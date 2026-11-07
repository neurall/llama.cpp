// Self-check of the planner's estimator (src/llama-rls-fork.h): one RLS per path, Thompson draws, residual-driven drift.
// Synthetic two-path cost series, no GPU, no model.  build: cmake --build <dir> --target test-rls-fork;  run: ctest -R test-rls-fork
#include "../src/llama-rls-fork.h"

#include <cstdio>
#include <random>

static int n_failed = 0;
static void check(const char * what, bool ok, double v) {
    printf("  %-64s %s  (%.3f)\n", what, ok ? "OK" : "FAILED", v);
    if (!ok) { n_failed++; }
}

int main() {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);
    // two paths of a pass of n tokens at c cells with upload U (us): A = 200 ms + 1.5 ms/token + U, B = 50 ms + 0.9 ms/token + 1.5 x U (uploads that stall the compute): B is 13% faster at 2k, 32% at 10k
    auto truth = [](int p, double n, double c, double U) { return p == 0 ? 200e3 + 1500.0 * n + 0.01 * n * c + U : 50e3 + 900.0 * n + 0.01 * n * c + 1.5 * U; };
    // both start from the same wrong prior (compute 1 ms per token, upload coefficient 1): the planner does not know which is faster
    const double th0[4] = { 0.0, 1000.0, 0.0, 1.0 }, fl[4] = { 64.0 * 1000.0, 0.0, 0.0, 0.0 };
    dsf_rls r[2];
    rls_init(r[0], 4, th0, fl);
    rls_init(r[1], 4, th0, fl);
    int picks_b = 0, late = 0;
    for (int it = 0; it < 200; ++it) {
        const double n = (it % 2) ? 2000.0 : 10000.0, c = n * 0.5 + (it % 3) * 4096.0, U = 1.5e6;
        const double phi[4] = { 1.0, n, n * c, U };
        double t[2];
        for (int p = 0; p < 2; ++p) { rls_draw(r[p], phi, rls_mix(1000 + it) ^ (uint64_t) p, &t[p]); }
        const int p = t[1] < t[0] ? 1 : 0;
        const double y = truth(p, n, c, U) * (1.0 + 0.03 * noise(rng));
        rls_update(r[p], phi, y);
        if (it >= 100) { late++; picks_b += p; }
    }
    check("after warm-up the faster path (B) is picked >= 90% of calls", picks_b >= 0.9 * late, (double) picks_b / late);
    {
        const double phi[4] = { 1.0, 10000.0, 10000.0 * 5000.0, 1.5e6 };
        double v = 0.0;
        const double mu = rls_pred(r[1], phi, &v);
        const double y = truth(1, 10000.0, 5000.0, 1.5e6);
        check("B's posterior mean within 5% of the truth", std::fabs(mu - y) < 0.05 * y, mu / y);
        check("B is determined (n >= k + 1)", r[1].det(), r[1].n);
    }
    // drift: the path becomes 1.6x slower (a power cap): the first surprising residual must re-open P
    {
        const double phi[4] = { 1.0, 10000.0, 10000.0 * 5000.0, 1.5e6 };
        double tr0 = 0.0, tr1 = 0.0;
        for (int i = 0; i < 4; ++i) { tr0 += r[1].P[i * 4 + i] * phi[i] * phi[i]; }
        const double nis = rls_update(r[1], phi, 1.6 * truth(1, 10000.0, 5000.0, 1.5e6));
        for (int i = 0; i < 4; ++i) { tr1 += r[1].P[i * 4 + i] * phi[i] * phi[i]; }
        check("drift: the surprising sample has NIS > 1", nis > 1.0, nis);
        check("drift: residual-driven inflation grows P", tr1 > tr0, tr1 / tr0);
        check("drift: the running means restart at n = k + 1", r[1].n == r[1].k + 1, r[1].n);
        // and the model follows within a few samples
        for (int i = 0; i < 8; ++i) { rls_update(r[1], phi, 1.6 * truth(1, 10000.0, 5000.0, 1.5e6) * (1.0 + 0.03 * noise(rng))); }
        double v = 0.0;
        const double mu = rls_pred(r[1], phi, &v);
        check("drift: the mean follows to within 5% after 8 samples", std::fabs(mu / (1.6 * truth(1, 10000.0, 5000.0, 1.5e6)) - 1.0) < 0.05, mu / (1.6 * truth(1, 10000.0, 5000.0, 1.5e6)));
    }
    // persistence round trip
    {
        dsf_rls q;
        const bool ok = rls_parse(q, 4, rls_str(r[1]));
        check("state string round trip keeps th", ok && std::fabs(q.th[1] - r[1].th[1]) <= 1e-5 * std::fabs(r[1].th[1]), q.th[1] / r[1].th[1]);
    }
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
