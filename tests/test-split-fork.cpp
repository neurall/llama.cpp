// Unit tests of the dense split planner's layer model and search (src/llama-splitplan-fork.h): how many tokens the helper card takes (s), how many of each layer's experts
// move to it (f; 0 = the experts are skipped on the helper, the offload card computes them all), and whether splitting beats one card. Per-layer numbers in, plan out;
// no GPU and no model.
//
// build: cmake --build <dir> --target test-split-fork;  run: ctest -R test-split-fork   or   test-split-fork [iterations] [seed]
#include "../src/llama-splitplan-fork.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

static int  n_failed = 0;
static long g_iters  = 300;
static unsigned long g_seed = 1;

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-62s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}

// a model of L layers, the first n_dense_first of them dense (no experts); GLM-like sizes: 800 MB of experts per MoE layer, 5 ms of compute per 1k tokens per layer
static dsx_in model(int64_t T, int L = 40, int n_dense_first = 3, double Bm = 24.0, double Bh = 6.0, double rel = 1.0) {
    dsx_in in;
    in.T = T; in.L = L; in.Bm = Bm; in.Bh = Bh; in.rel = rel;
    in.w.assign(L, 1.0); in.xf.assign(L, 0.6); in.E.assign(L, 800e6); in.kv_b.assign(L, 0.0); in.att_w.assign(L, 1.0 / L);
    in.held_m.assign(L, 0.0); in.held_h.assign(L, 0.0);
    for (int il = 0; il < n_dense_first; ++il) { in.E[il] = 0.0; in.xf[il] = 0.0; }
    in.w_sum = L; in.c_base = 5.0; in.att_tok = 0.0;   // us per token for the whole model share of a layer = c_base * w / w_sum
    in.c_base = 5.0 * L;
    in.b_in = in.b_out = 6144.0 * 2.0;
    return in;
}

static void test_plan_shape() {
    printf("Testing the dense split plan:\n");
    std::mt19937_64 rng(g_seed);
    long bad = 0, n_f = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        auto in = model(256 + (int64_t) (rng() % 30000), 8 + (int) (rng() % 60), (int) (rng() % 6), 4.0 + (double) (rng() % 300) / 10.0, 2.0 + (double) (rng() % 300) / 10.0, 0.3 + (double) (rng() % 200) / 100.0);
        for (int il = 0; il < in.L; ++il) { in.held_m[il] = (double) (rng() % 100) / 100.0; in.held_h[il] = (double) (rng() % 100) / 100.0; in.kv_b[il] = (double) (rng() % 50) * 1e6; }
        const dsx_out o = dsx_plan(in);
        if (!(o.t_one > 0.0 && o.best_split > 0.0)) { fail("non-positive time"); continue; }
        if (o.best_s < 0.2 - 1e-9 || o.best_s > 0.8 + 1e-9) { fail(fmt("helper token share %.3f outside 0.2..0.8", o.best_s)); }
        if ((int) o.best_f.size() != in.L) { fail("f per layer"); continue; }
        for (int il = 0; il < in.L; ++il) {
            const double f = o.best_f[il];
            if (f < 0.0 || f > 1.0) { fail(fmt("layer %d: f %.2f", il, f)); }
            if (std::fabs(f * 20.0 - std::round(f * 20.0)) > 1e-9) { fail(fmt("layer %d: f %.3f is not a step of 5%%", il, f)); }
            if (in.E[il] <= 0.0 && f != 0.0) { fail(fmt("dense layer %d: the helper was given %.2f of its (no) experts", il, f)); }
            if (f > 0.0) {   // experts move only for a clear win (10% of the layer)
                n_f++;
                if (!(dsx_layer_us(in, il, o.best_s, f) < 0.90 * dsx_layer_us(in, il, o.best_s, 0.0))) { fail(fmt("layer %d: f %.2f is not a 10%% win", il, f)); }
            }
        }
    }
    report("s in 0.2..0.8; f in steps of 5%; dense layers f = 0; f > 0 only for a 10% win", bad == 0, first);
    report("the run had layers that took experts (not vacuous)", n_f > 100, fmt("%ld layer plans", n_f));
}

static void test_behaviour() {
    printf("Testing what the plan does when the cards differ (the experts are skipped on a slow helper):\n");
    {   // a helper behind a very slow link takes no experts, and splitting is no gain
        auto in = model(8000, 40, 3, 24.0, 0.05);
        const dsx_out o = dsx_plan(in);
        double fsum = 0.0; for (double f : o.best_f) { fsum += f; }
        report("helper link 0.05 GB/s: no experts on the helper", fsum == 0.0, fmt("sum f %.2f", fsum));
        report("helper link 0.05 GB/s: the split is slower than one card", o.best_split >= o.t_one, fmt("%.0f vs %.0f ms", o.best_split / 1e3, o.t_one / 1e3));
    }
    {   // experts already resident on a fast helper, compute-bound pass: it takes a share of them
        auto in = model(16000, 40, 3, 24.0, 24.0);
        in.c_base *= 6.0;
        for (int il = 0; il < in.L; ++il) { in.held_h[il] = 1.0; }
        const dsx_out o = dsx_plan(in);
        double fsum = 0.0; for (double f : o.best_f) { fsum += f; }
        report("fast helper that holds the experts: it computes some of them", fsum > 0.0, fmt("sum f %.2f", fsum));
    }
    {   // equal cards, no hops, no experts: the tokens split in half
        auto in = model(4000, 20, 20, 24.0, 24.0);
        in.b_in = in.b_out = 0.0;
        const dsx_out o = dsx_plan(in);
        report("equal cards, dense work only: half the tokens each", std::fabs(o.best_s - 0.5) < 0.021, fmt("s %.2f", o.best_s));
        report("equal cards, dense work only: the split beats one card by ~2x", o.best_split < 0.6 * o.t_one, fmt("%.0f vs %.0f ms", o.best_split / 1e3, o.t_one / 1e3));
    }
    {   // a faster helper takes more tokens
        auto a = dsx_plan(model(8000, 40, 40, 24.0, 24.0, 1.0)), b = dsx_plan(model(8000, 40, 40, 24.0, 24.0, 2.0));
        report("a helper twice as fast takes a larger share of the tokens", b.best_s >= a.best_s, fmt("%.2f vs %.2f", b.best_s, a.best_s));
    }
    {   // [the first model: 80-85% of the tokens on the helper, measured best 77%/less] the hand-in keeps the helper's share moderate for equal cards
        auto in = model(8800, 40, 3, 24.0, 24.0);
        const dsx_out o = dsx_plan(in);
        report("equal cards with experts: the helper never holds most of the tokens", o.best_s <= 0.6, fmt("s %.2f", o.best_s));
    }
    {   // time grows with the pass length
        bool mono = true; double p1 = 0, p2 = 0;
        for (int64_t T : { 512, 1024, 4096, 16384 }) { const dsx_out o = dsx_plan(model(T)); mono &= o.t_one > p1 && o.best_split > p2; p1 = o.t_one; p2 = o.best_split; }
        report("one-card and split times grow with the tokens", mono);
    }
    {   // the KV of a long context read over the helper's link makes the helper's range dearer
        auto a = model(8000, 40, 40, 24.0, 6.0), b = a;
        for (auto & x : b.kv_b) { x = 2e9; }
        report("a KV cache the helper must read over its link raises the split's time", dsx_plan(b).best_split > dsx_plan(a).best_split);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_plan_shape();
    test_behaviour();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
