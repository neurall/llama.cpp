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
    // a restart keeps what one sample pinned: the reloaded posterior predicts the measured call as narrow as before (diag(P) alone: about +-55%)
    {
        const double th0[4] = { 2e5, 300.0, 0.0, 1.0 }, fl[4] = { 1e5, 50.0, 0.001, 0.5 };
        dsf_rls a; rls_init(a, 4, th0, fl);
        const double phi[4] = { 1.0, 4497.0, 4497.0 * 2000.0, 2.0e6 };
        rls_update(a, phi, 6.0e6);
        double v0 = 0.0, v1 = 0.0;
        rls_pred(a, phi, &v0);
        dsf_rls q;
        const bool ok = rls_parse(q, 4, rls_str(a));
        rls_pred(q, phi, &v1);
        check("one sample: the measured call is narrow (sd < 10%)", std::sqrt(v0) < 0.10 * 6.0e6, std::sqrt(v0) / 6.0e6);
        check("restart: the reloaded posterior is as narrow there (within 1%)", ok && std::fabs(std::sqrt(v1) / std::sqrt(v0) - 1.0) < 0.01, std::sqrt(v1) / std::sqrt(v0));
        dsf_rls d;   // an older state line (no off-diagonal section) still loads, as diag(P)
        const std::string s = rls_str(a);
        check("old format without off-diagonals loads", rls_parse(d, 4, s.substr(0, s.rfind('|'))) && d.P[1] == 0.0, d.P[1]);
    }
    {   // K3 1k prompt, request 2 (predmax1k, 2026-11-15): k x1 ran once on its prior (45.9 s estimated, 70.4 s took) and its model says 95.2 s; layer parallel,
        // mix and stream never ran and kept the prior (71.6, 71.6, 71.7 s); layer parallel won and took 77.4 s
        double mu[4]  = { 95.2, 71.6, 71.6, 71.7 }, var[4] = { 17.8 * 17.8, 6.3e5 * 6.3e5, 79.5 * 79.5, 2.4e4 * 2.4e4 };
        const int nsamp[4] = { 1, 0, 0, 0 };
        const bool ok[4] = { true, true, true, true }, known[4] = { false, false, false, false };
        bool trial = false;
        check("without the calibration a never-run prior wins (the K3 miss)", rls_voi_pick(mu, var, ok, 4, 2.0, &trial, known) != 0, 0.0);
        rls_prior_cal(mu, var, nsamp, 4, 70.4 / 45.9);
        check("priors x the measured miss (70.4 / 45.9): the arm that ran wins", rls_voi_pick(mu, var, ok, 4, 2.0, &trial, known) == 0, mu[1]);
        check("the arm that ran keeps its own model's mean", mu[0] == 95.2, mu[0]);
        double m2[2] = { 10.0, 20.0 }, v2[2] = { 1.0, 1.0 }; const int s2[2] = { 0, 0 };
        rls_prior_cal(m2, v2, s2, 2, 0.0);
        check("no calibration yet: the priors stay", m2[0] == 10.0 && m2[1] == 20.0, m2[0]);
    }
    {   // GLM 12k on one Vulkan card, request 2 (2026-11-17): layer parallel ran on its prior, built from the guessed compute (12 ms a token: 152.3 s), took 28.6 s
        // (x0.19); the pass measured the compute (2.30 ms a token) and the stream, never run, was built again from it: 32.5 s, then x0.19 on top: 6.1 s. It won
        // and took 29.3 s. An old miss prices only the arms still built from the compute that prior had
        double mu[3] = { 28.56, 32.51, 160.0 }, var[3] = { 1.56 * 1.56, 28.9 * 28.9, 150.0 * 150.0 };
        const int nsamp[3] = { 1, 0, 0 };
        const double built[3] = { 12000.0, 2296.7, 12000.0 };   // us a token each arm's prior was built from (the last one never built again)
        rls_prior_cal(mu, var, nsamp, 3, 28557.0 / 152300.0, built, 12000.0);
        check("an arm built again from the measured compute keeps its prior", mu[1] == 32.51, mu[1]);
        check("an arm still on the old guess gets the miss", std::fabs(mu[2] - 160.0 * 28557.0 / 152300.0) < 1e-9, mu[2]);
        const bool ok[3] = { true, true, true }, known[3] = { false, false, false };
        bool trial = false;
        check("the path that ran stays the pick", rls_voi_pick(mu, var, ok, 3, 3.0, &trial, known) == 0, mu[1]);
    }
    // one card against the split (dsf_pick, the split planner's pick): cases from logged calls (stats.db 2026-11-05..15), the engine's own PLN inputs, seconds.
    // Of 116 real one-vs-two losses with logged inputs: 63 lost trials (exploration), 33 estimates on the wrong side (GLM 3.0 cold one card ~ its upload alone:
    // 4.0 s against 5.5 measured at 2268 tokens), 2 coin flips (K3 995 cold, priors 0.1% apart); none a rule picking against estimates that were right.
    {
        const double s = 1e6;
        auto pass = [&](double T, double U, double mu1, double sd1, bool kn1, int n1, double mu2 = 0, double sd2 = 0) {
            dsf_arm_pass p; p.T = T; p.U = U * s; p.mu1 = mu1 * s; p.v1 = sd1 * sd1 * s * s; p.kn1 = kn1; p.n1 = n1; p.mu2 = mu2 * s; p.v2 = sd2 * sd2 * s * s; return p; };
        dsf_rls meas; meas.n = 3; meas.s2 = 0.01;   // a split measured three times, 10% noise
        // K3 995 tokens cold (hot_tree8, 11-13): neither side ran; the plan's split / one ratio 1.31 is capped at 1 (above 1 the plan is wrong on most models:
        // median 1.45 on GLM 3.5, whose split measures 0.73-1.09 of one card), so the split's prior is one card's and a trial of it is the exploration (measured 87.1 s vs 77.4)
        dsf_call k3; k3.ratio = 1.0;
        const dsf_pick_out a = dsf_pick({ pass(995, 22.2, 81.1, 11.0, false, 1) }, k3);
        check("K3 995 cold, equal priors: the split is a trial, never a mean pick", a.choice == 1 && a.trial, a.am[1] / s);
        dsf_call k3m; k3m.ratio = 1.0; k3m.split_prior = false; k3m.split = &meas;
        const dsf_pick_out b = dsf_pick({ pass(995, 22.2, 77.4, 2.0, true, 3, 87.1, 3.0) }, k3m);
        check("K3 995, the split measured slower (87.1 vs 77.4 s): one card, no trial", b.choice == 0 && !b.trial, b.am[0] / s);
        // GLM 3.0 2268 tokens cold (var1, 11-09): one card priced 4.02 +- 1.45 against its upload 3.99, fill 0.74 owed: upload-bound, no trial of the split, one card;
        // measured one card 5.51, split 4.88 with its fill: the miss is the estimate (one card's compute left out), not the rule
        dsf_call g; g.ratio = 1.0; g.f_mu = 0.74 * s; g.f_v = 0.74 * 0.74 * s * s;
        const dsf_pick_out c = dsf_pick({ pass(2268, 3.99, 4.02, 1.45, false, 0) }, g);
        check("GLM 3.0 2268 cold as logged: upload-bound, one card", c.choice == 0 && c.upload_bound, c.am[0] / s);
        const dsf_pick_out d0 = dsf_pick({ pass(2268, 3.99, 5.51, 0.3, true, 3) }, g);
        check("GLM 3.0 2268, one card known at its measured 5.51 s, the plan's capped ratio 1.00: still one card (the ratio is the miss)", d0.choice == 0, d0.am[1] / s);
        g.ratio = 0.90;   // GLM 3.0's measured split / one card (12 classes, staging=0 logs): the split is tried
        const dsf_pick_out d = dsf_pick({ pass(2268, 3.99, 5.51, 0.3, true, 3) }, g);
        check("GLM 3.0 2268, one card known at 5.51 s, the measured ratio 0.90: the split is tried", d.choice == 1 && d.trial, d.am[1] / s);
        // the rampool13 deadlock guard: neither known, the split's prior a hair above one card by the fill - still tried where the stake pays
        dsf_call h; h.ratio = 1.0; h.f_mu = 0.2 * s; h.f_v = 0.04 * s * s;
        const dsf_pick_out e = dsf_pick({ pass(10144, 4.0, 14.0, 5.0, false, 1) }, h);
        check("GLM 3.0 12k, split prior one x 1.00 + fill above one card: a trial of the split", e.choice == 1 && e.trial, e.am[1] / s);
        // the bench cap (MiMo 3k, 11-08): one card never measured, its cold prior 8.7 s against 4.0-4.3 measured: capped to max(upload, bench compute)
        dsf_call mi; mi.ratio = 0.9; mi.c_one_bench = 4.2 * s / 3000.0;
        const dsf_pick_out f = dsf_pick({ pass(3000, 2.0, 8.7, 3.0, false, 0) }, mi);
        check("MiMo 3k: a never-measured one card is capped by the bench (8.7 -> 4.2 s)", std::fabs(f.mu1_0 / s - 4.2) < 1e-9, f.mu1_0 / s);
        // crossover audit X4 (qn1m ns_probe_cells, 3536 tokens, 2026-11-12): the split's model said 2.06 +- 0.34 s against one card 2.09 +- 0.10, the split ran,
        // and this size had measured one card 1628 t/s (2.17 s) and the split 1389 t/s (2.55 s): the passes measured at this size price the pass, not the model
        dsf_call qn; qn.ratio = 0.96; qn.split_prior = false; qn.split = &meas;
        dsf_arm_pass q = pass(3536, 0.5, 2.09, 0.10, true, 4, 2.06, 0.34);
        check("qn1m 3536 as logged, nothing measured at this size: the split (the models' pick)", dsf_pick({ q }, qn).choice == 1, 0.0);
        q.m1 = 3536.0 / 1628.0 * s; q.nm1 = 2; q.m2 = 3536.0 / 1389.0 * s; q.nm2 = 2;
        const dsf_pick_out qa = dsf_pick({ q }, qn);
        check("qn1m 3536, one card 2.17 s and the split 2.55 s measured at this size: one card, no trial", qa.choice == 0 && !qa.trial, qa.am[1] / s);
        // one sample of each at this size stays with the models (qni3 engram_late_fill 2442: the split measured once at 1253 t/s ran at 2025)
        dsf_arm_pass r = pass(2442, 0.4, 1.62, 0.10, true, 4, 1.15, 0.20);
        r.m1 = 2442.0 / 1514.0 * s; r.nm1 = 1; r.m2 = 2442.0 / 1253.0 * s; r.nm2 = 1;
        check("qni3 2442, one sample of each at this size: the models decide (the split)", dsf_pick({ r }, qn).choice == 1, 0.0);
    }
    {   // INV phi at a call's end (GLM, two Vulkan cards, 2026-11-16): 9472 tokens in 6 passes (trial passes of 1664, then 1536), features for 1536
        check("INV phi: the prompt's passes taught the arm, its end's average is no sample", !rls_report_phi_off(1536.0, 9472.0 / 6.0, 4, true), 9472.0 / 6.0);
        check("INV phi: no pass taught it, the end would teach 43 tokens off its features", rls_report_phi_off(1536.0, 9472.0 / 6.0, 0, true), 9472.0 / 6.0);
        check("INV phi: passes of the planned size", !rls_report_phi_off(1536.0, 3072.0 / 2.0, 0, true), 1536.0);
    }
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
