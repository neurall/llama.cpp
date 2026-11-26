// Self-check of the planner's estimator (src/llama-rls-fork.h): one RLS per path, Thompson draws, residual-driven drift.
// Synthetic two-path cost series, no GPU, no model.  build: cmake --build <dir> --target test-rls-fork;  run: ctest -R test-rls-fork
#include "../src/llama-rls-fork.h"
#include "../src/llama-estimate-fork.h"   // the placement pick, for the second simulator

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
        // GLM 12k on two Vulkan cards, request 2 (handoff_serial_cal_vk, 2026-11-17): the split measured once (28.38 +- 2.41 s, fill 1.94 +- 0.80 owed), one card
        // never ran; the plan's ratio 0.77 (the blocking hand-offs' bytes priced, the drains around them not) put the mirror at 28.38 / 0.77 = 37 s, above the
        // split: the split on every request (31.4 s with the fill; one card runs 26.5)
        dsf_rls once; once.n = 1; once.s2 = 0.01;
        dsf_call vg; vg.ratio = 0.77; vg.split_prior = false; vg.split = &once; vg.f_mu = 1.94 * s; vg.f_v = 0.80 * 0.80 * s * s; vg.seen_before = 1;
        const dsf_arm_pass gp = pass(12434, 3.95, 11.68, 4.20, false, 0, 28.38, 2.41);
        check("GLM 12k Vulkan as logged (ratio 0.77): the split", dsf_pick({ gp }, vg).choice == 1, dsf_pick({ gp }, vg).mu1_0 / s);
        // the plan's ratio where the hand-offs block (their bytes and the expert uploads they hold back, dsxn_layer_us): 1.4 stands for what it gives here
        vg.ratio = dsf_prior_ratio(1.4, 1.0, true);
        check("hand-offs that block: the plan's ratio, above 1 too", vg.ratio == 1.4 && dsf_prior_ratio(0.77, 1.0, true) == 0.77, vg.ratio);
        check("GLM 12k Vulkan request 2, the blocking ratio: one card runs", dsf_pick({ gp }, vg).choice == 0, dsf_pick({ gp }, vg).mu1_0 / s);
        check("hand-offs in stream order: the bench's ratio, at most 1", dsf_prior_ratio(0.77, 1.0, false) == 0.77 && dsf_prior_ratio(1.31, 1.0, false) == 1.0, 0.0);
        // request 1 (user 2026-11-17: "there will be no second chance on same prompt. the first planner choice must be best one"): neither ran, one card
        // 11.68 +- 4.20 s, upload 3.95 s; at the bytes-only 0.77 the split's prior won and took 31.6 s (one card 26.5); at a blocking ratio of 1 it was a trial
        dsf_call g1; g1.f_mu = 0.18 * s; g1.f_v = 0.18 * 0.18 * s * s;
        const dsf_arm_pass g1p = pass(12434, 3.95, 11.68, 4.20, false, 0);
        g1.ratio = 0.77;
        check("GLM 12k Vulkan request 1 at 0.77: the split (the miss)", dsf_pick({ g1p }, g1).choice == 1, dsf_pick({ g1p }, g1).am[1] / s);
        g1.ratio = 1.0;
        check("GLM 12k Vulkan request 1 at 1: the split as a trial (the miss)", dsf_pick({ g1p }, g1).choice == 1 && dsf_pick({ g1p }, g1).trial, dsf_pick({ g1p }, g1).am[1] / s);
        g1.ratio = dsf_prior_ratio(1.4, 1.0, true);
        const dsf_pick_out g1o = dsf_pick({ g1p }, g1);
        check("GLM 12k Vulkan request 1, the blocking ratio: one card, no trial", g1o.choice == 0 && !g1o.trial, g1o.am[1] / s);
        // the plan with each range's attention by position (dsxn_att_pos) said 1.05 there: a pair whose hand-offs block keeps a ratio above 1 as a measured cost,
        // so the split is not tried against it (the trial ran 31.6 s against one card's 26.5); a pair in stream order at its cap of 1 still is
        g1.ratio = 1.05; g1.blocks = true;
        const dsf_pick_out g1b = dsf_pick({ g1p }, g1);
        check("GLM 12k Vulkan request 1 at 1.05, hand-offs block: one card, no trial", g1b.choice == 0 && !g1b.trial, g1b.am[1] / s);
        g1.ratio = 1.0; g1.blocks = false;
        check("the same at 1 in stream order: the split as a trial", dsf_pick({ g1p }, g1).trial, dsf_pick({ g1p }, g1).am[1] / s);
        // DeepSeek 4.1 13571 tokens, fresh state (equal_prompt_passes relgate 2026-11-25): no token-range graph (deepseek41 is no mirror arch), so its "split" shortens
        // no compute; the plan's ratio 0.56 priced it at 54.69 s against one card 97.18 +- 34.99 (upload 4.18): the split ran 90.6 s, one card runs 86.6 and was never tried
        dsf_call ds; ds.ratio = 0.56;
        const dsf_arm_pass dsp = pass(13571, 4.18, 97.18, 34.99, false, 0);
        check("DeepSeek 13571 fresh at the plan's 0.56: the split (the miss)", dsf_pick({ dsp }, ds).choice == 1, dsf_pick({ dsp }, ds).am[1] / s);
        ds.ratio = dsf_prior_ratio(0.56, 1.0, false, false); ds.ranges = false;
        const dsf_pick_out dso = dsf_pick({ dsp }, ds);
        check("DeepSeek 13571 fresh, no token ranges: ratio 1, one card, no trial", ds.ratio == 1.0 && dso.choice == 0 && !dso.trial, dso.am[1] / s);
        // MiMo 11k on two CUDA cards, request 1: upload 3.36 s, one card 4.00 +- 1.44, fill 0.10; at ratio 0.84 the split is bound by its upload and its mean
        // (3.46 with the fill) is the lower: the upload rule forbids a trial of the split, not the split where it is ahead (it refused it at 0.84, took it at 0.87)
        dsf_call m1; m1.ratio = 0.84; m1.f_mu = 0.10 * s; m1.f_v = 0.10 * 0.10 * s * s;
        const dsf_pick_out m1o = dsf_pick({ pass(11011, 3.36, 4.00, 1.44, false, 0) }, m1);
        check("MiMo 11k CUDA request 1, bound by the upload, the lower mean: the split", m1o.choice == 1 && m1o.upload_bound && !m1o.trial, m1o.am[1] / s);
    }
    {   // INV phi at a call's end (GLM, two Vulkan cards, 2026-11-16): 9472 tokens in 6 passes (trial passes of 1664, then 1536), features for 1536
        check("INV phi: the prompt's passes taught the arm, its end's average is no sample", !rls_report_phi_off(1536.0, 9472.0 / 6.0, 4, true), 9472.0 / 6.0);
        check("INV phi: no pass taught it, the end would teach 43 tokens off its features", rls_report_phi_off(1536.0, 9472.0 / 6.0, 0, true), 9472.0 / 6.0);
        check("INV phi: passes of the planned size", !rls_report_phi_off(1536.0, 3072.0 / 2.0, 0, true), 1536.0);
    }
    {   // closed-loop pick simulator (user 2026-11-25: "use simulator to see how our curr code would behave without need for real run"): a fake machine with the
        // measured one-card and split times of a logged miss (fork/docs/bug-log.md), a cold start, six requests of the same size; dsf_pick decides each, the arm
        // that ran feeds its model (one card: the measured mean, known from 2 samples; the split: its RLS, the fill paid once). Pass: at most one losing request
        // (the trial) and the last three on the faster arm. Only cases a real OLMoE run (restricted VRAM, power limit) cannot make: an arch without token ranges, Vulkan
        // hand-offs that block; the rest are real rows (user 2026-11-25: "real thing is always better").
        struct sim_case { const char * name; double T, U, one, split, prior1, sd1, ratio, fill; bool blocks, ranges, old = false; };   // old: the inputs before the fix, must lose
        const sim_case cases[] = {
            { "DeepSeek 4.1 13571, no token ranges (bug-log 2026-11-25)", 13571, 4.18, 86.6,  90.6, 97.18, 34.99, 0.56, 0.0,  false, false },
            { "old: DeepSeek priced as if it had token ranges (the miss)", 13571, 4.18, 86.6,  90.6, 97.18, 34.99, 0.56, 0.0,  false, true, true },
            { "GLM 12k two Vulkan cards, hand-offs block (2026-11-17)",   12434, 3.95, 26.5,  29.7, 11.68, 4.20,  1.40, 1.94, true,  true  },
            { "MiMo 11k two Vulkan cards, hand-offs block (#375)",        11011, 3.40, 11.0,  15.5, 11.0,  3.00,  1.24, 0.0,  true,  true  },
        };
        const double s = 1e6;
        for (const auto & k : cases) {
            dsf_rls split; const double c = k.prior1 * s / k.T * k.ratio, th0[4] = { 0, c, 1, 1 }, fl[4] = { 64 * c, 0, 0, 0 }; rls_init(split, 4, th0, fl);
            std::vector<double> one_runs; bool fill_owed = k.fill > 0;
            int lost = 0, last_bad = 0, n_split = 0; std::string seq;
            for (int r = 0; r < 6; ++r) {
                dsf_arm_pass q; q.T = k.T; q.U = k.U * s;
                if (one_runs.empty()) { q.mu1 = k.prior1 * s; q.v1 = k.sd1 * k.sd1 * s * s; }
                else { double m = 0; for (double x : one_runs) { m += x; } m /= one_runs.size(); q.mu1 = m; q.v1 = 0.03 * 0.03 * m * m; q.kn1 = one_runs.size() >= 2; q.n1 = (int) one_runs.size(); }
                const double sphi[4] = { 1, k.T, 0, k.U * s };
                q.mu2 = rls_pred(split, sphi, &q.v2);
                if (!one_runs.empty()) { q.m1 = k.one * s; q.nm1 = (int) one_runs.size(); }   // the passes measured at this size (dsf_plan's B.us_tok)
                if (n_split > 0)       { q.m2 = k.split * s; q.nm2 = n_split; }
                dsf_call dc; dc.ratio = dsf_prior_ratio(k.ratio, 1.0, k.blocks, k.ranges); dc.blocks = k.blocks; dc.ranges = k.ranges; dc.split_prior = split.n == 0; dc.split = &split; dc.seen_before = r;   // the ratio as dsf_plan forms it
                if (fill_owed) { dc.f_mu = k.fill * s; dc.f_v = 0.4 * 0.4 * k.fill * k.fill * s * s; }
                const int ch = dsf_pick({ q }, dc).choice;
                const bool bad = ch == 1 ? k.split > k.one : k.one > k.split;
                lost += bad; last_bad += bad && r >= 3; seq += ch ? 'S' : '1';
                if (ch == 1) { rls_update(split, sphi, k.split * s); fill_owed = false; n_split++; } else { one_runs.push_back(k.one * s); }
            }
            char what[200]; snprintf(what, sizeof what, "sim %s: %s", k.name, seq.c_str());
            check(what, (lost <= 1 && last_bad == 0) != k.old, lost);
        }
    }
    {   // the same closed loop for the other one-vs-two pick: one card or the layer split for a model that fits one card (place_one_card_pick). The loop
        // above missed its bad pick (relgate1k 2026-11-25: Qwen3.6 ran one card cold, -31% prompt) because it never calls it: dsf_pick splits a prompt pass
        // inside a loaded server, where both arms run in one process and each case starts from priors. The placement is fixed before the load, one start
        // runs one arm, and a -cs cell is a cold first start, which none of those cases is. Here: a cold state (as -cs gives), then six server starts that
        // each read the records the starts before them wrote and run one request; chat prompts under 1024 tokens leave the prompt untimed. The rates are the
        // request totals the records keep, from the server logs of 2026-11-25 (Qwen3.6 two 3090s: one card passes of 1718 tokens at 5.7-7.1k i/s, request
        // 5810.5 i/s 152.5 o/s; layer split passes of 1728 at 9.9-11.9k i/s, request 8207.5 i/s 162.0 o/s) and 2026-11-23 (Inkling 2 Mini 12k chat: one card
        // 1372 i/s 56.9 o/s, split 1074 i/s 33.9 o/s). Pass: start 1 runs the split and at most one start runs the slower arm, so the split wins from start 3
        // and a faster one card from start 2 (until 2026-11-25; now the first start is the estimate's pick); no decision line says "measured" before both arms have a record.
        struct req { double n_p, n_g; };
        const req chat = { 700, 400 }, gate = { 13744, 1024 }, decode_bound = { 300, 4096 };
        // Since 2026-11-25 the first start runs the estimate's pick (place_estimate) from the inputs the decision has: the split's weights left in RAM by
        // the fitter (Inkling 6.2 GiB, Qwen3.6 none), the experts' used share, RAM read and upload rates, the split's passes. The second start tries the
        // other arm once, then the measured totals decide.
        const double GiB = 1073741824.0;
        struct place_sim { const char * name; double one_i, one_o, spl_i, spl_o; req odd, even; double host, used, pass; };
        const place_sim cases[] = {
            { "Qwen3.6 cold, chat and 12k starts",                  5810.5, 152.5, 8207.5, 162.0, chat, gate, 0, 8.0 / 256, 4096 },
            { "Qwen3.6 cold, 12k gate starts (relgate1k)",          5810.5, 152.5, 8207.5, 162.0, gate, gate, 0, 8.0 / 256, 4096 },
            { "Inkling 2 Mini cold, chat and 12k starts (one card)", 1372,  56.9,  1074,   33.9,  chat, gate, 6.2 * GiB, 8.0 / 128, 2048 },
            { "Inkling 2 Mini cold, decode-bound starts (one card)", 1372,  56.9,  1074,   33.9,  decode_bound, decode_bound, 6.2 * GiB, 8.0 / 128, 2048 },
        };
        for (const auto & k : cases) {
            double rec[2][4] = { { -1, -1, 0, 0 }, { -1, -1, 0, 0 } };   // [one card, split]: p, g (ms a token, -1: not timed), n_p, n_g as the state keeps them
            std::string seq, bad_line;
            int lost = 0;
            place_est_in ei; ei.host_bytes = k.host; ei.used_share = k.used; ei.ram_gbs = 111.92; ei.link_gbs = 26.61; ei.pass_tokens = k.pass; ei.passes_per_call = 2;
            const bool est_one = place_estimate_pick(place_estimate(ei, k.odd.n_p), k.odd.n_p, k.odd.n_g);
            const bool want_one = k.odd.n_p / k.one_i + k.odd.n_g / k.one_o < k.odd.n_p / k.spl_i + k.odd.n_g / k.spl_o;
            for (int st = 0; st < 6; ++st) {
                const req q = st % 2 ? k.even : k.odd;
                const bool h1 = place_record_counts(rec[0][0], rec[0][1]), h2 = place_record_counts(rec[1][0], rec[1][1]);
                const double np = h1 && h2 ? (rec[0][2] + rec[1][2]) / 2 : 0, ng = h1 && h2 ? (rec[0][3] + rec[1][3]) / 2 : 0;   // as common.cpp weighs them
                const bool one = place_one_card_pick(h1, rec[0][0], rec[0][1], h2, rec[1][0], rec[1][1], np, ng, est_one);
                const place_line pl = place_pick_line(one, h1, rec[0][0], rec[0][1], h2, rec[1][0], rec[1][1], np, ng, "estimate");
                if (!(h1 && h2) && (pl.label + pl.why).find("measured") != std::string::npos && bad_line.empty()) { bad_line = pl.label + " (" + pl.why + ")"; }
                const double t1 = q.n_p / k.one_i + q.n_g / k.one_o, t2 = q.n_p / k.spl_i + q.n_g / k.spl_o;
                lost += one ? t1 > t2 : t2 > t1;
                seq += one ? '1' : 'S';
                double * r = rec[one ? 0 : 1];
                place_record_update(r[0], r[1], r[2], r[3], 1000.0 / (one ? k.one_i : k.spl_i), (int) q.n_p, 1000.0 / (one ? k.one_o : k.spl_o), (int) q.n_g);
            }
            char what[512];
            snprintf(what, sizeof what, "place sim %s: %s (1 = one card, S = split)", k.name, seq.c_str());
            check(what, (seq[0] == '1') == want_one && seq[1] != seq[0] && lost <= 1, lost);   // first the faster arm by the estimate, then the other once
            snprintf(what, sizeof what, "place sim %s: no line says measured before both are timed%s%s", k.name, bad_line.empty() ? "" : ": ", bad_line.c_str());
            check(what, bad_line.empty(), 0);
        }
    }
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
