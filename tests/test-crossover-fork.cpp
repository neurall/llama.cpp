// The planner's crossover picks (fork/docs/planner-crossovers.md, fork/crosstest.py X2-X5) in closed loop, CPU only: a simulated server starts from a
// cold state and runs requests; the engine's own pick code decides each one, the arm that ran learns as the engine learns, and the times are the
// measured ones of real runs (sources by each case), with noise. The crossover audit of Qwen3.8 IQ4_XS (cold_split_first_tested, 2026-11-25) left
// X2-X5 unverified: the planner never ran the other arm. Each check asks (a) after its trials the pick is within 5% of the faster arm, (b) the other
// arm is tried, or, where by design it never is, every pick is the faster arm on the measured numbers, (c) no decision treats an untimed arm as known.
// build: cmake --build <dir> --target test-crossover-fork;  run: ctest -R test-crossover-fork
// standalone: g++ -std=c++17 -O2 tests/test-crossover-fork.cpp -o /tmp/test-crossover-fork && /tmp/test-crossover-fork
#include "../src/llama-rls-fork.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;
static void check(const char * what, bool ok, double v) {
    printf("  %-100s %s  (%.3f)\n", what, ok ? "OK" : "FAILED", v);
    if (!ok) { n_failed++; }
}

// X4: one card against the dense split (dsf_pick), priced as dsf_plan prices a request: the prompt pass and the rest of the request's cells (its
// decode tokens) as a second pass of the horizon ("2 passes left, stake 2" in the logged plans), the one-card arm from the moe-cache's model of
// the path (an RLS from its cold prior, as llama_moe_cache_pp_one_draw_fork hands it over), the split from its own RLS (the plan's modelled
// split per token as its prior, dsf_plan), the passes measured at this size from the runs so far (g_buckets), the fill owed until the split's
// first pass. Six requests of the same prompt from a cold state, 200 seeds of 3% noise (the repeat spread of a pass, RLS_S2_0's source).
struct x4_case {
    const char * name;
    double T, T2, U;             // the prompt pass, the request's decode cells (the horizon's second pass), the expert upload (s)
    double one_c, one_w;         // one card's pass: the first request (cold) and the later ones (s)
    double spl_c, spl_w;         // the split's pass, the same (s)
    double fill;                 // the fill the split owes before its first pass (s)
    double bench_us, mc_us, split_us, ratio;   // the bench's one-card compute, the moe-cache's cold prior, the plan's split, per token (us); the plan's split/one ratio
    char first;                  // the arm the logged plan picked on the first request
    double log_one, log_split;   // the logged plan's first-request means of its first pass (s), 0: none logged
    bool one_ran = false;        // one card ran once before (the logged first request): the six requests start from there
};

static void x4_sims() {
    // Qwen3.8 IQ4_XS on two 3090s. The plan's inputs from cold_split_first_tested-qni4.server.log (2026-11-25): upload 1.41 s, one card 2.32 +- 0.84 s
    // at 13782 tokens (the bench cap: 168.3 us a token, +-36%), moe-cache stream prior 23455 ms (1600 us a token over the upload), split prior 139.9 us
    // a token, ratio 0.83, fill 0.08 s owed; the split's passes 5.44 s cold, 5.01 s warm (rel_multissd4-qni4: 2750 t/s for the call). One card at
    // 13782: 2113-2241 t/s cold, 2410-2460 warm (pv2_winmove_sync-pv2-qni4-on, poolv2_ff-pv2-qni4-on/off, one card alone, 2026-11-22).
    // At 3536 (rel_multissd4-qni4-io, 2026-11-20): one card 2313 t/s, the split's pass 1991 t/s and its fill 0.12 s, streaming upload 1.40 s.
    // Two Vulkan cards at 13782 (rel_tuners_take_turns-vk-qni4-i, 2026-11-18): request 1 ran one card (8.30 s a pass), request 2 the split on its
    // prior (one x 0.89 = 7.35 s, upload 1.43 s, fill 0.08 s), which took 10.49 s a pass (1314 t/s): from there.
    const x4_case cases[] = {
        { "Qwen3.8 IQ4_XS 13782 (the split 11-13% faster)", 13782, 1024, 1.41, 13782 / 2125.0, 13782 / 2440.0, 5.44, 5.01, 0.18, 168.3, 1600, 139.9, 0.83, 'S', 2.32, 1.93 },
        { "Qwen3.8 IQ4_XS 3536 (one card 16% faster)",        3536, 1024, 1.40, 3536 / 2313.0,  3536 / 2313.0,  3540 / 1991.0, 3540 / 1991.0, 0.12, 168.3, 1600, 139.9, 0.83, '1', 0, 0 },
        { "Qwen3.8 IQ4_XS 13782 two Vulkan cards (one card 26% faster)", 13782, 1024, 1.43, 8.30, 8.30, 10.49, 10.49, 0.08, 0, 1600, 139.9, 0.89, 'S', 8.30, 7.35, true },
    };
    const double s = 1e6;
    for (const auto & k : cases) {
        int good = 0, other_ran = 0, b_ok = 0, bad_known = 0, first_ok = 0;
        double p_one = 0, p_split = 0;   // request 1's means as priced (the same in every seed: nothing ran yet)
        std::string seq1;
        for (int seed = 1; seed <= 200; ++seed) {
            std::mt19937 rng(seed); std::normal_distribution<double> g(0.0, 1.0);
            dsf_rls one, split;
            { const double th0[4] = { 0, k.mc_us, 0, 1 }, fl[4] = { 64 * k.mc_us, 0, 0, 0 }; rls_init(one, 4, th0, fl); }
            { const double th0[4] = { 0, k.split_us, 1, 1 }, fl[4] = { 64 * k.split_us, 0, 0, 0 }; rls_init(split, 4, th0, fl); }
            double sum1 = 0, sum2 = 0; int n1 = 0, n2 = 0; bool owed = k.fill > 0;
            if (k.one_ran) { const double phi[4] = { 1, k.T, 0, k.U * s }; rls_update(one, phi, k.one_c * s); sum1 = k.one_c * s; n1 = 1; }
            int lost = 0, lost_late = 0; bool ran[2] = {}, right = true;
            std::string seq;
            for (int r = 0; r < 6; ++r) {
                std::vector<dsf_arm_pass> passes;
                for (const double T : { k.T, k.T2 }) {
                    const double phi[4] = { 1, T, 0, k.U * s };
                    dsf_arm_pass q; q.T = T; q.U = k.U * s;
                    q.mu1 = rls_pred(one, phi, &q.v1); q.kn1 = rls_known(one, q.mu1, q.v1); q.n1 = one.n;
                    if (split.n > 0) { q.mu2 = rls_pred(split, phi, &q.v2); }
                    if (T == k.T && n1 > 0) { q.m1 = sum1 / n1; q.nm1 = n1; }
                    if (T == k.T && n2 > 0) { q.m2 = sum2 / n2; q.nm2 = n2; }
                    passes.push_back(q);
                }
                dsf_call dc; dc.c_one_bench = k.bench_us; dc.ratio = dsf_prior_ratio(k.ratio, 1.0, false); dc.split_prior = split.n == 0; dc.split = &split; dc.seen_before = r + (k.one_ran ? 1 : 0);
                if (owed) { dc.f_mu = k.fill * s; dc.f_v = k.fill * k.fill * s * s; }
                const dsf_pick_out o = dsf_pick(passes, dc);
                if (r == 0) { p_one = o.mu1_0 / s; p_split = o.mu2_0 / s; }
                bad_known += (o.known1 && n1 == 0) || (o.known2 && n2 == 0);   // (c): an arm that never ran here is never known here
                const double b1 = r == 0 ? k.one_c : k.one_w, b2 = (r == 0 ? k.spl_c : k.spl_w) + (owed ? k.fill : 0.0);
                const bool slow = o.choice == 1 ? b2 > 1.05 * b1 : b1 > 1.05 * b2;
                right = right && (o.choice == 1 ? b2 <= b1 : b1 <= b2);
                lost += slow; lost_late += slow && r >= 2;
                seq += o.choice ? 'S' : '1';
                ran[o.choice] = true;
                const double phi[4] = { 1, k.T, 0, k.U * s }, f = 1.0 + 0.03 * g(rng);
                if (o.choice == 1) { const double t = (r == 0 ? k.spl_c : k.spl_w) * f; rls_update(split, phi, t * s); sum2 += t * s; n2++; owed = false; }
                else               { const double t = b1 * f;                            rls_update(one, phi, t * s);   sum1 += t * s; n1++; }
            }
            good += lost <= 1 && lost_late == 0;
            other_ran += ran[0] && ran[1];
            b_ok += (ran[0] && ran[1]) || right;
            first_ok += seq[0] == k.first;
            if (seed == 1) { seq1 = seq; }
        }
        char what[300];
        if (k.log_one > 0) {
            snprintf(what, sizeof what, "X4 %s: request 1 priced as logged: one card %.2f s (log %.2f), split %.2f (log %.2f)", k.name, p_one, k.log_one, p_split, k.log_split);
            check(what, std::fabs(p_one / k.log_one - 1) < 0.02 && std::fabs(p_split / k.log_split - 1) < 0.02, p_one);
        }
        snprintf(what, sizeof what, "X4 %s: first pick as logged (%c), seed 1 runs %s (1 one card, S split)", k.name, k.first, seq1.c_str());
        check(what, first_ok == 200, first_ok);
        snprintf(what, sizeof what, "X4 %s: (a) at most one losing request, none from the third, in 95%% of seeds", k.name);
        check(what, good >= 190, good);
        // (b): the other arm runs, or (one card never tried at 13782: stake 2 against one card at the split / 0.83, its mirror) every pick is the faster arm
        snprintf(what, sizeof what, "X4 %s: (b) the other arm tried in %d of 200 seeds, else every pick the faster arm", k.name, other_ran);
        check(what, b_ok == 200, b_ok);
        snprintf(what, sizeof what, "X4 %s: (c) no pick takes an arm that never ran here as known", k.name);
        check(what, bad_known == 0, bad_known);
    }
}

int main() {
    x4_sims();
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
