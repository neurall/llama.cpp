// The planner's crossover picks (fork/docs/planner-crossovers.md, fork/crosstest.py X2-X5) in closed loop, CPU only: a simulated server starts from a
// cold state and runs requests; the engine's own pick code decides each one, the arm that ran learns as the engine learns, and the times are the
// measured ones of real runs (sources by each case), with noise. The crossover audit of Qwen3.8 IQ4_XS (cold_split_first_tested, 2026-11-25) left
// X2-X5 unverified: the planner never ran the other arm. Each check asks (a) after its trials the pick is within 5% of the faster arm, (b) the other
// arm is tried, or, where by design it never is, every pick is the faster arm on the measured numbers, (c) no decision treats an untimed arm as known.
// build: cmake --build <dir> --target test-crossover-fork;  run: ctest -R test-crossover-fork
// standalone: g++ -std=c++17 -O2 tests/test-crossover-fork.cpp -o /tmp/test-crossover-fork && /tmp/test-crossover-fork
#include "../src/llama-estimate-fork.h"   // pp_path_rules, pp_pick (X2, X3); llama-rls-fork.h with it (dsf_pick: X4)

#include <cstdio>
#include <cstring>
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

// X2 (the stream against the chains and the k arms) and X3 (the stream alone against the stream with the mix): the moe-cache planner's own rules and
// pick (pp_path_rules, pp_pick: llama-estimate-fork.h, as llama_moe_cache_pp_decide_fork calls them). A reason that says "known here" names the pick
// as the lowest mean among the paths that ran here at comparable features (rls_known): a path that never ran that way is no such pick.
static bool claims_known(const char * why) { return std::strstr(why, "known here") != nullptr; }

// Two plans as logged, every path's mean and spread from the plan line ("plan: prompt call of N tokens ...", ms).
static void x23_replays() {
    double mu[PP_PATHS], var[PP_PATHS];
    bool ok[PP_PATHS], kn[PP_PATHS], known[PP_PATHS], tonly[PP_PATHS];
    int runs[PP_PATHS];
    auto clear = [&] { for (int p = 0; p < PP_PATHS; ++p) { mu[p] = var[p] = 0.0; ok[p] = kn[p] = false; runs[p] = 0; } };
    auto arm = [&](int p, double m_ms, double sd_ms) { ok[p] = true; mu[p] = m_ms * 1e3; var[p] = sd_ms * sd_ms * 1e6; };
    // Qwen3.8 IQ4_XS on two 3090s, the first prompt call (cold_split_first_tested-qni4.server.log, 2026-11-25): 13782 tokens, nothing ran yet (n=0 on
    // every path), R1's crossover 256 (max-batch 51, the chain's calculated 763 capped at 256). Logged: "picked stream, layer parallel", the reason
    // "lowest mean among the paths known here" - none was: the mix's 22639 ms was the lowest mean, but a trial only past PP_MIX_COLD.
    for (const bool sw : { false, true }) {
        clear();
        arm(PP_RING, 23455, 22120); arm(PP_MIX, 22639, 22082); arm(PP_CHAIN, 22760, 22086); arm(PP_DEC, 44773, 31727);
        arm(PP_K, 23865, 22151); for (int i = 1; i <= 5; ++i) { arm(PP_K + i, 23876, 22152); }
        arm(PP_LP, 22640, 22082);
        pp_path_rules(13782, 256, sw, false, runs, kn, ok, known, tonly);
        const pp_pick_out o = pp_pick(mu, var, ok, known, tonly, 1, nullptr);
        char what[300];
        snprintf(what, sizeof what, "X2/X3 replay Qwen3.8 IQ4_XS 13782 cold (router switch %s): layer parallel as logged, no trial", sw ? "on" : "off");
        check(what, o.pick == PP_LP && !o.trial, o.pick);
        snprintf(what, sizeof what, "X2/X3 replay Qwen3.8 IQ4_XS 13782 cold (router switch %s): (c) no path known here, the reason says none is: \"%s\"", sw ? "on" : "off", o.why);
        check(what, !claims_known(o.why), 0);
    }
    // Qwen3.6 on two cards, 1280 tokens after the dense split ran this size (split_variant_measured-qwen36-2cards-expertcache-on.server.log): the prompt
    // chain n=1, the rest n=0, the split's passes per variant all 0 (not run yet). Logged: "stream variant stream, not the drawn stream, mix", then the
    // reason "lowest mean among the paths known here": the split variant rule took it, no mean and no known path.
    {
        clear();
        arm(PP_RING, 3726, 3576); arm(PP_MIX, 3565, 3572); arm(PP_CHAIN, 3750, 2768); arm(PP_DEC, 7129, 5054);
        arm(PP_K, 3766, 3579); arm(PP_K + 1, 3771, 3579); arm(PP_K + 2, 3773, 3579); arm(PP_K + 3, 3774, 3579); arm(PP_K + 4, 3775, 3579);
        arm(PP_LP, 3565, 3572);
        runs[PP_CHAIN] = 1;
        const double took[3] = { 0.0, 0.0, 0.0 };
        pp_path_rules(1280, 58, false, false, runs, kn, ok, known, tonly);
        const pp_pick_out o = pp_pick(mu, var, ok, known, tonly, 1, took);
        check("X3 replay Qwen3.6 1280 after a dense split: the stream variant, not the drawn mix, as logged", o.pick == PP_RING && o.drawn == PP_MIX && !o.trial, o.pick);
        char what[300];
        snprintf(what, sizeof what, "X3 replay Qwen3.6 1280 after a dense split: (c) the reason names the split's variant rule, no mean: \"%s\"", o.why);
        check(what, !claims_known(o.why) && std::strstr(o.why, "lowest mean") == nullptr, 0);
    }
}

// In closed loop: requests of one size from a cold state on one card with a slower second card (the mix's), the dense split not taking the call.
// Each path's model is an RLS built as pp_rls builds it: th0 = (0, compute per token, 1, 1) over phi = (1, n, n x a x cells, the path's calculated
// upload), the attention feature left 0; the compute is the logged guess (1600 us a token) until a pass measured it (105 us a token, the same log),
// and a never-run path is built again when it moved; a never-run mix or k arm takes the stream's posterior once the stream ran (pp_rls); the paths
// that never ran are priced by the last prior's miss at this size (rls_prior_cal, the same compute only); the path that ran learns its time (3% noise).
// The calculated uploads per pass are the 13782-token plan's (stream 1405 ms, mix 589, prompt chain 710, decode chain 22722, k x1 1815, k x2-x32 1826).
struct x23_case {
    const char * name;
    double n, cold, warm;   // the stream's pass, the first request and the later ones (s)
    double r_mix, r_k;      // the mix's and the k arms' pass over the stream's
    bool mix_tried;         // (b): the mix must be tried (it is faster, or within 5%: a run settles it)
};

static void x23_sims() {
    const double c_guess = 1600.0, c_meas = 105.0, s = 1e6;
    double Up[PP_PATHS] = {};
    Up[PP_RING] = 1405e3; Up[PP_MIX] = 589e3; Up[PP_CHAIN] = 710e3; Up[PP_DEC] = 22722e3; Up[PP_K] = 1815e3;
    for (int i = 1; i <= 5; ++i) { Up[PP_K + i] = 1826e3; }
    bool cand[PP_PATHS] = {};
    for (const int p : { (int) PP_RING, (int) PP_MIX, (int) PP_CHAIN, (int) PP_DEC }) { cand[p] = true; }
    for (int i = 0; i <= 5; ++i) { cand[PP_K + i] = true; }
    // Qwen3.8 IQ4_XS one card at 13782: 6.49 s cold, 5.65 s warm (pv2_winmove_sync-pv2-qni4-on and poolv2_ff, one card alone, 2026-11-22); the k arms
    // 14% slower (planner-crossovers.md, qni4 forced cells: stream -> k -14%, loses all six); the mix 2% faster at 16k (stream -> mix: wins 16k cold and
    // warm). At 3536: one card 2313 t/s (rel_multissd4-qni4-io). The mix 18% faster there (Qwen3.8 IQ3 1.5-4.5k: +18-47%, the engine's PP_MIX_COLD
    // rows) or 1.88x slower at 13782 (MiMo 12k: 495 against 933 t/s): the same model's times with another model's mix ratio, the two sides of X3.
    const x23_case cases[] = {
        { "Qwen3.8 IQ4_XS 13782: k 14% slower, the mix 2% faster", 13782, 6.49, 5.65, 0.98, 1.14, true },
        { "Qwen3.8 IQ4_XS 13782: the mix 1.88x slower (MiMo 12k's ratio)", 13782, 6.49, 5.65, 1.88, 1.14, false },
        { "Qwen3.8 IQ4_XS 3536: the mix 18% faster (Qwen3.8 IQ3's ratio)", 3536, 3536 / 2313.0, 3536 / 2313.0, 1 / 1.18, 1.14, true },
    };
    const int NR = 12;
    for (const auto & k : cases) {
        int late_ok = 0, bad_why = 0, mix_ran = 0, k_ran = 0, chain_ran = 0, worst_lost = 0, one_lost = 0;
        std::string seq1;
        for (int seed = 1; seed <= 200; ++seed) {
            std::mt19937 rng(seed); std::normal_distribution<double> g(0.0, 1.0);
            dsf_rls arm[PP_PATHS];
            double built[PP_PATHS] = {}, c = c_guess, cal = 0.0, cal_c = 0.0;
            bool c_measured = false, ran[PP_PATHS] = {};
            int lost = 0, lost_late = 0;
            std::string seq;
            for (int r = 0; r < NR; ++r) {
                double mu[PP_PATHS] = {}, var[PP_PATHS] = {}, cal_on[PP_PATHS] = {};
                bool ok[PP_PATHS] = {}, kn[PP_PATHS] = {}, known[PP_PATHS], tonly[PP_PATHS];
                int runs[PP_PATHS] = {};
                for (int p = 0; p < PP_PATHS; ++p) {
                    if (!cand[p]) { continue; }
                    if (arm[p].k == 0 || (arm[p].n == 0 && c != built[p])) {
                        const double th0[4] = { 0.0, c, 1.0, 1.0 }, fl[4] = { 64.0 * c, 0.0, 0.0, 0.0 };
                        rls_init(arm[p], 4, th0, fl); built[p] = c;
                    }
                    if ((p == PP_MIX || p >= PP_K) && arm[p].n == 0 && arm[PP_RING].n > 0) { arm[p] = arm[PP_RING]; arm[p].n = 0; arm[p].nis = 1.0; }
                    const double phi[4] = { 1.0, k.n, 0.0, Up[p] };
                    mu[p] = std::max(c_measured ? c * k.n : 0.0, rls_pred(arm[p], phi, &var[p]));
                    ok[p] = true;
                }
                int ns[PP_PATHS];
                for (int p = 0; p < PP_PATHS; ++p) { ns[p] = cand[p] ? arm[p].n : 1; }
                rls_prior_cal(mu, var, ns, PP_PATHS, cal, built, cal_c);
                for (int p = 0; p < PP_PATHS; ++p) {
                    cal_on[p] = ns[p] == 0 ? (cal > 0.0 && built[p] == cal_c ? cal : 1.0) : 0.0;
                    kn[p] = rls_known(arm[p], mu[p], var[p]); runs[p] = arm[p].n;
                }
                pp_path_rules(k.n, 256, false, false, runs, kn, ok, known, tonly);
                const pp_pick_out o = pp_pick(mu, var, ok, known, tonly, r + 1, nullptr);
                bad_why += claims_known(o.why) && !known[o.pick];
                const double base = r == 0 ? k.cold : k.warm;
                auto t_of = [&](int p) { return base * (p == PP_MIX ? k.r_mix : p >= PP_K ? k.r_k : p == PP_RING ? 1.0 : 2.66); };   // the chains: the decode chain's 266%
                double best = t_of(PP_RING);
                for (int p = 0; p < PP_PATHS; ++p) { if (cand[p] && p != PP_CHAIN) { best = std::min(best, t_of(p)); } }   // the prompt chain: not measured on this model
                const int pk = o.pick;
                const bool slow = pk < 0 || t_of(pk) > 1.05 * best;
                lost += slow; lost_late += slow && r >= NR - 4;
                seq += pk == PP_RING ? 'S' : pk == PP_MIX ? 'M' : pk == PP_CHAIN ? 'C' : pk == PP_DEC ? 'D' : pk >= PP_K ? (char) ('0' + pk - PP_K) : '?';
                if (pk < 0) { continue; }
                ran[pk] = true;
                const double took = t_of(pk) * (1.0 + 0.03 * g(rng)) * s, phi[4] = { 1.0, k.n, 0.0, Up[pk] };
                if (cal_on[pk] > 0.0 && mu[pk] > 0.0) { cal = took / (mu[pk] / cal_on[pk]); cal_c = built[pk]; }   // it ran on its prior: the prior's miss prices the never-run arms
                rls_update(arm[pk], phi, took);
                c = c_meas; c_measured = true;   // a pass measured the compute
            }
            late_ok += lost_late == 0;
            mix_ran += ran[PP_MIX]; chain_ran += ran[PP_CHAIN] || ran[PP_DEC];
            bool any_k = false;
            for (int i = 0; i < PP_KN; ++i) { any_k = any_k || ran[PP_K + i]; }
            k_ran += any_k;
            worst_lost = std::max(worst_lost, lost); one_lost += lost <= 1;
            if (seed == 1) { seq1 = seq; }
        }
        char what[300];
        snprintf(what, sizeof what, "X2/X3 %s: (a) the last 4 of %d requests within 5%% of the best path, in 95%% of seeds (seed 1 runs %s; S stream, M mix, 0-5 k x1-x32)", k.name, NR, seq1.c_str());
        check(what, late_ok >= 190, late_ok);
        // the k arms take the stream's posterior and its spread (the cold first pass in it), so a k arm that never ran stays a wide unknown: each is tried
        // once the calls of this size so far make its stake pay, k x1 first. The doc's X2 target: at most one trial per size band per process.
        snprintf(what, sizeof what, "X2/X3 %s: at most one request lost (a slower path's trial) in 95%% of seeds; the most in a seed %d", k.name, worst_lost);
        check(what, one_lost >= 190, one_lost);
        if (k.mix_tried) {
            snprintf(what, sizeof what, "X3 %s: (b) the never-run mix tried in every seed", k.name);
            check(what, mix_ran == 200, mix_ran);
        }
        snprintf(what, sizeof what, "X2 %s: (b) the chains past the crossover never run (R1: no trial there); a k arm ran in %d of 200 seeds", k.name, k_ran);
        check(what, chain_ran == 0, chain_ran);
        snprintf(what, sizeof what, "X2/X3 %s: (c) no reason calls a pick known here that is not", k.name);
        check(what, bad_why == 0, bad_why);
    }
}

int main() {
    x23_replays();
    x23_sims();
    x4_sims();
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
