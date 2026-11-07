// The planner's replay gate (user 2026-11-07: "a planner change is only built when the replay shows no pick losing >5% vs the measured best").
// Feeds logged requests, in their order and from a fresh state, through the planner's estimator and decision kernels (src/llama-rls-fork.h: rls_*,
// rls_argmin, rls_split_choice - the same functions llama-moecache-fork.cpp and llama-dsplit-fork.cpp decide with) and scores each pick against the
// measured best arm of that call class (size, cold/warm request).
//
//   test-planner-replay-fork [events.csv truth.csv [seeds]]      (fork/planner_events.py writes both from the q5h stats logs)
// Without arguments: the GLM 3.5 12k x16+x4 medians of 2026-11-07 (63 logged calls, 17 starts, builds up9_merged2 .. rampool10) are built in.
//
// Arms: one-card paths stream / mix / predictions k x1..x32 / decode chain / prompt chain with the moe-cache's features, the dense split with its fill.
// The outcome of a pick is the measured median of its arm class (stream, mix and predictions share the stream's: the logs have too few of each);
// an arm class never measured at that call class is unscored (its pick is reported, no sample is fed). CPU only, no model.
#include "../src/llama-rls-fork.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct ev { std::string log; int req; double n, cells, Ts, Um, up, ch, dec_max = 1e9; };
enum { A_RING, A_MIX, A_CHAIN, A_DEC, A_K, A_N = A_K + 6 };
static const int N_USED = 8, N_EXP = 512;
static const int PICK_MODE = 3;   // the mode the planner runs (keep equal to the code)   // GLM 3.5: k steps x1..x32 (the arms rampool9/10 listed)

static std::vector<std::string> csv_row(const std::string & l) {
    std::vector<std::string> o; std::string c; bool q = false;
    for (char ch : l) { if (ch == '"') { q = !q; } else if (ch == ',' && !q) { o.push_back(c); c.clear(); } else { c += ch; } }
    o.push_back(c); return o;
}
static std::string cls_of(double n) { return n < 3000 ? "2k" : "10k"; }
static const char * arm_class(int a) { return a == A_DEC ? "dec" : a == A_CHAIN ? "chain" : "stream"; }

struct result { int calls = 0, scored = 0, bad = 0; double loss_sum = 0, loss_max = 0; std::map<std::string, std::pair<int, double>> by; };

// mode: 3 value of information (rls_voi_pick, the planner's), 0 Thompson draws, 1 posterior means, 2 lower confidence bound mu - sigma (an arm the posterior allows to be faster is tried; once measured its sigma is the noise)
static result replay(const std::vector<ev> & E, std::map<std::string, double> & T, double fill_ms, uint64_t seed, int mode, bool verbose) {
    result R;
    std::string cur;
    dsf_rls one[A_N], split, fill;
    bool copies = false;
    std::vector<double> seen;
    uint64_t rseed = 0; int last_req = -1;
    for (const auto & e : E) {
        if (e.log != cur) {   // a new start: fresh state (no state file), copies not in place
            cur = e.log; copies = false; last_req = -1; seen.clear();
            for (auto & r : one) { r = dsf_rls(); }
            split = dsf_rls(); fill = dsf_rls();
        }
        if (e.req != last_req) { rseed = rls_mix(seed ^ ((uint64_t) e.req * 1315423911ull) ^ std::hash<std::string>{}(e.log)); last_req = e.req; }
        const std::string st = e.req == 0 ? "c" : "w", cl = cls_of(e.n);
        // the moe-cache's candidates and features (llama_moe_cache_pp_decide_fork); attention not fitted in these logs (feature 0)
        double phi[A_N][4] = {}; bool ok[A_N] = {};
        auto cand = [&](int a, double U) { ok[a] = true; phi[a][0] = 1; phi[a][1] = e.n; phi[a][2] = 0; phi[a][3] = U * 1e3; };
        cand(A_RING, e.Ts); cand(A_MIX, e.Um);
        cand(A_CHAIN, e.up); cand(A_DEC, e.up + e.ch);   // the chains at every size, as the planner (planner-learned)
        for (int i = 0; A_K + i < A_N && N_USED * (1 << i) < N_EXP; ++i) { cand(A_K + i, e.Ts * std::max(0.0, 1.0 - (double) (1 << i) * N_USED / N_EXP)); }
        const double ctok = e.ch > 0 ? e.ch * 1e3 / e.n : 1000.0;   // the prior compute per token: a token's expert bytes / the read rate (= the chunks' per token)
        double mu[A_N], var[A_N], dr[A_N];
        for (int a = 0; a < A_N; ++a) {
            if (!ok[a]) { continue; }
            if (one[a].k == 0) { const double th0[4] = { 0, ctok, 1, 1 }, fl[4] = { 64 * ctok, 0, 0, 0 }; rls_init(one[a], 4, th0, fl); }
            if ((a == A_MIX || a >= A_K) && one[a].n == 0 && one[A_RING].n > 0) { one[a] = one[A_RING]; one[a].n = 0; one[a].nis = 1; }
            mu[a] = rls_pred(one[a], phi[a], &var[a]);
            rls_draw(one[a], phi[a], rls_mix(rseed + a), &dr[a]);
        }
        double lcb[A_N];
        for (int a = 0; a < A_N; ++a) { lcb[a] = ok[a] ? mu[a] - std::sqrt(std::max(0.0, var[a])) : 0; }
        int Rn = 1;   // this call and the earlier ones of the same size in this start
        for (const auto & q : seen) { Rn += q == e.n; }
        seen.push_back(e.n);
        bool meas[A_N];
        for (int a = 0; a < A_N; ++a) { meas[a] = ok[a] && rls_known(one[a], mu[a], var[a]); }
        const int pick1 = mode == 3 ? rls_voi_pick(mu, var, ok, A_N, Rn, nullptr, meas) : rls_argmin(mode == 0 ? dr : mode == 1 ? mu : lcb, ok, A_N);
        // the split (dsf_plan): one pass per call here (the logged calls ran as one pass)
        const double sphi[4] = { 1, e.n, 0, e.Ts * 1e3 };
        if (split.k == 0) { const double c = 2034.2, th0[4] = { 0, c, 1, 1 }, fl[4] = { 64 * c, 0, 0, 0 }; rls_init(split, 4, th0, fl); }   // rampool10's logged split prior
        const double est = copies ? 0.0 : 740e3;   // the planner's fill estimate (us) on these cards
        if (fill.k == 0) { const double o = 1; rls_init(fill, 1, &o, nullptr); }
        double vs, ds, fd = 0, fv, dd1 = mode == 0 ? std::max(0.0, dr[pick1]) : mode == 1 ? mu[pick1] : lcb[pick1];
        double ms = rls_pred(split, sphi, &vs); rls_draw(split, sphi, rls_mix(rseed + 0x51), &ds);
        if (mode == 3 && split.n == 0) { ms = mu[pick1] * RLS_SPLIT_ONE_RATIO; vs = ms * ms * RLS_SPLIT_ONE_SDLOG * RLS_SPLIT_ONE_SDLOG + var[pick1] * RLS_SPLIT_ONE_RATIO * RLS_SPLIT_ONE_RATIO; }   // the data prior (dsf_plan)
        if (est > 0) { const double fm = rls_pred(fill, &est, &fv); if (mode == 0) { rls_draw(fill, &est, rls_mix(rseed + 0xf1), &fd); fd = std::max(0.0, fd); } else { fd = fm; } }
        const double d2 = mode == 0 ? ds : mode == 1 ? ms : ms - std::sqrt(std::max(0.0, vs));
        int sp = rls_split_choice(dd1 - d2, std::max(0.0, dd1 - d2), fd);
        if (mode == 3) {   // two arms for the call: the one-card pick's mean and the split's mean plus the fill owed
            const double m2[2] = { mu[pick1], ms + (est > 0 ? rls_pred(fill, &est, &fv) : 0) }, v2[2] = { var[pick1], vs + (est > 0 ? fv : 0) };
            const bool o2[2] = { true, true };
            const bool me2[2] = { rls_known(one[pick1], m2[0], v2[0]), split.n > 0 && rls_known(split, m2[1] - (est > 0 ? rls_pred(fill, &est, &fv) : 0), vs) };
            sp = rls_voi_pick(m2, v2, o2, 2, Rn, nullptr, me2);
            if (getenv("REPLAY_DBG") && verbose) { printf("    dbg2 n %.0f R %d one %.0f +- %.0f (n %d) split %.0f +- %.0f (n %d)\n", e.n, Rn, m2[0] / 1e3, std::sqrt(v2[0]) / 1e3, one[pick1].n, m2[1] / 1e3, std::sqrt(v2[1]) / 1e3, split.n); }
            if (getenv("REPLAY_DBG") && verbose) { printf("    dbg n %.0f R %d one %.0f +- %.0f split %.0f +- %.0f\n", e.n, Rn, m2[0] / 1e3, std::sqrt(v2[0]) / 1e3, m2[1] / 1e3, std::sqrt(v2[1]) / 1e3); }
        }
        // outcome and score
        const std::string picked = sp ? "dense" : arm_class(pick1);
        auto tr = [&](const std::string & a) { auto it = T.find(cl + "," + st + "," + a); return it == T.end() ? -1.0 : it->second; };
        // the dense medians are the split pass without its fill (fork/planner_events.py): the fill is added where this replayed start still owes it
        auto call_t = [&](const std::string & a) { const double t = tr(a); return t > 0 && a == "dense" && !copies ? t + fill_ms : t; };
        double best = -1; std::string ba;
        for (const char * a : { "stream", "dec", "chain", "dense" }) { const double t = call_t(a); if (t > 0 && (best < 0 || t < best)) { best = t; ba = a; } }
        double took = call_t(picked);
        R.calls++;
        const std::string key = cl + "-" + st + " " + picked;
        if (took > 0 && best > 0) {
            const double loss = took / best - 1.0;
            R.scored++; R.loss_sum += loss; R.loss_max = std::max(R.loss_max, loss); R.bad += loss > 0.05;
            R.by[key].first++; R.by[key].second += loss;
            if (verbose) { printf("  %-34s req %d %5.0f tok %s: picked %-6s (%s) %6.0f ms, best %-6s %6.0f ms, loss %5.1f%%\n", e.log.c_str(), e.req, e.n, st.c_str(), picked.c_str(), sp ? "split" : "one card", took, ba.c_str(), best, 100 * loss); }
            // the sample the planner learns from (the code's definitions: split pass = call - fill wall; fill = wall)
            if (sp) { rls_update(split, sphi, tr("dense") * 1e3); if (!copies) { rls_update(fill, &est, fill_ms * 1e3); } copies = true; }
            else    { rls_update(one[pick1], phi[pick1], took * 1e3); }
        } else if (verbose) { printf("  %-34s req %d %5.0f tok %s: picked %-6s unscored (never measured there)\n", e.log.c_str(), e.req, e.n, st.c_str(), picked.c_str()); }
    }
    return R;
}

int main(int argc, char ** argv) {
    std::vector<ev> E;
    std::map<std::string, double> T;
    double fill_ms = 1478;
    if (argc >= 3) {
        std::ifstream f(argv[1]); std::string l; std::getline(f, l);
        while (std::getline(f, l)) { auto c = csv_row(l); if (c.size() < 11) continue; E.push_back({ c[0], atoi(c[1].c_str()), atof(c[2].c_str()), atof(c[3].c_str()), atof(c[4].c_str()), atof(c[5].c_str()), atof(c[6].c_str()), atof(c[7].c_str()), c.size() > 11 ? atof(c[11].c_str()) : 1e9 }); }
        std::ifstream g(argv[2]);
        while (std::getline(g, l)) { auto c = csv_row(l); if (c.size() >= 4) { T[c[0] + "," + c[1] + "," + c[2]] = atof(c[3].c_str()); } }
    } else {   // built in: GLM 3.5 12k medians (truth.csv of 2026-11-07) and one start's call features (rampool10 rp0)
        T = { { "10k,c,dense", 15844 - 1478 }, { "10k,c,stream", 17329 }, { "10k,w,dense", 13280 - 1478 }, { "10k,w,stream", 17387 }, { "2k,c,dec", 19304 }, { "2k,c,dense", 15932 - 8904 },
              { "2k,c,stream", 8354 }, { "2k,w,dec", 20248 }, { "2k,w,dense", 8450 - 1397 }, { "2k,w,stream", 7338 } };
        for (int s = 0; s < 4; ++s) {
            const std::string lg = "builtin" + std::to_string(s);
            for (int r = 0; r < 2; ++r) { E.push_back({ lg, r, 2034, 1017, 8244, 7421, 12000, 3350 }); E.push_back({ lg, r, 10400, 7234, 23226 - 15000, 6660, 11688, 18144 }); }
        }
    }
    const int seeds = argc >= 4 ? atoi(argv[3]) : 20;
    int fails = 0;
    for (int mode = 0; mode < 4; ++mode) {
        result tot; int worst_bad = 0;
        for (int s = 0; s < seeds; ++s) {
            result r = replay(E, T, fill_ms, 1000 + s, mode, s == 0 && argc >= 3 && mode == PICK_MODE);
            tot.calls += r.calls; tot.scored += r.scored; tot.bad += r.bad; tot.loss_sum += r.loss_sum; tot.loss_max = std::max(tot.loss_max, r.loss_max);
            worst_bad = std::max(worst_bad, r.bad);
            for (auto & kv : r.by) { tot.by[kv.first].first += kv.second.first; tot.by[kv.first].second += kv.second.second; }
        }
        printf("%s: %d calls x %d seeds, %d scored, picks losing > 5%%: %d (%.1f%%), mean loss %.1f%%, worst %.1f%%\n", mode == 0 ? "draws (Thompson)" : mode == 1 ? "posterior means" : mode == 2 ? "lower bound mu - sigma" : "value of information (EI R > EL)",
               tot.calls / seeds, seeds, tot.scored, tot.bad, 100.0 * tot.bad / std::max(1, tot.scored), 100.0 * tot.loss_sum / std::max(1, tot.scored), 100.0 * tot.loss_max);
        for (auto & kv : tot.by) { printf("    %-22s %5d picks, mean loss %5.1f%%\n", kv.first.c_str(), kv.second.first, 100.0 * kv.second.second / kv.second.first); }
        if (mode == PICK_MODE) { fails = tot.bad; }
    }
    printf("gate (the planner's pick, mode %d): %s\n", PICK_MODE, fails ? "FAIL: picks losing more than 5% against the measured best" : "pass");
    return 0;   // a report, not a ctest failure: the gate is read before a build (run.py), the build itself is not blocked by ctest
}
