// The planner's replay gate (user 2026-11-07: "a planner change is only built when the replay shows no pick losing >5% vs the measured best").
// Feeds logged requests, in their order and from a fresh state, through the planner's estimator and decision kernels (src/llama-rls-fork.h: rls_*,
// rls_argmin, rls_split_choice - the same functions llama-moecache-fork.cpp and llama-dsplit-fork.cpp decide with) and scores each pick against the
// measured best arm of that call class (size, cold/warm request).
//
//   test-planner-replay-fork [events.csv truth.csv [seeds]]      (fork/planner_events.py writes both from the q5h stats logs; --all: every kept log, per model)
// Without arguments: the GLM 3.5 12k x16+x4 medians of 2026-11-07 (63 logged calls, 17 starts, builds up9_merged2 .. rampool10) are built in.
//
// Arms: one-card paths stream / mix / predictions k x1..x32 / decode chain / prompt chain with the moe-cache's features, layer parallel (the stream with each
// card computing its own layers: a one-card path in the pick, two cards in the outcome), the dense split with its fill.
// The outcome of a pick is the measured median of its arm class (stream, mix and predictions share the stream's: the logs have too few of each);
// an arm class never measured at that call class is unscored (its pick is reported, no sample is fed). CPU only, no model.
// Events with a model (planner_events.py --all) carry their model's experts, picks, fill and split prior, and their class's medians are ms per token: with
// them the replay lists per model the picks losing > 5% and every pick on the wrong side of one card against two (with the inputs that made it;
// REPLAY_DBG=1: every call's inputs).
#include "../src/llama-rls-fork.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// the defaults past U are the built-in GLM 3.5 12k's: 8 of 512 experts (the gate's numbers; that GGUF says 288), the planner's fill estimate on these
// cards, the measured fill wall, rampool10's logged split prior
struct ev { std::string log; int req; double n, cells, Ts, Um, up, ch, dec_max = 1e9, U = 0;
            std::string model{}, cls{}; int n_used = 8, n_exp = 512; double fill_est = 740e3, fill_ms = 1478, split_prior = 2034.2, lp = 0, n_call = 0;
            std::string path{}; double took = 0; };   // what the logged call ran and took (the report's)
enum { A_RING, A_MIX, A_CHAIN, A_DEC, A_K, A_KN = 6, A_LP = A_K + A_KN, A_N };
static const int PICK_MODE = 3;
static bool g_carry = false;   // replay with the previous start's stored state (R4), else every start fresh   // the mode the planner runs (keep equal to the code)   // GLM 3.5: k steps x1..x32 (the arms rampool9/10 listed)
static std::map<std::string, std::array<double, 3>> g_range;   // per class and arm: samples, least and most ms per token (the report's: a median of two far apart is fragile)

static std::vector<std::string> csv_row(const std::string & l) {
    std::vector<std::string> o; std::string c; bool q = false;
    for (char ch : l) { if (ch == '"') { q = !q; } else if (ch == ',' && !q) { o.push_back(c); c.clear(); } else { c += ch; } }
    o.push_back(c); return o;
}
static std::string cls_of(const ev & e) { return !e.cls.empty() ? e.cls : e.n < 3000 ? "2k" : "10k"; }
static const char * arm_class(int a) { return a == A_DEC ? "dec" : a == A_CHAIN ? "chain" : a == A_LP ? "lp" : "stream"; }
// a stream variant's own outcome name (fork/planner_events.py writes ring, mix, k1 .. k32); "" for the others
static std::string arm_variant(int a) { return a == A_RING ? "ring" : a == A_MIX ? "mix" : a >= A_K && a < A_LP ? "k" + std::to_string(1 << (a - A_K)) : ""; }
static bool two_cards(const std::string & a) { return a == "dense" || a == "lp"; }

struct result { int calls = 0, scored = 0, bad = 0; double loss_sum = 0, loss_max = 0; std::map<std::string, std::pair<int, double>> by;
                std::map<std::string, std::array<int, 4>> models; std::vector<std::string> x4; };   // per model: calls, scored, > 5%, one against two cards; the latter's lines

// mode: 3 value of information (rls_voi_pick, the planner's), 0 Thompson draws, 1 posterior means, 2 lower confidence bound mu - sigma (an arm the posterior allows to be faster is tried; once measured its sigma is the noise)
static result replay(const std::vector<ev> & E, std::map<std::string, double> & T, uint64_t seed, int mode, bool verbose) {
    result R;
    std::string cur;
    dsf_rls one[A_N], split, fill;
    bool copies = false;
    std::vector<double> seen;
    std::map<int, double> cal;   // per size bucket: what a prior missed by (process-local, as g_pp_cal)
    uint64_t rseed = 0; int last_req = -1;
    for (const auto & e : E) {
        if (e.log != cur) {   // a new start: copies not in place; fresh state, or (carry) what the previous start of this arm stored (R4)
            const std::string arm = e.log.substr(e.log.rfind('-') + 1), prev_arm = cur.empty() ? "" : cur.substr(cur.rfind('-') + 1);
            cur = e.log; copies = false; last_req = -1; seen.clear(); cal.clear();
            if (g_carry && arm == prev_arm) {   // the state file round trip: every posterior with a sample (R4: its P carries any surprise), as rls_str / rls_parse (diag P), the rest fresh
                auto keep = [](dsf_rls & r, int k) { dsf_rls q; if (r.k > 0 && r.n >= 1 && rls_parse(q, k, rls_str(r))) { r = q; } else { r = dsf_rls(); } };
                for (auto & r : one) { keep(r, 4); }
                keep(split, 4); keep(fill, 1);
            } else {
                for (auto & r : one) { r = dsf_rls(); }
                split = dsf_rls(); fill = dsf_rls();
            }
        }
        if (e.req != last_req) { rseed = rls_mix(seed ^ ((uint64_t) e.req * 1315423911ull) ^ std::hash<std::string>{}(e.log)); last_req = e.req; }
        const std::string st = e.req == 0 ? "c" : "w", cl = cls_of(e);
        // the moe-cache's candidates and features (llama_moe_cache_pp_decide_fork); attention not fitted in these logs (feature 0)
        double phi[A_N][4] = {}; bool ok[A_N] = {};
        auto cand = [&](int a, double U) { ok[a] = true; phi[a][0] = 1; phi[a][1] = e.n; phi[a][2] = 0; phi[a][3] = U * 1e3; };
        cand(A_RING, e.Ts); cand(A_MIX, e.Um);
        cand(A_CHAIN, e.up); cand(A_DEC, e.up + e.ch);   // the chains at every size, as the planner (planner-learned)
        for (int i = 0; i < A_KN && e.n_used * (1 << i) < e.n_exp; ++i) { cand(A_K + i, e.Ts * std::max(0.0, 1.0 - (double) (1 << i) * e.n_used / e.n_exp)); }
        // layer parallel where the planner priced it (the CAL line's per-pass estimate). shortcut: that is its time (each card's upload or compute share,
        // whichever longer), not the upload the planner's feature is - equal while no pass measured the compute; log the upload part to separate them
        if (e.lp > 0) { cand(A_LP, e.lp); }
        const double ctok = e.ch > 0 ? e.ch * 1e3 / e.n : 1000.0;   // the prior compute per token: a token's expert bytes / the read rate (= the chunks' per token)
        double mu[A_N], var[A_N], dr[A_N];
        for (int a = 0; a < A_N; ++a) {
            if (!ok[a]) { continue; }
            if (one[a].k == 0) { const double th0[4] = { 0, ctok, 1, 1 }, fl[4] = { 64 * ctok, 0, 0, 0 }; rls_init(one[a], 4, th0, fl); }
            if ((a == A_MIX || a >= A_K) && one[a].n == 0 && one[A_RING].n > 0) { one[a] = one[A_RING]; one[a].n = 0; one[a].nis = 1; }
            mu[a] = rls_pred(one[a], phi[a], &var[a]);
            rls_draw(one[a], phi[a], rls_mix(rseed + a), &dr[a]);
        }
        // the never-run arms priced by what a prior missed by at this size, as the planner (rls_prior_cal; REPLAY_NOCAL=1: without, the gate's before)
        const auto bkt = [](double n) { int b = 0; while (b < 31 && n >= (double) (2ull << b)) { ++b; } return b; };
        double con[A_N] = {};
        if (!getenv("REPLAY_NOCAL")) {
            int ns[A_N];
            for (int a = 0; a < A_N; ++a) { ns[a] = ok[a] ? one[a].n : 1; }
            double c = 0.0;
            for (int d = 0, b = bkt(e.n); d < 2 && c <= 0.0; ++d) {
                if (cal.count(b - d)) { c = cal[b - d]; } else if (cal.count(b + d)) { c = cal[b + d]; }
            }
            rls_prior_cal(mu, var, ns, A_N, c);
            for (int a = 0; a < A_N; ++a) { con[a] = ns[a] == 0 ? (c > 0.0 ? c : 1.0) : 0.0; }
        }
        double lcb[A_N];
        for (int a = 0; a < A_N; ++a) { lcb[a] = ok[a] ? mu[a] - std::sqrt(std::max(0.0, var[a])) : 0; }
        int Rn = 1;   // this call and the earlier ones of the same size in this start
        for (const auto & q : seen) { Rn += q == e.n; }
        seen.push_back(e.n);
        bool tonly[A_N];   // stream variants that never ran: trials only, never a mean pick (rampool13)
        for (int a = 0; a < A_N; ++a) { tonly[a] = (a == A_MIX || (a >= A_K && a < A_LP)) && one[a].n == 0; }   // layer parallel is a mean pick (the planner's tonly: the mix only)
        bool meas[A_N];
        for (int a = 0; a < A_N; ++a) { meas[a] = ok[a] && rls_known(one[a], mu[a], var[a]); }
        {   // R1: above the measured CPU crossover (g_mb_auto = experts / picks, 64 for the built-in GLM) the chains are only a trial against a known arm
            bool any = false;
            for (int a = 0; a < A_N; ++a) { any = any || (meas[a] && a != A_CHAIN && a != A_DEC); }
            for (const int a : { (int) A_CHAIN, (int) A_DEC }) { if (ok[a] && e.n > e.n_exp / e.n_used && !meas[a] && !any) { ok[a] = false; } }
        }
        const int pick1 = mode == 3 ? rls_voi_pick(mu, var, ok, A_N, Rn, nullptr, meas, tonly) : rls_argmin(mode == 0 ? dr : mode == 1 ? mu : lcb, ok, A_N);
        // the split (dsf_plan): one pass per call here (the logged calls ran as one pass)
        const double sphi[4] = { 1, e.n, 0, e.Ts * 1e3 };
        if (split.k == 0) { const double c = e.split_prior, th0[4] = { 0, c, 1, 1 }, fl[4] = { 64 * c, 0, 0, 0 }; rls_init(split, 4, th0, fl); }   // the logged split prior
        const double est = copies ? 0.0 : e.fill_est;   // the planner's fill estimate (us)
        if (fill.k == 0) { const double o = 1; rls_init(fill, 1, &o, nullptr); }
        double vs, ds, fd = 0, fv, dd1 = mode == 0 ? std::max(0.0, dr[pick1]) : mode == 1 ? mu[pick1] : lcb[pick1];
        double ms = rls_pred(split, sphi, &vs); rls_draw(split, sphi, rls_mix(rseed + 0x51), &ds);
        if (mode == 3 && split.n == 0) { ms = mu[pick1] * RLS_SPLIT_ONE_RATIO; vs = ms * ms * RLS_SPLIT_ONE_SDLOG * RLS_SPLIT_ONE_SDLOG + var[pick1] * RLS_SPLIT_ONE_RATIO * RLS_SPLIT_ONE_RATIO; }   // the data prior (dsf_plan)   // never ran (a loaded posterior keeps its n)
        if (est > 0) { const double fm = rls_pred(fill, &est, &fv); if (mode == 0) { rls_draw(fill, &est, rls_mix(rseed + 0xf1), &fd); fd = std::max(0.0, fd); } else { fd = fm; } }
        const double d2 = mode == 0 ? ds : mode == 1 ? ms : ms - std::sqrt(std::max(0.0, vs));
        int sp = rls_split_choice(dd1 - d2, std::max(0.0, dd1 - d2), fd);
        double m2[2] = {}, v2[2] = {}; bool me2[2] = {};   // mode 3's two arms, kept for the report
        if (mode == 3) {   // two arms for the call: the one-card pick's mean and the split's mean plus the fill owed
            m2[0] = mu[pick1]; m2[1] = ms + (est > 0 ? rls_pred(fill, &est, &fv) : 0); v2[0] = var[pick1]; v2[1] = vs + (est > 0 ? fv : 0);
            const bool o2[2] = { true, true };
            me2[0] = rls_known(one[pick1], m2[0], v2[0]); me2[1] = split.n > 0 && rls_known(split, m2[1] - (est > 0 ? rls_pred(fill, &est, &fv) : 0), vs);
            sp = rls_voi_pick(m2, v2, o2, 2, Rn, nullptr, me2);
            // R2 (planner-crossovers.md): neither side known here - split iff the one-card pass is compute-bound by r over the LINK-bound upload
            // (max(U, C1) > r U; C1 = the one-card mean) and, with REPLAY_R2_FILL, its predicted gain beats the fill owed
            if (getenv("REPLAY_R2") && !me2[0] && !me2[1] && e.U > 0) {
                const double ratio = std::max(e.U * 1e3, m2[0]) / (e.U * 1e3);
                sp = ratio > 1.83 && (!getenv("REPLAY_R2_FILL") || m2[0] - (m2[1] - (est > 0 ? rls_pred(fill, &est, &fv) : 0)) > (est > 0 ? rls_pred(fill, &est, &fv) : 0)) ? 1 : 0;
            }
            if (getenv("REPLAY_DBG") && verbose) { printf("    dbg2 n %.0f R %d one %.0f +- %.0f (n %d) split %.0f +- %.0f (n %d)\n", e.n, Rn, m2[0] / 1e3, std::sqrt(v2[0]) / 1e3, one[pick1].n, m2[1] / 1e3, std::sqrt(v2[1]) / 1e3, split.n); }
            if (getenv("REPLAY_DBG") && verbose) { printf("    dbg n %.0f R %d one %.0f +- %.0f split %.0f +- %.0f\n", e.n, Rn, m2[0] / 1e3, std::sqrt(v2[0]) / 1e3, m2[1] / 1e3, std::sqrt(v2[1]) / 1e3); }
        }
        // outcome and score
        std::string picked = sp ? "dense" : arm_class(pick1);
        if (!sp && !arm_variant(pick1).empty() && T.count(cl + "," + st + "," + arm_variant(pick1))) { picked = arm_variant(pick1); }   // the variant's own measurement where it exists
        auto tr = [&](const std::string & a) { auto it = T.find(cl + "," + st + "," + a); return it == T.end() ? -1.0 : it->second * (e.cls.empty() ? 1.0 : e.n); };   // a model's medians are per token
        // the dense medians are the split pass without its fill (fork/planner_events.py): the fill is added where this replayed start still owes it
        auto call_t = [&](const std::string & a) { const double t = tr(a); return t > 0 && a == "dense" && !copies ? t + e.fill_ms : t; };
        double best = -1; std::string ba;
        for (const char * a : { "stream", "dec", "chain", "dense", "lp", "ring", "mix", "k1", "k2", "k4", "k8", "k16", "k32" }) { const double t = call_t(a); if (t > 0 && (best < 0 || t < best)) { best = t; ba = a; } }
        double took = call_t(picked);
        R.calls++;
        if (!e.model.empty()) { R.models[e.model][0]++; }
        const std::string key = cl + "-" + st + " " + picked;
        auto inputs = [&]() {   // what the pick saw (mode 3): each candidate path's mean +- sd, samples, known here, trial only; then one card against the split
            std::string o;
            char b[256];
            for (int a = 0; a < A_N; ++a) {
                if (!ok[a]) { continue; }
                snprintf(b, sizeof b, "%s%s%s %.1f +- %.1f s (n %d%s%s)", o.empty() ? "" : ", ", a == pick1 ? "*" : "", arm_variant(a).empty() ? arm_class(a) : arm_variant(a).c_str(),
                         mu[a] / 1e6, std::sqrt(std::max(0.0, var[a])) / 1e6, one[a].n, meas[a] ? ", known" : "", tonly[a] ? ", trial only" : "");
                o += b;
            }
            snprintf(b, sizeof b, "      one card or split: one %.1f +- %.1f s%s, split %.1f + fill %.1f = %.1f +- %.1f s%s (split n %d%s) -> %s\n", m2[0] / 1e6, std::sqrt(std::max(0.0, v2[0])) / 1e6,
                     me2[0] ? " known" : "", ms / 1e6, (m2[1] - ms) / 1e6, m2[1] / 1e6, std::sqrt(std::max(0.0, v2[1])) / 1e6, me2[1] ? " known" : "", split.n, split.n == 0 ? ", prior one x 0.96" : "",
                     sp ? "split" : "one card");
            double cf = 0;   // the factor rls_prior_cal put on the never-run arms (1: none measured at this size yet)
            for (int a = 0; a < A_N; ++a) { if (ok[a] && con[a] > 0 && con[a] != 1) { cf = con[a]; } }
            return "      path pick (R " + std::to_string(Rn) + (cf > 0 ? ", never-run priors x " + std::to_string(cf).substr(0, 4) : "") + "): " + o + "\n" + b;
        };
        if (verbose && mode == 3 && getenv("REPLAY_DBG") && !e.model.empty()) { printf("%s", inputs().c_str()); }
        if (took > 0 && best > 0) {
            const double loss = took / best - 1.0;
            R.scored++; R.loss_sum += loss; R.loss_max = std::max(R.loss_max, loss); R.bad += loss > 0.05;
            R.by[key].first++; R.by[key].second += loss;
            if (verbose) { printf("  %-34s req %d %5.0f tok %s: picked %-6s (%s) %6.0f ms, best %-6s %6.0f ms, loss %5.1f%%\n", e.log.c_str(), e.req, e.n, st.c_str(), picked.c_str(), sp ? "split" : pick1 == A_LP ? "layer parallel" : "one card", took, ba.c_str(), best, 100 * loss); }
            if (!e.model.empty()) {
                auto & m = R.models[e.model];
                m[1]++; m[2] += loss > 0.05;
                if (loss > 0.05 && two_cards(picked) != two_cards(ba)) {   // the wrong side of one card against two: the call, the outcome, what the pick saw
                    m[3]++;
                    if (verbose) {
                        const double k = e.n_call > e.n ? e.n_call / e.n : 1.0;   // a call of several passes: its time is the passes' per token
                        auto rng = [&](const std::string & a) {   // the samples behind an outcome, for this call's tokens
                            const auto it = g_range.find(cl + "," + st + "," + a);
                            if (it == g_range.end()) { return std::string(); }
                            char r[96]; snprintf(r, sizeof r, " (n %.0f, %.1f-%.1f s)", it->second[0], it->second[1] * e.n * k / 1e3, it->second[2] * e.n * k / 1e3); return std::string(r);
                        };
                        char b[640];
                        snprintf(b, sizeof b, "  %s %s: %.0f tokens%s, req %d (%s): picked %s (%s) %.1f s%s, best %s %.1f s%s, loss %.0f%%; logged: %s %.1f s\n", cl.c_str(), st.c_str(), k > 1 ? e.n_call : e.n,
                                 k > 1 ? (" in passes of " + std::to_string((int) e.n)).c_str() : "", e.req, e.log.c_str(), picked.c_str(), sp ? "split" : pick1 == A_LP ? "layer parallel" : "one card",
                                 took * k / 1e3, rng(picked).c_str(), ba.c_str(), best * k / 1e3, rng(ba).c_str(), 100 * loss, e.path.c_str(), e.took / 1e3);
                        R.x4.push_back(b + inputs());
                    }
                }
            }
            // the sample the planner learns from (the code's definitions: split pass = call - fill wall; fill = wall)
            if (sp) { rls_update(split, sphi, tr("dense") * 1e3); if (!copies && est > 0) { rls_update(fill, &est, e.fill_ms * 1e3); } copies = true; }
            else    {
                if (con[pick1] > 0.0 && mu[pick1] > 0.0) { cal[bkt(e.n)] = took * 1e3 / (mu[pick1] / con[pick1]); }   // it ran on its prior: the prior's miss
                rls_update(one[pick1], phi[pick1], took * 1e3);
            }
        } else if (verbose) { printf("  %-34s req %d %5.0f tok %s: picked %-6s unscored (never measured there)\n", e.log.c_str(), e.req, e.n, st.c_str(), picked.c_str()); }
    }
    if (verbose && !R.models.empty()) {
        printf("\nper model, the planner's pick (mode %d): calls, scored (the pick and a best measured twice at the call's class), picks losing > 5%%, of those one card against two\n", mode);
        for (const auto & kv : R.models) { printf("  %-7s %5d calls %5d scored %5d losing > 5%% %4d one card against two\n", kv.first.c_str(), kv.second[0], kv.second[1], kv.second[2], kv.second[3]); }
        printf("\none card against two: picks losing > 5%% where the best measured arm runs on the other side (dense, lp = two cards); * = the path pick\n");
        for (const auto & s : R.x4) { printf("%s", s.c_str()); }
        printf("\n");
    }
    return R;
}

int main(int argc, char ** argv) {
    std::vector<ev> E;
    std::map<std::string, double> T;
    if (argc >= 3) {
        std::ifstream f(argv[1]); std::string l; std::getline(f, l);
        while (std::getline(f, l)) {
            auto c = csv_row(l); if (c.size() < 11) continue;
            ev e{ c[0], atoi(c[1].c_str()), atof(c[2].c_str()), atof(c[3].c_str()), atof(c[4].c_str()), atof(c[5].c_str()), atof(c[6].c_str()), atof(c[7].c_str()), c.size() > 11 ? atof(c[11].c_str()) : 1e9, c.size() > 12 ? atof(c[12].c_str()) : 0 };
            if (c.size() >= 22) {   // planner_events.py --all: the model and its machine's numbers
                e.model = c[13]; e.cls = c[14]; e.n_used = atoi(c[15].c_str()); e.n_exp = atoi(c[16].c_str()); e.fill_est = atof(c[17].c_str()) * 1e3; e.fill_ms = atof(c[18].c_str());
                e.split_prior = atof(c[19].c_str()); e.lp = atof(c[20].c_str()); e.n_call = atof(c[21].c_str()); e.path = c[8]; e.took = atof(c[9].c_str());
            }
            E.push_back(e);
        }
        std::ifstream g(argv[2]);
        // a class measured once is no outcome to score against (one cold 2k call of k x8 at 6.5 s would set "the best"): two samples or more
        while (std::getline(g, l)) {
            auto c = csv_row(l);
            if (c.size() >= 4 && (c.size() < 5 || atoi(c[4].c_str()) >= 2)) { T[c[0] + "," + c[1] + "," + c[2]] = atof(c[3].c_str()); }
            if (c.size() >= 7) { g_range[c[0] + "," + c[1] + "," + c[2]] = { atof(c[4].c_str()), atof(c[5].c_str()), atof(c[6].c_str()) }; }
        }
    } else {   // built in: GLM 3.5 12k medians (truth.csv of 2026-11-07) and one start's call features (rampool10 rp0)
        T = { { "10k,c,dense", 15844 - 1478 }, { "10k,c,stream", 17329 }, { "10k,w,dense", 13280 - 1478 }, { "10k,w,stream", 17387 }, { "2k,c,dec", 19304 }, { "2k,c,dense", 15932 - 8904 },
              { "2k,c,stream", 8354 }, { "2k,w,dec", 20248 }, { "2k,w,dense", 8450 - 1397 }, { "2k,w,stream", 7338 } };
        for (int s = 0; s < 4; ++s) {
            const std::string lg = "builtin" + std::to_string(s);
            for (int r = 0; r < 2; ++r) { E.push_back({ lg, r, 2034, 1017, 8244, 7421, 12000, 3350 }); E.push_back({ lg, r, 10400, 7234, 23226 - 15000, 6660, 11688, 18144 }); }
        }
    }
    const int seeds = argc >= 4 ? atoi(argv[3]) : 20;
    g_carry = getenv("REPLAY_CARRY") != nullptr;
    printf("state: %s\n", g_carry ? "carried from the previous start of the same arm (R4)" : "fresh at every start");
    int fails = 0;
    for (int mode = 0; mode < 4; ++mode) {
        result tot; int worst_bad = 0;
        for (int s = 0; s < seeds; ++s) {
            result r = replay(E, T, 1000 + s, mode, s == 0 && argc >= 3 && mode == PICK_MODE);
            tot.calls += r.calls; tot.scored += r.scored; tot.bad += r.bad; tot.loss_sum += r.loss_sum; tot.loss_max = std::max(tot.loss_max, r.loss_max);
            worst_bad = std::max(worst_bad, r.bad);
            for (auto & kv : r.by) { tot.by[kv.first].first += kv.second.first; tot.by[kv.first].second += kv.second.second; }
        }
        printf("%s: %d calls x %d seeds, %d scored, picks losing > 5%%: %d (%.1f%%), mean loss %.1f%%, worst %.1f%%\n", mode == 0 ? "draws (Thompson)" : mode == 1 ? "posterior means" : mode == 2 ? "lower bound mu - sigma" : "value of information (EI R > EL)",
               tot.calls / seeds, seeds, tot.scored, tot.bad, 100.0 * tot.bad / std::max(1, tot.scored), 100.0 * tot.loss_sum / std::max(1, tot.scored), 100.0 * tot.loss_max);
        for (auto & kv : tot.by) { if (E.empty() || E[0].model.empty()) { printf("    %-22s %5d picks, mean loss %5.1f%%\n", kv.first.c_str(), kv.second.first, 100.0 * kv.second.second / kv.second.first); } }   // per class: the GLM gate's (with models: the per-model lists)
        if (mode == PICK_MODE) { fails = tot.bad; }
    }
    printf("gate (the planner's pick, mode %d): %s\n", PICK_MODE, fails ? "FAIL: picks losing more than 5% against the measured best" : "pass");
    return 0;   // a report, not a ctest failure: the gate is read before a build (run.py), the build itself is not blocked by ctest
}
