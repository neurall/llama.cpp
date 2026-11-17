#pragma once
// One learned cost model per prompt path (fork/docs/rampool.md, "Robust planner design"): recursive least squares with its own covariance.
// Shared by the dense split planner (llama-dsplit-fork.cpp: the split pass, the copy fill) and the moe-cache's one-card paths (llama-moecache-fork.cpp):
// both sides of a pick are the same estimator, so neither can be overruled by a differently built number (faults A-G of the design).
//
// A call's time is linear in measured features, t = th . phi, phi = (1, tokens, tokens x cells, the path's calculated upload in us) - or (fill estimate)
// for the copy fill. No tuned constant: the prior is the planner's own calculation with an uncertainty as large as itself, and every later number is a
// measurement weighted by the covariance it earned. Internal header (static inline, no public API).
//
// Errors are relative: each sample is weighted by 1 / its own time (phi / y against 1), so a 4-token call and a 10k-token call teach alike. Replayed over
// every prompt call in runs.db (975 calls, 240 runs, 2026-11-07), absolute least squares let the big calls own the fit and missed the small ones by
// 100-2000% (decode chain median 101%); relative weighting: decode chain 21%, stream 21%, prompt chain 5%, dense 4%, predictions 7% (leave-one-out medians).
//
//   u = phi / y;  e = 1 - th.u;  q = u' P u;  g = P u / (1 + q);  th += g e;  P -= g u' P
//   s2  running mean of e^2 / (1 + q)        (the relative residual variance; the prior s2_0 = RLS_S2_0, the measured repeat noise, counts as sample 0)
//   NIS = e^2 / (s2 (1 + q))                 (1 on average when the model is right)
//   NIS > 1: P *= NIS and the running means restart at n = k + 1: a changed machine re-opens the variance by exactly the surprise, no timers, no windows.
// The parameter covariance is s2 P; a prediction's variance is s2 (mu^2 + phi' P phi).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Priors (weak: each is replaced by the run's own samples as they come; source given for each):
//   RLS_S2_0             the noise, until two samples of a path exist (then its own residuals)
//   RLS_SPLIT_ONE_RATIO  the split's mean before it ran once on this build (then its own RLS)
//   th0 / P0             each path's coefficients from the measured rates (moe-cache, split plan), +- their own size
//   fill coefficient 1   the planner's fill estimate, +- itself, until the first fill
// The prior relative noise of a pass time: the repeat spread measured on the logs, not a tuning. 2026-11-07, GLM 3.5 12k x16+x4, 63 calls of 17 starts
// (up9_merged2 .. rampool10), the same arm, call size and request state: pooled relative MAD 0.030, sd 0.045 (tests/test-planner-replay-fork.cpp). With
// s2_0 = 1 (rampool10) one sample left a path +- 70% wide and a 25% faster split lost the warm 10400-token call to a draw (row 23769, pp 600 -> 497).
static constexpr double RLS_S2_0 = 0.045 * 0.045;

struct dsf_rls {
    int    k = 0;           // features in use (4 for a pass, 1 for the fill)
    double th[4] = {};      // the coefficients
    double P[16] = {};      // their covariance over s2 (row-major k x k inside 4x4)
    double s2 = 0.0;        // relative residual variance; 0 = not primed yet
    double nis = 1.0;       // running mean of the normalised innovation squared
    double e2 = 0.0;        // sum of e^2 / (1 + q) over the samples since the last restart (s2 from the samples alone once two exist)
    int    n = 0;           // samples (restarts at k + 1 on drift)
    bool det() const { return n >= k + 1; }   // identifiable: at least k + 1 samples since the last restart
    // R4: stored for the next process whenever it has a sample - not only once determined and quiet. A request gives a path 1-3 samples, so no
    // one-card path ever reached n >= k + 1 (moe-state.ini 2026-11-07 held dsf.rls.split/fill only), and dropping a posterior after a surprise left
    // the other side's carried posterior unopposed (replay, carried state: 80 of 1700 picks lost up to 31%). The stored diag(P) carries the doubt.
    bool storable() const { return n >= 1; }
};

// P0 = diag(max(th0^2, floor^2)) / s2_0: every prior coefficient is uncertain by its own size (the logged planner errors were 2x).
static inline void rls_init(dsf_rls & r, int k, const double * th0, const double * floor) {
    r = dsf_rls();
    r.k = k;
    for (int i = 0; i < k; ++i) {
        r.th[i] = th0[i];
        const double f = floor ? floor[i] : 0.0;
        r.P[i * 4 + i] = std::fmax(th0[i] * th0[i], f * f) / RLS_S2_0;   // the parameter covariance is s2 P: th0^2 at the prior
    }
    r.s2 = RLS_S2_0;
}

static inline double rls_dot(const dsf_rls & r, const double * phi) {
    double m = 0.0;
    for (int i = 0; i < r.k; ++i) { m += r.th[i] * phi[i]; }
    return m;
}
static inline double rls_q(const dsf_rls & r, const double * phi) {
    double q = 0.0;
    for (int i = 0; i < r.k; ++i) { for (int j = 0; j < r.k; ++j) { q += phi[i] * r.P[i * 4 + j] * phi[j]; } }
    return q;
}

// A coefficient whose feature was 0 so far has a zero row and column in P (no sample moved it), so a zero prior gets its uncertainty when its feature
// first shows up: the coefficient that alone would explain the whole call, (mu / phi_i)^2 (or y's at the first sample when the prior predicts 0). Exact,
// not an approximation: nothing has been learned about it.
static inline void rls_prime(dsf_rls & r, const double * phi, double y_or_0) {
    if (r.s2 <= 0.0) { r.s2 = RLS_S2_0; }
    const double mu = std::fabs(rls_dot(r, phi)), s = mu > 0.0 ? mu : std::fabs(y_or_0);
    for (int i = 0; i < r.k; ++i) {
        if (r.P[i * 4 + i] <= 0.0 && phi[i] != 0.0 && s > 0.0) {
            const double c = s / std::fabs(phi[i]);
            r.P[i * 4 + i] = c * c / r.s2;
        }
    }
}

// mean and predictive variance s2 (mu^2 + phi' P phi) of the time at phi
static inline double rls_pred(dsf_rls & r, const double * phi, double * var) {
    rls_prime(r, phi, 0.0);
    const double mu = rls_dot(r, phi);
    if (var) { *var = r.s2 * (mu * mu + rls_q(r, phi)); }
    return mu;
}

// one measured time y at phi; returns the sample's NIS (-1: not used)
static inline double rls_update(dsf_rls & r, const double * phi, double y) {
    if (!(y > 0.0) || !std::isfinite(y)) { return -1.0; }
    rls_prime(r, phi, y);
    double u[4] = {}, Pu[4] = {};
    for (int i = 0; i < r.k; ++i) { u[i] = phi[i] / y; }
    for (int i = 0; i < r.k; ++i) { for (int j = 0; j < r.k; ++j) { Pu[i] += r.P[i * 4 + j] * u[j]; } }
    double q = 0.0;
    for (int i = 0; i < r.k; ++i) { q += u[i] * Pu[i]; }
    const double e = 1.0 - rls_dot(r, u), d = 1.0 + q;
    const double nis = e * e / (r.s2 * d);
    for (int i = 0; i < r.k; ++i) { r.th[i] += Pu[i] / d * e; }
    for (int i = 0; i < r.k; ++i) { for (int j = 0; j < r.k; ++j) { r.P[i * 4 + j] -= Pu[i] * Pu[j] / d; } }
    r.n++;
    // running means with the prior as sample 0: a single lucky first sample cannot make the model sure of itself
    // the noise: the prior RLS_S2_0 weighs as one sample until two samples exist, then the run's own residuals alone (a weak prior: the first live
    // samples override the logged number)
    r.e2 += e * e / d;
    r.s2  = r.n >= 2 ? r.e2 / (double) r.n : (RLS_S2_0 + r.e2) / (double) (r.n + 1);
    r.nis += (nis - r.nis) / (double) (r.n + 1);
    // Drift: the model is surprised on average, not by one sample (a correct model's single NIS is above 1 a third of the time: chi-square, 1 degree), and
    // only once determined (before n >= k + 1 a large residual is what the samples are still teaching; inflating on it compounded, P x 1000 per early
    // sample once s2_0 is the measured noise). Then P grows by the running surprise and the means restart at k + 1. Covers C6 (a first sample is no drift).
    if (r.n > r.k + 1 && r.nis > 1.0) {
        for (int i = 0; i < r.k * 4; ++i) { r.P[i] *= r.nis; }
        r.n = r.k + 1;
        r.nis = 1.0;
        r.e2 = r.s2 * (double) r.n;
    }
    return nis;
}

// one Gaussian draw of th from N(th, s2 P), applied to phi (Thompson sampling). The seed fixes the draw: the same request and path see the same view.
static inline uint64_t rls_mix(uint64_t x) {   // splitmix64
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
static inline double rls_draw(dsf_rls & r, const double * phi, uint64_t seed, double * t) {
    rls_prime(r, phi, 0.0);
    double mu = rls_dot(r, phi);
    // Cholesky of s2 P (k <= 4); a pivot that rounding left at or below 0 contributes nothing
    double L[16] = {};
    for (int i = 0; i < r.k; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = r.s2 * 0.5 * (r.P[i * 4 + j] + r.P[j * 4 + i]);
            for (int m = 0; m < j; ++m) { s -= L[i * 4 + m] * L[j * 4 + m]; }
            if (i == j) { L[i * 4 + i] = s > 0.0 ? std::sqrt(s) : 0.0; }
            else        { L[i * 4 + j] = L[j * 4 + j] > 0.0 ? s / L[j * 4 + j] : 0.0; }
        }
    }
    double z[4] = {};
    uint64_t x = seed;
    for (int i = 0; i < r.k; ++i) {   // Box-Muller from two uniforms per normal
        x = rls_mix(x); const double u1 = ((double) (x >> 11) + 0.5) / 9007199254740992.0;
        x = rls_mix(x); const double u2 = ((double) (x >> 11) + 0.5) / 9007199254740992.0;
        z[i] = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
    double v = mu;
    for (int i = 0; i < r.k; ++i) { double d = 0.0; for (int m = 0; m <= i; ++m) { d += L[i * 4 + m] * z[m]; } v += d * phi[i]; }
    *t = v;
    return mu;
}

// "th0 th1 th2 th3 | P00 P11 P22 P33 | s2 n nis | P01 P02 P03 P12 P13 P23": the log line and the state file. The off-diagonals carry what the samples
// pinned: one sample fixes the time at its own features to +-6% while every coefficient alone stays near its prior, so diag(P) alone predicted that same
// call +-55% after a restart (GLM 3.5 4k split, n=1: 6.29 +- 3.49 s) and every start re-tried the arm it had measured (MiMo 12k one card +- 18.8 s on 12 s).
// Older files without the last section load as diag(P); full precision because phi' P phi at a measured point is a small difference of large terms.
static inline std::string rls_str(const dsf_rls & r) {
    std::string o;
    char b[64];
    for (int i = 0; i < r.k; ++i) { snprintf(b, sizeof b, "%s%.6g", i ? " " : "", r.th[i]); o += b; }
    o += " |";
    for (int i = 0; i < r.k; ++i) { snprintf(b, sizeof b, " %.17g", r.P[i * 4 + i]); o += b; }
    snprintf(b, sizeof b, " | %.6g %d %.3g |", r.s2, r.n, r.nis);
    o += b;
    for (int i = 0; i < r.k; ++i) { for (int j = i + 1; j < r.k; ++j) { snprintf(b, sizeof b, " %.17g", r.P[i * 4 + j]); o += b; } }
    return o;
}
// back from rls_str (k must match)
static inline bool rls_parse(dsf_rls & r, int k, const std::string & v) {
    dsf_rls o; o.k = k;
    const char * c = v.c_str(); char * e = nullptr;
    for (int i = 0; i < k; ++i) { o.th[i] = strtod(c, &e); if (e == c) { return false; } c = e; }
    while (*c == ' ') { ++c; } if (*c++ != '|') { return false; }
    for (int i = 0; i < k; ++i) { o.P[i * 4 + i] = strtod(c, &e); if (e == c || !(o.P[i * 4 + i] >= 0.0)) { return false; } c = e; }
    while (*c == ' ') { ++c; } if (*c++ != '|') { return false; }
    o.s2 = strtod(c, &e); if (e == c || !(o.s2 > 0.0)) { return false; }
    // the samples behind the stored posterior count (capped at k + 1, so this start's samples still weigh): a loaded path with n = 0 was never
    // 'known here' and never a candidate against a trial, and a loaded split read as 'never ran' got the 0.96 x one-card prior back (R4 replay)
    c = e;
    { char * e2 = nullptr; const long nn = strtol(c, &e2, 10); o.n = e2 != c && nn > 0 ? (int) std::min<long>(nn, k + 1) : 0; c = e2; }
    strtod(c, &e);   // nis: a new start begins its own running mean
    c = e;
    while (*c == ' ') { ++c; }
    if (*c == '|') {   // the off-diagonals (older files end before them: diag(P) only)
        ++c;
        double off[6]; int m = 0; bool ok = true;
        for (int i = 0; i < k && ok; ++i) { for (int j = i + 1; j < k; ++j) { off[m++] = strtod(c, &e); if (e == c) { ok = false; break; } c = e; } }
        for (int i = 0, t = 0; ok && i < k; ++i) { for (int j = i + 1; j < k; ++j, ++t) { o.P[i * 4 + j] = o.P[j * 4 + i] = off[t]; } }
    }
    o.nis = 1.0; o.e2 = o.s2 * (double) o.n;
    r = o;
    return true;
}

// The two decision kernels, shared by the planner (llama-moecache-fork.cpp, llama-dsplit-fork.cpp) and the offline replay (tests/test-planner-replay-fork.cpp),
// so the replay gate runs the code that decides, not a copy of it.
// One-card path: the candidate with the lowest value (a draw, or a mean); -1 when none is eligible.
static inline int rls_argmin(const double * v, const bool * ok, int n) {
    int b = -1;
    for (int i = 0; i < n; ++i) { if (ok[i] && (b < 0 || v[i] < v[b])) { b = i; } }
    return b;
}
// Split against one card for the pass about to run: its gain one - split > 0, and the positive gains of the call's passes beat the copy fill (0 when
// the copies are in place), paid once.
static inline int rls_split_choice(double g_this, double g_pos, double fill) { return g_this > 0.0 && g_pos > fill ? 1 : 0; }

// The split's prior before it ever ran on this build: the one-card mean of the same pass x the split/one time ratio measured across models
// (fork/bench planner dataset 2026-11-07, x16+x4, 32 sizes of DeepSeek, GLM 3.0/3.5, MiMo, Qwen3.6/3.8 with both arms measured: median 0.96, robust
// sd(log) 0.36). The plan's own split time (t_raw_ms) was 1.8x high on GLM 3.5 (2034 us per token against 1135 measured) and kept the split from ever
// being tried; the release's working rule was the same idea (the plan trusted for the ratio, the size from one card measured: rule 41b).
static constexpr double RLS_SPLIT_ONE_RATIO = 0.96, RLS_SPLIT_ONE_SDLOG = 0.36;

// Exploration by value of information (replaces the Thompson draws: rows 23768/23769, a 25% faster split lost the warm call to a draw). Against the best
// mean b, an arm t ~ N(mu, var) is expected to gain EI = E[max(0, b - t)] per call and to lose EL = E[max(0, t - b)] = EI + mu - b on the trial. It is
// tried when the gain over the calls this decision recurs in (R: this call and the earlier ones like it in the process) beats the trial's loss:
// EI x R > EL. A known slower arm (sigma = the noise) is never tried, an unknown one only where it can pay; R grows as the decision repeats. No constant.
// (the 5% loss bound is the replay gate's and the cells' acceptance metric, tests/test-planner-replay-fork.cpp; at run time the stake decides)
static inline double rls_ncdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
static inline double rls_ei(double mu, double var, double b) {
    const double sd = std::sqrt(std::fmax(0.0, var));
    if (sd <= 0.0) { return std::fmax(0.0, b - mu); }
    const double z = (b - mu) / sd;
    // a pass cannot take less than 0, so the gain over the best arm b is at most b: an arm with a wild posterior (MiMo 64k, 2026-11-08: one card
    // 1170 s against the split's 19.7 s, variance in the thousands) otherwise wins a trial through its variance alone
    return std::min(std::fmax(0.0, b), (b - mu) * rls_ncdf(z) + sd * std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.141592653589793));
}
// Known here: the model's parameter uncertainty at this phi is no larger than the noise (phi' P phi <= mu^2, i.e. var <= 2 s2 mu^2): the arm ran at
// comparable features. A measured arm extrapolated far from where it ran is not known here (the split measured at 10k says little at 2k).
// Two samples at least (user 2026-11-08: "decisions on such noisy data? sample 2-3 times to denoise"): MiMo 64k warm let ONE sample of the decode chain
// (a 137x miss of its calculation) stand as that arm's mean and posterior, and the next calls priced from it.
static inline bool rls_known(const dsf_rls & r, double mu, double var) { return r.n > 1 && var <= 2.0 * r.s2 * mu * mu; }

// The arms that never ran (n_samples 0) priced by what a prior missed by where one ran at this size (cal = took / its prior's estimate; 0: none yet): the
// priors share their inputs (SSD rate, resident bytes, link rates), so one measured miss is every prior's (user 2026-11-15: "planner shall calculate it").
// K3 1k prompt: the stream's k x1 arm estimated 45.9 s on its prior and took 70.4 s; only its own model learned it, the untried arms kept the same 35%-low
// prior (71.6 s against its 95.2 s at the next call) and layer parallel won the next call: 77.4 s.
// built (optional): the compute per token each arm's prior was built from, cal_built the one the prior that missed had. An arm built again since from a measured
// compute is not priced by the old miss: that miss was the guess's, the measurement replaced it (GLM 12k on one Vulkan card, 2026-11-17: layer parallel's prior
// at a guessed 12 ms a token missed x0.19, the stream rebuilt at the measured 2.3 ms got x0.19 on top: planned 6.1 s, took 29.3, request 2 left the faster path)
static inline void rls_prior_cal(double * mu, double * var, const int * n_samples, int n, double cal, const double * built = nullptr, double cal_built = 0.0) {
    if (!(cal > 0.0)) { return; }
    for (int i = 0; i < n; ++i) { if (n_samples[i] == 0 && (!built || built[i] == cal_built)) { mu[i] *= cal; var[i] *= cal * cal; } }
}

// The pick. Among the arms known here: the lowest mean (a learned estimate is never overruled: no tie margin, no veto). Any other arm (a prior, or a
// model extrapolated far from where it ran) is a trial, run only when its stake pays: expected gain over the calls of this size so far against its
// expected loss, EI x R >= EL - all three learned. With no arm known here the lowest mean stands (nothing better is known).
// explore_unknown: with nothing known here the lowest mean is still the reference and the other arms get the same stake test (GLM 3.0 12k, rampool13:
// one card never became known, so the split's prior one x 0.96 + fill stayed a hair above it on every pass and was never tried - a deadlock, since only a
// run makes it known; rel_b14244 ran the split there at 783 against 663 tokens/s). Off for the prompt paths, whose never-run variants flip on noise.
static inline int rls_voi_pick(const double * mu, const double * var, const bool * ok, int n, double R, bool * trial, const bool * known = nullptr,
                               const bool * trial_only = nullptr, bool explore_unknown = false) {
    if (trial) { *trial = false; }
    int b = -1;
    if (known) {
        for (int i = 0; i < n; ++i) { if (ok[i] && known[i] && (b < 0 || mu[i] < mu[b])) { b = i; } }
    }
    if (b < 0) {   // nothing known here: the lowest mean, never an arm that is a trial only (a stream variant that never ran: rampool12 flipped between them)
        for (int i = 0; i < n; ++i) { if (ok[i] && !(trial_only && trial_only[i]) && (b < 0 || mu[i] < mu[b])) { b = i; } }
        if (b < 0) { return rls_argmin(mu, ok, n); }
        if (!explore_unknown) { return b; }
    }
    int t = b; double best = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!ok[i] || i == b) { continue; }
        const double ei = rls_ei(mu[i], var[i], mu[b]), el = ei + mu[i] - mu[b];
        if (ei * R - el >= best && ei * R >= el) { best = ei * R - el; t = i; }
    }
    if (trial) { *trial = t != b; }
    return t;
}

// One card against the split for the passes left of a call: the split planner's pick (dsf_plan, llama-dsplit-fork.cpp) and the replay gate's
// (tests/test-planner-replay-fork.cpp), one code (user 2026-11-15: "make picking one or 2 gpus unit test ... picks correctly").
// Per pass: its tokens T, the link-bound expert upload U (us), the one-card estimate (the moe-cache's pick, or the split planner's own one-card model where
// that is narrower; kn1 known here, n1 its samples) and the split model's prediction (unused while the split never ran). Per call: the bench's one-card
// compute, the split / one ratio, the split's model (rls_known), the fill owed and how often a one-pass call of this size came before in this process.
struct dsf_arm_pass { double T = 0.0, U = 0.0, mu1 = 0.0, v1 = 0.0, mu2 = 0.0, v2 = 0.0; bool kn1 = false; int n1 = 0;
                      double m1 = 0.0, m2 = 0.0; int nm1 = 0, nm2 = 0; };   // m1, m2: the pass's time from the passes measured at this size (us), nm1, nm2 their count; 0: none
struct dsf_call {
    double c_one_bench = 0.0;             // the bench's one-card compute per token (us), 0: none
    double ratio = RLS_SPLIT_ONE_RATIO;   // the split / one-card compute ratio (the plan's matmul bench and link probe, else the measured constant)
    bool split_prior = true;              // the split never ran on this build: its prior
    const dsf_rls * split = nullptr;      // the split's model (known here: rls_known)
    double f_mu = 0.0, f_v = 0.0;         // the fill owed (us, us^2); 0: the copies are in place
    int seen_before = 0;                  // calls of one pass in this size class (dsf_size_class) decided before in this process
};
struct dsf_pick_out {
    int choice = 0, np = 0;               // 0 one card, 1 the split; the passes priced
    bool trial = false, upload_bound = false, known1 = false, known2 = false;
    double stake = 1.0, am[2] = {}, av[2] = {};                         // one card, the split plus the fill, over the passes left
    double mu1_0 = 0.0, v1_0 = 0.0, mu2_0 = 0.0, v2_0 = 0.0, u0 = 0.0;  // the first pass's, priced (the log's)
};
// The split prior's compute ratio (split / one card) from the plan's bench: at most 1 (rule 41b: trusted for the ratio, never for the size; -1: no plan).
// Where the hand-offs block (the start's copy probe, dsplit_xcopy_probe) the plan prices their bytes but not the drains around them, so it claims no gain: 1.
// GLM 12k on two Vulkan cards (2026-11-17): the plan said 0.77, the split measured 28.4 s, and the mirror held the never-run one card at 28.4 / 0.77 = 37 s;
// one card runs 26.5 s
static inline double dsf_prior_ratio(double c_split, double c_one, bool blocks) {
    if (!(c_split > 0.0 && c_one > 0.0)) { return -1.0; }
    return blocks ? 1.0 : std::min(1.0, c_split / c_one);
}
static inline int dsf_size_class(int64_t n_tokens) { return (int) std::floor(4.0 * std::log2((double) std::max<int64_t>(1, n_tokens))); }   // a quarter octave
static inline dsf_pick_out dsf_pick(const std::vector<dsf_arm_pass> & passes, const dsf_call & c) {
    dsf_pick_out o;
    const double sd2 = RLS_SPLIT_ONE_SDLOG * RLS_SPLIT_ONE_SDLOG;
    double m1 = 0.0, v1s = 0.0, m2 = 0.0, v2s = 0.0;
    for (const dsf_arm_pass & p : passes) {
        double mu1 = p.mu1, v1 = p.v1, mu2 = p.mu2, v2 = p.v2;
        // MiMo 12k relcand2: one card measured 11.98 +- 0.76 s, the cap made it 11.97 +- 4.31, nothing was known, and the wide one card won a trial every request; the split (9 s) never ran.
        // So the cap is for an arm never measured (n1 0), not one measured once (rls_known wants more). One card never measured here: the moe-cache's cold prior (MiMo 3k,
        // no-map-reads 2026-11-08: 8.7 s for a pass b14929 ran in 4.0-4.3 s on one card) sent every call to the split (5.2-5.5 s) and one pass of stake never paid the trial.
        // Same rule as the split's prior: max(upload, the bench's one-card compute), uncertain by RLS_SPLIT_ONE_SDLOG
        if (!p.kn1 && p.n1 == 0 && c.c_one_bench > 0.0) {
            const double b = std::max(p.U, p.T * c.c_one_bench);
            if (b < mu1) { mu1 = b; v1 = b * b * sd2; }
        }
        if (c.split_prior) {
            // The split shortens the compute only: a pass bound by its expert upload U gains nothing from it (the release's rule, dsf_decide: max(U, compute)).
            // So the prior is max(U, one card x the bench's split/one compute ratio), not a flat ratio: at 2k GLM 3.0 one card is its 4.5 s upload and the
            // split's fill can only lose, at 10k one card is 14 s of compute and the split saves its share from the first pass (rel_b14244 split there,
            // 783 against 663 tokens/s; rampool13's flat 0.96 prior sat above one card by the fill and was never tried). The crossover is where the
            // compute passes the upload: computed per pass from measured link rate and bytes, no size constant.
            mu2 = std::max(p.U, mu1 * c.ratio);
            v2  = mu2 * mu2 * sd2 + v1 * c.ratio * c.ratio;
        }
        // the mirror of the split's prior: a split measured against a one card never measured, its bench figure counts neither the hand-offs nor the other card's layers
        // (GLM 3.0 12k, first request: split 6.46 s measured, one card 4.90 +- 1.76 from the bench, picked, took 9.4 s)
        if (!p.kn1 && p.n1 == 0 && !c.split_prior && mu2 / c.ratio > mu1) { mu1 = std::max(p.U, mu2 / c.ratio); v1 = mu1 * mu1 * sd2; }
        // the passes measured at this size price the pass where each arm ran twice or more here (two samples: the user's denoise rule, rls_known): the models fit
        // every size at once and miss at some (qn1m 3536 tokens, crossover audit X4: the split's model said 2.06 s, the split measured 2.55 s at this size and
        // took 2.71, one card measured 2.17); one sample of an arm stays with its model (qni3 2442: the split measured once at 1253 t/s, ran at 2025)
        bool m_kn1 = false, m_kn2 = false;
        if (p.nm1 >= 2 && p.m1 > 0.0 && p.nm2 >= 2 && p.m2 > 0.0) {
            mu1 = p.m1; v1 = std::min(v1, mu1 * mu1 * 0.06 * 0.06); m_kn1 = true;
            mu2 = p.m2; v2 = std::min(v2 > 0.0 ? v2 : mu2 * mu2, mu2 * mu2 * 0.06 * 0.06); m_kn2 = true;
        }
        if (o.np == 0) {
            o.mu1_0 = mu1; o.v1_0 = v1; o.mu2_0 = mu2; o.v2_0 = v2; o.u0 = p.U;
            o.known1 = p.kn1 || m_kn1; o.known2 = m_kn2 || (!c.split_prior && c.split && rls_known(*c.split, mu2, v2));
        }
        m1 += mu1; v1s += v1; m2 += mu2; v2s += v2;
        ++o.np;
    }
    o.am[0] = m1; o.am[1] = m2 + c.f_mu; o.av[0] = v1s; o.av[1] = v2s + c.f_v;
    // No trial of the split on a call its expert upload bounds (prior = U): the split shortens the compute only, so the trial cannot win there, and its sample
    // teaches the split model an upload-bound time it then extrapolates (GLM 3.5 12k, both-splits 2026-11-08: the 2194-token first call, U 5.32 s, was the
    // trial; the 10240-token call then saw split 17.0 +- 9.2 s against one card 6.0 and stayed one card, 648 against 821-839 t/s on the builds that split).
    // And no trial where even a perfect split saves less than the fill it owes: it can only shorten the compute above the upload (GLM 3.0, 100-token prompt: one card 3.99 s,
    // upload 3.76 s, fill 0.74 s; the trial ran the split with the fill, 5.8 s against about 4)
    o.upload_bound = c.split_prior && (o.mu1_0 * c.ratio <= o.u0 || (o.mu1_0 - o.u0) * (double) std::max(1, o.np) <= c.f_mu);
    const bool aok[2] = { true, !o.upload_bound }, akn[2] = { o.known1, o.known2 };
    // the trial's stake: the passes left in this call. The count of passes of this size seen so far grew with every pass and forced a trial of the
    // never-known one-card arm once per request (GLM 3.5 64k, 2026-11-08: a 25.9 s one-card pass at 36k cells against 11.9 s split, 10% of the prompt);
    // what a trial learns pays over the passes still to come here, and the next request starts from the stored posterior (R4) anyway.
    // A one-pass call is the exception (user 2026-11-08, Qwen3.8 3.5k edit prompt: the dense split ran 1127 t/s for every repeat, 3.1 s, while one card, never tried, is 2.4 s in b14244; its prior
    // was a cold 11.4 s and one pass of stake never paid a trial): its next calls come with the same features, so the trial's two samples make the arm known and end the trials, and the
    // calls of this size class so far are the stake. Multi-pass calls keep the passes left: their features move with every pass, the arm never becomes known (GLM 3.5 64k).
    // + 1 for a one-pass call: a trial's sample is stored with the posteriors (dsf.rls.*), so it also pays the next call of this size, in this process or the
    // next one. Without it a stake of 1 never tries anything (EI x 1 >= EI + mu_t - mu_b only where the trial is already the lower mean), and every server
    // starts with seen_before 0: GLM 3.0 12k ran one card on every build since rampool11 against 782 t/s split on b14244 (prior one x 1.00 + fill, p_split 0.45-0.49)
    o.stake = o.np == 1 ? 2.0 + (double) c.seen_before : (double) std::max(1, o.np);
    // the split is tried where its stake pays even before one card is known here. Only when one card has the lower mean: with the split already lower the same step ran the reverse trial,
    // a whole first request on one card (first / second request, 200 W: GLM 3.0 12k 538 / 703 t/s, Qwen3.8 IQ4_XS 13.7k 1387 / 1782)
    o.choice = rls_voi_pick(o.am, o.av, aok, 2, o.stake, &o.trial, akn, nullptr, o.am[1] >= o.am[0]);
    return o;
}

// INV phi (I6): a call's end teaches its arm only when none of its passes did (llama_moe_cache_pp_pass_learn_fork) and it streamed, and only then must its
// average pass be the size its features were built for. A prompt of calls decided at different sizes averages between them (GLM on two Vulkan cards,
// 2026-11-16: trial passes of 1664 tokens, then 1536; 9472 tokens in 6 passes is 1579 a pass, the passes had taught the arm, the check failed the row)
static inline bool rls_report_phi_off(double planned, double n1, int learned, bool streamed) {
    return learned == 0 && streamed && std::fabs(planned - n1) > 32.0;
}

// the normal CDF, for the logged p_split only (never a decision input)
static inline double rls_phi_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
