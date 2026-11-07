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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

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
    int    n = 0;           // samples (restarts at k + 1 on drift)
    bool det() const { return n >= k + 1; }   // identifiable: at least k + 1 samples since the last restart
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
    r.s2  += (e * e / d - r.s2) / (double) (r.n + 1);
    r.nis += (nis - r.nis) / (double) (r.n + 1);
    // Drift: the model is surprised on average, not by one sample (a correct model's single NIS is above 1 a third of the time: chi-square, 1 degree), and
    // only once determined (before n >= k + 1 a large residual is what the samples are still teaching; inflating on it compounded, P x 1000 per early
    // sample once s2_0 is the measured noise). Then P grows by the running surprise and the means restart at k + 1. Covers C6 (a first sample is no drift).
    if (r.n > r.k + 1 && r.nis > 1.0) {
        for (int i = 0; i < r.k * 4; ++i) { r.P[i] *= r.nis; }
        r.n = r.k + 1;
        r.nis = 1.0;
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

// "th0 th1 th2 th3 | P00 P11 P22 P33 | s2 n nis": the log line and the state file (diag(P) only: the prior a next start begins from)
static inline std::string rls_str(const dsf_rls & r) {
    std::string o;
    char b[64];
    for (int i = 0; i < r.k; ++i) { snprintf(b, sizeof b, "%s%.6g", i ? " " : "", r.th[i]); o += b; }
    o += " |";
    for (int i = 0; i < r.k; ++i) { snprintf(b, sizeof b, " %.6g", r.P[i * 4 + i]); o += b; }
    snprintf(b, sizeof b, " | %.6g %d %.3g", r.s2, r.n, r.nis);
    return o + b;
}
// back from rls_str (k must match); the loaded state is a prior: n restarts at 0 so the means and NIS keep their weight for this start's samples
static inline bool rls_parse(dsf_rls & r, int k, const std::string & v) {
    dsf_rls o; o.k = k;
    const char * c = v.c_str(); char * e = nullptr;
    for (int i = 0; i < k; ++i) { o.th[i] = strtod(c, &e); if (e == c) { return false; } c = e; }
    while (*c == ' ') { ++c; } if (*c++ != '|') { return false; }
    for (int i = 0; i < k; ++i) { o.P[i * 4 + i] = strtod(c, &e); if (e == c || !(o.P[i * 4 + i] >= 0.0)) { return false; } c = e; }
    while (*c == ' ') { ++c; } if (*c++ != '|') { return false; }
    o.s2 = strtod(c, &e); if (e == c || !(o.s2 > 0.0)) { return false; }
    o.n = 0; o.nis = 1.0;
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
static constexpr double RLS_TRIAL_LOSS_MAX = 0.05;   // the user's rule (2026-11-07): a wrong pick may cost at most 5% of the call against the measured best
static inline double rls_ncdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
static inline double rls_ei(double mu, double var, double b) {
    const double sd = std::sqrt(std::fmax(0.0, var));
    if (sd <= 0.0) { return std::fmax(0.0, b - mu); }
    const double z = (b - mu) / sd;
    return (b - mu) * rls_ncdf(z) + sd * std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.141592653589793);
}
// Known here: the model's parameter uncertainty at this phi is no larger than the noise (phi' P phi <= mu^2, i.e. var <= 2 s2 mu^2): the arm ran at
// comparable features. A measured arm extrapolated far from where it ran is not known here (the split measured at 10k says little at 2k).
static inline bool rls_known(const dsf_rls & r, double mu, double var) { return r.n > 0 && var <= 2.0 * r.s2 * mu * mu; }

// The pick (planner-simple). Among the arms known here: the lowest mean, a tie within the combined spread to the lower variance (the release's
// margin). An arm not known here (a prior, or a model extrapolated) replaces the best known only as a trial; with no arm known here the lowest mean
// stands (nothing better is known). A trial (any arm other than that pick) runs only when its expected loss EL is at most RLS_TRIAL_LOSS_MAX of the
// call (the user's rule: a wrong pick never costs more than 5%) and its expected gain over the calls of this size so far pays for it: EI x R >= EL.
static inline int rls_voi_pick(const double * mu, const double * var, const bool * ok, int n, double R, bool * trial, const bool * known = nullptr) {
    if (trial) { *trial = false; }
    int b = -1;
    if (known) {
        for (int i = 0; i < n; ++i) { if (ok[i] && known[i] && (b < 0 || mu[i] < mu[b])) { b = i; } }
        for (int i = 0; b >= 0 && i < n; ++i) {
            if (ok[i] && known[i] && i != b && var[i] < var[b] && mu[i] - mu[b] < std::sqrt(var[i] + var[b])) { b = i; }
        }
    }
    if (b < 0) { return rls_argmin(mu, ok, n); }   // nothing known here: the lowest mean
    int t = b; double best = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!ok[i] || i == b) { continue; }
        const double ei = rls_ei(mu[i], var[i], mu[b]), el = ei + mu[i] - mu[b];
        if (el > RLS_TRIAL_LOSS_MAX * mu[b]) { continue; }
        if (ei * R - el >= best && ei * R >= el) { best = ei * R - el; t = i; }
    }
    if (trial) { *trial = t != b; }
    return t;
}

// the normal CDF, for the logged p_split only (never a decision input)
static inline double rls_phi_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
