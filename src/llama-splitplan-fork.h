// The dense split planner's model of one split layer and its search, pure functions of per-layer numbers so tests/test-split-fork.cpp can run them with no GPU and no
// model. llama-dsplit-fork.cpp (dsplit_plan_compute) fills dsx_in from the model and the probe and calls dsx_plan; the reasoning behind each term is in the comments below.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

struct dsx_in {
    int64_t T = 0;                       // tokens of the pass
    int     L = 0;                       // layers
    std::vector<double> w, xf, E;        // per layer: compute weight (elements), share of it that is experts, expert bytes
    std::vector<double> kv_b, att_w;     // per layer: bytes of its KV cache the other card's range reads, share of the context attention
    std::vector<double> held_m, held_h;  // per layer: fraction of the experts already resident on the offload card / on the helper
    double w_sum = 0.0;
    double c_base = 0.0, att_tok = 0.0;  // compute per token without / with the part that is attention over the context (us)
    double rel = 1.0;                    // the helper's compute against the offload card's
    double Bm = 0.0, Bh = 0.0;           // upload rate of the offload card / the helper (GB/s)
    double b_in = 0.0, b_out = 0.0;      // bytes per token that cross per layer, there and back
};
struct dsx_out { double t_one = 0.0, best_split = -1.0, best_s = 0.55; std::vector<double> best_f; };

// One split layer, as the graph runs it (build_moe_ffn, range mode). Times in us from the start of the layer:
//   dense    both cards at once, each on its own tokens: the helper s x D / rel, the offload card (1 - s) x D
//   hand-in  an expert chain reads the hidden state of ALL tokens, so it starts when its own card's dense part is done AND the other card's tokens
//            have crossed the helper's link. This is what the first version of this model missed: it let the offload card compute experts while the
//            helper was still in its dense part, and so put 80-85% of the tokens on the helper. Measured, Qwen3.8-Flash-Next, 8.8k tokens, no experts
//            on the x4: 85% of the tokens there 135 ms per layer, 77% 127 ms (model then: 89 ms; this one: 150 and 142, the same slope).
//   chains   the offload card (1 - f) x X, the helper f x X / rel; a chain cannot end before its card's uploads of this layer have arrived
//   hand-out every range gets the other card's sum for its tokens back over the helper's link
//   link     the helper's link carries its expert uploads and all four hand-offs: the layer is never shorter than their sum
// s: the helper's share of the tokens, f: the helper's share of the layer's experts.
static inline double dsx_layer_us(const dsx_in & in, int il, double s, double f) {
    const double T = (double) in.T;
    const double cb = in.c_base * T * in.w[il] / in.w_sum;
    const double D = cb * (1.0 - in.xf[il]) + in.att_tok * in.att_w[il] * T, X = cb * in.xf[il];
    const double kv_in = in.kv_b[il] / (in.Bh * 1e3);   // the cache of this layer to the range on the other card
    const double tok_h = s * T, tok_m = (1.0 - s) * T;
    const double dense_h = s * D / in.rel + kv_in, dense_m = (1.0 - s) * D;
    const double in_m  = f < 1.0 ? tok_h * in.b_in  / (in.Bh * 1e3) : 0.0;   // the helper's tokens, to the offload card's chain
    const double in_h  = f > 0.0 ? tok_m * in.b_in  / (in.Bh * 1e3) : 0.0;   // the offload card's tokens, to the helper's chain
    const double out_h = f < 1.0 ? tok_h * in.b_out / (in.Bh * 1e3) : 0.0;   // the offload card's sum for the helper's tokens
    const double out_m = f > 0.0 ? tok_m * in.b_out / (in.Bh * 1e3) : 0.0;   // the helper's sum for the offload card's tokens
    const double up_m  = in.E[il] * (1.0 - f) * (1.0 - in.held_m[il]) / (in.Bm * 1e3);
    const double up_h  = in.E[il] * f         * (1.0 - in.held_h[il]) / (in.Bh * 1e3);
    const double end_m = f < 1.0 ? std::max(std::max(dense_m, dense_h + in_m) + (1.0 - f) * X,   up_m) : dense_m;
    const double end_h = f > 0.0 ? std::max(std::max(dense_h, dense_m + in_h) + f * X / in.rel,  up_h) : dense_h;
    const double t = std::max(std::max(end_h, end_m + out_h), std::max(end_m, f > 0.0 ? end_h + out_m : 0.0));
    return std::max(t, up_h + in_m + in_h + out_h + out_m + kv_in);
}

// One card's time (every layer on the offload card), and the best token share s of the helper with the best expert share f per layer. Experts move to the helper
// only for a clear win (10% of the layer): its hops and their stream synchronisations cost more than the model counts (Qwen3.8-Flash-Next, 3k tokens: 5% of the
// experts on the x4 cost about 3 ms per layer, measured). A dense layer (no experts) never gets any.
static inline dsx_out dsx_plan(const dsx_in & in) {
    dsx_out o;
    o.best_f.assign(in.L, 0.0);
    std::vector<double> cur_f(in.L, 0.0);
    for (int il = 0; il < in.L; ++il) {
        const double cl = (in.c_base * in.w[il] / in.w_sum + in.att_tok * in.att_w[il]) * (double) in.T;   // us: the layer on one card
        const double up = in.E[il] * (1.0 - in.held_m[il]) / (in.Bm * 1e3);
        o.t_one += std::max(cl, up);
    }
    for (double s = 0.20; s <= 0.801; s += 0.02) {
        double sum = 0.0;
        for (int il = 0; il < in.L; ++il) {
            double best_l = -1.0, bf = 0.0;
            for (int k = 0; k <= 20; ++k) {
                if (in.E[il] <= 0.0 && k > 0) { break; }   // a dense layer: no experts to share
                const double t = dsx_layer_us(in, il, s, k / 20.0);
                if (best_l < 0.0 || (k == 0 ? t < best_l : t < 0.90 * best_l)) { best_l = t; bf = k / 20.0; }
            }
            cur_f[il] = bf;
            sum += best_l;
        }
        if (o.best_split < 0.0 || sum < o.best_split) { o.best_split = sum; o.best_s = s; o.best_f = cur_f; }
    }
    return o;
}

// ---- any number of cards (2026-10-23): the shares of the live balance and the planner's model, for N cards of any compute, link and memory ----------------
// A split pass gives every card that takes part one range of the tokens; one card, the expert card x (the fastest link), computes the experts of every
// layer for all tokens (the dense split alone; experts on the other cards are a measured trial, llama-dsplit-fork.cpp). Per card i, in microseconds:
//   a_i  per token of its range: its dense compute (measured: its dense time over its tokens; planned: the expert card's compute over its probed speed),
//        plus, for every card but x, the hand-offs of that token once per layer with experts over the slower of its own and x's link: its hidden state to
//        x (b_in bytes) and the experts' sum back (b_out). The way back holds up its next layer as much as the way there holds up the experts: both count.
//        This is what the first balance missed: on 2x RTX 3090 (x16 + x4) it equalised the dense compute alone and put 52-54% of the tokens on the x4 card,
//        where a fixed 45% was the fastest of 15/30/45/60% (GLM-5.3-Flash 3.5-bit, 15k: 420 prompt t/s against 387-389 at 30% and 60%).
//   k_i  per pass whatever its share: the KV cache of every attention layer that another card holds, read over the slower of the two links.
// The cards take equally long when k_i + s_i T a_i = lambda for every card that has a share and the shares add up to 1, s_i = (lambda - k_i) / (T a_i).
// A card takes part only when even min_share of the tokens does not make it the slowest: lambda >= k_i + min_share T a_i, its threshold (a card so slow,
// or whose fixed cost is so high, that it would hold the others up with a few tokens gets none). The sum of the shares grows with lambda and jumps by at
// least min_share where a card's threshold is passed, so lambda is the smallest value where the sum reaches 1 (found by bisection); when that is at a jump,
// the card that joins there takes what is left (it finishes a little early, the others at lambda). The expert card always keeps at least min_share (the
// router and the expert chains of the pass run there).
struct dsb_card { double a = 0.0, k = 0.0; };
static inline std::vector<double> dsb_target(const std::vector<dsb_card> & c, double T, double min_share, int x) {
    const int n = (int) c.size();
    std::vector<double> s(n, 0.0);
    if (n == 0) { return s; }
    if (T <= 0.0) { T = 1.0; }
    auto share = [&](int i, double lam) {   // card i's share at lambda (0 below its threshold)
        if (c[i].a <= 0.0) { return 0.0; }
        const double v = (lam - c[i].k) / (T * c[i].a);
        if (i == x) { return std::max(min_share, v); }
        return lam >= c[i].k + min_share * T * c[i].a ? v : 0.0;
    };
    auto total = [&](double lam) { double t = 0.0; for (int i = 0; i < n; ++i) { t += share(i, lam); } return t; };
    double lo = 0.0, hi = 0.0;
    for (int i = 0; i < n; ++i) { if (c[i].a > 0.0) { hi = std::max(hi, c[i].k + T * c[i].a); } }   // any card alone with all the tokens
    if (hi <= 0.0) { if (x >= 0 && x < n) { s[x] = 1.0; } return s; }
    if (total(lo) >= 1.0) { hi = lo; }
    for (int it = 0; it < 200 && hi - lo > 1e-12 * hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        (total(mid) >= 1.0 ? hi : lo) = mid;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { s[i] = share(i, hi); sum += s[i]; }
    if (sum > 1.0 + 1e-9) {   // at a jump: the cards that joined between lo and hi take what is left
        double joined = 0.0;
        for (int i = 0; i < n; ++i) { if (i != x && s[i] > 0.0 && share(i, lo) == 0.0) { joined += s[i]; } }
        const double left = std::max(0.0, joined - (sum - 1.0));
        for (int i = 0; i < n; ++i) { if (i != x && s[i] > 0.0 && share(i, lo) == 0.0) { s[i] *= joined > 0.0 ? left / joined : 0.0; } }
        sum = 0.0;
        for (double v : s) { sum += v; }
    }
    if (sum <= 0.0) { s.assign(n, 0.0); if (x >= 0 && x < n) { s[x] = 1.0; } return s; }
    for (double & v : s) { v /= sum; }
    return s;
}

// The tokens of every card for a pass of T tokens: s_i T, rounded so that they add up to T (largest remainder). A card that would get fewer than min_tok
// gives its share to the others (a range that small costs more in hand-offs than it saves); the expert card keeps at least min(min_tok, T).
static inline std::vector<int64_t> dsb_counts(int64_t T, const std::vector<double> & s, int64_t min_tok, int x) {
    const int n = (int) s.size();
    std::vector<double>  w(s);
    std::vector<int64_t> out(n, 0);
    if (T <= 0 || n == 0) { return out; }
    for (int round = 0; round <= n; ++round) {
        double sum = 0.0;
        for (double v : w) { sum += std::max(0.0, v); }
        if (sum <= 0.0) { out.assign(n, 0); if (x >= 0 && x < n) { out[x] = T; } return out; }
        int64_t used = 0;
        std::vector<std::pair<double, int>> rem;
        for (int i = 0; i < n; ++i) {
            const double e = std::max(0.0, w[i]) / sum * (double) T;
            out[i] = (int64_t) std::floor(e);
            used  += out[i];
            if (w[i] > 0.0) { rem.push_back({ e - std::floor(e), i }); }
        }
        std::sort(rem.begin(), rem.end(), [](const std::pair<double, int> & p, const std::pair<double, int> & q) {
            return p.first > q.first || (p.first == q.first && p.second < q.second);
        });
        for (size_t j = 0; used < T && !rem.empty(); j = (j + 1) % rem.size()) { out[rem[j].second]++; ++used; }
        int worst = -1;
        for (int i = 0; i < n; ++i) {
            if (i != x && w[i] > 0.0 && out[i] < min_tok && (worst < 0 || out[i] < out[worst])) { worst = i; }
        }
        if (worst < 0) { break; }
        w[worst] = 0.0;
    }
    if (x >= 0 && x < n) {   // the expert card's minimum, taken from the biggest ranges; a range that falls under min_tok by that goes to the expert card whole
        const int64_t need = std::min<int64_t>(min_tok, T);
        while (out[x] < need) {
            int big = -1;
            for (int i = 0; i < n; ++i) { if (i != x && out[i] > 0 && (big < 0 || out[i] > out[big])) { big = i; } }
            if (big < 0) { out[x] = T; break; }
            const int64_t take = std::min(need - out[x], out[big]);
            out[big] -= take; out[x] += take;
        }
        for (int i = 0; i < n; ++i) { if (i != x && out[i] > 0 && out[i] < min_tok) { out[x] += out[i]; out[i] = 0; } }
    }
    return out;
}

// One step of the live balance, from the shares s the measured pass ran with toward the target: the first measured pass of a size goes all the way (the cards'
// speeds are known only now), later passes half way (one pass's times have noise). Nothing moves while the cards' times t_i (those with a share) are within
// `hold` of each other, relative to the longest. Returns that gap.
static inline double dsb_step(std::vector<double> & s, const std::vector<double> & target, const std::vector<double> & t, bool first, double hold) {
    double tmax = 0.0, tmin = -1.0;
    for (size_t i = 0; i < t.size() && i < s.size(); ++i) {
        if (s[i] > 0.0 && t[i] > 0.0) { tmax = std::max(tmax, t[i]); tmin = tmin < 0.0 ? t[i] : std::min(tmin, t[i]); }
    }
    const double gap = tmax > 0.0 ? (tmax - tmin) / tmax : 0.0;
    if (!first && gap < hold) { return gap; }
    double sum = 0.0;
    for (size_t i = 0; i < s.size() && i < target.size(); ++i) {
        s[i] = std::max(0.0, first ? target[i] : s[i] + 0.5 * (target[i] - s[i]));
        sum += s[i];
    }
    if (sum > 0.0) { for (double & v : s) { v /= sum; } }
    return gap;
}

// The experts' shares toward equal expert times: a card's expert time per whole share of the experts is m_i / f_i, so the shares that make those times equal
// are proportional to f_i / m_i (their sum stays what it was). A card without experts (f_i = 0) or without a measurement (m_i = 0) keeps its share.
static inline std::vector<double> dsb_expert_target(const std::vector<double> & f, const std::vector<double> & m) {
    std::vector<double> out(f);
    double sum_w = 0.0, sum_f = 0.0;
    for (size_t i = 0; i < f.size() && i < m.size(); ++i) { if (f[i] > 0.0 && m[i] > 0.0) { sum_w += f[i] / m[i]; sum_f += f[i]; } }
    if (sum_w <= 0.0) { return out; }
    for (size_t i = 0; i < f.size() && i < m.size(); ++i) { if (f[i] > 0.0 && m[i] > 0.0) { out[i] = sum_f * (f[i] / m[i]) / sum_w; } }
    return out;
}

// The planner's model of a split pass over N cards (the dense split alone: x computes the experts of every layer for all tokens). Per layer, the cards'
// dense parts run at once, each on its own tokens; x's expert chain starts when every range's hidden state has arrived; its uploads run ahead of the compute.
struct dsxn_in {
    int64_t T = 0;
    int     L = 0;
    std::vector<double> w, xf, E, kv_b, att_w;   // per layer, as in dsx_in
    std::vector<int>    lcard;                   // per layer: the card (index) that holds its KV cache, -1 when none of the cards does
    std::vector<double> lB;                      // per layer: the upload rate of the card that holds its KV cache (GB/s)
    double w_sum = 0.0, c_base = 0.0, att_tok = 0.0, b_in = 0.0, b_out = 0.0;
    std::vector<double> rel, B;                  // per card: compute against the expert card's, upload rate (GB/s)
    std::vector<std::vector<double>> held;       // per card, per layer: the share of the layer's experts in that card's pool
    int    x = 0;                                // the expert card
    double min_share = 0.05;
};
struct dsxn_out { double t_one = 0.0, t_split = -1.0; std::vector<double> s; };

// a_i and k_i of every card over the whole pass (see dsb_card)
static inline std::vector<dsb_card> dsxn_cards(const dsxn_in & in) {
    const int n = (int) in.rel.size();
    std::vector<dsb_card> c(n);
    for (int i = 0; i < n; ++i) {
        for (int il = 0; il < in.L; ++il) {
            const double D = in.c_base * in.w[il] / in.w_sum * (1.0 - in.xf[il]) + in.att_tok * in.att_w[il];   // us per token of the layer's dense part
            c[i].a += D / std::max(1e-9, in.rel[i]);
            if (in.E[il] > 0.0 && i != in.x) { c[i].a += (in.b_in + in.b_out) / (std::min(in.B[i], in.B[in.x]) * 1e3); }
            if (in.kv_b[il] > 0.0 && in.lcard[il] != i) { c[i].k += in.kv_b[il] / (std::min(in.B[i], in.lB[il]) * 1e3); }
        }
    }
    return c;
}

static inline double dsxn_layer_us(const dsxn_in & in, int il, const std::vector<double> & s) {
    const double T  = (double) in.T;
    const double cb = in.c_base * T * in.w[il] / in.w_sum;
    const double D  = cb * (1.0 - in.xf[il]) + in.att_tok * in.att_w[il] * T, X = cb * in.xf[il];
    const bool   moe = in.E[il] > 0.0;
    double front = 0.0;   // when the expert card holds every range's hidden state (a layer without experts: when the last card is done)
    for (int i = 0; i < (int) s.size(); ++i) {
        if (s[i] <= 0.0) { continue; }
        double t = s[i] * D / std::max(1e-9, in.rel[i]);
        if (in.kv_b[il] > 0.0 && in.lcard[il] != i) { t += in.kv_b[il] / (std::min(in.B[i], in.lB[il]) * 1e3); }
        if (moe && i != in.x) { t += s[i] * T * (in.b_in + in.b_out) / (std::min(in.B[i], in.B[in.x]) * 1e3); }
        front = std::max(front, t);
    }
    if (!moe) { return front; }
    const double up = in.E[il] * (1.0 - in.held[in.x][il]) / (in.B[in.x] * 1e3);
    return std::max(front + X, up);
}

// one card (everything on the expert card) and the split with the shares that make the cards take equally long
static inline dsxn_out dsxn_plan(const dsxn_in & in) {
    dsxn_out o;
    for (int il = 0; il < in.L; ++il) {
        const double cl = (in.c_base * in.w[il] / in.w_sum + in.att_tok * in.att_w[il]) * (double) in.T;
        const double up = in.E[il] * (1.0 - in.held[in.x][il]) / (in.B[in.x] * 1e3);
        o.t_one += std::max(cl, up);
    }
    o.s = dsb_target(dsxn_cards(in), (double) in.T, in.min_share, in.x);
    o.t_split = 0.0;
    for (int il = 0; il < in.L; ++il) { o.t_split += dsxn_layer_us(in, il, o.s); }
    return o;
}
