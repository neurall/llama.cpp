// The dense split planner's model of one split layer and its search, pure functions of per-layer numbers so tests/test-split-fork.cpp can run them with no GPU and no
// model. llama-dsplit-fork.cpp (dsplit_plan_compute) fills dsx_in from the model and the probe and calls dsx_plan; the reasoning behind each term is in the comments below.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
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
