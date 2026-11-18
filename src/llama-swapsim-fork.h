// The decode cache's swap rules as a model that replays recorded expert picks (user 2026-11-18: SWAP_LEAD took about 1.9k tokens to decide by a live A/B, "thats
// abnormal cant it be faster"; "for sure do d", "way faster and cheaper during run so nobrainer"). A thread count acts within one token, so two counts alternate token
// by token; the pay-back margin changes which experts the cache holds, so a live trial has to wait for the cache to turn over under each value (320-token slices, 96
// dropped, 3 a value). Replayed on the same picks from the same starting cache, two margins differ by nothing but the rule: no noise, no token spent on the worse one.
//
// What is mirrored (src/llama-moecache-fork.cpp, decode, the default policy "add"):
//   score        uses in the last `window` tokens + gweight x lifetime share (of the layer's busiest expert; of the pool's busiest when layers compete for a block)
//   in the token a layer's misses, dearest on the CPU first, go to the CPU or to an upload by the time model T(k) = max(k uploads at the link rate, the rest at the
//                CPU rate): the k that finishes the layer first. An upload takes a free block of the layer's class, else the layer's lowest-scored expert this
//                token does not use, and only when the newcomer scores at least the victim plus the margin (jit 1; jit 2: any miss)
//   between tokens the misses left, hottest first, with at least `admit` uses in the window, take a free block or the class's lowest-scored expert (any layer), again
//                only past the margin, at most `budget` evictions a token over all layers; such an upload serves from the next token
// What is not: prompt passes, the predictor's prefetch, pinned experts, experts read through a mapping, the upload gate. The engine checks the replay of the setting in
// use against its own CPU and upload byte counters before it believes the comparison (swapsim_trusted).
#pragma once

#include "../ggml/src/ggml-hist-fork.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

struct swapsim_knobs {
    int    window  = 64;     // tokens of the recent-use window
    double gweight = 16.0;   // the lifetime share's weight in the score
    double admit   = 2.0;    // uses in the window before a miss may take a block between tokens
    int    budget  = 8;      // evictions between two tokens, all layers together
    int    jit     = 1;      // uploads inside the token: 0 none, 1 admitted like a swap, 2 any miss
    int    policy  = 0;      // the score: 0 "add" (window + lifetime share), 1 "crf" (ClassLRU, upstream #31233: a use count that halves every 128 tokens gives a
                             // class, the score is 2^class / (age + 1); no pay-back margin), 2 "stay" (the three last-10 rings of ggml-hist-fork.h: the reads
                             // a stay is expected to serve, from the expert's last stays, else from its recall rate; the newcomer must expect more than the
                             // victim, no window count and no margin), 3 "rate" (the recall ring alone: uses a token over the span of the last 10 recalls,
                             // measured up to now, so both how often and how lately count; one recall is no rate), 4 "rate+stay" (the same score; a newcomer
                             // whose last stays served fewer reads than an upload costs, `payback`, must beat the victim by that shortfall)
                             // 5 "lru" (the time of the last use), 6 "slru" (upstream's cache from #30637 to #31233: a hit moves an expert to the protected half of
                             // what its layer holds, whose oldest goes back; LRU inside each half); both as the engine's policy=lru|slru, no margin
    double payback = 5.0;    // reads a stay must serve to pay for its upload (the upload's cost over a CPU miss's: CPU GB/s / link GB/s)
    int    spend   = 0;      // the budget between tokens: 0 by layer in turn (the engine), 1 best first: the hottest newcomers of any layer
};

struct swapsim_layer {
    int    n = 0, cls = 0, link = 0;     // experts, block class, upload link
    double bytes = 0;                    // one expert (up + gate + down)
    std::vector<uint8_t>  in, coming;    // cached; queued between tokens (serves from the next token)
    std::vector<uint32_t> win, glob;     // uses in the window, lifetime uses
    uint64_t gmax = 1;                   // the layer's busiest expert's lifetime uses
    std::vector<float>    crf;           // policy crf: the use count, halving every 128 tokens, as of crf_t
    std::vector<uint32_t> crf_t;         // the layer's token of the last use
    uint32_t tok = 0;                    // tokens this layer saw
    std::vector<uint8_t>  prot;          // policy slru: in the protected half
    int n_prot = 0;
    std::vector<hist10>   h;             // last 10 recalls, last 10 evictions, reads served in each of the last 10 stays (kept while out)
    std::deque<std::vector<int32_t>> recent;
};

struct swapsim_out {
    double   cpu_bytes = 0, jit_bytes = 0, bg_bytes = 0;   // computed on the CPU; uploaded inside the token; uploaded between tokens
    double   us = 0;                                       // the layers' miss time by the time model, summed
    uint64_t tokens = 0, picks = 0, hits = 0, evictions = 0, d_margin = 0, d_budget = 0, d_admit = 0;
};

struct swapsim {
    swapsim_knobs              k;
    std::vector<swapsim_layer> L;
    std::vector<int>           free_blocks;      // per class: blocks holding nothing
    std::vector<double>        link_bytes_us;    // per link: bytes a microsecond
    double                     cpu_bytes_us = 0; // the CPU's expert rate, bytes a microsecond (0: nothing is computed on the CPU, every miss is uploaded)
    std::vector<int>           margin;           // per link: the pay-back margin in uses
    size_t                     rr = 0;
    swapsim_out                out;
    uint32_t                   clock = 0;                        // tokens seen, prompt tokens too: the rings' time
    double                     dec_adm = 0, dec_tok = 0, n_blocks = 0;   // admissions and tokens, fading over about one pool of tokens; the pool's blocks

    // tokens a resident stays: the pool's blocks over the admissions a token (as the RAM row tier's rows_turnover)
    double turnover() const { return n_blocks / std::max(dec_tok > 0 ? dec_adm / dec_tok : 1.0, 1e-3); }
    void admitted(swapsim_layer & l, int32_t e) { hist_admit(l.h[e]); dec_adm += 1.0; }
    void evicted(swapsim_layer & l, int32_t e) { l.in[e] = 0; hist_evict(l.h[e], clock); out.evictions++; if (l.prot[e]) { l.prot[e] = 0; l.n_prot--; } }
    bool by_window() const { return k.policy < 2 || k.policy >= 5; }   // a miss needs `admit` uses in the window (the history policies decide by their rings)
    // may a newcomer scoring s take the block of a victim scoring v
    bool beats(const swapsim_layer & l, int32_t e, double s, double v) const {
        if (k.policy == 4) {   // an expert that did not pay for its last stays needs that much more to come back
            const hist10 & h = l.h[e];
            double m = k.payback;
            if (h.n_evict > 0) { double sum = 0; for (int i = 0; i < h.n_evict; ++i) { sum += h.served[i]; } m = sum / h.n_evict; }
            return s > v * std::max(1.0, k.payback / std::max(m, 0.5));
        }
        return k.policy >= 2 ? s > v : s >= v + margin_of(l);
    }
    void tick() {
        if (n_blocks <= 0) {
            for (int f : free_blocks) { n_blocks += f; }
            for (const auto & l : L) { for (int e = 0; e < l.n; ++e) { n_blocks += l.in[e] || l.coming[e]; } }
            n_blocks = std::max(1.0, n_blocks);
        }
        clock++;
        const double d = std::exp(-1.0 / n_blocks);
        dec_adm *= d; dec_tok = dec_tok * d + 1.0;
    }

    void add_layer(int n, int cls, int link, double bytes) {
        swapsim_layer l; l.n = n; l.cls = cls; l.link = link; l.bytes = bytes;
        l.in.assign(n, 0); l.coming.assign(n, 0); l.win.assign(n, 0); l.glob.assign(n, 0); l.crf.assign(n, 0.0f); l.crf_t.assign(n, 0); l.h.assign(n, hist10()); l.prot.assign(n, 0);
        L.push_back(std::move(l));
        if ((int) free_blocks.size() <= cls) { free_blocks.resize(cls + 1, 0); }
    }
    uint64_t gmax_all() const { uint64_t g = 1; for (const auto & l : L) { g = std::max(g, l.gmax); } return g; }
    double score(const swapsim_layer & l, int32_t e, uint64_t gmax) const {
        if (k.policy == 2) { return hist_expect(l.h[e], clock, turnover()); }
        if (k.policy == 5) { return (double) l.crf_t[e]; }
        if (k.policy == 6) { return (l.prot[e] ? 1e15 : 0.0) + (double) l.crf_t[e]; }
        if (k.policy >= 3) { return hist_rate(l.h[e], clock); }
        if (k.policy == 1) {   // as the engine's policy::crf
            const uint32_t age = l.tok - l.crf_t[e];
            return (double) (1u << std::min(7, (int) std::floor(std::log2(std::max(1.0f, l.crf[e] * std::exp2(-(float) age / 128.0f)))))) / (double) (age + 1);
        }
        return l.win[e] + k.gweight * ((double) l.glob[e] / (double) gmax);
    }
    int margin_of(const swapsim_layer & l) const { return k.policy != 0 ? 0 : l.link < (int) margin.size() ? margin[l.link] : 0; }

    // a token's picks of one layer enter its counts: the window, the lifetime uses, the halving count
    void note(swapsim_layer & l, const std::vector<int32_t> & cur) const {
        l.tok++;
        if ((int) l.recent.size() >= k.window) {
            for (int32_t o : l.recent.front()) { l.win[o]--; }
            l.recent.pop_front();
        }
        l.recent.emplace_back();
        for (int32_t e : cur) {
            if (e < 0 || e >= l.n) { continue; }
            l.win[e]++; l.gmax = std::max<uint64_t>(l.gmax, ++l.glob[e]);
            l.crf[e] = l.crf[e] * std::exp2(-(float) (l.tok - l.crf_t[e]) / 128.0f) + 1.0f; l.crf_t[e] = l.tok;
            if (k.policy == 6 && l.in[e] && !l.prot[e]) {   // a hit: to the protected half of what the layer holds, its oldest back to the other
                l.prot[e] = 1; l.n_prot++;
                int held = 0; for (int x = 0; x < l.n; ++x) { held += l.in[x]; }
                if (l.n_prot > std::max(1, held / 2)) {
                    int32_t oldest = -1;
                    for (int32_t x = 0; x < l.n; ++x) { if (l.prot[x] && x != e && (oldest < 0 || l.crf_t[x] < l.crf_t[oldest])) { oldest = x; } }
                    if (oldest >= 0) { l.prot[oldest] = 0; l.n_prot--; }
                }
            }
            hist_recall(l.h[e], clock);
            l.recent.back().push_back(e);
        }
    }
    // a prompt token: counted, nothing swapped or priced (prompt passes have their own paths)
    void observe(const std::vector<std::vector<int32_t>> & picks) {
        tick();
        for (size_t li = 0; li < L.size() && li < picks.size(); ++li) { note(L[li], picks[li]); }
    }

    // one decode token: picks[layer] = the experts its router chose
    void token(const std::vector<std::vector<int32_t>> & picks) {
        out.tokens++;
        tick();
        for (auto & l : L) {   // what was queued between the last two tokens serves now
            for (int e = 0; e < l.n; ++e) { if (l.coming[e]) { l.coming[e] = 0; l.in[e] = 1; } }
        }
        std::vector<std::vector<int32_t>> pending(L.size());
        for (size_t li = 0; li < L.size() && li < picks.size(); ++li) {
            swapsim_layer & l = L[li];
            const std::vector<int32_t> & cur = picks[li];
            note(l, cur);
            std::vector<int32_t> miss;
            for (int32_t e : cur) {
                if (e < 0 || e >= l.n) { continue; }
                out.picks++;
                if (l.in[e]) { out.hits++; } else if (std::find(miss.begin(), miss.end(), e) == miss.end()) { miss.push_back(e); }
            }
            if (miss.empty()) { continue; }
            pending[li] = miss;
            const int    m  = (int) miss.size();
            const double S  = l.bytes, rc = cpu_bytes_us, rl = l.link < (int) link_bytes_us.size() ? link_bytes_us[l.link] : 0.0;
            // one token routes each expert once: every miss costs the CPU the same, the hotter goes up first
            std::sort(miss.begin(), miss.end(), [&](int32_t a, int32_t b) { return score(l, a, l.gmax) > score(l, b, l.gmax); });
            auto T = [&](int kk) { const double tl = rl > 0 ? kk * S / rl : (kk ? 1e300 : 0.0), tc = rc > 0 ? (m - kk) * S / rc : (m - kk ? 1e300 : 0.0); return std::max(tl, tc); };
            int kb = 0;
            if (k.jit > 0 && rl > 0) { for (int kk = 1; kk <= m; ++kk) { if (T(kk) < T(kb)) { kb = kk; } } }
            if (rc <= 0 && rl > 0) { kb = m; }
            int n_up = 0;
            for (int i = 0; i < kb; ++i) {
                const int32_t id = miss[i];
                int32_t victim = -1;
                if (free_blocks[l.cls] > 0) {
                    free_blocks[l.cls]--;
                } else {
                    double best = 1e300;
                    for (int32_t x = 0; x < l.n; ++x) {
                        if (!l.in[x] || std::find(cur.begin(), cur.end(), x) != cur.end()) { continue; }
                        const double c = score(l, x, l.gmax);
                        if (c < best) { best = c; victim = x; }
                    }
                    if (victim < 0) { break; }   // every block of the layer is in use by this token: the rest stay with the CPU
                    if (k.jit < 2 && !beats(l, id, score(l, id, l.gmax), best)) { out.d_margin++; continue; }
                    evicted(l, victim);
                }
                l.in[id] = 1; admitted(l, id); n_up++;
            }
            out.jit_bytes += n_up * S; out.cpu_bytes += (m - n_up) * S;
            out.us += std::max(rl > 0 ? n_up * S / rl : 0.0, rc > 0 ? (m - n_up) * S / rc : 0.0);
        }
        // between tokens: the misses left, hottest first
        const uint64_t ga = gmax_all();
        int budget = k.budget;
        if (L.empty()) { return; }
        rr = (rr + 1) % L.size();
        auto coldest = [&](const swapsim_layer & l, swapsim_layer *& lo, int32_t & victim, double & best) {   // of the class, whichever layer holds it
            lo = nullptr; victim = -1; best = 1e300;
            for (auto & o : L) {
                if (o.cls != l.cls) { continue; }
                for (int32_t x = 0; x < o.n; ++x) {
                    if (!o.in[x]) { continue; }
                    const double c = score(o, x, ga);
                    if (c < best) { best = c; victim = x; lo = &o; }
                }
            }
        };
        if (k.spend == 1) {   // best first: every layer's admitted misses in one order, the hottest takes the next eviction
            std::vector<std::pair<double, std::pair<size_t, int32_t>>> cand;
            for (size_t li = 0; li < L.size(); ++li) {
                for (int32_t id : pending[li]) {
                    if (L[li].in[id] || L[li].coming[id]) { continue; }
                    if (by_window() && L[li].win[id] < k.admit) { out.d_admit++; continue; }
                    cand.push_back({ score(L[li], id, ga), { li, id } });
                }
            }
            std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (const auto & c : cand) {
                swapsim_layer & l = L[c.second.first];
                const int32_t id = c.second.second;
                if (free_blocks[l.cls] > 0) { free_blocks[l.cls]--; l.coming[id] = 1; admitted(l, id); out.bg_bytes += l.bytes; continue; }
                swapsim_layer * lo; int32_t victim; double best;
                coldest(l, lo, victim, best);
                if (victim < 0) { continue; }
                if (!beats(l, id, c.first, best)) { out.d_margin++; continue; }
                if (budget <= 0) { out.d_budget++; continue; }
                budget--;
                evicted(*lo, victim);
                l.coming[id] = 1; admitted(l, id); out.bg_bytes += l.bytes;
            }
            return;
        }
        for (size_t q = 0; q < L.size(); ++q) {
            const size_t li = (rr + q) % L.size();
            swapsim_layer & l = L[li];
            std::vector<int32_t> & p = pending[li];
            if (p.empty()) { continue; }
            std::sort(p.begin(), p.end(), [&](int32_t a, int32_t b) { return score(l, a, l.gmax) > score(l, b, l.gmax); });
            for (size_t i = 0; i < p.size(); ++i) {
                const int32_t id = p[i];
                if (l.in[id] || l.coming[id]) { continue; }
                if (by_window() && l.win[id] < k.admit) { out.d_admit++; continue; }
                if (free_blocks[l.cls] > 0) { free_blocks[l.cls]--; l.coming[id] = 1; admitted(l, id); out.bg_bytes += l.bytes; continue; }
                swapsim_layer * lo; int32_t victim; double best;
                coldest(l, lo, victim, best);
                if (victim < 0) { break; }
                if (!beats(l, id, score(l, id, ga), best)) { out.d_margin += p.size() - i; break; }   // sorted: nothing hotter follows
                if (budget-- <= 0) { out.d_budget += p.size() - i; break; }
                evicted(*lo, victim);
                l.coming[id] = 1; admitted(l, id); out.bg_bytes += l.bytes;
            }
        }
    }
};

// What the three kinds of miss bytes cost a token, in microseconds a byte: fitted on the engine's own decode tokens (token time against the bytes its counters saw:
// computed on the CPU, uploaded inside the token, uploaded between tokens), so the replay prices an arm the way this machine, link and backend really pay. The time
// model alone cannot: by it an upload never loses to the CPU inside its token, yet on CUDA the stricter margin won 6% (GLM, MiMo) and on Vulkan the eager one 20%.
struct swapsim_cost { double cpu = 0, jit = 0, bg = 0, base = 0; bool fitted = false; };

// least squares of t = base + cpu x c + jit x j + bg x b over the tokens (ridge 1e-9 of each column's own scale); a cost never comes out negative. Fitted from 32
// tokens on and only when the tokens differ in what they did (a column that never varies gets 0)
static inline swapsim_cost swapsim_fit(const std::vector<double> & t_us, const std::vector<double> & cpu_b, const std::vector<double> & jit_b, const std::vector<double> & bg_b) {
    swapsim_cost w;
    const size_t n = t_us.size();
    if (n < 32 || cpu_b.size() != n || jit_b.size() != n || bg_b.size() != n) { return w; }
    const std::vector<double> * X[3] = { &cpu_b, &jit_b, &bg_b };
    double m[4] = { 0, 0, 0, 0 }, sd[3] = { 0, 0, 0 };
    for (size_t i = 0; i < n; ++i) { m[3] += t_us[i]; for (int c = 0; c < 3; ++c) { m[c] += (*X[c])[i]; } }
    for (double & x : m) { x /= (double) n; }
    for (size_t i = 0; i < n; ++i) { for (int c = 0; c < 3; ++c) { const double d = (*X[c])[i] - m[c]; sd[c] += d * d; } }
    double A[3][4] = {};   // centred normal equations, columns scaled to unit variance
    bool use[3];
    for (int c = 0; c < 3; ++c) { sd[c] = std::sqrt(sd[c] / (double) n); use[c] = sd[c] > 1e-9 * std::max(1.0, std::fabs(m[c])); }
    for (size_t i = 0; i < n; ++i) {
        double z[3];
        for (int c = 0; c < 3; ++c) { z[c] = use[c] ? ((*X[c])[i] - m[c]) / sd[c] : 0.0; }
        for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) { A[r][c] += z[r] * z[c]; } A[r][3] += z[r] * (t_us[i] - m[3]); }
    }
    for (int r = 0; r < 3; ++r) { A[r][r] += use[r] ? 1e-9 * (double) n : 1.0; }
    for (int c = 0; c < 3; ++c) {   // Gauss-Jordan, 3 unknowns
        int piv = c;
        for (int r = c + 1; r < 3; ++r) { if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) { piv = r; } }
        if (std::fabs(A[piv][c]) < 1e-12) { return w; }
        if (piv != c) { for (int q = 0; q < 4; ++q) { std::swap(A[piv][q], A[c][q]); } }
        for (int r = 0; r < 3; ++r) {
            if (r == c) { continue; }
            const double f = A[r][c] / A[c][c];
            for (int q = c; q < 4; ++q) { A[r][q] -= f * A[c][q]; }
        }
    }
    double b[3];
    for (int c = 0; c < 3; ++c) { b[c] = use[c] ? std::max(0.0, A[c][3] / A[c][c] / sd[c]) : 0.0; }
    w.cpu = b[0]; w.jit = b[1]; w.bg = b[2];
    w.base = m[3] - b[0] * m[0] - b[1] * m[1] - b[2] * m[2];
    w.fitted = true;
    return w;
}

// Two margins on the same picks from the same cache: what each costs a token, and which is cheaper. Priced by the fitted costs; with none, by the time model's layer
// times alone (uploads between tokens free). `settle`: tokens at the start that count for neither (each arm's cache first turns over under its own rule).
struct swapsim_verdict { double us_a = 0, us_b = 0; swapsim_out a, b; bool b_wins = false; double gain = 0; };
static inline swapsim_verdict swapsim_compare(const swapsim & start, const std::vector<std::vector<std::vector<int32_t>>> & tokens, const std::vector<int> & margin_a,
                                              const std::vector<int> & margin_b, const swapsim_cost & w = swapsim_cost(), size_t settle = 0) {
    swapsim_verdict v;
    swapsim sa = start, sb = start;
    sa.margin = margin_a; sb.margin = margin_b;
    sa.out = swapsim_out(); sb.out = swapsim_out();
    swapsim_out a0, b0;
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (t == settle) { a0 = sa.out; b0 = sb.out; }
        sa.token(tokens[t]); sb.token(tokens[t]);
    }
    auto cost = [&](const swapsim_out & o, const swapsim_out & o0) {
        return w.fitted ? w.cpu * (o.cpu_bytes - o0.cpu_bytes) + w.jit * (o.jit_bytes - o0.jit_bytes) + w.bg * (o.bg_bytes - o0.bg_bytes) : o.us - o0.us;
    };
    const double n = (double) std::max<size_t>(1, tokens.size() > settle ? tokens.size() - settle : 0);
    v.a = sa.out; v.b = sb.out;
    v.us_a = cost(sa.out, a0) / n; v.us_b = cost(sb.out, b0) / n;
    v.b_wins = v.us_b < v.us_a;
    v.gain = v.us_a > 0 ? (v.us_a - v.us_b) / v.us_a : 0.0;
    return v;
}

// The replay of the setting in use against the engine's own counters over the same tokens: believed when both the CPU's bytes and the uploaded bytes are within a
// quarter (or both sides hold next to nothing: under 1% of the picked bytes)
static inline bool swapsim_trusted(double sim_cpu, double real_cpu, double sim_up, double real_up, double picked_bytes) {
    auto near = [&](double s, double r) { return std::max(s, r) < 0.01 * picked_bytes || (s >= 0.75 * r && s <= 1.25 * r); };
    return near(sim_cpu, real_cpu) && near(sim_up, real_up);
}
