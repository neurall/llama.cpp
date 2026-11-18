#pragma once
// The prompt planner's strategy estimators. Pure functions: no globals, no locks, no logging, so each one is unit tested alone (tests/test-estimate-fork.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---- strategy estimators (user 2026-11-10: "all our strategy deciders implemented as properly named functions taking all key metrics and returning new estimated time per token,
// so planner can combine more strategies working in parallel") ----
// One function per strategy. In: the measured rates and the byte counts of the call being planned (pp_metrics, filled once in llama_moe_cache_pp_decide_fork). Out: us per prompt
// token, or us per call where the name says time. Nothing here decides or stores: the planner compares them, teaches its models with the upload part, and may run two side by side.
// What a model's experts add up to for a pass of n tokens. Filled one expert at a time by pp_sums_add: from the loaded model (llama_moe_cache_pp_decide_fork) and from a
// machine made of numbers (pp_metrics_for), the same code for both.
struct pp_miss { double l1p, us, cu, sm; };   // an expert no card holds: log(1 - its share of one token's picks), its upload over its layer's link (us), the CPU's time for one token routed to it (us), its bytes only the SSD has
struct pp_sums {
    double U = 0.0, M = 0.0;     // every expert the fastest card does not hold: upload over that link (us), bytes only the SSD has
    double Um = 0.0, Mm = 0.0;   // the same for the experts no card holds
    std::vector<double> Uk, Mk;  // the same for a prediction of 2^i x the picks, per i: what the pass's tokens rank that high
    std::vector<double> Ul, Ml, Lb;   // layer parallel, per card: what its layers' card does not hold, over that card's link (us); of it the bytes only the SSD has; all its layers' expert bytes
    std::vector<pp_miss> miss;
    explicit pp_sums(size_t n_predictions = 0, size_t n_cards = 0) : Uk(n_predictions, 0.0), Mk(n_predictions, 0.0), Ul(n_cards, 0.0), Ml(n_cards, 0.0), Lb(n_cards, 0.0) {}
};
// One expert: S its bytes, Sm the part only the SSD has, on_card whether its layer's card holds it, fastest whether that card has the fastest link, p1 its share of one token's
// picks; r0, rl, rc the fastest link's, its layer's link's and the CPU's rate in bytes per us (0: not known, the term is left out); card its layer's card (index of its link).
// rl prices the misses (one expert at a time), rb the layer's link in bulk (layer parallel streams a card's layers like the stream: its probed rate; the per-expert
// average of decode's uploads is 2 against 22 GB/s on Vulkan, and layer parallel was priced at 39.6 s for an 11.07 s pass)
static inline void pp_sums_add(pp_sums & s, double n, double S, double Sm, bool on_card, bool fastest, double p1, double r0, double rl, double rb, double rc, size_t card) {
    if (card < s.Lb.size()) {
        s.Lb[card] += S;
        if (!on_card && rb > 0.0) { s.Ul[card] += S / rb; s.Ml[card] += Sm; }
    }
    if (r0 > 0.0 && !(on_card && fastest)) {   // the fastest link's card holds it: copied there, else it streams
        s.U += S / r0; s.M += Sm;
        for (size_t i = 0; i < s.Uk.size(); ++i) {
            const double hit = 1.0 - std::exp(n * std::log1p(-std::min(1.0, p1 * (double) (1 << i))));
            s.Uk[i] += hit * S / r0; s.Mk[i] += hit * Sm;
        }
    }
    if (r0 > 0.0 && !on_card) { s.Um += S / r0; s.Mm += Sm; }   // the mix: computed on its home card if a card holds it
    if (!on_card && rl > 0.0) { s.miss.push_back({ std::log1p(-std::min(1.0, p1)), S / rl, rc > 0.0 ? S / rc : 0.0, Sm }); }
}
// The misses of a pass of n tokens: uploaded (once per expert the pass touches, us), on the CPU (once per token routed to them, us), and their bytes only the SSD has
static inline void pp_sums_misses(const pp_sums & s, double n, double & up, double & cpu, double & ssd_bytes) {
    up = 0.0; cpu = 0.0; ssd_bytes = 0.0;
    for (const auto & x : s.miss) {
        const double touched = 1.0 - std::exp(n * x.l1p);
        up += touched * x.us;
        cpu -= std::expm1(x.l1p) * n * x.cu;
        ssd_bytes += touched * x.sm;
    }
}

struct pp_metrics {
    double n           = 0.0;   // tokens of one pass
    double comp_us_tok = 0.0;   // one token's compute on the card, all layers, measured (0: no pass measured it yet)
    double ssd_gbs     = 0.0;   // the SSD's rate for expert reads (pp_ssd_rate_gbs)
    double U = 0.0,  M  = 0.0;  // every expert the fastest card does not hold: its upload over that link (us), and the bytes of it only the SSD has
    double Um = 0.0, Mm = 0.0;  // the same for the experts no card holds (the mix computes the others where they are)
    double up       = 0.0;      // the experts this pass is expected to touch and no card holds: uploaded over their layer's link (us)
    double cpu      = 0.0;      // the same misses computed on the CPU, once per token routed to them (us)
    double ssd_miss = 0.0;      // the same misses' bytes only the SSD has, at its rate (us)
    double wait_us  = 0.0;      // what a router pass of this size waited for its ids, measured (0: none yet)
    double chunk_us = 0.0;      // decode chain: the weight reads of its 4-token chunks at this size (us)
    double dec_f = 0.0, dec_r = 1.0, dec_kc = 1.0;   // decode chain, learned: fixed part (us), misses ratio, chunks ratio
    const double * Uk = nullptr, * Mk = nullptr;      // prediction of 2^i x the picks: upload (us) and SSD bytes, per i
    std::vector<double> Ul, Ml, share;                // layer parallel, per card: upload over its own link (us), bytes only the SSD has, its layers' share of the compute
};
// the per-card part of the sums into the metrics (both pp_metrics_for and the planner)
static inline void pp_metrics_cards(pp_metrics & m, const pp_sums & s) {
    double all = 0.0;
    for (const double b : s.Lb) { all += b; }
    m.Ul = s.Ul; m.Ml = s.Ml; m.share.assign(s.Lb.size(), 0.0);
    for (size_t c = 0; c < s.Lb.size(); ++c) { m.share[c] = all > 0.0 ? s.Lb[c] / all : 0.0; }
}
static inline double pp_ssd_us(const pp_metrics & m, double bytes) { return m.ssd_gbs > 0.0 ? bytes / (m.ssd_gbs * 1e3) : 0.0; }
// The misses of one pass on the card (us): the link and the SSD deliver side by side, the slower one is the time.
static inline double estimate_moe_miss_ongpu_time(const pp_metrics & m) { return std::max(m.up, m.ssd_miss); }
// The misses of one pass on the CPU (us): each token routed to a missing expert reads it once.
static inline double estimate_moe_miss_oncpu_time(const pp_metrics & m) { return m.cpu; }
// Full stream: every expert uploaded while the pass computes, no router wait. The upload part, then the pass.
static inline double estimate_fullstream_upload_time(const pp_metrics & m)    { return std::max(m.U, pp_ssd_us(m, m.M)); }
static inline double estimate_fullstream_time_per_token(const pp_metrics & m) { return std::max(estimate_fullstream_upload_time(m), m.comp_us_tok * m.n) / m.n; }
// The mix: as the full stream, less the experts a slower card holds and computes itself.
static inline double estimate_mix_upload_time(const pp_metrics & m)           { return std::max(m.Um, pp_ssd_us(m, m.Mm)); }
static inline double estimate_mix_time_per_token(const pp_metrics & m)        { return std::max(estimate_mix_upload_time(m), m.comp_us_tok * m.n) / m.n; }
// Router path (prompt chain): each layer waits for its router's ids, uploads only the misses among the picks, then computes: the three add up.
static inline double estimate_router_time_per_token(const pp_metrics & m)     { return (estimate_moe_miss_ongpu_time(m) + m.wait_us + m.comp_us_tok * m.n) / m.n; }
// Router or full stream (user 2026-11-10): which of the two the planner strikes out for a call of m.n tokens. A path measured at this size counts with its real time
// (ran_r the router path, ran_s the stream: us a pass, 0 never ran here), the other with its estimate; the stream as soon as it costs the same or less. measured: an
// input the estimates rest on was measured in this process (a pass timed the compute, a router pass its waits). Two estimates from calculated inputs alone strike
// nothing: the calculated compute is a lower bound and an unknown router wait counts 0, so the router always looked faster (Qwen3.8 IQ3_S, 2442 tokens, first
// request, 2026-11-14: compute calculated 78 us a token against about 560 measured, the router 339 ms against the stream's 1334; the chain took 2449 ms, the
// release's stream and mix 1767). Then the planner's older rules pick (R1: no chain past the crossover before it ran).
enum { PP_SWITCH_NONE = 0, PP_SWITCH_ROUTER_OUT = 1, PP_SWITCH_STREAM_OUT = 2 };
static inline int pp_switch_pick(const pp_metrics & m, double ran_r, double ran_s, bool measured) {
    if (ran_r <= 0.0 && ran_s <= 0.0 && !measured) { return PP_SWITCH_NONE; }
    const double t_r = ran_r > 0.0 ? ran_r : estimate_router_time_per_token(m) * m.n;
    const double t_s = ran_s > 0.0 ? ran_s : estimate_fullstream_time_per_token(m) * m.n;
    return t_s <= t_r ? PP_SWITCH_ROUTER_OUT : PP_SWITCH_STREAM_OUT;
}
// Decode chain: the misses go to the CPU and to the uploads side by side (a x b / (a + b)), the GPU work runs in 4-token chunks that read every token's experts again.
static inline double estimate_decodechain_time(const pp_metrics & m) {
    const double both = m.up + m.cpu > 0.0 ? m.up * m.cpu / (m.up + m.cpu) : 0.0;
    return m.dec_f + both * m.dec_r + std::max(m.chunk_us * m.dec_kc, m.comp_us_tok * m.n) + m.ssd_miss;
}
static inline double estimate_decodechain_time_per_token(const pp_metrics & m) { return estimate_decodechain_time(m) / m.n; }
// Predictor: 2^i x the picks uploaded ahead of the routers, what the tokens rank that high; late_factor is what the late fixes and the ranking nodes cost on top (PP_K_LATE).
static inline double estimate_predictor_upload_time(const pp_metrics & m, int i, double late_factor) { return late_factor * std::max(m.Uk[i], pp_ssd_us(m, m.Mk[i])); }
static inline double estimate_predictor_time_per_token(const pp_metrics & m, int i, double late_factor) { return std::max(estimate_predictor_upload_time(m, i, late_factor), m.comp_us_tok * m.n) / m.n; }
// Layer parallel (the old staging=0 layout, --moe moe-split=home): each card computes its own layers' experts, the ones it holds where they sit, the others over its own link.
// The cards take turns, so a pass is each card's turn in a row, a turn as long as its uploads (link or SSD) or its compute, whichever is longer.
// shortcut: no head start counted (a card's ring fills its first layers during the other card's turn), so this is the upper bound; count the ring's depth if picks show it too high.
static inline double estimate_layerparallel_upload_time(const pp_metrics & m) {
    double t = 0.0;
    for (size_t c = 0; c < m.Ul.size(); ++c) { t += std::max(m.Ul[c], pp_ssd_us(m, c < m.Ml.size() ? m.Ml[c] : 0.0)); }
    return t;
}
static inline double estimate_layerparallel_time(const pp_metrics & m) {
    double t = 0.0;
    for (size_t c = 0; c < m.Ul.size(); ++c) {
        t += std::max({ m.Ul[c], pp_ssd_us(m, c < m.Ml.size() ? m.Ml[c] : 0.0), m.comp_us_tok * m.n * (c < m.share.size() ? m.share[c] : 0.0) });
    }
    return t;
}
static inline double estimate_layerparallel_time_per_token(const pp_metrics & m) { return estimate_layerparallel_time(m) / m.n; }

// A pass the pool refused (its graph and the KV did not fit a card's pool). Kimi K3.1, 4588 tokens, 2026-11-15: layer parallel (never measured, picked on its prior,
// 45.7 s against the one card's 46.6) wanted a 5329.6 MiB graph on CUDA1 beside the KV in a 6206.6 MiB pool; the refusal became an out-of-memory ceiling for
// every path, so the call and the next ran 2304 + 2436-token passes at 31-33 t/s where one card had run 3936-4190-token passes at 57-63 t/s.
// A path with another path to fall back on (layer parallel, the mix, the k arms: the stream on one card takes their call) keeps the ceiling as its own, and the
// call runs again at the same pass size on the others; a refusal of the last resort halves the passes of every path, as before.
struct pp_refusal { uint64_t tc; bool path_only; uint32_t retry_pass; };
static inline pp_refusal pp_refusal_of(bool has_fallback, uint32_t n_tokens, uint64_t cells, double share, uint32_t smaller_pass) {
    const uint64_t tc = (uint64_t) ((double) n_tokens * (double) std::max<uint64_t>(cells, 1) * std::min(share, 0.9));   // tokens x cells this path may ask for
    return has_fallback ? pp_refusal{ tc, true, n_tokens } : pp_refusal{ tc, false, smaller_pass };
}
// a path's ceiling (0: none) admits a pass of n tokens that ends at `cells`
static inline bool pp_pass_under(uint64_t tc, double n, double cells) { return tc == 0 || n * std::max(1.0, cells) <= (double) tc; }
// The record of graph sizes every path's pass cap reads (llama-moecache-fork.cpp need_tok: per pass size the most a graph of that size borrowed on a card) took the
// refused graph's size before the pool refused it. A path-only refusal takes it back out: that graph never ran and its layout is the refused path's (layer parallel
// computes layers 47-92's experts on CUDA1: 5329.6 MiB for 4588 tokens, the stream's graph there 0.953 MiB a token). Kept, it was CUDA1's only real size, the start-up
// sizes left the fit, and the stream's cap read 1.16 MiB a token: 4512 tokens, the call ran as 2304 + 2284 in 151.4 s where its plan before the refusal was one pass
// (Kimi K3.1, 2026-11-15). The last resort's refusal keeps it: its next pass is that same layout, smaller.
template <class M> static inline void pp_refusal_drop(const pp_refusal & r, uint32_t n_tokens, M & sizes) { if (r.path_only) { sizes.erase((int32_t) n_tokens); } }

// ---- a machine and a model as numbers (user 2026-11-10: tune on unit tests without a model: simulated SSD, memory and PCIe rates, card count, card memory, compute speed) ----
// pp_metrics_for builds a pass's metrics from them through the same sums llama_moe_cache_pp_decide_fork fills from the loaded model (pp_sums_add): the layers go to the cards,
// each card holds the hottest experts of its layers, the RAM the hottest of the rest, what is left stays on the SSD. tests/test-estimate-fork.cpp sweeps machines with it.
struct pp_machine {
    double ssd_gbs = 0.0;                 // SSD read rate
    double cpu_gbs = 0.0;                 // the CPU's read rate over expert weights
    double ram_gb  = 0.0;                 // RAM that holds expert slices (the pinned pool)
    std::vector<double> link_gbs;         // per card, the fastest first: PCIe upload rate
    std::vector<double> vram_gb;          // per card: memory left for experts
    std::vector<double> comp_us_tok;      // per card: one token's compute, all layers
    double router_wait_us_layer = 0.0;    // what one layer's router ids cost a router pass
};
struct pp_model {
    int    layers = 0, experts = 0, used = 0;   // experts per layer, picked per token
    double expert_bytes = 0.0;
    double heat_skew = 0.0;                     // how unevenly the picks fall on a layer's experts: expert of rank r gets 1 / (r + 1)^skew of them (0: all equal)
};
static inline pp_metrics pp_metrics_for(const pp_machine & hw, const pp_model & md, double n, pp_sums * sums_out = nullptr) {
    pp_metrics m;
    const size_t n_cards = std::max<size_t>(1, hw.vram_gb.size());
    pp_sums s(6, n_cards);
    const double layer_bytes = (double) md.experts * md.expert_bytes, total = (double) md.layers * layer_bytes;
    double vram_all = 0.0, held = 0.0;
    for (const double v : hw.vram_gb) { vram_all += v * 1e9; held += std::min(v * 1e9, total / (double) n_cards); }
    const double ram_share = total > held ? std::min(1.0, hw.ram_gb * 1e9 / (total - held)) : 1.0;   // of what no card holds, the hottest part the RAM takes
    std::vector<double> w((size_t) md.experts);
    double w_all = 0.0;
    for (int e = 0; e < md.experts; ++e) { w[e] = std::pow((double) e + 1.0, -md.heat_skew); w_all += w[e]; }
    const double r0 = hw.link_gbs.empty() ? 0.0 : hw.link_gbs[0] * 1e3, rc = hw.cpu_gbs * 1e3;
    for (int l = 0; l < md.layers; ++l) {
        // the layers go to the cards by their expert memory, the fastest link's card first; each card holds the hottest experts of its layers
        size_t card = 0;
        for (double acc = 0.0; card + 1 < n_cards; ++card) { acc += hw.vram_gb[card] * 1e9; if ((double) l / md.layers < acc / std::max(1.0, vram_all)) { break; } }
        const double layers_here = std::max(1.0, md.layers * (vram_all > 0.0 ? hw.vram_gb[card] * 1e9 / vram_all : 1.0));
        const double card_share  = hw.vram_gb.empty() ? 0.0 : std::min(1.0, hw.vram_gb[card] * 1e9 / (layers_here * layer_bytes));
        const int n_on_card = (int) std::floor(card_share * md.experts + 0.5);
        const int n_in_ram  = (int) std::floor(ram_share * (md.experts - n_on_card) + 0.5);
        const double rl = card < hw.link_gbs.size() ? hw.link_gbs[card] * 1e3 : r0;
        for (int e = 0; e < md.experts; ++e) {
            const bool on_card = e < n_on_card, in_ram = !on_card && e < n_on_card + n_in_ram;
            pp_sums_add(s, n, md.expert_bytes, on_card || in_ram ? 0.0 : md.expert_bytes, on_card, card == 0, (double) md.used * w[e] / w_all, r0, rl, rl, rc, card);
        }
    }
    double ssd_bytes = 0.0;
    pp_sums_misses(s, n, m.up, m.cpu, ssd_bytes);
    m.n = n;
    m.comp_us_tok = hw.comp_us_tok.empty() ? 0.0 : hw.comp_us_tok[0];
    m.ssd_gbs = hw.ssd_gbs;
    m.U = s.U; m.M = s.M; m.Um = s.Um; m.Mm = s.Mm;
    pp_metrics_cards(m, s);
    m.ssd_miss = pp_ssd_us(m, ssd_bytes);
    m.wait_us = hw.router_wait_us_layer * md.layers;
    if (sums_out) { *sums_out = s; m.Uk = sums_out->Uk.data(); m.Mk = sums_out->Mk.data(); }
    return m;
}
// The pass size from which the full stream costs no more per token than the router path: the crossover the planner switches at. 0: the stream from the first token, limit + 1: never.
static inline double pp_crossover_tokens(const pp_machine & hw, const pp_model & md, double limit) {
    for (double n = 1.0; n <= limit; n = std::floor(n * 1.25) + 1.0) {
        const pp_metrics m = pp_metrics_for(hw, md, n);
        if (estimate_fullstream_time_per_token(m) <= estimate_router_time_per_token(m)) { return n == 1.0 ? 0.0 : n; }
    }
    return limit + 1.0;
}

// ---- paired A/B of one multiplier on token times (user 2026-11-15: "or do abab every 100 tokens") ----
// The CPU/GPU split of a layer's misses (llama_moe_cache_jit_fork) weighs the CPU side by this multiplier: 2 counts the CPU twice as dear (more uploads), 0.5 half (more on the
// CPU). The token times decide it, as the decode thread search does: tokens run base and candidate in turn, b c b c b, every c against the mean of its two b neighbours (a
// warming cache cancels); the candidate wins only when it was faster in every one of 3 pairs by the batch thread search's noise-scaled bar (0.5% to 3% from the b tokens'
// spread: with the decode search's looser rule, a mixed result then more wins than losses, +-5% noise alone walked the multiplier 1 -> 8 in 2000 tokens). A comparison in
// which the candidate split no layer differently is a loss. As in the batch thread search, two won comparisons in a row replace the base (one alone: flat +-5% noise moved
// the multiplier in 79 of 200 runs of 1000 tokens). Candidates base*2 first, then base/2; a won step goes on in its direction until it loses or reaches a bound.
// Settled, the search runs again after `every` tokens; the first one starts at once.
// shortcut: steps of x2 only, bisect between two steps when the measured optimum sits between them
struct ab_tune {
    double base = 1.0, cand = 1.0, lo = 1.0 / 16, hi = 16.0;
    int every = 100, since = 100;   // tokens between two searches; since the last one
    int st = 0, dir = 0;            // 0 settled, 1 sampling; the direction tried (+1 base*2, -1 base/2)
    bool won = false, pending = false;   // a step of this search won: no turn to the other direction; the candidate won once, the next comparison confirms it
    int pairs = 3, k = 0, diff = 0; // pairs sampled, tokens into the sequence, tokens on which the two values split some layer differently
    std::vector<double> sb, sc;
    double last_b = 0.0, last_c = 0.0, last_base = 1.0, last_cand = 0.0; int last_wins = 0, last_pairs = 0, last_diff = 0;   // the last comparison, for the caller's log
    double now() const { return st == 1 && (k % 2) == 1 ? cand : base; }
    double other() const { return st == 1 ? ((k % 2) == 1 ? base : cand) : base; }   // the value the token is compared against
    bool plan(int d) {
        const double c = base * (d > 0 ? 2.0 : 0.5);
        if (c > hi * (1.0 + 1e-9) || c < lo * (1.0 - 1e-9)) { return false; }
        cand = c; dir = d; k = 0; pairs = 3; diff = 0; sb.clear(); sc.clear(); st = 1;
        return true;
    }
    void settle() { st = 0; since = 0; }
    void abort() { if (st == 1) { st = 0; since = every; } }   // the samples are void (the thread count changed under them): the search starts again
    // t: the time of the token that ran at now(); differs: on that token some layer's split under other() would have differed. True when a comparison finished (last_* set).
    bool feed(double t, bool differs) {
        if (st == 0) {
            if (++since < every) { return false; }
            won = false;
            if (!plan(+1) && !plan(-1)) { since = 0; }
            return false;   // this token ran before the search: not a sample
        }
        (k % 2 == 0 ? sb : sc).push_back(t);
        diff += differs;
        if (k < 2 * pairs) { ++k; return false; }
        const double bmin = *std::min_element(sb.begin(), sb.end()), bmax = *std::max_element(sb.begin(), sb.end());
        const double bar = std::min(0.03, std::max(0.005, 0.5 * (bmax / std::max(bmin, 1e-9) - 1.0)));
        int wins = 0;
        for (int i = 0; i < pairs; ++i) { wins += sc[i] < (1.0 - bar) * 0.5 * (sb[i] + sb[i + 1]); }
        double mb = 0.0, mc = 0.0;
        for (double x : sb) { mb += x; }
        for (double x : sc) { mc += x; }
        last_b = mb / (double) sb.size(); last_c = mc / (double) sc.size(); last_base = base; last_cand = cand; last_wins = wins; last_pairs = pairs; last_diff = diff;
        if (diff > 0 && wins == pairs && !pending) {
            pending = true; plan(dir);   // the same candidate again
        } else if (diff > 0 && wins == pairs) {
            base = cand; won = true; pending = false;
            if (!plan(dir)) { settle(); }
        } else {
            pending = false;
            if (!(!won && dir > 0 && plan(-1))) { settle(); }
        }
        return true;
    }
};

// ---- decode thread search (user 2026-10-26: alternate the count in use with one lower candidate, token by token) ----
// 2026-11-18 (user: "we need quick in max 12t solution", answers of 64-100 tokens): the search starts at the computed count, not on every hardware thread: with GPUs the
// true cores (decode reads expert weights and SMT siblings add no bandwidth: on PC1's 3975WX 32 threads ran within 1% of the best on CUDA, 64 22-30% slower), CPU only
// all threads (common_thread_start_fork), or the count an earlier server stored. Tokens run b c b c ...; the only candidate is b's half (64 -> 32 -> 16). An answer
// samples 2 warm-up tokens and at most per_request pairs (12 tokens); an undecided comparison goes on with its samples in the next answer, up to max_pairs. A clear win
// (the half z standard errors faster) adopts it, and the next answer compares it with its own half (won); a clear loss, or max_pairs undecided, keeps b and ends the walk.
// The sign of an undecided comparison never picks (MiMo 2026-11-18: 64 against 32 undecided at +0.88 +- 0.51 ms over 64 pairs kept 64 by its sign, then 48 took over).
// A comparison fits token time = a + arm + CPU MiB + upload MiB + trend by least squares and decides on the arm's coefficient (the candidate's ms per token over b's at
// the same work): most of a token's scatter is its own work, the bytes of experts it computes on the CPU and the bytes it uploads explain 84-94% of the variance of
// 10-token windows (2026-11-15 TOK lines; before that, 2 pairs decided by majority settled DeepSeek on 63 threads where 30 ran 8% faster).
// Least squares of y on the columns of X (row-major, n rows, p columns), a tiny ridge on columns from 2 on (a covariate without variance gets 0). Returns false when singular;
// beta: the coefficients, se1: the standard error of coefficient 1.
static inline bool small_ols(const std::vector<double> & X, const std::vector<double> & y, int n, int p, std::vector<double> & beta, double & se1) {
    std::vector<double> A((size_t) p * p, 0.0), r((size_t) p, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) { r[j] += X[(size_t) i * p + j] * y[i]; for (int l = 0; l < p; ++l) { A[(size_t) j * p + l] += X[(size_t) i * p + j] * X[(size_t) i * p + l]; } }
    }
    for (int j = 2; j < p; ++j) { A[(size_t) j * p + j] += 1e-9 * (double) n + 1e-12; }
    // solve A [beta | e1] by Gauss-Jordan with partial pivoting
    std::vector<double> M((size_t) p * (p + 2));
    for (int j = 0; j < p; ++j) { for (int l = 0; l < p; ++l) { M[(size_t) j * (p + 2) + l] = A[(size_t) j * p + l]; } M[(size_t) j * (p + 2) + p] = r[j]; M[(size_t) j * (p + 2) + p + 1] = j == 1 ? 1.0 : 0.0; }
    for (int c = 0; c < p; ++c) {
        int piv = c;
        for (int j = c + 1; j < p; ++j) { if (std::fabs(M[(size_t) j * (p + 2) + c]) > std::fabs(M[(size_t) piv * (p + 2) + c])) { piv = j; } }
        if (std::fabs(M[(size_t) piv * (p + 2) + c]) < 1e-300) { return false; }
        if (piv != c) { for (int l = 0; l < p + 2; ++l) { std::swap(M[(size_t) c * (p + 2) + l], M[(size_t) piv * (p + 2) + l]); } }
        const double d = M[(size_t) c * (p + 2) + c];
        for (int l = 0; l < p + 2; ++l) { M[(size_t) c * (p + 2) + l] /= d; }
        for (int j = 0; j < p; ++j) {
            if (j == c) { continue; }
            const double f = M[(size_t) j * (p + 2) + c];
            if (f != 0.0) { for (int l = 0; l < p + 2; ++l) { M[(size_t) j * (p + 2) + l] -= f * M[(size_t) c * (p + 2) + l]; } }
        }
    }
    beta.assign((size_t) p, 0.0);
    for (int j = 0; j < p; ++j) { beta[j] = M[(size_t) j * (p + 2) + p]; }
    double rss = 0.0;
    for (int i = 0; i < n; ++i) { double f = 0.0; for (int j = 0; j < p; ++j) { f += X[(size_t) i * p + j] * beta[j]; } rss += (y[i] - f) * (y[i] - f); }
    se1 = n > p ? std::sqrt(std::max(0.0, rss / (n - p) * M[(size_t) 1 * (p + 2) + p + 1])) : 0.0;
    return true;
}
struct thread_search {
    int st = 0, b = 0, c = 0, k = 0, warm = 0, left = 0;   // st 0 idle, 1 comparing, 3 settled; k: this comparison's samples (odd: the next runs on c); left: pairs this answer may still sample
    int lead = 0;                     // the count the tokens between samples run on: the faster one by the comparison so far (no decision: the samples go on)
    int min_pairs = 5, max_pairs = 64, per_request = 5; double z = 2.5;
    bool won = false;                 // the last comparison adopted the half: the next answer compares it with its own half
    std::vector<double> tt, tx, tu;   // this comparison's tokens: time, CPU MiB, upload MiB
    double last_b = 0.0, last_c = 0.0, last_d = 0.0, last_se = 0.0; int last_nb = 0, last_nc = 0, last_pairs = 0;   // the last comparison, for the caller's log
    bool sampling() const { return st == 1 && left > 0; }
    int now() const { return st != 1 ? b : !sampling() || warm > 0 ? lead : (k % 2) == 1 ? c : b; }
    bool settled() const { return st == 3; }
    // compare n with its half (nothing above n can run: the pool's size); 1 has no half
    void start(int n) { b = lead = n; c = n / 2; won = false; k = 0; tt.clear(); tx.clear(); tu.clear(); st = c >= 1 ? 1 : 3; resume(); }
    void settle(int n) { b = lead = n; won = false; st = 3; }   // a count an earlier server settled on: no comparison
    // an answer's first token: 2 warm-up tokens (the cache refills after the prompt), then at most per_request pairs; a pair the last answer left half done is dropped
    void resume() {
        if (st != 1) { return; }
        if (k % 2 == 1) { tt.pop_back(); tx.pop_back(); tu.pop_back(); --k; }
        warm = 2; left = per_request;
    }
    // the candidate's ms per token over b's at the same CPU and upload work (and trend), with its standard error
    void effect(double & d, double & se) const {
        const int n = (int) tt.size(), p = 5;
        double mx = 0.0, mu = 0.0;
        for (int i = 0; i < n; ++i) { mx += tx[i]; mu += tu[i]; }
        mx /= n; mu /= n;
        std::vector<double> X((size_t) n * p), beta;
        for (int i = 0; i < n; ++i) {
            double * row = &X[(size_t) i * p];
            row[0] = 1.0; row[1] = i % 2 == 1 ? 1.0 : 0.0; row[2] = tx[i] - mx; row[3] = tu[i] - mu; row[4] = i - 0.5 * (n - 1);
        }
        d = 0.0; se = 1e30;
        if (small_ols(X, tt, n, p, beta, se)) { d = beta[1]; }
    }
    // t: the time of the token that ran on now(); x, u: its CPU and upload MiB. True when the comparison finished (last_* set, won says how)
    bool feed(double t, double x = 0.0, double u = 0.0) {
        if (!sampling()) { return false; }
        if (warm > 0) { --warm; return false; }
        tt.push_back(t); tx.push_back(x); tu.push_back(u);
        if (k++ % 2 == 0) { return false; }   // the pair's token on b
        --left;
        const int n = k / 2;
        if (n < min_pairs) { return false; }
        double d = 0.0, se = 0.0;
        effect(d, se);
        const bool decided = std::fabs(d) > z * se;
        lead = d < -se ? c : b;   // one standard error: the sign alone ran the noisier half too often (no byte counters, flat tops)
        if (!decided && n < max_pairs) { return false; }
        double mb = 0.0, mc = 0.0;
        for (size_t i = 0; i < tt.size(); ++i) { (i % 2 == 1 ? mc : mb) += tt[i]; }
        last_b = mb / n; last_c = mc / n; last_d = d; last_se = se; last_nb = b; last_nc = c; last_pairs = n;
        won = decided && d < 0.0;
        if (won) { b = c; }
        st = 3; left = 0;
        return true;
    }
};

// The watch beside a settled thread search: windows of 64 decode tokens; the box changed (other load) when a window runs 10% above the best window at comparable work, twice
// in a row. Work = CPU + upload MiB a token (the expert cache's byte counters); a window with over 25% more work than the best is no evidence either way (2026-11-15: MiMo's
// uploads come and go in ~320-token phases at +12% token time and restarted the search from 64 threads once a cell). reset() at a new request: its context and expert mix
// make its own baseline. shortcut: a load change between two requests is not seen; upgrade to a best per context band if that matters (tests/test-estimate-fork.cpp).
// Tokens while the expert cache refills after a prompt (refilling: its background uploads queued or in flight) are no sample and restart the baseline: GLM two-card
// Vulkan 2026-11-17, every slow run 2 restarted the search from 64 threads during the refill ("64 tokens at 90.58 ms/token against 73.83 settled", 11-14 t/s against 17-18).
struct drift_watch {
    int n = 64, need = 2; double slow = 1.10, comparable = 0.25;
    int wn = 0, drift = 0; double wsum = 0.0, wwork = 0.0, best = 0.0, best_work = 0.0;
    double last_m = 0.0, last_best = 0.0;   // the window and the best when it last fired (for the log line)
    void reset() { wn = 0; drift = 0; wsum = 0.0; wwork = 0.0; best = 0.0; best_work = 0.0; }
    bool feed(double t, double work, bool refilling = false) {   // one token's time and work; true: search again
        if (refilling) { reset(); return false; }   // paid whatever the box: no sample, the baseline starts after the refill
        wsum += t; wwork += work;
        if (++wn < n) { return false; }
        const double m = wsum / n, w = wwork / n;
        wn = 0; wsum = 0.0; wwork = 0.0;
        if (best <= 0.0) { best = m; best_work = w; drift = 0; return false; }
        if (w > best_work * (1.0 + comparable)) { return false; }   // more work: slower for that alone
        if (m < best) { best = 0.7 * best + 0.3 * m; best_work = 0.7 * best_work + 0.3 * w; drift = 0; return false; }
        drift = m > slow * best ? drift + 1 : 0;
        if (drift >= need) { last_m = m; last_best = best; reset(); return true; }
        return false;
    }
};

// The self-tuner's knob order (src/llama-moecache-fork.cpp self_tune): it tries one knob at a time, so the transient knobs (36-token slices, decided in
// 72-216 decode tokens) go before the state-carrying ones (320-token slices, 640-1920 tokens), each group in its listed order. GLM two-card Vulkan
// 2026-11-18: with SWAP_LEAD first, UPLOAD_WAIT (gate 0 53.6-55.0 against gate 3 63.8-71.3 ms/token) was never tried in 3 of 6 cells over two 1k
// answers, and their decode stayed at 11-14 t/s against 18 (tests/test-estimate-fork.cpp).
template <class K> void tune_order(std::vector<K> & knobs) {
    std::stable_partition(knobs.begin(), knobs.end(), [](const K & k) { return !k.slow; });
}

// The self-tuner's verdict on one knob from its slices (mean ms per token of each slice, one list per value; inc: the value in use when the knob's turn
// began). An alternative 6% slower than the incumbent after one slice each is dropped at once (exploring costs tokens on the worse setting). From min_s
// slices each, the incumbent keeps the knob unless an alternative beats it by more than 2 standard errors of both AND by at least 1%. The verdict is final
// when sure or at max_s slices each; unsure, the value in use stays.
struct tune_verdict { bool done = false, sure = false; size_t pick = 0, n_min = 0; std::vector<double> mean; };
static inline tune_verdict tune_decide(const std::vector<std::vector<double>> & s, size_t inc, size_t min_s, size_t max_s) {
    tune_verdict r;
    r.pick = inc;
    const size_t nv = s.size();
    auto avg = [](const std::vector<double> & v) { double m = 0; for (double x : v) { m += x; } return m / (double) v.size(); };
    size_t n_min = SIZE_MAX;
    for (const auto & v : s) { n_min = std::min(n_min, v.size()); }
    if (n_min >= 1 && n_min < min_s) {
        const double mi = avg(s[inc]);
        bool worse = true;
        for (size_t c = 0; c < nv; ++c) {
            if (c != inc && !(avg(s[c]) > 1.06 * mi)) { worse = false; }
        }
        if (worse) { n_min = max_s; }   // falls into the verdict below with the incumbent
    }
    r.n_min = n_min;
    if (n_min < min_s) { return r; }
    std::vector<double> se(nv);
    r.mean.assign(nv, 0.0);
    for (size_t c = 0; c < nv; ++c) {
        const auto & v = s[c];
        const double m = avg(v);
        double q = 0;
        for (double x : v) { q += (x - m) * (x - m); }
        r.mean[c] = m; se[c] = v.size() > 1 ? std::sqrt(q / (v.size() - 1) / v.size()) : 1e300;
    }
    size_t b = inc;
    for (size_t c = 0; c < nv; ++c) { if (c != inc && r.mean[c] < r.mean[b]) { b = c; } }
    bool sure = true;
    for (size_t c = 0; c < nv; ++c) {
        if (c != b && r.mean[c] - 2 * se[c] <= r.mean[b] + 2 * se[b]) { sure = false; }
    }
    if (b != inc && (!sure || r.mean[b] > 0.99 * r.mean[inc])) { b = inc; }
    r.pick = b; r.sure = sure; r.done = sure || n_min >= max_s;
    return r;
}

// The two decode tuners take turns. The thread search (the decode thread count, llama_context::thread_tune_feed) and the self-tuner (the expert cache's
// knobs, one at a time, src/llama-moecache-fork.cpp self_tune) both read token times, and each one's changes read as the other's signal: a self-tune slice
// counts a token only while the thread search is not sampling, and the drift watch does not restart the thread search while a self-tune cycle is open (its
// arms change token times by design). MiMo CUDA 12k/4k 2026-11-18 (b15807 37.9/38.5 t/s, upload_wait_tuned_first 34.5/33.6): UPLOAD_WAIT's 36-token slices
// ran inside the first thread search, then a drift restart (28.2 against 25.3 ms/token) ran inside SWAP_LEAD's slices; SWAP_LEAD -2, a clear 10% win on
// b15807, ended unsure after 8 slices each and stayed 0. The CPU/GPU split's A/B already runs on settled thread counts only (llama_moe_cache_split_feed_fork).
static inline bool tune_slice_counts(bool cycle_open, bool prefill, bool threads_quiet) { return cycle_open && !prefill && threads_quiet; }
static inline bool tune_drift_held(bool refilling, bool cycle_open) { return refilling || cycle_open; }

// The refill after a prompt is a phase of its own (user 2026-11-18). The DDR gate (UPLOAD_WAIT: 3 the budget with its chunks, 2 or 1 the CPU phase; the upload
// worker in src/llama-moecache-fork.cpp) holds an upload while the CPU computes uncached experts, so a swap does not slow that phase. A refill upload pays a
// debt instead: each expert it puts back removes a CPU miss from every later token, so it goes at the link's rate, ungated; steady swaps keep the gate.
static inline int upload_gate_for(int gate, bool refill) { return refill ? 0 : gate; }

// The self-tuner's stored verdicts for one knob (user 2026-11-18: resume where the last server left off): its last `keep` clear verdicts, comma-separated, newest
// last. A server starts the knob at the most frequent of them, the newest on a tie; an unsure verdict is not kept (the value in use stays).
static inline void tune_history_push(std::string & h, double v, size_t keep = 8) {
    char b[32];
    snprintf(b, sizeof b, "%g", v);
    h += (h.empty() ? "" : ",") + std::string(b);
    while ((size_t) std::count(h.begin(), h.end(), ',') >= keep) { h.erase(0, h.find(',') + 1); }
}
static inline bool tune_history_pick(const std::string & h, double & v) {
    std::vector<double> xs;
    for (size_t a = 0; a <= h.size();) {
        size_t e = h.find(',', a);
        if (e == std::string::npos) { e = h.size(); }
        const std::string tok = h.substr(a, e - a);
        char * end = nullptr;
        const double x = strtod(tok.c_str(), &end);
        if (tok.empty() || end != tok.c_str() + tok.size()) { return false; }   // a damaged entry: no pick
        xs.push_back(x);
        a = e + 1;
    }
    int best = -1;
    for (size_t i = xs.size(); i-- > 0;) {   // newest first: a tie keeps the newest
        const int n = (int) std::count(xs.begin(), xs.end(), xs[i]);
        if (n > best) { best = n; v = xs[i]; }
    }
    return true;
}
