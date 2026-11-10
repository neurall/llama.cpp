#pragma once
// The prompt planner's strategy estimators. Pure functions: no globals, no locks, no logging, so each one is unit tested alone (tests/test-estimate-fork.cpp).
#include <algorithm>
#include <cmath>
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
    std::vector<pp_miss> miss;
    explicit pp_sums(size_t n_predictions = 0) : Uk(n_predictions, 0.0), Mk(n_predictions, 0.0) {}
};
// One expert: S its bytes, Sm the part only the SSD has, on_card whether its layer's card holds it, fastest whether that card has the fastest link, p1 its share of one token's
// picks; r0, rl, rc the fastest link's, its layer's link's and the CPU's rate in bytes per us (0: not known, the term is left out)
static inline void pp_sums_add(pp_sums & s, double n, double S, double Sm, bool on_card, bool fastest, double p1, double r0, double rl, double rc) {
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
};
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
// Decode chain: the misses go to the CPU and to the uploads side by side (a x b / (a + b)), the GPU work runs in 4-token chunks that read every token's experts again.
static inline double estimate_decodechain_time(const pp_metrics & m) {
    const double both = m.up + m.cpu > 0.0 ? m.up * m.cpu / (m.up + m.cpu) : 0.0;
    return m.dec_f + both * m.dec_r + std::max(m.chunk_us * m.dec_kc, m.comp_us_tok * m.n) + m.ssd_miss;
}
static inline double estimate_decodechain_time_per_token(const pp_metrics & m) { return estimate_decodechain_time(m) / m.n; }
// Predictor: 2^i x the picks uploaded ahead of the routers, what the tokens rank that high; late_factor is what the late fixes and the ranking nodes cost on top (PP_K_LATE).
static inline double estimate_predictor_upload_time(const pp_metrics & m, int i, double late_factor) { return late_factor * std::max(m.Uk[i], pp_ssd_us(m, m.Mk[i])); }
static inline double estimate_predictor_time_per_token(const pp_metrics & m, int i, double late_factor) { return std::max(estimate_predictor_upload_time(m, i, late_factor), m.comp_us_tok * m.n) / m.n; }

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
    pp_sums s(6);
    const size_t n_cards = std::max<size_t>(1, hw.vram_gb.size());
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
            pp_sums_add(s, n, md.expert_bytes, on_card || in_ram ? 0.0 : md.expert_bytes, on_card, card == 0, (double) md.used * w[e] / w_all, r0, rl, rc);
        }
    }
    double ssd_bytes = 0.0;
    pp_sums_misses(s, n, m.up, m.cpu, ssd_bytes);
    m.n = n;
    m.comp_us_tok = hw.comp_us_tok.empty() ? 0.0 : hw.comp_us_tok[0];
    m.ssd_gbs = hw.ssd_gbs;
    m.U = s.U; m.M = s.M; m.Um = s.Um; m.Mm = s.Mm;
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
