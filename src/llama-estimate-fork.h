#pragma once
// The prompt planner's strategy estimators. Pure functions: no globals, no locks, no logging, so each one is unit tested alone (tests/test-estimate-fork.cpp).
#include <algorithm>
#include <cmath>
#include <vector>

// ---- strategy estimators (user 2026-11-10: "all our strategy deciders implemented as properly named functions taking all key metrics and returning new estimated time per token,
// so planner can combine more strategies working in parallel") ----
// One function per strategy. In: the measured rates and the byte counts of the call being planned (pp_metrics, filled once in llama_moe_cache_pp_decide_fork). Out: us per prompt
// token, or us per call where the name says time. Nothing here decides or stores: the planner compares them, teaches its models with the upload part, and may run two side by side.
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
// pp_metrics_for builds a pass's metrics from them the way llama_moe_cache_pp_decide_fork builds them from the loaded model, with every expert equally hot: the experts fill
// the cards' expert memory (the fastest link first), then the RAM, the rest stays on the SSD. tests/test-estimate-fork.cpp sweeps machines with it.
// shortcut: uniform heat and one link rate for the uploads; pp_decide's per-expert loop (heat, per-layer links) moves here when the dense split's estimators do.
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
};
static inline pp_metrics pp_metrics_for(const pp_machine & hw, const pp_model & md, double n) {
    pp_metrics m;
    const double total = (double) md.layers * md.experts * md.expert_bytes;
    double left = total, on0 = 0.0, on_cards = 0.0;
    for (size_t i = 0; i < hw.vram_gb.size(); ++i) {
        const double h = std::min(left, hw.vram_gb[i] * 1e9);
        if (i == 0) { on0 = h; }
        on_cards += h; left -= h;
    }
    const double in_ram = std::min(left, hw.ram_gb * 1e9), on_ssd = left - in_ram;
    const double link = hw.link_gbs.empty() ? 0.0 : hw.link_gbs[0] * 1e3;   // bytes per us
    const double touched = 1.0 - std::pow(1.0 - (double) md.used / (double) md.experts, n);   // the share of a layer's experts a pass of n tokens asks for
    m.n = n;
    m.comp_us_tok = hw.comp_us_tok.empty() ? 0.0 : hw.comp_us_tok[0];
    m.ssd_gbs = hw.ssd_gbs;
    if (link > 0.0) {
        m.U  = (total - on0) / link;      m.M  = on_ssd;
        m.Um = (total - on_cards) / link; m.Mm = on_ssd;
        m.up = touched * (total - on_cards) / link;
    }
    m.ssd_miss = pp_ssd_us(m, touched * on_ssd);
    if (hw.cpu_gbs > 0.0) { m.cpu = n * md.used * md.layers * md.expert_bytes * (1.0 - on_cards / total) / (hw.cpu_gbs * 1e3); }
    m.wait_us = hw.router_wait_us_layer * md.layers;
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
