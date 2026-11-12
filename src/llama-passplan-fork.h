// The small formulas of the prompt pass planning, pure functions so tests/test-pass-fork.cpp can run them with no GPU and no model; llama-moecache-fork.cpp and
// llama-graph.cpp call these (no second copy).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// A graph's bytes as a + b x tokens from the sizes measured so far (tokens, bytes, all at one cell count). ONE rule for the pass cap (pp_cap_to_pool), find_slot's
// limit (llama_moe_cache_kv_tail_cap_n_fork) and the least lend (kv_prompt_lend0), so they cannot disagree. The least-squares line through every size; a graph's
// fixed part is never negative, so the slope never exceeds the least bytes per token measured: clamped there, a refit. Two passes of nearly one size give a slope of
// noise (GLM 3.0, 6552 and 6554 tokens: 124.8 MB per token where the graphs measured 1.4). Until 2026-11-12 the two limits took the line through the two largest
// sizes and cut a negative fixed part to 0 while keeping the slope: Qwen3.8 --moe staging=0 had run 6656 and 6860 tokens (0.60 MiB per token between them, the graphs
// measure 0.42), a 7064-token pass was sized 4.2 GiB instead of 2.9 and find_slot refused it ("n_tokens = 7064 > size = 0"), the call ran as two passes, 25% slower.
// One size, or no rising line: the most bytes per token (not below the real slope). False: no size.
static inline bool ppx_lend_line(const std::vector<std::pair<int32_t, size_t>> & pts, double & a, double & b) {
    a = b = 0.0;
    if (pts.empty()) { return false; }
    double mx = 0.0, my = 0.0, rmin = 0.0, rmax = 0.0;
    for (const auto & p : pts) {
        const double r = (double) p.second / (double) p.first;
        rmin = rmin > 0.0 ? std::min(rmin, r) : r;
        rmax = std::max(rmax, r);
        mx += (double) p.first; my += (double) p.second;
    }
    mx /= (double) pts.size(); my /= (double) pts.size();
    double sxx = 0.0, sxy = 0.0;
    for (const auto & p : pts) { sxx += ((double) p.first - mx) * ((double) p.first - mx); sxy += ((double) p.first - mx) * ((double) p.second - my); }
    if (sxx > 0.0 && sxy > 0.0) { b = sxy / sxx; a = std::max(0.0, my - b * mx); }
    if (b <= 0.0) { b = rmax; a = 0.0; }
    if (b > rmin) { b = rmin; a = std::max(0.0, my - b * mx); }
    return b > 0.0;
}

// Prefetch staging: the slots (full expert tensors uploaded ahead on a second stream) are lent blocks behind the graph and its scratch. As many as the pool still holds behind
// them (cap_b: what the pool can lend at most, stage_off: where the slots start), up to the configured number; never a cut micro-batch. The scheduler keeps slots - 1 uploads
// in flight (ggml_backend_sched_prefetch_prepare), so one slot overlaps nothing: 2 is the least that hides an upload behind a matmul (double buffering), fewer means none.
// plan_k >= 0 (prompt graphs): the pass planner's limit.
static inline int ppx_stage_slots(size_t slot_b, size_t cap_b, size_t stage_off, int n_cfg, int plan_k) {
    int fit = slot_b > 0 && cap_b > stage_off ? (int) std::min<size_t>((size_t) std::max(0, n_cfg), (cap_b - stage_off) / slot_b) : 0;
    if (plan_k >= 0) { fit = std::min(fit, plan_k); }
    return fit >= 2 ? fit : 0;
}

// The fallback pass size of a prompt of n tokens with micro-batch ub_max when the cache path (max batch mb) takes the tail: full passes of ub_max when the tail fits the cache
// path; otherwise equal passes, so no pass is nearly empty (each re-uploads the experts).
static inline uint32_t ppx_equal_pass(uint32_t n, uint32_t ub_max, uint32_t mb) {
    if (n <= ub_max || ub_max <= mb) { return ub_max; }
    if (n % ub_max <= mb) { return ub_max; }
    const uint32_t passes = (n + ub_max - 1) / ub_max;
    return (n + passes - 1) / passes;
}

// Dense split: the token where the first range (the card with the slower link) ends. share of the tokens, but each range keeps at least min_tokens (a range smaller than
// that costs more in hand-offs than it saves). Both ranges are never empty: a pass of exactly min_tokens tokens gave the first range all of them and the second none.
static inline int64_t ppx_ds_edge(int64_t n_tokens, int64_t min_tokens, double share) {
    if (n_tokens < 2) { return n_tokens; }   // nothing to split: the caller treats edge >= n_tokens as one range
    const int64_t e = std::max<int64_t>(min_tokens, std::min<int64_t>(n_tokens - min_tokens, (int64_t) llround(share * (double) n_tokens)));
    return std::max<int64_t>(1, std::min<int64_t>(e, n_tokens - 1));
}

// ---- --moe staging=0: the stream into the cache's blocks (llama-moecache-fork.cpp cs_*) ----------------------------------------------------------------------------
// The ring at the pool tail holds two entries of the most common size class (the one in the kernels, the next on its way), or one of the largest class when that is
// more, plus two blocks (a grid block across the ring's start is not the ring's). unit bytes: an entry's experts x the class's block; an entry: a layer's experts / chunks.
static inline size_t ppx_ring_reserve(size_t unit_max_b, size_t unit_com_b, size_t bmax) { return std::max(unit_max_b, 2 * unit_com_b) + 2 * bmax; }

// The entries a card computes of one layer beyond the first, with the layer in `chunks` chunks and the card holding `share` of its experts (1: no expert split). A part
// is cut only as far as an entry asks (llama-graph.cpp): half a layer at 2 chunks is one entry.
static inline int ppx_extra_entries(int chunks, double share) { return std::max(1, (int) std::ceil(share * (double) chunks - 1e-9)) - 1; }

// What a prompt of n tokens takes as equal stream passes of at most cap tokens (us): a pass uploads every expert beside its compute (up_us) and computes its tokens
// (comp_us_tok each): the longer of the two. *pass: the pass size.
static inline double ppx_stream_us(uint32_t n, uint32_t cap, double up_us, double comp_us_tok, uint32_t * pass = nullptr) {
    if (n == 0 || cap == 0) { if (pass) { *pass = 0; } return 0.0; }
    const uint32_t k = (n + cap - 1) / cap, p = (n + k - 1) / k;
    if (pass) { *pass = p; }
    return (double) k * std::max(up_us, comp_us_tok * (double) p);
}

// The chunk count of a stream call: candidate i (listed by rising chunk count) leaves passes of at most cap[i] tokens (0: it cannot run) and costs extra[i] entries
// per layer, each one more pass over every token of the pass (entry_us_tok: zeroed outputs, quantized inputs, a sum). Among the candidates within 1% of the lowest
// time: the fewest passes (the upload a pass really takes is what the links' rates promise or more, never less), then the fewest chunks. us[i] (if given): each
// candidate's time, < 0 where it cannot run. -1: none can run.
static inline int ppx_pick_chunks(uint32_t n, const uint32_t * cap, const int * extra, int n_cand, double up_us, double comp_us_tok, double entry_us_tok, double * us = nullptr) {
    double t_min = -1.0;
    for (int i = 0; i < n_cand; ++i) {
        const double t = cap[i] > 0 ? ppx_stream_us(n, cap[i], up_us, comp_us_tok + entry_us_tok * (double) extra[i]) : -1.0;
        if (us) { us[i] = t; }
        if (t >= 0.0 && (t_min < 0.0 || t < t_min)) { t_min = t; }
    }
    int best = -1;
    uint32_t kb = 0;
    for (int i = 0; i < n_cand; ++i) {
        if (cap[i] == 0) { continue; }
        const double   t = ppx_stream_us(n, cap[i], up_us, comp_us_tok + entry_us_tok * (double) extra[i]);
        const uint32_t k = (n + cap[i] - 1) / cap[i];
        if (t <= 1.01 * t_min && (best < 0 || k < kb)) { best = i; kb = k; }
    }
    return best;
}
