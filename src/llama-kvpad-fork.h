// How many cells a decode graph reads. Pure functions, tested by tests/test-kvpad-fork.cpp.
//
// A graph reads the cache's cells rounded up to 256 (llama_kv_cache::get_n_kv), so that it can be reused: in decode the count steps once per 256 tokens, and
// each step builds the whole graph again and captures its CUDA graphs again. Qwen3.8-Flash-Next on one RTX 3090 (2026-11-19, perf of the decode thread):
// 62 ms a step with the card idle (kernel launches under capture 18, the scheduler's tables 11, capture host code 16, build + split + allocation 9) against
// 12.9 ms a token: 2% of the decode time. A one-token pass therefore rounds up further, over rows the KV zone already backs and keeps zero.
#pragma once
#include <algorithm>
#include <cstdint>

// The pad of a one-token pass from what a rebuild costs against a token (both measured by the context, microseconds): one rebuild per pad tokens may take half
// a percent of the decode time. 256 (the rule for every other pass) until both are known. Never past 2048: the cells past the ones in use are read and masked,
// by attention that reads every cell at a cost per cell (the measured models: under 0.4 us a cell for a token, a few percent of what the rebuilds cost).
static inline uint32_t llama_kv_decode_pad_fork(double rebuild_us, double token_us) {
    if (!(rebuild_us > 0.0) || !(token_us > 0.0)) { return 256; }
    const double need = rebuild_us / (0.005 * token_us);
    uint32_t p = 256;
    while (p < 2048 && (double) p < need) { p *= 2; }
    return p;
}

// The pad a context goes on with after a measurement. It starts at 1024 (the measured models: a rebuild costs 50 to 100 ms, a token 13 to 55 ms, a
// padded cell under 0.4 us) and only gives the wider reads up when rebuilds turned out cheap (CPU decode: no capture, slow tokens): a quarter of the
// current pad or less. Every other change would build the graph once more for a gain inside the noise.
static inline uint32_t llama_kv_decode_pad_next_fork(uint32_t cur, double rebuild_us, double token_us) {
    const uint32_t m = llama_kv_decode_pad_fork(rebuild_us, token_us);
    return (uint64_t) m * 4 <= cur ? m : cur;
}

// The cells a one-token pass reads. `base`: the count of every other pass (the cells in use rounded up to 256, at most the cache's size). `backed`: the cells
// whose rows are the cache's and finite on every card (the zone's horizon; everything for a cache laid out whole). Never below `base`: when the zone backs
// fewer, the pass that needs them grows it as before.
static inline uint32_t llama_kv_n_kv_decode_fork(uint32_t base, uint32_t size, uint32_t pad, uint64_t backed) {
    if (pad <= 256) { return base; }
    const uint64_t wide = ((uint64_t) base + pad - 1) / pad * pad;
    const uint64_t cap  = backed / 256 * 256;
    return (uint32_t) std::min<uint64_t>(size, std::max<uint64_t>(base, std::min(wide, cap)));
}

// The first cell of a one-token pass's mask (llama_kv_cache::kq_mask_k0: the cells before it are visible to the token). `k0_first`: the multiple of 256 at or
// before the token's cell, the start of every other pass. The mask keeps one start and one width (the pad) for as long as the graph keeps its cells: a start
// that follows the token would move every 256 tokens and the graph would be built again for it.
static inline uint32_t llama_kv_decode_k0_fork(uint32_t k0_first, uint32_t n_kv, uint32_t pad) {
    if (pad <= 256) { return k0_first; }
    return std::min(k0_first, n_kv > pad ? n_kv - pad : 0u);
}
