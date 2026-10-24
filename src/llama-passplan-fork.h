// The small formulas of the prompt pass planning, pure functions so tests/test-pass-fork.cpp can run them with no GPU and no model; llama-moecache-fork.cpp and
// llama-graph.cpp call these (no second copy).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

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
