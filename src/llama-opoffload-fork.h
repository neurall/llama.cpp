#pragma once
// fork: op offload off for a context whose graph's fixed part does not fit (llama_context::sched_reserve). Op offload copies the host expert weights into the
// compute buffer, a part that does not shrink with the micro-batch: two measures (n_tokens and n_tokens / 2) give that fixed part and the per-token slope. When
// fixed + the margins + a 512-token pass do not fit the card's free VRAM, the context runs without op offload, and its pass then takes at most half of what the
// margins leave (the KV gets the other half). RTX 4060 8 GB, Inkling 2 Mini 12k (W10, 2026-11-26): graph 10147 MiB at 8192 tokens, 8356 at 4096 = 6566 MiB fixed,
// free 5950 < 8737 MiB: HTTP 500 before; without op offload at 8192 tokens the KV got 10240 cells and the 18642-token prompt did not fit.
// test: tests/test-opoffload-fork.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>

struct opoff_plan_fork {
    bool     off     = false;   // run this context without op offload
    uint32_t ubatch  = 0;       // the micro-batch without op offload (0: keep the configured one)
    double   fixed   = 0.0;     // bytes of the graph that do not grow with tokens
    double   per_tok = 0.0;     // bytes a token
    double   need    = 0.0;     // what a 512-token pass with op offload needs beside the margins
    double   left    = 0.0;     // bytes left beside the margins without op offload
};

// the op offload fit check also runs for passes of 128 to 512 tokens on a GPU run: common.cpp retries a failed context at smaller micro-batches, and a check
// that stopped above 512 let the 512-token try through to a context whose every request failed with "Compute error" (Inkling 2 Mini, 8 GB 4060, 2026-11-27)
static inline bool opoff_check_due_fork(bool op_offload, int n_gpu_layers, uint32_t n_tokens) {
    return n_tokens > 512 || (op_offload && n_gpu_layers > 0 && n_tokens >= 128);
}

// lt / lt2: the graph's bytes on the card at n_tokens and n_tokens / 2; margin: the card's VRAM margin; kv_margin, kv_fixed, kv_row: the KV arena's margin,
// fixed part and bytes a cell (llama_moe_cache_kv_arena_shape_fork)
static inline opoff_plan_fork opoff_plan_fork_calc(size_t lt, size_t lt2, uint32_t n_tokens, size_t free_b, size_t margin, size_t kv_margin, size_t kv_fixed, size_t kv_row) {
    opoff_plan_fork p;
    if (lt == 0 || lt2 >= lt || n_tokens < 2) { return p; }
    p.per_tok = (double) (lt - lt2) / (n_tokens - n_tokens / 2);
    p.fixed   = std::max(0.0, (double) lt - p.per_tok * n_tokens);
    const uint32_t n_min = std::min<uint32_t>(512, n_tokens);   // a pass smaller than 512 tokens asks for its own size
    p.need    = p.fixed + std::max(margin, kv_margin) + kv_fixed + n_min * (p.per_tok + kv_row);
    if ((double) free_b >= p.need) { return p; }
    p.off  = true;
    p.left = (double) free_b - (p.need - n_min * (p.per_tok + kv_row) - p.fixed);
    const uint32_t u_off = p.left > 0 ? (uint32_t) std::min<double>(n_tokens, p.left / 2 / p.per_tok) / 32 * 32 : 512;
    if (u_off < n_tokens) { p.ubatch = std::max<uint32_t>(512, u_off); }
    return p;
}
