#pragma once
// fork: checks on the size of a pass's graph, pure (tests/test-graphcheck-fork.cpp).

#include <cstddef>
#include <cstdint>

// A pass-size probe that asks for far more per token than the context's first reserve (the plain call's graph, which also holds the staged experts as inputs, so
// a prompt pass's graph is no bigger) holds a tensor that lives too long: 2026-11-14, GLM two-card dense split, first reserve 8201 MiB at 16384 tokens on CUDA0,
// the probe 58347 MiB at the same size (every expert layer's card chain in the graph with no reader); passes were capped at 3776 tokens.
// The first reserve's bytes scale through the origin (as the free-VRAM cut reads them); 512 MiB of slack keeps small passes with their fixed part clear.
inline bool llama_graph_jump_fork(size_t base_bytes, uint32_t base_tokens, size_t probe_bytes, uint32_t probe_tokens) {
    if (base_bytes == 0 || base_tokens == 0 || probe_tokens == 0) {
        return false;   // no first reserve to hold it against
    }
    const double at = (double) base_bytes * probe_tokens / base_tokens;
    return (double) probe_bytes > 2.0 * at + 512.0 * 1048576.0;
}
