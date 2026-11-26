#pragma once
// fork: CUDA's room before anything can load a kernel (src/llama-moecache-fork.cpp: the pool's start, "CUDA's room from the start", and room_lend at a lend).
// A kernel loads on its first use (lazy loading): at a graph's first run, and before that already when its scratch is planned (alias_scratch: the backend
// picks each node's kernel and queries it as a real call does; flash attention: cudaFuncSetAttribute, cudaOccupancyMaxActiveBlocksPerMultiprocessor).
// The plan runs outside a graph, where an out-of-memory aborts (ggml_cuda_error), and a load that fails breaks its kernel for good anyway. So the VRAM a
// load takes comes from the pool, lent before: CUDA gets the whole tail area (what every kernel of the build takes, measured by the probe processes, plus
// the page tables of the pins still to come: the area was sized from both) at the pool's start, where nothing is seen yet, and again at the lend of a
// shape class not seen yet, before its plan; the free VRAM beside the pool is never counted on. Until 2026-11-26 the start gave the pins' share only and
// the lend grew the room after the plan: Hy4's first call (the 2-token warmup) planned its flash attention with what the pins had left (4 MiB beside the
// pool) and aborted, "CUDA error: out of memory" at cudaFuncSetAttribute, at every micro-batch the start retried. 11-24 ran because that kernel fit then.
// test: tests/test-room-fork.cpp

#include "llama-poolalloc-fork.h"

#include <algorithm>

// The room's first byte before a step that may load kernels: the block edge at or below tail_real - room_max (the whole tail area) for a shape class not
// seen yet, else what is asked already (want). A room that is enough already gets no extra block; the room only grows here (a call gives back what CUDA
// did not take, llama_moe_cache_vram_watch_fork).
static inline size_t room_lo_before_plan(size_t want, bool seen, size_t vol_lo, size_t tail_real, size_t room_max, size_t base, size_t step) {
    if (seen || room_max == 0) { return want; }
    return std::min(want, poolx_room_edge_down(tail_real - std::min(tail_real, room_max), vol_lo, tail_real, base, step));
}
