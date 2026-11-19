// The scheduler's per-tensor tables between two graphs (ggml_backend_sched_reset). Pure, tested by tests/test-sched-reset-fork.cpp.
//
// The tables are sized for the largest graph a context may build (1.5 million slots for a 16384-token micro-batch) and were cleared whole before every
// graph: 11 ms of memset for a decode graph that fills about ten thousand slots (perf of the decode thread, Qwen3.8, 2026-11-19: once per rebuilt graph, the
// card idle meanwhile). Every table entry is reached through the hash set (hash_id marks its slot used), so only the marked slots can hold anything.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// `used`: the hash set's marks, 32 slots a word, `size` slots. Entries of marked slots go back to their empty values (backend id -1, no copies, not read
// across backends), then the marks are cleared. The tables must have been cleared whole once (ggml_backend_sched_new).
static inline void ggml_sched_reset_used_fork(uint32_t * used, size_t size, int * backend_ids, void ** copies, size_t per, bool * flags) {
    const size_t n_words = (size + 31) / 32;
    for (size_t w = 0; w < n_words; ++w) {
        uint32_t bits = used[w];
        if (!bits) { continue; }
        used[w] = 0;
        for (size_t b = 0; bits; ++b, bits >>= 1) {
            if (!(bits & 1u)) { continue; }
            const size_t i = w * 32 + b;
            backend_ids[i] = -1;
            memset(&copies[i * per], 0, per * sizeof(void *));
            flags[i] = false;
        }
    }
}
