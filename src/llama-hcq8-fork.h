// fork: --moe hc-q8=1: the hyper-connection up/down matrices (Qwen3.8 hc_attn_*, hc_ffn_*, output_hc_*), stored BF16, are converted to Q8_0 at load.
//
// They are read every decode token by float matvecs: on Qwen3.8 IQ3_S 194 of them, 1.27 GB a token (2026-11-19 trace:
// mul_mat_vec_f 3.19 ms a token). As Q8_0 (offline test copy): decode +3.5% (67.9 vs 65.6 t/s), perplexity 1.3913 vs 1.3953 (10 x 1024).
// Only tensors in device buffers: a host buffer may map the file's bytes in place, which then no longer match the tensor. Pure: tests/test-hcq8-fork.cpp.
#pragma once

#include <cstdint>
#include <cstring>

static inline bool llama_hcq8_name_fork(const char * name) {
    const char * hc[] = { "hc_attn_up.weight", "hc_attn_down.weight", "hc_ffn_up.weight", "hc_ffn_down.weight", "output_hc_up.weight", "output_hc_down.weight" };
    const size_t n = strlen(name);
    for (const char * s : hc) {
        const size_t m = strlen(s);
        if (n >= m && strcmp(name + n - m, s) == 0 && (n == m || name[n - m - 1] == '.')) { return true; }
    }
    return false;
}

// convert this tensor? bf16: its type is BF16; ne0: its row length (Q8_0 blocks of 32); host: its buffer is host memory
static inline bool llama_hcq8_wanted_fork(bool on, const char * name, bool bf16, int64_t ne0, bool host) {
    return on && bf16 && !host && ne0 % 32 == 0 && llama_hcq8_name_fork(name);
}
