#pragma once
// fork: the scheduler's card stamps class each split as expert (moe) or dense time by its first node (ggml-backend.cpp); the dense split's token shares come from the
// dense time per token, so an expert card's work over all tokens must not read as dense. Header-only: tests/test-pass-fork.cpp checks it without a GPU.
#include <cstring>

// the first node of a staging=0 stream part (its hook waits for the ring's copies)
static inline bool ggml_sched_cs_split_name(const char * name) { return strncmp(name, "ffn_moe_cs_slots-", 17) == 0; }

// expert time: an expert matmul first in a stream pass or over an expert range, a stream part's start, or a stream part's reduce over all tokens (ffn_moe_cs_rel:
// staged keeps that reduce in the matmuls' split; 2026-11-14 Qwen3.8 IQ4_XS 12k staging=0 read it as dense: the expert card 324 us/token against staged's 270, the
// helper took 51% of the tokens instead of 47%, and the expert card waited 1437 ms of the next pass against 944)
static inline bool ggml_sched_split_moe(bool mul_mat_id, bool cs_pass, bool expert_range, const char * name) {
    return (mul_mat_id && (cs_pass || expert_range)) || ggml_sched_cs_split_name(name) || strncmp(name, "ffn_moe_cs_rel-", 15) == 0;
}
