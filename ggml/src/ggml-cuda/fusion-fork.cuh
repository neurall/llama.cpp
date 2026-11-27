#pragma once
// fork: what the fork's fusion code (fusion.cu) and ggml-cuda.cu share beside upstream's fusion.cuh

#include "common.cuh"
#include "q8reuse-fork.h"
#include "mix-match-fork.h"
#include "ctrl-match-fork.h"

#include <atomic>
#include <cstdint>
#include <vector>

// ggml_cuda_try_fuse with the fork's nodes_fused: a node a fusion ran ahead of its turn is marked, the evaluate loop skips it there (fusion.cu)
int ggml_cuda_try_fuse_fork(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i, std::vector<bool> & nodes_fused);

// the matvec grouping's members however far ahead, for the stream note of the launch that ran them (ggml-cuda.cu)
extern std::vector<int> g_mm_multi_far;

// defined in ggml-cuda.cu, used by the fusions in fusion.cu as well
bool ggml_cuda_fusion_disabled();   // GGML_CUDA_DISABLE_FUSION
extern std::atomic<bool> g_obs_on;          // expert use counters: stock-placement observer (decode)
extern std::atomic<bool> g_obs_prompt_on;   // and the expert cache's prompt heat
void ggml_cuda_obs_count(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * ids);
void ggml_cuda_mul_mat_id_begin(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool zero = true);   // what a MUL_MAT_ID does before its kernel, fused or not

// Q8 reuse (GGML_CUDA_Q8_REUSE=1): the slot buffers, a slot for an input, a node's writes dropping the slots it overlaps, and the claim a fused norm
// makes again after the graph loop marks its nodes written (ggml_cuda_q8_revive_fork)
extern char * g_q8r_buf[GGML_CUDA_MAX_DEVICES];
void * ggml_cuda_q8_reuse_fork(int device, int lane, const void * src, const int64_t * ne, const int64_t * strides, size_t src_bytes, size_t q8_bytes, bool * hit);
void ggml_cuda_q8_written_fork(int device, const ggml_tensor * t);
struct ggml_cuda_q8_revive_fork_t { int device = -1, lane = 0; q8_reuse_key k; uintptr_t lo = 0, hi = 0; };
extern ggml_cuda_q8_revive_fork_t g_q8r_revive;

// the fork fusions that read inputs of the nodes they skip, and what graph_optimize keeps alive for them (ggml_cuda_fusion_add_alloc_deps, fusion.cu)
bool ggml_cuda_mix_fuse_on_fork();    // GGML_CUDA_MIX_FUSE, mix-fork.cu
bool ggml_cuda_ctrl_fuse_on_fork();   // GGML_CUDA_CTRL_FUSE, ctrl-fork.cu
int  ggml_cuda_conv_step_keep_fork(const ggml_cgraph * cgraph, int i, const ggml_tensor ** keep, int & n_keep);   // conv-step-fork.cu: the group's last node or -1
