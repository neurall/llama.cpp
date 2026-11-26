#pragma once

#include "common.cuh"

// fork: ggml_rel_bias_mask, out[kv, t, h] = table[h, rel_pos[kv, t], t] + mask[kv, t]
void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
