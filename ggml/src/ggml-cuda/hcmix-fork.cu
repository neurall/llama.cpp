// fork: one token's hyper-connection mix in two kernels instead of seven (Qwen3.8 / qwen4exp, build_hc_mix).
//
// The graph: lo = silu(scale * (W_down . xn)); gate = W_up . lo; mixed[d] = pre_scale * sum_c xn[c, d] * sigmoid(gate[c, d]).
// Unfused (trace 2026-11-19, RTX 3090): quantize, mmvq down, scale, silu, quantize, mmvq up, hc_pre = 7 launches, ~19.7 us a mix, 97 mixes a token.
// The weights are 2 x 3.3 MB, ~7 us at the card's bandwidth: the rest is launch gaps and two small kernels each too short to fill the card.
//
// Here (inspired by Strata's split down / up kernels, own code):
//   hc_down: grid (row groups, K splits); each block holds its K slice of xn in shared memory and writes partial sums part[ks][r]
//   hc_up:   every block sums the partials of all rows (L2, a few KB), applies scale + silu, then its rows of W_up, the sigmoid gate and the stream mean
// The input stays f32 (no q8_1 quantize): the dot reads Q8_0 weights against f32 activations.
// Matched in ggml_cuda_try_fuse on the MUL_MAT of W_down (GGML_CUDA_HC_FUSE=1); any other shape or type runs the graph as before.
#include "common.cuh"
#include "ggml-impl.h"

#include <cstdlib>

static constexpr int HCF_PAD = 36;   // floats per 32-element block in shared memory: a lane reading its own block with float4 loads hits distinct banks

// group of G lanes (G = 1..32, power of two) sums a dot of `nb` Q8_0 blocks of `row` against the padded shared vector xs (block b at xs + b*HCF_PAD)
template <int G>
static __device__ __forceinline__ float hcf_group_dot(const block_q8_0 * __restrict__ row, int nb, const float * __restrict__ xs, int gl) {
    float s = 0.0f;
    for (int b = gl; b < nb; b += G) {
        const block_q8_0 & q = row[b];
        const float d = __half2float(q.d);
        const uint16_t * q16 = (const uint16_t *) q.qs;
        const float4 * x4 = (const float4 *) (xs + b * HCF_PAD);
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint16_t a = q16[2 * k], c = q16[2 * k + 1];
            const float4 xv = x4[k];
            acc += (float) (int8_t) (a & 0xff) * xv.x + (float) (int8_t) (a >> 8) * xv.y + (float) (int8_t) (c & 0xff) * xv.z + (float) (int8_t) (c >> 8) * xv.w;
        }
        s += d * acc;
    }
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o, G);
    }
    return s;
}

// grid (ceil(R / blockDim.y), KS), block (32, rows per block)
static __global__ void hcf_down(const block_q8_0 * __restrict__ w, int64_t row_blocks, const float * __restrict__ x, int nb, int R, int nbs, float * __restrict__ part) {
    extern __shared__ float xs[];
    const int kb0 = blockIdx.y * nbs;
    const int kb1 = min(nb, kb0 + nbs);
    const int nthr = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < (kb1 - kb0) * 32; i += nthr) {
        xs[(i >> 5) * HCF_PAD + (i & 31)] = x[kb0 * 32 + i];
    }
    __syncthreads();
    const int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= R) {
        return;
    }
    const float s = hcf_group_dot<32>(w + r * row_blocks + kb0, kb1 - kb0, xs, threadIdx.x);
    if (threadIdx.x == 0) {
        part[blockIdx.y * R + r] = s;
    }
}

// grid ceil(n_embd / D), block (32, NW); G lanes per W_up row
template <int G>
static __global__ void hcf_up(const block_q8_0 * __restrict__ w, int64_t row_blocks, int nb_up, const float * __restrict__ part, int KS, int R,
                              float sc, float bi, const float * __restrict__ xn, float * __restrict__ dst, int n_embd, int hc, int D, float pre_scale) {
    extern __shared__ float sh[];
    float * lo   = sh;                          // nb_up * HCF_PAD
    float * gate = sh + nb_up * HCF_PAD;        // D * hc
    const int nthr = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int r = tid; r < nb_up * 32; r += nthr) {
        float v = 0.0f;
        if (r < R) {
            for (int k = 0; k < KS; ++k) { v += part[k * R + r]; }
            v = v * sc + bi;
            v = v / (1.0f + expf(-v));
        }
        lo[(r >> 5) * HCF_PAD + (r & 31)] = v;
    }
    __syncthreads();
    const int d0 = blockIdx.x * D;
    const int nrows = D * hc;
    const int groups = nthr / G, g = tid / G, gl = tid % G;
    for (int j = g; j < nrows; j += groups) {
        const int c = j / D, dd = j % D;
        const int d = d0 + dd;
        float s = 0.0f;
        if (d < n_embd) {
            s = hcf_group_dot<G>(w + (int64_t) (c * n_embd + d) * row_blocks, nb_up, lo, gl);
        }
        if (gl == 0) {
            gate[j] = s;
        }
    }
    __syncthreads();
    for (int dd = tid; dd < D; dd += nthr) {
        const int d = d0 + dd;
        if (d >= n_embd) { continue; }
        float s = 0.0f;
        for (int c = 0; c < hc; ++c) {
            s += xn[c * n_embd + d] / (1.0f + expf(-gate[c * D + dd]));
        }
        dst[d] = pre_scale * s;
    }
}

static bool hcf_is_view(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool hcf_reaches(const ggml_tensor * t, const ggml_tensor * target) {   // t is target or a chain of views of it
    for (; t; t = t->view_src) {
        if (t == target) { return true; }
        if (!hcf_is_view(t->op)) { return false; }
    }
    return false;
}

// returns the nodes to skip after i (0: no match)
int ggml_cuda_try_fuse_hc_mix_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_CUDA_HC_FUSE") && atoi(getenv("GGML_CUDA_HC_FUSE")) != 0;
    if (!on) {
        return 0;
    }
    const ggml_tensor * down = cgraph->nodes[i];
    if (down->op != GGML_OP_MUL_MAT || down->src[0]->type != GGML_TYPE_Q8_0 || down->src[1]->type != GGML_TYPE_F32 || down->ne[1] != 1 || ggml_nrows(down->src[1]) != 1) {
        return 0;
    }
    int idx[4];
    for (int k = 0, j = i + 1; k < 4; ++j) {
        if (j >= cgraph->n_nodes) { return 0; }
        const ggml_op op = cgraph->nodes[j]->op;
        if (hcf_is_view(op)) { continue; }
        idx[k++] = j;
    }
    const ggml_tensor * scale = cgraph->nodes[idx[0]];
    const ggml_tensor * silu  = cgraph->nodes[idx[1]];
    const ggml_tensor * up    = cgraph->nodes[idx[2]];
    const ggml_tensor * pre   = cgraph->nodes[idx[3]];
    if (scale->op != GGML_OP_SCALE || scale->src[0] != down || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != scale ||
        up->op != GGML_OP_MUL_MAT || up->src[1] != silu || up->src[0]->type != GGML_TYPE_Q8_0 || pre->op != GGML_OP_DSV4_HC_PRE ||
        !hcf_reaches(pre->src[1], up) || pre->src[0]->data != down->src[1]->data) {
        return 0;
    }
    const ggml_tensor * xs = pre->src[0];
    const int n_embd = (int) xs->ne[0], hc = (int) xs->ne[1];
    if (xs->ne[2] != 1 || ggml_get_op_params_i32(pre, 1) == 0 || !ggml_is_contiguous(xs) || !ggml_is_contiguous(down->src[1]) || !ggml_is_contiguous(pre) ||
        (int64_t) n_embd * hc != up->ne[0] || down->ne[0] != up->src[0]->ne[0] || up->src[0]->ne[0] % 32 != 0 || down->src[0]->ne[0] % 32 != 0) {
        return 0;
    }
    // each intermediate feeds only the next node of the chain (the views between them included)
    for (int j = i; j < idx[3]; ++j) {
        if (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT) { return 0; }
        if (ggml_node_get_use_count(cgraph, j) != 1) { return 0; }
    }
    const int K = (int) down->src[0]->ne[0], R = (int) down->src[0]->ne[1];
    const int nb = K / 32, nb_up = R / 32;
    if (R % 32 != 0 || nb_up > 32) {
        return 0;
    }
    static const int KS  = getenv("GGML_CUDA_HC_KS")  ? atoi(getenv("GGML_CUDA_HC_KS"))  : 8;
    static const int RPB = getenv("GGML_CUDA_HC_RPB") ? atoi(getenv("GGML_CUDA_HC_RPB")) : 8;
    static const int D   = getenv("GGML_CUDA_HC_D")   ? atoi(getenv("GGML_CUDA_HC_D"))   : 8;
    static const int NW  = getenv("GGML_CUDA_HC_NW")  ? atoi(getenv("GGML_CUDA_HC_NW"))  : 4;
    const int nbs = (nb + KS - 1) / KS;
    static bool said = false;
    if (!said) {
        said = true;
        GGML_LOG_WARN("hc fuse: %s K=%d R=%d n_embd=%d hc=%d: down grid %dx%d of %d rows, up grid %d of %d columns, 2 kernels for 7\n",
                      down->name, K, R, n_embd, hc, (R + RPB - 1) / RPB, KS, RPB, (n_embd + D - 1) / D, D);
    }
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) KS * R);

    float sc, bi;
    memcpy(&sc, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&bi, (const float *) scale->op_params + 1, sizeof(float));
    const float pre_scale = ggml_get_op_params_f32(pre, 0);

    hcf_down<<<dim3((R + RPB - 1) / RPB, KS), dim3(32, RPB), nbs * HCF_PAD * sizeof(float), st>>>(
        (const block_q8_0 *) down->src[0]->data, down->src[0]->nb[1] / sizeof(block_q8_0), (const float *) down->src[1]->data, nb, R, nbs, part.get());
    const size_t sh_up = ((size_t) nb_up * HCF_PAD + (size_t) D * hc) * sizeof(float);
    const dim3 gu((n_embd + D - 1) / D), bu(32, NW);
    const block_q8_0 * wu = (const block_q8_0 *) up->src[0]->data;
    const int64_t rbu = up->src[0]->nb[1] / sizeof(block_q8_0);
#define HCF_UP(G) hcf_up<G><<<gu, bu, sh_up, st>>>(wu, rbu, nb_up, part.get(), KS, R, sc, bi, (const float *) xs->data, (float *) pre->data, n_embd, hc, D, pre_scale)
    if (nb_up <= 8)       { HCF_UP(8); }
    else if (nb_up <= 16) { HCF_UP(16); }
    else                  { HCF_UP(32); }
#undef HCF_UP
    CUDA_CHECK(cudaGetLastError());
    return idx[3] - i;
}
