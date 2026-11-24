// fork: one token's hyper-connection mix in two kernels instead of seven (Qwen3.8 / qwen4exp, build_hc_mix).
//
// The graph: lo = silu(scale * (W_down . xn)); gate = W_up . lo; mixed[d] = pre_scale * sum_c xn[c, d] * sigmoid(gate[c, d]).
// Unfused (trace 2026-11-19, RTX 3090): quantize, mmvq down, scale, silu, quantize, mmvq up, hc_pre = 7 launches, ~19.7 us a mix, 97 mixes a token.
// The weights are 2 x 3.3 MB, ~7 us at the card's bandwidth: the rest is launch gaps and two small kernels each too short to fill the card.
//
// Here:
//   hc_down: grid (row groups, K splits); each block holds its K slice of xn in shared memory and writes partial sums part[ks][r]
//   hc_up:   every block sums the partials of all rows (L2, a few KB), applies scale + silu, then its rows of W_up, the sigmoid gate and the stream mean
// The input stays f32 (no q8_1 quantize): the dot reads Q8_0 weights against f32 activations.
// Matched in ggml_cuda_try_fuse on the MUL_MAT of W_down (on by default, GGML_CUDA_HC_FUSE=0 turns it off; decode perplexity against unquantized HC weights -0.16 +- 0.03%, the unfused graph -0.51 +- 0.08%: f32 activations stay closer than q8_1 ones; 9 us a layer less, 2026-11-19); any other shape or type runs the graph as before.
#include "common.cuh"
#include "ggml-impl.h"

#include <cstdlib>

// Weight loads first: every lane loads all its Q8_0 words of RPW rows x S steps into registers before the block stages its input
// and syncs, so the weight latency hides behind the staging (the first version loaded after the sync: 350-410 GB/s).
// Lanes 8b..8b+7 read one Q8_0 block, 4 values each: a warp step covers 4 consecutive blocks (136 contiguous bytes) of a row.
static __device__ __forceinline__ uint32_t hcf_ld16(const uint8_t * p) { return *(const uint16_t *) p; }

// grid (R / (NW*RPW), K / (128*S)), block (32, NW); x: K floats; part[kc][R]
template <int RPW, int S>
static __global__ void hcf_down(const uint8_t * __restrict__ w, int64_t row_bytes, const float * __restrict__ x, int R, float * __restrict__ part) {
    constexpr int KC = 128 * S;
    __shared__ __align__(16) float xs[KC];
    const int lane = threadIdx.x, warp = threadIdx.y, sub = lane & 7, bl = lane >> 3;
    const int kc = blockIdx.y;
    const int row0 = (blockIdx.x * blockDim.y + warp) * RPW;
    uint32_t q[RPW][S];
    float dq[RPW][S];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const uint8_t * row = w + (int64_t) min(row0 + r, R - 1) * row_bytes + (int64_t) kc * (KC / 32) * 34;
#pragma unroll
        for (int st = 0; st < S; ++st) {
            const uint8_t * blk = row + (4 * st + bl) * 34;
            dq[r][st] = __half2float(__ushort_as_half((unsigned short) hcf_ld16(blk)));
            q[r][st]  = hcf_ld16(blk + 2 + 4 * sub) | hcf_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    const int tid = warp * 32 + lane, nthr = 32 * blockDim.y;
    for (int i = tid; i < KC / 4; i += nthr) {
        *(float4 *) &xs[4 * i] = *(const float4 *) &x[(int64_t) kc * KC + 4 * i];
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int st = 0; st < S; ++st) {
            const float4 xv = *(const float4 *) &xs[32 * (4 * st + bl) + 4 * sub];
            const uint32_t v = q[r][st];
            acc += dq[r][st] * ((float) (int8_t) (v & 255u) * xv.x + (float) (int8_t) ((v >> 8) & 255u) * xv.y +
                                (float) (int8_t) ((v >> 16) & 255u) * xv.z + (float) (int8_t) (v >> 24) * xv.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { acc += ggml_cuda_shfl_xor(acc, o, 32); }
        if (lane == 0 && row0 + r < R) { part[(int64_t) kc * R + row0 + r] = acc; }
    }
}

// grid n_embd / D with D*hc = NW*RPW rows a block (row j: stream j / D, column d0 + j % D), block (32, NW); K_up = R = 32*NB
template <int RPW, int NB>
static __global__ void hcf_up(const uint8_t * __restrict__ w, int64_t row_bytes, const float * __restrict__ part, int KS, int R,
                              float sc, float bi, const float * __restrict__ xn, float * __restrict__ dst, int n_embd, int hc, int D, float pre_scale) {
    constexpr int S = (NB + 3) / 4;
    extern __shared__ __align__(16) float sh[];
    float * lo   = sh;               // 128*S (zero past R)
    float * gate = sh + 128 * S;     // D*hc
    const int lane = threadIdx.x, warp = threadIdx.y, sub = lane & 7, bl = lane >> 3;
    const int d0 = blockIdx.x * D;
    uint32_t q[RPW][S];
    float dq[RPW][S];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int j = warp * RPW + r, c = j / D, d = min(d0 + j % D, n_embd - 1);
        const uint8_t * row = w + (int64_t) (c * n_embd + d) * row_bytes;
#pragma unroll
        for (int st = 0; st < S; ++st) {
            const int b = 4 * st + bl;
            const uint8_t * blk = row + min(b, NB - 1) * 34;
            dq[r][st] = b < NB ? __half2float(__ushort_as_half((unsigned short) hcf_ld16(blk))) : 0.0f;
            q[r][st]  = hcf_ld16(blk + 2 + 4 * sub) | hcf_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    const int tid = warp * 32 + lane, nthr = 32 * blockDim.y;
    for (int r = tid; r < 128 * S; r += nthr) {
        float v = 0.0f;
        if (r < R) {
            for (int k = 0; k < KS; ++k) { v += part[k * R + r]; }
            v = v * sc + bi;
            v = v / (1.0f + expf(-v));
        }
        lo[r] = v;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int st = 0; st < S; ++st) {
            const float4 xv = *(const float4 *) &lo[32 * (4 * st + bl) + 4 * sub];
            const uint32_t v = q[r][st];
            acc += dq[r][st] * ((float) (int8_t) (v & 255u) * xv.x + (float) (int8_t) ((v >> 8) & 255u) * xv.y +
                                (float) (int8_t) ((v >> 16) & 255u) * xv.z + (float) (int8_t) (v >> 24) * xv.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { acc += ggml_cuda_shfl_xor(acc, o, 32); }
        if (lane == 0) { gate[warp * RPW + r] = acc; }
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
    static const bool on = !getenv("GGML_CUDA_HC_FUSE") || atoi(getenv("GGML_CUDA_HC_FUSE")) != 0;   // on unless GGML_CUDA_HC_FUSE=0 (see the top)
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
    constexpr int NW = 8, RPW = 4, S = 5, NB = 10;   // shortcut: one shape (KC 640, K_up 320) compiled; other shapes keep the unfused graph until tuned shapes are added
    const int rows_up = NW * RPW, D = rows_up / hc;
    if (K % (128 * S) != 0 || R % (NW * RPW) != 0 || R != 32 * NB || rows_up % hc != 0 || n_embd % D != 0 ||
        down->src[0]->nb[1] != (size_t) (K / 32) * 34 || up->src[0]->nb[1] != (size_t) NB * 34) {
        return 0;
    }
    const int KS = K / (128 * S);
    static bool said = false;
    if (!said) {
        said = true;
        GGML_LOG_WARN("hc fuse: %s K=%d R=%d n_embd=%d hc=%d: down grid %dx%d, up grid %d of %d columns, 2 kernels for 7\n",
                      down->name, K, R, n_embd, hc, R / (NW * RPW), KS, n_embd / D, D);
    }
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) KS * R);
    float sc, bi;
    memcpy(&sc, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&bi, (const float *) scale->op_params + 1, sizeof(float));
    const float pre_scale = ggml_get_op_params_f32(pre, 0);
    hcf_down<RPW, S><<<dim3(R / (NW * RPW), KS), dim3(32, NW), 0, st>>>(
        (const uint8_t *) down->src[0]->data, down->src[0]->nb[1], (const float *) down->src[1]->data, R, part.get());
    const size_t sh_up = (128 * ((NB + 3) / 4) + (size_t) D * hc) * sizeof(float);
    hcf_up<RPW, NB><<<n_embd / D, dim3(32, NW), sh_up, st>>>(
        (const uint8_t *) up->src[0]->data, up->src[0]->nb[1], part.get(), KS, R, sc, bi, (const float *) xs->data, (float *) pre->data, n_embd, hc, D, pre_scale);
    CUDA_CHECK(cudaGetLastError());
    return idx[3] - i;
}
