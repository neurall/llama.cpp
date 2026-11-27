// fork: Inkling 2 depth mix as one kernel. llama_model_inkling2::graph::mix builds, per call,
//   states = rms_norm(concat(stack, cur))                         [E, S, T]
//   p      = soft_max(scale * mul_mat(states, repeat(query)))      [S, Q, T]
//   out    = mul_mat(cont(transpose(states)), p)                   [E, Q, T]
// 7 kernels; decode runs 120 such calls a token (2026-11-23: 840 of 5265 kernels a token on Inkling 2 Mini).
// Two kernels, both spread over the card: one block per state row (sum of squares and the Q query dots in one read), then
// blocks over embd chunks (softmax over S from those partials, each thread sums its column over the S rows).
// One block per token was 22 us a call (one SM reads all S rows twice). The nodes are found by their links,
// not by graph order, and every node between the first and the last must belong to the mix (views aside).
// GGML_CUDA_MIX_FUSE=0 turns it off.
#include "common.cuh"
#include "ggml-impl.h"
#include "mix-match-fork.h"
#include "fuse-apart-fork.h"

#include <cstdlib>

namespace {

constexpr int MIX_MAX_S = GGML_MIX_MAX_S;
constexpr int MIX_MAX_Q = GGML_MIX_MAX_Q;

struct mix_args {
    const char * a; int64_t a_nb0, a_nb1, a_nb2; int sa;   // the first sa state rows (the stack, or all states without a concat)
    const char * b; int64_t b_nb0, b_nb1, b_nb2;           // the rows after them (cur)
    const char * q; int64_t q_nb0, q_nb1, q_nb2; int q_ne2;
    char * d;       int64_t d_nb0, d_nb1, d_nb2;
    int S, Q, E;
    float eps, scale;
};

static __device__ __forceinline__ float mix_x(const mix_args & A, int s, int t, int e) {
    return s < A.sa ? *(const float *) (A.a + e*A.a_nb0 + s*A.a_nb1 + t*A.a_nb2)
                    : *(const float *) (A.b + e*A.b_nb0 + (s - A.sa)*A.b_nb1 + t*A.b_nb2);
}

// pass 1: one block per (state row, token): sum of squares and the Q query dots, to part[t][s][0..Q]
static __global__ void mix_rows_kernel(const __grid_constant__ mix_args A, float * part) {
    const int s = blockIdx.x, t = blockIdx.y;
    const char * qt = A.q + (t % A.q_ne2)*A.q_nb2;
    float v[1 + MIX_MAX_Q] = {0.0f};
    for (int e = threadIdx.x; e < A.E; e += blockDim.x) {
        const float x = mix_x(A, s, t, e);
        v[0] += x*x;
#pragma unroll
        for (int q = 0; q < MIX_MAX_Q; ++q) {
            if (q < A.Q) { v[1 + q] += x * *(const float *) (qt + e*A.q_nb0 + q*A.q_nb1); }
        }
    }
    __shared__ float red[1 + MIX_MAX_Q][32];
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int k = 0; k < 1 + MIX_MAX_Q; ++k) {
        v[k] = warp_reduce_sum(v[k]);
        if (lane == 0) { red[k][warp] = v[k]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < 1 + MIX_MAX_Q; ++k) {
            float r = lane < nw ? red[k][lane] : 0.0f;
            r = warp_reduce_sum(r);
            if (lane == 0 && k <= A.Q) { part[((int64_t) t*A.S + s)*(1 + MIX_MAX_Q) + k] = r; }
        }
    }
}

// pass 2: blocks over (embd chunk, token): rms scale and softmax over S from the partials, then the chunk's columns
static __global__ void mix_out_kernel(const __grid_constant__ mix_args A, const float * part) {
    const int t = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    __shared__ float inv[MIX_MAX_S];
    __shared__ float p[MIX_MAX_Q][MIX_MAX_S];
    const float * pt = part + (int64_t) t*A.S*(1 + MIX_MAX_Q);
    for (int s = threadIdx.x; s < A.S; s += blockDim.x) {
        const float r = rsqrtf(pt[s*(1 + MIX_MAX_Q)]/A.E + A.eps);   // rms_norm's scale
        inv[s] = r;
        for (int q = 0; q < A.Q; ++q) { p[q][s] = A.scale*r*pt[s*(1 + MIX_MAX_Q) + 1 + q]; }
    }
    __syncthreads();
    if (warp < A.Q) {
        float mx = -INFINITY;
        for (int s = lane; s < A.S; s += WARP_SIZE) { mx = fmaxf(mx, p[warp][s]); }
        mx = warp_reduce_max(mx);
        float sum = 0.0f;
        for (int s = lane; s < A.S; s += WARP_SIZE) { const float v = expf(p[warp][s] - mx); p[warp][s] = v; sum += v; }
        sum = warp_reduce_sum(sum);
        for (int s = lane; s < A.S; s += WARP_SIZE) { p[warp][s] /= sum; }
    }
    __syncthreads();
    const int e = blockIdx.x*blockDim.x + threadIdx.x;
    if (e >= A.E) { return; }
    float o[MIX_MAX_Q] = {0.0f};
    for (int s = 0; s < A.S; ++s) {
        const float v = mix_x(A, s, t, e)*inv[s];
#pragma unroll
        for (int q = 0; q < MIX_MAX_Q; ++q) { if (q < A.Q) { o[q] += p[q][s]*v; } }
    }
#pragma unroll
    for (int q = 0; q < MIX_MAX_Q; ++q) {
        if (q < A.Q) { *(float *) (A.d + e*A.d_nb0 + q*A.d_nb1 + t*A.d_nb2) = o[q]; }
    }
}

}  // namespace

bool ggml_cuda_mix_fuse_on_fork() {
    static const bool on = !getenv("GGML_CUDA_MIX_FUSE") || atoi(getenv("GGML_CUDA_MIX_FUSE")) != 0;
    return on;
}

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_mix_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_cuda_mix_fuse_on_fork();
    ggml_mix_match_fork m;
    if (!on || !ggml_mix_match_fork_at(cgraph, i, m)) {
        return 0;
    }
    // tripwire, should never fire: graph_optimize kept sa, sb and the query alive until mm2 (ggml_cuda_fusion_add_alloc_deps), so mm2 cannot sit on
    // them; a graph that reached here without those deps runs its nodes one by one rather than read its own writes
    if (!ggml_cuda_fuse_apart_fork(m.mm2, { m.sa, m.sb, m.qry }, false)) {
        static int said = 0;
        ggml_cuda_fuse_apart_warn_fork("depth mix", m.mm2, said);
        return 0;
    }
    const int64_t E = m.R->ne[0], S = m.R->ne[1], T = m.R->ne[2], Q = m.mm1->ne[1];
    mix_args A{};
    A.a = (const char *) m.sa->data; A.a_nb0 = m.sa->nb[0]; A.a_nb1 = m.sa->nb[1]; A.a_nb2 = m.sa->nb[2]; A.sa = (int) m.sa->ne[1];
    if (m.sb) {
        A.b = (const char *) m.sb->data; A.b_nb0 = m.sb->nb[0]; A.b_nb1 = m.sb->nb[1]; A.b_nb2 = m.sb->nb[2];
    }
    A.q = (const char *) m.qry->data; A.q_nb0 = m.qry->nb[0]; A.q_nb1 = m.qry->nb[1]; A.q_nb2 = m.qry->nb[2]; A.q_ne2 = (int) m.qry->ne[2];
    A.d = (char *) m.mm2->data; A.d_nb0 = m.mm2->nb[0]; A.d_nb1 = m.mm2->nb[1]; A.d_nb2 = m.mm2->nb[2];
    A.S = (int) S; A.Q = (int) Q; A.E = (int) E;
    memcpy(&A.eps, m.R->op_params, sizeof(float));
    memcpy(&A.scale, m.sm->op_params, sizeof(float));
    ggml_cuda_pool_alloc<float> part(ctx.pool(), T*S*(1 + MIX_MAX_Q));
    mix_rows_kernel<<<dim3((unsigned) S, (unsigned) T), 256, 0, ctx.stream()>>>(A, part.get());
    mix_out_kernel<<<dim3((unsigned) ((E + 127)/128), (unsigned) T), 128, 0, ctx.stream()>>>(A, part.get());
    return m.last - i;
}

