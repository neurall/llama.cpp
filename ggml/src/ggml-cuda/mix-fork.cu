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

#include <cstdlib>

namespace {

constexpr int MIX_MAX_S = 64;
constexpr int MIX_MAX_Q = 4;
constexpr int MIX_MAX_T = 16;   // decode and short batches; long prompt passes keep the batched matmuls

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

static bool mix_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static bool mix_f32(const ggml_tensor * t) { return t->type == GGML_TYPE_F32; }

}  // namespace

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_mix_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = !getenv("GGML_CUDA_MIX_FUSE") || atoi(getenv("GGML_CUDA_MIX_FUSE")) != 0;
    const ggml_tensor * n0 = cgraph->nodes[i];
    if (!on || (n0->op != GGML_OP_CONCAT && n0->op != GGML_OP_RMS_NORM)) {
        return 0;
    }
    // the next non-view nodes from i: at most concat, rms_norm, cont, repeat, mul_mat, soft_max, mul_mat
    int idx[8];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < 7; ++j) {
        if (!mix_is_view(cgraph->nodes[j])) { idx[n++] = j; }
    }
    const ggml_tensor * cat = n0->op == GGML_OP_CONCAT ? n0 : nullptr;
    const ggml_tensor * R   = cat ? (n > 1 ? cgraph->nodes[idx[1]] : nullptr) : n0;
    if (!R || R->op != GGML_OP_RMS_NORM || (cat && (R->src[0] != cat || ggml_get_op_params_i32(cat, 0) != 1))) {
        return 0;
    }
    const ggml_tensor * cont = nullptr, * rep = nullptr, * mm1 = nullptr, * sm = nullptr, * mm2 = nullptr;
    int last = -1;
    for (int k = cat ? 2 : 1; k < n && !mm2; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_TRANSPOSE && t->src[0]->src[0] == R && !cont) { cont = t; }
        else if (t->op == GGML_OP_REPEAT && !rep) { rep = t; }
        else if (t->op == GGML_OP_MUL_MAT && t->src[0] == R && !mm1) { mm1 = t; }
        else if (t->op == GGML_OP_SOFT_MAX && mm1 && t->src[0] == mm1 && !t->src[1] && !t->src[2] && !sm) { sm = t; }
        else if (t->op == GGML_OP_MUL_MAT && cont && sm && t->src[0] == cont && t->src[1] == sm) { mm2 = t; last = idx[k]; }
        else { return 0; }   // a node of something else inside the window
    }
    if (!mm2 || !mm1 || (rep && mm1->src[1] != rep)) {
        return 0;
    }
    const ggml_tensor * qry = rep ? rep->src[0] : mm1->src[1];
    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));

    const int64_t E = R->ne[0], S = R->ne[1], T = R->ne[2], Q = mm1->ne[1];
    if (max_bias != 0.0f || S > MIX_MAX_S || Q > MIX_MAX_Q || T > MIX_MAX_T || R->ne[3] != 1 || qry->ne[0] != E || qry->ne[1] != Q || qry->ne[3] != 1 ||
            T % qry->ne[2] != 0 || mm2->ne[0] != E || mm2->ne[1] != Q || mm2->ne[2] != T ||
            !mix_f32(R) || !mix_f32(qry) || !mix_f32(mm1) || !mix_f32(sm) || !mix_f32(mm2) || !mix_f32(R->src[0]) || (cat && (!mix_f32(cat->src[0]) || !mix_f32(cat->src[1])))) {
        return 0;
    }
    // every skipped node is read only inside the mix
    for (int j = i; j < last; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return 0; }
        int uses = 0;
        for (int k = j + 1; k <= last; ++k) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[k]->src[s] == t; }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) { return 0; }
    }

    mix_args A{};
    const ggml_tensor * sa = cat ? cat->src[0] : R->src[0];
    A.a = (const char *) sa->data; A.a_nb0 = sa->nb[0]; A.a_nb1 = sa->nb[1]; A.a_nb2 = sa->nb[2]; A.sa = (int) sa->ne[1];
    if (cat) {
        const ggml_tensor * sb = cat->src[1];
        A.b = (const char *) sb->data; A.b_nb0 = sb->nb[0]; A.b_nb1 = sb->nb[1]; A.b_nb2 = sb->nb[2];
        if (sa->ne[2] != T || sb->ne[2] != T) { return 0; }
    } else if (sa->ne[1] != S || sa->ne[2] != T) {
        return 0;
    }
    A.q = (const char *) qry->data; A.q_nb0 = qry->nb[0]; A.q_nb1 = qry->nb[1]; A.q_nb2 = qry->nb[2]; A.q_ne2 = (int) qry->ne[2];
    A.d = (char *) mm2->data; A.d_nb0 = mm2->nb[0]; A.d_nb1 = mm2->nb[1]; A.d_nb2 = mm2->nb[2];
    A.S = (int) S; A.Q = (int) Q; A.E = (int) E;
    memcpy(&A.eps, R->op_params, sizeof(float));
    memcpy(&A.scale, sm->op_params, sizeof(float));
    ggml_cuda_pool_alloc<float> part(ctx.pool(), T*S*(1 + MIX_MAX_Q));
    mix_rows_kernel<<<dim3((unsigned) S, (unsigned) T), 256, 0, ctx.stream()>>>(A, part.get());
    mix_out_kernel<<<dim3((unsigned) ((E + 127)/128), (unsigned) T), 128, 0, ctx.stream()>>>(A, part.get());
    return last - i;
}
