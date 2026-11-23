// fork: Inkling 2 loop controller as two kernels. llama_model_inkling2::graph::control builds, per call,
//   g      = concat(rms_norm(cur), rms_norm(msg), 0)                 [2E, T]
//   h      = mul_mat(W_down, g)                                       [R, T]   W_down [2E, R] f32
//   gain   = tanh(mul_mat(W_gain, h)), forget = tanh(mul_mat(W_forget, h))   [E, T]  W_* [R, E] f32
//   out    = (cur - forget*cur) + gain*msg
// 12 kernels; decode runs 48 calls a token on Inkling 2 Mini. Kernel 1: one block per (R row, token) does both sums of
// squares and both half dots in one read. Kernel 2: one warp per (embd column, token) does the gain and forget dots and the output.
// Nodes are found by their links from the anchor rms_norm; every non-view node in the range must be one of the 12.
// GGML_CUDA_CTRL_FUSE=0 turns it off.
#include "common.cuh"
#include "ggml-impl.h"

#include <cstdlib>

namespace {

constexpr int CT_MAX_R = 128;
constexpr int CT_MAX_T = 16;

struct ct_args {
    const char * c; int64_t c_nb1;   // cur rows
    const char * m; int64_t m_nb1;   // msg rows
    const float * wd;                // [2E, R]
    const float * wg; const float * wf;   // [R, E]
    char * d; int64_t d_nb1;
    int E, R;
    float eps_c, eps_m;
};

static __global__ void ct_down_kernel(const __grid_constant__ ct_args A, float * h) {
    const int j = blockIdx.x, t = blockIdx.y;
    const float * c = (const float *) (A.c + t*A.c_nb1);
    const float * m = (const float *) (A.m + t*A.m_nb1);
    const float * w = A.wd + (int64_t) j*2*A.E;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // ss cur, ss msg, dot cur, dot msg
    for (int e = threadIdx.x; e < A.E; e += blockDim.x) {
        const float xc = c[e], xm = m[e];
        v[0] += xc*xc; v[1] += xm*xm; v[2] += xc*w[e]; v[3] += xm*w[A.E + e];
    }
    __shared__ float red[4][32];
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = warp_reduce_sum(v[k]);
        if (lane == 0) { red[k][warp] = v[k]; }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = warp_reduce_sum(lane < nw ? red[k][lane] : 0.0f);
        }
        if (lane == 0) {
            h[t*A.R + j] = v[2]*rsqrtf(v[0]/A.E + A.eps_c) + v[3]*rsqrtf(v[1]/A.E + A.eps_m);
        }
    }
}

static __global__ void ct_out_kernel(const __grid_constant__ ct_args A, const float * h) {
    const int t = blockIdx.y;
    const int lane = threadIdx.x % WARP_SIZE;
    const int e = blockIdx.x*(blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    __shared__ float hs[CT_MAX_R];
    for (int j = threadIdx.x; j < A.R; j += blockDim.x) { hs[j] = h[t*A.R + j]; }
    __syncthreads();
    if (e >= A.E) { return; }
    const float * wg = A.wg + (int64_t) e*A.R;
    const float * wf = A.wf + (int64_t) e*A.R;
    float g = 0.0f, f = 0.0f;
    for (int j = lane; j < A.R; j += WARP_SIZE) { g += wg[j]*hs[j]; f += wf[j]*hs[j]; }
    g = warp_reduce_sum(g);
    f = warp_reduce_sum(f);
    if (lane == 0) {
        const float xc = ((const float *) (A.c + t*A.c_nb1))[e];
        const float xm = ((const float *) (A.m + t*A.m_nb1))[e];
        ((float *) (A.d + t*A.d_nb1))[e] = (xc - tanhf(f)*xc) + tanhf(g)*xm;
    }
}

static bool ct_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static bool ct_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// a row-contiguous f32 [E, T] operand
static bool ct_rows(const ggml_tensor * t, int64_t E, int64_t T) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == E && t->ne[1] == T && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
}

}  // namespace

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_ctrl_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = !getenv("GGML_CUDA_CTRL_FUSE") || atoi(getenv("GGML_CUDA_CTRL_FUSE")) != 0;
    const ggml_tensor * n0 = cgraph->nodes[i];
    if (!on || n0->op != GGML_OP_RMS_NORM) {
        return 0;
    }
    // the next 12 non-view nodes from i
    constexpr int N = 12;
    int idx[N];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < N; ++j) {
        if (!ct_is_view(cgraph->nodes[j])) { idx[n++] = j; }
    }
    if (n < N) {
        return 0;
    }
    // the add that ends it is one of them; walk back from it
    const ggml_tensor * add = nullptr;
    int last = -1;
    for (int k = 0; k < N; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (t->op == GGML_OP_ADD && t->src[0]->op == GGML_OP_SUB && t->src[1]->op == GGML_OP_MUL) { add = t; last = idx[k]; }
    }
    if (!add) {
        return 0;
    }
    const ggml_tensor * sub = add->src[0], * mgm = add->src[1];
    const ggml_tensor * cur = sub->src[0], * mfc = sub->src[1];
    if (mfc->op != GGML_OP_MUL || mfc->src[1] != cur || !ct_unary(mfc->src[0], GGML_UNARY_OP_TANH) || !ct_unary(mgm->src[0], GGML_UNARY_OP_TANH)) {
        return 0;
    }
    const ggml_tensor * tf = mfc->src[0], * tg = mgm->src[0], * msg = mgm->src[1];
    const ggml_tensor * mmf = tf->src[0], * mmg = tg->src[0];
    if (mmf->op != GGML_OP_MUL_MAT || mmg->op != GGML_OP_MUL_MAT || mmf->src[1] != mmg->src[1]) {
        return 0;
    }
    const ggml_tensor * mmd = mmf->src[1];
    if (mmd->op != GGML_OP_MUL_MAT || mmd->src[1]->op != GGML_OP_CONCAT || ggml_get_op_params_i32(mmd->src[1], 0) != 0) {
        return 0;
    }
    const ggml_tensor * cat = mmd->src[1];
    const ggml_tensor * rc = cat->src[0], * rm = cat->src[1];
    if (rc->op != GGML_OP_RMS_NORM || rm->op != GGML_OP_RMS_NORM || rc->src[0] != cur || rm->src[0] != msg) {
        return 0;
    }
    const ggml_tensor * set[N] = {rc, rm, cat, mmd, mmf, tf, mfc, sub, mmg, tg, mgm, add};
    for (int k = 0; k < N; ++k) {
        bool found = false;
        for (int s = 0; s < N; ++s) { found |= cgraph->nodes[idx[k]] == set[s]; }
        if (!found) { return 0; }
    }
    const int64_t E = cur->ne[0], T = cur->ne[1], R = mmd->ne[0];
    const ggml_tensor * wd = mmd->src[0], * wg = mmg->src[0], * wf = mmf->src[0];
    if (T > CT_MAX_T || R > CT_MAX_R || !ct_rows(cur, E, T) || !ct_rows(msg, E, T) || !ct_rows(add, E, T) ||
            wd->type != GGML_TYPE_F32 || wd->ne[0] != 2*E || wd->ne[1] != R || !ggml_is_contiguous(wd) ||
            wg->type != GGML_TYPE_F32 || wg->ne[0] != R || wg->ne[1] != E || !ggml_is_contiguous(wg) ||
            wf->type != GGML_TYPE_F32 || wf->ne[0] != R || wf->ne[1] != E || !ggml_is_contiguous(wf)) {
        return 0;
    }
    // every skipped node but the add is read only inside the controller
    for (int k = 0; k < N; ++k) {
        const int j = idx[k];
        const ggml_tensor * t = cgraph->nodes[j];
        if (t == add) { continue; }
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return 0; }
        int uses = 0;
        for (int q = i; q <= last; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[q]->src[s] == t; }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) { return 0; }
    }

    ct_args A{};
    A.c = (const char *) cur->data; A.c_nb1 = cur->nb[1];
    A.m = (const char *) msg->data; A.m_nb1 = msg->nb[1];
    A.wd = (const float *) wd->data; A.wg = (const float *) wg->data; A.wf = (const float *) wf->data;
    A.d = (char *) add->data; A.d_nb1 = add->nb[1];
    A.E = (int) E; A.R = (int) R;
    memcpy(&A.eps_c, rc->op_params, sizeof(float));
    memcpy(&A.eps_m, rm->op_params, sizeof(float));
    ggml_cuda_pool_alloc<float> h(ctx.pool(), T*R);
    ct_down_kernel<<<dim3((unsigned) R, (unsigned) T), 256, 0, ctx.stream()>>>(A, h.get());
    ct_out_kernel<<<dim3((unsigned) ((E + 7)/8), (unsigned) T), 256, 0, ctx.stream()>>>(A, h.get());
    return last - i;
}
