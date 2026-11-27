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
#include "ctrl-match-fork.h"
#include "fuse-apart-fork.h"

#include <cstdlib>

namespace {

constexpr int CT_MAX_R = GGML_CT_MAX_R;

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

}  // namespace

bool ggml_cuda_ctrl_fuse_on_fork() {
    static const bool on = !getenv("GGML_CUDA_CTRL_FUSE") || atoi(getenv("GGML_CUDA_CTRL_FUSE")) != 0;
    return on;
}

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_ctrl_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_cuda_ctrl_fuse_on_fork();
    ggml_ctrl_match_fork m;
    if (!on || !ggml_ctrl_match_fork_at(cgraph, i, m)) {
        return 0;
    }
    // tripwire, should never fire: graph_optimize kept cur and msg alive until the add (ggml_cuda_fusion_add_alloc_deps); the output kernel reads each
    // element's own cur and msg before it writes it, so the add exactly in place over one of them would be fine, any other overlap is not
    if (!ggml_cuda_fuse_apart_fork(m.add, { m.cur, m.msg }, true)) {
        static int said = 0;
        ggml_cuda_fuse_apart_warn_fork("loop controller", m.add, said);
        return 0;
    }
    const int64_t E = m.cur->ne[0], T = m.cur->ne[1], R = m.mmd->ne[0];
    ct_args A{};
    A.c = (const char *) m.cur->data; A.c_nb1 = m.cur->nb[1];
    A.m = (const char *) m.msg->data; A.m_nb1 = m.msg->nb[1];
    A.wd = (const float *) m.wd->data; A.wg = (const float *) m.wg->data; A.wf = (const float *) m.wf->data;
    A.d = (char *) m.add->data; A.d_nb1 = m.add->nb[1];
    A.E = (int) E; A.R = (int) R;
    memcpy(&A.eps_c, m.rc->op_params, sizeof(float));
    memcpy(&A.eps_m, m.rm->op_params, sizeof(float));
    ggml_cuda_pool_alloc<float> h(ctx.pool(), T*R);
    ct_down_kernel<<<dim3((unsigned) R, (unsigned) T), 256, 0, ctx.stream()>>>(A, h.get());
    ct_out_kernel<<<dim3((unsigned) ((E + 7)/8), (unsigned) T), 256, 0, ctx.stream()>>>(A, h.get());
    return m.last - i;
}
