// fork: one-token short convolution step as one kernel. The graph builds, per convolve (Inkling 2, also any build_conv_state_store model):
//   x    = concat(past, new, 0)          [n_conv, C, S]    past = the last n_conv - 1 inputs from the recurrent state
//   cpy(view(x, last n_conv - 1), cell)                      the state store
//   out  = ssm_conv(x, w)                [C, 1, S]
// 3 kernels; Inkling 2 Mini decode runs 192 a token. Here one thread a (channel, seq) does all three and still writes x, so later views of
// x (dynamic taps) read it as before. The cpy and ssm_conv are found as the next non-view nodes after the concat.
// GGML_CUDA_CONV_STEP=0 turns it off.
#include "common.cuh"
#include "ew-epi-fork.cuh"
#include "ggml-impl.h"

#include <cstdlib>

namespace {

constexpr int CS_MAX_CONV = 8;

struct cs_args {
    const char * past; int64_t p_nb0, p_nb1, p_nb2;
    const char * nw;   int64_t n_nb1, n_nb2;            // the new input, one column
    char * x;          int64_t x_nb1, x_nb2;            // the concat, contiguous in dim 0
    const char * w;    int64_t w_nb1;                   // [n_conv, C]
    char * out;        int64_t o_nb0, o_nb2;            // [C, 1, S]
    char * cell;       int64_t c_nb1;                   // [n_cols*C, S]
    int n_conv, C, S, cell_off;                         // cell_off: first x column the store copies
};

template <bool epi>
static __device__ __forceinline__ void cs_body(const cs_args & A, const ewc_prog * P) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= A.C) { return; }
    float v[CS_MAX_CONV];
#pragma unroll
    for (int j = 0; j < CS_MAX_CONV; ++j) {
        if (j < A.n_conv - 1) { v[j] = *(const float *) (A.past + j*A.p_nb0 + c*A.p_nb1 + s*A.p_nb2); }
        else if (j == A.n_conv - 1) { v[j] = *(const float *) (A.nw + c*A.n_nb1 + s*A.n_nb2); }
    }
    float * xr = (float *) (A.x + c*A.x_nb1 + s*A.x_nb2);
    const float * wr = (const float *) (A.w + c*A.w_nb1);
    float * cr = (float *) (A.cell + s*A.c_nb1) + (int64_t) c*(A.n_conv - A.cell_off);
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < CS_MAX_CONV; ++j) {
        if (j < A.n_conv) {
            xr[j] = v[j];
            sum += v[j]*wr[j];
            if (j >= A.cell_off) { cr[j - A.cell_off] = v[j]; }
        }
    }
    if constexpr (epi) {
        ewc_run_at(*P, (int64_t) s*A.C + c, sum);   // the elementwise run after the conv, its value as operand (out is [C, 1, S] contiguous)
    } else {
        *(float *) (A.out + c*A.o_nb0 + s*A.o_nb2) = sum;
    }
}

static __global__ void cs_kernel(const __grid_constant__ cs_args A) { cs_body<false>(A, nullptr); }
static __global__ void cs_kernel_epi(const __grid_constant__ cs_args A, const __grid_constant__ ewc_prog P) { cs_body<true>(A, &P); }

static bool cs_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

}  // namespace

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_conv_step_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = !getenv("GGML_CUDA_CONV_STEP") || atoi(getenv("GGML_CUDA_CONV_STEP")) != 0;
    const ggml_tensor * x = cgraph->nodes[i];
    if (!on || x->op != GGML_OP_CONCAT || ggml_get_op_params_i32(x, 0) != 0 || x->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * past = x->src[0], * nw = x->src[1];
    const int n_conv = (int) x->ne[0];
    if (!past || !nw || past->type != GGML_TYPE_F32 || nw->type != GGML_TYPE_F32 || nw->ne[0] != 1 || n_conv > CS_MAX_CONV || n_conv < 2 ||
            x->ne[3] != 1 || x->nb[0] != sizeof(float) || !ggml_is_contiguous(x)) {
        return 0;
    }
    // the next two non-view nodes: the state store cpy and the ssm_conv, in either order
    const ggml_tensor * cpy = nullptr, * conv = nullptr;
    int last = -1, i_conv = -1;
    for (int j = i + 1; j < cgraph->n_nodes && (!cpy || !conv); ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (cs_is_view(t)) { continue; }
        if (t->op == GGML_OP_CPY && !cpy && t->src[0]->view_src == x) { cpy = t; last = j; continue; }
        if (t->op == GGML_OP_SSM_CONV && !conv && t->src[0] == x) { conv = t; last = j; i_conv = j; continue; }
        return 0;
    }
    if (!cpy || !conv) {
        return 0;
    }
    // the store: the last n_cols columns of x, as [n_cols, C, S] contiguous rows into f32 cells [n_cols*C, S]
    const ggml_tensor * lv = cpy->src[0], * cell = cpy->src[1];
    const int64_t C = x->ne[1], S = x->ne[2];
    const int64_t n_cols = lv->ne[0];
    const int64_t off = (int64_t) lv->view_offs / (int64_t) sizeof(float);
    if (cell->type != GGML_TYPE_F32 || lv->ne[1] != C || lv->ne[2] != S || lv->nb[1] != x->nb[1] || lv->nb[2] != x->nb[2] ||
            off + n_cols != n_conv || lv->view_offs % sizeof(float) != 0 || cell->ne[0] != n_cols*C || cell->nb[0] != sizeof(float) || ggml_nrows(cell) != S) {
        return 0;
    }
    const ggml_tensor * w = conv->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != n_conv || w->ne[1] != C || w->nb[0] != sizeof(float) || conv->type != GGML_TYPE_F32 || conv->ne[1] != 1) {
        return 0;
    }
    // nodes between that are not the three or views stop it (nothing else may run in between)
    for (int j = i + 1; j < last; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t != cpy && t != conv && !cs_is_view(t)) { return 0; }
    }

    cs_args A{};
    A.past = (const char *) past->data; A.p_nb0 = past->nb[0]; A.p_nb1 = past->nb[1]; A.p_nb2 = past->nb[2];
    A.nw = (const char *) nw->data; A.n_nb1 = nw->nb[1]; A.n_nb2 = nw->nb[2];
    A.x = (char *) x->data; A.x_nb1 = x->nb[1]; A.x_nb2 = x->nb[2];
    A.w = (const char *) w->data; A.w_nb1 = w->nb[1];
    A.out = (char *) conv->data; A.o_nb0 = conv->nb[0]; A.o_nb2 = conv->nb[2];
    A.cell = (char *) cell->data; A.c_nb1 = cell->nb[1];
    A.n_conv = n_conv; A.C = (int) C; A.S = (int) S; A.cell_off = (int) off;
    const dim3 grid((unsigned) ((C + 127)/128), (unsigned) S);
    // the elementwise run after the conv (residual add, scale, dynamic taps) in the same kernel, element (c, s) by the thread that has the conv value
    static const bool epi_on = !getenv("GGML_CUDA_MM_EPI") || atoi(getenv("GGML_CUDA_MM_EPI")) != 0;
    ewc_prog P;
    int end = 0;
    if (epi_on && conv->nb[0] == sizeof(float) && conv->nb[2] == C*sizeof(float) && ggml_cuda_ew_after_fork(cgraph, i_conv, last, P, end)) {
        const ggml_tensor * out = cgraph->nodes[end];
        auto apart = [&](const ggml_tensor * t) {
            const char * a0 = (const char *) out->data, * b0 = (const char *) t->data;
            return a0 + ggml_nbytes(out) <= b0 || b0 + ggml_nbytes(t) <= a0;
        };
        // other threads read and write these while one thread writes its element of the run's output
        // the new column may be the output itself (an in-place residual add): thread (c, s) reads its element, then writes it
        const bool nw_own = nw->data == out->data && nw->nb[1] == sizeof(float) && out->nb[0] == sizeof(float) && ggml_nelements(out) == C*S &&
                            (S == 1 || nw->nb[2] == C*sizeof(float));
        if (apart(x) && apart(cell) && apart(past) && (apart(nw) || nw_own)) {
            cs_kernel_epi<<<grid, 128, 0, ctx.stream()>>>(A, P);
            return end - i;
        }
    }
    cs_kernel<<<grid, 128, 0, ctx.stream()>>>(A);
    return last - i;
}
