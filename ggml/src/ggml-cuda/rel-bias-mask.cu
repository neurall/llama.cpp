#include "rel-bias-mask.cuh"
#include "convert.cuh"

// one block per (token, head) row, the threads over the KV cells: the cell positions and the token's mask row are read once per head (L2), the table
// slice of a token is n_head x n_dist floats. The distance bucket is computed here from the two position lists (a dense [n_kv, n_tok] index input was
// 3.7 GiB at 112k cells and an 8192-token pass, filled on the host)
template <typename T_mask, typename T_out>
static __global__ void k_rel_bias_mask(const char * __restrict__ table, const int32_t * __restrict__ kv_pos, const int32_t * __restrict__ tok_pos,
        const char * __restrict__ mask, char * __restrict__ dst, const int64_t n_kv, const int64_t ext,
        const size_t tb1, const size_t tb2, const size_t mk1, const size_t d1, const size_t d2) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int64_t tp = tok_pos[t];
    const T_mask  * mk = (const T_mask  *) (mask + t*mk1);
    const char    * tb = table + h*sizeof(float) + t*tb2;
    T_out         * o  = (T_out *) (dst + t*d1 + h*d2);
    for (int64_t kv = threadIdx.x; kv < n_kv; kv += blockDim.x) {
        const int64_t dd = tp - kv_pos[kv];
        const int64_t d  = dd >= 0 && dd < ext ? dd : ext;
        const float v = *(const float *) (tb + d*tb1) + ggml_cuda_cast<float>(mk[kv]);
        o[kv] = ggml_cuda_cast<T_out>(v);
    }
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table   = dst->src[0];
    const ggml_tensor * kv_pos  = dst->src[1];
    const ggml_tensor * tok_pos = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];

    const int64_t n_kv   = dst->ne[0];
    const int64_t n_tok  = dst->ne[1];
    const int64_t n_head = dst->ne[2];
    GGML_ASSERT(n_tok <= INT32_MAX && n_head <= 65535);

    const dim3 grid((unsigned) n_tok, (unsigned) n_head, 1);
    const int  nth = (int) std::min<int64_t>(256, (n_kv + 31)/32*32);
    cudaStream_t stream = ctx.stream();

    const char    * tb = (const char *) table->data;
    const int32_t * kp = (const int32_t *) kv_pos->data;
    const int32_t * tp = (const int32_t *) tok_pos->data;
    const char    * mk = (const char *) mask->data;
    char          * o  = (char *) dst->data;
    const int64_t ext  = table->ne[1] - 1;

#define RBM_LAUNCH(TM, TO) k_rel_bias_mask<TM, TO><<<grid, nth, 0, stream>>>(tb, kp, tp, mk, o, n_kv, ext, table->nb[1], table->nb[2], mask->nb[1], dst->nb[1], dst->nb[2])
    if (mask->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        RBM_LAUNCH(half, half);
    } else if (mask->type == GGML_TYPE_F16) {
        RBM_LAUNCH(half, float);
    } else if (dst->type == GGML_TYPE_F16) {
        RBM_LAUNCH(float, half);
    } else {
        RBM_LAUNCH(float, float);
    }
#undef RBM_LAUNCH
}
