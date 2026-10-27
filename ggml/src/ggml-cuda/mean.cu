#include "mean.cuh"
#include "reduce_rows.cuh"

#ifdef GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif  // GGML_CUDA_USE_CUB

template <typename T> __global__ void divide_by_count(T * result, size_t count) {
    *result /= static_cast<T>(count);
}

void ggml_cuda_op_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

// Special case for reducing vectors
#ifdef GGML_CUDA_USE_CUB
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus iscapturing;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &iscapturing));
#endif // USE_CUDA_GRAPH
    if ((nrows == 1) &&
#ifdef USE_CUDA_GRAPH
            // Determine if CUDA graphs are effectively disabled for this context
            // (no graph instance exists and we're not capturing, OR graphs are explicitly enabled)
            (((ncols > 65536) &&
              (((!ctx.any_cuda_graph_has_instance()) && (iscapturing == cudaStreamCaptureStatusNone)) ||
               ctx.any_cuda_graph_enabled())) ||
            // CUDA graphs are enabled - use lower threshold
             ((ncols > 32768) &&
              !(((!ctx.any_cuda_graph_has_instance()) && (iscapturing == cudaStreamCaptureStatusNone)) ||
                ctx.any_cuda_graph_enabled())))) {
#else
        (ncols > 65536)) {
#endif // USE_CUDA_GRAPH
        // Single row - use device-wide reduction
        size_t           tmp_size = 0;
        ggml_cuda_pool & pool     = ctx.pool();

        DeviceReduce::Sum(nullptr, tmp_size, src0_d, dst_d, ncols, stream);

        ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
        DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, src0_d, dst_d, ncols, stream);

        // Divide by ncols
        divide_by_count<float><<<1, 1, 0, stream>>>(dst_d, ncols);
        return;
    }
#endif // GGML_CUDA_USE_CUB

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // Heuristic for block size selection to optimize occupancy.
    // See discussion in: https://github.com/ggml-org/llama.cpp/pull/15132
    dim3 block_dims;
    if ((nrows / nsm) < 2) {
        block_dims = dim3(512, 1, 1);
    } else {
        block_dims = dim3(ncols < 1024 ? 32 : 128, 1, 1);
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    if (ggml_is_contiguous(src0)) {
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/true>, launch_params, src0_d, dst_d, ncols);
        return;
    }

    const char * src0_d_bytes = (const char *) src0->data;
    ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/true>, launch_params, src0_d_bytes, dst_d, ncols,
            src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
}

// fork: the scratch the expert-cache pool lends a graph is computed per node before the graph runs (ggml_backend_cuda_nodes_scratch_fork, ggml-cuda.cu). This
// mirrors ggml_cuda_op_mean: one long row reduces with CUB (its temporary storage from the pool, CUB's own size query with the same types). Whether a row of
// 32769..65536 columns does depends on the CUDA graph state when it runs, not known before: from 32769 columns on. tests/test-scratch-fork.cpp checks it.
size_t ggml_cuda_mean_scratch_fork(const ggml_tensor * dst, size_t top) {
#ifdef GGML_CUDA_USE_CUB
    const ggml_tensor * src0 = dst->src[0];
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
#ifdef USE_CUDA_GRAPH
    const int64_t min_cols = 32768;
#else
    const int64_t min_cols = 65536;
#endif // USE_CUDA_GRAPH
    if (nrows == 1 && ncols > min_cols) {
        size_t tmp_size = 0;
        CUDA_CHECK(DeviceReduce::Sum(nullptr, tmp_size, (const float *) nullptr, (float *) nullptr, ncols, (cudaStream_t) nullptr));
        if (tmp_size > 0) {
            top = GGML_PAD(top, 256) + tmp_size;
        }
    }
#else
    GGML_UNUSED(dst);
#endif // GGML_CUDA_USE_CUB
    return top;
}
