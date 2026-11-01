#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// fork: only the rows `rows[0 .. n_rows)` of a tensor to f16, each at its place in a whole copy (the layout of to_fp16_nc); -1 skips a slot, and row 0 always goes too.
// false: no kernel for this type, nothing was launched. Blocks of ne00/qk per row, strides s01, s02, s03 in blocks.
bool ggml_cuda_to_fp16_rows(ggml_type type, const void * x, half * y, const int32_t * rows, int64_t n_rows,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03, int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, nv_bfloat162> && std::is_same_v<dst_t, float2>) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}
