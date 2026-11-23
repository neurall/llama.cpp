// fork: the elementwise ops of the ew-chain fuser (ew-chain-fork.cu) as one device function, shared by its own kernel and by the matvec epilogue
// (ggml_cuda_ew_epi in common.cuh: a chain run on each matvec result as mmvq writes it, instead of a second kernel).
#pragma once
#include "common.cuh"
#include "unary.cuh"

enum ewc_op : uint8_t {
    EWC_ADD, EWC_SUB, EWC_MUL, EWC_DIV, EWC_SCALE, EWC_SQR, EWC_SQRT, EWC_LOG, EWC_SIN, EWC_COS, EWC_CLAMP, EWC_FILL,
    EWC_ABS, EWC_NEG, EWC_RELU, EWC_TANH, EWC_SIGMOID, EWC_SILU, EWC_GELU, EWC_EXP,
};

static __host__ __device__ __forceinline__ bool ewc_binary(uint8_t op) { return op <= EWC_DIV; }

// math per op as the single-op kernels do it (unary.cu, binbcast.cu, scale.cu, clamp.cu)
static __device__ __forceinline__ float ewc_eval(uint8_t op, float a, float b, float p0, float p1) {
    switch (op) {
        case EWC_ADD:     return a + b;
        case EWC_SUB:     return a - b;
        case EWC_MUL:     return a * b;
        case EWC_DIV:     return a / b;
        case EWC_SCALE:   return p0 * a + p1;
        case EWC_SQR:     return a * a;
        case EWC_SQRT:    return sqrtf(a);
        case EWC_LOG:     return logf(a);
        case EWC_SIN:     return sinf(a);
        case EWC_COS:     return cosf(a);
        case EWC_CLAMP:   return fminf(fmaxf(a, p0), p1);
        case EWC_FILL:    return p0;
        case EWC_ABS:     return fabsf(a);
        case EWC_NEG:     return -a;
        case EWC_RELU:    return fmaxf(a, 0);
        case EWC_TANH:    return tanhf(a);
        case EWC_SIGMOID: return ggml_det_sigmoid(a);
        case EWC_SILU:    return ggml_cuda_op_silu_single(a);
        case EWC_GELU:    return ggml_cuda_op_gelu_single(a);
        default:          return expf(a);   // EWC_EXP
    }
}

static __device__ __forceinline__ float ewc_epi_get(const ggml_cuda_ew_epi & E, const float * r, int ref, float acc, int row) {
    if (ref >= 0) { return r[ref]; }
    const ggml_cuda_ew_epi::opd & o = E.o[-1 - ref];
    return o.p ? *(const float *) (o.p + (int64_t) (row % o.ne0) * o.nb0) : acc;
}

// runs the program on matvec row `row` whose value is acc, and writes the chain's last value
static __device__ __forceinline__ void ewc_epi_write(const ggml_cuda_ew_epi & E, float acc, int row) {
    float r[ggml_cuda_ew_epi::MAX_ST];
#pragma unroll 1
    for (int s = 0; s < E.n_st; ++s) {
        const ggml_cuda_ew_epi::step & S = E.st[s];
        const float a = S.op == EWC_FILL ? 0.0f : ewc_epi_get(E, r, S.a, acc, row);
        const float b = ewc_binary(S.op) ? ewc_epi_get(E, r, S.b, acc, row) : 0.0f;
        r[s] = ewc_eval(S.op, a, b, S.p0, S.p1);
    }
    *(float *) (E.dst + (int64_t) row * E.dst_nb0) = r[E.n_st - 1];
}
