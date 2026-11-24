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

// the full program (ew-chain's own kernel, and producers that run the chain after them: conv step). Element e of the last node is walked in its
// shape; operand k reads at i_d mod ne_d through its strides; an operand with p == nullptr is the producer's value for element e (acc).
constexpr int EWC_MAX_OPD  = 8;
constexpr int EWC_MAX_STEP = 16;

struct ewc_opd  { const char * p; int ne[4]; int64_t nb[4]; };
struct ewc_step { uint8_t op; int8_t a, b; float p0, p1; };   // a, b: >= 0 the value of step a, < 0 operand -1 - a
struct ewc_prog {
    ewc_opd  opd[EWC_MAX_OPD];
    ewc_step st[EWC_MAX_STEP];
    int      n_st;
    int      ne[4];
    int64_t  nb[4];
    char *   dst;
};

static __device__ __forceinline__ float ewc_get(const ewc_prog & P, const float * r, int ref, const int i[4], float acc) {
    if (ref >= 0) { return r[ref]; }
    const ewc_opd & o = P.opd[-1 - ref];
    if (!o.p) { return acc; }
    return *(const float *) (o.p + (i[0] % o.ne[0]) * o.nb[0] + (i[1] % o.ne[1]) * o.nb[1] + (i[2] % o.ne[2]) * o.nb[2] + (i[3] % o.ne[3]) * o.nb[3]);
}

static __device__ __forceinline__ void ewc_run_at(const ewc_prog & P, int64_t e, float acc) {
    int i[4];
    i[0] = (int) (e % P.ne[0]); e /= P.ne[0];
    i[1] = (int) (e % P.ne[1]); e /= P.ne[1];
    i[2] = (int) (e % P.ne[2]); e /= P.ne[2];
    i[3] = (int) e;
    float r[EWC_MAX_STEP];
#pragma unroll 1
    for (int s = 0; s < P.n_st; ++s) {
        const ewc_step & S = P.st[s];
        const float a = S.op == EWC_FILL ? 0.0f : ewc_get(P, r, S.a, i, acc);
        const float b = ewc_binary(S.op) ? ewc_get(P, r, S.b, i, acc) : 0.0f;
        r[s] = ewc_eval(S.op, a, b, S.p0, S.p1);
    }
    *(float *) (P.dst + i[0] * P.nb[0] + i[1] * P.nb[1] + i[2] * P.nb[2] + i[3] * P.nb[3]) = r[P.n_st - 1];
}

// host (ew-chain-fork.cu): the elementwise run that starts at the first non-view node after node i_from and reads the value of node i_src (directly
// or through views) only as operands, element for element (its last node walks src's elements in order). Its src operands get p == nullptr. end: the
// run's last node index. Nothing else reads src or its views.
bool ggml_cuda_ew_after_fork(const ggml_cgraph * cgraph, int i_src, int i_from, ewc_prog & P, int & end);
