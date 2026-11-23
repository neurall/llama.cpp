// fork: generic elementwise chain fuser. A run of consecutive graph nodes made of f32 elementwise ops (add, sub, mul, div, scale, sqr, sqrt, log,
// sin, cos, clamp, fill, unary) runs as ONE kernel that interprets a short program per output element, instead of one launch per op.
//
// Rules (ggml_cuda_try_fuse_ew_chain_fork):
//   - every node of the run but the last is used only inside the run and is no graph output: nothing outside reads the skipped tensors
//   - shapes broadcast by repetition (ggml's binary rule): each node's shape divides the last node's shape in every dim, each operand's divides its
//     node's. The kernel walks the last node's elements; node and operand k read at index i_d mod ne_d, which composes ((i mod a) mod b = i mod b, b | a)
//   - operands outside the run are read through their own strides (views, transposes); one that overlaps the output and is not laid out exactly like
//     it (each thread reads its own element before it writes it) stops the fusion
// Math per op is the one the single-op kernels use (unary.cu, binbcast.cu, scale.cu, clamp.cu), so a fused chain gives the unfused result.
// Inkling 2 decode (2026-11-23): 7430 kernels a token, ~6000 of them 1-3 us elementwise or small ops (clip, control, convolve chains).
// GGML_CUDA_EW_CHAIN=0 turns it off; GGML_CUDA_EW_CHAIN_LOG=1 logs each fused run once per shape.
#include "common.cuh"
#include "unary.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <set>
#include <string>

namespace {

constexpr int EWC_MAX_OPD  = 8;
constexpr int EWC_MAX_STEP = 16;

enum ewc_op : uint8_t {
    EWC_ADD, EWC_SUB, EWC_MUL, EWC_DIV, EWC_SCALE, EWC_SQR, EWC_SQRT, EWC_LOG, EWC_SIN, EWC_COS, EWC_CLAMP, EWC_FILL,
    EWC_ABS, EWC_NEG, EWC_RELU, EWC_TANH, EWC_SIGMOID, EWC_SILU, EWC_GELU, EWC_EXP,
};

struct ewc_opd  { const char * p; int ne[4]; int64_t nb[4]; };
struct ewc_step { uint8_t op; int8_t a, b; float p0, p1; };   // a, b: >= 0 the value of step a, < 0 operand -1 - a
struct ewc_prog {
    ewc_opd  opd[EWC_MAX_OPD];
    ewc_step st[EWC_MAX_STEP];
    int      n_st;
    int      n_opd;
    int      ne[4];
    int64_t  nb[4];
    char *   dst;
};

static __device__ __forceinline__ float ewc_get(const float * x, const float * r, int ref) {
    return ref >= 0 ? r[ref] : x[-1 - ref];
}

static __global__ void ewc_kernel(const __grid_constant__ ewc_prog P, const int n) {   // n <= INT32_MAX (ewc_node_ok): 32-bit index math
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) {
        int i[4];
        int t = e;
        i[0] = t % P.ne[0]; t /= P.ne[0];
        i[1] = t % P.ne[1]; t /= P.ne[1];
        i[2] = t % P.ne[2]; t /= P.ne[2];
        i[3] = t;
        // every operand loaded up front: independent loads in flight together, not one load latency per step
        float x[EWC_MAX_OPD];
#pragma unroll
        for (int k = 0; k < EWC_MAX_OPD; ++k) {
            if (k < P.n_opd) {
                const ewc_opd & o = P.opd[k];
                x[k] = *(const float *) (o.p + (i[0] % o.ne[0]) * o.nb[0] + (i[1] % o.ne[1]) * o.nb[1] + (i[2] % o.ne[2]) * o.nb[2] + (i[3] % o.ne[3]) * o.nb[3]);
            }
        }
        float r[EWC_MAX_STEP];
#pragma unroll 1
        for (int s = 0; s < P.n_st; ++s) {
            const ewc_step & S = P.st[s];
            const float a = S.op == EWC_FILL ? 0.0f : ewc_get(x, r, S.a);
            float v;
            switch (S.op) {
                case EWC_ADD:     v = a + ewc_get(x, r, S.b); break;
                case EWC_SUB:     v = a - ewc_get(x, r, S.b); break;
                case EWC_MUL:     v = a * ewc_get(x, r, S.b); break;
                case EWC_DIV:     v = a / ewc_get(x, r, S.b); break;
                case EWC_SCALE:   v = S.p0 * a + S.p1; break;
                case EWC_SQR:     v = a * a; break;
                case EWC_SQRT:    v = sqrtf(a); break;
                case EWC_LOG:     v = logf(a); break;
                case EWC_SIN:     v = sinf(a); break;
                case EWC_COS:     v = cosf(a); break;
                case EWC_CLAMP:   v = fminf(fmaxf(a, S.p0), S.p1); break;
                case EWC_FILL:    v = S.p0; break;
                case EWC_ABS:     v = fabsf(a); break;
                case EWC_NEG:     v = -a; break;
                case EWC_RELU:    v = fmaxf(a, 0); break;
                case EWC_TANH:    v = tanhf(a); break;
                case EWC_SIGMOID: v = ggml_det_sigmoid(a); break;
                case EWC_SILU:    v = ggml_cuda_op_silu_single(a); break;
                case EWC_GELU:    v = ggml_cuda_op_gelu_single(a); break;
                default:          v = expf(a); break;   // EWC_EXP
            }
            r[s] = v;
        }
        *(float *) (P.dst + i[0] * P.nb[0] + i[1] * P.nb[1] + i[2] * P.nb[2] + i[3] * P.nb[3]) = r[P.n_st - 1];
    }
}

// the program op of a node (-1: not elementwise here) and how many sources it reads (fill reads none: src0 gives the shape only)
static int ewc_op_of(const ggml_tensor * t, int & n_src) {
    n_src = 1;
    switch (t->op) {
        case GGML_OP_ADD:   n_src = 2; return EWC_ADD;
        case GGML_OP_SUB:   n_src = 2; return EWC_SUB;
        case GGML_OP_MUL:   n_src = 2; return EWC_MUL;
        case GGML_OP_DIV:   n_src = 2; return EWC_DIV;
        case GGML_OP_SCALE: return EWC_SCALE;
        case GGML_OP_SQR:   return EWC_SQR;
        case GGML_OP_SQRT:  return EWC_SQRT;
        case GGML_OP_LOG:   return EWC_LOG;
        case GGML_OP_SIN:   return EWC_SIN;
        case GGML_OP_COS:   return EWC_COS;
        case GGML_OP_CLAMP: return EWC_CLAMP;
        case GGML_OP_FILL:  n_src = 0; return EWC_FILL;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_ABS:     return EWC_ABS;
                case GGML_UNARY_OP_NEG:     return EWC_NEG;
                case GGML_UNARY_OP_RELU:    return EWC_RELU;
                case GGML_UNARY_OP_TANH:    return EWC_TANH;
                case GGML_UNARY_OP_SIGMOID: return EWC_SIGMOID;
                case GGML_UNARY_OP_SILU:    return EWC_SILU;
                case GGML_UNARY_OP_GELU:    return EWC_GELU;
                case GGML_UNARY_OP_EXP:     return EWC_EXP;
                default:                    return -1;
            }
        default: return -1;
    }
}

static bool ewc_divides(const ggml_tensor * small, const ggml_tensor * big) {
    for (int d = 0; d < 4; ++d) { if (small->ne[d] <= 0 || big->ne[d] % small->ne[d] != 0) { return false; } }
    return true;
}

static bool ewc_node_ok(const ggml_tensor * t) {
    int n_src;
    if (ewc_op_of(t, n_src) < 0 || !(t->flags & GGML_TENSOR_FLAG_COMPUTE) || t->type != GGML_TYPE_F32 || t->view_src != nullptr || ggml_nelements(t) > INT32_MAX) { return false; }
    for (int s = 0; s < 2; ++s) {
        if (t->src[s] && t->src[s]->type != GGML_TYPE_F32) { return false; }
    }
    if (t->op == GGML_OP_SCALE && ggml_nelements(t->src[0]) != ggml_nelements(t)) { return false; }   // same shape, the kernel walks dst
    return true;
}

static bool ewc_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

}  // namespace

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_ew_chain_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on  = !getenv("GGML_CUDA_EW_CHAIN") || atoi(getenv("GGML_CUDA_EW_CHAIN")) != 0;
    static const bool log = getenv("GGML_CUDA_EW_CHAIN_LOG") && atoi(getenv("GGML_CUDA_EW_CHAIN_LOG")) != 0;
    if (!on || !ewc_node_ok(cgraph->nodes[i])) {
        return 0;
    }
    // the longest run of fusable nodes from i
    int end = i;
    while (end + 1 < cgraph->n_nodes && end + 1 - i < EWC_MAX_STEP && ewc_node_ok(cgraph->nodes[end + 1])) { ++end; }
    // shrink until every node before the last is read only inside the run, shapes divide the last node's, and the operands fit
    for (; end > i; --end) {
        const ggml_tensor * last = cgraph->nodes[end];
        bool ok = true;
        for (int k = i; k < end && ok; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || !ewc_divides(t, last)) { ok = false; break; }
            int uses = 0;
            for (int j = k + 1; j <= end; ++j) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[j]->src[s] == t; }
            }
            ok = uses == ggml_node_get_use_count(cgraph, k);
        }
        if (!ok) { continue; }
        ewc_prog P{};
        int n_opd = 0;
        for (int k = i; k <= end && ok; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            int n_src;
            ewc_step & S = P.st[k - i];
            S.op = (uint8_t) ewc_op_of(t, n_src);
            S.a = S.b = 0;
            S.p0 = S.p1 = 0.0f;
            if (t->op == GGML_OP_SCALE || t->op == GGML_OP_CLAMP) { memcpy(&S.p0, (const float *) t->op_params, sizeof(float)); memcpy(&S.p1, (const float *) t->op_params + 1, sizeof(float)); }
            if (t->op == GGML_OP_FILL) { memcpy(&S.p0, t->op_params, sizeof(float)); }
            for (int s = 0; s < n_src && ok; ++s) {
                const ggml_tensor * src = t->src[s];
                int ref = 0;
                bool inside = false;
                for (int j = i; j < k; ++j) { if (cgraph->nodes[j] == src) { ref = j - i; inside = true; break; } }
                if (!inside) {
                    if (!ewc_divides(src, t)) { ok = false; break; }
                    int o = 0;
                    for (; o < n_opd; ++o) { if (P.opd[o].p == src->data && !memcmp(P.opd[o].nb, src->nb, sizeof(P.opd[o].nb)) && P.opd[o].ne[0] == src->ne[0] && P.opd[o].ne[1] == src->ne[1] && P.opd[o].ne[2] == src->ne[2] && P.opd[o].ne[3] == src->ne[3]) { break; } }
                    if (o == n_opd) {
                        if (n_opd == EWC_MAX_OPD) { ok = false; break; }
                        // an operand overlapping the output: only when laid out exactly like it (each thread reads its element, then writes it)
                        if (ewc_overlap(src, last) && (src->data != last->data || memcmp(src->ne, last->ne, sizeof(src->ne)) || memcmp(src->nb, last->nb, sizeof(src->nb)))) { ok = false; break; }
                        P.opd[o].p = (const char *) src->data;
                        for (int d = 0; d < 4; ++d) { P.opd[o].ne[d] = (int) src->ne[d]; P.opd[o].nb[d] = (int64_t) src->nb[d]; }
                        ++n_opd;
                    }
                    ref = -1 - o;
                }
                (s == 0 ? S.a : S.b) = (int8_t) ref;
            }
        }
        if (!ok) { continue; }
        P.n_st = end - i + 1;
        P.n_opd = n_opd;
        for (int d = 0; d < 4; ++d) { P.ne[d] = (int) last->ne[d]; P.nb[d] = (int64_t) last->nb[d]; }
        P.dst = (char *) last->data;
        const int64_t n = ggml_nelements(last);
        const int block = 256;
        const int grid = (int) std::min<int64_t>((n + block - 1) / block, 4096);
        ewc_kernel<<<grid, block, 0, ctx.stream()>>>(P, n);
        if (log) {
            static std::set<std::string> said;
            std::string l;
            for (int k = i; k <= end; ++k) { l += std::string(k == i ? "" : " ") + ggml_op_desc(cgraph->nodes[k]); }
            l += " [" + std::to_string(last->ne[0]) + "," + std::to_string(last->ne[1]) + "," + std::to_string(last->ne[2]) + "]";
            if (said.insert(l).second) { GGML_LOG_WARN("ew chain: %d nodes in one kernel: %s\n", end - i + 1, l.c_str()); }
        }
        return end - i;
    }
    return 0;
}
