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
#include "ew-epi-fork.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <set>
#include <string>

namespace {

static __global__ void ewc_kernel(const __grid_constant__ ewc_prog P, const int64_t n) {
    for (int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (int64_t) gridDim.x * blockDim.x) {
        ewc_run_at(P, e, 0.0f);
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

static bool ewc_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
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

// the program of the longest fusable run from node i: P, the run's node indices ix[0..m), its last node index end (false: none of 2+ nodes)
static bool ewc_build(const ggml_cgraph * cgraph, int i, ewc_prog & P, int * ix, int & m, int & end, int min_m = 2) {
    if (!ewc_node_ok(cgraph->nodes[i])) {
        return false;
    }
    // the longest run of fusable nodes from i; view nodes of tensors outside the run (operand views) compute nothing and are passed over
    int n_ix = 0;
    ix[n_ix++] = i;
    for (int j = i + 1; j < cgraph->n_nodes && n_ix < EWC_MAX_STEP; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ewc_is_view(t)) {
            bool of_run = false;
            for (int k = 0; k < n_ix; ++k) { of_run |= t->view_src == cgraph->nodes[ix[k]] || t->src[0] == cgraph->nodes[ix[k]]; }
            if (of_run) { break; }
            continue;
        }
        if (!ewc_node_ok(t)) { break; }
        ix[n_ix++] = j;
    }
    // shrink until every node before the last is read only inside the run, shapes divide the last node's, and the operands fit
    for (m = n_ix; m >= min_m; --m) {
        end = ix[m - 1];
        const ggml_tensor * last = cgraph->nodes[end];
        bool ok = true;
        for (int kk = 0; kk < m - 1 && ok; ++kk) {
            const ggml_tensor * t = cgraph->nodes[ix[kk]];
            if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || !ewc_divides(t, last)) { ok = false; break; }
            int uses = 0;
            for (int jj = kk + 1; jj < m; ++jj) {
                for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[ix[jj]]->src[s] == t; }
            }
            ok = uses == ggml_node_get_use_count(cgraph, ix[kk]);
        }
        if (!ok) { continue; }
        P = ewc_prog{};
        int n_opd = 0;
        for (int kk = 0; kk < m && ok; ++kk) {
            const ggml_tensor * t = cgraph->nodes[ix[kk]];
            int n_src;
            ewc_step & S = P.st[kk];
            S.op = (uint8_t) ewc_op_of(t, n_src);
            S.a = S.b = 0;
            S.p0 = S.p1 = 0.0f;
            if (t->op == GGML_OP_SCALE || t->op == GGML_OP_CLAMP) { memcpy(&S.p0, (const float *) t->op_params, sizeof(float)); memcpy(&S.p1, (const float *) t->op_params + 1, sizeof(float)); }
            if (t->op == GGML_OP_FILL) { memcpy(&S.p0, t->op_params, sizeof(float)); }
            for (int s = 0; s < n_src && ok; ++s) {
                const ggml_tensor * src = t->src[s];
                int ref = 0;
                bool inside = false;
                for (int jj = 0; jj < kk; ++jj) { if (cgraph->nodes[ix[jj]] == src) { ref = jj; inside = true; break; } }
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
        P.n_st = m;
        for (int d = 0; d < 4; ++d) { P.ne[d] = (int) last->ne[d]; P.nb[d] = (int64_t) last->nb[d]; }
        P.dst = (char *) last->data;
        return true;
    }
    return false;
}

// returns the nodes to skip after i (0: no fusion)
int ggml_cuda_try_fuse_ew_chain_fork(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on  = getenv("GGML_CUDA_EW_CHAIN") && atoi(getenv("GGML_CUDA_EW_CHAIN")) != 0;   // off by default: lost every measured on/off pair (2026-11-25)
    static const bool log = getenv("GGML_CUDA_EW_CHAIN_LOG") && atoi(getenv("GGML_CUDA_EW_CHAIN_LOG")) != 0;
    static const bool log_all = getenv("GGML_CUDA_EW_CHAIN_LOG") && atoi(getenv("GGML_CUDA_EW_CHAIN_LOG")) >= 2;   // every run, to count them
    ewc_prog P;
    int ix[EWC_MAX_STEP];
    int m = 0, end = 0;
    if (!on || !ewc_build(cgraph, i, P, ix, m, end)) {
        return 0;
    }
    const ggml_tensor * last = cgraph->nodes[end];
    {
        const int64_t n = ggml_nelements(last);
        const int block = 256;
        const int grid = (int) std::min<int64_t>((n + block - 1) / block, 4096);
        ewc_kernel<<<grid, block, 0, ctx.stream()>>>(P, n);
        if (log) {
            static std::set<std::string> said;
            std::string l;
            for (int kk = 0; kk < m; ++kk) { l += std::string(kk == 0 ? "" : " ") + ggml_op_desc(cgraph->nodes[ix[kk]]); }
            l += " <-";   // producers of the operands from outside the run
            for (int kk = 0; kk < m; ++kk) {
                const ggml_tensor * t = cgraph->nodes[ix[kk]];
                for (int s = 0; s < 2 && t->src[s]; ++s) {
                    const ggml_tensor * src = t->src[s];
                    bool inside = false;
                    for (int jj = 0; jj < kk; ++jj) { inside |= cgraph->nodes[ix[jj]] == src; }
                    if (inside) { continue; }
                    const ggml_tensor * pr = src;
                    while (pr->view_src || (ewc_is_view(pr) && pr->src[0])) { pr = pr->view_src ? pr->view_src : pr->src[0]; }
                    l += std::string(" ") + ggml_op_desc(pr) + (pr->op == GGML_OP_MUL_MAT ? "(" + std::string(ggml_type_name(pr->src[0]->type)) + ")" : "");
                }
            }
            l += " [" + std::to_string(last->ne[0]) + "," + std::to_string(last->ne[1]) + "," + std::to_string(last->ne[2]) + "]";
            if (said.insert(l).second || log_all) { GGML_LOG_WARN("ew chain: %d nodes in one kernel: %s\n", m, l.c_str()); }
        }
        return end - i;
    }
    return 0;
}

bool ggml_cuda_ew_after_fork(const ggml_cgraph * cgraph, int i_src, int i_from, ewc_prog & P, int & end) {
    static const bool log = getenv("GGML_CUDA_EW_CHAIN_LOG") && atoi(getenv("GGML_CUDA_EW_CHAIN_LOG")) != 0;
    const ggml_tensor * src = cgraph->nodes[i_src];
    if (src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) || (src->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_nelements(src) > INT32_MAX) {
        return false;
    }
    int j = i_from + 1;
    while (j < cgraph->n_nodes && ewc_is_view(cgraph->nodes[j])) { ++j; }
    int ix[EWC_MAX_STEP];
    int m = 0;
    if (j >= cgraph->n_nodes || !ewc_build(cgraph, j, P, ix, m, end, 1)) {
        return false;
    }
    const ggml_tensor * last = cgraph->nodes[end];
    if (ggml_nelements(last) != ggml_nelements(src) || !ggml_is_contiguous(last)) {
        return false;
    }
    // src and the view nodes of it up to the run's end: every read of them is by one another or by run nodes
    constexpr int MAX_V = 16;
    int vs[MAX_V];
    int n_v = 0;
    vs[n_v++] = i_src;
    for (int k = i_src + 1; k <= end && n_v < MAX_V; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (!ewc_is_view(t)) { continue; }
        for (int q = 0; q < n_v; ++q) {
            if (t->view_src == cgraph->nodes[vs[q]] || t->src[0] == cgraph->nodes[vs[q]]) { vs[n_v++] = k; break; }
        }
    }
    for (int q = 0; q < n_v; ++q) {
        const ggml_tensor * t = cgraph->nodes[vs[q]];
        int uses = 0;
        for (int k = 0; k < m; ++k) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[ix[k]]->src[s] == t; }
        }
        for (int r = 0; r < n_v; ++r) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[vs[r]]->src[s] == t; }
        }
        if (uses != ggml_node_get_use_count(cgraph, vs[q]) || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
    }
    // outside operands must be ready when src is: a node produced after everything src reads would make the fused launch wait for it, and the
    // stream plan keep it on the main chain (Inkling 2: the gate matvec beside attention took the MUL by the attention output, -1% decode)
    const auto base_of = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
    const auto index_of = [&](const ggml_tensor * t, int below) {
        for (int k = below; k >= 0; --k) { if (cgraph->nodes[k] == t) { return k; } }
        return -1;   // a leaf (weights, inputs)
    };
    int i_dep = -1;   // the latest node src reads
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (src->src[s]) { i_dep = std::max(i_dep, index_of(base_of(src->src[s]), i_src - 1)); }
    }
    for (int k = 0; k < m; ++k) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * o = cgraph->nodes[ix[k]]->src[s];
            if (!o) { continue; }
            bool inside = false;
            for (int r = 0; r < m && !inside; ++r) { inside = o == cgraph->nodes[ix[r]]; }
            for (int q = 0; q < n_v && !inside; ++q) { inside = o == cgraph->nodes[vs[q]]; }
            if (inside || base_of(o) == src) { continue; }
            if (index_of(base_of(o), end) > i_dep) {
                return false;
            }
        }
    }
    // operands over src's memory: only src's elements in the last node's own order (contiguous, same shape as the last node)
    const char * s0 = (const char *) src->data, * s1 = s0 + ggml_nbytes(src);
    int n_src_opd = 0;
    for (int o = 0; o < EWC_MAX_OPD && P.opd[o].p; ++o) {
        ewc_opd & d = P.opd[o];
        int64_t span = sizeof(float);
        for (int k = 0; k < 4; ++k) { span += (int64_t) (d.ne[k] - 1) * d.nb[k]; }
        if (!(d.p < s1 && s0 < d.p + span)) { continue; }
        bool same = d.p == s0 && d.nb[0] == (int64_t) sizeof(float);
        for (int k = 0; k < 4 && same; ++k) {
            same = d.ne[k] == last->ne[k] && (k == 0 || d.nb[k] == d.nb[k - 1] * d.ne[k - 1]);
        }
        if (!same) {
            return false;
        }
        d.p = nullptr;
        ++n_src_opd;
    }
    if (n_src_opd == 0) {
        return false;
    }
    if (log) {
        static std::set<std::string> said;
        std::string l = std::string(ggml_op_desc(src)) + " +";
        for (int k = 0; k < m; ++k) { l += std::string(" ") + ggml_op_desc(cgraph->nodes[ix[k]]); }
        l += " [" + std::to_string(last->ne[0]) + "," + std::to_string(last->ne[1]) + "," + std::to_string(last->ne[2]) + "," + std::to_string(last->ne[3]) + "]";
        if (said.insert(l).second) { GGML_LOG_WARN("ew after: %s\n", l.c_str()); }
    }
    return true;
}

// fork: the chain after matvec node i as its epilogue (ggml_cuda_ew_epi, run by mmvq as it writes each row): ggml_cuda_ew_after_fork's run whose last
// node is the matvec's rows [rows,1,1,1] and fits the epilogue's step and operand counts. A run of one node counts here (the launch is the kernel saved).
bool ggml_cuda_ew_epi_build_fork(const ggml_cgraph * cgraph, int i, ggml_cuda_ew_epi & E, int & end) {
    const ggml_tensor * mm = cgraph->nodes[i];
    const int64_t R = mm->ne[0];
    if (ggml_nelements(mm) != R) {
        return false;
    }
    ewc_prog P;
    if (!ggml_cuda_ew_after_fork(cgraph, i, i, P, end) || P.n_st > ggml_cuda_ew_epi::MAX_ST || P.ne[0] != R) {
        return false;
    }
    E = ggml_cuda_ew_epi{};
    for (int o = 0; o < EWC_MAX_OPD; ++o) {
        if (o >= 1 && !P.opd[o].p && !P.opd[o].nb[0]) { break; }   // past the used operands (a src operand has p == nullptr but its strides set)
        if (o >= ggml_cuda_ew_epi::MAX_OPD || P.opd[o].nb[0] > INT32_MAX) { return false; }
        E.o[o].p = P.opd[o].p;
        E.o[o].ne0 = P.opd[o].ne[0];
        E.o[o].nb0 = (int) P.opd[o].nb[0];
    }
    for (int s = 0; s < P.n_st; ++s) {
        E.st[s].op = P.st[s].op; E.st[s].a = P.st[s].a; E.st[s].b = P.st[s].b; E.st[s].p0 = P.st[s].p0; E.st[s].p1 = P.st[s].p1;
    }
    E.n_st = P.n_st;
    E.dst = P.dst;
    E.dst_nb0 = (int) P.nb[0];
    return true;
}
