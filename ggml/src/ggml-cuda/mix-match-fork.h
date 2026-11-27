#pragma once
// test: tests/test-fuse-deps-fork.cpp (no GPU)
#include "ggml.h"
#include "../ggml-impl.h"   // ggml_node_get_use_count, ggml_get_op_params_i32

#include <cstring>

// fork: the node match of Inkling 2's depth mix (mix-fork.cu), shared by the fusion and by graph_optimize, which keeps the tensors the fused kernels
// read alive until the mix's last node (ggml_cuda_fusion_add_alloc_deps): the kernels read the concat's inputs while they write the last matmul, and
// in the unfused order those inputs die at the concat, so their bytes were free for that matmul's output (experts on the CPU: decode PPL 1.61).
//   states = rms_norm(concat(stack, cur))                         [E, S, T]
//   p      = soft_max(scale * mul_mat(states, repeat(query)))      [S, Q, T]
//   out    = mul_mat(cont(transpose(states)), p)                   [E, Q, T]
constexpr int GGML_MIX_MAX_S = 64;
constexpr int GGML_MIX_MAX_Q = 4;
constexpr int GGML_MIX_MAX_T = 16;   // decode and short batches; long prompt passes keep the batched matmuls

struct ggml_mix_match_fork {
    const ggml_tensor * cat = nullptr, * R = nullptr, * cont = nullptr, * rep = nullptr, * mm1 = nullptr, * sm = nullptr, * mm2 = nullptr;
    const ggml_tensor * sa  = nullptr;   // the first sa state rows (the stack, or all states without a concat)
    const ggml_tensor * sb  = nullptr;   // the rows after them (cur), with a concat
    const ggml_tensor * qry = nullptr;
    int last = -1;                       // mm2's node
};

static inline bool ggml_mix_is_view_fork(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// the depth mix that starts at node i (a concat or the rms_norm), or false
static inline bool ggml_mix_match_fork_at(const ggml_cgraph * cgraph, int i, ggml_mix_match_fork & m) {
    const ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_CONCAT && n0->op != GGML_OP_RMS_NORM) {
        return false;
    }
    // the next non-view nodes from i: at most concat, rms_norm, cont, repeat, mul_mat, soft_max, mul_mat
    int idx[8];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < 7; ++j) {
        if (!ggml_mix_is_view_fork(cgraph->nodes[j])) { idx[n++] = j; }
    }
    const ggml_tensor * cat = n0->op == GGML_OP_CONCAT ? n0 : nullptr;
    const ggml_tensor * R   = cat ? (n > 1 ? cgraph->nodes[idx[1]] : nullptr) : n0;
    if (!R || R->op != GGML_OP_RMS_NORM || (cat && (R->src[0] != cat || ggml_get_op_params_i32(cat, 0) != 1))) {
        return false;
    }
    const ggml_tensor * cont = nullptr, * rep = nullptr, * mm1 = nullptr, * sm = nullptr, * mm2 = nullptr;
    int last = -1;
    for (int k = cat ? 2 : 1; k < n && !mm2; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_TRANSPOSE && t->src[0]->src[0] == R && !cont) { cont = t; }
        else if (t->op == GGML_OP_REPEAT && !rep) { rep = t; }
        else if (t->op == GGML_OP_MUL_MAT && t->src[0] == R && !mm1) { mm1 = t; }
        else if (t->op == GGML_OP_SOFT_MAX && mm1 && t->src[0] == mm1 && !t->src[1] && !t->src[2] && !sm) { sm = t; }
        else if (t->op == GGML_OP_MUL_MAT && cont && sm && t->src[0] == cont && t->src[1] == sm) { mm2 = t; last = idx[k]; }
        else { return false; }   // a node of something else inside the window
    }
    if (!mm2 || !mm1 || (rep && mm1->src[1] != rep)) {
        return false;
    }
    const ggml_tensor * qry = rep ? rep->src[0] : mm1->src[1];
    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));

    const auto f32 = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32; };
    const int64_t E = R->ne[0], S = R->ne[1], T = R->ne[2], Q = mm1->ne[1];
    if (max_bias != 0.0f || S > GGML_MIX_MAX_S || Q > GGML_MIX_MAX_Q || T > GGML_MIX_MAX_T || R->ne[3] != 1 || qry->ne[0] != E || qry->ne[1] != Q || qry->ne[3] != 1 ||
            T % qry->ne[2] != 0 || mm2->ne[0] != E || mm2->ne[1] != Q || mm2->ne[2] != T ||
            !f32(R) || !f32(qry) || !f32(mm1) || !f32(sm) || !f32(mm2) || !f32(R->src[0]) || (cat && (!f32(cat->src[0]) || !f32(cat->src[1])))) {
        return false;
    }
    // every skipped node is read only inside the mix
    for (int j = i; j < last; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
        int uses = 0;
        for (int k = j + 1; k <= last; ++k) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[k]->src[s] == t; }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) { return false; }
    }
    const ggml_tensor * sa = cat ? cat->src[0] : R->src[0];
    const ggml_tensor * sb = cat ? cat->src[1] : nullptr;
    if (cat ? (sa->ne[2] != T || sb->ne[2] != T) : (sa->ne[1] != S || sa->ne[2] != T)) {
        return false;
    }
    m = { cat, R, cont, rep, mm1, sm, mm2, sa, sb, qry, last };
    return true;
}

// what graph_optimize keeps alive until the mix's last node: every tensor the fused kernels read while they write mm2 (the states and the query)
static inline int ggml_mix_keep_fork(const ggml_mix_match_fork & m, const ggml_tensor * keep[3]) {
    int n = 0;
    keep[n++] = m.sa;
    if (m.sb) { keep[n++] = m.sb; }
    keep[n++] = m.qry;
    return n;
}
