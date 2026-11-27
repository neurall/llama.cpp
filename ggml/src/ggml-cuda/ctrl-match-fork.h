#pragma once
// test: tests/test-fuse-deps-fork.cpp (no GPU)
#include "ggml.h"
#include "../ggml-impl.h"   // ggml_node_get_use_count, ggml_get_op_params_i32

// fork: the node match of Inkling 2's loop controller (ctrl-fork.cu), shared by the fusion and by graph_optimize, which keeps cur and msg (read by the
// output kernel while it writes the add) alive until the add (ggml_cuda_fusion_add_alloc_deps; the same hazard as the depth mix, mix-match-fork.h).
//   g      = concat(rms_norm(cur), rms_norm(msg), 0)                 [2E, T]
//   h      = mul_mat(W_down, g)                                       [R, T]   W_down [2E, R] f32
//   gain   = tanh(mul_mat(W_gain, h)), forget = tanh(mul_mat(W_forget, h))   [E, T]  W_* [R, E] f32
//   out    = (cur - forget*cur) + gain*msg
constexpr int GGML_CT_MAX_R = 128;
constexpr int GGML_CT_MAX_T = 16;

struct ggml_ctrl_match_fork {
    const ggml_tensor * cur = nullptr, * msg = nullptr, * add = nullptr;
    const ggml_tensor * rc = nullptr, * rm = nullptr, * mmd = nullptr;   // the two rms_norms and the down matmul (eps and R)
    const ggml_tensor * wd = nullptr, * wg = nullptr, * wf = nullptr;
    int last = -1;   // the add's node
};

static inline bool ggml_ct_is_view_fork(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static inline bool ggml_ct_unary_fork(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// a row-contiguous f32 [E, T] operand
static inline bool ggml_ct_rows_fork(const ggml_tensor * t, int64_t E, int64_t T) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == E && t->ne[1] == T && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
}

// the loop controller that starts at node i (its rms_norm of cur), or false
static inline bool ggml_ctrl_match_fork_at(const ggml_cgraph * cgraph, int i, ggml_ctrl_match_fork & m) {
    const ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_RMS_NORM) {
        return false;
    }
    // the next 12 non-view nodes from i
    constexpr int N = 12;
    int idx[N];
    int n = 0;
    for (int j = i; j < cgraph->n_nodes && n < N; ++j) {
        if (!ggml_ct_is_view_fork(cgraph->nodes[j])) { idx[n++] = j; }
    }
    if (n < N) {
        return false;
    }
    // the add that ends it is one of them; walk back from it
    const ggml_tensor * add = nullptr;
    int last = -1;
    for (int k = 0; k < N; ++k) {
        const ggml_tensor * t = cgraph->nodes[idx[k]];
        if (t->op == GGML_OP_ADD && t->src[0]->op == GGML_OP_SUB && t->src[1]->op == GGML_OP_MUL) { add = t; last = idx[k]; }
    }
    if (!add) {
        return false;
    }
    const ggml_tensor * sub = add->src[0], * mgm = add->src[1];
    const ggml_tensor * cur = sub->src[0], * mfc = sub->src[1];
    if (mfc->op != GGML_OP_MUL || mfc->src[1] != cur || !ggml_ct_unary_fork(mfc->src[0], GGML_UNARY_OP_TANH) || !ggml_ct_unary_fork(mgm->src[0], GGML_UNARY_OP_TANH)) {
        return false;
    }
    const ggml_tensor * tf = mfc->src[0], * tg = mgm->src[0], * msg = mgm->src[1];
    const ggml_tensor * mmf = tf->src[0], * mmg = tg->src[0];
    if (mmf->op != GGML_OP_MUL_MAT || mmg->op != GGML_OP_MUL_MAT || mmf->src[1] != mmg->src[1]) {
        return false;
    }
    const ggml_tensor * mmd = mmf->src[1];
    if (mmd->op != GGML_OP_MUL_MAT || mmd->src[1]->op != GGML_OP_CONCAT || ggml_get_op_params_i32(mmd->src[1], 0) != 0) {
        return false;
    }
    const ggml_tensor * cat = mmd->src[1];
    const ggml_tensor * rc = cat->src[0], * rm = cat->src[1];
    if (rc->op != GGML_OP_RMS_NORM || rm->op != GGML_OP_RMS_NORM || rc->src[0] != cur || rm->src[0] != msg) {
        return false;
    }
    const ggml_tensor * set[N] = {rc, rm, cat, mmd, mmf, tf, mfc, sub, mmg, tg, mgm, add};
    for (int k = 0; k < N; ++k) {
        bool found = false;
        for (int s = 0; s < N; ++s) { found |= cgraph->nodes[idx[k]] == set[s]; }
        if (!found) { return false; }
    }
    const int64_t E = cur->ne[0], T = cur->ne[1], R = mmd->ne[0];
    const ggml_tensor * wd = mmd->src[0], * wg = mmg->src[0], * wf = mmf->src[0];
    if (T > GGML_CT_MAX_T || R > GGML_CT_MAX_R || !ggml_ct_rows_fork(cur, E, T) || !ggml_ct_rows_fork(msg, E, T) || !ggml_ct_rows_fork(add, E, T) ||
            wd->type != GGML_TYPE_F32 || wd->ne[0] != 2*E || wd->ne[1] != R || !ggml_is_contiguous(wd) ||
            wg->type != GGML_TYPE_F32 || wg->ne[0] != R || wg->ne[1] != E || !ggml_is_contiguous(wg) ||
            wf->type != GGML_TYPE_F32 || wf->ne[0] != R || wf->ne[1] != E || !ggml_is_contiguous(wf)) {
        return false;
    }
    // every skipped node but the add is read only inside the controller
    for (int k = 0; k < N; ++k) {
        const int j = idx[k];
        const ggml_tensor * t = cgraph->nodes[j];
        if (t == add) { continue; }
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
        int uses = 0;
        for (int q = i; q <= last; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) { uses += cgraph->nodes[q]->src[s] == t; }
        }
        if (uses != ggml_node_get_use_count(cgraph, j)) { return false; }
    }
    m = { cur, msg, add, rc, rm, mmd, wd, wg, wf, last };
    return true;
}

// what graph_optimize keeps alive until the add: cur and msg, read by the output kernel while it writes the add
static inline int ggml_ctrl_keep_fork(const ggml_ctrl_match_fork & m, const ggml_tensor * keep[2]) {
    keep[0] = m.cur;
    keep[1] = m.msg;
    return 2;
}
