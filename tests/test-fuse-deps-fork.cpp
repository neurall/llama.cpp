// Unit test of the fork fusions that read inputs of the nodes they skip (ggml/src/ggml-cuda/mix-match-fork.h, ctrl-match-fork.h, fuse-apart-fork.h).
// No GPU. Inkling 2's depth mix with experts on the CPU (2026-11-27, decode PPL 1.60-1.63, different every run): the layer state cur is the
// scheduler's copy of the CPU split's output, whose only reader is the concat, so in the unfused order it dies there and the allocator may lay the
// mix's output (the last matmul) over it while the fused kernel still reads it. The rule: graph_optimize keeps every tensor the fused kernels read
// alive until the group's last node (ggml_mix_keep_fork / ggml_ctrl_keep_fork); the byte check stays as a tripwire.
// build: cmake --build <dir> --target test-fuse-deps-fork;  run: ctest -R test-fuse-deps-fork
#include "../ggml/src/ggml-cuda/mix-match-fork.h"
#include "../ggml/src/ggml-cuda/ctrl-match-fork.h"
#include "../ggml/src/ggml-cuda/fuse-apart-fork.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static bool keeps(const ggml_tensor * const * keep, int n, const ggml_tensor * t) {
    for (int k = 0; k < n; ++k) { if (keep[k] == t) { return true; } }
    return false;
}

static bool overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static int node_index(ggml_cgraph * g, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) { if (ggml_graph_node(g, i) == t) { return i; } }
    return -1;
}

int main() {
    const int64_t E = 64, S0 = 3, Q = 1;   // one decode token: a stack of 3 states, the layer's state, one query
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, 1, 1);
    ggml_tensor * stack = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, S0, 1);
    ggml_tensor * qry   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, Q, 1);
    ggml_tensor * res   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, Q, 1);
    ggml_set_input(in); ggml_set_input(stack); ggml_set_input(qry); ggml_set_input(res);
    ggml_tensor * cur  = ggml_scale(ctx, in, 1.0f);                            // the layer state: read by the concat only
    ggml_tensor * cat  = ggml_concat(ctx, stack, cur, 1);
    ggml_tensor * R    = ggml_rms_norm(ctx, cat, 1e-6f);
    ggml_tensor * mm1  = ggml_mul_mat(ctx, R, qry);
    ggml_tensor * sm   = ggml_soft_max_ext(ctx, mm1, nullptr, 0.125f, 0.0f);
    ggml_tensor * cont = ggml_cont(ctx, ggml_transpose(ctx, R));
    ggml_tensor * mm2  = ggml_mul_mat(ctx, cont, sm);
    ggml_tensor * out  = ggml_add(ctx, mm2, res);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // the matcher finds the mix at the concat, and the keep list holds what the kernels read: stack, cur, query (the old rule kept nothing)
    const int i_cat = node_index(gf, cat);
    ggml_mix_match_fork m;
    CHECK(i_cat >= 0 && ggml_mix_match_fork_at(gf, i_cat, m));
    CHECK(m.mm2 == mm2 && ggml_graph_node(gf, m.last) == mm2);
    // the same expectations against the rule before the fix (graph_optimize kept nothing alive for the mix): it must fail them
    auto keep_rule_fails = [&](int (*rule)(const ggml_mix_match_fork &, const ggml_tensor * k[3])) {
        const ggml_tensor * k[3] = {};
        const int n = rule(m, k);
        return (int) !(n == 3) + (int) !keeps(k, n, stack) + (int) !keeps(k, n, cur) + (int) !keeps(k, n, qry);
    };
    const int fail_new = keep_rule_fails(ggml_mix_keep_fork);
    const int fail_old = keep_rule_fails([](const ggml_mix_match_fork &, const ggml_tensor * k[3]) { (void) k; return 0; });
    printf("mix keep rule: new %d failed expectations, before the fix %d\n", fail_new, fail_old);
    CHECK(fail_new == 0 && fail_old > 0);
    const ggml_tensor * keep[3];
    const int n_keep = ggml_mix_keep_fork(m, keep);
    // the controller's keep list: cur and msg
    {
        ggml_ctrl_match_fork cm;
        cm.cur = cur; cm.msg = stack;
        const ggml_tensor * ck[2];
        CHECK(ggml_ctrl_keep_fork(cm, ck) == 2 && ck[0] == cur && ck[1] == stack);
    }

    // the allocator with the deps as the scheduler adds them (a view node after mm2 whose sources are the kept tensors): mm2 shares no byte with them
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        ggml_tensor * dep = ggml_view_tensor(ctx, const_cast<ggml_tensor *>(keep[0]));
        for (int k = 0; k < n_keep; ++k) { dep->src[k] = const_cast<ggml_tensor *>(keep[k]); }
        ggml_cgraph * g2 = ggml_new_graph(ctx);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_graph_add_node(g2, ggml_graph_node(gf, i));
            if (ggml_graph_node(gf, i) == mm2) { ggml_graph_add_node(g2, dep); }
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_dev_buffer_type(cpu));
        CHECK(ggml_gallocr_alloc_graph(ga, g2));
        CHECK(!overlap(mm2, cur) && !overlap(mm2, stack) && !overlap(mm2, qry));
        CHECK(ggml_cuda_fuse_apart_fork(mm2, { m.sa, m.sb, m.qry }, false));   // the tripwire stays quiet
        ggml_gallocr_free(ga);
    } else {
        printf("no CPU device: allocator part skipped\n");
    }

    // the tripwire's rule: an output on an op-NONE input (a scheduler copy) one row in, or exactly on it, is refused; apart is fine; an in-place
    // element-wise output (same address and layout) is fine where the kernel allows it (the loop controller's add)
    {
        alignas(256) static char buf[1 << 16];
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
        ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
        CHECK(a->op == GGML_OP_NONE);
        a->data = buf;       b->data = buf + 64;   CHECK(!ggml_cuda_fuse_apart_fork(b, { a }, false));
        b->data = buf;                             CHECK(!ggml_cuda_fuse_apart_fork(b, { a }, false));
        CHECK(ggml_cuda_fuse_apart_fork(b, { a }, true));
        c->data = buf;                             CHECK(!ggml_cuda_fuse_apart_fork(b, { c }, true));   // same address, other layout
        b->data = buf + 256;                       CHECK(ggml_cuda_fuse_apart_fork(b, { a }, false));   // touching, no shared byte
        CHECK(ggml_cuda_fuse_apart_fork(b, { nullptr }, false));
    }

    ggml_free(ctx);
    printf("test-fuse-deps-fork: %s (%d failures)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
