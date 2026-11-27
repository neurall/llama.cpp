// The CPU plan of the host steps of a decode layer (ggml/src/ggml-cpu/ggml-cpu.c: ggml_get_n_tasks, ggml_graph_plan).
// A decode layer of the MoE cache has two host steps: the router's picks and the layer input copied to the host (10 ids, 2560 floats), and the chain of the
// experts the card does not hold. Both asked for the whole thread team: a copy counted as many tasks as threads, and so did the chain's products even when every
// pick is on the card and the op only zeroes its output. nsys, Qwen3.8 IQ3_S decode, build b15843, 2026-11-18: 96 such steps a token and 8% of the main
// thread's samples in OpenMP's fork and barrier (about 1.5 ms of a 19.2 ms token).
// A step with nothing to compute plans one thread; a step with an expert to compute keeps the team.
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-op-slots-fork.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

static const int N_IN = 64, N_OUT = 32, N_AS = 6, N_USED = 3, N_SLOTS = 4;

struct chain {
    ggml_context * ctx;
    ggml_cgraph  * gf;
    ggml_tensor  * out;
};

// the host chain's first product: ids pick N_USED of N_AS experts; table[e] == N_SLOTS means expert e is not on the card
static chain make_chain(const std::vector<int32_t> & ids_v, const std::vector<int32_t> & table_v, bool ids_from_this_graph) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    chain c; c.ctx = ggml_init(ip); c.gf = ggml_new_graph(c.ctx);
    ggml_tensor * as    = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, N_IN, N_OUT, N_AS);
    ggml_tensor * x     = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, N_IN, 1, 1);
    ggml_tensor * ids   = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, N_USED, 1);
    ggml_tensor * table = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, N_AS);
    for (int64_t i = 0; i < ggml_nelements(as); ++i) { ((float *) as->data)[i] = 1.0f; }
    for (int64_t i = 0; i < ggml_nelements(x);  ++i) { ((float *) x->data)[i]  = 1.0f; }
    memcpy(ids->data, ids_v.data(), ids_v.size() * sizeof(int32_t));
    memcpy(table->data, table_v.data(), table_v.size() * sizeof(int32_t));
    if (ids_from_this_graph) { ids = ggml_cont(c.ctx, ids); }   // the picks are computed by this very graph: the plan cannot read them
    c.out = ggml_mul_mat_id(c.ctx, as, x, ids);
    c.out->src[3] = table;
    ggml_fork_mmid_set_cache_absent(c.out, N_SLOTS);
    ggml_build_forward_expand(c.gf, c.out);
    return c;
}

int main() {
    const int T = 8;
    {   // the copy of the picks and of the layer input to the host: no team
        ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip); ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_cont(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 10, 1)));
        ggml_build_forward_expand(gf, ggml_cont(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 1, 1)));
        const int n = ggml_graph_plan(gf, T, nullptr).n_threads;
        printf("copies of 10 ids and 2560 floats: %d threads\n", n);
        assert(n == 1);
        ggml_free(ctx);
    }
    {   // a big copy keeps the team
        ggml_init_params ip = { 64u * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip); ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_cont(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1024)));
        const int n = ggml_graph_plan(gf, T, nullptr).n_threads;
        printf("copy of 2560 x 1024 floats: %d threads\n", n);
        assert(n == T);
        ggml_free(ctx);
    }
    const std::vector<int32_t> all_on_card = { 0, 1, 2, 3, 0, 1 }, one_missing = { 0, 1, N_SLOTS, 3, 0, 1 };
    {   // every pick is on the card: the chain only zeroes its output, on one thread
        chain c = make_chain({ 0, 1, 3 }, all_on_card, false);
        ggml_cplan p = ggml_graph_plan(c.gf, T, nullptr);
        printf("chain, every pick on the card: %d threads\n", p.n_threads);
        assert(p.n_threads == 1);
        std::vector<uint8_t> work(p.work_size + 1); p.work_data = work.data();
        for (int64_t i = 0; i < ggml_nelements(c.out); ++i) { ((float *) c.out->data)[i] = 7.0f; }
        assert(ggml_graph_compute(c.gf, &p) == GGML_STATUS_SUCCESS);
        for (int64_t i = 0; i < ggml_nelements(c.out); ++i) { assert(((float *) c.out->data)[i] == 0.0f); }
        ggml_free(c.ctx);
    }
    {   // negative ids (deferred, skipped) are no host work either
        chain c = make_chain({ -1, 1, -1 }, one_missing, false);
        assert(ggml_graph_plan(c.gf, T, nullptr).n_threads == 1);
        ggml_free(c.ctx);
    }
    {   // one pick is not on the card: the chain computes it, with the team
        chain c = make_chain({ 0, 2, 3 }, one_missing, false);
        ggml_cplan p = ggml_graph_plan(c.gf, T, nullptr);
        printf("chain, one pick missing: %d threads\n", p.n_threads);
        assert(p.n_threads == T);
        std::vector<uint8_t> work(p.work_size + 1); p.work_data = work.data();
        assert(ggml_graph_compute(c.gf, &p) == GGML_STATUS_SUCCESS);
        const float * o = (const float *) c.out->data;   // [N_OUT, N_USED, 1]: the row of the missing expert is computed, the others are zero
        assert(o[0] == 0.0f && o[N_OUT] == (float) N_IN && o[2 * N_OUT] == 0.0f);
        ggml_free(c.ctx);
    }
    {   // the picks come out of this graph: unknown when the plan is made, the team stays
        chain c = make_chain({ 0, 1, 3 }, all_on_card, true);
        const int n = ggml_graph_plan(c.gf, T, nullptr).n_threads;
        printf("chain, picks computed by the same graph: %d threads\n", n);
        assert(n == T);
        ggml_free(c.ctx);
    }
    printf("ok\n");
    return 0;
}
