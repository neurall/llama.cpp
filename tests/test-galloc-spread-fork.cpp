// fork: frees held back in the graphs a backend runs on several streams (ggml-alloc.c, ggml_gallocr_set_spread_fork).
// Stock hands a freed buffer to the next node that fits: branches of a layer that do not need each other's results come to share memory, and
// a plan that keeps launches on the same memory in graph order (ggml-streams-fork.h) makes them wait for each other.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

extern "C" { GGML_API void ggml_gallocr_set_spread_fork(int n_nodes, size_t max_node_bytes, bool host_too); }

// `branches` chains off one input that do not depend on each other (sqr, sqrt and scale share a buffer in place, sum ends the chain), their sums added up
static ggml_cgraph * build(ggml_context * ctx, int n, int branches) {
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_tensor * acc = nullptr;
    for (int b = 0; b < branches; ++b) {
        ggml_tensor * s = ggml_sum(ctx, ggml_scale(ctx, ggml_sqrt(ctx, ggml_sqr(ctx, x)), 1.0f + (float) b));
        acc = acc ? ggml_add(ctx, acc, s) : s;
    }
    ggml_set_output(acc);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(gf, acc);
    return gf;
}

// pairs of nodes on the same memory of which the later one does not descend from the earlier one: no data dependency, a wait all the same
static int false_pairs(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);
    std::vector<std::set<const ggml_tensor *>> anc(n);
    int bad = 0;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = ggml_graph_node(gf, j);
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = t->src[k];
            if (!s) { continue; }
            anc[j].insert(s);
            for (int i = 0; i < j; ++i) {
                if (ggml_graph_node(gf, i) == s) { anc[j].insert(anc[i].begin(), anc[i].end()); }
            }
        }
        for (int i = 0; i < j; ++i) {
            const ggml_tensor * u = ggml_graph_node(gf, i);
            if (t->view_src || u->view_src || !t->data || !u->data) { continue; }
            const char * a0 = (const char *) u->data, * a1 = a0 + ggml_nbytes(u);
            const char * b0 = (const char *) t->data, * b1 = b0 + ggml_nbytes(t);
            if (a0 < b1 && b0 < a1 && !anc[j].count(u)) { ++bad; }
        }
    }
    return bad;
}

// reserve a graph of n_first elements and first_branches branches (another shape than the second: a graph that fits the plan of the same shape is not
// planned again), then place the 10-branch graph of 256 elements; its false pairs
static int run(int n_first, int first_branches) {
    ggml_init_params ip = { 16u << 20, nullptr, true };
    ggml_context * c1 = ggml_init(ip), * c2 = ggml_init(ip);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    assert(ggml_gallocr_reserve(ga, build(c1, n_first, first_branches)));
    ggml_cgraph * gf = build(c2, 256, 10);
    assert(ggml_graph_n_nodes(gf) >= 32);
    assert(ggml_gallocr_alloc_graph(ga, gf));
    const int bad = false_pairs(gf);
    ggml_gallocr_free(ga);
    ggml_free(c1); ggml_free(c2);
    return bad;
}

int main() {
    const int stock = run(100000, 12);
    assert(stock > 0);   // stock: the second branch takes the first one's buffer
    ggml_gallocr_set_spread_fork(1024, (size_t) 1 << 20, true);
    const int spread = run(100000, 12);
    assert(spread == 0);   // the buffer the bigger graph left holds every branch apart
    const int tight = run(256, 1);
    assert(tight > 0);   // a buffer that holds one branch only: the plan as stock makes it, nothing grows for the spread
    ggml_gallocr_set_spread_fork(0, 0, false);
    assert(run(100000, 12) == stock);
    printf("ok: pairs of independent nodes on the same memory: stock %d, frees held back %d, buffer too small %d\n", stock, spread, tight);
    return 0;
}
