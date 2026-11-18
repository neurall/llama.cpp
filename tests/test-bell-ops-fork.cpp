// The decode bell's two ops on real backends (ggml_moe_bell, ggml_moe_rows; ggml/src/ggml-bell-fork.h).
// A graph of 48 "layers", each: the bell (picks -> blocks, published to the host), the card's own rows, then the rows of the missing picks from the host.
// The graph is launched once per token and this thread serves the bells while it runs, as the engine will. Run on the CPU backend and on every GPU found:
// the blocks, the published picks and the rows must be the same everywhere, and a token with nothing missing must finish with nobody serving.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-bell-fork.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const int N_EMBD = 2560, N_USED = 10, N_EXPERT = 512, N_LAYER = 48;

static float row_value(int layer, int id, float x0) { return 1000.0f * layer + id + x0; }

static bool run(ggml_backend_dev_t dev) {
    const char * name = ggml_backend_dev_name(dev);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto b_alloc = reg ? (void * (*)(size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_alloc") : nullptr;
    auto b_free  = reg ? (void (*)(void *))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_free")  : nullptr;
    const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!cpu && !b_alloc) { printf("%-10s no bell memory on this backend: skipped\n", name); return true; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    assert(be);
    std::vector<ggml_bell *> bells;
    for (int l = 0; l < N_LAYER; ++l) {
        void * m = cpu ? calloc(1, ggml_bell_size(N_USED, N_EMBD)) : b_alloc(ggml_bell_size(N_USED, N_EMBD));
        assert(m);
        bells.push_back(ggml_bell_init(m, N_USED, N_EMBD));
    }
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (N_LAYER * 8 + 8) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1); ggml_set_input(x);
    std::vector<ggml_tensor *> ids(N_LAYER), table(N_LAYER), own(N_LAYER), slots(N_LAYER), out(N_LAYER);
    for (int l = 0; l < N_LAYER; ++l) {
        ids[l]   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);       ggml_set_input(ids[l]);
        table[l] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_EXPERT);        ggml_set_input(table[l]);
        own[l]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, N_USED, 1); ggml_set_input(own[l]);
        slots[l] = ggml_moe_bell(ctx, ids[l], table[l], x, bells[l]);
        out[l]   = ggml_moe_rows(ctx, ggml_scale(ctx, own[l], 1.0f), ids[l], slots[l], bells[l]);   // the scale stands for the card's expert chain
        ggml_set_output(slots[l]); ggml_set_output(out[l]);
        ggml_build_forward_expand(gf, out[l]);
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) { printf("%-10s does not take %s: FAILED\n", name, ggml_op_name(ggml_graph_node(gf, i)->op)); return false; }
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    assert(ggml_gallocr_alloc_graph(ga, gf));

    std::mt19937 rng(11);
    std::vector<float> ownv(N_EMBD * N_USED, -2.0f), xv(N_EMBD, 0.0f), got(N_EMBD * N_USED);
    std::vector<std::vector<int32_t>> idv(N_LAYER, std::vector<int32_t>(N_USED)), tbv(N_LAYER, std::vector<int32_t>(N_EXPERT));
    std::vector<uint32_t> seen(N_LAYER, 0);
    long n_miss = 0, n_host = 0, bad = 0;
    double us_all_hit = 0, us_miss = 0;
    const int n_tokens = 60;
    for (int t = 0; t < n_tokens; ++t) {
        const bool all_hit = t % 3 == 0;   // every third token has every pick on the card
        xv[0] = (float) t;
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        for (int l = 0; l < N_LAYER; ++l) {
            for (int e = 0; e < N_EXPERT; ++e) { tbv[l][e] = !all_hit && rng() % 100 < 3 ? -1 : (int32_t) (e % 300); }
            for (int i = 0; i < N_USED; ++i) { idv[l][i] = i == 7 && t % 5 == 0 ? -1 : (int32_t) (rng() % N_EXPERT); }
            ggml_backend_tensor_set(ids[l], idv[l].data(), 0, ggml_nbytes(ids[l]));
            ggml_backend_tensor_set(table[l], tbv[l].data(), 0, ggml_nbytes(table[l]));
            ggml_backend_tensor_set(own[l], ownv.data(), 0, ggml_nbytes(own[l]));
        }
        ggml_backend_synchronize(be);
        const auto t0 = std::chrono::steady_clock::now();
        if (all_hit) {   // nobody serves: the graph must come back on its own
            assert(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(be);
            for (int l = 0; l < N_LAYER; ++l) { assert(bells[l]->seq != seen[l] && bells[l]->n_miss == 0); seen[l] = bells[l]->seq; }
        } else {
            bool rang = true;
            std::thread host([&] {   // the engine serves from the thread that launched the graph; a thread of its own also serves the CPU backend, whose compute call returns only at the end
                for (int l = 0; l < N_LAYER; ++l) {
                    ggml_bell * b = bells[l];
                    if (!ggml_bell_host_wait(b, seen[l], 10 * 1000 * 1000)) {
                        rang = false;
                        for (int k = 0; k < N_LAYER; ++k) { ggml_bell_host_done(bells[k], bells[k]->seq); }   // let a waiting card go
                        return;
                    }
                    seen[l] = b->seq;
                    int32_t todo[N_USED];
                    const int n = ggml_bell_host_misses(b, todo);
                    n_host += n;
                    if (n == 0) { continue; }
                    bad += ggml_bell_x(b)[0] != (float) t;   // the layer input came with the picks
                    for (int i = 0; i < N_USED; ++i) {
                        if (todo[i] >= 0) { for (int j = 0; j < N_EMBD; ++j) { ggml_bell_rows(b)[i * N_EMBD + j] = row_value(l, todo[i], (float) t); } }
                    }
                    ggml_bell_host_done(b, seen[l]);
                }
            });
            assert(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
            host.join();
            if (!rang) { printf("%-10s a layer never rang: FAILED\n", name); return false; }
            ggml_backend_synchronize(be);
        }
        (all_hit ? us_all_hit : us_miss) += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        for (int l = 0; l < N_LAYER; ++l) {
            std::vector<int32_t> sv(N_USED);
            ggml_backend_tensor_get(slots[l], sv.data(), 0, ggml_nbytes(slots[l]));
            ggml_backend_tensor_get(out[l], got.data(), 0, ggml_nbytes(out[l]));
            for (int i = 0; i < N_USED; ++i) {
                const int32_t id = idv[l][i], s = id >= 0 && tbv[l][id] >= 0 ? tbv[l][id] : -1;
                const bool miss = id >= 0 && s < 0;
                n_miss += miss;
                bad += sv[i] != s;
                const float want = miss ? row_value(l, id, (float) t) : -2.0f;
                for (int j = 0; j < N_EMBD; ++j) { bad += got[i * N_EMBD + j] != want; }
            }
        }
    }
    printf("%-10s %d tokens x %d layers: %ld picks missing, %ld computed by the host, %ld wrong values; a token: %.0f us with every pick on the card, %.0f us with misses\n",
           name, n_tokens, N_LAYER, n_miss, n_host, bad, us_all_hit / (n_tokens / 3), us_miss / (n_tokens - n_tokens / 3));
    ggml_gallocr_free(ga); ggml_free(ctx); ggml_backend_free(be);
    for (auto * b : bells) { if (cpu) { free(b); } else { b_free(b); } }
    return bad == 0 && n_miss == n_host && n_miss > 0;
}

int main() {
    ggml_backend_load_all();
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { ok = run(ggml_backend_dev_get(i)) && ok; }
    printf(ok ? "ok\n" : "FAILED\n");
    return ok ? 0 : 1;
}
