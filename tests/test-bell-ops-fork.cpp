// The decode bell's two ops on real backends (ggml_moe_bell, ggml_moe_rows; ggml/src/ggml-bell-fork.h).
// A graph of 48 "layers", each: the bell (picks -> blocks, published to the host), the card's own rows, then the rows of the missing picks from the host.
// The graph is launched once per token and this thread serves the bells while it runs, as the engine will. Run on the CPU backend and on every GPU found:
// the blocks, the published picks and the rows must be the same everywhere, and a token with nothing missing must finish with nobody serving.
// Then the engine's way (run_sched): the graph goes through the scheduler and the scheduler has each split's bells served (ggml_backend_set_split_bell_callback),
// with a card's token as long as a real one.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-bell-fork.h"

#undef NDEBUG
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

static const int N_EMBD = 2560, N_USED = 10, N_EXPERT = 512, N_LAYER = 48;

static float row_value(int layer, int id, float x0) { return 1000.0f * layer + id + x0; }

// the host's step for one layer of token t: waits for the picks, writes the rows of the missing ones. false: the layer never rang
static bool serve(ggml_bell * b, uint32_t & seen, int l, int t, long & n_host, long & bad) {
    if (!ggml_bell_host_wait(b, seen, 10 * 1000 * 1000)) { return false; }
    seen = b->seq;
    int32_t todo[N_USED];
    const int n = ggml_bell_host_misses(b, todo);
    n_host += n;
    if (n == 0) { return true; }
    bad += ggml_bell_x(b)[0] != (float) t;   // the layer input came with the picks
    for (int i = 0; i < N_USED; ++i) {
        if (todo[i] >= 0) { for (int j = 0; j < N_EMBD; ++j) { ggml_bell_rows(b)[i * N_EMBD + j] = row_value(l, todo[i], (float) t); } }
    }
    ggml_bell_host_done(b, seen);
    return true;
}

static bool run(ggml_backend_dev_t dev) {
    const char * name = ggml_backend_dev_name(dev);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto b_alloc = reg ? (void * (*)(ggml_backend_dev_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_alloc") : nullptr;
    auto b_free  = reg ? (void (*)(void *))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_free")  : nullptr;
    const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!cpu && !b_alloc) { printf("%-10s no bell memory on this backend: skipped\n", name); return true; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    assert(be);
    std::vector<ggml_bell *> bells;
    for (int l = 0; l < N_LAYER; ++l) {
        void * m = cpu ? calloc(1, ggml_bell_size(N_USED, N_EMBD)) : b_alloc(dev, ggml_bell_size(N_USED, N_EMBD));
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
            std::thread host([&] {   // a thread of its own, started before the launch (the CPU backend's compute call returns only at the end)
                for (int l = 0; l < N_LAYER; ++l) {
                    if (!serve(bells[l], seen[l], l, t, n_host, bad)) {
                        rang = false;
                        for (int k = 0; k < N_LAYER; ++k) { ggml_bell_host_done(bells[k], bells[k]->seq); }   // let a waiting card go
                        return;
                    }
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

// The engine's way. A layer's own work is N_FILL kernels here, so a card's token is some 3000 launches. A CUDA stream queues 1022 calls behind a running kernel,
// the next call blocks until that kernel ends (fork/bench/launchblock.cu). Served by the launching thread after the launch, the first token with a missing pick
// never came back: a graph's first call is launched kernel by kernel (its CUDA graph is captured from the second call on), and the launch stood in cuLaunchKernel
// behind a rows kernel that waited for the host (Qwen3.8 UD-IQ4_XS, 25 layers and about 1500 kernels a card, 2026-11-18; OLMoE's 8 layers fitted the queue).
static const int N_FILL = 60;

struct sched_host {
    std::vector<ggml_bell *> bells;
    std::vector<uint32_t>    seen;
    int  t = 0;
    long n_host = 0, bad = 0;
    bool rang = true;
};

static void serve_split(ggml_backend_t, struct ggml_cgraph * g, void * ud) {
    sched_host & h = *(sched_host *) ud;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        const ggml_tensor * n = ggml_graph_node(g, i);
        if (n->op != GGML_OP_MOE_BELL) { continue; }
        ggml_bell * b; memcpy(&b, n->op_params, sizeof(b));
        const size_t l = (size_t) (std::find(h.bells.begin(), h.bells.end(), b) - h.bells.begin());
        assert(l < h.bells.size());
        if (!serve(b, h.seen[l], (int) l, h.t, h.n_host, h.bad)) {
            h.rang = false;
            for (auto * k : h.bells) { ggml_bell_host_done(k, k->seq); }
            return;
        }
    }
}

static std::atomic<int64_t> g_deadline_ms{0};   // a token under way has to be back by then (0: none)
static const char *         g_dev_name = "";
static int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool run_sched(ggml_backend_dev_t dev) {
    const char * name = ggml_backend_dev_name(dev);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto b_alloc = reg ? (void * (*)(ggml_backend_dev_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_alloc") : nullptr;
    auto b_free  = reg ? (void (*)(void *))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_bell_free")  : nullptr;
    const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!cpu && !b_alloc) { return true; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t be_cpu = cpu ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(be && (cpu || be_cpu));
    ggml_backend_t bes[2] = { be, be_cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, cpu ? 1 : 2, 8192, false, true);
    sched_host h;
    for (int l = 0; l < N_LAYER; ++l) {
        void * m = cpu ? calloc(1, ggml_bell_size(N_USED, N_EMBD)) : b_alloc(dev, ggml_bell_size(N_USED, N_EMBD));
        assert(m);
        h.bells.push_back(ggml_bell_init(m, N_USED, N_EMBD));
    }
    h.seen.assign(N_LAYER, 0);
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (N_LAYER * (N_FILL + 8) + 8) + ggml_graph_overhead_custom(8192, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1); ggml_set_input(x);
    std::vector<ggml_tensor *> ids(N_LAYER), table(N_LAYER), own(N_LAYER), slots(N_LAYER), out(N_LAYER);
    for (int l = 0; l < N_LAYER; ++l) {
        ids[l]   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);       ggml_set_input(ids[l]);
        table[l] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_EXPERT);        ggml_set_input(table[l]);
        own[l]   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, N_USED, 1); ggml_set_input(own[l]);
        slots[l] = ggml_moe_bell(ctx, ids[l], table[l], x, h.bells[l]);
        ggml_tensor * cur = own[l];
        for (int f = 0; f < N_FILL; ++f) { cur = ggml_scale(ctx, cur, 1.0f); }
        out[l]   = ggml_moe_rows(ctx, cur, ids[l], slots[l], h.bells[l]);
        ggml_set_output(slots[l]); ggml_set_output(out[l]);
        ggml_build_forward_expand(gf, out[l]);
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) { ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(gf, i), be); }   // inputs alone would leave it all on the CPU
    assert(ggml_backend_sched_alloc_graph(sched, gf));
    ggml_backend_set_split_bell_callback(serve_split, &h);

    std::mt19937 rng(12);
    std::vector<float> ownv(N_EMBD * N_USED, -2.0f), xv(N_EMBD, 0.0f), got(N_EMBD * N_USED);
    std::vector<std::vector<int32_t>> idv(N_LAYER, std::vector<int32_t>(N_USED)), tbv(N_LAYER, std::vector<int32_t>(N_EXPERT));
    long n_miss = 0, bad = 0;
    double us_all_hit = 0, us_miss = 0;
    const int n_tokens = 30;
    g_dev_name = name;
    for (int t = 0; t < n_tokens; ++t) {
        const bool all_hit = t % 3 == 2;   // the first token has misses: it is the one launched kernel by kernel
        h.t = t;
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
        g_deadline_ms = now_ms() + 20 * 1000;
        assert(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
        g_deadline_ms = 0;
        (all_hit ? us_all_hit : us_miss) += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (!h.rang) { printf("%-10s scheduler: a layer never rang: FAILED\n", name); return false; }
        for (int l = 0; l < N_LAYER; ++l) {
            assert(h.seen[l] == h.bells[l]->seq);   // every bell of the token was served
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
    bad += h.bad;
    printf("%-10s scheduler, %d tokens x %d layers x %d kernels: %ld picks missing, %ld computed by the host, %ld wrong values; a token: %.0f us with every pick on the card, %.0f us with misses\n",
           name, n_tokens, N_LAYER, N_FILL + 2, n_miss, h.n_host, bad, us_all_hit / (n_tokens / 3), us_miss / (n_tokens - n_tokens / 3));
    ggml_backend_set_split_bell_callback(nullptr, nullptr);
    ggml_backend_sched_free(sched); ggml_free(ctx); ggml_backend_free(be);
    if (be_cpu) { ggml_backend_free(be_cpu); }
    for (auto * b : h.bells) { if (cpu) { free(b); } else { b_free(b); } }
    return bad == 0 && n_miss == h.n_host && n_miss > 0;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // a line is out before a hang or an assert
    ggml_backend_load_all();
    bool ok = true;
    std::thread([] {   // a token that hangs is a failure, not a hung test
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const int64_t d = g_deadline_ms.load();
            if (d != 0 && now_ms() > d) {
                printf("%-10s scheduler: a token did not come back in 20 s (its launch stands behind a kernel that waits for the host): FAILED\n", g_dev_name);
                fflush(stdout);
                _exit(1);
            }
        }
    }).detach();
    // every card is brought up before any bell is made, as in the engine. Made while another card was current, a bell was out of the second card's range
    // (Xid 13); this test missed it at first because it made that card's bells before the card had a context
    std::vector<ggml_backend_t> up;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { up.push_back(ggml_backend_dev_init(ggml_backend_dev_get(i), nullptr)); }
    for (size_t i = ggml_backend_dev_count(); i-- > 0; ) { ok = run(ggml_backend_dev_get(i)) && ok; }   // and the last card first
    for (size_t i = ggml_backend_dev_count(); i-- > 0; ) { ok = run_sched(ggml_backend_dev_get(i)) && ok; }
    for (auto * b : up) { if (b) { ggml_backend_free(b); } }
    printf(ok ? "ok\n" : "FAILED\n");
    return ok ? 0 : 1;
}
