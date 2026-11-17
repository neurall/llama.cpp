// The engine's bandwidth probes against the most this machine reaches on the same path, measured here another way. Every rate the planner calculates
// with comes from these probes, so a probe that times a slower path than the real traffic prices every upload or read wrong. A probe outside 90-110% of
// its reference fails.
//   PCIe upload, per GPU: the moe cache's link probe (llama_moe_link_gbs_fork) and the GPU probe (llama_gpu_probe_fork, alone and with every GPU of its
//     backend at once) against the expert stream's path: async uploads from the card's pinned host buffer type, back to back, one wait
//     (ggml_backend_tensor_set_async + ggml_backend_synchronize, as fork/iobench/uploadbench.cpp); the cards at once start together on a barrier
//   VRAM, per GPU: the memory-speed probe (llama_moe_vram_probe_gbs_fork) against the faster of a ggml copy kernel and the backend's own
//     device-to-device copy, 256 MiB, bytes read plus written
//   RAM read: the machine probe (llama_moe_ram_probe_gbs_fork) against a sequential read of 1 GiB by 1/4, 1/2, 3/4 and all hardware threads, started
//     together on a barrier, the fastest
// PC1 2026-11-17 (2x RTX 3090, PCIe 4.0 x16): the Vulkan link probe read 15.0-15.7 GB/s (a synchronous write into BAR-mapped VRAM is a CPU copy), the
// stream 26.7 on both cards and both queues; CUDA 26.7 both ways.
#include "ggml.h"
#include "ggml-backend.h"
#include "../src/llama-gpu-probe-fork.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

double llama_moe_link_gbs_fork(ggml_backend_dev_t d);         // src/llama-moecache-fork.cpp
double llama_moe_vram_probe_gbs_fork(ggml_backend_dev_t d);
double llama_moe_ram_probe_gbs_fork(void);

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int g_fails = 0;
static void check(const std::string & who, const char * what, double probe, double ref) {
    const bool ok = ref > 0.0 && probe >= 0.9 * ref && probe <= 1.1 * ref;
    printf("%-8s %-30s probe %7.2f GB/s, reference %7.2f GB/s (%4.0f%%): %s\n", who.c_str(), what, probe, ref, ref > 0.0 ? 100.0 * probe / ref : 0.0,
           ok ? "ok" : "FAIL");
    g_fails += ok ? 0 : 1;
}

constexpr size_t UP_BYTES = 64u << 20;

struct card {
    ggml_backend_dev_t    dev  = nullptr;
    ggml_backend_t        be   = nullptr;
    ggml_backend_buffer_t host = nullptr, vram = nullptr;
    ggml_context *        ctx  = nullptr;
    ggml_tensor *         t    = nullptr;
    const char *          src  = nullptr;
    bool init(ggml_backend_dev_t d) {
        dev = d;
        be = ggml_backend_dev_init(d, nullptr);
        ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(d);
        host = hb ? ggml_backend_buft_alloc_buffer(hb, UP_BYTES) : nullptr;
        vram = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d), UP_BYTES);
        if (!be || !host || !vram) {
            return false;
        }
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) UP_BYTES);
        ggml_backend_tensor_alloc(vram, t, ggml_backend_buffer_get_base(vram));
        src = (const char *) ggml_backend_buffer_get_base(host);
        return true;
    }
    ~card() {
        if (ctx) { ggml_free(ctx); }
        if (host) { ggml_backend_buffer_free(host); }
        if (vram) { ggml_backend_buffer_free(vram); }
        if (be) { ggml_backend_free(be); }
    }
};

// `copies` async uploads of 64 MiB back to back, one wait: GB/s
static double upload(card & c, int copies) {
    const double t0 = now_s();
    for (int k = 0; k < copies; ++k) {
        ggml_backend_tensor_set_async(c.be, c.t, c.src, 0, UP_BYTES);
    }
    ggml_backend_synchronize(c.be);
    return (double) copies * UP_BYTES / (now_s() - t0) / 1e9;
}

static double upload_alone(card & c) {
    upload(c, 2);   // the first use of the host buffer by this card
    double best = 0.0;
    for (int r = 0; r < 3; ++r) {
        best = std::max(best, upload(c, 8));
    }
    return best;
}

// every card of `cards` uploading at once, a thread each, started on a barrier: each card's rate
static std::vector<double> upload_together(std::vector<card *> & cards) {
    std::vector<double> rate(cards.size(), 0.0);
    std::atomic<size_t> ready{0};
    std::atomic<bool>   go{false};
    std::vector<std::thread> th;
    for (size_t i = 0; i < cards.size(); ++i) {
        th.emplace_back([&, i] {
            upload(*cards[i], 1);
            ready.fetch_add(1);
            while (!go.load()) { std::this_thread::yield(); }
            rate[i] = upload(*cards[i], 16);
        });
    }
    while (ready.load() < cards.size()) { std::this_thread::yield(); }
    go.store(true);
    for (auto & t : th) { t.join(); }
    return rate;
}

// the faster of a ggml copy kernel and the backend's device-to-device copy, 256 MiB each way: GB/s of bytes read plus written
static double vram_reference(ggml_backend_dev_t d, ggml_backend_t be) {
    const int64_t row = 16384, rows = (256ll << 20) / (row * (int64_t) sizeof(float));
    ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, rows);
    ggml_tensor * y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row, rows);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, x, y));
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(d));
    double kernel = 0.0, copy = 0.0;
    const double bytes = 2.0 * (double) ggml_nbytes(x);
    for (int r = 0; b && r < 6; ++r) {   // the first round of each warms up
        double t0 = now_s();
        ggml_backend_graph_compute(be, gf);
        const double tk = now_s() - t0;
        t0 = now_s();
        ggml_backend_tensor_copy(x, y);
        const double tc = now_s() - t0;
        if (r > 0) {
            kernel = std::max(kernel, bytes / tk / 1e9);
            copy   = std::max(copy, bytes / tc / 1e9);
        }
    }
    if (b) { ggml_backend_buffer_free(b); }
    ggml_free(ctx);
    return std::max(kernel, copy);
}

// a sequential read of 1 GiB by 1/4, 1/2, 3/4 and all hardware threads, started together on a barrier: the fastest
static double ram_reference() {
    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    std::vector<uint64_t> buf(((size_t) 1 << 30) / 8, 1);
    double best = 0.0;
    for (unsigned nt : { hw / 4, hw / 2, hw * 3 / 4, hw }) {
        nt = std::max(1u, nt);
        for (int pass = 0; pass < 3; ++pass) {
            std::atomic<unsigned> ready{0};
            std::atomic<bool> go{false};
            std::atomic<uint64_t> sink{0};
            std::vector<std::thread> th;
            for (unsigned i = 0; i < nt; ++i) {
                th.emplace_back([&, i] {
                    const size_t a = buf.size() / nt * i, b = i + 1 == nt ? buf.size() : buf.size() / nt * (i + 1);
                    ready.fetch_add(1);
                    while (!go.load(std::memory_order_acquire)) {}
                    uint64_t x = 0;
                    for (size_t k = a; k < b; ++k) { x += buf[k]; }
                    sink += x;
                });
            }
            while (ready.load() < nt) {}
            const double t0 = now_s();
            go.store(true, std::memory_order_release);
            for (auto & t : th) { t.join(); }
            best = std::max(best, (double) (buf.size() * 8) / (now_s() - t0) / 1e9);
        }
    }
    return best;
}

int main() {
    ggml_backend_load_all();
    check("RAM", "read", llama_moe_ram_probe_gbs_fork(), ram_reference());

    std::vector<std::vector<ggml_backend_dev_t>> by_reg;   // the GPUs of each backend
    std::vector<ggml_backend_reg_t> regs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t rg = ggml_backend_dev_backend_reg(d);
        size_t r = 0;
        while (r < regs.size() && regs[r] != rg) { ++r; }
        if (r == regs.size()) { regs.push_back(rg); by_reg.emplace_back(); }
        by_reg[r].push_back(d);
    }
    int gpus = 0;
    for (const auto & devs : by_reg) {
        std::vector<card> cards(devs.size());
        std::vector<card *> ok;
        std::vector<double> alone;
        for (size_t i = 0; i < devs.size(); ++i) {
            if (!cards[i].init(devs[i])) {
                printf("%-8s no backend or no pinned host buffer: skipped\n", ggml_backend_dev_name(devs[i]));
                continue;
            }
            ok.push_back(&cards[i]);
            alone.push_back(upload_alone(cards[i]));
        }
        const std::vector<double> together = ok.size() > 1 ? upload_together(ok) : alone;
        std::vector<ggml_backend_dev_t> probe_devs;
        for (card * c : ok) { probe_devs.push_back(c->dev); }
        const std::vector<llama_gpu_probe_fork_result> probe = llama_gpu_probe_fork(probe_devs);
        for (size_t i = 0; i < ok.size(); ++i) {
            const std::string nm = ggml_backend_dev_name(ok[i]->dev);
            check(nm, "PCIe upload, link probe", llama_moe_link_gbs_fork(ok[i]->dev), alone[i]);
            check(nm, "PCIe upload, GPU probe", probe[i].h2d_gbs, alone[i]);
            if (ok.size() > 1) {
                check(nm, "PCIe upload, all cards at once", probe[i].h2d_conc_gbs, together[i]);
            }
            check(nm, "VRAM read + write", llama_moe_vram_probe_gbs_fork(ok[i]->dev), vram_reference(ok[i]->dev, ok[i]->be));
            ++gpus;
        }
    }
    printf("%d GPUs, %d checks failed\n", gpus, g_fails);
    return g_fails ? 1 : 0;
}
