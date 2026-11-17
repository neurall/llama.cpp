// uploadbench DIR [MiB]: host -> card uploads through the ggml backend API, ggml_backend_tensor_set_async (what the expert stream calls), on every GPU of every
// backend in DIR (a build's libggml-cuda.so, libggml-vulkan.so), from a backend's own pinned host buffer (ggml_backend_dev_host_buffer_type) into a card buffer.
// Per card: the first upload from a fresh host buffer (Vulkan imports host memory into each card at its first use), then warm copies of several sizes issued
// back to back (the stream's expert slices are 0.3-10 MB); then the other card's host buffer (the RAM pool is allocated from one card's host buffer type and
// both cards read it), then both cards of a backend at once (a thread each). GB/s = bytes / (first issue to synchronize); issue us = the host time per call.
// Env as the engine sees it: GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1 puts Vulkan's uploads on the transfer queue.
//
// build: g++ -O2 -std=c++17 -pthread -I ggml/include fork/iobench/uploadbench.cpp -L DIR -lggml -lggml-base -Wl,-rpath,DIR -o uploadbench
// run (PC1, the Vulkan cells' driver): LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 LD_LIBRARY_PATH=DIR:/p/bw/p2p/userspace-595-gl/usr/lib/x86_64-linux-gnu
//   VK_ICD_FILENAMES=/p/bw/p2p/userspace-595-gl/nvidia_icd_595.json ./uploadbench DIR   (without the preload the drivers' LLVM crashes in call_once:
//   the build links its C++ runtime statically)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct card {
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t be = nullptr;
    ggml_backend_buffer_t host = nullptr, vram = nullptr;
    ggml_context * ctx = nullptr;
    ggml_tensor * t = nullptr;
    char * src = nullptr;
    size_t bytes = 0;
};

// the host buffer is allocated after every card's backend exists, as the engine's RAM pool is (a Vulkan card imports host memory made before its device at
// its first use); alloc_ms: the allocation's time (with the fork's import into every card at allocation it carries that cost)
static double g_alloc_ms = 0.0;
static bool card_init(card & c, size_t bytes) {
    const ggml_backend_dev_t dev = c.dev;
    c.bytes = bytes;
    ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(dev);
    if (!c.be || !hb) { fprintf(stderr, "%s: no backend or no host buffer type\n", ggml_backend_dev_name(dev)); return false; }
    const double t0 = now_s();
    c.host = ggml_backend_buft_alloc_buffer(hb, bytes);
    g_alloc_ms = (now_s() - t0) * 1e3;
    c.vram = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), bytes);
    if (!c.host || !c.vram) { fprintf(stderr, "%s: allocation failed\n", ggml_backend_dev_name(dev)); return false; }
    c.src = (char *) ggml_backend_buffer_get_base(c.host);
    for (size_t i = 0; i < bytes; i += 4096) { c.src[i] = (char) (i >> 12); }   // every page touched: no first-touch fault inside a timing
    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    c.ctx = ggml_init(ip);
    c.t = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) bytes);
    return ggml_backend_tensor_alloc(c.vram, c.t, ggml_backend_buffer_get_base(c.vram)) == GGML_STATUS_SUCCESS;
}

struct res { double gbs, issue_us; };

// copies of `chunk` bytes from `src` until `total` bytes, then wait
static res run(card & c, const char * src, size_t chunk, size_t total) {
    const size_t n = total / chunk;
    const double t0 = now_s();
    size_t off = 0;
    for (size_t i = 0; i < n; ++i) {
        if (off + chunk > c.bytes) { off = 0; }
        ggml_backend_tensor_set_async(c.be, c.t, src + off, off, chunk);
        off += chunk;
    }
    const double t1 = now_s();
    ggml_backend_synchronize(c.be);
    const double t2 = now_s();
    return { (double) (n * chunk) / (t2 - t0) / 1e9, (t1 - t0) / (double) n * 1e6 };
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BUILD_DIR [MiB per card buffer, 1024]\n", argv[0]); return 1; }
    ggml_backend_load_all_from_path(argv[1]);
    const size_t bytes = (size_t) (argc > 2 ? atoi(argv[2]) : 1024) << 20;
    std::vector<std::vector<card>> by_reg;   // the GPUs of each backend
    std::vector<std::string> reg_name;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        const std::string rn = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
        size_t r = 0;
        while (r < reg_name.size() && reg_name[r] != rn) { ++r; }
        if (r == reg_name.size()) { reg_name.push_back(rn); by_reg.emplace_back(); }
        by_reg[r].emplace_back();
        by_reg[r].back().dev = d;
        by_reg[r].back().be = ggml_backend_dev_init(d, nullptr);
    }
    for (auto & cards : by_reg) {
        for (auto & c : cards) {
            if (!card_init(c, bytes)) { return 1; }
            printf("%-10s host buffer of %zu MiB allocated in %.1f ms\n", ggml_backend_dev_name(c.dev), bytes >> 20, g_alloc_ms);
        }
    }
    printf("%-10s %-28s %10s %9s %9s\n", "card", "test", "chunk KiB", "GB/s", "issue us");
    const size_t chunks[] = { 256u << 20, 16u << 20, 4u << 20, 1u << 20, 256u << 10 };
    for (auto & cards : by_reg) {
        for (auto & c : cards) {
            const char * nm = ggml_backend_dev_name(c.dev);
            const res f = run(c, c.src, 256u << 20, 256u << 20);   // the first use of this host buffer by this card
            printf("%-10s %-28s %10zu %9.2f %9.1f\n", nm, "first use of the host buffer", (size_t) 256 << 10, f.gbs, f.issue_us);
            for (size_t ch : chunks) {
                const res r = run(c, c.src, ch, 1u << 30);
                printf("%-10s %-28s %10zu %9.2f %9.1f\n", nm, "own host buffer", ch >> 10, r.gbs, r.issue_us);
            }
        }
        if (cards.size() >= 2) {
            for (size_t i = 0; i < cards.size(); ++i) {   // the other card's host buffer, first use then warm
                card & c = cards[i];
                const char * other = cards[(i + 1) % cards.size()].src;
                const res f = run(c, other, 256u << 20, 256u << 20);
                printf("%-10s %-28s %10zu %9.2f %9.1f\n", ggml_backend_dev_name(c.dev), "other card's buffer, 1st", (size_t) 256 << 10, f.gbs, f.issue_us);
                const res r = run(c, other, 4u << 20, 1u << 30);
                printf("%-10s %-28s %10zu %9.2f %9.1f\n", ggml_backend_dev_name(c.dev), "other card's buffer", (size_t) 4 << 10, r.gbs, r.issue_us);
            }
            for (size_t ch : { (size_t) 256u << 20, (size_t) 4u << 20, (size_t) 1u << 20 }) {   // every card at once, a thread each
                std::vector<res> rr(cards.size());
                std::vector<std::thread> th;
                const double t0 = now_s();
                for (size_t i = 0; i < cards.size(); ++i) { th.emplace_back([&, i] { rr[i] = run(cards[i], cards[i].src, ch, 1u << 30); }); }
                for (auto & t : th) { t.join(); }
                const double sum = (double) cards.size() * (double) (1u << 30) / (now_s() - t0) / 1e9;
                for (size_t i = 0; i < cards.size(); ++i) {
                    printf("%-10s %-28s %10zu %9.2f %9.1f\n", ggml_backend_dev_name(cards[i].dev), "all cards at once", ch >> 10, rr[i].gbs, rr[i].issue_us);
                }
                printf("%-10s %-28s %10zu %9.2f\n", "sum", "all cards at once", ch >> 10, sum);
            }
        }
    }
    return 0;
}
