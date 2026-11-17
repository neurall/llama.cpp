// The engine's link probe (llama_moe_link_gbs_fork = early_link_gbs in src/llama-moecache-fork.cpp) against the rate the expert stream gets: async uploads
// from the card's pinned host buffer type, back to back, one wait (ggml_backend_tensor_set_async + ggml_backend_synchronize, as fork/iobench/uploadbench.cpp).
// The probe prices every upload the planner calculates with, so it has to read the stream's path.
// PC1 2026-11-17 (2x RTX 3090, PCIe 4.0 x16): the Vulkan probe read 15.0-15.7 GB/s (a synchronous write into the BAR-mapped VRAM is a CPU copy), the
// stream 26.7 on both cards and both queues; CUDA 26.7 both ways.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

double llama_moe_link_gbs_fork(ggml_backend_dev_t d);   // src/llama-moecache-fork.cpp

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    ggml_backend_load_all();
    const size_t n = 64u << 20;
    int gpus = 0, fails = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(d);
        ggml_backend_t be = ggml_backend_dev_init(d, nullptr);
        ggml_backend_buffer_t hbuf = hb ? ggml_backend_buft_alloc_buffer(hb, n) : nullptr;
        ggml_backend_buffer_t dbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d), n);
        if (!be || !hbuf || !dbuf) {
            printf("%-8s no backend or no pinned host buffer: skipped\n", ggml_backend_dev_name(d));
            continue;
        }
        ++gpus;
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) n);
        ggml_backend_tensor_alloc(dbuf, t, ggml_backend_buffer_get_base(dbuf));
        const char * src = (const char *) ggml_backend_buffer_get_base(hbuf);
        double stream = 0.0;
        for (int r = 0; r < 4; ++r) {   // round 0 warms up (first use of the host buffer by this card)
            const double t0 = now_s();
            for (int k = 0; k < 8; ++k) {
                ggml_backend_tensor_set_async(be, t, src, 0, n);
            }
            ggml_backend_synchronize(be);
            if (r > 0) {
                stream = std::max(stream, 8.0 * (double) n / (now_s() - t0) / 1e9);
            }
        }
        const double probe = llama_moe_link_gbs_fork(d);
        const bool ok = probe >= 0.9 * stream;
        printf("%-8s probe %6.2f GB/s, stream path %6.2f GB/s: %s\n", ggml_backend_dev_name(d), probe, stream,
               ok ? "ok" : "FAIL (the probe reads below 90% of the stream's rate)");
        fails += ok ? 0 : 1;
        ggml_free(ctx);
        ggml_backend_buffer_free(hbuf);
        ggml_backend_buffer_free(dbuf);
        ggml_backend_free(be);
    }
    printf("%d GPUs, %d failed\n", gpus, fails);
    return fails ? 1 : 0;
}
