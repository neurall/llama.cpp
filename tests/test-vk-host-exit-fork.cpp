// A pinned host buffer that is still allocated when the process exits. The engine keeps some for the life of the process (cstream's id slots
// g_cs_pio.buf, the table snapshots mc->cs_snap_buf), and CUDA leaves such memory to the end of the process. Vulkan freed what was left in the
// static destructor of its instance (vk_instance_t::host_allocations), when the driver had already torn itself down: every Vulkan llama-server and
// llama-perplexity that streamed a prompt died with SIGSEGV on its way out (kernel log "segfault at ...3f650 ... error 14", 2026-11-17).
//
// The driver's own exit handler unloads its core library (libnvidia-eglcore, loaded once the card has run work) before that destructor runs, and
// freeing the buffer then jumped into the unloaded code (gdb: the driver's dlclose calls from __run_exit_handlers, then SIGSEGV in
// vk_buffer_struct::~vk_buffer_struct called from vk_instance_t::~vk_instance_t).
//
// The test runs an add on the first Vulkan card and copies its result into a pinned buffer (as cstream's ids come back), frees the backend and the
// card's buffer (as a context does), keeps the pinned buffer and returns 0: a crash at exit fails it (ctest sees the signal).
//
// build: cmake --build <dir> --target test-vk-host-exit-fork;  run: test-vk-host-exit-fork (skipped without a Vulkan card)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-vk-host-exit-fork: no Vulkan device, skipped\n");
        return 0;
    }
    ggml_backend_dev_t         dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t hb  = ggml_backend_dev_host_buffer_type(dev);
    if (!hb) {
        printf("test-vk-host-exit-fork: %s has no pinned host buffer type, skipped\n", ggml_backend_dev_name(dev));
        return 0;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_t keep = ggml_backend_buft_alloc_buffer(hb, 8u << 20);
    if (!keep) {
        printf("test-vk-host-exit-fork: FAILED: no pinned buffer of 8 MiB on %s\n", ggml_backend_dev_name(dev));
        return 1;
    }

    // work on the card: c = a + b, then c into a tensor of the pinned buffer
    const int64_t n = 1 << 16;
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> av(n, 1.0f), bv(n, 2.0f);
    ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("test-vk-host-exit-fork: FAILED: the add did not run on %s\n", ggml_backend_dev_name(dev));
        return 1;
    }
    ggml_context * hctx = ggml_init(ip);
    ggml_tensor * h = ggml_new_tensor_1d(hctx, GGML_TYPE_F32, n);
    ggml_tallocr ta = ggml_tallocr_new(keep);
    ggml_tallocr_alloc(&ta, h);
    ggml_backend_tensor_copy(c, h);
    const float * hp = (const float *) h->data;
    if (hp[0] != 3.0f || hp[n - 1] != 3.0f) {
        printf("test-vk-host-exit-fork: FAILED: the sum in the pinned buffer is %g, not 3\n", hp[0]);
        return 1;
    }
    ggml_backend_buffer_free(dbuf);
    ggml_free(ctx);
    ggml_backend_free(be);
    printf("test-vk-host-exit-fork: %s: a pinned buffer of 8 MiB kept until exit; a crash after this line is the bug\n", ggml_backend_dev_name(dev));
    return 0;   // the pinned buffer (and hctx) are never freed, as the engine's process-long ones
}
