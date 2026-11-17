// A Vulkan buffer holds what the driver takes in one allocation. ggml-vulkan reported its default 1 GiB "suballocation block" as the buffer type's
// largest buffer, while the RTX 3090's driver (595) takes 1 TiB in one buffer: ggml-alloc cut every graph and every weight set into 1 GiB buffers,
// and a graph planned into one region lent by the pool was refused when it passed 1 GiB (OLMoE at 32k cells, 2026-11-17).
//
// The test asks the card for one 2 GiB buffer; a driver that refuses it skips the test. Then the buffer type's limit must be at least that, and a
// graph of 1.5 GiB (c = a + b, three tensors of 512 MiB alive at once) must be planned in one buffer and compute right at its end.
//
// build: cmake --build <dir> --target test-vk-max-size-fork;  run: test-vk-max-size-fork (skipped without a Vulkan card)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-vk-max-size-fork: no Vulkan device, skipped\n");
        return 0;
    }
    ggml_backend_dev_t         dev  = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t big = (size_t) 2 << 30;
    ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(buft, big);
    if (!probe) {
        printf("test-vk-max-size-fork: %s takes no 2 GiB buffer, skipped\n", ggml_backend_dev_name(dev));
        return 0;
    }
    ggml_backend_buffer_free(probe);
    const size_t max = ggml_backend_buft_get_max_size(buft);
    printf("test-vk-max-size-fork: %s: one 2 GiB buffer allocated; the buffer type's largest buffer is %.2f GiB\n", ggml_backend_dev_name(dev), max / 1073741824.0);
    if (max < big) {
        printf("test-vk-max-size-fork: FAILED: the buffer type reports a smaller limit than the driver takes\n");
        return 1;
    }

    // a graph of 1.5 GiB planned by ggml-alloc: one buffer, and the sum is right at the end of c (past 1 GiB in that buffer)
    const int64_t ne0 = 4096, ne1 = 32768;   // 512 MiB of f32 each
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_set_output(a);   // outputs too: c may not take their memory, all three stay alive
    ggml_set_output(b);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_gallocr_t ga = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        printf("test-vk-max-size-fork: FAILED: the 1.5 GiB graph was not allocated\n");
        return 1;
    }
    if (a->buffer != b->buffer || b->buffer != c->buffer) {
        printf("test-vk-max-size-fork: FAILED: the 1.5 GiB graph was cut into several buffers (a, b, c in %s)\n",
               a->buffer == b->buffer ? "two" : "three");
        return 1;
    }
    const size_t n = (size_t) ne0 * ne1;
    std::vector<float> av(n), bv(n, 1.0f), cv(ne0);
    for (size_t i = 0; i < n; ++i) { av[i] = (float) (i % 4099); }
    ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("test-vk-max-size-fork: FAILED: the add did not run\n");
        return 1;
    }
    ggml_backend_tensor_get(c, cv.data(), ggml_nbytes(c) - ne0 * sizeof(float), ne0 * sizeof(float));   // the last row
    for (int64_t i = 0; i < ne0; ++i) {
        const float want = (float) ((n - ne0 + i) % 4099) + 1.0f;
        if (cv[i] != want) {
            printf("test-vk-max-size-fork: FAILED: c[%zu] is %g, not %g\n", (size_t) (n - ne0 + i), cv[i], want);
            return 1;
        }
    }
    printf("test-vk-max-size-fork: the 1.5 GiB graph lies in one buffer of %.2f GiB and its last row is right\n", ggml_gallocr_get_buffer_size(ga, 0) / 1073741824.0);
    ggml_gallocr_free(ga);
    ggml_backend_free(be);
    ggml_free(ctx);
    return 0;
}
