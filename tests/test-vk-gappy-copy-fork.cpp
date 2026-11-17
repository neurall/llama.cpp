// Vulkan copies a gappy view into a compact tensor the way CUDA does (ggml_backend_cuda_cpy_tensor_async, rows_copy): a layer's K or V inside the token-major KV
// record has one row per cell with the other layers' rows in between, the scheduler gives its copy on another backend a compact layout
// (ggml_dup_tensor_layout, ggml_backend_tensor_is_gappy) and calls the destination backend's cpy_tensor_async on the pair. Only the rows may cross, and nothing
// past the destination may change. Before 2026-11-17 Vulkan copied the view's whole byte span flat: the wrong layout, and bytes written past the destination.
//
// Cases per Vulkan card: the source in a buffer of the same card (another context on it) and in pinned host memory (ggml_backend_vk_host_buffer_type, from the
// CPU backend); f16 and q8_0 rows; a small copy (the host path's CPU copy below 128 KiB) and a large one.
//
// build: cmake --build <dir> --target test-vk-gappy-copy-fork;  run: test-vk-gappy-copy-fork
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_run = 0, n_failed = 0;

static void check(const char * what, ggml_backend_t src_be, ggml_backend_buffer_type_t src_buft, ggml_backend_t dst_be, ggml_type type, int64_t cells) {
    const int64_t D = 128, H = 4;
    const int     layers = 6, layer = 3;
    const size_t  rowb = ggml_row_size(type, D);
    const size_t  rec  = (size_t) layers * 2 * H * rowb;   // a cell's record: K and V of every layer, all heads
    char name[200];
    snprintf(name, sizeof(name), "%s, %s, %lld cells", what, ggml_type_name(type), (long long) cells);
    n_run++;

    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * sctx = ggml_init(ip);
    ggml_context * dctx = ggml_init(ip);
    ggml_tensor * record = ggml_new_tensor_2d(sctx, type, D, cells * layers * 2 * H);   // the KV record, one row per (cell, layer, K/V, head)
    ggml_tensor * view   = ggml_view_3d(sctx, record, D, cells, H, rec, rowb, (size_t) layer * 2 * H * rowb);   // the layer's K: [D, cells, heads]
    ggml_tensor * dst    = ggml_new_tensor_3d(dctx, type, D, cells, H);
    ggml_tensor * guard  = ggml_new_tensor_1d(dctx, GGML_TYPE_I8, 1 << 20);
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, src_buft);
    ggml_backend_buffer_t dbuf = ggml_backend_alloc_ctx_tensors(dctx, dst_be);
    if (!sbuf || !dbuf || (char *) guard->data < (char *) dst->data + ggml_nbytes(dst)) {
        printf("  %-60s SKIPPED (no buffers, or the guard does not follow the destination)\n", name);
        n_run--;
    } else {
        std::mt19937 rng(7);
        std::vector<uint8_t> src_bytes(ggml_nbytes(record));
        for (auto & b : src_bytes) { b = (uint8_t) rng(); }
        ggml_backend_tensor_set(record, src_bytes.data(), 0, src_bytes.size());
        std::vector<uint8_t> pat(ggml_nbytes(dst), 0xA5), gpat(ggml_nbytes(guard), 0x5A);
        ggml_backend_tensor_set(dst, pat.data(), 0, pat.size());
        ggml_backend_tensor_set(guard, gpat.data(), 0, gpat.size());

        bool took = dst_be->iface.cpy_tensor_async && dst_be->iface.cpy_tensor_async(src_be, dst_be, view, dst);
        ggml_backend_synchronize(src_be);
        ggml_backend_synchronize(dst_be);

        std::vector<uint8_t> out(ggml_nbytes(dst)), gout(ggml_nbytes(guard)), want(ggml_nbytes(dst));
        ggml_backend_tensor_get(dst, out.data(), 0, out.size());
        ggml_backend_tensor_get(guard, gout.data(), 0, gout.size());
        const size_t voff = (size_t) ((char *) view->data - (char *) record->data);
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t c = 0; c < cells; ++c) {
                memcpy(want.data() + h * dst->nb[2] + c * dst->nb[1], src_bytes.data() + voff + h * view->nb[2] + c * view->nb[1], rowb);
            }
        }
        size_t bad_rows = 0, bad_guard = 0;
        for (int64_t r = 0; r < cells * H; ++r) { bad_rows += memcmp(out.data() + r * rowb, want.data() + r * rowb, rowb) != 0; }
        for (auto b : gout) { bad_guard += b != 0x5A; }
        if (!took) {
            n_failed++;
            printf("  %-60s FAILED: the backend refused the copy (CUDA takes the rows)\n", name);
        } else if (bad_rows > 0 || bad_guard > 0) {
            n_failed++;
            printf("  %-60s FAILED: %zu of %lld rows wrong, %zu bytes past the destination changed\n", name, bad_rows, (long long) (cells * H), bad_guard);
        } else {
            printf("  %-60s OK\n", name);
        }
    }
    if (sbuf) { ggml_backend_buffer_free(sbuf); }
    if (dbuf) { ggml_backend_buffer_free(dbuf); }
    ggml_free(sctx);
    ggml_free(dctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-vk-gappy-copy-fork: no Vulkan device, skipped\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
        ggml_backend_t a = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);   // another context on the card
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
        printf("%s:\n", ggml_backend_dev_name(dev));
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int64_t cells : { 8, 600 }) {
                check("from a buffer of the same card", a, ggml_backend_get_default_buffer_type(a), b, type, cells);
                if (host) { check("from pinned host memory", cpu, host, b, type, cells); }
            }
        }
        ggml_backend_free(a);
        ggml_backend_free(b);
    }
    ggml_backend_free(cpu);
    printf("test-vk-gappy-copy-fork: %d checks, %d failed\n", n_run, n_failed);
    return n_failed > 0 ? 1 : 0;
}
