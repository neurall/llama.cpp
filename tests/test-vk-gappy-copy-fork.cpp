// Vulkan copies a gappy view into a compact tensor the way CUDA does (ggml_backend_cuda_cpy_tensor_async, rows_copy): a layer's K or V inside the token-major KV
// record has one row per cell with the other layers' rows in between, the scheduler gives its copy on another backend a compact layout
// (ggml_dup_tensor_layout, ggml_backend_tensor_is_gappy) and calls the destination backend's cpy_tensor_async on the pair. Only the rows may cross, and nothing
// past the destination may change. Before 2026-11-17 Vulkan copied the view's whole byte span flat: the wrong layout, and bytes written past the destination.
//
// Cases per Vulkan card: the source in a buffer of the same card (another context on it) and in pinned host memory (ggml_backend_vk_host_buffer_type, from the
// CPU backend); f16 and q8_0 rows; a small copy (the host path's CPU copy below 128 KiB) and a large one.
//
// The synchronous copy (ggml_backend_tensor_copy) is the scheduler's copy when the destination backend refuses the pair: Vulkan to another card, a card to the
// CPU. It ran one transfer per row, a submission and a fence wait each: MiMo V2.6 on two Vulkan cards (2026-11-17) took ~16 s per layer for its K and V at an 11k
// prompt, 16 t/s for the whole prompt. Its cases copy an 11k-cell view in both layouts of a layer inside the record (heads outer, heads inside a cell) and fail
// above 500 ms (a few transfers take ~10 ms).
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

// sync: ggml_backend_tensor_copy (timed) instead of the destination's cpy_tensor_async; cell_major: the view [D, heads, cells] (a cell's heads one run) instead of
// [D, cells, heads]
static void check(const char * what, ggml_backend_t src_be, ggml_backend_buffer_type_t src_buft, ggml_backend_t dst_be, ggml_type type, int64_t cells,
                  bool sync = false, bool cell_major = false) {
    const int64_t D = 128, H = 4;
    const int     layers = 6, layer = 3;
    const size_t  rowb = ggml_row_size(type, D);
    const size_t  rec  = (size_t) layers * 2 * H * rowb;   // a cell's record: K and V of every layer, all heads
    char name[200];
    snprintf(name, sizeof(name), "%s, %s, %lld cells%s", what, ggml_type_name(type), (long long) cells, cell_major ? ", heads inside a cell" : "");
    n_run++;

    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * sctx = ggml_init(ip);
    ggml_context * dctx = ggml_init(ip);
    ggml_tensor * record = ggml_new_tensor_2d(sctx, type, D, cells * layers * 2 * H);   // the KV record, one row per (cell, layer, K/V, head)
    const size_t koff = (size_t) layer * 2 * H * rowb;
    ggml_tensor * view   = cell_major ? ggml_view_3d(sctx, record, D, H, cells, rowb, rec, koff)    // the layer's K: [D, heads, cells]
                                      : ggml_view_3d(sctx, record, D, cells, H, rec, rowb, koff);   //                 [D, cells, heads]
    ggml_tensor * dst    = ggml_new_tensor_3d(dctx, type, D, view->ne[1], view->ne[2]);
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

        const int64_t t0 = ggml_time_us();
        bool took = true;
        if (sync) {
            ggml_backend_tensor_copy(view, dst);
        } else {
            took = dst_be->iface.cpy_tensor_async && dst_be->iface.cpy_tensor_async(src_be, dst_be, view, dst);
        }
        ggml_backend_synchronize(src_be);
        ggml_backend_synchronize(dst_be);
        const double ms = (ggml_time_us() - t0) / 1e3;

        std::vector<uint8_t> out(ggml_nbytes(dst)), gout(ggml_nbytes(guard)), want(ggml_nbytes(dst));
        ggml_backend_tensor_get(dst, out.data(), 0, out.size());
        ggml_backend_tensor_get(guard, gout.data(), 0, gout.size());
        const size_t voff = (size_t) ((char *) view->data - (char *) record->data);
        for (int64_t i2 = 0; i2 < view->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < view->ne[1]; ++i1) {
                memcpy(want.data() + i2 * dst->nb[2] + i1 * dst->nb[1], src_bytes.data() + voff + i2 * view->nb[2] + i1 * view->nb[1], rowb);
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
        } else if (sync && ms > 500.0) {
            n_failed++;
            printf("  %-60s FAILED: %.0f ms for %lld rows (the scheduler makes this copy for every layer of a pass)\n", name, ms, (long long) (cells * H));
        } else if (sync) {
            printf("  %-60s OK (%.1f ms)\n", name, ms);
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
    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    std::vector<ggml_backend_t> cards;
    for (size_t d = 0; d < n_dev; ++d) { cards.push_back(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, d), nullptr)); }
    for (size_t d = 0; d < n_dev; ++d) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
        ggml_backend_t a = cards[d];
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);   // another context on the card
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
        printf("%s:\n", ggml_backend_dev_name(dev));
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            for (int64_t cells : { 8, 600 }) {
                check("from a buffer of the same card", a, ggml_backend_get_default_buffer_type(a), b, type, cells);
                if (host) { check("from pinned host memory", cpu, host, b, type, cells); }
            }
        }
        for (bool cell_major : { false, true }) {
            check("sync copy to the CPU", a, ggml_backend_get_default_buffer_type(a), cpu, GGML_TYPE_F16, 11264, true, cell_major);
            if (n_dev > 1) {
                for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
                    check("sync copy to the next card", a, ggml_backend_get_default_buffer_type(a), cards[(d + 1) % n_dev], type, 11264, true, cell_major);
                }
            }
        }
        ggml_backend_free(b);
    }
    for (ggml_backend_t c : cards) { ggml_backend_free(c); }
    ggml_backend_free(cpu);
    printf("test-vk-gappy-copy-fork: %d checks, %d failed\n", n_run, n_failed);
    return n_failed > 0 ? 1 : 0;
}
