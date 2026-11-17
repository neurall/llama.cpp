// Vulkan flash attention over a layer's K and V inside the token-major KV record (one record per cell holds every layer's K and V, kv-tail forces this
// layout): the views' byte span is the cells times the record, past maxStorageBufferRange (4 GiB) at long contexts (GLM, about 188 KB a cell at f16: 23k
// cells). CUDA indexes with 64-bit strides; Vulkan bound the whole span to a 32-bit pipeline and its shaders multiply cell by stride in 32 bits. Now such a
// call runs in chunks of cells (ggml_vk_flash_attn, fa_chunks).
//
// Each case: a record buffer of the given size on the first Vulkan card, K and V of 8192 cells at its start (filler between them), flash attention with 8
// query heads on 4 KV heads, D 128; the CPU backend computes the same on compact K/V. Cases cover f16 and q8_0, 1 token (decode, grouped heads), 4 tokens
// (grouped heads), 16 tokens (rows of their own), and the tail mask (k0: the mask covers the cells from 4096 on, every cell before is visible; a tenth of
// the masked cells are -inf). Error as in test-backend-ops (normalized mean squared error, bound 5e-4).
//
// build: cmake --build <dir> --target test-vk-fa-span-fork;  run: test-vk-fa-span-fork (about 10 GiB of VRAM, skipped without a Vulkan card)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_run = 0, n_failed = 0;

static void check(ggml_backend_t vk, ggml_backend_t cpu, ggml_type type, double span_gib, int64_t N, int64_t k0) {
    const int64_t D = 128, H = 4, HQ = 8, n_kv = 8192;
    const size_t  headb = ggml_row_size(type, D);                                       // one head's K (or V) row of a cell
    const size_t  rowb  = (size_t) H * headb;                                           // one cell's K (or V) of this layer, all heads
    size_t unit = 256;   // a record is whole blocks of the type (its rows are) and a multiple of 256 bytes
    while (unit % ggml_type_size(type) != 0) { unit += 256; }
    const size_t  rec   = (size_t) (span_gib * (1ull << 30)) / n_kv / unit * unit;   // record bytes a cell
    char name[160];
    snprintf(name, sizeof(name), "%s, span %.1f GiB, %2lld token%s, mask from cell %lld", ggml_type_name(type), (double) rec * n_kv / (1ull << 30),
             (long long) N, N > 1 ? "s" : " ", (long long) k0);
    n_run++;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> kf((size_t) n_kv * H * D), vf((size_t) n_kv * H * D);   // [cell][head][d]
    for (auto & x : kf) { x = u(rng); }
    for (auto & x : vf) { x = u(rng); }
    std::vector<uint8_t> kb((size_t) n_kv * rowb), vb((size_t) n_kv * rowb);   // the same in the K/V type, rows [cell][head]
    ggml_quantize_chunk(type, kf.data(), kb.data(), 0, n_kv * H, D, nullptr);
    ggml_quantize_chunk(type, vf.data(), vb.data(), 0, n_kv * H, D, nullptr);
    std::vector<float> qh((size_t) D * N * HQ);
    for (auto & x : qh) { x = u(rng); }
    const int64_t mcols = n_kv - k0;
    std::vector<ggml_fp16_t> mh((size_t) mcols * N);
    for (auto & x : mh) { x = ggml_fp32_to_fp16(rng() % 10 == 0 ? -INFINITY : 0.25f * u(rng)); }
    const float scale = 1.0f / sqrtf((float) D);

    // Vulkan: the record, K and V as views with the cell stride of the record
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * vctx = ggml_init(ip);
    const int64_t rec_blocks = (int64_t) ((rec * n_kv + ggml_type_size(type) - 1) / ggml_type_size(type));
    ggml_tensor * record = ggml_new_tensor_1d(vctx, type, rec_blocks * ggml_blck_size(type));
    ggml_tensor * q  = ggml_new_tensor_3d(vctx, GGML_TYPE_F32, D, N, HQ);
    ggml_tensor * m  = ggml_new_tensor_2d(vctx, GGML_TYPE_F16, mcols, N);
    ggml_tensor * k  = ggml_view_3d(vctx, record, D, n_kv, H, rec, headb, 0);
    ggml_tensor * v  = ggml_view_3d(vctx, record, D, n_kv, H, rec, headb, rowb);
    ggml_tensor * o  = ggml_flash_attn_ext(vctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(vctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t vbuf = ggml_backend_alloc_ctx_tensors(vctx, vk);
    if (!vbuf) {
        printf("  %-62s SKIPPED (no buffer of that size)\n", name);
        n_run--;
        ggml_free(vctx);
        return;
    }
    const bool claimed = ggml_backend_supports_op(vk, o);
    ggml_backend_buffer_clear(vbuf, 0);
    for (int64_t c = 0; c < n_kv; ++c) {
        ggml_backend_tensor_set(record, kb.data() + (size_t) c * rowb, (size_t) c * rec,        rowb);
        ggml_backend_tensor_set(record, vb.data() + (size_t) c * rowb, (size_t) c * rec + rowb, rowb);
    }
    ggml_backend_tensor_set(q, qh.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(m, mh.data(), 0, ggml_nbytes(m));

    // CPU: compact K and V [D, cell, head]
    ggml_context * cctx = ggml_init(ip);
    ggml_tensor * cq = ggml_new_tensor_3d(cctx, GGML_TYPE_F32, D, N, HQ);
    ggml_tensor * cm = ggml_new_tensor_2d(cctx, GGML_TYPE_F16, mcols, N);
    ggml_tensor * ck = ggml_new_tensor_3d(cctx, type, D, n_kv, H);
    ggml_tensor * cv = ggml_new_tensor_3d(cctx, type, D, n_kv, H);
    ggml_tensor * co = ggml_flash_attn_ext(cctx, cq, ck, cv, cm, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(co, GGML_PREC_F32);
    ggml_cgraph * cg = ggml_new_graph(cctx);
    ggml_build_forward_expand(cg, co);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu);
    std::vector<uint8_t> kc(kb.size()), vc(vb.size());
    for (int64_t c = 0; c < n_kv; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            memcpy(kc.data() + ((size_t) h * n_kv + c) * headb, kb.data() + ((size_t) c * H + h) * headb, headb);
            memcpy(vc.data() + ((size_t) h * n_kv + c) * headb, vb.data() + ((size_t) c * H + h) * headb, headb);
        }
    }
    ggml_backend_tensor_set(cq, qh.data(), 0, ggml_nbytes(cq));
    ggml_backend_tensor_set(cm, mh.data(), 0, ggml_nbytes(cm));
    ggml_backend_tensor_set(ck, kc.data(), 0, ggml_nbytes(ck));
    ggml_backend_tensor_set(cv, vc.data(), 0, ggml_nbytes(cv));

    const ggml_status st = ggml_backend_graph_compute(vk, gf);
    ggml_backend_graph_compute(cpu, cg);
    std::vector<float> got(ggml_nelements(o)), want(ggml_nelements(co));
    if (st == GGML_STATUS_SUCCESS) { ggml_backend_tensor_get(o, got.data(), 0, ggml_nbytes(o)); }
    ggml_backend_tensor_get(co, want.data(), 0, ggml_nbytes(co));
    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < want.size(); ++i) { err += (got[i] - want[i]) * (double) (got[i] - want[i]); ref += want[i] * (double) want[i]; }
    const double nmse = ref > 0.0 ? err / ref : err;
    if (st != GGML_STATUS_SUCCESS || !(nmse <= 5e-4)) {
        n_failed++;
        printf("  %-62s FAILED: %s, error %.3g (bound 5e-4), the backend %s the op\n", name, st == GGML_STATUS_SUCCESS ? "computed" : "compute failed", nmse,
               claimed ? "claimed" : "refused");
    } else {
        printf("  %-62s OK (error %.3g)\n", name, nmse);
    }
    ggml_backend_buffer_free(vbuf);
    ggml_backend_buffer_free(cbuf);
    ggml_free(vctx);
    ggml_free(cctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-vk-fa-span-fork: no Vulkan device, skipped\n");
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t vk  = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    printf("%s:\n", ggml_backend_dev_name(dev));
    struct { ggml_type type; double span; int64_t N, k0; } cases[] = {
        { GGML_TYPE_F16,  2.0,  1,    0 }, { GGML_TYPE_F16,  2.0, 16, 4096 },                                     // under 4 GiB: one binding, as before
        { GGML_TYPE_F16,  5.0,  1,    0 }, { GGML_TYPE_F16,  5.0,  4, 4096 }, { GGML_TYPE_F16, 5.0, 16, 0 },     // two chunks (6400 + 1792 cells)
        { GGML_TYPE_F16,  5.0, 16, 4096 },
        { GGML_TYPE_F16,  9.0,  1, 4096 }, { GGML_TYPE_F16,  9.0, 16, 4096 }, { GGML_TYPE_F16, 9.0, 16, 0 },     // three (3584 + 3584 + 1024), the first unmasked
        { GGML_TYPE_Q8_0, 5.0,  1,    0 }, { GGML_TYPE_Q8_0, 5.0, 16, 4096 },
    };
    for (const auto & c : cases) {
        check(vk, cpu, c.type, c.span, c.N, c.k0);
    }
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    printf("test-vk-fa-span-fork: %d checks, %d failed\n", n_run, n_failed);
    return n_failed > 0 ? 1 : 0;
}
