// GPU test of Vulkan's temporaries (prealloc_x, prealloc_y, prealloc_split_k and the add+rms partials) inside a region of a buffer the caller lends: the region the
// expert pool lends each graph behind its tensors (llama_moe_cache_alias_fork), as on CUDA. Before 2026-11-17 Vulkan made them on demand outside the pool (the VRAM
// guard named 9 such buffers in an OLMoE request); with the region they are views into the lent buffer, sized per graph by the sizing pass (scratch_measure).
//
// Every op that takes temporaries runs on each Vulkan card without a region (the backend's own buffers) and with one, in a buffer filled with a pattern first:
//  - the output is the same, bit for bit (only where the temporaries live differs);
//  - no byte of the lent buffer outside [offset, offset + what ggml_backend_fork_nodes_scratch_fork computed) changed;
//  - the high-water mark the backend reports equals what was computed;
//  - no device memory was made while the region was lent (VRAM guard).
// A region lent to another context of the card is not used (its mark stays 0), and one too small is not used either (the op takes its own buffers, the mark says
// how much it asked, so the pool learns it was short).
//
// build: cmake --build <dir> --target test-vk-scratch-fork;  run: test-vk-scratch-fork [card]   (no Vulkan card: skipped, exit 0)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

extern "C" void ggml_backend_vram_guard_fork(int mode);
extern "C" void ggml_backend_vram_guard_stats_fork(uint64_t * n, uint64_t * bytes);

using scratch_t = size_t (*)(int, ggml_backend_buffer_t, size_t, size_t, ggml_backend_t);   // device, region buffer, offset in it, capacity, the backend it serves
using nodes_t   = size_t (*)(int, const ggml_tensor * const *, int);

static scratch_t scratch_fn = nullptr;
static nodes_t   nodes_fn   = nullptr;
static int n_failed = 0, n_run = 0, n_skipped = 0;

struct tcase {
    const char * name;
    int n_expert;   // the I32 leaves (expert ids) hold distinct experts below this per row
    std::function<ggml_tensor *(ggml_context *)> build;
};

// random contents for every input the build made: floats, halves, quantized weights (from random floats), expert ids
static void fill_inputs(ggml_context * ctx, int n_expert, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE || t->buffer == nullptr || t->view_src != nullptr) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        std::vector<uint8_t> data(ggml_nbytes(t));
        if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> experts(n_expert);
            int32_t * d = (int32_t *) data.data();
            for (int64_t r = 0; r < n / t->ne[0]; ++r) {
                std::iota(experts.begin(), experts.end(), 0);
                std::shuffle(experts.begin(), experts.end(), rng);
                for (int64_t i = 0; i < t->ne[0]; ++i) { d[r * t->ne[0] + i] = experts[i]; }
            }
        } else {
            std::vector<float> f(n);
            for (float & x : f) { x = u(rng); }
            if (t->type == GGML_TYPE_F32) {
                memcpy(data.data(), f.data(), data.size());
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) data.data(), n);
            } else {
                ggml_quantize_chunk(t->type, f.data(), data.data(), 0, n / t->ne[0], t->ne[0], nullptr);
            }
        }
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
}

static std::vector<uint8_t> compute(ggml_backend_t be, ggml_cgraph * gf, ggml_tensor * y) {
    std::vector<uint8_t> out(ggml_nbytes(y));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        out.clear();
        return out;
    }
    ggml_backend_synchronize(be);
    ggml_backend_tensor_get(y, out.data(), 0, out.size());
    return out;
}

static double nmse(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const float * x = (const float *) a.data();
    const float * y = (const float *) b.data();
    double err = 0, ref = 0;
    for (size_t i = 0; i < a.size() / sizeof(float); ++i) {
        err += ((double) x[i] - y[i]) * ((double) x[i] - y[i]);
        ref += (double) x[i] * x[i];
    }
    return ref > 0 ? err / ref : err;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-vk-scratch-fork: no Vulkan card, skipped\n");
        return 0;
    }
    scratch_fn = (scratch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fork_scratch_fork");
    nodes_fn   = (nodes_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_fork_nodes_scratch_fork");
    if (!scratch_fn || !nodes_fn) {
        printf("test-vk-scratch-fork: FAILED: the Vulkan backend does not export the scratch region (ggml_backend_fork_scratch_fork, ggml_backend_fork_nodes_scratch_fork)\n");
        return 1;
    }
    const int only = argc > 1 ? atoi(argv[1]) : -1;

    const tcase cases[] = {
        { "mul_mat q4_K 4096 x 4096, 64 tokens (activations converted)", 0, [](ggml_context * c) {
            return ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_Q4_K, 4096, 4096), ggml_new_tensor_2d(c, GGML_TYPE_F32, 4096, 64)); } },
        { "mul_mat q8_0 16384 x 32, 16 tokens (split k)", 0, [](ggml_context * c) {
            return ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, 16384, 32), ggml_new_tensor_2d(c, GGML_TYPE_F32, 16384, 16)); } },
        { "mul_mat f16 weight with strided rows (made contiguous first)", 0, [](ggml_context * c) {
            ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 512, 512);
            return ggml_mul_mat(c, ggml_view_2d(c, w, 256, 512, w->nb[1], 0), ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 32)); } },
        { "mul_mat_id q4_K 16 experts, 4 used, 64 tokens", 16, [](ggml_context * c) {
            return ggml_mul_mat_id(c, ggml_new_tensor_3d(c, GGML_TYPE_Q4_K, 2048, 1024, 16), ggml_new_tensor_3d(c, GGML_TYPE_F32, 2048, 4, 64),
                                   ggml_new_tensor_2d(c, GGML_TYPE_I32, 4, 64)); } },
        { "mul_mat_id q4_K 16 experts, 4 used, 1 token", 16, [](ggml_context * c) {
            return ggml_mul_mat_id(c, ggml_new_tensor_3d(c, GGML_TYPE_Q4_K, 2048, 1024, 16), ggml_new_tensor_3d(c, GGML_TYPE_F32, 2048, 4, 1),
                                   ggml_new_tensor_2d(c, GGML_TYPE_I32, 4, 1)); } },
        { "flash_attn_ext f16 K/V, 1 query, 4096 cached (split k)", 0, [](ggml_context * c) {
            return ggml_flash_attn_ext(c, ggml_new_tensor_4d(c, GGML_TYPE_F32, 128, 1, 32, 1), ggml_new_tensor_4d(c, GGML_TYPE_F16, 128, 4096, 8, 1),
                                       ggml_new_tensor_4d(c, GGML_TYPE_F16, 128, 4096, 8, 1), nullptr, 0.0884f, 0.0f, 0.0f); } },
        { "flash_attn_ext q8_0 K/V, 8 queries, 4096 cached", 0, [](ggml_context * c) {
            return ggml_flash_attn_ext(c, ggml_new_tensor_4d(c, GGML_TYPE_F32, 128, 8, 32, 1), ggml_new_tensor_4d(c, GGML_TYPE_Q8_0, 128, 4096, 8, 1),
                                       ggml_new_tensor_4d(c, GGML_TYPE_Q8_0, 128, 4096, 8, 1), nullptr, 0.0884f, 0.0f, 0.0f); } },
        { "soft_max 65536 wide", 0, [](ggml_context * c) {
            return ggml_soft_max(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 65536, 8)); } },
        { "argsort 8192 wide", 0, [](ggml_context * c) {
            return ggml_argsort(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 8192, 16), GGML_SORT_ORDER_DESC); } },
        { "top_k 8 of 32768", 0, [](ggml_context * c) {
            return ggml_top_k(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 32768, 4), 8); } },
        { "cumsum 65536 wide", 0, [](ggml_context * c) {
            return ggml_cumsum(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 65536, 4)); } },
        { "cross_entropy_loss 32000 x 8", 0, [](ggml_context * c) {
            return ggml_cross_entropy_loss(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 32000, 8), ggml_new_tensor_2d(c, GGML_TYPE_F32, 32000, 8)); } },
    };

    for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) {
        if (only >= 0 && (int) d != only) { continue; }
        ggml_backend_dev_t dev   = ggml_backend_reg_dev_get(reg, d);
        ggml_backend_t     be    = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_t     other = ggml_backend_dev_init(dev, nullptr);   // another context on the card (a draft model's): the region is not lent to it
        if (!be || !other) { continue; }
        // the lent buffer: the region starts at a 256-byte step inside it (the pool lends behind the graph's tensors), the pattern marks every byte around it
        const size_t pool_size = (size_t) 256 << 20, off = ((size_t) 64 << 20) + 256, cap = (size_t) 160 << 20;
        ggml_backend_buffer_t pool = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), pool_size);
        if (!pool) { printf("card %zu: no memory for the lent buffer, skipped\n", d); continue; }
        ggml_init_params pip = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * pctx  = ggml_init(pip);
        ggml_tensor  * whole = ggml_new_tensor_1d(pctx, GGML_TYPE_I8, (int64_t) pool_size);
        ggml_backend_tensor_alloc(pool, whole, ggml_backend_buffer_get_base(pool));
        std::vector<uint8_t> host(pool_size);
        const uint8_t pat = 0xA5;
        // bytes of the lent buffer outside [off, off + used) that are no longer the pattern
        auto spilled = [&](size_t used, size_t & first) {
            ggml_backend_tensor_get(whole, host.data(), 0, pool_size);
            size_t n = 0;
            first = SIZE_MAX;
            for (size_t i = 0; i < pool_size; ++i) {
                if ((i < off || i >= off + used) && host[i] != pat) { n++; first = std::min(first, i); }
            }
            return n;
        };
        printf("Testing Vulkan's temporaries in a lent region, card %zu %s:\n", d, ggml_backend_dev_name(dev));
        {   // the pattern itself: every byte of the cleared buffer holds it (a clear that filled 32-bit words with the value left A5 00 00 00, 2026-11-17)
            size_t first = 0;
            ggml_backend_buffer_clear(pool, pat);
            const size_t n = spilled(pool_size - off, first);   // every byte before off is checked
            n_run++;
            if (n > 0) {
                n_failed++;
                printf("  FAILED: ggml_backend_buffer_clear(0x%02x) left %zu bytes of the first %zu without the value (first at %zu)\n", pat, n, off, first);
            }
        }
        std::mt19937 rng(42);
        int with_temps = 0;
        size_t none_total = 0;   // what an op without temporaries is computed to take (the add+rms partials only)
        {
            ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * c = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
            ggml_tensor * y = ggml_add(c, a, a);
            none_total = nodes_fn((int) d, &y, 1);
            ggml_free(c);
        }
        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
            const tcase & tc = cases[ci];
            ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor  * y   = tc.build(ctx);
            if (!ggml_backend_supports_op(be, y)) {
                printf("  %-62s not supported, skipped\n", tc.name);
                n_skipped++;
                ggml_free(ctx);
                continue;
            }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            fill_inputs(ctx, tc.n_expert, rng);
            std::vector<ggml_tensor *> nodes(ggml_graph_n_nodes(gf));
            for (int i = 0; i < (int) nodes.size(); ++i) { nodes[i] = ggml_graph_node(gf, i); }
            const size_t computed = nodes_fn((int) d, nodes.data(), (int) nodes.size());
            if (computed > none_total) { with_temps++; }

            scratch_fn((int) d, nullptr, 0, 0, nullptr);   // no region: the backend's own buffers
            const std::vector<uint8_t> ref  = compute(be, gf, y);
            const std::vector<uint8_t> ref2 = compute(be, gf, y);
            const bool exact = ref == ref2;   // an op whose result varies from run to run is compared by its error instead
            auto same = [&](const std::vector<uint8_t> & out) {
                if (out.size() != ref.size()) { return false; }
                if (exact) { return out == ref; }
                return y->type == GGML_TYPE_F32 ? nmse(ref, out) < 1e-10 : true;   // indices that vary from run to run are not compared
            };
            char what[160];
            auto fail = [&](const char * why) { n_failed++; printf("  FAILED %s: %s\n", what, why); };

            // 1. lent to this context: the temporaries live in [off, off + computed), nothing else is touched and no device memory is made
            snprintf(what, sizeof(what), "%s, region lent", tc.name);
            n_run++;
            ggml_backend_buffer_clear(pool, pat);
            scratch_fn((int) d, pool, off, cap, be);
            uint64_t g0 = 0, g1 = 0, gb = 0;
            ggml_backend_vram_guard_fork(1);
            ggml_backend_vram_guard_stats_fork(&g0, &gb);
            const std::vector<uint8_t> out = compute(be, gf, y);
            ggml_backend_vram_guard_stats_fork(&g1, &gb);
            ggml_backend_vram_guard_fork(0);
            const size_t asked = scratch_fn((int) d, nullptr, 0, 0, nullptr);
            size_t first = 0;
            const size_t outside = spilled(computed, first);
            if (!same(out))          { fail("the output differs from the run on the backend's own buffers"); }
            if (outside > 0)         { char m[160]; snprintf(m, sizeof(m), "%zu bytes outside the region changed (first at %zu, region %zu + %zu)", outside, first, off, computed); fail(m); }
            if (asked != computed)   { char m[160]; snprintf(m, sizeof(m), "the op asked for %zu bytes, %zu were computed", asked, computed); fail(m); }
            if (g1 != g0)            { char m[160]; snprintf(m, sizeof(m), "%llu device buffers made while the region was lent", (unsigned long long) (g1 - g0)); fail(m); }
            printf("  %-62s %10zu bytes in the region%s\n", tc.name, computed, exact ? "" : " (result varies between runs: compared by its error)");

            // 2. lent to another context on the card: this one keeps its own buffers and does not count in the mark
            snprintf(what, sizeof(what), "%s, region lent to another context", tc.name);
            n_run++;
            ggml_backend_buffer_clear(pool, pat);
            scratch_fn((int) d, pool, off, cap, other);
            const std::vector<uint8_t> out2 = compute(be, gf, y);
            const size_t asked2 = scratch_fn((int) d, nullptr, 0, 0, nullptr);
            const size_t outside2 = spilled(0, first);
            if (!same(out2))   { fail("the output differs"); }
            if (outside2 > 0)  { char m[160]; snprintf(m, sizeof(m), "%zu bytes of the lent buffer changed (first at %zu)", outside2, first); fail(m); }
            if (asked2 != 0)   { char m[160]; snprintf(m, sizeof(m), "the mark is %zu, its asks count only for the context the region is lent to", asked2); fail(m); }

            // 3. a region too small: not used at all (the op takes its own buffers), the mark says what it asked for
            if (computed > none_total) {
                snprintf(what, sizeof(what), "%s, region too small", tc.name);
                n_run++;
                ggml_backend_buffer_clear(pool, pat);
                scratch_fn((int) d, pool, off, computed - 256, be);
                const std::vector<uint8_t> out3 = compute(be, gf, y);
                const size_t asked3 = scratch_fn((int) d, nullptr, 0, 0, nullptr);
                const size_t outside3 = spilled(0, first);
                if (!same(out3))          { fail("the output differs"); }
                if (outside3 > 0)         { char m[160]; snprintf(m, sizeof(m), "%zu bytes of the lent buffer changed (first at %zu)", outside3, first); fail(m); }
                if (asked3 != computed)   { char m[160]; snprintf(m, sizeof(m), "the mark is %zu, the op asked for %zu", asked3, computed); fail(m); }
            }
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        if (with_temps < 3) {
            n_failed++;
            printf("  FAILED: only %d of the ops took temporaries on this card: the test exercises nothing\n", with_temps);
        }
        scratch_fn((int) d, nullptr, 0, 0, nullptr);
        ggml_free(pctx);
        ggml_backend_buffer_free(pool);
        ggml_backend_free(other);
        ggml_backend_free(be);
    }
    printf("test-vk-scratch-fork: %d checks, %d failed, %d skipped\n", n_run, n_failed, n_skipped);
    return n_failed == 0 ? 0 : 1;
}
