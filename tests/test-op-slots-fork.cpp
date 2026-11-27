// fork: the op_params slots the fork gives a meaning of its own (ggml/src/ggml-op-slots-fork.h). Each is set through the fork's own setter;
// upstream's accessors of the op must still read what they read before, no other slot may change, and the fork's reader gets the value back.
// Upstream gave such slots a meaning three times (MUL_MAT slot 1 hint, MUL_MAT_ID slot 3 src1 precision, FLASH_ATTN_EXT slot 5 indexed
// bias); fork/op-slot-check finds that in the source, this test in the built library. CPU only, no model.

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-op-slots-fork.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fails; } } while (0)

static std::set<std::string> g_covered;

// every slot but [slot, slot + width) as it was
static bool others_unchanged(const int32_t * before, const ggml_tensor * t, int slot, int width) {
    for (int i = 0; i < (int) (GGML_MAX_OP_PARAMS/sizeof(int32_t)); ++i) {
        if ((i < slot || i >= slot + width) && before[i] != t->op_params[i]) {
            fprintf(stderr, "  slot %d changed: %d -> %d\n", i, before[i], t->op_params[i]);
            return false;
        }
    }
    return true;
}

static void fill(ggml_tensor * t, float scale) {
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        const float v = scale*(float) ((i*7919) % 101 - 50)/50.0f;
        if (t->type == GGML_TYPE_F16) { ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(v); } else { ((float *) t->data)[i] = v; }
    }
}

static std::vector<float> compute(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    CHECK(ggml_graph_compute_with_ctx(ctx, gf, 2) == GGML_STATUS_SUCCESS);
    return std::vector<float>((const float *) out->data, (const float *) out->data + ggml_nelements(out));
}

// FLASH_ATTN_EXT: the sparse gate (ggml_flash_attn_ext_set_sparse_min, read by ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse)
static void test_fa_sparse_min(ggml_context * ctx) {
    g_covered.insert("fa_sparse_min");
    const int D = 64, H = 4, HKV = 2, KV = 256, NB = 4, NKV_MAX = 32;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NB, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, HKV, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, HKV, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, NB, 1, 1);
    fill(q, 1.0f); fill(k, 0.5f); fill(v, 1.0f);
    for (int64_t r = 0; r < NB; ++r) {
        for (int64_t c = 0; c < KV; ++c) {   // at most NKV_MAX visible cells a row: a sparse attention's mask
            ((ggml_fp16_t *) m->data)[r*KV + c] = ggml_fp32_to_fp16(c % (KV/NKV_MAX) == r % (KV/NKV_MAX) ? 0.0f : -INFINITY);
        }
    }
    auto make = [&]() {
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_n_kv_max(out, NKV_MAX);
        return out;
    };

    ggml_tensor * out = make();
    int32_t before[GGML_MAX_OP_PARAMS/sizeof(int32_t)];
    memcpy(before, out->op_params, sizeof(before));
    const bool          bias0 = ggml_flash_attn_ext_has_indexed_bias(out);
    const enum ggml_prec prec0 = ggml_prec_get_acc(out);

    ggml_flash_attn_ext_set_sparse_min(out, 2048);

    // upstream's accessors read what they read before (slot 5: the indexed bias flag, the gate's slot until 2026-11-26)
    CHECK(ggml_flash_attn_ext_has_indexed_bias(out) == bias0);
    CHECK(!ggml_flash_attn_ext_has_indexed_bias(out));
    CHECK(ggml_prec_get_acc(out) == prec0);
    CHECK(others_unchanged(before, out, GGML_FORK_SLOT_FA_SPARSE_MIN, 1));
    // the fork's reader gets the value back
    CHECK(ggml_fork_fa_sparse_min(out) == 2048);

    // upstream's setters leave the gate alone
    ggml_prec_set_acc(out, GGML_PREC_F32);
    CHECK(ggml_prec_get_acc(out) == GGML_PREC_F32 && ggml_fork_fa_sparse_min(out) == 2048);
    ggml_tensor * table   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, H, NB, 1);
    ggml_tensor * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, KV, NB);
    ggml_tensor * with_bias = make();
    ggml_flash_attn_ext_set_sparse_min(with_bias, 2048);
    ggml_flash_attn_ext_add_indexed_bias(with_bias, table, indices);
    CHECK(ggml_flash_attn_ext_has_indexed_bias(with_bias) && ggml_fork_fa_sparse_min(with_bias) == 2048);

    if (g_fails == 0) {
        // the CPU runs a gated attention as the plain one (with the gate in the bias flag it took the bias path, no table: a crash)
        ggml_tensor * plain = make();
        ggml_tensor * gated = make();
        ggml_flash_attn_ext_set_sparse_min(gated, 1);
        const std::vector<float> a = compute(ctx, plain), b = compute(ctx, gated);
        CHECK(a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
        bool finite = true;
        for (float x : b) { finite = finite && std::isfinite(x); }
        CHECK(finite);
    }
}

// MUL_MAT_ID: the zero-first flag, the expert range, the MoE cache's absent value; and the MUL_MAT slice a backend cuts from it
static void test_mmid(ggml_context * ctx) {
    g_covered.insert("mmid_neg_ids"); g_covered.insert("mmid_range_lo"); g_covered.insert("mmid_range_hi"); g_covered.insert("mmid_cache_absent");
    const int K = 32, N = 16, N_AS = 4, N_USED = 2, T = 3;
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, N_AS);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N_USED, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, T);
    fill(as, 1.0f); fill(b, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(ids); ++i) { ((int32_t *) ids->data)[i] = (int32_t) ((i*3 + 1) % N_AS); }

    struct setter { const char * name; int slot, width; void (*set)(ggml_tensor *); bool (*read_back)(const ggml_tensor *); };
    const setter setters[] = {
        { "mmid_neg_ids",      GGML_FORK_SLOT_MMID_NEG_IDS,      1, [](ggml_tensor * t) { ggml_fork_mmid_set_neg_ids(t); },
                                                                    [](const ggml_tensor * t) { return ggml_fork_mmid_neg_ids(t) == 1; } },
        { "mmid_range",        GGML_FORK_SLOT_MMID_RANGE_LO,     2, [](ggml_tensor * t) { ggml_fork_mmid_set_range(t, 1, 3); },
                                                                    [](const ggml_tensor * t) { return ggml_fork_mmid_range_lo(t) == 1 && ggml_fork_mmid_range_hi(t) == 3; } },
        { "mmid_cache_absent", GGML_FORK_SLOT_MMID_CACHE_ABSENT, 1, [](ggml_tensor * t) { ggml_fork_mmid_set_cache_absent(t, 37); },
                                                                    [](const ggml_tensor * t) { return ggml_fork_mmid_cache_absent(t) == 37; } },
    };
    static_assert(GGML_FORK_SLOT_MMID_RANGE_HI == GGML_FORK_SLOT_MMID_RANGE_LO + 1, "the range is two adjacent slots");

    for (const setter & s : setters) {
        ggml_tensor * t = ggml_mul_mat_id(ctx, as, b, ids);
        int32_t before[GGML_MAX_OP_PARAMS/sizeof(int32_t)];
        memcpy(before, t->op_params, sizeof(before));
        const enum ggml_prec acc0 = ggml_prec_get_acc(t), src0 = ggml_prec_get_src(t, 1);
        s.set(t);
        // upstream's accessors: the accumulation precision (slot 0, the cache's absent value until 2026-11-26) and the src1 precision (slot 3,
        // the expert range until the move to 8 and 9)
        if (ggml_prec_get_acc(t) != acc0 || ggml_prec_get_src(t, 1) != src0) { fprintf(stderr, "  %s\n", s.name); }
        CHECK(ggml_prec_get_acc(t) == acc0 && ggml_prec_get_src(t, 1) == src0);
        CHECK(others_unchanged(before, t, s.slot, s.width));
        CHECK(s.read_back(t));
        // upstream's setters leave it alone
        ggml_prec_set_acc(t, GGML_PREC_F32);
        ggml_prec_set_src(t, GGML_PREC_BF16, 1);
        CHECK(ggml_prec_get_acc(t) == GGML_PREC_F32 && ggml_prec_get_src(t, 1) == GGML_PREC_BF16);
        CHECK(s.read_back(t));
    }

    // every flag at once on one matmul, then the MUL_MAT slice of it (ggml_cuda_mul_mat_id: the tensor copied, op MUL_MAT): on a MUL_MAT slot 1
    // is upstream's hint (GGML_HINT_SRC0_IS_HADAMARD == 1 ran the FWHT with no src[0]: segfault, 2026-11-24); the precision must carry over
    ggml_tensor * t = ggml_mul_mat_id(ctx, as, b, ids);
    for (const setter & s : setters) { s.set(t); }
    ggml_prec_set_acc(t, GGML_PREC_F32);
    ggml_tensor slice = *t;
    slice.op = GGML_OP_MUL_MAT;
    ggml_fork_mm_slice_params(&slice);
    CHECK(slice.op_params[1] == GGML_HINT_NONE);   // read so by ggml_compute_forward_mul_mat, ggml_cuda_op_mul_mat_use_fwht, ggml_vk_can_use_fwht
    CHECK(ggml_prec_get_acc(&slice) == GGML_PREC_F32);
    for (const setter & s : setters) { CHECK(s.read_back(t)); }

    if (g_fails == 0) {
        // the CPU's product with every flag set equals the plain one (all ids valid, no cache table)
        ggml_tensor * plain = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_tensor * flagged = ggml_mul_mat_id(ctx, as, b, ids);
        for (const setter & s : setters) { s.set(flagged); }
        const std::vector<float> x = compute(ctx, plain), y = compute(ctx, flagged);
        CHECK(x.size() == y.size() && memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0);
    }
}

// the KQ mask input: the cells of its graph (llama-graph.cpp build_attn_inp_kq_mask, can_reuse_kq_mask)
static void test_mask(ggml_context * ctx) {
    g_covered.insert("mask_n_kv");
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 8, 1, 1);
    ggml_set_input(m);
    int32_t before[GGML_MAX_OP_PARAMS/sizeof(int32_t)];
    memcpy(before, m->op_params, sizeof(before));
    ggml_fork_mask_set_n_kv(m, 1234);
    CHECK(m->op == GGML_OP_NONE && (m->flags & GGML_TENSOR_FLAG_INPUT));
    CHECK(others_unchanged(before, m, GGML_FORK_SLOT_MASK_N_KV, 1));
    CHECK(ggml_fork_mask_n_kv(m) == 1234);
}

// MOE_BELL, MOE_ROWS (fork ops): the bell pointer
static void test_bell(ggml_context * ctx) {
    g_covered.insert("bell_ptr");
    static int bell_dummy;
    void * bell = &bell_dummy;
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
    ggml_tensor * x     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * down  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    for (ggml_tensor * t : { ggml_moe_bell(ctx, ids, table, x, bell), ggml_moe_rows(ctx, down, ids, slots, bell) }) {
        void * p;
        memcpy(&p, &t->op_params[GGML_FORK_SLOT_BELL_PTR], sizeof(p));   // as ggml-cpu.c, moe-bell.cu read it
        CHECK(p == bell);
    }
}

int main() {
    ggml_init_params ip = { 64u*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    test_fa_sparse_min(ctx);
    test_mmid(ctx);
    test_mask(ctx);
    test_bell(ctx);

    // every slot in the registry has a case above
    int n = 0;
    const ggml_fork_op_slot * slots = ggml_fork_op_slots(&n);
    for (int i = 0; i < n; ++i) {
        if (!g_covered.count(slots[i].name)) { fprintf(stderr, "FAIL: registry slot %s (%s %d) has no case in this test\n", slots[i].name, ggml_op_name(slots[i].op), slots[i].slot); ++g_fails; }
    }

    ggml_free(ctx);
    printf("%s: %d registry slots, %d failure(s)\n", g_fails ? "FAIL" : "OK", n, g_fails);
    return g_fails ? 1 : 0;
}
