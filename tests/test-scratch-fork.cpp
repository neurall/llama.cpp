// GPU test of the CUDA scratch the expert pool lends a graph (ggml_backend_cuda_node_scratch_fork: ggml-cuda.cu, mmq.cu, fattn-common.cuh, top-k.cu, argsort.cu).
// The pool computes what a graph's ops take from the scratch region BEFORE the graph runs and lends exactly that; a wrong number is either an out-of-memory in the first pass (too small: the rest comes
// from the device pool beside the pool, DeepSeek 4.1 edit cell, 2026-10-27) or lent blocks that experts could hold (too big: the dense split lent cuda0x4 the
// scratch of the whole batch).
//
// This test RUNS the real ops on every CUDA card with the scratch region set and compares the high-water mark the backend reports (what the op really took, in its
// order, with the 256-byte start of every allocation) with what the function computes from the node alone. They must be EQUAL: used = required.
//  - MUL_MAT_ID: every weight type, a range of shapes and token counts (every path: MMVQ, MMQ, MMF, the sorting path). The sorting path multiplies one expert at a
//    time, so what it takes depends on the routing, which is not known before the graph runs: the function returns the most any routing takes. Routing every token
//    to the same experts reaches it (equal, on every card where a busier expert never takes less: all NVIDIA cards since Turing); a spread routing stays at or below.
//  - MUL_MAT: the same types through the dense dispatch (MMVQ, MMQ, cuBLAS with its conversions, batched pointers and chunks), which the sorting path runs per expert.
//  - FLASH_ATTN_EXT (the kernel choice and its launch run in plan mode: parallel blocks, stream-k fixup, mask scan, sparse gather), TOP_K and ARGSORT (CUB's own
//    size queries).
// Whoever changes an op's allocations without its mirror fails here, on the first GPU run.
//
// build: cmake --build <dir> --target test-scratch-fork;  run: ctest -R test-scratch-fork   or   test-scratch-fork [card]   (no CUDA card: skipped, exit 0)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

using scratch_t = size_t (*)(int, void *, size_t, ggml_backend_t);   // device, region, capacity, the backend the region serves (nullptr: any)
using node_t    = size_t (*)(int, const ggml_tensor *);
using nodes_t   = size_t (*)(int, const ggml_tensor * const *, int);
static scratch_t scratch_fn = nullptr;
static ggml_backend_t g_owner = nullptr;   // run_op lends the region to this backend instead of the one running the op (the owner case)
static node_t    node_fn    = nullptr;
static nodes_t   nodes_fn   = nullptr;

static int n_failed = 0, n_run = 0, n_skipped = 0;

// the pool asks the scheduler for the most scratch the nodes of a card take; the scheduler keeps each split's value until it splits again (ggml-backend.cpp)
GGML_API size_t ggml_backend_sched_node_max_dev(ggml_backend_sched_t sched, ggml_backend_dev_t dev, size_t (*fn)(void * user, struct ggml_tensor * const * nodes, int n),
        size_t (*copy_fn)(ggml_backend_t src_backend, ggml_backend_t dst_backend, const struct ggml_tensor * src, const struct ggml_tensor * dst), void * user);
using copy_t = size_t (*)(ggml_backend_t, ggml_backend_t, const ggml_tensor *, const ggml_tensor *);
static copy_t copy_fn = nullptr;
static int n_asked = 0;
static size_t count_nodes_fn(void * user, struct ggml_tensor * const * nodes, int n) {
    n_asked++;
    return nodes_fn(*(const int *) user, nodes, n);
}

// Builds one op in a fresh context (build returns the output and sets *ids for MUL_MAT_ID), uploads the ids and runs it with the region set: asked = what it took,
// computed = what the backend says it takes. The weights and activations stay as allocated: what the op takes depends on shapes and types only.
static bool run_op(ggml_backend_t be, int d, void * base, size_t cap, const std::function<ggml_tensor *(ggml_context *, ggml_tensor **)> & build,
                   const std::vector<int32_t> & ids_v, size_t & asked, size_t & computed) {
    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ids = nullptr;
    ggml_tensor * y   = build(ctx, &ids);
    if (!ggml_backend_supports_op(be, y)) {   // e.g. a flash attention shape no CUDA kernel takes
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    if (ids) {
        ggml_backend_tensor_set(ids, ids_v.data(), 0, ggml_nbytes(ids));
    }
    ggml_backend_t owner = g_owner ? g_owner : be;
    scratch_fn(d, base, cap, owner);                // set the region, clear the mark
    ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    asked    = scratch_fn(d, base, cap, owner);     // the mark of that one op: what it took, fitting or not
    computed = node_fn(d, y);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("test-scratch-fork: no CUDA card, skipped\n");
        return 0;
    }
    scratch_fn = (scratch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_scratch_fork");
    node_fn    = (node_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_node_scratch_fork");
    nodes_fn   = (nodes_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nodes_scratch_fork");
    copy_fn    = (copy_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_scratch_fork");
    if (!scratch_fn || !node_fn || !nodes_fn || !copy_fn) {
        printf("test-scratch-fork: FAILED: the CUDA backend does not export the scratch functions\n");
        return 1;
    }
    const int only = argc > 1 ? atoi(argv[1]) : -1;

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_M,
                                GGML_TYPE_MXFP4, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32 };
    struct shape { int64_t ne00, ne01, n_expert, n_used; };
    const shape shapes[] = { { 2048, 1024, 16, 4 }, { 5120, 2304, 32, 6 }, { 4096, 1536, 64, 8 } };   // input width, rows, experts, used (DeepSeek 4.1's real width is 5120 x 2304)
    const int64_t tokens[] = { 1, 2, 4, 8, 17, 64, 130, 512, 2049 };   // decode batches to a prompt pass: every MUL_MAT_ID path
    const int64_t cols[]   = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1000, 2049 };   // around every dense kernel switch

    for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) {
        if (only >= 0 && (int) d != only) { continue; }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, d);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { continue; }
        // the region the pool would lend: big enough for every case, so nothing spills (the high-water mark counts what was asked for anyway)
        const size_t cap = (size_t) 3 << 30;
        ggml_backend_buffer_t region = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), cap);
        if (!region) { printf("card %zu: no memory for the scratch region, skipped\n", d); ggml_backend_free(be); continue; }
        void * base = ggml_backend_buffer_get_base(region);
        printf("Testing the scratch the matmuls take against what is computed, card %zu %s:\n", d, ggml_backend_dev_name(dev));
        int bad_card = 0;
        char what[256];
        auto check = [&](size_t asked, size_t computed, bool equal) {
            n_run++;
            if (equal ? asked == computed : asked <= computed) { return; }
            n_failed++;
            if (++bad_card <= 40) {
                printf("  FAILED %s: the op took %zu bytes, computed %zu (%+lld)\n", what, asked, computed, (long long) computed - (long long) asked);
            }
        };
        const std::vector<int32_t> no_ids;

        for (ggml_type type : types) {
            for (const shape & sh : shapes) {
                if (sh.ne00 % ggml_blck_size(type) != 0) { n_skipped++; continue; }
                for (int64_t nt : tokens) {
                    for (int64_t ne11 : { (int64_t) 1, sh.n_used }) {   // gate and up read one activation row per token, down one per used expert
                        for (int routing = 0; routing < 2; ++routing) {   // 0: every token to experts 0..n_used-1 (an expert gets every token), 1: spread
                            std::vector<int32_t> iv((size_t) (sh.n_used * nt));
                            for (size_t i = 0; i < iv.size(); ++i) {
                                iv[i] = routing == 0 ? (int32_t) (i % sh.n_used) : (int32_t) ((i * 7 + i / sh.n_used) % sh.n_expert);
                            }
                            size_t asked = 0, computed = 0;
                            if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor ** ids) {
                                    ggml_tensor * w = ggml_new_tensor_3d(ctx, type, sh.ne00, sh.ne01, sh.n_expert);
                                    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.ne00, ne11, nt);
                                    *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sh.n_used, nt);
                                    return ggml_mul_mat_id(ctx, w, x, *ids);
                                }, iv, asked, computed)) { n_skipped++; continue; }
                            snprintf(what, sizeof(what), "MUL_MAT_ID %-8s %5lld x %-5lld %3lld experts %lld used, %5lld tokens, %lld activation rows, %s routing",
                                     ggml_type_name(type), (long long) sh.ne00, (long long) sh.ne01, (long long) sh.n_expert, (long long) sh.n_used, (long long) nt,
                                     (long long) ne11, routing == 0 ? "same-expert" : "spread");
                            check(asked, computed, routing == 0);
                        }
                    }
                }
                for (int64_t t : cols) {   // the dense dispatch on the same weight shape (one expert's matrix)
                    size_t asked = 0, computed = 0;
                    if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                            return ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, sh.ne00, sh.ne01), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.ne00, t));
                        }, no_ids, asked, computed)) { n_skipped++; continue; }
                    snprintf(what, sizeof(what), "MUL_MAT %-8s %5lld x %-5lld, %5lld columns", ggml_type_name(type), (long long) sh.ne00, (long long) sh.ne01, (long long) t);
                    check(asked, computed, true);
                }
            }
        }
        // the region serves only the context it is lent to: another context on the same card (a DSpark draft between two target steps) takes its scratch from
        // the device pool and adds nothing to the region's mark (2026-10-27: the draft's asks were taken from, and counted against, the target's lend)
        if (ggml_backend_t other = ggml_backend_dev_init(dev, nullptr)) {
            size_t asked = 0, computed = 0;
            g_owner = other;
            const bool ran = run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                    return ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 2048, 1024), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 8));
                }, no_ids, asked, computed);
            g_owner = nullptr;
            if (ran) {
                snprintf(what, sizeof(what), "MUL_MAT Q4_0 2048 x 1024, 8 columns, the region lent to another context on the card");
                check(asked, 0, true);
                n_run++;
                if (computed == 0) { n_failed++; printf("  FAILED: the owner case needs an op that takes scratch (computed 0)\n"); }
            } else {
                n_skipped++;
            }
            ggml_backend_free(other);
        }
        // cuBLAS with what the cases above do not reach: a broadcast batch (attention's KQ: pointer arrays), f16 activations (no conversion of either input), and
        // a weight whose f32 copy is over the conversion limit (precision f32: converted in chunks, one after another)
        for (int64_t t : cols) {
            size_t asked = 0, computed = 0;
            if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                    return ggml_mul_mat(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 512, 4), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, t, 16));
                }, no_ids, asked, computed)) {
                snprintf(what, sizeof(what), "MUL_MAT f16 128 x 512 x 4 heads, f32 %lld columns x 16 heads (broadcast)", (long long) t);
                check(asked, computed, true);
            } else { n_skipped++; }
            if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                    return ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 512), ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, t));
                }, no_ids, asked, computed)) {
                snprintf(what, sizeof(what), "MUL_MAT f16 256 x 512, f16 %lld columns", (long long) t);
                check(asked, computed, true);
            } else { n_skipped++; }
        }
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
            for (int64_t t : { (int64_t) 1, (int64_t) 9, (int64_t) 64 }) {
                size_t asked = 0, computed = 0;
                if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                        ggml_tensor * y = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, 8192, 16416), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, t));
                        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
                        return y;
                    }, no_ids, asked, computed)) { n_skipped++; continue; }
                snprintf(what, sizeof(what), "MUL_MAT %s 8192 x 16416 precision f32, %lld columns (chunks)", ggml_type_name(type), (long long) t);
                check(asked, computed, true);
            }
        }
        // flash attention: decode (vector kernel, parallel blocks) to prompt passes (tensor-core kernel, stream-k fixup), head sizes of the models in use
        // (DeepSeek 4.1: 512 x 64 heads over one shared K/V head), grouped queries, f32 precision, quantized K/V, the mask scan and the sparse gather
        struct fa_shape { int64_t dk, dv, n_head, n_head_kv; ggml_type kv; bool f32_prec; int32_t n_kv_max; };
        const fa_shape fa_shapes[] = {
            { 128, 128, 32,  8, GGML_TYPE_F16,  false,   0 },
            { 128, 128, 32,  8, GGML_TYPE_F16,  true,    0 },
            {  64,  64, 16, 16, GGML_TYPE_F16,  false,   0 },
            { 256, 256, 16,  2, GGML_TYPE_F16,  false,   0 },
            { 128, 128, 32,  8, GGML_TYPE_Q8_0, false,   0 },
            { 512, 512, 64,  1, GGML_TYPE_F16,  false,   0 },
            { 512, 512, 64,  1, GGML_TYPE_F16,  false, 512 },
            { 576, 512, 16,  1, GGML_TYPE_F16,  false,   0 },
        };
        for (const fa_shape & fs : fa_shapes) {
            for (int64_t n_kv : { (int64_t) 256, (int64_t) 4096, (int64_t) 16384 }) {
                for (int64_t n_q : { (int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 27, (int64_t) 64, (int64_t) 512, (int64_t) 1024, (int64_t) 3389 }) {
                    size_t asked = 0, computed = 0;
                    if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, fs.dk, n_q, fs.n_head, 1);
                            ggml_tensor * k = ggml_new_tensor_4d(ctx, fs.kv, fs.dk, n_kv, fs.n_head_kv, 1);
                            ggml_tensor * v = ggml_new_tensor_4d(ctx, fs.kv, fs.dv, n_kv, fs.n_head_kv, 1);
                            ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
                            ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, m, 0.125f, 0.0f, 0.0f);
                            if (fs.f32_prec) { ggml_prec_set_acc(y, GGML_PREC_F32); }
                            if (fs.n_kv_max > 0) { ggml_flash_attn_ext_set_n_kv_max(y, fs.n_kv_max); }
                            return y;
                        }, no_ids, asked, computed)) { n_skipped++; continue; }
                    snprintf(what, sizeof(what), "FLASH_ATTN_EXT K %lld V %lld, %lld heads over %lld, %s K/V%s%s, %lld cells, %lld queries", (long long) fs.dk,
                             (long long) fs.dv, (long long) fs.n_head, (long long) fs.n_head_kv, ggml_type_name(fs.kv), fs.f32_prec ? ", f32 precision" : "",
                             fs.n_kv_max > 0 ? ", sparse" : "", (long long) n_kv, (long long) n_q);
                    check(asked, computed, true);
                }
            }
        }
        // top-k (router picks, the sparse attention's index selection) and argsort: the bitonic sort, the radix select, CUB
        for (int64_t ncols : { (int64_t) 32, (int64_t) 256, (int64_t) 512, (int64_t) 1000, (int64_t) 3389, (int64_t) 5000, (int64_t) 16384, (int64_t) 65536 }) {
            for (int64_t nrows : { (int64_t) 1, (int64_t) 2, (int64_t) 27, (int64_t) 82, (int64_t) 100, (int64_t) 3389 }) {
                for (int64_t k : { (int64_t) 8, (int64_t) 512 }) {
                    if (k > ncols) { continue; }
                    size_t asked = 0, computed = 0;
                    if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                            return ggml_top_k(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows), (int) k);
                        }, no_ids, asked, computed)) { n_skipped++; continue; }
                    snprintf(what, sizeof(what), "TOP_K %lld of %lld columns, %lld rows", (long long) k, (long long) ncols, (long long) nrows);
                    check(asked, computed, true);
                }
                for (ggml_sort_order order : { GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC }) {
                    if (ncols * nrows > ((int64_t) 1 << 26)) { continue; }   // the test's own buffers, not the op's
                    size_t asked = 0, computed = 0;
                    if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                            return ggml_argsort(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows), order);
                        }, no_ids, asked, computed)) { n_skipped++; continue; }
                    snprintf(what, sizeof(what), "ARGSORT %s %lld columns, %lld rows", order == GGML_SORT_ORDER_ASC ? "ascending" : "descending", (long long) ncols, (long long) nrows);
                    check(asked, computed, true);
                }
            }
        }
        // the other ops that take scratch: softmax (only its cooperative path: few long rows, no mask), sum and mean (CUB over long rows), cumsum (CUB scans),
        // the triangular solve (cuBLAS path: pointer arrays) and the Mamba-2 scan (the SSD path for long sequences: seven work buffers)
        for (int64_t ncols : { (int64_t) 1000, (int64_t) 5000, (int64_t) 30000, (int64_t) 50000, (int64_t) 70000, (int64_t) 200000 }) {
            for (int64_t nrows : { (int64_t) 1, (int64_t) 3 }) {
                size_t asked = 0, computed = 0;
                auto mat = [&](ggml_context * ctx) { return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows); };
                if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) { return ggml_soft_max(ctx, mat(ctx)); }, no_ids, asked, computed)) {
                    snprintf(what, sizeof(what), "SOFT_MAX %lld columns, %lld rows", (long long) ncols, (long long) nrows);
                    check(asked, computed, true);
                } else { n_skipped++; }
                if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) { return ggml_sum(ctx, mat(ctx)); }, no_ids, asked, computed)) {
                    snprintf(what, sizeof(what), "SUM %lld x %lld", (long long) ncols, (long long) nrows);
                    check(asked, computed, true);
                } else { n_skipped++; }
                if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) { return ggml_mean(ctx, mat(ctx)); }, no_ids, asked, computed)) {
                    snprintf(what, sizeof(what), "MEAN %lld columns, %lld rows", (long long) ncols, (long long) nrows);
                    check(asked, computed, nrows > 1 || ncols <= 32768 || ncols > 65536);   // in between the path depends on the CUDA graph state: the larger
                } else { n_skipped++; }
                if (run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) { return ggml_cumsum(ctx, mat(ctx)); }, no_ids, asked, computed)) {
                    snprintf(what, sizeof(what), "CUMSUM %lld columns, %lld rows", (long long) ncols, (long long) nrows);
                    check(asked, computed, true);
                } else { n_skipped++; }
            }
        }
        for (int64_t n : { (int64_t) 32, (int64_t) 64, (int64_t) 100 }) {
            for (int64_t k : { (int64_t) 16, (int64_t) 64 }) {
                for (int64_t nb : { (int64_t) 1, (int64_t) 12 }) {
                    size_t asked = 0, computed = 0;
                    if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor **) {
                            return ggml_solve_tri(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, n, nb), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, nb), true, true, false);
                        }, no_ids, asked, computed)) { n_skipped++; continue; }
                    snprintf(what, sizeof(what), "SOLVE_TRI %lld x %lld, %lld columns, %lld batches", (long long) n, (long long) n, (long long) k, (long long) nb);
                    check(asked, computed, true);
                }
            }
        }
        for (int64_t n_tok : { (int64_t) 64, (int64_t) 129, (int64_t) 600, (int64_t) 2048 }) {   // the scan path, then the SSD path: one odd chunk, a partial second chunk, aligned chunks
            for (int64_t n_seqs : { (int64_t) 1, (int64_t) 2 }) {
                const int64_t d_state = 128, head_dim = 64, n_head = 8, n_group = 1;
                size_t asked = 0, computed = 0;
                std::vector<int32_t> seq_ids((size_t) n_seqs);
                for (int64_t i = 0; i < n_seqs; ++i) { seq_ids[(size_t) i] = (int32_t) i; }
                if (!run_op(be, (int) d, base, cap, [&](ggml_context * ctx, ggml_tensor ** pid) {
                        ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                        ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs);
                        ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok, n_seqs);
                        ggml_tensor * A   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head);   // Mamba-2: one A per head
                        ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok, n_seqs);
                        ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok, n_seqs);
                        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                        *pid = ids;   // the sequences' state slots: uploaded by run_op
                        return ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, 1);
                    }, seq_ids, asked, computed)) { n_skipped++; continue; }
                snprintf(what, sizeof(what), "SSM_SCAN Mamba-2 %lld tokens, %lld sequences", (long long) n_tok, (long long) n_seqs);
                check(asked, computed, true);
            }
        }
        // what the computation costs: the pool asks it for every node of every graph (a decode step too), so each call must stay cheap. Printed, not checked.
        {
            struct ggml_init_params ip = { 32 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 1, 64, 1);
            ggml_tensor * kv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 4096, 1, 1);
            ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4096, 1, 1, 1);
            ggml_tensor * w  = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, 5120, 2304, 32);
            ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5120, 1, 1);
            ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, 1);
            const struct { const char * name; ggml_tensor * t; } nodes[] = {
                { "FLASH_ATTN_EXT 512 x 64 heads, 4096 cells, 1 query", ggml_flash_attn_ext(ctx, q, kv, kv, m, 0.125f, 0.0f, 0.0f) },
                { "TOP_K 512 of 4096, 1 row",                           ggml_top_k(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1), 512) },
                { "MUL_MAT q4_K 5120 x 2304, 1 column",                 ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 5120, 2304), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1)) },
                { "MUL_MAT_ID iq2_xxs 5120 x 2304 x 32, 1 token",       ggml_mul_mat_id(ctx, w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5120, 1, 1), id) },
                { "ADD (no scratch)",                                   ggml_add(ctx, x, x) },
            };
            std::vector<const ggml_tensor *> all;
            size_t most = 0;
            for (const auto & nd : nodes) {
                const int n = 2000;
                const int64_t t0 = ggml_time_us();
                size_t sink = 0;
                for (int i = 0; i < n; ++i) { sink += node_fn((int) d, nd.t); }
                printf("  cost: %-52s %6.2f us per call (%zu bytes)\n", nd.name, (ggml_time_us() - t0) / (double) n, sink / n);
                all.push_back(nd.t);
                most = std::max(most, node_fn((int) d, nd.t));
            }
            {   // the pool's way: a split's nodes at once, the device set once per call; the most of them must equal the most of the single calls
                const int n = 2000;
                const int64_t t0 = ggml_time_us();
                size_t got = 0;
                for (int i = 0; i < n; ++i) { got = nodes_fn((int) d, all.data(), (int) all.size()); }
                printf("  cost: %-52s %6.2f us per node (%zu bytes)\n", "the nodes above in one call", (ggml_time_us() - t0) / ((double) n * all.size()), got);
                snprintf(what, sizeof(what), "the nodes above in one call");
                check(got, most, true);
            }
            ggml_free(ctx);
        }
        // the scheduler keeps the value per split until it splits again (split uid): the same graph asked twice costs one call per split, a new split of other
        // shapes is computed again. A stale value is a lend too small for the graph that runs.
        {
            auto expect = [&](bool ok, const char * msg) { n_run++; if (!ok) { n_failed++; bad_card++; printf("  FAILED scheduler: %s\n", msg); } };
            ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            ggml_backend_t bes[2] = { be, cpu };
            ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, 2, 64, false, false);
            struct ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * wctx = ggml_init(wp);
            ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_K, 5120, 2304);
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);   // the weight on the card: the matmul runs there
            ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            int user = (int) d;
            size_t first = 0;
            for (int64_t ncols : { (int64_t) 1, (int64_t) 512 }) {
                struct ggml_init_params gp = { 4 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * gctx = ggml_init(gp);
                ggml_tensor * y = ggml_mul_mat(gctx, w, ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 5120, ncols));
                ggml_cgraph * gf = ggml_new_graph(gctx);
                ggml_build_forward_expand(gf, y);
                ggml_backend_sched_reset(sched);
                expect(ggml_backend_sched_alloc_graph(sched, gf), "the graph was not allocated");
                const int    n0 = n_asked;
                const size_t v1 = ggml_backend_sched_node_max_dev(sched, dev, count_nodes_fn, copy_fn, &user);
                const int    n1 = n_asked;
                const size_t v2 = ggml_backend_sched_node_max_dev(sched, dev, count_nodes_fn, copy_fn, &user);
                expect(n1 > n0,            "a new split was not computed");
                expect(n_asked == n1,      "the same split was computed again");
                expect(v1 == v2,           "the kept value differs from the computed one");
                expect(v1 == node_fn((int) d, y), "the split's value is not its node's scratch");
                if (ncols == 1) { first = v1; } else { expect(v1 != first, "a new split of other shapes kept the old value"); }
                ggml_free(gctx);
            }
            ggml_backend_sched_free(sched);
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(cpu);
        }
        scratch_fn((int) d, nullptr, 0, nullptr);   // give the region back
        ggml_backend_buffer_free(region);
        ggml_backend_free(be);
        printf("  card %zu: %s\n", d, bad_card ? "FAILED" : "OK");
    }
    // the scheduler's input copies between cards: a gappy view on card 0 (1 KB of every 4 KB row, like a layer's K inside the token-major KV record) read by a
    // node on card 1 is gathered into a compact block on card 0 first (ggml_backend_cuda_cpy_tensor_async). Card 0 runs no op, so what it took is that block;
    // the scheduler must compute the same for card 0, and nothing for card 1 (its copy of a compact view takes nothing).
    if (ggml_backend_reg_dev_count(reg) >= 2 && only < 0) {
        ggml_backend_dev_t dv[2] = { ggml_backend_reg_dev_get(reg, 0), ggml_backend_reg_dev_get(reg, 1) };
        ggml_backend_t bk[3] = { ggml_backend_dev_init(dv[0], nullptr), ggml_backend_dev_init(dv[1], nullptr), ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr) };
        const size_t rcap = (size_t) 64 << 20;
        ggml_backend_buffer_t rg[2];
        void * rb[2];
        for (int c = 0; c < 2; ++c) {
            rg[c] = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dv[c]), rcap);
            rb[c] = ggml_backend_buffer_get_base(rg[c]);
        }
        ggml_backend_sched_t sched = ggml_backend_sched_new(bk, nullptr, 3, 64, false, false);
        struct ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        ggml_tensor * rec = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1024, 4096);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, bk[0]);
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        struct ggml_init_params gp = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * view = ggml_view_2d(gctx, rec, 256, 4096, rec->nb[1], 0);
        ggml_tensor * y    = ggml_cont(gctx, view);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_sched_set_tensor_backend(sched, y, bk[1]);   // the reader on card 1
        size_t computed[2] = { 0, 0 }, took[2] = { 0, 0 };
        const bool alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
        if (alloc_ok) {
            for (int c = 0; c < 2; ++c) {
                int user = c;
                computed[c] = ggml_backend_sched_node_max_dev(sched, dv[c], count_nodes_fn, copy_fn, &user);
                scratch_fn(c, rb[c], rcap, bk[c]);   // set the region, clear the mark
            }
            ggml_backend_sched_graph_compute(sched, gf);
            ggml_backend_sched_synchronize(sched);
            for (int c = 0; c < 2; ++c) {
                took[c] = scratch_fn(c, rb[c], rcap, bk[c]);
                scratch_fn(c, nullptr, 0, nullptr);
            }
        }
        const size_t block = ggml_row_size(GGML_TYPE_F32, 256) * 4096;
        const bool ok = alloc_ok && took[0] == computed[0] && took[1] == computed[1] && computed[0] == block;
        n_run++;
        if (!ok) { n_failed++; }
        printf("Testing a gappy view copied from card 0 to card 1: card 0 took %zu, computed %zu (the block is %zu); card 1 took %zu, computed %zu: %s\n",
               took[0], computed[0], block, took[1], computed[1], ok ? "OK" : "FAILED");
        ggml_free(gctx);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        for (int c = 0; c < 2; ++c) { ggml_backend_buffer_free(rg[c]); }
        for (ggml_backend_t b : bk) { ggml_backend_free(b); }
    }
    printf("test-scratch-fork: %d cases run, %d skipped, %d failed\n", n_run, n_skipped, n_failed);
    return n_failed ? 1 : 0;
}
