// Unit test of the CUDA rows copy pitch (ggml/src/ggml-cuda/rows-copy-fork.h). No GPU: tensors in a no_alloc context.
// build: cmake --build <dir> --target test-rows-copy-fork;  run: ctest -R test-rows-copy-fork
#include "../ggml/src/ggml-cuda/rows-copy-fork.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // Inkling 2 + DSpark draft on 2 cards (2026-11-26): a one-row view of a [4096 x 64] f32 tensor, its nb[1] taken from a parent whose rows are wider
    ggml_tensor * parent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, 64);
    ggml_tensor * one    = ggml_view_2d(ctx, parent, 4096, 1, parent->nb[1], 0);
    ggml_tensor * dst1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
    CHECK(ggml_cuda_rows_pitch(one) == ggml_row_size(GGML_TYPE_F32, 4096));   // own width, not the parent's nb[1]
    CHECK(ggml_cuda_rows_copy_ok(one, dst1));
    // a single row whose nb[1] is below its width (a view laid over the next row) still copies: its pitch is its own width
    ggml_tensor * narrow = ggml_view_2d(ctx, parent, 4096, 1, 16, 0);
    CHECK(ggml_cuda_rows_pitch(narrow) == ggml_row_size(GGML_TYPE_F32, 4096));
    CHECK(ggml_cuda_rows_copy_ok(narrow, dst1));
    // several gappy rows keep nb[1] as the pitch
    ggml_tensor * rows = ggml_view_2d(ctx, parent, 4096, 64, parent->nb[1], 0);
    ggml_tensor * dst  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 64);
    CHECK(ggml_cuda_rows_pitch(rows) == parent->nb[1]);
    CHECK(ggml_cuda_rows_copy_ok(rows, dst));
    // rows that overlap (pitch below the width) are no rows copy
    ggml_tensor * ovl = ggml_view_2d(ctx, parent, 4096, 2, 16, 0);
    ggml_tensor * dst2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 2);
    CHECK(!ggml_cuda_rows_copy_ok(ovl, dst2));
    // same layout: a plain copy, no rows copy
    CHECK(!ggml_cuda_rows_copy_ok(dst, dst));
    ggml_free(ctx);
    if (g_fail) { fprintf(stderr, "test-rows-copy-fork: %d failed\n", g_fail); return 1; }
    printf("test-rows-copy-fork: ok\n");
    return 0;
}
