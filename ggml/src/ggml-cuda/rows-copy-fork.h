#pragma once
// test: tests/test-rows-copy-fork.cpp (no GPU)
#include "ggml.h"
#include "../ggml-impl.h"   // ggml_are_same_layout

// fork: a gappy source view copied as rows (one 2D copy a plane) into a compact destination. The source pitch is nb[1], except for a single row,
// whose nb[1] can be anything: cudaMemcpy2D rejects a pitch below the row width ("invalid pitch argument", Inkling 2 + DSpark draft on 2 cards,
// 2026-11-26), so one row uses its own width; rows that overlap (pitch below the width) are no rows copy, the scheduler copies them plainly
static inline size_t ggml_cuda_rows_pitch(const ggml_tensor * src) {
    return src->ne[1] > 1 ? src->nb[1] : ggml_row_size(src->type, src->ne[0]);
}

static inline bool ggml_cuda_rows_copy_ok(const ggml_tensor * src, const ggml_tensor * dst) {
    return !ggml_are_same_layout(src, dst) && src->type == dst->type && ggml_are_same_shape(src, dst) &&
           src->nb[0] == dst->nb[0] && ggml_is_contiguous(dst) && ggml_cuda_rows_pitch(src) >= ggml_row_size(src->type, src->ne[0]);
}
