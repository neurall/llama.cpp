#pragma once
// fork: may a separate draft model (-md) load beside this target? Not when the target is bigger than the free VRAM: then the expert pool takes the
// cards (the pool is the whole VRAM) and the draft's compute buffer has no place in it yet, so its first decode found no VRAM: Qwen3.8 Next IQ4_XS on
// two 3090s (1.9x free VRAM) + its MTP draft aborted at load, 2026-11-25 ("VRAM outside the block pool: CUDA1 1264.69 MiB", GGML_ASSERT(buffer)).
// The draft inside the pool is the next step (draft-in-pool); --spec-draft-n-max N still forces the draft. Built-in MTP (no -md) is not decided here.
// ratio = target file size / free VRAM of the cards it may use. test: tests/test-spec-guard-fork.cpp
static inline bool spec_draft_refused_fork(double ratio, bool has_dft, bool n_max_user) {
    return has_dft && !n_max_user && ratio > 1.0;
}
