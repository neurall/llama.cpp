#pragma once
// fork: when fit's memory measure of an extra model (a draft) fails, may it retry against a dry target context?
// A draft that borrows the target's embeddings/head (Qwen3.8 MTP, DSpark, DFlash, EAGLE3) cannot make a context alone.
// llama_init_from_model logs that error ("... requires ctx_other ...") and returns null, so fit throws FIT_ERR_NO_CTX;
// a check for "requires ctx_other" alone never fired (2026-11-25: Qwen3.8 MTP and Inkling DSpark drafts counted 0 MiB).
// test: tests/test-fit-retry-fork.cpp

#include <cstring>

#define FIT_ERR_NO_CTX "failed to create llama_context from model"

static inline bool fit_retry_with_dry_target_fork(const char * what) {
    return what != nullptr && (strstr(what, "requires ctx_other") != nullptr || strstr(what, FIT_ERR_NO_CTX) != nullptr);
}
