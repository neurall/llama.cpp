#pragma once
// fork: the ubatch of a draft model's context. Its compute buffer grows with ubatch x its vocab (logits, markov rows): with the target's ubatch
// (4096 on Inkling 2 Mini, vocab 197k) it asked 6.3 GiB for a 0.5 GB draft. The largest power of two <= the target's ubatch whose vocab-wide rows
// (8 bytes a vocab entry a token) stay within the draft's own weight size, at least 64 (a draft block batch); a draft with a prompt ring
// (DSV4: ring = its window + a draft block) never takes more than the ring in one call. Never above the target's ubatch.
// The runtime (common_speculative_init_result) and fit's measure of the draft (common_fit_apply) both use it: fit measured the draft at the
// target's ubatch, counted 6.3 GiB it never takes and left the context at its 4096 minimum (Inkling 2 + DSpark, 2026-11-25).
// test: tests/test-draft-ubatch-fork.cpp

#include "llama.h"

#include <cstdint>
#include <cstdlib>

static inline uint32_t draft_ubatch_fork(uint64_t n_vocab, uint64_t draft_bytes, uint32_t n_ubatch_tgt, uint32_t ring /* 0: no prompt ring */) {
    const uint64_t per_tok = n_vocab * 8;
    const uint64_t want    = per_tok ? draft_bytes / per_tok : n_ubatch_tgt;
    uint32_t ub = 64;
    while (ub * 2 <= n_ubatch_tgt && ub * 2 <= want) { ub *= 2; }
    if (ring > 0) {
        while (ub > 64 && ub / 2 >= ring) { ub /= 2; }
    }
    return ub < n_ubatch_tgt ? ub : n_ubatch_tgt;
}

// the rule for a loaded (or dry-loaded, no_alloc) draft model
static inline uint32_t common_speculative_draft_ubatch(const llama_model * model_dft, uint32_t n_ubatch_tgt) {
    // a DSV4 draft with the prompt ring never takes more than its window (n_swa rows injected) plus a draft block in one call:
    // ubatch 4096 reserved 15.5 GiB outside the block pool on DeepSeek 4.1 (1 MiB of vocab rows a token)
    char hc[32];
    const bool ring = llama_model_meta_val_str(model_dft, "dflash.hyper_connection.count", hc, sizeof(hc)) >= 0 && std::atoi(hc) > 0;
    return draft_ubatch_fork(llama_vocab_n_tokens(llama_model_get_vocab(model_dft)), llama_model_size(model_dft), n_ubatch_tgt,
                             ring ? (uint32_t) llama_model_n_swa(model_dft) + 64 : 0);
}
