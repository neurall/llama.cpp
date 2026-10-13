#pragma once

// Fork: dense split of a prompt pass over several GPUs (see llama-dsplit-fork.cpp for the reasoning).

#include "ggml.h"
#include "ggml-backend.h"

struct llama_model;

// --moe dense-split=1: copies the dense (non-expert) weights of every trunk layer once to every other GPU, at load. Nothing when the option is off or
// fewer than two GPUs hold layers.
void llama_dsplit_mirror_build_fork(const llama_model & model);
void llama_dsplit_mirror_free_fork(const llama_model & model);

// true when the copies exist (the graph builder may cut a prompt pass into token ranges)
bool llama_dsplit_ready_fork(const llama_model & model);

// the copy of weight t on device dev; t itself when t lives on dev or has no copy (routed experts, tensors in host memory)
ggml_tensor * llama_dsplit_mirror_fork(const llama_model & model, ggml_tensor * t, ggml_backend_dev_t dev);

// --moe dense-split=auto: which configuration a prompt pass of n_tokens runs in, chosen from measured pass times (see llama-dsplit-fork.cpp).
// Called before the graph of a pass is built; returns true when the choice differs from the one the last graph was built with (the graph must be rebuilt).
bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens);
// the choice for the graph being built: -1 not managed (option off, no copies, a pass under 64 tokens), 0 one card, 1 split over the cards (dense and experts)
int  llama_dsplit_cur_fork();
// the graphs reserved at context creation are sized for the split (the larger graph) whenever the split may be used
void llama_dsplit_reserve_fork(const llama_model & model);
// wall time of the prompt tokens evaluated since the last report (llama_context::synchronize)
void llama_dsplit_report_fork(int64_t n_tokens, int64_t us);
