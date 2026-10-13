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
