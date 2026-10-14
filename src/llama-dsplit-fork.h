#pragma once

// Fork: dense split of a prompt pass over several GPUs (see llama-dsplit-fork.cpp for the reasoning).

#include "ggml.h"
#include "ggml-backend.h"

struct llama_model;

// --moe dense-split=1: copies the dense (non-expert) weights of every trunk layer once to every other GPU, at load. Nothing when the option is off or
// fewer than two GPUs hold layers.
void llama_dsplit_mirror_build_fork(llama_model & model);   // also gives every copy the precision rule of its original (model.prec_policy)
void llama_dsplit_mirror_free_fork(const llama_model & model);

// true when the copies exist (the graph builder may cut a prompt pass into token ranges)
bool llama_dsplit_ready_fork(const llama_model & model);

// the copy of weight t on device dev; t itself when t lives on dev or has no copy (routed experts, tensors in host memory)
ggml_tensor * llama_dsplit_mirror_fork(const llama_model & model, ggml_tensor * t, ggml_backend_dev_t dev);

// a layer as seen from card dev: every dense weight replaced by its copy there (llama_layer holds tensor pointers only)
struct llama_layer;
llama_layer llama_dsplit_layer_fork(const llama_model & model, const llama_layer & l, ggml_backend_dev_t dev);

// the split planner (see llama-dsplit-fork.cpp): for prompt passes of n_tokens, does the second card pay, and with which start shares. valid = it could be
// computed (a measured compute per token exists, two GPUs).
struct llama_dsplit_plan_fork {
    bool   valid      = false;
    bool   split      = false;   // the split beats one card by more than 5%
    double share      = 0.55;    // tokens of the first range (the card with the slower link)
    double t_one_ms   = 0.0;
    double t_split_ms = 0.0;
};
llama_dsplit_plan_fork llama_dsplit_plan_get_fork(const llama_model & model, int64_t n_tokens);
// --moe split-experts: -1 auto (planner + tuner), 0 dense split only (helpers hold no experts), 1 dense split plus experts (link-ratio start + controller)
int llama_dsplit_experts_mode_fork();
// --moe split-f16: how the hidden states and the experts' sums cross between the cards. 0: f32 both ways; 1: f16 both ways; 2 (split-f16=in): f16 to the
// card that computes the experts, f32 back (see llama-dsplit-fork.cpp)
int llama_dsplit_hop_mode_fork();
// the helper's share of the tokens from the plan of that pass size (-1: no plan)
double llama_dsplit_plan_share_fork(int64_t n_tokens);
// the helper's share of layer il's experts from the plan of that pass size (-1: no plan)
double llama_dsplit_plan_helper_f_fork(int il, int64_t n_tokens);

// --moe dense-split=auto: which configuration a prompt pass of n_tokens runs in, chosen from measured pass times (see llama-dsplit-fork.cpp).
// Called before the graph of a pass is built; returns true when the choice differs from the one the last graph was built with (the graph must be rebuilt).
bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens);
// the choice for the graph being built: -1 not managed (option off, no copies, a pass under 64 tokens), 0 one card, 1 split over the cards (dense and experts)
int  llama_dsplit_cur_fork();
// the graphs reserved at context creation are sized for the split (the larger graph) whenever the split may be used
void llama_dsplit_reserve_fork(const llama_model & model);
// wall time of the prompt tokens evaluated since the last report (llama_context::synchronize)
void llama_dsplit_report_fork(int64_t n_tokens, int64_t us);
