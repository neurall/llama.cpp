#pragma once

// Next-layer MoE router predictor (experimental sandbox).
//
// For every MoE layer L and lookahead k (1..LLAMA_PRED_AHEAD) an F32 matrix [n_embd, n_expert],
// started as layer L+k's router, predicts L+k's router logits from layer L's MoE input. The graph
// trains it by NLMS when L+k's real logits exist (w += mu/|x|^2 * x (x) (logits - pred)) and counts
// how many of the predicted top-k experts L+k really selects, all on the GPU.
//
//   LLAMA_PRED_AHEAD=k   lookahead layers (0/unset: off)
//   LLAMA_PRED_TRAIN=N   train every N decode steps (0: frozen, default 1)
//   LLAMA_PRED_MU=m      NLMS step size (default 0.5)
//   LLAMA_PRED_SCORE=N   count top-k hits every N decode steps (0: off, default 1)
//
// The graph holds the train and score nodes only on the steps that need them, so steps doing neither
// run the bare prediction.

#include <cstdint>

struct llama_model;
struct ggml_tensor;

void llama_pred_init(const llama_model & model); // once; no-op when off
void llama_pred_free();

// before each decode: sets this step's mu, reports the overlap every 256 decode tokens
void llama_pred_step(int64_t n_tokens);

// largest batch the predictor runs on (decode, small speculative batches)
int64_t llama_pred_max_batch();

int           llama_pred_ahead();
bool          llama_pred_train_now();      // this step trains
bool          llama_pred_score_now();      // this step counts hits
bool          llama_pred_graph_reusable(); // the graph was built for this step's train/score mode
void          llama_pred_graph_built();
ggml_tensor * llama_pred_w(int il);        // [n_embd, n_expert * K]: layer il's predictors of layers il+1..il+K stacked, or nullptr
int           llama_pred_nk(int il);       // K for layer il
ggml_tensor * llama_pred_mu();             // [1]
ggml_tensor * llama_pred_stats();          // [8]: summed top-k hits per lookahead
