#pragma once

// Fork: per-GPU probe and the prefill layer plan built from it (see llama-gpu-probe-fork.cpp for the reasoning).

#include "ggml-backend.h"

#include <string>
#include <vector>

struct llama_gpu_probe_fork_result {
    ggml_backend_dev_t dev = nullptr;
    std::string name;
    double h2d_gbs      = 0.0;   // bulk host->device upload (64 MB, pinned source) with this card alone
    double h2d_conc_gbs = 0.0;   // the same upload while every probed card uploads at once: shared root complex, chipset uplink and RAM contention
    double lat_us       = 0.0;   // median time of one small (4 KB) synchronous upload: the fixed cost of a hand-off between layers
    double wtps         = 0.0;   // compute speed in "weight bytes x tokens per second": a quantised mul_mat over 64 MB of uploaded weights and 512 tokens
};

// Runs the probe on every GPU (about 1 s per card) and remembers the results for llama_gpu_probe_find_fork().
std::vector<llama_gpu_probe_fork_result> llama_gpu_probe_fork(const std::vector<ggml_backend_dev_t> & gpus);

// Results of an earlier llama_gpu_probe_fork() for this device, or nullptr.
const llama_gpu_probe_fork_result * llama_gpu_probe_find_fork(ggml_backend_dev_t dev);

// What the plan needs to know about the model (all from tensor sizes and hyper-parameters, available before any buffer is allocated).
struct llama_prefill_model_fork {
    int    n_layer             = 0;
    double expert_bytes_layer  = 0.0;   // routed experts of one layer, bytes
    double dense_bytes_layer   = 0.0;   // everything else of one layer (attention, norms, router, shared expert), bytes
    double n_expert            = 0.0;
    double n_used              = 0.0;   // experts used per token
    double n_embd              = 0.0;   // activation width: what crosses between two cards at a layer boundary, per token
    double n_tokens            = 0.0;   // tokens of the prefill pass the plan is made for
};

// Shares of the layers (sum 1, one entry per probe result, in the order given) that minimise the modelled time of one prefill pass.
// A share of 0 means the card is left out of prefill. Empty when fewer than two cards were probed or the probe data is unusable.
// `log` receives the lines explaining the numbers.
std::vector<float> llama_prefill_plan_fork(const std::vector<const llama_gpu_probe_fork_result *> & cards, const llama_prefill_model_fork & m, std::vector<std::string> * log);
