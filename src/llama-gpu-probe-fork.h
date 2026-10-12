#pragma once

// Fork: per-GPU probe and the prefill layer plan built from it (see llama-gpu-probe-fork.cpp for the reasoning).

#include "ggml-backend.h"

#include <string>
#include <vector>

struct llama_gpu_probe_fork_result {
    ggml_backend_dev_t dev = nullptr;
    size_t index = 0;            // position in the vector given to llama_gpu_probe_fork() (the order of the subset table)
    std::string name;
    double h2d_gbs      = 0.0;   // bulk host->device upload (32 MB, pinned source) with this card alone
    double h2d_conc_gbs = 0.0;   // the same upload while every probed card uploads at once: shared root complex, chipset uplink and RAM contention
    double lat_us       = 0.0;   // median time of one small (4 KB) synchronous upload: the fixed cost of a hand-off between layers
    double compute      = 0.0;   // compute throughput of a small quantised matmul through the backend's own kernels (weight bytes x tokens per second); only the ratio between cards means anything
};

// Runs the probe on every GPU (about 1 s per card) and remembers the results for llama_gpu_probe_find_fork().
// `conc` (optional) receives, for every subset of cards (bit i of the mask = card i, mask 1..2^n-1), the upload rate in GB/s of each member while exactly that
// subset uploads at once (conc[mask][i], 0 for non-members). A subset of one is the card alone. This is the measured cost of sharing the root complex, the
// chipset uplink and RAM: how much a card takes away from the others.
std::vector<llama_gpu_probe_fork_result> llama_gpu_probe_fork(const std::vector<ggml_backend_dev_t> & gpus, std::vector<std::vector<double>> * conc = nullptr);

// The subset table of the last llama_gpu_probe_fork() call (see its `conc` argument).
const std::vector<std::vector<double>> & llama_gpu_probe_conc_fork();

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
    double kv_bytes_token_layer = 0.0;  // K and V bytes one token adds to one layer: they move from a card left out of the prefill to the layer's owner afterwards
    double n_tokens            = 0.0;   // tokens of the prompt the plan is made for
    double pass_tokens         = 0.0;   // tokens of one prefill pass (micro-batch); the prompt runs in ceil(n_tokens / pass_tokens) passes; 0 = one pass
    double fastest_prefill_tps = 0.0;   // measured prefill tokens per second of the fastest card alone on this model (0 = unknown: estimated from the synthetic probe)
    std::vector<double> owner_share;    // share of the layers each card owns in the decode layout (the default split); empty = equal
};

// Shares of the layers (sum 1, one entry per probe result, in the order given) that minimise the modelled time of one prefill pass of m.n_tokens tokens.
// A share of 0 means the card is left out of prefill: nothing is uploaded to it during the pass, so the remaining cards get its bandwidth. Empty when fewer
// than two cards were probed or the probe data is unusable. `conc` is the subset table of llama_gpu_probe_fork() (empty: the all-cards rate of each card is
// used for every subset). `log` receives the lines explaining the numbers, including for each card of the chosen set what it steals, what it pays back per token
// and after how many tokens its fixed costs are paid back.
std::vector<float> llama_prefill_plan_fork(const std::vector<const llama_gpu_probe_fork_result *> & cards, const std::vector<std::vector<double>> & conc,
                                           const llama_prefill_model_fork & m, std::vector<std::string> * log);

// Memory side of the prefill on one card (a P-card, a card that computes the prompt): keep its dense weights resident, or stream them through a window of two
// layers like the experts? Streaming frees VRAM for a longer micro-batch and for the K/V of a long prompt, and costs dense_bytes / upload rate of extra traffic per pass.
struct llama_prefill_mem_fork {
    double vram_bytes             = 0.0;   // VRAM the pass may use on this card (total minus the CUDA context and what the system needs)
    double dense_bytes_total      = 0.0;   // dense weights of every layer this card computes in the prefill
    double dense_bytes_layer      = 0.0;   // dense weights of one layer
    double ring_bytes             = 0.0;   // the expert staging ring (3 expert tensors)
    double kv_bytes_per_token     = 0.0;   // K and V of one token for all layers this card computes
    double scratch_bytes_per_token = 0.0;  // compute scratch of one token of a micro-batch (about 65 KB measured on GLM)
    double n_tokens               = 0.0;   // prompt length
    double ub_cap                 = 0.0;   // the largest micro-batch worth considering (the context, or the user's -ub)
    double pass_fixed_us          = 0.0;   // `a`: cost of one pass that does not scale with the tokens and is not hidden (measured: upload exposure, launch gaps)
    double upload_gbs             = 0.0;   // the card's upload rate
};
struct llama_prefill_mem_choice {
    bool   feasible      = false;   // some micro-batch fits at all
    bool   stream_dense  = false;
    double ub            = 0.0;     // micro-batch the choice allows
    double passes        = 0.0;
    double extra_us      = 0.0;     // cost that depends on the choice: passes x (a + streamed dense traffic)
    const char * reason  = "";
};
llama_prefill_mem_choice llama_prefill_mem_plan_fork(const llama_prefill_mem_fork & m);
