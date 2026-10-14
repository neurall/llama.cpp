#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step_fork(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>

#include "ggml-backend.h"

#include <vector>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer_fork {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (last slot all zeros)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;

    // router prediction (LLAMA_MOE_CACHE_PRED): the next layer's router weights; the graph applies
    // them to this layer's MoE input and hands the logits to the CPU expert op (src[4])
    ggml_tensor * pred_w = nullptr;
    // learned predictors: pred_all (fp16 [n_embd, n_expert * pred_ahead]; block k initialized from the router of layer
    // il+1+k) predicts that layer's router logits from this layer's MoE input and is trained in the
    // graph by NLMS with step size pred_mu (1 element, set per token by the cache; 0 = no training)
    int           pred_ahead = 0;
    bool          pred_on    = true;    // predict from this layer: one of its target layers misses often enough (updated every 256 steps)
    ggml_tensor * pred_all   = nullptr; // [n_embd, n_expert * pred_ahead] fp16: the predictors of the next layers, stacked
    ggml_tensor * pred_mu    = nullptr;
    ggml_tensor * pred_q     = nullptr; // Q8_0 copy of pred_all used for predicting (half the bytes, same accuracy); re-quantized after each update

    // JIT pool (layers cached on a slower-link GPU): jit_n expert slots (+1 zero slot) on the fastest-link GPU, filled
    // just in time with this token's misses and computed there by a second small chain; jit_table maps expert -> pool slot
    int32_t       jit_n      = 0;
    ggml_tensor * jit_up_c   = nullptr;
    ggml_tensor * jit_gate_c = nullptr;
    ggml_tensor * jit_down_c = nullptr;
    ggml_tensor * jit_table  = nullptr;
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
void llama_moe_cache_init_fork(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t prefetch_slots = 0, int32_t window = 64,
                          int32_t predict = 0, int32_t predict_train = 0);
void llama_moe_cache_free_fork();

// Heat of a start that runs the stock placement (no cache): the CPU-side expert matmuls already see their routed ids in host memory, so counting them costs a few integer
// increments per layer and token (no GPU read-back, no sync). The counts are saved at exit in the same state lines the cache reads (hot.<layer>), so the first start that
// runs the cache begins with a preheated set. Off with LLAMA_MOE_OBSERVE=0, without a state file, and for models without experts.
void llama_moe_observe_start(const llama_model & model);
void llama_moe_observe_step();   // once per decode call: ends the counting after observe-steps calls
void llama_moe_observe_save();

// largest batch (tokens) that uses the cache; bigger ones take the stock path
// (LLAMA_MOE_CACHE_MAX_BATCH, default 31: below the CUDA op-offload threshold)
int64_t llama_moe_cache_max_batch_fork();
// --moe max-batch=auto: node budget cap, start-up probe of the cache path (timed 16- and 64-token batches), result -> the computed limit
int64_t llama_moe_cache_max_batch_cap_fork();
bool    llama_moe_cache_probe_wanted_fork();
void    llama_moe_cache_probe_begin_fork();
void    llama_moe_cache_probe_end_fork(const double * cn, const double * cus, int nc, const double * sn, const double * sus, int ns);
void    llama_moe_cache_probe_stream_fork(bool on);   // force passes above 0 tokens onto the streaming path (probe)

// ubatch for one decode call of n_tokens new tokens, at most ub_max (the compute buffer is sized for it): equal passes, and a tail that fits the
// cache path (<= max-batch) is left to it instead of a nearly empty streaming pass that pays a whole expert upload. `--moe prefill-equal=0` = ub_max.
uint32_t llama_moe_cache_prefill_ubatch_fork(uint32_t n_tokens, uint32_t ub_max);
// the measured time of the prompt that the pass planner sized last (all its passes): the planner reads the compute per token from it
void llama_moe_cache_prefill_report_fork(uint32_t n_tokens, int64_t us);
// for the per-prompt decision line (llama_dsplit_report_fork): the pass plan of that prompt, and the expert split decided at load
bool llama_moe_cache_pass_plan_fork(uint32_t n_tokens, uint32_t & passes, uint32_t & pass_tokens, int & slots);
bool llama_moe_cache_expert_split_fork(double * speedup);
// multi-GPU prefill: alpha (1 = experts over all GPUs by link bandwidth, 0 = fastest GPU only), measured or LLAMA_PREFILL_SPLIT
float llama_moe_cache_prefill_alpha_fork();
// a GPU's measured upload link GB/s (startup probe, refined by uploads); 0 if unknown
double llama_moe_cache_link_gbs_fork(ggml_backend_dev_t dev);
double llama_moe_cache_resident_frac_fork(int il, ggml_backend_dev_t dev);   // share of layer il's experts ready in dev's pool (0: the layer's cache is elsewhere)
double llama_moe_cache_compute_us_fork();                                     // measured compute of one prompt token on the offload card, all layers, us (0: not yet)
// before the first compute-buffer reserve: time each GPU's host link, reserve the prefill split only when it can pay
void llama_moe_cache_prefill_decide_fork(const std::vector<ggml_backend_dev_t> & gpus);
// --moe layer-heat=1: per layer device in / out; true when a layer moved (the most concentrated expert layers to the GPU with the slower host link)
bool llama_moe_layer_heat_fork(const llama_model & model, std::vector<ggml_backend_dev_t> & layer_dev);
// layers il_a and il_b trade GPUs: their expert caches trade cards (cached experts move along as far as free blocks allow); false: not possible
bool llama_moe_cache_swap_layers_fork(int il_a, int il_b);
// compute alias (--moe alias=1): the pool buffers whose first blocks the scheduler may use for its compute graphs, and the bytes it may use
struct llama_moe_alias_buf_fork { ggml_backend_buffer_t buf; ggml_backend_dev_t dev; size_t capacity; };
std::vector<llama_moe_alias_buf_fork> llama_moe_cache_alias_buffers_fork();
// --moe mirror-pool: the dense split's copies at the start of dev's pool (the pass planner keeps room), and per graph where its tensors start (behind them
// when it splits, else 0); the lent range covers the copies too
void llama_moe_cache_alias_mirror_fork(ggml_backend_dev_t dev, size_t bytes);
void llama_moe_cache_alias_offset_fork(ggml_backend_dev_t dev, size_t off);
// after a graph is allocated, before its inputs are set: lend the blocks it reaches to it (evict their experts)
void llama_moe_cache_alias_fork(ggml_backend_sched_t sched, int32_t n_tokens);
// after a worst-case reserve with the alias: does the graph (+ CUDA scratch) fit in the lent blocks, and the prefetch staging slots too? need / cap in bytes
struct llama_moe_alias_fit_fork { bool graph = true; bool stage = true; size_t need = 0; size_t cap = 0; };
llama_moe_alias_fit_fork llama_moe_cache_alias_fit_fork(ggml_backend_sched_t sched, int32_t n_tokens);
// the pool cannot lend the smallest worst-case graph: the alias stays off from now on
void llama_moe_cache_alias_drop_fork();
bool    llama_moe_cache_active_fork();
bool    llama_moe_cache_wants_probs_fork();   // --moe policy=r: the gate op of every cached layer also gets the router probabilities (src[6]) for the relevance score
// startup probe: how many reader threads reach 95% of the CPU's RAM read rate (0: not measured)
int32_t llama_moe_cache_cpu_sat_threads_fork();
// live cache state for llama_get_tuning_info(): false when the cache is off
bool    llama_moe_cache_get_info_fork(struct llama_tuning_info * info);

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer_fork * llama_moe_cache_lookup_fork(const ggml_tensor * up_exps);
// JIT miss offload (LLAMA_MOE_CACHE_UPLOAD_NOW): called with a layer's router ids on the host before its cache chain launches
void llama_moe_cache_jit_fork(const llama_moe_cache_layer_fork * pub, const ggml_tensor * ids);

// apply throttled LRU updates; call between graph executions only
void llama_moe_cache_step_fork();

// learned predictors: the next decode trains them (the graph holds the update nodes only then, so a
// non-training step costs just the prediction); graph reuse must not cross a change of this
bool llama_moe_cache_pred_train_now_fork();
bool llama_moe_cache_graph_reusable_fork();
void llama_moe_cache_graph_built_fork();
