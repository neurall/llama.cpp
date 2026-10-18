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
#include <functional>
#include <string>
#include <vector>

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
bool llama_moe_layer_heat_fork(const llama_model & model, std::vector<ggml_backend_dev_t> & layer_dev, const std::vector<double> & lay_dense_b,
        const std::vector<double> & lay_exp_b);
// layers il_a and il_b trade GPUs: their expert caches trade cards (cached experts move along as far as free blocks allow); false: not possible
bool llama_moe_cache_swap_layers_fork(int il_a, int il_b);
// compute alias (--moe alias=1): the pool buffers whose first blocks the scheduler may use for its compute graphs, and the bytes it may use
struct llama_moe_alias_buf_fork { ggml_backend_buffer_t buf; ggml_backend_dev_t dev; size_t capacity; };
std::vector<llama_moe_alias_buf_fork> llama_moe_cache_alias_buffers_fork();
// --moe mirror-pool: the dense split's copies at the start of dev's pool (the pass planner keeps room), and per graph where its tensors start (behind them
// when it splits, else 0); the lent range covers the copies too
void llama_moe_cache_alias_mirror_fork(ggml_backend_dev_t dev, size_t bytes);
void llama_moe_cache_alias_offset_fork(ggml_backend_dev_t dev, size_t off);
// after a graph is allocated, before its inputs are set: lend the blocks it reaches to it (evict their experts). Returns true when the graph needed more than the
// space in front of the KV records (kv-tail): the records were moved forward to make room, so the graph (its views of the KV) must be built again.
bool llama_moe_cache_alias_fork(ggml_backend_sched_t sched, int32_t n_tokens);
// the reason the last call refused its graph (the compute graph and the KV do not fit the pool: the caller fails the decode), nullptr when it did not
const char * llama_moe_cache_alias_err_fork();
// kv-tail, before a graph is built (process_ubatch): grow the records' space when the cells in use pass half of it, and make room in front of them for the lend
// a graph of this size needed before. Returns true when the records moved (the previous graph's views are stale: no reuse).
bool llama_moe_cache_kv_tail_prepare_fork(ggml_backend_sched_t sched, int32_t n_tokens);
// after a worst-case reserve with the alias: does the graph (+ CUDA scratch) fit in the lent blocks, and the prefetch staging slots too? need / cap in bytes
struct llama_moe_alias_fit_fork { bool graph = true; bool stage = true; size_t need = 0; size_t cap = 0; };
llama_moe_alias_fit_fork llama_moe_cache_alias_fit_fork(ggml_backend_sched_t sched, int32_t n_tokens);
// the pool cannot lend the smallest worst-case graph: the alias stays off from now on
void llama_moe_cache_alias_drop_fork();

// KV cache in the pool (--moe kv-tail=1, token-major record, forward rows: cell c is row c; fork/docs/kv-grow.md). Layout of a card's pool buffer:
//   [ compute lend from byte 0 ][ record 1: H cells ][ record 2: H cells ] .. [ expert blocks ]
// The cache registers its record of one card: `bytes` is the virtual size (all cells of the context), `row` the bytes of one cell, `rows()` the cells in use
// (highest used cell + 1), `step` the growth step in cells (profile from the GGUF: 2k for mostly full-attention models, 10k otherwise), `detach()` frees the
// small backing buffer the cache starts with, `attach(base)` points the record at `base` (row 0). The pool backs H rows of every record (H grows by `step`
// when the cells in use pass H/2) and moves the records when H or the lend in front of them grows; the blocks under them are taken from the experts.
struct llama_kv_tail_fork {
    void *             owner = nullptr;
    ggml_backend_dev_t dev   = nullptr;
    size_t             bytes = 0;
    size_t             row   = 0;
    size_t             pad   = 1;   // the attention reads the cells rounded up to this many (masked, but read): the claim covers them
    size_t             step  = 0;   // growth step in cells (0: 10240)
    float              bpv   = 2.0f;   // bytes per KV value of this record's types (f16 2.0, q8_0 1.0625, q4_0 0.5625)
    std::function<size_t()>                  rows;
    std::function<void()>                    detach;
    std::function<void(ggml_backend_buffer_t, uint8_t *, size_t)> attach;   // (buffer, address of row 0 of this record, bytes from one cell to the next)
    bool               attached = false;   // placed in a pool's tail or in an arena
    // qkv-ram-fallback in place (fork/docs/qkv-ram-fallback.md): the record's K and V rows as typed segments of its cell row (layer order, K then V).
    // layout(tk, tv, segs): the row bytes the record would have with K type tk and V type tv, and its segments (0: those types cannot hold it, e.g. a row that is
    // not a whole number of quant blocks). retype(tk, tv): the record tensor and its views take those types and that layout (the next attach sets their data).
    // Without them (or convertible false: an indexer's key cache, whose reads are the indexer's own) the record keeps its types when the KV changes tier.
    struct seg { ggml_type type; int64_t ne0; size_t off; bool v; int64_t head; int32_t lay; };   // head: elements per head (the attention kernel choice depends on it); lay: layer of the record
    // mask: layers whose K/V live in host memory (the RAM part of the ladder): they take no room in the row
    std::function<size_t(ggml_type, ggml_type, const std::vector<char> *, std::vector<seg> *)> layout;
    // retype(tk, tv, mask, host, hbuf): new types; layers in mask point at host[2*layer + is_v] = (cell 0, cell stride) in buffer hbuf (mask/host nullptr: unchanged)
    std::function<void(ggml_type, ggml_type, const std::vector<char> *, const std::vector<std::pair<uint8_t *, size_t>> *, ggml_backend_buffer_t)> retype;
    ggml_type          type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;   // the types the record has now
    bool               convertible = false;
};
void llama_moe_cache_kv_tail_add_fork(const llama_kv_tail_fork & rec);
// a cache rebuilt with other KV types (llama_kv_refit) takes the place of the old one in the running pool
void llama_moe_cache_kv_tail_rebind_fork();
// qkv-ram-fallback (fork/docs/qkv-ram-fallback.md): the tokens the pool can hold as KV with the expert cache at its floor, per tier, the smallest over the cards,
// from the exact row layouts of each tier. valid is false before the pool exists. row_f16: bytes of one token's KV over all cards as f16.
struct llama_kv_fit_caps { bool valid = false; size_t f16 = 0, k8v8 = 0, k4v4 = 0, row_f16 = 0, ram = 0; };   // ram: tokens with every growing layer's KV (tier 2 types) in host memory, from free RAM less 4 GiB
llama_kv_fit_caps llama_moe_cache_kv_fit_caps_fork();
// qkv-ram-fallback in place: the convertible records of every card take K type tk and V type tv inside the pool. The rows in use are converted on the card
// (through a staging range borrowed at the front of the pool), the zone takes the new rows' size and the blocks it leaves go back to the experts (converting up,
// it takes the blocks it needs first). No cache object, graph buffer or allocation is made or freed; the next graphs are built with the new types. false: nothing
// changed (note says why: no pool yet, the rows do not fit, a record that cannot take those types).
bool llama_moe_cache_kv_retype_fork(ggml_backend_sched_t sched, int32_t tier, char * note, size_t note_size);   // tier 0: the types the cache was created with, 1: q8_0/q8_0, 2: q4_0/q4_0
// qkv-ram-fallback, RAM part: the KV of whole layers moves to a host buffer (allocated once) until every card holds n_cells; the pool zone shrinks by their rows
// (the experts take it). gpu_reads: attention stays on the GPU and reads them over PCIe; else the views sit in a host buffer and the scheduler runs their attention on
// the CPU. false: nothing changed (note says why).
bool llama_moe_cache_kv_ram_fork(ggml_backend_sched_t sched, size_t n_cells, char * note, size_t note_size);
// does an expert pool (the compute alias lend) exist for this context? Without it (a model that fits in VRAM, KV in the dense arena) the graph buffers are real allocations:
// reserving them at the growth horizon makes every pass beyond it reallocate (Qwen3.6, 7 reallocs x 80 ms = +0.55 s on a 13.7k prompt), so the reserve is made at the full size.
void llama_moe_cache_kv_pool_expected_fork(bool on);
bool llama_moe_cache_kv_pool_expected_fork();

void llama_moe_cache_kv_tail_remove_fork(void * owner);
// before the cache writes cells up to `max_cell`: do the claimed blocks cover them? (the cache asserts it)
bool llama_moe_cache_kv_tail_covers_fork(void * owner, size_t max_cell);
// before the pool is created: the small backing buffers go back to the card (the pool is sized with that memory)
void llama_moe_cache_kv_tail_detach_fork();
// records that no expert pool took (a dense model, or a card without experts): one buffer per card of its free VRAM (less the margin) is allocated, once, the records sit at its tail and
// a request may use as many tokens as the buffer holds. Called after the pool start at the first real decode.
void llama_moe_cache_kv_arena_start_fork();
// what the cache and the pool hold now, per card (the per-request log line and the "stats" field of a response)
struct llama_kv_tail_stat_fork {
    std::string dev;
    size_t kv_cells = 0, kv_bytes = 0;          // cells in use, bytes the pool holds for them
    size_t cap_tokens = 0;                      // tokens the pool (or arena) of this card can back for its records now (the smallest over its records)
    size_t moe_bytes = 0;                       // bytes of the blocks the experts can use now
    size_t blocks = 0, blocks_free = 0, blocks_kv = 0, blocks_lent = 0;   // expert blocks of the pool: all, free, under the KV, lent to the compute graph
    size_t claim_n = 0, release_n = 0, evicted = 0;   // since the last call of this function with reset: blocks claimed, released, experts evicted by claims
};
std::vector<llama_kv_tail_stat_fork> llama_moe_cache_kv_tail_stats_fork(bool reset);
void llama_moe_cache_kv_tail_report_fork(const char * when);   // the same as one log line per card (info level)
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
