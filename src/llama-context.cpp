#include "llama-context.h"
#include "llama-kvtier-fork.h"
void llama_moe_set_opt(const char * key, const char * value);
#include <fstream>
#include <filesystem>
#include <map>
#include <thread>

#include "llama-moecache-fork.h"
#include "llama-dsplit-fork.h"
#include "llama-kv-cache.h"
#include "llama-moestate-fork.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

bool llama_split_ctl_poll_fork(int64_t n_expert, int64_t n_embd);   // fork: llama-graph.cpp, the share controller of the expert split

// fork: defined in ggml-backend.cpp, declared here (a line in a shared ggml header rebuilds every CUDA object): zeros in the row padding behind the last
// expert of a copied expert tensor
extern "C" {
    GGML_API void ggml_backend_tensor_zero_tail_fork(ggml_backend_t backend, struct ggml_tensor * tensor);
}

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static uint32_t g_decode_n_tokens = 0;   // tokens of the decode call in progress, for the stats of llama_process
void llama_moe_token_time_fork(double ms);   // src/llama-moestate-fork.cpp
void llama_moe_prompt_done_fork(uint32_t n_tokens, double ms);
void llama_moe_prompt_begin_fork(uint32_t n_tokens, uint32_t ubatch);
void     llama_dsplit_note_limit_fork(uint32_t cap);   // src/llama-dsplit-fork.cpp: the one-card pass limit of this call
void     llama_dsplit_note_card_max_fork(uint32_t cap);   // src/llama-dsplit-fork.cpp: the context's card limit (what %limit is of)
size_t   llama_moe_cache_alias_room_fork(ggml_backend_dev_t dev, uint32_t n_tokens);   // src/llama-moecache-fork.cpp: what a card's pool lends a prompt graph of n tokens
void     llama_moe_cache_note_graph_fork(ggml_backend_dev_t dev, uint32_t n_tokens, size_t graph_bytes);   // src/llama-moecache-fork.cpp: a load-time measure-only reserve is a measured graph size for the pass planner
size_t   llama_moe_cache_graph_scratch_fork(ggml_backend_sched_t sched, ggml_backend_dev_t dev);   // src/llama-moecache-fork.cpp: the computed CUDA scratch of the graph the scheduler split last, on dev's pool
uint32_t llama_dsplit_pass_size_fork(uint32_t cap);   // the full pass of the split: the fastest card takes the limit
uint32_t llama_moe_cache_pass_fit_fork(uint32_t cells, uint32_t ub);
void     llama_dsplit_bench_fork(const llama_model & model);   // llama-dsplit-fork.cpp: matmul + attention latency per GPU, before this context takes VRAM
void     llama_dsplit_note_pass_fork(int64_t pos0);   // llama-dsplit-fork.cpp (dsf): every prompt pass is timed for the split factors
bool     llama_dsplit_pass_open_fork();
void     llama_dsplit_pass_end_fork();
void     llama_dsplit_pass_abandon_fork();   // a pass that ran out of CUDA memory: no sample
void     llama_dsplit_note_fill_fork(double ms);
uint32_t llama_dsplit_max_pass_fork(const llama_model * model, uint32_t n_ubatch);   // llama-dsplit-fork.cpp: the largest dense-split pass (sliding-window cells)        // src/llama-moecache-fork.cpp: the largest pass every pool can lend for
bool     llama_moe_cache_refusal_fork(size_t * need, size_t * room);        // the last refused lend: need and room (bytes)
extern "C" void ggml_backend_sched_upload_add_fork(ggml_backend_dev_t dev, uint64_t n);   // ggml-backend.cpp: bytes a card received from host memory (pass statistics)
extern "C" void ggml_backend_sched_link_add_fork(ggml_backend_dev_t dst, ggml_backend_dev_t src, uint64_t n);   // the same for any input: src NULL = host memory
static thread_local bool g_split_sub = false;          // decode() runs one pass of a call sized pass by pass (--moe split-pass=1)
bool llama_dsplit_in_pass_loop_fork() { return g_split_sub; }   // llama-graph.cpp ds_plan: the pass was sized for all cards by the loop below (graph built on this thread)
void llama_moe_prompt_progress_fork(uint32_t done);
uint32_t llama_moe_ubadapt_cache_fork(uint32_t n_tokens, uint32_t planned, uint32_t ceil);
uint32_t llama_moe_ubadapt_next_fork(uint32_t n_tokens, uint32_t ceil);
void llama_moe_decision_fork(const char * name, const char * value, const char * why);
void llama_moe_cache_vram_watch_fork(bool prompt);   // src/llama-moecache-fork.cpp: the lowest free VRAM per pool card
void llama_moe_cache_context_live_fork(const void * ctx, const llama_model * model, bool live);   // src/llama-moecache-fork.cpp: the live contexts' models

static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

// fork: nsys range hooks for the scheduler (GGML_NVTX=1), defined in ggml-backend.cpp
GGML_API void ggml_backend_nvtx_set_fork(void (*)(const char *), void (*)(void));

void llama_kv_reserve_override_fork(uint32_t n);   // llama-kv-cache.cpp
// host buffer of a pass: H(u, k) = u * (a * k + b) + c for u pass tokens and k cells, fitted from three small reserves at startup (host_calibrate), file level: one context per process
static double g_host_a = 0.0, g_host_b = 0.0, g_host_c = 0.0, g_host_d = 0.0;
void llama_moe_cache_pp_cells_fork(uint64_t cells_end);
void llama_moe_cache_mask_bpc_fork(double bytes_per_token_cell);   // llama-moecache-fork.cpp: the graph lend's growth with the cells (the cards' copies of the masks)   // llama-moecache-fork.cpp: the cells the next planned call ends at (its compute grows with them); declared here, no header edit
static bool   g_host_cal = false, g_host_ok = false;
static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);
    llama_dsplit_bench_fork(model);   // fork: once per process, while the VRAM is still free (the KV cache and the expert pool come below)

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    // the load-time fit measures a context with no_alloc: it must see the small context common handed over, not the tail's per-request size (it counted the KV of 82k tokens and moved experts to the CPU)
    if (const char * kt = llama_moe_get_opt("kv-tail"); kt && atoi(kt) != 0 && !model.no_alloc()) {   // the context of a request is calculated from the request, -c is not used (fork/docs/kv-tail.md)
        const char * ac = llama_moe_get_opt("kv-auto-ctx");   // --moe kv-auto-ctx=0: -c is accepted again (the cap of the virtual record)
        const bool tail_possible = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && params.n_seq_max <= 1 && params.type_k == params.type_v && !model.devices.empty();
        const char * given = llama_moe_get_opt("kv-ctx-given");   // set by the common code: 1 when the user passed -c
        if ((!ac || atoi(ac) != 0) && tail_possible) {
            {   // what the cache goes back to when it cannot use the tail layout (llama_kv_cache): the user's -c, else the calculated context of the normal layout (as in the branch below).
                // Not cparams.n_ctx: without -c that is the model's trained maximum, "going back" to it kept a static KV of a million cells (the structures that scale with it
                // filled the host) whenever the cache declined the tail after this check (shared layers, transposed V, no offload)
                uint32_t user = cparams.n_ctx;
                if (params.n_ctx == 0 && !(given && atoi(given) != 0)) {
                    const char * bc = llama_moe_get_opt("kv-budget-ctx");
                    user = std::min<uint32_t>(hparams.n_ctx_train, bc && atoi(bc) > 0 ? (uint32_t) atoi(bc) : 17408);
                }
                llama_moe_set_opt("kv-user-ctx", std::to_string(user).c_str());
            }
            if (!(given && atoi(given) != 0)) {   // no -c: the smaller of the model's maximum and what the machine can hold (a 1M-token model must not get a 1M-token context: the structures that scale with it, the masks, fill the host)
                uint64_t row = 0, fixed_b = 0, free_b = 0;   // KV bytes per context token (the layers whose cache grows), the bytes of the sliding-window layers (a ring of the window plus a batch: they do not grow with the context), free VRAM over all GPUs
                for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
                    if (!hparams.has_kv(il)) { continue; }
                    const uint64_t r = ggml_row_size(params.type_k, hparams.n_embd_k_gqa(il)) + ggml_row_size(params.type_v, hparams.n_embd_v_gqa(il));
                    if (hparams.is_swa(il)) { fixed_b += r * ((uint64_t) hparams.n_swa + llama_dsplit_max_pass_fork(&model, params.n_ubatch)); }   // MiMo: 39 of 48 layers are sliding-window; counting them per token made the cap 10x too low (196k instead of 1M)
                    else                    { row += r; }
                }
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { size_t f = 0, t = 0; ggml_backend_dev_memory(d, &f, &t); free_b += f; }
                }
                uint64_t fit = row > 0 && free_b > fixed_b ? (free_b - fixed_b) / row : (uint64_t) hparams.n_ctx_train;
                fit = std::max<uint64_t>(4096, fit / 256 * 256);
                // No up-front cap (user 2026-10-17): the pool claims KV cells per request, and running out of pool space is what refuses a request (and later what
                // moves it to quantised KV, then RAM KV). A declared context computed here from guessed terms was wrong for sliding-window models (10x too low on MiMo) and
                // is not needed: the reserve graph uses min(size, step) cells, so the model's own maximum costs nothing until a request needs it. `fit` is only logged.
                cparams.n_ctx = hparams.n_ctx_train;
                if (const char * fb0 = llama_moe_get_opt("kv-fallback")) { LLAMA_LOG_INFO("%s: --moe kv-fallback=%s (the KV tiers after f16 that may be used)\n", __func__, fb0); }
                // --moe kv-fallback=none: no tier past f16 in the pool, so the context is what VRAM holds (llama_kv_fit_select refuses the rest)
                if (const char * fb = llama_moe_get_opt("kv-fallback"); fb && strcmp(fb, "none") == 0) { cparams.n_ctx = (uint32_t) std::min<uint64_t>(hparams.n_ctx_train, fit); }
                LLAMA_LOG_WARN("%s: kv-tail: no -c: context %u tokens (the model maximum; the pool refuses a request it cannot hold; free VRAM %.1f GiB would hold %llu tokens of KV beside nothing else)\n", __func__, cparams.n_ctx, free_b / 1073741824.0, (unsigned long long) (row ? free_b / row : 0));
            }
        } else if (!tail_possible) {
            // the common code leaves n_ctx at 0 in tail mode (the tail needs no default), and 0 means the model's trained maximum: a static KV (and context-scaled graph
            // buffers: 73 GB at 1M cells on the GLM stand-in) of a million cells. Without -c the normal layout gets the calculated context (common_auto_ctx) instead.
            if (params.n_ctx == 0 && !(given && atoi(given) != 0)) {
                const char * bc = llama_moe_get_opt("kv-budget-ctx");   // common_auto_ctx: the largest context that leaves the expert cache most of its VRAM
                const uint32_t fit_ctx = bc ? (uint32_t) atoi(bc) : 0;
                cparams.n_ctx = std::min<uint32_t>(hparams.n_ctx_train, fit_ctx ? fit_ctx : 17408);
            }
            LLAMA_LOG_WARN("%s: kv-tail is not possible here (needs flash attention, one sequence, equal K and V types, no sparse-attention indexer, a GPU): the normal KV cache with n_ctx = %u\n", __func__, cparams.n_ctx);
        }
    }
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP || model.arch == LLM_ARCH_GLM5_NEXT) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.prefetch_experts_slots = params.prefetch_experts_slots;
    cparams.moe_cache = params.n_moe_cache_slots != 0;
    {   // GGML_NVTX=1: the CUDA backend's nsys range functions for the scheduler (declared here, see ggml_backend_nvtx_set_fork in ggml-backend.cpp)
        const char * nv = getenv("GGML_NVTX");
        ggml_backend_reg_t reg = nv && atoi(nv) >= 1 ? ggml_backend_reg_by_name("CUDA") : nullptr;
        if (reg) {
            ggml_backend_nvtx_set_fork((void (*)(const char *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nvtx_push_fork"),
                                       (void (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nvtx_pop_fork"));
        }
    }
    cparams.kv_unified = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;   // upstream's expert cache size (inert here, the fork's own cache is --moe)


    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends; the offload GPU goes first: the scheduler offloads big-batch
        // ops on host weights to the first backend that accepts them
        std::vector<llama_device> devs = model.devices;
        if (model.dev_offload < devs.size()) {
            std::rotate(devs.begin(), devs.begin() + model.dev_offload, devs.begin() + model.dev_offload + 1);
        }
        for (const auto & dev : devs) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }
        // a context that runs on tensors of another model (a draft's token embeddings or lm head, shared through ctx_other) needs a backend on every card that
        // holds them: a DSpark draft given -devd CUDA1 aborted on the target's output.weight in CUDA0 ("pre-allocated tensor (output.weight) in a buffer (CUDA0)
        // that cannot run the operation", 2026-10-26). The extra backend only runs the ops on those tensors; the draft's own layers stay on the cards -devd names.
        if (cparams.ctx_other) {
            for (const auto & od : llama_get_model(cparams.ctx_other)->devices) {
                if (od.is_meta || std::any_of(devs.begin(), devs.end(), [&](const llama_device & d) { return d.dev == od.dev; })) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(od.dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(od.dev)));
                }
                LLAMA_LOG_INFO("%s: %s added for the tensors this context shares with its target\n", __func__, ggml_backend_dev_name(od.dev));
                backends.emplace_back(backend);
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    kv_type_k = kv_type_k0 = params.type_k; kv_type_v = kv_type_v0 = params.type_v; kv_swa_full = params.swa_full;
    llama_moe_cache_kv_pool_expected_fork(cparams.moe_cache);   // before the memory module: its get_size_reserve reads it
    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ kv_type_k,
            /*.type_v    =*/ kv_type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            /*.no_alloc  =*/ model.no_alloc(),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the offload device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[model.dev_offload < model.devices.size() ? model.dev_offload : 0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            // --moe pipe-stock=1 (research): the stock micro-batch pipeline across the cards although the experts are host-resident (tensor overrides). A prompt of several
            // micro-batches (-b > -ub) then has card 1 work on micro-batch k+1 while card 2 works on k: the overlap the single-pass traces lack.
            (!model.has_tensor_overrides() || [] { const char * e = llama_moe_get_opt("pipe-stock"); return e && atoi(e) != 0; }()) &&
            cparams.moe_cache_size == 0 && // not supported by the upstream MoE cache
            // --moe pipeline=0: no micro-batch pipeline (research A/B against the dense split)
            [] { const char * e = llama_moe_get_opt("pipeline"); return !(e && atoi(e) == 0); }();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        // why the micro-batch pipeline is on or off: the conditions above, one line (a layer layout of the fork may not meet the stock ones)
        LLAMA_LOG_WARN("%s: pipeline parallelism %s (devices %d, gpu layers %d of %d, split mode %d, offload_kqv %d, tensor overrides %d, pipe-stock %d, moe cache slots %d)\n", __func__,
                pipeline_parallel ? "on" : "off", (int) model.n_devices(), (int) model.n_gpu_layers(), (int) model.hparams.n_layer_all, (int) model.split_mode(),
                (int) cparams.offload_kqv, (int) model.has_tensor_overrides(), llama_moe_get_opt("pipe-stock") ? atoi(llama_moe_get_opt("pipe-stock")) : 0, (int) cparams.moe_cache_size);
        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache) {
            // multi-GPU prefill split: decided (and reserved) before the first compute-buffer reserve
            std::vector<ggml_backend_dev_t> gpus;
            for (auto & backend : backends) {
                ggml_backend_dev_t d = ggml_backend_get_device(backend.get());
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpus.push_back(d); }
            }
            llama_moe_cache_prefill_decide_fork(gpus);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size);
        }

        sched_reserve();

        // the expert cache starts on the first non-warmup decode (moe_cache_start), so auto
        // sizing (-1) sees the free VRAM left after every model and context is loaded,
        // e.g. a speculative draft model the server loads after this context
        cparams.moe_cache_slots   = params.n_moe_cache_slots;
        cparams.moe_cache_inserts = params.n_moe_cache_inserts;
        cparams.moe_cache_window  = params.n_moe_cache_window > 0 ? params.n_moe_cache_window : 64;
        cparams.moe_predict       = std::max(0, params.n_moe_predict);
        cparams.moe_predict_train = std::max(0, params.n_moe_predict_train);

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}


// research (--moe log=l, log=l64): every generated token (l) or every 64th (l64), per layer the layer's output embedding (1 byte per float, hex) and the expert ids its router selected,
// appended to layertrace.csv in the log directory: model,pos,layer,embhex,xhex,ids. Unsampled tokens answer "do not observe", so the graph runs as without it
namespace {
struct layer_trace_state {
    llama_context * ctx = nullptr;
    ggml_backend_sched_eval_callback user_cb = nullptr;
    void * user_data = nullptr;
    int every = 0;
    bool ids_only = false;   // log=x: the selected expert ids only, no embeddings
    std::string model;
    std::string file;   // the model file name without .gguf and split suffix plus the start time, for <file>.<time>.routing
    size_t nl_first = 0, k_first = 0, ne_first = 0;   // log=x: layers and experts per layer of the first complete sample; others are skipped
    int last_pos = -1, seq = 0;   // log=x: a position that goes down is a new prompt: its own file (<file>-<seq>.routing)
    std::map<int, std::string> hex, xhex, ids;
    std::map<int, std::vector<int32_t>> idv;
    std::map<int, std::vector<uint16_t>> prv;   // log=x: per layer the router probabilities of all experts (fp16): the relevance of every expert to this token   // log=x: per layer the selected experts of the sampled token   // per layer: output embedding, router input, selected experts
};
layer_trace_state g_lt;

bool layer_trace_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const bool is_out  = strncmp(t->name, "l_out-", 6) == 0;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_probs = g_lt.ids_only && strncmp(t->name, "ffn_moe_probs-", 14) == 0;   // after the gating function, before the selection bias
    const bool is_x    = strncmp(t->name, "ffn_norm-", 9) == 0;   // the router's (and the experts') input: the normalised residual
    bool mine = (g_lt.ids_only ? (is_topk || is_probs) : (is_out || is_topk || is_x)) && t->ne[1] == 1 && g_lt.ctx->trace_n_eval() > 0 && g_lt.ctx->trace_n_eval() % g_lt.every == 0;
    if (ask) {
        const bool theirs = g_lt.user_cb ? g_lt.user_cb(t, true, g_lt.user_data) : false;
        return mine || theirs;
    }
    if (mine) {
        const int il = atoi(strchr(t->name, '-') + 1);
        if (is_probs) {
            if (t->type == GGML_TYPE_F32) {
                std::vector<float> f(t->ne[0]);
                ggml_backend_tensor_get(t, f.data(), 0, f.size()*sizeof(float));
                std::vector<uint16_t> h(f.size());
                ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) h.data(), (int64_t) f.size());
                g_lt.prv[il] = std::move(h);
            }
        } else if (is_topk && t->type == GGML_TYPE_I32 && t->ne[0] == (int64_t) g_lt.ctx->get_model().hparams.n_expert_used(il)) {   // one token's experts (a warm-up graph can pass n_tokens x k as one row)
            std::vector<int32_t> v(t->ne[0]);
            ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(int32_t));
            if (g_lt.ids_only) {
                g_lt.idv[il] = v;
                if (il == (int) g_lt.ctx->get_model().hparams.n_layer() - 1) {   // the last layer closes the sample
                    if (g_lt.nl_first == 0) { g_lt.nl_first = g_lt.idv.size(); g_lt.k_first = v.size(); g_lt.ne_first = g_lt.prv.size() == g_lt.idv.size() ? g_lt.prv.begin()->second.size() : 0; }
                    if (g_lt.idv.size() != g_lt.nl_first || v.size() != g_lt.k_first || (g_lt.ne_first && (g_lt.prv.size() != g_lt.idv.size() || g_lt.prv.begin()->second.size() != g_lt.ne_first))) { g_lt.idv.clear(); g_lt.prv.clear(); return g_lt.user_cb ? g_lt.user_cb(t, false, g_lt.user_data) : true; }   // incomplete sample
                    // version 2 (this): header "MOER", u16 2, u16 layers, u16 k, u16 n_expert (0: no probabilities), u16 layer numbers; per token i32 pos, layers x k u16 ids, then
                    // layers x n_expert fp16 router probabilities (the relevance of every expert to the token, from the router the model computes anyway)
                    // <model file>[.<tag>].<YYYYmmdd-HHMMSS>.routing (tag = --moe tag=NAME, run.py sets test-prompt; one file per prompt: a position that goes down starts <file>-<n>.routing), little endian: once "MOER", u16 version 1, u16 layers, u16 k, u16 layer numbers;
                    // then per sampled token i32 pos and layers x k u16 expert ids (layer order as in the header). A run starts where pos goes down.
                    { const int cur = g_lt.ctx->trace_n_p_eval() + g_lt.ctx->trace_n_eval(); if (cur < g_lt.last_pos) { g_lt.seq++; } g_lt.last_pos = cur; }
                    const std::filesystem::path path = moe_log_file(("routing/" + g_lt.file + (g_lt.seq ? "-" + std::to_string(g_lt.seq) : std::string()) + ".routing").c_str());
                    std::error_code ec;
                    std::filesystem::create_directories(path.parent_path(), ec);   // <logdir>/routing/
                    const bool fresh = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
                    const uint16_t k = (uint16_t) v.size();
                    std::string out;
                    auto put = [&](const void * p, size_t n) { out.append((const char *) p, n); };
                    if (fresh) {
                        const uint16_t ver = 2, nl = (uint16_t) g_lt.idv.size(), ne = (uint16_t) g_lt.ne_first;
                        put("MOER", 4); put(&ver, 2); put(&nl, 2); put(&k, 2); put(&ne, 2);
                        for (const auto & e : g_lt.idv) { const uint16_t l = (uint16_t) e.first; put(&l, 2); }
                    }
                    const int32_t pos = g_lt.ctx->trace_n_p_eval() + g_lt.ctx->trace_n_eval();
                    put(&pos, 4);
                    for (const auto & e : g_lt.idv) {
                        for (uint16_t i = 0; i < k; ++i) { const uint16_t id = i < e.second.size() ? (uint16_t) e.second[i] : 0xffff; put(&id, 2); }
                    }
                    if (g_lt.ne_first) {
                        for (const auto & e : g_lt.prv) { put(e.second.data(), e.second.size()*2); }
                    }
                    std::ofstream f(path, std::ios::app | std::ios::binary);
                    f.write(out.data(), (std::streamsize) out.size());
                    g_lt.idv.clear(); g_lt.prv.clear();
                }
            } else {
                std::string o;
                for (size_t i = 0; i < v.size(); ++i) { o += (i ? " " : "") + std::to_string(v[i]); }
                g_lt.ids[il] = o;
            }
        } else if (is_x && t->type == GGML_TYPE_F32) {
            // fixed scale, clipped at +-16 (int8, 2 hex characters per dim): unlike the per-token absmax scale of the output embedding, the outlier dims do not hide the rest
            std::vector<float> v(t->ne[0]);
            ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
            std::string h;
            h.reserve(v.size()*2);
            char b[4];
            for (float x : v) { snprintf(b, sizeof b, "%02x", (unsigned) (uint8_t) (int8_t) std::lround(std::max(-16.0f, std::min(16.0f, x))*(127.0f/16.0f))); h += b; }
            g_lt.xhex[il] = h;
        } else if (is_out && t->type == GGML_TYPE_F32) {
            std::vector<float> v(t->ne[0]);
            ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
            float mx = 1e-20f;
            for (float x : v) { mx = std::max(mx, std::fabs(x)); }
            std::string h;
            h.reserve(v.size()*2);
            char b[4];
            for (float x : v) { snprintf(b, sizeof b, "%02x", (unsigned) (uint8_t) (int8_t) std::lround(x/mx*127.0f)); h += b; }
            g_lt.hex[il] = h;
            if (il == (int) g_lt.ctx->get_model().hparams.n_layer() - 1) {   // the last layer closes the sample
                const std::filesystem::path path = moe_log_file("layertrace.csv");
                std::error_code ec;
                const bool fresh = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
                std::string out = fresh ? "model,pos,layer,embhex,xhex,ids\n" : "";
                const int pos = g_lt.ctx->trace_n_p_eval() + g_lt.ctx->trace_n_eval();
                for (const auto & e : g_lt.hex) { out += g_lt.model + "," + std::to_string(pos) + "," + std::to_string(e.first) + "," + e.second + "," + (g_lt.xhex.count(e.first) ? g_lt.xhex[e.first] : "") + "," + (g_lt.ids.count(e.first) ? g_lt.ids[e.first] : "") + "\n"; }
                std::ofstream o(path, std::ios::app | std::ios::binary);
                o.write(out.data(), (std::streamsize) out.size());
                g_lt.hex.clear(); g_lt.xhex.clear(); g_lt.ids.clear();
            }
        }
    }
    return g_lt.user_cb ? g_lt.user_cb(t, false, g_lt.user_data) : true;
}
}  // namespace

// research (--moe log=t): per-layer timing of every prompt batch (more than one token). The scheduler is asked for the output of each layer;
// it computes up to such a node, synchronises the card that ran it and calls back, so the time between two callbacks is that layer on that card including the upload
// waits and the hand-off copy from the previous card. Appended to prefill-timing.csv in the log directory: file,pass,n_tokens,layer,backend,dt_us,cum_us. The
// synchronisation at every layer breaks the overlap of uploads across layers a little, so the totals are somewhat above an undisturbed pass; the point is the
// shares: how long a layer takes on each card, and what a card boundary costs.
namespace {
struct prefill_trace_state {
    llama_context * ctx = nullptr;
    ggml_backend_sched_eval_callback user_cb = nullptr;
    void * user_data = nullptr;
    int64_t t0 = 0, t_prev = 0;   // start of the pass (its first layer's output) and the previous callback
    int pass = 0, last_il = -1;
    std::string file;
    std::string rows;
};
prefill_trace_state g_pt;

bool prefill_trace_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const bool mine = strncmp(t->name, "l_out-", 6) == 0 && t->ne[1] > 1;   // (inp_embd is an input leaf, the scheduler never calls back for it)
    if (ask) {
        const bool theirs = g_pt.user_cb ? g_pt.user_cb(t, true, g_pt.user_data) : false;
        return mine || theirs;
    }
    if (mine) {
        const int64_t now = ggml_time_us();
        const int il = atoi(strchr(t->name, '-') + 1);
        if (g_pt.t0 == 0 || il <= g_pt.last_il) {   // the layer number went down (or this is the first callback): a new pass starts, its first layer has no start time and gets dt 0
            g_pt.t0 = g_pt.t_prev = now;
            g_pt.pass++;
            g_pt.rows.clear();
        }
        g_pt.last_il = il;
        g_pt.rows += g_pt.file + "," + std::to_string(g_pt.pass) + "," + std::to_string(t->ne[1]) + "," + std::to_string(il) + "," + g_pt.ctx->trace_backend_name(t) + "," +
                     std::to_string(now - g_pt.t_prev) + "," + std::to_string(now - g_pt.t0) + "\n";
        g_pt.t_prev = now;
        if (il == (int) g_pt.ctx->get_model().hparams.n_layer() - 1) {   // the last layer closes the pass
            const std::filesystem::path path = moe_log_file("prefill-timing.csv");
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            const bool fresh = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
            std::ofstream o(path, std::ios::app | std::ios::binary);
            if (fresh) { o << "file,pass,n_tokens,layer,backend,dt_us,cum_us\n"; }
            o.write(g_pt.rows.data(), (std::streamsize) g_pt.rows.size());
            g_pt.rows.clear();
            g_pt.t0 = 0;
            g_pt.last_il = -1;
        }
    }
    return g_pt.user_cb ? g_pt.user_cb(t, false, g_pt.user_data) : true;
}
}  // namespace

const char * llama_context::trace_backend_name(const ggml_tensor * t) const {
    ggml_backend_t b = sched ? ggml_backend_sched_get_tensor_backend(sched.get(), const_cast<ggml_tensor *>(t)) : nullptr;
    return b ? ggml_backend_name(b) : "?";
}

void llama_context::set_eval_cb() {
    if ((moe_log_has("l") || moe_log_has("x")) && moe_state_enabled() && (g_lt.ctx == nullptr || g_lt.ctx == this)) {   // the first context only: a draft context has its own layers and would interleave
        g_lt.ids_only = !moe_log_has("l");
        g_lt.ctx = this; g_lt.user_cb = cparams.cb_eval; g_lt.user_data = cparams.cb_eval_user_data;
        g_lt.every = std::max(1, moe_log_num(g_lt.ids_only ? 'x' : 'l', 1));   // log=l: every token, log=l64: every 64th; log=x: ids only, same sampling
        if (g_lt.file.empty()) {   // once per process: the file name (model, tag, start time)
            g_lt.model = moe_state_section(model);
            for (char & c : g_lt.model) { if (c == ',' || c == '\n') { c = '_'; } }
            g_lt.file = moe_run_stem(g_lt.model);   // the same stem for every research file of the run (routing, cache pressure)
        }
        ggml_backend_sched_set_eval_callback(sched.get(), layer_trace_cb, nullptr);
    } else if (moe_log_has("t") && (g_pt.ctx == nullptr || g_pt.ctx == this)) {   // the first context only
        g_pt.ctx = this; g_pt.user_cb = cparams.cb_eval; g_pt.user_data = cparams.cb_eval_user_data;
        if (g_pt.file.empty()) {
            g_pt.file = moe_state_section(model);
            for (char & c : g_pt.file) { if (c == ',' || c == '\n') { c = '_'; } }
            g_pt.file = moe_run_stem(g_pt.file);
        }
        ggml_backend_sched_set_eval_callback(sched.get(), prefill_trace_cb, nullptr);
    } else {
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
    }
}

llama_context::~llama_context() {
    if (g_lt.ctx == this) { g_lt.ctx = nullptr; }   // research trace (log=l, log=x): the callback state must not point at a freed context
    if (moe_embsnap_enabled()) {   // research (--moe log=e): at exit only, the last layer embedding of the last token of the last graph; nothing is read while running
        synchronize();
        llm_graph_result * r = gf_res_prev_active ? gf_res_prev_active : gf_res_prev[0].get();
        ggml_tensor * te = r ? r->get_embd() : nullptr;
        if (te && te->buffer && te->ne[1] > 0) {
            std::vector<float> row(te->ne[0]);
            ggml_backend_tensor_get(te, row.data(), (size_t) (te->ne[1] - 1)*te->nb[1], row.size()*sizeof(float));
            moe_embsnap_set(row.data(), (int) row.size());
        }
    }
    if (cparams.moe_cache) {
        llama_moe_cache_free_fork();
    } else if (moe_observe_started) {
        llama_moe_observe_save();
    }
    if (moe_decoded) { moe_snap_save(moe_state_section(model)); }   // research (--moe log=s): the model's state section after the run's last write
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.no_alloc() && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

// fork: layers a and b trade GPUs without allocating anything. Every pair of their tensors (weights in VRAM, memory tensors: KV, indexer caches,
// recurrent states) must match in type and shape. Each pair is copied crosswise through host memory, then the two tensors trade storage
// (data, buffer, extra), so layer a's tensors point at the memory on b's GPU that now holds a's data. The expert caches trade cards too
// (llama_moe_cache_swap_layers_fork). Then the layer -> device map changes and the scheduler plans the graph again. Call between batches.
bool llama_context::swap_layers_fork(int a, int b) {
    if (a == b || a < 0 || b < 0 || a >= (int) model.hparams.n_layer_all || b >= (int) model.hparams.n_layer_all || model.dev_layer(a) == model.dev_layer(b)) {
        LLAMA_LOG_WARN("%s: layers %d and %d: not two layers on different devices\n", __func__, a, b);
        return false;
    }
    synchronize();
    const int64_t t0 = ggml_time_us();
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> pairs;
    auto pair_up = [&](const std::map<std::string, ggml_tensor *> & by_name, const std::string & pa, const std::string & pb, bool suffix) -> bool {
        for (const auto & [name, ta] : by_name) {
            const bool mine = suffix ? (name.size() > pa.size() && name.compare(name.size() - pa.size(), pa.size(), pa) == 0)
                                     : name.compare(0, pa.size(), pa) == 0;
            if (!mine) {
                continue;
            }
            const std::string nb = suffix ? name.substr(0, name.size() - pa.size()) + pb : pb + name.substr(pa.size());
            auto it = by_name.find(nb);
            if (it == by_name.end()) {
                LLAMA_LOG_WARN("%s: %s has no counterpart %s\n", __func__, name.c_str(), nb.c_str());
                return false;
            }
            ggml_tensor * tb = it->second;
            if (!ta->buffer || !tb->buffer) {
                continue;
            }
            const bool ha = ggml_backend_buffer_is_host(ta->buffer), hb = ggml_backend_buffer_is_host(tb->buffer);
            if (ha && hb) {
                continue;   // both in RAM (experts with --cpu-moe): nothing moves
            }
            if (ha != hb || ta->type != tb->type || !ggml_are_same_shape(ta, tb) || ggml_nbytes(ta) != ggml_nbytes(tb) || ta->view_src || tb->view_src) {
                LLAMA_LOG_WARN("%s: %s and %s differ in type, shape or residence\n", __func__, name.c_str(), nb.c_str());
                return false;
            }
            pairs.push_back({ ta, tb });
        }
        return true;
    };
    std::map<std::string, ggml_tensor *> weights;
    for (const auto & [name, t] : model.tensors_by_name) { weights[name] = t; }
    std::map<std::string, ggml_tensor *> mem;
    if (memory && !memory->visit_tensors_fork([&](ggml_tensor * t) { mem[ggml_get_name(t)] = t; })) {
        LLAMA_LOG_WARN("%s: this memory type cannot move layers yet\n", __func__);
        return false;
    }
    if (!pair_up(weights, "blk." + std::to_string(a) + ".", "blk." + std::to_string(b) + ".", false) ||
        !pair_up(mem, "_l" + std::to_string(a), "_l" + std::to_string(b), true)) {
        return false;
    }
    if (!llama_moe_cache_swap_layers_fork(a, b)) {
        return false;
    }
    size_t bytes = 0;
    std::vector<uint8_t> da, db;
    for (auto & [ta, tb] : pairs) {
        const size_t n = ggml_nbytes(ta);
        da.resize(n); db.resize(n);
        ggml_backend_tensor_get(ta, da.data(), 0, n);
        ggml_backend_tensor_get(tb, db.data(), 0, n);
        ggml_backend_tensor_set(tb, da.data(), 0, n);
        ggml_backend_tensor_set(ta, db.data(), 0, n);
        std::swap(ta->data, tb->data);
        std::swap(ta->buffer, tb->buffer);
        std::swap(ta->extra, tb->extra);
        bytes += 2*n;
    }
    const_cast<llama_model &>(model).swap_layer_dev_fork(a, b);   // the context holds the model const; the placement is the one thing a swap changes
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;
    sched_need_reserve = true;
    sched_reserve();
    LLAMA_LOG_WARN("%s: layers %d and %d traded GPUs: %zu tensors, %.1f MiB copied, %.0f ms\n", __func__, a, b, pairs.size(), bytes / 1048576.0, (ggml_time_us() - t0) / 1000.0);
    return true;
}

// fork: --moe max-batch=auto: how long a 16- and a 64-token batch take on the cache path, and a streamed pass of two larger sizes (second run of each: graphs built, uploads settled),
// on varied tokens so the routing is not one fixed expert set. The context is cleared after each run.
void llama_context::moe_probe_costs_fork() {
    const int64_t t_start = ggml_time_us();
    const llama_token n_vocab = (llama_token) model.vocab.n_tokens();
    llama_moe_cache_probe_begin_fork();
    auto run = [&](int n) -> double {
        llama_batch_ext batch(this);
        for (int i = 0; i < n; ++i) {
            const int32_t idx = batch.add_token(0);
            llama_pos pos[GGML_MROPE_SECTIONS] = { i, i, i, 0 };
            if (idx < 0 || !batch.set_token_id(idx, (llama_token) ((1000 + (int64_t) i * 7919) % std::max<llama_token>(n_vocab, 2))) ||
                !batch.set_token_pos(idx, pos) || !batch.set_output(idx, i == n - 1)) {
                return -1.0;
            }
        }
        synchronize();
        const int64_t t0 = ggml_time_us();
        const int ret = decode(batch);
        synchronize();
        const double dt = (double) (ggml_time_us() - t0);
        memory->clear(true);
        return ret == 0 ? dt : -1.0;
    };
    const int cap = (int) llama_moe_cache_max_batch_cap_fork();
    const double cn[3] = { 16, 64, (double) cap };
    double cus[3] = { -1, -1, -1 };
    if (run(4) > 0) {
        for (int i = 0; i < 3; ++i) {
            run((int) cn[i]);
            cus[i] = run((int) cn[i]);
        }
    }
    // streamed passes (above 0 tokens forced onto the streaming path), sizes up to the micro-batch
    double sn[4], sus[4];
    int ns = 0;
    if (cus[2] > 0) {
        llama_moe_cache_probe_stream_fork(true);
        for (int n : { 64, 128, 256, 512 }) {
            if (n > (int) cparams.n_ubatch) { break; }
            run(n);
            sn[ns] = n; sus[ns] = run(n); ++ns;
        }
        llama_moe_cache_probe_stream_fork(false);
    }
    llama_moe_cache_probe_end_fork(cn, cus, 3, sn, sus, ns);
    LLAMA_LOG_WARN("%s: cache-path probe took %.1f s\n", __func__, (ggml_time_us() - t_start) / 1e6);
}

// fork: the compute alias lends the pool's first blocks to the worst-case graph, its CUDA scratch and the prefetch staging slots. When they do not fit
// the micro-batch gets smaller (scaled by lent / needed bytes, down to 512) and the scheduler is reserved again; at 512 the staging slots are skipped for graphs that cannot
// lend them (prefetch off, no allocation), and if even the graph does not fit the alias is dropped and the scheduler keeps its own buffers.
// Returns true when sched_reserve() must run again.
bool llama_context::moe_alias_shrink_fork(bool graph_fits, bool stage_fits, size_t need, size_t cap) {
    if (graph_fits && stage_fits) {
        return false;
    }
    // The staging slots alone never cut the micro-batch (--moe stage-shrink=1: the old rule). Every pass streams every expert the cards do not hold, so
    // a pass more costs a whole upload, while the slots only hide uploads under compute: MiMo-V2.6-Flash 3-bit, 4k prompt, one x16 card, prompt tokens/s:
    // 6 slots 127 (cut to 512-token passes, 8 passes), 3 slots 128 (same), 2 slots 471 (2208, 2 passes), no slots 544 (5504, one pass). A graph gets as
    // many slots as the pool still lends next to it (llama_moe_cache_alias_fork, at least 3), and the prompt's passes are sized by the pass planner
    // (llama_moe_cache_prefill_ubatch_fork), which knows that a big pass may run without slots.
    static const bool stage_shrink = [] { const char * e = llama_moe_get_opt("stage-shrink"); return e && atoi(e) != 0; }();
    if (graph_fits && !stage_shrink) {
        static bool said = false;
        if (!said) {
            said = true;
            LLAMA_LOG_WARN("%s: the pool lends %.0f MiB, graph + scratch + all staging slots need %.0f MiB at micro-batch %u: big passes get fewer or no slots, the micro-batch stays\n",
                    __func__, cap / 1048576.0, need / 1048576.0, cparams.n_ubatch);
        }
        return false;
    }
    // a graph that does not fit is sized in sched_reserve (fit_ub: measured graphs against each pool's room, no guessed cut); here only what is left: the alias
    // off when not even 512 tokens fit, or the staging slots with stage-shrink=1 (the old rule), which steps down by one 32-token granule
    if (graph_fits && cparams.n_ubatch > 512) {
        const uint32_t nu = cparams.n_ubatch - 32;
        LLAMA_LOG_WARN("%s: the pool lends %.0f MiB, graph + scratch + prefetch staging need %.0f MiB at micro-batch %u: micro-batch %u (stage-shrink=1)\n", __func__,
                cap / 1048576.0, need / 1048576.0, cparams.n_ubatch, nu);
        cparams.n_ubatch = nu;
        sched_need_reserve = true;
        return true;
    }
    if (!graph_fits) {
        LLAMA_LOG_WARN("%s: the pool cannot lend the graph even at micro-batch 512: compute buffers stay separate (--moe alias=0)\n", __func__);
        llama_moe_cache_alias_drop_fork();
        moe_alias_bufs.clear();
        sched_need_reserve = true;
        return true;
    }
    LLAMA_LOG_WARN("%s: no room for the prefetch staging slots at micro-batch 512: prefetch is skipped for graphs that cannot lend them\n", __func__);
    return false;
}

// fork: --moe alias=1: the GPUs' compute graphs go into the expert pool buffers (llama_moe_cache_alias_fork lends the blocks they reach)
void llama_context::apply_moe_alias() {
    // prefetch overlap: --prefetch-experts-slots / --moe pf-slots; unset (-1) = the expert tensors of one layer (up, gate and down; 2 when gate and up are
    // one tensor): the next layer's uploads run while this one computes. They are only filled when expert weights are host-resident and a prompt batch computes
    // them on a GPU, so there is nothing to allocate otherwise; how many a graph gets is what its pool holds behind it (ppx_stage_slots). 0 = off
    {
        int n = cparams.prefetch_experts_slots;
        if (n < 0) {
            n = 0;
            for (const auto & l : model.layers) {
                n = std::max(n, (l.ffn_up_exps != nullptr) + (l.ffn_gate_exps != nullptr) + (l.ffn_down_exps != nullptr) + (l.ffn_gate_up_exps != nullptr));
            }
        }
        ggml_backend_sched_set_prefetch_experts_slots(sched.get(), n);
    }
    // a new scheduler reserves its worst case: a pass that splits (behind the dense split's copies, --moe mirror-pool) when the split may be used
    moe_alias_off.assign(moe_alias_bufs.size(), 0);
    for (size_t i = 0; i < moe_alias_bufs.size(); ++i) {
        const auto & ab = moe_alias_bufs[i];
        const size_t off = std::min(ab.capacity, llama_dsplit_mirror_offset_fork(model, ab.dev, true));
        moe_alias_off[i] = off;
        llama_moe_cache_alias_offset_fork(ab.dev, off);
        for (auto & be : backends) {
            if (ggml_backend_get_device(be.get()) == ab.dev) {
                ggml_backend_sched_set_external_buffer(sched.get(), be.get(), ab.buf, off, ab.capacity - off);
            }
        }
    }
}

// --moe mirror-pool, before a graph is placed: a pass that splits goes behind the dense split's copies in each card's pool, any other graph to the start
// of the pool (over the copies, which are then gone). A change only costs the allocator a new placement plan of that graph, nothing is allocated or freed.
void llama_context::moe_mirror_offsets() {
    for (size_t i = 0; i < moe_alias_bufs.size() && i < moe_alias_off.size(); ++i) {
        const auto & ab = moe_alias_bufs[i];
        const size_t off = std::min(ab.capacity, llama_dsplit_mirror_offset_fork(model, ab.dev, false));
        if (off == 0) {
            llama_dsplit_mirror_invalidate_fork(model, ab.dev);
        }
        if (off == moe_alias_off[i]) {
            continue;
        }
        moe_alias_off[i] = off;
        llama_moe_cache_alias_offset_fork(ab.dev, off);
        for (auto & be : backends) {
            if (ggml_backend_get_device(be.get()) == ab.dev) {
                ggml_backend_sched_set_external_buffer(sched.get(), be.get(), ab.buf, off, ab.capacity - off);
            }
        }
    }
}

void llama_moe_cache_ledger_fork(ggml_backend_dev_t dev, const char * kind, const void * ptr, size_t bytes, const char * who);   // llama-moecache-fork.cpp (no header edit)

// --moe mirror-pool, after the pool lent this graph its blocks: copies that the region lost since they were last filled are copied in again from the cards
// that hold the originals (card to card). The graph runs only after this on the receiving card's stream. Every card's copies are issued before anything waits
// (2026-10-23): a copy runs on its source card's stream, so the cards fill each other at once, both directions of a link in use (one card after the other
// with a wait after each, GLM-5.3-Flash took 848 + 750 ms per request on x16 + x4).
void llama_context::moe_mirror_fill() {
    const int64_t t0 = ggml_time_us();
    std::string txt;
    size_t total = 0;
    for (size_t i = 0; i < moe_alias_bufs.size(); ++i) {
        const auto & ab = moe_alias_bufs[i];
        // a pass that does not split has its graph at the start of the pool, where the copies are: filling them now would overwrite that graph's tensors
        if (i < moe_alias_off.size() && moe_alias_off[i] == 0) {
            continue;
        }
        const auto todo = llama_dsplit_mirror_take_upload_fork(model, ab.dev);
        if (todo.empty()) {
            continue;
        }
        ggml_backend_t bd = nullptr;
        for (auto & be : backends) {
            if (ggml_backend_get_device(be.get()) == ab.dev) { bd = be.get(); }
        }
        size_t bytes = 0;
        for (const auto & pr : todo) {
            ggml_backend_dev_t ds = pr.first->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pr.first->buffer)) : nullptr;
            ggml_backend_t bs = nullptr;
            for (auto & be : backends) {
                if (ggml_backend_get_device(be.get()) == ds) { bs = be.get(); }
            }
            llama_moe_cache_ledger_fork(ab.dev, "mirror", pr.second->data, ggml_nbytes(pr.second), pr.second->name);   // --moe ledger=1
            if (bs != nullptr && bd != nullptr) {
                ggml_backend_tensor_copy_async(bs, bd, pr.first, pr.second);
            } else {
                ggml_backend_tensor_copy(pr.first, pr.second);
            }
            // the copy sits in pool blocks that held experts until now: the row padding behind a quantized copy holds their bytes, and the matmuls read
            // past the last row there (a WEIGHTS buffer is not cleared at use): zeros behind it (see ggml_backend_tensor_zero_tail_fork)
            if (bd != nullptr && ggml_is_quantized(pr.second->type)) {
                ggml_backend_tensor_zero_tail_fork(bd, pr.second);
            }
            // the card's link traffic of this prompt's first pass (the split pass report: from host memory, or from the card that holds the weight)
            ggml_backend_sched_link_add_fork(ab.dev, ds && ggml_backend_dev_type(ds) == GGML_BACKEND_DEVICE_TYPE_GPU ? ds : nullptr, ggml_nbytes(pr.first));
            bytes += ggml_nbytes(pr.first);
        }
        txt += format("%s%.1f MiB on %s", txt.empty() ? "" : ", ", bytes / 1048576.0, ggml_backend_dev_name(ab.dev));
        total += bytes;
    }
    if (total == 0) {
        return;
    }
    ggml_backend_sched_synchronize(sched.get());
    for (auto & be : backends) {
        ggml_backend_synchronize(be.get());
    }
    LLAMA_LOG_WARN("%s: dense split: copies filled at once: %s in %.0f ms (pool blocks lent to this prompt)\n", __func__, txt.c_str(), (ggml_time_us() - t0) / 1e3);
    llama_dsplit_note_fill_fork((ggml_time_us() - t0) / 1e3);   // the split's per-prompt cost, kept apart from the pass's time per token (dsf)
}

void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);
}

// qkv-ram-fallback: the tier a request of n_tokens (prompt + answer) needs, and the rebuild when it differs from the cache's. Returns the tier (0 f16, 1 k8v8, 2 k4v4, 3 ram),
// -1 when no enabled tier holds it (note says why), -2 when the pool does not exist yet (the first decode starts it: this request runs as the cache is). Bit 8 is set when the cache was
// rebuilt (the caller drops what it knew of the old one).
int32_t llama_context::kv_fit_select(uint32_t n_tokens, int32_t min_tier, char * note, size_t note_size) {
    auto say = [&](const char * fmt, auto... args) { if (note && note_size) { snprintf(note, note_size, fmt, args...); } };
    const llama_kv_fit_caps cp = llama_moe_cache_kv_fit_caps_fork();
    if (!cp.valid) { return -2; }
    const char * lst = llama_moe_get_opt("kv-fallback");
    std::string list = lst ? lst : "k8v8,k4v4,ram";
    std::vector<std::string> unknown;
    const std::vector<int> allowed = kvx_allowed(list, &unknown);
    static bool warned_list = false;
    if (!unknown.empty() && !warned_list) {   // once: the words that are no tier (k8v8, k4v4, ram, or none) are left out
        warned_list = true;
        std::string w; for (auto & u : unknown) { w += (w.empty() ? "'" : ", '") + u + "'"; }
        LLAMA_LOG_WARN("%s: --moe kv-fallback=%s: not a tier: %s (tiers are k8v8, k4v4, ram; none = f16 only)\n", __func__, list.c_str(), w.c_str());
    }   // tiers tried after f16, in place: q8_0, q4_0 in the pool, then whole layers' KV in host memory (llama-kvtier-fork.h)
    const int tier = kvx_select(cp, allowed, kv_tier, min_tier, n_tokens);
    if (tier < 0) {
        say("kv-fit: %u tokens fit no enabled tier: f16 %zu, q8_0 %zu, q4_0 %zu, RAM %zu tokens (list %s; --moe kv-fallback=k8v8,k4v4,ram)", n_tokens, cp.f16, cp.k8v8, cp.k4v4, cp.ram, list.c_str());
        return -1;
    }
    if (tier == 3) {   // q4_0 in the pool first (if the list has it), then whole layers to host memory until the request fits
        synchronize();
        char why[384] = { 0 };
        if (kvx_ram_needs_q4(allowed, kv_tier)) {
            if (!llama_moe_cache_kv_retype_fork(sched.get(), 2, why, sizeof(why))) { say("%s", why); return -1; }
            kv_tier = 2;
        }
        if (!llama_moe_cache_kv_ram_fork(sched.get(), n_tokens, why, sizeof(why))) { say("%s", why); return -1; }
        kv_tier = 3;
        say("kv-fit: %u tokens: %s", n_tokens, why);
        llama_moe_decision_fork("kv", "part in RAM", why);   // shown live by run.py (decided kv ...)
        return 3 | 256;
    }
    if (tier == kv_tier) { return tier; }
    if (kv_tier == 3) { return kv_tier; }   // layers in host memory stay there for now (moving them back: not written yet)
    synchronize();
    char why[256] = { 0 };
    if (!llama_moe_cache_kv_retype_fork(sched.get(), tier, why, sizeof(why))) {
        if (tier < kv_tier) {   // back down (a shorter request) while the cache still holds more than the lower tier fits: stay, nothing is lost
            say("kv-fit: staying %s: %s", kv_tier == 1 ? "q8_0/q8_0" : "q4_0/q4_0", why);
            return kv_tier;
        }
        say("%s", why[0] ? why : "kv-fit: the KV could not change tier");
        return -1;
    }
    kv_tier = tier;
    static const char * names[] = { "f16", "q8_0/q8_0", "q4_0/q4_0" };
    say("kv-fit: %u tokens: KV now %s in place (f16 holds %zu, q8_0 %zu, q4_0 %zu)%s", n_tokens, names[tier], cp.f16, cp.k8v8, cp.k4v4, tier ? ", small quality loss" : "");
    { char why[160]; snprintf(why, sizeof why, "%u tokens: f16 holds %zu, q8_0 %zu, q4_0 %zu tokens%s", n_tokens, cp.f16, cp.k8v8, cp.k4v4, tier ? " (small quality loss)" : "");
      llama_moe_decision_fork("kv", names[tier], why); }
    return tier | 256;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    sched_new(max_nodes, cparams.pipeline_parallel);
    apply_moe_alias();

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    // fork: a prompt pass outputs one token per sequence (the last token's logits; embeddings pool per sequence). With the expert cache the compute buffers
    // are sized for that: before the pool exists they are given back to the cards when it is sized, and after it each graph gets its lend when it runs (a
    // graph with more outputs too). The default n_outputs_max (n_batch, 131072 in the server cells) sized GLM 3.0's 16384-token reserve with 16384 x 151552
    // logits: a 30 GiB buffer, refused, and the retry halved the micro-batch for good (2026-10-24).
    const uint32_t n_outputs_pp = cparams.moe_cache ? n_seqs : std::min(n_tokens, cparams.n_outputs_max);

    // fork: with the pool lending the graphs, the micro-batch is the largest prompt pass whose graph every card's pool holds, computed (user 2026-10-24:
    // "simply calculate max ubatch properly"; before, a reserve that did not fit was cut by a fixed 0.78 and 0.97: GLM 3.0 8192 -> 6176 where 8192 and 16384
    // had run). Each size is checked with a measure-only reserve (nothing allocated) of the real graph, plus the CUDA scratch its nodes take on each card
    // (computed from them), against llama_moe_cache_alias_room_fork: the pool up to the KV zone of those cells, behind the dense split's copies. The first
    // probe below the configured size is the line through two measured sizes (graph bytes = a + b x tokens per card) against each card's room; then
    // bisection in 32-token steps between the largest size that fit and the smallest that did not. 0: not even 512 tokens fit.
    // once per process: what the prompt graph of each card is made of (its largest tensors), so a size that does not fit names its cause (GLM 3.0: 1.84 MiB per
    // token at the start, 0.5 in the real passes, 2026-10-24)
    std::vector<size_t> lt(backend_ptrs.size(), 0);
    auto list_top = [&](ggml_cgraph * mg, uint32_t u) {
        static bool listed = false;
        if (listed || !mg) { return; }
        listed = true;
        for (size_t bi = 0; bi < backend_ptrs.size(); ++bi) {
            ggml_backend_dev_t dv = ggml_backend_get_device(backend_ptrs[bi]);
            if (ggml_backend_dev_type(dv) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            std::vector<const ggml_tensor *> ts;
            for (int k = 0; k < ggml_graph_n_nodes(mg); ++k) {
                ggml_tensor * t = ggml_graph_node(mg, k);
                if (ggml_backend_sched_get_tensor_backend(sched.get(), t) == backend_ptrs[bi]) { ts.push_back(t); }
            }
            std::sort(ts.begin(), ts.end(), [](const ggml_tensor * x, const ggml_tensor * y) { return ggml_nbytes(x) > ggml_nbytes(y); });
            std::string top;
            for (size_t k = 0; k < ts.size() && k < 8; ++k) {
                top += format("%s %s [%lld,%lld,%lld,%lld] %.0f MiB; ", ts[k]->name, ggml_op_desc(ts[k]), (long long) ts[k]->ne[0], (long long) ts[k]->ne[1],
                        (long long) ts[k]->ne[2], (long long) ts[k]->ne[3], ggml_nbytes(ts[k]) / 1048576.0);
            }
            LLAMA_LOG_WARN("%s: prompt graph of %u tokens on %s: largest tensors: %s\n", __func__, u, ggml_backend_dev_name(dv), top.c_str());
        }
    };
    if (cparams.moe_cache && !model.no_alloc() && n_tokens > 512) {   // before the first real reserve (which allocates): the list, measure-only
        list_top(graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, lt.data()), n_tokens);
    }
    auto fit_ub = [&](uint32_t hi) -> uint32_t {
        std::vector<std::pair<size_t, ggml_backend_dev_t>> gi;   // backends whose card has a pool
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_dev_t d = ggml_backend_get_device(backend_ptrs[i]);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && llama_moe_cache_alias_room_fork(d, 1) > 0) { gi.push_back({ i, d }); }
        }
        if (gi.empty()) {
            return hi;
        }
        std::vector<size_t> s(backend_ptrs.size(), 0);
        int probes = 0;
        auto fits = [&](uint32_t u) {
            std::fill(s.begin(), s.end(), 0);
            ggml_cgraph * mg = graph_reserve(u, n_seqs, n_seqs, mctx.get(), true, s.data());
            probes++;
            list_top(mg, u);
            if (mg) {
                for (const auto & [i, d] : gi) { s[i] += llama_moe_cache_graph_scratch_fork(sched.get(), d); }   // the reserve's splits are still in the scheduler
            }
            for (const auto & [i, d] : gi) { llama_moe_cache_note_graph_fork(d, u, s[i]); }
            for (const auto & [i, d] : gi) { if (s[i] > llama_moe_cache_alias_room_fork(d, u)) { return false; } }
            return true;
        };
        std::string line;
        auto say = [&](uint32_t u, const char * how) {
            for (const auto & [i, d] : gi) {
                line += format(" | %s: graph + scratch %.0f MiB, room %.0f MiB", ggml_backend_dev_name(d), s[i] / 1048576.0, llama_moe_cache_alias_room_fork(d, u) / 1048576.0);
            }
            LLAMA_LOG_WARN("%s: micro-batch from the pools: %u tokens (%s, %d measure-only reserves)%s\n", __func__, u, how, probes, line.c_str());
        };
        if (fits(hi)) {
            say(hi, "the configured size fits");
            return hi;
        }
        const std::vector<size_t> s_hi = s;
        uint32_t lo = 0, bad = hi;
        const uint32_t u1 = std::max<uint32_t>(512, hi / 2 / 32 * 32);
        if (u1 < hi) {
            if (fits(u1)) { lo = u1; } else { bad = u1; }
        }
        if (lo == 0 && bad <= 512) {
            say(0, "not even 512 tokens fit");
            return 0;
        }
        if (lo > 0) {   // the line through the two sizes, per card, against the card's room at each size: the first probe
            uint32_t a0 = lo, b0 = bad;
            while (b0 - a0 > 32) {
                const uint32_t m = (a0 + b0) / 2 / 32 * 32;
                bool ok = true;
                for (const auto & [i, d] : gi) {
                    const double b = s_hi[i] > s[i] ? (double) (s_hi[i] - s[i]) / (double) (hi - u1) : (double) s_hi[i] / (double) hi;
                    const double a = (double) s_hi[i] - b * (double) hi;
                    if (a + b * (double) m > (double) llama_moe_cache_alias_room_fork(d, m)) { ok = false; }
                }
                if (ok) { a0 = m; } else { b0 = m; }
            }
            if (a0 > lo && a0 < bad) {
                if (fits(a0)) { lo = a0; } else { bad = a0; }
                if (lo == a0 && a0 + 32 < bad) {   // the line was right if the next step does not fit
                    if (fits(a0 + 32)) { lo = a0 + 32; } else { bad = a0 + 32; }
                }
            }
        }
        while (bad - lo > 32) {
            const uint32_t m = std::max<uint32_t>(512, (lo + bad) / 2 / 32 * 32);
            if (m <= lo || m >= bad) { break; }
            if (fits(m)) { lo = m; } else { bad = m; }
        }
        if (lo > 0) { fits(lo); }   // the sizes of the answer for the log line
        say(lo, lo > 0 ? "the largest that fits" : "not even 512 tokens fit");
        return lo;
    };
    static const llama_context * ub_fitted = nullptr;   // computed once per context, when its pools first lend the graphs
    if (!moe_alias_bufs.empty() && !model.no_alloc() && ub_fitted != this && n_tokens > 512) {
        ub_fitted = this;
        const uint32_t u = fit_ub(n_tokens);
        if (u >= 512 && u < n_tokens) {
            cparams.n_ubatch = u;
            sched_need_reserve = true;
            sched_reserve();
            return;
        }
    }

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                apply_moe_alias();
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf && !moe_alias_bufs.empty()) {   // the graph does not fit in the lent blocks: the largest that does (computed), or the alias off
                const uint32_t u = n_tokens > 512 ? fit_ub(n_tokens - 32) : 0;
                if (u >= 512) {
                    cparams.n_ubatch = u;
                    sched_need_reserve = true;
                    sched_reserve();
                    return;
                }
                if (moe_alias_shrink_fork(false, true, 0, 0)) {
                    sched_reserve();
                    return;
                }
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.no_alloc());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // The pinned host buffer of a pass (the attention masks and other inputs that live in host memory) grows with the pass tokens u and the cells k. Measure how, from the real graph of this
    // model, with three small measure-only reserves (nothing is allocated, so nothing can stall): (256, k1) and (512, k1) give c and the per-token bytes at k1; (256, k2) gives how they grow
    // with the cells. H = u * (a * k + b) + c then holds for any model (any mask type, any number of masks) and the pass cap divides the live pinnable budget by it.
    if (!g_host_cal) {
        static int said = 0;
        if (said++ < 3) { LLAMA_LOG_WARN("%s: host calibration check: moe_cache %d, no_alloc %d, memory %d\n", __func__, (int) cparams.moe_cache, (int) model.no_alloc(), (int) (memory != nullptr)); }
    }
    if (cparams.moe_cache && !model.no_alloc() && !g_host_cal && memory) {
        int cpu_i = -1;
        for (size_t bi = 0; bi < backend_ptrs.size(); ++bi) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[bi])) == GGML_BACKEND_DEVICE_TYPE_CPU) { cpu_i = (int) bi; break; }
        }
        if (cpu_i >= 0) {
            g_host_cal = true;   // measured once, when the cache is on and the memory exists
            auto measure = [&](uint32_t u, uint32_t k) -> double {
                llama_kv_reserve_override_fork(k);
                auto mc = memory->init_full();
                std::vector<size_t> sz(backend_ptrs.size(), 0);
                graph_reserve(u, 1, 1, mc.get(), true, sz.data());
                llama_kv_reserve_override_fork(0);
                return (double) sz[cpu_i];
            };
            // four points: H = u x (a x k + b) + d x k + c; d holds what grows with the cells alone (the k-pool indexer's pool index lists, GLM-5.3 and
            // Qwen3.8-Flash-Next), which three points would have folded into b
            const double k1 = 32768, k2 = 65536, u1 = 256, u2 = 512;
            const double h11 = measure((uint32_t) u1, (uint32_t) k1), h21 = measure((uint32_t) u2, (uint32_t) k1);
            const double h12 = measure((uint32_t) u1, (uint32_t) k2), h22 = measure((uint32_t) u2, (uint32_t) k2);
            const double s1 = (h21 - h11) / (u2 - u1);               // bytes per pass token at k1 cells
            const double s2 = (h22 - h12) / (u2 - u1);               // bytes per pass token at k2 cells
            g_host_a = std::max(0.0, (s2 - s1) / (k2 - k1));
            g_host_b = std::max(0.0, s1 - g_host_a * k1);
            const double r1 = h11 - u1 * s1, r2 = h12 - u1 * s2;     // what does not grow with the tokens, at k1 and k2
            g_host_d = std::max(0.0, (r2 - r1) / (k2 - k1));
            g_host_c = std::max(0.0, r1 - g_host_d * k1);
            g_host_ok = h21 > 0 && s1 > 0;
            if (g_host_ok) { llama_moe_cache_mask_bpc_fork(g_host_a); }   // the cards get copies of the same inputs: the pool's lend estimate grows with the cells by as much
            LLAMA_LOG_WARN("%s: host buffer of a pass: %.0f MiB at (%.0f tokens, %.0f cells), %.0f MiB at (%.0f, %.0f), %.0f MiB at (%.0f, %.0f), %.0f MiB at (%.0f, %.0f) -> %.3f bytes per token per cell + %.0f bytes per token + %.3f bytes per cell + %.0f KiB\n", __func__,
                    h11 / 1048576.0, u1, k1, h21 / 1048576.0, u2, k1, h12 / 1048576.0, u1, k2, h22 / 1048576.0, u2, k2, g_host_a, g_host_b, g_host_d, g_host_c / 1024.0);
        }
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.no_alloc());
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.no_alloc());
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    if (!moe_alias_bufs.empty() && !model.no_alloc()) {
        const llama_moe_alias_fit_fork fit = llama_moe_cache_alias_fit_fork(sched.get(), (int32_t) n_tokens);
        if (!fit.graph && n_tokens > 512) {   // the graph does not fit (the pool changed since the micro-batch was computed): the largest that does
            const uint32_t u = fit_ub(n_tokens - 32);
            if (u >= 512) {
                cparams.n_ubatch = u;
                sched_need_reserve = true;
                sched_reserve();
                return;
            }
        }
        if (moe_alias_shrink_fork(fit.graph, fit.stage, fit.need, fit.cap)) {
            sched_reserve();
            return;
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        thread_tune_feed(ggml_time_us() - t_compute_start_us);
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        if (!fork_resident) {
            llama_dsplit_report_fork(n_queued_tokens, ggml_time_us() - t_compute_start_us);   // fork: --moe dense-split=auto learns which configuration is faster
            llama_moe_cache_prefill_report_fork((uint32_t) n_queued_tokens, ggml_time_us() - t_compute_start_us);   // fork: the pass planner learns the compute per token
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

bool llama_context::set_mtp_dsa_index_share(bool enabled) {
    if (model.arch != LLM_ARCH_GLM5_NEXT || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || memory == nullptr) {
        return false;
    }

    enabled = enabled && model.hparams.indexer_index_share_mtp;

    auto * mem = static_cast<llama_memory_hybrid_idx *>(memory.get());
    if (mem->get_mtp_dsa_index_share() != enabled) {
        mem->set_mtp_dsa_index_share(enabled);
        sched_need_reserve = true;
    }

    return enabled;
}

bool llama_context::set_mtp_dsa_selection(const int32_t * data, size_t size) {
    if (model.arch != LLM_ARCH_GLM5_NEXT || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || memory == nullptr) {
        return false;
    }

    auto * mem = static_cast<llama_memory_hybrid_idx *>(memory.get());
    if (!mem->get_mtp_dsa_index_share()) {
        return false;
    }

    mem->set_mtp_dsa_selection(data, size);
    return true;
}

const int32_t * llama_context::get_mtp_dsa_selection(size_t * size) {
    if (size != nullptr) {
        *size = 0;
    }
    if (mtp_dsa_sel_raw.empty() || mtp_dsa_sel_raw.size() != mtp_dsa_sel_mask.size() || mtp_dsa_sel_width == 0) {
        return nullptr;
    }
    GGML_ASSERT(memory != nullptr && model.arch == LLM_ARCH_GLM5_NEXT);
    GGML_ASSERT(mtp_dsa_sel_raw.size() == mtp_dsa_sel_width*mtp_dsa_sel_seq.size());

    auto * mem = static_cast<llama_memory_hybrid_idx *>(memory.get());
    auto * mem_idx = mem->get_mem_idx();
    GGML_ASSERT(mem_idx != nullptr);

    mtp_dsa_sel.resize(mtp_dsa_sel_raw.size());
    for (size_t row = 0; row < mtp_dsa_sel_seq.size(); ++row) {
        const llama_seq_id seq_id = mtp_dsa_sel_seq[row];
        GGML_ASSERT(seq_id >= 0);
        const auto & cells = mem_idx->get_cells(seq_id);

        for (size_t j = 0; j < mtp_dsa_sel_width; ++j) {
            const size_t i = row*mtp_dsa_sel_width + j;
            const int32_t cell = mtp_dsa_sel_raw[i];
            if (mtp_dsa_sel_mask[i] != 0.0f) {
                mtp_dsa_sel[i] = -1;
                continue;
            }
            GGML_ASSERT(cell >= 0 && (uint32_t) cell < cells.size());
            GGML_ASSERT(!cells.is_empty((uint32_t) cell) && cells.seq_has((uint32_t) cell, seq_id));
            mtp_dsa_sel[i] = cells.pos_get((uint32_t) cell);
        }
    }
    if (size != nullptr) {
        *size = mtp_dsa_sel.size();
    }
    return mtp_dsa_sel.data();
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

// Decode thread autotune. The default count is a formula (cores minus one per GPU); the best count depends on the CPU, its
// memory and how much of the model runs on it. Every `hold` tokens: candidates {base, base-2, base+2} in interleaved
// 32-token slices (3 rounds, 8 warm-up tokens each). Decode times are noisy and correlated, so the decision uses slice means
// paired by round: a candidate must beat the base in EVERY round (by half the base's own spread, 0.5-3%), the cycle is void when the base's own slices differ by
// more than 25%, and a candidate must win two cycles in a row before it replaces the base. The interval doubles (4096 ..
// 65536 tokens) while the base is confirmed. LLAMA_THREAD_AUTOTUNE=0 turns it off.
// the tuned thread counts of this model on this machine are kept in the state file as "<default> <tuned>", valid while the default
// is unchanged: a short session (one reply, no time for a cycle) starts from what an earlier one found
// how much of the expert work runs on the CPU, in quarters of the layers (4 = all of it, the --cpu-moe case)
static int threads_cpu_quarter(const llama_model & model) {
    int n = 0, host = 0;
    for (const auto & l : model.layers) {
        if (l.ffn_up_exps && l.ffn_up_exps->buffer) { n++; host += ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) ? 1 : 0; }
    }
    return n > 0 ? (4 * host + n / 2) / n : 4;
}

// key of a tuned count: the card set and links it was tuned on (place.g<gpus>.<id>.v3) and the CPU share of the experts
static std::string threads_key(const llama_model & model, const char * key) {
    const std::string & pl = moe_state_place();
    const int q = threads_cpu_quarter(model);
    return (pl.empty() ? std::string() : pl + ".") + (q >= 4 ? std::string(key) : std::string(key) + "_cpu" + std::to_string(q));
}

static int threads_saved(const llama_model & model, const char * key, int dflt) {
    // ini-fix (2026-10-22): no saved count: every start searches from the same default (a saved count, re-explored every 8th session, made a start depend on how
    // many sessions came before it)
    if (true) { (void) model; (void) key; (void) dflt; return 0; }
    std::string v;
    int d = 0, t = 0;
    return moe_state_get(moe_state_section(model), key, v) && sscanf(v.c_str(), "%d %d", &d, &t) == 2 && d == dflt && t >= 2 ? t : 0;
}

// how many sessions used this saved value: it is explored again every 8th session, the others start from it and measure nothing
static int threads_runs_bump(const llama_model & model, const char * key) {
    if (true) { (void) model; (void) key; return 0; }   // ini-fix (2026-10-22): see threads_saved
    std::string v;
    int n = 0;
    if (moe_state_get(moe_state_section(model), key, v)) { n = atoi(v.c_str()); }
    n++;
    moe_state_set(moe_state_section(model), { { key, std::to_string(n) } });
    return n;
}

static void threads_save(const llama_model & model, const char * key, int dflt, int tuned) {
    if (true) { (void) model; (void) key; (void) dflt; (void) tuned; return; }   // ini-fix (2026-10-22): see threads_saved
    moe_state_set(moe_state_section(model), { { key, std::to_string(dflt) + " " + std::to_string(tuned) } });
}

void llama_context::set_thread_autotune(bool on) {
    const char * e = moe_opt("thread-tune");
    if (e && atoi(e) == 0) { on = false; }
    // no expert work on the CPU (everything on the GPUs): the decode threads only wait, so the default count costs nothing and there is nothing to tune or store.
    // (A count of 2 tuned in that state was later reused with experts on the CPU: 0.7x of stock decode on a small card.)
    if (on && threads_cpu_quarter(model) == 0 && !cparams.moe_cache) { on = false; }   // the expert cache computes its misses on the CPU
    thr.on    = on;
    thr.base0 = (int) cparams.n_threads;
    thr.base  = thr.base0;
    thr.hold  = 8;   // the first tokens run on the default (the counter runs across requests)
    // the thread pool is created for the default count after this call: a saved count above it cannot run, so only lower ones apply
    if (on) {
        if (const int s = threads_saved(model, threads_key(model, "threads").c_str(), thr.base0)) {
            thr.base = std::min(s, thr.base0);
            cparams.n_threads = (uint32_t) thr.base;
        }
    }
}

// the startup probe knows how many reader threads saturate the RAM read rate; decode does arithmetic on every byte it reads (dequantizing), so it needs
// a few times that many (PC1: saturated by 2, best decode 6) but not the cores-minus-GPUs formula's 60 on a 64-core host, where every thread adds a
// barrier to each small expert matmul. A tuned count saved by an earlier session stands.
void llama_context::thread_cap_from_probe(int32_t n_sat) {
    if (!thr.on || n_sat <= 0 || thr.base != thr.base0) {
        return;
    }
    const int cap = std::max(4, 3*(int) n_sat);
    if (cap < thr.base) {
        LLAMA_LOG_INFO("%s: decode threads %d -> %d (the RAM read rate is saturated by %d threads; the tuner starts from here)\n", __func__, thr.base, cap, (int) n_sat);
        thr.base = cap;
        cparams.n_threads = (uint32_t) cap;
    }
}

void llama_context::set_batch_thread_autotune(bool on, int32_t n_max) {
    const char * e = moe_opt("thread-tune");
    if (e && atoi(e) == 0) { on = false; }
    // off unless asked for (LLAMA_BATCH_THREAD_AUTOTUNE=1): a cycle needs 15 full batches after a two-batch hold, a prompt of up to ~16k tokens ends before it can decide
    // (and so before anything is saved), so every start explored five thread counts, SMT siblings included, and paid ~5% of the prompt (CPU only, OLMoE, 4k tokens)
    const char * b = moe_opt("batch-tune");
    if (!(b && atoi(b) != 0)) { on = false; }
    thrb.on    = on;
    thrb.n_max = n_max;
    thrb.base0 = (int) cparams.n_threads_batch;
    thrb.base  = thrb.base0;
    if (on) {
        if (const int s = threads_saved(model, threads_key(model, "threads_batch").c_str(), thrb.base0)) {
            thrb.base = std::min(s, std::max(n_max, thrb.base0));
            cparams.n_threads_batch = (uint32_t) thrb.base;
            // a count measured by an earlier session (adopted, or the default confirmed best) stands: exploring five thread counts costs a
            // few percent of every short prompt, so it is only repeated every 8th session
            if (threads_runs_bump(model, "threads_batch_runs") % 8 != 0) {
                thrb.hold_len = 1 << 30;
                thrb.hold = thrb.hold_len;
                return;
            }
        }
    }
    thrb.hold = 2;  // the first full batches run on the default
}

void llama_context::thread_tune_feed_batch(int64_t dt_us, int64_t n_tokens) {
    if (!thrb.on || cparams.n_threads_batch < 2 || n_tokens < 64) {
        return;
    }
    thrb.n_full = std::max(thrb.n_full, n_tokens);
    if (n_tokens < thrb.n_full*9/10) {
        return; // a short tail batch: its per-token time is not comparable
    }
    auto set = [&](int n) { cparams.n_threads_batch = (uint32_t) n; };
    if (thrb.cand.empty()) {
        if (thrb.hold > 0 && --thrb.hold > 0) { return; }
        const int hw   = std::max(2, (int) std::thread::hardware_concurrency());
        const int base = thrb.base;
        // an attached thread pool has a fixed size: never ask for more threads than it has (the caller says how many)
        const int cap = threadpool_batch ? std::max(base, thrb.n_max) : hw;
        std::vector<int> c = { base };
        for (int n : { base - 2, base + 2, base + 4, hw }) {
            n = std::min(n, cap);
            if (n >= 2 && n != base && std::find(c.begin(), c.end(), n) == c.end()) { c.push_back(n); }
        }
        if (c.size() < 2) { thrb.hold = thrb.hold_len; return; }
        thrb.cand = c; thrb.slot = 0;
        thrb.slice.assign(c.size(), {});
        set(c[0]);
        return;
    }
    // this batch ran on cand[slot % n]
    const size_t k = (size_t) thrb.slot % thrb.cand.size();
    thrb.slice[k].push_back((double) dt_us / (double) n_tokens);
    if (++thrb.slot < (int) thrb.cand.size()*3) {
        set(thrb.cand[(size_t) thrb.slot % thrb.cand.size()]);
        return;
    }
    // decide on paired rounds: a candidate must beat the base in EVERY round; two winning cycles in a row to replace it
    auto mean = [&](size_t i) { double m = 0; for (double x : thrb.slice[i]) { m += x; } return m/std::max<size_t>(1, thrb.slice[i].size()); };
    const auto & b0 = thrb.slice[0];
    const double bmin = *std::min_element(b0.begin(), b0.end()), bmax = *std::max_element(b0.begin(), b0.end());
    const bool noisy = bmax > 1.25*bmin;
    // same rule as the decode tuner: more threads must win in every round by the noise-scaled bar, fewer threads must not lose by more than 1%
    int best = -1, fewest = -1;
    const double win = std::min(0.03, std::max(0.005, 0.5*(bmax/bmin - 1))); // noise-scaled bar for more threads, 0.5% to 3%
    if (!noisy) {
        for (size_t i = 1; i < thrb.cand.size(); ++i) {
            const bool fewer = thrb.cand[i] < thrb.cand[0];
            bool all = thrb.slice[i].size() == b0.size();
            for (size_t r = 0; all && r < b0.size(); ++r) { all = thrb.slice[i][r] < (fewer ? 1.01 : 1.0 - win)*b0[r]; }
            if (!all) { continue; }
            if (fewer) { if (fewest < 0 || thrb.cand[i] < thrb.cand[(size_t) fewest]) { fewest = (int) i; } }
            else if (best < 0 || mean(i) < mean((size_t) best)) { best = (int) i; }
        }
    }
    if (best < 0) { best = fewest; }
    const int cand_best = best >= 0 ? thrb.cand[(size_t) best] : -1;
    const bool adopt = cand_best >= 0 && cand_best == thrb.pending;
    std::string msg;
    for (size_t i = 1; i < thrb.cand.size(); ++i) { msg += " | " + std::to_string(thrb.cand[i]) + ": " + std::to_string(1e6/mean(i)).substr(0, 6); }
    LLAMA_LOG_INFO("%s: batch thread autotune: base %d %.1f tokens/s%s%s -> %d%s\n", __func__, thrb.cand[0], 1e6/mean(0), msg.c_str(),
            noisy ? " (noisy, void)" : "", adopt ? cand_best : thrb.cand[0], (cand_best >= 0 && !adopt) ? " (leads, confirming)" : "");
    if (adopt) { thrb.base = cand_best; thrb.pending = -1; thrb.hold_len = 8; thrb.hold = thrb.hold_len; threads_save(model, threads_key(model, "threads_batch").c_str(), thrb.base0, cand_best); }
    else if (cand_best >= 0) { thrb.pending = cand_best; thrb.hold = 1; }   // confirm at the next full batch
    else {
        thrb.pending = -1; thrb.hold_len = std::min(thrb.hold_len*2, 64); thrb.hold = thrb.hold_len;
        // a clean cycle in which nothing beat the base: remember the base, the next sessions do not repeat the measurement
        if (!noisy) { threads_save(model, threads_key(model, "threads_batch").c_str(), thrb.base0, thrb.base); }
    }
    set(thrb.base);
    thrb.cand.clear();
}

void llama_context::thread_tune_feed(int64_t dt_us) {
    if (!thr.on || cparams.n_threads < 2) {
        return;
    }
    // First tokens: find the decode thread count by ALTERNATING between the best count so far (b) and one lower candidate (c) on consecutive tokens (user 2026-10-26).
    // The search starts on every hardware thread (common_init_result): too many threads costs little (barriers, SMT siblings), too few costs a lot, and on a multi-die
    // CPU only all of them reach every die's memory path. One token is a noisy sample (expert cache hits differ per token), so c is never judged on a run of its own:
    //   tokens   b c b c b   (2 pairs; 3 pairs when the step is small, 4*(b-c) <= b, where the gap is near the noise)
    // and every c token is compared with the mean of the two b tokens around it, so the expert cache warming up (later tokens are faster by themselves) cancels.
    // c is adopted only if it was faster than its neighbours in EVERY pair (a tie or a mixed result keeps the higher count; a mixed result gets one more pair first,
    // then c needs more wins than losses). Candidates: b/2 first (16 -> 8 -> 4); a lost candidate becomes the lower bound L and the next one is the midpoint of
    // b and L (8 vs 4 lost -> 6, then 5): about 4-6 tokens per candidate, the count settles in about 20 tokens. The cost in tokens does not matter (user), the
    // quality of the decision does. Then the +-2 tuner below stays off (settled, with the drift watch). File-local state: one decoding context per process.
    static struct { int st = 0, b = 0, c = 0, lo = 0, pairs = 0, k = 0, warm = 0, drift = 0, wn = 0; std::vector<double> sb, sc; double best = 0, wsum = 0; } d;
    if (d.st != 3) {   // 0 start, 1 sampling the alternation; 3 settled (below)
        auto plan = [&]() {   // the next candidate below b, or finish
            const int c = d.lo == 0 ? d.b/2 : (d.b + d.lo)/2;
            if (c < 1 || c >= d.b || c <= d.lo) {
                LLAMA_LOG_INFO("%s: thread search: %d -> %d threads (alternating b/c tokens; the next lower candidate was not clearly faster)\n", __func__, thr.base, d.b);
                llama_moe_decision_fork("threads_decode", std::to_string(d.b).c_str(), "from every hardware thread down, alternating with a lower count token by token; a lower count must win every pair (too few costs more than too many)");
                if (d.b != thr.base) { thr.base = d.b; threads_save(model, "threads", thr.base0, d.b); }
                cparams.n_threads = (uint32_t) thr.base;
                d.st = 3; d.drift = 0; d.wn = 0; d.wsum = 0; d.best = 0;   // settled: the count stays until the token time drifts (below)
                return;
            }
            d.c = c; d.pairs = 4*(d.b - c) <= d.b ? 3 : 2; d.k = 0; d.sb.clear(); d.sc.clear();
            cparams.n_threads = (uint32_t) d.b;   // the sequence starts on b
            char v[48]; snprintf(v, sizeof v, "%d vs %d (testing)", d.b, c);
            llama_moe_decision_fork("threads_decode", v, "searching: alternating the best count so far with one lower candidate, token by token");
        };
        if (d.st == 0) { d.b = thr.base; d.lo = 0; d.st = 1; d.warm = 2; plan(); if (d.st != 1) { return; } }   // the first tokens warm the expert cache: not sampled
        if (d.warm > 0) { --d.warm; return; }
        (d.k % 2 == 0 ? d.sb : d.sc).push_back((double) dt_us);   // token k ran on b (even k) or on c (odd k)
        if (d.k < 2*d.pairs) { ++d.k; cparams.n_threads = (uint32_t) (d.k % 2 == 0 ? d.b : d.c); return; }
        // the sequence b c b c (b) is complete: every c against the mean of its two neighbouring b tokens
        int wins = 0, losses = 0;
        for (int i = 0; i < d.pairs; ++i) { (d.sc[i] < 0.5*(d.sb[i] + d.sb[i + 1]) ? wins : losses)++; }
        if (wins > 0 && losses > 0 && d.pairs < 4) { ++d.pairs; ++d.k; cparams.n_threads = (uint32_t) d.c; return; }   // mixed: one more pair, the last b token is its first neighbour
        double mb = 0, mc = 0;
        for (double x : d.sb) { mb += x; } for (double x : d.sc) { mc += x; }
        LLAMA_LOG_INFO("%s: thread search: %d threads %.2f ms/token vs %d threads %.2f ms/token, %d of %d pairs won by the lower count -> %s\n", __func__,
                d.b, mb/d.sb.size()/1000.0, d.c, mc/d.sc.size()/1000.0, wins, wins + losses, wins > losses ? "lower" : "keep");
        if (wins > losses) { d.b = d.c; } else { d.lo = d.c; }
        plan();
        return;
    }
    if (d.st == 3) {   // settled: stay, and watch. Windows of 64 tokens; a mean 10% above the best seen, twice in a row, means the box changed (other load, a new prompt mix): search again from the default
        d.wsum += (double) dt_us;
        if (++d.wn < 64) { return; }
        const double m = d.wsum/64.0; d.wsum = 0; d.wn = 0;
        if (d.best <= 0 || m < d.best) { d.best = d.best <= 0 ? m : 0.7*d.best + 0.3*m; d.drift = 0; return; }
        d.drift = m > 1.10*d.best ? d.drift + 1 : 0;
        if (d.drift >= 2) {
            LLAMA_LOG_INFO("%s: thread search again: %.2f ms/token against %.2f settled\n", __func__, m/1000.0, d.best/1000.0);
            thr.base = thr.base0;
            d.st = 0; d.b = d.c = d.lo = 0; d.best = 0;
        }
        return;
    }
    auto set = [&](int n) { cparams.n_threads = (uint32_t) n; };
    if (thr.cand.empty()) {
        if (thr.hold > 0) {
            if (--thr.hold > 0) { return; }
        }
        const int hw   = std::max(2, (int) std::thread::hardware_concurrency());
        const int base = thr.base;
        std::vector<int> c = { base };
        // an attached thread pool (llama-cli) has a fixed size: never ask for more threads than it has
        const int cap = (threadpool || threadpool_batch) ? base : hw;
        // climb by 2: the start is already bounded by the probe (thread_cap_from_probe), a win is followed at once by the next step (below)
        const int step = 2;
        const int lo = std::max(2, base - step), hi = std::min(cap, base + step), lo2 = std::max(2, base - 2*step);
        if (lo != base) { c.push_back(lo); }
        // a far lower count costs a slow slice per round: only when the nearer one looked close to the base
        if (thr.try_lo2 && lo2 != lo && lo2 != base) { c.push_back(lo2); }
        if (hi != base) { c.push_back(hi); }
        // far candidates (+4, +8, +16) only while the last cycle showed a trend (base+2 ahead in every round): a slow gradient shows at four to eight times the size,
        // but a far count costs a slice per round, so it is not tried blindly
        if (thr.try_far) {
            for (int d : { 4, 8, 16 }) {
                const int n = std::min(cap, base + d);
                if (n > base && std::find(c.begin(), c.end(), n) == c.end()) { c.push_back(n); }
            }
        }
        // the first cycle also tries base/2 and base/4: a start far above the useful count (container CPU quota, SMT siblings) costs every token until
        // the +-2 steps could walk down, and the first tokens are the ones a short request is made of
        if (thr.cycles == 0 && thr.pending < 0) {
            for (int d : { 2, 4 }) {
                const int n = std::max(2, base/d);
                if (n < base && std::find(c.begin(), c.end(), n) == c.end()) { c.push_back(n); }
            }
        }
        if (c.size() < 2) { thr.hold = thr.hold_len; return; }
        // the first cycle of a session is short (12-token slices, 3 skipped) so one-shot runs get a decision; later ones use 32 / 8
        thr.slice_len = thr.cycles == 0 ? 12 : 32; thr.warm_len = thr.cycles == 0 ? 3 : 8;
        thr.cand = c; thr.slot = 0; thr.tok = 0; thr.warm = thr.warm_len;
        thr.cur_sum.assign(1, 0.0);
        thr.slice.assign(c.size(), {});
        set(c[0]);
        return;
    }
    if (thr.warm > 0) { --thr.warm; return; }
    thr.cur_sum[0] += (double) dt_us;
    if (++thr.tok < thr.slice_len) { return; }
    const size_t k = (size_t) thr.slot % thr.cand.size();
    thr.slice[k].push_back(thr.cur_sum[0]/(double) thr.slice_len);
    thr.cur_sum[0] = 0.0; thr.tok = 0; thr.warm = thr.warm_len;
    if (++thr.slot < (int) thr.cand.size()*3) {
        set(thr.cand[(size_t) thr.slot % thr.cand.size()]);
        return;
    }
    // decide on paired slice means
    auto mean = [&](size_t i) { double m = 0; for (double x : thr.slice[i]) { m += x; } return m/std::max<size_t>(1, thr.slice[i].size()); };
    const auto & b0 = thr.slice[0];
    const double bmin = *std::min_element(b0.begin(), b0.end()), bmax = *std::max_element(b0.begin(), b0.end());
    const bool noisy = bmax > 1.25*bmin;
    // more threads must win in every round by a noise-scaled bar (0.5-3%); fewer threads only must not lose by more than 1% (a tie favors fewer: less power,
    // cores left for launches and uploads). The fastest winner with more threads beats the lowest count among the ties.
    int best = -1, fewest = -1;
    // the bar for more threads follows the measured noise: half the base's own round-to-round spread, between 0.5% and 3%
    const double win = std::min(0.03, std::max(0.005, 0.5*(bmax/bmin - 1)));
    if (!noisy) {
        for (size_t i = 1; i < thr.cand.size(); ++i) {
            const bool fewer = thr.cand[i] < thr.cand[0];
            // more threads must pay for the cores they take: at least 0.25% per added thread (+2: 0.5%, +8: 2%, +16: 4%), on top of the noise bar
            const double bar = fewer ? 0.0 : std::min(0.08, std::max(win, 0.0025*(thr.cand[i] - thr.cand[0])));
            bool all = thr.slice[i].size() == b0.size();
            for (size_t r = 0; all && r < b0.size(); ++r) { all = thr.slice[i][r] < (fewer ? 1.01 : 1.0 - bar)*b0[r]; }
            if (!all) { continue; }
            if (fewer) { if (fewest < 0 || thr.cand[i] < thr.cand[(size_t) fewest]) { fewest = (int) i; } }
            else if (best < 0 || mean(i) < mean((size_t) best)) { best = (int) i; }
        }
    }
    if (best < 0) { best = fewest; }
    thr.try_far = false;
    for (size_t i = 1; i < thr.cand.size(); ++i) {   // a trend: base+2 ahead of the base in every round, by any amount
        if (thr.cand[i] != thr.cand[0] + 2 || thr.slice[i].size() != b0.size()) { continue; }
        bool ahead = true;
        for (size_t r = 0; r < b0.size(); ++r) { ahead = ahead && thr.slice[i][r] < b0[r]; }
        thr.try_far = thr.try_far || ahead;
    }
    thr.try_lo2 = false;
    for (size_t i = 1; i < thr.cand.size(); ++i) {
        if (thr.cand[i] >= thr.cand[0] || thr.slice[i].size() != b0.size()) { continue; }
        bool close = true;
        for (size_t r = 0; r < b0.size(); ++r) { close = close && thr.slice[i][r] < 1.03*b0[r]; }
        thr.try_lo2 = thr.try_lo2 || close;
    }
    const int cand_best = best >= 0 ? thr.cand[(size_t) best] : -1;
    // two cycles in a row must win in the same direction (a win at +8 and then at +4 is still a gain): the more conservative of the two is adopted
    const bool coarse = cand_best >= 0 && std::abs(cand_best - thr.cand[0]) > 4 && mean((size_t) best) < 0.9*mean(0);   // a 10% win from a far count needs no confirmation
    const bool adopt = coarse || cand_best >= 0 && thr.pending >= 0 && ((cand_best > thr.cand[0]) == (thr.pending > thr.cand[0]));
    const int adopted = !adopt ? thr.cand[0] : coarse ? cand_best : (cand_best > thr.cand[0] ? std::min(cand_best, thr.pending) : std::max(cand_best, thr.pending));
    std::string msg;
    for (size_t i = 1; i < thr.cand.size(); ++i) { msg += " | " + std::to_string(thr.cand[i]) + ": " + std::to_string(mean(i)/1000.0).substr(0, 5); }
    LLAMA_LOG_INFO("%s: thread autotune: base %d %.2f ms/token%s%s -> %d%s\n", __func__, thr.cand[0], mean(0)/1000.0, msg.c_str(),
            noisy ? " (noisy, void)" : "", adopted, (cand_best >= 0 && !adopt) ? " (leads, confirming)" : "");
    // an adopted step keeps climbing: the next cycle follows after 128 tokens; a cycle without a win is the plateau, then the rest grows as before
    if (adopt) { thr.base = adopted; thr.hold_len = 4096; thr.pending = -1; thr.hold = 128; threads_save(model, threads_key(model, "threads").c_str(), thr.base0, adopted); }
    else if (cand_best >= 0) { thr.pending = cand_best; thr.hold = 128; }   // second cycle soon
    else { thr.pending = -1; thr.hold_len = noisy ? 4096 : std::min(thr.hold_len*2, 65536); thr.hold = noisy ? 1024 : thr.hold_len; }
    set(thr.base);
    thr.cycles++;
    thr.cand.clear();
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_embeddings_layer_inp_last(uint32_t n) {
    cparams.embeddings_layer_inp_last = n;
    sched_need_reserve = true;   // the graph's outputs change (llm_graph_result::set_outputs)
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

extern void llama_dsplit_veto_fork(int n_tokens);   // llama-dsplit-fork.cpp (declared here, no shared header)

// fork (llama-dsplit-fork.cpp): the balance step of the pass just run, and the helpers' experts dropped from a split pass that does not fit; declared here so
// that no shared header changes
bool llama_dsplit_prev_split_fork();
void llama_dsplit_pass_done_fork();
bool llama_dsplit_drop_experts_fork(int64_t n_tokens);

// fork: --moe idx-check=1 (debug, slow: the card is drained and every index input read back before each graph). Before a graph runs, in every context (the
// target and a draft): each row index the host set for it (SET_ROWS / GET_ROWS fed by an input tensor: the KV write rows, k_idxs / v_idxs, the DSV4 compressor's
// state and row indices, token ids) must lie inside the tensor it addresses, and each view must still point where its source lives (a graph built before its
// source moved reads and writes the old place). An index outside its tensor reads or writes the neighbour of that record in the pool: the DSpark draft's KV
// sits between the target's lend and the target's growing rows (2026-10-27, 'illegal memory access' at spec depth 2 and 3 that launch blocking hid). The
// first bad index or view is logged with its node and the run aborts before the kernel runs; a line per context for the first graphs and every 256th says
// the check runs.
static void fork_idx_check(ggml_backend_sched_t sched, ggml_cgraph * gf, const char * who, int64_t n_graph) {
    ggml_backend_sched_synchronize(sched);   // inputs set asynchronously are on the card before they are read back
    static std::vector<uint8_t> buf;
    int64_t n_idx = 0, n_view = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->view_src && t->data && t->view_src->data) {
            n_view++;
            if ((const char *) t->data != (const char *) t->view_src->data + t->view_offs) {
                LLAMA_LOG_ERROR("idx-check: %s graph %lld: node %d '%s' (%s) is a view of '%s' at %p, but its source now lives at %p + %zu = %p: the graph was built before the source moved\n",
                        who, (long long) n_graph, i, t->name, ggml_op_name(t->op), t->view_src->name, t->data, t->view_src->data, t->view_offs,
                        (const void *) ((const char *) t->view_src->data + t->view_offs));
                GGML_ABORT("idx-check: stale view");
            }
        }
        const ggml_tensor * idx = nullptr;
        const ggml_tensor * tab = nullptr;
        if (t->op == GGML_OP_SET_ROWS) { idx = t->src[1]; tab = t->src[2]; }   // set_rows: src[0] the rows, src[1] the indices, src[2] the destination
        if (t->op == GGML_OP_GET_ROWS) { idx = t->src[1]; tab = t->src[0]; }
        if (!idx || !tab || !(idx->flags & GGML_TENSOR_FLAG_INPUT) || !idx->buffer || !idx->data || !ggml_is_contiguous(idx) ||
                (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64)) {
            continue;
        }
        const int64_t n = ggml_nelements(idx);
        if (n == 0) {
            continue;
        }
        buf.resize(ggml_nbytes(idx));
        ggml_backend_tensor_get(idx, buf.data(), 0, buf.size());
        int64_t mn = INT64_MAX, mx = INT64_MIN;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t v = idx->type == GGML_TYPE_I32 ? (int64_t) ((const int32_t *) buf.data())[k] : ((const int64_t *) buf.data())[k];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        n_idx++;
        if (mn < 0 || mx >= tab->ne[1]) {
            LLAMA_LOG_ERROR("idx-check: %s graph %lld: node %d '%s' (%s): index input '%s' holds [%lld, %lld] (%lld values), but '%s' has %lld rows (data %p, row stride %zu B, %s): "
                    "the out-of-range rows lie in its neighbour's memory\n",
                    who, (long long) n_graph, i, t->name, ggml_op_name(t->op), idx->name, (long long) mn, (long long) mx, (long long) n, tab->name, (long long) tab->ne[1],
                    tab->data, tab->nb[1], tab->buffer ? ggml_backend_buffer_name(tab->buffer) : "no buffer");
            GGML_ABORT("idx-check: row index outside its tensor");
        }
    }
    if (n_graph < 4 || n_graph % 256 == 0) {
        LLAMA_LOG_WARN("idx-check: %s graph %lld: %lld index inputs inside their tensors, %lld views at their sources\n", who, (long long) n_graph, (long long) n_idx, (long long) n_view);
    }
}

// fork: --moe dynhot keys (user 2026-10-28: the key of a request's hot map is the activations at the prompt's last token, "literally on prefill end", before
// the logits: every token before it is mixed in, nothing of the template; "pool all last token layers activations"). The pass that ends a prompt copies its
// last token's router input (the source of ffn_gate_inp) of every MoE layer and its final hidden state (result_norm, what the logits are made from) into
// small outputs (dynkey-0.., dynkey-final); after the pass they are fetched, and the next call hands them to the MoE cache (llama_moe_topic_act_fork).
// Other graphs stay as they were: an eval callback would turn the scheduler's prefetch off.
bool llama_moe_topic_key_wanted_fork(void);
void llama_moe_topic_act_fork(const float * const * rows, int n_rows, int64_t n_embd);
namespace {
struct dynkey_state {
    std::vector<std::vector<float>> rows;   // the MoE layers' router inputs in layer order, then the final hidden state (empty: not in the graph)
    bool pending = false;                   // fetched after a prompt's last pass, not handed over yet
} g_dynkey;

// t's last row (a 2-D float tensor) copied out as an output named name
bool dynkey_out(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * t, const char * name) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->ne[1] < 1 || t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != sizeof(float)) { return false; }
    ggml_tensor * out = ggml_cont(ctx, ggml_view_1d(ctx, t, t->ne[0], (size_t) (t->ne[1] - 1) * t->nb[1]));
    ggml_set_name(out, name);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return true;
}

// the copies of a prompt's last pass: the router input of every MoE layer found in the graph, and the final hidden state
void dynkey_add(ggml_context * ctx, ggml_cgraph * gf, int n_layer) {
    std::vector<std::pair<int, ggml_tensor *>> rin;   // per MoE layer (once each): its router's input
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        int il = -1, end = 0;
        if (t->op != GGML_OP_MUL_MAT || t->src[0] == nullptr || sscanf(t->src[0]->name, "blk.%d.ffn_gate_inp.weight%n", &il, &end) != 1 || end == 0 || il < 0 || il >= n_layer) {
            continue;
        }
        bool seen = false;
        for (const auto & r : rin) { seen = seen || r.first == il; }
        if (!seen) { rin.emplace_back(il, t->src[1]); }
    }
    std::sort(rin.begin(), rin.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    int n = 0;
    for (const auto & r : rin) {
        char name[32];
        snprintf(name, sizeof(name), "dynkey-%d", n);
        n += dynkey_out(ctx, gf, r.second, name);
    }
    dynkey_out(ctx, gf, ggml_graph_get_tensor(gf, "result_norm"), "dynkey-final");
}

// after the pass: the copies into host rows, queued on their backends (the next call synchronizes before it hands them over)
void dynkey_fetch(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    std::vector<ggml_tensor *> rows;
    ggml_tensor * fin = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (strncmp(t->name, "dynkey-", 7) != 0) { continue; }
        if (strcmp(t->name + 7, "final") == 0) { fin = t; continue; }
        const int k = atoi(t->name + 7);
        if (k >= (int) rows.size()) { rows.resize(k + 1, nullptr); }
        rows[k] = t;
    }
    rows.push_back(fin);
    g_dynkey.rows.resize(rows.size());
    bool any = false;
    for (size_t k = 0; k < rows.size(); ++k) {
        g_dynkey.rows[k].clear();
        if (rows[k] == nullptr) { continue; }
        g_dynkey.rows[k].resize((size_t) rows[k]->ne[0]);
        ggml_backend_tensor_get_async(ggml_backend_sched_get_tensor_backend(sched, rows[k]), rows[k], g_dynkey.rows[k].data(), 0, ggml_nbytes(rows[k]));
        any = true;
    }
    g_dynkey.pending = any;
}
}  // namespace

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    llama_kv_cache_set_mask_threads_fork(ubatch.n_tokens > 1 ? (int) cparams.n_threads_batch : (int) cparams.n_threads);   // the mask fill follows the compute threads
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // fork: --moe kv-tail: the KV zone of each card grows (cells past half of it) and makes room for the lend a graph of this size needed before, BEFORE the graph
    // is built: a graph never points at rows that move (a move changes the reuse epoch below)
    if (cparams.moe_cache) {
        llama_moe_cache_kv_tail_prepare_fork(sched.get(), (int32_t) ubatch.n_tokens);
        // a card whose pool cannot hold the KV zone at all (no gap between the zero blocks takes the fixed part and a cell): the graph would point at records that were never
        // attached (the allocator then aborted: 'cache_kv_record already has a buffer', OLMoE with 0.47 GiB VRAM left). The request fails with the reason instead.
        extern const char * llama_moe_cache_kv_tail_unplaced_fork(int32_t n_tokens);
        if (const char * why = llama_moe_cache_kv_tail_unplaced_fork((int32_t) ubatch.n_tokens)) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // fork: the pass before this one was a split pass: its card times are complete now (waited for), one balance step for it (llama-dsplit-fork.cpp)
    if (!fork_resident && ubatch.n_tokens > 1 && llama_dsplit_prev_split_fork()) {
        ggml_backend_sched_synchronize(sched.get());
        llama_dsplit_pass_done_fork();
    }
    // fork: every prompt pass is timed for the split factors (llama-dsplit-fork.cpp, dsf): the pass decided last ends at this synchronize, one card or split
    if (!fork_resident && ubatch.n_tokens > 1 && llama_dsplit_pass_open_fork()) {
        ggml_backend_sched_synchronize(sched.get());
        llama_dsplit_pass_end_fork();
    }
    if (!fork_resident) { llama_dsplit_note_pass_fork(ubatch.n_tokens > 1 && ubatch.pos ? (int64_t) ubatch.pos[0] : -1); }
    // fork: --moe dense-split: the configuration of this pass (one card or split over the cards), chosen before the graph is built; a changed choice rebuilds
    const bool ds_changed = !fork_resident && llama_dsplit_decide_fork(model, ubatch.n_tokens);   // the per-pass split decision (and its log line) is part of what the exit skips
    // fork: the share controller of the expert split (llama-graph.cpp) moves the cards' expert counts from the measurement of an earlier pass; a changed count
    // needs a new graph
    const bool ctl_changed = ubatch.n_tokens >= 32 && llama_split_ctl_poll_fork((int64_t) model.hparams.n_expert, (int64_t) model.hparams.n_embd);
    const int64_t t_graph0 = ggml_time_us();
    bool graph_rebuilt = false;
    bool reuse = !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams) && llama_moe_cache_graph_reusable_fork() && !ds_changed && !ctl_changed;
    // fork (--moe dynhot): the pass that ends a prompt (its last token has the output) copies the key rows out; a reused graph without the copies is built again
    const bool want_key = cparams.moe_cache_started && !cparams.ctx_other && ubatch.output != nullptr && ubatch.output[ubatch.n_tokens - 1] &&
                          llama_moe_topic_key_wanted_fork();
    if (reuse && want_key && ggml_graph_get_tensor(res->get_gf(), "dynkey-final") == nullptr && ggml_graph_get_tensor(res->get_gf(), "dynkey-0") == nullptr) {
        reuse = false;
    }
    if (reuse) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        // fork: only the context whose graphs live in the pool (moe_alias_bufs) lends from it. A draft context (DSpark) has its own compute buffers and its KV
        // in its own arena; its call set the lend to its small graph and the card's CUDA scratch just behind it, both over the target's graph: the blocks under the
        // target's graph went back to the free list, the after-prompt refill uploaded experts into them, and the draft's scratch writes (its own streams) raced
        // the target's verify step: illegal memory access at depth 2 and 3 (DeepSeek 4.1 + Lucebox draft, tetris prompt, reasoning off, 2026-10-27)
        if (!moe_alias_bufs.empty() && llama_moe_cache_alias_fork(sched.get(), (int32_t) ubatch.n_tokens)) { reuse = false; }   // the KV zone moved behind this graph: build it again
        if (const char * why = moe_alias_bufs.empty() ? nullptr : llama_moe_cache_alias_err_fork()) {   // fork: the graph and the KV do not fit the pool: this pass is refused (the request fails, the server lives)
            LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
    }
    if (!reuse) {
        gf_res_prev_active = nullptr;
        res->reset();

        // fork: a split pass places its graph behind the dense split's copies, where the pool has less room than for a one-card pass. When that graph does not
        // fit (a request failed with "graph needs 2597 MiB, the external buffer holds 2366 MiB"), the passes of this size run on one card from now on and the graph
        // is built again, instead of the request failing.
        for (int attempt = 0; ; ++attempt) {
            ggml_backend_sched_reset(sched.get());
            set_eval_cb();

            //const auto t_start_us = ggml_time_us();

            gf = model.build_graph(gparams);
            llama_moe_cache_graph_built_fork();

            //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

            if (!gf) {
                LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (want_key) {
                dynkey_add(res->get_ctx(), gf, (int) model.hparams.n_layer());
            }

            moe_mirror_offsets();   // fork: --moe mirror-pool: a pass that splits behind the dense split's copies, any other graph over them

            static const bool dbg_rb = getenv("LLAMA_REBUILD_DEBUG") != nullptr;   // squeeze repro (Qwen3.6, 4.7 GiB): which build of the graph the inputs belong to
            if (dbg_rb) { LLAMA_LOG_WARN("%s: graph build attempt %d: %d tokens, %d nodes, %d splits so far\n", __func__, attempt, (int) ubatch.n_tokens, ggml_graph_n_nodes(gf), ggml_backend_sched_get_n_splits(sched.get())); }
            if (ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                // fork: --moe alias=1 lends the pool blocks this graph's tensors reach to it; when the graph reached into the KV zone (kv-tail), the zone moved behind
                // it and this graph's views of the KV point at the old place: it is built again (the next graph of this size gets its lend before it is built)
                const bool moved = !moe_alias_bufs.empty() && llama_moe_cache_alias_fork(sched.get(), (int32_t) ubatch.n_tokens);   // only the pool's own context (see above)
                if (const char * why = moe_alias_bufs.empty() ? nullptr : llama_moe_cache_alias_err_fork()) {   // the graph and the KV do not fit the pool: refuse this pass instead of aborting
                    LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
                if (dbg_rb) {
                    int n_use = 0, n_buf = 0;   // graph nodes reading a tensor named out_ids, and how many of those have a buffer
                    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                        for (int j = 0; j < GGML_MAX_SRC; ++j) {
                            const ggml_tensor * sj = ggml_graph_node(gf, i)->src[j];
                            if (sj && strncmp(sj->name, "out_ids", 7) == 0) { n_use++; n_buf += sj->buffer != nullptr; if (n_use <= 3) { LLAMA_LOG_WARN("   reader %s <- src %d '%s' %p buffer %p data %p flags 0x%x\n", ggml_graph_node(gf, i)->name, j, sj->name, (const void *) sj, (void *) sj->buffer, sj->data, (unsigned) sj->flags); } }
                        }
                    }
                    LLAMA_LOG_WARN("%s: attempt %d allocated, alias_fork %s; nodes reading out_ids %d (with a buffer %d)\n", __func__, attempt, moved ? "moved the zone: build again" : "no move", n_use, n_buf);
                }
                if (moved && attempt < 4) {
                    res->reset();
                    continue;
                }
                break;
            }
            // fork: a split pass with experts on the helpers that does not fit: the experts go back to the expert card (the dense split alone) and the graph is
            // built again; only a dense split alone that does not fit gives the split up for passes of this size
            if (attempt == 0 && !fork_resident && llama_dsplit_cur_fork() == 1 && llama_dsplit_drop_experts_fork(ubatch.n_tokens)) {
                (void) llama_dsplit_decide_fork(model, ubatch.n_tokens);
                res->reset();
                continue;
            }
            if (attempt <= 1 && !fork_resident && llama_dsplit_cur_fork() == 1) {
                LLAMA_LOG_WARN("%s: a split pass of %d tokens does not fit behind the dense-split copies: one card for passes of this size\n", __func__, (int) ubatch.n_tokens);
                llama_dsplit_veto_fork((int) ubatch.n_tokens);
                (void) llama_dsplit_decide_fork(model, ubatch.n_tokens);
                res->reset();
                continue;
            }
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
        graph_rebuilt = true;
    }
    if (graph_rebuilt && ubatch.n_tokens >= 32) {
        // what a rebuild costs a pass (the share controller pays it every time it moves a count): LLAMA_SPLIT_CTL_DEBUG=1 prints it
        static const bool dbg = getenv("LLAMA_SPLIT_CTL_DEBUG") != nullptr;
        if (dbg) {
            LLAMA_LOG_WARN("prefill split: graph of %u tokens built and placed in %.1f ms%s\n", ubatch.n_tokens, (ggml_time_us() - t_graph0) / 1e3,
                    ctl_changed ? " (the expert counts changed)" : ds_changed ? " (one card / split choice changed)" : "");
        }
    }
    moe_mirror_fill();   // fork: --moe mirror-pool: the dense split's copies, when the region lost them since they were last filled

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    // fork: --moe idx-check=1: every index input and view of this graph checked before it runs (fork_idx_check)
    static const bool idx_check = [] { const char * e = llama_moe_get_opt("idx-check"); return e && atoi(e) != 0; }();
    if (idx_check) {
        static std::map<const llama_context *, int64_t> n_graphs;   // per context, here and not a member (no header edit for a debug switch)
        fork_idx_check(sched.get(), res->get_gf(), cparams.ctx_other ? "draft" : "target", n_graphs[this]++);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }
    if (want_key) {
        dynkey_fetch(sched.get(), res->get_gf());
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we skip position continuity checks and output all tokens
    //       a decoder without memory (see decode()) outputs the marked tokens, unless it extracts embeddings
    if (!balloc->init(batch_inp, model.vocab, memory || cparams.embeddings)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_all);
        return -2;
    };

    const auto & out_ids = balloc->get_out_ids();
    for (uint32_t i = 0; i < n_outputs_all; ++i) {
        output_ids[out_ids[i]] = i;
    }

    n_outputs = n_outputs_all;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_outputs*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_outputs*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_outputs*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_outputs*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_outputs*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

// fork (user 2026-10-25: "we don't want an out-of-memory crash, we want try/catch recovery where it happens, ideally prevented"): a CUDA out-of-memory
// inside a pass comes back from the backend as GGML_STATUS_ALLOC_FAILED (ggml_cuda_oom, ggml-cuda.cu) instead of killing the process. decode then takes
// the whole call back (every position the call added: its later passes cannot be redone on top of the earlier ones' outputs), lowers the pass ceiling
// below and returns 1, and llama_decode / llama_process run the same batch again at once (g_oom_retry): the caller never sees it. The ceiling is in
// tokens x cells, since the pass's attention, mask and their temporaries grow with both: after a failure at T tokens and C cells a pass may have
// T x C x s, s the share that fits (the pool's bytes and the card's free ones, ggml_cuda_oom_fit; 0.9 of it at most, 0.25 at least, 0.5 where it is not known): fewer
// tokens at that context and more at a shorter one (a later short prompt keeps its full passes). It stays for the life of the process and is lowered again
// at every failure; a pass of 256 tokens that still does not fit is an error, not a loop.
static std::atomic<uint64_t> g_oom_tc    { 0 };   // the ceiling, tokens x cells, 0 = none
static std::atomic<int>      g_oom_retry { 0 };   // set by decode: the call was taken back, run it again
void llama_moe_cache_vram_oom_fork(int cuda_device, size_t bytes);   // llama-moecache-fork.cpp: the failed request counts as VRAM need outside the pool (declared here, no header edit)
static int fork_cuda_last_oom(int * device, size_t * bytes, double * fit) {   // ggml-cuda.cu: the number of out-of-memory errors it caught, the last one's device, size and the share of the pass that fits
    static int (*fn)(int *, size_t *, double *) = [] {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        return reg ? (int (*)(int *, size_t *, double *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_last_oom_fork") : nullptr;
    }();
    return fn ? fn(device, bytes, fit) : 0;
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    moe_decoded = true;
    // fork (--moe split-pass, default 1; 2 = equal passes, 0 = the old pass, A/B; dense split forced; user 2026-10-24): a prompt pass of the dense split was cut by the ONE-card limit, so
    // with two cards each computed about half of what its buffers hold, for every upload of the experts. Here a call is run pass by pass, each pass sized
    // just before it runs: the expert card (fastest link) takes the whole one-card limit, every other card what it finishes in the expert card's time
    // (the zero-wait rule, llama-dsplit-fork.cpp dsplit_zw; llama_dsplit_pass_size_fork), the last pass what is left. The step of the pass before runs
    // first, so the other cards' tokens follow the measured times pass by pass.
    // (The KV cache cuts every pass of a call when the call starts, so per-pass sizes need one call per pass.) Only for one sequence whose only output
    // is its last token (a prompt): the earlier passes' outputs would be lost, and a failed later pass cannot take back the earlier ones (KDA state).
    if (!g_split_sub && cparams.moe_cache && memory && !cparams.embeddings && batch_inp.tokens.size() > cparams.n_ubatch) {
        // default OFF since 2026-10-25: with the equal dense-time shares the share-cut split beat the loop (Qwen3.8 60k 1991 vs 1843, GLM 3.0 708-755 vs 696
        // prompt t/s), and the loop's big lends left GLM's pools short: a 10336-token call at 62.5k cells found no KV slot and failed (bug log). --moe split-pass=1: on
        static const int  split_pass = [] { const char * e = llama_moe_get_opt("split-pass"); return e ? atoi(e) : 0; }();
        static const bool forced     = [] { const char * e = llama_moe_get_opt("dense-split"); return e && strcmp(e, "1") == 0; }();
        const auto & tk = batch_inp.tokens;
        const int32_t n = (int32_t) tk.size();
        // llama_dsplit_max_pass_fork says whether passes beyond the micro-batch are allowed at all: it is also what the sliding-window ring was sized for
        // (no bigger pass for models with sliding-window layers: their ring holds n_swa + the micro-batch, a bigger pass finds no slot)
        bool ok = split_pass > 0 && forced && llama_dsplit_ready_fork(model) && tk[0].seq_ids.size() == 1 && llama_dsplit_max_pass_fork(&model, cparams.n_ubatch) > cparams.n_ubatch;
        for (int32_t i = 0; ok && i < n; ++i) {
            ok = tk[i].seq_ids == tk[0].seq_ids && (!tk[i].output || i == n - 1);
        }
        if (ok) {
            const uint32_t cap = cparams.n_ubatch;
            llama_dsplit_note_limit_fork(cap);
            llama_dsplit_note_card_max_fork(cap);
            int32_t off = 0, last_off = 0, k = 0;
            int32_t cnt_fit = 0;   // set when the pool refused a pass: the largest pass that fit since (0: none refused in this call)
            int oom_k = 0;         // passes a CUDA out-of-memory took back in this call (each runs again under the ceiling, g_oom_tc)
            bool ran = false;      // a pass ran since the last balance step (a refused pass is no measurement)
            g_split_sub = true;
            while (off < n) {
                if (ran && llama_dsplit_prev_split_fork()) {   // the pass just run: its balance step now, before the next pass is sized
                    ggml_backend_sched_synchronize(sched.get());
                    llama_dsplit_pass_done_fork();
                }
                ran = false;
                // the biggest pass now: the expert card (fastest link) takes the whole one-card limit, every other card what it keeps up with (its balanced
                // tokens, at most the limit). split-pass=1: every pass that big, the rest of the call in a last pass. split-pass=2: the fewest passes that
                // size allows, all of the same size (the measured 60k GLM 3.0 run: a short last pass costs the experts' whole upload, 3.2 s for 1974
                // tokens, where 15k-token passes take 6.5 s; equal passes leave no pass that short and need no more passes).
                const int32_t R = n - off;
                // The card limit of THIS pass: the most tokens the expert card's range can take now. The pool holds the cached experts, the pass's graph (lent)
                // and the request's KV, which grows with every pass (15 KiB per token per card on Qwen3.8-Flash-Next): the pass of the context's card limit can
                // stop fitting as the prompt goes on (a 31k pass at 68k cells needed 20.5 GiB lent + the KV, refused, 2026-10-24). Every pool's fit
                // (llama_moe_cache_pass_fit_fork, the planner's per-pool cap: measured graph bytes per token against what the pool lends beside the KV of the
                // pass's cells) bounds the pass; when it binds, the card limit is scaled down until the pass it gives fits, so the expert card still takes its
                // whole (smaller) limit and the others their balanced share. Each step shrinks the limit, so the walk ends.
                uint32_t cap_p = cap;
                uint32_t W = llama_dsplit_pass_size_fork(cap_p);
                for (uint32_t fit = llama_moe_cache_pass_fit_fork(W, W); fit < W && cap_p > 1; fit = llama_moe_cache_pass_fit_fork(W, W)) {
                    cap_p = std::max<uint32_t>(1, (uint32_t) ((uint64_t) cap_p * fit / W));
                    W = llama_dsplit_pass_size_fork(cap_p);
                }
                if (cnt_fit > 0 && (int32_t) W > cnt_fit) {   // a pool refused a bigger pass in this call
                    cap_p = std::max<uint32_t>(1, (uint32_t) ((uint64_t) cap_p * (uint64_t) cnt_fit / W));
                    W = llama_dsplit_pass_size_fork(cap_p);
                }
                llama_dsplit_note_limit_fork(cap_p);
                int32_t cnt = std::min<int32_t>((int32_t) W, R);
                if (split_pass == 2) {
                    const int32_t k_r = (R + (int32_t) W - 1) / (int32_t) W;
                    cnt = (R + k_r - 1) / k_r;
                }
                LLAMA_LOG_WARN("%s: dense split pass %d: %d tokens (pass limit %u = the expert card's card limit %u + what the others keep up with%s%s%s), %d of %d left\n",
                        __func__, ++k, cnt, W, cap_p, split_pass == 2 ? "; equal passes" : "", "",
                        cap_p < cap ? format("; the card limit %u of the context does not fit the pools now", cap).c_str() : "", R - cnt, n);
                llama_batch_ext sub(batch_inp);
                sub.tokens.assign(tk.begin() + off, tk.begin() + off + cnt);
                const int r = decode(sub);
                size_t need = 0, room = 0;
                if (r != 0 && llama_moe_cache_refusal_fork(&need, &room) && room > 0 && room < need) {
                    // A pool refused the pass (its graph and the KV of its cells do not fit). Not the call: the same pass again, smaller by room / need (the graph
                    // grows about linearly with the tokens; a second refusal shrinks again), so the server never halves the whole call (2026-10-24: a refused
                    // 31k pass sent the server into calls of 3-19k tokens, every pass of them small). The failed sub-call took its cells back (decode's rollback).
                    const int32_t c2 = (int32_t) ((double) cnt * (double) room / (double) need);
                    if (c2 > 0 && c2 < cnt) {
                        LLAMA_LOG_WARN("%s: dense split pass %d refused by a pool (needs %.0f MiB, room %.0f MiB): again with %d tokens\n", __func__, k, need / 1048576.0, room / 1048576.0, c2);
                        cnt_fit = c2;
                        --k;
                        continue;
                    }
                }
                if (r == 1 && g_oom_retry.exchange(0) && ++oom_k <= 8) {   // a CUDA out-of-memory took this pass back (decode): the same pass again, under the ceiling
                    --k;
                    continue;
                }
                if (r != 0) {
                    g_split_sub = false;
                    if (off == 0) { return r; }
                    LLAMA_LOG_ERROR("%s: dense split pass %d failed (%d) after %d tokens of this call ran: the call fails\n", __func__, k, r, off);
                    return -2;
                }
                last_off = off;
                off += cnt;
                ran = true;
            }
            g_split_sub = false;
            if (last_off > 0 && tk[n - 1].output) {   // the output of the last pass: its index in the call, not in the pass
                const int32_t cnt = n - last_off;
                std::vector<int32_t> head(output_ids.begin(), output_ids.begin() + cnt);
                std::fill(output_ids.begin(), output_ids.end(), -1);
                std::copy(head.begin(), head.end(), output_ids.begin() + last_off);
            }
            return 0;
        }
    }
    if (!fork_resident_checked) {
        fork_resident_checked = true;
        bool all_gpu = !cparams.moe_cache && !model.tensors_by_name.empty();
        for (const auto & kv : model.tensors_by_name) {
            const ggml_tensor * t = kv.second;
            if (!all_gpu) { break; }
            if (!t || !t->buffer || kv.first == "token_embd.weight") { continue; }   // the input embedding stays in host memory in stock too
            ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            all_gpu = d != nullptr && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        const char * rx = moe_opt("resident-exit");   // --moe resident-exit=0 keeps the hooks for a model that sits in VRAM (A/B of this exit)
        const char * fo = moe_opt("fork-off");        // --fork off: the same exit whatever the placement is
        if ((all_gpu && (!rx || atoi(rx) != 0)) || (fo && atoi(fo) != 0)) {
            fork_resident = true;
        }
        if (fork_resident) {
            thr.on  = false;   // the decode and batch thread tuners: no CPU op in the graph
            thrb.on = false;
            LLAMA_LOG_INFO("%s: the fork's per-call hooks are skipped (stock path): %s\n", __func__, fo && atoi(fo) != 0 ? "--fork off" : "all weights are on the GPU and there is no expert cache");
        }
    }
    if (!fork_resident && !cparams.moe_cache && !moe_observe_started && !moe_cache_defer) {
        moe_observe_started = true;
        llama_moe_observe_start(model);
    }
    if (moe_observe_started && batch_inp.tokens.size() <= 8) {   // the answer, not the question: prompt batches are not counted (and do not use up the step budget)
        llama_moe_observe_step();
    }
    if (cparams.moe_cache && !cparams.moe_cache_started && !moe_cache_defer) {
        cparams.moe_cache_started = true;
        const char * alias_opt = llama_moe_get_opt("alias");
        const bool   alias     = !alias_opt || atoi(alias_opt) != 0;   // default on; the pool decides whether there is anything to lend
        if (alias) {   // the compute buffers go back to the devices first: the pool is sized with that memory and lends it back to the graphs
            for (auto & be : backends) {
                if (ggml_backend_dev_type(ggml_backend_get_device(be.get())) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    ggml_backend_sched_set_external_buffer(sched.get(), be.get(), nullptr, 0, 0);
                }
            }
        }
        llama_moe_cache_kv_tail_detach_fork();   // --moe kv-tail=1: the KV records' small buffers go back to the card before the pool is sized
        llama_moe_cache_init_fork(model, cparams.moe_cache_slots, cparams.moe_cache_inserts, cparams.prefetch_experts_slots < 0 ? 3 : cparams.prefetch_experts_slots, cparams.moe_cache_window,
                cparams.moe_predict, cparams.moe_predict_train);
        if (alias) {
            moe_alias_bufs = llama_moe_cache_alias_buffers_fork();   // applied to every scheduler created from now on (sched_reserve makes a new one)
        }
        // --moe mirror-pool: the dense split's copies get their places now that the pools exist (the start of each card's pool, or a buffer of their own);
        // the reserve below then sizes the worst case behind them
        moe_mirrors_placed = true;
        llama_dsplit_mirror_place_fork(model, moe_alias_bufs);
        if (alias) {
            apply_moe_alias();
            sched_need_reserve = true;
        }
        // no cap from the RAM probe (3 x the threads that saturate the read rate): the decode search starts on every hardware thread and steps down
        // itself (user 2026-10-26: start from the most threads; too many costs little, too few a lot)
        sched_reserve(); // the compute graph now includes the cache chain
    }
    if (g_dynkey.pending && cparams.moe_cache_started && !cparams.ctx_other) {   // fork (--moe dynhot): the key rows of the prompt's last pass to the MoE cache
        synchronize();
        std::vector<const float *> rows;
        for (const auto & r : g_dynkey.rows) { rows.push_back(r.empty() ? nullptr : r.data()); }
        llama_moe_topic_act_fork(rows.data(), (int) rows.size(), (int64_t) model.hparams.n_embd);
        g_dynkey.pending = false;
    }
    if (cparams.moe_cache_started && !cparams.ctx_other) {   // fork (--moe dynhot=1, hot maps per topic): the first call of a prompt looks up the stored requests nearest to it
        std::vector<int32_t> ids;
        ids.reserve(batch_inp.tokens.size());
        for (const auto & t : batch_inp.tokens) { if (!t.has_embd && t.id != LLAMA_TOKEN_NULL) { ids.push_back(t.id); } }
        llama_moe_cache_topic_begin_fork(model, ids.data(), (int32_t) ids.size());
    }
    {   // --moe kv-tail=1: KV records no expert pool took (dense models) get their buffer, once, at the first real decode. Only this context's records: the records of
        // the context that expects the pool wait for it (a draft decoding at load took the target's records into a fixed full-context arena, 7 GiB on DeepSeek 4.1);
        // that context hands the ones its pool did not take to an arena once the pool started (or when it decodes with the start deferred, as before)
        extern void llama_moe_cache_kv_arena_start_own_fork(bool pool_records);
        llama_moe_cache_kv_arena_start_own_fork(cparams.moe_cache && (cparams.moe_cache_started || moe_cache_defer));
    }
    {
        extern const char * llama_moe_cache_kv_arena_unplaced_fork();
        if (const char * why = llama_moe_cache_kv_arena_unplaced_fork()) {   // no memory for the KV: refuse, do not build a graph over records that have none
            LLAMA_LOG_ERROR("%s: %s\n", __func__, why);
            return -2;
        }
    }
    if (!fork_resident && !cparams.moe_cache && !moe_mirrors_placed) {
        moe_mirrors_placed = true;
        llama_dsplit_mirror_place_fork(model, moe_alias_bufs);   // no pool: the copies get buffers of their own
        if (llama_dsplit_ready_fork(model)) {
            sched_need_reserve = true;   // the split may be used from now on: its worst case
            sched_reserve();
        }
    }
    if (cparams.moe_cache_started && !moe_probe_done) {   // --moe max-batch=auto: time the cache path once, before the first real batch
        moe_probe_done = true;
        if (llama_moe_cache_probe_wanted_fork() && memory && memory->seq_pos_max(0) < 0) {
            moe_probe_costs_fork();
        }
    }
    if (cparams.moe_cache_started && !moe_swap_test_done) {   // --moe swap-test=A:B: swap two layers once, before the first batch (test of the swap)
        moe_swap_test_done = true;
        if (const char * e = llama_moe_get_opt("swap-test")) {
            int a = -1, b = -1;
            if (sscanf(e, "%d:%d", &a, &b) == 2) {
                swap_layers_fork(a, b);
            }
        }
    }
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    if (!mtp_dsa_sel_raw.empty()) {
        synchronize();
    }
    mtp_dsa_sel_raw.clear();
    mtp_dsa_sel_mask.clear();
    mtp_dsa_sel_seq.clear();
    mtp_dsa_sel.clear();
    mtp_dsa_sel_width = 0;

    sched_reserve();

    bool did_optimize = false;
    bool did_retier   = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    // prompt batches: equal passes sized from the number of new tokens, a tail the cache path can take is left to it (n_ubatch is the maximum)
    // The attention mask is a pinned host buffer of (pass tokens x cells the pass sees x 2 bytes), re-pinned whenever the horizon grows: a 255k-token prompt in passes of 13.6k
    // tokens wanted 6.5 GiB of it, and the kernel could not pin that beside the 108 GiB weight pool (PIN_STALL, exit 75). The pass planner weighs uploads and graph bytes but
    // not this buffer, so the pass size is capped here: mask budget / (2 x cells at the end of this call). --moe mask-mb=N overrides the calculated budget (0 = no cap).
    { static bool said = false; if (!said && n_tokens_all >= 64) { said = true; LLAMA_LOG_WARN("%s: pass cap inputs: moe_cache %d, memory %d, n_tokens_all %u, n_ubatch %u\n", __func__, (int) cparams.moe_cache, (int) (memory != nullptr), n_tokens_all, cparams.n_ubatch); } }
    uint32_t n_ubatch_lim = cparams.n_ubatch;
    if (const uint64_t tc = g_oom_tc.load(); tc > 0 && memory && n_tokens_all > 1) {   // after a CUDA out-of-memory (g_oom_tc): at most tc / cells tokens a pass
        const uint64_t n_kv_end = (uint64_t) (std::max<llama_pos>(memory->seq_pos_max(0), -1) + 1) + n_tokens_all;
        const uint32_t lim = (uint32_t) std::min<uint64_t>(n_ubatch_lim, std::max<uint64_t>(256, tc / std::max<uint64_t>(1, n_kv_end) / 32 * 32));
        if (lim < n_ubatch_lim) {
            static uint32_t said = 0;
            if (said != lim) {
                said = lim;
                char v[64], why[200];
                snprintf(v, sizeof v, "%u tokens (CUDA out-of-memory ceiling)", lim);
                snprintf(why, sizeof why, "a pass at %llu cells may have %llu tokens x cells: half of the pass that ran out of VRAM", (unsigned long long) n_kv_end, (unsigned long long) tc);
                llama_moe_decision_fork("prompt pass", v, why);
                LLAMA_LOG_WARN("%s: prompt pass %s: %s\n", __func__, v, why);
            }
            n_ubatch_lim = lim;
        }
    }
    double   mask_budget_mib = 0.0;
    if (cparams.moe_cache && memory) {
        const char * mm = llama_moe_get_opt("mask-mb");
        // Calculated, not guessed: the mask is pinned host memory. A pin served by free 1 GiB pool pages is immediate and certain (a 5.0 GiB mask took 0.3 s from 5 free pages),
        // every page beyond that depends on the kernel finding pinnable memory (the 6.5 GiB request stalled for 180 s). So the budget is the free pool pages right now (at least 1 GiB),
        // or, with no pool, the available RAM less the 4 GiB margin the weight loader keeps. --moe mask-mb=N overrides.
        uint64_t budget = 0;
        uint64_t pool_free = 0, pool_total = 0, avail = 0;
        if (FILE * f = fopen("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages", "r")) { unsigned long long v = 0; if (fscanf(f, "%llu", &v) == 1) { pool_free = v; } fclose(f); }
        if (FILE * f = fopen("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages", "r"))   { unsigned long long v = 0; if (fscanf(f, "%llu", &v) == 1) { pool_total = v; } fclose(f); }
        if (FILE * f = fopen("/proc/meminfo", "r")) {
            char line[256];
            while (fgets(line, sizeof line, f)) { unsigned long long kb = 0; if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { avail = kb << 10; break; } }
            fclose(f);
        }
        if (mm) {
            budget = (uint64_t) atoll(mm) << 20;
        } else if (const char * sr = getenv("GGML_SIM_RAM_MB")) {   // --moe sim-ram=MB: this much can be pinned (test runs)
            budget = (uint64_t) atoll(sr) << 20;
        } else {
            const uint64_t margin = (uint64_t) 4 << 30;
            // ONE pinned host buffer holds a pass's graph inputs (the startup fit measures it: 2 bytes per pass token per cell, one f16 mask; MiMo's 2.5 GiB pin of a
            // 16k pass at 32k cells is that buffer). It is pinned from free pool pages rounded up to whole pages, keeping one page spare
            // (ggml_cuda_host_malloc_registered); the old buffer is freed before the new one is pinned, so the pages free now, less one, are what it can have.
            // (Until 2026-10-17 the budget was divided by the number of GPUs, from a misread stall: the allocator rounded the pool part down and pinned the
            // tail from transparent huge pages, and that tail stalled. It halved the passes' size at long context: 63 passes instead of about 35 at 255k.)
            // The buffer the scheduler holds now is freed before the next one is pinned, so its pool pages count as free (they were budgeted from the pool when it
            // was pinned: whole pages, rounded up); before 2026-10-17 they did not, and the budget was short by the current buffer (2 GiB of 4 at MiMo's start).
            uint64_t held_p = 0;
            if (pool_total > 0 && backend_cpu) {
                const size_t hb = ggml_backend_sched_get_buffer_size(sched.get(), backend_cpu);
                held_p = hb > ((size_t) 512 << 20) ? (uint64_t) ((hb + ((size_t) 1 << 30) - 1) >> 30) : 0;
            }
            budget = pool_total > 0 ? std::max<uint64_t>((uint64_t) 1 << 30, (pool_free + held_p > 1 ? pool_free + held_p - 1 : 0) << 30)
                                    : std::max<uint64_t>((uint64_t) 1 << 30, avail > margin ? avail - margin : 0);
        }
        const uint64_t n_kv_end = (uint64_t) (std::max<llama_pos>(memory->seq_pos_max(0), -1) + 1) + n_tokens_all;
        mask_budget_mib = budget / 1048576.0;
        if (budget > 0 && n_kv_end > 0 && g_host_ok) {
            const double per_tok = std::max(1.0, g_host_a * (double) n_kv_end + g_host_b);   // bytes of host buffer per pass token at that many cells (fitted at startup)
            const uint64_t cap = std::max<uint64_t>(256, (uint64_t) std::max(0.0, ((double) budget - g_host_c - g_host_d * (double) n_kv_end) / per_tok) / 32 * 32);
            // user 2026-10-22: the pinned host buffer no longer caps the pass (GLM 3.0 with its weights pinned to the last GiB had a 1.0-1.9 GiB budget: its 12k
            // prompt ran as 5 passes instead of b13600's 3, 520 against 641 prompt t/s). The loader keeps the RAM of a full pass's host buffer free instead
            // (common_pass_buffer_reserve); --moe mask-cap=1 brings the cap back. The pinned-memory rewrite after this release replaces both.
            static const bool mask_cap = [] { const char * e = llama_moe_get_opt("mask-cap"); return e && atoi(e) != 0; }();
            if (mask_cap && cap < n_ubatch_lim) { n_ubatch_lim = (uint32_t) cap; }
            // A slower prompt rather than none (user 2026-10-25): the pass's host buffer grows with the context (the f16 mask is 2 bytes per pass token per cell: 6.3 GiB
            // for a 16384-token pass at 203k cells) and was pinned beside 108 GiB of pinned weights: RAM ran out, the box thrashed (PSI 97%) and the MiMo 200k prompt
            // failed. Pinning does not fail, the kernel swaps and stalls, so there is nothing to catch: the pass is cut BEFORE it is pinned, to the largest that fits, while its buffer (the
            // startup fit, the same numbers as the check below) would leave less than the RAM margin (4 GiB). RAM = available + free pool pages + the
            // buffer the scheduler holds now (freed before the new one is pinned). Each halving re-uploads the experts once more per pass: slower, never stuck.
            {
                uint64_t ram = avail + (pool_free << 30) + (backend_cpu ? (uint64_t) ggml_backend_sched_get_buffer_size(sched.get(), backend_cpu) : 0);
                if (const char * sr = getenv("GGML_SIM_RAM_MB")) { ram = (uint64_t) atoll(sr) << 20; }
                // 4 GiB (user 2026-10-25, later the same day: 8 GiB cut DeepSeek-V4.1 passes to ~4.2k tokens beside 101 GiB pinned). Earlier that day 4 GiB froze the box
                // at 164k cells with 3.2 GiB left (MiMo 200k, 108 GiB pinned): pin less (-nm) for such prompts, or raise it with ram-margin-mb
                // --moe ram-margin-mb=N overrides (e.g. 12288 for a 12 GiB margin)
                const char * rm = llama_moe_get_opt("ram-margin-mb");
                const uint64_t ram_margin = rm ? (uint64_t) atoll(rm) << 20 : (uint64_t) 4 << 30;
                const double room = ram > ram_margin ? (double) (ram - ram_margin) : 0.0;
                auto need = [&](uint32_t u) { return (double) u * per_tok + g_host_d * (double) n_kv_end + g_host_c; };
                uint32_t lim = n_ubatch_lim;
                if (need(lim) > room) {
                    // the largest pass that fits, not the next power of two down (user 2026-10-25): the buffer is linear in the pass tokens (the startup fit above), so the pass
                    // that leaves exactly the margin is room less the fixed parts over the bytes per token. The caller cuts a call into equal passes of at most this many tokens
                    // (32768 tokens: 3 passes of 10944 for a limit of 12000, not 4 of 8192 as halving gave).
                    const double u_fit = (room - g_host_c - g_host_d * (double) n_kv_end) / per_tok;
                    lim = std::max<uint32_t>(256, (uint32_t) std::max(0.0, std::min((double) lim, u_fit)) / 32 * 32);
                }
                if (lim < n_ubatch_lim) {
                    char v[64], why[256];
                    snprintf(v, sizeof v, "%u tokens (cut from %u to fit)", lim, n_ubatch_lim);
                    snprintf(why, sizeof why, "RAM: a %u-token pass needs %.0f MiB of host buffer at %llu cells, %.0f MiB of RAM is left beside the %.0f GiB margin",
                             n_ubatch_lim, need(n_ubatch_lim) / 1048576.0, (unsigned long long) n_kv_end, room / 1048576.0, ram_margin / 1073741824.0);
                    llama_moe_decision_fork("prompt pass", v, why);
                    LLAMA_LOG_WARN("%s: prompt pass %s: %s\n", __func__, v, why);
                    n_ubatch_lim = lim;
                }
            }
        }
    }
    if (cparams.moe_cache && memory) { llama_moe_cache_pp_cells_fork((uint64_t) (std::max<llama_pos>(memory->seq_pos_max(0), -1) + 1) + n_tokens_all); }
    g_decode_n_tokens = n_tokens_all;
    // fork: the dense split sizes its ranges against the one-card limit (no card gets more than it, ds_plan); a pass of a call sized pass by pass
    // (--moe split-pass=1, the loop at the top of decode) runs as ONE pass: the pass planner still runs for its claim of the prompt's KV (kv-tail)
    if (n_tokens_all >= 64 && cparams.moe_cache && memory && !g_split_sub) { llama_dsplit_note_limit_fork(n_ubatch_lim); }
    const uint32_t n_ubatch_split = g_split_sub && cparams.moe_cache ? (llama_moe_cache_prefill_ubatch_fork(n_tokens_all, n_tokens_all), n_tokens_all) : 0;
    const uint32_t n_ubatch_call = n_ubatch_split > 0 ? n_ubatch_split : cparams.moe_cache ? llama_moe_ubadapt_cache_fork(n_tokens_all, llama_moe_cache_prefill_ubatch_fork(n_tokens_all, n_ubatch_lim), n_ubatch_lim) : llama_moe_ubadapt_next_fork(n_tokens_all, n_ubatch_lim);
    if (n_tokens_all > 1) { llama_moe_prompt_begin_fork((uint32_t) n_tokens_all, n_ubatch_call); }
    if (cparams.moe_cache && memory && n_tokens_all >= 64) {   // the mask math, checkable against the host pins in the same log (--moe log=m not needed: one line per call)
        const uint64_t kv_end = (uint64_t) (std::max<llama_pos>(memory->seq_pos_max(0), -1) + 1) + n_tokens_all;
        const uint64_t mask_b = (uint64_t) GGML_PAD(n_ubatch_call, 64) * GGML_PAD(kv_end, 256) * 2;
        LLAMA_LOG_WARN("%s: pass size: %u tokens in this call, %llu cells at its end, pass of %u (limit %u from the pinned-mask budget %.0f MiB), the mask of a full pass %.0f MiB\n", __func__,
                n_tokens_all, (unsigned long long) kv_end, n_ubatch_call, n_ubatch_lim, mask_budget_mib, mask_b / 1048576.0);
        // the startup fit checked against the real graphs, every model: the host buffer the scheduler holds now (the largest graph reserved so far) against
        // the fit at this call's pass and cells. Actual well above predicted = an input the fit does not see (it grows with something the calibration did not vary).
        if (g_host_ok && backend_cpu) {
            const double pred = (double) n_ubatch_call * (g_host_a * (double) kv_end + g_host_b) + g_host_d * (double) kv_end + g_host_c;
            const double act  = (double) ggml_backend_sched_get_buffer_size(sched.get(), backend_cpu);
            LLAMA_LOG_WARN("%s: host buffer check: fit %.0f MiB for this call's pass, scheduler holds %.0f MiB (the largest graph so far)%s\n", __func__,
                    pred / 1048576.0, act / 1048576.0, act > 1.10 * pred + 64.0 * 1048576.0 ? " -- MORE than the fit: an input the calibration misses" : "");
        }
    }

    while (true) {
        mctx = memory->init_batch(*balloc, n_ubatch_call, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    if (cparams.moe_cache && !did_retier) {   // qkv-ram-fallback in place: the KV of this tier is full: the next tier, same batch again
                        did_retier = true;
                        char note[512] = { 0 };
                        llama_pos hist = -1;   // the longest history of any sequence (-np > 1: the batch may belong to another one than sequence 0)
                        for (uint32_t sq = 0; sq < cparams.n_seq_max; ++sq) { hist = std::max(hist, memory->seq_pos_max((llama_seq_id) sq)); }
                        const uint32_t need = (uint32_t) (std::max<llama_pos>(hist, -1) + 1) + n_tokens_all;
                        const int32_t r = kv_fit_select(need, -1, note, sizeof(note));
                        if (note[0]) { LLAMA_LOG_WARN("%s: %s\n", __func__, note); }
                        if (r >= 0 && (r & 256)) { continue; }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // the first position of every sequence in this call: a CUDA out-of-memory takes the whole call back (g_oom_tc)
    std::vector<std::pair<llama_seq_id, llama_pos>> call_pos_min;
    for (uint32_t sq = 0; sq < cparams.n_seq_max && sq < (uint32_t) LLAMA_MAX_SEQ; ++sq) {
        const llama_pos p0 = balloc->seq_pos_min((llama_seq_id) sq);
        if (p0 >= 0) { call_pos_min.emplace_back((llama_seq_id) sq, p0); }
    }

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        if (thrb.on) {
            // batch thread tuner: the time from the start of the previous prompt ubatch to the start of this one is the previous one's cost
            // (prompt-only batches are not synchronized one by one, so the compute time itself is not observable here)
            const int64_t now = ggml_time_us();
            if (thrb.last_tokens >= 64 && now - thrb.last_us < 20*1000*1000) {
                thread_tune_feed_batch(now - thrb.last_us, thrb.last_tokens);
            }
            thrb.last_us = now;
            thrb.last_tokens = ubatch.n_tokens > 1 ? ubatch.n_tokens : 0;
        }

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        // CUDA out-of-memory errors the backend caught (ggml_cuda_oom), also one it got past by itself (a CUDA graph that did not fit is run without
        // one): the next start leaves room for them
        static int oom_seen = 0;
        int oom_dev = -1; size_t oom_bytes = 0; double oom_fit = 0.0;
        const int  oom_n   = fork_cuda_last_oom(&oom_dev, &oom_bytes, &oom_fit);
        const bool oom_new = oom_n > oom_seen;
        if (oom_new) {
            oom_seen = oom_n;
            if (oom_dev >= 0) { llama_moe_cache_vram_oom_fork(oom_dev, oom_bytes); }
        }
        if (!res && status == GGML_STATUS_ALLOC_FAILED && memory && oom_new) {   // a CUDA out-of-memory (not a graph the scheduler could not place)
            // the pool could not grow (oom_bytes > 0): the same call in passes of the share of the tokens x cells that fits (the pool's bytes and the card's free ones, at most
            // 0.9 of it as later temporaries may be bigger, at least a quarter; half where the share is not known); a CUDA graph could not be instantiated after part
            // of the graph ran: the same call as it was, the graph now runs without a CUDA graph
            const bool inst = oom_bytes == 0;
            llama_pos cells = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) { cells = std::max(cells, ubatch.pos[i] + 1); }
            bool back = inst || ubatch.n_tokens > 256;   // 256 tokens that do not fit: an error (see g_oom_tc)
            for (size_t k = 0; back && k < call_pos_min.size(); ++k) { back = memory->seq_rm(call_pos_min[k].first, call_pos_min[k].second, -1); }
            if (back) {
                if (!inst) {
                    const double share = oom_fit > 0.0 ? std::min(0.9, std::max(0.25, oom_fit * 0.9)) : 0.5;
                    const uint64_t tc = (uint64_t) ((double) ubatch.n_tokens * (double) std::max<llama_pos>(cells, 1) * share), prev = g_oom_tc.load();
                    g_oom_tc = prev ? std::min(prev, tc) : tc;
                }
                g_oom_retry = 1;
                // the failed call is no measurement: the split factors and the balance would time the failure and the retry as one pass, the pass planner
                // and the prompt counters would count its tokens twice. What the other cards still run of the failed graph ends first; the retry is timed
                // from its own start (an earlier call not synchronized yet loses its sample: it cannot be told apart from the failure).
                llama_dsplit_pass_abandon_fork();
                ggml_backend_sched_synchronize(sched.get());
                n_queued_tokens    = 0;
                t_compute_start_us = 0;
                thrb.last_tokens   = 0;
                if (inst) {
                    LLAMA_LOG_WARN("%s: CUDA graph out of VRAM on CUDA%d in a %u-token pass at %d cells: the call is taken back and runs again without that CUDA graph\n",
                            __func__, oom_dev, ubatch.n_tokens, (int) cells);
                } else {
                    LLAMA_LOG_WARN("%s: CUDA out of memory on CUDA%d in a %u-token pass at %d cells: the call is taken back and runs again with at most %llu tokens x cells a pass\n",
                            __func__, oom_dev, ubatch.n_tokens, (int) cells, (unsigned long long) g_oom_tc.load());
                }
                return 1;
            }
        }
        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                // fork: a graph the pool could not lend next to the KV (llama_moe_cache_alias_err_fork) is 'no room for this many tokens': 1 makes the caller try a smaller batch, as it
                // does when the KV has no slot; the ubatch's entries were removed above. Not a compute error (the server answered HTTP 500 "Compute error" for a 40k-token prompt on W10).
                case GGML_STATUS_FAILED:       return llama_moe_cache_alias_err_fork() ? 1 : -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        auto * t_mtp_sel  = res->get_mtp_dsa_sel();
        auto * t_mtp_mask = res->get_mtp_dsa_mask();
        if (t_mtp_sel != nullptr || t_mtp_mask != nullptr) {
            GGML_ASSERT(t_mtp_sel != nullptr && t_mtp_mask != nullptr);
            GGML_ASSERT(t_mtp_sel->type == GGML_TYPE_I32 && t_mtp_mask->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(t_mtp_sel) && ggml_is_contiguous(t_mtp_mask));
            GGML_ASSERT(t_mtp_sel->ne[0] == t_mtp_mask->ne[0] && t_mtp_sel->ne[1] == (int64_t) ubatch.n_tokens);
            GGML_ASSERT(ggml_nelements(t_mtp_sel) == ggml_nelements(t_mtp_mask));

            const size_t width = (size_t) t_mtp_sel->ne[0];
            if (mtp_dsa_sel_width == 0) {
                mtp_dsa_sel_width = width;
                mtp_dsa_sel_raw.resize(width*n_tokens_all);
                mtp_dsa_sel_mask.resize(width*n_tokens_all);
                mtp_dsa_sel_seq.resize(n_tokens_all, -1);
            }
            GGML_ASSERT(width == mtp_dsa_sel_width);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                GGML_ASSERT(ubatch.n_seq_id[i] == 1);
                mtp_dsa_sel_seq[(size_t) n_tokens_prev + i] = ubatch.seq_id[i][0];
            }

            const size_t offset = width*(size_t) n_tokens_prev;
            ggml_backend_t backend_sel = ggml_backend_sched_get_tensor_backend(sched.get(), t_mtp_sel);
            ggml_backend_t backend_mask = ggml_backend_sched_get_tensor_backend(sched.get(), t_mtp_mask);
            GGML_ASSERT(backend_sel != nullptr && backend_mask != nullptr);
            ggml_backend_tensor_get_async(backend_sel, t_mtp_sel, mtp_dsa_sel_raw.data() + offset, 0, ggml_nbytes(t_mtp_sel));
            ggml_backend_tensor_get_async(backend_mask, t_mtp_mask, mtp_dsa_sel_mask.data() + offset, 0, ggml_nbytes(t_mtp_mask));
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
        if (n_tokens_all > 1) { llama_moe_prompt_progress_fork((uint32_t) n_tokens_prev); }
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    // apply throttled MoE expert-cache updates between graph executions; wait for
    // the GPU first so no queued read of a slot races the worker refilling it
    static const bool mc_sync = !moe_opt("sync") || atoi(moe_opt("sync")) != 0;
    // only the context that owns the cache steps it (not e.g. an MTP draft context)
    if (cparams.moe_cache) {
        if (mc_sync && llama_moe_cache_active_fork()) {
            ggml_backend_sched_synchronize(sched.get());
        }
        llama_moe_cache_step_fork();
    }

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }
    // fork: only the last rows of a ubatch are written (embeddings_layer_inp_last): the layer inputs go to lazily paged memory, not into the pinned,
    // cleared output buffer (DeepSeek-V4.1 + DSpark at 16384 rows: 1 GB pinned and cleared for 129 rows per pass the draft reads)
    const bool   layer_inp_lazy   = cparams.embeddings_layer_inp_last > 0 && embd_layer_inp_float_count > 0;
    const size_t layer_inp_pinned = layer_inp_lazy ? 0 : embd_layer_inp_float_count;
    if (layer_inp_lazy && embd_layer_inp_lazy_n < embd_layer_inp_float_count) {
        embd_layer_inp_lazy.reset(new float[embd_layer_inp_float_count]);
        embd_layer_inp_lazy_n = embd_layer_inp_float_count;
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + layer_inp_pinned + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    size_t lazy_offset = 0;
    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il] && layer_inp_lazy) {
            embd_layer_inp[il] = buffer_view<float>{embd_layer_inp_lazy.get() + lazy_offset, (size_t) n_embd * n_batch};
            lazy_offset += embd_layer_inp[il].size;
        } else if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);

        const size_t row_floats = model.hparams.n_embd;
        GGML_ASSERT(nfloats % row_floats == 0);
        const size_t rows = nfloats / row_floats;   // the ubatch's rows, or only its last ones (cparams.embeddings_layer_inp_last)
        GGML_ASSERT(rows > 0 && rows <= n_tokens);
        const size_t dst_offset = (token_offset + n_tokens - rows) * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    // MoE expert cache chain, per layer: slot lookup and CPU pinning, plus up/gate/
    // activation/down/concat per 4-token chunk of batches up to max_batch. The
    // scheduler is sized before the cache exists, so budget for it up front.
    if (cparams.moe_cache) {
        const uint32_t nt = (uint32_t) std::min<int64_t>(n_tokens, llama_moe_cache_max_batch_cap_fork());
        res += model.hparams.n_layer() * (16u + 12u * ((nt + 3) / 4));
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    llama_dsplit_reserve_fork(model);   // fork: reserve for the larger (split) graph when the dense split may be used
    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        { const int64_t t_ids = ggml_time_us(); ggml_backend_synchronize(backend); ggml_backend_sched_sync_stat_add(3, ggml_time_us() - t_ids); }

        // let an expert cache see which experts offloaded batches use (the old sched loop did this; the move to user code dropped it)
        void * obs_ud = nullptr;
        ggml_moe_obs_cb_t obs_cb = ggml_get_moe_obs_callback(&obs_ud);
        if (obs_cb && strstr(src->name, "ffn_gate_exps")) {
            ggml_tensor ids_host = *ids;
            ids_host.data = st.ids_data.data();
            obs_cb(src->name, &ids_host, nullptr, obs_ud);
        }

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id < n_expert);
                if (id >= 0) { // negative: skipped expert (2-GPU prefill, MoE cache)
                    st.used[id] = true;
                }
            }
        }

        st.ids = ids;
    }

    // experts already resident on this device (the MoE expert cache) are copied device-to-device,
    // runs of the rest go over PCIe as before
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    auto bytes_view = [](ggml_backend_buffer_t buf, void * data, size_t n) {
        ggml_tensor t = {};
        t.type   = GGML_TYPE_I8;
        t.buffer = buf;
        t.data   = data;
        t.ne[0] = (int64_t) n; t.ne[1] = t.ne[2] = t.ne[3] = 1;
        t.nb[0] = 1; t.nb[1] = t.nb[2] = t.nb[3] = n;
        return t;
    };

    // group consecutive experts and copy them together
    auto copy_run = [&](int64_t first, int64_t last) {
        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
                ggml_backend_sched_upload_add_fork(ggml_backend_get_device(backend), (uint64_t) (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, (last - first + 1)*expert_size + padding);
        ggml_backend_sched_upload_add_fork(ggml_backend_get_device(backend), (uint64_t) (last - first + 1)*expert_size);
        if (last == n_expert - 1) {
            ggml_backend_tensor_zero_tail_fork(backend, dst);   // fork: the last expert has no neighbour: zeros in the tensor's own row padding (see ggml-backend.cpp)
        }
    };

    // Slice copies inside the device (cache -> staged tensor, staged tensor -> cache) are collected and issued as ONE kernel launch when the backend has one
    // (CUDA: ggml_backend_cuda_copy_slices_fork), instead of one driver call per expert and tensor.
    typedef bool (*copy_slices_t)(ggml_backend_t, const void * const *, void * const *, size_t, size_t);
    static std::map<ggml_backend_dev_t, copy_slices_t> slices_fn;
    copy_slices_t copy_slices = nullptr;
    {
        auto it = slices_fn.find(dev);
        if (it == slices_fn.end()) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            it = slices_fn.emplace(dev, reg ? (copy_slices_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_slices_fork") : nullptr).first;
        }
        copy_slices = it->second;
    }
    struct slice_pair { const void * s; void * d; ggml_backend_buffer_t sb; ggml_backend_buffer_t db; };
    std::vector<slice_pair> pairs;
    auto flush_slices = [&]() {
        if (pairs.empty()) { return; }
        std::vector<const void *> srcs; std::vector<void *> dsts;
        for (const auto & p : pairs) { srcs.push_back(p.s); dsts.push_back(p.d); }
        if (!(copy_slices && copy_slices(backend, srcs.data(), dsts.data(), pairs.size(), expert_size))) {
            for (const auto & p : pairs) {
                ggml_tensor s = bytes_view(p.sb, (void *) p.s, expert_size);
                ggml_tensor d = bytes_view(p.db, p.d, expert_size);
                ggml_backend_tensor_copy_async(backend, backend, &s, &d);
            }
        }
        pairs.clear();
    };

    int64_t first = -1;
    int64_t last  = -1;
    for (int64_t id = 0; id < n_expert; ++id) {
        if (!st.used[id]) {
            continue;
        }

        const void *          dev_data = nullptr;
        ggml_backend_buffer_t dev_buf  = nullptr;
        if (ggml_backend_moe_src(src, (int32_t) id, dev, &dev_data, &dev_buf)) {
            pairs.push_back({ dev_data, (uint8_t *) dst->data + (size_t) id*expert_size, dev_buf, dst->buffer });   // cache block -> slot in the staged tensor
            if (id == n_expert - 1) {
                ggml_backend_tensor_zero_tail_fork(backend, dst);   // the last expert, taken from the cache: the row padding behind it is the tensor's own
            }
            // keep the MMQ padding rule: the unused neighbour must not hold NaNs
            if (id < n_expert - 1 && !st.used[id + 1]) {
                const size_t off = (size_t) (id + 1)*expert_size;
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + off, off, std::min<size_t>(expert_size, 512));
            }
            continue;
        }

        if (first >= 0 && id == last + 1) {
            last = id;
            continue;
        }
        if (first >= 0) {
            copy_run(first, last);
        }
        first = id;
        last  = id;
    }
    if (first >= 0) {
        copy_run(first, last);
    }

    flush_slices();   // the resident experts, in one launch (they are read by the matmul after the uploads of the runs above, on the same stream)

    // the used experts are on the device now: let the expert cache keep some of them
    // (device-to-device, queued after the copies above on the same stream, one launch for all of them)
    for (int64_t id = 0; id < n_expert; ++id) {
        if (!st.used[id]) {
            continue;
        }
        void *                dst_data = nullptr;
        ggml_backend_buffer_t dst_buf  = nullptr;
        if (ggml_backend_moe_fill(src, (int32_t) id, dev, &dst_data, &dst_buf)) {
            pairs.push_back({ (uint8_t *) dst->data + (size_t) id*expert_size, dst_data, dst->buffer, dst_buf });
        }
    }
    flush_slices();

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // --moe expert-owner=1 (research, prompt batches only): the expert matmuls of a layer run on the card that owns the layer, and so do their uploads from host memory,
        // instead of all on the offload card (the scheduler sends any expert weight in host memory to the first backend that offloads). With the layers shared out by
        // upload bandwidth (--moe split=bw) every link then works on its own layers at the same time, and activations cross cards where a card's block of layers
        // begins and ends, not twice per layer of the slow card.
        static const bool owner_on = [] { const char * e = llama_moe_get_opt("expert-owner"); return e && atoi(e) != 0; }();
        if (owner_on && il >= 0 && ubatch.n_tokens >= 32 && (strcmp(name, "ffn_moe_up") == 0 || strcmp(name, "ffn_moe_gate") == 0 || strcmp(name, "ffn_moe_down") == 0 ||
                                                            strcmp(name, "ffn_moe_gate_up") == 0)) {
            const auto dev = model.dev_layer(il);
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_supports_op(backend.get(), cur)) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                }
            }
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.n_moe_cache_slots           =*/ 0,
        /*.n_moe_cache_inserts         =*/ 2,
        /*.n_moe_cache_window          =*/ 64,
        /*.n_moe_predict               =*/ 0,
        /*.n_moe_predict_train         =*/ 0,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.prefetch_experts_slots      =*/ -1,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        llama_moe_cache_context_live_fork(ctx, model, true);   // fork: the expert pool keys its stored VRAM need by the other live contexts (a draft)
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    llama_moe_cache_context_live_fork(ctx, nullptr, false);
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_thread_autotune(llama_context * ctx, bool on) {
    ctx->set_thread_autotune(on);
}

void llama_set_batch_thread_autotune(llama_context * ctx, bool on, int32_t n_max) {
    ctx->set_batch_thread_autotune(on, n_max);
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_moe_cache_defer_fork(llama_context * ctx, bool defer) {
    ctx->moe_cache_defer = defer;
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_embeddings_layer_inp_last(llama_context * ctx, uint32_t n) {
    ctx->set_embeddings_layer_inp_last(n);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

bool llama_set_mtp_dsa_index_share(llama_context * ctx, bool enabled) {
    return ctx != nullptr && ctx->set_mtp_dsa_index_share(enabled);
}

bool llama_set_mtp_dsa_selection(llama_context * ctx, const int32_t * data, size_t size) {
    return ctx != nullptr && ctx->set_mtp_dsa_selection(data, size);
}

const int32_t * llama_get_mtp_dsa_selection(llama_context * ctx, size_t * size) {
    if (ctx == nullptr) {
        if (size != nullptr) {
            *size = 0;
        }
        return nullptr;
    }
    ctx->synchronize();
    return ctx->get_mtp_dsa_selection(size);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int64_t t_us = ggml_time_us();
    int ret = ctx->decode(batch);
    for (int t = 0; ret == 1 && t < 8 && g_oom_retry.exchange(0); ++t) { ret = ctx->decode(batch); }   // fork: taken back after a CUDA out-of-memory (g_oom_tc)
    if (ret == 0) {
        const double ms = (ggml_time_us() - t_us) / 1000.0;
        if (batch.n_tokens == 1) { llama_moe_token_time_fork(ms); } else { llama_moe_prompt_done_fork((uint32_t) batch.n_tokens, ms); }
        llama_moe_cache_vram_watch_fork(batch.n_tokens > 1);
    }
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

static void llama_log_tuning(const char * fn, const llama_context * ctx);

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);
    llama_moe_cache_kv_tail_report_fork("end of run");   // --moe kv-tail=1: KV cells and the expert cache per card (-lv 3)

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    llama_log_tuning(__func__, ctx);
}

int32_t llama_kv_fit_select(struct llama_context * ctx, uint32_t n_tokens, int32_t min_tier, char * note, size_t note_size) {
    return ctx->kv_fit_select(n_tokens, min_tier, note, note_size);
}

void llama_get_tuning_info(const llama_context * ctx, llama_tuning_info * info) {
    *info = {};
    info->margin = info->gate = info->wait = info->big = info->predict = info->self_tune = -1;
    info->moe_slots_min = info->moe_slots_max = info->moe_layers = -1;
    info->moe_active = llama_moe_cache_get_info_fork(info);
    info->kvt_ctx_train = ctx->get_model().hparams.n_ctx_train;
    info->n_threads       = (int32_t) ctx->get_cparams().n_threads;
    info->n_threads_batch = (int32_t) ctx->get_cparams().n_threads_batch;
    info->n_batch         = (int32_t) ctx->n_batch();
    info->n_ubatch        = (int32_t) ctx->n_ubatch();
}

static void llama_log_tuning(const char * fn, const llama_context * ctx) {
    llama_tuning_info t;
    llama_get_tuning_info(ctx, &t);
    if (t.moe_active) {
        const uint64_t n = t.moe_hits + t.moe_misses;
        LLAMA_LOG_INFO("%s:    moe cache = on, hit %.1f%% (%llu/%llu experts), %d layers, %d..%d slots/layer, %llu uploads\n", fn,
                n ? 100.0*t.moe_hits/n : 0.0, (unsigned long long) t.moe_hits, (unsigned long long) n, t.moe_layers, t.moe_slots_min,
                t.moe_slots_max, (unsigned long long) t.moe_uploads);
        LLAMA_LOG_INFO("%s:    tuned     = margin %d, gate %d, wait %d, big %d, predict %d, self-tune %s\n", fn, t.margin, t.gate, t.wait, t.big,
                t.predict, t.self_tune ? "on" : "off");
    } else {
        LLAMA_LOG_INFO("%s:    moe cache = off\n", fn);
    }
    LLAMA_LOG_INFO("%s:    threads   = %d decode / %d batch, batch %d, ubatch %d\n", fn, t.n_threads, t.n_threads_batch, t.n_batch, t.n_ubatch);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: {
            // the server's decode path: per-token and per-prompt-call times for GET /moe/stats
            const int64_t t_us = ggml_time_us();
            // debug (bug 153, draft crash): LLAMA_PROC_DBG=1 log each call, 2 + sync the draft context after it, 3 + sync the target
            // context around it, 4 + sync before it too. Staged so the level that hides the crash names the racing step.
            static const int dbg = getenv("LLAMA_PROC_DBG") ? atoi(getenv("LLAMA_PROC_DBG")) : 0;
            llama_context * other = ctx->get_cparams().ctx_other;
            if (dbg >= 4) { llama_synchronize(ctx); }
            if (dbg >= 3 && other) { llama_synchronize(other); }
            if (dbg >= 1) { LLAMA_LOG_WARN("procdbg: %s decode begin\n", other ? "draft" : "target"); }
            int ret = ctx->decode(*batch);
            if (dbg >= 2 && other) { llama_synchronize(ctx); }
            if (dbg >= 3 && other) { llama_synchronize(other); }
            if (dbg >= 1) { LLAMA_LOG_WARN("procdbg: %s decode end ret %d n %d\n", other ? "draft" : "target", ret, (int) g_decode_n_tokens); }
            for (int t = 0; ret == 1 && t < 8 && g_oom_retry.exchange(0); ++t) { ret = ctx->decode(*batch); }   // fork: taken back after a CUDA out-of-memory (g_oom_tc)
            if (ret == 0) {
                const int32_t n = (int32_t) g_decode_n_tokens;
                const double ms = (ggml_time_us() - t_us) / 1000.0;
                if (n == 1) { llama_moe_token_time_fork(ms); } else { llama_moe_prompt_done_fork((uint32_t) n, ms); }
                llama_moe_cache_vram_watch_fork(n > 1);
            }
            return ret;
        }
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
