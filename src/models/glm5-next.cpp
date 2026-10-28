#include "models.h"
#include "../llama-dsplit-fork.h"
#include "../llama-moecache-fork.h"

#include <algorithm>
#include "llama-memory-hybrid-idx.h"

// fork (llama-graph.cpp), declared here so that no shared header changes: the index in ds_r of the expert card's range (where it sits in token order
// is ds_plan's choice: the first)
size_t llama_ds_xr_fork(const std::vector<llm_graph_context::ds_range_fork> & r);


// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.

void llama_model_glm5_next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound, false);

    // the MLA cache holds the compressed latent
    hparams.n_embd_head_v_full = hparams.n_lora_kv;

    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    // DSA indexer with k-pool compression
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,        hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,        hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,             hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,             hparams.indexer_kpool);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, hparams.indexer_kpool_select_tail, false);
    GGML_ASSERT(hparams.indexer_kpool > 1 && hparams.indexer_top_k % hparams.indexer_kpool == 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 1);
    ml.get_key_or_arr(LLM_KV_ATTENTION_INDEXER_TYPES, hparams.is_indexer_full_impl, hparams.n_layer(), false);

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult == 4 && "mHC with hc_mult != 4 is not supported");

    switch (hparams.n_layer()) {
        case 45: type = LLM_TYPE_320B_A18B; break; // GLM-5.3-Flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5_next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_mix_dim = (2 + hc)*hc;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        const bool is_nextn = i >= n_layer;
        const int flags = is_nextn ? mtp_flags : trunk_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        if (i < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc*n_embd, hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc*n_embd, hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);
        }

        const int64_t head_dim = hparams.n_embd_head_kda;
        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = head_dim * n_head;

        if (hparams.is_recr(i)) {
            auto conv = [&](llm_tensor tid) {
                ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
                return t ? t : create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner}, flags);
            };
            layer.ssm_q_conv = conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = conv(LLM_TENSOR_SSM_CONV1D_V);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, flags);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd, head_dim}, flags);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {head_dim, d_inner}, flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, flags);

            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {n_head}, flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, flags);

            layer.ssm_g_a    = create_tensor(tn(LLM_TENSOR_SSM_G_A,  "weight", i), {n_embd, head_dim}, flags);
            layer.ssm_g_b    = create_tensor(tn(LLM_TENSOR_SSM_G_B,  "weight", i), {head_dim, d_inner}, flags);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, flags);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, flags);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, flags);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, flags);
            layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, flags);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, flags);

            const int64_t n_indexer_head = hparams.indexer_n_head;
            const int64_t n_embd_indexer = hparams.indexer_head_size;
            const int64_t kpool          = hparams.indexer_kpool;

            const bool full = i >= n_layer || hparams.is_indexer_full(i);
            const int  iflags = flags | (full ? 0 : TENSOR_NOT_REQUIRED);

            layer.indexer_k_norm     = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "weight", i), {n_embd_indexer}, iflags);
            layer.indexer_k_norm_b   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,     "bias",   i), {n_embd_indexer}, iflags);
            layer.indexer_proj       = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,       "weight", i), {n_embd, n_indexer_head}, iflags);
            layer.indexer_attn_k     = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,     "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_attn_q_b   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B,   "weight", i), {q_lora_rank, n_indexer_head * n_embd_indexer}, iflags);
            layer.indexer_kpool_gate = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_GATE, "weight", i), {n_embd, n_embd_indexer}, iflags);
            layer.indexer_kpool_ape  = create_tensor(tn(LLM_TENSOR_INDEXER_KPOOL_APE,  "weight", i), {n_embd_indexer, kpool}, iflags);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, flags);
        } else {
            const int64_t n_ff_exp        = hparams.n_ff_exp(i);
            const int64_t n_expert_shared = hparams.n_expert_shared;

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm5_next::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Causal conv1d over one of Q/K/V
static ggml_tensor * glm5_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                 ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                 int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                 int64_t d_conv, int64_t head_dim, int64_t n_head,
                                 int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head,
                                 int64_t mem_size, int64_t K_rs) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // group s holds the conv window s tokens back.
    // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
    for (int64_t s = 0; s < K_rs; ++s) {
        const int64_t s_idx = std::max<int64_t>(0, n_seq_tokens - s);
        ggml_tensor * conv_x_s = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], s_idx * conv_x->nb[0]);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, conv_x_s,
                ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1)   * ggml_element_size(conv_states_all),
                    n_embd_r_total * ggml_element_size(conv_states_all),
                    ((s * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}


// K-pool indexer inputs
class llama_model_glm5_next::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, sel_mask, new_pool_idxs, new_pool_rep, ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // The scatter mask shape follows n_kv.
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == std::max(mctx->get_n_kpool_new(), 1u);

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64     [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32     [n_pool]         cell caching each pool's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32     [kpool, n_pool]  member cells per pool, n_kv sentinel for the padded pools
    ggml_tensor * pool_mask     = nullptr; // F32/F16 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr; // I32     [kpool - 1, n_tokens]
    ggml_tensor * sel_mask      = nullptr; // F32     [n_sel, 1, 1, n_tokens] 0 for live selection slots, -inf for dead ones
    // n_new is never below 1, see build_inp_kpool
    ggml_tensor * new_pool_idxs = nullptr; // I32     [kpool*n_new]    members of the pools completed this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64     [n_new]          cell to write each new pooled key into

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0;
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
};

llama_model_glm5_next::llm_graph_input_kpool * llama_model_glm5_next::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();
    const uint32_t n_kv   = mctx_idx->get_n_kv();
    // a ubatch that completes no pool still builds one dummy entry, so the graph does not
    // change shape every kpool tokens
    const uint32_t n_new = std::max(mctx_hyb->get_n_kpool_new(), 1u);

    // the fused lightning indexer wants an f16 mask
    const auto type_mask = cparams.fused_lid ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask  = ggml_new_tensor_2d(ctx0, type_mask, n_pool, n_tokens);
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv = n_kv;

    // selection width: the top pools plus the optional tail, also the sparse attention bound
    {
        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        const int64_t n_sel      = kpool*n_top_pool + (hparams.indexer_kpool_select_tail ? kpool - 1 : 0);
        inp->n_sel = (uint32_t) n_sel;

        // the scatter maps the dead slots of this mask to dump rows, the fused mask op reads liveness from the scores instead
        if (!cparams.fused_top_k_mask) {
            inp->sel_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_sel, 1, 1, n_tokens);
            ggml_set_input(inp->sel_mask);
            // set_input_kpool always fills the mask, so it stays allocated in every graph
            ggml_build_forward_expand(gf, inp->sel_mask);
        }
    }

    inp->n_new = n_new;

    inp->new_pool_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, kpool*n_new);
    ggml_set_input(inp->new_pool_idxs);
    // one scatter row per new pool, each a distinct rep row (see kpool_build_state)
    inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new);
    ggml_set_input(inp->new_pool_rep);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

llama_model_glm5_next::graph::hc_pre
llama_model_glm5_next::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    ggml_tensor * post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    post = dsv4_hc_affine(ctx0, post, scale_post, base_post);
    post = ggml_sigmoid(ctx0, post);
    post = ggml_scale(ctx0, post, 2.0f);
    cb(post, "hc_post", il);

    ggml_tensor * comb = nullptr;
    if (cparams.fused_dsv4_hc_comb) {
        comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        comb = dsv4_hc_affine(ctx0, comb, scale_comb, base_comb);
        comb = ggml_reshape_3d(ctx0, comb, hc, hc, nt);
        comb = build_hc_sinkhorn(comb, il);
    }
    cb(comb, "hc_comb", il);

    return { build_hc_pre(x, pre, il), post, comb };
}

ggml_tensor * llama_model_glm5_next::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

// construct the graph helpers without building the trunk, so graph_mtp can reuse them
llama_model_glm5_next::graph::graph(const llama_model & model, const llm_graph_params & params, no_build) :
    llm_build_delta_net_base(params), model(model) {
}

llama_model_glm5_next::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build{}) {

    GGML_ASSERT(hparams.n_layer_nextn > 0 && "GLM5-Next MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "GLM5-Next MTP currently supports a single NextN block");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");

    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm &&
                "GLM5-Next MTP block is missing - load the model with MTP enabled");

    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) n_embd && "GLM5-Next MTP hidden width mismatch");

    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0,
                layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * h_norm = build_norm(h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0), layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // the NextN block has a full indexer, nothing to carry in
    dsa_out dsa = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, nullptr, il);
    cur = dsa.cur;
    cb(cur, "mtp_attn_out", il);

    // narrow to the output tokens before the position-wise FFN; unmasked nextn embeddings need all rows
    if (crop_before_nextn(inp_out_ids)) {
        cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm && "GLM5-Next MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "GLM5-Next MTP: missing both nextn.shared_head_head and output");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

llama_model_glm5_next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_rs    = inp_hyb->get_recr();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda   = hparams.n_head();
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = n_head_kda * head_dim;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * prev_sel = nullptr;

    // ---- fork: dense split (llama-dsplit-fork.cpp) ---------------------------------------------------------------------------------------------------------
    // A prompt pass of one sequence is cut into token ranges, one per card (ds_plan, the dense split kit in llama-graph.cpp: any number of cards, each range
    // sized by the live balance so that every card takes equally long; the expert card, the fastest link, holds the last range). Every card computes the
    // dense part of its range of every layer with its own copy of the dense weights, and its range's residual stream stays on it for the whole pass. Only
    // what depends on other tokens crosses:
    //   KDA layers   range after range: the conv window (the last d_conv-1 projected tokens) and the recurrent state (4.2 MB on GLM-5.3-Flash) of a range are
    //                the start of the next range's; the last range writes both to the memory. The scan itself is therefore sequential over the ranges.
    //   DSA layers   (sparse attention) per range on its card (see dsa_split below), or all tokens on the layer's own card.
    //   MoE          build_moe_ffn runs in range mode (moe_ranges): every card's chain reads all ranges, every range gets the sum of all cards' partial results.
    // Exact (the same operations on the same tokens; only the summation order of the partial results changes). Off for several sequences in one pass, for
    // rollback snapshots (n_rs_seq > 0) and for passes shorter than --moe dense-split-min tokens (default 512); every range has at least 4 x d_conv tokens.
    struct drange { ggml_backend_t b; ggml_backend_dev_t dev; int64_t t0, t1; };
    std::vector<drange> R;
    if (!cparams.embeddings_nextn && n_seqs == 1 && ds_plan(4 * d_conv)) {
        for (const auto & r : ds_r) { R.push_back({ r.b, r.dev, r.t0, r.t1 }); }
    }
    // Everything created for a range is pinned to the range's card: pin() walks the tensors of ctx0 created since the last mark. Views take the card of their
    // source; writes into the memory (cpy into a view of a KV / recurrent-state tensor) run where the memory is.
    ggml_tensor * pin_mark = nullptr;
    auto skip = [&]() {
        ggml_tensor * t = pin_mark ? pin_mark : ggml_get_first_tensor(ctx0);
        if (t == nullptr) { return; }
        for (ggml_tensor * n = ggml_get_next_tensor(ctx0, t); n; n = ggml_get_next_tensor(ctx0, n)) { t = n; }
        pin_mark = t;
    };
    auto pin = [&](ggml_backend_t b) {
        for (ggml_tensor * t = pin_mark ? ggml_get_next_tensor(ctx0, pin_mark) : ggml_get_first_tensor(ctx0); t; t = ggml_get_next_tensor(ctx0, t)) {
            pin_mark = t;
            if (b == nullptr || t->op == GGML_OP_NONE || t->buffer != nullptr) { continue; }
            if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) { continue; }
            if (t->view_src != nullptr && t->view_src->buffer != nullptr) { continue; }
            ggml_backend_sched_set_tensor_backend(sched, t, b);
        }
    };
    auto backend_of = [&](ggml_backend_dev_t dev) -> ggml_backend_t {
        for (int i = 0; sched && i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(b) == dev) { return b; }
        }
        return nullptr;
    };
    // the layer as seen from card dev: every dense weight replaced by its copy there (llama_layer holds tensor pointers only)
    static_assert(sizeof(llama_layer) % sizeof(ggml_tensor *) == 0, "llama_layer is expected to hold tensor pointers only");
    auto layer_on = [&](const llama_layer & l, ggml_backend_dev_t dev) {
        llama_layer m = l;
        ggml_tensor ** p = reinterpret_cast<ggml_tensor **>(&m);
        for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
            if (p[i]) { p[i] = llama_dsplit_mirror_fork(model, p[i], dev); }
        }
        return m;
    };

    // Sparse-attention (DSA) layers per range too: every query attends to at most n_sel selected keys (the indexer's top pools and the tail), so the work is
    // linear in the queries and splits by token range. Per range on its card: the MLA and indexer projections, the selection (indexer scores against the
    // pooled keys, top pools, mask) and the attention over the layer's K cache; once, on the memory's card: the cache writes of both ranges and the pooling
    // of the pools this pass completes. Only for the masked flash-attention path (prompt passes); --moe dense-split-attn=0 runs these layers for all tokens on
    // the layer's card. (nsys, GLM-5.3-Flash 5.5k: the attention layer for all tokens took 125 ms on the x16 card while the x4 card waited 130 ms.)
    static const bool dsa_opt = [] { const char * e = llama_moe_get_opt("dense-split-attn"); return !e || atoi(e) != 0; }();
    const bool dsa_split = dsa_opt && cparams.flash_attn && cparams.fused_lid;

    if (!R.empty()) {
        const size_t NR = R.size();
        std::vector<ggml_tensor *> x(NR), res(NR), post(NR), comb(NR), cr(NR);
        std::vector<ggml_tensor *> prev_sel_r(NR, nullptr);   // the selection mask of the last full-indexer layer, per range
        // The per-range slices of the attention inputs (masks and cache indices) are cut ONCE per pass: a slice made per layer is a new tensor each time and
        // the scheduler copied it to the range's card for every attention layer (kq_mask 34 MB, gather_mask 25 MB, pool_mask 8.5 MB per layer for 3k tokens),
        // as an upload queued behind that card's expert uploads (nsys, GLM-5.3-Flash 5.5k: the x16 card waited 64 ms of a 231 ms attention layer for it).
        std::vector<ggml_tensor *> dsa_kqm(NR), dsa_pm(NR), dsa_gm(NR), dsa_tail(NR), dsa_kim(NR), dsa_kil(NR);
        if (dsa_split) {
            ggml_tensor * kq_mask = inp_attn->get_kq_mask();
            ggml_tensor * k_mla   = inp_attn->get_k_idxs();
            ggml_tensor * k_lid   = inp_kpool->k_idxs;
            const int64_t n_kv_   = mctx_hyb->get_idx()->get_n_kv();
            const int64_t n_pool_ = inp_kpool->pool_cells->ne[0];
            // sel_mask exists only without the fused top-k mask op (upstream #30640): the fused op reads liveness from the scores instead,
            // so with it the ranges get no slice (b13838 dereferenced the missing mask here: segfault on every dense-split GLM pass)
            ggml_tensor * sel_m   = inp_kpool->sel_mask;
            const int64_t n_sel_  = sel_m ? sel_m->ne[0] : 0;
            for (size_t r = 0; r < NR; ++r) {
                const int64_t nr = R[r].t1 - R[r].t0, t0 = R[r].t0;
                dsa_kqm[r]  = ggml_view_2d(ctx0, kq_mask, n_kv_, nr, kq_mask->nb[1], t0 * kq_mask->nb[1]);
                dsa_pm[r]   = ggml_view_2d(ctx0, inp_kpool->pool_mask, n_pool_, nr, inp_kpool->pool_mask->nb[1], t0 * inp_kpool->pool_mask->nb[1]);
                dsa_gm[r]   = sel_m ? ggml_reshape_2d(ctx0, ggml_view_4d(ctx0, sel_m, n_sel_, 1, 1, nr, sel_m->nb[1],
                                    sel_m->nb[2], sel_m->nb[3], t0 * sel_m->nb[3]), n_sel_, nr) : nullptr;
                dsa_tail[r] = ggml_view_2d(ctx0, inp_kpool->tail_idxs, inp_kpool->tail_idxs->ne[0], nr, inp_kpool->tail_idxs->nb[1], t0 * inp_kpool->tail_idxs->nb[1]);
                dsa_kim[r]  = ggml_view_1d(ctx0, k_mla, nr, t0 * ggml_element_size(k_mla));
                dsa_kil[r]  = ggml_view_1d(ctx0, k_lid, nr, t0 * ggml_element_size(k_lid));
            }
        }
        skip();
        for (size_t r = 0; r < NR; ++r) {
            const int64_t nr = R[r].t1 - R[r].t0;
            ggml_tensor * e = ggml_cont(ctx0, ggml_view_2d(ctx0, inp, n_embd, nr, inp->nb[1], R[r].t0 * inp->nb[1]));
            e = ggml_reshape_3d(ctx0, e, n_embd, 1, nr);
            x[r] = ggml_repeat_4d(ctx0, e, n_embd, hc, nr, 1);
            pin(R[r].b);
        }
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            std::vector<llama_layer> L;
            for (size_t r = 0; r < NR; ++r) { L.push_back(layer_on(layer, R[r].dev)); }

            for (size_t r = 0; r < NR; ++r) {
                skip();
                res[r] = x[r];
                ggml_tensor * c = build_hc_pre(x[r], L[r].hc_attn_fn, L[r].hc_attn_scale, L[r].hc_attn_base, &post[r], &comb[r], il);
                cr[r] = build_norm(c, L[r].attn_norm, nullptr, LLM_NORM_RMS, il);
                pin(R[r].b);
            }

            if (hparams.is_recr(il)) {
                const auto * mctx_cur = inp_rs->mctx;
                const auto   kv_head  = mctx_cur->get_head();
                ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
                ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
                skip();
                ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
                ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
                state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);
                skip();   // the memory reads run where the memory is
                const int64_t conv_state_size = (d_conv - 1) * d_inner;
                const int64_t n_embd_r_total  = 3 * conv_state_size;
                const size_t  es_s = ggml_element_size(conv_state_all);
                const size_t  es_m = ggml_element_size(conv_states_all);
                ggml_tensor * tail[3] = { nullptr, nullptr, nullptr };
                for (size_t r = 0; r < NR; ++r) {
                    const int64_t nr   = R[r].t1 - R[r].t0;
                    const bool    last = r + 1 == NR;
                    const llama_layer & Lr = L[r];
                    ggml_tensor * qkv[3];
                    for (int q = 0; q < 3; ++q) {
                        ggml_tensor * W  = q == 0 ? Lr.wq : q == 1 ? Lr.wk : Lr.wv;
                        ggml_tensor * Cw = q == 0 ? Lr.ssm_q_conv : q == 1 ? Lr.ssm_k_conv : Lr.ssm_v_conv;
                        ggml_tensor * prev = tail[q] ? tail[q] : ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
                                (d_conv - 1) * es_s, n_embd_r_total * es_s, q * conv_state_size * es_s);
                        ggml_tensor * xp = ggml_mul_mat(ctx0, W, cr[r]);
                        ggml_tensor * x3 = ggml_reshape_3d(ctx0, xp, d_inner, nr, n_seqs);
                        ggml_tensor * cx = ggml_concat(ctx0, prev, ggml_transpose(ctx0, x3), 0);
                        // the last d_conv-1 tokens: the next range's conv window, or (last range) the conv state written to the memory
                        ggml_tensor * win = ggml_cont(ctx0, ggml_view_3d(ctx0, cx, d_conv - 1, d_inner, n_seqs, cx->nb[1], cx->nb[2], nr * cx->nb[0]));
                        if (last) {
                            ggml_build_forward_expand(gf, ggml_cpy(ctx0, win, ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                                    (d_conv - 1) * es_m, n_embd_r_total * es_m, (kv_head * n_embd_r_total + q * conv_state_size) * es_m)));
                        } else {
                            tail[q] = win;
                        }
                        ggml_tensor * X = ggml_ssm_conv(ctx0, cx, ggml_reshape_2d(ctx0, Cw, d_conv, d_inner));
                        X = ggml_silu(ctx0, ggml_reshape_2d(ctx0, X, d_inner, nr));
                        qkv[q] = ggml_reshape_4d(ctx0, X, head_dim, n_head_kda, nr, n_seqs);
                    }
                    ggml_tensor * f_a = ggml_mul_mat(ctx0, Lr.ssm_f_a, cr[r]);
                    ggml_tensor * g1  = ggml_add(ctx0, ggml_mul_mat(ctx0, Lr.ssm_f_b, f_a), Lr.ssm_dt_b);
                    ggml_tensor * A   = ggml_reshape_3d(ctx0, Lr.ssm_a, 1, n_head_kda, 1);
                    if (hparams.kda_gate_lower_bound > -INFINITY) {
                        g1 = ggml_mul(ctx0, ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, nr), A);
                        g1 = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f)), hparams.kda_gate_lower_bound);
                    } else {
                        g1 = ggml_mul(ctx0, ggml_reshape_3d(ctx0, ggml_softplus(ctx0, g1), head_dim, n_head_kda, nr), A);
                    }
                    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, nr, n_seqs);
                    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, ggml_mul_mat(ctx0, Lr.ssm_beta, cr[r]), 1, n_head_kda, nr, n_seqs));
                    ggml_tensor * Qc = build_gdn_l2_norm(ctx0, qkv[0], 1e-6f);
                    ggml_tensor * Kc = build_gdn_l2_norm(ctx0, qkv[1], 1e-6f);
                    auto os = build_delta_net(Qc, Kc, qkv[2], g1, beta, state, il);
                    ggml_tensor * output    = ggml_cont(ctx0, os.first);
                    ggml_tensor * new_state = ggml_cont(ctx0, os.second);
                    if (last) {
                        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs,
                                ssm_states_all->nb[1], kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
                    } else {
                        state = new_state;   // the next range continues from here
                    }
                    ggml_tensor * g2 = ggml_mul_mat(ctx0, Lr.ssm_g_b, ggml_mul_mat(ctx0, Lr.ssm_g_a, cr[r]));
                    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, nr);
                    ggml_tensor * normed = build_norm(ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, nr), Lr.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
                    ggml_tensor * gated  = ggml_cont_2d(ctx0, ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2)), d_inner, nr);
                    cr[r] = ggml_mul_mat(ctx0, Lr.wo, gated);
                    pin(R[r].b);
                }
            } else if (dsa_split) {
                const auto * mctx_mla = mctx_hyb->get_attn();
                const auto * mctx_lid = mctx_hyb->get_idx();
                const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
                const int64_t kv_lora_rank        = hparams.n_lora_kv;
                const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - hparams.n_rot();
                const float   kq_scale            = 1.0f / sqrtf((float) n_embd_head_k_mla);
                const int64_t n_indexer_head      = hparams.indexer_n_head;
                const int64_t n_embd_indexer      = hparams.indexer_head_size;
                const int64_t kpool               = hparams.indexer_kpool;
                const int64_t n_pool              = inp_kpool->pool_cells->ne[0];
                const int64_t n_new               = inp_kpool->n_new;
                const int64_t n_kv                = mctx_lid->get_n_kv();
                const bool    full_idx            = hparams.is_indexer_full(il);
                ggml_tensor * kq_mask   = inp_attn->get_kq_mask();
                std::vector<ggml_tensor *> qa(NR), iq(NR), iw(NR);
                // 1. per range on its card: projections; the cache writes run where the caches are (both ranges, before anything reads them)
                for (size_t r = 0; r < NR; ++r) {
                    const int64_t nr = R[r].t1 - R[r].t0;
                    const llama_layer & Lr = L[r];
                    skip();
                    ggml_tensor * qr = build_norm(ggml_mul_mat(ctx0, Lr.wq_a, cr[r]), Lr.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                    ggml_tensor * q  = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Lr.wq_b, qr), n_embd_head_qk_nope, n_head, nr);
                    ggml_tensor * kv = build_norm(ggml_mul_mat(ctx0, Lr.wkv_a_mqa, cr[r]), Lr.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
                    kv = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, nr);
                    ggml_tensor * qp = ggml_permute(ctx0, ggml_mul_mat(ctx0, Lr.wk_b, ggml_permute(ctx0, q, 0, 2, 1, 3)), 0, 2, 1, 3);
                    qa[r] = qp;
                    ggml_tensor * packed = nullptr;
                    if (full_idx) {
                        iq[r] = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Lr.indexer_attn_q_b, qr), n_embd_indexer, n_indexer_head, nr);
                        ggml_tensor * ik = build_norm(ggml_mul_mat(ctx0, Lr.indexer_attn_k, cr[r]), Lr.indexer_k_norm, Lr.indexer_k_norm_b, LLM_NORM, il);
                        ggml_tensor * ig = ggml_mul_mat(ctx0, Lr.indexer_kpool_gate, cr[r]);
                        ggml_tensor * pz = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, nr), 0.0f);
                        packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pz, 0), 3*n_embd_indexer, 1, nr);
                        iw[r] = ggml_scale(ctx0, ggml_mul_mat(ctx0, Lr.indexer_proj, cr[r]), 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
                    }
                    pin(R[r].b);
                    if (packed) {
                        ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, dsa_kil[r], il));
                    }
                    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv, dsa_kim[r], il));
                    skip();   // the cache writes and their index views run where the caches are
                }
                // 2. the pools this pass completes, once, where the indexer cache is (they span the ranges)
                ggml_tensor * pooled = nullptr;
                if (full_idx) {
                    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, n_embd_indexer);
                    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
                    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_new);
                    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], 0);
                    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));
                    ggml_tensor * lg = ggml_cont(ctx0, ggml_permute(ctx0, ggml_add(ctx0, pg, layer.indexer_kpool_ape), 1, 0, 2, 3));
                    ggml_tensor * pr = ggml_reshape_3d(ctx0, ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, lg, kpool, n_embd_indexer*n_new)), kpool, n_embd_indexer, n_new);
                    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
                    ggml_tensor * pooled_new = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, pr, pk)), n_embd_indexer, n_new);
                    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
                    pooled = ggml_reshape_3d(ctx0, kpool_cache.gather_pooled(inp_kpool->pool_cells), n_embd_indexer, 1, n_pool);
                    ggml_build_forward_expand(gf, pooled);
                    skip();
                }
                // 3. per range on its card: selection mask and attention over the cache
                const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
                for (size_t r = 0; r < NR; ++r) {
                    const int64_t nr = R[r].t1 - R[r].t0;
                    const llama_layer & Lr = L[r];
                    skip();
                    ggml_tensor * kq_mask_r = dsa_kqm[r];
                    ggml_tensor * sel = nullptr;
                    if (full_idx) {
                        ggml_tensor * pmask = dsa_pm[r];
                        ggml_tensor * score = ggml_lightning_indexer(ctx0, iq[r], pooled, iw[r], pmask);
                        ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool);
                        if (cparams.fused_top_k_mask) {
                            // same as the one-card path (build_kpool_select): the fused op builds the range's mask from its slice of kq_mask
                            ggml_tensor * tail = hparams.indexer_kpool_select_tail ? dsa_tail[r] : nullptr;
                            GGML_ASSERT(kpool*n_top_pool + (tail ? tail->ne[0] : 0) == (int64_t) inp_kpool->n_sel);
                            sel = build_fused_top_k_mask(kq_mask_r, top_k, score, inp_kpool->pool_idxs, tail, nullptr, il);
                        } else {
                            ggml_tensor * sel_score = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, score, 1, n_pool, nr), top_k);
                            ggml_tensor * sel_order = ggml_argsort(ctx0, ggml_reshape_2d(ctx0, sel_score, n_top_pool, nr), GGML_SORT_ORDER_DESC);
                            top_k = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ggml_cast(ctx0, top_k, GGML_TYPE_F32), 1, n_top_pool, nr), sel_order);
                            top_k = ggml_cast(ctx0, ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, n_top_pool, nr)), GGML_TYPE_I32);
                            ggml_tensor * sel_idx = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, inp_kpool->pool_idxs, ggml_reshape_1d(ctx0, top_k, n_top_pool*nr)), kpool*n_top_pool, nr);
                            if (hparams.indexer_kpool_select_tail) {
                                sel_idx = ggml_concat(ctx0, sel_idx, dsa_tail[r], 0);
                            }
                            const int64_t n_sel = sel_idx->ne[0];
                            GGML_ASSERT(inp_kpool->sel_mask->ne[0] == n_sel);
                            ggml_tensor * mask_all = ggml_repeat_4d(ctx0, ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1), -INFINITY), n_kv + n_sel, nr, 1, 1);
                            mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, nr);
                            ggml_tensor * zeros = ggml_repeat_4d(ctx0, ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1), 0.0f), n_sel, nr, 1, 1);
                            zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, nr);
                            ggml_tensor * live  = ggml_exp(ctx0, dsa_gm[r]);
                            ggml_tensor * dump  = ggml_arange(ctx0, (float) n_kv, (float) (n_kv + n_sel), 1.0f);
                            ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
                            idx_f = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
                            sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, ggml_cast(ctx0, idx_f, GGML_TYPE_I32), n_sel, nr, 1));
                            sel = ggml_view_2d(ctx0, sel, n_kv, nr, sel->nb[2], 0);
                            sel = ggml_add(ctx0, sel, kq_mask_r);
                        }
                        prev_sel_r[r] = sel;
                    } else {
                        GGML_ASSERT(prev_sel_r[r] != nullptr && "shared indexer layer must follow a full indexer layer");
                        sel = prev_sel_r[r];
                    }
                    ggml_tensor * mask = ggml_reshape_4d(ctx0, sel, n_kv, nr, 1, 1);
                    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
                    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
                    ggml_tensor * o = build_attn_mha(qa[r], k, v, nullptr, mask, nullptr, Lr.wv_b, inp_kpool->n_sel, kq_scale, il);
                    cr[r] = ggml_mul_mat(ctx0, Lr.wo, o);
                    pin(R[r].b);
                }
            } else {
                // sparse attention, first version: all tokens on the layer's own card
                ggml_backend_t bl = backend_of(model.dev_layer(il));
                skip();
                ggml_tensor * all = cr[0];
                for (size_t r = 1; r < NR; ++r) { all = ggml_concat(ctx0, all, cr[r], 1); }
                pin(bl);
                ggml_tensor * out = build_dsa_layer(all, layer, mctx_hyb, inp_attn, inp_kpool, &prev_sel, il);
                skip();
                for (size_t r = 0; r < NR; ++r) {
                    cr[r] = ggml_cont(ctx0, ggml_view_2d(ctx0, out, n_embd, R[r].t1 - R[r].t0, out->nb[1], R[r].t0 * out->nb[1]));
                }
                pin(bl);
            }

            for (size_t r = 0; r < NR; ++r) {
                skip();
                x[r]   = build_hc_post(cr[r], res[r], post[r], comb[r], il);
                res[r] = x[r];
                ggml_tensor * c = build_hc_pre(x[r], L[r].hc_ffn_fn, L[r].hc_ffn_scale, L[r].hc_ffn_base, &post[r], &comb[r], il);
                cr[r] = build_norm(c, L[r].ffn_norm, nullptr, LLM_NORM_RMS, il);
                pin(R[r].b);
            }

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                for (size_t r = 0; r < NR; ++r) {
                    skip();
                    cr[r] = build_ffn(cr[r], L[r].ffn_up, nullptr, nullptr, L[r].ffn_gate, nullptr, nullptr, L[r].ffn_down, nullptr, nullptr,
                            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    pin(R[r].b);
                }
            } else {
                // router logits per range on its card; the selection runs on the expert card's range for all tokens (288 x tokens floats cross, not the hidden state)
                std::vector<ggml_tensor *> lg(NR);
                for (size_t r = 0; r < NR; ++r) {
                    skip();
                    lg[r] = build_lora_mm(L[r].ffn_gate_inp, cr[r]);
                    if ((llama_expert_gating_func_type) hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                        ggml_prec_set_acc(lg[r], GGML_PREC_F32);
                    }
                    pin(R[r].b);
                }
                skip();
                ggml_tensor * logits = lg[0];
                ggml_tensor * call   = cr[0];
                for (size_t r = 1; r < NR; ++r) {
                    logits = ggml_concat(ctx0, logits, lg[r], 1);
                    call   = ggml_concat(ctx0, call, cr[r], 1);   // the shape of the whole pass; computed only if a path of build_moe_ffn reads it
                }
                pin(R[llama_ds_xr_fork(ds_r)].b);
                moe_ranges.clear();
                for (size_t r = 0; r < NR; ++r) { moe_ranges.push_back({ R[r].b, R[r].t0, R[r].t1, cr[r], nullptr }); }
                build_moe_ffn(call,
                        llama_dsplit_mirror_fork(model, layer.ffn_gate_inp, R[llama_ds_xr_fork(ds_r)].dev),
                        layer.ffn_up_exps,
                        layer.ffn_gate_exps,
                        layer.ffn_down_exps,
                        llama_dsplit_mirror_fork(model, layer.ffn_exp_probs_b, R[llama_ds_xr_fork(ds_r)].dev),
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il, logits);
                std::vector<ggml_tensor *> mo(NR);
                for (size_t r = 0; r < NR; ++r) { mo[r] = moe_ranges[r].out; }
                moe_ranges.clear();
                skip();   // the router's selection follows its input (the logits, on the expert card's range); the expert chains are pinned inside
                for (size_t r = 0; r < NR; ++r) {
                    skip();
                    ggml_tensor * sh = build_ffn(cr[r], L[r].ffn_up_shexp, nullptr, nullptr, L[r].ffn_gate_shexp, nullptr, nullptr, L[r].ffn_down_shexp, nullptr, nullptr,
                            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cr[r] = ggml_add(ctx0, mo[r], sh);
                    pin(R[r].b);
                }
            }

            for (size_t r = 0; r < NR; ++r) {
                skip();
                x[r] = build_cvec(build_hc_post(cr[r], res[r], post[r], comb[r], il), il);
                pin(R[r].b);
                ggml_build_forward_expand(gf, x[r]);
            }
        }
        // the ranges back together for the output head, on the output card
        skip();
        inpL = x[0];
        for (size_t r = 1; r < NR; ++r) { inpL = ggml_concat(ctx0, inpL, x[r], 2); }
        pin(backend_of(model.dev_output()));
    }


    for (int il = 0; il < n_layer && R.empty(); ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        hc_pre attn_hc = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il);
        cur = attn_hc.cur;
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            dsa_out dsa = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, prev_sel, il);
            cur = dsa.cur;
            prev_sel = dsa.sel;
        }

        inpL = build_hc_post(cur, residual, attn_hc.post, attn_hc.comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        hc_pre ffn_hc = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il);
        cur = ffn_hc.cur;
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, ffn_hc.post, ffn_hc.comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // narrow to the output tokens, then collapse the streams
    // Unmasked nextn embeddings need all rows.
    if (crop_before_nextn(inp_out_ids)) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_mean(inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // the post-norm hidden state feeds the draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// KDA layer, g_a/g_b output gate

ggml_tensor * llama_model_glm5_next::graph::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    const int64_t mem_size = mctx_cur->get_size();
    const int64_t K_rs     = (int64_t) cparams.n_rs_seq + 1;

    ggml_tensor * Qcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Kcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Vcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // Decay gate, ssm_a holds -exp(A_log)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    // Match FLA l2 norm
    constexpr float l2_eps = 1e-6f;
    Qcur = build_gdn_l2_norm(ctx0, Qcur, l2_eps);
    Kcur = build_gdn_l2_norm(ctx0, Kcur, l2_eps);

    ggml_tensor * output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
    output = ggml_cont(ctx0, output);
    cb(output, "kda_scan_out", il);

    // output gate, then RMSNorm(o) * Sigmoid(g2)
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_tokens);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// Scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an additive mask

ggml_tensor * llama_model_glm5_next::graph::build_kpool_select(
        ggml_tensor * cur, ggml_tensor * qr, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_kpool * inp_kpool, int il) {

    const auto * mctx_lid = mctx_hyb->get_idx();

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;
    const int64_t n_pool         = inp_kpool->pool_cells->ne[0];
    const int64_t n_new          = inp_kpool->n_new;

    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);
    cb(iq, "indexer_q", il);

    // Per-token key and pool gate scores, cached together
    ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
    ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);
    cb(ig, "indexer_gate", il);

    // Cache rows store key | gate | pooled
    ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_indexer, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_concat(ctx0, ggml_concat(ctx0, ik, ig, 0), pzero, 0);
    packed = ggml_reshape_3d(ctx0, packed, 3*n_embd_indexer, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, n_embd_indexer);
    const int64_t n_kv = mctx_lid->get_n_kv();

    // flash attention requires an f16 mask
    const ggml_type type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // Pool the entries completed by this ubatch. The last one is a dummy when the ubatch completes none.
    ggml_tensor * rows = kpool_cache.gather_key_gate(inp_kpool->new_pool_idxs);
    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_new);

    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_new, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));

    ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_new]
    // fold the pool axis into rows, soft_max maps ne[2] to CUDA gridDim.y which caps at 65535
    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, logits, kpool, n_embd_indexer*n_new));
    probs = ggml_reshape_3d(ctx0, probs, kpool, n_embd_indexer, n_new);

    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    ggml_tensor * pooled_new = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk)); // [1, head_dim, n_new]
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, n_embd_indexer, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    // scatter the fresh pooled keys into their rep rows, then gather all n_pool of them by cell:
    // the older pools come from the rows earlier ubatches wrote
    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    ggml_tensor * pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    pooled = ggml_reshape_3d(ctx0, pooled, n_embd_indexer, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * sel_idx = nullptr;
    {
        ggml_tensor * weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
        cb(weights, "indexer_weights", il);

        ggml_tensor * score = nullptr;
        if (cparams.fused_lid) {
            score = ggml_lightning_indexer(ctx0, iq, pooled, weights, inp_kpool->pool_mask);
            res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        } else {
            ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
            ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

            ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_head]
            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));   // [n_head, n_tokens, n_pool]
            score = ggml_relu(ctx0, kq);
            score = ggml_mul(ctx0, score, weights);
            score = ggml_sum_rows(ctx0, score);                          // [1, n_tokens, n_pool]
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
            score = ggml_add(ctx0, score, inp_kpool->pool_mask);
        }
        cb(score, "indexer_score", il);

        const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
        ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], UNORDERED

        // keep the cells of the live picked pools and the tail, -inf elsewhere
        // all retained cells are in the token's sequence, so the kq_mask is not needed
        if (cparams.fused_top_k_mask) {
            ggml_tensor * tail = hparams.indexer_kpool_select_tail ? inp_kpool->tail_idxs : nullptr;
            GGML_ASSERT(kpool*n_top_pool + (tail ? tail->ne[0] : 0) == (int64_t) inp_kpool->n_sel);

            ggml_tensor * keep = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, type_mask, n_kv), 0.0f);
            ggml_tensor * sel  = ggml_top_k_mask(ctx0, keep, top_k, score, inp_kpool->pool_idxs, tail); // [n_kv, n_tokens]
            res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, sel, il});
            cb(sel, "indexer_sel", il);

            return sel;
        }

        // The selection mask marks the first min(nv, n_top_pool) slots as the visible pools, so order the set by descending score.
        ggml_tensor * sel_score = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
        ggml_tensor * sel_order = ggml_argsort(ctx0,
                ggml_reshape_2d(ctx0, sel_score, n_top_pool, n_tokens), GGML_SORT_ORDER_DESC);
        top_k = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, ggml_cast(ctx0, top_k, GGML_TYPE_F32), 1, n_top_pool, n_tokens), sel_order);
        top_k = ggml_cast(ctx0, ggml_cont(ctx0, ggml_reshape_2d(ctx0, top_k, n_top_pool, n_tokens)), GGML_TYPE_I32);
        cb(top_k, "indexer_top_k", il);

        sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
                ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens));  // [kpool, n_top_pool*n_tokens]
        sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);

        if (hparams.indexer_kpool_select_tail) {
            // Append the incomplete tail with n_kv for missing cells.
            sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
        }
    }
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == (int64_t) inp_kpool->n_sel);

    ggml_build_forward_expand(gf, sel_idx);

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, type_mask, n_kv + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, type_mask, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // Live slots (visible pools, real tail cells) address disjoint cells. Each dead slot writes its own dump row
    // n_kv + slot, so the scatter indices of a token are unique: idx = dump + live*(idx - dump), live = exp(mask).
    GGML_ASSERT(inp_kpool->sel_mask->ne[0] == n_sel && inp_kpool->sel_mask->ne[3] == n_tokens);
    ggml_tensor * live  = ggml_reshape_2d(ctx0, ggml_exp(ctx0, inp_kpool->sel_mask), n_sel, n_tokens);
    ggml_tensor * dump  = ggml_arange(ctx0, (float) n_kv, (float) (n_kv + n_sel), 1.0f);
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);
    sel = ggml_cont(ctx0, sel);
    cb(sel, "indexer_sel", il);

    return sel;
}

// Nope MLA layer with sparse attention over the indexer selection

llama_model_glm5_next::graph::dsa_out
llama_model_glm5_next::graph::build_dsa_layer(
        ggml_tensor * cur, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kpool, ggml_tensor * prev_sel, int il) {

    const auto * mctx_mla = mctx_hyb->get_attn();

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank      = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - hparams.n_rot();
    const float   kq_scale = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5-Next MLA is nope-only");

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens);

    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b so the cache holds only the latent
    ggml_tensor * q_absorbed = ggml_permute(ctx0, q, 0, 2, 1, 3);
    q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_absorbed);
    q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);
    cb(q_absorbed, "q_absorbed", il);

    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * sel = prev_sel;
    if (il >= (int) hparams.n_layer() || hparams.is_indexer_full(il)) { // the NextN block always has a full indexer
        sel = build_kpool_select(cur, qr, layer, mctx_hyb, inp_kpool, il);

        // the selection replaces the kq_mask: split it into the streams of the K view
        const int64_t n_stream = k->ne[3];
        GGML_ASSERT(sel->ne[0] == k->ne[2] && sel->ne[1] % n_stream == 0);
        sel = ggml_reshape_4d(ctx0, sel, sel->ne[0], sel->ne[1]/n_stream, 1, n_stream);
    } else {
        GGML_ASSERT(sel != nullptr && "shared indexer layer must follow a full indexer layer");
    }

    ggml_build_forward_expand(gf, q_absorbed);
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));

    cb(sel, "kq_mask_dsa", il);

    ggml_tensor * out = build_attn_mha(q_absorbed, k, v, nullptr, sel, nullptr, layer.wv_b, inp_kpool->n_sel, kq_scale, il);
    cb(out, "kqv_out", il);

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "attn_out", il);

    return { out, sel };
}
