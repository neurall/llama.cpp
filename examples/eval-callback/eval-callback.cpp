#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ROUTER_DUMP=<file>: instead of printing every tensor, append each MoE layer's router input
// (ffn_norm-<il>, or attn_post_norm-<il> in the qwen35 graphs) and router logits (ffn_moe_logits*-<il>) as records: u32 name length, name,
// i64 ne0, i64 ne1, ne0*ne1 f32. Used to measure how well layer L+1's router predicts from layer L.
static FILE * g_dump = nullptr;

static bool router_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    const bool want = t->type == GGML_TYPE_F32 &&
        (strncmp(t->name, "ffn_norm-", 9) == 0 || strncmp(t->name, "attn_post_norm-", 15) == 0 ||
         strncmp(t->name, "ffn_moe_logits", 14) == 0);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1] * t->ne[2] * t->ne[3];
    std::vector<float> buf(ne0 * ne1);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    const uint32_t n = (uint32_t) strlen(t->name);
    fwrite(&n, sizeof(n), 1, g_dump);
    fwrite(t->name, 1, n, g_dump);
    fwrite(&ne0, sizeof(ne0), 1, g_dump);
    fwrite(&ne1, sizeof(ne1), 1, g_dump);
    fwrite(buf.data(), sizeof(float), buf.size(), g_dump);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    // router dump: every token's output, so the last layer runs for all tokens too
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, g_dump != nullptr || i + 1 == tokens.size());
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval;
    params.cb_eval_user_data = &cb_data;
    if (const char * dump = getenv("ROUTER_DUMP")) {
        g_dump = fopen(dump, "wb");
        params.cb_eval = router_dump_cb;
        params.cb_eval_user_data = nullptr;
    }
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    if (g_dump) {
        fclose(g_dump);
    }
    llama_backend_free();

    return 0;
}
