#include "llama-pred.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int MAXK = 8;                             // max lookahead

struct pred_state {
    int     ahead = 0;
    int     train = 1;
    int     score = 1;
    int     mode = 0, mode_built = -1;      // bit 0 train, bit 1 score
    float   mu    = 0.5f;
    float   cur_mu = -1.0f;
    int     n_used = 0;
    int64_t max_batch = 512;
    std::string file;                       // LLAMA_PRED_FILE: weights loaded at init, saved at free

    std::map<std::pair<int, int>, ggml_tensor *> w; // (il, k) -> predictor
    ggml_tensor * mu_t    = nullptr;
    ggml_tensor * stats_t = nullptr;
    int           n_pairs[MAXK] = {};               // predictors per lookahead

    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    uint64_t n_steps = 0, n_tok = 0, n_tok_win = 0;
    float    last[MAXK] = {};                       // stats at the last report
    double   tot_hits[MAXK] = {};
    uint64_t tot_tok = 0;
};

pred_state * g_pred = nullptr;
bool         g_pred_init = false;

// weights file: magic, count, then per predictor (il, k, type, nbytes, data)
constexpr uint32_t PRED_MAGIC = 0x33525050; // "PPR3"

void pred_save(const pred_state * ps) {
    const std::string tmp = ps->file + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        return;
    }
    const uint32_t n = (uint32_t) ps->w.size();
    fwrite(&PRED_MAGIC, 4, 1, f);
    fwrite(&n, 4, 1, f);
    std::vector<uint8_t> buf;
    for (const auto & [key, t] : ps->w) {
        const int32_t hdr[3] = { key.first, key.second, (int32_t) t->type };
        const uint64_t nb = ggml_nbytes(t);
        buf.resize(nb);
        ggml_backend_tensor_get(t, buf.data(), 0, nb);
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(&nb, sizeof(nb), 1, f);
        fwrite(buf.data(), 1, nb, f);
    }
    fclose(f);
    rename(tmp.c_str(), ps->file.c_str());
}

bool pred_load(pred_state * ps) {
    FILE * f = fopen(ps->file.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t magic = 0, n = 0;
    bool ok = fread(&magic, 4, 1, f) == 1 && fread(&n, 4, 1, f) == 1 && magic == PRED_MAGIC && n == ps->w.size();
    std::vector<uint8_t> buf;
    for (uint32_t i = 0; ok && i < n; ++i) {
        int32_t hdr[3];
        uint64_t nb;
        ok = fread(hdr, sizeof(hdr), 1, f) == 1 && fread(&nb, sizeof(nb), 1, f) == 1;
        auto it = ok ? ps->w.find({ hdr[0], hdr[1] }) : ps->w.end();
        ok = ok && it != ps->w.end() && (int32_t) it->second->type == hdr[2] && nb == ggml_nbytes(it->second);
        if (ok) {
            buf.resize(nb);
            ok = fread(buf.data(), 1, nb, f) == nb;
            if (ok) {
                ggml_backend_tensor_set(it->second, buf.data(), 0, nb);
            }
        }
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_WARN("%s: %s does not match these predictors, starting from the routers\n", __func__, ps->file.c_str());
    }
    return ok;
}

} // namespace

void llama_pred_init(const llama_model & model) {
    if (g_pred_init) {
        return;
    }
    // the fit-params dry run builds contexts on weights with no data: wait for the real model
    for (const auto & l : model.layers) {
        if (l.ffn_gate_inp && !l.ffn_gate_inp->data) {
            return;
        }
    }
    g_pred_init = true;
    const char * a = getenv("LLAMA_PRED_AHEAD");
    const int ahead = a ? std::max(0, std::min(MAXK, atoi(a))) : 0;
    if (ahead == 0) {
        return;
    }
    auto * ps = new pred_state;
    ps->ahead = ahead;
    if (const char * t = getenv("LLAMA_PRED_TRAIN")) { ps->train = atoi(t); }
    if (const char * m = getenv("LLAMA_PRED_MU"))    { ps->mu    = (float) atof(m); }
    if (const char * s = getenv("LLAMA_PRED_SCORE")) { ps->score = atoi(s); }
    if (const char * b = getenv("LLAMA_PRED_MAX_BATCH")) { ps->max_batch = atoll(b); }
    if (const char * f = getenv("LLAMA_PRED_FILE"))  { ps->file = f; }
    // fp16 weights (default): half the memory traffic of the predict + update per token
    const char * f16 = getenv("LLAMA_PRED_F16");
    const ggml_type wtype = f16 && atoi(f16) == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ps->n_used = (int) model.hparams.n_expert_used_max();

    const int n_layer = (int) model.layers.size();
    ggml_backend_buffer_type_t buft0 = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const ggml_tensor * r0 = model.layers[il].ffn_gate_inp;
        if (!r0 || !r0->buffer) {
            continue;
        }
        for (int k = 1; k <= ahead && il + k < n_layer; ++k) {
            const ggml_tensor * r = model.layers[il + k].ffn_gate_inp;
            if (!r || !r->buffer || r->ne[0] != r0->ne[0]) {
                break;
            }
            ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, r->ne[0], r->ne[1]);
            ggml_format_name(w, "pred-%d-%d", il, k);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(r->buffer);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                ggml_free(ctx);
                break;
            }
            buft0 = buft0 ? buft0 : buft;
            ps->ctxs.push_back(ctx);
            ps->bufs.push_back(buf);
            // start as the target layer's router
            std::vector<uint8_t> raw(ggml_nbytes(r));
            ggml_backend_tensor_get(r, raw.data(), 0, raw.size());
            std::vector<float> f(ggml_nelements(r));
            if (r->type == GGML_TYPE_F32) {
                memcpy(f.data(), raw.data(), raw.size());
            } else {
                ggml_get_type_traits(r->type)->to_float(raw.data(), f.data(), (int64_t) f.size());
            }
            if (wtype == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> h(f.size());
                ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
                ggml_backend_tensor_set(w, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
            } else {
                ggml_backend_tensor_set(w, f.data(), 0, f.size()*sizeof(float));
            }
            ps->w[{il, k}] = w;
            ps->n_pairs[k - 1]++;
        }
    }
    if (ps->w.empty()) {
        delete ps;
        return;
    }
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ps->mu_t    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ps->stats_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MAXK);
    ggml_set_name(ps->mu_t, "pred-mu");
    ggml_set_name(ps->stats_t, "pred-stats");
    ps->ctxs.push_back(ctx);
    ps->bufs.push_back(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft0));
    const float zero[MAXK] = {};
    ggml_backend_tensor_set(ps->stats_t, zero, 0, sizeof(zero));
    const bool loaded = !ps->file.empty() && pred_load(ps);
    g_pred = ps;
    LLAMA_LOG_WARN("%s: next-layer router predictors: %zu (lookahead %d, %s), training every %d steps, scoring every %d steps (0: off), "
            "mu %.2f, batches up to %" PRId64 " tokens%s\n", __func__, ps->w.size(), ahead, ggml_type_name(wtype), ps->train, ps->score,
            ps->mu, ps->max_batch, loaded ? ", weights loaded" : "");
}

static void pred_report(pred_state * ps, const char * what, const double * hits, uint64_t n_tok) {
    std::string msg;
    for (int k = 0; k < ps->ahead && n_tok > 0; ++k) {
        const double den = (double) n_tok * ps->n_pairs[k] * ps->n_used;
        msg += format(" L+%d %.1f%%", k + 1, den > 0 ? 100.0*hits[k]/den : 0.0);
    }
    LLAMA_LOG_WARN("pred: %s top-%d overlap:%s (%" PRIu64 " tokens)\n", what, ps->n_used, msg.c_str(), n_tok);
}

void llama_pred_step(int64_t n_tokens) {
    pred_state * ps = g_pred;
    if (!ps || n_tokens > llama_pred_max_batch()) {
        return;
    }
    // the previous decode is done (the caller runs this before the next one): read the counters
    if (ps->score > 0 && ps->n_tok_win >= 256) {
        float s[MAXK];
        ggml_backend_tensor_get(ps->stats_t, s, 0, sizeof(s));
        double win[MAXK];
        for (int k = 0; k < MAXK; ++k) {
            win[k] = s[k] - ps->last[k];
            ps->tot_hits[k] += win[k];
            ps->last[k] = s[k];
        }
        ps->tot_tok += ps->n_tok_win;
        pred_report(ps, "last window", win, ps->n_tok_win);
        ps->n_tok_win = 0;
        // keep the F32 counters small
        const float zero[MAXK] = {};
        ggml_backend_tensor_set(ps->stats_t, zero, 0, sizeof(zero));
        std::fill(std::begin(ps->last), std::end(ps->last), 0.0f);
    }
    ps->n_steps++;
    const bool train = ps->train > 0 && ps->n_steps % ps->train == 0;
    const bool score = ps->score > 0 && ps->n_steps % ps->score == 0;
    ps->mode = (train ? 1 : 0) | (score ? 2 : 0);
    // a batch of n tokens applies the average of their NLMS updates (their sum overshoots on similar tokens);
    // one token (decode) is exact sequential NLMS
    // LLAMA_PRED_BATCH_MU: avg (mu/n, default), sqrt (mu/sqrt(n)), sum (mu)
    static const char * bm = getenv("LLAMA_PRED_BATCH_MU");
    const float div = !bm || strcmp(bm, "avg") == 0 ? (float) n_tokens : strcmp(bm, "sqrt") == 0 ? sqrtf((float) n_tokens) : 1.0f;
    const float mu = ps->mu / div;
    if (train && ps->cur_mu != mu) {
        ggml_backend_tensor_set(ps->mu_t, &mu, 0, sizeof(float));
        ps->cur_mu = mu;
    }
    ps->n_tok += n_tokens;
    if (score) {
        ps->n_tok_win += n_tokens; // the overlap is over the scored tokens
    }
}

void llama_pred_free() {
    pred_state * ps = g_pred;
    if (!ps) {
        return;
    }
    if (ps->score > 0) {
        float s[MAXK];
        ggml_backend_tensor_get(ps->stats_t, s, 0, sizeof(s));
        for (int k = 0; k < MAXK; ++k) {
            ps->tot_hits[k] += s[k] - ps->last[k];
        }
        pred_report(ps, "total", ps->tot_hits, ps->tot_tok + ps->n_tok_win);
    }
    if (!ps->file.empty()) {
        pred_save(ps);
    }
    for (auto * b : ps->bufs) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * c : ps->ctxs) { ggml_free(c); }
    delete ps;
    g_pred = nullptr;
    g_pred_init = false;
}

int64_t llama_pred_max_batch() {
    return g_pred ? g_pred->max_batch : 8;
}

bool llama_pred_train_now() {
    return g_pred && (g_pred->mode & 1);
}

bool llama_pred_score_now() {
    return g_pred && (g_pred->mode & 2);
}

bool llama_pred_graph_reusable() {
    return !g_pred || g_pred->mode == g_pred->mode_built;
}

void llama_pred_graph_built() {
    if (g_pred) {
        g_pred->mode_built = g_pred->mode;
    }
}

int llama_pred_ahead() {
    return g_pred ? g_pred->ahead : 0;
}

ggml_tensor * llama_pred_w(int il, int k) {
    if (!g_pred) {
        return nullptr;
    }
    auto it = g_pred->w.find({il, k});
    return it == g_pred->w.end() ? nullptr : it->second;
}

ggml_tensor * llama_pred_mu() {
    return g_pred ? g_pred->mu_t : nullptr;
}

ggml_tensor * llama_pred_stats() {
    return g_pred ? g_pred->stats_t : nullptr;
}
