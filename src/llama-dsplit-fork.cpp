// Fork: dense split of a prompt pass over several GPUs.
//
// Problem. In the multi-GPU prompt split the routed experts of a layer are shared out by id range (every card uploads and computes its own range), but the
// dense part of the layer (hyper-connection mixing, norms, attention or the recurrent KDA block, router, shared expert) runs on the one card that holds the
// layer: on GLM-5.3-Flash at 8k tokens that is 150-165 ms of a 255-285 ms layer, and the other card waits for it (nsys, lanes chart: the x4 card idle 59% of
// the layer). Dense work is per token, so it splits by TOKENS: every card computes the dense part of its own token range of every layer, at the same time.
//
// What that needs:
//   weights  every card needs the dense weights of every layer. They are copied here ONCE at load (GLM-5.3-Flash: 184 MB per layer, 6.8 GB onto the x4
//            card, 1.5 GB onto the x16 card) and never sent again; the routed experts are not copied (they stay shared by id range).
//   ranges   the graph builder cuts the pass into token ranges, one per card (llama_model_glm5_next::graph, --moe dense-split-share), and keeps every range's
//            residual stream on its card for the whole pass. Per layer only what really depends on other tokens crosses: the recurrent state and the conv
//            window of the KDA layers (4.2 MB + 295 KB, range after range), the hidden states that the expert chains of the other cards need, the partial
//            sums that come back, and (first version) the attention layers, which run on the layer's own card for all tokens.
//
// Memory: the copies come out of the expert cache pool of the receiving card (the pool is sized from the memory that is free after the model is loaded).

#include "llama-dsplit-fork.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct dsplit_set {
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> bufs;
    std::map<ggml_backend_dev_t, std::unordered_map<const ggml_tensor *, ggml_tensor *>> mirror;
};

std::mutex                                 g_mu;
std::map<const llama_model *, dsplit_set>  g_sets;

bool dsplit_opt_on() {
    const char * e = llama_moe_get_opt("dense-split");
    return e && (atoi(e) != 0 || strcmp(e, "auto") == 0);
}

bool dsplit_opt_auto() {
    const char * e = llama_moe_get_opt("dense-split");
    return e && strcmp(e, "auto") == 0;
}

}  // namespace

void llama_dsplit_mirror_build_fork(const llama_model & model) {
    {
        // the per-pass choice (--moe dense-split=auto) is managed for any model on two or more GPUs, with or without copies
        std::lock_guard<std::mutex> lk(g_mu);
        g_sets[&model];
    }
    if (!dsplit_opt_on() || (model.arch != LLM_ARCH_GLM5_NEXT && model.arch != LLM_ARCH_QWEN3NEXT)) {
        return;   // a token-range graph exists for these architectures only: no copies for other models (they get the expert split alone)
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (const auto & d : model.devices) {
        if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpus.push_back(d.dev); }
    }
    if (gpus.size() < 2) {
        LLAMA_LOG_WARN("%s: dense split needs two GPUs, %zu found: off\n", __func__, gpus.size());
        return;
    }
    const int n_layer = (int) model.hparams.n_layer();
    dsplit_set S;
    for (ggml_backend_dev_t D : gpus) {
        std::vector<ggml_tensor *> src;
        size_t bytes = 0;
        for (const auto & nt : model.tensors_by_name) {
            const std::string & name = nt.first;
            ggml_tensor * t = nt.second;
            int il = -1;
            if (sscanf(name.c_str(), "blk.%d.", &il) != 1 || il < 0 || il >= n_layer) { continue; }   // trunk layers only (no MTP head)
            if (name.find("_exps") != std::string::npos) { continue; }                               // routed experts stay shared by id range
            if (t == nullptr || t->buffer == nullptr) { continue; }
            ggml_backend_dev_t td = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            if (td == nullptr || td == D || ggml_backend_dev_type(td) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            src.push_back(t);
            bytes += ggml_nbytes(t);
        }
        if (src.empty()) { continue; }
        ggml_init_params ip = { ggml_tensor_overhead() * (src.size() + 8), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        std::vector<ggml_tensor *> dst;
        dst.reserve(src.size());
        for (ggml_tensor * t : src) {
            ggml_tensor * m = ggml_dup_tensor(ctx, t);
            ggml_set_name(m, t->name);
            dst.push_back(m);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(D));
        if (buf == nullptr) {
            LLAMA_LOG_WARN("%s: no room for %.1f MiB of dense weights on %s: dense split off\n", __func__, bytes / 1048576.0, ggml_backend_dev_name(D));
            ggml_free(ctx);
            for (auto & b : S.bufs) { ggml_backend_buffer_free(b.second); ggml_free(b.first); }
            return;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const int64_t t0 = ggml_time_us();
        for (size_t i = 0; i < src.size(); ++i) {
            ggml_backend_tensor_copy(src[i], dst[i]);
            S.mirror[D][src[i]] = dst[i];
        }
        S.bufs.push_back({ ctx, buf });
        LLAMA_LOG_INFO("%s: dense split: %zu dense tensors (%.1f MiB) copied to %s in %.2f s\n", __func__, src.size(), bytes / 1048576.0,
                ggml_backend_dev_name(D), (ggml_time_us() - t0) / 1e6);
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_sets[&model] = std::move(S);
}

namespace {
int dsplit_n_gpus(const llama_model & model) {
    int n = 0;
    for (const auto & d : model.devices) { n += ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU; }
    return n;
}
}  // namespace

void llama_dsplit_mirror_free_fork(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end()) { return; }
    for (auto & b : it->second.bufs) {
        ggml_backend_buffer_free(b.second);
        ggml_free(b.first);
    }
    g_sets.erase(it);
}

bool llama_dsplit_ready_fork(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    return it != g_sets.end() && !it->second.mirror.empty();
}

ggml_tensor * llama_dsplit_mirror_fork(const llama_model & model, ggml_tensor * t, ggml_backend_dev_t dev) {
    if (t == nullptr) { return nullptr; }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end()) { return t; }
    auto mt = it->second.mirror.find(dev);
    if (mt == it->second.mirror.end()) { return t; }
    auto tt = mt->second.find(t);
    return tt == mt->second.end() ? t : tt->second;
}

// ---- --moe dense-split=auto: which cards a prompt pass runs on, per pass size, from measured pass times ------------------------------------------------------
// Whether a second card pays back depends on the pass size: a short pass is bound by the uploads of the experts (every pass uploads all of them), where the
// second card's link adds bandwidth but its share of the experts must fit its link, and every layer pays fixed hand-off costs; a long pass is bound by
// compute, which the split divides. Measured on GLM-5.3-Flash, x16 + x4 (prompt tokens per second, one card / split): 1.8k 403 / 354, 5.5k 589 / 776,
// 8k 566 / 790. The break-even depends on the cards, their links and the model, so it is measured, not assumed:
//   configurations  0 = one card (the default layout: experts and dense on the fastest link), 1 = dense + expert split over the cards
//   buckets         pass sizes in steps of sqrt(2) (256, 362, 512, ... 8192, 11585, ...); a bucket keeps the mean wall time per prompt token of each
//                   configuration (from llama_context::synchronize: all passes of one decode call, attributed to the size of its largest pass)
//   choice          both measured: the faster, and the slower one again every 16 passes of the bucket (to follow the machine); one measured: the other
//                   one (once); none: the nearest bucket with both measured; no data at all: the split first. No switch point is built in: it depends
//                   on the model (expert bytes per layer, quantisation, dense-to-expert ratio, attention) and on the cards, so every size is measured.
//   models          any model on two or more GPUs: config 1 is the expert split, plus the dense split where the model has a token-range graph
//                   (glm5-next); config 0 is one card.
// --moe dense-split=1 / =0 force the split on / off for every pass of --moe dense-split-min tokens or more.
namespace {
struct dsplit_bucket { double us_tok[2] = { 0.0, 0.0 }; int n[2] = { 0, 0 }; int passes = 0; };
std::map<int, dsplit_bucket> g_buckets;
int     g_cur      = -1;   // choice for the graph being built
int     g_built    = -1;   // choice the last graph was built with
int     g_pending  = -1;   // choice of the passes not reported yet (-2: mixed)
int64_t g_pend_max = 0;    // their largest pass

int dsplit_bucket_of(int64_t n) {
    int b = 0;
    double edge = 256.0;
    while ((double) n > edge * 1.2 && b < 40) { edge *= 1.41421356; ++b; }
    return b;
}
}  // namespace

bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    int choice = -1;
    const bool managed = it != g_sets.end() && dsplit_n_gpus(model) >= 2 && (dsplit_opt_auto() || !it->second.mirror.empty());
    if (managed && n_tokens >= 64) {
        if (!dsplit_opt_auto()) {
            choice = 1;   // forced on (the graph builder still applies dense-split-min)
        } else {
            const int b = dsplit_bucket_of(n_tokens);
            dsplit_bucket & B = g_buckets[b];
            if (B.n[0] > 0 && B.n[1] > 0) {
                const int best = B.us_tok[1] < B.us_tok[0] ? 1 : 0;
                choice = (B.passes % 16 == 15) ? 1 - best : best;
            } else if (B.n[0] > 0 || B.n[1] > 0) {
                choice = B.n[0] > 0 ? 1 : 0;
            } else {
                int found = -1, dist = 1 << 20;
                for (const auto & kv : g_buckets) {
                    if (kv.second.n[0] > 0 && kv.second.n[1] > 0 && std::abs(kv.first - b) < dist) {
                        dist = std::abs(kv.first - b);
                        found = kv.second.us_tok[1] < kv.second.us_tok[0] ? 1 : 0;
                    }
                }
                choice = found >= 0 ? found : 1;   // nothing measured anywhere yet: the split first, one card next
            }
            B.passes++;
        }
    }
    g_cur = choice;
    if (choice >= 0) {
        g_pending  = g_pending == -1 || g_pending == choice ? choice : -2;
        g_pend_max = std::max(g_pend_max, n_tokens);
    }
    const bool changed = choice != g_built;
    g_built = choice;
    return changed;
}

int llama_dsplit_cur_fork() {
    return g_cur;
}

void llama_dsplit_reserve_fork(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    g_cur = it != g_sets.end() && dsplit_n_gpus(model) >= 2 && (dsplit_opt_auto() || !it->second.mirror.empty()) ? 1 : -1;
}

void llama_dsplit_report_fork(int64_t n_tokens, int64_t us) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_pending >= 0 && n_tokens > 0 && us > 0 && dsplit_opt_auto()) {
        dsplit_bucket & B = g_buckets[dsplit_bucket_of(g_pend_max)];
        const int c = g_pending;
        const double v = (double) us / (double) n_tokens;
        B.us_tok[c] = B.n[c] == 0 ? v : 0.5 * (B.us_tok[c] + v);   // the newer measurement weighs half
        B.n[c]++;
        static int logged = 0;
        if (logged++ < 64) {
            LLAMA_LOG_WARN("dense split: %lld prompt tokens (largest pass %lld) %s: %.1f tokens/s; this size: one card %s, split %s\n", (long long) n_tokens,
                    (long long) g_pend_max, c ? "split" : "one card", 1e6 / v,
                    B.n[0] ? std::to_string((int) (1e6 / B.us_tok[0])).c_str() : "-", B.n[1] ? std::to_string((int) (1e6 / B.us_tok[1])).c_str() : "-");
        }
    }
    g_pending  = -1;
    g_pend_max = 0;
}

llama_layer llama_dsplit_layer_fork(const llama_model & model, const llama_layer & l, ggml_backend_dev_t dev) {
    static_assert(sizeof(llama_layer) % sizeof(ggml_tensor *) == 0, "llama_layer is expected to hold tensor pointers only");
    llama_layer m = l;
    ggml_tensor ** p = reinterpret_cast<ggml_tensor **>(&m);
    for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
        if (p[i]) { p[i] = llama_dsplit_mirror_fork(model, p[i], dev); }
    }
    return m;
}
