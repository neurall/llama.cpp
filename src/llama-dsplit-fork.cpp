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

#include "llama-impl.h"
#include "llama-model.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
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
    return e && atoi(e) != 0;
}

}  // namespace

void llama_dsplit_mirror_build_fork(const llama_model & model) {
    if (!dsplit_opt_on()) {
        return;
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
