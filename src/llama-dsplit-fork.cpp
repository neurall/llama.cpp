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
#include "llama-splitplan-fork.h"
#include "llama-gpu-probe-fork.h"
#include "llama-moecache-fork.h"
#include "llama-moestate-fork.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// --moe mirror-pool (default on with the compute alias): one card's copies live at the start of its expert pool buffer, in the blocks a prompt pass that
// splits borrows anyway (llama_moe_cache_alias_fork lends them, the graph is placed behind them). Decode never reads a copy (a layer's dense part runs on
// the card that holds the layer), so in decode those blocks hold experts again, and the next pass that splits copies the weights back in (cross-card,
// card to card): MiMo-V2.6-Flash keeps ~9 GB of copies, which a load-time buffer took from the expert cache for the whole session.
struct dsplit_copies {
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> pairs;   // original, copy
    std::vector<size_t> off;       // byte offset of each copy in the region
    size_t              bytes  = 0;
    ggml_context *      ctx    = nullptr;
    bool                placed = false;   // the copies have memory: the pool region, or a buffer of their own
    bool                pooled = false;   // in the pool region: they hold the weights only from an upload until a graph that does not split reuses it
    bool                valid  = false;
};
struct dsplit_set {
    std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> bufs;
    std::map<ggml_backend_dev_t, std::unordered_map<const ggml_tensor *, ggml_tensor *>> mirror;
    std::map<ggml_backend_dev_t, dsplit_copies> copies;
    bool resident = false;   // every weight is in VRAM: no dense split at all (no copies, no split graph reserved, no per-pass decision)
    bool failed = false;   // a card's copies got no memory: the dense split stays off
};

bool dsplit_pool_mode() {
    const char * e = llama_moe_get_opt("mirror-pool");
    const char * a = llama_moe_get_opt("alias");
    const char * p = llama_moe_get_opt("pool");
    return !(e && atoi(e) == 0) && !(a && atoi(a) == 0) && !(p && atoi(p) == 0);
}

// the model's precision rules are keyed by the weight tensor: a copy takes the rule of its original
void dsplit_copy_prec(llama_model & model, const ggml_tensor * src, ggml_tensor * dst) {
    // (llama_prec_policy: "this weight cannot take 4-bit activations"). Without it the copies of such weights ran with 4-bit activations on the other card
    // (Qwen3.8-Flash-Next, the linear-attention weights: perplexity 4360 where one card gives 1.125, with copies that were byte for byte right).
    auto p1 = model.prec_policy.prec_src1.find(src);
    if (p1 != model.prec_policy.prec_src1.end()) { const ggml_prec v = p1->second; model.prec_policy.prec_src1[dst] = v; }
    auto pa = model.prec_policy.prec_acc.find(src);
    if (pa != model.prec_policy.prec_acc.end()) { const ggml_prec v = pa->second; model.prec_policy.prec_acc[dst] = v; }
}

std::mutex                                 g_mu;
std::map<const llama_model *, dsplit_set>  g_sets;

// --moe dense-split=auto|1|0|one. Default auto (user 2026-10-14: the prompt features are on by default, as every release before): on two or more GPUs the dense
// weights are copied to every card once at load (out of the expert pool) and the planner / measured choice picks one card or the split per prompt size.
// dense-split=0 turns it off (no copies, no pool cost, the expert split alone).
const char * dsplit_opt() {
    const char * e = llama_moe_get_opt("dense-split");
    return e ? e : "auto";
}

bool dsplit_opt_on() {
    const char * e = dsplit_opt();
    return atoi(e) != 0 || strcmp(e, "auto") == 0 || strcmp(e, "one") == 0;
}

// --moe dense-split=one (tests): the copies are made, every prompt pass runs on one card (with --moe one-range=1: as one token range on the fastest card)
bool dsplit_opt_one() {
    return strcmp(dsplit_opt(), "one") == 0;
}

bool dsplit_opt_auto() {
    return strcmp(dsplit_opt(), "auto") == 0;
}

}  // namespace

void llama_dsplit_mirror_build_fork(llama_model & model) {
    {
        // the per-pass choice (--moe dense-split=auto) is managed for any model on two or more GPUs, with or without copies
        std::lock_guard<std::mutex> lk(g_mu);
        g_sets[&model];
    }
    if (!dsplit_opt_on() || (model.arch != LLM_ARCH_GLM5_NEXT && model.arch != LLM_ARCH_QWEN3NEXT && model.arch != LLM_ARCH_QWEN4EXP && model.arch != LLM_ARCH_MIMO2 && model.arch != LLM_ARCH_DEEPSEEK4 && model.arch != LLM_ARCH_QWEN35MOE)) {
        return;   // a token-range graph exists for these architectures only: no copies for other models (they get the expert split alone)
    }
    if (dsplit_opt_auto()) {
        // every weight in VRAM: the layer-split pipeline (stock's path, the micro-batches overlap over the cards) is the fast prompt path and the split graph's worst-case
        // reserve (Qwen3.6-35B: 27.7 GiB on a 24 GiB card) fails and drops the pipeline for good: 2750 against 4474 prompt t/s. The copies pay off where experts stream.
        bool all_gpu = !model.tensors_by_name.empty();
        for (const auto & nt : model.tensors_by_name) {
            const ggml_tensor * t = nt.second;
            if (!t || !t->buffer || nt.first == "token_embd.weight") { continue; }   // the input embedding stays in host memory in stock too: a row lookup, no matmul
            ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            if (d == nullptr || ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) { all_gpu = false; break; }
        }
        if (all_gpu) {
            { std::lock_guard<std::mutex> lk(g_mu); g_sets[&model].resident = true; }
            LLAMA_LOG_INFO("%s: all weights are in VRAM: no dense split (the layer-split pipeline is faster); --moe dense-split=1 forces it\n", __func__);
            return;
        }
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (const auto & d : model.devices) {
        if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpus.push_back(d.dev); }
    }
    if (gpus.size() < 2) {
        if (llama_moe_get_opt("dense-split")) {   // a default-on switch is not worth a warning on a one-GPU machine
            LLAMA_LOG_WARN("%s: dense split needs two GPUs, %zu found: off\n", __func__, gpus.size());
        }
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
        if (dsplit_pool_mode()) {
            // the copies wait for this card's expert pool (llama_dsplit_mirror_place_fork): only their places are laid out here, nothing is allocated
            dsplit_copies c;
            c.ctx = ctx;
            ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(D);
            size_t cur = 0;
            for (size_t i = 0; i < src.size(); ++i) {
                cur = GGML_PAD(cur, 256);
                c.pairs.push_back({ src[i], dst[i] });
                c.off.push_back(cur);
                cur += ggml_backend_buft_get_alloc_size(bt, dst[i]);
                S.mirror[D][src[i]] = dst[i];
                dsplit_copy_prec(model, src[i], dst[i]);
            }
            c.bytes = GGML_PAD(cur, (size_t) 2 << 20);
            LLAMA_LOG_WARN("%s: dense split: %zu dense tensors (%.1f MiB) for %s, held in its expert pool only while a prompt pass splits\n", __func__, src.size(),
                    c.bytes / 1048576.0, ggml_backend_dev_name(D));
            S.copies[D] = std::move(c);
            continue;
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
            dsplit_copy_prec(model, src[i], dst[i]);
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
    for (auto & kv : it->second.copies) {
        if (kv.second.ctx) { ggml_free(kv.second.ctx); }   // the pooled copies' headers (a buffer of their own took its context into bufs)
    }
    g_sets.erase(it);
}

namespace {
// Bytes of mirror copies that have to be written before the next split pass (max over the cards: both fill at once, each over its own link). The copies live
// in lent cache blocks (mirror-pool), decode reuses those blocks for experts, so EVERY request that splits starts with this copy. Measured: GLM 3-bit 1k-token
// prompt 419 prompt t/s split against 512 on one card, MiMo 538 against 691 (a difference of 0.44 s and 0.40 s, 2.3-2.7 GB at ~6.6 GB/s over the x4 link).
static std::atomic<double> g_mirror_fill_bytes { 0.0 };
static void dsplit_update_fill(const dsplit_set & S) {
    double mx = 0.0;
    for (const auto & kv : S.copies) { if (kv.second.pooled && !kv.second.valid) { mx = std::max(mx, (double) kv.second.bytes); } }
    g_mirror_fill_bytes.store(mx);
}

// the copies can be used: made, and every card's placed (mirror-pool places them at the first decode)
bool dsplit_copies_ready(const dsplit_set & S) {
    if (S.mirror.empty() || S.failed) { return false; }
    for (const auto & kv : S.copies) {
        if (!kv.second.placed) { return false; }
    }
    return true;
}
}  // namespace

bool llama_dsplit_ready_fork(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    return it != g_sets.end() && dsplit_copies_ready(it->second);
}

void llama_dsplit_mirror_place_fork(const llama_model & model, const std::vector<llama_moe_alias_buf_fork> & pools) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end()) {
        return;
    }
    dsplit_set & S = it->second;
    for (auto & kv : S.copies) {
        dsplit_copies & c = kv.second;
        if (c.placed) {
            continue;
        }
        const llama_moe_alias_buf_fork * pb = nullptr;
        for (const auto & p : pools) {
            if (p.dev == kv.first) { pb = &p; }
        }
        // at most half of what the pool lends: the graph, its scratch and the staging slots of a split pass go behind the copies
        if (pb != nullptr && c.bytes <= pb->capacity / 2) {
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(pb->buf);
            for (size_t i = 0; i < c.pairs.size(); ++i) {
                ggml_tensor * t = c.pairs[i].second;
                t->buffer = pb->buf;
                t->data   = base + c.off[i];
            }
            c.placed = true;
            c.pooled = true;
            c.valid  = false;
            dsplit_update_fill(S);
            llama_moe_cache_alias_mirror_fork(kv.first, c.bytes);   // the pass planner keeps room for them
            LLAMA_LOG_WARN("%s: dense split: the %.1f MiB of copies for %s sit at the start of its expert pool (%.1f GiB lendable): filled when a prompt pass "
                    "splits, expert blocks again in decode\n", __func__, c.bytes / 1048576.0, ggml_backend_dev_name(kv.first), pb->capacity / 1073741824.0);
            continue;
        }
        // no pool on this card, or the copies would take more than half of it: a buffer of their own, filled now (the behaviour before mirror-pool)
        // a card without room to spare (small card, long context): the copies would eat what the graph buffers need later (measured: Qwen3.6 on a simulated 8 GB
        // card, 1.5 GiB of copies, the next graph allocation failed and decode ran at 0.6x of stock). The split is a prefill gain only: leave them out.
        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(kv.first, &mem_free, &mem_total);
        if (mem_free < 2 * c.bytes + ((size_t) 1 << 30)) {
            LLAMA_LOG_WARN("%s: no room to spare for %.1f MiB of dense weights on %s (%.1f MiB free): dense split off\n", __func__, c.bytes / 1048576.0,
                    ggml_backend_dev_name(kv.first), mem_free / 1048576.0);
            S.failed = true;
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_dev_buffer_type(kv.first));
        if (buf == nullptr) {
            LLAMA_LOG_WARN("%s: no room for %.1f MiB of dense weights on %s: dense split off\n", __func__, c.bytes / 1048576.0, ggml_backend_dev_name(kv.first));
            S.failed = true;
            continue;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & pr : c.pairs) {
            ggml_backend_tensor_copy(pr.first, pr.second);
        }
        S.bufs.push_back({ c.ctx, buf });
        c.ctx    = nullptr;
        c.placed = true;
        c.pooled = false;
        c.valid  = true;
        LLAMA_LOG_WARN("%s: dense split: %.1f MiB of copies for %s in a buffer of their own (%s)\n", __func__, c.bytes / 1048576.0, ggml_backend_dev_name(kv.first),
                pb ? "more than half of the pool" : "no expert pool on the card");
    }
}

std::vector<std::pair<ggml_tensor *, ggml_tensor *>> llama_dsplit_mirror_take_upload_fork(const llama_model & model, ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end()) {
        return {};
    }
    auto ct = it->second.copies.find(dev);
    if (ct == it->second.copies.end() || !ct->second.pooled || ct->second.valid) {
        return {};
    }
    ct->second.valid = true;   // the caller fills them before the graph runs
    dsplit_update_fill(it->second);
    return ct->second.pairs;
}

void llama_dsplit_mirror_invalidate_fork(const llama_model & model, ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end()) {
        return;
    }
    auto ct = it->second.copies.find(dev);
    if (ct != it->second.copies.end() && ct->second.pooled) {
        ct->second.valid = false;
        dsplit_update_fill(it->second);
    }
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
struct dsplit_bucket { double us_tok[2] = { 0.0, 0.0 }; int n[2] = { 0, 0 }; int passes = 0; bool prior[2] = { false, false }; };   // prior: loaded from the state file, replaced by the first live measurement
std::map<int, dsplit_bucket> g_buckets;
int     g_cur      = -1;   // choice for the graph being built
int     g_built    = -1;   // choice the last graph was built with
int     g_pending  = -1;   // choice of the passes not reported yet (-2: mixed)
int64_t g_pend_max = 0;    // their largest pass
// the per-prompt decision line (llama_dsplit_report_fork): why the passes not reported yet run as they do, and the cards (names, once)
const char * g_why      = "";
std::string  g_dev_off, g_dev_rest, g_dev_all;
int          g_ngpus = 0, g_pf_choice = -1;   // for GET /moe/stats: cards the split can use, and the last prompt pass's choice (g_cur is reset to -1 by every decode pass)
std::string  g_pf_why;

}  // namespace
uint64_t llama_moe_cache_pp_cells_get_fork();   // llama-moecache-fork.cpp: the cells the planned call ends at
namespace {
// A size bucket is the pass size AND the context it runs at (2026-10-17): the time per token grows with the cells (attention), so a speed measured at 2k cells
// said nothing about the same pass size at 200k, and once stored it decided every later run. Bands: under 16k cells, then each doubling up to 512k.
int dsplit_bucket_of(int64_t n) {
    int b = 0;
    double edge = 256.0;
    while ((double) n > edge * 1.2 && b < 40) { edge *= 1.41421356; ++b; }
    const uint64_t cells = llama_moe_cache_pp_cells_get_fork();
    int band = 0;
    for (uint64_t c = 16384; cells >= c && band < 6; c *= 2) { ++band; }
    return b + 64 * band;
}
}  // namespace

// LLAMA_DS_VERIFY=1 (debug): once, before the first prompt pass, every copy is read back and compared with its original, byte for byte. A copy that differs
// was made before its original had its final content (a weight that is filled or rewritten after the load).
static void dsplit_verify(const dsplit_set & S) {
    size_t n = 0, n_bad = 0;
    std::vector<uint8_t> a, b;
    for (const auto & dm : S.mirror) {
        for (const auto & pr : dm.second) {
            const ggml_tensor * src = pr.first;
            const ggml_tensor * dst = pr.second;
            const size_t nb = ggml_nbytes(src);
            a.resize(nb); b.resize(nb);
            ggml_backend_tensor_get(src, a.data(), 0, nb);
            ggml_backend_tensor_get(dst, b.data(), 0, nb);
            ++n;
            if (memcmp(a.data(), b.data(), nb) != 0) {
                size_t first = 0, cnt = 0;
                for (size_t i = 0; i < nb; ++i) { if (a[i] != b[i]) { if (cnt++ == 0) { first = i; } } }
                if (n_bad++ < 24) {
                    LLAMA_LOG_WARN("dense split: copy of %s (%s, %zu bytes) on %s differs from its original: %zu bytes, the first at %zu\n", src->name,
                            ggml_type_name(src->type), nb, ggml_backend_dev_name(dm.first), cnt, first);
                }
            }
        }
    }
    LLAMA_LOG_WARN("dense split: %zu copies of dense weights checked against their originals, %zu differ\n", n, n_bad);
}

// ---- the split planner: does a second card pay for a prompt pass of n tokens, and with which shares ---------------------------------------------------
// The choice that comes first (the measured choice below only confirms it): the split runs only where this model says the second card pays on THIS model
// and THESE cards, and it hands the split its start shares. Per layer and card, from what is known at run time:
//   compute  c: the measured compute of one prompt token on the offload card, all layers (the pass planner, llama_moe_cache_compute_us_fork). Split per
//            layer by the weights a token reads: the experts' part X = active expert elements over all elements of the layer (Qwen3.8-Flash-Next: 39%;
//            nsys at 3k tokens: 17 of 41 ms), the dense part D the rest. The helper computes at its probed speed relative to the offload card.
//   uploads  every pass streams the experts a card computes and does not hold: f x E_l x (1 - held_l) / link rate, held_l = the share of layer l's
//            experts in that card's pool (llama_moe_cache_resident_frac_fork). They run ahead of the compute, so a card's layer time is the larger of the two.
//   hops     the token ranges' hidden states cross the helper's link: to the card that computes the experts and back (f16, n_embd per token); both cards
//            wait for them (they sit on the layer's critical path).
//   layer    one card: max(c_l x T, upload_l). Split with the helper's token share s and expert share f_l: the larger of the two cards' times.
// The search: s from 0.30 to 0.85, per layer the f_l that gives the shortest layer (0 to 1 in steps of 5%); the split is taken when its sum beats one card
// by 5%. Measured (Qwen3.8-Flash-Next, 3k tokens, x16 + x4, ms per layer): one card 51, split with f = 0 and s = 0.62: 42 (model 39); f = 5%: slower,
// as the model says (the x4's link carries the extra hops); the controller's 21% of the experts on the x4: 48.7.
namespace {
struct dsplit_plan_cache {
    int64_t n_tokens = 0;
    double  c_us     = 0.0;
    double  fill     = -1.0;   // the pending mirror fill the plan was made with: the first pass of a request pays it, the later ones do not
    llama_dsplit_plan_fork plan;
    std::vector<double> f_help;   // per layer
};
std::map<int, dsplit_plan_cache> g_plans;   // per pass-size bucket
std::mutex g_plan_mu;
// The split tuner (under g_plan_mu): the planner gives the start, the measured pass times of a size decide (the model is good to 20-25% on the split;
// a share it gets wrong costs every pass of that size). Per pass-size bucket, one parameter at a time, steps halved when the time gets worse:
//   phase 0  gx, a factor on the plan's per-layer expert shares of the helper (0 = no experts on it, the dense split alone): step 0.5, down to 0.125
//   phase 1  s, the helper's share of the tokens: step 0.08, down to 0.02
//   phase 2  settled: the best measured (gx, s) stays for the rest of the run; at most 12 measurements per size
// Every change rebuilds the graph of the next pass of that size (llama_dsplit_decide_fork says "changed").
struct dsplit_tune {
    int    phase = 0, dir = -1, n = 0;
    double gx = 1.0, s = -1.0, step = 0.5;
    double best_gx = 1.0, best_s = -1.0, best_t = 0.0;
};
std::map<int, dsplit_tune> g_tune;
bool g_tune_changed = false;
}

static std::atomic<double> g_split_calib { 1.0 };   // measured over modelled split time (EMA over measured split passes)
// The calibration is learned from the first measured split pass (it was a 30% step per pass from 1.0: GLM 3-bit at 3.2k tokens ran 1.68x its plan, so the first
// requests kept choosing a split that lost to one card by 18%) and kept per model in the state file, so a restart starts from what was measured.
static std::atomic<int> g_calib_n { 0 };
static std::string g_calib_section;
// Everything learned here is stored under a key of the HARDWARE it was learned on: the cards in model order, each as name, PCI bus id (the physical slot), memory and
// the measured link class (rounded to 5 GB/s). Move a card to another slot, add or remove one, or reorder the devices and the key changes: the old entries stay in the
// file, are not used, and the planner starts again from its model. Same cards in the same slots find their entries again.
static std::string g_hw_key;
static std::string dsplit_pci_file(const std::string & busid, const char * file) {   // one line of /sys/bus/pci/devices/<id>/<file> (Linux), "" elsewhere or when missing
    std::string path = "/sys/bus/pci/devices/" + busid + "/" + file, line;
    for (auto & ch : path) { ch = (char) std::tolower((unsigned char) ch); }
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { return ""; }
    char buf[128] = {0}; if (fgets(buf, sizeof buf, f)) { line = buf; }
    fclose(f);
    while (!line.empty() && (line.back() == '\n' || line.back() == ' ')) { line.pop_back(); }
    return line;
}
static std::string dsplit_hw_key(const llama_model & model) {
    std::string desc;
    for (const auto & d : model.devices) {
        if (ggml_backend_dev_type(d.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        ggml_backend_dev_props p; ggml_backend_dev_get_props(d.dev, &p);
        const std::string bus = p.device_id ? p.device_id : "";
        // vendor + device id and the slot say which card sits where; "generation x width" the link it has there (4x16 = PCIe 4.0 x16): short, stable, no measuring
        std::string ven = dsplit_pci_file(bus, "vendor"), dev = dsplit_pci_file(bus, "device"), spd = dsplit_pci_file(bus, "current_link_speed"), wid = dsplit_pci_file(bus, "current_link_width");
        if (ven.compare(0, 2, "0x") == 0) { ven = ven.substr(2); }
        if (dev.compare(0, 2, "0x") == 0) { dev = dev.substr(2); }
        const double gt = atof(spd.c_str());   // "16.0 GT/s PCIe"
        const int gen = gt >= 31.0 ? 5 : gt >= 15.0 ? 4 : gt >= 7.0 ? 3 : gt >= 4.0 ? 2 : gt > 0.0 ? 1 : 0;
        std::string one = ven.empty() ? std::string(p.name ? p.name : "?") : ven + dev;   // no sysfs (Windows): the name stands in
        one += "@" + (bus.size() > 5 ? bus.substr(5) : bus);   // 0000:0b:00.0 -> 0b:00.0
        one += "/" + (gen ? std::to_string(gen) : std::string("?")) + "x" + (wid.empty() ? std::string("?") : wid);
        desc += one + ";";
    }
    uint64_t h = 1469598103934665603ull; for (char ch : desc) { h = (h ^ (uint8_t) ch) * 1099511628211ull; }
    char k[24]; snprintf(k, sizeof k, "%08x", (unsigned) (h ^ (h >> 32)));
    LLAMA_LOG_WARN("split planner: hardware key %s = %s\n", k, desc.c_str());
    return k;
}
// State keys carry a format number (2026-10-17: calib3, arm2): the calibration was learned against the planner's old model (no attention, a stale compute per token),
// and the arms were bucketed by pass size only. Read under the new model, MiMo's stored 1.714 made every split look 1.7x slower, the planner never chose one, and
// only a measured split pass can change the calibration: stuck. The old entries stay in the file, unused.
static void dsplit_calib_load(const llama_model & model) {
    if (!g_calib_section.empty()) { return; }
    g_calib_section = moe_state_section(model);
    g_hw_key = dsplit_hw_key(model);
    std::string v;
    if (!g_calib_section.empty() && moe_state_get(g_calib_section, "dsplit_calib3@" + g_hw_key, v)) {
        const double c = atof(v.c_str());
        if (c >= 0.5 && c <= 3.0) { g_split_calib.store(c); g_calib_n = 0; LLAMA_LOG_WARN("split planner: calibration %.2f from the state file (a prior: the first measured split pass replaces it)\n", c); }
    }
}
// the measured per-token time of each arm in a size bucket, kept per hardware: "us_one n_one us_split n_split" (counts capped at 3 so new measurements still move it)
static std::set<int> g_bucket_loaded;
static void dsplit_bucket_load(int b) {
    if (g_calib_section.empty() || g_hw_key.empty() || !g_bucket_loaded.insert(b).second) { return; }
    std::string v;
    if (!moe_state_get(g_calib_section, "dsplit_arm2_b" + std::to_string(b) + "@" + g_hw_key, v)) { return; }
    double u0, u1; int n0, n1;
    if (sscanf(v.c_str(), "%lf %d %lf %d", &u0, &n0, &u1, &n1) == 4 && u0 >= 0 && u1 >= 0) {
        dsplit_bucket & B = g_buckets[b];
        if (n0 > 0 && B.n[0] == 0) { B.us_tok[0] = u0; B.n[0] = n0; B.prior[0] = true; }
        if (n1 > 0 && B.n[1] == 0) { B.us_tok[1] = u1; B.n[1] = n1; B.prior[1] = true; }
        LLAMA_LOG_WARN("split planner: size bucket %d from the state file: one card %s, split %s\n", b, n0 > 0 ? (std::to_string((int) (1e6 / u0)) + " tokens/s").c_str() : "-", n1 > 0 ? (std::to_string((int) (1e6 / u1)) + " tokens/s").c_str() : "-");
    }
}
static void dsplit_bucket_save(int b) {
    if (g_calib_section.empty() || g_hw_key.empty()) { return; }
    const dsplit_bucket & B = g_buckets[b];
    char v[96]; snprintf(v, sizeof v, "%.3f %d %.3f %d", B.us_tok[0], std::min(B.n[0], 3), B.us_tok[1], std::min(B.n[1], 3));
    moe_state_set(g_calib_section, { { "dsplit_arm2_b" + std::to_string(b) + "@" + g_hw_key, v } });
}

double   llama_moe_cache_compute_att_us_fork();   // llama-moecache-fork.cpp: the attention part of the compute per token at the planned cells (declared here, no header edit)
uint64_t llama_moe_cache_pp_cells_get_fork();     // the cells the planned call ends at
static void dsplit_plan_compute(const llama_model & model, int64_t T, double c_us, llama_dsplit_plan_fork & out, std::vector<double> & f_help) {
    out = llama_dsplit_plan_fork();
    f_help.clear();
    std::vector<ggml_backend_dev_t> gpus;
    for (const auto & d : model.devices) {
        if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { gpus.push_back(d.dev); }
    }
    if (gpus.size() != 2 || model.dev_offload >= model.devices.size() || c_us <= 0.0 || T <= 0) {
        return;   // two cards only for now (the token ranges are two, see ds_plan)
    }
    ggml_backend_dev_t dm = model.devices[model.dev_offload].dev;
    ggml_backend_dev_t dh = gpus[0] == dm ? gpus[1] : gpus[0];
    const double Bm = llama_moe_cache_link_gbs_fork(dm), Bh = llama_moe_cache_link_gbs_fork(dh);
    if (Bm <= 0.0 || Bh <= 0.0) {
        return;
    }
    double rel = 1.0;   // the helper's compute against the offload card's
    {
        const auto * pm = llama_gpu_probe_find_fork(dm);
        const auto * ph = llama_gpu_probe_find_fork(dh);
        if (pm && ph && pm->compute > 0 && ph->compute > 0) { rel = ph->compute / pm->compute; }
    }
    const int    L      = (int) model.hparams.n_layer();
    const double n_used = (double) std::max<uint32_t>(1, model.hparams.n_expert_used_max());
    const double n_exp  = (double) std::max<uint32_t>(1, model.hparams.n_expert);
    // per layer: the share of the layer's compute in the whole token (by weight elements), its expert share, its expert bytes
    std::vector<double> w(L, 0.0), xf(L, 0.0), E(L, 0.0);
    double w_sum = 0.0, dense_bytes = 0.0;
    for (int il = 0; il < L && il < (int) model.layers.size(); ++il) {
        const llama_layer & ly = model.layers[il];
        const ggml_tensor * const * p = reinterpret_cast<const ggml_tensor * const *>(&ly);
        double dense = 0.0, act = 0.0;
        for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
            const ggml_tensor * t = p[i];
            if (!t || ggml_n_dims(t) < 2) { continue; }
            if (strstr(t->name, "_exps")) {
                act  += (double) ggml_nelements(t) * n_used / n_exp;
                E[il] += (double) ggml_nbytes(t);
            } else if (!strstr(t->name, "nextn")) {
                dense += (double) ggml_nelements(t);
                dense_bytes += (double) ggml_nbytes(t);
            }
        }
        w[il]  = dense + act;
        xf[il] = w[il] > 0.0 ? act / w[il] : 0.0;
        w_sum += w[il];
    }
    if (w_sum <= 0.0) {
        return;
    }
    // Attention over the context (2026-10-17): c_us is the compute per token at the planned cells, base + a x cells (llama-moecache-fork.cpp, pp_c_at). The
    // a x cells part is attention, which grows with the context and is spread over the layers evenly, not by weights; it is dense work, so a split shares it by
    // tokens. MiMo 3-bit at 230k cells: 4.6 ms of a token's compute, three quarters of it attention; the planner without this saw 8 us and an upload-bound pass.
    // A split range attends to the whole cache of its layer, which sits on the layer's own card: the other card's range reads it over the helper's link, every
    // layer, every pass (f16 K and V of every cell the layer sees; a sliding-window layer only its window).
    const double att_tok = std::min(c_us, std::max(0.0, llama_moe_cache_compute_att_us_fork()));
    const double c_base  = c_us - att_tok;
    const double cells   = (double) llama_moe_cache_pp_cells_get_fork();
    // recurrent layers (Qwen3.5/3.6, Qwen3-Next: linear attention) hold no KV and do no attention over the cells: the attention goes to the others
    std::vector<double> kv_b(L, 0.0), att_w(L, 0.0);
    int n_att = 0;
    for (int il = 0; il < L; ++il) { if (!model.hparams.is_recr(il)) { n_att++; } }
    for (int il = 0; il < L; ++il) {
        if (model.hparams.is_recr(il)) { continue; }
        att_w[il] = 1.0 / std::max(1, n_att);
        const double cl_seen = model.hparams.is_swa(il) ? std::min(cells, (double) model.hparams.n_swa + (double) T) : cells;
        // MLA (DeepSeek, GLM-DSA, Kimi): the cache holds the compressed latent and its rope part, one K row per cell and no V
        const double row = model.hparams.is_mla() ? (double) (model.hparams.n_lora_kv + model.hparams.n_rot())
                                                  : (double) (model.hparams.n_embd_k_gqa(il) + model.hparams.n_embd_v_gqa(il));
        kv_b[il] = cl_seen * row * 2.0;
    }
    // bytes per token that cross per layer, there and back (llama_dsplit_hop_mode_fork: f16 both ways, f16 there and f32 back, or f32 both ways)
    const int    hop_mode = llama_dsplit_hop_mode_fork();
    dsx_in in;   // the model of a split layer and its search: llama-splitplan-fork.h
    in.T = T; in.L = L; in.w = w; in.xf = xf; in.E = E; in.kv_b = kv_b; in.att_w = att_w; in.w_sum = w_sum;
    in.c_base = c_base; in.att_tok = att_tok; in.rel = rel; in.Bm = Bm; in.Bh = Bh;
    in.b_in = (double) model.hparams.n_embd * (hop_mode != 0 ? 2.0 : 4.0); in.b_out = (double) model.hparams.n_embd * (hop_mode == 1 ? 2.0 : 4.0);
    in.held_m.resize(L); in.held_h.resize(L);
    for (int il = 0; il < L; ++il) { in.held_m[il] = llama_moe_cache_resident_frac_fork(il, dm); in.held_h[il] = llama_moe_cache_resident_frac_fork(il, dh); }
    const dsx_out so = dsx_plan(in);
    const double t_one = so.t_one;
    double best_split = so.best_split;
    const double best_s = so.best_s;
    const std::vector<double> best_f = so.best_f;
    // the split's estimate is scaled by what earlier split passes measured against their plan (llama_dsplit_report_fork); one card's estimate held to
    // within 4% (Qwen3.8-Flash-Next: 2603 / 2697 ms at 3k, 2038 / 2104 at 1.9k), the split's was 21-26% short (waits the model does not see)
    const double raw_split = best_split;
    best_split *= g_split_calib.load();
    // What the split costs after the prompt: its mirror copies of the dense weights sit in lent cache blocks, and the experts they displaced have to be
    // uploaded again before decode is back at full speed. How much of that a prompt really displaces grows with its length and saturates: the fraction of a
    // layer's experts that n tokens touch, 1 - (1 - k/E)^n (a 1-token prompt displaces nothing worth counting, a long one the whole lent area). The refill
    // is one more upload of the lent bytes over the offload card's link, charged to the split's time (one-off seconds, same unit as the pass).
    // Both are paid once per request, so they are spread over its passes (before 2026-10-17 every pass carried the whole refill: 63 times over in a 255k prompt).
    // The passes of the request so far are about cells / T (the call's cells include the earlier calls of the same prompt).
    const double n_pass = std::max(1.0, cells / std::max<double>(1.0, (double) T));
    best_split += g_mirror_fill_bytes.load() / (std::min(Bm, Bh) * 1e3) / n_pass;   // the copy of the dense weights into the lent blocks, before the first layer (slower link)
    {
        const double touched = 1.0 - std::pow(1.0 - n_used / n_exp, (double) T);
        best_split += touched * dense_bytes / (Bm * 1e3) * llama_moe_live_get("refill_weight") / n_pass;   // live: POST /moe/params
    }
    out.valid      = true;
    out.split      = best_split > 0.0 && best_split < 0.95 * t_one;
    out.share      = best_s;
    out.t_one_ms   = t_one / 1e3;
    out.t_split_ms = best_split / 1e3;
    out.t_raw_ms   = raw_split / 1e3;
    out.t_add_ms   = (best_split - raw_split * g_split_calib.load()) / 1e3;
    f_help = best_f;
}

llama_dsplit_plan_fork llama_dsplit_plan_get_fork(const llama_model & model, int64_t n_tokens) {
    static const bool plan_on = [] { const char * e = llama_moe_get_opt("split-plan"); return !e || atoi(e) != 0; }();   // --moe split-plan=0: no model, the measured choice and the old start shares
    if (!plan_on) {
        return llama_dsplit_plan_fork();
    }
    const double c_us = llama_moe_cache_compute_us_fork();
    std::lock_guard<std::mutex> lk(g_plan_mu);
    dsplit_calib_load(model);
    const int b = dsplit_bucket_of(n_tokens);
    auto & pc = g_plans[b];
    const double fill_now = g_mirror_fill_bytes.load();
    if (pc.n_tokens != n_tokens || pc.c_us != c_us || pc.fill != fill_now) {
        pc.fill     = fill_now;
        pc.n_tokens = n_tokens;
        pc.c_us     = c_us;
        dsplit_plan_compute(model, n_tokens, c_us, pc.plan, pc.f_help);
        static int logged = 0;
        if (pc.plan.valid && logged++ < 32) {
            double fa = 0.0; for (double f : pc.f_help) { fa += f; } fa /= std::max<size_t>(1, pc.f_help.size());
            LLAMA_LOG_WARN("split planner: %lld tokens: one card %.0f ms, split %.0f ms (helper: %.0f%% of the tokens, %.0f%% of the experts on average): %s\n",
                    (long long) n_tokens, pc.plan.t_one_ms, pc.plan.t_split_ms, 100.0 * pc.plan.share, 100.0 * fa, pc.plan.split ? "split" : "one card");
        }
    }
    return pc.plan;
}

double llama_dsplit_plan_share_fork(int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    const int b = dsplit_bucket_of(n_tokens);
    auto tt = g_tune.find(b);
    if (tt != g_tune.end() && tt->second.s >= 0.0) { return tt->second.s; }
    auto it = g_plans.find(b);
    return it == g_plans.end() || !it->second.plan.valid ? -1.0 : it->second.plan.share;
}

// --moe split-experts=auto|0|1: who decides whether the helpers in a split also take experts. auto (default): the split planner and the tuner, per layer and
// pass size. 0: dense split only, the helpers take token ranges of the dense part and no experts (the card with the fastest link computes them all).
// 1: dense split plus experts as before the planner (the link-ratio start and the per-layer controller), whatever the planner says.
int llama_dsplit_experts_mode_fork() {
    static const int mode = [] {
        const char * e = llama_moe_get_opt("split-experts");
        if (!e || strcmp(e, "auto") == 0) { return -1; }
        return atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

// --moe split-f16=0|in|1 (default 0): the precision of what crosses between the cards in a split layer (expert split and dense split alike). Two tensors cross per
// layer, both [n_embd, tokens]: the hidden state, to the card that computes experts on it, and the experts' weighted sum, back to the card that holds the
// tokens. As f16 they are half the bytes on the helper's link, where they sit on the layer's critical path (GLM-5.3-Flash, 8k tokens: 131 MB each way,
// 20 ms + 21 ms of a x4 link per layer as f32).
//   1   f16 both ways: half the bytes
//   in  f16 there, f32 back: the experts' matmuls quantise their input to 8 bits anyway, the sum that goes into the residual stream keeps all its bits
//   0   f32 both ways: bit-for-bit what one card computes, up to the order of the additions
int llama_dsplit_hop_mode_fork() {
    static const int mode = [] {
        const char * e = llama_moe_get_opt("split-f16");
        if (!e) { return 0; }   // f32 by default: f16 hops cost up to +0.4% perplexity where the dense split ran (Next IQ3_S, 10 chunks paired, f16 vs f32 +0.33 +- 0.17%); split-f16=1 trades that for half the link bytes
        if (strcmp(e, "in") == 0) { return 2; }
        return atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

double llama_dsplit_plan_helper_f_fork(int il, int64_t n_tokens) {
    const int mode = llama_dsplit_experts_mode_fork();
    if (mode == 0) { return 0.0; }    // dense only
    if (mode == 1) { return -1.0; }   // the old start and controller
    std::lock_guard<std::mutex> lk(g_plan_mu);
    auto it = g_plans.find(dsplit_bucket_of(n_tokens));
    if (it == g_plans.end() || !it->second.plan.valid || il < 0 || il >= (int) it->second.f_help.size()) {
        return -1.0;
    }
    auto tt = g_tune.find(dsplit_bucket_of(n_tokens));
    const double gx = tt != g_tune.end() ? tt->second.gx : 1.0;
    return std::min(1.0, it->second.f_help[il] * gx);
}

// a split pass whose graph did not fit behind the copies (llama_context::process_ubatch): passes of this many tokens or more run on one card from now on
static int g_veto_tokens = INT32_MAX;
void llama_dsplit_veto_fork(int n_tokens) {
    g_veto_tokens = std::min(g_veto_tokens, n_tokens);
}

// GET /moe/stats: the dense split of the last prompt pass ("prefill": 1 split over all cards, 0 one card, -1 not managed), the cards, the reason
std::string llama_dsplit_state_json_fork() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::string why = g_pf_why;
    for (char & c : why) { if (c == '"' || c == '\\' || c < ' ') { c = ' '; } }
    char b[400];
    snprintf(b, sizeof b, "\"split\":{\"gpus\":%d,\"devs\":\"%s\",\"prefill\":%d,\"prefill_gpus\":%d,\"why\":\"%s\"}", g_ngpus, g_dev_all.c_str(), g_pf_choice,
             g_pf_choice == 1 ? g_ngpus : g_pf_choice == 0 ? 1 : 0, why.c_str());
    return b;
}

bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    {
        static bool verified = false;
        if (!verified && n_tokens >= 64 && it != g_sets.end() && getenv("LLAMA_DS_VERIFY")) { verified = true; dsplit_verify(it->second); }
    }
    if (g_dev_all.empty() && model.dev_offload < model.devices.size()) {
        for (const auto & d : model.devices) {
            if (ggml_backend_dev_type(d.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            const std::string nm = ggml_backend_dev_name(d.dev);
            g_dev_all += (g_dev_all.empty() ? "" : "+") + nm;
            if (d.dev != model.devices[model.dev_offload].dev) { g_dev_rest += (g_dev_rest.empty() ? "" : "+") + nm; }
        }
        g_dev_off = ggml_backend_dev_name(model.devices[model.dev_offload].dev);
    }
    int choice = -1;
    const char * why = "";
    const bool managed = it != g_sets.end() && dsplit_n_gpus(model) >= 2 && ((dsplit_opt_auto() && !it->second.resident) || !it->second.mirror.empty()) &&
                         (it->second.mirror.empty() || dsplit_copies_ready(it->second));   // mirror-pool: not before the copies are placed
    if (managed && n_tokens >= 64) {
        if (dsplit_opt_one()) {
            choice = 0;   // forced to one card (tests)
            why = "forced: dense-split=one";
        } else if (!dsplit_opt_auto()) {
            choice = 1;   // forced on (the graph builder still applies dense-split-min)
            why = "forced: dense-split=1";
        } else {
            const int b = dsplit_bucket_of(n_tokens);
            dsplit_bucket & B = g_buckets[b];
            // the split planner first (see above): sure when it puts the two more than 15% apart; then nothing is tried against it
            const llama_dsplit_plan_fork plan = llama_dsplit_plan_get_fork(model, n_tokens);
            dsplit_bucket_load(b);   // what an earlier run measured on these cards in this size bucket
            const bool plan_sure = plan.valid && std::fabs(plan.t_split_ms - plan.t_one_ms) >= 0.15 * plan.t_one_ms;
            // No trials by default: a pass is never run in the configuration the model expects to be slower just to find out. The planner decides; the split must be
            // predicted clearly ahead (8%) to be used, a close call stays on one card. --moe split-explore=1 brings the old trial-and-error back (the other
            // configuration once per size bucket, the slower one again every 16th pass).
            static const bool explore = [] { const char * e = llama_moe_get_opt("split-explore"); return e && atoi(e) != 0; }();
            if (B.n[0] > 0 && B.n[1] > 0) {
                const int best = B.us_tok[1] < B.us_tok[0] ? 1 : 0;
                choice = (explore && B.passes % 16 == 15) ? 1 - best : best;
                why = choice == best ? "measured: faster at this size" : "measured: the slower one again (every 16th pass of a size)";
            } else if (plan_sure) {
                choice = plan.split ? 1 : 0;
                why = "planner: more than 15% apart";
            } else if (explore && (B.n[0] > 0 || B.n[1] > 0)) {
                choice = B.n[0] > 0 ? 1 : 0;
                why = "measuring: the other one once";
            } else if (plan.valid) {
                choice = (plan.t_split_ms <= (1.0 - llama_moe_live_get("split_min_gain")) * plan.t_one_ms) ? 1 : 0;   // live: POST /moe/params
                why = choice ? "planner: split predicted clearly ahead (split_min_gain)" : "planner: close call, one card (no trial)";
            } else {
                int found = -1, dist = 1 << 20;
                for (const auto & kv : g_buckets) {
                    if (kv.second.n[0] > 0 && kv.second.n[1] > 0 && std::abs(kv.first - b) < dist) {
                        dist = std::abs(kv.first - b);
                        found = kv.second.us_tok[1] < kv.second.us_tok[0] ? 1 : 0;
                    }
                }
                choice = found >= 0 ? found : 0;   // nothing measured and no plan yet: one card first (it measures the compute the planner needs)
                why = found >= 0 ? "measured: the nearest size" : "no plan or measurement yet: one card first";
            }
            B.passes++;
        }
    }
    if (choice == 1 && n_tokens >= g_veto_tokens) {
        choice = 0;   // the split graph of this size did not fit behind the mirror copies once: one card
        why    = "one card: the split pass did not fit behind the copies";
    }
    g_cur = choice;
    g_ngpus = dsplit_n_gpus(model);
    if (choice >= 0) { g_pf_choice = choice; g_pf_why = why; }
    if (choice >= 0) {
        g_pending  = g_pending == -1 || g_pending == choice ? choice : -2;
        g_pend_max = std::max(g_pend_max, n_tokens);
        g_why      = why;
    }
    bool tuned = false;
    {
        std::lock_guard<std::mutex> pk(g_plan_mu);
        tuned = g_tune_changed && choice == 1;
        if (choice == 1) { g_tune_changed = false; }
    }
    const bool changed = choice != g_built || tuned;
    g_built = choice;
    return changed;
}

int llama_dsplit_cur_fork() {
    return g_cur;
}

void llama_dsplit_reserve_fork(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    g_cur = it != g_sets.end() && dsplit_n_gpus(model) >= 2 && ((dsplit_opt_auto() && !it->second.resident) || !it->second.mirror.empty()) &&
            (it->second.mirror.empty() || dsplit_copies_ready(it->second)) ? 1 : -1;
}

// one measured split pass of size bucket b, us per token: the tuner's step (see dsplit_tune)
static void dsplit_tune_step(int b, double v) {
    std::lock_guard<std::mutex> pk(g_plan_mu);
    auto pit = g_plans.find(b);
    if (pit == g_plans.end() || !pit->second.plan.valid) {
        return;
    }
    dsplit_tune & T = g_tune[b];
    if (T.phase >= 2) {
        return;
    }
    if (T.s < 0.0) { T.s = pit->second.plan.share; T.best_s = T.s; }
    T.n++;
    const bool better = T.best_t == 0.0 || v < T.best_t;
    if (better) { T.best_t = v; T.best_gx = T.gx; T.best_s = T.s; }
    bool any_f = false;
    for (double f : pit->second.f_help) { any_f = any_f || f > 0.0; }
    const double gx0 = T.gx, s0 = T.s;
    auto next_phase = [&]() {
        T.gx = T.best_gx; T.s = T.best_s;
        T.phase++; T.dir = -1; T.step = T.phase == 1 ? 0.08 : 0.0;
        if (T.phase == 1) { T.s = std::max(0.10, std::min(0.90, T.best_s - T.step)); }
    };
    if (T.phase == 0 && (!any_f || llama_dsplit_experts_mode_fork() >= 0)) {
        next_phase();   // the plan puts no experts on the helper, or split-experts fixes them: nothing to scale
    } else if (T.phase == 0) {
        if (T.n > 1 && !better) { T.dir = -T.dir; T.step *= 0.5; }
        if (T.step < 0.125) { next_phase(); }
        else { T.gx = std::max(0.0, std::min(4.0, T.best_gx + T.dir * T.step)); }
    } else if (T.phase == 1) {
        if (!better) { T.dir = -T.dir; T.step *= 0.5; }
        if (T.step < 0.02) { next_phase(); }
        else { T.s = std::max(0.10, std::min(0.90, T.best_s + T.dir * T.step)); }
    }
    if (T.n >= 12 && T.phase < 2) { T.gx = T.best_gx; T.s = T.best_s; T.phase = 2; }
    if (T.phase >= 2) {
        LLAMA_LOG_WARN("split tuner: passes of up to %lld tokens settled after %d measurements: helper %.0f%% of the tokens, its experts x%.2f of the plan, %.1f prompt tokens/s\n",
                (long long) pit->second.n_tokens, T.n, 100.0 * T.s, T.gx, 1e6 / T.best_t);
    }
    if (T.gx != gx0 || T.s != s0) { g_tune_changed = true; }
}

void llama_dsplit_report_fork(int64_t n_tokens, int64_t us) {
    std::lock_guard<std::mutex> lk(g_mu);
    static const bool tune_on = [] { const char * e = llama_moe_get_opt("split-tune"); return !e || atoi(e) != 0; }();   // --moe split-tune=0: the plan's shares stay
    if (g_pending == 1 && n_tokens > 0 && us > 0 && tune_on) {
        dsplit_tune_step(dsplit_bucket_of(g_pend_max), (double) us / (double) n_tokens);
    }
    if (g_pending >= 0 && n_tokens > 0 && us > 0 && dsplit_opt_auto()) {
        dsplit_bucket & B = g_buckets[dsplit_bucket_of(g_pend_max)];
        const int c = g_pending;
        const double v = (double) us / (double) n_tokens;
        if (c == 1) {
            // the split planner's calibration: this measurement against the plan it ran with (both per token)
            std::lock_guard<std::mutex> pk(g_plan_mu);
            auto pit = g_plans.find(dsplit_bucket_of(g_pend_max));
            if (pit != g_plans.end() && pit->second.plan.valid && pit->second.plan.t_split_ms > 0.0 && pit->second.n_tokens > 0) {
                // the factor scales the modelled pass only: the one-off copy and refill terms are taken out of the measurement first (they were counted twice before:
                // GLM at 3.2k learned 1.68, applied to 12k where the true factor is about 1.3 it made the planner choose one card and lose 17%)
                const double model_us_tok = pit->second.plan.t_raw_ms * 1e3 / (double) pit->second.n_tokens;
                const double add_us_tok   = pit->second.plan.t_add_ms * 1e3 / (double) pit->second.n_tokens;
                const double r = std::max(0.5, std::min(3.0, std::max(0.0, v - add_us_tok) / std::max(1e-9, model_us_tok)));
                // the first measurement sets the factor, later ones average in half and half
                const double nc = g_calib_n.fetch_add(1) == 0 ? r : 0.5 * g_split_calib.load() + 0.5 * r;
                g_split_calib.store(nc);
                if (!g_calib_section.empty()) { char b2[32]; snprintf(b2, sizeof b2, "%.3f", nc); moe_state_set(g_calib_section, { { "dsplit_calib3@" + g_hw_key, b2 } }); }
                pit->second.c_us = -1.0;   // recompute the plan of this size with the new calibration
            }
        }
        B.us_tok[c] = (B.n[c] == 0 || B.prior[c]) ? v : 0.5 * (B.us_tok[c] + v);   // the newer measurement weighs half; a stored value is only a prior and is replaced
        if (B.prior[c]) { B.prior[c] = false; B.n[c] = 0; }
        B.n[c]++;
        dsplit_bucket_save(dsplit_bucket_of(g_pend_max));
        static int logged = 0;
        if (logged++ < 64) {
            LLAMA_LOG_WARN("dense split: %lld prompt tokens (largest pass %lld) %s: %.1f tokens/s; this size: one card %s, split %s\n", (long long) n_tokens,
                    (long long) g_pend_max, c ? "split" : "one card", 1e6 / v,
                    B.n[0] ? std::to_string((int) (1e6 / B.us_tok[0])).c_str() : "-", B.n[1] ? std::to_string((int) (1e6 / B.us_tok[1])).c_str() : "-");
        }
    }
    if (n_tokens >= 64 && us > 0) {
        // One line per decode call with prompt passes, always (a warning-level line: libllama info lines are hidden at the default -lv 3): which cards the prompt ran on, how it was split and why. fork/run.py records the fields
        // (run-history.csv, runs.db), so a choice that goes against what was measured before shows up (a second card skipped where the split was faster).
        //   mode     one-card: the experts of every layer on the card with the fastest link (the other cards only run the dense part of their own layers);
        //            dense-split / dense+expert-split: the dense part of every layer on all cards by token ranges, the experts on the fastest card only / on
        //            all; expert-split: the experts by link ratio (decided at load); mixed: passes of both kinds in one call
        //   helper_token_share  the other cards' share of the tokens; helper_expert_share  their mean share of a layer's experts (-1: the per-layer controller)
        uint32_t passes = 0, pass = 0;
        int slots = -1;
        if (!llama_moe_cache_pass_plan_fork((uint32_t) n_tokens, passes, pass, slots)) {
            pass   = (uint32_t) (g_pend_max > 0 ? g_pend_max : n_tokens);
            passes = (uint32_t) ((n_tokens + pass - 1) / std::max<uint32_t>(1, pass));
        }
        std::string mode = "one-card", gpus = g_dev_off.empty() ? "?" : g_dev_off, skipped = g_dev_rest.empty() ? "-" : g_dev_rest, by = g_why;
        double tok_share = 0.0, exp_share = 0.0;
        if (g_pending == 1 || g_pending == -2) {
            const int b = dsplit_bucket_of(g_pend_max);
            std::lock_guard<std::mutex> pk(g_plan_mu);
            const auto pit = g_plans.find(b);
            const auto tt  = g_tune.find(b);
            const bool have_plan = pit != g_plans.end() && pit->second.plan.valid;
            const double gx = tt != g_tune.end() ? tt->second.gx : 1.0;
            tok_share = tt != g_tune.end() && tt->second.s > 0.0 ? tt->second.s : have_plan ? pit->second.plan.share : 0.55;
            const int em = llama_dsplit_experts_mode_fork();
            if (em == 0) {
                exp_share = 0.0;
            } else if (em == 1 || !have_plan || pit->second.f_help.empty()) {
                exp_share = -1.0;
            } else {
                double s = 0.0;
                for (double f : pit->second.f_help) { s += std::min(1.0, f * gx); }
                exp_share = s / (double) pit->second.f_help.size();
            }
            mode    = g_pending == -2 ? "mixed" : exp_share == 0.0 ? "dense-split" : "dense+expert-split";
            gpus    = g_dev_all.empty() ? "?" : g_dev_all;
            skipped = "-";
            if (tt != g_tune.end() && tt->second.phase >= 2) { by += ", tuner settled"; } else if (tt != g_tune.end() && tt->second.n > 0) { by += ", tuner measuring"; }
        } else if (g_pending == -1) {
            double sp = 0.0;
            const bool es = llama_moe_cache_expert_split_fork(&sp);
            if (es && !g_dev_rest.empty()) { mode = "expert-split"; gpus = g_dev_all.empty() ? "?" : g_dev_all; skipped = "-"; }
            by = format("dense split not set up (--moe dense-split=0, or fewer than two GPUs, or an architecture without a token-range graph); expert split %s at load (transfer speedup %.2fx)", es ? "reserved" : "not reserved", sp);
        }
        std::string one = "-", spl = "-";
        if (g_pend_max > 0) {
            const auto bit = g_buckets.find(dsplit_bucket_of(g_pend_max));
            if (bit != g_buckets.end()) {
                if (bit->second.n[0]) { one = std::to_string((int) (1e6 / bit->second.us_tok[0])); }
                if (bit->second.n[1]) { spl = std::to_string((int) (1e6 / bit->second.us_tok[1])); }
            }
        }
        LLAMA_LOG_WARN("prefill decision: prompt_tokens=%lld passes=%u pass_tokens=%u staging_slots=%d mode=%s gpus=%s skipped=%s helper_token_share=%.2f "
                "helper_expert_share=%.2f prompt_tps=%.1f one_card_tps=%s split_tps=%s chosen_by=\"%s\"\n", (long long) n_tokens, passes, pass, slots, mode.c_str(),
                gpus.c_str(), skipped.c_str(), tok_share, exp_share, (double) n_tokens * 1e6 / (double) us, one.c_str(), spl.c_str(), by.c_str());
    }
    g_pending  = -1;
    g_pend_max = 0;
    g_why      = "";
}

llama_layer llama_dsplit_layer_fork(const llama_model & model, const llama_layer & l, ggml_backend_dev_t dev) {
    static_assert(sizeof(llama_layer) % sizeof(ggml_tensor *) == 0, "llama_layer is expected to hold tensor pointers only");
    llama_layer m = l;
    ggml_tensor ** p = reinterpret_cast<ggml_tensor **>(&m);
    // LLAMA_DS_ORIG=a,b (debug): weights whose name contains one of these pieces are read from the layer's own card instead of the copy
    static const std::vector<std::string> keep = [] {
        std::vector<std::string> v;
        const char * e = getenv("LLAMA_DS_ORIG");
        for (const char * q = e; q && *q; ) {
            const char * c = strchr(q, ',');
            v.push_back(c ? std::string(q, c - q) : std::string(q));
            q = c ? c + 1 : nullptr;
        }
        return v;
    }();
    for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
        if (!p[i]) { continue; }
        bool orig = false;
        for (const auto & k : keep) { orig = orig || strstr(p[i]->name, k.c_str()) != nullptr; }
        if (!orig) { p[i] = llama_dsplit_mirror_fork(model, p[i], dev); }
    }
    return m;
}

// (after g_cur) the copies' bytes the graph being built keeps in front of it (see llama-dsplit-fork.h)
size_t llama_dsplit_mirror_offset_fork(const llama_model & model, ggml_backend_dev_t dev, bool reserve) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_sets.find(&model);
    if (it == g_sets.end() || (!reserve && g_cur != 1)) {
        return 0;
    }
    auto ct = it->second.copies.find(dev);
    return ct != it->second.copies.end() && ct->second.pooled ? ct->second.bytes : 0;
}
