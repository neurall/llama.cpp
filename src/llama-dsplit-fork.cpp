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

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
    bool failed = false;   // the copies could not be made at load: the dense split stays off
    // cards whose copies got no memory (a small card, a long context): they take no token range, the other cards still split (2026-10-23: one such card
    // used to turn the split off for all of them)
    std::set<ggml_backend_dev_t> bad;
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

// The probed upload rate of a card, read ONCE: cards are not hot-swapped, so the rate (and with it the card order, the expert card, the planner's and the
// balance's hand-off costs) is fixed by the first valid probe and never re-derived per decision, plan or pass. Own mutex: callers hold g_mu or g_plan_mu.
std::mutex                                  g_link_mu;
std::map<ggml_backend_dev_t, double>        g_link_rate;
void dsplit_link_forget() { std::lock_guard<std::mutex> lk(g_link_mu); g_link_rate.clear(); }   // a changed card or link: the next ask reads the rates again
double dsplit_link_once(ggml_backend_dev_t d) {
    auto & mu = g_link_mu; auto & rate = g_link_rate;
    std::lock_guard<std::mutex> lk(mu);
    auto it = rate.find(d);
    if (it != rate.end()) { return it->second; }
    const double g = llama_moe_cache_link_gbs_fork(d);
    if (g > 0.0) { rate[d] = g; }   // before the probe the rate is 0: not stored, asked again
    return g;
}

// The cards of a split pass (2026-10-23, any number of cards, any links and compute): every GPU whose copies have memory, in range order, the slowest link
// first and the expert card last. The expert card computes the experts of every layer while the helpers hold none (the dense split alone); it is the card
// with the fastest link, since every pass uploads the experts it does not hold (on a tie: the card the one-card passes run on). Nothing here is about two
// cards: three cards on x16, x8 and x4 links get three ranges, the x16 card computes the experts, and the live balance gives each card the tokens that make
// them take equally long (llama-splitplan-fork.h, dsb_target). Under g_mu.
std::vector<ggml_backend_dev_t> g_cards;
std::vector<ggml_backend_dev_t> g_card_order;   // every GPU in range order, fixed at the first update: a measured rate that wobbles must not reorder the cards
void dsplit_cards_update(const llama_model & model, const dsplit_set & S) {
    if (g_card_order.empty()) {
        std::vector<std::pair<double, ggml_backend_dev_t>> v;
        for (const auto & d : model.devices) {
            if (ggml_backend_dev_type(d.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            bool seen = false;
            for (const auto & p : v) { seen = seen || p.second == d.dev; }
            if (!seen) { v.push_back({ dsplit_link_once(d.dev), d.dev }); }
        }
        if (v.size() < 2) { g_cards.clear(); return; }
        double top = 0.0;
        for (const auto & p : v) { top = std::max(top, p.first); }
        if (top <= 0.0) { g_cards.clear(); return; }   // no link rates yet (before the probe): no split
        // rates within 10% of each other count as the same link (two x16 cards measure 25.8 and 26.0 GB/s): their order is the model's. Classes from the
        // fastest down, a new class where a rate is under 90% of its class's first; then the slowest class first (a strict order for the sort).
        std::vector<double> desc;
        for (const auto & p : v) { desc.push_back(p.first); }
        std::sort(desc.begin(), desc.end(), std::greater<double>());
        auto cls = [&](double r) { int c = 0; double head = desc[0]; for (double d : desc) { if (d < 0.9 * head) { ++c; head = d; } if (d == r) { break; } } return c; };
        std::stable_sort(v.begin(), v.end(), [&](const auto & a, const auto & b) { return cls(a.first) > cls(b.first); });
        // the expert card: the fastest link; among the cards within 10% of it, the card the one-card passes run on
        size_t xi = v.size() - 1;
        if (model.dev_offload < model.devices.size()) {
            const ggml_backend_dev_t off = model.devices[model.dev_offload].dev;
            for (size_t i = 0; i < v.size(); ++i) { if (v[i].second == off && v[i].first >= 0.9 * top) { xi = i; } }
        }
        std::rotate(v.begin() + xi, v.begin() + xi + 1, v.end());   // the expert card to the end, the others keep their order
        for (const auto & p : v) { g_card_order.push_back(p.second); }
        std::string txt;
        for (const auto & p : v) { txt += format(" %s %.1f GB/s", ggml_backend_dev_name(p.second), p.first); }
        LLAMA_LOG_WARN("dense split: cards in range order:%s (the last computes the experts)\n", txt.c_str());
    }
    g_cards.clear();
    for (ggml_backend_dev_t d : g_card_order) { if (!S.bad.count(d)) { g_cards.push_back(d); } }
    if (g_cards.size() < 2 || S.bad.count(g_card_order.back())) { g_cards.clear(); }   // fewer than two cards with copies, or the expert card has none: no split
}
}  // namespace

// the cards of a split pass, the expert card last (empty: no split; see g_cards). For the graph builders, which have no model at hand.
void llama_dsplit_cards_fork(std::vector<ggml_backend_dev_t> & out) {
    std::lock_guard<std::mutex> lk(g_mu);
    out = g_cards;
}

// The order in which a graph builder issues its ranges' attention over one layer's KV cache: the ranges on other cards first, the range on the cache's own card
// last. A range on another card reads the cache through a copy, and the scheduler queues that copy on the CACHE's card stream, behind whatever that card has
// queued by then. Issued in token order, the copy for the x16 card stood behind the x4 card's own attention whenever the layer's cache was on the x4 card, so
// the two attentions ran one after the other: Qwen3.8-Flash-Next 60k prompt (2026-10-24), x16 + x4, the x16 card idle 397 ms in front of its 285 ms attention
// split, behind the x4's 182 ms one. Issued first, the copy leaves right behind the K/V writes and both cards attend at the same time. Any order is correct: the
// writes of every range come first, and the ranges' attentions do not read each other. range_devs: each range's card, in token order; cache: a view of the
// layer's cache (its K).
std::vector<size_t> llama_dsplit_attn_order_fork(const std::vector<ggml_backend_dev_t> & range_devs, const ggml_tensor * cache) {
    const ggml_tensor * base = cache && cache->view_src ? cache->view_src : cache;
    const ggml_backend_dev_t cd = base && base->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(base->buffer)) : nullptr;
    std::vector<size_t> order;
    for (size_t r = 0; r < range_devs.size(); ++r) { if (range_devs[r] != cd) { order.push_back(r); } }
    for (size_t r = 0; r < range_devs.size(); ++r) { if (range_devs[r] == cd) { order.push_back(r); } }
    return order;
}

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
// What filling the mirror copies costs before the next split pass (us). The copies live in lent cache blocks (mirror-pool), decode reuses those blocks for
// experts, so EVERY request that splits starts with this copy. Measured: GLM 3-bit 1k-token prompt 419 prompt t/s split against 512 on one card, MiMo 538
// against 691 (a difference of 0.44 s and 0.40 s, 2.3-2.7 GB at ~6.6 GB/s over the x4 link).
// llama_context::moe_mirror_fill issues the copies of every card before it waits (2026-10-23; before, one card after the other with a wait after each:
// Qwen3.8-Flash-Next 13.7k, 2078 MiB on the x4 card in 365 ms, then 1878 MiB on the x16 in 325 ms; GLM-5.3-Flash 4523 MiB in 848 ms, then 4000 MiB in 750 ms).
// A copy crosses the source card's link out and the receiving card's link in, and a link carries both directions at once, so the fill takes as long as the
// busiest link in its busier direction: max over the cards of max(bytes in, bytes out) / its upload rate (with two cards on x16 + x4: the x4 card's larger
// direction alone, not both cards' sum).
static std::atomic<double> g_mirror_fill_us { 0.0 };
static void dsplit_update_fill(const dsplit_set & S) {
    std::map<ggml_backend_dev_t, double> in, out;
    for (const auto & kv : S.copies) {
        if (!kv.second.pooled || kv.second.valid) { continue; }
        for (const auto & pr : kv.second.pairs) {
            const double nb = (double) ggml_nbytes(pr.first);
            in[kv.first] += nb;
            ggml_backend_dev_t ds = pr.first->buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pr.first->buffer)) : nullptr;
            if (ds) { out[ds] += nb; }
        }
    }
    double t = 0.0;
    auto busy = [&](ggml_backend_dev_t d, double b) {
        const double g = dsplit_link_once(d);
        if (g > 0.0) { t = std::max(t, b / (g * 1e3)); }
    };
    for (const auto & kv : in)  { busy(kv.first, std::max(kv.second, out.count(kv.first) ? out[kv.first] : 0.0)); }
    for (const auto & kv : out) { if (!in.count(kv.first)) { busy(kv.first, kv.second); } }
    g_mirror_fill_us.store(t);
}

// the copies can be used: made, and every card's placed or left out (mirror-pool places them at the first decode)
bool dsplit_copies_ready(const dsplit_set & S) {
    if (S.mirror.empty() || S.failed) { return false; }
    for (const auto & kv : S.copies) {
        if (!kv.second.placed && !S.bad.count(kv.first)) { return false; }
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
            LLAMA_LOG_WARN("%s: no room to spare for %.1f MiB of dense weights on %s (%.1f MiB free): %s takes no token range, the other cards split\n", __func__,
                    c.bytes / 1048576.0, ggml_backend_dev_name(kv.first), mem_free / 1048576.0, ggml_backend_dev_name(kv.first));
            S.bad.insert(kv.first);
            continue;
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, ggml_backend_dev_buffer_type(kv.first));
        if (buf == nullptr) {
            LLAMA_LOG_WARN("%s: no room for %.1f MiB of dense weights on %s: it takes no token range, the other cards split\n", __func__, c.bytes / 1048576.0,
                    ggml_backend_dev_name(kv.first));
            S.bad.insert(kv.first);
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
    if (it == g_sets.end() || it->second.bad.count(dev)) { return t; }   // a card left out has copies without memory: never handed out
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
    std::vector<double> s_cards;  // the plan's token share of every card of the split (g_cards order)
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
    // phase 0 (2026-10-23, user's design): K whole MoE layers give their experts to the helper, in g_layer_order; K starts at 0 (the dense split alone: its
    // copies were uploaded once, the helper computes without uploading). K grows 1, 2, 4 ... while the pass gets faster, then bisects between the best K and
    // the first slower one: the helper's share grows until its time reaches the offload card's. The settled K and s are stored (dsplit_tune_save).
    int    K = 0, best_K = 0, k_hi = -1, k_step = 1;
    // phase 2 never stops (2026-10-23, user's design): the two levers keep drifting. A probe moves one lever one small step (lever 0: the helper's share of the
    // tokens, lever 1: the number of MoE layers whose experts the helper takes), runs two passes, and is taken when it is 1% faster per token than the base
    // measured the same way; if not, that lever goes the other way next time and after two misses its step halves. The helper (x4) late means a share that is
    // too big: the probes walk it down until it is not late any more, and back up when the other card is. A step that has shrunk to its floor probes rarely.
    bool   dr_init = false, dr_probe = false;
    int    dr_lever = 0, dr_nb = 0, dr_np = 0, dr_base_K = 0, dr_fail[2] = {0, 0}, dr_dir[2] = {-1, -1};
    double dr_bt = 0.0, dr_pt = 0.0, dr_ds = 0.02, dr_base_s = 0.0;
};
std::map<int, dsplit_tune> g_tune;
bool g_tune_changed = false;
std::vector<int> g_layer_order;   // MoE layers in the order the helper takes them: the layers whose experts its pool holds most first (they cost it the fewest uploads)
std::vector<int> g_layer_rank;    // per layer: its place in g_layer_order, -1 for a layer without experts
}

// measured over modelled split time (EMA over measured split passes). Starts at 1.25, not 1.0: the model leaves out waits it cannot see, and every split
// measured on 2x RTX 3090 (x16 + x4) ran longer than planned: Qwen3.8-Flash-Next 2026-10-23 +7% (3k), +9% (3k), +20% (12k), +27% (12k); earlier 21-26%
// (comment at dsplit_plan_compute). A split the model puts more than 25% ahead is still tried; the first measured split pass replaces the prior.
static std::atomic<double> g_split_calib { 1.25 };
// The calibration is learned from the first measured split pass (it was a 30% step per pass from 1.0: GLM 3-bit at 3.2k tokens ran 1.68x its plan, so the first
// requests kept choosing a split that lost to one card by 18%) and kept per model in the state file, so a restart starts from what was measured.
static std::atomic<int> g_calib_n { 0 };
// Per pass size (2026-10-23): the measured over the modelled time of each size, for the split and for one card, from that size's passes (first sets it, later ones
// average in half). The model misses different things at different sizes (the split at 4096 tokens of GLM-5.3-Flash 3.5-bit ran 1.94x its plan, at 4233
// 0.98x; one card at 3.5k tokens on Qwen3.8-Flash-Next 0.73x, at 13.7k 1.00x), so one global factor fit no size. A size not measured takes the nearest
// measured one, else the prior (split 1.25, one card 1.0). Kept per model, cards and build in the state file. Under g_plan_mu.
static std::map<int, double> g_calib_split_b, g_calib_one_b;
static std::map<int, double> g_one_raw_b;   // the model's one-card time per token of the last plan of each size (us), raw: what a measured one-card pass is set against
static double calib_near(const std::map<int, double> & m, int b, double dflt) {
    if (m.empty()) { return dflt; }
    auto it = m.find(b);
    if (it != m.end()) { return it->second; }
    int best = -1, dist = 1 << 30;
    for (const auto & kv : m) { if (std::abs(kv.first - b) < dist) { dist = std::abs(kv.first - b); best = kv.first; } }
    return best >= 0 ? m.at(best) : dflt;
}
static std::string g_calib_section;
// Everything learned here is stored under a key of the HARDWARE it was learned on: the cards in model order, each as name, PCI bus id (the physical slot), memory and
// the measured link class (rounded to 5 GB/s). Move a card to another slot, add or remove one, or reorder the devices and the key changes: the old entries stay in the
// file, are not used, and the planner starts again from its model. Same cards in the same slots find their entries again.
int llama_dsplit_hop_mode_range_fork();   // below: the dense split's hand-off precision (f16 both ways by default)
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
static std::string dsplit_hw_key(const llama_model & model, bool log = true, bool with_gen = true) {
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
        one += "/" + (!with_gen ? std::string("") : gen ? std::to_string(gen) : std::string("?")) + "x" + (wid.empty() ? std::string("?") : wid);   // with_gen=false (the run-time check): a link drops its generation when idle
        desc += one + ";";
    }
    uint64_t h = 1469598103934665603ull; for (char ch : desc) { h = (h ^ (uint8_t) ch) * 1099511628211ull; }
    char k[24]; snprintf(k, sizeof k, "%08x", (unsigned) (h ^ (h >> 32)));
    if (log) { LLAMA_LOG_WARN("split planner: hardware key %s = %s\n", k, desc.c_str()); }
    return k;
}
// State keys carry a format number (2026-10-17: calib3, arm2): the calibration was learned against the planner's old model (no attention, a stale compute per token),
// and the arms were bucketed by pass size only. Read under the new model, MiMo's stored 1.714 made every split look 1.7x slower, the planner never chose one, and
// only a measured split pass can change the calibration: stuck. The old entries stay in the file, unused.
// Stored again since 2026-10-23 (the ini-fix of 2026-10-22 kept the calibration and the measured arms per session only). Without them every server start
// paid one split pass the planner predicted ahead and the cards then measured slower: Qwen3.8-Flash-Next on 2x RTX 3090 (x16 + x4), second prompt of each
// start, one card -> split 1421 -> 1091 (3k, -23%), 1358 -> 1177 (3k, -13%), 1535 -> 1459 (12k, -5%) prompt t/s. Both are measurements on these cards, inputs.
// Key: the model's section + cards/slots/links + build: a new build, card or slot measures again; older keys stay in the file, unused.
static std::string dsplit_state_key(const std::string & what) {
    const char * bid = llama_moe_get_opt("build-id");
    return what + "@" + g_hw_key + "#" + (bid ? bid : "");
}
static bool dsplit_state_on() { return !g_calib_section.empty() && !g_hw_key.empty() && llama_moe_get_opt("build-id"); }   // no build id (a tool without common's setup): session only
// ---- dense-split factors learned from every request (user 2026-10-25: "any request measures and stores ... per GPU per token times"; "decide dense on or off
// from them for a given prompt length") ------------------------------------------------------------------------------------------------------------------------
// Every prompt pass, one card or split, is timed: wall time from its decision to the synchronize before the next pass (or the end of its call), the copy
// fill and the expert card's waits for expert uploads taken out, per token: the pass's compute per token (the upload is added back per planned pass size
// as max(upload, compute), llama_moe_cache_pass_upload_ms_fork). Kept per context bucket of 8192 cells, because a token costs more the more cells it attends to and the sparse-attention gate
// makes that curve jump (GLM 3.0: 1302 us/token at 24-33k cells, 924 at 33-42k): one-card passes in dsf.one.k<b>, split passes in dsf.split.k<b>. A split
// pass also keeps each card's dense us/token (dsf.dense.<card>.k<b>). Only these go to the state file, per model, cards AND build (user: a new build can be faster
// or slower, it learns its own; "store only what is hard to get, like attention": what a start measures cheaply - link rates, hand-off bytes, the copy fill,
// a card's matmul rate - is measured again at every start and kept in memory). Loaded at the first decision.
// The decision for a pass at context c with R tokens left in its call: split when R x (one(c) - split(c)) beats the copy fill still to pay (0 when the copies
// are valid) by split_min_gain of the one-card time. A side not measured at c takes its nearest measured context; a side never measured comes from the other
// side and the cards' dense factors: with equal dense time on every card a token's dense part costs 1 / sum_i 1 / (d_i + hop_i) instead of d_x, so the
// split saves sigma = d_x - 1 / sum_i 1 / (d_i + hop_i) per token. Nothing measured yet (a new build's first request): the planner as before.
// ponytail: the nearest context is taken as it is (no attention correction between buckets), one value per bucket whatever the pass size.
// a measured value: the median of its last DSF_KEEP samples (user 2026-10-25: "us/token are noisy as well": one slow pass - cold caches, a clock dip, a
// background job - cannot move a median of five), n = samples seen
static constexpr size_t DSF_KEEP = 5;
struct dsf_pt {
    std::vector<double> s;
    int n = 0;
    double us = 0.0;   // the median of s
};
static std::mutex g_dsf_mu;   // innermost lock (taken under g_mu and g_plan_mu, never the other way)
static std::map<int, dsf_pt> g_dsf_one, g_dsf_split;
static std::map<std::string, std::map<int, dsf_pt>> g_dsf_dense;
static std::map<std::string, dsf_pt> g_dsf_hop;
static dsf_pt g_dsf_fill;
static bool g_dsf_loaded = false;
static constexpr int64_t DSF_CELLS = 8192;
struct dsf_pass { int choice = -1; int64_t T = 0, pos0 = -1, t0 = 0; double fill_ms = 0.0; ggml_backend_dev_t xdev = nullptr; double U_us = -1.0; };
double llama_moe_cache_pass_upload_ms_fork(uint32_t n_tokens, uint32_t n_used, uint32_t n_expert);   // llama-moecache-fork.cpp: a pass's expert upload (ms)
static dsf_pass g_dsf_ps;                // the prompt pass decided last (open until the synchronize after it)
static int64_t  g_dsf_next_pos0 = -1;    // the first position of the pass about to be decided (llama_context)
static void dsf_med(dsf_pt & p) {
    std::vector<double> t = p.s;
    std::sort(t.begin(), t.end());
    p.us = t.empty() ? 0.0 : (t.size() % 2 ? t[t.size() / 2] : 0.5 * (t[t.size() / 2 - 1] + t[t.size() / 2]));
}
static void dsf_upd(dsf_pt & p, double v) {
    p.s.push_back(v);
    if (p.s.size() > DSF_KEEP) { p.s.erase(p.s.begin()); }
    p.n = std::min(p.n + 1, 1000000);
    dsf_med(p);
}
static std::string dsf_val(const dsf_pt & p) {   // "n v1 v2 ...": samples seen, then the kept samples (oldest first)
    std::string o = std::to_string(p.n);
    for (double v : p.s) { char b[32]; snprintf(b, sizeof b, " %.3f", v); o += b; }
    return o;
}
static bool dsf_parse(const std::string & v, dsf_pt & p) {
    const char * c = v.c_str();
    char * e = nullptr;
    p.n = (int) strtol(c, &e, 10);
    if (e == c) { return false; }
    for (c = e; ; c = e) { const double x = strtod(c, &e); if (e == c) { break; } p.s.push_back(x); }
    if (p.s.size() > DSF_KEEP) { p.s.erase(p.s.begin(), p.s.end() - DSF_KEEP); }
    dsf_med(p);
    return !p.s.empty();
}
static void dsf_save(const std::vector<std::pair<std::string, std::string>> & kv) {   // under g_dsf_mu
    if (!dsplit_state_on() || kv.empty()) { return; }
    std::vector<std::pair<std::string, std::string>> o;
    for (const auto & e : kv) { o.push_back({ dsplit_state_key(e.first), e.second }); }
    moe_state_set(g_calib_section, o);
}
// the samples are kept live in memory; the state file is written once at the end of a call (user 2026-10-25: "observe the last per-token times live, store
// them at the request's end, if not stored already"), only the buckets that got a sample since
static std::set<std::string> g_dsf_dirty;                  // keys (dsf.one.k3, dsf.dense.CUDA0.k3, ...) with samples not written yet
static std::map<std::string, const dsf_pt *> g_dsf_ref;    // the value behind each key (std::map nodes do not move)
static void dsf_mark(const std::string & key, const dsf_pt & p) { g_dsf_dirty.insert(key); g_dsf_ref[key] = &p; }
static void dsf_flush() {   // under g_dsf_mu
    if (g_dsf_dirty.empty()) { return; }
    std::vector<std::pair<std::string, std::string>> kv;
    for (const auto & k : g_dsf_dirty) { auto it = g_dsf_ref.find(k); if (it != g_dsf_ref.end() && it->second) { kv.push_back({ k, dsf_val(*it->second) }); } }
    g_dsf_dirty.clear();
    dsf_save(kv);
}
static void dsf_load() {   // under g_dsf_mu: this build's factors from the state file, once
    if (g_dsf_loaded || !dsplit_state_on()) { return; }
    g_dsf_loaded = true;
    std::vector<std::pair<std::string, std::string>> kv;
    if (!moe_state_section_kv(g_calib_section, kv)) { return; }
    const std::string suf = dsplit_state_key("");
    auto kidx = [](const std::string & w) { const size_t q = w.rfind(".k"); return q == std::string::npos ? -1 : atoi(w.c_str() + q + 2); };
    for (const auto & e : kv) {
        const std::string & k = e.first;
        if (k.rfind("dsf.", 0) != 0 || k.size() <= suf.size() + 4 || k.compare(k.size() - suf.size(), suf.size(), suf) != 0) { continue; }
        const std::string w = k.substr(4, k.size() - suf.size() - 4);   // one.k3, split.k3, dense.CUDA0.k3, hop.CUDA0, fill
        dsf_pt pt;
        if (!dsf_parse(e.second, pt)) { continue; }
        if (w.rfind("one.k", 0) == 0) { g_dsf_one[kidx(w)] = pt; }
        else if (w.rfind("split.k", 0) == 0) { g_dsf_split[kidx(w)] = pt; }
        else if (w.rfind("dense.", 0) == 0 && w.rfind(".k") != std::string::npos) { g_dsf_dense[w.substr(6, w.rfind(".k") - 6)][kidx(w)] = pt; }
        else if (w.rfind("hop.", 0) == 0) { g_dsf_hop[w.substr(4)] = pt; }
        else if (w == "fill") { g_dsf_fill = pt; }
    }
    LLAMA_LOG_WARN("split factors: %zu one-card and %zu split contexts, dense curves of %zu cards, copy fill %.0f ms (%d prompts) from the state file (this build)\n",
            g_dsf_one.size(), g_dsf_split.size(), g_dsf_dense.size(), g_dsf_fill.us, g_dsf_fill.n);
}

static void dsplit_calib_load(const llama_model & model) {
    if (!g_calib_section.empty()) { return; }
    g_calib_section = moe_state_section(model);
    g_hw_key = dsplit_hw_key(model);
    std::string v;
    if (dsplit_state_on()) {   // every size's stored factors (this model, cards and build)
        std::vector<std::pair<std::string, std::string>> kv;
        const std::string suf = dsplit_state_key("");
        if (moe_state_section_kv(g_calib_section, kv)) {
            for (const auto & p : kv) {
                const std::string & k = p.first;
                if (k.size() <= suf.size() || k.compare(k.size() - suf.size(), suf.size(), suf) != 0) { continue; }
                const double c = atof(p.second.c_str());
                if (c < 0.3 || c > 3.0) { continue; }
                if (k.rfind("dsplit_calib5_b", 0) == 0) { g_calib_split_b[atoi(k.c_str() + 15)] = c; }
                if (k.rfind("dsplit_calib1_b", 0) == 0) { g_calib_one_b[atoi(k.c_str() + 15)] = c; }
            }
            if (!g_calib_split_b.empty() || !g_calib_one_b.empty()) {
                LLAMA_LOG_WARN("split planner: %zu split and %zu one-card size factors from the state file\n", g_calib_split_b.size(), g_calib_one_b.size());
            }
        }
    }
    if (dsplit_state_on() && moe_state_get(g_calib_section, dsplit_state_key("dsplit_calib4"), v)) {
        const double c = atof(v.c_str());
        if (c >= 0.5 && c <= 3.0) { g_split_calib.store(c); g_calib_n = 0; LLAMA_LOG_WARN("split planner: calibration %.2f from the state file (a prior: the first measured split pass replaces it)\n", c); }
    }
}
static void dsplit_calib_b_save(const char * what, int b, double c) {   // a size's factor (dsplit_calib5_b: split, dsplit_calib1_b: one card)
    if (!dsplit_state_on()) { return; }
    char v[32]; snprintf(v, sizeof v, "%.4f", c);
    moe_state_set(g_calib_section, { { dsplit_state_key(std::string(what) + std::to_string(b)), v } });
}
static void dsplit_calib_save(double c) {
    if (!dsplit_state_on()) { return; }
    char v[32]; snprintf(v, sizeof v, "%.3f", c);
    moe_state_set(g_calib_section, { { dsplit_state_key("dsplit_calib4"), v } });
}
// the measured per-token time of each arm in a size bucket, kept per hardware and build: "us_one n_one us_split n_split" (counts capped at 3 so new measurements still move it)
static std::set<int> g_bucket_loaded;
static void dsplit_bucket_load(int b) {
    if (!dsplit_state_on() || !g_bucket_loaded.insert(b).second) { return; }
    std::string v;
    if (!moe_state_get(g_calib_section, dsplit_state_key("dsplit_arm3_b" + std::to_string(b)), v)) { return; }
    double u0, u1; int n0, n1;
    if (sscanf(v.c_str(), "%lf %d %lf %d", &u0, &n0, &u1, &n1) == 4 && u0 >= 0 && u1 >= 0) {
        dsplit_bucket & B = g_buckets[b];
        if (n0 > 0 && B.n[0] == 0) { B.us_tok[0] = u0; B.n[0] = n0; B.prior[0] = true; }
        if (n1 > 0 && B.n[1] == 0) { B.us_tok[1] = u1; B.n[1] = n1; B.prior[1] = true; }
        LLAMA_LOG_WARN("split planner: size bucket %d from the state file: one card %s, split %s\n", b, n0 > 0 ? (std::to_string((int) (1e6 / u0)) + " tokens/s").c_str() : "-", n1 > 0 ? (std::to_string((int) (1e6 / u1)) + " tokens/s").c_str() : "-");
    }
}
// every stored bucket of this model, cards and build, once per process: the first pass of a size never measured takes the nearest measured size's arm
// ("measured: the nearest size") instead of one card. Before (2026-10-23), only the pass's own bucket was read, so GLM-5.3-Flash 3.5-bit ran the first
// 4233-token chunk of a 12k prompt on one card at 240 t/s while the stored 4096-token bucket said split, 312 t/s.
static void dsplit_bucket_load_all() {
    static bool done = false;
    if (done || !dsplit_state_on()) { return; }
    done = true;
    std::vector<std::pair<std::string, std::string>> kv;
    if (!moe_state_section_kv(g_calib_section, kv)) { return; }
    const std::string pre = "dsplit_arm3_b", suf = dsplit_state_key("");   // keys are pre + bucket + "@<hw>#<build>"
    for (const auto & p : kv) {
        const std::string & k = p.first;
        if (k.compare(0, pre.size(), pre) != 0 || k.size() <= suf.size() || k.compare(k.size() - suf.size(), suf.size(), suf) != 0) { continue; }
        dsplit_bucket_load(atoi(k.c_str() + pre.size()));
    }
}
static void dsplit_bucket_save(int b) {
    if (!dsplit_state_on()) { return; }
    const dsplit_bucket & B = g_buckets[b];
    char v[96]; snprintf(v, sizeof v, "%.3f %d %.3f %d", B.us_tok[0], std::min(B.n[0], 3), B.us_tok[1], std::min(B.n[1], 3));
    moe_state_set(g_calib_section, { { dsplit_state_key("dsplit_arm3_b" + std::to_string(b)), v } });
}

double   llama_moe_cache_compute_att_us_fork();   // llama-moecache-fork.cpp: the attention part of the compute per token at the planned cells (declared here, no header edit)
uint64_t llama_moe_cache_pp_cells_get_fork();     // the cells the planned call ends at
// the planner's numbers for the live balance (llama-splitplan-fork.h, dsb_card), set with every plan (under g_plan_mu): the bytes per token that cross between
// a range's card and the expert card, there and back, over all layers with experts; and per attention layer the card that holds its KV cache and the bytes
// of it that a range on another card reads (each pass)
static double g_hop_bytes_tok = 0.0;
static std::vector<std::pair<ggml_backend_dev_t, double>> g_kv_layers;
static double g_xf_avg = -1.0;   // the model's expert share of a token's compute: active expert weights over all weights a token reads (dsplit_zw), -1: not known
static double g_w_bytes_tok = 0.0;   // the weight bytes one token reads (dense + its active experts): the compute per token from a probed matmul rate
// Both from the model, for every split pass whatever chose it (llama_dsplit_decide_fork; the first version set them in the planner only, so a forced split
// balanced on the dense compute alone and put 56% of GLM-5.3-Flash's tokens on the x4 card): the layers with experts and the attention layers' KV rows
// are read once, the cells the KV holds come with every call (under g_plan_mu).
static void dsplit_link_model(const llama_model & model, int64_t T) {
    struct att_layer { ggml_backend_dev_t dev; double row; bool swa; };
    static const llama_model *    cached = nullptr;
    static int                    n_moe  = 0;
    static std::vector<att_layer> att;
    if (cached != &model) {
        cached = &model;
        n_moe  = 0;
        att.clear();
        const int L = (int) model.hparams.n_layer();
        const double n_used = (double) std::max<uint32_t>(1, model.hparams.n_expert_used_max());
        const double n_exp  = (double) std::max<uint32_t>(1, model.hparams.n_expert);
        double w_act = 0.0, w_all = 0.0, w_b = 0.0;   // the weights a token reads: its active experts' and everything else's (as dsplit_plan_compute counts them), and their bytes
        for (int il = 0; il < L && il < (int) model.layers.size(); ++il) {
            const llama_layer & ly = model.layers[il];
            const ggml_tensor * const * p = reinterpret_cast<const ggml_tensor * const *>(&ly);
            bool moe = false;
            for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *) && !moe; ++i) { moe = p[i] && strstr(p[i]->name, "_exps"); }
            n_moe += moe;
            for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
                const ggml_tensor * t = p[i];
                if (!t || ggml_n_dims(t) < 2 || strstr(t->name, "nextn")) { continue; }
                const double sh = strstr(t->name, "_exps") ? n_used / n_exp : 1.0;
                const double e  = (double) ggml_nelements(t) * sh;
                w_all += e;
                w_b   += (double) ggml_nbytes(t) * sh;
                if (strstr(t->name, "_exps")) { w_act += e; }
            }
            if (!model.hparams.is_recr(il)) {
                // MLA (DeepSeek, GLM-DSA, Kimi): the cache holds the compressed latent and its rope part, one K row per cell and no V
                const double row = model.hparams.is_mla() ? (double) (model.hparams.n_lora_kv + model.hparams.n_rot())
                                                          : (double) (model.hparams.n_embd_k_gqa(il) + model.hparams.n_embd_v_gqa(il));
                att.push_back({ model.dev_layer(il), row, model.hparams.is_swa(il) });
            }
        }
        g_xf_avg = w_all > 0.0 && w_act > 0.0 ? w_act / w_all : -1.0;
        g_w_bytes_tok = w_b;
    }
    const int    hop_mode = llama_dsplit_hop_mode_range_fork();
    const double b_in  = (double) model.hparams.n_embd * (hop_mode != 0 ? 2.0 : 4.0);
    const double b_out = (double) model.hparams.n_embd * (hop_mode == 1 ? 2.0 : 4.0);
    g_hop_bytes_tok = (double) n_moe * (b_in + b_out);
    const double cells = (double) llama_moe_cache_pp_cells_get_fork();
    g_kv_layers.clear();
    for (const auto & a : att) {
        const double seen = a.swa ? std::min(cells, (double) model.hparams.n_swa + (double) T) : cells;   // a sliding-window layer: only its window
        if (seen > 0.0) { g_kv_layers.push_back({ a.dev, seen * a.row * 2.0 }); }
    }
}
static double g_att_calc_us = 0.0;   // the attention of one token at the planned cells from the fitted latency curve (expert card), until passes are timed; under g_plan_mu
static double dsplit_mm_rate(ggml_backend_dev_t d);   // below: the measured matmul rate of a card (weight bytes x tokens per second)
static void dsplit_plan_compute(const llama_model & model, int64_t T, double c_us, llama_dsplit_plan_fork & out, std::vector<double> & f_help, std::vector<double> & s_cards) {
    out = llama_dsplit_plan_fork();
    f_help.clear();
    s_cards.clear();
    const std::vector<ggml_backend_dev_t> cards = g_cards;   // the cards of a split pass, the expert card last (the caller holds g_mu)
    const int n = (int) cards.size(), x = n - 1;
    if (n < 2 || c_us <= 0.0 || T <= 0) {
        return;
    }
    const ggml_backend_dev_t dm = cards[x];
    std::vector<double> B(n, 0.0), rel(n, 1.0);   // per card: upload rate, compute against the expert card's (probed at load)
    for (int i = 0; i < n; ++i) {
        B[i] = dsplit_link_once(cards[i]);
        if (B[i] <= 0.0) {
            return;
        }
        // compute against the expert card's: the measured matmul latencies (dsplit_mm_bench), else the probe's short burst
        const double ri = dsplit_mm_rate(cards[i]), rx = dsplit_mm_rate(dm);
        if (ri > 0.0 && rx > 0.0) { rel[i] = ri / rx; }
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
    // A split range attends to the whole cache of its layer, which sits on the layer's own card: a range on another card reads it over the slower of the two
    // links, every layer, every pass (f16 K and V of every cell the layer sees; a sliding-window layer only its window).
    const double att_meas = llama_moe_cache_compute_att_us_fork();   // measured from timed prompt passes (base + a x cells), 0 before the first
    const double att_tok = std::min(c_us, std::max(0.0, att_meas > 0.0 ? att_meas : g_att_calc_us));
    const double c_base  = c_us - att_tok;
    const double cells   = (double) llama_moe_cache_pp_cells_get_fork();
    // recurrent layers (Qwen3.5/3.6, Qwen3-Next: linear attention) hold no KV and do no attention over the cells: the attention goes to the others
    std::vector<double> kv_b(L, 0.0), att_w(L, 0.0), lB(L, 1e9);
    std::vector<int> lcard(L, -1);
    int n_att = 0;
    for (int il = 0; il < L; ++il) { if (!model.hparams.is_recr(il)) { n_att++; } }
    for (int il = 0; il < L; ++il) {
        const ggml_backend_dev_t ld = model.dev_layer(il);   // where the layer's KV cache or recurrent state is
        for (int i = 0; i < n; ++i) { if (cards[i] == ld) { lcard[il] = i; } }
        if (ld && ggml_backend_dev_type(ld) == GGML_BACKEND_DEVICE_TYPE_GPU) { lB[il] = dsplit_link_once(ld); }
        if (lB[il] <= 0.0) { lB[il] = 1e9; }
        if (model.hparams.is_recr(il)) { continue; }
        att_w[il] = 1.0 / std::max(1, n_att);
        const double cl_seen = model.hparams.is_swa(il) ? std::min(cells, (double) model.hparams.n_swa + (double) T) : cells;
        // MLA (DeepSeek, GLM-DSA, Kimi): the cache holds the compressed latent and its rope part, one K row per cell and no V
        const double row = model.hparams.is_mla() ? (double) (model.hparams.n_lora_kv + model.hparams.n_rot())
                                                  : (double) (model.hparams.n_embd_k_gqa(il) + model.hparams.n_embd_v_gqa(il));
        kv_b[il] = cl_seen * row * 2.0;
    }
    // bytes per token that cross per layer, there and back (llama_dsplit_hop_mode_fork: f16 both ways, f16 there and f32 back, or f32 both ways)
    const int    hop_mode = llama_dsplit_hop_mode_range_fork();
    dsxn_in in;   // the model of a split pass over the cards and its shares: llama-splitplan-fork.h
    in.T = T; in.L = L; in.w = w; in.xf = xf; in.E = E; in.kv_b = kv_b; in.att_w = att_w; in.w_sum = w_sum; in.lcard = lcard; in.lB = lB;
    in.c_base = c_base; in.att_tok = att_tok; in.rel = rel; in.B = B; in.x = x;
    in.b_in = (double) model.hparams.n_embd * (hop_mode != 0 ? 2.0 : 4.0); in.b_out = (double) model.hparams.n_embd * (hop_mode == 1 ? 2.0 : 4.0);
    in.held.assign(n, std::vector<double>(L, 0.0));
    for (int i = 0; i < n; ++i) { for (int il = 0; il < L; ++il) { in.held[i][il] = llama_moe_cache_resident_frac_fork(il, cards[i]); } }
    if (g_layer_order.empty()) {   // the layer tuner's order (split-balance=0), fixed at the first plan: the first helper's own layers first
        for (int il = 0; il < L; ++il) { if (E[il] > 0.0) { g_layer_order.push_back(il); } }
        std::stable_sort(g_layer_order.begin(), g_layer_order.end(), [&](int a, int b) { return in.held[0][a] > in.held[0][b]; });
        g_layer_rank.assign(L, -1);
        for (size_t r = 0; r < g_layer_order.size(); ++r) { g_layer_rank[g_layer_order[r]] = (int) r; }
    }
    const dsxn_out so = dsxn_plan(in);
    // both estimates scaled by what passes of this size (or the nearest measured size) measured against the model (g_calib_split_b, g_calib_one_b)
    const int    bk      = dsplit_bucket_of(T);
    g_one_raw_b[bk] = so.t_one / (double) T;
    const double c_one   = calib_near(g_calib_one_b, bk, 1.0);
    const double c_split = calib_near(g_calib_split_b, bk, g_split_calib.load());
    const double t_one   = so.t_one * c_one;
    // the split's estimate is scaled by what earlier split passes measured against their plan (llama_dsplit_report_fork); one card's estimate held to
    // within 4% (Qwen3.8-Flash-Next: 2603 / 2697 ms at 3k, 2038 / 2104 at 1.9k), the split's was 21-26% short (waits the model does not see)
    const double raw_split  = so.t_split;
    double       best_split = raw_split * c_split;
    // What the split costs after the prompt: its mirror copies of the dense weights sit in lent cache blocks, and the experts they displaced have to be
    // uploaded again before decode is back at full speed. How much of that a prompt really displaces grows with its length and saturates: the fraction of a
    // layer's experts that n tokens touch, 1 - (1 - k/E)^n (a 1-token prompt displaces nothing worth counting, a long one the whole lent area). The refill
    // is one more upload of the lent bytes over the expert card's link, charged to the split's time (one-off seconds, same unit as the pass).
    // Both are paid once per request, so they are spread over its passes (before 2026-10-17 every pass carried the whole refill: 63 times over in a 255k prompt).
    // The passes of the request so far are about cells / T (the call's cells include the earlier calls of the same prompt).
    const double n_pass = std::max(1.0, cells / std::max<double>(1.0, (double) T));
    best_split += g_mirror_fill_us.load() / n_pass;   // the copy of the dense weights into the lent blocks, before the first layer (see dsplit_update_fill)
    {
        const double touched = 1.0 - std::pow(1.0 - n_used / n_exp, (double) T);
        best_split += touched * dense_bytes / (B[x] * 1e3) * llama_moe_live_get("refill_weight") / n_pass;   // live: POST /moe/params
    }
    out.valid      = true;
    // clearly ahead after the calibration (20%), else one card: a split predicted 5-15% ahead measured slower on every Qwen3.8-Flash-Next size (2026-10-23)
    out.split      = best_split > 0.0 && best_split < 0.80 * t_one;
    out.share      = 1.0 - so.s[x];   // the helpers' share of the tokens together
    out.t_one_ms   = t_one / 1e3;
    out.t_split_ms = best_split / 1e3;
    out.t_raw_ms   = raw_split / 1e3;
    out.t_add_ms   = (best_split - raw_split * c_split) / 1e3;
    f_help.assign(L, 0.0);   // the plan keeps the experts on the expert card: experts on the helpers are a measured trial (the live balance)
    s_cards = so.s;
    dsplit_link_model(model, T);   // what the live balance models per card (dsplit_bal_step)
}

static void dsplit_tune_load(int b);   // below: a size settled by an earlier start
// ---- matmul latency per card, measured once before the first prompt (user 2026-10-25: "simulate a simple matmul, we need just latencies") --------------------
// The probe's matmul (a fixed Q4_K shape at 512 tokens, before the model is known) set the cards' compute ratio and, with no prompt pass timed yet, nothing
// else. Here each card of the split times ONE matmul shaped like the model's largest dense weight (its type, its width; a zeroed weight: the speed does not
// depend on the values) at two token counts. The slope between them is the card's cost per token at a prompt pass's size (one count alone mixes in the fixed
// launch and weight-read cost, which dominates at few tokens). Result: weight bytes x tokens per second per card, in g_mm_rate; about 100 MB of VRAM for a
// few milliseconds. A card whose buffers cannot be allocated keeps the probe's rate.
static std::map<ggml_backend_dev_t, double> g_mm_rate;   // under g_plan_mu
// The timing runs when the first context is created - at startup that is the parameter fit, which mutes the log - so its lines are kept here and printed once
// at the first prompt decision (llama_dsplit_decide_fork), where fork/run.py reads them (the attention layer kinds go to models.ini).
static std::string g_bench_log;   // under g_plan_mu
static bool        g_bench_logged = false;
#define BENCH_LOG(...) do { const std::string bl_ = format(__VA_ARGS__); g_bench_log += bl_; LLAMA_LOG_WARN("%s", bl_.c_str()); } while (0)
// attention per card (user 2026-10-25: "time just attn, the only nonlinear term"; "fit a curve by 3 points, extract the nonlinear coefs"): one flash-attention
// op of the model's first full-attention layer (its heads and head sizes; MLA: the compressed cache, one KV head) for ATT_T query tokens over 2k, 8k and 32k cells,
// fitted exactly by t(c) = a + b c + g c^2 (us for ATT_T tokens, one layer). b is the cost per cell, g what grows faster than the cells (cache effects; ~0 when
// attention is linear per token). The per-token cost of a layer at c cells is t(c) / ATT_T.
struct dsplit_att_fit { double a = 0.0, b = 0.0, g = 0.0; bool ok = false; };
static std::map<ggml_backend_dev_t, dsplit_att_fit> g_att_fit;   // under g_plan_mu
static constexpr int64_t ATT_T = 512;
static bool g_mm_done = false;
static double dsplit_mm_rate(ggml_backend_dev_t d) {
    auto it = g_mm_rate.find(d);
    if (it != g_mm_rate.end() && it->second > 0.0) { return it->second; }
    const auto * p = llama_gpu_probe_find_fork(d);
    return p ? (p->compute_sust > 0.0 ? p->compute_sust : p->compute) : 0.0;
}
// ---- sparse attention gate (models whose attention selects cells: an indexer's top-k, hp.indexer_top_k > 0) ---------------------------------------------
// Their attention runs with n_kv_max = the selection (glm5-next, qwen4exp), and the CUDA kernel gathers just those cells - but only from a K/V length its own
// rule fixes (max(4096, 2 x queries per tile x selection): ~33k cells for GLM-5.3-Flash prompt passes). Below that it attends to every cell with a mask, and
// the cost per token grows with the context: GLM 3.0 on an RTX 3090, 1302 us per token at 24-33k cells on the masked path, 924 at 33-42k on the gather.
// Here both paths are timed at load for the model's attention (its heads and head sizes) at 4k, 8k, 16k and 32k cells, over a mask that gives every
// query a DIFFERENT selection (the union the gather collects per tile is then the largest it gets: the gate found is where the gather wins even so), and
// the gate is set where the gather starts to win (interpolated), the largest over the cards. --moe sparse-gate=0: the kernel's own rule.
static std::vector<int32_t> g_sparse_min_card;   // under g_plan_mu: one entry per measured card, 0 = the gather did not win up to 32k cells there
static int32_t g_sparse_min = 0;
int32_t llama_dsplit_sparse_min_fork() {
    static const bool on = [] { const char * e = llama_moe_get_opt("sparse-gate"); return !e || atoi(e) != 0; }();
    return on ? g_sparse_min : 0;
}
static void dsplit_sparse_bench(const llama_model & model, ggml_backend_dev_t d, ggml_backend_t be, int64_t hsk, int64_t hsv, int64_t nh, int64_t nkv, bool mla) {
    const auto & hp = model.hparams;
    if (hp.indexer_top_k == 0) { return; }
    const int64_t kpool = std::max<int64_t>(1, hp.indexer_kpool);
    const int64_t C[4] = { 4096, 8192, 16384, 32768 };
    double td[4], ts[4];
    int64_t nsel_last = 0;
    for (int k = 0; k < 4; ++k) {
        const int64_t n_pool = C[k] / kpool;
        const int64_t n_sel  = kpool * std::min<int64_t>(n_pool, (int64_t) hp.indexer_top_k / kpool) + (hp.indexer_kpool_select_tail ? kpool - 1 : 0);
        nsel_last = n_sel;
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, ATT_T, nh, 1);
        ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, C[k], nkv, 1);
        ggml_tensor * vt = mla ? ggml_view_4d(ctx, kt, hsv, C[k], nkv, 1, kt->nb[1], kt->nb[2], kt->nb[3], 0)
                               : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsv, C[k], nkv, 1);
        ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, C[k], ATT_T, 1, 1);
        const float scale = 1.0f / sqrtf((float) hsk);
        ggml_tensor * fd = ggml_flash_attn_ext(ctx, q, kt, vt, m, scale, 0.0f, 0.0f);
        ggml_tensor * fs = ggml_flash_attn_ext(ctx, q, kt, vt, m, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_n_kv_max(fs, (int32_t) n_sel);
        ggml_flash_attn_ext_set_sparse_min(fs, 1);   // the gather whatever the length
        ggml_cgraph * gd = ggml_new_graph(ctx); ggml_build_forward_expand(gd, fd);
        ggml_cgraph * gs = ggml_new_graph(ctx); ggml_build_forward_expand(gs, fs);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        td[k] = ts[k] = 1e30;
        if (buf) {
            ggml_backend_buffer_clear(buf, 0);
            std::vector<ggml_fp16_t> mh((size_t) C[k] * ATT_T, ggml_fp32_to_fp16(-INFINITY));
            for (int64_t r = 0; r < ATT_T; ++r) {
                for (int64_t j = 0; j < n_sel; ++j) { mh[(size_t) r * C[k] + (size_t) ((r * n_sel + j) % C[k])] = ggml_fp32_to_fp16(0.0f); }
            }
            ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
            ggml_cgraph * g2[2] = { gd, gs };
            double * t2[2] = { &td[k], &ts[k] };
            for (int a = 0; a < 2; ++a) {
                for (int r = 0; r < 2; ++r) { ggml_backend_graph_compute(be, g2[a]); }
                ggml_backend_synchronize(be);
                for (int r = 0; r < 5; ++r) {
                    const int64_t t0 = ggml_time_us();
                    ggml_backend_graph_compute(be, g2[a]);
                    ggml_backend_synchronize(be);
                    *t2[a] = std::min(*t2[a], (double) (ggml_time_us() - t0));
                }
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        if (!buf) { BENCH_LOG("split planner: sparse attention gate on %s not measured (no buffer at %lld cells)\n", ggml_backend_dev_name(d), (long long) C[k]); return; }
    }
    // the gate: where the gather starts to win, interpolated between the two lengths around the change (0: it never won up to 32k cells)
    int32_t gate = 0;
    for (int k = 0; k < 4 && gate == 0; ++k) {
        if (ts[k] >= td[k]) { continue; }
        if (k == 0) { gate = (int32_t) C[0]; break; }
        const double a = td[k - 1] - ts[k - 1], b = td[k] - ts[k];   // a <= 0 < b
        gate = (int32_t) ((double) C[k - 1] + (double) (C[k] - C[k - 1]) * (-a) / std::max(1e-9, b - a));
    }
    g_sparse_min_card.push_back(gate);
    bool all = !g_sparse_min_card.empty();
    int32_t mx = 0;
    for (int32_t v : g_sparse_min_card) { all = all && v > 0; mx = std::max(mx, v); }
    g_sparse_min = all ? mx : 0;
    BENCH_LOG("split planner: sparse attention gate on %s (selection %lld cells, %lld query heads, K %lld V %lld%s, %lld tokens): masked / gather %.0f/%.0f, %.0f/%.0f, "
                   "%.0f/%.0f, %.0f/%.0f us at 4k/8k/16k/32k cells: the gather from %s (the gate now: %d cells)\n", ggml_backend_dev_name(d), (long long) nsel_last,
            (long long) nh, (long long) hsk, (long long) hsv, mla ? ", MLA" : "", (long long) ATT_T, td[0], ts[0], td[1], ts[1], td[2], ts[2], td[3], ts[3],
            gate > 0 ? (std::to_string(gate) + " cells").c_str() : "never up to 32k (the kernel's own rule stays)", g_sparse_min);
}

static void dsplit_att_bench(const llama_model & model, ggml_backend_dev_t d, ggml_backend_t be) {
    const auto & hp = model.hparams;
    {   // which layers grow with the context, from the GGUF (user 2026-10-25: "extract from the gguf how many are quadratic, the rest linear"; run.py stores the line in models.ini)
        static const llama_model * said = nullptr;
        if (said != &model) {
            said = &model;
            int full = 0, swa = 0, lin = 0;
            for (uint32_t i = 0; i < hp.n_layer(); ++i) { if (hp.is_recr(i)) { lin++; } else if (hp.is_swa(i)) { swa++; } else { full++; } }
            BENCH_LOG("split planner: attention layers: %d full (grow with the context), %d sliding-window (window %u), %d linear (no context term), sparse top-k %u%s\n",
                    full, swa, hp.n_swa, lin, hp.indexer_top_k,
                    hp.indexer_top_k > 0 ? " (the attention gathers only the selected cells above the sparse gate, dsplit_sparse_bench)" : "");
        }
    }
    int il = -1;
    for (uint32_t i = 0; i < hp.n_layer(); ++i) { if (!hp.is_recr(i) && !hp.is_swa(i)) { il = (int) i; break; } }
    if (il < 0) { for (uint32_t i = 0; i < hp.n_layer(); ++i) { if (!hp.is_recr(i)) { il = (int) i; break; } } }
    if (il < 0) { return; }   // no attention layer (all recurrent): nothing grows with the context
    const bool mla = hp.is_mla();
    const int64_t hsk = mla ? (int64_t) (hp.n_lora_kv + hp.n_rot()) : (int64_t) hp.n_embd_head_k(il);
    const int64_t hsv = mla ? (int64_t) hp.n_lora_kv : (int64_t) hp.n_embd_head_v(il);
    const int64_t nh = hp.n_head(il), nkv = mla ? 1 : std::max<int64_t>(1, hp.n_head_kv(il));
    if (hsk <= 0 || hsv <= 0 || nh <= 0) { return; }
    const int64_t C[3] = { 2048, 8192, 32768 };
    ggml_init_params ip = { 24 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, ATT_T, nh, 1);
    ggml_cgraph * g[3];
    for (int k = 0; k < 3; ++k) {
        ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsk, C[k], nkv, 1);
        ggml_tensor * vt = mla ? ggml_view_4d(ctx, kt, hsv, C[k], nkv, 1, kt->nb[1], kt->nb[2], kt->nb[3], 0)
                               : ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hsv, C[k], nkv, 1);
        ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, C[k], ATT_T, 1, 1);   // all zero: every cell visible
        g[k] = ggml_new_graph(ctx);
        ggml_build_forward_expand(g[k], ggml_flash_attn_ext(ctx, q, kt, vt, m, 1.0f / sqrtf((float) hsk), 0.0f, 0.0f));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    double t[3] = { 1e30, 1e30, 1e30 };
    if (buf) {
        ggml_backend_buffer_clear(buf, 0);
        for (int k = 0; k < 3; ++k) {
            for (int r = 0; r < 2; ++r) { ggml_backend_graph_compute(be, g[k]); }   // warm (the card is at its sustained clocks from the matmul loop)
            ggml_backend_synchronize(be);
            for (int r = 0; r < 5; ++r) {
                const int64_t t0 = ggml_time_us();
                ggml_backend_graph_compute(be, g[k]);
                ggml_backend_synchronize(be);
                t[k] = std::min(t[k], (double) (ggml_time_us() - t0));
            }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    if (!buf) { BENCH_LOG("split planner: attention latency on %s not measured (no buffer)\n", ggml_backend_dev_name(d)); return; }
    // exact quadratic through the three points (divided differences)
    const double d1 = (t[1] - t[0]) / (double) (C[1] - C[0]), d2 = (t[2] - t[1]) / (double) (C[2] - C[1]);
    dsplit_att_fit f;
    f.g = (d2 - d1) / (double) (C[2] - C[0]);
    f.b = d1 - f.g * (double) (C[0] + C[1]);
    f.a = t[0] - f.b * (double) C[0] - f.g * (double) C[0] * (double) C[0];
    f.ok = f.b > 0.0 || f.g > 0.0;
    g_att_fit[d] = f;
    dsplit_sparse_bench(model, d, be, hsk, hsv, nh, nkv, mla);
    BENCH_LOG("split planner: attention latency on %s (layer %d, %lld heads / %lld KV heads, K %lld V %lld%s, %lld tokens): %.0f / %.0f / %.0f us at 2k / 8k / 32k cells: "
                   "t(c) = %.1f + %.5f c + %.3g c^2 us, %.4f us per token per cell (at 32k cells %.0f%% of it from the c^2 term)\n",
            ggml_backend_dev_name(d), il, (long long) nh, (long long) nkv, (long long) hsk, (long long) hsv, mla ? ", MLA" : "", (long long) ATT_T, t[0], t[1], t[2],
            f.a, f.b, f.g, f.b / (double) ATT_T, t[2] > 0.0 ? 100.0 * f.g * 32768.0 * 32768.0 / t[2] : 0.0);
}
// the attention of ONE token over the whole model at c cells on card d (us): every full-attention layer at c, a sliding-window layer at its window (0 = no fit)
static double dsplit_att_us_tok(const llama_model & model, ggml_backend_dev_t d, double c, double T) {
    auto it = g_att_fit.find(d);
    if (it == g_att_fit.end() || !it->second.ok) { return 0.0; }
    const auto & f = it->second;
    const auto & hp = model.hparams;
    double us = 0.0;
    for (uint32_t i = 0; i < hp.n_layer(); ++i) {
        if (hp.is_recr(i)) { continue; }
        const double ci = hp.is_swa(i) ? std::min(c, (double) hp.n_swa + T) : c;
        us += std::max(0.0, f.a + f.b * ci + f.g * ci * ci) / (double) ATT_T;
    }
    return us;
}

static void dsplit_mm_bench(const llama_model & model, const std::vector<ggml_backend_dev_t> & devs) {
    if (g_mm_done || devs.size() < 2) { return; }
    g_mm_done = true;
    const ggml_tensor * big = nullptr;   // the largest dense weight of the model's first layers: its type and width stand for the dense part
    for (int il = 0; il < (int) model.layers.size() && il < 4; ++il) {
        const ggml_tensor * const * pt = reinterpret_cast<const ggml_tensor * const *>(&model.layers[il]);
        for (size_t i = 0; i < sizeof(llama_layer) / sizeof(ggml_tensor *); ++i) {
            const ggml_tensor * t = pt[i];
            if (!t || ggml_n_dims(t) != 2 || strstr(t->name, "_exps") || strstr(t->name, "nextn")) { continue; }
            if (!big || ggml_nbytes(t) > ggml_nbytes(big)) { big = t; }
        }
    }
    if (!big) { return; }
    const int64_t K = big->ne[0], N = std::min<int64_t>(big->ne[1], 8192);
    const int64_t T0 = 256, T1 = 1024;
    for (ggml_backend_dev_t d : devs) {
        ggml_backend_t be = ggml_backend_dev_init(d, nullptr);
        if (!be) { continue; }
        ggml_init_params ip = { 8 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W  = ggml_new_tensor_2d(ctx, big->type, K, N);
        ggml_tensor * X0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T0);
        ggml_tensor * X1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T1);
        ggml_cgraph * g0 = ggml_new_graph(ctx); ggml_build_forward_expand(g0, ggml_mul_mat(ctx, W, X0));
        ggml_cgraph * g1 = ggml_new_graph(ctx); ggml_build_forward_expand(g1, ggml_mul_mat(ctx, W, X1));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        double best[2] = { 1e30, 1e30 };
        if (buf) {
            ggml_backend_buffer_clear(buf, 0);
            ggml_cgraph * gs[2] = { g0, g1 };
            for (int k = 0; k < 2; ++k) {
                // a long loop first (user 2026-10-25): 300 ms of the same matmul, so the card runs at the clocks its power cap holds in a long pass, not its
                // boost clocks (the probe's short burst read a x4 card at 220 W 6% fast, its 3 s burst 20%)
                for (const int64_t t_end = ggml_time_us() + 300000; ggml_time_us() < t_end; ) { ggml_backend_graph_compute(be, gs[k]); ggml_backend_synchronize(be); }
                for (int r = 0; r < 5; ++r) {
                    const int64_t t0 = ggml_time_us();
                    ggml_backend_graph_compute(be, gs[k]);
                    ggml_backend_synchronize(be);
                    best[k] = std::min(best[k], (double) (ggml_time_us() - t0));
                }
            }
            ggml_backend_buffer_free(buf);
        }
        if (buf && best[1] > best[0]) {
            const double rate = (double) ggml_nbytes(W) * (double) (T1 - T0) / ((best[1] - best[0]) / 1e6);   // weight bytes x tokens per second
            g_mm_rate[d] = rate;
            BENCH_LOG("split planner: matmul latency on %s (%s %lldx%lld): %.0f us at %lld tokens, %.0f us at %lld: %.2f us per token, %.1f GB*token/s (probe %.1f)\n",
                    ggml_backend_dev_name(d), ggml_type_name(big->type), (long long) K, (long long) N, best[0], (long long) T0, best[1], (long long) T1,
                    (best[1] - best[0]) / (double) (T1 - T0), rate / 1e9, [&] { const auto * p = llama_gpu_probe_find_fork(d); return p ? (p->compute_sust > 0 ? p->compute_sust : p->compute) / 1e9 : 0.0; }());
        } else {
            BENCH_LOG("split planner: matmul latency on %s not measured (no buffer): the probe's rate stands\n", ggml_backend_dev_name(d));
        }
        ggml_free(ctx);
        dsplit_att_bench(model, d, be);
        ggml_backend_free(be);
    }
}

// The matmul and attention latencies are timed when the first context is created, before its KV cache and the expert pool take the VRAM. At the first prompt
// (where they ran in pre7/pre8) the pool had left 96-358 MiB on each card: the timing's own backend, buffers and kernel scratch did not fit, and auto mode died
// with "CUDA error: out of memory" in cuMemCreate (MiMo 60k, pre7; forced dense split never reached the planner, so it never timed and never failed).
void llama_dsplit_bench_fork(const llama_model & model) {
    std::vector<ggml_backend_dev_t> devs;
    for (const auto & d : model.devices) { if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) { devs.push_back(d.dev); } }
    std::lock_guard<std::mutex> lk(g_plan_mu);
    dsplit_mm_bench(model, devs);
}

llama_dsplit_plan_fork llama_dsplit_plan_get_fork(const llama_model & model, int64_t n_tokens) {
    static const bool plan_on = [] { const char * e = llama_moe_get_opt("split-plan"); return !e || atoi(e) != 0; }();   // --moe split-plan=0: no model, the measured choice and the old start shares
    if (!plan_on) {
        return llama_dsplit_plan_fork();
    }
    double c_us = llama_moe_cache_compute_us_fork();
    std::lock_guard<std::mutex> lk(g_plan_mu);
    // Calculated, not measured, for the first prompt (user 2026-10-25: "calculated split first pass"): before any prompt pass is timed the compute per token
    // is the weight bytes a token reads (dense weights + its active experts, dsplit_link_model) over the expert card's matmul rate measured at start (probe:
    // weight bytes x tokens per second, the sustained burst when there is one). Without it the plan was invalid and the first prompt ran on one card
    // ("no plan or measurement yet: one card first"). The first measured pass replaces it (the probe's small Q4_K graph is only an estimate of the real kernels).
    if (c_us <= 0.0 && g_w_bytes_tok > 0.0 && !g_cards.empty()) {
        {
            const double rate = dsplit_mm_rate(g_cards.back());
            if (rate > 0.0) {
                // the attention at the cells this call ends at, less half the call (its tokens see from the start to the end of the call: the average)
                const double cells = std::max(0.0, (double) llama_moe_cache_pp_cells_get_fork() - 0.5 * (double) n_tokens);
                g_att_calc_us = dsplit_att_us_tok(model, g_cards.back(), cells, (double) n_tokens);
                c_us = g_w_bytes_tok / rate * 1e6 + g_att_calc_us;
                static bool said = false;
                if (!said) { said = true; LLAMA_LOG_WARN("split planner: compute per token calculated (no prompt pass timed yet): %.0f MB of weights per token / %.1f GB*token/s (matmul latency) + attention %.1f us at %.0f cells (fitted curve) on %s = %.1f us per token\n",
                        g_w_bytes_tok / 1e6, rate / 1e9, g_att_calc_us, cells, ggml_backend_dev_name(g_cards.back()), c_us); }
            }
        }
    }
    dsplit_calib_load(model);
    const int b = dsplit_bucket_of(n_tokens);
    auto & pc = g_plans[b];
    const double fill_now = g_mirror_fill_us.load();
    if (pc.n_tokens != n_tokens || pc.c_us != c_us || pc.fill != fill_now) {
        pc.fill     = fill_now;
        pc.n_tokens = n_tokens;
        pc.c_us     = c_us;
        dsplit_plan_compute(model, n_tokens, c_us, pc.plan, pc.f_help, pc.s_cards);
        dsplit_tune_load(b);   // a size settled by an earlier start (layer count, token share)
        static int logged = 0;
        if (pc.plan.valid && logged++ < 32) {
            std::string sh;   // every card's planned token share, in range order (the expert card last)
            for (size_t i = 0; i < pc.s_cards.size() && i < g_cards.size(); ++i) {
                sh += format("%s%s %.0f%%", i ? ", " : "", ggml_backend_dev_name(g_cards[i]), 100.0 * pc.s_cards[i]);
            }
            LLAMA_LOG_WARN("split planner: %lld tokens: one card %.0f ms, split %.0f ms (tokens: %s; experts on %s): %s\n",
                    (long long) n_tokens, pc.plan.t_one_ms, pc.plan.t_split_ms, sh.c_str(), g_cards.empty() ? "-" : ggml_backend_dev_name(g_cards.back()),
                    pc.plan.split ? "split" : "one card");
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

// --moe split-experts=auto|0|1: who decides whether the helpers in a split also take experts. 0 (default, user's order 2026-10-23: tune the dense split alone
// first, so its timing problems are not mixed with the experts'; the experts on the prefill cards come after): dense split only, the helpers take token
// ranges of the dense part and no experts (the card with the fastest link computes them all). auto: the split planner, the balance's trial and the tuner decide,
// per layer and pass size. 1: dense split plus experts as before the planner (the link-ratio start and the per-layer controller), whatever the planner says.
int llama_dsplit_experts_mode_fork() {
    static const int mode = [] {
        const char * e = llama_moe_get_opt("split-experts");
        if (!e) { return 0; }
        if (strcmp(e, "auto") == 0) { return -1; }
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

// The hand-offs of the DENSE split (token ranges; the expert-only split keeps llama_dsplit_hop_mode_fork): f16 both ways unless --moe split-f16 says otherwise
// (user, 2026-10-23: enable it only in the dense split and see what the balance comes to). The helper's tokens cross to the expert card and their experts' sum
// comes back every MoE layer, 2 x n_embd x 4 bytes per token as f32: MiMo's x4 card spent 238 us per token on that, more than its dense compute (182 us).
// Perplexity: f16 hops measured +0.33 +- 0.17% (Next IQ3_S, 10 chunks paired) before; the paired check on the models that split is still to be repeated.
int llama_dsplit_hop_mode_range_fork() {
    static const int mode = [] {
        const char * e = llama_moe_get_opt("split-f16");
        return e ? llama_dsplit_hop_mode_fork() : 1;
    }();
    return mode;
}

// ---- the live balance of the dense split (2026-10-23, user's design; any number of cards) ---------------------------------------------------------------
// Two kinds of share per card, each balanced on its own part of the cards' busy time in the pass just run (ggml_sched_card_stat: device stamps in front of
// and behind every split, the expert chains apart from the rest):
//   s_i  card i's share of the tokens (its range): the dense parts. A token of its range costs card i its measured dense time over its tokens, plus the
//        hand-offs of that token over its link to the expert card and back (modelled from the probed link rates and the bytes per token: a copy has no
//        stamp), plus once per pass the KV caches of the attention layers other cards hold, read over its link (llama-splitplan-fork.h, dsb_card). The
//        shares that make every card take equally long are the target (dsb_target): the first measured pass of a size moves all the way there, later
//        passes half way, and nothing moves while the cards end within 3% of each other: fast at the start, then small moves that follow the drift for
//        as long as the server runs. A card too slow to help (even split-min-share of the tokens would make it the last) gets no range; it comes back
//        when the others get slower. Without the hand-offs the first balance put 52-54% of GLM-5.3-Flash's tokens on the x4 card of 2x RTX 3090, where a
//        fixed 45% was the fastest of 15/30/45/60% (15k prompt: 420 against 387-389 prompt t/s at 30% and 60%).
//   f_i  card i's share of each layer's experts: the MoE parts. All on the expert card (the dense split alone) until the token shares have held for two
//        passes; then the experts go on at the same ratio, f_i = s_i, and are kept only if the pass gets faster per token than the dense split alone.
//        Slower: the helpers' f halve, pass by pass, and below 2% together the experts are abandoned: the dense split alone again. Kept: f is balanced
//        on the cards' expert times as s is on the dense ones. A pass whose graph does not fit with the helpers' experts drops them at once.
// Before anything is measured every card takes an equal share (user, 2026-10-23: from the card count, 50% each on two cards; the start from the link ratio,
// 30% on the x4 card, was wrong: the dense weights are copied once, a token's dense work does not cross a link). A size not measured yet starts from the
// nearest measured size, else from the last shares of an earlier run (state file). Per pass-size bucket, stored per model, cards and build
// (dsplit_state_key). --moe split-balance=0: the planner's shares and no experts on the helpers; --moe dense-split-share=X fixes the first card's share
// by hand (tests: the other cards split the rest equally).
extern "C" {
    struct dsplit_card_stat { ggml_backend_t backend; int64_t dense_us, moe_us, first_us, last_us; int n_dense, n_moe; int64_t wait_dense_us, wait_moe_us, max_wait_us;
                              int64_t up_wait_us, xwait_us, host_us, link_us; };   // must match ggml_sched_card_stat (ggml-backend.cpp)
    GGML_API int ggml_backend_sched_card_stats(struct dsplit_card_stat * out, int cap, long * gen);
    GGML_API uint64_t ggml_backend_sched_upload_bytes_fork(ggml_backend_dev_t dev);   // a card's uploaded bytes since the last read (pass statistics)
    GGML_API void     ggml_backend_sched_link_bytes_fork(ggml_backend_dev_t dev, uint64_t * in_host, uint64_t * in_card, uint64_t * out_card);   // its other link traffic
}
namespace {
struct dsplit_bal {
    std::vector<ggml_backend_dev_t> dev;   // the cards, in range order (the expert card last)
    std::vector<double> s, f;              // the next pass's token and expert shares
    std::vector<double> s_run, f_run;      // the shares the measured pass ran with
    std::vector<double> c;                 // dense us per token of each card's own range (the last measured; kept while a card has no range)
    std::vector<double> hop, k;            // modelled per card: hand-offs (us per token), KV reads (us per pass)
    std::vector<double> d, m;              // dense and expert ms of each card in the measured pass
    std::vector<double> wd, wm, wx;        // measured idle ms of each card before its dense / expert splits, and its longest single wait (ggml_sched_card_stat)
    std::vector<double> up;                // ms each card's link spent on host uploads in the measured pass: its uploaded bytes / its measured link rate
    // The pass on each card by cause, all measured by stream stamps (ggml_sched_card_stat; user 2026-10-24: "capture total, upload and compute time, the rest
    // is stall"): d + m (kernels) + uw (waiting for its expert uploads) + xw (waiting for another card's result) + st (the rest: nothing queued, and the
    // time before its first and after its last split) = span, the pass on the cards. ln: its upload stream's busy time (overlaps the kernels).
    std::vector<double> uw, xw, st, ln;
    std::vector<double> gb_up, gb_host, gb_in, gb_out;   // GB over its link in the pass: expert uploads, other inputs from host memory, from other cards, to other cards
    double span = 0.0;                     // ms, the measured pass: first split start to last split end over all cards
    std::vector<int64_t> tok;              // tokens of each card in the measured pass
    uint32_t cap = 0;                      // the card limit that pass ran under (the limit changes from pass to pass when the pools shrink it): %limit is of this, not of a later pass's
    uint32_t plim = 0;                     // the pass limit that pass ran under (all cards): a pass with fewer tokens is its call's last, the rest of the prompt
    int    phase = 0, n = 0, still = 0;    // phase: 0 balancing the token shares, 1 experts on trial, 2 the dense split alone (experts abandoned), 3 with experts
    double ref_v = 0.0;                    // us per token of the dense split alone: the trial's reference
    double v = 0.0;                        // us per token of the last pass (its span on the cards)
    std::string last;                      // what the last step did (/moe/stats, the log)
};
std::map<int, dsplit_bal>       g_bal;                // under g_plan_mu

// ---- share climb (user 2026-10-25: "raise the work given to the second card while the total time per token keeps dropping, otherwise take work away") ----------
// On top of the equal-dense-time shares (llama_dsplit_bal_shares_fork) a learned offset of the other cards' share: after every pass cut by the shares the
// offset moves one step; the direction flips and the step halves whenever the pass got slower per token. The objective is the pass's time per token on
// the cards (span / tokens) over the expert card's dense time per token: a long prompt's passes get slower as the context grows (GLM 3.0 60k: 8.3 -> 14.0 s for
// the same 8192 tokens), and the dense time per token grows with it, so the ratio leaves what the shares did. --moe share-climb=0 turns it off.
// ponytail: one offset for all pass sizes and helpers, not stored between runs; a per-model stored offset when it proves out.
struct dsplit_climb { double adj = 0.0, step = 0.04, prev_adj = 0.0; int dir = +1; double last = 0.0; bool cut = false, settled = false; };
dsplit_climb g_climb;   // under g_plan_mu
static bool dsplit_climb_on() {
    static const bool on = [] { const char * e = llama_moe_get_opt("share-climb"); return !e || atoi(e) != 0; }();
    return on;
}
static void dsplit_climb_step(const dsplit_bal & B, size_t n) {
    if (!dsplit_climb_on() || g_climb.settled || !g_climb.cut || n < 2 || B.v <= 0.0 || B.c.size() != n || B.c[n - 1] <= 0.0) { return; }
    const double m = B.v / B.c[n - 1];
    const double adj0 = g_climb.adj;
    // Settling (user 2026-10-25: "settle after a few noisy ups and downs"): every reversal halves the step; once it would drop below 0.01 (three reversals
    // from 0.04) the offset stays at the middle of the last two tried and the climb stops for the run, so noise between passes cannot keep it moving.
    if (g_climb.last > 0.0 && m > g_climb.last) {
        g_climb.dir = -g_climb.dir;
        g_climb.step /= 2.0;
        if (g_climb.step < 0.01) {
            g_climb.adj = 0.5 * (g_climb.adj + g_climb.prev_adj);
            g_climb.settled = true;
            LLAMA_LOG_WARN("split climb: settled at the other cards' share offset %+.3f (pass %.1f us/token, %.3f x the expert card's dense us/token)\n", g_climb.adj, B.v, m);
            return;
        }
    }
    g_climb.prev_adj = g_climb.adj;
    g_climb.adj  = std::min(0.45, std::max(-0.45, g_climb.adj + g_climb.dir * g_climb.step));
    LLAMA_LOG_WARN("split climb: pass %.1f us/token = %.3f x the expert card's dense us/token (before %.3f): the other cards' share offset %+.3f -> %+.3f (step %.3f)\n",
            B.v, m, g_climb.last, adj0, g_climb.adj, g_climb.step);
    g_climb.last = m;
}
bool                            g_bal_used = false;   // a graph builder takes its shares from here
int                             g_bal_last = -1;      // the bucket of the last step
long long                       g_bal_t_ms = 0;       // wall time (ms since the epoch) of the last step: run.py tells this cell's passes from an earlier cell's
std::vector<ggml_backend_dev_t> g_rng_dev;            // the cards of the last split graph, in range order (the cards with no range included)
std::vector<int64_t>            g_rng_tok;            // and their tokens
int64_t                         g_rng_seq = 0;        // split graphs built so far (one per pass whose size changed) and when the last was built: the pass running now
int64_t                         g_rng_t0  = 0;
long                            g_card_seen = 0;
const char * dsplit_bal_phase(int p) {
    return p == 0 ? "balancing the token shares" : p == 1 ? "experts on trial" : p == 2 ? "dense split alone" : "dense split with experts";
}
dsplit_bal & dsplit_bal_get(int b, const std::vector<ggml_backend_dev_t> & dev) {
    auto it = g_bal.find(b);
    if (it != g_bal.end() && it->second.dev == dev) { return it->second; }
    dsplit_bal & B = g_bal[b];
    B = dsplit_bal();   // new, or the cards changed (a card left out for memory): start over
    const size_t n = dev.size();
    B.dev = dev;
    B.s.assign(n, n ? 1.0 / (double) n : 0.0);
    B.f.assign(n, 0.0);
    if (n) { B.f[n - 1] = 1.0; }
    B.c.assign(n, 0.0); B.hop.assign(n, 0.0); B.k.assign(n, 0.0); B.d.assign(n, 0.0); B.m.assign(n, 0.0); B.wd.assign(n, 0.0); B.wm.assign(n, 0.0); B.wx.assign(n, 0.0); B.tok.assign(n, 0);
    B.s_run = B.s; B.f_run = B.f;
    B.phase = 2;   // the dense split alone: every expert on the expert card
    B.last = "nothing measured yet";
    (void) b;      // the shares come from the zero-wait rule (dsplit_zw), the same for every pass size; B keeps the measured times of the size's last pass
    return B;
}
double dsplit_link(ggml_backend_dev_t d) {
    return d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU ? dsplit_link_once(d) : 0.0;
}
// what a card's link adds, modelled from the planner's numbers (g_hop_bytes_tok, g_kv_layers) and the probed link rates: the stamps time compute, not copies
void dsplit_bal_costs(dsplit_bal & B) {
    const size_t n = B.dev.size();
    if (n == 0) { return; }
    const double bx = dsplit_link(B.dev[n - 1]);
    for (size_t i = 0; i < n; ++i) {
        const double bi = dsplit_link(B.dev[i]);
        B.hop[i] = i + 1 < n && bi > 0.0 && bx > 0.0 ? g_hop_bytes_tok / (std::min(bi, bx) * 1e3) : 0.0;
        B.k[i]   = 0.0;
        for (const auto & kl : g_kv_layers) {
            if (kl.first == B.dev[i] || bi <= 0.0) { continue; }
            double bl = dsplit_link(kl.first);
            if (bl <= 0.0) { bl = bi; }
            B.k[i] += kl.second / (std::min(bi, bl) * 1e3);
        }
    }
}
// --moe split-pass=1 (default; user 2026-10-24: the expert card uses its whole micro-batch, not half): passes sized pass by pass with the expert card
// at its one-card limit (llama_context::decode), the other cards at what they finish in the expert card's time (dsplit_zw).
// 2: equal passes of at most that size; 0: the old pass of one card's limit split by the shares (A/B only)
int dsplit_split_pass_mode() {
    static const int m = [] { const char * e = llama_moe_get_opt("split-pass"); return e ? atoi(e) : 1; }();
    return m;
}
bool dsplit_split_pass_on() { return dsplit_split_pass_mode() != 0; }
// Two limits, not one (user 2026-10-24): what ONE card holds (its micro-batch: the most tokens its range of a pass takes, the buffers allocated
// on it) and what a pass of ALL cards takes (the expert card's limit + what each other card keeps up with, llama_dsplit_pass_size_fork).
std::atomic<uint32_t> g_card_lim { 0 };   // the card limit of the decode call in progress (llama_dsplit_note_limit_fork), 0 = not known
// The context's card limit: the largest micro-batch one card holds (cparams.n_ubatch, what a pass may take before the pools' room cuts it). Every pass's share
// is reported against THIS one number (user 2026-10-24: "%limit" against each pass's own, cut limit showed 6176 and 4536 tokens both as 100%).
std::atomic<uint32_t> g_card_max { 0 };
std::atomic<uint32_t> g_pass_lim { 0 };   // the pass limit last sized from it, 0 = not sized

// ---- zero-wait token shares (user 2026-10-24: the rule for every split pass, the first one included) ------------------------------------------------------
// The expert card takes as many tokens as fit its card limit. Every other card takes only the tokens it can finish -- its compute, its uploads and the hand-offs
// over its link, there and back -- in the time the expert card works without needing that card's result, so the expert card never waits. Per helper i:
//   tokens_i = f x k_i x the expert card's tokens,   k_i = w_x / (d_i + h_i + u_i), per token:
//     w_x  the expert card's work that does not need the other cards' results: its own dense part (its range comes first in token order, ds_plan; the one
//          expert op of the layer then needs every range), or with --moe split-pieces=1 its own range's experts (computed before the others' ranges)
//     d_i  card i's dense time;  h_i  its hand-offs over the slower of its and the expert card's link;  u_i  its other uploads (mask rows, KV reads)
// Before any pass is measured, every value comes from what the server measured at start, nothing assumed: the model's compute per token on the expert card
// (c_us, the pass planner's timed batches) split by the model's expert share of a token's weights (g_xf_avg), the cards' sustained compute ratio and link
// rates (probe), the hand-off bytes per token (g_hop_bytes_tok). After every pass: that pass's own measured times. f is what the measurement cannot see (how
// much of a card's chain really overlaps the expert card's work), one per card, kept as the target of that card's chain as a share of the expert card's window
// (see dsplit_zw_step): it starts at 0.75 (a little low, then up: user) and moves on the expert card's wait above its floor. Stored per model, cards and build
// (the next start begins where this one ended). The pass size follows from the shares: the expert card's limit / its share (llama_dsplit_pass_size_fork).
struct dsplit_zw {
    std::vector<ggml_backend_dev_t> dev;   // card order, the expert card last
    std::vector<double> k;                 // per card: its tokens per expert-card token from the measured times (the expert card's own: 1)
    std::vector<double> f;                 // per card: the target of its chain as a share of the expert card's window (starts at 0.75)
    std::vector<std::vector<std::pair<double, double>>> hist;   // on the expert card's slot: (the longest chain / window, its wait in ms) of the last passes (the floor)
    bool   measured = false;
    std::string src;                       // where k came from (the log)
};
dsplit_zw g_zw;
bool dsplit_zw_pieces() {   // --moe split-pieces=1: each range's experts as their own op, the expert card's own range first
    static const bool on = [] { const char * e = llama_moe_get_opt("split-pieces"); return e && atoi(e) != 0; }();
    return on;
}
// k of every card from the start-of-server measurements (see above); without the compute per token (not timed yet) the hand-offs cannot be put in time and are
// left out (k too high by their share): the 0.75 start and the first pass's wait take that back
void dsplit_zw_prior(const std::vector<ggml_backend_dev_t> & dev, std::vector<double> & k, std::string & src) {
    const size_t n = dev.size(), x = n - 1;
    k.assign(n, 0.0);
    k[x] = 1.0;
    const double xf = g_xf_avg > 0.0 && g_xf_avg < 1.0 ? g_xf_avg : 0.5;
    const double wf = dsplit_zw_pieces() ? xf : 1.0 - xf;   // the expert card's window: its own experts (pieces) or its own dense part, as a share of its compute
    const double c_us = llama_moe_cache_compute_us_fork();
    const auto rate = [](ggml_backend_dev_t d) { const auto * p = llama_gpu_probe_find_fork(d); return p ? (p->compute_sust > 0.0 ? p->compute_sust : p->compute) : 0.0; };
    const double sx = rate(dev[x]), bx = dsplit_link(dev[x]);
    for (size_t i = 0; i < x; ++i) {
        const double si = rate(dev[i]), bi = dsplit_link(dev[i]);
        const double slow = si > 0.0 && sx > 0.0 ? sx / si : 1.0;   // card i's dense time per token against the expert card's
        // the expert card's compute per token: the pass planner's timed batches, else the model's weight bytes a token reads over the expert card's probed
        // quantised-matmul rate (weight bytes x tokens per second, llama-gpu-probe-fork.cpp); the hand-offs need it to be put in time (2026-10-24: left out
        // while the compute per token was not timed yet, the first pass gave the x4 card 0.69 x the x16's tokens where its measured k was 0.65 x 0.75)
        const double c_x = c_us > 0.0 ? c_us : (sx > 0.0 && g_w_bytes_tok > 0.0 ? g_w_bytes_tok / sx * 1e6 : 0.0);
        const double h_i = bi > 0.0 && bx > 0.0 ? g_hop_bytes_tok / (std::min(bi, bx) * 1e3) : 0.0;
        k[i] = c_x > 0.0 ? c_x * wf / (c_x * (1.0 - xf) * slow + h_i) : wf / ((1.0 - xf) * slow);
    }
    src = c_us > 0.0 ? format("start-of-server measurements: %.0f us/token compute, experts %.0f%% of it, hand-offs %.0f bytes/token", c_us, 100.0 * xf, g_hop_bytes_tok)
                     : format("start-of-server measurements: %.0f us/token compute from the probed matmul rate, experts %.0f%% of the weights, hand-offs %.0f bytes/token",
                              sx > 0.0 && g_w_bytes_tok > 0.0 ? g_w_bytes_tok / sx * 1e6 : 0.0, 100.0 * xf, g_hop_bytes_tok);
}
void dsplit_zw_save() {   // "CUDA0:0.7500:0.6500,CUDA1:1.0000:1.0000" (card:factor:k)
    if (!dsplit_state_on()) { return; }
    std::string v;
    for (size_t i = 0; i < g_zw.dev.size() && i < g_zw.f.size() && i < g_zw.k.size(); ++i) {
        v += format("%s%s:%.4f:%.4f", i ? "," : "", ggml_backend_dev_name(g_zw.dev[i]), g_zw.f[i], g_zw.k[i]);
    }
    moe_state_set(g_calib_section, { { dsplit_state_key("dsplit_zw5"), v } });
}
// the token shares of these cards for a pass (empty: the rule is off)
std::vector<double> dsplit_zw_shares(const std::vector<ggml_backend_dev_t> & dev) {
    if (dev.size() < 2) { return {}; }
    if (g_zw.dev != dev) {   // the first pass, or the cards changed: start from the start-of-server measurements and the stored factor
        g_zw = dsplit_zw();
        g_zw.dev = dev;
        g_zw.f.assign(dev.size(), 0.75);
        g_zw.hist.assign(dev.size(), {});
        std::vector<double> stored_k;   // the k an earlier run of this model on these cards measured: it replaces the start-of-server estimate
        std::string v;
        if (dsplit_state_on() && moe_state_get(g_calib_section, dsplit_state_key("dsplit_zw5"), v)) {   // the stored factors, by card name
            for (size_t p = 0; p < v.size(); ) {
                size_t e = v.find(',', p);
                if (e == std::string::npos) { e = v.size(); }
                const std::string item = v.substr(p, e - p);
                const size_t c = item.find(':'), c2 = c == std::string::npos ? c : item.find(':', c + 1);
                for (size_t i = 0; c != std::string::npos && i < dev.size(); ++i) {
                    if (item.compare(0, c, ggml_backend_dev_name(dev[i])) != 0) { continue; }
                    const double f  = atof(item.c_str() + c + 1);
                    const double kk = c2 != std::string::npos ? atof(item.c_str() + c2 + 1) : 0.0;
                    if (f > 0.05 && f < 4.0) { g_zw.f[i] = f; }
                    if (kk > 0.0 && kk < 10.0) { stored_k.resize(dev.size(), 0.0); stored_k[i] = kk; }
                }
                p = e + 1;
            }
        }
        bool all_k = stored_k.size() == dev.size();
        for (size_t i = 0; all_k && i + 1 < dev.size(); ++i) { all_k = stored_k[i] > 0.0; }
        if (all_k) {   // every other card has a measured k (a card stored without one starts from the estimate with the rest)
            g_zw.k = stored_k;
            g_zw.k[dev.size() - 1] = 1.0;
            g_zw.measured = true;
            g_zw.src = "the measured times of an earlier run (state file)";
        }
    }
    if (!g_zw.measured) { dsplit_zw_prior(dev, g_zw.k, g_zw.src); }   // again until a pass is measured: the compute per token may be timed by now
    const size_t n = dev.size(), x = n - 1;
    std::vector<double> s(n, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) { s[i] = i == x ? 1.0 : g_zw.f[i] * g_zw.k[i]; sum += s[i]; }
    for (double & v : s) { v /= sum; }
    return s;
}
// after a measured pass (B: its times per card, card order as g_zw.dev): k from the pass's own times, f from the expert card's wait for the others
void dsplit_zw_step(const dsplit_bal & B, int64_t T) {
    const size_t n = B.dev.size();
    if (n < 2 || g_zw.dev != B.dev || T <= 0) { return; }
    const size_t x = n - 1;
    if (B.tok.size() != n || B.tok[x] <= 0 || B.m[x] <= 0.0) { return; }
    static const bool detail = getenv("GGML_SCHED_PASS_STATS") && atoi(getenv("GGML_SCHED_PASS_STATS")) == 1;   // the waits by cause are measured
    // the expert card's window per own token: its own range's experts (pieces; it computes every token's experts, the same time per token) or its own dense part
    const double w_x = dsplit_zw_pieces() ? B.m[x] * 1e3 / (double) T : B.d[x] * 1e3 / (double) B.tok[x];
    std::vector<double> k = g_zw.k;
    std::vector<double> chain(n, 0.0);   // per card: its chain in this pass, ms (its kernels, its uploads, its hand-offs)
    std::string txt;
    for (size_t i = 0; i < x; ++i) {
        if (B.tok[i] <= 0) { continue; }   // no range this pass: its k stays
        const double uw   = i < B.uw.size() ? B.uw[i] : 0.0;
        const double cost = (B.d[i] + uw) * 1e3 / (double) B.tok[i] + B.hop[i];
        if (cost > 0.0) { k[i] = w_x / cost; }
        chain[i] = B.d[i] + uw + B.hop[i] * (double) B.tok[i] / 1e3;
        txt += format("%s %s k %.2f (the expert card's %s %.0f us/token / its dense %.0f + uploads %.0f + hand-offs %.0f)", txt.empty() ? "" : ",", ggml_backend_dev_name(B.dev[i]),
                k[i], dsplit_zw_pieces() ? "experts" : "dense part", w_x, B.d[i] * 1e3 / (double) B.tok[i], uw * 1e3 / (double) B.tok[i], B.hop[i]);
    }
    // the expert card's wait for the others (measured by cause with --moe pass-stats; without it the gaps before its dense splits)
    const double w = detail && x < B.xw.size() ? B.xw[x] : B.wd[x];
    const double window = w_x * (double) B.tok[x] / 1e3;   // ms of the expert card's own work the other cards' chains run beside
    const std::vector<double> f0 = g_zw.f;
    // f is the target of a card's chain as a share of the expert card's window (k is measured every pass, so f x k x the expert card's tokens gives that share).
    // The expert card's wait has a floor that no share removes (Qwen3.8-Flash-Next 60k, x16 + x4, 2026-10-24: 0.40-0.47 s a pass with the x4's chain at 9-42%
    // of the window, 0.95 s at 105%): the floor is its wait in passes where every other card's chain stayed under 60% of the window (they cannot have caused
    // it). Above the floor by more than 3% of the pass: the cards with the longest chains drop 10%; else every card rises 5% a pass, up to its whole window.
    double worst = 0.0;   // the longest chain against the window
    for (size_t i = 0; i < x; ++i) { if (B.tok[i] > 0 && window > 0.0) { worst = std::max(worst, chain[i] / window); } }
    auto & h = g_zw.hist[x];   // (the longest chain / window, the expert card's wait in ms) of the last passes
    h.push_back({ worst, w });
    if (h.size() > 8) { h.erase(h.begin()); }
    double floor_ms = -1.0;
    for (const auto & p : h) { if (p.first < 0.6) { floor_ms = floor_ms < 0.0 ? p.second : std::min(floor_ms, p.second); } }
    const double wd = std::max(0.0, w - std::max(0.0, floor_ms));   // the part of the wait the other cards' tokens cause (all of it while no floor is known)
    const bool waited = wd > 0.03 * B.span;
    std::string fx;
    for (size_t i = 0; i < x; ++i) {
        if (B.tok[i] <= 0) { continue; }   // no range this pass: its factor stays
        const bool late = waited && window > 0.0 && chain[i] / window >= 0.9 * worst;
        if (late) { g_zw.f[i] = std::max(0.05, g_zw.f[i] * 0.9); }
        else      { g_zw.f[i] = std::min(1.0, g_zw.f[i] * 1.05); }
        fx += format("%s %s %.2f -> %.2f%s", fx.empty() ? "" : ",", ggml_backend_dev_name(B.dev[i]), f0[i], g_zw.f[i], late ? " (waited for)" : "");
    }
    g_zw.k = k;
    g_zw.measured = true;
    g_zw.src = "the last pass's measured times";
    LLAMA_LOG_WARN("split shares (zero wait): %s; the expert card waited %.0f ms of %.0f (%.0f%%), %.0f ms of it the floor (any tokens), %.0f ms caused by the tokens; "
            "the others' chains %.0f%% of its window %.0f ms; factors%s\n", txt.c_str(), w, B.span, B.span > 0.0 ? 100.0 * w / B.span : 0.0, std::max(0.0, floor_ms), wd,
            100.0 * worst, window, fx.c_str());
    dsplit_zw_save();
}
// one measured split pass of bucket b: the step of the balance (under g_plan_mu)
void dsplit_bal_step(int b) {
    dsplit_card_stat st[16]; long gen = 0;
    const int nst = ggml_backend_sched_card_stats(st, 16, &gen);
    if (nst < 1 || gen == g_card_seen || g_rng_dev.empty() || g_rng_dev.size() != g_rng_tok.size()) { return; }
    g_card_seen = gen;
    const size_t n = g_rng_dev.size();
    std::vector<const dsplit_card_stat *> cs(n, nullptr);
    for (int j = 0; j < nst; ++j) {
        const ggml_backend_dev_t d = ggml_backend_get_device(st[j].backend);
        for (size_t i = 0; i < n; ++i) { if (g_rng_dev[i] == d) { cs[i] = &st[j]; } }
    }
    int64_t T = 0;
    int64_t t_first = 0, t_last = 0;
    for (size_t i = 0; i < n; ++i) {
        if (g_rng_tok[i] <= 0) { continue; }
        if (!cs[i] || cs[i]->dense_us <= 0) { return; }   // a card with tokens and no dense time: not this pass's measurement
        T += g_rng_tok[i];
        t_first = t_first == 0 ? cs[i]->first_us : std::min(t_first, cs[i]->first_us);
        t_last  = std::max(t_last, cs[i]->last_us);
    }
    if (T <= 0 || t_last <= t_first) { return; }
    dsplit_bal & B = dsplit_bal_get(b, g_rng_dev);
    g_bal_last = b;
    g_bal_t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    dsplit_bal_costs(B);
    B.n++;
    B.v     = (double) (t_last - t_first) / (double) T;   // the pass's span on the cards over its tokens
    B.s_run = B.s;
    B.f_run = B.f;
    B.tok   = g_rng_tok;
    B.cap   = g_card_lim.load();
    B.plim  = g_pass_lim.load();
    double c_max = 0.0;
    B.up.assign(n, 0.0);
    B.uw.assign(n, 0.0); B.xw.assign(n, 0.0); B.st.assign(n, 0.0); B.ln.assign(n, 0.0);
    B.gb_up.assign(n, 0.0); B.gb_host.assign(n, 0.0); B.gb_in.assign(n, 0.0); B.gb_out.assign(n, 0.0);
    B.span = (double) (t_last - t_first) / 1e3;
    for (size_t i = 0; i < n; ++i) {
        // the card's uploads since the last pass step (staging fills and range uploads, counted by the scheduler), as time at its measured link rate
        const double gbs = dsplit_link_once(g_rng_dev[i]);
        const uint64_t ub = ggml_backend_sched_upload_bytes_fork(g_rng_dev[i]);
        B.up[i] = gbs > 0.0 ? (double) ub / (gbs * 1e6) : 0.0;
        {
            uint64_t hb = 0, ib = 0, ob = 0;
            ggml_backend_sched_link_bytes_fork(g_rng_dev[i], &hb, &ib, &ob);
            B.gb_up[i] = ub / 1e9; B.gb_host[i] = hb / 1e9; B.gb_in[i] = ib / 1e9; B.gb_out[i] = ob / 1e9;
        }
        B.d[i] = cs[i] ? cs[i]->dense_us / 1e3 : 0.0;
        B.m[i] = cs[i] ? cs[i]->moe_us / 1e3 : 0.0;
        B.wd[i] = cs[i] ? cs[i]->wait_dense_us / 1e3 : 0.0;
        B.wm[i] = cs[i] ? cs[i]->wait_moe_us / 1e3 : 0.0;
        B.wx[i] = cs[i] ? cs[i]->max_wait_us / 1e3 : 0.0;
        if (cs[i]) {
            B.uw[i] = cs[i]->up_wait_us / 1e3;
            B.xw[i] = cs[i]->xwait_us / 1e3;
            B.ln[i] = cs[i]->link_us / 1e3;
            // nothing queued between its splits, plus the pass before its first split and after its last (the other cards were working then)
            B.st[i] = (cs[i]->host_us + std::max<int64_t>(0, cs[i]->first_us - t_first) + std::max<int64_t>(0, t_last - cs[i]->last_us)) / 1e3;
        }
        if (g_rng_tok[i] > 0 && cs[i]) { B.c[i] = (double) cs[i]->dense_us / (double) g_rng_tok[i]; }
        c_max = std::max(c_max, B.c[i]);
    }
    (void) c_max;
    dsplit_climb_step(B, n);
    {   // this split pass's per-card factors (dsf): dense us/token at its context; hand-off; and for the experts on a card (dense + experts on cards whose links can
        // stream them, x16 / PCIe 5; user 2026-10-25): its expert compute per token of its expert share, and its measured upload rate
        std::lock_guard<std::mutex> dl(g_dsf_mu);
        dsf_load();
        if (g_dsf_ps.choice == 1 && g_dsf_ps.pos0 >= 0) {
            const int kb = (int) ((g_dsf_ps.pos0 + g_dsf_ps.T / 2) / DSF_CELLS);
            for (size_t i = 0; i < n && i < B.c.size(); ++i) {
                if (g_rng_tok[i] <= 0 || B.c[i] <= 0.0) { continue; }
                const std::string nm = ggml_backend_dev_name(g_rng_dev[i]);
                dsf_pt & dd = g_dsf_dense[nm][kb];
                dsf_upd(dd, B.c[i]);
                dsf_mark("dsf.dense." + nm + ".k" + std::to_string(kb), dd);
                if (i < B.hop.size() && B.hop[i] > 0.0) { dsf_upd(g_dsf_hop[nm], B.hop[i]); }   // in memory only: bytes per token / link rate, known at every start
                if (i < B.f_run.size() && B.f_run[i] > 0.0 && i < B.m.size() && B.m[i] > 0.0) {   // us per token of this card's share of a layer's experts
                    static std::map<std::string, dsf_pt> moe;
                    dsf_pt & m = moe[nm];
                    dsf_upd(m, B.m[i] * 1e3 / ((double) T * B.f_run[i]));   // in memory only (user 2026-10-25: store only what is hard to get, like attention)
                }
                if (i < B.ln.size() && B.ln[i] > 0.0 && i < B.gb_up.size() && B.gb_up[i] > 0.01) {   // measured upload GB/s on this card's link (expert uploads)
                    static std::map<std::string, dsf_pt> up;
                    dsf_pt & u = up[nm];
                    dsf_upd(u, B.gb_up[i] / (B.ln[i] / 1e3));   // in memory only: the probe measures the links at every start
                }
            }
        }
    }
    // the next pass's shares: the zero-wait rule from this pass's measured times (dsplit_zw); every expert stays on the expert card (the dense split alone)
    dsplit_zw_step(B, T);
    const std::string msg = format("zero-wait shares (%s)", g_zw.src.c_str());
    B.last = msg;
    std::string cards;
    for (size_t i = 0; i < n; ++i) {
        cards += format("%s%s %lld tokens (%.0f%%) %.0f%% of the experts, dense %.0f ms (%.1f us/token + hand-offs %.1f) moe %.0f ms, waits %.0f ms before dense + %.0f ms before experts (longest %.1f ms)", i ? " | " : "",
                ggml_backend_dev_name(B.dev[i]), (long long) g_rng_tok[i], 100.0 * B.s_run[i], 100.0 * B.f_run[i], B.d[i], B.c[i], B.hop[i], B.m[i], B.wd[i], B.wm[i], B.wx[i]);
        if (B.cap > 0) { cards += format(", %.0f%% of its card limit %u", 100.0 * (double) g_rng_tok[i] / (double) B.cap, B.cap); }
        if (i < B.up.size()) { cards += format(", upload %.0f ms", B.up[i]); }
        if (i < B.uw.size()) {   // the pass on this card by cause (measured): the five add up to the pass
            cards += format("; pass %.0f ms = kernels %.0f + upload wait %.0f + wait for other cards %.0f + stall %.0f; link busy %.0f ms; link GB: experts %.2f, other host %.2f, from cards %.2f, to cards %.2f",
                    B.span, B.d[i] + B.m[i], B.uw[i], B.xw[i], B.st[i], B.ln[i], B.gb_up[i], B.gb_host[i], B.gb_in[i], B.gb_out[i]);
        }
    }
    LLAMA_LOG_WARN("split balance: %s | pass %.1f us/token | %s: %s\n", cards.c_str(), B.v, dsplit_bal_phase(B.phase), msg.c_str());
    g_tune_changed = true;   // the next pass is built with the new shares
}
void dsplit_pass_step_locked() {   // under g_plan_mu: the balance step of the split pass just run
    int64_t n = 0;
    for (int64_t t : g_rng_tok) { n += t; }
    if (n > 0) { dsplit_bal_step(dsplit_bucket_of(n)); }
}
}  // namespace

// the token share of every card (in the order of `dev`, the cards of llama_dsplit_cards_fork) for a split pass of n_tokens: the zero-wait rule (dsplit_zw)
bool llama_dsplit_in_pass_loop_fork();   // llama-context.cpp: the pass was sized for all cards by the forced pass loop
std::vector<double> llama_dsplit_bal_shares_fork(int64_t n_tokens, const std::vector<ggml_backend_dev_t> & dev) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    g_bal_used = true;
    std::vector<double> s = dsplit_zw_shares(dev);
    // A pass cut by the shares (not one the forced pass loop sized for all cards): equal DENSE time per card. Each card's tokens are proportional to 1 / (its
    // measured dense us per token + its hand-offs per token); the experts run on the expert card after the dense ranges anyway. The zero-wait rule above
    // starved the other cards here: it shrinks their factor while the expert card waits, but that wait (1.4-1.6 s a pass on GLM 3.0) did not come from
    // their tokens (GLM 3.0 60k: factor 0.75 -> 0.33, cuda0x4 24% of the tokens, 2.9 s dense of a 16 s pass, idle 80%; both cards ~1.1 ms per token dense).
    // Before every card has a measured range (the first pass), the zero-wait shares stand.
    // In the forced pass loop too (2026-10-25): the expert card keeps its whole card limit and every other card gets the tokens it finishes in the same dense
    // time, so the pass is limit / the expert card's share. The zero-wait shares starved the other cards there as well (Qwen3.8 60k forced loop 1330-1358
    // prompt t/s against 1910 for the share-cut split with these shares).
    if (dev.size() >= 2) {
        const dsplit_bal & B = dsplit_bal_get(dsplit_bucket_of(n_tokens), dev);
        std::vector<double> r(dev.size(), 0.0);
        double sum = 0.0;
        bool ok = B.c.size() == dev.size();
        for (size_t i = 0; ok && i < dev.size(); ++i) {
            const double per = B.c[i] + (i < B.hop.size() ? B.hop[i] : 0.0);
            ok = per > 0.0;
            if (ok) { r[i] = 1.0 / per; sum += r[i]; }
        }
        if (ok && sum > 0.0) {
            for (size_t i = 0; i < dev.size(); ++i) { s[i] = r[i] / sum; }
            // the climb's offset on the other cards' share (the expert card, last, takes the rest)
            const double h = 1.0 - s.back(), h2 = dsplit_climb_on() ? std::min(0.95, std::max(0.02, h + g_climb.adj)) : h;
            for (size_t i = 0; i + 1 < dev.size(); ++i) { s[i] *= h > 0.0 ? h2 / h : 0.0; }
            s.back() = 1.0 - h2;
        }
    }
    g_climb.cut = !llama_dsplit_in_pass_loop_fork();
    if (s.size() != dev.size()) { s.assign(dev.size(), dev.empty() ? 0.0 : 1.0 / (double) dev.size()); }   // fewer than two cards: nothing to share
    dsplit_bal_get(dsplit_bucket_of(n_tokens), dev).s = s;   // the shares the pass report shows
    return s;
}
// The pass of a forced dense split (--moe split-pass=1, research, user 2026-10-24). Until now a split pass was cut by the ONE-card pass limit (what one card's
// buffers hold), so with two cards each computed about half of what it holds, for every upload of the experts. Now the expert card (the fastest link,
// last of the cards) takes the whole limit and every other card limit x its share / the expert card's share (what it keeps up with in the same time: a card that keeps up with 90%
// of the fastest gets 0.9 x the limit), so a full pass is limit / largest share; a prompt runs as full passes and one remainder pass split by the balance.
// ponytail: one limit for every card (the one-card limit of the context); cards with different free VRAM would need a limit each.
// The most tokens one dense-split pass can hold: the expert card takes the card limit and every other card at most that (llama_dsplit_pass_size_fork starts
// from limit x cards). A sliding-window KV cache (MiMo: 39 of 48 layers) holds the window plus the cells of one pass, which it writes before it attends: made for
// one card's micro-batch (16384) it refused the split pass of 27623 tokens ("failed to find a memory slot for batch of size 27623", MiMo 60k prompt, 2026-10-24).
// Its extra cells (1.5 GiB per card on MiMo) come out of the expert cache: the KV zone lies over expert blocks of any class (no zero blocks between the classes).
// Only a FORCED dense split (--moe dense-split=1) runs passes beyond one card's micro-batch (llama_context::decode, split-pass); the automatic choice cuts the
// ubatch into token ranges inside it, so the window cache needs n_swa + n_ubatch cells there. Sized for the split pass always (spass29..32) it cost every MiMo
// start 1.5 GiB of expert cache per card for nothing: decode 29.4 -> 26.6 t/s on mimo-i (2026-10-24).
// Cards with copies only; a model without a split gets n_ubatch.
uint32_t llama_dsplit_max_pass_fork(const llama_model * model, uint32_t n_ubatch) {
    const char * e = llama_moe_get_opt("dense-split");
    const char * sp = llama_moe_get_opt("split-pass");   // 0 = the old one-card pass: no pass beyond the micro-batch, so the small ring (A/B of the pass size, ring included)
    if (!e || strcmp(e, "1") != 0 || (sp && strcmp(sp, "0") == 0)) { return n_ubatch; }
    // Max pass by default, except for models with sliding-window layers (MiMo): their ring doubles with the pass, which costs ~1.5 GiB of expert cache per card
    // and attention over the bigger ring; 60k prompt, forced split: 16k pass 865-886 t/s vs 32k pass 684-758 (MiMo), while Qwen3.8 gained +7% from the 32k pass.
    for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) { if (model->hparams.is_swa(il)) { return n_ubatch; } }
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_sets.find(model);
    if (it == g_sets.end() || it->second.failed || it->second.resident || it->second.mirror.size() < 2) { return n_ubatch; }
    return n_ubatch * (uint32_t) (it->second.mirror.size() - std::min(it->second.bad.size(), it->second.mirror.size() - 1));
}
void     llama_dsplit_note_limit_fork(uint32_t cap) { g_card_lim.store(cap); }
void     llama_dsplit_note_card_max_fork(uint32_t cap) { g_card_max.store(cap); }
uint32_t llama_dsplit_card_lim_fork() { return g_card_lim.load(); }
uint32_t llama_dsplit_pass_size_fork(uint32_t cap) {
    std::vector<ggml_backend_dev_t> devs;
    llama_dsplit_cards_fork(devs);
    if (devs.size() < 2 || cap == 0) { return cap; }
    // The shares are kept per pass size (bucket): the pass is sized with the shares of the size it comes out at, as ds_plan reads them. Each bucket
    // gives one size, so the walk ends at a size that maps to itself or at the first size seen twice (two buckets pointing at each other).
    int64_t T = (int64_t) cap * (int64_t) devs.size();
    std::vector<int64_t> seen;
    for (;;) {
        const std::vector<double> s = llama_dsplit_bal_shares_fork(T, devs);
        const double se = s.empty() ? 0.0 : s.back();   // the expert card (the fastest link, last of the cards): it takes the whole limit, every pass
        if (se <= 0.0) { return cap; }
        double w = 0.0;
        for (size_t i = 0; i < s.size(); ++i) { w += i + 1 == s.size() ? (double) cap : std::min((double) cap, std::floor((double) cap * s[i] / se + 0.5)); }
        const int64_t W = std::max<int64_t>(cap, (int64_t) w);
        if (W == T || std::find(seen.begin(), seen.end(), W) != seen.end()) { T = W; break; }
        seen.push_back(T);
        T = W;
    }
    g_pass_lim.store((uint32_t) T);
    return (uint32_t) T;
}
// the cards of the split graph just built and the tokens of each (0: no range this pass), in range order: the balance relates each card's time to its tokens
void llama_dsplit_note_ranges_fork(const std::vector<ggml_backend_dev_t> & dev, const std::vector<int64_t> & tok) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    g_rng_dev = dev;
    g_rng_tok = tok;
    g_rng_seq++;
    g_rng_t0 = ggml_time_us();
}

// One balance step per pass (2026-10-23): the report below runs once per llama_decode call and saw only the last pass of it (a 30k prompt = 4 passes gave 2 steps).
// llama_context::process_ubatch asks before it builds the next pass: was the pass just run a split one, and if so take its card times now.
bool llama_dsplit_prev_split_fork() {
    return g_cur == 1;
}
void llama_dsplit_pass_done_fork() {
    std::lock_guard<std::mutex> pk(g_plan_mu);
    dsplit_pass_step_locked();
}

// a split pass with experts on the helpers did not fit (llama_context::process_ubatch): the experts go back to the expert card for passes of this size (the
// dense split alone) and the graph is built again, before the split itself is given up; true when there was something to drop
bool llama_dsplit_drop_experts_fork(int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    auto bt = g_bal.find(dsplit_bucket_of(n_tokens));
    if (bt == g_bal.end() || bt->second.f.empty()) { return false; }
    dsplit_bal & B = bt->second;
    double fh = 0.0;
    for (size_t i = 0; i + 1 < B.f.size(); ++i) { fh += B.f[i]; }
    if (fh <= 0.0) { return false; }
    std::fill(B.f.begin(), B.f.end(), 0.0);
    B.f.back() = 1.0;
    B.phase = 2;
    B.last = "the helpers' experts did not fit in memory: the dense split alone";
    LLAMA_LOG_WARN("split balance: a pass of %lld tokens with %.0f%% of the experts on the helpers does not fit: the experts stay on %s\n", (long long) n_tokens,
            100.0 * fh, ggml_backend_dev_name(B.dev.back()));
    g_tune_changed = true;
    return true;
}

// the share of layer il's experts that card dev computes in a split pass of n_tokens (llama-graph.cpp, the split controller's counts): the live balance's,
// 0 for a card it gives none; -1 when nothing sets it (the controller decides)
double llama_dsplit_card_f_fork(int il, int64_t n_tokens, ggml_backend_dev_t dev) {
    const int mode = llama_dsplit_experts_mode_fork();
    if (mode == 0) { return 0.0; }    // dense only
    if (mode == 1) { return -1.0; }   // the old start and controller
    {
        std::lock_guard<std::mutex> lk(g_plan_mu);
        if (g_bal_used) {
            auto bt = g_bal.find(dsplit_bucket_of(n_tokens));
            if (bt == g_bal.end()) { return 0.0; }
            for (size_t i = 0; i < bt->second.dev.size(); ++i) { if (bt->second.dev[i] == dev) { return bt->second.f[i]; } }
            return 0.0;
        }
    }
    return llama_dsplit_plan_helper_f_fork(il, n_tokens);   // --moe split-balance=0: the layer tuner's
}

double llama_dsplit_plan_helper_f_fork(int il, int64_t n_tokens) {
    const int mode = llama_dsplit_experts_mode_fork();
    if (mode == 0) { return 0.0; }    // dense only
    if (mode == 1) { return -1.0; }   // the old start and controller
    // auto (2026-10-23): whole MoE layers, the first K of g_layer_order, give their experts to the helper (1: all of the layer's experts but the one the owner
    // keeps); every other layer none. K comes from the layer tuner (phase 0 of dsplit_tune_step) and starts at 0: the dense split alone.
    std::lock_guard<std::mutex> lk(g_plan_mu);
    if (g_bal_used) {   // the live balance: the first helper's share (llama_dsplit_card_f_fork gives every card's)
        auto bt = g_bal.find(dsplit_bucket_of(n_tokens));
        return bt != g_bal.end() && !bt->second.f.empty() ? bt->second.f[0] : 0.0;
    }
    if (il < 0 || il >= (int) g_layer_rank.size() || g_layer_rank[il] < 0) { return 0.0; }
    auto tt = g_tune.find(dsplit_bucket_of(n_tokens));
    const int K = tt != g_tune.end() ? tt->second.K : 0;
    return g_layer_rank[il] < K ? 1.0 : 0.0;
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
    std::lock_guard<std::mutex> pk(g_plan_mu);
    int n_rng = 0;   // the cards that computed a range of the last split pass
    for (int64_t t : g_rng_tok) { n_rng += t > 0; }
    std::string out = format("\"split\":{\"gpus\":%d,\"devs\":\"%s\",\"prefill\":%d,\"prefill_gpus\":%d,\"why\":\"%s\"", g_ngpus, g_dev_all.c_str(), g_pf_choice,
             g_pf_choice == 1 ? (n_rng > 0 ? n_rng : g_ngpus) : g_pf_choice == 0 ? 1 : 0, why.c_str());
    // the live balance of the last measured split pass: every card's shares, dense and MoE times, the phase, what the last step did
    auto bt = g_bal_last >= 0 ? g_bal.find(g_bal_last) : g_bal.end();
    if (bt != g_bal.end() && !bt->second.dev.empty()) {
        const dsplit_bal & B = bt->second;
        std::string last = B.last;
        for (char & c : last) { if (c == '"' || c == '\\' || c < ' ') { c = ' '; } }
        static const bool detail = getenv("GGML_SCHED_PASS_STATS") && atoi(getenv("GGML_SCHED_PASS_STATS")) == 1;   // --moe pass-stats: the breakdown below is measured
        std::string zsrc = g_zw.src;
        for (char & c : zsrc) { if (c == '"' || c == '\\' || c < ' ') { c = ' '; } }
        out += format(",\"bal\":{\"t_ms\":%lld,\"phase\":\"%s\",\"n\":%d,\"us_tok\":%.2f,\"ref_us_tok\":%.2f,\"span_ms\":%.1f,\"cap\":%u,\"plim\":%u,\"detail\":%d,\"last\":\"%s\",\"zw_src\":\"%s\",\"cards\":[",
                g_bal_t_ms, dsplit_bal_phase(B.phase), B.n, B.v, B.ref_v, B.span, B.cap, B.plim, detail ? 1 : 0, last.c_str(), zsrc.c_str());
        const auto at = [](const std::vector<double> & v, size_t i) { return i < v.size() ? v[i] : 0.0; };
        // The cards' measured speeds against the best card's (user 2026-10-24: "make those scores 1 normalized ... 0.25 for upload and 0.8 for compute"): compute from the
        // probe's sustained matmul burst (power-capped clocks included), upload from the probed host-to-card rate; both measured once at server start, 1.00 = the best card
        std::vector<double> sc_c(B.dev.size(), 0.0), sc_u(B.dev.size(), 0.0);
        double mc = 0.0, mu = 0.0;
        for (size_t i = 0; i < B.dev.size(); ++i) {
            const auto * pr = llama_gpu_probe_find_fork(B.dev[i]);
            sc_c[i] = pr ? (pr->compute_sust > 0.0 ? pr->compute_sust : pr->compute) : 0.0;
            sc_u[i] = dsplit_link_once(B.dev[i]);
            mc = std::max(mc, sc_c[i]); mu = std::max(mu, sc_u[i]);
        }
        for (size_t i = 0; i < B.dev.size(); ++i) { sc_c[i] = mc > 0.0 ? sc_c[i] / mc : 0.0; sc_u[i] = mu > 0.0 ? sc_u[i] / mu : 0.0; }
        for (size_t i = 0; i < B.dev.size(); ++i) {
            out += format("%s{\"dev\":\"%s\",\"experts\":%s,\"tok\":%lld,\"s\":%.4f,\"s_next\":%.4f,\"f\":%.4f,\"f_next\":%.4f,\"dense_ms\":%.1f,\"dense_us_tok\":%.2f,"
                    "\"hop_us_tok\":%.2f,\"kv_ms\":%.1f,\"moe_ms\":%.1f,\"wait_dense_ms\":%.1f,\"wait_moe_ms\":%.1f,\"wait_max_ms\":%.2f,\"link_gbs\":%.2f,\"lim_pct\":%.1f,\"up_ms\":%.1f,"
                    "\"upw_ms\":%.1f,\"xw_ms\":%.1f,\"stall_ms\":%.1f,\"link_ms\":%.1f,\"gb_up\":%.3f,\"gb_host\":%.3f,\"gb_in\":%.3f,\"gb_out\":%.3f,\"sc_compute\":%.3f,\"sc_upload\":%.3f,\"zw_k\":%.3f,\"zw_f\":%.3f}", i ? "," : "", ggml_backend_dev_name(B.dev[i]), i + 1 == B.dev.size() ? "true" : "false",
                    (long long) (i < B.tok.size() ? B.tok[i] : 0), B.s_run[i], B.s[i], B.f_run[i], B.f[i], B.d[i], B.c[i], B.hop[i], B.k[i] / 1e3, B.m[i], B.wd[i], B.wm[i], B.wx[i], dsplit_link_once(B.dev[i]),
                    g_card_max.load() > 0 && i < B.tok.size() ? 100.0 * (double) B.tok[i] / (double) g_card_max.load() : 0.0, at(B.up, i),   // lim_pct: of the context's card limit   // link_gbs: this start's probe (a swapped or upgraded card is measured again)
                    at(B.uw, i), at(B.xw, i), at(B.st, i), at(B.ln, i), at(B.gb_up, i), at(B.gb_host, i), at(B.gb_in, i), at(B.gb_out, i), sc_c[i], sc_u[i], g_zw.dev == B.dev && i < g_zw.k.size() ? g_zw.k[i] : 0.0, g_zw.dev == B.dev && i < g_zw.f.size() ? g_zw.f[i] : 0.0);
        }
        out += "]}";
    }
    // the pass running now (the ranges of the split graph built last), before it is measured: run.py shows every pass as it starts
    if (!g_rng_dev.empty() && g_rng_dev.size() == g_rng_tok.size()) {
        out += format(",\"cur\":{\"seq\":%lld,\"age_ms\":%.0f,\"cap\":%u,\"cap_max\":%u,\"pass_lim\":%u,\"cards\":[", (long long) g_rng_seq, (ggml_time_us() - g_rng_t0) / 1e3, g_card_lim.load(), g_card_max.load(), g_pass_lim.load());
        for (size_t i = 0; i < g_rng_dev.size(); ++i) {
            out += format("%s{\"dev\":\"%s\",\"tok\":%lld}", i ? "," : "", ggml_backend_dev_name(g_rng_dev[i]), (long long) g_rng_tok[i]);
        }
        out += "]}";
    }
    return out + "}";
}

// Every split decision looks at the cards' identity and links again (vendor, device, slot, PCIe generation x width: sysfs reads, nothing touches the cards). A
// card swapped or upgraded between runs is a new key at load, so the stored shares of the old hardware are never used (dsplit_state_key); this catches a change
// while the server runs (a link retrained to another width, a card gone): the link rates, the card order, the plans, the balance and the tuner are recalculated
// from the cards as they are now.
static void dsplit_hw_check(const llama_model & model) {
    if (g_hw_key.empty()) { return; }
    static std::string base;   // the key without the generation, as of the first decision
    const std::string k = dsplit_hw_key(model, false, false);
    if (base.empty()) { base = k; }
    if (k == base) { return; }
    LLAMA_LOG_WARN("split planner: the cards or their links changed (%s -> %s): link rates, card order, plans and balance are recalculated\n", base.c_str(), k.c_str());
    base = k;
    g_hw_key = dsplit_hw_key(model);
    dsplit_link_forget();
    g_card_order.clear();
    std::lock_guard<std::mutex> pk(g_plan_mu);
    g_plans.clear(); g_bal.clear(); g_tune.clear();
}

// A curve of us/token over the context, fitted to the buckets' medians (user 2026-10-25: "store all per token times in an array, then extract the linear and
// the nonlinear part: the token number where it changes and its term"): one line a + b c, or two lines with a break at c0 - the sparse-attention gate makes
// a token's cost drop there (GLM 3.0: 1302 us/token at 24-33k cells, 924 at 33-42k). The break is kept only when it beats the single line by AIC (n ln(SSE/n)
// + 2k, k the parameters: 2 for a line, 5 for two lines and the break), so noise between buckets does not make one; at least 3 buckets on each side.
struct dsf_fit {
    bool   ok = false;
    double a = 0.0, b = 0.0, c0 = -1.0, a2 = 0.0, b2 = 0.0;
    double at(double c) const { return c0 >= 0.0 && c >= c0 ? a2 + b2 * c : a + b * c; }
};
static void dsf_line(const std::vector<std::pair<double, double>> & p, size_t i0, size_t i1, double & a, double & b, double & sse) {
    const double n = (double) (i1 - i0);
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = i0; i < i1; ++i) { sx += p[i].first; sy += p[i].second; sxx += p[i].first * p[i].first; sxy += p[i].first * p[i].second; }
    const double den = n * sxx - sx * sx;
    b = den > 0.0 ? (n * sxy - sx * sy) / den : 0.0;
    a = (sy - b * sx) / n;
    sse = 0.0;
    for (size_t i = i0; i < i1; ++i) { const double r = p[i].second - (a + b * p[i].first); sse += r * r; }
}
static dsf_fit dsf_fit_curve(const std::map<int, dsf_pt> & m) {
    std::vector<std::pair<double, double>> p;   // (the context at the bucket's middle, its median us/token), in context order
    for (const auto & e : m) { if (e.second.n > 0) { p.push_back({ ((double) e.first + 0.5) * (double) DSF_CELLS, e.second.us }); } }
    dsf_fit f;
    if (p.empty()) { return f; }
    f.ok = true;
    if (p.size() == 1) { f.a = p[0].second; return f; }
    double sse = 0.0;
    dsf_line(p, 0, p.size(), f.a, f.b, sse);
    const double n = (double) p.size();
    auto aic = [&](double e, int k) { return n * std::log(std::max(e, 1e-9) / n) + 2.0 * k; };
    double best = aic(sse, 2);
    for (size_t k = 3; k + 3 <= p.size(); ++k) {
        double a1, b1, s1, a2, b2, s2;
        dsf_line(p, 0, k, a1, b1, s1);
        dsf_line(p, k, p.size(), a2, b2, s2);
        const double v = aic(s1 + s2, 5);
        if (v < best) { best = v; f.a = a1; f.b = b1; f.c0 = 0.5 * (p[k - 1].first + p[k].first); f.a2 = a2; f.b2 = b2; }
    }
    return f;
}
static std::string dsf_fit_str(const dsf_fit & f) {
    if (!f.ok) { return "-"; }
    if (f.c0 < 0.0) { return format("%.1f + %.5f c", f.a, f.b); }
    return format("%.1f + %.5f c below %.0fk cells, %.1f + %.5f c above", f.a, f.b, f.c0 / 1024.0, f.a2, f.b2);
}
// the choice for a pass from the learned factors (-1: not enough measured); under g_mu
static int dsf_decide(const llama_model & model, int64_t n_tokens) {
    std::lock_guard<std::mutex> dl(g_dsf_mu);
    dsf_load();
    if (g_cards.size() < 2 || g_dsf_next_pos0 < 0) { return -1; }
    const dsf_fit f1 = dsf_fit_curve(g_dsf_one), f2 = dsf_fit_curve(g_dsf_split);
    std::map<std::string, dsf_fit> fd;   // every card's dense curve (split passes)
    for (const auto & e : g_dsf_dense) { fd[e.first] = dsf_fit_curve(e.second); }
    const std::string xdev = ggml_backend_dev_name(g_cards.back());
    // the dense us a split saves per token at context c: equal dense time on every card (see dsf)
    auto sigma = [&](double c) -> double {
        auto ix = fd.find(xdev);
        if (ix == fd.end() || !ix->second.ok) { return -1.0; }
        const double dx = ix->second.at(c);
        if (dx <= 0.0) { return -1.0; }
        double inv = 1.0 / dx;
        int others = 0;
        for (const auto & e : fd) {
            if (e.first == xdev || !e.second.ok) { continue; }
            const double d = e.second.at(c);
            if (d <= 0.0) { continue; }
            auto h = g_dsf_hop.find(e.first);
            inv += 1.0 / (d + (h != g_dsf_hop.end() ? h->second.us : 0.0));
            others++;
        }
        return others ? dx - 1.0 / inv : -1.0;
    };
    if (!f1.ok && !f2.ok) { return -1; }
    const char * src = f1.ok && f2.ok ? "both fitted" : f1.ok ? "split from the cards' dense curves" : "one card from the cards' dense curves";
    if ((!f1.ok || !f2.ok) && sigma((double) g_dsf_next_pos0) < 0.0) { return -1; }
    auto one_at   = [&](double c) { return f1.ok ? f1.at(c) : f2.at(c) + sigma(c); };
    auto split_at = [&](double c) { return f2.ok ? f2.at(c) : f1.at(c) - sigma(c); };
    // the passes left in this call (passes of n_tokens, the last one what remains), each max(its expert upload, its compute): the dense split shortens the
    // compute only, so it gains nothing on a pass whose upload is longer (user 2026-10-25: upload against compute decides; Qwen3.8 2k: split -40%, 60k: +28%)
    const int64_t end = std::max<int64_t>((int64_t) llama_moe_cache_pp_cells_get_fork(), g_dsf_next_pos0 + n_tokens);
    const int64_t R = end - g_dsf_next_pos0;
    const uint32_t k_used = std::max<uint32_t>(1, model.hparams.n_expert_used_max()), n_exp = std::max<uint32_t>(1, model.hparams.n_expert);
    double t_one = 0.0, t_split = 0.0, u_first = -1.0;
    for (int64_t c = g_dsf_next_pos0; c < end; c += n_tokens) {
        const int64_t T = std::min<int64_t>(n_tokens, end - c);
        const double cm = (double) c + 0.5 * (double) T;
        const double U = std::max(0.0, llama_moe_cache_pass_upload_ms_fork((uint32_t) T, k_used, n_exp)) * 1e3;   // us
        if (u_first < 0.0) { u_first = U; }
        t_one   += std::max(U, (double) T * one_at(cm));
        t_split += std::max(U, (double) T * split_at(cm));
    }
    const double fill_us = g_mirror_fill_us.load() > 0.0 ? (g_dsf_fill.n ? g_dsf_fill.us * 1e3 : g_mirror_fill_us.load()) : 0.0;   // copies to fill first (0: valid)
    const double gain = t_one - t_split;
    const int c = gain - fill_us > llama_moe_live_get("split_min_gain") * t_one ? 1 : 0;
    static int logged = 0;
    if (logged++ < 64) {
        LLAMA_LOG_WARN("split factors: pass of %lld tokens from %lldk cells, %lld tokens left (%s): compute per token one card %s; split %s; expert upload %.0f ms "
                       "per pass; over the passes left one card %.0f ms, split %.0f ms + copy fill %.0f ms: %s\n", (long long) n_tokens,
                (long long) (g_dsf_next_pos0 / 1024), (long long) R, src, dsf_fit_str(f1).c_str(), dsf_fit_str(f2).c_str(), u_first / 1e3,
                t_one / 1e3, t_split / 1e3, fill_us / 1e3, c ? "split" : "one card");
    }
    return c;
}
// every prompt pass is timed: the pass decided last ends at the synchronize before the next prompt pass and at the end of its call (llama_context)
void llama_dsplit_note_pass_fork(int64_t pos0) { std::lock_guard<std::mutex> dl(g_dsf_mu); g_dsf_next_pos0 = pos0; }
bool llama_dsplit_pass_open_fork() { std::lock_guard<std::mutex> dl(g_dsf_mu); return g_dsf_ps.choice >= 0; }
void llama_dsplit_pass_end_fork() {
    std::lock_guard<std::mutex> dl(g_dsf_mu);
    const dsf_pass p = g_dsf_ps;
    g_dsf_ps = dsf_pass();
    if (p.choice < 0 || p.T < 64 || p.pos0 < 0 || p.t0 <= 0) { return; }
    dsf_load();
    double us = (double) (ggml_time_us() - p.t0) - p.fill_ms * 1e3;   // the copy fill is the split's per-prompt cost, kept apart (dsf.fill)
    // Only a compute-bound pass tells its compute: one that took no longer than its expert upload is upload-bound, its time per token is the upload's, not
    // the compute's (Qwen3.8 2k: 2441 tokens in 2.2 s against a 2.8 s upload estimate; recorded as compute it made one card look 988 us per token at the
    // start of the context, where big passes measure 450-650, and the split won the 4k request it lost by 20%). The decision adds the upload as max(U, T c).
    if (p.U_us > 0.0 && us <= p.U_us) { return; }
    // the compute side of the pass: less the time the expert card's stream waited for expert uploads (stream stamps of this pass, when it was measured).
    // An upload-bound pass then still tells its compute; the decision adds the upload back as max(upload, compute) for the pass size it plans.
    {
        dsplit_card_stat st[16]; long gen = 0;
        static long last_gen = -1;
        const int nst = ggml_backend_sched_card_stats(st, 16, &gen);
        if (gen != last_gen && p.xdev) {
            for (int j = 0; j < nst; ++j) { if (st[j].backend && ggml_backend_get_device(st[j].backend) == p.xdev) { us -= (double) st[j].up_wait_us; } }
        }
        last_gen = gen;
    }
    if (us <= 0.0) { return; }
    const int k = (int) ((p.pos0 + p.T / 2) / DSF_CELLS);
    dsf_pt & d = (p.choice == 1 ? g_dsf_split : g_dsf_one)[k];
    dsf_upd(d, us / (double) p.T);
    dsf_mark(std::string(p.choice == 1 ? "dsf.split.k" : "dsf.one.k") + std::to_string(k), d);
}
void llama_dsplit_note_fill_fork(double ms) {   // llama_context::moe_mirror_fill: the copies of this prompt were filled (part of the split pass decided last)
    std::lock_guard<std::mutex> dl(g_dsf_mu);
    dsf_load();
    g_dsf_ps.fill_ms += ms;
    dsf_upd(g_dsf_fill, ms);   // in memory only: before the first fill of a run the planner's estimate (copy bytes / link rate) stands
}

bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (n_tokens >= 64) {   // the load-time timing's lines, once (they were written while the parameter fit muted the log)
        std::lock_guard<std::mutex> pk(g_plan_mu);
        if (!g_bench_logged && !g_bench_log.empty()) { g_bench_logged = true; LLAMA_LOG_WARN("%s", g_bench_log.c_str()); }
    }
    dsplit_hw_check(model);
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
    if (it != g_sets.end()) { dsplit_cards_update(model, it->second); }   // the cards that can take a range now (copies with memory)
    // with copies: two or more cards that can take a range, every card's copies placed or left out (mirror-pool places them at the first decode); without
    // copies (no token-range graph for the model): the choice between one card and the expert split over two or more GPUs
    const bool managed = it != g_sets.end() && dsplit_n_gpus(model) >= 2 &&
                         (it->second.mirror.empty() ? (dsplit_opt_auto() && !it->second.resident) : (dsplit_copies_ready(it->second) && g_cards.size() >= 2));
    if (managed && n_tokens >= 64 && !it->second.mirror.empty()) {
        std::lock_guard<std::mutex> pk(g_plan_mu);
        dsplit_link_model(model, n_tokens);   // the balance's hand-off and KV numbers, in auto and forced mode alike
    }
    if (managed && n_tokens >= 64) {
        {   // the state file's section and hardware key for the split factors, also in forced runs (they teach the factors too)
            std::lock_guard<std::mutex> pk(g_plan_mu);
            dsplit_calib_load(model);
        }
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
            llama_dsplit_plan_fork plan = llama_dsplit_plan_get_fork(model, n_tokens);
            dsplit_bucket_load(b);   // what an earlier run measured on these cards in this size bucket
            dsplit_bucket_load_all();   // and in every other size, for the nearest-size fallback below
            // one card measured at this size: the split is weighed against that, not against the model's one-card time (2026-10-23, Qwen3.8-Flash-Next 3k:
            // the model said 3346 / 3369 ms, the pass just measured took 2457 / 2573 ms; the inflated one-card time made a slower split look 11-19% ahead)
            // The split's estimate is scaled by the same correction (measured over modelled one-card time): the model is trusted for the ratio of the two, not
            // their size. Replacing the one-card time alone compared a measured time against a modelled one that is off the same way (GLM 3.0 60k, 10368 tokens:
            // model one card 24022 / split 19790 ms, one card measured 18100 ms: split "slower", one card chosen; the split measured 677 against 570 prompt t/s).
            if (plan.valid && B.n[0] > 0) {
                const double t_one = B.us_tok[0] * (double) n_tokens / 1e3;
                if (plan.t_one_ms > 0.0) { plan.t_split_ms *= t_one / plan.t_one_ms; }
                plan.t_one_ms = t_one;
            }
            const bool plan_sure = plan.valid && std::fabs(plan.t_split_ms - plan.t_one_ms) >= 0.15 * plan.t_one_ms;
            // No trials by default: a pass is never run in the configuration the model expects to be slower just to find out. The planner decides; the split must be
            // predicted clearly ahead (8%) to be used, a close call stays on one card. --moe split-explore=1 brings the old trial-and-error back (the other
            // configuration once per size bucket, the slower one again every 16th pass).
            static const bool explore = [] { const char * e = llama_moe_get_opt("split-explore"); return e && atoi(e) != 0; }();
            if (B.n[0] > 0 && B.n[1] > 0) {
                // the split must win by split_min_gain on the passes measured, a tie or a close call stays on one card (never on while slower)
                const int best = B.us_tok[1] <= (1.0 - llama_moe_live_get("split_min_gain")) * B.us_tok[0] ? 1 : 0;
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
            // the factors learned from earlier requests decide whenever both configurations can be estimated at this pass's context (see dsf)
            {
                const int fc = dsf_decide(model, n_tokens);
                if (fc >= 0) {
                    choice = fc;
                    why = fc ? "factors: the split is faster at this context over the tokens left" : "factors: one card is faster at this context over the tokens left";
                }
            }
            B.passes++;
        }
    }
    // (Until 2026-10-25 a size needed a measured one-card pass before the split was tried: "never split on a prediction alone", after the model was 2.2 s per
    // pass off on Qwen Next. User 2026-10-25: the first pass takes the calculated split. The prediction is now the compute calculated from the probe until a pass
    // is timed, and once one card is measured the split's estimate is scaled by the same correction (the ratio is trusted, not the size), so a first pass
    // the model misjudges is corrected by the measured passes after it.)
    if (choice == 1 && n_tokens >= g_veto_tokens) {
        choice = 0;   // the split graph of this size did not fit behind the mirror copies once: one card
        why    = "one card: the split pass did not fit behind the copies";
    }
    g_cur = choice;
    g_ngpus = dsplit_n_gpus(model);
    if (choice >= 0 && n_tokens >= 64) {   // the prompt pass about to run: timed until the synchronize after it (dsf)
        // its expert upload, computed before the factors' lock (it reads the cache under the cache's own lock)
        const double U_us = llama_moe_cache_pass_upload_ms_fork((uint32_t) n_tokens, std::max<uint32_t>(1, model.hparams.n_expert_used_max()),
                                                                std::max<uint32_t>(1, model.hparams.n_expert)) * 1e3;
        std::lock_guard<std::mutex> dl(g_dsf_mu);
        g_dsf_ps = dsf_pass();
        g_dsf_ps.choice = choice; g_dsf_ps.T = n_tokens; g_dsf_ps.pos0 = g_dsf_next_pos0; g_dsf_ps.t0 = ggml_time_us();
        g_dsf_ps.xdev = g_cards.empty() ? nullptr : g_cards.back();
        g_dsf_ps.U_us = U_us;   // this pass's expert upload (<0: not known)
        g_dsf_next_pos0 = -1;
    }
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
    if (it != g_sets.end()) { dsplit_cards_update(model, it->second); }
    // --moe split-reserve=1 (research): reserve the compute buffers for the SPLIT graph at load although the copies are not placed yet (they are placed after the
    // context exists). Both cards get compute buffers; a split pass gives each card only its range's dense part, so the graph that fits at a micro-batch may
    // be bigger than the one-card graph (GLM-5.3-Flash on 2x RTX 3090: the one-card reserve at 16384 asks 29.4 GiB of the expert card and fails, 8192 is kept).
    static const bool early = [] { const char * e = llama_moe_get_opt("split-reserve"); return e && atoi(e) != 0; }();
    const bool ready = it != g_sets.end() && (it->second.mirror.empty() ? (dsplit_opt_auto() && !it->second.resident)
                                                                          : ((dsplit_copies_ready(it->second) || early) && g_cards.size() >= 2));
    g_cur = ready && dsplit_n_gpus(model) >= 2 ? 1 : -1;
}

// one measured split pass of size bucket b, us per token: the tuner's step (see dsplit_tune)
// a settled size (layer count and token share) per model, cards and build (dsplit_state_key): the next start uses it at once instead of measuring again
static void dsplit_tune_save(int b, const dsplit_tune & T) {
    if (!dsplit_state_on()) { return; }
    char v[64]; snprintf(v, sizeof v, "%d %.4f", T.K, T.s);
    moe_state_set(g_calib_section, { { dsplit_state_key("dsplit_tune2_b" + std::to_string(b)), v } });
}
static void dsplit_tune_load(int b) {
    if (!dsplit_state_on() || g_tune.count(b)) { return; }
    std::string v;
    if (!moe_state_get(g_calib_section, dsplit_state_key("dsplit_tune2_b" + std::to_string(b)), v)) { return; }
    int K = 0; double sh = -1.0;
    if (sscanf(v.c_str(), "%d %lf", &K, &sh) == 2 && K >= 0 && sh > 0.0 && sh < 1.0) {
        dsplit_tune & T = g_tune[b];
        T.K = T.best_K = K; T.s = T.best_s = sh; T.phase = 2;
        LLAMA_LOG_WARN("split tuner: passes of size bucket %d from the state file: the experts of %d MoE layers on the helper, %.0f%% of the tokens\n", b, K, 100.0 * sh);
    }
}

static void dsplit_drift_step(int b, dsplit_tune & T, double v, int kmax) {
    static const bool drift_on = [] { const char * e = llama_moe_get_opt("split-drift"); return !e || atoi(e) != 0; }();   // --moe split-drift=0: settled stays settled
    if (!drift_on) { return; }
    if (!T.dr_init) { T.dr_init = true; T.dr_base_s = T.s; T.dr_base_K = T.K; T.dr_probe = false; T.dr_nb = 0; }
    if (!T.dr_probe) {   // a pass of the base setting
        T.dr_bt = T.dr_nb == 0 ? v : 0.5 * (T.dr_bt + v); T.dr_nb++;
        if (T.dr_nb < (T.dr_ds <= 0.005 ? 8 : 2)) { return; }   // a step at its floor probes rarely: it costs a (slightly) worse pass each time
        for (int tries = 0; tries < 2; ++tries) {
            if (T.dr_lever == 0) {
                double ns = T.dr_base_s + T.dr_dir[0] * T.dr_ds;
                if (ns < 0.10 || ns > 0.90) { T.dr_dir[0] = -T.dr_dir[0]; ns = T.dr_base_s + T.dr_dir[0] * T.dr_ds; }
                T.s = std::max(0.10, std::min(0.90, ns)); T.K = T.dr_base_K;
            } else {
                int nk = T.dr_base_K + T.dr_dir[1];
                if (nk < 0 || nk > kmax) { T.dr_dir[1] = -T.dr_dir[1]; nk = T.dr_base_K + T.dr_dir[1]; }
                if (nk < 0 || nk > kmax) { T.dr_lever = 0; continue; }   // no layer lever (fixed experts or none)
                T.K = nk; T.s = T.dr_base_s;
            }
            break;
        }
        T.dr_probe = true; T.dr_np = 0; g_tune_changed = true;
        return;
    }
    T.dr_pt = T.dr_np == 0 ? v : 0.5 * (T.dr_pt + v); T.dr_np++;
    if (T.dr_np < 2) { return; }
    const int lv = T.dr_lever;
    if (T.dr_pt < T.dr_bt * 0.99) {   // taken: the new base, same direction again
        LLAMA_LOG_INFO("split drift: %s %.0f%% -> %.0f%% of the tokens, %d -> %d layers: %.1f -> %.1f prompt tokens/s\n", lv == 0 ? "share" : "layers", 100.0 * T.dr_base_s, 100.0 * T.s,
                T.dr_base_K, T.K, 1e6 / T.dr_bt, 1e6 / T.dr_pt);
        T.dr_base_s = T.s; T.dr_base_K = T.K; T.dr_bt = T.dr_pt; T.dr_nb = 2; T.dr_fail[lv] = 0;
        dsplit_tune_save(b, T);
    } else {   // not better: the other way next time on this lever; two misses in a row: a smaller step
        T.dr_dir[lv] = -T.dr_dir[lv];
        if (++T.dr_fail[lv] >= 2 && lv == 0) { T.dr_ds = std::max(0.005, T.dr_ds * 0.5); T.dr_fail[lv] = 0; }
        T.s = T.dr_base_s; T.K = T.dr_base_K; T.dr_nb = 1;
    }
    T.dr_probe = false; T.dr_lever = 1 - T.dr_lever; g_tune_changed = true;
}

static void dsplit_tune_step(int b, double v) {
    std::lock_guard<std::mutex> pk(g_plan_mu);
    auto pit = g_plans.find(b);
    if (pit == g_plans.end() || !pit->second.plan.valid) {
        return;
    }
    dsplit_tune & T = g_tune[b];
    if (T.phase >= 2) {
        dsplit_drift_step(b, T, v, llama_dsplit_experts_mode_fork() >= 0 ? 0 : (int) g_layer_order.size());
        return;
    }
    if (T.s < 0.0) { T.s = pit->second.plan.share; T.best_s = T.s; }
    T.n++;
    const bool better = T.best_t == 0.0 || v < T.best_t;
    if (better) { T.best_t = v; T.best_gx = T.gx; T.best_s = T.s; T.best_K = T.K; }
    const double gx0 = T.gx, s0 = T.s;
    const int    K0  = T.K;
    auto next_phase = [&]() {
        T.gx = T.best_gx; T.s = T.best_s; T.K = T.best_K;
        T.phase++; T.dir = -1; T.step = T.phase == 1 ? 0.08 : 0.0;
        if (T.phase == 1) { T.s = std::max(0.10, std::min(0.90, T.best_s - T.step)); }
    };
    if (T.phase == 0 && (llama_dsplit_experts_mode_fork() >= 0 || g_layer_order.empty())) {
        next_phase();   // split-experts fixes the helper's experts, or the model has no MoE layers: nothing to count
    } else if (T.phase == 0) {
        // the layer count: K just measured. Growing (no slower count yet): 1, 2, 4 ... more layers while the pass gets faster. Then bisect between the best
        // count and the first slower one, until they are neighbours.
        const int kmax = (int) g_layer_order.size();
        int next = -1;
        if (T.k_hi < 0) {
            if (better && T.K < kmax) { next = std::min(kmax, T.K + T.k_step); T.k_step *= 2; }
            else if (!better)         { T.k_hi = T.K; }
        }
        if (T.k_hi >= 0) {
            if (!better && T.K < T.k_hi) { T.k_hi = T.K; }
            const int mid = (T.best_K + T.k_hi) / 2;
            next = (mid > T.best_K && mid < T.k_hi) ? mid : -1;
        }
        if (next < 0) { next_phase(); } else { T.K = next; }
    } else if (T.phase == 1) {
        if (!better) { T.dir = -T.dir; T.step *= 0.5; }
        if (T.step < 0.02) { next_phase(); }
        else { T.s = std::max(0.10, std::min(0.90, T.best_s + T.dir * T.step)); }
    }
    if (T.n >= 16 && T.phase < 2) { T.gx = T.best_gx; T.s = T.best_s; T.K = T.best_K; T.phase = 2; }
    if (T.phase >= 2) {
        LLAMA_LOG_WARN("split tuner: passes of up to %lld tokens settled after %d measurements: helper %.0f%% of the tokens, the experts of %d of %zu MoE layers on it, %.1f prompt tokens/s\n",
                (long long) pit->second.n_tokens, T.n, 100.0 * T.s, T.K, g_layer_order.size(), 1e6 / T.best_t);
        dsplit_tune_save(b, T);
    }
    if (T.gx != gx0 || T.s != s0 || T.K != K0) { g_tune_changed = true; }
}

void llama_dsplit_report_fork(int64_t n_tokens, int64_t us) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_pending == 1 && n_tokens > 0 && us > 0) {   // the live balance (and the cards' times for /moe/stats)
        std::lock_guard<std::mutex> pk(g_plan_mu);
        dsplit_pass_step_locked();   // (the last pass of the call; the passes before it were taken by llama_dsplit_pass_done_fork)
    }
    llama_dsplit_pass_end_fork();   // the call's last prompt pass ends here (synchronized): its time per token for the split factors
    { std::lock_guard<std::mutex> dl(g_dsf_mu); dsf_flush(); }   // and the call's new samples go to the state file, once
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
                dsplit_calib_save(nc);
                const int bk = dsplit_bucket_of(g_pend_max);   // and this size's own factor
                auto ct = g_calib_split_b.find(bk);
                g_calib_split_b[bk] = ct == g_calib_split_b.end() ? r : 0.5 * ct->second + 0.5 * r;
                dsplit_calib_b_save("dsplit_calib5_b", bk, g_calib_split_b[bk]);
                pit->second.c_us = -1.0;   // recompute the plan of this size with the new calibration
            }
        } else {
            // one card: this size's measured over the model's one-card time
            std::lock_guard<std::mutex> pk(g_plan_mu);
            const int bk = dsplit_bucket_of(g_pend_max);
            auto rt = g_one_raw_b.find(bk);
            if (rt != g_one_raw_b.end() && rt->second > 0.0) {
                const double r1 = std::max(0.3, std::min(3.0, v / rt->second));
                auto ct = g_calib_one_b.find(bk);
                g_calib_one_b[bk] = ct == g_calib_one_b.end() ? r1 : 0.5 * ct->second + 0.5 * r1;
                dsplit_calib_b_save("dsplit_calib1_b", bk, g_calib_one_b[bk]);
                auto pit = g_plans.find(bk);
                if (pit != g_plans.end()) { pit->second.c_us = -1.0; }   // recompute the plan of this size with it
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
        std::string mode = "one-card", gpus = g_dev_off.empty() ? "?" : g_dev_off, skipped = g_dev_rest.empty() ? "-" : g_dev_rest, by = g_why, cards = "-";
        double tok_share = 0.0, exp_share = 0.0;
        if (g_pending == 1 || g_pending == -2) {
            const int b = dsplit_bucket_of(g_pend_max);
            std::lock_guard<std::mutex> pk(g_plan_mu);
            const auto pit = g_plans.find(b);
            const auto tt  = g_tune.find(b);
            const auto bt  = g_bal_used && !g_rng_dev.empty() ? g_bal.find(dsplit_bucket_of([&] { int64_t s = 0; for (int64_t t : g_rng_tok) { s += t; } return s; }())) : g_bal.end();
            const bool have_plan = pit != g_plans.end() && pit->second.plan.valid;
            const double gx = tt != g_tune.end() ? tt->second.gx : 1.0;
            const int em = llama_dsplit_experts_mode_fork();
            if (bt != g_bal.end() && !bt->second.dev.empty()) {
                // the live balance's shares of the last pass: the helpers' (every card but the expert card) together, and every card's own
                const dsplit_bal & B = bt->second;
                tok_share = 1.0 - B.s_run.back();
                exp_share = em == 1 ? -1.0 : 1.0 - B.f_run.back();
                cards.clear();
                for (size_t i = 0; i < B.dev.size(); ++i) {
                    cards += format("%s%s:%.2f:%.2f", i ? "," : "", ggml_backend_dev_name(B.dev[i]), B.s_run[i], B.f_run[i]);
                }
            } else {
                tok_share = tt != g_tune.end() && tt->second.s > 0.0 ? tt->second.s : have_plan ? pit->second.plan.share : 0.55;
                if (em == 0) {
                    exp_share = 0.0;
                } else if (em == 1 || !have_plan || pit->second.f_help.empty()) {
                    exp_share = -1.0;
                } else {
                    double s = 0.0;
                    for (double f : pit->second.f_help) { s += std::min(1.0, f * gx); }
                    exp_share = s / (double) pit->second.f_help.size();
                }
            }
            mode    = g_pending == -2 ? "mixed" : exp_share == 0.0 ? "dense-split" : "dense+expert-split";
            gpus    = g_dev_all.empty() ? "?" : g_dev_all;
            skipped = "-";
            if (bt != g_bal.end()) { by += format(", balance: %s", dsplit_bal_phase(bt->second.phase)); }
            else if (tt != g_tune.end() && tt->second.phase >= 2) { by += ", tuner settled"; } else if (tt != g_tune.end() && tt->second.n > 0) { by += ", tuner measuring"; }
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
        // cards=NAME:token share:expert share,... (every card of the last split pass, in range order, the expert card last)
        LLAMA_LOG_WARN("prefill decision: prompt_tokens=%lld passes=%u pass_tokens=%u staging_slots=%d mode=%s gpus=%s skipped=%s helper_token_share=%.2f "
                "helper_expert_share=%.2f cards=%s prompt_tps=%.1f one_card_tps=%s split_tps=%s chosen_by=\"%s\"\n", (long long) n_tokens, passes, pass, slots, mode.c_str(),
                gpus.c_str(), skipped.c_str(), tok_share, exp_share, cards.c_str(), (double) n_tokens * 1e6 / (double) us, one.c_str(), spl.c_str(), by.c_str());
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
