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
    const auto * pm = llama_gpu_probe_find_fork(dm);
    for (int i = 0; i < n; ++i) {
        B[i] = dsplit_link_once(cards[i]);
        if (B[i] <= 0.0) {
            return;
        }
        const auto * pi = llama_gpu_probe_find_fork(cards[i]);
        if (pm && pi && pm->compute > 0 && pi->compute > 0) { rel[i] = pi->compute / pm->compute; }
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
    const double att_tok = std::min(c_us, std::max(0.0, llama_moe_cache_compute_att_us_fork()));
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
std::vector<double> llama_dsplit_bal_shares_fork(int64_t n_tokens, const std::vector<ggml_backend_dev_t> & dev) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    g_bal_used = true;
    std::vector<double> s = dsplit_zw_shares(dev);
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

bool llama_dsplit_decide_fork(const llama_model & model, int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_mu);
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
            if (plan.valid && B.n[0] > 0) { plan.t_one_ms = B.us_tok[0] * (double) n_tokens / 1e3; }
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
            B.passes++;
        }
    }
    {
        // Never split on a prediction alone: this size needs a measured one-card pass to compare with (2026-10-23: the model was 2.2 s per pass off on Qwen Next).
        // The first pass of a size runs on one card; the split is tried once the planner, now scaled by that measurement, still puts it clearly ahead.
        const auto bk = g_buckets.find(dsplit_bucket_of(n_tokens));
        if (choice == 1 && dsplit_opt_auto() && managed && (bk == g_buckets.end() || bk->second.n[0] == 0)) {
            choice = 0;
            why    = "one card first: no measured one-card pass at this size to compare the split with";
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
