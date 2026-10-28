#include "llama-graph.h"
#include "llama-passplan-fork.h"
#include "llama-splitplan-fork.h"

#include <typeinfo>

#include "llama-moecache-fork.h"
#include "llama-dsplit-fork.h"

#include "llama-impl.h"
#include "llama-moestate-fork.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <map>
#include <mutex>
#include <tuple>

int32_t llama_dsplit_sparse_min_fork();   // llama-dsplit-fork.cpp: the measured K/V length from which the sparse attention gather runs (0: backend rule)

// fork (llama-dsplit-fork.cpp): the cards of a dense split pass and their shares, declared here so that no shared header changes
void   llama_dsplit_cards_fork(std::vector<ggml_backend_dev_t> & out);   // the cards that take a range, the expert card last (empty: no split)
std::vector<size_t> llama_dsplit_attn_order_fork(const std::vector<ggml_backend_dev_t> & range_devs, const ggml_tensor * cache);   // ranges' attention order: the cache's card last

// fork: the index in ds_r of the expert card's range (the card that computes the experts: the last of llama_dsplit_cards_fork). Where it sits in token order
// is ds_plan's choice (first), so the builders ask here instead of assuming a position. The expert card always has a range.
size_t llama_ds_xr_fork(const std::vector<llm_graph_context::ds_range_fork> & r) {
    std::vector<ggml_backend_dev_t> cards;
    llama_dsplit_cards_fork(cards);
    for (size_t i = 0; i < r.size() && !cards.empty(); ++i) {
        if (r[i].dev == cards.back()) { return i; }
    }
    return r.empty() ? 0 : r.size() - 1;
}
std::vector<double> llama_dsplit_bal_shares_fork(int64_t n_tokens, const std::vector<ggml_backend_dev_t> & dev);   // their token shares (the live balance)
uint32_t llama_dsplit_card_lim_fork();   // the card limit of the decode call in progress: the most tokens one card's range takes (0: not known)
void   llama_dsplit_note_ranges_fork(const std::vector<ggml_backend_dev_t> & dev, const std::vector<int64_t> & tok);   // the ranges a graph was built with
double llama_dsplit_card_f_fork(int il, int64_t n_tokens, ggml_backend_dev_t dev);   // a card's share of a layer's experts (-1: the controller decides)
int    llama_dsplit_hop_mode_range_fork();   // the dense split's hand-off precision (llama-dsplit-fork.cpp)
int    llama_dsplit_hop_mode_fork();

// ---- fork: share controller of the multi-GPU prompt split ---------------------------------------------------------------------------------------------------
// The routed experts of a layer are split over the cards in contiguous id ranges (see build_moe_ffn). How many experts each card holds decides who waits for
// whom: the layer's owner (the card with the dense weights) adds the partial sums, so it waits when a helper's sum arrives after its own chain has ended, and
// the helper idles when it arrives before. No formula gets this right for every model, card and link: the balance depends on each card's compute, the bytes
// of the layer's expert tensors (they differ between layers: mixed quantisations), the hop of the hidden state over the helper's link, the power caps.
// So it is measured and tuned, per layer, as the user specified (2026-10-13):
//   start   a card that is not on the fastest link starts with its upload-balanced share (its upload rate over the sum of all rates: every link finishes
//           the layer's uploads together), the cards on the fastest link share the rest. No fitted constants: the best share depends on the prompt
//           length AND on the model (expert bytes per layer, quantisation, dense-to-expert ratio) and the cards; the bisection below finds it per layer and
//           per pass size, and a size never seen starts from the nearest tuned size.
//   tune    after every prompt pass the scheduler reports, per layer and card, when the card's expert chain started and ended and when a helper's partial
//           sum had left it (time stamps in stream order, ggml_sched_chain_stat). delta = t_out(helper) - t_end(owner): positive, the owner waited, the
//           helper gets fewer experts; negative, the helper waited and gets more. One expert moved changes the gap by tau = the owner's time per expert
//           + the helper's time per expert, so the move that closes the gap is delta / tau experts: 75% of it is taken (--moe split-gain; the times per
//           expert include waits that do not scale with the count), at most `step` experts, never less than one. One pass's gap is good to about a
//           millisecond, which is several experts, so what is acted on is the mean of the last three estimates of the balance point (count - delta /
//           tau, --moe split-window), and nothing moves while that smoothed gap is inside half of tau or 1% of the layer time. `step` starts at 25% of the experts (--moe split-step) and is halved every time the wait changes sides, so a wrong
//           estimate still closes like a bisection. The move is taken from the count the measurement was made with: a measurement that arrives a pass
//           late cannot add up twice. (The first version moved a fixed 5% per pass whatever the gap: four passes to switch a helper off, each of them
//           with the wrong shares.)
//   apply   a pass asks for new counts before its graph is reused (llama_split_ctl_poll_fork, from llama_context::process_ubatch): the graph is rebuilt
//           only when a count changed, a pass with unchanged counts reuses the graph of the pass before. (Read while a graph was being built, as before,
//           new counts reached a pass only when something else forced a rebuild, and the same measurement was applied again on every reuse.)
//   settle  a pass size is settled when its measured layer time stays within 1.5% (--moe split-settle) for two measurements in a row, or after 8
//           measurements (--moe split-passes) whatever they say: the counts of the fastest measurement are taken and kept for the rest of the run (the
//           prompt length is what it is, the answer has been found). (Judged by the counts instead, a size never settled: timing jitter of a
//           millisecond moves a few experts in most layers on every pass, while the layer time has long stopped changing.)
//   cost    Qwen3.8-Flash-Next, 5.5k tokens, ten passes of the same prompt, prompt tokens/s: 1219, 1264, 1257, 1264, 1300, 1303, 1300 ... with the
//           tuning, 1222, 1260, 1264, 1284, 1280, 1279 ... with the start shares, 906 on one card. A rebuild costs 7 ms of a 4.2 s pass and no pass was
//           slower than with the start shares: the tuning is free, so nothing is stored between runs (a stored answer would have to be keyed by the
//           model, the cards and their slots, and would go stale with any of them). Every run measures and stops when it has settled.
//           --moe split-auto=0: no tuning, the start shares for the whole run.
//   skip    a helper whose count falls under one expert, or whose experts save the owner less time than the owner spends on the two copies with it, holds
//           nothing in that layer: no chain, no hop, no upload there (the go / no-go, per layer, measured). It is tried again 32 measured passes later.
//   length  the counts are kept per pass size (one bucket per power of two of the tokens in the pass): hops and compute grow with the tokens, the expert bytes
//           do not, so the balance moves with the prompt length. A size never seen starts from the nearest tuned size of the same layer and card.
//   link    a helper cannot hold more experts than its link uploads in one layer: (layer period - its hops) x upload rate / bytes per expert, with 15%
//           spare. The period is measured (the owner's chain start of two consecutive layers, minus what the owner waited). Short passes are bound by
//           this, not by compute: GLM-5.3-Flash at 2k tokens has a 108 ms layer and 8.3 MB per expert, a x4 link moves 23% of the experts in that time;
//           seeded with 35% the owner waited 70 ms per layer for the helper's uploads (316 prompt tokens per second against 406 on one card).
// Any number of cards; nothing refers to a slot width or to "two cards". Without the stamps (a backend that has no host function in stream order) the start
// values stay. --moe split-slow-share=F fixes the shares by hand (tests).
extern "C" {
    struct ggml_sched_chain_stat {      // must match ggml-backend.cpp
        int            layer;
        ggml_backend_t backend;
        int            lo, hi;
        int64_t        t_start, t_end, t_out;
        int64_t        n_tokens;
    };
    GGML_API int ggml_backend_sched_chain_stats(struct ggml_sched_chain_stat * out, int cap, long * gen);
}
namespace {

struct llama_split_ctl {
    std::mutex mu;
    long gen_seen  = 0;
    long n_updates = 0;
    typedef std::tuple<int, int, ggml_backend_dev_t> key_t;   // pass-size bucket, layer, helper card
    std::map<key_t, double> held;     // experts the helper holds (a fraction is kept, the graph rounds)
    std::map<key_t, double> step;     // the next step of this layer and card, in experts
    std::map<key_t, int>    dir;      // direction of the last step: the step is halved when the direction turns
    std::map<key_t, long>   off_at;   // skipped since this update
    std::map<int, double>   bytes_expert;   // per layer: bytes of one expert over the three expert tensors
    std::map<int, double>   ms_tok;         // per pass-size bucket: measured layer period per token (us), to cap a start value by the link
    std::map<int, int>      calm;           // per pass-size bucket: measurements in a row whose layer time stayed within the tolerance
    std::map<int, int>      n_meas;         // per pass-size bucket: measurements read
    std::map<int, double>   period;         // per pass-size bucket: the layer time of the last measurement (us)
    std::map<int, double>   best_period;    // per pass-size bucket: the shortest layer time measured so far
    std::map<key_t, double> used;           // the counts the last graph was built with (what a measurement belongs to)
    std::map<key_t, double> best;           // the counts of the measurement with the shortest layer time
    std::map<key_t, std::vector<double>> est;   // the last few estimates of the balance point (count - gap / time per expert)
    std::map<int, bool>     frozen;         // per pass-size bucket: settled, the counts stay for the rest of the run
    long                    n_rebuilds = 0; // measurements that changed a count (each one rebuilds the graph of the next pass)
};
llama_split_ctl g_split_ctl;

int split_bucket(int64_t n_tokens) {
    int b = 0;
    while (((int64_t) 64 << b) < n_tokens && b < 16) { ++b; }
    return b;
}

double split_opt(const char * key, double def) {
    const char * e = llama_moe_get_opt(key);
    return e ? atof(e) : def;
}

// a finished measurement is read once and every layer's helper counts are moved; true when a count changed (the next graph of that size is rebuilt)
bool split_ctl_update(int64_t n_expert, int64_t n_embd) {
    llama_split_ctl & c = g_split_ctl;
    long gen = 0;
    ggml_backend_sched_chain_stats(nullptr, 0, &gen);
    if (gen == c.gen_seen) { return false; }
    static std::vector<ggml_sched_chain_stat> st(2048);
    const int n = ggml_backend_sched_chain_stats(st.data(), (int) st.size(), &gen);
    c.gen_seen = gen;
    if (n <= 0) { return false; }
    c.n_updates++;
    static const double step0 = split_opt("split-step", 0.25);
    static const double gain  = split_opt("split-gain", 0.75);
    std::map<int, int> big_moves, seen;   // per pass-size bucket of this measurement: layers moved by more than one expert, layers measured
    std::map<int, std::vector<const ggml_sched_chain_stat *>> by_layer;
    for (int i = 0; i < n; ++i) { by_layer[st[i].layer].push_back(&st[i]); }
    double sum_wait = 0, sum_early = 0, sum_share = 0, sum_period = 0;
    int n_h = 0, n_moved = 0, n_off = 0, n_capped = 0, n_period = 0;
    int64_t tokens = 0;
    // the reference chain's start per layer (the owner's: the chain without a hand-off; in range mode the card holding the most experts): two
    // consecutive layers give the layer period
    std::map<int, int64_t> o_start;
    for (const auto & kv : by_layer) {
        const ggml_sched_chain_stat * ref = nullptr;
        for (const auto * e : kv.second) { if (e->t_out == 0) { ref = e; } }
        if (!ref) { for (const auto * e : kv.second) { if (!ref || (e->hi - e->lo) > (ref->hi - ref->lo)) { ref = e; } } }
        if (ref && ref->t_start != 0) { o_start[kv.first] = ref->t_start; }
    }
    for (const auto & kv : by_layer) {
        const ggml_sched_chain_stat * o = nullptr;
        int n_owner = 0;
        for (const auto * e : kv.second) { if (e->t_out == 0) { o = e; ++n_owner; } }
        // dense split (range mode): every card hands its partial results on (no chain without t_out); the reference is the card holding the most experts
        // and the balance is between the chain ENDS: both cards' partial results are needed by both ranges
        const bool range_mode = n_owner == 0;
        if (range_mode) {
            for (const auto * e : kv.second) { if (!o || (e->hi - e->lo) > (o->hi - o->lo)) { o = e; } }
            n_owner = o ? 1 : 0;
        }
        if (n_owner != 1 || o->t_start == 0 || o->t_end <= o->t_start || o->hi <= o->lo) {
            static int dbg = 0;
            if (dbg++ < 8 && getenv("LLAMA_SPLIT_CTL_DEBUG")) {
                LLAMA_LOG_WARN("prefill split: layer %d not measured: %zu chains, owners %d, t_start %lld, t_end %lld, range %d-%d\n", kv.first, kv.second.size(), n_owner,
                        o ? (long long) o->t_start : -1LL, o ? (long long) o->t_end : -1LL, o ? o->lo : -1, o ? o->hi : -1);
            }
            continue;
        }
        tokens = o->n_tokens;
        double period = 0.0;   // us
        {
            auto nx = o_start.find(kv.first + 1), pv = o_start.find(kv.first - 1);
            if (nx != o_start.end() && nx->second > o->t_start)      { period = (double) (nx->second - o->t_start); }
            else if (pv != o_start.end() && o->t_start > pv->second) { period = (double) (o->t_start - pv->second); }
        }
        const double per_o = (double) (o->t_end - o->t_start) / (double) (o->hi - o->lo);
        const double gbs_o = llama_moe_cache_link_gbs_fork(ggml_backend_get_device(o->backend));
        const double hop_o = gbs_o > 0 ? 2.0 * (double) o->n_tokens * (double) n_embd * 4.0 / (gbs_o * 1e3) : 0.0;   // us: the owner's halves of the two copies
        for (const auto * h : kv.second) {
            if (h == o || h->t_start == 0 || h->t_end <= h->t_start || h->hi <= h->lo) { continue; }
            const int bk_h = split_bucket(h->n_tokens);
            if (c.frozen[bk_h]) { continue; }   // settled: nothing moves for this pass size
            // dense split with a plan: the planner's counts stay. Equal expert-chain durations are not the shortest layer there: on Qwen3.8-Flash-Next the
            // experts are 17 of 41 ms, and the balance of the chains left 21% of them on the x4 card, link-bound (48.7 ms per layer against 42 with none).
            if (range_mode && llama_dsplit_card_f_fork(kv.first, h->n_tokens, ggml_backend_get_device(h->backend)) >= 0.0) { continue; }
            const llama_split_ctl::key_t key(bk_h, kv.first, ggml_backend_get_device(h->backend));
            auto it = c.held.find(key);
            if (it == c.held.end()) {
                static int dbg2 = 0;
                if (dbg2++ < 8 && getenv("LLAMA_SPLIT_CTL_DEBUG")) {
                    LLAMA_LOG_WARN("prefill split: layer %d helper %s has no count for bucket %d (tokens %lld)\n", kv.first, ggml_backend_name(h->backend),
                            split_bucket(h->n_tokens), (long long) h->n_tokens);
                }
                continue;
            }
            const double n_meas = (double) (h->hi - h->lo);
            const double per_h  = (double) (h->t_end - h->t_start) / n_meas;
            const double tau    = per_o + per_h;
            // range mode: the chains start at different times (each waits for the other card's ranges), so their DURATIONS are compared: a card whose
            // chain takes longer (its upload or its compute) holds too many experts. (Compared by their ends, the x4 card looked early because the x16's
            // chain waited for the x4's tokens, and its share ran away to more than half.)
            const double delta  = range_mode ? (double) ((h->t_end - h->t_start) - (o->t_end - o->t_start)) : (double) (h->t_out - o->t_end);
            // Every measurement, at whatever count it was made, says where the balance lies: count - delta / tau. The mean of the last few of these
            // estimates (--moe split-window, 3) takes the timing jitter out before anything moves: one pass's gap is good to a millisecond or so, which
            // is several experts.
            double move = 0.0;
            {
                static const size_t window = (size_t) std::max(1.0, split_opt("split-window", 3));
                auto & est = c.est[key];
                est.push_back(std::max(0.0, std::min((double) n_expert - 1.0, n_meas - delta / std::max(1e-9, tau))));
                if (est.size() > window) { est.erase(est.begin()); }
                const double target = std::accumulate(est.begin(), est.end(), 0.0) / (double) est.size();
                const double gap    = (n_meas - target) * tau;   // the smoothed gap, us: positive, the helper holds too many
                if (std::fabs(gap) >= std::max(0.5 * tau, 0.01 * period)) {   // period 0 (not measured): half of tau alone
                    auto sit = c.step.find(key);
                    if (sit == c.step.end()) { sit = c.step.emplace(key, step0 * (double) n_expert).first; }
                    const int d = gap > 0 ? 1 : -1;
                    int & last = c.dir[key];
                    if (last != 0 && last != d) { sit->second *= 0.5; }   // the gap changed sides: the target lies inside the last step
                    last = d;
                    move = d * std::max(1.0, std::min(sit->second, gain * std::fabs(n_meas - target)));   // towards the smoothed balance, at most the step
                }
            }
            double v = std::max(0.0, std::min((double) n_expert - 1.0, n_meas - move));
            // the link cap: what this card's link uploads in one layer period (without what the owner waited), less its two hops, 15% spare
            const double gbs_h = llama_moe_cache_link_gbs_fork(ggml_backend_get_device(h->backend));
            auto be = c.bytes_expert.find(kv.first);
            if (period > 0 && gbs_h > 0 && be != c.bytes_expert.end() && be->second > 0) {
                const double p_eff = std::max(0.0, period - std::max(0.0, delta));
                const double hop_h = 2.0 * (double) h->n_tokens * (double) n_embd * 2.0 / (gbs_h * 1e3);   // us, f16 both ways
                const double cap_n = 0.85 * std::max(0.0, p_eff - hop_h) * gbs_h * 1e3 / be->second;
                if (v > cap_n) { v = cap_n; ++n_capped; }
                sum_period += p_eff; ++n_period;
            }
            if (v < 1.0 || v * per_o < hop_o) { v = 0.0; c.off_at[key] = c.n_updates; ++n_off; }
            if (llround(v) != llround(n_meas)) { ++n_moved; }
            if (std::llabs(llround(v) - llround(n_meas)) > 1) { big_moves[bk_h]++; }
            seen[bk_h]++;
            it->second = v;
            (delta > 0 ? sum_wait : sum_early) += std::fabs(delta);
            sum_share += v / (double) n_expert;
            ++n_h;
        }
    }
    if (n_period > 0 && tokens > 0) { c.ms_tok[split_bucket(tokens)] = sum_period / n_period / (double) tokens; }
    if (n_h > 0 && (c.n_updates <= 12 || c.n_updates % 16 == 0)) {
        LLAMA_LOG_WARN("prefill split: pass %ld of %lld tokens measured over %d layer chains: layer %.1f ms, the owner waited %.2f ms per layer, a helper %.2f ms; helper share now %.1f%% on average, %d layers moved, %d capped by the link, %d skipped\n",
                c.n_updates, (long long) tokens, n_h, n_period ? sum_period / n_period / 1e3 : 0.0, sum_wait / n_h / 1e3, sum_early / n_h / 1e3, 100.0 * sum_share / n_h, n_moved, n_capped, n_off);
    }
    // settle (see the top of this file): by the layer time of the pass size most layers of this measurement ran at
    bool restored = false;
    {
        int bk_main = -1, n_main = 0;
        for (const auto & sb : seen) { if (sb.second > n_main) { n_main = sb.second; bk_main = sb.first; } }
        if (bk_main >= 0 && n_main >= 4 && n_period > 0) {
            static const double tol   = split_opt("split-settle", 0.015);
            static const int    limit = (int) split_opt("split-passes", 8);
            const double p = sum_period / n_period;
            double & last = c.period[bk_main];
            int & calm    = c.calm[bk_main];
            int & meas    = c.n_meas[bk_main];
            ++meas;
            calm = (last > 0.0 && std::fabs(p - last) <= tol * last) ? calm + 1 : 0;
            last = p;
            auto bp = c.best_period.find(bk_main);
            if (bp == c.best_period.end() || p < bp->second) {
                c.best_period[bk_main] = p;
                for (const auto & u : c.used) { if (seen.count(std::get<0>(u.first))) { c.best[u.first] = u.second; } }   // the counts this measurement ran with
            }
            if (calm >= 2 || meas >= limit) {
                for (const auto & sb : seen) { c.frozen[sb.first] = true; }
                for (const auto & bst : c.best) {
                    if (!seen.count(std::get<0>(bst.first))) { continue; }
                    auto h = c.held.find(bst.first);
                    if (h != c.held.end() && llround(h->second) != llround(bst.second)) { h->second = bst.second; restored = true; }
                }
                LLAMA_LOG_WARN("prefill split: passes of up to %lld tokens are settled after %d measurements (layer time %.1f ms, the best %.1f ms): the expert counts of the best one are kept for the rest of the run\n",
                        (long long) ((int64_t) 64 << bk_main), meas, p / 1e3, c.best_period[bk_main] / 1e3);
            }
        }
    }
    if (n_moved > 0 || restored) { c.n_rebuilds++; }
    return n_moved > 0 || restored;
}

// experts each card holds in layer il for a pass of n_tokens (cards: device and upload rate; the owner keeps what the helpers do not hold)
std::vector<int64_t> split_ctl_counts(int il, int64_t n_tokens, const std::vector<std::pair<ggml_backend_dev_t, double>> & cards, size_t owner, int64_t n_expert, int64_t n_embd, double bytes_layer, bool range_mode) {
    llama_split_ctl & c = g_split_ctl;
    std::lock_guard<std::mutex> lk(c.mu);
    if (bytes_layer > 0 && n_expert > 0) { c.bytes_expert[il] = bytes_layer / (double) n_expert; }

    // Start share: the upload-balanced share (replaces the fixed 1.4 x link ratio of the first version and a factor fitted to GLM-5.3-Flash: measured
    // there, the best x4 share was 15% at 1.8k tokens and 25% at 5.5k against 19.7% upload-balanced, but that curve belongs to that model and those
    // cards; the bisection tunes each layer and pass size from here).
    const double len_f = 1.0;
    static const long   retry = (long) split_opt("split-retry", 32);
    const int bk = split_bucket(n_tokens);
    std::vector<double> v(cards.size(), 0.0);
    // Whether the card owns the layer or helps does not matter (GLM-5.3-Flash on x16 + x4: the x4 card owns five layers; seeded as "the helper gets a
    // multiple of its rate over the owner's", the x16 helper was cut to an equal split there and the x4 uploaded 1.2 GB per layer over its link); the cards
    // on the fastest link share the rest equally.
    double u_max = 0.0, u_sum = 0.0, slow_sum = 0.0;
    int    n_fast = 0;
    for (const auto & cd : cards) { u_max = std::max(u_max, cd.second); u_sum += cd.second; }
    const auto slow_start = [&](size_t i) { return std::min(1.0, len_f * cards[i].second / std::max(1e-9, u_sum)); };
    for (size_t i = 0; i < cards.size(); ++i) { if (cards[i].second < 0.999 * u_max) { slow_sum += slow_start(i); } else { ++n_fast; } }
    const double slow_scale = slow_sum > 0.9 ? 0.9 / slow_sum : 1.0;
    const auto start_share = [&](size_t i) {
        return cards[i].second < 0.999 * u_max ? slow_scale * slow_start(i) : (1.0 - slow_scale * slow_sum) / std::max(1, n_fast);
    };
    const auto seed = [&](size_t i) {
        // dense split (token ranges): the split planner's share of this layer's experts for the helper, when it has a plan for this pass size (it counts the
        // uploads net of what each card holds and the hops on the helper's link; Qwen3.8-Flash-Next on x16 + x4: none on the x4)
        if (range_mode) {
            const double fp = llama_dsplit_card_f_fork(il, n_tokens, cards[i].first);
            if (fp >= 0.0) { return fp * (double) n_expert; }
        }
        double v0 = start_share(i) * (double) n_expert;
        // The link cap for a pass size not measured yet: the layer period per token of the nearest measured size, scaled to this size (the period grows
        // about with the tokens). No size measured at all: the start share stands, and the first measured pass corrects it.
        if (cards[i].second < 0.999 * u_max && bytes_layer > 0) {
            for (int d = 0; d <= 6; ++d) {
                auto mt = c.ms_tok.find(bk - d);
                if (mt == c.ms_tok.end()) { mt = c.ms_tok.find(bk + d); }
                if (mt != c.ms_tok.end()) {
                    const double period = mt->second * (double) n_tokens;                                               // us
                    const double hop    = 2.0 * (double) n_tokens * (double) n_embd * 2.0 / (cards[i].second * 1e3);    // us
                    v0 = std::min(v0, 0.85 * std::max(0.0, period - hop) * cards[i].second * 1e3 / (bytes_layer / (double) n_expert));
                    break;
                }
            }
        }
        return v0;
    };
    double sum_h = 0.0;
    bool   fresh = false;
    for (size_t i = 0; i < cards.size(); ++i) {
        if (i == owner) { continue; }
        const llama_split_ctl::key_t key(bk, il, cards[i].first);
        // dense split with a plan: the planner's share for this layer, scaled by the split tuner (llama-dsplit-fork.cpp), read on every graph build
        if (range_mode) {
            const double fp = llama_dsplit_card_f_fork(il, n_tokens, cards[i].first);
            if (fp >= 0.0) {
                v[i] = fp * (double) n_expert;
                sum_h += v[i];
                continue;
            }
        }
        auto it = c.held.find(key);
        if (it == c.held.end()) {
            it = c.held.emplace(key, seed(i)).first;
            fresh = true;
        } else if (it->second <= 0.0 && !c.frozen[bk]) {   // a settled size keeps its switched-off helpers
            auto ot = c.off_at.find(key);
            if (ot == c.off_at.end() || c.n_updates - ot->second >= retry) { it->second = seed(i); c.off_at.erase(key); fresh = true; }
        }
        v[i] = it->second;
        sum_h += v[i];
    }
    // the owner always keeps one expert
    (void) fresh;
    const double max_h = (double) n_expert - 1.0;
    if (sum_h > max_h && sum_h > 0.0) {
        for (size_t i = 0; i < cards.size(); ++i) {
            if (i != owner) { v[i] *= max_h / sum_h; }
        }
    }
    std::vector<int64_t> out(cards.size(), 0);
    int64_t used = 0;
    for (size_t i = 0; i < cards.size(); ++i) {
        if (i == owner) { continue; }
        out[i] = std::max<int64_t>(0, std::min<int64_t>(n_expert - 1 - used, llround(v[i])));
        used += out[i];
        c.used[llama_split_ctl::key_t(bk, il, cards[i].first)] = (double) out[i];
    }
    out[owner] = n_expert - used;
    return out;
}

}  // namespace

// fork: asked before a prompt pass reuses the graph of the pass before (llama_context::process_ubatch): reads a finished measurement of the expert chains and
// moves the cards' expert counts (see "apply" at the top of this file). true: a count changed, the graph has to be rebuilt.
bool llama_split_ctl_poll_fork(int64_t n_expert, int64_t n_embd) {
    static const bool tune = split_opt("split-auto", 1) != 0;
    if (!tune || n_expert <= 0) { return false; }
    std::lock_guard<std::mutex> lk(g_split_ctl.mu);
    return split_ctl_update(n_expert, n_embd);
}

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    ASSERT_EMBD_OR_TOKEN(*ubatch);

    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    GGML_ASSERT(ubatch->embd_state && "this graph requires a state embedding, see llama_batch_ext_set_embd_state()");
    GGML_ASSERT(n_embd_state == h->ne[0]);

    ggml_backend_tensor_set(h, ubatch->embd_state, 0, n_tokens*n_embd_state*ggml_element_size(h));
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token)      || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)       || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd_state) || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        const bool has_embd = ubatch->is_mixed() || ubatch->token == nullptr;
        if (rope_section_order == LLAMA_ROPE_SECTION_ORDER_TYXZ || !has_embd) {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
            return;
        }

        // input is always [t, y, x, z]
        // token entries are expanded by the batch to [p, p, p, 0]
        GGML_ASSERT(n_pos_per_embd == 4);

        // slot index per section, slots are t = 0, y = 1, x = 2, z = 3
        std::array<int64_t, 4> slot_of_section = { 0, 1, 2, 3 };
        switch (rope_section_order) {
            case LLAMA_ROPE_SECTION_ORDER_TYXZ: break;
            case LLAMA_ROPE_SECTION_ORDER_ZYXT: slot_of_section = { 3, 1, 2, 0 }; break;
            default: GGML_ABORT("unsupported rope section order");
        }

        std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = ubatch->is_mixed() ? ubatch->type[i] != 0 : true;
            for (int64_t s = 0; s < 4; ++s) {
                const int64_t slot = is_embd ? slot_of_section[s] : s;
                pos_data[s*n_tokens + i] = ubatch->pos[slot*n_tokens + i];
            }
        }
        ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        if (attn_temp_pow) {
            // (1 + (pos + offset) / length) ^ power
            for (int i = 0; i < n_tokens; ++i) {
                const double pos = ubatch->pos[i];
                attn_scale_data[i] = (float) std::exp((double) f_attn_temp_scale *
                    std::log1p((pos + f_attn_temp_offset) / (double) n_attn_temp_floor_scale));
            }
        } else {
            for (int i = 0; i < n_tokens; ++i) {
                const float pos = ubatch->pos[i];
                attn_scale_data[i] = std::log(
                    std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
                ) * f_attn_temp_scale + 1.0;
            }
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

bool llm_graph_input_attn_temp::can_reuse(const llm_graph_params & params) {
    // only the power form opts in; the log form keeps no reuse
    return attn_temp_pow && attn_scale != nullptr && attn_scale->ne[2] == params.ubatch.n_tokens;
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    if (!out_ids->buffer) {   // fork: say what this input is when the scheduler did not allocate it (qwen3.6 4.7 GiB squeeze repro, 2026-10-18)
        LLAMA_LOG_ERROR("%s: out_ids has no buffer: name '%s' tensor %p ne0 %lld flags 0x%x data %p view_src %p ubatch n_tokens %u n_outputs %d (this=%p)\n", __func__, out_ids->name, (void *) out_ids,
                (long long) out_ids->ne[0], (unsigned) out_ids->flags, out_ids->data, (void *) out_ids->view_src, ubatch->n_tokens, (int) n_outputs, (void *) this);
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = self_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type, bool causal) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (causal && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    // see llama_non_causal_type
    const bool causal     = cparams.causal_attn;
    const bool causal_swa = cparams.causal_attn || hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY;

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE, causal);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE, causal);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type, causal_swa);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type, causal_swa);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    // the idxs are left unallocated when the KV cache is bypassed during training
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    // leave unallocated when the top-k layers use the indexer mask
    if (self_kq_mask_mla->buffer) {
        mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);
    }

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    // left unallocated when the indexer does not use the rotation
    if (self_k_rot_lid && self_k_rot_lid->buffer) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    // leave the mask unallocated if all attention layers have their own (e.g. qwen4exp QSA)
    if (inp_attn->self_kq_mask->buffer) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    // leave the mask unallocated if all attention layers have their own (e.g. glm5-next DSA)
    if (inp_attn->self_kq_mask->buffer) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;
    t_mtp_dsa_sel  = nullptr;
    t_mtp_dsa_mask = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();
    hdmd_inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    if (t_mtp_dsa_sel != nullptr) {
        ggml_set_output(t_mtp_dsa_sel);
    }
    if (t_mtp_dsa_mask != nullptr) {
        ggml_set_output(t_mtp_dsa_mask);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                // fork: a ubatch longer than embeddings_layer_inp_last keeps only its last rows as the output, as a copy, so only those are copied
                // out. The copy runs at the end of the graph, so the whole tensor still lives to the end: a model whose layer inputs are big takes
                // only the last rows itself (deepseek4.cpp: DeepSeek-V4.1 + DSpark, 3 layers x 320 MiB at a 16384-row pass)
                const int64_t last = params.cparams.embeddings_layer_inp_last;
                ggml_tensor * t = t_layer_inp[il];
                if (last > 0 && t->ne[1] > last && ggml_n_dims(t) <= 2 && ggml_is_contiguous(t)) {
                    ggml_context * ctx = ctx_compute.get();
                    t = ggml_cont(ctx, ggml_view_2d(ctx, t, t->ne[0], last, t->nb[1], (size_t) (t->ne[1] - last) * t->nb[1]));
                    ggml_build_forward_expand(gf, t);
                    t_layer_inp[il] = t;
                }
                ggml_set_output(t);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

ggml_tensor * llm_graph_result::get_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot) const {
    const auto it = hdmd_inputs.find({ cur, rot });
    return it == hdmd_inputs.end() ? nullptr : it->second;
}

void llm_graph_result::set_hdmd_input(const ggml_tensor * cur, const ggml_tensor * rot, ggml_tensor * res) {
    hdmd_inputs[{ cur, rot }] = res;
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    moe_cache        (params.moe_cache),
    prec_policy      (params.prec_policy),
    hdmd             (params.hdmd),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_hadamard_input(
          ggml_tensor * w,
          ggml_tensor * cur) const {
    if (!hdmd) {
        return cur;
    }
    const auto it = hdmd->rot.find(w);
    if (it == hdmd->rot.end()) {
        return cur;
    }
    const auto & t = it->second;
    if (ggml_tensor * x = res->get_hdmd_input(cur, t.rot)) {
        return x;
    }
    ggml_tensor * x = cur;
    if (t.perm_rep > 1) {
        // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
        x = ggml_is_contiguous(x) ? x : ggml_cont(ctx0, x);
        const int64_t ne1 = x->ne[1], ne2 = x->ne[2], ne3 = x->ne[3];
        x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
    }
    if (t.signs) {
        x = ggml_mul(ctx0, x, t.signs);
    }
    x = llama_mul_mat_hadamard(ctx0, x, t.rot);
    res->set_hdmd_input(cur, t.rot, x);
    return x;
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    // a Hadamard-folded weight reads the transformed input; a LoRA delta lives in the primal basis and reads `cur`
    ggml_tensor * cur_mm = build_hadamard_input(w, cur);
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur_mm);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

// fork: in a multi-GPU prompt split each card computes its share of a layer's experts (mm_id_backend set): its ids hold -1 in the slots whose expert
// another card computes. The matmul skips them (op_params[1]: their rows come out zero), but a gather BY those ids (per-expert weight scales, expert
// biases) reads row -1, i.e. the bytes in front of the gathered tensor in the graph buffer: whatever another tensor left there. 0 x NaN is NaN:
// Qwen3.8-Flash-Next (per-expert scales, GSQ-RCO) answered '!!!!' after long prompts whenever the planner gave the helper card a few experts in some
// layers, and never with no helper experts or with one card; which runs failed depended on what happened to lie in front of the scale tensor
// (2026-10-18). The gathers take the ids clamped to a real expert: the rows they scale or shift are the zero rows of skipped slots, any expert is right.
static ggml_tensor * moe_ids_for_gather(ggml_context * ctx, ggml_tensor * ids, int64_t n_expert, ggml_backend_sched_t sched, ggml_backend_t b) {
    ggml_tensor * f = ggml_cast(ctx, ids, GGML_TYPE_F32);
    ggml_tensor * c = ggml_clamp(ctx, f, 0.0f, (float) (n_expert - 1));
    ggml_tensor * i = ggml_cast(ctx, c, GGML_TYPE_I32);
    if (sched && b) {
        for (ggml_tensor * t : { f, c, i }) { ggml_backend_sched_set_tensor_backend(sched, t, b); }
    }
    return i;
}

llm_graph_amax_src1 llm_graph_context::build_amax_src1_f16(
          ggml_tensor * cur) const {
    llm_graph_amax_src1 result = {
        /*.res     =*/ cur,
        /*.scale_s =*/ nullptr,
    };

    const int64_t n_embd = cur->ne[0];

    // amax = max|cur| along the rows for each [expert, token] -> [1, n_expert, n_tokens]
    ggml_tensor * ab = ggml_abs(ctx0, cur);
    ggml_tensor * amax = ggml_pool_1d(ctx0, ab, GGML_OP_POOL_MAX, n_embd, n_embd, 0);

    // ref: https://github.com/ggml-org/llama.cpp/pull/26223
    constexpr float k_log2_e = 1.4426950408889634f; // 1 / ln(2)
    constexpr float k_ln_2   = 0.6931471805599453f;

    ggml_tensor * e;
    // note: this can technically be:
    // e = ggml_scale(ctx0, amax, 1.0f/32768.0f);
    // but we leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks
    e = ggml_scale(ctx0, amax, 1.0f/1024.0f);
    e = ggml_log  (ctx0, e);
    e = ggml_scale(ctx0, e, k_log2_e);
    e = ggml_ceil (ctx0, e);
    e = ggml_clamp(ctx0, e, 0.0f, 127.0f);
    e = ggml_scale(ctx0, e, k_ln_2);
    e = ggml_exp  (ctx0, e);

    result = {
        /*.res     =*/ ggml_div(ctx0, cur, e),
        /*.scale_s =*/ e,
    };

    return result;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s,
          ggml_tensor * slots) const {
    // as in build_lora_mm: the folded weight reads the transformed input, the LoRA delta reads `cur`
    ggml_tensor * cur_mm = build_hadamard_input(w, cur);
    // the experts in the MoE cache are selected by their slots
    ggml_tensor * res = slots == nullptr ?
        ggml_mul_mat_id(ctx0, w, cur_mm, ids) :
        ggml_mul_mat_id(ctx0, moe_cache->get_experts(w), cur_mm, slots);
    if (mm_id_backend) {
        res->op_params[1] = 1; // ids may be -1: zero the output first
        if (mm_id_hi > mm_id_lo) { res->op_params[8] = mm_id_lo; res->op_params[9] = mm_id_hi; }
        ggml_backend_sched_set_tensor_backend(sched, res, mm_id_backend);
    }

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, mm_id_backend ? moe_ids_for_gather(ctx0, ids, n_expert, sched, mm_id_backend) : ids);   // -1 ids: see moe_ids_for_gather
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 || arch == LLM_ARCH_GLM5_NEXT || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_GELU_ERF:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_GEGLU_ERF:
            {
                cur = ggml_geglu_erf(ctx0, cur);
                cb(cur, "ffn_geglu_erf", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

// the architectures whose routed experts use the clamped swiglu (the stock path, the expert cache and its JIT pool must agree)
static bool moe_uses_swiglu_clamp(llm_arch arch, const llama_hparams & hparams) {
    return arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41 || arch == LLM_ARCH_GLM5_NEXT ||
           (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in,
        post_exp_norm,
        post_exp_norm_eps
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * post_exp_norm,
               float   post_exp_norm_eps) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    // LLAMA_MOE_DEFER: the previous MoE layer's deferred host experts go into the graph here, after this layer's
    // attention and before its router, so the host computes them while the GPU runs the attention
    ggml_tensor * defer_prev = moe_defer_pending;
    moe_defer_pending = nullptr;
    if (defer_prev) {
        ggml_build_forward_expand(gf, cur);
        ggml_build_forward_expand(gf, defer_prev);
    }

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb(logits, "ffn_moe_logits", il);

        // MoE expert cache: train the predictors that targeted this layer (NLMS on the router logits):
        // w += mu/|x|^2 * x (x) (logits - pred), on the predictor's GPU, no host work
        auto it = moe_pred_todo.find(il);
        if (it != moe_pred_todo.end()) {
            for (int src : it->second) {
                auto & e = moe_pred_srcs[src];
                if (logits->ne[1] != e.pred->ne[1] || logits->ne[0] * e.K != e.pred->ne[0]) {
                    e.K = -1;   // rows don't match (output-only last layer): no update for this source
                    continue;
                }
                e.real.push_back(logits);
                if ((int) e.real.size() != e.K) {
                    continue;
                }
                // all targets seen: one stacked update, w += mu/|x|^2 * x (x) (logits - pred)
                std::vector<ggml_tensor *> nodes;
                ggml_tensor * real = e.real[0];
                for (size_t i = 1; i < e.real.size(); ++i) {
                    real = ggml_concat(ctx0, real, e.real[i], 0);                        // [n_expert * K, n_tokens]
                    nodes.push_back(real);
                }
                ggml_tensor * err = ggml_sub(ctx0, real, e.pred);
                ggml_tensor * nrm = ggml_sum_rows(ctx0, ggml_sqr(ctx0, e.x));            // [1, n_tokens]
                ggml_tensor * g   = ggml_mul(ctx0, ggml_div(ctx0, err, nrm), e.mu);
                ggml_tensor * dw  = ggml_out_prod(ctx0, e.x, g);                         // [n_embd, n_expert * K]
                ggml_tensor * sum = ggml_add(ctx0, e.w, dw);
                ggml_tensor * upd = ggml_cpy(ctx0, sum, e.w);
                for (ggml_tensor * t : { err, nrm, nrm->src[0], g, g->src[0], dw, sum, upd }) {
                    nodes.push_back(t);
                }
                ggml_tensor * requant = nullptr;
                if (e.q) {
                    // refresh the Q8_0 prediction copy from the updated weights
                    ggml_tensor * f32 = ggml_cast(ctx0, sum, GGML_TYPE_F32);
                    requant = ggml_cpy(ctx0, f32, e.q);
                    nodes.push_back(f32);
                    nodes.push_back(requant);
                }
                for (ggml_tensor * t : nodes) {
                    ggml_backend_sched_set_tensor_backend(sched, t, e.backend);
                }
                ggml_build_forward_expand(gf, upd);
                if (requant) {
                    ggml_build_forward_expand(gf, requant);
                }
            }
            moe_pred_todo.erase(it);
        }
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = selected_experts_in;
    if (selected_experts == nullptr) {
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
    }
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (arch == LLM_ARCH_DEEPSEEK41) {
            // model.py Gate.forward: weights / (sum + 1e-20); the sum can fall below the F16 clamp below
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0f, 1e-20f);
        } else {
            // Avoid division by zero, clamp to smallest number representable by F16
            weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        }
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    // MoE expert cache (see llama-moecache-fork.h): during decode and small batches
    // (short prefills, speculative verify) on a layer whose experts live in host
    // memory, run a parallel mul_mat_id chain over a device-resident cache of hot
    // experts. Cached ids are skipped by the CPU chain (src[3] table) and served
    // by the cache chain; uncached ids map to the cache's zero slot. The two
    // outputs sum to the exact result. Larger batches keep the stock path, where
    // the scheduler streams the used experts to the GPU instead.
    const llama_moe_cache_layer_fork * mcache = nullptr;
    ggml_tensor * mc_ids_all = nullptr; // LLAMA_MOE_DEFER: full ids (router observation) and the deferred ids
    ggml_tensor * mc_ids_def = nullptr;
    ggml_tensor * mc_w_host  = nullptr;
    intptr_t      mc_defer_pk = 0;      // protected (immediate) experts per token
    ggml_tensor * mc_defer_tbl = nullptr;
    ggml_tensor * mc_defer_mask = nullptr;
    ggml_tensor * mc_slot_ids = nullptr;
    ggml_backend_t mc_cpu_backend = nullptr;
    ggml_tensor *  mc_pred        = nullptr;
    if (cparams.moe_cache && n_tokens > 0 && n_tokens <= llama_moe_cache_max_batch_fork() && !gate_up_exps && gate_exps && down_exps &&
        !up_exps_b && !gate_exps_b && !down_exps_b &&
        !up_exps_s && !gate_exps_s && !down_exps_s &&
        type_op == LLM_FFN_SILU && !weight_before_ffn && loras->empty()) {
        mcache = llama_moe_cache_lookup_fork(up_exps);
    }

    // the experts of host-resident layers may be read from the MoE cache
    ggml_tensor * slots = build_moe_cache_slots(selected_experts, up_exps, gate_exps, down_exps, gate_up_exps, il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    ggml_tensor * mc_inp = cur;

    // Expert cache: overlap the GPU cache chain with the CPU chain for the
    // uncached experts. Copy the CPU chain's inputs to CPU in their own split
    // first (that copy syncs the GPU while only the router is queued), then
    // queue the GPU chain, then run the CPU chain on the copies only, so the
    // scheduler has no reason to wait for the GPU before computing on CPU.
    ggml_tensor * down_g = nullptr;
    if (mcache) {
        ggml_backend_t cpu_backend = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu_backend = b;
            }
        }
        mc_cpu_backend = cpu_backend;
        ggml_tensor * pw = mcache->pred_all ? mcache->pred_all : mcache->pred_w;
        if (pw && pw->buffer && mcache->pred_on) {
            // the next layers' router logits from this layer's input (learned predictors, else the next
            // router), on their GPU in the split before the CPU copy; handed to the CPU gate op as src[4]
            // ([n_expert*ahead, n_tokens]), where the cache's observer reads them
            ggml_backend_dev_t wdev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(pw->buffer));
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
                if (wdev && ggml_backend_get_device(b) == wdev && b != cpu_backend) {
                    ggml_tensor * x2 = ggml_reshape_2d(ctx0, mc_inp, n_embd, n_tokens);
                    // all lookaheads in one matmul ([n_expert * K, n_tokens], the stacked layout the cache reads)
                    mc_pred = ggml_mul_mat(ctx0, mcache->pred_q ? mcache->pred_q : pw, x2);
                    ggml_backend_sched_set_tensor_backend(sched, mc_pred, b);
                    // train on decode and small batches only: a big prefill batch would sum too many steps
                    if (mcache->pred_all && mcache->pred_mu && n_tokens <= 8 && llama_moe_cache_pred_train_now_fork()) {
                        moe_pred_srcs[il] = { mcache->pred_all, mcache->pred_q, x2, mc_pred, mcache->pred_mu, b, mcache->pred_ahead, {} };
                        for (int k = 0; k < mcache->pred_ahead; ++k) {
                            moe_pred_todo[il + 1 + k].push_back(il);
                        }
                    }
                    cb(mc_pred, "ffn_moe_pred", il);
                    if (cpu_backend) {
                        // copied with the CPU chain's inputs, so the gate op doesn't wait for the GPU chain
                        mc_pred = ggml_cont(ctx0, mc_pred);
                        ggml_backend_sched_set_tensor_backend(sched, mc_pred, cpu_backend);
                        cb(mc_pred, "ffn_moe_pred_cpu", il);
                    }
                    ggml_build_forward_expand(gf, mc_pred);
                    break;
                }
            }
        }
        ggml_tensor * sel_dev = selected_experts;
        if (cpu_backend) {
            cur = ggml_cont(ctx0, cur);
            selected_experts = ggml_cont(ctx0, selected_experts);
            ggml_backend_sched_set_tensor_backend(sched, cur, cpu_backend);
            ggml_backend_sched_set_tensor_backend(sched, selected_experts, cpu_backend);
            cb(cur, "ffn_moe_cpu_inp", il);
            ggml_build_forward_expand(gf, cur);
            ggml_build_forward_expand(gf, selected_experts);
            // LLAMA_MOE_DEFER=n (decode, not the last layer): the host misses among the n lowest-scored of the
            // n_expert_used selected experts (argsort order: best first) are computed while the next layer's attention
            // runs, and their output joins the next MoE layer's output (ktransformers' expert deferral)
            static const int defer_n = [] { const char * e = moe_opt("defer"); return e ? atoi(e) : 0; }();
            if (defer_n > 0 && defer_n < n_expert_used && n_tokens == 1 && il < (int) n_layer - 1 && !selected_experts_in &&
                    !up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps && !weight_before_ffn) {
                mc_ids_all = selected_experts;
                mc_defer_pk = 1; // deferral on for this layer
                // router weights on the host with this split's inputs: the deferred chain then has no GPU input, so it
                // doesn't wait (sync) for the GPU work queued after this layer (the next layer's attention)
                mc_w_host = ggml_cont(ctx0, weights);
                ggml_backend_sched_set_tensor_backend(sched, mc_w_host, cpu_backend);
                cb(mc_w_host, "ffn_moe_defer_w", il);
                ggml_build_forward_expand(gf, mc_w_host);
                // deferral mask, decided once, before the JIT and from the same table the GPU chain uses: host misses in
                // ascending router weight while the count stays <= LLAMA_MOE_DEFER and their summed weight stays <=
                // LLAMA_MOE_DEFER_MASS of the layer's total (default 1: count only). Cheap-in-quality misses first.
                mc_defer_mask = ggml_map_custom2(ctx0, selected_experts, mc_w_host,
                        [](ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * w, int ith, int, void * ud) {
                            if (ith != 0) { return; }
                            const llama_moe_cache_layer_fork * mc = (const llama_moe_cache_layer_fork *) ud;
                            static const int   dn   = [] { const char * e = moe_opt("defer"); return e ? atoi(e) : 0; }();
                            static const float mass = [] { const char * e = moe_opt("defer-mass"); return e ? (float) atof(e) : 1.0f; }();
                            const int32_t * tbl = (const int32_t *) mc->host_table->data;
                            for (int64_t t = 0; t < a->ne[1]; ++t) {
                                std::vector<std::pair<float, int64_t>> miss; // (weight, slot i)
                                float total = 0.0f;
                                for (int64_t i = 0; i < a->ne[0]; ++i) {
                                    const int32_t v = *(const int32_t *) ((const char *) a->data + t*a->nb[1] + i*a->nb[0]);
                                    const float   x = *(const float *) ((const char *) w->data + i*w->nb[1] + t*w->nb[2]);
                                    total += x;
                                    *(int32_t *) ((char *) dst->data + t*dst->nb[1] + i*dst->nb[0]) = 0;
                                    if (v >= 0 && tbl[v] == mc->n_slots) { miss.emplace_back(x, i); }
                                }
                                std::sort(miss.begin(), miss.end());
                                float cum = 0.0f;
                                int   cnt = 0;
                                for (const auto & m : miss) {
                                    if (cnt >= dn || cum + m.first > mass*total) { break; }
                                    cum += m.first;
                                    cnt++;
                                    *(int32_t *) ((char *) dst->data + t*dst->nb[1] + m.second*dst->nb[0]) = 1;
                                }
                            }
                        }, 1, (void *) mcache);
                ggml_backend_sched_set_tensor_backend(sched, mc_defer_mask, cpu_backend);
                cb(mc_defer_mask, "ffn_moe_defer_mask", il);
                ggml_build_forward_expand(gf, mc_defer_mask);
                // immediate ids: the layer's selection without the deferred experts
                selected_experts = ggml_map_custom2(ctx0, selected_experts, mc_defer_mask,
                        [](ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * m, int ith, int, void *) {
                            if (ith != 0) { return; }
                            for (int64_t t = 0; t < a->ne[1]; ++t) {
                                for (int64_t i = 0; i < a->ne[0]; ++i) {
                                    const int32_t v = *(const int32_t *) ((const char *) a->data + t*a->nb[1] + i*a->nb[0]);
                                    const int32_t d = *(const int32_t *) ((const char *) m->data + t*m->nb[1] + i*m->nb[0]);
                                    *(int32_t *) ((char *) dst->data + t*dst->nb[1] + i*dst->nb[0]) = d ? -1 : v;
                                }
                            }
                        }, 1, nullptr);
                ggml_backend_sched_set_tensor_backend(sched, selected_experts, cpu_backend);
                cb(selected_experts, "ffn_moe_imm_ids", il);
                ggml_build_forward_expand(gf, selected_experts);
            }
            // JIT miss offload: with the ids on the host (this CPU split), the cache may upload some of this layer's
            // misses on the chain's stream before the chain below is launched (llama_moe_cache_jit_fork)
            ggml_tensor * jit = ggml_map_custom1(ctx0, selected_experts,
                    [](ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
                        GGML_UNUSED(nth);
                        if (ith == 0) {
                            llama_moe_cache_jit_fork((const llama_moe_cache_layer_fork *) ud, a);
                            memcpy(dst->data, a->data, ggml_nbytes(a));
                        }
                    }, 1, (void *) mcache);
            ggml_backend_sched_set_tensor_backend(sched, jit, cpu_backend);
            cb(jit, "ffn_moe_cache_jit", il);
            ggml_build_forward_expand(gf, jit);
        }

        // slot ids from the device table, after the CPU split above so they see the JIT's table updates;
        // get_rows batches over ids->ne[1] only when the table has matching ne[2]: flatten
        {
            ggml_tensor * ids_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, sel_dev), n_expert_used*n_tokens);
            mc_slot_ids = ggml_get_rows(ctx0, mcache->dev_table, ids_flat); // [1, n_expert_used*n_tokens]
            mc_slot_ids = ggml_reshape_2d(ctx0, mc_slot_ids, n_expert_used, n_tokens);
            cb(mc_slot_ids, "ffn_moe_cache_slots", il);
        }
        // JIT pool (layer cached on a slower-link GPU): the pool's slot ids, on the fastest-link GPU
        ggml_tensor * mc_pool_ids = nullptr;
        if (mcache->jit_n > 0) {
            ggml_tensor * ids_flat = ggml_reshape_1d(ctx0, ggml_cont(ctx0, sel_dev), n_expert_used*n_tokens);
            mc_pool_ids = ggml_get_rows(ctx0, mcache->jit_table, ids_flat);
            mc_pool_ids = ggml_reshape_2d(ctx0, mc_pool_ids, n_expert_used, n_tokens);
            cb(mc_pool_ids, "ffn_moe_jit_slots", il);
        }

        // device-side chain over the cached experts, mirroring the LLM_FFN_SILU
        // activation above (the only type_op the cache path is enabled for).
        // Uncached ids all map to one zero slot, so an id repeats within a token; only
        // the vector kernels (MMVQ/MMVF) handle that, MMQ sizes its grid for distinct
        // ids. Run batches in chunks of 4 tokens, below every type's MMVQ id limit.
        constexpr int64_t mc_chunk = 4;
        for (int64_t t0 = 0; t0 < n_tokens; t0 += mc_chunk) {
            const int64_t nt = std::min<int64_t>(mc_chunk, n_tokens - t0);
            // a single chunk (every decode step) takes the tensors as-is: views here
            // measured ~1.7% slower decode (likely they block CUDA backend fusion)
            ggml_tensor * inp_c = n_tokens <= mc_chunk ? mc_inp :
                ggml_view_3d(ctx0, mc_inp, n_embd, 1, nt, mc_inp->nb[1], mc_inp->nb[2], t0*mc_inp->nb[2]);
            ggml_tensor * ids_c = n_tokens <= mc_chunk ? mc_slot_ids :
                ggml_view_2d(ctx0, mc_slot_ids, n_expert_used, nt, mc_slot_ids->nb[1], t0*mc_slot_ids->nb[1]);

            ggml_tensor * up_g   = ggml_mul_mat_id(ctx0, mcache->up_c,   inp_c, ids_c);
            ggml_tensor * gate_g = ggml_mul_mat_id(ctx0, mcache->gate_c, inp_c, ids_c);
            cb(up_g,   "ffn_moe_cache_up",   il);
            cb(gate_g, "ffn_moe_cache_gate", il);

            ggml_tensor * act_g = nullptr;
            {
                const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                constexpr float eps = 1e-6f;
                if (limit > eps) {
                    if (moe_uses_swiglu_clamp(arch, hparams)) {
                        // exactly the op the CPU chain uses for these archs
                        act_g = ggml_swiglu_clamp(ctx0, gate_g, up_g, limit);
                    } else {
                        up_g = ggml_clamp(ctx0, up_g, -limit, limit);
                        ggml_tensor * ga = ggml_silu(ctx0, gate_g);
                        ga    = ggml_clamp(ctx0, ga, -INFINITY, limit);
                        act_g = ggml_mul(ctx0, ga, up_g);
                    }
                } else {
                    act_g = ggml_swiglu_split(ctx0, gate_g, up_g);
                }
            }

            ggml_tensor * down_c = ggml_mul_mat_id(ctx0, mcache->down_c, act_g, ids_c); // [n_embd, n_expert_used, nt]
            cb(down_c, "ffn_moe_cache_down", il);
            if (mc_pool_ids) {
                ggml_tensor * ids_p = n_tokens <= mc_chunk ? mc_pool_ids :
                    ggml_view_2d(ctx0, mc_pool_ids, n_expert_used, nt, mc_pool_ids->nb[1], t0*mc_pool_ids->nb[1]);
                ggml_tensor * up_p   = ggml_mul_mat_id(ctx0, mcache->jit_up_c,   inp_c, ids_p);
                ggml_tensor * gate_p = ggml_mul_mat_id(ctx0, mcache->jit_gate_c, inp_c, ids_p);
                const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                ggml_tensor * act_p = nullptr;
                if (limit > 1e-6f) {
                    if (moe_uses_swiglu_clamp(arch, hparams)) {
                        act_p = ggml_swiglu_clamp(ctx0, gate_p, up_p, limit);
                    } else {
                        up_p = ggml_clamp(ctx0, up_p, -limit, limit);
                        ggml_tensor * ga = ggml_clamp(ctx0, ggml_silu(ctx0, gate_p), -INFINITY, limit);
                        act_p = ggml_mul(ctx0, ga, up_p);
                    }
                } else {
                    act_p = ggml_swiglu_split(ctx0, gate_p, up_p);
                }
                ggml_tensor * down_p = ggml_mul_mat_id(ctx0, mcache->jit_down_c, act_p, ids_p);
                cb(down_p, "ffn_moe_jit_down", il);
                down_c = ggml_add(ctx0, down_c, down_p);
            }
            down_g = down_g ? ggml_concat(ctx0, down_g, down_c, 2) : down_c;
        }

        ggml_build_forward_expand(gf, down_g);
    }

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    // the routed experts' chain (up/gate, activation, down) for expert ids ids_x
    auto expert_chain = [&](ggml_tensor * cur_in, ggml_tensor * ids_x) -> ggml_tensor * {
        ggml_tensor * cur = cur_in;
        // the expert biases are gathered by id: a card's share of the experts has -1 ids (see moe_ids_for_gather)
        ggml_tensor * ids_b = nullptr;
        auto bias_ids = [&]() { if (!ids_b) { ids_b = mm_id_backend ? moe_ids_for_gather(ctx0, ids_x, n_expert, sched, mm_id_backend) : ids_x; } return ids_b; };
        ggml_tensor * up = nullptr;
        ggml_tensor * experts = nullptr;

        if (gate_up_exps) {
            // merged gate_up path: one mul_mat_id, then split into gate and up views
            ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, ids_x, up_exps_s, slots); // [n_ff*2, n_expert_used, n_tokens]
            cb(gate_up, "ffn_moe_gate_up", il);

            if (up_exps_s) {
                cb(gate_up, "ffn_moe_gate_up_scaled", il);
            }

            if (gate_up_exps_b) {
                gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, bias_ids());
                cb(gate_up, "ffn_moe_gate_up_biased", il);
            }

            const int64_t n_ff = gate_up->ne[0] / 2;
            cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            cb(cur, "ffn_moe_gate", il);
            up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
            cb(up, "ffn_moe_up", il);
        } else {
            // separate gate and up path
            up = build_lora_mm_id(up_exps, cur, ids_x, up_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
            cb(up, "ffn_moe_up", il);

            if (mcache) {
                up->src[3] = mcache->host_table;
                up->op_params[0] = mcache->n_slots;
                if (mc_cpu_backend) {
                    // must run where the skip table is honored (CPU), never op-offloaded
                    ggml_backend_sched_set_tensor_backend(sched, up, mc_cpu_backend);
                }
            }

            if (up_exps_s) {
                cb(up, "ffn_moe_up_scaled", il);
            }

            if (up_exps_b) {
                up = ggml_add_id(ctx0, up, up_exps_b, bias_ids());
                cb(up, "ffn_moe_up_biased", il);
            }

            if (gate_exps) {
                cur = build_lora_mm_id(gate_exps, cur, ids_x, gate_exps_s, slots); // [n_ff, n_expert_used, n_tokens]
                cb(cur, "ffn_moe_gate", il);

                if (mcache) {
                    if (mc_pred) {
                        cur->src[4] = mc_pred;
                    }
                    if (probs && llama_moe_cache_wants_probs_fork()) {
                        cur->src[6] = probs;   // policy r: the router's probability of every expert for these tokens, the relevance the cache scores by
                    }
                    cur->src[3] = mcache->host_table;
                    cur->op_params[0] = mcache->n_slots;
                    if (mc_cpu_backend) {
                        ggml_backend_sched_set_tensor_backend(sched, cur, mc_cpu_backend);
                    }
                }
            } else {
                cur = up;
            }

            if (gate_exps_s) {
                cb(cur, "ffn_moe_gate_scaled", il);
            }

            if (gate_exps_b) {
                cur = ggml_add_id(ctx0, cur, gate_exps_b, bias_ids());
                cb(cur, "ffn_moe_gate_biased", il);
            }
        }

        const bool has_gate = gate_exps || gate_up_exps;

        switch (type_op) {
            case LLM_FFN_SILU:
                if (gate_exps) {
                    if (il >= 0) {
                        const float limit = hparams.swiglu_clamp_exp[il];
                        constexpr float eps = 1e-6f;
                        if (limit > eps) {
                            if (moe_uses_swiglu_clamp(arch, hparams)) {
                                cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                            } else {
                                up = ggml_clamp(ctx0, up, -limit, limit);
                                cb(up, "ffn_moe_up_clamped", il);
                                ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                                cb(gate_act, "ffn_moe_silu", il);
                                gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                                cb(gate_act, "ffn_moe_silu_clamped", il);
                                cur = ggml_mul(ctx0, gate_act, up);
                            }
                            cb(cur, "ffn_moe_swiglu_limited", il);
                            break;
                        }
                    }
                }

                if (has_gate) {
                    cur = ggml_swiglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_SITU:
                {
                    // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                    GGML_ASSERT(has_gate);
                    const float beta = hparams.situ_beta;
                    const float lb   = hparams.situ_linear_beta;

                    ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                    act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                    if (lb > 0.0f) {
                        up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                    }
                    cur = ggml_mul(ctx0, act, up);
                    cb(cur, "ffn_moe_situ", il);
                } break;
            case LLM_FFN_GELU:
                if (has_gate) {
                    cur = ggml_geglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_moe_gelu", il);
                } break;
            case LLM_FFN_SWIGLU_OAI_MOE:
                {
                    // TODO: move to hparams?
                    constexpr float alpha = 1.702f;
                    constexpr float limit = 7.0f;
                    cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                    cb(cur, "ffn_moe_swiglu_oai", il);
                } break;
            case LLM_FFN_RELU:
                if (has_gate) {
                    cur = ggml_reglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_reglu", il);
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_moe_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                if (has_gate) {
                    // TODO: add support for gated squared relu
                    GGML_ABORT("fatal error: gated squared relu not implemented");
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_moe_relu_sqr", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        // upstream (#30656): src1 of the down projection can exceed the F16 range on these architectures, so it is rescaled by a power of two and the product scaled back
        // (the amax op cannot be split across devices, so these architectures are denied -sm tensor [TAG_MOE_AMAX_SCALE])
        llm_graph_amax_src1 amax_src1 = { nullptr, nullptr };
        if (arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_MELLUM || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_LAGUNA) {
            amax_src1 = build_amax_src1_f16(cur);
            cur = amax_src1.res;
        }
        experts = build_lora_mm_id(down_exps, cur, ids_x, down_exps_s, slots); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);
        if (amax_src1.scale_s) {
            experts = ggml_mul(ctx0, experts, amax_src1.scale_s);
        }
        return experts;
    };

    // multi-GPU prefill (big batches, no MoE cache chain): the routed experts are split over the GPUs in contiguous id
    // ranges, each GPU streams its own range over its own PCIe link and computes it; ids outside a GPU's range are -1
    // (skipped, zero rows); each GPU weights and sums its experts and the partial outputs are added. Shares:
    // (1 - alpha) * [fastest link] + alpha * (link GB/s / sum), alpha measured by the MoE cache's prefill tuner
    // (1 = all GPUs by bandwidth, 0 = fastest only) or fixed by LLAMA_PREFILL_SPLIT
    struct split_part { ggml_backend_t b; double gbs; int64_t lo, hi; };
    std::vector<split_part> parts;
    if (!mcache && n_tokens >= 32 && sched && !up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps_b && loras->empty()) {
        const float alpha = llama_moe_cache_prefill_alpha_fork();
        // fork: a dense split pass (moe_ranges) shares the experts over the cards of its ranges by the live balance's per-card shares (all on the expert card
        // while the helpers hold none), whatever the expert-only split's alpha says. alpha is 0 by default since 2026-10-23 (MiMo fix), and this block then
        // never ran in a dense split: every expert stayed on the expert card and the balance's expert trial changed nothing but its log line.
        const bool range_mode = !moe_ranges.empty();
        if ((alpha > 0.0f || range_mode) && llama_dsplit_cur_fork() != 0) {   // fork: --moe dense-split=auto chose one card for this pass: no expert split either
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
                ggml_backend_dev_t d = ggml_backend_get_device(b);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    bool in_ranges = !range_mode;   // range mode: only the cards that hold a range of this pass
                    // (all ranges of the pass, ds_r, not only this call's: a builder may call once per range, see llama_model_qwen4exp::graph)
                    for (const auto & rg : ds_r) { in_ranges = in_ranges || rg.b == b; }
                    const double g = llama_moe_cache_link_gbs_fork(d);
                    if (g > 0 && in_ranges) { parts.push_back({ b, g, 0, 0 }); }
                }
            }
            if (parts.size() >= 2) {
                double sum = 0; size_t fast = 0;
                for (size_t i = 0; i < parts.size(); ++i) { sum += parts[i].gbs; if (parts[i].gbs > parts[fast].gbs) { fast = i; } }
                // slowest first: their chains are issued first, so they stream while the faster GPUs work
                const ggml_backend_t fast_b = parts[fast].b;
                // --moe split-order=fast issues the fastest link's chain first. With the staging slots in place the slow link has nothing left to "stream while the others
                // work", and a chain issued first holds the others back: nsys, GLM 8k, split 0.5: the x4's chain (30 ms) ran alone, then the x16's (111 ms), per layer.
                static const bool fast_first = [] { const char * e = llama_moe_get_opt("split-order"); return e && strcmp(e, "fast") == 0; }();
                std::sort(parts.begin(), parts.end(), [](const split_part & x, const split_part & y) { return fast_first ? x.gbs > y.gbs : x.gbs < y.gbs; });
                // --moe split-slow-share=F: the cards that are not on the fastest link hold F of the experts together (e.g. 0.25: a quarter of the experts is uploaded to and
                // computed on the x4 card, the rest on the x16 card), split among themselves by link rate. Without it the share is alpha x the link share, which for
                // x16 25.8 + x4 6.5 GB/s is at most 20% (alpha 1) and 10% at alpha 0.5: measured by nsys, GLM 8k, the x4 card then received 11% of the uploaded bytes.
                static const double slow_share = [] { const char * e = llama_moe_get_opt("split-slow-share"); return e ? atof(e) : -1.0; }();
                double slow_sum = 0.0, top = 0.0;
                for (const auto & pt : parts) { top = std::max(top, pt.gbs); }
                for (const auto & pt : parts) { if (pt.gbs < top) { slow_sum += pt.gbs; } }
                // The counts come from the share controller (see llama_split_ctl above) when the layer has an owner among the cards and nothing fixes the shares by
                // hand; otherwise from the upload rates as before.
                static const bool ctl_on = split_opt("split-chunks", 1) <= 1;   // --moe split-auto=0 stops the tuning (llama_split_ctl_poll_fork), not the table
                std::vector<int64_t> n_of;
                if (ctl_on && slow_share < 0.0 && gate_inp && gate_inp->buffer) {
                    // range mode (dense split): every card hands its partial results on, the router runs on the expert card's range: that card is the reference
                    const ggml_backend_dev_t od = !moe_ranges.empty() ? ggml_backend_get_device(ds_r[llama_ds_xr_fork(ds_r)].b)
                                                                      : ggml_backend_buft_get_device(ggml_backend_buffer_get_type(gate_inp->buffer));
                    std::vector<std::pair<ggml_backend_dev_t, double>> cards;
                    size_t owner_i = parts.size();
                    for (size_t i = 0; i < parts.size(); ++i) {
                        cards.push_back({ ggml_backend_get_device(parts[i].b), parts[i].gbs });
                        if (od && cards.back().first == od) { owner_i = i; }
                    }
                    const double bytes_layer = (double) ((up_exps ? ggml_nbytes(up_exps) : 0) + (gate_exps ? ggml_nbytes(gate_exps) : 0) + (down_exps ? ggml_nbytes(down_exps) : 0));
                    if (owner_i < parts.size()) { n_of = split_ctl_counts(il, n_tokens, cards, owner_i, n_expert, n_embd, bytes_layer, !moe_ranges.empty()); }
                    // --moe split-pattern=a:b:c (tests): the experts every helper holds, per layer, cycled over the layers (0 = helper off, the owner keeps the
                    // rest and at least one). The controller's results depend on timing; this gives every shape it can produce on demand: a layer with one
                    // card only next to split layers, a card with a single expert, a card with all but one. Each must give the same perplexity as one card.
                    static const std::vector<int64_t> pattern = [] {
                        std::vector<int64_t> p;
                        const char * e = llama_moe_get_opt("split-pattern");
                        for (const char * s = e; s && *s; ) { p.push_back(atoll(s)); s = strchr(s, ':'); if (s) { ++s; } }
                        return p;
                    }();
                    // --moe split-pattern-after=N: the pattern applies once the controller has read N measurements (the start shares before that). A change
                    // of shape from one graph to the next is what the controller does after its first measurement; this makes that change repeatable.
                    static const long pattern_after = [] { const char * e = llama_moe_get_opt("split-pattern-after"); return e ? atol(e) : 0L; }();
                    if (!pattern.empty() && owner_i < parts.size() && g_split_ctl.n_updates >= pattern_after) {
                        n_of.assign(parts.size(), 0);
                        int64_t used = 0;
                        for (size_t i = 0; i < parts.size(); ++i) {
                            if (i == owner_i) { continue; }
                            n_of[i] = std::max<int64_t>(0, std::min<int64_t>(n_expert - 1 - used, pattern[(size_t) il % pattern.size()]));
                            used += n_of[i];
                        }
                        n_of[owner_i] = n_expert - used;
                    }
                    if (!n_of.empty() && getenv("LLAMA_SPLIT_CTL_DEBUG") && atoi(getenv("LLAMA_SPLIT_CTL_DEBUG")) >= 2) {
                        std::string txt;
                        for (size_t i = 0; i < parts.size(); ++i) {
                            txt += std::string(" ") + ggml_backend_name(parts[i].b) + (i == owner_i ? "(owner) " : " ") + std::to_string(n_of[i]);
                        }
                        LLAMA_LOG_WARN("prefill split: layer %d, %lld tokens:%s\n", il, (long long) n_tokens, txt.c_str());
                    }
                }
                int64_t lo = 0;
                std::vector<split_part> kept;
                for (size_t i = 0; i < parts.size(); ++i) {
                    double share = (1.0 - alpha) * (parts[i].b == fast_b) + alpha * parts[i].gbs / sum;
                    if (slow_share >= 0.0 && slow_sum > 0.0) {
                        share = parts[i].gbs < top ? slow_share * parts[i].gbs / slow_sum : (1.0 - slow_share) * parts[i].gbs / (sum - slow_sum);
                    }
                    int64_t hi = i + 1 == parts.size() ? n_expert : std::min<int64_t>(n_expert, lo + llround(share * n_expert));
                    if (!n_of.empty()) { hi = std::min<int64_t>(n_expert, lo + n_of[i]); }
                    if (hi > lo) { kept.push_back({ parts[i].b, parts[i].gbs, lo, hi }); }
                    lo = hi;
                }
                parts = kept;
                // --moe split-chunks=N: every card's expert range is cut into N blocks (more parts on the same card). A block is staged and uploaded on its own and its
                // matmul starts when ITS experts have arrived, while the next block uploads: the card computes in step with its transfer inside the layer, instead of
                // waiting for the whole range (nsys, GLM 8k: the x16 card did no compute while a layer's tensor of 1.8 GB arrived, about 70 ms per tensor). The blocks of a
                // card keep their order after the other card's blocks of the same layer; each is one more mask, one more MUL_MAT_ID per tensor and one more sum.
                static const int chunks = [] { const char * e = llama_moe_get_opt("split-chunks"); return e ? std::max(1, atoi(e)) : 1; }();
                if (chunks > 1) {
                    std::vector<split_part> cut;
                    for (const auto & pt : parts) {
                        for (int ci = 0; ci < chunks; ++ci) {
                            const int64_t clo = pt.lo + (pt.hi - pt.lo) * ci / chunks, chi = pt.lo + (pt.hi - pt.lo) * (ci + 1) / chunks;
                            if (chi > clo) { cut.push_back({ pt.b, pt.gbs, clo, chi }); }
                        }
                    }
                    parts = cut;
                }
                // One card may be left here (the share controller switched the other cards off for this layer, or a share rounded to no expert): the layer
                // stays in the split path with that one card holding every expert. It used to drop to the generic chain, whose matmuls are not pinned: the
                // scheduler then gave them to the first backend that takes host weights, whichever card the controller had meant, in the middle of a graph
                // whose other layers are pinned. Measured on Qwen3.8-Flash-Next with the x4-linked card listed first: 17 of 48 layers switched off after the
                // first pass, their experts all went to the x4 card, and the second perplexity chunk came out NaN (one card: 1.1086). A graph is now split
                // in every expert layer or in none. --moe split-single=generic restores the old behaviour (to reproduce that fault).
                static const bool single_generic = [] { const char * e = llama_moe_get_opt("split-single"); return e && strcmp(e, "generic") == 0; }();
                if (single_generic && parts.size() < 2) { parts.clear(); }
            } else {
                parts.clear();   // one GPU: nothing to split
            }
        }
    }
    // fork: dense split (moe_ranges, set by the model builder): the split path is taken in any case; without an expert split, the expert card (the fastest
    // link, see llama_model_glm5_next::graph; its range by llama_ds_xr_fork) holds all experts, also when this call carries only another card's range
    if (!moe_ranges.empty() && parts.empty() && sched) {
        parts.push_back({ ds_r[llama_ds_xr_fork(ds_r)].b, llama_moe_cache_link_gbs_fork(ggml_backend_get_device(ds_r[llama_ds_xr_fork(ds_r)].b)), 0, n_expert });
    }
    ggml_tensor * experts = nullptr;
    if (!parts.empty()) {
        // The OWNER of the layer is the card that holds its dense weights (attention, norms, router): the router weight says which. Nothing below assumes two cards or
        // a particular link: any number of cards, each with its own expert range, and the owner may be any of them (or none: a layer whose dense part is on the CPU).
        // Three rules keep the owner's compute stream free of the other cards (nsys, OLMoE 3.4k tokens, x16 owner + x4 helper, the owner idle 27 ms of a 77 ms layer):
        //  1. The id helpers f and f1 are put in the graph here, right behind the router, so they are computed by the owner. Left to the first id mask that reads
        //     them, f1 landed among the FIRST chain's pinned nodes, i.e. on the helper card, and the owner's own id mask then waited for a copy back from the helper.
        //  2. The owner's id mask is built first (it joins the owner's dense split), the owner's expert chain last: the helpers' input copies are issued before the
        //     owner's matmuls are queued, so they leave the owner right after the dense part.
        //  3. The partial sums are added on the owner AFTER its own chain: sum = owner's part, then + each helper's part. The scheduler starts a new split at an add
        //     that reads another card's tensor (see pass 5 in ggml_backend_sched_split_graph), so the owner's own matmuls never wait for a helper's result.
        ggml_backend_t owner_b = nullptr;
        if (gate_inp && gate_inp->buffer) {
            const ggml_backend_dev_t owner_dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(gate_inp->buffer));
            for (const auto & pt : parts) { if (owner_dev && ggml_backend_get_device(pt.b) == owner_dev) { owner_b = pt.b; } }
        }
        if (!moe_ranges.empty()) {
            // range mode: the expert card (llama_ds_xr_fork) plays the owner for the id helpers
            owner_b = nullptr;
            for (const auto & pt : parts) { if (pt.b == ds_r[llama_ds_xr_fork(ds_r)].b) { owner_b = pt.b; } }
        }
        if (owner_b) {
            std::stable_partition(parts.begin(), parts.end(), [&](const split_part & pt) { return pt.b != owner_b; });   // helpers first, the owner's chain(s) last
        }
        // id in [lo, hi) -> kept, else -1 (on the integer-valued floats: clamp(id - lo + 1, 0, 1) * clamp(hi - id, 0, 1))
        ggml_tensor * f  = ggml_cast(ctx0, ggml_cont(ctx0, selected_experts), GGML_TYPE_F32);
        ggml_tensor * f1 = ggml_scale_bias(ctx0, f, 1.0f, 1.0f);
        ggml_build_forward_expand(gf, f1);   // rule 1
        std::vector<ggml_tensor *> ids(parts.size());
        // Rule 4: a helper takes its own copy of the router weights together with its id mask, at the very start of its chain. Read straight from the owner by the
        // mul behind the helper's `down` matmul, the 110 KB copy was issued behind that matmul and stood on the helper's link behind the helper's own expert upload
        // of the next layer (nsys, same run: 3.5 ms, which the owner then waited for the partial sum). Early, it travels with the id helpers while the link is free.
        std::vector<ggml_tensor *> w_part(parts.size(), nullptr);
        // Rule 5: what crosses the cards crosses as f16. The hidden state goes to a helper and its partial sum comes back, both [n_embd, n_tokens] f32, over
        // the helper's link in both directions: GLM-5.3-Flash at 8k tokens 131 MB each way, 20 ms + 21 ms of a x4 link per layer, which is what bounds the
        // share the helper can finish in time (and 5 ms + 5 ms of the owner's stream). Half the bytes as f16: the owner casts once for all helpers, a helper
        // casts back in its mask split (so the hidden state arrives with the id helpers, at the very start of its chain), and returns its sum as f16.
        // The expert matmuls quantise their input to 8 bits anyway; the partial sum loses bits 12 and below. --moe split-f16=0 keeps f32, split-f16=in
        // sends the hidden state as f16 and takes the sum back as f32 (llama_dsplit_hop_mode_fork: what each choice costs in perplexity and speed).
        // the dense split (token ranges) has its own default, f16 both ways (llama_dsplit_hop_mode_range_fork); the expert-only split keeps f32 unless asked
        const int  hop_mode    = moe_ranges.empty() ? llama_dsplit_hop_mode_fork() : llama_dsplit_hop_mode_range_fork();
        const bool hop_f16     = hop_mode != 0;   // the hidden state, to the card that computes experts on it
        const bool hop_f16_out = hop_mode == 1;   // the experts' sum, back to the card that holds the tokens
        ggml_tensor * cur16 = nullptr;
        std::vector<ggml_tensor *> cur_part(parts.size(), nullptr);
        if (owner_b && hop_f16 && cur->type == GGML_TYPE_F32 && moe_ranges.empty()) {
            for (const auto & pt : parts) { if (pt.b != owner_b) { cur16 = cur16 ? cur16 : ggml_cast(ctx0, cur, GGML_TYPE_F16); } }
            if (cur16) {
                ggml_backend_sched_set_tensor_backend(sched, cur16, owner_b);
                ggml_build_forward_expand(gf, cur16);
            }
        }
        std::vector<size_t> mask_order;      // rule 2: the owner's masks first
        for (size_t i = 0; i < parts.size(); ++i) { if (parts[i].b == owner_b) { mask_order.push_back(i); } }
        for (size_t i = 0; i < parts.size(); ++i) { if (parts[i].b != owner_b) { mask_order.push_back(i); } }
        for (size_t i : mask_order) {
            // Every operation of a card's id mask is pinned to that card. Left free, the scheduler put them next to the first chain, so the second chain took
            // its ids from the first chain's split and had to wait for it (nsys, GLM 8k: the x16's chain started when the x4's ended).
            auto pin = [&](ggml_tensor * t) { ggml_backend_sched_set_tensor_backend(sched, t, parts[i].b); return t; };
            ggml_tensor * lo_m = pin(ggml_clamp(ctx0, pin(ggml_scale_bias(ctx0, f,  1.0f, (float) (1 - parts[i].lo))), 0.0f, 1.0f));
            ggml_tensor * hi_m = pin(ggml_clamp(ctx0, pin(ggml_scale_bias(ctx0, f, -1.0f, (float) parts[i].hi)),       0.0f, 1.0f));
            ggml_tensor * m = pin(ggml_mul(ctx0, lo_m, hi_m));
            ids[i] = pin(ggml_cast(ctx0, pin(ggml_scale_bias(ctx0, pin(ggml_mul(ctx0, m, f1)), 1.0f, -1.0f)), GGML_TYPE_I32));
            cb(ids[i], "ffn_moe_ids_part", il);
            // all id sets before any chain: the scheduler copies only the used experts of a host weight when the
            // MUL_MAT_ID is the first node of its split (the ids must exist before the split starts)
            ggml_build_forward_expand(gf, ids[i]);
            if (owner_b && parts[i].b != owner_b && !weight_before_ffn) {   // rule 4
                w_part[i] = pin(ggml_cont(ctx0, weights));
                ggml_build_forward_expand(gf, w_part[i]);
            }
            if (cur16 && parts[i].b != owner_b) {                           // rule 5
                cur_part[i] = pin(ggml_cast(ctx0, cur16, GGML_TYPE_F32));
                ggml_build_forward_expand(gf, cur_part[i]);
            }
        }
        // each GPU weights and sums its experts ([n_embd, n_tokens] crosses between the GPUs, not [n_embd, n_expert_used, n_tokens])
        auto reduce = [&](ggml_tensor * e, ggml_backend_t b, ggml_tensor * w) {
            if (!weight_before_ffn) {
                e = ggml_mul(ctx0, e, w ? w : weights);
                ggml_backend_sched_set_tensor_backend(sched, e, b);
            }
            ggml_tensor * sum = ggml_view_2d(ctx0, e, n_embd, n_tokens, e->nb[2], 0);
            for (int64_t i = 1; i < (int64_t) hparams.n_expert_used(il); ++i) {
                sum = ggml_add(ctx0, sum, ggml_view_2d(ctx0, e, n_embd, n_tokens, e->nb[2], i*e->nb[1]));
                ggml_backend_sched_set_tensor_backend(sched, sum, b);
            }
            if (hparams.n_expert_used(il) == 1) {
                sum = ggml_cont(ctx0, sum);
                ggml_backend_sched_set_tensor_backend(sched, sum, b);
            }
            return sum;
        };
        // fork: dense split. Every card's expert chain reads the hidden state of ALL tokens: the ranges on its own card as they are, the other cards' ranges
        // copied over as f16 (cast once on the range's card, back to f32 on the reading card), concatenated in token order on the reading card.
        std::map<ggml_backend_t, ggml_tensor *> rin_cache;
        std::vector<ggml_tensor *> r16(moe_ranges.size(), nullptr);
        auto range_in = [&](ggml_backend_t b) -> ggml_tensor * {
            auto it = rin_cache.find(b);
            if (it != rin_cache.end()) { return it->second; }
            ggml_tensor * acc = nullptr;
            for (size_t r = 0; r < moe_ranges.size(); ++r) {
                ggml_tensor * x = moe_ranges[r].cur;   // [n_embd, tokens of the range] on the range's card
                if (moe_ranges[r].b != b) {
                    if (hop_f16 && x->type == GGML_TYPE_F32) {
                        if (!r16[r]) {
                            r16[r] = ggml_cast(ctx0, x, GGML_TYPE_F16);
                            ggml_backend_sched_set_tensor_backend(sched, r16[r], moe_ranges[r].b);
                            ggml_build_forward_expand(gf, r16[r]);
                        }
                        x = ggml_cast(ctx0, r16[r], GGML_TYPE_F32);
                    } else {
                        x = ggml_cont(ctx0, x);
                    }
                    ggml_backend_sched_set_tensor_backend(sched, x, b);
                }
                if (acc) {
                    acc = ggml_concat(ctx0, acc, x, 1);
                    ggml_backend_sched_set_tensor_backend(sched, acc, b);
                } else {
                    acc = x;
                }
            }
            acc = ggml_reshape_3d(ctx0, acc, n_embd, 1, n_tokens);   // the chain's input layout (cur is [n_embd, 1, n_tokens] here)
            rin_cache[b] = acc;
            return acc;
        };
        if (!moe_ranges.empty()) {
            // Every card's input is put in the graph BEFORE any expert chain, so the copies of the other cards' ranges are issued right behind their dense part.
            // Built next to its chain, the copy of the x4 card's range for the x16's chain was queued on the x4's stream behind the x4's whole chain (the read
            // half of a copy runs on the source card's stream, in order): the two chains ran one after the other (nsys, GLM-5.3-Flash 5.5k, dense split: the x16
            // waited 36 ms, then the x4 49 ms, per layer).
            for (size_t i = 0; i < parts.size(); ++i) { ggml_build_forward_expand(gf, range_in(parts[i].b)); }
        }
        std::vector<ggml_tensor *> partial(parts.size());
        for (size_t i = 0; i < parts.size(); ++i) {
            mm_id_backend = parts[i].b;
            mm_id_lo = (int32_t) parts[i].lo;
            mm_id_hi = (int32_t) parts[i].hi;
            ggml_tensor * chain_in = !moe_ranges.empty() ? range_in(parts[i].b) : (cur_part[i] ? cur_part[i] : cur);
            partial[i] = reduce(expert_chain(chain_in, ids[i]), parts[i].b, w_part[i]);
            mm_id_backend = nullptr;
            mm_id_lo = mm_id_hi = 0;
            if (cur_part[i] && hop_f16_out) {   // rule 5: the partial sum leaves the helper as f16 and becomes f32 again on the owner
                ggml_tensor * p16 = ggml_cast(ctx0, partial[i], GGML_TYPE_F16);
                ggml_backend_sched_set_tensor_backend(sched, p16, parts[i].b);
                ggml_build_forward_expand(gf, p16);
                partial[i] = ggml_cast(ctx0, p16, GGML_TYPE_F32);
                ggml_backend_sched_set_tensor_backend(sched, partial[i], owner_b);
                continue;
            }
            ggml_build_forward_expand(gf, partial[i]); // in this order: the helpers' chains issued first (slowest link first), the owner's last
        }
        if (!moe_ranges.empty()) {
            // fork: dense split. Each range gets the sum of all cards' partial results for its tokens, on the range's card: its own card's part first, the other
            // cards' parts of it cross as f16 (cast on the card that computed them, back to f32 on the range's card).
            for (auto & rg : moe_ranges) {
                const int64_t nr = rg.t1 - rg.t0;
                ggml_tensor * acc = nullptr;
                for (int pass = 0; pass < 2; ++pass) {
                    for (size_t i = 0; i < parts.size(); ++i) {
                        if ((parts[i].b == rg.b) != (pass == 0)) { continue; }
                        ggml_tensor * v = ggml_view_2d(ctx0, partial[i], n_embd, nr, partial[i]->nb[1], rg.t0 * partial[i]->nb[1]);
                        if (parts[i].b != rg.b) {
                            v = hop_f16_out ? ggml_cast(ctx0, v, GGML_TYPE_F16) : ggml_cont(ctx0, v);
                            ggml_backend_sched_set_tensor_backend(sched, v, parts[i].b);
                            ggml_build_forward_expand(gf, v);
                            if (hop_f16_out) {
                                v = ggml_cast(ctx0, v, GGML_TYPE_F32);
                                ggml_backend_sched_set_tensor_backend(sched, v, rg.b);
                            }
                        }
                        if (acc) {
                            acc = ggml_add(ctx0, acc, v);
                            ggml_backend_sched_set_tensor_backend(sched, acc, rg.b);
                        } else {
                            acc = v;
                        }
                    }
                }
                if (acc && acc->op == GGML_OP_VIEW) {   // one card holds all experts and it is this range's card: a contiguous copy of the rows
                    acc = ggml_cont(ctx0, acc);
                    ggml_backend_sched_set_tensor_backend(sched, acc, rg.b);
                }
                cb(acc, "ffn_moe_out", il);
                rg.out = acc;
            }
            return moe_ranges.back().out;
        }
        // rule 3: every partial sum goes straight to the owner and is added there, the owner's own part first. (Added as the chains were built, the sum of two
        // helpers was computed on the second helper: one card's result crossed to another helper and only then to the owner.)
        ggml_tensor * moe_out = nullptr;
        auto add_part = [&](size_t i) {
            if (!moe_out) { moe_out = partial[i]; return; }
            moe_out = ggml_add(ctx0, moe_out, partial[i]);
            if (owner_b) { ggml_backend_sched_set_tensor_backend(sched, moe_out, owner_b); }
        };
        for (size_t i = 0; i < parts.size(); ++i) { if (parts[i].b == owner_b) { add_part(i); } }
        for (size_t i = 0; i < parts.size(); ++i) { if (parts[i].b != owner_b) { add_part(i); } }
        cb(moe_out, "ffn_moe_out", il);
        return moe_out;
    } else {
        experts = expert_chain(cur, selected_experts);
    }

    if (mcache) {
        experts->src[3] = mcache->host_table;
        experts->op_params[0] = mcache->n_slots;
        if (mc_cpu_backend) {
            ggml_backend_sched_set_tensor_backend(sched, experts, mc_cpu_backend);
        }

        experts = ggml_add(ctx0, experts, down_g);
        cb(experts, "ffn_moe_cache_merged", il);
    }

    if (mc_defer_pk > 0) {
        // deferred ids resolved inside this layer's immediate host split, where the tables can't change (they only
        // change in split callbacks and step()) and match what the GPU chain used: cached ids -> -1, so the deferred
        // chain computes exactly the host misses among the deferred experts, whatever gets published before it runs
        // the immediate host run's down op (before the merge with the GPU chain) orders this after that run
        ggml_tensor * imm_dep = (experts->op == GGML_OP_ADD && experts->src[0]->op == GGML_OP_MUL_MAT_ID) ? experts->src[0] : experts;
        mc_ids_def = ggml_map_custom3(ctx0, mc_ids_all, mc_defer_mask, imm_dep,
                [](ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * m, const ggml_tensor *, int ith, int, void * ud) {
                    if (ith != 0) { return; }
                    const llama_moe_cache_layer_fork * mc = (const llama_moe_cache_layer_fork *) ud;
                    const int32_t * tbl = (const int32_t *) mc->host_table->data;
                    for (int64_t t = 0; t < a->ne[1]; ++t) {
                        for (int64_t i = 0; i < a->ne[0]; ++i) {
                            const int32_t v = *(const int32_t *) ((const char *) a->data + t*a->nb[1] + i*a->nb[0]);
                            const int32_t d = *(const int32_t *) ((const char *) m->data + t*m->nb[1] + i*m->nb[0]);
                            const bool keep = d && v >= 0 && tbl[v] == mc->n_slots; // deferred and still uncached
                            *(int32_t *) ((char *) dst->data + t*dst->nb[1] + i*dst->nb[0]) = keep ? v : -1;
                        }
                    }
                }, 1, (void *) mcache);
        ggml_backend_sched_set_tensor_backend(sched, mc_ids_def, mc_cpu_backend);
        cb(mc_ids_def, "ffn_moe_defer_ids", il);
        ggml_build_forward_expand(gf, mc_ids_def);
        // frozen "nothing cached" table for the deferred chain (the live table may change before it runs)
        const int32_t n_slots_c = mcache->n_slots;
        ggml_tensor * frozen = ggml_map_custom1(ctx0, mcache->host_table,
                [](ggml_tensor * dst, const ggml_tensor *, int ith, int, void * ud) {
                    if (ith != 0) { return; }
                    const int32_t v = (int32_t) (intptr_t) ud;
                    for (int64_t i = 0; i < ggml_nelements(dst); ++i) { ((int32_t *) dst->data)[i] = v; }
                }, 1, (void *) (intptr_t) n_slots_c);
        ggml_backend_sched_set_tensor_backend(sched, frozen, mc_cpu_backend);
        cb(frozen, "ffn_moe_defer_tbl", il);
        mc_defer_tbl = frozen;
    }
    if (mc_ids_def) {
        // host MUL_MAT_IDs of an expert chain, found from its down op
        std::function<void(ggml_tensor *, int, const std::function<void(ggml_tensor *)> &)> each_mmid =
                [&](ggml_tensor * t, int depth, const std::function<void(ggml_tensor *)> & f) {
            if (!t || depth > 4) { return; }
            if (t->op == GGML_OP_MUL_MAT_ID) { f(t); }
            for (int j = 0; j < 2; ++j) { each_mmid(t->src[j], depth + 1, f); }
        };
        // the immediate gate op observes the layer's full selection for the cache (its own ids hide the deferred ones)
        each_mmid(experts, 0, [&](ggml_tensor * t) {
            if (t->src[2] == selected_experts && strstr(t->src[0]->name, "gate_exps")) { t->src[5] = mc_ids_all; if (probs && llama_moe_cache_wants_probs_fork()) { t->src[6] = probs; } }
        });
        ggml_tensor * dexp = expert_chain(cur, mc_ids_def);
        each_mmid(dexp, 0, [&](ggml_tensor * t) {
            t->src[3] = mc_defer_tbl;
            t->op_params[0] = mcache->n_slots;
            t->src[4] = nullptr; // no predictor handoff on the deferred gate op
            const char * kind = strstr(t->src[0]->name, "gate_exps") ? "gate" : strstr(t->src[0]->name, "up_exps") ? "up" : "down";
            ggml_format_name(t, "ffn_moe_defer_%s-%d", kind, il);
            ggml_backend_sched_set_tensor_backend(sched, t, mc_cpu_backend);
        });
        // weighted sum over the deferred experts in one host op (rows of cached / immediate experts are zero; the
        // per-op dispatch of a mul + n adds cost more than the math)
        ggml_tensor * dsum = ggml_map_custom3(ctx0, ggml_view_2d(ctx0, dexp, n_embd, n_tokens, dexp->nb[2], 0), dexp, mc_w_host,
                [](ggml_tensor * dst, const ggml_tensor *, const ggml_tensor * x, const ggml_tensor * w, int ith, int nth, void *) {
                    const int64_t ne = x->ne[0], nu = x->ne[1], nt = x->ne[2];
                    const int64_t r0 = ne*ith/nth, r1 = ne*(ith + 1)/nth;
                    for (int64_t t = 0; t < nt; ++t) {
                        float * d = (float *) ((char *) dst->data + t*dst->nb[1]);
                        for (int64_t i = r0; i < r1; ++i) { d[i] = 0.0f; }
                        for (int64_t e = 0; e < nu; ++e) {
                            const float we = *(const float *) ((const char *) w->data + e*w->nb[1] + t*w->nb[2]);
                            const float * xe = (const float *) ((const char *) x->data + e*x->nb[1] + t*x->nb[2]);
                            if (we == 0.0f) { continue; }
                            for (int64_t i = r0; i < r1; ++i) { d[i] += we*xe[i]; }
                        }
                    }
                }, GGML_N_TASKS_MAX, nullptr);
        ggml_backend_sched_set_tensor_backend(sched, dsum, mc_cpu_backend);
        cb(dsum, "ffn_moe_defer_out", il);
        moe_defer_pending = dsum; // expanded at the next MoE layer, after its attention
    }

    if (down_exps_s) {
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // an expert output can be scaled by a norm weight, then normalized, before the router weights apply
    if (post_exp_norm) {
        experts = ggml_mul(ctx0, experts, post_exp_norm);
        experts = ggml_rms_norm(ctx0, experts, post_exp_norm_eps);
        cb(experts, "ffn_moe_post_exp_norm", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    // Use per-layer n_expert_used to bound the graph even during warmup (avoids
    // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
    // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
    const uint32_t n_expert_used_il = hparams.n_expert_used(il);
    for (uint32_t i = 0; i < n_expert_used_il; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < n_expert_used_il; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used_il == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    if (defer_prev) {
        moe_out = ggml_add(ctx0, moe_out, defer_prev);
        cb(moe_out, "ffn_moe_out_with_deferred", il);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

ggml_tensor * llm_graph_context::build_moe_cache_slots(
         ggml_tensor * selected_experts,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * gate_up_exps,
                 int   il) const {
    if (moe_cache == nullptr) {
        return nullptr;
    }

    ggml_tensor * slot_map = moe_cache->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0]);
    if (slot_map == nullptr) {
        return nullptr;
    }
    for (ggml_tensor * w : { up_exps, gate_exps, down_exps, gate_up_exps }) {
        if (w != nullptr && moe_cache->get_experts(w) == nullptr) {
            return nullptr;
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(moe_cache->backend(il), slots)) {
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(sched, slots, moe_cache->backend(il));
    cb(slots, "ffn_moe_slots", il);

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(gf, slots);

    return ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]); // [n_expert_used, n_tokens]
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;

    const int64_t n_tok_rows = has_mixed ? llm_graph_n_tok_rows(ubatch) : 0;

    // we have 3 standard paths to produce the input embeddings for the first layer:
    // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
    // - embd1: pass raw embeddings, skipping the `tok_embd`
    // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
    ggml_tensor * embd0 = nullptr;
    ggml_tensor * embd1 = nullptr;
    ggml_tensor * embd2 = nullptr;

    // construct the input tensors
    {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        if (has_mixed) {
            inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
            cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
            ggml_set_input(inp->mixed_tokens);
        }
    }

    // the embeddings placeholders for the 3 paths
    // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
    // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
    {
        embd0 = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        ggml_build_forward_order(gf, embd0);

        embd1 = inp->embd;

        if (has_mixed) {
            embd2 = ggml_get_rows(ctx0, tok_embd, inp->mixed_tokens);
            ggml_build_forward_order(gf, embd2);
        }
    }

    // helper for extracting token embeddings with lora and padding
    // TODO: when lora is active, this is likely going to cause issues similar to https://github.com/ggml-org/llama.cpp/pull/30160
    //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
    auto build_tok = [&](ggml_tensor * cur, ggml_tensor * ids) {
        // a Hadamard-latent table stores rotated rows: restore the primal basis, h = s * (H z)
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd); it != hdmd->inv.end()) {
                cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);
                if (it->second.signs) {
                    cur = ggml_mul(ctx0, cur, it->second.signs);
                }
            }
        }

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(embd0, inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = embd1;

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    if (has_mixed) {
        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        ggml_tensor * embd_mixed = build_tok(embd2, inp->mixed_tokens);
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), embd_mixed, inp->mixed_slots);

        assert(ggml_are_same_shape (inps[0], inps[2]));
        assert(ggml_are_same_stride(inps[0], inps[2]));
    }

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    const bool scale_tok_only = hparams.f_embedding_scale != 0.0f && hparams.n_deepstack_layers > 0;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f && !scale_tok_only) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale*(scale_tok_only ? hparams.f_embedding_scale : 1.0f);
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd(), hparams.rope_section_order);

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset, hparams.attn_temp_pow);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    cb(cur, "inp_attn_scale", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);
    ggml_set_name(cur, "cross_embd");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_enc");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_dec");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        // fork: from which K/V length the gather runs, as measured at load (llama-dsplit-fork.cpp dsplit_att_bench; 0: the backend's own rule)
        if (const int32_t smin = llama_dsplit_sparse_min_fork(); smin > 0 && n_kv_max > 0) { ggml_flash_attn_ext_set_sparse_min(cur, smin); }
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_prec_set_acc(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the KV values are the tokens of the batch, padded like the KV cache (see llama_kv_cache::get_n_kv)
    const int64_t n_kv = GGML_PAD(n_tokens, 256);

    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    cb(inp->self_kq_mask, "self_kq_mask", -1);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_kv, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    // the KV values and the KQ bias are padded with zeros up to the KV dimension of the mask
    const int64_t n_kv_pad = kq_mask->ne[0] - k->ne[2];

    k = ggml_pad(ctx0, k, 0, 0, n_kv_pad, 0);
    v = ggml_pad(ctx0, v, 0, 0, n_kv_pad, 0);

    if (kq_b) {
        kq_b = ggml_pad(ctx0, kq_b, n_kv_pad, 0, 0, 0);
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    ggml_tensor * q = q_cur;
    ggml_tensor * k;
    ggml_tensor * v;

    if (cparams.training) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);

        k = k_cur;
        v = v_cur;
    } else {
        {
            const auto & k_idxs = inp->get_k_idxs();
            const auto & v_idxs = inp->get_v_idxs();

            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
        }

        k = mctx_cur->get_k(ctx0, il);
        v = mctx_cur->get_v(ctx0, il);
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_fused_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
                int   il) const {
    top_k = ggml_reshape_4d(ctx0, top_k, top_k->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);

    ggml_tensor * cur = ggml_top_k_mask(ctx0, kq_mask, top_k, nullptr, nullptr, nullptr);
    res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, cur, il});

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    // top_k indexes the cells, and the indexer mask of the same type is built the same way
    // let's use it, so when no other layer reads the MLA mask is not allocated
    ggml_tensor * kq_mask = inp->get_kq_mask_mla();
    if (inp->get_kq_mask_lid()->type == kq_mask->type) {
        kq_mask = inp->get_kq_mask_lid();
    }

    ggml_tensor * kq_mask_top_k = nullptr;
    if (cparams.fused_top_k_mask) {
        kq_mask_top_k = build_fused_top_k_mask(kq_mask, top_k, il);
    } else {
        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        // reshape KQ mask into tensor with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
        // this will be our source of zero values for unmasking top k mask elements
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        // modify KQ mask by unmasking elements that are in top_k indices
        // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
        kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // reshape to restore the original shape of KQ mask:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // combine with the original kq mask
        kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // whole seq fits into batch in training mode
    const bool use_kv_cur = cparams.training && k_cur && v_cur;
    if (use_kv_cur) {
        GGML_ASSERT(mctx_cur->get_n_kv() == n_tokens);
    }

    const bool store_kv = !use_kv_cur || hparams.n_layer_kv_from_start >= 0;

    // optionally store to KV cache
    if (store_kv && k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (store_kv && v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = use_kv_cur ? k_cur : mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = use_kv_cur ? v_cur : mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

    ggml_tensor * output_states = get_state_rows ?
        get_state_rows(ctx0, states, state_copy_main) :
        ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    inp->s_copy_main = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;

    return build_rs(s, state_copy, inp->s_copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        // a graph without logits still computes t_embd, so that it is never empty
        if (res->t_logits == nullptr && res->t_embd != nullptr) {
            ggml_build_forward_expand(gf, res->t_embd);
        }
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (hparams.pooling_type_cls == LLAMA_POOLING_TYPE_MEAN) {
                    // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_unary(ctx0, cur, hparams.act_cls);
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_logits() const {
    // a pooled embeddings context outputs one vector per sequence, so its graph skips the LM head
    if (cparams.embeddings && pooling_type != LLAMA_POOLING_TYPE_NONE) {
        res->t_logits = nullptr;
    }

    // the models declare t_logits without expanding it, so the LM head is computed only when it is expanded here
    if (res->t_logits != nullptr) {
        ggml_build_forward_expand(gf, res->t_logits);
    }
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // the padding row gives the chains without a row of the ubatch a valid input, even with no output at all
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> no_rows;

    // every sampler builds n_outputs_max_per_seq chains, like the reserve, so the graph keeps its topology
    // whatever rows the ubatch outputs: a chain without a row works on the first row and is not selected
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        const auto & rows = it != sampling_rows.end() ? it->second : no_rows;

        for (uint32_t i = 0; i < cparams.n_outputs_max_per_seq; ++i) {
            const bool     active = i < rows.size();
            const uint32_t row    = active ? rows[i] : 0;
            const int      i_out  = active ? 1       : 0;

            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[row] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[row] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[row] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[row] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}

// ---- fork: dense split kit ------------------------------------------------------------------------------------------------------------------------------------
// Shared by the model builders that cut a prompt pass into token ranges, one per card (see llama-dsplit-fork.cpp for the why and the measurements). A builder
// calls ds_plan() once; if it returns true, it builds every per-token block once per range between ds_skip() and ds_pin(card) (with the range's copy of the
// layer, llama_dsplit_layer_fork), the KV-cache attention with ds_attn_kv(), the experts with build_moe_ffn in range mode (moe_ranges), and joins the ranges
// for the output head with ds_join(). Recurrent blocks hand their state from range to range in the builder (the state of a range is the start of the next).

bool llama_dsplit_in_pass_loop_fork();   // llama-context.cpp

bool llm_graph_context::ds_plan(int64_t min_tokens) const {
    ds_r.clear();
    ds_cut.clear();
    ds_mark = nullptr;
    static const double  share0  = [] { const char * e = llama_moe_get_opt("dense-split-share"); return e ? atof(e) : -1.0; }();   // set: fixed by hand (tests)
    static const int64_t min_tok = [] { const char * e = llama_moe_get_opt("dense-split-min");   return e ? (int64_t) atoll(e) : (int64_t) 512; }();
    // --moe one-range=1 (research, off): a pass that runs on one card (the split planner or the measured choice said so) still takes the token-range path, as
    // ONE range on the card with the fastest link: every layer's dense part then runs there from that card's copies of the other cards' layers, instead of on
    // the card that owns the layer with two hand-offs per layer (nsys, Qwen3.8-Flash-Next 3k tokens on one x16: the x4's 8 layers cost about 0.27 s of a
    // 2.46 s pass). Measured (same model and prompt, four passes each): same perplexity as the plain one-card pass (1.1251 / 1.1086), but SLOWER, 1091-1146
    // prompt tokens/s against 1132-1211: the x16 card then computes the 8 layers the x4 card ran, and its pool is 635 MiB smaller by the copies.
    static const bool one_range = [] { const char * e = llama_moe_get_opt("one-range"); return e && atoi(e) != 0; }();
    const int cur = llama_dsplit_cur_fork();
    const bool single = cur == 0 && one_range;
    if (!sched || (cur != 1 && !single) || ubatch.n_seqs != 1 || cparams.n_rs_seq != 0 || n_tokens < std::max(min_tok, min_tokens)) {
        return false;
    }
    if (!loras->empty()) {
        return false;   // an adapter is looked up by the weight tensor: the copies on the other cards would run without it
    }
    // the cards of the split (llama-dsplit-fork.cpp, g_cards): every GPU whose copies have memory, the slowest link first and the expert card (the fastest
    // link) last; any number of cards, any links and compute. Each takes one range of the tokens, sized by the live balance (equal time per card).
    std::vector<ggml_backend_dev_t> cards;
    llama_dsplit_cards_fork(cards);
    std::vector<ggml_backend_t> be;
    std::vector<ggml_backend_dev_t> devs;
    for (ggml_backend_dev_t d : cards) {
        ggml_backend_t b = ds_backend(d);
        if (b) { be.push_back(b); devs.push_back(d); }
    }
    if (devs.size() < 2) {
        return false;
    }
    if (single) {
        ds_r.push_back({ be.back(), devs.back(), 0, n_tokens });
        return true;
    }
    // the token shares: the live balance's; --moe dense-split-share=X (tests) gives the first card X and the others the rest in equal parts
    std::vector<double> share;
    if (share0 >= 0.0) {
        share.assign(devs.size(), devs.size() > 1 ? (1.0 - std::min(1.0, share0)) / (double) (devs.size() - 1) : 1.0);
        share[0] = std::min(1.0, share0);
    } else {
        share = llama_dsplit_bal_shares_fork(n_tokens, devs);
    }
    if (share.size() != devs.size()) {
        share.assign(devs.size(), 1.0 / (double) devs.size());
    }
    // Tokens per card (user 2026-10-24): the expert card takes 100% of what fits it in EVERY pass, min(its card limit, the pass), and only the other cards vary.
    // They share the rest of the pass by the zero-wait shares, each at most what it finishes in the expert card's time (llama_dsplit_pass_size_fork sizes a
    // full pass so that the rest is exactly that). A pass no bigger than the expert card's limit -- a short prompt, the last pass of a call -- runs on the
    // expert card alone (one range). --moe dense-split-share (tests): the old cut by fixed shares.
    const size_t  xi   = devs.size() - 1;
    const int64_t minr = std::max<int64_t>(1, min_tokens);
    const int64_t lim  = (int64_t) llama_dsplit_card_lim_fork();
    std::vector<int64_t> cnt(devs.size(), 0);
    // Only a pass the pass loop sized for all cards (forced split, llama_context::decode) is bigger than the card limit: there the expert card takes its whole
    // limit. Every other pass (auto mode, --moe split-pass=0, a pass at most one card's micro-batch) is cut by the shares: with the expert-card-first rule there,
    // the expert card took the whole pass and the other cards got 0 tokens (GLM 3.0, 60k prompt: cuda0x4 0 tokens at a 36% share, waiting 53% of every pass).
    const bool pass_loop = llama_dsplit_in_pass_loop_fork();
    if (share0 < 0.0 && pass_loop) {
        cnt[xi] = lim > 0 ? std::min<int64_t>(lim, n_tokens) : n_tokens;
        const int64_t rest = n_tokens - cnt[xi];
        double hs = 0.0;
        for (size_t i = 0; i < xi; ++i) { hs += share[i]; }
        int64_t given = 0;
        int     last  = -1;
        for (size_t i = 0; i < xi && rest > 0; ++i) {
            cnt[i] = hs > 0.0 ? (int64_t) ((double) rest * share[i] / hs) : 0;
            given += cnt[i];
            last   = (int) i;
        }
        if (rest > 0) { if (last >= 0) { cnt[last] += rest - given; } else { cnt[xi] += rest; } }
        // a range shorter than the builder's minimum (the conv window) takes it from the expert card's range: only a call's last few tokens come out that short
        for (size_t i = 0; i < xi; ++i) {
            if (cnt[i] > 0 && cnt[i] < minr && cnt[xi] - (minr - cnt[i]) >= minr) { cnt[xi] -= minr - cnt[i]; cnt[i] = minr; }
        }
    } else {
        cnt = dsb_counts(n_tokens, share, minr, (int) xi);
    }
    // No card's range is bigger than the one-card pass limit (what one card's buffers hold, llama_dsplit_card_lim_fork): the tokens over a card's limit go to
    // the cards under theirs, by their shares (a helper whose share of the rest is over the limit, more than two cards).
    if (const int64_t cap = lim; cap > 0 && n_tokens > cap) {
        for (int round = 0; round < (int) cnt.size(); ++round) {   // the excess of one card can push another over its limit (more than two cards)
            int64_t extra = 0;
            for (auto & c : cnt) { if (c > cap) { extra += c - cap; c = cap; } }
            if (extra == 0) { break; }
            double room = 0.0;
            for (size_t i = 0; i < cnt.size(); ++i) { if (cnt[i] < cap) { room += share[i]; } }
            if (room <= 0.0) { cnt.back() += extra; break; }   // every card at its limit: cannot happen, a pass is at most limit / largest share
            int64_t given = 0;
            int last = -1;
            for (size_t i = 0; i < cnt.size(); ++i) {
                if (cnt[i] < cap && share[i] > 0.0) { const int64_t g = (int64_t) ((double) extra * share[i] / room); cnt[i] += g; given += g; last = (int) i; }
            }
            if (last >= 0) { cnt[last] += extra - given; } else { cnt.back() += extra; break; }
        }
    }
    // The ranges in token order, the expert card's range FIRST (2026-10-24; the other cards first before). A range continues the recurrent state, the conv
    // window and the KV of every range before it, layer by layer: the card with the later tokens waits, in every layer, for the cards with the earlier ones.
    // With the helpers first, that card was the expert card: Qwen3.8-Flash-Next 60k prompt on x16 + x4, the x16 card waited 19-37% of every pass for the x4
    // card's scan (13 + 30 ms a layer), while the x4 card waited for the x16's experts anyway (12k prompt, measured: 1832 -> 1925 prompt t/s, the x16's
    // wait 20% -> 14%). With the expert card first, its tokens depend on nothing the other cards compute in this pass; the other cards wait for it instead.
    // The expert card is found by llama_ds_xr_fork, not by position.
    std::vector<size_t> ord;
    ord.push_back(devs.size() - 1);
    for (size_t i = 0; i + 1 < devs.size(); ++i) { ord.push_back(i); }
    int64_t t0 = 0;
    for (size_t i : ord) {
        if (cnt[i] <= 0) { continue; }
        ds_r.push_back({ be[i], devs[i], t0, t0 + cnt[i] });
        t0 += cnt[i];
    }
    llama_dsplit_note_ranges_fork(devs, cnt);
    static int logged = 0;
    if (logged++ < 4) {
        std::string txt;
        for (const auto & r : ds_r) { txt += format(" %s %lld-%lld", ggml_backend_name(r.b), (long long) r.t0, (long long) r.t1); }
        LLAMA_LOG_WARN("dense split: graph of %lld tokens:%s\n", (long long) n_tokens, txt.c_str());
    }
    return true;
}

void llm_graph_context::ds_skip() const {
    ggml_tensor * t = ds_mark ? ds_mark : ggml_get_first_tensor(ctx0);
    if (t == nullptr) { return; }
    for (ggml_tensor * n = ggml_get_next_tensor(ctx0, t); n; n = ggml_get_next_tensor(ctx0, n)) { t = n; }
    ds_mark = t;
}

void llm_graph_context::ds_pin(ggml_backend_t b) const {
    for (ggml_tensor * t = ds_mark ? ggml_get_next_tensor(ctx0, ds_mark) : ggml_get_first_tensor(ctx0); t; t = ggml_get_next_tensor(ctx0, t)) {
        ds_mark = t;
        if (b == nullptr || t->op == GGML_OP_NONE || t->buffer != nullptr) { continue; }
        if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) { continue; }
        if (t->view_src != nullptr && t->view_src->buffer != nullptr) { continue; }   // writes into the memory (KV cache, recurrent state) run where it is
        ggml_backend_sched_set_tensor_backend(sched, t, b);
    }
}

ggml_backend_t llm_graph_context::ds_backend(ggml_backend_dev_t dev) const {
    for (int i = 0; sched && i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(b) == dev) { return b; }
    }
    return nullptr;
}

ggml_tensor * llm_graph_context::ds_rows(ggml_tensor * x, size_t r) const {
    const auto key = std::make_pair((const ggml_tensor *) x, r);
    auto it = ds_cut.find(key);
    if (it != ds_cut.end()) { return it->second; }
    int d = GGML_MAX_DIMS - 1;
    while (d > 0 && x->ne[d] == 1) { --d; }   // the token dim: the last one that is not 1
    int64_t ne[4] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
    ne[d] = ds_r[r].t1 - ds_r[r].t0;
    ggml_tensor * v = ggml_view_4d(ctx0, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2], x->nb[3], ds_r[r].t0 * x->nb[d]);
    ds_cut[key] = v;
    return v;
}

std::vector<ggml_tensor *> llm_graph_context::ds_slice(ggml_tensor * x) const {
    std::vector<ggml_tensor *> out;
    for (size_t r = 0; r < ds_r.size(); ++r) { out.push_back(ds_rows(x, r)); }
    return out;
}

std::vector<ggml_tensor *> llm_graph_context::ds_pos(ggml_tensor * inp_pos) const {
    const int64_t npe = hparams.n_pos_per_embd();
    if (npe <= 1) { return ds_slice(inp_pos); }
    const int64_t n_all = inp_pos->ne[0] / npe;
    const size_t  es    = ggml_element_size(inp_pos);
    std::vector<ggml_tensor *> out(ds_r.size());
    for (size_t r = 0; r < ds_r.size(); ++r) {
        const int64_t nr = ds_r[r].t1 - ds_r[r].t0;
        ds_skip();
        ggml_tensor * t = nullptr;
        for (int64_t k = 0; k < npe; ++k) {
            ggml_tensor * v = ggml_view_1d(ctx0, inp_pos, nr, (size_t) (k * n_all + ds_r[r].t0) * es);
            t = t ? ggml_concat(ctx0, t, v, 0) : v;
        }
        out[r] = t;
        ds_pin(ds_r[r].b);
    }
    return out;
}

std::vector<ggml_tensor *> llm_graph_context::ds_conv_chain(ggml_tensor * rows, ggml_tensor * states_all, size_t dst_off, const std::vector<ggml_tensor *> & xs,
                                                             int64_t state_cols, int64_t channels) const {
    std::vector<ggml_tensor *> out(ds_r.size());
    ds_skip();
    ggml_tensor * tail = ggml_reshape_3d(ctx0, rows, state_cols, channels, 1);   // the stored history, where the memory is
    for (size_t r = 0; r < ds_r.size(); ++r) {
        const int64_t nr = ds_r[r].t1 - ds_r[r].t0;
        ds_skip();
        ggml_tensor * in  = ggml_concat(ctx0, tail, ggml_transpose(ctx0, xs[r]), 0);
        ggml_tensor * win = ggml_cont(ctx0, ggml_view_3d(ctx0, in, state_cols, channels, 1, in->nb[1], in->nb[2], (size_t) nr * in->nb[0]));
        out[r] = in;
        ds_pin(ds_r[r].b);
        if (r + 1 == ds_r.size()) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, win, ggml_view_2d(ctx0, states_all, state_cols * channels, 1, states_all->nb[1], dst_off)));
            ds_skip();   // the write runs where the memory is
        } else {
            tail = win;
        }
    }
    return out;
}

ggml_tensor * llm_graph_context::ds_join(const std::vector<ggml_tensor *> & xs, int dim, ggml_backend_t b) const {
    ds_skip();
    ggml_tensor * acc = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) { acc = ggml_concat(ctx0, acc, xs[i], dim); }
    ds_pin(b);
    return acc;
}

std::vector<ggml_tensor *> llm_graph_context::ds_attn_kv(llm_graph_input_attn_kv * inp, const std::vector<ggml_tensor *> & q, const std::vector<ggml_tensor *> & k,
                                                          const std::vector<ggml_tensor *> & v, const std::vector<ggml_tensor *> & sinks, float kq_scale, int il) const {
    GGML_ASSERT(!inp->self_k_rot && !inp->self_v_rot);
    return ds_attn_cache(inp->mctx, inp->get_k_idxs(), inp->get_v_idxs(), inp->get_kq_mask(), q, k, v, sinks, kq_scale, il);
}

std::vector<ggml_tensor *> llm_graph_context::ds_attn_kv(llm_graph_input_attn_kv_iswa * inp, const std::vector<ggml_tensor *> & q, const std::vector<ggml_tensor *> & k,
                                                          const std::vector<ggml_tensor *> & v, const std::vector<ggml_tensor *> & sinks, float kq_scale, int il) const {
    const bool swa = hparams.is_swa(il);
    GGML_ASSERT(!(swa ? inp->self_k_rot_swa : inp->self_k_rot) && !(swa ? inp->self_v_rot_swa : inp->self_v_rot));
    return ds_attn_cache(swa ? inp->mctx->get_swa() : inp->mctx->get_base(), swa ? inp->get_k_idxs_swa() : inp->get_k_idxs(),
                         swa ? inp->get_v_idxs_swa() : inp->get_v_idxs(), swa ? inp->get_kq_mask_swa() : inp->get_kq_mask(), q, k, v, sinks, kq_scale, il);
}

std::vector<ggml_tensor *> llm_graph_context::ds_attn_cache(const llama_kv_cache_context * mctx_cur, ggml_tensor * k_idxs, ggml_tensor * v_idxs, ggml_tensor * kq_mask,
                                                             const std::vector<ggml_tensor *> & q, const std::vector<ggml_tensor *> & k, const std::vector<ggml_tensor *> & v,
                                                             const std::vector<ggml_tensor *> & sinks, float kq_scale, int il) const {
    GGML_ASSERT(cparams.flash_attn && !cparams.training);
    // 1. every range's K/V into the cache (these writes run where the cache is, behind the range's projections)
    ds_skip();
    for (size_t r = 0; r < ds_r.size(); ++r) {
        const int64_t nr = ds_r[r].t1 - ds_r[r].t0;
        (void) nr;
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k[r], ds_rows(k_idxs, r), il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v[r], ds_rows(v_idxs, r), il));
    }
    ds_skip();
    // 2. every range's queries over the whole cache, with its rows of the mask (the cache view is read after both writes: they were issued first)
    std::vector<ggml_tensor *> out(ds_r.size());
    std::vector<ggml_backend_dev_t> rdev;
    for (const auto & rr : ds_r) { rdev.push_back(rr.dev); }
    for (size_t r : llama_dsplit_attn_order_fork(rdev, mctx_cur->get_k(ctx0, il))) {   // the range on the cache's own card last (see there)
        const int64_t nr = ds_r[r].t1 - ds_r[r].t0;
        ds_skip();
        ggml_tensor * kc = mctx_cur->get_k(ctx0, il);
        ggml_tensor * vc = mctx_cur->get_v(ctx0, il);
        (void) nr;
        ggml_tensor * m  = ds_rows(kq_mask, r);   // its rows of the mask, cut once for all layers
        out[r] = build_attn_mha(q[r], kc, vc, nullptr, m, sinks.empty() ? nullptr : sinks[r], nullptr, 0, kq_scale, il);
        ds_pin(ds_r[r].b);
    }
    return out;
}
