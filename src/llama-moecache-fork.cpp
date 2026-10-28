#include <csignal>
#include "llama-moecache-fork.h"
#include "llama-fork-hooks.h"
#include "llama-gpu-probe-fork.h"
#include "llama-moestate-fork.h"
#include "llama-dsplit-fork.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <cctype>
#include <cstdarg>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <set>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include "llama-poolalloc-fork.h"
#include "llama-moetopic-fork.h"
#include "llama-passplan-fork.h"
#include "llama-kvtier-fork.h"

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// learned predictors: this decode trains them / the current graph was built with the update nodes
// values that depend on the card set and its links are stored under the place prefix (place.g<gpus>.<id>.v3, set by the tool), like the link rates
static std::string hw_key(const std::string & key) {
    const std::string & pl = moe_state_place();
    return pl.empty() ? key : pl + "." + key;
}

// a card's own name in a state key: its PCI address (0000_04_00_0), which stays with the card whatever order the devices are listed in and tells two cards
// of one model apart; the device name (CUDA0) when the backend has no address, the link index when there is no device. Values measured per card (the VRAM a
// start needs outside the pool, the link bandwidth) are stored under it: under the model name alone, two RTX 3090 overwrote each other (2026-10-28)
static std::string dev_pci_id(ggml_backend_dev_t dev, int k) {
    std::string id = "link" + std::to_string(k);
    if (dev) {
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        id = props.device_id ? props.device_id : ggml_backend_dev_name(dev);
    }
    for (char & c : id) { if (!isalnum((unsigned char) c)) { c = '_'; } }
    return id;
}

static bool g_pred_train_now = false, g_pred_train_built = false;
static uint64_t g_pred_epoch = 0, g_pred_epoch_built = 0;   // bumped when a layer's predictor is switched on or off
// 2-GPU prefill: the second GPU's share of the experts in big batches (0: one GPU); set by the prefill tuner, baked into
// the graph, so a change bumps the epoch and the next graph is rebuilt
static float    g_split_share = 0.0f; // alpha: 1 = all GPUs by link bandwidth, 0 = fastest GPU only
static uint64_t g_split_epoch = 0, g_split_epoch_built = 0;
static bool     g_split_enabled = false; // the compute buffers were reserved for a split (decided before the first reserve)
static double   g_split_speedup = 0.0;   // the transfer speedup that decision was judged by (the per-prompt decision line)
static std::vector<std::pair<ggml_backend_dev_t, double>> g_early_gbs; // early link probe, until the cache's own probe

namespace {

// --moe log=c (LLAMA_MOE_CACHE_TRACE=<prefix> moves the files): TRACE=<steps> (ctl file or env) records that many decode steps, after
// TRACE_AFTER steps, as a Chrome/Perfetto trace <prefix>.<n>.json (ui.perfetto.dev): the token steps, GPU split
// starts, CPU expert phases, predictions, every upload per worker, publishes (and how late) -- gaps show up there
struct trace_ev { int64_t ts, dur; int tid; std::string name, args; };
struct tracer {
    std::string            prefix;
    std::atomic<int64_t>   skip{0};  // steps before recording starts
    std::atomic<int64_t>   left{0};  // steps still to record
    std::mutex             mtx;
    std::vector<trace_ev>  ev;
    std::map<int, std::string> names;
    int                    n_files = 0;
    int64_t                t_step  = 0;
};
tracer g_tr;
enum { TR_SCHED = 1, TR_CPU = 2, TR_PRED = 3, TR_DDR = 4, TR_WORKER = 10 };

inline bool tr_on() { return g_tr.left > 0 && g_tr.skip == 0; }

void tr(int tid, std::string name, int64_t ts, int64_t dur = -1, std::string args = "") {
    std::lock_guard<std::mutex> lk(g_tr.mtx);
    g_tr.ev.push_back({ ts, dur, tid, std::move(name), std::move(args) });
}

void tr_dump() {
    std::vector<trace_ev> ev;
    {
        std::lock_guard<std::mutex> lk(g_tr.mtx);
        ev.swap(g_tr.ev);
    }
    if (ev.empty()) {
        return;
    }
    const std::string path = g_tr.prefix + "." + std::to_string(g_tr.n_files++) + ".json";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        return;
    }
    const int64_t t0 = ev.front().ts;
    fprintf(f, "{\"traceEvents\":[\n");
    for (const auto & n : g_tr.names) {
        fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s\"}},\n", n.first, n.second.c_str());
    }
    for (size_t i = 0; i < ev.size(); ++i) {
        const auto & e = ev[i];
        fprintf(f, "{\"name\":\"%s\",\"pid\":1,\"tid\":%d,\"ts\":%" PRId64 ",", e.name.c_str(), e.tid, e.ts - t0);
        if (e.dur >= 0) {
            fprintf(f, "\"ph\":\"X\",\"dur\":%" PRId64, e.dur);
        } else if (e.dur == -2) {
            fprintf(f, "\"ph\":\"C\"");
        } else {
            fprintf(f, "\"ph\":\"i\",\"s\":\"t\"");
        }
        fprintf(f, ",\"args\":{%s}}%s\n", e.args.c_str(), i + 1 < ev.size() ? "," : "");
    }
    fprintf(f, "]}\n");
    fclose(f);
    LLAMA_LOG_WARN("moe-cache: trace of %zu events written to %s\n", ev.size(), path.c_str());
}

std::string tr_fmt(const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

constexpr int MAX_LINKS = 16; // GPUs with a cache (their host->GPU links are measured, never assumed)

struct layer_state {
    llama_moe_cache_layer_fork pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    std::vector<uint32_t> expert_count;   // expert id -> uses, halved every LLAMA_MOE_CACHE_HALVE_EVERY steps
    std::vector<uint32_t> win_count;      // expert id -> uses in the last 64 tokens
    std::vector<uint64_t> expert_last;    // expert id -> clock of its last use (policy lru)
    std::vector<float>    relevance;      // expert id -> router probability for the current token, smoothed: rel = decay * rel + p (policy r)
    std::vector<uint8_t>  ghost_flag;     // expert id -> evicted within the last n_cache evictions of this layer (regret counter)
    std::deque<int32_t>   ghost_q;
    std::deque<std::vector<int32_t>> recent; // ids of those tokens, oldest first
    // where up/gate/down live on disk (fd < 0: unknown, upload from host memory)
    int    src_fd[3]   = { -1, -1, -1 };
    size_t src_offs[3] = { 0, 0, 0 };

    std::vector<uint64_t> glob_count;     // expert id -> lifetime uses
    std::vector<uint64_t> glob_base;      // the counts a profile preloaded at start (research emb=1 records this run's counts: glob_count - glob_base)
    std::vector<uint32_t> topic_cnt;      // expert id -> uses in the request being served (prompt and answer): its own term of the eviction score (hot maps per topic, --moe dynhot)
    std::vector<uint32_t> topic_dec;      // expert id -> uses in its answer (the decode steps the hit rate counts): its record keeps the hottest of these (topic_hottest)
    std::vector<uint32_t> topic_think;    // the same while it thinks (between <think> and </think>): the record's second map
    uint32_t              topic_max = 0;  // the largest topic_cnt of this layer (the score's scale: g_topic_tmax, the largest of all layers)
    std::vector<float>    topic_heat;     // expert id -> heat in the map merged from the stored requests nearest to this prompt (0..1 on one scale for all layers; 0: none)
    uint64_t              glob_max = 1;
    std::vector<bool>     sticky;         // expert id -> never evicted (most-used by lifetime count)
    std::vector<bool>     hot;            // expert id -> always hot: the fewest experts covering LLAMA_MOE_CACHE_PIN_HOT of the layer's picks
    bool                  slow = false;   // this layer's cache sits on a GPU with a slower upload link than the fastest one
    int                   link = 0;       // upload link (GPU) of this layer's cache: 0 = the offload / fastest GPU, 1.. the others
    int32_t               n_cache = 0;    // cache slots (the rest stream predicted experts); block pool: this layer's nominal share, used only by heuristics (hot set, adopt limit)
    int32_t               pcls    = -1;   // block pool: the size class this layer's experts are cached in (index into moe_cache::pool), -1 = own tensors (--moe pool=0)
    int32_t               n_own   = 0;    // block pool: blocks of the class this layer owns now
    std::vector<uint8_t>  is_stream;      // slot -> holds streamed (predicted) experts, not the cache
    std::unique_ptr<std::atomic<const char *>[]> pin_ptr[3];   // tensor (up, gate, down) -> expert -> its pinned copy (--moe pinr), nullptr = not pinned
    std::vector<int32_t>  stream_slots;   // those slots, round robin
    std::vector<float>    stream_score;   // slot -> predictor score of the streamed expert (SLOTKEEP)
    std::vector<uint64_t> stream_step;    // slot -> step it was streamed for
    int32_t               stream_next = 0;
    std::vector<int32_t>  stream_hit;     // experts used from a stream slot this step (promotion candidates)

    // prefill preheat: experts adopted from a GPU-offloaded prompt batch (copied device-to-device
    // from the scheduler's expert copy), published at the next step once up/gate/down all arrived
    std::vector<int32_t>  adopt_slot;     // expert id -> slot being filled, -1 none
    std::vector<uint8_t>  adopt_mask;     // expert id -> bit per tensor copied (1 up, 2 gate, 4 down)
    std::vector<int32_t>  adopted;        // expert ids with adopt_slot set, this batch
    std::vector<int32_t>  adopt_victims;  // experts evicted for them (table entry reset at the step)

    // drop-cached (LLAMA_MOE_CACHE_DROP): RAM pages are dropped only for experts that stayed in VRAM
    std::vector<uint64_t> cached_since;   // expert id -> step it was published (0: not cached)
    std::vector<bool>     dropped;        // expert id -> its RAM pages were dropped

    // router prediction: expert id -> upload queued or running (no second copy), published by the predictor
    std::vector<uint8_t>  queued;
    std::vector<uint8_t>  prefetched;

    // swap pay-back accounting, per expert: when its current copy finished uploading (0: not uploaded,
    // e.g. adopted from prefill), what that upload cost, and the decode hits it served since
    std::vector<int64_t>  up_t;
    std::vector<float>    up_cost;
    std::vector<uint32_t> up_hits;

    // JIT pool on the fastest-link GPU (pub.jit_n slots): pool slot -> expert, expert -> pool slot; the host table marks
    // pool experts n_slots + 1 (the CPU op skips them, the main device table maps them to "uncached")
    std::vector<int32_t>  pool_expert;
    std::vector<int32_t>  pool_slot;
    bool                  pool_dirty = false;

    uint64_t n_hit  = 0;
    uint64_t pred_seen_hit = 0, pred_seen_miss = 0; // n_hit / n_miss at the last predictor on/off decision
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
    bool    urgent = false; // router prediction: published between layers, not at the next step
    float   read_us = 0;    // disk / page-cache read into staging (0 when copied straight from host memory)
    float   copy_us = 0;    // host -> device copy over the layer's PCIe link
    int64_t t_done  = 0;    // ggml_time_us() when the upload finished
    bool    stream  = false; // predicted upload into a stream slot, for layer_idx in this step
    bool    skipped = false; // dropped by the worker: its layer had started (slot freed at publish)
    uint64_t step   = 0;     // step the prediction was made in
    int64_t t_queued = 0;    // ggml_time_us() when queued (trace)
    bool    bg      = false; // refill of a free block after a prompt (reheat_free_blocks): the step does not wait for it, a worker takes it only when no other
                             // job waits on its link, and it does not count against the swap budget
};

struct moe_cache {
    std::string settings;   // key settings that are no knobs and not tuned (upload path, bounce, workers, slots, policy), the first part of the run.knobs line of the state snapshot
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;
    int32_t window      = 64; // recent-usage window in tokens (--moe-cache-window)

    uint64_t clock   = 0;
    uint64_t n_steps = 0;
    bool     boot_done = false;    // a prompt batch was seen: the admit boot is over
    uint64_t boot_from = 0;        // the admit boot window counts from here (the start of the process: a cold cache; a prompt batch fills the cache by adoption and does not restart it, GLM pf12k decode -5%)

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    // block pool: per size class, which layer owns each block (-1 free) and the free blocks. No block is reserved for a layer: a layer takes a
    // free block, or the block of the lowest-ranked cached expert of ANY layer of the class (see pick_slot)
    struct pool_state {
        std::vector<int32_t> owner;   // block -> index into layers, -1 free
        std::vector<int32_t> free;    // free blocks (stack)
        ggml_backend_dev_t   dev = nullptr;   // the card of the class's buffer (pool_fill_log)
        size_t               B   = 0;         // bytes of one block
        size_t               off[3] = { 0, 0, 0 }, len[3] = { 0, 0, 0 };   // up / gate / down slice: offset in a block and bytes
        size_t               past[3] = { 0, 0, 0 };   // bytes a quantized matmul reads past the slice's last row (rows not a multiple of 512 values), see pool_layout
        int32_t              active_end = -1; // kv-tail: the blocks from here on are all under the KV rows (owner -3): scans stop here (-1: the whole class)
        int32_t              n_active() const { return active_end < 0 ? (int32_t) owner.size() : active_end; }
    };
    std::vector<pool_state> pool;
    int64_t  n_expert_used = 0;   // experts per token (the compute alias sizes the CUDA scratch from it)
    uint64_t gmax_all = 1;
    // compute alias (--moe alias=1): per pool buffer, the first size class can lend its first blocks to the scheduler's compute buffer.
    // Blocks [0, blocked) of that class are compute memory now (owner -2): no expert, not free; `want` is what the last graph needed.
    struct alias_pool {
        ggml_backend_buffer_t buf  = nullptr;
        ggml_backend_dev_t    dev  = nullptr;
        int32_t               pcls = -1;
        size_t                B    = 0;
        int32_t               n    = 0;
        int32_t               blocked = 0;
        int32_t               want    = 0;
        int                   cuda_dev = -1;    // CUDA device index (the backend's scratch region lives in the lent blocks), -1: other backend
        size_t                hwm       = 0;    // the largest scratch a graph asked for so far (reported with log=a)
        int32_t               last_n    = 0;    // the previous graph: its tokens and the scratch it was lent (checked against what it asked for)
        size_t                last_scr  = 0;
        size_t                refused_room = 0;   // what the pool could lend beside the KV when refused_b was refused
        size_t                refused_b = 0;    // the least graph lend (bytes) refused as not fitting the pool or reaching the KV zone: a same or bigger lend is refused again until the zone moves (the blocks are already lent, so the lend check alone would pass it)
        size_t                slot_b    = 0;    // staging slot size of this card (the largest staged expert range), 0 until a prompt graph was seen
        int                   n_stage   = 0;    // staging slots the last graph got (0: it ran without prefetch)
        int                   n_cfg     = 0;    // staging slots a graph may use at most (the scheduler's ring size, --moe pf-slots)
        size_t                graph_tok = 0;    // graph bytes per batch token of the biggest prompt graphs seen (the prompt pass planner)
        uint32_t              graph_tok_n = 0;  // tokens of the graph graph_tok was taken from: only a graph at least as big replaces it
        size_t                scr_tok   = 0;    // the computed CUDA scratch per batch token of that same graph (the prompt pass planner)
        // Lending from every size class (--moe alias-all, default on): the classes lie back to back in the buffer (the first one at 0, nothing between them),
        // so the lent range [0, bytes) runs past the first class into the next ones, and each class gives the blocks the range covers. One class only (the
        // old rule) capped the micro-batch at what the biggest class holds (MiMo-V2.6-Flash: 4 GB of a 15 GB pool, Qwen3.8-Flash-Next: 4.8 GB of 19 GB).
        struct seg { int32_t pcls = -1; size_t base = 0; size_t B = 0; int32_t n = 0; int32_t blocked = 0; int32_t hwm = 0; };   // hwm: the most blocks the lend took in this class
        std::vector<seg>      segs;
        size_t                span   = 0;       // lendable bytes: up to the last class's last expert block
        size_t                want_b = 0;       // bytes the last graph needed (what alias_release keeps)
        size_t                mirror_b = 0;     // the dense split's copies at the start of the buffer (--moe mirror-pool, see llama_dsplit_mirror_place_fork)
        size_t                base_off = 0;     // where the graph being built starts: mirror_b when it splits (the copies before it), else 0
        // KV zone (--moe kv-tail=1, fork/docs/kv-grow.md): the records of the caches on this card interleaved in one combined row per cell (kv_row bytes, record r at
        // byte `prefix` of it), [kv_start, kv_start + kv_h * kv_row) of the buffer, behind the compute lend and in front of the experts; blocks under it are owner -3
        struct kvt_run { void * owner = nullptr; size_t bytes = 0, row = 0, pad = 1, prefix = 0, cells_max = 0, ratio = 1; bool fixed = false; std::function<size_t()> rows; std::function<void(ggml_backend_buffer_t, uint8_t *, size_t)> attach; size_t cells = 0, plan = 0; float bpv = 2.0f;   // plan: the cells the prompt being planned will fill; bpv: bytes per value of the record's types
                         std::function<size_t(ggml_type, ggml_type, const std::vector<char> *, std::vector<llama_kv_tail_fork::seg> *)> layout;   // qkv-ram-fallback in place (llama_kv_tail_fork)
                         std::function<void(ggml_type, ggml_type, const std::vector<char> *, const std::vector<std::pair<uint8_t *, size_t>> *, ggml_backend_buffer_t)> retype;
                         std::vector<char> out, out_next; std::vector<std::pair<uint8_t *, size_t>> host;   // RAM part: layers in host memory now / after the next relayout; where
                         ggml_type tk = GGML_TYPE_F16, tv = GGML_TYPE_F16, tk0 = GGML_TYPE_F16, tv0 = GGML_TYPE_F16; bool convertible = false; size_t unit = 256; };   // tk0/tv0: the types it was created with (tier 0); unit: its row and its place in the combined row are multiples of it
        std::vector<kvt_run>  kvt;
        size_t                tail_end  = 0;    // end of the last class's expert blocks (the zone never reaches the last two blocks before it)
        size_t                span_full = 0;    // the lend's reach without a KV zone
        size_t                kv_row   = 0;     // bytes of one combined cell (the sum of the growing records' rows)
        size_t                kv_fixed = 0;     // bytes of the fixed records (fewer cells than the largest record, e.g. MiMo's sliding-window cache: laid out at full size, first in the zone)
        size_t                kv_start = 0;     // where the zone starts (the lend's reach): moves only when a graph needs more lend than ever before
        size_t                kv_h     = 0;     // cells the zone backs (the horizon)
        size_t                kv_step  = 10240; // horizon step (from the layer pattern, llama_kv_cache)
        size_t                kv_zhi   = 0, kv_last_cells = 0;   // the zeroed pad window ends here; the cells at the last zeroing
        std::map<int32_t, size_t> need_tok;     // the most a graph of n tokens borrowed from the start of the buffer (graph + scratch + staging slots)
        std::map<int32_t, size_t> need_cells;   // the cells in use when that most was measured (the graph's inputs grow with them: need_at_cells)
        size_t                kv_blocks = 0;    // blocks under the KV zone now
        size_t                dec_need  = 0;    // the most a decode graph (< 64 tokens) borrowed from the start of the buffer: the KV never takes that
        size_t                dec_cells = 0;    // the cells in use when dec_need was measured
        // quantized K/V (qkv-ram-fallback): the attention converts a layer's K and V to f16 inside the graph's lend (fattn-common.cuh, f16_extra), so a graph's lend grows
        // by these bytes per cell (the largest layer's K + V elements x 2). Prompt graphs always take that path; a decode graph only when its head size has no vector kernel
        // for quantized K/V (192, 576 ...). 0 for float K/V.
        size_t                kv_lend_pc = 0, kv_lend_pc_dec = 0;
        // RAM part of the KV ladder: one host buffer per card, allocated once at the first move (never freed); host_cells: cells every host layer array holds
        ggml_backend_buffer_t hbuf = nullptr;
        size_t                hsize = 0, hused = 0, host_cells = 0;
    };
    std::atomic<bool> kv_gave_back{false};   // kv-tail: blocks came back from the KV (a new, shorter request): the next step refills them with the hottest missing experts
    std::atomic<bool> starved{false};   // kv-tail: the KV holds every expert block of the cards that have a record: no expert can be cached, the per-step cache logic is off
    std::vector<alias_pool> alias;   // block pool: the lifetime count of the busiest expert of ANY layer, the common scale of scores compared across layers
    bool pf_count_on      = false;   // prompt hit counters on the cards (see pf-count)
    int  pf_count_layers  = 0;
    int  pf_count_experts = 0;
    std::map<const ggml_tensor *, size_t> by_up_src;
    std::map<const ggml_tensor *, std::pair<size_t, int>> by_src; // host weight -> (layer, 0 up / 1 gate / 2 down)

    // big-batch prefill: experts served device-to-device from the cache vs over PCIe
    uint64_t n_src_hit   = 0;
    uint64_t n_src_query = 0;
    uint64_t n_adopted   = 0; // prefill preheat: experts kept from prompt batches
    uint64_t n_evictions = 0; // experts evicted from the cache (all kinds)
    // cache pressure counters (per layer-step unless said otherwise), exposed as llama_tuning_info::moe_pr:
    //  0 rel_seen    layer-steps observed with router probabilities (policy r; 0 after a few steps = they do not arrive and the default score is used)
    //  3 pending     misses that reached the swap decision          4 free    admitted into an empty slot            5 evict   admitted by evicting a victim
    //  6 d_admit     declined: fewer than ADMIT uses in the window  7 d_margin declined: not better than the victim by the pay-back margin
    //  8 d_budget    declined: swap budget of the step used up      9 d_victim declined: every slot in flight     10 d_big declined: the victim is a lifetime regular
    // 11 d_dup       already cached, queued or score 0              12 regret  a miss of an expert evicted within the last cache turnover (an eviction that was wrong)
    // 13 ev_unpaid   evicted uploads that never paid back their copy cost   14 ev_zero  evicted uploads that were never hit
    uint64_t pr[16] = {};

    // --moe pinr=N (default 6): a model that does not fit RAM keeps its weights mmap'd, and pinned copies of the slices of the experts that get uploaded are kept for as
    // long as N/10 of the RAM lasts: the next upload of an expert reads its pinned copy (direct DMA, 2.2x the pageable speed on the x16 link), the cold ones stay in the page cache
    ggml_backend_buffer_type_t pin_buft = nullptr;
    ggml_backend_buffer_type_t lazy_buft = nullptr;   // --moe pin-lazy: the GPU's pinned host buffer type
    size_t                     pin_budget = 0, pin_used = 0, pin_left = 0;
    char *                     pin_cur = nullptr;
    uint64_t                   pin_n = 0;
    std::vector<ggml_backend_buffer_t> pin_chunks;
    std::mutex                 pin_mtx;
    std::thread                lazy_thr;          // --moe pin-lazy: reads expert tensors into pinned buffers in the background (pin_prefetch_run)
    std::atomic<bool>          lazy_stop{false};

    // model bigger than RAM, weights mmap'd: an expert in VRAM doesn't need its RAM copy. Its pages
    // are dropped when it is cached and read back (async readahead) when it is evicted, so RAM holds
    // the experts the CPU computes instead of the kernel's LRU choice (LLAMA_MOE_CACHE_DROP=1/0)
    bool drop_cached = false;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::vector<std::thread> workers;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<int>         worker_link; // worker -> upload link it serves (layer_state::link), -1: all
    int                      n_links = 1; // GPUs holding cache layers
    ggml_backend_dev_t       link_dev[MAX_LINKS] = {}; // the GPU behind each link
    ggml_backend_t           link_be[MAX_LINKS]  = {}; // its compute backend (seen at the split callback): JIT uploads go on its stream
    uint64_t                 n_jit = 0, n_jit_layers = 0, n_jit_miss = 0, n_pool_up = 0, n_pool_hit = 0; // JIT: experts uploaded, layer calls that uploaded, misses seen
    // layer clock (resource saturator deadlines): EMA of the time from layer i's CPU expert op to layer i+1's
    std::vector<double>      layer_dt;
    int64_t                  clk_t  = 0;
    int64_t                  clk_li = -1;
    uint64_t                 clk_step = 0;
    std::vector<upload_job>  done;
    bool                     stop = false;
    int                      in_flight = 0; // jobs popped by a worker, not yet in done
    int                      in_flight_stream = 0; // of those, stream and background jobs (the step doesn't wait for them)
    int32_t                  graph_tokens_last = 0; // tokens of the last graph that asked for lent blocks (llama_moe_cache_alias_fork): a decode graph or a prompt pass
    uint64_t                 n_reheat = 0;          // experts queued by reheat_free_blocks, all prompts
    int64_t                  fill_from = -1;        // decode step at which the last prompt was over (pool_fill_log follows the refill), -1: not watching
    int64_t                  fill_t0   = 0;
    std::condition_variable  dcv;           // signalled when a job lands in done

    // swap cost accounting (guarded by wmtx for the upload side)
    double   upload_us    = 0;  // EMA of one expert upload (up+gate+down)
    uint64_t n_uploads    = 0;
    double   link_us[MAX_LINKS] = {}; // same, per upload link: [0] fastest GPU, [1] slower ones (layer_state::slow)
    uint64_t n_link[MAX_LINKS] = {};
    int64_t  last_step_us = 0;
    double   step_us_busy = 0;  // EMA of token time while uploads were in flight
    double   step_us_idle = 0;  // EMA of token time with no uploads in flight
    size_t   rr           = 0;  // round-robin start layer for the swap budget
    int      last_budget  = 0;
    int      last_margin  = 0;
    int      link_margin[MAX_LINKS] = {}; // pay-back margin per upload link
    double   read_link[MAX_LINKS] = {};     // EMA of the read part of one upload, per link
    double   copy_link[MAX_LINKS] = {};     // EMA of the PCIe copy part, per link
    // DDR4 phases: the CPU computing uncached experts (ggml moe phase callback)
    // stream self-tuning per link (AUTO): extra lead in layers, candidates per layer, off; window counters
    int                  lead_extra[MAX_LINKS] = {};
    int                  auto_m[MAX_LINKS] = {};   // 0: STREAM_M
    bool                 stream_off[MAX_LINKS] = {};
    std::atomic<uint64_t> st_up[MAX_LINKS] {};
    std::atomic<uint64_t> st_late[MAX_LINKS] {};
    std::atomic<uint64_t> st_hit[MAX_LINKS] {};
    std::atomic<bool>    cpu_busy{false};
    // DDR4 load meter: demand now (MB/s: the CPU phase at its measured rate + each running upload at its link rate)
    // and bytes read, CPU expert matmuls vs uploads
    std::atomic<int64_t> ddr_mbs{0};
    int64_t              cpu_mbs_on = 0;     // what the running CPU phase added
    double               cpu_gbs_meas = 0;   // EMA of the CPU phases' read rate
    std::atomic<int64_t> cpu_bytes_phase{0};
    std::atomic<int64_t> cpu_bytes{0}, up_bytes{0};
    std::atomic<int64_t> pf_hit_bytes{0};   // prompt passes: bytes of experts found in the cache and copied from it on the card (the upload skipped), the counterpart of pf_bytes
    std::atomic<int64_t> pf_ask[MAX_LINKS] = {}, pf_found[MAX_LINKS] = {}, pf_wrong[MAX_LINKS] = {};   // staging lookups per card (link): experts asked, found on that card, asked of a card whose cache is not this one
    std::atomic<int64_t> pf_bytes{0}, pf_link[MAX_LINKS] = {};   // prompt passes: bytes of experts not found in the cache, which the pass streams from RAM over the link (live view in GET /moe/stats only, not in the decode churn columns)
    std::atomic<int64_t> link_bytes[MAX_LINKS] = {};   // uploaded bytes per link, for GET /moe/stats
    int64_t              seg_cpu_bytes = 0, seg_up_bytes = 0;
    std::atomic<int64_t> busy_t0{0};
    std::atomic<int64_t> busy_us{0};      // total busy time
    std::atomic<int64_t> busy_n{0};       // busy phases
    uint64_t up_overlap = 0;              // uploads that ran while the CPU was busy (at start or end)
    std::atomic<int64_t> gate_wait_us{0}; // time workers waited for the CPU phase to end (GATE=1)
    int64_t  seg_gate_us = 0;
    int64_t  seg_busy_us = 0, seg_busy_n = 0; uint64_t seg_overlap = 0, seg_up2 = 0, seg_steps2 = 0; int64_t seg_t0 = 0;
    double   gbs_all      = 0;            // EMA of upload throughput, GB/s (expert sizes differ per layer, times don't compare)
    double   gbs_link[MAX_LINKS] = {};     // same, per link
    // evicted uploads, per link: count, decode hits served, uploads whose hits*cpu_eval_us covered their cost, lifetime
    uint64_t ev_n[MAX_LINKS] = {}, ev_hits[MAX_LINKS] = {}, ev_paid[MAX_LINKS] = {}; // paid: hits x CPU eval time (layer bytes / CPU_GBS) >= upload time
    double   ev_life_us[MAX_LINKS] = {}, ev_cost_us[MAX_LINKS] = {};

    // LLAMA_MOE_CACHE_CTL: settings file re-read on change; stats are reported per segment between changes
    std::string                     ctl;
    std::filesystem::file_time_type ctl_mtime{};
    uint64_t seg_steps = 0, seg_hit = 0, seg_miss = 0, seg_up = 0;

    // pinned staging per upload worker, for file-backed uploads (pread -> pinned -> DMA)
    std::vector<ggml_backend_buffer_t> staging;
    size_t bounce_bytes = 0;   // --moe bounce=MB: uploads from mmap'd memory are copied to the pinned staging buffer in chunks of this size first, then DMA'd from there (0: DMA straight from the mmap pages)
    size_t staging_size = 0;

    // hot-set profile: per-expert lifetime uses saved at shutdown, preloaded at the next start
    std::string profile;

    // router prediction (LLAMA_MOE_CACHE_PRED=M): at layer L's CPU expert op a helper thread applies
    // layer L+1's router to layer L's input (top-8 overlap 0.61 on GLM-5.3-Flash) and uploads up to
    // LLAMA_MOE_CACHE_PREDICT_MAX of its top-M experts that aren't cached. They are published before
    // layer L+1's GPU split if the upload finished by then, else at the next step
    int32_t pred_m   = 0;
    int32_t pred_max = 2;
    float   pred_margin = 0.1f; // min score lead over the M-th prediction (LLAMA_MOE_CACHE_PREDICT_MARGIN)
    bool    pred_ra     = false; // LLAMA_MOE_CACHE_PREDICT_RA=1: readahead of predicted uncached experts (mmap'd weights)
    std::vector<std::vector<float>> pred_b;  // layer idx -> its selection bias (empty: raw logits)
    struct pred_req {
        size_t             li;
        uint64_t           step;
        int64_t            n_tok;
        int64_t            n_blk; // predicted layers li+1 .. li+n_blk
        std::vector<float> x;     // [n_tok][n_blk][n_expert] logits
    };
    std::deque<pred_req>    preq;
    std::mutex              pmtx;
    std::condition_variable pcv;
    std::thread             pred_thr;
    bool                    pred_stop   = false;

    // L3 prefetch (L3PF=M): after layer L's CPU phase, one thread per CCX reads that CCX's half of the rows of the
    // top-M predicted uncached experts of layer L+1 into its L3, while the GPU runs L+1's attention (DDR idle)
    struct l3pf_job { size_t li = 0; std::vector<int32_t> ex; uint64_t gen = 0; };
    std::mutex              l3_mtx;
    std::condition_variable l3_cv;
    l3pf_job                l3_job;
    bool                    l3_stop = false;
    std::vector<std::thread> l3_thr;
    std::atomic<uint64_t>   l3_n_ex{0}, l3_done{0}, l3_late{0}, l3_dropped{0};
    // live A/B slices (AUTO_L3): state 0 = off, 1 = on; token time sums per state; decided value
    int      ab_state = 0, ab_slices = 0, ab_decided = -1;
    int      cpu_sat_threads = 0; // startup probe: reader threads that reach 95% of the CPU's RAM read rate
    // SELF_TUNE=1 (self_tune): the knob under test, its candidate in use, the running slice, per-candidate slice means
    int      lv_k = -1, lv_c = 0;
    int      lv_inc = 0;           // the incumbent candidate of the knob under test (the value in use when its slice cycle began)
    int64_t  lv_slice_len = 36, lv_warm = 4; // slice length and skipped tokens of the knob under test (state-carrying knobs need long ones)
    std::string tuned_text;        // the self-tuner's decisions (NAME=value lines), kept in the state file: the next start begins from them
    uint64_t lv_cycle = 0;         // cycles finished: the candidate order rotates so no candidate always runs first (cold cache)
    int64_t  lv_slice_n = 0;
    double   lv_slice_sum = 0;
    uint64_t lv_rest_until = 0;    // the first cycle starts at once (a saved tuner state sets a rest instead); the incumbent runs first, so exploring costs little
    uint64_t lv_rest = 0;          // tokens between cycles: doubles while cycles confirm their decisions
    uint64_t hr_h = 0, hr_m = 0;   // hits and misses at the last hit-rate window boundary (regime-change trigger)
    double   hr_ref = -1;          // the window hit rate (%) when the last cycle ended: the level its decisions were made at
    uint64_t hr_at = 0;            // step of the last window boundary
    bool     lv_changed = false;   // a decision in this cycle differs from the previous cycle's
    std::vector<double> lv_prev;   // previous cycle's decisions per tunable
    std::vector<std::vector<double>> lv_samples;
    bool     last_prefill = false; // the last observed batch was a prompt batch (not a decode token)
    int64_t  last_tokens  = 0;     // its token count
    bool     prev_prefill = false; // the batch before it was one too (its step time is then pure batch time, no idle gap)
    // prefill split tuner: candidate shares, per-candidate ubatch throughputs (t/s), the candidate in use, rest period
    std::vector<float>               ps_vals;
    std::vector<std::vector<double>> ps_samples;
    int      ps_c = 0;
    bool     ps_done = false;
    int64_t  ps_full = 0;          // largest prompt batch seen: only full batches are compared
    int      ps_rest = 0, ps_rest_left = 0;
    double   ab_sum[2] = { 0, 0 };
    uint64_t ab_n[2] = { 0, 0 }, ab_since = 0;
    double   l3pf_set = 0; // the configured L3PF the slices test against 0
    int64_t                 gpu_started = -1; // layer idx whose GPU split may have started this step (mtx)
    int64_t                 last_obs    = -1; // layer idx of the last CPU expert op this step (mtx)
    uint64_t n_pred_up = 0, n_pred_pub = 0, n_pred_used = 0, n_pred_late = 0, n_stream_stale = 0, n_promoted = 0, n_tbp = 0;

    // learned predictors (in-graph NLMS, see llama_moe_cache_layer_fork::pred_m): trained every pred_train
    // tokens (--lrn-prd N, 0 = frozen) with step size pred_mu; weights persist in pred_file
    int32_t  pred_ahead = 0;
    int32_t  pred_train = 0;
    float    pred_mu    = 0.5f;
    float    cur_mu     = 0.0f;
    uint64_t n_train    = 0;
    std::string pred_file;
    std::vector<std::vector<float>> last_pred; // layer idx -> last token-0 prediction [n_blk][n_expert]
    std::vector<uint64_t>           last_pred_step;
    std::vector<int64_t>            last_pred_blk;
    uint64_t acc_hit[8] = {}, acc_tot[8] = {}; // top-k overlap per lookahead, lifetime
    // prefetch candidates, L+1: the top-M predicted experts among those not cached, M = 1..3: how many the router
    // then picked (precision), and how many of the real misses they cover (recall)
    uint64_t pm_pick[3] = {}, pm_hit[3] = {}, pm_miss = 0, pm_cov[3] = {};
    uint64_t win_hit[8] = {}, win_tot[8] = {}; // same, since the last report
    std::vector<ggml_context *>        pctxs;
    std::vector<ggml_backend_buffer_t> pbufs;
};

// Where a tensor's raw GGUF bytes live (recorded by the model loader).
bool file_location(const llama_model & model, const ggml_tensor * t, int * fd, size_t * off) {
    auto it = model.exps_file_locs.find(t);
    if (it == model.exps_file_locs.end()) {
        return false;
    }
    *fd  = it->second.fd;
    *off = it->second.offs;
    return true;
}

static size_t mem_available_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    size_t kb = 0;
    char line[256];
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) { break; }
    }
    if (f) { fclose(f); }
    return kb * 1024;
}

// the pinned copy of expert e's slice of tensor k, or nullptr. Made on the SECOND upload of the expert (the first only marks it: an expert that is never evicted and uploaded again gains
// nothing from a copy, and every copy duplicates the page cache) while the pinr budget lasts
const char * pin_slice(moe_cache * mc, layer_state & ls, int k, int32_t e, size_t sz, const ggml_tensor * src) {
    if (!mc->pin_buft || !ls.pin_ptr[k]) {
        return nullptr;
    }
    const char * const seen = (const char *) 1;   // marker: uploaded once
    const char * p0 = ls.pin_ptr[k][e].load(std::memory_order_acquire);
    if (p0 > seen) {
        return p0;
    }
    if (p0 == nullptr) {
        ls.pin_ptr[k][e].store(seen, std::memory_order_release);
        return nullptr;
    }
    char * dst = nullptr;
    {
        std::lock_guard<std::mutex> lk(mc->pin_mtx);
        if (ls.pin_ptr[k][e].load(std::memory_order_acquire) > seen) {
            return ls.pin_ptr[k][e].load();
        }
        if (mc->pin_used + sz > mc->pin_budget) {
            return nullptr;
        }
        if (mc->pin_left < sz) {
            const size_t chunk = (size_t) 256 << 20;
            if (mem_available_bytes() < chunk + ((size_t) 6 << 30)) {   // never push the system into swapping
                LLAMA_LOG_WARN("moe-cache: pinr: stopped at %.1f GiB pinned (RAM available is low)\n", mc->pin_used / 1073741824.0);
                mc->pin_budget = mc->pin_used;
                return nullptr;
            }
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(mc->pin_buft, chunk);
            if (!b) {
                mc->pin_budget = mc->pin_used;
                return nullptr;
            }
            mc->pin_chunks.push_back(b);
            mc->pin_cur  = (char *) ggml_backend_buffer_get_base(b);
            mc->pin_left = chunk;
        }
        dst = mc->pin_cur;
        mc->pin_cur  += sz;
        mc->pin_left -= sz;
        mc->pin_used += sz;
        if (++mc->pin_n % 2048 == 0) {
            LLAMA_LOG_WARN("moe-cache: pinr: %.1f of %.1f GiB pinned (%" PRIu64 " expert slices)\n", mc->pin_used / 1073741824.0, mc->pin_budget / 1073741824.0, mc->pin_n);
        }
    }
    memcpy(dst, (const char *) src->data + (size_t) e*sz, sz);
    ls.pin_ptr[k][e].store(dst, std::memory_order_release);
    return dst;
}

bool pread_full(int fd, void * dst, size_t n, size_t off);

// --moe pin-lazy (default with a pin budget on a MoE model): the loader maps the weights and reads nothing, this thread then pins whole expert tensors while the
// server already answers: one pinned buffer per tensor (cudaHostAlloc costs ~3.5 GiB/s whatever the size, measured 2026-10-19), filled by 16 threads of 4 MiB preads
// (the SSD gives 6.5 GiB/s that way, the mapping's page faults 2.8), then the tensor's data pointer moves to the pinned copy: CPU compute and uploads read
// it from there (direct DMA), and the mapping's pages and the page cache of that range are dropped. Layer order, until the budget or the RAM margin ends it.
// The data pointer is swapped while the graph may read it: both copies hold the same bytes, and an op that started on the old address still finds it mapped.
static void pin_prefetch_run(moe_cache * mc, size_t budget) {
    size_t used = 0, total = 0;
    for (auto & ls : mc->layers) { for (int k = 0; k < 3; ++k) { total += ggml_nbytes((k == 0 ? ls.pub.up_src : k == 1 ? ls.pub.gate_src : ls.pub.down_src)); } }
    const int64_t t0 = ggml_time_us();
    size_t next_log = (size_t) 10 << 30;
    int n_pinned = 0;
    for (auto & ls : mc->layers) {
        for (int k = 0; k < 3; ++k) {
            if (mc->lazy_stop.load()) { return; }
            ggml_tensor * t = const_cast<ggml_tensor *>(k == 0 ? ls.pub.up_src : k == 1 ? ls.pub.gate_src : ls.pub.down_src);
            if (ls.src_fd[k] < 0 || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) { continue; }
            const size_t nb = ggml_nbytes(t);
            if (used + nb > budget) { continue; }   // a smaller tensor further on may still fit
            if (mem_available_bytes() < nb + ((size_t) 8 << 30)) {   // never push the system into swapping
                LLAMA_LOG_WARN("moe-cache: pin-lazy: stopped at %.1f GiB pinned (RAM available is low)\n", used / 1073741824.0);
                return;
            }
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(mc->lazy_buft, nb);
            if (!b) { LLAMA_LOG_WARN("moe-cache: pin-lazy: pinned allocation of %.0f MiB failed at %.1f GiB pinned\n", nb / 1048576.0, used / 1073741824.0); return; }
            char * base = (char *) ggml_backend_buffer_get_base(b);
            std::atomic<size_t> nxt{0};
            std::atomic<bool>   fail{false};
            const size_t req = (size_t) 4 << 20;
            auto work = [&]() {
                for (;;) {
                    const size_t o = nxt.fetch_add(req);
                    if (o >= nb || fail.load() || mc->lazy_stop.load()) { return; }
                    if (!pread_full(ls.src_fd[k], base + o, std::min(req, nb - o), ls.src_offs[k] + o)) { fail.store(true); return; }
                }
            };
            std::vector<std::thread> th;
            for (int i = 0; i < 16; ++i) { th.emplace_back(work); }
            for (auto & x : th) { x.join(); }
            if (fail.load() || mc->lazy_stop.load()) { ggml_backend_buffer_free(b); if (fail.load()) { continue; } return; }
            void * old = t->data;
            __atomic_store_n(&t->data, (void *) base, __ATOMIC_RELEASE);
            {   std::lock_guard<std::mutex> lk(mc->pin_mtx); mc->pin_chunks.push_back(b); }
#ifndef _WIN32
            const uintptr_t pg = 4096, a0 = ((uintptr_t) old + pg - 1) & ~(pg - 1), a1 = ((uintptr_t) old + nb) & ~(pg - 1);
            if (a1 > a0) { madvise((void *) a0, a1 - a0, MADV_DONTNEED); }   // the mapping's pages of this tensor
            posix_fadvise(ls.src_fd[k], (off_t) ls.src_offs[k], (off_t) nb, POSIX_FADV_DONTNEED);   // and the page cache behind them
#endif
            used += nb; ++n_pinned;
            if (used >= next_log) {
                LLAMA_LOG_WARN("moe-cache: pin-lazy: %.1f of %.1f GiB pinned (%d tensors, %.1f s, %.2f GiB/s)\n", used / 1073741824.0, std::min(budget, total) / 1073741824.0,
                        n_pinned, (ggml_time_us() - t0) / 1e6, used / 1073741824.0 / std::max(1e-3, (ggml_time_us() - t0) / 1e6));
                next_log += (size_t) 10 << 30;
            }
        }
    }
    LLAMA_LOG_WARN("moe-cache: pin-lazy: done, %.1f GiB of the %.1f GiB expert weights pinned (%d tensors) in %.1f s, the rest stays mapped\n",
            used / 1073741824.0, total / 1073741824.0, n_pinned, (ggml_time_us() - t0) / 1e6);
}

bool pread_full(int fd, void * dst, size_t n, size_t off) {
#ifdef _WIN32
    // no file locations are recorded on Windows; uploads use host memory
    GGML_UNUSED(fd); GGML_UNUSED(dst); GGML_UNUSED(n); GGML_UNUSED(off);
    return false;
#else
    uint8_t * d = (uint8_t *) dst;
    while (n > 0) {
        const ssize_t r = pread(fd, d, n, (off_t) off);
        if (r <= 0) {
            return false;
        }
        d += r; n -= (size_t) r; off += (size_t) r;
    }
    return true;
#endif
}

// tunables, from LLAMA_MOE_CACHE_<NAME> at start; with LLAMA_MOE_CACHE_CTL=<file> also re-read from that file
// (NAME=value lines, checked every 16 steps when it changes) so one server can A/B settings without a reload
struct knobs_t {
    double swap_frac = 0.25;  // upload time per step as a fraction of the token time (the swap budget); tuned live
    double hot_frac  = 0;     // tier: pin the always-hot set on slower links; off: never won in A/B
    double sticky    = 0.0; // off: +1.6% on GLM chat, within noise
    double slow_stay = 0;     // tier min stay (steps); off: never won in A/B
    double margin    = -1;  // fixed pay-back margin; -1: 0 (deterministic mode 4); -2: from upload timing (CPU GB/s / link GB/s)
    double budget    = -1;  // fixed swaps per step, -1: from upload timing
    double cpu_gbs   = 38;  // CPU read rate alone (ddrbw, 6 threads, DDR4-3200 ECC); pay-back margin, DDR demand before measured
    double link      = 1;   // pay-back margin per upload link (0: one margin from the average upload)
    double gate      = 0;   // uploads wait (up to GATE_MAX_US per tensor) while the CPU computes uncached experts: no DDR4 contention
    double gate_max_us = -1;  // max wait for DDR room per tensor copy; -1: the measured mean layer time
    double big         = 0;    // an expert may only evict one with at most its lifetime use (no weaklings evicting big boys)
    double l3pf        = 0;    // L3 prefetch: experts per layer (0 off); needs GGML_MOE_CCX_SPLIT=<domains> and placed threads
    double auto_l3     = 0;    // live A/B: alternate L3PF on/off every 64 tokens, keep the faster (the CPU kernel may be compute-bound)
    double tbp         = 0;    // token-boundary prefetch: per early layer, stream up to this many of the last token's misses
    double tbp_layers  = 6;    // ... for the first N MoE layers (their uploads run while the output head, sampling and the
                               //     dense layers keep DDR idle, ~7-10 ms on GLM)
    double auto_tune   = 1;    // 1: adjust stream lead and STREAM_M per link from the measured late share and precision (thresholds);
                               // 2: uploads per target layer = measured time until it / measured link time per expert
    double wait        = 1;    // the step waits for queued cache swaps (0: never; finished uploads are published at splits)
    double chunk_kb    = -1;    // GATE=3: copy in chunks of this size, re-checking the DDR budget before each (0: whole tensor)
    double margin_mb   = 0;     // VRAM kept free per GPU after the cache is sized; 0: 384 (the engine sets it from the measured run-time growth)
    double ddr_gbs     = 44;   // fork/bench/ddrbw: CPU + both DMAs together peak at 44-45 GB/s (DDR4-3200 ECC, 2 ch); GATE=3: an upload starts only while DDR demand + its link rate stays under this
    double stream    = 1e9; // predicted uploads use up to this many stream slots per layer (0: evict cache slots, old path)
    double stream_m  = 12;  // candidates per target layer in stream mode (over-predict; no confidence cut)
    double slotkeep  = 0;   // a prediction may only replace a stream slot holding a lower-scored one of this step
    double predict   = 0;   // the learned predictors (allocated with --moe-predict) run at all: off by default and never switched on by the tuner (user rule: predictor always off); PRED=1 / --moe pred=1 for research
    double self_tune = 1;   // self-tuner: A/B the streaming knobs one at a time on real decode token time (self_tune())
    double offset      = 1; // predicted uploads only for layers far enough ahead to land in time on their link
    double stream_slow = 1; // stream onto slow-link (x4) layers too
    double trace       = 0; // LLAMA_MOE_CACHE_TRACE set: record this many steps (re-armed whenever a ctl file sets it)
    double trace_after = 0; // ... starting after this many steps
    double admit       = 2; // a missed expert needs ADMIT uses in the last 64 tokens before it may take a slot. Default 2 (of the experts used in a 64-token window 69% reach 2 uses, 52% reach 3, and the 3rd use predicts further use only 16% better; Qwen Next: 2 and 3 equal); was 0 (any miss) until 6 Oct, 2 from 3 Oct to 4 Oct
                            // 3.0-bit (ppl comparator, 3 runs, ranges apart): a single recent use of a historically popular expert no longer evicts one that is hot now
    double admit_boot  = 6; // ADMIT_BOOT=N: the admit used for the first ADMIT_BOOT_STEPS decode steps at the start of the process, while the cache is still being filled: strict entry saves
                            // uploads that cost more than they return (Qwen3.6 game4, 100 tokens: admit 8 is +5.8% over 2, hit rate flat; at 1024 tokens the gain is gone); 0 or <= ADMIT: off
    double admit_boot_steps = 128;
    double admit_slow  = 0; // ADMIT_SLOW=N: the admission rule of layers on the slower link (their uploads cost 4-7x more; 0: same as ADMIT)
    double lead        = 1; // LEAD=1: the predicted layers per link are the nearest ones whose upload still lands in time: the lookahead window starts at
                            // ceil(measured upload time per expert on that link / measured layer time) and is LEAD_SPAN + 1 layers wide; the predictor's lookahead
                            // count is derived from the probed link speeds (PREDICT_AHEAD overrides). 0: every lookahead up to PREDICT_AHEAD (the old behaviour)
    double lead_span   = 1; // LEAD_SPAN: layers beyond the nearest feasible one that are also used
    double admit_jit   = 0; // ADMIT_JIT=N: the same admission rule for just-in-time uploads (a miss of this token goes to the GPU only after N uses in the last 64 tokens); 0: any
    double jit         = 1; // JIT miss offload (JIT=0 off): once a layer's router ids reach the host, upload the k misses (k from the
                            // measured link / CPU / combined RAM rates) that make CPU + link finish soonest into cache slots on
                            // the chain's stream, so the GPU computes them this token while the CPU computes the rest
};

// a knob name as typed: case and '-' / '_' do not matter (swap-lead = SWAP_LEAD)
// the --moe key of a knob: lowercase with '-' (SWAP_LEAD = swap-lead)
std::string knob_key(std::string name) {
    for (char & c : name) { c = c == '_' ? '-' : (char) tolower((unsigned char) c); }
    return name;
}

std::string knob_canon(std::string name) {
    for (char & c : name) { c = c == '-' ? '_' : (char) toupper((unsigned char) c); }
    return name;
}

static const std::vector<std::pair<const char *, double knobs_t::*>> & knob_fields() {
    static const std::vector<std::pair<const char *, double knobs_t::*>> fields = {
        { "UPLOAD_SHARE", &knobs_t::swap_frac }, { "PIN_HOT", &knobs_t::hot_frac }, { "STAY_BONUS", &knobs_t::sticky }, { "SLOW_MIN_STAY", &knobs_t::slow_stay },
        { "SWAP_LEAD", &knobs_t::margin }, { "SWAPS_PER_STEP", &knobs_t::budget }, { "CPU_RAM_GBS", &knobs_t::cpu_gbs }, { "SWAP_LEAD_PER_LINK", &knobs_t::link },
        { "UPLOAD_WAIT", &knobs_t::gate }, { "UPLOAD_WAIT_MAX_US", &knobs_t::gate_max_us }, { "RAM_CEILING_GBS", &knobs_t::ddr_gbs },
        { "VRAM_RESERVE_MB", &knobs_t::margin_mb }, { "MARGIN_MB", &knobs_t::margin_mb }, { "WAIT_SWAPS", &knobs_t::wait }, { "PRED_AUTO", &knobs_t::auto_tune },
        { "EV_CLD", &knobs_t::big }, { "L3_PF", &knobs_t::l3pf }, { "L3_AUTO", &knobs_t::auto_l3 }, { "IDLE_UP", &knobs_t::tbp }, { "IDLE_UP_N", &knobs_t::tbp_layers }, { "UPLOAD_CHUNK_KB", &knobs_t::chunk_kb },
        { "PRED_ISO", &knobs_t::stream }, { "PRED_N", &knobs_t::stream_m }, { "PRED_KEEP", &knobs_t::slotkeep }, { "PRED_TUNE", &knobs_t::self_tune }, { "PRED", &knobs_t::predict },
        { "PRED_LEAD", &knobs_t::offset }, { "PRED_SLOW", &knobs_t::stream_slow },
        { "TRACE_N", &knobs_t::trace }, { "TRACE_SKIP", &knobs_t::trace_after }, { "UPLOAD_NOW", &knobs_t::jit },
        { "LEAD", &knobs_t::lead }, { "LEAD_SPAN", &knobs_t::lead_span }, { "ADMIT", &knobs_t::admit }, { "ADMIT_BOOT", &knobs_t::admit_boot }, { "ADMIT_BOOT_STEPS", &knobs_t::admit_boot_steps }, { "ADMIT_SLOW", &knobs_t::admit_slow }, { "ADMIT_JIT", &knobs_t::admit_jit },
    };
    return fields;
}

bool knob_set(knobs_t & k, const std::string & name_in, double v) {
    const std::string name = knob_canon(name_in);
    for (const auto & f : knob_fields()) {
        if (name == f.first) {
            k.*f.second = v;
            return true;
        }
    }
    return false;
}

// knobs the user set (LLAMA_MOE_CACHE_<NAME> or --moe NAME=value): the self-tuner and the saved tuner state leave them alone
static bool g_autotune_off = false; // -at off / --moe autotune=0 / cache=0: the tuner list stays empty, the knobs keep their defaults

std::set<std::string> & user_knobs() {
    static std::set<std::string> s;
    return s;
}

knobs_t & knobs() {
    static knobs_t k = [] {
        knobs_t r;
        for (const char * n : { "VRAM_RESERVE_MB", "UPLOAD_SHARE", "PIN_HOT", "STAY_BONUS", "SLOW_MIN_STAY", "SWAP_LEAD", "SWAPS_PER_STEP", "CPU_RAM_GBS", "SWAP_LEAD_PER_LINK", "UPLOAD_WAIT", "UPLOAD_WAIT_MAX_US", "PRED_ISO", "PRED_N", "TRACE_N", "TRACE_SKIP", "LEAD", "LEAD_SPAN", "ADMIT", "ADMIT_BOOT", "ADMIT_BOOT_STEPS", "ADMIT_SLOW", "ADMIT_JIT", "PRED_LEAD", "PRED_SLOW", "RAM_CEILING_GBS", "WAIT_SWAPS", "UPLOAD_CHUNK_KB", "PRED_AUTO", "IDLE_UP", "IDLE_UP_N", "EV_CLD", "L3_PF", "L3_AUTO", "PRED_KEEP", "PRED_TUNE", "PRED", "UPLOAD_NOW" }) {
            if (const char * e = moe_opt(knob_key(n).c_str())) {
                knob_set(r, n, atof(e));
                user_knobs().insert(n);
            }
        }
        return r;
    }();
    return k;
}

// "NAME=value ..." of every knob the cache runs with at this moment (tuner decisions included); a * marks the ones the user set. Written to the state snapshot as run.knobs (after the key settings).
std::string knob_dump() {
    std::string out;
    char b[64];
    for (const auto & f : knob_fields()) {
        snprintf(b, sizeof b, "%g", knobs().*f.second);
        out += std::string(out.empty() ? "" : " ") + f.first + (user_knobs().count(f.first) ? "*=" : "=") + b;
    }
    return out;
}

enum class policy { halve, window, hybrid, add, lru, rel };
std::atomic<bool> g_rel_dead{false};   // policy r without router probabilities: the cache would never fill (relevance 0 everywhere), so it falls back to the default score
// hot maps per topic (--moe dynhot=1, set at the first prompt): the lifetime term of the score is the topic's map plus this request's own uses (scale: its busiest
// expert of any layer, g_topic_tmax) instead of the one map of every request ever served
std::atomic<bool>     g_topic_on{false};
std::atomic<bool>     g_topic_thinking{false};   // the request being served is thinking (between <think> and </think>): its answer's uses count in the thinking map
std::atomic<bool>     g_topic_answer{false};     // its prompt is over (its key came): from here on its uses are the answer's (a short prompt runs as decode steps too)

// --moe expids=N (0: off, the default): per expert of every layer, the last N token ids the router sent to it (user idea 2026-10-28: per expert the
// pooled input embedding of the tokens that activated it, for a predictor that plans expert uploads from the coming tokens). The ids, not vectors: a
// token's layer-0 input is its embedding row, so any pooling (max, mean) is computed from them later, and 100 ids per expert are 4.4 MB on DeepSeek-V4.1
// (43 layers x 256 experts) where 4-bit max-pooled vectors would be 28 MB. Decode calls (and any call of one micro-batch) only: a prompt pass on the GPU
// reports which experts it used, not which token used which (its routing stays on the card). Kept in <state dir>/hotmap/<model>.expids, read at the
// first prompt, written when a request ends (the server calls llama_moe_request_end_fork), at the start of each later prompt and at the end (exptok_save).
struct exptok_state {
    int32_t               n = -1;      // ring size per expert; 0 = off; -1 = not set up yet
    uint32_t              n_layers = 0, n_exp = 0;
    std::vector<int32_t>  ring;        // n_layers x n_exp x n token ids
    std::vector<uint32_t> seen;        // n_layers x n_exp: tokens routed to the expert so far (the ring's next slot = seen % n)
    std::vector<int32_t>  batch;       // the token ids of the call being computed: row t of a layer's routing is batch[t]
    std::string           path;
    bool                  dirty = false;
} g_exptok;

// --moe tokdump=PATH (experiment, user 2026-10-28: "store all selected (8) of all layers moes per layer 0 activation ... see how much bigger hitrate we
// would have if we used it to prefetch too"): per answer token, the input of the first and of the last MoE layer (fp16; the first already holds the whole
// context, through that layer's attention) and the experts every MoE layer picked for it, each with whether it was cached then. run.py tokpred replays the
// file as a nearest-neighbour memory (activation -> the experts of all layers) and counts the misses a prefetch from it would have saved; nothing here
// changes what the cache does. Decode steps only: a prompt pass on the GPU never shows its routing to the host.
// File: "MOETOKD1", then n_embd, MoE layers, experts per token, experts per layer (uint32 each); per token: its request's number (uint32), the two inputs
// (n_embd fp16 each), per MoE layer the experts it picked (int16) and whether each was cached (uint8).
struct tokdump_state {
    int      on = -1;                 // -1: not decided yet
    FILE *   f  = nullptr;
    uint32_t n_embd = 0, n_lay = 0, n_used = 0;
    uint32_t req = 0;                 // requests begun (llama_moe_cache_topic_begin_fork)
    uint64_t step = UINT64_MAX;       // the step whose tokens are being filled
    int64_t  n_tok = 0;               // its tokens (0: none)
    std::vector<ggml_fp16_t> k0, kl;  // per token: the input of the first and of the last MoE layer
    std::vector<int16_t>     ids;     // per token: MoE layers x experts per token
    std::vector<uint8_t>     hit;     // per token: MoE layers x experts per token, 1 = cached
} g_tokdump;
std::atomic<uint32_t> g_topic_tmax{1};

policy get_policy() {
    static const policy p = [] {
        const char * e = moe_opt("policy");
        if (e && strcmp(e, "window") == 0) return policy::window;
        if (e && strcmp(e, "hybrid") == 0) return policy::hybrid;
        if (e && strcmp(e, "halve")  == 0) return policy::halve;
        if (e && (strcmp(e, "r") == 0 || strcmp(e, "relevance") == 0)) return policy::rel;   // only the router's relevance of the expert to the current context: no recency, window or hit counts
        if (e && strcmp(e, "lru")    == 0) return policy::lru;   // plain least recently used: the score is the time of the last use, no lifetime counts, no pay-back margin
        return policy::add;
    }();
    return p;
}

// eviction score, in "uses" units so the pay-back margin applies to all policies
// gmax: the scale of the lifetime-count term. Within a layer its own busiest expert (ls.glob_max); when experts of different layers compete
// for a block (block pool) one common scale, else a layer that spreads its traffic over many experts gets all of them scored near the top and
// a layer with a few very hot experts loses its second tier (measured MiMo stand-in, 25% cache: layer 4 fell from 64 to 13 experts, hit 68 -> 43%)
double score(const layer_state & ls, int32_t id, uint64_t gmax) {
    switch (get_policy()) {
        case policy::window: return ls.win_count[id];
        case policy::lru:    return (double) ls.expert_last[id];
        case policy::rel:
            if (!g_rel_dead.load(std::memory_order_relaxed)) { return (double) ls.relevance[id]; }
            return ls.win_count[id] + 16.0 * ((double) ls.glob_count[id] / (double) gmax);   // no router probabilities reached the cache: the default score
        case policy::hybrid: return ls.win_count[id] * ((double) ls.glob_count[id] / (double) gmax);
        case policy::add: {
            static const double k = [] {
                const char * w = moe_opt("global-weight");
                return w ? atof(w) : 16.0;
            }();
            // hot maps per topic: the map of the stored requests nearest to this prompt (0..1, 0 for a new topic) and this request's own uses take the
            // lifetime counts' place: the counts of every past request kept other topics' experts in the cache ("stubborn"), so a new topic evicted them
            // before its own cold ones
            // (scale of the own uses: the busiest expert of any layer as of the last step, at least this layer's: current within a prompt's first pass)
            if (g_topic_on.load(std::memory_order_relaxed)) {
                const double tmax = (double) std::max(g_topic_tmax.load(std::memory_order_relaxed), ls.topic_max);
                return ls.win_count[id] + k * ((double) ls.topic_heat[id] + (double) ls.topic_cnt[id] / tmax);
            }
            return ls.win_count[id] + k * ((double) ls.glob_count[id] / (double) gmax);
        }
        case policy::halve:  return ls.expert_count[id];
    }
    return 0;
}

double score(const layer_state & ls, int32_t id) {
    return score(ls, id, ls.glob_max);
}

// the pay-back margin is in uses, which has no meaning for the time of the last use: plain LRU swaps whenever the candidate was used more recently
double pay_back(int margin) {
    return get_policy() == policy::lru || get_policy() == policy::rel ? 0.0 : (double) margin;
}

// Hot-set profile: which experts this model used, so the next start preloads them into VRAM
// instead of warming the cache over the first requests. Keyed by model name + file size.
// LLAMA_MOE_CACHE_PROFILE=<path> overrides the location, =0 disables.
std::string cache_file(const llama_model & model, const char * prefix, const std::string & suffix = "") {
#ifdef _WIN32
    const char * base = getenv("LOCALAPPDATA");
    const std::string dir = base ? std::string(base) + "\\llama.cpp" : "";
#else
    const char * xdg  = getenv("XDG_CACHE_HOME");
    const char * home = getenv("HOME");
    const std::string dir = xdg ? std::string(xdg) + "/llama.cpp" : home ? std::string(home) + "/.cache/llama.cpp" : "";
#endif
    if (dir.empty()) {
        return "";
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string name = model.name;
    for (char & c : name) {
        if (!isalnum((unsigned char) c) && c != '-' && c != '.') {
            c = '_';
        }
    }
    return dir + "/" + prefix + name + "-" + std::to_string(model.size()) + suffix + ".bin";
}

// the model's section in the state file; empty when the state is off (LLAMA_MOE_STATE=0 / LLAMA_MOE_CACHE_PROFILE=0)
int parse_layer_from_name(const char * name);

std::string profile_path(const llama_model & model) {
    return moe_state_enabled() ? moe_state_section(model) : "";
}

void profile_save(const moe_cache * mc) {
    if (mc->profile.empty() || mc->n_steps < 64) { // too little use to be worth keeping
        return;
    }
    // raw lifetime activation counts per expert (the cache normalizes by the layer max when scoring); a layer is halved
    // while its max exceeds 2^31, which keeps the ratios. One line per layer: hot.<layer> = c0 c1 c2 ...
    std::vector<std::pair<std::string, std::string>> kv;
    for (size_t il = 0; il < mc->layers.size(); ++il) {
        const auto & ls = mc->layers[il];
        int shift = 0;
        while ((ls.glob_max >> shift) > (1ull << 31)) {
            shift++;
        }
        std::string v;
        for (size_t e = 0; e < ls.glob_count.size(); ++e) {
            v += (e ? " " : "") + std::to_string((uint32_t) (ls.glob_count[e] >> shift));
        }
        kv.emplace_back("hot." + std::to_string(parse_layer_from_name(ls.pub.up_src->name)), v);   // real layer: dense layers are not cache layers
    }
    if (false && !mc->tuned_text.empty()) {   // ini-fix (2026-10-22): the self-tuner's decisions are not stored (see the reader)
        // the self-tuner's decisions of this GPU count: tuned.l<links> = NAME=value NAME=value ...
        std::string v = mc->tuned_text;
        for (char & c : v) { if (c == '\n') { c = ' '; } }
        kv.emplace_back(hw_key("tuned.l" + std::to_string(mc->n_links)), v);
    }
    // what this run did, per name the last 9 runs: run.<name> = <median> <last> | <oldest> ... <newest> (hit rate of the cache, evictions, uploads and
    // uploaded MiB per step, share of the prefill's experts found in the cache)
    {
        uint64_t h = 0, m = 0;
        for (const auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        const double steps = (double) mc->n_steps;
        const std::pair<const char *, double> run[] = {
            { "hit_pct",       h + m ? 100.0*h/(h + m) : 0.0 },
            { "evict_step",    mc->n_evictions/steps },
            { "upload_step",   mc->n_uploads/steps },
            { "up_mib_step",   std::max<int64_t>(0, mc->up_bytes.load())/1048576.0/steps },
            { "prefill_hit_pct", mc->n_src_query ? 100.0*mc->n_src_hit/mc->n_src_query : 0.0 },
            // the predictor's own result (only when it uploaded something): share of its uploads that were used, share of its candidates too late to start,
            // and the share of the real misses its top-1 / top-2 candidates would have covered
            { "pred_used_pct", mc->n_pred_up ? 100.0*mc->n_pred_used/mc->n_pred_up : -1.0 },
            { "pred_late_pct", mc->n_pred_up + mc->n_pred_late ? 100.0*mc->n_pred_late/(mc->n_pred_up + mc->n_pred_late) : -1.0 },
            { "pred_cov1_pct", mc->pm_miss ? 100.0*mc->pm_cov[0]/mc->pm_miss : -1.0 },
            { "pred_cov2_pct", mc->pm_miss ? 100.0*mc->pm_cov[1]/mc->pm_miss : -1.0 },
            // hit rate points the predictor added: each predicted upload used once is a miss that became a hit (estimate, no second run needed)
            { "pred_boost_pts", mc->n_pred_up && h + m ? 100.0*mc->n_pred_used/(h + m) : -1.0 },
        };
        for (const auto & r : run) {
            if (r.second < 0) { continue; }   // no predictor this run: nothing to record
            std::vector<double> v;
            std::string old;
            if (moe_state_get(mc->profile, std::string("run.") + r.first, old)) {
                const size_t bar = old.find('|');
                std::istringstream is(bar == std::string::npos ? "" : old.substr(bar + 1));
                for (double x; is >> x; ) { v.push_back(x); }
            }
            v.push_back(r.second);
            if (v.size() > 9) { v.erase(v.begin(), v.end() - 9); }
            std::vector<double> t = v;
            std::sort(t.begin(), t.end());
            char nb[48];
            snprintf(nb, sizeof nb, "%.3f %.3f |", t[t.size()/2], v.back());
            std::string out = nb;
            for (double x : v) { snprintf(nb, sizeof nb, " %.3f", x); out += nb; }
            kv.emplace_back(std::string("run.") + r.first, out);
        }
    }
    kv.emplace_back("run.knobs", (mc->settings.empty() ? std::string() : mc->settings + " | ") + knob_dump());   // key settings, then every knob   // the knobs of this run at its end (what was selected, tuner decisions included)
    moe_state_set(mc->profile, kv);
    if (moe_embsnap_enabled()) {   // research: this run's counts alone (lifetime counts minus what the preloaded profile brought in)
        std::vector<std::pair<std::string, std::string>> hot;
        for (const auto & ls : mc->layers) {
            std::string v;
            for (size_t e = 0; e < ls.glob_count.size(); ++e) {
                const uint64_t b = e < ls.glob_base.size() ? ls.glob_base[e] : 0;
                v += (e ? " " : "") + std::to_string(ls.glob_count[e] > b ? ls.glob_count[e] - b : 0);
            }
            hot.emplace_back("hot." + std::to_string(parse_layer_from_name(ls.pub.up_src->name)), v);
        }
        moe_embsnap_write(hot);
    }
}

// not evictable: sticky, or cached on a slow-link layer for less than LLAMA_MOE_CACHE_SLOW_MIN_STAY steps (default 1024):
// on a slow link a swap costs more, so an expert must stay long enough to pay for its upload
bool pinned(const moe_cache * mc, const layer_state & ls, int32_t e);

// seed usage from the profile and queue the most-used experts of each layer for upload;
// the first step() waits for them and publishes them. Returns the number queued.
// Always-hot experts (formatting, common tokens) stay in VRAM: the most-used experts by lifetime
// count, up to LLAMA_MOE_CACHE_STAY_BONUS (default 0, e.g. 0.3) of each layer's slots, are never evicted.
// the always-hot set: the fewest experts (by lifetime count) covering LLAMA_MOE_CACHE_PIN_HOT (default 0.75) of the picks
void refresh_hot(layer_state & ls) {
    const double hot_frac = knobs().hot_frac;
    ls.hot.assign(ls.glob_count.size(), false);
    std::vector<int32_t> ids(ls.glob_count.size());
    uint64_t tot = 0;
    for (int32_t e = 0; e < (int32_t) ids.size(); ++e) { ids[e] = e; tot += ls.glob_count[e]; }
    if (tot == 0) {
        return;
    }
    std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
    uint64_t acc = 0;
    for (int32_t e : ids) {
        if (acc >= hot_frac * tot) {
            break;
        }
        ls.hot[e] = true;
        acc += ls.glob_count[e];
    }
}

void refresh_sticky(layer_state & ls) {
    const double frac = knobs().sticky;
    refresh_hot(ls);
    ls.sticky.assign(ls.glob_count.size(), false);
    if (ls.slow) {
        // slow upload link: its always-hot experts stay (the most-used first, up to the slots), swaps are rare
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < (int32_t) ls.hot.size(); ++e) {
            if (ls.hot[e]) {
                ids.push_back(e);
            }
        }
        std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
        for (int32_t i = 0; i < (int32_t) ids.size() && i < ls.n_cache; ++i) {
            ls.sticky[ids[i]] = true;
        }
        return;
    }
    const int32_t k = (int32_t) (frac * ls.n_cache);
    if (k <= 0) {
        return;
    }
    std::vector<int32_t> ids;
    for (int32_t e = 0; e < (int32_t) ls.glob_count.size(); ++e) {
        if (ls.glob_count[e] > 0) {
            ids.push_back(e);
        }
    }
    const int32_t n = std::min<int32_t>(k, (int32_t) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
    for (int32_t i = 0; i < n; ++i) {
        ls.sticky[ids[i]] = true;
    }
}

int parse_layer_from_name(const char * name);

}  // namespace (reopened below)

namespace {
struct stock_heat_t {
    std::mutex mtx;
    bool on = false;
    std::string section;
    std::vector<std::vector<uint64_t>> counts;   // [layer][expert]
    uint64_t tokens = 0;                         // routed tokens seen (summed over observed layers)
    int64_t steps = 0;                           // decode calls seen while counting
} g_heat;
std::atomic<bool> g_heat_live{false};            // counting right now (stops after observe-steps decode calls; the counts stay for the save): read without the lock once per decode call

void * cuda_obs_proc(const char * name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

void stock_obs_cb(const char * name, const struct ggml_tensor * ids, const struct ggml_tensor *, void *) {
    const int il = parse_layer_from_name(name);
    if (il < 0 || ids->type != GGML_TYPE_I32 || ids->ne[1] > 8) {   // > 8 tokens is a prompt batch: the experts a question uses are not the ones the answer uses
        return;
    }
    std::lock_guard<std::mutex> lk(g_heat.mtx);
    if ((size_t) il >= g_heat.counts.size()) {
        return;
    }
    auto & c = g_heat.counts[il];
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id >= 0 && (size_t) id < c.size()) {
                c[id]++;
            }
        }
    }
    g_heat.tokens += ids->ne[1];
}
// a file with no hot map of its own yet starts from the one of the same file name or the same family and shape (other quant, other build of the file): it is copied
// once into this file's own section, which then learns on its own; only layers with the model's expert count are taken
void moe_seed_hot(const std::string & section, size_t n_expert) {
    // ini-fix (2026-10-22): no hot map is copied from another file's record (other quant, other build): a model starts from its own use counts or none
    (void) section; (void) n_expert;
    if (true) { return; }
    std::vector<std::pair<std::string, std::string>> own, donor, hot;
    if (section.empty() || n_expert == 0 || !moe_state_enabled()) {
        return;
    }
    moe_state_section_kv(section, own);
    if (std::any_of(own.begin(), own.end(), [](const auto & p) { return p.first.compare(0, 4, "hot.") == 0; }) || !moe_state_section_kv(section, donor, true)) {
        return;
    }
    for (const auto & p : donor) {
        if (p.first.compare(0, 4, "hot.") == 0 && (size_t) std::count(p.second.begin(), p.second.end(), ' ') + 1 == n_expert) { hot.push_back(p); }
    }
    if (!hot.empty() && moe_state_set(section, hot)) {
        LLAMA_LOG_INFO("moe-cache: no hot map for [%s]: seeded %zu layers from a same-name or same-family record\n", section.c_str(), hot.size());
    }
}

}  // namespace

void llama_moe_observe_start(const llama_model & model) {
    const char * off = moe_opt("observe");
    if ((off && off[0] == '0') || model.hparams.n_expert == 0 || !moe_state_enabled() || llama_moe_cache_active_fork()) {
        return;
    }
    // a model whose experts all sit on GPUs keeps the stock placement for good: the cache never runs on it, so there is no hot map to learn
    bool any_host_experts = false;
    for (const auto & l : model.layers) {
        if (!l.ffn_up_exps || !l.ffn_up_exps->buffer) {
            continue;
        }
        ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_up_exps->buffer));
        if (ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) || (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            any_host_experts = true;
            break;
        }
    }
    if (!any_host_experts) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_heat.mtx);
    g_heat.section = moe_state_section(model);
    moe_seed_hot(g_heat.section, model.hparams.n_expert);
    g_heat.counts.assign(model.hparams.n_layer(), std::vector<uint64_t>(model.hparams.n_expert, 0));
    g_heat.tokens = 0;
    g_heat.steps = 0;
    g_heat.on = true;
    g_heat_live = true;
    ggml_set_moe_obs_callback(stock_obs_cb, nullptr);
    if (auto fn = (void (*)(bool)) cuda_obs_proc("ggml_backend_cuda_obs_enable")) {
        fn(true);   // GPU-resident layers count on the device, read back once in llama_moe_observe_save
    }
}

// The hot map is that of the answer (decode batches of at most 8 tokens: a prompt batch routes differently and is not counted). It is stable after a few hundred tokens. The counting itself costs every token (a lock and a read-back of the router ids on CPU layers, one extra
// kernel per layer on GPU layers), so it stops after observe-steps decode calls (--moe observe-steps=N, default 512, 0: never) and the run goes on at full speed.
void llama_moe_observe_step() {
    if (!g_heat_live.load(std::memory_order_relaxed)) {
        return;
    }
    static const int64_t cap = [] { const char * v = moe_opt("observe-steps"); return v ? (int64_t) atoll(v) : (int64_t) 512; }();
    std::lock_guard<std::mutex> lk(g_heat.mtx);
    if (cap <= 0 || ++g_heat.steps < cap) {
        return;
    }
    g_heat_live = false;
    ggml_set_moe_obs_callback(nullptr, nullptr);
    if (auto en = (void (*)(bool)) cuda_obs_proc("ggml_backend_cuda_obs_enable")) {
        en(false);   // the device counters keep what they have; llama_moe_observe_save reads them
    }
}

void llama_moe_observe_save() {
    std::vector<std::pair<std::string, std::string>> kv;
    std::vector<std::pair<std::string, std::string>> embhot;   // research: embsnap
    {
        std::lock_guard<std::mutex> lk(g_heat.mtx);
        if (!g_heat.on) {
            return;
        }
        g_heat.on = false;
        ggml_set_moe_obs_callback(nullptr, nullptr);
        if (auto en = (void (*)(bool)) cuda_obs_proc("ggml_backend_cuda_obs_enable")) {
            en(false);
        }
        if (auto rd = (bool (*)(int, int, uint64_t *)) cuda_obs_proc("ggml_backend_cuda_obs_read")) {
            const int nl = (int) g_heat.counts.size(), ne = (int) g_heat.counts[0].size();
            std::vector<uint64_t> gpu((size_t) nl*ne, 0);
            if (rd(nl, ne, gpu.data())) {
                for (int il = 0; il < nl; ++il) {
                    for (int e = 0; e < ne; ++e) {
                        g_heat.counts[il][e] += gpu[(size_t) il*ne + e];
                    }
                }
                g_heat.tokens += 64;   // GPU layers counted: keep even when the CPU saw few tokens
            }
        }
        if (g_heat.tokens < 64) {   // too little use to be worth keeping
            return;
        }
        for (size_t il = 0; il < g_heat.counts.size(); ++il) {
            auto c = g_heat.counts[il];
            uint64_t sum = 0;
            for (auto x : c) { sum += x; }
            if (sum == 0) {
                continue;   // a layer whose experts run on the GPU: nothing was seen
            }
            if (moe_embsnap_enabled()) {   // research: this run's counts alone
                std::string r;
                for (size_t e = 0; e < c.size(); ++e) { r += (e ? " " : "") + std::to_string(c[e]); }
                embhot.emplace_back("hot." + std::to_string(il), r);
            }
            std::string old;   // earlier runs' counts of this layer are added
            if (moe_state_get(g_heat.section, "hot." + std::to_string(il), old)) {
                std::istringstream is(old);
                for (size_t e = 0; e < c.size(); ++e) {
                    uint64_t v = 0;
                    if (!(is >> v)) { break; }
                    c[e] += v;
                }
            }
            uint64_t mx = 0;
            for (auto x : c) { mx = std::max(mx, x); }
            int shift = 0;
            while ((mx >> shift) > (1ull << 31)) { shift++; }
            std::string v;
            for (size_t e = 0; e < c.size(); ++e) {
                v += (e ? " " : "") + std::to_string((uint32_t) (c[e] >> shift));
            }
            kv.emplace_back("hot." + std::to_string(il), v);
        }
    }
    if (!kv.empty()) {
        moe_state_set(g_heat.section, kv);
        moe_embsnap_write(embhot);
    }
}

namespace {
void set_table_entry(llama_moe_cache_layer_fork & pub, int32_t expert, int32_t slot_or_dummy);
void page_hint(const moe_cache * mc, const layer_state & ls, int32_t expert, bool drop);
void note_evict(moe_cache * mc, layer_state & ls, int32_t e);
double score(const layer_state & ls, int32_t id, uint64_t gmax);
double score(const layer_state & ls, int32_t id);
// score of an expert competing with experts of other layers for a block: one common scale (moe_cache::gmax_all) in the pool
inline double xscore(const moe_cache * mc, const layer_state & ls, int32_t id) {
    return ls.pcls >= 0 ? score(ls, id, std::max<uint64_t>(1, mc->gmax_all)) : score(ls, id);
}

// ---- block pool runtime ----------------------------------------------------------------------------------------------------------------------------
// With the block pool the slot index used everywhere (tables, upload offsets, the slot arrays of a layer) is a block of the layer's size class.
// Invariant: ls.slot_expert[s] >= 0 or ls.slot_in_flight[s] only in the layer that owns block s. Without the pool (--moe pool=0) a slot is one of
// the layer's own tensor slots and these helpers keep the old per-layer behaviour.
struct slot_pick {
    int32_t slot  = -1;
    int32_t owner = -1;   // layer whose cached expert sits in the block; -1: the block is free or empty
};

// the block an expert of layer li may take: a free (or owned but empty) block if there is one, else the block whose cached expert has the lowest
// rank(owner layer, owner index, expert) >= 0 (< 0: may not be evicted). Pool: every block of li's class, whichever layer owns it: no layer has a
// quota. best: the rank of the returned victim, 0 for a free block
template <class Rank>
slot_pick pick_slot(moe_cache * mc, size_t li, Rank rank, double & best) {
    layer_state & ls = mc->layers[li];
    best = 1e300;
    slot_pick p;
    if (ls.pcls >= 0) {
        const poolx_pick pk = poolx_pick_slot(mc->pool[ls.pcls], mc->pool[ls.pcls].n_active(),
                [&](int32_t o, int32_t b) { return mc->layers[o].slot_in_flight[b] || mc->layers[o].is_stream[b]; },
                [&](int32_t o, int32_t b) { return mc->layers[o].slot_expert[b]; },
                [&](int32_t o, int32_t e) { return rank(mc->layers[o], (size_t) o, e); }, best);
        p.slot = pk.slot; p.owner = pk.owner;   // rules and their reasons: llama-poolalloc-fork.h
        return p;
    }
    for (int32_t b = 0; b < ls.pub.n_slots; ++b) {
        if (ls.slot_in_flight[b] || ls.is_stream[b]) {
            continue;
        }
        const int32_t e = ls.slot_expert[b];
        if (e < 0) { p.slot = b; best = 0; return p; }
        const double r = rank(ls, li, e);
        if (r >= 0 && r < best) { best = r; p.slot = b; p.owner = (int32_t) li; }
    }
    return p;
}

// block b now belongs to layer li (the expert its previous owner cached there must be evicted first)
void pool_take(moe_cache * mc, size_t li, int32_t b) {
    layer_state & ls = mc->layers[li];
    if (ls.pcls < 0) {
        return;
    }
    auto & ps = mc->pool[ls.pcls];
    const int32_t o = ps.owner[b];
    GGML_ASSERT(o != -3 && "kv-tail: an expert is assigned to a block under the KV rows");
    if (o == (int32_t) li) {
        return;
    }
    if (o < 0) {
        auto it = std::find(ps.free.rbegin(), ps.free.rend(), b);
        if (it != ps.free.rend()) {
            ps.free.erase(std::next(it).base());
        }
    } else {
        mc->layers[o].n_own--;
    }
    ps.owner[b] = (int32_t) li;
    ls.n_own++;
}

// layer li gives the empty block b back: any layer of the class may take it
void pool_release(moe_cache * mc, size_t li, int32_t b) {
    layer_state & ls = mc->layers[li];
    if (ls.pcls < 0 || mc->pool[ls.pcls].owner[b] != (int32_t) li) {
        return;
    }
    mc->pool[ls.pcls].owner[b] = -1;
    mc->pool[ls.pcls].free.push_back(b);
    ls.n_own--;
}

// evict the expert layer lo caches in block b; its table entry goes back to uncached now (call at a sync point)
void evict_slot(moe_cache * mc, layer_state & lo, int32_t b) {
    const int32_t v = lo.slot_expert[b];
    if (v < 0) {
        return;
    }
    lo.expert_slot[v] = -1;
    lo.slot_expert[b] = -1;
    set_table_entry(lo.pub, v, lo.pub.n_slots);
    note_evict(mc, lo, v);
    lo.cached_since[v] = 0;
    if (lo.dropped[v]) { page_hint(mc, lo, v, false); lo.dropped[v] = false; }
}

size_t profile_preload(moe_cache * mc, const llama_model & model) {
    std::vector<std::vector<uint64_t>> counts;
    const char * from = mc->profile.c_str();
    bool ok = !mc->profile.empty();
    std::vector<std::pair<std::string, std::string>> sect;
    moe_seed_hot(mc->profile, mc->layers.empty() ? 0 : mc->layers[0].glob_count.size());
    ok = ok && moe_state_section_kv(mc->profile, sect);
    for (size_t il = 0; ok && il < mc->layers.size(); ++il) {
        const std::string key = "hot." + std::to_string(parse_layer_from_name(mc->layers[il].pub.up_src->name));
        const auto it = std::find_if(sect.begin(), sect.end(), [&](const auto & p) { return p.first == key; });
        ok = it != sect.end();
        const std::string v = ok ? it->second : "";
        if (ok) {
            std::vector<uint64_t> c;
            const char * p = v.c_str();
            char * end = nullptr;
            for (unsigned long long x = strtoull(p, &end, 10); end != p; x = strtoull(p, &end, 10)) { c.push_back(x); p = end; }
            ok = c.size() == mc->layers[il].glob_count.size();
            counts.push_back(std::move(c));
        }
    }
    if (!ok) {
        if (!mc->profile.empty() && moe_state_enabled()) {
            LLAMA_LOG_INFO("moe-cache: no usage profile for [%s] (or another cache layout)\n", from);
        }
        counts.clear();
    }
    if (!ok && !model.moe_expert_usage.empty()) {
        // no local profile: the model's own usage table (GGUF key moe_cache.expert_usage)
        const size_t n_expert = mc->layers.empty() ? 0 : mc->layers[0].glob_count.size();
        counts.clear();
        ok = n_expert > 0;
        for (size_t il = 0; ok && il < mc->layers.size(); ++il) {
            const size_t layer = (size_t) parse_layer_from_name(mc->layers[il].pub.up_src->name);
            ok = mc->layers[il].glob_count.size() == n_expert && (layer + 1)*n_expert <= model.moe_expert_usage.size();
            if (ok) {
                const uint32_t * u = model.moe_expert_usage.data() + layer*n_expert;
                counts.emplace_back(u, u + n_expert);
            }
        }
        from = "GGUF moe_cache.expert_usage";
    }
    if (!ok) {
        return 0;
    }
    LLAMA_LOG_WARN("moe-cache: usage profile from %s\n", from);   // a warning: the server's log keeps no library INFO lines, and whether the cache starts filled is worth seeing
    size_t queued = 0;
    std::lock_guard<std::mutex> lk(mc->wmtx);
    // each layer's hottest experts (its share of the pool), then placed hottest first over ALL layers: pick_slot hands out the highest free block,
    // so the hottest experts of the model sit at the back of the pool and a prompt, which borrows from the front, evicts the coldest ones
    // (placed layer by layer, layer 0's hottest experts took the first blocks: the first ones every prompt pass borrows)
    struct pick { double heat; size_t il; int32_t e; };
    std::vector<pick> picks;
    for (size_t il = 0; il < mc->layers.size(); ++il) {
        auto & ls = mc->layers[il];
        ls.glob_count = counts[il];
        ls.glob_base  = counts[il];
        ls.glob_max   = std::max<uint64_t>(1, *std::max_element(ls.glob_count.begin(), ls.glob_count.end()));
        refresh_sticky(ls);
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < (int32_t) ls.glob_count.size(); ++e) {
            if (ls.glob_count[e] > 0) {
                ids.push_back(e);
            }
        }
        std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
        uint64_t total = 0;
        for (uint64_t c : ls.glob_count) { total += c; }
        for (int32_t i = 0; i < std::min<int32_t>(ls.n_cache, (int32_t) ids.size()); ++i) {
            picks.push_back({ (double) ls.glob_count[ids[i]] / (double) std::max<uint64_t>(1, total), il, ids[i] });   // share of the layer's picks: comparable across layers
        }
    }
    std::stable_sort(picks.begin(), picks.end(), [](const pick & a, const pick & b) { return a.heat > b.heat; });
    for (const auto & pk : picks) {
        auto & ls = mc->layers[pk.il];
        double best;
        const int32_t s = pick_slot(mc, pk.il, [](layer_state &, size_t, int32_t) { return -1.0; }, best).slot;   // free blocks only
        if (s < 0) {
            continue;   // this layer's class is full
        }
        pool_take(mc, pk.il, s);
        ls.slot_in_flight[s] = true;
        ls.queued[pk.e]      = 1;
        upload_job pj { pk.il, pk.e, s };
        pj.t_queued = ggml_time_us();
        mc->todo.push_back(pj);
        queued++;
    }
    mc->wcv.notify_all();
    return queued;
}

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

// ---- pool write ledger (--moe ledger=1): who writes which bytes of a card's pool, when and how much ----------------------------------------------------
// Every writer of the one pool buffer per card leaves a line in <logdir>/pool-ledger-<pid>.csv: t_us, card, kind, lo, hi (byte offsets in that card's pool),
// bytes, who. Kinds: lend (a graph's tensors, CUDA scratch and staging slots, the dense split's copies in front), release (lent blocks back to the experts),
// zero (a class's zero block restored), kv-zone (the zone laid out), kv-move (rows copied), kv-pad (pad rows zeroed), up-start / up-done (an expert upload,
// "bg" for a background refill), keep-hot (an expert moved out of blocks a prompt borrows), mirror (the dense split's copies filled). The file is the
// replay: fork/pool_ledger.py (main tree) rebuilds who owned which bytes at any time and lists every overlap.
// The live check runs at each write against what belongs to someone else right now and logs "pool-ledger: OVERLAP" at once, with both owners:
//   an upload whose block is lent to a graph (-2), under the KV zone (-3) or owned by another layer, at its start or at its end (a lend or a zone that
//   took the block while the copy ran); a zone that reaches into the lend or covers a class's zero block (the decode tables point uncached experts there);
//   a zero-block restore or a keep-hot destination inside the zone or the lend.
// Off by default: a mutex and a formatted line per upload (~2 us against a ~0.1-1 ms copy).
static bool led_on() {
    static const bool on = [] { const char * e = moe_opt("ledger"); return e && atoi(e) != 0; }();
    return on;
}
static std::mutex g_led_mu;
static FILE *     g_led_f = nullptr;
static int64_t    g_led_n_overlap = 0;
static void led_line(const char * card, const char * kind, size_t lo, size_t hi, const std::string & who) {
    if (!led_on()) { return; }
    std::lock_guard<std::mutex> lk(g_led_mu);
    if (!g_led_f) {
        const std::string path = moe_log_file(("pool-ledger-" + std::to_string((long long) time(nullptr)) + ".csv").c_str());
        g_led_f = fopen(path.c_str(), "w");
        if (!g_led_f) { return; }
        fprintf(g_led_f, "t_us,card,kind,lo,hi,bytes,who\n");
        LLAMA_LOG_WARN("pool-ledger: writing %s\n", path.c_str());
    }
    fprintf(g_led_f, "%lld,%s,%s,%zu,%zu,%zu,%s\n", (long long) ggml_time_us(), card, kind, lo, hi, hi > lo ? hi - lo : 0, who.c_str());
    if (strcmp(kind, "OVERLAP") == 0 || strcmp(kind, "lend") == 0) { fflush(g_led_f); }   // a killed server keeps everything up to its last graph
}
static void led_overlap(const char * card, size_t lo, size_t hi, const std::string & what) {
    if (!led_on()) { return; }
    {
        std::lock_guard<std::mutex> lk(g_led_mu);
        if (++g_led_n_overlap > 200) { return; }   // the first ones say it; the file keeps the rest of the story
    }
    LLAMA_LOG_WARN("pool-ledger: OVERLAP on %s [%.3f, %.3f) MiB: %s\n", card, lo / 1048576.0, hi / 1048576.0, what.c_str());
    led_line(card, "OVERLAP", lo, hi, what);
}
static const char * led_card(const moe_cache::alias_pool & ap) { return ap.dev ? ggml_backend_dev_name(ap.dev) : "?"; }
static size_t led_kv_end(const moe_cache::alias_pool & ap) { return poolx_kv_end(ap); }
// the pool and the byte range of block b of size class pcls
static const moe_cache::alias_pool * led_block(const moe_cache * mc, int32_t pcls, int32_t b, size_t & lo, size_t & hi) {
    for (const auto & ap : mc->alias) {
        for (const auto & sg : ap.segs) {
            if (sg.pcls == pcls) { lo = sg.base + (size_t) b * sg.B; hi = lo + sg.B; return &ap; }
        }
    }
    return nullptr;
}
// an upload into block `slot` of layer li's class: at its start and at its end the block must still be that layer's (not lent, not the KV's, not another
// layer's) and outside the zone's bytes
static void led_upload(moe_cache * mc, const char * kind, size_t li, int32_t expert, int32_t slot, bool bg) {
    if (!led_on() || li >= mc->layers.size()) { return; }
    const int32_t pcls = mc->layers[li].pcls;
    if (pcls < 0) { return; }
    size_t lo = 0, hi = 0;
    const moe_cache::alias_pool * ap = led_block(mc, pcls, slot, lo, hi);
    if (!ap) { return; }
    const std::string who = tr_fmt("L%zu e%d block %d class %d%s", li, expert, slot, pcls, bg ? " bg" : "");
    led_line(led_card(*ap), kind, lo, hi, who);
    const int32_t o = slot < (int32_t) mc->pool[pcls].owner.size() ? mc->pool[pcls].owner[slot] : -9;
    if (o == -2) { led_overlap(led_card(*ap), lo, hi, kind + std::string(" ") + who + ": the block is lent to a graph"); }
    else if (o == -3) { led_overlap(led_card(*ap), lo, hi, kind + std::string(" ") + who + ": the block is under the KV zone"); }
    else if (o >= 0 && (size_t) o != li) { led_overlap(led_card(*ap), lo, hi, kind + std::string(" ") + who + tr_fmt(": the block belongs to layer %d", o)); }
    if (ap->kv_h > 0 && lo < led_kv_end(*ap) && hi > ap->kv_start) {
        led_overlap(led_card(*ap), lo, hi, kind + std::string(" ") + who + tr_fmt(": inside the KV zone [%.1f, %.1f) MiB", ap->kv_start / 1048576.0, led_kv_end(*ap) / 1048576.0));
    }
}
// The live region map of one card's pool: where each region starts and ends RIGHT NOW. Computed from the layout fields every layout change updates (lend,
// zone, class segments), so it cannot go stale; ledger=1 logs it as "map" lines at each change (kv_layout, lend) and the replay tool checks every write
// against the map that was current at that time. Regions: lend [0, want_b) (graph + scratch + staging, the blocks in it are owner -2), zone (KV rows,
// blocks under it are owner -3), blocks of each class (experts; each block B bytes = slices + the read-past guard of the last slice), the class's zero block.
using pool_region = poolx_region;
static std::vector<pool_region> pool_map(const moe_cache::alias_pool & ap) { return poolx_regions(ap); }
// The gate every write into an expert block consults first (always on, not only with ledger=1): the block's range must lie in the blocks region, outside the
// zone and off every zero block. A violating write is refused and counted (logged once per 64), never done: a layout bug then costs an upload, not data.
// Use counts per byte range (poolx_ranges, llama-poolalloc-fork.h): upload DMAs, keep-hot copies and graphs enter a range, a zone move waits for its users and holds new ones back.
static poolx_ranges<const void *> g_ranges;
static void pool_enter(const moe_cache::alias_pool & ap, size_t lo, size_t hi, bool w) { g_ranges.enter(&ap, lo, hi, w); }
// A worker never waits for a move: the mover waits for the cache's in-flight uploads (kv_blocks_apply), and a queued upload blocked here is one of them (deadlock,
// 2026-10-18: "an upload into the blocks taken for the KV never finished"). Inside a move range the job is skipped instead, like a stale stream job.
static bool pool_try_enter(const moe_cache::alias_pool & ap, size_t lo, size_t hi, bool w) { return g_ranges.try_enter(&ap, lo, hi, w); }
static void pool_leave(const moe_cache::alias_pool & ap, size_t lo, size_t hi, bool w) { g_ranges.leave(&ap, lo, hi, w); }
static void pool_move_begin(const moe_cache::alias_pool & ap, size_t lo, size_t hi) {
    if (!g_ranges.move_begin(&ap, lo, hi, 10)) {
        // going on would copy under a DMA or a graph still reading the range: corrupt rows or a corrupt expert block, found many tokens later. Stop here with the numbers.
        GGML_ABORT("pool-map: a move of [%.1f, %.1f) MiB on %s waited 10 s for users of the range (a stuck upload or graph): not moving under them",
                   lo / 1048576.0, hi / 1048576.0, ggml_backend_dev_name(ap.dev));
    }
}
static void pool_move_end(const moe_cache::alias_pool & ap) {
    static int64_t shown_us = 0;
    if (g_ranges.st_enter_us.load() + g_ranges.st_move_us.load() != shown_us) {   // always when the waiting grew: the time lost to the map is never silent
        shown_us = g_ranges.st_enter_us.load() + g_ranges.st_move_us.load();
        LLAMA_LOG_WARN("pool-map: stalls so far: %lld of %lld range entries waited for a move (%.2f ms, longest %.2f ms); %lld moves waited for users (%.2f ms)\n", (long long) g_ranges.st_enter_n.load(),
                (long long) g_ranges.st_calls.load(), g_ranges.st_enter_us.load() / 1e3, g_ranges.st_enter_max.load() / 1e3, (long long) g_ranges.st_move_n.load(), g_ranges.st_move_us.load() / 1e3);
    }
    g_ranges.move_end(&ap);
}
static std::atomic<int64_t> g_gate_refused{0};
static bool pool_gate(const moe_cache::alias_pool & ap, size_t lo, size_t hi, const char * who) {
    size_t rlo = 0, rhi = 0;
    const char * kind = poolx_gate_kind(ap, lo, hi, &rlo, &rhi);   // no allocation, no string compares: this runs for every upload
    if (!kind) { return true; }
    if ((g_gate_refused++ & 63) == 0) {
        LLAMA_LOG_WARN("pool-gate: refused %s [%.3f, %.3f) MiB on %s: it reaches the %s [%.3f, %.3f) MiB (%lld refused)\n", who, lo / 1048576.0, hi / 1048576.0,
                ggml_backend_dev_name(ap.dev), kind, rlo / 1048576.0, rhi / 1048576.0, (long long) g_gate_refused.load());
    }
    return false;
}
// Fill ids (the dump and the ledger): graph (cb) 1, CUDA scratch (sc) 2, prefetch staging (ub) 3, KV zone rows 4, expert blocks 5; the zero blocks stay 0. The first NaN the
// scheduler finds (GGML_SCHED_NAN_CHECK=1) writes pool-dump-<card>.txt into the log directory: the region map and, per MiB, the byte that fills most of it, how much of the MiB
// that is, and the writers named in it.
// --moe poison=3|4 (debug; 1 and 2 are the same as 3: the older fills, finite ids and a plain 0xFF, either failed nothing or named nothing): every region of the pool is filled on
// allocation and on every resize with NaN words, and a few words of it carry the name of the filling code, so a read of bytes nobody wrote fails at once (the NaN check names
// the tensor) and the writers named inside it say where the bytes came from. Only the reads by design stay finite: the pad window, the weight read-past guards, the zero blocks.
static bool poison_on() {
    static const bool on = [] { const char * e = moe_opt("poison"); return e && atoi(e) != 0; }();
    return on;
}
// poison=3: NaN words with the writer's name in a few of them. Pool word w (byte offset / 4) is 0xFFFFFFFF, except w % 22 == 20 and 21: the four letters of the filling code
// (GRPH graph, SCRT scratch, STGE staging, KVRW KV rows, XBLK expert block never uploaded) and the serial of that fill call (logged in the ledger as "fill"). Uppercase ASCII
// is a finite float and a finite half pair, the serial a denormal: a read of unwritten bytes is NaN at once and the letters left in the bad tensor name who filled those bytes and
// in which call (the NaN report and the pool dump count them). The phase follows the absolute offset: the same pattern wherever a region starts.
static bool poison_ids() { return poison_on(); }
// poison=4: the same fill, but the two marked words hold the 8-byte return address of the code that called pool_fill (the call site itself, no table to keep in step). The ledger's
// "fill" line logs that address next to pool_fill's own, so the library's load address cancels: call site = &pool_fill + (ra - anchor), resolved with nm / addr2line.
static bool poison_addr() {
    static const bool on = [] { const char * e = moe_opt("poison"); return e && atoi(e) >= 4; }();
    return on;
}
static const char * poison_tag(uint8_t v) {   // v: the fill ids (see the dump header)
    static const char * t[] = { "ZERO", "GRPH", "SCRT", "STGE", "KVRW", "XBLK", "OTHR", "OTHR" };
    return t[v & 7];
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void pool_fill(const moe_cache::alias_pool & ap, size_t lo, size_t hi, uint8_t v) {
    if (!poison_on() || !ap.buf || hi <= lo) { return; }
#if defined(__GNUC__) || defined(__clang__)
    const uint64_t ra = (uint64_t) (uintptr_t) __builtin_return_address(0);
#else
    const uint64_t ra = 0;
#endif
    ggml_tensor z = {};
    z.type = GGML_TYPE_I8; z.buffer = ap.buf; z.ne[1] = z.ne[2] = z.ne[3] = 1; z.nb[0] = 1;
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    static std::atomic<uint32_t> serial{0};
    const uint32_t ser = ++serial;
    uint32_t tag; memcpy(&tag, poison_tag(v), 4);
    if (led_on()) { led_line(led_card(ap), "fill", lo, hi, tr_fmt("%s serial %u ra 0x%llx anchor 0x%llx", poison_tag(v), ser, (unsigned long long) ra, (unsigned long long) (uintptr_t) &pool_fill)); }
    static thread_local std::vector<uint8_t> chunk;
    chunk.resize((size_t) 4 << 20);
    for (size_t o = lo; o < hi; ) {
        const size_t n = std::min(chunk.size(), hi - o);
        const bool addr = poison_addr();
        auto word = [&](size_t w) -> uint32_t { const size_t k = w % 22; return k == 20 ? (addr ? (uint32_t) ra : tag) : k == 21 ? (addr ? (uint32_t) (ra >> 32) : ser) : 0xFFFFFFFFu; };
        if (((o | n) & 3) == 0) {   // whole words (the usual case): 4 bytes per step
            uint32_t * c = (uint32_t *) chunk.data();
            for (size_t j = 0, w = o >> 2; j < n >> 2; ++j, ++w) { c[j] = word(w); }
        } else {
            for (size_t i = 0; i < n; ++i) { const size_t a = o + i; const uint32_t x = word(a >> 2); chunk[i] = (uint8_t) (x >> (8 * (a & 3))); }
        }
        z.data = base + o; z.ne[0] = (int64_t) n; z.nb[1] = z.nb[2] = z.nb[3] = n;
        ggml_backend_tensor_set(&z, chunk.data(), 0, n);
        o += n;
    }
}
static void led_map(const moe_cache::alias_pool & ap, const char * why) {
    if (!led_on()) { return; }
    for (const auto & r : pool_map(ap)) { led_line(led_card(ap), "map", r.lo, r.hi, tr_fmt("%s %s class %d", why, r.kind, (int) r.pcls)); }
}
// a range that must not touch the zone or the lend [0, want_b)
static void led_check_range(const moe_cache::alias_pool & ap, const char * kind, size_t lo, size_t hi, const std::string & who, bool vs_zone, bool vs_lend) {
    if (!led_on() || hi <= lo) { return; }
    if (vs_zone && ap.kv_h > 0 && lo < led_kv_end(ap) && hi > ap.kv_start) {
        led_overlap(led_card(ap), lo, hi, tr_fmt("%s %s: inside the KV zone [%.1f, %.1f) MiB", kind, who.c_str(), ap.kv_start / 1048576.0, led_kv_end(ap) / 1048576.0));
    }
    if (vs_lend && ap.want_b > 0 && lo < ap.want_b) {
        led_overlap(led_card(ap), lo, hi, tr_fmt("%s %s: inside the lend [0, %.1f) MiB", kind, who.c_str(), ap.want_b / 1048576.0));
    }
}

// the NaN hook (ggml_sched_set_nan_hook_fork, set at pool allocation with --moe poison): the region map and the per-MiB fill of every pool, to the log directory
extern "C" { GGML_API void ggml_sched_set_nan_hook_fork(void (*fn)(void)); }
static void pool_dump_all(void) {
    moe_cache * mc = g_cache;
    if (!mc) { return; }
    static std::atomic<int> once{0};
    if (once++) { return; }   // the first NaN only
    for (auto & ap : mc->alias) {
        if (!ap.buf) { continue; }
        const std::string path = moe_log_file(("pool-dump-" + std::string(led_card(ap)) + "-" + std::to_string((long long) time(nullptr)) + ".txt").c_str());
        FILE * f = fopen(path.c_str(), "w");
        if (!f) { continue; }
        const size_t size = ggml_backend_buffer_get_size(ap.buf);
        fprintf(f, "# %s pool, %zu bytes. regions (kind class lo hi bytes):\n", led_card(ap), size);
        const auto map = pool_map(ap);
        for (const auto & r : map) { fprintf(f, "region %s %d %zu %zu %zu\n", r.kind, (int) r.pcls, r.lo, r.hi, r.hi - r.lo); }
        fprintf(f, "# anchor (pool_fill) 0x%llx; call site = pool_fill + (address - anchor) when the address is a poison=4 call-site address\n", (unsigned long long) (uintptr_t) &pool_fill);
        fprintf(f, "# per MiB: index, region at its start, dominant byte, share of the MiB with that byte (poison ids: 1 graph, 2 scratch, 3 staging, 4 KV rows, 5 expert block never uploaded, 0 zero)\n");
        std::vector<uint8_t> buf((size_t) 1 << 20);
        ggml_tensor z = {};
        z.type = GGML_TYPE_I8; z.buffer = ap.buf; z.ne[1] = z.ne[2] = z.ne[3] = 1; z.nb[0] = 1;
        for (size_t o = 0; o < size; o += buf.size()) {
            const size_t n = std::min(buf.size(), size - o);
            z.data = (uint8_t *) ggml_backend_buffer_get_base(ap.buf) + o; z.ne[0] = (int64_t) n; z.nb[1] = z.nb[2] = z.nb[3] = n;
            ggml_backend_tensor_get(&z, buf.data(), 0, n);
            size_t cnt[256] = {};
            for (size_t i = 0; i < n; ++i) { cnt[buf[i]]++; }
            int best = 0; for (int v = 1; v < 256; ++v) { if (cnt[v] > cnt[best]) { best = v; } }
            const char * reg = "-"; int rc = -1;
            for (const auto & r : map) { if (o >= r.lo && o < r.hi && strcmp(r.kind, "lend") != 0) { reg = r.kind; rc = r.pcls; } }
            std::map<std::string, std::pair<size_t, uint32_t>> wr;   // poison=3: writer tag -> (words, first serial) of this MiB
            if (poison_ids()) {
                for (size_t i = 0; i + 8 <= n; i += 4) {
                    bool up = true; for (int k = 0; k < 4; ++k) { up = up && buf[i + k] >= 'A' && buf[i + k] <= 'Z'; }
                    if (!up) { continue; }
                    uint32_t sv; memcpy(&sv, &buf[i + 4], 4);
                    auto & e = wr[std::string((const char *) &buf[i], 4)]; if (e.first++ == 0) { e.second = sv; }
                }
            }
            fprintf(f, "%zu %s%d %d %.3f", o >> 20, reg, rc, best, (double) cnt[best] / (double) n);
            for (const auto & e : wr) { fprintf(f, " %s:%zu@%u", e.first.c_str(), e.second.first, e.second.second); }
            if (poison_addr()) {   // poison=4: 8-byte call-site addresses (high word 0x5000..0x7fff = a user-space address)
                std::map<uint64_t, size_t> ad;
                for (size_t i = 0; i + 8 <= n; i += 4) { uint32_t a0, a1; memcpy(&a0, &buf[i], 4); memcpy(&a1, &buf[i + 4], 4); if (a1 >= 0x5000 && a1 <= 0x7fff) { ad[((uint64_t) a1 << 32) | a0]++; } }
                for (const auto & e : ad) { fprintf(f, " 0x%llx:%zu", (unsigned long long) e.first, e.second); }
            }
            fprintf(f, "\n");
        }
        fclose(f);
        LLAMA_LOG_WARN("pool-dump: %s written\n", path.c_str());
    }
    // stop here: under a debugger break at the bad tensor, else end the process (the rest of the run would only print '!!!!' over the evidence)
    bool traced = false;
#ifndef _WIN32   // no /proc and no SIGTRAP there: the process just ends
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) { int pid = 0; if (sscanf(line, "TracerPid: %d", &pid) == 1) { traced = pid != 0; break; } }
        fclose(f);
    }
    if (traced) { LLAMA_LOG_WARN("pool-dump: debugger attached, SIGTRAP\n"); raise(SIGTRAP); return; }
#endif
    LLAMA_LOG_ERROR("pool-dump: first NaN, pool dumped, exiting (--moe poison)\n");
    fflush(nullptr);
    _exit(3);
}

void ddr_trace(moe_cache * mc, int64_t t) {
    if (tr_on()) {
        tr(TR_DDR, "DDR demand", t, -2, tr_fmt("\"GB/s\":%.1f", mc->ddr_mbs / 1e3));
    }
}

void moe_phase_cb(int busy, size_t bytes, void * ud) {
    auto * mc = (moe_cache *) ud;
    const int64_t t = ggml_time_us();
    if (busy) {
        mc->busy_t0 = t;
        mc->cpu_busy = true;
        mc->cpu_bytes += (int64_t) bytes;
        mc->cpu_bytes_phase = (int64_t) bytes;
        mc->cpu_mbs_on = (int64_t) ((mc->cpu_gbs_meas > 0 ? mc->cpu_gbs_meas : knobs().cpu_gbs) * 1e3);
        mc->ddr_mbs += mc->cpu_mbs_on;
    } else {
        mc->cpu_busy = false;
        mc->busy_us += t - mc->busy_t0;
        mc->busy_n++;
        mc->ddr_mbs -= mc->cpu_mbs_on;
        const int64_t dt = t - mc->busy_t0;
        if (mc->cpu_bytes_phase > 0 && dt > 0) {
            const double gbs = (double) mc->cpu_bytes_phase / dt / 1e3;
            mc->cpu_gbs_meas = mc->cpu_gbs_meas > 0 ? 0.95*mc->cpu_gbs_meas + 0.05*gbs : gbs;
        }
        if (tr_on()) {
            tr(TR_CPU, "cpu experts", mc->busy_t0, dt, tr_fmt("\"MB\":%.1f", mc->cpu_bytes_phase / 1e6));
        }
    }
    ddr_trace(mc, t);
}

// --moe tokdump: one MoE layer of a decode step (from the observer, mc->mtx held): the experts its tokens picked, at the first and the last MoE layer
// their input too; the step's tokens go to the file after its last MoE layer
static void tokdump_layer(moe_cache * mc, const layer_state & ls, size_t li, const ggml_tensor * ids, const ggml_tensor * op) {
    auto & d = g_tokdump;
    if (d.on < 0) {
        const char * p = moe_opt("tokdump");
        d.f  = p && *p ? fopen(p, "ab") : nullptr;   // appended: a server started again adds its requests (run.py tokpred numbers them anew)
        d.on = d.f != nullptr;
        if (p && *p && d.f == nullptr) { LLAMA_LOG_WARN("moe-cache: tokdump: cannot write %s\n", p); }
    }
    const ggml_tensor * x = op ? op->src[1] : nullptr;   // the experts' input: [n_embd, 1, tokens] (or [n_embd, tokens])
    const int64_t n_tok = ids->ne[1], n_ids = ids->ne[0];
    if (d.on == 0 || x == nullptr || x->type != GGML_TYPE_F32 || x->data == nullptr || (x->ne[2] != n_tok && x->ne[1] != n_tok)) { return; }
    if (d.n_embd == 0) {
        d.n_embd = (uint32_t) x->ne[0];
        d.n_lay  = (uint32_t) mc->layers.size();
        d.n_used = (uint32_t) n_ids;
        const uint32_t h[4] = { d.n_embd, d.n_lay, d.n_used, (uint32_t) ls.expert_slot.size() };
        fseek(d.f, 0, SEEK_END);
        if (ftell(d.f) == 0) {   // a new file: the header first
            fwrite("MOETOKD1", 1, 8, d.f);
            fwrite(h, sizeof(uint32_t), 4, d.f);
        }
    }
    if (x->ne[0] != (int64_t) d.n_embd || n_ids != (int64_t) d.n_used || li >= d.n_lay) { return; }
    const size_t ne = d.n_embd, nc = (size_t) d.n_lay * d.n_used;
    if (li == 0 || d.n_tok != n_tok) {   // a new step (not by mc->n_steps: it moves on after the first MoE layer, and the first layer's key was lost)
        d.step  = mc->n_steps;
        d.n_tok = n_tok;
        d.k0.assign((size_t) n_tok * ne, 0);
        d.kl.assign((size_t) n_tok * ne, 0);
        d.ids.assign((size_t) n_tok * nc, -1);
        d.hit.assign((size_t) n_tok * nc, 0);
    }
    const size_t xs = x->ne[2] == n_tok ? x->nb[2] : x->nb[1];
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * row = (const float *) ((const char *) x->data + t * xs);
        if (li == 0)           { ggml_fp32_to_fp16_row(row, d.k0.data() + (size_t) t * ne, (int64_t) ne); }
        if (li + 1 == d.n_lay) { ggml_fp32_to_fp16_row(row, d.kl.data() + (size_t) t * ne, (int64_t) ne); }
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + i * ids->nb[0]);
            const size_t  c  = (size_t) t * nc + li * d.n_used + (size_t) i;
            d.ids[c] = (int16_t) id;
            d.hit[c] = id >= 0 && id < (int32_t) ls.expert_slot.size() && ls.expert_slot[id] >= 0;
        }
    }
    if (li + 1 == d.n_lay) {   // the step's last MoE layer: its tokens to the file
        for (int64_t t = 0; t < n_tok; ++t) {
            fwrite(&d.req, sizeof(uint32_t), 1, d.f);
            fwrite(d.k0.data() + (size_t) t * ne, sizeof(ggml_fp16_t), ne, d.f);
            fwrite(d.kl.data() + (size_t) t * ne, sizeof(ggml_fp16_t), ne, d.f);
            fwrite(d.ids.data() + (size_t) t * nc, sizeof(int16_t), nc, d.f);
            fwrite(d.hit.data() + (size_t) t * nc, 1, nc, d.f);
        }
        fflush(d.f);
        d.n_tok = 0;
    }
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, const struct ggml_tensor * op, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    // big-batch prefill ids are observed too: they warm the cache before decode
    // (hits aren't counted there, the cache graph isn't built for big batches)
    const bool prefill = n_tokens > llama_moe_cache_max_batch_fork();

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->last_prefill = prefill;
    if (prefill) { mc->boot_done = true; }   // a big prompt batch fills the cache by adoption: the strict entry of the boot is for a cold cache only
    mc->last_tokens  = n_tokens;
    if (get_policy() == policy::rel && op && op->src[6] && op->src[6]->type == GGML_TYPE_F32 && op->src[6]->data && op->src[6]->ne[0] == (int64_t) ls->relevance.size()) {
        static const float d = [] { const char * v = moe_opt("rel-decay"); return v ? (float) atof(v) : 0.5f; }();   // 0.5: halve the old relevance, add the current one
        const ggml_tensor * pr = op->src[6];
        for (int64_t t = 0; t < std::min<int64_t>(n_tokens, pr->ne[1]); ++t) {
            const float * p = (const float *) ((const char *) pr->data + t*pr->nb[1]);
            for (size_t e = 0; e < ls->relevance.size(); ++e) { ls->relevance[e] = d*ls->relevance[e] + p[e]; }
        }
        mc->pr[0]++;   // observed layer-steps with router probabilities
    }
    static const bool count_prompt = [] { const char * v = moe_opt("count-prompt"); return !v || atoi(v) != 0; }();   // --moe count-prompt=0: prompt batches warm the window but not the lifetime counts
    for (int64_t t = 0; t < n_tokens; ++t) {
        if ((int32_t) ls->recent.size() >= mc->window) {
            for (int32_t old : ls->recent.front()) {
                ls->win_count[old]--;
            }
            ls->recent.pop_front();
        }
        ls->recent.emplace_back();
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            ls->expert_count[id]++;
            ls->win_count[id]++;
            ls->expert_last[id] = ++mc->clock;
            ls->recent.back().push_back(id);
            if (g_exptok.n > 0 && n_tokens == (int64_t) g_exptok.batch.size() && (uint32_t) il < g_exptok.n_layers && (uint32_t) id < g_exptok.n_exp) {
                const size_t c = (size_t) il * g_exptok.n_exp + (size_t) id;   // --moe expids: this token to this expert's ring
                g_exptok.ring[c * (size_t) g_exptok.n + g_exptok.seen[c] % (uint32_t) g_exptok.n] = g_exptok.batch[t];
                g_exptok.seen[c]++;
                g_exptok.dirty = true;
            }
            ls->topic_max = std::max(ls->topic_max, ++ls->topic_cnt[id]);   // the request's own map (hot maps per topic): prompt and answer, whatever the lifetime counts take
            if (!prefill && g_topic_answer.load(std::memory_order_relaxed)) {   // what its record stores: the answer's uses, per phase (not a short prompt's tokens)
                (g_topic_thinking.load(std::memory_order_relaxed) ? ls->topic_think : ls->topic_dec)[id]++;
            }
            if (!prefill || count_prompt) {   // the lifetime counts (eviction score, saved hot map): the answer's experts, not the question's, with count-prompt=0
                ls->glob_max = std::max(ls->glob_max, ++ls->glob_count[id]);
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                if (!prefill && ls->is_stream[slot] &&
                        std::find(ls->stream_hit.begin(), ls->stream_hit.end(), id) == ls->stream_hit.end()) {
                    ls->stream_hit.push_back(id);
                    if (ls->prefetched[id]) {
                        mc->st_hit[ls->link]++;
                    }
                }
                ls->n_hit += !prefill;
                ls->up_hits[id] += !prefill;
                ls->slot_last_use[slot] = ++mc->clock;
                if (ls->prefetched[id] && !prefill) {
                    mc->n_pred_used++;
                    ls->prefetched[id] = 0;
                }
            } else {
                ls->n_miss += !prefill;
                if (!prefill && ls->ghost_flag[id]) { mc->pr[12]++; ls->ghost_flag[id] = 0; }   // evicted recently and wanted again
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }

    // router prediction for the next layers, off this thread (the other CPU threads wait at a barrier)
    const size_t li = ls - mc->layers.data();
    if (!prefill && g_tokdump.on != 0) {
        tokdump_layer(mc, *ls, li, ids, op);   // --moe tokdump (experiment)
    }
    if (!prefill && tr_on()) {
        int n_miss = 0, n_str = 0;
        std::string idl;
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + i*ids->nb[0]);
            const int32_t sl = ls->expert_slot[id];
            n_miss += sl < 0;
            n_str  += sl >= 0 && ls->is_stream[sl];
            idl += tr_fmt("%s%d%s", i ? " " : "", id, sl < 0 ? "-" : ls->is_stream[sl] ? "+" : "");
        }
        tr(TR_CPU, tr_fmt("router L%zu: %d miss", li, n_miss), ggml_time_us(), -1,
                tr_fmt("\"ids (- miss, + streamed)\":\"%s\",\"streamed hits\":%d", idl.c_str(), n_str));
    }
    const ggml_tensor * pred = op ? op->src[4] : nullptr;
    // (sampled: 1 token in 16 -- this runs while the other CPU threads wait at the barrier)
    if (!prefill && mc->pred_m > 0 && mc->n_steps % 16 == 0) {
        // this layer's real choice (token 0) scores the predictions made for it at earlier layers
        const auto & b = mc->pred_b[li];
        const int64_t ne = (int64_t) ls->expert_slot.size();
        std::vector<int32_t> idx(ne);
        for (int k = 0; k < 8 && (size_t) k + 1 <= li; ++k) {
            const size_t src = li - 1 - k;
            if (mc->last_pred_step[src] != mc->n_steps || mc->last_pred_blk[src] <= k) {
                continue;
            }
            const float * lg = mc->last_pred[src].data() + k*ne;
            auto sc = [&](int32_t e) { return b.empty() ? lg[e] : 1.0f/(1.0f + expf(-lg[e])) + b[e]; };
            for (int32_t e = 0; e < ne; ++e) { idx[e] = e; }
            std::partial_sort(idx.begin(), idx.begin() + n_ids, idx.end(), [&](int32_t x, int32_t y) { return sc(x) > sc(y); });
            uint64_t hit = 0;
            for (int64_t i = 0; i < n_ids; ++i) {
                for (int64_t j = 0; j < n_ids; ++j) {
                    hit += idx[i] == *(const int32_t *) ((const char *) ids->data + j*ids->nb[0]);
                }
            }
            mc->acc_hit[k] += hit; mc->acc_tot[k] += n_ids;
            mc->win_hit[k] += hit; mc->win_tot[k] += n_ids;
            if (k == 0) {
                // uncached experts ranked by predicted score: what an L3 / stream prefetch of M experts would fetch
                std::vector<int32_t> unc;
                for (int32_t e = 0; e < ne; ++e) {
                    if (ls->expert_slot[e] < 0) { unc.push_back(e); }
                }
                const int32_t m = std::min<int32_t>(3, (int32_t) unc.size());
                std::partial_sort(unc.begin(), unc.begin() + m, unc.end(), [&](int32_t x, int32_t y) { return sc(x) > sc(y); });
                auto used = [&](int32_t e) {
                    for (int64_t j = 0; j < n_ids; ++j) {
                        if (*(const int32_t *) ((const char *) ids->data + j*ids->nb[0]) == e) { return true; }
                    }
                    return false;
                };
                uint64_t n_miss = 0;
                for (int64_t j = 0; j < n_ids; ++j) {
                    n_miss += ls->expert_slot[*(const int32_t *) ((const char *) ids->data + j*ids->nb[0])] < 0;
                }
                mc->pm_miss += n_miss;
                uint64_t h = 0;
                for (int32_t M = 1; M <= 3; ++M) {
                    if (M <= m) { h += used(unc[M - 1]); }
                    mc->pm_pick[M - 1] += std::min<int32_t>(M, m);
                    mc->pm_hit[M - 1]  += h;
                    mc->pm_cov[M - 1]  += h;
                }
            }
        }
    }
    if (!prefill && mc->pred_m > 0 && pred && pred->type == GGML_TYPE_F32 && pred->ne[1] == n_tokens && li + 1 < mc->layers.size()) {
        const int64_t ne    = (int64_t) mc->layers[li + 1].expert_slot.size();
        const int64_t n_blk = pred->ne[0] / ne;
        moe_cache::pred_req r { li, mc->n_steps, n_tokens, n_blk, std::vector<float>(n_blk*ne*n_tokens) };
        for (int64_t t = 0; t < n_tokens; ++t) {
            memcpy(r.x.data() + t*n_blk*ne, (const char *) pred->data + t*pred->nb[1], n_blk*ne*sizeof(float));
        }
        mc->last_pred[li].assign(r.x.begin(), r.x.begin() + n_blk*ne);
        mc->last_pred_step[li] = mc->n_steps;
        mc->last_pred_blk[li]  = n_blk;
        mc->last_obs = (int64_t) li;
        {
            const int64_t t = ggml_time_us();
            if (mc->layer_dt.size() != mc->layers.size()) {
                mc->layer_dt.assign(mc->layers.size(), 0.0);
            }
            if (mc->clk_step == mc->n_steps && mc->clk_li == (int64_t) li - 1 && li > 0) {
                double & d = mc->layer_dt[li - 1];
                const double dt = (double) (t - mc->clk_t);
                d = d > 0 ? 0.95*d + 0.05*dt : dt;
            }
            mc->clk_t = t; mc->clk_li = (int64_t) li; mc->clk_step = mc->n_steps;
        }
        if (knobs().l3pf > 0 && !mc->l3_thr.empty()) {
            // top-M predicted experts of layer li+1 that aren't cached (block 0 = one layer ahead)
            auto & tl = mc->layers[li + 1];
            const auto & b = mc->pred_b[li + 1];
            const float * lg = r.x.data();
            auto sc = [&](int32_t e) { return b.empty() ? lg[e] : 1.0f/(1.0f + expf(-lg[e])) + b[e]; };
            std::vector<int32_t> unc;
            for (int32_t e = 0; e < (int32_t) ne; ++e) {
                if (tl.expert_slot[e] < 0 && !tl.queued[e]) { unc.push_back(e); }
            }
            const int32_t m = std::min<int32_t>((int32_t) knobs().l3pf, (int32_t) unc.size());
            std::partial_sort(unc.begin(), unc.begin() + m, unc.end(), [&](int32_t x, int32_t y) { return sc(x) > sc(y); });
            unc.resize(m);
            {
                std::lock_guard<std::mutex> lk(mc->l3_mtx);
                mc->l3_job.li = li + 1;
                mc->l3_job.ex = std::move(unc);
                mc->l3_job.gen++;
            }
            mc->l3_cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> plk(mc->pmtx);
            mc->preq.push_back(std::move(r));
        }
        mc->pcv.notify_one();
    }
}

// publish a finished upload (mtx held, no graph reading this layer's table)
void publish_job(moe_cache * mc, const upload_job & j) {
    auto & ls = mc->layers[j.layer_idx];
    if (j.skipped) {
        ls.slot_in_flight[j.slot] = false;
        ls.queued[j.expert]       = 0;
        return;
    }
    ls.slot_expert[j.slot]     = j.expert;
    ls.expert_slot[j.expert]   = j.slot;
    ls.slot_last_use[j.slot]   = ++mc->clock;
    ls.slot_in_flight[j.slot]  = false;
    ls.queued[j.expert]        = 0;
    ls.prefetched[j.expert]    = j.urgent;
    set_table_entry(ls.pub, j.expert, j.slot);
    ls.cached_since[j.expert] = mc->n_steps + 1;
    ls.up_t[j.expert]    = j.t_done;
    ls.up_cost[j.expert] = j.read_us + j.copy_us;
    ls.up_hits[j.expert] = 0;
}

// an expert leaves VRAM: did its upload pay back (decode hits x CPU eval time saved >= upload time)?
void note_evict(moe_cache * mc, layer_state & ls, int32_t e) {
    mc->n_evictions++;
    if (!ls.ghost_flag.empty()) {
        ls.ghost_flag[e] = 1; ls.ghost_q.push_back(e);
        while ((int32_t) ls.ghost_q.size() > std::max(1, ls.n_cache)) { ls.ghost_flag[ls.ghost_q.front()] = 0; ls.ghost_q.pop_front(); }
    }
    if (ls.up_t[e] == 0) {
        return;
    }
    mc->pr[14] += ls.up_hits[e] == 0;
    const int k = ls.link;
    mc->ev_n[k]++;
    mc->ev_hits[k]    += ls.up_hits[e];
    const double cpu_us = (double) (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]) / (knobs().cpu_gbs * 1e3);
    mc->ev_paid[k]    += ls.up_hits[e] * cpu_us >= ls.up_cost[e];
    mc->pr[13]        += !(ls.up_hits[e] * cpu_us >= ls.up_cost[e]);
    mc->ev_life_us[k] += (double) (ggml_time_us() - ls.up_t[e]);
    mc->ev_cost_us[k] += ls.up_cost[e];
    ls.up_t[e] = 0;
}

// LLAMA_MOE_CACHE_CTL: apply the file's NAME=value lines when it changed, report the segment that just ended
void ctl_poll(moe_cache * mc) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(mc->ctl, ec);
    if (ec || t == mc->ctl_mtime) {
        return;
    }
    mc->ctl_mtime = t;
    knobs_t k = knobs();
    std::string line, applied;
    std::ifstream f(mc->ctl);
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string name = line.substr(0, eq);
        if (name.rfind("LLAMA_MOE_CACHE_", 0) == 0) {
            name = name.substr(16);
        }
        if (knob_set(k, name, atof(line.c_str() + eq + 1))) {
            applied += " " + line;
        }
    }
    // uploads still in VRAM count too (evictions alone would bias pay-back low: the good ones stay)
    for (auto & ls : mc->layers) {
        for (int32_t e = 0; e < (int32_t) ls.up_t.size(); ++e) {
            if (ls.up_t[e] && ls.expert_slot[e] >= 0) {
                note_evict(mc, ls, e); // settled here with its hits so far; not tracked in the next segment
            }
        }
    }
    uint64_t h = 0, m = 0;
    for (const auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
    const uint64_t dh = h - mc->seg_hit, dm = m - mc->seg_miss;
    std::string per_link;
    for (int i = 0; i < mc->n_links; ++i) {
        per_link += tr_fmt(" link %d: %llu (%.1f hits, paid %.0f%%)", i, (unsigned long long) mc->ev_n[i],
            mc->ev_n[i] ? (double) mc->ev_hits[i]/mc->ev_n[i] : 0.0, mc->ev_n[i] ? 100.0*mc->ev_paid[i]/mc->ev_n[i] : 0.0);
    }
    LLAMA_LOG_WARN("moe-cache: ctl: segment of %" PRIu64 " steps: hit-rate %.1f%%, %" PRIu64 " uploads, uploads settled%s | now%s\n",
            mc->n_steps - mc->seg_steps, dh + dm ? 100.0*dh/(dh + dm) : 0.0, mc->n_uploads - mc->seg_up, per_link.c_str(), applied.c_str());
    mc->seg_steps = mc->n_steps; mc->seg_hit = h; mc->seg_miss = m; mc->seg_up = mc->n_uploads;
    for (int i = 0; i < mc->n_links; ++i) {
        mc->ev_n[i] = mc->ev_hits[i] = mc->ev_paid[i] = 0;
        mc->ev_life_us[i] = mc->ev_cost_us[i] = 0;
    }
    const bool resort = k.hot_frac != knobs().hot_frac || k.sticky != knobs().sticky;
    knobs() = k;
    for (int i = 0; i < mc->n_links; ++i) { // new settings: fresh measurement window (the learned lead / M are kept)
        mc->st_up[i] = mc->st_late[i] = mc->st_hit[i] = 0;
    }
    if (applied.find(" TRACE=") != std::string::npos && !g_tr.prefix.empty()) {
        g_tr.skip = (int64_t) k.trace_after;
        g_tr.left = (int64_t) k.trace;
    }
    if (resort) {
        for (auto & ls : mc->layers) {
            refresh_sticky(ls);
        }
    }
}

void readahead(const layer_state & ls, int32_t expert) {
#ifndef _WIN32
    static const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
        const uintptr_t a   = (uintptr_t) t->data + (size_t) expert*t->nb[2];
        const uintptr_t beg = a & ~(uintptr_t) (pg - 1);
        madvise((void *) beg, a + t->nb[2] - beg, MADV_WILLNEED);
    }
#else
    GGML_UNUSED(ls); GGML_UNUSED(expert);
#endif
}

constexpr uint32_t PRED_MAGIC = 0x3252504d; // "MPR2"

// learned predictors: per layer, the stacked [n_embd, n_expert * lookahead] fp16 matrix
void pred_save(const moe_cache * mc) {
    if (mc->pred_file.empty() || mc->n_train == 0) {
        return;
    }
    const std::string tmp = mc->pred_file + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        return;
    }
    const uint32_t hdr[3] = { PRED_MAGIC, (uint32_t) mc->layers.size(), (uint32_t) mc->pred_ahead };
    bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1;
    std::vector<uint8_t> buf;
    for (const auto & ls : mc->layers) {
        const ggml_tensor * t = ls.pub.pred_all;
        const uint64_t n = t ? (uint64_t) ggml_nbytes(t) : 0;
        ok = ok && fwrite(&n, sizeof(n), 1, f) == 1;
        if (ok && n) {
            buf.resize(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n);
            ok = fwrite(buf.data(), 1, n, f) == n;
        }
    }
    ok = fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, mc->pred_file, ec);
        LLAMA_LOG_INFO("moe-cache: saved the router predictors (%" PRIu64 " training steps) to %s\n", mc->n_train, mc->pred_file.c_str());
    } else {
        std::filesystem::remove(tmp, ec);
    }
}

// Q8_0 prediction copy <- fp16 master (at start and after loading saved weights; the graph re-quantizes after updates)
void pred_requant(llama_moe_cache_layer_fork & pub) {
    if (!pub.pred_q || !pub.pred_all) {
        return;
    }
    const int64_t n_per_row = pub.pred_all->ne[0], nrows = pub.pred_all->ne[1];
    std::vector<ggml_fp16_t> h(n_per_row * nrows);
    ggml_backend_tensor_get(pub.pred_all, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
    std::vector<float> f(h.size());
    ggml_fp16_to_fp32_row(h.data(), f.data(), (int64_t) f.size());
    std::vector<uint8_t> q(ggml_nbytes(pub.pred_q));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    ggml_backend_tensor_set(pub.pred_q, q.data(), 0, q.size());
}

bool pred_load(const moe_cache * mc) {
    FILE * f = mc->pred_file.empty() ? nullptr : fopen(mc->pred_file.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t hdr[3] = {};
    bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == PRED_MAGIC && hdr[1] == mc->layers.size() && hdr[2] == (uint32_t) mc->pred_ahead;
    std::vector<std::vector<uint8_t>> data;
    for (const auto & ls : mc->layers) {
        const ggml_tensor * t = ls.pub.pred_all;
        uint64_t n = 0;
        ok = ok && fread(&n, sizeof(n), 1, f) == 1 && n == (t ? (uint64_t) ggml_nbytes(t) : 0);
        data.emplace_back(n);
        ok = ok && fread(data.back().data(), 1, n, f) == n;
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_WARN("moe-cache: ignoring router predictors %s (other model or lookahead)\n", mc->pred_file.c_str());
        return false;
    }
    size_t i = 0;
    for (const auto & ls : mc->layers) {
        if (ls.pub.pred_all && !data[i].empty()) {
            ggml_backend_tensor_set(ls.pub.pred_all, data[i].data(), 0, data[i].size());
        }
        ++i;
    }
    return true;
}

// L3 domains from sysfs (cpuN/cache/index3/shared_cpu_list), in order of first CPU; for each the highest CPU that is
// the first SMT thread of its core. Linux only; elsewhere (or on failure) one domain, no pinning.
// measured mean layer time (layer clock), else token time / layers, else 1 ms before anything was measured
double mean_layer_us(const moe_cache * mc) {
    double sum = 0; int n = 0;
    for (double d : mc->layer_dt) { if (d > 0) { sum += d; n++; } }
    if (n > 0) { return sum / n; }
    if (mc->step_us_idle > 0 && !mc->layers.empty()) { return mc->step_us_idle / (double) mc->layers.size(); }
    return 1000.0;
}

// measured mean CPU expert phase (phase callback), else a quarter layer
double mean_cpu_phase_us(const moe_cache * mc) {
    const int64_t n = mc->busy_n;
    return n > 16 ? (double) mc->busy_us / (double) n : mean_layer_us(mc) / 4;
}

// resource saturator, startup probe (~0.5 s): measured instead of assumed. Reads the model's expert weights (the
// same memory decode reads) with CPU threads, DMA-copies them into a still-empty cache slot of every GPU, alone and
// all at once. Sets CPU_GBS (CPU read rate), DDR_GBS (CPU + all links together) unless given, and seeds each link's
// GB/s. LLAMA_MOE_CACHE_PROBE=0 skips it.
// Helper threads (probe, upload workers, predictor) run anywhere the process may: on Linux a new thread inherits its
// creator's affinity, and with strict CPU placement ggml pins the main thread to one compute core. Measured with
// LLAMA_AUTO_PLACE=1 before this: probe CPU 26 instead of 39 GB/s, uploads 12 / 3.4 instead of 24 / 5.2 GB/s.
// The mask is taken at library load, before any pinning (a user's taskset still applies). Windows threads start with
// the process mask; other platforms have no thread pinning.
#if defined(__linux__)
static cpu_set_t g_proc_mask = [] {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0) {
        CPU_ZERO(&cs);
    }
    return cs;
}();
#endif

#if defined(__linux__)
static const cpu_set_t * helper_cpu_set_mc() {
    static const cpu_set_t cs = [] {
        cpu_set_t c;
        CPU_ZERO(&c);
        if (const char * e = getenv("LLAMA_HELPER_CPUS")) {   // see common: a user CPU mask -C, plus the SMT siblings of its cores
            for (const char * p = e; *p; ) {
                char * end = nullptr;
                const long a = strtol(p, &end, 10);
                if (end == p) { break; }
                long b = a;
                if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); }
                for (long i = a; i <= b && i < CPU_SETSIZE; ++i) { CPU_SET((int) i, &c); }
                p = *end == ',' ? end + 1 : end;
                if (end == p && *end != ',') { break; }
            }
        }
        return c;
    }();
    return CPU_COUNT(&cs) > 0 ? &cs : nullptr;
}
#endif

void unpin_helper() {
#if defined(__linux__)
    if (const cpu_set_t * hs = helper_cpu_set_mc()) {
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), hs);
        return;
    }
    if (CPU_COUNT(&g_proc_mask) > 0) {
        pthread_setaffinity_np(pthread_self(), sizeof(g_proc_mask), &g_proc_mask);
    }
#endif
}

std::vector<std::vector<int>> l3_domain_cores();

// NUMA map and per-node read rates (log only, no decision yet): which node each GPU hangs on (sysfs numa_node of its PCI id), and the RAM read rate of
// every reader node against every buffer node. Each node's buffer is first-touched by a thread pinned to that node (no mempolicy call: containers refuse
// those), the readers are pinned to one physical core each. The local diagonal is what a GPU's node can feed it; the off-diagonal what a cross-node read costs
void numa_probe(moe_cache * mc) {
#if defined(__linux__)
    const char * ne = moe_opt("numa-probe");
    if (ne && ne[0] == '0') { return; }
    auto read = [](const std::string & path) { std::ifstream f(path); std::string v; std::getline(f, v); return v; };
    auto parse = [](const std::string & list) {
        std::vector<int> r;
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            const std::string part = list.substr(i, j == std::string::npos ? std::string::npos : j - i);
            const size_t dash = part.find('-');
            if (!part.empty()) {
                const int a = atoi(part.c_str()), b = dash == std::string::npos ? a : atoi(part.c_str() + dash + 1);
                for (int k = a; k <= b; ++k) { r.push_back(k); }
            }
            if (j == std::string::npos) { break; }
            i = j + 1;
        }
        return r;
    };
    std::vector<int> phys;   // first SMT thread of every physical core
    for (const auto & d : l3_domain_cores()) { phys.insert(phys.end(), d.begin(), d.end()); }
    std::vector<std::vector<int>> node_cores;   // physical cores of each node that has CPUs
    std::vector<int> node_id;
    for (int n = 0; n < 64; ++n) {
        const std::string l = read("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (l.empty()) { if (n > 0 && node_cores.empty()) { break; } continue; }
        std::vector<int> c;
        for (int cpu : parse(l)) { if (std::find(phys.begin(), phys.end(), cpu) != phys.end()) { c.push_back(cpu); } }
        if (!c.empty()) { node_cores.push_back(c); node_id.push_back(n); }
    }
    const int N = (int) node_cores.size();
    if (N < 2) { return; }
    std::string gpus;
    for (int k = 0; k < mc->n_links; ++k) {
        ggml_backend_dev_props pr = {};
        if (mc->link_dev[k]) { ggml_backend_dev_get_props(mc->link_dev[k], &pr); }
        std::string id = pr.device_id ? pr.device_id : "";
        for (auto & ch : id) { ch = (char) tolower((unsigned char) ch); }
        const std::string nn = id.empty() ? "" : read("/sys/bus/pci/devices/" + id + "/numa_node");
        gpus += tr_fmt(" link %d -> node %s", k, nn.empty() || nn[0] == '-' ? "?" : nn.c_str());
    }
    const size_t buf_n = (size_t) 256 << 20;
    std::vector<uint8_t *> buf(N, nullptr);
    auto pin = [](int cpu) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    };
    for (int n = 0; n < N; ++n) {   // first touch by a thread of the node
        std::thread([&, n] {
            pin(node_cores[n][0]);
            buf[n] = (uint8_t *) malloc(buf_n);
            if (buf[n]) { memset(buf[n], 1, buf_n); }
        }).join();
        if (!buf[n]) { for (auto * b : buf) { free(b); } return; }
    }
    auto cell = [&](int r, int b, double secs) {   // GB/s: the cores of node r read the buffer of node b
        std::atomic<bool> stop{false};
        const int nt = (int) node_cores[r].size();
        std::vector<double> got(nt, 0.0);
        std::vector<std::thread> th;
        for (int t = 0; t < nt; ++t) {
            th.emplace_back([&, t] {
                pin(node_cores[r][t]);
                uint64_t x = 0;
                size_t o = buf_n / nt * t;
                const size_t lim = buf_n / nt * (t + 1);
                while (!stop) {   // 1 MiB per check, wrapping inside this thread's slice
                    const size_t end = std::min(o + ((size_t) 1 << 20), lim);
                    for (; o < end; o += 64) { x += *(const volatile uint64_t *) (buf[b] + o); }
                    got[t] += (double) ((size_t) 1 << 20);
                    if (o >= lim) { o = buf_n / nt * t; }
                }
                got[t] += (double) (x & 1);
            });
        }
        const int64_t t0 = ggml_time_us();
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        stop = true;
        for (auto & x : th) { x.join(); }
        const double dt = (double) (ggml_time_us() - t0);
        double g = 0;
        for (double v : got) { g += v / dt / 1e3; }
        return g;
    };
    std::string rs;
    for (int r = 0; r < N; ++r) {
        rs += tr_fmt("%snode %d (%zu cores):", r ? "; " : "", node_id[r], node_cores[r].size());
        for (int b = 0; b < N; ++b) {
            if (N > 4 && b != r && b != (r + 1) % N) { continue; }   // many nodes: local and the next one only, to keep the start short
            rs += tr_fmt(" %d:%.1f", node_id[b], cell(r, b, 0.12));
        }
    }
    for (auto * b : buf) { free(b); }
    LLAMA_LOG_WARN("moe-cache: numa: %d nodes;%s; read GB/s (reader node, buffer node:rate) %s\n", N, gpus.c_str(), rs.c_str());
#else
    (void) mc;
#endif
}

void resource_probe(moe_cache * mc) {
    const char * pe = moe_opt("probe");
    if ((pe && pe[0] == '0') || mc->layers.empty()) {
        return;
    }
    // physical cores (first SMT thread of each): the ramp's upper bound (Linux sysfs / Windows; else logical CPUs)
    int n_phys = 0;
    for (const auto & d : l3_domain_cores()) {
        n_phys += (int) d.size();
    }
    // reader threads are spread over the L3 domains (one core per domain in turn): on a NUMA host with several memory controllers (a 4-node 64-core EPYC)
    // unpinned readers share a few of them and the ramp flattens at a fraction of the RAM rate (blg: 27 GB/s probed, 52-72 GB/s streamed by all cores)
    std::vector<int> spread;
    {
        const auto doms = l3_domain_cores();
        for (size_t k = 0; ; ++k) {
            bool any = false;
            for (const auto & d : doms) { if (k < d.size()) { spread.push_back(d[k]); any = true; } }
            if (!any) { break; }
        }
    }
    // source chunks: whole experts across layers, far more than any L3
    struct chunk { const uint8_t * p; size_t n; };
    std::vector<chunk> src;
    for (auto & ls : mc->layers) {
        for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
            if (!ggml_backend_buffer_is_host(t->buffer)) { return; }
            for (int64_t e = 0; e < t->ne[2]; e += 7) {
                src.push_back({ (const uint8_t *) t->data + (size_t) e*t->nb[2], t->nb[2] });
            }
        }
    }
    // one target per link: slot 0 of the first layer on it (empty now; the preload fills it after the probe)
    std::vector<int> link_layer(mc->n_links, -1);
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        if (link_layer[mc->layers[li].link] < 0 && mc->layers[li].pub.n_slots > 0) { link_layer[mc->layers[li].link] = (int) li; }
    }
    auto run = [&](int n_thr, bool dma, double secs, double & cpu_gbs, std::vector<double> & link_gbs) {
        std::atomic<bool> stop{false};
        std::atomic<size_t> next{0};
        std::vector<double> got(n_thr + mc->n_links, 0.0);
        std::vector<std::thread> th;
        {
            for (int t = 0; t < n_thr; ++t) {
                th.emplace_back([&, t] {
                    unpin_helper();
#if defined(__linux__)
                    if (!spread.empty()) {
                        cpu_set_t cs;
                        CPU_ZERO(&cs);
                        CPU_SET(spread[t % spread.size()], &cs);
                        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                    }
#endif
                    uint64_t x = 0;
                    while (!stop) {
                        const chunk & c = src[next.fetch_add(1) % src.size()];
                        for (size_t o = 0; o < c.n; o += 64) { x += *(const volatile uint64_t *) (c.p + o); }
                        got[t] += (double) c.n;
                    }
                    got[t] += (double) (x & 1);
                });
            }
        }
        if (dma) {
            for (int k = 0; k < mc->n_links; ++k) {
                if (link_layer[k] < 0) { continue; }
                th.emplace_back([&, k] {
                    unpin_helper();
                    auto & ls = mc->layers[link_layer[k]];
                    size_t i = (size_t) k * 977;
                    while (!stop) {
                        const ggml_tensor * t = ls.pub.up_src;
                        const int64_t e = (int64_t) (i++ % (size_t) t->ne[2]);
                        ggml_backend_tensor_set(ls.pub.up_c, (const uint8_t *) t->data + (size_t) e*t->nb[2], 0, t->nb[2]);
                        got[n_thr + k] += (double) t->nb[2];
                    }
                });
            }
        }
        const int64_t t0 = ggml_time_us();
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        stop = true;
        for (auto & x : th) { x.join(); }
        const double dt = (double) (ggml_time_us() - t0);
        cpu_gbs = 0;
        for (int t = 0; t < n_thr; ++t) { cpu_gbs += got[t] / dt / 1e3; }
        link_gbs.assign(mc->n_links, 0.0);
        for (int k = 0; k < mc->n_links; ++k) { link_gbs[k] = got[n_thr + k] / dt / 1e3; }
    };
    // step 0 runs the links alone; then the CPU readers are measured from the top down: all physical cores first, then half, a quarter, ... until the rate
    // falls under 95% of the best, then a bisection between the last count below and the first at it. Measuring from the top cannot stop early on a
    // plateau of a few threads (a ramp up from 1 did on a 4-node EPYC host); peak CPU alone = the CPU's RAM read rate, the smallest count within 95% of it =
    // cores that saturate RAM; peak CPU + links = the RAM budget
    double cpu_alone = 0, total = 0;
    std::vector<double> link_alone, link_mixed, tmp;
    std::map<int, std::pair<double, double>> meas;   // reader threads -> (CPU alone, CPU + links)
    const int n_max = std::max(1, n_phys > 0 ? n_phys : (int) std::thread::hardware_concurrency());
    run(0, true, 0.15, cpu_alone, link_alone);
    cpu_alone = 0;
    auto measure = [&](int nt) {
        if (meas.count(nt)) { return; }
        double g = 0, gm = 0;
        run(nt, false, 0.08, g, tmp);
        run(nt, true, 0.08, gm, link_mixed);
        for (double l : link_mixed) { gm += l; }
        meas[nt] = { g, gm };
        cpu_alone = std::max(cpu_alone, g);
        total     = std::max(total, gm);
    };
    int hi = n_max, lo = 0;   // hi: a count within 95% of the best so far, lo: the largest one measured below it
    measure(n_max);
    for (int nt = n_max / 2; nt >= 1; nt /= 2) {
        measure(nt);
        if (meas[nt].first >= 0.95 * cpu_alone) { hi = nt; } else { lo = nt; break; }
    }
    while (hi - lo > 1) {
        const int mid = (hi + lo) / 2;
        measure(mid);
        if (meas[mid].first >= 0.95 * cpu_alone) { hi = mid; } else { lo = mid; }
    }
    int n_sat = n_max;
    for (const auto & m : meas) {
        if (m.second.first >= 0.95 * cpu_alone) { n_sat = m.first; break; }
    }
    mc->cpu_sat_threads = n_sat;
    if (!moe_opt("cpu-ram-gbs")) { knobs().cpu_gbs = cpu_alone; }
    if (!moe_opt("ram-ceiling-gbs")) { knobs().ddr_gbs = std::max(cpu_alone, total); }
    // the DDR budget gate (GATE=3) needs the budget just measured: on by default then (GLM short chat +6%, MiMo +16% in a pair); the tuner can turn it off
    if (!user_knobs().count("UPLOAD_WAIT")) { knobs().gate = 3; }
    std::string links, rs;
    for (int k = 0; k < mc->n_links; ++k) {
        mc->gbs_link[k] = link_alone[k];
        links += tr_fmt(" link %d %.1f", k, link_alone[k]);
    }
    // the link bandwidth seen on this hardware, rounded to half octaves (x1.41), kept with its placement records: a probe that differs by a factor of 2 or more
    // (a shared uplink, a throttled or reconfigured slot, a bifurcation the PCI ids do not show) drops those records, so the next start measures again
    if (!mc->profile.empty() && !moe_state_place().empty()) {
        const std::string pre = moe_state_place();
        for (int k = 0; k < mc->n_links; ++k) {
            if (link_alone[k] <= 0) { continue; }
            const double seen = std::pow(2.0, std::round(2.0*std::log2(link_alone[k]))/2.0);
            // keyed by the card's PCI address, not the link's index: the index is the cache's order (link 0 = the card prompts go to, then the device
            // order), so when that choice or the device order changes, a card's speed would be compared with another card's (PC1: 24.9 against 5.7 GB/s,
            // a change of 4x) and the placement records dropped for nothing
            const std::string key = pre + ".link." + dev_pci_id(mc->link_dev[k], k);
            std::string old;
            const double o = moe_state_get(mc->profile, key, old) ? atof(old.c_str()) : 0.0;
            if (o > 0 && (link_alone[k] >= 2*o || link_alone[k] <= o/2)) {
                LLAMA_LOG_WARN("moe-cache: link %d now %.1f GB/s, it was %.1f when the placement was measured: those records are dropped, the next start measures again\n", k, link_alone[k], o);
                moe_state_erase(mc->profile, pre);
            }
            if (o <= 0 || link_alone[k] >= 2*o || link_alone[k] <= o/2) {
                char nb[32];
                snprintf(nb, sizeof nb, "%.1f", seen);
                moe_state_set(mc->profile, {{ key, nb }});
            }
        }
    }
    {
        size_t i = 0;
        for (const auto & m : meas) { rs += tr_fmt("%s%d:%.1f/%.1f", i++ ? " " : "", m.first, m.second.first, m.second.second); }
    }
    LLAMA_LOG_WARN("moe-cache: probe: links alone%s GB/s; CPU threads:GB/s alone/with links %s -> CPU %.1f GB/s "
        "(saturated by %d threads), RAM budget %.1f GB/s\n", links.c_str(), rs.c_str(), cpu_alone, n_sat, knobs().ddr_gbs);
}

// physical cores (first SMT thread of each) per L3 domain, domains in order of their first CPU; empty if unknown
std::vector<std::vector<int>> l3_domain_cores() {
    std::vector<std::vector<int>> dom;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    std::vector<char> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        return dom;
    }
    std::vector<KAFFINITY> l3, cores; // processor group 0 only (<= 64 logical CPUs)
    for (DWORD off = 0; off < len; ) {
        auto * e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        if (e->Relationship == RelationCache && e->Cache.Level == 3 && e->Cache.GroupMask.Group == 0) {
            l3.push_back(e->Cache.GroupMask.Mask);
        }
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupMask[0].Group == 0) {
            cores.push_back(e->Processor.GroupMask[0].Mask);
        }
        off += e->Size;
    }
    for (KAFFINITY m : l3) {
        std::vector<int> d;
        for (KAFFINITY c : cores) {
            if (c != 0 && (c & m) == c) {
                int first = 0;
                while (!((c >> first) & 1)) { ++first; }
                d.push_back(first);
            }
        }
        std::sort(d.begin(), d.end());
        if (!d.empty()) { dom.push_back(d); }
    }
    std::sort(dom.begin(), dom.end());
#elif defined(__linux__)
    auto read = [](const std::string & path) { std::ifstream f(path); std::string v; std::getline(f, v); return v; };
    auto parse = [](const std::string & list) {
        std::vector<int> r;
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            const std::string part = list.substr(i, j == std::string::npos ? std::string::npos : j - i);
            const size_t dash = part.find('-');
            if (!part.empty()) {
                const int a = atoi(part.c_str()), b = dash == std::string::npos ? a : atoi(part.c_str() + dash + 1);
                for (int k = a; k <= b; ++k) r.push_back(k);
            }
            if (j == std::string::npos) break;
            i = j + 1;
        }
        return r;
    };
    std::vector<std::string> seen;
    for (int cpu = 0; cpu < 1024; ++cpu) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        const std::string sib = read(base + "/topology/thread_siblings_list");
        if (sib.empty()) { break; }
        const std::vector<int> sl = parse(sib);
        if (sl.empty() || sl[0] != cpu) { continue; }
        const std::string l3 = read(base + "/cache/index3/shared_cpu_list");
        auto it = std::find(seen.begin(), seen.end(), l3);
        if (it == seen.end()) { seen.push_back(l3); dom.emplace_back(); it = seen.end() - 1; }
        dom[it - seen.begin()].push_back(cpu);
    }
#endif
    return dom;
}

std::vector<int> l3_prefetch_cpus() {
    std::vector<int> out;
    for (const auto & d : l3_domain_cores()) {
        out.push_back(d.back()); // the domain's highest physical core
    }
    if (out.empty()) out.push_back(-1);
    return out;
}

// L3 prefetch thread for one CCX (side 0: first half of every expert matrix's rows, side 1: second half, matching
// GGML_MOE_CCX_SPLIT): waits for the current CPU expert phase to end, then reads the predicted experts' rows
void l3pf_loop(moe_cache * mc, int side, int n_dom, int cpu) {
    unpin_helper();
#if defined(__linux__)
    if (cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
#elif defined(_WIN32)
    if (cpu >= 0 && cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
    }
#endif // ponytail: macOS has no thread pinning (AUTO_L3 turns the prefetch off if it doesn't pay)
    uint64_t seen = 0;
    volatile uint64_t sink = 0;
    for (;;) {
        moe_cache::l3pf_job job;
        {
            std::unique_lock<std::mutex> lk(mc->l3_mtx);
            mc->l3_cv.wait(lk, [&]() { return mc->l3_stop || mc->l3_job.gen != seen; });
            if (mc->l3_stop) {
                return;
            }
            job  = mc->l3_job;
            seen = job.gen;
        }
        // the posting layer's CPU phase still reads DDR: start after it (bounded wait)
        const int64_t t0 = ggml_time_us();
        while (mc->cpu_busy && ggml_time_us() - t0 < (int64_t) (2*mean_layer_us(mc))) {
            std::this_thread::yield();
        }
        const auto & ls = mc->layers[job.li];
        bool dropped = false;
        for (int32_t e : job.ex) {
            for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
                const int64_t  rows = t->ne[1];
                const uint8_t * base = (const uint8_t *) t->data + (size_t) e*t->nb[2];
                const uint8_t * a = base + (size_t) (rows*side/n_dom)*t->nb[1];
                const uint8_t * b = base + (size_t) (rows*(side + 1)/n_dom)*t->nb[1];
                uint64_t x = 0;
                for (const uint8_t * q = a; q < b; q += 64) {
                    x += *(const volatile uint64_t *) q;
                }
                sink = sink + x;
                if (mc->l3_job.gen != seen) { // ponytail: unlocked read, a newer job only makes this one stop early
                    dropped = true;
                    break;
                }
            }
            if (dropped) {
                break;
            }
            if (side == 0) {
                mc->l3_n_ex++;
            }
        }
        if (side == 0) {
            // the target layer's CPU expert op has run its observer: the phase started before we finished
            if (dropped) {
                mc->l3_dropped++;
            } else if (mc->last_obs >= (int64_t) job.li) {
                mc->l3_late++;
            } else {
                mc->l3_done++;
            }
        }
    }
}

void pred_loop(moe_cache * mc) {
    unpin_helper();
    std::vector<float>   lg;
    std::vector<int32_t> idx;
    for (;;) {
        moe_cache::pred_req r;
        {
            std::unique_lock<std::mutex> lk(mc->pmtx);
            mc->pcv.wait(lk, [mc]() { return mc->pred_stop || !mc->preq.empty(); });
            if (mc->pred_stop) {
                return;
            }
            r = std::move(mc->preq.front());
            mc->preq.pop_front();
        }
        const int64_t t_pop = ggml_time_us();
        const int64_t ne = (int64_t) mc->layers[r.li + 1].expert_slot.size();
        // candidates per predicted layer, nearest first
        std::vector<std::vector<int32_t>> cands(r.n_blk);
        std::vector<std::vector<float>>   cscore(r.n_blk); // the candidates' predictor scores
        for (int64_t k = 0; k < r.n_blk && r.li + 1 + k < mc->layers.size(); ++k) {
            const size_t tl = r.li + 1 + k;
            auto & ls = mc->layers[tl];
            if (knobs().lead != 0) {
                // the nearest layers whose upload still lands: lead = upload time per expert on this layer's link / layer time
                const double ul = mc->link_us[ls.link] * (1 + (knobs().auto_tune == 1 ? mc->lead_extra[ls.link] : 0));
                const int lead = ul > 0 ? std::max(1, (int) std::ceil(ul / std::max(1.0, mean_layer_us(mc)))) : 1;
                if (k + 1 < lead || k + 1 > lead + (int) knobs().lead_span) {
                    continue;
                }
            }
            const auto & b = mc->pred_b[tl];
            auto & cand = cands[k];
            lg.resize(ne);
            idx.resize(ne);
            for (int64_t t = 0; t < r.n_tok; ++t) {
                const float * src = r.x.data() + (t*r.n_blk + k)*ne;
                for (int64_t e = 0; e < ne; ++e) {
                    lg[e] = b.empty() ? src[e] : 1.0f/(1.0f + expf(-src[e])) + b[e];
                }
                for (int32_t e = 0; e < ne; ++e) { idx[e] = e; }
                // stream slots: over-predict (a wrong guess costs idle bandwidth only, never a cached expert)
                const bool stream = knobs().stream > 0 && !ls.stream_slots.empty();
                const int32_t sm = knobs().auto_tune == 1 && mc->auto_m[ls.link] > 0 ? mc->auto_m[ls.link] : (int32_t) knobs().stream_m;
                const int32_t m = std::min<int32_t>(stream ? sm : mc->pred_m, (int32_t) ne);
                std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), [&](int32_t a, int32_t c) { return lg[a] > lg[c]; });
                if (mc->pred_ra) {
                    // model bigger than RAM: start reading the predicted experts the CPU will compute
                    for (int32_t j = 0; j < m; ++j) {
                        if (ls.expert_slot[idx[j]] < 0) {
                            readahead(ls, idx[j]);
                        }
                    }
                }
                // only confident predictions: precision 0.28 near the cut, 0.88 with a 0.1 lead (GLM, offline)
                for (int32_t j = 0; j < m; ++j) {
                    if (!stream && lg[idx[j]] - lg[idx[m - 1]] < mc->pred_margin) {
                        break;
                    }
                    if (std::find(cand.begin(), cand.end(), idx[j]) == cand.end()) {
                        cand.push_back(idx[j]);
                        cscore[k].push_back(lg[idx[j]]);
                    }
                }
            }
        }
        {
            // a backlog would finish too late and delay the step's own uploads
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            if (knobs().stream <= 0 && !mc->todo.empty()) {
                continue; // stream jobs instead go first on their link and are dropped once their layer started
            }
        }
        std::vector<upload_job> jobs;
        {
            std::lock_guard<std::mutex> lock(mc->mtx);
            if (r.step != mc->n_steps) {
                mc->n_pred_late++;
                continue;
            }
            for (int64_t k = 0; k < r.n_blk; ++k) {
                const size_t tl = r.li + 1 + k;
                // too late: that layer's GPU split may be running
                if (cands[k].empty() || mc->gpu_started >= (int64_t) tl) {
                    mc->n_pred_late += !cands[k].empty();
                    continue;
                }
                auto & ls = mc->layers[tl];
                const auto & cand = cands[k];
                int32_t n_job = 0;
                int32_t link_cap = INT32_MAX; // AUTO=2: uploads this link can land before layer tl
                // smart offset: the upload must land before the layer; it is k+1 layers ahead, a layer takes ~token/layers
                if (knobs().offset != 0 && mc->step_us_idle > 0 && mc->link_us[ls.link] > 0) {
                    // time until layer tl's experts are needed: the measured layer clock from the predicting layer on
                    // (falls back to token time / layers before the clock has samples)
                    double until = 0;
                    const double avg = mc->step_us_idle / (double) mc->layers.size();
                    for (size_t q = r.li; q < tl; ++q) {
                        until += q < mc->layer_dt.size() && mc->layer_dt[q] > 0 ? mc->layer_dt[q] : avg;
                    }
                    const bool cap_mode = knobs().auto_tune >= 2;
                    const double need = mc->link_us[ls.link] * (1 + (knobs().auto_tune == 1 ? mc->lead_extra[ls.link] : 0));
                    if (until < need) {
                        mc->n_pred_late += !cand.empty();
                        continue;
                    }
                    if (cap_mode) {
                        // AUTO=2: as many uploads as this link can finish before the layer, from measured times only (no
                        // thresholds): (time until the layer) / (one expert on this link), minus what the link has queued
                        int64_t queued = 0;
                        {
                            std::lock_guard<std::mutex> wlk(mc->wmtx);
                            for (const auto & q : mc->todo) {
                                queued += mc->layers[q.layer_idx].link == ls.link;
                            }
                        }
                        link_cap = (int32_t) std::max<int64_t>(0, (int64_t) (until / mc->link_us[ls.link]) - queued);
                    }
                }
                if (knobs().stream > 0 && ((ls.slow && knobs().stream_slow == 0) || (knobs().auto_tune == 1 && mc->stream_off[ls.link]))) {
                    continue;
                }
                const int32_t n_ss = knobs().stream > 0 ? std::min<int32_t>((int32_t) knobs().stream, (int32_t) ls.stream_slots.size()) : 0;
                for (size_t ci = 0; ci < cand.size(); ++ci) {
                    const int32_t id = cand[ci];
                    const float   sc = cscore[k][ci];
                    if (n_job >= (n_ss > 0 ? n_ss : mc->pred_max) || n_job >= link_cap) {
                        break;
                    }
                    if (ls.expert_slot[id] >= 0 || ls.queued[id] || ls.adopt_slot[id] >= 0) {
                        continue;
                    }
                    if (n_ss > 0) {
                        // next free stream slot, round robin; its previous (streamed) expert leaves
                        int32_t slot = -1;
                        if (knobs().slotkeep != 0) {
                            // an empty slot or one streamed for an earlier step first, else the lowest-scored one, and only
                            // if this candidate scores higher (candidates come best first: a refusal ends this layer)
                            float lo = 0;
                            for (int32_t t = 0; t < n_ss; ++t) {
                                const int32_t sl = ls.stream_slots[t];
                                if (ls.slot_in_flight[sl]) {
                                    continue;
                                }
                                if (ls.slot_expert[sl] < 0 || ls.stream_step[sl] != r.step) {
                                    slot = sl;
                                    break;
                                }
                                if (slot < 0 || ls.stream_score[sl] < lo) {
                                    slot = sl;
                                    lo   = ls.stream_score[sl];
                                }
                            }
                            if (slot >= 0 && ls.slot_expert[slot] >= 0 && ls.stream_step[slot] == r.step && lo >= sc) {
                                slot = -1;
                            }
                        } else {
                            for (int32_t t = 0; t < n_ss && slot < 0; ++t) {
                                const int32_t sl = ls.stream_slots[ls.stream_next++ % n_ss];
                                slot = ls.slot_in_flight[sl] ? -1 : sl;
                            }
                        }
                        if (slot < 0) {
                            break;
                        }
                        ls.stream_score[slot] = sc;
                        ls.stream_step[slot]  = r.step;
                        const int32_t victim = ls.slot_expert[slot];
                        if (victim >= 0) {
                            ls.expert_slot[victim] = -1;
                            ls.slot_expert[slot]   = -1;
                            set_table_entry(ls.pub, victim, ls.pub.n_slots);
                            ls.cached_since[victim] = 0;
                            ls.prefetched[victim]   = 0;
                        }
                        ls.slot_in_flight[slot] = true;
                        ls.queued[id]           = 1;
                        upload_job j { tl, id, slot };
                        j.urgent = true;
                        j.stream = true;
                        j.step   = r.step;
                        jobs.push_back(j);
                        n_job++;
                        continue;
                    }
                    // victim: an empty slot, else the lowest-scored cached expert not predicted now
                    int32_t slot = -1;
                    double best = 1e300;
                    for (int32_t sl = 0; sl < ls.pub.n_slots; ++sl) {
                        if (ls.slot_in_flight[sl]) {
                            continue;
                        }
                        const int32_t v = ls.slot_expert[sl];
                        if (v < 0) { slot = sl; break; }
                        if (pinned(mc, ls, v) || std::find(cand.begin(), cand.end(), v) != cand.end()) {
                            continue;
                        }
                        const double c = score(ls, v);
                        if (c < best) { best = c; slot = sl; }
                    }
                    if (slot < 0) {
                        break;
                    }
                    const int32_t victim = ls.slot_expert[slot];
                    if (victim >= 0) {
                        ls.expert_slot[victim] = -1;
                        ls.slot_expert[slot]   = -1;
                        set_table_entry(ls.pub, victim, ls.pub.n_slots);
                        note_evict(mc, ls, victim);
                        ls.cached_since[victim] = 0;
                        ls.prefetched[victim]   = 0;
                        if (ls.dropped[victim]) { page_hint(mc, ls, victim, false); ls.dropped[victim] = false; }
                    }
                    ls.slot_in_flight[slot] = true;
                    ls.queued[id]           = 1;
                    upload_job j { tl, id, slot };
                    j.urgent = true;
                    jobs.push_back(j);
                    n_job++;
                }
            }
        }
        if (!jobs.empty()) {
            {
                std::lock_guard<std::mutex> wlk(mc->wmtx);
                const int64_t now = ggml_time_us();
                for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) {
                    it->t_queued = now;
                    mc->todo.push_front(*it);
                }
                if (tr_on()) {
                    std::string tg;
                    for (const auto & j : jobs) {
                        tg += tr_fmt("%sL%zu:e%d", tg.empty() ? "" : " ", j.layer_idx, j.expert);
                    }
                    tr(TR_PRED, tr_fmt("predict from L%zu: %zu uploads", r.li, jobs.size()), t_pop, now - t_pop, tr_fmt("\"jobs\":\"%s\"", tg.c_str()));
                }
                mc->n_pred_up += jobs.size();
            }
            mc->wcv.notify_all();
        }
    }

}

// Prefill preheat: the scheduler just copied the used experts of `weight` to `dev` for a prompt
// batch. Keep the ones this prompt uses more than what is cached: they are copied device-to-device
// into a slot (no extra PCIe traffic) and published at the next step. Up to LLAMA_MOE_CACHE_ADOPT
// (default 1.0) of a layer's slots per batch; only layers whose cache is on `dev`.
bool moe_fill_cb(const ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev,
                void ** data, ggml_backend_buffer_t * buffer, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    layer_state & ls = mc->layers[it->second.first];
    const int kind = it->second.second;
    ggml_tensor * c = kind == 0 ? ls.pub.up_c : kind == 1 ? ls.pub.gate_c : ls.pub.down_c;
    if (!c || !c->buffer || c->nb[2] < weight->nb[2] || c->nb[1] != weight->nb[1] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        return false;
    }
    static const double frac = [] {
        const char * e = moe_opt("adopt");
        return e ? atof(e) : 1.0; // measured: 1/16 +0.3 hit pts, 1/4 +3.8, 1 +11.4 (GLM 12k, 1 GPU)
    }();
    std::lock_guard<std::mutex> lock(mc->mtx);
    if (expert < 0 || expert >= (int32_t) ls.expert_slot.size()) {
        return false;
    }
    int32_t slot = ls.adopt_slot[expert];
    if (slot < 0) {
        if (ls.expert_slot[expert] >= 0 || (int32_t) ls.adopted.size() >= (int32_t) (frac * ls.n_cache)) {
            return false;
        }
        // victim: a free block, else the non-sticky cached expert with the least lifetime use (block pool: of any layer of the class),
        // and only if this expert (whose use this prompt just counted) was used more. Hot maps per topic: the topic's map and this request's
        // uses stand for the lifetime use, as in score()
        auto life = [&](const layer_state & l, int32_t v) {
            return g_topic_on.load(std::memory_order_relaxed)
                ? (double) l.topic_heat[v] + (double) l.topic_cnt[v] / (double) std::max(g_topic_tmax.load(std::memory_order_relaxed), l.topic_max)
                : (double) l.glob_count[v];
        };
        double least;
        const size_t ali = it->second.first;
        const slot_pick pk = pick_slot(mc, ali, [&](layer_state & lo, size_t, int32_t v) { return pinned(mc, lo, v) ? -1.0 : life(lo, v); }, least);
        slot = pk.slot;
        if (slot < 0 || least >= life(ls, expert)) {
            return false;
        }
        if (pk.owner >= 0) {
            layer_state & lo = mc->layers[pk.owner];
            const int32_t victim = lo.slot_expert[slot];
            lo.expert_slot[victim] = -1; // no later batch reads the slot as the victim
            lo.slot_expert[slot]   = -1;
            lo.adopt_victims.push_back(victim);   // its table entry is reset at the next step
        }
        pool_take(mc, ali, slot);
        ls.slot_in_flight[slot] = true;
        ls.adopt_slot[expert]   = slot;
        ls.adopted.push_back(expert);
    }
    ls.adopt_mask[expert] |= (uint8_t) (1 << kind);
    *data   = (uint8_t *) c->data + (size_t) slot*c->nb[2];
    *buffer = c->buffer;
    return true;
}

// drop (cached in VRAM) or prefetch (evicted, the CPU needs it again) an expert's mmap'd pages
void page_hint(const moe_cache * mc, const layer_state & ls, int32_t expert, bool drop) {
#ifndef _WIN32
    if (!mc->drop_cached || expert < 0) {
        return;
    }
    static const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
        const uintptr_t a   = (uintptr_t) t->data + (size_t) expert*t->nb[2];
        const uintptr_t beg = (a + pg - 1) & ~(uintptr_t) (pg - 1);
        const uintptr_t end = (a + t->nb[2]) & ~(uintptr_t) (pg - 1);
        if (end <= beg) {
            continue;
        }
#ifdef MADV_PAGEOUT
        const int advice = drop ? MADV_PAGEOUT : MADV_WILLNEED;
#else
        const int advice = drop ? MADV_DONTNEED : MADV_WILLNEED;
#endif
        madvise((void *) beg, end - beg, advice);
    }
#else
    GGML_UNUSED(mc); GGML_UNUSED(ls); GGML_UNUSED(expert); GGML_UNUSED(drop);
#endif
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

// The CUDA backend's copy without the stream synchronisation (see ggml_backend_cuda_set_tensor_nosync_fork in ggml-cuda.cu for why: one sync per expert slice
// was 34k syncs in a 1641-token GLM prompt). nullptr when there is no CUDA backend; the caller then uses the synchronous ggml_backend_tensor_set.
// A worker that used it calls cuda_thread_sync() once before it reports the job done.
typedef void (*cuda_set_nosync_t)(ggml_tensor *, const void *, size_t, size_t);
cuda_set_nosync_t cuda_set_nosync() {
    static const cuda_set_nosync_t f = (cuda_set_nosync_t) cuda_obs_proc("ggml_backend_cuda_set_tensor_nosync_fork");
    return f;
}
void cuda_thread_sync() {
    static void (* const f)() = (void (*)()) cuda_obs_proc("ggml_backend_cuda_thread_sync_fork");
    if (f) { f(); }
}

// table writes go to the host table (host memory, read by the CPU expert op) and mark the layer; the device
// copy is flushed in one transfer per dirty layer at the next sync point (split callback, end of step) instead of
// one synchronous 4-byte copy per entry
std::mutex                                g_tbl_mtx;
std::vector<llama_moe_cache_layer_fork *>      g_tbl_dirty;

void set_table_entry(llama_moe_cache_layer_fork & pub, int32_t expert, int32_t slot_or_dummy) {
    if (pub.jit_n > 0 && slot_or_dummy != pub.n_slots + 1 && g_cache) {
        // the main cache takes (or drops) this expert: it leaves the JIT pool, or the pool chain would add it twice
        for (auto & l : g_cache->layers) {
            if (&l.pub == &pub) {
                const int32_t ps = l.pool_slot.empty() ? -1 : l.pool_slot[expert];
                if (ps >= 0) {
                    l.pool_slot[expert] = -1;
                    l.pool_expert[ps]   = -1;
                    l.pool_dirty        = true;
                }
                break;
            }
        }
    }
    ((int32_t *) pub.host_table->data)[expert] = slot_or_dummy;
    std::lock_guard<std::mutex> lk(g_tbl_mtx);
    if (std::find(g_tbl_dirty.begin(), g_tbl_dirty.end(), &pub) == g_tbl_dirty.end()) {
        g_tbl_dirty.push_back(&pub);
    }
}


void flush_tables() {
    std::vector<llama_moe_cache_layer_fork *> d;
    {
        std::lock_guard<std::mutex> lk(g_tbl_mtx);
        d.swap(g_tbl_dirty);
    }
    std::vector<int32_t> v;
    for (auto * pub : d) {
        const int32_t * h = (const int32_t *) pub->host_table->data;
        v.assign(h, h + ggml_nelements(pub->dev_table));
        for (auto & x : v) {
            x = x >= pub->n_slots ? -1 : x;   // uncached (n_slots) and JIT pool experts (n_slots + 1): -1, which every kernel skips (no zero block is read)
        }
        ggml_backend_tensor_set(pub->dev_table, v.data(), 0, v.size()*sizeof(int32_t));
    }
    if (g_cache) {
        for (auto & l : g_cache->layers) {
            if (!l.pool_dirty) {
                continue;
            }
            l.pool_dirty = false;
            std::vector<int32_t> t(l.pool_slot.size());
            for (size_t e = 0; e < t.size(); ++e) {
                t[e] = l.pool_slot[e] >= 0 ? l.pool_slot[e] : -1;
            }
            ggml_backend_tensor_set(l.pub.jit_table, t.data(), 0, t.size()*sizeof(int32_t));
        }
    }
}

} // namespace

// --moe ledger=1: a write into a card's pool by code outside this file (the dense split's copies, llama-context.cpp), logged with its byte range; every
// block under it must be lent to the graph (-2) at that moment, and it must not touch the KV zone or a zero block. Declared where it is called (no header edit).
void llama_moe_cache_ledger_fork(ggml_backend_dev_t dev, const char * kind, const void * ptr, size_t bytes, const char * who) {
    moe_cache * mc = g_cache;
    if (!led_on() || !mc || !ptr) { return; }
    const std::string w = who ? who : "";
    for (const auto & ap : mc->alias) {
        if (ap.dev != dev || !ap.buf) { continue; }
        const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(ap.buf);
        const uint8_t * p    = (const uint8_t *) ptr;
        if (p < base || p >= base + ggml_backend_buffer_get_size(ap.buf)) { continue; }
        const size_t lo = (size_t) (p - base), hi = lo + bytes;
        led_line(led_card(ap), kind, lo, hi, w);
        led_check_range(ap, kind, lo, hi, w, true, false);
        for (const auto & sg : ap.segs) {
            const auto & ps = mc->pool[sg.pcls];
            for (int32_t b = 0; b < sg.n; ++b) {
                const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                if (b1 > lo && b0 < hi && ps.owner[b] != -2) {
                    led_overlap(led_card(ap), b0, b1, tr_fmt("%s %s: block %d of class %d is not lent to the graph (owner %d)", kind, w.c_str(), b, sg.pcls, (int) ps.owner[b]));
                    break;
                }
            }
        }
    }
}

// Big-batch prefill offloads expert matmuls and copies the used experts to the
// GPU; experts published in this cache on that GPU are copied from their slot
// (device-to-device) instead. Called from the scheduler on the decode thread;
// published slots are stable until the next step(), which runs after a sync.
// Router prediction: before a GPU split launches, publish the predicted experts whose upload
// finished. No split is executing (the previous CPU split synchronized), so tables can change.
static void moe_split_cb(ggml_backend_t backend, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    for (int k = 0; k < mc->n_links; ++k) { // main thread only; read by the JIT op on the same thread
        if (!mc->link_be[k] && mc->link_dev[k] == ggml_backend_get_device(backend)) {
            mc->link_be[k] = backend;
        }
    }
    if (mc->pred_m == 0 && knobs().jit == 0) {
        return;
    }
    std::vector<upload_job> ready;
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        for (size_t i = 0; i < mc->done.size(); ) {
            if (mc->done[i].urgent || knobs().wait == 0) {
                ready.push_back(mc->done[i]);
                mc->done[i] = mc->done.back();
                mc->done.pop_back();
            } else {
                ++i;
            }
        }
    }
    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->gpu_started = mc->last_obs + 1;
    const bool trc = tr_on();
    const int64_t now = trc ? ggml_time_us() : 0;
    if (trc) {
        tr(TR_SCHED, tr_fmt("gpu split from L%lld", (long long) mc->gpu_started), now);
    }
    for (const auto & j : ready) {
        publish_job(mc, j);
        mc->n_pred_pub += !j.skipped;
        if (trc && !j.skipped) {
            tr(TR_SCHED, tr_fmt("publish L%zu e%d%s", j.layer_idx, j.expert, (int64_t) j.layer_idx < mc->gpu_started ? " LATE" : ""), now, -1,
                    tr_fmt("\"ahead (layers)\":%lld", (long long) j.layer_idx - mc->gpu_started));
        }
    }
    flush_tables();
}

// The prompt pass's staging fill (ggml-backend.cpp, the prefetch upload) asks which experts of an id range this card already holds, and copies those on the card
// instead of uploading them again (every pass used to upload every expert, also the ones in this pool). Same lookup as moe_src_cb below, for a whole range under
// one lock, and WITHOUT its side effects: nothing is counted, the admit boot is not ended (a pass asks for every expert of every layer). A block whose upload
// is still in flight counts as missing. --moe pf-resident=0 does not register it (the old single upload of the range).
extern "C" {
    typedef int (*ggml_backend_moe_where_fn_t)(const ggml_tensor *, int32_t, int32_t, ggml_backend_dev_t, const void **, ggml_backend_buffer_t *, void *);
    GGML_API void ggml_backend_set_moe_where_fn_fork(ggml_backend_moe_where_fn_t fn, void * user_data);   // ggml-backend.cpp (declared there, no shared header)
}
static void moe_where_count(int64_t range, int64_t found, int64_t inflight, int64_t wrongdev);
static int moe_where_cb(const ggml_tensor * weight, int32_t lo, int32_t hi, ggml_backend_dev_t dev, const void ** data, ggml_backend_buffer_t * buffer, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return 0;
    }
    layer_state & ls = mc->layers[it->second.first];
    std::lock_guard<std::mutex> lock(mc->mtx);
    int lk = -1;
    for (int k = 0; k < mc->n_links; ++k) { if (mc->link_dev[k] == dev) { lk = k; break; } }
    ggml_tensor * c = it->second.second == 0 ? ls.pub.up_c : it->second.second == 1 ? ls.pub.gate_c : ls.pub.down_c;
    if (!c || !c->buffer || c->nb[2] < weight->nb[2] || c->nb[1] != weight->nb[1] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        moe_where_count(hi - lo, 0, 0, hi - lo);
        if (lk >= 0) { mc->pf_ask[lk] += hi - lo; mc->pf_wrong[lk] += hi - lo; }
        return 0;
    }
    int n = 0, inflight = 0;
    for (int32_t e = std::max<int32_t>(lo, 0); e < hi && e < (int32_t) ls.expert_slot.size(); ++e) {
        const int32_t s = ls.expert_slot[e];
        if (s < 0) {
            continue;
        }
        if ((size_t) s < ls.slot_in_flight.size() && ls.slot_in_flight[s]) {
            inflight++;
            continue;
        }
        data[e - lo] = (const uint8_t *) c->data + (size_t) s*c->nb[2];
        ++n;
    }
    moe_where_count(hi - lo, n, inflight, 0);
    if (lk >= 0) { mc->pf_ask[lk] += hi - lo; mc->pf_found[lk] += n; }
    *buffer = c->buffer;
    return n;
}

static bool moe_src_cb(const ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev,
                const void ** data, ggml_backend_buffer_t * buffer, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    layer_state & ls = mc->layers[it->second.first];
    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->boot_done = true;   // a GPU prompt batch (this callback only runs for those): the cache is warmed by adoption, the admit boot is over
    // GPU-offloaded prompt batches never reach the CPU observer (moe_obs_cb): count the prompt's
    // experts here (once per expert and batch, on up) for the lifetime usage and the saved profile.
    // Not queued for upload: extra uploads after a long prompt measured slower (they compete with decode)
    if (it->second.second == 0 && expert >= 0 && expert < (int32_t) ls.expert_slot.size()) {
        ls.expert_count[expert]++;
        ls.topic_max = std::max(ls.topic_max, ++ls.topic_cnt[expert]);
        ls.glob_max = std::max(ls.glob_max, ++ls.glob_count[expert]);
    }
    // an expert the pass takes from RAM: its bytes cross this card's link (live bandwidth of the prompt phase)
    auto streamed = [&]() {
        mc->pf_bytes += (int64_t) weight->nb[2];
        for (int k = 0; k < mc->n_links; ++k) { if (mc->link_dev[k] == dev) { mc->pf_link[k] += (int64_t) weight->nb[2]; break; } }
        return false;
    };
    ggml_tensor * c = it->second.second == 0 ? ls.pub.up_c : it->second.second == 1 ? ls.pub.gate_c : ls.pub.down_c;
    if (!c || !c->buffer || c->nb[2] < weight->nb[2] || c->nb[1] != weight->nb[1] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        return streamed();
    }
    mc->n_src_query++;
    if (expert < 0 || expert >= (int32_t) ls.expert_slot.size() || ls.expert_slot[expert] < 0) {
        return streamed();
    }
    mc->n_src_hit++;
    mc->pf_hit_bytes += (int64_t) weight->nb[2];
    *data   = (const uint8_t *) c->data + (size_t) ls.expert_slot[expert]*c->nb[2];
    *buffer = c->buffer;
    return true;
}

// ---- compute alias ---------------------------------------------------------------------------------------------------------------------------
// --moe alias (default 1, 0 = off): the scheduler places the compute graphs of a GPU at the start of that GPU's expert pool buffer (the first blocks of its biggest
// size class) instead of a buffer of its own. After each graph is allocated, the blocks its tensors reach are taken from the cache (their experts
// evicted, uploads into them awaited) and lent to it; at the next step (a sync point) the blocks above what the last graph needed come back. A
// decode graph needs a few blocks, a big prompt batch many, only while it runs: no buffer is allocated or resized, so the prompt ubatch is
// bounded by the pool, not by a separate compute buffer.
// default on (with the block pool); --moe alias=0 keeps the scheduler's own compute buffers
static std::atomic<bool> g_alias_dropped { false };   // the pool could not lend even the smallest worst-case graph: the scheduler keeps its own buffers
static bool alias_on() {
    static const bool on = [] { const char * e = moe_opt("alias"); return !e || atoi(e) != 0; }();
    return on && !g_alias_dropped.load(std::memory_order_relaxed);
}

void llama_moe_cache_alias_drop_fork() {
    g_alias_dropped = true;
}

// the dense split's copies sit at the start of dev's pool buffer (llama_dsplit_mirror_place_fork): the pass planner keeps room for them
void llama_moe_cache_alias_mirror_fork(ggml_backend_dev_t dev, size_t bytes) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev == dev) { ap.mirror_b = bytes; }
    }
}

// before a graph is allocated (and before a reserve): its tensors start at off in dev's pool buffer, the lent range covers [0, off) too
void llama_moe_cache_alias_offset_fork(ggml_backend_dev_t dev, size_t off) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev == dev) { ap.base_off = off; }
    }
}

static size_t kv_lend_potential(const moe_cache::alias_pool & ap, size_t min_cells = 0);   // below (KV zone)
std::vector<llama_moe_alias_buf_fork> llama_moe_cache_alias_buffers_fork() {
    std::vector<llama_moe_alias_buf_fork> out;
    moe_cache * mc = g_cache;
    if (!mc || !alias_on()) {
        return out;
    }
    for (const auto & ap : mc->alias) {
        out.push_back({ ap.buf, ap.dev, kv_lend_potential(ap) });   // up to the last class's last expert block, less the KV zone wherever it sits (moe_alias moves it behind a bigger graph)
    }
    return out;
}

GGML_API size_t ggml_backend_sched_node_max_dev(ggml_backend_sched_t sched, ggml_backend_dev_t dev, size_t (*fn)(void * user, struct ggml_tensor * const * nodes, int n),
        size_t (*copy_fn)(ggml_backend_t src_backend, ggml_backend_t dst_backend, const struct ggml_tensor * src, const struct ggml_tensor * dst), void * user);   // ggml-backend.cpp

// The CUDA scratch a graph is lent on this card, before it runs: the most any node that runs on THIS card takes, each computed by the backend from the node itself
// (ggml_backend_cuda_nodes_scratch_fork: the matmuls, flash attention, top-k and argsort mirror their own pool allocations; tests/test-scratch-fork.cpp runs each op
// and checks that what it took equals what was computed). The scheduler keeps the value per split until it splits again (ggml_backend_sched_node_max_dev), so a
// reused graph (a decode step) asks nothing. A card that computes a share of the tokens (the dense split) is lent the scratch of its share: the
// whole batch lent cuda0x4 648 MiB where its share asked for 119 MiB (DeepSeek 4.1, 3389 tokens, 2026-10-27). Nothing is learned or guessed on top: a rate
// learned from a graph that was lent too little (a 2-token decode graph whose flash attention took 2.02 MiB) once lent a 3389-token pass 3416 MiB on each card
// where its nodes took at most 678 (DeepSeek 4.1, 2026-10-27). The scheduler holds the graph's splits: the graph about to run (llama_moe_cache_alias_fork), the
// reserved one (llama_moe_cache_alias_fit_fork) or a measure-only reserve (llama_moe_cache_graph_scratch_fork). A card that is no CUDA device gets none.
// --moe log=a: what computing the scratch costs the lend (host time, calls, splits whose nodes were asked: a reused graph asks none)
static std::atomic<int64_t> g_scr_us{0}, g_scr_n{0}, g_scr_splits{0};
static size_t node_scratch(void * user, struct ggml_tensor * const * nodes, int n) {
    using nodes_fn_t = size_t (*)(int, const ggml_tensor * const *, int);
    static nodes_fn_t fn = [] {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        return reg ? (nodes_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nodes_scratch_fork") : nullptr;
    }();
    g_scr_splits++;
    return fn ? fn(*(const int *) user, nodes, n) : 0;
}
static size_t alias_scratch(const moe_cache::alias_pool & ap, ggml_backend_sched_t sched) {
    const char * nm = ggml_backend_dev_name(ap.dev);
    if (ap.cuda_dev < 0 && (!nm || strncmp(nm, "CUDA", 4) != 0)) {
        return 0;
    }
    using copy_fn_t = size_t (*)(ggml_backend_t, ggml_backend_t, const ggml_tensor *, const ggml_tensor *);
    static copy_fn_t copy_fn = [] {   // the inputs other cards copy from this one: a gappy view is gathered here first (ggml_backend_cuda_copy_scratch_fork)
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
        return reg ? (copy_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_copy_scratch_fork") : nullptr;
    }();
    const int64_t t0 = ggml_time_us();
    int dev = ap.cuda_dev >= 0 ? ap.cuda_dev : atoi(nm + 4);
    const size_t need = ggml_backend_sched_node_max_dev(sched, ap.dev, node_scratch, copy_fn, &dev);
    g_scr_us += ggml_time_us() - t0;
    g_scr_n++;
    return need;
}

// the CUDA scratch of the graph the scheduler split last, on dev's pool (0: dev has no pool). llama_context::sched_reserve adds it to each measure-only reserve's
// graph bytes: what a pass of that size borrows from the pool is its tensors and that scratch.
size_t llama_moe_cache_graph_scratch_fork(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    moe_cache * mc = g_cache;
    if (!mc || !alias_on()) {
        return 0;
    }
    for (const auto & ap : mc->alias) {
        if (ap.dev == dev) { return alias_scratch(ap, sched); }
    }
    return 0;
}

// After a worst-case reserve: do the graph, its scratch and the prefetch staging slots fit in what the pool lends? (graph: needed, stage: optional)
llama_moe_alias_fit_fork llama_moe_cache_alias_fit_fork(ggml_backend_sched_t sched, int32_t n_tokens) {
    llama_moe_alias_fit_fork fit;
    moe_cache * mc = g_cache;
    if (!mc || !alias_on() || mc->alias.empty()) {
        return fit;
    }
    for (auto & ap : mc->alias) {
        size_t peak = 0;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(be) == ap.dev) { peak = ggml_backend_sched_graph_peak(sched, be); }
        }
        const size_t cap     = kv_lend_potential(ap, (size_t) n_tokens);
        const size_t scratch = alias_scratch(ap, sched);
        const size_t base    = GGML_PAD(ap.base_off + GGML_PAD(peak, 256) + scratch, 256);   // behind the dense split's copies when the reserved graph splits
        {   // what the shrink-to-fit decided on, once per reserve size (a graph of this many tokens against what the pool can lend)
            static int said_tok = -1;
            if (said_tok != n_tokens) {
                said_tok = n_tokens;
                LLAMA_LOG_INFO("moe-cache: alias fit %s: %d tokens: graph peak %.1f MiB, CUDA scratch %.1f MiB, base offset %.1f MiB -> need %.1f MiB, the pool can lend %.1f MiB (pool %.1f MiB)\n",
                        ggml_backend_dev_name(ap.dev), n_tokens, peak / 1048576.0, scratch / 1048576.0, ap.base_off / 1048576.0, base / 1048576.0, cap / 1048576.0,
                        ap.kvt.empty() ? ap.span / 1048576.0 : ap.tail_end / 1048576.0);
            }
        }
        fit.cap  = std::min(fit.cap ? fit.cap : cap, cap);
        fit.need = std::max(fit.need, base);
        if (base > cap) {
            fit.graph = false;
        }
        {
            // A staging slot holds one whole expert tensor. The scheduler only knows its slot size once a prompt batch has been computed, so at reserve time it
            // says 0 and the slots looked free: the micro-batch was never shrunk for them, and the real prompt graph then found no room, ran WITHOUT prefetch and
            // paid its whole upload on top of the compute (GLM 12k, one x16: 649 prompt t/s at the auto micro-batch, 769 at 8192 where the slots fit; nsys: 0% of the copy
            // time hidden behind kernels). Until the scheduler knows, the largest expert tensor of the model is the slot size.
            size_t slot_b = ggml_backend_sched_prefetch_slot_bytes_dev(sched, ap.dev);   // 0 for a card that runs no staged expert matmul in this graph
            if (slot_b == 0 && ap.dev == mc->link_dev[0]) {
                for (const auto & l : mc->layers) {
                    for (const ggml_tensor * t : { l.pub.up_src, l.pub.gate_src, l.pub.down_src }) {
                        if (t) { slot_b = std::max(slot_b, GGML_PAD(ggml_nbytes(t), 256)); }
                    }
                }
            }
            // Seed the pass planner (pp_plan) from this worst-case reserve: it needs the slot size and the graph bytes per token of the card, and used to learn
            // them only when a prompt graph first asked for lent blocks. The first prompt of every start was therefore planned by the old equal-passes rule, and
            // a benchmark (or a user) who looks at the first request saw that plan. --moe pp-seed=0 switches the seeding off.
            static const bool seed = [] { const char * e = moe_opt("pp-seed"); return !e || atoi(e) != 0; }();
            if (seed && n_tokens >= 64) {
                if (ap.slot_b == 0 && slot_b > 0) { ap.slot_b = slot_b; }
                if (ap.graph_tok == 0 && peak > 0) {
                    ap.graph_tok   = (peak    + (size_t) n_tokens - 1) / (size_t) n_tokens;
                    ap.scr_tok     = (scratch + (size_t) n_tokens - 1) / (size_t) n_tokens;
                    ap.graph_tok_n = (uint32_t) n_tokens;
                }
            }
            const size_t stage = slot_b * (size_t) std::max(0, ggml_backend_sched_get_prefetch_experts_slots(sched));
            if (stage > 0 && base + stage > cap) {
                fit.stage = false;
                fit.need  = std::max(fit.need, base + stage);
            } else {
                fit.need  = std::max(fit.need, base + stage);
            }
        }
    }
    return fit;
}

// What the pool of `dev` can lend the graph of a prompt pass of n_tokens tokens: up to the KV zone of those cells, behind the dense split's copies (the worst
// case: a pass that splits), less one block of the lend's rounding to whole blocks (kv_lend_round). The staging slots are not counted: they take what the
// graph leaves (ppx_stage_slots), the micro-batch is never cut for them. 0: no pool on dev. llama_context::sched_reserve sizes the micro-batch with it against
// the measured graph of each size plus its computed CUDA scratch (llama_moe_cache_graph_scratch_fork).
size_t llama_moe_cache_alias_room_fork(ggml_backend_dev_t dev, uint32_t n_tokens) {
    moe_cache * mc = g_cache;
    if (!mc || !alias_on()) {
        return 0;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev != dev) {
            continue;
        }
        const size_t cap  = kv_lend_potential(ap, (size_t) n_tokens);
        size_t bmax = 0;
        for (const auto & sg : ap.segs) { bmax = std::max(bmax, sg.B); }
        const size_t used = ap.mirror_b + bmax;
        return cap > used ? cap - used : 0;
    }
    return 0;
}

// A measure-only reserve of the load-time micro-batch fit (llama_context::sched_reserve) knows the real graph of that many tokens before any prompt ran. It goes into
// need_tok like a graph that ran, so the first prompt plans its passes from measured sizes: with the 1160-token warm-up call as the only point the planner took
// that graph's bytes per token (its fixed part spread over 1160 tokens) as the slope and cut a 16384-token prompt to 3 passes of 5472, although the load had measured
// the 16384-token graph (9962 MiB on a pool that lends 17139) and the next prompt, with two points, ran it as ONE pass (DS4.1 IQ2_XXS, 2026-10-26). The reserve's
// bytes are the graph's tensors plus the CUDA scratch its nodes take on this card (the caller adds llama_moe_cache_graph_scratch_fork); a real graph of the same
// size later replaces the point when it is bigger (alias, same rule as a graph that ran).
void llama_moe_cache_note_graph_fork(ggml_backend_dev_t dev, uint32_t n_tokens, size_t graph_bytes) {
    moe_cache * mc = g_cache;
    if (!mc || !alias_on() || n_tokens < 64 || graph_bytes == 0) {
        return;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev != dev) {
            continue;
        }
        size_t & nt = ap.need_tok[(int32_t) n_tokens];
        nt = std::max(nt, graph_bytes);
        return;
    }
}

static std::atomic<int> g_pp_k { -1 };   // staging slots the pass planner gave the prompt it sized last (-1: no plan, a prompt graph takes what fits)

// --moe log=f: what the blocks of the pool hold, per card: before a prompt's first pass, when the prompt is over (its borrowed blocks back) and 8 / 32 / 128
// decode steps later. Blocks holding a published expert, being uploaded, free, lent to a graph (its tensors, the CUDA scratch, the staging slots), and the
// card's free memory. A prompt must leave nothing behind: after it the lent count falls back to what a decode graph borrows and the refill
// (reheat_free_blocks) turns the free blocks into experts again.
static void pool_fill_log(moe_cache * mc, const char * when) {
    static const bool on = moe_log_has("f");
    if (!on) {
        return;
    }
    struct agg { ggml_backend_dev_t dev = nullptr; size_t n = 0, held = 0, up = 0, fr = 0, lent = 0, kv = 0; double b_held = 0, b_all = 0; };
    std::vector<agg> a;
    for (const auto & ps : mc->pool) {
        agg * g = nullptr;
        for (auto & x : a) { if (x.dev == ps.dev) { g = &x; } }
        if (g == nullptr) { a.emplace_back(); g = &a.back(); g->dev = ps.dev; }
        for (size_t b = 0; b < ps.owner.size(); ++b) {
            const int32_t o = ps.owner[b];
            g->n++;
            g->b_all += (double) ps.B;
            if (o == -2)                                     { g->lent++; }
            else if (o == -3)                                { g->kv++; }
            else if (o < 0)                                  { g->fr++; }
            else if (mc->layers[o].slot_in_flight[b])        { g->up++; }
            else if (mc->layers[o].slot_expert[b] >= 0)      { g->held++; g->b_held += (double) ps.B; }
            else                                             { g->fr++; }   // owned and empty (an adoption that did not complete)
        }
    }
    std::string s;
    for (const auto & g : a) {
        size_t fm = 0, tm = 0;
        if (g.dev != nullptr) { ggml_backend_dev_memory(g.dev, &fm, &tm); }
        s += tr_fmt(" %s: %zu of %zu blocks hold experts (%.2f of %.2f GB), %zu uploading, %zu free, %zu lent, %zu under the KV rows, card %.2f GB free;",
                g.dev ? ggml_backend_dev_name(g.dev) : "?", g.held, g.n, g.b_held / 1e9, g.b_all / 1e9, g.up, g.fr, g.lent, g.kv, fm / 1e9);
    }
    LLAMA_LOG_WARN("moe-cache: pool %s:%s\n", when, s.c_str());
}

// --moe keep-hot (default 1, 0 = off): a prompt pass borrows the first blocks of the pool (its graph, scratch and staging slots) and the experts in them
// are evicted, whatever their heat: the block's position decides, not the expert. After a long prompt decode then starts without part of its hottest
// experts and pays their upload in its first tokens (12k prompt + 32 tokens, passes as long as the context: Qwen3.8-Flash-Next 35.5 decode t/s against
// 51.0 with 2k passes, GLM-5.3-Flash 14.4 vs 16.3; the prompt itself 1092 vs 850 and 501 vs 355 t/s). Here, before the blocks [from, to) of a class
// are lent, each expert cached in them is moved, card to card inside the pool buffer, into the block outside the range that holds the coldest
// expert of the class (a free block first), as long as it scores higher than that expert (xscore, the scale the refill and eviction use). So the
// prompt evicts the coldest experts of the pool instead of the ones that happened to sit in front. A block copy runs at the card's memory speed
// (a 15 GB move is ~0.05 s on a 3090) against the PCIe refill of the same bytes (~1.3 s on x16). mc->mtx held.
// --moe log=a: where the alias spends host time (us, calls): the lend itself, the hot-expert moves, the release at the first decode steps, the sync it forces
static struct alias_cost {
    std::atomic<int64_t> lend_us{0}, lend_n{0}, keephot_us{0}, keephot_n{0}, release_us{0}, release_n{0}, sync_us{0}, sync_n{0}, drain_us{0}, evict_n{0};
    std::atomic<int64_t> where_calls{0}, where_range{0}, where_found{0}, where_inflight{0}, where_wrongdev{0};
} g_ac;
static void moe_where_count(int64_t range, int64_t found, int64_t inflight, int64_t wrongdev) {
    g_ac.where_calls++; g_ac.where_range += range; g_ac.where_found += found; g_ac.where_inflight += inflight; g_ac.where_wrongdev += wrongdev;
}
static void alias_cost_log(const char * why) {
    if (!moe_log_has("a")) { return; }   // not a static: the first call comes before the --moe options are set, a cached false hid the counters for the whole run
    LLAMA_LOG_WARN("moe-cache: alias cost (%s): lend %lld calls %.1f ms (scratch %lld calls %.1f ms, %lld splits asked) | keep-hot %lld calls %.1f ms | sync %lld calls %.1f ms | in-flight drain %.1f ms | release %lld calls %.1f ms | %lld blocks evicted\n",
            why, (long long) g_ac.lend_n.load(), g_ac.lend_us.load() / 1e3, (long long) g_scr_n.load(), g_scr_us.load() / 1e3, (long long) g_scr_splits.load(),
            (long long) g_ac.keephot_n.load(), g_ac.keephot_us.load() / 1e3,
            (long long) g_ac.sync_n.load(), g_ac.sync_us.load() / 1e3, g_ac.drain_us.load() / 1e3, (long long) g_ac.release_n.load(), g_ac.release_us.load() / 1e3,
            (long long) g_ac.evict_n.load());
    LLAMA_LOG_WARN("moe-cache: staging lookups: %lld range calls, %lld experts asked, %lld found on the card, %lld found but still in flight, %lld asked of the wrong card\n",
            (long long) g_ac.where_calls.load(), (long long) g_ac.where_range.load(), (long long) g_ac.where_found.load(), (long long) g_ac.where_inflight.load(), (long long) g_ac.where_wrongdev.load());
}
static void keep_hot_move(moe_cache * mc, ggml_backend_sched_t sched, moe_cache::alias_pool & ap, const moe_cache::alias_pool::seg & sg, int32_t from, int32_t to) {
    static const bool on = [] { const char * e = moe_opt("keep-hot"); return !e || atoi(e) != 0; }();
    if (!on || to <= from) {
        return;
    }
    const int64_t kh_t0 = ggml_time_us();
    struct kh_done { int64_t t0; ~kh_done() { g_ac.keephot_us += ggml_time_us() - t0; g_ac.keephot_n++; } } kh_guard{kh_t0};
    auto & ps = mc->pool[sg.pcls];
    struct ent { double sc; int32_t b; };
    std::vector<ent> hot, cold;   // experts in the range (hottest first), blocks behind it (free, then coldest first)
    for (int32_t b = from; b < to; ++b) {
        const int32_t o = ps.owner[b];
        if (o < 0) { continue; }
        layer_state & lo = mc->layers[o];
        const int32_t e = lo.slot_expert[b];
        if (e < 0 || lo.slot_in_flight[b] || lo.is_stream[b]) { continue; }
        hot.push_back({ xscore(mc, lo, e), b });
    }
    if (hot.empty()) {
        return;
    }
    for (int32_t b = to, nb = ps.n_active(); b < nb; ++b) {
        const int32_t o = ps.owner[b];
        if (o == -2 || o == -3) { continue; }
        if (o < 0) { cold.push_back({ -1.0, b }); continue; }
        layer_state & lo = mc->layers[o];
        if (lo.slot_in_flight[b] || lo.is_stream[b]) { continue; }
        const int32_t e = lo.slot_expert[b];
        cold.push_back({ e < 0 ? -1.0 : xscore(mc, lo, e), b });
    }
    std::sort(hot.begin(),  hot.end(),  [](const ent & a, const ent & b) { return a.sc > b.sc; });
    std::sort(cold.begin(), cold.end(), [](const ent & a, const ent & b) { return a.sc < b.sc; });
    // a decode graph may still read the blocks that are overwritten (their experts are evicted now)
    { const int64_t s0 = ggml_time_us(); ggml_backend_sched_synchronize(sched); g_ac.sync_us += ggml_time_us() - s0; g_ac.sync_n++; }
    const int64_t t0 = ggml_time_us();
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf) + sg.base;
    auto view = [&](int32_t b) {
        ggml_tensor t = {};
        t.type   = GGML_TYPE_I8;
        t.buffer = ap.buf;
        t.data   = base + (size_t) b * sg.B;
        t.ne[0]  = (int64_t) sg.B; t.ne[1] = t.ne[2] = t.ne[3] = 1;
        t.nb[0]  = 1; t.nb[1] = t.nb[2] = t.nb[3] = sg.B;
        return t;
    };
    size_t n_moved = 0;
    for (size_t i = 0; i < hot.size() && i < cold.size() && hot[i].sc > cold[i].sc; ++i) {
        const int32_t src = hot[i].b, dst = cold[i].b;
        const size_t  li  = (size_t) ps.owner[src];
        layer_state & ls  = mc->layers[li];
        const int32_t e   = ls.slot_expert[src];
        if (ps.owner[dst] >= 0) {
            evict_slot(mc, mc->layers[ps.owner[dst]], dst);
        }
        if (!pool_gate(ap, sg.base + (size_t) dst * sg.B, sg.base + (size_t) (dst + 1) * sg.B, "keep-hot")) { continue; }
        pool_take(mc, li, dst);
        if (led_on()) {
            const size_t d0 = sg.base + (size_t) dst * sg.B;
            const std::string who = tr_fmt("L%zu e%d block %d -> %d class %d", li, e, src, dst, sg.pcls);
            led_line(led_card(ap), "keep-hot", d0, d0 + sg.B, who);
            led_check_range(ap, "keep-hot", d0, d0 + sg.B, who, true, false);
        }
        ggml_tensor s = view(src), d = view(dst);
        pool_enter(ap, sg.base + (size_t) dst * sg.B, sg.base + (size_t) (dst + 1) * sg.B, true);
        ggml_backend_tensor_copy(&s, &d);
        pool_leave(ap, sg.base + (size_t) dst * sg.B, sg.base + (size_t) (dst + 1) * sg.B, true);
        ls.slot_expert[dst]   = e;
        ls.expert_slot[e]     = dst;
        ls.slot_last_use[dst] = ls.slot_last_use[src];
        ls.slot_expert[src]   = -1;   // the range's eviction below finds the block empty
        set_table_entry(ls.pub, e, dst);
        n_moved++;
    }
    static int logged = 0;
    if (n_moved > 0 && logged++ < 16) {
        LLAMA_LOG_WARN("moe-cache: keep-hot %s: %zu of %zu experts in the %d blocks a prompt borrows moved to colder blocks in %.1f ms (%.2f GB)\n",
                ggml_backend_dev_name(ap.dev), n_moved, hot.size(), to - from, (ggml_time_us() - t0) / 1e3, (double) n_moved * (double) sg.B / 1e9);
    }
}


// ---- KV cache at the tail of the pool (--moe kv-tail=1, fork/docs/kv-tail.md) ----
static std::mutex                  g_kvt_mtx;
static std::vector<llama_kv_tail_fork> g_kvt;
static struct { std::atomic<size_t> claim_n{0}, release_n{0}, evicted{0}; } g_kvs;

size_t llama_kv_cache_tail_ratio_fork(const void * kv);   // llama-kv-cache.cpp: the tokens one row of a record covers (declared here, no header edit)
// Which records of a card are laid out at full size in front of the growing rows (fixed) and which grow with the zone, one combined row per cell. A record whose rows
// cover fewer tokens than the largest is a sliding-window ring (its cells are reused): fixed. A compressed cache (DSV4: one row per `ratio` tokens, written at row
// pos / ratio) covers the whole context with fewer rows; until 2026-10-26 the rule "fewer cells than the largest" took it for a ring and laid it out at full size:
// DeepSeek 4.1's ratio-2 cache held 1920 MiB of the x16 card's pool at every context (524288 rows x 3840 B, 9.4% of the pool) and every zone move copied it whole.
// It now grows with the zone: its row i sits in combined row i (all records share the row index), so it takes its full row per cell and leaves the rows past its
// own end unused; that is less than the full size below rows_max cells of context and up to `ratio` times more at the model's maximum. A compressed cache
// whose full size is at most one growth step of its rows stays fixed (DSV4 V4's ratio-128 cache, 8192 rows): growing would never make it smaller.
static bool kv_rec_fixed(size_t cells_max, size_t ratio, size_t tokens_max, size_t step) {
    return ratio > 1 ? cells_max <= step : cells_max < tokens_max;
}

static std::atomic<bool> g_kv_pool_expected { true };
void llama_moe_cache_kv_pool_expected_fork(bool on) { g_kv_pool_expected = on; }
bool llama_moe_cache_kv_pool_expected_fork() { return g_kv_pool_expected.load(); }
// The records a context registered while it expected an expert pool (the target): they belong to that pool's tail. The flag above is set by every context at its
// creation, so it is captured per record here and asked per cache afterwards. A draft context (no pool, ctx_other) created after the target set it to false,
// and with one global flag that broke the target twice (DeepSeek 4.1 + DSpark draft, 2026-10-26):
//  - the target's reserves (get_size_reserve) sized its KV at the full context instead of one growth step: the prompt graph grew to 13.5 GiB on CUDA0 and the
//    micro-batch fell from 16384 to 640 (prompt 59 -> 22-37 t/s);
//  - the draft's first decode (at load, before the target's first request starts the pool) took the target's records into a fixed full-context arena:
//    7.05 GiB for 1048576 tokens, the expert cache 28.6 instead of 41.3 GiB.
// Only the target takes its records into an arena, once its pool started and left them (kv_arena_start); the pool's detach only checks them.
static std::mutex             g_kvt_pb_mtx;   // only around this set: get_size_reserve asks it from graph reserves
static std::set<const void *> g_kvt_pool_bound;
static bool kvt_pool_bound(const void * owner) { std::lock_guard<std::mutex> lk(g_kvt_pb_mtx); return g_kvt_pool_bound.count(owner) > 0; }
bool llama_moe_cache_kv_pool_bound_fork(const void * owner) { return kvt_pool_bound(owner); }   // declared where it is used (llama-kv-cache.cpp), no header edit
void llama_moe_cache_kv_tail_add_fork(const llama_kv_tail_fork & rec) {
    std::lock_guard<std::mutex> lk(g_kvt_mtx);
    g_kvt.push_back(rec);
    if (g_kv_pool_expected) { std::lock_guard<std::mutex> lk2(g_kvt_pb_mtx); g_kvt_pool_bound.insert(rec.owner); }
}
void llama_moe_cache_kv_tail_remove_fork(void * owner) {
    std::lock_guard<std::mutex> lk(g_kvt_mtx);
    { std::lock_guard<std::mutex> lk2(g_kvt_pb_mtx); g_kvt_pool_bound.erase(owner); }
    g_kvt.erase(std::remove_if(g_kvt.begin(), g_kvt.end(), [&](const llama_kv_tail_fork & r) { return r.owner == owner; }), g_kvt.end());
    if (moe_cache * mc = g_cache) {
        for (auto & ap : mc->alias) {
            ap.kvt.erase(std::remove_if(ap.kvt.begin(), ap.kvt.end(), [&](const moe_cache::alias_pool::kvt_run & r) { return r.owner == owner; }), ap.kvt.end());
        }
    }
}
void llama_moe_cache_kv_tail_report_fork(const char * when) {
    for (const auto & st : llama_moe_cache_kv_tail_stats_fork(false)) {
        LLAMA_LOG_WARN("moe-cache: %s %s: KV %zu cells (%.1f MiB), expert cache %.2f GiB in %zu of %zu blocks (%zu free, %zu lent to the graph, %zu under the KV)\n", when, st.dev.c_str(),
                st.kv_cells, st.kv_bytes / 1048576.0, st.moe_bytes / 1073741824.0, st.blocks - st.blocks_kv - st.blocks_lent, st.blocks, st.blocks_free, st.blocks_lent, st.blocks_kv);
    }
}
void llama_moe_cache_kv_tail_detach_fork() {
    std::lock_guard<std::mutex> lk(g_kvt_mtx);
    for (auto & r : g_kvt) { if (r.detach && kvt_pool_bound(r.owner)) { r.detach(); } }   // the pool's own records: a draft's KV in its arena may hold cells
}

// ---- the KV zone of a card (kv-tail, fork/docs/kv-grow.md) ----
// [kv_start, kv_start + kv_h * kv_row) of the pool buffer holds the combined rows of the card's records (kv_row = the sum of their rows, row c = cell c). The
// blocks under it belong to the KV (owner -3): experts in blocks the zone takes are evicted (uploads into them dropped or awaited, as the compute alias does),
// blocks it leaves go back to the free list.
static void kv_blocks_apply(moe_cache * mc, moe_cache::alias_pool & ap, size_t lo, size_t hi, bool do_take, bool do_give,
                            size_t & n_take, size_t & n_give, size_t & n_evict) {
    std::vector<std::pair<size_t, size_t>> rng;
    if (hi > lo) { rng.push_back({ lo, hi }); }
    for (auto & sg : ap.segs) {
        auto & ps = mc->pool[sg.pcls];
        std::vector<int32_t> take, give;
        for (int32_t b = 0; b < sg.n; ++b) {
            const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
            bool want = false;
            for (const auto & q : rng) { if (q.first < b1 && q.second > b0) { want = true; break; } }
            const int32_t o = ps.owner[b];
            if (want) { if (o != -3 && do_take) { take.push_back(b); } }
            else if (o == -3 && do_give) { give.push_back(b); }
        }
        if (!give.empty()) {
            std::lock_guard<std::mutex> lk(mc->mtx);
            n_give += poolx_zone_give(ps, give);
        }
        if (take.empty()) { continue; }
        const int32_t want_pcls = sg.pcls;
        auto in_zone = [&](const upload_job & j) {
            return j.layer_idx < mc->layers.size() && mc->layers[j.layer_idx].pcls == want_pcls && std::binary_search(take.begin(), take.end(), j.slot);
        };
        auto drop_job = [&](const upload_job & j) {
            layer_state & lo = mc->layers[j.layer_idx];
            lo.slot_in_flight[j.slot] = false;
            if (j.expert >= 0 && j.expert < (int32_t) lo.queued.size()) { lo.queued[j.expert] = 0; }
        };
        for (int spin = 0; ; ++spin) {
            bool busy = false;
            {
                std::lock_guard<std::mutex> wlk(mc->wmtx);
                std::lock_guard<std::mutex> lk(mc->mtx);
                for (auto it = mc->todo.begin(); it != mc->todo.end(); ) {
                    if (in_zone(*it)) { drop_job(*it); it = mc->todo.erase(it); } else { ++it; }
                }
                for (size_t k = 0; k < mc->done.size(); ) {
                    if (in_zone(mc->done[k])) { drop_job(mc->done[k]); mc->done[k] = mc->done.back(); mc->done.pop_back(); } else { ++k; }
                }
                for (int32_t b : take) {
                    const int32_t o = ps.owner[b];
                    if (o < 0 || !mc->layers[o].slot_in_flight[b]) { continue; }
                    layer_state & lo = mc->layers[o];
                    bool adoption = false;
                    for (size_t i = 0; i < lo.adopted.size(); ++i) {
                        const int32_t e = lo.adopted[i];
                        if (lo.adopt_slot[e] == b) {
                            lo.adopt_slot[e] = -1; lo.adopt_mask[e] = 0;
                            lo.adopted.erase(lo.adopted.begin() + (ptrdiff_t) i);
                            lo.slot_in_flight[b] = false;
                            adoption = true;
                            break;
                        }
                    }
                    if (!adoption) { busy = true; break; }   // a worker is copying into it right now
                }
            }
            if (!busy) { break; }
            GGML_ASSERT(spin < 200000 && "kv-tail: an upload into the blocks taken for the KV never finished");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        std::lock_guard<std::mutex> lk(mc->mtx);
        poolx_zone_take(ps, take, &sg.blocked, [&](int32_t o, int32_t b) {
            layer_state & lo = mc->layers[o];
            evict_slot(mc, lo, b);
            lo.n_own--;
            n_evict++;
        });
        n_take += take.size();
    }
}


#include "llama-kvzone-fork.h"
static size_t kv_bmax(const moe_cache::alias_pool & ap) { return kvz_bmax(ap); }
static size_t kv_lend_potential(const moe_cache::alias_pool & ap, size_t min_cells) { return kvz_lend_potential(ap, min_cells); }
static size_t kv_gap_fit(const moe_cache::alias_pool & ap, size_t bmax, size_t lo_min, size_t row) { return kvz_gap_fit(ap, bmax, lo_min, row); }
static size_t kv_gap_place(const moe_cache::alias_pool & ap, size_t bmax, size_t lo_min, size_t want, size_t size) { return kvz_gap_place(ap, bmax, lo_min, want, size); }
static void kv_zone_check(const moe_cache::alias_pool & ap, size_t lend_end, const char * who) {   // a violation used to show up as '!!!!' or a NaN many tokens later; now it stops here, with the numbers
    if (const char * bad = kvz_bad(ap, lend_end)) {
        GGML_ABORT("kv-zone invariant (%s) on %s: %s: zone [%zu, %zu) fixed %zu h %zu row %zu, lend end %zu, tail_end %zu",
                   who, ggml_backend_dev_name(ap.dev), bad, ap.kv_start, ap.kv_start + ap.kv_fixed + ap.kv_h * ap.kv_row, ap.kv_fixed, (size_t) ap.kv_h, ap.kv_row, lend_end, ap.tail_end);
    }
}

// A graph's lend grows with the cells too: its copies of the attention masks (pass tokens x cells x 2 bytes for an f16 mask) are inputs the scheduler copies into
// each card's compute buffer. The bytes per pass token per cell are the host-buffer fit's (llama-context.cpp, the same inputs on the host side), set at startup.
// need_tok holds the most a size borrowed, measured at some cells: at more cells it needs that much more (MiMo 255k: a 2368-token pass carries a 1.15 GB mask
// on each card, which a size measured at 32k cells did not have).
static std::atomic<double> g_mask_bpc { 0.0 };
void llama_moe_cache_mask_bpc_fork(double bytes_per_token_cell) { g_mask_bpc.store(std::max(0.0, bytes_per_token_cell)); }
static size_t need_at_cells(const moe_cache::alias_pool & ap, int32_t n_tokens, size_t need, size_t cells) {
    const auto it = ap.need_cells.find(n_tokens);
    const size_t c0 = it != ap.need_cells.end() ? it->second : cells;
    const double pc = g_mask_bpc.load() * (double) n_tokens + (double) (n_tokens < 64 ? ap.kv_lend_pc_dec : ap.kv_lend_pc);   // + the f16 copies of quantized K/V
    return cells > c0 ? need + (size_t) (pc * (double) (cells - c0)) : need;
}

// where a lend of `bytes` from the start of the buffer really ends: it takes whole blocks of every size class its range overlaps (the classes are stacked in the buffer, each with its own
// block size), so the end is the furthest of base + ceil((bytes - base) / B) * B over those classes. Rounding by the LARGEST block size (kv_bmax) is wrong when a smaller-block class
// sits at the start (MiMo: class 0 has 9.2 MB blocks, the largest class 12.6 MB sits at the end: a 10.8 MB lend ended at 18.3 MB, not 12.6 MB)
static size_t kv_lend_round(const moe_cache::alias_pool & ap, size_t bytes) { return poolx_lend_round(ap, bytes); }

// n bytes from `from` to `to` inside the pool buffer, on the card (a device copy at ~800 GB/s: 1 GB of KV in ~2.5 ms on an RTX 3090), in chunks that never
// overlap their own source
static void kv_move_bytes(moe_cache::alias_pool & ap, size_t from, size_t to, size_t n) {
    if (n == 0 || from == to) { return; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    const size_t d = from > to ? from - to : to - from;
    const size_t chunk = std::min<size_t>(d, (size_t) 256 << 20);
    auto cp = [&](size_t s0, size_t t0, size_t len) {
        ggml_tensor a = {}, b = {};
        for (ggml_tensor * t : { &a, &b }) {
            t->type = GGML_TYPE_I8; t->buffer = ap.buf;
            t->ne[0] = (int64_t) len; t->ne[1] = t->ne[2] = t->ne[3] = 1;
            t->nb[0] = 1; t->nb[1] = t->nb[2] = t->nb[3] = len;
        }
        a.data = base + s0; b.data = base + t0;
        ggml_backend_tensor_copy(&a, &b);
    };
    if (to > from) {   // moving up: copy from the end
        for (size_t left = n; left > 0; ) { const size_t len = std::min(chunk, left); left -= len; cp(from + left, to + left, len); }
    } else {
        for (size_t done = 0; done < n; ) { const size_t len = std::min(chunk, n - done); cp(from + done, to + done, len); done += len; }
    }
}

static std::string g_alias_err;   // why the last llama_moe_cache_alias_fork refused its graph (empty: it did not)
static std::atomic<size_t> g_refuse_need { 0 }, g_refuse_room { 0 };   // the last refused lend: what the graph needed, what the pool could lend beside the KV
// the last refused lend's need and room (bytes), once: the dense split's pass loop shrinks the refused pass by room / need and runs it again (0: none refused)
bool llama_moe_cache_refusal_fork(size_t * need, size_t * room) {
    *need = g_refuse_need.exchange(0);
    *room = g_refuse_room.exchange(0);
    return *need > 0;
}
static std::atomic<int> g_kv_epoch{0};   // bumped when a KV zone moves: graphs built before point at the old place (no reuse)
static int g_kv_epoch_built = 0;

// lay the zone out for h cells behind a lend of l bytes (rounded up to the largest block): take the new blocks first (their experts out before rows land there),
// copy the rows in use when the zone moves, give back what it left, point the records at it. Called before a graph is built (prepare, request start) or between
// a graph's allocation and its inputs (moe_alias, which then has the graph built again). No allocation: the zone is part of the one pool buffer.
// Where the KV zone may lie: anywhere behind the lend up to the end of the expert blocks, over blocks of any class (the classes lie back to back: no zero blocks
// since 2026-10-24, kvz_gaps), its start on a block edge of the class it falls in (kvz_block_ceil: a start inside a block of a smaller class would share that block
// with an expert or the lend). The zone is placed nearest to `want`, or, when it does not fit, made as large as the room and the cache refuses the cells past it.
// --moe poison: after a layout change of a pool (the KV zone placed or moved, lent blocks given back) every block that holds nothing (free: no expert, not lent,
// not under the zone) is filled with poison (5), so a kernel that reads a block it does not own reads 5s, not the bytes some buffer left there (user 2026-10-24:
// "poison reads after each major buffer reshuffling"; the buffers move a lot). Each run of free blocks is re-checked under the cache lock while its byte range is
// entered for writing (an upload DMA into it waits); when the lock is busy the fill is skipped this time, never done over a block that was just claimed.
static void poison_free_blocks(moe_cache * mc, const moe_cache::alias_pool & ap) {
    if (!poison_on()) { return; }
    for (const auto & sg : ap.segs) {
        const auto & ps = mc->pool[sg.pcls];
        int32_t b = 0;
        while (b < sg.n) {
            if (ps.owner[b] != -1) { ++b; continue; }
            int32_t e = b;
            while (e < sg.n && ps.owner[e] == -1) { ++e; }
            const size_t lo = sg.base + (size_t) b * sg.B, hi = sg.base + (size_t) e * sg.B;
            pool_enter(ap, lo, hi, true);
            if (mc->mtx.try_lock()) {
                std::vector<std::pair<size_t, size_t>> runs;   // still free now that no DMA can start in the range
                for (int32_t x = b; x < e; ) {
                    if (ps.owner[x] != -1) { ++x; continue; }
                    int32_t y = x;
                    while (y < e && ps.owner[y] == -1) { ++y; }
                    runs.push_back({ sg.base + (size_t) x * sg.B, sg.base + (size_t) y * sg.B });
                    x = y;
                }
                mc->mtx.unlock();
                for (const auto & r : runs) { pool_fill(ap, r.first, r.second, 5); }
            }
            pool_leave(ap, lo, hi, true);
            b = e;
        }
    }
}

static bool kv_layout(moe_cache * mc, moe_cache::alias_pool & ap, size_t h, size_t l, size_t cells, bool dry, size_t lend_min = 0) {
    if (!dry) { ap.refused_b = 0; }   // the zone is laid out again: an earlier refusal no longer holds
    const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
    l = kv_lend_round(ap, l);   // the lend's end on its block edge (poolx_lend_round)
    lend_min = std::min(l, kv_lend_round(ap, lend_min));   // the zone may go back down to the end of the lend, never into it
    // The attention reads the growing records up to the cells in use rounded up to the pad (256 with flash attention); those pad rows must be the zone's
    // (kv_tail_pad_zero zeroes them only inside the zone). A horizon clamped to the bare cell count left them in the expert blocks behind the zone: MiMo,
    // 15 GiB free per card, 11673 cells -> the zone ended at cell 11673, the last query chunk (rows 11520+) read expert bytes as K/V, NaN in layer 0, '!!!!'
    // from ~9k tokens up (2026-10-18). Whatever clamps h (room, the pool end, the zero-block rule below), it never goes under this.
    // cmax: the most cells any growing record has (a compressed cache has fewer rows than the context: it must not cap the others' horizon at its own rows)
    size_t pad = 1, cmax = 0, need = cells;   // need: the cells in use or planned for the prompt being laid out, whichever is more
    for (const auto & r : ap.kvt) { if (!r.fixed) { pad = std::max(pad, r.pad); cmax = std::max(cmax, r.cells_max); need = std::max(need, r.plan); } }
    const size_t h_min = std::min((need + pad - 1) / pad * pad, cmax);
    const size_t h_pool = ap.kv_row > 0 ? kv_gap_fit(ap, bmax, lend_min, ap.kv_row) : (size_t) -1;   // the largest zone any gap can hold
    if (led_on()) { LLAMA_LOG_WARN("kv-gap %s: h %zu h_min %zu h_pool %zu lend_min %.1f MiB want %.1f MiB fixed %.1f MiB row %zu tail_end %.1f MiB segs %zu\n", led_card(ap), h, h_min, h_pool, lend_min / 1048576.0, l / 1048576.0, ap.kv_fixed / 1048576.0, ap.kv_row, ap.tail_end / 1048576.0, ap.segs.size()); }
    if (!dry && h_pool < h_min) {   // no gap behind this lend holds the padded rows of the cells in use: the pass is refused here, not written and aborted in set_input_k_idxs (Qwen3.6 on PC1 with 8 GiB, 13720 cells, gap of 13780, padded read 13824, 2026-10-18)
        g_alias_err = "kv-tail: " + std::to_string(cells) + " cells of KV (read as " + std::to_string(h_min) + ") do not fit in any gap behind a lend of " + std::to_string(l / 1048576) + " MiB (the largest holds " + std::to_string(h_pool) + ")";
        const size_t short_b = (h_min - h_pool) * ap.kv_row;   // the lend must be this much smaller for the rows to fit
        g_refuse_need.store(l);
        g_refuse_room.store(l > short_b ? l - short_b : 0);
        return false;   // the zone stays where it is: moving it into a gap smaller than the cells in use cut rows off the KV (W10 OLMoE, 19712 cells in use, zone 20258 -> 19608)
    }
    h = std::min(std::max(h, std::min(h_min, h_pool)), h_pool);   // never past what a gap holds: there the cache refuses the cells (llama_moe_cache_kv_tail_covers_fork checks the padded count)
    {
        const size_t ls = kv_gap_place(ap, bmax, lend_min, l, ap.kv_fixed + h * ap.kv_row);
        if (ls != (size_t) -1) { l = ls; }
    }
    if (h == ap.kv_h && l == ap.kv_start) { return false; }
    if (dry) { return true; }
    const size_t lo_old = ap.kv_start, h_old = ap.kv_h, lo = l, hi = l + ap.kv_fixed + h * ap.kv_row;
    const bool   first  = h_old == 0;   // nothing laid out yet: no rows to move, the fixed part is zeroed below
    const size_t n_used = first ? 0 : ap.kv_fixed + std::min(cells, std::min(h, h_old)) * ap.kv_row;   // the fixed part and the growing rows in use: one contiguous range
    size_t n_take = 0, n_give = 0, n_evict = 0;
    pool_move_begin(ap, std::min(lo, lo_old), std::max(hi, h_old > 0 ? led_kv_end(ap) : hi));   // no DMA runs in either place, none starts until the move ends
    kv_blocks_apply(mc, ap, lo, hi, true, false, n_take, n_give, n_evict);
    if (led_on()) {
        const std::string who = tr_fmt("%zu -> %zu cells, %zu in use, row %zu B, fixed %zu B", h_old, h, cells, ap.kv_row, ap.kv_fixed);
        led_line(led_card(ap), "kv-zone", lo, hi, who);
        if (lo != lo_old && n_used > 0) { led_line(led_card(ap), "kv-move", lo, lo + n_used, tr_fmt("rows from [%zu, %zu)", lo_old, lo_old + n_used)); }
    }
    if (lo != lo_old) { kv_move_bytes(ap, lo_old, lo, n_used); }
    kv_blocks_apply(mc, ap, lo, hi, false, true, n_take, n_give, n_evict);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    auto zero = [&](size_t z0, size_t z1) {   // bytes [z0, z1) of the pool to 0 (a device memset)
        if (z1 <= z0) { return; }
        ggml_tensor z = {};
        z.type = GGML_TYPE_I8; z.buffer = ap.buf; z.data = base + z0;
        z.ne[0] = (int64_t) (z1 - z0); z.ne[1] = z.ne[2] = z.ne[3] = 1;
        z.nb[0] = 1; z.nb[1] = z.nb[2] = z.nb[3] = z1 - z0;
        ggml_backend_tensor_memset(&z, 0, 0, z1 - z0);
    };
    if (first && ap.kv_fixed > 0) { zero(lo, lo + ap.kv_fixed); }   // the fixed records are read up to their padded cells from the first token on: zero, once (the attention multiplies masked rows too)
    for (auto & r : ap.kvt) {
        if (!r.attach) { continue; }
        if (r.fixed) { r.attach(ap.buf, base + lo + r.prefix, r.row); }
        else         { r.attach(ap.buf, base + lo + ap.kv_fixed + r.prefix, ap.kv_row); }
    }
    ap.kv_h = h; ap.kv_start = lo; ap.span = lo;
    kv_zone_check(ap, lend_min, "kv_layout");
    led_map(ap, "kv_layout");
    pool_fill(ap, lo + std::max(n_used, ap.kv_fixed), hi, 4);   // --moe poison: rows nobody wrote yet are 4 (the fixed part is zeroed above)
    // A compressed record shares the combined rows with records that fill them faster (DSV4: its row i covers tokens 2i, 2i+1): its pad rows, which its attention
    // reads past its own rows in use, lie BELOW the cells in use of the others, where the pad window (kv_tail_pad_zero, whole rows from the largest count up) never
    // comes. Those rows must be finite from the moment the zone holds them, so every byte the zone gets here beyond the rows it brought along is zeroed (also over a
    // poison fill: they are read by design). One device memset of the zone's free rows per layout change (10240 cells x 5120 B = 50 MiB on DeepSeek 4.1's x16 card).
    for (const auto & r : ap.kvt) { if (!r.fixed && r.ratio > 1) { zero(lo + std::max(n_used, ap.kv_fixed), hi); break; } }
    pool_move_end(ap);
    ap.kv_zhi = 0;   // the pad window (rows past the cells in use that the attention reads) is zeroed again
    ap.blocked = ap.segs.empty() ? 0 : ap.segs[0].blocked;
    size_t held = 0;
    for (auto & sg : ap.segs) { for (int32_t b = 0; b < sg.n; ++b) { if (mc->pool[sg.pcls].owner[b] == -3) { held++; } } }
    ap.kv_blocks = held;
    for (auto & sg : ap.segs) { mc->pool[sg.pcls].active_end = -1; }   // the zone sits between the lend and the experts: scans cover the whole class (they skip -3)
    g_kvs.claim_n += n_take; g_kvs.release_n += n_give; g_kvs.evicted += n_evict;
    if (n_give > 0) { mc->kv_gave_back = true; }
    if (lo != lo_old) { g_kv_epoch++; }
    if (moe_log_has("k") || lo != lo_old) {
        LLAMA_LOG_WARN("moe-cache: kv-tail %s: zone %zu -> %zu cells at %.1f -> %.1f MiB (%zu B per cell, %zu cells in use%s), %zu blocks taken (%zu experts evicted), %zu given back\n",
                ggml_backend_dev_name(ap.dev), h_old, h, lo_old / 1048576.0, lo / 1048576.0, ap.kv_row, cells,
                lo != lo_old ? tr_fmt(", %.1f MiB of rows moved", n_used / 1048576.0).c_str() : "", n_take, n_evict, n_give);
    }
    {   // no block left for an expert on any card that holds a KV zone (the blocks lent to the graph are not the experts' either)
        bool any = false, usable = false;
        for (auto & a2 : mc->alias) {
            if (a2.kvt.empty()) { continue; }
            any = true;
            for (auto & sg : a2.segs) {
                const auto & ps = mc->pool[sg.pcls];
                for (int32_t b = 0; b < sg.n && !usable; ++b) { if (ps.owner[b] != -3 && ps.owner[b] != -2) { usable = true; } }
            }
        }
        const bool now = any && !usable;
        if (now != mc->starved.load()) {
            mc->starved = now;
            LLAMA_LOG_WARN("moe-cache: kv-tail: %s\n", now ? "the KV holds every expert block: the cache logic is off, experts are streamed" : "expert blocks are free again: the cache logic is on");
        }
    }
    return true;
}

// the horizon of every card: the cells in use (or planned for the prompt) rounded to the pad, plus half a step at least; it grows by kv_step before the rows are
// needed, at a point where no graph is built (a prompt call plans all its tokens up front, a decode grows it half a step ahead), and shrinks only at a request
// start. Until 2026-10-17 it kept TWICE the cells in use: at a 255k-token MiMo prompt 4.7 GB of the two pools sat under empty rows (about 550 expert blocks).
// The lend in front of it is the most a graph of this size needed before (moe_alias remembers it), so a known size never moves the zone after its graph is built.
static bool kv_tail_plan(moe_cache * mc, size_t plan_extra, bool reset_plan, bool allow_shrink, int32_t n_tokens, bool dry) {
    bool any = false;
    for (auto & ap : mc->alias) {
        if (ap.kvt.empty() || ap.kv_row == 0) { continue; }
        size_t eff = 0, cells = 0, cells_max = 0;   // cells_max: the most cells of any growing record (a compressed cache's fewer rows never cap the others)
        for (auto & r : ap.kvt) {
            if (r.fixed) { continue; }   // laid out at full size
            const size_t c = r.rows ? r.rows() : 0;
            if (!dry) {
                r.cells = c;
                if (plan_extra > 0) { r.plan = c + plan_extra; } else if (reset_plan) { r.plan = 0; }
            }
            const size_t plan = !dry ? r.plan : (plan_extra > 0 ? c + plan_extra : (reset_plan ? 0 : r.plan));
            size_t e = std::max(c, plan);
            e = e > 0 ? (e + r.pad - 1) / r.pad * r.pad : 0;
            eff = std::max(eff, e);
            cells = std::max(cells, c);
            cells_max = std::max(cells_max, r.cells_max);
        }
        const size_t step = std::max<size_t>(256, ap.kv_step);
        size_t h = (allow_shrink || ap.kv_h == 0) ? step : ap.kv_h;
        while (eff + step / 2 > h) { h += step; }
        const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
        // the zone stays where it is unless the lend of this graph size reaches it. Rounding its own start again to the block grids of all classes (they do not
        // line up) moved it one block up at every call: 584 MB of rows copied and the graph built again for every decoded token, until it crept over a zero
        // block (2026-10-18, pool ledger). The lend's end is rounded as kv_layout lays it out (the block edge it ends on).
        size_t lr = ap.kv_start;
        size_t lend_min = kv_lend_round(ap, ap.dec_need);   // the least any graph borrows (a decode graph's, measured): the zone may go back down to here
        if (n_tokens > 0) {
            const auto it = ap.need_tok.find(n_tokens);
            if (it != ap.need_tok.end()) {
                const size_t nb = kv_lend_round(ap, need_at_cells(ap, n_tokens, it->second, eff));
                lr = std::max(lr, nb); lend_min = std::max(lend_min, nb);
            }
        }
        // the end of the pool is a hard limit (nothing past tail_end is the KV's: the claim check cannot even see it). When the cells this call needs do not fit
        // behind the lend, the lend shrinks (the call's passes get smaller: pp_cap_to_pool sees the zone), never below what a decode graph borrows.
        if (ap.tail_end <= lr + ap.kv_fixed + eff * ap.kv_row) {
            const size_t floor_l = kv_lend_round(ap, ap.dec_need);   // what a decode graph borrows (measured), on its block edge
            const size_t zone_b  = ap.kv_fixed + eff * ap.kv_row;
            if (ap.tail_end > zone_b + floor_l) { lr = std::min(lr, std::max(floor_l, kvz_block_floor(ap, ap.tail_end - zone_b))); }
        }
        const size_t hard = ap.tail_end > lr + ap.kv_fixed ? (ap.tail_end - lr - ap.kv_fixed) / ap.kv_row : 0;
        const size_t room = hard;   // the zone may take every expert block behind the lend (KV and micro-batch first, the cache gets what is left)
        h = std::min({ h, cells_max, std::max(room, eff), hard });
        if (eff > room && !dry) {
            LLAMA_LOG_WARN("moe-cache: kv-tail %s: %zu cells do not fit behind a lend of %.1f MiB (room for %zu, %zu at most): the experts give up their last blocks%s\n",
                    ggml_backend_dev_name(ap.dev), eff, lr / 1048576.0, room, hard, eff > hard ? "; the pool cannot hold this call (the cache refuses the cells past it)" : "");
        }
        // the zone starts at the lend rounded up to whole blocks: from the unrounded end of the graph (12566400 B on MiMo, 16 KB short of 12 MiB) its first block was also the
        // last lent block (owner -2), never claimable for the KV (owner -3): the first request after the server's own warm-up decode aborted
        const bool moved = kv_layout(mc, ap, h, lr, cells, dry, lend_min);
        if (moved && !dry) { poison_free_blocks(mc, ap); }   // --moe poison: the zone's old place holds nothing now
        any = moved || any;
    }
    return any;
}

// the reason, when a card that keeps its KV in the pool has no zone at all after the layout for this graph (kv_h == 0: the fixed part and one cell fit in no gap, or the pool is
// smaller than the graph's lend plus the zone); nullptr when every zone is laid out. The text is for the person who started the server: what to change.
const char * llama_moe_cache_kv_tail_unplaced_fork(int32_t n_tokens) {
    static thread_local std::string msg;
    moe_cache * mc = g_cache;
    if (!mc) { return nullptr; }
    for (auto & ap : mc->alias) {
        if (ap.kvt.empty() || ap.kv_row == 0 || ap.kv_h > 0) { continue; }
        msg = tr_fmt("kv-tail %s: the pool of %.1f MiB cannot hold the KV zone for a %d-token graph (fixed part %.1f MiB, %zu B per cell, the graph borrows %.1f MiB): leave more VRAM free, or use a smaller context or micro-batch",
                ggml_backend_dev_name(ap.dev), ap.tail_end / 1048576.0, (int) n_tokens, ap.kv_fixed / 1048576.0, ap.kv_row, kv_lend_round(ap, std::max(ap.dec_need, (size_t) 1)) / 1048576.0);
        return msg.c_str();
    }
    return nullptr;
}

// a record's unit: its row and its place in the combined row are multiples of it (as tm_layout_fork in llama-kv-cache.cpp lays its rows out: 256 bytes for float K/V,
// lcm(16, block bytes) for quantized K/V, whose cell strides must be whole blocks for the attention's f16 conversion)
static size_t kv_type_unit(ggml_type t) { return ggml_is_quantized(t) ? std::lcm<size_t>(16, ggml_type_size(t)) : 256; }
static size_t kv_rec_unit(ggml_type tk, ggml_type tv) { return std::lcm(kv_type_unit(tk), kv_type_unit(tv)); }

// where the records sit in the zone: fixed ones (fewer cells than the largest: a sliding-window ring) first, each at full size on 256 bytes (F bytes in all); the
// growing ones side by side in one combined row of R bytes, each at a multiple of its unit, R rounded to the least common multiple of their units. All float: the
// plain sum (as before quantized tiers).
using kvt_runs = std::vector<moe_cache::alias_pool::kvt_run>;
static void kv_zone_offsets(kvt_runs & kvt, size_t & F, size_t & R) {
    F = 0; R = 0;
    size_t u_all = 256;
    for (auto & kr : kvt) {
        if (kr.fixed) { kr.prefix = F; F += (kr.cells_max * kr.row + 255) / 256 * 256; continue; }
        kr.prefix = (R + kr.unit - 1) / kr.unit * kr.unit;
        R = kr.prefix + kr.row;
        u_all = std::lcm(u_all, kr.unit);
    }
    R = (R + u_all - 1) / u_all * u_all;
}

// the records as KV tier `tier` holds them: 0 the types each record was created with, 1 q8_0 K and V, 2 q4_0 K and V (convertible records only: an indexer's key
// cache keeps its own), with their rows, units and places in the zone (F, R). false: a record's rows are not whole blocks of those types.
static bool kv_records_as(const kvt_runs & kvt, int tier, kvt_runs & out, size_t & F, size_t & R) {
    out = kvt;
    for (auto & r : out) {
        if (!r.convertible) { continue; }
        const ggml_type tk = tier == 3 ? r.tk : tier == 1 ? GGML_TYPE_Q8_0 : tier == 2 ? GGML_TYPE_Q4_0 : r.tk0;   // 3: the types now (the RAM part moves layers, not types)
        const ggml_type tv = tier == 3 ? r.tv : tier == 1 ? GGML_TYPE_Q8_0 : tier == 2 ? GGML_TYPE_Q4_0 : r.tv0;
        const size_t row = r.layout(tk, tv, &r.out_next, nullptr);
        if (row == 0) { return false; }
        r.tk = tk; r.tv = tv; r.row = row; r.unit = kv_rec_unit(tk, tv); r.bytes = row * r.cells_max;
        r.bpv = (float) ggml_row_size(tk, 256) / 256.0f;
    }
    kv_zone_offsets(out, F, R);
    return true;
}

// the bytes per cell a graph's lend grows by while the growing records hold quantized K/V: the attention converts one layer's K and V to f16 inside the lend
// (fattn-common.cuh, f16_extra; an MLA cache has no V row, its V is K's copy). Prompt graphs always convert (pc); a decode graph (one query token) reads q8_0 / q4_0
// directly when the head size has a vector kernel (64 .. 256 in steps of 64, not 192, equal K and V heads: fattn.cu), else it converts too (pc_dec).
static void kv_lend_pc_of(const kvt_runs & kvt, size_t & pc_out, size_t & pc_dec_out) {
    size_t pc = 0;
    bool   vec = true;
    for (auto & kr : kvt) {
        if (kr.fixed || !kr.layout || (!ggml_is_quantized(kr.tk) && !ggml_is_quantized(kr.tv))) { continue; }
        std::vector<llama_kv_tail_fork::seg> s;
        kr.layout(kr.tk, kr.tv, nullptr, &s);   // every layer: the attention copies a host layer's K/V into the lend too
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i].v) { continue; }
            const bool has_v = i + 1 < s.size() && s[i + 1].v;
            pc = std::max(pc, (size_t) (s[i].ne0 + (has_v ? s[i + 1].ne0 : 0)) * 2);
            const int64_t hk = s[i].head, hv = has_v ? s[i + 1].head : -1;
            if (hk != hv || hk > 256 || hk % 64 != 0 || hk == 192) { vec = false; }
        }
    }
    pc_out = pc; pc_dec_out = vec ? 0 : pc;
}

// bind the KV records of the cards' caches that no pool has taken yet (g_kvt, attached == false) to this card's pool: one zone of combined rows behind the compute lend, in front
// of the experts (fork/docs/kv-grow.md). At pool creation, and again when a cache is rebuilt with other types (qkv-ram-fallback: llama_kv_refit). Backs no cells yet: the first graph
// lays it out (llama_moe_cache_kv_tail_prepare_fork), the first graph that needs more lend moves it behind that.
static void kv_tail_bind(moe_cache::alias_pool & ap) {
    std::lock_guard<std::mutex> klk(g_kvt_mtx);
    ap.kv_row = 0; ap.kv_fixed = 0;
    for (auto & r : g_kvt) {
        if (r.dev != ap.dev || r.attached) { continue; }
        if (ap.span != ap.tail_end && ap.kv_start == 0 && ap.kv_h == 0 && ap.kvt.empty()) {
            LLAMA_LOG_ERROR("%s: kv-tail needs --moe alias-all (the zone is laid out across every size class)\n", __func__);
            GGML_ABORT("kv-tail without alias-all");
        }
        moe_cache::alias_pool::kvt_run kr;
        kr.owner = r.owner; kr.bytes = r.bytes; kr.row = r.row; kr.pad = std::max<size_t>(1, r.pad); kr.rows = r.rows; kr.attach = r.attach; kr.bpv = r.bpv;
        kr.layout = r.layout; kr.retype = r.retype; kr.tk = r.type_k; kr.tv = r.type_v; kr.convertible = r.convertible && r.layout && r.retype;
        kr.unit = r.layout ? kv_rec_unit(r.type_k, r.type_v) : 256;
        kr.cells_max = r.row ? r.bytes / r.row : 0;
        kr.ratio = llama_kv_cache_tail_ratio_fork(r.owner);
        if (r.step > 0) { ap.kv_step = r.step; }
        ap.kvt.push_back(kr);
        r.attached = true;
    }
    if (ap.kvt.empty()) { return; }
    // records covering the most tokens (the model's context) grow together in one combined row; a sliding-window cache (its cells are a ring of the window plus
    // a batch) is fixed: laid out at full size in front of the growing part. A compressed cache covers the whole context with fewer rows: kv_rec_fixed
    size_t cmax = 0;   // tokens
    for (auto & kr : ap.kvt) { cmax = std::max(cmax, kr.cells_max * kr.ratio); }
    for (auto & kr : ap.kvt) { kr.fixed = kv_rec_fixed(kr.cells_max, kr.ratio, cmax, ap.kv_step); kr.tk0 = kr.tk; kr.tv0 = kr.tv; }
    kv_zone_offsets(ap.kvt, ap.kv_fixed, ap.kv_row);
    kv_lend_pc_of(ap.kvt, ap.kv_lend_pc, ap.kv_lend_pc_dec);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    for (auto & kr : ap.kvt) {   // row 0 at the start of the buffer until the first layout
        if (kr.attach) { kr.attach(ap.buf, base + (kr.fixed ? kr.prefix : ap.kv_fixed + kr.prefix), kr.fixed ? kr.row : ap.kv_row); }
    }
    ap.kv_start = 0; ap.kv_h = 0;
    size_t bmax = 0; for (auto & sg2 : ap.segs) { bmax = std::max(bmax, sg2.B); }
    const size_t cap = ap.kv_row && ap.tail_end > 4 * bmax + ap.kv_fixed ? (ap.tail_end - 4 * bmax - ap.kv_fixed) / ap.kv_row : 0;
    LLAMA_LOG_WARN("%s: kv-tail: %zu record(s) on %s: fixed part %.1f MiB, growing part %zu B per cell in steps of %zu cells: room for %zu tokens in the pool of %.1f MiB (model cap %zu)\n",
            __func__, ap.kvt.size(), ggml_backend_dev_name(ap.dev), ap.kv_fixed / 1048576.0, ap.kv_row, ap.kv_step, cap, ap.tail_end / 1048576.0, cmax);
}

bool llama_moe_cache_kv_tail_prepare_fork(ggml_backend_sched_t sched, int32_t n_tokens) {
    moe_cache * mc = g_cache;
    if (!mc) { return false; }
    static bool tiers_logged = false;
    if (!tiers_logged) {
        const llama_kv_fit_caps cp = llama_moe_cache_kv_fit_caps_fork();
        if (cp.valid) {
            tiers_logged = true;
            LLAMA_LOG_WARN("kv-fit: KV tiers with the expert cache at its floor: f16 up to %zu tokens, q8_0/q8_0 up to %zu, q4_0/q4_0 up to %zu (%.1f KB of f16 KV per token over all cards); beyond that the KV would have to live in RAM\n",
                    cp.f16, cp.k8v8, cp.k4v4, cp.row_f16 / 1024.0);
        }
    }
    const bool reset = n_tokens < 64;   // a decode graph ends the plan of the last prompt
    if (!kv_tail_plan(mc, 0, reset, false, n_tokens, true)) {
        if (reset) { kv_tail_plan(mc, 0, true, false, n_tokens, false); }   // only the plan resets (no change of the zones)
        return false;
    }
    if (sched) { ggml_backend_sched_synchronize(sched); }   // the previous graph may still read the rows and the blocks that change hands
    kv_tail_plan(mc, 0, reset, false, n_tokens, false);
    flush_tables();
    return g_kv_epoch.load() != g_kv_epoch_built;
}

// ---- KV arena for the cards without an expert pool (dense models): one buffer per card, the records interleaved in one combined row per cell ----
struct kv_arena_run { void * owner; std::function<size_t()> rows; size_t cells_max; bool fixed; };
struct kv_arena_t { ggml_backend_dev_t dev = nullptr; ggml_backend_buffer_t buf = nullptr; size_t size = 0, row = 0, fixed = 0, pad = 1; std::vector<kv_arena_run> runs; };
// the cells an arena can hold: whole pads (the attention reads the cells rounded up to the pad, those rows must lie inside the buffer: same rule as the pool zone)
static size_t kv_arena_cells(const kv_arena_t & ar) { return ar.size / std::max<size_t>(1, ar.row) / ar.pad * ar.pad; }
static std::vector<kv_arena_t> g_kv_arena;
static std::string g_kv_arena_err;   // why a card got no arena (the records stay unattached: a graph built over them aborts in the allocator); asked for by decode()

// pool_records: whether the records of the context that expects the expert pool may go to an arena (see g_kvt_pool_bound). That context passes true once its
// pool started (the records the pool did not take); every other context (a draft) false, so it takes only its own records.
static void kv_arena_start(bool pool_records) {
    std::lock_guard<std::mutex> lk(g_kvt_mtx);
    auto take = [&](const llama_kv_tail_fork & r) { return !r.attached && (pool_records || !kvt_pool_bound(r.owner)); };
    std::vector<ggml_backend_dev_t> devs;
    for (auto & r : g_kvt) { if (take(r) && r.dev && std::find(devs.begin(), devs.end(), r.dev) == devs.end()) { devs.push_back(r.dev); } }
    for (ggml_backend_dev_t dev : devs) {
        // growing records in one combined row, sliding-window rings fixed at full size in front (as in the pool, kv_rec_fixed); tmax: the most tokens a record
        // covers, cmax: the most rows of a growing record (the combined rows the arena needs)
        size_t row = 0, fixed = 0, cmax = 0, tmax = 0;
        auto fixed_of = [&](const llama_kv_tail_fork & r) { return kv_rec_fixed(r.bytes / r.row, llama_kv_cache_tail_ratio_fork(r.owner), tmax, r.step > 0 ? r.step : 10240); };
        for (auto & r : g_kvt) { if (take(r) && r.dev == dev && r.row) { tmax = std::max(tmax, r.bytes / r.row * llama_kv_cache_tail_ratio_fork(r.owner)); } }
        for (auto & r : g_kvt) {
            if (!take(r) || r.dev != dev || !r.row) { continue; }
            if (fixed_of(r)) { fixed += (r.bytes + 255) / 256 * 256; } else { row += r.row; cmax = std::max(cmax, r.bytes / r.row); }
        }
        if (row == 0) { continue; }
        size_t free_b = 0, total = 0;
        ggml_backend_dev_memory(dev, &free_b, &total);
        const size_t margin = (size_t) (knobs().margin_mb > 0 ? knobs().margin_mb : 384) * 1024 * 1024;
        size_t pad = 256;   // the attention reads the cells rounded up to the pad: the buffer holds at least one whole pad of rows behind the fixed part
        for (auto & r : g_kvt) { if (take(r) && r.dev == dev && r.row) { pad = std::max<size_t>(pad, r.pad); } }
        const size_t least = fixed + pad * row;
        if (free_b <= margin + (64u << 20) || free_b - margin < least) {
            LLAMA_LOG_ERROR("kv-arena: %s has %.1f MiB free: no room for the KV\n", ggml_backend_dev_name(dev), free_b / 1048576.0);
            g_kv_arena_err = tr_fmt("kv-arena: %s has %.1f MiB of VRAM free, the KV cache needs the %.0f MiB safety margin plus %.1f MiB (the fixed part and %zu cells): leave more VRAM free, or use a smaller model or context",
                    ggml_backend_dev_name(dev), free_b / 1048576.0, margin / 1048576.0, least / 1048576.0, pad);
            continue;
        }
        // never more than the model's context needs (taking all free VRAM left nothing for the compute buffers: Qwen3.6 on a 8 GB card, 4124 MiB arena, the graph reserve failed);
        // when the allocation fails (the free figure over-reports, e.g. under WDDM) retry with less, down to the least
        const size_t need = fixed + cmax * row;
        size_t size = std::min(free_b - margin, need);
        const int64_t t_alloc0 = ggml_time_us();
        ggml_backend_buffer_t buf = nullptr;
        for (;;) {
            buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), size);   // the one allocation of the KV, at start
            if (buf || size <= least) { break; }
            LLAMA_LOG_WARN("kv-arena: could not allocate %.1f MiB on %s, trying less\n", size / 1048576.0, ggml_backend_dev_name(dev));
            size = std::max(least, size - size / 4);
        }
        if (!buf) {
            LLAMA_LOG_ERROR("kv-arena: could not allocate %.1f MiB on %s\n", size / 1048576.0, ggml_backend_dev_name(dev));
            g_kv_arena_err = tr_fmt("kv-arena: could not allocate %.1f MiB of KV cache on %s: leave more VRAM free, or use a smaller context", size / 1048576.0, ggml_backend_dev_name(dev));
            continue;
        }
        ggml_backend_buffer_clear(buf, 0);   // the attention reads the cells rounded up to the pad: those rows must be finite
        if (fixed + row > size) {
            LLAMA_LOG_ERROR("kv-arena: %s: the fixed records (%.1f MiB) do not fit in %.1f MiB\n", ggml_backend_dev_name(dev), fixed / 1048576.0, size / 1048576.0);
            g_kv_arena_err = tr_fmt("kv-arena: %s: the fixed part of the KV cache (%.1f MiB) does not fit in the %.1f MiB of VRAM left: leave more VRAM free", ggml_backend_dev_name(dev), fixed / 1048576.0, size / 1048576.0);
            ggml_backend_buffer_free(buf);
            continue;
        }
        g_kv_arena.push_back(kv_arena_t{ dev, buf, size - fixed, row, fixed, pad, {} });
        kv_arena_t & ar = g_kv_arena.back();
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        size_t prefix = 0, fpos = 0;
        for (auto & r : g_kvt) {
            if (!take(r) || r.dev != dev || !r.row) { continue; }
            const bool fx = fixed_of(r);
            if (r.attach) { r.attach(buf, fx ? base + fpos : base + fixed + prefix, fx ? r.row : row); }
            r.attached = true;
            ar.runs.push_back(kv_arena_run{ r.owner, r.rows, r.bytes / r.row, fx });
            if (fx) { fpos += (r.bytes + 255) / 256 * 256; } else { prefix += r.row; }
        }
        LLAMA_LOG_WARN("kv-arena: %s: allocation + clear of %.1f MiB took %.1f ms\n", ggml_backend_dev_name(dev), size / 1048576.0, (ggml_time_us() - t_alloc0) / 1000.0);
        LLAMA_LOG_WARN("kv-arena: %s: %.1f MiB, %zu records (fixed part %.1f MiB, growing part %zu B per cell): room for %zu tokens\n", ggml_backend_dev_name(dev),
                (size) / 1048576.0, ar.runs.size(), fixed / 1048576.0, row, kv_arena_cells(ar));
    }
}
void llama_moe_cache_kv_arena_start_fork() { kv_arena_start(true); }
void llama_moe_cache_kv_arena_start_own_fork(bool pool_records) { kv_arena_start(pool_records); }   // declared where it is called (llama-context.cpp), no header edit

// the reason, when a card that keeps its KV outside an expert pool got no arena (a record still unattached after the start): decode() then fails with this text instead of building a
// graph over records without memory (the allocator aborted: 'cache_kv_record already has a buffer', OLMoE with 0.47 GiB VRAM left, 2026-10-18)
const char * llama_moe_cache_kv_arena_unplaced_fork() {
    if (g_kv_arena_err.empty()) { return nullptr; }
    std::lock_guard<std::mutex> lk(g_kvt_mtx);
    for (auto & r : g_kvt) { if (!r.attached && r.row) { return g_kv_arena_err.c_str(); } }
    return nullptr;
}

// What stays in front of a growing KV zone when a PROMPT runs: the lend of its smallest pass (64 tokens; from the two largest graph sizes measured, at the current
// cells, as pp_cap_to_pool fits them), else what a decode graph borrows, plus three blocks (two stay the experts', one for the rounding). With quantized K/V the lend
// grows with the cells (kv_lend_pc, the attention's f16 copies): this is the lend extrapolated to no cells, the growth is charged per cell (kv_prompt_cells). The KV
// limit for prompts (find_slot), the tier capacities (kv-fit) and the "fits" line of a request all use it, so they cannot disagree.
static size_t kv_prompt_lend0(const moe_cache::alias_pool & ap) {
    const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
    size_t cells = 0;
    for (auto & q : ap.kvt) { if (!q.fixed) { cells = std::max(cells, q.rows ? q.rows() : 0); } }
    std::vector<std::pair<int32_t, size_t>> pts;
    for (const auto & kv : ap.need_tok) { if (kv.first >= 64) { pts.push_back({ kv.first, need_at_cells(ap, kv.first, kv.second, cells) }); } }
    const size_t dpc = ap.kv_lend_pc_dec * ap.dec_cells;
    size_t lend = ap.dec_need > dpc ? ap.dec_need - dpc : 0;   // no prompt graph seen yet: a decode graph's (measured)
    if (!pts.empty()) {
        double a = 0.0, b = 0.0;
        const auto & p1 = pts.back();
        if (pts.size() >= 2 && pts[pts.size() - 2].first != p1.first && p1.second > pts[pts.size() - 2].second) {
            const auto & p0 = pts[pts.size() - 2];
            b = (double) (p1.second - p0.second) / (double) (p1.first - p0.first);
            a = std::max(0.0, (double) p1.second - b * (double) p1.first);
        } else {
            b = (double) p1.second / (double) p1.first;
        }
        const size_t l64 = (size_t) (a + 64.0 * b), pcc = ap.kv_lend_pc * cells;
        lend = std::max(lend, l64 > pcc ? l64 - pcc : 0);
    }
    (void) bmax;
    return kv_lend_round(ap, lend);   // on the block edge the zone then starts on
}
// the cells a prompt's KV can grow to on this card with a zone of fixed part F and combined row R while the lend grows by pc per cell (not capped by the model's
// context: the log shows what really fits)
static size_t kv_prompt_cells(const moe_cache::alias_pool & ap, size_t F, size_t R, size_t pc) {
    return kvx_cells(ap.tail_end, kv_prompt_lend0(ap), F, R, pc);
}
static bool kv_any_out(const moe_cache::alias_pool & ap);
static size_t kv_mem_available();
static size_t kv_prompt_room(const moe_cache::alias_pool & ap) {
    const size_t c = kv_prompt_cells(ap, ap.kv_fixed, ap.kv_row, ap.kv_lend_pc);
    // the zone lies in ONE gap between the class zero blocks, in whole pads (bug-log #9, #10): the pool's bytes over the row size over-reports. The gap behind the lend
    // of this floor (the floor less the fixed part and the three blocks is the lend itself)
    const size_t fl = kv_prompt_lend0(ap) + ap.kv_fixed;   // (the old kv_prompt_floor: the lend of the least pass, the fixed records, three blocks)
    const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
    size_t pad = 1;
    for (const auto & q : ap.kvt) { if (!q.fixed) { pad = std::max<size_t>(pad, q.pad); } }
    const size_t lend = fl - ap.kv_fixed;   // the lend of the least pass (kv_prompt_lend0)
    const size_t cg = ap.kv_row ? std::min(c, kv_gap_fit(ap, bmax, lend, ap.kv_row)) / pad * pad : c;
    return kv_any_out(ap) ? std::min(cg, ap.host_cells) : cg;   // layers in host memory: their arrays hold host_cells
}

// the tier capacities from the exact layouts each tier would give the records (whole-block rows, the fixed ring converted too, the attention's f16 copies in the lend)
llama_kv_fit_caps llama_moe_cache_kv_fit_caps_fork() {
    llama_kv_fit_caps out;
    moe_cache * mc = g_cache;
    if (!mc) { return out; }
    size_t cap[3] = { (size_t) -1, (size_t) -1, (size_t) -1 }, row0 = 0;
    bool any = false;
    for (auto & ap : mc->alias) {
        if (ap.kvt.empty() || ap.kv_row == 0) { continue; }
        any = true;
        for (int t = 0; t < 3; ++t) {
            kvt_runs v; size_t F = 0, R = 0, pc = 0, pcd = 0;
            if (!kv_records_as(ap.kvt, t, v, F, R)) { cap[t] = 0; continue; }
            kv_lend_pc_of(v, pc, pcd);
            cap[t] = std::min(cap[t], kv_prompt_cells(ap, F, R, pc));
            if (t == 0) { row0 += R; }
        }
    }
    if (!any) { return out; }
    out.valid = true; out.f16 = cap[0]; out.k8v8 = cap[1]; out.k4v4 = cap[2]; out.row_f16 = row0;
    {   // RAM part: every growing layer's KV in q4_0 in host memory (the free RAM less 4 GiB, or the host buffers already made), the pool keeps the fixed part
        size_t row_all = 0, have = 0; bool made = false;
        for (auto & ap : mc->alias) {
            if (ap.kvt.empty() || ap.kv_row == 0) { continue; }
            kvt_runs v; size_t F = 0, R = 0;
            if (!kv_records_as(ap.kvt, 2, v, F, R)) { continue; }
            for (auto & r : v) { if (!r.fixed && r.convertible) { row_all += r.layout(r.tk, r.tv, nullptr, nullptr); } }
            if (ap.hbuf) { made = true; have = have ? std::min(have, ap.host_cells) : ap.host_cells; }
        }
        out.ram = made ? have : (row_all ? kv_mem_available() / row_all : 0);
    }
    return out;
}

// ---- qkv-ram-fallback in place: the KV changes its types inside the pool (fork/docs/qkv-ram-fallback.md) ----
// User design 2026-10-17/18: no new cache object, no graph reallocation, nothing allocated or freed. When the KV of a tier fills the pool, the rows in use are
// converted on the card to the next tier's types (f16 -> q8_0 -> q4_0: 0.53, then 0.28 of the f16 bytes) and the zone shrinks to the new rows: the blocks it leaves
// go back to the experts (refilled with the hottest missing ones). A later request that fits a lower tier converts the rows it keeps back up.
// Conversion: in chunks of rows, each converted into a staging range (old type -> f32 -> new type: CUDA copies have no direct f16 -> q8_0) borrowed from the compute
// lend in front of the zone (idle between graphs), then copied to its place. Shrinking runs in ascending order (a new row never lies above its old one, so a chunk's
// copy never lands on rows not read yet), growing in descending order. The graphs run on a backend of their own per card (the conversion graphs differ every chunk
// and would disturb the scheduler backend's CUDA-graph state). The log line gives bytes and time of each switch.
// (Alternative weighed 2026-10-18: a quantized shadow written alongside in the zone's free end, moved to the start at the switch. It holds 1.53x the KV bytes from
// the first shadowed row on, taken from the experts, and ends the f16 tier at ~65% of its capacity; the one-shot switch costs ~0.1 s once.)
static ggml_backend_t kv_conv_backend(ggml_backend_dev_t dev) {
    static std::mutex mtx;
    static std::map<ggml_backend_dev_t, ggml_backend_t> m;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = m.find(dev);
    if (it != m.end()) { return it->second; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    m[dev] = be;
    return be;
}

// a staging range: the longest run of whole blocks of one size class between the dense split's copies and the KV zone (the compute lend), at most 256 MiB; never a
// class's zero block (it sits after the class's last block). Its experts are evicted for the conversion, the zone's final layout gives the blocks back free.
static bool kv_stage_range(const moe_cache::alias_pool & ap, size_t & lo, size_t & hi) {
    const size_t from = kv_lend_round(ap, ap.mirror_b);
    lo = hi = 0;
    for (const auto & sg : ap.segs) {
        const size_t c0 = std::max(sg.base, from), c1 = std::min(sg.base + (size_t) sg.n * sg.B, ap.kv_start);
        if (c1 <= c0 || sg.B == 0) { continue; }
        const size_t b0 = sg.base + (c0 - sg.base + sg.B - 1) / sg.B * sg.B;
        if (b0 >= c1) { continue; }
        const size_t len = std::min<size_t>((c1 - b0) / sg.B * sg.B, (size_t) 256 << 20);
        if (len > hi - lo) { lo = b0; hi = b0 + len; }
    }
    return hi > lo;
}

struct kv_cv_part { const std::vector<llama_kv_tail_fork::seg> * so; const std::vector<llama_kv_tail_fork::seg> * sn; size_t so_add, sn_add; };

// rows [0, n_rows) of a run: old row i at byte src + i*ss of the buffer, new row i at dst + i*ds, through the staging range [stg, stg + stg_b). up: descending chunks.
static bool kv_convert_run(ggml_backend_t be, moe_cache::alias_pool & ap, size_t src, size_t ss, size_t dst, size_t ds, size_t n_rows,
                           const std::vector<kv_cv_part> & parts, size_t stg, size_t stg_b, bool up, size_t & moved) {
    if (n_rows == 0) { return true; }
    size_t f32_row = 0, n_seg = 0;
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.so->size(); ++i) {
            n_seg++;
            if ((*p.so)[i].type != (*p.sn)[i].type) { f32_row += (size_t) (*p.so)[i].ne0 * 4; }
        }
    }
    if (n_seg == 0) { return true; }
    const size_t slack = (n_seg + 2) * 256;
    if (stg_b <= slack) { return false; }
    const size_t chunk = std::min(n_rows, (stg_b - slack) / (ds + f32_row));
    if (chunk == 0) { return false; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    const size_t n_chunks = (n_rows + chunk - 1) / chunk, n_nodes = n_seg * 4 + 8;
    for (size_t k = 0; k < n_chunks; ++k) {
        const size_t ci = up ? n_chunks - 1 - k : k;
        const size_t a = ci * chunk, n = std::min(chunk, n_rows - a);
        ggml_init_params ip = { n_seg * 6 * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false) + 4096, nullptr, true };
        ggml_context * cx = ggml_init(ip);
        if (!cx) { return false; }
        ggml_cgraph * gf = ggml_new_graph_custom(cx, n_nodes, false);
        size_t tmp = GGML_PAD(stg + n * ds, 256);   // f32 temporaries behind the converted rows, one per segment (no memory shared inside a graph)
        auto mk = [&](ggml_type t, int64_t ne0, size_t at, size_t stride) {
            ggml_tensor * x = ggml_new_tensor_2d(cx, t, ne0, (int64_t) n);
            x->nb[1] = stride; x->nb[2] = x->nb[3] = stride * n;
            x->data = base + at; x->buffer = ap.buf;
            return x;
        };
        for (const auto & p : parts) {
            for (size_t i = 0; i < p.so->size(); ++i) {
                const auto & o = (*p.so)[i];
                const auto & w = (*p.sn)[i];
                const size_t s_at = src + a * ss + p.so_add + o.off;
                const size_t d_at = stg + p.sn_add + w.off;
                if (o.type == w.type) {   // the same type: its bytes, copied as halves (every row is a whole number of 2-byte units)
                    const int64_t nh = (int64_t) (ggml_row_size(o.type, o.ne0) / 2);
                    ggml_build_forward_expand(gf, ggml_cpy(cx, mk(GGML_TYPE_F16, nh, s_at, ss), mk(GGML_TYPE_F16, nh, d_at, ds)));
                } else {
                    ggml_tensor * t32 = mk(GGML_TYPE_F32, o.ne0, tmp, (size_t) o.ne0 * 4);
                    tmp = GGML_PAD(tmp + (size_t) o.ne0 * 4 * n, 256);
                    ggml_build_forward_expand(gf, ggml_cpy(cx, ggml_cpy(cx, mk(o.type, o.ne0, s_at, ss), t32), mk(w.type, w.ne0, d_at, ds)));
                }
            }
        }
        GGML_ASSERT(tmp <= stg + stg_b && "kv-retype: staging overrun");
        for (ggml_tensor * t = ggml_get_first_tensor(cx); t != nullptr; t = ggml_get_next_tensor(cx, t)) { if (!t->buffer) { t->buffer = ap.buf; } }
        const ggml_status st = ggml_backend_graph_compute(be, gf);   // synchronous
        ggml_free(cx);
        if (st != GGML_STATUS_SUCCESS) { return false; }
        kv_move_bytes(ap, stg, dst + a * ds, n * ds);   // the converted rows to their place (synchronous device copy)
        moved += n * ds;
    }
    return true;
}

bool llama_moe_cache_kv_retype_fork(ggml_backend_sched_t sched, int32_t tier, char * note, size_t note_size) {
    auto say = [&](const char * fmt, auto... args) { if (note && note_size) { snprintf(note, note_size, fmt, args...); } };
    moe_cache * mc = g_cache;
    if (!mc) { say("kv-fit: no expert pool"); return false; }
    const int64_t t0 = ggml_time_us();
    struct card_plan { moe_cache::alias_pool * ap; kvt_runs nk; size_t F = 0, R = 0, pc = 0, pcd = 0, cells = 0, h = 0, s_lo = 0, s_hi = 0; bool up = false; ggml_backend_t be = nullptr; };
    std::vector<card_plan> plans;
    for (auto & ap : mc->alias) {   // every card is planned and checked first: nothing changes unless all of them can take the new layout
        if (ap.kvt.empty() || ap.kv_row == 0) { continue; }
        card_plan cp; cp.ap = &ap;
        const char * dn = ggml_backend_dev_name(ap.dev);
        if (!kv_records_as(ap.kvt, tier, cp.nk, cp.F, cp.R)) { say("kv-fit: %s: a record's rows are not whole blocks of tier %d's types", dn, tier); return false; }
        kv_lend_pc_of(cp.nk, cp.pc, cp.pcd);
        bool le = cp.F <= ap.kv_fixed && cp.R <= ap.kv_row, ge = cp.F >= ap.kv_fixed && cp.R >= ap.kv_row;
        for (size_t i = 0; i < ap.kvt.size(); ++i) {
            le = le && cp.nk[i].prefix <= ap.kvt[i].prefix && cp.nk[i].row <= ap.kvt[i].row;
            ge = ge && cp.nk[i].prefix >= ap.kvt[i].prefix && cp.nk[i].row >= ap.kvt[i].row;
        }
        bool same = le && ge;
        for (size_t i = 0; i < ap.kvt.size() && same; ++i) { same = cp.nk[i].tk == ap.kvt[i].tk && cp.nk[i].tv == ap.kvt[i].tv && cp.nk[i].out_next == ap.kvt[i].out; }
        if (same) { continue; }
        for (size_t i = 0; i < ap.kvt.size(); ++i) {   // a record that cannot be retyped keeps its views: the metadata must not say otherwise
            const auto & r = ap.kvt[i];
            if (!r.retype && (cp.nk[i].tk != r.tk || cp.nk[i].tv != r.tv || cp.nk[i].out_next != r.out)) { say("kv-fit: %s: a record cannot change its types to tier %d's", dn, tier); return false; }
        }
        if (!le && !ge) { say("kv-fit: %s: tier %d's layout is neither smaller nor larger everywhere (records keep their order)", dn, tier); return false; }
        cp.up = !le;
        for (auto & r : ap.kvt) { if (!r.fixed) { cp.cells = std::max(cp.cells, r.rows ? r.rows() : 0); } }
        const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
        const size_t room = ap.tail_end > ap.kv_start + cp.F ? (ap.tail_end - ap.kv_start - cp.F) / std::max<size_t>(1, cp.R) : 0;   // every block behind the zone's start
        if (cp.cells > room) { say("kv-fit: %s: %zu cells do not fit the zone as tier %d (room for %zu)", dn, cp.cells, tier, room); return false; }
        cp.h = std::min(std::max(ap.kv_h, cp.cells), std::max(room, cp.cells));
        {   // the new extent must stay inside the one gap between class zero blocks that holds the zone (kvz_bad): bigger rows (converting up) reach past it, over a zero block, where
            // uncached experts read zero rows and a zero restore would wipe KV. Refused here, before anything changes.
            size_t gap_end = 0;
            for (const auto & g : kvz_gaps(ap, 0, 1)) { if (ap.kv_start >= g.first && ap.kv_start < g.second) { gap_end = g.second; } }
            const size_t gap_h = gap_end > ap.kv_start + cp.F && cp.R > 0 ? (gap_end - ap.kv_start - cp.F) / cp.R : 0;
            if (gap_end == 0 && !cp.up) { /* no gap found around the zone start: the check cannot say anything, and a smaller layout cannot overrun one */ }
            else if (cp.cells > gap_h) { say("kv-fit: %s: %zu cells do not fit the gap of the zone as tier %d (room for %zu)", dn, cp.cells, tier, gap_h); return false; }
            if (gap_end != 0) { cp.h = std::min(cp.h, std::max(gap_h, cp.cells)); }
        }
        if (!kv_stage_range(ap, cp.s_lo, cp.s_hi)) { say("kv-fit: %s: no lend in front of the KV to stage the conversion", dn); return false; }
        cp.be = kv_conv_backend(ap.dev);
        if (!cp.be) { say("kv-fit: %s: no backend for the conversion", dn); return false; }
        plans.push_back(std::move(cp));
    }
    if (plans.empty()) { return true; }
    if (sched) { ggml_backend_sched_synchronize(sched); }   // nothing may read the rows or the blocks that change hands
    size_t moved = 0, b_old = 0, b_new = 0, n_give_all = 0, n_take_all = 0;
    for (auto & cp : plans) {
        auto & ap = *cp.ap;
        size_t nt = 0, ng = 0, ne = 0;
        const size_t Z = ap.kv_start;
        b_old += ap.kv_fixed + cp.cells * ap.kv_row;
        kv_blocks_apply(mc, ap, cp.s_lo, cp.s_hi, true, false, nt, ng, ne);   // the staging: its experts out (uploads into it dropped or awaited)
        if (cp.up) { kv_blocks_apply(mc, ap, Z, Z + cp.F + cp.h * cp.R, true, false, nt, ng, ne); }   // converting up: the new extent's experts out before rows land there
        const size_t nr = ap.kvt.size();
        std::vector<std::vector<llama_kv_tail_fork::seg>> so(nr), sn(nr);
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; auto & q = cp.nk[i];
            if (r.layout) { r.layout(r.tk, r.tv, &r.out, &so[i]); q.layout(q.tk, q.tv, &q.out_next, &sn[i]); }
            else          { so[i] = sn[i] = { { GGML_TYPE_F16, (int64_t) (r.row / 2), 0, false, 0, 0 } }; }   // no layout: the row moves as bytes
        }
        // the RAM part: layers leaving the row for host memory (q.out_next, places in q.host) are copied there first (the cells in use, by the card: its kernels
        // write host memory directly), then dropped from the old segments: the compaction below moves only the layers that stay
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; auto & q = cp.nk[i];
            if (r.fixed || q.out_next.empty()) { continue; }
            std::vector<llama_kv_tail_fork::seg> keep;
            const size_t n_seg = so[i].size();
            ggml_init_params ip = { (n_seg * 3 + 4) * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_seg * 2 + 4, false) + 4096, nullptr, true };
            ggml_context * hx = ggml_init(ip);
            ggml_cgraph * hg = ggml_new_graph_custom(hx, n_seg * 2 + 4, false);
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
            bool any = false;
            for (const auto & sg : so[i]) {
                const bool leaves = sg.lay < (int32_t) q.out_next.size() && q.out_next[sg.lay] && !(sg.lay < (int32_t) r.out.size() && r.out[sg.lay]);
                if (!leaves) { keep.push_back(sg); continue; }
                if (cp.cells == 0) { continue; }
                const auto & h = q.host[2 * sg.lay + (sg.v ? 1 : 0)];
                const int64_t nh = (int64_t) (ggml_row_size(sg.type, sg.ne0) / 2);
                ggml_tensor * a = ggml_new_tensor_2d(hx, GGML_TYPE_F16, nh, (int64_t) cp.cells);
                a->nb[1] = ap.kv_row; a->nb[2] = a->nb[3] = ap.kv_row * cp.cells; a->data = base + Z + ap.kv_fixed + r.prefix + sg.off; a->buffer = ap.buf;
                ggml_tensor * b = ggml_new_tensor_2d(hx, GGML_TYPE_F16, nh, (int64_t) cp.cells);
                b->nb[1] = h.second; b->nb[2] = b->nb[3] = h.second * cp.cells; b->data = h.first; b->buffer = ap.buf;
                ggml_build_forward_expand(hg, ggml_cpy(hx, a, b));
                any = true;
                moved += cp.cells * (size_t) nh * 2;
            }
            for (ggml_tensor * t = ggml_get_first_tensor(hx); t != nullptr; t = ggml_get_next_tensor(hx, t)) { if (!t->buffer) { t->buffer = ap.buf; } }
            if (any && ggml_backend_graph_compute(cp.be, hg) != GGML_STATUS_SUCCESS) { GGML_ABORT("kv-ram: copying layers to host memory failed (the KV is lost)"); }
            ggml_free(hx);
            so[i] = keep;
        }
        struct run { size_t src, ss, dst, ds, n; std::vector<kv_cv_part> parts; };
        std::vector<run> fixed_runs;
        run grow { Z + ap.kv_fixed, ap.kv_row, Z + cp.F, cp.R, cp.cells, {} };
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; auto & q = cp.nk[i];
            if (r.fixed) { fixed_runs.push_back({ Z + r.prefix, r.row, Z + q.prefix, q.row, r.cells_max, { { &so[i], &sn[i], 0, 0 } } }); }
            else         { grow.parts.push_back({ &so[i], &sn[i], r.prefix, q.prefix }); }
        }
        std::vector<run *> order;   // shrinking: lowest addresses first (the fixed part, then the growing rows); growing: the reverse
        for (auto & fr : fixed_runs) { order.push_back(&fr); }
        order.push_back(&grow);
        if (cp.up) { std::reverse(order.begin(), order.end()); }
        for (run * rr : order) {
            if (!kv_convert_run(cp.be, ap, rr->src, rr->ss, rr->dst, rr->ds, rr->n, rr->parts, cp.s_lo, cp.s_hi - cp.s_lo, cp.up, moved)) {
                GGML_ABORT("kv-retype: a conversion graph failed half-way (the KV is lost)");
            }
        }
        // the records take their new types, rows and places
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; const auto & q = cp.nk[i];
            if (r.retype && (r.tk != q.tk || r.tv != q.tv || r.out != q.out_next)) {
                static const bool cpu_attn = [] { const char * e = moe_opt("kv-ram-attn"); return e && strcmp(e, "cpu") == 0; }();
                // host layers: gpu (default) keeps the views in the pool's buffer, so their attention stays on the card and reads host memory over PCIe; cpu gives
                // them the host buffer, and the scheduler runs their attention on the CPU (RAM speed, the link stays free for expert uploads)
                r.retype(q.tk, q.tv, &q.out_next, &q.host, cpu_attn && ap.hbuf ? ap.hbuf : ap.buf);
            }
            r.out = q.out_next; r.host = q.host;
            r.row = q.row; r.bytes = q.bytes; r.prefix = q.prefix; r.tk = q.tk; r.tv = q.tv; r.unit = q.unit; r.bpv = q.bpv;
        }
        {
            std::lock_guard<std::mutex> klk(g_kvt_mtx);
            for (auto & g : g_kvt) {
                for (auto & r : ap.kvt) { if (g.owner == r.owner && g.dev == ap.dev) { g.row = r.row; g.bytes = r.bytes; g.type_k = r.tk; g.type_v = r.tv; g.bpv = r.bpv; } }
            }
        }
        // the lend measured so far, moved to the new per-cell growth (it was measured with or without the attention's f16 copies)
        for (auto & kv : ap.need_tok) {
            const auto it = ap.need_cells.find(kv.first);
            const int64_t c0 = it != ap.need_cells.end() ? (int64_t) it->second : 0;
            const int64_t d = kv.first < 64 ? (int64_t) cp.pcd - (int64_t) ap.kv_lend_pc_dec : (int64_t) cp.pc - (int64_t) ap.kv_lend_pc;
            kv.second = (size_t) std::max<int64_t>(0, (int64_t) kv.second + d * c0);
        }
        ap.dec_need = (size_t) std::max<int64_t>(0, (int64_t) ap.dec_need + ((int64_t) cp.pcd - (int64_t) ap.kv_lend_pc_dec) * (int64_t) ap.dec_cells);
        ap.kv_fixed = cp.F; ap.kv_row = cp.R; ap.kv_lend_pc = cp.pc; ap.kv_lend_pc_dec = cp.pcd;
        // the zone's new extent: blocks it left (and the staging) back to the experts
        kv_blocks_apply(mc, ap, Z, Z + cp.F + cp.h * cp.R, true, true, nt, ng, ne);
        n_give_all += ng; n_take_all += nt;
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
        for (auto & r : ap.kvt) { if (r.attach) { r.attach(ap.buf, base + Z + (r.fixed ? r.prefix : cp.F + r.prefix), r.fixed ? r.row : cp.R); } }
        for (const auto & r : ap.kvt) {   // a compressed record's pad rows lie below the others' cells in use: the rows past the converted ones must be finite (as in kv_layout)
            if (r.fixed || r.ratio <= 1) { continue; }
            const size_t z0 = Z + cp.F + cp.cells * cp.R, z1 = Z + cp.F + cp.h * cp.R;
            if (z1 > z0) {
                ggml_tensor z = {};
                z.type = GGML_TYPE_I8; z.buffer = ap.buf; z.data = base + z0;
                z.ne[0] = (int64_t) (z1 - z0); z.ne[1] = z.ne[2] = z.ne[3] = 1;
                z.nb[0] = 1; z.nb[1] = z.nb[2] = z.nb[3] = z1 - z0;
                ggml_backend_tensor_memset(&z, 0, 0, z1 - z0);
            }
            break;
        }
        ap.kv_h = cp.h; ap.kv_zhi = 0; ap.kv_last_cells = 0;
        for (auto & sg : ap.segs) {   // a lent block past the lend (the staging took lent ones: kv_blocks_apply lowered `blocked`) is nobody's: free
            auto & ps = mc->pool[sg.pcls];
            std::lock_guard<std::mutex> lk(mc->mtx);
            for (int32_t b = sg.blocked; b < sg.n; ++b) { if (ps.owner[b] == -2) { ps.owner[b] = -1; ps.free.push_back(b); } }
            ps.active_end = -1;
        }
        size_t held = 0;
        for (auto & sg : ap.segs) { for (int32_t b = 0; b < sg.n; ++b) { if (mc->pool[sg.pcls].owner[b] == -3) { held++; } } }
        ap.kv_blocks = held;
        b_new += cp.F + cp.cells * cp.R;
    }
    g_kv_epoch++;   // graphs built before point at the old types and places
    flush_tables();
    if (n_give_all > 0) { mc->kv_gave_back = true; }
    const char * tn = tier == 1 ? "q8_0/q8_0" : tier == 2 ? "q4_0/q4_0" : "the cache's own types";
    LLAMA_LOG_WARN("kv-fit: KV converted in place to %s on %zu card(s): %.1f -> %.1f MiB of rows (%.1f MiB staged) in %.1f ms; %zu blocks back to the experts, %zu taken\n",
            tn, plans.size(), b_old / 1048576.0, b_new / 1048576.0, moved / 1048576.0, (ggml_time_us() - t0) / 1000.0, n_give_all, n_take_all);
    say("kv-fit: KV converted in place to %s (%.1f -> %.1f MiB)", tn, b_old / 1048576.0, b_new / 1048576.0);
    return true;
}

// ---- qkv-ram-fallback, RAM part: whole layers' KV to host memory (user design 2026-10-18) ----
// When q4_0 is full too, the KV of whole layers leaves the pool for one host buffer per card (allocated once, at the first move, from free RAM less 4 GiB; never
// freed). A layer's K and V are separate views, so each can live elsewhere with its own stride: cells cannot be split between VRAM and RAM (one attention view
// needs one address range), layers can. The layers with the most bytes per cell go first, only as many as the request needs; the zone shrinks by their rows
// and the experts take the space. Attention of those layers reads host memory (--moe kv-ram-attn=gpu: by the card over PCIe, default; cpu: on the CPU).
static size_t kv_mem_available() {
    size_t b = 0;
#ifdef _WIN32
    MEMORYSTATUSEX ms = {}; ms.dwLength = sizeof(ms);   // (W10: the RAM tier held 0 tokens, there is no /proc/meminfo)
    // the commit limit counts too: W10 has no page file (32 GB virtual = 32 GB RAM) and the server's pinned weights are committed, so a free-physical reading of 20 GB left
    // 17 GB of commit and the 9-11 GiB host buffer of the RAM tier failed to allocate (2026-10-19)
    if (GlobalMemoryStatusEx(&ms)) { b = (size_t) std::min<unsigned long long>(ms.ullAvailPhys, ms.ullAvailPageFile); }
#else
    size_t kb = 0;
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) { if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) { break; } }
        fclose(f);
    }
    b = kb * 1024;
#endif
    static bool warned = false;
    if (b == 0 && !warned) { warned = true; LLAMA_LOG_WARN("moe-cache: kv-ram: the free RAM could not be read: the RAM tier of the KV holds nothing\n"); }
    const size_t margin = (size_t) 4 << 30;
    return b > margin ? b - margin : 0;
}
static size_t kv_row_all(const moe_cache::alias_pool & ap) {   // bytes per cell of every growing layer of this card in its current types (all of them in host memory)
    size_t row = 0;
    for (auto & r : ap.kvt) { if (!r.fixed && r.convertible) { row += r.layout(r.tk, r.tv, nullptr, nullptr); } }
    return row;
}
static bool kv_any_out(const moe_cache::alias_pool & ap) {
    for (auto & r : ap.kvt) { for (char c : r.out) { if (c) { return true; } } }
    return false;
}

bool llama_moe_cache_kv_ram_fork(ggml_backend_sched_t sched, size_t n_cells, char * note, size_t note_size) {
    auto say = [&](const char * fmt, auto... args) { if (note && note_size) { snprintf(note, note_size, fmt, args...); } };
    moe_cache * mc = g_cache;
    if (!mc) { say("kv-ram: no expert pool"); return false; }
    size_t n_cards = 0;
    for (auto & ap : mc->alias) { if (!ap.kvt.empty() && ap.kv_row) { n_cards++; } }
    if (n_cards == 0) { say("kv-ram: no KV in a pool"); return false; }
    size_t avail = kv_mem_available();
    std::vector<std::pair<moe_cache::alias_pool *, size_t>> undo;
    std::string what;
    // a failure on a later card leaves the earlier ones as they were: their host places are given back and their layers stay in the pool
    auto rollback = [&]() { for (auto & [ap, hu] : undo) { ap->hused = hu; for (auto & r : ap->kvt) { r.out_next = r.out; } } };
    for (auto & ap : mc->alias) {
        if (ap.kvt.empty() || ap.kv_row == 0 || kv_prompt_room(ap) >= n_cells) { continue; }
        const char * dn = ggml_backend_dev_name(ap.dev);
        const size_t r_all = kv_row_all(ap);
        if (r_all == 0) { say("kv-ram: %s: no layer can move", dn); rollback(); return false; }
        if (!ap.hbuf) {   // once: the host buffer, a share of the free RAM
            size_t cmax = 0;
            for (auto & r : ap.kvt) { if (!r.fixed) { cmax = std::max(cmax, r.cells_max); } }
            const size_t budget = avail / n_cards;
            size_t n_seg = 0;
            for (auto & r : ap.kvt) { if (!r.fixed && r.convertible) { std::vector<llama_kv_tail_fork::seg> sg; r.layout(r.tk, r.tv, nullptr, &sg); n_seg += sg.size(); } }
            const size_t hc = kvx_host_cells(budget, n_seg, r_all, cmax);
            if (hc < n_cells) { say("kv-ram: %s: free RAM holds %zu cells of all layers' KV, %zu needed", dn, hc, n_cells); rollback(); return false; }
            const size_t sz = kvx_host_size(hc, r_all, n_seg);
            const int64_t ta = ggml_time_us();
            ap.hbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(ap.dev), sz);
            if (!ap.hbuf) { say("kv-ram: %s: could not allocate %.1f MiB of host memory", dn, sz / 1048576.0); rollback(); return false; }
            ggml_backend_buffer_clear(ap.hbuf, 0);   // rows past the cells in use are read (masked) by the attention: finite
            ap.hsize = sz; ap.hused = 0; ap.host_cells = hc;
            avail = avail > sz ? avail - sz : 0;
            LLAMA_LOG_WARN("kv-ram: %s: host buffer %.1f MiB for %zu cells (allocated once, %.0f ms)\n", dn, sz / 1048576.0, hc, (ggml_time_us() - ta) / 1000.0);
        }
        if (n_cells > ap.host_cells) { say("kv-ram: %s: the host buffer holds %zu cells, %zu needed", dn, ap.host_cells, n_cells); rollback(); return false; }
        undo.push_back({ &ap, ap.hused });
        struct cand { size_t rec; int32_t lay; size_t bytes; };
        std::vector<cand> c;
        for (size_t i = 0; i < ap.kvt.size(); ++i) {
            auto & r = ap.kvt[i];
            if (r.fixed || !r.convertible) { continue; }
            std::vector<llama_kv_tail_fork::seg> sg;
            r.layout(r.tk, r.tv, &r.out, &sg);
            std::map<int32_t, size_t> by;
            for (auto & q : sg) { by[q.lay] += ggml_row_size(q.type, q.ne0); }
            for (auto & [l, b] : by) { c.push_back({ i, l, b }); }
        }
        std::sort(c.begin(), c.end(), [](const cand & a, const cand & b) { return a.bytes > b.bytes; });
        uint8_t * hb = (uint8_t *) ggml_backend_buffer_get_base(ap.hbuf);
        size_t n_mv = 0, b_mv = 0;
        bool met = false;   // the layers chosen hold the request: without this the loop could move every layer and still return success
        for (auto & k : c) {
            auto & r = ap.kvt[k.rec];
            std::vector<llama_kv_tail_fork::seg> all;
            r.layout(r.tk, r.tv, nullptr, &all);
            int32_t n_lay = 0;
            for (auto & q : all) { n_lay = std::max(n_lay, q.lay + 1); }
            if (r.out_next.size() < (size_t) n_lay) { r.out_next.resize(n_lay, 0); }
            if (r.host.size() < (size_t) (2 * n_lay)) { r.host.resize(2 * n_lay, { nullptr, 0 }); }
            r.out_next[k.lay] = 1;
            for (auto & q : all) {
                if (q.lay != k.lay) { continue; }
                const size_t st = ggml_row_size(q.type, q.ne0);
                r.host[2 * q.lay + (q.v ? 1 : 0)] = { hb + ap.hused, st };
                ap.hused += (ap.host_cells * st + 255) / 256 * 256;
            }
            GGML_ASSERT(ap.hused <= ap.hsize);
            n_mv++; b_mv += k.bytes;
            kvt_runs v; size_t F = 0, R = 0, pc = 0, pcd = 0;
            if (kv_records_as(ap.kvt, 3, v, F, R)) { kv_lend_pc_of(v, pc, pcd); if (kv_prompt_cells(ap, F, R, pc) >= n_cells) { met = true; break; } }
        }
        if (!met) { say("kv-ram: %s: with every layer's KV in host memory %zu cells still do not fit", dn, n_cells); rollback(); return false; }
        what += tr_fmt("%s%s: %zu layer(s), %zu B per cell", what.empty() ? "" : "; ", dn, n_mv, b_mv);
    }
    if (undo.empty()) { return true; }
    char why[384] = { 0 };
    if (!llama_moe_cache_kv_retype_fork(sched, 3, why, sizeof(why))) {
        rollback();
        say("kv-ram: %s", why);
        return false;
    }
    LLAMA_LOG_WARN("kv-ram: KV of %s moved to host memory (attention reads it %s)\n", what.c_str(),
            [] { const char * e = moe_opt("kv-ram-attn"); return e && strcmp(e, "cpu") == 0; }() ? "on the CPU" : "over PCIe on the card");
    say("kv-ram: layers to host memory: %s", what.c_str());
    return true;
}

// find_slot's limit: the tokens the zone may hold. The KV may take the whole expert cache except what a decode graph borrows and two blocks (starved experts are
// streamed experts, still faster than a starved KV): a context that would take more is "full" for the cache, the server stops the answer or refuses the prompt.
size_t llama_moe_cache_kv_tail_cap_fork(void * owner) {
    size_t res = (size_t) -1;   // the least over every card the cache has records on: arenas (dense cards) and pools alike, a cell is written on all of them
    for (auto & ar : g_kv_arena) {
        for (auto & q : ar.runs) { if (q.owner == owner) { res = std::min(res, q.fixed ? q.cells_max : std::min(kv_arena_cells(ar), q.cells_max)); break; } }
    }
    moe_cache * mc = g_cache;
    if (!mc) { return res; }
    // pool cards, same rule (2026-10-18 OLMoE: card 1 held 1752 cells, card 0 2079, the first card's limit let 1792 through: NaN)
    for (auto & ap : mc->alias) {
        for (auto & r : ap.kvt) {
            if (r.owner != owner) { continue; }
            if (r.fixed || ap.kv_row == 0) { res = std::min<size_t>(res, r.cells_max); break; }   // laid out at full size
            const size_t bmax = kv_bmax(ap);
            const size_t dpc = ap.kv_lend_pc_dec * ap.dec_cells;   // a decode graph's lend at no cells (quantized K/V: its f16 copies grow per cell)
            const size_t floor_b = kv_lend_round(ap, ap.dec_need > dpc ? ap.dec_need - dpc : 0) + ap.kv_fixed;   // the least lend (a decode graph's, measured) on its block edge, then the fixed part
            const size_t room = ap.tail_end > floor_b ? ap.tail_end - floor_b : 0;
            // the zone lies in ONE gap between the class zero blocks and the attention reads its cells rounded up to the pad: what the pool can really hold is the
            // largest gap behind the least lend, in whole pads. Not the pool's bytes over the row size (OLMoE 1.0 GiB free: "room for 6308 tokens" on a card whose
            // largest gap held 3300 rows, the 3440-token prompt was admitted and failed at cell 2949, 2026-10-18)
            size_t pad = 1;
            for (const auto & q : ap.kvt) { if (!q.fixed) { pad = std::max<size_t>(pad, q.pad); } }
            const size_t lend_floor = kv_lend_round(ap, ap.dec_need);   // a decode graph's lend (measured) on its block edge
            const size_t gap_cells  = kv_gap_fit(ap, bmax, lend_floor, ap.kv_row);
            size_t cap = std::min(std::min(room / (ap.kv_row + ap.kv_lend_pc_dec), gap_cells) / pad * pad, (size_t) r.cells_max);   // never more than the record's cells (the model's context)
            if (kv_any_out(ap)) { cap = std::min(cap, ap.host_cells); }   // layers in host memory: their arrays hold host_cells
            res = std::min(res, cap);
            break;
        }
    }
    return res;   // (-1 when the pool does not exist yet, before the first decode: no limit)
}

// The same limit for a prompt ubatch (find_slot, n_tokens >= 64): its pass needs a prompt graph's lend in front of the KV, not a decode graph's. Before
// 2026-10-17 a prompt could grow the KV to the decode limit, the pass planner then had no room for even a 64-token pass and the lend check aborted ("the compute
// graph and the KV do not fit in the pool") instead of the cache refusing the cells. The lend of the smallest pass (64 tokens) is the line: from the two largest
// sizes measured (need_tok, at the current cells), as pp_cap_to_pool fits them. A server that gets the refusal retries with smaller batches, which are decode-sized.
size_t llama_moe_cache_kv_tail_cap_n_fork(void * owner, uint32_t n_tokens) {
    const size_t cap = llama_moe_cache_kv_tail_cap_fork(owner);
    moe_cache * mc = g_cache;
    if (!mc || n_tokens < 64) { return cap; }
    size_t res = cap;   // the least over the cards, as in cap_fork
    for (auto & ap : mc->alias) {
        for (auto & r : ap.kvt) {
            if (r.owner != owner) { continue; }
            if (r.fixed || ap.kv_row == 0) { break; }
            if (ap.need_tok.empty()) { break; }
            bool any = false;
            for (const auto & kv : ap.need_tok) { if (kv.first >= 64) { any = true; } }
            if (!any) { break; }
            const size_t bmax = std::max<size_t>(1, kv_bmax(ap));
            size_t pad = 1;
            for (const auto & q : ap.kvt) { if (!q.fixed) { pad = std::max<size_t>(pad, q.pad); } }
            // the lend of a pass of n_tokens when the cache holds c cells (the graph's inputs grow with them), from the two largest sizes measured as pp_cap_to_pool fits them
            auto lend_at = [&](size_t c) -> size_t {
                // a size measured as this very pass size: the layout (kv_tail_plan) lends exactly that, so the fit must too (OLMoE 1900-token prompt retried with 512: the line
                // through the two largest sizes gave less than the 121.6 MiB the 512 graph borrows, the zone was cut to 1481 under a read of 1536 cells)
                const auto ex = ap.need_tok.find((int32_t) n_tokens);
                if (ex != ap.need_tok.end()) { return kv_lend_round(ap, need_at_cells(ap, ex->first, ex->second, c)); }
                std::vector<std::pair<int32_t, size_t>> pts;
                for (const auto & kv : ap.need_tok) { if (kv.first >= 64) { pts.push_back({ kv.first, need_at_cells(ap, kv.first, kv.second, c) }); } }
                double a = 0.0, b = 0.0;
                const auto & p1 = pts.back();
                if (pts.size() >= 2 && pts[pts.size() - 2].first != p1.first && p1.second > pts[pts.size() - 2].second) {
                    const auto & p0 = pts[pts.size() - 2];
                    b = (double) (p1.second - p0.second) / (double) (p1.first - p0.first);
                    a = std::max(0.0, (double) p1.second - b * (double) p1.first);
                } else {
                    b = (double) p1.second / (double) p1.first;
                }
                return kv_lend_round(ap, (size_t) (a + (double) n_tokens * b));   // on the block edge the zone starts on
            };
            // the most cells c (whole pads) for which a zone of c cells lies in ONE gap between the zero blocks behind THE LEND OF THIS PASS AT c CELLS, and the pool holds lend + zone
            // + the experts' two blocks and one for the rounding. Not at the decode lend or the 64-token lend: the pass the call really runs borrows more, and fewer gaps hold the zone
            // (OLMoE 1900-token prompt, 0.8 GiB free: advertised 2048 cells, the 64-token pass at 1088 cells borrowed 125 MiB, the gap behind it held 1481, cell 1343 was written to a
            // zone read to 1536, 2026-10-18). The lend only grows with c and the gap only shrinks with the lend: bisection.
            auto fits = [&](size_t c) -> bool {
                const size_t lend = lend_at(c);
                return lend + ap.kv_fixed + c * ap.kv_row <= ap.tail_end && kv_gap_fit(ap, bmax, lend, ap.kv_row) >= c;   // c is in whole pads: the attention reads exactly c rows   // one pad of margin: the lend is a fit through measured sizes and the real graph came 0.3% bigger (Qwen3.6 PC1, gap 13780 for 13824 cells)
            };
            size_t lo = 0, hi = cap / pad;   // in pads: lo fits (0 cells), hi is the advertised limit
            while (lo < hi) {
                const size_t mid = (lo + hi + 1) / 2;
                if (fits(mid * pad)) { lo = mid; } else { hi = mid - 1; }
            }
            res = std::min(res, lo * pad);
            break;
        }
    }
    return res;
}

// why kv_tail_covers_fork says no, for the message before its assert (declared locally in llama-kv-cache.cpp: no header change)
std::string llama_moe_cache_kv_tail_why_fork(void * owner, size_t max_cell) {
    moe_cache * mc = g_cache;
    char buf[400];
    if (!mc) { return "no cache"; }
    for (auto & ap : mc->alias) {
        for (auto & r : ap.kvt) {
            if (r.owner != owner) { continue; }
            snprintf(buf, sizeof buf, "%s: record %s, cell %zu: cells_max %zu, horizon kv_h %zu, kv_start %zu, kv_fixed %zu, kv_row %zu, row %zu, tail_end %zu",
                     ggml_backend_dev_name(ap.dev), r.fixed ? "fixed (sliding-window ring)" : "growing", max_cell, (size_t) r.cells_max, (size_t) ap.kv_h, (size_t) ap.kv_start, (size_t) ap.kv_fixed, (size_t) ap.kv_row, (size_t) r.row, (size_t) ap.tail_end);
            std::string more;
            {   // the first block under the range that is not the KV's
                size_t lo = r.fixed ? ap.kv_start + r.prefix : ap.kv_start + ap.kv_fixed;
                size_t hi = lo + (max_cell + 1) * (r.fixed ? r.row : ap.kv_row);
                for (auto & sg : ap.segs) {
                    const auto & ps = mc->pool[sg.pcls];
                    for (int32_t b = 0; b < sg.n; ++b) {
                        const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                        if (b1 > lo && b0 < hi && ps.owner[b] != -3) {
                            char t[300];
                            snprintf(t, sizeof t, "; first unclaimed block %d of class %d [%zu, %zu) owner %d, range [%zu, %zu), B %zu, span %zu, want_b %zu, base_off %zu, mirror_b %zu", b, sg.pcls, b0, b1, (int) ps.owner[b], lo, hi, (size_t) sg.B, (size_t) ap.span, (size_t) ap.want_b, (size_t) ap.base_off, (size_t) ap.mirror_b);
                            more = t; break;
                        }
                    }
                    if (!more.empty()) { break; }
                }
            }
            {   // the size classes of this card's pool
                std::string cl = "; classes:";
                for (auto & sg : ap.segs) { char t[120]; snprintf(t, sizeof t, " [pcls %d base %zu B %zu n %d blocked %d]", sg.pcls, (size_t) sg.base, (size_t) sg.B, (int) sg.n, (int) sg.blocked); cl += t; }
                char t2[80]; snprintf(t2, sizeof t2, " kv_bmax %zu", kv_bmax(ap)); more += cl + t2;
            }
            return std::string(buf) + more;
        }
    }
    return "owner not found";
}

bool llama_moe_cache_kv_tail_covers_fork(void * owner, size_t max_cell) {
    moe_cache * mc = g_cache;
    if (mc) {
        for (auto & ap : mc->alias) {
            for (auto & r : ap.kvt) {
                if (r.owner != owner) { continue; }
                size_t lo, hi;   // every block under [lo, hi) must be the KV's (owner -3)
                if (r.fixed) {
                    if (max_cell + 1 > r.cells_max) { return false; }
                    lo = ap.kv_start + r.prefix; hi = lo + (max_cell + 1) * r.row;
                } else {
                    // the attention reads up to the cells rounded up to the pad: all of those rows must be the zone's, not just the cells written
                    const size_t rd = std::min((max_cell + 1 + r.pad - 1) / r.pad * r.pad, (size_t) r.cells_max);
                    if (rd > ap.kv_h) { return false; }
                    if (kv_any_out(ap) && max_cell + 1 > ap.host_cells) { return false; }
                    lo = ap.kv_start + ap.kv_fixed; hi = lo + rd * ap.kv_row;
                }
                for (auto & sg : ap.segs) {
                    const auto & ps = mc->pool[sg.pcls];
                    for (int32_t b = 0; b < sg.n; ++b) {
                        const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                        if (b1 > lo && b0 < hi && ps.owner[b] != -3) { return false; }
                    }
                }
                break;   // this card covers the cell: the cache has records on the other cards of the pool too (all must)
            }
        }
    }
    for (auto & ar : g_kv_arena) {   // every arena the cache has records on must cover the cell, not the first
        for (auto & q : ar.runs) {
            if (q.owner != owner) { continue; }
            if (q.fixed ? (max_cell + 1) > q.cells_max : (max_cell + 1 + ar.pad - 1) / ar.pad * ar.pad > kv_arena_cells(ar)) { return false; }
            break;
        }
    }
    return true;
}


// The attention reads the cells rounded up to the pad (256): the rows between the cells in use and that boundary are masked, but they are read, and in the zone
// they are bytes of evicted experts (NaN/Inf in a half float: 0 x NaN is NaN in the softmax). Only that window is zeroed (whole combined rows: every record of
// the card): no block is cleared at a claim; the window moves once per 256 generated tokens and is set up again when a request starts or the zone moves.
static void kv_tail_pad_zero(moe_cache * mc) {
    for (auto & ap : mc->alias) {
        if (ap.kvt.empty() || ap.kv_row == 0) { continue; }
        size_t cells = 0, pad = 1;
        for (auto & r : ap.kvt) { if (!r.fixed) { cells = std::max(cells, r.rows ? r.rows() : 0); pad = std::max(pad, r.pad); } }
        if (cells < ap.kv_last_cells) { ap.kv_zhi = 0; }   // the cache shrank (a new request): the window is set up again
        ap.kv_last_cells = cells;
        if (cells == 0) { continue; }
        const size_t hi = std::min<size_t>((cells + pad - 1) / pad * pad, ap.kv_h);
        const size_t lo = std::max(ap.kv_zhi, cells);
        if (hi <= lo) { continue; }
        ggml_tensor z = {};
        z.type = GGML_TYPE_I8; z.buffer = ap.buf;
        z.data = (uint8_t *) ggml_backend_buffer_get_base(ap.buf) + ap.kv_start + ap.kv_fixed + lo * ap.kv_row;
        led_line(led_card(ap), "kv-pad", ap.kv_start + ap.kv_fixed + lo * ap.kv_row, ap.kv_start + ap.kv_fixed + hi * ap.kv_row, tr_fmt("rows %zu-%zu zeroed", lo, hi));
        z.ne[0] = (int64_t) ((hi - lo) * ap.kv_row); z.ne[1] = z.ne[2] = z.ne[3] = 1;
        z.nb[0] = 1; z.nb[1] = z.nb[2] = z.nb[3] = (size_t) z.ne[0];
        ggml_backend_tensor_memset(&z, 0, 0, (size_t) z.ne[0]);
        ap.kv_zhi = hi;
    }
}

std::vector<llama_kv_tail_stat_fork> llama_moe_cache_kv_tail_stats_fork(bool reset) {
    std::vector<llama_kv_tail_stat_fork> out;
    moe_cache * mc = g_cache;
    if (!mc) { return out; }
    std::lock_guard<std::mutex> lk(mc->mtx);
    for (auto & ap : mc->alias) {
        llama_kv_tail_stat_fork st;
        st.dev = ggml_backend_dev_name(ap.dev);
        size_t cap = (size_t) -1;
        for (auto & r : ap.kvt) { st.kv_cells = std::max(st.kv_cells, r.cells); st.kv_bytes += std::min(r.bytes, r.cells * r.row); if (!r.fixed) { cap = std::min(cap, kv_prompt_room(ap)); } }   /* a fixed record (a sliding-window ring) does not limit how long a request can be: MiMo reported 16640 tokens, the ring size */   // cells: the longest cache (a model with a window cache and a global one holds the same tokens in both)
        st.cap_tokens = cap == (size_t) -1 ? 0 : cap;
        for (auto & sg : ap.segs) {
            const auto & ps = mc->pool[sg.pcls];
            for (int32_t b = 0; b < sg.n; ++b) {
                st.blocks++;
                const int32_t o = ps.owner[b];
                if (o == -1) { st.blocks_free++; } else if (o == -2) { st.blocks_lent++; } else if (o == -3) { st.blocks_kv++; }
                if (o != -2 && o != -3) { st.moe_bytes += sg.B; }   // blocks the experts can use now (cached, uploading or free)
            }
        }
        out.push_back(std::move(st));
    }
    for (auto & ar : g_kv_arena) {   // dense: the cards with an arena
        llama_kv_tail_stat_fork st;
        st.dev = ggml_backend_dev_name(ar.dev);
        size_t cap = (size_t) -1;
        for (auto & q : ar.runs) { const size_t c = q.rows ? q.rows() : 0; st.kv_cells = std::max(st.kv_cells, c); if (!q.fixed) { cap = std::min(cap, std::min(kv_arena_cells(ar), q.cells_max)); } }
        st.kv_bytes = st.kv_cells * ar.row + ar.fixed;
        st.cap_tokens = cap == (size_t) -1 ? 0 : cap;
        out.push_back(std::move(st));
    }
    if (!out.empty()) {
        out[0].claim_n = g_kvs.claim_n.load(); out[0].release_n = g_kvs.release_n.load(); out[0].evicted = g_kvs.evicted.load();
        if (reset) { g_kvs.claim_n = 0; g_kvs.release_n = 0; g_kvs.evicted = 0; }
    }
    return out;
}


const char * llama_moe_cache_alias_err_fork() { return g_alias_err.empty() ? nullptr : g_alias_err.c_str(); }

bool llama_moe_cache_alias_fork(ggml_backend_sched_t sched, int32_t n_tokens) {
    g_alias_err.clear();
    moe_cache * mc = g_cache;
    if (mc && !mc->alias.empty() && !alias_on()) {
        // the pool could not lend the graph (separate compute buffers) but the KV zones still live in it: the pad rows the attention reads past the cells in use must be
        // zeroed all the same. Skipped, they held whatever the blocks held before (poison: NaN, so the first graph's attention was NaN; OLMoE at 0.786 GiB VRAM left,
        // 2026-10-18; without poison stale expert bytes, the MiMo '!!!!' mechanism).
        kv_tail_pad_zero(mc);
    }
    if (!mc || !alias_on() || mc->alias.empty()) {
        return false;
    }
    struct lend_done { int64_t t0 = ggml_time_us(); int32_t n; ~lend_done() { g_ac.lend_us += ggml_time_us() - t0; g_ac.lend_n++; if (n > 64 && (g_ac.lend_n.load() % 8) == 0) { alias_cost_log("prompt"); } } } lend_guard{ggml_time_us(), n_tokens};
    {
        const int32_t mb = (int32_t) std::max<int64_t>(8, llama_moe_cache_max_batch_fork());
        if (n_tokens > mb && mc->graph_tokens_last <= mb) {
            pool_fill_log(mc, tr_fmt("before a prompt (first pass %d tokens)", n_tokens).c_str());
            mc->fill_from = -1;
        }
    }
    mc->graph_tokens_last = n_tokens;
    struct pfill { moe_cache::alias_pool * ap; size_t lo, hi; uint8_t v; };
    std::vector<pfill> pfills;   // --moe poison: applied after the scheduler sync below
    bool changed = false;   // (the KV zones were laid out for this graph before it was built: llama_moe_cache_kv_tail_prepare_fork)
    kv_tail_pad_zero(mc);   // the KV rows this graph writes first: the lend below stops in front of them (ap.span); a decode graph ends the plan of the last prompt
    for (auto & ap : mc->alias) {
        size_t peak = 0;
        ggml_backend_t ap_be = nullptr;   // this context's backend on the card: the only one the scratch region below serves
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(be) == ap.dev) { peak = ggml_backend_sched_graph_peak(sched, be); ap_be = be; }
        }
        // the backend's scratch goes into the lent blocks right after the graph's tensors. Its size is computed per card from the graph's own nodes
        // (alias_scratch); every graph then reports what it really asked for (fitting or not), and one that asked for more than it was lent is logged as an
        // error with both numbers: an op whose scratch mirror is missing or wrong (GGML_CUDA_SCRATCH_CHECK=1 names it). Nothing is learned from it.
        // the region serves only the backend passed with it (ap_be): another context on the card (a DSpark draft between two target steps) takes its scratch
        // from the device pool, and its asks do not count in the high-water mark compared below
        static size_t (*scratch_fn)(int, void *, size_t, ggml_backend_t) = [] {
            ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
            return reg ? (size_t (*)(int, void *, size_t, ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_scratch_fork") : nullptr;
        }();
        if (ap.cuda_dev < 0 && scratch_fn) {
            const char * nm = ggml_backend_dev_name(ap.dev);
            if (nm && strncmp(nm, "CUDA", 4) == 0) { ap.cuda_dev = atoi(nm + 4); }
        }
        size_t scratch = 0;
        if (ap.cuda_dev >= 0) {
            const size_t asked = scratch_fn(ap.cuda_dev, nullptr, 0, nullptr);   // what the previous graph asked for, fitting or not
            ap.hwm = std::max(ap.hwm, asked);
            if (asked > ap.last_scr && ap.last_n > 0) {
                // the previous graph asked for more than it was lent: the rest came from the device pool (the VRAM beside the pool)
                static std::atomic<int> said { 0 };
                if (said++ < 16) {
                    LLAMA_LOG_ERROR("moe-cache: alias %s: SCRATCH SHORT: a %d-token graph asked for %zu bytes, %zu were computed and lent (%zu short, taken from the device pool): "
                            "an op's scratch mirror is missing or wrong, GGML_CUDA_SCRATCH_CHECK=1 names it\n",
                            ggml_backend_dev_name(ap.dev), ap.last_n, asked, ap.last_scr, asked - ap.last_scr);
                }
            }
            scratch = alias_scratch(ap, sched);   // computed per card from its own nodes, exactly (no slack on top)
            ap.last_n   = n_tokens;
            ap.last_scr = scratch;
        }
        // the graph's tensors start at base_off: behind the dense split's copies when the graph splits (llama_moe_cache_alias_offset_fork), else at 0
        const size_t peak_al = GGML_PAD(ap.base_off + peak, 256);
        // prefetch overlap: the staging slots of the prompt batch (full expert tensors uploaded ahead on a second stream) are lent blocks too,
        // after the graph and its scratch; only on the GPU that prompt batches are offloaded to
        size_t stage = 0;
        const size_t stage_off = GGML_PAD(peak_al + scratch, 256);
        // every card that runs staged expert matmuls in this graph gets its slots from its own pool (the multi-GPU prompt split runs half of the experts on each card).
        // As many slots as the class still holds behind the graph, up to the configured number, at least 3 (the scheduler's ring adapts to the region it gets,
        // see ggml_backend_sched_prefetch_prepare): a big micro-batch runs with fewer slots, or none, instead of the micro-batch being cut for them.
        {
            const size_t slot_b = ggml_backend_sched_prefetch_slot_bytes_dev(sched, ap.dev);
            const size_t cap_b  = kv_lend_potential(ap, (size_t) n_tokens);   // the slots may reach past the KV zone: it moves behind them (above)
            const int    n_cfg  = std::max(0, ggml_backend_sched_get_prefetch_experts_slots(sched));
            const int    fit    = ppx_stage_slots(slot_b, cap_b, stage_off, n_cfg, n_tokens >= 64 ? g_pp_k.load() : -1);   // the pass planner's choice: more slots evict more cached experts
            stage = slot_b * (size_t) fit;
            ap.slot_b  = slot_b;
            ap.n_stage = fit;
            ap.n_cfg   = n_cfg;
            // graph bytes per token (the prompt pass planner), from the biggest graph seen. Not the largest ratio of any graph: a 64-token graph costs 2.6x more per
            // token than a 512-token one (its fixed part is spread over 8x fewer tokens), and the max-batch calibration of a first start runs 64..512-token graphs
            // before the first prompt, so the max made every pass look 2.6x heavier: a 512-token pass "did not fit" the 1.0 GiB pool and the first prompts ran in
            // passes of 320 (Qwen3.6-35B 12k, 934 prompt t/s against 1490 with a state file, which skips that calibration).
            if (n_tokens >= 64 && (uint32_t) n_tokens >= ap.graph_tok_n) {
                ap.graph_tok   = (peak    + n_tokens - 1) / n_tokens;
                ap.scr_tok     = (scratch + n_tokens - 1) / n_tokens;
                ap.graph_tok_n = (uint32_t) n_tokens;
            }
        }
        const size_t need_b = stage_off + stage;   // the bytes this graph borrows from the start of the buffer: its tensors, the CUDA scratch, the staging slots
        if (!ap.need_tok.empty() || !ap.kvt.empty()) {   // remembered per graph size: the next graph of this size gets its lend before it is built (prepare)
            size_t cells_now = 0;
            for (auto & r : ap.kvt) { if (!r.fixed) { cells_now = std::max(cells_now, r.rows ? r.rows() : 0); } }
            // the graph's own bytes (tensors + CUDA scratch, stage_off), not the staging slots: the pass planner sizes passes on them and the slots take what the
            // pool holds behind the graph (user rule 2026-10-24). With the slots in (GLM 3.0: 3 x 2.6 GiB) the fit cut the expert card's passes from 6176 to 3776
            // tokens while 3.5 GiB of the lendable pool stayed free.
            size_t & nt = ap.need_tok[n_tokens];
            if (stage_off >= nt) { ap.need_cells[n_tokens] = cells_now; }
            nt = std::max(nt, stage_off);
        }
        // the lend takes whole blocks: a graph of need_b bytes holds ceil(need_b / bmax) * bmax of the buffer (the largest size class), so it reaches the KV zone when THAT passes the
        // zone's start, not need_b (MiMo: a 10.8 MB lend took the 9.2 MB blocks 0 and 1, up to 18.3 MB, while the zone began at 12.6 MB inside block 1 and was never claimed)
        const size_t lend_end = ap.kvt.empty() ? need_b : kv_lend_round(ap, need_b);
        if (ap.refused_b && need_b >= ap.refused_b) {   // refused before and the zone has not moved since: the same answer, though the blocks are lent already
            if (g_alias_err.empty()) { g_alias_err = "kv-tail: a " + std::to_string(n_tokens) + "-token pass needs " + std::to_string(need_b / 1048576) + " MiB lent from the start of the pool: refused before (the compute graph and the KV do not fit)"; }
            g_refuse_need.store(need_b);
            g_refuse_room.store(ap.refused_room);
            break;
        }
        if (!ap.kvt.empty() && lend_end > ap.span) {
            // the graph reaches into the KV zone: the zone moves behind it (a device copy of the rows in use), and since this graph's views of the KV point at the old
            // place it is built again (the caller). Never silently over the KV, and no abort: only when the zone cannot fit behind the graph at all.
            const size_t bmax = kv_bmax(ap);
            size_t cells = 0;
            for (auto & r : ap.kvt) { if (!r.fixed) { cells = std::max(cells, r.rows ? r.rows() : 0); } }
            {   // the attention reads the cells in use rounded up to the pad: the zone must hold those rows too (OLMoE on W10, 2026-10-18: 33568 cells in use, a zone of 33740 after the lend grew, padded read 33792, abort in set_input_k_idxs)
                size_t pad = 1, cmx = 0;   // cmx: the most cells of a growing record (a compressed cache's fewer rows never cap the others)
                for (auto & r : ap.kvt) { if (!r.fixed) { pad = std::max(pad, r.pad); cmx = std::max(cmx, r.cells_max); } }
                cells = std::min((cells + pad - 1) / pad * pad, cmx);
            }
            // the zone moves behind exactly this graph's lend, its start on the block edge the lend ends on (no headroom: a bigger graph later moves it again, a
            // device copy of the rows in use, ~2.5 ms per GB; 25% headroom pushed MiMo's zone 3.4 GiB further back and the decode then found no room for its cells)
            const size_t keep = ap.kv_fixed + std::max(cells, (size_t) 1) * ap.kv_row;   // the zone as it is now must still fit behind the lend
            const size_t lr_max = ap.tail_end > keep ? kvz_block_floor(ap, ap.tail_end - keep) : 0;
            const size_t lr = kv_lend_round(ap, need_b);   // the zone starts where the lend really ends
            if (lr + keep > ap.tail_end) {
                LLAMA_LOG_ERROR("moe-cache: kv-tail %s: a %d-token graph needs %.1f MiB lent from the start of the pool, and %zu cells of KV (%.1f MiB) do not fit behind it in the pool of %.1f MiB\n",
                        ggml_backend_dev_name(ap.dev), n_tokens, need_b / 1048576.0, cells, cells * ap.kv_row / 1048576.0, ap.tail_end / 1048576.0);
                // a clean refusal of this pass (the caller fails the decode, the server answers the request with an error) instead of killing the process
                g_alias_err = "kv-tail: the compute graph and the KV do not fit in the pool: a " + std::to_string(n_tokens) + "-token pass needs " + std::to_string((size_t) (need_b / 1048576)) +
                              " MiB lent from the start of the pool and " + std::to_string(cells) + " cells of KV do not fit behind it";
                g_refuse_need.store(need_b);   // what the dense split's pass loop shrinks the pass by (llama_moe_cache_refusal_fork)
                g_refuse_room.store(lr_max);
                ap.refused_room = lr_max;
                ap.refused_b = ap.refused_b ? std::min(ap.refused_b, need_b) : need_b;
                break;   // not return: cards before this one may have lent blocks in this call, and the flush and sync below must still cover them
            }
            ggml_backend_sched_synchronize(sched);   // the previous graph may still read the rows
            size_t h = ap.kv_h;
            const size_t room = (ap.tail_end - lr - ap.kv_fixed) / std::max<size_t>(1, ap.kv_row);
            h = std::min(h, std::max(room, cells));
            kv_layout(mc, ap, h, lr, cells, false, lr);
            poison_free_blocks(mc, ap);   // --moe poison: the zone's old place holds nothing now
            flush_tables();
            return true;
        }
        ap.want_b = std::min(need_b, ap.span);
        if (n_tokens < 64) {
            if (need_b >= ap.dec_need) { size_t c = 0; for (auto & r : ap.kvt) { if (!r.fixed) { c = std::max(c, r.rows ? r.rows() : 0); } } ap.dec_cells = c; }
            ap.dec_need = std::max(ap.dec_need, need_b);
        }
        ap.want   = ap.segs.empty() ? 0 : (int32_t) std::min<size_t>((size_t) ap.n, (ap.want_b + ap.B - 1) / ap.B);
        if (ap.cuda_dev >= 0) {
            scratch_fn(ap.cuda_dev, scratch > 0 ? (char *) ggml_backend_buffer_get_base(ap.buf) + peak_al : nullptr, scratch, ap_be);
        }
        {
            // the staging slots come only from lent blocks: if the pool cannot lend them, this card's splits of this graph run without prefetch (never a new buffer)
            if (stage > 0 && need_b <= ap.span) {
                ggml_backend_sched_set_prefetch_region_dev(sched, ap.dev, ap.buf, stage_off, stage, true);
            } else {
                ggml_backend_sched_set_prefetch_region_dev(sched, ap.dev, nullptr, 0, 0, true);
            }
        }
        static const bool dbg = moe_log_has("a");   // --moe log=a: the compute alias per graph
        if (dbg) {
            LLAMA_LOG_WARN("moe-cache: alias %s: %d tokens, graph %.1f MiB + scratch %.1f MiB + %d staging slots (largest scratch asked so far %.1f MiB) = %.1f of %.1f MiB lendable\n",
                    ggml_backend_dev_name(ap.dev), n_tokens, peak / 1048576.0, scratch / 1048576.0, ap.n_stage, ap.hwm / 1048576.0, need_b / 1048576.0, ap.span / 1048576.0);
        }
        for (auto & sg : ap.segs) {
            const int32_t need = poolx_lend_blocks(sg, need_b);   // the blocks of this class the range covers
            if (need <= sg.blocked) {
                continue;
            }
            auto & ps = mc->pool[sg.pcls];
            const int32_t blocked = sg.blocked;
            // uploads into these blocks: a queued one is dropped, a finished but unpublished one is discarded, a running one is awaited (one copy);
            // an adoption is cancelled (its device copy was queued before this graph on the same stream). The step publishes uploads only later,
            // so waiting for the publish would never end.
            const int32_t want_pcls = sg.pcls;
            auto in_zone = [&](const upload_job & j) {
                return j.layer_idx < mc->layers.size() && mc->layers[j.layer_idx].pcls == want_pcls && j.slot >= blocked && j.slot < need;
            };
            auto drop_job = [&](const upload_job & j) {
                layer_state & lo = mc->layers[j.layer_idx];
                lo.slot_in_flight[j.slot] = false;
                if (j.expert >= 0 && j.expert < (int32_t) lo.queued.size()) { lo.queued[j.expert] = 0; }
            };
            for (int spin = 0; ; ++spin) {
                bool busy = false;
                {
                    std::lock_guard<std::mutex> wlk(mc->wmtx);
                    std::lock_guard<std::mutex> lk(mc->mtx);
                    for (auto it = mc->todo.begin(); it != mc->todo.end(); ) {
                        if (in_zone(*it)) { drop_job(*it); it = mc->todo.erase(it); } else { ++it; }
                    }
                    for (size_t k = 0; k < mc->done.size(); ) {
                        if (in_zone(mc->done[k])) { drop_job(mc->done[k]); mc->done[k] = mc->done.back(); mc->done.pop_back(); } else { ++k; }
                    }
                    for (int32_t b = blocked; b < need && !busy; ++b) {
                        const int32_t o = ps.owner[b];
                        if (o < 0 || !mc->layers[o].slot_in_flight[b]) {
                            continue;
                        }
                        layer_state & lo = mc->layers[o];
                        bool adoption = false;
                        for (size_t i = 0; i < lo.adopted.size(); ++i) {
                            const int32_t e = lo.adopted[i];
                            if (lo.adopt_slot[e] == b) {
                                lo.adopt_slot[e] = -1;
                                lo.adopt_mask[e] = 0;
                                lo.adopted.erase(lo.adopted.begin() + (ptrdiff_t) i);
                                lo.slot_in_flight[b] = false;
                                adoption = true;
                                break;
                            }
                        }
                        busy = !adoption;   // a worker is copying into it right now
                    }
                }
                if (!busy) {
                    break;
                }
                GGML_ASSERT(spin < 200000 && "compute alias: an upload into the borrowed blocks never finished");
                g_ac.drain_us += 50;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            std::lock_guard<std::mutex> lk(mc->mtx);
            if (n_tokens > llama_moe_cache_max_batch_fork()) {
                keep_hot_move(mc, sched, ap, sg, blocked, need);
            }
            if (poolx_lend(ps, blocked, need, [&](int32_t o, int32_t b) {
                    layer_state & lo = mc->layers[o];
                    evict_slot(mc, lo, b);
                    lo.n_own--;
                    g_ac.evict_n++;
                }) > 0) {
                LLAMA_LOG_WARN("pool-gate: a lend of blocks %d-%d of class %d reached blocks under the KV zone on %s (not taken)\n", (int) blocked, (int) need, (int) sg.pcls, ggml_backend_dev_name(ap.dev));
                // the graph is placed over those blocks all the same: it would overwrite live KV rows ('!!!!' / NaN many tokens later). Refuse this pass (the caller fails the decode).
                ap.refused_b = ap.refused_b ? std::min(ap.refused_b, need_b) : need_b;
                if (g_alias_err.empty()) { g_alias_err = std::string("kv-tail: a lend of blocks ") + std::to_string(blocked) + "-" + std::to_string(need) + " of class " + std::to_string(sg.pcls) + " reaches blocks under the KV zone on " + ggml_backend_dev_name(ap.dev); }
            }
            sg.blocked = need;
            sg.hwm = std::max(sg.hwm, need);
            changed = true;
        }
        ap.blocked = ap.segs.empty() ? 0 : ap.segs[0].blocked;
        if (led_on()) {   // the bytes this graph uses from the start of the pool, and what is in them
            const std::string who = tr_fmt("graph %d tokens: copies [0, %zu) graph [%zu, %zu) scratch [%zu, %zu) staging %d x %zu [%zu, %zu)", n_tokens, ap.base_off,
                    ap.base_off, peak_al, peak_al, stage_off, ap.n_stage, ap.slot_b, stage_off, need_b);
            led_line(led_card(ap), "lend", 0, need_b, who);
            led_check_range(ap, "lend", 0, kv_lend_round(ap, need_b), who, true, false);
        }
        if (poison_on()) { pfills.push_back({ &ap, ap.base_off, peak_al, 1 }); pfills.push_back({ &ap, peak_al, stage_off, 2 }); pfills.push_back({ &ap, stage_off, need_b, 3 }); }
        if (ap.kv_h > 0 && kv_lend_round(ap, need_b) > ap.kv_start) {
            LLAMA_LOG_WARN("pool-gate: lend of %.1f MiB on %s reaches the KV zone at %.1f MiB\n", kv_lend_round(ap, need_b) / 1048576.0, ggml_backend_dev_name(ap.dev), ap.kv_start / 1048576.0);
            if (g_alias_err.empty()) { g_alias_err = std::string("kv-tail: a lend of ") + std::to_string(kv_lend_round(ap, need_b) / 1048576) + " MiB on " + ggml_backend_dev_name(ap.dev) + " reaches the KV zone at " + std::to_string(ap.kv_start / 1048576) + " MiB"; }
        }
    }
    if (changed) {
        const int64_t s0 = ggml_time_us();
        flush_tables();
        ggml_backend_sched_synchronize(sched);   // the previous graph may still read those blocks; this graph's inputs are written next
        for (const auto & f : pfills) { pool_fill(*f.ap, f.lo, f.hi, f.v); }   // --moe poison
        g_ac.sync_us += ggml_time_us() - s0; g_ac.sync_n++;
    }
    return false;
}

// at a sync point: blocks above what the last graph needed go back to the cache. Returns how many came back.
static size_t alias_release(moe_cache * mc) {
    struct rel_done { int64_t t0 = ggml_time_us(); ~rel_done() { g_ac.release_us += ggml_time_us() - t0; g_ac.release_n++; } } rel_guard;
    size_t n_back = 0;
    for (auto & ap : mc->alias) {
        for (auto & sg : ap.segs) {
            const size_t  cov  = ap.want_b > sg.base ? ap.want_b - sg.base : 0;
            const int32_t keep = (int32_t) std::min<size_t>((size_t) sg.n, (cov + sg.B - 1) / sg.B);
            if (keep < sg.blocked) {
                auto & ps = mc->pool[sg.pcls];
                n_back += poolx_release(ps, keep, sg.blocked);
                led_line(led_card(ap), "release", sg.base + (size_t) keep * sg.B, sg.base + (size_t) sg.blocked * sg.B, tr_fmt("class %d blocks %d-%d back to the experts", sg.pcls, keep, sg.blocked));
                sg.blocked = keep;
            }
        }
        ap.blocked = ap.segs.empty() ? 0 : ap.segs[0].blocked;
    }
    if (n_back > 0) { alias_cost_log("after a prompt"); }
    return n_back;
}

// After a prompt: fill the blocks it borrowed.
// The passes of a prompt take pool blocks for their graph, the backend's scratch and the staging slots (llama_moe_cache_alias_fork) and drop the experts
// that sat in them; the blocks come back free at the first decode steps (alias_release). Left alone they fill slowly: a missed expert takes even a free
// block only after `admit` uses within 64 tokens, a step schedules a few uploads per layer, and nothing the prompt staged is kept on the staged path
// (GGML_PF_COUNT). So every answer after a long prompt started with a part of its cache empty, and its first tokens computed those experts on the CPU.
// Here every free block of a size class gets the best expert that is not cached, by the cache's own score on the scale that compares layers (xscore:
// recent use, which the prompt's hit counters feed, plus the lifetime share that the saved hot map seeds), best first over all layers of the class, so
// the blocks go where the score says an expert is missed most, as in the steady state. The uploads are background jobs (upload_job::bg): the decode step
// does not wait for them, a worker takes one only when no miss-driven upload waits on its link, and they cost no swap budget (a free block evicts
// nothing). The link is idle right after a prompt: the refill of 10 GB takes about 0.5 s of a x16 link, 2 s of a x4.
// --moe reheat=0 turns it off. With --moe fixed=1 (repeatable runs) the jobs are ordinary ones the step waits for and publishes in order.
static size_t reheat_free_blocks(moe_cache * mc, bool wait, bool ini_only = false) {   // ini_only: by the lifetime counts alone (the state ini's hot map and this run's use), no recent window
    static const bool on = [] { const char * e = moe_opt("reheat"); return !e || atoi(e) != 0; }();
    if (!on) {
        return 0;
    }
    struct cand { double sc; size_t li; int32_t e; };
    std::vector<upload_job> jobs;
    double bytes = 0.0;
    for (size_t pc = 0; pc < mc->pool.size(); ++pc) {
        const size_t n_free = mc->pool[pc].free.size();
        if (n_free == 0) {
            continue;
        }
        std::vector<cand> c;
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            const layer_state & ls = mc->layers[li];
            if (ls.pcls != (int32_t) pc) {
                continue;
            }
            for (int32_t e = 0; e < (int32_t) ls.expert_slot.size(); ++e) {
                if (ls.expert_slot[e] >= 0 || ls.queued[e] || ls.adopt_slot[e] >= 0) {
                    continue;
                }
                // (hot maps per topic: the score, whose lifetime term is the topic's map and the request's own uses, not the ini's map)
                const double sc = ini_only && !g_topic_on.load(std::memory_order_relaxed) ? (double) ls.glob_count[e] / (double) std::max<uint64_t>(1, mc->gmax_all)
                                                                                         : xscore(mc, ls, e);
                if (sc > 0.0) {
                    c.push_back({ sc, li, e });
                }
            }
        }
        const size_t n = std::min(n_free, c.size());
        std::partial_sort(c.begin(), c.begin() + (ptrdiff_t) n, c.end(), [](const cand & a, const cand & b) { return a.sc > b.sc; });
        for (size_t i = 0; i < n; ++i) {
            layer_state & ls = mc->layers[c[i].li];
            double best;
            const int32_t slot = pick_slot(mc, c[i].li, [](layer_state &, size_t, int32_t) { return -1.0; }, best).slot;   // free blocks only
            if (slot < 0) {
                break;
            }
            pool_take(mc, c[i].li, slot);
            ls.slot_in_flight[slot] = true;
            ls.queued[c[i].e]       = 1;
            upload_job j { c[i].li, c[i].e, slot };
            j.bg       = !wait;
            j.t_queued = ggml_time_us();
            jobs.push_back(j);
            bytes += (double) (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
        }
    }
    if (jobs.empty()) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        for (const auto & j : jobs) {
            mc->todo.push_back(j);
        }
        mc->n_reheat += jobs.size();
    }
    mc->wcv.notify_all();
    static int logged = 0;
    if (logged++ < 16) {
        LLAMA_LOG_WARN("moe-cache: after the prompt: %zu experts (%.2f GB) queued to refill the blocks it borrowed, best score first%s\n", jobs.size(), bytes / 1e9,
                wait ? "" : " (background uploads)");
    }
    return jobs.size();
}

// ---- hot maps per topic (task 158; user idea 2026-10-27; the store: llama-moetopic-fork.h) ---------------------------------------------------------
// --moe dynhot=full|store|read|off (1 = full, 0 = off; default store, user 2026-10-28: "dynhot full/onlystore/onlyread/off meaning only store uses ini hotmoes
// for decode"): the hottest experts of every request's answer (topic_dec, as many as the cache holds: topic_hottest) and its decode hit rate are stored
// under its prompt's keys, the mean and the max of the input embeddings of the request's own text (the chat template's user parts; a /completion prompt
// with the template's markers cut); a prompt stored before changes its record only for a better hit rate. store collects only: the cache runs on the
// ini's hot experts, so the maps the ini steers and the maps dynhot steers (full) can be compared (each record says which); read steers from a frozen file
// and stores nothing (reg tests). With full and read the first pass of the next prompt looks up the 4 stored records nearest to it and merges their maps,
// each weighted by its cosine (topic_heat, 0..1 per expert): a query not asked before takes the maps of the nearest topics, at least as near to it as the
// ini's map of everything (user 2026-10-28: "cos dist always picks one that is simmilar to what ini had maybe even better"); --moe dynhot-min=X takes only
// records at least X alike. With nothing stored the ini picks, as before. From then on:
//   - score() takes the topic's map plus this request's own uses as its lifetime term, not the counts of every request ever served (the ini's hot map):
//     those kept other topics' experts in the cache, so a new topic evicted them before its own cold ones (user: "so tetris doesnt spend 2nd run evicting
//     stubborn irrelevant moes, just his own cold");
//   - the prompt runs on whatever the cache holds (an expert it finds there is one it does not stream);
//   - at the first decode step after it, every cached expert that neither the map rates nor the request used goes (topic_drop_foreign), and the refill after
//     a prompt (reheat_free_blocks, by score) puts the topic's and the request's experts in those blocks.
// Every prompt logs its best similarity: the threshold is calibrated from real prompts. The store lies in the user's cache directory (hotmaps/, one file per
// model; --moe hotmaps=DIR moves it, state=PATH does not: one set of maps collects from every run, clean-state benchmark cells included); the ini's hot map
// is still counted and saved as before.
static struct topic_t {
    int                 on = -1;          // -1: not decided yet (the first prompt opens the store)
    moe_topic_store     store;
    std::vector<float>  key;              // the key of the request being served: the rows at its prompt's last token as the model gave them (n_rows x n_embd)
    uint32_t            n_rows = 0;       // its rows: the router input of every MoE layer, then the final hidden state
    moe_topic_text      text;             // what it said, thought and answered (the first few hundred bytes of each: the record's line keeps them)
    uint64_t            phash = 0;        // a hash of its prompt's tokens: the same prompt is the same record
    bool                act_wait = false; // its prompt began: the layers' keys come with its last pass (llama_moe_topic_act_fork)
    bool                act_ok   = false; // they came: the request can be stored
    int32_t             n_prompt = 0;     // its prompt's tokens (the log)
    bool                own = false;      // its prompt text is the request's own text (the chat template's user parts), not the whole prompt
    llama_token         tok_open = -2, tok_close = -2;   // the model's <think> and </think> (LLAMA_TOKEN_NULL: none; -2: not looked up yet)
    bool                switched = false; // a </think> came in the answer: the answer map counts from there (topic_phase)
    uint64_t            h_sw = 0, m_sw = 0;   // the decode hits and misses at that switch: the two phases' hit rates
    // lookup: the 4 nearest records (only those at least min_cos alike; -1: any); store: a request at least merge_cos alike joins the record. Default 2:
    // never, every prompt a record of its own (user 2026-10-28: merging is offline, a merged map must hit about as well as its parts and keep the hottest of both)
    float               min_cos = -1.0f, merge_cos = 2.0f;
    int32_t             prev_n = 0;       // tokens of the previous decode call: a prompt starts where a call of more than a step follows a step
    bool                after_prompt = false;   // a prompt began: its first decode step drops the experts foreign to the topic and the request
    uint64_t h0 = 0, m0 = 0;   // decode hits and misses when the request began
    enum { OFF, FULL, STORE, READ };
    int  mode = STORE;
    bool steer = false;   // this request: stored maps pick the cache's experts (full or read, and a map found)
    bool use()  const { return mode == FULL || mode == READ; }    // the maps pick the cache's experts
    bool keep() const { return mode == FULL || mode == STORE; }   // requests are stored
} g_topic;

// up to 400 bytes of text into t (cut at a character boundary): what a record keeps of a request's words
static void topic_text_add(std::string & t, const std::string & piece) {
    if (t.size() >= 400) { return; }
    t += piece;
    if (t.size() > 400) {
        size_t n = 400;
        while (n > 0 && ((unsigned char) t[n] & 0xC0) == 0x80) { --n; }
        t.resize(n);
    }
}

// the hottest of a request's answer counts (cnt: layer x expert, cut in place), as many as the cache had room for (user 2026-10-28: "store only hottest
// for him as any other is just stealing precious moe cache space"; the cache's lifetime counts are the state over many requests, a record is one
// request's): per size class of the block pool the most used experts of all its layers, as many as the class has blocks for experts (free or holding one;
// not lent to compute, not under the KV) - the layers of a class compete for its blocks, so these are the fill that would have hit most; a layer with its
// own slots (--moe pool=0) keeps its n_cache most used. Counted over the whole answer, not read from the cache at its end: what most of its tokens needed
// may be gone by its last token. Returns the experts kept. Called with mc->mtx held.
static size_t topic_hottest(moe_cache * mc, std::vector<uint32_t> & cnt) {
    const uint32_t ne = g_topic.store.n_exp;
    size_t kept = 0;
    auto keep_top = [&](std::vector<size_t> & cells, size_t n) {   // all but the n most used of cells to 0
        if (cells.size() > n) {
            std::nth_element(cells.begin(), cells.begin() + (ptrdiff_t) n, cells.end(), [&](size_t a, size_t b) { return cnt[a] > cnt[b]; });
            for (size_t i = n; i < cells.size(); ++i) { cnt[cells[i]] = 0; }
        }
        kept += std::min(n, cells.size());
    };
    std::vector<std::vector<size_t>> by_cls(mc->pool.size());
    for (const auto & ls : mc->layers) {
        if (ls.pub.il < 0 || (uint32_t) ls.pub.il >= g_topic.store.n_layers) { continue; }
        std::vector<size_t>   own;
        std::vector<size_t> & cells = ls.pcls >= 0 && (size_t) ls.pcls < by_cls.size() ? by_cls[ls.pcls] : own;
        for (uint32_t e = 0; e < ne; ++e) {
            const size_t c = (size_t) ls.pub.il * ne + e;
            if (cnt[c] > 0) { cells.push_back(c); }
        }
        if (&cells == &own) { keep_top(own, (size_t) std::max(0, ls.n_cache)); }
    }
    for (size_t p = 0; p < by_cls.size(); ++p) {
        size_t blocks = 0;
        for (int32_t b = 0; b < mc->pool[p].n_active(); ++b) { blocks += mc->pool[p].owner[b] >= -1; }
        keep_top(by_cls[p], blocks);
    }
    return kept;
}

// the request just served: the hottest experts of its answer under its key, a record of its own (merging is offline: user 2026-10-28)
static void topic_store_pending(moe_cache * mc) {
    if (g_topic.key.empty() || g_topic.store.n_exp == 0 || !g_topic.keep() || !g_topic.act_ok) {
        if (!g_topic.key.empty() && g_topic.keep() && !g_topic.act_ok) {
            LLAMA_LOG_WARN("moe-cache: dynhot: no key from the prompt's last pass (%s): the request is not stored\n", g_topic.act_wait ? "it never came" : "not all three rows");
        }
        g_topic.key.clear();
        g_topic.act_wait = g_topic.act_ok = false;
        return;
    }
    // the two maps: the answer's uses (after </think>, or all of it) and the thinking phase's, each cut to what the cache holds
    std::vector<uint32_t> cnt(g_topic.store.rec_cells(), 0), cth(g_topic.store.rec_cells(), 0);
    uint64_t tot = 0, all = 0, n_used = 0, tot_t = 0, all_t = 0, n_used_t = 0;   // per map: uses kept, uses, experts used
    size_t   kept = 0, kept_t = 0;
    uint64_t h1 = 0, m1 = 0;
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & ls : mc->layers) {
            h1 += ls.n_hit;
            m1 += ls.n_miss;
            if (ls.pub.il < 0 || (uint32_t) ls.pub.il >= g_topic.store.n_layers) { continue; }
            for (size_t e = 0; e < ls.topic_dec.size() && e < g_topic.store.n_exp; ++e) {
                const size_t c = (size_t) ls.pub.il * g_topic.store.n_exp + e;
                cnt[c]    = ls.topic_dec[e];
                all      += ls.topic_dec[e];
                n_used   += ls.topic_dec[e] > 0;
                cth[c]    = ls.topic_think[e];
                all_t    += ls.topic_think[e];
                n_used_t += ls.topic_think[e] > 0;
            }
        }
        kept   = topic_hottest(mc, cnt);
        kept_t = all_t > 0 ? topic_hottest(mc, cth) : 0;
        for (const uint32_t c : cnt) { tot += c; }
        for (const uint32_t c : cth) { tot_t += c; }
    }
    if (tot > 0 || tot_t > 0) {
        const int64_t t0 = ggml_time_us();
        // the phases' decode hit rates (unknown: < 0): thinking from the request's start to the </think>, the answer after it (all of it without thinking)
        auto rate = [](uint64_t h, uint64_t m) { return h + m > 0 ? (float) ((double) h / (double) (h + m)) : -1.0f; };
        const bool  thought = all_t > 0;
        const uint64_t he = thought && g_topic.switched ? g_topic.h_sw : h1, me = thought && g_topic.switched ? g_topic.m_sw : m1;
        const float ht = thought ? rate(he - g_topic.h0, me - g_topic.m0) : -1.0f;
        const float hr = !thought ? rate(h1 - g_topic.h0, m1 - g_topic.m0) : g_topic.switched ? rate(h1 - he, m1 - me) : -1.0f;
        const float  mcos = g_topic.merge_cos;
        const size_t at = g_topic.store.add(g_topic.key.data(), g_topic.n_rows, cnt.data(), mcos, tot > 0 ? hr : -1.0f, g_topic.steer ? 1u : 0u, g_topic.phash,
                                            thought ? cth.data() : nullptr, thought ? ht : -1.0f, &g_topic.text);
        if (at == SIZE_MAX) {
            LLAMA_LOG_WARN("moe-cache: dynhot: a key of %u rows, the store holds keys of %u: the request is not stored\n", g_topic.n_rows, g_topic.store.n_rows);
            g_topic.key.clear();
            g_topic.act_ok = false;
            return;
        }
        const auto & st = g_topic.store;
        LLAMA_LOG_WARN("moe-cache: dynhot: request stored (hit %.1f%% with the cache on %s; its %zu hottest of %llu experts the answer used, %.1f%% of its uses%s): "
                "nearest record at cosine %.3f (joins at >= %.2f), maps overlap %.2f: %s record %zu (its hit %.1f%%, %u requests, %zu records, %.2f ms)\n",
                100.0 * hr, g_topic.steer ? "the hot maps" : "the ini's hot experts", kept, (unsigned long long) n_used, 100.0 * (double) tot / (double) std::max<uint64_t>(1, all),
                thought ? tr_fmt("; thinking: hit %.1f%%, its %zu hottest of %llu experts, %.1f%% of its uses%s", 100.0 * ht, kept_t, (unsigned long long) n_used_t,
                    100.0 * (double) tot_t / (double) std::max<uint64_t>(1, all_t), g_topic.switched ? "" : ", never closed").c_str() : "",
                st.last_cos, mcos, st.last_best, st.last_kept ? "kept (hit rate not better)" : st.last_same ? "the same prompt: map replaced in" : st.last_merged ? "joined" : "new",
                at, 100.0 * st.hit[at], st.n_req[at], st.size(), (ggml_time_us() - t0) / 1e3);
    }
    g_topic.key.clear();
    g_topic.act_ok = false;
}

// the prompt is over: every cached expert that neither the topic's map rates nor the request used goes (not a pinned one, not one an upload is writing), its
// block back to the free list for the refill that follows. Called by the step with mc->mtx held (a sync point; the step flushes the tables at its end).
static size_t topic_drop_foreign(moe_cache * mc) {
    size_t n = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        layer_state & ls = mc->layers[li];
        for (int32_t b = 0; b < (int32_t) ls.slot_expert.size(); ++b) {
            const int32_t e = ls.slot_expert[b];
            if (e < 0 || ls.slot_in_flight[b] || ls.is_stream[b] || pinned(mc, ls, e) || ls.topic_heat[e] > 0.0f || ls.topic_cnt[e] > 0) { continue; }
            evict_slot(mc, ls, b);
            pool_release(mc, li, b);
            n++;
        }
    }
    return n;
}

// the directory of the hot maps: --moe hotmaps=DIR (a frozen set for reg tests, a scratch set for an A/B), else hotmaps/ in the user's cache directory,
// whatever state file the run uses; "" with state=0 (no learned state at all) or no cache directory
static std::string hotmap_dir() {
    if (const char * d = moe_opt("hotmaps")) {
        return strcmp(d, "0") == 0 ? "" : d;
    }
    if (const char * s = moe_opt("state"); s != nullptr && strcmp(s, "0") == 0) {
        return "";
    }
    const std::string u = moe_user_dir();
    return u.empty() ? "" : u + "/hotmaps";
}

// <hot map dir>/<model section><ext>: the files of the hot maps per topic and of the per-expert tokens, one per model ("" when there is no directory)
static std::string hotmap_path(const llama_model & model, const char * ext) {
    const std::string dir = hotmap_dir();
    if (dir.empty()) {
        return "";
    }
    std::string sec = moe_state_section(model);
    for (char & ch : sec) { if (!isalnum((unsigned char) ch) && ch != '-' && ch != '.') { ch = '_'; } }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir + "/" + sec + ext;
}

// --moe expids: the rings to the file (written beside it and renamed over it, so a crash leaves the old file whole); only when tokens came in since
static void exptok_save() {
    if (g_exptok.n <= 0 || !g_exptok.dirty || g_exptok.path.empty()) {
        return;
    }
    const std::string tmp = g_exptok.path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        return;
    }
    static const char magic[8] = { 'M', 'O', 'E', 'X', 'P', 'I', 'D', '1' };
    const uint32_t h[4] = { g_exptok.n_layers, g_exptok.n_exp, (uint32_t) g_exptok.n, 0 };
    bool ok = fwrite(magic, 1, 8, f) == 8 && fwrite(h, sizeof(uint32_t), 4, f) == 4
           && fwrite(g_exptok.seen.data(), sizeof(uint32_t), g_exptok.seen.size(), f) == g_exptok.seen.size()
           && fwrite(g_exptok.ring.data(), sizeof(int32_t), g_exptok.ring.size(), f) == g_exptok.ring.size();
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::filesystem::rename(tmp, g_exptok.path);
    } else {
        std::remove(tmp.c_str());
    }
    g_exptok.dirty = false;
}

// --moe expids, every decode call: set up at the first (read the file of an earlier run: same layers and experts; another ring size keeps the most recent
// tokens that fit), the call's token ids for the observer, and at the start of a prompt the tokens so far to the file
static void exptok_call(const llama_model & model, const int32_t * ids, int32_t n, bool first) {
    if (g_exptok.n < 0) {
        const char * v = moe_opt("expids");
        g_exptok.n = v != nullptr && model.hparams.n_expert > 0 ? std::max(0, atoi(v)) : 0;
        if (g_exptok.n > 0) {
            const uint32_t L = (uint32_t) model.hparams.n_layer(), X = (uint32_t) model.hparams.n_expert, N = (uint32_t) g_exptok.n;
            g_exptok.n_layers = L;
            g_exptok.n_exp    = X;
            g_exptok.ring.assign((size_t) L * X * N, -1);
            g_exptok.seen.assign((size_t) L * X, 0);
            g_exptok.path = hotmap_path(model, ".expids");
            const char * how = "a new file";
            if (FILE * f = g_exptok.path.empty() ? nullptr : fopen(g_exptok.path.c_str(), "rb")) {
                char m[8]; uint32_t h[4];
                if (fread(m, 1, 8, f) == 8 && memcmp(m, "MOEXPID1", 8) == 0 && fread(h, sizeof(uint32_t), 4, f) == 4 && h[0] == L && h[1] == X && h[2] > 0) {
                    std::vector<uint32_t> seen((size_t) L * X);
                    std::vector<int32_t>  ring((size_t) L * X * h[2]);
                    if (fread(seen.data(), sizeof(uint32_t), seen.size(), f) == seen.size() && fread(ring.data(), sizeof(int32_t), ring.size(), f) == ring.size()) {
                        for (size_t c = 0; c < seen.size(); ++c) {   // the most recent min(seen, old size, new size) tokens, each at its place in the new ring
                            const uint32_t keep = std::min({ seen[c], h[2], N });
                            for (uint32_t k = keep; k >= 1; --k) {
                                g_exptok.ring[c * N + (seen[c] - k) % N] = ring[c * h[2] + (seen[c] - k) % h[2]];
                            }
                            g_exptok.seen[c] = seen[c];
                        }
                        how = "read";
                    }
                } else {
                    how = "of another model or layer split: not used, not written";
                    g_exptok.path.clear();   // never overwrite a file this run cannot read
                }
                fclose(f);
            }
            LLAMA_LOG_WARN("moe-cache: expids: the last %d token ids per expert (%u layers x %u experts, %.1f MiB) in %s: %s\n", g_exptok.n, L, X,
                    (double) g_exptok.ring.size() * sizeof(int32_t) / 1048576.0, g_exptok.path.empty() ? "(no file)" : g_exptok.path.c_str(), how);
        }
    }
    if (g_exptok.n <= 0) {
        return;
    }
    if (first) {
        exptok_save();   // a new prompt: the request before it to the file
    }
    if (ids != nullptr && n > 0) {
        g_exptok.batch.assign(ids, ids + n);
    } else {
        g_exptok.batch.clear();
    }
}

// a request started (the server's hook): the next call starts its prompt, however short it is (topic_begin). Not set before the first request: the
// server's warm-up decodes at load are no prompt (a record of their 2 tokens would be stored at every start)
static std::atomic<bool> g_req_boundary{false};
static std::vector<int32_t> g_req_prompt;   // the tokens of the topic key of the request that started (the server's thread, as decode)
static bool                 g_req_own = false;   // they are the request's own text (a chat request's user input), not its whole prompt

// the server's hooks (llama-fork-hooks.h): a request starts; a request ended: the per-expert token ids to the file, so the file is whole while the server waits
void llama_moe_request_begin_fork(const int32_t * prompt, int32_t n_prompt, bool own_text) {
    g_req_prompt.assign(prompt, prompt + (prompt != nullptr ? std::max(0, n_prompt) : 0));
    g_req_own = own_text;
    g_req_boundary.store(true);
}

void llama_moe_request_end_fork(void) {
    exptok_save();
    if (g_cache != nullptr) {
        topic_store_pending(g_cache);   // the request's map now, not when the next prompt begins: a server's last request is kept too
    }
}

static void topic_phase(const llama_model & model, const int32_t * ids, int32_t n, bool first);   // below: the thinking phase

void llama_moe_cache_topic_begin_fork(const llama_model & model, const int32_t * ids, int32_t n) {
    moe_cache * mc = g_cache;
    const int32_t mb = (int32_t) std::max<int64_t>(8, llama_moe_cache_max_batch_fork());
    // the first call of a prompt: the first call after a request started (the server says so: a prompt shorter than the cache's max batch is one too,
    // 37 tokens on restricted Qwen3.6 where that batch is larger), or a call above the max batch after steps (llama-cli and the like)
    bool first = n > mb && g_topic.prev_n <= mb;
    if (n > 0 && g_req_boundary.exchange(false)) {
        first = true;
    }
    g_topic.prev_n = n;
    exptok_call(model, ids, n, first);   // --moe expids: every call
    g_tokdump.req += first;              // --moe tokdump: the answer tokens that follow belong to a new request
    topic_phase(model, ids, n, first);   // --moe dynhot: <think> / </think> among the tokens fed (prompt and answer)
    if (mc == nullptr || !first || g_topic.on == 0) {
        return;
    }
    if (g_topic.on < 0) {   // the first prompt: open the store
        size_t max_rec = 1024;
        const std::string m = moe_opt("dynhot") ? moe_opt("dynhot") : "store";
        if      (m == "full" || m == "1") { g_topic.mode = topic_t::FULL; }
        else if (m == "read")             { g_topic.mode = topic_t::READ; }
        else if (m == "off"  || m == "0") { g_topic.mode = topic_t::OFF; }
        else if (m == "store")            { g_topic.mode = topic_t::STORE; }
        else { LLAMA_LOG_WARN("moe-cache: dynhot=%s: not full, store, read or off; store\n", m.c_str()); }
        g_topic.on = g_topic.mode != topic_t::OFF && !hotmap_dir().empty() && model.tok_embd != nullptr && model.hparams.n_expert > 0 ? 1 : 0;
        if (g_topic.on == 0) {
            g_req_prompt.clear();
            return;
        }
        if (const char * m = moe_opt("dynhot-max"))   { max_rec = (size_t) atoll(m); }   // records kept: 0 = no limit (while testing)
        if (const char * m = moe_opt("dynhot-min"))   { g_topic.min_cos   = (float) atof(m); }
        if (const char * m = moe_opt("dynhot-merge")) { g_topic.merge_cos = (float) atof(m); }   // the 4-bit cosine at which a request joins a record (default 2: never)
        if (const char * m = moe_opt("dynhot-global")) { g_topic.store.use_glob = atoi(m) != 0; }
        if (const char * m = moe_opt("dynhot-same"))   { g_topic.store.same_pool = strcmp(m, "pool") == 0; }   // the same prompt again: replace (a better run's map) | pool (every run's counts)   // 0: the lookup blends the nearest topics only (the A/B against the global map)
        const std::string path = hotmap_path(model, ".js");   // <hot map dir>/<model>.js (one assignment per line; the binary .hotmap before it is deleted)
        g_topic.store.max_records = max_rec;   // a record is the keys and a count per expert of every layer: 49 KB on DeepSeek-V4.1, 49 MB of RAM at the default 1024
        if (!g_topic.store.open(path, (uint32_t) model.hparams.n_embd, (uint32_t) model.hparams.n_layer(), (uint32_t) model.hparams.n_expert)) {
            LLAMA_LOG_WARN("moe-cache: dynhot: %s holds maps of another shape (model or layer count): not used, not written\n", path.c_str());
        }
        static const char * what[] = { "off", "full: the nearest maps pick the cache's experts, every request stored",
            "store: requests stored, the ini's hot experts pick the cache's", "read: the nearest maps pick the cache's experts, nothing stored (frozen)" };
        LLAMA_LOG_WARN("moe-cache: dynhot: %zu records and a global map of %u requests in %s; dynhot=%s (lookup: the 4 nearest%s%s; a request joins a record at "
                "cosine >= %.2f)\n", g_topic.store.size(), g_topic.store.glob_req, path.c_str(), what[g_topic.mode],
                g_topic.min_cos > -1.0f ? tr_fmt(" at cosine >= %.2f", g_topic.min_cos).c_str() : "", g_topic.store.use_glob ? " and the global map" : "", g_topic.merge_cos);
    }
    topic_store_pending(mc);   // the request before this prompt, if its end did not store it
    // the server's prompt (the request's own text when the chat template's parts are known, else the whole prompt; its prompt cache may decode only the
    // tail: the 4 tokens of a repeated prompt were a new topic), else this call's tokens: the hash that finds the same prompt again, and its text
    const std::vector<int32_t> whole = std::move(g_req_prompt);
    g_req_prompt.clear();
    const int32_t * kt = whole.empty() ? ids : whole.data();
    const int32_t   kn = whole.empty() ? n : (int32_t) whole.size();
    uint64_t ph = 1469598103934665603ull;   // FNV-1a over the token ids
    for (int32_t i = 0; i < kn; ++i) { ph = (ph ^ (uint64_t) (uint32_t) kt[i]) * 1099511628211ull; }
    moe_topic_text tx;
    for (int32_t i = 0; i < kn && tx.prompt.size() < 400; ++i) { topic_text_add(tx.prompt, model.vocab.token_to_piece(kt[i])); }
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (auto & ls : mc->layers) {
            std::fill(ls.topic_cnt.begin(), ls.topic_cnt.end(), 0u);
            std::fill(ls.topic_dec.begin(), ls.topic_dec.end(), 0u);
            std::fill(ls.topic_think.begin(), ls.topic_think.end(), 0u);
            std::fill(ls.topic_heat.begin(), ls.topic_heat.end(), 0.0f);
            ls.topic_max = 0;
        }
        g_topic_tmax.store(1);
        g_topic.steer = false;   // the prompt runs on the ini's hot experts: the maps are looked up at its last pass (llama_moe_topic_act_fork)
        g_topic_on.store(false);
        g_topic.h0 = g_topic.m0 = 0;   // the decode hits so far: this request's hit rate is what comes after
        for (const auto & ls : mc->layers) { g_topic.h0 += ls.n_hit; g_topic.m0 += ls.n_miss; }
    }
    g_topic_answer.store(false);
    g_topic.key.clear();
    g_topic.n_rows       = 0;
    g_topic.text         = std::move(tx);
    g_topic.phash        = ph;
    g_topic.act_wait     = true;
    g_topic.act_ok       = false;
    g_topic.after_prompt = false;
    g_topic.n_prompt     = kn;
    g_topic.own          = !whole.empty() && g_req_own;
}

// --moe dynhot=full|read: the maps of the records nearest to the request's key (their thinking maps or their answer maps) into the cache's score, and the
// drop and refill of the next decode step (topic_drop_foreign, reheat_free_blocks) put them in; nothing stored: the ini picks. Returns the records used
static size_t topic_steer(moe_cache * mc, bool thinking, float * best, size_t * rated) {
    std::vector<float> heat;
    if (best) { *best = -1.0f; }
    if (g_topic.store.n_rows != 0 && g_topic.store.n_rows != g_topic.n_rows) { return 0; }   // keys of another row count: nothing to compare with
    const std::vector<float> u = g_topic.store.n_rows ? g_topic.store.unit_rows(g_topic.key.data()) : std::vector<float>();
    const size_t used = u.empty() ? 0 : g_topic.store.merge(u.data(), 4, g_topic.min_cos, heat, best, thinking);
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (auto & ls : mc->layers) {
            const bool in = used > 0 && ls.pub.il >= 0 && (uint32_t) ls.pub.il < g_topic.store.n_layers;
            for (size_t e = 0; e < ls.topic_heat.size(); ++e) {
                ls.topic_heat[e] = in && e < g_topic.store.n_exp ? heat[(size_t) ls.pub.il * g_topic.store.n_exp + e] : 0.0f;
                n += ls.topic_heat[e] > 0.0f;
            }
        }
        g_topic.steer = used > 0 || g_topic.store.last_wg > 0.0f;   // nothing stored: the ini picks for this request
        g_topic_on.store(g_topic.steer);
    }
    g_topic.after_prompt = g_topic.steer;
    if (rated) { *rated = n; }
    return used;
}

// --moe dynhot: the thinking phase (user 2026-10-28: "store 2 maps per emb ... if think tokens detected"; "load nonthinking hot map on thinking end").
// Every token fed to the model passes here, the prompt's and the answer's: <think> starts the phase, </think> ends it. A </think> in the answer (after
// the prompt's last pass) is the switch: the answer's uses count in the answer map from then on, and with full or read the nearest records' answer maps
// replace their thinking maps in the cache. A model without the tags never thinks here: one map
static void topic_phase(const llama_model & model, const int32_t * ids, int32_t n, bool first) {
    if (g_topic.tok_open == -2) {
        g_topic.tok_open  = model.vocab.text_to_token("<think>");
        g_topic.tok_close = model.vocab.text_to_token("</think>");
    }
    if (first) {
        g_topic_thinking.store(false);
        g_topic.switched = false;
    }
    const bool tags   = g_topic.tok_open != LLAMA_TOKEN_NULL && g_topic.tok_close != LLAMA_TOKEN_NULL;
    const bool answer = g_topic_answer.load() && g_topic.on == 1;   // the prompt is over: these are the answer's tokens, its words go to the record
    bool th = g_topic_thinking.load(), closed = false;
    for (int32_t i = 0; i < n; ++i) {
        if (tags && ids[i] == g_topic.tok_open)  { th = true; continue; }
        if (tags && ids[i] == g_topic.tok_close) { closed = closed || th; th = false; continue; }
        if (answer && ids[i] >= 0) { topic_text_add(th ? g_topic.text.thinking : g_topic.text.answer, model.vocab.token_to_piece(ids[i])); }
    }
    g_topic_thinking.store(th);
    moe_cache * mc = g_cache;
    if (!closed || mc == nullptr || g_topic.on != 1 || g_topic.act_wait || g_topic.key.empty() || g_topic.switched) {
        return;   // no switch, or a </think> of the prompt itself (thinking turned off in the template)
    }
    g_topic.switched = true;
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        g_topic.h_sw = g_topic.m_sw = 0;
        for (const auto & ls : mc->layers) { g_topic.h_sw += ls.n_hit; g_topic.m_sw += ls.n_miss; }
    }
    if (g_topic.use()) {
        float  best  = -1.0f;
        size_t rated = 0;
        const size_t used = topic_steer(mc, false, &best, &rated);
        LLAMA_LOG_WARN("moe-cache: dynhot: </think>: the answer maps of %zu records (nearest at cosine %.3f) replace their thinking maps, they rate %zu experts\n",
                used, best, rated);
    }
}

// asked by the server's context before each pass: does the pass that ends this prompt copy its last token's key rows out (llama-context.cpp, dynkey_add)
bool llama_moe_topic_key_wanted_fork(void) {
    return g_cache != nullptr && g_topic.on == 1 && g_topic.act_wait;
}

// the prompt's last pass is over: its last token's router input of every MoE layer in layer order and then its final hidden state (rows, n_rows of
// n_embd floats; nullptr: not in the graph) become the request's layer keys, each at unit length and raw (which normalization tells topics apart is
// decided from the collected records): the first and the middle MoE layer's, the final state, and all MoE layers' pooled (each at unit length first, so
// no layer outweighs the others). Then the lookup: with full or read the nearest maps pick the cache's experts for the answer (topic_drop_foreign at its
// first decode step and the refill after it), with store only the log says how near the nearest record is. With --moe tokdump every row also goes to
// <tokdump>.acts (per request: its prompt hash, the row count, the rows in fp16), so other poolings can be tried on them later
void llama_moe_topic_act_fork(const float * const * rows, int n_rows, int64_t n_embd) {
    moe_cache * mc = g_cache;
    if (mc == nullptr || g_topic.on != 1 || !g_topic.act_wait || n_embd != (int64_t) g_topic.store.n_embd || n_rows < 1) {
        return;
    }
    g_topic.act_wait = false;
    {   // the answer starts here: its hit rate and its maps count from now (a prompt shorter than the cache's max batch ran as decode steps)
        std::lock_guard<std::mutex> lk(mc->mtx);
        g_topic.h0 = g_topic.m0 = 0;
        for (const auto & ls : mc->layers) { g_topic.h0 += ls.n_hit; g_topic.m0 += ls.n_miss; }
        for (auto & ls : mc->layers) {
            std::fill(ls.topic_dec.begin(), ls.topic_dec.end(), 0u);
            std::fill(ls.topic_think.begin(), ls.topic_think.end(), 0u);
        }
    }
    g_topic_answer.store(true);
    const int64_t t0 = ggml_time_us();
    // the key: every row as the model gave it (each compared only with the same row of a stored key, the cosines averaged; moe_topic_store); a row the
    // graph did not have makes the request one this store cannot keep
    g_topic.key.assign((size_t) n_rows * n_embd, 0.0f);
    g_topic.n_rows = (uint32_t) n_rows;
    int n_ok = 0;
    for (int r = 0; r < n_rows; ++r) {
        if (rows[r] == nullptr) { continue; }
        std::copy(rows[r], rows[r] + n_embd, g_topic.key.begin() + (ptrdiff_t) r * n_embd);
        n_ok++;
    }
    g_topic.act_ok = n_ok == n_rows && n_rows > 1;
    if (const char * dp = moe_opt("tokdump"); dp && *dp) {   // --moe tokdump: every row of the prompt's last token beside the token dump
        static FILE * fa = nullptr;
        if (fa == nullptr) {
            fa = fopen((std::string(dp) + ".acts").c_str(), "ab");
            if (fa != nullptr) {
                fseek(fa, 0, SEEK_END);
                if (ftell(fa) == 0) {
                    const uint32_t h = (uint32_t) n_embd;
                    fwrite("MOEACTS1", 1, 8, fa);
                    fwrite(&h, sizeof(h), 1, fa);
                }
            }
        }
        if (fa != nullptr) {
            const uint32_t nr = (uint32_t) n_rows;
            fwrite(&g_topic.phash, sizeof(g_topic.phash), 1, fa);
            fwrite(&nr, sizeof(nr), 1, fa);
            std::vector<ggml_fp16_t> h(n_embd);
            for (int i = 0; i < n_rows; ++i) {
                if (rows[i] != nullptr) { ggml_fp32_to_fp16_row(rows[i], h.data(), n_embd); } else { std::fill(h.begin(), h.end(), ggml_fp32_to_fp16(0.0f)); }
                fwrite(h.data(), sizeof(ggml_fp16_t), n_embd, fa);
            }
            fflush(fa);
        }
    }
    float  best = -1.0f;
    size_t used = 0, rated = 0;
    const bool thinking = g_topic_thinking.load();   // the answer starts with thinking: the records' thinking maps first, their answer maps at </think>
    if (g_topic.use()) {
        used = topic_steer(mc, thinking, &best, &rated);
    } else if (g_topic.act_ok && g_topic.store.n_rows == g_topic.n_rows) {   // store: only how near the nearest record is (the log)
        const std::vector<float> u = g_topic.store.unit_rows(g_topic.key.data());
        const auto nb = g_topic.store.nearest(u.data(), 1);
        best = nb.empty() ? -1.0f : nb[0].first;
    }
    LLAMA_LOG_WARN("moe-cache: dynhot: prompt of %d tokens (%s), key: %d of %d rows at its last token (MoE layers and the final state), nearest of %zu stored "
            "requests at similarity %.3f%s: %s (%.1f ms)\n", g_topic.n_prompt, g_topic.own ? "the request's own text" : "the whole prompt", n_ok, n_rows,
            g_topic.store.size(), best, thinking ? ", it starts thinking" : "",
            !g_topic.use() ? "store only, the ini's hot experts pick the cache's" : g_topic.steer ? tr_fmt("%zu maps merged and the global map at weight %.2f, "
                "they rate %zu experts", used, g_topic.store.last_wg, rated).c_str() : "nothing stored: the ini's hot experts pick the cache's",
            (ggml_time_us() - t0) / 1e3);
}

// ---- block pool layout -------------------------------------------------------------------------------------------------------------------------
// The expert cache of a device group is ONE buffer of blocks. A block holds one expert: its up, gate and down slices one after the other. Layers whose
// experts have the same types and sizes form a size class; a class is a run of equal blocks, and any block of it can hold an expert of any layer of
// the class (the owner table, moe_cache::pool, says which). Nothing is reserved per layer: the class gets (layers x the slot count) blocks and the
// layers take them as their experts earn them. The cache tensors of a layer are strided views of its class: data = class start + the slice's offset
// in a block, nb[2] = block bytes, ne[2] = blocks of the class. The classes lie back to back: the device tables mark uncached experts -1 and the kernels skip them,
// so no class needs a zero block behind it.
// The MMVQ / MMQ kernels take the expert stride from nb[2] as nb[2] / type size, so a block is padded to a multiple of the quant block sizes of its
// class (test-backend-ops: MUL_MAT_ID with strided experts). The first blocks of the buffer are one contiguous byte range a compute buffer can take.
struct pool_class {
    size_t B      = 0;               // bytes of one block
    size_t off[3] = { 0, 0, 0 };     // up / gate / down: byte offset inside a block
    size_t len[3] = { 0, 0, 0 };     // up / gate / down: bytes of the slice
    size_t past[3] = { 0, 0, 0 };    // up / gate / down: bytes read past the slice's last row
    size_t base   = 0;               // byte offset of the class in the buffer
    std::vector<size_t> members;     // indices into the group's layer list
};
struct pool_layout_t {
    std::vector<pool_class> cls;
    std::vector<size_t>     lay_cls;      // layer -> class
    size_t round_bytes = 0;               // the bytes of one block per layer (what one more slot per layer costs)
};


static pool_layout_t pool_layout(const std::vector<const llama_layer *> & ls) {
    pool_layout_t out;
    out.lay_cls.resize(ls.size());
    std::map<std::vector<size_t>, size_t> key2cls;
    for (size_t i = 0; i < ls.size(); ++i) {
        const ggml_tensor * t[3] = { ls[i]->ffn_up_exps, ls[i]->ffn_gate_exps, ls[i]->ffn_down_exps };
        const std::vector<size_t> key = { (size_t) t[0]->type, (size_t) t[1]->type, (size_t) t[2]->type, t[0]->nb[2], t[1]->nb[2], t[2]->nb[2] };
        auto it = key2cls.find(key);
        if (it == key2cls.end()) {
            it = key2cls.emplace(key, out.cls.size()).first;
            pool_class c;
            poolx_slice sl[3];
            for (int k = 0; k < 3; ++k) { sl[k] = { t[k]->nb[2], ggml_type_size(t[k]->type), (size_t) ggml_blck_size(t[k]->type), (size_t) t[k]->ne[0] }; }
            const poolx_block pb = poolx_block_class(sl);   // sizing rules and their reasons (Qwen3.8 down rows of 640): llama-poolalloc-fork.h
            c.B = pb.B;
            for (int k = 0; k < 3; ++k) { c.off[k] = pb.off[k]; c.len[k] = pb.len[k]; c.past[k] = pb.past[k]; }
            out.cls.push_back(c);
        }
        out.lay_cls[i] = it->second;
        out.cls[it->second].members.push_back(i);
        out.round_bytes += out.cls[it->second].B;
    }
    return out;
}

// ---- VRAM outside the expert pool -----------------------------------------------------------------------------------------------------------------------
// The pool takes a card's free VRAM minus a margin for what CUDA allocates after it is laid out: kernels loaded on first use, cuBLAS handles, the decode
// graphs' scratch, prompt scratch that does not fit the lent region (the device pool keeps its peak), the pool allocation's own page tables. Until
// 2026-10-22 the margin was a fixed 384 MiB everywhere (the 0.1 MiB-per-micro-batch-token rule in common/ set margin_mb, which no engine setting read),
// and the pool grew in whole rounds (one block per layer of the card, GLM 3.0: 182 MiB), so 495-661 MiB per 24 GB card stayed free at the peak.
// Now every session measures what it really took outside the pool: free VRAM when the pool was sized - the pool - the lowest free VRAM seen. The
// figure is stored per card, driver and build (an input: a new driver or build measures again) and the next start leaves that plus a safety.
// The watch also logs its first lows with the call that made them: when CUDA's own use stops growing.
struct vram_dev_t {
    ggml_backend_dev_t dev = nullptr;
    size_t  free_at_size = 0;      // free VRAM when the pool was sized
    size_t  pool         = 0;      // bytes the pool took
    size_t  low          = SIZE_MAX;
    int64_t stored_mb    = -1;     // the need stored for this card, driver and build (-1: none yet)
    bool    from_cold    = false;  // a warm start sized with the cold figure: its own figure is written at its first decode sample
    int64_t peak_mb      = 0;      // the largest need this session
    int     n_lows       = 0;
    std::string section, key;
};
static std::vector<vram_dev_t> g_vram;
static std::mutex g_vram_mtx;

// the NVIDIA driver: the kernel module's version on Linux, else the CUDA driver API version the backend reports
static const std::string & vram_driver() {
    static const std::string d = [] {
        std::string v;
#ifndef _WIN32
        if (FILE * f = fopen("/proc/driver/nvidia/version", "r")) {   // "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.104.02  Release Build ..."
            char line[512] = {};
            if (fgets(line, sizeof line, f)) {   // the first word made of digits and dots
                for (const char * p = line; *p && v.empty(); ) {
                    while (*p == ' ') { ++p; }
                    const char * w = p;
                    bool num = isdigit((unsigned char) *p) != 0, dot = false;
                    while (*p && *p != ' ' && *p != '\n') { num = num && (isdigit((unsigned char) *p) || *p == '.'); dot = dot || *p == '.'; ++p; }
                    if (num && dot) { v.assign(w, p); }
                }
            }
            fclose(f);
        }
#endif
        if (v.empty()) {
            if (auto fn = (int (*)(void)) cuda_obs_proc("ggml_backend_cuda_driver_version_fork")) { v = "api" + std::to_string(fn()); }
        }
        return v.empty() ? std::string("unknown") : v;
    }();
    return d;
}

// The live llama_contexts and their models (llama_init_from_model adds, llama_free removes). What the other contexts of the process (a DFlash or
// DSpark draft, an MTP head) take on a card after the pool is sized is part of the need stored below, so the need is stored apart for each set of
// them: DeepSeek-V4.1 + DSpark depth 3 (2026-10-27): plain runs stored 38 MiB on cuda0x4; the draft's compute buffer grows by 256 MiB after the pool
// is sized (its first decode after the speculative setup turned on its extra outputs) and did not fit in the 104 MiB left: cudaMalloc failed at load,
// at every start, as a draft run's target calls are all verify batches of several tokens and it never stored a figure of its own.
static std::mutex g_live_mtx;
static std::map<const void *, const llama_model *> g_live_ctx;
static uint64_t g_vram_others = 0;   // bytes of the other contexts' models when the pool was sized (0: the cache's context is the only one)

void llama_moe_cache_context_live_fork(const void * ctx, const llama_model * model, bool live) {
    std::lock_guard<std::mutex> lk(g_live_mtx);
    if (live) { g_live_ctx[ctx] = model; } else { g_live_ctx.erase(ctx); }
}

// cold: this start runs the max-batch probe (no stored timings), whose calibration graphs take ~100 MiB more outside the pool than a warm start
// (GLM 3.0 140 against 49 MiB, MiMo 111 against 36, Qwen3.8 166 against 52, 2026-10-22): the two are stored apart
// The card's PCI address is part of the key: two cards of one model (PC1's two RTX 3090) had the same key, so the second card's figure overwrote the
// first's (35 over 290 MiB, 2026-10-28) and the next start left cuda0x4 a 99 MiB margin where the DSpark draft's setup needs 260 MiB: cudaMalloc failed at load.
static std::string vram_key(ggml_backend_dev_t dev, bool cold) {
    std::string gpu = std::string(ggml_backend_dev_description(dev)) + "." + dev_pci_id(dev, 0);
    for (char & c : gpu) { if (!isalnum((unsigned char) c)) { c = '_'; } }
    const char * b = moe_opt("build-id");
    const std::string with = g_vram_others ? ".with" + std::to_string(g_vram_others >> 20) : "";   // a plain run keeps the key it always had
    return hw_key("vram_need." + gpu + ".drv" + vram_driver() + ".b" + (b ? b : "0") + with + (cold ? ".cold" : ".warm"));
}

// the margin a pool leaves on a card: --moe vram-reserve-mb / margin_mb if set, else the need an earlier session measured on this card, driver and build
// plus a safety (64 MiB or an eighth of the need), else 384 MiB. Windows never goes below 384: the display and other programs take VRAM at any time (WDDM).
static size_t vram_margin(ggml_backend_dev_t dev, const std::string & section, bool cold, int64_t * stored_mb, bool * from_cold) {
    const size_t MiB = 1024 * 1024;
    if (user_knobs().count("VRAM_RESERVE_MB") || user_knobs().count("MARGIN_MB")) {
        return (size_t) std::max(0.0, knobs().margin_mb) * MiB;
    }
    size_t m = 384 * MiB;
    std::string v;
    // a warm start with no warm figure yet takes the cold one (it is larger)
    bool own = !section.empty() && moe_state_get(section, vram_key(dev, cold), v);
    if (!own && !cold && !section.empty() && moe_state_get(section, vram_key(dev, true), v) && from_cold) { *from_cold = true; }
    if ((own || (from_cold && *from_cold)) && atoll(v.c_str()) > 0) {
        const int64_t need = atoll(v.c_str());
        if (stored_mb) { *stored_mb = need; }
        m = (size_t) (need + std::max<int64_t>(64, need / 8)) * MiB;
    }
#ifdef _WIN32
    m = std::max(m, 384 * MiB);
#endif
    return m;
}

// called by the decode path after every call (llama-context.cpp): prompt calls always, decode tokens every 32nd
void llama_moe_cache_vram_watch_fork(bool prompt) {
    static std::atomic<uint32_t> n_calls{0};
    const uint32_t k = ++n_calls;
    if (!prompt && (k & 31) != 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_vram_mtx);
    for (auto & d : g_vram) {
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(d.dev, &f, &t);
        if (f < d.low) {
            d.low = f;
            const int64_t outside = (int64_t) d.free_at_size - (int64_t) d.pool - (int64_t) f;
            d.peak_mb = std::max<int64_t>(d.peak_mb, (outside + (1 << 20) - 1) >> 20);
            if (d.n_lows++ < 24) {
                LLAMA_LOG_WARN("moe-cache: VRAM watch %s: call %u (%s): free %.0f MiB, outside the pool since it was sized %lld MiB\n", ggml_backend_dev_name(d.dev), k,
                        prompt ? "prompt" : "decode", f / 1048576.0, (long long) d.peak_mb);
            }
        }
        // stored once a decode token was sampled (a whole request seen), then at every new peak 16 MiB over the stored figure. A warm start sized with the
        // cold figure (no warm one yet) writes its own, smaller figure at its first decode sample.
        const bool first = d.stored_mb < 0 || d.from_cold;
        if (!d.section.empty() && ((first && !prompt) || (!first && d.peak_mb > d.stored_mb + 16))) {
            moe_state_set(d.section, { { d.key, std::to_string(d.peak_mb) } });
            LLAMA_LOG_WARN("moe-cache: VRAM need outside the pool on %s: %lld MiB, stored for the next start (was %lld)\n", ggml_backend_dev_name(d.dev),
                    (long long) d.peak_mb, (long long) d.stored_mb);
            d.stored_mb = d.peak_mb;
            d.from_cold = false;
        }
    }
}

// fork: a CUDA out-of-memory the backend caught (llama_context::decode after ggml_cuda_oom): the watch above never sees a request that failed, so the need
// stored for the next start stayed what the passes that fitted took (MiMo: 40 MiB learned, then a pass wanted more with 82 MiB free and the process died).
// The failed request counts at once: what is in use outside the pool now plus what was asked for, or, when the size is not known (a CUDA graph's
// instantiation), plus all that was still free (it was not enough). Stored right away: the retry may fail again and the next start must leave the room.
void llama_moe_cache_vram_oom_fork(int cuda_device, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_vram_mtx);
    const std::string name = "CUDA" + std::to_string(cuda_device);
    for (auto & d : g_vram) {
        if (name != ggml_backend_dev_name(d.dev)) { continue; }
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(d.dev, &f, &t);
        const int64_t outside = std::max<int64_t>(0, (int64_t) d.free_at_size - (int64_t) d.pool - (int64_t) f);
        const int64_t need_mb = (outside + (int64_t) (bytes ? bytes : f) + (2 << 20) - 1) >> 20;   // rounded up to whole MiB
        d.peak_mb = std::max<int64_t>(d.peak_mb, need_mb);
        LLAMA_LOG_WARN("moe-cache: VRAM need outside the pool on %s after a CUDA out-of-memory: %lld MiB (%lld in use outside the pool, %s)%s\n",
                ggml_backend_dev_name(d.dev), (long long) d.peak_mb, (long long) (outside >> 20),
                bytes ? ("+" + std::to_string(bytes >> 20) + " MiB asked for").c_str() : ("+" + std::to_string(f >> 20) + " MiB that was free and not enough").c_str(),
                d.section.empty() ? "" : ", stored for the next start");
        if (!d.section.empty() && d.peak_mb > d.stored_mb) {
            moe_state_set(d.section, { { d.key, std::to_string(d.peak_mb) } });
            d.stored_mb = d.peak_mb;
            d.from_cold = false;
        }
    }
}

void llama_moe_cache_init_fork(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t prefetch_slots, int32_t window,
                          int32_t predict, int32_t predict_train) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        LLAMA_LOG_WARN("moe-cache: another context already has the expert cache (it is one per process): this context runs without it\n");
        return;
    }
    [&]() {
        if (n_slots == 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        mc->n_expert_used = (int64_t) model.hparams.n_expert_used();
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        mc->window = std::max(1, window);

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer) {
                continue;
            }
            {
                // host or CPU-repacked (non-host but CPU device) experts are cacheable
                ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_up_exps->buffer));
                const bool on_cpu = ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) ||
                                    (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!on_cpu) {
                    continue; // experts already on a device: nothing to cache
                }
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        // LLAMA_MOE_CACHE_JIT_POOL=n: JIT pool slots per slower-link layer on the fastest-link GPU. Default 0: the pool
        // chain's cross-GPU input/output copies cost ~10% on GLM (2x3090, 20 x4 layers) even with nothing in the pool
        // --moe pool=0: own cache tensors per layer with a fixed slot count each, as before the block pool (reference for A/B)
        const bool pool_on = [] { const char * e = moe_opt("pool"); return !e || atoi(e) != 0; }() && predict <= 0;   // the predictor's stream slots are per layer
        const int32_t jit_pool = [] { const char * e = moe_opt("jit-pool"); return e ? std::max(0, atoi(e)) : 0; }();
        // LLAMA_MOE_CACHE_JIT_POOL_ALL=1: every layer gets a pool (JIT never evicts the main cache)
        const bool jit_pool_all = [] { const char * e = moe_opt("jit-pool-all"); return e && atoi(e) != 0; }();
        // n_slots < 0: fill each device's free VRAM (called after KV/compute buffers
        // exist), leaving a margin for the compute graph growing by the cache chain.
        // ponytail: one slot count per device, uniform over that device's layers.
        ggml_backend_dev_t sz_dev = nullptr;   // the last sizing (slots_for): the block pool fills the rest of its last round from it
        size_t sz_free = 0, sz_margin = 0;
        int64_t sz_stored = -1;
        bool sz_from_cold = false;
        const std::string vram_section = profile_path(model);
        std::string probe_v;
        const bool vram_cold = vram_section.empty() || !moe_state_get(vram_section, hw_key("batch.cost2"), probe_v);   // the max-batch probe runs this start
        {   // the other contexts now (the server creates the draft before the target's first decode, which sizes the pool): their models key the stored need
            std::lock_guard<std::mutex> lk(g_live_mtx);
            bool own = false;
            g_vram_others = 0;
            for (const auto & lc : g_live_ctx) {
                if (lc.second == &model && !own) { own = true; continue; }   // the cache's own context (an MTP head runs a second one on the same model)
                g_vram_others += llama_model_size(lc.second);
            }
            if (g_vram_others) {
                LLAMA_LOG_WARN("moe-cache: other contexts on the process hold models of %.0f MiB (a draft or MTP head): the VRAM need outside the pool is "
                        "stored apart for them\n", g_vram_others / 1048576.0);
            }
        }
        auto slots_for = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands) -> int32_t {
            if (n_slots > 0) {
                return n_slots;
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (!dev) {
                return 0;
            }
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            // what CUDA takes after the pool (vram_margin: set by the user, else measured by an earlier session, else 384 MiB)
            int64_t stored_mb = -1;
            bool from_cold = false;
            size_t margin = vram_margin(dev, vram_section, vram_cold, &stored_mb, &from_cold);
            // --prefetch-experts-slots: the scheduler lazily allocates N full expert
            // tensors on the device big batches are offloaded to
            if (prefetch_slots >= 2 && !alias_on()) {   // with the compute alias the staging slots are lent blocks: no permanent reserve
                if (model.dev_offload < model.devices.size() && dev == model.devices[model.dev_offload].dev) {
                    size_t max_tensor = 0;
                    for (const auto & c : all) {
                        max_tensor = std::max({max_tensor, ggml_nbytes(c.l->ffn_up_exps), ggml_nbytes(c.l->ffn_gate_exps), ggml_nbytes(c.l->ffn_down_exps)});
                    }
                    margin += (size_t) std::min(prefetch_slots, 4) * max_tensor;
                }
            }
            // JIT pools of the slower-link GPUs' layers live on the fastest-link (offload) GPU
            if (jit_pool > 0 && model.dev_offload < model.devices.size() && dev == model.devices[model.dev_offload].dev) {
                for (const auto & g2 : groups) {
                    if (ggml_backend_buft_get_device(g2.first) == dev && !jit_pool_all) {
                        continue;
                    }
                    for (const auto & c : g2.second) {
                        margin += (size_t) (jit_pool + 1) * (c.l->ffn_up_exps->nb[2] + c.l->ffn_gate_exps->nb[2] + c.l->ffn_down_exps->nb[2]);
                    }
                }
            }
            size_t per_slot = 0;
            int64_t n_expert = 0;
            for (const auto & c : cands) {
                per_slot += c.l->ffn_up_exps->nb[2] + c.l->ffn_gate_exps->nb[2] + c.l->ffn_down_exps->nb[2];
                n_expert = c.l->ffn_up_exps->ne[2];
            }
            if (pool_on) {   // the rounds are padded (alignment, quant block sizes): count what the buffer really takes
                std::vector<const llama_layer *> ll;
                for (const auto & c : cands) { ll.push_back(c.l); }
                per_slot = pool_layout(ll).round_bytes;
            }
            // the learned predictors (fp16 [n_embd, n_expert * ahead] per layer, allocated after the cache) need room too
            {
                const char * a = moe_opt("predict-ahead");
                const int64_t ahead = std::max(0, std::min(8, a ? atoi(a) : 2));
                if (predict > 0 && ahead > 0 && !cands.empty()) {
                    const int64_t n_embd = cands[0].l->ffn_up_exps->ne[0];
                    const char * st = moe_opt("predict-stride");
                    const size_t n_src = (cands.size() + std::max(1, st ? atoi(st) : 1) - 1) / std::max(1, st ? atoi(st) : 1);
                    // fp16 master + Q8_0 prediction copy (34 bytes per 32 values)
                    margin += (size_t) (n_src * n_embd * n_expert * ahead * (sizeof(ggml_fp16_t) + 34.0/32)) + 64u * 1024 * 1024;
                }
            }
            sz_dev = dev; sz_free = free; sz_margin = margin; sz_stored = stored_mb; sz_from_cold = from_cold;
            if (free <= margin + per_slot) {
                return 0;
            }
            // one extra (dummy) slot per layer is allocated on top
            const int64_t fit = (int64_t) ((free - margin) / per_slot) - 1;
            return (int32_t) std::max<int64_t>(0, std::min<int64_t>(fit, n_expert - 1));
        };

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            const int32_t slots = tables_only ? 0 : slots_for(buft, cands);
            if (!tables_only && slots <= 0) {
                LLAMA_LOG_WARN("%s: no room for MoE cache slots on %s\n", __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.n_slots = slots;
                    // slower upload link than the fastest GPU (the one prompt processing is sent to); set here,
                    // not when the layer is created: that is the tables-only pass on the CPU buffer type
                    ggml_backend_dev_t dv = ggml_backend_buft_get_device(buft);
                    ls->slow = groups.size() > 1 && model.dev_offload < model.devices.size() && dv && dv != model.devices[model.dev_offload].dev;
                    {
                        int idx = 0;
                        for (size_t di = 0; di < model.devices.size(); ++di) {
                            if (model.devices[di].dev == dv) { idx = (int) di; }
                        }
                        const int off = (int) model.dev_offload;
                        ls->link = std::min(MAX_LINKS - 1, idx == off ? 0 : (idx < off ? idx + 1 : idx));
                        mc->n_links = std::max(mc->n_links, ls->link + 1);
                        mc->link_dev[ls->link] = dv;
                    }
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            // block pool: the up / gate / down cache tensors of all layers are strided views of one buffer (see pool_layout); the remaining
            // tensors of the context (device tables) are allocated below as before. The tensor fields are set by hand: init_tensor would clear
            // the quantization padding behind a tensor, which here is the next layer's data.
            if (pool_on && !tables_only) {
                std::vector<const llama_layer *> ll;
                for (const auto & c : cands) { ll.push_back(c.l); }
                pool_layout_t pl = pool_layout(ll);
                std::vector<int32_t> nblk(pl.cls.size());
                size_t total = 0;
                // The classes lie back to back: no zero block after a class. The device tables mark an uncached expert -1 and every kernel skips negative ids,
                // so nothing reads a slot behind a class's last block; the zero block that sat there (the "uncached" target before negative ids) was the one
                // thing in the pool nothing could move, and the KV zone had to fit between two of them. MiMo-V2.6-Flash has 9 size classes on the x4 card (the
                // largest 5.3 GiB): its 3 GiB sliding-window part fitted no class behind an 11 GiB lend with 9.5 GiB of the pool behind it ("do not fit in any
                // gap", the pass fell from 27623 to 2609 tokens, 2026-10-24). A read past a class's blocks now reads the next class's experts (poison: 5), not
                // zeros that hide it.
                // the class with the most layers first: its first blocks are the ones the compute buffer can borrow (alias)
                std::vector<size_t> order(pl.cls.size());
                for (size_t c = 0; c < order.size(); ++c) { order[c] = c; }
                std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return pl.cls[x].members.size() > pl.cls[y].members.size(); });
                for (size_t c : order) { nblk[c] = slots * (int32_t) pl.cls[c].members.size(); }   // the class's share of the memory, owned by no layer
                // the slot count is whole rounds (a block per layer of the card, GLM 3.0: 182 MiB): what the last round leaves under the margin becomes
                // blocks of the class with the most layers instead of staying unused
                int32_t extra = 0;
                if (n_slots < 0 && !order.empty() && sz_dev == ggml_backend_buft_get_device(buft)) {
                    size_t want = 4096;
                    for (size_t c : order) { want += (size_t) nblk[c] * pl.cls[c].B; }
                    if (sz_free > sz_margin + want) { extra = (int32_t) ((sz_free - sz_margin - want) / pl.cls[order[0]].B); }
                    nblk[order[0]] += extra;
                }
                for (size_t c : order) {
                    pl.cls[c].base = total;
                    total         += (size_t) nblk[c] * pl.cls[c].B;
                }
                total += 4096;   // MMQ reads whole 512-wide tiles: the end of the last tensor may be read past
                ggml_backend_buffer_t pbuf = ggml_backend_buft_alloc_buffer(buft, total);
                if (!pbuf) {
                    LLAMA_LOG_WARN("%s: failed to allocate the MoE cache pool (%.1f MiB) on %s - cache disabled\n", __func__, total / 1048576.0, ggml_backend_buft_name(buft));
                    return false;
                }
                if (sz_dev == ggml_backend_buft_get_device(buft)) {   // the VRAM watch: free at sizing, the pool, and what the allocation itself took
                    size_t f_after = 0, t_after = 0;
                    ggml_backend_dev_memory(sz_dev, &f_after, &t_after);
                    LLAMA_LOG_WARN("moe-cache: VRAM on %s: %.0f MiB free at sizing, margin %.0f MiB (%s, %s start), pool %.0f MiB (+%d blocks from the last round), "
                            "free after it %.0f MiB (the allocation took %.1f MiB beyond the pool)\n", ggml_backend_dev_name(sz_dev), sz_free / 1048576.0,
                            sz_margin / 1048576.0, sz_stored >= 0 ? tr_fmt("stored need %lld MiB + safety", (long long) sz_stored).c_str() : "no stored need",
                            vram_cold ? "cold" : "warm",
                            total / 1048576.0, extra, f_after / 1048576.0, ((double) sz_free - (double) total - (double) f_after) / 1048576.0);
                    std::lock_guard<std::mutex> lk(g_vram_mtx);
                    vram_dev_t d;
                    d.dev = sz_dev; d.free_at_size = sz_free; d.pool = total; d.stored_mb = sz_stored; d.from_cold = sz_from_cold; d.section = vram_section; d.key = vram_key(sz_dev, vram_cold);
                    g_vram.push_back(d);
                }
                ggml_backend_buffer_clear(pbuf, 0);
                if (poison_on()) {   // expert blocks are 5 until an upload lands; the read-past tail stays 0
                    moe_cache::alias_pool tmp; tmp.buf = pbuf;
                    pool_fill(tmp, 0, total - 4096, 5);
                }
                if (poison_on()) { ggml_sched_set_nan_hook_fork(pool_dump_all); }
                ggml_backend_buffer_set_usage(pbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(pbuf);
                const size_t pool0 = mc->pool.size();
                std::string cls_txt;
                for (size_t c = 0; c < pl.cls.size(); ++c) {
                    moe_cache::pool_state ps;
                    ps.dev = ggml_backend_buft_get_device(buft);
                    ps.B   = pl.cls[c].B;
                    for (int k = 0; k < 3; ++k) { ps.off[k] = pl.cls[c].off[k]; ps.len[k] = pl.cls[c].len[k]; ps.past[k] = pl.cls[c].past[k]; }
                    ps.owner.assign(nblk[c], -1);
                    for (int32_t b = nblk[c] - 1; b >= 0; --b) { ps.free.push_back(b); }
                    mc->pool.push_back(std::move(ps));
                    cls_txt += tr_fmt("%s%d x %.2f MiB (%zu layers)", c ? ", " : "", nblk[c], pl.cls[c].B / 1048576.0, pl.cls[c].members.size());
                }
                for (size_t i = 0; i < cands.size(); ++i) {
                    layer_state * ls = nullptr;
                    for (auto & l : mc->layers) {
                        if (l.pub.il == cands[i].il) { ls = &l; break; }
                    }
                    const size_t       ci = pl.lay_cls[i];
                    const pool_class & cl = pl.cls[ci];
                    const int32_t      n  = nblk[ci];
                    ggml_tensor * t3[3] = { ls->pub.up_c, ls->pub.gate_c, ls->pub.down_c };
                    for (int k = 0; k < 3; ++k) {
                        t3[k]->ne[2]  = n;   // the class's blocks: the tables hold 0 .. n - 1 or -1 (uncached, skipped)
                        t3[k]->nb[2]  = cl.B;
                        t3[k]->nb[3]  = cl.B * (size_t) n;
                        t3[k]->buffer = pbuf;
                        t3[k]->data   = base + cl.base + cl.off[k];
                    }
                    ls->pub.n_slots = n;                      // the class's blocks; n marks "uncached" in the host tables (-1 in the device tables)
                    ls->pcls        = (int32_t) (pool0 + ci);
                    ls->n_cache     = slots;                  // nominal share, for the heuristics only
                }
                LLAMA_LOG_WARN("moe-cache: block pool on %s: %zu layers, %.1f MiB, size classes: %s; no blocks reserved per layer\n", ggml_backend_buft_name(buft),
                        cands.size(), total / 1048576.0, cls_txt.c_str());
                {
                    moe_cache::alias_pool ap;
                    ap.buf  = pbuf;
                    ap.dev  = ggml_backend_buft_get_device(buft);
                    ap.pcls = (int32_t) (pool0 + order[0]);
                    ap.B    = pl.cls[order[0]].B;
                    ap.n    = nblk[order[0]];
                    static const bool alias_all = [] { const char * e = moe_opt("alias-all"); return !e || atoi(e) != 0; }();
                    for (size_t c : order) {
                        moe_cache::alias_pool::seg sg;
                        sg.pcls = (int32_t) (pool0 + c);
                        sg.base = pl.cls[c].base;
                        sg.B    = pl.cls[c].B;
                        sg.n    = nblk[c];
                        ap.segs.push_back(sg);
                        ap.span = sg.base + (size_t) sg.n * sg.B;
                        if (!alias_all) { break; }
                    }
                    ap.tail_end  = pl.cls[order.back()].base + (size_t) nblk[order.back()] * pl.cls[order.back()].B;
                    ap.span_full = ap.span;
                    kv_tail_bind(ap);
                    mc->alias.push_back(ap);
                }
                mc->bufs.push_back(pbuf);
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            if (!tables_only) {
                // the scheduler places ops by the residency of weight buffers only;
                // without this the cache chain follows its neighbours onto the CPU
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            mc->bufs.push_back(buf);
            return true;
        };

        for (auto & g : groups) {
            ggml_backend_dev_t d = ggml_backend_buft_get_device(g.first);
            LLAMA_LOG_WARN("moe-cache: %zu expert layers cached on %s (%s), upload link %s\n", g.second.size(), ggml_backend_buft_name(g.first),
                    d ? ggml_backend_dev_name(d) : "?",
                    groups.size() > 1 && model.dev_offload < model.devices.size() && d && d != model.devices[model.dev_offload].dev ? "slower" : "fastest");
        }
        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (ok && jit_pool > 0 && (groups.size() > 1 || jit_pool_all) && model.dev_offload < model.devices.size()) {
            ggml_backend_dev_t fd = model.devices[model.dev_offload].dev;
            ggml_backend_buffer_type_t fb = nullptr;
            for (auto & g : groups) {
                if (ggml_backend_buft_get_device(g.first) == fd) { fb = g.first; }
            }
            std::vector<layer_state *> slow_ls;
            for (auto & l : mc->layers) {
                if ((l.link != 0 || jit_pool_all) && l.pub.up_c) { slow_ls.push_back(&l); }
            }
            if (fb && !slow_ls.empty()) {
                ggml_init_params ip = { ggml_tensor_overhead()*(slow_ls.size()*4 + 8), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                for (auto * l : slow_ls) {
                    const ggml_tensor * u = l->pub.up_src, * g = l->pub.gate_src, * d = l->pub.down_src;
                    l->pub.jit_up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], jit_pool + 1);
                    l->pub.jit_gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], jit_pool + 1);
                    l->pub.jit_down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], jit_pool + 1);
                    l->pub.jit_table  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(l->pub.jit_up_c,   "moe_jit_up.%d",   l->pub.il);
                    ggml_format_name(l->pub.jit_gate_c, "moe_jit_gate.%d", l->pub.il);
                    ggml_format_name(l->pub.jit_down_c, "moe_jit_down.%d", l->pub.il);
                    ggml_format_name(l->pub.jit_table,  "moe_jit_tbl.%d",  l->pub.il);
                }
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, fb);
                if (buf) {
                    ggml_backend_buffer_clear(buf, 0);
                    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    mc->bufs.push_back(buf);
                    mc->ctxs.push_back(ctx);
                    for (auto * l : slow_ls) { l->pub.jit_n = jit_pool; }
                    LLAMA_LOG_INFO("moe-cache: JIT pool %d slots x %zu slower-link layers on %s (%.0f MiB)\n", jit_pool, slow_ls.size(),
                            ggml_backend_dev_name(fd), ggml_backend_buffer_get_size(buf)/1048576.0);
                } else {
                    for (auto * l : slow_ls) { l->pub.jit_up_c = l->pub.jit_gate_c = l->pub.jit_down_c = l->pub.jit_table = nullptr; }
                    ggml_free(ctx);
                }
            }
        }

        // unquantized (F16/BF16/F32) experts: uncached routes share one dummy slot, and the batched mul_mat_id paths
        // for those types (and MMQ/mmf, host sort) break on duplicate ids: only quantized mmvq/mmvf are dup-safe
        if (ok) {
            for (auto & ls : mc->layers) {
                for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
                    if (!ggml_is_quantized(t->type)) {
                        LLAMA_LOG_WARN("%s: expert cache disabled: %s experts are not quantized (duplicate dummy slots are unsafe)\n",
                                __func__, ggml_type_name(t->type));
                        ok = false;
                    }
                    if (!ok) { break; }
                }
                if (!ok) { break; }
            }
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached: n_slots in the host table, -1 in the device table)
        // with router prediction on, the top 4 slots of each layer take the
        // predicted uploads, so a wrong guess never evicts a cached expert
        const int32_t n_stream = predict > 0 ? 4 : 0;
        size_t vram = 0;
        std::vector<ggml_backend_buffer_t> pool_seen;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            const int32_t ns = ls.pub.n_slots;
            ls.n_cache = ls.pcls >= 0 ? ls.n_cache : ns > 2*n_stream ? ns - n_stream : ns;   // pool: the share set at allocation (heuristics only)
            ls.is_stream.assign(ns, 0);
            ls.stream_score.assign(ns, 0.0f);
            ls.stream_step.assign(ns, 0);
            ls.stream_slots.clear();
            for (int32_t sl = ls.n_cache; ls.pcls < 0 && sl < ns; ++sl) {
                ls.is_stream[sl] = 1;
                ls.stream_slots.push_back(sl);
            }
            ls.slot_expert.assign(ns, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ns, 0);
            ls.slot_in_flight.assign(ns, false);
            ls.expert_count.assign(n_expert, 0);
            ls.win_count.assign(n_expert, 0);
            ls.expert_last.assign(n_expert, 0);
            ls.relevance.assign(n_expert, 0.0f);
            ls.ghost_flag.assign(n_expert, 0);
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                if (!file_location(model, srcs[k], &ls.src_fd[k], &ls.src_offs[k])) {
                    ls.src_fd[k] = -1;
                }
            }
            ls.glob_count.assign(n_expert, 0);
            ls.topic_cnt.assign(n_expert, 0);
            ls.topic_dec.assign(n_expert, 0);
            ls.topic_think.assign(n_expert, 0);
            ls.topic_heat.assign(n_expert, 0.0f);
            ls.sticky.assign(n_expert, false);
            ls.adopt_slot.assign(n_expert, -1);
            ls.cached_since.assign(n_expert, 0);
            ls.up_t.assign(n_expert, 0);
            ls.up_cost.assign(n_expert, 0.0f);
            ls.up_hits.assign(n_expert, 0);
            ls.dropped.assign(n_expert, false);
            ls.queued.assign(n_expert, 0);
            ls.prefetched.assign(n_expert, 0);
            ls.adopt_mask.assign(n_expert, 0);

            std::vector<int32_t> dummy(n_expert, ns);
            std::vector<int32_t> neg(n_expert, -1);
            ggml_backend_tensor_set(ls.pub.dev_table, neg.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));
            if (ls.pub.jit_n > 0) {
                ls.pool_expert.assign(ls.pub.jit_n, -1);
                ls.pool_slot.assign(n_expert, -1);
                std::vector<int32_t> none(n_expert, -1);
                ggml_backend_tensor_set(ls.pub.jit_table, none.data(), 0, n_expert*sizeof(int32_t));
            }

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_src[ls.pub.up_src]   = { (size_t) (&ls - mc->layers.data()), 0 };
            mc->by_src[ls.pub.gate_src] = { (size_t) (&ls - mc->layers.data()), 1 };
            mc->by_src[ls.pub.down_src] = { (size_t) (&ls - mc->layers.data()), 2 };
            if (ls.pcls < 0) { vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c); }
            else if (ls.pub.up_c->buffer && std::find(pool_seen.begin(), pool_seen.end(), ls.pub.up_c->buffer) == pool_seen.end()) {   // pool: the layers' tensors are views of one buffer, counted once (the sum of the views said 546 GB for a 4.5 GB pool)
                pool_seen.push_back(ls.pub.up_c->buffer);
                vram += ggml_backend_buffer_get_size(ls.pub.up_c->buffer);
            }
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        {
            size_t max_expert = 0;
            bool   any_file   = false;
            bool   any_repack = false; // host memory isn't the file layout: must upload from the file
            for (auto & ls : mc->layers) {
                max_expert = std::max(max_expert, ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
                any_file   |= ls.src_fd[0] >= 0 && ls.src_fd[1] >= 0 && ls.src_fd[2] >= 0;
                any_repack |= !ggml_backend_buffer_is_host(ls.pub.up_src->buffer);
            }
            ggml_backend_dev_t gpu = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
            // default: copy from the mmap'd host memory (measured faster); pread path is opt-in
            const char * pr = moe_opt("pread");
            const bool use_pread = (pr && pr[0] == '1') || any_repack;
            const char * bn = moe_opt("bounce"), * bk = moe_opt("bounce-kb");
            const size_t bounce_kb = bk ? (size_t) std::max(0, atoi(bk)) : bn ? (size_t) std::max(0, atoi(bn)) * 1024 : 0;   // --moe bounce-kb=KB or bounce=MB; default off (0): 256 KB halves the upload time but cost GLM 3.0 16% (memcpy competes with the CPU experts for RAM bandwidth)
            ggml_backend_buffer_type_t hbuft = (((use_pread && any_file) || bounce_kb > 0) && gpu) ? ggml_backend_dev_host_buffer_type(gpu) : nullptr;
            const char * nt = moe_opt("upload-threads");
            // one set of workers per upload link when layers sit behind both: a slow (x4) copy doesn't hold up the fast link's queue
            const int n_links   = mc->n_links > 1 ? mc->n_links : 1;
            const int n_workers = std::max(1, nt ? atoi(nt) : 1) * n_links;
            for (int w = 0; w < n_workers; ++w) {
                mc->worker_link.push_back(n_links > 1 ? w % n_links : -1);
                ggml_backend_buffer_t b = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, max_expert) : nullptr;
                mc->staging.push_back(b);
            }
            // weights loaded without mmap (--load-mode pin) sit in the GPU's pinned host buffer: direct DMA
            const bool src_pinned = gpu && ggml_backend_buffer_get_type(mc->layers[0].pub.up_src->buffer) == ggml_backend_dev_host_buffer_type(gpu);
            if (bounce_kb > 0 && mc->staging[0] && !use_pread && !src_pinned) {   // pinned weights are DMA'd directly: a bounce would only add a copy
                mc->staging_size  = max_expert;
                mc->bounce_bytes  = std::min<size_t>(bounce_kb << 10, max_expert);
            }
            {   // --moe pinr=N: pinned copies of the uploaded experts' slices, N/10 of the RAM (default 0 = off, postponed); only for weights that are mmap'd
                const char * pv = moe_opt("pinr");
                const double pinr = pv ? atof(pv) : 0.0;
                size_t mem_total = 0;
                if (FILE * mf = fopen("/proc/meminfo", "r")) {
                    char line[256]; size_t kb = 0;
                    while (fgets(line, sizeof line, mf)) { if (sscanf(line, "MemTotal: %zu kB", &kb) == 1) { mem_total = kb * 1024; break; } }
                    fclose(mf);
                }
                if (gpu && !src_pinned && !mc->staging[0] && pinr > 0 && mem_total) {
                    mc->pin_buft   = ggml_backend_dev_host_buffer_type(gpu);
                    mc->pin_budget = (size_t) (std::min(pinr, 10.0) / 10.0 * (double) mem_total);
                    for (auto & ls2 : mc->layers) {
                        const ggml_tensor * ps[3] = { ls2.pub.up_src, ls2.pub.gate_src, ls2.pub.down_src };
                        for (int k2 = 0; k2 < 3; ++k2) {
                            const size_t ne = (size_t) ps[k2]->ne[2];
                            ls2.pin_ptr[k2].reset(new std::atomic<const char *>[ne]);
                            for (size_t e2 = 0; e2 < ne; ++e2) { ls2.pin_ptr[k2][e2].store(nullptr); }
                        }
                    }
                    LLAMA_LOG_WARN("moe-cache: pinr=%.1f: up to %.1f GiB of expert slices pinned as they are uploaded, the rest stays mmap'd\n", pinr, mc->pin_budget / 1073741824.0);
                }
            }
            {   // --moe pin-lazy: the loader mapped the weights without reading them; pin the expert tensors in the background up to --moe pin-budget
                const char * pl = moe_opt("pin-lazy"); const char * pb = moe_opt("pin-budget");
                const size_t budget = pb ? (size_t) strtoull(pb, nullptr, 10) : 0;
                if (gpu && !src_pinned && pl && atoi(pl) && budget > 0) {
                    mc->lazy_buft = ggml_backend_dev_host_buffer_type(gpu);
                    mc->lazy_thr  = std::thread(pin_prefetch_run, mc, budget);
                    LLAMA_LOG_WARN("moe-cache: pin-lazy: pinning expert tensors in the background, up to %.1f GiB, the rest stays mapped\n", budget / 1073741824.0);
                }
            }
            LLAMA_LOG_WARN("moe-cache: %d upload workers%s, %s\n", n_workers, n_links > 1 ? " (per link)" : "",
                    mc->bounce_bytes ? "memcpy -> pinned staging (bounce) -> GPU" :
                    mc->staging[0] ? "pread -> pinned staging -> GPU" :
                    src_pinned     ? "from pinned host memory (direct DMA)" :
                                     "from pageable host memory (mmap; --load-mode pin: faster uploads)");
            {
                static const char * const pol[] = { "halve", "window", "hybrid", "add", "lru", "rel" };
                char sb[320];
                snprintf(sb, sizeof sb, "slots=%d workers=%d links=%d upload=%s bounce_kb=%zu policy=%s", (int) mc->n_slots, n_workers, n_links,
                        mc->bounce_bytes ? "bounce" : mc->staging[0] ? "pread" : src_pinned ? "pinned-dma" : "mmap-pageable",
                        (size_t) (mc->bounce_bytes >> 10), pol[(int) get_policy()]);
                mc->settings = sb;
            }
        }

        for (size_t w = 0; w < mc->staging.size(); ++w)
        mc->workers.emplace_back([mc, w]() {
            unpin_helper();
            uint8_t * staging_ptr = mc->staging[w] ? (uint8_t *) ggml_backend_buffer_get_base(mc->staging[w]) : nullptr;
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    // the first queued job on this worker's link (-1: any)
                    const int link = mc->worker_link[w];
                    auto mine = [mc, link]() {
                        auto bg = mc->todo.end();   // the first background refill, taken only when no other job waits on this link
                        for (auto it = mc->todo.begin(); it != mc->todo.end(); ++it) {
                            if (link >= 0 && mc->layers[it->layer_idx].link != link) {
                                continue;
                            }
                            if (!it->bg) {
                                return it;
                            }
                            if (bg == mc->todo.end()) {
                                bg = it;
                            }
                        }
                        return bg;
                    };
                    mc->wcv.wait(lk, [mc, &mine]() { return mc->stop || mine() != mc->todo.end(); });
                    if (mc->stop) {
                        return;
                    }
                    auto it = mine();
                    j = *it;
                    mc->todo.erase(it);
                    mc->in_flight++;
                    mc->in_flight_stream += j.stream || j.bg;
                }
                auto & ls = mc->layers[j.layer_idx];
                // ponytail: unlocked reads of gpu_started / n_steps; a race only delays or keeps one stale job
                if (j.stream && (j.step != mc->n_steps || mc->gpu_started >= (int64_t) j.layer_idx)) {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.skipped = true;
                    j.done    = true;
                    mc->done.push_back(j);
                    mc->in_flight--;
                    mc->in_flight_stream -= j.stream || j.bg;
                    mc->n_stream_stale++;
                    mc->dcv.notify_all();
                    if (tr_on()) {
                        tr(TR_WORKER + (int) w, tr_fmt("drop L%zu e%d (layer started)", j.layer_idx, j.expert), ggml_time_us());
                    }
                    continue;
                }
                if (ls.pcls >= 0) {
                    size_t blo = 0, bhi = 0;
                    const moe_cache::alias_pool * bap = led_block(mc, ls.pcls, j.slot, blo, bhi);
                    if (bap && !pool_gate(*bap, blo, bhi, "upload")) {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        j.skipped = true; j.done = true;
                        mc->done.push_back(j);
                        mc->in_flight--;
                        mc->in_flight_stream -= j.stream || j.bg;
                        mc->dcv.notify_all();
                        continue;
                    }
                }
                size_t ulo = 0, uhi = 0;
                const moe_cache::alias_pool * uap = ls.pcls >= 0 ? led_block(mc, ls.pcls, j.slot, ulo, uhi) : nullptr;
                if (uap && !pool_try_enter(*uap, ulo, uhi, true)) {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.skipped = true; j.done = true;
                    mc->done.push_back(j);
                    mc->in_flight--;
                    mc->in_flight_stream -= j.stream || j.bg;
                    mc->dcv.notify_all();
                    continue;
                }
                const int64_t t0 = ggml_time_us();
                led_upload(mc, "up-start", j.layer_idx, j.expert, j.slot, j.bg);
                const bool busy_start = mc->cpu_busy;
                ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
                const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
                int64_t t_read = 0;
                bool nosync_used = false;   // some tensor of this job was queued without a sync: wait for all of them once, after the loop
                bool up_failed   = false;   // a tensor of this job could not be uploaded: the job is dropped (skipped), the slot must not be published as holding the expert
                for (int k = 0; k < 3; ++k) {
                    const size_t sz = srcs[k]->nb[2];
                    const int64_t link_mbs = (int64_t) ((mc->gbs_link[ls.link] > 0 ? mc->gbs_link[ls.link] : 10.0) * 1e3);
                    if (knobs().gate >= 3 && mc->ddr_mbs + link_mbs > (int64_t) (knobs().ddr_gbs * 1e3)) {
                        // budget gate: wait until DDR demand leaves room for this copy
                        const int64_t tw = ggml_time_us();
                        while (mc->ddr_mbs + link_mbs > (int64_t) (knobs().ddr_gbs * 1e3) && ggml_time_us() - tw < (int64_t) (knobs().gate_max_us >= 0 ? knobs().gate_max_us : mean_layer_us(mc))) {
                            std::this_thread::yield();
                        }
                        mc->gate_wait_us += ggml_time_us() - tw;
                        if (tr_on()) {
                            tr(TR_WORKER + (int) w, "DDR budget wait", tw, ggml_time_us() - tw);
                        }
                    }
                    mc->ddr_mbs += link_mbs;
                    mc->up_bytes += (int64_t) sz; mc->link_bytes[ls.link] += (int64_t) sz;
                    ddr_trace(mc, ggml_time_us());
                    struct ddr_release { moe_cache * mc; int64_t v; ~ddr_release() { mc->ddr_mbs -= v; ddr_trace(mc, ggml_time_us()); } } rel { mc, link_mbs };
                    if ((knobs().gate == 2 || (knobs().gate == 1 && j.stream)) && mc->cpu_busy) {
                        // ponytail: spin-yield; a condition variable if waits get long
                        const int64_t tw = ggml_time_us();
                        while (mc->cpu_busy && ggml_time_us() - tw < (int64_t) (knobs().gate_max_us >= 0 ? knobs().gate_max_us : mean_layer_us(mc))) {
                            std::this_thread::yield();
                        }
                        mc->gate_wait_us += ggml_time_us() - tw;
                        if (tr_on()) {
                            tr(TR_WORKER + (int) w, "gate wait", tw, ggml_time_us() - tw);
                        }
                    }
                    const int64_t tr = ggml_time_us();
                    if (staging_ptr && mc->bounce_bytes == 0 && ls.src_fd[k] >= 0 &&
                            pread_full(ls.src_fd[k], staging_ptr, sz, ls.src_offs[k] + (size_t) j.expert*sz)) {
                        t_read += ggml_time_us() - tr;
                        ggml_backend_tensor_set(dsts[k], staging_ptr, (size_t) j.slot*dsts[k]->nb[2], sz);
                    } else if (mc->bounce_bytes > 0 && staging_ptr && ggml_backend_buffer_is_host(srcs[k]->buffer)) {
                        // BOUNCE: the mmap'd slice goes through this worker's pinned staging buffer: the DMA then runs from pinned memory
                        const char * bsrc = (const char *) srcs[k]->data + (size_t) j.expert*sz;
                        const size_t bdst = (size_t) j.slot*dsts[k]->nb[2];
                        for (size_t off = 0; off < sz; off += mc->bounce_bytes) {
                            const size_t n = std::min(mc->bounce_bytes, sz - off);
                            memcpy(staging_ptr, bsrc + off, n);
                            ggml_backend_tensor_set(dsts[k], staging_ptr, bdst + off, n);
                        }
                    } else if (ggml_backend_buffer_is_host(srcs[k]->buffer)) {
                        // chunk: set, or the bytes this link moves in a quarter of the measured mean CPU phase (the budget is
                        // re-checked ~4 times per phase)
                        const size_t ch = knobs().gate < 3 || knobs().chunk_kb == 0 ? 0 :
                            knobs().chunk_kb > 0 ? (size_t) knobs().chunk_kb * 1024 :
                            (size_t) std::max(64.0*1024, (mc->gbs_link[ls.link] > 0 ? mc->gbs_link[ls.link] : 1.0) * 1e3 * mean_cpu_phase_us(mc) / 4);
                        const char * pinned = pin_slice(mc, ls, k, j.expert, sz, srcs[k]);   // --moe pinr: the pinned copy of this slice, made now if the budget allows
                        if (ch == 0 || ch >= sz) {
                            // the source (pinned slice copy or the model's own mapping) stays put, so the copy can be queued without waiting for it: the
                            // job syncs once after its three tensors (nosync_used) instead of after each of them
                            const cuda_set_nosync_t ns = ggml_backend_buffer_is_host(dsts[k]->buffer) ? nullptr : cuda_set_nosync();
                            if (ns) {
                                const size_t doff = (size_t) j.slot*dsts[k]->nb[2];
                                ns(dsts[k], pinned ? pinned : (const char *) srcs[k]->data + (size_t) j.expert*sz, doff, sz);
                                nosync_used = true;
                            }
                            else if (pinned) { ggml_backend_tensor_set(dsts[k], pinned, (size_t) j.slot*dsts[k]->nb[2], sz); }
                            else        { upload_slice(dsts[k], srcs[k], j.expert, j.slot); }
                        } else {
                            // re-check the DDR budget per chunk: a CPU phase that started meanwhile makes the rest wait
                            const char * src = pinned ? pinned : (const char *) srcs[k]->data + (size_t) j.expert*sz;
                            const size_t dst = (size_t) j.slot*dsts[k]->nb[2];
                            for (size_t off = 0; off < sz; off += ch) {
                                const int64_t tw = ggml_time_us();
                                while (mc->ddr_mbs > (int64_t) (knobs().ddr_gbs * 1e3) && ggml_time_us() - tw < (int64_t) (knobs().gate_max_us >= 0 ? knobs().gate_max_us : mean_layer_us(mc))) {
                                    std::this_thread::yield();
                                }
                                mc->gate_wait_us += ggml_time_us() - tw;
                                // the chunk source stays put as well (pinned copy or the model's mapping): queue the chunk, the job syncs once at its end.
                                // The budget wait above still paces the chunks, which is all the per-chunk sync was good for here.
                                const cuda_set_nosync_t ns = ggml_backend_buffer_is_host(dsts[k]->buffer) ? nullptr : cuda_set_nosync();
                                if (ns) { ns(dsts[k], src + off, dst + off, std::min(ch, sz - off)); nosync_used = true; }
                                else    { ggml_backend_tensor_set(dsts[k], src + off, dst + off, std::min(ch, sz - off)); }
                            }
                        }
                    } else {
                        LLAMA_LOG_ERROR("moe-cache: can't upload repacked '%s' without its file location\n", srcs[k]->name);
                        up_failed = true;
                    }
                    // what a matmul reads past this slice's last row (rows not a multiple of 512 values, see pool_layout): the block reserves it, and it gets
                    // the bytes that follow the slice in the model's tensor (the next expert), as the tensor itself would give; behind the last expert the
                    // tensor's padding, zeros. Same stream as the slice, before the next slice (up's read region reaches into gate, which is written next).
                    if (ls.pcls >= 0 && mc->pool[ls.pcls].past[k] > 0) {
                        static const std::vector<uint8_t> zeros(1u << 16, 0);
                        const size_t pb   = std::min(mc->pool[ls.pcls].past[k], zeros.size());
                        const bool   next = j.expert + 1 < srcs[k]->ne[2] && ggml_backend_buffer_is_host(srcs[k]->buffer);
                        const char * from = next ? (const char *) srcs[k]->data + (size_t) (j.expert + 1) * sz : (const char *) zeros.data();
                        const size_t doff = (size_t) j.slot * dsts[k]->nb[2] + sz;
                        const cuda_set_nosync_t ns = ggml_backend_buffer_is_host(dsts[k]->buffer) ? nullptr : cuda_set_nosync();
                        if (ns) { ns(dsts[k], from, doff, pb); nosync_used = true; }
                        else    { ggml_backend_tensor_set(dsts[k], from, doff, pb); }
                    }
                }
                if (nosync_used) { cuda_thread_sync(); }
                if (up_failed) { j.skipped = true; }
                if (uap) { pool_leave(*uap, ulo, uhi, true); }
                led_upload(mc, "up-done", j.layer_idx, j.expert, j.slot, j.bg);
                j.t_done  = ggml_time_us();
                if (j.stream) {
                    mc->st_up[ls.link]++;
                    mc->st_late[ls.link] += mc->gpu_started >= (int64_t) j.layer_idx;
                }
                if (tr_on()) {
                    const int64_t gs = mc->gpu_started;
                    tr(TR_WORKER + (int) w, tr_fmt("L%zu e%d %s%s", j.layer_idx, j.expert, j.stream ? "stream" : j.urgent ? "predicted" : "swap",
                                (j.stream || j.urgent) && gs >= (int64_t) j.layer_idx ? " LATE" : ""), t0, j.t_done - t0,
                            tr_fmt("\"queued us\":%lld,\"read us\":%lld,\"link\":\"%s\",\"ahead at finish\":%lld",
                                (long long) (t0 - j.t_queued), (long long) t_read, tr_fmt("%d", ls.link).c_str(), (long long) j.layer_idx - gs));
                }
                const bool overlap = busy_start || mc->cpu_busy;
                const double dt = (double) (j.t_done - t0);
                j.read_us = (float) t_read;
                j.copy_us = (float) (dt - t_read);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    mc->upload_us = mc->n_uploads++ ? 0.9*mc->upload_us + 0.1*dt : dt;
                    mc->up_overlap += overlap;
                    double & lu = mc->link_us[ls.link];
                    const bool first = mc->n_link[ls.link]++ == 0;
                    lu = first ? dt : 0.9*lu + 0.1*dt;
                    double & lr = mc->read_link[ls.link];
                    double & lc = mc->copy_link[ls.link];
                    lr = first ? j.read_us : 0.9*lr + 0.1*j.read_us;
                    lc = first ? j.copy_us : 0.9*lc + 0.1*j.copy_us;
                    const double gbs = (double) (srcs[0]->nb[2] + srcs[1]->nb[2] + srcs[2]->nb[2]) / std::max(1.0, dt) / 1e3;
                    double & lg = mc->gbs_link[ls.link];
                    lg = first ? gbs : 0.9*lg + 0.1*gbs;
                    mc->gbs_all = mc->n_uploads == 1 ? gbs : 0.9*mc->gbs_all + 0.1*gbs;
                    j.done = true;
                    mc->done.push_back(j);
                    mc->in_flight--;
                    mc->in_flight_stream -= j.stream || j.bg;
                }
                mc->dcv.notify_all();
            }
        });

        mc->profile = profile_path(model);
        {
            // decisions of the last self-tune on this machine for this model (the profile file's footer): start from them (an explicit
            // LLAMA_MOE_CACHE_<NAME> wins) and check again after a while instead of exploring right away
            const char * te = moe_opt("tuned");
            std::string txt;
            // ini-fix (2026-10-22): the stored decisions are not read: a start whose tuner began from another build's values behaved unlike a cold one,
            // and the knobs' defaults change between builds. Every start tunes from the defaults.
            (void) te;
            for (char & c : txt) { if (c == ' ') { c = '\n'; } }
            int n = 0;
            size_t pos = 0;
            while (pos < txt.size()) {
                const size_t nl = txt.find('\n', pos);
                const std::string line = txt.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
                pos = nl == std::string::npos ? txt.size() : nl + 1;
                const size_t eq = line.find('=');
                if (eq == std::string::npos) { continue; }
                const std::string name = line.substr(0, eq);
                if (!user_knobs().count(name) && knob_set(knobs(), name, atof(line.c_str() + eq + 1))) { n++; }
            }
            if (n > 0) {
                mc->tuned_text = txt;
                mc->lv_rest = 16384; mc->lv_rest_until = 16384;
                LLAMA_LOG_WARN("moe-cache: self-tune: %d saved settings loaded from the state file, next check after 16384 tokens\n", n);
            }
        }
        if (const char * c = moe_opt("ctl")) {
            mc->ctl = c;
        }
        if (moe_log_has("c")) {   // --moe log=c: decode-step trace, files moe-trace* in the log directory
            static const std::string trace_prefix = moe_log_file("moe-trace");
            g_tr.prefix = trace_prefix.c_str();
            g_tr.names[TR_SCHED] = "sched: tokens, GPU splits, publishes";
            g_tr.names[TR_CPU]   = "CPU experts + router";
            g_tr.names[TR_PRED]  = "predictor";
            g_tr.names[TR_DDR]   = "DDR demand";
            for (size_t w = 0; w < mc->worker_link.size(); ++w) {
                const int l = mc->worker_link[w];
                g_tr.names[TR_WORKER + (int) w] = tr_fmt("upload worker %zu (link %s)", w, l < 0 ? "any" : std::to_string(l).c_str());
            }
            g_tr.skip = (int64_t) knobs().trace_after;
            g_tr.left = (int64_t) knobs().trace;
        }
#ifndef _WIN32
        {
            const char * e = moe_opt("drop");
            const double ram = (double) sysconf(_SC_PHYS_PAGES) * (double) sysconf(_SC_PAGESIZE);
            ggml_backend_dev_t gpu = mc->bufs.empty() ? nullptr : ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
            const bool mmapd = !mc->layers.empty() && ggml_backend_buffer_is_host(mc->layers[0].pub.up_src->buffer) &&
                !(gpu && ggml_backend_buffer_get_type(mc->layers[0].pub.up_src->buffer) == ggml_backend_dev_host_buffer_type(gpu));
            // opt-in: MiMo on 1 GPU measured -9% (a small, churning cache drops and re-reads too often);
            // next: drop only experts that stayed cached long
            mc->drop_cached = e && atoi(e) != 0 && mmapd;
            GGML_UNUSED(ram);
            if (mc->drop_cached) {
                LLAMA_LOG_INFO("moe-cache: model bigger than RAM: VRAM-cached experts' pages are dropped from RAM\n");
            }
        }
#endif
        resource_probe(mc);
        numa_probe(mc);
        if (const size_t n = profile_preload(mc, model)) {
            LLAMA_LOG_WARN("moe-cache: preloading %zu experts (usage profile)\n", n);
        } else {
            LLAMA_LOG_WARN("moe-cache: nothing preloaded: no usage profile of this model and layout in the state file, the cache fills from the first prompt\n");
        }
        {
            size_t n_slow = 0, n_hot = 0, n_stk = 0;
            for (auto & ls : mc->layers) {
                if (ls.slow) {
                    n_slow++;
                    for (bool h : ls.hot) n_hot += h;
                    for (bool k : ls.sticky) n_stk += k;
                }
            }
            if (n_slow > 0) {
                LLAMA_LOG_WARN("moe-cache: %zu layers on slower-link GPUs: always-hot experts %.1f/layer, pinned %.1f/layer, min stay %s steps\n",
                        n_slow, (double) n_hot / n_slow, (double) n_stk / n_slow, std::to_string((int64_t) knobs().slow_stay).c_str());
            }
        }

        {
            const char * x = moe_opt("predict-max");
            mc->pred_m   = predict;
            mc->pred_max = x ? atoi(x) : 2;
            {
                const char * ra = moe_opt("predict-ra");
                ggml_backend_dev_t gpu = mc->bufs.empty() ? nullptr : ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
                const auto * src = mc->layers.empty() ? nullptr : mc->layers[0].pub.up_src;
                mc->pred_ra = ra && atoi(ra) != 0 && src && ggml_backend_buffer_is_host(src->buffer) &&
                    !(gpu && ggml_backend_buffer_get_type(src->buffer) == ggml_backend_dev_host_buffer_type(gpu));
            }
            if (const char * g = moe_opt("predict-margin")) {
                mc->pred_margin = (float) atof(g);
            }
            {
                const char * a  = moe_opt("predict-ahead");
                const char * mu = moe_opt("predict-mu");
                int ahead = 2;
                if (!a && knobs().lead != 0 && !mc->layers.empty() && mc->layers[0].pub.up_src && mc->layers[0].pub.gate_src && mc->layers[0].pub.down_src) {
                    // lookahead long enough for the slowest link: its upload time (probed alone speed, x2 for the contention seen under load) over a 1 ms layer
                    const auto & p0 = mc->layers[0].pub;
                    const double eb = (double) (p0.up_src->nb[2] + p0.gate_src->nb[2] + p0.down_src->nb[2]);
                    int lead_max = 1;
                    for (int k = 0; k < mc->n_links; ++k) {
                        if (mc->gbs_link[k] > 0) { lead_max = std::max(lead_max, (int) std::ceil(2.0 * eb / (mc->gbs_link[k] * 1e9) * 1e3)); }
                    }
                    ahead = std::max(2, std::min(8, lead_max + (int) knobs().lead_span));
                }
                mc->pred_ahead = std::max(0, std::min(8, a ? atoi(a) : ahead));
                mc->pred_train = predict_train;
                if (mu) { mc->pred_mu = (float) atof(mu); }
                const char * pf = moe_opt("predict-file");
                mc->pred_file = pf ? (strcmp(pf, "0") == 0 ? "" : pf) : cache_file(model, "moe-pred-", "-a" + std::to_string(mc->pred_ahead));
            }
            const size_t nl = mc->layers.size();
            mc->pred_b.assign(nl, {});
            mc->last_pred.assign(nl, {});
            mc->last_pred_step.assign(nl, UINT64_MAX);
            mc->last_pred_blk.assign(nl, 0);
            for (size_t li = 0; mc->pred_m > 0 && li < nl; ++li) {
                const int il = mc->layers[li].pub.il;
                const ggml_tensor * b = il < (int) model.layers.size() ? model.layers[il].ffn_exp_probs_b : nullptr;
                if (b && b->buffer && b->type == GGML_TYPE_F32 && ggml_nelements(b) == (int64_t) mc->layers[li].expert_slot.size()) {
                    mc->pred_b[li].resize(ggml_nelements(b));
                    ggml_backend_tensor_get(b, mc->pred_b[li].data(), 0, ggml_nbytes(b));
                }
            }
            size_t n_pred = 0, n_learn = 0;
            // LLAMA_MOE_CACHE_PREDICT_STRIDE=N: predict only from every Nth layer (each covers its next `ahead`
            // layers, so ahead >= N keeps every layer predicted): N times fewer predictor matmuls and weights
            const char * pst = moe_opt("predict-stride");
            const size_t stride = (size_t) std::max(1, pst ? atoi(pst) : 1);
            const char * pq8 = moe_opt("predict-q8");
            const bool use_q8 = !(pq8 && pq8[0] == '0');
            for (size_t li = 0; mc->pred_m > 0 && li + 1 < nl; ++li) {
                if (li % stride != 0) {
                    continue;
                }
                auto & pub = mc->layers[li].pub;
                const int64_t n_embd = pub.up_src->ne[0];
                std::vector<ggml_tensor *> routers;
                for (int k = 0; k < std::max(1, mc->pred_ahead) && li + 1 + k < nl; ++k) {
                    const int il = mc->layers[li + 1 + k].pub.il;
                    if (il != pub.il + 1 + k || il >= (int) model.layers.size()) {
                        break;
                    }
                    ggml_tensor * w = model.layers[il].ffn_gate_inp;
                    if (!w || !w->buffer || w->ne[0] != n_embd || w->ne[1] != (int64_t) mc->layers[li + 1 + k].expert_slot.size()) {
                        break;
                    }
                    routers.push_back(w);
                }
                if (routers.empty()) {
                    continue;
                }
                pub.pred_w = routers[0];
                n_pred++;
                if (mc->pred_ahead == 0) {
                    continue;
                }
                // learned predictors start as the target layers' routers, next to them on their device
                // all lookaheads stacked in one fp16 tensor: one matmul predicts them, one update trains them
                const int64_t ne_x = routers[0]->ne[1];
                while (!routers.empty() && routers.back()->ne[1] != ne_x) {
                    routers.pop_back();
                }
                ggml_init_params ip = { ggml_tensor_overhead()*3, nullptr, true };
                ggml_context * pctx = ggml_init(ip);
                mc->pctxs.push_back(pctx);
                pub.pred_all = ggml_new_tensor_2d(pctx, GGML_TYPE_F16, n_embd, ne_x * (int64_t) routers.size());
                ggml_format_name(pub.pred_all, "moe_pred-%d", pub.il);
                pub.pred_mu = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, 1);
                ggml_set_name(pub.pred_mu, "moe_pred_mu");
                if (use_q8 && n_embd % 32 == 0) { // Q8_0 blocks of 32
                    pub.pred_q = ggml_new_tensor_2d(pctx, GGML_TYPE_Q8_0, n_embd, ne_x * (int64_t) routers.size());
                    ggml_format_name(pub.pred_q, "moe_pred_q8-%d", pub.il);
                }
                ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors_from_buft(pctx, ggml_backend_buffer_get_type(routers[0]->buffer));
                if (!pbuf) {
                    pub.pred_all = nullptr;
                    pub.pred_mu  = nullptr;
                    pub.pred_q   = nullptr;
                    continue;
                }
                mc->pbufs.push_back(pbuf);
                if (n_learn == 0) {
                    LLAMA_LOG_WARN("moe-cache: predictor weights in %s (router %s in %s)\n", ggml_backend_buffer_name(pbuf),
                            routers[0]->name, ggml_backend_buffer_name(routers[0]->buffer));
                }
                std::vector<ggml_fp16_t> h(n_embd * ne_x * routers.size());
                for (size_t k = 0; k < routers.size(); ++k) {
                    const ggml_tensor * w = routers[k];
                    std::vector<uint8_t> raw(ggml_nbytes(w));
                    ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
                    std::vector<float> f(ggml_nelements(w));
                    if (w->type == GGML_TYPE_F32) {
                        memcpy(f.data(), raw.data(), raw.size());
                    } else {
                        ggml_get_type_traits(w->type)->to_float(raw.data(), f.data(), (int64_t) f.size());
                    }
                    ggml_fp32_to_fp16_row(f.data(), h.data() + k * n_embd * ne_x, (int64_t) f.size());   // rows k*E.. = layer +k's router
                }
                ggml_backend_tensor_set(pub.pred_all, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
                const float zero = 0.0f;
                ggml_backend_tensor_set(pub.pred_mu, &zero, 0, sizeof(float));
                pub.pred_ahead = (int) routers.size();
                n_learn++;
            }
            if (n_learn > 0) {
                const bool loaded = pred_load(mc);
                for (auto & ls : mc->layers) {
                    pred_requant(ls.pub);
                }
                LLAMA_LOG_WARN("moe-cache: learned router predictors for %zu layers (every %zu%s), %d layers ahead, %s, training %s\n",
                        n_learn, stride, use_q8 ? ", Q8_0 copy" : ", fp16", mc->pred_ahead, loaded ? ("loaded " + mc->pred_file).c_str() : "new (= the routers)",
                        mc->pred_train > 0 ? ("every " + std::to_string(mc->pred_train) + " tokens").c_str() : "off (--lrn-prd N)");
            }
            if (n_pred > 0) {
                mc->pred_thr = std::thread(pred_loop, mc);
                {
                    // L3 prefetch threads, idle until L3PF > 0: one per L3 domain, pinned to that domain's highest
                    // physical core (LLAMA_MOE_L3PF_CPUS=a,b,.. overrides); rows split like GGML_MOE_CCX_SPLIT=<domains>
                    std::vector<int> cpus = l3_prefetch_cpus();
                    if (const char * c = moe_opt("l3pf-cpus")) {
                        cpus.clear();
                        for (const char * q = c; *q; ) {
                            cpus.push_back(atoi(q));
                            while (*q && *q != ',') { ++q; }
                            if (*q == ',') { ++q; }
                        }
                    }
                    for (size_t d = 0; d < cpus.size(); ++d) {
                        mc->l3_thr.emplace_back(l3pf_loop, mc, (int) d, (int) cpus.size(), cpus[d]);
                    }
                    LLAMA_LOG_INFO("moe-cache: %zu L3 domains (prefetch CPUs%s), use GGML_MOE_CCX_SPLIT=%zu with L3PF\n", cpus.size(),
                        [&] { std::string r; for (int c : cpus) r += " " + std::to_string(c); return r; }().c_str(), cpus.size());
                }
                ggml_backend_set_split_callback(moe_split_cb, mc);
                LLAMA_LOG_WARN("moe-cache: router prediction for %zu layers: top-%d, up to %d uploads per layer\n",
                        n_pred, mc->pred_m, mc->pred_max);
            } else {
                mc->pred_m = 0;
                ggml_backend_set_split_callback(moe_split_cb, mc); // JIT: the GPU backends and the table flush before chains
            }
        }
        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        ggml_set_moe_phase_callback(moe_phase_cb, mc);
        {
            const char * e = moe_opt("prefill-d2d");
            if (!e || e[0] != '0') {
                ggml_backend_set_moe_src_callback(moe_src_cb, mc);
                ggml_backend_set_moe_fill_callback(moe_fill_cb, mc);
            }
            const char * r = moe_opt("pf-resident");   // the staging fill of a prompt pass copies what the card holds (moe_where_cb); 0 = upload everything
            if (!r || r[0] != '0') {
                ggml_backend_set_moe_where_fn_fork(moe_where_cb, mc);
            }
            // Prompt hit counting on the cards (--moe pf-count, default on; 0 = off): the CUDA backend counts the expert ids of every prompt expert matmul in
            // per-layer histograms (one small kernel per layer), read once per prompt in llama_moe_cache_prefill_report_fork: which experts the prompt used goes
            // into the cache's use counts (its heat for the answer), as the host callback did when it was on (GGML_PF_COUNT), without its host cost.
            const char * pc = moe_opt("pf-count");
            if ((!pc || pc[0] != '0') && !mc->layers.empty() && mc->layers[0].pub.up_src) {
                ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
                typedef bool (*obs_init_t)(int, int, int);
                obs_init_t init = reg ? (obs_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_obs_prompt_init_fork") : nullptr;
                int n_layer = 0;
                for (const auto & l : mc->layers) { n_layer = std::max(n_layer, l.pub.il + 1); }
                const int n_expert = (int) mc->layers[0].pub.up_src->ne[2];
                for (size_t i = 0; init && i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    const char * nm = ggml_backend_dev_name(d);
                    if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && nm && strncmp(nm, "CUDA", 4) == 0) {
                        mc->pf_count_on = init(atoi(nm + 4), n_layer, n_expert) || mc->pf_count_on;
                    }
                }
                mc->pf_count_layers  = n_layer;
                mc->pf_count_experts = n_expert;
            }
        }
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_WARN("%s: MoE expert cache enabled: %zu layers, slots/layer %d..%d, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(),
                std::min_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                std::max_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                mc->max_inserts, vram/1024.0/1024.0);
    }();
}

// JIT miss offload. Runs on the CPU (graph custom op) as soon as layer `pub`'s router ids are on the host, before the
// cache chain of that layer is launched. For m uncached experts of size S, CPU alone takes m*S/rc. Uploading k of them
// over the layer's link (rate rl) while the CPU computes the rest at the contended rate rcc = min(rc, ram - rl) (then rc
// once the link is done) finishes at T(k); the k with the smallest T wins, 0 when nothing beats the CPU alone. All rates
// are measured (link and CPU phase rates live, RAM budget by the startup probe). Uploaded experts become cache entries
// (same eviction score as the step's swaps), so the upload also serves later tokens.
void llama_moe_cache_jit_fork(const llama_moe_cache_layer_fork * pub, const ggml_tensor * ids) {
    moe_cache * mc = g_cache;
    if (!mc || knobs().jit == 0 || ids->ne[1] > 4) {
        return;
    }
    std::lock_guard<std::mutex> lock(mc->mtx);
    layer_state * lsp = nullptr;
    for (auto & l : mc->layers) {
        if (&l.pub == pub) { lsp = &l; break; }
    }
    if (!lsp) {
        return;
    }
    layer_state & ls = *lsp;
    const bool pool = ls.pub.jit_n > 0; // slower-link layer: the pool on the fastest-link GPU instead of its own cache
    const int  lk   = pool ? 0 : ls.link;
    ggml_backend_t be = mc->link_be[lk];
    const double rl = mc->gbs_link[lk];
    if (!be || rl <= 0) {
        return;
    }
    std::vector<int32_t> cur, miss;
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (e < 0 || e >= (int32_t) ls.expert_slot.size() || std::find(cur.begin(), cur.end(), e) != cur.end()) {
                continue;
            }
            cur.push_back(e);
            if (pool && ls.pool_slot[e] >= 0) {
                mc->n_pool_hit++;
            } else if (ls.expert_slot[e] < 0 && !ls.queued[e]) {
                miss.push_back(e);
            }
        }
    }
    mc->n_jit_miss += miss.size();
    if (miss.empty()) {
        return;
    }
    const double S   = (double) (pub->up_src->nb[2] + pub->gate_src->nb[2] + pub->down_src->nb[2]);
    const double rc  = (mc->cpu_gbs_meas > 0 ? mc->cpu_gbs_meas : knobs().cpu_gbs) * 1e3; // bytes/us
    const double rcc = std::min(rc, std::max(0.0, knobs().ddr_gbs * 1e3 - rl * 1e3));
    const double rlb = rl * 1e3;
    const int    m   = (int) miss.size();
    auto T = [&](int k) {
        const double tl = k * S / rlb, bc = (m - k) * S;
        if (k == 0) { return bc / rc; }
        return bc <= tl * rcc ? std::max(tl, rcc > 0 ? bc / rcc : 1e300) : tl + (bc - tl * rcc) / rc;
    };
    int kb = 0;
    for (int k = 1; k <= m; ++k) {
        if (T(k) < T(kb)) { kb = k; }
    }
    if (kb == 0) {
        return;
    }
    std::sort(miss.begin(), miss.end(), [&](int32_t a, int32_t b) { return score(ls, a) > score(ls, b); });
    int n_up = 0;
    if (pool) {
        for (int i = 0; i < kb; ++i) {
            const int32_t id = miss[i];
            // pool slot: an empty one, else the lowest-scored pool expert this token doesn't use
            int32_t ps = -1;
            double best = 1e300;
            for (int32_t s = 0; s < ls.pub.jit_n; ++s) {
                const int32_t x = ls.pool_expert[s];
                if (x < 0) { ps = s; break; }
                if (std::find(cur.begin(), cur.end(), x) != cur.end()) {
                    continue;
                }
                const double c = score(ls, x);
                if (c < best) { best = c; ps = s; }
            }
            if (ps < 0) {
                break;
            }
            const int32_t old = ls.pool_expert[ps];
            if (old >= 0) {
                ls.pool_slot[old] = -1;
                ls.pool_expert[ps] = -1;
                set_table_entry(ls.pub, old, ls.pub.n_slots); // back to the CPU
            }
            const ggml_tensor * srcs[3] = { pub->up_src,   pub->gate_src,   pub->down_src   };
            ggml_tensor *       dsts[3] = { pub->jit_up_c, pub->jit_gate_c, pub->jit_down_c };
            for (int k = 0; k < 3; ++k) {
                const size_t sz = srcs[k]->nb[2];
                ggml_backend_tensor_set_async(be, dsts[k], (const char *) srcs[k]->data + (size_t) id*sz, (size_t) ps*dsts[k]->nb[2], sz);
            }
            ls.pool_expert[ps] = id;
            ls.pool_slot[id]   = ps;
            ls.pool_dirty      = true;
            set_table_entry(ls.pub, id, ls.pub.n_slots + 1); // CPU skips it, the pool chain computes it
            mc->up_bytes += (int64_t) S; mc->link_bytes[ls.link] += (int64_t) S;
            ++n_up;
        }
        mc->n_pool_up += n_up;
        mc->n_jit += n_up;
        mc->n_jit_layers += n_up > 0;
        return;
    }
    for (int i = 0; i < kb; ++i) {
        const int32_t id = miss[i];
        if (knobs().admit_jit > 0 && ls.win_count[id] < knobs().admit_jit) {
            continue;  // not seen often enough in the last 64 tokens: the CPU computes it, it takes no slot
        }
        // slot: a free block, else the lowest-scored cached expert of THIS layer this token doesn't use (mid-graph: another layer's
        // table may already be on the GPU, so its blocks are not taken here; the step at the token boundary moves blocks between layers)
        double best;
        const size_t jli = (size_t) (&ls - mc->layers.data());
        const slot_pick pk = pick_slot(mc, jli, [&](layer_state & lo, size_t o, int32_t x) {
            return o != jli || pinned(mc, lo, x) || std::find(cur.begin(), cur.end(), x) != cur.end() ? -1.0 : score(lo, x);
        }, best);
        const int32_t slot = pk.slot;
        if (slot < 0) {
            break;
        }
        const int32_t victim = pk.owner >= 0 ? ls.slot_expert[slot] : -1;
        if (victim >= 0) {
            // JIT=1: admitted only as the step's swap would be (no churn): hotter than the victim by the link's pay-back
            // margin; JIT=2: any miss may evict the lowest-scored expert this token doesn't use
            if (knobs().jit < 2 && ((knobs().big != 0 && ls.glob_count[id] < ls.glob_count[victim]) ||
                    score(ls, id) < score(ls, victim) + pay_back(mc->link_margin[ls.link]))) {
                continue;
            }
            evict_slot(mc, ls, slot);
        }
        pool_take(mc, jli, slot);
        // async on the chain's stream: the chain (launched next) runs after the copies
        const ggml_tensor * srcs[3] = { pub->up_src, pub->gate_src, pub->down_src };
        ggml_tensor *       dsts[3] = { pub->up_c,   pub->gate_c,   pub->down_c   };
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            ggml_backend_tensor_set_async(be, dsts[k], (const char *) srcs[k]->data + (size_t) id*sz, (size_t) slot*dsts[k]->nb[2], sz);
        }
        ls.slot_expert[slot]    = id;
        ls.expert_slot[id]      = slot;
        ls.slot_last_use[slot]  = ++mc->clock;
        set_table_entry(ls.pub, id, slot); // host table now: the CPU op skips it; device table flushed before the chain
        ls.cached_since[id] = mc->n_steps + 1;
        ls.up_t[id]    = ggml_time_us();
        ls.up_cost[id] = (float) (S / rlb);
        ls.up_hits[id] = 0;
        mc->up_bytes  += (int64_t) S; mc->link_bytes[ls.link] += (int64_t) S;
        ++n_up;
    }
    mc->n_jit += n_up;
    mc->n_jit_layers += n_up > 0;
}

const llama_moe_cache_layer_fork * llama_moe_cache_lookup_fork(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    if (g_cache->starved.load()) {
        return nullptr;   // kv-tail: the KV holds every expert block: the graph is built without the cache chain (no work on experts that cannot be cached), the plain path
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

// SELF_TUNE=1: the streaming knobs are found on the running system, one at a time: candidate values alternate every SLICE
// decode tokens, each slice's mean token time is one sample; a knob is decided when the fastest candidate beats every
// other by two standard errors (or after MAX_S slices each), then the next knob; the whole cycle repeats after REST
// tokens so a changed workload is followed (4096, doubling up to 65536 while cycles confirm themselves). SLICE, MAX_S and
// 2 SE are the test's parameters, not tuning thresholds. On by default; LLAMA_MOE_CACHE_PRED_TUNE=0 / ctl SELF_TUNE=0: off.
// multi-GPU prefill, measured (LLAMA_PREFILL_SPLIT unset): alpha 1 = experts over all GPUs by link bandwidth, 0 = the
// fastest GPU only. Full prompt batches alternate between two alphas, each batch's tokens/s is one sample; a pair is
// decided when the faster mean beats the other by 2 SE (or after MAX_S batches each). Search: 1 vs 0; if all GPUs lose,
// step down: 0.5 vs 0, then 0.75 vs 0.5 or 0.25 vs 0. The result is held for a rest period (doubling while re-checks
// confirm it). Works for any GPU count and link widths; a box where splitting doesn't pay ends at 0.
static void prefill_split_tune(moe_cache * mc, double batch_us) {
    static const char * env = moe_opt("prefill-split");
    if (env || !g_split_enabled) {
        return; // fixed by LLAMA_PREFILL_SPLIT, or the buffers weren't reserved for a split
    }
    int n_known = 0;
    for (int k = 0; k < mc->n_links; ++k) { n_known += mc->gbs_link[k] > 0 && mc->link_dev[k]; }
    if (n_known < 2 || batch_us <= 0) {
        return;
    }
    auto set = [&](float a) { if (g_split_share != a) { g_split_share = a; g_split_epoch++; } };
    if (mc->ps_vals.empty()) {            // start: pair (all GPUs, one GPU)
        mc->ps_vals = { 1.0f, 0.0f };
        mc->ps_samples.assign(2, {});
        mc->ps_c = 0;
        set(mc->ps_vals[0]);
        return;                           // this batch ran before the tuner started
    }
    mc->ps_full = std::max(mc->ps_full, mc->last_tokens);
    if (mc->last_tokens < mc->ps_full) {
        return;                           // a prompt's last, partial batch
    }
    if (mc->ps_done) {
        if (--mc->ps_rest_left > 0) {
            return;
        }
        mc->ps_done = false;              // re-check from the top
        mc->ps_vals = { 1.0f, 0.0f };
        mc->ps_samples.assign(2, {});
        mc->ps_c = 0;
        set(mc->ps_vals[0]);
        return;
    }
    mc->ps_samples[mc->ps_c].push_back((double) mc->last_tokens / (batch_us / 1e6));
    const size_t MIN_S = 3, MAX_S = 6;
    const size_t n_min = std::min(mc->ps_samples[0].size(), mc->ps_samples[1].size());
    if (n_min >= MIN_S) {
        double mean[2], se[2];
        for (int c = 0; c < 2; ++c) {
            const auto & v = mc->ps_samples[c];
            double m = 0, q = 0;
            for (double x : v) { m += x; }
            m /= v.size();
            for (double x : v) { q += (x - m)*(x - m); }
            mean[c] = m; se[c] = sqrt(q / (v.size() - 1) / v.size());
        }
        const int w = mean[1] > mean[0] ? 1 : 0;
        const bool sure = mean[w] - 2*se[w] > mean[1 - w] + 2*se[1 - w];
        if (sure || n_min >= MAX_S) {
            const float a = mc->ps_vals[0], b = mc->ps_vals[1], win = mc->ps_vals[w];
            LLAMA_LOG_WARN("moe-cache: prefill split alpha %.2f: %.0f t/s vs %.2f: %.0f t/s -> %.2f (%s)\n",
                a, mean[0], b, mean[1], win, sure ? "clear" : "best mean");
            // next pair, stepping down from "all GPUs" only while the one-GPU side keeps winning
            std::vector<float> next;
            if (a == 1.0f && b == 0.0f && win == 0.0f)  { next = { 0.5f, 0.0f }; }
            else if (a == 0.5f && b == 0.0f)            { next = win == 0.5f ? std::vector<float>{ 0.75f, 0.5f } : std::vector<float>{ 0.25f, 0.0f }; }
            if (!next.empty()) {
                mc->ps_vals = next;
                mc->ps_samples.assign(2, {});
                mc->ps_c = 0;
                set(mc->ps_vals[0]);
                return;
            }
            const bool same = g_split_share == win;
            set(win);
            mc->ps_done = true;
            mc->ps_rest = mc->ps_rest == 0 ? 64 : (same ? std::min(4096, 2*mc->ps_rest) : 64);
            mc->ps_rest_left = mc->ps_rest;
            LLAMA_LOG_WARN("moe-cache: prefill split: alpha %.2f, re-check after %d prompt batches\n", win, mc->ps_rest);
            return;
        }
    }
    mc->ps_c ^= 1;
    set(mc->ps_vals[mc->ps_c]);
}

float llama_moe_cache_prefill_alpha_fork() {
    return g_split_share;
}

// The share of layer il's experts that card `dev` holds ready in its pool: 0 when the layer's cache lives on another card (the prompt split's planner:
// a card that holds a layer's experts computes them without uploading them).
double llama_moe_cache_resident_frac_fork(int il, ggml_backend_dev_t dev) {
    moe_cache * mc = g_cache;
    if (!mc || !dev) {
        return 0.0;
    }
    std::lock_guard<std::mutex> lock(mc->mtx);
    for (const auto & l : mc->layers) {
        if (l.pub.il != il || !l.pub.up_src || !l.pub.up_c || !l.pub.up_c->buffer) {
            continue;
        }
        if (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.pub.up_c->buffer)) != dev) {
            return 0.0;
        }
        int64_t held = 0;
        for (size_t e = 0; e < l.expert_slot.size(); ++e) {
            const int32_t sl = l.expert_slot[e];
            if (sl >= 0 && !((size_t) sl < l.slot_in_flight.size() && l.slot_in_flight[sl])) { held++; }
        }
        return (double) held / (double) std::max<int64_t>(1, l.pub.up_src->ne[2]);
    }
    return 0.0;
}

double llama_moe_cache_link_gbs_fork(ggml_backend_dev_t dev) {
    moe_cache * mc = g_cache; // set once at init
    if (mc && dev) {
        for (int k = 0; k < mc->n_links; ++k) {
            if (mc->link_dev[k] == dev && mc->gbs_link[k] > 0) {
                return mc->gbs_link[k];
            }
        }
    }
    for (const auto & e : g_early_gbs) {
        if (e.first == dev) {
            return e.second;
        }
    }
    return 0;
}

// Before the first compute-buffer reserve: the multi-GPU prefill split needs its compute buffers on every GPU, and they
// can't be taken back from the expert cache later, so the split is reserved only when it can pay: an early probe times
// 64 MB pinned-host -> device copies per GPU, and splitting's best transfer speedup is (sum of link GB/s) / (fastest).
// Reserved at >= 1.25x (2 x16: 2x; x16 + x4: 1.22x, where the split measured slower); alpha then starts at 1 and the
// tuner measures. LLAMA_PREFILL_SPLIT=<alpha> overrides (0: never).
// pinned host -> device GB/s of a GPU: 64 MB copies, measured once per process (the layer placement asks at load, the prefill split later)
static double early_link_gbs(ggml_backend_dev_t d) {
    for (const auto & e : g_early_gbs) {
        if (e.first == d) {
            return e.second;
        }
    }
    const size_t n = 64u << 20;
    ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(d);
    ggml_backend_buffer_t hbuf = hb ? ggml_backend_buft_alloc_buffer(hb, n) : nullptr;
    ggml_backend_buffer_t dbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d), n);
    double gbs = 0;
    if (hbuf && dbuf) {
        ggml_tensor t = {};
        t.type = GGML_TYPE_I8; t.buffer = dbuf; t.data = ggml_backend_buffer_get_base(dbuf);
        t.ne[0] = (int64_t) n; t.ne[1] = t.ne[2] = t.ne[3] = 1; t.nb[0] = 1; t.nb[1] = t.nb[2] = t.nb[3] = n;
        const void * src = ggml_backend_buffer_get_base(hbuf);
        ggml_backend_tensor_set(&t, src, 0, n); // warm-up
        const int64_t t0 = ggml_time_us();
        for (int r = 0; r < 4; ++r) { ggml_backend_tensor_set(&t, src, 0, n); }
        gbs = 4.0 * n / (double) std::max<int64_t>(1, ggml_time_us() - t0) / 1e3;
    }
    if (hbuf) { ggml_backend_buffer_free(hbuf); }
    if (dbuf) { ggml_backend_buffer_free(dbuf); }
    g_early_gbs.push_back({ d, gbs });
    return gbs;
}

// --moe layer-heat=1|fit: which expert layers each GPU holds for decode, by how HOT they are. Any number of GPUs.
//
// Why. A decode step computes a missed expert on the CPU and uploads it for later; the step waits for the uploads of the step before. Behind a slow link
// that wait is what a layer costs: MiMo-V2.6-Flash, x16 + x4, layers by share in model order, decode after a 12k prompt: 30% of the layers on the x4
// card 28.8 tokens/s, 40% 24.3, 50% 20.5, at the same 82-84% hit rate (at 50% the x4's layers are 37% resident, their misses cross the x4 link). A layer
// whose expert use is concentrated misses rarely with few of its experts in VRAM, and the saved hot maps (state file, hot.<layer>) say that layers
// differ a lot: the hottest half of a layer's experts carries 80% of its uses in the first layers of Qwen3.8-Flash-Next and 97% in layers 15 and 31
// (MiMo 69-97%, GLM-5.3-Flash 63-87%; the first layers are the flattest in all three, and the plain split gives exactly those to the first GPU).
//
// Rule. The weakest card takes the hottest layers.
//   cards   ranked by ONE measured number, the time to feed a card and get its result: the upload of one expert (its bytes over the bulk rate plus the
//           fixed latency of a small copy) plus the result latency of the probe (one token's activations up, a small quantised matmul, the result
//           down, end to end). It holds link rate, link latency and compute, so a card ranks low whatever makes it slow: a fast card in a narrow
//           slot, a slow card in a wide one. Capability = 1 / that time; any mix of cards and links.
//   layers  ranked by the area under their coverage curve (the share of uses the hottest k experts carry, averaged over k): hottest first.
//   =1      every card keeps the NUMBER of layers the split gave it; only which layers changes.
//   =fit    the numbers follow from memory and the cards' miss budgets: a card may miss in proportion to its capability (the strongest card lambda,
//           a card that is c times weaker lambda / c), a layer needs its dense weights plus the experts that keep its misses under that budget
//           (coverage curve), and each card, weakest first, takes the hottest remaining layers that fit; the strongest takes the rest. lambda is the
//           smallest value for which the rest fits the strongest card (bisection). A slow card so ends up with as many hot layers as fit nearly
//           whole, a fast one with the flat layers at whatever residency its memory allows.
// Without a saved hot map nothing moves (the first run of a model learns it). The placement is made at load: weights and memory are allocated for it.
bool llama_moe_layer_heat_fork(const llama_model & model, std::vector<ggml_backend_dev_t> & layer_dev, const std::vector<double> & lay_dense_b,
        const std::vector<double> & lay_exp_b) {
    const char * e = moe_opt("layer-heat");
    const int mode = !e ? 0 : strcmp(e, "fit") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    if (mode == 0) {
        return false;
    }
    std::vector<ggml_backend_dev_t> gpus;
    for (ggml_backend_dev_t d : layer_dev) {
        if (d && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU && std::find(gpus.begin(), gpus.end(), d) == gpus.end()) {
            gpus.push_back(d);
        }
    }
    if (gpus.size() < 2) {
        return false;
    }
    std::vector<std::pair<std::string, std::string>> kv;
    std::string name = model.name;
    for (char & c : name) {
        if (c == '[' || c == ']' || c == '\n' || c == '\r') { c = '_'; }
    }
    if (!moe_state_section_kv(name + " 0", kv, true)) {
        LLAMA_LOG_WARN("moe-cache: layer-heat: no saved usage profile for %s, layers stay where the split put them\n", name.c_str());
        return false;
    }
    // the layers: coverage curve and heat
    struct lay { int il; double heat; std::vector<double> cov; };   // cov[k]: share of the uses carried by the k + 1 hottest experts
    std::vector<lay> lays;
    double exp_b_sum = 0.0, n_exp_sum = 0.0;
    for (const auto & p : kv) {
        if (p.first.compare(0, 4, "hot.") != 0) {
            continue;
        }
        const int il = atoi(p.first.c_str() + 4);
        if (il < 0 || il >= (int) layer_dev.size() || std::find(gpus.begin(), gpus.end(), layer_dev[il]) == gpus.end()) {
            continue;
        }
        std::vector<double> c;
        std::istringstream is(p.second);
        for (double x; is >> x; ) { c.push_back(x); }
        const double tot = std::accumulate(c.begin(), c.end(), 0.0);
        if (c.empty() || tot <= 0) {
            continue;
        }
        std::sort(c.rbegin(), c.rend());
        lay l; l.il = il; l.heat = 0.0; l.cov.resize(c.size());
        double run = 0.0;
        for (size_t k = 0; k < c.size(); ++k) { run += c[k]; l.cov[k] = run / tot; l.heat += l.cov[k]; }
        l.heat /= (double) c.size();
        if ((size_t) il < lay_exp_b.size()) { exp_b_sum += lay_exp_b[il]; n_exp_sum += (double) c.size(); }
        lays.push_back(std::move(l));
    }
    if (lays.empty()) {
        return false;
    }
    std::sort(lays.begin(), lays.end(), [](const lay & a, const lay & b) { return a.heat > b.heat; });   // hottest first
    // the cards: capability = 1 / (one expert upload + the result latency), weakest first
    const double exp_b = n_exp_sum > 0 ? exp_b_sum / n_exp_sum : 2e6;   // bytes of one expert (up + gate + down)
    struct card { ggml_backend_dev_t dev; double cap, gbs, res_us, free_b; int n_now; };
    std::vector<card> cards;
    bool all_res = true;   // the result latency of every card or of none (it must be the same measurement on all of them to compare)
    for (ggml_backend_dev_t d : gpus) { const auto * r = llama_gpu_probe_find_fork(d); all_res = all_res && r && r->res_us > 0; }
    for (ggml_backend_dev_t d : gpus) {
        const auto * r = llama_gpu_probe_find_fork(d);
        const double gbs = r && r->h2d_gbs > 0 ? r->h2d_gbs : early_link_gbs(d);
        const double lat = r && r->lat_us > 0 ? r->lat_us : 0.0;
        const double up_us = lat + exp_b / (std::max(0.1, gbs) * 1e3);   // one expert upload, us
        const double res = all_res ? r->res_us : 0.0;
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(d, &f, &t);
        cards.push_back({ d, 1.0 / (up_us + res), gbs, res, (double) f, 0 });
    }
    for (const auto & l : lays) {
        for (auto & c : cards) { c.n_now += layer_dev[l.il] == c.dev; }
    }
    std::sort(cards.begin(), cards.end(), [](const card & a, const card & b) { return a.cap < b.cap; });   // weakest first
    const double cap_max = cards.back().cap;
    std::vector<int> n_take(cards.size(), 0);
    double lambda = -1.0;
    if (mode == 1) {
        for (size_t g = 0; g < cards.size(); ++g) { n_take[g] = cards[g].n_now; }
    } else {
        // experts a layer needs (as a share of its experts) to miss at most m of its uses
        const auto need_frac = [](const lay & l, double m) {
            const size_t k = (size_t) (std::lower_bound(l.cov.begin(), l.cov.end(), 1.0 - m) - l.cov.begin());
            return (double) std::min(l.cov.size(), k + 1) / (double) l.cov.size();
        };
        const auto bytes_of = [&](const lay & l, double m) {
            const double d = (size_t) l.il < lay_dense_b.size() ? lay_dense_b[l.il] : 0.0, x = (size_t) l.il < lay_exp_b.size() ? lay_exp_b[l.il] : 0.0;
            return d + x * need_frac(l, m);
        };
        static const double reserve = 2.0 * (1ull << 30);   // per card: compute buffers, K/V, the backend's own pool (the pool takes what is left after load)
        // the rest fits the strongest card at its miss budget lam? fills take[] on the way
        const auto place = [&](double lam, std::vector<int> & take) {
            size_t li = 0;
            for (size_t g = 0; g + 1 < cards.size(); ++g) {
                const double m = lam * cards[g].cap / cap_max;   // a weaker card may miss less
                double left = cards[g].free_b - reserve;
                take[g] = 0;
                while (li < lays.size() && bytes_of(lays[li], m) <= left) { left -= bytes_of(lays[li], m); ++li; ++take[g]; }
            }
            double need = 0.0;
            for (size_t i = li; i < lays.size(); ++i) { need += bytes_of(lays[i], lam); }
            take.back() = (int) (lays.size() - li);
            return need <= cards.back().free_b - reserve;
        };
        std::vector<int> take(cards.size(), 0);
        if (!place(1.0, take)) {
            LLAMA_LOG_WARN("moe-cache: layer-heat=fit: the layers do not fit the cards even with every expert outside VRAM, layers stay where the split put them\n");
            return false;
        }
        double lo = 0.0, hi = 1.0;
        for (int it = 0; it < 30; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (place(mid, take)) { hi = mid; } else { lo = mid; }
        }
        place(hi, n_take);
        lambda = hi;
    }
    size_t li = 0;
    std::string txt;
    for (size_t g = 0; g < cards.size(); ++g) {
        std::vector<int> mine;
        for (int k = 0; k < n_take[g] && li < lays.size(); ++k, ++li) {
            layer_dev[lays[li].il] = cards[g].dev;
            mine.push_back(lays[li].il);
        }
        std::sort(mine.begin(), mine.end());
        std::string ls;
        for (int il : mine) { ls += tr_fmt(" %d", il); }
        txt += tr_fmt("%s%s (capability %.2f of the strongest: link %.1f GB/s, result latency %.0f us): %zu layers:%s", g ? "; " : "",
                ggml_backend_dev_name(cards[g].dev), cards[g].cap / cap_max, cards[g].gbs, cards[g].res_us, mine.size(), ls.c_str());
    }
    LLAMA_LOG_WARN("moe-cache: layer-heat=%s: hottest layers on the weakest card%s: %s\n", mode == 2 ? "fit" : "1",
            lambda >= 0 ? tr_fmt(" (miss budget of the strongest card %.3f)", lambda).c_str() : "", txt.c_str());
    return true;
}

// Layers il_a and il_b trade GPUs (llama_context::swap_layers_fork moves their weights and memory). Their expert caches trade cards
// too: each layer's views, device table, pool class and link become the other's. Cached experts move in one direction only: those of the
// layer that goes to the card with the slower link (it would have to upload them over that link again), the highest-scored first, as many as
// that card's class has free blocks for; the layer that goes to the faster link refills from RAM if it needs to. Nothing is allocated (the copy
// goes through host memory, read before the blocks are given up).
// Call at a sync point (no graph running); waits for in-flight uploads of the two layers.
bool llama_moe_cache_swap_layers_fork(int il_a, int il_b) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return true;   // no expert cache: nothing to move
    }
    size_t ia = SIZE_MAX, ib = SIZE_MAX;
    for (size_t i = 0; i < mc->layers.size(); ++i) {
        if (mc->layers[i].pub.il == il_a) { ia = i; }
        if (mc->layers[i].pub.il == il_b) { ib = i; }
    }
    if (ia == SIZE_MAX || ib == SIZE_MAX) {
        return ia == ib;   // neither layer is cached: fine; only one: the caller checked the layouts, this cannot be swapped
    }
    layer_state & A = mc->layers[ia];
    layer_state & B = mc->layers[ib];
    for (int k = 0; k < 3; ++k) {
        const ggml_tensor * sa = k == 0 ? A.pub.up_src : k == 1 ? A.pub.gate_src : A.pub.down_src;
        const ggml_tensor * sb = k == 0 ? B.pub.up_src : k == 1 ? B.pub.gate_src : B.pub.down_src;
        if (sa->type != sb->type || sa->nb[2] != sb->nb[2]) {
            LLAMA_LOG_WARN("moe-cache: swap %d <-> %d: the experts differ in size, not swapped\n", il_a, il_b);
            return false;
        }
    }
    if ((A.pcls < 0) != (B.pcls < 0) || A.pub.jit_n > 0 || B.pub.jit_n > 0) {
        return false;
    }
    // wait for the uploads still running into either layer
    for (int spin = 0; ; ++spin) {
        {
            std::lock_guard<std::mutex> lk(mc->mtx);
            bool busy = false;
            for (const layer_state * l : { &A, &B }) {
                for (size_t s = 0; s < l->slot_in_flight.size() && !busy; ++s) { busy = l->slot_in_flight[s]; }
            }
            if (!busy) { break; }
        }
        if (spin > 20000) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::lock_guard<std::mutex> lk(mc->mtx);
    // the cached experts of each layer, hottest first, read to host
    struct moved { int32_t e; std::vector<uint8_t> data[3]; };
    auto take = [&](layer_state & L, size_t li, bool keep) {
        std::vector<std::pair<double, int32_t>> c;
        for (size_t s = 0; s < L.slot_expert.size(); ++s) {
            if (L.slot_expert[s] >= 0) { c.push_back({ xscore(mc, L, L.slot_expert[s]), (int32_t) s }); }
        }
        std::sort(c.rbegin(), c.rend());
        std::vector<moved> out;
        ggml_tensor * t3[3] = { L.pub.up_c, L.pub.gate_c, L.pub.down_c };
        for (const auto & p : c) {
            if (!keep) {
                break;
            }
            moved m;
            m.e = L.slot_expert[p.second];
            for (int k = 0; k < 3; ++k) {
                const ggml_tensor * src = k == 0 ? L.pub.up_src : k == 1 ? L.pub.gate_src : L.pub.down_src;
                m.data[k].resize(src->nb[2]);
                ggml_backend_tensor_get(t3[k], m.data[k].data(), (size_t) p.second * t3[k]->nb[2], src->nb[2]);
            }
            out.push_back(std::move(m));
        }
        for (const auto & p : c) {   // the layer leaves this card: its experts there are gone, its blocks free
            const int32_t e = L.slot_expert[p.second];
            L.expert_slot[e] = -1;
            L.slot_expert[p.second] = -1;
            L.cached_since[e] = 0;
            pool_release(mc, li, p.second);
        }
        return out;
    };
    // after the swap A has B's link and B has A's: the layer arriving on the slower link (link != 0) keeps its experts
    std::vector<moved> ma = take(A, ia, B.link != 0), mb = take(B, ib, A.link != 0);
    // the layers trade their card-bound cache state
    std::swap(A.pub.up_c, B.pub.up_c);
    std::swap(A.pub.gate_c, B.pub.gate_c);
    std::swap(A.pub.down_c, B.pub.down_c);
    std::swap(A.pub.dev_table, B.pub.dev_table);
    std::swap(A.pub.n_slots, B.pub.n_slots);
    std::swap(A.pcls, B.pcls);
    std::swap(A.n_own, B.n_own);
    std::swap(A.link, B.link);
    std::swap(A.slow, B.slow);
    std::swap(A.slot_expert, B.slot_expert);
    std::swap(A.slot_last_use, B.slot_last_use);
    std::swap(A.slot_in_flight, B.slot_in_flight);
    std::swap(A.is_stream, B.is_stream);
    std::swap(A.stream_score, B.stream_score);
    std::swap(A.stream_step, B.stream_step);
    std::swap(A.stream_slots, B.stream_slots);
    for (layer_state * L : { &A, &B }) {   // every expert uncached on the new card first
        for (int32_t e = 0; e < (int32_t) L->expert_slot.size(); ++e) { set_table_entry(L->pub, e, L->pub.n_slots); }
    }
    // the moved experts into free blocks of the new card's class
    int n_moved = 0, n_dropped = 0;
    auto put = [&](layer_state & L, size_t li, std::vector<moved> & ms) {
        ggml_tensor * t3[3] = { L.pub.up_c, L.pub.gate_c, L.pub.down_c };
        for (auto & m : ms) {
            double best;
            const int32_t s = pick_slot(mc, li, [](layer_state &, size_t, int32_t) { return -1.0; }, best).slot;   // free blocks only
            if (s < 0) { n_dropped++; continue; }
            pool_take(mc, li, s);
            for (int k = 0; k < 3; ++k) {
                ggml_backend_tensor_set(t3[k], m.data[k].data(), (size_t) s * t3[k]->nb[2], m.data[k].size());
            }
            L.slot_expert[s] = m.e;
            L.expert_slot[m.e] = s;
            L.slot_last_use[s] = ++mc->clock;
            L.cached_since[m.e] = mc->n_steps + 1;
            set_table_entry(L.pub, m.e, s);
            n_moved++;
        }
    };
    put(A, ia, ma);
    put(B, ib, mb);
    flush_tables();
    LLAMA_LOG_WARN("moe-cache: swap %d <-> %d: %d cached experts moved with their layers, %d dropped (no free block on the new card)\n", il_a, il_b, n_moved, n_dropped);
    return true;
}

void llama_moe_cache_prefill_decide_fork(const std::vector<ggml_backend_dev_t> & gpus) {
    if (const char * e = moe_opt("prefill-split")) {
        g_split_share   = (float) atof(e);
        g_split_enabled = g_split_share > 0;
    }
    if (gpus.size() < 2) {
        return;
    }
    double sum = 0, best = 0, worst = 1e30;
    std::string log;
    for (ggml_backend_dev_t d : gpus) {
        const double gbs = early_link_gbs(d);
        sum += gbs; best = std::max(best, gbs); worst = std::min(worst, gbs);
        log += tr_fmt(" %s %.1f", ggml_backend_dev_name(d), gbs);
    }
    if (moe_opt("prefill-split")) {
        return;
    }
    const double speedup = best > 0 ? sum / best : 1.0;
    // Off by default since 2026-10-23: the expert-only split (every card uploads and computes its share of each layer's experts) is not started from the probe's
    // verdict any more. It pays twice: the expert bytes go over both links, and every layer waits for the slower card, the same cost as extra passes (user,
    // 2026-10-23). The verdict ("saved compute > hop", no margin) sat on the line for MiMo-V2.6-Flash on x16 + x4: one start said 1.24x and stayed on one card,
    // the next said 1.25x and split, 17% slower at 3k (598/627 against 725/754 prompt t/s) and 24% slower on the cold 12k pass. Models with a dense split
    // try that first (its copies are uploaded once and the second card computes for free, llama-dsplit-fork.cpp); --moe prefill-split=F still forces this one.
    const int verdict = llama_prefill_split_verdict_fork();
    g_split_enabled = false;
    g_split_share   = 0.0f;
    g_split_speedup = speedup;
    LLAMA_LOG_WARN("moe-cache: prefill links (GB/s):%s -> split's transfer speedup %.2fx, probe verdict %s: one GPU (the expert-only split is off by default; --moe prefill-split=F forces it)\n",
        log.c_str(), speedup, verdict < 0 ? "none" : verdict == 1 ? "pays" : "does not pay");
    (void) worst;
}

// PREDICT switched: every layer's predictor off (or on, until the per-layer miss rule runs again), graph rebuilt once
static void apply_predict(moe_cache * mc) {
    const bool on = knobs().predict != 0;
    for (auto & ls : mc->layers) {
        if (ls.pub.pred_on != on) {
            ls.pub.pred_on = on;
            g_pred_epoch++;
        }
    }
}

static void self_tune(moe_cache * mc) {
    // slow: the knob changes what the cache holds, and its effect builds up over hundreds of tokens, so a candidate needs a long
    // slice (its first tokens run on the previous candidate's contents) instead of the 36-token slice of the transient knobs
    struct tunable { const char * name; double knobs_t::* f; std::vector<double> vals; bool slow = false; };
    // the streaming knobs only matter with the predictor; the swap path's are always live
    static const std::vector<tunable> T_stream = {
        { "PRED_N", &knobs_t::stream_m,  { 4, 6, 8, 12 } },
        { "PRED_AUTO",     &knobs_t::auto_tune, { 0, 1 } },
        { "IDLE_UP",      &knobs_t::tbp,       { 0, 2 } },
    };
    static const std::vector<tunable> T_swap = {
        { "UPLOAD_WAIT",     &knobs_t::gate,      { 0, 3 } },
        { "UPLOAD_SHARE", &knobs_t::swap_frac, { 0.25, 0.5 }, true }, // the upload budget: more uploads need DDR headroom, which differs per machine
        { "WAIT_SWAPS",     &knobs_t::wait,      { 0, 1 } },
    };
    static const std::vector<tunable> T = [&] {
        std::vector<tunable> t;
        // whatever the user set (LLAMA_MOE_CACHE_<NAME> in the environment) is theirs: never tuned. Deterministic mode fixes them all.
        const char * dm = moe_opt("fixed");
        const bool det_mode = dm && atoi(dm) != 0;
        auto add = [&](const tunable & x) {
            if (!det_mode && !g_autotune_off && !user_knobs().count(x.name)) { t.push_back(x); }
        };
        // the learned predictor is never switched on by the tuner (it costs performance): no PRED candidate
        if (mc->pred_m > 0 && knobs().stream > 0) { for (const auto & x : T_stream) { add(x); } }
        // the swap margin: 0 = any hotter expert may enter, -2 = pay-back margin from measured upload / CPU time. Which one wins depends
        // on the machine (slow link, RAM headroom), so it is tuned live; a state-carrying knob, hence the long slices
        add({ "SWAP_LEAD", &knobs_t::margin, { 0, -2 }, true });
        for (const auto & x : T_swap) { add(x); }
        return t;
    }();
    if (T.empty()) { return; } // -at off, every knob set by the user or deterministic mode: nothing to tune
    const size_t MIN_S = 3, MAX_S = 8;
    // a cycle costs ~2k tokens per state-carrying knob (half of them on the worse candidate): rest long enough to keep that under ~10%
    const uint64_t REST_MIN = 16384, REST_MAX = 131072;
    auto begin = [&](int k) {
        // the candidate in use runs first (the static default or the last decision, never a cold alternative); the others follow
        mc->lv_k = k; mc->lv_c = 0;
        for (size_t i = 0; i < T[k].vals.size(); ++i) { if (T[k].vals[i] == knobs().*T[k].f) { mc->lv_c = (int) i; } }
        mc->lv_inc = mc->lv_c;
        mc->lv_slice_len = T[k].slow ? 320 : 36; mc->lv_warm = T[k].slow ? 96 : 4;
        mc->lv_samples.assign(T[k].vals.size(), {});
        mc->lv_slice_n = 0; mc->lv_slice_sum = 0;
        knobs().*T[k].f = T[k].vals[mc->lv_c];
        apply_predict(mc);
    };
    // the window hit rate (256 steps): over the rest the workload can change (a new topic, a longer context). When it falls 5 points below the level the last cycle decided at,
    // the decisions are stale and the cycle starts now instead of when the rest timer ends
    auto window_hit = [&]() {
        uint64_t h = 0, m = 0;
        for (const auto & l : mc->layers) { h += l.n_hit; m += l.n_miss; }
        const uint64_t dh = h - mc->hr_h, dm = m - mc->hr_m;
        mc->hr_h = h; mc->hr_m = m; mc->hr_at = mc->n_steps;
        return dh + dm > 0 ? 100.0*dh/(double) (dh + dm) : -1.0;
    };
    if (mc->lv_k < 0) {
        if (mc->n_steps >= mc->lv_rest_until) { begin(0); return; }
        if (mc->n_steps >= mc->hr_at + 256) {
            const double w = window_hit();
            if (mc->hr_ref < 0) {
                mc->hr_ref = w;
            } else if (w >= 0 && w < mc->hr_ref - 5.0 && mc->n_steps > 512) {
                LLAMA_LOG_WARN("moe-cache: self-tune: hit rate %.1f%% against %.1f%% at the last decision: workload changed, tuning again now\n", w, mc->hr_ref);
                mc->hr_ref = -1;
                mc->lv_rest_until = mc->n_steps;
                begin(0);
            }
        }
        return;
    }
    if (mc->lv_slice_n < mc->lv_slice_len) {
        return;
    }
    const tunable & t = T[mc->lv_k];
    mc->lv_samples[mc->lv_c].push_back(mc->lv_slice_sum / (double) (mc->lv_slice_n - mc->lv_warm));
    mc->lv_slice_n = 0; mc->lv_slice_sum = 0;
    const size_t nv = t.vals.size();
    size_t n_min = SIZE_MAX;
    for (const auto & v : mc->lv_samples) { n_min = std::min(n_min, v.size()); }
    // a clearly worse alternative (6% slower than the incumbent after one slice each) is dropped at once: exploring costs tokens on the worse setting
    if (n_min >= 1 && n_min < MIN_S) {
        const auto & vi = mc->lv_samples[mc->lv_inc];
        double mi = 0; for (double x : vi) { mi += x; } mi /= vi.size();
        bool worse = true;
        for (size_t c = 0; c < nv; ++c) {
            if ((int) c == mc->lv_inc) { continue; }
            const auto & v = mc->lv_samples[c]; double m = 0; for (double x : v) { m += x; } m /= v.size();
            if (!(m > 1.06*mi)) { worse = false; }
        }
        if (worse) { n_min = MAX_S; }   // falls into the decision below with the incumbent
    }
    if (n_min >= MIN_S) {
        std::vector<double> mean(nv), se(nv);
        for (size_t c = 0; c < nv; ++c) {
            const auto & v = mc->lv_samples[c];
            double m = 0, q = 0;
            for (double x : v) { m += x; }
            m /= v.size();
            for (double x : v) { q += (x - m)*(x - m); }
            mean[c] = m; se[c] = v.size() > 1 ? sqrt(q / (v.size() - 1) / v.size()) : 1e300;
        }
        // mean: ms per token, lower is better. The incumbent keeps the knob unless an alternative is better by more than the noise (2 standard
        // errors of both) AND by at least 1%: a tuner that is not sure leaves the value that was in use
        const size_t inc = (size_t) mc->lv_inc;
        size_t b = inc;
        for (size_t c = 0; c < nv; ++c) { if (c != inc && mean[c] < mean[b]) { b = c; } }
        bool sure = true;
        for (size_t c = 0; c < nv; ++c) {
            if (c != b && mean[c] - 2*se[c] <= mean[b] + 2*se[b]) { sure = false; }
        }
        if (b != inc && (!sure || mean[b] > 0.99*mean[inc])) { b = inc; }
        if (sure || n_min >= MAX_S) {
            knobs().*t.f = t.vals[b];
            apply_predict(mc);
            std::string res;
            for (size_t c = 0; c < nv; ++c) { res += tr_fmt(" %g:%.2f", t.vals[c], mean[c]/1e3); }
            LLAMA_LOG_WARN("moe-cache: self-tune: %s ms/token%s -> %g (%s, %zu slices each)\n", t.name, res.c_str(), t.vals[b],
                sure ? "clear" : "best mean", n_min);
            if (mc->lv_prev.size() != T.size()) { mc->lv_prev.assign(T.size(), -1e300); }
            mc->lv_changed |= mc->lv_prev[mc->lv_k] != t.vals[b];
            mc->lv_prev[mc->lv_k] = t.vals[b];
            // next tunable; with the predictor off the streaming knobs change nothing: skip them this cycle
            int next = mc->lv_k + 1;
            while (next < (int) T.size() && knobs().predict == 0 &&
                    (T[next].f == &knobs_t::stream_m || T[next].f == &knobs_t::auto_tune || T[next].f == &knobs_t::tbp)) {
                next++;
            }
            if (next < (int) T.size()) {
                const double keep = knobs().*t.f;
                begin(next);
                knobs().*t.f = keep;
                apply_predict(mc);
            } else {
                // exploring costs tokens on worse settings: when a cycle confirms the last one, re-check half as often
                mc->lv_rest = mc->lv_changed ? REST_MIN : std::min(REST_MAX, std::max(REST_MIN, 2*mc->lv_rest));
                mc->lv_changed = false;
                mc->lv_k = -1;
                mc->hr_ref = -1; mc->hr_at = mc->n_steps;   // the next full window after the cycle sets the reference level
                { uint64_t h = 0, m = 0; for (const auto & l : mc->layers) { h += l.n_hit; m += l.n_miss; } mc->hr_h = h; mc->hr_m = m; }
                mc->lv_cycle++;
                mc->lv_rest_until = mc->n_steps + mc->lv_rest;
                LLAMA_LOG_WARN("moe-cache: self-tune: next check in %llu tokens\n", (unsigned long long) mc->lv_rest);
                mc->tuned_text.clear();
                for (const auto & tt : T) { mc->tuned_text += std::string(tt.name) + "=" + tr_fmt("%g\n", knobs().*tt.f); }
            }
            return;
        }
    }
    mc->lv_c = (mc->lv_c + 1) % (int) nv;
    knobs().*t.f = t.vals[mc->lv_c];
    apply_predict(mc);
}

void llama_moe_cache_step_fork() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    struct flush_at_exit { ~flush_at_exit() { flush_tables(); } } flush_guard; // table changes reach the GPU before the next graph
    if (mc->starved.load()) {
        return;   // kv-tail: the KV holds every expert block, nothing to score, admit or upload
    }

    // LLAMA_MOE_CACHE_DETERMINISTIC=1 (benchmarks): publish exactly the uploads
    // scheduled by the previous step, in a fixed order, with a fixed swap budget
    // and margin, so a run no longer depends on upload timing and repeats its
    // output. Uploads still overlap the token's compute; the wait is only for
    // the ones not finished by then.
    static const bool det = [] {
        const char * e = moe_opt("fixed");
        return e && atoi(e) != 0;
    }();
    // wait for and publish the previous step's uploads (default; LLAMA_MOE_CACHE_WAIT_SWAPS=0
    // disables): a cache that is current every step beats the time the wait costs,
    // fixed-text decode +16%, hit rate 66% -> 77%
    const bool wait_publish = det || knobs().wait != 0; // LLAMA_MOE_CACHE_WAIT_SWAPS / ctl WAIT

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::unique_lock<std::mutex> wlk(mc->wmtx);
        // queued stream jobs were for the token that just ended: drop them (their slots are freed at publish)
        for (auto it = mc->todo.begin(); it != mc->todo.end(); ) {
            if (it->stream) {
                upload_job j = *it;
                j.skipped = true;
                j.done    = true;
                mc->done.push_back(j);
                mc->n_stream_stale++;
                it = mc->todo.erase(it);
            } else {
                ++it;
            }
        }
        if (wait_publish) {
            // wait for the cache swaps only; a stream copy still running lands later and is published at a split
            mc->dcv.wait(wlk, [mc]() {
                return mc->in_flight == mc->in_flight_stream &&
                    std::none_of(mc->todo.begin(), mc->todo.end(), [](const upload_job & j) { return !j.stream && !j.bg; });
            });
            std::sort(mc->done.begin(), mc->done.end(), [](const upload_job & a, const upload_job & b) {
                return a.layer_idx != b.layer_idx ? a.layer_idx < b.layer_idx : a.slot < b.slot;
            });
        }
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            publish_job(mc, j);
        }
        mc->done.clear();

        // prefill preheat: the scheduler's device-to-device copies are done (the decode step syncs
        // the GPU first); publish experts whose up/gate/down all arrived, free the rest
        size_t n_adopt = 0;
        for (auto & ls : mc->layers) {
            for (int32_t v : ls.adopt_victims) {
                set_table_entry(ls.pub, v, ls.pub.n_slots);
                note_evict(mc, ls, v);
                ls.cached_since[v] = 0;
                if (ls.dropped[v]) { page_hint(mc, ls, v, false); ls.dropped[v] = false; }
            }
            ls.adopt_victims.clear();
            for (int32_t e : ls.adopted) {
                const int32_t slot = ls.adopt_slot[e];
                if (ls.adopt_mask[e] == 7) {
                    ls.slot_expert[slot]   = e;
                    ls.expert_slot[e]      = slot;
                    ls.slot_last_use[slot] = ++mc->clock;
                    set_table_entry(ls.pub, e, slot);
                    ls.cached_since[e] = mc->n_steps + 1;
                    n_adopt++;
                } else {
                    pool_release(mc, (size_t) (&ls - mc->layers.data()), slot);
                }
                ls.slot_in_flight[slot] = false;
                ls.adopt_slot[e] = -1;
                ls.adopt_mask[e] = 0;
            }
            ls.adopted.clear();
        }
        mc->n_adopted += n_adopt;
    }

    // swap budget for this step, from measured costs: keep upload time within
    // LLAMA_MOE_CACHE_UPLOAD_SHARE of the token time, minus what is still queued
    const double frac = knobs().swap_frac;
    int budget_total;
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        const int64_t now = ggml_time_us();
        if (mc->last_step_us && mc->last_prefill && mc->prev_prefill) {
            prefill_split_tune(mc, (double) (now - mc->last_step_us));
        }
        mc->prev_prefill = mc->last_prefill;
        if (mc->last_step_us) {
            const double dt = (double) (now - mc->last_step_us);
            if (dt < 5e6) { // ignore idle gaps between requests
                double & ema = mc->todo.empty() ? mc->step_us_idle : mc->step_us_busy;
                ema = ema ? 0.9*ema + 0.1*dt : dt;
                if (knobs().auto_l3 != 0) {
                    mc->ab_sum[mc->ab_state] += dt;
                    mc->ab_n[mc->ab_state]++;
                }
                if (knobs().self_tune != 0 && mc->lv_k >= 0 && !mc->last_prefill) {
                    if (++mc->lv_slice_n > mc->lv_warm) { // the first tokens after a switch still run on the previous setting's slots
                        mc->lv_slice_sum += dt;
                    }
                }
            }
        }
        // AUTO_L3: alternate L3PF off/on every 64 tokens; after 8 slices each keep the faster, re-test every 4096 tokens
        if (knobs().auto_l3 != 0) {
            if (mc->l3pf_set <= 0) { mc->l3pf_set = knobs().l3pf > 0 ? knobs().l3pf : 2; }
            if (mc->ab_decided >= 0 && mc->n_steps - mc->ab_since > 4096) {
                mc->ab_decided = -1; mc->ab_slices = 0; mc->ab_sum[0] = mc->ab_sum[1] = 0; mc->ab_n[0] = mc->ab_n[1] = 0;
            }
            if (mc->ab_decided < 0 && mc->n_steps % 64 == 0) {
                if (++mc->ab_slices >= 16 && mc->ab_n[0] && mc->ab_n[1]) {
                    const double t0 = mc->ab_sum[0] / mc->ab_n[0], t1 = mc->ab_sum[1] / mc->ab_n[1];
                    mc->ab_decided = t1 < t0 ? 1 : 0;
                    mc->ab_since = mc->n_steps;
                    LLAMA_LOG_WARN("moe-cache: auto L3 prefetch: off %.2f ms/token, on (%g experts) %.2f ms/token -> %s\n",
                        t0/1e3, mc->l3pf_set, t1/1e3, mc->ab_decided ? "on" : "off");
                } else {
                    mc->ab_state ^= 1;
                }
            }
            const int st = mc->ab_decided >= 0 ? mc->ab_decided : mc->ab_state;
            knobs().l3pf = st ? mc->l3pf_set : 0;
        }
        if (knobs().self_tune != 0) {
            self_tune(mc);
        }
        mc->last_step_us = now;
        const double step_us = std::max(mc->step_us_idle, mc->step_us_busy);
        if (mc->upload_us <= 0 || step_us <= 0) {
            budget_total = (int) mc->layers.size(); // bootstrap: ~1 per layer until measured
        } else {
            budget_total = (int) (frac*step_us*mc->workers.size()/mc->upload_us) - (int) std::count_if(mc->todo.begin(), mc->todo.end(), [](const upload_job & j) { return !j.bg; });
        }
        // static budget: LLAMA_MOE_CACHE_SWAPS_PER_STEP swaps per step (deterministic mode: default 8)
        const int fixed_budget = (int) knobs().budget;
        if (fixed_budget >= 0 || det) {
            budget_total = fixed_budget >= 0 ? fixed_budget : 8;
        }
        budget_total = std::max(0, budget_total);
        mc->last_budget = budget_total;
    }

    // a swap must pay for itself: over the count's horizon the candidate saves
    // (count_c - count_v) CPU expert evals of t_cpu each, and costs one upload
    const double cpu_gbs = knobs().cpu_gbs;
    // per upload link: a slow (x4) link pays more per swap, so its experts are only replaced by clearly hotter ones
    uint32_t margin = 1;
    int link_margin[MAX_LINKS] = {};
    // upload time / CPU eval time of the same expert = CPU GB/s / link GB/s, whatever the layer's expert size
    if (mc->gbs_all > 0) {
        margin = (uint32_t) std::max(1.0, std::ceil(cpu_gbs / mc->gbs_all));
        for (int k = 0; k < mc->n_links; ++k) {
            link_margin[k] = mc->gbs_link[k] > 0 && knobs().link != 0 ? (int) std::max(1.0, std::ceil(cpu_gbs / mc->gbs_link[k])) : (int) margin;
        }
    }
    // static pay-back margin: LLAMA_MOE_CACHE_SWAP_LEAD, default 0 (deterministic mode: 4). The timing margin above (-2)
    // charges an upload as CPU time, but the DMA runs beside the CPU and only costs DDR contention: on GLM topic-switching
    // chats (PC1, x4 link margin 9-11) it held the cache at 64.7% hit / 19.93 t/s vs 71.2% / 21.26 t/s with 0
    const int fixed_margin = knobs().margin == -1 ? (det ? 4 : 0) : (int) knobs().margin;
    if (fixed_margin >= 0) {
        margin = (uint32_t) fixed_margin;
        link_margin[0] = link_margin[1] = (int) margin;
    }
    mc->last_margin = (int) margin;
    mc->link_margin[0] = link_margin[0];
    mc->link_margin[1] = link_margin[1];

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;
    if (!mc->ctl.empty() && mc->n_steps % 16 == 0) {
        ctl_poll(mc);
    }
    if (!g_tr.prefix.empty()) {
        const int64_t now = ggml_time_us();
        if (g_tr.skip > 0) {
            g_tr.skip--;
        } else if (g_tr.left > 0) {
            if (g_tr.t_step) {
                tr(TR_SCHED, tr_fmt("token %llu", (unsigned long long) mc->n_steps), g_tr.t_step, now - g_tr.t_step);
            }
            if (--g_tr.left == 0) {
                tr_dump();
            }
        }
        g_tr.t_step = now;
    }
    // learned predictors: this token trains them (in the graph) when mu > 0
    if (mc->pred_ahead > 0 && mc->pred_m > 0) {
        const float want = mc->pred_train > 0 && mc->n_steps % mc->pred_train == 0 ? mc->pred_mu : 0.0f;
        g_pred_train_now = want > 0;
        if (want != mc->cur_mu) {
            for (auto & ls : mc->layers) {
                if (ls.pub.pred_mu) {
                    ggml_backend_tensor_set(ls.pub.pred_mu, &want, 0, sizeof(float));
                }
            }
            mc->cur_mu = want;
        }
        mc->n_train += want > 0;
    }
    if (mc->pred_m > 0 && mc->n_steps % 1024 == 0 && mc->win_tot[0] > 0) {
        std::string msg;
        for (int k = 0; k < 8 && mc->win_tot[k] > 0; ++k) {
            msg += format(" L+%d %.1f%%", k + 1, 100.0*mc->win_hit[k]/mc->win_tot[k]);
            mc->win_hit[k] = mc->win_tot[k] = 0;
        }
        LLAMA_LOG_INFO("moe-cache: predicted top-k overlap, last 1024 tokens:%s (%" PRIu64 " training steps)\n", msg.c_str(), mc->n_train);
    }
    mc->last_obs    = -1;
    mc->gpu_started = -1;
    // drop-cached: only experts that stayed in VRAM for LLAMA_MOE_CACHE_DROP_STAY steps (default 1024)
    // lose their RAM pages; churning ones never do (dropping and re-reading them cost more than it saved)
    if (mc->drop_cached && mc->n_steps % 256 == 0) {
        static const uint64_t stay = [] {
            const char * e = moe_opt("drop-stay");
            return (uint64_t) (e ? atoll(e) : 1024);
        }();
        for (auto & ls : mc->layers) {
            for (int32_t sl = 0; sl < ls.pub.n_slots; ++sl) {
                const int32_t e = ls.slot_expert[sl];
                if (e >= 0 && !ls.dropped[e] && ls.cached_since[e] && mc->n_steps - ls.cached_since[e] >= stay) {
                    page_hint(mc, ls, e, true);
                    ls.dropped[e] = true;
                }
            }
        }
    }
    // predict only where it can pay: a layer's predictor runs while one of its target layers misses
    // more than LLAMA_MOE_CACHE_PREDICT_MISS (default 0.05) of its expert uses (windows of 256 steps)
    if (mc->pred_ahead > 0 && mc->n_steps % 256 == 0) {
        static const double thr = [] {
            const char * e = moe_opt("predict-miss");
            return e ? atof(e) : 0.05;
        }();
        const size_t nl = mc->layers.size();
        std::vector<double> miss(nl, 0.0);
        for (size_t li = 0; li < nl; ++li) {
            auto & ls = mc->layers[li];
            const uint64_t h = ls.n_hit - ls.pred_seen_hit, m = ls.n_miss - ls.pred_seen_miss;
            miss[li] = h + m ? (double) m / (h + m) : 0.0;
            ls.pred_seen_hit = ls.n_hit; ls.pred_seen_miss = ls.n_miss;
        }
        size_t n_on = 0;
        for (size_t li = 0; li < nl; ++li) {
            auto & pub = mc->layers[li].pub;
            bool on = false;
            for (int k = 1; k <= std::max(1, pub.pred_ahead) && li + k < nl; ++k) {
                on = on || miss[li + k] > thr;
            }
            on = on && knobs().predict != 0;
            if (on != pub.pred_on) {
                pub.pred_on = on;
                g_pred_epoch++;
            }
            n_on += on;
        }
        if (mc->n_steps % 4096 == 0) {
            LLAMA_LOG_INFO("moe-cache: predicting from %zu of %zu layers (target miss rate > %.2f)\n", n_on, nl, thr);
        }
    }
    if (mc->n_steps % 4096 == 0) { // follow the workload: re-pick the always-hot set from lifetime counts
        for (auto & ls : mc->layers) {
            refresh_sticky(ls);
        }
    }

    // 1b) promotion: a streamed expert that was used and outscores the weakest evictable cached expert takes its
    //     place by swapping the two slots' roles (no copy): its slot joins the cache, the victim's becomes a stream slot
    for (auto & ls : mc->layers) {
        for (int32_t id : ls.stream_hit) {
            const int32_t s_str = ls.expert_slot[id];
            if (s_str < 0 || !ls.is_stream[s_str]) {
                continue;
            }
            int32_t s_vic = -1;
            double best = 1e300;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s] || ls.is_stream[s]) {
                    continue;
                }
                const int32_t v = ls.slot_expert[s];
                if (v < 0) { s_vic = s; best = -1; break; }
                if (pinned(mc, ls, v)) {
                    continue;
                }
                const double c = score(ls, v);
                if (c < best) { best = c; s_vic = s; }
            }
            if (s_vic < 0 || score(ls, id) <= best) {
                continue;
            }
            if (knobs().big != 0 && ls.slot_expert[s_vic] >= 0 && ls.glob_count[id] < ls.glob_count[ls.slot_expert[s_vic]]) {
                continue;
            }
            const int32_t v = ls.slot_expert[s_vic];
            if (v >= 0) {
                ls.expert_slot[v]     = -1;
                ls.slot_expert[s_vic] = -1;
                set_table_entry(ls.pub, v, ls.pub.n_slots);
                note_evict(mc, ls, v);
                ls.cached_since[v] = 0;
                if (ls.dropped[v]) { page_hint(mc, ls, v, false); ls.dropped[v] = false; }
            }
            ls.is_stream[s_str] = 0;
            ls.is_stream[s_vic] = 1;
            std::replace(ls.stream_slots.begin(), ls.stream_slots.end(), s_str, s_vic);
            ls.prefetched[id] = 0;
            mc->n_promoted++;
        }
        ls.stream_hit.clear();
    }

    // 1c) stream self-tuning per link, every 64 steps: lead from the late share, candidates from the precision
    if (knobs().auto_tune == 1 && knobs().stream > 0 && mc->n_steps % 64 == 0) {
        for (int k = 0; k < mc->n_links; ++k) {
            const uint64_t up = mc->st_up[k], late = mc->st_late[k], hit = mc->st_hit[k];
            if (up < 16 || mc->stream_off[k]) {
                continue;
            }
            int & m = mc->auto_m[k];
            if (m == 0) {
                m = (int) knobs().stream_m;
            }
            const double fl = (double) late / up, pr = (double) hit / up;
            if (fl > 0.25) {
                mc->lead_extra[k] = std::min(mc->lead_extra[k] + 1, 2); // late share counts finished copies only (not queue drops)
            } else if (fl < 0.05 && mc->lead_extra[k] > 0) {
                mc->lead_extra[k]--;
            }
            if (pr < 0.35) {
                m = std::max(2, m - 1);
            } else if (pr > 0.6) {
                m = std::min(12, m + 1);
            }
            mc->stream_off[k] = m == 2 && pr < 0.2;
            LLAMA_LOG_WARN("moe-cache: auto: %s link %llu streams, %.0f%% late, %.0f%% used -> lead +%d, M %d%s\n", tr_fmt("GPU link %d", k).c_str(),
                    (unsigned long long) up, 100*fl, 100*pr, mc->lead_extra[k], m, mc->stream_off[k] ? ", streaming off" : "");
            mc->st_up[k] = mc->st_late[k] = mc->st_hit[k] = 0;
        }
    }

    // 1d) token-boundary prefetch: the first MoE layers' misses of the token that just ended go to stream slots now,
    //     while the output head, sampling and the dense layers leave DDR idle; consecutive tokens share many experts
    if (knobs().tbp > 0 && knobs().stream > 0) {
        std::vector<upload_job> bj;
        const size_t nb = std::min(mc->layers.size(), (size_t) knobs().tbp_layers);
        for (size_t li = 0; li < nb; ++li) {
            auto & ls = mc->layers[li];
            const int32_t n_ss = std::min<int32_t>((int32_t) knobs().stream, (int32_t) ls.stream_slots.size());
            if (n_ss <= 0) {
                continue;
            }
            std::vector<int32_t> c = ls.pending;
            std::sort(c.begin(), c.end(), [&](int32_t a, int32_t b) { return score(ls, a) > score(ls, b); });
            int n = 0;
            for (int32_t id : c) {
                if (n >= (int) knobs().tbp || n >= n_ss) {
                    break;
                }
                if (ls.expert_slot[id] >= 0 || ls.queued[id] || ls.adopt_slot[id] >= 0) {
                    continue;
                }
                int32_t slot = -1;
                for (int32_t t = 0; t < n_ss && slot < 0; ++t) {
                    const int32_t sl = ls.stream_slots[ls.stream_next++ % n_ss];
                    slot = ls.slot_in_flight[sl] ? -1 : sl;
                }
                if (slot < 0) {
                    break;
                }
                const int32_t victim = ls.slot_expert[slot];
                if (victim >= 0) {
                    ls.expert_slot[victim] = -1;
                    ls.slot_expert[slot]   = -1;
                    set_table_entry(ls.pub, victim, ls.pub.n_slots);
                    ls.cached_since[victim] = 0;
                    ls.prefetched[victim]   = 0;
                }
                ls.slot_in_flight[slot] = true;
                ls.queued[id]           = 1;
                upload_job j { li, id, slot };
                j.urgent   = true;
                j.stream   = true;
                j.step     = mc->n_steps;
                j.t_queued = ggml_time_us();
                bj.push_back(j);
                n++;
            }
        }
        if (!bj.empty()) {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            for (auto it = bj.rbegin(); it != bj.rend(); ++it) {
                mc->todo.push_front(*it);
            }
            mc->n_tbp += bj.size();
            mc->wcv.notify_all();
        }
    }

    const size_t n_back = alias_release(mc);
    if (poison_on() && n_back > 0) { for (const auto & ap : mc->alias) { poison_free_blocks(mc, ap); } }   // the blocks given back hold nothing now
    if (mc->kv_gave_back.exchange(false)) {   // the expert cache grew at the start of a request: the freed blocks get the hottest experts the cache misses (scores seeded from the state ini)
        reheat_free_blocks(mc, det, true);
    }
    mc->gmax_all = 1;
    uint32_t tmax = 1;
    for (const auto & l : mc->layers) { mc->gmax_all = std::max(mc->gmax_all, l.glob_max); tmax = std::max(tmax, l.topic_max); }
    g_topic_tmax.store(tmax, std::memory_order_relaxed);
    const bool decode_graph = mc->graph_tokens_last <= (int32_t) std::max<int64_t>(8, llama_moe_cache_max_batch_fork());
    // hot maps per topic: the first decode step after a prompt drops the cached experts neither the topic's map nor the request used; the refill below
    // fills their blocks too
    size_t n_drop = 0;
    if (g_topic.after_prompt && decode_graph) {
        g_topic.after_prompt = false;
        const int64_t t0 = ggml_time_us();
        n_drop = topic_drop_foreign(mc);
        LLAMA_LOG_WARN("moe-cache: dynhot: the prompt is over: %zu cached experts that neither the topic's map nor the request used dropped (%.2f ms; the refill follows)\n",
                n_drop, (ggml_time_us() - t0) / 1e3);
    }
    // blocks came back and the graph that just ran was a decode graph: the prompt is over (between the passes of a prompt nothing comes back, and
    // after a smaller pass that is still a prompt pass the next pass may take the blocks again)
    if ((n_back > 0 || n_drop > 0) && decode_graph) {
        pool_fill_log(mc, "after a prompt (its blocks back)");
        reheat_free_blocks(mc, det);
        mc->fill_from = (int64_t) mc->n_steps;
        mc->fill_t0   = ggml_time_us();
    } else if (mc->fill_from >= 0) {
        const int64_t d = (int64_t) mc->n_steps - mc->fill_from;
        if (d == 8 || d == 32 || d == 128) {
            pool_fill_log(mc, tr_fmt("%" PRId64 " decode steps (%.0f ms) after a prompt", d, (ggml_time_us() - mc->fill_t0) / 1e3).c_str());
            if (d == 128) { mc->fill_from = -1; }
        }
    }
    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    const size_t n_layers = mc->layers.size();
    mc->rr = (mc->rr + 1) % n_layers;
    for (size_t k = 0; k < n_layers; ++k) {
        const size_t li = (mc->rr + k) % n_layers;
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        // hottest candidates first; the budget only throttles evictions,
        // filling empty slots is free
        std::sort(ls.pending.begin(), ls.pending.end(), [&](int32_t a, int32_t b) {
            return score(ls, a) > score(ls, b);
        });
        mc->pr[3] += ls.pending.size();
        if (get_policy() == policy::rel && !g_rel_dead.load() && mc->pr[0] == 0 && mc->n_steps > 4) {
            g_rel_dead = true;
            LLAMA_LOG_WARN("moe-cache: policy r: no router probabilities reached the cache after %" PRIu64 " steps: using the default score instead\n", mc->n_steps);
        }
        // swaps cost PCIe bandwidth and latency: cap evictions per layer per step,
        // and none at all while earlier uploads are still queued
        int & budget = budget_total;
        for (auto it = ls.pending.begin(); it != ls.pending.end(); ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.queued[id] || score(ls, id) <= 0) {
                mc->pr[11]++;
                continue;
            }
            double admit_now = ls.slow && knobs().admit_slow > 0 ? knobs().admit_slow : knobs().admit;
            if (!mc->boot_done && knobs().admit_boot > admit_now && (double) (mc->n_steps - mc->boot_from) < knobs().admit_boot_steps) { admit_now = knobs().admit_boot; }
            if (ls.win_count[id] < admit_now) {
                mc->pr[6]++;
                continue;
            }

            // victim: a free block if any, else the lowest-scored cached expert (block pool: of ANY layer of the class, so the blocks go to the
            // layers whose experts score highest; without the pool: of this layer)
            double best;
            const slot_pick pk = pick_slot(mc, li, [&](layer_state & lo, size_t, int32_t e) { return pinned(mc, lo, e) ? -1.0 : xscore(mc, lo, e); }, best);
            const int32_t slot = pk.slot;
            if (slot < 0) {
                mc->pr[9] += ls.pending.end() - it;
                break; // every slot is in flight; try again next step
            }

            layer_state & lo = pk.owner >= 0 ? mc->layers[pk.owner] : ls;
            const int32_t victim = pk.owner >= 0 ? lo.slot_expert[slot] : -1;
            if (victim >= 0) {
                if (knobs().big != 0 && ls.glob_count[id] < lo.glob_count[victim]) {
                    mc->pr[10]++;
                    continue; // lifetime regular stays; a later candidate may still fit
                }
                if (xscore(mc, ls, id) < xscore(mc, lo, victim) + pay_back(link_margin[ls.link])) {
                    mc->pr[7] += ls.pending.end() - it;   // this and every later (colder) candidate
                    break; // candidates are sorted: nothing hotter than what's cached
                }
                if (budget-- <= 0) {
                    mc->pr[8] += ls.pending.end() - it;
                    break;
                }
                mc->pr[5]++;
                evict_slot(mc, lo, slot);
            }
            pool_take(mc, li, slot);
            mc->pr[4] += victim < 0;
            ls.slot_in_flight[slot] = true;
            ls.queued[id]           = 1;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            upload_job sj { li, id, slot };
            sj.t_queued = ggml_time_us();
            mc->todo.push_back(sj);
        }
        ls.pending.clear();
    }
    mc->wcv.notify_all();

    // decay: counts halve every N steps, so recent use dominates old use
    static const uint64_t halve_every = [] {
        const char * h = moe_opt("halve-every");
        return (uint64_t) std::max(1, h ? atoi(h) : 64);
    }();
    if (mc->n_steps % halve_every == 0) {
        for (auto & ls : mc->layers) {
            for (auto & c : ls.expert_count) { c >>= 1; }
        }
    }

    static const bool stats = moe_log_has("p");   // --moe log=p: periodic cache stats on the log, every 64 steps or every N with log=pN
    static const int stats_every = std::max(1, moe_log_num('p', 64));
    if (stats && mc->n_steps % stats_every == 0) {
        static uint64_t ph = 0, pm = 0;
        uint64_t h = 0, m = 0, filled = 0, inflight = 0, total = 0;
        for (auto & ls : mc->layers) {
            h += ls.n_hit; m += ls.n_miss; total += ls.pcls >= 0 ? ls.n_own : ls.pub.n_slots;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                filled   += ls.slot_expert[s] >= 0;
                inflight += ls.slot_in_flight[s];
            }
        }
        size_t queued;
        {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            queued = mc->todo.size();
        }
        const uint64_t dh = h - ph, dm = m - pm;
        ph = h; pm = m;
        {   // the same numbers as a CSV next to the routing trace: routing/<stem>.cache.csv, one line per interval (log=p1: one per token). demand = met + unmet = the selected experts of
            // all layers), met = already in VRAM, unmet = selected but not in VRAM (computed on the CPU, a swap candidate), load = mean over layers of (experts used in the last 64 tokens) / slots: above 1 the cache cannot hold the working set
            static uint64_t prev[16] = {}, pup = 0, pev = 0;
            double load = 0;
            for (auto & ls : mc->layers) {
                int act = 0;
                for (uint32_t c : ls.win_count) { act += c > 0; }
                load += (ls.pcls >= 0 ? ls.n_own : ls.pub.n_slots) > 0 ? (double) act / (ls.pcls >= 0 ? ls.n_own : ls.pub.n_slots) : 0;
            }
            load /= std::max<size_t>(1, mc->layers.size());
            const std::filesystem::path path = moe_log_file(("routing/" + moe_run_stem(mc->profile) + ".cache.csv").c_str());
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            const bool fresh = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
            std::string out = fresh ? "step,demand,met,unmet,load,filled,slots,uploads,evictions,pending,admit_free,admit_evict,d_admit,d_margin,d_budget,d_victim,d_big,d_dup,regret,ev_unpaid,ev_zero,t_ms\n" : "";
            char b[512];
            snprintf(b, sizeof b, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64, mc->n_steps, dh + dm, dh, dm, load, filled, total, mc->n_uploads - pup, mc->n_evictions - pev);
            out += b;
            for (int i : {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) { out += "," + std::to_string(mc->pr[i] - prev[i]); prev[i] = mc->pr[i]; }
            // wall-clock time of the interval end in ms since the epoch (microsecond resolution), the same clock as the GPU sampler of fork/run.py: a change of speed lines up with power, clocks and swaps
            snprintf(b, sizeof b, ",%.3f", std::chrono::duration<double, std::micro>(std::chrono::system_clock::now().time_since_epoch()).count() / 1e3);
            out += b;
            out += "\n";
            pup = mc->n_uploads; pev = mc->n_evictions;
            std::ofstream cf(path, std::ios::app);
            cf << out;
        }
        LLAMA_LOG_WARN("moe-cache: step %" PRIu64 " filled %" PRIu64 "/%" PRIu64 " inflight %" PRIu64 " queued %zu window-hit %.1f%% | upload %.2f ms/expert, token %.1f ms idle / %.1f ms busy, swap budget %d, min count gain %d (fast link %.2f ms -> %d, slow link %.2f ms -> %d)\n",
                mc->n_steps, filled, total, inflight, queued, dh + dm ? 100.0*dh/(dh + dm) : 0.0,
                mc->upload_us/1e3, mc->step_us_idle/1e3, mc->step_us_busy/1e3, mc->last_budget, mc->last_margin,
                mc->link_us[0]/1e3, mc->link_margin[0], mc->link_us[1]/1e3, mc->link_margin[1]);
        if (knobs().jit != 0) {
            static uint64_t pj = 0, pl = 0, pmiss = 0;
            static uint64_t ppu = 0, pph = 0;
            LLAMA_LOG_WARN("moe-cache: JIT: %.2f experts/token uploaded for immediate use in %.2f layers/token, of %.2f misses/token; pool %.2f uploads, %.2f hits/token\n",
                    (mc->n_jit - pj)/64.0, (mc->n_jit_layers - pl)/64.0, (mc->n_jit_miss - pmiss)/64.0, (mc->n_pool_up - ppu)/64.0, (mc->n_pool_hit - pph)/64.0);
            pj = mc->n_jit; pl = mc->n_jit_layers; pmiss = mc->n_jit_miss; ppu = mc->n_pool_up; pph = mc->n_pool_hit;
        }
        for (int k = 0; k < mc->n_links; ++k) {
            if (!mc->n_link[k]) {
                continue;
            }
            const double n = (double) std::max<uint64_t>(1, mc->ev_n[k]);
            LLAMA_LOG_WARN("moe-cache: %s link: upload %.2f ms = read %.2f + copy %.2f, %.1f GB/s (CPU %.0f GB/s) | evicted %" PRIu64 ": %.1f hits each, paid back %.0f%%, cost %.2f ms, lived %.1f s\n",
                    tr_fmt("GPU link %d", k).c_str(), mc->link_us[k]/1e3, mc->read_link[k]/1e3, mc->copy_link[k]/1e3, mc->gbs_link[k], knobs().cpu_gbs,
                    mc->ev_n[k], mc->ev_hits[k]/n, 100.0*mc->ev_paid[k]/n, mc->ev_cost_us[k]/n/1e3, mc->ev_life_us[k]/n/1e6);
        }
        {
            // DDR4 phases over the last 64 steps: CPU busy on uncached experts vs wall time, and uploads colliding with it
            const int64_t now = ggml_time_us();
            const int64_t bu = mc->busy_us, bn = mc->busy_n;
            const double  st = (double) std::max<uint64_t>(1, mc->n_steps - mc->seg_steps2);
            const uint64_t up = mc->n_uploads - mc->seg_up2, ov = mc->up_overlap - mc->seg_overlap;
            if (mc->seg_t0) {
                LLAMA_LOG_WARN("moe-cache: DDR: CPU experts %.0f MB/token at %.1f GB/s in their phases, uploads %.0f MB/token | avg %.1f GB/s\n",
                        (mc->cpu_bytes - mc->seg_cpu_bytes)/st/1e6, mc->cpu_gbs_meas, (mc->up_bytes - mc->seg_up_bytes)/st/1e6,
                        (double) (mc->cpu_bytes - mc->seg_cpu_bytes + mc->up_bytes - mc->seg_up_bytes) / std::max<int64_t>(1, now - mc->seg_t0) / 1e3);
                LLAMA_LOG_WARN("moe-cache: phases: token %.1f ms wall, CPU expert matmuls %.1f ms in %.1f phases (%.0f%%) | %.1f uploads/token, %.0f%% overlapping CPU phases, gate waits %.2f ms/token\n",
                        (now - mc->seg_t0)/st/1e3, (bu - mc->seg_busy_us)/st/1e3, (bn - mc->seg_busy_n)/st,
                        100.0*(bu - mc->seg_busy_us)/std::max<int64_t>(1, now - mc->seg_t0), up/st, up ? 100.0*ov/up : 0.0,
                        (mc->gate_wait_us - mc->seg_gate_us)/st/1e3);
            }
            mc->seg_t0 = now; mc->seg_busy_us = bu; mc->seg_busy_n = bn; mc->seg_steps2 = mc->n_steps; mc->seg_up2 = mc->n_uploads; mc->seg_overlap = mc->up_overlap;
            mc->seg_gate_us = mc->gate_wait_us;
            mc->seg_cpu_bytes = mc->cpu_bytes; mc->seg_up_bytes = mc->up_bytes;
        }
        {
            // DDR4 phases over the last 64 steps: CPU busy on uncached experts vs wall time, and uploads colliding with it
            const int64_t now = ggml_time_us();
            const int64_t bu = mc->busy_us, bn = mc->busy_n;
            const double  st = (double) std::max<uint64_t>(1, mc->n_steps - mc->seg_steps2);
            const uint64_t up = mc->n_uploads - mc->seg_up2, ov = mc->up_overlap - mc->seg_overlap;
            if (mc->seg_t0) {
                LLAMA_LOG_WARN("moe-cache: phases: token %.1f ms wall, CPU expert matmuls %.1f ms in %.1f phases (%.0f%%) | %.1f uploads/token, %.0f%% overlapping CPU phases\n",
                        (now - mc->seg_t0)/st/1e3, (bu - mc->seg_busy_us)/st/1e3, (bn - mc->seg_busy_n)/st,
                        100.0*(bu - mc->seg_busy_us)/std::max<int64_t>(1, now - mc->seg_t0), up/st, up ? 100.0*ov/up : 0.0);
            }
            mc->seg_t0 = now; mc->seg_busy_us = bu; mc->seg_busy_n = bn; mc->seg_steps2 = mc->n_steps; mc->seg_up2 = mc->n_uploads; mc->seg_overlap = mc->up_overlap;
        }
    }

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

void llama_moe_set_options(const char * opts) {
    // "NAME=value,NAME=value": tuner knobs (MARGIN, GATE, WAIT, BIG, ...); a name given here is never tuned
    knobs(); // environment first
    std::string s = opts ? opts : "";
    for (size_t pos = 0; pos < s.size();) {
        const size_t comma = s.find(',', pos);
        const std::string kv = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? s.size() : comma + 1;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) { continue; }
        std::string name = kv.substr(0, eq);
        name = knob_canon(name);
        if (name == "AUTOTUNE") {
            g_autotune_off = atof(kv.c_str() + eq + 1) == 0;
        } else if (knob_set(knobs(), name, atof(kv.c_str() + eq + 1))) {
            user_knobs().insert(name);
        } else {
            // every other setting is read by the engine by its --moe key (policy, predict-ahead, ...)
            llama_moe_set_opt(knob_key(name).c_str(), kv.substr(eq + 1).c_str());
        }
    }
}

bool llama_moe_cache_active_fork() {
    return g_cache != nullptr;
}

bool llama_moe_cache_wants_probs_fork() {
    return get_policy() == policy::rel;
}

bool llama_moe_cache_get_info_fork(struct llama_tuning_info * info) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return false;
    }
    info->moe_hits = info->moe_misses = 0;
    for (auto & ls : mc->layers) { info->moe_hits += ls.n_hit; info->moe_misses += ls.n_miss; }
    info->moe_uploads = mc->n_uploads;
    for (int i = 0; i < 16; ++i) { info->moe_pr[i] = mc->pr[i]; }
    info->moe_evictions = mc->n_evictions; info->moe_up_bytes = (uint64_t) std::max<int64_t>(0, (int64_t) mc->up_bytes);
    info->moe_layers  = (int32_t) mc->layers.size();
    info->moe_slots_min = INT32_MAX; info->moe_slots_max = 0;
    for (auto & ls : mc->layers) {
        info->moe_slots_min = std::min(info->moe_slots_min, ls.pcls >= 0 ? ls.n_own : ls.pub.n_slots);
        info->moe_slots_max = std::max(info->moe_slots_max, ls.pcls >= 0 ? ls.n_own : ls.pub.n_slots);
    }
    info->margin = mc->last_margin; // the value in use (the tuner's pick, a fixed MARGIN, or the timing rule's)
    info->gate = (int32_t) knobs().gate; info->wait = (int32_t) knobs().wait; info->big = (int32_t) knobs().big;
    info->predict = (int32_t) knobs().predict; info->self_tune = (int32_t) knobs().self_tune;
    {
        const auto kt = llama_moe_cache_kv_tail_stats_fork(false);
        info->kvt_n = (int32_t) std::min<size_t>(kt.size(), 4);
        for (int i = 0; i < info->kvt_n; ++i) {
            snprintf(info->kvt_dev[i], sizeof info->kvt_dev[i], "%s", kt[i].dev.c_str());
            info->kvt_cells[i] = kt[i].kv_cells; info->kvt_cap[i] = kt[i].cap_tokens; info->kvt_kv_mib[i] = kt[i].kv_bytes / 1048576.0; info->kvt_moe_gib[i] = kt[i].moe_bytes / 1073741824.0;
            info->kvt_blocks[i] = kt[i].blocks; info->kvt_free[i] = kt[i].blocks_free; info->kvt_lent[i] = kt[i].blocks_lent; info->kvt_kv_blocks[i] = kt[i].blocks_kv;
        }
        if (!kt.empty()) { info->kvt_cap_min = info->kvt_cap[0]; for (int i = 1; i < info->kvt_n; ++i) { info->kvt_cap_min = std::min(info->kvt_cap_min, info->kvt_cap[i]); } }
        if (!kt.empty()) { info->kvt_claim_n = kt[0].claim_n; info->kvt_release_n = kt[0].release_n; info->kvt_evicted = kt[0].evicted; }
    }
    return true;
}

// max-batch: the largest batch (tokens) that still goes through the cache path. Larger batches take the prompt path, which uploads the experts they use.
//
// Why a threshold exists and where it should sit (measured GLM-5.3-Flash 3.0-bit, 2x RTX 3090, x16 + x4, DDR4-3200 ~44 GB/s; fitted 2026-10-11):
//   cache path   time ~ n * c      c = CPU time per token for the experts the cache misses, close to the decode step (about 40-50 ms)
//   prompt path  time ~ a + n * b  a = upload of the used experts per pass (3.5-4 s here, saturates once most experts are used), b = GPU compute per token
//                                  (about 1.7 ms: prefill of 1k/2k/4k-token ubatches took 5.8/7.5/11.1 s)
//   crossover    n* = a / (c - b)  about 90 tokens on this machine; 120 tokens measured 32 t/s streaming against 25 t/s through the cache.
// The crossover scales with the ratio of RAM bandwidth to PCIe bandwidth, because c is bound by the CPU reading the missed experts from RAM and a by the
// link: a box with 100 GB/s RAM on PCIe 3.0 (ratio about 8) has its optimum at several hundred tokens (a fork user measured 512 best), ours (ratio 2-3.5)
// at about 90, a laptop on a slow link and fast RAM higher still, a 4x16 GPU server with slow RAM lower. The decode batches (1 token) and the speculative
// verify batches (a few tokens) must stay on the cache path, so never go below about 8.
// Default 31 is conservative for this machine; deriving it from the probes (link GB/s, CPU GB/s, measured decode step and layer times) is planned.
// --moe max-batch: unset or auto = the limit is computed at the first decode (crossover below; 31 until then, and 31 if the probe fails); a number fixes it.
// The graph node budget is sized for the cap, so the computed value may move between 8 and the cap without a rebuild.
static std::atomic<int64_t> g_mb_auto { 31 };      // the computed limit
static std::atomic<int64_t> g_mb_probe { -1 };     // >= 0 while the start-up probe forces the cache path for up to this many tokens
static bool mb_is_auto() {
    static const bool a = [] { const char * e = moe_opt("max-batch"); return !e || strcmp(e, "auto") == 0; }();
    return a;
}
static int64_t mb_fixed() {
    static const int64_t v = [] { const char * e = moe_opt("max-batch"); return (e && strcmp(e, "auto") != 0) ? (int64_t) atoi(e) : (int64_t) -1; }();
    return v;
}

int64_t llama_moe_cache_max_batch_cap_fork() {
    return std::max<int64_t>(128, mb_fixed());   // graph node budget at reserve time
}

int64_t llama_moe_cache_max_batch_fork() {
    const int64_t p = g_mb_probe.load(std::memory_order_relaxed);
    if (p >= 0) {
        return p;
    }
    if (mb_fixed() >= 0) {
        return mb_fixed();
    }
    return g_mb_auto.load(std::memory_order_relaxed);
}

// The streaming path costs a(n) + n*b: a(n) uploads every distinct expert the batch touches, over the offload GPU's link (prompt batches are
// offloaded to the fastest card): layers x distinct experts u(n) x expert bytes / link speed, with u(n) = N (1 - (1 - k/N)^n) for N experts per
// layer and k used per token; b is the small GPU compute per token. The cache path costs c per token (measured). Crossover n*: n c = a(n) + n b.
// Check, GLM-5.3-Flash 3.0-bit on 2x RTX 3090 (link 25.5 GB/s): a(120) = 3.8 s formula vs 3.7 s measured; qwen3.8 cut-down 0.43 vs 0.47 s.
// Measured, not modelled: the probe times batches of several sizes on both paths (the cache keeps part of the experts, so the streamed pass uploads less than
// a formula over all touched experts says). Each path is a piecewise-linear curve through its points, extended past the last point with the last slope;
// the switch is the smallest n where streaming is not slower than the cache path (cap when it never is within the cap).
struct mb_pts { std::vector<double> n, us; };
// batch time cannot fall as the batch grows: pool adjacent violators (average neighbours that break the order) smooths timing noise
static void mb_monotone(mb_pts & p) {
    std::vector<double> v, w;
    for (double u : p.us) {
        v.push_back(u); w.push_back(1);
        while (v.size() > 1 && v[v.size() - 2] > v.back()) {
            const double t = w[w.size() - 2] + w.back();
            v[v.size() - 2] = (v[v.size() - 2] * w[w.size() - 2] + v.back() * w.back()) / t;
            w[w.size() - 2] = t;
            v.pop_back(); w.pop_back();
        }
    }
    size_t k = 0;
    for (size_t i = 0; i < v.size(); ++i) { for (int j = 0; j < (int) w[i]; ++j) { p.us[k++] = v[i]; } }
}
// below the first point the cache path scales with the tokens (through the origin), a streamed pass stays at its first point (the upload does not shrink)
static double mb_eval(const mb_pts & p, double n, bool flat_below) {
    const size_t m = p.n.size();
    if (n <= p.n[0]) {
        return flat_below ? p.us[0] : p.us[0] * n / p.n[0];
    }
    size_t i = 1;
    while (i + 1 < m && n > p.n[i]) {
        ++i;
    }
    const double slope = (p.us[i] - p.us[i - 1]) / (p.n[i] - p.n[i - 1]);
    return std::max(0.0, p.us[i - 1] + slope * (n - p.n[i - 1]));
}
static int64_t crossover(const mb_pts & cache, const mb_pts & stream) {
    const int64_t cap = llama_moe_cache_max_batch_cap_fork();
    if (cache.n.empty() || stream.n.empty()) {
        return 31;
    }
    for (int64_t n = 8; n <= cap; ++n) {
        if (mb_eval(stream, (double) n, true) <= mb_eval(cache, (double) n, false)) {
            return n;
        }
    }
    return cap;
}
static std::string mb_encode(int slots, const mb_pts & c, const mb_pts & s) {
    std::string r = std::to_string(slots) + " " + std::to_string(c.n.size());
    for (size_t i = 0; i < c.n.size(); ++i) { r += " " + std::to_string((long long) c.n[i]) + ":" + std::to_string((long long) c.us[i]); }
    r += " " + std::to_string(s.n.size());
    for (size_t i = 0; i < s.n.size(); ++i) { r += " " + std::to_string((long long) s.n[i]) + ":" + std::to_string((long long) s.us[i]); }
    return r;
}
static bool mb_decode(const std::string & v, int slots, mb_pts & c, mb_pts & s) {
    std::istringstream is(v);
    int sl = 0;
    size_t cnt = 0;
    if (!(is >> sl >> cnt) || sl != slots || cnt < 2 || cnt > 16) {
        return false;
    }
    auto get = [&](mb_pts & p, size_t m) {
        for (size_t i = 0; i < m; ++i) {
            std::string t; double n, u;
            if (!(is >> t) || sscanf(t.c_str(), "%lf:%lf", &n, &u) != 2 || n <= 0 || u <= 0) { return false; }
            p.n.push_back(n); p.us.push_back(u);
        }
        return true;
    };
    size_t ns = 0;
    if (!get(c, cnt) || !(is >> ns) || ns < 2 || ns > 16 || !get(s, ns)) {
        return false;
    }
    return true;
}

bool llama_moe_cache_probe_wanted_fork() {
    moe_cache * mc = g_cache;
    if (!mc || !mb_is_auto()) {
        return false;
    }
    std::string v;
    mb_pts c, st;
    if (!mc->profile.empty() && moe_state_get(mc->profile, hw_key("batch.cost2"), v) && mb_decode(v, mc->n_slots, c, st)) {   // stored by an earlier start: no probe, just compute
        mb_monotone(c); mb_monotone(st);
        g_mb_auto = crossover(c, st);
        LLAMA_LOG_WARN("moe-cache: max-batch auto = %lld (stored probe)\n", (long long) g_mb_auto.load());
        return false;
    }
    return true;
}

void llama_moe_cache_probe_begin_fork() {
    g_mb_probe = llama_moe_cache_max_batch_cap_fork();
}

void llama_moe_cache_probe_stream_fork(bool on) {
    g_mb_probe = on ? 0 : llama_moe_cache_max_batch_cap_fork();
}

void llama_moe_cache_probe_end_fork(const double * cn, const double * cus, int nc, const double * sn, const double * sus, int ns) {
    g_mb_probe = -1;
    moe_cache * mc = g_cache;
    mb_pts c, st;
    for (int i = 0; i < nc; ++i) { if (cus[i] > 0) { c.n.push_back(cn[i]); c.us.push_back(cus[i]); } }
    for (int i = 0; i < ns; ++i) { if (sus[i] > 0) { st.n.push_back(sn[i]); st.us.push_back(sus[i]); } }
    if (!mc || c.n.size() < 2 || st.n.size() < 2) {
        return;   // probe failed: the limit stays at 31
    }
    mb_monotone(c); mb_monotone(st);
    g_mb_auto = crossover(c, st);
    std::string msg;
    for (size_t i = 0; i < c.n.size(); ++i) { msg += " c" + std::to_string((int) c.n[i]) + "=" + std::to_string((int) (c.us[i] / 1e3)) + "ms"; }
    for (size_t i = 0; i < st.n.size(); ++i) { msg += " s" + std::to_string((int) st.n[i]) + "=" + std::to_string((int) (st.us[i] / 1e3)) + "ms"; }
    LLAMA_LOG_WARN("moe-cache: max-batch auto = %lld (timed batches, cache path c / streamed s:%s)\n", (long long) g_mb_auto.load(), msg.c_str());
    if (!mc->profile.empty()) {
        moe_state_set(mc->profile, { { hw_key("batch.cost2"), mb_encode((int) mc->n_slots, c, st) } });
    }
}

// ---- prompt pass planner ------------------------------------------------------------------------------------------------------------------------------------
// A prompt runs in passes; every pass walks all layers and streams every expert the card does not hold (Qwen3.8-Flash-Next: 34 GB of 49 per pass on the
// x16 card, 1.3 s; MiMo-V2.6-Flash 3-bit: about 90 GB, 3.5 s). The staging slots hide that upload under the compute, but they take pool memory next to the
// graph, and a big graph leaves room for fewer of them, or none. So the number of passes is a trade, decided per prompt from what is known:
//   U      the upload of one pass on the offload card: the expert bytes of every layer minus what this card's pool holds of them, over its link (measured)
//   C(u)   the compute of a pass of u tokens: c x u, c measured from earlier prompt passes on this card (unknown at first: then fewest passes)
//   k(u)   staging slots that fit next to a graph of u tokens: (what the class lends - graph and scratch bytes per token x u) / slot size, at most the
//          configured number; fewer than 3 = no prefetch
//   pass   k >= 6: max(U, C) (uploads hidden); 3..5: max(U, C) + a quarter of the smaller (the next layer's copies wait for this layer's matmuls);
//          no slots: U + C
//   prompt P passes of ceil(n / P) tokens: P x pass; P from the fewest (n / ub_max) up to four more, the cheapest wins.
// Measured on MiMo, 4k prompt, one x16: 8 passes with 6 slots 127 t/s, 2 passes with 2 slots 471, one pass without slots 544: one pass wins when the upload
// dominates; Qwen3.8-Flash-Next at 8k (compute 6.4 s against 1.3 s of upload) wins with two staged passes. --moe pass-plan=0: equal passes as before.
static std::mutex g_pp_mu;
static double     g_pp_c_us  = 0.0;   // compute of one prompt token on the offload card in a pass (us), 0 = not measured yet
static void pp_c_us_stored();   // an earlier start's value until this start measured its own (below)
static uint32_t   g_pp_n     = 0;     // the last planned prompt: its tokens, passes, pass size and upload estimate (to read its measurement)
static uint32_t   g_pp_P     = 0;
static uint32_t   g_pp_ub    = 0;
static double     g_pp_U_us  = 0.0;
// The compute of a prompt token grows with the cells its pass attends to (MiMo 3-bit, 255k-token prompt, x16 + x4: 1.4 ms per token at 32k cells, 4.6 ms at 230k;
// the attention over the whole context). One number per model cannot plan both ends, so the measurement is fitted as c(cells) = c0 + a x cells:
//   g_pp_c_us  c0, the compute per token with no context
//   g_pp_a_us  a, per token and cell (0 until two measurements far enough apart in cells say otherwise)
// The server sends a long prompt as several decode calls (n_batch tokens each) and synchronizes only after the last one, so the context's report covers all of them:
// every planned call adds its tokens, passes, upload estimate and token-weighted cells to the totals below, and the report reads the totals (before 2026-10-17 it
// compared the report with the last call alone, and prompts longer than one call never taught it anything: a stale 8 us per token planned a 255k prompt).
static double     g_pp_a_us  = 0.0;
static double     g_pp_Usum  = 0.0;   // sum over the planned passes of their upload estimate (us)
static double     g_pp_xc    = 0.0;   // sum over the planned tokens of the cells each one's pass attends to (about the cells at the middle of its call)
static uint64_t   g_pp_cells = 0;     // the cells at the end of the call being planned (set by the context, llama_moe_cache_pp_cells_fork)
static double     g_fit[5]   = {0, 0, 0, 0, 0};   // decayed sums for the fit: weight, x, y, x*x, x*y (x = cells, y = us per token)
void llama_moe_cache_pp_cells_fork(uint64_t cells_end) { std::lock_guard<std::mutex> lk(g_pp_mu); g_pp_cells = cells_end; }
// the compute per token at the cells of the call being planned (under g_pp_mu)
static double pp_c_at() { return g_pp_c_us > 0.0 ? g_pp_c_us + g_pp_a_us * (double) g_pp_cells : 0.0; }
// one planned call (under g_pp_mu): added to what the next report measures
static void pp_add(uint32_t n, uint32_t ub, double U, int k) {
    const uint32_t P = (n + std::max<uint32_t>(1, ub) - 1) / std::max<uint32_t>(1, ub);
    g_pp_n += n; g_pp_P += P; g_pp_ub = std::min(ub, n); g_pp_U_us = U; g_pp_k = k;
    g_pp_Usum += (double) P * U;
    g_pp_xc   += (double) n * std::max(0.0, (double) g_pp_cells - 0.5 * (double) n);
}

// The experts a prompt pass streams to `dev`: every layer's expert bytes (E), and how many of those bytes this card's pool holds now (R), ready, not in flight.
static void pp_expert_bytes(moe_cache * mc, ggml_backend_dev_t dev, double & E, double & R) {
    E = 0.0; R = 0.0;
    std::lock_guard<std::mutex> lock(mc->mtx);
    for (const auto & l : mc->layers) {
        if (!l.pub.up_src || !l.pub.gate_src || !l.pub.down_src) {
            continue;
        }
        const double  lb    = (double) ggml_nbytes(l.pub.up_src) + (double) ggml_nbytes(l.pub.gate_src) + (double) ggml_nbytes(l.pub.down_src);
        const int64_t n_exp = std::max<int64_t>(1, l.pub.up_src->ne[2]);
        int64_t held = 0;
        const ggml_tensor * c = l.pub.up_c;
        if (c && c->buffer && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) == dev) {
            for (size_t e = 0; e < l.expert_slot.size(); ++e) {
                const int32_t sl = l.expert_slot[e];
                if (sl >= 0 && !((size_t) sl < l.slot_in_flight.size() && l.slot_in_flight[sl])) { held++; }
            }
        }
        E += lb;
        R += lb * (double) std::min<int64_t>(held, n_exp) / (double) n_exp;
    }
}

// The upload of a prompt pass (ms): the expert bytes it touches - every layer's experts x the touched fraction 1 - (1 - k/E)^n of a pass of n tokens with k
// of E experts per token - less what the expert card's pool holds, over that card's measured link rate. A pass takes at least this long (the uploads overlap
// the compute, they do not shorten); the dense split leaves it as it is (the experts stay on the expert card) and only shortens the compute, so a split pays
// only when a pass's compute is longer than this (user 2026-10-25: "is the time it takes to upload the same as computing 7k tokens?"). -1: not known.
double llama_moe_cache_pass_upload_ms_fork(uint32_t n_tokens, uint32_t n_used, uint32_t n_expert) {
    moe_cache * mc = g_cache;
    if (!mc || mc->n_links <= 0 || !mc->link_dev[0] || n_expert == 0 || n_tokens == 0) { return -1.0; }
    double E = 0.0, R = 0.0;
    pp_expert_bytes(mc, mc->link_dev[0], E, R);
    const double gbs = llama_moe_cache_link_gbs_fork(mc->link_dev[0]);
    if (gbs <= 0.0 || E <= 0.0) { return -1.0; }
    const double frac = 1.0 - std::pow(1.0 - std::min(1.0, (double) n_used / (double) n_expert), (double) n_tokens);
    return frac * std::max(0.0, E - R) / (gbs * 1e6);
}


// what a prompt sized without the planner is measured against: its passes and the upload of one (the compute per token is read from every prompt pass on
// one card, also the first ones, which the split planner needs before it can say anything)
static uint32_t pp_record(moe_cache * mc, uint32_t n, uint32_t ub) {
    double U = 0.0;
    if (mc->n_links > 0 && mc->link_dev[0]) {
        const double gbs = llama_moe_cache_link_gbs_fork(mc->link_dev[0]);
        double E = 0.0, R = 0.0;
        pp_expert_bytes(mc, mc->link_dev[0], E, R);
        U = gbs > 0.0 ? std::max(0.0, E - R) / (gbs * 1e3) : 0.0;
    }
    std::lock_guard<std::mutex> lk(g_pp_mu);
    pp_add(n, ub, U, -1);
    return ub;
}

// A pass whose graph does not fit the pool is a crash (kv-tail: "the compute graph and the KV do not fit in the pool", seen on an 8 GiB card with a 12k prompt planned
// as ONE pass of 13720 tokens: 5.1 GiB of graph for a 4.6 GiB pool). The lend of a graph (moe_alias) must satisfy exactly
//     poolx_lend_round(mirror_b + need_b) + kv_fixed + max(KV rows in use, KV horizon) * kv_row <= tail_end   (the lend ends on the block edge of its class)
// and need_b grows linearly with the tokens of the pass (graph tensors, CUDA scratch): a + b * tokens, taken from the two largest graph sizes the pool has lent for so far
// (a single size: b = need / tokens, a = 0, which is not below the real slope). The largest pass is the biggest u with that inequality, no margin; a graph that still
// misses (the staging slots are variable) meets the same check in moe_alias.
static uint32_t pp_cap_to_pool(const moe_cache::alias_pool & ap, uint32_t n_prompt, uint32_t ub) {
    if (ap.kvt.empty()) { return ub; }
    size_t pad = 1;
    for (const auto & r : ap.kvt) { if (!r.fixed) { pad = std::max<size_t>(pad, r.pad); } }
    // No prompt-sized graph of this run lent yet (a start that took the max-batch probe from the state file runs no calibration graphs before its first
    // prompt): the planner's bytes per pass token, which the startup reserve seeds (graph_tok + scr_tok), still keeps that first pass and the zone for
    // the prompt's KV inside the pool. Without it the first prompt of a warm start was planned with no room for its KV (Qwen3.6-35B, 3 GiB free per card:
    // a 4579-token pass left no gap for the zone, the server retried with halved batches: 1860 prompt t/s against 3706 on the cold start, 2026-10-22).
    // The graph bytes per token leave out the staging slots (a fixed part), added when known. No headroom on top (user 2026-10-22: no cap may cost a pass
    // where fewer were possible; GLM 12k ran 5 passes instead of 3 with 1.35x): a pass the estimate lets past the KV zone's room fails its call and the
    // server retries it smaller, which only small cards (3 GiB free per card) met.
    auto seeded = [&]() -> uint32_t {   // the graph alone: the staging slots take what the pool holds behind it (ppx_stage_slots), they never cut a pass
        return ap.graph_tok > 0 ? kvz_pass_cap(ap, n_prompt, ub, 0.0, (double) (ap.graph_tok + ap.scr_tok), pad, ap.mirror_b) : ub;
    };
    if (ap.need_tok.empty()) { return seeded(); }
    // every measured size at the cells this call ends at (need_at_cells): the fit below then includes the masks' growth with the context
    size_t cells_end = (size_t) n_prompt;
    for (auto & r : ap.kvt) { if (!r.fixed) { cells_end = std::max(cells_end, (r.rows ? r.rows() : 0) + (size_t) n_prompt); } }
    std::vector<std::pair<int32_t, size_t>> pts;
    for (const auto & kv : ap.need_tok) { if (kv.first >= 64) { pts.push_back({ kv.first, need_at_cells(ap, kv.first, kv.second, cells_end) }); } }
    if (pts.empty()) { return seeded(); }
    // graph bytes = a + b x tokens, the least-squares line through every measured size (at the cells this call ends at). Not the line through the two largest
    // sizes: two passes of nearly the same size (GLM 3.0: 8995 and 8973 tokens) gave a slope of noise, negative, and the fallback (all bytes per token, a = 0)
    // made the graph 1.44 MiB per token where the passes measured 1.5 GiB + 0.33 MiB per token. One size only, or no rising line: bytes per token of the
    // largest graph (a = 0), which is not below the real slope.
    double a = 0.0, b = 0.0;
    if (pts.size() >= 2) {
        double mx = 0.0, my = 0.0;
        for (const auto & p : pts) { mx += p.first; my += (double) p.second; }
        mx /= (double) pts.size(); my /= (double) pts.size();
        double sxx = 0.0, sxy = 0.0;
        for (const auto & p : pts) { sxx += (p.first - mx) * (p.first - mx); sxy += (p.first - mx) * ((double) p.second - my); }
        if (sxx > 0.0 && sxy > 0.0) { b = sxy / sxx; a = std::max(0.0, my - b * mx); }
    }
    if (b <= 0.0) {
        for (const auto & p : pts) { b = std::max(b, (double) p.second / (double) p.first); }
        a = 0.0;
    }
    // A graph's fixed part is never negative (a >= 0), so the slope can never exceed any measured graph's bytes per token: clamp it there and refit a.
    // Two passes of nearly the same size (GLM 3.0, 40k prompt: the 6552- and 6554-token passes, 2 tokens apart) gave a slope of noise, 124.8 MB per
    // token where the graphs measured 1.4 MB per token: the next call's passes were capped to 96 tokens (60k prompt: 239 passes, the cell ran 20+ min).
    if (b > 0.0) {
        double bmax = 0.0, mx = 0.0, my = 0.0;
        for (const auto & p : pts) { const double r = (double) p.second / (double) p.first; bmax = bmax > 0.0 ? std::min(bmax, r) : r; mx += p.first; my += (double) p.second; }
        if (b > bmax) { b = bmax; a = std::max(0.0, my / (double) pts.size() - b * mx / (double) pts.size()); }
    }
    if (b <= 0.0) { return ub; }
    // Past the largest measured size the line is an extrapolation from small graphs (a cold start measures 64-512-token calibration graphs): the same
    // headroom there (Qwen3.6-35B, 3 GiB free per card, cold: a 4128-token pass lent 1222 of the x4 card's 1282 MiB and its KV did not fit, 2026-10-22)
    if (moe_log_has("w")) {   // the cap's inputs: a 22381-token call of GLM (split-pass=0, 55k cells) was capped to 96 tokens (239 passes)
        LLAMA_LOG_WARN("moe-cache: pass cap: n %u cells_end %zu pts %zu a %.0f b %.0f mirror %zu lend_potential %zu -> %u\n", n_prompt, cells_end, pts.size(), a, b, ap.mirror_b,
                (size_t) kvz_lend_potential(ap, (std::max<size_t>(n_prompt, ap.kv_h) + pad - 1) / pad * pad), kvz_pass_cap(ap, n_prompt, ub, a, b, pad, ap.mirror_b));
        for (const auto & p : pts) { LLAMA_LOG_WARN("moe-cache: pass cap point: %d tokens -> %zu B\n", p.first, p.second); }
    }
    return kvz_pass_cap(ap, n_prompt, ub, a, b, pad, ap.mirror_b);   // the zone for the padded prompt rows lies in a gap: the lend can reach only what kvz_lend_potential says (OLMoE 3440-token prompt, 1.0 GiB free: planned from the pool end, the zone then held 2962 cells behind a 451-token pass: no slot, 2026-10-18)
}

// The dense split's pass loop (llama_context::decode, --moe split-pass) sizes each pass itself and runs it as one sub-call, so the planner's pool cap (pp_plan)
// never cut it: a 31k-token pass at 68k cells asked CUDA1's pool for more than the graph and the KV left (2026-10-24), the call failed and the server retried it
// in halves (passes of 3-19k tokens). This is that cap for one pass: the largest pass, up to ub, that every card's pool can lend its graph for with the KV
// zone of `cells` cells (the cells at the end of the pass) behind it.
uint32_t llama_moe_cache_pass_fit_fork(uint32_t cells, uint32_t ub) {
    moe_cache * mc = g_cache;
    if (!mc || mc->alias.empty()) { return ub; }
    for (const auto & a : mc->alias) { ub = pp_cap_to_pool(a, cells, ub); }
    return ub;
}

static uint32_t pp_plan(moe_cache * mc, uint32_t n, uint32_t ub_max, uint32_t ub_old) {
    static const bool on = [] { const char * e = moe_opt("pass-plan"); return !e || atoi(e) != 0; }();
    const moe_cache::alias_pool * ap = nullptr;
    if (!mc->alias.empty() && mc->link_dev[0]) {
        for (const auto & a : mc->alias) { if (a.dev == mc->link_dev[0]) { ap = &a; } }
    }
    if (ap) {   // bounded whatever the plan below does (also before the first prompt graph is known; see pp_cap_to_pool)
        const uint32_t ub_old0 = ub_old, ub_max0 = ub_max;
        // EVERY card's pool lends its part of the pass and holds the KV of its layers: the smallest cap wins. With the offload card's pool alone the x4 card's
        // KV zone was never checked (Qwen3.6-35B, 3 GiB free per card: 1222 of its 1282 MiB lent to a 4352-token pass, its KV did not fit, 2026-10-22)
        for (const auto & a : mc->alias) {
            ub_old = pp_cap_to_pool(a, n, ub_old);
            ub_max = pp_cap_to_pool(a, n, ub_max);
        }
        // the cap cuts the equal passes of the call: split the call into equal passes again under it. Uncut, a 9580-token call capped at 4160 ran as
        // 4160 + 4160 + 1260 (the last pass streamed every expert for 1260 tokens): 3700 prompt t/s against 4020 for b13600's equal passes on the same
        // small card (Qwen3.6-35B, 3 GiB free per card, 2026-10-22); as 3 x 3200 the passes are equal and leave the staging slots their room
        if (ub_old > 0 && ub_old < n) {
            const uint32_t passes = (n + ub_old - 1) / ub_old;
            ub_old = std::min<uint32_t>(ub_old, ((n + passes - 1) / passes + 31) / 32 * 32);
        }
        if (moe_log_has("w")) { LLAMA_LOG_WARN("moe-cache: pass plan input: n %u ub_max %u -> %u, ub_old %u -> %u, span %zu, graph_tok %zu, scr_tok %zu, slot_b %zu, mirror_b %zu, need_tok %zu\n", n, ub_max0, ub_max, ub_old0, ub_old, (size_t) ap->span, (size_t) ap->graph_tok, (size_t) ap->scr_tok, (size_t) ap->slot_b, (size_t) ap->mirror_b, ap->need_tok.size()); }
    }
    if (!on || mc->alias.empty() || mc->n_links <= 0 || !mc->link_dev[0]) {
        return pp_record(mc, n, ub_old);
    }
    if (!ap || ap->slot_b == 0 || ap->graph_tok == 0 || ap->span == 0) {
        return pp_record(mc, n, ub_old);   // no prompt graph seen yet: slot and graph sizes unknown
    }
    const double gbs = llama_moe_cache_link_gbs_fork(mc->link_dev[0]);
    if (gbs <= 0.0) {
        return pp_record(mc, n, ub_old);
    }
    // with the KV zone (kv-tail) ap->span is only the lend's reach so far (the zone's start), which a bigger graph moves: the plan must see what the lend can reach
    // (the pool up to the experts' end, less the zone), else the first prompts are planned into the small span they happen to start with (GLM: 2048-token passes, 0 slots)
    const double span_eff = (double) std::max(ap->span, kv_lend_potential(*ap));
    double E = 0.0, R_now = 0.0;
    pp_expert_bytes(mc, mc->link_dev[0], E, R_now);
    // what the pool holds per byte that is not lent now: lending L bytes leaves rho x (span - L) of the experts in place (a lent block's experts are evicted)
    const double free_now = span_eff - (double) std::min(ap->want_b, ap->span);
    const double rho      = free_now > 0.0 ? std::min(1.0, R_now / free_now) : 0.0;
    double c_us;
    {
        std::lock_guard<std::mutex> lk(g_pp_mu);
        pp_c_us_stored();
        c_us = pp_c_at();   // for the log line only: nothing below decides on it
    }
    const double per_t = (double) (ap->graph_tok + ap->scr_tok);
    // The fewest passes whose graph fits the pool, never more (user rules 2026-10-23: no more passes than b13838; 2026-10-24: the micro-batch first, the staging
    // slots take what the graph leaves, ppx_stage_slots). Every pass streams the experts again, so a further pass costs a whole upload. A pass of u tokens lends
    // mirror_b (the dense split's copies, counted for every pass: the plan is made before the split choice) + per_t x u. There is no time model any more: it chose
    // between fewer slots and smaller passes (guessed overlap 0.25, compute 0.5 x upload when unmeasured, 6 slots for a full overlap, 2% to change), and with the
    // slots never cutting the pass there is nothing left for one to decide.
    const uint32_t P0 = std::max<uint32_t>(1, (n + ub_max - 1) / ub_max);
    uint32_t best_ub = ub_old, best_P = 0;
    double   best_U  = 0.0;
    for (uint32_t P = P0; P <= n && best_P == 0; ++P) {
        const uint32_t u = std::min<uint32_t>(ub_max, ((n + P - 1) / P + 31) / 32 * 32);
        const double   L = per_t * (double) u + (double) ap->mirror_b;
        if (L > span_eff) { continue; }
        best_P  = P;
        best_ub = u;
        best_U  = std::max(0.0, E - rho * (span_eff - L)) / (gbs * 1e3);   // the upload of the experts the pool does not hold after this lend (log)
    }
    if (best_P == 0) {
        return pp_record(mc, n, ub_old);
    }
    const int best_k = -1;   // the slots: whatever the pool holds behind the graph (llama_moe_cache_alias_fork, ppx_stage_slots)
    const double best_t = 0.0;
    static const bool dbg = moe_log_has("w");   // --moe log=w: the pass planner's choice per prompt
    if (dbg) {
        LLAMA_LOG_WARN("moe-cache: pass plan: %u tokens: %u passes of %u, staging slots: what the pool holds behind the graph, upload %.0f ms per pass (the pool holds %.1f of %.1f GB), compute %.1f us per token%s\n",
                n, (n + best_ub - 1) / best_ub, best_ub, best_U / 1e3, rho * (span_eff - (per_t * best_ub + (double) ap->mirror_b)) / 1e9, E / 1e9,
                c_us, c_us > 0 ? "" : " (not measured yet)");
        (void) best_t;
    }
    std::lock_guard<std::mutex> lk(g_pp_mu);
    pp_add(n, best_ub, best_U, best_k);
    return best_ub;
}

// The measured time of a planned prompt (all its passes, from the context's synchronize): the compute per token is read from it. With slots that hide the
// upload a pass takes max(U, C): above U it is the compute; at U it only says the compute is not more (an upper bound, kept when lower). Without slots C = t - U.
// A split pass (two cards) says nothing about one card: not used.
// The per-prompt decision line (llama_dsplit_report_fork, called just before the report below) reads the plan of the prompt it reports: its passes, pass size
// and staging slots. False when the last plan is not of this prompt (no alias, --moe pass-plan=0, or already read).
bool llama_moe_cache_pass_plan_fork(uint32_t n_tokens, uint32_t & passes, uint32_t & pass_tokens, int & slots) {
    if (n_tokens != g_pp_n || g_pp_P == 0 || g_pp_ub == 0) {
        return false;
    }
    passes      = g_pp_P;
    pass_tokens = g_pp_ub;
    slots       = g_pp_k.load();
    return true;
}

// the expert split of prompt batches over the cards' links, as decided at load (llama_moe_cache_prefill_decide_fork); *speedup: the transfer speedup it was judged by
bool llama_moe_cache_expert_split_fork(double * speedup) {
    if (speedup) { *speedup = g_split_speedup; }
    return g_split_enabled;
}


// kv-tail, at the end of a prompt (its preheat counters were just read): the hottest experts the prompt used that the cache does not hold replace the coldest
// cached experts the prompt did not use, but only where the prompt's expert scores higher (the cache's common score, which the prompt counts just raised). At most a twentieth of a class's blocks per prompt, background uploads (the link is idle after a prompt): the cache
// follows the request without churning it. The expert that leaves is evicted now (its table goes back to uncached at the next step's flush).
static std::vector<upload_job> preheat_swap(moe_cache * mc, const std::vector<uint8_t> & used, int n_exp) {   // mc->mtx held; the caller queues the jobs after releasing it (lock order: wmtx before mtx)
    bool tail = false;
    for (auto & ap : mc->alias) { tail = tail || !ap.kvt.empty(); }
    std::vector<upload_job> jobs;
    // After a prompt only the blocks the prompt borrowed and gave back (empty) are refilled (reheat_free_blocks); a cached expert is never evicted for the
    // prompt's own, which are mostly not the answer's. --moe pswap=1 brings the swap back.
    static const bool swap_on = [] { const char * e = moe_opt("pswap"); return e && atoi(e) != 0; }();
    if (!swap_on || !tail || mc->starved.load()) { return jobs; }
    for (size_t pc = 0; pc < mc->pool.size(); ++pc) {
        struct cand { double sc; size_t li; int32_t e; int32_t b; };
        std::vector<cand> hot, cold;
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            layer_state & ls = mc->layers[li];
            if (ls.pcls != (int32_t) pc || ls.pub.il < 0) { continue; }
            const uint8_t * u = used.data() + (size_t) ls.pub.il * n_exp;
            for (int32_t e = 0; e < (int32_t) ls.expert_slot.size() && e < n_exp; ++e) {
                const int32_t b = ls.expert_slot[e];
                if (b < 0) {
                    if (u[e] && !ls.queued[e] && ls.adopt_slot[e] < 0) { hot.push_back({ xscore(mc, ls, e), li, e, -1 }); }
                } else if (!u[e] && !ls.slot_in_flight[b] && !ls.is_stream[b]) {
                    cold.push_back({ xscore(mc, ls, e), li, e, b });
                }
            }
        }
        const size_t budget = std::max<size_t>(1, mc->pool[pc].owner.size() / 20);
        const size_t n = std::min({ hot.size(), cold.size(), budget });
        if (n == 0) { continue; }
        std::partial_sort(hot.begin(), hot.begin() + (ptrdiff_t) n, hot.end(), [](const cand & a, const cand & b) { return a.sc > b.sc; });
        std::partial_sort(cold.begin(), cold.begin() + (ptrdiff_t) n, cold.end(), [](const cand & a, const cand & b) { return a.sc < b.sc; });
        for (size_t i = 0; i < n; ++i) {
            if (hot[i].sc <= cold[i].sc) { break; }   // the question's experts are not always the answer's: a swap only when the prompt's expert scores higher than the cached one
            layer_state & lo = mc->layers[cold[i].li];
            evict_slot(mc, lo, cold[i].b);
            pool_take(mc, hot[i].li, cold[i].b);
            layer_state & ls = mc->layers[hot[i].li];
            ls.slot_in_flight[cold[i].b] = true;
            ls.queued[hot[i].e] = 1;
            upload_job j { hot[i].li, hot[i].e, cold[i].b };
            j.bg = true;
            j.t_queued = ggml_time_us();
            jobs.push_back(j);
        }
    }
    return jobs;
}

void llama_moe_cache_prefill_report_fork(uint32_t n_tokens, int64_t us) {
    moe_cache * mc = g_cache;
    if (mc && mc->pf_count_on) {
        // the prompt's expert use, counted on the cards: once per expert and prompt, as the host callback counted it (the cards are idle here: after synchronize)
        typedef bool (*obs_read_t)(int, int, uint8_t *);
        static obs_read_t rd = [] {
            ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
            return reg ? (obs_read_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_obs_prompt_read_fork") : (obs_read_t) nullptr;
        }();
        std::vector<uint8_t> used((size_t) mc->pf_count_layers * mc->pf_count_experts, 0);
        std::vector<upload_job> swaps;
        if (rd && rd(mc->pf_count_layers, mc->pf_count_experts, used.data())) {
            std::lock_guard<std::mutex> lock(mc->mtx);
            for (auto & ls : mc->layers) {
                if (ls.pub.il < 0 || ls.pub.il >= mc->pf_count_layers) {
                    continue;
                }
                const uint8_t * u = used.data() + (size_t) ls.pub.il * mc->pf_count_experts;
                for (int e = 0; e < mc->pf_count_experts && e < (int) ls.expert_count.size(); ++e) {
                    if (u[e]) {
                        ls.expert_count[e]++;
                        ls.topic_max = std::max(ls.topic_max, ++ls.topic_cnt[e]);
                        ls.glob_max = std::max(ls.glob_max, ++ls.glob_count[e]);
                    }
                }
            }
            swaps = preheat_swap(mc, used, mc->pf_count_experts);
        }
        if (!swaps.empty()) {
            {
                std::lock_guard<std::mutex> wlk(mc->wmtx);
                for (const auto & j : swaps) { mc->todo.push_back(j); }
                mc->n_reheat += swaps.size();
            }
            mc->wcv.notify_all();
            if (moe_log_has("k")) { LLAMA_LOG_WARN("moe-cache: kv-tail: after the prompt %zu cold experts replaced by the prompt's hottest missing ones\n", swaps.size()); }
        }
    }
    std::lock_guard<std::mutex> lk(g_pp_mu);
    // a split pass (two cards) says nothing about one card, and a report that is not of the planned calls (an unplanned prompt) neither: the totals start again
    if (!mc || us <= 0 || llama_dsplit_cur_fork() == 1 || n_tokens != g_pp_n || g_pp_P == 0 || g_pp_ub == 0) {
        g_pp_n = 0; g_pp_P = 0; g_pp_Usum = 0.0; g_pp_xc = 0.0;
        return;
    }
    int k = 0;
    for (const auto & a : mc->alias) { if (mc->n_links > 0 && a.dev == mc->link_dev[0]) { k = a.n_stage; } }
    // totals over the calls since the last report (pp_add): with staging slots a pass takes max(U, C), without them U + C
    const double Usum = g_pp_Usum, nt = (double) g_pp_n, x = g_pp_xc / std::max(1.0, nt);
    double c_new = -1.0;
    if (k >= 2) {   // slots hide uploads (2 is the least that overlaps, ppx_stage_slots)
        if ((double) us > 1.3 * Usum) {
            c_new = (double) us / nt;
        } else if (g_pp_c_us > 0.0) {
            c_new = std::min(g_pp_c_us + g_pp_a_us * x, (double) us / nt);
        }
    } else {
        c_new = std::max(0.0, (double) us - Usum) / nt;
    }
    if (c_new >= 0.0) {
        // the fit c = c0 + a x cells over the measurements, older ones decayed (x 0.7 each): the slope only from points at least 8k cells apart; a stays >= 0
        for (double & f : g_fit) { f *= 0.7; }
        g_fit[0] += 1.0; g_fit[1] += x; g_fit[2] += c_new; g_fit[3] += x * x; g_fit[4] += x * c_new;
        const double var = g_fit[0] * g_fit[3] - g_fit[1] * g_fit[1];   // n^2 x variance of the cells
        if (g_fit[0] > 1.2 && var > g_fit[0] * g_fit[0] * 4096.0 * 4096.0) {
            g_pp_a_us = std::max(0.0, (g_fit[0] * g_fit[4] - g_fit[1] * g_fit[2]) / var);
        }
        const double c0_new = std::max(0.0, c_new - g_pp_a_us * x);
        g_pp_c_us = g_pp_c_us > 0.0 ? 0.5 * (g_pp_c_us + c0_new) : c0_new;
        LLAMA_LOG_WARN("moe-cache: prompt compute measured: %.0f tokens in %.0f ms over %u passes (upload estimate %.0f ms) at %.0f cells: %.1f us per token -> c0 %.1f us + %.5f us per cell\n",
                nt, us / 1e3, g_pp_P, Usum / 1e3, x, c_new, g_pp_c_us, g_pp_a_us);
        // ini-fix (2026-10-22): not written to the state file any more (see pp_c_us_stored): a start plans from its own measurement only
    }
    g_pp_n = 0; g_pp_P = 0; g_pp_Usum = 0.0; g_pp_xc = 0.0;   // read once
}

// The compute of one prompt token measured by an earlier start on these cards (state file), until this start measures its own (under g_pp_mu). Without it the
// first prompt of every start had no plan (the split planner needs this number), ran on one card to measure it, and only the next prompts could split
// (Qwen3.8-Flash-Next 8.8k on x16 + x4: the first prompt 1201 prompt t/s on one card, the planner's split 1432 from the second, 1590 tuned).
// Keys pp_c0_us / pp_a1_us (2026-10-17): until then pp_c_us held the whole compute per token of the last prompt it could read, attention included (MiMo: 1265 us
// measured at about 250k cells); read as c0, the base of the fit, it would plan every short prompt as 20x heavier. Old entries stay in the file, unused.
static void pp_c_us_stored() {
    // ini-fix (2026-10-22): no stored value is read. The state held values from older builds (22591 us per token next to 1212 measured live) that the split
    // planner read as truth; the session measures its own (llama_moe_cache_prefill_report_fork) and nothing is written for the next start.
    return;
    static bool tried = false;
    if (tried || g_pp_c_us > 0.0 || !g_cache || g_cache->profile.empty() || moe_state_place().empty()) {
        return;
    }
    tried = true;
    std::string v;
    if (moe_state_get(g_cache->profile, moe_state_place() + ".pp_c0_us", v) && atof(v.c_str()) > 0.0) {
        g_pp_c_us = atof(v.c_str());
        if (moe_state_get(g_cache->profile, moe_state_place() + ".pp_a1_us", v)) { g_pp_a_us = std::max(0.0, atof(v.c_str())); }
        LLAMA_LOG_WARN("moe-cache: compute of one prompt token %.2f us (stored by an earlier start on these cards): the first prompt is planned from it\n", g_pp_c_us);
    }
}

// the pass planner's measured compute of one prompt token on the offload card, all layers, us (0 = not measured yet)
double llama_moe_cache_compute_us_fork() {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    pp_c_us_stored();
    return pp_c_at();
}
// the cells the call being planned ends at (0 = not set yet)
uint64_t llama_moe_cache_pp_cells_get_fork() { std::lock_guard<std::mutex> lk(g_pp_mu); return g_pp_cells; }
// the part of it that is attention over the context (a x cells): the split planner shares it by tokens like the rest of the dense part
double llama_moe_cache_compute_att_us_fork() {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    return g_pp_c_us > 0.0 ? g_pp_a_us * (double) g_pp_cells : 0.0;
}

uint32_t llama_moe_cache_prefill_ubatch_fork(uint32_t n, uint32_t ub_max) {
    if (g_cache && n > 0) {   // kv-tail: the prompt's KV is claimed before its passes are planned, so the planner sees what the pool can still lend
        const bool changed = kv_tail_plan(g_cache, n, false, true, 0, false);   // the zone shrinks to this request (a new request) or grows for its prompt
        for (auto & ap : g_cache->alias) { for (auto & sg : ap.segs) { sg.hwm = 0; } }   // the lend's reach is this request's (the micro-batch is fixed per request): placement follows it
        if (changed) { flush_tables(); }   // between requests nothing runs: the tables reach the GPU, no scheduler sync needed
    }
    static const bool on = [] { const char * e = moe_opt("prefill-equal"); return !e || atoi(e) != 0; }();
    const uint32_t mb = (uint32_t) std::max<int64_t>(1, llama_moe_cache_max_batch_fork());
    if (on && g_cache && n > mb && ub_max > mb) {
        // the pass planner (above); its fallback is the rule below
        uint32_t ub_old = ub_max;
        if (n > ub_max && n % ub_max > mb) {
            const uint32_t passes = (n + ub_max - 1) / ub_max;
            ub_old = (n + passes - 1) / passes;
        }
        return pp_plan(g_cache, n, ub_max, ub_old);
    }
    return on ? ppx_equal_pass(n, ub_max, mb) : ub_max;
}


// Blocks of the expert pools that hold an expert now, and all blocks (lent and free ones count in the total): how full the cache is when a prompt starts.
void llama_moe_cache_fill_fork(uint64_t * filled, uint64_t * total) {
    *filled = *total = 0;
    moe_cache * mc = g_cache;
    if (!mc) { return; }
    std::lock_guard<std::mutex> lk(mc->mtx);
    for (const auto & p : mc->pool) {
        *total += p.owner.size();
        for (int32_t o : p.owner) { *filled += o >= 0; }
    }
}

// What the prompt passes asked the cache for since the start (the per-pass log line takes the difference): experts asked, found on the card they are needed on (copied
// there, the upload skipped), and asked of the wrong card (the layer's cache lives on the other card: nothing to copy, the expert comes from RAM). Both lookups count:
// the staging fill of a pass (moe_where_cb, the live path: whole layers per pass) and the scheduler's per-expert query (moe_src_cb). False without a cache.
// the same per card (link k): its name and counts, false past the last card. The staging fill of a card asks for the experts of every layer its tokens route to; the share
// "wrong" belongs to layers whose cache is on the other card (the dense split runs every layer on every card, on different tokens)
bool llama_moe_cache_staging_card_fork(int k, const char ** name, int64_t * asked, int64_t * found, int64_t * wrong) {
    moe_cache * mc = g_cache;
    if (!mc || k < 0 || k >= mc->n_links || !mc->link_dev[k]) { return false; }
    *name = ggml_backend_dev_name(mc->link_dev[k]);
    *asked = mc->pf_ask[k]; *found = mc->pf_found[k]; *wrong = mc->pf_wrong[k];
    return true;
}

bool llama_moe_cache_staging_counts_fork(int64_t * asked, int64_t * found, int64_t * wrong) {
    moe_cache * mc = g_cache;
    if (!mc) { return false; }
    *asked = g_ac.where_range.load() + (int64_t) mc->n_src_query;
    *found = g_ac.where_found.load() + (int64_t) mc->n_src_hit;
    *wrong = g_ac.where_wrongdev.load();
    return true;
}

// GET /moe/stats, bandwidth part: RAM and each PCIe link in use (rate since the previous call, at least 0.5 s apart) against the probed maximum, bytes uploaded per link
std::string llama_moe_bw_json_fork() {
    moe_cache * mc = g_cache;
    if (!mc) { return "\"bandwidth\":null,\"hit\":null"; }
    static std::mutex mu; static int64_t t0 = 0, cpu0 = 0, up0 = 0, lk0[MAX_LINKS] = {};
    static double ram = 0, cpu = 0, lk[MAX_LINKS] = {}, win = 0, hr_recent = -1;
    static uint64_t h0 = 0, m0 = 0, sq0 = 0, sh0 = 0; static double pf_recent = -1;
    static int64_t pfb0 = 0, pfl0[MAX_LINKS] = {};
    std::lock_guard<std::mutex> g(mu);
    uint64_t hh = 0, mm = 0;
    for (const auto & l : mc->layers) { hh += l.n_hit; mm += l.n_miss; }
    const int64_t now = ggml_time_us();
    if (now - t0 >= 500000) {
        if (t0) {
            const double dt = (now - t0) / 1e6;
            cpu = (mc->cpu_bytes - cpu0) / dt / 1e9;
            ram = cpu + (mc->up_bytes - up0 + mc->pf_bytes - pfb0) / dt / 1e9;
            for (int k = 0; k < mc->n_links; ++k) { lk[k] = (mc->link_bytes[k] - lk0[k] + mc->pf_link[k] - pfl0[k]) / dt / 1e9; }
            if (mc->n_src_query > sq0) { pf_recent = 100.0 * (mc->n_src_hit - sh0) / (mc->n_src_query - sq0); }
            win = dt;
            hr_recent = (hh - h0) + (mm - m0) > 0 ? 100.0 * (hh - h0) / ((hh - h0) + (mm - m0)) : hr_recent;
        }
        h0 = hh; m0 = mm;
        t0 = now; cpu0 = mc->cpu_bytes; up0 = mc->up_bytes; pfb0 = mc->pf_bytes; sq0 = mc->n_src_query; sh0 = mc->n_src_hit;
        for (int k = 0; k < mc->n_links; ++k) { lk0[k] = mc->link_bytes[k]; pfl0[k] = mc->pf_link[k]; }
    }
    char b[256];
    snprintf(b, sizeof b, "\"bandwidth\":{\"gpus\":%d,\"window_s\":%.2f,\"ram_gbs\":%.2f,\"ram_max_gbs\":%.2f,\"cpu_gbs\":%.2f,\"uploaded_mib\":%.0f,\"links\":[",
             mc->n_links, win, ram, knobs().ddr_gbs, cpu, (double) mc->up_bytes / 1048576.0);
    std::string o = b;
    for (int k = 0; k < mc->n_links; ++k) {
        snprintf(b, sizeof b, "%s{\"link\":%d,\"dev\":\"%s\",\"gbs\":%.2f,\"max_gbs\":%.2f,\"uploaded_mib\":%.0f}", k ? "," : "", k,
                 mc->link_dev[k] ? ggml_backend_dev_name(mc->link_dev[k]) : "?", lk[k], mc->gbs_link[k], (double) mc->link_bytes[k] / 1048576.0);
        o += b;
    }
    snprintf(b, sizeof b, "]},\"hit\":{\"all\":%.2f,\"recent\":%.2f,\"prompt_all\":%.2f,\"prompt_recent\":%.2f}", hh + mm ? 100.0 * hh / (hh + mm) : 0.0, hr_recent,
             mc->n_src_query ? 100.0 * mc->n_src_hit / mc->n_src_query : 0.0, pf_recent);
    return o + b;
}

void llama_moe_cache_free_fork() {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    if (g_topic.on > 0) { topic_store_pending(mc); }   // the last request's map (hot maps per topic)
    exptok_save();   // --moe expids: the tokens of the last request
    for (const auto & ap : mc->alias) {   // end of run: the whole pool map of every card (what each byte range of the pool holds when the run ends), always logged, once
        size_t total = 0;
        for (const auto & r : pool_map(ap)) { total = std::max(total, r.hi); }
        std::string line = tr_fmt("pool-map end %s: pool %.1f MiB:", led_card(ap), total / 1048576.0);
        for (const auto & r : pool_map(ap)) {
            line += tr_fmt(" [%s%s %.1f-%.1f MiB]", r.kind, r.pcls >= 0 ? tr_fmt(" c%d", (int) r.pcls).c_str() : "", r.lo / 1048576.0, r.hi / 1048576.0);
        }
        if (ap.kv_h > 0) { line += tr_fmt(" | KV zone %zu cells x %zu B per cell", (size_t) ap.kv_h, (size_t) ap.kv_row); }
        LLAMA_LOG_WARN("%s\n", line.c_str());
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    ggml_set_moe_phase_callback(nullptr, nullptr);
    ggml_backend_set_moe_src_callback(nullptr, nullptr);
    ggml_backend_set_moe_fill_callback(nullptr, nullptr);
    ggml_backend_set_moe_where_fn_fork(nullptr, nullptr);
    ggml_backend_set_split_callback(nullptr, nullptr);
    if (!mc->l3_thr.empty()) {
        {
            std::lock_guard<std::mutex> lk(mc->l3_mtx);
            mc->l3_stop = true;
        }
        mc->l3_cv.notify_all();
        for (auto & t : mc->l3_thr) { t.join(); }
        mc->l3_thr.clear();
        if (mc->l3_n_ex) {
            LLAMA_LOG_WARN("moe-cache: L3 prefetch: %llu experts, %llu read before their layer's CPU phase, %llu late, %llu dropped (newer job)\n",
                (unsigned long long) mc->l3_n_ex.load(), (unsigned long long) mc->l3_done.load(),
                (unsigned long long) mc->l3_late.load(), (unsigned long long) mc->l3_dropped.load());
        }
    }
    if (mc->pred_thr.joinable()) {
        {
            std::lock_guard<std::mutex> plk(mc->pmtx);
            mc->pred_stop = true;
        }
        mc->pcv.notify_all();
        mc->pred_thr.join();
    }
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_all();
    for (auto & t : mc->workers) {
        t.join();
    }
    profile_save(mc);
    uint64_t h = 0, m = 0;
    for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
    LLAMA_LOG_WARN("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%, prefill experts from cache %" PRIu64 "/%" PRIu64 ", kept from prefill %" PRIu64 "\n",
            mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0, mc->n_src_hit, mc->n_src_query, mc->n_adopted);
    LLAMA_LOG_WARN("moe-cache: staging lookups: %lld range calls, %lld experts asked, %lld found on the card, %lld found but still in flight, %lld asked of the wrong card\n",
            (long long) g_ac.where_calls.load(), (long long) g_ac.where_range.load(), (long long) g_ac.where_found.load(), (long long) g_ac.where_inflight.load(), (long long) g_ac.where_wrongdev.load());
    alias_cost_log("at exit");
    // --moe layer-stats=1: how many cache slots each layer holds filled at the end of the run, and its hit rate (does a fixed per-layer share fill up evenly?)
    if (const char * e = moe_opt("layer-stats"); e && atoi(e) != 0) {
        for (const auto & ls : mc->layers) {
            int filled = 0;
            for (int32_t x : ls.slot_expert) { filled += x >= 0; }
            LLAMA_LOG_WARN("moe-cache: layer %d: %d experts cached (%s %d), hits %" PRIu64 " misses %" PRIu64 " (%.1f%%)\n", ls.pub.il, filled,
                    ls.pcls >= 0 ? "pool class" : "slots", ls.pcls >= 0 ? ls.pcls : ls.pub.n_slots,
                    (uint64_t) ls.n_hit, (uint64_t) ls.n_miss, ls.n_hit + ls.n_miss ? 100.0*ls.n_hit/(ls.n_hit + ls.n_miss) : 0.0);
        }
    }
    if (mc->pm_pick[0]) {
        LLAMA_LOG_WARN("moe-cache: prefetch candidates L+1 (top-M predicted uncached): M=1 %.0f%% used, M=2 %.0f%%, M=3 %.0f%%; "
                "they cover %.0f%% / %.0f%% / %.0f%% of the real misses (%.2f misses per layer)\n",
                100.0*mc->pm_hit[0]/mc->pm_pick[0], 100.0*mc->pm_hit[1]/std::max<uint64_t>(1, mc->pm_pick[1]),
                100.0*mc->pm_hit[2]/std::max<uint64_t>(1, mc->pm_pick[2]),
                100.0*mc->pm_cov[0]/std::max<uint64_t>(1, mc->pm_miss), 100.0*mc->pm_cov[1]/std::max<uint64_t>(1, mc->pm_miss),
                100.0*mc->pm_cov[2]/std::max<uint64_t>(1, mc->pm_miss), (double) mc->pm_miss/std::max<uint64_t>(1, mc->acc_tot[0]/8));
    }
    if (mc->acc_tot[0]) {
        std::string msg;
        for (int k = 0; k < 8 && mc->acc_tot[k] > 0; ++k) {
            msg += format(" L+%d %.1f%%", k + 1, 100.0*mc->acc_hit[k]/mc->acc_tot[k]);
        }
        LLAMA_LOG_WARN("moe-cache: predicted top-k overlap:%s (%" PRIu64 " training steps)\n", msg.c_str(), mc->n_train);
    }
    pred_save(mc);
    if (mc->n_pred_up || mc->pm_miss) {
        // one line for the harness (fork/run.py (exp)): the predictor's result of this run
        uint64_t hh = 0, mm = 0;
        for (auto & ls : mc->layers) { hh += ls.n_hit; mm += ls.n_miss; }
        const double hr = hh + mm ? 100.0*hh/(hh + mm) : 0.0, boost = hh + mm ? 100.0*mc->n_pred_used/(hh + mm) : 0.0;
        LLAMA_LOG_WARN("moe-pred: hit=%.1f hit_without_est=%.1f boost_pts=%.1f up=%" PRIu64 " pub=%" PRIu64 " used=%" PRIu64 " late=%" PRIu64 " stale=%" PRIu64 " cov1=%.1f cov2=%.1f cov3=%.1f miss_per_layer=%.2f\n",
                hr, hr - boost, boost, mc->n_pred_up, mc->n_pred_pub, mc->n_pred_used, mc->n_pred_late, mc->n_stream_stale,
                100.0*mc->pm_cov[0]/std::max<uint64_t>(1, mc->pm_miss), 100.0*mc->pm_cov[1]/std::max<uint64_t>(1, mc->pm_miss),
                100.0*mc->pm_cov[2]/std::max<uint64_t>(1, mc->pm_miss), (double) mc->pm_miss/std::max<uint64_t>(1, mc->acc_tot[0]/8));
    }
    if (mc->n_pred_up) {
        LLAMA_LOG_WARN("moe-cache: router prediction: %" PRIu64 " uploads, %" PRIu64 " published before their layer, %" PRIu64 " used, %" PRIu64 " too late to start, %" PRIu64 " dropped in the queue (layer started), %" PRIu64 " promoted into the cache, %" PRIu64 " token-boundary prefetches\n",
                mc->n_pred_up, mc->n_pred_pub, mc->n_pred_used, mc->n_pred_late, mc->n_stream_stale, mc->n_promoted, mc->n_tbp);
    }
    mc->lazy_stop.store(true);
    if (mc->lazy_thr.joinable()) { mc->lazy_thr.join(); }
    for (auto * b : mc->staging) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * b : mc->pin_chunks) { ggml_backend_buffer_free(b); }
    for (auto * b : mc->pbufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->pctxs) { ggml_free(c); }
    for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->ctxs) { ggml_free(c); }
    delete mc;
    g_cache     = nullptr;
    g_init_done = false;
}

bool llama_moe_cache_pred_train_now_fork() {
    return g_pred_train_now;
}

static bool g_starved_built = false;
bool llama_moe_cache_graph_reusable_fork() {
    const bool starved = g_cache && g_cache->starved.load();
    return g_pred_train_now == g_pred_train_built && g_pred_epoch == g_pred_epoch_built && g_split_epoch == g_split_epoch_built && starved == g_starved_built &&
           g_kv_epoch.load() == g_kv_epoch_built;   // a moved KV zone: the old graph's views point at the old place
}

int32_t llama_moe_cache_cpu_sat_threads_fork() {
    return g_cache ? g_cache->cpu_sat_threads : 0;
}

void llama_moe_cache_graph_built_fork() {
    g_starved_built = g_cache && g_cache->starved.load();
    g_pred_train_built = g_pred_train_now;
    g_pred_epoch_built = g_pred_epoch;
    g_split_epoch_built = g_split_epoch;
    g_kv_epoch_built = g_kv_epoch.load();
}

namespace {
bool pinned(const moe_cache * mc, const layer_state & ls, int32_t e) {
    const uint64_t stay = (uint64_t) knobs().slow_stay;
    return ls.sticky[e] || (ls.slow && ls.cached_since[e] && mc->n_steps + 1 - ls.cached_since[e] < stay);
}
} // namespace

// C-linkage, exported entry for the server (llama.h): the planner above is a C++ function other engine files declare locally
uint32_t llama_moe_prefill_pass_fork(uint32_t n_tokens, uint32_t ub_max) { return llama_moe_cache_prefill_ubatch_fork(n_tokens, ub_max); }
