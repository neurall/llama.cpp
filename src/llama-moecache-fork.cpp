#include <csignal>
#include "../ggml/src/ggml-read-fork.h"
#include "../ggml/src/ggml-rampool-fork.h"
#include "llama-moecache-fork.h"
#include "llama-backend-fork.h"
#include "llama-fork-hooks.h"
#include "llama-gpu-probe-fork.h"
#include "llama-moestate-fork.h"
#include "llama-dsplit-fork.h"
#include "llama-rls-fork.h"   // the planner's estimator (one RLS per prompt path)
#include "llama-estimate-fork.h"   // the planner's strategy estimators: metrics in, time per token out

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <climits>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "llama-poolalloc-fork.h"
#include "llama-moetopic-fork.h"
#include "llama-passplan-fork.h"
#include "llama-kvtier-fork.h"

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/file.h>
#include <fcntl.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#if defined(__linux__) && __has_include(<linux/aio_abi.h>)
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#define MOE_HAVE_AIO 1   // kernel AIO by raw syscalls: no library, passes Docker's default seccomp (io_uring does not) and runs the same speed as io_uring on O_DIRECT reads (fio, 2026-11-02)
#if !defined(SYS_io_getevents)   // 32-bit ABIs without the plain syscall: the pread pool
#undef MOE_HAVE_AIO
#endif
#endif
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
void llama_moe_decision_fork(const char * name, const char * value, const char * why);   // exported in llama.h; "stage" writes one span of the start-up to stats.log
void llama_moe_ssd_max_fork(double gbs);   // llama-moestate-fork.cpp: the drive's measured maximum, for the share logged on SSD spans
double llama_moe_ssd_max_get_fork();        // the same, read: the call planner's SSD rate until a stream pass measured its own
void llama_moe_span_fork(const char * what, double t0, double t1, uint64_t bytes, const char * where);
void llama_moe_stats_emit_at_fork(const char * kind, const char * text, double at);   // llama-moestate-fork.cpp: a line of stats.log with the time of the event
void llama_moe_cache_fill_fork(uint64_t * filled, uint64_t * total);
   // llama-moestate-fork.cpp: one line of stats.log for one big operation
double llama_moe_span_now_fork();   // seconds since the start of the process, the time base of the spans

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
static int g_early_rd_threads = 0;   // the O_DIRECT read thread count the early machine benchmark found (0 = it did not run); read_threads_probe is not asked again
// The rate the model file was read at while the pin loader filled RAM (GB/s, 0 = no pin loader ran): the start of the stream's SSD cost until a stream pass has measured its own (llama_moe_cache_pp_decide_fork)
static std::atomic<double> g_load_read_gbs{ 0.0 };
// The server is warm once its first filling prompt call or its first request is over (user 2026-11-03: "ignore lower than expected vals as those are first cold outliers or start measuring after n tokens when warm"):
// until then the CPU phases do not feed the CPU rate, the start-up probe's value stands (DeepSeek's first phases read mapped experts cold: 1.8 GB/s against 31.6 probed, then 7.9, then 19)
static std::atomic<bool> g_warm{ false };

void        llama_moe_stats_log_fork(const char * kind, const char * text);   // llama-moestate-fork.cpp: --moe stats
std::string llama_moe_stats_io_fork(int who);                                 // what the disk, the memory and the CPU did since this caller's last call
bool        llama_moe_log_profile_fork(const char * name);                    // --moe log-profile

LLAMA_API void llama_model_wait_loaded_fork(const struct llama_model * m);

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
    bool                  src_ready = true; // its expert sources are read (llama_moe_host_pending_fork); kept by layer_src_ready under wmtx
    std::vector<uint32_t> expert_count;   // expert id -> uses, halved every LLAMA_MOE_CACHE_HALVE_EVERY steps
    std::vector<uint32_t> win_count;      // expert id -> uses in the last 64 tokens
    std::vector<uint64_t> expert_last;    // expert id -> clock of its last use (policy lru)
    std::vector<float>    crf;            // expert id -> uses, halving every 128 tokens (policy crf: ClassLRU, upstream PR 31233)
    std::vector<uint32_t> crf_t;          // expert id -> this layer's token of its last use
    uint32_t              tok = 0;        // tokens this layer has seen
    std::vector<uint8_t>  prot;           // expert id -> in the protected segment (policy slru)
    int32_t               n_prot = 0;
    std::vector<float>    relevance;      // expert id -> router probability for the current token, smoothed: rel = decay * rel + p (policy r)
    std::vector<uint8_t>  ghost_flag;     // expert id -> evicted within the last n_cache evictions of this layer (regret counter)
    std::deque<int32_t>   ghost_q;
    std::deque<std::vector<int32_t>> recent; // ids of those tokens, oldest first
    // where up/gate/down live on disk (fd < 0: unknown, upload from host memory)
    int    src_fd[3]   = { -1, -1, -1 };
    int    src_dfd[3]  = { -1, -1, -1 };   // --moe read-direct: the file again with O_DIRECT, set only for a tensor that stayed mapped (not in pinned memory)
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
    const char * const *  rp_tab[3] = { nullptr, nullptr, nullptr };   // --moe rampool: the pool's table of each tensor (expert -> its block, nullptr: out), nullptr: not in the pool
    std::vector<uint32_t> vis;            // --moe rampool heat: per expert the last RP_VIS decode-step stamps, oldest first (0: none), saved as vis.<layer> in the heat file
    std::vector<uint64_t> heat_cnt;       // --moe rampool: per expert its decode uses over all runs (the heat file's hot.<layer>), ranks experts the ring has not seen
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
                         std::function<ggml_tensor *(int32_t, bool)> rot;
                         std::vector<char> out, out_next; std::vector<std::pair<uint8_t *, size_t>> host;   // RAM part: layers in host memory now / after the next relayout; where
                         std::vector<int8_t> kind; std::vector<int32_t> l_il, l_part; std::function<void(bool)> parts_set;   // --moe attn-split: see llama_kv_tail_fork
                         bool gather = false;   // see llama_kv_tail_fork
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
    bool                       rp_on = false;      // --moe rampool: the RAM pool holds the expert slices (ggml-rampool-fork.h)
    bool                       rp_evict = false;   // --moe rampool=2: the pool evicts and admits during the run; 1: the fill's set stays (ggml_rampool_evict_fork)
    bool                       rp_filling = false; // the first fill runs: only then rp_heat ranks what a card holds after all others
    std::vector<ggml_backend_buffer_t> rp_bufs;    // its pinned chunks
    std::vector<ggml_backend_buffer_t> rp_pre;     // 512 MiB pieces pinned before the VRAM pool is sized, handed out as chunks by rp_alloc (rp_prepin)
    std::vector<std::pair<char *, size_t>> rp_raw; // chunks the fill reads into before they are pinned (rp_alloc), pinned in place after it (rp_pin_raw); size 0: not pinned
    size_t                     rp_raw_bytes = 0;
    size_t                     rp_vram_reserve = 0;   // VRAM the pin after the sizing will take on each card (rp_prepin measures one piece); the sizing leaves it free
    bool (*rp_pin)(void *, size_t) = nullptr;      // the GPU backend's pin of a filled range (ggml_backend_cuda_pin_range_fork), nullptr: the pool pins before the fill as before
    void (*rp_unpin)(void *) = nullptr;
    uint32_t                   vis_base = 0;       // the heat clock where the heat file left it
    std::string                heat_file;          // <state dir>/<model>/globalhot.ini: the pool's heat, apart from the state file (whose values get reset or rolled back by policy)
    std::atomic<uint32_t>      vis_now{1};         // the heat clock: vis_base + decode steps
    ggml_backend_buffer_type_t lazy_buft = nullptr;   // --moe pin-lazy: the GPU's pinned host buffer type
    size_t                     pin_budget = 0, pin_used = 0, pin_left = 0;
    char *                     pin_cur = nullptr;
    uint64_t                   pin_n = 0;
    std::vector<ggml_backend_buffer_t> pin_chunks;
    std::vector<std::pair<ggml_tensor *, void *>> lazy_moved;   // pin-lazy: a weight whose data pointer moved into a pin chunk, and its mapped address (restored before the chunks are freed)
    std::mutex                 pin_mtx;
    std::thread                lazy_thr;          // --moe pin-lazy: reads expert tensors into pinned buffers in the background (pin_prefetch_run)
    std::atomic<bool>          lazy_stop{false};
    std::thread                stats_thr;          // --moe stats: the status lines (stats_status_run)
    std::thread                preheat_thr;        // pinned copies of the hot experts that stayed mapped, read once with O_DIRECT before request 1 (preheat_run)
    std::atomic<bool>          profile_ready{false};   // profile_preload has set the lifetime counts (ls.glob_count): the preheat may rank the experts
    bool                       pin_on_upload = false;   // --moe pinr>0: pin_slice also copies an expert on its second upload; off when only the preheat fills pin_ptr
    std::atomic<bool>          stats_stop{false};
    std::thread                fast_thr;           // the fast engine windows (fast_window_run, planner profile), joined at free

    // model bigger than RAM, weights mmap'd: an expert in VRAM doesn't need its RAM copy. Its pages
    // are dropped when it is cached and read back (async readahead) when it is evicted, so RAM holds
    // the experts the CPU computes instead of the kernel's LRU choice (LLAMA_MOE_CACHE_DROP=1/0)
    bool drop_cached = false;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;
    std::vector<ggml_backend_buffer_t>  miss_bufs;   // --moe miss-dev: the imports of the host weights
    ggml_context *                      miss_ctx = nullptr;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::vector<std::thread> workers;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    uint32_t                 hpend_seen = ~0u;   // g_hpend_gen the layers' src_ready were last set at (wmtx)
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
    double               cpu_gbs_meas = 0;   // EMA of the CPU phases' read rate (layers whose experts sit in pinned RAM)
    double               cpu_gbs_mapped = 0; // the same for layers whose experts stayed mapped: the CPU reads them through page faults from the SSD (GLM 3.5 2026-11-07: 0.5-0.9 GB/s against 41 pinned)
    bool                 phase_mapped = false;   // set by llama_moe_cache_jit_fork for the layer whose CPU phase comes next
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
    // --moe read-direct: one pinned buffer per upload worker and one O_DIRECT descriptor per model file, for the expert slices that stayed mapped (see read_direct)
    std::vector<ggml_backend_buffer_t> rd_staging;
    size_t rd_size = 0;
    std::map<int, int> rd_fds;
    std::map<const ggml_tensor *, std::pair<int, size_t>> rd_src;   // a weight that stayed mapped -> (its O_DIRECT descriptor, where it starts in the file)
    // JIT uploads of a mapped expert (no pinned copy, not in the pool) read the slice with O_DIRECT into one of these pinned slots and DMA from there, never from the mapping
    // (jit_direct_src): 3 slots per expert a token uses, each reused once the copy that last read it is done (its event)
    struct jit_slot { ggml_backend_buffer_t buf = nullptr; ggml_backend_event_t ev = nullptr; ggml_backend_t be = nullptr; };
    std::vector<jit_slot> jit_ring;
    size_t jit_ring_i = 0;
    uint64_t jit_rd_n = 0, jit_rd_b = 0, jit_rd_us = 0, jit_map_n = 0;   // per request: slices read O_DIRECT for JIT uploads, their bytes and time; slices that had to come from the mapping
    double   jit_rd_gbs = 0;   // the measured O_DIRECT rate of those reads (EMA): the planner's upload cost of a mapped expert when it is below the link's rate
    int rd_threads = 8;   // --moe read-threads: at 1 MiB blocks, queue depth 8 gave 5.5 GB/s, 2 gave 3.3 (fio on this file)
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
#ifdef _WIN32
    MEMORYSTATUSEX st = {};   // /proc/meminfo is Linux only: on Windows this returned 0 and every pinning path (rampool, pinr, preheat) refused
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#endif
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
// the next sz bytes of the pinned chunk pool (--moe pinr / the preheat), nullptr when the budget is spent or RAM is low
static char * pin_alloc(moe_cache * mc, size_t sz, size_t align = 1) {   // align 4096: the block starts on a page, so an O_DIRECT read can land in it
    std::lock_guard<std::mutex> lk(mc->pin_mtx);
    const size_t pad = align > 1 ? (size_t) ((align - ((uintptr_t) mc->pin_cur & (align - 1))) & (align - 1)) : 0;
    if (mc->pin_left >= pad + sz && pad > 0) { mc->pin_cur += pad; mc->pin_left -= pad; mc->pin_used += pad; }
    if (mc->pin_used + sz > mc->pin_budget) {
        return nullptr;
    }
    if (mc->pin_left < sz) {
        // the preheat's arena is ONE registered buffer of the whole budget (one allocation, one register, huge pages while free 2 MiB blocks last); --moe pinr grows in 256 MiB steps
        const size_t chunk = mc->pin_on_upload ? (size_t) 256 << 20 : std::max<size_t>(sz, (mc->pin_budget - mc->pin_used + 4095) & ~(size_t) 4095);
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
        mc->pin_cur  = (char *) ggml_backend_buffer_get_base(b);   // the registered anonymous memory of ggml_cuda_host_malloc_registered: huge pages while free 2 MiB blocks last, one register
        mc->pin_left = chunk;
    }
    char * dst = mc->pin_cur;
    mc->pin_cur  += sz;
    mc->pin_left -= sz;
    mc->pin_used += sz;
    if (++mc->pin_n % 2048 == 0) {
        LLAMA_LOG_WARN("moe-cache: pinr: %.1f of %.1f GiB pinned (%" PRIu64 " expert slices)\n", mc->pin_used / 1073741824.0, mc->pin_budget / 1073741824.0, mc->pin_n);
    }
    return dst;
}

static size_t expert_past(const moe_cache * mc, const layer_state & ls, int k, int32_t expert, const ggml_tensor * src, const char * & from);

const char * pin_slice(moe_cache * mc, layer_state & ls, int k, int32_t e, size_t sz, const ggml_tensor * src) {
    if (!mc->pin_buft || !ls.pin_ptr[k]) {
        return nullptr;
    }
    const char * const seen = (const char *) 1;   // marker: uploaded once
    const char * p0 = ls.pin_ptr[k][e].load(std::memory_order_acquire);
    if (p0 > seen) {
        return p0;
    }
    if (!mc->pin_on_upload) {   // only the preheat fills the copies
        return nullptr;
    }
    if (p0 == nullptr) {
        ls.pin_ptr[k][e].store(seen, std::memory_order_release);
        return nullptr;
    }
    const char * from = nullptr;
    const size_t pb = expert_past(mc, ls, k, e, src, from);
    if (pb > 0 && e + 1 >= src->ne[2]) {   // the last expert's past bytes are padding, not the mapping's: no copy
        return nullptr;
    }
    char * dst = pin_alloc(mc, sz + pb);
    if (!dst) {
        return nullptr;
    }
    memcpy(dst, (const char *) src->data + (size_t) e*sz, sz + pb);   // with the bytes a matmul reads past the slice (the next expert's), like the preheat's copies
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
#if defined(__GNUC__) || defined(__clang__)
            __atomic_store_n(&t->data, (void *) base, __ATOMIC_RELEASE);
#else
            std::atomic_thread_fence(std::memory_order_release); t->data = (void *) base;
#endif
            {   std::lock_guard<std::mutex> lk(mc->pin_mtx); mc->pin_chunks.push_back(b); mc->lazy_moved.push_back({ t, old }); }
#ifndef _WIN32
            const uintptr_t pg = 4096, a0 = ((uintptr_t) old + pg - 1) & ~(pg - 1), a1 = ((uintptr_t) old + nb) & ~(pg - 1);
            if (a1 > a0) { madvise((void *) a0, a1 - a0, MADV_DONTNEED); }   // the mapping's pages of this tensor
            posix_fadvise(ls.src_fd[k], (off_t) ls.src_offs[k], (off_t) nb, POSIX_FADV_DONTNEED);   // and the page cache behind them
#endif
            used += nb; ++n_pinned;
            g_load_read_gbs.store((double) used / (double) std::max<int64_t>(1, ggml_time_us() - t0) / 1e3);
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

// O_DIRECT reads of one byte range, two engines: the pread pool (several threads, each one block at a time) and kernel AIO (ONE thread, 32 blocks of 1 MiB in flight).
// fio, PC1 3700X: 8 sync threads 5.9 GB/s, libaio 1 thread qd32 4.7 GB/s and the same with 16 busy cores; PC2 3600: 3.6 against 3.46 (3.3 with 12 cores busy, 4 sync threads 3.4).
// The async engine needs no free cores, the pool wins when cores are free: rd_pick_mode starts from the free cores and then follows what each engine measured on the real reads.
// Ladder when the async engine is not there (old kernel, seccomp, Windows): the pread pool. Windows overlapped I/O would be the same shape (completion port), not written.

#ifdef MOE_HAVE_AIO
// The engine for the next big read: 0 = pread pool, 1 = kernel AIO. Starts from the cores free now (pool needs rd_threads of them), then each engine's measured GB/s (moving average
// over its last reads) decides; every 16th big read runs the other engine so both numbers stay fresh and a changed load is seen. Switches are logged.
static std::atomic<int>      g_rd_mode{-1};
static std::atomic<uint32_t> g_rd_calls{0};
static std::atomic<double>   g_rd_gbs[2] = { 0.0, 0.0 };
static int free_cores_now() {
#ifdef __linux__
    static std::mutex mx; static uint64_t last_busy = 0, last_all = 0; static int64_t last_us = 0; static int cached = -1;
    std::lock_guard<std::mutex> lk(mx);
    const int64_t now = ggml_time_us();
    if (cached >= 0 && now - last_us < 500000) { return cached; }
    FILE * f = fopen("/proc/stat", "r");
    if (!f) { return cached = (int) std::thread::hardware_concurrency(); }
    unsigned long long v[8] = { 0 };
    const int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    fclose(f);
    if (n < 4) { return cached = (int) std::thread::hardware_concurrency(); }
    uint64_t all = 0; for (int i = 0; i < 8; ++i) { all += v[i]; }
    const uint64_t idle = v[3] + v[4], busy = all - idle;
    const int hw = (int) std::thread::hardware_concurrency();
    if (last_all != 0 && all > last_all) { cached = (int) std::lround((double) hw * (1.0 - (double) (busy - last_busy) / (double) (all - last_all))); }
    else { cached = hw; }
    last_busy = busy; last_all = all; last_us = now;
    return cached;
#else
    return (int) std::thread::hardware_concurrency();
#endif
}
static int rd_pick_mode(int rd_threads, bool & explore) {
    explore = false;
    if (!g_aio_ok.load()) { return 0; }
    int m = g_rd_mode.load();
    if (m < 0) { m = free_cores_now() >= rd_threads ? 0 : 1; g_rd_mode.store(m); }
    if (g_rd_calls.fetch_add(1) % 16 == 15) { explore = true; return 1 - m; }
    const double a = g_rd_gbs[m].load(), b = g_rd_gbs[1 - m].load();
    if (a > 0 && b > 1.1 * a) {
        g_rd_mode.store(1 - m);
        LLAMA_LOG_WARN("moe-cache: read engine %s -> %s (%.2f against %.2f GB/s, %d cores free)\n", m ? "kernel AIO" : "pread pool", m ? "pread pool" : "kernel AIO", b, a, free_cores_now());
        return 1 - m;
    }
    return m;
}
static void rd_note(int mode, size_t bytes, int64_t us) {
    if (us <= 0) { return; }
    const double g = (double) bytes / ((double) us * 1e3), old = g_rd_gbs[mode].load();
    g_rd_gbs[mode].store(old > 0 ? 0.7 * old + 0.3 * g : g);
}
#endif

// --moe read-direct: bytes of a mapped expert tensor read with O_DIRECT into a worker's pinned buffer. Through the mapping a slice is read by page faults and 128 KiB
// readahead (DeepSeek 2 workers: 1.3 GB/s, fio on the same file: 3.3 GB/s at queue depth 2 with 1 MiB blocks, 5.5 at 8) and every byte passes the page cache, whose
// reclaim then swaps out the process when RAM is tight (2026-11-02). Returns where the bytes start inside the buffer, nullptr when the caller must use the mapping.
static const char * read_direct(int dfd, uint8_t * buf, size_t cap, size_t off, size_t n) {
    const size_t a = direct_align(), beg = off & ~(a - 1), end = (off + n + a - 1) & ~(a - 1), need = off + n - beg;
    if (end - beg > cap) { return nullptr; }
    if (!sr_read(dfd, (char *) buf, beg, end - beg, need)) { return nullptr; }   // llama-spaced-read-fork.h: the platform's async engine or one read at a time, the spaced order
    return (const char *) buf + (off - beg);
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
    double cpu_gbs   = 0;   // CPU read rate alone, 0 = not measured yet: resource_probe sets it at the start, the CPU phases refine it live (cpu_gbs_meas); --moe cpu-ram-gbs fixes it for a test
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
    double margin_mb   = 0;     // VRAM kept free per GPU after the cache is sized; 0: 484 (the engine sets it from the measured run-time growth)
    double ddr_gbs     = 0;    // CPU + the DMAs together at their peak, 0 = not measured yet (resource_probe sets it); GATE=3: an upload starts only while DDR demand + its link rate stays under this
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
    double gpu_shape   = -1; // GPU_SHAPE=0|1: the CPU computes with the GPU's accumulation order and math (GGML_CPU_GPU_SHAPE), so a miss computed on the CPU has the GPU's bits; -1: as the environment says. Switchable while running (ctl file)
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
        { "GPU_SHAPE", &knobs_t::gpu_shape },
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
        for (const char * n : { "VRAM_RESERVE_MB", "UPLOAD_SHARE", "PIN_HOT", "STAY_BONUS", "SLOW_MIN_STAY", "SWAP_LEAD", "SWAPS_PER_STEP", "CPU_RAM_GBS", "SWAP_LEAD_PER_LINK", "UPLOAD_WAIT", "UPLOAD_WAIT_MAX_US", "PRED_ISO", "PRED_N", "TRACE_N", "TRACE_SKIP", "LEAD", "LEAD_SPAN", "ADMIT", "ADMIT_BOOT", "ADMIT_BOOT_STEPS", "ADMIT_SLOW", "ADMIT_JIT", "PRED_LEAD", "PRED_SLOW", "RAM_CEILING_GBS", "WAIT_SWAPS", "UPLOAD_CHUNK_KB", "PRED_AUTO", "IDLE_UP", "IDLE_UP_N", "EV_CLD", "L3_PF", "L3_AUTO", "PRED_KEEP", "PRED_TUNE", "PRED", "UPLOAD_NOW", "GPU_SHAPE" }) {
            if (const char * e = moe_opt(knob_key(n).c_str())) {
                knob_set(r, n, atof(e));
                user_knobs().insert(n);
            }
        }
        return r;
    }();
    return k;
}
// The CPU's read rate now (user 2026-11-03: "in any circumstance a constant ... people have each different hw"): the live average of its phases, else the start-up probe's; 0 while neither exists,
// and no decision may lean on a number then
static double cpu_gbs_now(const moe_cache * mc) { return mc->cpu_gbs_meas > 0 ? mc->cpu_gbs_meas : knobs().cpu_gbs; }


// GPU_SHAPE knob: flips the CPU backend's GPU-shaped math (quants.c, ggml_cpu_set_gpu_shape). Between decode steps, so one graph compute never mixes both.
static void apply_gpu_shape(double v) {
    if (v < 0) { return; }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    auto set = reg ? (void (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_gpu_shape") : nullptr;
    if (set) {
        set(v != 0);
        LLAMA_LOG_WARN("moe-cache: CPU math in the GPU's order: %s\n", v != 0 ? "on" : "off");
    }
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

enum class policy { halve, window, hybrid, add, lru, rel, crf, slru };
std::atomic<bool> g_rel_dead{false};   // policy r without router probabilities: the cache would never fill (relevance 0 everywhere), so it falls back to the default score
// hot maps per topic (--moe dynhot=1, set at the first prompt): the lifetime term of the score is the topic's map plus this request's own uses (scale: its busiest
// expert of any layer, g_topic_tmax) instead of the one map of every request ever served
std::atomic<bool>     g_topic_on{false};
std::atomic<bool>     g_topic_thinking{false};   // the request being served is thinking (between <think> and </think>): its answer's uses count in the thinking map
static std::atomic<uint64_t> g_ppc_picked{0}, g_ppc_all{0}, g_ppc_miss{0};   // prompt chain, this call: experts its routers picked, experts of those layers, picks not in a block
static std::atomic<uint64_t> g_pjit_miss{0}, g_pjit_up{0}, g_pjit_layers{0};   // prompt calls: missed experts seen by the JIT, uploaded, layer-steps
static std::atomic<uint64_t> g_pjit_left_l[256], g_pjit_pool_l[256];   // per layer: misses left to the CPU in prompt calls; 1 if the layer uses the pool branch
static std::atomic<bool> g_prompt_call{false};   // inside a decode call that is a prompt (llama_moe_cache_prompt_call_fork, set by llama_context::decode)
static std::atomic<int64_t> g_mb_auto { 0 };       // outside a prompt call, the largest pass on the cache path: experts per layer / experts per token, the tokens that can touch them all
static std::atomic<bool> g_pp_short{false};      // the lend of this prompt call left a layer fewer blocks than its pass may use: its passes leave the prompt chain
static std::atomic<uint32_t> g_pp_last_n{0}, g_pp_last_ub{0};   // the last decision's call (tokens, pass ceiling): decided again when the chain is refused after it
static std::atomic<bool> g_pp_dec{false};        // this prompt call runs on the decode chain, misses may stay on the CPU (llama_moe_cache_pp_decide_fork)
static bool              g_pp_dec_built = false; // the same when the last graph was finished
static bool              g_pp_graph = false;     // the graph being built (or the one reused) has layers on the prompt chain
static bool              g_pp_built = false;     // the same when the last graph was finished: a graph is reused only for the same chain
static std::atomic<bool> g_pp_mix_call{true};    // this prompt call may run the prompt mix (llama_moe_cache_pp_decide_fork; a forced path keeps it)
static std::atomic<bool> g_pp_no_dense{false};   // this prompt call stays on one card: a one-card path measured at this size beat the dense split there
static std::atomic<int>    g_pp_path_call{0};    // this prompt call's one-card path (PP_*, below), set by llama_moe_cache_pp_decide_fork: what the dense split is weighed against
static std::atomic<double> g_pp_save_us{0.0};    // what that path is calculated to save over the stream (us): the dense split planner's one-card time less this
static bool              g_pp_mix_call_built = true;   // the same when the last graph was finished
static bool              g_pp_k_built = false;   // the last graph carries the rankings of predicted staging
static double            g_pp_k_m = 0.0;         // and its multiplier (ggml_backend_pred_get_fork when it was built): the report comes after the prompt call ended
extern "C" {
    GGML_API void   ggml_backend_pred_set_fork(double m);   // ggml-backend.cpp: predicted staging of the passes to come
    GGML_API double ggml_backend_pred_get_fork();
}
static bool              g_pp_obs   = false;     // the observer is called by the prompt chain: these ids are a prompt's
static int64_t           g_pp_want  = 0;         // tokens x experts per token of the pass being built: the most blocks a layer of it can use
static std::atomic<int64_t> g_pp_mb{0};          // this prompt call: the largest pass the prompt chain takes (llama_moe_cache_pp_decide_fork)
static bool              g_pp_mix_graph = false; // the graph being built has layers on the prompt mix (llama_moe_cache_pp_mix_fork)
static bool              g_pp_mix_built = false; // the same for the graph that runs: the staging fill leaves out what such a layer's home card holds
static std::atomic<double>  g_pp_up_us{0.0};     // upload time of the misses of the prompt chain's passes since the last report: the pass's time without it is its compute
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
        if (e && strcmp(e, "crf")    == 0) return policy::crf;   // ClassLRU: a use count that halves every 128 tokens gives the class, the score is 2^class / age
        if (e && strcmp(e, "slru")   == 0) return policy::slru;  // two segments: a hit moves an expert to the protected half, whose oldest goes back to the other; LRU inside each
        return policy::add;
    }();
    return p;
}

// eviction score, in "uses" units so the pay-back margin applies to all policies
// gmax: the scale of the lifetime-count term. Within a layer its own busiest expert (ls.glob_max); when experts of different layers compete
// for a block (block pool) one common scale, else a layer that spreads its traffic over many experts gets all of them scored near the top and
// a layer with a few very hot experts loses its second tier (measured MiMo stand-in, 25% cache: layer 4 fell from 64 to 13 experts, hit 68 -> 43%)
// replay prefetch (user 2026-10-29: "for sure prefetch, that's major benefit"; --moe replay=1, with jit-all=1 for a deterministic replay): every
// request's answer picks (per model layer, 16 per token) are stored under its prompt's hash (<store>/trace/<hash>.ids); a request with a stored trace
// plays it: each step uploads the experts the next tokens of the trace need, as many as the link moves in --moe replay-frac (0.9) of a token's time,
// into the blocks whose expert the trace needs latest or never (Belady), and every eviction ranks by the same next use. The token being computed
// takes its own misses just in time (JIT): an expert in flight during its token would be computed by the CPU.
struct replay_state {
    bool   on      = false;                       // --moe replay=1
    bool   active  = false;                       // this request plays a stored trace
    int    ahead   = 4;                           // --moe replay-ahead=N (GLM 3.0bit tetrisc: 1, 2, 4 tokens 29.2 t/s, 8 28.7, 16 27.2): tokens of the trace the prefetch looks at; an expert the trace needs
                                                  // within them is never evicted for a prefetch (that would only move the miss)
    int    horizon = 512;                         // tokens of the trace the next-use ranks see
    double frac    = 0.9;                         // --moe replay-frac
    std::vector<std::vector<int16_t>> rec;        // per model layer: this request's answer picks, 16 per token (-1 pads)
    std::vector<std::vector<int16_t>> play;       // the stored trace being played, same layout
    std::vector<int32_t>              pos;        // per model layer: answer tokens computed so far = the row of the next one
    std::vector<std::vector<int32_t>> nu;         // per model layer and expert: tokens from pos to its next use in the trace (INT32_MAX: none seen)
    uint64_t picks = 0, matched = 0, n_pref = 0, jit0 = 0, jitm0 = 0, n_tok = 0;
    uint64_t n_cpu = 0;                           // answer picks the CPU computed (not in VRAM when their token ran, not uploaded just in time)
    int64_t  t_end  = 0;                          // the last step's end (its prefetch queued): the next step's start minus this = the token's compute
    double   cu_ema = 0;                          // that compute time, smoothed (us): the link budget of the prefetch is a share of it, not of the step
                                                  // (the step also waits for the uploads: sized by the step, the budget grew with its own wait)
    int32_t  first_mis = -1, first_mis_layer = -1; // the first answer token whose picks differ from the stored trace's
    int32_t  first_cpu = -1, first_cpu_layer = -1; // the first answer token a pick of which the CPU computed
};
replay_state g_rp;

double score(const layer_state & ls, int32_t id, uint64_t gmax) {
    if (g_rp.active && ls.pub.il >= 0 && (size_t) ls.pub.il < g_rp.nu.size() && (size_t) id < g_rp.nu[ls.pub.il].size()) {
        const int32_t d = g_rp.nu[ls.pub.il][id];   // replay: the trace's next use ranks (sooner = higher, never again = 0)
        return d == INT32_MAX ? 0.0 : 1e9 / (1.0 + (double) d);
    }
    switch (get_policy()) {
        case policy::window: return ls.win_count[id];
        case policy::lru:    return (double) ls.expert_last[id];
        case policy::crf: {
            static const double hot = [] { const char * w = moe_opt("crf-hot"); return w ? atof(w) : 0.0; }();   // --moe crf-hot=K: plus K x the lifetime share (the ini's counts and this run's)
            const uint32_t age = ls.tok - ls.crf_t[id];
            return (double) (1u << std::min(7, (int) std::floor(std::log2(std::max(1.0f, ls.crf[id] * std::exp2(-(float) age / 128.0f)))))) / (double) (age + 1) +
                   hot * ((double) ls.glob_count[id] / (double) gmax);
        }
        case policy::slru:   return (ls.prot[id] ? 1e15 : 0.0) + (double) ls.expert_last[id];
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
    return get_policy() == policy::lru || get_policy() == policy::rel || get_policy() == policy::crf || get_policy() == policy::slru ? 0.0 : (double) margin;
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

// --moe rampool heat (user 2026-11-07): per expert a fixed ring of its last RP_VIS decode-step stamps. Expected next use = last stamp + mean gap between the stamps; one stamp: now + its
// age; past due: now + how long overdue. Never seen: after every seen expert, by the lifetime count. A card holds it: HUGE_VAL (a RAM copy pays nothing).
// Ten stamps (user 2026-11-08: "consider last 10 stored revisits and evict rare visitors first, not just by global hot count"): a rare visitor's mean gap is long, so it is
// expected back last and leaves the pool first; the lifetime count only orders the never-seen. A heat file written with a narrower ring keeps its stamps (rp_heat_load).
// Not built yet (user 2026-11-08, "keep note"): beside the step stamps, the last 10 request ids an expert served. An expert hot inside one request but absent from the
// other recent requests is topical, not general: once its request is over its step estimate is stale, and it should be judged by how many of the recent requests it served
// (expected next use = now + mean request length x requests seen / requests served), so the general experts outlive the topical ones across requests. Needs a test we do not
// have: a sequence of prompts never seen (`--gen in=2000:6000,out=256,task=code|prose`, 8+ runs on one server) reading the RPL line's misses-from-RAM share per request.
static constexpr int RP_VIS = 10;
static void vis_stamp(layer_state & ls, int32_t e, uint32_t now) {
    if (ls.vis.empty()) { return; }
    uint32_t * v = ls.vis.data() + (size_t) e * RP_VIS;
    if (v[RP_VIS - 1] == now) { return; }   // the same step again (another token of the call)
    memmove(v, v + 1, (RP_VIS - 1) * sizeof(uint32_t));
    v[RP_VIS - 1] = now;
}
static double vis_next(const moe_cache * mc, const layer_state & ls, int32_t e) {
    const double now = (double) mc->vis_now.load(std::memory_order_relaxed);
    const uint32_t * v = ls.vis.empty() ? nullptr : ls.vis.data() + (size_t) e * RP_VIS;
    int n = 0;
    while (v && n < RP_VIS && v[RP_VIS - 1 - n] != 0) { ++n; }
    if (n == 0) { return 1e15 + 1e12 / (1.0 + (double) (ls.heat_cnt.empty() ? ls.glob_count[e] : ls.heat_cnt[e])); }
    const double last = (double) v[RP_VIS - 1];
    if (n == 1) { return now + (now - last); }
    const double nxt = last + (last - (double) v[RP_VIS - n]) / (n - 1);
    return nxt >= now ? nxt : now + (now - nxt);
}
static double rp_heat(const void * key, int64_t e, void * ud) {   // a pool reader's thread, the pool's lock held: never takes mc->mtx (racy reads of plain ints)
    const moe_cache * mc = (const moe_cache *) ud;
    const auto it = mc->by_src.find((const ggml_tensor *) key);
    if (it == mc->by_src.end()) { return HUGE_VAL; }
    const layer_state & ls = mc->layers[it->second.first];
    // The fill with eviction on puts what a card holds after all others: the most unique content at the start (the pool admits them later when a card drops them). Eviction
    // itself does not: a card holds an expert only for a while (GLM 3.5 12k decode: 31 VRAM evictions per token), and every eviction took such a block, so the expert came
    // back from the SSD once the card dropped it (pinpool-default 2026-11-08: 15424 demand reads, 13.0 s waited in a 74 s decode, decode 13.7 against 14.7 without the
    // pool). Without eviction the fill ranks card-held experts by heat too: a hot one left out would never come back (rampool3: 17 GiB of card-held slices out of the pool).
    if (mc->rp_evict && mc->rp_filling && (ls.expert_slot[e] >= 0 || ls.queued[e])) { return HUGE_VAL; }
    return vis_next(mc, ls, (int32_t) e);
}

// The heat file (user 2026-11-07: expert heat in its own file, not mixed with tuning values): <state dir>/<model>/globalhot.ini, lines clock=N, hot.<layer>=uses..., vis.<layer>=stamps...
// (RP_VIS per expert, oldest first). Written beside it and renamed over it under <file>.lock, so a crash or a second server never leaves half a file.
struct heat_lock {
    int fd = -1;
    explicit heat_lock(const std::string & f) {
#ifndef _WIN32
        fd = open((f + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
        if (fd >= 0) { while (flock(fd, LOCK_EX) != 0 && errno == EINTR) {} }
#endif
    }
    ~heat_lock() { if (fd >= 0) { close(fd); } }
};
static void rp_heat_save(const moe_cache * mc) {
    if (!mc->rp_on || mc->heat_file.empty()) { return; }
    std::string o = "clock=" + std::to_string(mc->vis_now.load()) + "\n";
    for (const auto & ls : mc->layers) {
        const std::string l = std::to_string(parse_layer_from_name(ls.pub.up_src->name));
        o += "hot." + l + "=";
        for (size_t e = 0; e < ls.heat_cnt.size(); ++e) { o += (e ? " " : "") + std::to_string(ls.heat_cnt[e]); }
        o += "\nvis." + l + "=";
        for (size_t i = 0; i < ls.vis.size(); ++i) { o += (i ? " " : "") + std::to_string(ls.vis[i]); }
        o += "\n";
    }
    heat_lock lk(mc->heat_file);
    const std::string tmp = mc->heat_file + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) { return; }
    const bool ok = fwrite(o.data(), 1, o.size(), f) == o.size();
    if (fclose(f) == 0 && ok) { std::error_code ec; std::filesystem::rename(tmp, mc->heat_file, ec); if (!ec) { return; } }
    std::remove(tmp.c_str());
}
// false: no file (the counts start from the state's lifetime counts, once); a line of another layout is skipped (that layer starts empty)
static bool rp_heat_load(moe_cache * mc) {
    std::map<std::string, std::string> kv;
    {
        heat_lock lk(mc->heat_file);
        std::ifstream in(mc->heat_file);
        if (!in) { return false; }
        for (std::string line; std::getline(in, line); ) {
            const size_t eq = line.find('=');
            if (eq != std::string::npos) { kv[line.substr(0, eq)] = line.substr(eq + 1); }
        }
    }
    if (kv.count("clock")) { mc->vis_base = (uint32_t) strtoul(kv["clock"].c_str(), nullptr, 10); mc->vis_now.store(mc->vis_base + 1); }
    for (auto & ls : mc->layers) {
        const std::string l = std::to_string(parse_layer_from_name(ls.pub.up_src->name));
        std::vector<uint64_t> h; std::vector<uint32_t> v;
        { std::istringstream is(kv["hot." + l]); for (uint64_t x; is >> x; ) { h.push_back(x); } }
        { std::istringstream is(kv["vis." + l]); for (uint32_t x; is >> x; ) { v.push_back(x); } }
        if (h.size() == ls.heat_cnt.size()) { ls.heat_cnt = h; }
        if (v.size() == ls.vis.size()) { ls.vis = v; }
        else if (!ls.vis.empty() && !v.empty() && v.size() % ls.heat_cnt.size() == 0) {   // a file from a narrower (or wider) ring: the newest stamps of each expert, oldest first
            const size_t w = v.size() / ls.heat_cnt.size(), keep = std::min<size_t>(w, RP_VIS);
            for (size_t e = 0; e < ls.heat_cnt.size(); ++e) { for (size_t i = 0; i < keep; ++i) { ls.vis[e * RP_VIS + RP_VIS - keep + i] = v[e * w + w - keep + i]; } }
        }
    }
    return true;
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
    rp_heat_save(mc);
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

void * cpu_proc_fork(const char * name) {   // the CPU backend's fork entry points (ggml-cpu.cpp, ggml_backend_cpu_get_proc_address)
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}
void * cuda_obs_proc(const char * name) {   // a process-wide hook from whichever backend has it (llama-backend-fork.h)
    return llama_fork_proc(name);
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
}

}  // namespace

// --moe uniqdump=PATH (user 2026-10-29: "full 60k set collect"; "inc counters from router selections to get full set per req"): the GPU histograms of the
// stock observer (every expert matmul, prompt batches too) are read when a request ends; the delta since the last request is its full set. One line per
// request and MoE layer goes to PATH ("req N layer L uses U unique K: counts of every expert"), the log gets the unique experts per layer (number, % of
// the layer's experts) and the share of the uses the 8 hottest carry. Needs --moe observe=0 (the observer would switch the counters off after N steps).
namespace {
struct uniq_state_t {
    bool                  on = false;
    std::string           path;
    int                   nl = 0, ne = 0;
    uint64_t              n_req = 0;
    std::vector<uint64_t> ans, prompt;           // this request's answer and prompt picks per layer x expert
    bool                  in_prompt = false;
    int64_t               dec = 0, seg0 = 0, seg = 256;   // answer tokens so far, the current segment's first, segment length
    int                   trace = 0;                     // --moe uniqtrace=N: answer tokens traced per request (0: off)
    bool                  host = false;                  // the expert cache runs: its observer gives the true expert ids (the GPU sees cache slots)
    std::mutex            mtx;                           // host mode: the observer adds, the decode thread reads
    std::vector<uint64_t> hcnt;                          // host mode: picks since the last read, layer x expert
    std::vector<std::vector<int16_t>> htrace;            // host mode: per layer the answer's picks, 16 per token (-1 pads)
    int                   acts = 0;                      // --moe acts=N: the answer's first N tokens' expert inputs of every layer to <store>/acts (0: off)
    int                   n_embd = 0;
    std::vector<std::vector<uint16_t>> hacts;            // per layer: the answer tokens' expert inputs (f16, n_embd per token)
    std::vector<std::vector<int16_t>>  hids;             // per layer: their picks, 16 per token (-1 pads)
} g_uniq;

// host mode (expert cache): the observer's ids of one MoE layer and batch, and with --moe acts the experts' input rows of the answer's tokens
static void uniq_host_obs(int il, const ggml_tensor * ids, const ggml_tensor * op, bool prefill) {
    auto & u = g_uniq;
    if (!u.on || il < 0 || il >= u.nl) { return; }
    std::lock_guard<std::mutex> lk(u.mtx);
    if (!u.host) {   // the cache started after uniqdump did (deferred setup): its observer counts from now on, the GPU counters stop
        u.host = true;
        if (auto off = (void (*)(void)) cuda_obs_proc("ggml_backend_fork_obs_full_off_fork")) { off(); }
        if (u.trace == 0) { if (const char * tr = moe_opt("uniqtrace"); tr && atoi(tr) > 0) { u.trace = atoi(tr); } }
        if (u.acts == 0)  { if (const char * ac = moe_opt("acts"); ac && atoi(ac) > 0) { u.acts = atoi(ac); } }
    }
    const int64_t n_ids = ids->ne[0], n_tok = ids->ne[1];
    auto & tr = u.htrace[il];
    for (int64_t t = 0; t < n_tok; ++t) {
        const bool keep = !prefill && u.trace > 0 && tr.size() < (size_t) u.trace * 16;
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id >= 0 && id < u.ne) { u.hcnt[(size_t) il*u.ne + id]++; }
            if (keep && i < 16) { tr.push_back((int16_t) id); }
        }
        for (int64_t i = n_ids; keep && i < 16; ++i) { tr.push_back(-1); }
    }
    const ggml_tensor * x = op ? op->src[1] : nullptr;   // the experts' input: [n_embd, 1, tokens] (or [n_embd, tokens]), F32 on the host
    if (prefill || u.acts <= 0 || x == nullptr || x->type != GGML_TYPE_F32 || x->data == nullptr || (x->ne[2] != n_tok && x->ne[1] != n_tok)) { return; }
    if (u.n_embd == 0) { u.n_embd = (int) x->ne[0]; }
    if (x->ne[0] != u.n_embd) { return; }
    const size_t xs = x->ne[2] == n_tok ? x->nb[2] : x->nb[1];
    auto & ha = u.hacts[il];
    auto & hi = u.hids[il];
    for (int64_t t = 0; t < n_tok && hi.size() < (size_t) u.acts * 16; ++t) {
        const float * row = (const float *) ((const char *) x->data + t * xs);
        for (int c = 0; c < u.n_embd; ++c) { ha.push_back(ggml_fp32_to_fp16(row[c])); }
        for (int64_t i = 0; i < 16; ++i) {
            hi.push_back(i < n_ids ? (int16_t) *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]) : (int16_t) -1);
        }
    }
}

static std::string topic_store_dir();   // below: the model's hot map directory ("" without one)

// --moe acts: the request's answer tokens to <store>/acts/<n>.f16 (per layer in layer order: its tokens' expert inputs, n_embd f16 each) and <n>.ids
// (the same tokens' picks, 16 int16 per token, -1 pads), described by <n>.json (layers, tokens per layer, n_embd); n = the next free number
static void uniq_acts_dump() {
    auto & u = g_uniq;
    std::lock_guard<std::mutex> lk(u.mtx);
    if (u.acts <= 0 || u.n_embd <= 0) { return; }
    size_t tot = 0;
    for (const auto & v : u.hids) { tot += v.size(); }
    const std::string base = topic_store_dir();
    if (tot == 0 || base.empty()) { for (auto & v : u.hacts) { v.clear(); } for (auto & v : u.hids) { v.clear(); } return; }
    const std::filesystem::path dir = std::filesystem::path(base) / "acts";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    int n = 0;
    while (std::filesystem::exists(dir / (std::to_string(n) + ".json"), ec)) { n++; }
    const std::string stem = (dir / std::to_string(n)).string();
    FILE * fa = fopen((stem + ".f16").c_str(), "wb");
    FILE * fi = fopen((stem + ".ids").c_str(), "wb");
    FILE * fj = fopen((stem + ".json").c_str(), "w");
    if (fa && fi && fj) {
        fprintf(fj, "{\"n_embd\": %d, \"k\": 16, \"layers\": [", u.n_embd);
        bool first = true;
        for (int il = 0; il < u.nl; ++il) {
            if (u.hids[il].empty()) { continue; }
            fwrite(u.hacts[il].data(), sizeof(uint16_t), u.hacts[il].size(), fa);
            fwrite(u.hids[il].data(), sizeof(int16_t), u.hids[il].size(), fi);
            fprintf(fj, "%s[%d, %zu]", first ? "" : ", ", il, u.hids[il].size() / 16);
            first = false;
        }
        fprintf(fj, "]}\n");
        LLAMA_LOG_WARN("moe-cache: acts: request's answer inputs and picks of every layer to %s.f16/.ids (%.1f MB)\n", stem.c_str(), (double) tot / 16 * u.n_embd * 2 / 1e6);
    }
    if (fa) { fclose(fa); }
    if (fi) { fclose(fi); }
    if (fj) { fclose(fj); }
    for (auto & v : u.hacts) { v.clear(); }
    for (auto & v : u.hids) { v.clear(); }
}
}

// the counts since the last read under a label (prompt, answer 0-255, ...): one line per layer to the file, a summary to the log; also added to the
// request's total. Returns nothing; a segment with no picks writes nothing.
static void uniq_report(const std::vector<uint64_t> & now, const char * label) {
    auto & u = g_uniq;
    if (u.path.empty()) { return; }   // --moe acts alone: no uniqdump file, no lines
    FILE * f = u.path.empty() ? nullptr : fopen(u.path.c_str(), "a");
    uint64_t uses = 0, top8 = 0, uniq_sum = 0;
    int      layers = 0, umin = INT32_MAX, umax = 0;
    for (int il = 0; il < u.nl; ++il) {
        std::vector<uint64_t> c(now.begin() + (size_t) il*u.ne, now.begin() + (size_t) (il + 1)*u.ne);
        uint64_t tot = 0;
        int      k   = 0;
        for (int e = 0; e < u.ne; ++e) { tot += c[e]; k += c[e] > 0; }
        if (tot == 0) { continue; }
        if (f) {
            fprintf(f, "req %llu %s layer %d uses %llu unique %d:", (unsigned long long) u.n_req, label, il, (unsigned long long) tot, k);
            for (int e = 0; e < u.ne; ++e) { fprintf(f, " %llu", (unsigned long long) c[e]); }
            fputc('\n', f);
        }
        std::partial_sort(c.begin(), c.begin() + std::min(8, u.ne), c.end(), std::greater<uint64_t>());
        for (int i = 0; i < std::min(8, u.ne); ++i) { top8 += c[i]; }
        uses += tot; uniq_sum += k; layers++; umin = std::min(umin, k); umax = std::max(umax, k);
    }
    if (f) { fclose(f); }
    if (layers > 0) {
        LLAMA_LOG_WARN("moe-cache: uniqdump: request %llu %s: %llu expert uses, unique experts per layer %.1f of %d (%.1f%%; min %d, max %d, %d layers), the 8 hottest of a layer carry %.1f%% of its uses\n",
                (unsigned long long) u.n_req, label, (unsigned long long) uses, (double) uniq_sum / layers, u.ne, 100.0 * (double) uniq_sum / layers / u.ne, umin, umax, layers,
                100.0 * (double) top8 / (double) uses);
    }
}

// read the cards' counters (cleared by the read) as one segment: into the request's answer total unless it is the prompt
static void uniq_flush(const char * label, bool answer) {
    auto & u = g_uniq;
    if (!u.on) { return; }
    std::vector<uint64_t> now((size_t) u.nl*u.ne, 0);   // the picks since the last read (the counters are cleared by the read)
    if (u.host) {
        std::lock_guard<std::mutex> lk(u.mtx);
        now.swap(u.hcnt);
        u.hcnt.assign(now.size(), 0);
        if (std::all_of(now.begin(), now.end(), [](uint64_t c) { return c == 0; })) { return; }
    } else {
        auto rd = (bool (*)(int, int, uint64_t *)) cuda_obs_proc("ggml_backend_fork_obs_full_read_fork");
        if (!rd || !rd(u.nl, u.ne, now.data())) { return; }
    }
    uniq_report(now, label);
    for (size_t i = 0; i < now.size(); ++i) { (answer ? u.ans : u.prompt)[i] += now[i]; }
}

// every decode call (before its graph runs): the batches up to the first one of at most 8 tokens are the prompt; the answer is cut into segments of
// --moe uniqseg=N tokens (default 256) to see experts hot early and cold late (user 2026-10-29)
void llama_moe_uniq_step(int n_tokens) {
    auto & u = g_uniq;
    if (!u.on) { return; }
    if (n_tokens > 8) { u.in_prompt = true; return; }
    if (u.in_prompt) { uniq_flush("prompt", false); u.in_prompt = false; u.seg0 = u.dec = 0; }
    if (u.dec - u.seg0 >= u.seg) {
        const std::string l = "answer " + std::to_string(u.seg0) + "-" + std::to_string(u.dec - 1);
        uniq_flush(l.c_str(), true);
        u.seg0 = u.dec;
    }
    u.dec += n_tokens;
}

// --moe uniqtrace=N: the request's answer picks per layer in token order to PATH.trace, one line per layer: "req R layer L tokens T: a,b,..,h a,b,..,h ..."
// (the k picks of each token, comma separated; tokens space separated)
static void uniq_trace_dump() {
    auto & u = g_uniq;
    if (!u.on || u.trace <= 0) { return; }
    std::vector<int32_t> n_tok(u.nl, 0);
    std::vector<int16_t> ids((size_t) u.nl * u.trace * 16, -1);
    if (u.host) {
        std::lock_guard<std::mutex> lk(u.mtx);
        for (int il = 0; il < u.nl; ++il) {
            n_tok[il] = (int32_t) (u.htrace[il].size() / 16);
            std::copy(u.htrace[il].begin(), u.htrace[il].end(), ids.begin() + (size_t) il * u.trace * 16);
            u.htrace[il].clear();
        }
    } else {
        auto rd = (bool (*)(int, int32_t *, int16_t *)) cuda_obs_proc("ggml_backend_fork_obs_trace_read_fork");
        if (!rd || !rd(u.nl, n_tok.data(), ids.data())) { return; }
    }
    FILE * f = fopen((u.path + ".trace").c_str(), "a");
    if (!f) { return; }
    for (int il = 0; il < u.nl; ++il) {
        if (n_tok[il] <= 0) { continue; }
        fprintf(f, "req %llu layer %d tokens %d:", (unsigned long long) u.n_req, il, n_tok[il]);
        for (int t = 0; t < n_tok[il]; ++t) {
            const int16_t * r = ids.data() + ((size_t) il * u.trace + t) * 16;
            fputc(' ', f);
            for (int i = 0, first = 1; i < 16; ++i) {
                if (r[i] < 0) { continue; }
                fprintf(f, first ? "%d" : ",%d", r[i]);
                first = 0;
            }
        }
        fputc('\n', f);
    }
    fclose(f);
}

static void uniq_request_end() {
    auto & u = g_uniq;
    if (!u.on) { return; }
    uniq_trace_dump();
    uniq_acts_dump();
    if (u.in_prompt) { uniq_flush("prompt", false); u.in_prompt = false; }
    const std::string l = "answer " + std::to_string(u.seg0) + "-" + std::to_string(u.dec > 0 ? u.dec - 1 : 0);
    uniq_flush(l.c_str(), true);
    uniq_report(u.ans, "answer all");
    for (size_t i = 0; i < u.ans.size(); ++i) { u.ans[i] += u.prompt[i]; }
    uniq_report(u.ans, "whole request");
    std::fill(u.ans.begin(), u.ans.end(), 0);
    std::fill(u.prompt.begin(), u.prompt.end(), 0);
    u.seg0 = u.dec = 0;
    u.n_req++;
}

void llama_moe_uniq_start(const llama_model & model) {   // called for every model, also one fully on the GPUs (no cache, no observer)
    if (g_uniq.on) { return; }
    const char * up = moe_opt("uniqdump");
    const char * ac = moe_opt("acts");
    if (((up && *up) || (ac && atoi(ac) > 0)) && model.hparams.n_expert > 0) {   // --moe acts works without uniqdump too
        g_uniq.on = true;
        g_uniq.path = up && *up ? up : "";
        g_uniq.nl = (int) model.hparams.n_layer();
        g_uniq.ne = (int) model.hparams.n_expert;
        g_uniq.ans.assign((size_t) g_uniq.nl*g_uniq.ne, 0);
        g_uniq.prompt.assign((size_t) g_uniq.nl*g_uniq.ne, 0);
        if (const char * sg = moe_opt("uniqseg"); sg && atoi(sg) > 0) { g_uniq.seg = atoi(sg); }
        g_uniq.host = llama_moe_cache_active_fork();   // the cache's observer gives true expert ids; the GPU counters would see cache slots
        g_uniq.hcnt.assign((size_t) g_uniq.nl*g_uniq.ne, 0);
        g_uniq.htrace.assign(g_uniq.nl, {});
        g_uniq.hacts.assign(g_uniq.nl, {});
        g_uniq.hids.assign(g_uniq.nl, {});
        if (const char * tr = moe_opt("uniqtrace"); tr && atoi(tr) > 0 && g_uniq.host) { g_uniq.trace = atoi(tr); }
        if (const char * ac = moe_opt("acts"); ac && atoi(ac) > 0 && g_uniq.host) { g_uniq.acts = atoi(ac); }
        if (auto in = g_uniq.host ? nullptr : (bool (*)(int, int, int)) cuda_obs_proc("ggml_backend_fork_obs_full_init_fork")) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {   // every GPU whose backend has the hook
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_is_gpu(d) && llama_fork_proc_dev(d, "ggml_backend_fork_obs_full_init_fork")) { in(llama_fork_dev_index(d), g_uniq.nl, g_uniq.ne); }
            }
        }
        if (const char * tr = moe_opt("uniqtrace"); tr && atoi(tr) > 0 && !g_uniq.host) {
            if (auto ti = (bool (*)(int, int, int)) cuda_obs_proc("ggml_backend_fork_obs_trace_init_fork")) {
                g_uniq.trace = atoi(tr);
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_is_gpu(d) && llama_fork_proc_dev(d, "ggml_backend_fork_obs_trace_init_fork")) { ti(llama_fork_dev_index(d), g_uniq.nl, g_uniq.trace); }
                }
            }
        }
    }
}

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
    if (auto fn = (void (*)(bool)) cuda_obs_proc("ggml_backend_fork_obs_enable")) {
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
    if (auto en = (void (*)(bool)) cuda_obs_proc("ggml_backend_fork_obs_enable")) {
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
        if (auto en = (void (*)(bool)) cuda_obs_proc("ggml_backend_fork_obs_enable")) {
            en(false);
        }
        if (auto rd = (bool (*)(int, int, uint64_t *)) cuda_obs_proc("ggml_backend_fork_obs_read")) {
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

static double early_link_gbs(ggml_backend_dev_t d);   // pinned host to device GB/s, measured once per process (defined below, outside the anonymous namespace resource_probe sits in)
static void miss_dev_setup(moe_cache * mc);   // defined below, next to it (--moe miss-dev)
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

// --moe prompt-victim=1: which cached expert a prompt pass's miss takes, lowest rank first (free blocks and the experts the layer uses now are the callers' part). 0: the old order.
// all: the pass visits nearly all of this layer's experts, so every expert comes back next pass and hotness says nothing: the farthest next use goes first, that is this layer's
// unused blocks, then the layers just finished, most recent first, then the layers furthest ahead. A smaller pass: the coldest by hotness among those, wherever it lies (equal: the
// later layer). The next two layers give blocks last in both: they are needed soonest.
inline double prompt_victim_rank(const moe_cache * mc, const layer_state & lo, size_t o, size_t jli, int32_t x, bool all) {
    static const bool on = [] { const char * e = moe_opt("prompt-victim"); return e && atoi(e) != 0; }();
    // the ranks below leave room for scores under 9e5, and the clock of the time-based policies runs past that: their score goes in as a share of the clock
    const auto bounded = [&](double s) {
        const policy p = get_policy();
        if (p == policy::lru)  { return 9e5 * (double) lo.expert_last[x] / (double) std::max<uint64_t>(mc->clock, 1); }
        if (p == policy::slru) { return (lo.prot[x] ? 4.5e5 : 0.0) + 4.4e5 * (double) lo.expert_last[x] / (double) std::max<uint64_t>(mc->clock, 1); }
        return std::min(s, 9e5);
    };
    if (!on) {
        const double s = bounded(score(lo, x));
        return o > jli ? 1e12 - (double) o * 1e6 + s : s;
    }
    const double s = bounded(xscore(mc, lo, x));
    if (o > jli && o <= jli + 2) { return 1e12 - (double) o * 1e6 + s; }
    if (all) { return o <= jli ? (double) (jli - o) * 1e6 + s : 1e11 - (double) o * 1e6 + s; }
    return s + 1e-3 * (1.0 - (double) o / (double) mc->layers.size());
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
    mc->profile_ready.store(true, std::memory_order_release);
    std::stable_sort(picks.begin(), picks.end(), [](const pick & a, const pick & b) { return a.heat > b.heat; });
    const bool fixed = [] { const char * e = moe_opt("fixed"); return e && atoi(e) != 0; }();   // --moe fixed=1: ordinary jobs the step waits for (repeatable runs)
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
        // a background job (as the refill after a prompt): no step waits for the start-up fill. As ordinary jobs the first decode (the server's 2-token warm-up)
        // waited for all of them: the slow card's layers are read last and its x4 link took 1.5-3.3 s more (GLM, MiMo, qni4 2026-11-10), the call itself 0.4 s
        pj.bg = !fixed;
        mc->todo.push_back(pj);
        queued++;
    }
    mc->wcv.notify_all();
    return queued;
}

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

// fork: host buffers an asynchronous load still reads (LLAMA_ASYNC_LOAD): the model registers its tensors' buffers before its read thread starts, the
// loader's pin thread marks each one read once its last read landed and it is pinned, the read thread marks the rest at its end. While any load reads, a
// worker takes only jobs whose expert sources sit in buffers marked read: a buffer the registration missed waits for the end of the load, as before,
// instead of being uploaded unread. Until 2026-11-10 the workers waited for the whole read, so the start-up fill (profile_preload, ~40 GB on GLM 3.0)
// began when it ended and the first call (the seq_rm probe at load) waited for all of it: GLM 3.0 2.7 s, MiMo 3.1 s on the x4 card's link.
static std::mutex                g_hpend_mtx;
static std::vector<const void *> g_hpend;           // bases of the buffers being read
static std::vector<const void *> g_hread;           // bases of the buffers marked read (a registration drops its base from here: a reused address)
static std::atomic<uint32_t>     g_hpend_gen{0};    // bumped on every change, before g_hpend_n
static std::atomic<size_t>       g_hpend_n{0};
}  // namespace (reopened below)
LLAMA_API void llama_moe_host_pending_fork(const void * base, bool pending) {
    std::lock_guard<std::mutex> lk(g_hpend_mtx);
    auto & from = pending ? g_hread : g_hpend;
    auto & to   = pending ? g_hpend : g_hread;
    if (const auto it = std::find(from.begin(), from.end(), base); it != from.end()) { from.erase(it); }
    if (std::find(to.begin(), to.end(), base) == to.end()) { to.push_back(base); }
    g_hpend_gen.fetch_add(1, std::memory_order_release);
    g_hpend_n.store(g_hpend.size(), std::memory_order_release);
}
namespace {
// mc->wmtx held: the layer's up/gate/down sources are read: no load reads, or each sits in a buffer marked read (never one only left unregistered)
static bool layer_src_ready(moe_cache * mc, size_t li) {
    const uint32_t g = g_hpend_gen.load(std::memory_order_acquire);
    if (mc->hpend_seen != g) {
        mc->hpend_seen = g;
        std::lock_guard<std::mutex> lk(g_hpend_mtx);
        for (auto & ls : mc->layers) {
            ls.src_ready = true;
            for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
                if (!t) { continue; }
                const void * b = t->buffer ? ggml_backend_buffer_get_base(t->buffer) : nullptr;
                if (!g_hpend.empty() && (!b || std::find(g_hread.begin(), g_hread.end(), b) == g_hread.end())) { ls.src_ready = false; }
            }
        }
    }
    return mc->layers[li].src_ready;
}

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
// Columns (2026-10-28, so a line says who did what to which bytes, when): t_us, thread (the host thread: a name it gave itself, e.g. up-w0 for upload worker
// 0, else t<n> in the order threads first wrote a line), card, kind, rw (w: writes the bytes, o: changes who owns them, rw: a graph reads and writes its lend,
// x: a violation), state (start / done for work with a duration: an upload, a graph from its lend to the step after it; once for the rest), lo, hi, bytes,
// who. The graph's window runs from its "lend" line (queued) to its "graph-done" line (the step after the context synchronised), so a write by another
// thread into those bytes inside that window is a race. The file is flushed every --moe ledger-flush-ms (default 2000) and at exit, so a hang or a kill
// keeps all but the last seconds.
static bool led_on() {
    static const bool on = [] { const char * e = moe_opt("ledger"); return e && atoi(e) != 0; }();
    return on;
}
// --moe race-stress=N (debug): every upload worker sleeps a random 0..N ms before each copy and again after it, before the job is reported done. Timing
// windows that are rare at normal speed (an upload still running when a lend or a zone change takes its block, a report arriving after a graph started) stay
// open long enough to be hit, so a race shows in every run instead of one in a hundred. Off (0) by default.
static void race_stress_sleep() {
    static const int max_ms = [] { const char * e = moe_opt("race-stress"); return e ? std::max(0, atoi(e)) : 0; }();
    if (max_ms <= 0) { return; }
    static thread_local uint64_t x = 0x9E3779B97F4A7C15ull ^ (uint64_t) (uintptr_t) &x;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;   // xorshift: no shared generator between the workers
    std::this_thread::sleep_for(std::chrono::microseconds((int64_t) (x % ((uint64_t) max_ms * 1000 + 1))));
}
static std::mutex g_led_mu;
static FILE *     g_led_f = nullptr;
static int64_t    g_led_n_overlap = 0;
static int64_t    g_led_flushed_us = 0;
static thread_local std::string g_led_thread;   // set by led_thread(); empty: t<n>
static std::atomic<int> g_led_n_threads{0};
static void led_thread(const std::string & name) { g_led_thread = name; }
static const std::string & led_thread_name() {
    if (g_led_thread.empty()) { g_led_thread = "t" + std::to_string(++g_led_n_threads); }
    return g_led_thread;
}
// what a kind of line does to its bytes (rw) and whether it opens or closes a window (state)
static void led_kind(const char * kind, const char *& rw, const char *& st) {
    struct k { const char * kind, * rw, * st; };
    static const k t[] = {
        { "up-start", "w", "start" }, { "up-done", "w", "done" }, { "lend", "rw", "start" }, { "graph-done", "rw", "done" },
        { "release", "o", "once" }, { "kv-zone", "o", "once" }, { "map", "o", "once" }, { "fill", "w", "once" }, { "kv-pad", "w", "once" },
        { "kv-move", "w", "once" }, { "keep-hot", "w", "once" }, { "mirror", "w", "once" }, { "zero", "w", "once" }, { "OVERLAP", "x", "once" },
        { "cuda-capture", "rw", "once" }, { "cuda-replay", "rw", "once" },
        { "chain-up", "w", "once" }, { "jit-up", "w", "once" }, { "adopt", "w", "once" },
        { "node-w", "w", "once" }, { "node-r", "r", "once" }, { "mix-read", "r", "once" }, { "split-replay", "rw", "once" }, { "verify-bad", "x", "once" },
        { "dma-start", "w", "start" }, { "dma-done", "w", "done" }, { "table", "w", "once" },
    };
    for (const auto & e : t) { if (strcmp(kind, e.kind) == 0) { rw = e.rw; st = e.st; return; } }
    rw = "?"; st = "once";
}
static int64_t led_flush_us() {
    static const int64_t v = [] { const char * e = moe_opt("ledger-flush-ms"); return (int64_t) (e ? std::max(1, atoi(e)) : 2000) * 1000; }();
    return v;
}
static void led_line(const char * card, const char * kind, size_t lo, size_t hi, const std::string & who) {
    if (!led_on()) { return; }
    const std::string & th = led_thread_name();
    std::lock_guard<std::mutex> lk(g_led_mu);
    if (!g_led_f) {
        const std::string path = moe_log_file(("pool-ledger-" + std::to_string((long long) time(nullptr)) + ".csv").c_str());
        g_led_f = fopen(path.c_str(), "w");
        if (!g_led_f) { return; }
        fprintf(g_led_f, "t_us,thread,card,kind,rw,state,lo,hi,bytes,who\n");
        LLAMA_LOG_WARN("pool-ledger: writing %s\n", path.c_str());
        std::atexit([] { std::lock_guard<std::mutex> lk2(g_led_mu); if (g_led_f) { fflush(g_led_f); } });
    }
    const char * rw = "?", * st = "once";
    led_kind(kind, rw, st);
    const int64_t now = ggml_time_us();
    fprintf(g_led_f, "%lld,%s,%s,%s,%s,%s,%zu,%zu,%zu,%s\n", (long long) now, th.c_str(), card, kind, rw, st, lo, hi, hi > lo ? hi - lo : 0, who.c_str());
    // a killed server keeps everything up to its last graph (lend) and every violation at once; the rest every ledger-flush-ms
    if (strcmp(kind, "OVERLAP") == 0 || strcmp(kind, "lend") == 0 || now - g_led_flushed_us >= led_flush_us()) {
        fflush(g_led_f);
        g_led_flushed_us = now;
    }
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

// --moe ledger: the CUDA graph cache reports each capture and replay with the scratch region its kernels use (ggml_backend_cuda_set_graph_hook_fork, set at
// pool allocation). Logged as offsets of that card's pool: a cuda-replay whose range lies outside the lend of its graph (the lend line before it) is a replay
// that reads and writes bytes it no longer owns, the bug of 2026-10-28 (fork/pool_ledger.py checks it).
static void led_cuda_graph_hook(int device, const void * graph, const char * what, const void * base, size_t cap) {
    moe_cache * mc = g_cache;
    if (!mc || !base || !led_on()) { return; }
    for (const auto & ap : mc->alias) {
        if (ap.cuda_dev != device || !ap.buf) { continue; }
        const uint8_t * b0 = (const uint8_t *) ggml_backend_buffer_get_base(ap.buf);
        const uint8_t * p  = (const uint8_t *) base;
        if (p < b0 || p >= b0 + ggml_backend_buffer_get_size(ap.buf)) { continue; }
        const size_t lo = (size_t) (p - b0);
        led_line(led_card(ap), what, lo, lo + cap, tr_fmt("cuda graph %p scratch", graph));
        return;
    }
}

// --moe ledger=2: on top of the writers above, what the graphs themselves touch: per split, one node-w line per pool range a node writes and one node-r line per input it reads
// there (card, byte range, split, node, op, tensor names). A split seen before (same uid, so the same addresses) gives one split-replay line. So `fork/pool_ledger.py LEDGER --addr
// CARD:MIB` answers who last wrote and who read a byte. Debug only: 5-10 thousand lines per new graph. The kernels' own reads of the expert blocks go by the tables: mix-read lines.
static int led_level() {
    static const int v = [] { const char * e = moe_opt("ledger"); return e ? atoi(e) : 0; }();
    return v;
}
static void led_split_cb(ggml_backend_t be, struct ggml_cgraph * g, int split_id, int n_splits, uint64_t uid) {
    moe_cache * mc = g_cache;
    if (!mc) { return; }
    const ggml_backend_dev_t dev = ggml_backend_get_device(be);
    for (auto & ap : mc->alias) {
        if (ap.dev != dev || !ap.buf) { continue; }
        const char * card = led_card(ap);
        static std::mutex mu;
        static std::unordered_set<uint64_t> seen;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (uid != 0 && !seen.insert(uid).second) {
                led_line(card, "split-replay", 0, 0, tr_fmt("split %d of %d uid %llu", split_id, n_splits, (unsigned long long) uid));
                return;
            }
        }
        const char * base = (const char *) ggml_backend_buffer_get_base(ap.buf);
        const size_t size = ggml_backend_buffer_get_size(ap.buf);
        const auto line = [&](const char * kind, const ggml_tensor * t, const std::string & who) {
            if (!t || !t->data) { return; }
            const char * d = (const char *) t->data;
            if (d < base || d >= base + size) { return; }
            const size_t lo = (size_t) (d - base);
            led_line(card, kind, lo, lo + ggml_nbytes(t), who);
        };
        for (int j = 0; j < ggml_graph_n_nodes(g); ++j) {
            const ggml_tensor * t = ggml_graph_node(g, j);
            const std::string who = tr_fmt("split %d of %d node %d %s %s", split_id, n_splits, j, ggml_op_name(t->op), t->name);
            const bool view = t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
            if (!view) { line("node-w", t, who); }
            for (int k = 0; k < GGML_MAX_SRC && !view; ++k) {
                if (t->src[k]) { line("node-r", t->src[k], who + tr_fmt(" reads src %d %s", k, t->src[k]->name)); }
            }
        }
        return;
    }
}
extern "C" { GGML_API void ggml_sched_set_split_hook_fork(void (*fn)(ggml_backend_t, struct ggml_cgraph *, int, int, uint64_t)); }
// a copy into a pool as its stream runs it (ggml_cuda_dma in ggml-cuda.cu), on CUDA's own thread: start and end in pool offsets, the writer's pointer and length with them.
// pool_ledger.py --dma lays the windows over the lend, the KV zone and each other. Copies outside a pool (weights on a card) are left out.
extern "C" {
    typedef void (*ggml_backend_dma_hook_t)(int device, const void * dst, size_t bytes, const char * who, int done);
    GGML_API void ggml_backend_set_dma_hook_fork(ggml_backend_dma_hook_t fn);
}
static void led_dma(int device, const void * dst, size_t bytes, const char * who, int done) {
    GGML_UNUSED(device);
    moe_cache * mc = g_cache;
    if (!mc) { return; }
    for (const auto & ap : mc->alias) {
        if (!ap.buf) { continue; }
        const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(ap.buf);
        const uint8_t * p    = (const uint8_t *) dst;
        if (p < base || p >= base + ggml_backend_buffer_get_size(ap.buf)) { continue; }
        const size_t lo = (size_t) (p - base);
        led_line(led_card(ap), done ? "dma-done" : "dma-start", lo, lo + bytes, who);
        return;
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
        mc->cpu_mbs_on = (int64_t) (cpu_gbs_now(mc) * 1e3);
        mc->ddr_mbs += mc->cpu_mbs_on;
    } else {
        mc->cpu_busy = false;
        mc->busy_us += t - mc->busy_t0;
        mc->busy_n++;
        mc->ddr_mbs -= mc->cpu_mbs_on;
        const int64_t dt = t - mc->busy_t0;
        if (mc->cpu_bytes_phase > 0 && dt > 0 && g_warm.load(std::memory_order_relaxed)) {
            const double gbs = (double) mc->cpu_bytes_phase / dt / 1e3;
            double & ema = mc->phase_mapped ? mc->cpu_gbs_mapped : mc->cpu_gbs_meas;
            ema = ema > 0 ? 0.95*ema + 0.05*gbs : gbs;
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
    const bool prefill = n_tokens > llama_moe_cache_max_batch_fork() || g_pp_obs;

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
    uniq_host_obs(il, ids, op, prefill);   // --moe uniqdump with the cache: every router pick (true expert ids), --moe acts: the answer's inputs
    if (g_rp.on && !prefill && g_topic_answer.load(std::memory_order_relaxed) && il >= 0 && (size_t) il < g_rp.rec.size()) {   // replay: record, follow
        for (int64_t t = 0; t < n_tokens; ++t) {
            const int32_t row = g_rp.pos[il]++;
            const bool    pl  = g_rp.active && (size_t) (row + 1) * 16 <= g_rp.play[il].size();
            for (int64_t i = 0; i < 16; ++i) {
                const int32_t id = i < n_ids ? *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]) : -1;
                g_rp.rec[il].push_back((int16_t) id);
                if (id < 0) { continue; }
                g_rp.picks++;
                if (pl) {
                    const int16_t * r = g_rp.play[il].data() + (size_t) row * 16;
                    const bool m = std::find(r, r + 16, (int16_t) id) != r + 16;
                    g_rp.matched += m;
                    if (!m && (g_rp.first_mis < 0 || row < g_rp.first_mis)) { g_rp.first_mis = row; g_rp.first_mis_layer = il; }
                }
            }
        }
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
        ls->tok++;
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
            if (get_policy() == policy::crf) {
                ls->crf[id] = ls->crf[id] * std::exp2(-(float) (ls->tok - ls->crf_t[id]) / 128.0f) + 1.0f;
                ls->crf_t[id] = ls->tok;
            } else if (get_policy() == policy::slru && ls->expert_slot[id] >= 0 && !ls->prot[id]) {   // a hit: to the protected half of what this layer holds, its oldest back to the other
                ls->prot[id] = 1; ls->n_prot++;
                if (ls->n_prot > std::max(1, ls->n_cache / 2)) {
                    int32_t oldest = -1;
                    for (int32_t x = 0; x < (int32_t) ls->prot.size(); ++x) {
                        if (ls->prot[x] && x != id && (oldest < 0 || ls->expert_last[x] < ls->expert_last[oldest])) { oldest = x; }
                    }
                    if (oldest >= 0) { ls->prot[oldest] = 0; ls->n_prot--; }
                }
            }
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
            if (!prefill) { vis_stamp(*ls, id, mc->vis_now.load(std::memory_order_relaxed)); if (!ls->heat_cnt.empty()) { ls->heat_cnt[id]++; } }   // the rampool heat: decode visits only (a prompt pass reads every expert anyway)
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
                if (g_rp.on && !prefill && g_topic_answer.load(std::memory_order_relaxed) && il >= 0 && (size_t) il < g_rp.pos.size()) {
                    g_rp.n_cpu++;
                    const int32_t row = g_rp.pos[il] - 1;   // this token's row (the replay block above already advanced pos for this batch)
                    if (g_rp.first_cpu < 0 || row < g_rp.first_cpu) { g_rp.first_cpu = row; g_rp.first_cpu_layer = il; }
                }
                {   // --moe jit-all: every expert a decode token computes on the CPU breaks the deterministic replay: say which and why (first 40)
                    static const bool ja = [] { const char * v = moe_opt("jit-all"); return v && atoi(v) > 0; }();
                    static int n_said = 0;
                    if (ja && !prefill && n_tokens <= 4 && n_said < 40) {
                        n_said++;
                        LLAMA_LOG_WARN("moe-cache: jit-all: CPU computes layer %d expert %d (step %llu, batch %lld tokens, queued %d, slots %d)\n",
                                il, id, (unsigned long long) mc->n_steps, (long long) n_tokens, (int) ls->queued[id], ls->pub.n_slots);
                    }
                }
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
    if (ls.expert_slot[j.expert] >= 0 && ls.expert_slot[j.expert] != j.slot) {   // --moe jit-all: the token needed it before this copy landed, the JIT
        ls.slot_in_flight[j.slot] = false;                                          // gave it a block of its own; this block goes back
        ls.queued[j.expert]       = 0;
        pool_release(mc, j.layer_idx, j.slot);
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

// an upload's pay-back is settled (decode hits x CPU eval time saved >= upload time): at its eviction, or at a segment end for an expert still cached (ctl_poll)
static void note_payback(moe_cache * mc, layer_state & ls, int32_t e) {
    if (ls.up_t[e] == 0) {
        return;
    }
    mc->pr[14] += ls.up_hits[e] == 0;
    const int k = ls.link;
    mc->ev_n[k]++;
    mc->ev_hits[k]    += ls.up_hits[e];
    const double cg = cpu_gbs_now(mc);
    const double cpu_us = cg > 0.0 ? (double) (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]) / (cg * 1e3) : 0.0;
    mc->ev_paid[k]    += ls.up_hits[e] * cpu_us >= ls.up_cost[e];
    mc->pr[13]        += !(ls.up_hits[e] * cpu_us >= ls.up_cost[e]);
    mc->ev_life_us[k] += (double) (ggml_time_us() - ls.up_t[e]);
    mc->ev_cost_us[k] += ls.up_cost[e];
    ls.up_t[e] = 0;
}
// an expert leaves VRAM: the eviction bookkeeping (count, protected bit, ghost list), then its pay-back
void note_evict(moe_cache * mc, layer_state & ls, int32_t e) {
    mc->n_evictions++;
    if (!ls.prot.empty() && ls.prot[e]) { ls.prot[e] = 0; ls.n_prot--; }
    if (!ls.ghost_flag.empty()) {
        ls.ghost_flag[e] = 1; ls.ghost_q.push_back(e);
        while ((int32_t) ls.ghost_q.size() > std::max(1, ls.n_cache)) { ls.ghost_flag[ls.ghost_q.front()] = 0; ls.ghost_q.pop_front(); }
    }
    note_payback(mc, ls, e);
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
                note_payback(mc, ls, e); // settled here with its hits so far; not tracked in the next segment (the expert stays cached: no eviction bookkeeping)
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
    apply_gpu_shape(k.gpu_shape);
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
        if (!ok) { break; }   // a size that failed its check never sizes an allocation
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

void resource_probe(moe_cache * mc, const llama_model & model) {
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
    if (!mc->rp_pre.empty()) {   // --moe rampool: the expert tensors are the unpopulated mapping (the pool fills after this probe), so a read of them is a page fault from the SSD, not RAM
        // (GLM 3.5 2026-11-08: the probe said RAM 22.4 GB/s against 45.0 with the tensors pinned; the planner then kept the uploads at a quarter and the CPU took the misses, decode
        // 60.5 against a possible 16+ t/s). The pieces pinned for the pool are populated RAM: the probe reads and DMAs from them instead, in slices of an expert's size.
        const size_t n = mc->layers[0].pub.up_src->nb[2];
        src.clear();
        const size_t stride = mc->rp_pre.size() >= 8 ? 7 : 1;   // one piece (the pool pins after the fill, rp_prepin): all of its 1 GiB, far past any cache
        for (ggml_backend_buffer_t b : mc->rp_pre) {
            const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(b);
            for (size_t o = 0, sz = ggml_backend_buffer_get_size(b); o + n <= sz; o += stride * n) { src.push_back({ base + o, n }); }
        }
        LLAMA_LOG_WARN("moe-cache: resource probe: reads and DMAs from the pool's %zu pinned pieces (%zu slices of %.1f MiB), not from the mapped expert tensors\n", mc->rp_pre.size(), src.size(), n / 1048576.0);
    }
    if (src.empty()) { return; }
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
                        const chunk & c = src[i++ % src.size()];   // the same sources as the CPU readers: pinned slices, with the pool its pieces (never the mapping)
                        const size_t n = std::min(c.n, (size_t) ls.pub.up_src->nb[2]);
                        ggml_backend_tensor_set(ls.pub.up_c, c.p, 0, n);
                        got[n_thr + k] += (double) n;
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
    // fork: the probe's three results (CPU RAM rate alone, with the links, threads that saturate it) are a near-static fact of this CPU and these cards: kept per CPU thread count
    // and PCI addresses of the link cards, used again at later starts (0.8 s on PC1) unless --moe probe-remeasure=1 (stored-values policy: denoised, keyed by hardware)
    std::string pkey = format("probe_cpu.%u", std::max(2u, std::thread::hardware_concurrency()));
    for (int k = 0; k < mc->n_links; ++k) { pkey += "." + dev_pci_id(mc->link_dev[k], k); }
    int n_sat = n_max;
    bool probe_stored = false;
    {
        std::string pv; double s_cpu = 0, s_tot = 0; int s_sat = 0;
        if (!moe_opt("probe-remeasure") && moe_state_get("machine", pkey, pv) && sscanf(pv.c_str(), "%lf %lf %d", &s_cpu, &s_tot, &s_sat) == 3 && s_cpu > 0 && s_tot > 0 && s_sat > 0) {
            cpu_alone = s_cpu; total = s_tot; n_sat = s_sat; probe_stored = true;
            link_alone.assign(mc->n_links, 0.0);
        }
    }
    if (!probe_stored) {
    llama_model_wait_loaded_fork(&model);   // fork: LLAMA_ASYNC_LOAD: the probe reads the weights' memory
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
    for (const auto & m : meas) {
        if (m.second.first >= 0.95 * cpu_alone) { n_sat = m.first; break; }
    }
    moe_state_set("machine", {{ pkey, format("%.3f %.3f %d", cpu_alone, total, n_sat) }});
    }   // !probe_stored
    mc->cpu_sat_threads = n_sat;
    if (!moe_opt("cpu-ram-gbs")) { knobs().cpu_gbs = cpu_alone; }
    if (!moe_opt("ram-ceiling-gbs")) { knobs().ddr_gbs = std::max(cpu_alone, total); }
    // the DDR budget gate (GATE=3) needs the budget just measured: on by default then (GLM short chat +6%, MiMo +16% in a pair); the tuner can turn it off
    if (!user_knobs().count("UPLOAD_WAIT")) { knobs().gate = 3; }
    // The link rates are the early benchmark's (early_link_gbs: measured once per process, the fastest of four copies), shared, not measured a second time; the loop above only had them alone for the CPU/links mix
    for (int k = 0; k < mc->n_links; ++k) {
        if (const double e = mc->link_dev[k] ? early_link_gbs(mc->link_dev[k]) : 0.0; e > 0.0) { link_alone[k] = e; }
    }
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
        "(saturated by %d threads), RAM budget %.1f GB/s%s\n", links.c_str(), probe_stored ? "(stored)" : rs.c_str(), cpu_alone, n_sat, knobs().ddr_gbs, probe_stored ? " (stored)" : "");
    llama_moe_decision_fork("stage", "link and CPU speed probe", "");
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
        std::vector<int64_t> queued_by_link(MAX_LINKS, 0);   // counted under wmtx BEFORE mtx: the file's lock order is wmtx then mtx (kv_blocks_apply, pp_settle, the alias lend); the reverse deadlocked
        if (knobs().auto_tune >= 2) {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            for (const auto & q : mc->todo) { queued_by_link[mc->layers[q.layer_idx].link]++; }
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
                        const int64_t queued = queued_by_link[ls.link];
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

// What a matmul reads past a slice's last row (rows that are not a multiple of 512 values, Qwen3.8's and Qwen Next's down rows of 640: see pool_layout): the block keeps room for it, and it
// must hold what the tensor itself would give: the bytes that follow the slice, which are the next expert's, behind the last expert the tensor's padding, zeros. Every writer of a block
// writes it through here. Returns how many bytes and where they come from; 0 when the class reads nothing past. A JIT upload wrote only the slice until 2026-11-02: the read past it found
// what the block held before, a graph's or the KV's bytes, and the last value of each output row came out wrong or NaN (found by --moe verify on Qwen Next).
static size_t expert_past(const moe_cache * mc, const layer_state & ls, int k, int32_t expert, const ggml_tensor * src, const char * & from) {
    if (ls.pcls < 0 || mc->pool[ls.pcls].past[k] == 0) { return 0; }
    static const std::vector<uint8_t> zeros(1u << 16, 0);
    const bool next = expert + 1 < src->ne[2] && ggml_backend_buffer_is_host(src->buffer);
    from = next ? (const char *) src->data + (size_t) (expert + 1) * src->nb[2] : (const char *) zeros.data();
    return std::min(mc->pool[ls.pcls].past[k], zeros.size());
}
// Those bytes as a tensor of their own: the class's view ends with the last block's slice, so a write or read through the view would overrun it, while the bytes are inside the block.
static ggml_tensor expert_past_view(const ggml_tensor * dst, size_t off, size_t n) {
    ggml_tensor z = {};
    z.type = GGML_TYPE_I8; z.buffer = dst->view_src ? dst->view_src->buffer : dst->buffer; z.data = (char *) dst->data + off;
    z.ne[0] = (int64_t) n; z.ne[1] = z.ne[2] = z.ne[3] = 1;
    z.nb[0] = 1; z.nb[1] = z.nb[2] = z.nb[3] = n;
    return z;
}

// Prefill preheat: the scheduler just copied the used experts of `weight` to `dev` for a prompt
// batch. Keep the ones this prompt uses more than what is cached: they are copied device-to-device
// into a slot (no extra PCIe traffic) and published at the next step. Up to LLAMA_MOE_CACHE_ADOPT
// (default 1.0) of a layer's slots per batch; only layers whose cache is on `dev`.
static const void * moe_read_fn(const ggml_tensor * weight, size_t off, size_t n, void * dst, size_t cap, void * ud);

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
        led_upload(mc, "adopt", ali, expert, slot, false);
        ls.slot_in_flight[slot] = true;
        ls.adopt_slot[expert]   = slot;
        ls.adopted.push_back(expert);
    }
    ls.adopt_mask[expert] |= (uint8_t) (1 << kind);
    *data   = (uint8_t *) c->data + (size_t) slot*c->nb[2];
    *buffer = c->buffer;
    {   // the caller copies the slice, device to device; what a matmul reads behind it comes from the host
        const char * from = nullptr;
        if (const size_t pb = expert_past(mc, ls, kind, expert, weight, from)) {
            ggml_tensor z = expert_past_view(c, (size_t) slot*c->nb[2] + (size_t) weight->nb[2], pb);
            // fork: the next expert's bytes from the pool's block (it carries them) or an O_DIRECT read, not the mapping (DS forced stream k x1: map-guard caught this read, 2026-11-10)
            const size_t sz = (size_t) weight->nb[2];
            const bool past_next = from == (const char *) weight->data + (size_t) (expert + 1)*sz;
            const char * rp = past_next && mc->rp_on ? ggml_rampool_use_fork(weight, expert) : nullptr;
            const void * rd = nullptr;
            thread_local std::vector<uint8_t> rbuf;
            if (past_next && !rp) {
                rbuf.resize(pb + 3*direct_align());
                rd = moe_read_fn(weight, (size_t) (expert + 1)*sz, pb, rbuf.data(), rbuf.size(), mc);
            }
            ggml_backend_tensor_set(&z, rp ? rp + sz : rd ? (const char *) rd : from, 0, pb);
        }
    }
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
    static const cuda_set_nosync_t f = (cuda_set_nosync_t) cuda_obs_proc("ggml_backend_fork_set_tensor_nosync_fork");
    return f;
}
// the CUDA device index of a device, -1 when the device is not one of the CUDA backend's (Vulkan, HIP, CPU, ...): decided by the registry, not the name
// the device's index inside its backend when that backend has the scratch hook (what the hook takes), else -1
static int cuda_dev_index(ggml_backend_dev_t dev) {
    return llama_fork_proc_dev(dev, "ggml_backend_fork_scratch_fork") ? llama_fork_dev_index(dev) : -1;
}
// the no-sync copy for a destination buffer: from the buffer's own backend (a cache layer on a backend without it uses the synchronous set)
static cuda_set_nosync_t cuda_set_nosync_for(ggml_backend_buffer_t buf) {
    if (!buf || ggml_backend_buffer_is_host(buf)) { return nullptr; }
    return (cuda_set_nosync_t) llama_fork_proc_dev(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)), "ggml_backend_fork_set_tensor_nosync_fork");
}
void cuda_thread_sync() {
    static void (* const f)() = (void (*)()) cuda_obs_proc("ggml_backend_fork_thread_sync_fork");
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
        led_line("?", "table", 0, v.size()*sizeof(int32_t), tr_fmt("L%d device table", pub->il));
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
    typedef const void * (*ggml_backend_moe_read_fn_t)(const ggml_tensor *, size_t, size_t, void *, size_t, void *);
    GGML_API void ggml_backend_set_moe_read_fn_fork(ggml_backend_moe_read_fn_t fn, void * user_data);   // ggml-backend.cpp (declared there too, no shared header)
    GGML_API void ggml_backend_upload_stats_fork(uint64_t * out6, bool reset);  // pool bytes, host bytes, file bytes, file reads, read us, slot wait us
}
// The O_DIRECT read thread count from the machine, not a constant (user 2026-11-02: "8 t is nice on epyc but hardly on 3600"): 1 MiB blocks on n threads for 0.2 s each, n = 1, 2, 4, 8 and
// 16 up to the hardware threads / 2; the smallest n within 7% of the best bandwidth wins, so a drive that one or two threads fill leaves the other cores to compute.
// Costs about 1 s at start. Falls back to 8 (the old default) when the file cannot be read direct.
static int read_threads_probe(const std::map<int, int> & fds, const std::string & section, bool measure = false) {   // measure: skip the stored answer (a new drive)
#ifdef _WIN32
    GGML_UNUSED(fds);
    return 8;
#else
    int dfd = -1;
    for (const auto & kv : fds) { if (kv.second >= 0) { dfd = kv.second; break; } }
    struct stat st;
    if (dfd < 0 || fstat(dfd, &st) != 0 || st.st_size < ((off_t) 1 << 30)) { return 8; }
    // kept with the model's other start measurements (links, RAM), keyed by the hardware threads: a later start skips the probe
    const std::string key = format("read_threads.%u", std::thread::hardware_concurrency());
    const std::string mkey = format("ssd_max_gbs.%u", std::thread::hardware_concurrency());
    std::string old;
    if (!measure && !section.empty() && moe_state_get(section, key, old) && atoi(old.c_str()) > 0) {
        std::string mx;
        if (moe_state_get(section, mkey, mx) && atof(mx.c_str()) > 0.0) { llama_moe_ssd_max_fork(atof(mx.c_str())); }
        LLAMA_LOG_WARN("moe-cache: read-direct: %d threads (stored probe)\n", atoi(old.c_str()));
        return atoi(old.c_str());
    }
    const size_t block = (size_t) 1 << 20, span = (size_t) st.st_size / 32 / block * block;   // each thread its own region, 32 regions in the file
    const int hw = std::max(2u, std::thread::hardware_concurrency()), cap = std::min(32, std::max(1, hw / 2));   // 32 regions in the file: no thread past the last
    std::vector<int> cands;
    for (int n = 1; n < cap; n *= 2) { cands.push_back(n); }
    cands.push_back(cap);
    std::vector<double> gbs;
    for (const int n : cands) {
        std::atomic<size_t> bytes{0};
        std::atomic<bool>   stop{false}, fail{false};
        std::vector<std::thread> th;
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < n; ++i) {
            th.emplace_back([&, i]() {
                void * buf = nullptr;
                if (posix_memalign(&buf, 4096, block) != 0) { fail.store(true); return; }
                size_t o = (size_t) i * span;
                const size_t end = o + span;
                while (!stop.load(std::memory_order_relaxed) && o + block <= end) {
                    const ssize_t r = pread(dfd, buf, block, (off_t) o);
                    if (r <= 0) { fail.store(true); break; }
                    bytes.fetch_add((size_t) r);
                    o += block;
                }
                free(buf);
            });
        }
        while (ggml_time_us() - t0 < 200000 && !fail.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        stop.store(true);
        for (auto & t : th) { t.join(); }
        if (fail.load()) { return 8; }
        gbs.push_back((double) bytes.load() / ((ggml_time_us() - t0) / 1e6) / 1e9);
    }
    const double best = *std::max_element(gbs.begin(), gbs.end());
    size_t pick = cands.size() - 1;
    for (size_t k = 0; k < cands.size(); ++k) { if (gbs[k] >= 0.93 * best) { pick = k; break; } }
    std::string row;
    for (size_t k = 0; k < cands.size(); ++k) { row += format(" %d:%.2f", cands[k], gbs[k]); }
    LLAMA_LOG_WARN("moe-cache: read-direct: probe GB/s by threads%s -> %d threads (the smallest within 7%% of the best %.2f GB/s)\n", row.c_str(), cands[pick], best);
    llama_moe_decision_fork("stage", "SSD read probe", "");
    llama_moe_ssd_max_fork(best);
    if (!section.empty()) { moe_state_set(section, {{ key, std::to_string(cands[pick]) }, { mkey, format("%.3f", best) }}); }
    return cands[pick];
#endif
}

// The scheduler's prompt uploads (ggml_backend_sched_upload_weight): the same read as read_direct in 1 MiB blocks on several threads. Threads start per chunk, as pin_prefetch_run does per tensor.
// ponytail: 7 thread starts per 32 MiB, about 2% of a core at 5 GB/s; a pool if that ever shows.
static const void * moe_read_fn(const ggml_tensor * weight, size_t off, size_t n, void * dst, size_t cap, void * ud) {
#ifdef _WIN32
    GGML_UNUSED(weight); GGML_UNUSED(off); GGML_UNUSED(n); GGML_UNUSED(dst); GGML_UNUSED(cap); GGML_UNUSED(ud);
    return nullptr;
#else
    moe_cache * mc = (moe_cache *) ud;
    const auto it = mc->rd_src.find(weight);
    if (it == mc->rd_src.end() || it->second.first < 0) { return nullptr; }
    if (!dst) { return mc; }   // only asked
    const size_t a = direct_align(), file_off = it->second.second + off, beg = file_off & ~(a - 1), end = (file_off + n + a - 1) & ~(a - 1), need = file_off + n - beg;
    uint8_t * buf = (uint8_t *) (((uintptr_t) dst + a - 1) & ~(uintptr_t) (a - 1));
    if ((size_t) (buf - (uint8_t *) dst) + (end - beg) > cap) { return nullptr; }
    const int dfd = it->second.first;
    const size_t total = end - beg, block = (size_t) 1 << 20;
#ifdef MOE_HAVE_AIO
    bool explore = false;
    const int mode = total >= ((size_t) 4 << 20) ? rd_pick_mode(mc->rd_threads, explore) : 0;
    if (mode == 1) {
        const int64_t t0 = ggml_time_us();
        if (async_read_range(dfd, buf, beg, total, need)) { rd_note(1, total, ggml_time_us() - t0); return (const void *) (buf + (file_off - beg)); }
    }
    const int64_t t_pool0 = ggml_time_us();
#endif
    const std::vector<sr_piece> pcs = sr_pieces(total);   // the threads take the pieces in the spaced order of llama-spaced-read-fork.h
    std::atomic<size_t> next{0};
    std::atomic<bool>   fail{false};
    auto work = [&]() {
        for (;;) {
            const size_t pi = next.fetch_add(1);
            if (pi >= pcs.size() || fail.load(std::memory_order_relaxed)) { return; }
            const size_t o = pcs[pi].off, want = pcs[pi].n;
            const ssize_t r = pread(dfd, buf + o, want, (off_t) (beg + o));
            if (r < 0 || ((size_t) r < want && o + (size_t) r < need)) { fail.store(true); return; }   // short only past the end of the file
        }
    };
    const int n_th = total <= block ? 0 : std::min<int>(mc->rd_threads - 1, (int) (total / block));
    std::vector<std::thread> th;
    th.reserve(n_th);
    for (int i = 0; i < n_th; ++i) { th.emplace_back([&work]() { unpin_helper(); work(); }); }
    work();
    for (auto & x : th) { x.join(); }
#ifdef MOE_HAVE_AIO
    if (total >= ((size_t) 4 << 20) && !fail.load()) { rd_note(0, total, ggml_time_us() - t_pool0); }
#endif
    return fail.load() ? nullptr : (const void *) (buf + (file_off - beg));
#endif
}

extern "C" {
    typedef int (*ggml_backend_moe_where_fn_t)(const ggml_tensor *, int32_t, int32_t, ggml_backend_dev_t, const void **, ggml_backend_buffer_t *, void *);
    GGML_API void ggml_backend_set_moe_where_fn_fork(ggml_backend_moe_where_fn_t fn, void * user_data);   // ggml-backend.cpp (declared there, no shared header)
}
static void moe_where_count(int64_t range, int64_t found, int64_t inflight, int64_t wrongdev);
#define GGML_MOE_ELSEWHERE ((const void *) 1)   // ggml-backend.cpp: this expert needs no staging, no id of the graph reads it on this card
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
    if (c && c->buffer && g_pp_mix_built && lk == 0 && ls.link > 0) {
        // the prompt mix: the home card computes what its table says it holds (the table is frozen for the pass, llama_moe_cache_pp_freeze_fork); those experts take no staging
        const int32_t * t = (const int32_t *) ls.pub.host_table->data;
        for (int32_t e = std::max<int32_t>(lo, 0); e < hi && e < (int32_t) ls.expert_slot.size(); ++e) {
            if (t[e] >= 0 && t[e] < ls.pub.n_slots) { data[e - lo] = GGML_MOE_ELSEWHERE; }
        }
    }
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

// Bumped when a graph's place in a card's pool changes: the dense split's copies in front of the graph grew or shrank, or a graph starts at another offset.
// A graph reused from before keeps its tensors where they were allocated, so it is not reused across a change (llama_moe_cache_graph_reusable_fork), the
// same rule as the KV zone's epoch. Until 2026-10-28 only a moved KV zone stopped a reuse.
static std::atomic<int> g_lend_epoch{0};
static int g_lend_epoch_built = 0;

// the dense split's copies sit at the start of dev's pool buffer (llama_dsplit_mirror_place_fork): the pass planner keeps room for them
void llama_moe_cache_alias_mirror_fork(ggml_backend_dev_t dev, size_t bytes) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev == dev) {
            if (ap.mirror_b != bytes) { g_lend_epoch++; }
            ap.mirror_b = bytes;
        }
    }
}

// before a graph is allocated (and before a reserve): its tensors start at off in dev's pool buffer, the lent range covers [0, off) too
void llama_moe_cache_alias_offset_fork(ggml_backend_dev_t dev, size_t off) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    for (auto & ap : mc->alias) {
        if (ap.dev == dev) {
            if (ap.base_off != off) { g_lend_epoch++; }
            ap.base_off = off;
        }
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
    static nodes_fn_t fn = (nodes_fn_t) llama_fork_proc("ggml_backend_fork_nodes_scratch_fork");
    g_scr_splits++;
    return fn ? fn(*(const int *) user, nodes, n) : 0;
}
static size_t alias_scratch(const moe_cache::alias_pool & ap, ggml_backend_sched_t sched) {
    if (ap.cuda_dev < 0 && cuda_dev_index(ap.dev) < 0) {
        return 0;
    }
    using copy_fn_t = size_t (*)(ggml_backend_t, ggml_backend_t, const ggml_tensor *, const ggml_tensor *);
    static copy_fn_t copy_fn = (copy_fn_t) llama_fork_proc("ggml_backend_fork_copy_scratch_fork");   // the inputs other cards copy from this one: a gappy view is gathered here first
    const int64_t t0 = ggml_time_us();
    int dev = ap.cuda_dev >= 0 ? ap.cuda_dev : cuda_dev_index(ap.dev);
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
        // this graph holds one growth step of KV (get_size_reserve), its masks too: at more cells it needs more, like a graph that ran (need_at_cells). With no cells noted it
        // did not grow, and the fit through it and the graphs that ran gave a fixed part of 2.2 GiB at 98k cells: the rest of a 128k prompt ran in passes of 64 tokens at
        // 189 t/s instead of about 4400 tokens at the 2900 t/s of the big passes (Qwen3.6, 4.5 GiB per card, 2026-11-01)
        if (graph_bytes >= nt) { ap.need_cells[(int32_t) n_tokens] = ap.kv_step; }
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
void llama_moe_cache_kv_pool_expected_fork(bool on) {
    g_kv_pool_expected = on;
    // --moe fattn-kernel=vec|tile|mma: the kernel of decode-sized attention nodes, where it can run them (fork/docs/fork-knobs.md); values of BEST_FATTN_KERNEL_*
    static const int k = [] { const char * e = moe_opt("fattn-kernel"); return !e ? 0 : !strcmp(e, "vec") ? 100 : !strcmp(e, "tile") ? 200 : !strcmp(e, "mma") ? 400 : 0; }();
    if (k) { if (auto fn = (void (*)(int)) cuda_obs_proc("ggml_backend_fork_fattn_force_fork")) { fn(k); } }
    if (const char * e = moe_opt("fattn-rows"); e && atoi(e) == 0) { if (auto fn = (void (*)(int)) cuda_obs_proc("ggml_backend_fork_fattn_rows_fork")) { fn(0); } }   // a gather converts all of quantized K/V
}
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
                   who, ggml_backend_dev_name(ap.dev), bad, ap.kv_start, kvz_end(ap), ap.kv_fixed, (size_t) ap.kv_h, ap.kv_row, lend_end, ap.tail_end);
    }
}

// A graph's lend grows with the cells too: its copies of the attention masks (pass tokens x cells x 2 bytes for an f16 mask) are inputs the scheduler copies into
// each card's compute buffer. The bytes per pass token per cell are the host-buffer fit's (llama-context.cpp, the same inputs on the host side), set at startup.
// need_tok holds the most a size borrowed, measured at some cells: at more cells it needs that much more (MiMo 255k: a 2368-token pass carries a 1.15 GB mask
// on each card, which a size measured at 32k cells did not have).
static std::atomic<double> g_mask_bpc { 0.0 };
void llama_moe_cache_mask_bpc_fork(double bytes_per_token_cell) { g_mask_bpc.store(std::max(0.0, bytes_per_token_cell)); }
static std::atomic<bool> g_mask_tail { false };   // a graph's mask covers only its pass's own cells (llama_kv_cache::kq_mask_k0)
void llama_moe_cache_mask_tail_fork() { g_mask_tail.store(true, std::memory_order_relaxed); }
static size_t need_at_cells(const moe_cache::alias_pool & ap, int32_t n_tokens, size_t need, size_t cells) {
    const auto it = ap.need_cells.find(n_tokens);
    const size_t c0 = it != ap.need_cells.end() ? it->second : cells;
    if (cells <= c0) { return need; }
    const double pc = (double) (n_tokens < 64 ? ap.kv_lend_pc_dec : ap.kv_lend_pc);   // the f16 copies of quantized K/V
    double cols = (double) (cells - c0);   // the mask's columns the graph gains
    if (g_mask_tail.load(std::memory_order_relaxed)) {   // never more than the pass's own cells and the padding on both sides
        const double most = (double) GGML_PAD(n_tokens, 256) + 256.0;
        cols = std::max(0.0, std::min((double) cells, most) - std::min((double) c0, most));
    }
    return need + (size_t) (g_mask_bpc.load() * (double) n_tokens * cols + pc * (double) (cells - c0));
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
    const size_t h_min = kvz_read_cells(need, pad, cmax);
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
        const size_t ls = kv_gap_place(ap, bmax, lend_min, l, kvz_bytes(ap, h));
        if (ls != (size_t) -1) { l = ls; }
    }
    if (h == ap.kv_h && l == ap.kv_start) { return false; }
    if (dry) { return true; }
    const size_t lo_old = ap.kv_start, h_old = ap.kv_h, lo = l, hi = l + kvz_bytes(ap, h);
    const bool   first  = h_old == 0;   // nothing laid out yet: no rows to move, the fixed part is zeroed below
    const size_t n_used = first ? 0 : kvz_bytes(ap, std::min(cells, std::min(h, h_old)));   // the fixed part and the growing rows in use: one contiguous range
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
        const std::string zone = format("kv-tail %s: zone %zu -> %zu cells at %.1f -> %.1f MiB (%zu B per cell, %zu cells in use%s), %zu blocks taken (%zu experts evicted), %zu given back",
                ggml_backend_dev_name(ap.dev), h_old, h, lo_old / 1048576.0, lo / 1048576.0, ap.kv_row, cells,
                lo != lo_old ? tr_fmt(", %.1f MiB of rows moved", n_used / 1048576.0).c_str() : "", n_take, n_evict, n_give);
        LLAMA_LOG_WARN("moe-cache: %s\n", zone.c_str());
        llama_moe_stats_log_fork("KVZ", zone.c_str());
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
            e = kvz_read_cells(e, r.pad);
            eff = std::max(eff, e);
            cells = std::max(cells, c);
            cells_max = std::max(cells_max, r.cells_max);
        }
        const size_t step = std::max<size_t>(256, ap.kv_step);
        size_t h = (allow_shrink || ap.kv_h == 0) ? step : ap.kv_h;
        while (eff + step / 2 > h) { h += step; }
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
        if (ap.tail_end <= lr + kvz_bytes(ap, eff)) {
            const size_t floor_l = kv_lend_round(ap, ap.dec_need);   // what a decode graph borrows (measured), on its block edge
            const size_t zone_b  = kvz_bytes(ap, eff);
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
    size_t u_all = 1;   // (from 256 a q8_0 row of 5440 B took 8704: lcm(256, 272))
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
        vec = vec && !kr.gather;   // a gather over quantized K/V: decode has the copies too, though only the rows it reads are written (fattn-common.cuh)
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i].v) { continue; }
            if (s[i].lay < (int32_t) kr.kind.size() && kr.kind[s[i].lay] != 0 && kr.out_next[s[i].lay]) { continue; }   // --moe attn-split: the other layout's entry: no graph reads it
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
        kr.layout = r.layout; kr.retype = r.retype; kr.rot = r.rot; kr.tk = r.type_k; kr.tv = r.type_v; kr.convertible = r.convertible && r.layout && r.retype;
        kr.kind = r.kind; kr.l_il = r.l_il; kr.l_part = r.l_part; kr.parts_set = r.parts_set; kr.gather = r.gather;
        if (!r.mask0.empty()) { kr.out = r.mask0; kr.out_next = r.mask0; static uint8_t no_place[8]; kr.host.assign(2 * r.mask0.size(), { no_place, r.row }); }
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
            llama_moe_decision_fork("stage", "KV size fit computed", "");
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
        const size_t margin = (size_t) (knobs().margin_mb > 0 ? knobs().margin_mb : 484) * 1024 * 1024;
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
        out.pool += ap.tail_end;   // the pool of this card: what the experts and the KV zone share
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

struct kv_cv_part { const std::vector<llama_kv_tail_fork::seg> * so; const std::vector<llama_kv_tail_fork::seg> * sn; size_t so_add, sn_add; const std::function<ggml_tensor *(int32_t, bool)> * rot; };

// the rotation a segment needs when it changes type: rows are rotated while quantized (llama_kv_cache, n_rot_k_tier), so only a change between float and quantized rotates
static ggml_tensor * kv_cv_rot(const kv_cv_part & p, size_t i) {
    const auto & o = (*p.so)[i];
    const auto & w = (*p.sn)[i];
    return p.rot && *p.rot && ggml_is_quantized(o.type) != ggml_is_quantized(w.type) ? (*p.rot)(o.lay, o.v) : nullptr;
}

// rows [0, n_rows) of a run: old row i at byte src + i*ss of the buffer, new row i at dst + i*ds, through the staging range [stg, stg + stg_b). up: descending chunks.
static bool kv_convert_run(ggml_backend_t be, moe_cache::alias_pool & ap, size_t src, size_t ss, size_t dst, size_t ds, size_t n_rows,
                           const std::vector<kv_cv_part> & parts, size_t stg, size_t stg_b, bool up, size_t & moved) {
    if (n_rows == 0) { return true; }
    size_t f32_row = 0, n_seg = 0, n_rot = 0;
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.so->size(); ++i) {
            n_seg++;
            if ((*p.so)[i].type == (*p.sn)[i].type) { continue; }
            const bool rotates = kv_cv_rot(p, i) != nullptr;
            f32_row += (size_t) (*p.so)[i].ne0 * (rotates ? 8 : 4);
            n_rot   += rotates ? 1 : 0;
        }
    }
    if (n_seg == 0) { return true; }
    const size_t slack = (n_seg + n_rot + 2) * 256;
    if (stg_b <= slack) { return false; }
    const size_t chunk = std::min(n_rows, (stg_b - slack) / (ds + f32_row));
    if (chunk == 0) { return false; }
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(ap.buf);
    const size_t n_chunks = (n_rows + chunk - 1) / chunk, n_nodes = n_seg * 4 + n_rot * 2 + 8;
    for (size_t k = 0; k < n_chunks; ++k) {
        const size_t ci = up ? n_chunks - 1 - k : k;
        const size_t a = ci * chunk, n = std::min(chunk, n_rows - a);
        ggml_init_params ip = { (n_seg + n_rot) * 6 * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_nodes, false) + 4096, nullptr, true };
        ggml_context * cx = ggml_init(ip);
        if (!cx) { return false; }
        ggml_cgraph * gf = ggml_new_graph_custom(cx, n_nodes, false);
        size_t tmp = GGML_PAD(stg + n * ds, 256);   // f32 temporaries behind the converted rows, one per segment (no memory shared inside a graph)
        auto mk = [&](ggml_type t, int64_t ne0, size_t at, size_t stride, int64_t rows = 0) {
            if (rows == 0) { rows = (int64_t) n; }
            ggml_tensor * x = ggml_new_tensor_2d(cx, t, ne0, rows);
            x->nb[1] = stride; x->nb[2] = x->nb[3] = stride * (size_t) rows;
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
                    const size_t row32 = (size_t) o.ne0 * 4, at32 = tmp;
                    tmp = GGML_PAD(tmp + row32 * n, 256);
                    ggml_tensor * x = ggml_cpy(cx, mk(o.type, o.ne0, s_at, ss), mk(GGML_TYPE_F32, o.ne0, at32, row32));
                    if (ggml_tensor * rot = kv_cv_rot(p, i)) {   // the same blocks as the graph's rotation: a row is contiguous, n_rot floats at a time
                        const int64_t nr = rot->ne[0];
                        GGML_ASSERT(o.ne0 % nr == 0 && "kv-retype: a row that is not a whole number of rotation blocks");
                        ggml_build_forward_expand(gf, x);
                        const int64_t blocks = o.ne0 / nr * (int64_t) n;
                        ggml_tensor * ro = ggml_mul_mat(cx, rot, mk(GGML_TYPE_F32, nr, at32, (size_t) nr * 4, blocks));
                        ggml_mul_mat_set_hint(ro, GGML_HINT_SRC0_IS_HADAMARD);
                        ro->data = base + tmp; ro->buffer = ap.buf;
                        ggml_build_forward_expand(gf, ro);
                        x = mk(GGML_TYPE_F32, o.ne0, tmp, row32);
                        tmp = GGML_PAD(tmp + row32 * n, 256);
                    }
                    ggml_build_forward_expand(gf, ggml_cpy(cx, x, mk(w.type, w.ne0, d_at, ds)));
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

static bool g_kv_rows_by_caller = false;   // llama_moe_cache_kv_parts_fork writes the rows itself: the retype changes the layout only, nothing is staged or moved on the card
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
        // the staging is for rows to convert: an empty KV (a request's start) has none, and with the dense split's copies in front there may be no blocks to stage in
        bool has_rows = cp.cells > 0;
        for (const auto & r : ap.kvt) { has_rows = has_rows || r.fixed; }
        if (has_rows && !g_kv_rows_by_caller && !kv_stage_range(ap, cp.s_lo, cp.s_hi)) { say("kv-fit: %s: no lend in front of the KV to stage the conversion", dn); return false; }
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
        b_old += kvz_bytes(ap, cp.cells);
        kv_blocks_apply(mc, ap, cp.s_lo, cp.s_hi, true, false, nt, ng, ne);   // the staging: its experts out (uploads into it dropped or awaited)
        if (cp.up) { kv_blocks_apply(mc, ap, Z, Z + cp.F + cp.h * cp.R, true, false, nt, ng, ne); }   // converting up: the new extent's experts out before rows land there
        const size_t nr = ap.kvt.size();
        std::vector<std::vector<llama_kv_tail_fork::seg>> so(nr), sn(nr);
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; auto & q = cp.nk[i];
            if (r.layout) {
                r.layout(r.tk, r.tv, &r.out, &so[i]); q.layout(q.tk, q.tv, &q.out_next, &sn[i]);
            }
            else          { so[i] = sn[i] = { { GGML_TYPE_F16, (int64_t) (r.row / 2), 0, false, 0, 0 } }; }   // no layout: the row moves as bytes
        }
        // the RAM part: layers leaving the row for host memory (q.out_next, places in q.host) are copied there first (the cells in use, by the card: its kernels
        // write host memory directly), then dropped from the old segments: the compaction below moves only the layers that stay
        for (size_t i = 0; i < nr; ++i) {
            auto & r = ap.kvt[i]; auto & q = cp.nk[i];
            if (g_kv_rows_by_caller || r.fixed || q.out_next.empty()) { continue; }
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
                a->nb[1] = ap.kv_row; a->nb[2] = a->nb[3] = ap.kv_row * cp.cells; a->data = base + kvz_row_at(ap, 0) + r.prefix + sg.off; a->buffer = ap.buf;
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
            if (r.fixed) { fixed_runs.push_back({ Z + r.prefix, r.row, Z + q.prefix, q.row, r.cells_max, { { &so[i], &sn[i], 0, 0, &r.rot } } }); }
            else         { grow.parts.push_back({ &so[i], &sn[i], r.prefix, q.prefix, &r.rot }); }
        }
        std::vector<run *> order;   // shrinking: lowest addresses first (the fixed part, then the growing rows); growing: the reverse
        for (auto & fr : fixed_runs) { order.push_back(&fr); }
        order.push_back(&grow);
        if (cp.up) { std::reverse(order.begin(), order.end()); }
        for (run * rr : order) {
            if (g_kv_rows_by_caller) { break; }
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
bool llama_moe_cache_kv_any_host_fork() {
    moe_cache * mc = g_cache;
    if (!mc) { return false; }
    for (const auto & ap : mc->alias) { if (kv_any_out(ap)) { return true; } }
    return false;
}

static bool kv_any_out(const moe_cache::alias_pool & ap) {
    for (const auto & r : ap.kvt) {
        for (size_t i = 0; i < r.out.size(); ++i) { if (r.out[i] && !(i < r.kind.size() && r.kind[i] != 0)) { return true; } }   // (an attn-split entry out of the row is no host layer)
    }
    return false;
}

// --moe attn-split: the rows of the attention layers that can split move between their whole entries (all KV heads on the layer's home card) and their parts (each card its heads).
// Every card's zone is read into RAM (pageable memory: no CUDA allocation at run time, nothing staged in the pool), the zones take their new layout (retype: layers that stay keep
// their rows), and the rows that enter are put together from the images and written back: a whole K or V row is its parts' rows side by side. No graph runs meanwhile.
// Only when a request passes the threshold (llama_context::decode): nothing to read for a new prompt; a prefix in use costs a read and a write of its rows over each card's link.
static ggml_tensor kv_bytes_of_pool(moe_cache::alias_pool & ap, size_t at, size_t n) {
    ggml_tensor z = {};
    z.type = GGML_TYPE_I8; z.buffer = ap.buf; z.data = (uint8_t *) ggml_backend_buffer_get_base(ap.buf) + at;
    z.ne[0] = (int64_t) n; z.ne[1] = z.ne[2] = z.ne[3] = 1;
    z.nb[0] = 1; z.nb[1] = z.nb[2] = z.nb[3] = n;
    return z;
}

bool llama_moe_cache_kv_parts_fork(ggml_backend_sched_t sched, bool on, char * note, size_t note_size) {
    auto say = [&](const char * fmt, auto... args) { if (note && note_size) { snprintf(note, note_size, fmt, args...); } };
    moe_cache * mc = g_cache;
    if (!mc) { say("attn-split: no expert pool"); return false; }
    const int64_t t0 = ggml_time_us();
    if (sched) { ggml_backend_sched_synchronize(sched); }
    struct image { moe_cache::alias_pool * ap = nullptr; size_t cells = 0, row = 0; std::unique_ptr<uint8_t[]> b; };
    struct place { size_t im, at, st; };   // an entry's K or V in the old rows: which image, its place in the row, its bytes per row
    std::vector<image> img;
    using key_t = std::tuple<void *, int32_t, bool, int32_t>;
    const auto key_of = [](const moe_cache::alias_pool::kvt_run & r, const llama_kv_tail_fork::seg & q) -> key_t {   // a layer that can split: by model layer and part (-1 whole); a plain one: as it is
        const int8_t k = q.lay < (int32_t) r.kind.size() ? r.kind[q.lay] : 0;
        return k == 0 ? key_t { r.owner, q.lay, q.v, -2 } : key_t { r.owner, r.l_il[q.lay], q.v, k == 1 ? -1 : r.l_part[q.lay] };
    };
    std::map<key_t, place> old;
    size_t bytes = 0;
    for (auto & ap : mc->alias) {
        bool any = false;
        for (auto & r : ap.kvt) { any = any || !r.kind.empty(); }
        if (!any) { continue; }
        image im; im.ap = &ap; im.row = ap.kv_row;
        for (auto & r : ap.kvt) { if (!r.fixed) { im.cells = std::max(im.cells, r.rows ? r.rows() : 0); } }
        for (auto & r : ap.kvt) {
            if (r.fixed || !r.layout) { continue; }
            std::vector<llama_kv_tail_fork::seg> cur;
            r.layout(r.tk, r.tv, &r.out, &cur);
            for (const auto & q : cur) { old[key_of(r, q)] = { img.size(), r.prefix + q.off, ggml_row_size(q.type, q.ne0) }; }
        }
        const size_t n = im.cells * im.row;
        if (n > 0) {
            im.b.reset(new (std::nothrow) uint8_t[n]);
            if (!im.b) { say("attn-split: no room in RAM for %.1f MiB of rows", n / 1048576.0); return false; }
            const ggml_tensor z = kv_bytes_of_pool(ap, ap.kv_start + ap.kv_fixed, n);
            ggml_backend_tensor_get(&z, im.b.get(), 0, n);
            bytes += n;
        }
        img.push_back(std::move(im));
    }
    if (img.empty()) { say("attn-split: no layer can split"); return false; }
    static uint8_t no_place[8];   // an entry out of the row: no graph reads it
    for (auto & im : img) {
        for (auto & r : im.ap->kvt) {
            if (r.kind.empty() || r.fixed) { continue; }
            r.out_next.resize(r.kind.size(), 0);
            r.host.assign(2 * r.kind.size(), { no_place, r.row });
            for (size_t i = 0; i < r.kind.size(); ++i) {
                if (r.kind[i] == 1) { r.out_next[i] = on ? 1 : 0; }
                if (r.kind[i] == 2) { r.out_next[i] = on ? 0 : 1; }
            }
        }
    }
    char why[384] = { 0 };
    g_kv_rows_by_caller = true;
    const bool laid_out = llama_moe_cache_kv_retype_fork(sched, 3, why, sizeof(why));
    g_kv_rows_by_caller = false;
    if (!laid_out) {
        for (auto & im : img) { for (auto & r : im.ap->kvt) { if (!r.kind.empty() && !r.fixed) { r.out_next = r.out; } } }
        say("attn-split: %s", why);
        return false;
    }
    for (auto & im : img) {
        auto & ap = *im.ap;
        const size_t R = ap.kv_row, n = im.cells * R, z_at = ap.kv_start + ap.kv_fixed;
        if (n == 0) { continue; }
        std::unique_ptr<uint8_t[]> nb(new (std::nothrow) uint8_t[n]);
        if (!nb) { GGML_ABORT("attn-split: no room in RAM for the new rows (the KV is lost)"); }
        const ggml_tensor z = kv_bytes_of_pool(ap, z_at, n);
        const auto rows_from = [&](const place & w, size_t w_add, size_t to, size_t st) {
            const image & si = img[w.im];
            const size_t c_n = std::min(im.cells, si.cells);
            for (size_t c = 0; c < c_n; ++c) { memcpy(nb.get() + c * R + to, si.b.get() + c * si.row + w.at + w_add, st); }
        };
        for (auto & r : ap.kvt) {
            if (r.fixed || !r.layout) { continue; }
            std::vector<llama_kv_tail_fork::seg> cur;
            r.layout(r.tk, r.tv, &r.out, &cur);
            for (const auto & q : cur) {
                const int8_t k = q.lay < (int32_t) r.kind.size() ? r.kind[q.lay] : 0;
                const size_t st = ggml_row_size(q.type, q.ne0), to = r.prefix + q.off;
                if (k == 0) { rows_from(old.at(key_of(r, q)), 0, to, st); continue; }   // a plain layer: its place in the row may have changed
                const int32_t il = r.l_il[q.lay];
                if (k == 2) {   // a part: its bytes of the whole row
                    rows_from(old.at(key_t { r.owner, il, q.v, -1 }), (size_t) r.l_part[q.lay] * st, to, st);
                } else {        // a whole layer: its parts' rows side by side
                    size_t off = 0;
                    for (int32_t p = 0; ; ++p) {
                        const auto it = old.find(key_t { r.owner, il, q.v, p });
                        if (it == old.end()) { break; }
                        rows_from(it->second, 0, to + off, it->second.st);
                        off += it->second.st;
                    }
                    GGML_ASSERT(off == st && "attn-split: a whole row is not its parts' rows");
                }
            }
        }
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(&z), nb.get(), 0, n);
    }
    for (auto & im : img) { for (auto & r : im.ap->kvt) { if (r.parts_set) { r.parts_set(on); } } }
    g_kv_epoch.fetch_add(1);   // graphs built before read the other entries
    say("attn-split: %zu cells of KV moved to %s in %.1f ms (%.1f MiB through RAM)", img[0].cells, on ? "the parts on their cards" : "the whole layers on their home cards", (ggml_time_us() - t0) / 1000.0, bytes / 1048576.0);
    return true;
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
                return lend + kvz_bytes(ap, c) <= ap.tail_end && kv_gap_fit(ap, bmax, lend, ap.kv_row) >= c;   // c is in whole pads: the attention reads exactly c rows   // one pad of margin: the lend is a fit through measured sizes and the real graph came 0.3% bigger (Qwen3.6 PC1, gap 13780 for 13824 cells)
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

// The bytes record r's reads cover while cells [0, max_cell] are in use: for a fixed record (a sliding-window ring with its own row) the cells themselves, for a
// growing record whole pads of combined rows (flash attention reads up to the pad). One function for the check (kv_tail_covers_fork) and its message (kv_tail_why_fork):
// until 2026-10-28 the message described the unpadded range while the check tested the padded one.
static void kv_read_range(const moe_cache::alias_pool & ap, const moe_cache::alias_pool::kvt_run & r, size_t max_cell, size_t & lo, size_t & hi) {
    if (r.fixed) { lo = ap.kv_start + r.prefix; hi = lo + (max_cell + 1) * r.row; return; }
    lo = kvz_row_at(ap, 0);
    hi = kvz_row_at(ap, kvz_read_cells(max_cell + 1, r.pad, r.cells_max));
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
                size_t lo, hi;
                kv_read_range(ap, r, max_cell, lo, hi);
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
                if (r.fixed) {
                    if (max_cell + 1 > r.cells_max) { return false; }
                } else {
                    // the attention reads up to the cells rounded up to the pad: all of those rows must be the zone's, not just the cells written
                    if (kvz_read_cells(max_cell + 1, r.pad, r.cells_max) > ap.kv_h) { return false; }
                    if (kv_any_out(ap) && max_cell + 1 > ap.host_cells) { return false; }
                }
                size_t lo, hi;   // every block under [lo, hi) must be the KV's (owner -3)
                kv_read_range(ap, r, max_cell, lo, hi);
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
            if (q.fixed ? (max_cell + 1) > q.cells_max : kvz_read_cells(max_cell + 1, ar.pad) > kv_arena_cells(ar)) { return false; }
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
        const size_t hi = kvz_read_cells(cells, pad, ap.kv_h);
        const size_t lo = std::max(ap.kv_zhi, cells);
        if (hi <= lo) { continue; }
        ggml_tensor z = {};
        z.type = GGML_TYPE_I8; z.buffer = ap.buf;
        z.data = (uint8_t *) ggml_backend_buffer_get_base(ap.buf) + kvz_row_at(ap, lo);
        led_line(led_card(ap), "kv-pad", kvz_row_at(ap, lo), kvz_row_at(ap, hi), tr_fmt("rows %zu-%zu zeroed", lo, hi));
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
// llama_context::process_ubatch: the graph of a pass did not fit the blocks the pool lent (ggml_gallocr_reserve: "graph needs X MiB, the external buffer holds Y"); decode then
// takes the smaller-pass path as for a refusal (DeepSeek 64k, 2026-11-08: a 16384-token pass at 73k cells needed 25.4 GiB against a 20.2 GiB lend and the request died with HTTP 500)
void llama_moe_cache_alias_refused_fork(uint32_t n_tokens) {
    if (g_alias_err.empty()) { g_alias_err = "the compute graph of a " + std::to_string(n_tokens) + "-token pass did not fit the blocks the pool lent (its attention grows with the cells)"; }
}

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
        static size_t (*scratch_fn)(int, void *, size_t, ggml_backend_t) = (size_t (*)(int, void *, size_t, ggml_backend_t)) llama_fork_proc("ggml_backend_fork_scratch_fork");
        if (ap.cuda_dev < 0 && scratch_fn) {
            ap.cuda_dev = cuda_dev_index(ap.dev);
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
            if (moe_log_has("w")) {   // what a graph of this size borrows, against what its size held so far
                LLAMA_LOG_WARN("moe-cache: lend %s: %d tokens at %zu cells: graph %.1f MiB + CUDA scratch %.1f MiB behind %.1f MiB, staging %.1f MiB (this size so far %.1f MiB at %zu cells)\n", ggml_backend_dev_name(ap.dev), n_tokens, cells_now,
                        peak / 1048576.0, scratch / 1048576.0, ap.base_off / 1048576.0, stage / 1048576.0, nt / 1048576.0, ap.need_cells.count(n_tokens) ? ap.need_cells[n_tokens] : (size_t) 0);
            }
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
            size_t cells = 0;
            for (auto & r : ap.kvt) { if (!r.fixed) { cells = std::max(cells, r.rows ? r.rows() : 0); } }
            {   // the attention reads the cells in use rounded up to the pad: the zone must hold those rows too (OLMoE on W10, 2026-10-18: 33568 cells in use, a zone of 33740 after the lend grew, padded read 33792, abort in set_input_k_idxs)
                size_t pad = 1, cmx = 0;   // cmx: the most cells of a growing record (a compressed cache's fewer rows never cap the others)
                for (auto & r : ap.kvt) { if (!r.fixed) { pad = std::max(pad, r.pad); cmx = std::max(cmx, r.cells_max); } }
                cells = kvz_read_cells(cells, pad, cmx);
            }
            // the zone moves behind exactly this graph's lend, its start on the block edge the lend ends on (no headroom: a bigger graph later moves it again, a
            // device copy of the rows in use, ~2.5 ms per GB; 25% headroom pushed MiMo's zone 3.4 GiB further back and the decode then found no room for its cells)
            const size_t keep = kvz_bytes(ap, std::max(cells, (size_t) 1));   // the zone as it is now must still fit behind the lend
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
            bool stuck = false;
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
                if (spin >= 200000) { stuck = true; break; }   // 10 s: a worker still copies into these blocks
                g_ac.drain_us += 50;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            if (stuck) {   // refuse this pass (the caller fails the request, the server lives); the blocks stay unlent, so the late copy lands in a cache block, not under the graph
                LLAMA_LOG_ERROR("moe-cache: compute alias %s: an upload into blocks %d-%d of class %d did not finish in 10 s, this pass is refused\n", ggml_backend_dev_name(ap.dev), (int) blocked, (int) need, (int) sg.pcls);
                if (g_alias_err.empty()) { g_alias_err = std::string("compute alias: an upload into the borrowed blocks did not finish in 10 s on ") + ggml_backend_dev_name(ap.dev); }
                continue;
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
        // the previous graph may still read those blocks and the tables that point into them: it finishes first, then the tables change (flushed before
        // the sync, a still-running previous graph saw the evicted experts as uncached in its later layers and dropped them from that token)
        ggml_backend_sched_synchronize(sched);
        flush_tables();   // this graph's inputs are written next
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
// --moe dynhot=full|store|read|off (1 = full, 0 = off; default off since 2026-11-02; store (user 2026-10-28: "dynhot full/onlystore/onlyread/off meaning only store uses ini hotmoes
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
    int                 k = 1;                              // records merged for the lookup (--moe dynhot-k)
    int32_t             prev_n = 0;       // tokens of the previous decode call: a prompt starts where a call of more than a step follows a step
    bool                after_prompt = false;   // a prompt began: its first decode step drops the experts foreign to the topic and the request
    uint64_t h0 = 0, m0 = 0;   // decode hits and misses when the request began
    enum { OFF, FULL, STORE, READ };
    int  mode = STORE;
    bool steer = false;   // this request: stored maps pick the cache's experts (full or read, and a map found)
    bool use()  const { return mode == FULL || mode == READ; }    // the maps pick the cache's experts
    bool keep() const { return mode == FULL || mode == STORE; }   // requests are stored
} g_topic;

namespace {
static std::string topic_store_dir() { return g_topic.store.path; }   // declared above the uniqdump section (same namespace)
}

namespace {
static std::string replay_path(uint64_t h) {
    char hx[32];
    snprintf(hx, sizeof hx, "%016llx", (unsigned long long) h);
    return (std::filesystem::path(g_topic.store.path) / "trace" / (std::string(hx) + ".ids")).string();
}

// the answer starts (the prompt's last pass is over): record from here, and play the prompt's stored trace if there is one. mc->mtx held.
static void replay_begin(moe_cache * mc) {
    auto & r = g_rp;
    if (!r.on || g_topic.store.path.empty()) { return; }
    if (const char * v = moe_opt("replay-ahead")) { r.ahead = std::max(1, atoi(v)); }
    if (const char * v = moe_opt("replay-frac"))  { r.frac  = std::max(0.05, atof(v)); }
    int nl = 0;
    size_t ne = 0;
    for (const auto & ls : mc->layers) { nl = std::max(nl, ls.pub.il + 1); ne = std::max(ne, ls.expert_slot.size()); }
    r.rec.assign(nl, {});
    r.pos.assign(nl, 0);
    r.play.assign(nl, {});
    r.nu.assign(nl, std::vector<int32_t>(ne, INT32_MAX));
    r.active = false;
    r.picks = r.matched = r.n_pref = r.n_tok = r.n_cpu = 0;
    r.first_mis = r.first_mis_layer = r.first_cpu = r.first_cpu_layer = -1;
    r.jit0  = mc->n_jit;
    r.jitm0 = mc->n_jit_miss;
    FILE * f = fopen(replay_path(g_topic.phash).c_str(), "rb");
    if (!f) { return; }
    int32_t n = 0;
    bool ok = fread(&n, sizeof n, 1, f) == 1 && n == nl;
    std::vector<int32_t> rows(ok ? n : 0);
    ok = ok && fread(rows.data(), sizeof(int32_t), rows.size(), f) == rows.size();
    for (int il = 0; ok && il < nl; ++il) {
        if (rows[il] < 0 || rows[il] > (1 << 24)) { ok = false; break; }   // a count from a file is bounded before it sizes anything
        r.play[il].resize((size_t) rows[il] * 16);
        ok = fread(r.play[il].data(), sizeof(int16_t), r.play[il].size(), f) == r.play[il].size();
    }
    fclose(f);
    if (!ok) { for (auto & v : r.play) { v.clear(); } return; }
    r.active = true;
    int32_t mx = 0;
    for (int32_t x : rows) { mx = std::max(mx, x); }
    LLAMA_LOG_WARN("moe-cache: replay: this prompt's stored trace plays (%d answer tokens): the steps upload what its next tokens need\n", mx);
}

// the next use of every expert from each layer's position, looking horizon tokens ahead (the step: all layers are at the same token). mc->mtx held.
static void replay_next_use() {
    auto & r = g_rp;
    for (size_t il = 0; il < r.play.size(); ++il) {
        auto & nu = r.nu[il];
        std::fill(nu.begin(), nu.end(), INT32_MAX);
        const int32_t rows = (int32_t) (r.play[il].size() / 16), p = r.pos[il];
        for (int32_t t = std::min(rows, p + r.horizon) - 1; t >= p; --t) {
            const int16_t * row = r.play[il].data() + (size_t) t * 16;
            for (int i = 0; i < 16; ++i) {
                if (row[i] >= 0 && (size_t) row[i] < nu.size()) { nu[row[i]] = t - p; }
            }
        }
    }
}

// upload what the next tokens of the trace need, soonest first, into the blocks the trace needs latest (never a block whose expert is needed no
// later than the candidate, never an expert the token about to run needs: the JIT gives it its own). Returns the uploads queued. mc->mtx held.
static int replay_prefetch(moe_cache * mc, double * bud) {
    auto & r = g_rp;
    int n = 0;
    std::vector<upload_job> jobs;
    for (int d = 1; d <= r.ahead; ++d) {
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            layer_state & ls = mc->layers[li];
            const int il = ls.pub.il;
            if (il < 0 || (size_t) il >= r.play.size()) { continue; }
            const int32_t t = r.pos[il] + d;
            if ((size_t) (t + 1) * 16 > r.play[il].size()) { continue; }
            const int16_t * row = r.play[il].data() + (size_t) t * 16;
            const double S = (double) (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
            const int    lk = ls.link >= 0 && ls.link < MAX_LINKS ? ls.link : 0;
            for (int i = 0; i < 16 && bud[lk] >= S; ++i) {
                const int32_t id = row[i];
                if (id < 0 || (size_t) id >= ls.expert_slot.size() || ls.expert_slot[id] >= 0 || ls.queued[id] || r.nu[il][id] == 0) { continue; }
                double best;
                const slot_pick pk = pick_slot(mc, li, [&](layer_state & lo, size_t, int32_t x) {
                    if (pinned(mc, lo, x)) { return -1.0; }
                    const int32_t dx = lo.pub.il >= 0 && (size_t) lo.pub.il < r.nu.size() ? r.nu[lo.pub.il][x] : INT32_MAX;
                    return dx <= r.ahead ? -1.0 : (dx == INT32_MAX ? 0.0 : 1e9 / (1.0 + (double) dx));   // needed within the window: kept
                }, best);
                if (pk.slot < 0) { continue; }
                layer_state & lo = pk.owner >= 0 ? mc->layers[pk.owner] : ls;
                if (pk.owner >= 0 && lo.slot_expert[pk.slot] >= 0) { evict_slot(mc, lo, pk.slot); }
                pool_take(mc, li, pk.slot);
                ls.slot_in_flight[pk.slot] = true;
                ls.queued[id]              = 1;
                upload_job j { li, id, pk.slot };
                j.t_queued = ggml_time_us();
                jobs.push_back(j);
                bud[lk] -= S;
                n++;
            }
        }
    }
    if (!jobs.empty()) {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        for (const auto & j : jobs) { mc->todo.push_back(j); }
        mc->wcv.notify_all();
    }
    r.n_pref += n;
    r.n_tok++;
    return n;
}

// the request is over: its trace replaces the prompt's stored one, and the log says how the play went
static void replay_end(moe_cache * mc) {
    auto & r = g_rp;
    if (!r.on || r.rec.empty() || g_topic.store.path.empty() || g_topic.phash == 0) { return; }
    size_t tot = 0;
    for (const auto & v : r.rec) { tot += v.size(); }
    if (tot == 0) { return; }
    uint64_t jit, jitm;
    {
        std::lock_guard<std::mutex> lk(mc->mtx);
        jit = mc->n_jit - r.jit0; jitm = mc->n_jit_miss - r.jitm0;
    }
    const std::string path = replay_path(g_topic.phash);
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    if (FILE * f = fopen(path.c_str(), "wb")) {
        const int32_t n = (int32_t) r.rec.size();
        fwrite(&n, sizeof n, 1, f);
        for (const auto & v : r.rec) { const int32_t rows = (int32_t) (v.size() / 16); fwrite(&rows, sizeof rows, 1, f); }
        for (const auto & v : r.rec) { fwrite(v.data(), sizeof(int16_t), v.size(), f); }
        fclose(f);
    }
    LLAMA_LOG_WARN("moe-cache: replay: first pick unlike the trace at answer token %d (layer %d); %llu answer picks computed by the CPU, the first at token %d (layer %d)\n",
            r.first_mis, r.first_mis_layer, (unsigned long long) r.n_cpu, r.first_cpu, r.first_cpu_layer);
    LLAMA_LOG_WARN("moe-cache: replay: %s: %llu picks, %.2f%% of them as the stored trace had them; %llu prefetched ahead (%.1f per token), %llu of the picks "
            "missed the cache when their token ran (%.2f%%: %llu uploaded just in time), the trace stored for the next request of this prompt\n",
            r.active ? "played the stored trace" : "recorded (no stored trace)", (unsigned long long) r.picks, r.active ? 100.0 * (double) r.matched / (double) std::max<uint64_t>(1, r.picks) : 0.0,
            (unsigned long long) r.n_pref, (double) r.n_pref / (double) std::max<uint64_t>(1, r.n_tok), (unsigned long long) jitm,
            100.0 * (double) jitm / (double) std::max<uint64_t>(1, r.picks), (unsigned long long) jit);
    r.active = false;
    for (auto & v : r.rec) { v.clear(); }
}
}

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
    uint64_t h1 = 0, m1 = 0, have_map = 0, rec_hit = 0, rec_hit_t = 0;   // experts the recalled map rates, uses of the request's kept experts they cover
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
        // the record keeps every expert the routers picked (user 2026-10-29: "what routers picked in each layer at any point must be captured, stored
        // and 100% recalled per that req"); the cut to what the cache holds is the fill's business (heat order), so it is only counted here for the log
        std::vector<uint32_t> cut = cnt, cut_t = cth;
        kept   = topic_hottest(mc, cut);
        kept_t = all_t > 0 ? topic_hottest(mc, cut_t) : 0;
        for (const uint32_t c : cut)   { tot   += c; }
        for (const uint32_t c : cut_t) { tot_t += c; }
        // recall: of all this request's uses, the share at experts the recalled map rates; greedy sampling
        // (temperature 0, top-k 1) walks a repeated prompt through the same experts, so a map stored for it must recall 100% (user 2026-10-29)
        for (const auto & ls : mc->layers) {
            if (ls.pub.il < 0 || (uint32_t) ls.pub.il >= g_topic.store.n_layers) { continue; }
            for (size_t e = 0; e < ls.topic_heat.size() && e < g_topic.store.n_exp; ++e) {
                const size_t c = (size_t) ls.pub.il * g_topic.store.n_exp + e;
                have_map   += ls.topic_heat[e] > 0.0f;
                rec_hit    += ls.topic_heat[e] > 0.0f ? cnt[c] : 0;
                rec_hit_t  += ls.topic_heat[e] > 0.0f ? cth[c] : 0;
            }
        }
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
        if (have_map > 0) {
            LLAMA_LOG_WARN("moe-cache: dynhot: recall of the map used: %.1f%% of the answer's uses, %.1f%% of the thinking's (%llu experts rated)\n",
                    100.0 * (double) rec_hit / (double) std::max<uint64_t>(1, all), 100.0 * (double) rec_hit_t / (double) std::max<uint64_t>(1, all_t),
                    (unsigned long long) have_map);
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
        std::error_code ec;
        std::filesystem::rename(tmp, g_exptok.path, ec);
        if (ec) { std::remove(tmp.c_str()); }
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
                if (fread(m, 1, 8, f) == 8 && memcmp(m, "MOEXPID1", 8) == 0 && fread(h, sizeof(uint32_t), 4, f) == 4 && h[0] == L && h[1] == X && h[2] > 0 && h[2] <= 65536) {
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
// --moe stats: a status line every --moe stats-ms (default 10000; was 1000, user 2026-11-02: "writes a status line every second maybe hurting perf not needed so often 10-30s is fine"). Earlier: "we need to see
// stats samples more often than planner makes decisions"). RAM as this engine moves it (this CPU shows no memory counters without root: the CPU's expert reads, the uploads, the prompt streams; all other traffic is
// missing), each link's uploads against its probed maximum, the cache, then disk, memory and CPU (llama_moe_stats_io_fork). What every card really moved, its load, temperature and free VRAM: the gpu lines from run.py.
static unsigned long long pp_self_read_bytes();
// Before the first request (user 2026-11-03: "load all hot moes to ram to preheat pagecache prior req"): the experts the state's hot map counts most, which stayed
// mapped (the pin and the GPU cache hold the rest), are asked for from the file in the order of their counts, so request 1 finds them in the page cache instead of
// faulting them in at 0.8 GB/s (DeepSeek cold: a 2-token call took 10 s). The budget is the share of the model the pin planner already leaves to the page cache
// (--moe pin-cache, 0.15 of the mapped experts); the reads are kept to 512 MiB ahead of what the disk has delivered, wait while a prompt call runs and stop when
// a request has been served or the budget is spent. Off by default; --moe preheat=1 switches it on. No hot map (empty state) means nothing to read.
static unsigned long long swap_out_pages() {   // /proc/vmstat pswpout: pages the kernel has swapped out since boot
    unsigned long long v = 0;
    if (FILE * f = fopen("/proc/vmstat", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f)) { if (sscanf(line, "pswpout %llu", &v) == 1) { break; } }
        fclose(f);
    }
    return v;
}

// Before the first request (user 2026-11-03: "load all hot moes to ram ... prior req", then "stick to our array of pinned moe buffers"): the experts the state's hot map
// counts most, which stayed mapped (the pin and the GPU cache hold the rest), are read ONCE with O_DIRECT (no page cache, no second copy) into pinned slots, hottest
// first (ls.pin_ptr, the same array --moe pinr fills). The uploads of a miss then come from pinned RAM, not from the SSD at 0.8 to 1.6 GB/s; a mapped pointer is
// never touched (a touch pulls in one page, then 128 KiB). The budget is the share the pin planner leaves to the page cache (--moe pin-cache, 0.15 of the mapped
// experts). It waits during prompt calls, stops when decode starts, when the budget is spent, or at the first sign of swapping (a swap-out, or free memory under what it
// would be after the whole budget). Off by default; --moe preheat=1 switches it on. No hot map (empty state) means nothing to read.
static void preheat_run(moe_cache * mc) {
    // the hot counts come from the usage profile (profile_preload), which runs after the cache is set up: wait for it, give up when a request has started or the engine stops
    while (!mc->profile_ready.load(std::memory_order_acquire)) {
        if (mc->stats_stop.load() || mc->n_steps > 0) { return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    struct cand { uint64_t n; size_t li; int32_t e; };
    std::vector<cand> c;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        if (ls.src_dfd[0] < 0 || ls.src_dfd[1] < 0 || ls.src_dfd[2] < 0 || !ls.pin_ptr[0] || !ls.pin_ptr[1] || !ls.pin_ptr[2]) { continue; }
        for (int32_t e = 0; e < (int32_t) ls.expert_slot.size(); ++e) {
            if (ls.glob_count[e] > 0 && ls.expert_slot[e] < 0) { c.push_back({ ls.glob_count[e], li, e }); }
        }
    }
    if (c.empty() || mc->pin_budget == 0) {
        size_t mapped_layers = 0, with_counts = 0;
        for (const auto & ls : mc->layers) { mapped_layers += ls.src_dfd[0] >= 0 && ls.src_dfd[1] >= 0 && ls.src_dfd[2] >= 0; for (uint64_t g : ls.glob_count) { with_counts += g > 0; } }
        LLAMA_LOG_WARN("moe-cache: preheat: nothing to read (%zu candidate experts, budget %.1f GiB, %zu of %zu layers mapped for direct reads, %zu experts with hot counts)\n", c.size(), mc->pin_budget / 1073741824.0, mapped_layers, mc->layers.size(), with_counts);
        return;
    }
    std::sort(c.begin(), c.end(), [](const cand & a, const cand & b) { return a.n > b.n; });
    const int64_t t0 = ggml_time_us();
    const unsigned long long sw0 = swap_out_pages();
    size_t n = 0, n_seen = 0, n_fail = 0, n_last = 0;
    int read_errno = 0;
    bool spent = false;
    const char * why = "budget spent";
    for (const cand & x : c) {
        if (mc->stats_stop.load() || mc->n_steps > 0) { why = "a request is running"; break; }
        if ((++n_seen & 31) == 0 && swap_out_pages() > sw0) { why = "the system swapped, swap guard"; break; }   // the arena itself is checked once by pin_alloc; O_DIRECT reads add no memory
        while (g_prompt_call.load(std::memory_order_relaxed) && !mc->stats_stop.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        layer_state & ls = mc->layers[x.li];
        bool ok = true;
        const char * dst[3] = { nullptr, nullptr, nullptr };
        const char * from = nullptr;
        for (int k = 0; k < 3 && ok; ++k) {
            const ggml_tensor * t = k == 0 ? ls.pub.up_src : k == 1 ? ls.pub.gate_src : ls.pub.down_src;
            const size_t sz = t->nb[2];
            const size_t pb = expert_past(mc, ls, k, x.e, t, from);   // bytes a matmul reads past the slice: the next expert's, read along (as the upload worker does)
            if (pb > 0 && x.e + 1 >= t->ne[2]) { ok = false; n_last++; break; }   // the last expert's past bytes are padding, not in the file: it stays with the mapping
            const size_t off = ls.src_offs[k] + (size_t) x.e*sz, beg = off & ~(size_t) 4095, end = (off + sz + pb + 4095) & ~(size_t) 4095;
            char * d = pin_alloc(mc, end - beg, 4096);   // the O_DIRECT read lands in the pinned block itself: no staging buffer, no copy, no page cache
            if (!d) { ok = false; spent = true; break; }
            size_t got = 0;
            if (sr_read(ls.src_dfd[k], d, beg, end - beg, off + sz + pb - beg)) { got = off + sz + pb - beg; }
            else if (!read_errno) { read_errno = errno; }
            if (got < off + sz + pb - beg) { ok = false; n_fail++; break; }   // a failed read: the expert stays mapped, its block stays unused
            dst[k] = d + (off - beg);
        }
        if (spent) { break; }
        if (!ok) { continue; }   // a partly copied expert is simply not published (its pinned bytes stay unused)
        for (int k = 0; k < 3; ++k) { ls.pin_ptr[k][x.e].store(dst[k], std::memory_order_release); }
        n++;
    }
    LLAMA_LOG_WARN("moe-cache: preheat: %zu hot experts, %.1f of %.1f GiB pinned in %.1f s, hottest first; stopped: %s; %zu failed reads%s%s, %zu last experts left mapped\n", n, mc->pin_used / 1073741824.0, mc->pin_budget / 1073741824.0, (ggml_time_us() - t0) / 1e6, why,
            n_fail, read_errno ? ", first: " : "", read_errno ? strerror(read_errno) : "", n_last);
}

// Fast engine windows (user 2026-11-05: "for every 10 tokens we for sure know how much data we sent where", "and what time per token was, and saturation of ram, vram and pcie bandwidth"): every
// --moe stats-fast-ms (default 100 ms, 0 = off) the engine's own byte counters are read and the change since the last window is kept in memory, written as UPL lines of stats.log every 10 s with the
// window's own time. A line has the decode steps of the window and the time per token they took, RAM traffic (CPU compute, uploads, prompt streaming) against the RAM ceiling, every link's
// traffic against its probed maximum, and how full the cache in VRAM is. Windows where nothing moved are not written. A 3 s prompt call so shows its own rates, not a 10 s average.
static void fast_window_run(moe_cache * mc) {
    const char * o = moe_opt("stats-fast-ms");
    const int period_ms = o ? atoi(o) : 100;
    if (period_ms <= 0) { return; }
    struct win { double t, dt; uint64_t steps; int64_t cpu, up, pf, lk[MAX_LINKS]; uint64_t filled, total; };
    std::vector<win> buf;
    int64_t t0 = ggml_time_us(), cpu0 = mc->cpu_bytes.load(), up0 = mc->up_bytes.load(), pf0 = mc->pf_bytes.load(), lk0[MAX_LINKS] = {};
    for (int k = 0; k < mc->n_links; ++k) { lk0[k] = mc->link_bytes[k].load() + mc->pf_link[k].load(); }
    uint64_t n0 = mc->n_steps;
    double last_flush = llama_moe_span_now_fork();
    while (!mc->stats_stop.load(std::memory_order_relaxed)) {   // ends with the cache: llama_moe_cache_free_fork joins this thread before it deletes mc
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
        const int64_t t = ggml_time_us();
        win w = {};
        w.t = llama_moe_span_now_fork(); w.dt = (double) (t - t0) / 1e6;
        const int64_t cpu = mc->cpu_bytes.load(), up = mc->up_bytes.load(), pf = mc->pf_bytes.load();
        const uint64_t n = mc->n_steps;
        w.steps = n - n0; w.cpu = cpu - cpu0; w.up = up - up0; w.pf = pf - pf0;
        bool any = w.steps || w.cpu || w.up || w.pf;
        for (int k = 0; k < mc->n_links; ++k) {
            const int64_t b = mc->link_bytes[k].load() + mc->pf_link[k].load();
            w.lk[k] = b - lk0[k]; lk0[k] = b; any = any || w.lk[k];
        }
        t0 = t; cpu0 = cpu; up0 = up; pf0 = pf; n0 = n;
        if (any) { llama_moe_cache_fill_fork(&w.filled, &w.total); buf.push_back(w); }
        if (w.t - last_flush >= 10.0) {
            last_flush = w.t;
            for (const auto & x : buf) {
                std::string links;
                for (int k = 0; k < mc->n_links; ++k) {
                    const double g = (double) x.lk[k] / x.dt / 1e9;
                    links += format(" %s %.2f GB/s (%.0f%% of %.1f)", mc->link_dev[k] ? ggml_backend_dev_name(mc->link_dev[k]) : "?", g, mc->gbs_link[k] > 0 ? 100.0 * g / mc->gbs_link[k] : 0.0, mc->gbs_link[k]);
                }
                const double ram = (double) (x.cpu + x.up + x.pf) / x.dt / 1e9, ceil = knobs().ddr_gbs;
                llama_moe_stats_emit_at_fork("UPL", format("window %.0f ms: steps %llu%s; ram %.2f GB/s (%.0f%% of %.1f; cpu %.2f, upload %.2f, stream %.2f); links%s; cache %llu/%llu blocks (%.0f%%)",
                        x.dt * 1e3, (unsigned long long) x.steps, x.steps ? format(" (%.1f ms per token)", x.dt * 1e3 / (double) x.steps).c_str() : "", ram, ceil > 0 ? 100.0 * ram / ceil : 0.0, ceil,
                        (double) x.cpu / x.dt / 1e9, (double) x.up / x.dt / 1e9, (double) x.pf / x.dt / 1e9, links.c_str(), (unsigned long long) x.filled, (unsigned long long) x.total,
                        x.total ? 100.0 * (double) x.filled / (double) x.total : 0.0).c_str(), x.t - x.dt);
            }
            buf.clear();
        }
    }
}
static void stats_status_run(moe_cache * mc) {
    const char * o = moe_opt("stats-ms");
    const int64_t period = (int64_t) std::max(50, o ? atoi(o) : 10000) * 1000;
    int64_t t0 = ggml_time_us(), cpu0 = mc->cpu_bytes.load(), up0 = mc->up_bytes.load(), pf0 = mc->pf_bytes.load(), lk0[MAX_LINKS] = {};
    for (int k = 0; k < mc->n_links; ++k) { lk0[k] = mc->link_bytes[k].load() + mc->pf_link[k].load(); }
    uint64_t h0 = 0, m0 = 0, n0 = mc->n_steps, u0 = mc->n_uploads, e0 = mc->n_evictions;
    while (!mc->stats_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const int64_t t = ggml_time_us();
        if (t - t0 < period) { continue; }
        const double dt = (double) (t - t0) / 1e6;
        const int64_t cpu = mc->cpu_bytes.load(), up = mc->up_bytes.load(), pf = mc->pf_bytes.load();
        uint64_t h = 0, m = 0;
        for (const auto & l : mc->layers) { h += l.n_hit; m += l.n_miss; }
        std::string links;
        for (int k = 0; k < mc->n_links; ++k) {
            const int64_t b = mc->link_bytes[k].load() + mc->pf_link[k].load();
            links += format(" %s %.2f/%.1f", mc->link_dev[k] ? ggml_backend_dev_name(mc->link_dev[k]) : "?", (double) (b - lk0[k]) / dt / 1e9, mc->gbs_link[k]);
            lk0[k] = b;
        }
        size_t todo = 0;
        int in_flight = 0;
        {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            todo = mc->todo.size();
            in_flight = mc->in_flight;
        }
        const uint64_t n = mc->n_steps, u = mc->n_uploads, e = mc->n_evictions;
        const std::string line = format("ram %.2f/%.1f GB/s (cpu %.2f, upload %.2f, stream %.2f); links%s GB/s; hit %.1f%%, steps %.1f/s, uploads %.0f/s, evictions %.0f/s, queue %zu+%d; %s",
                (double) (cpu - cpu0 + up - up0 + pf - pf0) / dt / 1e9, knobs().ddr_gbs, (double) (cpu - cpu0) / dt / 1e9, (double) (up - up0) / dt / 1e9, (double) (pf - pf0) / dt / 1e9, links.c_str(),
                h + m > h0 + m0 ? 100.0 * (double) (h - h0) / (double) (h + m - h0 - m0) : 0.0, (double) (n - n0) / dt, (double) (u - u0) / dt, (double) (e - e0) / dt, todo, in_flight, llama_moe_stats_io_fork(1).c_str());
        llama_moe_stats_log_fork("STA", line.c_str());
        t0 = t; cpu0 = cpu; up0 = up; pf0 = pf; h0 = h; m0 = m; n0 = n; u0 = u; e0 = e;
    }
}

void llama_moe_request_begin_fork(const int32_t * prompt, int32_t n_prompt, bool own_text) {
    llama_moe_stats_log_fork("REQ", format("begin: %d prompt tokens", n_prompt).c_str());
    llama_moe_stats_log_fork("PHA", format("idle -> prefill: %d prompt tokens", n_prompt).c_str());
    g_req_prompt.assign(prompt, prompt + (prompt != nullptr ? std::max(0, n_prompt) : 0));
    g_req_own = own_text;
    g_req_boundary.store(true);
}

void llama_tokheat_dump_fork(void);   // llama-context.cpp: --moe tokheat=DIR

static void rp_log(const char * when);
// --moe rampool (fork/docs/rampool.md): expert slices that stayed mapped (the loader pinned no expert tensor, see llama-model-loader.cpp) go to a pinned pool chosen by heat
static constexpr size_t RP_PIECE = (size_t) 1 << 30;   // the pool's chunk ceiling (ggml_rampool_open_fork) and the prepin's piece: 1 GiB takes the CUDA host allocator's fast pin
static void * rp_alloc(size_t bytes, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    if (bytes <= RP_PIECE && bytes > RP_PIECE / 2 && !mc->rp_pre.empty()) {   // a full chunk: a piece pinned before the VRAM sizing (rp_prepin); a class's small last chunk is pinned on its own
        ggml_backend_buffer_t b = mc->rp_pre.back();
        mc->rp_pre.pop_back();
        mc->rp_bufs.push_back(b);
        return ggml_backend_buffer_get_base(b);
    }
    const size_t floor = (size_t) 6 << 30;   // the floor pin_alloc keeps: never push the system into swapping
    if (mem_available_bytes() < mc->rp_raw_bytes + bytes + floor) { return nullptr; }   // the raw chunks below hold no page yet: their bytes still count as available
#ifndef _WIN32
    if (mc->rp_pin) {   // read first, pinned after (rp_pin_raw): the O_DIRECT read faults the pages in at the drive's speed, so no populate pass (90 GiB: 16.8 s against 27.3 s pinned first)
        void * m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { return nullptr; }
#ifdef MLOCK_ONFAULT
        // locked as the fill's reads fault the pages in: none swapped out before the pin, and off the LRU list after it (pin-mlock)
        if (const char * ml = moe_opt("pin-mlock"); !(ml && ml[0] == '0')) { (void) mlock2(m, bytes, MLOCK_ONFAULT); }
#endif
        mc->rp_raw.push_back({ (char *) m, bytes });
        mc->rp_raw_bytes += bytes;
        return m;
    }
#endif
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(mc->pin_buft, bytes);
    if (!b) { return nullptr; }
    char * base = (char *) ggml_backend_buffer_get_base(b);
#ifndef _WIN32
    if (const char * ml = moe_opt("pin-mlock"); !(ml && ml[0] == '0')) { (void) mlock((void *) ((uintptr_t) base & ~(uintptr_t) 4095), bytes); }   // as the pinned weights (pin-mlock)
#endif
    mc->rp_bufs.push_back(b);
    return base;
}
#ifndef _WIN32
// --moe map-guard: the guarded ranges, and the SIGSEGV report that names the tensor and the stack of the reader left on the mapping (a test mode: snprintf in a handler is fine here)
struct mg_range { uintptr_t a, b; const char * name; };
static std::vector<mg_range> & mg_ranges() { static std::vector<mg_range> v; return v; }
static void mg_on_segv(int sig, siginfo_t * si, void * uc) {
    const uintptr_t x = (uintptr_t) si->si_addr;
    static std::atomic<bool> first{false};
    if (first.exchange(true)) { for (;;) { pause(); } }   // several CPU threads fault at once: the first prints, the others wait for its re-raise
    char buf[640];
    for (const auto & r : mg_ranges()) {
        if (x >= r.a && x < r.b) {
            uintptr_t ip = 0;   // the instruction that read: backtrace() came back empty in these handlers (2026-11-08), this alone names the kernel (addr2line -e <module> <offset>)
#if defined(__x86_64__)
            ip = (uintptr_t) ((ucontext_t *) uc)->uc_mcontext.gregs[REG_RIP];
#endif
            Dl_info di = {};
            const bool mod = ip != 0 && dladdr((void *) ip, &di) != 0 && di.dli_fname != nullptr;
            const int n = snprintf(buf, sizeof buf, "map-guard: read of %s at offset %zu through the mapping, by %s+0x%zx, stack:\n", r.name, (size_t) (x - r.a),
                    mod ? di.dli_fname : "?", mod ? (size_t) (ip - (uintptr_t) di.dli_fbase) : (size_t) ip);
            if (n > 0) { (void) !write(2, buf, (size_t) n); }
            void * st[48];
            backtrace_symbols_fd(st, backtrace(st, 48), 2);
            break;
        }
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
#endif
// The CPU kernel's copy of expert e. Most paths ask the pool before the kernel (llama_moe_cache_jit_fork); a layer the prompt chain could not take runs the plain path, whose small passes
// stay on the CPU with no such step (GLM 3.5 map-guard 2026-11-08: the 2-token call read blk.3's gate tensor through the mapping in vec_dot). So a miss is asked here, the wait a page fault
// would have been, in pinned RAM instead of the page cache. The demand takes a block even from the fill's set (rampool=1 keeps it otherwise; DeepSeek 12k map-guard
// 2026-11-08: 22 of 123 GiB out of the pool, the tail call's CPU experts read the down tensor through the mapping).
static const void * rp_cpu_ptr(const ggml_tensor * src0, int64_t e) {
    if (const char * p = ggml_rampool_get_fork(src0, e)) { return p; }
    if (const moe_cache * mc = g_cache) {
        const int32_t id = (int32_t) e;
        for (const auto & l : mc->layers) {
            if (l.pub.up_src == src0 || l.pub.gate_src == src0 || l.pub.down_src == src0) {
                const void * keys[3] = { l.pub.up_src, l.pub.gate_src, l.pub.down_src };
                ggml_rampool_demand_fork(keys, 3, &id, 1);
                return ggml_rampool_get_fork(src0, e);
            }
        }
    }
    return nullptr;
}
static void rp_pre_free(moe_cache * mc) {   // pieces the pool did not take go back
    for (auto * b : mc->rp_pre) { ggml_backend_buffer_free(b); }
    mc->rp_pre.clear();
}
static void rp_raw_free(moe_cache * mc) {   // after ggml_rampool_close_fork: no reader left
#ifndef _WIN32
    for (auto & r : mc->rp_raw) {
        if (r.second == 0) { continue; }
        if (mc->rp_unpin) { mc->rp_unpin(r.first); }
        munmap(r.first, r.second);
    }
#endif
    mc->rp_raw.clear();
    mc->rp_raw_bytes = 0;
}
extern "C" GGML_API void ggml_rampool_on_chunk_fork(void (*cb)(void * chunk, void * ud));   // ggml-rampool-fork.cpp, declared here: no header edit
// The chunks the fill read into, pinned in place: the read faulted their pages in, so the register only pins them. 4 threads pin each chunk as its last fill read lands
// (ggml_rampool_on_chunk_fork), under the rest of the fill; what is left (a chunk no fill read touched) after it. One that is refused stays pageable (its uploads go
// through the driver's staging, slower but right). DS 100.3 GiB: 7.69 s on 4 threads after the fill before.
struct rp_pinner {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<size_t> q;                      // rp_raw entries whose fill reads are done
    bool end = false;
    std::unordered_map<char *, size_t> at;     // chunk -> its rp_raw entry
    std::vector<uint8_t> done;
    std::atomic<size_t> refused{0}, under{0};  // under: bytes pinned as their reads ended
    std::vector<std::thread> th;
};
static rp_pinner * g_rp_pinner = nullptr;
static void rp_chunk_read(void * chunk, void * /*ud*/) {
    rp_pinner * pp = g_rp_pinner;
    if (!pp) { return; }
    const auto it = pp->at.find((char *) chunk);
    if (it == pp->at.end()) { return; }   // a piece pinned before the sizing (rp_prepin)
    { std::lock_guard<std::mutex> lk(pp->mu); pp->q.push_back(it->second); }
    pp->cv.notify_one();
}
static void rp_pin_start(moe_cache * mc, rp_pinner & pp) {
    if (mc->rp_raw.empty()) { return; }
    for (size_t i = 0; i < mc->rp_raw.size(); ++i) { pp.at[mc->rp_raw[i].first] = i; }
    pp.done.assign(mc->rp_raw.size(), 0);
    g_rp_pinner = &pp;
    ggml_rampool_on_chunk_fork(rp_chunk_read);
    for (int t = 0; t < 4; ++t) {
        pp.th.emplace_back([mc, &pp] {
            for (;;) {
                size_t i;
                {
                    std::unique_lock<std::mutex> lk(pp.mu);
                    pp.cv.wait(lk, [&] { return pp.end || !pp.q.empty(); });
                    if (pp.q.empty()) { return; }
                    i = pp.q.front();
                    pp.q.pop_front();
                }
                if (!mc->rp_pin(mc->rp_raw[i].first, mc->rp_raw[i].second)) { pp.refused++; }
                pp.done[i] = 1;
                pp.under += mc->rp_raw[i].second;
            }
        });
    }
}
static void rp_pin_raw(moe_cache * mc, rp_pinner & pp) {
    if (mc->rp_raw.empty()) { return; }
    const int64_t t0 = ggml_time_us();
    { std::lock_guard<std::mutex> lk(pp.mu); pp.end = true; }   // the fill is done: the queue drains, then the threads end
    pp.cv.notify_all();
    for (auto & t : pp.th) { t.join(); }
    ggml_rampool_on_chunk_fork(nullptr);
    g_rp_pinner = nullptr;
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next++) < mc->rp_raw.size(); ) {
            if (!pp.done[i] && !mc->rp_pin(mc->rp_raw[i].first, mc->rp_raw[i].second)) { pp.refused++; }
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < 4; ++t) { th.emplace_back(work); }
    work();
    for (auto & t : th) { t.join(); }
    LLAMA_LOG_WARN("moe-cache: rampool: %.1f GiB pinned in place on 4 threads (%zu chunks): %.1f GiB as their fill reads ended, the rest in %.2f s after the fill%s\n",
            mc->rp_raw_bytes / 1073741824.0, mc->rp_raw.size(), pp.under.load() / 1073741824.0, (ggml_time_us() - t0) / 1e6,
            pp.refused.load() ? tr_fmt(", %zu refused: pageable", pp.refused.load()).c_str() : "");
}
// Pinned host memory costs every card VRAM for its mapping (GLM 3.5, 101.9 GiB pinned: ~150 MiB per card). Pinned after the expert pool was sized, it ate the
// margin and cuBLAS found no memory at the first prompt (2026-11-07). So the pool's bytes are pinned here, before the sizing, whose free VRAM then includes that cost.
static void rp_prepin(moe_cache * mc, ggml_backend_dev_t gpu) {
    const char * rv = moe_opt("rampool");
    if (!rv || atoi(rv) == 0 || !gpu) { return; }
    const char * pb = moe_opt("pin-budget"), * pu = moe_opt("pin-used");
    const size_t budget = pb ? (size_t) strtoull(pb, nullptr, 10) : 0, used = pu ? (size_t) strtoull(pu, nullptr, 10) : 0;
    const size_t want = budget > used ? budget - used : 0;
    if (want == 0) { return; }
    mc->pin_buft = ggml_backend_dev_host_buffer_type(gpu);
    const int64_t t0 = ggml_time_us();
#ifndef _WIN32
    ggml_backend_reg_t greg = ggml_backend_dev_backend_reg(gpu);
    mc->rp_pin   = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(greg, "ggml_backend_cuda_pin_range_fork");
    mc->rp_unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(greg, "ggml_backend_cuda_unpin_range_fork");
    if (!mc->rp_unpin) { mc->rp_pin = nullptr; }
    if (mc->rp_pin && want >= RP_PIECE) {
        // The pool's chunks are pinned after the fill (rp_alloc, rp_pin_raw), so after the sizing: one piece is pinned here, and what it took of each card's VRAM
        // (PC1: 2 MiB per GiB, a GPU page table entry per 4 KiB page) is left free at the sizing for every other GiB of the budget. shortcut: reserved for the
        // whole budget, a pool that turns out smaller leaves the difference unused (2 MiB per GiB per card), a reserve from the fill's own plan if that shows.
        std::vector<std::pair<ggml_backend_dev_t, size_t>> f0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_backend_reg(d) != greg || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) { continue; }
            size_t f = 0, t = 0;
            ggml_backend_dev_memory(d, &f, &t);
            f0.push_back({ d, f });
        }
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(mc->pin_buft, RP_PIECE);
        if (b) {
            size_t per = 0;
            for (const auto & x : f0) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(x.first, &f, &t);
                per = std::max(per, x.second > f ? x.second - f : 0);
            }
            // the driver takes page-table memory in 2 MiB steps, so one piece reads 0 or 2 MiB: a 0 left no reserve, the pin under the fill then took ~214 MiB of each
            // card and the first pass ran out of memory (DS 2026-11-10, 3 of 42 starts read 0: restart at micro-batch 2048, +40 s). Never below one 8 B entry per 4 KiB page.
            per = std::max(per, RP_PIECE / 4096 * 8);
            mc->rp_pre.push_back(b);
            const size_t n_rest = want / RP_PIECE - 1 + 8;   // the classes' short last chunks are pinned on their own: a few more ranges than full pieces
            mc->rp_vram_reserve = per * n_rest;
            LLAMA_LOG_WARN("moe-cache: rampool: 1 GiB pinned before the VRAM sizing in %.2f s, it took %.1f MiB of each card's VRAM: %.0f MiB left free per card at the sizing "
                    "for the pin after the fill (budget %.1f GiB)\n", (ggml_time_us() - t0) / 1e6, per / 1048576.0, mc->rp_vram_reserve / 1048576.0, want / 1073741824.0);
            return;
        }
        mc->rp_pin = nullptr;   // no piece: the old way below (it stops at the first piece that fails too)
    }
#endif
    // one piece per full chunk, never more than the budget (left over ones are freed after the open). The pieces are populated and pinned on several threads:
    // one thread took 35 s for GLM 3.0's 101 GiB (0.2 s populate + 0.07 s register per GiB, before the first token); --moe pin-threads=N (default 4, 1 = the old serial loop).
    // shortcut: a piece that fails ends the run of pieces after it too (the prefix that got a buffer stays), a refused pin falls back to pageable inside the allocator as before.
    const char * ptv = moe_opt("pin-threads");
    const int nthr = std::max(1, std::min(16, ptv ? atoi(ptv) : 4));
    const size_t n_want = want / RP_PIECE;
    std::vector<ggml_backend_buffer_t> got_b(n_want, nullptr);
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    auto worker = [&] {
        for (size_t i; !stop && (i = next++) < n_want; ) {
            if (mem_available_bytes() < RP_PIECE + ((size_t) 6 << 30)) { stop = true; break; }
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(mc->pin_buft, RP_PIECE);
            if (!b) { stop = true; break; }
#ifndef _WIN32
            if (const char * ml = moe_opt("pin-mlock"); !(ml && ml[0] == '0')) { (void) mlock(ggml_backend_buffer_get_base(b), RP_PIECE); }
#endif
            got_b[i] = b;
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < nthr; ++t) { th.emplace_back(worker); }
    worker();
    for (auto & t : th) { t.join(); }
    size_t got = 0;
    for (size_t i = 0; i < n_want; ++i) {
        if (!got_b[i]) { for (size_t j = i + 1; j < n_want; ++j) { if (got_b[j]) { ggml_backend_buffer_free(got_b[j]); } } break; }
        mc->rp_pre.push_back(got_b[i]);
        got += RP_PIECE;
    }
    LLAMA_LOG_WARN("moe-cache: rampool: %.1f GiB pinned before the VRAM sizing in %.1f s on %d threads (budget %.1f GiB)\n", got / 1073741824.0, (ggml_time_us() - t0) / 1e6, nthr, want / 1073741824.0);
}
static void rp_setup(moe_cache * mc, const llama_model & model) {
    const char * rv = moe_opt("rampool");
    if (!rv || atoi(rv) == 0) { return; }
    if (const std::string sf = moe_state_file(); !sf.empty()) {   // <state dir>/<model>/globalhot.ini (user 2026-11-07; the model's directory beside the state file, as its config.ini will be)
        std::string m = moe_state_section(model);
        for (char & ch : m) { if (!isalnum((unsigned char) ch) && ch != '-' && ch != '.') { ch = '_'; } }
        const std::string dir = std::filesystem::path(sf).parent_path().string() + "/" + m;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        mc->heat_file = dir + "/globalhot.ini";
    }
    for (auto & ls : mc->layers) { ls.heat_cnt = ls.glob_count; }   // no heat file yet: the state's lifetime counts, once
    const bool heat_from_file = !mc->heat_file.empty() && rp_heat_load(mc);
    const char * pb = moe_opt("pin-budget"), * pu = moe_opt("pin-used");
    const size_t budget = pb ? (size_t) strtoull(pb, nullptr, 10) : 0, used = pu ? (size_t) strtoull(pu, nullptr, 10) : 0;
    ggml_backend_dev_t gpu = mc->bufs.empty() ? nullptr : ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
    size_t n_t = 0, bytes = 0, n_past = 0;
    for (auto & ls : mc->layers) {
        const ggml_tensor * ts[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3; ++k) {
            const char * from = nullptr;
            if (ls.src_dfd[k] < 0 || ls.src_fd[k] < 0) { continue; }
            // bytes its matmul reads past the slice (DS: 40 of 120 tensors): the block carries them, read along (the next expert's; zeros behind the last). Until
            // 2026-11-08 these tensors stayed out of the pool and every read of them went to the mapping: DS 12k rampool=1 decode 16.2 against 20.9 t/s, 0.5 GB/s SSD
            const size_t pb = expert_past(mc, ls, k, 0, ts[k], from);
            n_past += pb > 0;
            ggml_rampool_add_fork(ts[k], ls.src_fd[k], ls.src_offs[k], ts[k]->ne[2], ts[k]->nb[2], pb, (int32_t) ts[k]->type);
            n_t++; bytes += ggml_nbytes(ts[k]);
        }
    }
    const size_t avail = mem_available_bytes(), want = budget > used ? budget - used : 0;
    if (!gpu || n_t == 0 || want == 0 || mc->rd_threads <= 0) {
        LLAMA_LOG_WARN("moe-cache: rampool: off (%s; pin budget %.1f GiB, pinned by the loader %.1f GiB, %zu expert tensors mapped for direct reads, %zu read past their slice)\n",
                !gpu ? "no GPU" : n_t == 0 ? "no expert tensor stayed mapped: the model fits the pin budget" : want == 0 ? "no budget left" : "no direct reads", budget / 1073741824.0, used / 1073741824.0, n_t, n_past);
        ggml_rampool_close_fork();
        rp_pre_free(mc);
        return;
    }
    mc->pin_buft = ggml_backend_dev_host_buffer_type(gpu);
    const int64_t t0 = ggml_time_us();
    if (!ggml_rampool_open_fork(want, rp_alloc, rp_heat, mc, mc->rd_threads)) {
        LLAMA_LOG_WARN("moe-cache: rampool: off (no pinned memory for it, %.1f GiB available)\n", avail / 1073741824.0);
        rp_pre_free(mc);
        rp_raw_free(mc);
        return;
    }
    rp_pre_free(mc);
    // shortcut: rampool=1 keeps the fill's set for the run (heat moves it between runs through the heat file); rampool=2 evicts by heat, slower on GLM 3.5 (pool churn, fork/docs/rampool.md)
    mc->rp_evict = atoi(rv) >= 2;
    ggml_rampool_evict_fork(mc->rp_evict);
    mc->rp_filling = true;
    rp_pinner pp;
    rp_pin_start(mc, pp);
    ggml_rampool_fill_fork();
    mc->rp_filling = false;
    rp_pin_raw(mc, pp);   // before the tables below: nothing copies from the pool until then
    struct ggml_rampool_stats st;
    ggml_rampool_stats_fork(&st, true);
    for (auto & ls : mc->layers) {
        const ggml_tensor * ts[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        for (int k = 0; k < 3; ++k) { ls.rp_tab[k] = ggml_rampool_table_fork(ts[k]); }
    }
    if (auto set = (void (*)(const void * (*)(const ggml_tensor *, int64_t))) cpu_proc_fork("ggml_backend_cpu_set_expert_ptr_hook")) { set(rp_cpu_ptr); }
    mc->rp_on = true;
    // --moe map-guard=1 (user 2026-11-08: "we have no control over individual moes in [the mapping] and can't evict ... shall read/prefetch intelligently instead of faults"): the
    // expert tensors' mapping pages become unreadable, so a reader left on the mapping stops the server at once instead of faulting the SSD in silence. A test mode for the
    // sane and gate lists: with rampool=0 the mapping is the source on purpose. Whole pages inside each tensor only: an edge page may hold a neighbour the loader still reads.
    if (const char * mg = moe_opt("map-guard"); mg && atoi(mg) > 0) {
#ifndef _WIN32
        size_t guarded = 0;
        for (auto & ls : mc->layers) {
            const ggml_tensor * ts[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                if (!ls.rp_tab[k] || !ts[k]->data) { continue; }
                const uintptr_t a = ((uintptr_t) ts[k]->data + 4095) & ~(uintptr_t) 4095, b = ((uintptr_t) ts[k]->data + ggml_nbytes(ts[k])) & ~(uintptr_t) 4095;
                if (b > a && mprotect((void *) a, b - a, PROT_NONE) == 0) { guarded += b - a; mg_ranges().push_back({ a, b, ts[k]->name }); }
            }
        }
        { void * st[1]; (void) backtrace(st, 1); }   // loads libgcc now: its first call allocates, which a signal handler must not
        struct sigaction sa = {};
        sa.sa_sigaction = mg_on_segv; sa.sa_flags = SA_SIGINFO; sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, nullptr);   // the reader's name and stack in the server log, then the default crash
        LLAMA_LOG_WARN("moe-cache: rampool: map-guard: %.1f GiB of the expert mapping made unreadable (PROT_NONE): a read of it is a bug and stops the server\n", guarded / 1073741824.0);
#else
        LLAMA_LOG_WARN("moe-cache: rampool: map-guard: not on Windows\n");
#endif
    }
#ifndef _WIN32
    // The loader read the dense weights through the mapping, and their pages stay in the page cache for nothing (GLM 3.5 12k: 10.7 GiB beside a 101 GiB pool, 9 GiB of RAM
    // left, 2.4 GiB swapped out). Every later read of the file is O_DIRECT (the fill, the demand reads, the JIT ring), so the file's cached pages go now; the dense weights
    // through pinned staging instead of the mapping is the next step (user 2026-11-08: no mmap).
    {
        std::vector<int> fds;
        for (const auto & ls : mc->layers) { for (int k = 0; k < 3; ++k) { if (ls.src_fd[k] >= 0 && std::find(fds.begin(), fds.end(), ls.src_fd[k]) == fds.end()) { fds.push_back(ls.src_fd[k]); } } }
        for (int fd : fds) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); }
    }
#endif
    size_t ringed = 0;
    for (const auto & ls : mc->layers) { for (size_t i = RP_VIS - 1; i < ls.vis.size(); i += RP_VIS) { ringed += ls.vis[i] != 0; } }
    LLAMA_LOG_WARN("moe-cache: rampool: %.1f of %.1f GiB expert slices in RAM (%.1f GiB filled, %d size classes, %zu tensors) = pin budget %.1f - pinned by the loader %.1f GiB, RAM available %.1f GiB, in %.1f s; "
            "picked by heat (%s): %zu experts with a visit ring (clock %u), the rest by lifetime count; %zu tensors read past their slice (their blocks carry those bytes)\n",
            st.bytes / 1073741824.0, bytes / 1073741824.0, st.used / 1073741824.0, st.classes, n_t, budget / 1073741824.0, used / 1073741824.0, avail / 1073741824.0, (ggml_time_us() - t0) / 1e6,
            heat_from_file ? mc->heat_file.c_str() : "no heat file yet, the state's lifetime counts", ringed, mc->vis_now.load(), n_past);
    llama_moe_decision_fork("stage", "rampool fill", "");
}
// stats.log RPL and the log: what the pool served since the last line (per request, at exit). RAM share = asked found in the pool; SSD = read into the pool now + refused (old path)
static void rp_log(const char * when) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->rp_on) { return; }
    struct ggml_rampool_stats st;
    ggml_rampool_stats_fork(&st, true);
    const double a = (double) std::max<uint64_t>(1, st.asked), pp = (double) std::max<uint64_t>(1, st.pp_hit + st.pp_miss);
    const std::string line = format("%s: misses %llu, from RAM %.1f%%, from SSD %.1f%% (read into the pool %llu, refused %llu, %.2f GiB read, waited %.1f ms), evicted %llu; prompt copies from RAM %.1f%% (%llu of %llu); pool %.1f of %.1f GiB used",
            when, (unsigned long long) st.asked, 100.0 * st.hit / a, 100.0 * (st.cold + st.refused) / a, (unsigned long long) st.cold, (unsigned long long) st.refused, st.cold_bytes / 1073741824.0,
            st.wait_us / 1e3, (unsigned long long) st.evicted, 100.0 * st.pp_hit / pp, (unsigned long long) st.pp_hit, (unsigned long long) (st.pp_hit + st.pp_miss), st.used / 1073741824.0, st.bytes / 1073741824.0);
    llama_moe_stats_log_fork("RPL", line.c_str());
    LLAMA_LOG_WARN("moe-cache: rampool: %s\n", line.c_str());
}

void llama_moe_request_end_fork(void) {
    llama_moe_stats_log_fork("REQ", "end");
    rp_log("request");
    g_warm.store(true);
    if (g_pjit_layers.load() > 0) {   // prompt calls of this request: what the JIT saw and did (uploads against CPU-computed misses)
        const uint64_t ms = g_pjit_miss.exchange(0), up = g_pjit_up.exchange(0), ly = g_pjit_layers.exchange(0);
        std::string pl;
        for (int l = 0; l < 256; ++l) { const uint64_t v = g_pjit_left_l[l].exchange(0); if (v) { pl += tr_fmt(" L%d%s:%" PRIu64, l, g_pjit_pool_l[l].load() ? "p" : "", v); } }
        LLAMA_LOG_WARN("moe-cache: prompt calls: left to the CPU per layer (p: pool branch):%s\n", pl.c_str());
        LLAMA_LOG_WARN("moe-cache: prompt calls: %" PRIu64 " layer-steps, %" PRIu64 " missed experts, %" PRIu64 " uploaded, %" PRIu64 " left to the CPU (%.1f%%)\n", ly, ms, up, ms - std::min(ms, up), ms ? 100.0*(ms - std::min(ms, up))/ms : 0.0);
    }
    if (moe_cache * jc = g_cache; jc && (jc->jit_rd_n || jc->jit_map_n)) {   // this request's JIT uploads of mapped experts: how they were read
        LLAMA_LOG_WARN("moe-cache: jit: %" PRIu64 " mapped expert slices read O_DIRECT for uploads, %.2f GiB at %.1f GB/s, %" PRIu64 " DMA'd from the mapping (no O_DIRECT source)\n",
                jc->jit_rd_n, jc->jit_rd_b / 1073741824.0, jc->jit_rd_us ? jc->jit_rd_b / (double) jc->jit_rd_us / 1e3 : 0.0, jc->jit_map_n);
        jc->jit_rd_n = jc->jit_rd_b = jc->jit_rd_us = jc->jit_map_n = 0;
    }
    exptok_save();
    llama_tokheat_dump_fork();
    uniq_request_end();
    if (g_cache != nullptr) { replay_end(g_cache); }
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
    // --moe req-reset=1: a new request says what the last one's window was worth: nothing when the topic changed. Replay of mixed topics at a 10% cache: the first 8 tokens of a
    // request hit 12% with the old window, 43% with it emptied, the whole answer gains 1 to 2 points. ClassLRU's classes drop by two for the same reason; an expert's age stays.
    static const bool req_reset = [] { const char * e = moe_opt("req-reset"); return e && atoi(e) != 0; }();
    if (first && req_reset && mc != nullptr) {
        std::lock_guard<std::mutex> lock(mc->mtx);
        for (auto & ls : mc->layers) {
            ls.recent.clear();
            std::fill(ls.win_count.begin(), ls.win_count.end(), 0u);
            for (float & c : ls.crf) { c *= 0.25f; }
        }
    }
    exptok_call(model, ids, n, first);   // --moe expids: every call
    g_tokdump.req += first;              // --moe tokdump: the answer tokens that follow belong to a new request
    topic_phase(model, ids, n, first);   // --moe dynhot: <think> / </think> among the tokens fed (prompt and answer)
    if (mc == nullptr || !first || g_topic.on == 0) {
        return;
    }
    if (g_topic.on < 0) {   // the first prompt: open the store
        size_t max_rec = 1024;
        const std::string m = moe_opt("dynhot") ? moe_opt("dynhot") : "off";   // default off (user 2026-11-02): nothing stored, the router beat the stored maps in the sim; --moe dynhot=full|store|read
        if      (m == "full" || m == "1") { g_topic.mode = topic_t::FULL; }
        else if (m == "read")             { g_topic.mode = topic_t::READ; }
        else if (m == "off"  || m == "0") { g_topic.mode = topic_t::OFF; }
        else if (m == "store")            { g_topic.mode = topic_t::STORE; }
        else { LLAMA_LOG_WARN("moe-cache: dynhot=%s: not full, store, read or off; store\n", m.c_str()); }
        g_topic.on = g_topic.mode != topic_t::OFF && !hotmap_dir().empty() && model.tok_embd != nullptr && model.hparams.n_expert > 0 ? 1 : 0;
        g_rp.on = g_topic.on == 1 && [] { const char * v = moe_opt("replay"); return v && atoi(v) > 0; }();   // --moe replay=1 (needs the hot map store)
        if (g_topic.on == 0) {
            g_req_prompt.clear();
            return;
        }
        if (const char * m = moe_opt("dynhot-max"))   { max_rec = (size_t) atoll(m); }   // records kept: 0 = no limit (while testing)
        if (const char * m = moe_opt("dynhot-min"))   { g_topic.min_cos   = (float) atof(m); }
        if (const char * m = moe_opt("dynhot-k"))     { g_topic.k = std::max(1, atoi(m)); }
        if (const char * m = moe_opt("dynhot-merge")) { g_topic.merge_cos = (float) atof(m); }   // the 4-bit cosine at which a request joins a record (default 2: never)
        if (const char * m = moe_opt("dynhot-global")) { g_topic.store.use_glob = atoi(m) != 0; }
        if (const char * m = moe_opt("dynhot-near")) { g_topic.store.near_cos = (float) atof(m); }   // the averaged cosine from which the nearest records are used as they are
        if (const char * m = moe_opt("dynhot-same")) {   // pool: the same prompt's maps added; a number: the similarity at which a request is one stored before
            g_topic.store.same_pool = strcmp(m, "pool") == 0;
            if (std::isdigit((unsigned char) m[0])) { g_topic.store.same_cos = (float) atof(m); }
        }   // the same prompt again: replace (a better run's map) | pool (every run's counts)   // 0: the lookup blends the nearest topics only (the A/B against the global map)
        const std::string path = hotmap_path(model, "");   // <hot map dir>/<model>/: keys.js, a file of sets per layer, global.js (stores of earlier layouts stay untouched)
        g_topic.store.max_records = max_rec;   // a record is the keys and a count per expert of every layer: 49 KB on DeepSeek-V4.1, 49 MB of RAM at the default 1024
        if (!g_topic.store.open(path, (uint32_t) model.hparams.n_embd, (uint32_t) model.hparams.n_layer(), (uint32_t) model.hparams.n_expert)) {
            LLAMA_LOG_WARN("moe-cache: dynhot: %s holds maps of another shape (model or layer count): not used, not written\n", path.c_str());
        }
        static const char * what[] = { "off", "full: the nearest maps pick the cache's experts, every request stored",
            "store: requests stored, the ini's hot experts pick the cache's", "read: the nearest maps pick the cache's experts, nothing stored (frozen)" };
        LLAMA_LOG_WARN("moe-cache: dynhot: %zu records and a global map of %u requests in %s; dynhot=%s (lookup: the %d nearest%s%s; a request joins a record at "
                "cosine >= %.2f)\n", g_topic.store.size(), g_topic.store.glob_requests(), path.c_str(), what[g_topic.mode], g_topic.k,
                g_topic.min_cos > -1.0f ? tr_fmt(" at cosine >= %.2f", g_topic.min_cos).c_str() : "", g_topic.store.use_glob ? " (the global map when none is near)" : "", g_topic.merge_cos);
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
    const size_t used = u.empty() ? 0 : g_topic.store.merge(u.data(), (size_t) g_topic.k, g_topic.min_cos, heat, best, thinking);
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
    if (g_rp.on) {   // replay: record this answer, play the prompt's stored trace
        std::lock_guard<std::mutex> lk(mc->mtx);
        replay_begin(mc);
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
            if (auto fn = (int (*)(void)) cuda_obs_proc("ggml_backend_fork_driver_version_fork")) { v = "api" + std::to_string(fn()); }
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
    // LLAMA_ASYNC_LOAD sizes the pool before the card's first kernels and cuBLAS handle exist (182 MiB more free at sizing on GLM 3.0): a normal start's need is too small for it (cublasCreate failed)
    extern bool g_async_load_fork;
    return hw_key("vram_need." + gpu + ".drv" + vram_driver() + ".b" + (b ? b : "0") + with + (g_async_load_fork ? ".async" : "") + (cold ? ".cold" : ".warm"));
}

// the margin a pool leaves on a card: --moe vram-reserve-mb / margin_mb if set, else the need an earlier session measured on this card, driver and build
// plus a safety (64 MiB or an eighth of the need), else 484 MiB. Windows never goes below 484: the display and other programs take VRAM at any time (WDDM).
static size_t vram_margin(ggml_backend_dev_t dev, const std::string & section, bool cold, int64_t * stored_mb, bool * from_cold);
// the context's first reserve (llama_context::sched_reserve): the VRAM margin a card keeps, so the first micro-batch is computed from free VRAM instead of failing into it
size_t llama_moe_cache_vram_margin_fork(ggml_backend_dev_t dev) { return vram_margin(dev, "", false, nullptr, nullptr); }
static size_t vram_margin(ggml_backend_dev_t dev, const std::string & section, bool cold, int64_t * stored_mb, bool * from_cold) {
    const size_t MiB = 1024 * 1024;
    if (user_knobs().count("VRAM_RESERVE_MB") || user_knobs().count("MARGIN_MB")) {
        return (size_t) std::max(0.0, knobs().margin_mb) * MiB;
    }
    size_t m = 484 * MiB;
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
    for (auto & d : g_vram) {   // the device number is the backend's own (the OOM hook's), matched through the registry, not a name
        if (llama_fork_dev_index(d.dev) != cuda_device || !llama_fork_proc_dev(d.dev, "ggml_backend_fork_last_oom_fork")) { continue; }
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

        apply_gpu_shape(knobs().gpu_shape);   // --moe gpu-shape=N / LLAMA_MOE_CACHE_GPU_SHAPE at start; the ctl file switches it later
        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        mc->n_expert_used = (int64_t) model.hparams.n_expert_used();
        if (mc->n_expert_used > 0) { g_mb_auto = (int64_t) model.hparams.n_expert / mc->n_expert_used; }
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
        rp_prepin(mc, ggml_backend_buft_get_device(groups.begin()->first));
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
            free -= std::min(free, mc->rp_vram_reserve);   // the RAM pool's pin after the fill (rp_prepin): as if done, so the VRAM watch does not count it outside the pool either
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
                // --moe pool-floor=1: every class can hold one whole layer's experts, paid by the class with the most layers. The prompt chain refuses a pass a class cannot hold
                // (GLM 3.0, one card: "layer 3 has 111 blocks, wants 288"), so without this it never runs on such a model above a few dozen tokens.
                // shortcut: off by default and tests only, it costs the big class decode hits; make it the default if the chain against stream rows pay for it
                static const bool pool_floor = [] { const char * e = moe_opt("pool-floor"); return e && atoi(e) != 0; }();
                if (pool_floor && !order.empty()) {
                    const size_t  big = order[0];
                    const int32_t big_E = (int32_t) ll[pl.cls[big].members[0]]->ffn_up_exps->ne[2];
                    for (size_t c : order) {
                        const int32_t E = (int32_t) ll[pl.cls[c].members[0]]->ffn_up_exps->ne[2];
                        if (c == big || nblk[c] >= E) { continue; }
                        const int32_t take = (int32_t) (((size_t) (E - nblk[c]) * pl.cls[c].B + pl.cls[big].B - 1) / pl.cls[big].B);
                        if (nblk[big] - take < big_E) { LLAMA_LOG_WARN("moe-cache: pool-floor: no room for a %.2f MiB class of %d blocks\n", pl.cls[c].B / 1048576.0, E); continue; }
                        nblk[big] -= take;
                        nblk[c]    = E;
                    }
                }
                for (size_t c : order) {
                    pl.cls[c].base = total;
                    total         += (size_t) nblk[c] * pl.cls[c].B;
                }
                total += 4096;   // MMQ reads whole 512-wide tiles: the end of the last tensor may be read past
                // One buffer when the backend takes it (CUDA: no limit). A backend with a buffer limit (Vulkan: 1 GiB blocks, 2 GiB on the Radeon 780M) gets one
                // buffer per size class, each class cut to the limit: a class's blocks must stay one strided run (the kernels index them from one pointer).
                // shortcut: the compute alias and the KV tail assume one buffer; they are off on those backends (common_moe_cache_auto_impl), lift when ported
                const size_t max_b = ggml_backend_buft_get_max_size(buft);
                const bool   split = total > max_b;
                std::vector<ggml_backend_buffer_t> cbufs(pl.cls.size(), nullptr);
                ggml_backend_buffer_t pbuf = nullptr;
                if (split) {
                    // a class past the limit is split into classes of fewer layers (each a buffer of its own) rather than cut: a layer belongs to one class
                    {
                        std::vector<pool_class> ncls;
                        std::vector<int32_t>    nnblk;
                        std::vector<size_t>     norder;
                        for (size_t c : order) {
                            const pool_class & oc = pl.cls[c];
                            const int32_t cap   = (int32_t) ((max_b - 4096) / oc.B);
                            const size_t  m     = oc.members.size();
                            const size_t  parts = std::max<size_t>(1, std::min(m, cap > 0 ? ((size_t) nblk[c] + cap - 1) / cap : m));
                            for (size_t q = 0; q < parts; ++q) {
                                pool_class nc = oc;
                                nc.members.assign(oc.members.begin() + q * m / parts, oc.members.begin() + (q + 1) * m / parts);
                                nc.base = 0;
                                for (size_t i : nc.members) { pl.lay_cls[i] = ncls.size(); }
                                norder.push_back(ncls.size());
                                nnblk.push_back((int32_t) ((size_t) nblk[c] * nc.members.size() / m));
                                ncls.push_back(std::move(nc));
                            }
                            if (parts > 1) {
                                LLAMA_LOG_WARN("moe-cache: size class of %.1f MiB blocks (%zu layers, %d blocks) split into %zu buffers: the %s buffer limit is %.0f MiB\n",
                                        oc.B / 1048576.0, m, nblk[c], parts, ggml_backend_buft_name(buft), max_b / 1048576.0);
                            }
                        }
                        pl.cls = std::move(ncls);
                        nblk   = std::move(nnblk);
                        order  = std::move(norder);
                        cbufs.assign(pl.cls.size(), nullptr);
                    }
                    total = 0;
                    for (size_t c : order) {
                        const int32_t cap = (int32_t) ((max_b - 4096) / pl.cls[c].B);
                        if (nblk[c] > cap) {
                            LLAMA_LOG_WARN("moe-cache: size class of %.1f MiB blocks cut from %d to %d blocks: the %s buffer limit is %.0f MiB\n",
                                    pl.cls[c].B / 1048576.0, nblk[c], cap, ggml_backend_buft_name(buft), max_b / 1048576.0);
                            nblk[c] = cap;
                        }
                        pl.cls[c].base = 0;
                        const size_t sz = (size_t) nblk[c] * pl.cls[c].B + 4096;
                        cbufs[c] = nblk[c] > 0 ? ggml_backend_buft_alloc_buffer(buft, sz) : nullptr;
                        if (!cbufs[c]) {
                            LLAMA_LOG_WARN("%s: failed to allocate the MoE cache pool (class of %.1f MiB) on %s - cache disabled\n", __func__, sz / 1048576.0, ggml_backend_buft_name(buft));
                            for (auto * b : cbufs) { if (b) { ggml_backend_buffer_free(b); } }
                            return false;
                        }
                        total += sz;
                    }
                    pbuf = cbufs[order[0]];
                } else {
                    pbuf = ggml_backend_buft_alloc_buffer(buft, total);
                    if (!pbuf) {
                        LLAMA_LOG_WARN("%s: failed to allocate the MoE cache pool (%.1f MiB) on %s - cache disabled\n", __func__, total / 1048576.0, ggml_backend_buft_name(buft));
                        return false;
                    }
                    for (auto & b : cbufs) { b = pbuf; }
                }
                std::vector<ggml_backend_buffer_t> ubufs;   // the distinct buffers
                for (auto * b : cbufs) { if (std::find(ubufs.begin(), ubufs.end(), b) == ubufs.end()) { ubufs.push_back(b); } }
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
                for (auto * b : ubufs) { ggml_backend_buffer_clear(b, 0); }
                if (poison_on() && !split) {   // expert blocks are 5 until an upload lands; the read-past tail stays 0
                    moe_cache::alias_pool tmp; tmp.buf = pbuf;
                    pool_fill(tmp, 0, total - 4096, 5);
                }
                if (poison_on()) { ggml_sched_set_nan_hook_fork(pool_dump_all); }
                if (led_level() >= 2) { ggml_sched_set_split_hook_fork(led_split_cb); }
                if (led_on()) { ggml_backend_set_dma_hook_fork(led_dma); }
                if (led_on()) {   // CUDA graph captures and replays with their scratch range go into the ledger
                    typedef void (*graph_hook_t)(int, const void *, const char *, const void *, size_t);
                    typedef void (*set_graph_hook_t)(graph_hook_t);
                    if (auto set = (set_graph_hook_t) cuda_obs_proc("ggml_backend_fork_set_graph_hook_fork")) { set(led_cuda_graph_hook); }
                }
                for (auto * b : ubufs) { ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); }
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
                        t3[k]->buffer = cbufs[ci];
                        t3[k]->data   = (uint8_t *) ggml_backend_buffer_get_base(cbufs[ci]) + cl.base + cl.off[k];
                    }
                    ls->pub.n_slots = n;                      // the class's blocks; n marks "uncached" in the host tables (-1 in the device tables)
                    ls->pcls        = (int32_t) (pool0 + ci);
                    ls->n_cache     = slots;                  // nominal share, for the heuristics only
                }
                LLAMA_LOG_WARN("moe-cache: block pool on %s: %zu layers, %.1f MiB, size classes: %s; no blocks reserved per layer\n", ggml_backend_buft_name(buft),
                        cands.size(), total / 1048576.0, cls_txt.c_str());
                llama_moe_decision_fork("stage", "VRAM expert pool of one card sized", "");
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
                        if (!alias_all || split) { break; }
                    }
                    ap.tail_end  = split ? ap.span : pl.cls[order.back()].base + (size_t) nblk[order.back()] * pl.cls[order.back()].B;
                    ap.span_full = ap.span;
                    kv_tail_bind(ap);
                    mc->alias.push_back(ap);
                }
                for (auto * b : ubufs) { mc->bufs.push_back(b); }
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
            ls.crf.assign(n_expert, 0.0f);
            ls.crf_t.assign(n_expert, 0);
            ls.prot.assign(n_expert, 0);
            ls.relevance.assign(n_expert, 0.0f);
            ls.ghost_flag.assign(n_expert, 0);
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                if (!file_location(model, srcs[k], &ls.src_fd[k], &ls.src_offs[k])) {
                    ls.src_fd[k] = -1;
                }
            }
            ls.glob_count.assign(n_expert, 0);
            if (const char * rv = moe_opt("rampool"); rv && atoi(rv) != 0) { ls.vis.assign((size_t) n_expert * RP_VIS, 0); }
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

        // fork: LLAMA_ASYNC_LOAD: the rest of the init needs no weight bytes on the default path, so it and the graph reserves after it run under the reads. What does read them waits
        // itself: the upload workers (below), resource_probe when it measures, the predictor's routers; the rampool setup fills from them
        if (const char * rv = moe_opt("rampool"); rv && atoi(rv) != 0) { llama_model_wait_loaded_fork(&model); }
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
                const bool ph_on = [] { const char * v = moe_opt("preheat"); return v && v[0] == '1'; }();   // default off (preheat5, 2026-11-03: the server's own warm-up stops it and request 1 follows at once: 2 experts read in 11 s, while its 6.3 GiB arena leaves the page cache); =1 on
                // the preheat works per layer (src_dfd: that layer's experts stayed mapped), so a model whose first layers are pinned in tensor order still gets it; pinr wants all weights mapped
                if (gpu && !mc->staging[0] && ((pinr > 0 && !src_pinned) || ph_on) && mem_total) {
                    mc->pin_buft   = ggml_backend_dev_host_buffer_type(gpu);
                    mc->pin_on_upload = pinr > 0 && !src_pinned;
                    if (mc->pin_on_upload) {
                        mc->pin_budget = (size_t) (std::min(pinr, 10.0) / 10.0 * (double) mem_total);
                    } else {   // the preheat: the share of the mapped experts the pin planner leaves to the page cache (--moe pin-cache, 0.15)
                        double mapped = 0.0;
                        for (const auto & ls3 : mc->layers) {
                            for (const ggml_tensor * t3 : { ls3.pub.up_src, ls3.pub.gate_src, ls3.pub.down_src }) {
                                if (t3->buffer && ggml_backend_buffer_get_type(t3->buffer) != mc->pin_buft) { mapped += (double) t3->nb[2] * (double) t3->ne[2]; }   // only the tensors that stayed mapped
                            }
                        }
                        const char * pc = llama_moe_get_opt("pin-cache");
                        mc->pin_budget = (size_t) (mapped * (pc ? std::min(0.9, std::max(0.0, atof(pc))) : 0.15));
                    }
                    for (auto & ls2 : mc->layers) {
                        const ggml_tensor * ps[3] = { ls2.pub.up_src, ls2.pub.gate_src, ls2.pub.down_src };
                        for (int k2 = 0; k2 < 3; ++k2) {
                            const size_t ne = (size_t) ps[k2]->ne[2];
                            ls2.pin_ptr[k2].reset(new std::atomic<const char *>[ne]);
                            for (size_t e2 = 0; e2 < ne; ++e2) { ls2.pin_ptr[k2][e2].store(nullptr); }
                        }
                    }
                    LLAMA_LOG_WARN("moe-cache: %s: up to %.1f GiB of expert slices pinned%s, the rest stays mmap'd\n", mc->pin_on_upload ? "pinr" : "preheat", mc->pin_budget / 1073741824.0, mc->pin_on_upload ? " as they are uploaded" : " from the hot map before request 1");
                }
            }
            // --moe read-direct=1: the expert tensors that did not fit the pin budget are read from their file past the page cache (read_direct)
            size_t rd_mapped = 0;
#ifndef _WIN32
            if (const char * rd = moe_opt("read-direct"); !(rd && rd[0] == '0') && gpu && !mc->staging[0]) {   // default on (user 2026-11-02, with pin-mlock: DS prompt 263 -> 341 t/s); =0 switches it off
                ggml_backend_buffer_type_t pinned_buft = ggml_backend_dev_host_buffer_type(gpu);
                for (auto & ls2 : mc->layers) {
                    const ggml_tensor * ps[3] = { ls2.pub.up_src, ls2.pub.gate_src, ls2.pub.down_src };
                    for (int k2 = 0; k2 < 3; ++k2) {
                        if (ls2.src_fd[k2] < 0 || !ps[k2]->buffer || ggml_backend_buffer_get_type(ps[k2]->buffer) == pinned_buft) { continue; }
                        auto it = mc->rd_fds.find(ls2.src_fd[k2]);
                        if (it == mc->rd_fds.end()) {
                            it = mc->rd_fds.emplace(ls2.src_fd[k2], open_direct_fd(ls2.src_fd[k2])).first;
                            if (it->second < 0) { LLAMA_LOG_WARN("moe-cache: read-direct: no O_DIRECT on this file (%s), it stays mapped\n", strerror(errno)); }
                        }
                        ls2.src_dfd[k2] = it->second;
                        if (it->second >= 0) { rd_mapped += ggml_nbytes(ps[k2]); }
                        mc->rd_src[ps[k2]] = { it->second, ls2.src_offs[k2] };
                    }
                }
#ifdef MOE_HAVE_AIO
                if (const char * ra = moe_opt("read-aio")) { g_aio_ok.store(atoi(ra) != 0 ? 1 : 0); }
                // --moe read-blk=KB (default 256) and read-gap=KB (default 64, 0 = no gaps: plain 256K blocks in file order); set before the first read caches them
                if (const char * rb = moe_opt("read-blk")) { setenv("LLAMA_MOE_RD_BLK_KB", rb, 1); }
                // --moe read-engine=uring: io_uring instead of kernel AIO for the spaced reads (default aio; the release image's seccomp blocks io_uring)
                if (const char * re = moe_opt("read-engine")) { setenv("LLAMA_MOE_RD_ENGINE", re, 1); }
            if (const char * rq = moe_opt("read-qd"))     { setenv("LLAMA_MOE_RD_QD", rq, 1); }
                if (const char * rg = moe_opt("read-gap")) { setenv("LLAMA_MOE_RD_GAP_KB", rg, 1); }
#endif
                if (const char * rt = moe_opt("read-threads")) { mc->rd_threads = std::max(1, atoi(rt)); }
                else if (rd_mapped > 0) { mc->rd_threads = g_early_rd_threads > 0 ? g_early_rd_threads : read_threads_probe(mc->rd_fds, mc->profile); }
                if (rd_mapped > 0) {
                    mc->rd_size = max_expert + ((size_t) 1 << 18);   // a slice, what is read past it, and the block grid on both sides
                    for (size_t w = 0; w < mc->staging.size(); ++w) { mc->rd_staging.push_back(ggml_backend_buft_alloc_buffer(pinned_buft, mc->rd_size)); }
                }
            }
            // --moe pin-mlock=1: pinned weights are locked as well. A pinned page cannot be evicted, but it stays on the LRU list: reclaim scans it again and again
            // (DeepSeek at 88 GiB pinned: 95 GiB on the inactive anon list, up to 8 M pages/s scanned) and each refault of the page cache pushes more of that pressure onto real anon pages.
            if (const char * ml = moe_opt("pin-mlock"); !(ml && ml[0] == '0') && gpu) {   // default on; =0 switches it off
                ggml_backend_buffer_type_t pinned_buft = ggml_backend_dev_host_buffer_type(gpu);
                std::set<ggml_backend_buffer_t> seen;
                size_t locked = 0, failed = 0;
                const int64_t t_ml = ggml_time_us();
                for (auto & ls2 : mc->layers) {
                    for (const ggml_tensor * t : { ls2.pub.up_src, ls2.pub.gate_src, ls2.pub.down_src }) {
                        if (!t->buffer || ggml_backend_buffer_get_type(t->buffer) != pinned_buft || !seen.insert(t->buffer).second) { continue; }
                        const size_t n = ggml_backend_buffer_get_size(t->buffer);
                        const uintptr_t b0 = (uintptr_t) ggml_backend_buffer_get_base(t->buffer) & ~(uintptr_t) 4095;
                        // on fault: a buffer the loader's deferred pin left locked that way (ggml_backend_cuda_pin_defer) has the same flags, so the kernel returns at
                        // once (3.8 s for GLM 3.0's 100.5 GiB with a plain mlock, which walks every page again); the resident pages of any other buffer are locked as before
#ifdef MLOCK_ONFAULT
                        if (mlock2((void *) b0, n, MLOCK_ONFAULT) == 0) { locked += n; } else { failed += n; }
#else
                        if (mlock((void *) b0, n) == 0) { locked += n; } else { failed += n; }
#endif
                    }
                }
                LLAMA_LOG_WARN("moe-cache: pin-mlock: %.1f GiB of pinned weights locked in %.1f ms, %.1f GiB refused (RLIMIT_MEMLOCK)\n", locked / 1073741824.0, (ggml_time_us() - t_ml) / 1e3, failed / 1073741824.0);
                llama_moe_decision_fork("stage", "pinned weights lock step", "");
            }
#endif
            {   // --moe pin-lazy: the loader mapped the weights without reading them; pin the expert tensors in the background up to --moe pin-budget
                const char * pl = moe_opt("pin-lazy"); const char * pb = moe_opt("pin-budget");
                const size_t budget = pb ? (size_t) strtoull(pb, nullptr, 10) : 0;
                if (gpu && !src_pinned && pl && atoi(pl) && budget > 0) {
                    mc->lazy_buft = ggml_backend_dev_host_buffer_type(gpu);
                    mc->lazy_thr  = std::thread(pin_prefetch_run, mc, budget);
                    LLAMA_LOG_WARN("moe-cache: pin-lazy: pinning expert tensors in the background, up to %.1f GiB, the rest stays mapped\n", budget / 1073741824.0);
                }
            }
            if (llama_moe_log_profile_fork("stats")) { mc->stats_thr = std::thread(stats_status_run, mc); }
            if (llama_moe_log_profile_fork("planner")) { mc->fast_thr = std::thread(fast_window_run, mc); }
            if (const char * ph = moe_opt("preheat"); ph && ph[0] == '1' && gpu && !mc->rd_staging.empty()) { mc->preheat_thr = std::thread(preheat_run, mc); }   // default off; =1 on; rd_staging: some experts stayed mapped
            if (!mc->rd_staging.empty()) {
                LLAMA_LOG_WARN("moe-cache: read-direct: %.1f GiB of expert weights stayed mapped, read with O_DIRECT: prompt uploads on %d threads, %d cache workers\n", rd_mapped / 1073741824.0, mc->rd_threads, n_workers);
                // the CPU's rate on a mapped expert starts at the drive's measured maximum, not at the RAM rate: a cold mapped slice is read by page faults from the SSD, so the
                // first tokens' misses on mapped layers go to uploads instead of the CPU (stats.log 2026-11-08: "all on the CPU (42.5 GB/s, mapped 42.5)" until a phase was measured)
                if (rd_mapped > 0 && mc->cpu_gbs_mapped <= 0.0 && llama_moe_ssd_max_get_fork() > 0.0) {
                    mc->cpu_gbs_mapped = llama_moe_ssd_max_get_fork();
                    LLAMA_LOG_WARN("moe-cache: read-direct: the CPU reads a mapped expert at %.1f GB/s until measured (the drive's maximum), a pinned one at the RAM rate\n", mc->cpu_gbs_mapped);
                }
            }
            LLAMA_LOG_WARN("moe-cache: %d upload workers%s, %s\n", n_workers, n_links > 1 ? " (per link)" : "",
                    mc->bounce_bytes ? "memcpy -> pinned staging (bounce) -> GPU" :
                    mc->staging[0] ? "pread -> pinned staging -> GPU" :
                    src_pinned     ? "from pinned host memory (direct DMA)" :
                                     "from pageable host memory (mmap; --load-mode pin: faster uploads)");
            {
                static const char * const pol[] = { "halve", "window", "hybrid", "add", "lru", "rel", "crf", "slru" };
                char sb[320];
                snprintf(sb, sizeof sb, "slots=%d workers=%d links=%d upload=%s bounce_kb=%zu policy=%s", (int) mc->n_slots, n_workers, n_links,
                        mc->bounce_bytes ? "bounce" : mc->staging[0] ? "pread" : src_pinned ? "pinned-dma" : "mmap-pageable",
                        (size_t) (mc->bounce_bytes >> 10), pol[(int) get_policy()]);
                mc->settings = sb;
            }
        }

        for (size_t w = 0; w < mc->staging.size(); ++w)
        mc->workers.emplace_back([mc, w]() {
            // fork: LLAMA_ASYNC_LOAD: no wait for the whole read, a job waits for its own layer's buffers (layer_src_ready)
            unpin_helper();
            led_thread("up-w" + std::to_string(w));   // --moe ledger: this worker's lines say which upload thread wrote
            uint8_t * staging_ptr = mc->staging[w] ? (uint8_t *) ggml_backend_buffer_get_base(mc->staging[w]) : nullptr;
            uint8_t * rd_ptr = w < mc->rd_staging.size() && mc->rd_staging[w] ? (uint8_t *) (((uintptr_t) ggml_backend_buffer_get_base(mc->rd_staging[w]) + 4095) & ~(uintptr_t) 4095) : nullptr;
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
                            if (!layer_src_ready(mc, it->layer_idx)) {
                                continue;   // its weights are still being read
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
                    for (;;) {
                        const bool reading = g_hpend_n.load(std::memory_order_acquire) > 0;   // read before mine(): a load that ends after it is seen next round
                        if (mc->stop || mine() != mc->todo.end()) { break; }
                        // nobody notifies when a buffer's read lands: looked at again every 20 ms while a load reads
                        if (reading) { mc->wcv.wait_for(lk, std::chrono::milliseconds(20)); } else { mc->wcv.wait(lk); }
                    }
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
                race_stress_sleep();   // --moe race-stress: an upload starts late (a lend or a zone change may happen meanwhile)
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
                    if (ls.rp_tab[k]) {   // --moe rampool: the pool's block (read in now if the pool admits it), held until the copy is done
                        if (const char * pc = ggml_rampool_take_fork(srcs[k], j.expert)) {
                            ggml_backend_tensor_set(dsts[k], pc, (size_t) j.slot*dsts[k]->nb[2], sz);
                            const char * pc_from = nullptr;
                            if (const size_t pb = expert_past(mc, ls, k, j.expert, srcs[k], pc_from)) {   // the block carries the bytes read past the slice
                                ggml_tensor z = expert_past_view(dsts[k], (size_t) j.slot * dsts[k]->nb[2] + sz, pb);
                                ggml_backend_tensor_set(&z, pc + sz, 0, pb);
                            }
                            ggml_rampool_drop_fork(srcs[k], j.expert);
                            continue;
                        }
                    }
                    if (ls.pin_ptr[k]) {   // the preheat's pinned copy of this expert (no read at all): straight to the card
                        const char * pc = ls.pin_ptr[k][j.expert].load(std::memory_order_acquire);
                        const char * pc_from = nullptr;
                        const size_t pb = pc > (const char *) 1 ? expert_past(mc, ls, k, j.expert, srcs[k], pc_from) : 0;
                        if (pc > (const char *) 1 && (pb == 0 || j.expert + 1 < srcs[k]->ne[2])) {   // the preheat read the past bytes along (the last expert is never preheated)
                            ggml_backend_tensor_set(dsts[k], pc, (size_t) j.slot*dsts[k]->nb[2], sz);
                            if (pb > 0) {
                                ggml_tensor z = expert_past_view(dsts[k], (size_t) j.slot * dsts[k]->nb[2] + sz, pb);
                                ggml_backend_tensor_set(&z, pc + sz, 0, pb);
                            }
                            continue;
                        }
                    }
                    if (rd_ptr && ls.src_dfd[k] >= 0) {
                        // a slice that stayed mapped: the file into this worker's pinned buffer, then DMA. The bytes a matmul reads past the slice come with it.
                        const char * past = nullptr;
                        const size_t pb = expert_past(mc, ls, k, j.expert, srcs[k], past);
                        const bool past_in_file = pb > 0 && j.expert + 1 < srcs[k]->ne[2];
                        const size_t off = ls.src_offs[k] + (size_t) j.expert*sz;
                        if (const char * got = read_direct(ls.src_dfd[k], rd_ptr, mc->rd_size - 4096, off, sz + (past_in_file ? pb : 0))) {
                            const int64_t dr = ggml_time_us() - tr;
                            t_read += dr;
                            ggml_backend_tensor_set(dsts[k], got, (size_t) j.slot*dsts[k]->nb[2], sz);
                            if (pb > 0) {
                                ggml_tensor z = expert_past_view(dsts[k], (size_t) j.slot * dsts[k]->nb[2] + sz, pb);
                                ggml_backend_tensor_set(&z, past_in_file ? got + sz : past, 0, pb);
                            }
                            continue;
                        }
                        // a failed read falls through to the mapping below
                    }
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
                            const cuda_set_nosync_t ns = cuda_set_nosync_for(dsts[k]->buffer);
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
                                const cuda_set_nosync_t ns = cuda_set_nosync_for(dsts[k]->buffer);
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
                    const char * from = nullptr;
                    if (const size_t pb = expert_past(mc, ls, k, j.expert, srcs[k], from)) {
                        ggml_tensor z = expert_past_view(dsts[k], (size_t) j.slot * dsts[k]->nb[2] + sz, pb);
                        const cuda_set_nosync_t ns = cuda_set_nosync_for(dsts[k]->buffer);
                        if (ns) { ns(&z, from, 0, pb); nosync_used = true; }
                        else    { ggml_backend_tensor_set(&z, from, 0, pb); }
                    }
                }
                if (nosync_used) { cuda_thread_sync(); }
                race_stress_sleep();   // --moe race-stress: the copy is done, its report comes late
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
        // ini-fix (2026-10-22): the self-tuner's stored decisions are not read: a start whose tuner began from another build's values behaved unlike a cold one. Every start tunes from the defaults.
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
        resource_probe(mc, model);
        numa_probe(mc);
        if (const size_t n = profile_preload(mc, model)) {
            LLAMA_LOG_WARN("moe-cache: preloading %zu experts (usage profile)\n", n);
        } else {
            LLAMA_LOG_WARN("moe-cache: nothing preloaded: no usage profile of this model and layout in the state file, the cache fills from the first prompt\n");
        }
        rp_setup(mc, model);
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
            if (mc->pred_m > 0) { llama_model_wait_loaded_fork(&model); }   // fork: LLAMA_ASYNC_LOAD: the predictor copies the routers' weights
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
        ggml_backend_set_moe_read_fn_fork(mc->rd_src.empty() ? nullptr : moe_read_fn, mc);
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
                typedef bool (*obs_init_t)(int, int, int);
                obs_init_t init = (obs_init_t) llama_fork_proc("ggml_backend_fork_obs_prompt_init_fork");
                int n_layer = 0;
                for (const auto & l : mc->layers) { n_layer = std::max(n_layer, l.pub.il + 1); }
                const int n_expert = (int) mc->layers[0].pub.up_src->ne[2];
                for (size_t i = 0; init && i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    const int cd = cuda_dev_index(d);
                    if (cd >= 0) {
                        mc->pf_count_on = init(cd, n_layer, n_expert) || mc->pf_count_on;
                    }
                }
                mc->pf_count_layers  = n_layer;
                mc->pf_count_experts = n_expert;
            }
        }
        g_cache = mc;
        g_init_done = true;

        miss_dev_setup(mc);
        LLAMA_LOG_WARN("%s: MoE expert cache enabled: %zu layers, slots/layer %d..%d, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(),
                std::min_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                std::max_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                mc->max_inserts, vram/1024.0/1024.0);
        llama_moe_decision_fork("stage", "expert cache enabled", "");
    }();
}

// JIT miss offload. Runs on the CPU (graph custom op) as soon as layer `pub`'s router ids are on the host, before the
// cache chain of that layer is launched. For m uncached experts of size S, CPU alone takes m*S/rc. Uploading k of them
// over the layer's link (rate rl) while the CPU computes the rest at the contended rate rcc = min(rc, ram - rl) (then rc
// once the link is done) finishes at T(k); the k with the smallest T wins, 0 when nothing beats the CPU alone. All rates
// are measured (link and CPU phase rates live, RAM budget by the startup probe). Uploaded experts become cache entries
// (same eviction score as the step's swaps), so the upload also serves later tokens.
// A CPU-repacked expert tensor (non-host buffer of the CPU device) holds another byte layout than the GPU's blocks: ->data must not be uploaded. Only the upload workers can serve it, from the file.
static bool experts_in_file_layout(const llama_moe_cache_layer_fork * pub) {
    return ggml_backend_buffer_is_host(pub->up_src->buffer) && ggml_backend_buffer_is_host(pub->gate_src->buffer) && ggml_backend_buffer_is_host(pub->down_src->buffer);
}
// A JIT upload of an expert that stayed mapped (no pinned copy, not in the pool): the slice is read with O_DIRECT into a pinned ring slot and DMA'd from there. A DMA
// from the mapping itself page-faults the SSD through the page cache and bounces (GLM 3.5 2026-11-08, 101 GiB pinned, 17.9 GiB mapped: 257k-390k major faults per 12k
// request, decode 10.7 t/s against 15-16 with the tail cached), and the pool design wants no second copy in the page cache at all (user 2026-11-08). The ring holds 3
// slots per expert a token uses (every slice of one token's misses in flight at once); a slot is reused once the copy that last read it is done (its event).
// Returns where the bytes start, or nullptr when the weight has no O_DIRECT source (the caller falls back to the mapping and the count says so).
static const char * jit_direct_src(moe_cache * mc, ggml_backend_t be, const ggml_tensor * w, size_t off, size_t n, int n_used) {
    if (mc->rd_size == 0 || mc->rd_src.find(w) == mc->rd_src.end()) { return nullptr; }
    if (mc->jit_ring.empty()) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(be));
        for (int i = 0; i < 3 * std::max(1, n_used) && buft; ++i) {
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, mc->rd_size);
            if (!b) { break; }
            mc->jit_ring.push_back({ b, nullptr, nullptr });
        }
        if (mc->jit_ring.empty()) { mc->rd_size = 0; return nullptr; }   // no pinned slot: the mapping, and no second try
        LLAMA_LOG_WARN("moe-cache: jit: %zu pinned slots of %.1f MiB (%.0f MiB) for O_DIRECT reads of mapped experts, never a DMA from the mapping\n",
                mc->jit_ring.size(), mc->rd_size / 1048576.0, mc->jit_ring.size() * mc->rd_size / 1048576.0);
    }
    auto & s = mc->jit_ring[mc->jit_ring_i++ % mc->jit_ring.size()];
    if (s.ev) {   // the copy that last read this slot
        ggml_backend_event_synchronize(s.ev);
        if (s.be != be) { ggml_backend_event_free(s.ev); s.ev = nullptr; }   // an event belongs to its device
    }
    const int64_t t0 = ggml_time_us();
    const void * p = moe_read_fn(w, off, n, ggml_backend_buffer_get_base(s.buf), ggml_backend_buffer_get_size(s.buf), mc);
    if (!p) { mc->jit_map_n++; return nullptr; }
    const int64_t dt = ggml_time_us() - t0;
    mc->jit_rd_n++; mc->jit_rd_b += n; mc->jit_rd_us += (uint64_t) dt;
    if (dt > 0) { const double g = (double) n / (double) dt / 1e3; mc->jit_rd_gbs = mc->jit_rd_gbs > 0 ? 0.9 * mc->jit_rd_gbs + 0.1 * g : g; }
    s.be = be;
    return (const char *) p;
}
static void jit_direct_done(moe_cache * mc, ggml_backend_t be) {   // the copies from the slot taken last are queued: it is free again when they are done
    auto & s = mc->jit_ring[(mc->jit_ring_i - 1) % mc->jit_ring.size()];
    if (!s.ev) { s.ev = ggml_backend_event_new(ggml_backend_get_device(be)); }
    if (s.ev) { ggml_backend_event_record(s.ev, be); }
}

void llama_moe_cache_jit_fork(const llama_moe_cache_layer_fork * pub, const ggml_tensor * ids) {
    moe_cache * mc = g_cache;
    static const bool jit_all_mode = [] { const char * v = moe_opt("jit-all"); return v && atoi(v) > 0; }();   // --moe jit-all: every batch the cache serves as decode, for deterministic replay
    // Upload or CPU, per layer and per call, from measured costs (user 2026-10-31: no fixed rule): an upload costs bytes / link rate whatever the expert's token
    // count, the CPU costs bytes / CPU rate for every token routed to the expert, so a prompt call (many tokens per expert) uploads nearly everything and a decode
    // step (one token) keeps the CPU where it is faster. In a prompt call the block for an upload is a free one or one of a layer that already ran (its chain is
    // done: this layer's router ids came back through the same stream), never one of the layers still to come. A prompt call is several tokens with at most one
    // output (llama_context::decode).
    const bool prompt = g_prompt_call.load(std::memory_order_relaxed);
    // --moe rampool: this layer's misses into the pool before the CPU kernel or an upload reads them (no lock of the cache held: the reads take ms). Every decode step;
    // a prompt call only where it leaves before need_left below can (no JIT, no link rate yet): the CPU kernel would read those misses through the mapping
    // (GLM 3.5 map-guard 2026-11-08: the 2-token call at start segfaulted in 6 CPU threads on the PROT_NONE expert mapping)
    auto need_all = [&]() {
        if (!mc || !mc->rp_on) { return; }
        for (const auto & l : mc->layers) {
            if (&l.pub != pub || !l.rp_tab[0]) { continue; }
            std::vector<int32_t> need;
            for (int64_t t = 0; t < ids->ne[1]; ++t) {
                for (int64_t i = 0; i < ids->ne[0]; ++i) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                    if (e >= 0 && e < (int32_t) l.expert_slot.size() && l.expert_slot[e] < 0 && std::find(need.begin(), need.end(), e) == need.end()) { need.push_back(e); }
                }
            }
            if (!need.empty()) {   // the three tensors in one call: their reads run together and the step waits once, not three times in a row
                const void * keys[3] = { pub->up_src, pub->gate_src, pub->down_src };
                ggml_rampool_need_fork(keys, 3, need.data(), (int) need.size());
            }
            break;
        }
    };
    if (!prompt) { need_all(); }
    if (!mc || knobs().jit == 0 || !experts_in_file_layout(pub)) {   // repacked: the misses stay with the CPU, which runs that layout
        if (prompt) { need_all(); }
        return;
    }
    std::unique_lock<std::mutex> lock(mc->mtx);   // released before a pool wait (need_left below)
    layer_state * lsp = nullptr;
    for (auto & l : mc->layers) {
        if (&l.pub == pub) { lsp = &l; break; }
    }
    if (!lsp) {
        return;
    }
    layer_state & ls = *lsp;
    mc->phase_mapped = ls.src_dfd[0] >= 0 && !ls.rp_tab[0];   // this layer's CPU phase reads mapped experts (moe_phase_cb keeps their rate apart); the pool's are in RAM after the need above
    const bool pool = ls.pub.jit_n > 0; // slower-link layer: the pool on the fastest-link GPU instead of its own cache
    const int  lk   = pool ? 0 : ls.link;
    ggml_backend_t be = mc->link_be[lk];
    const double rl = mc->gbs_link[lk] > 0 ? mc->gbs_link[lk] : jit_all_mode ? 1.0 : 0.0;   // jit-all: before the link is measured too
    if (!be || rl <= 0) {
        if (prompt) { lock.unlock(); need_all(); }
        return;
    }
    std::vector<int32_t> cur, miss, inflight;
    std::vector<int32_t> ntok(ls.expert_slot.size(), 0);   // tokens of this call routed to each expert: the CPU computes every one of them, an upload costs the same for 1 or 100
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (e >= 0 && e < (int32_t) ntok.size()) { ntok[e]++; }
        }
    }
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (e < 0 || e >= (int32_t) ls.expert_slot.size() || std::find(cur.begin(), cur.end(), e) != cur.end()) {
                continue;
            }
            cur.push_back(e);
            if (pool && ls.pool_slot[e] >= 0) {
                mc->n_pool_hit++;
            } else if (ls.expert_slot[e] < 0 && (!ls.queued[e] || jit_all_mode)) {   // jit-all: an expert still in flight too (its own copy now; publish_job drops the late one)
                miss.push_back(e);
            } else if (ls.expert_slot[e] < 0) {
                inflight.push_back(e);
            }
        }
    }
    mc->n_jit_miss += miss.size();
    if (prompt) { g_pjit_layers++; g_pjit_miss += miss.size(); }
    // an expert whose upload is still in flight (the load's preload) has no slot when the slot ids are read: the CPU kernel computes it, so with the pool on it needs a
    // pool block too (GLM 3.5 map-guard 2026-11-08: the 2-token call at start read blk.3.ffn_gate_exps through the mapping while 3694 preloads were queued)
    auto need_inflight = [&]() {
        if (!prompt || !mc->rp_on || !ls.rp_tab[0] || inflight.empty()) { return; }
        const void * keys[3] = { pub->up_src, pub->gate_src, pub->down_src };
        if (lock.owns_lock()) { lock.unlock(); }
        ggml_rampool_need_fork(keys, 3, inflight.data(), (int) inflight.size());
    };
    if (miss.empty()) {
        need_inflight();
        return;
    }
    const double S   = (double) (pub->up_src->nb[2] + pub->gate_src->nb[2] + pub->down_src->nb[2]);
    const double rc  = cpu_gbs_now(mc) * 1e3; // bytes/us, 0 = not measured: nothing is computed on the CPU then
    const double rcc = knobs().ddr_gbs > 0.0 ? std::min(rc, std::max(0.0, knobs().ddr_gbs * 1e3 - rl * 1e3)) : rc;   // the RAM budget, when it is known
    const double rlb = rl * 1e3;
    const int    m   = (int) miss.size();
    // A prompt pass's misses that get no VRAM block run on the CPU. With the pool on they come from a pool block, admitted and stamped before the kernel (user 2026-11-08: no
    // expert is read through the mapping; before this, the CPU read them from the mapping: no need call stamps a prompt pass). The cache lock goes first: the reads take ms and
    // the upload workers need the lock. Called once, last, on every path out of here.
    auto need_left = [&](int from) {
        if (!prompt || !mc->rp_on || !ls.rp_tab[0]) { return; }
        if (from < m) {
            const void * keys[3] = { pub->up_src, pub->gate_src, pub->down_src };
            lock.unlock();
            ggml_rampool_need_fork(keys, 3, miss.data() + from, m - from);
        }
        need_inflight();
    };
    // A mapped expert (its tensor stayed mapped, no pinned copy) is read by the CPU through page faults from the SSD, at the rate measured for such layers (cpu_gbs_mapped),
    // not at the RAM rate: its bytes count as the RAM-rate equivalent (user 2026-11-07: the planner is "crippled by ... assumptions"; GLM 3.5 decode sent 2531 of 5651
    // single misses to the CPU at 0.5-0.9 GB/s while the x16 link ran at 10%).
    const double rcm = mc->cpu_gbs_mapped > 0 ? mc->cpu_gbs_mapped * 1e3 : rc;
    // held by the pool only when all three tensors are (the pool admits an expert whole since 2026-11-08; before, an expert with its down tensor out counted as held and the
    // CPU kernel read that tensor through the mapping: DS 12k 1000 major faults/s)
    auto in_pool = [&](int32_t e) { return ls.rp_tab[0] && ls.rp_tab[0][e] && ls.rp_tab[1] && ls.rp_tab[1][e] && ls.rp_tab[2] && ls.rp_tab[2][e]; };
    auto mapped = [&](int32_t e) { return ls.src_dfd[0] >= 0 && !(ls.pin_ptr[0] && ls.pin_ptr[0][e].load(std::memory_order_acquire)) && !in_pool(e); };
    // bytes at the RAM rate that cost the same CPU time. With the pool on, the mapping is not a source (user 2026-11-08: no second copy in the page cache): an expert the pool does
    // not hold is uploaded, read O_DIRECT into the pinned ring (jit_direct_src), never computed from page faults, so its CPU cost is unbounded here and the sort puts it first
    auto cpu_eq = [&](int32_t e) { return mapped(e) && mc->rp_on ? 1e300 : (double) ntok[e] * S * (mapped(e) && rcm > 0 ? rc / rcm : 1.0); };
    int n_mapped = 0;
    for (int32_t e : miss) { n_mapped += mapped(e); }
    // the experts dearest on the CPU (most tokens, mapped ones first: their time grows with tokens and with faults, an upload's does not) are uploaded first
    std::sort(miss.begin(), miss.end(), [&](int32_t a, int32_t b) { const double ca = cpu_eq(a), cb = cpu_eq(b); return ca != cb ? ca > cb : score(ls, a) > score(ls, b); });
    std::vector<double> cpu_tail(m + 1, 0.0);   // cpu_tail[k]: RAM-rate bytes the CPU still reads if the first k are uploaded (each expert once per token routed to it)
    for (int i = m - 1; i >= 0; --i) { cpu_tail[i] = cpu_tail[i + 1] + cpu_eq(miss[i]); }
    // a mapped expert's upload runs at the O_DIRECT read rate when that is below the link's (jit_direct_src, measured), a pinned or pool one at the link's rate
    const double rdm = mc->jit_rd_gbs > 0 ? std::min(rlb, mc->jit_rd_gbs * 1e3) : rlb;
    std::vector<double> up_head(m + 1, 0.0);   // up_head[k]: the upload time of the first k
    for (int i = 0; i < m; ++i) { up_head[i + 1] = up_head[i] + S / (mapped(miss[i]) ? rdm : rlb); }
    auto T = [&](int k) {
        const double tl = up_head[k], bc = cpu_tail[k];
        if (k == 0) { return bc / rc; }
        return bc <= tl * rcc ? std::max(tl, rcc > 0 ? bc / rcc : 1e300) : tl + (bc - tl * rcc) / rc;
    };
    int kb = 0;
    for (int k = 1; k <= m; ++k) {   // measured, not a rule: the k that finishes the layer first, for a prompt call or a decode step alike
        if (T(k) < T(kb)) { kb = k; }
    }
    // --moe jit-all=1 (user 2026-10-29: "no point until you achieve fully deterministic moe replay"): every miss of this token goes to the GPU, none to
    // the CPU: the CPU's float results differ from the GPU's, and greedy runs of one prompt diverged after a median 5 tokens when the CPU computed some
    static const bool jit_all = [] { const char * v = moe_opt("jit-all"); return v && atoi(v) > 0; }();
    if (jit_all || rc <= 0.0) { kb = m; }
    {   // stats.log SWT: the call's misses switch between the CPU, a mix and uploads (user 2026-11-02: "in stats.log log switching from missed moes from cpu to uploads"); one line per switch, not per layer
        static std::atomic<int> last_mode[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };   // per link: the cards' pass sizes differ, the rule can flip on one and not on the other
        const int mode = kb == 0 ? 0 : kb == m ? 2 : 1;
        if (last_mode[lk & 7].exchange(mode) != mode) {
            static const char * const nm[3] = { "CPU", "CPU and uploads", "uploads" };
            llama_moe_stats_log_fork("SWT", format("%s call, link %d (%.1f GB/s), layer %d: misses go to %s (%d of %d uploaded, %d mapped, %d tokens; layer takes %.2f ms with them, %.2f ms all on the CPU (%.1f GB/s, mapped %.1f), %.2f ms all uploaded)",
                    prompt ? "prompt" : "decode", lk, rl, pub->il, nm[mode], kb, m, n_mapped, (int) ids->ne[1], T(kb) / 1e3, T(0) / 1e3, rc / 1e3, rcm / 1e3, T(m) / 1e3).c_str());
        }
    }
    if (kb == 0) {
        need_left(0);
        return;
    }
    int n_up = 0, left = kb;   // miss[] is ordered by tokens routed to the expert (most first), then by score; miss[left..m) stay with the CPU
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
                left = i;
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
                const char * rp = mc->rp_on ? ggml_rampool_use_fork(srcs[k], id) : nullptr;   // the pool's pinned block, not the mapping (see the slot path below)
                const char * dr = rp ? nullptr : jit_direct_src(mc, be, srcs[k], (size_t) id*sz, sz, (int) ids->ne[0]);   // else an O_DIRECT read into a pinned slot
                ggml_backend_tensor_set_async(be, dsts[k], rp ? rp : dr ? dr : (const char *) srcs[k]->data + (size_t) id*sz, (size_t) ps*dsts[k]->nb[2], sz);
                if (dr) { jit_direct_done(mc, be); }
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
        if (prompt) { g_pjit_up += n_up; if (pub->il >= 0 && pub->il < 256) { g_pjit_left_l[pub->il] += (uint64_t) (m - n_up); g_pjit_pool_l[pub->il] = 1; } }
        need_left(left);
        return;
    }
    for (int i = 0; i < kb; ++i) {
        const int32_t id = miss[i];
        if (!prompt && knobs().admit_jit > 0 && ls.win_count[id] < knobs().admit_jit) {
            continue;  // not seen often enough in the last 64 tokens: the CPU computes it, it takes no slot
        }
        // slot: a free block, else the lowest-scored cached expert of THIS layer this token doesn't use (mid-graph: another layer's
        // table may already be on the GPU, so its blocks are not taken here; the step at the token boundary moves blocks between layers)
        double best;
        const size_t jli = (size_t) (&ls - mc->layers.data());
        const slot_pick pk = pick_slot(mc, jli, [&](layer_state & lo, size_t o, int32_t x) {
            if (prompt) {
                // first a free block, then a layer that already ran or this layer's blocks the pass does not use; only when none is left (the first layers of a
                // card have no layer behind them) a block of the layer furthest ahead, so the layers that follow soonest keep what they cached
                if (pinned(mc, lo, x) || (o == jli && std::find(cur.begin(), cur.end(), x) != cur.end())) { return -1.0; }
                return prompt_victim_rank(mc, lo, o, jli, x, cur.size() * 8 >= ls.expert_slot.size() * 7);
            }
            return o != jli || pinned(mc, lo, x) || std::find(cur.begin(), cur.end(), x) != cur.end() ? -1.0 : score(lo, x);
        }, best);
        const int32_t slot = pk.slot;
        if (slot < 0) {
            left = i;
            break;
        }
        const int32_t victim = pk.owner >= 0 ? mc->layers[pk.owner].slot_expert[slot] : -1;
        if (victim >= 0 && prompt) {
            evict_slot(mc, pk.owner >= 0 ? mc->layers[pk.owner] : ls, slot);   // any block of a past layer: no pay-back test, the pass needs this expert now
        } else if (victim >= 0) {
            // JIT=1: admitted only as the step's swap would be (no churn): hotter than the victim by the link's pay-back
            // margin; JIT=2: any miss may evict the lowest-scored expert this token doesn't use
            if (knobs().jit < 2 && ((knobs().big != 0 && ls.glob_count[id] < ls.glob_count[victim]) ||
                    score(ls, id) < score(ls, victim) + pay_back(mc->link_margin[ls.link]))) {
                continue;
            }
            evict_slot(mc, ls, slot);
        }
        pool_take(mc, jli, slot);
        led_upload(mc, "jit-up", jli, id, slot, false);
        // async on the chain's stream: the chain (launched next) runs after the copies
        const ggml_tensor * srcs[3] = { pub->up_src, pub->gate_src, pub->down_src };
        ggml_tensor *       dsts[3] = { pub->up_c,   pub->gate_c,   pub->down_c   };
        for (int k = 0; k < 3; ++k) {
            const size_t sz = srcs[k]->nb[2];
            // the pool's pinned block when it holds the expert (stamped: it stays until the graph after next): with rampool the tensor's own pointer is the
            // unpopulated mapping, and a DMA from it page-faults the SSD and bounces (GLM 3.5 rampool5: first 1938-token call 43.8 s against 17.6, 78 GB of faults)
            const char * from = nullptr;
            const size_t pb = expert_past(mc, ls, k, id, srcs[k], from);
            const bool past_next = pb > 0 && from == (const char *) srcs[k]->data + (size_t) (id + 1) * sz;   // the bytes past the slice are the next expert's: in the file right after it, read along
            const char * rp = mc->rp_on ? ggml_rampool_use_fork(srcs[k], id) : nullptr;
            const char * dr = rp ? nullptr : jit_direct_src(mc, be, srcs[k], (size_t) id*sz, sz + (past_next ? pb : 0), (int) ids->ne[0]);   // else an O_DIRECT read into a pinned slot
            ggml_backend_tensor_set_async(be, dsts[k], rp ? rp : dr ? dr : (const char *) srcs[k]->data + (size_t) id*sz, (size_t) slot*dsts[k]->nb[2], sz);
            if (pb) {
                ggml_tensor z = expert_past_view(dsts[k], (size_t) slot*dsts[k]->nb[2] + sz, pb);
                ggml_backend_tensor_set_async(be, &z, rp ? rp + sz : dr && past_next ? dr + sz : from, 0, pb);   // the pool's block carries the past bytes
            }
            if (dr) { jit_direct_done(mc, be); }
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
    if (prompt) { g_pjit_up += n_up; if (pub->il >= 0 && pub->il < 256) { g_pjit_left_l[pub->il] += (uint64_t) (m - n_up); } }
    if (prompt && n_up < m) {   // no block to take: these few are all the CPU still computes in a prompt
        static int shown = 0;
        if (shown++ < 5) { LLAMA_LOG_WARN("moe-cache: prompt: layer %d: %d of %d missed experts found no block (past layers' blocks and free ones used up), the CPU computes them\n", pub->il, m - n_up, m); }
    }
    need_left(left);
}

// The cache of a prompt pass (user 2026-10-31: prefill and decode want their own cache logic, hot experts stay but nothing stands in the way of a miss). The decode chain
// costs a prompt per token, whatever it finds in the cache: its GPU part runs in 4-token chunks on the vector kernels and the CPU chain's activations run over every token
// of the pass even with nothing left to compute (Qwen3.6 2 ms per token, qni4 5, GLM and MiMo 16, DeepSeek 19, about ten times the prompt path's compute). The prompt chain is
// the GPU alone: one matmul per expert tensor over the whole pass (the same kernels as the prompt path), and every expert the pass misses is uploaded here, right behind the
// router, before that matmul. The CPU computes nothing, no admit test, no pay-back test: a hit costs nothing, a miss costs its upload once.
static bool pp_chain_on() {
    static const bool on = [] { const char * e = moe_opt("pp-chain"); return !e || atoi(e) != 0; }();   // --moe pp-chain=0: prompt passes on the decode chain, as before
    return on;
}
static bool pp_chain_allowed() {
    return pp_chain_on() && g_prompt_call.load(std::memory_order_relaxed) && !g_pp_short.load(std::memory_order_relaxed) && !g_pp_dec.load(std::memory_order_relaxed);
}

// the blocks of a layer's class an expert may take now: not lent to a graph, not under the KV zone, no upload in flight
static int64_t pp_blocks(moe_cache * mc, layer_state & ls) {
    int64_t n = 0;
    if (ls.pcls >= 0) {
        const auto & ps = mc->pool[ls.pcls];
        for (int32_t b = 0, nb = ps.n_active(); b < nb; ++b) {
            const int32_t o = ps.owner[b];
            n += o == -1 || (o >= 0 && !mc->layers[o].slot_in_flight[b] && !mc->layers[o].is_stream[b]);
        }
        return n;
    }
    for (int32_t b = 0; b < ls.pub.n_slots; ++b) { n += !ls.slot_in_flight[b] && !ls.is_stream[b]; }
    return n;
}

// A chain pass starts with nothing on its way: the decode step's uploads not started yet are dropped, the running ones land and are published, so "not in a block" means
// "not on the card" for the uploads below.
static void pp_settle(moe_cache * mc) {
    std::unique_lock<std::mutex> wlk(mc->wmtx);
    for (auto & j : mc->todo) {
        j.skipped = true;
        j.done    = true;
        mc->done.push_back(j);
    }
    mc->todo.clear();
    mc->dcv.wait(wlk, [mc]() { return mc->in_flight == 0; });
    std::lock_guard<std::mutex> lk(mc->mtx);
    for (const auto & j : mc->done) { publish_job(mc, j); }
    mc->done.clear();
}

// graph build: 1, this layer of a prompt pass takes the prompt chain. The pass may use every distinct expert of the layer at once, so the blocks must hold that many (no quota: all
// blocks of the class count, a past layer's are free to take); -1, a layer that cannot hold them stays on the prompt path, which streams its experts through the staging slots;
// 0, not a prompt pass.
// the links' backends, from the scheduler: before the first split callback (the first graph of a run) link_be is empty and the prompt chain was refused for that graph
void llama_moe_cache_link_backends_fork(ggml_backend_sched_t sched) {
    moe_cache * mc = g_cache;
    if (!mc || !sched) { return; }
    for (int k = 0; k < mc->n_links; ++k) {
        for (int i = 0; !mc->link_be[k] && i < ggml_backend_sched_get_n_backends(sched); ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_get_device(b) == mc->link_dev[k]) { mc->link_be[k] = b; }
        }
    }
}
int llama_moe_cache_pp_chain_fork(const llama_moe_cache_layer_fork * pub, int64_t n_tokens, int64_t n_used) {
    moe_cache * mc = g_cache;
    if (!mc || !pp_chain_allowed() || pub->jit_n > 0 || !experts_in_file_layout(pub)) {   // repacked: the chain has no CPU branch and its uploads copy ->data; the plain path keeps the CPU
        return 0;
    }
    if (!g_pp_graph) { pp_settle(mc); }
    std::lock_guard<std::mutex> lock(mc->mtx);
    for (auto & l : mc->layers) {
        if (&l.pub != pub) {
            continue;
        }
        // --moe pp-off=F: F of the pairs stay off the card, so the pass needs at most (1 - F) of the experts' blocks (the least used ones are what stays off)
        static const double off_f = [] { const char * e = moe_opt("pp-off"); return e ? atof(e) : 0.0; }();
        const int64_t want = (int64_t) std::ceil((1.0 - off_f) * (double) std::min<int64_t>((int64_t) l.expert_slot.size(), n_tokens * n_used));
        if (!mc->link_be[l.link] || pp_blocks(mc, l) < want) {
            static int n_log = 0;
            if (n_log++ < 3) { LLAMA_LOG_WARN("moe-cache: prompt chain refused: layer %d has %lld blocks, wants %lld\n", pub->il, (long long) pp_blocks(mc, l), (long long) want); }
            return -1;
        }
        g_pp_want  = n_tokens * n_used;
        g_pp_graph = true;
        return 1;
    }
    return 0;
}

// A stream pass (no prompt chain) of a prompt call, for a layer cached behind a slower link: the fastest link's card takes what the home card does not hold. 0 fastest = no mix.
static bool pp_mix_opt() {
    // On by default again: the P8 crash (CUDA illegal memory access on GLM 3.5 12k, rows 23770, 23786, 23788) was the mul_mat_id tile_spec seams left
    // unwritten for the -1 ids of a split (fork/docs/mix-crash.md, fixed in mmq.cu / mmf.cu); forced mix ran clean after it. --moe pp-mix=0 turns it off.
    static const bool on = [] {
        const char * e = moe_opt("pp-mix");
        const bool v = !e || atoi(e) != 0;
        if (!v) { LLAMA_LOG_WARN("moe-cache: feas: mix: off by --moe pp-mix=0\n"); }
        return v;
    }();
    return on;
}
static bool pp_mix_allowed() {
    return pp_mix_opt() && pp_chain_on() && g_prompt_call.load(std::memory_order_relaxed) && g_pp_mix_call.load(std::memory_order_relaxed);
}
ggml_backend_dev_t llama_moe_cache_pp_mix_fork(const llama_moe_cache_layer_fork * pub) {
    moe_cache * mc = g_cache;
    if (!mc || !pp_mix_allowed() || pub->jit_n > 0) {
        return nullptr;
    }
    for (auto & l : mc->layers) {
        if (&l.pub == pub) {
            if (l.link == 0 || !mc->link_dev[0]) { return nullptr; }
            g_pp_mix_graph = true;
            return mc->link_dev[0];
        }
    }
    return nullptr;
}

// Right before a graph runs (behind its lend): the tables of a mix graph are what the two chains split the experts by, the home card's by the device table, the ring's staging
// by the host table. Nothing may change either during the pass, so both start from the same state: the uploads on their way land, the rest of the decode step's are dropped.
void llama_moe_cache_pp_freeze_fork() {
    moe_cache * mc = g_cache;
    if (!mc || !g_pp_mix_built) {
        return;
    }
    pp_settle(mc);
    flush_tables();
    if (led_level() >= 2) {   // what the home cards read in this pass, by their tables: one line per expert, and led_upload checks the block is still the layer's
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            const auto & ls = mc->layers[li];
            if (ls.link == 0) { continue; }
            for (int32_t e = 0; e < (int32_t) ls.expert_slot.size(); ++e) {
                if (ls.expert_slot[e] >= 0) { led_upload(mc, "mix-read", li, e, ls.expert_slot[e], false); }
            }
        }
    }
}

// --moe verify=N (debug, slow): weights never change and the host holds the true copy, so every block the tables say holds an expert is read back and compared with it, past
// the slice too (what a matmul reads behind its last row). 1: before a graph of the prompt mix or chain, 2: before every graph of 64 tokens or more, 3: before every graph. A bad
// block is named with its layer, expert and bytes, and goes into the ledger: `pool_ledger.py LEDGER --addr CARD:MIB` then says who wrote it. Clean means the experts are right and the
// fault is in the kernels or their inputs. Nothing is stopped, the run goes on.
void llama_moe_cache_verify_fork(ggml_backend_sched_t sched, int32_t n_tokens) {
    static const int mode = [] { const char * e = moe_opt("verify"); return e ? atoi(e) : 0; }();
    moe_cache * mc = g_cache;
    if (mode <= 0 || !mc) { return; }
    if (mode < 3 && !(mode >= 2 && n_tokens >= 64) && !g_pp_mix_built && !g_pp_built) { return; }
    const int64_t t0 = ggml_time_us();
    ggml_backend_sched_synchronize(sched);
    for (int i = 0; i < mc->n_links; ++i) { if (mc->link_be[i]) { ggml_backend_synchronize(mc->link_be[i]); } }
    static const char * const parts[3] = { "up", "gate", "down" };
    std::vector<uint8_t> got;
    size_t n_read = 0, n_bad = 0, n_bad_past = 0;
    int said = 0;
    std::lock_guard<std::mutex> lk(mc->mtx);
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
        const ggml_tensor * dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
        for (int32_t e = 0; e < (int32_t) ls.expert_slot.size(); ++e) {
            const int32_t slot = ls.expert_slot[e];
            if (slot < 0 || ls.slot_in_flight[slot]) { continue; }
            n_read++;
            for (int k = 0; k < 3; ++k) {
                if (!srcs[k] || !dsts[k] || !ggml_backend_buffer_is_host(srcs[k]->buffer)) { continue; }
                const size_t sz  = srcs[k]->nb[2];
                const size_t off = (size_t) slot * dsts[k]->nb[2];
                const char * from = nullptr;
                const size_t pb  = expert_past(mc, ls, k, e, srcs[k], from);
                const size_t n   = sz + pb;
                got.resize(n);
                ggml_backend_tensor_get(dsts[k], got.data(), off, sz);
                if (pb) { ggml_tensor z = expert_past_view(dsts[k], off + sz, pb); ggml_backend_tensor_get(&z, got.data() + sz, 0, pb); }
                const uint8_t * want = (const uint8_t *) srcs[k]->data + (size_t) e * sz;
                const uint8_t * next = (const uint8_t *) from;
                size_t diff = 0, first = n;
                for (size_t i = 0; i < n; ++i) {
                    if (got[i] != (i < sz ? want[i] : next[i - sz])) { diff++; first = std::min(first, i); }
                }
                if (diff == 0) { continue; }
                const bool in_slice = first < sz;
                (in_slice ? n_bad : n_bad_past)++;
                size_t lo = 0, hi = 0;
                const moe_cache::alias_pool * ap = ls.pcls >= 0 ? led_block(mc, ls.pcls, slot, lo, hi) : nullptr;
                const std::string who = tr_fmt("L%zu il %d e%d %s block %d: %zu of %zu bytes differ, the first at +%zu (%s)", li, ls.pub.il, e, parts[k], slot, diff, n, first, in_slice ? "in the slice" : "only past the slice");
                if (ap) { led_line(led_card(*ap), "verify-bad", lo, hi, who); }
                if (said++ < 16) { LLAMA_LOG_ERROR("pool-verify: BAD %s [%zu, %zu) of %s: %s\n", dsts[k]->name, lo, hi, ap ? led_card(*ap) : "?", who.c_str()); }
            }
        }
    }
    LLAMA_LOG_WARN("pool-verify: a %d-token graph: %zu cached experts read back and compared with the host's copy in %.0f ms: %zu slices bad, %zu only past the slice\n",
            (int) n_tokens, n_read, (ggml_time_us() - t0) / 1e3, n_bad, n_bad_past);
}

bool llama_moe_cache_pp_chain_pass_fork() { return g_pp_built || g_pp_k_built; }
// the dense split stays out of this pass: its path is the chain or a prediction, or the dense split was measured at this size and something else is faster, or nothing else was measured to weigh it against
bool llama_moe_cache_pp_chain_takes_fork(int64_t n_tokens) {
    return g_prompt_call.load(std::memory_order_relaxed) && pp_chain_on() && (n_tokens <= llama_moe_cache_max_batch_fork() || ggml_backend_pred_get_fork() > 0.0 || g_pp_no_dense.load(std::memory_order_relaxed));
}

// after the lend of a built graph (llama_context::process_ubatch): the blocks the graph took are gone. True: a layer of the chain lost too many, the prompt call leaves the chain
// and the graph is built again (the check at build time could not see this lend; a layer short of blocks would have experts nobody computes).
void llama_moe_cache_pp_decide_fork(uint32_t n_tokens, uint32_t ub_max);
bool llama_moe_cache_pp_short_fork() {
    moe_cache * mc = g_cache;
    if (!mc || !g_pp_built) {
        return false;
    }
    std::string refused;
    {
        std::lock_guard<std::mutex> lock(mc->mtx);
        for (auto & l : mc->layers) {
            if (l.pub.up_c && pp_blocks(mc, l) < std::min<int64_t>((int64_t) l.expert_slot.size(), g_pp_want) && !moe_opt("pp-off")) {   // --moe pp-off: misses go to the miss device, the chain is not short of blocks
                g_pp_short = true;
                g_pp_built = false;
                refused = format("prompt chain: layer %d keeps %lld blocks beside the graph, the pass may use %lld experts: the call is decided again without the chain",
                        l.pub.il, (long long) pp_blocks(mc, l), (long long) std::min<int64_t>((int64_t) l.expert_slot.size(), g_pp_want));
                break;
            }
        }
    }
    if (refused.empty()) { return false; }
    LLAMA_LOG_WARN("moe-cache: %s\n", refused.c_str());
    llama_moe_stats_log_fork("CHN", refused.c_str());
    // the pass has not run: the call is decided again among the other arms (Qwen3.8 200 tokens, both-splits 2026-11-08: the refused chain fell to the plain stream,
    // 1510 ms, while the mix was calculated at 302 ms and took 574 on the next request)
    llama_moe_cache_pp_decide_fork(g_pp_last_n.load(std::memory_order_relaxed), g_pp_last_ub.load(std::memory_order_relaxed));
    return true;
}

// the misses of a chain layer, on the chain's stream, right behind its router (a graph custom op: the ids are on the host). The blocks: free ones, then the layers that already
// ran and this layer's experts the pass does not use, then the layer furthest ahead: the same order as the JIT's prompt rule. An expert a slow layer or the hot set holds on to
// is the last choice, never refused: nothing in a prompt may stop a miss.
static std::atomic<int64_t> g_ppoff_pairs{0}, g_ppoff_dev{0}, g_ppoff_all{0};   // --moe pp-off: kept-off pairs, of them on the device, all pairs
void llama_moe_cache_pp_upload_fork(const llama_moe_cache_layer_fork * pub, const ggml_tensor * ids) {
    moe_cache * mc = g_cache;
    GGML_ASSERT(mc != nullptr);
    if (mc->rp_on) {   // this pass's misses into the pool first, no lock of the cache held (the reads take ms): the uploads below are DMAs from its pinned blocks, not from the unpopulated mapping
        std::vector<int32_t> need;
        {
            std::lock_guard<std::mutex> lock(mc->mtx);
            for (const auto & l : mc->layers) {
                if (&l.pub != pub || !l.rp_tab[0]) { continue; }
                std::vector<uint8_t> seen(l.expert_slot.size(), 0);
                for (int64_t t = 0; t < ids->ne[1]; ++t) {
                    for (int64_t i = 0; i < ids->ne[0]; ++i) {
                        const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                        if (e >= 0 && e < (int32_t) seen.size() && !seen[e]) { seen[e] = 1; if (l.expert_slot[e] < 0) { need.push_back(e); } }
                    }
                }
                break;
            }
        }
        if (!need.empty()) {
            const void * keys[3] = { pub->up_src, pub->gate_src, pub->down_src };
            ggml_rampool_need_fork(keys, 3, need.data(), (int) need.size());
        }
    }
    {
        std::lock_guard<std::mutex> lock(mc->mtx);
        layer_state * lsp = nullptr;
        for (auto & l : mc->layers) {
            if (&l.pub == pub) { lsp = &l; break; }
        }
        GGML_ASSERT(lsp != nullptr);
        layer_state & ls = *lsp;
        const size_t  jli = (size_t) (&ls - mc->layers.data());
        std::vector<uint8_t> used(ls.expert_slot.size(), 0);
        std::vector<int64_t> cnt(ls.expert_slot.size(), 0);
        for (int64_t t = 0; t < ids->ne[1]; ++t) {
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e >= 0 && e < (int32_t) used.size()) { used[e] = 1; cnt[e]++; }
            }
        }
        std::vector<int32_t> miss;
        size_t n_visited = 0;
        for (int32_t e = 0; e < (int32_t) used.size(); ++e) {
            n_visited += used[e] ? 1 : 0;
            if (used[e] && ls.expert_slot[e] < 0) { miss.push_back(e); }
        }
        // --moe pp-off=F[,pp-off-dev=R]: the misses with the fewest tokens (the most upload per token of work), up to F of the pass's expert-token pairs, stay on the
        // host; of those the R most used (by pairs) run on the miss device (up_m), the rest on the CPU. The graph built the decode chain's shape for this (mc_ppk).
        static const double off_f = [] { const char * e = moe_opt("pp-off"); return e ? atof(e) : 0.0; }();
        static const double off_r = [] { const char * e = moe_opt("pp-off-dev"); return e ? atof(e) : 0.5; }();
        if (off_f > 0.0) {
            ls.pub.miss_on_dev.assign(used.size(), 0);
            std::sort(miss.begin(), miss.end(), [&](int32_t a, int32_t b) { return cnt[a] < cnt[b]; });
            const double budget = off_f * (double) (ids->ne[0] * ids->ne[1]);
            double off = 0; size_t k = 0;
            while (k < miss.size() && off + (double) cnt[miss[k]] <= budget + 1e-9) { off += (double) cnt[miss[k]]; ++k; }
            if (ls.pub.up_m) {   // most used first onto the device, up to R of the kept pairs
                double dev = 0;
                for (size_t j = k; j-- > 0; ) {
                    if (dev + (double) cnt[miss[j]] > off_r * off + 1e-9) { continue; }
                    dev += (double) cnt[miss[j]];
                    ls.pub.miss_on_dev[miss[j]] = 1;
                }
                g_ppoff_dev += (int64_t) dev;
            }
            g_ppoff_pairs += (int64_t) off; g_ppoff_all += ids->ne[0] * ids->ne[1];
            miss.erase(miss.begin(), miss.begin() + (ptrdiff_t) k);
        }
        mc->n_jit_miss += miss.size();
        g_pjit_layers++;
        g_pjit_miss += miss.size();
        g_ppc_picked += n_visited; g_ppc_all += used.size(); g_ppc_miss += miss.size();
        if (!miss.empty()) {
            struct cand { double rank; int32_t slot, owner; };
            std::vector<cand> c;
            auto add = [&](int32_t o, int32_t b) {
                layer_state & lo = mc->layers[o];
                if (lo.slot_in_flight[b] || lo.is_stream[b]) { return; }
                const int32_t x = lo.slot_expert[b];
                if (x < 0) { c.push_back({ -1.0, b, -1 }); return; }
                if ((size_t) o == jli && used[x]) { return; }
                const double r = prompt_victim_rank(mc, lo, (size_t) o, jli, x, n_visited * 8 >= used.size() * 7);
                c.push_back({ pinned(mc, lo, x) ? 2e12 + r : r, b, o });
            };
            if (ls.pcls >= 0) {
                const auto & ps = mc->pool[ls.pcls];
                for (int32_t b = 0, nb = ps.n_active(); b < nb; ++b) {
                    const int32_t o = ps.owner[b];
                    if (o == -1) { c.push_back({ -2.0 - (double) b, b, -1 }); }   // the highest free block first: prompts borrow the pool from its start
                    else if (o >= 0) { add(o, b); }
                }
            } else {
                for (int32_t b = 0; b < ls.pub.n_slots; ++b) { add((int32_t) jli, b); }
            }
            if (c.size() < miss.size()) {
                GGML_ABORT("moe-cache: prompt chain: layer %d: %zu blocks for %zu missed experts (the check at the graph's build failed to see this)", pub->il, c.size(), miss.size());
            }
            std::partial_sort(c.begin(), c.begin() + (ptrdiff_t) miss.size(), c.end(), [](const cand & a, const cand & b) { return a.rank < b.rank; });
            ggml_backend_t be = mc->link_be[ls.link];
            const double   S   = (double) (pub->up_src->nb[2] + pub->gate_src->nb[2] + pub->down_src->nb[2]);
            const double   rlb = mc->gbs_link[ls.link] * 1e3;
            const ggml_tensor * srcs[3] = { pub->up_src, pub->gate_src, pub->down_src };
            ggml_tensor *       dsts[3] = { pub->up_c,   pub->gate_c,   pub->down_c   };
            for (size_t i = 0; i < miss.size(); ++i) {
                const int32_t id = miss[i], slot = c[i].slot;
                if (c[i].owner >= 0) { evict_slot(mc, mc->layers[c[i].owner], slot); }
                pool_take(mc, jli, slot);
                led_upload(mc, "chain-up", jli, id, slot, false);
                for (int k = 0; k < 3; ++k) {
                    const size_t sz = srcs[k]->nb[2];
                    const char * rp = mc->rp_on ? ggml_rampool_use_fork(srcs[k], id) : nullptr;   // the pool's pinned block (asked above), not the mapping; the block carries the past bytes too
                    ggml_backend_tensor_set_async(be, dsts[k], rp ? rp : (const char *) srcs[k]->data + (size_t) id*sz, (size_t) slot*dsts[k]->nb[2], sz);
                    const char * from = nullptr;
                    if (const size_t pb = expert_past(mc, ls, k, id, srcs[k], from)) {
                        ggml_tensor z = expert_past_view(dsts[k], (size_t) slot*dsts[k]->nb[2] + sz, pb);
                        ggml_backend_tensor_set_async(be, &z, rp ? rp + sz : from, 0, pb);
                    }
                }
                ls.slot_expert[slot]   = id;
                ls.expert_slot[id]     = slot;
                ls.slot_last_use[slot] = ++mc->clock;
                set_table_entry(ls.pub, id, slot);
                ls.cached_since[id] = mc->n_steps + 1;
                ls.up_t[id]    = ggml_time_us();
                ls.up_cost[id] = rlb > 0 ? (float) (S / rlb) : 0.0f;
                ls.up_hits[id] = 0;
                mc->up_bytes  += (int64_t) S; mc->link_bytes[ls.link] += (int64_t) S;
            }
            mc->n_jit += miss.size();
            mc->n_jit_layers++;
            g_pjit_up += miss.size();
            if (rlb > 0) { g_pp_up_us.store(g_pp_up_us.load() + (double) miss.size() * S / rlb); }
        }
    }
    // the CPU's expert op is not in this graph and the GPU's prompt counters do not see slot ids: the observer gets the pass's ids here (heat for the answer)
    g_pp_obs = true;
    moe_obs_cb(pub->gate_src->name, ids, nullptr, mc);
    g_pp_obs = false;
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
// --moe miss-dev=NAME: the experts the cache does not hold are computed on this device (a GPU sharing the host memory) instead of on the CPU: every expert tensor is imported in
// place, one buffer per tensor (no copy, the Vulkan import needs page-aligned ends). All or nothing: a refused import leaves the CPU chain as it was.
// one miss chain (up, gate, activation, down of one expert for one token) round trip on a device, us: the median of 16, the input written and the output read like the real hop
static double miss_chain_us(ggml_backend_dev_t dev, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) {
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { return -1.0; }
    if (auto set = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads")) {
        set(be, std::max(1, (int) std::thread::hardware_concurrency() / 2));   // shortcut: half the hardware threads, not the decode's count; upgrade if the verdict is ever close
    }
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, up->ne[0], 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
    ggml_tensor * u = ggml_mul_mat_id(ctx, up, x, ids), * g = ggml_mul_mat_id(ctx, gate, x, ids);
    ggml_tensor * out = ggml_mul_mat_id(ctx, down, ggml_swiglu_split(ctx, g, u), ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    double res = -1.0;
    if (ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be)) {
        std::vector<float>   xv(up->ne[0], 0.5f), yv(ggml_nelements(out));
        const std::vector<int32_t> idv = { 16, -1, -1, -1, -1, -1, -1, -1 };   // one miss, as at a high hit rate
        std::vector<double> t;
        for (int i = 0; i < 16; ++i) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { t.clear(); break; }
            ggml_backend_tensor_get(out, yv.data(), 0, ggml_nbytes(out));
            t.push_back((double) (ggml_time_us() - t0));
        }
        if (!t.empty()) { std::sort(t.begin(), t.end()); res = t[t.size() / 2]; }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
    return res;
}

static void miss_dev_setup(moe_cache * mc) {
    const char * name = moe_opt("miss-dev");
    if (!name) { return; }
    const bool autop = strcmp(name, "auto") == 0;
    ggml_backend_dev_t dev = autop ? nullptr : ggml_backend_dev_by_name(name);
    for (size_t i = 0; autop && !dev && i < ggml_backend_dev_count(); ++i) {   // a GPU sharing the host memory that can import it, whichever backend it is on
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_dev_props pr;
        ggml_backend_dev_get_props(d, &pr);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU && pr.caps.buffer_from_host_ptr) { dev = d; }
    }
    if (!dev) { return; }
    name = ggml_backend_dev_name(dev);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr) { LLAMA_LOG_WARN("moe-cache: miss-dev %s cannot import host memory: the misses stay on the CPU\n", name); return; }
    ggml_init_params ip = { mc->layers.size() * 3 * ggml_tensor_overhead(), nullptr, true };
    mc->miss_ctx = ggml_init(ip);
    const size_t page = 65536;   // covers minImportedHostPointerAlignment (4 KiB) and the Windows view granularity
    size_t bytes = 0;
    auto import = [&](const ggml_tensor * t) -> ggml_tensor * {
        if (!t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) { return nullptr; }
        const uintptr_t s0 = (uintptr_t) t->data / page * page, e0 = GGML_PAD((uintptr_t) t->data + ggml_nbytes(t), 4096);
        ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (void *) s0, e0 - s0, ggml_nbytes(t));
        if (!buf) { return nullptr; }
        mc->miss_bufs.push_back(buf);
        ggml_tensor * m = ggml_new_tensor(mc->miss_ctx, t->type, GGML_MAX_DIMS, t->ne);
        ggml_set_name(m, t->name);
        if (ggml_backend_tensor_alloc(buf, m, (char *) ggml_backend_buffer_get_base(buf) + ((uintptr_t) t->data - s0)) != GGML_STATUS_SUCCESS) { return nullptr; }
        bytes += e0 - s0;
        return m;
    };
    bool ok = true;
    const char * ml = moe_opt("miss-layers");   // --moe miss-layers=N: only the first N layers (the driver may not hold them all resident); the others stay on the CPU
    int left = ml ? atoi(ml) : INT_MAX;
    for (auto & l : mc->layers) {
        if (left-- <= 0) { break; }
        l.pub.up_m   = ok ? import(l.pub.up_src)   : nullptr;
        l.pub.gate_m = l.pub.up_m   ? import(l.pub.gate_src) : nullptr;
        l.pub.down_m = l.pub.gate_m ? import(l.pub.down_src) : nullptr;
        ok = l.pub.down_m != nullptr;
    }
    if (!ok) {
        for (auto & l : mc->layers) { l.pub.up_m = l.pub.gate_m = l.pub.down_m = nullptr; }
        for (auto * b : mc->miss_bufs) { ggml_backend_buffer_free(b); }
        mc->miss_bufs.clear();
        LLAMA_LOG_WARN("moe-cache: miss-dev %s refused an import of the host weights: the misses stay on the CPU\n", name);
        return;
    }
    if (autop) {   // the faster of the device and the CPU, measured with the real weights
        const auto & p0 = mc->layers[0].pub;
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const double us_m = miss_chain_us(dev, p0.up_m, p0.gate_m, p0.down_m), us_c = cpu ? miss_chain_us(cpu, p0.up_src, p0.gate_src, p0.down_src) : -1.0;
        LLAMA_LOG_WARN("moe-cache: one miss: %s %.0f us, CPU %.0f us: %s\n", name, us_m, us_c, us_m > 0 && us_c > 0 && us_m < us_c ? "misses on the device" : "misses stay on the CPU");
        if (!(us_m > 0 && us_c > 0 && us_m < us_c)) {
            for (auto & l : mc->layers) { l.pub.up_m = l.pub.gate_m = l.pub.down_m = nullptr; }
            for (auto * b : mc->miss_bufs) { ggml_backend_buffer_free(b); }
            mc->miss_bufs.clear();
            return;
        }
    }
    LLAMA_LOG_WARN("moe-cache: misses computed on %s: %.1f GiB of expert weights imported in %zu buffers\n", name, bytes / 1073741824.0, mc->miss_bufs.size());
}

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
        // each copy timed alone, the fastest kept (user 2026-11-03: "upload moes sized data to gpu 3 times and pick and keep in mem fastest"): contention and clock dips only lower a reading, and the one
        // value serves the placement, the prefill split and every rate the planner calculates with
        for (int r = 0; r < 4; ++r) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(&t, src, 0, n);
            gbs = std::max(gbs, (double) n / (double) std::max<int64_t>(1, ggml_time_us() - t0) / 1e3);
        }
    }
    if (hbuf) { ggml_backend_buffer_free(hbuf); }
    if (dbuf) { ggml_backend_buffer_free(dbuf); }
    g_early_gbs.push_back({ d, gbs });
    return gbs;
}

#if defined(SR_HAVE_AIO) || defined(_WIN32)
// The async engine's best shape on the model's drive: block size, gap and reads in flight (O_DIRECT, about 0.15 s per combination), measured once per drive and kept in [machine] as
// rd_blk_kb.<dev>, rd_gap_kb.<dev>, rd_qd.<dev>; every reader starts from them (sr_blk, sr_gap, sr_qd in llama-spaced-read-fork.h). The smallest block within 7% of the best wins, as in
// read_threads_probe, and ssd_max_gbs.<dev> is lifted when the engine beats the pread probe (user 2026-11-07: "find these vals if not present", "all our uploads as fast as optimal").
static void rd_shape_bench(const char * path, unsigned long long dev, const std::string & sec) {
    const std::string kb = format("rd_blk_kb.%llu", dev), kg = format("rd_gap_kb.%llu", dev), kq = format("rd_qd.%llu", dev);
    std::string b, g, q;
    if (moe_state_get(sec, kb, b) && moe_state_get(sec, kg, g) && moe_state_get(sec, kq, q) && atoi(b.c_str()) > 0 && atoi(q.c_str()) > 0) {
        g_sr_blk = (size_t) atol(b.c_str()) << 10; g_sr_gap = (size_t) atol(g.c_str()) << 10; g_sr_qd = atoi(q.c_str());
        LLAMA_LOG_WARN("moe-cache: machine: SSD reads %s KiB blocks, %s KiB gaps, %s in flight (stored)\n", b.c_str(), g.c_str(), q.c_str());
        return;
    }
    const int dfd = open_direct(path);
    if (dfd < 0) { return; }
#ifdef _WIN32
    struct { long long st_size; } st;
    st.st_size = _lseeki64(dfd, 0, SEEK_END);   // stat's 32-bit size fails on a model file
    if (st.st_size < (1LL << 30)) { close(dfd); return; }
#else
    struct stat st;
    if (fstat(dfd, &st) != 0 || st.st_size < ((off_t) 1 << 30)) { close(dfd); return; }
#endif
#ifdef _WIN32
    const size_t range = (size_t) 64 << 20;   // Windows: the block size is what matters (256 KiB: 28 s cold load of 11.4 GB, 16 MiB: 12.7 s), so the blocks tried are big
#else
    const size_t range = (size_t) 32 << 20;
#endif
    const double s0 = llama_moe_span_now_fork();
    const size_t file_end = (size_t) st.st_size / 4096 * 4096;
    void * buf = nullptr;
#ifdef _WIN32
    buf = _aligned_malloc(range, 4096);
    if (!buf) { close(dfd); return; }
#else
    if (posix_memalign(&buf, 4096, range) != 0) { close(dfd); return; }
#endif
    struct combo { size_t blk, gap; int qd; double gbs; };
    std::vector<combo> cs;
#ifdef _WIN32
    for (int qd : { 8, 2 }) { for (size_t blk : { (size_t) 1 << 20, (size_t) 4 << 20, (size_t) 16 << 20 }) { cs.push_back({ blk, 0, qd, 0.0 }); } }
#else
    for (int qd : { 32, 8 }) { for (size_t blk : { (size_t) 256 << 10, (size_t) 1 << 20 }) { for (size_t gap : { (size_t) 64 << 10, (size_t) 0 }) { cs.push_back({ blk, gap, qd, 0.0 }); } } }
#endif
    size_t off = (size_t) st.st_size / 2 / 4096 * 4096;   // the middle of the file: read_threads_probe read the heads of its regions (O_DIRECT: the page cache is out, the drive's own cache is not)
    for (combo & c : cs) {
        const int64_t t0 = ggml_time_us(); size_t bytes = 0;
        while (ggml_time_us() - t0 < 150000) {
            if (off + range > file_end) { off = 4096; }
            if (!async_read_range(dfd, (uint8_t *) buf, off, range, range, c.qd, c.blk, c.gap)) { bytes = 0; break; }
            bytes += range; off += range;
        }
        c.gbs = bytes ? (double) bytes / ((double) (ggml_time_us() - t0) / 1e6) / 1e9 : 0.0;
    }
#ifdef _WIN32
    _aligned_free(buf);
#else
    free(buf);
#endif
    close(dfd);
    double best = 0.0;
    for (const combo & c : cs) { best = std::max(best, c.gbs); }
    if (best <= 0.0) { LLAMA_LOG_WARN("moe-cache: machine: SSD read shape: the async engine failed, the readers keep their defaults\n"); return; }
    std::sort(cs.begin(), cs.end(), [](const combo & x, const combo & y) { return x.blk != y.blk ? x.blk < y.blk : x.qd != y.qd ? x.qd < y.qd : x.gap > y.gap; });   // preference at a tie: small block, few in flight, gaps on
    const combo * pick = nullptr; std::string row;
    for (const combo & c : cs) { if (!pick && c.gbs >= 0.93 * best) { pick = &c; } row += format(" %zuK/%zuK/%d:%.2f", c.blk >> 10, c.gap >> 10, c.qd, c.gbs); }
    g_sr_blk = pick->blk; g_sr_gap = pick->gap; g_sr_qd = pick->qd;
    LLAMA_LOG_WARN("moe-cache: machine: SSD read shape GB/s by block/gap/in flight%s -> %zuK/%zuK/%d (the smallest within 7%% of the best %.2f GB/s)\n", row.c_str(), pick->blk >> 10, pick->gap >> 10, pick->qd, best);
    std::vector<std::pair<std::string, std::string>> kv = {{ kb, std::to_string(pick->blk >> 10) }, { kg, std::to_string(pick->gap >> 10) }, { kq, std::to_string(pick->qd) }};
    const std::string km = format("ssd_max_gbs.%llu", dev);
    std::string mx;
    if (!moe_state_get(sec, km, mx) || atof(mx.c_str()) < best) { kv.push_back({ km, format("%.3f", best) }); llama_moe_ssd_max_fork(best); }
    moe_state_set(sec, kv);
    llama_moe_span_fork("machine benchmark: SSD read shape (block, gap, reads in flight)", s0, llama_moe_span_now_fork(), 0, "SSD probe");
}
#endif

// The machine's three bandwidths, measured as early as possible (right after the model file size is read, before the layers are placed) and kept in the state file section [machine]
// (user 2026-11-05: "measure maxssd speed as early as possible, the same for big vram uploads, measure and store achieved ram read bw and pcie achieved bw"). Every later span of stats.log
// that reads the SSD logs its rate as a share of the SSD maximum, so an inefficient big read stands out.
//   ssd_max_gbs.<dev>          O_DIRECT 1 MiB reads, best of 1..hardware threads/2 (read_threads_probe), per block device of the model file; measured once, then stored
//   ram_read_gbs.<threads>     sequential read of 1 GiB by hardware threads/2 threads, best of three passes; measured once, then stored
//   h2d_gbs.<gpu index>.<name> pinned host to device, 64 MiB copies, the fastest of four (early_link_gbs); measured at every start (0.1 s), stored with the value before
//   rd_blk_kb/rd_gap_kb/rd_qd.<dev>  the async engine's read shape on the drive (rd_shape_bench); measured once, then stored
// A GPU's own memory speed: one 256 MiB copy inside its memory (read + write), the fastest of four. Next to the RAM read rate it says whether an expert cache
// can pay at all: a GPU that shares system RAM (780M, Strix Halo) copies at about RAM speed, a discrete card several times faster (user 2026-11-08:
// "measure gpus latency and speed and compare to ram and if not bigger disable moe cache")
static double g_ram_read_gbs = 0.0;
static std::vector<std::pair<ggml_backend_dev_t, double>> g_vram_gbs;
static double vram_copy_gbs(ggml_backend_dev_t d) {
    const size_t n = 256u << 20;
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(d), 2 * n);
    if (!b) { return 0.0; }
    char * base = (char *) ggml_backend_buffer_get_base(b);
    ggml_tensor x = {}, y = {};
    for (ggml_tensor * t : { &x, &y }) {
        t->type = GGML_TYPE_I8; t->buffer = b;
        t->ne[0] = (int64_t) n; t->ne[1] = t->ne[2] = t->ne[3] = 1; t->nb[0] = 1; t->nb[1] = t->nb[2] = t->nb[3] = n;
    }
    x.data = base; y.data = base + n;
    double best = 0.0;
    for (int r = 0; r < 4; ++r) {   // the first copy warms up; contention only lowers a reading
        const int64_t t0 = ggml_time_us();
        ggml_backend_tensor_copy(&x, &y);
        best = std::max(best, 2.0 * (double) n / (double) std::max<int64_t>(1, ggml_time_us() - t0) / 1e3);
    }
    ggml_backend_buffer_free(b);
    return best;
}
LLAMA_API double llama_moe_ram_read_gbs_fork(void) { return g_ram_read_gbs; }
LLAMA_API double llama_moe_vram_gbs_fork(ggml_backend_dev_t d) {
    for (const auto & e : g_vram_gbs) { if (e.first == d) { return e.second; } }
    return 0.0;
}

LLAMA_API void llama_moe_early_bench_fork(const char * model_path) {
    if (!moe_state_enabled() || !model_path || !*model_path) { return; }
    const double t0 = llama_moe_span_now_fork();
    const std::string sec = "machine";
    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    std::string v;
#ifndef _WIN32
    struct stat st;
    if (stat(model_path, &st) == 0) {
        const std::string key = format("ssd_max_gbs.%llu", (unsigned long long) st.st_dev);
        const double s0 = llama_moe_span_now_fork();
        // a drive reads slower as it fills (user 2026-11-10: "remeasure baseline max model read speed occasionally"): the stored maximum holds for the fill it was measured at,
        // 5 points of fill later it is measured again
        const std::string fkey = format("ssd_fill_pct.%llu", (unsigned long long) st.st_dev);
        struct statvfs fs;
        const int fill = statvfs(model_path, &fs) == 0 && fs.f_blocks > 0 ? (int) (100.0 - 100.0 * (double) fs.f_bavail / (double) fs.f_blocks) : -1;
        const char * fresh = llama_moe_get_opt("refresh-ssd");   // a file just rewritten reads at another rate: measure again
        const bool fill_moved = (fresh && atoi(fresh) > 0) || (fill >= 0 && (!moe_state_get(sec, fkey, v) || std::abs(atoi(v.c_str()) - fill) >= 5));
        if (!fill_moved && moe_state_get(sec, key, v) && atof(v.c_str()) > 0.0 && moe_state_get(sec, format("read_threads.%u", hw), v) && atoi(v.c_str()) > 0) {
            g_early_rd_threads = atoi(v.c_str());
            moe_state_get(sec, key, v);
            llama_moe_ssd_max_fork(atof(v.c_str()));
            LLAMA_LOG_WARN("moe-cache: machine: SSD maximum %.2f GB/s, %d read threads (stored)\n", atof(v.c_str()), g_early_rd_threads);
        } else {
            const int dfd = open_direct(model_path);
            if (dfd >= 0) {
                std::map<int, int> fds{{ 0, dfd }};
                g_early_rd_threads = read_threads_probe(fds, sec, /*measure =*/ true);   // this drive is new to the state: measured, not another drive's stored answer; stores read_threads.<hw> and ssd_max_gbs.<hw>
                close(dfd);
                if (moe_state_get(sec, format("ssd_max_gbs.%u", hw), v)) { moe_state_set(sec, {{ key, v }}); }   // read_threads_probe keys the maximum by threads; the early bench by drive
                if (fill >= 0) { moe_state_set(sec, {{ fkey, std::to_string(fill) }}); }
                LLAMA_LOG_WARN("moe-cache: machine: SSD maximum measured %.2f GB/s at %d%% full\n", llama_moe_ssd_max_get_fork(), fill);
                llama_moe_span_fork("machine benchmark: SSD maximum (O_DIRECT, best thread count)", s0, llama_moe_span_now_fork(), 0, "SSD probe");
            }
        }
#ifdef SR_HAVE_AIO
        rd_shape_bench(model_path, (unsigned long long) st.st_dev, sec);
#endif
    }
#else
    {
        struct _stat64 st;
        if (_stat64(model_path, &st) == 0) { rd_shape_bench(model_path, (unsigned long long) st.st_dev, sec); }
    }
#endif
    {   // RAM read: the pinned weights and the stream ring are read by the CPU and the DMA engines at this rate at most
        const std::string key = format("ram_read_gbs.%u", hw / 2);
        const double s0 = llama_moe_span_now_fork();
        if (moe_state_get(sec, key, v) && atof(v.c_str()) > 0.0) {
            g_ram_read_gbs = atof(v.c_str());
            LLAMA_LOG_WARN("moe-cache: machine: RAM read %.2f GB/s with %u threads (stored)\n", atof(v.c_str()), hw / 2);
        } else {
            const size_t n = (size_t) 1 << 30;
            std::vector<uint64_t> buf(n / 8, 1);   // touched here, so the passes read resident memory
            const unsigned nt = hw / 2;
            double best = 0.0;
            for (int pass = 0; pass < 3; ++pass) {
                std::vector<std::thread> th;
                std::atomic<uint64_t> sink{0};
                const int64_t t1 = ggml_time_us();
                for (unsigned i = 0; i < nt; ++i) {
                    th.emplace_back([&, i]() {
                        const size_t a = buf.size() / nt * i, b = i + 1 == nt ? buf.size() : buf.size() / nt * (i + 1);
                        uint64_t x = 0;
                        for (size_t k = a; k < b; ++k) { x += buf[k]; }
                        sink += x;
                    });
                }
                for (auto & t : th) { t.join(); }
                best = std::max(best, (double) n / (double) std::max<int64_t>(1, ggml_time_us() - t1) / 1e3);
            }
            moe_state_set(sec, {{ key, format("%.3f", best) }});
            g_ram_read_gbs = best;
            LLAMA_LOG_WARN("moe-cache: machine: RAM read %.2f GB/s with %u threads (measured)\n", best, nt);
            llama_moe_span_fork("machine benchmark: RAM read (1 GiB, all threads)", s0, llama_moe_span_now_fork(), n, "RAM probe");
        }
    }
    {   // PCIe: every card, measured now (the value is also the one the placement and the splits use, see early_link_gbs)
        const double s0 = llama_moe_span_now_fork();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!ggml_backend_dev_is_gpu(d)) { continue; }
            const double g = early_link_gbs(d);
            const std::string key = format("h2d_gbs.%zu.%s", i, ggml_backend_dev_name(d));
            std::string old;
            moe_state_get(sec, key, old);
            moe_state_set(sec, {{ key, format("%.3f", g) }});
            LLAMA_LOG_WARN("moe-cache: machine: PCIe upload to %s %.2f GB/s (stored before: %s)\n", ggml_backend_dev_name(d), g, old.empty() ? "none" : old.c_str());
            const std::string kv = format("vram_gbs.%zu.%s", i, ggml_backend_dev_name(d));   // the memory speed of a card does not change: measured once, then stored
            double vg = moe_state_get(sec, kv, old) ? atof(old.c_str()) : 0.0;
            const bool stored = vg > 0.0;
            if (!stored) { vg = vram_copy_gbs(d); if (vg > 0.0) { moe_state_set(sec, {{ kv, format("%.3f", vg) }}); } }
            g_vram_gbs.push_back({ d, vg });
            LLAMA_LOG_WARN("moe-cache: machine: %s memory %.2f GB/s (%s), RAM read %.2f GB/s\n", ggml_backend_dev_name(d), vg, stored ? "stored" : "measured", g_ram_read_gbs);
        }
        llama_moe_span_fork("machine benchmark: PCIe uploads to every card", s0, llama_moe_span_now_fork(), 0, "PCIe probe");
    }
    llama_moe_decision_fork("stage", "machine bandwidths measured (SSD maximum, RAM read, PCIe uploads)", "");
    GGML_UNUSED(t0);
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
    if (!moe_state_section_kv(moe_state_section(model), kv, true)) {   // the section common set (file name + size), else the model's own name; the hot.* keys are written there
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
                const ggml_tensor * s3[3] = { L.pub.up_src, L.pub.gate_src, L.pub.down_src };
                const char * from = nullptr;
                if (const size_t pb = expert_past(mc, L, k, m.e, s3[k], from)) {
                    ggml_tensor z = expert_past_view(t3[k], (size_t) s * t3[k]->nb[2] + m.data[k].size(), pb);
                    ggml_backend_tensor_set(&z, from, 0, pb);
                }
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
            }
            return;
        }
    }
    mc->lv_c = (mc->lv_c + 1) % (int) nv;
    knobs().*t.f = t.vals[mc->lv_c];
    apply_predict(mc);
}

// Expert tables against block owners: every cached entry of a layer's table points at a block of that layer's size class that the layer owns and that holds
// that expert. A table entry into a block lent to a graph (-2), under the KV zone (-3), free (-1) or another layer's is how a stale mapping makes a matmul
// read someone else's bytes (uploads, tables and slots: 21 of the 145 bug-log rows by 2026-10-28). Host tables are the source of the device copies, and only
// the step (this thread) changes them and the owners. Every step with --moe ledger or poison, else every 64th step (~23k compares on DeepSeek 4.1, ~20 us).
// The first violations are logged with the numbers (and a ledger line); with --moe poison the process stops at the first, so the pool dump shows it.
static void tables_check(moe_cache * mc) {
    static int n_bad = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        const layer_state & ls = mc->layers[li];
        if (ls.pcls < 0 || ls.pcls >= (int32_t) mc->pool.size() || !ls.pub.host_table) { continue; }
        const auto & ps = mc->pool[ls.pcls];
        const int32_t * h = (const int32_t *) ls.pub.host_table->data;
        const int64_t ne = ggml_nelements(ls.pub.host_table);
        for (int64_t e = 0; e < ne; ++e) {
            const int32_t s = h[e];
            if (s < 0 || s >= ls.pub.n_slots) { continue; }   // uncached (n_slots) or in the JIT pool (n_slots + 1)
            const int32_t own = s < (int32_t) ps.owner.size() ? ps.owner[s] : -9;
            const int32_t has = s < (int32_t) ls.slot_expert.size() ? ls.slot_expert[s] : -9;
            if (own == (int32_t) li && has == (int32_t) e) { continue; }
            if (n_bad++ < 20) {
                LLAMA_LOG_ERROR("moe-cache: table check: layer %zu expert %lld -> block %d of class %d, but the block's owner is %d (%s) and it holds expert %d\n",
                        li, (long long) e, s, ls.pcls, own, own == -1 ? "free" : own == -2 ? "lent to a graph" : own == -3 ? "under the KV zone" : own == -9 ? "out of range" : "another layer", has);
                size_t lo = 0, hi = 0;
                if (const moe_cache::alias_pool * ap = led_block(mc, ls.pcls, s, lo, hi)) {
                    led_line(led_card(*ap), "OVERLAP", lo, hi, tr_fmt("table: L%zu e%lld points at block %d, owner %d holds e%d", li, (long long) e, s, own, has));
                }
            }
            if (poison_on()) { GGML_ABORT("moe-cache: table check failed (layer %zu expert %lld block %d owner %d holds %d)", li, (long long) e, s, own, has); }
        }
    }
}

void llama_moe_cache_step_fork() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    if (led_on() || poison_on() || mc->n_steps % 64 == 0) {
        tables_check(mc);
    }
    if (led_on()) {   // the graph that ran is over (the context synchronised before this step): its window on the bytes it was lent closes here
        for (const auto & ap : mc->alias) {
            if (ap.want_b > 0) { led_line(led_card(ap), "graph-done", 0, ap.want_b, tr_fmt("graph %d tokens", (int) mc->graph_tokens_last)); }
        }
    }
    struct flush_at_exit { ~flush_at_exit() { flush_tables(); } } flush_guard; // table changes reach the GPU before the next graph
    {   // stats.log: decode speed and hit rate per 256 steps. Prompt calls count no hits and take long, so a window with one in it says little: read it next to the call lines
        static const bool on = llama_moe_log_profile_fork("planner");
        static int64_t t0 = 0, n0 = 0;
        static uint64_t h0 = 0, m0 = 0;
        if (on && (t0 == 0 || mc->n_steps - n0 >= 256)) {
            uint64_t h = 0, m = 0;
            for (const auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
            const int64_t t = ggml_time_us();
            if (t0 > 0) {
                llama_moe_stats_log_fork("DEW", format("%lld steps: %.1f t/s, %.1f ms per token, hit %.1f%%; %s", (long long) (mc->n_steps - n0), (double) (mc->n_steps - n0) * 1e6 / (double) std::max<int64_t>(1, t - t0),
                        (double) (t - t0) / 1e3 / (double) std::max<int64_t>(1, mc->n_steps - n0),
                        h + m > h0 + m0 ? 100.0 * (double) (h - h0) / (double) (h + m - h0 - m0) : 0.0, llama_moe_stats_io_fork(2).c_str()).c_str());
            }
            t0 = t; n0 = mc->n_steps; h0 = h; m0 = m;
        }
    }
    if (mc->starved.load()) {
        return;   // kv-tail: the KV holds every expert block, nothing to score, admit or upload
    }
    if (g_rp.active && g_rp.t_end > 0) {   // replay: the token's compute (from the last step's end to now, before this step waits for uploads)
        const double cu = (double) (ggml_time_us() - g_rp.t_end);
        g_rp.cu_ema = g_rp.cu_ema > 0 ? 0.8 * g_rp.cu_ema + 0.2 * cu : cu;
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
    if (g_pp_built && g_prompt_call.load(std::memory_order_relaxed)) {
        // behind a pass of the prompt chain: it uploaded what it missed, there is no swap to plan and no decode step to publish for. What the prompt borrowed still comes back here (a sync
        // point): a prompt that ends on the chain may have no decode step behind it, and its blocks stayed empty into the next prompt.
        std::lock_guard<std::mutex> lock(mc->mtx);
        const size_t n_back = alias_release(mc);
        if (mc->kv_gave_back.exchange(false)) { reheat_free_blocks(mc, det, true); }
        if (n_back > 0) { reheat_free_blocks(mc, det); }
        return;
    }
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
            // --moe jit-all: the background refills too (one still in flight during a token is computed by the CPU: no deterministic replay)
            static const bool jit_all_wait = [] { const char * v = moe_opt("jit-all"); return v && atoi(v) > 0; }();
            mc->dcv.wait(wlk, [mc]() {
                return (jit_all_wait ? mc->in_flight == 0 : mc->in_flight == mc->in_flight_stream) &&
                    std::none_of(mc->todo.begin(), mc->todo.end(), [](const upload_job & j) { return !j.stream && (!j.bg || jit_all_wait); });
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
        // --moe jit-all=1: no background swaps; every expert a token needs is uploaded just in time and computed by the GPU (an expert whose
        // swap is still in flight during the token would be computed by the CPU, whose floats differ: no deterministic replay)
        static const bool jit_all_step = [] { const char * v = moe_opt("jit-all"); return v && atoi(v) > 0; }();
        if (jit_all_step) { budget_total = 0; }
        mc->last_budget = budget_total;
    }

    // a swap must pay for itself: over the count's horizon the candidate saves
    // (count_c - count_v) CPU expert evals of t_cpu each, and costs one upload
    const double cpu_gbs = cpu_gbs_now(mc);   // the live rate: the pay-back margin follows the CPU as it is now
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
        for (int k = 0; k < mc->n_links; ++k) { link_margin[k] = (int) margin; }
    }
    mc->last_margin = (int) margin;
    for (int k = 0; k < mc->n_links; ++k) { mc->link_margin[k] = link_margin[k]; }   // every link (a third card's layers read mc->link_margin[2] in the JIT admission)

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;
    if (mc->rp_on) {   // the graph that ran is over (synchronised): the pool may evict what the one before it asked for
        mc->vis_now.store(mc->vis_base + 1 + (uint32_t) mc->n_steps, std::memory_order_relaxed);
        ggml_rampool_graph_fork();
    }
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
    if (g_topic.after_prompt && decode_graph && !g_rp.active) {
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
    if (g_rp.active) {   // replay: the trace says what comes; the reactive swaps below would only fight it
        replay_next_use();
        double bud[MAX_LINKS];   // bytes each link may move during the next token: a share of the token's compute
        const double cu = g_rp.cu_ema > 0 ? g_rp.cu_ema : 20000.0;
        for (int k = 0; k < MAX_LINKS; ++k) { bud[k] = g_rp.frac * cu * (mc->gbs_link[k] > 0 ? mc->gbs_link[k] : 1.0) * 1e3; }
        replay_prefetch(mc, bud);
        for (auto & ls : mc->layers) { ls.pending.clear(); }
        g_rp.t_end = ggml_time_us();
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
            static const bool own_layer = [] { const char * e = moe_opt("own-layer"); return e && atoi(e) != 0; }();   // --moe own-layer=1: a layer evicts only its own experts between tokens too
            const slot_pick pk = pick_slot(mc, li, [&](layer_state & lo, size_t o, int32_t e) { return (own_layer && o != li) || pinned(mc, lo, e) ? -1.0 : xscore(mc, lo, e); }, best);
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
static std::atomic<int64_t> g_mb_probe { -1 };     // >= 0 while the start-up probe forces the cache path for up to this many tokens
static bool mb_is_auto() {
    static const bool a = [] { const char * e = moe_opt("max-batch"); return !e || strcmp(e, "auto") == 0; }();
    return a;
}
static int64_t mb_fixed() {
    static const int64_t v = [] { const char * e = moe_opt("max-batch"); return (e && strcmp(e, "auto") != 0) ? (int64_t) atoi(e) : (int64_t) -1; }();
    return v;
}

// The graph budget: outside a prompt call the limit above; a prompt call may take the decode chain as far as the CPU can beat the links: experts / picks x CPU rate / slowest link's rate,
// past that one token more costs the CPU more than uploading every expert it touches. Taken once, from the start-up probe: the node budget cannot follow later rates.
int64_t llama_moe_cache_max_batch_cap_fork() {
    static int64_t dec = -1;
    if (dec < 0 && g_cache && pp_chain_on()) {
        double r = 0.0;
        for (int k = 0; k < g_cache->n_links; ++k) {
            if (g_cache->gbs_link[k] > 0.0 && (r <= 0.0 || g_cache->gbs_link[k] < r)) { r = g_cache->gbs_link[k]; }
        }
        dec = r > 0.0 && knobs().cpu_gbs > 0.0 ? g_mb_auto.load(std::memory_order_relaxed) * (int64_t) std::ceil(knobs().cpu_gbs / r) : 0;
    }
    return std::max({ g_mb_auto.load(std::memory_order_relaxed), mb_fixed(), dec });
}

// Exact mode (the CPU backend's GPU-shaped math on, GPU_SHAPE knob / GGML_CPU_GPU_SHAPE): the CPU copies the GPU's one-token kernel (mul_mat_vec_q, 4 warps
// over K). For 2..8 tokens the GPU computes a cached expert with another kernel (mul_mat_vec_q_moe, one warp per token) and with MMQ above that, so a
// multi-token pass that mixes cached (GPU) and missed (CPU) experts gives bits that depend on what is cached: the 4-token tail of a server prompt moved
// token 0's logprob with the cache size (Qwen3.6, 3000 against 4000 MiB). Here only one-token passes take the cache path; larger ones take the prompt
// path, all experts on the GPU, the same bits at any cache size (400 of 400 tokens equal at 89.5% and 96.8% hits).
static bool cpu_gpu_shape_on() {
    static int (*get)(void) = nullptr;   // looked up once the CPU backend is loaded
    if (!get) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        get = reg ? (int (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_gpu_shape") : nullptr;
    }
    return get && get() != 0;
}

extern "C" { GGML_API void ggml_backend_set_link_rate_fork(ggml_backend_dev_t dev, double gbs); }   // ggml-backend.cpp (declared there, no shared header)
void llama_moe_cache_prompt_call_fork(bool on) {
    g_prompt_call.store(on, std::memory_order_relaxed);
    g_pp_dec = false;
    if (on) { g_pp_short = false; } else { ggml_backend_pred_set_fork(0.0); }
    if (on && g_cache) {
        for (int k = 0; k < g_cache->n_links; ++k) {
            if (g_cache->link_dev[k]) { ggml_backend_set_link_rate_fork(g_cache->link_dev[k], g_cache->gbs_link[k]); }
        }
    }
}

int64_t llama_moe_cache_max_batch_fork() {
    const int64_t p = g_mb_probe.load(std::memory_order_relaxed);
    if (p >= 0) {
        return p;
    }
    if (cpu_gpu_shape_on()) {
        return 1;
    }
    if (g_prompt_call.load(std::memory_order_relaxed) && pp_chain_on()) {   // a prompt call: the prompt chain, as far as it pays; none once the lend left it too few blocks
        return g_pp_short.load(std::memory_order_relaxed) ? 0 : mb_fixed() >= 0 ? mb_fixed() : g_pp_mb.load(std::memory_order_relaxed);
    }
    if (mb_fixed() >= 0) {
        // with the prompt chain a fixed limit is the prompts' limit: other passes (a reserve, a pass with many outputs) keep the decode chain's own, or a test's forced chain would
        // reserve the CPU chain's buffers for a whole micro-batch
        return pp_chain_on() ? std::min(mb_fixed(), g_mb_auto.load(std::memory_order_relaxed)) : mb_fixed();
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
        return g_mb_auto.load(std::memory_order_relaxed);
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
    if (!mc || !mb_is_auto() || pp_chain_on()) {   // the prompt chain decides per prompt call (llama_moe_cache_pp_decide_fork); the probe is for --moe pp-chain=0
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
        return;   // probe failed: the limit stays
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
// What a stream pass reads from the SSD (user 2026-11-03: "count the stream's compute and SSD read in its calculated time, using the measured rates"). DeepSeek (232 GiB on 125 GiB of RAM) streamed
// 30 to 49 GB through the mapping in a 277-token call, 13.6 to 17.5 s at 2.1 to 2.9 GB/s, where the upload over the link alone said 4.5 s. g_pp_ssd_m: the bytes of the experts this call's stream
// takes from tensors that are still mapped (set where the call is decided, used where it ends); g_pp_ssd_gbs: the mapped bytes per second the stream passes got through (read_bytes over 1 GiB says the SSD was read at all).
static double              g_pp_ssd_m   = 0.0;
static std::atomic<double> g_pp_ssd_gbs{ 0.0 };   // atomic: the dense split's upload estimate reads it under its own lock, not g_pp_mu
// The SSD's rate for a path that reads mapped experts: what a stream pass measured, else the loader's, else the machine benchmark's drive maximum (2026-11-07: GLM 3.5 planned every
// call "from the SSD at 0.00 GB/s", both were 0 with the O_DIRECT loader). One rate for every path's price (2026-11-10: the mix and the dense split's upload had no SSD floor, the
// stream and the decode chain had: DeepSeek 117 tokens, the mix priced 4.9 s and the split 4.75 s against 20.9 s for the stream, the call took 19.2 s).
static double pp_ssd_rate_gbs() {
    const double m = g_pp_ssd_gbs.load(), l = g_load_read_gbs.load();
    return m > 0.0 ? m : l > 0.0 ? l : llama_moe_ssd_max_get_fork();
}
static unsigned long long  g_pp_rb0     = 0;
// The decode chain's calculated time against what it took (user 2026-11-03: "it must be measured live as perf fluctuates and the cpu path can become slower ... when we remove mmap this algo will autoadapt"):
// g_pp_de0 the calculation of the call being decided, g_pp_dec_ratio the moving average of took / calculated over the decode-chain calls so far (0 = none yet, read as 1). A faster memory path under the
// CPU makes the ratio fall and the decode chain reach further, a slower one the other way.
static double              g_pp_de0       = 0.0;
static double              g_pp_dec_ratio = 0.0;
// 2026-11-06: the ratio above learned from every decode-chain call, whatever its size, and applied to every size: one 14.2k-token call that took 57x its calculation
// (the calculation had no compute) made the next 4-token call read "decode chain to 0 tokens, 4424 ms calculated" and leave the decode chain where it is the fastest.
// The time is now three parts, each learned from the calls it dominates (llama_moe_cache_pp_decide_fork): the misses x g_pp_dec_ratio (calls where the misses
// dominate), the 4-token chunks' weight reads x g_pp_dec_kc (calls where the chunks dominate), and a fixed part per pass g_pp_dec_f (the least any call left over).
static double              g_pp_dec_kc    = 1.0;    // measured over calculated chunk reads (0.05 to 64)
static double              g_pp_dec_f     = -1.0;   // the decode chain's fixed part per pass (us), -1 = no call measured yet
static double              g_pp_dec_chunk = 0.0;    // the call being decided: its chunks' calculated weight reads (us)
static unsigned long long pp_self_read_bytes() {
#ifdef __linux__
    unsigned long long v = 0;
    if (FILE * f = fopen("/proc/self/io", "r")) {
        char line[128];
        while (fgets(line, sizeof(line), f)) { if (sscanf(line, "read_bytes: %llu", &v) == 1) { break; } }
        fclose(f);
    }
    return v;
#else
    return 0;
#endif
}
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
// The compute of a pass, by its size: the last measurement per power of two of the tokens (in memory, never stored; user 2026-10-30: the last measured times, log-spaced, no formula. A
// pass has a large fixed part: 55 ms + 0.28 ms per token on Qwen3.6, so a 4-token tail says nothing about a 1000-token pass). A pass of the prompt chain gives the true value, its time
// less its uploads, which come before the compute; a streamed pass only an upper bound, since its compute hides behind its uploads: it fills a size no chain pass measured.
static struct { double n = 0.0, us = 0.0; bool chain = false; } g_pp_pass[32];
static void pp_pass_add(uint32_t n, double us, bool chain) {
    int b = 0;
    while ((n >> (b + 1)) != 0) { ++b; }
    auto & p = g_pp_pass[b];
    if (chain || !p.chain) { p.n = (double) n; p.us = us; p.chain = chain; }
}
// through the measured points; below the first the pass stays as it is (its fixed part), above the last it grows with the tokens. 0 with no measurement yet
static double pp_pass_us(double n) {
    double n0 = 0.0, u0 = 0.0;
    for (const auto & p : g_pp_pass) {
        if (p.n <= 0.0) { continue; }
        if (n <= p.n) { return n0 > 0.0 ? u0 + (p.us - u0) * (n - n0) / (p.n - n0) : p.us; }
        n0 = p.n; u0 = p.us;
    }
    return n0 > 0.0 ? u0 * n / n0 : 0.0;
}
// The largest chain pass measured at or below n tokens: its true compute (us), 0 = none. A size the chain never ran costs at least this (pp_chain_pays).
static double pp_pass_floor(double n) {
    double u = 0.0;
    for (const auto & p : g_pp_pass) { if (p.chain && p.n <= n) { u = p.us; } }
    return u;
}
// The time of a whole prompt pass, per path and size: the last measurement per power of two of the tokens (in memory, never stored: what the cards hold changes it, and a try costs one
// pass). This is what auto decides by (user 2026-11-01: per size, from measured values). The paths are told apart by the graph that ran, a forced path measures as well.
// PP_RING: the stream, every expert staged on the fastest link's card. PP_MIX: the stream with the prompt mix. PP_DEC: the decode chain. PP_K + i: the stream with predicted staging, q2 up to 2^i x a token's picks.
enum { PP_RING = 0, PP_MIX = 1, PP_CHAIN = 2, PP_DENSE = 3, PP_DEC = 4, PP_K = 5, PP_PATHS = PP_K + 8 };
static constexpr double PP_MIX_COLD = 4096.0;   // below it the prompt mix competes on its mean from the first request, above it only a trial (shortcut: a fixed size, the data has wins to 4.5k and losses at 12k; compute it from the cross-card traffic when 5-10k is measured)
static constexpr double PP_MIN_BATCH = 32.0;   // GGML_OP_OFFLOAD_MIN_BATCH's default: under it a pass's experts run on the CPU, whichever path was planned
static constexpr double PP_K_LATE = 1.3;   // predicted staging's real cost over the upload of the experts it calculates; the forced sweep's break-even at 77% of a layer's experts needed (2026-10-31, campaign predictor-upload-bw-reducer)
static struct { double n = 0.0, us = 0.0; } g_pp_took[PP_PATHS][32];
static bool g_pp_no_chain[32] = {};   // per size: the chain's graph refused such a pass
// ---- the learned planner (fork/docs/rampool.md "Robust planner design"; --moe split-model=0 keeps the per-size rules above for one release) ----
// One RLS per path over phi = (1, n, n x cells, the path's calculated upload) predicts a pass of any size: nothing keyed to a size (fault B), the path's own
// calculation is a feature whose coefficient is learned (the decode chain's "11.3 s calculated" took 19-25 s: its upload coefficient learns ~2, fault A/D).
// The pick is a Thompson draw per path from its posterior, kept for the request: exploration shrinks with confidence and regrows on drift by itself.
static dsf_rls  g_pp_rls[PP_PATHS];
static bool     g_pp_rls_loaded[PP_PATHS] = {};   // the state file was consulted for this path (once the split planner's section is known)
static bool     g_pp_rls_own[PP_PATHS] = {};      // and held a posterior for it
static double   g_pp_ctok_prior = 0.0;            // compute per token before any stream pass measured it: a token's expert bytes / the card's read rate (us)
static double   g_pp_phi[PP_PATHS][4] = {};       // the call's features per candidate path (phi[1] == 0: not a candidate), for the report to update the path that ran
static double   g_pp_pmu[PP_PATHS] = {}, g_pp_var[PP_PATHS] = {}, g_pp_drw[PP_PATHS] = {};   // the call's predicted mean, variance and draw per candidate
static int      g_pp_pick = -1;                   // the path the draws picked for this call (-1: not decided by the draws)
static uint64_t g_pp_seed = 0;                    // the draws of the running request (logged only since planner-simple)
static std::map<uint32_t, int> g_pp_size_seen;    // calls per pass size in this process: how often a pick recurs (the trial's R)
static bool     g_pp_trial = false;               // the call's pick is a trial (value of information), not the lowest mean
static int64_t  g_pp_seen_c0 = -1;                // the first cell of the last call: a lower one is a new request
static bool     g_pp_redraw = true;               // a pass surprised its model (NIS > 1): draw again at the next call
static uint32_t g_pp_chain_refused = UINT32_MAX;  // the chain's graph refused a pass of this size or more (a fact per pass size, not a bucket)
static bool pp_split_model_on() { static const bool on = [] { const char * o = moe_opt("split-model"); return !o || atoi(o) != 0; }(); return on; }
bool llama_dsplit_state_get_fork(const char * what, std::string & v);         // llama-dsplit-fork.cpp: this model/cards/build's state value, false: not known yet
void llama_dsplit_state_set_fork(const char * what, const std::string & v);   // and store one
void llama_dsplit_plan_book_fork(double regret_us, bool hit);                // /moe/stats plan_picks, plan_hits, plan_regret_s (both planners book there)
double llama_dsplit_split_mu_fork(int64_t pos0);                             // the split's predicted mean for the pass at pos0 (us), -1: not weighed
// the planner's invariants (fork/docs/planner-rules.md section 3): logged once per class to stats.log as INV, never an abort
static void pp_inv(const char * name, const std::string & text) {
    static std::set<std::string> seen;
    if (!seen.insert(name).second) { return; }   // under g_pp_mu
    const std::string t = std::string(name) + ": " + text;
    LLAMA_LOG_WARN("moe-cache: INV %s\n", t.c_str());
    llama_moe_stats_log_fork("INV", t.c_str());
}
// a path's estimator, initialised at its first use: th0 = (0, compute per token measured, attention per token and cell measured, upload coefficient 1)
// (both 0 until a stream pass measured them: a zero prior gets its uncertainty from the first call, rls_prime); a stored posterior of this build replaces it
static std::string pp_path_name(int p);
static double g_pp_c_chain = 0.0;                 // compute per token bounded by the prompt chain before a stream pass measured it: a chain pass's time past its upload, the least seen (us)
static double g_pp_rls_c[PP_PATHS] = {};          // the compute per token an arm's prior was built from
// compute per token, the best known: the stream's measured, else the chain's bound, else the card's weight read rate over a token's expert bytes
static double pp_c_guess() { return g_pp_c_us > 0.0 ? g_pp_c_us : g_pp_c_chain > 0.0 ? g_pp_c_chain : g_pp_ctok_prior; }
static dsf_rls & pp_rls(int p) {
    dsf_rls & r = g_pp_rls[p];
    const double c = pp_c_guess();
    // an arm that never ran is built again when the compute known moves: built once from the load-time weight read rate, the stream's prior said 2728 ms
    // for a 512-token OLMoE pass that takes 138, the chain (measured on the first call) always had the lower mean and the stream never ran (touch18 tiny rows)
    if (r.k == 0 || (r.n == 0 && !g_pp_rls_own[p] && c != g_pp_rls_c[p])) {
        // Every prior coefficient from a measured rate, never 0: a zero prior got its uncertainty from the first call that showed its feature, and a 2-token
        // call at 1 cell made the per-token and per-cell coefficients 10 s wide (rampool9 first run: +- 21000 s on a 2k call). Compute per token: pp_c_guess;
        // attention: the feature is the measured attention time (n x a x cells, the moe-cache's fit), its coefficient 1 +- 1 like the upload's.
        const double th0[4] = { 0.0, c, 1.0, 1.0 }, fl[4] = { 64.0 * c, 0.0, 0.0, 0.0 };   // fixed part: one 64-token pass (the least managed pass)
        rls_init(r, 4, th0, fl);
        g_pp_rls_c[p] = c;
    }
    if (!g_pp_rls_loaded[p] && r.n == 0) {
        std::string v;
        if (llama_dsplit_state_get_fork(("pp.rls." + std::to_string(p)).c_str(), v)) {
            g_pp_rls_loaded[p] = true;
            dsf_rls s;
            if (rls_parse(s, 4, v)) {
                r = s; g_pp_rls_own[p] = true;
                LLAMA_LOG_WARN("moe-cache: stored posterior of %s loaded (n=%d, this build and regime): %s\n", pp_path_name(p).c_str(), r.n, v.c_str());
            }
        }
    }
    // the mix and the predictions are the stream with something added: until one ran (or a stored posterior exists) the stream's posterior is their prior,
    // so a stream already known makes them a near tie to be tried, not nine wide unknowns
    if ((p == PP_MIX || p >= PP_K) && r.n == 0 && !g_pp_rls_own[p] && g_pp_rls[PP_RING].n > 0) {
        r = g_pp_rls[PP_RING];
        r.n = 0; r.nis = 1.0;
    }
    return r;
}
static std::string pp_path_name(int p) {
    return p == PP_RING ? "stream" : p == PP_MIX ? "stream, mix" : p == PP_CHAIN ? "prompt chain" : p == PP_DENSE ? "dense" : p == PP_DEC ? "decode chain" : format("stream, k x%d", 1 << (p - PP_K));
}
// What a try may risk (user 2026-11-02: "often smaller changes ... with big changes to react faster ... to not lose too much perf if decisions were bad"). A running pass cannot change its path, so a
// try that loses costs its whole call: DeepSeek's mix tried first on 16384 tokens, 134 s against 120 s. So a try runs when the call takes at most 1 s anyway, or at most 5% of this server's
// prompt time so far, less what tries lost already: a small call soon, a big one after enough work to pay for it. A loss is taken back at the next call of that size (the measured times decide).
// --moe pp-try=0: every size tries, as before.
static double g_pp_total_us = 0.0, g_pp_lost_us = 0.0;
static double g_pp_wait_us[32] = {}, g_pp_up_call = 0.0;   // per size: the least a prompt chain pass spent on top of its upload and compute, waiting for the router's ids (0 = no sample); the call's calculated upload
static bool pp_try_ok(double known_us) {
    static const char * const o = moe_opt("pp-try");
    return (o && atoi(o) == 0) || known_us <= std::max(1e6, 0.05 * g_pp_total_us - g_pp_lost_us);
}
static int pp_bucket(double n) {
    int b = 0;
    while (b < 31 && n >= (double) (2ull << b)) { ++b; }
    return b;
}
// the router wait of a pass of n tokens: this size's, else the nearest size measured; 0 = never measured (the router is then assumed free, so it runs and gets measured)
static double pp_wait_us(double n) {
    const int b = pp_bucket(n);
    for (int d = 0; d < 32; ++d) {
        if (b - d >= 0 && g_pp_wait_us[b - d] > 0.0) { return g_pp_wait_us[b - d]; }
        if (b + d < 32 && g_pp_wait_us[b + d] > 0.0) { return g_pp_wait_us[b + d]; }
    }
    return 0.0;
}
// the path's time at n tokens, through its measured points as pp_pass_us; 0 when it never ran near this size
static double pp_took_us(int path, double n) {
    if (g_pp_took[path][pp_bucket(n)].n <= 0.0) { return 0.0; }
    double n0 = 0.0, u0 = 0.0, n1 = 0.0, u1 = 0.0;
    for (const auto & p : g_pp_took[path]) {
        if (p.n <= 0.0) { continue; }
        if (n <= p.n) { return n0 > 0.0 ? u0 + (p.us - u0) * (n - n0) / (p.n - n0) : p.us; }
        n1 = n0; u1 = u0;
        n0 = p.n; u0 = p.us;
    }
    // above the last point: on through the last two, a pass has a fixed part (Qwen3.6: 281 ms at 606 tokens scaled from zero said 407 ms at 902, it took 340, and the mix lost a size it won)
    return n1 > 0.0 && u0 > u1 ? u0 + (u0 - u1) * (n - n0) / (n0 - n1) : u0 * n / n0;
}
// the fastest path but one (the chain, unless told) at n tokens, measured there; 0 when none ran there
static double pp_other_us(double n, int skip = PP_CHAIN) {
    double o = 0.0;
    for (int p = 0; p < PP_PATHS; ++p) {
        const double t = p == skip ? 0.0 : pp_took_us(p, n);
        if (t > 0.0 && (o <= 0.0 || t < o)) { o = t; }
    }
    return o;
}
// the path's time at n tokens: measured there, else the largest size measured below, scaled by the tokens; 0 when it never ran at or below this size
static double pp_scaled_us(int path, double n) {
    if (const double t = pp_took_us(path, n); t > 0.0) { return t; }
    for (int e = pp_bucket(n) - 1; e >= 0; --e) {
        const auto & p = g_pp_took[path][e];
        if (p.n > 0.0) { return p.us * n / p.n; }
    }
    return 0.0;
}
// The multiplier of predicted staging for a stream pass of n tokens, 0 = none (user 2026-11-01: k as multiples of the router's picks, found per model and size, DeepSeek took 5 x its 6 where
// Qwen3.8 took 1 x its 8). The stream runs first: it is what a prediction must beat. Then one try per step, 1 x, 2 x, 4 x the picks, while the step before won and the ranks do not reach all
// experts; of the steps measured, the fastest stays (a step can lose and the next one win: Qwen3.8 at 958 tokens, x1 lost, x4 won by 15%). Prediction shortens the upload alone, so a size is tried only while the upload is longer than what the compute takes at least.
// A new size starts on x1, no stream first (DeepSeek 152 tokens: 20 t/s on x1, 12 on the stream, a request lost to measure it); the stream comes last, as a try under pp_try_ok.
// The prompt mix for a stream pass of n tokens (user 2026-11-01: the second card should give a boost, and a partial upload slower than all of it is a bug). The plain stream runs first, it is
// what the last release runs; then one try. It saves the uploads of what the slower link's card holds and pays for moving each of its layers' tokens over that link, so its gain shrinks with
// the tokens: measured here, the faster; the nearest smaller size that lost says lost here (GLM, 4630 tokens: 9.9 s against 5.3 s, 26 s of host wait on the cross-card copies against 0.7).
// A pass whose picks cannot fill a layer stages nothing, so there is nothing to save.
static bool pp_mix_pick(double n, int n_used, int n_expert, const char ** why) {
    const double r = pp_took_us(PP_RING, n), m = pp_took_us(PP_MIX, n);
    if (r <= 0.0) { return false; }
    if (m > 0.0) { return m < r; }
    if (n * (double) n_used < (double) n_expert) { return false; }
    for (int e = pp_bucket(n) - 1; e >= 0; --e) {
        if (g_pp_took[PP_MIX][e].n <= 0.0 || g_pp_took[PP_RING][e].n <= 0.0) { continue; }
        if (g_pp_took[PP_MIX][e].us / g_pp_took[PP_MIX][e].n >= g_pp_took[PP_RING][e].us / g_pp_took[PP_RING][e].n) { return false; }
        break;
    }
    if (!pp_try_ok(r)) { *why = "no try of the mix: this call could lose too much"; return false; }
    *why = "one try of the mix";
    return true;
}
static double pp_k_pick(double n, double r, double U, int n_used, int n_expert, const char ** why) {
    double best = 0.0, t_best = r > 0.0 ? r : std::numeric_limits<double>::max();
    bool last_won = true;
    for (int i = 0; PP_K + i < PP_PATHS && (double) n_used * (double) (1 << i) < (double) n_expert; ++i) {
        const double t = pp_took_us(PP_K + i, n);
        if (t <= 0.0) {
            if (last_won && U > pp_pass_floor(n) && (n * (double) n_used < (double) n_expert || pp_try_ok(best > 0.0 ? t_best : r))) { *why = "one try of a prediction"; return (double) (1 << i); }
            break;
        }
        last_won = t < t_best;
        if (last_won) { best = (double) (1 << i); t_best = t; }
    }
    if (r <= 0.0 && best > 0.0 && pp_try_ok(t_best)) { *why = "one try of the stream"; return 0.0; }
    return best;
}
// Whether a pass of n tokens takes the prompt chain. Measured at this size on both sides: the faster. Else the nearest size measured on both sides, when it says so for this one too: the
// chain pays for its uploads per expert missed, so its loss only grows with the tokens: lost below means lost here, won above means won here. Else one try, while the chain can still win:
// up is what its misses cost by the picks so far (the real chain pays more on a miss-heavy pass, less once its misses stayed on the card); its compute costs at least pp_pass_floor, and at least
// what a stream of this size took beyond its whole upload, which it can at most have spent in sequence. The other side is what this size took, or the upload streaming needs (no staging can
// beat that). A try that loses costs one pass and settles the size and everything above it (GLM, 4684 tokens: 17.4 s against 8 s, before the stream's time bounded the compute). No miss at
// all: the chain, it uploads nothing and needs no staging.
static bool pp_chain_pays(double n, double up, double U, bool stream, const char ** why) {
    const int b = pp_bucket(n);
    if (g_pp_no_chain[b]) { *why = "its graph refused this size"; return false; }
    const double c = pp_took_us(PP_CHAIN, n), o = pp_other_us(n);
    if (c > 0.0 && o > 0.0) { *why = "measured at this size"; return c < o; }
    for (int d = 1; d < 32; ++d) {
        for (const int e : { b - d, b + d }) {
            if (e < 0 || e > 31 || g_pp_took[PP_CHAIN][e].n <= 0.0) { continue; }
            const double m = g_pp_took[PP_CHAIN][e].n, co = pp_other_us(m);
            if (co <= 0.0) { continue; }
            const bool won = g_pp_took[PP_CHAIN][e].us < co;
            if (won == (e > b)) { *why = won ? "won at a larger size" : "lost at a smaller size"; return won; }
        }
    }
    if (up <= 0.0) { *why = "nothing to upload"; return true; }
    // measured here, nothing to weigh it against: the other side once: the stream, or the decode chain where that is expected to win (4 tokens, Qwen3.8: 200 ms against 45)
    if (c > 0.0) { *why = stream ? "the stream once, to weigh the chain against" : "the decode chain once, to weigh the chain against"; return !stream; }
    *why = "one try";
    double est = std::max(pp_pass_floor(n), o - U) + up;
    // a chain pass measured at a smaller size says more than the picks: its misses are the ones that did not stay on the card (Qwen3.6: 138 ms at 62 tokens, up alone said 200 ms at 82)
    if (const double sc = pp_scaled_us(PP_CHAIN, n); sc > 0.0) { est = std::min(est, sc); }
    if (stream && !pp_try_ok(std::max(o, U))) { *why = "no try of the chain: this call could lose too much"; return false; }
    return est < (o > 0.0 ? o : U);
}

// one planned call (under g_pp_mu): added to what the next report measures
static void pp_add(uint32_t n, uint32_t ub, double U, int k) {
    const uint32_t P = (n + std::max<uint32_t>(1, ub) - 1) / std::max<uint32_t>(1, ub);
    g_pp_n += n; g_pp_P += P; g_pp_ub = std::min(ub, n); g_pp_U_us = U; g_pp_k = k;
    g_pp_Usum += (double) P * U;
    g_pp_xc   += (double) n * std::max(0.0, (double) g_pp_cells - 0.5 * (double) n);
}

// Expert e of this layer, as far as its tensors are neither pinned nor in the RAM pool (the SSD delivers them).
static double pp_mapped_expert_bytes(const layer_state & l, ggml_backend_buffer_type_t pinned_buft, int64_t e) {
    double Sm = 0.0;
    for (int k = 0; k < 3; ++k) {
        // the RAM pool holds this slice: DeepSeek, 101 of 123 GiB in the pool, all 123 priced as SSD reads put every path of a 117-token call at 21 s and hid the
        // decode chain (2.2 s forced) behind the stream (20 s taken), 2026-11-10
        if (l.rp_tab[k] && l.rp_tab[k][e]) { continue; }
        const ggml_tensor * t = k == 0 ? l.pub.up_src : k == 1 ? l.pub.gate_src : l.pub.down_src;
        // a tensor in the card's pinned host buffer is resident (read straight into place, locked): it never comes off the SSD. Counting it as mapped priced
        // GLM 3.0's 100.5 GiB of locked weights as SSD reads at 6.3 GB/s (stream 17.5 s, decode chain 19.7 s for a 100-token call) and only the mix escaped it
        if (l.src_fd[k] < 0 || !t->buffer || (pinned_buft && ggml_backend_buffer_get_type(t->buffer) == pinned_buft)) { continue; }
        const char * b = (const char *) ggml_backend_buffer_get_base(t->buffer);
        if ((const char *) t->data >= b && (const char *) t->data < b + ggml_backend_buffer_get_size(t->buffer)) { Sm += (double) t->nb[2]; }
    }
    return Sm;
}

// The experts a prompt pass streams to `dev`: every layer's expert bytes (E), and how many of those bytes this card's pool holds now (R), ready, not in flight;
// Mm (if asked): of the bytes it does not hold, the ones the SSD still has to deliver.
static void pp_expert_bytes(moe_cache * mc, ggml_backend_dev_t dev, double & E, double & R, double * Mm = nullptr) {
    E = 0.0; R = 0.0;
    if (Mm) { *Mm = 0.0; }
    std::lock_guard<std::mutex> lock(mc->mtx);
    ggml_backend_buffer_type_t pinned_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
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
        if (Mm) {
            const bool mine = c && c->buffer && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) == dev;
            for (int64_t e = 0; e < n_exp; ++e) {
                const int32_t sl = mine && (size_t) e < l.expert_slot.size() ? l.expert_slot[e] : -1;
                if (sl >= 0 && !((size_t) sl < l.slot_in_flight.size() && l.slot_in_flight[sl])) { continue; }
                *Mm += pp_mapped_expert_bytes(l, pinned_buft, e);
            }
        }
    }
}

// The upload of a prompt pass (ms): the expert bytes it touches - every layer's experts x the touched fraction 1 - (1 - k/E)^n of a pass of n tokens with k
// of E experts per token - less what the expert card's pool holds, over that card's measured link rate. A pass takes at least this long (the uploads overlap
// the compute, they do not shorten); the dense split leaves it as it is (the experts stay on the expert card) and only shortens the compute, so a split pays
// only when a pass's compute is longer than this (user 2026-10-25: "is the time it takes to upload the same as computing 7k tokens?"). -1: not known.
double llama_moe_cache_pass_upload_ms_fork(uint32_t n_tokens, uint32_t n_used, uint32_t n_expert) {
    moe_cache * mc = g_cache;
    if (!mc || mc->n_links <= 0 || !mc->link_dev[0] || n_expert == 0 || n_tokens == 0) { return -1.0; }
    double E = 0.0, R = 0.0, Mm = 0.0;
    pp_expert_bytes(mc, mc->link_dev[0], E, R, &Mm);
    const double gbs = llama_moe_cache_link_gbs_fork(mc->link_dev[0]);
    if (gbs <= 0.0 || E <= 0.0) { return -1.0; }
    const double frac = 1.0 - std::pow(1.0 - std::min(1.0, (double) n_used / (double) n_expert), (double) n_tokens);
    // the link is not the only limit: the bytes still mapped come off the SSD first (2026-11-10 DeepSeek 117 tokens: the split planned 4.75 s over the link, the call took 19.2 s at 2.1 GB/s)
    const double ssd = pp_ssd_rate_gbs();
    return frac * std::max(std::max(0.0, E - R) / (gbs * 1e6), ssd > 0.0 ? Mm / (ssd * 1e6) : 0.0);
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
                (size_t) kvz_lend_potential(ap, kvz_read_cells(std::max<size_t>(n_prompt, ap.kv_h), pad)), kvz_pass_cap(ap, n_prompt, ub, a, b, pad, ap.mirror_b));
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
        static const bool own_layer_pp = [] { const char * e = moe_opt("own-layer"); return e && atoi(e) != 0; }();
        if (n == 0 || own_layer_pp) { continue; }   // (this pass moves blocks between layers)
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

// the prompt path a pass of this call runs on (the split's own pass is PP_DENSE)
static int pp_cur_path() {
    int path = llama_dsplit_cur_fork() == 1 ? PP_DENSE : g_pp_dec_built ? PP_DEC : g_pp_built ? PP_CHAIN : g_pp_mix_built ? PP_MIX : PP_RING;
    if ((path == PP_RING || path == PP_MIX) && g_pp_k_built) {
        path = PP_K;
        for (double m = g_pp_k_m; m > 1.0; m /= 2.0) { ++path; }
    }
    return path;
}
static int g_pp_pass_learned = 0;   // passes of this call that taught their path (the call's end then adds no second sample)
// Every prompt pass teaches its path at once (llama_moe_prompt_progress_fork), not only the call's end: the split planner re-plans every pass against the
// one-card arm, and a stale one-card mean held for a whole call (GLM 3.0 64k, 2026-11-08: a stored stream posterior of one sample said 4 s for a 10144-token
// pass, one-card passes took 15 s, the 56k call ran one card from the first pass to the last at 491-581 tokens/s while the split does 750-800).
void llama_moe_cache_pp_pass_learn_fork(uint32_t n, double us) {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    if (!g_cache || us <= 0.0 || n < 64) { return; }
    const int path = pp_cur_path();
    if (path == PP_DENSE || g_pp_phi[path][1] <= 0.0) { return; }
    const double planned = g_pp_phi[path][1];
    if ((double) n < 0.8 * planned || (double) n > 1.25 * planned) { return; }   // a pass of another size than the features were built for (the call's tail)
    dsf_rls & rp = pp_rls(path);
    const double nis = rls_update(rp, g_pp_phi[path], us);
    if (nis > 1.0) { g_pp_redraw = true; }
    if (rp.storable()) { llama_dsplit_state_set_fork(("pp.rls." + std::to_string(path)).c_str(), rls_str(rp)); }
    g_pp_pass_learned++;
    if (moe_log_has("w")) {
        double v = 0.0;
        const double m = rls_pred(rp, g_pp_phi[path], &v);
        LLAMA_LOG_WARN("moe-cache: pass of %u tokens on %s took %.2f s: its arm now %.2f +- %.2f s (n=%d, nis %.2f)\n", n, pp_path_name(path).c_str(), us / 1e6, m / 1e6, std::sqrt(std::max(0.0, v)) / 1e6, rp.n, nis);
    }
}
void llama_moe_cache_prefill_report_fork(uint32_t n_tokens, int64_t us) {
    moe_cache * mc = g_cache;
    if (mc && mc->pf_count_on) {
        // the prompt's expert use, counted on the cards: once per expert and prompt, as the host callback counted it (the cards are idle here: after synchronize)
        typedef bool (*obs_read_t)(int, int, uint8_t *);
        static obs_read_t rd = (obs_read_t) llama_fork_proc("ggml_backend_fork_obs_prompt_read_fork");
        std::vector<uint8_t> used((size_t) mc->pf_count_layers * mc->pf_count_experts, 0);
        std::vector<upload_job> swaps;
        const bool counted = rd && rd(mc->pf_count_layers, mc->pf_count_experts, used.data());
        if (moe_log_has("k")) {   // zero on a pass that ran: the card's counters saw none of its expert matmuls
            LLAMA_LOG_WARN("moe-cache: prompt counters: %zu of %zu layer experts counted\n", (size_t) std::count(used.begin(), used.end(), (uint8_t) 1), used.size());
        }
        if (counted) {
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
    const double chain_up = g_pp_up_us.exchange(0.0);
    // a call of P passes: one pass of its size. A call whose passes were not counted here (the chains, a fixed -ub): the passes of the size its decision was made for.
    // Without it such a call read as one pass (OLMoE 2761 tokens on the prompt chain: "predicted 197 ms per pass, took 1105"), so neither its time per size nor the router's wait was ever learned
    const int path = pp_cur_path();
    const double phi_n = path >= 0 && path < PP_PATHS ? g_pp_phi[path][1] : 0.0;
    const double P = n_tokens == g_pp_n && g_pp_P > 0 ? (double) g_pp_P : phi_n >= 1.0 && (double) n_tokens > phi_n ? std::ceil((double) n_tokens / phi_n) : 1.0;
    if (mc && us > 0 && n_tokens > 0 && llama_dsplit_cur_fork() != 1 && !g_pp_dec_built) {
        pp_pass_add((uint32_t) std::lround((double) n_tokens / P), (g_pp_built ? std::max(0.0, (double) us - chain_up) : (double) us) / P, g_pp_built);
    }
    if (mc && us > 0 && n_tokens > 1) {
        const int learned = g_pp_pass_learned;
        g_pp_pass_learned = 0;
        // the SSD rate of the stream: the bytes the process read from storage over this call (only a stream pass, which reads the mapped experts; the chains read few)
        if (g_pp_ssd_m > 0.0 && path != PP_DEC && path != PP_CHAIN && path != PP_DENSE && (double) n_tokens >= PP_MIN_BATCH) {
            const unsigned long long rb = pp_self_read_bytes();
            if (rb > g_pp_rb0 + (1ull << 30)) {
                const double g = g_pp_ssd_m / ((double) us * 1e3);   // mapped bytes through the pass per second: the page cache holds a part of them (DeepSeek 2026-11-03: 30 GB read of 110 GB mapped), so the bytes read alone would over-count
                g_pp_ssd_gbs = g_pp_ssd_gbs > 0.0 ? 0.7 * g_pp_ssd_gbs + 0.3 * g : g;
            }
        }
        g_pp_ssd_m = 0.0;
        // stats.log: what this call took against the best known at its size, read before its own measurement goes in. The first run of a path that loses counts against the budget of later tries (pp_try_ok)
        {
            const double n1 = (double) n_tokens / P, took = (double) us / P, was = pp_took_us(path, n1);
            // the learned planner: the path that ran learns from this pass with the features it was decided with (a split pass learns in the split planner)
            std::string pr;
            if (path != PP_DENSE && g_pp_phi[path][1] > 0.0) {
                dsf_rls & rp = pp_rls(path);
                const double mu = g_pp_pmu[path], sd = std::sqrt(std::max(0.0, g_pp_var[path]));
                // (a tiny call on the stream family ran on the CPU: its time says nothing about a stream. The 2-token call at start, priced against the uncalibrated link rate, left the upload coefficient about 17x low)
                // nor does a tiny decode-chain call teach its arm: the 2-token start call (cold kernels, 420 ms) put a 100-token GLM call at 13.9 s and the stream won it (2026-11-10)
                const bool streamed = n1 >= PP_MIN_BATCH || path == PP_CHAIN;
                const double nis = learned > 0 || !streamed ? rp.nis : rls_update(rp, g_pp_phi[path], took);   // the passes taught the arm already (llama_moe_cache_pp_pass_learn_fork)
                if (nis > 1.0) { g_pp_redraw = true; }   // the posterior moved more than it expected: the next call draws again
                // stored only when determined and quiet (feedback_stored_values_policy): a next start begins from it as its prior
                if (rp.storable()) { llama_dsplit_state_set_fork(("pp.rls." + std::to_string(path)).c_str(), rls_str(rp)); }
                double lo = -1.0;   // the lowest predicted mean among this call's candidates AND the split's (C17: a one-card loss to the split is regret too)
                for (int p = 0; p < PP_PATHS; ++p) { if (g_pp_phi[p][1] > 0.0 && (lo < 0.0 || g_pp_pmu[p] < lo)) { lo = g_pp_pmu[p]; } }
                if (const double ms = llama_dsplit_split_mu_fork(g_pp_seen_c0); ms > 0.0 && (lo < 0.0 || ms < lo)) { lo = ms; }
                // INV sure (I9): a path with two samples or fewer must not claim to know its time within ~20%
                if (rp.n <= 2 && rp.nis <= 0.1 && g_pp_var[path] < 0.05 * mu * mu) { pp_inv("sure", format("%s after %d samples: %.0f +- %.0f ms", pp_path_name(path).c_str(), rp.n, mu / 1e3, sd / 1e3)); }
                // INV phi (I6): the sample's pass size is the one the features were built for
                if (std::fabs(g_pp_phi[path][1] - (double) n_tokens / P) > 32.0) { pp_inv("phi", format("%s: features for %.0f tokens, the pass had %.0f", pp_path_name(path).c_str(), g_pp_phi[path][1], (double) n_tokens / P)); }
                const bool best = lo >= 0.0 && mu <= lo;
                const double regret = best ? 0.0 : std::max(0.0, took - lo);
                if (g_pp_pick == path) { llama_dsplit_plan_book_fork(regret * P, took <= lo + sd); }
                pr = format("; predicted %.0f +- %.0f ms per pass, took %.0f, nis %.2f, regret %.1f s (model n=%d, nis %.2f)", mu / 1e3, sd / 1e3, took / 1e3, nis, regret * P / 1e6, rp.n, rp.nis);
            }
            int b = -1;
            for (int p = 0; p < PP_PATHS; ++p) {
                const double t = pp_took_us(p, n1);
                if (t > 0.0 && (b < 0 || t < pp_took_us(b, n1))) { b = p; }
            }
            const double best = b >= 0 ? pp_took_us(b, n1) : 0.0;
            g_pp_total_us += (double) us;
            if (was <= 0.0 && best > 0.0 && took > best) { g_pp_lost_us += (took - best) * P; }
            // only when the sample's pass is the one its upload was calculated for (a fixed -ub cuts a call into passes the planner does not count); the call's
            // last pass is shorter, so the passes average below the decided size: the band of llama_moe_cache_pp_pass_learn_fork (it was 32 tokens, and
            // OLMoE's 2761 tokens in 6 passes of 512, 460 on average, never taught the wait)
            if (path == PP_CHAIN && n1 >= PP_MIN_BATCH && n1 >= 0.8 * phi_n && n1 <= 1.25 * phi_n) {
                const double rest = std::max(1.0, took - g_pp_up_call);   // the pass past its upload: the router's wait and the compute
                if (g_pp_c_us > 0.0) {
                    double & w = g_pp_wait_us[pp_bucket(n1)];
                    const double got = std::max(1.0, rest - g_pp_c_us * n1);
                    w = w > 0.0 ? std::min(w, got) : got;
                } else {   // no stream measured the compute yet: the rest bounds it (as a wait it would be counted twice once a stream measures the compute)
                    g_pp_c_chain = g_pp_c_chain > 0.0 ? std::min(g_pp_c_chain, rest / n1) : rest / n1;
                }
            }
            const uint64_t pk = g_ppc_picked.exchange(0), pa = g_ppc_all.exchange(0), pm = g_ppc_miss.exchange(0);
            const std::string picks = pa ? format(", routers picked %.1f%% of the experts, %.1f%% of those uploaded", 100.0 * (double) pk / (double) pa, pk ? 100.0 * (double) pm / (double) pk : 0.0) : std::string();
            // where the stream's weight uploads came from and what the scheduler thread waited for (ggml_backend_sched_upload_weight): a pass bound by its file reads shows here
            uint64_t ups[6] = { 0 };
            ggml_backend_upload_stats_fork(ups, true);
            const std::string upl = ups[0] + ups[1] + ups[2] == 0 ? std::string() : format("; weight uploads: RAM pool %.1f GB, host %.1f GB, file %.1f GB in %llu reads (%.2f GB/s while reading: %.0f ms of the scheduler thread, %.0f ms more waiting for a read chunk)",
                    ups[0] / 1e9, ups[1] / 1e9, ups[2] / 1e9, (unsigned long long) ups[3], ups[4] ? (double) ups[2] / ((double) ups[4] * 1e3) : 0.0, ups[4] / 1e3, ups[5] / 1e3);
            const std::string done = format("prompt call of %u tokens: %s took %.0f ms (%.0f t/s, %.2f ms per token%s)%s%s; %s", n_tokens, pp_path_name(path).c_str(), (double) us / 1e3, (double) n_tokens * 1e6 / (double) us, (double) us / 1e3 / std::max(1u, n_tokens), picks.c_str(),
                    (best > 0.0 ? format("; best known here %s %.0f ms, %.2f x%s", pp_path_name(b).c_str(), best / 1e3, took / best, took > 1.3 * best ? ", SLOW" : "") : std::string("; first of this size")).append(pr).c_str(),
                    upl.c_str(), llama_moe_stats_io_fork(0).c_str());
            llama_moe_stats_log_fork("DON", done.c_str());
            { const double t1 = llama_moe_span_now_fork(); llama_moe_span_fork(format("prompt call: %s, %u tokens", pp_path_name(path).c_str(), n_tokens).c_str(), t1 - (double) us / 1e6, t1, 0, ""); }
        }
        // the first pass of a server that fills a layer runs cold whatever its path (Qwen3.6: 348 ms on the chain, 130 warm; the stream 150 ms over its repeats) and the size would keep that
        // number until the path runs again: not a measurement
        // the chain was asked and its graph refused (a layer's blocks cannot hold the pass, pp_chain_fork): the size asks no more
        if (path != PP_CHAIN && path != PP_DENSE && path != PP_DEC && g_pp_mb.load(std::memory_order_relaxed) > 0) {
            g_pp_no_chain[pp_bucket((double) n_tokens / P)] = true;
            g_pp_chain_refused = std::min(g_pp_chain_refused, (uint32_t) ((double) n_tokens / P));
        }
        // the first call on the decode chain neither: 2 tokens took GLM 3065 ms next to the preload's uploads, 106 ms for 4 tokens later, and that call may be the only one of its size
        static bool cold = true, cold_dec = true;
        const bool fills = !mc->layers.empty() && (double) n_tokens / P * (double) mc->n_expert_used >= (double) mc->layers[0].expert_slot.size();
        if (cold && fills) {
            cold = false;
            g_warm.store(true);
        } else if (path == PP_DEC && cold_dec) {
            cold_dec = false;
        } else {
            auto & t = g_pp_took[path][pp_bucket((double) n_tokens / P)];
            t.n = (double) n_tokens / P; t.us = (double) us / P;
            if (path == PP_DEC && g_pp_de0 > 0.0) {   // the first call of the path is not counted above (it pays the preload's uploads): neither here
                // each part learns from the calls it dominates (see g_pp_dec_kc): a 15k-token call teaches the chunks, a 4-token call the misses
                const double took = (double) us / P, d0 = g_pp_de0, ch = g_pp_dec_chunk;
                const double r = g_pp_dec_ratio > 0.0 ? g_pp_dec_ratio : 1.0, f = std::max(0.0, g_pp_dec_f);
                if (ch > 0.0 && ch * g_pp_dec_kc >= d0 * r) {   // the chunks dominate: only they learn (a fixed part from such a call would lock small calls out)
                    // (no floor of 1: the card's weight read rate is a load-time matmul timing, 95 to 20762 GB/s in the logs of 2026-11-10; OLMoE 118 tokens: 702 ms calculated, the whole call took 132)
                    const double q = std::max(0.05, (took - f - d0 * r) / ch);
                    g_pp_dec_kc = std::min(64.0, 0.7 * g_pp_dec_kc + 0.3 * q);
                } else {   // the misses dominate: they learn, and the fixed part is the least such a call left over
                    const double q = std::max(0.25, std::min(8.0, (took - f - ch * g_pp_dec_kc) / d0));
                    g_pp_dec_ratio = g_pp_dec_ratio > 0.0 ? 0.7 * g_pp_dec_ratio + 0.3 * q : q;
                    const double rest = std::max(0.0, took - d0 * g_pp_dec_ratio - ch * g_pp_dec_kc);
                    g_pp_dec_f = g_pp_dec_f < 0.0 ? rest : std::min(g_pp_dec_f, rest);
                }
                // kept between starts: a server otherwise began every time at the calculated chunks (OLMoE 58 tokens: 418 ms calculated, 97 taken, on every start)
                llama_dsplit_state_set_fork("pp.dec", format("%.4f %.4f %.1f", g_pp_dec_ratio, g_pp_dec_kc, g_pp_dec_f));
            }
        }
    }
    // a chain pass neither: its uploads add to its compute, as one token's compute it told the split planner one card takes 1288 ms where it streams in 500
    if (!mc || us <= 0 || llama_dsplit_cur_fork() == 1 || g_pp_built || g_pp_k_built || g_pp_dec_built || n_tokens != g_pp_n || g_pp_P == 0 || g_pp_ub == 0) {
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

// Whether the prompt chain takes this call, from what is measured now (user 2026-10-31: the crossovers come from the machine and the model, no per-model constants; the start-up probe
// could not see them: it repeats synthetic tokens, so its cache path found every expert and looked faster than it is; user 2026-11-01: from measured values, per size). pp_chain_pays
// decides; what it needs from the cache is what the chain's uploads cost at this size:
//   prompt chain   C(n) + sum over the experts not on the card of P(the pass touches it) x its upload over its layer's link: uploads wait for the router, so they add to the compute
//   streaming      max(C(n), U): U is the upload of everything no card holds, over the fastest link; the staging slots hide it under the compute
// P = 1 - (1 - p)^n, with p the expert's share of the layer's picks so far: k (count + 1) / (all counts + experts), so a model nobody ran yet counts as uniform.
double llama_dsplit_read_rate_fork(ggml_backend_dev_t d);   // llama-dsplit-fork.cpp: the card's weight read rate from the matmul timing at load (bytes/s), 0 = not timed

void llama_moe_cache_pp_decide_fork(uint32_t n_tokens, uint32_t ub_max) {
    moe_cache * mc = g_cache;
    g_pp_path_call = PP_RING;   // until decided: the stream, nothing saved against it
    g_pp_save_us = 0.0;
    g_pp_last_n.store(n_tokens, std::memory_order_relaxed); g_pp_last_ub.store(ub_max, std::memory_order_relaxed);
    // a refused chain (g_pp_short) only takes the chain arm out below: the other arms are still decided (llama_moe_cache_pp_short_fork)
    if (!mc || !pp_chain_on() || !g_prompt_call.load(std::memory_order_relaxed) || g_pp_dec.load(std::memory_order_relaxed) || !mb_is_auto() || ub_max == 0 || mc->n_links <= 0) {
        return;
    }
    const double R_read = llama_dsplit_read_rate_fork(mc->link_dev[0]);   // before any lock: it takes the split planner's (which calls into the cache under it)
    double A = 0.0;   // the expert bytes one token reads, all layers: what the decode chain's chunks read again for every token
    pp_sums sums((size_t) (PP_PATHS - PP_K));   // llama-estimate-fork.h: every expert added below, the same sums a simulated machine fills
    std::vector<pp_miss> & v = sums.miss;      // the experts no card holds
    double & U = sums.U, & Um = sums.Um;       // the stream's upload, without and with the prompt mix
    double & M = sums.M, & Mm = sums.Mm;       // of those bytes, the ones only the SSD has
    double * const Uk = sums.Uk.data(), * const Mk = sums.Mk.data();   // the same two for predicted staging with 2^i x the picks
    const double n = (double) std::min(n_tokens, ub_max);
    int n_expert = 0;
    bool has_slower = false;
    bool cpu_known = false;   // the CPU's read rate is measured: without it the decode chain's time cannot be calculated, and no prompt call takes it
    bool chain_fits = true;   // every layer has the blocks a prompt chain pass of this size takes (the graph's own check, llama_moe_cache_pp_chain_fork)
    {
        std::lock_guard<std::mutex> lock(mc->mtx);
        const double r0 = mc->gbs_link[0] * 1e3;   // bytes / us
        const double rc = cpu_gbs_now(mc) * 1e3;   // as llama_moe_cache_jit_fork
        ggml_backend_buffer_type_t pinned_buft = mc->link_dev[0] ? ggml_backend_dev_host_buffer_type(mc->link_dev[0]) : nullptr;
        cpu_known = rc > 0.0;
        for (const auto & l : mc->layers) {
            if (!l.pub.up_src || !l.pub.gate_src || !l.pub.down_src) {
                continue;
            }
            n_expert = std::max(n_expert, (int) l.expert_slot.size());
            has_slower = has_slower || l.link > 0;
            // a chain the graph then refuses falls to the decode chain's CPU side: DeepSeek, 62-token call, chain calculated 2.1 s, ran 18.5 s that way (2026-11-10)
            if (chain_fits && (!mc->link_be[l.link] || pp_blocks(mc, const_cast<layer_state &>(l)) < std::min<int64_t>((int64_t) l.expert_slot.size(), (int64_t) std::ceil(n) * (int64_t) mc->n_expert_used))) { chain_fits = false; }
            const double S  = (double) (l.pub.up_src->nb[2] + l.pub.gate_src->nb[2] + l.pub.down_src->nb[2]);
            A += (double) mc->n_expert_used * S;
            const double rl = mc->gbs_link[l.link] * 1e3;
            // how often each expert is picked: the state's lifetime counts plus the RAM pool's heat file. A fresh state has no counts, every expert then looked equally likely
            // and a 117-token DeepSeek pass was priced as touching 97% of each layer, the SSD-only cold ones among them (decode chain 6.95 s planned, 2.1 s taken; 65 of 256
            // experts missed in layer 0), while the pool had been filled by that very heat (2026-11-10)
            auto picks_of = [&](size_t e) { return (double) l.glob_count[e] + (e < l.heat_cnt.size() ? (double) l.heat_cnt[e] : 0.0); };
            double all = (double) l.glob_count.size();
            for (size_t e = 0; e < l.glob_count.size(); ++e) { all += picks_of(e); }
            for (size_t e = 0; e < l.expert_slot.size(); ++e) {
                const bool on_card = l.expert_slot[e] >= 0;
                const double Sm = pp_mapped_expert_bytes(l, pinned_buft, (int64_t) e);   // what the SSD delivers of it: nothing where it is pinned or in the RAM pool
                // its share of one token's picks: k (count + 1) / (all counts + experts), so a model nobody ran yet counts as uniform
                pp_sums_add(sums, n, S, Sm, on_card, l.link == 0, (double) mc->n_expert_used * (picks_of(e) + 1.0) / all, r0, rl, rc);
            }
        }
    }
    std::lock_guard<std::mutex> lk(g_pp_mu);
    double up = 0.0, cpu = 0.0, miss_ssd_bytes = 0.0;   // what the call's misses cost uploaded (once per expert it touches) and on the CPU (once per token routed to them); their bytes only the SSD has
    pp_sums_misses(sums, n, up, cpu, miss_ssd_bytes);
    const char * why = "";
    // A chain pass drops the uploads that have not started (pp_settle). Right after a start those are the hot set on its way, the slower link last: the first 2-token call threw away most of what
    // the x4 card was to hold, its blocks stayed reserved and counted as full (Qwen3.8: 17% of the asked experts on the fast card, 31% without the chain, a fifth more bytes per pass).
    bool busy = false;
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        busy = !mc->todo.empty() || mc->in_flight > 0;
    }
    // --moe pp-path=stream|mix|chain (test rows): every prompt call takes this path, whatever was measured. The planner picks by time, so a path may run once in a long request or
    // never: the prompt mix's first try came on the last call of the second request, and a bug on it looked like a bug of the second request (2026-11-02). fork/runs/paths.csv forces each one.
    static const char * const force = moe_opt("pp-path");
    // Upload or CPU is decided behind each layer's router, for every pass, from the real picks and misses x the known costs (llama_moe_cache_jit_fork, user 2026-11-02: "number of miss moes
    // multiplied by known cost"). Here, before any router ran, only the graph is chosen: the decode chain has a CPU side, the prompt chain and the stream have none. The odds of a miss say too little
    // for that alone: the uploads and the CPU work on the same misses side by side, so the decode chain's time is up x cpu / (up + cpu) (2026-11-02, GLM, MiMo, DeepSeek, cold and warm, 100-token
    // prompts: 2438 / 2300, 1824 / 2000, 2318 / 2100, 1576 / 1600, 6727 / 7000, 2669 / 2700 ms calculated / taken; a cold DeepSeek with its CPU rate at 1.9 GB/s among them). The stream's time is its
    // upload U, the most it can take: what the decode chain must beat. Both follow the rates, so the size where they meet moves with the link and the RAM, and comes out as a number: U / time per token
    // (user 2026-11-02: "we just need to find calc max threshol token count to do misses cpu higher upload and gpu"). What was measured at this size counts first. Then the prompt chain if its uploads can win.
    const double td = pp_took_us(PP_DEC, n), to = pp_other_us(n, PP_DEC);
    // The stream reads the experts no card holds: over the link, and from the SSD where their tensors are still mapped. Both run side by side (the reads fill the staging slots the link empties), so the
    // slower one is its time; the SSD's rate is what the stream passes read at so far, none before the first one.
    // a stream pass's own rate, until one ran: the loader's, else the machine benchmark's drive maximum (2026-11-07: GLM 3.5 planned every call "from the SSD at 0.00 GB/s", both were 0 with the O_DIRECT loader)
    const double ssd_rate = pp_ssd_rate_gbs();
    const double Ts = ssd_rate > 0.0 ? std::max(U, M / (ssd_rate * 1e3)) : U;
    g_pp_ssd_m = M;
    g_pp_rb0 = pp_self_read_bytes();
    const double de0 = up + cpu > 0.0 ? up * cpu / (up + cpu) : 0.0;
    g_pp_de0 = de0;
    // 2026-11-06: both sides carry the pass's GPU work. The stream hides its upload under its compute (the staging slots): max(Ts, C). The decode chain adds its GPU
    // work to the misses, and that work runs in 4-token chunks (llama_context::graph_max_nodes: up/gate/activation/down per chunk) that read every token's experts
    // again: the token's expert bytes A at the card's weight read rate (llama_dsplit_read_rate_fork), x what such calls measured, never under the pass's compute.
    // Without it the calculation saw the misses only and sent 1k-16k-token calls to the decode chain at 4-57x their calculated time (Qwen3.6 14.2k: 266 ms
    // calculated, 15.2 s taken; Qwen3.8-Flash-Next 15.5k: 1360 ms, 35.0 s, streamed 10.1 s). Small calls keep it: next to their misses the chunks are small, so the
    // size where the two meet comes out per model (2026-11-06 logs: about 200 tokens on Qwen3.6, 128-512 on GLM-5.3, about 256 on MiMo, past 1k on DeepSeek).
    auto comp_at  = [&](double nn) { return nn * pp_c_at(); };   // the pass's compute on one card, measured (0 until a stream pass measured it)
    auto chunk_at = [&](double nn) { return R_read > 0.0 ? nn * A / R_read * 1e6 : 0.0; };
    auto de0_at   = [&](double nn) {
        double u = 0.0, c = 0.0;
        for (const auto & x : v) { u += (1.0 - std::exp(nn * x.l1p)) * x.us; c -= std::expm1(x.l1p) * nn * x.cu; }
        return u + c > 0.0 ? u * c / (u + c) : 0.0;
    };
    {   // the decode chain's learned parts of an earlier start (asked until the section answers or this start learned its own)
        static bool dec_loaded = false;
        std::string sv;
        double r0 = 0.0, k0 = 0.0, f0 = 0.0;
        if (!dec_loaded && g_pp_dec_ratio == 0.0 && g_pp_dec_kc == 1.0 && g_pp_dec_f < 0.0 && llama_dsplit_state_get_fork("pp.dec", sv)) {
            dec_loaded = true;
            if (sscanf(sv.c_str(), "%lf %lf %lf", &r0, &k0, &f0) == 3 && k0 >= 0.05 && k0 <= 64.0 && r0 >= 0.0 && r0 <= 8.0) {
                g_pp_dec_ratio = r0; g_pp_dec_kc = k0; g_pp_dec_f = f0;
                LLAMA_LOG_WARN("moe-cache: stored decode chain parts loaded: misses x%.2f, chunks x%.2f, fixed %.0f ms\n", r0 > 0.0 ? r0 : 1.0, k0, std::max(0.0, f0) / 1e3);
            }
        }
    }
    const double dec_r = g_pp_dec_ratio > 0.0 ? g_pp_dec_ratio : 1.0, dec_f = std::max(0.0, g_pp_dec_f);
    auto de_at = [&](double nn, double d0) { return dec_f + d0 * dec_r + std::max(chunk_at(nn) * g_pp_dec_kc, comp_at(nn)); };
    auto ot_at = [&](double nn) { return std::max(Ts, comp_at(nn)); };
    g_pp_dec_chunk = chunk_at(n);
    const double de = td > 0.0 ? td : de_at(n, de0), ot = to > 0.0 ? to : ot_at(n);   // the decode chain against the fastest other path, measured or calculated
    // where the two meet, for the line: the largest pass the decode chain is calculated to win, by halving between 1 and the pass limit
    // no cap beside it (user 2026-11-03, DeepSeek: a node budget of 384 refused the chain at 409 tokens where it was calculated 2.7 times faster): the crossover decides, up to the pass limit
    double dec_max = 0.0;
    {
        double lo = 1.0, hi = (double) ub_max;
        if (de_at(hi, de0_at(hi)) < ot_at(hi)) {
            dec_max = hi;
        } else if (de_at(lo, de0_at(lo)) < ot_at(lo)) {
            for (int it = 0; it < 24 && hi - lo > 1.0; ++it) {
                const double mid = std::floor(0.5 * (lo + hi));
                (de_at(mid, de0_at(mid)) < ot_at(mid) ? lo : hi) = mid;
            }
            dec_max = lo;
        }
    }
    static const bool k_auto = moe_opt("pp-k") == nullptr;   // --moe pp-k fixes it, pp-k=0 turns it off
    // The learned planner's candidates: each path's features for a pass of this call, its posterior and its draw. Computed in every mode (a forced path or
    // the old rules still teach the path that ran); picked by only with split-model on. The eligibility below is facts, not rules: the mix needs a slower
    // card and passes that fill a layer, the prompt chain no uploads under way and a pass its graph took, the decode chain a measured CPU rate, the
    // predictions ranks short of all experts.
    const bool learned = pp_split_model_on() && !force;
    {
        const int64_t c0 = (int64_t) g_pp_cells - (int64_t) n_tokens;
        if (c0 <= g_pp_seen_c0 || g_pp_redraw || g_pp_seed == 0) { g_pp_seed = rls_mix((uint64_t) ggml_time_us()); g_pp_redraw = false; }   // a new request, or the last pass surprised its model
        g_pp_seen_c0 = c0;
    }
    const double cells_mid = std::max(0.0, (double) g_pp_cells - 0.5 * (double) n_tokens);
    if (R_read > 0.0 && A > 0.0) { g_pp_ctok_prior = A / R_read * 1e6; }   // the cells a pass of this call attends to, about
    for (auto & f : g_pp_phi) { f[1] = 0.0; }
    g_pp_pick = -1;
    auto cand = [&](int p, double Up) { g_pp_phi[p][0] = 1.0; g_pp_phi[p][1] = n; g_pp_phi[p][2] = n * g_pp_a_us * cells_mid; g_pp_phi[p][3] = Up; };
    // U per path: the stream's bound by its link and by the SSD reads (Ts; the design names U, Ts is U wherever the experts are not mapped), the mix's
    // upload, the misses the prompt chain uploads behind the router, the decode chain's misses plus its 4-token chunk reads, the predictions the stream's
    // the chains' misses also wait for the SSD where their tensors are still mapped (planner-rules.md C1/R3: rp=1 cold 2k took 43-86 s against
    // 5.5-11 s calculated without it): the misses' mapped bytes over the stream's SSD rate, the same rate Ts uses
    const double ssd_miss = ssd_rate > 0.0 ? miss_ssd_bytes / (ssd_rate * 1e3) : 0.0;
    pp_metrics pm;
    pm.n = n; pm.comp_us_tok = g_pp_c_us > 0.0 ? g_pp_c_us : g_pp_c_chain; pm.ssd_gbs = ssd_rate;   // the chain's bound until a stream measures it (not the load-time read rate: 20x high on OLMoE)
    pm.U = U; pm.M = M; pm.Um = Um; pm.Mm = Mm; pm.up = up; pm.cpu = cpu; pm.ssd_miss = ssd_miss;
    pm.wait_us = pp_wait_us(n); pm.chunk_us = chunk_at(n); pm.dec_f = dec_f; pm.dec_r = dec_r; pm.dec_kc = g_pp_dec_kc;
    pm.Uk = Uk; pm.Mk = Mk;
    cand(PP_RING, estimate_fullstream_upload_time(pm));
    if (has_slower && pp_mix_opt() && n * (double) mc->n_expert_used >= (double) n_expert) { cand(PP_MIX, estimate_mix_upload_time(pm)); }
    // (the chains at any size are candidates again: the calculated crossover dec_max overruled their learned estimates; an unknown chain is now only a
    // trial whose stake pays, and a known slow one loses on its mean - replay: no pick changed). The prompt chain's uploads wait for the same SSD reads.
    if (!busy && (uint32_t) n < g_pp_chain_refused && chain_fits) { cand(PP_CHAIN, estimate_moe_miss_ongpu_time(pm)); }
    // its calculated time (uploads and CPU side by side, the chunks, the learned ratios), not its uploads alone: the uploads alone priced a 100-token GLM call at 4.8 s, the
    // calculation said 1.6 s, it took 2.1 s, and the stream (3.3 s) won every such call (2026-11-10, two cards)
    if (cpu_known) { cand(PP_DEC, estimate_decodechain_time(pm)); }
    // a prediction of 2^i x the picks uploads what the call's tokens rank that high (the same SSD rule as Ts), not the whole layer: the stream's upload at a
    // long prompt where every expert is touched, a fraction at a short one. x PP_K_LATE: what the forced sweep shows it pays beyond that (late fixes, the ranking nodes).
    // Was the stream's upload x (1 - 2^i x picks / experts), upside down: k x32 cost nothing, the planner tried it first (rows 2026-11-08: 57 k x32 picks against 5 k x1).
    for (int i = 0; k_auto && PP_K + i < PP_PATHS && (double) mc->n_expert_used * (double) (1 << i) < (double) n_expert; ++i) {
        cand(PP_K + i, estimate_predictor_upload_time(pm, i, PP_K_LATE));
    }
    int best_mu = -1;
    for (int p = 0; p < PP_PATHS; ++p) {
        if (g_pp_phi[p][1] <= 0.0) { continue; }
        dsf_rls & rp = pp_rls(p);
        // no path beats the compute itself: a few short samples extrapolated to a long call gave the chain -494459 ms (GLM, 4475 tokens, 2026-11-10), and a negative mean always won
        g_pp_pmu[p] = std::max(g_pp_c_us > 0.0 ? g_pp_c_us * (double) n : 0.0, rls_pred(rp, g_pp_phi[p], &g_pp_var[p]));
        rls_draw(rp, g_pp_phi[p], rls_mix(g_pp_seed + (uint64_t) p), &g_pp_drw[p]);
    }
    {
        bool ok[PP_PATHS];
        for (int p = 0; p < PP_PATHS; ++p) { ok[p] = g_pp_phi[p][1] > 0.0; }
        if (g_pp_short.load(std::memory_order_relaxed)) { ok[PP_CHAIN] = false; }   // the lend left a chain layer too few blocks for this call
        // a call under the GPU's smallest expert batch (GGML_OP_OFFLOAD_MIN_BATCH, 32 tokens) streams nothing: its expert matmuls run on the CPU, each expert read from the pool or the SSD. The prompt's
        // 4-token tail (2026-11-08) took 1.1 s on GLM 3.5 and 1.7 s on DeepSeek that way, 0.5 and 0.3 s on the decode chain: about 10% of a 3k prompt. A chain whenever one can run.
        if (n < PP_MIN_BATCH && (ok[PP_CHAIN] || ok[PP_DEC])) {
            for (int p = 0; p < PP_PATHS; ++p) { if (p != PP_CHAIN && p != PP_DEC) { ok[p] = false; } }
        }
        // deterministic: the lowest mean among the paths known here, a trial only where its stake pays over the calls of this size so far
        // (rls_voi_pick; planner-simple/learned, replacing the Thompson draws that lost rows 23768/23769)
        const double R = (double) ++g_pp_size_seen[(uint32_t) n];
        bool known[PP_PATHS];   // ran at comparable features (rls_known): the arms a trial is weighed against
        for (int p = 0; p < PP_PATHS; ++p) { known[p] = ok[p] && rls_known(g_pp_rls[p], g_pp_pmu[p], g_pp_var[p]); }
        // R1 (fork/docs/planner-crossovers.md): above the MEASURED CPU crossover g_mb_auto (the release's max-batch, experts / picks) the chains are no
        // cold pick - no logged class on Qwen3.6/3.8, GLM, MiMo has a chain win there (-97% to -308%), DeepSeek only below 3k. There they are a trial
        // against an arm known here, and a chain measured here (known) competes on its mean. (a448602c58 had dropped this as a ballpark; the
        // ballpark was the calculated dec_max, this is the measured one.)
        static const bool pp_switch_r1 = [] { const char * e = moe_opt("pp-switch"); return !e || atoi(e) != 0; }();
        for (const int p : { PP_CHAIN, PP_DEC }) {
            // the router path has no CPU side to run away with (the 1170 s below was the decode chain): with the switch on, time per token decides it at any size
            // (Qwen3.8-Flash-Next IQ4_XS, 512-token passes: router 2.9-3.0 ms per token against the stream's 3.6-3.8, never tried past 256 under this rule)
            if (p == PP_CHAIN && pp_switch_r1) { continue; }
            // (g_mb_auto = experts / picks is 32 on these models, and the chains win well past it. Forced sweep 2026-10-31, t/s, cache / stream: MiMo 100 tokens 59/28, 200 64/55, 300 62/82; GLM 3.0 56/27, 62/54, 60/80;
            // DeepSeek 200 53/13, 500 52/27, 1000 57/57. So up to the calculated crossover dec_max, at most 256 tokens: dec_max alone says thousands before a stream pass measured the compute, and a chain loses there; a chain that ran competes on its mean at any size)
            // shortcut: 256 is the sweep's crossover on MiMo and GLM, not computed (DeepSeek's is near 1000); use dec_max alone once it is built from a measured compute and link rate
            if (ok[p] && n > std::max((double) g_mb_auto.load(std::memory_order_relaxed), std::min(dec_max, 256.0)) && !known[p]) { ok[p] = false; }   // no trial past the crossover: MiMo 64k warm, a 16384-token decode chain trial calculated 8.6 s took 1170 s (the stream known, so the old test let it run)
        }
        // R1's mirror: at or below that crossover the chains are the cold pick, so an arm that streams and never ran here is no trial there, while a chain can
        // run. DeepSeek 117 tokens (2026-11-10, third start of two builds): two decode chain samples of 3.7 s left a mean of 14.2 s (the first sample's error
        // went to the coefficient of the calculated time, which then read 4.4 s instead of 1.1), an unknown stream, k x1 drew 3.5 s and took 29.5 s (49 GB
        // read from the SSD). A stream measured here competes on its mean at any size.
        if (ok[PP_DEC] || ok[PP_CHAIN]) {
            const double xo = std::max((double) g_mb_auto.load(std::memory_order_relaxed), std::min(dec_max, 256.0));
            for (int p = 0; p < PP_PATHS; ++p) { if (p != PP_CHAIN && p != PP_DEC && ok[p] && !known[p] && n <= xo) { ok[p] = false; } }
        }
        // The predictions compete on their calculated upload (user 2026-11-08: "if we are not trying it prior full stream than it is bug"), the lowest k first: it stages the least,
        // a higher k only after it ran. The mix competes on its mean up to PP_MIX_COLD (2026-11-08 auto rows: MiMo 418 tokens 3018 ms against the stream's 3834, 1016 tokens 3187
        // against 3907; Qwen3.8 IQ3 1.5-4.5k +18-47%, GLM 3.5 2.3k +10%); past it a mix that never ran here is a trial only, never a mean pick: its prior is the stream's (means
        // equal to noise, rampool12 rows 23784-23789) and the cross-card traffic grows with the tokens (MiMo 12k 495 against 933 t/s, DS 12k 304 against 530)
        bool tonly[PP_PATHS];
        for (int p = 0; p < PP_PATHS; ++p) { tonly[p] = p == PP_MIX && g_pp_rls[p].n == 0 && n > PP_MIX_COLD; }
        // The mix's freeze drops the uploads not started yet (llama_moe_cache_pp_freeze_fork, pp_settle), as the prompt chain's graph does: right after a start that is the hot set on its
        // way, and the fast card stays cold for the session (2026-11-08 forced mix cells, Qwen3.8: 24% of the asked experts found on it against 44%, every pass slower than the stream).
        // So no try of a mix not known here while uploads run; a known one keeps its mean.
        if (busy && ok[PP_MIX] && !known[PP_MIX]) { ok[PP_MIX] = false; }
        for (int p = PP_K + 1; p < PP_PATHS; ++p) { if (ok[p] && g_pp_rls[p - 1].n == 0) { ok[p] = false; } }
        // Router or full stream (user 2026-11-10): time per token of the path that ran here against the estimate of the other, and the stream as soon as it costs the same or less.
        // A path measured at this size counts with its real time. The estimates: the router waits for its ids, uploads its picks and computes; the full stream uploads every expert
        // (one expert's upload x the experts) while it computes, so it takes the longer of the two. No stored threshold. --moe pp-switch=0: both compete on their means, as before.
        static const bool pp_switch = [] { const char * e = moe_opt("pp-switch"); return !e || atoi(e) != 0; }();
        g_pp_up_call = up;
        if (pp_switch && ok[PP_RING] && ok[PP_CHAIN]) {
            const double wait = pp_wait_us(n), comp = pm.comp_us_tok * n;
            const double ran_r = pp_took_us(PP_CHAIN, n), ran_s = pp_took_us(PP_RING, n);
            // both carry the SSD now (it was left out while the SSD term counted locked and pool weights as file reads: 93 GB of GLM's at 6.3 GB/s, 14.8 s against a 3.9 s upload)
            const double t_r = ran_r > 0.0 ? ran_r : estimate_router_time_per_token(pm) * n, t_s = ran_s > 0.0 ? ran_s : estimate_fullstream_time_per_token(pm) * n;
            const bool stream = t_s <= t_r;
            ok[stream ? PP_CHAIN : PP_RING] = false;
            llama_moe_stats_log_fork("DEC", format("prompt call of %.0f tokens: %s: router %.3f ms per token (%s; picks %.0f, wait %.0f, compute %.0f ms), full stream %.3f ms per token (%s; upload %.0f ms)", n, stream ? "full stream" : "router",
                    t_r / 1e3 / n, ran_r > 0.0 ? "measured" : "estimate", up / 1e3, wait / 1e3, comp / 1e3, t_s / 1e3 / n, ran_s > 0.0 ? "measured" : "estimate", U / 1e3).c_str());
        }
        g_pp_pick = rls_voi_pick(g_pp_pmu, g_pp_var, ok, PP_PATHS, R, &g_pp_trial, known, tonly);
        best_mu   = rls_argmin(g_pp_pmu, ok, PP_PATHS);
    }
    bool chain, dec, mix;
    double m;
    if (learned && g_pp_pick >= 0) {
        chain = g_pp_pick == PP_CHAIN;
        dec   = g_pp_pick == PP_DEC;
        mix   = g_pp_pick == PP_MIX;
        m     = g_pp_pick >= PP_K ? (double) (1 << (g_pp_pick - PP_K)) : 0.0;
        why   = g_pp_trial ? "trial: its expected gain over the calls of this size beats its expected loss" : g_pp_pick == best_mu ? "lowest mean" : "lowest mean among the paths known here";
        g_pp_dec = dec;
        g_pp_mb = chain || dec ? ub_max : 0;
        g_pp_mix_call = mix;
    } else {
    g_pp_pick = -1;   // the old rules decide: no regret is booked against the draws
    const bool dec_ok = cpu_known && de < ot;
    chain = !g_pp_short.load(std::memory_order_relaxed) && (force ? strcmp(force, "chain") == 0 : !busy && (!dec_ok || td > 0.0) && pp_chain_pays(n, up, U, !dec_ok, &why));   // a refused chain stays out of the second decision
    dec = force ? strcmp(force, "dec") == 0 : dec_ok && !chain;
    if (force) { why = "--moe pp-path"; } else if (busy) { why = "uploads on their way"; } else if (dec) { why = "the decode chain is faster"; }
    g_pp_dec = dec;
    g_pp_mb = chain || dec ? ub_max : 0;
    const char * why_mix = "";
    mix = force ? strcmp(force, "mix") == 0 && has_slower && pp_mix_opt() : !chain && !dec && has_slower && pp_mix_opt() && pp_mix_pick(n, (int) mc->n_expert_used, n_expert, &why_mix);
    g_pp_mix_call = mix;
    if (why_mix[0]) { why = why_mix; }
    const double r = pp_took_us(mix ? PP_MIX : PP_RING, n);
    m = k_auto && !force && !chain && !dec && !why_mix[0] ? pp_k_pick(n, r, mix ? Um : U, (int) mc->n_expert_used, n_expert, &why) : 0.0;
    }
    if (k_auto) { ggml_backend_pred_set_fork(m); }
    double o = 0.0;   // the fastest one-card stream measured here
    for (int p = 0; p < PP_PATHS; ++p) {
        const double t = p == PP_CHAIN || p == PP_DENSE ? 0.0 : pp_took_us(p, n);
        if (t > 0.0 && (o <= 0.0 || t < o)) { o = t; }
    }
    const double d = pp_took_us(PP_DENSE, n);
    // only a one-card path measured at this size can veto the split; with none measured the dense split planner weighs the split against the planned one-card time
    // (2026-11-07 GLM 3.5 12k rampool6 warm: the split measured 15.4 s at 10496 tokens and no stream there, so the pass was forced to one card, 17.3 s, pp 583 -> 508)
    g_pp_no_dense = !learned && d > 0.0 && o > 0.0 && o <= d;   // the learned planner has no veto: the split weighs every path's posterior (fault B)
    // the path this call takes and what it is expected to save over the stream (the dense split planner weighs its one-card time less this, llama_dsplit_decide_fork):
    // measured at this size where it ran; else the decode chain's calculation, the chain's compute plus its misses' uploads; a first try of a prediction saves nothing yet
    {
        int pth = PP_RING;
        if (chain) { pth = PP_CHAIN; } else if (dec) { pth = PP_DEC; } else if (m > 0.0) { pth = PP_K; for (double q = m; q > 1.0; q /= 2.0) { ++pth; } } else if (mix) { pth = PP_MIX; }
        const double tm = pp_took_us(pth, n);
        const double tp = tm > 0.0 ? tm : pth == PP_DEC ? de : pth == PP_CHAIN ? comp_at(n) + up : ot;
        g_pp_path_call = pth;
        g_pp_save_us = std::max(0.0, ot - tp);
    }
    std::string ks;
    for (int i = 0; i < PP_PATHS - PP_K; ++i) {
        if (const double t = pp_took_us(PP_K + i, n); t > 0.0) { ks += format(", k x%d %.0f ms", 1 << i, t / 1e3); }
    }
    std::string links;
    for (int k = 0; k < mc->n_links; ++k) { links += format("%s%.1f", k ? "/" : "", mc->gbs_link[k]); }
    const std::string pick = format("prompt call of %.0f tokens: %s (%s; decode chain to %.0f tokens, %.0f ms calculated against %.0f; took: decode chain %.0f ms, chain %.0f ms, stream %.0f ms, mix %.0f ms, dense %.0f ms%s; streaming upload %.0f ms, with the mix %.0f ms, misses uploaded %.0f ms or on the CPU %.0f ms over %zu experts off the card; links %s GB/s, CPU %.1f GB/s; stream reads %.1f GB from the SSD at %.2f GB/s; decode chain: misses x%.2f, 4-token chunks %.0f ms x%.2f at %.0f GB/s weight read, fixed %.0f ms; compute %.0f ms)",
            n, chain ? "prompt chain" : dec ? "decode chain" : format("stream%s%s", mix ? ", mix" : "", m > 0.0 ? format(", k x%.0f", m).c_str() : "").c_str(), why, std::min(dec_max, (double) ub_max), de / 1e3, ot / 1e3, td / 1e3, pp_took_us(PP_CHAIN, n) / 1e3, pp_took_us(PP_RING, n) / 1e3, pp_took_us(PP_MIX, n) / 1e3,
            pp_took_us(PP_DENSE, n) / 1e3, ks.c_str(), U / 1e3, Um / 1e3, up / 1e3, cpu / 1e3, v.size(), links.c_str(), mc->cpu_gbs_meas > 0 ? mc->cpu_gbs_meas : knobs().cpu_gbs, M / 1e9, ssd_rate, dec_r, g_pp_dec_chunk / 1e3, g_pp_dec_kc, R_read / 1e9, dec_f / 1e3, comp_at(n) / 1e3);
    LLAMA_LOG_WARN("moe-cache: %s\n", pick.c_str());
    llama_moe_stats_log_fork("CAL", pick.c_str());
    {   // every candidate's posterior and draw (feedback_planner_explains_picks): what the pick saw
        std::string cs;
        for (int p = 0; p < PP_PATHS; ++p) {
            if (g_pp_phi[p][1] <= 0.0) { continue; }
            cs += format("%s%s %.0f +- %.0f ms, draw %.0f (n=%d, nis %.2f)", cs.empty() ? "" : "; ", pp_path_name(p).c_str(), g_pp_pmu[p] / 1e3, std::sqrt(std::max(0.0, g_pp_var[p])) / 1e3,
                    g_pp_drw[p] / 1e3, g_pp_rls[p].n, g_pp_rls[p].nis);
        }
        const std::string pl = format("plan: prompt call of %.0f tokens at %.0f cells: %s; %s%s", n, cells_mid, cs.c_str(), learned ? "picked " : "old rules (split-model=0 or a forced path) picked ",
                (chain ? std::string("prompt chain") : dec ? std::string("decode chain") : format("stream%s%s", mix ? ", mix" : "", m > 0.0 ? format(", k x%.0f", m).c_str() : "")).c_str());
        LLAMA_LOG_WARN("moe-cache: %s\n", pl.c_str());
        llama_moe_stats_log_fork("PLN", pl.c_str());
    }
    llama_moe_stats_io_fork(0);   // the reads of this call start counting here
}

// For the learned split planner (dsf_plan): the lowest drawn one-card time of a pass of n tokens at c cells over this call's candidates, by the same draws the
// call's pick used (one consistent view per request); its posterior mean and variance in *mu, *var. -1: no candidate (the call was not planned here).
// shortcut: a candidate's upload is the call's pass size's (U at n of the call); passes of other sizes in one call are the tail only. Upgrade if calls mix sizes.
double llama_moe_cache_pp_one_draw_fork(int64_t n, int64_t c, double * mu, double * var, int * path, bool * known, int * n_obs) {
    // planner-simple: the one-card side is the path this call picked, its posterior mean and variance at this pass (no draws); -1 when none
    std::lock_guard<std::mutex> lk(g_pp_mu);
    const int p = g_pp_pick >= 0 ? g_pp_pick : -1;
    if (p < 0 || g_pp_phi[p][1] <= 0.0) { return -1.0; }
    const double phi[4] = { 1.0, (double) n, (double) n * g_pp_a_us * (double) c, g_pp_phi[p][3] };
    double v = 0.0;
    dsf_rls & rp = pp_rls(p);
    const double m = rls_pred(rp, phi, &v);
    if (known) { *known = rls_known(rp, m, v); }
    if (n_obs) { *n_obs = rp.n; }
    if (mu) { *mu = m; }
    if (var) { *var = v; }
    if (path) { *path = p; }
    return std::max(0.0, m);
}
// this call has one-card candidates (the split planner learns its own one-card model otherwise)
bool llama_moe_cache_pp_planned_fork() { std::lock_guard<std::mutex> lk(g_pp_mu); for (const auto & f : g_pp_phi) { if (f[1] > 0.0) { return true; } } return false; }
// the pass size the call runs with, once the context has decided it (planner-rules.md C19/C21: pp_decide runs before the pass planner and its
// features were for min(call, limit)); the candidates keep their pick and draws, the features follow the pass that runs
void llama_moe_cache_pp_repass_fork(uint32_t n_tokens, uint32_t ub) {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    const double n = (double) std::min(n_tokens, std::max<uint32_t>(1, ub));
    for (auto & f : g_pp_phi) {
        if (f[1] <= 0.0 || f[1] == n) { continue; }
        f[2] *= n / f[1];   // n x a x cells: the cells stay, the tokens change
        f[1] = n;
    }
}
double llama_moe_cache_att_us_cell_fork() { std::lock_guard<std::mutex> lk(g_pp_mu); return g_pp_a_us; }   // attention per token and cell (us), 0 = not fitted yet
std::string llama_moe_cache_pp_path_name_fork(int p) { return p >= 0 && p < PP_PATHS ? pp_path_name(p) : std::string("-"); }

// For the dense split planner (llama_dsplit_decide_fork), which weighs the split against the one-card path this call took (2026-11-06)
bool llama_moe_cache_pp_dec_pass_fork() { return g_pp_dec_built; }
bool llama_moe_cache_pp_no_dense_fork() { return g_pp_no_dense.load(std::memory_order_relaxed); }
double llama_moe_cache_pp_saving_us_fork() { return g_pp_save_us.load(std::memory_order_relaxed); }
double llama_moe_cache_pp_path_us_fork(int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    return pp_took_us(g_pp_path_call.load(std::memory_order_relaxed), (double) n_tokens);
}
// the fastest one-card path measured at this pass size (us): what one card does best there; the decode chain is no candidate at the sizes the split runs
double llama_moe_cache_pp_best_one_us_fork(int64_t n_tokens) {
    std::lock_guard<std::mutex> lk(g_pp_mu);
    double o = 0.0;
    for (int p = 0; p < PP_PATHS; ++p) {
        if (p == PP_DENSE || p == PP_DEC) { continue; }
        const double t = pp_took_us(p, (double) n_tokens);
        if (t > 0.0 && (o <= 0.0 || t < o)) { o = t; }
    }
    return o;
}
// the split is expected faster than this call's path: the call streams instead (no decode chain, no prompt chain, no predicted staging), so the split can run under it
bool llama_moe_cache_pp_yield_fork(const char * why) {
    static const bool k_auto = moe_opt("pp-k") == nullptr, forced = moe_opt("pp-path") != nullptr;
    if (forced || (!k_auto && ggml_backend_pred_get_fork() > 0.0)) { return false; }   // a path the user set keeps its passes
    g_pp_dec = false;
    g_pp_mb = 0;
    if (k_auto) { ggml_backend_pred_set_fork(0.0); }
    g_pp_path_call = g_pp_mix_call.load(std::memory_order_relaxed) ? PP_MIX : PP_RING;
    g_pp_save_us = 0.0;
    const std::string s = format("prompt call: the dense split takes it from its one-card path (%s)", why);
    LLAMA_LOG_WARN("moe-cache: %s\n", s.c_str());
    llama_moe_stats_log_fork("CAL", s.c_str());
    return true;
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
// the engine's byte counters, for the per-token lines of stats.log (llama-moestate-fork.cpp): cpu compute, uploads, prompt streaming, then every link's bytes; 0 when no cache runs
int llama_moe_cache_counters_fork(int64_t * v, int cap) {
    moe_cache * mc = g_cache;
    if (!mc || cap < 3 + mc->n_links) { return 0; }
    v[0] = mc->cpu_bytes.load(); v[1] = mc->up_bytes.load(); v[2] = mc->pf_bytes.load();
    for (int k = 0; k < mc->n_links; ++k) { v[3 + k] = mc->link_bytes[k].load() + mc->pf_link[k].load(); }
    return 3 + mc->n_links;
}

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
    ggml_backend_set_moe_read_fn_fork(nullptr, nullptr);
    ggml_backend_set_dma_hook_fork(nullptr);
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
    if (g_ppoff_all > 0) {
        LLAMA_LOG_WARN("moe-cache: pp-off: %.1f%% of the prompt's expert-token pairs kept off the card: %.1f%% on the miss device, %.1f%% on the CPU\n",
                100.0*g_ppoff_pairs/g_ppoff_all, 100.0*g_ppoff_dev/g_ppoff_all, 100.0*(g_ppoff_pairs - g_ppoff_dev)/g_ppoff_all);
    }
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
    mc->stats_stop.store(true);
    if (mc->stats_thr.joinable()) { mc->stats_thr.join(); }
    if (mc->rp_on) {   // the readers stop before their chunks go; the CPU kernels go back to the weights' own pointers
        rp_log("exit");
        if (auto set = (void (*)(const void * (*)(const ggml_tensor *, int64_t))) cpu_proc_fork("ggml_backend_cpu_set_expert_ptr_hook")) { set(nullptr); }
        ggml_rampool_close_fork();
        for (auto * b : mc->rp_bufs) { ggml_backend_buffer_free(b); }
        mc->rp_bufs.clear();
        mc->rp_on = false;
    }
    rp_pre_free(mc);
    rp_raw_free(mc);
    if (mc->preheat_thr.joinable()) { mc->preheat_thr.join(); }
    if (mc->fast_thr.joinable()) { mc->fast_thr.join(); }
    { std::lock_guard<std::mutex> lk(g_vram_mtx); g_vram.clear(); }   // the watch entries of this cache: a re-init adds its own (stale ones were sampled and stored on)
    mc->lazy_stop.store(true);
    if (mc->lazy_thr.joinable()) { mc->lazy_thr.join(); }
    for (auto * b : mc->staging) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * b : mc->rd_staging) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto & s : mc->jit_ring) { if (s.ev) { ggml_backend_event_synchronize(s.ev); ggml_backend_event_free(s.ev); } if (s.buf) { ggml_backend_buffer_free(s.buf); } }
#ifndef _WIN32
    for (auto & f : mc->rd_fds) { if (f.second >= 0) { close(f.second); } }
#endif
    for (auto & mv : mc->lazy_moved) { mv.first->data = mv.second; }   // the model outlives this cache (a new context, a re-init): its weights point at the mapping again, not at freed pinned memory
    for (auto * b : mc->pin_chunks) { ggml_backend_buffer_free(b); }
    for (auto * b : mc->pbufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->pctxs) { ggml_free(c); }
    for (auto * b : mc->miss_bufs) { ggml_backend_buffer_free(b); }
    if (mc->miss_ctx) { ggml_free(mc->miss_ctx); }
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
           g_kv_epoch.load() == g_kv_epoch_built &&   // a moved KV zone: the old graph's views point at the old place
           g_lend_epoch.load() == g_lend_epoch_built &&   // a graph placed elsewhere in the pool since: the old graph's tensors sit at the old place
           (!g_pp_built || pp_chain_allowed()) && g_pp_dec_built == g_pp_dec.load(std::memory_order_relaxed) && (!g_pp_mix_built || pp_mix_allowed()) && g_pp_k_built == (ggml_backend_pred_get_fork() > 0.0) && g_pp_mix_call_built == g_pp_mix_call.load(std::memory_order_relaxed);       // a graph of the prompt chain has no CPU chain: only a prompt call may run it
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
    g_lend_epoch_built = g_lend_epoch.load();
    g_pp_built = g_pp_graph;
    g_pp_dec_built = g_pp_dec.load(std::memory_order_relaxed);
    g_pp_k_m = ggml_backend_pred_get_fork();
    g_pp_k_built = g_pp_k_m > 0.0;
    g_pp_mix_call_built = g_pp_mix_call.load(std::memory_order_relaxed);
    g_pp_graph = false;
    g_pp_mix_built = g_pp_mix_graph;
    g_pp_mix_graph = false;
}

namespace {
bool pinned(const moe_cache * mc, const layer_state & ls, int32_t e) {
    const uint64_t stay = (uint64_t) knobs().slow_stay;
    return ls.sticky[e] || (ls.slow && ls.cached_since[e] && mc->n_steps + 1 - ls.cached_since[e] < stay);
}
} // namespace

// C-linkage, exported entry for the server (llama.h): the planner above is a C++ function other engine files declare locally
uint32_t llama_moe_prefill_pass_fork(uint32_t n_tokens, uint32_t ub_max) { return llama_moe_cache_prefill_ubatch_fork(n_tokens, ub_max); }
