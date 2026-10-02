#include "llama-moecache.h"
#include "llama-moestate.h"

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
#include <set>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
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
static bool g_pred_train_now = false, g_pred_train_built = false;
static uint64_t g_pred_epoch = 0, g_pred_epoch_built = 0;   // bumped when a layer's predictor is switched on or off
// 2-GPU prefill: the second GPU's share of the experts in big batches (0: one GPU); set by the prefill tuner, baked into
// the graph, so a change bumps the epoch and the next graph is rebuilt
static float    g_split_share = 0.0f; // alpha: 1 = all GPUs by link bandwidth, 0 = fastest GPU only
static uint64_t g_split_epoch = 0, g_split_epoch_built = 0;
static bool     g_split_enabled = false; // the compute buffers were reserved for a split (decided before the first reserve)
static std::vector<std::pair<ggml_backend_dev_t, double>> g_early_gbs; // early link probe, until the cache's own probe

namespace {

// LLAMA_MOE_CACHE_TRACE=<prefix>: TRACE=<steps> (ctl file or env) records that many decode steps, after
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
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    std::vector<uint32_t> expert_count;   // expert id -> uses, halved every LLAMA_MOE_CACHE_HALVE_EVERY steps
    std::vector<uint32_t> win_count;      // expert id -> uses in the last 64 tokens
    std::deque<std::vector<int32_t>> recent; // ids of those tokens, oldest first
    // where up/gate/down live on disk (fd < 0: unknown, upload from host memory)
    int    src_fd[3]   = { -1, -1, -1 };
    size_t src_offs[3] = { 0, 0, 0 };

    std::vector<uint64_t> glob_count;     // expert id -> lifetime uses
    uint64_t              glob_max = 1;
    std::vector<bool>     sticky;         // expert id -> never evicted (most-used by lifetime count)
    std::vector<bool>     hot;            // expert id -> always hot: the fewest experts covering LLAMA_MOE_CACHE_HOT_FRAC of the layer's picks
    bool                  slow = false;   // this layer's cache sits on a GPU with a slower upload link than the fastest one
    int                   link = 0;       // upload link (GPU) of this layer's cache: 0 = the offload / fastest GPU, 1.. the others
    int32_t               n_cache = 0;    // cache slots (the rest stream predicted experts)
    std::vector<uint8_t>  is_stream;      // slot -> holds streamed (predicted) experts, not the cache
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
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;
    int32_t window      = 64; // recent-usage window in tokens (--moe-cache-window)

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;
    std::map<const ggml_tensor *, std::pair<size_t, int>> by_src; // host weight -> (layer, 0 up / 1 gate / 2 down)

    // big-batch prefill: experts served device-to-device from the cache vs over PCIe
    uint64_t n_src_hit   = 0;
    uint64_t n_src_query = 0;
    uint64_t n_adopted   = 0; // prefill preheat: experts kept from prompt batches
    uint64_t n_evictions = 0; // experts evicted from the cache (all kinds)

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
    int                      in_flight_stream = 0; // of those, stream jobs (the step doesn't wait for them)
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

    // hot-set profile: per-expert lifetime uses saved at shutdown, preloaded at the next start
    std::string profile;

    // router prediction (LLAMA_MOE_CACHE_PREDICT=M): at layer L's CPU expert op a helper thread applies
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
    int64_t  lv_slice_len = 36, lv_warm = 4; // slice length and skipped tokens of the knob under test (state-carrying knobs need long ones)
    std::string tuned_text;        // the self-tuner's decisions (NAME=value lines), kept in the state file: the next start begins from them
    uint64_t lv_cycle = 0;         // cycles finished: the candidate order rotates so no candidate always runs first (cold cache)
    int64_t  lv_slice_n = 0;
    double   lv_slice_sum = 0;
    uint64_t lv_rest_until = 0;
    uint64_t lv_rest = 0;          // tokens between cycles: doubles while cycles confirm their decisions
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

    // learned predictors (in-graph NLMS, see llama_moe_cache_layer::pred_m): trained every pred_train
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
    double ddr_gbs     = 44;   // tools/bench/ddrbw: CPU + both DMAs together peak at 44-45 GB/s (DDR4-3200 ECC, 2 ch); GATE=3: an upload starts only while DDR demand + its link rate stays under this
    double stream    = 1e9; // predicted uploads use up to this many stream slots per layer (0: evict cache slots, old path)
    double stream_m  = 12;  // candidates per target layer in stream mode (over-predict; no confidence cut)
    double slotkeep  = 0;   // a prediction may only replace a stream slot holding a lower-scored one of this step
    double predict   = 1;   // the learned predictors (allocated with --moe-predict) run at all; SELF_TUNE decides it
    double self_tune = 1;   // self-tuner: A/B the streaming knobs one at a time on real decode token time (self_tune())
    double offset      = 1; // predicted uploads only for layers far enough ahead to land in time on their link
    double stream_slow = 1; // stream onto slow-link (x4) layers too
    double trace       = 0; // LLAMA_MOE_CACHE_TRACE set: record this many steps (re-armed whenever a ctl file sets it)
    double trace_after = 0; // ... starting after this many steps
    double jit         = 1; // JIT miss offload (JIT=0 off): once a layer's router ids reach the host, upload the k misses (k from the
                            // measured link / CPU / combined RAM rates) that make CPU + link finish soonest into cache slots on
                            // the chain's stream, so the GPU computes them this token while the CPU computes the rest
};

bool knob_set(knobs_t & k, const std::string & name, double v) {
    static const std::pair<const char *, double knobs_t::*> fields[] = {
        { "SWAP_FRAC", &knobs_t::swap_frac }, { "HOT_FRAC", &knobs_t::hot_frac }, { "STICKY", &knobs_t::sticky }, { "SLOW_STAY", &knobs_t::slow_stay },
        { "MARGIN", &knobs_t::margin }, { "BUDGET", &knobs_t::budget }, { "CPU_GBS", &knobs_t::cpu_gbs }, { "LINK", &knobs_t::link },
        { "GATE", &knobs_t::gate }, { "GATE_MAX_US", &knobs_t::gate_max_us }, { "DDR_GBS", &knobs_t::ddr_gbs },
        { "MARGIN_MB", &knobs_t::margin_mb }, { "WAIT", &knobs_t::wait }, { "AUTO", &knobs_t::auto_tune },
        { "BIG", &knobs_t::big }, { "L3PF", &knobs_t::l3pf }, { "AUTO_L3", &knobs_t::auto_l3 }, { "TBP", &knobs_t::tbp }, { "TBP_LAYERS", &knobs_t::tbp_layers }, { "CHUNK_KB", &knobs_t::chunk_kb },
        { "STREAM", &knobs_t::stream }, { "STREAM_M", &knobs_t::stream_m }, { "SLOTKEEP", &knobs_t::slotkeep }, { "SELF_TUNE", &knobs_t::self_tune }, { "PREDICT", &knobs_t::predict },
        { "OFFSET", &knobs_t::offset }, { "STREAM_SLOW", &knobs_t::stream_slow },
        { "TRACE", &knobs_t::trace }, { "TRACE_AFTER", &knobs_t::trace_after }, { "JIT", &knobs_t::jit },
    };
    for (const auto & f : fields) {
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
        for (const char * n : { "MARGIN_MB", "SWAP_FRAC", "HOT_FRAC", "STICKY", "SLOW_STAY", "MARGIN", "BUDGET", "CPU_GBS", "LINK", "GATE", "GATE_MAX_US", "STREAM", "STREAM_M", "TRACE", "TRACE_AFTER", "OFFSET", "STREAM_SLOW", "DDR_GBS", "WAIT", "CHUNK_KB", "AUTO", "TBP", "TBP_LAYERS", "BIG", "L3PF", "AUTO_L3", "SLOTKEEP", "SELF_TUNE", "PREDICT", "JIT" }) {
            if (const char * e = getenv((std::string("LLAMA_MOE_CACHE_") + n).c_str())) {
                knob_set(r, n, atof(e));
                user_knobs().insert(n);
            }
        }
        return r;
    }();
    return k;
}

enum class policy { halve, window, hybrid, add };

policy get_policy() {
    static const policy p = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_POLICY");
        if (e && strcmp(e, "window") == 0) return policy::window;
        if (e && strcmp(e, "hybrid") == 0) return policy::hybrid;
        if (e && strcmp(e, "halve")  == 0) return policy::halve;
        return policy::add;
    }();
    return p;
}

// eviction score, in "uses" units so the pay-back margin applies to all policies
double score(const layer_state & ls, int32_t id) {
    switch (get_policy()) {
        case policy::window: return ls.win_count[id];
        case policy::hybrid: return ls.win_count[id] * ((double) ls.glob_count[id] / (double) ls.glob_max);
        case policy::add: {
            static const double k = [] {
                const char * w = getenv("LLAMA_MOE_CACHE_GLOBAL_WEIGHT");
                return w ? atof(w) : 16.0;
            }();
            return ls.win_count[id] + k * ((double) ls.glob_count[id] / (double) ls.glob_max);
        }
        case policy::halve:  return ls.expert_count[id];
    }
    return 0;
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
        kv.emplace_back("hot." + std::to_string(il), v);
    }
    if (!mc->tuned_text.empty()) {
        // the self-tuner's decisions of this GPU count: tuned.l<links> = NAME=value NAME=value ...
        std::string v = mc->tuned_text;
        for (char & c : v) { if (c == '\n') { c = ' '; } }
        kv.emplace_back("tuned.l" + std::to_string(mc->n_links), v);
    }
    moe_state_set(mc->profile, kv);
}

// not evictable: sticky, or cached on a slow-link layer for less than LLAMA_MOE_CACHE_SLOW_STAY steps (default 1024):
// on a slow link a swap costs more, so an expert must stay long enough to pay for its upload
bool pinned(const moe_cache * mc, const layer_state & ls, int32_t e);

// seed usage from the profile and queue the most-used experts of each layer for upload;
// the first step() waits for them and publishes them. Returns the number queued.
// Always-hot experts (formatting, common tokens) stay in VRAM: the most-used experts by lifetime
// count, up to LLAMA_MOE_CACHE_STICKY (default 0, e.g. 0.3) of each layer's slots, are never evicted.
// the always-hot set: the fewest experts (by lifetime count) covering LLAMA_MOE_CACHE_HOT_FRAC (default 0.75) of the picks
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
void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy);
void page_hint(const moe_cache * mc, const layer_state & ls, int32_t expert, bool drop);

size_t profile_preload(moe_cache * mc, const llama_model & model) {
    std::vector<std::vector<uint64_t>> counts;
    const char * from = mc->profile.c_str();
    bool ok = !mc->profile.empty();
    std::vector<std::pair<std::string, std::string>> sect;
    ok = ok && moe_state_section_kv(mc->profile, sect);
    for (size_t il = 0; ok && il < mc->layers.size(); ++il) {
        const std::string key = "hot." + std::to_string(il);
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
    LLAMA_LOG_INFO("moe-cache: usage profile from %s\n", from);
    size_t queued = 0;
    std::lock_guard<std::mutex> lk(mc->wmtx);
    for (size_t il = 0; il < mc->layers.size(); ++il) {
        auto & ls = mc->layers[il];
        ls.glob_count = counts[il];
        ls.glob_max   = std::max<uint64_t>(1, *std::max_element(ls.glob_count.begin(), ls.glob_count.end()));
        refresh_sticky(ls);
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < (int32_t) ls.glob_count.size(); ++e) {
            if (ls.glob_count[e] > 0) {
                ids.push_back(e);
            }
        }
        std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
        for (int32_t s = 0; s < std::min<int32_t>(ls.n_cache, (int32_t) ids.size()); ++s) {
            ls.slot_in_flight[s] = true;
            ls.queued[ids[s]]    = 1;
            upload_job pj { il, ids[s], s };
            pj.t_queued = ggml_time_us();
            mc->todo.push_back(pj);
            queued++;
        }
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

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, const struct ggml_tensor * op, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    // big-batch prefill ids are observed too: they warm the cache before decode
    // (hits aren't counted there, the cache graph isn't built for big batches)
    const bool prefill = n_tokens > llama_moe_cache_max_batch();

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
    mc->last_tokens  = n_tokens;
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
            ls->recent.back().push_back(id);
            ls->glob_max = std::max(ls->glob_max, ++ls->glob_count[id]);
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
    if (ls.up_t[e] == 0) {
        return;
    }
    const int k = ls.link;
    mc->ev_n[k]++;
    mc->ev_hits[k]    += ls.up_hits[e];
    const double cpu_us = (double) (ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]) / (knobs().cpu_gbs * 1e3);
    mc->ev_paid[k]    += ls.up_hits[e] * cpu_us >= ls.up_cost[e];
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
void pred_requant(llama_moe_cache_layer & pub) {
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

void unpin_helper() {
#if defined(__linux__)
    if (CPU_COUNT(&g_proc_mask) > 0) {
        pthread_setaffinity_np(pthread_self(), sizeof(g_proc_mask), &g_proc_mask);
    }
#endif
}

std::vector<std::vector<int>> l3_domain_cores();

void resource_probe(moe_cache * mc) {
    const char * pe = getenv("LLAMA_MOE_CACHE_PROBE");
    if ((pe && pe[0] == '0') || mc->layers.empty()) {
        return;
    }
    // physical cores (first SMT thread of each): the ramp's upper bound (Linux sysfs / Windows; else logical CPUs)
    int n_phys = 0;
    for (const auto & d : l3_domain_cores()) {
        n_phys += (int) d.size();
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
    // one ramp: step 0 runs the links alone, then each step adds a CPU reader thread and measures it alone and
    // together with every link, until neither rises (< 3% twice in a row). Peak CPU alone = the CPU's RAM read rate,
    // the thread count reaching 95% of it = cores that saturate RAM; peak CPU + links = the RAM budget
    double cpu_alone = 0, total = 0;
    std::vector<double> link_alone, link_mixed, tmp, ramp, ramp_all;
    const int n_max = std::max(1, n_phys > 0 ? n_phys : (int) std::thread::hardware_concurrency());
    int flat = 0;
    for (int nt = 0; nt <= n_max && flat < 2; ++nt) {
        double g = 0, gm = 0;
        if (nt == 0) {
            run(0, true, 0.15, g, link_alone);
            continue;
        }
        run(nt, false, 0.08, g, tmp);
        run(nt, true, 0.08, gm, link_mixed);
        for (double l : link_mixed) { gm += l; }
        flat = g < cpu_alone * 1.03 && gm < total * 1.03 ? flat + 1 : 0;
        cpu_alone = std::max(cpu_alone, g);
        total     = std::max(total, gm);
        ramp.push_back(g);
        ramp_all.push_back(gm);
    }
    int n_sat = (int) ramp.size();
    for (int i = 0; i < (int) ramp.size(); ++i) {
        if (ramp[i] >= 0.95 * cpu_alone) { n_sat = i + 1; break; }
    }
    mc->cpu_sat_threads = n_sat;
    if (!getenv("LLAMA_MOE_CACHE_CPU_GBS")) { knobs().cpu_gbs = cpu_alone; }
    if (!getenv("LLAMA_MOE_CACHE_DDR_GBS")) { knobs().ddr_gbs = std::max(cpu_alone, total); }
    // the DDR budget gate (GATE=3) needs the budget just measured: on by default then (GLM short chat +6%, MiMo +16% in a pair); the tuner can turn it off
    if (!user_knobs().count("GATE")) { knobs().gate = 3; }
    std::string links, rs;
    for (int k = 0; k < mc->n_links; ++k) {
        mc->gbs_link[k] = link_alone[k];
        links += tr_fmt(" link %d %.1f", k, link_alone[k]);
    }
    for (size_t i = 0; i < ramp.size(); ++i) {
        rs += tr_fmt("%s%zu:%.1f/%.1f", i ? " " : "", i + 1, ramp[i], ramp_all[i]);
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
    if (!c || !c->buffer || c->nb[2] != weight->nb[2] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        return false;
    }
    static const double frac = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_ADOPT");
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
        // victim: empty slot, else the non-sticky cached expert with the least lifetime use,
        // and only if this expert (whose use this prompt just counted) was used more
        uint64_t least = UINT64_MAX;
        for (int32_t sl = 0; sl < ls.pub.n_slots; ++sl) {
            if (ls.slot_in_flight[sl] || ls.is_stream[sl]) {
                continue;
            }
            const int32_t v = ls.slot_expert[sl];
            if (v < 0) { slot = sl; least = 0; break; }
            if (pinned(mc, ls, v)) {
                continue;
            }
            if (ls.glob_count[v] < least) { least = ls.glob_count[v]; slot = sl; }
        }
        if (slot < 0 || least >= ls.glob_count[expert]) {
            return false;
        }
        const int32_t victim = ls.slot_expert[slot];
        if (victim >= 0) {
            ls.expert_slot[victim] = -1; // no later batch reads the slot as the victim
            ls.slot_expert[slot]   = -1;
            ls.adopt_victims.push_back(victim);
        }
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

// table writes go to the host table (host memory, read by the CPU expert op) and mark the layer; the device
// copy is flushed in one transfer per dirty layer at the next sync point (split callback, end of step) instead of
// one synchronous 4-byte copy per entry
std::mutex                                g_tbl_mtx;
std::vector<llama_moe_cache_layer *>      g_tbl_dirty;

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
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

// the device table marks uncached experts -1 instead of the dummy slot when every expert type goes through the
// quantized mat-vec kernels, which skip negative ids (no read of the dummy slot's weights for every miss)
bool g_neg_ids = false;

void flush_tables() {
    std::vector<llama_moe_cache_layer *> d;
    {
        std::lock_guard<std::mutex> lk(g_tbl_mtx);
        d.swap(g_tbl_dirty);
    }
    std::vector<int32_t> v;
    for (auto * pub : d) {
        const int32_t * h = (const int32_t *) pub->host_table->data;
        v.assign(h, h + ggml_nelements(pub->dev_table));
        for (auto & x : v) {
            x = x >= pub->n_slots ? (g_neg_ids ? -1 : pub->n_slots) : x; // n_slots + 1: JIT pool expert, uncached here
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
                t[e] = l.pool_slot[e] >= 0 ? l.pool_slot[e] : (g_neg_ids ? -1 : l.pub.jit_n);
            }
            ggml_backend_tensor_set(l.pub.jit_table, t.data(), 0, t.size()*sizeof(int32_t));
        }
    }
}

} // namespace

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

static bool moe_src_cb(const ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev,
                const void ** data, ggml_backend_buffer_t * buffer, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    layer_state & ls = mc->layers[it->second.first];
    std::lock_guard<std::mutex> lock(mc->mtx);
    // GPU-offloaded prompt batches never reach the CPU observer (moe_obs_cb): count the prompt's
    // experts here (once per expert and batch, on up) for the lifetime usage and the saved profile.
    // Not queued for upload: extra uploads after a long prompt measured slower (they compete with decode)
    if (it->second.second == 0 && expert >= 0 && expert < (int32_t) ls.expert_slot.size()) {
        ls.expert_count[expert]++;
        ls.glob_max = std::max(ls.glob_max, ++ls.glob_count[expert]);
    }
    ggml_tensor * c = it->second.second == 0 ? ls.pub.up_c : it->second.second == 1 ? ls.pub.gate_c : ls.pub.down_c;
    if (!c || !c->buffer || c->nb[2] != weight->nb[2] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        return false;
    }
    mc->n_src_query++;
    if (expert < 0 || expert >= (int32_t) ls.expert_slot.size() || ls.expert_slot[expert] < 0) {
        return false;
    }
    mc->n_src_hit++;
    *data   = (const uint8_t *) c->data + (size_t) ls.expert_slot[expert]*c->nb[2];
    *buffer = c->buffer;
    return true;
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t prefetch_slots, int32_t window,
                          int32_t predict, int32_t predict_train) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots == 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
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
        const int32_t jit_pool = [] { const char * e = getenv("LLAMA_MOE_CACHE_JIT_POOL"); return e ? std::max(0, atoi(e)) : 0; }();
        // LLAMA_MOE_CACHE_JIT_POOL_ALL=1: every layer gets a pool (JIT never evicts the main cache)
        const bool jit_pool_all = [] { const char * e = getenv("LLAMA_MOE_CACHE_JIT_POOL_ALL"); return e && atoi(e) != 0; }();
        // n_slots < 0: fill each device's free VRAM (called after KV/compute buffers
        // exist), leaving a margin for the compute graph growing by the cache chain.
        // ponytail: one slot count per device, uniform over that device's layers.
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
            // default 384 MiB (measured peak growth after load: 54 to 170 MiB per GPU); the engine measures the growth of this model
            // per batch token once and sets margin_mb (LLAMA_MOE_CACHE_MARGIN_MB / --moe margin_mb=N override)
            size_t margin = (size_t) (knobs().margin_mb > 0 ? knobs().margin_mb : 384) * 1024 * 1024;
            // --prefetch-experts-slots: the scheduler lazily allocates N full expert
            // tensors on the device big batches are offloaded to
            if (prefetch_slots >= 2) {
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
            // the learned predictors (fp16 [n_embd, n_expert * ahead] per layer, allocated after the cache) need room too
            {
                const char * a = getenv("LLAMA_MOE_CACHE_PREDICT_AHEAD");
                const int64_t ahead = std::max(0, std::min(8, a ? atoi(a) : 2));
                if (predict > 0 && ahead > 0 && !cands.empty()) {
                    const int64_t n_embd = cands[0].l->ffn_up_exps->ne[0];
                    const char * st = getenv("LLAMA_MOE_CACHE_PREDICT_STRIDE");
                    const size_t n_src = (cands.size() + std::max(1, st ? atoi(st) : 1) - 1) / std::max(1, st ? atoi(st) : 1);
                    // fp16 master + Q8_0 prediction copy (34 bytes per 32 values)
                    margin += (size_t) (n_src * n_embd * n_expert * ahead * (sizeof(ggml_fp16_t) + 34.0/32)) + 64u * 1024 * 1024;
                }
            }
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

        {
            const char * ng = getenv("LLAMA_MOE_CACHE_NEG_IDS");
            g_neg_ids = !(ng && ng[0] == '0');
            for (auto & ls : mc->layers) {
                for (const ggml_tensor * t : { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src }) {
                    g_neg_ids = g_neg_ids && ggml_is_quantized(t->type);
                }
            }
        }
        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        // with router prediction on, the top LLAMA_MOE_CACHE_STREAM_SLOTS (default 4) slots of each layer take the
        // predicted uploads, so a wrong guess never evicts a cached expert
        int32_t n_stream = 0;
        {
            const char * ss = getenv("LLAMA_MOE_CACHE_STREAM_SLOTS");
            n_stream = predict > 0 ? std::max(0, ss ? atoi(ss) : 4) : 0;
        }
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            const int32_t ns = ls.pub.n_slots;
            ls.n_cache = ns > 2*n_stream ? ns - n_stream : ns;
            ls.is_stream.assign(ns, 0);
            ls.stream_score.assign(ns, 0.0f);
            ls.stream_step.assign(ns, 0);
            ls.stream_slots.clear();
            for (int32_t sl = ls.n_cache; sl < ns; ++sl) {
                ls.is_stream[sl] = 1;
                ls.stream_slots.push_back(sl);
            }
            ls.slot_expert.assign(ns, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ns, 0);
            ls.slot_in_flight.assign(ns, false);
            ls.expert_count.assign(n_expert, 0);
            ls.win_count.assign(n_expert, 0);
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                if (!file_location(model, srcs[k], &ls.src_fd[k], &ls.src_offs[k])) {
                    ls.src_fd[k] = -1;
                }
            }
            ls.glob_count.assign(n_expert, 0);
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
            if (g_neg_ids) {
                std::vector<int32_t> neg(n_expert, -1);
                ggml_backend_tensor_set(ls.pub.dev_table, neg.data(), 0, n_expert*sizeof(int32_t));
            } else {
                ggml_backend_tensor_set(ls.pub.dev_table, dummy.data(), 0, n_expert*sizeof(int32_t));
            }
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));
            if (ls.pub.jit_n > 0) {
                ls.pool_expert.assign(ls.pub.jit_n, -1);
                ls.pool_slot.assign(n_expert, -1);
                std::vector<int32_t> none(n_expert, g_neg_ids ? -1 : ls.pub.jit_n);
                ggml_backend_tensor_set(ls.pub.jit_table, none.data(), 0, n_expert*sizeof(int32_t));
            }

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_src[ls.pub.up_src]   = { (size_t) (&ls - mc->layers.data()), 0 };
            mc->by_src[ls.pub.gate_src] = { (size_t) (&ls - mc->layers.data()), 1 };
            mc->by_src[ls.pub.down_src] = { (size_t) (&ls - mc->layers.data()), 2 };
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
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
            const char * pr = getenv("LLAMA_MOE_CACHE_PREAD");
            const bool use_pread = (pr && pr[0] == '1') || any_repack;
            ggml_backend_buffer_type_t hbuft = (use_pread && any_file && gpu) ? ggml_backend_dev_host_buffer_type(gpu) : nullptr;
            const char * nt = getenv("LLAMA_MOE_CACHE_UPLOAD_THREADS");
            // one set of workers per upload link when layers sit behind both: a slow (x4) copy doesn't hold up the fast link's queue
            const char * lw = getenv("LLAMA_MOE_CACHE_LINK_WORKERS");
            const int n_links   = mc->n_links > 1 && !(lw && lw[0] == '0') ? mc->n_links : 1;
            const int n_workers = std::max(1, nt ? atoi(nt) : 1) * n_links;
            for (int w = 0; w < n_workers; ++w) {
                mc->worker_link.push_back(n_links > 1 ? w % n_links : -1);
                ggml_backend_buffer_t b = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, max_expert) : nullptr;
                mc->staging.push_back(b);
            }
            // weights loaded without mmap (--load-mode pin) sit in the GPU's pinned host buffer: direct DMA
            const bool src_pinned = gpu && ggml_backend_buffer_get_type(mc->layers[0].pub.up_src->buffer) == ggml_backend_dev_host_buffer_type(gpu);
            LLAMA_LOG_WARN("moe-cache: %d upload workers%s, %s\n", n_workers, n_links > 1 ? " (per link)" : "",
                    mc->staging[0] ? "pread -> pinned staging -> GPU" :
                    src_pinned     ? "from pinned host memory (direct DMA)" :
                                     "from pageable host memory (mmap; --load-mode pin: faster uploads)");
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
                        auto it = mc->todo.begin();
                        while (it != mc->todo.end() && link >= 0 && mc->layers[it->layer_idx].link != link) {
                            ++it;
                        }
                        return it;
                    };
                    mc->wcv.wait(lk, [mc, &mine]() { return mc->stop || mine() != mc->todo.end(); });
                    if (mc->stop) {
                        return;
                    }
                    auto it = mine();
                    j = *it;
                    mc->todo.erase(it);
                    mc->in_flight++;
                    mc->in_flight_stream += j.stream;
                }
                auto & ls = mc->layers[j.layer_idx];
                // ponytail: unlocked reads of gpu_started / n_steps; a race only delays or keeps one stale job
                if (j.stream && (j.step != mc->n_steps || mc->gpu_started >= (int64_t) j.layer_idx)) {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    j.skipped = true;
                    j.done    = true;
                    mc->done.push_back(j);
                    mc->in_flight--;
                    mc->in_flight_stream -= j.stream;
                    mc->n_stream_stale++;
                    mc->dcv.notify_all();
                    if (tr_on()) {
                        tr(TR_WORKER + (int) w, tr_fmt("drop L%zu e%d (layer started)", j.layer_idx, j.expert), ggml_time_us());
                    }
                    continue;
                }
                const int64_t t0 = ggml_time_us();
                const bool busy_start = mc->cpu_busy;
                ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
                const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
                int64_t t_read = 0;
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
                    mc->up_bytes += (int64_t) sz;
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
                    if (staging_ptr && ls.src_fd[k] >= 0 &&
                            pread_full(ls.src_fd[k], staging_ptr, sz, ls.src_offs[k] + (size_t) j.expert*sz)) {
                        t_read += ggml_time_us() - tr;
                        ggml_backend_tensor_set(dsts[k], staging_ptr, (size_t) j.slot*dsts[k]->nb[2], sz);
                    } else if (ggml_backend_buffer_is_host(srcs[k]->buffer)) {
                        // chunk: set, or the bytes this link moves in a quarter of the measured mean CPU phase (the budget is
                        // re-checked ~4 times per phase)
                        const size_t ch = knobs().gate < 3 || knobs().chunk_kb == 0 ? 0 :
                            knobs().chunk_kb > 0 ? (size_t) knobs().chunk_kb * 1024 :
                            (size_t) std::max(64.0*1024, (mc->gbs_link[ls.link] > 0 ? mc->gbs_link[ls.link] : 1.0) * 1e3 * mean_cpu_phase_us(mc) / 4);
                        if (ch == 0 || ch >= sz) {
                            upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                        } else {
                            // re-check the DDR budget per chunk: a CPU phase that started meanwhile makes the rest wait
                            const char * src = (const char *) srcs[k]->data + (size_t) j.expert*sz;
                            const size_t dst = (size_t) j.slot*dsts[k]->nb[2];
                            for (size_t off = 0; off < sz; off += ch) {
                                const int64_t tw = ggml_time_us();
                                while (mc->ddr_mbs > (int64_t) (knobs().ddr_gbs * 1e3) && ggml_time_us() - tw < (int64_t) (knobs().gate_max_us >= 0 ? knobs().gate_max_us : mean_layer_us(mc))) {
                                    std::this_thread::yield();
                                }
                                mc->gate_wait_us += ggml_time_us() - tw;
                                ggml_backend_tensor_set(dsts[k], src + off, dst + off, std::min(ch, sz - off));
                            }
                        }
                    } else {
                        LLAMA_LOG_ERROR("moe-cache: can't upload repacked '%s' without its file location\n", srcs[k]->name);
                    }
                }
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
                    mc->in_flight_stream -= j.stream;
                }
                mc->dcv.notify_all();
            }
        });

        mc->profile = profile_path(model);
        {
            // decisions of the last self-tune on this machine for this model (the profile file's footer): start from them (an explicit
            // LLAMA_MOE_CACHE_<NAME> wins) and check again after a while instead of exploring right away
            const char * te = getenv("LLAMA_MOE_CACHE_TUNED");
            std::string txt;
            if ((!te || atoi(te) != 0) && !mc->profile.empty()) { moe_state_get(mc->profile, "tuned.l" + std::to_string(mc->n_links), txt); }
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
        if (const char * c = getenv("LLAMA_MOE_CACHE_CTL")) {
            mc->ctl = c;
        }
        if (const char * t = getenv("LLAMA_MOE_CACHE_TRACE")) {
            g_tr.prefix = t;
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
            const char * e = getenv("LLAMA_MOE_CACHE_DROP");
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
        if (const size_t n = profile_preload(mc, model)) {
            LLAMA_LOG_INFO("moe-cache: preloading %zu experts (usage profile)\n", n);
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
            const char * x = getenv("LLAMA_MOE_CACHE_PREDICT_MAX");
            mc->pred_m   = predict;
            mc->pred_max = x ? atoi(x) : 2;
            {
                const char * ra = getenv("LLAMA_MOE_CACHE_PREDICT_RA");
                ggml_backend_dev_t gpu = mc->bufs.empty() ? nullptr : ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
                const auto * src = mc->layers.empty() ? nullptr : mc->layers[0].pub.up_src;
                mc->pred_ra = ra && atoi(ra) != 0 && src && ggml_backend_buffer_is_host(src->buffer) &&
                    !(gpu && ggml_backend_buffer_get_type(src->buffer) == ggml_backend_dev_host_buffer_type(gpu));
            }
            if (const char * g = getenv("LLAMA_MOE_CACHE_PREDICT_MARGIN")) {
                mc->pred_margin = (float) atof(g);
            }
            {
                const char * a  = getenv("LLAMA_MOE_CACHE_PREDICT_AHEAD");
                const char * mu = getenv("LLAMA_MOE_CACHE_PREDICT_MU");
                mc->pred_ahead = std::max(0, std::min(8, a ? atoi(a) : 2));
                mc->pred_train = predict_train;
                if (mu) { mc->pred_mu = (float) atof(mu); }
                const char * pf = getenv("LLAMA_MOE_CACHE_PREDICT_FILE");
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
            const char * pst = getenv("LLAMA_MOE_CACHE_PREDICT_STRIDE");
            const size_t stride = (size_t) std::max(1, pst ? atoi(pst) : 1);
            const char * pq8 = getenv("LLAMA_MOE_CACHE_PREDICT_Q8");
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
                    if (const char * c = getenv("LLAMA_MOE_L3PF_CPUS")) {
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
            const char * e = getenv("LLAMA_MOE_CACHE_PREFILL_D2D");
            if (!e || e[0] != '0') {
                ggml_backend_set_moe_src_callback(moe_src_cb, mc);
                ggml_backend_set_moe_fill_callback(moe_fill_cb, mc);
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
void llama_moe_cache_jit(const llama_moe_cache_layer * pub, const ggml_tensor * ids) {
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
            mc->up_bytes += (int64_t) S;
            ++n_up;
        }
        mc->n_pool_up += n_up;
        mc->n_jit += n_up;
        mc->n_jit_layers += n_up > 0;
        return;
    }
    for (int i = 0; i < kb; ++i) {
        const int32_t id = miss[i];
        // slot: an empty one, else the lowest-scored cached expert this token doesn't use
        int32_t slot = -1;
        double best = 1e300;
        for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
            if (ls.slot_in_flight[s] || ls.is_stream[s]) {
                continue;
            }
            const int32_t x = ls.slot_expert[s];
            if (x < 0) { slot = s; break; }
            if (pinned(mc, ls, x) || std::find(cur.begin(), cur.end(), x) != cur.end()) {
                continue;
            }
            const double c = score(ls, x);
            if (c < best) { best = c; slot = s; }
        }
        if (slot < 0) {
            break;
        }
        const int32_t victim = ls.slot_expert[slot];
        if (victim >= 0) {
            // JIT=1: admitted only as the step's swap would be (no churn): hotter than the victim by the link's pay-back
            // margin; JIT=2: any miss may evict the lowest-scored expert this token doesn't use
            if (knobs().jit < 2 && ((knobs().big != 0 && ls.glob_count[id] < ls.glob_count[victim]) ||
                    score(ls, id) < score(ls, victim) + mc->link_margin[ls.link])) {
                continue;
            }
            ls.expert_slot[victim] = -1;
            set_table_entry(ls.pub, victim, ls.pub.n_slots);
            note_evict(mc, ls, victim);
            ls.cached_since[victim] = 0;
            if (ls.dropped[victim]) { page_hint(mc, ls, victim, false); ls.dropped[victim] = false; }
        }
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
        mc->up_bytes  += (int64_t) S;
        ++n_up;
    }
    mc->n_jit += n_up;
    mc->n_jit_layers += n_up > 0;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
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
// 2 SE are the test's parameters, not tuning thresholds. On by default; LLAMA_MOE_CACHE_SELF_TUNE=0 / ctl SELF_TUNE=0: off.
// multi-GPU prefill, measured (LLAMA_PREFILL_SPLIT unset): alpha 1 = experts over all GPUs by link bandwidth, 0 = the
// fastest GPU only. Full prompt batches alternate between two alphas, each batch's tokens/s is one sample; a pair is
// decided when the faster mean beats the other by 2 SE (or after MAX_S batches each). Search: 1 vs 0; if all GPUs lose,
// step down: 0.5 vs 0, then 0.75 vs 0.5 or 0.25 vs 0. The result is held for a rest period (doubling while re-checks
// confirm it). Works for any GPU count and link widths; a box where splitting doesn't pay ends at 0.
static void prefill_split_tune(moe_cache * mc, double batch_us) {
    static const char * env = getenv("LLAMA_PREFILL_SPLIT");
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

float llama_moe_cache_prefill_alpha() {
    return g_split_share;
}

double llama_moe_cache_link_gbs(ggml_backend_dev_t dev) {
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
void llama_moe_cache_prefill_decide(const std::vector<ggml_backend_dev_t> & gpus) {
    if (const char * e = getenv("LLAMA_PREFILL_SPLIT")) {
        g_split_share   = (float) atof(e);
        g_split_enabled = g_split_share > 0;
    }
    if (gpus.size() < 2) {
        return;
    }
    const size_t n = 64u << 20;
    double sum = 0, best = 0, worst = 1e30;
    std::string log;
    for (ggml_backend_dev_t d : gpus) {
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
        sum += gbs; best = std::max(best, gbs); worst = std::min(worst, gbs);
        log += tr_fmt(" %s %.1f", ggml_backend_dev_name(d), gbs);
    }
    if (getenv("LLAMA_PREFILL_SPLIT")) {
        return;
    }
    const double speedup = best > 0 ? sum / best : 1.0;
    // only GPUs on equally fast links (x16 with x16): a x4 or chipset-shared link (PC1 IQ3_S: 6.2 vs 25.5 GB/s, speedup 1.24x) never splits
    g_split_enabled = speedup >= 1.25 && worst >= 0.7 * best;
    g_split_share   = g_split_enabled ? 1.0f : 0.0f;
    LLAMA_LOG_WARN("moe-cache: prefill links (GB/s):%s -> split's transfer speedup %.2fx: %s\n", log.c_str(), speedup,
        g_split_enabled ? "reserved, alpha measured on prompt batches" : "one GPU (the split's compute buffers would cost more cache)");
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
        { "STREAM_M", &knobs_t::stream_m,  { 4, 6, 8, 12 } },
        { "AUTO",     &knobs_t::auto_tune, { 0, 1 } },
        { "TBP",      &knobs_t::tbp,       { 0, 2 } },
    };
    static const std::vector<tunable> T_swap = {
        { "GATE",     &knobs_t::gate,      { 0, 3 } },
        { "SWAP_FRAC", &knobs_t::swap_frac, { 0.25, 0.5 }, true }, // the upload budget: more uploads need DDR headroom, which differs per machine
        { "WAIT",     &knobs_t::wait,      { 0, 1 } },
        { "BIG",      &knobs_t::big,       { 0, 1 } },
    };
    static const std::vector<tunable> T = [&] {
        std::vector<tunable> t;
        // whatever the user set (LLAMA_MOE_CACHE_<NAME> in the environment) is theirs: never tuned. Deterministic mode fixes them all.
        const char * dm = getenv("LLAMA_MOE_CACHE_DETERMINISTIC");
        const bool det_mode = dm && atoi(dm) != 0;
        auto add = [&](const tunable & x) {
            if (!det_mode && !g_autotune_off && !user_knobs().count(x.name)) { t.push_back(x); }
        };
        if (mc->pred_m > 0) {
            add({ "PREDICT", &knobs_t::predict, { 0, 1 } }); // first: the streaming knobs are tuned with it on
        }
        if (mc->pred_m > 0 && knobs().stream > 0) { for (const auto & x : T_stream) { add(x); } }
        // the swap margin: 0 = any hotter expert may enter, -2 = pay-back margin from measured upload / CPU time. Which one wins depends
        // on the machine (slow link, RAM headroom), so it is tuned live; a state-carrying knob, hence the long slices
        add({ "MARGIN", &knobs_t::margin, { 0, -2 }, true });
        for (const auto & x : T_swap) { add(x); }
        return t;
    }();
    if (T.empty()) { return; } // -at off, every knob set by the user or deterministic mode: nothing to tune
    const size_t MIN_S = 3, MAX_S = 8;
    // a cycle costs ~2k tokens per state-carrying knob (half of them on the worse candidate): rest long enough to keep that under ~10%
    const uint64_t REST_MIN = 16384, REST_MAX = 131072;
    auto begin = [&](int k) {
        mc->lv_k = k; mc->lv_c = (int) (mc->lv_cycle % T[k].vals.size());
        mc->lv_slice_len = T[k].slow ? 320 : 36; mc->lv_warm = T[k].slow ? 96 : 4;
        mc->lv_samples.assign(T[k].vals.size(), {});
        mc->lv_slice_n = 0; mc->lv_slice_sum = 0;
        knobs().*T[k].f = T[k].vals[mc->lv_c];
        apply_predict(mc);
    };
    if (mc->lv_k < 0) {
        if (mc->n_steps >= mc->lv_rest_until) { begin(0); }
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
    if (n_min >= MIN_S) {
        std::vector<double> mean(nv), se(nv);
        for (size_t c = 0; c < nv; ++c) {
            const auto & v = mc->lv_samples[c];
            double m = 0, q = 0;
            for (double x : v) { m += x; }
            m /= v.size();
            for (double x : v) { q += (x - m)*(x - m); }
            mean[c] = m; se[c] = sqrt(q / (v.size() - 1) / v.size());
        }
        size_t b = 0;
        for (size_t c = 1; c < nv; ++c) { if (mean[c] < mean[b]) { b = c; } }
        bool sure = true;
        for (size_t c = 0; c < nv; ++c) {
            if (c != b && mean[c] - 2*se[c] <= mean[b] + 2*se[b]) { sure = false; }
        }
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

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    struct flush_at_exit { ~flush_at_exit() { flush_tables(); } } flush_guard; // table changes reach the GPU before the next graph

    // LLAMA_MOE_CACHE_DETERMINISTIC=1 (benchmarks): publish exactly the uploads
    // scheduled by the previous step, in a fixed order, with a fixed swap budget
    // and margin, so a run no longer depends on upload timing and repeats its
    // output. Uploads still overlap the token's compute; the wait is only for
    // the ones not finished by then.
    static const bool det = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_DETERMINISTIC");
        return e && atoi(e) != 0;
    }();
    // wait for and publish the previous step's uploads (default; LLAMA_MOE_CACHE_WAIT=0
    // disables): a cache that is current every step beats the time the wait costs,
    // fixed-text decode +16%, hit rate 66% -> 77%
    const bool wait_publish = det || knobs().wait != 0; // LLAMA_MOE_CACHE_WAIT / ctl WAIT

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
                    std::none_of(mc->todo.begin(), mc->todo.end(), [](const upload_job & j) { return !j.stream; });
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
    // LLAMA_MOE_CACHE_SWAP_FRAC of the token time, minus what is still queued
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
            budget_total = (int) (frac*step_us*mc->workers.size()/mc->upload_us) - (int) mc->todo.size();
        }
        // static budget: LLAMA_MOE_CACHE_BUDGET swaps per step (deterministic mode: default 8)
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
    // static pay-back margin: LLAMA_MOE_CACHE_MARGIN, default 0 (deterministic mode: 4). The timing margin above (-2)
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
            const char * e = getenv("LLAMA_MOE_CACHE_DROP_STAY");
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
            const char * e = getenv("LLAMA_MOE_CACHE_PREDICT_MISS");
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
        // swaps cost PCIe bandwidth and latency: cap evictions per layer per step,
        // and none at all while earlier uploads are still queued
        int & budget = budget_total;
        for (auto it = ls.pending.begin(); it != ls.pending.end(); ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.queued[id] || score(ls, id) <= 0) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the least-called cached expert
            int32_t slot = -1;
            double best = 1e300;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s] || ls.is_stream[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (pinned(mc, ls, ls.slot_expert[s])) {
                    continue;
                }
                const double c = score(ls, ls.slot_expert[s]);
                if (c < best) { best = c; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                if (knobs().big != 0 && ls.glob_count[id] < ls.glob_count[victim]) {
                    continue; // lifetime regular stays; a later candidate may still fit
                }
                if (score(ls, id) < score(ls, victim) + link_margin[ls.link] || budget-- <= 0) {
                    break; // candidates are sorted: nothing hotter than what's cached
                }
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, ls.pub.n_slots);
                note_evict(mc, ls, victim);
                ls.cached_since[victim] = 0;
                if (ls.dropped[victim]) { page_hint(mc, ls, victim, false); ls.dropped[victim] = false; }
            }
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
        const char * h = getenv("LLAMA_MOE_CACHE_HALVE_EVERY");
        return (uint64_t) std::max(1, h ? atoi(h) : 64);
    }();
    if (mc->n_steps % halve_every == 0) {
        for (auto & ls : mc->layers) {
            for (auto & c : ls.expert_count) { c >>= 1; }
        }
    }

    static const bool stats = getenv("LLAMA_MOE_CACHE_STATS") != nullptr;
    if (stats && mc->n_steps % 64 == 0) {
        static uint64_t ph = 0, pm = 0;
        uint64_t h = 0, m = 0, filled = 0, inflight = 0, total = 0;
        for (auto & ls : mc->layers) {
            h += ls.n_hit; m += ls.n_miss; total += ls.pub.n_slots;
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
        for (char & c : name) { c = (char) toupper((unsigned char) c); }
        if (name == "AUTOTUNE") {
            g_autotune_off = atof(kv.c_str() + eq + 1) == 0;
        } else if (knob_set(knobs(), name, atof(kv.c_str() + eq + 1))) {
            user_knobs().insert(name);
        } else {
            // every other setting is an environment variable of the engine (POLICY, PREDICT_AHEAD, ...): --moe name=value sets LLAMA_MOE_CACHE_NAME, and wins over the environment
            for (char & c : name) { if (c == '-') { c = '_'; } }
            const std::string var = "LLAMA_MOE_CACHE_" + name, val = kv.substr(eq + 1);
#ifdef _WIN32
            _putenv_s(var.c_str(), val.c_str());
#else
            setenv(var.c_str(), val.c_str(), 1);
#endif
            LLAMA_LOG_INFO("moe-cache: --moe %s=%s sets %s\n", kv.substr(0, eq).c_str(), val.c_str(), var.c_str());
        }
    }
}

bool llama_moe_cache_active() {
    return g_cache != nullptr;
}

bool llama_moe_cache_get_info(struct llama_tuning_info * info) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return false;
    }
    info->moe_hits = info->moe_misses = 0;
    for (auto & ls : mc->layers) { info->moe_hits += ls.n_hit; info->moe_misses += ls.n_miss; }
    info->moe_uploads = mc->n_uploads;
    info->moe_evictions = mc->n_evictions; info->moe_up_bytes = (uint64_t) std::max<int64_t>(0, (int64_t) mc->up_bytes);
    info->moe_layers  = (int32_t) mc->layers.size();
    info->moe_slots_min = INT32_MAX; info->moe_slots_max = 0;
    for (auto & ls : mc->layers) {
        info->moe_slots_min = std::min(info->moe_slots_min, ls.pub.n_slots);
        info->moe_slots_max = std::max(info->moe_slots_max, ls.pub.n_slots);
    }
    info->margin = mc->last_margin; // the value in use (the tuner's pick, a fixed MARGIN, or the timing rule's)
    info->gate = (int32_t) knobs().gate; info->wait = (int32_t) knobs().wait; info->big = (int32_t) knobs().big;
    info->predict = (int32_t) knobs().predict; info->self_tune = (int32_t) knobs().self_tune;
    return true;
}

int64_t llama_moe_cache_max_batch() {
    static const int64_t v = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_MAX_BATCH");
        return (int64_t) (e ? atoi(e) : 31);
    }();
    return v;
}

void llama_moe_cache_free() {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    ggml_set_moe_phase_callback(nullptr, nullptr);
    ggml_backend_set_moe_src_callback(nullptr, nullptr);
    ggml_backend_set_moe_fill_callback(nullptr, nullptr);
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
    if (mc->n_pred_up) {
        LLAMA_LOG_WARN("moe-cache: router prediction: %" PRIu64 " uploads, %" PRIu64 " published before their layer, %" PRIu64 " used, %" PRIu64 " too late to start, %" PRIu64 " dropped in the queue (layer started), %" PRIu64 " promoted into the cache, %" PRIu64 " token-boundary prefetches\n",
                mc->n_pred_up, mc->n_pred_pub, mc->n_pred_used, mc->n_pred_late, mc->n_stream_stale, mc->n_promoted, mc->n_tbp);
    }
    for (auto * b : mc->staging) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * b : mc->pbufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->pctxs) { ggml_free(c); }
    for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->ctxs) { ggml_free(c); }
    delete mc;
    g_cache     = nullptr;
    g_init_done = false;
}

bool llama_moe_cache_pred_train_now() {
    return g_pred_train_now;
}

bool llama_moe_cache_graph_reusable() {
    return g_pred_train_now == g_pred_train_built && g_pred_epoch == g_pred_epoch_built && g_split_epoch == g_split_epoch_built;
}

int32_t llama_moe_cache_cpu_sat_threads() {
    return g_cache ? g_cache->cpu_sat_threads : 0;
}

void llama_moe_cache_graph_built() {
    g_pred_train_built = g_pred_train_now;
    g_pred_epoch_built = g_pred_epoch;
    g_split_epoch_built = g_split_epoch;
}

namespace {
bool pinned(const moe_cache * mc, const layer_state & ls, int32_t e) {
    const uint64_t stay = (uint64_t) knobs().slow_stay;
    return ls.sticky[e] || (ls.slow && ls.cached_since[e] && mc->n_steps + 1 - ls.cached_since[e] < stay);
}
} // namespace
