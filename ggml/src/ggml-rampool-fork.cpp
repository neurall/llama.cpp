// The RAM pool (--moe rampool=1): expert slices in pinned host memory chosen by heat, read straight from the SSD, no mmap read and no page cache.
//
// Why. A model that does not fit pinned RAM used to be pinned in tensor order: the first tensors pinned, the rest mapped. So cold experts of the first layers held RAM while hot
// experts of the last layers came from the SSD. Here the pool holds the experts expected back soonest (the caller's heat callback: the revisit ring of each expert), the GPU
// cache holds the hottest, and the rest stays on the SSD (fork/docs/rampool.md).
//
// What it is. A store of small equal fragments, one per expert slice (up, gate, down each their own): a class per (slice bytes, type), blocks of a class in pinned chunks, a table
// element -> block, and "free" is a change of owner. An element's record is its tensor's file handle and offset plus its index: a block can always be dropped, the bytes are on the SSD.
//
// Rules.
//   - One writer function, rp_read, and it writes only inside the block it owns.
//   - Check and load BEFORE the kernel (ggml_rampool_need_fork), never on a memory access.
//   - A block is taken from an element only when nobody can still read it: not asked by this graph or the one before it (a copy of the previous graph can still run), no holds.
//   - Eviction: of 64 sampled evictable blocks of the class the one expected back last, and every demanded element gets one. Refusing the newcomers expected back later than
//     that block (until 2026-11-08) refused every expert never seen before, and its caller read it from the mapping: GLM 3.5 12k, 4899 of 12871 cold misses per request,
//     16 GiB of page cache next to a 101 GiB pool, 1000 major faults per second and swapping through the decode. The read into the pool costs the same SSD bytes and leaves nothing behind.
//   - Elements of a class lie behind each other as the file has them after the fill, so one DMA serves a run. A matmul that reads past a slice needs the real next expert's
//     bytes there (llama-moecache-fork.cpp, expert_past): such tensors are not given to the pool.
#include "ggml-rampool-fork.h"
#include "ggml-impl.h"
#include "ggml-read-fork.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstddef>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
#endif

namespace {

enum : uint8_t { RP_OUT = 0, RP_READING = 1, RP_IN = 2 };

struct rp_rec {
    int32_t  slot    = -1;      // block of its class, -1: out
    uint32_t graph   = 0;       // the graph that last asked for it
    uint16_t holds   = 0;       // readers outside a graph (ggml_rampool_take_fork)
    uint8_t  state   = RP_OUT;
    bool     claimed = false;   // a reader has it (the queue can hold a second job for it)
    bool     prob    = false;   // on probation: it came in without passing the admission rule (rp_take_block), asked again in a later graph it stays (rp_promote)
    bool     ahead   = false;   // read ahead for a predicted later layer (ggml_rampool_ahead_fork), not asked for yet
};

struct rp_tensor {
    const void * key  = nullptr;
    int          fd   = -1, dfd = -1;   // the file; the same file opened O_DIRECT, -1: not possible here
    size_t       off  = 0, esz = 0;     // where element 0 lies, element bytes
    size_t       pad  = 0;              // bytes a kernel reads past an element: its block is esz + pad and carries them (the next element's; zeros behind the last)
    int64_t      n    = 0;
    int          cls  = 0;
    int32_t      idx  = 0;
    std::vector<rp_rec> rec;
    std::unique_ptr<std::atomic<const char *>[]> ptr;   // the table the readers use without the lock (see ggml_rampool_table_fork)
};

struct rp_owner { int32_t t = -1, e = -1; };

struct rp_class {
    size_t   size  = 0;
    int32_t  kind  = 0;
    uint64_t elems = 0;                 // elements of this class in the model
    uint32_t n_slots = 0, per_chunk = 0;   // n_slots: blocks that exist (block numbers behind a short chunk do not)
    std::vector<char *>   chunks;       // capacity reserved at open for every chunk the class can get: a reader's at() outside the lock never sees it move (ggml_rampool_grow_fork)
    std::vector<uint32_t> chunk_n;      // blocks of each chunk: per_chunk, a short one less
    std::vector<rp_owner> owner;        // block -> element
    std::vector<uint32_t> free;         // blocks without an owner (a stack)
    uint32_t sample = 0;                // where the next eviction sample starts
    uint64_t cold   = 0;                // bytes this class read from the file since the last resize: where a new chunk pays most
    std::deque<uint32_t> prob;          // probation blocks, oldest first: a declined newcomer takes the oldest one nobody reads
    std::vector<uint8_t> left;          // block -> 1 once an element left it (evicted): the blocks still 0 and owned hold what came in first and never left (ggml_rampool_map_fork)
    std::vector<int64_t> t_in;          // block -> when its element came (rp_now_us): the live map's churn split (ggml_rampool_live_fork); grown on first use
    char * at(uint32_t s) const { return chunks[s / per_chunk] + (size_t) (s % per_chunk) * size; }
};

struct rp_job { rp_tensor * t; int64_t e; };

struct rp_pool {
    std::vector<std::unique_ptr<rp_tensor>>      tensors;
    std::unordered_map<const void *, rp_tensor *> by_key;
    std::vector<rp_class> cls;
    ggml_rampool_heat_t heat = nullptr;
    void * ud = nullptr;
    std::map<int, int> dfd_of;          // file handle -> its O_DIRECT handle
    int    n_threads = 0;

    std::mutex              mu;         // tables, queue and records; held for microseconds
    std::condition_variable work_cv, done_cv;
    std::deque<rp_job>      demand;     // what callers wait for
    std::deque<rp_job>      pf;         // reads ahead for predicted later layers: a reader takes one only when no demand waits
    std::vector<std::thread> readers;
    bool                    stop = false;
    std::atomic<uint32_t>   graph{ 2 };
    std::atomic<size_t>     memal{ 4 };  // memory alignment O_DIRECT takes here: 4 bytes on NVMe (queue/dma_alignment); an EINVAL raises it, the reader then goes through its bounce buffer
    size_t                  rq = 0;      // bytes of one read request (the device's own limit, see rp_request_bytes)
    bool                    evict = false;   // ggml_rampool_evict_fork: blocks change owner during the run; off: the fill's set stays, a slice out of it is read the old way

    std::atomic<uint64_t> st_asked{0}, st_hit{0}, st_cold{0}, st_refused{0}, st_evicted{0}, st_cold_b{0}, st_wait_us{0}, st_pp_hit{0}, st_pp_miss{0};
    std::atomic<uint64_t> wait_total_us{0};   // st_wait_us without the per-request reset (ggml_rampool_wait_us_fork)
    std::atomic<uint64_t> rd_kept_b{0}, rd_hit_b{0}, rd_cold_b{0}, rd_ppmiss_b{0};   // bytes served since open, never reset (ggml_rampool_live_fork): hits on kept blocks, all hits, file reads, prompt-pass misses left to the SSD path
    bool (*admit)(const void * key, int64_t e, double victim, void * ud) = nullptr;   // ggml_rampool_admit_fork: may element e evict a block expected back at `victim`
    std::atomic<uint64_t> st_admitted{0}, st_declined{0};
    std::atomic<uint64_t> st_ahead{0}, st_ahead_used{0};   // reads ahead started; of those, asked for before their block left
    uint64_t pool_bytes = 0;
};

rp_pool * g_rp = nullptr;

// file offsets of an O_DIRECT read in whole units of this: 4096 is a multiple of every sector size in use
constexpr size_t RP_SECTOR = 4096;

int64_t rp_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

rp_tensor * rp_find(const void * key) {
    if (g_rp == nullptr) { return nullptr; }
    const auto it = g_rp->by_key.find(key);
    return it == g_rp->by_key.end() ? nullptr : it->second;
}

// The size of one read request: what the device takes in one command (queue/max_sectors_kb). PC1, a Gen4 x4 SSD, 32 threads: 512 KiB requests 5.2 GB/s, 2.4 MB requests (an expert slice in
// one pread, split by the kernel anyway) 4.0 GB/s at any thread count. 0: not known, one request per read.
size_t rp_request_bytes(int fd) {
#ifdef __linux__
    struct stat st;
    if (fstat(fd, &st) != 0) { return 0; }
    const std::string d = "/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev));
    for (const char * sub : { "/queue/max_sectors_kb", "/../queue/max_sectors_kb" }) {
        if (FILE * f = fopen((d + sub).c_str(), "r")) {
            unsigned long kb = 0;
            const int got = fscanf(f, "%lu", &kb);
            fclose(f);
            if (got == 1 && kb > 0) { return (size_t) kb << 10; }
        }
    }
#else
    (void) fd;
#endif
    return 0;
}

// THE WRITER: bytes [off, off + n) of the file into [dst, dst + n). Whole sectors go straight into dst; the sectors a slice shares with its neighbours (the slices of a tensor do not start on
// sector edges) go through the reader's bounce buffer so nothing is written outside dst. Requests of p->rq bytes each. Without O_DIRECT a plain pread.
bool rp_read(rp_pool * p, int dfd, int fd, char * dst, size_t n, uint64_t off, char * bounce, size_t bounce_n) {
    if (dfd < 0) {
        for (size_t d = 0; d < n; ) {
            const ptrdiff_t r = sr_pread1(fd, dst + d, n - d, off + d);
            if (r <= 0) { return false; }
            d += (size_t) r;
        }
        return true;
    }
    const uint64_t end = off + n;
    uint64_t pos = off / RP_SECTOR * RP_SECTOR;
    while (pos < end) {
        const uint64_t edge = (pos + RP_SECTOR);
        size_t want;
        if (pos < off || end < edge) {
            want = RP_SECTOR;   // a sector shared with a neighbour
        } else {
            want = (size_t) ((end - pos) / RP_SECTOR * RP_SECTOR);   // the whole-sector middle in one go: sr_read splits it into the drive's measured shape (256 KiB blocks, 32 in flight)
        }
        const uint64_t u0 = std::max(pos, off), u1 = std::min(pos + want, end);   // the bytes of this request that belong to dst
        char * d = dst + (size_t) (u0 - off);
        const size_t ma = p->memal.load(std::memory_order_relaxed);
        const bool whole = u0 == pos && u1 == pos + want;
        const bool direct = whole && ((uintptr_t) d & (ma - 1)) == 0;
        if (!direct) { want = std::min(want, std::min(bounce_n, p->rq > 0 ? p->rq / RP_SECTOR * RP_SECTOR : bounce_n)); }
        // the platform's async engine (ggml-read-fork.h: kernel AIO, io_uring or overlapped I/O, the stored rd_blk_kb / rd_qd shape) for a whole-sector span straight into the block
        // (user 2026-11-08: the pool reads through the one reader wrapper, at the measured shape, not 512 KiB preads); one read at a time only for the sectors shared with a neighbour
        ptrdiff_t r = direct && want > RP_SECTOR ? (sr_read(dfd, d, pos, want, want) ? (ptrdiff_t) want : (ptrdiff_t) -1) : sr_pread1(dfd, direct ? d : bounce, want, pos);
        if (r < 0 && errno == EINVAL && direct) {   // this file system or device wants more alignment than dma_alignment says: the bounce buffer from now on
            p->memal.store(RP_SECTOR, std::memory_order_relaxed);
            continue;
        }
        if (r <= 0) { return false; }
        const uint64_t got = (uint64_t) r;
        if (!direct) {
            const uint64_t c1 = std::min(pos + got, end);
            if (c1 > u0) { memcpy(d, bounce + (size_t) (u0 - pos), (size_t) (c1 - u0)); }
        }
        if (pos + got >= end) { return true; }
        if (got % RP_SECTOR != 0) { return false; }   // a short read off a sector edge: only the end of the file does that, and that is past `end`
        pos += got;
    }
    return true;
}

void rp_thread_setup() {
#ifdef __linux__
    // a reader sleeps on the SSD: any core will do. It starts with the affinity of the thread that made it, which can be one compute core
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const long nc = sysconf(_SC_NPROCESSORS_ONLN);
    for (long c = 0; c < nc && c < CPU_SETSIZE; ++c) { CPU_SET(c, &cs); }
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
}

char * rp_bounce_new(size_t n) {
#ifndef _WIN32
    void * b = nullptr;
    return posix_memalign(&b, RP_SECTOR, n) == 0 ? (char *) b : nullptr;
#else
    (void) n;
    return nullptr;
#endif
}

// out of a graph's reach: not read by this graph or the one before it, and no reader outside a graph. force (a CPU kernel's demand): only this graph's are kept, the one
// before it has been synchronised (the counter moves after the sync); DS 12k relcand: the 13.5k stream call asked for every block, so the tail's demand found none
bool rp_evictable(const rp_pool * p, const rp_rec & r, bool force = false) {
    return r.state == RP_IN && r.holds == 0 && r.graph + (force ? 0u : 1u) < p->graph.load(std::memory_order_relaxed);
}

// block s of class c leaves its element. The caller holds the lock.
void rp_drop_owner(rp_pool * p, rp_class & c, uint32_t s) {
    const rp_owner o = c.owner[s];
    rp_tensor & t = *p->tensors[o.t];
    t.ptr[o.e].store(nullptr, std::memory_order_release);
    t.rec[o.e].slot  = -1;
    t.rec[o.e].state = RP_OUT;
    t.rec[o.e].ahead = false;   // read ahead and never asked for: a wasted read
    c.owner[s] = rp_owner();
    if (s < c.left.size()) { c.left[s] = 1; }
    p->st_evicted++;
}

// A block for element ne of tensor nt (class c), -1: none. The caller holds the lock. A free block first. Else the victim: of 64 sampled evictable blocks the one expected
// back last. With an admission rule (ggml_rampool_admit_fork, the VRAM cache's: uses in the last 64 tokens, expected back before the victim) a newcomer that fails it is on
// probation: it takes the oldest probation block nobody reads, so a one-off miss pushes out another one-off miss, not a resident that keeps coming back. Without such a
// block a CPU demand (cpu: its kernel reads the bytes from the pool) still takes the victim; an upload is refused and reads the file into its own buffer.
// Kimi K3.1 12k decode, every miss admitted (2026-11-10): 834k reads and 834k evictions in 432 tokens (1708 GiB), the pool served 21% of the misses.
int32_t rp_take_block(rp_pool * p, rp_class & c, bool force, bool cpu, const rp_tensor & nt, int64_t ne, bool & prob) {
    prob = false;
    if (!c.free.empty()) {
        const uint32_t s = c.free.back();
        c.free.pop_back();
        return (int32_t) s;
    }
    if (!p->evict && !force) { return -1; }
    const uint32_t n = (uint32_t) c.owner.size();
    if (n == 0) { return -1; }
    int32_t best = -1;
    double  bh   = -1.0;
    int     seen = 0;
    for (uint32_t i = 0; i < n && seen < 64; ++i) {   // shortcut: a stride walk, not random; enough while the classes are big
        const uint32_t s = (c.sample + i * 7919u) % n;
        const rp_owner o = c.owner[s];
        if (o.t < 0) { continue; }
        rp_tensor & t = *p->tensors[o.t];
        if (!rp_evictable(p, t.rec[o.e], force)) { continue; }
        seen++;
        const double h = p->heat(t.key, o.e, p->ud);
        if (h > bh) { bh = h; best = (int32_t) s; }
    }
    c.sample = (c.sample + 1) % n;
    if (p->admit != nullptr) {
        if (best >= 0 && p->admit(nt.key, ne, bh, p->ud)) {
            p->st_admitted++;
        } else {
            prob = true;
            p->st_declined++;
            while (!c.prob.empty()) {
                const uint32_t s = c.prob.front();
                if (s >= c.owner.size() || c.owner[s].t < 0) { c.prob.pop_front(); continue; }   // gone since: a shrink, the sampled rule, a failed read
                const rp_owner o = c.owner[s];
                const rp_rec & r = p->tensors[o.t]->rec[o.e];
                if (!r.prob || r.slot != (int32_t) s) { c.prob.pop_front(); continue; }   // asked again since (promoted), or the block changed hands
                if (!rp_evictable(p, r, force)) { break; }   // the oldest is still read: the newer ones were asked later
                c.prob.pop_front();
                rp_drop_owner(p, c, s);
                return (int32_t) s;
            }
            if (!cpu) { return -1; }
        }
    }
    if (best < 0) { return -1; }
    rp_drop_owner(p, c, (uint32_t) best);
    return best;
}

// block s of class c got a new element now (the caller holds the lock)
static void rp_tin(rp_class & c, int32_t s) {
    if (c.t_in.size() < c.owner.size()) { c.t_in.resize(c.owner.size(), 0); }
    c.t_in[s] = rp_now_us();
}
// a hit on element r of tensor t: its bytes, and whether its block still holds what came in first (the hot set that never left). The caller holds the lock.
static void rp_hit_bytes(rp_pool * p, const rp_tensor & t, const rp_rec & r) {
    p->rd_hit_b += t.esz;
    const rp_class & c = p->cls[t.cls];
    if (r.slot >= 0 && (size_t) r.slot < c.left.size() && !c.left[r.slot]) { p->rd_kept_b += t.esz; }
}

// asked again in a later graph than the one that read it in: off probation (the block stays when probation is evicted). The caller holds the lock.
void rp_promote(rp_rec & r, uint32_t last_graph, uint32_t g) {
    if (r.prob && last_graph != g) { r.prob = false; }
}

// element e leaves for a block and a reader. The caller holds the lock. false: no block (refused). force: a block is taken from the fill's set even with eviction
// off (a CPU kernel's demand: the only other source is the mapping, ggml_rampool_demand_fork)
bool rp_start(rp_pool * p, rp_tensor & t, int64_t e, bool force = false, bool cpu = false, bool ahead = false) {
    if (!force && p->heat(t.key, e, p->ud) == HUGE_VAL) { return false; }   // a card holds it (the fill): a RAM copy pays nothing
    rp_class & c = p->cls[t.cls];
    rp_rec & r = t.rec[e];
    bool prob = false;
    const int32_t s = rp_take_block(p, c, force, cpu, t, e, prob);
    if (s < 0) { return false; }
    r.slot    = s;
    r.state   = RP_READING;
    r.claimed = false;
    r.prob    = prob;
    r.ahead   = ahead;
    c.owner[s] = { t.idx, (int32_t) e };
    rp_tin(c, s);
    if (prob) { c.prob.push_back((uint32_t) s); }
    (ahead ? p->pf : p->demand).push_back({ &t, e });
    return true;
}

// the read rp_start just queued is taken back: the lock has been held since, so no reader claimed it. The block goes to the free list (its evicted owner is gone: a rare loss).
void rp_unstart(rp_pool * p, rp_tensor & t, int64_t e) {
    rp_rec & r = t.rec[e];
    rp_class & c = p->cls[t.cls];
    auto & q = r.ahead ? p->pf : p->demand;
    for (auto it = q.rbegin(); it != q.rend(); ++it) {
        if (it->t == &t && it->e == e) { q.erase(std::next(it).base()); break; }
    }
    r.ahead = false;
    c.owner[r.slot] = rp_owner();
    c.free.push_back((uint32_t) r.slot);
    r.slot  = -1;
    r.state = RP_OUT;
}

void rp_reader(rp_pool * p) {
    rp_thread_setup();
    const size_t bn = std::max<size_t>(p->rq, RP_SECTOR);
    char * bounce = rp_bounce_new(bn);
    for (;;) {
        rp_job j;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->work_cv.wait(lk, [p]() { return p->stop || !p->demand.empty() || !p->pf.empty(); });
            if (p->stop) { break; }
            auto & q = p->demand.empty() ? p->pf : p->demand;
            j = q.front();
            q.pop_front();
            rp_rec & r = j.t->rec[j.e];
            if (r.state != RP_READING || r.claimed) { continue; }
            r.claimed = true;
        }
        rp_tensor & t = *j.t;
        rp_rec & r = t.rec[j.e];
        rp_class & c = p->cls[t.cls];
        char * dst = c.at((uint32_t) r.slot);   // the block is ours: its owner changes only under RP_IN
        // O_DIRECT: a buffered read leaves a second copy in the page cache, which then pushes out the slices the pool does not hold (rampool3, GLM 3.5 warm decode 13.0 -> 6.5 t/s)
        const size_t pad = j.e + 1 < t.n ? t.pad : 0;   // the next element's first bytes come along; behind the last element zeros
        if (t.pad > pad) { memset(dst + t.esz, 0, t.pad); }
        const bool ok = bounce != nullptr && rp_read(p, t.dfd, t.fd, dst, t.esz + pad, t.off + (uint64_t) j.e * t.esz, bounce, bn);
        {
            std::lock_guard<std::mutex> lk(p->mu);
            if (ok) {
                t.ptr[j.e].store(dst, std::memory_order_release);
                r.state = RP_IN;
                p->st_cold_b += t.esz;
                p->rd_cold_b += t.esz;
                c.cold += t.esz;
            } else {
                GGML_LOG_WARN("rampool: a read of %zu bytes at %llu failed: %s\n", t.esz, (unsigned long long) (t.off + (uint64_t) j.e * t.esz), strerror(errno));
                c.owner[r.slot] = rp_owner();
                c.free.push_back((uint32_t) r.slot);
                r.slot  = -1;
                r.state = RP_OUT;
            }
        }
        p->done_cv.notify_all();
    }
    free(bounce);
}

} // namespace

extern "C" {

void ggml_rampool_add_fork(const void * key, int fd, size_t file_off, int64_t n, size_t esz, size_t pad, int32_t kind) {
    if (g_rp == nullptr) { g_rp = new rp_pool(); }
    rp_pool * p = g_rp;
    auto t = std::make_unique<rp_tensor>();
    t->key = key; t->fd = fd; t->off = file_off; t->n = n; t->esz = esz; t->pad = pad; t->idx = (int32_t) p->tensors.size();
    t->rec.resize((size_t) n);
    t->ptr.reset(new std::atomic<const char *>[(size_t) n]);
    for (int64_t e = 0; e < n; ++e) { t->ptr[e].store(nullptr, std::memory_order_relaxed); }
    int c = -1;
    for (size_t i = 0; i < p->cls.size(); ++i) { if (p->cls[i].size == esz + pad && p->cls[i].kind == kind) { c = (int) i; } }
    if (c < 0) {
        c = (int) p->cls.size();
        p->cls.emplace_back();
        p->cls.back().size = esz + pad;
        p->cls.back().kind = kind;
    }
    t->cls = c;
    p->cls[c].elems += (uint64_t) n;
    auto d = p->dfd_of.find(fd);
    if (d == p->dfd_of.end()) { d = p->dfd_of.emplace(fd, open_direct_fd(fd)).first; }
    t->dfd = d->second;
    if (p->rq == 0) { p->rq = rp_request_bytes(fd); }
    p->by_key[key] = t.get();
    p->tensors.push_back(std::move(t));
}

bool ggml_rampool_open_fork(size_t bytes, ggml_rampool_alloc_t alloc, ggml_rampool_heat_t heat, void * ud, int n_threads) {
    rp_pool * p = g_rp;
    if (p == nullptr) { return false; }
    p->heat = heat; p->ud = ud; p->n_threads = std::max(1, n_threads);
    uint64_t total = 0;
    for (const auto & c : p->cls) { total += c.elems * c.size; }
    const double share = total > bytes ? (double) bytes / (double) total : 1.0;   // every class the same share of its elements
    bool any = false;
    for (auto & c : p->cls) {
        uint64_t want = std::min<uint64_t>(c.elems, (uint64_t) ((double) c.elems * share));
        // chunks of at most 1 GiB: an allocation unit, not a limit (a big register stalled at 25.9 GiB, project_ssd_reader_pinned_slots); a full chunk with its extra block
        // fits the caller's 1 GiB prepinned piece, which the CUDA host allocator pins through its fast path (>= 1 GiB: populate + register, not cudaMallocHost)
        c.per_chunk = (uint32_t) std::max<uint64_t>(1, std::min<uint64_t>(std::max<uint64_t>(want, 1), (((size_t) 1 << 30) - c.size) / c.size));
        if (share < 1.0) { want -= std::min(want, (want + c.per_chunk - 1) / c.per_chunk); }   // the extra block of every chunk counts against the share: the pool never pins more than `bytes`
        c.chunks.reserve((size_t) ((c.elems + c.per_chunk - 1) / c.per_chunk) + 2);
        uint64_t got = 0;
        while (got < want) {
            const uint64_t k = std::min<uint64_t>(c.per_chunk, want - got);   // the last chunk only as big as what is left
            // one block more than it hands out: what is read past the last block stays inside the registered range
            char * m = (char *) alloc((size_t) k * c.size + c.size, ud);
            if (m == nullptr) { GGML_LOG_WARN("rampool: no more pinned memory at class %zu bytes, %llu of %llu blocks\n", c.size, (unsigned long long) got, (unsigned long long) want); break; }
            memset(m + (size_t) k * c.size, 0, c.size);
            c.chunks.push_back(m);
            c.chunk_n.push_back((uint32_t) k);
            got += k;
        }
        c.n_slots = (uint32_t) std::min(got, c.elems);
        c.owner.assign(c.n_slots, rp_owner());
        c.left.assign(c.n_slots, 0);
        c.free.clear();
        for (uint32_t s = c.n_slots; s > 0; --s) { c.free.push_back(s - 1); }
        p->pool_bytes += (uint64_t) c.n_slots * c.size;
        any |= c.n_slots > 0;
    }
    if (!any) {
        ggml_rampool_close_fork();
        return false;
    }
    return true;
}

// the owner's hook for a chunk whose last fill read landed (on that reader's thread): it pins the chunk then, under the rest of the fill. Declared where it is set
// (llama-moecache-fork.cpp), no header edit. A chunk no fill read touches gets no call.
static void (*g_rp_chunk_done)(void * chunk, void * ud) = nullptr;
extern "C" GGML_API void ggml_rampool_on_chunk_fork(void (*cb)(void * chunk, void * ud)) { g_rp_chunk_done = cb; }

// The first fill: the elements expected back soonest of each class (none a card holds), laid out in file order, read in jobs of one full queue of the async engine
// (sr_blk x sr_qd: 8 MiB) on n_threads threads, the jobs in spaced order (the even ones, then the odd ones: neighbours in the file are never in flight together,
// ggml-read-fork.h). Jobs of p->rq (512 KiB) before: 2 of the 32 blocks in flight per thread, DS 101 GiB at 3.36 GB/s against the loader's 4.9.
void ggml_rampool_fill_fork(void) {
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    const int64_t t0 = rp_now_us();
    uint64_t n_card = 0, n_card_in = 0;
    for (auto & c : p->cls) {
        struct el { double h; int32_t t, e; };
        std::vector<el> v;
        v.reserve((size_t) c.elems);
        for (auto & t : p->tensors) {
            if (&p->cls[t->cls] != &c) { continue; }
            for (int64_t e = 0; e < t->n; ++e) {
                const double h = p->heat(t->key, e, p->ud);
                n_card += h == HUGE_VAL;   // a card holds it: after all others, so blocks left over still hold it (prompt passes churn the cards' caches; without a RAM copy it came back from the SSD)
                v.push_back({ h, t->idx, (int32_t) e });
            }
        }
        std::stable_sort(v.begin(), v.end(), [](const el & a, const el & b) { return a.h < b.h; });
        if (v.size() > c.free.size()) { v.resize(c.free.size()); }
        for (const el & x : v) { n_card_in += x.h == HUGE_VAL; }
        std::sort(v.begin(), v.end(), [](const el & a, const el & b) { return a.t != b.t ? a.t < b.t : a.e < b.e; });
        // blocks in ascending order: file neighbours become pool neighbours
        std::sort(c.free.begin(), c.free.end(), std::greater<uint32_t>());
        for (const el & x : v) {
            const uint32_t s = c.free.back();
            c.free.pop_back();
            rp_tensor & t = *p->tensors[x.t];
            t.rec[x.e].slot  = (int32_t) s;
            t.rec[x.e].state = RP_READING;
            t.rec[x.e].claimed = true;
            c.owner[s] = { x.t, x.e };
            rp_tin(c, (int32_t) s);
        }
    }
    struct range { rp_tensor * t; char * dst; size_t n; uint64_t off; uint32_t ck; };
    std::vector<range> jobs;
    uint64_t bytes = 0;
    const size_t piece = std::max<size_t>(p->rq, sr_blk() * (size_t) sr_qd()) / RP_SECTOR * RP_SECTOR;
    std::vector<uint32_t> ck0(p->cls.size() + 1, 0);   // the first global chunk number of each class
    for (size_t i = 0; i < p->cls.size(); ++i) { ck0[i + 1] = ck0[i] + (uint32_t) p->cls[i].chunks.size(); }
    std::unique_ptr<std::atomic<uint32_t>[]> ck_left(new std::atomic<uint32_t>[ck0.back() + 1]);   // fill jobs not yet done per chunk
    for (uint32_t i = 0; i <= ck0.back(); ++i) { ck_left[i] = 0; }
    for (auto & tp : p->tensors) {
        rp_tensor & t = *tp;
        rp_class & c = p->cls[t.cls];
        for (int64_t e = 0; e < t.n; ) {
            if (t.rec[e].slot < 0) { ++e; continue; }
            int64_t b = e + 1;
            while (t.pad == 0 && b < t.n && t.rec[b].slot == t.rec[e].slot + (b - e) && (uint32_t) t.rec[b].slot / c.per_chunk == (uint32_t) t.rec[e].slot / c.per_chunk) { ++b; }   // padded blocks: one element per range (file neighbours are esz apart, their blocks esz + pad)
            const size_t pad = b < t.n ? t.pad : 0;   // the next element's first bytes come along; behind the last element zeros
            uint64_t off = t.off + (uint64_t) e * t.esz, end = t.off + (uint64_t) b * t.esz + pad;
            char * dst = c.at((uint32_t) t.rec[e].slot);
            const uint32_t ck = ck0[t.cls] + (uint32_t) t.rec[e].slot / c.per_chunk;   // a range never crosses a chunk (the run above stops at its edge)
            if (t.pad > pad) { memset(dst + t.esz, 0, t.pad); }
            while (off < end) {
                const uint64_t cut = std::min(end, (off / piece + 1) * piece);
                jobs.push_back({ &t, dst, (size_t) (cut - off), off, ck });
                ck_left[ck]++;
                dst += (size_t) (cut - off);
                off = cut;
            }
            bytes += end - (t.off + (uint64_t) e * t.esz);
            e = b;
        }
    }
    std::vector<range> spaced;
    spaced.reserve(jobs.size());
    // spaced in 16 segments of the file, each on its own: chunks complete through the whole fill and their pins run under it. One spacing over the whole
    // list completed every chunk in its second half, the pin threads fell behind (DS: 2.4 s of pins after the fill)
    const size_t seg = std::max<size_t>(2, (jobs.size() + 15) / 16);
    for (size_t s0 = 0; s0 < jobs.size(); s0 += seg) {
        const size_t s1 = std::min(jobs.size(), s0 + seg);
        for (size_t pass = 0; pass < 2; ++pass) { for (size_t i = s0 + pass; i < s1; i += 2) { spaced.push_back(jobs[i]); } }
    }
    std::atomic<size_t> next{ 0 };
    std::atomic<uint64_t> bad{ 0 };
    auto work = [&]() {
        rp_thread_setup();
        const size_t bn = std::max<size_t>(p->rq, RP_SECTOR);
        char * bounce = rp_bounce_new(bn);
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= spaced.size()) { break; }
            const range & r = spaced[i];
            if (bounce == nullptr || !rp_read(p, r.t->dfd, r.t->fd, r.dst, r.n, r.off, bounce, bn)) { bad++; }
            if (ck_left[r.ck].fetch_sub(1) == 1 && g_rp_chunk_done) {   // the chunk's last read: a failed one too (a failed fill drops the blocks below, the pin is harmless)
                const rp_class & c = p->cls[r.t->cls];
                g_rp_chunk_done(c.chunks[r.ck - ck0[r.t->cls]], p->ud);
            }
        }
        free(bounce);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < p->n_threads; ++i) { th.emplace_back(work); }
    for (auto & x : th) { x.join(); }
    if (bad.load() > 0) { GGML_LOG_ERROR("rampool: %llu reads of the fill failed, those blocks are dropped\n", (unsigned long long) bad.load()); }
    for (auto & tp : p->tensors) {
        rp_tensor & t = *tp;
        rp_class & c = p->cls[t.cls];
        for (int64_t e = 0; e < t.n; ++e) {
            rp_rec & r = t.rec[e];
            if (r.slot < 0) { continue; }
            r.claimed = false;
            if (bad.load() > 0) {   // shortcut: which range failed is not kept, so a failed fill drops every block (the pool then fills by demand)
                c.owner[r.slot] = rp_owner();
                c.free.push_back((uint32_t) r.slot);
                r.slot = -1; r.state = RP_OUT;
                continue;
            }
            r.state = RP_IN;
            t.ptr[e].store(c.at((uint32_t) r.slot), std::memory_order_release);
        }
    }
    const double s = std::max(1e-3, (rp_now_us() - t0) / 1e6);
    GGML_LOG_WARN("rampool: fill %.1f GiB read in %.1f s (%.2f GB/s, %d threads, the async engine at %zu KiB blocks x %d in flight, edge sectors in %zu KiB reads%s), %llu of %llu slices a card holds also in RAM (the blocks left after all others)\n", bytes / 1073741824.0, s, bytes / 1e9 / s,
            p->n_threads, sr_blk() >> 10, sr_qd(), p->rq >> 10, p->tensors.empty() || p->tensors[0]->dfd >= 0 ? "" : ", no O_DIRECT here", (unsigned long long) n_card_in, (unsigned long long) n_card);
    for (int i = 0; i < p->n_threads; ++i) { p->readers.emplace_back(rp_reader, p); }
}

void ggml_rampool_close_fork(void) {
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->stop = true;
    }
    p->work_cv.notify_all();
    for (auto & t : p->readers) { t.join(); }
#ifndef _WIN32
    for (const auto & kv : p->dfd_of) { if (kv.second >= 0) { close(kv.second); } }
#endif
    delete p;
    g_rp = nullptr;
}

const char * const * ggml_rampool_table_fork(const void * key) {
    rp_tensor * t = rp_find(key);
    return t ? reinterpret_cast<const char * const *>(t->ptr.get()) : nullptr;
}
static_assert(sizeof(std::atomic<const char *>) == sizeof(const char *), "the table is read as plain pointers");

void ggml_rampool_evict_fork(bool on) {
    if (g_rp) { g_rp->evict = on; }
}

// declared where they are used (llama-moecache-fork.cpp): no header edit
GGML_API void ggml_rampool_admit_fork(bool (*admit)(const void * key, int64_t e, double victim, void * ud)) {
    if (g_rp) { std::lock_guard<std::mutex> lk(g_rp->mu); g_rp->admit = admit; }
}
GGML_API void ggml_rampool_admit_stats_fork(uint64_t * admitted, uint64_t * declined, bool reset) {
    *admitted = g_rp ? g_rp->st_admitted.load() : 0;
    *declined = g_rp ? g_rp->st_declined.load() : 0;
    if (reset && g_rp) { g_rp->st_admitted = 0; g_rp->st_declined = 0; }
}

// Reads ahead for a later layer the predictor expects (--moe pred-top): queued behind every demand, no wait (user 2026-11-10: the SSD reads layer after layer, nonstop;
// the step waits only for what its own layer still lacks). A block only as the admission allows (else a probation block), never a resident's for a guess; an element
// starts in all of its tensors or in none, as in rp_need. Returns the elements refused (no block). upto: stop at the first refusal (the stream's reads ahead,
// one class: the rest would be refused too, and each refusal samples 64 victims under the lock while the scheduler thread waits; 30% slower passes, rdahead rd2)
GGML_API int ggml_rampool_ahead_fork(const void * const * keys, int nk, const int32_t * ids, int n, bool upto) {
    rp_pool * p = g_rp;
    if (p == nullptr || n <= 0 || !p->evict) { return 0; }
    rp_tensor * ts[8];
    int nt = 0;
    for (int k = 0; k < nk && nt < 8; ++k) { if (rp_tensor * t = rp_find(keys[k])) { ts[nt++] = t; } }
    if (nt == 0) { return 0; }
    const uint32_t g = p->graph.load(std::memory_order_relaxed);
    int started = 0, refused = 0;
    std::lock_guard<std::mutex> lk(p->mu);
    for (int i = 0; i < n; ++i) {
        const int64_t e = ids[i];
        int begun[8]; int nb = 0; bool fail = false;
        for (int k = 0; k < nt && !fail; ++k) {
            rp_tensor * t = ts[k];
            if (e < 0 || e >= t->n || t->rec[e].state != RP_OUT) { continue; }
            if (rp_start(p, *t, e, false, false, true)) { t->rec[e].graph = g; begun[nb++] = k; } else { fail = true; }   // stamped: not evicted before its layer asks
        }
        if (fail) {
            for (int j = 0; j < nb; ++j) { rp_unstart(p, *ts[begun[j]], e); }
            refused++;
            if (upto) { refused += n - i - 1; break; }
        } else { started += nb; }
    }
    p->st_ahead += (uint64_t) started;
    if (started > 0) { p->work_cv.notify_all(); }
    return refused;
}

// The copies of these elements have landed (a stream layer's uploads, their event synchronized by the caller): their blocks may go now, inside the graph that asked
// for them. Without it a pass larger than the pool found every block stamped by its own earlier layers, and the pool refused the rest (cs_copy then read the
// mapping: Qwen3.6 4k-token stream at a 2-3 GiB pool, 60-70k refused a request). The pool's own rules pick among them; a later ask stamps them again.
GGML_API void ggml_rampool_done_fork(const void * const * keys, int nk, const int32_t * ids, int n) {
    rp_pool * p = g_rp;
    if (p == nullptr || n <= 0) { return; }
    rp_tensor * ts[8];
    int nt = 0;
    for (int k = 0; k < nk && nt < 8; ++k) { if (rp_tensor * t = rp_find(keys[k])) { ts[nt++] = t; } }
    std::lock_guard<std::mutex> lk(p->mu);
    for (int k = 0; k < nt; ++k) {
        for (int i = 0; i < n; ++i) {
            const int64_t e = ids[i];
            if (e >= 0 && e < ts[k]->n && ts[k]->rec[e].state == RP_IN && ts[k]->rec[e].holds == 0) { ts[k]->rec[e].graph = 0; }
        }
    }
}

GGML_API void ggml_rampool_ahead_stats_fork(uint64_t * started, uint64_t * used, bool reset) {
    *started = g_rp ? g_rp->st_ahead.load() : 0;
    *used    = g_rp ? g_rp->st_ahead_used.load() : 0;
    if (reset && g_rp) { g_rp->st_ahead = 0; g_rp->st_ahead_used = 0; }
}

void ggml_rampool_graph_fork(void) {
    if (g_rp) { g_rp->graph.fetch_add(1, std::memory_order_relaxed); }
}

static void rp_need(const void * const * keys, int nk, const int32_t * ids, int n, bool force);
void ggml_rampool_need_fork(const void * const * keys, int nk, const int32_t * ids, int n) { rp_need(keys, nk, ids, n, false); }
// A CPU kernel's demand: the element is read into the pool whatever the fill's set or the eviction switch (DeepSeek 12k map-guard, build short-paths 2026-11-08:
// the tail call's missed experts found no VRAM block, the CPU computed them, and 22 of 123 GiB of experts were out of the pool with eviction off, so
// ggml_vec_dot_iq1_m read blk.3's down tensor through the mapping). The block comes from the sampled coldest of the class.
void ggml_rampool_demand_fork(const void * const * keys, int nk, const int32_t * ids, int n) { rp_need(keys, nk, ids, n, true); }
static void rp_need(const void * const * keys, int nk, const int32_t * ids, int n, bool force) {
    rp_pool * p = g_rp;
    if (p == nullptr || n <= 0) { return; }
    rp_tensor * ts[8];
    int nt = 0;
    for (int k = 0; k < nk && nt < 8; ++k) { if (rp_tensor * t = rp_find(keys[k])) { ts[nt++] = t; } }
    if (nt == 0) { return; }
    const uint32_t g = p->graph.load(std::memory_order_relaxed);
    int reading = 0, started = 0, hit = 0, refused = 0, bumped = 0;
    std::unique_lock<std::mutex> lk(p->mu);
    // An element is in RAM in all of its tensors or in none: with one tensor out (its class found no colder victim while the others did) the caller takes the
    // element as held and the CPU kernel reads the missing tensor through the mapping (DS 12k, build past-in-pool: all major faults in the down tensor's vec_dot,
    // 1000/s). So one element's reads start together, and when one of them finds no block the ones started are taken back and the element stays out whole.
    for (int i = 0; i < n; ++i) {
        const int64_t e = ids[i];
        int begun[8]; int nb = 0, n_out = 0; bool fail = false;
        for (int k = 0; k < nt; ++k) {
            rp_tensor * t = ts[k];
            if (e < 0 || e >= t->n) { continue; }
            rp_rec & r = t->rec[e];
            const uint32_t g0 = r.graph;
            r.graph = g;
            const bool was_ahead = r.ahead;
            if (was_ahead) { r.ahead = false; p->st_ahead_used++; }
            if (r.state == RP_IN) { rp_promote(r, g0, g); hit++; rp_hit_bytes(p, *t, r); continue; }
            if (r.state == RP_READING) {   // a read ahead still queued behind other reads ahead goes into the demand queue too: the first reader to come claims it
                reading++;
                if (was_ahead && !r.claimed) { p->demand.push_back({ t, e }); bumped++; }
                continue;
            }
            n_out++;
            if (!fail && rp_start(p, *t, e, force, true)) { begun[nb++] = k; } else { fail = true; }
        }
        if (fail) { for (int j = 0; j < nb; ++j) { rp_unstart(p, *ts[begun[j]], e); } refused += n_out; }
        else { started += nb; reading += nb; }
    }
    p->st_asked += (uint64_t) n * nt; p->st_hit += (uint64_t) hit; p->st_cold += (uint64_t) started; p->st_refused += (uint64_t) refused;
    if (started + bumped > 0) { if (started + bumped > 1) { p->work_cv.notify_all(); } else { p->work_cv.notify_one(); } }
    if (reading > 0) {
        const int64_t t0 = rp_now_us();
        int k = 0, i = 0;
        p->done_cv.wait(lk, [&]() {
            for (; k < nt; ++k, i = 0) {
                const rp_tensor * t = ts[k];
                while (i < n && (ids[i] < 0 || ids[i] >= t->n || t->rec[ids[i]].state != RP_READING)) { ++i; }
                if (i < n) { return false; }
            }
            return true;
        });
        const uint64_t w = (uint64_t) (rp_now_us() - t0);
        p->st_wait_us += w; p->wait_total_us += w;
    }
}

const char * ggml_rampool_get_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || e < 0 || e >= t->n) { return nullptr; }
    const rp_rec & r = t->rec[e];
    // asked by this graph or the one before it: no eviction before the graph after next (the graph counter moves only between graphs). Without eviction a block in stays in, so
    // no stamp is needed: a prompt pass's CPU op (no need before it) reads the pool too, not the mapping (rampool3: first 1938-token pass 46.8 s vs 16.7 s without the pool)
    const uint32_t g = g_rp->graph.load(std::memory_order_relaxed);
    if (g_rp->evict && r.graph + 1 < g) { return nullptr; }
    // eviction off: a CPU kernel's read is stamped, so a forced demand in this graph (another thread, another expert) never takes the block it reads
    // shortcut: no lock, so a demand that checked the stamp just before this store can still take the block; nanoseconds wide, forced demands only; a hazard flag if it shows
    if (!g_rp->evict && r.graph != g) { __atomic_store_n(const_cast<uint32_t *>(&r.graph), g, __ATOMIC_SEQ_CST); }
    return t->ptr[e].load(std::memory_order_acquire);
}

const char * ggml_rampool_use_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || e < 0 || e >= t->n) { return nullptr; }
    rp_pool * p = g_rp;
    std::lock_guard<std::mutex> lk(p->mu);
    rp_rec & r = t->rec[e];
    if (r.state != RP_IN) { p->st_pp_miss++; p->rd_ppmiss_b += t->esz; return nullptr; }
    const uint32_t g = p->graph.load(std::memory_order_relaxed);
    rp_promote(r, r.graph, g);
    r.graph = g;
    p->st_pp_hit++;
    rp_hit_bytes(p, *t, r);
    return t->ptr[e].load(std::memory_order_relaxed);
}

const char * ggml_rampool_take_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || e < 0 || e >= t->n) { return nullptr; }
    rp_pool * p = g_rp;
    rp_rec & r = t->rec[e];
    std::unique_lock<std::mutex> lk(p->mu);
    r.holds++;
    p->st_asked++;
    if (r.state == RP_IN) { rp_promote(r, r.graph, p->graph.load(std::memory_order_relaxed)); p->st_hit++; rp_hit_bytes(p, *t, r); return t->ptr[e].load(std::memory_order_relaxed); }
    if (r.state == RP_OUT) {
        if (!rp_start(p, *t, e)) {
            r.holds--;
            p->st_refused++;
            return nullptr;
        }
        p->st_cold++;
        p->work_cv.notify_one();
    } else if (r.state == RP_READING && r.ahead && !r.claimed) {   // a read ahead still queued: into the demand queue too (rp_need does the same)
        r.ahead = false;
        p->st_ahead_used++;
        p->demand.push_back({ t, e });
        p->work_cv.notify_one();
    }
    const int64_t t0 = rp_now_us();
    p->done_cv.wait(lk, [&]() { return r.state != RP_READING; });
    const uint64_t w = (uint64_t) (rp_now_us() - t0);
    p->st_wait_us += w; p->wait_total_us += w;
    if (r.state == RP_IN) { return t->ptr[e].load(std::memory_order_relaxed); }
    r.holds--;
    return nullptr;
}

void ggml_rampool_drop_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || e < 0 || e >= t->n) { return; }
    std::lock_guard<std::mutex> lk(g_rp->mu);
    t->rec[e].holds--;
}

// The pool's size follows the RAM the system leaves (llama-moecache-fork.cpp rp_resize, between requests; user 2026-11-02: grow while the floor holds, give back when pressure
// shows). A class's pressure: bytes it read from the file since the last resize per byte it holds.
static double rp_pressure(const rp_class & c) { return (double) c.cold / (double) std::max<uint64_t>(c.size, (uint64_t) c.n_slots * c.size); }

GGML_API size_t ggml_rampool_grow_fork(size_t bytes, ggml_rampool_alloc_t alloc) {   // declared where it is used (llama-moecache-fork.cpp), no header edit
    rp_pool * p = g_rp;
    if (p == nullptr || alloc == nullptr) { return 0; }
    size_t added = 0;
    std::vector<uint8_t> done(p->cls.size(), 0);   // one chunk per class and call: the next request shows where reads still go
    for (;;) {
        int best = -1;
        {
            std::lock_guard<std::mutex> lk(p->mu);
            for (size_t i = 0; i < p->cls.size(); ++i) {
                const rp_class & c = p->cls[i];
                if (done[i] || c.cold == 0 || c.n_slots >= c.elems) { continue; }
                if (best < 0 || rp_pressure(c) > rp_pressure(p->cls[best])) { best = (int) i; }
            }
        }
        if (best < 0) { break; }
        done[best] = 1;
        rp_class & c = p->cls[best];
        if (c.chunks.empty()) {   // the open gave it nothing: chunks of up to 1 GiB, as there (no block exists, no reader can look at the vector)
            c.per_chunk = (uint32_t) std::max<size_t>(1, (((size_t) 1 << 30) - c.size) / c.size);
            c.chunks.reserve((size_t) ((c.elems + c.per_chunk - 1) / c.per_chunk) + 2);
        }
        const uint64_t k = std::min<uint64_t>({ (uint64_t) c.per_chunk, c.elems - c.n_slots, (uint64_t) ((bytes - added) / c.size) });
        if (k == 0 || c.chunks.size() >= c.chunks.capacity()) { continue; }
        char * m = (char *) alloc((size_t) k * c.size + c.size, p->ud);   // one block more, as at open: what is read past the last block stays inside the range
        if (m == nullptr) { break; }
        memset(m + (size_t) k * c.size, 0, c.size);
        {
            std::lock_guard<std::mutex> lk(p->mu);
            const uint32_t base = (uint32_t) c.chunks.size() * c.per_chunk;
            c.chunks.push_back(m);
            c.chunk_n.push_back((uint32_t) k);
            c.owner.resize((size_t) base + k);
            c.left.resize((size_t) base + k, 0);
            for (uint32_t s = (uint32_t) k; s > 0; --s) { c.free.push_back(base + s - 1); }   // empty: the next misses of the class land here before anything is evicted
            c.n_slots += (uint32_t) k;
            p->pool_bytes += k * c.size;
        }
        added += (size_t) k * c.size;
    }
    std::lock_guard<std::mutex> lk(p->mu);
    for (auto & c : p->cls) { c.cold = 0; }
    return added;
}

GGML_API size_t ggml_rampool_shrink_fork(size_t bytes, void (*free_chunk)(void * chunk, void * ud)) {
    rp_pool * p = g_rp;
    if (p == nullptr || free_chunk == nullptr || bytes == 0) { return 0; }
    size_t freed = 0;
    std::vector<char *> gone;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        std::vector<uint8_t> skip(p->cls.size(), 0);
        while (freed < bytes) {
            int best = -1;
            for (size_t i = 0; i < p->cls.size(); ++i) {
                const rp_class & c = p->cls[i];
                if (skip[i] || c.chunks.empty()) { continue; }
                if (best < 0 || rp_pressure(c) < rp_pressure(p->cls[best])) { best = (int) i; }
            }
            if (best < 0) { break; }
            rp_class & c = p->cls[best];
            const uint32_t base = (uint32_t) (c.chunks.size() - 1) * c.per_chunk, k = c.chunk_n.back();
            bool ok = true;
            for (uint32_t s = base; s < base + k && ok; ++s) {   // the last chunk goes whole or not at all: no block of it read by this graph or the one before, held or being read
                const rp_owner o = c.owner[s];
                ok = o.t < 0 || rp_evictable(p, p->tensors[o.t]->rec[o.e]);
            }
            if (!ok) { skip[best] = 1; continue; }
            for (uint32_t s = base; s < base + k; ++s) {
                const rp_owner o = c.owner[s];
                if (o.t < 0) { continue; }
                rp_tensor & t = *p->tensors[o.t];
                t.ptr[o.e].store(nullptr, std::memory_order_release);
                t.rec[o.e].slot  = -1;
                t.rec[o.e].state = RP_OUT;
                p->st_evicted++;
            }
            c.free.erase(std::remove_if(c.free.begin(), c.free.end(), [base](uint32_t s) { return s >= base; }), c.free.end());
            c.owner.resize(base);
            c.left.resize(base);
            c.n_slots -= k;
            p->pool_bytes -= (uint64_t) k * c.size;
            gone.push_back(c.chunks.back());
            c.chunks.pop_back();
            c.chunk_n.pop_back();
            freed += (size_t) k * c.size;
        }
        for (auto & c : p->cls) { c.cold = 0; }
    }
    for (char * m : gone) { free_chunk(m, p->ud); }   // outside the lock: an unpin syncs the driver
    return freed;
}

uint64_t ggml_rampool_wait_us_fork(void) {
    return g_rp ? g_rp->wait_total_us.load(std::memory_order_relaxed) : 0;
}

// the live map (run.py): bytes of blocks holding the element that came in first and never left (the hot set the pool keeps), and of blocks on probation
extern "C" GGML_API void ggml_rampool_map_fork(uint64_t * kept, uint64_t * prob) {
    *kept = *prob = 0;
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    std::lock_guard<std::mutex> lk(p->mu);
    for (const auto & c : p->cls) {
        for (size_t s = 0; s < c.owner.size(); ++s) {
            const rp_owner o = c.owner[s];
            if (o.t < 0) { continue; }
            if (p->tensors[o.t]->rec[o.e].prob) { *prob += c.size; } else if (s < c.left.size() && !c.left[s]) { *kept += c.size; }
        }
    }
}

// the live map (GET /moe/stats, user 2026-11-11: "rampool how many are being swapped frequently and how many stick around longer"): the pool's blocks class by
// class as n cells, each the kind of most of its bytes: K kept (the element that came in first, never left), E there churn_us or longer, e newer (the churn),
// p probation, f free. parts: bytes of K E e p f and the pool; rd: bytes served since open by source (kept hits, all hits, file reads, prompt misses to the SSD path).
extern "C" GGML_API void ggml_rampool_live_fork(char * cells, int n, uint64_t * parts, uint64_t * rd, int64_t churn_us) {
    for (int i = 0; i < 6; ++i) { parts[i] = 0; }
    for (int i = 0; i < 4; ++i) { rd[i] = 0; }
    for (int i = 0; i < n; ++i) { cells[i] = 'f'; }
    rp_pool * p = g_rp;
    if (p == nullptr || n <= 0) { return; }
    rd[0] = p->rd_kept_b; rd[1] = p->rd_hit_b; rd[2] = p->rd_cold_b; rd[3] = p->rd_ppmiss_b;
    static const char kind[5] = { 'K', 'E', 'e', 'p', 'f' };
    std::lock_guard<std::mutex> lk(p->mu);
    const int64_t now = rp_now_us();
    uint64_t total = 0;
    for (const auto & c : p->cls) { total += (uint64_t) c.n_slots * c.size; }
    if (total == 0) { return; }
    std::vector<double> cb((size_t) n * 5, 0.0);
    uint64_t at = 0;
    for (const auto & c : p->cls) {
        for (uint32_t s = 0; s < c.n_slots; ++s, at += c.size) {
            const rp_owner o = s < c.owner.size() ? c.owner[s] : rp_owner();
            int k = 4;
            if (o.t >= 0) {
                if (p->tensors[o.t]->rec[o.e].prob) { k = 3; }
                else if (s < c.left.size() && !c.left[s]) { k = 0; }
                else { k = s < c.t_in.size() && now - c.t_in[s] < churn_us ? 2 : 1; }
            }
            parts[k] += c.size;
            for (uint64_t lo = at, hi = at + c.size; lo < hi; ) {   // the block's bytes over the cells it covers
                const uint64_t ci = std::min<uint64_t>((uint64_t) n - 1, lo * (uint64_t) n / total), end = std::min<uint64_t>(hi, (ci + 1) * total / (uint64_t) n);
                cb[ci * 5 + k] += (double) (end > lo ? end - lo : hi - lo);
                lo = end > lo ? end : hi;
            }
        }
    }
    parts[5] = total;
    for (int i = 0; i < n; ++i) {
        int best = 4;
        for (int k = 0; k < 5; ++k) { if (cb[(size_t) i * 5 + k] > cb[(size_t) i * 5 + best]) { best = k; } }
        cells[i] = kind[best];
    }
}

void ggml_rampool_stats_fork(struct ggml_rampool_stats * s, bool reset) {
    memset(s, 0, sizeof(*s));
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    s->asked = p->st_asked; s->hit = p->st_hit; s->cold = p->st_cold; s->refused = p->st_refused; s->evicted = p->st_evicted;
    s->cold_bytes = p->st_cold_b; s->wait_us = p->st_wait_us; s->pp_hit = p->st_pp_hit; s->pp_miss = p->st_pp_miss;
    s->bytes = p->pool_bytes; s->classes = (int) p->cls.size();
    {
        std::lock_guard<std::mutex> lk(p->mu);
        for (const auto & c : p->cls) { s->used += (uint64_t) (c.n_slots - c.free.size()) * c.size; }
    }
    if (reset) {
        p->st_asked = 0; p->st_hit = 0; p->st_cold = 0; p->st_refused = 0; p->st_evicted = 0; p->st_cold_b = 0; p->st_wait_us = 0; p->st_pp_hit = 0; p->st_pp_miss = 0;
    }
}

} // extern "C"
