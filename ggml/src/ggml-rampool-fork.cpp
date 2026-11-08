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
//   - Eviction: of 64 sampled evictable blocks of the class the one expected back last; a newcomer gets it only when it is expected back sooner (else it stays out, "refused",
//     and its caller reads it the old way): a cold expert read once does not push a warm one out.
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
};

struct rp_tensor {
    const void * key  = nullptr;
    int          fd   = -1, dfd = -1;   // the file; the same file opened O_DIRECT, -1: not possible here
    size_t       off  = 0, esz = 0;     // where element 0 lies, element bytes
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
    uint32_t n_slots = 0, per_chunk = 0;
    std::vector<char *>   chunks;
    std::vector<rp_owner> owner;        // block -> element
    std::vector<uint32_t> free;         // blocks without an owner (a stack)
    uint32_t sample = 0;                // where the next eviction sample starts
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
    std::vector<std::thread> readers;
    bool                    stop = false;
    std::atomic<uint32_t>   graph{ 2 };
    std::atomic<size_t>     memal{ 4 };  // memory alignment O_DIRECT takes here: 4 bytes on NVMe (queue/dma_alignment); an EINVAL raises it, the reader then goes through its bounce buffer
    size_t                  rq = 0;      // bytes of one read request (the device's own limit, see rp_request_bytes)
    bool                    evict = false;   // ggml_rampool_evict_fork: blocks change owner during the run; off: the fill's set stays, a slice out of it is read the old way

    std::atomic<uint64_t> st_asked{0}, st_hit{0}, st_cold{0}, st_refused{0}, st_evicted{0}, st_cold_b{0}, st_wait_us{0}, st_pp_hit{0}, st_pp_miss{0};
    std::atomic<uint64_t> wait_total_us{0};   // st_wait_us without the per-request reset (ggml_rampool_wait_us_fork)
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

// out of a graph's reach: not read by this graph or the one before it, and no reader outside a graph
bool rp_evictable(const rp_pool * p, const rp_rec & r) {
    return r.state == RP_IN && r.holds == 0 && r.graph + 1 < p->graph.load(std::memory_order_relaxed);
}

// A block for an element of class c expected back at `want` (the heat clock), -1: none. The caller holds the lock. A free block first; else of 64 sampled evictable blocks the
// one expected back last, taken only when the newcomer is expected back sooner.
int32_t rp_take_block(rp_pool * p, rp_class & c, double want) {
    if (!c.free.empty()) {
        const uint32_t s = c.free.back();
        c.free.pop_back();
        return (int32_t) s;
    }
    if (!p->evict) { return -1; }
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
        if (!rp_evictable(p, t.rec[o.e])) { continue; }
        seen++;
        const double h = p->heat(t.key, o.e, p->ud);
        if (h > bh) { bh = h; best = (int32_t) s; }
    }
    c.sample = (c.sample + 1) % n;
    if (best < 0 || !(want < bh)) { return -1; }
    const rp_owner o = c.owner[best];
    rp_tensor & t = *p->tensors[o.t];
    t.ptr[o.e].store(nullptr, std::memory_order_release);
    t.rec[o.e].slot  = -1;
    t.rec[o.e].state = RP_OUT;
    c.owner[best] = rp_owner();
    p->st_evicted++;
    return best;
}

// element e leaves for a block and a reader. The caller holds the lock. false: no block (refused).
bool rp_start(rp_pool * p, rp_tensor & t, int64_t e) {
    const double want = p->heat(t.key, e, p->ud);
    if (want == HUGE_VAL) { return false; }   // a card holds it: a RAM copy pays nothing
    rp_class & c = p->cls[t.cls];
    rp_rec & r = t.rec[e];
    const int32_t s = rp_take_block(p, c, want);
    if (s < 0) { return false; }
    r.slot    = s;
    r.state   = RP_READING;
    r.claimed = false;
    c.owner[s] = { t.idx, (int32_t) e };
    p->demand.push_back({ &t, e });
    return true;
}

void rp_reader(rp_pool * p) {
    rp_thread_setup();
    const size_t bn = std::max<size_t>(p->rq, RP_SECTOR);
    char * bounce = rp_bounce_new(bn);
    for (;;) {
        rp_job j;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->work_cv.wait(lk, [p]() { return p->stop || !p->demand.empty(); });
            if (p->stop) { break; }
            j = p->demand.front();
            p->demand.pop_front();
            rp_rec & r = j.t->rec[j.e];
            if (r.state != RP_READING || r.claimed) { continue; }
            r.claimed = true;
        }
        rp_tensor & t = *j.t;
        rp_rec & r = t.rec[j.e];
        rp_class & c = p->cls[t.cls];
        char * dst = c.at((uint32_t) r.slot);   // the block is ours: its owner changes only under RP_IN
        // O_DIRECT: a buffered read leaves a second copy in the page cache, which then pushes out the slices the pool does not hold (rampool3, GLM 3.5 warm decode 13.0 -> 6.5 t/s)
        const bool ok = bounce != nullptr && rp_read(p, t.dfd, t.fd, dst, t.esz, t.off + (uint64_t) j.e * t.esz, bounce, bn);
        {
            std::lock_guard<std::mutex> lk(p->mu);
            if (ok) {
                t.ptr[j.e].store(dst, std::memory_order_release);
                r.state = RP_IN;
                p->st_cold_b += t.esz;
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

void ggml_rampool_add_fork(const void * key, int fd, size_t file_off, int64_t n, size_t esz, int32_t kind) {
    if (g_rp == nullptr) { g_rp = new rp_pool(); }
    rp_pool * p = g_rp;
    auto t = std::make_unique<rp_tensor>();
    t->key = key; t->fd = fd; t->off = file_off; t->n = n; t->esz = esz; t->idx = (int32_t) p->tensors.size();
    t->rec.resize((size_t) n);
    t->ptr.reset(new std::atomic<const char *>[(size_t) n]);
    for (int64_t e = 0; e < n; ++e) { t->ptr[e].store(nullptr, std::memory_order_relaxed); }
    int c = -1;
    for (size_t i = 0; i < p->cls.size(); ++i) { if (p->cls[i].size == esz && p->cls[i].kind == kind) { c = (int) i; } }
    if (c < 0) {
        c = (int) p->cls.size();
        p->cls.emplace_back();
        p->cls.back().size = esz;
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
        uint64_t got = 0;
        while (got < want) {
            const uint64_t k = std::min<uint64_t>(c.per_chunk, want - got);   // the last chunk only as big as what is left
            // one block more than it hands out: what is read past the last block stays inside the registered range
            char * m = (char *) alloc((size_t) k * c.size + c.size, ud);
            if (m == nullptr) { GGML_LOG_WARN("rampool: no more pinned memory at class %zu bytes, %llu of %llu blocks\n", c.size, (unsigned long long) got, (unsigned long long) want); break; }
            memset(m + (size_t) k * c.size, 0, c.size);
            c.chunks.push_back(m);
            got += k;
        }
        c.n_slots = (uint32_t) std::min(got, c.elems);
        c.owner.assign(c.n_slots, rp_owner());
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

// The first fill: the elements expected back soonest of each class (none a card holds), laid out in file order, read in requests of p->rq bytes on n_threads threads, the requests
// in spaced order (the even ones, then the odd ones: neighbours in the file are never in flight together, ggml-read-fork.h).
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
        }
    }
    struct range { rp_tensor * t; char * dst; size_t n; uint64_t off; };
    std::vector<range> jobs;
    uint64_t bytes = 0;
    const size_t piece = p->rq > 0 ? p->rq / RP_SECTOR * RP_SECTOR : (size_t) 1 << 22;
    for (auto & tp : p->tensors) {
        rp_tensor & t = *tp;
        rp_class & c = p->cls[t.cls];
        for (int64_t e = 0; e < t.n; ) {
            if (t.rec[e].slot < 0) { ++e; continue; }
            int64_t b = e + 1;
            while (b < t.n && t.rec[b].slot == t.rec[e].slot + (b - e) && (uint32_t) t.rec[b].slot / c.per_chunk == (uint32_t) t.rec[e].slot / c.per_chunk) { ++b; }
            uint64_t off = t.off + (uint64_t) e * t.esz, end = t.off + (uint64_t) b * t.esz;
            char * dst = c.at((uint32_t) t.rec[e].slot);
            while (off < end) {
                const uint64_t cut = std::min(end, (off / piece + 1) * piece);
                jobs.push_back({ &t, dst, (size_t) (cut - off), off });
                dst += (size_t) (cut - off);
                off = cut;
            }
            bytes += end - (t.off + (uint64_t) e * t.esz);
            e = b;
        }
    }
    std::vector<range> spaced;
    spaced.reserve(jobs.size());
    for (int pass = 0; pass < 2; ++pass) { for (size_t i = pass; i < jobs.size(); i += 2) { spaced.push_back(jobs[i]); } }
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

void ggml_rampool_graph_fork(void) {
    if (g_rp) { g_rp->graph.fetch_add(1, std::memory_order_relaxed); }
}

void ggml_rampool_need_fork(const void * key, const int32_t * ids, int n) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || n <= 0) { return; }
    rp_pool * p = g_rp;
    const uint32_t g = p->graph.load(std::memory_order_relaxed);
    int reading = 0, started = 0, hit = 0, refused = 0;
    std::unique_lock<std::mutex> lk(p->mu);
    for (int i = 0; i < n; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= t->n) { continue; }
        rp_rec & r = t->rec[e];
        r.graph = g;
        if (r.state == RP_IN) { hit++; continue; }
        if (r.state == RP_READING) { reading++; continue; }
        if (rp_start(p, *t, e)) { started++; reading++; } else { refused++; }
    }
    p->st_asked += (uint64_t) n; p->st_hit += (uint64_t) hit; p->st_cold += (uint64_t) started; p->st_refused += (uint64_t) refused;
    if (started > 0) { if (started > 1) { p->work_cv.notify_all(); } else { p->work_cv.notify_one(); } }
    if (reading > 0) {
        const int64_t t0 = rp_now_us();
        int i = 0;
        p->done_cv.wait(lk, [&]() {
            while (i < n && (ids[i] < 0 || ids[i] >= t->n || t->rec[ids[i]].state != RP_READING)) { ++i; }
            return i == n;
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
    if (g_rp->evict && r.graph + 1 < g_rp->graph.load(std::memory_order_relaxed)) { return nullptr; }
    return t->ptr[e].load(std::memory_order_acquire);
}

const char * ggml_rampool_use_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr || e < 0 || e >= t->n) { return nullptr; }
    rp_pool * p = g_rp;
    std::lock_guard<std::mutex> lk(p->mu);
    rp_rec & r = t->rec[e];
    if (r.state != RP_IN) { p->st_pp_miss++; return nullptr; }
    r.graph = p->graph.load(std::memory_order_relaxed);
    p->st_pp_hit++;
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
    if (r.state == RP_IN) { p->st_hit++; return t->ptr[e].load(std::memory_order_relaxed); }
    if (r.state == RP_OUT) {
        if (!rp_start(p, *t, e)) {
            r.holds--;
            p->st_refused++;
            return nullptr;
        }
        p->st_cold++;
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

uint64_t ggml_rampool_wait_us_fork(void) {
    return g_rp ? g_rp->wait_total_us.load(std::memory_order_relaxed) : 0;
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
        for (const auto & c : p->cls) { s->used += (uint64_t) (c.owner.size() - c.free.size()) * c.size; }
    }
    if (reset) {
        p->st_asked = 0; p->st_hit = 0; p->st_cold = 0; p->st_refused = 0; p->st_evicted = 0; p->st_cold_b = 0; p->st_wait_us = 0; p->st_pp_hit = 0; p->st_pp_miss = 0;
    }
}

} // extern "C"
