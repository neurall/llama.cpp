// The RAM pool: expert weights in pinned host memory, read straight from the SSD, no mmap and no page cache.
//
// Why. A model that does not fit pinned RAM used to be split by tensor order: the first tensors pinned, the rest mapped. DeepSeek V4.1 Flash on 125 GiB of RAM: 123 GiB of experts,
// 101 GiB pinned, so 7 of 40 layers came through the mapping, through 9 GiB of page cache that the next pass evicted again. Those layers took about 7 of the 12 s of a pass: the link
// moves 26 GB/s from pinned memory and about half of that from pageable memory, the page cache read the SSD at 2.3 GB/s, and each pageable copy waits for the stream first.
// Here every byte of an expert goes one of two ways: it is in the pool (direct DMA, full link) or it is read into the pool first (32 threads, about 5.2 GB/s on PC1). Nothing is held
// twice, and no page fault is ever taken on the way.
//
// What it is. A store of small equal fragments, one per expert slice (up, gate, down each their own). Unlike the GPU pool nothing here grows, moves or is lent, so it shares no code
// with it: a class per (slice bytes, type), blocks of a class in pinned chunks, a table element -> block, and "free" is a change of owner. The record of an element is its tensor's file
// handle and offset plus its index, which is all a rehydration needs: the block can always be dropped, the bytes are on the SSD.
//
// Rules.
//   - One writer function, rp_read, and it writes only inside the block it owns: [slot, slot + class size). Everything else here changes tables.
//   - Check and load BEFORE the kernel (ggml_rampool_need_fork, GPU ring and CPU op alike), never on a memory access. The readers fetch ahead (ggml_rampool_ahead_fork) so the check finds
//     the bytes there; what it waits for is what the prediction missed.
//   - A block is taken from an element only when nobody can still read it: not asked by this graph or the one before it (a ring copy of the previous graph can still run), no holds.
//     Coldest first: elements a card already holds, then the least often asked. An element that finds no block stays out and its caller reads it the old way (the stats count them).
//   - Elements of a class lie directly behind each other as the file has them, so one DMA serves a run. Read past the end of a run (a quantized matmul reads whole tiles) lands in the
//     next block, always the same type and whole blocks of it, or zeros in the slack at the end of a chunk: any finite bytes will do there, the rule is only "no NaN" (fork: MMQ padding).
#include "ggml-rampool-fork.h"
#include "ggml-impl.h"
#include "ggml-read-fork.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
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
    uint32_t uses    = 0;       // how often it was asked for, over all runs: the coldest goes first
    uint32_t graph   = 0;       // the graph that last asked for it
    uint16_t holds   = 0;       // readers outside a graph (ggml_rampool_take_fork)
    uint8_t  state   = RP_OUT;
    bool     claimed = false;   // a reader has it (the queues can hold a second job for it: a wait moves an element up to the demand queue)
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
    std::vector<uint32_t> victims;      // blocks in eviction order, as of the last look; stale entries are checked at the pop
    size_t   vpos = 0;
    char * at(uint32_t s) const { return chunks[s / per_chunk] + (size_t) (s % per_chunk) * size; }
};

struct rp_job { rp_tensor * t; int64_t e; };

struct rp_pool {
    std::vector<std::unique_ptr<rp_tensor>>      tensors;
    std::unordered_map<const void *, rp_tensor *> by_key;
    std::vector<rp_class> cls;
    ggml_rampool_low_t low = nullptr;
    void * ud = nullptr;
    std::map<int, int> dfd_of;          // file handle -> its O_DIRECT handle
    int    n_threads = 0;

    std::mutex              mu;         // tables, queues and records; held for microseconds (a rebuild of the eviction order: a few ms, only when a class ran out)
    std::condition_variable work_cv, done_cv;
    std::deque<rp_job>      demand, ahead;   // a caller waits for what is in demand; ahead is what the readers fetch while nobody waits, in the order it was asked
    std::vector<std::thread> readers;
    bool                    stop = false;
    std::atomic<uint32_t>   graph{ 2 };
    std::atomic<size_t>     memal{ 4 };  // memory alignment O_DIRECT takes here: 4 bytes on NVMe (queue/dma_alignment); an EINVAL raises it, the reader then goes through its bounce buffer
    size_t                  rq = 0;      // bytes of one read request (the device's own limit, see rp_request_bytes)

    std::atomic<uint64_t> st_asked{0}, st_cold{0}, st_ahead{0}, st_evicted{0}, st_lost{0}, st_cold_b{0}, st_wait_us{0};
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
            const ssize_t r = sr_pread1(fd, dst + d, n - d, off + d);
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
            want = (size_t) ((end - pos) / RP_SECTOR * RP_SECTOR);
            if (p->rq > 0) { want = std::min(want, p->rq / RP_SECTOR * RP_SECTOR); }
        }
        const uint64_t u0 = std::max(pos, off), u1 = std::min(pos + want, end);   // the bytes of this request that belong to dst
        char * d = dst + (size_t) (u0 - off);
        const size_t ma = p->memal.load(std::memory_order_relaxed);
        const bool whole = u0 == pos && u1 == pos + want;
        const bool direct = whole && ((uintptr_t) d & (ma - 1)) == 0;
        if (!direct) { want = std::min(want, bounce_n); }
        ssize_t r = sr_pread1(dfd, direct ? d : bounce, want, pos);
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

// A block for an element of class c, -1: none. The caller holds the lock. only_low: only from an element a card holds (a background fetch never evicts what the compute reads).
int32_t rp_take_block(rp_pool * p, rp_class & c, bool only_low) {
    if (!c.free.empty()) {
        const uint32_t s = c.free.back();
        c.free.pop_back();
        return (int32_t) s;
    }
    for (int round = 0; round < 2; ++round) {
        while (c.vpos < c.victims.size()) {
            const uint32_t s = c.victims[c.vpos];
            const rp_owner o = c.owner[s];
            if (o.t < 0) { c.vpos++; continue; }
            rp_tensor & t = *p->tensors[o.t];
            rp_rec & r = t.rec[o.e];
            if (!rp_evictable(p, r)) { c.vpos++; continue; }
            if (only_low && !(p->low && p->low(t.key, o.e, p->ud))) { return -1; }   // the order puts the low ones first: none left
            c.vpos++;
            t.ptr[o.e].store(nullptr, std::memory_order_release);
            r.slot  = -1;
            r.state = RP_OUT;
            c.owner[s] = rp_owner();
            p->st_evicted++;
            return (int32_t) s;
        }
        if (round == 1) { break; }
        // the order as of now: what a card holds first, then the least asked; among equals the highest tensor, the one a scan of the layers needs last
        struct cand { uint64_t rank; uint32_t s; };
        std::vector<cand> v;
        v.reserve(c.owner.size());
        for (uint32_t s = 0; s < (uint32_t) c.owner.size(); ++s) {
            const rp_owner o = c.owner[s];
            if (o.t < 0) { continue; }
            const rp_tensor & t = *p->tensors[o.t];
            const rp_rec & r = t.rec[o.e];
            if (!rp_evictable(p, r)) { continue; }
            const bool lo = p->low && p->low(t.key, o.e, p->ud);
            v.push_back({ ((uint64_t) (lo ? 0 : 1) << 63) | ((uint64_t) r.uses << 31) | (uint64_t) (0x7fffffffu - (uint32_t) o.t), s });
        }
        std::sort(v.begin(), v.end(), [](const cand & a, const cand & b) { return a.rank < b.rank; });
        c.victims.resize(v.size());
        for (size_t i = 0; i < v.size(); ++i) { c.victims[i] = v[i].s; }
        c.vpos = 0;
    }
    return -1;
}

// element e leaves for a block and a reader. The caller holds the lock. false: no block.
bool rp_start(rp_pool * p, rp_tensor & t, int64_t e, bool front, bool only_low) {
    rp_class & c = p->cls[t.cls];
    rp_rec & r = t.rec[e];
    const int32_t s = rp_take_block(p, c, only_low);
    if (s < 0) { return false; }
    r.slot    = s;
    r.state   = RP_READING;
    r.claimed = false;
    c.owner[s] = { t.idx, (int32_t) e };
    (front ? p->demand : p->ahead).push_back({ &t, e });
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
            p->work_cv.wait(lk, [p]() { return p->stop || !p->demand.empty() || !p->ahead.empty(); });
            if (p->stop) { break; }
            auto & q = !p->demand.empty() ? p->demand : p->ahead;
            j = q.front();
            q.pop_front();
            rp_rec & r = j.t->rec[j.e];
            if (r.state != RP_READING || r.claimed) { continue; }   // the other job of an element that moved up
            r.claimed = true;
        }
        rp_tensor & t = *j.t;
        rp_rec & r = t.rec[j.e];
        rp_class & c = p->cls[t.cls];
        char * dst = c.at((uint32_t) r.slot);   // the block is ours: its owner changes only under RP_IN
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

// elements `ids` are asked for. The lock is taken once per call (one call per tensor): a graph makes a few hundred of these. After it, what is in stays in until the graph after next.
void rp_ask(rp_pool * p, rp_tensor & t, const int32_t * ids, int n, bool count, bool wait, bool only_low) {
    const uint32_t g = p->graph.load(std::memory_order_relaxed);
    int reading = 0, started = 0, lost = 0;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        for (int i = 0; i < n; ++i) {
            const int64_t e = ids[i];
            rp_rec & r = t.rec[e];
            if (count) { r.uses++; }
            r.graph = g;
            if (r.state == RP_IN) { continue; }
            if (r.state == RP_READING) {
                if (wait && !r.claimed) { p->demand.push_back({ &t, e }); started++; }   // still in the ahead queue: a caller waits for it now
                reading++;
                continue;
            }
            if (rp_start(p, t, e, wait, only_low)) { started++; reading++; } else { lost += wait; }
        }
    }
    if (count) { p->st_asked += (uint64_t) n; }
    if (lost > 0) { p->st_lost += (uint64_t) lost; }
    if (started > 0) {
        (wait ? p->st_cold : p->st_ahead) += (uint64_t) started;
        if (started > 1) { p->work_cv.notify_all(); } else { p->work_cv.notify_one(); }
    }
    if (wait && reading > 0) {
        const int64_t t0 = rp_now_us();
        std::unique_lock<std::mutex> lk(p->mu);
        int i = 0;
        p->done_cv.wait(lk, [&]() {
            while (i < n && t.rec[ids[i]].state != RP_READING) { ++i; }
            return i == n;
        });
        p->st_wait_us += (uint64_t) (rp_now_us() - t0);
    }
}

// a thread's list of element ids (the call that asks for a whole tensor or by counts)
std::vector<int32_t> & rp_ids() {
    static thread_local std::vector<int32_t> v;
    return v;
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

bool ggml_rampool_open_fork(size_t bytes, size_t chunk_bytes, ggml_rampool_alloc_t alloc, ggml_rampool_low_t low, void * ud, int n_threads) {
    rp_pool * p = g_rp;
    if (p == nullptr) { return false; }
    p->low = low; p->ud = ud; p->n_threads = n_threads;
    uint64_t total = 0;
    for (const auto & c : p->cls) { total += c.elems * c.size; }
    const double share = total > bytes ? (double) bytes / (double) total : 1.0;   // every class the same share of its elements
    bool any = false;
    for (auto & c : p->cls) {
        const uint64_t want = (uint64_t) ((double) c.elems * share);
        c.per_chunk = (uint32_t) std::max<size_t>(1, chunk_bytes / c.size);
        uint64_t got = 0;
        while (got < want) {
            // one block more than it hands out: what is read past the last block (see the rules) stays inside the registered range
            char * m = (char *) alloc((size_t) c.per_chunk * c.size + c.size, ud);
            if (m == nullptr) { GGML_LOG_WARN("rampool: no more pinned memory at class %zu bytes, %llu of %llu blocks\n", c.size, (unsigned long long) got, (unsigned long long) want); break; }
            memset(m + (size_t) c.per_chunk * c.size, 0, c.size);
            c.chunks.push_back(m);
            got += c.per_chunk;
        }
        c.n_slots = (uint32_t) std::min(got, c.elems);
        c.owner.assign(c.n_slots, rp_owner());
        c.free.clear();
        for (uint32_t s = c.n_slots; s > 0; --s) { c.free.push_back(s - 1); }
        p->pool_bytes += (uint64_t) c.chunks.size() * c.per_chunk * c.size;
        any |= c.n_slots > 0;
    }
    if (!any) {
        ggml_rampool_close_fork();
        return false;
    }
    return true;
}

void ggml_rampool_seed_fork(const void * key, int64_t e, uint32_t uses) {
    if (rp_tensor * t = rp_find(key)) { t->rec[e].uses = uses; }
}

// The first fill, before anything else reads the pool: the most used elements of each class, laid out in file order, read by the same writer in requests of p->rq bytes on n_threads threads.
void ggml_rampool_fill_fork(void) {
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    const int64_t t0 = rp_now_us();
    for (auto & c : p->cls) {
        struct el { uint32_t uses; int32_t t, e; };
        std::vector<el> v;
        v.reserve((size_t) c.elems);
        for (auto & t : p->tensors) {
            if (&p->cls[t->cls] != &c) { continue; }
            for (int64_t e = 0; e < t->n; ++e) { v.push_back({ t->rec[e].uses, t->idx, (int32_t) e }); }
        }
        std::stable_sort(v.begin(), v.end(), [](const el & a, const el & b) { return a.uses > b.uses; });
        if (v.size() > c.n_slots) { v.resize(c.n_slots); }
        std::sort(v.begin(), v.end(), [](const el & a, const el & b) { return a.t != b.t ? a.t < b.t : a.e < b.e; });
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
    // runs of elements that are neighbours in the file and in a chunk are one range, cut into requests here so the threads share them evenly
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
                const uint64_t cut = std::min(end, (off / piece + 1) * piece);   // on request edges, so all but the first and last are whole sectors
                jobs.push_back({ &t, dst, (size_t) (cut - off), off });
                dst += (size_t) (cut - off);
                off = cut;
            }
            bytes += end - (t.off + (uint64_t) e * t.esz);
            e = b;
        }
    }
    std::atomic<size_t> next{ 0 };
    std::atomic<uint64_t> bad{ 0 };
    auto work = [&]() {
        rp_thread_setup();
        const size_t bn = std::max<size_t>(p->rq, RP_SECTOR);
        char * bounce = rp_bounce_new(bn);
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= jobs.size()) { break; }
            const range & r = jobs[i];
            if (bounce == nullptr || !rp_read(p, r.t->dfd, r.t->fd, r.dst, r.n, r.off, bounce, bn)) { bad++; }
        }
        free(bounce);
    };
    std::vector<std::thread> th;
    for (int i = 0; i < p->n_threads; ++i) { th.emplace_back(work); }
    for (auto & x : th) { x.join(); }
    if (bad.load() > 0) { GGML_LOG_ERROR("rampool: %llu reads of the fill failed\n", (unsigned long long) bad.load()); }
    for (auto & tp : p->tensors) {
        rp_tensor & t = *tp;
        rp_class & c = p->cls[t.cls];
        for (int64_t e = 0; e < t.n; ++e) {
            rp_rec & r = t.rec[e];
            if (r.slot < 0) { continue; }
            r.claimed = false;
            r.state   = RP_IN;
            t.ptr[e].store(c.at((uint32_t) r.slot), std::memory_order_release);
        }
    }
    const double s = std::max(1e-3, (rp_now_us() - t0) / 1e6);
    GGML_LOG_WARN("rampool: %.1f GiB read in %.1f s (%.2f GB/s, %d threads, requests of %zu KiB%s)\n", bytes / 1073741824.0, s, bytes / 1e9 / s, p->n_threads, p->rq >> 10,
            p->tensors.empty() || p->tensors[0]->dfd >= 0 ? "" : ", no O_DIRECT here");
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
    return t ? reinterpret_cast<const char * const *>(t->ptr.get()) : nullptr;   // std::atomic<T *> is a plain pointer here (static_assert below): a reader's load after a need is a plain load
}
static_assert(sizeof(std::atomic<const char *>) == sizeof(const char *), "the table is read as plain pointers");

void ggml_rampool_graph_fork(void) {
    if (g_rp) { g_rp->graph.fetch_add(1, std::memory_order_relaxed); }
}

void ggml_rampool_need_fork(const void * key, const int32_t * ids, int n, bool count) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return; }
    if (ids == nullptr) {
        auto & v = rp_ids();
        v.resize((size_t) t->n);
        for (int64_t e = 0; e < t->n; ++e) { v[e] = (int32_t) e; }
        ids = v.data();
        n = (int) t->n;
    }
    rp_ask(g_rp, *t, ids, n, count, true, false);
}

void ggml_rampool_need_counts_fork(const void * key, const int64_t * counts, int n) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return; }
    auto & v = rp_ids();
    v.clear();
    for (int e = 0; e < n && e < t->n; ++e) { if (counts[e] > 0) { v.push_back(e); } }
    rp_ask(g_rp, *t, v.data(), (int) v.size(), true, true, false);
}

void ggml_rampool_ahead_fork(const void * key, const int32_t * ids, int n, bool only_low) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return; }
    if (ids == nullptr) {
        auto & v = rp_ids();
        v.clear();
        for (int64_t e = 0; e < t->n; ++e) { if (t->rec[e].state == RP_OUT) { v.push_back((int32_t) e); } }
        ids = v.data();
        n = (int) v.size();
    }
    rp_ask(g_rp, *t, ids, n, false, false, only_low);
}

int64_t ggml_rampool_run_fork(const void * key, int64_t e, int64_t end) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return 1; }
    const rp_class & c = g_rp->cls[t->cls];
    const char * p0 = t->ptr[e].load(std::memory_order_acquire);
    const int32_t s0 = t->rec[e].slot;
    int64_t b = e + 1;
    while (b < end && t->ptr[b].load(std::memory_order_acquire) == p0 + (size_t) (b - e) * t->esz && t->rec[b].slot == s0 + (b - e) && (uint32_t) t->rec[b].slot / c.per_chunk == (uint32_t) s0 / c.per_chunk) { ++b; }
    return b - e;
}

const char * ggml_rampool_take_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return nullptr; }
    rp_pool * p = g_rp;
    rp_rec & r = t->rec[e];
    std::unique_lock<std::mutex> lk(p->mu);
    r.holds++;
    r.uses++;
    p->st_asked++;
    if (r.state == RP_IN) { return t->ptr[e].load(std::memory_order_relaxed); }
    if (r.state == RP_OUT) {
        if (!rp_start(p, *t, e, true, false)) {
            r.holds--;
            p->st_lost++;
            return nullptr;
        }
        p->st_cold++;
    } else if (!r.claimed) {
        p->demand.push_back({ t, e });
        p->st_cold++;
    }
    p->work_cv.notify_one();
    const int64_t t0 = rp_now_us();
    p->done_cv.wait(lk, [&]() { return r.state != RP_READING; });
    p->st_wait_us += (uint64_t) (rp_now_us() - t0);
    if (r.state == RP_IN) { return t->ptr[e].load(std::memory_order_relaxed); }
    r.holds--;
    return nullptr;
}

void ggml_rampool_drop_fork(const void * key, int64_t e) {
    rp_tensor * t = rp_find(key);
    if (t == nullptr) { return; }
    std::lock_guard<std::mutex> lk(g_rp->mu);
    t->rec[e].holds--;
}

void ggml_rampool_stats_fork(struct ggml_rampool_stats * s, bool reset) {
    memset(s, 0, sizeof(*s));
    rp_pool * p = g_rp;
    if (p == nullptr) { return; }
    s->asked = p->st_asked; s->cold = p->st_cold; s->ahead = p->st_ahead; s->evicted = p->st_evicted; s->lost = p->st_lost;
    s->cold_bytes = p->st_cold_b; s->wait_us = p->st_wait_us; s->bytes = p->pool_bytes;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        for (const auto & c : p->cls) { s->used += (uint64_t) (c.owner.size() - c.free.size()) * c.size; }
    }
    if (reset) {
        p->st_asked = 0; p->st_cold = 0; p->st_ahead = 0; p->st_evicted = 0; p->st_lost = 0; p->st_cold_b = 0; p->st_wait_us = 0;
    }
}

} // extern "C"
