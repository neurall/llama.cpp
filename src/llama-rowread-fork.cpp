// fork: lazy row tables read straight from the file instead of through the mapping.
//
// DeepSeek V4.1's Engram table holds 768M rows of 144 bytes (103 GiB). Each token reads 48 of them at hashed, random places. Through the mapping every row cost a
// 4 KiB page of page cache (28x its size) next to the RAM pool, read by madvise(WILLNEED) or by a page fault with readahead (DS 13.5k-token call: about 115k major
// faults of 24 KiB in its first 4 s; user 2026-11-10: "the pressure was caused by our mmap use"). Here the rows a pass needs are read as O_DIRECT pages in one batch
// (ggml-read-fork.h sr_read_pages: kernel AIO on Linux, overlapped ReadFile on Windows) into a compact table that the graph gathers from. No page cache, nothing mapped is read.
//
// LLAMA_ROWREAD=0 keeps the mapping (A/B).

#include "llama-impl.h"
#include "llama-rowread-fork.h"
#include "../ggml/src/ggml-read-fork.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
struct row_src { int fd = -1; uint64_t off = 0; };
uint8_t * page_alloc(size_t n) {
#ifdef _WIN32
    return (uint8_t *) _aligned_malloc(n, direct_align());
#else
    void * p = nullptr;
    return posix_memalign(&p, direct_align(), n) == 0 ? (uint8_t *) p : nullptr;
#endif
}
void page_free(uint8_t * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
std::mutex g_mu;
std::unordered_map<const ggml_tensor *, row_src> g_src;
}

// rows kept in the RAM pool by their histories (ggml-rampool-fork.cpp, rows part); declared where used, as in llama-moecache-fork.cpp
extern "C" void   ggml_rampool_rows_add_fork(const void * key, size_t rs);
extern "C" size_t ggml_rampool_rows_get_fork(const void * key, const int32_t * rows, size_t n, uint8_t * dst, uint8_t * hit);
extern "C" void   ggml_rampool_rows_put_fork(const void * key, const int32_t * rows, size_t n, const uint8_t * src);
extern "C" void   ggml_rampool_rows_has_fork(const void * key, const int32_t * rows, size_t n, uint8_t * in);
extern "C" void   ggml_rampool_rows_stats_fork(const void * key, uint64_t * v, double * turnover);

// the table t starts at byte `off` of the file open as fd_file (the loader's descriptor): opened once more past the page cache
bool llama_rowread_add_fork(const ggml_tensor * t, int fd_file, uint64_t off) {
    const char * e = getenv("LLAMA_ROWREAD");
    if (e && atoi(e) == 0) { return false; }
    const int fd = open_direct_fd(fd_file);
    if (fd < 0) { return false; }
    // the engine must work here: one page read now, or the caller keeps the mapping
    const size_t pg = direct_align();
    uint8_t * buf = page_alloc(pg);
    const uint64_t p0 = off / pg * pg;
    const bool ok = buf && sr_read_pages(fd, &p0, 1, pg, buf);
    page_free(buf);
    if (!ok) { LLAMA_LOG_WARN("%s: %s stays read through the mapping (no async page reads here)\n", __func__, ggml_get_name(t)); return false; }
    std::lock_guard<std::mutex> lk(g_mu);
    g_src[t] = { fd, off };
    const char * ep = getenv("LLAMA_ROWREAD_POOL");   // =0: every pass reads all its rows (A/B)
    const bool pool = !ep || atoi(ep) != 0;
    if (pool) { ggml_rampool_rows_add_fork(t, ggml_row_size(t->type, t->ne[0])); }
    LLAMA_LOG_INFO("%s: %s read straight from the file (O_DIRECT pages, kernel AIO), not through the mapping%s\n", __func__, ggml_get_name(t),
            pool ? "; rows read again soon stay in RAM by their last-10 histories" : "");
    return true;
}

bool llama_rowread_has_fork(const ggml_tensor * t) {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_src.count(t) > 0;
}

// rows (sorted, unique) of t into dst, row i at dst + i*row size. False: no source or a read failed (dst incomplete).
static bool rowread_gather(const ggml_tensor * t, const int32_t * rows, size_t n, uint8_t * dst) {
    row_src s;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_src.find(t);
        if (it == g_src.end()) { return false; }
        s = it->second;
    }
    const size_t rs = ggml_row_size(t->type, t->ne[0]);
    const uint64_t pg = direct_align();
    constexpr size_t MAXP = 16384;   // pages per batch: 64 MiB of bounce buffer
    GGML_ASSERT(rs <= (MAXP - 1)*pg);
    thread_local uint8_t * buf = nullptr;
    if (!buf) { buf = page_alloc(MAXP*pg); if (!buf) { return false; } }   // one per gathering thread, kept
    std::vector<uint64_t> pages;
    pages.reserve(MAXP);
    size_t i0 = 0;
    while (i0 < n) {
        // the rows of this batch and their pages, in file order; a row's pages sit next to each other in the buffer as in the file (a page two rows share, once)
        pages.clear();
        size_t i1 = i0;
        while (i1 < n) {
            const uint64_t o = s.off + (uint64_t) rows[i1]*rs;
            const uint64_t a = o / pg * pg, b = (o + rs - 1) / pg * pg;
            const size_t add = (size_t) ((b - a) / pg) + 1 - (!pages.empty() && pages.back() == a ? 1 : 0);
            if (pages.size() + add > MAXP) { break; }
            for (uint64_t q = a; q <= b; q += pg) { if (pages.empty() || pages.back() < q) { pages.push_back(q); } }
            ++i1;
        }
        if (!sr_read_pages(s.fd, pages.data(), pages.size(), pg, buf)) { return false; }
        size_t j = 0;
        for (size_t i = i0; i < i1; ++i) {
            const uint64_t o = s.off + (uint64_t) rows[i]*rs;
            const uint64_t a = o / pg * pg;
            while (pages[j] != a) { ++j; }
            memcpy(dst + i*rs, buf + j*pg + (o - a), rs);
        }
        i0 = i1;
    }
    return true;
}

// The read-ahead: a pass's rows are read on one reader thread from the pass's start (llama_rowread_ahead_fork, right after the memory apply), beside its graph build,
// KV moves and copy fill; its fill takes them from RAM and reads only what the read-ahead lacks (qni4 12k pass: 402-455 ms of reads before layer 0, the SSD does
// 125-160k random 4 KiB reads a second, so more readers do not help; hiding them does).
namespace {
struct row_job {
    const ggml_tensor * t = nullptr;
    int part = 0;                 // which gather of the pass (DS: the Engram layer), read in the order asked
    std::vector<int32_t> u;       // the rows, sorted and distinct
    std::vector<uint8_t> data;    // their bytes, u.size() rows
    bool started = false, done = false, ok = false;
};
struct row_reader {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::shared_ptr<row_job>> q;   // jobs in the order asked; a fill takes its table's newest
    bool stop = false;
    std::thread th;
    ~row_reader() {
        { std::lock_guard<std::mutex> lk(mu); stop = true; }
        cv.notify_all();
        if (th.joinable()) { th.join(); }
    }
};
std::atomic<bool> g_ahead_used{false};

void reader_loop(row_reader * r) {
    std::unique_lock<std::mutex> lk(r->mu);
    for (;;) {
        std::shared_ptr<row_job> j;
        r->cv.wait(lk, [&] {
            if (r->stop) { return true; }
            for (auto & x : r->q) { if (!x->started) { j = x; return true; } }
            return false;
        });
        if (r->stop) { return; }
        j->started = true;
        lk.unlock();
        j->data.resize(j->u.size()*ggml_row_size(j->t->type, j->t->ne[0]));
        const bool ok = rowread_gather(j->t, j->u.data(), j->u.size(), j->data.data());
        lk.lock();
        j->ok   = ok;
        j->done = true;
        r->cv.notify_all();
    }
}

row_reader & reader() {
    static row_reader r;
    static std::once_flag once;
    std::call_once(once, [] { r.th = std::thread(reader_loop, &r); });
    return r;
}
}

// the rows ids[0..n) of t for gather `part` of the next pass, read on the reader thread from now (parts in the order asked): the pass finds them in RAM. A job of
// the same t and part that has not started, or was read and never taken (its pass gathered nothing), gives way.
void llama_rowread_ahead_fork(const ggml_tensor * t, int part, const int32_t * ids, size_t n) {
    if (n == 0 || !llama_rowread_has_fork(t)) { return; }
    auto j = std::make_shared<row_job>();
    j->t = t;
    j->part = part;
    j->u.assign(ids, ids + n);
    std::sort(j->u.begin(), j->u.end());
    j->u.erase(std::unique(j->u.begin(), j->u.end()), j->u.end());
    {   // rows in RAM now are not read; the fill takes them from there (nothing evicts between: only a fill of t puts rows of t)
        std::vector<uint8_t> in(j->u.size());
        ggml_rampool_rows_has_fork(t, j->u.data(), j->u.size(), in.data());
        size_t w = 0;
        for (size_t k = 0; k < j->u.size(); ++k) { if (!in[k]) { j->u[w++] = j->u[k]; } }
        j->u.resize(w);
    }
    row_reader & r = reader();
    {
        std::lock_guard<std::mutex> lk(r.mu);
        r.q.erase(std::remove_if(r.q.begin(), r.q.end(), [&](const std::shared_ptr<row_job> & x) { return x->t == t && x->part == part && (!x->started || x->done); }), r.q.end());
        r.q.push_back(j);
    }
    g_ahead_used = true;
    r.cv.notify_all();
}

// per model architecture: the function that names a ubatch's rows (the same hash its graph input uses) and asks for them (llama_rowread_ahead_fork). Registered by
// the model's source at start-up, called by llama_context::process_ubatch after the memory apply.
struct llama_model;
struct llama_memory_context_i;
struct llama_ubatch;
using llama_rowread_ahead_fn = void (*)(const llama_model & model, llama_memory_context_i * mctx, const llama_ubatch & ubatch);
static std::vector<std::pair<int, llama_rowread_ahead_fn>> & ahead_fns() { static std::vector<std::pair<int, llama_rowread_ahead_fn>> v; return v; }

bool llama_rowread_ahead_register_fork(int arch, llama_rowread_ahead_fn fn) {
    ahead_fns().emplace_back(arch, fn);
    return true;
}

void llama_rowread_ahead_ubatch_fork(int arch, const llama_model & model, llama_memory_context_i * mctx, const llama_ubatch & ubatch) {
    static const bool on = [] { const char * e = getenv("LLAMA_ROWREAD_AHEAD"); return !e || atoi(e) != 0; }();   // =0: the A/B arm without it
    if (!on) { return; }
    for (const auto & f : ahead_fns()) { if (f.first == arch) { f.second(model, mctx, ubatch); } }
}

// The same for tokens that are not decoded yet (the candidates of the next token; llama_context::synchronize asks while the caller samples). No pass, so no
// memory context: the architecture's function names the rows from the memory itself, with the hash its pass uses, and asks for them as gather part 0.
struct llama_memory_i;
using llama_rowread_next_fn = void (*)(const llama_model & model, llama_memory_i * mem, const llama_ubatch & nxt);
static std::vector<std::pair<int, llama_rowread_next_fn>> & next_fns() { static std::vector<std::pair<int, llama_rowread_next_fn>> v; return v; }

bool llama_rowread_next_register_fork(int arch, llama_rowread_next_fn fn) {
    next_fns().emplace_back(arch, fn);
    return true;
}

void llama_rowread_next_fork(int arch, const llama_model & model, llama_memory_i * mem, const llama_ubatch & nxt) {
    for (const auto & f : next_fns()) { if (f.first == arch) { f.second(model, mem, nxt); } }
}

// Rows u[0..nu) of t (ascending, distinct) into dst, row k at dst + k*row size: each read from the file (a failed read: from the mapping, logged once). Rows in RAM
// come from the pool, rows a read-ahead of (t, part) holds from it (its newest job, waited for), the rest are read now. False: a read failed.
bool llama_rowread_fill_rows_fork(const ggml_tensor * t, int part, const int32_t * u, size_t nu, uint8_t * dst) {
    const size_t rs = ggml_row_size(t->type, t->ne[0]);
    // rows in RAM first (each recall goes into its history), then the misses: from the read-ahead, else read now; the misses are handed to the pool, which keeps
    // the ones whose histories say they will be read again during a stay
    std::vector<uint8_t> hit(nu);
    const size_t n_ram = ggml_rampool_rows_get_fork(t, u, nu, dst, hit.data());
    std::vector<int32_t> mr;   // the misses, ascending
    std::vector<size_t>  mk;   // their places in u
    mr.reserve(nu - n_ram);
    mk.reserve(nu - n_ram);
    for (size_t k = 0; k < nu; ++k) { if (!hit[k]) { mr.push_back(u[k]); mk.push_back(k); } }
    std::vector<uint8_t> mb(mr.size()*rs);
    std::shared_ptr<row_job> j;
    const int64_t t0 = ggml_time_us();
    if (g_ahead_used.load()) {
        row_reader & r = reader();
        std::unique_lock<std::mutex> lk(r.mu);
        for (auto it = r.q.rbegin(); it != r.q.rend() && !j; ++it) { if ((*it)->t == t && (*it)->part == part) { j = *it; } }
        if (j) {
            r.cv.wait(lk, [&] { return j->done; });
            const auto it = std::find(r.q.begin(), r.q.end(), j);   // a newer read-ahead of t may have dropped it already
            if (it != r.q.end()) { r.q.erase(it); }
        }
    }
    const int64_t t1 = ggml_time_us();
    bool ok = true;
    size_t n_miss = mr.size();
    if (!mr.empty()) {
        std::vector<size_t> rest;
        if (j && j->ok) {
            rest = rowread_take(j->u, j->data.data(), mr, rs, mb.data());
        } else {
            rest.resize(mr.size());
            for (size_t i = 0; i < mr.size(); ++i) { rest[i] = i; }
        }
        n_miss = rest.size();
        if (!rest.empty()) {
            std::vector<int32_t> rr(rest.size());
            for (size_t i = 0; i < rest.size(); ++i) { rr[i] = mr[rest[i]]; }
            std::vector<uint8_t> rb(rest.size()*rs);
            ok = rowread_gather(t, rr.data(), rr.size(), rb.data());
            for (size_t i = 0; ok && i < rest.size(); ++i) { memcpy(mb.data() + rest[i]*rs, rb.data() + i*rs, rs); }
        }
        if (ok) {
            for (size_t i = 0; i < mr.size(); ++i) { memcpy(dst + mk[i]*rs, mb.data() + i*rs, rs); }
            ggml_rampool_rows_put_fork(t, mr.data(), mr.size(), mb.data());
        }
    }
    static const bool st = getenv("LLAMA_STEP_TIME") != nullptr;   // dev: small calls (decode) summed, every 256
    if (st && nu < 512) {
        static size_t c_n = 0, c_rows = 0, c_ram = 0, c_ahead = 0, c_now = 0; static int64_t c_us = 0;
        c_n++; c_rows += nu; c_ram += n_ram; c_ahead += mr.size() - n_miss; c_now += n_miss; c_us += ggml_time_us() - t0;
        if (c_n == 256) {
            LLAMA_LOG_WARN("%s: %s: 256 small calls, rows %.1f a call: in RAM %.1f, read-ahead %.1f, read now %.1f; %.0f us a call after the RAM lookup\n", __func__,
                    ggml_get_name(t), c_rows / 256.0, c_ram / 256.0, c_ahead / 256.0, c_now / 256.0, c_us / 256.0);
            c_n = c_rows = c_ram = c_ahead = c_now = 0; c_us = 0;
        }
    }
    static const bool dbg = getenv("LLAMA_SPLIT_CTL_DEBUG") != nullptr;
    if (dbg && nu >= 512) {
        uint64_t v[10];
        double turn = 0;
        ggml_rampool_rows_stats_fork(t, v, &turn);
        LLAMA_LOG_WARN("%s: %s part %d: %zu rows, %zu in RAM, %zu from the read-ahead (waited %.1f ms for it), %zu read now; %.1f ms. Rows in RAM %llu of %llu, "
                "histories %llu, %.1f MiB, kept %llu, declined %llu, evicted %llu, forgotten %llu, a stay lasts %.0f rows recalled\n", __func__, ggml_get_name(t), part, nu,
                n_ram, mr.size() - n_miss, (t1 - t0) / 1e3, n_miss, (ggml_time_us() - t0) / 1e3, (unsigned long long) v[0], (unsigned long long) v[2],
                (unsigned long long) v[1], v[9] / 1048576.0, (unsigned long long) v[5], (unsigned long long) v[6], (unsigned long long) v[7],
                (unsigned long long) v[8], turn);
    }
    if (!ok) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) { LLAMA_LOG_WARN("%s: a row read of %s failed: this pass takes its rows from the mapping\n", __func__, ggml_get_name(t)); }
        for (size_t k = 0; k < nu; ++k) { memcpy(dst + k*rs, (const uint8_t *) t->data + (size_t) u[k]*rs, rs); }
    }
    return ok;
}

// A pass's rows for a graph that gathers from `tab` (an input of the pass: t's row size, at least as many rows as ids) instead of the mapped t: each distinct row
// among ids[0..n) once, ascending (llama_rowread_fill_rows_fork, gather part 0). Returns them: row k of tab holds row u[k] of t, so a caller maps its ids to tab's
// rows with lower_bound.
std::vector<int32_t> llama_rowread_fill_fork(const ggml_tensor * t, ggml_tensor * tab, const int32_t * ids, size_t n) {
    std::vector<int32_t> u(ids, ids + n);
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    GGML_ASSERT((int64_t) u.size() <= tab->ne[1]);
    const size_t rs = ggml_row_size(tab->type, tab->ne[0]);
    const bool host = ggml_backend_buffer_is_host(tab->buffer);
    // kept per thread, never zeroed: a decode token stages 16 rows (0.5 to 0.6 MB on Qwen3.8) and a new vector was that many bytes of fresh, zero-filled pages
    // on the thread the card waits for; every byte sent below is written by the fill first
    static thread_local std::vector<uint8_t> tmp;
    if (!host && tmp.size() < u.size()*rs) { tmp.resize(u.size()*rs); }
    uint8_t * dst = host ? (uint8_t *) tab->data : tmp.data();
    llama_rowread_fill_rows_fork(t, 0, u.data(), u.size(), dst);
    if (!host) { ggml_backend_tensor_set(tab, tmp.data(), 0, u.size()*rs); }
    return u;
}
