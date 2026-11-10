// SSD read order shared by every place that reads model bytes from the file into RAM (expert slices, big ranges, preheat, the pinned model load).
//
// Measured 2026-11-06 on the 2 TB Maxio MAP1602 drive (fork/experiments, runs.db table experiments): reads that are contiguous in the file and in flight together are
// slow on data that is some time old (1.7-2.3 GB/s), while the same reads with any hole between them (4 KiB was enough) run at 3.3-4.0 GB/s, and the same bytes freshly
// rewritten read at 6+ GB/s either way. The Linux block layer sees the same 256 KiB requests in both cases (blktrace), so the cause is inside the drive.
// So a range is read as blocks of RD_BLK with RD_GAP skipped after each, and then the skipped gaps as small reads of their own: no two neighbouring pieces are issued one after the other.
//   LLAMA_MOE_RD_BLK_KB (default 256), LLAMA_MOE_RD_GAP_KB (default 64; 0 = plain blocks in file order, as before; with BLK 1024 that is the old 1 MiB behaviour)
// Offsets are relative to the start of the range; every piece starts on a multiple of 64 KiB, so an O_DIRECT range that starts on a 4096 byte boundary keeps its alignment.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <string>
#ifdef _WIN32
// Windows: the same API on a CRT descriptor (_open_osfhandle), read with overlapped ReadFile; a handle from open_direct has FILE_FLAG_NO_BUFFERING, Windows' O_DIRECT
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#if defined(__linux__) && __has_include(<linux/aio_abi.h>)
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#if defined(SYS_io_getevents)
#define SR_HAVE_AIO 1   // kernel AIO by raw syscalls: no library, passes Docker's default seccomp (io_uring does not)
#endif
#endif
#endif

struct sr_piece { size_t off; size_t n; };

static inline size_t sr_env_kb(const char * name, size_t dflt) {
    const char * v = getenv(name);
    return v && *v ? (size_t) atol(v) << 10 : dflt << 10;
}
// The machine benchmark's picks for this drive (bytes, reads in flight; 0 = not measured, state file [machine] rd_blk_kb / rd_gap_kb / rd_qd). LLAMA_MOE_RD_* (--moe read-blk, read-gap, read-qd) override them.
inline size_t g_sr_blk = 0, g_sr_gap = (size_t) -1; inline int g_sr_qd = 0;
static inline size_t sr_blk() { static const size_t e = sr_env_kb("LLAMA_MOE_RD_BLK_KB", 0); return std::max<size_t>(e ? e : g_sr_blk ? g_sr_blk : (size_t) 256 << 10, 65536); }
static inline size_t sr_gap() { static const size_t e = sr_env_kb("LLAMA_MOE_RD_GAP_KB", (size_t) -1 >> 10); return e != ((size_t) -1 >> 10) << 10 ? e : g_sr_gap != (size_t) -1 ? g_sr_gap : (size_t) 64 << 10; }
static inline int    sr_qd()  { static const int e = getenv("LLAMA_MOE_RD_QD") ? atoi(getenv("LLAMA_MOE_RD_QD")) : 0; return e > 0 ? e : g_sr_qd > 0 ? g_sr_qd : 32; }

// the pieces of [0, total) in read order
static inline std::vector<sr_piece> sr_pieces(size_t total, size_t blk = sr_blk(), size_t gap = sr_gap()) {
    std::vector<sr_piece> p;
    if (gap == 0 || total <= blk) {
        for (size_t o = 0; o < total; o += blk) { p.push_back({ o, total - o < blk ? total - o : blk }); }
        return p;
    }
    const size_t period = blk + gap;
    for (size_t o = 0; o < total; o += period) { p.push_back({ o, total - o < blk ? total - o : blk }); }          // pass 1: the blocks
    for (size_t o = blk; o < total; o += period) { p.push_back({ o, total - o < gap ? total - o : gap }); }       // pass 2: the gaps
    return p;
}

// One API for every platform (user 2026-11-07: "one read-code wrapper, multiarch in one place"): the callers name a file by an int descriptor and never see the platform.
//   open_direct(path) / open_direct_fd(fd)  the file past the page cache (Linux O_DIRECT, Windows FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED), -1: not possible here
//   direct_align()                          what such a read needs: buffer, file offset and length on this grid (4096 is a multiple of every sector size in use)
//   sr_pread1                               one positioned read, pread's contract (bytes read, 0 at the end of the file, -1 and errno on an error)
//   async_read_range                        many reads in flight from ONE thread (Linux kernel AIO, Windows overlapped I/O); false: not here or failed, the caller falls back
//   sr_read                                 THE read: the async engine for big ranges, else sr_pread
static constexpr size_t direct_align() { return 4096; }

static inline int open_direct(const char * path) {
#if defined(_WIN32)
    const int nw = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);   // the paths are UTF-8, as ggml_fopen takes them
    if (nw <= 0) { return -1; }
    std::wstring w((size_t) nw, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path, -1, &w[0], nw);
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                 FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) { return -1; }
    const int fd = _open_osfhandle((intptr_t) h, _O_RDONLY);
    if (fd < 0) { CloseHandle(h); }
    return fd;
#elif defined(O_DIRECT)
    return open(path, O_RDONLY | O_DIRECT);
#else
    (void) path;
    return -1;   // macOS and others: no O_DIRECT (F_NOCACHE is a different contract), the callers keep their mapping
#endif
}

// the already open file `fd` once more, past the page cache (the loader and the cache have the file only as a descriptor)
static inline int open_direct_fd(int fd) {
#if defined(_WIN32)
    const HANDLE h0 = (HANDLE) _get_osfhandle(fd);
    if (h0 == INVALID_HANDLE_VALUE) { return -1; }
    const HANDLE h = ReOpenFile(h0, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
    if (h == INVALID_HANDLE_VALUE) { return -1; }
    const int d = _open_osfhandle((intptr_t) h, _O_RDONLY);
    if (d < 0) { CloseHandle(h); }
    return d;
#elif defined(__linux__)
    return open(("/proc/self/fd/" + std::to_string(fd)).c_str(), O_RDONLY | O_DIRECT);
#else
    (void) fd;
    return -1;
#endif
}

#ifdef _WIN32
// a manual-reset event per thread for the one-at-a-time reads: a handle opened overlapped needs one to wait on, and one per call would be a kernel object per read
struct sr_event_t { HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr); ~sr_event_t() { if (h) { CloseHandle(h); } } };
#endif

static inline int64_t sr_pread1(int fd, void * buf, size_t n, size_t off) {
#ifdef _WIN32
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    thread_local sr_event_t ev;
    if (h == INVALID_HANDLE_VALUE || !ev.h) { errno = EBADF; return -1; }
    OVERLAPPED ov; memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD) (uint64_t) off; ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32); ov.hEvent = ev.h;
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD) std::min<size_t>(n, (size_t) 1 << 30), nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) {
        if (GetLastError() == ERROR_HANDLE_EOF) { return 0; }
        errno = EIO; return -1;
    }
    if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
        if (GetLastError() == ERROR_HANDLE_EOF) { return 0; }
        errno = EIO; return -1;
    }
    return (int64_t) got;
#else
    return (int64_t) pread(fd, buf, n, (off_t) off);
#endif
}

// bytes [beg, beg + total) of fd into buf in that order, one read at a time; `need` = bytes that exist (a short piece past it is the end of the file). False: a read failed.
static inline bool sr_pread(int fd, char * buf, size_t beg, size_t total, size_t need) {
    for (const sr_piece & p : sr_pieces(total)) {
        size_t g = 0;
        while (g < p.n) {
            const int64_t r = sr_pread1(fd, buf + p.off + g, p.n - g, beg + p.off + g);
            if (r < 0 && errno == EINTR) { continue; }
            if (r <= 0) { break; }
            g += (size_t) r;
            if (g < p.n && ((p.off + g) & 4095) && (beg & 4095) == 0) { return false; }   // a short read off the block grid: the next one would fail with O_DIRECT
        }
        if (g < p.n && p.off + g < need) { return false; }
    }
    return true;
}

// --moe read-aio=0 turns the async engine off on every platform (it also switches itself off when the system refuses it); on since 2026-11-06: one thread, 32 reads in flight, the spaced order
inline std::atomic<int> g_aio_ok{1};

// kernel AIO, ONE thread, 32 reads in flight (a new one the moment one finishes), the pieces in the spaced order
#ifdef SR_HAVE_AIO
struct aio_ctx_t { aio_context_t ctx = 0; ~aio_ctx_t() { if (ctx) { syscall(SYS_io_destroy, ctx); } } };
// bytes [beg, beg + total) of fd into buf (4096-aligned); `need` = bytes that exist (a short block past it is the end of the file). False: nothing usable, the caller reads again with pread.
static inline bool aio_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need, int qd = sr_qd(), size_t blk = sr_blk(), size_t gap = sr_gap()) {
    constexpr int QDMAX = 32;
    const int QD = std::min(std::max(qd, 1), QDMAX);
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    thread_local aio_ctx_t c;
    if (!c.ctx && syscall(SYS_io_setup, QDMAX, &c.ctx) != 0) { c.ctx = 0; g_aio_ok.store(0); return false; }
    iocb cbs[QDMAX]; iocb * ps[1]; io_event ev[QDMAX];
    int free_slot[QDMAX]; int n_free = QD;
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    const std::vector<sr_piece> pcs = sr_pieces(total, blk, gap);   // blocks with gaps, then the gaps: no two neighbours in flight together
    size_t next = 0; int infl = 0; bool bad = false;
    while (next < pcs.size() || infl > 0) {
        while (!bad && next < pcs.size() && n_free > 0) {
            const int k = free_slot[--n_free];
            const sr_piece & pc = pcs[next]; const size_t want = pc.n;
            memset(&cbs[k], 0, sizeof(iocb));
            cbs[k].aio_lio_opcode = IOCB_CMD_PREAD; cbs[k].aio_fildes = (uint32_t) fd;
            cbs[k].aio_buf = (uint64_t) (uintptr_t) (buf + pc.off); cbs[k].aio_nbytes = want; cbs[k].aio_offset = (int64_t) (beg + pc.off);
            cbs[k].aio_data = (uint64_t) pc.off;
            ps[0] = &cbs[k];
            if (syscall(SYS_io_submit, c.ctx, 1L, ps) != 1) {
                ++n_free; bad = true;
                if (infl == 0 && next == 0) { g_aio_ok.store(0); }   // the very first submit refused: not on this kernel
                break;
            }
            ++next; ++infl;
        }
        if (infl == 0) { break; }
        const long got = syscall(SYS_io_getevents, c.ctx, 1L, (long) QD, ev, (struct timespec *) nullptr);
        if (got < 0) { if (errno == EINTR) { continue; } bad = true; g_aio_ok.store(0); break; }   // a lost context cannot be drained
        for (long e = 0; e < got; ++e) {
            const iocb * cb = (const iocb *) (uintptr_t) ev[e].obj;
            const size_t off = (size_t) ev[e].data, want = cb->aio_nbytes;
            if (ev[e].res < 0 || ((size_t) ev[e].res < want && off + (size_t) ev[e].res < need)) { bad = true; }
            free_slot[n_free++] = (int) (cb - cbs); --infl;
        }
    }
    return !bad;
}
#endif

// n pages of fd, page i of pg bytes at offs[i] (pg-aligned, pg = direct_align()) into buf + i*pg (buf pg-aligned): the gathers of lazy row tables (DeepSeek V4.1
// Engram: 144-byte rows at random places of a 103 GiB table, a 12k-token pass reads about as many pages as rows). Kernel AIO, ONE thread, up to 128 reads in flight,
// submitted in batches; a short read at the end of the file counts as read. False: no async engine here (the caller keeps its own path) or a read failed.
// Windows: overlapped ReadFile, 32 in flight (win_read_range's pattern). False also when AIO is refused here or switched off (--moe read-aio=0): the caller keeps its own path.
static inline bool sr_read_pages(int fd, const uint64_t * offs, size_t n, size_t pg, uint8_t * buf) {
#ifdef SR_HAVE_AIO
    constexpr int QD = 128;
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    struct ctx_t { aio_context_t ctx = 0; ~ctx_t() { if (ctx) { syscall(SYS_io_destroy, ctx); } } };
    thread_local ctx_t c;
    if (!c.ctx && syscall(SYS_io_setup, QD, &c.ctx) != 0) { c.ctx = 0; return false; }
    iocb cbs[QD]; iocb * ps[QD]; io_event ev[QD];
    int free_slot[QD]; int n_free = QD;
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    size_t next = 0; int infl = 0; bool bad = false;
    while (next < n || infl > 0) {
        long np = 0;
        while (!bad && next < n && n_free > 0) {
            const int k = free_slot[--n_free];
            memset(&cbs[k], 0, sizeof(iocb));
            cbs[k].aio_lio_opcode = IOCB_CMD_PREAD; cbs[k].aio_fildes = (uint32_t) fd;
            cbs[k].aio_buf = (uint64_t) (uintptr_t) (buf + next*pg); cbs[k].aio_nbytes = pg; cbs[k].aio_offset = (int64_t) offs[next];
            ps[np++] = &cbs[k]; ++next;
        }
        if (np > 0) {
            const long s = syscall(SYS_io_submit, c.ctx, np, ps);
            const long ok = s < 0 ? 0 : s;
            for (long j = ok; j < np; ++j) { free_slot[n_free++] = (int) (ps[j] - cbs); }   // not taken: back to the free slots, read again
            next -= (size_t) (np - ok);
            infl += (int) ok;
            if (ok == 0) { bad = true; }
        }
        if (infl == 0) { break; }
        const long got = syscall(SYS_io_getevents, c.ctx, 1L, (long) QD, ev, (struct timespec *) nullptr);
        if (got < 0) { if (errno == EINTR) { continue; } g_aio_ok.store(0); return false; }   // a lost context cannot be drained
        for (long e = 0; e < got; ++e) {
            const iocb * cb = (const iocb *) (uintptr_t) ev[e].obj;
            if (ev[e].res <= 0) { bad = true; }
            free_slot[n_free++] = (int) (cb - cbs); --infl;
        }
    }
    return !bad;
#elif defined(_WIN32)
    constexpr int QD = 32;   // WaitForMultipleObjects takes up to 64 handles
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { return false; }
    struct evs_t { HANDLE e[QD]; bool ok = true;
        evs_t() { for (HANDLE & x : e) { x = CreateEventW(nullptr, TRUE, FALSE, nullptr); ok = ok && x; } }
        ~evs_t() { for (HANDLE x : e) { if (x) { CloseHandle(x); } } } };
    thread_local evs_t ev;
    if (!ev.ok) { return false; }
    OVERLAPPED ov[QD]; int act[QD]; HANDLE wh[QD]; int free_slot[QD]; int n_free = QD;
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    size_t next = 0; int infl = 0; bool bad = false;
    auto done = [&](int k, BOOL ok, DWORD got) { if (!ok || got == 0) { bad = true; } free_slot[n_free++] = k; };   // 0 bytes: past the end, or failed
    while (next < n || infl > 0) {
        while (!bad && next < n && n_free > 0) {
            const int k = free_slot[--n_free];
            memset(&ov[k], 0, sizeof(OVERLAPPED));
            ov[k].Offset = (DWORD) offs[next]; ov[k].OffsetHigh = (DWORD) (offs[next] >> 32); ov[k].hEvent = ev.e[k];
            if (ReadFile(h, buf + next*pg, (DWORD) pg, nullptr, &ov[k])) {   // done already
                DWORD got = 0; const BOOL ok = GetOverlappedResult(h, &ov[k], &got, FALSE); done(k, ok, got);
            } else if (GetLastError() == ERROR_IO_PENDING) {
                act[infl++] = k;
            } else {
                done(k, FALSE, 0);
            }
            ++next;
        }
        if (infl == 0) { if (bad || next >= n) { break; } continue; }
        for (int i = 0; i < infl; ++i) { wh[i] = ev.e[act[i]]; }
        const DWORD w = WaitForMultipleObjects((DWORD) infl, wh, FALSE, INFINITE);
        if (w - WAIT_OBJECT_0 < (DWORD) infl) {
            const int i = (int) (w - WAIT_OBJECT_0), k = act[i];
            act[i] = act[--infl];
            DWORD got = 0; const BOOL ok = GetOverlappedResult(h, &ov[k], &got, FALSE); done(k, ok, got);
        } else {   // the wait itself failed: wait for each read on its own so no OVERLAPPED outlives this frame
            for (int i = 0; i < infl; ++i) { DWORD got = 0; GetOverlappedResult(h, &ov[act[i]], &got, TRUE); free_slot[n_free++] = act[i]; }
            infl = 0; bad = true;
        }
    }
    return !bad;
#else
    (void) fd; (void) offs; (void) n; (void) pg; (void) buf;
    return false;
#endif
}

// io_uring by raw syscalls (no liburing), same contract and order as the AIO engine. --moe read-engine=uring (LLAMA_MOE_RD_ENGINE=uring); the default stays kernel AIO,
// which is what the release image allows (Docker's default seccomp blocks io_uring).
#if defined(SR_HAVE_AIO) && defined(SYS_io_uring_setup) && __has_include(<linux/io_uring.h>)
#include <linux/io_uring.h>
#include <sys/mman.h>
#define SR_HAVE_URING 1
inline std::atomic<int> g_uring_ok{1};
struct uring_t {
    int fd = -1; void * sq = nullptr, * cq = nullptr, * sqe = nullptr; size_t sqsz = 0, cqsz = 0, sqesz = 0;
    unsigned * sq_tail = nullptr, * sq_mask = nullptr, * sq_array = nullptr, * cq_head = nullptr, * cq_tail = nullptr, * cq_mask = nullptr;
    io_uring_sqe * sqes = nullptr; io_uring_cqe * cqes = nullptr;
    ~uring_t() {
        if (sqes) { munmap(sqes, sqesz); }
        if (cq) { munmap(cq, cqsz); }
        if (sq) { munmap(sq, sqsz); }
        if (fd >= 0) { close(fd); }
    }
    bool init(unsigned entries) {
        io_uring_params p; memset(&p, 0, sizeof p);
        fd = (int) syscall(SYS_io_uring_setup, entries, &p);
        if (fd < 0) { return false; }
        sqsz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cqsz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        sqesz = p.sq_entries * sizeof(io_uring_sqe);
        sq = mmap(nullptr, sqsz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        cq = mmap(nullptr, cqsz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        sqes = (io_uring_sqe *) mmap(nullptr, sqesz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == (io_uring_sqe *) MAP_FAILED) { sq = cq = nullptr; sqes = nullptr; return false; }
        char * s = (char *) sq, * c = (char *) cq;
        sq_tail = (unsigned *) (s + p.sq_off.tail); sq_mask = (unsigned *) (s + p.sq_off.ring_mask); sq_array = (unsigned *) (s + p.sq_off.array);
        cq_head = (unsigned *) (c + p.cq_off.head); cq_tail = (unsigned *) (c + p.cq_off.tail); cq_mask = (unsigned *) (c + p.cq_off.ring_mask);
        cqes = (io_uring_cqe *) (c + p.cq_off.cqes);
        return true;
    }
};
// bytes [beg, beg + total) of fd (4096-aligned) into buf, 32 reads in flight, the pieces in the spaced order. False: nothing usable, the caller falls back.
static inline bool uring_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need, int qd = sr_qd(), size_t blk = sr_blk(), size_t gap = sr_gap()) {
    constexpr unsigned QDMAX = 32;
    const unsigned QD = (unsigned) std::min(std::max(qd, 1), (int) QDMAX);
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    thread_local uring_t r; thread_local bool made = false;
    if (!made) { made = true; if (!r.init(QDMAX)) { g_uring_ok.store(0); } }
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    const std::vector<sr_piece> pcs = sr_pieces(total, blk, gap);
    size_t offs[QDMAX]; size_t wants[QDMAX]; unsigned free_slot[QDMAX]; unsigned n_free = QD;
    for (unsigned i = 0; i < QD; ++i) { free_slot[i] = i; }
    size_t next = 0; unsigned infl = 0; bool bad = false;
    while (next < pcs.size() || infl > 0) {
        unsigned queued = 0; unsigned tail = *r.sq_tail;
        while (!bad && next < pcs.size() && n_free > 0) {
            const unsigned k = free_slot[--n_free];
            const sr_piece & pc = pcs[next];
            offs[k] = pc.off; wants[k] = pc.n;
            const unsigned idx = tail & *r.sq_mask;
            io_uring_sqe * s = &r.sqes[idx];
            memset(s, 0, sizeof *s);
            s->opcode = IORING_OP_READ; s->fd = fd; s->addr = (uint64_t) (uintptr_t) (buf + pc.off); s->len = (uint32_t) pc.n;
            s->off = (uint64_t) (beg + pc.off); s->user_data = k;
            r.sq_array[idx] = idx;
            ++tail; ++queued; ++next; ++infl;
        }
        if (queued) {
            __atomic_store_n(r.sq_tail, tail, __ATOMIC_RELEASE);
            if (syscall(SYS_io_uring_enter, r.fd, queued, 0u, 0u, nullptr, 0) < 0 && errno != EINTR) { bad = true; g_uring_ok.store(0); break; }
        }
        if (infl == 0) { break; }
        unsigned head = *r.cq_head;
        if (__atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE) == head) {   // nothing done yet: wait for one
            if (syscall(SYS_io_uring_enter, r.fd, 0u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0) < 0 && errno != EINTR) { bad = true; g_uring_ok.store(0); break; }
            continue;
        }
        while (head != __atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE)) {
            const io_uring_cqe * c = &r.cqes[head & *r.cq_mask];
            const unsigned k = (unsigned) c->user_data;
            if (c->res < 0 || ((size_t) c->res < wants[k] && offs[k] + (size_t) c->res < need)) { bad = true; }
            free_slot[n_free++] = k; --infl; ++head;
        }
        __atomic_store_n(r.cq_head, head, __ATOMIC_RELEASE);
    }
    return !bad;
}
#endif

#ifdef _WIN32
// Windows overlapped I/O, the same contract and order as the kernel AIO engine: ONE thread, up to sr_qd() reads in flight, each with its own OVERLAPPED and event, a new one the moment
// one finishes (WaitForMultipleObjects takes up to 64 handles, 32 is the cap). With FILE_FLAG_NO_BUFFERING buf, beg and every piece must sit on the sector grid (direct_align()).
// Every read in flight is waited for before the function returns: the OVERLAPPED blocks live on this stack frame.
static inline bool win_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need, int qd, size_t blk, size_t gap) {
    constexpr int QDMAX = 32;
    const int QD = std::min(std::max(qd, 1), QDMAX);
    const HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { return false; }
    struct evs_t { HANDLE e[QDMAX]; bool ok = true;
        evs_t() { for (HANDLE & x : e) { x = CreateEventW(nullptr, TRUE, FALSE, nullptr); ok = ok && x; } }
        ~evs_t() { for (HANDLE x : e) { if (x) { CloseHandle(x); } } } };
    thread_local evs_t ev;
    if (!ev.ok) { g_aio_ok.store(0); return false; }
    OVERLAPPED ov[QDMAX]; size_t offs[QDMAX], wants[QDMAX]; int act[QDMAX]; HANDLE wh[QDMAX];
    int free_slot[QDMAX]; int n_free = QD;
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    const std::vector<sr_piece> pcs = sr_pieces(total, blk, gap);
    size_t next = 0; int infl = 0; bool bad = false;
    auto done = [&](int k, BOOL ok, DWORD got) {   // one finished read: short only past the end of the file
        if (!ok && GetLastError() != ERROR_HANDLE_EOF) { bad = true; }
        if ((size_t) got < wants[k] && offs[k] + got < need) { bad = true; }
        free_slot[n_free++] = k;
    };
    while (next < pcs.size() || infl > 0) {
        while (!bad && next < pcs.size() && n_free > 0) {
            const int k = free_slot[--n_free];
            const sr_piece & pc = pcs[next++];
            offs[k] = pc.off; wants[k] = pc.n;
            const uint64_t o = (uint64_t) (beg + pc.off);
            memset(&ov[k], 0, sizeof(OVERLAPPED));
            ov[k].Offset = (DWORD) o; ov[k].OffsetHigh = (DWORD) (o >> 32); ov[k].hEvent = ev.e[k];
            if (ReadFile(h, buf + pc.off, (DWORD) pc.n, nullptr, &ov[k])) {   // done already (a handle without FILE_FLAG_OVERLAPPED, or data in the drive's cache)
                DWORD got = 0; const BOOL ok = GetOverlappedResult(h, &ov[k], &got, FALSE); done(k, ok, got);
            } else if (GetLastError() == ERROR_IO_PENDING) {
                act[infl++] = k;
            } else {
                done(k, FALSE, 0);   // ERROR_HANDLE_EOF counts as 0 bytes, anything else as a failed read
            }
        }
        if (bad) { next = pcs.size(); }   // stop submitting; what is in flight is still drained below
        if (infl == 0) { continue; }   // nothing in flight: either more to submit, or the loop ends
        for (int i = 0; i < infl; ++i) { wh[i] = ev.e[act[i]]; }
        const DWORD w = WaitForMultipleObjects((DWORD) infl, wh, FALSE, INFINITE);
        if (w - WAIT_OBJECT_0 < (DWORD) infl) {
            const int i = (int) (w - WAIT_OBJECT_0), k = act[i];
            act[i] = act[--infl];
            DWORD got = 0; const BOOL ok = GetOverlappedResult(h, &ov[k], &got, FALSE); done(k, ok, got);
        } else {   // the wait itself failed: wait for each read on its own so no OVERLAPPED outlives this frame
            for (int i = 0; i < infl; ++i) { DWORD got = 0; GetOverlappedResult(h, &ov[act[i]], &got, TRUE); free_slot[n_free++] = act[i]; }
            infl = 0; bad = true;
        }
    }
    return !bad;
}
#endif

// The async engine of this platform (kernel AIO on Linux, overlapped I/O on Windows, none elsewhere): bytes [beg, beg + total) of a direct descriptor into buf, the spaced order.
// False: no engine here, switched off (--moe read-aio=0), or a read failed; the caller reads again its own way. io_uring is reached only through sr_read (LLAMA_MOE_RD_ENGINE=uring).
static inline bool async_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need, int qd = sr_qd(), size_t blk = sr_blk(), size_t gap = sr_gap()) {
#if defined(SR_HAVE_AIO)
    return aio_read_range(fd, buf, beg, total, need, qd, blk, gap);
#elif defined(_WIN32)
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    return win_read_range(fd, buf, beg, total, need, qd, blk, gap);
#else
    (void) fd; (void) buf; (void) beg; (void) total; (void) need; (void) qd; (void) blk; (void) gap;
    return false;
#endif
}

// THE read: bytes [beg, beg + total) of fd (4096-aligned beg and total for O_DIRECT) into buf, spaced order: io_uring (LLAMA_MOE_RD_ENGINE=uring) or the async engine, else one read at a time.
// False: a read failed, nothing usable.
static inline bool sr_read(int fd, char * buf, size_t beg, size_t total, size_t need) {
#ifdef SR_HAVE_URING
    static const bool uring = [] { const char * e = getenv("LLAMA_MOE_RD_ENGINE"); return e && !strcmp(e, "uring"); }();
    if (uring && total >= ((size_t) 512 << 10) && uring_read_range(fd, (uint8_t *) buf, beg, total, need)) { return true; }
#endif
    if (total >= ((size_t) 512 << 10) && async_read_range(fd, (uint8_t *) buf, beg, total, need)) { return true; }
    return sr_pread(fd, buf, beg, total, need);
}
