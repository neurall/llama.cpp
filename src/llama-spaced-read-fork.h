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
#ifndef _WIN32
#include <cerrno>
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
static inline size_t sr_blk() { static const size_t v = std::max<size_t>(sr_env_kb("LLAMA_MOE_RD_BLK_KB", 256), 65536); return v; }
static inline size_t sr_gap() { static const size_t v = sr_env_kb("LLAMA_MOE_RD_GAP_KB", 64); return v; }

// the pieces of [0, total) in read order
static inline std::vector<sr_piece> sr_pieces(size_t total) {
    const size_t blk = sr_blk(), gap = sr_gap();
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

#ifndef _WIN32
// bytes [beg, beg + total) of fd into buf in that order with pread; `need` = bytes that exist (a short piece past it is the end of the file). False: a read failed.
static inline bool sr_pread(int fd, char * buf, size_t beg, size_t total, size_t need) {
    for (const sr_piece & p : sr_pieces(total)) {
        size_t g = 0;
        while (g < p.n) {
            const ssize_t r = pread(fd, buf + p.off + g, p.n - g, (off_t) (beg + p.off + g));
            if (r < 0 && errno == EINTR) { continue; }
            if (r <= 0) { break; }
            g += (size_t) r;
            if (g < p.n && ((p.off + g) & 4095) && (beg & 4095) == 0) { return false; }   // a short read off the block grid: the next one would fail with O_DIRECT
        }
        if (g < p.n && p.off + g < need) { return false; }
    }
    return true;
}
#endif

#ifndef _WIN32
// kernel AIO, ONE thread, 32 reads in flight (a new one the moment one finishes), the pieces in the spaced order
inline std::atomic<int> g_aio_ok{1};   // --moe read-aio=0 turns the async engine off (it also switches itself off when the kernel refuses it); on since 2026-11-06: one thread, 32 reads in flight, in the spaced order of llama-spaced-read-fork.h
#ifdef SR_HAVE_AIO
struct aio_ctx_t { aio_context_t ctx = 0; ~aio_ctx_t() { if (ctx) { syscall(SYS_io_destroy, ctx); } } };
// bytes [beg, beg + total) of fd into buf (4096-aligned); `need` = bytes that exist (a short block past it is the end of the file). False: nothing usable, the caller reads again with pread.
static inline bool aio_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need) {
    constexpr int QD = 32;
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    thread_local aio_ctx_t c;
    if (!c.ctx && syscall(SYS_io_setup, QD, &c.ctx) != 0) { c.ctx = 0; g_aio_ok.store(0); return false; }
    iocb cbs[QD]; iocb * ps[1]; io_event ev[QD];
    int free_slot[QD]; int n_free = QD;
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    const std::vector<sr_piece> pcs = sr_pieces(total);   // blocks with gaps, then the gaps: no two neighbours in flight together
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
static inline bool uring_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need) {
    constexpr unsigned QD = 32;
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    thread_local uring_t r; thread_local bool made = false;
    if (!made) { made = true; if (!r.init(QD)) { g_uring_ok.store(0); } }
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    const std::vector<sr_piece> pcs = sr_pieces(total);
    size_t offs[QD]; size_t wants[QD]; unsigned free_slot[QD]; unsigned n_free = QD;
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

// THE read: bytes [beg, beg + total) of fd (4096-aligned beg and total for O_DIRECT) into buf, spaced order: io_uring or kernel AIO (LLAMA_MOE_RD_ENGINE=uring picks io_uring), else pread. False: a read failed, nothing usable.
static inline bool sr_read(int fd, char * buf, size_t beg, size_t total, size_t need) {
#ifdef SR_HAVE_URING
    static const bool uring = [] { const char * e = getenv("LLAMA_MOE_RD_ENGINE"); return e && !strcmp(e, "uring"); }();
    if (uring && total >= ((size_t) 512 << 10) && uring_read_range(fd, (uint8_t *) buf, beg, total, need)) { return true; }
#endif
#ifdef SR_HAVE_AIO
    if (total >= ((size_t) 512 << 10) && aio_read_range(fd, (uint8_t *) buf, beg, total, need)) { return true; }
#endif
    return sr_pread(fd, buf, beg, total, need);
}
#endif
