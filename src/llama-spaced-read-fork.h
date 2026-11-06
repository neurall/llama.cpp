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

// THE read: bytes [beg, beg + total) of fd (4096-aligned beg and total for O_DIRECT) into buf, spaced order, kernel AIO when the kernel has it, else pread. False: a read failed, nothing usable.
static inline bool sr_read(int fd, char * buf, size_t beg, size_t total, size_t need) {
#ifdef SR_HAVE_AIO
    if (total >= ((size_t) 512 << 10) && aio_read_range(fd, (uint8_t *) buf, beg, total, need)) { return true; }
#endif
    return sr_pread(fd, buf, beg, total, need);
}
#endif
