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
#include <cstdio>
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
            if (g < p.n && ((p.off + g) & 4095) && (beg & 4095) == 0) {   // a short read off the block grid: the next one would fail with O_DIRECT
                if (p.off + g >= need) { break; }   // ... unless it reached the end of the file with every byte needed (the W10 loader's last chunk, 2026-11-27)
                return false;
            }
        }
        if (g < p.n && p.off + g < need) { return false; }
    }
    return true;
}

// ---- several drives: llama-server --multissd DIR[:SHARE],... (env LLAMA_MOE_SSD; LLAMA_MOE_MIRROR=DIR, the older one-copy name) ----
// A model file may lie once more in other directories, under its name and size, on other drives: whole, or trimmed to a share of every layer's experts (a sparse
// file, what it lacks is a hole: fork/ssd_place.py writes them, SHARE is its; SEEK_DATA tells this reader which bytes a file holds). One read keeps a queue on
// every drive (sr_sched): each drive takes the next piece it holds while it has fewer than qd of this read in flight, the least busy drive first, so N drives read
// side by side and a faster drive, which empties its queue sooner, takes more of the pieces two drives hold. Every reader of this wrapper gets the drives: the
// RAM pool's fill and readers, the stream's reads, the loader. The original stays whole (any piece has a file to go to; the edge sectors and mapped reads only
// know the original): a trimmed original is refused, and a copy whose sampled bytes differ from the original's, or that is sparse where the file system cannot
// say where its holes are, is not used.
// Kimi K3.1 on PC1, 2026-11-20: one drive 5.4-6.1 GB/s; a whole copy on a second drive (4.3 GB/s) through one in-order queue: 3.97 + 3.16 GB/s in the prompt stream.
constexpr int SR_MAX_SRC = 16;   // files of one model file, its own included; also the most drives
struct sr_srcs {
    int      n = 0, n_drives = 1;
    uint64_t size = 0;
    int      fd[SR_MAX_SRC];                                    // [0] is -1: the caller's own descriptor
    int      drive[SR_MAX_SRC];                                 // index of its drive (block device)
    std::vector<std::pair<uint64_t, uint64_t>> held[SR_MAX_SRC];   // the sorted [start, end) ranges the file holds; empty: all of it
};

// [lo, hi) inside one of the sorted ranges r; an empty list holds everything
static inline bool sr_holds(const std::vector<std::pair<uint64_t, uint64_t>> & r, uint64_t lo, uint64_t hi) {
    if (r.empty() || lo >= hi) { return true; }
    const auto it = std::upper_bound(r.begin(), r.end(), lo, [](uint64_t v, const std::pair<uint64_t, uint64_t> & x) { return v < x.first; });
    return it != r.begin() && hi <= std::prev(it)->second;
}

// the file a block goes to: of the n files, those that hold it (holds[i]), the one whose drive has the fewest bytes in flight (q[drive[i]]); ties: the first. -1: none
static inline int sr_pick(int n, const bool * holds, const int * drive, const uint64_t * q) {
    int best = -1;
    for (int i = 0; i < n; ++i) { if (holds[i] && (best < 0 || q[drive[i]] < q[drive[best]])) { best = i; } }
    return best;
}

// "DIR[:SHARE],DIR[:SHARE],..." -> (directory, share of each layer's experts it holds; 0: not given, the placement takes each drive's share of the summed read
// speed). The share is the text after the last ':' when it is a number in (0, 1], so Windows paths (C:\m) keep their drive letter. Trailing separators are dropped.
static inline std::vector<std::pair<std::string, double>> sr_parse_ssd(const char * s) {
    std::vector<std::pair<std::string, double>> v;
    if (!s) { return v; }
    const std::string all(s);
    for (size_t pos = 0; pos < all.size();) {
        size_t c = all.find(',', pos);
        if (c == std::string::npos) { c = all.size(); }
        std::string e = all.substr(pos, c - pos);
        pos = c + 1;
        while (!e.empty() && e.front() == ' ') { e.erase(0, 1); }
        while (!e.empty() && e.back() == ' ') { e.pop_back(); }
        double share = 0.0;
        const size_t k = e.rfind(':');
        if (k != std::string::npos && k + 1 < e.size()) {
            char * end = nullptr;
            const double x = strtod(e.c_str() + k + 1, &end);
            if (end && *end == 0 && x > 0.0 && x <= 1.0) { share = x; e.resize(k); }
        }
        while (e.size() > 1 && (e.back() == '/' || e.back() == '\\')) { e.pop_back(); }
        if (!e.empty()) { v.push_back({ e, share }); }
    }
    return v;
}

inline std::atomic<uint64_t> g_sr_q[SR_MAX_SRC];   // bytes in flight per drive (the readers of this library)
inline thread_local bool g_sr_one_file = false;     // this thread reads only the file it names (the drive shape probe measures one drive)
inline std::atomic<uint64_t> g_sr_b[SR_MAX_SRC];   // bytes read per drive

#if defined(__linux__)
#include <map>
#include <mutex>
#include <sys/stat.h>
inline std::mutex g_sr_mu;
inline std::vector<uint64_t> g_sr_dev;   // drive index -> its block device
inline std::map<std::pair<uint64_t, uint64_t>, sr_srcs *> g_sr_of;   // (device, inode) of a file -> its files elsewhere (nullptr: none); built once, kept

// shortcut: drives are told apart by the file system's device (st_dev), so two partitions of one disk count as two drives; map through /sys/dev/block when that matters
static inline int sr_drive_of(uint64_t dev) {   // under g_sr_mu
    for (size_t i = 0; i < g_sr_dev.size(); ++i) { if (g_sr_dev[i] == dev) { return (int) i; } }
    if (g_sr_dev.size() >= (size_t) SR_MAX_SRC) { return -1; }
    g_sr_dev.push_back(dev);
    return (int) g_sr_dev.size() - 1;
}

// the byte ranges a file holds (SEEK_DATA / SEEK_HOLE); empty: all of it (no hole, or the file system cannot say)
static inline std::vector<std::pair<uint64_t, uint64_t>> sr_data_ranges(int fd, uint64_t size) {
    std::vector<std::pair<uint64_t, uint64_t>> v;
    for (off_t pos = 0; (uint64_t) pos < size;) {
        const off_t d = lseek(fd, pos, SEEK_DATA);
        if (d < 0) {
            if (errno == ENXIO) { break; }
            return {};
        }
        off_t h = lseek(fd, d, SEEK_HOLE);
        if (h < 0) { h = (off_t) size; }
        v.push_back({ (uint64_t) d, (uint64_t) h });
        pos = h;
    }
    if (v.size() == 1 && v[0].first == 0 && v[0].second >= size) { v.clear(); }
    return v;
}

// a copy holds the original's bytes: 4 KiB at the start, middle and end of its first, middle and last range (of the whole file when it holds all) compared
static inline bool sr_same_bytes(int fa, int fb, const std::vector<std::pair<uint64_t, uint64_t>> & held, uint64_t size) {
    std::vector<std::pair<uint64_t, uint64_t>> r = held;
    if (r.empty()) { r.push_back({ 0, size }); }
    void * m = nullptr;
    if (posix_memalign(&m, 4096, 8192) != 0) { return false; }
    char * a = (char *) m, * b = a + 4096;
    bool same = true;
    for (size_t k : { (size_t) 0, r.size() / 2, r.size() - 1 }) {
        const uint64_t lo = (r[k].first + 4095) / 4096 * 4096, hi = r[k].second / 4096 * 4096;
        if (hi < lo + 4096) { continue; }
        for (uint64_t o : { lo, (lo + (hi - lo) / 2) / 4096 * 4096, hi - 4096 }) {
            if (sr_pread1(fa, a, 4096, o) != 4096 || sr_pread1(fb, b, 4096, o) != 4096 || memcmp(a, b, 4096) != 0) { same = false; }
        }
    }
    free(m);
    return same;
}
#endif

// the files of the file behind fd on other drives, once per file (nullptr: no --multissd, or the file lies nowhere else)
static inline const sr_srcs * sr_srcs_of(int fd) {
#if defined(__linux__)
    static const std::vector<std::pair<std::string, double>> dirs = [] {
        const char * s = getenv("LLAMA_MOE_SSD");
        return sr_parse_ssd(s && *s ? s : getenv("LLAMA_MOE_MIRROR"));
    }();
    struct stat st;
    if (dirs.empty() || g_sr_one_file || fd < 0 || fstat(fd, &st) != 0) { return nullptr; }
    std::lock_guard<std::mutex> lk(g_sr_mu);
    const auto key = std::make_pair((uint64_t) st.st_dev, (uint64_t) st.st_ino);
    if (const auto it = g_sr_of.find(key); it != g_sr_of.end()) { return it->second; }
    sr_srcs * s = nullptr;
    char link[64], path[4096];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    const ssize_t pn = readlink(link, path, sizeof path - 1);
    const char * base = pn > 0 ? (path[pn] = 0, strrchr(path, '/')) : nullptr;
    if (base) {
        auto * t = new sr_srcs();
        t->size = (uint64_t) st.st_size;
        t->n = 1; t->fd[0] = -1; t->drive[0] = sr_drive_of((uint64_t) st.st_dev); t->held[0] = sr_data_ranges(fd, t->size);
        if (!t->held[0].empty()) {   // the original must hold every byte: the edge sectors and mapped reads do not come through here yet
            fprintf(stderr, "read: --multissd: %s has holes (a trimmed original): its copies are not used, keep the original whole and trim the copies\n", path);
            delete t;
            g_sr_of[key] = nullptr;
            return nullptr;
        }
        std::string row;
        for (const auto & d : dirs) {
            const std::string p = d.first + base;
            struct stat b;
            if (t->n >= SR_MAX_SRC || stat(p.c_str(), &b) != 0 || (uint64_t) b.st_size != t->size || (b.st_dev == st.st_dev && b.st_ino == st.st_ino)) { continue; }
            const int dr = sr_drive_of((uint64_t) b.st_dev);
            const int f2 = dr >= 0 ? open_direct(p.c_str()) : -1;
            if (f2 < 0) { continue; }
            t->fd[t->n] = f2; t->drive[t->n] = dr; t->held[t->n] = sr_data_ranges(f2, t->size);
            uint64_t hb = 0;
            for (const auto & r : t->held[t->n]) { hb += r.second - r.first; }
            if (t->held[t->n].empty()) { hb = t->size; }
            const char * why = t->held[t->n].empty() && (uint64_t) b.st_blocks * 512 + ((uint64_t) 1 << 20) < t->size
                ? "is sparse, but its file system cannot say where the holes are" : sr_same_bytes(fd, f2, t->held[t->n], t->size) ? nullptr : "differs from the original";
            if (why) {   // its holes would read as zeros, or it is another version of the file: wrong weights, never slower ones
                fprintf(stderr, "read: --multissd: %s %s: not used\n", p.c_str(), why);
                close(f2);
                continue;
            }
            row += " " + p + " (drive " + std::to_string(dr) + ", " + std::to_string((int) (100.0 * (double) hb / (double) std::max<uint64_t>(1, t->size) + 0.5)) + "% of its bytes)";
            t->n++;
        }
        bool seen[SR_MAX_SRC] = {};
        t->n_drives = 0;
        for (int i = 0; i < t->n; ++i) { if (!seen[t->drive[i]]) { seen[t->drive[i]] = true; t->n_drives++; } }
        if (t->n > 1) {
            fprintf(stderr, "read: --multissd: %s is read from %d drives:%s\n", path, t->n_drives, row.c_str());
            s = t;
        } else {
            delete t;
        }
    }
    g_sr_of[key] = s;
    return s;
#else
    (void) fd;   // shortcut: Linux only (readlink of /proc/self/fd, SEEK_DATA); Windows reads its one drive
    return nullptr;
#endif
}

// --multissd given: the async engines size their queues for several drives (128 reads in flight a thread), else for one (32: the AIO contexts every reader
// thread keeps stay as small as before, fs.aio-max-nr counts them)
static inline bool sr_multi_on() {
    static const bool on = [] { const char * s = getenv("LLAMA_MOE_SSD"), * m = getenv("LLAMA_MOE_MIRROR"); return (s && *s) || (m && *m); }();
    return on;
}

// the drives the file behind fd is read from (1: no other)
static inline int sr_drives(int fd) { const sr_srcs * s = sr_srcs_of(fd); return s ? s->n_drives : 1; }

// the descriptor and drive for the block [lo, hi) of the file behind fd: one of its files that holds it, the least busy drive. drive -1: fd itself, no accounting
static inline int sr_src_fd(const sr_srcs * s, int fd, uint64_t lo, uint64_t hi, int * drive) {
    *drive = -1;
    if (!s) { return fd; }
    bool h[SR_MAX_SRC]; uint64_t q[SR_MAX_SRC];
    hi = std::min<uint64_t>(hi, s->size);
    for (int i = 0; i < s->n; ++i) { h[i] = sr_holds(s->held[i], lo, hi); }
    for (int d = 0; d < SR_MAX_SRC; ++d) { q[d] = g_sr_q[d].load(std::memory_order_relaxed); }
    const int i = sr_pick(s->n, h, s->drive, q);
    if (i < 0) { return fd; }   // the original holds everything (sr_srcs_of): not reached
    *drive = s->drive[i];
    return i == 0 ? fd : s->fd[i];
}

// The issue order of one read [beg, beg + total) cut into pieces pcs (the spaced order), over the files of S (nullptr: the one file, pieces in order): next() gives the
// file and piece to read now, from the drives with fewer than per_drive of this read in flight, the one with the fewest bytes in flight (q[drive], every reader)
// first; each drive takes the next piece in order that it holds and no drive took. done() when a read of that file finished.
struct sr_sched {
    const sr_srcs * S; const std::vector<sr_piece> & pcs; uint64_t beg; int per_drive; size_t left;
    std::vector<uint8_t> taken; size_t cur[SR_MAX_SRC] = {}; int on[SR_MAX_SRC] = {};
    sr_sched(const sr_srcs * s, const std::vector<sr_piece> & p, uint64_t b, int pd) : S(s), pcs(p), beg(b), per_drive(std::max(pd, 1)), left(p.size()), taken(s ? p.size() : 0, 0) {}
    int drive(int f) const { return S ? S->drive[f] : 0; }
    bool next(const uint64_t * q, int * file, size_t * piece) {
        int best = -1; size_t bj = 0;
        for (int f = 0; f < (S ? S->n : 1) && left > 0; ++f) {
            if (on[drive(f)] >= per_drive) { continue; }
            size_t j = cur[f];
            if (S) {
                while (j < pcs.size() && (taken[j] || !sr_holds(S->held[f], beg + pcs[j].off, std::min<uint64_t>(beg + pcs[j].off + pcs[j].n, S->size)))) { ++j; }
                cur[f] = j;
            }
            if (j < pcs.size() && (best < 0 || (q && q[drive(f)] < q[drive(best)]))) { best = f; bj = j; }
        }
        if (best < 0) { return false; }
        if (S) { taken[bj] = 1; }
        cur[best] = bj + 1; on[drive(best)]++; left--;
        *file = best; *piece = bj;
        return true;
    }
    void done(int f) { on[drive(f)]--; }
};

// "drive i: X GiB" of every drive this library read from, "" when it read from one
static inline std::string sr_drive_bytes() {
    std::string r;
    int n = 0;
    for (int d = 0; d < SR_MAX_SRC; ++d) { if (g_sr_b[d].load() > 0) { n++; } }
    if (n < 2) { return r; }
    for (int d = 0; d < SR_MAX_SRC; ++d) {
        const uint64_t b = g_sr_b[d].load();
        if (b > 0) { r += (r.empty() ? "" : ", ") + std::string("drive ") + std::to_string(d) + ": " + std::to_string((int) (b / 1073741824.0 * 10 + 0.5) / 10.0).substr(0, 6) + " GiB"; }
    }
    return r;
}

// --moe read-aio=0 turns the async engine off on every platform (it also switches itself off when the system refuses it); on since 2026-11-06: one thread, 32 reads in flight, the spaced order
inline std::atomic<int> g_aio_ok{1};

// kernel AIO, ONE thread, 32 reads in flight on each drive the file is read from (a new one the moment one finishes), the pieces in the spaced order (sr_sched)
#ifdef SR_HAVE_AIO
struct aio_ctx_t { aio_context_t ctx = 0; ~aio_ctx_t() { if (ctx) { syscall(SYS_io_destroy, ctx); } } };
// bytes [beg, beg + total) of fd into buf (4096-aligned); `need` = bytes that exist (a short block past it is the end of the file). False: nothing usable, the caller reads again with pread.
static inline bool aio_read_range(int fd, uint8_t * buf, size_t beg, size_t total, size_t need, int qd = sr_qd(), size_t blk = sr_blk(), size_t gap = sr_gap()) {
    constexpr int QDMAX = 128;   // qd in flight on each drive the file is read from (--multissd: 4 drives at 32; more drives share the 128)
    if (!g_aio_ok.load(std::memory_order_relaxed)) { return false; }
    const int cap = sr_multi_on() ? QDMAX : 32;
    const sr_srcs * S = sr_srcs_of(fd);
    const int nd = S ? S->n_drives : 1, per = std::max(1, std::min(std::max(qd, 1), cap / nd)), QD = std::min(per * nd, cap);
    thread_local aio_ctx_t c;
    if (!c.ctx && syscall(SYS_io_setup, cap, &c.ctx) != 0) { c.ctx = 0; g_aio_ok.store(0); return false; }
    iocb cbs[QDMAX]; iocb * ps[1]; io_event ev[QDMAX];
    int free_slot[QDMAX]; int n_free = QD; int slot_file[QDMAX];
    for (int i = 0; i < QD; ++i) { free_slot[i] = i; }
    const std::vector<sr_piece> pcs = sr_pieces(total, blk, gap);   // blocks with gaps, then the gaps: no two neighbours in flight together
    sr_sched sc(S, pcs, beg, per);
    uint64_t q[SR_MAX_SRC] = {};
    int infl = 0; bool bad = false, sent = false;
    while (sc.left > 0 || infl > 0) {
        int f = 0; size_t j = 0;
        while (!bad && n_free > 0) {
            if (S) { for (int d = 0; d < SR_MAX_SRC; ++d) { q[d] = g_sr_q[d].load(std::memory_order_relaxed); } }
            if (!sc.next(S ? q : nullptr, &f, &j)) { break; }
            const int k = free_slot[--n_free];
            const sr_piece & pc = pcs[j]; const size_t want = pc.n;
            slot_file[k] = f;
            memset(&cbs[k], 0, sizeof(iocb));
            cbs[k].aio_lio_opcode = IOCB_CMD_PREAD; cbs[k].aio_fildes = (uint32_t) (f == 0 ? fd : S->fd[f]);
            cbs[k].aio_buf = (uint64_t) (uintptr_t) (buf + pc.off); cbs[k].aio_nbytes = want; cbs[k].aio_offset = (int64_t) (beg + pc.off);
            cbs[k].aio_data = (uint64_t) pc.off;
            ps[0] = &cbs[k];
            if (syscall(SYS_io_submit, c.ctx, 1L, ps) != 1) {
                ++n_free; sc.done(f); bad = true;
                if (!sent) { g_aio_ok.store(0); }   // the very first submit refused: not on this kernel
                break;
            }
            sent = true; ++infl;
            if (S) { g_sr_q[S->drive[f]].fetch_add(want, std::memory_order_relaxed); }
        }
        if (infl == 0) { bad = bad || sc.left > 0; break; }   // pieces no file holds (not reached: the original holds all of them), or a refused submit
        const long got = syscall(SYS_io_getevents, c.ctx, 1L, (long) QD, ev, (struct timespec *) nullptr);
        if (got < 0) { if (errno == EINTR) { continue; } bad = true; g_aio_ok.store(0); break; }   // a lost context cannot be drained
        for (long e = 0; e < got; ++e) {
            const iocb * cb = (const iocb *) (uintptr_t) ev[e].obj;
            const int k = (int) (cb - cbs);
            const size_t off = (size_t) ev[e].data, want = cb->aio_nbytes;
            if (ev[e].res < 0 || ((size_t) ev[e].res < want && off + (size_t) ev[e].res < need)) { bad = true; }
            sc.done(slot_file[k]);
            if (S) {
                g_sr_q[S->drive[slot_file[k]]].fetch_sub(want, std::memory_order_relaxed);
                if (ev[e].res > 0) { g_sr_b[S->drive[slot_file[k]]].fetch_add((uint64_t) ev[e].res, std::memory_order_relaxed); }
            }
            free_slot[n_free++] = k; --infl;
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
    constexpr unsigned QDMAX = 128;   // as aio_read_range: qd on each drive the file is read from
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    const int cap = sr_multi_on() ? (int) QDMAX : 32;
    const sr_srcs * S = sr_srcs_of(fd);
    const int nd = S ? S->n_drives : 1, per = std::max(1, std::min(std::max(qd, 1), cap / nd));
    const unsigned QD = (unsigned) std::min(per * nd, cap);
    thread_local uring_t r; thread_local bool made = false;
    if (!made) { made = true; if (!r.init((unsigned) cap)) { g_uring_ok.store(0); } }
    if (!g_uring_ok.load(std::memory_order_relaxed)) { return false; }
    const std::vector<sr_piece> pcs = sr_pieces(total, blk, gap);
    sr_sched sc(S, pcs, beg, per);
    uint64_t q[SR_MAX_SRC] = {};
    size_t offs[QDMAX]; size_t wants[QDMAX]; unsigned free_slot[QDMAX]; unsigned n_free = QD; int slot_file[QDMAX];
    for (unsigned i = 0; i < QD; ++i) { free_slot[i] = i; }
    unsigned infl = 0; bool bad = false;
    while (sc.left > 0 || infl > 0) {
        unsigned queued = 0; unsigned tail = *r.sq_tail;
        int f = 0; size_t j = 0;
        while (!bad && n_free > 0) {
            if (S) { for (int d = 0; d < SR_MAX_SRC; ++d) { q[d] = g_sr_q[d].load(std::memory_order_relaxed); } }
            if (!sc.next(S ? q : nullptr, &f, &j)) { break; }
            const unsigned k = free_slot[--n_free];
            const sr_piece & pc = pcs[j];
            offs[k] = pc.off; wants[k] = pc.n; slot_file[k] = f;
            if (S) { g_sr_q[S->drive[f]].fetch_add(pc.n, std::memory_order_relaxed); }
            const unsigned idx = tail & *r.sq_mask;
            io_uring_sqe * s = &r.sqes[idx];
            memset(s, 0, sizeof *s);
            s->opcode = IORING_OP_READ; s->fd = f == 0 ? fd : S->fd[f]; s->addr = (uint64_t) (uintptr_t) (buf + pc.off); s->len = (uint32_t) pc.n;
            s->off = (uint64_t) (beg + pc.off); s->user_data = k;
            r.sq_array[idx] = idx;
            ++tail; ++queued; ++infl;
        }
        if (queued) {
            __atomic_store_n(r.sq_tail, tail, __ATOMIC_RELEASE);
            if (syscall(SYS_io_uring_enter, r.fd, queued, 0u, 0u, nullptr, 0) < 0 && errno != EINTR) { bad = true; g_uring_ok.store(0); break; }
        }
        if (infl == 0) { bad = bad || sc.left > 0; break; }
        unsigned head = *r.cq_head;
        if (__atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE) == head) {   // nothing done yet: wait for one
            if (syscall(SYS_io_uring_enter, r.fd, 0u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0) < 0 && errno != EINTR) { bad = true; g_uring_ok.store(0); break; }
            continue;
        }
        while (head != __atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE)) {
            const io_uring_cqe * c = &r.cqes[head & *r.cq_mask];
            const unsigned k = (unsigned) c->user_data;
            if (c->res < 0 || ((size_t) c->res < wants[k] && offs[k] + (size_t) c->res < need)) { bad = true; }
            sc.done(slot_file[k]);
            if (S) {
                g_sr_q[S->drive[slot_file[k]]].fetch_sub(wants[k], std::memory_order_relaxed);
                if (c->res > 0) { g_sr_b[S->drive[slot_file[k]]].fetch_add((uint64_t) c->res, std::memory_order_relaxed); }
            }
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
    int d = -1;   // one read at a time: from the least busy drive that holds the whole span (--multissd), the original when that read fails
    const int f = sr_src_fd(sr_srcs_of(fd), fd, beg, beg + total, &d);
    if (d >= 0) { g_sr_q[d].fetch_add(total, std::memory_order_relaxed); }
    const bool ok = sr_pread(f, buf, beg, total, need);
    if (d >= 0) { g_sr_q[d].fetch_sub(total, std::memory_order_relaxed); if (ok) { g_sr_b[d].fetch_add(total, std::memory_order_relaxed); } }
    return ok || (f != fd && sr_pread(fd, buf, beg, total, need));
}
