// mmapdma: how fast does one MoE up tensor (MiMo: 704643072 bytes) reach VRAM from a file, cold and hot page cache?
//   M1 mmap-pageable   cudaMemcpy straight from the mmap (the driver stages it through its own pinned buffers; cold = page faults)
//   M2 mmap-register   cudaHostRegister the mmap, then one cudaMemcpyAsync (register time counted)
//   M3 direct          O_DIRECT pread into pinned chunks, DMA each chunk (SSD -> pinned -> VRAM; no page cache; no GPUDirect: nvidia_fs is not loaded)
//   M4 hybrid          threads memcpy chunks from the mmap into pinned chunks, DMA each chunk when full (the proposal)
//   M5 pread           buffered pread into pinned chunks, DMA each chunk
//   M0 pinned          pinned RAM -> VRAM, the link ceiling
// Each pipeline thread owns SLOTS pinned chunks and one stream; a slot is refilled after its DMA event completed (reads overlap DMA).
// Cold = posix_fadvise(DONTNEED) on the range before a fresh mmap; hot = the run right after, pages cached.
// build: nvcc -O2 -o mmapdma mmapdma.cu -lpthread      run: ./mmapdma FILE OFFSET BYTES [dev]
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define _GNU_SOURCE
#include <liburing.h>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#include <algorithm>
#include <atomic>
#include <csignal>
#include <sched.h>
#include <sys/syscall.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static const char * g_file; static size_t g_off, g_len; static void * g_dev; static int g_fd, g_fdd;

static void drop_cache() {
    sync();
    posix_fadvise(g_fd, (off_t) g_off, (off_t) g_len, POSIX_FADV_DONTNEED);
}
static const uint8_t * map_range(void ** base, size_t * mlen) {
    const size_t pg = 4096, a = g_off / pg * pg;
    *mlen = g_len + (g_off - a);
    *base = mmap(nullptr, *mlen, PROT_READ | PROT_WRITE, MAP_PRIVATE, g_fd, (off_t) a);
    if (*base == MAP_FAILED) { perror("mmap"); exit(1); }
    return (const uint8_t *) *base + (g_off - a);
}
static bool verify(const uint8_t * ref) {   // first, middle and last 1 MiB of the device copy against the file
    std::vector<uint8_t> h(1 << 20);
    for (size_t o : {(size_t) 0, g_len / 2 / 4096 * 4096, g_len - (1 << 20)}) {
        CK(cudaMemcpy(h.data(), (uint8_t *) g_dev + o, 1 << 20, cudaMemcpyDeviceToHost));
        if (memcmp(h.data(), ref + o, 1 << 20)) { return false; }
    }
    return true;
}

// mode: 0 memcpy from the mmap, 1 buffered pread, 2 O_DIRECT pread, 3 from a pinned source buffer (M0)
static double pipeline(int mode, size_t chunk, int nthr, int slots, const uint8_t * src) {
    const size_t nch = (g_len + chunk - 1) / chunk;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    const double t0 = now();
    for (int t = 0; t < nthr; t++) {
        th.emplace_back([&, t]() {
            cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            std::vector<uint8_t *> buf(slots); std::vector<cudaEvent_t> ev(slots);
            for (int s = 0; s < slots; s++) {
                CK(cudaHostAlloc((void **) &buf[s], chunk, cudaHostAllocDefault));
                CK(cudaEventCreateWithFlags(&ev[s], cudaEventDisableTiming));
                CK(cudaEventRecord(ev[s], st));
            }
            if (mode == 4) {   // io_uring, O_DIRECT: `slots` reads in flight per thread, each finished read is DMA'd, its slot is read again once that DMA event is done
                struct io_uring ring; if (io_uring_queue_init(slots, &ring, 0)) { perror("io_uring"); exit(1); }
                std::vector<size_t> ci(slots); int inflight = 0;
                auto submit = [&](int sx) {
                    const size_t i = next.fetch_add(1); if (i >= nch) { return false; }
                    const size_t o = i * chunk, n = std::min(chunk, g_len - o), rn = (n + 4095) / 4096 * 4096;
                    CK(cudaEventSynchronize(ev[sx]));
                    struct io_uring_sqe * q = io_uring_get_sqe(&ring); io_uring_prep_read(q, g_fdd, buf[sx], (unsigned) rn, g_off + o); q->user_data = (unsigned long) sx; ci[sx] = i; inflight++; return true;
                };
                for (int k = 0; k < slots; k++) { submit(k); }
                io_uring_submit(&ring);
                while (inflight > 0) {
                    struct io_uring_cqe * c; if (io_uring_wait_cqe(&ring, &c)) { perror("wait_cqe"); exit(1); }
                    const int sx = (int) c->user_data; const size_t o = ci[sx] * chunk, n = std::min(chunk, g_len - o);
                    if (c->res < (int) n) { fprintf(stderr, "short read %d of %zu\n", c->res, n); exit(1); }
                    io_uring_cqe_seen(&ring, c); inflight--;
                    CK(cudaMemcpyAsync((uint8_t *) g_dev + o, buf[sx], n, cudaMemcpyHostToDevice, st)); CK(cudaEventRecord(ev[sx], st));
                    if (submit(sx)) { io_uring_submit(&ring); }
                }
                io_uring_queue_exit(&ring);
                CK(cudaStreamSynchronize(st));
                for (int k = 0; k < slots; k++) { cudaFreeHost(buf[k]); cudaEventDestroy(ev[k]); }
                cudaStreamDestroy(st);
                return;
            }
            int s = 0;
            for (size_t i; (i = next.fetch_add(1)) < nch; s = (s + 1) % slots) {
                const size_t o = i * chunk, n = std::min(chunk, g_len - o);
                CK(cudaEventSynchronize(ev[s]));
                if (mode == 0) { memcpy(buf[s], src + o, n); }
                else if (mode == 1 || mode == 2) {
                    size_t got = 0;
                    const int fd = mode == 2 ? g_fdd : g_fd;
                    const size_t rn = mode == 2 ? (n + 4095) / 4096 * 4096 : n;   // O_DIRECT: whole sectors (the tail read runs past the range, inside the file)
                    while (got < rn) { const ssize_t r = pread(fd, buf[s] + got, rn - got, (off_t) (g_off + o + got)); if (r <= 0) { perror("pread"); exit(1); } got += (size_t) r; }
                }
                else { /* mode 3: the DMA reads the pinned source in place below */ }
                CK(cudaMemcpyAsync((uint8_t *) g_dev + o, mode == 3 ? src + o : buf[s], n, cudaMemcpyHostToDevice, st));
                CK(cudaEventRecord(ev[s], st));
            }
            CK(cudaStreamSynchronize(st));
            for (int k = 0; k < slots; k++) { cudaFreeHost(buf[k]); cudaEventDestroy(ev[k]); }
            cudaStreamDestroy(st);
        });
    }
    for (auto & x : th) { x.join(); }
    return now() - t0;
}


// M6 fake pointer: userspace paging. A PROT_NONE view (the "fake base", same offsets as in the file range) is handed to cudaMemcpy; the first touch of a
// chunk raises SIGSEGV inside the copy, the handler pread()s that chunk (file offset = range offset + fault offset) into a second RW view of the same memfd and
// opens the first view for reading, then the copy retries. Compared with M4/M5 (the same reads into pinned buffers) this keeps the driver's own staging copy.
static uint8_t * g_fb, * g_fw; static size_t g_fchunk; static std::atomic<int> * g_fst; static std::atomic<long> g_faults{0};
static void segv(int, siginfo_t * si, void *) {
    const uint8_t * a = (const uint8_t *) si->si_addr;
    if (a < g_fb || a >= g_fb + g_len) { signal(SIGSEGV, SIG_DFL); return; }   // not ours: die the normal way
    const size_t k = (size_t) (a - g_fb) / g_fchunk, o = k * g_fchunk, n = std::min(g_fchunk, g_len - o);
    int z = 0;
    if (g_fst[k].compare_exchange_strong(z, 1)) {
        size_t got = 0;
        while (got < n) { const ssize_t r = pread(g_fd, g_fw + o + got, n - got, (off_t) (g_off + o + got)); if (r <= 0) { _exit(3); } got += (size_t) r; }
        mprotect(g_fb + o, n, PROT_READ);   // the chunk becomes visible to every thread only now, fully read
        g_fst[k].store(2);
        g_faults++;
    } else { while (g_fst[k].load() != 2) { sched_yield(); } }
}
static double fake_run(size_t chunk, int nthr, bool reuse, bool * ok) {
    static int mfd = -1;
    if (!reuse) {   // a fresh fake range: nothing loaded
        if (g_fb) { munmap(g_fb, g_len); munmap(g_fw, g_len); delete[] g_fst; close(mfd); }
        mfd = (int) syscall(SYS_memfd_create, "fake", 0); if (mfd < 0 || ftruncate(mfd, (off_t) g_len)) { perror("memfd"); exit(1); }
        if (getenv("PREALLOC")) { fallocate(mfd, 0, 0, (off_t) g_len); }   // pages allocated and zeroed here, not in the fault handler
        g_fw = (uint8_t *) mmap(nullptr, g_len, PROT_READ | PROT_WRITE, MAP_SHARED | (getenv("PREALLOC") ? MAP_POPULATE : 0), mfd, 0);
        g_fb = (uint8_t *) mmap(nullptr, g_len, PROT_NONE, MAP_SHARED, mfd, 0);
        g_fchunk = chunk; g_fst = new std::atomic<int>[(g_len + chunk - 1) / chunk]; for (size_t i = 0; i < (g_len + chunk - 1) / chunk; i++) { g_fst[i] = 0; }
        struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = segv; sa.sa_flags = SA_SIGINFO | SA_NODEFER; sigemptyset(&sa.sa_mask); sigaction(SIGSEGV, &sa, nullptr);
        g_faults = 0;
    }
    std::atomic<size_t> next{0}; const size_t nch = (g_len + chunk - 1) / chunk;
    std::vector<std::thread> th; const double t0 = now();
    for (int t = 0; t < nthr; t++) {
        th.emplace_back([&]() {
            cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            for (size_t i; (i = next.fetch_add(1)) < nch;) {
                const size_t o = i * chunk, n = std::min(chunk, g_len - o);
                CK(cudaMemcpyAsync((uint8_t *) g_dev + o, g_fb + o, n, cudaMemcpyHostToDevice, st));
                CK(cudaStreamSynchronize(st));
            }
            cudaStreamDestroy(st);
        });
    }
    for (auto & x : th) { x.join(); }
    const double t = now() - t0;
    void * base; size_t ml; const uint8_t * m = map_range(&base, &ml); *ok = verify(m); munmap(base, ml);
    return t;
}

// BIGPIN=1: the "RAM pool" case. One big pinned buffer for the whole tensor. Pass 1 streams the file straight into it (each thread reads its chunks to pin+offset, no staging
// copy) and uploads every chunk the moment it landed; pass 2 is DMA from the pinned buffer alone (the pool hit). variants: B1 buffered pread, B2 O_DIRECT pread, B3 io_uring O_DIRECT.
static double bigpass(int var, size_t chunk, int nthr, int qd, uint8_t * pin) {
    const size_t nch = (g_len + chunk - 1) / chunk; std::atomic<size_t> next{0}; std::vector<std::thread> th; const double t0 = now();
    for (int t = 0; t < nthr; t++) th.emplace_back([&]() {
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        if (var == 3) {
            struct io_uring ring; if (io_uring_queue_init(qd, &ring, 0)) { perror("io_uring"); exit(1); }
            int inflight = 0;
            auto sub = [&]() { const size_t i = next.fetch_add(1); if (i >= nch) { return false; } const size_t o = i * chunk, n = std::min(chunk, g_len - o);
                struct io_uring_sqe * q = io_uring_get_sqe(&ring); io_uring_prep_read(q, g_fdd, pin + o, (unsigned) ((n + 4095) / 4096 * 4096), g_off + o); q->user_data = i; inflight++; return true; };
            for (int k = 0; k < qd; k++) { sub(); } io_uring_submit(&ring);
            while (inflight > 0) {
                struct io_uring_cqe * c; if (io_uring_wait_cqe(&ring, &c)) { exit(1); }
                const size_t o = c->user_data * chunk, n = std::min(chunk, g_len - o); if (c->res < (int) n) { fprintf(stderr, "short read\n"); exit(1); }
                io_uring_cqe_seen(&ring, c); inflight--;
                CK(cudaMemcpyAsync((uint8_t *) g_dev + o, pin + o, n, cudaMemcpyHostToDevice, st));
                if (sub()) { io_uring_submit(&ring); }
            }
            io_uring_queue_exit(&ring);
        } else {
            for (size_t i; (i = next.fetch_add(1)) < nch;) {
                const size_t o = i * chunk, n = std::min(chunk, g_len - o), rn = var == 2 ? (n + 4095) / 4096 * 4096 : n; size_t got = 0;
                while (got < rn) { const ssize_t r = pread(var == 2 ? g_fdd : g_fd, pin + o + got, rn - got, (off_t) (g_off + o + got)); if (r <= 0) { perror("pread"); exit(1); } got += (size_t) r; }
                CK(cudaMemcpyAsync((uint8_t *) g_dev + o, pin + o, n, cudaMemcpyHostToDevice, st));
            }
        }
        CK(cudaStreamSynchronize(st)); cudaStreamDestroy(st);
    });
    for (auto & x : th) { x.join(); }
    return now() - t0;
}
static double dma_pass(size_t chunk, int nthr, const uint8_t * pin) {   // the pool hit: pinned -> VRAM only (nthr 0: one cudaMemcpy of everything)
    if (nthr == 0) { const double t0 = now(); CK(cudaMemcpy(g_dev, pin, g_len, cudaMemcpyHostToDevice)); return now() - t0; }
    return pipeline(3, chunk, nthr, 1, pin);
}
static void bigreg();
static void bigpin() {
    if (getenv("BIGREG_ONLY")) { { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); } bigreg(); return; }
    uint8_t * pin; CK(cudaHostAlloc((void **) &pin, g_len, cudaHostAllocDefault));
    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }
    const char * nm[] = {"", "B1 pread", "B2 direct", "B3 uring"};
    printf("%-10s %5s %4s %3s | %-22s | %-24s | pass 2 (pinned -> VRAM only)\n", "pass 1", "chunk", "thr", "qd", "cold stream+upload", "hot (page cache) stream");
    for (int var : {1, 2, 3}) for (size_t ch : {(size_t) 256 << 10, (size_t) 1 << 20, (size_t) 4 << 20}) for (int th : {1, 2, 4, 8, 16}) for (int qd : (var == 3 ? std::vector<int>{4, 16, 32, 64} : std::vector<int>{0})) {
        std::vector<double> c, h;
        for (int r = 0; r < 3; r++) { drop_cache(); memset(pin, 0, 0); c.push_back((double) g_len / bigpass(var, ch, th, qd, pin) / 1e9); }
        bool ok; { void * b; size_t ml; const uint8_t * m = map_range(&b, &ml); ok = verify(m) && !memcmp(pin, m, 1 << 20) && !memcmp(pin + g_len - (1 << 20), m + g_len - (1 << 20), 1 << 20); munmap(b, ml); }
        if (var == 1) { for (int r = 0; r < 3; r++) { h.push_back((double) g_len / bigpass(var, ch, th, qd, pin) / 1e9); } }   // direct variants never use the cache
        std::sort(c.begin(), c.end()); std::sort(h.begin(), h.end());
        double p1 = 0, p4 = 0; { std::vector<double> a, b; for (int r = 0; r < 3; r++) { a.push_back((double) g_len / dma_pass(0, 0, pin) / 1e9); b.push_back((double) g_len / dma_pass(1 << 20, 4, pin) / 1e9); } std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); p1 = a.back(); p4 = b.back(); }
        char hs[40] = "-"; if (!h.empty()) { snprintf(hs, sizeof hs, "%.2f / %.2f / %.2f", h.front(), h[1], h.back()); }
        printf("%-10s %4zuK %4d %3d | %5.2f / %5.2f / %5.2f | %-24s | one memcpy %.1f, 4 thr 1M %.1f GB/s %s\n", nm[var], ch >> 10, th, qd, c.front(), c[1], c.back(), hs, p1, p4, ok ? "" : "DATA MISMATCH");
        fflush(stdout);
    }
    bigreg();
}

// B4 registered mmap: the file range is mapped and cudaHostRegister'd in slices by several threads (the register call faults the pages in and pins them), every slice is
// uploaded right after its register; pass 2 is DMA from the registered mapping alone. Rows: cold and hot page cache, register slice size x threads.
static void bigreg() {
    printf("--- B4 registered mmap by threads: 'reg' = register faults the pages in itself; 'pre+reg' = threads MADV_POPULATE_READ the slices first, then register\n%-8s %4s %-7s | %-5s | %-24s | %-8s | %-8s | pass 2 DMA only from the registered map\n", "slice", "thr", "variant", "temp", "total GB/s (min/med/max)", "pre ms", "unreg ms");
    for (int pre = 0; pre < 2; pre++) for (size_t sl : {(size_t) 16 << 20, (size_t) 64 << 20, g_len}) for (int th : {1, 4, 8, 16}) {
        if (sl == g_len && th > 1) { continue; }
        for (int hot = 0; hot < 2; hot++) {
            std::vector<double> a; double un = 0, p1 = 0, p4 = 0, pop = 0; bool ok = true;
            for (int r = 0; r < 3; r++) {
                if (!hot) { drop_cache(); } else if (r == 0) { void * b; size_t m2; const uint8_t * mm = map_range(&b, &m2); volatile uint8_t sink = 0; for (size_t o = 0; o < g_len; o += 4096) { sink += mm[o]; } munmap(b, m2); }
                void * base; size_t ml; const uint8_t * m = map_range(&base, &ml);
                const size_t ns = (g_len + sl - 1) / sl; std::atomic<size_t> next{0}; std::vector<std::thread> tv; const double t0 = now();
                if (pre) {   // phase 1: the threads read the slices into the page cache and map them (no pinning yet)
                    std::atomic<size_t> pn{0}; std::vector<std::thread> pv;
                    for (int t = 0; t < th; t++) pv.emplace_back([&]() { for (size_t i; (i = pn.fetch_add(1)) < ns;) { const size_t o = i * sl, n = std::min(sl, g_len - o); const size_t a0 = ((size_t) (m + o)) & ~(size_t) 4095, e0 = (((size_t) (m + o + n)) + 4095) & ~(size_t) 4095;   /* the tensor starts 3072 bytes into a page */
                    if (madvise((void *) a0, e0 - a0, MADV_POPULATE_READ)) { perror("madvise"); exit(1); } } });
                    for (auto & x : pv) { x.join(); }
                    pop = (now() - t0) * 1e3;
                }
                for (int t = 0; t < th; t++) tv.emplace_back([&]() {
                    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
                    for (size_t i; (i = next.fetch_add(1)) < ns;) { const size_t o = i * sl, n = std::min(sl, g_len - o);
                        CK(cudaHostRegister((void *) (m + o), n, cudaHostRegisterPortable)); CK(cudaMemcpyAsync((uint8_t *) g_dev + o, m + o, n, cudaMemcpyHostToDevice, st)); }
                    CK(cudaStreamSynchronize(st)); cudaStreamDestroy(st);
                });
                for (auto & x : tv) { x.join(); }
                a.push_back((double) g_len / (now() - t0) / 1e9);
                if (r == 0) { ok = verify(m); }
                // pass 2: DMA only, sequential per slice (one stream) and 4 threads
                double best1 = 0, best4 = 0;
                for (int k = 0; k < 3; k++) {
                    double t1 = now(); for (size_t o = 0; o < g_len; o += sl) { CK(cudaMemcpy((uint8_t *) g_dev + o, m + o, std::min(sl, g_len - o), cudaMemcpyHostToDevice)); } best1 = std::max(best1, (double) g_len / (now() - t1) / 1e9);
                    std::atomic<size_t> nx{0}; std::vector<std::thread> t4; t1 = now();
                    for (int t = 0; t < 4; t++) t4.emplace_back([&]() { for (size_t i; (i = nx.fetch_add(1)) < ns;) { const size_t o = i * sl; CK(cudaMemcpy((uint8_t *) g_dev + o, m + o, std::min(sl, g_len - o), cudaMemcpyHostToDevice)); } });
                    for (auto & x : t4) { x.join(); } best4 = std::max(best4, (double) g_len / (now() - t1) / 1e9);
                }
                p1 = std::max(p1, best1); p4 = std::max(p4, best4);
                const double tu = now(); for (size_t o = 0; o < g_len; o += sl) { cudaHostUnregister((void *) (m + o)); } un = (now() - tu) * 1e3;
                munmap(base, ml);
            }
            std::sort(a.begin(), a.end());
            printf("%5zuM %4d %-7s | %-5s | %5.2f / %5.2f / %5.2f | %6.0f   | %6.0f   | one stream %.1f, 4 threads %.1f GB/s %s\n", sl >> 20, th, pre ? "pre+reg" : "reg", hot ? "hot" : "cold", a.front(), a[1], a.back(), pop, un, p1, p4, ok ? "" : "DATA MISMATCH");
            fflush(stdout);
        }
    }
}

// SWEEP=1: M4 (mmap -> pinned chunks), M5 (pread -> pinned chunks) and M7 (io_uring O_DIRECT) over threads x chunk x slots, REPS runs each, min / median / max GB/s
static void sweep() {
    const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 5;
    const int ths[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
    const size_t chs[] = {64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20};
    void * base; size_t ml;
    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }
    printf("%-4s %-5s %5s %4s %5s | %7s %7s %7s GB/s (min median max of %d)\n", "meth", "temp", "chunk", "thr", "slots", "min", "median", "max", reps);
    std::vector<int> meths = {4, 5, 7}; if (getenv("METHS")) { meths.clear(); for (const char * c = getenv("METHS"); *c; c++) { if (*c >= '0' && *c <= '9') { meths.push_back(*c - '0'); } } }   // METHS=6: O_DIRECT pread only
    for (int meth : meths) {
        for (size_t ch : chs) for (int th : ths) for (int sl : (meth == 7 ? std::vector<int>{2, 4, 8, 16, 32, 64} : std::vector<int>{2, 4})) {
            for (int hot = 0; hot < 2; hot++) {
                if ((meth == 7 || meth == 6) && hot) { continue; }   // O_DIRECT never uses the page cache
                std::vector<double> g; bool ok = true;
                if (hot) { drop_cache(); void * b; size_t m2; const uint8_t * mm = map_range(&b, &m2); volatile uint8_t sink = 0; for (size_t o = 0; o < g_len; o += 4096) { sink += mm[o]; } munmap(b, m2); }
                for (int r = 0; r < reps; r++) {
                    if (!hot) { drop_cache(); }
                    double t;
                    if (meth == 4) { const uint8_t * m = map_range(&base, &ml); t = pipeline(0, ch, th, sl, m); if (r == 0) { ok = verify(m); } munmap(base, ml); }
                    else { t = pipeline(meth == 5 ? 1 : meth == 6 ? 2 : 4, ch, th, sl, nullptr); if (r == 0) { const uint8_t * m = map_range(&base, &ml); ok = verify(m); munmap(base, ml); } }
                    g.push_back((double) g_len / t / 1e9);
                }
                std::sort(g.begin(), g.end());
                printf("M%-3d %-5s %4zuK %4d %5d | %7.2f %7.2f %7.2f %s\n", meth, hot ? "hot" : "cold", ch >> 10, th, sl, g.front(), g[g.size() / 2], g.back(), ok ? "" : "DATA MISMATCH");
                fflush(stdout);
            }
        }
    }
}

static void row(const char * name, const char * cfg, const char * temp, double s, bool ok) {
    printf("%-14s %-22s %-5s %7.0f ms  %6.2f GB/s  %s\n", name, cfg, temp, s * 1e3, (double) g_len / s / 1e9, ok ? "ok" : "DATA MISMATCH");
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s FILE OFFSET BYTES [dev]\n", argv[0]); return 1; }
    g_file = argv[1]; g_off = strtoull(argv[2], 0, 0); g_len = strtoull(argv[3], 0, 0);
    CK(cudaSetDevice(argc > 4 ? atoi(argv[4]) : 0));
    g_fd = open(g_file, O_RDONLY); g_fdd = open(g_file, O_RDONLY | O_DIRECT);
    if (g_fd < 0 || g_fdd < 0) { perror("open"); return 1; }
    CK(cudaMalloc(&g_dev, g_len));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, argc > 4 ? atoi(argv[4]) : 0));
    printf("%s, tensor %zu bytes at %zu of %s\n", p.name, g_len, g_off, g_file);
    if (getenv("SWEEP")) { sweep(); return 0; }
    if (getenv("BIGPIN")) { bigpin(); return 0; }
    char cfg[64];
    void * base; size_t ml; const uint8_t * m; bool ok; double t;

    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }   // warm-up: context + first copy
    // M0: link ceiling (pinned source, 1 MiB chunks, 1 and 4 threads)
    { uint8_t * pin; CK(cudaHostAlloc((void **) &pin, g_len, cudaHostAllocDefault)); m = map_range(&base, &ml); memcpy(pin, m, g_len); munmap(base, ml);
      for (int th : {1, 4}) { t = pipeline(3, 1 << 20, th, 2, pin); snprintf(cfg, sizeof cfg, "1M x%d thr", th); row("M0 pinned", cfg, "-", t, verify(pin)); }
      cudaFreeHost(pin); }

    for (int rep = 0; rep < 2; rep++) {   // twice: the second pass shows run-to-run spread; cold first inside each
        printf("--- repeat %d\n", rep + 1);
        if (!getenv("FAKE_ONLY")) for (int hot = 0; hot < 2; hot++) {
            const char * tmp = hot ? "hot" : "cold";
            if (!hot) { drop_cache(); }
            m = map_range(&base, &ml);
            t = now(); CK(cudaMemcpy(g_dev, m, g_len, cudaMemcpyHostToDevice)); t = now() - t;
            row("M1 mmap-pageable", "one cudaMemcpy", tmp, t, verify(m));
            munmap(base, ml);
            if (!hot) { drop_cache(); }
            m = map_range(&base, &ml);
            t = now(); CK(cudaHostRegister((void *) m, g_len, cudaHostRegisterPortable)); t = now() - t;
            printf("%-14s %-22s %-5s %7.0f ms  (register only)\n", "M2 mmap-register", "cudaHostRegister", tmp, t * 1e3);
            for (int k = 0; k < 3; k++) {   // DMA only, from the registered mmap: 1st may fault cold pages in, later ones are the steady state
                t = now(); CK(cudaMemcpy(g_dev, m, g_len, cudaMemcpyHostToDevice)); t = now() - t;
                snprintf(cfg, sizeof cfg, "DMA only, copy %d", k + 1); row("M2 mmap-register", cfg, tmp, t, verify(m));
            }
            cudaHostUnregister((void *) m); munmap(base, ml);
        }
        if (!getenv("FAKE_ONLY")) for (size_t ch : {(size_t) 256 << 10, (size_t) 1 << 20}) {
            for (int th : {1, 2, 4, 8}) {
                snprintf(cfg, sizeof cfg, "%zuK x%d thr x2 slots", ch >> 10, th);
                drop_cache(); t = pipeline(2, ch, th, 2, nullptr); { m = map_range(&base, &ml); ok = verify(m); munmap(base, ml); } row("M3 direct", cfg, "cold", t, ok);
                for (int hot = 0; hot < 2; hot++) {
                    const char * tmp = hot ? "hot" : "cold";
                    if (!hot) { drop_cache(); }
                    m = map_range(&base, &ml);
                    t = pipeline(0, ch, th, 2, m); ok = verify(m); munmap(base, ml); row("M4 hybrid-mmap", cfg, tmp, t, ok);
                    if (!hot) { drop_cache(); }
                    t = pipeline(1, ch, th, 2, nullptr); m = map_range(&base, &ml); ok = verify(m); munmap(base, ml); row("M5 pread", cfg, tmp, t, ok);
                }
            }
        }

        for (size_t ch : {(size_t) 256 << 10, (size_t) 1 << 20, (size_t) 2752512}) {   // 2752512 = one MiMo up expert (704643072 / 256)
            for (int th : {1, 4, 8}) {
                snprintf(cfg, sizeof cfg, "%zuK x%d thr", ch >> 10, th);
                drop_cache(); t = fake_run(ch, th, false, &ok); row("M6 fake-ptr", cfg, "cold", t, ok);
                t = fake_run(ch, th, true, &ok); row("M6 fake-ptr", "all loaded, copy again", "-", t, ok);   // steady state: pageable copy from resident memory
                drop_cache(); { m = map_range(&base, &ml); (void) m; munmap(base, ml); }
                { void * b2; size_t m2; const uint8_t * mm = map_range(&b2, &m2); volatile uint8_t sink = 0; for (size_t o = 0; o < g_len; o += 4096) { sink += mm[o]; } munmap(b2, m2); }   // warm the page cache
                t = fake_run(ch, th, false, &ok); row("M6 fake-ptr", cfg, "hot", t, ok);
            }
        }
    }
    return 0;
}
