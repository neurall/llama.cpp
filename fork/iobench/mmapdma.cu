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
#include <atomic>
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
    char cfg[64];
    void * base; size_t ml; const uint8_t * m; bool ok; double t;

    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }   // warm-up: context + first copy
    // M0: link ceiling (pinned source, 1 MiB chunks, 1 and 4 threads)
    { uint8_t * pin; CK(cudaHostAlloc((void **) &pin, g_len, cudaHostAllocDefault)); m = map_range(&base, &ml); memcpy(pin, m, g_len); munmap(base, ml);
      for (int th : {1, 4}) { t = pipeline(3, 1 << 20, th, 2, pin); snprintf(cfg, sizeof cfg, "1M x%d thr", th); row("M0 pinned", cfg, "-", t, verify(pin)); }
      cudaFreeHost(pin); }

    for (int rep = 0; rep < 2; rep++) {   // twice: the second pass shows run-to-run spread; cold first inside each
        printf("--- repeat %d\n", rep + 1);
        for (int hot = 0; hot < 2; hot++) {
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
        for (size_t ch : {(size_t) 256 << 10, (size_t) 1 << 20}) {
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
    }
    return 0;
}
