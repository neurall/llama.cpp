// hugefault: SSD-cold file mmap with 2 MB pages (ext4 large folios + MADV_HUGEPAGE) vs 4 KB pages vs io_uring O_DIRECT, one MiMo up tensor (256 experts of 2752512 bytes).
//   T1 sequential: threads touch one byte per page of the whole tensor (reads it in), GB/s of the tensor
//   T2 random experts (64 of 256): fault only, useful GB/s = expert bytes / time   (page-size over-read shows up here)
//   T3 same experts: each thread cudaMemcpyAsync's the expert from the mapping (the driver faults it in) to VRAM
//   T4 same experts: io_uring O_DIRECT reads of exactly the expert into pinned slots, DMA each when it lands
// build: nvcc -O2 -std=c++17 -o hugefault hugefault.cu -lpthread -luring      run: ./hugefault FILE OFFSET BYTES [dev]
#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const size_t HP = 2u << 20, EXP = 2752512;
static size_t g_off, g_len; static int g_fd, g_fdd; static void * g_dev; static uint8_t * g_map, * g_res; static size_t g_resl, g_mapoff;

static double resident_pct();
static long rbytes() { FILE * f = fopen("/proc/self/io", "r"); char l[128]; long v = 0; while (fgets(l, sizeof l, f)) { if (!strncmp(l, "read_bytes", 10)) { v = atol(l + 11); } } fclose(f); return v; }
static void drop() { sync(); posix_fadvise(g_fd, 0, 0, POSIX_FADV_DONTNEED); if (system("sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'")) { fprintf(stderr, "drop_caches failed\n"); }   /* fadvise alone leaves PMD-mapped large folios cached */ const double r = resident_pct(); if (r > 1.0) { fprintf(stderr, "WARNING: %.1f%% still cached after drop\n", r); } }
static uint8_t * map_tensor(bool huge) {   // returns a pointer at the tensor start; the mapping is 2 MB aligned in both file offset and address
    g_mapoff = g_off / HP * HP; const size_t end = g_off + g_len, mlen = (end - g_mapoff + HP - 1) / HP * HP;
    g_resl = mlen + HP; g_res = (uint8_t *) mmap(nullptr, g_resl, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    g_map = (uint8_t *) (((size_t) g_res + HP - 1) & ~(HP - 1));
    if (mmap(g_map, mlen, PROT_READ, MAP_SHARED | MAP_FIXED, g_fd, (off_t) g_mapoff) == MAP_FAILED) { perror("mmap"); exit(1); }
    if (huge && madvise(g_map, mlen, MADV_HUGEPAGE)) { perror("madvise hugepage"); }
    return g_map + (g_off - g_mapoff);
}
static void unmap_tensor() { munmap(g_res, g_resl); }
static size_t pmd_kb() { FILE * f = fopen("/proc/self/smaps_rollup", "r"); char l[200]; size_t kb = 0; while (fgets(l, sizeof l, f)) { if (!strncmp(l, "FilePmdMapped", 13)) kb = strtoull(l + 14, 0, 10); } fclose(f); return kb; }

// touch every page of [p, p+n) with the given page step, by nthr threads over chunk-aligned work items of `step` bytes
static double touch(const uint8_t * base, const std::vector<std::pair<size_t, size_t>> & ranges, size_t step, int nthr) {
    std::vector<size_t> items; for (auto & r : ranges) { for (size_t o = r.first & ~(step - 1); o < r.first + r.second; o += step) items.push_back(o); }
    std::atomic<size_t> nx{0}; std::vector<std::thread> th; const double t0 = now();
    for (int t = 0; t < nthr; t++) th.emplace_back([&]() { volatile uint8_t s = 0; for (size_t i; (i = nx.fetch_add(1)) < items.size();) { s += base[items[i]]; } });
    for (auto & x : th) { x.join(); }
    return now() - t0;
}
static double resident_pct() {   // share of the tensor range still in the page cache (mincore on a throwaway mapping)
    const size_t a = g_off & ~(size_t) 4095, n = g_len + (g_off - a); void * m = mmap(nullptr, n, PROT_READ, MAP_SHARED, g_fd, (off_t) a); std::vector<unsigned char> v((n + 4095) / 4096);
    mincore(m, n, v.data()); size_t c = 0; for (auto x : v) c += x & 1; munmap(m, n); return 100.0 * c / v.size();
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// correctness: reference bytes read once with O_DIRECT; every method's result is compared to them (outside the timing). g_bad counts mismatches per table row.
static uint8_t * g_ref; static int g_bad; static std::vector<std::pair<size_t, size_t>> g_exps;
static void chk_map_exps(const uint8_t * m) { for (size_t i = 0; i < g_exps.size(); i++) { if (memcmp(m + g_exps[i].first, g_ref + i * EXP, EXP)) { g_bad++; } } }
static void chk_map_samples(const uint8_t * m) {   // 48 spread-out 4 KiB samples of the tensor against an O_DIRECT read of the same bytes
    uint8_t * b = (uint8_t *) aligned_alloc(4096, 4096);
    for (int k = 0; k < 48; k++) { const size_t o = (g_len / 48) * k / 4096 * 4096; if (pread(g_fdd, b, 4096, (off_t) (g_off + o)) != 4096 || memcmp(b, m + o, 4096)) { g_bad++; } }
    free(b);
}
static void chk_dev_exps() { std::vector<uint8_t> h(EXP); for (size_t i = 0; i < g_exps.size(); i++) { CK(cudaMemcpy(h.data(), (uint8_t *) g_dev + g_exps[i].first, EXP, cudaMemcpyDeviceToHost)); if (memcmp(h.data(), g_ref + i * EXP, EXP)) { g_bad++; } } }

int main(int argc, char ** argv) {
    g_off = strtoull(argv[2], 0, 0); g_len = strtoull(argv[3], 0, 0); CK(cudaSetDevice(argc > 4 ? atoi(argv[4]) : 0));
    g_fd = open(argv[1], O_RDONLY); g_fdd = open(argv[1], O_RDONLY | O_DIRECT); CK(cudaMalloc(&g_dev, g_len));
    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }
    const size_t nexp = g_len / EXP;
    std::vector<size_t> ids(nexp); std::iota(ids.begin(), ids.end(), 0); std::mt19937 rng(42); std::shuffle(ids.begin(), ids.end(), rng); ids.resize(128);
    std::vector<std::pair<size_t, size_t>> exps, all = {{0, g_len}}; for (size_t id : ids) { exps.push_back({id * EXP, EXP}); }
    g_exps = exps; g_ref = (uint8_t *) aligned_alloc(4096, exps.size() * EXP);
    for (size_t i = 0; i < exps.size(); i++) { if (pread(g_fdd, g_ref + i * EXP, EXP, (off_t) (g_off + exps[i].first)) != (ssize_t) EXP) { perror("reference read"); return 1; } }
    const double useful = 128.0 * EXP / 1e9;
    printf("tensor %zu bytes at %zu, file offset %% 2MB = %zu; %zu experts of %zu, 128 random ones for T2-T4\n", g_len, g_off, g_off % HP, nexp, EXP);

    printf("\nT1 sequential cold fault of the whole tensor (one touch per page), GB/s, median of 3\n%-22s", "pages \\ threads");
    const int ths[] = {1, 2, 4, 8, 16, 32}; for (int th : ths) printf(" %6d", th); printf("\n");
    for (int huge = 1; huge >= 0; huge--) {
        printf("%-22s", huge ? "2 MB (MADV_HUGEPAGE)" : "4 KB");
        for (int th : ths) { std::vector<double> r; size_t pk = 0; long rd = 0; for (int k = 0; k < 3; k++) { drop(); const uint8_t * m = map_tensor(huge); long b0 = rbytes(); double t = touch(m, all, huge ? HP : 4096, th); pk = pmd_kb() + (rbytes() - b0) / 1024 * 0; rd = rbytes() - b0; chk_map_samples(m); r.push_back((double) g_len / t / 1e9); unmap_tensor(); } printf(" %6.2f", med(r)); if (th == 32) { printf("   (storage read %ld MB for a %zu MB tensor; FilePmdMapped %zu MB)", rd >> 20, g_len >> 20, pk >> 10); } fflush(stdout); }
        printf("   verify: %s\n", g_bad ? "MISMATCH" : "ok"); g_bad = 0;
    }

    printf("\nT2 128 random experts, fault only, useful GB/s (expert bytes / time), median of 3\n%-22s", "pages \\ threads"); for (int th : ths) printf(" %6d", th); printf("\n");
    for (int huge = 1; huge >= 0; huge--) {
        printf("%-22s", huge ? "2 MB (MADV_HUGEPAGE)" : "4 KB");
        for (int th : ths) { std::vector<double> r; long rd = 0; for (int k = 0; k < 3; k++) { drop(); const uint8_t * m = map_tensor(huge); long b0 = rbytes(); double t = touch(m, exps, huge ? HP : 4096, th); rd = rbytes() - b0; chk_map_exps(m); r.push_back(useful / t); unmap_tensor(); } printf(" %6.2f", med(r)); if (th == 32) { printf("   (storage read %ld MB for %.0f MB of experts = %.2fx)", rd >> 20, useful * 1e3 / 1.048576, rd / (useful * 1e9)); } fflush(stdout); }
        printf("   verify: %s\n", g_bad ? "MISMATCH" : "ok"); g_bad = 0;
    }

    printf("\nT3 128 random experts, cudaMemcpyAsync from the mapping (driver faults it in), useful GB/s, median of 3\n%-22s", "pages \\ threads"); for (int th : ths) printf(" %6d", th); printf("\n");
    for (int huge = 1; huge >= 0; huge--) {
        printf("%-22s", huge ? "2 MB (MADV_HUGEPAGE)" : "4 KB");
        for (int th : ths) {
            std::vector<double> r;
            for (int k = 0; k < 3; k++) {
                drop(); CK(cudaMemset(g_dev, 0, g_len)); const uint8_t * m = map_tensor(huge); std::atomic<size_t> nx{0}; std::vector<std::thread> tv; const double t0 = now();
                for (int t = 0; t < th; t++) tv.emplace_back([&]() { cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
                    for (size_t i; (i = nx.fetch_add(1)) < exps.size();) { CK(cudaMemcpyAsync((uint8_t *) g_dev + exps[i].first, m + exps[i].first, EXP, cudaMemcpyHostToDevice, st)); CK(cudaStreamSynchronize(st)); }
                    cudaStreamDestroy(st); });
                for (auto & x : tv) { x.join(); }
                const double dt = now() - t0; chk_dev_exps(); r.push_back(useful / dt); unmap_tensor();
            }
            printf(" %6.2f", med(r)); fflush(stdout);
        }
        printf("   verify: %s\n", g_bad ? "MISMATCH" : "ok"); g_bad = 0;
    }

    printf("\nT4 128 random experts, io_uring O_DIRECT of exactly the expert into pinned slots + DMA, useful GB/s, median of 3 (rows: queue depth per thread)\n%-22s", "depth \\ threads"); const int t4[] = {1, 2, 4, 8, 16}; for (int th : t4) printf(" %6d", th); printf("\n");
    for (int qd : {4, 16, 32}) {
        printf("%-22d", qd);
        for (int th : t4) {
            std::vector<double> r;
            for (int k = 0; k < 3; k++) {
                drop(); CK(cudaMemset(g_dev, 0, g_len)); std::atomic<size_t> nx{0}; std::vector<std::thread> tv; std::atomic<int> ready{0}; std::atomic<bool> go{false}; double t0 = 0;
                for (int t = 0; t < th; t++) tv.emplace_back([&]() {
                    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking)); struct io_uring ring; io_uring_queue_init(qd, &ring, 0);
                    std::vector<uint8_t *> buf(qd); std::vector<cudaEvent_t> ev(qd); std::vector<size_t> ei(qd);
                    for (int s = 0; s < qd; s++) { CK(cudaHostAlloc((void **) &buf[s], EXP + 4096, 0)); CK(cudaEventCreateWithFlags(&ev[s], cudaEventDisableTiming)); CK(cudaEventRecord(ev[s], st)); }
                    ready++; while (!go.load()) { sched_yield(); }   // slots allocated: the clock starts when all threads are ready
                    int inflight = 0;
                    auto sub = [&](int s) { const size_t i = nx.fetch_add(1); if (i >= exps.size()) { return false; } CK(cudaEventSynchronize(ev[s])); ei[s] = i;
                        struct io_uring_sqe * q = io_uring_get_sqe(&ring); io_uring_prep_read(q, g_fdd, buf[s], EXP, g_off + exps[i].first); q->user_data = (unsigned long) s; inflight++; return true; };
                    for (int s = 0; s < qd; s++) { sub(s); } io_uring_submit(&ring);
                    while (inflight > 0) { struct io_uring_cqe * c; io_uring_wait_cqe(&ring, &c); const int s = (int) c->user_data; if (c->res != (int) EXP) { fprintf(stderr, "short read %d\n", c->res); exit(1); } io_uring_cqe_seen(&ring, c); inflight--;
                        CK(cudaMemcpyAsync((uint8_t *) g_dev + exps[ei[s]].first, buf[s], EXP, cudaMemcpyHostToDevice, st)); CK(cudaEventRecord(ev[s], st)); if (sub(s)) { io_uring_submit(&ring); } }
                    CK(cudaStreamSynchronize(st)); io_uring_queue_exit(&ring); for (int s = 0; s < qd; s++) { cudaFreeHost(buf[s]); cudaEventDestroy(ev[s]); } cudaStreamDestroy(st); });
                while (ready.load() < th) { sched_yield(); } t0 = now(); go = true;
                for (auto & x : tv) { x.join(); }
                const double dt = now() - t0; chk_dev_exps(); r.push_back(useful / dt);
            }
            printf(" %6.2f", med(r)); fflush(stdout);
        }
        printf("   verify: %s\n", g_bad ? "MISMATCH" : "ok"); g_bad = 0;
    }
    return 0;
}
