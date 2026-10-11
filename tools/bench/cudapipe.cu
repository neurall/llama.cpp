// cudapipe.cu: the two-GPU prefill pipeline and its building blocks on dummy CUDA, with the real load sizes and timings (no model).
//
//   bench=bw     link bandwidth per GPU (pinned / pageable / file-backed host memory, alone and both GPUs at once, duplex), the stash crossing
//   bench=lat    latency of a small host-to-device copy (24 KB, 4 MB, 50 MB) while a big upload runs on the same card (pageable / pinned, one copy per 1 GB block or chunked)
//   bench=sync   sync and launch costs: stream/event sync, empty kernel round trip, cross-device event wait, queued kernels, cudaMalloc
//   bench=up     upload sweep from file-backed pages (a fresh mmap of file=PATH per test, like the first touch of the mmap'd model): direct = T threads, each one whole-slice
//                cudaMemcpyAsync from the mapped pages; bounce = T threads, each copies C-byte chunks into its own ring of pinned buffers and DMAs them; pinned = the source in pinned
//                memory (upper bound); dma = the bounce ring without the CPU copy (the DMA limit of a chunk size); pread / pread_d / uring / uring_d = the file read straight into the
//                pinned ring (buffered / O_DIRECT; io_uring with registered buffers, reads and DMAs of different buffers overlap). One GPU alone and all GPUs at once. keys: file=PATH region_mb=256 reps=3 threads=1,2,3,4 chunks_kb=64,256,1024,4096,16384,32768 ring=4
//   bench=pipe   the pass pipeline: groups of blocks alternate between the lanes (one GPU each); a lane = a host thread + an uploader thread; every step
//                waits for its weights (uploaded through K slots), for the stash of the group before (pushed through pinned host memory by the other
//                lane), computes (a kernel that spins for the measured time), and pushes its stash on. Reports the wall time and tokens/s, and the
//                one-card run of the same load (lanes=1) for comparison.
//
// build: nvcc -O2 -std=c++17 -arch=sm_86 cudapipe.cu -o cudapipe -lpthread -luring     run: CUDA_DEVICE_ORDER=PCI_BUS_ID ./cudapipe bench=pipe key=value ...
// pipe keys (defaults = full GLM 3.0bit, 12k prompt in 4 parts): blocks=45 dense=3 dsa_every=4 parts=4 gsz=1 w_mb=2550 pieces=3 stash_mb=198
//   t_kda_ms=77 t_dsa_ms=154 (per part-step on lane 0) speed1=1.0 (lane 1 relative speed) dev0=1 dev1=0 (CUDA_DEVICE_ORDER=PCI_BUS_ID: 1 = the x16 card)
//   lanes=2 split=alt|share share1=0.4 slots=9 src=pinned|pageable|file file=PATH host_gb=8 tokens=12416
//   hostwork_ms=0 (CPU time per step before the launch) inputsync=0 (wait for the lane's last step before the next launch: one input copy)
//   replan=0 (a DSA step ends with a stream sync, the allocator's re-plan) kernels=1 (kernels per step) uploader=1 (own thread per lane) reps=1
#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void   cpu_spin_ms(double ms) { const double t0 = now_s(); while ((now_s() - t0) * 1e3 < ms) { } }

static std::map<std::string, std::string> g_kv;
static double      D(const char * k, double def) { auto it = g_kv.find(k); return it == g_kv.end() ? def : atof(it->second.c_str()); }
static std::string S(const char * k, const char * def) { auto it = g_kv.find(k); return it == g_kv.end() ? def : it->second; }

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void spin_kernel(unsigned long long ns) { const unsigned long long t0 = gtimer(); while (gtimer() - t0 < ns) { } }
__global__ void empty_kernel() { }

static int n_sm(int dev) { int v = 0; CK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev)); return v; }

// ---- host memory of the weights: pinned, pageable (malloc), or file-backed pages (like the mmap'd model) ----
struct HostMem { char * p = nullptr; size_t size = 0; std::string kind; };
static HostMem make_host(const std::string & kind, size_t size) {
    HostMem h; h.kind = kind; h.size = size;
    if (kind == "pinned") {
        CK(cudaHostAlloc((void **) &h.p, size, cudaHostAllocPortable));
    } else if (kind == "file") {
        const std::string path = S("file", "/dev/shm/cudapipe.bin");
        // never write into, resize or truncate an existing file: a model passed as file= was cut to host_gb and one byte per page overwritten once. An existing file at least
        // `size` long is mapped read-only and private (the pages come from the page cache, which is what "pageable file" stands for); a shorter one is refused; a missing one is created.
        struct stat st;
        const bool exists = stat(path.c_str(), &st) == 0;
        if (exists && (size_t) st.st_size >= size) {
            int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) { perror("file"); exit(1); }
            h.p = (char *) mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (h.p == MAP_FAILED) { perror("mmap"); exit(1); }
            close(fd);
            for (size_t i = 0; i < size; i += 4096) { (void) *(volatile const char *) (h.p + i); } // touch every page (read only)
            return h;
        }
        if (exists) { fprintf(stderr, "file=%s exists and is shorter than host_gb: refusing to resize it\n", path.c_str()); exit(1); }
        int fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
        if (fd < 0 || ftruncate(fd, size) != 0) { perror("file"); exit(1); }
        h.p = (char *) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (h.p == MAP_FAILED) { perror("mmap"); exit(1); }
        close(fd);
    } else {
        h.p = (char *) malloc(size);
    }
    for (size_t i = 0; i < size; i += 4096) { h.p[i] = (char) (i >> 12); } // touch every page
    return h;
}

// ---- bench=bw ----
static double h2d_once(int dev, void * dst, const char * src, size_t bytes, cudaStream_t st, double * call_ms) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, st));
    const double t0 = now_s();
    CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st));
    if (call_ms) { *call_ms = (now_s() - t0) * 1e3; }
    CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    (void) dev;
    return ms;
}

static void bench_bw() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t mb = (size_t) D("mb", 512) << 20; const int reps = (int) D("reps", 6);
    printf("bench=bw: %d devices, %zu MiB per copy, %d reps\n", nd, mb >> 20, reps);
    std::vector<HostMem> hm; for (const char * k : { "pinned", "pageable", "file" }) { hm.push_back(make_host(k, (size_t) D("host_gb", 4) << 30)); }
    std::vector<void *> dbuf(nd); std::vector<cudaStream_t> st(nd);
    for (int d = 0; d < nd; ++d) { CK(cudaSetDevice(d)); CK(cudaMalloc(&dbuf[d], mb)); CK(cudaStreamCreate(&st[d])); }
    printf("%-34s %8s %10s %12s\n", "test", "GB/s", "ms/copy", "call_ms");
    for (int d = 0; d < nd; ++d) {
        CK(cudaSetDevice(d));
        for (auto & h : hm) {
            double tot = 0, call = 0;
            h2d_once(d, dbuf[d], h.p, mb, st[d], nullptr);
            for (int r = 0; r < reps; ++r) { double c; tot += h2d_once(d, dbuf[d], h.p + (size_t) (r % 3) * mb, mb, st[d], &c); call += c; }
            printf("H2D dev%d %-8s alone            %8.2f %10.1f %12.1f\n", d, h.kind.c_str(), mb * reps / (tot * 1e6), tot / reps, call / reps);
        }
        // D2H pinned
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); CK(cudaEventRecord(a, st[d]));
        for (int r = 0; r < reps; ++r) { CK(cudaMemcpyAsync(hm[0].p, dbuf[d], mb, cudaMemcpyDeviceToHost, st[d])); }
        CK(cudaEventRecord(b, st[d])); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b));
        printf("D2H dev%d pinned alone              %8.2f %10.1f\n", d, mb * reps / (ms * 1e6), ms / reps);
        // duplex: H2D and D2H at the same time on two streams
        cudaStream_t s2; CK(cudaStreamCreate(&s2)); CK(cudaEventRecord(a, st[d]));
        for (int r = 0; r < reps; ++r) { CK(cudaMemcpyAsync(dbuf[d], hm[0].p, mb, cudaMemcpyHostToDevice, st[d])); CK(cudaMemcpyAsync(hm[0].p + mb, dbuf[d], mb, cudaMemcpyDeviceToHost, s2)); }
        CK(cudaEventRecord(b, st[d])); CK(cudaStreamSynchronize(s2)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
        printf("duplex dev%d pinned H2D+D2H         %8.2f %10.1f   (both directions counted)\n", d, 2.0 * mb * reps / (ms * 1e6), ms / reps);
    }
    // all GPUs at once, per host kind: one host thread per GPU
    for (auto & h : hm) {
        std::vector<double> gbs(nd, 0), call(nd, 0); std::vector<std::thread> th; std::atomic<int> go{0};
        for (int d = 0; d < nd; ++d) {
            th.emplace_back([&, d]() {
                CK(cudaSetDevice(d)); h2d_once(d, dbuf[d], h.p, mb, st[d], nullptr);
                go++; while (go < nd) { }
                double tot = 0, c = 0, cc;
                const double t0 = now_s();
                for (int r = 0; r < reps; ++r) { tot += h2d_once(d, dbuf[d], h.p + (size_t) ((r + d) % 3) * mb, mb, st[d], &cc); c += cc; }
                gbs[d] = mb * reps / ((now_s() - t0) * 1e9); call[d] = c / reps;
            });
        }
        for (auto & t : th) { t.join(); }
        for (int d = 0; d < nd; ++d) { printf("H2D dev%d %-8s ALL GPUs at once %8.2f %10s %12.1f\n", d, h.kind.c_str(), gbs[d], "", call[d]); }
    }
    // stash crossing through pinned host memory: D2H on dev a, H2D on dev b, ordered by events; and cudaMemcpyPeer
    if (nd >= 2) {
        const size_t sb = (size_t) D("stash_mb", 198) << 20;
        for (int a = 0; a < 2; ++a) {
            const int b = 1 - a; void *da, *db; CK(cudaSetDevice(a)); CK(cudaMalloc(&da, sb)); CK(cudaSetDevice(b)); CK(cudaMalloc(&db, sb));
            cudaStream_t sa, sb_; cudaEvent_t ev; CK(cudaSetDevice(a)); CK(cudaStreamCreate(&sa)); CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            CK(cudaSetDevice(b)); CK(cudaStreamCreate(&sb_));
            double t_host = 0, t_peer = 0;
            for (int r = 0; r < reps + 1; ++r) {
                double t0 = now_s();
                CK(cudaSetDevice(a)); CK(cudaMemcpyAsync(hm[0].p, da, sb, cudaMemcpyDeviceToHost, sa)); CK(cudaEventRecord(ev, sa));
                CK(cudaSetDevice(b)); CK(cudaStreamWaitEvent(sb_, ev, 0)); CK(cudaMemcpyAsync(db, hm[0].p, sb, cudaMemcpyHostToDevice, sb_)); CK(cudaStreamSynchronize(sb_));
                if (r) { t_host += now_s() - t0; }
                t0 = now_s(); CK(cudaMemcpyPeerAsync(db, b, da, a, sb, sb_)); CK(cudaStreamSynchronize(sb_));
                if (r) { t_peer += now_s() - t0; }
            }
            printf("crossing dev%d -> dev%d %zu MiB: via pinned host %7.1f ms (%.2f GB/s end to end), cudaMemcpyPeer %7.1f ms\n", a, b, sb >> 20, t_host / reps * 1e3,
                   sb * reps / (t_host * 1e9), t_peer / reps * 1e3);
        }
    }
}

// ---- bench=sync ----
static void bench_sync() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const int N = (int) D("n", 2000);
    printf("bench=sync: %d devices, %d iterations\n", nd, N);
    for (int d = 0; d < nd; ++d) {
        CK(cudaSetDevice(d)); cudaStream_t s; CK(cudaStreamCreate(&s)); cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        empty_kernel<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s));
        double t0 = now_s(); for (int i = 0; i < N; ++i) { CK(cudaStreamSynchronize(s)); } const double t_sync = (now_s() - t0) / N;
        t0 = now_s(); for (int i = 0; i < N; ++i) { CK(cudaEventRecord(e, s)); CK(cudaEventSynchronize(e)); } const double t_ev = (now_s() - t0) / N;
        t0 = now_s(); for (int i = 0; i < N; ++i) { empty_kernel<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s)); } const double t_rt = (now_s() - t0) / N;
        t0 = now_s(); for (int i = 0; i < N; ++i) { empty_kernel<<<1, 32, 0, s>>>(); } const double t_launch = (now_s() - t0) / N; CK(cudaStreamSynchronize(s));
        t0 = now_s(); for (int i = 0; i < N; ++i) { empty_kernel<<<n_sm(d), 256, 0, s>>>(); } CK(cudaStreamSynchronize(s)); const double t_q = (now_s() - t0) / N;
        // a busy stream: sync while a 5 ms kernel runs
        t0 = now_s(); spin_kernel<<<n_sm(d), 32, 0, s>>>(5000000ull); CK(cudaStreamSynchronize(s)); const double t_5 = now_s() - t0;
        void * p; t0 = now_s(); CK(cudaMalloc(&p, (size_t) 1296 << 20)); const double t_malloc = now_s() - t0; t0 = now_s(); CK(cudaFree(p)); const double t_free = now_s() - t0;
        t0 = now_s(); CK(cudaMalloc(&p, (size_t) 1296 << 20)); CK(cudaMemsetAsync(p, 0, (size_t) 1296 << 20, s)); CK(cudaStreamSynchronize(s)); const double t_set = now_s() - t0; CK(cudaFree(p));
        printf("dev%d: idle stream sync %6.2f us, event record+sync %6.2f us, empty kernel + sync %6.2f us, launch (async) %6.2f us, queued %6.2f us each,"
               " 5 ms spin kernel + sync %7.3f ms, cudaMalloc 1296 MiB %6.2f ms, cudaFree %6.2f ms, malloc+memset %6.2f ms\n", d, t_sync * 1e6, t_ev * 1e6, t_rt * 1e6,
               t_launch * 1e6, t_q * 1e6, t_5 * 1e3, t_malloc * 1e3, t_free * 1e3, t_set * 1e3);
    }
    if (nd >= 2) { // a stream on dev b waits for an event recorded on dev a after a 2 ms kernel: latency of the cross-device ordering
        cudaStream_t sa, sb; cudaEvent_t ea; CK(cudaSetDevice(0)); CK(cudaStreamCreate(&sa)); CK(cudaEventCreateWithFlags(&ea, cudaEventDisableTiming)); CK(cudaSetDevice(1)); CK(cudaStreamCreate(&sb));
        double tot = 0, tot0 = 0;
        for (int i = 0; i < 200; ++i) {
            CK(cudaSetDevice(0)); const double t0 = now_s(); spin_kernel<<<n_sm(0), 32, 0, sa>>>(2000000ull); CK(cudaEventRecord(ea, sa));
            CK(cudaSetDevice(1)); CK(cudaStreamWaitEvent(sb, ea, 0)); empty_kernel<<<1, 32, 0, sb>>>(); CK(cudaStreamSynchronize(sb)); tot += now_s() - t0;
            CK(cudaSetDevice(0)); const double t1 = now_s(); spin_kernel<<<n_sm(0), 32, 0, sa>>>(2000000ull); CK(cudaStreamSynchronize(sa)); tot0 += now_s() - t1;
        }
        printf("cross-device order: 2 ms kernel on dev0, then an empty kernel on dev1 after cudaStreamWaitEvent: %.3f ms (the kernel alone with sync: %.3f ms)\n", tot / 200 * 1e3, tot0 / 200 * 1e3);
    }
}

// ---- bench=pipe ----
struct Grp { int b0, b1, lane; double t_ms[2]; bool moe, dsa; int np; };
struct Gate { std::mutex m; std::condition_variable cv; std::vector<char> v;
    void init(size_t n) { v.assign(n, 0); }
    void set(size_t i) { { std::lock_guard<std::mutex> lk(m); v[i] = 1; } cv.notify_all(); }
    void wait(size_t i) { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&] { return v[i] != 0; }); } };

struct LaneS {
    int dev = 0; double speed = 1.0; cudaStream_t comp, copy, cross;
    std::vector<void *> slot; std::vector<cudaEvent_t> ready, freed; std::vector<void *> st_in, st_out;
    std::vector<int> groups; // indices in lane order
    int sms = 0;
};

static double run_pipe(int n_lanes, bool verbose) {
    const int blocks = (int) D("blocks", 45), dense = (int) D("dense", 3), dsa_every = (int) D("dsa_every", 4), P = (int) D("parts", 4), gsz = (int) D("gsz", 1);
    const size_t w_bytes = (size_t) D("w_mb", 2550) << 20; const int pieces = (int) D("pieces", 3); const size_t Wp = w_bytes / pieces; const size_t stash = (size_t) D("stash_mb", 198) << 20;
    const double t_kda = D("t_kda_ms", 77), t_dsa = D("t_dsa_ms", 154), speed1 = D("speed1", 1.0), hostwork = D("hostwork_ms", 0);
    const int K = (int) D("slots", 9), nker = (int) D("kernels", 1); const bool inputsync = D("inputsync", 0) != 0, replan = D("replan", 0) != 0, uploader = D("uploader", 1) != 0;
    const int dev0 = (int) D("dev0", 1), dev1 = (int) D("dev1", 0); const double tokens = D("tokens", 12416);
    const std::string split = S("split", "alt"); const double share1 = D("share1", 0.4);
    // groups from the last block backwards; lane by alternation or by share (smooth weighted round robin), the last group on lane 0
    std::vector<Grp> grp; double c0 = 0, c1 = 0; int turn = 0;
    for (int b1 = blocks; b1 > 0; ) {
        int b0 = std::max(0, b1 - gsz); Grp g{};
        if (b1 <= dense) { b0 = 0; }                  // the dense lead is one group
        else if (b0 < dense) { b0 = dense; }          // a group does not straddle the lead
        g.b0 = b0; g.b1 = b1;
        if (grp.size() > 4096) { fprintf(stderr, "too many groups\n"); exit(1); }
        if (n_lanes == 1) { g.lane = 0; }
        else if (split == "alt") { g.lane = turn++ % 2; }
        else { c0 += 1 - share1; c1 += share1; if (grp.empty() || c0 >= c1) { g.lane = 0; c0 -= 1; } else { g.lane = 1; c1 -= 1; } }
        double t = 0; int np = 0; bool dsa = false;
        for (int b = g.b0; b < g.b1; ++b) { const bool is_dsa = (b % dsa_every) == dsa_every - 1; dsa |= is_dsa; t += is_dsa ? t_dsa : t_kda; np += b >= dense ? pieces : 0; }
        g.t_ms[0] = t; g.t_ms[1] = t / speed1; g.dsa = dsa; g.moe = np > 0; g.np = np; grp.push_back(g); b1 = g.b0;
    }
    std::reverse(grp.begin(), grp.end());
    const int G = (int) grp.size();
    std::vector<LaneS> lane(n_lanes);
    HostMem hm = make_host(S("src", "pinned"), (size_t) D("host_gb", 8) << 30);
    char * hpin_stash = nullptr; CK(cudaHostAlloc((void **) &hpin_stash, (size_t) P * stash, cudaHostAllocPortable));
    for (int L = 0; L < n_lanes; ++L) {
        LaneS & l = lane[L]; l.dev = L == 0 ? dev0 : dev1; l.speed = L == 0 ? 1.0 : speed1; CK(cudaSetDevice(l.dev)); l.sms = n_sm(l.dev);
        CK(cudaStreamCreate(&l.comp)); CK(cudaStreamCreate(&l.copy)); CK(cudaStreamCreate(&l.cross));
        for (int k = 0; k < K; ++k) { void * p; CK(cudaMalloc(&p, Wp)); l.slot.push_back(p); cudaEvent_t a, b; CK(cudaEventCreateWithFlags(&a, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&b, cudaEventDisableTiming)); l.ready.push_back(a); l.freed.push_back(b); }
        for (int u = 0; u < P; ++u) { void *a, *b; CK(cudaMalloc(&a, stash)); CK(cudaMalloc(&b, stash)); l.st_in.push_back(a); l.st_out.push_back(b); }
    }
    for (int g = 0; g < G; ++g) { lane[grp[g].lane].groups.push_back(g); }
    // per-step events and flags
    std::vector<cudaEvent_t> done(G * P), pushev(G * P), pullev(G * P);
    for (int g = 0; g < G; ++g) { for (int u = 0; u < P; ++u) {
        CK(cudaSetDevice(lane[grp[g].lane].dev)); CK(cudaEventCreateWithFlags(&done[g * P + u], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&pushev[g * P + u], cudaEventDisableTiming));
        const int nl = g + 1 < G ? lane[grp[g + 1].lane].dev : lane[grp[g].lane].dev; CK(cudaSetDevice(nl)); CK(cudaEventCreateWithFlags(&pullev[g * P + u], cudaEventDisableTiming)); } }
    Gate issued, pushed, ready_issued, free_issued; issued.init(G * P); pushed.init(G * P);
    // pieces in lane order: piece q of the lane -> slot q % K
    std::vector<std::vector<int>> piece_group(n_lanes); std::vector<std::vector<int>> group_q0(n_lanes, std::vector<int>(G, -1));
    for (int L = 0; L < n_lanes; ++L) { int q = 0; for (int g : lane[L].groups) { group_q0[L][g] = q; for (int i = 0; i < grp[g].np; ++i) { piece_group[L].push_back(g); ++q; } } }
    std::vector<size_t> nq(n_lanes); size_t qmax = 0; for (int L = 0; L < n_lanes; ++L) { nq[L] = piece_group[L].size(); qmax = std::max(qmax, nq[L]); }
    ready_issued.init(n_lanes * (qmax + 1)); free_issued.init(n_lanes * (qmax + 1));
    const auto QI = [&](int L, size_t q) { return (size_t) L * (qmax + 1) + q; };
    const size_t host_pieces = hm.size / Wp;
    std::vector<double> busy(n_lanes, 0), bytes_up(n_lanes, 0), t_end(n_lanes, 0);
    std::atomic<int> go{0};
    const auto upload_piece = [&](int L, size_t q) {
        LaneS & l = lane[L]; const int s = (int) (q % K);
        if (q >= (size_t) K) { free_issued.wait(QI(L, q - K)); CK(cudaStreamWaitEvent(l.copy, l.freed[s], 0)); }
        CK(cudaMemcpyAsync(l.slot[s], hm.p + ((q * 7 + L * 3) % host_pieces) * Wp, Wp, cudaMemcpyHostToDevice, l.copy)); CK(cudaEventRecord(l.ready[s], l.copy));
        bytes_up[L] += Wp; ready_issued.set(QI(L, q));
    };
    std::vector<std::thread> th;
    for (int L = 0; L < n_lanes; ++L) {
        if (uploader) { th.emplace_back([&, L]() { CK(cudaSetDevice(lane[L].dev)); go++; while (go < 2 * n_lanes) { } for (size_t q = 0; q < nq[L]; ++q) { upload_piece(L, q); } }); }
        th.emplace_back([&, L]() {
            LaneS & l = lane[L]; CK(cudaSetDevice(l.dev)); go++; while (go < (uploader ? 2 : 1) * n_lanes) { }
            cudaEvent_t prev = nullptr;
            for (int g : l.groups) {
                const Grp & G_ = grp[g]; const int q0 = group_q0[L][g];
                for (int i = 0; i < G_.np; ++i) { const size_t q = (size_t) q0 + i; if (!uploader) { upload_piece(L, q); } ready_issued.wait(QI(L, q)); CK(cudaStreamWaitEvent(l.comp, l.ready[q % K], 0)); }
                for (int u = 0; u < P; ++u) {
                    const size_t idx = (size_t) g * P + u;
                    if (g > 0 && grp[g - 1].lane != L) { // the stash of the group before comes from the other lane through pinned host memory
                        pushed.wait((size_t) (g - 1) * P + u); CK(cudaStreamWaitEvent(l.cross, pushev[(g - 1) * P + u], 0));
                        CK(cudaMemcpyAsync(l.st_in[u], hpin_stash + (size_t) u * stash, stash, cudaMemcpyHostToDevice, l.cross)); CK(cudaEventRecord(pullev[(g - 1) * P + u], l.cross));
                        CK(cudaStreamWaitEvent(l.comp, pullev[(g - 1) * P + u], 0));
                    } else if (g > 0) { issued.wait((size_t) (g - 1) * P + u); }
                    if (hostwork > 0) { cpu_spin_ms(hostwork); }
                    if (inputsync && prev) { CK(cudaEventSynchronize(prev)); }
                    const unsigned long long ns = (unsigned long long) (G_.t_ms[L] * 1e6 / nker);
                    for (int k = 0; k < nker; ++k) { spin_kernel<<<l.sms, 32, 0, l.comp>>>(ns); }
                    CK(cudaEventRecord(done[idx], l.comp)); prev = done[idx]; busy[L] += G_.t_ms[L] * 1e-3;
                    if (g + 1 < G && grp[g + 1].lane != L) {
                        CK(cudaStreamWaitEvent(l.cross, done[idx], 0)); CK(cudaMemcpyAsync(hpin_stash + (size_t) u * stash, l.st_out[u], stash, cudaMemcpyDeviceToHost, l.cross));
                        CK(cudaEventRecord(pushev[idx], l.cross)); pushed.set(idx);
                    }
                    if (replan && G_.dsa) { CK(cudaStreamSynchronize(l.comp)); }
                    issued.set(idx);
                }
                for (int i = 0; i < G_.np; ++i) { const size_t q = (size_t) q0 + i; CK(cudaEventRecord(l.freed[q % K], l.comp)); free_issued.set(QI(L, q)); }
            }
            CK(cudaStreamSynchronize(l.comp)); t_end[L] = now_s();
        });
    }
    const double t0 = now_s();
    for (auto & t : th) { t.join(); }
    for (int L = 0; L < n_lanes; ++L) { CK(cudaSetDevice(lane[L].dev)); CK(cudaDeviceSynchronize()); }
    const double wall = now_s() - t0;
    if (verbose) {
        int n1 = 0; for (auto & g : grp) { n1 += g.lane == 1 ? g.b1 - g.b0 : 0; }
        printf("RESULT,pipe,lanes=%d,split=%s,share1=%.2f,gsz=%d,parts=%d,slots=%d,src=%s,replan=%d,inputsync=%d,hostwork_ms=%.1f,kernels=%d,uploader=%d,speed1=%.2f,blocks_lane1=%d,wall_s=%.2f,tok_s=%.0f",
               n_lanes, split.c_str(), share1, gsz, P, K, hm.kind.c_str(), (int) replan, (int) inputsync, hostwork, nker, (int) uploader, speed1, n1, wall, tokens / wall);
        for (int L = 0; L < n_lanes; ++L) { printf(",lane%d_compute_s=%.2f,lane%d_up_gb=%.1f,lane%d_end_s=%.2f", L, busy[L], L, bytes_up[L] / 1e9, L, t_end[L] - t0); }
        printf("\n");
    }
    for (int L = 0; L < n_lanes; ++L) { CK(cudaSetDevice(lane[L].dev)); CK(cudaDeviceSynchronize()); for (void * p : lane[L].slot) { cudaFree(p); } for (void * p : lane[L].st_in) { cudaFree(p); } for (void * p : lane[L].st_out) { cudaFree(p); } }
    CK(cudaFreeHost(hpin_stash));
    if (hm.kind == "pinned") { CK(cudaFreeHost(hm.p)); } else if (hm.kind == "file") { munmap(hm.p, hm.size); } else { free(hm.p); }
    return wall;
}

// ---- bench=up ----
static std::vector<size_t> list_of(const char * key, const char * def, size_t mul) {
    std::vector<size_t> v; std::string s = S(key, def); size_t i = 0;
    while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) { j = s.size(); } v.push_back((size_t) atof(s.substr(i, j - i).c_str()) * mul); i = j + 1; }
    return v;
}

struct UpResult { double gbs = 0; double s = 0; };

// one upload of `bytes` from src to the device buffer dst on device dev, with T threads; chunk == 0: each thread one cudaMemcpyAsync of its slice; else a pinned ring per thread
// io: 0 memcpy from the mapped pages (src), 1 pread (buffered) into the pinned buffer, 2 pread O_DIRECT, 3 io_uring READ_FIXED buffered, 4 io_uring READ_FIXED O_DIRECT (file_off: where the region starts in the file)
struct UpFiles { int fd = -1, fdd = -1; off_t off = 0; };
static UpResult upload_run(int dev, void * dst, const char * src, size_t bytes, int T, size_t chunk, int ring, bool skip_copy = false, int io = 0, UpFiles ff = UpFiles()) {
    std::vector<std::thread> th; std::atomic<int> ready{0}, go{0};
    const size_t slice = (bytes / T + 4095) & ~(size_t) 4095;
    double t_start = 0, t_end = 0;
    std::vector<double> t_done(T, 0);
    for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t]() {
            CK(cudaSetDevice(dev));
            cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
            const size_t lo = std::min(bytes, (size_t) t * slice), hi = std::min(bytes, lo + slice);
            std::vector<char *> pin; std::vector<cudaEvent_t> ev;
            if (chunk) {
                for (int k = 0; k < ring; ++k) { char * p; CK(cudaHostAlloc((void **) &p, chunk, cudaHostAllocPortable)); pin.push_back(p); cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); ev.push_back(e); CK(cudaEventRecord(e, st)); }
            }
            ready++; while (go.load() == 0) { }
            if (!chunk) {
                if (hi > lo) { CK(cudaMemcpyAsync((char *) dst + lo, src + lo, hi - lo, cudaMemcpyHostToDevice, st)); }
            } else if (io >= 3) {
                // io_uring: every pinned buffer is FREE -> READING (READ_FIXED submitted) -> DMA (cudaMemcpyAsync issued, event pending) -> FREE; reads and DMAs of different buffers overlap
                struct io_uring ur; if (io_uring_queue_init(ring, &ur, 0) != 0) { fprintf(stderr, "io_uring_queue_init failed\n"); exit(1); }
                std::vector<iovec> iov(ring); for (int k = 0; k < ring; ++k) { iov[k].iov_base = pin[k]; iov[k].iov_len = chunk; }
                if (io_uring_register_buffers(&ur, iov.data(), ring) != 0) { fprintf(stderr, "io_uring_register_buffers failed\n"); exit(1); }
                std::vector<int> state(ring, 0); std::vector<size_t> boff(ring, 0), bn(ring, 0); // 0 free, 1 reading, 2 dma
                size_t next = lo; size_t chunks_left = (hi - lo + chunk - 1) / chunk;
                const int fdx = io == 4 ? ff.fdd : ff.fd;
                while (chunks_left > 0) {
                    bool progress = false;
                    for (int k = 0; k < ring; ++k) {
                        if (state[k] == 0 && next < hi) {
                            const size_t n = std::min(chunk, hi - next);
                            struct io_uring_sqe * sqe = io_uring_get_sqe(&ur);
                            io_uring_prep_read_fixed(sqe, fdx, pin[k], (unsigned) n, ff.off + next, k); sqe->user_data = (unsigned long long) k;
                            state[k] = 1; boff[k] = next; bn[k] = n; next += n; progress = true;
                        }
                    }
                    if (progress) { io_uring_submit(&ur); }
                    struct io_uring_cqe * cqe;
                    while (io_uring_peek_cqe(&ur, &cqe) == 0) {
                        const int k = (int) cqe->user_data;
                        if (cqe->res < 0 || (size_t) cqe->res != bn[k]) { fprintf(stderr, "io_uring read: res %d for %zu bytes (%s)\n", cqe->res, bn[k], cqe->res < 0 ? strerror(-cqe->res) : "short"); exit(1); }
                        io_uring_cqe_seen(&ur, cqe);
                        CK(cudaMemcpyAsync((char *) dst + boff[k], pin[k], bn[k], cudaMemcpyHostToDevice, st)); CK(cudaEventRecord(ev[k], st));
                        state[k] = 2; progress = true;
                    }
                    for (int k = 0; k < ring; ++k) {
                        if (state[k] == 2 && cudaEventQuery(ev[k]) == cudaSuccess) { state[k] = 0; --chunks_left; progress = true; }
                    }
                    if (!progress) { sched_yield(); }
                }
                io_uring_queue_exit(&ur);
            } else {
                int k = 0;
                for (size_t off = lo; off < hi; off += chunk, k = (k + 1) % ring) {
                    const size_t n = std::min(chunk, hi - off);
                    CK(cudaEventSynchronize(ev[k]));               // the DMA that last used this buffer is done
                    if (io == 1 || io == 2) {   // pread of the file region straight into the pinned buffer (buffered: the kernel copies from the page cache; O_DIRECT: the NVMe writes into it)
                        size_t got = 0; const int fdx = io == 2 ? ff.fdd : ff.fd;
                        while (got < n) { ssize_t r = pread(fdx, pin[k] + got, n - got, ff.off + off + got); if (r <= 0) { fprintf(stderr, "pread failed: %s\n", strerror(errno)); exit(1); } got += (size_t) r; }
                    } else
                    if (!skip_copy) { memcpy(pin[k], src + off, n); } // page faults + the CPU copy (mode dma: skipped, the pinned buffer is DMA'd as it is: the DMA limit of this chunk size)
                    CK(cudaMemcpyAsync((char *) dst + off, pin[k], n, cudaMemcpyHostToDevice, st));
                    CK(cudaEventRecord(ev[k], st));
                }
            }
            CK(cudaStreamSynchronize(st));
            t_done[t] = now_s();
            for (auto p : pin) { CK(cudaFreeHost(p)); }
            for (auto e : ev) { CK(cudaEventDestroy(e)); }
            CK(cudaStreamDestroy(st));
        });
    }
    while (ready.load() < T) { }
    t_start = now_s(); go = 1;
    for (auto & x : th) { x.join(); }
    t_end = *std::max_element(t_done.begin(), t_done.end());
    UpResult r; r.s = t_end - t_start; r.gbs = bytes / (r.s * 1e9);
    return r;
}

static int io_of(const std::string & m) { return m == "pread" ? 1 : m == "pread_d" ? 2 : m == "uring" ? 3 : m == "uring_d" ? 4 : 0; }

static void bench_up() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t region = (size_t) D("region_mb", 256) << 20; const int reps = (int) D("reps", 3), ring = (int) D("ring", 4);
    const std::string path = S("file", "/p/bw/data/q5h/glm12/GLM12-3.0bit.gguf");
    std::vector<size_t> Ts = list_of("threads", "1,2,3,4", 1), Cs = list_of("chunks_kb", "64,256,1024,4096,16384,32768", 1024);
    int fd = open(path.c_str(), O_RDONLY); if (fd < 0) { perror("file"); exit(1); }
    int fdd = open(path.c_str(), O_RDONLY | O_DIRECT); if (fdd < 0) { perror("file O_DIRECT"); exit(1); }
    off_t fsize = lseek(fd, 0, SEEK_END);
    printf("bench=up: %d devices, file %s (%.1f GB), %zu MiB per upload, %d reps (each rep a fresh mapping, offsets differ), ring %d\n", nd, path.c_str(), fsize / 1e9, region >> 20, reps, ring);
    // warm the page cache for the regions used (reads the file once, not timed)
    { std::vector<char> tmp(1 << 20); size_t span = std::min<size_t>(fsize, region * (size_t) (reps + 1) * 8); for (size_t o = 0; o + tmp.size() <= span; o += tmp.size()) { if (pread(fd, tmp.data(), tmp.size(), o) < 0) { break; } } }
    std::vector<void *> dbuf(nd);
    for (int d = 0; d < nd; ++d) { CK(cudaSetDevice(d)); CK(cudaMalloc(&dbuf[d], region)); }
    HostMem pinned = make_host("pinned", region);
    printf("%-9s %-4s %-9s | %s\n", "mode", "thr", "chunk", "GB/s per GPU alone (mean of reps) | both GPUs at once (each)");
    struct Cfg { std::string mode; int T; size_t chunk; };
    std::vector<Cfg> cfgs;
    cfgs.push_back({ "pinned", 1, 0 });
    for (size_t T : Ts) { cfgs.push_back({ "direct", (int) T, 0 }); }
    for (size_t T : Ts) { for (size_t C : Cs) { cfgs.push_back({ "dma", (int) T, C }); } }
    for (size_t T : Ts) { for (size_t C : Cs) { cfgs.push_back({ "bounce", (int) T, C }); } }
    if (D("io", 1) != 0) {   // reads of the file into the pinned ring (key io=0 to skip): pread buffered / O_DIRECT, io_uring buffered / O_DIRECT; keys threads_io, chunks_io_kb
        std::vector<size_t> Ti = list_of("threads_io", "1,2,4", 1), Ci = list_of("chunks_io_kb", "256,1024,4096", 1024);
        for (const char * m : { "pread", "pread_d", "uring", "uring_d" }) { for (size_t T : Ti) { for (size_t C : Ci) { cfgs.push_back({ m, (int) T, C }); } } }
    }
    size_t next = 0; // the next unused 256 MiB window of the file, wrapped
    for (auto & c : cfgs) {
        std::vector<double> alone(nd, 0), both(nd, 0);
        for (int pass = 0; pass < 2; ++pass) {         // pass 0: one GPU at a time; pass 1: all at once
            std::vector<double> acc(nd, 0);
            for (int r = 0; r < reps; ++r) {
                // a fresh mapping: page tables empty, page cache hot: the first touch of the mmap'd model
                std::vector<const char *> maps(nd); std::vector<size_t> offs(nd);
                for (int d = 0; d < nd; ++d) { offs[d] = (next % ((size_t) fsize / region - 1)) * region; ++next; }
                for (int d = 0; d < nd; ++d) { void * m = mmap(nullptr, region, PROT_READ, MAP_PRIVATE, fd, offs[d]); if (m == MAP_FAILED) { perror("mmap"); exit(1); } maps[d] = (const char *) m; }
                if (pass == 0) {
                    for (int d = 0; d < nd; ++d) { UpResult u = upload_run(d, dbuf[d], c.mode == "pinned" ? pinned.p : maps[d], region, c.T, c.chunk, ring, c.mode == "dma", io_of(c.mode), UpFiles{ fd, fdd, (off_t) offs[d] }); acc[d] += u.gbs; }
                } else {
                    std::vector<std::thread> th; std::vector<UpResult> res(nd);
                    for (int d = 0; d < nd; ++d) { th.emplace_back([&, d]() { res[d] = upload_run(d, dbuf[d], c.mode == "pinned" ? pinned.p : maps[d], region, c.T, c.chunk, ring, c.mode == "dma", io_of(c.mode), UpFiles{ fd, fdd, (off_t) offs[d] }); }); }
                    for (auto & t : th) { t.join(); }
                    for (int d = 0; d < nd; ++d) { acc[d] += res[d].gbs; }
                }
                for (int d = 0; d < nd; ++d) { munmap((void *) maps[d], region); }
            }
            for (int d = 0; d < nd; ++d) { (pass == 0 ? alone : both)[d] = acc[d] / reps; }
        }
        char chunk[16]; if (c.chunk) { snprintf(chunk, sizeof chunk, "%zuK", c.chunk >> 10); } else { snprintf(chunk, sizeof chunk, "-"); }
        printf("%-9s %-4d %-9s |", c.mode.c_str(), c.T, chunk);
        for (int d = 0; d < nd; ++d) { printf(" dev%d %6.2f", d, alone[d]); }
        printf(" |");
        for (int d = 0; d < nd; ++d) { printf(" dev%d %6.2f", d, both[d]); }
        printf("  (sum %.2f)\n", both[0] + (nd > 1 ? both[1] : 0));
        fflush(stdout);
    }
}

// ---- bench=lat ----
// What does a small host-to-device copy cost while big uploads are running on the same card? (the lane thread's input copies and stash pull wait behind the upload thread's weights in the pipeline)
// A big-upload thread keeps copying `big_mb` blocks from the chosen source in the chosen way; a probe thread issues one small cudaMemcpyAsync (pinned source, own stream) every probe_ms and times it to completion.
static int g_probe_prio = 0; // bench=lat: 1 = the probe stream has the highest stream priority
static double pct(std::vector<double> v, double q) { if (v.empty()) { return 0; } std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; }

static void lat_run(int dev, const char * label, HostMem & src, size_t big, size_t chunk, int ring, const std::vector<size_t> & probes, double probe_ms, double secs) {
    CK(cudaSetDevice(dev));
    void * dbig = nullptr; if (big) { CK(cudaMalloc(&dbig, big)); }
    std::atomic<bool> stop{false};
    // idle latency of each probe size first
    std::vector<char *> ph(probes.size()); std::vector<void *> pd(probes.size());
    for (size_t i = 0; i < probes.size(); ++i) { CK(cudaHostAlloc((void **) &ph[i], probes[i], cudaHostAllocPortable)); CK(cudaMalloc(&pd[i], probes[i])); memset(ph[i], 1, probes[i]); }
    cudaStream_t sp; if (g_probe_prio) { int lo = 0, hi = 0; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); CK(cudaStreamCreateWithPriority(&sp, cudaStreamNonBlocking, hi)); }
    else { CK(cudaStreamCreateWithFlags(&sp, cudaStreamNonBlocking)); }
    auto probe_once = [&](size_t i) { const double t0 = now_s(); CK(cudaMemcpyAsync(pd[i], ph[i], probes[i], cudaMemcpyHostToDevice, sp)); CK(cudaStreamSynchronize(sp)); return (now_s() - t0) * 1e3; };
    std::vector<double> idle(probes.size()); for (size_t i = 0; i < probes.size(); ++i) { probe_once(i); std::vector<double> v; for (int r = 0; r < 20; ++r) { v.push_back(probe_once(i)); } idle[i] = pct(v, 0.5); }
    std::vector<std::vector<double>> lat(probes.size());
    std::atomic<size_t> bytes_done{0};
    std::thread up([&]() {   // the upload thread (big == 0: none, the probe alone)
        if (!big) { return; }
        CK(cudaSetDevice(dev)); cudaStream_t su; CK(cudaStreamCreateWithFlags(&su, cudaStreamNonBlocking));
        std::vector<char *> pin; std::vector<cudaEvent_t> ev;
        if (chunk) { for (int k = 0; k < ring; ++k) { char * p; CK(cudaHostAlloc((void **) &p, chunk, cudaHostAllocPortable)); pin.push_back(p); cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); ev.push_back(e); CK(cudaEventRecord(e, su)); } }
        size_t off_src = 0;
        while (!stop.load()) {
            if (off_src + big > src.size) { off_src = 0; }
            if (!chunk) {
                CK(cudaMemcpyAsync(dbig, src.p + off_src, big, cudaMemcpyHostToDevice, su)); CK(cudaStreamSynchronize(su));
            } else {
                int k = 0;
                for (size_t o = 0; o < big && !stop.load(); o += chunk, k = (k + 1) % ring) {
                    const size_t n = std::min(chunk, big - o);
                    CK(cudaEventSynchronize(ev[k]));
                    if (src.kind != "pinned") { memcpy(pin[k], src.p + off_src + o, n); }
                    CK(cudaMemcpyAsync((char *) dbig + o, src.kind == "pinned" ? src.p + off_src + o : pin[k], n, cudaMemcpyHostToDevice, su)); CK(cudaEventRecord(ev[k], su));
                }
                CK(cudaStreamSynchronize(su));
            }
            bytes_done += big; off_src += big;
        }
        CK(cudaStreamSynchronize(su));
    });
    const double t0 = now_s();
    while (now_s() - t0 < secs) {
        for (size_t i = 0; i < probes.size(); ++i) { lat[i].push_back(probe_once(i)); std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(probe_ms)); }
    }
    const double dt = now_s() - t0;
    stop = true; up.join();
    printf("%-34s upload %5.2f GB/s |", label, bytes_done.load() / (dt * 1e9));
    for (size_t i = 0; i < probes.size(); ++i) { printf(" %zuK: idle %.2f p50 %.2f p90 %.2f max %.2f ms |", probes[i] >> 10, idle[i], pct(lat[i], 0.5), pct(lat[i], 0.9), pct(lat[i], 1.0)); }
    printf("\n"); fflush(stdout);
    if (dbig) { CK(cudaFree(dbig)); }
}

static void bench_lat() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    const size_t big = (size_t) D("big_mb", 1024) << 20; const double probe_ms = D("probe_ms", 15), secs = D("secs", 3);
    std::vector<size_t> probes = list_of("probes_kb", "24,4096,51200", 1024);
    HostMem file = make_host("file", (size_t) D("host_gb", 4) << 30), pinned = make_host("pinned", (size_t) D("host_gb", 4) << 30);
    printf("bench=lat: big upload %zu MiB blocks, probe every %.0f ms for %.0f s per case; probe sizes (KiB):", big >> 20, probe_ms, secs); for (auto p : probes) { printf(" %zu", p >> 10); } printf("\n");
    const bool ring_cases = S("cases", "all") == "ring";   // cases=ring: only the queue-depth and stream-priority cases
    for (int d = 0; d < nd; ++d) {
        printf("--- dev%d\n", d);
        lat_run(d, "no upload (probe alone)", pinned, 0, 0, 0, probes, probe_ms, 0.5);
        if (ring_cases) {
            // does the probe get through when the uploader keeps at most `ring` chunks queued (ring 1: the stream idles for a moment between chunks)?
            for (int ring : { 1, 2 }) {
                for (size_t c : { (size_t) 256 << 10, (size_t) 1 << 20, (size_t) 4 << 20, (size_t) 32 << 20 }) {
                    char l[96]; snprintf(l, sizeof l, "pinned, chunks %zuK ring %d", c >> 10, ring); lat_run(d, l, pinned, big, c, ring, probes, probe_ms, secs);
                }
            }
            g_probe_prio = 1;
            lat_run(d, "pinned, one copy per block, probe high prio", pinned, big, 0, 0, probes, probe_ms, secs);
            lat_run(d, "pinned, chunks 4096K ring 4, probe high prio", pinned, big, (size_t) 4 << 20, 4, probes, probe_ms, secs);
            g_probe_prio = 0;
            continue;
        }
        lat_run(d, "pageable file, one copy per block", file, big, 0, 0, probes, probe_ms, secs);
        lat_run(d, "pinned, one copy per block", pinned, big, 0, 0, probes, probe_ms, secs);
        for (size_t c : { (size_t) 256 << 10, (size_t) 4 << 20, (size_t) 32 << 20 }) {
            char l[96]; snprintf(l, sizeof l, "pageable file, bounce ring %zuK x4", c >> 10); lat_run(d, l, file, big, c, 4, probes, probe_ms, secs);
        }
        for (size_t c : { (size_t) 256 << 10, (size_t) 4 << 20, (size_t) 32 << 20 }) {
            char l[96]; snprintf(l, sizeof l, "pinned, chunks %zuK", c >> 10); lat_run(d, l, pinned, big, c, 4, probes, probe_ms, secs);
        }
    }
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) { const char * e = strchr(argv[i], '='); if (e) { g_kv[std::string(argv[i], e - argv[i])] = e + 1; } }
    const std::string bench = S("bench", "pipe");
    if (bench == "bw") { bench_bw(); }
    else if (bench == "sync") { bench_sync(); }
    else if (bench == "up") { bench_up(); }
    else if (bench == "lat") { bench_lat(); }
    else {
        const int reps = (int) D("reps", 1);
        for (int r = 0; r < reps; ++r) { if (D("lanes", 2) >= 2 && D("baseline", 1) != 0) { run_pipe(1, true); } run_pipe((int) D("lanes", 2), true); }
    }
    return 0;
}
