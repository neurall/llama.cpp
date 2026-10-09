// cudapipe.cu: the two-GPU prefill pipeline and its building blocks on dummy CUDA, with the real load sizes and timings (no model).
//
//   bench=bw     link bandwidth per GPU (pinned / pageable / file-backed host memory, alone and both GPUs at once, duplex), the stash crossing
//   bench=sync   sync and launch costs: stream/event sync, empty kernel round trip, cross-device event wait, queued kernels, cudaMalloc
//   bench=pipe   the pass pipeline: groups of blocks alternate between the lanes (one GPU each); a lane = a host thread + an uploader thread; every step
//                waits for its weights (uploaded through K slots), for the stash of the group before (pushed through pinned host memory by the other
//                lane), computes (a kernel that spins for the measured time), and pushes its stash on. Reports the wall time and tokens/s, and the
//                one-card run of the same load (lanes=1) for comparison.
//
// build: nvcc -O2 -std=c++17 -arch=sm_86 cudapipe.cu -o cudapipe -lpthread     run: CUDA_DEVICE_ORDER=PCI_BUS_ID ./cudapipe bench=pipe key=value ...
// pipe keys (defaults = full GLM 3.0bit, 12k prompt in 4 parts): blocks=45 dense=3 dsa_every=4 parts=4 gsz=1 w_mb=2550 pieces=3 stash_mb=198
//   t_kda_ms=77 t_dsa_ms=154 (per part-step on lane 0) speed1=1.0 (lane 1 relative speed) dev0=1 dev1=0 (CUDA_DEVICE_ORDER=PCI_BUS_ID: 1 = the x16 card)
//   lanes=2 split=alt|share share1=0.4 slots=9 src=pinned|pageable|file file=PATH host_gb=8 tokens=12416
//   hostwork_ms=0 (CPU time per step before the launch) inputsync=0 (wait for the lane's last step before the next launch: one input copy)
//   replan=0 (a DSA step ends with a stream sync, the allocator's re-plan) kernels=1 (kernels per step) uploader=1 (own thread per lane) reps=1
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
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
        int fd = open(path.c_str(), O_RDWR | O_CREAT, 0644);
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
        int b0 = std::max(0, b1 - gsz); Grp g{}; g.b0 = b0; g.b1 = b1;
        if (b0 < dense && b0 > 0) { b0 = dense; g.b0 = b0; } // dense lead on its own at the front
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

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) { const char * e = strchr(argv[i], '='); if (e) { g_kv[std::string(argv[i], e - argv[i])] = e + 1; } }
    const std::string bench = S("bench", "pipe");
    if (bench == "bw") { bench_bw(); }
    else if (bench == "sync") { bench_sync(); }
    else {
        const int reps = (int) D("reps", 1);
        for (int r = 0; r < reps; ++r) { if (D("lanes", 2) >= 2 && D("baseline", 1) != 0) { run_pipe(1, true); } run_pipe((int) D("lanes", 2), true); }
    }
    return 0;
}
