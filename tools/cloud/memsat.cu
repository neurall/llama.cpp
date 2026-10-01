// memsat: where does host memory bandwidth saturate, and how much of it is left for the GPUs?
//  A: CPU read bandwidth with 1, 2, 4 ... threads (the buffer is written first, so it really comes from DRAM). Saturation = fewest threads within 5% of the best.
//  B: the same sweep while every GPU pulls host -> device copies at once (what the expert cache does while the CPU computes the missed experts):
//     CPU GB/s, GPU GB/s and their sum at each thread count.
// Build: nvcc -O3 -Xcompiler "-O3 -mavx2 -pthread" memsat.cu -o memsat ; run: ./memsat [GiB buffer = 8]
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint64_t scan(const uint64_t * p, size_t n) {   // 4 independent sums, vectorised by the compiler
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i + 4 <= n; i += 4) { a += p[i]; b += p[i + 1]; c += p[i + 2]; d += p[i + 3]; }
    return a + b + c + d;
}

static double cpu_read(uint64_t * buf, size_t n, int threads, double secs) {   // GB/s over `secs`
    std::atomic<bool> stop{false}; std::vector<double> bytes(threads, 0.0); std::vector<std::thread> th; std::atomic<uint64_t> sink{0};
    for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
        const size_t per = n / threads; const uint64_t * p = buf + per * t; uint64_t s = 0;
        while (!stop) { s += scan(p, per); bytes[t] += (double) per * 8; }
        sink += s;
    });
    double t0 = now(); std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop = true;
    for (auto & x : th) x.join();
    double dt = now() - t0, tot = 0; for (double b : bytes) tot += b;
    return tot / dt / 1e9;
}

int main(int argc, char ** argv) {
    const size_t gib = argc > 1 ? atoi(argv[1]) : 8, n = (gib << 30) / 8;
    uint64_t * buf = (uint64_t *) aligned_alloc(4096, n * 8);
    { std::vector<std::thread> th; int nt = (int) std::thread::hardware_concurrency(); for (int t = 0; t < nt; ++t) th.emplace_back([&, t] { for (size_t i = n / nt * t; i < n / nt * (t + 1); ++i) buf[i] = i * 2654435761u; }); for (auto & x : th) x.join(); }
    const int tl[] = {1, 2, 4, 8, 16, 32, 64, 128}; const int maxt = (int) std::thread::hardware_concurrency();
    std::vector<int> ts; for (int t : tl) if (t <= maxt) ts.push_back(t);
    printf("A: CPU read bandwidth alone (%zu GiB buffer)\n threads   GB/s\n", gib);
    std::vector<double> alone; double best = 0;
    for (int t : ts) { double b = cpu_read(buf, n, t, 0.6); alone.push_back(b); if (b > best) best = b; printf(" %7d  %6.1f\n", t, b); }
    for (size_t i = 0; i < ts.size(); ++i) if (alone[i] >= 0.95 * best) { printf("saturates at %d threads (%.1f GB/s, best %.1f)\n", ts[i], alone[i], best); break; }
    int ng = 0; cudaGetDeviceCount(&ng);
    if (ng == 0) return 0;
    printf("\nB: the same while %d GPU(s) copy host -> device at once\n threads  CPU GB/s  GPU GB/s  sum GB/s\n", ng);
    std::atomic<bool> stop{false}; std::vector<double> gbytes(ng, 0.0); std::vector<std::thread> gth; std::atomic<int> ready{0};
    const size_t cb = 256ull << 20;
    for (int g = 0; g < ng; ++g) gth.emplace_back([&, g] {
        cudaSetDevice(g); void * h = nullptr, * d = nullptr; cudaStream_t s; cudaMallocHost(&h, cb); cudaMalloc(&d, cb); cudaStreamCreate(&s); ready++;
        while (!stop) { cudaMemcpyAsync(d, h, cb, cudaMemcpyHostToDevice, s); cudaStreamSynchronize(s); gbytes[g] += cb; }
        cudaFreeHost(h); cudaFree(d);
    });
    while (ready < ng) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for (int t : ts) {
        double g0 = 0; for (double x : gbytes) g0 += x; double t0 = now();
        double c = cpu_read(buf, n, t, 0.6);
        double g1 = 0; for (double x : gbytes) g1 += x; double gb = (g1 - g0) / (now() - t0) / 1e9;
        printf(" %7d  %8.1f  %8.1f  %8.1f\n", t, c, gb, c + gb);
    }
    stop = true; for (auto & x : gth) x.join();
    return 0;
}
