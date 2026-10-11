// ddrbw: RAM read bandwidth of CPU threads and GPU DMA (host -> device), alone and at the same time.
// Build: nvcc -O3 -std=c++17 -Xcompiler -fopenmp,-march=native -o ddrbw ddrbw.cu
// Run:   ./ddrbw [seconds per test]
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 3.0;
    const size_t cpu_bytes = 4ull << 30, dma_bytes = 2ull << 30, chunk = 64ull << 20;
    int n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);

    // CPU source: ordinary memory, far bigger than the caches
    std::vector<uint64_t> cbuf(cpu_bytes / 8);
    for (size_t i = 0; i < cbuf.size(); ++i) cbuf[i] = i;
    // DMA source: pinned host memory, one slice per GPU
    char * hbuf = nullptr;
    cudaHostAlloc(&hbuf, dma_bytes * n_gpu, cudaHostAllocPortable);
    memset(hbuf, 1, dma_bytes * n_gpu);
    std::vector<char *> dbuf(n_gpu);
    for (int g = 0; g < n_gpu; ++g) { cudaSetDevice(g); cudaMalloc(&dbuf[g], chunk); }

    auto run = [&](const char * name, int n_cpu, std::vector<int> gpus) {
        std::atomic<bool> stop{false};
        std::vector<double> cpu_got(n_cpu, 0), gpu_got(n_gpu, 0);
        std::vector<std::thread> th;
        volatile uint64_t sink = 0;
        for (int t = 0; t < n_cpu; ++t) {
            th.emplace_back([&, t] {
                const size_t n = cbuf.size() / n_cpu, b = t * n;
                uint64_t s = 0;
                double got = 0;
                while (!stop) {
                    for (size_t i = b; i < b + n; i += 4) { s += cbuf[i] + cbuf[i + 1] + cbuf[i + 2] + cbuf[i + 3]; }
                    got += n * 8.0;
                }
                cpu_got[t] = got;
                sink = sink + s;
            });
        }
        for (int g : gpus) {
            th.emplace_back([&, g] {
                cudaSetDevice(g);
                cudaStream_t st;
                cudaStreamCreate(&st);
                double got = 0;
                size_t off = 0;
                while (!stop) {
                    cudaMemcpyAsync(dbuf[g], hbuf + g * dma_bytes + off, chunk, cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st);
                    got += chunk;
                    off = (off + chunk) % dma_bytes;
                }
                gpu_got[g] = got;
                cudaStreamDestroy(st);
            });
        }
        const double t0 = now();
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        stop = true;
        for (auto & t : th) t.join();
        const double dt = now() - t0;
        double c = 0, tot = 0;
        for (double v : cpu_got) c += v;
        std::string gs;
        tot = c;
        for (int g : gpus) {
            char b[64];
            snprintf(b, sizeof(b), "  GPU%d %5.1f", g, gpu_got[g] / dt / 1e9);
            gs += b;
            tot += gpu_got[g];
        }
        printf("%-28s CPU(%2d thr) %5.1f%s  | total %5.1f GB/s\n", name, n_cpu, c / dt / 1e9, gs.c_str(), tot / dt / 1e9);
        fflush(stdout);
    };

    run("cpu 6", 6, {});
    run("cpu 8", 8, {});
    run("cpu 16", 16, {});
    for (int g = 0; g < n_gpu; ++g) run(("dma GPU" + std::to_string(g)).c_str(), 0, {g});
    if (n_gpu > 1) run("dma both", 0, {0, 1});
    for (int g = 0; g < n_gpu; ++g) run(("cpu 6 + dma GPU" + std::to_string(g)).c_str(), 6, {g});
    if (n_gpu > 1) run("cpu 6 + dma both", 6, {0, 1});
    if (n_gpu > 1) run("cpu 16 + dma both", 16, {0, 1});
    return 0;
}
