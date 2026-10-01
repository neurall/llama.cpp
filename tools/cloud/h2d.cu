// h2d: host -> device copy bandwidth per GPU, one GPU alone and then all GPUs at once (what the expert cache does when it uploads misses),
// from pinned memory. A shared PCIe switch or a narrow link shows as a drop in the "all at once" column.
// Build: nvcc -O2 h2d.cu -o h2d ; run: ./h2d [MiB per copy = 256] [seconds = 2]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static double run(int dev, size_t bytes, double secs) {
    cudaSetDevice(dev);
    void * h = nullptr; void * d = nullptr; cudaStream_t s;
    if (cudaMallocHost(&h, bytes) != cudaSuccess || cudaMalloc(&d, bytes) != cudaSuccess) { return 0; }
    cudaStreamCreate(&s);
    cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s); cudaStreamSynchronize(s); // warm-up (also lifts an idle link to its full speed)
    size_t n = 0; auto t0 = std::chrono::steady_clock::now(); double dt = 0;
    while (dt < secs) {
        for (int i = 0; i < 4; ++i) { cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s); n++; }
        cudaStreamSynchronize(s);
        dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    cudaFreeHost(h); cudaFree(d); cudaStreamDestroy(s);
    return (double) n * bytes / dt / 1e9;
}

int main(int argc, char ** argv) {
    const size_t bytes = (size_t) (argc > 1 ? atoi(argv[1]) : 256) << 20; const double secs = argc > 2 ? atof(argv[2]) : 2.0;
    int n = 0; cudaGetDeviceCount(&n);
    std::vector<double> alone(n), all(n);
    for (int i = 0; i < n; ++i) { alone[i] = run(i, bytes, secs); }
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i) { th.emplace_back([&, i] { all[i] = run(i, bytes, secs); }); }
    for (auto & t : th) { t.join(); }
    double sa = 0, sl = 0;
    printf("gpu  name                            alone GB/s   all at once GB/s   pci bus\n");
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        char bus[32]; snprintf(bus, sizeof(bus), "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
        printf("%-4d %-30.30s %9.1f %18.1f   %s\n", i, p.name, alone[i], all[i], bus);
        sa += alone[i]; sl += all[i];
    }
    printf("total (sum)                                 %9.1f %18.1f   shared-link loss: %.0f%%\n", sa, sl, sa > 0 ? 100.0 * (1 - sl / sa) : 0.0);
    return 0;
}
