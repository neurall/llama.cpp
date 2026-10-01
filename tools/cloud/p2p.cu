// p2p: GPU <-> GPU copy bandwidth. Peer access (can one GPU read another's memory directly over PCIe) is listed per pair; without it the driver stages
// the copy through host memory, which shows as a much lower number. Then every pair alone, and a ring where each GPU sends to the next at once.
// Build: nvcc -O2 p2p.cu -o p2p ; run: ./p2p [MiB per copy = 256] [seconds per test = 1]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const size_t bytes = (size_t) (argc > 1 ? atoi(argv[1]) : 256) << 20; const double secs = argc > 2 ? atof(argv[2]) : 1.0;
    int n = 0; cudaGetDeviceCount(&n);
    std::vector<void *> buf(n), buf2(n); std::vector<cudaStream_t> st(n);
    for (int i = 0; i < n; ++i) { cudaSetDevice(i); cudaMalloc(&buf[i], bytes); cudaMalloc(&buf2[i], bytes); cudaStreamCreate(&st[i]); }
    printf("peer access (1 = direct GPU to GPU over PCIe, 0 = staged through host memory):\n      ");
    for (int j = 0; j < n; ++j) printf("to %-3d", j);
    printf("\n");
    for (int i = 0; i < n; ++i) {
        printf("from %d", i);
        for (int j = 0; j < n; ++j) {
            int ok = 0; if (i != j) { cudaDeviceCanAccessPeer(&ok, i, j); if (ok) { cudaSetDevice(i); cudaDeviceEnablePeerAccess(j, 0); cudaGetLastError(); } }
            printf("  %s  ", i == j ? "-" : (ok ? "1" : "0"));
        }
        printf("\n");
    }
    auto copy_bw = [&](int i, int j) {   // GPU i -> GPU j, one pair alone
        cudaSetDevice(i); cudaMemcpyPeerAsync(buf2[j], j, buf[i], i, bytes, st[i]); cudaStreamSynchronize(st[i]);
        double t0 = now(), dt = 0; size_t k = 0;
        while (dt < secs) { for (int r = 0; r < 4; ++r) { cudaMemcpyPeerAsync(buf2[j], j, buf[i], i, bytes, st[i]); k++; } cudaStreamSynchronize(st[i]); dt = now() - t0; }
        return (double) k * bytes / dt / 1e9;
    };
    printf("\nGB/s, one pair at a time (row = source, column = destination):\n      ");
    for (int j = 0; j < n; ++j) printf("to %-5d", j);
    printf("\n");
    for (int i = 0; i < n; ++i) {
        printf("from %d", i);
        for (int j = 0; j < n; ++j) { if (i == j) printf("  %6s ", "-"); else printf("  %6.1f ", copy_bw(i, j)); }
        printf("\n");
    }
    if (n > 1) {   // ring: i -> i+1, all at once
        double t0 = now(), dt = 0; size_t k = 0;
        for (int i = 0; i < n; ++i) { cudaSetDevice(i); cudaMemcpyPeerAsync(buf2[(i + 1) % n], (i + 1) % n, buf[i], i, bytes, st[i]); cudaStreamSynchronize(st[i]); }
        t0 = now();
        while (dt < secs) {
            for (int r = 0; r < 4; ++r) for (int i = 0; i < n; ++i) { cudaSetDevice(i); cudaMemcpyPeerAsync(buf2[(i + 1) % n], (i + 1) % n, buf[i], i, bytes, st[i]); }
            for (int i = 0; i < n; ++i) { cudaSetDevice(i); cudaStreamSynchronize(st[i]); }
            k += 4; dt = now() - t0;
        }
        double per = (double) k * bytes / dt / 1e9;
        printf("\nring (each GPU sends to the next, all at once): %.1f GB/s per GPU, %.1f GB/s total\n", per, per * n);
    }
    return 0;
}
