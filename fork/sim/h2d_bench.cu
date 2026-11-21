// H2D bandwidth vs chunk size and number of streams (pinned source), one device. usage: h2d DEV
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
int main(int argc, char ** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    const size_t total = (size_t) 2 << 30;   // 2 GiB moved per point
    char * h; CK(cudaHostAlloc(&h, total, cudaHostAllocDefault));
    for (size_t i = 0; i < total; i += 4096) { h[i] = (char) i; }
    char * d; CK(cudaMalloc(&d, total));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("device %d %s\n", dev, p.name);
    const size_t sizes[] = { 16<<10, 64<<10, 256<<10, 512<<10, 1<<20, 2<<20, 4<<20, 8<<20, 16<<20, 32<<20, 64<<20, 256<<20 };
    const int nst[] = { 1, 2, 4, 8 };
    printf("%-9s", "chunk\\str"); for (int s : nst) { printf(" %7d", s); } printf("   (GB/s)\n");
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (size_t cs : sizes) {
        printf("%-9s", cs >= (1<<20) ? (std::to_string(cs >> 20) + " MiB").c_str() : (std::to_string(cs >> 10) + " KiB").c_str());
        for (int ns : nst) {
            std::vector<cudaStream_t> st(ns); for (auto & s : st) { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
            const size_t n = total / cs;
            for (int rep = 0; rep < 2; ++rep) {   // warm, then measure
                CK(cudaEventRecord(e0, 0));
                for (size_t i = 0; i < n; ++i) { CK(cudaMemcpyAsync(d + (i * cs) % total, h + (i * cs) % total, cs, cudaMemcpyHostToDevice, st[i % ns])); }
                for (auto & s : st) { CK(cudaStreamSynchronize(s)); }
                CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
                if (rep == 1) { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); printf(" %7.2f", (double) total / (ms * 1e-3) / 1e9); }
            }
            for (auto & s : st) { CK(cudaStreamDestroy(s)); }
        }
        printf("\n"); fflush(stdout);
    }
    return 0;
}
