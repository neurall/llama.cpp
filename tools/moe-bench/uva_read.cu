// uva_read.cu: can a GPU kernel read pinned host memory (UVA zero-copy) as fast as a DMA copy?
// Question behind it: compute uncached MoE experts on the GPU straight from pinned RAM instead of
// copying them into cache slots first. Compares, per device and size:
//   memcpy : cudaMemcpyAsync host->device (the DMA path the cache uses)
//   uva16  : kernel, every thread reads 16 B grid-stride (best case coalescing)
//   row    : kernel, one warp per row of ROW bytes, lanes read 16 B each (mat-vec-like pattern)
// Build: nvcc -O2 -arch=sm_86 uva_read.cu -o uva_read ; run: ./uva_read [device] [MB]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void k_uva16(const int4 * __restrict__ p, size_t n, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) { *out = acc; }
}

// one warp per row: lanes stride the row in 16 B pieces (like mmvq reading quant blocks of one row)
__global__ void k_row(const char * __restrict__ p, int rows, int row_bytes, unsigned * out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (warp >= rows) { return; }
    const int4 * r = (const int4 *) (p + (size_t) warp * row_bytes);
    unsigned acc = 0;
    for (int i = lane; i < row_bytes / 16; i += 32) {
        const int4 v = r[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) { *out = acc; }
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    const size_t mb = argc > 2 ? atoi(argv[2]) : 8;
    CK(cudaSetDevice(dev));
    const size_t n = mb << 20;
    char * h; CK(cudaHostAlloc(&h, n, cudaHostAllocMapped | cudaHostAllocPortable));
    for (size_t i = 0; i < n; i += 4096) { h[i] = (char) i; }
    char * hd; CK(cudaHostGetDevicePointer((void **) &hd, h, 0));
    char * d; CK(cudaMalloc(&d, n));
    unsigned * o; CK(cudaMalloc(&o, 4));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    printf("device %d %s, %zu MB\n", dev, pr.name, mb);
    auto time = [&](const char * name, auto fn) {
        fn(); CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 10; ++r) {
            CK(cudaEventRecord(a)); fn(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = ms < best ? ms : best;
        }
        printf("  %-28s %8.1f us  %6.2f GB/s\n", name, best * 1e3, n / (best * 1e-3) / 1e9);
    };
    time("memcpy H2D", [&] { CK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice)); });
    for (int blocks : { 82, 164, 328, 656, 1312 }) {
        char nm[64]; snprintf(nm, sizeof nm, "uva16 %d blocks x 256", blocks);
        time(nm, [&] { k_uva16<<<blocks, 256>>>((const int4 *) hd, n / 16, o); });
    }
    for (int row : { 1344, 2688, 5376 }) {
        const int rows = (int) (n / row);
        char nm[64]; snprintf(nm, sizeof nm, "row %d B (%d rows)", row, rows);
        time(nm, [&] { k_row<<<(rows * 32 + 255) / 256, 256>>>(hd, rows, row, o); });
    }
    time("row 1344 B from VRAM (ref)", [&] { k_row<<<((int) (n / 1344) * 32 + 255) / 256, 256>>>(d, (int) (n / 1344), 1344, o); });
    return 0;
}
