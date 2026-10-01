// sync_lat.cu: GPU<->host handoff latency, the per-layer floor of the hybrid CPU/GPU decode (42 handoffs per token).
//   A  kernel + cudaStreamSynchronize
//   B  kernel writes a mapped host flag, host spin-polls it              (GGML_SCHED_D2H_ASYNC + PRELAUNCH path)
//   C  kernel + cudaLaunchHostFunc callback sets an atomic, host polls it (ktransformers' submit_with_cuda_stream)
//   D  host flag -> queued device spin-wait kernel -> kernel writes flag back, host polls (CPU->GPU->CPU round trip)
//   E  host launches a kernel then calls cudaStreamSynchronize (launch + sync from the host thread, base cost)
// build: nvcc -O2 sync_lat.cu -o sync_lat      run: sync_lat [iterations]
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static inline double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__global__ void k_empty() {}
__global__ void k_write(volatile uint32_t * f, uint32_t v) { *f = v; __threadfence_system(); }
__global__ void k_wait_then_write(const volatile uint32_t * in, uint32_t v, volatile uint32_t * out) {
    while (*in < v) { }
    *out = v;
    __threadfence_system();
}

static std::atomic<uint32_t> g_cb_flag;
static void CUDART_CB host_cb(void * ud) { g_cb_flag.store((uint32_t) (uintptr_t) ud, std::memory_order_release); }

static void report(const char * name, std::vector<double> & v) {
    std::sort(v.begin(), v.end());
    printf("%-52s p50 %7.1f us  p90 %7.1f  p99 %7.1f  min %7.1f\n", name, v[v.size()/2], v[v.size()*9/10], v[v.size()*99/100], v[0]);
}

int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 2000;
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    uint32_t * hf; CK(cudaHostAlloc((void **) &hf, 64, cudaHostAllocMapped));
    uint32_t * df; CK(cudaHostGetDevicePointer((void **) &df, hf, 0));
    volatile uint32_t * f1 = hf; volatile uint32_t * f2 = hf + 16;
    uint32_t * d1 = df; uint32_t * d2 = df + 16;
    hf[0] = 0; hf[16] = 0;
    for (int i = 0; i < 50; ++i) { k_empty<<<1, 1, 0, s>>>(); } CK(cudaStreamSynchronize(s));

    std::vector<double> A, B, C, D, E;
    for (int i = 1; i <= N; ++i) {
        double t0 = now_us();
        k_empty<<<1, 1, 0, s>>>();
        CK(cudaStreamSynchronize(s));
        E.push_back(now_us() - t0);
    }
    for (int i = 1; i <= N; ++i) {
        // A: a slightly bigger kernel so the sync isn't racing the launch
        double t0 = now_us();
        k_write<<<1, 1, 0, s>>>((volatile uint32_t *) d1, (uint32_t) i);
        CK(cudaStreamSynchronize(s));
        A.push_back(now_us() - t0);
    }
    for (int i = 1; i <= N; ++i) {
        const uint32_t v = 1000000 + i;
        double t0 = now_us();
        k_write<<<1, 1, 0, s>>>((volatile uint32_t *) d1, v);
        while (*f1 < v) { }
        B.push_back(now_us() - t0);
    }
    CK(cudaStreamSynchronize(s));
    for (int i = 1; i <= N; ++i) {
        const uint32_t v = 2000000 + i;
        double t0 = now_us();
        k_empty<<<1, 1, 0, s>>>();
        CK(cudaLaunchHostFunc(s, host_cb, (void *) (uintptr_t) v));
        while (g_cb_flag.load(std::memory_order_acquire) < v) { }
        C.push_back(now_us() - t0);
    }
    CK(cudaStreamSynchronize(s));
    for (int i = 1; i <= N; ++i) {
        const uint32_t v = 3000000 + i;
        k_wait_then_write<<<1, 1, 0, s>>>((const volatile uint32_t *) d1, v, (volatile uint32_t *) d2);
        double tw = now_us(); while (now_us() - tw < 300.0) { } // let the waiter start spinning on the device
        double t0 = now_us();
        *(volatile uint32_t *) f1 = v;
        while (*f2 < v) { }
        D.push_back(now_us() - t0);
    }
    CK(cudaStreamSynchronize(s));
    report("E launch empty kernel + cudaStreamSynchronize", E);
    report("A launch flag-kernel + cudaStreamSynchronize", A);
    report("B launch flag-kernel, host polls mapped flag", B);
    report("C launch + cudaLaunchHostFunc, host polls atomic", C);
    report("D host flag -> device spin kernel -> flag back", D);
    return 0;
}
