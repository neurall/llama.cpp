// launchblock.cu: what blocks on a card while one kernel of its compute stream waits for the host (the decode bell's rows kernel)?
//   build: nvcc -O2 -o launchblock launchblock.cu -lpthread      run: ./launchblock [device 0]
// Each case: a kernel W on stream S spins on a flag in mapped host memory; a thread raises the flag after HOLD seconds. Meanwhile this thread (and, in some
// cases, a second "uploader" thread on its own stream) issues work. A call that returns only when the flag is raised was blocked by W.
// RTX 3090, driver 595, CUDA 13.4 (2026-11-18): a stream takes 1021 calls behind W, the next one blocks until W ends: kernels, pinned copies and card-to-card
// copies alike (pageable copies block at once or after 963 small ones). Kernels on another stream and an uploader on its own stream are not held up (64 MiB
// pinned in 3 ms). So whoever answers W must not be the thread that still launches behind it (the decode bell: ggml_sched_bell_thread in ggml-backend.cpp).
#include <cuda_runtime.h>
#include <functional>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
static const double HOLD = 2.0;

__global__ void k_wait(const volatile unsigned * flag) { if (threadIdx.x == 0) { while (*flag == 0) { __nanosleep(100); } } }
__global__ void k_tiny(float * b) { b[threadIdx.x] += 1.0f; }

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
    cudaStream_t s, s2; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    float * d_buf, * d_big, * d_big2; CK(cudaMalloc(&d_buf, 1024 * sizeof(float))); CK(cudaMalloc(&d_big, 64 << 20)); CK(cudaMalloc(&d_big2, 64 << 20));
    CK(cudaMemset(d_buf, 0, 1024 * sizeof(float)));
    unsigned * h_flag; CK(cudaHostAlloc(&h_flag, 64, cudaHostAllocMapped));
    unsigned * m_flag; CK(cudaHostGetDevicePointer(&m_flag, h_flag, 0));
    char * pinned; CK(cudaHostAlloc(&pinned, 64 << 20, cudaHostAllocDefault)); memset(pinned, 1, 64 << 20);
    std::vector<char> pageable(64 << 20, 2);

    // a case: W launched, `work` issued by this thread; returns seconds the work's calls took and the index of the first call that took over half the hold
    auto run = [&](const char * name, int n, const std::function<void(int)> & work, const std::function<void()> & side = nullptr) {
        *(volatile unsigned *) h_flag = 0;
        k_wait<<<1, 32, 0, s>>>((const volatile unsigned *) m_flag);
        CK(cudaGetLastError());
        const double t0 = now_s();
        std::thread rel([&] { std::this_thread::sleep_for(std::chrono::duration<double>(HOLD)); *(volatile unsigned *) h_flag = 1; });
        std::thread sd; double side_s = -1;
        if (side) { sd = std::thread([&] { CK(cudaSetDevice(dev)); const double a = now_s(); side(); side_s = now_s() - a; }); }
        int first_blocked = -1; double worst = 0;
        for (int i = 0; i < n; ++i) {
            const double a = now_s();
            work(i);
            const double d = now_s() - a;
            if (d > worst) { worst = d; }
            if (d > HOLD / 2 && first_blocked < 0) { first_blocked = i; }
        }
        const double issued = now_s() - t0;
        rel.join();
        if (sd.joinable()) { sd.join(); }
        CK(cudaStreamSynchronize(s)); CK(cudaStreamSynchronize(s2));
        printf("%-58s %6d calls issued in %6.3f s, worst call %6.3f s%s", name, n, issued, worst, first_blocked >= 0 ? "" : "  not blocked");
        if (first_blocked >= 0) { printf("  BLOCKED at call %d", first_blocked); }
        if (side) { printf("; side thread took %.3f s%s", side_s, side_s > HOLD / 2 ? " (BLOCKED)" : ""); }
        printf("\n");
    };

    run("kernels on W's stream", 20000, [&](int) { k_tiny<<<1, 256, 0, s>>>(d_buf); });
    run("kernels on another stream", 20000, [&](int) { k_tiny<<<1, 256, 0, s2>>>(d_buf); });
    run("pinned 4 KiB host-to-card copies on W's stream", 2000, [&](int) { CK(cudaMemcpyAsync(d_big, pinned, 4096, cudaMemcpyHostToDevice, s)); });
    run("pageable 4 KiB host-to-card copies on W's stream", 2000, [&](int) { CK(cudaMemcpyAsync(d_big, pageable.data(), 4096, cudaMemcpyHostToDevice, s)); });
    run("pageable 1 MiB host-to-card copies on W's stream", 8, [&](int) { CK(cudaMemcpyAsync(d_big, pageable.data(), 1 << 20, cudaMemcpyHostToDevice, s)); });
    run("card-to-card 1 MiB copies on W's stream", 2000, [&](int) { CK(cudaMemcpyAsync(d_big, d_big2, 1 << 20, cudaMemcpyDeviceToDevice, s)); });
    run("4 KiB card-to-pageable-host copies on W's stream", 8, [&](int) { CK(cudaMemcpyAsync(pageable.data(), d_big, 4096, cudaMemcpyDeviceToHost, s)); });
    // an uploader on its own stream beside W: does its copy finish, and do this thread's launches go on while it waits?
    run("kernels on W's stream + uploader (64 MiB pinned, own stream, sync)", 20000, [&](int) { k_tiny<<<1, 256, 0, s>>>(d_buf); },
        [&] { CK(cudaMemcpyAsync(d_big2, pinned, 64 << 20, cudaMemcpyHostToDevice, cudaStreamPerThread)); CK(cudaStreamSynchronize(cudaStreamPerThread)); });
    run("kernels on W's stream + uploader (4 KiB pinned, own stream, sync)", 20000, [&](int) { k_tiny<<<1, 256, 0, s>>>(d_buf); },
        [&] { CK(cudaMemcpyAsync(d_big2, pinned, 4096, cudaMemcpyHostToDevice, cudaStreamPerThread)); CK(cudaStreamSynchronize(cudaStreamPerThread)); });
    run("kernels on W's stream + side kernel (own stream, sync)", 20000, [&](int) { k_tiny<<<1, 256, 0, s>>>(d_buf); },
        [&] { k_tiny<<<1, 256, 0, cudaStreamPerThread>>>(d_buf + 512); CK(cudaStreamSynchronize(cudaStreamPerThread)); });
    run("200000 kernels on W's stream", 200000, [&](int) { k_tiny<<<1, 256, 0, s>>>(d_buf); });
    return 0;
}
