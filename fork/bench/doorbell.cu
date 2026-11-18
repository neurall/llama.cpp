// doorbell.cu: what one host step per layer costs the card in decode, two ways (user 2026-11-18: take the host out of the per-layer loop).
//   build: nvcc -O2 -o doorbell fork/bench/doorbell.cu      run: ./doorbell [device 0] [layers 48] [work_us 150] [host_us 0] [tokens 200] [x_floats 2560] [row_floats 25600]
//   x_floats / row_floats: how much of the layer input goes to the host and how many floats of rows come back per layer (0 0 = a layer whose experts are all on the card)
// A token is LAYERS layers; a layer is some GPU work, then the host has to see the layer's 10 expert ids (and the layer input), may compute something,
// and hands rows back before the next layer may read them.
//   sync:     today's shape (nsys, Qwen3.8 IQ3_S, build b15843: 104 graph launches and 122 stream waits a token): a graph launch per layer, cudaStreamSynchronize,
//             two copies to the host, the host step, a copy to the card, the next launch.
//   doorbell: the whole token is ONE graph launched once. Behind each layer's work a kernel writes the ids and the input into mapped host memory and counts up a
//             word there; the host spins on that word, does its step, writes its rows into mapped memory and raises a flag; a kernel of the same graph spins on the
//             flag and then copies the rows. No driver call per layer.
// Printed: the time of a token with the work alone (one graph, no host), and what each way adds per layer on top of it.
#include <cuda_runtime.h>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <immintrin.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int N_EMBD = 2560, N_USED = 10;

__global__ void k_work(float * buf, int n, int iters) {   // about iters x a few ns on one block: stands for the layer's kernels
    float a = buf[threadIdx.x % n];
    for (int i = 0; i < iters; ++i) { a = a * 1.0000001f + 1e-7f; }
    buf[threadIdx.x % n] = a;
}

__global__ void k_publish(const int * ids, int * ids_host, const float * x, float * x_host, int n, volatile unsigned * seq) {
    if ((int) threadIdx.x < N_USED) { ids_host[threadIdx.x] = ids[threadIdx.x]; }
    for (int i = threadIdx.x; i < n; i += blockDim.x) { x_host[i] = x[i]; }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) { *seq = *seq + 1u; __threadfence_system(); }
}

__global__ void k_wait_rows(const volatile unsigned * flag, unsigned want, const float * rows_host, float * dst, int n) {
    if (threadIdx.x == 0) { while (*flag < want) { __nanosleep(100); } }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) { dst[i] += rows_host[i]; }
}

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void spin_us(double us) { const double t = now_us() + us; while (now_us() < t) { _mm_pause(); } }

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0, layers = argc > 2 ? atoi(argv[2]) : 48, tokens = argc > 5 ? atoi(argv[5]) : 200;
    const double work_us = argc > 3 ? atof(argv[3]) : 150, host_us = argc > 4 ? atof(argv[4]) : 0;
    const int n_x = argc > 6 ? atoi(argv[6]) : N_EMBD, n_rows = argc > 7 ? atoi(argv[7]) : N_EMBD * N_USED;
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleSpin));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    float * d_buf, * d_x, * d_dst; int * d_ids;
    CK(cudaMalloc(&d_buf, 1024 * sizeof(float))); CK(cudaMalloc(&d_x, N_EMBD * sizeof(float))); CK(cudaMalloc(&d_dst, N_EMBD * N_USED * sizeof(float))); CK(cudaMalloc(&d_ids, N_USED * sizeof(int)));
    CK(cudaMemset(d_buf, 0, 1024 * sizeof(float))); CK(cudaMemset(d_x, 0, N_EMBD * sizeof(float))); CK(cudaMemset(d_dst, 0, N_EMBD * N_USED * sizeof(float))); CK(cudaMemset(d_ids, 0, N_USED * sizeof(int)));
    // mapped host memory: the card reads and writes it by pointer
    int * h_ids; float * h_x, * h_rows; unsigned * h_words;
    CK(cudaHostAlloc(&h_ids, N_USED * sizeof(int), cudaHostAllocMapped)); CK(cudaHostAlloc(&h_x, N_EMBD * sizeof(float), cudaHostAllocMapped));
    CK(cudaHostAlloc(&h_rows, N_EMBD * N_USED * sizeof(float), cudaHostAllocMapped)); CK(cudaHostAlloc(&h_words, 2 * 64, cudaHostAllocMapped));
    int * m_ids; float * m_x, * m_rows; unsigned * m_words;
    CK(cudaHostGetDevicePointer(&m_ids, h_ids, 0)); CK(cudaHostGetDevicePointer(&m_x, h_x, 0)); CK(cudaHostGetDevicePointer(&m_rows, h_rows, 0)); CK(cudaHostGetDevicePointer(&m_words, h_words, 0));
    volatile unsigned * h_seq = h_words, * h_flag = h_words + 16;   // their own cache lines
    memset(h_rows, 0, N_EMBD * N_USED * sizeof(float));
    std::vector<float> p_x(N_EMBD), p_rows(N_EMBD * N_USED, 0.0f); std::vector<int> p_ids(N_USED);   // ordinary host memory, as the scheduler's host tensors

    // the work kernel's loop count for work_us
    int iters = 100000;
    for (int pass = 0; pass < 6; ++pass) {
        CK(cudaStreamSynchronize(s)); const double t0 = now_us();
        for (int i = 0; i < 20; ++i) { k_work<<<1, 256, 0, s>>>(d_buf, 1024, iters); }
        CK(cudaStreamSynchronize(s)); const double per = (now_us() - t0) / 20;
        iters = (int) (iters * work_us / per); if (iters < 1) { iters = 1; }
    }
    auto capture = [&](int kind, int n_layers) {   // kind 0: work alone; 1: work + doorbell
        cudaGraph_t g; cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < n_layers; ++l) {
            k_work<<<1, 256, 0, s>>>(d_buf, 1024, iters);
            if (kind == 1) {
                k_publish<<<1, 256, 0, s>>>(d_ids, m_ids, d_x, m_x, n_x, (volatile unsigned *) m_words);
                k_wait_rows<<<1, 256, 0, s>>>((const volatile unsigned *) (m_words + 16), (unsigned) (l + 1), m_rows, d_dst, n_rows);
            }
        }
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ex, g, 0)); CK(cudaGraphDestroy(g));
        return ex;
    };
    cudaGraphExec_t g_token = capture(0, layers), g_layer = capture(0, 1), g_bell = capture(1, layers);

    auto run = [&](const char * name, const std::function<void()> & token) {
        for (int i = 0; i < 20; ++i) { token(); }
        const double t0 = now_us();
        for (int i = 0; i < tokens; ++i) { token(); }
        const double us = (now_us() - t0) / tokens;
        printf("%-9s %9.1f us a token, %7.1f us a layer\n", name, us, us / layers);
        return us;
    };
    const double base = run("work", [&] { CK(cudaGraphLaunch(g_token, s)); CK(cudaStreamSynchronize(s)); });
    const double sync = run("sync", [&] {
        for (int l = 0; l < layers; ++l) {
            CK(cudaGraphLaunch(g_layer, s));
            CK(cudaMemcpyAsync(p_ids.data(), d_ids, N_USED * sizeof(int), cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
            if (n_x > 0) { CK(cudaMemcpyAsync(p_x.data(), d_x, n_x * sizeof(float), cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); }
            if (host_us > 0) { spin_us(host_us); }
            if (n_rows > 0) { CK(cudaMemcpyAsync(d_dst, p_rows.data(), n_rows * sizeof(float), cudaMemcpyHostToDevice, s)); }
        }
        CK(cudaStreamSynchronize(s));
    });
    unsigned long bad = 0;
    const double bell = run("doorbell", [&] {
        *h_seq = 0; *h_flag = 0;
        CK(cudaGraphLaunch(g_bell, s));
        for (int l = 0; l < layers; ++l) {
            while (*h_seq < (unsigned) (l + 1)) { _mm_pause(); }
            bad += h_ids[0] != 0;   // the ids are read here
            if (host_us > 0) { spin_us(host_us); }
            *h_flag = (unsigned) (l + 1);
        }
        CK(cudaStreamSynchronize(s));
    });
    printf("work kernel %.0f us, host step %.0f us, %d layers, %d floats out, %d floats back: the host step per layer costs the card  sync %.1f us   doorbell %.1f us   (a token: %.2f ms against %.2f ms)%s\n",
           work_us, host_us, layers, n_x, n_rows, (sync - base) / layers - host_us, (bell - base) / layers - host_us, sync / 1e3, bell / 1e3, bad ? "  IDS WRONG" : "");
    return 0;
}
