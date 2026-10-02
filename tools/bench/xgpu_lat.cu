// Round-trip latency of one cross-GPU hop on GPUs without peer access (GeForce): the cost an expert-parallel split pays per layer
// (send the layer input to the other GPU, get its partial result back). Three ways, same payload:
//   host-sync   cudaMemcpyPeerAsync + cudaStreamSynchronize on the host between the two hops (what the moe-net MOE_LOCAL_2GPU experiment did: ~190 us)
//   event-chain the same copies chained with events across the two streams, no host sync inside the loop
//   mapped-flag two persistent kernels (one per GPU) exchange the payload through pinned, mapped host memory and a flag, no copies, no CPU
// nvcc -O2 -arch=sm_86 -o xgpu_lat xgpu_lat.cu && ./xgpu_lat [gpu_a gpu_b iters]     (a = the layer's GPU, b = the helper GPU)
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void touch(float * p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { p[i] += v; }
}

#define SPIN_MAX 400000000LL   // a lost flag ends the kernel instead of hanging the GPU

__global__ void ping(volatile float * ab, volatile float * ba, volatile int * f_ab, volatile int * f_ba, int n, int iters, int * err) {
    for (int it = 1; it <= iters; ++it) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) { ab[i] = (float) it; }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            *f_ab = it;
            long long s = 0;
            while (*f_ba != it && ++s < SPIN_MAX) {}
            if (s >= SPIN_MAX) { *err = 1; }
        }
        __syncthreads();
        __threadfence_system();
        float sum = 0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) { sum += ba[i]; }
        if (sum == -1.f) { ab[0] = sum; }
        if (*err) { return; }
    }
}

__global__ void pong(volatile float * ab, volatile float * ba, volatile int * f_ab, volatile int * f_ba, int n, int iters, int * err) {
    for (int it = 1; it <= iters; ++it) {
        if (threadIdx.x == 0) {
            long long s = 0;
            while (*f_ab != it && ++s < SPIN_MAX) {}
            if (s >= SPIN_MAX) { *err = 1; }
        }
        __syncthreads();
        __threadfence_system();
        for (int i = threadIdx.x; i < n; i += blockDim.x) { ba[i] = ab[i] + 1.f; }  // a stand-in for the helper GPU's expert work
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) { *f_ba = it; }
        if (*err) { return; }
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int a = argc > 1 ? atoi(argv[1]) : 1, b = argc > 2 ? atoi(argv[2]) : 0, iters = argc > 3 ? atoi(argv[3]) : 2000;
    int can = 0;
    CK(cudaDeviceCanAccessPeer(&can, a, b));
    cudaDeviceProp pa, pb;
    CK(cudaGetDeviceProperties(&pa, a));
    CK(cudaGetDeviceProperties(&pb, b));
    printf("a = GPU %d (%s), b = GPU %d (%s), peer access %d, %d iterations per row\n", a, pa.name, b, pb.name, can, iters);
    printf("%10s %12s %12s %12s   (us per round trip: a -> b -> a)\n", "payload", "host-sync", "event-chain", "mapped-flag");

    for (int bytes : { 4096, 16384, 65536 }) {
        const int n = bytes / 4;
        float *bufA, *bufB;
        CK(cudaSetDevice(a)); CK(cudaMalloc(&bufA, bytes)); CK(cudaMemset(bufA, 0, bytes));
        CK(cudaSetDevice(b)); CK(cudaMalloc(&bufB, bytes)); CK(cudaMemset(bufB, 0, bytes));
        cudaStream_t sa, sb;
        cudaEvent_t ea, eb;
        CK(cudaSetDevice(a)); CK(cudaStreamCreate(&sa)); CK(cudaEventCreateWithFlags(&ea, cudaEventDisableTiming));
        CK(cudaSetDevice(b)); CK(cudaStreamCreate(&sb)); CK(cudaEventCreateWithFlags(&eb, cudaEventDisableTiming));
        const int blocks = (n + 255) / 256;

        auto host_sync = [&](int k) {
            for (int i = 0; i < k; ++i) {
                CK(cudaSetDevice(a)); touch<<<blocks, 256, 0, sa>>>(bufA, n, 1.f);
                CK(cudaMemcpyPeerAsync(bufB, b, bufA, a, bytes, sa)); CK(cudaStreamSynchronize(sa));
                CK(cudaSetDevice(b)); touch<<<blocks, 256, 0, sb>>>(bufB, n, 1.f);
                CK(cudaMemcpyPeerAsync(bufA, a, bufB, b, bytes, sb)); CK(cudaStreamSynchronize(sb));
            }
        };
        auto event_chain = [&](int k) {
            for (int i = 0; i < k; ++i) {
                CK(cudaSetDevice(a)); touch<<<blocks, 256, 0, sa>>>(bufA, n, 1.f);
                CK(cudaMemcpyPeerAsync(bufB, b, bufA, a, bytes, sa)); CK(cudaEventRecord(ea, sa));
                CK(cudaSetDevice(b)); CK(cudaStreamWaitEvent(sb, ea, 0)); touch<<<blocks, 256, 0, sb>>>(bufB, n, 1.f);
                CK(cudaMemcpyPeerAsync(bufA, a, bufB, b, bytes, sb)); CK(cudaEventRecord(eb, sb));
                CK(cudaSetDevice(a)); CK(cudaStreamWaitEvent(sa, eb, 0));
            }
            CK(cudaSetDevice(a)); CK(cudaStreamSynchronize(sa));
            CK(cudaSetDevice(b)); CK(cudaStreamSynchronize(sb));
        };
        auto timed = [&](auto && fn) {
            fn(50);
            std::vector<double> t;
            for (int r = 0; r < 5; ++r) { double t0 = now_us(); fn(iters); t.push_back((now_us() - t0) / iters); }
            return *std::min_element(t.begin(), t.end());
        };
        const double t_host = timed(host_sync), t_event = timed(event_chain);

        // mapped-flag: pinned, mapped host memory visible to both GPUs; one persistent kernel per GPU
        float * h_ab, * h_ba;
        int * h_f_ab, * h_f_ba, * h_err_a, * h_err_b;
        CK(cudaHostAlloc((void **) &h_ab, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &h_ba, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &h_f_ab, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &h_f_ba, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &h_err_a, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &h_err_b, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        double t_flag = -1;
        std::vector<double> tf;
        for (int r = 0; r < 5 && !(*h_err_a || *h_err_b); ++r) {
            *h_f_ab = 0; *h_f_ba = 0; *h_err_a = 0; *h_err_b = 0;
            CK(cudaSetDevice(b)); pong<<<1, 256, 0, sb>>>(h_ab, h_ba, h_f_ab, h_f_ba, n, iters, h_err_b);
            CK(cudaSetDevice(a));
            double t0 = now_us();
            ping<<<1, 256, 0, sa>>>(h_ab, h_ba, h_f_ab, h_f_ba, n, iters, h_err_a);
            CK(cudaStreamSynchronize(sa));
            double t1 = now_us();
            CK(cudaSetDevice(b)); CK(cudaStreamSynchronize(sb));
            if (!*h_err_a && !*h_err_b) { tf.push_back((t1 - t0) / iters); }
        }
        if (!tf.empty()) { t_flag = *std::min_element(tf.begin(), tf.end()); }
        printf("%8d B %12.1f %12.1f %12.1f%s\n", bytes, t_host, t_event, t_flag, t_flag < 0 ? "  (flag lost: kernel gave up)" : "");
        fflush(stdout);
    }
    return 0;
}
