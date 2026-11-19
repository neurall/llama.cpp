#include "moe-bell.cuh"
#include "../ggml-bell-fork.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

// The bells this process allocated: host pointer -> the card's pointer to the same memory. An op whose bell is not here stays off the cards (supports_op).
static std::mutex g_bell_mu;
static std::unordered_map<void *, void *> g_bells;

// The memory is mapped for the card that is current when it is allocated: a bell made with another card current gave that card's kernels an address out of
// their range (Xid 13 on the second card, the first layer there never rang: OLMoE, build bell_token_one_graph, 2026-11-18). So the layer's own card is set first.
void * ggml_cuda_bell_alloc(int device, size_t size) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(device); GGML_UNUSED(size);
    return nullptr;   // no host-mapped pinned memory there (as allreduce.cu)
#else
    ggml_cuda_set_device(device);
    void * host = nullptr, * dev = nullptr;
    if (cudaHostAlloc(&host, size, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
    if (cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) { (void) cudaGetLastError(); cudaFreeHost(host); return nullptr; }
    memset(host, 0, size);
    std::lock_guard<std::mutex> lk(g_bell_mu);
    g_bells[host] = dev;
    return host;
#endif
}

void ggml_cuda_bell_free(void * ptr) {
    if (!ptr) { return; }
    {
        std::lock_guard<std::mutex> lk(g_bell_mu);
        if (!g_bells.erase(ptr)) { return; }
    }
    cudaFreeHost(ptr);
}

static ggml_bell * bell_dev(const ggml_tensor * op) {
    void * host; memcpy(&host, op->op_params, sizeof(host));
    std::lock_guard<std::mutex> lk(g_bell_mu);
    const auto it = g_bells.find(host);
    return it == g_bells.end() ? nullptr : (ggml_bell *) it->second;
}

bool ggml_cuda_moe_bell_supported(const ggml_tensor * op) {
    const ggml_tensor * ids = op->op == GGML_OP_MOE_BELL ? op->src[0] : op->src[1];
    return bell_dev(op) != nullptr && ids->ne[0] <= GGML_BELL_MAX_USED;
}

// one block; as ggml_bell_publish: the picks and their blocks into the bell, the layer input when a pick has no block, then seq counts up
static __global__ void k_moe_bell(const int32_t * __restrict__ ids, const int32_t * __restrict__ table, const float * __restrict__ x,
                                  int32_t * __restrict__ slots, ggml_bell * b, const int n_used, const int n_embd) {
    const int i = threadIdx.x;
    int miss = 0;
    if (i < n_used) {
        const int32_t id = ids[i];
        int32_t s = id >= 0 ? table[id] : -1;
        s = s < 0 ? -1 : s;
        slots[i]    = s;
        b->ids[i]   = id;
        b->slots[i] = s;
        miss = id >= 0 && s < 0;
    }
    const int n_miss = __syncthreads_count(miss);
    if (n_miss > 0) {
        float * bx = (float *) (b + 1);
        for (int j = i; j < n_embd; j += blockDim.x) { bx[j] = x[j]; }
    }
    __threadfence_system();   // every thread's writes are in host memory before seq moves
    __syncthreads();
    if (i == 0) {
        b->n_miss = n_miss;
        __threadfence_system();
        b->seq = b->seq + 1u;
        __threadfence_system();
    }
}

// as ggml_bell_rows_take. The picks and their blocks are read from the card's own tensors: a layer with every pick on the card touches no host memory.
// Many blocks, 16-byte reads: the rows come over PCIe, ~1 us a read; one block of 4-byte reads moved ~1 KB a microsecond (k_moe_rows 29 us a layer,
// 1.40 ms a token on Qwen3.8, trace 2026-11-19). Every block waits for the flag itself (one polling thread each), then copies its share.
template <bool vec4>
static __global__ void k_moe_rows(float * __restrict__ down, const int32_t * __restrict__ ids, const int32_t * __restrict__ slots, ggml_bell * b,
                                  const int n_used, const int n_embd) {
    const int i = threadIdx.x;
    const int miss = i < n_used && ids[i] >= 0 && slots[i] < 0;
    if (__syncthreads_count(miss) == 0) {
        if (i == 0 && blockIdx.x == 0) { b->rows_seq = (int32_t) b->seq; }
        return;
    }
    if (i == 0) {
        while (b->flag != b->seq) {
#if defined(GGML_USE_HIP)
            for (volatile int w = 0; w < 64; ++w) { }
#else
            __nanosleep(100);
#endif
        }
    }
    __syncthreads();
    const float * rows = (const float *) (b + 1) + n_embd;
    const int n = vec4 ? n_embd / 4 : n_embd;
    for (int e = 0; e < n_used; ++e) {
        if (ids[e] < 0 || slots[e] >= 0) { continue; }
        for (int j = blockIdx.x * blockDim.x + i; j < n; j += gridDim.x * blockDim.x) {
            if constexpr (vec4) {
                const volatile float4 * src = (const volatile float4 *) (rows + (size_t) e * n_embd) + j;
                const float4 v = make_float4(src->x, src->y, src->z, src->w);
                ((float4 *) (down + (size_t) e * n_embd))[j] = v;
            } else {
                down[(size_t) e * n_embd + j] = rows[(size_t) e * n_embd + j];
            }
        }
    }
    if (blockIdx.x == 0) {
        __syncthreads();
        if (i == 0) { b->rows_seq = (int32_t) b->seq; }
    }
}

void ggml_cuda_op_moe_bell(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids = dst->src[0], * table = dst->src[1], * x = dst->src[2];
    ggml_bell * b = bell_dev(dst);
    GGML_ASSERT(b != nullptr);
    k_moe_bell<<<1, 256, 0, ctx.stream()>>>((const int32_t *) ids->data, (const int32_t *) table->data, (const float *) x->data, (int32_t *) dst->data, b,
                                             (int) ids->ne[0], (int) x->ne[0]);
}

void ggml_cuda_op_moe_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * down = dst->src[0], * ids = dst->src[1], * slots = dst->src[2];
    ggml_bell * b = bell_dev(dst);
    GGML_ASSERT(b != nullptr && dst->data == down->data);
    const int n_embd = (int) down->ne[0];
    const float * rows = (const float *) (b + 1) + n_embd;   // device-side address of the mapped rows
    const bool vec4 = n_embd % 4 == 0 && ((uintptr_t) rows & 15) == 0 && ((uintptr_t) dst->data & 15) == 0;
    static const int n_blocks = getenv("GGML_CUDA_ROWS_BLOCKS") ? atoi(getenv("GGML_CUDA_ROWS_BLOCKS")) : 16;   // dev: copy blocks per layer
    const int nb = std::max(1, std::min(n_blocks, (vec4 ? n_embd / 4 : n_embd) / 256 + 1));
    if (vec4) {
        k_moe_rows<true><<<nb, 256, 0, ctx.stream()>>>((float *) dst->data, (const int32_t *) ids->data, (const int32_t *) slots->data, b, (int) ids->ne[0], n_embd);
    } else {
        k_moe_rows<false><<<nb, 256, 0, ctx.stream()>>>((float *) dst->data, (const int32_t *) ids->data, (const int32_t *) slots->data, b, (int) ids->ne[0], n_embd);
    }
}
