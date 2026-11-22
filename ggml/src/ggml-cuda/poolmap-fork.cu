// fork (pool v2, user 2026-11-22): alloc on the card's pool map (src/llama-poolmap-fork.h). One block owns the map while it runs: every scan (free fit,
// coldest anchor, mover destinations, victims) is a parallel pass over the entries with a block-wide minimum; the edits (free + merge, split + place) shift
// the vector in place. The report goes to mapped host memory, `done` last. Same steps and tie-breaks as pv2_alloc_do (the host reference).
// A second kernel then copies the movers' bytes and points the expert tables at their new places (freed experts: -1).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#endif
#include "../../../src/llama-poolmap-fork.h"

#define PM_T 1024

struct pm_ctx { pm_entry * v; uint32_t * cnt; uint32_t cap; uint64_t lo, hi; };

static __device__ uint64_t pm_key(int64_t hot, uint32_t idx) {
    int64_t h = hot + 2147483648ll;
    h = h < 0 ? 0 : h > 4294967295ll ? 4294967295ll : h;
    return ((uint64_t) h << 32) | idx;
}
static __device__ uint64_t pm_align(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// block-wide: entries [k+1, n) one place left (entry k is overwritten)
static __device__ void pm_remove(pm_ctx & c, uint32_t k) {
    const uint32_t n = *c.cnt;
    for (uint32_t b = k; b + 1 < n; b += PM_T) {
        const uint32_t j = b + threadIdx.x;
        pm_entry t;
        const bool on = j + 1 < n;
        if (on) { t = c.v[j + 1]; }
        __syncthreads();
        if (on) { c.v[j] = t; }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) { *c.cnt = n - 1; }
    __syncthreads();
}
// block-wide: entries [k, n) m places right; false: no room
static __device__ bool pm_insert(pm_ctx & c, uint32_t k, uint32_t m) {
    const uint32_t n = *c.cnt;
    if (n + m > c.cap) { return false; }
    for (int64_t e = n; e > (int64_t) k; e -= PM_T) {
        const int64_t j = e - PM_T + threadIdx.x;
        pm_entry t;
        const bool on = j >= (int64_t) k && j < e;
        if (on) { t = c.v[j]; }
        __syncthreads();
        if (on) { c.v[j + m] = t; }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) { *c.cnt = n + m; }
    __syncthreads();
    return true;
}
// block-wide: entry i becomes free and merges with free neighbours
static __device__ void pm_free(pm_ctx & c, uint32_t i) {
    if (threadIdx.x == 0) { pm_entry & e = c.v[i]; e.type = PM_FREE; e.id = -1; e.hot = 0; e.flags = 0; e.u = 0; }
    __syncthreads();
    if (i + 1 < *c.cnt && c.v[i + 1].type == PM_FREE) {
        if (threadIdx.x == 0) { c.v[i].size += c.v[i + 1].size; }
        __syncthreads();
        pm_remove(c, i + 1);
    }
    if (i > 0 && c.v[i - 1].type == PM_FREE) {
        if (threadIdx.x == 0) { c.v[i - 1].size += c.v[i].size; }
        __syncthreads();
        pm_remove(c, i);
    }
}
// block-wide: index of the entry that starts at x (UINT32_MAX: none)
static __device__ uint32_t pm_find_start(pm_ctx & c, uint64_t x, uint32_t * sh) {
    if (threadIdx.x == 0) { *sh = UINT32_MAX; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < *c.cnt; i += PM_T) { if (c.v[i].off == x) { atomicMin(sh, i); } }
    __syncthreads();
    const uint32_t r = *sh;
    __syncthreads();
    return r;
}
// block-wide: place e at [st, st + e.size) inside the free entry that holds it; false: none holds it, or no room to split
static __device__ bool pm_place(pm_ctx & c, uint64_t st, pm_entry e, uint32_t * sh) {
    if (threadIdx.x == 0) { *sh = UINT32_MAX; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < *c.cnt; i += PM_T) {
        const pm_entry & f = c.v[i];
        if (f.type == PM_FREE && f.off <= st && st + e.size <= f.off + f.size) { atomicMin(sh, i); }
    }
    __syncthreads();
    const uint32_t i = *sh;
    __syncthreads();
    if (i == UINT32_MAX) { return false; }
    const pm_entry f = c.v[i];
    const uint64_t pre = st - f.off, post = f.off + f.size - (st + e.size);
    const uint32_t extra = (pre > 0) + (post > 0);
    __syncthreads();
    if (extra && !pm_insert(c, i + 1, extra)) { return false; }
    if (threadIdx.x == 0) {
        uint32_t k = i;
        if (pre)  { pm_entry p = f; p.size = pre; c.v[k++] = p; }
        e.off = st; c.v[k++] = e;
        if (post) { pm_entry q = f; q.off = st + e.size; q.size = post; c.v[k] = q; }
    }
    __syncthreads();
    return true;
}
// block-wide: does [st, st + n) cover only free entries and unpinned experts? (starting at entry i0, the one that holds st or one before it)
static __device__ bool pm_window_ok(const pm_ctx & c, uint32_t i0, uint64_t st, uint64_t n) {
    if (st < c.lo || st + n > c.hi) { return false; }
    const uint32_t cnt = *c.cnt;
    uint32_t j = i0;
    while (j < cnt && c.v[j].off + c.v[j].size <= st) { ++j; }
    for (; j < cnt && c.v[j].off < st + n; ++j) {
        const pm_entry & e = c.v[j];
        if (e.type == PM_FREE) { continue; }
        if (e.type != PM_MOE || (e.flags & PM_PIN)) { return false; }
    }
    return true;
}

static __global__ void k_pm_alloc(pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, uint64_t n, uint64_t a, int32_t id, int64_t hot, int pin, pm_result * res) {
    __shared__ unsigned long long sh_best, sh_key;
    __shared__ uint32_t sh_idx;
    __shared__ uint64_t cov[PM_MAX_MV]; __shared__ int64_t cov_hot[PM_MAX_MV]; __shared__ uint32_t n_cov;
    __shared__ uint32_t n_mv, n_fr, ovf;
    pm_ctx c = { v, cnt, cap, lo, hi };
    if (a == 0) { a = 1; }
    if (threadIdx.x == 0) { sh_best = ~0ull; sh_key = ~0ull; n_cov = 0; n_mv = 0; n_fr = 0; ovf = 0; }
    __syncthreads();
    // 1. a free entry that holds it, lowest address first
    for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
        const pm_entry & e = v[i];
        if (e.type != PM_FREE) { continue; }
        const uint64_t st = pm_align(e.off, a);
        if (st + n <= e.off + e.size) { atomicMin(&sh_best, (unsigned long long) st); }
    }
    __syncthreads();
    uint64_t s = sh_best;
    if (s == ~0ull) {
        // 2. the coldest unpinned expert whose window covers only free bytes and unpinned experts (ties: lowest address)
        for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
            const pm_entry & e = v[i];
            if (e.type != PM_MOE || (e.flags & PM_PIN)) { continue; }
            if (pm_window_ok(c, i, pm_align(e.off, a), n)) { atomicMin(&sh_key, (unsigned long long) pm_key(e.hot, i)); }
        }
        __syncthreads();
        if (sh_key != ~0ull) { s = pm_align(v[(uint32_t) (sh_key & 0xffffffffu)].off, a); }
        else {
            // 3. a window starting in free bytes (lowest address)
            for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
                const pm_entry & e = v[i];
                if (e.type != PM_FREE) { continue; }
                const uint64_t st = pm_align(e.off, a);
                if (pm_window_ok(c, i, st, n)) { atomicMin(&sh_best, (unsigned long long) st); }
            }
            __syncthreads();
            s = sh_best;
        }
    }
    __syncthreads();
    if (s == ~0ull) {
        if (threadIdx.x == 0) { res->s = UINT64_MAX; res->n_mv = 0; res->n_fr = 0; res->overflow = 0; __threadfence_system(); res->done = 1; }
        return;
    }
    // the experts the window covers, hottest first (ties: lowest address)
    for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
        const pm_entry & e = v[i];
        if (e.type == PM_MOE && e.off < s + n && e.off + e.size > s) {
            const uint32_t k = atomicAdd(&n_cov, 1u);
            if (k < PM_MAX_MV) { cov[k] = e.off; cov_hot[k] = e.hot; } else { ovf = 1; }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t m = n_cov < PM_MAX_MV ? n_cov : PM_MAX_MV;
        for (uint32_t x = 1; x < m; ++x) {   // insertion sort: a handful of entries
            const uint64_t o = cov[x]; const int64_t h = cov_hot[x]; int32_t y = (int32_t) x - 1;
            while (y >= 0 && (cov_hot[y] < h || (cov_hot[y] == h && cov[y] > o))) { cov[y + 1] = cov[y]; cov_hot[y + 1] = cov_hot[y]; --y; }
            cov[y + 1] = o; cov_hot[y + 1] = h;
        }
        n_cov = m;
    }
    __syncthreads();
    for (uint32_t q = 0; q < n_cov; ++q) {
        const uint32_t ix = pm_find_start(c, cov[q], &sh_idx);
        const pm_entry r = v[ix];
        const uint64_t u = r.u ? r.u : 1;
        if (threadIdx.x == 0) { sh_best = ~0ull; sh_key = ~0ull; }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {   // a free entry outside the window, lowest address
            const pm_entry & f = v[i];
            if (f.type != PM_FREE) { continue; }
            const uint64_t st = pm_align(f.off, u);
            if (st + r.size <= f.off + f.size && (st >= s + n || st + r.size <= s)) { atomicMin(&sh_best, (unsigned long long) st); }
        }
        __syncthreads();
        uint64_t to = sh_best;
        if (to == ~0ull) {   // else the coldest same-size unpinned expert outside the window, colder than it (ties: lowest address)
            for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
                const pm_entry & w = v[i];
                if (w.type != PM_MOE || (w.flags & PM_PIN) || w.size != r.size || w.off % u != 0 || (w.off < s + n && w.off + w.size > s) || w.hot >= r.hot) { continue; }
                atomicMin(&sh_key, (unsigned long long) pm_key(w.hot, i));
            }
            __syncthreads();
            if (sh_key != ~0ull) {
                const uint32_t vi = (uint32_t) (sh_key & 0xffffffffu);
                const pm_entry w = v[vi];
                to = w.off;
                if (threadIdx.x == 0) { if (n_fr < PM_MAX_FR) { res->fr[n_fr++] = { w.off, w.size, w.id, 0 }; } else { ovf = 1; } }
                __syncthreads();
                pm_free(c, vi);
            }
        }
        const uint32_t ix2 = pm_find_start(c, r.off, &sh_idx);   // the victim's merge may have shifted it
        pm_free(c, ix2);
        if (to == ~0ull) {
            if (threadIdx.x == 0) { if (n_fr < PM_MAX_FR) { res->fr[n_fr++] = { r.off, r.size, r.id, 0 }; } else { ovf = 1; } }
            __syncthreads();
            continue;
        }
        pm_entry m = r; m.off = to;
        if (!pm_place(c, to, m, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
        if (threadIdx.x == 0) { if (n_mv < PM_MAX_MV) { res->mv[n_mv++] = { r.off, to, r.size, r.id, (uint32_t) u }; } else { ovf = 1; } }
        __syncthreads();
    }
    pm_entry ne = {}; ne.size = n; ne.hot = hot; ne.id = id; ne.u = (uint32_t) a; ne.type = PM_MOE; ne.flags = pin ? PM_PIN : 0;
    if (!pm_place(c, s, ne, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
    if (threadIdx.x == 0) {
        res->s = s; res->n_mv = n_mv; res->n_fr = n_fr; res->overflow = ovf;
        __threadfence_system();
        res->done = 1;
    }
}

// the movers' bytes and the expert tables (tables[layer][expert] = offset / unit; freed: -1); one block per move or freed entry
static __global__ void k_pm_apply(const pm_result * res, char * base, int32_t * const * tables, int32_t E) {
    const uint32_t b = blockIdx.x;
    if (b < res->n_mv && b < PM_MAX_MV) {
        const pm_mv m = res->mv[b];
        if (base) {
            const char * src = base + m.from; char * dst = base + m.to;
            if (((m.from | m.to | m.n) & 15) == 0) { for (uint64_t i = threadIdx.x; i < m.n / 16; i += blockDim.x) { ((uint4 *) dst)[i] = ((const uint4 *) src)[i]; } }
            else { for (uint64_t i = threadIdx.x; i < m.n; i += blockDim.x) { dst[i] = src[i]; } }
        }
        __syncthreads();
        if (threadIdx.x == 0 && tables && m.id >= 0) { tables[m.id / E][m.id % E] = (int32_t) (m.to / (m.u ? m.u : 1)); }
    }
    const uint32_t f = b;
    if (threadIdx.x == 0 && tables && f < res->n_fr && f < PM_MAX_FR && res->fr[f].id >= 0) { tables[res->fr[f].id / E][res->fr[f].id % E] = -1; }
}

void ggml_cuda_poolmap_alloc_fork(cudaStream_t st, pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, uint64_t n, uint64_t a,
                                  int32_t id, int64_t hot, int pin, pm_result * res_dev, char * base, int32_t * const * tables, int32_t E) {
    k_pm_alloc<<<1, PM_T, 0, st>>>(v, cnt, cap, lo, hi, n, a, id, hot, pin, res_dev);
    CUDA_CHECK(cudaGetLastError());
    k_pm_apply<<<PM_MAX_FR, 256, 0, st>>>(res_dev, base, tables, E);
    CUDA_CHECK(cudaGetLastError());
}
