// fork (pool v2, user 2026-11-22): alloc on the card's pool map (src/llama-poolmap-fork.h). One block owns the map while it runs: every scan (free fit,
// coldest anchor, mover destinations, victims) is a parallel pass over the entries with a block-wide minimum; the edits (free + merge, split + place) shift
// the vector in place. The report goes to mapped host memory, `done` last. Same steps and tie-breaks as pv2_alloc_do (the host reference).
// A second kernel then copies the movers' bytes and points the expert tables at their new places (freed experts: -1).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
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

struct pm_shared { uint32_t cur, nofree, pick, pidx; unsigned long long sh_best, sh_key, sh_same; uint32_t sh_idx; uint64_t cov[PM_MAX_MV]; int64_t cov_hot[PM_MAX_MV]; uint32_t n_cov, n_mv, n_fr, ovf; };
// thread 0: the coldest unpinned expert at w (index vi, exactly n bytes at a multiple of a) leaves and the newcomer takes its entry in place
static __device__ void pm_take_inplace(pm_ctx & c, uint32_t vi, uint64_t a, int32_t id, int64_t hot, int pin, pm_result * res) {
    const pm_entry w = c.v[vi];
    res->fr[0] = { w.off, w.size, w.id, 0 };
    pm_entry & e = c.v[vi]; e.hot = hot; e.id = id; e.u = (uint32_t) a; e.type = PM_MOE; e.flags = pin ? PM_PIN : 0;
    res->s = w.off; res->n_mv = 0; res->n_fr = 1; res->overflow = 0;
    __threadfence_system();
    res->done = 1;
}
// the batch's takeable experts (key: heat, INT64_MAX for none; value: start, top bit: a window, not an exact slot), sorted coldest first (ties: lowest address)
struct pm_cands { const int64_t * key; const uint64_t * val; uint32_t n; uint64_t cn, ca; };
static __global__ void k_pm_cands(const pm_entry * v, const uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, const pm_req * rq, int64_t * key, uint64_t * val) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap) { return; }
    const uint32_t nc = *cnt; const uint64_t n = rq->n, a = rq->a ? rq->a : 1;
    int64_t k = INT64_MAX; uint64_t o = 0;
    if (i < nc && rq->kind != PM_ALLOC_KV && rq->kind != PM_ALLOC_WRK) {
        const pm_entry e = v[i];
        if (e.type == PM_MOE && !(e.flags & PM_PIN)) {
            pm_ctx c = { (pm_entry *) v, (uint32_t *) cnt, cap, lo, hi };
            if (e.size == n && e.off % a == 0) { k = e.hot; o = e.off; }
            else if (pm_window_ok(c, i, pm_align(e.off, a), n)) { k = e.hot; o = e.off | (1ull << 63); }
        }
    }
    key[i] = k; val[i] = o;
}
static __device__ void pm_alloc_one(pm_ctx & c, pm_shared & S, uint64_t n, uint64_t a, int32_t id, int64_t hot, int pin, int kind, pm_result * res, const pm_cands & cd) {
    const bool over = (pin & 2) == 0;   // pin bit 1: movers to free bytes only
    pin &= 1;
    unsigned long long & sh_best = S.sh_best; unsigned long long & sh_key = S.sh_key; uint32_t & sh_idx = S.sh_idx;
    uint64_t * cov = S.cov; int64_t * cov_hot = S.cov_hot; uint32_t & n_cov = S.n_cov; uint32_t & n_mv = S.n_mv; uint32_t & n_fr = S.n_fr; uint32_t & ovf = S.ovf;
    pm_entry * v = c.v; uint32_t * cnt = c.cnt;
    if (a == 0) { a = 1; }
    if (threadIdx.x == 0) { sh_best = ~0ull; sh_key = ~0ull; n_cov = 0; n_mv = 0; n_fr = 0; ovf = 0; }
    __syncthreads();
    const bool left = kind == PM_ALLOC_KV || kind == PM_ALLOC_WRK;   // user 2026-11-22: KV and work grow from the left end: the lowest window over free bytes and unpinned experts
    if (left) {
        for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
            const pm_entry & e = v[i];
            if (e.type != PM_FREE && (e.type != PM_MOE || (e.flags & PM_PIN))) { continue; }
            const uint64_t st = pm_align(e.off, a);
            if (pm_window_ok(c, i, st, n)) { atomicMin(&sh_best, (unsigned long long) st); }
        }
        __syncthreads();
    }
    // the stream's case, no scan: no free bytes hold n (the last request found none and changed no free bytes) and the batch's sorted takeable
    // experts are at hand: the coldest one still there leaves; an exact slot is taken in place, a window goes the long way below
    if (!left && cd.key && S.nofree && n == cd.cn && a == cd.ca) {
        if (threadIdx.x == 0) {
            S.pick = 0;
            const uint32_t nc = *cnt;
            while (S.cur < cd.n) {
                const int64_t k = cd.key[S.cur];
                if (k == INT64_MAX) { S.cur = cd.n; break; }
                const uint64_t o = cd.val[S.cur] & ~(1ull << 63); const bool win = cd.val[S.cur] >> 63;
                uint32_t l = 0, h = nc;
                while (l < h) { const uint32_t m = (l + h) / 2; if (v[m].off < o) { l = m + 1; } else { h = m; } }
                const bool ok = l < nc && v[l].off == o && v[l].type == PM_MOE && !(v[l].flags & PM_PIN) && v[l].hot == k;
                if (!ok) { S.cur++; continue; }
                if (win) { S.pick = 2; break; }
                S.pick = 1; S.pidx = l; S.cur++;
                break;
            }
            if (S.pick == 1) { res->path = 1; pm_take_inplace(c, S.pidx, a, id, hot, pin, res); }
        }
        __syncthreads();
        if (S.pick == 1) { return; }
    }
    if (threadIdx.x == 0) { S.nofree = 0; }
    // 1. a free entry that holds it, lowest address first; in the same pass (one read of the map: a single block is bound by load latency) the coldest
    // unpinned expert whose window covers only free bytes and unpinned experts (ties: lowest address), used by step 2 when no free entry holds it.
    // An expert of exactly n bytes at a multiple of a is its own window (no window check).
    {
        const uint32_t nc = *cnt;
        // user 2026-11-22: the coldest expert of the same size is swapped in place (no split, no merge, no shift of the vector); a window over other
        // sizes only when no same-size expert is takeable
        unsigned long long best = ~0ull, key = ~0ull, same = ~0ull;
        #pragma unroll 4
        for (uint32_t i = threadIdx.x; i < nc; i += PM_T) {
            const uint64_t off = v[i].off, sz = v[i].size; const uint8_t ty = v[i].type, fl = v[i].flags;
            if (ty == PM_FREE) {
                const uint64_t st = pm_align(off, a);
                if (st + n <= off + sz && st < best) { best = st; }
            } else if (!left && ty == PM_MOE && !(fl & PM_PIN)) {
                const uint64_t k = pm_key(v[i].hot, i);
                if (sz == n && off % a == 0) { if (k < same) { same = k; } }
                else if (k < key && pm_window_ok(c, i, pm_align(off, a), n)) { key = k; }
            }
        }
        if (threadIdx.x == 0) { S.sh_same = ~0ull; }
        __syncthreads();
        if (best != ~0ull) { atomicMin(&sh_best, best); }
        if (key != ~0ull) { atomicMin(&sh_key, key); }
        if (same != ~0ull) { atomicMin(&S.sh_same, same); }
    }
    __syncthreads();
    if (sh_best == ~0ull && !left && S.sh_same != ~0ull) { sh_key = S.sh_same; }   // uniform: every thread reads the same shared values
    uint64_t s = sh_best;
    if (s == ~0ull && !left) {
        // 2. the coldest takeable expert found above
        if (sh_key != ~0ull) {
            const uint32_t vi = (uint32_t) (sh_key & 0xffffffffu);
            s = pm_align(v[vi].off, a);
            // the stream's case: the coldest expert is exactly the size asked at its start; no same-size expert is colder (it would have won) and no free
            // bytes hold n, so it leaves and the newcomer takes its entry in place (no shift of the vector; same map and report as the general path)
            const pm_entry w = v[vi];
            if (w.off == s && w.size == n) {
                __syncthreads();
                if (threadIdx.x == 0) { res->path = 2; pm_take_inplace(c, vi, a, id, hot, pin, res); S.nofree = 1; }   // !left: an expert (no free bytes changed)
                __syncthreads();
                return;
            }
        }
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
        if (threadIdx.x == 0) { res->s = UINT64_MAX; res->n_mv = 0; res->n_fr = 0; res->overflow = 0; res->path = 4; __threadfence_system(); res->done = 1; S.nofree = 1; }   // no free bytes held it, the map is unchanged
        __syncthreads();
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
        if (to == ~0ull && over) {   // else the coldest same-size unpinned expert outside the window, colder than it (ties: lowest address)
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
    if (kind == PM_ALLOC_KV || kind == PM_ALLOC_WRK) { ne.type = kind == PM_ALLOC_KV ? PM_KV : PM_WRK; ne.id = -1; ne.hot = 0; ne.flags = PM_PIN; }   // held: KV never moves, work never in a pass
    if (!pm_place(c, s, ne, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
    if (threadIdx.x == 0) {
        res->s = s; res->n_mv = n_mv; res->n_fr = n_fr; res->overflow = ovf; res->path = 3;
        __threadfence_system();
        res->done = 1;
    }
    __syncthreads();
}
// user 2026-11-22: the host asks for a list of allocs (sizes may differ) in one launch and goes on; each gets its report, in order (later ones see the
// earlier ones' map)
static __global__ void __launch_bounds__(PM_T) k_pm_alloc(pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, const pm_req * rq, uint32_t n_rq, pm_result * res,
                                                         const int64_t * ckey, const uint64_t * cval) {
    __shared__ pm_shared S;
    pm_ctx c = { v, cnt, cap, lo, hi };
    const pm_cands cd = { ckey, cval, cap, rq[0].n, rq[0].a ? rq[0].a : 1 };
    if (threadIdx.x == 0) { S.cur = 0; S.nofree = 0; }
    __syncthreads();
    for (uint32_t k = 0; k < n_rq; ++k) {
        const pm_req q = rq[k];
        uint64_t t0 = 0; if (threadIdx.x == 0) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); }
        pm_alloc_one(c, S, q.n, q.a, q.id, q.hot, q.pin, q.kind, res + k, cd);
        if (threadIdx.x == 0) { uint64_t t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); res[k].ns = t1 - t0; }
    }
}

// the movers' bytes and the expert tables (tables[layer][expert] = offset / unit; freed: -1); one block per move or freed entry
static __global__ void k_pm_apply(const pm_result * resv, char * base, int32_t * const * tables, int32_t E) {
    const pm_result * res = resv;   // one request's report (the batch launches one apply per request, in order)
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

// the batch: one alloc launch for all requests, then one apply launch per request in order (request k's moves may read bytes request k-1 moved in)
// ev (optional): the alloc runs on st, the applies on st_apply after it (the table writes stay in order with that stream's table flushes and uploads)
void ggml_cuda_poolmap_alloc_batch_fork(cudaStream_t st, pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, const pm_req * rq_dev, uint32_t n_rq,
                                        pm_result * res_dev, char * base, int32_t * const * tables, int32_t E, cudaStream_t st_apply = 0, cudaEvent_t ev = nullptr,
                                        void * scratch = nullptr) {
    if (n_rq == 0) { return; }
    int64_t * ckey = nullptr; uint64_t * cval = nullptr;
    if (scratch && n_rq > 1) {   // scratch: pm_scratch_bytes(cap) bytes; the takeable experts for the first request's size, sorted by heat
        int64_t * k0 = (int64_t *) scratch; ckey = k0 + cap; uint64_t * v0 = (uint64_t *) (ckey + cap); cval = v0 + cap;
        void * tmp = cval + cap; size_t tb = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tb, k0, ckey, v0, cval, (int) cap, 0, 64, st));
        if (tb > ((size_t) 4 << 20)) { fprintf(stderr, "pool-map: the sort needs %zu bytes of scratch, %zu there\n", tb, (size_t) 4 << 20); abort(); }
        k_pm_cands<<<(cap + 255) / 256, 256, 0, st>>>(v, cnt, cap, lo, hi, rq_dev, k0, v0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tmp, tb, k0, ckey, v0, cval, (int) cap, 0, 64, st));
    }
    k_pm_alloc<<<1, PM_T, 0, st>>>(v, cnt, cap, lo, hi, rq_dev, n_rq, res_dev, ckey, cval);
    CUDA_CHECK(cudaGetLastError());
    if (ev) { CUDA_CHECK(cudaEventRecord(ev, st)); }
    if (!st_apply && ev) { return; }   // the caller queues the applies itself (ggml_cuda_poolmap_apply_fork) once it knows what they wait for
    if (ev) { CUDA_CHECK(cudaStreamWaitEvent(st_apply, ev, 0)); } else { st_apply = st; }
    for (uint32_t k = 0; k < n_rq; ++k) {
        k_pm_apply<<<dim3(PM_MAX_FR, 1), 256, 0, st_apply>>>(res_dev + k, base, tables, E);
        CUDA_CHECK(cudaGetLastError());
    }
}
// a batch's moves and frees as one flat list of independent ops (the host splits where one op depends on another): one block per op;
// n > 0: copy n bytes from -> to and set the table entry to / u; n == 0: a freed expert, table entry -1
static __global__ void k_pm_apply_ops(const pm_mv * ops, char * base, int32_t * const * tables, int32_t E) {
    const pm_mv m = ops[blockIdx.x];
    if (m.n) {
        const char * src = base + m.from; char * dst = base + m.to;
        if (((m.from | m.to | m.n) & 15) == 0) { for (uint64_t i = threadIdx.x; i < m.n / 16; i += blockDim.x) { ((uint4 *) dst)[i] = ((const uint4 *) src)[i]; } }
        else { for (uint64_t i = threadIdx.x; i < m.n; i += blockDim.x) { dst[i] = src[i]; } }
        __syncthreads();
    }
    if (threadIdx.x == 0 && m.id >= 0) { tables[m.id / E][m.id % E] = m.n ? (int32_t) (m.to / (m.u ? m.u : 1)) : -1; }
}
// one segment of a batch's ops on st (after what the caller queued there: the compute the freed and moved experts' layers wait for); ev: the alloc's
void ggml_cuda_poolmap_apply_fork(cudaStream_t st, cudaEvent_t ev, const pm_mv * ops_dev, uint32_t n_ops, char * base, int32_t * const * tables, int32_t E) {
    if (ev) { CUDA_CHECK(cudaStreamWaitEvent(st, ev, 0)); }
    if (n_ops == 0) { return; }
    k_pm_apply_ops<<<n_ops, 256, 0, st>>>(ops_dev, base, tables, E);
    CUDA_CHECK(cudaGetLastError());
}
void ggml_cuda_poolmap_alloc_fork(cudaStream_t st, pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, uint64_t n, uint64_t a,
                                  int32_t id, int64_t hot, int pin, pm_result * res_dev, char * base, int32_t * const * tables, int32_t E) {
    static pm_req * h = nullptr, * d = nullptr;   // one mapped request slot (the caller's thread owns the map: one alloc at a time)
    if (!h) { CUDA_CHECK(cudaHostAlloc((void **) &h, sizeof(pm_req), cudaHostAllocMapped | cudaHostAllocPortable)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &d, h, 0)); }
    CUDA_CHECK(cudaStreamSynchronize(st));   // the slot may still be read by the last launch
    *h = { n, a, hot, id, pin };
    ggml_cuda_poolmap_alloc_batch_fork(st, v, cnt, cap, lo, hi, d, 1, res_dev, base, tables, E);
}
