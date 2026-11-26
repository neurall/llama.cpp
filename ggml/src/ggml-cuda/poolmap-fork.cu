// fork (pool v2, user 2026-11-22): alloc on the card's pool map (src/llama-poolmap-fork.h). One block owns the map while it runs: every scan (free fit,
// coldest anchor, mover destinations, victims) is a parallel pass over the entries with a block-wide minimum; the edits (free + merge, split + place) shift
// the vector in place. The report goes to mapped host memory, `done` last. Same steps and tie-breaks as pv2_alloc_do (the host reference).
// A second kernel then copies the movers' bytes and points the expert tables at their new places (freed experts: -1).
#ifdef GGML_USE_HIP
#include "vendors/hip.h"
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#endif
#include <cstdio>
#include <cstdlib>
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#endif
#include "../../../src/llama-poolmap-fork.h"

#define PM_T 1024

struct pm_lists { pm_cls * cls; uint32_t * ncls; pm_le * buf; pm_le * pend; uint32_t * pre; uint32_t cap; bool ok; };
struct pm_ctx { pm_entry * v; uint32_t * cnt; uint32_t cap; uint64_t lo, hi; const uint32_t * busy = nullptr; uint32_t E = 1; pm_lists L = {}; };
// a layer still ahead in the pass (busy bit): its experts are takeable by nobody in this batch, as a pinned one
static __device__ bool pm_busy(const pm_ctx & c, int32_t id) { if (!c.busy || id < 0) { return false; } const uint32_t l = (uint32_t) id / c.E; return (c.busy[l >> 5] >> (l & 31)) & 1u; }
static __device__ bool pm_held(const pm_ctx & c, const pm_entry & e) { return (e.flags & PM_PIN) || (e.type == PM_MOE && pm_busy(c, e.id)); }

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
// block-wide: does [st, st + n) cover only free entries and unpinned experts? (starting at entry i0, the one that holds st or one before it)
static __device__ bool pm_window_ok(const pm_ctx & c, uint32_t i0, uint64_t st, uint64_t n) {
    if (st < c.lo || st + n > c.hi) { return false; }
    const uint32_t cnt = *c.cnt;
    uint32_t j = i0;
    while (j < cnt && c.v[j].off + c.v[j].size <= st) { ++j; }
    for (; j < cnt && c.v[j].off < st + n; ++j) {
        const pm_entry & e = c.v[j];
        if (e.type == PM_FREE) { continue; }
        if (e.type != PM_MOE || pm_held(c, e)) { return false; }
    }
    return true;
}

// the batch's block-wide state
struct pm_shared {
    uint64_t nf_n, nf_a; uint32_t pick, pidx;   // nf: no free bytes hold nf_n bytes at a multiple of nf_a (0: not known)
    unsigned long long sh_best, sh_key, sh_same; uint32_t sh_idx; uint64_t cov[PM_MAX_MV]; int64_t cov_hot[PM_MAX_MV]; uint32_t n_cov, n_mv, n_fr, ovf;
    long long ph; unsigned long long pk, cn; uint32_t pp;   // a pop's pending pick; a new class found by the rebuild
    pm_le sp[PM_PEND];   // a merge's pending newcomers, sorted here
    typename cub::BlockScan<uint32_t, PM_T>::TempStorage scan;
};

// user 2026-11-22: the size-class lists (llama-poolmap-fork.h, pm_le). A list's live elements [head, head + len) of buffer buf are in (heat, start) order;
// dead ones keep their keys (the order holds), a merge drops them
static __device__ uint64_t pm_lk(const pm_entry & e) { return (e.off << 24) | ((uint64_t) ((uint32_t) e.id & 0x7fffffu) << 1); }
static __device__ uint64_t pm_loff(uint64_t k) { return k >> 24; }
static __device__ int32_t pm_lid(uint64_t k) { return (int32_t) ((k >> 1) & 0x7fffffu); }
static __device__ bool pm_lless(int64_t h1, uint64_t k1, int64_t h2, uint64_t k2) { return h1 < h2 || (h1 == h2 && pm_loff(k1) < pm_loff(k2)); }
static __device__ pm_le * pm_lb(const pm_ctx & c, int j, uint32_t b) { return c.L.buf + ((size_t) b * PM_NC + j) * c.L.cap; }
static __device__ pm_le * pm_lp(const pm_ctx & c, int j) { return c.L.pend + (size_t) j * PM_PEND; }
// shortcut: a class is (bytes, unit) and a request of n bytes at a multiple of a asks class (n, a); an expert of those bytes with another unit that is also a
// multiple of a is not in that class (the server's experts of one size share one unit), its requests would miss it: give classes a unit set if that changes
static __device__ int pm_cls_of(const pm_ctx & c, uint64_t n, uint64_t u) {
    if (!c.L.ok) { return -1; }
    const uint32_t m = *c.L.ncls;
    for (uint32_t j = 0; j < m; ++j) { if (c.L.cls[j].n == n && c.L.cls[j].u == u) { return (int) j; } }
    return -1;
}
static __device__ int pm_elig(const pm_ctx & c, const pm_entry & e) { return e.type == PM_MOE && !(e.flags & PM_PIN) ? pm_cls_of(c, e.size, e.u ? e.u : 1) : -1; }
// lower bound of (h, k) in a sorted run
static __device__ uint32_t pm_lbound(const pm_le * a, uint32_t n, int64_t h, uint64_t k) {
    uint32_t l = 0, r = n;
    while (l < r) { const uint32_t m = (l + r) / 2; if (pm_lless(a[m].hot, a[m].k, h, k)) { l = m + 1; } else { r = m; } }
    return l;
}
// block-wide: class j's pending newcomers sorted (in shared memory) and merged with its live elements into its other buffer; each element moves once
static __device__ void pm_merge(const pm_ctx & c, pm_shared & S, int j) {
    __syncthreads();
    pm_cls & C = c.L.cls[j];
    const uint32_t np = C.np, len = C.len, head = C.head, b = C.buf;
    __syncthreads();
    if (np == 0 && len == 0) { return; }
    const pm_le * src = pm_lb(c, j, b) + head; pm_le * dst = pm_lb(c, j, b ^ 1u); const pm_le * pd = pm_lp(c, j);
    uint32_t P = 1; while (P < np) { P <<= 1; }
    for (uint32_t i = threadIdx.x; i < P; i += PM_T) { S.sp[i] = i < np ? pd[i] : pm_le { INT64_MAX, ~0ull }; }
    __syncthreads();
    for (uint32_t k = 2; k <= P; k <<= 1) {   // bitonic: at most PM_PEND elements
        for (uint32_t g = k >> 1; g > 0; g >>= 1) {
            for (uint32_t i = threadIdx.x; i < P; i += PM_T) {
                const uint32_t x = i ^ g;
                if (x > i) {
                    const pm_le p = S.sp[i], q = S.sp[x];
                    const bool up = (i & k) == 0;
                    if (up ? pm_lless(q.hot, q.k, p.hot, p.k) : pm_lless(p.hot, p.k, q.hot, q.k)) { S.sp[i] = q; S.sp[x] = p; }
                }
            }
            __syncthreads();
        }
    }
    // pre[i]: live elements before i
    uint32_t base = 0;
    for (uint32_t t = 0; t < len; t += PM_T) {
        const uint32_t i = t + threadIdx.x;
        const uint32_t live = i < len ? (uint32_t) ((src[i].k & 1u) == 0) : 0u;
        uint32_t ex, agg;
        cub::BlockScan<uint32_t, PM_T>(S.scan).ExclusiveSum(live, ex, agg);
        if (i < len) { c.L.pre[i] = base + ex; }
        base += agg;
        __syncthreads();
    }
    if (threadIdx.x == 0) { c.L.pre[len] = base; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < len; i += PM_T) {
        const pm_le e = src[i];
        if (e.k & 1u) { continue; }
        dst[c.L.pre[i] + pm_lbound(S.sp, np, e.hot, e.k)] = e;
    }
    for (uint32_t q = threadIdx.x; q < np; q += PM_T) {
        const pm_le e = S.sp[q];
        dst[c.L.pre[pm_lbound(src, len, e.hot, e.k)] + q] = e;
    }
    __syncthreads();
    if (threadIdx.x == 0) { C.len = base + np; C.head = 0; C.np = 0; C.buf = b ^ 1u; }
    __syncthreads();
}
// block-wide: e (as it is in the map) joins its class's list (pending)
static __device__ void pm_list_add(const pm_ctx & c, pm_shared & S, const pm_entry & e) {
    const int j = pm_elig(c, e);
    if (j < 0) { return; }
    __syncthreads();
    if (c.L.cls[j].np == PM_PEND) { pm_merge(c, S, j); }
    if (threadIdx.x == 0) { pm_cls & C = c.L.cls[j]; pm_lp(c, j)[C.np] = { e.hot, pm_lk(e) }; C.np++; }
    __syncthreads();
}
// block-wide: e (as it was in the map) leaves its class's list: dead in place, or out of the pending ones
static __device__ void pm_list_del(const pm_ctx & c, pm_shared & S, const pm_entry & e) {
    const int j = pm_elig(c, e);
    if (j < 0) { return; }
    __syncthreads();
    if (threadIdx.x == 0) {
        pm_cls & C = c.L.cls[j]; pm_le * a = pm_lb(c, j, C.buf) + C.head;
        const uint32_t i = pm_lbound(a, C.len, e.hot, pm_lk(e));
        S.pick = 0;
        if (i < C.len && pm_loff(a[i].k) == e.off && !(a[i].k & 1u)) {
            a[i].k |= 1u; S.pick = 1;
            while (C.len && (a[0].k & 1u)) { ++a; C.head++; C.len--; }
        }
        S.sh_idx = UINT32_MAX;
    }
    __syncthreads();
    if (!S.pick) {
        const pm_le * pd = pm_lp(c, j); const uint32_t np = c.L.cls[j].np;
        for (uint32_t q = threadIdx.x; q < np; q += PM_T) { if (pm_loff(pd[q].k) == e.off) { S.sh_idx = q; } }
        __syncthreads();
        if (threadIdx.x == 0 && S.sh_idx != UINT32_MAX) { pm_cls & C = c.L.cls[j]; pm_le * p = pm_lp(c, j); p[S.sh_idx] = p[C.np - 1]; C.np--; }
    }
    __syncthreads();
}
// block-wide: class j's coldest element that is takeable now (layer not busy, colder than cap_hot) leaves its list; its start, UINT64_MAX: none
static __device__ uint64_t pm_pop(const pm_ctx & c, pm_shared & S, int j, int64_t cap_hot) {
    __syncthreads();
    pm_cls & C = c.L.cls[j];
    const uint32_t len = C.len, np = C.np;
    pm_le * a = pm_lb(c, j, C.buf) + C.head; pm_le * pd = pm_lp(c, j);
    if (threadIdx.x == 0) { S.sh_idx = UINT32_MAX; S.ph = LLONG_MAX; S.pk = ~0ull; S.pp = UINT32_MAX; }
    __syncthreads();
    for (uint32_t t = 0; t < len; t += PM_T) {   // the list is in heat order: the first takeable one is its coldest
        const uint32_t i = t + threadIdx.x;
        if (i < len) { const pm_le e = a[i]; if (!(e.k & 1u) && e.hot < cap_hot && !pm_busy(c, pm_lid(e.k))) { atomicMin(&S.sh_idx, i); } }
        __syncthreads();
        const bool got = S.sh_idx != UINT32_MAX || a[t + PM_T - 1 < len ? t + PM_T - 1 : len - 1].hot >= cap_hot;
        __syncthreads();
        if (got) { break; }
    }
    for (uint32_t q = threadIdx.x; q < np; q += PM_T) { const pm_le e = pd[q]; if (e.hot < cap_hot && !pm_busy(c, pm_lid(e.k))) { atomicMin(&S.ph, (long long) e.hot); } }
    __syncthreads();
    for (uint32_t q = threadIdx.x; q < np; q += PM_T) { const pm_le e = pd[q]; if (e.hot == S.ph && e.hot < cap_hot && !pm_busy(c, pm_lid(e.k))) { atomicMin(&S.pk, (unsigned long long) pm_loff(e.k)); } }
    __syncthreads();
    for (uint32_t q = threadIdx.x; q < np; q += PM_T) { if (S.pk != ~0ull && pm_loff(pd[q].k) == S.pk) { S.pp = q; } }
    __syncthreads();
    uint64_t r = UINT64_MAX;
    const bool fl = S.sh_idx != UINT32_MAX, fp = S.pp != UINT32_MAX;
    const bool use_l = fl && (!fp || pm_lless(a[S.sh_idx].hot, a[S.sh_idx].k, pd[S.pp].hot, pd[S.pp].k));
    if (use_l) { r = pm_loff(a[S.sh_idx].k); } else if (fp) { r = pm_loff(pd[S.pp].k); }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (use_l) { a[S.sh_idx].k |= 1u; while (C.len && (a[0].k & 1u)) { ++a; C.head++; C.len--; } }
        else if (fp) { pd[S.pp] = pd[C.np - 1]; C.np--; }
    }
    __syncthreads();
    return r;
}
// block-wide: the lists from the map as it is (sent whole): classes from its takeable experts, every one of them pending, merged per PM_PEND
static __device__ void pm_rebuild(const pm_ctx & c, pm_shared & S) {
    if (threadIdx.x == 0) { *c.L.ncls = 0; for (int j = 0; j < PM_NC; ++j) { c.L.cls[j] = { 0, 0, 0, 0, 0, 0 }; } }
    __syncthreads();
    const uint32_t n = *c.cnt;
    for (uint32_t t = 0; t < n; t += PM_T) {
        const uint32_t i = t + threadIdx.x;
        while (true) {   // classes this tile brings, one at a time
            if (threadIdx.x == 0) { S.cn = ~0ull; }
            __syncthreads();
            if (i < n) { const pm_entry & e = c.v[i]; if (e.type == PM_MOE && !(e.flags & PM_PIN) && pm_cls_of(c, e.size, e.u ? e.u : 1) < 0) { atomicMin(&S.cn, (unsigned long long) i); } }
            __syncthreads();
            if (S.cn == ~0ull || *c.L.ncls >= (uint32_t) PM_NC) { __syncthreads(); break; }
            if (threadIdx.x == 0) { const uint32_t m = *c.L.ncls; const pm_entry & e = c.v[S.cn]; c.L.cls[m].n = e.size; c.L.cls[m].u = e.u ? e.u : 1; *c.L.ncls = m + 1; }
            __syncthreads();
        }
        for (int j = 0; j < (int) *c.L.ncls; ++j) { if (c.L.cls[j].np + PM_T > PM_PEND) { pm_merge(c, S, j); } }
        __syncthreads();
        if (i < n) {
            const pm_entry e = c.v[i]; const int j = pm_elig(c, e);
            if (j >= 0) { const uint32_t q = atomicAdd(&c.L.cls[j].np, 1u); pm_lp(c, j)[q] = { e.hot, pm_lk(e) }; }
        }
        __syncthreads();
    }
    for (int j = 0; j < (int) *c.L.ncls; ++j) { pm_merge(c, S, j); }
}

// block-wide: entries [k + m, n) m places left (entries [k, k + m) are overwritten)
static __device__ void pm_remove_n(pm_ctx & c, uint32_t k, uint32_t m) {
    const uint32_t n = *c.cnt;
    for (uint32_t b = k; b + m < n; b += PM_T) {
        const uint32_t j = b + threadIdx.x;
        pm_entry t;
        const bool on = j + m < n;
        if (on) { t = c.v[j + m]; }
        __syncthreads();
        if (on) { c.v[j] = t; }
        __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) { *c.cnt = n - m; }
    __syncthreads();
}
// block-wide: entry i becomes free and merges with free neighbours (an expert leaves its list)
static __device__ void pm_free(pm_ctx & c, pm_shared & S, uint32_t i) {
    const pm_entry old = c.v[i];
    pm_list_del(c, S, old);
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
// block-wide: place e at [st, st + e.size) inside the free entry that holds it; false: none holds it, or no room to split (it joins its list)
static __device__ bool pm_place(pm_ctx & c, pm_shared & S, uint64_t st, pm_entry e, uint32_t * sh) {
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
    e.off = st;
    pm_list_add(c, S, e);
    return true;
}
// thread 0: index of the entry that starts at x (binary search: the map is in address order); cnt when none
static __device__ uint32_t pm_bs(const pm_ctx & c, uint64_t x) {
    uint32_t l = 0, h = *c.cnt;
    while (l < h) { const uint32_t m = (l + h) / 2; if (c.v[m].off < x) { l = m + 1; } else { h = m; } }
    return l;
}
// block-wide: the host's changes since the last batch (pm_diff): each run of the card's entries replaced by the host's, the lists with them
static __device__ void pm_apply_delta(pm_ctx & c, pm_shared & S, const pm_batch & bt) {
    for (uint32_t q = 0; q < bt.n_ops; ++q) {
        const pm_delta d = bt.ops[q];
        if (threadIdx.x == 0) { S.pidx = pm_bs(c, d.a); S.sh_idx = pm_bs(c, d.b); }
        __syncthreads();
        const uint32_t i0 = S.pidx, i1 = S.sh_idx;
        __syncthreads();
        for (uint32_t i = i0; i < i1; ++i) { const pm_entry e = c.v[i]; pm_list_del(c, S, e); }
        const uint32_t old = i1 - i0;
        if (d.ne > old) { if (!pm_insert(c, i1, d.ne - old)) { if (threadIdx.x == 0) { S.ovf = 2; } __syncthreads(); return; } }
        else if (d.ne < old) { pm_remove_n(c, i0 + d.ne, old - d.ne); }
        for (uint32_t x = threadIdx.x; x < d.ne; x += PM_T) { c.v[i0 + x] = bt.ents[d.e0 + x]; }
        __syncthreads();
        for (uint32_t x = 0; x < d.ne; ++x) { const pm_entry e = c.v[i0 + x]; pm_list_add(c, S, e); }
    }
}

// thread 0: the expert at vi (exactly n bytes at a multiple of a, out of its list already) leaves and the newcomer takes its entry in place
static __device__ void pm_take_inplace(pm_ctx & c, uint32_t vi, uint64_t a, int32_t id, int64_t hot, int pin, pm_result * res) {
    const pm_entry w = c.v[vi];
    res->fr[0] = { w.off, w.size, w.id, 0 };
    pm_entry & e = c.v[vi]; e.hot = hot; e.id = id; e.u = (uint32_t) a; e.type = PM_MOE; e.flags = pin ? PM_PIN : 0;
    res->s = w.off; res->n_mv = 0; res->n_fr = 1; res->overflow = 0;
    __threadfence_system();
    res->done = 1;
}
static constexpr int64_t PM_TIER = 1000000000000000ll;   // the decode set's heat floor (pv2_refresh tiered: 1e15 + score)
static __device__ void pm_alloc_one(pm_ctx & c, pm_shared & S, uint64_t n, uint64_t a, int32_t id, int64_t hot, int pin, int kind, pm_result * res) {
    const bool over = (pin & 2) == 0;   // pin bit 1: movers to free bytes only
    // pin bit 2 (user 2026-11-22, an entry planned ahead): a newcomer below the decode tier takes no expert of it (heat >= PM_TIER) and opens no window
    const int64_t cap_hot = (pin & 8) ? hot : (pin & 4) && hot < PM_TIER ? PM_TIER : INT64_MAX;   // bit 3 (a KV stretch's resident moving out): only a victim colder than it
    // bit 4 (--moe tier-guard=1, A/B): a newcomer below the decode tier takes no same-size expert of it, but may open a window over colder bytes
    const int64_t cap_take = (pin & 16) && hot < PM_TIER && cap_hot > PM_TIER ? PM_TIER : cap_hot;
    // bit 5 (--moe xsize-pop=1, A/B): a newcomer below the decode tier takes a colder-than-tier expert of a larger size before a decode-set one of its own
    const bool xs = (pin & 32) && hot < PM_TIER;
    pin &= 1;
    unsigned long long & sh_best = S.sh_best; unsigned long long & sh_key = S.sh_key; uint32_t & sh_idx = S.sh_idx;
    uint64_t * cov = S.cov; int64_t * cov_hot = S.cov_hot; uint32_t & n_cov = S.n_cov; uint32_t & n_mv = S.n_mv; uint32_t & n_fr = S.n_fr; uint32_t & ovf = S.ovf;
    pm_entry * v = c.v; uint32_t * cnt = c.cnt;
    if (a == 0) { a = 1; }
    if (threadIdx.x == 0) { sh_best = ~0ull; sh_key = ~0ull; n_cov = 0; n_mv = 0; n_fr = 0; ovf = 0; }
    __syncthreads();
    const bool left = kind == PM_ALLOC_KV || kind == PM_ALLOC_WRK;   // user 2026-11-22: KV and work grow from the left end: the lowest window over free bytes and unpinned experts
    const int jc = left ? -1 : pm_cls_of(c, n, a);   // the size class's list: its coldest takeable expert in O(1)
    // the newcomer takes the popped expert's entry in place; path 1: no scan before (no free bytes hold n), 2: after the scan found none
    auto take_popped = [&](uint64_t o, uint32_t path) {
        if (threadIdx.x == 0) {
            const uint32_t vi = pm_bs(c, o); S.pidx = vi; res->path = path;
            if (vi < *cnt && v[vi].off == o && v[vi].size == n) { pm_take_inplace(c, vi, a, id, hot, pin, res); S.nf_n = n; S.nf_a = a; }
            else { res->s = UINT64_MAX; res->n_mv = 0; res->n_fr = 0; res->overflow = 3; __threadfence_system(); res->done = 1; S.pidx = UINT32_MAX; }   // a list out of step with the map: the host aborts
        }
        __syncthreads();
        if (S.pidx == UINT32_MAX) { return; }
        const pm_entry ne = v[S.pidx];
        pm_list_add(c, S, ne);
    };
    // the coldest below-tier expert of the next larger size that holds n at a multiple of a leaves; the newcomer lands at its aligned start, the rest of
    // its bytes free (path 6). false: none (the map unchanged)
    auto xpop = [&]() -> bool {
        if (!xs || !c.L.ok) { return false; }
        const uint32_t m = *c.L.ncls;
        uint64_t lo = n;
        for (uint32_t step = 0; step < m; ++step) {
            int jb = -1; uint64_t nb = UINT64_MAX;
            for (uint32_t j = 0; j < m; ++j) { const uint64_t nj = c.L.cls[j].n; if (nj > lo && nj < nb) { nb = nj; jb = (int) j; } }
            if (jb < 0) { return false; }
            lo = nb;
            const uint64_t o = pm_pop(c, S, jb, PM_TIER);
            if (o == UINT64_MAX) { continue; }
            if (threadIdx.x == 0) { S.pidx = pm_bs(c, o); }
            __syncthreads();
            const uint32_t vi = S.pidx;
            const bool at = vi < *cnt && v[vi].off == o;
            const pm_entry w = at ? v[vi] : pm_entry {};
            const uint64_t st = pm_align(o, a);
            __syncthreads();
            if (!at) {   // a list out of step with the map: the host aborts
                if (threadIdx.x == 0) { res->s = UINT64_MAX; res->n_mv = 0; res->n_fr = 0; res->overflow = 3; res->path = 6; __threadfence_system(); res->done = 1; }
                __syncthreads();
                return true;
            }
            if (st + n > o + w.size) { pm_list_add(c, S, w); continue; }   // its unit misses a: it stays
            pm_free(c, S, vi);
            pm_entry ne = {}; ne.size = n; ne.hot = hot; ne.id = id; ne.u = (uint32_t) a; ne.type = PM_MOE; ne.flags = pin ? PM_PIN : 0;
            if (!pm_place(c, S, st, ne, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
            if (threadIdx.x == 0) {
                res->fr[0] = { w.off, w.size, w.id, 0 };
                res->s = st; res->n_mv = 0; res->n_fr = 1; res->overflow = ovf; res->path = 6; S.nf_n = 0;
                __threadfence_system();
                res->done = 1;
            }
            __syncthreads();
            return true;
        }
        return false;
    };
    // own size below the tier, then a larger size below it, then own size up to the cap
    auto pop_take = [&](uint32_t path) -> bool {
        if (xs && cap_take > PM_TIER) {
            const uint64_t o = pm_pop(c, S, jc, PM_TIER);
            if (o != UINT64_MAX) { take_popped(o, path); return true; }
        }
        if (xpop()) { return true; }
        const uint64_t o = pm_pop(c, S, jc, cap_take);
        if (o != UINT64_MAX) { take_popped(o, path); return true; }
        return false;
    };
    if (left) {
        for (uint32_t i = threadIdx.x; i < *cnt; i += PM_T) {
            const pm_entry & e = v[i];
            if (e.type != PM_FREE && (e.type != PM_MOE || pm_held(c, e))) { continue; }
            const uint64_t st = pm_align(e.off, a);
            if (pm_window_ok(c, i, st, n)) { atomicMin(&sh_best, (unsigned long long) st); }
        }
        __syncthreads();
    }
    // the stream's case, no scan: no free bytes hold n (the last request found none and changed no free bytes): the class's coldest leaves
    if (jc >= 0 && S.nf_n == n && S.nf_a == a) {
        if (pop_take(1)) { return; }
    }
    __syncthreads();
    if (threadIdx.x == 0) { S.nf_n = 0; }
    // 1. a free entry that holds it, lowest address first; in the same pass, for a size with no list, the coldest same-size takeable expert
    {
        const uint32_t nc = *cnt;
        // user 2026-11-22: the coldest expert of the same size is swapped in place (no split, no merge, no shift of the vector); a window over other
        // sizes only when no same-size expert is takeable
        unsigned long long best = ~0ull, same = ~0ull;
        #pragma unroll 4
        for (uint32_t i = threadIdx.x; i < nc; i += PM_T) {
            const uint64_t off = v[i].off, sz = v[i].size; const uint8_t ty = v[i].type;
            if (ty == PM_FREE) {
                const uint64_t st = pm_align(off, a);
                if (st + n <= off + sz && st < best) { best = st; }
            } else if (!left && jc < 0 && ty == PM_MOE && !pm_held(c, v[i])) {
                const uint64_t k = pm_key(v[i].hot, i);
                if (sz == n && off % a == 0 && k < same && v[i].hot < cap_take) { same = k; }
            }
        }
        if (threadIdx.x == 0) { S.sh_same = ~0ull; }
        __syncthreads();
        if (best != ~0ull) { atomicMin(&sh_best, best); }
        if (same != ~0ull) { atomicMin(&S.sh_same, same); }
        __syncthreads();
        if (sh_best == ~0ull && jc >= 0) {
            if (pop_take(2)) { return; }
        }
        if (sh_best == ~0ull && S.sh_same == ~0ull && !left && cap_hot == INT64_MAX) {   // windows (a walk each) only when no free bytes and no same-size expert hold it
            unsigned long long key = ~0ull;
            for (uint32_t i = threadIdx.x; i < nc; i += PM_T) {
                if (v[i].type != PM_MOE || pm_held(c, v[i])) { continue; }
                const uint64_t k = pm_key(v[i].hot, i);
                if (k < key && pm_window_ok(c, i, pm_align(v[i].off, a), n)) { key = k; }
            }
            if (key != ~0ull) { atomicMin(&sh_key, key); }
        }
    }
    __syncthreads();
    if (sh_best == ~0ull && !left && S.sh_same != ~0ull) { sh_key = S.sh_same; }   // uniform: every thread reads the same shared values
    uint64_t s = sh_best;
    if (s == ~0ull && !left) {
        // 2. the coldest takeable expert found above
        if (sh_key != ~0ull) {
            const uint32_t vi = (uint32_t) (sh_key & 0xffffffffu);
            s = pm_align(v[vi].off, a);
            // the coldest expert is exactly the size asked at its start: it leaves and the newcomer takes its entry in place
            const pm_entry w = v[vi];
            if (w.off == s && w.size == n) {
                pm_list_del(c, S, w);
                if (threadIdx.x == 0) { res->path = 2; pm_take_inplace(c, vi, a, id, hot, pin, res); S.nf_n = n; S.nf_a = a; }   // !left: an expert (no free bytes changed)
                __syncthreads();
                const pm_entry ne = v[vi];
                pm_list_add(c, S, ne);
                return;
            }
        }
        else if (cap_hot == INT64_MAX) {   // an entry ahead below the tier opens no window (it may cover decode-set experts)
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
        if (threadIdx.x == 0) { res->s = UINT64_MAX; res->n_mv = 0; res->n_fr = 0; res->overflow = 0; res->path = 4; __threadfence_system(); res->done = 1; S.nf_n = n; S.nf_a = a; }   // no free bytes held it, the map is unchanged
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
                if (w.type != PM_MOE || pm_held(c, w) || w.size != r.size || w.off % u != 0 || (w.off < s + n && w.off + w.size > s) || w.hot >= r.hot) { continue; }
                atomicMin(&sh_key, (unsigned long long) pm_key(w.hot, i));
            }
            __syncthreads();
            if (sh_key != ~0ull) {
                const uint32_t vi = (uint32_t) (sh_key & 0xffffffffu);
                const pm_entry w = v[vi];
                to = w.off;
                if (threadIdx.x == 0) { if (n_fr < PM_MAX_FR) { res->fr[n_fr++] = { w.off, w.size, w.id, 0 }; } else { ovf = 1; } }
                __syncthreads();
                pm_free(c, S, vi);
            }
        }
        const uint32_t ix2 = pm_find_start(c, r.off, &sh_idx);   // the victim's merge may have shifted it
        pm_free(c, S, ix2);
        if (to == ~0ull) {
            if (threadIdx.x == 0) { if (n_fr < PM_MAX_FR) { res->fr[n_fr++] = { r.off, r.size, r.id, 0 }; } else { ovf = 1; } }
            __syncthreads();
            continue;
        }
        pm_entry m = r; m.off = to;
        if (!pm_place(c, S, to, m, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
        if (threadIdx.x == 0) { if (n_mv < PM_MAX_MV) { res->mv[n_mv++] = { r.off, to, r.size, r.id, (uint32_t) u }; } else { ovf = 1; } }
        __syncthreads();
    }
    pm_entry ne = {}; ne.size = n; ne.hot = hot; ne.id = id; ne.u = (uint32_t) a; ne.type = PM_MOE; ne.flags = pin ? PM_PIN : 0;
    if (kind == PM_ALLOC_KV || kind == PM_ALLOC_WRK) { ne.type = kind == PM_ALLOC_KV ? PM_KV : PM_WRK; ne.id = -1; ne.hot = 0; ne.flags = PM_PIN; }   // held: KV never moves, work never in a pass
    if (!pm_place(c, S, s, ne, &sh_idx)) { if (threadIdx.x == 0) { ovf = 2; } __syncthreads(); }
    if (threadIdx.x == 0) {
        res->s = s; res->n_mv = n_mv; res->n_fr = n_fr; res->overflow = ovf; res->path = n_cov ? 3 : 5;   // 5 ("other" in the host log): a fit in free bytes, no window
        __threadfence_system();
        res->done = 1;
    }
    __syncthreads();
}
static __device__ pm_lists pm_lists_of(void * scratch, uint32_t cap) {
    pm_lists L = {};
    if (!scratch) { return L; }
    char * b = (char *) scratch;
    L.cls = (pm_cls *) b; L.ncls = (uint32_t *) (b + 2048);
    L.buf = (pm_le *) (b + 4096); L.pend = L.buf + (size_t) 2 * PM_NC * cap; L.pre = (uint32_t *) (L.pend + (size_t) PM_NC * PM_PEND); L.cap = cap; L.ok = true;
    return L;
}
// alloc time for the report; shortcut: HIP has no %globaltimer, reports 0 ns there, use wall_clock64() and its rate if HIP timings are needed
static __device__ __forceinline__ uint64_t pm_now_ns() {
#ifdef GGML_USE_HIP
    return 0;
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

// user 2026-11-22: the host asks for a list of allocs (sizes may differ) in one launch and goes on; each gets its report, in order (later ones see the
// earlier ones' map). First the host's changes since the last batch (or the lists built from a map sent whole), last the pending newcomers merged
static __global__ void __launch_bounds__(PM_T) k_pm_alloc(pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, const pm_req * rq, uint32_t n_rq, pm_result * res,
                                                         void * scratch, pm_batch bt) {
    __shared__ pm_shared S;
    pm_ctx c = { v, cnt, cap, lo, hi, bt.busy, bt.E ? bt.E : 1u, pm_lists_of(scratch, cap) };
    if (threadIdx.x == 0) { S.nf_n = 0; S.ovf = 0; }
    __syncthreads();
    if (c.L.ok && bt.rebuild) { pm_rebuild(c, S); }
    pm_apply_delta(c, S, bt);
    for (uint32_t k = 0; k < n_rq; ++k) {
        const pm_req q = rq[k];
        uint64_t t0 = 0; if (threadIdx.x == 0) { t0 = pm_now_ns(); }
        pm_alloc_one(c, S, q.n, q.a, q.id, q.hot, q.pin, q.kind, res + k);
        if (threadIdx.x == 0) { res[k].ns = pm_now_ns() - t0; }
    }
    if (c.L.ok) { for (int j = 0; j < (int) *c.L.ncls; ++j) { if (c.L.cls[j].np) { pm_merge(c, S, j); } } }
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
// scratch (pm_scratch_bytes(cap), optional): the size-class lists, kept on the card between batches; bt: the host's changes, busy layers, rebuild
void ggml_cuda_poolmap_alloc_batch_fork(cudaStream_t st, pm_entry * v, uint32_t * cnt, uint32_t cap, uint64_t lo, uint64_t hi, const pm_req * rq_dev, uint32_t n_rq,
                                        pm_result * res_dev, char * base, int32_t * const * tables, int32_t E, cudaStream_t st_apply = 0, cudaEvent_t ev = nullptr,
                                        void * scratch = nullptr, pm_batch bt = {}) {
    if (n_rq == 0 && bt.n_ops == 0 && !bt.rebuild) { return; }
    k_pm_alloc<<<1, PM_T, 0, st>>>(v, cnt, cap, lo, hi, rq_dev, n_rq, res_dev, scratch, bt);
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
