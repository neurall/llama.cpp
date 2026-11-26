#pragma once
// fork (pool v2, user 2026-11-22): the pool map as it lives on the card: one vector of entries that tile the pool in address order, each
// {type: free, moe, kv, wrk; size; heat counter} (+ id, unit, offset). The alloc and defrag kernels read and update it on the card and report what they
// did (pv2_alloc_result); the host applies the report to its mirror (pv2_alloc_apply). With --moe map-check=1 the host reads the vector back after every
// alloc and compares (pm_same). The reference of the kernel is pv2_alloc_do on the same map (pm_to_heap / pm_from_heap).
// Canonical form: no zero sizes, no two free entries next to each other, offsets = running sum of sizes.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <atomic>
#include "llama-poolv2-fork.h"

// the card writes the flag in mapped host memory: a volatile read each spin, then acquire (GCC's __atomic_load_n is not in MSVC, the W10 host compiler)
static inline uint32_t pm_load_acquire(const volatile uint32_t * p) { const uint32_t v = *p; std::atomic_thread_fence(std::memory_order_acquire); return v; }

enum : uint8_t { PM_FREE = 0, PM_MOE = 1, PM_KV = 2, PM_WRK = 3 };
enum : uint8_t { PM_PIN = 1, PM_MOV = 2 };
// what an alloc is for (optional, default an expert): a KV page (never moved) or a working buffer (never moved in a pass), both held
enum : int32_t { PM_ALLOC_MOE = 0, PM_ALLOC_KV = 1, PM_ALLOC_WRK = 2 };

struct pm_entry {
    uint64_t off;
    uint64_t size;
    int64_t  hot;
    int32_t  id;     // moe: the expert id (layer * E + expert); kv/wrk: the owner's id; free: -1
    uint32_t u;      // moe: its start stays a multiple of it
    uint8_t  type;
    uint8_t  flags;
    uint8_t  pad[6];
};
static_assert(sizeof(pm_entry) == 40, "pm_entry layout is shared with the kernel");

struct pm_mv { uint64_t from, to, n; int32_t id; uint32_t u; };   // u: the mover's unit (its table entry = to / u)
struct pm_fr { uint64_t s, n; int32_t id; int32_t pad; };
static constexpr int PM_MAX_MV = 64, PM_MAX_FR = 128;
// user 2026-11-22: one list per size class on the card, kept in order (heat, then address; coldest at the head) by every change, never sorted whole:
// a take pops the head, a leaving expert's element is marked dead in place (found by binary search), a newcomer waits in its class's pending buffer and
// the pending ones merge into the list in one pass (each element moves once). Element: heat, and (start << 24) | (id << 1) | dead.
static constexpr int PM_NC = 4;               // size classes (bytes, unit) with a list; a request of another size scans the map
static constexpr uint32_t PM_PEND = 2048;     // a class's pending newcomers before a merge
struct pm_le { int64_t hot; uint64_t k; };
struct pm_cls { uint64_t n, u; uint32_t head, len, np, buf; };
// what changed on the host since the card's map was last in step: the card's entries [a, b) become ents[e0, e0 + ne) (a diff by address, pm_diff)
struct pm_delta { uint64_t a, b; uint32_t e0, ne; };
// a batch's inputs besides the requests (device pointers: mapped memory). rebuild: the map was just sent whole, the lists are built from it.
// busy: one bit per layer, its experts are taken by nobody in this batch (layers still ahead in the pass); E experts per layer
struct pm_batch { const pm_delta * ops = nullptr; const pm_entry * ents = nullptr; const uint32_t * busy = nullptr; uint32_t n_ops = 0, rebuild = 0, E = 1, pad_ = 0; };
// the card's scratch behind the map's count (map buffer: cap entries, 256 bytes for the count, then this): classes, two list buffers a class, pending, prefix
static inline size_t pm_scratch_bytes(uint32_t cap) { return 4096 + (size_t) 2 * PM_NC * cap * sizeof(pm_le) + (size_t) PM_NC * PM_PEND * sizeof(pm_le) + ((size_t) cap + 1) * 4; }
struct pm_req { uint64_t n = 0, a = 1; int64_t hot = 0; int32_t id = -1; int32_t pin = 1; int32_t kind = PM_ALLOC_MOE; };   // one alloc of a batch (sizes may differ): { n } asks a size only; pin bit 0: pinned, bit 1: movers to free bytes only; kind optional
struct pm_result {
    uint64_t s;          // the new area, UINT64_MAX: none
    uint32_t n_mv, n_fr;
    uint32_t overflow;   // more moves or frees than the lists hold: the host reads the whole map back
    uint32_t done;       // set last (the host spins on it, bell style)
    uint32_t path;       // dev timing: 1 popped from the sorted list, 2 scan then in place, 3 free bytes / moves, 4 refused
    uint32_t pad_;
    uint64_t ns;         // dev timing: the card's time for this request (globaltimer)
    pm_mv    mv[PM_MAX_MV];
    pm_fr    fr[PM_MAX_FR];
};

static inline std::vector<pm_entry> pm_from_heap(const pv2_heap & h) {
    std::vector<pm_entry> v;
    size_t at = h.lo;
    auto gi = h.gaps.begin(); auto ri = h.res.begin();
    while (at < h.hi) {
        pm_entry e = {};
        e.off = at;
        if (gi != h.gaps.end() && gi->first == at) { e.size = gi->second; e.type = PM_FREE; e.id = -1; ++gi; }
        else if (ri != h.res.end() && ri->first == at) {
            const pv2_res & r = ri->second;
            e.size = r.n; e.hot = r.heat; e.u = (uint32_t) r.u;
            if (r.id == PV2_HOLD) { e.type = r.kv ? PM_KV : PM_WRK; e.id = -1; e.flags = (uint8_t) (PM_PIN | (r.mov ? PM_MOV : 0)); }
            else { e.type = PM_MOE; e.id = r.id; e.flags = r.pin ? PM_PIN : 0; }
            ++ri;
        } else { return {}; }   // a hole: not a tiling heap
        at += e.size;
        v.push_back(e);
    }
    return v;
}
static inline pv2_res pm_res_of(const pm_entry & e) {
    pv2_res r; r.n = e.size; r.heat = e.hot; r.u = e.u ? e.u : 1; r.pin = (e.flags & PM_PIN) != 0; r.mov = (e.flags & PM_MOV) != 0; r.kv = e.type == PM_KV;
    r.id = e.type == PM_MOE ? e.id : PV2_HOLD;
    return r;
}
static inline pv2_heap pm_to_heap(const std::vector<pm_entry> & v, size_t lo, size_t hi) {
    pv2_heap h = pv2_make(lo, hi);
    for (const pm_entry & e : v) {
        if (e.type == PM_FREE) { continue; }
        pv2_add(h, e.off, pm_res_of(e));
    }
    return h;
}
// two maps agree: same entries in the same order (heat may differ: the card counts uses the host does not see)
static inline bool pm_same(const std::vector<pm_entry> & a, const std::vector<pm_entry> & b, size_t * first_diff = nullptr, bool pins = true) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const pm_entry & x = a[i], & y = b[i];
        if (x.off != y.off || x.size != y.size || x.type != y.type || x.id != y.id || (pins && (x.flags & PM_PIN) != (y.flags & PM_PIN))) { if (first_diff) { *first_diff = i; } return false; }
    }
    if (a.size() != b.size()) { if (first_diff) { *first_diff = n; } return false; }
    return true;
}
// canonical check (the kernel's own invariants)
static inline bool pm_canonical(const std::vector<pm_entry> & v, size_t lo, size_t hi) {
    size_t at = lo;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i].off != at || v[i].size == 0) { return false; }
        if (i && v[i].type == PM_FREE && v[i - 1].type == PM_FREE) { return false; }
        at += v[i].size;
    }
    return at == hi;
}
// user 2026-11-22: alloc asks a size and gets a pointer. alloc(r) returns at once a fake pointer (PM_FAKE | its number); buffered = true leaves it
// in the open batch, else the batch goes at once (submit: one launch for every queued request). place(p) gives the real offset (UINT64_MAX: no
// place); it waits only when that request's report has not landed, then replays the reports in order on the host (mirror, tables, owners), every
// finished one of the batch with it. The ring holds cap requests: a full ring first resolves its oldest
static constexpr uint64_t PM_FAKE = 1ull << 63;
struct pm_queue {
    pm_req * rq = nullptr; pm_result * rs = nullptr; uint32_t cap = 0;   // ring slots (mapped memory: the card reads rq, writes rs)
    std::function<void(uint32_t slot0, uint32_t n)> launch;              // one batch launch: slots [slot0, slot0 + n)
    std::function<void(uint64_t h, const pm_req &, const pm_result &)> replay;   // a landed report, in request order
    uint64_t next = 0, sent = 0, done = 0;   // requests [done, sent) on the card, [sent, next) queued
    std::unordered_map<uint64_t, uint64_t> at;   // resolved: request number -> offset. shortcut: kept until forget(), callers forget once they hold the place
    uint64_t alloc(const pm_req & r, bool buffered = false) {
        if (next - done == cap) { reap(done + 1); }
        const uint32_t k = (uint32_t) (next % cap);
        rq[k] = r; rs[k].done = 0;
        const uint64_t h = next++;
        if (!buffered) { submit(); }
        return PM_FAKE | h;
    }
    void submit() {
        while (sent < next) {   // a batch that wraps the ring goes as two launches (the kernel takes contiguous slots)
            const uint32_t s0 = (uint32_t) (sent % cap), n = (uint32_t) std::min<uint64_t>(next - sent, cap - s0);
            launch(s0, n);
            sent += n;
        }
    }
    void reap(uint64_t upto) {   // requests below upto resolved (waits for them), then every later one already landed
        if (upto > sent) { submit(); }
        for (; done < sent; ++done) {
            const uint32_t k = (uint32_t) (done % cap);
            if (done >= upto && pm_load_acquire(&rs[k].done) == 0) { break; }
            while (pm_load_acquire(&rs[k].done) == 0) { }
            if (replay) { replay(done, rq[k], rs[k]); }
            at[done] = rs[k].s;
        }
    }
    uint64_t place(uint64_t p) {
        if (!(p & PM_FAKE)) { return p; }
        const uint64_t h = p & ~PM_FAKE;
        if (h >= done) { reap(h + 1); }
        return at.at(h);
    }
    void forget(uint64_t p) { at.erase(p & ~PM_FAKE); }
};

// the card's map A in step with the host's B: runs of entries that differ, each [a, b) the same bytes on both sides (both tile the pool). Entries are the
// same when every field is (heat and pins included: they order and gate the lists)
static inline bool pm_eq(const pm_entry & x, const pm_entry & y) { return x.off == y.off && x.size == y.size && x.hot == y.hot && x.id == y.id && x.u == y.u && x.type == y.type && x.flags == y.flags; }
static inline void pm_diff(const std::vector<pm_entry> & A, const std::vector<pm_entry> & B, std::vector<pm_delta> & ops, std::vector<pm_entry> & ents) {
    ops.clear(); ents.clear();
    size_t i = 0, j = 0;
    while (i < A.size() && j < B.size()) {
        if (pm_eq(A[i], B[j])) { ++i; ++j; continue; }
        const size_t i0 = i, j0 = j;
        while (true) {   // to the next boundary both share, then on while the entries there still differ
            uint64_t ea = A[i].off + A[i].size, eb = B[j].off + B[j].size;
            while (ea != eb) { if (ea < eb) { ++i; ea = A[i].off + A[i].size; } else { ++j; eb = B[j].off + B[j].size; } }
            ++i; ++j;
            if (i == A.size() || j == B.size() || pm_eq(A[i], B[j])) { break; }
        }
        ops.push_back({ A[i0].off, A[i - 1].off + A[i - 1].size, (uint32_t) ents.size(), (uint32_t) (j - j0) });
        ents.insert(ents.end(), B.begin() + j0, B.begin() + j);
    }
}

// the host's model of the card's map takes a diff as the card does (pm_apply_delta): each run's residents go, the new entries' residents come
static inline bool pm_heap_delta(pv2_heap & h, const std::vector<pm_delta> & ops, const std::vector<pm_entry> & ents) {
    for (const pm_delta & d : ops) {
        for (auto it = h.res.lower_bound(d.a); it != h.res.end() && it->first < d.b; ) { const size_t x = it->first; ++it; pv2_drop(h, x); }   // runs start and end on shared boundaries: nothing crosses a
        for (uint32_t i = 0; i < d.ne; ++i) { const pm_entry & e = ents[d.e0 + i]; if (e.type != PM_FREE && !pv2_add(h, e.off, pm_res_of(e))) { return false; } }
    }
    return true;
}

// the kernel's report as the host's pv2_alloc_result
static inline pv2_alloc_result pm_report(const pm_result & r) {
    pv2_alloc_result o;
    o.s = r.s == UINT64_MAX ? PV2_NONE : (size_t) r.s;
    for (uint32_t k = 0; k < r.n_mv && k < (uint32_t) PM_MAX_MV; ++k) { o.moves.push_back({ (size_t) r.mv[k].from, (size_t) r.mv[k].to, (size_t) r.mv[k].n, r.mv[k].id }); }
    for (uint32_t k = 0; k < r.n_fr && k < (uint32_t) PM_MAX_FR; ++k) { o.freed.push_back({ (size_t) r.fr[k].s, (size_t) r.fr[k].n, r.fr[k].id }); }
    return o;
}
