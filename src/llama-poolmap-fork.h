#pragma once
// fork (pool v2, user 2026-11-22): the pool map as it lives on the card: one vector of entries that tile the pool in address order, each
// {type: free, moe, kv, wrk; size; heat counter} (+ id, unit, offset). The alloc and defrag kernels read and update it on the card and report what they
// did (pv2_alloc_result); the host applies the report to its mirror (pv2_alloc_apply). With --moe map-check=1 the host reads the vector back after every
// alloc and compares (pm_same). The reference of the kernel is pv2_alloc_do on the same map (pm_to_heap / pm_from_heap).
// Canonical form: no zero sizes, no two free entries next to each other, offsets = running sum of sizes.

#include <cstdint>
#include <cstddef>
#include <vector>
#include "llama-poolv2-fork.h"

enum : uint8_t { PM_FREE = 0, PM_MOE = 1, PM_KV = 2, PM_WRK = 3 };
enum : uint8_t { PM_PIN = 1, PM_MOV = 2 };

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
struct pm_req { uint64_t n, a; int64_t hot; int32_t id; int32_t pin; };   // one alloc of a batch (sizes may differ)
struct pm_result {
    uint64_t s;          // the new area, UINT64_MAX: none
    uint32_t n_mv, n_fr;
    uint32_t overflow;   // more moves or frees than the lists hold: the host reads the whole map back
    uint32_t done;       // set last (the host spins on it, bell style)
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
            if (r.id == PV2_HOLD) { e.type = r.mov ? PM_KV : PM_WRK; e.id = -1; e.flags = (uint8_t) (PM_PIN | (r.mov ? PM_MOV : 0)); }
            else { e.type = PM_MOE; e.id = r.id; e.flags = r.pin ? PM_PIN : 0; }
            ++ri;
        } else { return {}; }   // a hole: not a tiling heap
        at += e.size;
        v.push_back(e);
    }
    return v;
}
static inline pv2_heap pm_to_heap(const std::vector<pm_entry> & v, size_t lo, size_t hi) {
    pv2_heap h = pv2_make(lo, hi);
    for (const pm_entry & e : v) {
        if (e.type == PM_FREE) { continue; }
        pv2_res r; r.n = e.size; r.heat = e.hot; r.u = e.u ? e.u : 1; r.pin = (e.flags & PM_PIN) != 0; r.mov = (e.flags & PM_MOV) != 0;
        r.id = e.type == PM_MOE ? e.id : PV2_HOLD;
        pv2_add(h, e.off, r);
    }
    return h;
}
// two maps agree: same entries in the same order (heat may differ: the card counts uses the host does not see)
static inline bool pm_same(const std::vector<pm_entry> & a, const std::vector<pm_entry> & b, size_t * first_diff = nullptr) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const pm_entry & x = a[i], & y = b[i];
        if (x.off != y.off || x.size != y.size || x.type != y.type || x.id != y.id || (x.flags & PM_PIN) != (y.flags & PM_PIN)) { if (first_diff) { *first_diff = i; } return false; }
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
// the kernel's report as the host's pv2_alloc_result
static inline pv2_alloc_result pm_report(const pm_result & r) {
    pv2_alloc_result o;
    o.s = r.s == UINT64_MAX ? PV2_NONE : (size_t) r.s;
    for (uint32_t k = 0; k < r.n_mv && k < (uint32_t) PM_MAX_MV; ++k) { o.moves.push_back({ (size_t) r.mv[k].from, (size_t) r.mv[k].to, (size_t) r.mv[k].n, r.mv[k].id }); }
    for (uint32_t k = 0; k < r.n_fr && k < (uint32_t) PM_MAX_FR; ++k) { o.freed.push_back({ (size_t) r.fr[k].s, (size_t) r.fr[k].n, r.fr[k].id }); }
    return o;
}
