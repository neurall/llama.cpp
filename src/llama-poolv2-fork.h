// Pool v2 (user 2026-11-14, 2026-11-21): a card's experts in one byte heap, no size classes; every layer's expert table holds the expert's start in units.
//   allocation  an expert takes its up+gate+down slices at today's block offsets plus the bytes a matmul reads past down's last row, at a start that is a
//               multiple of its layer's unit: pv2_shape_of
//   heat order  coldest at the heap's low end (the KV edge), hottest at its top (next to the landing area); an expert's target start is where it would sit if
//               every hotter expert were packed above it: pv2_heat_dist, pv2_target
//   placement   within reach of the target: the nearest free window, else the coldest colder resident not in flight whose hole holds it (it leaves),
//               else the nearest window over several such residents, else none and the expert lands: pv2_place, pv2_commit. A hotter (or equally hot)
//               resident is never evicted.
//   landing     during a prompt a second heap above the main one takes the streamed experts with no place, reused layer after layer (its residents are
//               leftovers, heat -1, once their layer is computed); at the switch to decode it is refilled with the hottest experts no heap holds: pv2_refill
// tests/test-poolv2-fork.cpp holds the rules and replays them on recorded decode heat (--sim).
#pragma once
#include "llama-poolalloc-fork.h"   // poolx_block_class: the slice offsets and the read past a row, one rule for both pools

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

static constexpr size_t PV2_NONE = SIZE_MAX;

// One expert of a layer in the heap. The graph gives up, gate and down the same ids (one table per layer), so the three views share one nb[2], the unit u:
// a multiple of each slice type's block bytes (the kernels step a view by nb[2] / block bytes) and of 16 (vector loads; test-backend-ops MUL_MAT_ID unit
// cases pass on CUDA and Vulkan at units of whole blocks). The slices keep today's offsets in a block (each at a multiple of 128 from the expert's start);
// n ends behind down's slice and the largest read past a row (poolx_block_class), without today's rounding of a block to lcm(128, block bytes).
struct pv2_shape { size_t n = 0, u = 0, off[3] = { 0, 0, 0 }; };
static inline pv2_shape pv2_shape_of(const poolx_slice s[3]) {
    const poolx_block b = poolx_block_class(s);
    pv2_shape sh;
    sh.u = 16;
    for (int k = 0; k < 3; ++k) { sh.u = std::lcm(sh.u, s[k].ts); sh.off[k] = b.off[k]; }
    sh.n = b.off[2] + poolx_up(b.len[2], 128) + std::max({ b.past[0], b.past[1], b.past[2] });
    return sh;
}

struct pv2_res {
    int32_t id   = -1;      // the caller's expert id
    size_t  n    = 0;       // bytes held
    int64_t heat = 0;       // decode use count; -1: a landing leftover any newcomer may replace
    bool    pin  = false;   // in flight (its layer computes or waits to): never replaced
};
struct pv2_heap {
    size_t lo = 0, hi = 0;
    std::map<size_t, pv2_res> res;    // start -> resident
    std::map<size_t, size_t>  gaps;   // start -> bytes of a free extent; two extents never touch
};
static inline pv2_heap pv2_make(size_t lo, size_t hi) { pv2_heap h; h.lo = lo; h.hi = hi; if (hi > lo) { h.gaps[lo] = hi - lo; } return h; }

// [s, s+n) leaves the free map; false when it is not inside one free extent
static inline bool pv2_take(pv2_heap & h, size_t s, size_t n) {
    auto it = h.gaps.upper_bound(s);
    if (it == h.gaps.begin()) { return false; }
    --it;
    const size_t a = it->first, b = a + it->second;
    if (s + n > b) { return false; }
    h.gaps.erase(it);
    if (s > a)     { h.gaps[a] = s - a; }
    if (s + n < b) { h.gaps[s + n] = b - s - n; }
    return true;
}
// [s, s+n) back to the free map, merged with the extents it touches
static inline void pv2_give(pv2_heap & h, size_t s, size_t n) {
    auto nx = h.gaps.lower_bound(s);
    if (nx != h.gaps.end() && nx->first == s + n) { n += nx->second; nx = h.gaps.erase(nx); }
    if (nx != h.gaps.begin()) {
        auto pv = std::prev(nx);
        if (pv->first + pv->second == s) { pv->second += n; return; }
    }
    h.gaps.emplace_hint(nx, s, n);
}
static inline bool pv2_add(pv2_heap & h, size_t s, const pv2_res & r) { if (!pv2_take(h, s, r.n)) { return false; } h.res[s] = r; return true; }
static inline pv2_res pv2_drop(pv2_heap & h, size_t s) {
    auto it = h.res.find(s);
    const pv2_res r = it->second;
    h.res.erase(it);
    pv2_give(h, s, r.n);
    return r;
}

// the start that is a multiple of u nearest t for n bytes inside [a, b), PV2_NONE when none fits
static inline size_t pv2_fit(size_t a, size_t b, size_t n, size_t u, size_t t) {
    const size_t s0 = (a + u - 1) / u * u;
    if (b < n || s0 > b - n) { return PV2_NONE; }
    const size_t s1 = (b - n) / u * u;
    if (t <= s0) { return s0; }
    if (t >= s1) { return s1; }
    const size_t d = t / u * u;
    return t - d <= d + u - t ? d : d + u;
}
static inline size_t pv2_dist(size_t a, size_t b) { return a > b ? a - b : b - a; }

struct pv2_pick { size_t s = PV2_NONE; std::vector<size_t> evict; };   // s PV2_NONE: no window within reach; evict: starts of the residents it replaces

// where n bytes (start a multiple of u) of a newcomer of `heat` whose heat rank puts it at t go, within `reach` bytes of t (finite: the walks stop at it);
// a victim is a resident colder than the newcomer and not in flight:
//   1. a free window: the nearest one at an edge of its free extent, else the nearest one anywhere in it
//   2. (one) else one victim replaced: the coldest whose hole (it and the free bytes beside it) holds the newcomer, at the hole's edge nearer t; ties: the
//      fewest bytes left over, then the nearer
//   3. else a window over several victims (a newcomer bigger than any one hole): the nearest at an edge of its run between blockers or of a victim in it,
//      else the nearest anywhere in the run
// Measured on 50 prompts of changing heat (Qwen3.8 IQ3_S, one 3090, prompts 11-50): windows at the target itself split two residents and strand both
// remainders (1.4 GB of the 18.6 GB heap fit no expert), edges 1.2 GB; the coldest victim instead of the nearest: decode coverage 94.8 against 93.9%
// (ideal 95.5%), 0.6 GB stranded. Anywhere-windows keep "it lands only when nothing fits" true. one = false for the landing ring: the window nearest its
// cursor keeps a layer's experts in one run (its leftovers all hold heat -1, so the coldest would scatter them).
static inline pv2_pick pv2_place(const pv2_heap & h, size_t t, size_t n, size_t u, int64_t heat, size_t reach, bool one) {
    pv2_pick p;
    size_t ba = PV2_NONE, da = PV2_NONE, bb = PV2_NONE, db = PV2_NONE;   // the best edge window, the best window anywhere
    auto lim = [&]() { return std::min(reach, da); };
    auto offer = [&](size_t A, size_t B, size_t s, bool edge) {
        if (s == PV2_NONE || s < A || s > B || B - s < n || pv2_dist(s, t) > reach) { return; }
        size_t & d = edge ? da : db;
        if (pv2_dist(s, t) < d) { d = pv2_dist(s, t); (edge ? ba : bb) = s; }
    };
    auto edges = [&](size_t A, size_t B, size_t x) {   // the windows that start at x or end at x
        offer(A, B, (x + u - 1) / u * u, true);
        if (x >= n) { offer(A, B, (x - n) / u * u, true); }
    };
    auto run = [&](size_t A, size_t B) {   // false: too short for the newcomer
        if (B - A < n) { return false; }
        edges(A, B, A); edges(A, B, B); offer(A, B, pv2_fit(A, B, n, u, t), false);
        return true;
    };
    // free extents: from t to the right, then to the left, each side until the next extent is farther than the best
    const auto g0 = h.gaps.lower_bound(t);
    for (auto it = g0; it != h.gaps.end() && it->first - t <= lim(); ++it) { run(it->first, it->first + it->second); }
    for (auto it = g0; it != h.gaps.begin(); ) {
        --it;
        const size_t e = it->first + it->second;
        if (e < t && t - e > lim()) { break; }
        run(it->first, e);
    }
    if (ba != PV2_NONE || bb != PV2_NONE) { p.s = ba != PV2_NONE ? ba : bb; return p; }
    auto blocks = [&](const pv2_res & r) { return r.pin || r.heat >= heat; };
    if (one) {
        int64_t vh = std::numeric_limits<int64_t>::max();
        size_t  vl = PV2_NONE, vd = PV2_NONE, vs = PV2_NONE;
        auto it = h.res.lower_bound(t > reach + n ? t - reach - n : 0);
        if (it != h.res.begin()) { --it; }
        for (; it != h.res.end() && it->first <= t + reach + n; ++it) {
            if (blocks(it->second) || it->second.heat > vh) { continue; }
            const size_t A = it == h.res.begin() ? h.lo : std::prev(it)->first + std::prev(it)->second.n;
            const size_t B = std::next(it) == h.res.end() ? h.hi : std::next(it)->first;
            if (B - A < n) { continue; }
            for (size_t s : { (A + u - 1) / u * u, (B - n) / u * u }) {   // inside the hole every window covers the victim: a free one would have won above
                if (s < A || s > B || B - s < n || pv2_dist(s, t) > reach) { continue; }
                const int64_t h1 = it->second.heat;
                const size_t  l1 = B - A - n, d1 = pv2_dist(s, t);
                if (h1 < vh || (h1 == vh && (l1 < vl || (l1 == vl && d1 < vd)))) { vh = h1; vl = l1; vd = d1; vs = s; p.evict = { it->first }; }
            }
        }
        if (vs != PV2_NONE) { p.s = vs; return p; }
        p.evict.clear();
    }
    // runs between blockers (in flight, or not colder): the run reaching t or starting right of it first, then the runs left of it
    std::vector<size_t> anc;   // start and end of each replaceable resident in the run being walked
    auto close = [&](size_t A, size_t B) { if (run(A, B)) { for (size_t x : anc) { edges(A, B, x); } } anc.clear(); };
    size_t a0 = h.lo;
    auto lb = h.res.end();   // the blocker the first run starts behind
    for (auto it = h.res.upper_bound(t); it != h.res.begin(); ) {
        --it;
        if (blocks(it->second)) { a0 = it->first + it->second.n; lb = it; break; }
        if (t - it->first > reach) { a0 = it->first; break; }   // windows starting left of it are out of reach
    }
    size_t a = a0;
    for (auto it = h.res.lower_bound(a0); ; ++it) {
        if (a > t && a - t > lim()) { break; }
        if (it == h.res.end())         { close(a, h.hi); break; }
        if (it->first > t + lim() + n + u) { close(a, it->first); break; }   // windows over it or at its edges start out of reach
        if (!blocks(it->second)) { anc.push_back(it->first); anc.push_back(it->first + it->second.n); continue; }
        close(a, it->first);
        a = it->first + it->second.n;
    }
    anc.clear();
    if (lb != h.res.end()) {
        size_t b = lb->first;
        for (auto it = lb; ; ) {
            if (b < t && t - b > lim()) { break; }
            if (it == h.res.begin()) { close(h.lo, b); break; }
            --it;
            if (!blocks(it->second)) {
                anc.push_back(it->first); anc.push_back(it->first + it->second.n);
                if (t - it->first > lim()) { close(it->first, b); break; }   // windows starting left of it are out of reach
                continue;
            }
            close(it->first + it->second.n, b);
            b = it->first;
        }
    }
    p.s = ba != PV2_NONE ? ba : bb;
    if (p.s == PV2_NONE) { return p; }
    auto it = h.res.lower_bound(p.s);
    if (it != h.res.begin() && std::prev(it)->first + std::prev(it)->second.n > p.s) { --it; }
    for (; it != h.res.end() && it->first < p.s + n; ++it) { p.evict.push_back(it->first); }
    return p;
}

// the pick taken: the residents in its window leave (appended to *out), the newcomer moves in; false only for a pick of another heap state
static inline bool pv2_commit(pv2_heap & h, const pv2_pick & p, const pv2_res & r, std::vector<pv2_res> * out) {
    for (size_t s : p.evict) { const pv2_res v = pv2_drop(h, s); if (out) { out->push_back(v); } }
    return pv2_add(h, p.s, r);
}

// each expert's distance from the hot end of a heap that held every expert packed in heat order (ties: the lower id first)
static inline std::vector<size_t> pv2_heat_dist(const std::vector<int64_t> & heat, const std::vector<size_t> & n) {
    std::vector<int32_t> o(heat.size());
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int32_t a, int32_t b) { return heat[a] != heat[b] ? heat[a] > heat[b] : a < b; });
    std::vector<size_t> d(heat.size());
    size_t acc = 0;
    for (int32_t i : o) { d[i] = acc; acc += n[i]; }
    return d;
}
// the target start of n bytes at distance d below the top of [lo, top): the cold end lo when it would not fit
static inline size_t pv2_target(size_t lo, size_t top, size_t d, size_t n) { return top - lo >= d + n ? top - d - n : lo; }

struct pv2_cand { int32_t id; size_t n, u; int64_t heat; };
// The switch to decode (user 2026-11-21: "this tail layer sized area is fully refilled and reclaimed by hottest missing moes"): the landing area keeps no cold
// leftovers. Over its residents (decode heat set by the caller) and `miss` (experts no heap holds, hottest first), hottest first, a resident before a newcomer
// of the same heat (it costs no upload), as many as the area's bytes hold stay or come in; the other residents leave. A newcomer takes the free window nearest
// its place in that order (hottest at lo, next to the main heap's hot end). Returns the newcomers placed: the refill upload.
static inline std::vector<pv2_cand> pv2_refill(pv2_heap & h, const std::vector<pv2_cand> & miss) {
    std::vector<std::pair<int64_t, size_t>> rs;   // (heat, start), hottest first
    for (const auto & [s, r] : h.res) { rs.push_back({ r.heat, s }); }
    std::sort(rs.begin(), rs.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    std::vector<size_t> keep;
    std::vector<std::pair<size_t, size_t>> in;     // (miss index, its target)
    size_t used = 0, i = 0, j = 0;
    while (i < rs.size() || j < miss.size()) {
        const bool res = j == miss.size() || (i < rs.size() && rs[i].first >= miss[j].heat);
        const size_t nb = res ? h.res.at(rs[i].second).n : miss[j].n;
        if (used + nb > h.hi - h.lo) { break; }
        if (res) { keep.push_back(rs[i++].second); } else { in.push_back({ j++, h.lo + used }); }
        used += nb;
    }
    std::sort(keep.begin(), keep.end());
    for (size_t k = 0; k < rs.size(); ++k) {
        if (!std::binary_search(keep.begin(), keep.end(), rs[k].second)) { pv2_drop(h, rs[k].second); }
    }
    std::vector<pv2_cand> up;
    for (const auto & [jj, t] : in) {
        const pv2_cand & c = miss[jj];
        // the lowest heat: every resident blocks, so only free windows count (no newcomer replaces a kept one)
        const pv2_pick p = pv2_place(h, t, c.n, c.u, std::numeric_limits<int64_t>::min(), h.hi - h.lo, false);
        if (p.s != PV2_NONE && pv2_commit(h, p, { c.id, c.n, c.heat, false }, nullptr)) { up.push_back(c); }
    }
    return up;
}
