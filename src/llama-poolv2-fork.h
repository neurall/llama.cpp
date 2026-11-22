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
#include <array>
#include <cmath>
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

// Stage 1 (fork/docs/pool-v2.md: unit views over today's classes): a class of n blocks of B bytes seen through views that step by u (u divides B). Block b's
// table entry is b x B / u; the views reach the last block's slices (the backend checks (ne[2] - 1) x u + a slice's bytes); u = B gives today's views and tables.
static inline int64_t pv2_view_ne2(int64_t n, size_t B, size_t u) { return n > 0 ? (n - 1) * (int64_t) (B / u) + 1 : 0; }
static inline int32_t pv2_block_entry(int32_t b, size_t B, size_t u) { return b * (int32_t) (B / u); }
// stage 2 (the heap): every layer's views cover the whole pool, an expert of n bytes may start at any multiple of u up to span - n; the backend checks that the
// last view's slice, (ne[2] - 1) x u + bytes, ends inside the buffer
static inline int64_t pv2_heap_ne2(size_t span, size_t n, size_t u) { return span >= n ? (int64_t) ((span - n) / u) + 1 : 0; }

struct pv2_res {
    int32_t id   = -1;      // the caller's expert id
    size_t  n    = 0;       // bytes held
    int64_t heat = 0;       // decode use count; -1: a landing leftover any newcomer may replace
    bool    pin  = false;   // in flight (its layer computes or waits to): never replaced
    size_t  u    = 1;       // its unit: the start is a multiple of it (a move keeps that)
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

// Held stretches (stage 2): bytes of the card's buffer the heap lends out for a while (a graph's lend, the KV zone, a retype's staging). Each free piece of one is
// a pinned stand-in of id PV2_HOLD and the hottest heat: no placement crosses it, no refill drops it. The residents in a stretch go before it is held (pv2_overlap).
static constexpr int32_t PV2_HOLD = std::numeric_limits<int32_t>::min();
// the starts of the residents (not held stand-ins) that reach into [a, b)
static inline std::vector<size_t> pv2_overlap(const pv2_heap & h, size_t a, size_t b) {
    std::vector<size_t> out;
    auto it = h.res.lower_bound(a);
    if (it != h.res.begin() && std::prev(it)->first + std::prev(it)->second.n > a) { --it; }
    for (; it != h.res.end() && it->first < b; ++it) { if (it->second.id != PV2_HOLD) { out.push_back(it->first); } }
    return out;
}
// every free byte of [a, b) inside the heap becomes held, whatever resides there
static inline void pv2_hold_free(pv2_heap & h, size_t a, size_t b) {
    a = std::max(a, h.lo); b = std::min(b, h.hi);
    if (a >= b) { return; }
    std::vector<std::pair<size_t, size_t>> pieces;
    auto g = h.gaps.upper_bound(a);
    if (g != h.gaps.begin()) { --g; }
    for (; g != h.gaps.end() && g->first < b; ++g) {
        const size_t x = std::max(a, g->first), y = std::min(b, g->first + g->second);
        if (x < y) { pieces.push_back({ x, y - x }); }
    }
    for (const auto & [x, n] : pieces) { pv2_add(h, x, { PV2_HOLD, n, std::numeric_limits<int64_t>::max(), true }); }
}
// every free byte of [a, b) inside the heap becomes held; false (nothing changed) while a resident reaches into it
static inline bool pv2_hold(pv2_heap & h, size_t a, size_t b) {
    if (!pv2_overlap(h, std::max(a, h.lo), std::min(b, h.hi)).empty()) { return false; }
    pv2_hold_free(h, a, b);
    return true;
}
// every held byte of [a, b) becomes free; a stand-in reaching past the stretch keeps its outside part
static inline void pv2_unhold(pv2_heap & h, size_t a, size_t b) {
    std::vector<std::pair<size_t, size_t>> held;
    auto it = h.res.lower_bound(a);
    if (it != h.res.begin() && std::prev(it)->first + std::prev(it)->second.n > a) { --it; }
    for (; it != h.res.end() && it->first < b; ++it) { if (it->second.id == PV2_HOLD) { held.push_back({ it->first, it->second.n }); } }
    for (const auto & [s, n] : held) {
        pv2_drop(h, s);
        if (s < a)     { pv2_add(h, s, { PV2_HOLD, a - s, std::numeric_limits<int64_t>::max(), true }); }
        if (s + n > b) { pv2_add(h, b, { PV2_HOLD, s + n - b, std::numeric_limits<int64_t>::max(), true }); }
    }
}

// ---- stage 2b: the heap as a shadow of the size classes (no placement of its own: it hears where the classes put each expert and which blocks are held) ----
// a block of a size class as the shadow's check sees it: its start and bytes, the expert it holds (-1 none) and that expert's bytes, held (lent, KV, CUDA's room)
struct pv2_blk { size_t s, B; int32_t id; size_t n; bool held; };
struct pv2_at  { size_t s; int32_t id; size_t n; };   // a resident: start, id, bytes
// what the heap must hold for these blocks: a resident at each block's start that holds an expert, the union of the held blocks' bytes (sorted, disjoint)
static inline void pv2_expect(const std::vector<pv2_blk> & blks, std::vector<pv2_at> & res, std::vector<std::pair<size_t, size_t>> & u) {
    res.clear(); u.clear();
    std::vector<std::pair<size_t, size_t>> hb;
    for (const auto & b : blks) {
        if (b.held) { hb.push_back({ b.s, b.s + b.B }); } else if (b.id >= 0) { res.push_back({ b.s, b.id, b.n }); }
    }
    std::sort(hb.begin(), hb.end());
    for (const auto & r : hb) { if (!u.empty() && r.first <= u.back().second) { u.back().second = std::max(u.back().second, r.second); } else { u.push_back(r); } }
    std::sort(res.begin(), res.end(), [](const pv2_at & a, const pv2_at & b) { return a.s < b.s; });
}
// how far the heap is from `res` and `u` (pv2_expect): each resident it has that is not expected or expected and missing (start, id, bytes), each held range
// not exactly one of u and each range of u not exactly held; 0: the same
static inline size_t pv2_check(const pv2_heap & h, const std::vector<pv2_at> & res, const std::vector<std::pair<size_t, size_t>> & u) {
    size_t bad = 0, i = 0;
    std::vector<std::pair<size_t, size_t>> hu;
    for (const auto & [s, r] : h.res) {
        if (r.id == PV2_HOLD) {
            if (!hu.empty() && hu.back().second == s) { hu.back().second += r.n; } else { hu.push_back({ s, s + r.n }); }
            continue;
        }
        while (i < res.size() && res[i].s < s) { ++bad; ++i; }   // expected, missing
        if (i < res.size() && res[i].s == s && res[i].id == r.id && res[i].n == r.n) { ++i; } else { ++bad; }
    }
    bad += res.size() - i;
    for (const auto & r : hu) { bad += !std::binary_search(u.begin(), u.end(), r); }
    for (const auto & r : u)  { bad += !std::binary_search(hu.begin(), hu.end(), r); }
    return bad;
}
// the shadow hears expert `id` is now at s with n bytes (PV2_NONE: nowhere; where: id -> start): it leaves its old place, takes the new one. false: the new
// place is not free in the heap (the classes put it over an expert or held bytes the shadow still has: a missed move); nothing then changes but its leaving
static inline bool pv2_mirror(pv2_heap & h, std::map<int32_t, size_t> & where, int32_t id, size_t s, size_t n, int64_t heat = 0) {
    auto it = where.find(id);
    if (it != where.end()) {
        auto r = h.res.find(it->second);
        if (r != h.res.end() && r->second.id == id) {
            if (it->second == s && r->second.n == n) { return true; }
            pv2_drop(h, it->second);
        }
        where.erase(it);
    }
    if (s == PV2_NONE) { return true; }
    if (!pv2_add(h, s, { id, n, heat, false })) { return false; }
    where[id] = s;
    return true;
}
// the held stretches made exactly the union u (pv2_expect): what is held outside u comes back free, the free bytes in u are held; returns the residents that reach
// into u (the shadow: the classes lent or zoned bytes an expert of the shadow still has)
static inline size_t pv2_held_set(pv2_heap & h, const std::vector<std::pair<size_t, size_t>> & u) {
    pv2_unhold(h, h.lo, h.hi);
    size_t bad = 0;
    for (const auto & [a, b] : u) {
        bad += pv2_overlap(h, std::max(a, h.lo), std::min(b, h.hi)).size();
        pv2_hold_free(h, a, b);
    }
    return bad;
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

// every id, hottest first (ties: the lower id first)
static inline std::vector<int32_t> pv2_heat_order(const std::vector<int64_t> & heat) {
    std::vector<int32_t> o(heat.size());
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int32_t a, int32_t b) { return heat[a] != heat[b] ? heat[a] > heat[b] : a < b; });
    return o;
}
// c1 boot fill: one heat scale over all layers, a lifetime count's share of its layer's picks x 1e9
static inline int64_t pv2_share_heat(uint64_t count, uint64_t total) { return std::llround(1e9 * (double) count / (double) std::max<uint64_t>(1, total)); }
// c1 boot fill: the picked experts (heat >= 0) hottest first, ties to the lower id: the order pv2_heat_dist packs them down from the heap's top
static inline std::vector<int32_t> pv2_boot_order(const std::vector<int64_t> & heat) {
    std::vector<int32_t> o = pv2_heat_order(heat);
    o.erase(std::find_if(o.begin(), o.end(), [&](int32_t i) { return heat[i] < 0; }), o.end());
    return o;
}
// each expert's distance from the hot end of a heap that held every expert packed in heat order (ties: the lower id first)
static inline std::vector<size_t> pv2_heat_dist(const std::vector<int64_t> & heat, const std::vector<size_t> & n) {
    const std::vector<int32_t> o = pv2_heat_order(heat);
    std::vector<size_t> d(heat.size());
    size_t acc = 0;
    for (int32_t i : o) { d[i] = acc; acc += n[i]; }
    return d;
}
// the target start of n bytes at distance d below the top of [lo, top): the cold end lo when it would not fit
static inline size_t pv2_target(size_t lo, size_t top, size_t d, size_t n) { return top - lo >= d + n ? top - d - n : lo; }

// c2: the target in the main heap [lo, hi) whose landing hole [M, H) holds no expert: packed down from the top as if the hole were not there
static inline size_t pv2_target_main(size_t lo, size_t hi, size_t M, size_t H, size_t d, size_t n) {
    const size_t t = pv2_target(lo, hi - (H - M), d, n);
    return t >= M ? t + (H - M) : t;
}
// c2 decode: the heat of the decode cache's score (xscore) on the heap's scale; a victim must be colder than the newcomer by the pay-back margin
static inline int64_t pv2_decode_heat(double xscore) { return std::llround(1e6 * xscore); }
static inline int64_t pv2_victim_heat(double xscore, double margin) { return pv2_decode_heat(xscore - margin); }
// c2 decode: the targets (pv2_heat_dist, ~2 ms on 24.6k experts) are refreshed at the first step and every 64 steps, never per pick (last: PV2_NONE none yet)
// c2 moves (user 2026-11-21: VRAM copies are cheap, a sorted heap saves re-uploads): a chain for a newcomer of n bytes may copy at most the bytes whose copy
// time (copy_gbs: a device copy reads and writes, about half the VRAM bandwidth) equals one upload of the newcomer over the link (pcie_gbs), and no more than
// fit in the idle window at the safe point (idle_us; <= 0: no window, nothing moves). Feeds pv2_plan_move's budget.
static inline size_t pv2_move_budget(size_t n, double pcie_gbs, double copy_gbs, double idle_us) {
    if (!(pcie_gbs > 0.0) || !(copy_gbs > 0.0) || !(idle_us > 0.0)) { return 0; }
    const double pay = (double) n * copy_gbs / pcie_gbs, win = idle_us * copy_gbs * 1e3;
    return (size_t) std::min(pay, win);
}
static constexpr uint64_t PV2_DIST_EVERY = 64;
static inline bool pv2_dist_due(uint64_t step, uint64_t last) { return last == PV2_NONE || step >= last + PV2_DIST_EVERY; }
// c2: a main-heap window for n bytes near target t: the free window nearest t anywhere in the heap first, else (heat above INT64_MIN) residents colder than
// `heat` within reach (pv2_place)
static inline pv2_pick pv2_pick_main(const pv2_heap & h, size_t t, size_t n, size_t u, int64_t heat, size_t reach) {
    const pv2_pick p = pv2_place(h, t, n, u, std::numeric_limits<int64_t>::min(), h.hi - h.lo, true);
    if (p.s != PV2_NONE || heat == std::numeric_limits<int64_t>::min()) { return p; }
    return pv2_place(h, t, n, u, heat, reach, true);
}

// c2: a move plan (user 2026-11-21: "move, don't evict": only the coldest resident ever leaves; a chain of experts shifts toward the cold end by device copies
// and the newcomer goes last). The steps run in this order, each destination free when its step runs (a step may overlap its own source: the copy runs toward
// the destination); the evicted residents (starts, before any step) leave first.
struct pv2_step { int32_t id; size_t from, to; };
struct pv2_move { bool ok = false; std::vector<pv2_step> steps; std::vector<size_t> evict; size_t s = PV2_NONE, bytes = 0; };
static constexpr size_t PV2_MOVE_EVICT = 8;     // at most this many of the coldest are tried as victims
static constexpr size_t PV2_MOVE_CHAIN = 256;   // residents in the window the chain search covers (shortcut: a longer heap is searched around the target)
// the plan taken on the heap: evictions, steps in order, the newcomer; false when a step's destination is not free (a plan of another heap state)
static inline bool pv2_apply_move(pv2_heap & h, const pv2_move & m, const pv2_res & r) {
    for (size_t s : m.evict) { pv2_drop(h, s); }
    for (const pv2_step & st : m.steps) { const pv2_res v = pv2_drop(h, st.from); if (!pv2_add(h, st.to, v)) { return false; } }
    return pv2_add(h, m.s, r);
}
// 1. a free window anywhere: the nearest to t (ties: the lower start), nothing moves. 2. else for k = 0, 1, .. of the coldest eligible residents (heat below the
// newcomer's, not in flight, not held; ties: the lower id) evicted, the cheapest order-preserving layout of the rest and the newcomer: pinned ones stay, the
// newcomer sits where the fewest heat inversions are, every mover is packed against a stayer or the other movers (an expert's start is a multiple of its unit
// pv2_res::u); cost = bytes moved, then the newcomer's distance to t, then its start. The first k whose cost is within `budget` bytes wins; !ok when none.
static inline pv2_move pv2_plan_move(const pv2_heap & h, size_t t, size_t n, size_t u, int64_t heat, size_t budget = SIZE_MAX) {
    pv2_move mv;
    for (const auto & [a, len] : h.gaps) {
        const size_t s = pv2_fit(a, a + len, n, u, t);
        if (s != PV2_NONE && (mv.s == PV2_NONE || pv2_dist(s, t) < pv2_dist(mv.s, t))) { mv.s = s; mv.ok = true; }
    }
    if (mv.ok) { return mv; }
    std::vector<size_t> vic;
    for (const auto & [s, r] : h.res) { if (!r.pin && r.id != PV2_HOLD && r.heat < heat) { vic.push_back(s); } }
    std::sort(vic.begin(), vic.end(), [&](size_t a, size_t b) {
        const pv2_res & x = h.res.at(a), & y = h.res.at(b);
        return x.heat != y.heat ? x.heat < y.heat : x.id != y.id ? x.id < y.id : a < b; });
    using i64 = int64_t;
    const i64 INF = INT64_MAX / 4;
    struct el { size_t s, n, u; bool pin; i64 heat; int32_t id; };
    typedef std::array<i64, 3> key_t;   // cost, newcomer's distance to t, its start: lexicographic; the last two INF while the newcomer is not placed
    for (size_t k = 0; k <= std::min(vic.size(), PV2_MOVE_EVICT); ++k) {
        std::vector<size_t> gone(vic.begin(), vic.begin() + k);
        std::sort(gone.begin(), gone.end());
        std::vector<el> all;
        for (const auto & [s, r] : h.res) { if (!std::binary_search(gone.begin(), gone.end(), s)) { all.push_back({ s, r.n, r.u, r.pin, r.heat, r.id }); } }
        size_t w0 = 0, w1 = all.size();
        if (all.size() > PV2_MOVE_CHAIN) {
            size_t c = 0;
            while (c < all.size() && all[c].s < t) { ++c; }
            w0 = c > PV2_MOVE_CHAIN / 2 ? c - PV2_MOVE_CHAIN / 2 : 0;
            w1 = std::min(all.size(), w0 + PV2_MOVE_CHAIN);
        }
        const i64 LB = w0 ? (i64) (all[w0 - 1].s + all[w0 - 1].n) : (i64) h.lo, RB = w1 < all.size() ? (i64) all[w1].s : (i64) h.hi;
        const std::vector<el> E(all.begin() + w0, all.begin() + w1);
        const int m = (int) E.size();
        auto au = [](i64 x, size_t un) { return (x + (i64) un - 1) / (i64) un * (i64) un; };
        auto ad = [](i64 x, size_t un) { return x / (i64) un * (i64) un; };
        std::vector<int> allowed(m + 1, 0), pc(m + 1, 0);   // slots with the fewest heat inversions; pinned count
        {
            int mi = INT32_MAX;
            std::vector<int> inv(m + 1, 0);
            for (int j = 0; j <= m; ++j) {
                for (int i = 0; i < m; ++i) { inv[j] += i < j ? E[i].heat > heat : E[i].heat < heat; }
                mi = std::min(mi, inv[j]);
            }
            for (int j = 0; j <= m; ++j) { allowed[j] = inv[j] == mi; }
            for (int i = 0; i < m; ++i) { pc[i + 1] = pc[i] + E[i].pin; }
        }
        // left packing from the stayer a (-1: the left bound): Le[a+1][j] end after j movers, Lp position of mover a + j, Lc cost so far; right packing to the
        // stayer b (m: the right bound): Rs[b][j] start after j movers, the mover b - j at Rs[b][j]
        std::vector<std::vector<i64>> Le(m + 1), Lp(m + 1), Lc(m + 1), Rs(m + 1), Rc(m + 1);
        for (int a = -1; a < m; ++a) {
            auto & e = Le[a + 1]; auto & p = Lp[a + 1]; auto & c = Lc[a + 1];
            e.push_back(a < 0 ? LB : (i64) (E[a].s + E[a].n)); p.push_back(0); c.push_back(0);
            for (int i = a + 1; i < m && !E[i].pin; ++i) {
                const i64 pos = au(e.back(), E[i].u);
                p.push_back(pos); e.push_back(pos + (i64) E[i].n); c.push_back(c.back() + (pos != (i64) E[i].s ? (i64) E[i].n : 0));
            }
        }
        for (int b = 0; b <= m; ++b) {
            auto & r = Rs[b]; auto & c = Rc[b];
            r.push_back(b == m ? RB : (i64) E[b].s); c.push_back(0);
            for (int i = b - 1; i >= 0 && !E[i].pin; --i) {
                const i64 x = r.back() - (i64) E[i].n;
                if (x < 0) { break; }
                const i64 pos = ad(x, E[i].u);
                r.push_back(pos); c.push_back(c.back() + (pos != (i64) E[i].s ? (i64) E[i].n : 0));
            }
        }
        struct par { int a = -2, f = 0, p = 0; bool inc = false; };
        std::vector<std::array<key_t, 2>> dp(m + 2, { key_t { INF, INF, INF }, key_t { INF, INF, INF } });   // index b + 1
        std::vector<std::array<par, 2>> pr(m + 2);
        dp[0][0] = { 0, INF, INF };
        auto better = [](const key_t & x, const key_t & y) { return x < y; };
        for (int a = -1; a < m; ++a) {
            for (int b = a + 1; b <= m; ++b) {
                const int q = b - a - 1;
                if (q > 0 && pc[b] - pc[a + 1] > 0) { break; }
                if (q >= (int) Le[a + 1].size() || q >= (int) Rs[b].size()) { if (q >= (int) Le[a + 1].size()) { break; } continue; }
                for (int f = 0; f < 2; ++f) {
                    const key_t base = dp[a + 1][f];
                    if (base[0] >= INF) { continue; }
                    for (int p = 0; p <= q; ++p) {
                        const i64 le = Le[a + 1][p], rs = Rs[b][q - p], cost = base[0] + Lc[a + 1][p] + Rc[b][q - p];
                        if (le > rs) { continue; }
                        key_t k1 = { cost, base[1], base[2] };
                        if (better(k1, dp[b + 1][f])) { dp[b + 1][f] = k1; pr[b + 1][f] = { a, f, p, false }; }
                        if (f == 0 && allowed[a + 1 + p]) {
                            const size_t fs = pv2_fit((size_t) le, (size_t) rs, n, u, t);
                            if (fs != PV2_NONE) {
                                const key_t k2 = { cost, (i64) pv2_dist(fs, t), (i64) fs };
                                if (better(k2, dp[b + 1][1])) { dp[b + 1][1] = k2; pr[b + 1][1] = { a, 0, p, true }; }
                            }
                        }
                    }
                }
            }
        }
        const key_t best = dp[m + 1][1];
        if (best[0] >= INF || (size_t) best[0] > budget) { continue; }
        std::vector<pv2_step> steps;
        for (int b = m, f = 1; b >= 0 && pr[b + 1][f].a != -2;) {
            const par x = pr[b + 1][f];
            const int q = b - x.a - 1;
            for (int j = 1; j <= x.p; ++j) { const el & e = E[x.a + j]; if ((i64) e.s != Lp[x.a + 1][j]) { steps.push_back({ e.id, e.s, (size_t) Lp[x.a + 1][j] }); } }
            for (int j = 1; j <= q - x.p; ++j) { const el & e = E[b - j]; if ((i64) e.s != Rs[b][j]) { steps.push_back({ e.id, e.s, (size_t) Rs[b][j] }); } }
            f = x.f; b = x.a;
            if (b < 0) { break; }
        }
        // dependency order: a step runs when its destination is free but for its own source
        pv2_heap sim = h;
        for (size_t s : gone) { pv2_drop(sim, s); }
        std::vector<pv2_step> done, todo = steps;
        for (bool prog = true; prog && !todo.empty(); ) {
            prog = false;
            for (size_t i = 0; i < todo.size(); ) {
                const pv2_res v = pv2_drop(sim, todo[i].from);
                if (pv2_add(sim, todo[i].to, v)) { done.push_back(todo[i]); todo.erase(todo.begin() + i); prog = true; } else { pv2_add(sim, todo[i].from, v); ++i; }
            }
        }
        if (!todo.empty() || !pv2_take(sim, (size_t) best[2], n)) { continue; }
        mv.ok = true; mv.steps = done; mv.evict = gone; mv.s = (size_t) best[2]; mv.bytes = (size_t) best[0];
        return mv;
    }
    return mv;
}

// c2 background re-sort (user 2026-11-21): moves that bring residents toward their heat targets (tgt: id -> target start, pv2_target_main), at most `budget`
// bytes in all, run at the decode switch and between tokens (safe points only). Greedy: each step is the single move with the largest distance gained (the gain
// per byte moved), ties to the lower id then the lower start, into a window free but for the mover's own bytes; pinned or held ones and ids without a target
// stay. shortcut: single moves into free bytes only, so a full heap is not re-sorted; swaps through a scratch window when the measured misplacement needs them.
static inline std::vector<pv2_step> pv2_sort_step(const pv2_heap & h0, const std::map<int32_t, size_t> & tgt, size_t budget) {
    std::vector<pv2_step> out;
    pv2_heap h = h0;
    for (;;) {
        pv2_step best { 0, 0, 0 };
        size_t bg = 0;
        std::vector<size_t> starts;
        for (const auto & [s, r] : h.res) { starts.push_back(s); }
        for (size_t s : starts) {
            const pv2_res r = h.res.at(s);
            const auto it = tgt.find(r.id);
            if (r.pin || it == tgt.end() || r.n > budget) { continue; }
            pv2_drop(h, s);
            size_t x = PV2_NONE;
            for (const auto & [a, len] : h.gaps) {
                const size_t y = pv2_fit(a, a + len, r.n, r.u, it->second);
                if (y != PV2_NONE && (x == PV2_NONE || pv2_dist(y, it->second) < pv2_dist(x, it->second))) { x = y; }
            }
            pv2_add(h, s, r);
            if (x == PV2_NONE || pv2_dist(x, it->second) >= pv2_dist(s, it->second)) { continue; }
            const size_t g = pv2_dist(s, it->second) - pv2_dist(x, it->second);
            if (g > bg || (g == bg && r.id < best.id)) { bg = g; best = { r.id, s, x }; }
        }
        if (bg == 0) { return out; }
        const pv2_res v = pv2_drop(h, best.from);
        pv2_add(h, best.to, v);
        budget -= v.n;
        out.push_back(best);
    }
}

// The KV zone / lend / room take [0, Z) of the heap (user 2026-11-21: they must take the coldest, not whatever the heat order put at the low end; a 15.5k prompt
// threw out 348 experts above the 90th percentile of decode heat and 1.7k above the median). Residents overlapping [0, Z), hottest first (not pinned, not a hold,
// heat above min_heat), each moves above Z by pv2_plan_move on the heap cut at Z: a free window nearest Z, else over the colder residents there (the chain shifts,
// only a colder one leaves). Steps run in order: an eviction is a step with to == PV2_NONE, then a chain's steps, then the resident itself (`evicted` lists the ids); bytes copied stay within `budget`.
struct pv2_zone { std::vector<pv2_step> steps; std::vector<int32_t> evicted; size_t bytes = 0; };
static inline pv2_zone pv2_sort_range(const pv2_heap & h, size_t Z, size_t budget, int64_t min_heat) {
    pv2_zone z;
    pv2_heap sub = h;   // the heap as it is (the landing hole and every hold stay taken), cut at Z: nothing below Z is a destination
    for (auto it = sub.res.begin(); it != sub.res.end(); ) { if (it->first < Z) { it = sub.res.erase(it); } else { ++it; } }
    for (auto it = sub.gaps.begin(); it != sub.gaps.end(); ) {
        const size_t g = it->first, e = it->first + it->second;
        if (e <= Z) { it = sub.gaps.erase(it); }
        else if (g < Z) { const size_t len = e - Z; it = sub.gaps.erase(it); sub.gaps[Z] = len; ++it; }
        else { ++it; }
    }
    sub.lo = std::min(Z, h.hi);
    for (const auto & [s, r] : h.res) { if (s < Z && s + r.n > Z) { pv2_add(sub, Z, { PV2_HOLD, s + r.n - Z, 0, true, 1 }); } }   // the resident across Z still covers its bytes above it until it moves
    std::vector<size_t> cand;
    for (const auto & [s, r] : h.res) { if (s < Z && !r.pin && r.id != PV2_HOLD && r.heat > min_heat) { cand.push_back(s); } }
    std::stable_sort(cand.begin(), cand.end(), [&](size_t a, size_t b) { const pv2_res & x = h.res.at(a), & y = h.res.at(b); return x.heat != y.heat ? x.heat > y.heat : a < b; });
    // 1. exact swaps: a hot resident takes the place of the coldest resident above Z of the very same size and unit (the cold one leaves): no fragment, no chain, O(n log n)
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> by_size;   // (n, u) -> starts above Z, colder first
    for (const auto & [s, r] : sub.res) { if (!r.pin && r.id != PV2_HOLD) { by_size[{ r.n, r.u }].push_back(s); } }
    for (auto & [k, v] : by_size) { std::stable_sort(v.begin(), v.end(), [&](size_t a, size_t b) { const pv2_res & x = sub.res.at(a), & y = sub.res.at(b); return x.heat != y.heat ? x.heat < y.heat : a < b; }); }
    std::map<std::pair<size_t, size_t>, size_t> next_cold;
    std::vector<size_t> rest;
    for (size_t s : cand) {
        const pv2_res & r = h.res.at(s);
        if (z.bytes + r.n > budget) { break; }
        size_t fs = PV2_NONE;   // a free window above Z first: nothing leaves
        for (const auto & [ga, gl] : sub.gaps) {
            const size_t c = pv2_fit(ga, ga + gl, r.n, r.u, Z);
            if (c != PV2_NONE) { fs = c; break; }
        }
        if (fs != PV2_NONE) {
            z.steps.push_back({ r.id, s, fs });
            z.bytes += r.n;
            pv2_res moved = r; moved.pin = true;
            pv2_add(sub, fs, moved);
            continue;
        }
        auto it = by_size.find({ r.n, r.u });
        size_t & ix = next_cold[{ r.n, r.u }];
        if (it != by_size.end() && ix < it->second.size() && sub.res.at(it->second[ix]).heat < r.heat) {
            const size_t cs = it->second[ix++];
            const int32_t cid = sub.res.at(cs).id;
            z.evicted.push_back(cid);
            z.steps.push_back({ cid, cs, PV2_NONE });
            z.steps.push_back({ r.id, s, cs });
            z.bytes += r.n;
            pv2_res moved = r; moved.pin = true;
            sub.res[cs] = moved;   // same bytes, new owner: later passes neither evict nor shift it
        } else {
            rest.push_back(s);
        }
    }
    // 2. what no same-size cold resident could take: a free window, else a chain over colder residents (pv2_plan_move), a few only (each costs a search)
    size_t tried = 0;
    for (size_t s : rest) {
        if (tried++ >= 32) { break; }
        const pv2_res & r = h.res.at(s);
        if (z.bytes + r.n > budget) { break; }
        const pv2_move mv = pv2_plan_move(sub, Z, r.n, r.u, r.heat, budget - z.bytes - r.n);
        if (!mv.ok || mv.s == PV2_NONE) { continue; }
        for (size_t es : mv.evict) { z.evicted.push_back(sub.res.at(es).id); z.steps.push_back({ sub.res.at(es).id, es, PV2_NONE }); }
        pv2_apply_move(sub, mv, r);
        sub.res.at(mv.s).pin = true;
        for (const auto & st : mv.steps) { z.steps.push_back(st); }
        z.steps.push_back({ r.id, s, mv.s });
        z.bytes += mv.bytes + r.n;
    }
    return z;
}
struct pv2_cand { int32_t id; size_t n, u; int64_t heat; };
// The switch to decode (user 2026-11-21: "this tail layer sized area is fully refilled and reclaimed by hottest missing moes"): the landing area keeps no cold
// leftovers. Over its residents (decode heat set by the caller) and `miss` (experts no heap holds, hottest first), hottest first, a resident before a newcomer
// of the same heat (it costs no upload), as many as the area's bytes hold stay or come in; the other residents leave (appended to *left). A resident in flight
// or a held stretch stays whatever its heat, its bytes taken from the area first. A newcomer takes the free window nearest its place in that order (hottest at
// lo, next to the main heap's hot end). Returns the newcomers placed: the refill upload.
static inline std::vector<pv2_cand> pv2_refill(pv2_heap & h, const std::vector<pv2_cand> & miss, std::vector<pv2_res> * left = nullptr) {
    std::vector<std::pair<int64_t, size_t>> rs;   // (heat, start), hottest first: the residents that may leave
    size_t used = 0;
    for (const auto & [s, r] : h.res) { if (r.pin) { used += r.n; } else { rs.push_back({ r.heat, s }); } }
    std::sort(rs.begin(), rs.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    std::vector<size_t> keep;
    std::vector<std::pair<size_t, size_t>> in;     // (miss index, its target)
    size_t i = 0, j = 0;
    while (i < rs.size() || j < miss.size()) {
        const bool res = j == miss.size() || (i < rs.size() && rs[i].first >= miss[j].heat);
        const size_t nb = res ? h.res.at(rs[i].second).n : miss[j].n;
        if (used + nb > h.hi - h.lo) { break; }
        if (res) { keep.push_back(rs[i++].second); } else { in.push_back({ j++, h.lo + used }); }
        used += nb;
    }
    std::sort(keep.begin(), keep.end());
    for (size_t k = 0; k < rs.size(); ++k) {
        if (!std::binary_search(keep.begin(), keep.end(), rs[k].second)) { const pv2_res r = pv2_drop(h, rs[k].second); if (left) { left->push_back(r); } }
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

// c2 boot fill (profile_preload on the heap): the experts hottest first (d: the bytes of all hotter ones, pv2_heat_dist), each takes the free window nearest its
// target in the main heap [lo, hi) with the landing hole [M, H); one that finds no window is left out, no resident is replaced (the heap starts empty). The
// placed ones are residents in flight (their upload follows), returned in placement order.
struct pv2_boot_t { int32_t id; size_t n, u, d; int64_t heat; };
struct pv2_put_t  { int32_t id; size_t s; };
static inline std::vector<pv2_put_t> pv2_boot_fill(pv2_heap & h, const std::vector<pv2_boot_t> & ex, size_t M, size_t H) {
    std::vector<pv2_put_t> put;
    for (const pv2_boot_t & x : ex) {
        const size_t t = pv2_target_main(h.lo, h.hi, M, H, x.d, x.n);
        const pv2_pick p = pv2_pick_main(h, t, x.n, x.u, std::numeric_limits<int64_t>::min(), 0);
        if (p.s != PV2_NONE && pv2_add(h, p.s, { x.id, x.n, x.heat, true })) { put.push_back({ x.id, p.s }); }
    }
    return put;
}

// c2/c4: what a stretch needs before it is held (a graph's lend, the KV zone, a retype's staging). Each resident reaching into it is in one state of its upload
// (the caller's state function): a queued job is dropped, a finished one not yet published is discarded, a published expert is evicted, a running one is awaited
// (one copy, never cancelled); the stretch is ready when nothing is awaited. Ids in start order; held stand-ins are no residents of a plan.
enum pv2_job_t : int { PV2_CACHED = 0, PV2_QUEUED, PV2_DONE, PV2_RUNNING };
struct pv2_range_t {
    std::vector<int32_t> drop, discard, evict, wait;
    bool ready() const { return wait.empty(); }
};
template <class F> static inline pv2_range_t pv2_range_plan(const pv2_heap & h, size_t a, size_t b, F state) {
    pv2_range_t r;
    if (a >= b) { return r; }
    for (size_t s : pv2_overlap(h, a, b)) {
        const int32_t id = h.res.at(s).id;
        switch (state(id)) {
            case PV2_QUEUED:  r.drop.push_back(id);    break;
            case PV2_DONE:    r.discard.push_back(id); break;
            case PV2_CACHED:  r.evict.push_back(id);   break;
            case PV2_RUNNING: r.wait.push_back(id);    break;
        }
    }
    return r;
}

// Coalesced upload (user design 2026-11-22): the experts a layer streams are packed back to back into chunks of at most `chunk` bytes (the landing area is a
// double buffer of two chunks), each chunk goes up as ONE copy, then a scatter moves every expert from its chunk offset to its heat-placed heap start.
struct pv2_up_t    { int32_t id; size_t bytes, dst; };            // an expert to stream: its bytes and its final heap byte offset
struct pv2_upmove  { int32_t id; size_t src, dst, bytes; };       // src = byte offset inside the chunk
struct pv2_upchunk { size_t bytes = 0; std::vector<pv2_upmove> mv; };
static inline std::vector<pv2_upchunk> pv2_upchunks(const std::vector<pv2_up_t> & ex, size_t chunk) {
    std::vector<pv2_upchunk> r;
    for (const auto & x : ex) {
        if (x.bytes == 0 || x.bytes > chunk) { continue; }
        if (r.empty() || r.back().bytes + x.bytes > chunk) { r.emplace_back(); }
        r.back().mv.push_back({ x.id, r.back().bytes, x.dst, x.bytes });
        r.back().bytes += x.bytes;
    }
    return r;
}

// The 32 MiB tail reserve as a ring of `slots` slots (user 2026-11-22: "on transfer end the tail is empty and ready for the next transfer, the link saturated
// nonstop"): chunk c goes up into slot c % slots; its scatter (tail -> heat-placed heap starts) runs on a second stream as soon as its upload lands; the upload of
// chunk c may start once the scatter of chunk c - slots is done (wait_scatter), so the link only idles when every slot is still being scattered.
struct pv2_ring_op { size_t chunk, slot; int64_t wait_scatter; };   // wait_scatter: chunk whose scatter must be done first, -1 none
static inline std::vector<pv2_ring_op> pv2_upring(size_t n_chunks, size_t slots) {
    std::vector<pv2_ring_op> r;
    if (slots == 0) { return r; }
    for (size_t c = 0; c < n_chunks; ++c) { r.push_back({ c, c % slots, c >= slots ? (int64_t) (c - slots) : -1 }); }
    return r;
}
