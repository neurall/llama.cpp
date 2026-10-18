// KV zone placement in the expert pool, as templates over the pool type (anything with segs{base,n,B}, tail_end, kv_fixed, kv_h, kv_row, kvt, span) so that
// fork/tests/kvzone_fuzz.cpp can drive them with random pool shapes and no GPU. llama-moecache-fork.cpp includes this and wraps it for moe_cache::alias_pool.
#pragma once
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

template <class P> static size_t kvz_bmax(const P & ap) {
    size_t b = 0;
    for (const auto & sg : ap.segs) { b = std::max<size_t>(b, sg.B); }
    return b;
}

// The gaps between the class zero blocks: the only places a KV zone may lie (a class's zero block is read as zeros by every uncached expert and zeroed again at
// each release, so a zone over it is overwritten or overwrites). One walk for every question about where the zone fits: they used to carry a copy of this loop
// each, and a rule fixed in one (MiMo 12k, OLMoE 185 MiB pool, 2026-10-18) was missing in the next. [lo_min, tail_end) cut at each zero block, every start
// rounded up to `round` (ap.segs lie in ascending order of address).
template <class P> static std::vector<std::pair<size_t, size_t>> kvz_gaps(const P & ap, size_t lo_min, size_t round) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t a = lo_min;
    auto gap = [&](size_t g0, size_t g1) {
        g0 = (g0 + round - 1) / round * round;
        if (g1 > g0) { out.push_back({ g0, g1 }); }
    };
    for (const auto & sg : ap.segs) {
        const size_t z0 = sg.base + (size_t) sg.n * sg.B, z1 = z0 + sg.B;
        if (z0 > a) { gap(a, z0); }
        a = std::max(a, z1);
    }
    gap(a, ap.tail_end);
    return out;
}

// what the compute alias could lend if the KV zone moved behind the graph: the end of the gap that holds the zone, less the zone (fixed part and max(cells laid
// out, min_cells) rows), rounded down to whole largest blocks. Not "the end of the pool less the zone" (MiMo 12k, 2026-10-18: the fixed part fitted no gap behind it).
template <class P> static size_t kvz_lend_potential(const P & ap, size_t min_cells) {
    if (ap.kvt.empty()) { return ap.span; }
    const size_t bmax = std::max<size_t>(1, kvz_bmax(ap)), zone = ap.kv_fixed + std::max<size_t>(ap.kv_h, min_cells) * ap.kv_row;
    size_t best = 0;
    for (const auto & g : kvz_gaps(ap, 0, bmax)) {   // gap starts rounded up to whole blocks, as the placement rounds them: a gap that holds the zone only unaligned promised a lend the layout could not give (fuzz, 2026-10-18)
        if (g.second >= g.first + zone) { best = std::max(best, (g.second - zone) / bmax * bmax); }
    }
    return best;
}

// the most growing cells the largest gap behind lo_min can hold (after the fixed part)
template <class P> static size_t kvz_gap_fit(const P & ap, size_t bmax, size_t lo_min, size_t row) {
    size_t best = 0;
    for (const auto & g : kvz_gaps(ap, lo_min, bmax)) {
        if (row > 0 && g.second > g.first + ap.kv_fixed) { best = std::max(best, (g.second - g.first - ap.kv_fixed) / row); }
    }
    return best;
}

// start of a zone of `size` bytes in the gap nearest to `want` (block-aligned, behind lo_min), or (size_t) -1 when no gap holds it
template <class P> static size_t kvz_gap_place(const P & ap, size_t bmax, size_t lo_min, size_t want, size_t size) {
    size_t best = (size_t) -1, bestd = (size_t) -1;
    for (const auto & g : kvz_gaps(ap, lo_min, bmax)) {
        if (g.second < g.first + size) { continue; }
        const size_t s = std::min(std::max(want, g.first), (g.second - size) / bmax * bmax);
        if (s < g.first) { continue; }
        const size_t d = s > want ? s - want : want - s;
        if (d < bestd) { bestd = d; best = s; }
    }
    return best;
}

// the layout's invariants: nullptr when they hold, else what is broken. The zone lies in one gap (so over no zero block), behind the lend, inside the pool.
template <class P> static const char * kvz_bad(const P & ap, size_t lend_end) {
    if (ap.kv_h == 0) { return nullptr; }
    const size_t lo = ap.kv_start, hi = lo + ap.kv_fixed + ap.kv_h * ap.kv_row;
    if (hi > ap.tail_end)   { return "zone ends past the pool"; }
    if (lo < lend_end)      { return "zone starts inside the lend"; }
    for (const auto & g : kvz_gaps(ap, 0, 1)) { if (lo >= g.first && hi <= g.second) { return nullptr; } }
    return "zone is not inside one gap between the class zero blocks";
}
