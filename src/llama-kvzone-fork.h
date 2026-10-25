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

// The block edges of the pool: every class is a run of blocks of its own size, the classes back to back. A lend ends on the edge at or after its last byte and
// the KV zone starts on one (the blocks under each are whole); these two give the edge at or after x and at or before x (x itself past the blocks).
template <class P> static size_t kvz_block_ceil(const P & ap, size_t x) {
    for (const auto & sg : ap.segs) {
        const size_t end = sg.base + (size_t) sg.n * sg.B;
        if (sg.B > 0 && x > sg.base && x < end) { return sg.base + (x - sg.base + sg.B - 1) / sg.B * sg.B; }
    }
    return x;
}
template <class P> static size_t kvz_block_floor(const P & ap, size_t x) {
    for (const auto & sg : ap.segs) {
        const size_t end = sg.base + (size_t) sg.n * sg.B;
        if (sg.B > 0 && x > sg.base && x < end) { return sg.base + (x - sg.base) / sg.B * sg.B; }
    }
    return x;
}

// Where a KV zone may lie: [lo_min, tail_end), its start rounded up to `round`, over blocks of any size class (the classes lie back to back, nothing between them:
// the device tables mark uncached experts -1 and the kernels skip them, so no class keeps a zero block). Until 2026-10-24 each class ended in a zero block the zone
// had to avoid, which cut the pool into one gap per class: MiMo-V2.6-Flash's 3 GiB sliding-window part fitted none of its 9 classes on the x4 card behind an 11 GiB
// lend, with 9.5 GiB of the pool behind it. One walk for every question about where the zone fits (a rule fixed in one copy was once missing in the next, MiMo
// 12k, OLMoE 185 MiB pool, 2026-10-18).
template <class P> static std::vector<std::pair<size_t, size_t>> kvz_gaps(const P & ap, size_t lo_min, size_t round) {   // round > 1: the start on a block edge
    std::vector<std::pair<size_t, size_t>> out;
    const size_t g0 = round > 1 ? kvz_block_ceil(ap, lo_min) : lo_min;
    if (ap.tail_end > g0) { out.push_back({ g0, ap.tail_end }); }
    return out;
}

// what the compute alias could lend if the KV zone moved behind the graph: the end of the pool less the zone (fixed part and max(cells laid out, min_cells) rows),
// down to the block edge the zone then starts on: the largest lend end (a lend ends on a block edge, poolx_lend_round).
template <class P> static size_t kvz_lend_potential(const P & ap, size_t min_cells) {
    if (ap.kvt.empty()) { return ap.span; }
    const size_t bmax = std::max<size_t>(1, kvz_bmax(ap)), zone = ap.kv_fixed + std::max<size_t>(ap.kv_h, min_cells) * ap.kv_row;
    size_t best = 0;
    for (const auto & g : kvz_gaps(ap, 0, bmax)) {
        if (g.second >= g.first + zone) { best = std::max(best, kvz_block_floor(ap, g.second - zone)); }
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
        const size_t s = std::min(std::max(kvz_block_ceil(ap, want), g.first), kvz_block_floor(ap, g.second - size));
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
    return "zone is not inside the expert blocks";
}

// The largest prompt pass (tokens) whose graph still leaves the zone for the whole prompt: the lend is mirror_b (the dense split's copies) + a + b * tokens (graph and
// scratch), and it ends on the block edge at or after that (poolx_lend_round); the zone is the prompt's rows rounded up to the attention's pad. The potential is a block
// edge, so a lend ends at or before it exactly when its bytes do: no extra blocks. 64 is the least pass. See pp_cap_to_pool.
template <class P> static unsigned kvz_pass_cap(const P & ap, size_t n_prompt, unsigned ub, double a, double b, size_t pad, size_t mirror_b) {
    if (b <= 0.0) { return ub; }
    const size_t rows = (std::max<size_t>(n_prompt, ap.kv_h) + pad - 1) / pad * pad;
    const double room = (double) kvz_lend_potential(ap, rows) - (double) mirror_b;
    if (room <= a) { return std::min<unsigned>(ub, 64); }
    const double u = (room - a) / b;
    const unsigned cap = std::max<unsigned>(64, (unsigned) std::min(u, 4294967000.0) / 32 * 32);
    return std::min(ub, cap);
}
