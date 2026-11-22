#pragma once
// size classes picked per model (user 2026-11-22): an expert may sit in a bigger class's slot (waste = slot - expert bytes), so a model's sizes fold
// into at most k classes; the pick wastes the fewest bytes, weighted by how many experts of each size the model has (all layers' experts count alike)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct cp_size { uint64_t bytes; uint64_t n; };   // one expert size of the model and how many experts have it
struct cp_pick {
    std::vector<uint64_t> classes;   // slot sizes, ascending; the largest size is always one
    std::vector<size_t>   of;        // per input size (input order): index into classes
    uint64_t waste = 0, total = 0;   // padding bytes over all experts, all experts' own bytes
    double frac() const { return total ? (double) waste / (double) total : 0.0; }
};

// k >= 1; sizes may repeat or come unsorted
static inline cp_pick cp_pick_classes(const std::vector<cp_size> & in, size_t k) {
    std::vector<cp_size> s;   // distinct sizes, ascending, counts merged
    for (const cp_size & x : in) {
        auto it = std::find_if(s.begin(), s.end(), [&](const cp_size & y) { return y.bytes == x.bytes; });
        if (it == s.end()) { s.push_back(x); } else { it->n += x.n; }
    }
    std::sort(s.begin(), s.end(), [](const cp_size & a, const cp_size & b) { return a.bytes < b.bytes; });
    cp_pick r;
    if (s.empty() || k == 0) { return r; }
    const size_t n = s.size();
    k = std::min(k, n);
    // cost(i, j): sizes i..j in one class of size s[j]
    auto cost = [&](size_t i, size_t j) { uint64_t c = 0; for (size_t t = i; t <= j; ++t) { c += (s[j].bytes - s[t].bytes) * s[t].n; } return c; };
    const uint64_t INF = UINT64_MAX;
    // best[c][j]: sizes 0..j in c+1 classes, the last ending at j
    std::vector<std::vector<uint64_t>> best(k, std::vector<uint64_t>(n, INF));
    std::vector<std::vector<size_t>>   from(k, std::vector<size_t>(n, 0));
    for (size_t j = 0; j < n; ++j) { best[0][j] = cost(0, j); }
    for (size_t c = 1; c < k; ++c) {
        for (size_t j = c; j < n; ++j) {
            for (size_t i = c; i <= j; ++i) {   // the last class holds i..j
                if (best[c - 1][i - 1] == INF) { continue; }
                const uint64_t v = best[c - 1][i - 1] + cost(i, j);
                if (v < best[c][j]) { best[c][j] = v; from[c][j] = i; }
            }
        }
    }
    size_t c = 0;   // fewest classes reaching the least waste
    for (size_t t = 1; t < k; ++t) { if (best[t][n - 1] < best[c][n - 1]) { c = t; } }
    r.waste = best[c][n - 1];
    std::vector<size_t> ends;
    for (size_t j = n - 1;;) {
        ends.push_back(j);
        if (c == 0) { break; }
        const size_t i = from[c][j];
        j = i - 1; --c;
    }
    std::reverse(ends.begin(), ends.end());
    for (size_t e : ends) { r.classes.push_back(s[e].bytes); }
    for (const cp_size & x : s) { r.total += x.bytes * x.n; }
    for (const cp_size & x : in) {
        r.of.push_back((size_t) (std::lower_bound(r.classes.begin(), r.classes.end(), x.bytes) - r.classes.begin()));
    }
    return r;
}

// one block size for the whole pool (user 2026-11-22): the most numerous expert size is the block; a smaller expert fits in one block, a bigger one
// spans ceil(bytes / block) consecutive blocks. cp_block_waste: the padding bytes of block size b over all experts
static inline uint64_t cp_blocks_of(uint64_t bytes, uint64_t b) { return (bytes + b - 1) / b; }
static inline uint64_t cp_block_waste(const std::vector<cp_size> & in, uint64_t b) {
    uint64_t w = 0;
    for (const cp_size & x : in) { w += (cp_blocks_of(x.bytes, b) * b - x.bytes) * x.n; }
    return w;
}
static inline uint64_t cp_most_numerous(const std::vector<cp_size> & in) {
    uint64_t b = 0, n = 0;
    for (const cp_size & x : in) { if (x.n > n) { n = x.n; b = x.bytes; } }
    return b;
}
// the block size wasting least: candidates are the model's own expert sizes (a size whose experts all fit one block wastes nothing on them)
static inline uint64_t cp_best_block(const std::vector<cp_size> & in) {
    uint64_t b = 0, w = UINT64_MAX;
    for (const cp_size & x : in) { const uint64_t c = cp_block_waste(in, x.bytes); if (c < w || (c == w && x.bytes > b)) { w = c; b = x.bytes; } }
    return b;
}
