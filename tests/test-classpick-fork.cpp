#include "llama-classpick-fork.h"
#include <cstdio>
#include <cassert>
int main() {
    struct M { const char * name; std::vector<cp_size> s; };
    auto mb = [](double m) { return (uint64_t) (m * 1048576.0); };
    // n = layers (each layer of a model has the same expert count)
    std::vector<M> ms = {
        { "GLM 3.5",   { { mb(10.31), 34 }, { mb(7.88), 6 }, { mb(9.59), 1 }, { mb(12.45), 1 } } },
        { "MiMo",      { { mb(7.88), 18 }, { mb(8.74), 12 }, { mb(9.59), 1 }, { mb(10.31), 1 } } },
        { "Qwen3.8 IQ4", { { mb(2.24), 43 }, { mb(3.08), 4 }, { mb(3.32), 1 } } },
    };
    for (auto & m : ms) {
        uint64_t tot = 0; for (auto & x : m.s) tot += x.bytes * x.n;
        const uint64_t b = cp_most_numerous(m.s);
        printf("%-12s block = most numerous %.2f MiB: waste %.1f%%, spans:", m.name, b / 1048576.0, 100.0 * cp_block_waste(m.s, b) / tot);
        for (auto & x : m.s) printf(" %.2f->%llu", x.bytes / 1048576.0, (unsigned long long) cp_blocks_of(x.bytes, b));
        uint64_t bb = b, bw = cp_block_waste(m.s, b);
        for (auto & x : m.s) for (int d = 1; d <= 3; ++d) { uint64_t c = x.bytes / d; uint64_t w = cp_block_waste(m.s, c); if (w < bw) { bw = w; bb = c; } }
        printf(" | least-waste block %.2f MiB: %.1f%%, spans:", bb / 1048576.0, 100.0 * bw / tot);
        for (auto & x : m.s) printf(" %.2f->%llu", x.bytes / 1048576.0, (unsigned long long) cp_blocks_of(x.bytes, bb));
        for (size_t k = 1; k <= 3; ++k) { cp_pick p = cp_pick_classes(m.s, k); printf(" | %zu classes %.1f%%", p.classes.size(), 100 * p.frac()); }
        printf("\n");
    }
    // unit checks
    std::vector<cp_size> t = { { 10, 34 }, { 8, 6 }, { 12, 1 } };
    assert(cp_most_numerous(t) == 10 && cp_blocks_of(12, 10) == 2 && cp_blocks_of(8, 10) == 1);
    assert(cp_block_waste(t, 10) == 2 * 6 + 8 * 1);
    cp_pick p = cp_pick_classes(t, 2); assert(p.classes.size() == 2 && p.classes[0] == 10 && p.classes[1] == 12 && p.waste == 12);
    p = cp_pick_classes(t, 3); assert(p.waste == 0 && p.classes.size() == 3);
    { std::vector<cp_size> q = { { 338, 8 }, { 392, 8 } }; assert(cp_best_block(q) == 392); assert(cp_best_block({ { 224, 43 }, { 308, 4 }, { 332, 1 } }, 2) == 112); assert(cp_best_block(q, 1, 100) == 400); }
    printf("OK\n");
}
