// poolx_tail_give / poolx_class_gives (llama-poolalloc-fork.h): gapless classes giving tail blocks
#include "../src/llama-poolalloc-fork.h"
#include <cassert>
#include <cstdio>
#include <random>
int main() {
    {   // hot tail expert moves onto the coldest inside, cold tail expert leaves
        std::vector<double> h = { 5, 1, 7, 9, 0.5 }; std::vector<uint8_t> p(5, 0);
        poolx_give g = poolx_tail_give(h, p, 2);
        assert(g.n_left == 3 && g.moves.size() == 1 && g.moves[0] == std::make_pair(3, 1));
        assert(g.evict.size() == 2 && g.evict[0] == 1 && g.evict[1] == 4);
    }
    {   // empty blocks inside take movers first, nothing evicted for them
        std::vector<double> h = { 3, -1, 2, 8 }; std::vector<uint8_t> p(4, 0);
        poolx_give g = poolx_tail_give(h, p, 1);
        assert(g.moves.size() == 1 && g.moves[0] == std::make_pair(3, 1) && g.evict.empty());
    }
    {   // a pinned tail block stops the give; pinned inside is never a destination
        std::vector<double> h = { 0, 4, 6, 9 }; std::vector<uint8_t> p = { 1, 0, 1, 0 };
        poolx_give g = poolx_tail_give(h, p, 2);
        assert(g.n_left == 3 && g.moves.size() == 1 && g.moves[0] == std::make_pair(3, 1));
    }
    {   // class pick by heat per byte
        std::vector<std::vector<double>> h = { { 10, 20 }, { 12, 30 } }; std::vector<std::vector<uint8_t>> p = { { 0, 0 }, { 0, 0 } };
        assert(poolx_class_gives(h, p, { 1, 2 }) == 1);   // 12/2 < 10/1
        p[1][0] = 1; assert(poolx_class_gives(h, p, { 1, 2 }) == 0);   // 10 < 30/2
    }
    std::mt19937 rng(7);   // random: kept set = the hottest of all unpinned, block count right, no move onto a pinned block
    for (int it = 0; it < 20000; ++it) {
        const int n = 1 + (int) (rng() % 12), k = (int) (rng() % (n + 1));
        std::vector<double> h(n); std::vector<uint8_t> p(n);
        for (int i = 0; i < n; ++i) { h[i] = (rng() % 4 == 0) ? -1 : (double) (rng() % 20); p[i] = rng() % 6 == 0; }
        poolx_give g = poolx_tail_give(h, p, k);
        std::vector<double> nh(h.begin(), h.begin() + g.n_left);
        for (auto & [f, t] : g.moves) { assert(t < g.n_left && f >= g.n_left && !p[t] && h[f] > h[t]); nh[t] = h[f]; }
        assert(g.n_left >= n - k && g.n_left <= n);
        for (int b = g.n_left; b < n; ++b) { assert(!p[b]); }
        size_t ex = 0; for (int b = g.n_left; b < n; ++b) ex += h[b] >= 0;
        size_t lost = 0; for (int32_t e : g.evict) { (void) e; ++lost; }
        assert(g.moves.size() + lost >= ex);
        // no expert evicted is hotter than a kept unpinned one
        double kept_min = 1e9; for (int b = 0; b < g.n_left; ++b) if (!p[b] && nh[b] >= 0) kept_min = std::min(kept_min, nh[b]);
        for (int32_t e : g.evict) assert(h[e] <= kept_min);
    }
    printf("OK\n");
}
