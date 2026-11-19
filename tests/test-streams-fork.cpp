// The two-stream plan of a card's graph (ggml/src/ggml-streams-fork.h): pure, no backend.
//   clang++ -std=c++17 -I ggml/src tests/test-streams-fork.cpp -o test-streams-fork && ./test-streams-fork
#include "ggml-streams-fork.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <random>

// a unit that writes buffer `out` (0: none) and reads the buffers `in`; a buffer b is the memory [b*100, b*100 + 100)
static ggml_stream_unit unit(int leader, bool movable, float cost, int out, std::vector<int> in) {
    ggml_stream_unit u;
    u.leader = leader; u.movable = movable; u.cost = cost;
    if (out > 0) { u.writes.push_back({ (uintptr_t) out * 100, (uintptr_t) out * 100 + 100 }); }
    for (int b : in) { u.reads.push_back({ (uintptr_t) b * 100, (uintptr_t) b * 100 + 100 }); }
    return u;
}

int main() {
    const float sw = 1.0f;
    {   // a chain: nothing to gain, everything stays on the main stream and nobody waits
        std::vector<ggml_stream_unit> u = { unit(0, true, 5, 1, {}), unit(1, true, 5, 2, {1}), unit(2, true, 5, 3, {2}), unit(3, true, 5, 4, {3}) };
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(p.n_side == 0 && p.t_plan == p.t_serial);
        for (size_t i = 0; i < u.size(); ++i) { assert(p.stream[i] == 0 && p.wait[i] == -1 && !p.record[i]); }
        assert(ggml_stream_plan_check(u, cut, p));
    }
    // the layer of a decode token in small: A is the sublayer input; B1..B3 the routed chain (heavy, not movable: it rings the host);
    // C1, C2 the shared expert, which reads A alone and stands behind the routed chain in graph order; J joins both
    auto layer = [&](int c1_out) {
        return std::vector<ggml_stream_unit>{ unit(0, false, 5, 1, {}), unit(1, false, 10, 2, {1}), unit(2, false, 10, 3, {2}), unit(3, false, 10, 4, {3}),
                                              unit(4, true, 6, c1_out, {1}), unit(5, true, 6, 6, {c1_out}), unit(6, false, 2, 7, {4, 6}) };
    };
    {
        const std::vector<ggml_stream_unit> u = layer(5);
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(p.stream[4] == 1 && p.stream[5] == 1 && p.n_side == 2);            // the shared expert runs beside the routed chain
        assert(p.wait[4] == 0 && p.record[0]);                                    // behind A, not behind whatever the main stream has got to by then
        assert(p.wait[5] == -1);                                                  // its second unit follows on the same stream
        assert(p.stream[6] == 0 && p.wait[6] == 5 && p.record[5]);                // the join waits for the side stream's last unit
        assert(p.t_serial == 49 && p.t_plan == 37);                               // the 12 of the shared expert are off the chain
        assert(ggml_stream_plan_check(u, cut, p));
    }
    {   // the allocator gave C1's output the buffer B2 reads (free again behind B2 in graph order): C1 may not write it before B2 has run
        const std::vector<ggml_stream_unit> u = layer(2);
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(ggml_stream_plan_check(u, cut, p));
        assert(p.stream[4] == 0 || p.wait[4] >= 2);
    }
    {   // a unit that is not movable stays on the main stream however early it could start
        std::vector<ggml_stream_unit> u = layer(5);
        u[4].movable = u[5].movable = false;
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(p.n_side == 0 && ggml_stream_plan_check(u, cut, p));
    }
    {   // a cut before C1: everything before it has ended, so C1 waits for nothing of the chunk before and nothing of it is recorded for C1
        const std::vector<ggml_stream_unit> u = layer(5);
        std::vector<uint8_t> cut(u.size(), 0);
        cut[4] = 1;
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(p.wait[4] == -1 && !p.record[0] && ggml_stream_plan_check(u, cut, p));
    }
    {   // a wrong plan is seen: the shared expert beside the routed chain without its wait for A
        const std::vector<ggml_stream_unit> u = layer(5);
        std::vector<uint8_t> cut(u.size(), 0);
        ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        p.wait[4] = -1;
        assert(!ggml_stream_plan_check(u, cut, p));
    }
    // any graph, any allocation: the plan keeps every pair that touches the same memory in graph order, and is never slower than one stream by its own estimate
    std::mt19937 rng(7);
    long n_side = 0, n_wait = 0;
    double gain = 0;
    for (int it = 0; it < 4000; ++it) {
        const int n = 2 + (int) (rng() % 60), n_buf = 2 + (int) (rng() % 24);
        std::vector<ggml_stream_unit> u;
        std::vector<uint8_t> cut(n, 0);
        for (int i = 0; i < n; ++i) {
            std::vector<int> in;
            for (int k = (int) (rng() % 4); k > 0; --k) { in.push_back(1 + (int) (rng() % n_buf)); }
            u.push_back(unit(i, rng() % 3 != 0, 1.0f + (float) (rng() % 30), rng() % 8 ? 1 + (int) (rng() % n_buf) : 0, in));
            cut[i] = i > 0 && rng() % 25 == 0;
        }
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(ggml_stream_plan_check(u, cut, p));
        assert(p.t_plan <= p.t_serial + 1e-3f);
        n_side += p.n_side;
        for (int i = 0; i < n; ++i) { n_wait += p.wait[i] >= 0; }
        gain += 1.0 - p.t_plan / p.t_serial;
    }
    assert(n_side > 0 && n_wait > 0);
    printf("ok: 4000 random graphs, %ld units beside the main chain, %ld waits, estimated time %.1f%% shorter on average\n", n_side, n_wait, 100.0 * gain / 4000);
    return 0;
}
