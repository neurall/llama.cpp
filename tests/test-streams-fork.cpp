// The stream plan of a card's graph (ggml/src/ggml-streams-fork.h): pure, no backend.
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
// the unit u waits for (-1: none; -2: more than one)
static int waits_for(const ggml_stream_plan & p, int u) {
    int w = -1;
    for (int t = 0; t < GGML_STREAMS_MAX; ++t) {
        if (p.wait[u][t] >= 0) { w = w == -1 ? p.wait[u][t] : -2; }
    }
    return w;
}

int main() {
    const float sw = 1.0f;
    {   // a chain: nothing to gain, everything stays on the main stream and nobody waits
        std::vector<ggml_stream_unit> u = { unit(0, true, 5, 1, {}), unit(1, true, 5, 2, {1}), unit(2, true, 5, 3, {2}), unit(3, true, 5, 4, {3}) };
        std::vector<uint8_t> cut(u.size(), 0);
        for (int k = 2; k <= GGML_STREAMS_MAX; ++k) {
            const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw, k);
            assert(p.n_side == 0 && p.t_plan == p.t_serial);
            for (size_t i = 0; i < u.size(); ++i) { assert(p.stream[i] == 0 && waits_for(p, (int) i) == -1 && !p.record[i]); }
            assert(ggml_stream_plan_check(u, cut, p));
        }
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
        assert(waits_for(p, 4) == 0 && p.record[0]);                              // behind A, not behind whatever the main stream has got to by then
        assert(waits_for(p, 5) == -1);                                            // its second unit follows on the same stream
        assert(p.stream[6] == 0 && waits_for(p, 6) == 5 && p.record[5]);          // the join waits for the side stream's last unit
        assert(p.t_serial == 49 && p.t_plan == 37);                               // the 12 of the shared expert are off the chain
        assert(ggml_stream_plan_check(u, cut, p));
    }
    {   // the allocator gave C1's output the buffer B2 reads (free again behind B2 in graph order): C1 may not write it before B2 has run
        const std::vector<ggml_stream_unit> u = layer(2);
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        assert(ggml_stream_plan_check(u, cut, p));
        assert(p.stream[4] == 0 || waits_for(p, 4) >= 2);
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
        assert(waits_for(p, 4) == -1 && !p.record[0] && ggml_stream_plan_check(u, cut, p));
    }
    {   // a wrong plan is seen: the shared expert beside the routed chain without its wait for A
        const std::vector<ggml_stream_unit> u = layer(5);
        std::vector<uint8_t> cut(u.size(), 0);
        ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw);
        p.wait[4].fill(-1);
        assert(!ggml_stream_plan_check(u, cut, p));
    }
    // A delta-net layer as the plan dump of build side_prio shows it (2026-11-19, Qwen3.8 IQ3_S, units 6..21 of layer 0): A the layer input; Q the
    // input projection and Z the gate projection, both fill the card; CC, CV the conv chain; L1, L2 the q and k norms; G1..G3 alpha, add, softplus;
    // B1, B2 beta and its sigmoid; D the delta net; N the gated norm, the only reader of Z. With one side stream L2 and B1 landed behind Z on it.
    enum { A, Q, Z, CC, CV, L1, L2, G1, G2, G3, B1, B2, D, N, GDN_N };
    const auto gdn = [&] {
        std::vector<ggml_stream_unit> u(GDN_N);
        u[A]  = unit(A,  false, 1.5f, 1, {});
        u[Q]  = unit(Q,  true, 27.3f, 2, {1});   u[Q].fills = true;
        u[Z]  = unit(Z,  true, 12.1f, 3, {1});   u[Z].fills = true;
        u[CC] = unit(CC, true,  1.7f, 4, {2});
        u[CV] = unit(CV, false, 3.1f, 5, {4});
        u[L1] = unit(L1, true,  3.0f, 6, {5});
        u[L2] = unit(L2, true,  3.0f, 7, {5});
        u[G1] = unit(G1, true,  1.8f, 8, {1});
        u[G2] = unit(G2, true,  1.5f, 9, {8});
        u[G3] = unit(G3, true,  3.0f, 10, {9});
        u[B1] = unit(B1, true,  1.8f, 11, {1});
        u[B2] = unit(B2, true,  1.5f, 12, {11});
        u[D]  = unit(D,  false, 6.8f, 13, {6, 7, 10, 12});
        u[N]  = unit(N,  true,  3.1f, 14, {13, 3});
        return u;
    };
    for (int k = 2; k <= GGML_STREAMS_MAX; ++k) {
        const std::vector<ggml_stream_unit> u = gdn();
        std::vector<uint8_t> cut(u.size(), 0);
        const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw, k);
        assert(ggml_stream_plan_check(u, cut, p));
        assert(p.stream[Q] == 0 && p.stream[Z] >= 1);                             // Z beside the input projection
        for (int i = Z + 1; i < N; ++i) { assert(p.stream[i] != p.stream[Z]); }   // nothing queues behind Z before its reader
        assert(p.wait[N][p.stream[Z]] == Z);
        int beside = 0;
        for (int i = CC; i < D; ++i) { beside += p.stream[i] >= 1; }
        assert(beside >= (k == 2 ? 0 : k == 3 ? 3 : 6));   // one side stream: Z has it; two: the k norm and beta too; three: the gate chain as well
        if (k >= 4) {   // the gate chain starts behind the layer input, beside the input projection (a stream's units keep graph order: with fewer
                        // streams the k norm has taken the free one before the gate chain's turn, and nothing fills the gap in front of it)
            assert(p.stream[G1] >= 1 && p.wait[G1][0] == A && p.stream[G2] == p.stream[G1] && p.stream[G3] == p.stream[G1]);
            assert(p.stream[L2] >= 1 && p.stream[L2] != p.stream[G1]);
        }
        printf("delta-net layer, %d streams: %d units beside the main chain, estimate %.1f -> %.1f\n", k, p.n_side, p.t_serial, p.t_plan);
    }
    {   // the count is picked from the estimate: the delta-net layer takes four or more, the shared expert beside the routed chain one side stream
        const std::vector<ggml_stream_unit> g = gdn(), l = layer(5);
        std::vector<uint8_t> cg(g.size(), 0), cl(l.size(), 0);
        const ggml_stream_plan pg = ggml_stream_plan_pick(g, cg, sw, GGML_STREAMS_MAX), pl = ggml_stream_plan_pick(l, cl, sw, GGML_STREAMS_MAX);
        assert(pg.n_streams >= 4 && pg.t_plan == ggml_stream_plan_make(g, cg, sw, GGML_STREAMS_MAX).t_plan && ggml_stream_plan_check(g, cg, pg));
        assert(ggml_stream_plan_make(g, cg, sw, pg.n_streams - 1).t_plan > pg.t_plan);   // one stream fewer is slower by the estimate
        assert(pl.n_streams == 2 && pl.t_plan == 37 && ggml_stream_plan_check(l, cl, pl));
    }
    {   // more streams never make this layer's estimate longer
        const std::vector<ggml_stream_unit> u = gdn();
        std::vector<uint8_t> cut(u.size(), 0);
        float t = 1e30f;
        for (int k = 2; k <= GGML_STREAMS_MAX; ++k) {
            const float tk = ggml_stream_plan_make(u, cut, sw, k).t_plan;
            assert(tk <= t + 1e-3f);
            t = tk;
        }
    }
    // any graph, any allocation, any number of streams: the plan keeps every pair that touches the same memory in graph order, is never slower than one
    // stream by its own estimate, and puts nothing behind a unit that fills the card on a side stream before a unit that has to follow it
    std::mt19937 rng(7);
    for (int k = 2; k <= GGML_STREAMS_MAX; ++k) {
        long n_side = 0, n_wait = 0;
        double gain = 0;
        for (int it = 0; it < 4000; ++it) {
            const int n = 2 + (int) (rng() % 60), n_buf = 2 + (int) (rng() % 24);
            std::vector<ggml_stream_unit> u;
            std::vector<uint8_t> cut(n, 0);
            for (int i = 0; i < n; ++i) {
                std::vector<int> in;
                for (int j = (int) (rng() % 4); j > 0; --j) { in.push_back(1 + (int) (rng() % n_buf)); }
                u.push_back(unit(i, rng() % 3 != 0, 1.0f + (float) (rng() % 30), rng() % 8 ? 1 + (int) (rng() % n_buf) : 0, in));
                u.back().fills = rng() % 4 == 0;
                cut[i] = i > 0 && rng() % 25 == 0;
            }
            const ggml_stream_plan p = ggml_stream_plan_make(u, cut, sw, k);
            assert(ggml_stream_plan_check(u, cut, p));
            assert(p.t_plan <= p.t_serial + 1e-3f);
            int prev[GGML_STREAMS_MAX];
            for (int & x : prev) { x = -1; }
            for (int i = 0; i < n; ++i) {
                if (cut[i]) { for (int & x : prev) { x = -1; } }
                const int s = p.stream[i], h = prev[s];
                if (s >= 1 && h >= 0 && u[h].fills) {
                    bool followed = false;
                    for (int v = h + 1; v <= i && !followed; ++v) { followed = ggml_stream_conflict(u[h], u[v]); }
                    assert(followed);
                }
                prev[s] = i;
                for (int t = 0; t < GGML_STREAMS_MAX; ++t) { n_wait += p.wait[i][t] >= 0; }
            }
            n_side += p.n_side;
            gain += 1.0 - p.t_plan / p.t_serial;
        }
        assert(n_side > 0 && n_wait > 0);
        printf("ok: %d streams, 4000 random graphs, %ld units beside the main chain, %ld waits, estimated time %.1f%% shorter on average\n", k, n_side, n_wait, 100.0 * gain / 4000);
    }
    {   // the side streams' scratch only for a graph the plan may split. Kimi K3.1's 5056-token prompt graph (2026-11-20): its largest node 5056 x 7168 x 4 B,
        // the side product's scratch about 40 MB; the 6 x 40 MB it had been given cost the third pass its room beside the KV
        assert(ggml_stream_side_room(4, (size_t) 40 << 20, (size_t) 5056 * 7168 * 4) == 0);
        assert(ggml_stream_side_room(4, 1000, (size_t) 2 << 20) == (size_t) 3 * 2 * 1024);   // a decode graph, every node within 4 MiB: two pieces a side stream
        assert(ggml_stream_side_room(2, 256, GGML_STREAMS_NODE_MAX) == 2 * 256);
        assert(ggml_stream_side_room(0, 1000, 1024) == 0 && ggml_stream_side_room(1, 1000, 1024) == 0);   // streams off (GGML_CUDA_STREAMS=0)
        assert(ggml_stream_graph_may_split(GGML_STREAMS_NODE_MAX) && !ggml_stream_graph_may_split(GGML_STREAMS_NODE_MAX + 1));
        printf("ok: side-stream scratch only for graphs the plan may split\n");
    }
    return 0;
}
