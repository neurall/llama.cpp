// Two streams for one card's part of a graph (decode: a token is thousands of small kernels, user 2026-11-19: "fuse kernels and parallel streams first").
//
// A decode token on one stream runs its kernels one after another although many do not depend on each other: the shared expert beside the routed ones, the
// delta-net gates beside its input projection, a sublayer's inject weights beside the sublayer. Measured 2026-11-18 (nsys, Qwen3.8 IQ3_S, 2 x RTX 3090): our
// kernels' times add up to 13.8 ms a token and fill 13.6 ms; Strata's add up to 13.0 ms and fill 10.9 ms, its graph's branches overlap on a card.
//
// The plan is made from what was launched, not from the graph's ops: a unit is one launch of the backend (a node, or the nodes a fusion ran as one), with the
// memory it reads and writes. Two units that touch the same memory, at least one of them writing, keep their graph order whatever they are (a data dependency,
// a buffer the allocator handed out twice, a view of a state tensor): that is the only rule, so a plan is correct for any assignment. A unit goes to the side
// stream when it may (movable: a pure op that writes memory of its own) and the estimate says it starts earlier there. A unit on one stream that needs a unit of
// the other waits for an event recorded right behind that unit (not "now": the other stream has gone on meanwhile, and waiting for all of it would undo the gain).
// A cut (a chunk of a captured graph ends, the streams are joined) starts the plan afresh.
//
// Pure functions, no backend: tests/test-streams-fork.cpp. The CUDA backend records the units of a graph's first call and plans at capture (ggml-cuda.cu).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

struct ggml_stream_range {
    uintptr_t lo, hi;   // [lo, hi)
};

struct ggml_stream_unit {
    int   leader  = -1;      // graph node index of its first node
    bool  movable = false;   // may run beside the main chain
    float cost    = 0.0f;    // estimated time, in any unit that is the same for all
    float bus     = 0.0f;    // of it, the time it moves bytes at the card's bandwidth (0: a kernel short of the bus; see the plan)
    std::vector<ggml_stream_range> reads, writes;
};

struct ggml_stream_plan {
    std::vector<int8_t>  stream;   // per unit: 0 main, 1 side
    std::vector<int32_t> wait;     // per unit: the unit of the other stream whose end it waits for (-1: none)
    std::vector<uint8_t> record;   // per unit: a later unit waits for its end
    float t_serial = 0.0f;         // the units one after another
    float t_plan   = 0.0f;         // as planned
    int   n_side   = 0;
};

static inline bool ggml_stream_overlap(const std::vector<ggml_stream_range> & a, const std::vector<ggml_stream_range> & b) {
    for (const auto & x : a) {
        for (const auto & y : b) {
            if (x.lo < y.hi && y.lo < x.hi) { return true; }
        }
    }
    return false;
}

// v is before u in graph order: must v have ended before u starts?
static inline bool ggml_stream_conflict(const ggml_stream_unit & v, const ggml_stream_unit & u) {
    return ggml_stream_overlap(v.writes, u.reads) || ggml_stream_overlap(v.writes, u.writes) || ggml_stream_overlap(v.reads, u.writes);
}

// cut[u] != 0: every unit before u has ended when u starts (the streams were joined). switch_cost: what a wait across the streams costs, in cost units; a
// unit moves to the side stream only when it starts earlier there by more than that.
static inline ggml_stream_plan ggml_stream_plan_make(const std::vector<ggml_stream_unit> & units, const std::vector<uint8_t> & cut, float switch_cost) {
    const int n = (int) units.size();
    ggml_stream_plan p;
    p.stream.assign(n, 0);
    p.wait.assign(n, -1);
    p.record.assign(n, 0);
    std::vector<float> finish(n, 0.0f);
    float clock[2]  = { 0.0f, 0.0f };
    int   synced[2] = { -1, -1 };   // synced[s]: the latest unit of the other stream that stream s has waited for
    int   first     = 0;            // the chunk's first unit
    int   heavy     = -1;           // the main stream's latest unit that moves bytes (bus > 0)
    for (int u = 0; u < n; ++u) {
        if (u == 0 || (u < (int) cut.size() && cut[u])) {
            const float t = std::max(clock[0], clock[1]);
            clock[0] = clock[1] = t;
            synced[0] = synced[1] = u - 1;
            first = u;
            heavy = -1;
        }
        int dep[2] = { -1, -1 };   // the latest unit of each stream that has to end first
        for (int v = u - 1; v >= first && (dep[0] < 0 || dep[1] < 0); --v) {
            if (dep[p.stream[v]] < 0 && ggml_stream_conflict(units[v], units[u])) { dep[p.stream[v]] = v; }
        }
        // two kernels that each move bytes at the card's bandwidth do not overlap: both slow down and the main chain's one ends later. One that would
        // go beside the main chain waits for the main stream's latest such kernel, so it overlaps the small kernels behind it (2026-11-19 trace:
        // z's projection beside the input projection gained nothing; beside the conv chain's small kernels it is free). The main stream waits for no bus
        if (units[u].bus > 0.0f && heavy > dep[0]) { dep[0] = heavy; }
        float start[2];
        for (int s = 0; s < 2; ++s) {
            const int o = dep[1 - s];
            start[s] = std::max(clock[s], o >= 0 && o > synced[s] ? finish[o] + switch_cost : 0.0f);
        }
        const int s = units[u].movable && start[1] + switch_cost < start[0] ? 1 : 0;
        const int o = dep[1 - s];
        if (s == 0 && units[u].bus > 0.0f) { heavy = u; }
        if (o >= 0 && o > synced[s]) {
            p.wait[u]   = o;
            p.record[o] = 1;
            synced[s]   = o;
            // what o's stream had waited for by then, this stream has now waited for too: nothing to add, synced[] holds units of the other stream only
        }
        p.stream[u] = (int8_t) s;
        p.n_side   += s;
        finish[u]   = start[s] + units[u].cost;
        clock[s]    = finish[u];
        p.t_serial += units[u].cost;
    }
    p.t_plan = std::max(clock[0], clock[1]);
    return p;
}

// the rule itself, checked for every pair: units that touch the same memory (one of them writing) run in graph order. The order a plan gives: the units of a
// stream one after another, a unit behind the unit it waits for, everything behind a cut.
static inline bool ggml_stream_plan_check(const std::vector<ggml_stream_unit> & units, const std::vector<uint8_t> & cut, const ggml_stream_plan & p) {
    const int n = (int) units.size();
    // known[u][s]: the latest unit of stream s that has ended for certain when u starts
    std::vector<int> known0(n, -1), known1(n, -1);
    int last[2] = { -1, -1 };
    int first = 0;
    for (int u = 0; u < n; ++u) {
        if (u == 0 || (u < (int) cut.size() && cut[u])) { first = u; last[0] = last[1] = -1; }
        int k[2] = { first - 1, first - 1 };
        const int s = p.stream[u];
        if (last[s] >= 0) { k[0] = std::max(k[0], known0[last[s]]); k[1] = std::max(k[1], known1[last[s]]); k[s] = std::max(k[s], last[s]); }
        const int w = p.wait[u];
        if (w >= 0) {
            if (w >= u || w < first || p.stream[w] == s || !p.record[w]) { return false; }
            k[0] = std::max(k[0], known0[w]); k[1] = std::max(k[1], known1[w]); k[1 - s] = std::max(k[1 - s], w);
        }
        known0[u] = k[0]; known1[u] = k[1];
        for (int v = first; v < u; ++v) {
            if (ggml_stream_conflict(units[v], units[u]) && k[p.stream[v]] < v) { return false; }
        }
        if (s == 1 && !units[u].movable) { return false; }
        last[s] = u;
    }
    return true;
}
