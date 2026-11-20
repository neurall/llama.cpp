// Two streams for one card's part of a graph (decode: a token is thousands of small kernels, user 2026-11-19: "fuse kernels and parallel streams first").
//
// A decode token on one stream runs its kernels one after another although many do not depend on each other: the shared expert beside the routed ones, the
// delta-net gates beside its input projection, a sublayer's inject weights beside the sublayer. Measured 2026-11-18 (nsys, Qwen3.8 IQ3_S, 2 x RTX 3090): our
// kernels' times add up to 13.8 ms a token and fill 13.6 ms: next to nothing overlaps.
//
// The plan is made from what was launched, not from the graph's ops: a unit is one launch of the backend (a node, or the nodes a fusion ran as one), with the
// memory it reads and writes. Two units that touch the same memory, at least one of them writing, keep their graph order whatever they are (a data dependency,
// a buffer the allocator handed out twice, a view of a state tensor): that is the only rule, so a plan is correct for any assignment. A unit goes to the side
// stream when it may (movable: a pure op that writes memory of its own) and the estimate says it starts earlier there. A unit on one stream that needs a unit of
// the other waits for an event recorded right behind that unit (not "now": the other stream has gone on meanwhile, and waiting for all of it would undo the gain).
// A cut (a chunk of a captured graph ends, the streams are joined) starts the plan afresh.
//
// More than one stream beside the main one (2026-11-19, plan dump and trace of build side_prio): with one side stream the delta net's k norm and a gate
// projection were planned behind z (the gate projection, 8.85 MB) on it and waited for z's real end, which the plan does not know: a kernel that fills
// the card gets its blocks only when the main stream's kernel has handed out its last wave. So a unit that fills the card (fills) holds its side stream
// until a later unit waits for it; other units go to another side stream or stay on the main one.
//
// Pure functions, no backend: tests/test-streams-fork.cpp. The CUDA backend records the units of a graph's first call and plans at capture (ggml-cuda.cu).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#define GGML_STREAMS_MAX 8   // the main stream and up to seven beside it (the CUDA backend holds 8 streams a card)

struct ggml_stream_range {
    uintptr_t lo, hi;   // [lo, hi)
};

struct ggml_stream_unit {
    int   leader  = -1;      // graph node index of its first node
    bool  movable = false;   // may run beside the main chain
    bool  fills   = false;   // moves bytes for longer than a launch takes: it fills the card, the plan does not know when it ends
    float cost    = 0.0f;    // estimated time, in any unit that is the same for all
    std::vector<ggml_stream_range> reads, writes;
};

typedef std::array<int32_t, GGML_STREAMS_MAX> ggml_stream_ids;   // one unit index per stream (-1: none)

struct ggml_stream_plan {
    std::vector<int8_t>          stream;   // per unit: 0 main, 1.. beside it
    std::vector<ggml_stream_ids> wait;     // per unit and other stream: the unit of that stream whose end it waits for (-1: none)
    std::vector<uint8_t>         record;   // per unit: a later unit waits for its end
    float t_serial  = 0.0f;                // the units one after another
    float t_plan    = 0.0f;                // as planned
    int   n_side    = 0;
    int   n_streams = 2;
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
// unit moves beside the main chain only when it starts earlier there by more than that. n_streams: the main stream and the streams beside it.
static inline ggml_stream_plan ggml_stream_plan_make(const std::vector<ggml_stream_unit> & units, const std::vector<uint8_t> & cut, float switch_cost, int n_streams = 2) {
    const int n = (int) units.size();
    const int K = std::max(1, std::min(n_streams, GGML_STREAMS_MAX));
    ggml_stream_ids none; none.fill(-1);
    ggml_stream_plan p;
    p.n_streams = K;
    p.stream.assign(n, 0);
    p.wait.assign(n, none);
    p.record.assign(n, 0);
    std::vector<float> finish(n, 0.0f);
    std::vector<ggml_stream_ids> known(n, none);   // known[u][t]: the latest unit of stream t that has ended for certain when u starts
    float clock[GGML_STREAMS_MAX] = { 0.0f };
    int   last[GGML_STREAMS_MAX];                  // the chunk's latest unit of each stream
    int   hold[GGML_STREAMS_MAX];                  // a side stream's unit that fills the card and that no later unit has waited for yet
    int   first = 0;                               // the chunk's first unit
    for (int u = 0; u < n; ++u) {
        if (u == 0 || (u < (int) cut.size() && cut[u])) {
            float t = 0.0f;
            for (int s = 0; s < K; ++s) { t = std::max(t, clock[s]); }
            for (int s = 0; s < GGML_STREAMS_MAX; ++s) { clock[s] = t; last[s] = -1; hold[s] = -1; }
            first = u;
        }
        int dep[GGML_STREAMS_MAX];   // the latest unit of each stream that has to end first
        for (int & d : dep) { d = -1; }
        for (int v = u - 1, left = K; v >= first && left > 0; --v) {
            if (dep[p.stream[v]] < 0 && ggml_stream_conflict(units[v], units[u])) { dep[p.stream[v]] = v; --left; }
        }
        // what stream s knows to have ended when its next unit starts: what its latest unit knew, and that unit
        const auto base = [&](int s) {
            ggml_stream_ids k; k.fill(first - 1);
            if (last[s] >= 0) { k = known[last[s]]; k[s] = last[s]; }
            return k;
        };
        float start[GGML_STREAMS_MAX];
        for (int s = 0; s < K; ++s) {
            const ggml_stream_ids k = base(s);
            start[s] = clock[s];
            for (int t = 0; t < K; ++t) {
                if (t != s && dep[t] > k[t]) { start[s] = std::max(start[s], finish[dep[t]] + switch_cost); }
            }
        }
        int s = 0;
        for (int c = 1; c < K && units[u].movable; ++c) {
            const bool free = hold[c] < 0 || dep[c] >= hold[c];   // behind a unit that fills the card only what has to follow it anyway
            if (free && start[c] + switch_cost < start[0] && (s == 0 || start[c] < start[s])) { s = c; }
        }
        ggml_stream_ids k = base(s);
        for (;;) {   // the waits, the latest unit first: it knows the most, a wait for it may cover another
            int t = -1;
            for (int c = 0; c < K; ++c) {
                if (c != s && dep[c] > k[c] && (t < 0 || dep[c] > dep[t])) { t = c; }
            }
            if (t < 0) { break; }
            const int o = dep[t];
            p.wait[u][t] = o;
            p.record[o]  = 1;
            for (int r = 0; r < GGML_STREAMS_MAX; ++r) { k[r] = std::max(k[r], known[o][r]); }
            k[t] = std::max(k[t], (int32_t) o);
        }
        for (int c = 1; c < K; ++c) {
            if (hold[c] >= 0 && dep[c] >= hold[c]) { hold[c] = -1; }   // u runs behind it: its end is known from here on
        }
        known[u]    = k;
        p.stream[u] = (int8_t) s;
        p.n_side   += s > 0;
        finish[u]   = start[s] + units[u].cost;
        clock[s]    = finish[u];
        last[s]     = u;
        if (s > 0 && units[u].fills) { hold[s] = u; }
        p.t_serial += units[u].cost;
    }
    for (int s = 0; s < K; ++s) { p.t_plan = std::max(p.t_plan, clock[s]); }
    return p;
}

// The plan with the fewest streams that reaches the best estimate up to n_max streams: a graph without branches beside its main chain gets none, a layer
// with one branch one side stream (user 2026-11-19: the count is the planner's to decide, from a reasonable start). The estimate is the start value;
// what the card measures decides after it.
static inline ggml_stream_plan ggml_stream_plan_pick(const std::vector<ggml_stream_unit> & units, const std::vector<uint8_t> & cut, float switch_cost, int n_max) {
    ggml_stream_plan best = ggml_stream_plan_make(units, cut, switch_cost, n_max);
    for (int k = std::min(n_max, GGML_STREAMS_MAX) - 1; k >= 2; --k) {
        ggml_stream_plan p = ggml_stream_plan_make(units, cut, switch_cost, k);
        if (p.t_plan > best.t_plan + 1e-3f) { break; }
        best = std::move(p);
    }
    return best;
}

// the rule itself, checked for every pair: units that touch the same memory (one of them writing) run in graph order. The order a plan gives: the units of a
// stream one after another, a unit behind the units it waits for (and behind what those knew to have ended), everything behind a cut.
static inline bool ggml_stream_plan_check(const std::vector<ggml_stream_unit> & units, const std::vector<uint8_t> & cut, const ggml_stream_plan & p) {
    const int n = (int) units.size();
    const int K = std::max(1, std::min(p.n_streams, GGML_STREAMS_MAX));
    ggml_stream_ids none; none.fill(-1);
    std::vector<ggml_stream_ids> known(n, none);
    int last[GGML_STREAMS_MAX];
    for (int & l : last) { l = -1; }
    int first = 0;
    for (int u = 0; u < n; ++u) {
        if (u == 0 || (u < (int) cut.size() && cut[u])) {
            first = u;
            for (int & l : last) { l = -1; }
        }
        const int s = p.stream[u];
        if (s < 0 || s >= K) { return false; }
        ggml_stream_ids k; k.fill(first - 1);
        if (last[s] >= 0) { k = known[last[s]]; k[s] = last[s]; }
        for (int t = 0; t < GGML_STREAMS_MAX; ++t) {
            const int w = p.wait[u][t];
            if (w < 0) { continue; }
            if (t == s || t >= K || w >= u || w < first || p.stream[w] != t || !p.record[w]) { return false; }
            for (int r = 0; r < GGML_STREAMS_MAX; ++r) { k[r] = std::max(k[r], known[w][r]); }
            k[t] = std::max(k[t], (int32_t) w);
        }
        known[u] = k;
        for (int v = first; v < u; ++v) {
            if (ggml_stream_conflict(units[v], units[u]) && k[p.stream[v]] < v) { return false; }
        }
        if (s >= 1 && !units[u].movable) { return false; }
        last[s] = u;
    }
    return true;
}
