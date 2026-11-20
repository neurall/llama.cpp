// The RAM pool's fill sizes its own thread count from what it reads (ggml-rampool-fork.cpp, the fill). The read-thread probe times 1 MiB preads, one in flight
// a thread; the fill and the pool's readers issue one slice at a time through the async engine (ggml-read-fork.h) and wait for its last piece, so between two
// slices a thread has nothing on the drive. Kimi K3.1 on PC1, 2026-11-20: a probe that found the drive's rate at 4 threads (2: 5.38 GB/s, 4: 6.27) left the
// fill at 4.58 GB/s and the prompt stream at 4.4-4.5 GB/s, where b15843's probe had picked 16 threads and the same drive and files read 5.34 and 5.38.
// From the probe's count the fill doubles its threads while a window reads more than 5% faster than the one before; the doubling that bought less is taken
// back (the smallest count near the best, as the probe's own rule). The pool's readers start with the count the fill settled on.
#pragma once

#include <algorithm>

struct rp_thread_tune {
    int    n        = 1;       // threads now
    double last_gbs = 0.0;     // the rate of the last window (0: none yet)
    bool   settled  = false;   // no more doubling
};

// one window's rate at t.n threads: the next count (more threads to start when it grew), or settled
static inline rp_thread_tune rp_tune_step(rp_thread_tune t, double gbs, int cap) {
    if (t.settled) { return t; }
    if (t.last_gbs > 0.0 && gbs < 1.05 * t.last_gbs) { t.n = std::max(1, t.n / 2); t.settled = true; return t; }   // the doubling bought less than 5%: the count before it
    t.last_gbs = gbs;
    if (t.n >= cap) { t.settled = true; return t; }
    t.n = std::min(cap, t.n * 2);
    return t;
}

// The pool's start and its grows against the RAM a request needs beside it (llama-moecache-fork.cpp: rp_start_want, rp_resize). Kimi K3.1 on PC1, 2026-11-20:
// sessions of short requests grew the pool and stored 12.05 GiB kept beside it; the next start took 229.6 GiB where the equation gave 217.0, its 12.5k-token
// prompt needed 13.3 GiB beside the pool (the cross-card staging blocks of a full pass, its buffers) and RAM available fell to 1.5 GiB: the server was killed.
// The need is measured at each request's lowest point and the largest one kept; a learned start above the equation waits for one.
struct rp_size_rule {
    size_t eq;       // the equation's start (common_auto_pin's budget less what the loader pinned)
    size_t avail;    // RAM available now
    size_t kept;     // stored: the RAM the last session left beside its pool (0: none stored)
    size_t need;     // stored: the most RAM a request needed beside the pool, at its lowest point (0: none measured)
    size_t gfloor;   // what a grow leaves free (rp_grow_floor)
};
static inline size_t rp_start_pick(const rp_size_rule & r) {
    if (r.eq == 0 || r.kept == 0) { return r.eq; }
    const size_t keep = std::max(std::max(r.kept, r.gfloor), r.need > 0 ? r.need + r.gfloor : 0);
    if (r.avail <= keep) { return r.eq; }   // less RAM than that room: the equation, computed from what is available now
    const size_t learned = r.avail - keep;
    return r.need > 0 ? learned : std::min(learned, r.eq);   // no need measured: the learned start only ever lowers the equation's
}
// the RAM a request needed beside the pool: available at the start, less the pool, less the least available since
static inline size_t rp_need_now(size_t avail0, size_t pool, size_t low) {
    return avail0 > pool + low ? avail0 - pool - low : 0;
}
// a grow of `want` bytes stops where the measured need and the grow floor stay free (0 need: none measured, the grow's own rule)
// shortcut: a fresh state measures its first requests only, a later heavier one can still find the pool grown into its room; upgrade with a give-back during a request
static inline size_t rp_grow_cap(size_t want, size_t avail0, size_t pool, size_t need, size_t gfloor) {
    if (need == 0) { return want; }
    const size_t top = avail0 > need + gfloor ? avail0 - need - gfloor : 0;
    return pool >= top ? 0 : std::min(want, top - pool);
}
