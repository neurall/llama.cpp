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
