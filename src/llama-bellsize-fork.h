#pragma once
// fork: how many workers the decode bell's CPU pool gets (src/llama-bellpool-fork.h; src/llama-moecache-fork.cpp, bell_pool_n).
//
// A pool worker spins through a decode token (bellpool::spin_us, 20 ms, longer than a token), and so does the decode team when its graph computes on the
// CPU every token (OpenMP's KMP_BLOCKTIME, 200 ms, ggml-cpu.c; Hy4-Flash: 12 lightning indexers a token). A pool phase ends with its slowest worker: a
// worker that waits on a run queue holds the card's layer for a scheduler slice. A thread that wakes (an upload worker, a card's driver thread, the OS)
// looks for an idle CPU in its own L3 domain only (Linux: select_idle_sibling); with every CPU spinning it queues behind a spinner.
// Hy4-Flash 13k/1k, PC1 (64 CPUs, 32 cores, 8 L3 domains, decode team 32), 2026-11-26, a worker a core (the rule before): 64 spinning threads, run-queue
// wait 23-26% of a CPU (median), 7-24% of the tokens over 1.5x the median (p99 71-107 ms), 30-32 o/s and in 5 of 11 cells a slow phase under stock
// (21.2-24.4 o/s against 28.95/30.08); 28 workers: wait 5% (p90 30%); 24 workers: wait 3-4% (p90 11-18%), tokens over 1.5x 1% (p99 38-39 ms),
// 34.6/35.2 and 34.1/35.1 o/s; 16 workers: wait 1-2%, 34.9/35.3 o/s.
// The rule: the pool takes the CPUs the spinning decode team leaves, less one CPU an L3 domain for the threads that wake; at most a worker a core, at
// least the bell thread itself (worker 0). A decode team whose graph has no CPU work waits instead of spinning (n_team 0): a worker a core, as before.
// test: tests/test-bellsize-fork.cpp

#include <algorithm>

// n_allowed: CPUs the process may run on; n_physical: their cores; n_team: the decode team's threads when its graph computes on the CPU every token, else 0;
// n_l3: L3 domains of those CPUs
static inline int bell_pool_size(int n_allowed, int n_physical, int n_team, int n_l3) {
    return std::max(1, std::min(n_physical, n_allowed - n_team - n_l3));
}
