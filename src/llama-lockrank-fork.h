#pragma once
// fork: lock order for the planner's mutexes (tests/test-lockrank-fork.cpp). The order lived only in comments, and g_pp_mu was taken a second time on one thread:
// build lp_not_under_xsplit (2026-11-14) hung at start for 600 s. A thread takes ranked mutexes in rising rank only:
//   g_plan_mu (dense split planner) 10  <  g_dsf_mu (split estimator) 20  <  g_pp_mu (prompt planner) 30
// (g_dsf_mu's decide reads the prompt planner's cells and one-card draw under it; the first ranking, from a comment that called g_dsf_mu innermost, had
// these two the other way and failed every row of its first run)
// A second lock of a mutex the thread holds aborts at once with its name (a std::mutex hangs there for good). A lock below a held rank logs one
// "moe-cache: INV lock order" line per mutex, which fails the row (run.py ENGINE_CHECK); the lock is still taken (two threads must interleave to deadlock).
// Cost: a scan of the few ranks the thread holds, per lock.

#include "ggml.h"

#include <atomic>
#include <cstdlib>
#include <mutex>

extern "C" { GGML_API void ggml_log_internal(enum ggml_log_level level, const char * format, ...); }   // ggml-impl.h, exported by ggml-base

struct llama_ranked_mutex {
    std::mutex        m;
    const int         rank;
    const char *      name;
    std::atomic<bool> told{false};

    llama_ranked_mutex(int r, const char * n) : rank(r), name(n) {}

    struct held_t { const llama_ranked_mutex * m[8]; int n = 0; };
    static held_t & held() { static thread_local held_t h; return h; }

    void lock() {
        held_t & h = held();
        for (int i = 0; i < h.n; ++i) {
            if (h.m[i] == this) {
                ggml_log_internal(GGML_LOG_LEVEL_ERROR, "moe-cache: INV lock order: %s taken again on the thread that holds it (a std::mutex hangs here for good)\n", name);
                abort();
            }
            if (h.m[i]->rank >= rank && !told.exchange(true)) {
                ggml_log_internal(GGML_LOG_LEVEL_WARN, "moe-cache: INV lock order: %s (rank %d) taken while %s (rank %d) is held: a thread that takes them the other way deadlocks\n",
                        name, rank, h.m[i]->name, h.m[i]->rank);
            }
        }
        m.lock();
        if (h.n < 8) { h.m[h.n++] = this; }
    }

    void unlock() {
        held_t & h = held();
        for (int i = h.n - 1; i >= 0; --i) {
            if (h.m[i] == this) {
                for (int j = i; j + 1 < h.n; ++j) { h.m[j] = h.m[j + 1]; }
                h.n--;
                break;
            }
        }
        m.unlock();
    }
};
