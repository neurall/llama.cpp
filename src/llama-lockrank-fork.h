#pragma once
// fork: lock order check for the planner's mutexes (tests/test-lockrank-fork.cpp). The order is LEARNED from the locks the process takes, never
// declared: a lock taken while another is held records the edge "held, then taken"; a new edge that closes a cycle (B under A once, A under B now,
// also through other mutexes and on other threads) is a deadlock two threads can hit, and logs one "moe-cache: INV lock order" line per mutex,
// which fails the row (run.py ENGINE_CHECK); the lock is still taken. History: the order lived in comments, then in hand-set ranks; the ranks of
// 2026-11-14 had g_dsf_mu and g_pp_mu the wrong way round (the comment they came from was wrong) and failed every row of their first run.
// A second lock of a mutex the thread holds aborts at once with its name (a std::mutex hangs there for good: build lp_not_under_xsplit,
// 2026-11-14, the server hung at start for 600 s with g_pp_mu taken twice on one thread).
// Cost per lock: a scan of the few mutexes the thread holds and one atomic load each; the cycle search runs only when an edge is new (a few times
// per process). Up to 32 checked mutexes are tracked; more are still locked, without the order check.

#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>

extern "C" { GGML_API void ggml_log_internal(enum ggml_log_level level, const char * format, ...); }   // ggml-impl.h, exported by ggml-base

struct llama_checked_mutex {
    std::mutex        m;
    const char *      name;
    const int         id;
    std::atomic<bool> told{false};

    static constexpr int N = 32;

    explicit llama_checked_mutex(const char * n) : name(n), id(next_id().fetch_add(1)) {}

    static std::atomic<int> & next_id() { static std::atomic<int> v{0}; return v; }
    static std::atomic<uint32_t> * after() { static std::atomic<uint32_t> v[N]; return v; }   // after()[a] bit b: b was taken while a was held

    struct held_t { const llama_checked_mutex * m[8]; int n = 0; };
    static held_t & held() { static thread_local held_t h; return h; }

    // is `to` reachable from `from` over the learned edges
    static bool reaches(int from, int to) {
        uint32_t seen = 1u << from, todo = 1u << from;
        while (todo) {
            const int v = __builtin_ctz(todo);
            todo &= todo - 1;
            const uint32_t next = after()[v].load() & ~seen;
            if (next & (1u << to)) { return true; }
            seen |= next; todo |= next;
        }
        return false;
    }

    void lock() {
        held_t & h = held();
        for (int i = 0; i < h.n; ++i) {
            if (h.m[i] == this) {
                ggml_log_internal(GGML_LOG_LEVEL_ERROR, "moe-cache: INV lock order: %s taken again on the thread that holds it (a std::mutex hangs here for good)\n", name);
                abort();
            }
            const int a = h.m[i]->id;
            if (a >= N || id >= N || (after()[a].load() & (1u << id))) { continue; }
            after()[a].fetch_or(1u << id);   // new edge a -> this: does this already lead back to a?
            if (reaches(id, a) && !told.exchange(true)) {
                ggml_log_internal(GGML_LOG_LEVEL_WARN, "moe-cache: INV lock order: %s taken while %s is held, and %s was taken under %s before (directly or through others): two threads doing both deadlock\n",
                        name, h.m[i]->name, h.m[i]->name, name);
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
