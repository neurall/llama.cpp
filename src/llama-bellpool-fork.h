// fork: the decode bell's CPU experts on a pool of its own (inspired by Strata's per-core expert pool, our own code).
//
// The bell (src/llama-moecache-fork.cpp, bell_serve) computes a layer's missing experts on the host while the card computes the cached ones. Through
// ggml_backend_graph_compute that host chain (up, gate, swiglu, down) took ~100 us for 1.7 experts on Qwen3.8 IQ3_S (2026-11-19), the card waited in
// 19 of 48 layers a token (1.49 ms); the bytes alone are ~25 us at 127 GB/s. The time is the pool's wake-up and the op-by-op barriers.
//
// Here: workers stay up (spin, then sleep after spin_us idle), the input is quantized once, the rows of all missing experts are split over all
// workers in one phase (up, gate and swiglu of a row by one worker), the down rows in a second phase (each worker quantizes the hidden vectors it
// reads). Two waits a layer; the single-thread swiglu + quantize step (9-14 us a layer, 2026-11-19) is gone.
//
// Pure: no ggml backend calls. The dot and quantize functions come in as pointers (ggml's CPU type traits in the engine; plain float in the test,
// tests/test-bellpool-fork.cpp).
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

typedef void (*bellpool_dot_t)(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
typedef void (*bellpool_quant_t)(const float * x, void * y, int64_t n);
typedef void (*bellpool_dot2_t)(int n, float * s_a, float * s_b, const void * xa, const void * xb, const void * y);   // two weight rows against one input

struct bellpool_type {   // one weight type: its dot with the quantized input, and how the input is quantized for it
    bellpool_dot_t   dot   = nullptr;
    bellpool_quant_t quant = nullptr;   // nullptr: the dot reads float input
    size_t           q_row = 0;         // bytes of a quantized input row of n_in elements (or of floats)
    size_t           w_row = 0;         // bytes of one weight row
};

struct bellpool_layer {   // one layer's missing experts for one token
    int n_embd = 0, n_ff = 0;
    bellpool_type up, gate, down;
    bellpool_dot2_t upgate2 = nullptr;            // an up row and a gate row against the same quantized input in one pass (the input's bytes loaded once)
    int n = 0;                                    // missing experts
    const char * w_up[16], * w_gate[16], * w_down[16];   // expert matrices: row r at w + r * w_row
    float * out[16];                              // n_embd floats each
    const float * x = nullptr;                    // n_embd
};

// the rows [lo, hi) of a phase of `total` rows split over n workers, worker w: contiguous, sizes differ by at most one
static inline void bellpool_split(int64_t total, int n, int w, int64_t & lo, int64_t & hi) {
    const int64_t base = total / n, extra = total % n;
    lo = (int64_t) w * base + std::min<int64_t>(w, extra);
    hi = lo + base + (w < extra ? 1 : 0);
}

struct bellpool {
    int n_workers = 1;   // worker 0 is the caller
    int64_t spin_us = 20000;
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> arrived{0};
    std::atomic<int> phase_done[3];   // [1] unused
    std::atomic<bool> stop{false};
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> sleeping{0};
    const bellpool_layer * job = nullptr;
    std::vector<uint8_t> qx_up, qx_gate;       // quantized inputs
    bool same_q = false;                       // gate reads qx_up
    std::vector<std::vector<uint8_t>> qh;      // per worker: quantized hidden vectors
    std::vector<float> hu;                     // swiglu outputs, n * n_ff

    // pin_base >= 0: worker w runs on CPU pin_base + w - 1 only (one physical core each; Linux numbers the SMT siblings after all cores)
    explicit bellpool(int n, int pin_base = -1) : n_workers(std::max(1, n)) {
        for (auto & p : phase_done) { p = 0; }
        for (int w = 1; w < n_workers; ++w) {
            th.emplace_back([this, w] { loop(w); });
#if defined(__linux__)
            if (pin_base >= 0) {
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(pin_base + w - 1, &set);
                pthread_setaffinity_np(th.back().native_handle(), sizeof(set), &set);
            }
#endif
        }
    }
    ~bellpool() {
        stop = true;
        { std::lock_guard<std::mutex> lk(mtx); gen.fetch_add(1); }
        cv.notify_all();
        for (auto & t : th) { t.join(); }
    }

    // two phases; worker w does its share. Up and gate of a row by the same worker, swiglu inline: no single-thread step between the phases
    void up_gate(int w) {
        const bellpool_layer & L = *job;
        const void * qu = L.up.quant ? (const void *) qx_up.data() : (const void *) L.x;
        const void * qg = L.gate.quant ? (const void *) (same_q ? qx_up.data() : qx_gate.data()) : (const void *) L.x;
        int64_t lo, hi;
        bellpool_split((int64_t) L.n * L.n_ff, n_workers, w, lo, hi);
        const bool two = L.upgate2 != nullptr && qu == qg;
        for (int64_t t = lo; t < hi; ++t) {
            const int e = (int) (t / L.n_ff), r = (int) (t % L.n_ff);
            float u = 0.0f, g = 0.0f;
            if (two) {
                L.upgate2(L.n_embd, &u, &g, L.w_up[e] + (size_t) r * L.up.w_row, L.w_gate[e] + (size_t) r * L.gate.w_row, qu);
            } else {
                L.up.dot(L.n_embd, &u, 0, L.w_up[e] + (size_t) r * L.up.w_row, 0, qu, 0, 1);
                L.gate.dot(L.n_embd, &g, 0, L.w_gate[e] + (size_t) r * L.gate.w_row, 0, qg, 0, 1);
            }
            hu[(size_t) t] = g / (1.0f + std::exp(-g)) * u;
        }
    }
    void down(int w) {   // each worker quantizes the hidden vectors of the experts its rows touch, into its own buffer (n_ff values each: cheap)
        const bellpool_layer & L = *job;
        int64_t lo, hi;
        bellpool_split((int64_t) L.n * L.n_embd, n_workers, w, lo, hi);
        if (lo >= hi) { return; }
        uint8_t * qb = L.down.quant ? qh[w].data() : nullptr;
        if (L.down.quant) {
            for (int e = (int) (lo / L.n_embd); e <= (int) ((hi - 1) / L.n_embd); ++e) { L.down.quant(hu.data() + (size_t) e * L.n_ff, qb + (size_t) e * L.down.q_row, L.n_ff); }
        }
        for (int64_t t = lo; t < hi; ++t) {
            const int e = (int) (t / L.n_embd), r = (int) (t % L.n_embd);
            const void * q = L.down.quant ? (const void *) (qb + (size_t) e * L.down.q_row) : (const void *) (hu.data() + (size_t) e * L.n_ff);
            float s = 0.0f;
            L.down.dot(L.n_ff, &s, 0, L.w_down[e] + (size_t) r * L.down.w_row, 0, q, 0, 1);
            L.out[e][r] = s;
        }
    }

    void wait_phase(int k, int target) {
        while (phase_done[k].load(std::memory_order_acquire) < target) { /* spin: a layer's phases are microseconds */ }
    }

    void loop(int w) {
        uint64_t seen = 0;
        auto idle_from = std::chrono::steady_clock::now();
        for (;;) {
            uint64_t g = gen.load(std::memory_order_acquire);
            if (g == seen) {
                if (stop) { return; }
                if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - idle_from).count() > spin_us) {
                    std::unique_lock<std::mutex> lk(mtx);
                    sleeping++;
                    cv.wait(lk, [&] { return gen.load() != seen || stop; });
                    sleeping--;
                }
                continue;
            }
            seen = g;
            if (stop) { return; }
            up_gate(w);
            phase_done[0].fetch_add(1, std::memory_order_acq_rel);
            wait_phase(0, n_workers);
            down(w);
            phase_done[2].fetch_add(1, std::memory_order_acq_rel);
            idle_from = std::chrono::steady_clock::now();
        }
    }

    // dev: LLAMA_BELL_TIME=1 sums the phases of the caller's view (ns) and logs every 2048 layers
    struct phase_time { int64_t quant = 0, upg = 0, wait0 = 0, dn = 0, wait2 = 0; int64_t n = 0, experts = 0; } pt;
    static int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // one layer: returns when every out row is written
    void run(const bellpool_layer & L) {
        if (L.n <= 0) { return; }
        static const bool timed = [] { const char * e = getenv("LLAMA_BELL_TIME"); return e && atoi(e) != 0; }();
        int64_t t0 = timed ? now_ns() : 0, t1 = 0;
        job = &L;
        if (L.up.quant)   { qx_up.resize(L.up.q_row);     L.up.quant(L.x, qx_up.data(), L.n_embd); }
        same_q = L.up.quant && L.gate.quant == L.up.quant && L.gate.q_row == L.up.q_row;
        if (L.gate.quant && !same_q) { qx_gate.resize(L.gate.q_row); L.gate.quant(L.x, qx_gate.data(), L.n_embd); }
        hu.resize((size_t) L.n * L.n_ff);
        if (L.down.quant) { qh.resize(n_workers); for (auto & b : qh) { if (b.size() < (size_t) L.n * L.down.q_row) { b.resize((size_t) L.n * L.down.q_row); } } }
        for (auto & p : phase_done) { p.store(0, std::memory_order_relaxed); }
        {
            std::lock_guard<std::mutex> lk(mtx);
            gen.fetch_add(1, std::memory_order_acq_rel);
        }
        if (sleeping.load() > 0) { cv.notify_all(); }
        if (timed) { t1 = now_ns(); pt.quant += t1 - t0; t0 = t1; }
        up_gate(0);
        if (timed) { t1 = now_ns(); pt.upg += t1 - t0; t0 = t1; }
        phase_done[0].fetch_add(1, std::memory_order_acq_rel);
        wait_phase(0, n_workers);
        if (timed) { t1 = now_ns(); pt.wait0 += t1 - t0; t0 = t1; }
        down(0);
        if (timed) { t1 = now_ns(); pt.dn += t1 - t0; t0 = t1; }
        phase_done[2].fetch_add(1, std::memory_order_acq_rel);
        wait_phase(2, n_workers);
        if (timed) {
            t1 = now_ns(); pt.wait2 += t1 - t0; pt.n++; pt.experts += L.n;
            if (pt.n % 2048 == 0) {
                const double k = 1e-3 / (double) pt.n;
                fprintf(stderr, "bell pool time: %lld layers, %.2f experts a layer, us a layer: quantize+wake %.1f, up/gate+swiglu own %.1f, wait slowest %.1f, quant+down own %.1f, wait slowest %.1f (workers %d)\n",
                        (long long) pt.n, (double) pt.experts / pt.n, pt.quant * k, pt.upg * k, pt.wait0 * k, pt.dn * k, pt.wait2 * k, n_workers);
                pt = phase_time();
            }
        }
    }
};
