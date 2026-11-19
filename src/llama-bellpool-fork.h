// fork: the decode bell's CPU experts on a pool of its own (inspired by Strata's per-core expert pool, our own code).
//
// The bell (src/llama-moecache-fork.cpp, bell_serve) computes a layer's missing experts on the host while the card computes the cached ones. Through
// ggml_backend_graph_compute that host chain (up, gate, swiglu, down) took ~100 us for 1.7 experts on Qwen3.8 IQ3_S (2026-11-19), the card waited in
// 19 of 48 layers a token (1.49 ms); the bytes alone are ~25 us at 127 GB/s. The time is the pool's wake-up and the op-by-op barriers.
//
// Here: workers stay up (spin, then sleep after spin_us idle), the input is quantized once, the up and gate rows of all missing experts are split over
// all workers in one phase, swiglu + quantize of the hidden vector by one thread, the down rows in a second phase. Three waits a layer.
//
// Pure: no ggml backend calls. The dot and quantize functions come in as pointers (ggml's CPU type traits in the engine; plain float in the test,
// tests/test-bellpool-fork.cpp).
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

typedef void (*bellpool_dot_t)(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
typedef void (*bellpool_quant_t)(const float * x, void * y, int64_t n);

struct bellpool_type {   // one weight type: its dot with the quantized input, and how the input is quantized for it
    bellpool_dot_t   dot   = nullptr;
    bellpool_quant_t quant = nullptr;   // nullptr: the dot reads float input
    size_t           q_row = 0;         // bytes of a quantized input row of n_in elements (or of floats)
    size_t           w_row = 0;         // bytes of one weight row
};

struct bellpool_layer {   // one layer's missing experts for one token
    int n_embd = 0, n_ff = 0;
    bellpool_type up, gate, down;
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
    std::atomic<int> phase_done[3];
    std::atomic<bool> stop{false};
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> sleeping{0};
    const bellpool_layer * job = nullptr;
    std::vector<uint8_t> qx_up, qx_gate, qh;   // quantized inputs
    std::vector<float> hu, hg;                 // up and gate outputs, n * n_ff each

    explicit bellpool(int n) : n_workers(std::max(1, n)) {
        for (auto & p : phase_done) { p = 0; }
        for (int w = 1; w < n_workers; ++w) { th.emplace_back([this, w] { loop(w); }); }
    }
    ~bellpool() {
        stop = true;
        { std::lock_guard<std::mutex> lk(mtx); gen.fetch_add(1); }
        cv.notify_all();
        for (auto & t : th) { t.join(); }
    }

    // the three phases; worker w does its share, the caller (w = 0) the single-thread parts
    void up_gate(int w) {
        const bellpool_layer & L = *job;
        int64_t lo, hi;
        bellpool_split((int64_t) 2 * L.n * L.n_ff, n_workers, w, lo, hi);
        for (int64_t t = lo; t < hi; ++t) {
            const bool is_gate = t >= (int64_t) L.n * L.n_ff;
            const int64_t u = is_gate ? t - (int64_t) L.n * L.n_ff : t;
            const int e = (int) (u / L.n_ff), r = (int) (u % L.n_ff);
            const bellpool_type & ty = is_gate ? L.gate : L.up;
            const void * q = is_gate ? (ty.quant ? (const void *) qx_gate.data() : (const void *) L.x) : (ty.quant ? (const void *) qx_up.data() : (const void *) L.x);
            const char * wr = (is_gate ? L.w_gate[e] : L.w_up[e]) + (size_t) r * ty.w_row;
            float s = 0.0f;
            ty.dot(L.n_embd, &s, 0, wr, 0, q, 0, 1);
            (is_gate ? hg : hu)[(size_t) e * L.n_ff + r] = s;
        }
    }
    void swiglu_quant() {   // caller only
        const bellpool_layer & L = *job;
        for (int e = 0; e < L.n; ++e) {
            float * u = hu.data() + (size_t) e * L.n_ff;
            const float * g = hg.data() + (size_t) e * L.n_ff;
            for (int i = 0; i < L.n_ff; ++i) { u[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i]; }
            if (L.down.quant) { L.down.quant(u, qh.data() + (size_t) e * L.down.q_row, L.n_ff); }
        }
    }
    void down(int w) {
        const bellpool_layer & L = *job;
        int64_t lo, hi;
        bellpool_split((int64_t) L.n * L.n_embd, n_workers, w, lo, hi);
        for (int64_t t = lo; t < hi; ++t) {
            const int e = (int) (t / L.n_embd), r = (int) (t % L.n_embd);
            const void * q = L.down.quant ? (const void *) (qh.data() + (size_t) e * L.down.q_row) : (const void *) (hu.data() + (size_t) e * L.n_ff);
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
            wait_phase(1, 1);
            down(w);
            phase_done[2].fetch_add(1, std::memory_order_acq_rel);
            idle_from = std::chrono::steady_clock::now();
        }
    }

    // one layer: returns when every out row is written
    void run(const bellpool_layer & L) {
        if (L.n <= 0) { return; }
        job = &L;
        if (L.up.quant)   { qx_up.resize(L.up.q_row);     L.up.quant(L.x, qx_up.data(), L.n_embd); }
        if (L.gate.quant) { if (L.gate.quant == L.up.quant && L.gate.q_row == L.up.q_row) { qx_gate = qx_up; } else { qx_gate.resize(L.gate.q_row); L.gate.quant(L.x, qx_gate.data(), L.n_embd); } }
        hu.resize((size_t) L.n * L.n_ff); hg.resize((size_t) L.n * L.n_ff);
        if (L.down.quant) { qh.resize((size_t) L.n * L.down.q_row); }
        for (auto & p : phase_done) { p.store(0, std::memory_order_relaxed); }
        {
            std::lock_guard<std::mutex> lk(mtx);
            gen.fetch_add(1, std::memory_order_acq_rel);
        }
        if (sleeping.load() > 0) { cv.notify_all(); }
        up_gate(0);
        phase_done[0].fetch_add(1, std::memory_order_acq_rel);
        wait_phase(0, n_workers);
        swiglu_quant();
        phase_done[1].store(1, std::memory_order_release);
        down(0);
        phase_done[2].fetch_add(1, std::memory_order_acq_rel);
        wait_phase(2, n_workers);
    }
};
