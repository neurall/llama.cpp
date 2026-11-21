// fork: the decode bell's CPU experts on a pool of its own.
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

#include <algorithm>
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
    bellpool_dot2_t down2   = nullptr;            // two neighbouring down rows of an expert against its hidden vector in one pass
    int n = 0;                                    // missing experts
    const char * w_up[16], * w_gate[16], * w_down[16];   // expert matrices: row r at w + r * w_row
    float * out[16];                              // n_embd floats each
    const float * x = nullptr;                    // n_embd
    float limit = 0.0f;                           // > 0: ggml_swiglu_clamp's (GLM 5.3 Flash, DeepSeek 4.1): min(gate, limit), clamp(up, -limit, limit) first
    int   act   = 0;                              // 0: SwiGLU (limit above); 1: Kimi K3/K3.1's SITU, scales situ[0..3] (1/beta, beta, 1/lb, lb), situ_u: the up side's tanh
    float situ[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    bool  situ_u  = false;
};

// the rows [lo, hi) of a phase of `total` rows split over n workers, worker w: contiguous, sizes differ by at most one
static inline void bellpool_split(int64_t total, int n, int w, int64_t & lo, int64_t & hi) {
    const int64_t base = total / n, extra = total % n;
    lo = (int64_t) w * base + std::min<int64_t>(w, extra);
    hi = lo + base + (w < extra ? 1 : 0);
}

struct bellpool_pf {   // a layer's predicted misses: each worker reads its own rows of them into its caches before the layer's bell rings
    int n = 0, n_embd = 0, n_ff = 0;
    size_t up_row = 0, gate_row = 0, down_row = 0;   // bytes of one weight row
    const char * w_up[16], * w_gate[16], * w_down[16];
};

struct bellpool {
    int n_workers = 1;   // worker 0 is the caller
    int64_t spin_us = 20000;
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> arrived{0};
    std::atomic<int> phase_done[3];   // [1]: a prefetch's workers done
    std::atomic<bool> stop{false};
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> sleeping{0};
    std::atomic<int> kind{0};                  // the job of the last gen: 0 run (a layer's rows), 1 prefetch
    const bellpool_layer * job = nullptr;
    const bellpool_pf *    pf  = nullptr;
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
        bellpool_split(L.n_ff, n_workers, w, lo, hi);   // the same rows of every missing expert: a row's worker never depends on the other misses (prefetch)
        const bool two = L.upgate2 != nullptr && qu == qg;
        for (int e = 0; e < L.n; ++e) for (int64_t r = lo; r < hi; ++r) {
            const int64_t t = (int64_t) e * L.n_ff + r;
            float u = 0.0f, g = 0.0f;
            if (two) {
                L.upgate2(L.n_embd, &u, &g, L.w_up[e] + (size_t) r * L.up.w_row, L.w_gate[e] + (size_t) r * L.gate.w_row, qu);
            } else {
                L.up.dot(L.n_embd, &u, 0, L.w_up[e] + (size_t) r * L.up.w_row, 0, qu, 0, 1);
                L.gate.dot(L.n_embd, &g, 0, L.w_gate[e] + (size_t) r * L.gate.w_row, 0, qg, 0, 1);
            }
            if (L.act == 1) {   // SITU op by op as the CPU graph: scale, tanh, scale; sigmoid; mul; (up: scale, tanh, scale); mul
                const float a  = std::tanh(g * L.situ[0]) * L.situ[1];
                const float sg = 1.0f / (1.0f + std::exp(-g));
                const float uu = L.situ_u ? std::tanh(u * L.situ[2]) * L.situ[3] : u;
                hu[(size_t) t] = (a * sg) * uu;
                continue;
            }
            if (L.limit > 0.0f) { g = std::min(g, L.limit); u = std::clamp(u, -L.limit, L.limit); }   // as ggml_compute_forward_swiglu_clamp_f32
            hu[(size_t) t] = g / (1.0f + std::exp(-g)) * u;
        }
    }
    void down(int w) {   // each worker quantizes the hidden vectors of the experts its rows touch, into its own buffer (n_ff values each: cheap)
        const bellpool_layer & L = *job;
        int64_t lo, hi;
        bellpool_split(L.n_embd, n_workers, w, lo, hi);   // as up_gate: the same rows of every missing expert
        if (lo >= hi) { return; }
        uint8_t * qb = L.down.quant ? qh[w].data() : nullptr;
        if (L.down.quant) {
            for (int e = 0; e < L.n; ++e) { L.down.quant(hu.data() + (size_t) e * L.n_ff, qb + (size_t) e * L.down.q_row, L.n_ff); }
        }
        for (int e = 0; e < L.n; ++e) {
            const void * q = L.down.quant ? (const void *) (qb + (size_t) e * L.down.q_row) : (const void *) (hu.data() + (size_t) e * L.n_ff);
            for (int64_t r = lo; r < hi; ) {
                const char * wr = L.w_down[e] + (size_t) r * L.down.w_row;
                if (L.down2 != nullptr && r + 1 < hi) {   // the next row is this worker's
                    L.down2(L.n_ff, &L.out[e][r], &L.out[e][r + 1], wr, wr + L.down.w_row, q);
                    r += 2;
                } else {
                    float s = 0.0f;
                    L.down.dot(L.n_ff, &s, 0, wr, 0, q, 0, 1);
                    L.out[e][r] = s;
                    r += 1;
                }
            }
        }
    }

    // a prefetch: worker w's rows of each predicted miss, the rows up_gate and down give it, into its caches (a run after it reads them from there)
    static void pf_span(const char * p, size_t bytes) { for (size_t o = 0; o < bytes; o += 64) { __builtin_prefetch(p + o, 0, 3); } }
    void prefetch_share(int w) {
        const bellpool_pf & P = *pf;
        int64_t lo, hi;
        bellpool_split(P.n_ff, n_workers, w, lo, hi);
        for (int e = 0; e < P.n; ++e) {
            pf_span(P.w_up[e]   + (size_t) lo * P.up_row,   (size_t) (hi - lo) * P.up_row);
            pf_span(P.w_gate[e] + (size_t) lo * P.gate_row, (size_t) (hi - lo) * P.gate_row);
        }
        bellpool_split(P.n_embd, n_workers, w, lo, hi);
        for (int e = 0; e < P.n; ++e) { pf_span(P.w_down[e] + (size_t) lo * P.down_row, (size_t) (hi - lo) * P.down_row); }
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
            if (kind.load(std::memory_order_acquire) == 1) {
                prefetch_share(w);
                phase_done[1].fetch_add(1, std::memory_order_acq_rel);
                idle_from = std::chrono::steady_clock::now();
                continue;
            }
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

    // a layer's predicted misses into the workers' caches: returns when every worker has issued its share (the reads land behind it). Called between
    // runs from the thread that calls run, so no worker is inside a run then
    void prefetch(const bellpool_pf & P) {
        if (P.n <= 0) { return; }
        pf = &P;
        phase_done[1].store(0, std::memory_order_relaxed);
        kind.store(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mtx);
            gen.fetch_add(1, std::memory_order_acq_rel);
        }
        if (sleeping.load() > 0) { cv.notify_all(); }
        prefetch_share(0);
        phase_done[1].fetch_add(1, std::memory_order_acq_rel);
        wait_phase(1, n_workers);
    }

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
        kind.store(0, std::memory_order_relaxed);
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
