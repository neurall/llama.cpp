// l3pref: does prefetching a MoE expert into the L3 of the CCX that will compute it speed up the CPU expert
// matmul? Real ggml Q2_K dot kernel, GLM-5.3-Flash expert shape (gate/up 2048x4096, down 4096x2048, Q2_K ~8.26 MB).
// Compute threads on CPUs 0,1,2 (CCX0) and 4,5,6 (CCX1); rows split so CCX0 computes the first half of every
// matrix, CCX1 the second. Prefetch threads on CPU 3 (CCX0) and 7 (CCX1) read their CCX's half beforehand.
// Modes: cold (from DDR), same-CCX prefetch, wrong-CCX prefetch (control).
// Build: g++ -O2 -std=c++17 -I ggml/include l3pref.cpp -L build-link/bin -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$PWD/build-link/bin
#include "ggml.h"
#include "ggml-cpu.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

struct matrix { const uint8_t * data; int rows, cols; size_t row_bytes; };

int main() {
    const int n_embd = 4096, n_ff = 2048, n_exp = 64; // 64 experts ~530 MB: far bigger than the 32 MB of L3
    const auto * tt = ggml_get_type_traits_cpu(GGML_TYPE_Q2_K);
    const size_t rb_up = ggml_row_size(GGML_TYPE_Q2_K, n_embd), rb_dn = ggml_row_size(GGML_TYPE_Q2_K, n_ff);
    const size_t exp_bytes = 2 * n_ff * rb_up + n_embd * rb_dn;
    printf("expert %.2f MB, pool %.0f MB\n", exp_bytes / 1e6, n_exp * exp_bytes / 1e6);

    // quantize one row of each width, replicate it through the pool (values don't change the kernel's speed)
    std::vector<float> f(n_embd);
    for (int i = 0; i < n_embd; ++i) f[i] = (float) ((i * 37) % 101 - 50) / 50.0f;
    std::vector<uint8_t> row_up(rb_up), row_dn(rb_dn);
    ggml_quantize_chunk(GGML_TYPE_Q2_K, f.data(), row_up.data(), 0, 1, n_embd, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q2_K, f.data(), row_dn.data(), 0, 1, n_ff, nullptr);
    std::vector<uint8_t> pool(n_exp * exp_bytes);
    for (int e = 0; e < n_exp; ++e) {
        uint8_t * p = pool.data() + e * exp_bytes;
        for (int r = 0; r < 2 * n_ff; ++r) memcpy(p + r * rb_up, row_up.data(), rb_up);
        uint8_t * d = p + 2 * n_ff * rb_up;
        for (int r = 0; r < n_embd; ++r) memcpy(d + r * rb_dn, row_dn.data(), rb_dn);
    }
    // activations, q8_K
    const auto * t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::vector<uint8_t> y_up(ggml_row_size(GGML_TYPE_Q8_K, n_embd)), y_dn(ggml_row_size(GGML_TYPE_Q8_K, n_ff));
    t8->from_float(f.data(), y_up.data(), n_embd);
    t8->from_float(f.data(), y_dn.data(), n_ff);

    auto mats = [&](int e) {
        const uint8_t * p = pool.data() + e * exp_bytes;
        return std::vector<matrix>{ { p, n_ff, n_embd, rb_up }, { p + n_ff * rb_up, n_ff, n_embd, rb_up },
                                    { p + 2 * n_ff * rb_up, n_embd, n_ff, rb_dn } };
    };

    const int comp_cpu[6] = { 0, 1, 2, 4, 5, 6 };
    std::atomic<int> go{0}, done{0}, stop{0};
    std::atomic<int> cur_e{0};
    std::vector<double> sink(6);
    std::vector<std::thread> th;
    for (int t = 0; t < 6; ++t) {
        th.emplace_back([&, t] {
            pin(comp_cpu[t]);
            const int ccx = t / 3, lt = t % 3;
            int seen = 0;
            while (!stop) {
                while (go.load() == seen && !stop) {}
                if (stop) break;
                seen = go.load();
                float s = 0, acc = 0;
                for (const auto & m : mats(cur_e)) {
                    // CCX half of the rows, then this thread's third of that half
                    const int h0 = ccx * m.rows / 2, h1 = (ccx + 1) * m.rows / 2;
                    const int r0 = h0 + lt * (h1 - h0) / 3, r1 = h0 + (lt + 1) * (h1 - h0) / 3;
                    const void * y = m.cols == n_embd ? (const void *) y_up.data() : (const void *) y_dn.data();
                    for (int r = r0; r < r1; ++r) {
                        tt->vec_dot(m.cols, &s, 0, m.data + r * m.row_bytes, 0, y, 0, 1);
                        acc += s;
                    }
                }
                sink[t] += acc;
                done++;
            }
        });
    }
    auto prefetch = [&](int e, int ccx, int cpu) {
        pin(cpu);
        volatile uint64_t x = 0;
        for (const auto & m : mats(e)) {
            const uint8_t * a = m.data + (size_t) (ccx * m.rows / 2) * m.row_bytes;
            const uint8_t * b = m.data + (size_t) ((ccx + 1) * m.rows / 2) * m.row_bytes;
            for (const uint8_t * p = a; p < b; p += 64) x = x + *(const volatile uint64_t *) p;
        }
    };
    auto run = [&](const char * name, int mode) {
        double tc = 0, tp = 0; const int N = 48;
        for (int i = 0; i < N; ++i) {
            const int e = (i * 7 + mode * 17) % n_exp;  // a fresh expert each time: cold unless prefetched
            cur_e = e;
            if (mode > 0) {
                const double t0 = now_us();
                const int c0 = mode == 1 ? 3 : 7, c1 = mode == 1 ? 7 : 3; // mode 2: each half into the other CCX
                std::thread p0(prefetch, e, 0, c0), p1(prefetch, e, 1, c1);
                p0.join(); p1.join();
                tp += now_us() - t0;
            }
            done = 0;
            const double t0 = now_us();
            go++;
            while (done.load() < 6) {}
            tc += now_us() - t0;
        }
        printf("%-26s compute %6.0f us/expert = %5.1f GB/s", name, tc / N, exp_bytes / (tc / N) / 1e3);
        if (mode > 0) printf("   (prefetch %4.0f us = %5.1f GB/s)", tp / N, exp_bytes / (tp / N) / 1e3);
        printf("\n");
    };
    run("cold (DDR)", 0);
    run("prefetched, same CCX", 1);
    run("prefetched, wrong CCX", 2);
    run("cold (DDR) again", 0);
    stop = 1; go++;
    for (auto & t : th) t.join();
    return 0;
}
