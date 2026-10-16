// aggregate dot-product throughput of a type with T threads on separate rows of a 96 MB per thread buffer: where does it hit the memory wall?
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <atomic>
int main() {
    ggml_cpu_init();
    const int n = 4096, urows = 256;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> act(n), w((size_t) urows * n), imp(n, 1.0f);
    for (auto & v : act) v = nd(rng);
    for (auto & v : w) v = nd(rng) * 0.05f;
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS };
    const int ths[] = { 1, 2, 4, 6, 8, 12 };
    printf("%-9s", "GB/s"); for (int t : ths) printf(" %6dT", t); printf("\n");
    for (ggml_type t : types) {
        const auto * trc = ggml_get_type_traits_cpu(t); const auto * q8 = ggml_get_type_traits_cpu(trc->vec_dot_type);
        ggml_quantize_init(t);
        const size_t rb = ggml_row_size(t, n);
        std::vector<uint8_t> uq(rb * urows); ggml_quantize_chunk(t, w.data(), uq.data(), 0, urows, n, imp.data());
        std::vector<uint8_t> a8(ggml_row_size(trc->vec_dot_type, n)); q8->from_float(act.data(), a8.data(), n);
        printf("%-9s", ggml_type_name(t));
        for (int T : ths) {
            const size_t rows = ((size_t) 96 << 20) / rb;
            std::vector<std::vector<uint8_t>> bufs(T);
            for (int i = 0; i < T; ++i) { bufs[i].resize(rows * rb); for (size_t r = 0; r < rows; ++r) memcpy(bufs[i].data() + r * rb, uq.data() + (r % urows) * rb, rb); }
            std::atomic<int> go{0}; std::vector<double> secs(T); std::vector<std::thread> th;
            for (int i = 0; i < T; ++i) th.emplace_back([&, i] {
                while (!go.load()) {}
                float sink = 0; auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < 3; ++k) for (size_t r = 0; r < rows; ++r) { float s; trc->vec_dot(n, &s, 0, bufs[i].data() + r * rb, 0, a8.data(), 0, 1); sink += s; }
                secs[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); if (sink == 1.2345f) printf("!");
            });
            go = 1; for (auto & x : th) x.join();
            double mx = 0; for (double s : secs) if (s > mx) mx = s;
            printf(" %7.1f", (double) T * 3 * rows * rb / mx / 1e9);
        }
        printf("\n"); fflush(stdout);
    }
}
