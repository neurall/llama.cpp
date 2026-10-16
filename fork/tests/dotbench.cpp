// CPU dot-product speed of every quant type of the linked ggml-cpu, one thread, weights far larger than the caches (a 96 MB tiled buffer):
// ns per 256 weights, GB/s of weight bytes read per thread, and the single-thread memory read speed of the same buffer for reference.
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int main(int argc, char ** argv) {
    const double mb = argc > 1 ? atof(argv[1]) : 96.0;   // working set per type: 96 = memory-bound, 0.25 = in L2 (the pure compute speed of the kernel)
    ggml_cpu_init();
    const int n = 4096, urows = 256;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> act(n), w((size_t) urows * n), imp(n, 1.0f);
    for (auto & v : act) v = nd(rng);
    for (auto & v : w) v = nd(rng) * 0.05f;
    const ggml_type types[] = { GGML_TYPE_Q2_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL,
                                GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_MXFP4 };
    // memory read speed: sum a 96 MB buffer
    { std::vector<uint64_t> b(12 << 20, 1); uint64_t s = 0; auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 6; ++r) for (size_t i = 0; i < b.size(); ++i) s += b[i] * (i | 1);
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      printf("single-thread memory read: %.1f GB/s (%llu)\n", 6.0 * b.size() * 8 / sec / 1e9, (unsigned long long) (s & 1)); }
    printf("%-9s %8s %10s %8s\n", "type", "bpw", "ns/256w", "GB/s");
    for (ggml_type t : types) {
        const auto * trc = ggml_get_type_traits_cpu(t);
        const auto * q8  = ggml_get_type_traits_cpu(trc->vec_dot_type);
        if (!trc->vec_dot) { continue; }
        ggml_cpu_tune_vec_dot(t);   // pick the faster form of this type's dot product (GGML_CPU_KERNEL_TUNE=log prints it, =0 keeps the default form)
        ggml_quantize_init(t);
        const size_t rb = ggml_row_size(t, n);
        std::vector<uint8_t> uq(rb * urows);
        ggml_quantize_chunk(t, w.data(), uq.data(), 0, urows, n, imp.data());
        const size_t total = (size_t) (mb * (1 << 20)), rows = std::max<size_t>(1, total / rb);
        std::vector<uint8_t> big(rows * rb);
        for (size_t r = 0; r < rows; ++r) memcpy(big.data() + r * rb, uq.data() + (r % urows) * rb, rb);
        std::vector<uint8_t> a8(ggml_row_size(trc->vec_dot_type, n));
        q8->from_float(act.data(), a8.data(), n);
        float sink = 0;
        for (size_t r = 0; r < rows; ++r) { float s; trc->vec_dot(n, &s, 0, big.data() + r * rb, 0, a8.data(), 0, 1); sink += s; }   // warm
        const int reps = mb < 8 ? 3000 : 3; auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) for (size_t r = 0; r < rows; ++r) { float s; trc->vec_dot(n, &s, 0, big.data() + r * rb, 0, a8.data(), 0, 1); sink += s; }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double blocks = (double) reps * rows * (n / 256.0);
        printf("%-9s %8.2f %10.2f %8.2f%s\n", ggml_type_name(t), rb * 8.0 / n, sec * 1e9 / blocks, (double) reps * rows * rb / sec / 1e9, sink == 1.234f ? "!" : "");
    }
}
