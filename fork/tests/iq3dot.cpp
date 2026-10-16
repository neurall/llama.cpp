// micro test: IQ3_S / IQ3_XXS vec_dot of the linked ggml-cpu against a double-precision reference (dequantize + dot), and its speed. Built twice
// (with and without -DGGML_IQ3_FAST_AVX2) and run: compare error and ns per block.
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main() {
    ggml_cpu_init();
    const int n = 4096;                 // elements per row
    const int rows = 2048;              // weight rows
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> act(n); for (auto & v : act) v = nd(rng);
    std::vector<float> w((size_t) rows * n); for (auto & v : w) v = nd(rng) * 0.05f;
    const ggml_type types[2] = { GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS };
    for (ggml_type t : types) {
        const auto * tr  = ggml_get_type_traits(t);
        const auto * trc = ggml_get_type_traits_cpu(t);
        const auto * tq8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        ggml_quantize_init(t);
        const size_t rb = ggml_row_size(t, n);
        std::vector<uint8_t> q(rb * rows);
        std::vector<float> imp(n, 1.0f);
        ggml_quantize_chunk(t, w.data(), q.data(), 0, rows, n, imp.data());
        std::vector<uint8_t> a8(ggml_row_size(GGML_TYPE_Q8_K, n));
        tq8->from_float(act.data(), a8.data(), n);
        std::vector<float> deq(n);
        double maxrel = 0, sumerr = 0; std::vector<float> res(rows);
        for (int r = 0; r < rows; ++r) {
            float s = 0; trc->vec_dot(n, &s, 0, q.data() + r * rb, 0, a8.data(), 0, 1); res[r] = s;
            tr->to_float(q.data() + r * rb, deq.data(), n);
            double ref = 0; for (int i = 0; i < n; ++i) ref += (double) deq[i] * act[i];
            const double e = std::fabs(s - ref), rel = e / (std::fabs(ref) + 1e-3);
            sumerr += e; if (rel > maxrel) maxrel = rel;
        }
        // speed: all rows, repeated
        const int reps = 200; float sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) for (int r = 0; r < rows; ++r) { float s; trc->vec_dot(n, &s, 0, q.data() + r * rb, 0, a8.data(), 0, 1); sink += s; }
        auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double) reps * rows * (n / 256));
        double cs = 0; for (float v : res) cs += v;
        printf("%-8s mean|err| %.3e  max rel %.3e  checksum %.6f  %.2f ns/block%s\n", ggml_type_name(t), sumerr / rows, maxrel, cs, ns, sink == 12345.f ? "!" : "");
    }
    return 0;
}
