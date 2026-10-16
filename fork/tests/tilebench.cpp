// cache blocking of the MUL_MAT_ID loop (ggml-cpu.c, blck_0 weight rows x blck_1 tokens per tile): weights from a 96 MB buffer, T activation vectors per tile of R weight rows.
// Prints ns per (weight row x token) pair and the weight GB/s that implies; T = 1 is decode (no reuse possible), T > 1 is a batch (prompt, parallel requests).
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    ggml_cpu_init();
    const int n = 4096, urows = 256, Tmax = 32;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) urows * n), imp(n, 1.0f); for (auto & v : w) v = nd(rng) * 0.05f;
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL };
    const int Rs[] = { 4, 8, 16, 32 }, Ts[] = { 1, 4, 16, 32 };
    for (ggml_type t : types) {
        const auto * trc = ggml_get_type_traits_cpu(t); const auto * q8 = ggml_get_type_traits_cpu(trc->vec_dot_type);
        ggml_quantize_init(t);
        const size_t rb = ggml_row_size(t, n), ab = ggml_row_size(trc->vec_dot_type, n);
        std::vector<uint8_t> uq(rb * urows); ggml_quantize_chunk(t, w.data(), uq.data(), 0, urows, n, imp.data());
        const size_t rows = ((size_t) 96 << 20) / rb;
        std::vector<uint8_t> big(rows * rb); for (size_t r = 0; r < rows; ++r) memcpy(big.data() + r * rb, uq.data() + (r % urows) * rb, rb);
        std::vector<uint8_t> acts(ab * Tmax); for (int i = 0; i < Tmax; ++i) { std::vector<float> a(n); for (auto & v : a) v = nd(rng); q8->from_float(a.data(), acts.data() + i * ab, n); }
        printf("%s: ns per row x token (weight GB/s per token-pass), rows = tile weight rows\n%8s", ggml_type_name(t), "T \\ R");
        for (int R : Rs) printf(" %14d", R); printf("\n");
        for (int T : Ts) {
            printf("%8d", T);
            for (int R : Rs) {
                float sink = 0; auto t0 = std::chrono::steady_clock::now();
                for (size_t r0 = 0; r0 + R <= rows; r0 += R)
                    for (int tk = 0; tk < T; ++tk) for (int r = 0; r < R; ++r) { float s; trc->vec_dot(n, &s, 0, big.data() + (r0 + r) * rb, 0, acts.data() + tk * ab, 0, 1); sink += s; }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double pairs = (double) (rows / R * R) * T;
                printf(" %7.1f (%5.1f)", sec * 1e9 / pairs, pairs * rb / sec / 1e9 / T * T / T * 1.0 / 1.0);
                if (sink == 1.2f) printf("!");
            }
            printf("\n"); fflush(stdout);
        }
    }
}
