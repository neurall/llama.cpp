// ablation of the IQ3_XXS AVX2 dot product: where do the cycles go (lookup / sign / dot)?
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include "iq3xxs_ablation_signs.inc"
static inline float hsum8(__m256 x) { __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)); r = _mm_add_ps(r, _mm_movehl_ps(r, r)); r = _mm_add_ss(r, _mm_movehdup_ps(r)); return _mm_cvtss_f32(r); }
template <int V>
static float dot(int n, const block_iq3_xxs * x, const block_q8_K * y) {
    const uint64_t * signs64 = (const uint64_t *) keven_signs_q2xs;
    uint32_t aux32[2];
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 256; ++i) {
        const float d = 0.001f * y[i].d;
        const uint8_t * q3 = x[i].qs; const uint8_t * gas = x[i].qs + 64; const int8_t * q8 = y[i].qs;
        __m256i s1 = _mm256_setzero_si256(), s2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m256i a1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i a2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            __m256i g1, g2;
            if (V == 3) { g1 = _mm256_set1_epi32(q3[0] * 0x01010101); g2 = _mm256_set1_epi32(q3[8] * 0x01010101); q3 += 16; }
            else if (V == 1) {   // gather by scalar loads into a stack array, one vector load
                alignas(32) uint32_t t[16];
                for (int k = 0; k < 16; ++k) { t[k] = iq3xxs_grid[q3[k]]; }
                q3 += 16; g1 = _mm256_load_si256((const __m256i *) t); g2 = _mm256_load_si256((const __m256i *) (t + 8));
            } else {
                g1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]], iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]); q3 += 8;
                g2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]], iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]); q3 += 8;
            }
            memcpy(aux32, gas, 8); gas += 8;
            __m256i b1 = a1, b2 = a2;
            if (V == 4) {
                static const uint8_t m1[32] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3};
                static const uint8_t m2[32] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
                const __m256i mk1 = _mm256_loadu_si256((const __m256i *) m1), mk2 = _mm256_loadu_si256((const __m256i *) m2);
                auto expand = [&](uint32_t a, __m256i v) {
                    const uint32_t sb = ksigns_iq2xs[a & 127] | (ksigns_iq2xs[(a >> 7) & 127] << 8) | (ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint32_t) ksigns_iq2xs[(a >> 21) & 127] << 24);
                    const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) sb), mk1), mk2), mk2);
                    return _mm256_sub_epi8(_mm256_xor_si256(v, m), m);
                };
                b1 = expand(aux32[0], a1); b2 = expand(aux32[1], a2);
            } else if (V != 2) {
                const __m256i t1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127], signs64[(aux32[0] >> 7) & 127], signs64[aux32[0] & 127]);
                const __m256i t2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127], signs64[(aux32[1] >> 7) & 127], signs64[aux32[1] & 127]);
                b1 = _mm256_sign_epi8(a1, t1); b2 = _mm256_sign_epi8(a2, t2);
            }
            const __m256i d1 = _mm256_maddubs_epi16(g1, b1), d2 = _mm256_maddubs_epi16(g2, b2);
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(d1, _mm256_set1_epi16(2 * (aux32[0] >> 28) + 1)));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(d2, _mm256_set1_epi16(2 * (aux32[1] >> 28) + 1)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(s1, s2)), acc);
    }
    return hsum8(acc);
}
int main() {
    const int n = 4096, rows = 48; std::vector<block_iq3_xxs> x((size_t) rows * n / 256); std::vector<block_q8_K> y(n / 256);
    srand(1); for (auto & b : x) { for (auto & c : b.qs) c = rand(); b.d = 0x3c00; } for (auto & b : y) { for (auto & c : b.qs) c = rand(); b.d = 1; }
    float sink = 0; const char * nm[] = { "original (set_epi32 + sign table)", "lookups through a stack array", "no sign step", "no lookups (constant grid)", "sign by byte table + shuffle" };
    for (int v = 0; v < 5; ++v) {
        double best = 1e30;
        for (int r = 0; r < 9; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 40; ++rep) for (int row = 0; row < rows; ++row) {
                const block_iq3_xxs * xr = &x[(size_t) row * n / 256];
                sink += v == 0 ? dot<0>(n, xr, y.data()) : v == 1 ? dot<1>(n, xr, y.data()) : v == 2 ? dot<2>(n, xr, y.data()) : v == 3 ? dot<3>(n, xr, y.data()) : dot<4>(n, xr, y.data());
            }
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            if (ns < best) best = ns;
        }
        printf("%-36s %6.2f ns/256w\n", nm[v], best / (40.0 * rows * (n / 256)));
    }
    printf("check original %.6f sign-shuffle %.6f\n", dot<0>(n, x.data(), y.data()), dot<4>(n, x.data(), y.data()));
    if (sink == 1234.5f) puts("");
}
