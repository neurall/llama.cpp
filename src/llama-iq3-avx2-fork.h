// IQ3_XXS and IQ3_S x Q8_K dot products on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h), drop-ins for ggml's vec_dot.
// Bit-identical to ggml_vec_dot_iq3_xxs_q8_K / ggml_vec_dot_iq3_s_q8_K's AVX2 branches (the _fast ones in ggml/src/ggml-cpu/arch/x86/quants.c):
// every int32 lane holds the same sum (int adds in any order, no saturation: |maddubs pair| <= 2*62*128 for IQ3_XXS, 2*15*128 for IQ3_S)
// and the floats go in ggml's order (per block one fmadd of fp16(x.d) * y.d, then 0.25f * hsum_float_8 for IQ3_XXS, hsum_float_8 for IQ3_S).
// Per 64 values the signs of both 32-value halves come from one broadcast (IQ3_XXS: one srlv + parity pshufb for all eight 7-bit fields,
// ggml does it per half), each half's sign is one psignb, the 2s+1 scale is a table row read by vpmaddwd itself, the IQ3_S grid indices
// are one punpck{l,h}bw with a qh spread table instead of a shift, mask and or per index. clang 18 -O2, per 256-value block, mine vs ggml's
// haswell .so: IQ3_XXS 293 vs 383 instructions, 154 vs 200 memory operands, 78 vs 102 shuffles; IQ3_S 377 vs 534, 117 vs 216, 90 vs 86.
// One Zen 2 thread (3975WX), 2 x 1300 rows of 2560 (> L2), rows/s: IQ3_XXS ggml 3.26M, single 4.74M, dual 4.94M; IQ3_S 2.95M, 4.22M, 4.30M.
// Tried and slower there: vpgatherdd grids (-27%), broadcast + vpblendd grids (-17%), GPR byte extraction instead of movzx loads (+-3%).
// Reads stay inside the block (no over-read); every function needs AVX2 + FMA + F16C (vcvtph2ps) at run time.
// Test: tests/test-iq3-avx2-fork.cpp (against the CPU backend's own vec_dot).
#pragma once

#if defined(__AVX2__)

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if !defined(GGML_COMMON_DECL)
#define GGML_COMMON_DECL_CPP
#endif
#if !defined(GGML_COMMON_IMPL)
#define GGML_COMMON_IMPL_CPP
#endif
#include "../ggml/src/ggml-common.h"   // iq3xxs_grid, iq3s_grid are static const (one copy per TU), so every function here is static too: no ODR clash

#ifndef LLAMA_IQ3_PREFETCH
#define LLAMA_IQ3_PREFETCH 2048   // bytes ahead in the weight row, 0 = off; not measured here
#endif

namespace llama_iq3_fork {

struct alignas(64) tables_t {
    int16_t  sc[16][16];    // 4-bit scale s -> the 16 madd factors 2s+1 (ggml: _mm256_set1_epi16(2*ls+1))
    uint64_t hi[256];       // IQ3_S qh byte -> byte k = its bit k = bit 8 of grid index k
    constexpr tables_t() : sc{}, hi{} {
        for (int s = 0; s < 16; ++s) { for (int l = 0; l < 16; ++l) { sc[s][l] = (int16_t) (2 * s + 1); } }
        for (int b = 0; b < 256; ++b) {
            uint64_t h = 0;
            for (int k = 0; k < 8; ++k) { h |= (uint64_t) ((b >> k) & 1) << (8 * k); }
            hi[b] = h;
        }
    }
};
inline constexpr tables_t tables{};   // constant-initialized: no static constructor
static_assert(tables.sc[15][15] == 31 && tables.sc[0][0] == 1 && tables.hi[0x81] == 0x0100000000000001ull, "iq3 tables");

// one 32-value half's eight grid words, index k of the half -> int32 lane k (as ggml's _mm256_set_epi32)
template <typename I>
static inline __m256i grid8(const uint32_t * grid, I i) {
    return _mm256_set_epi32((int) grid[i(7)], (int) grid[i(6)], (int) grid[i(5)], (int) grid[i(4)],
                            (int) grid[i(3)], (int) grid[i(2)], (int) grid[i(1)], (int) grid[i(0)]);
}

// the two halves' q8 values signed in place: sv holds sign byte b of half h at byte spread_h[8b..] of each 128-bit lane;
// (byte & bit) +sat 0x7F is negative iff the bit is set and never 0, so one psignb negates where ggml's (q8 ^ m) - m does, -128 included
static inline void sign2(__m256i sv, __m256i spread0, __m256i spread1, __m256i & q0, __m256i & q1) {
    const __m256i sel = _mm256_set1_epi64x(0x8040201008040201ll);
    const __m256i bias = _mm256_set1_epi8(0x7F);
    q0 = _mm256_sign_epi8(q0, _mm256_adds_epu8(_mm256_and_si256(_mm256_shuffle_epi8(sv, spread0), sel), bias));
    q1 = _mm256_sign_epi8(q1, _mm256_adds_epu8(_mm256_and_si256(_mm256_shuffle_epi8(sv, spread1), sel), bias));
}

static inline void madd2(__m256i g0, __m256i g1, __m256i q0, __m256i q1, unsigned s0, unsigned s1, __m256i & acc0, __m256i & acc1) {
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(g0, q0), _mm256_load_si256((const __m256i *) tables.sc[s0])));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(g1, q1), _mm256_load_si256((const __m256i *) tables.sc[s1])));
}

// IQ3_XXS halves 2j and 2j+1: qs[16j..16j+15] grid bytes, then the two u32 words w0, w1 at qs[64+8j] (4 x 7-bit sign field + scale << 28)
static inline void pair(const block_iq3_xxs * x, int j, __m256i q0, __m256i q1, __m256i & acc0, __m256i & acc1) {
    const __m256i shifts = _mm256_setr_epi32(0, 0, 7, 7, 14, 14, 21, 21);
    const __m256i parity = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0));
    const __m256i spread0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
    const __m256i spread1 = _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12);

    uint64_t w;
    memcpy(&w, x->qs + QK_K / 4 + 8 * j, 8);
    // dword lanes (w0, w1) x 4 shifted by 0, 7, 14, 21: lane 2f + h = field f of half h; its byte 0 gets the even-parity bit 7 (ksigns_iq2xs)
    // from a nibble table on v ^ v >> 4 (bit 7 of that byte is 0 and bytes 1-3 are 0, so no mask: those look up parity[0] = 0)
    const __m256i v = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi64x((long long) w), shifts), _mm256_set1_epi32(127));
    const __m256i sv = _mm256_or_si256(v, _mm256_shuffle_epi8(parity, _mm256_xor_si256(v, _mm256_srli_epi32(v, 4))));
    sign2(sv, spread0, spread1, q0, q1);

    const uint8_t * q = x->qs + 16 * j;
    const __m256i g0 = grid8(iq3xxs_grid, [q](int k) { return q[k]; });
    const __m256i g1 = grid8(iq3xxs_grid, [q](int k) { return q[8 + k]; });
    madd2(g0, g1, q0, q1, (unsigned) (uint32_t) w >> 28, (unsigned) (w >> 60), acc0, acc1);
}

// IQ3_S halves 2j and 2j+1: grid index k of half h = qs[16j+8h+k] | bit k of qh[2j+h] << 8, sign bytes signs[8j..8j+7], scale nibbles scales[j]
static inline void pair(const block_iq3_s * x, int j, __m256i q0, __m256i q1, __m256i & acc0, __m256i & acc1) {
    const __m256i spread0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i spread1 = _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);

    uint64_t sb;
    memcpy(&sb, x->signs + 8 * j, 8);
    sign2(_mm256_set1_epi64x((long long) sb), spread0, spread1, q0, q1);

    const __m128i lo = _mm_loadu_si128((const __m128i *) (x->qs + 16 * j));
    const __m128i hi = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i *) &tables.hi[x->qh[2 * j]])),
                                                     (const double *) &tables.hi[x->qh[2 * j + 1]]));
    const __m128i ix0 = _mm_unpacklo_epi8(lo, hi), ix1 = _mm_unpackhi_epi8(lo, hi);   // eight u16 indices per half
    const uint64_t a0 = (uint64_t) _mm_cvtsi128_si64(ix0), b0 = (uint64_t) _mm_extract_epi64(ix0, 1);
    const uint64_t a1 = (uint64_t) _mm_cvtsi128_si64(ix1), b1 = (uint64_t) _mm_extract_epi64(ix1, 1);
    const __m256i g0 = grid8(iq3s_grid, [a0, b0](int k) { return (uint16_t) ((k < 4 ? a0 : b0) >> (16 * (k & 3))); });
    const __m256i g1 = grid8(iq3s_grid, [a1, b1](int k) { return (uint16_t) ((k < 4 ? a1 : b1) >> (16 * (k & 3))); });
    const unsigned s = x->scales[j];
    madd2(g0, g1, q0, q1, s & 15, s >> 4, acc0, acc1);
}

static inline void prefetch(const void * x) {
    if (LLAMA_IQ3_PREFETCH > 0) {   // two lines per 98- or 110-byte block, so every line of the row is asked for (past the row's end is harmless)
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ3_PREFETCH), _MM_HINT_T0);
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ3_PREFETCH + 64), _MM_HINT_T0);
    }
}

// ggml's step per block: d = fp16(x.d) * y.d, acc = fmadd(d, float(int32 lanes), acc)
static inline __m256 fma_block(uint16_t xd, const block_q8_K * y, __m256i s0, __m256i s1, __m256 acc) {
    const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(xd))) * y->d;
    return _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), acc);
}

// ggml's hsum_float_8, same order
static inline float hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// rows xa (and xb when two) against y; the q8 vectors of a pair loaded once. IQ3_XXS ends with 0.25f * hsum, IQ3_S with hsum alone
template <typename B, bool two>
static inline void dot(int n, float * s_a, float * s_b, const B * xa, const B * xb, const block_q8_K * y) {
    assert(n % QK_K == 0);
    __m256 acc_a = _mm256_setzero_ps(), acc_b = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        prefetch(xa + i);
        if constexpr (two) { prefetch(xb + i); }
        __m256i sum[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < 4; ++j) {   // kept rolled, as in llama-iq2s-avx2-fork.h
            const __m256i q0 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 64 * j));
            const __m256i q1 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 64 * j + 32));
            pair(xa + i, j, q0, q1, sum[0], sum[1]);
            if constexpr (two) { pair(xb + i, j, q0, q1, sum[2], sum[3]); }
        }
        acc_a = fma_block(xa[i].d, y + i, sum[0], sum[1], acc_a);
        if constexpr (two) { acc_b = fma_block(xb[i].d, y + i, sum[2], sum[3], acc_b); }
    }
    constexpr bool xxs = std::is_same_v<B, block_iq3_xxs>;
    *s_a = xxs ? 0.25f * hsum(acc_a) : hsum(acc_a);
    if constexpr (two) { *s_b = xxs ? 0.25f * hsum(acc_b) : hsum(acc_b); }
}

} // namespace llama_iq3_fork

// ggml_vec_dot_t: s[0] = row vx (n IQ3_XXS values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq3_xxs_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    llama_iq3_fork::dot<block_iq3_xxs, false>(n, s, nullptr, (const block_iq3_xxs *) vx, nullptr, (const block_q8_K *) vy);
}

// two IQ3_XXS rows (up and gate) against the same Q8_K input in one pass: bit-identical to two single calls
static inline void llama_dot2_iq3_xxs_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    llama_iq3_fork::dot<block_iq3_xxs, true>(n, s_a, s_b, (const block_iq3_xxs *) vxa, (const block_iq3_xxs *) vxb, (const block_q8_K *) vy);
}

// ggml_vec_dot_t: s[0] = row vx (n IQ3_S values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq3_s_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    llama_iq3_fork::dot<block_iq3_s, false>(n, s, nullptr, (const block_iq3_s *) vx, nullptr, (const block_q8_K *) vy);
}

// two IQ3_S rows against the same Q8_K input in one pass: bit-identical to two single calls
static inline void llama_dot2_iq3_s_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    llama_iq3_fork::dot<block_iq3_s, true>(n, s_a, s_b, (const block_iq3_s *) vxa, (const block_iq3_s *) vxb, (const block_q8_K *) vy);
}

#endif // __AVX2__
