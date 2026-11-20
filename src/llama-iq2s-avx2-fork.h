// IQ2_S x Q8_K dot product on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h), a drop-in for ggml's vec_dot.
// Bit-identical to ggml_vec_dot_iq2_s_q8_K's AVX2 branch (ggml/src/ggml-cpu/arch/x86/quants.c): every int32 lane holds the same sum
// (int adds in any order, no saturation: |maddubs pair| <= 2*43*128) and the floats go in ggml's order (per block one fmadd of
// fp16(x.d) * y.d, then 0.125f * hsum_float_8). Fewer instructions, loads and ALU ops than ggml (clang -O2: 56 vs 72 per 64 values),
// about as many shuffle uops: grid indices by one punpcklbw with a qh spread table, signs broadcast and expanded in place, scales one table row.
// Test: tests/test-iq2s-avx2-fork.cpp (against the CPU backend's own vec_dot).
#pragma once

#if defined(__AVX2__)

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if !defined(GGML_COMMON_DECL)
#define GGML_COMMON_DECL_CPP
#endif
#if !defined(GGML_COMMON_IMPL)
#define GGML_COMMON_IMPL_CPP
#endif
#include "../ggml/src/ggml-common.h"   // iq2s_grid is static const (one copy per TU), so every function here is static too: no ODR clash

#ifndef LLAMA_IQ2S_PREFETCH
#define LLAMA_IQ2S_PREFETCH 2048   // bytes ahead in the weight row, 0 = off; not measured here
#endif

namespace llama_iq2s_fork {

struct alignas(64) tables_t {
    uint64_t hi0[256];      // qh byte -> byte k (k < 4) = its bits 2k, 2k+1 = bits 8-9 of grid index k
    uint64_t hi1[256];      // the same in bytes 4-7, for the odd half of a pair (read 16 bytes wide: sc follows, so it stays inside)
    int16_t  sc[256][16];   // scale byte b -> the madd factors of its half: lanes 0-7 = 2*(b&15)+1, lanes 8-15 = 2*(b>>4)+1 (ggml: scales16 + get_scale_shuffle_k4)
    constexpr tables_t() : hi0{}, hi1{}, sc{} {
        for (int b = 0; b < 256; ++b) {
            for (int l = 0; l < 16; ++l) { sc[b][l] = (int16_t) (2 * (l < 8 ? (b & 15) : (b >> 4)) + 1); }
            uint64_t h = 0;
            for (int k = 0; k < 4; ++k) { h |= (uint64_t) ((b >> (2 * k)) & 3) << (8 * k); }
            hi0[b] = h;
            hi1[b] = h << 32;
        }
    }
};
inline constexpr tables_t tables{};   // constant-initialized: no static constructor
static_assert(tables.hi0[0xE4] == 0x03020100ull && tables.hi1[0x1B] == 0x0001020300000000ull &&
              tables.sc[0x2F][7] == 31 && tables.sc[0x2F][8] == 5, "iq2s tables");

// the four grid rows of one 32-value half from its four 16-bit indices packed in ix (as _mm256_set_epi64x, but each 128-bit half
// as a load plus a load-high: clang builds the set from four loads and three shuffles)
static inline __m128i grid2(uint64_t a, uint64_t b) {
    return _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i *) &iq2s_grid[a])), (const double *) &iq2s_grid[b]));
}
static inline __m256i grid4(uint64_t ix) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(grid2(ix & 0xffff, (ix >> 16) & 0xffff)), grid2((ix >> 32) & 0xffff, ix >> 48), 1);
}

// halves 2j and 2j+1 of block x against the q8 values q0, q1 of those halves: madd lanes into acc0, acc1
static inline void pair(const block_iq2_s * x, int j, __m256i q0, __m256i q1, __m256i & acc0, __m256i & acc1) {
    const __m256i sel = _mm256_set1_epi64x(0x8040201008040201ll);
    const __m256i bias = _mm256_set1_epi8(0x7F);
    const __m256i spread0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i spread1 = _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);

    // grid index k of a half = qs byte k | qh bits 2k,2k+1 << 8: one punpcklbw of the pair's 8 qs bytes with the spread qh bits
    const __m128i lo = _mm_loadl_epi64((const __m128i *) (x->qs + 8 * j));
    const __m128i hi = _mm_or_si128(_mm_loadl_epi64((const __m128i *) &tables.hi0[x->qh[2 * j]]),
                                    _mm_loadu_si128((const __m128i *) &tables.hi1[x->qh[2 * j + 1]]));   // a vpor from memory
    const __m128i ix = _mm_unpacklo_epi8(lo, hi);
    const __m256i g0 = grid4((uint64_t) _mm_cvtsi128_si64(ix));
    const __m256i g1 = grid4((uint64_t) _mm_extract_epi64(ix, 1));

    // the pair's 8 sign bytes broadcast, one pshufb per half; (byte & bit) +sat 0x7F is negative iff the bit is set and never 0, so
    // one psignb negates q8 value v where bit v%8 of byte v/8 is set (the same bytes as ggml's (q8 ^ m) - m, -128 included)
    uint64_t sb;
    memcpy(&sb, x->qs + QK_K / 8 + 8 * j, 8);
    const __m256i sv = _mm256_set1_epi64x((long long) sb);
    const __m256i s0 = _mm256_adds_epu8(_mm256_and_si256(_mm256_shuffle_epi8(sv, spread0), sel), bias);
    const __m256i s1 = _mm256_adds_epu8(_mm256_and_si256(_mm256_shuffle_epi8(sv, spread1), sel), bias);

    const __m256i sc0 = _mm256_load_si256((const __m256i *) tables.sc[x->scales[2 * j]]);
    const __m256i sc1 = _mm256_load_si256((const __m256i *) tables.sc[x->scales[2 * j + 1]]);
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(g0, _mm256_sign_epi8(q0, s0)), sc0));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(g1, _mm256_sign_epi8(q1, s1)), sc1));
}

// pair j of the q8 block q8 against row xa, and xb too when two: each q8 vector loaded once
template <bool two>
static inline void pairs(int j, const block_iq2_s * xa, const block_iq2_s * xb, const int8_t * q8, __m256i * acc) {
    const __m256i q0 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j));
    const __m256i q1 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j + 32));
    pair(xa, j, q0, q1, acc[0], acc[1]);
    if constexpr (two) { pair(xb, j, q0, q1, acc[2], acc[3]); }
}

static inline void prefetch(const block_iq2_s * x) {
    if (LLAMA_IQ2S_PREFETCH > 0) {   // two lines per 82-byte block, so every line of the row is asked for
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ2S_PREFETCH), _MM_HINT_T0);
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ2S_PREFETCH + 64), _MM_HINT_T0);
    }
}

// ggml's step per block: d = fp16(x.d) * y.d, acc = fmadd(d, float(int32 lanes), acc)
static inline __m256 fma_block(const block_iq2_s * x, const block_q8_K * y, __m256i s0, __m256i s1, __m256 acc) {
    const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x->d))) * y->d;
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

} // namespace llama_iq2s_fork

// ggml_vec_dot_t: s[0] = row vx (n IQ2_S values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq2_s_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq2s_fork;
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_iq2_s * x = (const block_iq2_s *) vx;
    const block_q8_K  * y = (const block_q8_K *) vy;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(x + i);
        __m256i sum[2] = { _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < 4; ++j) { f::pairs<false>(j, x + i, nullptr, y[i].qs, sum); }   // kept rolled: fully unrolled, clang hoists all four pairs and spills
        acc = f::fma_block(x + i, y + i, sum[0], sum[1], acc);
    }
    *s = 0.125f * f::hsum(acc);
}

// two rows (up and gate) against the same Q8_K input in one pass, each q8 vector loaded once: bit-identical to two single calls
static inline void llama_dot2_iq2_s_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq2s_fork;
    assert(n % QK_K == 0);
    const block_iq2_s * xa = (const block_iq2_s *) vxa;
    const block_iq2_s * xb = (const block_iq2_s *) vxb;
    const block_q8_K  * y  = (const block_q8_K *) vy;
    __m256 acc_a = _mm256_setzero_ps(), acc_b = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(xa + i);
        f::prefetch(xb + i);
        __m256i sum[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < 4; ++j) { f::pairs<true>(j, xa + i, xb + i, y[i].qs, sum); }   // kept rolled: fully unrolled, clang hoists all four pairs and spills
        acc_a = f::fma_block(xa + i, y + i, sum[0], sum[1], acc_a);
        acc_b = f::fma_block(xb + i, y + i, sum[2], sum[3], acc_b);
    }
    *s_a = 0.125f * f::hsum(acc_a);
    *s_b = 0.125f * f::hsum(acc_b);
}

#endif // __AVX2__
