// IQ2_XXS and IQ2_XS x Q8_K dot products on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h), drop-ins for ggml's vec_dot.
// Bit-identical to ggml_vec_dot_iq2_xxs_q8_K / ggml_vec_dot_iq2_xs_q8_K's AVX2 branches (ggml/src/ggml-cpu/arch/x86/quants.c): every int32
// lane holds the same sum (int adds in any order, no saturation: grid bytes are 8, 25 or 43, so |maddubs pair| <= 2*43*128 = 11008, and a
// block's lane <= 4 * 2*11008*31 per accumulator) and the floats go in ggml's order (per block one fmadd of fp16(x.d) * y.d on the sum of
// the two accumulators, then 0.125f * hsum_float_8).
// Per 64 values the eight 7-bit sign fields come from one 16-byte broadcast and two shifts (IQ2_XS) or two dword broadcasts, a blend and
// one srlv (IQ2_XXS), their 8th (parity) bit from one pshufb nibble lookup,
// and each 32-value sign vector from one pshufb spread + and + adds (no sign table loads, no cmpeq/or, no cross-lane shuffles); the scales
// are one table row as the madd's memory operand (IQ2_XXS: 16 rows of 2*l+1, IQ2_XS: the IQ2_S header's nibble-pair rows).
// clang -O2, per 256-value block (ggml's haswell build in brackets): IQ2_XXS 256 instrs, 71 loads, 48 shuffles (367, 127 + 11 spills, 56);
// IQ2_XS 272 instrs, 68 loads, 40 shuffles (249, 66, 95): more scalar index ops, far fewer shuffles. Zen 2, one thread, 64 MiB of rows:
// IQ2_XXS +47% single / +57% dual, IQ2_XS +57% / +64% rows per second at n = 2560.
// Grid indices come from GPR loads on purpose: read as bytes/words of the vector just loaded, clang extracts them with vpextrb/w (2 uops).
// Test: tests/test-iq2x-avx2-fork.cpp (against the CPU backend's own vec_dot).
#pragma once

#if defined(__AVX2__)

#include "llama-iq2s-avx2-fork.h"   // ggml-common.h, the IQ2_S scale rows (same nibble layout as IQ2_XS) and hsum

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef LLAMA_IQ2X_PREFETCH
#define LLAMA_IQ2X_PREFETCH 2048   // bytes ahead in the weight row, 0 = off; not measured here (Strata: 2048 on Zen 3, IQ3_S)
#endif

namespace llama_iq2x_fork {

struct alignas(64) tables_t {
    int16_t sc[16][16];   // IQ2_XXS scale nibble l -> 2*l+1 in all 16 madd lanes (ggml: _mm256_set1_epi16(2*ls+1))
    constexpr tables_t() : sc{} {
        for (int l = 0; l < 16; ++l) { for (int k = 0; k < 16; ++k) { sc[l][k] = (int16_t) (2 * l + 1); } }
    }
};
inline constexpr tables_t tables{};   // constant-initialized: no static constructor
static_assert(tables.sc[0][0] == 1 && tables.sc[15][15] == 31, "iq2x tables");

static inline uint32_t load32(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }

// four grid rows from four indices (as _mm256_set_epi64x, but each 128-bit half as a load plus a load-high)
static inline __m128i grid2(const uint64_t * t, uint32_t a, uint32_t b) {
    return _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i *) &t[a])), (const double *) &t[b]));
}
static inline __m256i grid4(const uint64_t * t, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(grid2(t, a, b)), grid2(t, c, d), 1);
}

// sign byte s (bits 0-6 = a 7-bit field, bit 7 = 0, other bytes of its 16/32-bit lane 0) -> s | even-parity bit 7 = ksigns_iq2xs[s]:
// x ^ (x >> 4) folds the 7 bits into a nibble of the same parity (bit 7 stays 0, so the pshufb never zeroes), one lookup gives 0x80 or 0
static inline __m256i with_parity(__m256i s, __m256i s_hi) {   // s_hi = the field's bits 4-6 at bits 0-2
    const __m256i odd = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                         0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
    return _mm256_or_si256(s, _mm256_shuffle_epi8(odd, _mm256_xor_si256(s, s_hi)));
}

// 32 sign bytes of one 32-value half from the sign bytes in sv: spread (byte k of a group = sign byte of the group) & bit k +sat 0x7F is
// negative iff the bit is set and never 0, so one psignb negates q8 where ggml's keven_signs_q2xs byte is -1 (-128 stays -128 in both)
static inline __m256i signs32(__m256i sv, __m256i spread) {
    const __m256i sel = _mm256_set1_epi64x(0x8040201008040201ll);
    return _mm256_adds_epu8(_mm256_and_si256(_mm256_shuffle_epi8(sv, spread), sel), _mm256_set1_epi8(0x7F));
}

static inline void dot32(__m256i g, __m256i q, __m256i s, const int16_t * sc, __m256i & acc) {
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(g, _mm256_sign_epi8(q, s)), _mm256_load_si256((const __m256i *) sc)));
}

// IQ2_XXS halves 2j (A) and 2j+1 (B): 16 bytes = A's 4 grid bytes, A's word (4 x 7 sign bits, scale in bits 28-31), B's grid bytes, B's word
static inline void pair(const block_iq2_xxs * x, int j, __m256i q0, __m256i q1, __m256i & acc0, __m256i & acc1) {
    const uint8_t * q = (const uint8_t *) (x->qs + 8 * j);
    // both 128-bit lanes = (wA, wA, wB, wB); low lane takes groups 0 and 1 of each word, high lane groups 2 and 3, so the spreads stay in-lane
    const __m256i w = _mm256_blend_epi32(_mm256_set1_epi32((int) load32(q + 4)), _mm256_set1_epi32((int) load32(q + 12)), 0xCC);
    const __m256i f = _mm256_and_si256(_mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 0, 7, 14, 21, 14, 21)), _mm256_set1_epi32(0x7F));
    const __m256i sv = with_parity(f, _mm256_srli_epi32(f, 4));
    const __m256i spread0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4);
    const __m256i spread1 = _mm256_setr_epi8(8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12);
    // grid bytes and scales from two GPR loads: read as bytes of q, clang extracts them from the xmm above with vpextrb (2 uops each)
    uint64_t a, b;
    memcpy(&a, q, 8);
    memcpy(&b, q + 8, 8);
    dot32(grid4(iq2xxs_grid, a & 255, (a >> 8) & 255, (a >> 16) & 255, (a >> 24) & 255), q0, signs32(sv, spread0), tables.sc[a >> 60], acc0);
    dot32(grid4(iq2xxs_grid, b & 255, (b >> 8) & 255, (b >> 16) & 255, (b >> 24) & 255), q1, signs32(sv, spread1), tables.sc[b >> 60], acc1);
}

// IQ2_XS halves 2j (A) and 2j+1 (B): 8 uint16 = 9-bit grid index | 7 sign bits << 9; scale byte per half (nibble per 16 values)
static inline void pair(const block_iq2_xs * x, int j, __m256i q0, __m256i q1, __m256i & acc0, __m256i & acc1) {
    const uint16_t * q = x->qs + 8 * j;
    const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) q));
    const __m256i sv = with_parity(_mm256_srli_epi16(v, 9), _mm256_srli_epi16(v, 13));   // sign byte of index k at byte 2k
    // low lane: indices 0,1 (A) / 4,5 (B); high lane: 2,3 / 6,7
    const __m256i spread0 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6);
    const __m256i spread1 = _mm256_setr_epi8(8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14);
    const auto & sc = llama_iq2s_fork::tables.sc;
    uint64_t a, b;   // grid indices from two GPR loads (read as q[k], clang extracts them from the xmm above with vpextrw, 2 uops each)
    memcpy(&a, q, 8);
    memcpy(&b, q + 4, 8);
    dot32(grid4(iq2xs_grid, a & 511, (a >> 16) & 511, (a >> 32) & 511, (a >> 48) & 511), q0, signs32(sv, spread0), sc[x->scales[2 * j]], acc0);
    dot32(grid4(iq2xs_grid, b & 511, (b >> 16) & 511, (b >> 32) & 511, (b >> 48) & 511), q1, signs32(sv, spread1), sc[x->scales[2 * j + 1]], acc1);
}

// pair j of the q8 block q8 against row xa, and xb too when two: each q8 vector loaded once
template <bool two, typename B>
static inline void pairs(int j, const B * xa, const B * xb, const int8_t * q8, __m256i * acc) {
    const __m256i q0 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j));
    const __m256i q1 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j + 32));
    pair(xa, j, q0, q1, acc[0], acc[1]);
    if constexpr (two) { pair(xb, j, q0, q1, acc[2], acc[3]); }
}

template <typename B>
static inline void prefetch(const B * x) {
    if (LLAMA_IQ2X_PREFETCH > 0) {   // two lines per 66- or 74-byte block, so every line of the row is asked for
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ2X_PREFETCH), _MM_HINT_T0);
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ2X_PREFETCH + 64), _MM_HINT_T0);
    }
}

// ggml's step per block: d = fp16(x.d) * y.d, acc = fmadd(d, float(sumi1 + sumi2), acc)
template <typename B>
static inline __m256 fma_block(const B * x, const block_q8_K * y, __m256i s0, __m256i s1, __m256 acc) {
    const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x->d))) * y->d;
    return _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(s0, s1)), acc);
}

template <typename B>
static inline float dot1(int n, const void * vx, const void * vy) {
    const B          * x = (const B *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        prefetch(x + i);
        __m256i sum[2] = { _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < 4; ++j) { pairs<false>(j, x + i, (const B *) nullptr, y[i].qs, sum); }
        acc = fma_block(x + i, y + i, sum[0], sum[1], acc);
    }
    return 0.125f * llama_iq2s_fork::hsum(acc);
}

template <typename B>
static inline void dot2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    const B          * xa = (const B *) vxa;
    const B          * xb = (const B *) vxb;
    const block_q8_K * y  = (const block_q8_K *) vy;
    __m256 acc_a = _mm256_setzero_ps(), acc_b = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_K; ++i) {
        prefetch(xa + i);
        prefetch(xb + i);
        __m256i sum[4] = { _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256() };
        for (int j = 0; j < 4; ++j) { pairs<true>(j, xa + i, xb + i, y[i].qs, sum); }
        acc_a = fma_block(xa + i, y + i, sum[0], sum[1], acc_a);
        acc_b = fma_block(xb + i, y + i, sum[2], sum[3], acc_b);
    }
    *s_a = 0.125f * llama_iq2s_fork::hsum(acc_a);
    *s_b = 0.125f * llama_iq2s_fork::hsum(acc_b);
}

} // namespace llama_iq2x_fork

// ggml_vec_dot_t: s[0] = row vx (n IQ2_XXS values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq2_xxs_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    *s = llama_iq2x_fork::dot1<block_iq2_xxs>(n, vx, vy);
}

// two IQ2_XXS rows (up and gate) against the same Q8_K input in one pass, each q8 vector loaded once: bit-identical to two single calls
static inline void llama_dot2_iq2_xxs_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    assert(n % QK_K == 0);
    llama_iq2x_fork::dot2<block_iq2_xxs>(n, s_a, s_b, vxa, vxb, vy);
}

// ggml_vec_dot_t: s[0] = row vx (n IQ2_XS values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq2_xs_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    *s = llama_iq2x_fork::dot1<block_iq2_xs>(n, vx, vy);
}

// two IQ2_XS rows against the same Q8_K input in one pass: bit-identical to two single calls
static inline void llama_dot2_iq2_xs_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    assert(n % QK_K == 0);
    llama_iq2x_fork::dot2<block_iq2_xs>(n, s_a, s_b, vxa, vxb, vy);
}

#endif // __AVX2__
