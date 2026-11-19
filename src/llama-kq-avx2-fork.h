// Q2_K, Q3_K and Q4_K x Q8_K dot products on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h), drop-ins for ggml's vec_dot,
// plus dual-row forms (up and gate against the same input, each q8 vector and the bsums loaded once; no faster than two single calls, see below).
// Bit-identical to the AVX2 branches of ggml_vec_dot_q{2,3,4}_K_q8_K (ggml/src/ggml-cpu/arch/x86/quants.c): every int32 lane holds ggml's sum
// (int adds in any order; int16 steps kept where ggml has them, bounds at each step below) and the floats go in ggml's order (per block
// Q2_K: fmadd(dmin) then fmadd(d); Q3_K: fmadd(d); Q4_K: 128-bit fmadd(dmin) into its own accumulator, fmadd(d); then ggml's hsums).
// What differs from ggml, per 256 values (loop instructions, clang -O2 vs ggml's gcc haswell build: Q2_K 76 vs 72 incl. 2 prefetches,
// Q3_K 112 vs 116, Q4_K 88 vs 96; shuffle-class ops 9 vs 12, 11 vs 14, 14 vs 19; multiplies the same 17 / 24 / 17):
//   Q2_K: scales one vbroadcasti128 + vpand and byte masks that zero-extend in the pshufb (ggml: pmovzxbw + two vpermq). Rest as ggml's.
//   Q3_K: the 12 scale bytes unpacked in vector registers (10 ops; ggml: ~20 scalar ops, vmovd + 3 vpinsrd, extract, 2 inserts), even scales
//         in the low lane and odd in the high one so one pshufb per 32 values picks both; high bit: vpandn + at most one shift (ggml: two).
//   Q4_K: the 12 scale bytes unpacked in vector registers (9 ops; ggml: ~14 scalar ops, vmovd + 3 vpinsrd), one pshufb per scale broadcast
//         (gcc turns ggml's into vpshufd/vpsrldq + vpbroadcastw: two).
//   All: a software prefetch 2 KB ahead (ggml has none): about half of the gain on cold weights.
// Measured (Zen 2 3975WX, one thread, n = 2560, 4096 rows per timing never touched since a 31-53 MB fill, ns per row, best of 3):
//   ggml / single / dual per row: Q2_K 78 / 65 / 66, Q3_K 111 / 88 / 91, Q4_K 110 / 86 / 87 (prefetch off: 77 / 72 / 69, 113 / 102 / 102,
//   104 / 109 / 96); in L1/L2: Q2_K 66 / 63 / 64, Q3_K 95 / 88 / 90, Q4_K 79 / 78 / 78.
//   The dual form gains nothing over two single calls: the q8 loads it shares are memory operands anyway, and with two rows clang spills.
// No load reads past the fields it uses (Q4_K's 16-byte load is d, dmin and the 12 scale bytes exactly).
// Test: tests/test-kq-avx2-fork.cpp (against the CPU backend's own vec_dot).
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
#include "../ggml/src/ggml-common.h"   // tables there are static const (one copy per TU), so every function here is static too: no ODR clash

#ifndef LLAMA_KQ_PREFETCH
#define LLAMA_KQ_PREFETCH 2048   // bytes ahead in the weight row, 0 = off
#endif

namespace llama_kq_fork {

struct alignas(64) tables_t {
    uint8_t q2[8][32];     // Q2_K, scale bytes in both lanes: chunk c (32 values) -> lanes 0-7 = int16 scale 2c, lanes 8-15 = scale 2c+1 (0x80 zero-extends)
    uint8_t w16[8][32];    // int16 scale vectors: word c of each lane broadcast over that lane (Q3_K: even/odd scales per lane; Q4_K: both lanes alike)
    constexpr tables_t() : q2{}, w16{} {
        for (int c = 0; c < 8; ++c) {
            for (int k = 0; k < 32; ++k) {
                q2[c][k]  = (uint8_t) ((k & 1) ? 0x80 : 2 * c + (k >> 4));
                w16[c][k] = (uint8_t) (2 * c + (k & 1));
            }
        }
    }
};
inline constexpr tables_t tables{};   // constant-initialized: no static constructor
static_assert(tables.q2[3][0] == 6 && tables.q2[3][1] == 0x80 && tables.q2[3][16] == 7 && tables.w16[5][31] == 11 && tables.w16[5][16] == 10, "kq tables");

static inline float h2f(const void * p) {   // fp16 -> fp32, exact (ggml's table lookup gives the same bits)
    uint16_t h;
    memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

// ggml's hsum_float_8, same order
static inline float hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

template <typename B> static inline void prefetch(const B * x) {
    if (LLAMA_KQ_PREFETCH > 0) {   // every line of the block: 84 / 110 / 144 bytes -> 2 / 2 / 3 lines
        for (size_t o = 0; o < sizeof(B); o += 64) { _mm_prefetch((const char *) x + LLAMA_KQ_PREFETCH + o, _MM_HINT_T0); }
    }
}

// the scale-broadcast masks through an empty asm: with constant masks clang rewrites each vpshufb into vpshuflw/vpshufhw + vpshufd + vpbroadcastd
// (Q4_K hot: 88 ns per 2560-row against ggml's 79); opaque, they stay one vpshufb with a memory operand
static inline const uint8_t * opaque(const uint8_t * p) {
#if defined(__GNUC__)
    __asm__("" : "+r"(p));
#endif
    return p;
}

template <int k> static inline __m256i srl16(__m256i v) { if constexpr (k == 0) { return v; } else { return _mm256_srli_epi16(v, k); } }
template <int k> static inline __m256i shift_to2(__m256i v) {   // bit k of each byte to bit 2 (16-bit shifts: a single set bit per byte never crosses into the other)
    if constexpr (k < 2) { return _mm256_slli_epi16(v, 2 - k); } else { return srl16<k - 2>(v); }
}

// ---- Q2_K: 16 sub-blocks of 16, 2-bit q (0..3), 4-bit scale and min per sub-block
// maddubs pair |q*q8| <= 2*3*128 = 768, times scale <= 15 in madd: no saturation anywhere

template <int R, int c>
static inline void q2_chunk(const uint8_t * mk, const __m256i * bits, const __m256i * sc, const int8_t * q8, __m256i * sum) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i y = _mm256_loadu_si256((const __m256i *) (q8 + 32 * c));
    const __m256i mask = _mm256_load_si256((const __m256i *) (mk + 32 * c));
    for (int r = 0; r < R; ++r) {
        const __m256i q = _mm256_and_si256(srl16<2 * (c & 3)>(bits[r]), m3);
        sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(_mm256_shuffle_epi8(sc[r], mask), _mm256_maddubs_epi16(q, y)));
    }
}

template <int R>
static inline void q2_block(const block_q2_K * const * x, const block_q8_K * y, __m256 * acc) {
    const uint8_t * mk = opaque(tables.q2[0]);
    const __m256i bs = _mm256_loadu_si256((const __m256i *) y->bsums);
    __m256i sc[R], bits[R], sum[R];
    for (int r = 0; r < R; ++r) {
        prefetch(x[r]);
        const __m128i raw = _mm_loadu_si128((const __m128i *) x[r]->scales);
        const __m256i mins = _mm256_srli_epi16(_mm256_cvtepu8_epi16(raw), 4);   // bytes zero-extended, >> 4 = the min nibble
        const float dmin = -y->d * h2f(x[r]->qs + QK_K / 4 + 2)   /* dmin follows d, which follows qs */;
        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(_mm256_madd_epi16(mins, bs)), acc[r]);
        sc[r] = _mm256_and_si256(_mm256_broadcastsi128_si256(raw), _mm256_set1_epi8(0x0F));
        sum[r] = _mm256_setzero_si256();
        bits[r] = _mm256_loadu_si256((const __m256i *) x[r]->qs);
    }
    q2_chunk<R, 0>(mk, bits, sc, y->qs, sum); q2_chunk<R, 1>(mk, bits, sc, y->qs, sum);
    q2_chunk<R, 2>(mk, bits, sc, y->qs, sum); q2_chunk<R, 3>(mk, bits, sc, y->qs, sum);
    for (int r = 0; r < R; ++r) { bits[r] = _mm256_loadu_si256((const __m256i *) (x[r]->qs + 32)); }
    q2_chunk<R, 4>(mk, bits, sc, y->qs, sum); q2_chunk<R, 5>(mk, bits, sc, y->qs, sum);
    q2_chunk<R, 6>(mk, bits, sc, y->qs, sum); q2_chunk<R, 7>(mk, bits, sc, y->qs, sum);
    for (int r = 0; r < R; ++r) {
        const float d = y->d * h2f(x[r]->qs + QK_K / 4);
        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum[r]), acc[r]);
    }
}

// ---- Q3_K: 16 sub-blocks of 16, q = 2 low bits + high bit (value q - 4 in -4..3), 6-bit signed scale (-32..31)

// int16 scales: low lane = scales 0, 2, .., 14, high lane = 1, 3, .., 15 (so tables.w16[c] picks 2c and 2c+1, what chunk c needs)
static inline __m256i q3_scales(const uint8_t * s) {
    const __m128i a = _mm_loadl_epi64((const __m128i *) s);                       // s0..s7
    const __m128i lo = _mm_and_si128(_mm_unpacklo_epi64(a, _mm_srli_epi16(a, 4)), _mm_set1_epi8(0x0F));   // k < 8: s[k] & 15, k >= 8: s[k-8] >> 4
    uint32_t h32;
    memcpy(&h32, s + 8, 4);
    // dword g = scales 4g..4g+3: bits 2g, 2g+1 of s[8+b] to bits 4, 5 (left 4, 2, 0, right 2: one shift each way so byte 3 keeps its top bits)
    const __m128i h = _mm_sllv_epi32(_mm_set1_epi32((int) h32), _mm_setr_epi32(4, 2, 0, 0));
    const __m128i hi = _mm_and_si128(_mm_srlv_epi32(h, _mm_setr_epi32(0, 0, 0, 2)), _mm_set1_epi8(0x30));
    const __m128i sc = _mm_sub_epi8(_mm_or_si128(lo, hi), _mm_set1_epi8(32));
    return _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)));
}

// chunk c (32 values) as ggml: maddubs(q3l, q8) - maddubs(4 where the high bit is clear, q8) in int16 (pairs <= 2*3*128 and 2*4*128, difference
// in int16); the high-bit term from one vpandn + at most one shift. (Tried for two rows: maddubs(q3l | 4*hbit) - a shared maddubs(4, q8), the same
// int16 with 8 fewer maddubs per block: 5% slower on Zen 2, the ALU ops it adds cost more than the multiplies it saves.)
template <int R, int c>
static inline void q3_chunk(const uint8_t * mk, const __m256i * bits, const __m256i * hb, const __m256i * sc, const int8_t * q8, __m256i * sum) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i y = _mm256_loadu_si256((const __m256i *) (q8 + 32 * c));
    const __m256i mask = _mm256_load_si256((const __m256i *) (mk + 32 * c));
    for (int r = 0; r < R; ++r) {
        const __m256i ql = _mm256_and_si256(srl16<2 * (c & 3)>(bits[r]), m3);
        const __m256i qh = shift_to2<c>(_mm256_andnot_si256(hb[r], _mm256_set1_epi8((char) (1 << c))));
        const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(ql, y), _mm256_maddubs_epi16(qh, y));
        sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(_mm256_shuffle_epi8(sc[r], mask), p));
    }
}

template <int R>
static inline void q3_block(const block_q3_K * const * x, const block_q8_K * y, __m256 * acc) {
    const uint8_t * mk = opaque(tables.w16[0]);
    __m256i sc[R], hb[R], bits[R], sum[R];
    for (int r = 0; r < R; ++r) {
        prefetch(x[r]);
        sc[r] = q3_scales(x[r]->scales);
        hb[r] = _mm256_loadu_si256((const __m256i *) x[r]->hmask);
        bits[r] = _mm256_loadu_si256((const __m256i *) x[r]->qs);
        sum[r] = _mm256_setzero_si256();
    }
    q3_chunk<R, 0>(mk, bits, hb, sc, y->qs, sum); q3_chunk<R, 1>(mk, bits, hb, sc, y->qs, sum);
    q3_chunk<R, 2>(mk, bits, hb, sc, y->qs, sum); q3_chunk<R, 3>(mk, bits, hb, sc, y->qs, sum);
    for (int r = 0; r < R; ++r) { bits[r] = _mm256_loadu_si256((const __m256i *) (x[r]->qs + 32)); }
    q3_chunk<R, 4>(mk, bits, hb, sc, y->qs, sum); q3_chunk<R, 5>(mk, bits, hb, sc, y->qs, sum);
    q3_chunk<R, 6>(mk, bits, hb, sc, y->qs, sum); q3_chunk<R, 7>(mk, bits, hb, sc, y->qs, sum);
    for (int r = 0; r < R; ++r) {
        const float d = y->d * h2f(&x[r]->d);
        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum[r]), acc[r]);
    }
}

// ---- Q4_K: 8 sub-blocks of 32, 4-bit q, 6-bit scale and min per sub-block (12 packed bytes)
// maddubs pair <= 2*15*128 = 3840, times scale <= 63 in madd: no saturation

// bytes 0-7 = scales, 8-15 = mins (ggml's utmp) from the block's first 16 bytes v = d, dmin, s0..s11:
//   sc j<4 = s[j] & 63, sc j>=4 = (s[j+4] & 15) | (s[j-4] >> 6) << 4; m j<4 = s[j+4] & 63, m j>=4 = (s[j+4] >> 4) | (s[j] >> 6) << 4
static inline __m128i q4_scales_mins(const __m128i v) {
    const __m128i w = _mm_shuffle_epi8(v, _mm_setr_epi8(4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));   // s0-3, s8-11, s4-7, s8-11
    const __m128i u = _mm_shuffle_epi8(v, _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11));   // the top-2-bit sources
    const __m128i lo = _mm_and_si128(w, _mm_setr_epi8(63, 63, 63, 63, 15, 15, 15, 15, 63, 63, 63, 63, 0, 0, 0, 0));
    const __m128i nib = _mm_and_si128(_mm_srli_epi16(w, 4), _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 15));
    const __m128i top = _mm_and_si128(_mm_srli_epi16(u, 2), _mm_set1_epi8(0x30));   // (s >> 6) << 4 = (s >> 2) & 0x30, zero lanes stay zero
    return _mm_or_si128(_mm_or_si128(lo, nib), top);
}

template <int R, int c>
static inline void q4_chunk(const uint8_t * mk, const block_q4_K * const * x, const __m256i * sc, const int8_t * q8, __m256i * sum) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i y = _mm256_loadu_si256((const __m256i *) (q8 + 32 * c));
    const __m256i mask = _mm256_load_si256((const __m256i *) (mk + 32 * c));
    for (int r = 0; r < R; ++r) {
        const __m256i bits = _mm256_loadu_si256((const __m256i *) (x[r]->qs + 32 * (c / 2)));   // the odd chunk reloads: a memory operand, no extra register
        const __m256i q = _mm256_and_si256(srl16<4 * (c & 1)>(bits), m4);
        sum[r] = _mm256_add_epi32(sum[r], _mm256_madd_epi16(_mm256_shuffle_epi8(sc[r], mask), _mm256_maddubs_epi16(q, y)));
    }
}

template <int R>
static inline void q4_block(const block_q4_K * const * x, const block_q8_K * y, __m256 * acc, __m128 * acc_m) {
    const __m256i bs = _mm256_loadu_si256((const __m256i *) y->bsums);
    const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));   // ggml's op: int16 pair sums, shared by the rows
    const uint8_t * mk = opaque(tables.w16[0]);
    __m256i sc[R], sum[R];
    for (int r = 0; r < R; ++r) {
        prefetch(x[r]);
        const __m256i ms = _mm256_cvtepu8_epi16(q4_scales_mins(_mm_loadu_si128((const __m128i *) x[r])));
        const float dmin = -y->d * h2f((const uint8_t *) x[r] + 2);
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(ms, 1), q8s);
        acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);
        sc[r] = _mm256_permute4x64_epi64(ms, 0x44);   // the 8 int16 scales in both lanes
        sum[r] = _mm256_setzero_si256();
    }
    q4_chunk<R, 0>(mk, x, sc, y->qs, sum); q4_chunk<R, 1>(mk, x, sc, y->qs, sum); q4_chunk<R, 2>(mk, x, sc, y->qs, sum); q4_chunk<R, 3>(mk, x, sc, y->qs, sum);
    q4_chunk<R, 4>(mk, x, sc, y->qs, sum); q4_chunk<R, 5>(mk, x, sc, y->qs, sum); q4_chunk<R, 6>(mk, x, sc, y->qs, sum); q4_chunk<R, 7>(mk, x, sc, y->qs, sum);
    for (int r = 0; r < R; ++r) {
        const float d = y->d * h2f(x[r]);
        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum[r]), acc[r]);
    }
}

// ---- row drivers: R rows of n values against one Q8_K row

template <int R, typename B, void (*block)(const B * const *, const block_q8_K *, __m256 *)>
static inline void rows(int n, const void * const * vx, const void * vy, float * out) {
    assert(n % QK_K == 0);
    const block_q8_K * y = (const block_q8_K *) vy;
    const B * x[R];
    __m256 acc[R];
    for (int r = 0; r < R; ++r) { x[r] = (const B *) vx[r]; acc[r] = _mm256_setzero_ps(); }
    for (int i = 0; i < n / QK_K; ++i) {
        const B * xi[R];
        for (int r = 0; r < R; ++r) { xi[r] = x[r] + i; }
        block(xi, y + i, acc);
    }
    for (int r = 0; r < R; ++r) { out[r] = hsum(acc[r]); }
}

template <int R>
static inline void rows_q4(int n, const void * const * vx, const void * vy, float * out) {
    assert(n % QK_K == 0);
    const block_q8_K * y = (const block_q8_K *) vy;
    __m256 acc[R];
    __m128 acc_m[R];
    for (int r = 0; r < R; ++r) { acc[r] = _mm256_setzero_ps(); acc_m[r] = _mm_setzero_ps(); }
    for (int i = 0; i < n / QK_K; ++i) {
        const block_q4_K * xi[R];
        for (int r = 0; r < R; ++r) { xi[r] = (const block_q4_K *) vx[r] + i; }
        q4_block<R>(xi, y + i, acc, acc_m);
    }
    for (int r = 0; r < R; ++r) {   // ggml: acc_m reduced by movehl + movehdup, then hsum_float_8(acc) + it
        __m128 m = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        out[r] = hsum(acc[r]) + _mm_cvtss_f32(m);
    }
}

} // namespace llama_kq_fork

// ggml_vec_dot_t: s[0] = row vx (n values) . vy (n Q8_K values); nrc == 1. dot2: two rows against the same vy in one pass, bit-identical to two single calls.
static inline void llama_dot_q2_K_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    llama_kq_fork::rows<1, block_q2_K, llama_kq_fork::q2_block<1>>(n, &vx, vy, s);
}
static inline void llama_dot2_q2_K_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    const void * vx[2] = { vxa, vxb };
    float s[2];
    llama_kq_fork::rows<2, block_q2_K, llama_kq_fork::q2_block<2>>(n, vx, vy, s);
    *s_a = s[0]; *s_b = s[1];
}

static inline void llama_dot_q3_K_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    llama_kq_fork::rows<1, block_q3_K, llama_kq_fork::q3_block<1>>(n, &vx, vy, s);
}
static inline void llama_dot2_q3_K_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    const void * vx[2] = { vxa, vxb };
    float s[2];
    llama_kq_fork::rows<2, block_q3_K, llama_kq_fork::q3_block<2>>(n, vx, vy, s);
    *s_a = s[0]; *s_b = s[1];
}

static inline void llama_dot_q4_K_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    llama_kq_fork::rows_q4<1>(n, &vx, vy, s);
}
static inline void llama_dot2_q4_K_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    const void * vx[2] = { vxa, vxb };
    float s[2];
    llama_kq_fork::rows_q4<2>(n, vx, vy, s);
    *s_a = s[0]; *s_b = s[1];
}

#endif // __AVX2__
