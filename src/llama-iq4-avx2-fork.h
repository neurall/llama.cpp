// IQ4_NL / MXFP4 / Q8_0 / Q2_0 x Q8_0 and IQ4_XS x Q8_K dot products on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h),
// drop-ins for ggml's vec_dot. Bit-identical to the AVX2 branches of ggml_vec_dot_{iq4_nl,mxfp4,q8_0,q2_0}_q8_0 and ggml_vec_dot_iq4_xs_q8_K
// (ggml/src/ggml-cpu/arch/x86/quants.c, the default ones, not the GGML_CPU_GPU_SHAPE dots: with that switch on, ggml's Q8_0 / Q2_0 / IQ4_NL / ...
// dots add in the CUDA order and these kernels must not stand in for them). Every int32 lane holds ggml's sum (int adds in any order; no
// saturation: |maddubs pair| <= 2*127*128 for IQ4, 2*12*128 MXFP4, 2*128*128 Q8_0 as ggml's own, 2*3*128 Q2_0) and the floats go in ggml's order:
// per 32-value block one fmadd of d = fp(y.d) * fp(x.d) into accumulator (block & 1) (Q8_0: block & 3, Q2_0: per 64-value block, both of its
// halves into accumulator (block & 1)), IQ4_XS per 256-value block one fmadd of fp16(x.d) * y.d, then ggml's hsum_float_8; an odd last
// IQ4_NL / MXFP4 block is ggml's scalar tail: one fmaf (clang contracts ggml's `sumf += d * (sumi1 + sumi2)`), with the true integer products.
// What differs from ggml (clang 18 -O2 vs the haswell .so, loop instructions / memory operands / shuffles):
//   IQ4_XS: the 6-bit sub-block scale is a 2 KB table row read by vpmaddwd itself (ggml: 5 scalar ops, vmovd, vpbroadcastw per 32 values):
//     157 / 29 / 16 per 256 values vs ggml 167 / 24 / 24. Dual: q8 loads and the block scale's y.d shared.
//   IQ4_NL, MXFP4: ggml's loop is already tight (clang merged its two 128-bit pshufb into one): single = ggml's code, 41 / 12 / 4 per 2 blocks
//     (MXFP4 40). Dual: 71 / 20 / 8 per 2 blocks of 2 rows (17.8 a row-block vs 20.5), q8 load and fp16(y.d) shared.
//   Q8_0, Q2_0: ggml's single row is minimal (Q8_0: 2 loads, 2 psignb, maddubs, madd, cvt, fmadd per block), so dual forms only; Q2_0's shares
//     the q8 loads, fp16(y.d) and maddubs(1, y): 34 vs 35.5 instructions a row-block, 7.5 vs 10 memory operands.
//   fp16 scales by a 256 KiB table as ggml (vcvtph2ps via a GPR cost 15%); the dual forms prefetch 2048 bytes ahead in both rows (without it two
//     interleaved row streams cost 30-48% on cold rows: the hardware prefetcher loses them).
// One Zen 2 thread (3975WX, other jobs running), M rows/s, cold = weights streamed from RAM (4096 rows of 640 / 1536 of 2560, not touched for
// 30 timings), dual counted as two rows; ggml / single / dual:
//   IQ4_NL n=640 25.4 / 24.9 / 30.1 (+19%), n=2560 7.11 / 6.97 / 7.76 (+9%); MXFP4 n=640 26.0 / 25.5 / 31.3 (+20%), n=2560 7.28 / 7.11 / 7.88 (+8%)
//   IQ4_XS n=2560 7.46 / 7.88 (+6%) / 8.06 (+8%), in L2 9.22 / 10.60 (+15%) / 9.78
//   Q2_0 n=640 25.0 / - / 30.7 (+23%), n=2560 6.82 / - / 8.14 (+19%); Q8_0 n=640 18.5 / - / 17.5 (-6%), n=2560 4.91 / - / 4.66 (-5%): no gain, keep ggml's.
// So for IQ4_NL / MXFP4 / Q2_0 the dual form is what gains, for IQ4_XS the single already does; single IQ4_NL / MXFP4 match ggml (-1..-3%).
// Reads: no over-read past a row or the q8 row, every load stays inside its block (prefetches may point past the row's end: hints, no fault).
// No alignment needed (unaligned loads). Needs AVX2 + FMA + F16C at run time.
// Test: tests/test-iq4-avx2-fork.cpp (against the CPU backend's own vec_dot and its AVX2 form; "bench" / "hot" print the numbers above).
#pragma once

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)

#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if !defined(GGML_COMMON_DECL)
#define GGML_COMMON_DECL_CPP
#endif
#if !defined(GGML_COMMON_IMPL)
#define GGML_COMMON_IMPL_CPP
#endif
#include "../ggml/src/ggml-common.h"   // kvalues_* are static const (one copy per TU), so every function here is static too: no ODR clash

#ifndef LLAMA_IQ4_PREFETCH
#define LLAMA_IQ4_PREFETCH 2048   // bytes ahead in each weight row of the dual forms, 0 = off (the single forms do not prefetch: no gain there)
#endif

namespace llama_iq4_fork {

// fp16 -> fp32 bits, exact (subnormals normalized; Inf / NaN kept as such)
constexpr uint32_t half_bits(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31;
    uint32_t m = h & 0x3FFu;
    if (e == 31) { return s | 0x7F800000u | (m << 13); }
    if (e != 0)  { return s | ((e + 112) << 23) | (m << 13); }
    if (m == 0)  { return s; }
    uint32_t x = 113;
    while (!(m & 0x400u)) { m <<= 1; --x; }
    return s | (x << 23) | ((m & 0x3FFu) << 13);
}

struct alignas(64) tables_t {
    uint32_t f16[65536];   // fp16 -> fp32 bits: one load per scale as ggml's ggml_table_f32_f16 (vcvtph2ps from a GPR was 15% slower: vmovd + 2-uop convert)
    int16_t  ls[64][16];   // IQ4_XS 6-bit sub-block scale s -> s - 32 in all 16 lanes: the madd factor (ggml: scalar - 32, vmovd, vpbroadcastw)
    uint32_t e8m0[256];    // MXFP4 e -> bits of ggml_e8m0_to_fp32_half(e) (ggml reads the same values from ggml_table_f32_e8m0_half)
    constexpr tables_t() : f16{}, ls{}, e8m0{} {
        for (uint32_t h = 0; h < 65536; ++h) { f16[h] = half_bits(h); }
        for (int s = 0; s < 64; ++s) { for (int l = 0; l < 16; ++l) { ls[s][l] = (int16_t) (s - 32); } }
        for (int e = 0; e < 256; ++e) { e8m0[e] = e < 2 ? 0x00200000u << e : (uint32_t) (e - 1) << 23; }
    }
};
inline constexpr tables_t tables{};   // constant-initialized (264 KiB of .rodata, one copy): no static constructor
static_assert(tables.f16[0x3C00] == 0x3F800000u && tables.f16[0x0001] == 0x33800000u && tables.f16[0x8400] == 0xB8800000u &&
              tables.f16[0x7BFF] == 0x477FE000u && tables.f16[0x03FF] == 0x387FC000u, "fp16 table");
static_assert(tables.ls[0][15] == -32 && tables.ls[63][0] == 31 && tables.e8m0[0] == 0x00200000u && tables.e8m0[1] == 0x00400000u &&
              tables.e8m0[128] == 0x3F800000u && tables.e8m0[255] == 0x7F000000u, "iq4 tables");

// _mm_load_ss reads through a may_alias type: a plain vmovss load, no GPR round trip
static inline float f16(ggml_half h) { return _mm_cvtss_f32(_mm_load_ss((const float *) &tables.f16[h])); }
static inline float e8m0(uint8_t e)  { return _mm_cvtss_f32(_mm_load_ss((const float *) &tables.e8m0[e])); }

template <int R>
static inline void prefetch(const void * p) {
    if (R > 1 && LLAMA_IQ4_PREFETCH > 0) { _mm_prefetch((const char *) p + LLAMA_IQ4_PREFETCH, _MM_HINT_T0); }
}

// ggml's hsum_float_8, same order
static inline float hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

static inline int hsum_i32(__m256i x) {   // integer: any order
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    r = _mm_add_epi32(r, _mm_unpackhi_epi64(r, r));
    return _mm_cvtsi128_si32(_mm_add_epi32(r, _mm_shuffle_epi32(r, 1)));
}

// the 16 nibble bytes of a 32-value block as codebook values in ggml's order: low nibbles in bytes 0-15, high nibbles in 16-31; ggml's
// load + srlw + vinserti128 + and + pshufb (a vbroadcasti128 + vpblendd form was 10% slower for IQ4_XS on Zen 2)
static inline __m256i nl_values(__m256i book, const uint8_t * qs) {
    const __m128i v = _mm_loadu_si128((const __m128i *) qs);
    return _mm256_shuffle_epi8(book, _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(v), _mm_srli_epi16(v, 4), 1), _mm256_set1_epi8(0x0F)));
}

// ggml's mul_add_epi8: maddubs(|k|, sign(q, k)), int16 pairs (for q = -128 and k < 0 this is -128 * |k|, as ggml: keep it, no |q| form)
static inline __m256i mul_add(__m256i k, __m256i q) { return _mm256_maddubs_epi16(_mm256_abs_epi8(k), _mm256_sign_epi8(q, k)); }

static inline __m256i madd1(__m256i p) { return _mm256_madd_epi16(p, _mm256_set1_epi16(1)); }

// ---- IQ4_NL and MXFP4: 32 4-bit codebook indices + one scale per block against Q8_0 (ggml's two code paths are the same but codebook and scale)
struct nl_iq4 {
    using block = block_iq4_nl;
    static const int8_t * book() { return kvalues_iq4nl; }
    static float scale(const block & x) { return f16(x.d); }
};
struct nl_mxfp4 {
    using block = block_mxfp4;
    static const int8_t * book() { return kvalues_mxfp4; }
    static float scale(const block & x) { return e8m0(x.e); }
};

// ggml's scalar tail for an odd last block: the true products (q = -128 included), so here maddubs(|q|, sign(k, q)), exact for |k| <= 127
template <class T>
static inline float nl_tail(__m256i book, const typename T::block & x, const block_q8_0 & y, float sumf) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) y.qs);
    const __m256i k = nl_values(book, x.qs);
    const int sumi = hsum_i32(madd1(_mm256_maddubs_epi16(_mm256_abs_epi8(q), _mm256_sign_epi8(k, q))));
    return std::fmaf(f16(y.d) * T::scale(x), (float) sumi, sumf);
}

// one 32-value block of up to two rows against q8 block y: ggml's fmadd into acc[r]
template <class T, int R>
static inline void nl_block(__m256i book, const typename T::block * const * x, int i, const block_q8_0 & y, __m256 * acc) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) y.qs);
    const float dy = f16(y.d);
    for (int r = 0; r < R; ++r) {
        const __m256i p = madd1(mul_add(nl_values(book, x[r][i].qs), q));
        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dy * T::scale(x[r][i])), _mm256_cvtepi32_ps(p), acc[r]);
    }
}

template <class T, int R>
static inline void nl_dot(int n, float * const * s, const typename T::block * const * x, const block_q8_0 * y) {
    const __m256i book = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) T::book()));
    const int nb = n / 32;
    __m256 even[R], odd[R];
    for (int r = 0; r < R; ++r) { even[r] = _mm256_setzero_ps(); odd[r] = _mm256_setzero_ps(); }
    int i = 0;
    for (; i + 1 < nb; i += 2) {   // two blocks per step, as ggml: block i into even, i + 1 into odd
        for (int r = 0; r < R; ++r) { prefetch<R>(x[r] + i); }   // 2 blocks = 36 (MXFP4 34) bytes per step < 64: every line asked for
        nl_block<T, R>(book, x, i, y[i], even);
        nl_block<T, R>(book, x, i + 1, y[i + 1], odd);
    }
    for (int r = 0; r < R; ++r) {
        float sumf = hsum(_mm256_add_ps(even[r], odd[r]));
        if (i < nb) { sumf = nl_tail<T>(book, x[r][i], y[i], sumf); }
        *s[r] = sumf;
    }
}

// ---- IQ4_XS x Q8_K: 8 sub-blocks of 32 with 6-bit scales (low 4 bits in scales_l nibbles, high 2 in scales_h), one fp16 d per 256
template <int R>
static inline void xs_dot(int n, float * const * s, const block_iq4_xs * const * x, const block_q8_K * y) {
    const __m256i book = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) kvalues_iq4nl));
    __m256 acc[R];
    for (int r = 0; r < R; ++r) { acc[r] = _mm256_setzero_ps(); }
    for (int i = 0; i < n / QK_K; ++i) {
        __m256i sum[R][2];
        uint32_t l[R], h[R];
        for (int r = 0; r < R; ++r) {
            prefetch<R>(x[r] + i);
            prefetch<R>((const char *) (x[r] + i) + 64);
            prefetch<R>((const char *) (x[r] + i) + 128);   // 136-byte blocks: three lines
            sum[r][0] = sum[r][1] = _mm256_setzero_si256();
            memcpy(&l[r], x[r][i].scales_l, 4);
            h[r] = x[r][i].scales_h;
        }
        const int8_t * q8 = y[i].qs;
        for (int j = 0; j < 4; ++j) {   // sub-blocks 2j, 2j+1: |sum lane| <= 8 * 32 * 2 * 32512 < 2^31
            const __m256i q0 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j));
            const __m256i q1 = _mm256_loadu_si256((const __m256i *) (q8 + 64 * j + 32));
            for (int r = 0; r < R; ++r) {
                const uint8_t * qs = x[r][i].qs + 32 * j;
                const int16_t * s0 = tables.ls[(l[r] & 15) | ((h[r] & 3) << 4)];
                const int16_t * s1 = tables.ls[((l[r] >> 4) & 15) | ((h[r] & 12) << 2)];
                l[r] >>= 8;
                h[r] >>= 4;
                sum[r][0] = _mm256_add_epi32(sum[r][0], _mm256_madd_epi16(mul_add(nl_values(book, qs), q0), _mm256_load_si256((const __m256i *) s0)));
                sum[r][1] = _mm256_add_epi32(sum[r][1], _mm256_madd_epi16(mul_add(nl_values(book, qs + 16), q1), _mm256_load_si256((const __m256i *) s1)));
            }
        }
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(f16(x[r][i].d) * y[i].d), _mm256_cvtepi32_ps(_mm256_add_epi32(sum[r][0], sum[r][1])), acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) { *s[r] = hsum(acc[r]); }
}

// ---- Q8_0 x Q8_0, two rows: ggml's four accumulators (block & 3 while four blocks remain, the rest into acc 0), mul_sum_i8_pairs_float
static inline void q8_block(const block_q8_0 & xa, const block_q8_0 & xb, const block_q8_0 & y, __m256 & aa, __m256 & ab) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) y.qs);
    const float dy = f16(y.d);
    const __m256i a = _mm256_loadu_si256((const __m256i *) xa.qs), b = _mm256_loadu_si256((const __m256i *) xb.qs);
    aa = _mm256_fmadd_ps(_mm256_set1_ps(f16(xa.d) * dy), _mm256_cvtepi32_ps(madd1(mul_add(a, q))), aa);
    ab = _mm256_fmadd_ps(_mm256_set1_ps(f16(xb.d) * dy), _mm256_cvtepi32_ps(madd1(mul_add(b, q))), ab);
}

// ---- Q2_0 x Q8_0: 64 2-bit values (v - 1) and one fp16 d per block, two Q8_0 blocks per block
static inline __m256i q2_unpack(const uint8_t * qs) {   // ggml's q2_0_unpack_32_avx2: byte b -> b & 3, (b >> 2) & 3, (b >> 4) & 3, b >> 6
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

// one Q2_0 block of rows a, b into aa, ab: per half ggml's fmadd of dx * dy with madd(maddubs(v, y) - maddubs(1, y)); maddubs(1, y) and dy shared
static inline void q2_block(const block_q2_0 & xa, const block_q2_0 & xb, const block_q8_0 * y, __m256 & aa, __m256 & ab) {
    const float da = f16(xa.d), db = f16(xb.d);
    for (int h = 0; h < 2; ++h) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) y[h].qs);
        const __m256i sy = _mm256_maddubs_epi16(_mm256_set1_epi8(1), q);
        const float dy = f16(y[h].d);
        const __m256 pa = _mm256_cvtepi32_ps(madd1(_mm256_sub_epi16(_mm256_maddubs_epi16(q2_unpack(xa.qs + 8 * h), q), sy)));
        const __m256 pb = _mm256_cvtepi32_ps(madd1(_mm256_sub_epi16(_mm256_maddubs_epi16(q2_unpack(xb.qs + 8 * h), q), sy)));
        aa = _mm256_fmadd_ps(_mm256_set1_ps(da * dy), pa, aa);
        ab = _mm256_fmadd_ps(_mm256_set1_ps(db * dy), pb, ab);
    }
}

} // namespace llama_iq4_fork

// ggml_vec_dot_t: s[0] = row vx (n IQ4_NL values) . vy (n Q8_0 values); nrc == 1
static inline void llama_dot_iq4_nl_q8_0_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq4_fork;
    assert(n % QK4_NL == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_iq4_nl * x[1] = { (const block_iq4_nl *) vx };
    float * out[1] = { s };
    f::nl_dot<f::nl_iq4, 1>(n, out, x, (const block_q8_0 *) vy);
}

// two rows against the same Q8_0 input in one pass (two consecutive down rows of one expert, or an up and a gate row): bit-identical to two single calls
static inline void llama_dot2_iq4_nl_q8_0_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq4_fork;
    assert(n % QK4_NL == 0);
    const block_iq4_nl * x[2] = { (const block_iq4_nl *) vxa, (const block_iq4_nl *) vxb };
    float * out[2] = { s_a, s_b };
    f::nl_dot<f::nl_iq4, 2>(n, out, x, (const block_q8_0 *) vy);
}

static inline void llama_dot_mxfp4_q8_0_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq4_fork;
    assert(n % QK_MXFP4 == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_mxfp4 * x[1] = { (const block_mxfp4 *) vx };
    float * out[1] = { s };
    f::nl_dot<f::nl_mxfp4, 1>(n, out, x, (const block_q8_0 *) vy);
}

static inline void llama_dot2_mxfp4_q8_0_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq4_fork;
    assert(n % QK_MXFP4 == 0);
    const block_mxfp4 * x[2] = { (const block_mxfp4 *) vxa, (const block_mxfp4 *) vxb };
    float * out[2] = { s_a, s_b };
    f::nl_dot<f::nl_mxfp4, 2>(n, out, x, (const block_q8_0 *) vy);
}

// ggml_vec_dot_t: s[0] = row vx (n IQ4_XS values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq4_xs_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq4_fork;
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_iq4_xs * x[1] = { (const block_iq4_xs *) vx };
    float * out[1] = { s };
    f::xs_dot<1>(n, out, x, (const block_q8_K *) vy);
}

static inline void llama_dot2_iq4_xs_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq4_fork;
    assert(n % QK_K == 0);
    const block_iq4_xs * x[2] = { (const block_iq4_xs *) vxa, (const block_iq4_xs *) vxb };
    float * out[2] = { s_a, s_b };
    f::xs_dot<2>(n, out, x, (const block_q8_K *) vy);
}

// Q8_0 x Q8_0, two rows (no single form: ggml's is already minimal)
static inline void llama_dot2_q8_0_q8_0_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq4_fork;
    assert(n % QK8_0 == 0);
    const block_q8_0 * xa = (const block_q8_0 *) vxa, * xb = (const block_q8_0 *) vxb, * y = (const block_q8_0 *) vy;
    const int nb = n / QK8_0;
    __m256 a[4], b[4];
    for (int k = 0; k < 4; ++k) { a[k] = _mm256_setzero_ps(); b[k] = _mm256_setzero_ps(); }
    int i = 0;
    for (; i + 3 < nb; i += 4) {
        for (int k = 0; k < 4; ++k) {
            f::prefetch<2>(xa + i + k);
            f::prefetch<2>(xb + i + k);
            f::q8_block(xa[i + k], xb[i + k], y[i + k], a[k], b[k]);
        }
    }
    for (; i < nb; ++i) { f::q8_block(xa[i], xb[i], y[i], a[0], b[0]); }
    *s_a = f::hsum(_mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3])));
    *s_b = f::hsum(_mm256_add_ps(_mm256_add_ps(b[0], b[1]), _mm256_add_ps(b[2], b[3])));
}

// Q2_0 x Q8_0, two rows (no single form: ggml's is already minimal)
static inline void llama_dot2_q2_0_q8_0_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq4_fork;
    assert(n % QK2_0 == 0);
    const block_q2_0 * xa = (const block_q2_0 *) vxa, * xb = (const block_q2_0 *) vxb;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / QK2_0;
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 1 < nb; i += 2) {   // as ggml: block i into accumulator 0, i + 1 into 1, an odd last one into 0
        f::prefetch<2>(xa + i);   // 36 bytes per step: every line asked for
        f::prefetch<2>(xb + i);
        f::q2_block(xa[i], xb[i], y + 2 * i, a0, b0);
        f::q2_block(xa[i + 1], xb[i + 1], y + 2 * i + 2, a1, b1);
    }
    if (i < nb) { f::q2_block(xa[i], xb[i], y + 2 * i, a0, b0); }
    *s_a = f::hsum(_mm256_add_ps(a0, a1));
    *s_b = f::hsum(_mm256_add_ps(b0, b1));
}

#endif // __AVX2__ && __FMA__ && __F16C__
