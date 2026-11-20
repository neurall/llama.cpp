// IQ1_S and IQ1_M x Q8_K dot products on AVX2 for the bell pool's CPU experts (src/llama-bellpool-fork.h), drop-ins for ggml's vec_dot.
// Kimi K3.1 stores 164 of its 184 up/gate and 90 of its 92 down tensors as IQ1_S, DeepSeek V4.1 Flash 32 up/gate and 28 down as IQ1_M.
// Bit-identical to the AVX2 branches of ggml_vec_dot_iq1_s_q8_K / ggml_vec_dot_iq1_m_q8_K (ggml/src/ggml-cpu/arch/x86/quants.c, haswell
// variant = BMI2 path): every int32 lane holds the same integer and the floats go in ggml's order (per block one fmadd of d per
// accumulator, the end hsum_float_8 of both and one fmadd 0.125f * delta hsum + main hsum, as clang contracts ggml's line).
// Integer changes, each exact:
// - no pdep: ggml builds the 11-bit grid indices with 4 pdep per 64 values, microcoded on Zen 1/2 (dozens of cycles each); here all 32
//   indices of a block come from a few shifts and masks of the 16 qh bytes and two punpcklbw with qs
// - grid . q8 as maddubs(scale bytes, sign(q8, grid)) summed in int16 over the block's 8 vectors, one madd by 1 per block: ggml's
//   maddubs(|g|, sign(q8, g)) then madd by the scale per vector gives the same int32 (|g| = 1 where sign(q8, g) != 0); bounds:
//   |sign(q8, g)| <= 128 (-128 stays -128 as in ggml), scale <= 15, so an int16 lane holds at most 8 * 2 * 15 * 128 = 30720
// - IQ1_S delta: madd(bsums, 2s+1) then psignd by qh bit 15 instead of madd(bsums, 1) * pmulld (exact for any
//   int16 bsums: |2 * 32768 * 15| < 2^31)
// - IQ1_M fp16 block scale: one multiply gathers the four top nibbles of the scales (pext's job) instead of ggml's shift-or chain
// Test: tests/test-iq1-avx2-fork.cpp (against the CPU backend's own vec_dot).
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
#include "../ggml/src/ggml-common.h"   // iq1s_grid is static const (one copy per TU), so every function here is static too: no ODR clash

#ifndef LLAMA_IQ1_PREFETCH
#define LLAMA_IQ1_PREFETCH 2048   // bytes ahead in the weight row, 0 = off; not measured here
#endif

namespace llama_iq1_fork {

struct alignas(32) tables_t {
    uint8_t dsel[4][32];    // IQ1_M delta signs: vector k of a half block = entries 4k..4k+3, 8 bytes each (low lane 4k, 4k+1; high 4k+2, 4k+3)
    uint8_t msel[8][32];    // IQ1_M scales: vector v's scale byte in ggml's srlv layout: 2*(v/2) + 8*(v%2) in both lanes
    constexpr tables_t() : dsel{}, msel{} {
        for (int k = 0; k < 4; ++k) { for (int l = 0; l < 32; ++l) { dsel[k][l] = (uint8_t) (4 * k + l / 8); } }
        for (int v = 0; v < 8; ++v) { for (int l = 0; l < 32; ++l) { msel[v][l] = (uint8_t) (2 * (v / 2) + 8 * (v % 2)); } }
    }
};
inline constexpr tables_t tables{};   // constant-initialized: no static constructor
static_assert(tables.dsel[1][15] == 5 && tables.dsel[3][31] == 15 && tables.msel[5][31] == 12 && tables.msel[6][0] == 6, "iq1 tables");

// IQ1_M fp16 block scale = top nibbles of the four scale words, in order: (s & 0xF000...) * M puts nibble k (bit 12+16k) at 48+4k, the
// other eleven partial products land below bit 48 in disjoint nibbles (no carry) or above bit 63
constexpr uint16_t iq1m_d16(uint64_t s) { return (uint16_t) (((s & 0xF000F000F000F000ull) * 0x0000001001001001ull) >> 48); }
static_assert(iq1m_d16(0x1000200030004000ull) == 0x1234 && iq1m_d16(0xFFFFFFFFFFFFFFFFull) == 0xFFFF && iq1m_d16(0x8FFF0FFF0FFF0FFFull) == 0x8000,
              "iq1m scale nibbles");

// four grid rows from four 16-bit indices packed in ix (as _mm256_set_epi64x, each 128-bit half a load plus a load-high)
static inline __m128i grid2(uint64_t a, uint64_t b) {
    return _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64((const __m128i *) &iq1s_grid[a])), (const double *) &iq1s_grid[b]));
}
static inline __m256i grid4(uint64_t ix) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(grid2(ix & 0xffff, (ix >> 16) & 0xffff)), grid2((ix >> 32) & 0xffff, ix >> 48), 1);
}

// the 32 grid indices of a block: qs byte | index bits 8-10 (hi, one byte per entry in entry order) << 8; ix[j] = entries 8j..8j+7
static inline void indices(const uint8_t * qs, __m256i hi, __m128i * ix) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) qs);
    const __m256i a = _mm256_unpacklo_epi8(q, hi);   // entries 0-7 | 16-23
    const __m256i b = _mm256_unpackhi_epi8(q, hi);   // entries 8-15 | 24-31
    ix[0] = _mm256_castsi256_si128(a);
    ix[1] = _mm256_castsi256_si128(b);
    ix[2] = _mm256_extracti128_si256(a, 1);
    ix[3] = _mm256_extracti128_si256(b, 1);
}

// the int16 products of vector v (32 values) against the q8 vector q: scale bytes sc times grid . q8 pairwise
static inline __m256i vdot(const __m128i * ix, int v, __m256i sc, __m256i q) {
    const uint64_t i4 = (v & 1) ? (uint64_t) _mm_extract_epi64(ix[v >> 1], 1) : (uint64_t) _mm_cvtsi128_si64(ix[v >> 1]);
    return _mm256_maddubs_epi16(sc, _mm256_sign_epi8(q, grid4(i4)));
}

static inline void prefetch(const void * x) {
    if (LLAMA_IQ1_PREFETCH > 0) {   // one line per block: blocks of 50 / 56 bytes, so every line of the row is asked for
        _mm_prefetch((const char *) ((uintptr_t) x + LLAMA_IQ1_PREFETCH), _MM_HINT_T0);
    }
}

static inline float fp16(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

// ggml's hsum_float_8, same order
static inline float hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// ggml's last line, hsum(main) + 0.125f * hsum(delta), compiled by clang as one vfmadd132ss
static inline float finish(__m256 main, __m256 delta) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(hsum(delta)), _mm_set_ss(0.125f), _mm_set_ss(hsum(main))));
}

// ---- IQ1_S: d, qs[32], qh[8] words (bits 0-11: index bits 8-10 of the group's 4 entries, 12-14: scale s, 15: delta sign)

struct prep_s {
    __m128i ix[4];
    __m256i ls;    // dword g: 2*s_g+1 in all four bytes
    __m256i dl;    // dword g: (bsums[2g] + bsums[2g+1]) * (2*s_g+1) * (bit 15 ? -1 : 1), ggml's int32 delta lane
};

static inline void prepare(const block_iq1_s * x, __m256i bsums, prep_s & p) {
    const __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *) x->qh));   // dword g = qh word g
    // entry k of group g takes bits 3k..3k+2 of w to byte k: t = w | w << 10 holds fields 0, 2 at bits 0, 16 and fields 1, 3 at 3, 19
    // (bits 16-31 of w are 0), so t << 5 holds fields 1, 3 at 8, 24
    const __m256i t = _mm256_or_si256(w, _mm256_slli_epi32(w, 10));
    const __m256i hi = _mm256_or_si256(_mm256_and_si256(t, _mm256_set1_epi32(0x00070007)), _mm256_and_si256(_mm256_slli_epi32(t, 5), _mm256_set1_epi32(0x07000700)));
    indices(x->qs, hi, p.ix);
    const __m256i l = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(w, 11), _mm256_set1_epi32(14)), _mm256_set1_epi32(1));
    p.ls = _mm256_shuffle_epi8(l, _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12));
    // (w << 16) | 1 is negative iff bit 15 and never 0
    p.dl = _mm256_sign_epi32(_mm256_madd_epi16(bsums, _mm256_and_si256(p.ls, _mm256_set1_epi16(0xFF))), _mm256_or_si256(_mm256_slli_epi32(w, 16), _mm256_set1_epi32(1)));
}

// block of row xa (and xb when two) against y: acc[0], acc[1] main and delta of a, acc[2], acc[3] of b
template <bool two>
static inline void block_s(const block_iq1_s * xa, const block_iq1_s * xb, const block_q8_K * y, __m256 * acc) {
    const __m256i bsums = _mm256_loadu_si256((const __m256i *) y->bsums);
    prep_s pa, pb;
    prepare(xa, bsums, pa);
    if constexpr (two) { prepare(xb, bsums, pb); }
    __m256i sa = _mm256_setzero_si256(), sb = _mm256_setzero_si256();
    _Pragma("GCC unroll 8")   // fully unrolled: index halves, scale and delta selects become constants (rolled clang keeps cmov + branches: -25%)
    for (int v = 0; v < 8; ++v) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) (y->qs + 32 * v));
        const __m256i sel = _mm256_set1_epi32(v);
        sa = _mm256_add_epi16(sa, vdot(pa.ix, v, _mm256_permutevar8x32_epi32(pa.ls, sel), q));
        if constexpr (two) { sb = _mm256_add_epi16(sb, vdot(pb.ix, v, _mm256_permutevar8x32_epi32(pb.ls, sel), q)); }
    }
    const __m256i one = _mm256_set1_epi16(1);
    const __m256 da = _mm256_set1_ps(y->d * fp16(xa->d));   // ggml: y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d)
    acc[0] = _mm256_fmadd_ps(da, _mm256_cvtepi32_ps(_mm256_madd_epi16(sa, one)), acc[0]);
    acc[1] = _mm256_fmadd_ps(da, _mm256_cvtepi32_ps(pa.dl), acc[1]);
    if constexpr (two) {
        const __m256 db = _mm256_set1_ps(y->d * fp16(xb->d));
        acc[2] = _mm256_fmadd_ps(db, _mm256_cvtepi32_ps(_mm256_madd_epi16(sb, one)), acc[2]);
        acc[3] = _mm256_fmadd_ps(db, _mm256_cvtepi32_ps(pb.dl), acc[3]);
    }
}

// ---- IQ1_M: qs[32], qh[16] (per entry a nibble: bits 0-2 index bits 8-10, bit 3 delta sign), scales[8] (4 words: 4 x 3-bit scale
//      per 16 values, top nibbles = fp16 block scale)

struct prep_m {
    __m128i ix[4];
    __m256i dlo, dhi;   // delta sign source per entry (0x78 + nibble: negative iff bit 3, never 0), entries 0-15 / 16-31 in both lanes
    __m256i sc;         // 2*s+1 per 16 values as int16, ggml's srlv layout (low lane: halves 2v, high lane: halves 2v+1)
    float   d;          // fp16 block scale
};

static inline void prepare(const block_iq1_m * x, prep_m & p) {
    const __m128i qh = _mm_loadu_si128((const __m128i *) x->qh);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(qh, m4), hi = _mm_and_si128(_mm_srli_epi16(qh, 4), m4);
    const __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));   // byte j = nibble of entry j
    indices(x->qs, _mm256_and_si256(nib, _mm256_set1_epi8(7)), p.ix);
    const __m256i ds = _mm256_add_epi8(nib, _mm256_set1_epi8(0x78));
    p.dlo = _mm256_permute2x128_si256(ds, ds, 0x00);
    p.dhi = _mm256_permute2x128_si256(ds, ds, 0x11);
    uint64_t s;
    memcpy(&s, x->scales, 8);
    const __m256i sh = _mm256_srlv_epi64(_mm256_set1_epi64x((long long) s), _mm256_set_epi64x(9, 3, 6, 0));
    p.sc = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(sh, _mm256_set1_epi16(7)), 1), _mm256_set1_epi16(1));
    p.d = fp16(iq1m_d16(s));
}

template <bool two>
static inline void block_m(const block_iq1_m * xa, const block_iq1_m * xb, const block_q8_K * y, __m256 * acc) {
    prep_m pa, pb;
    prepare(xa, pa);
    if constexpr (two) { prepare(xb, pb); }
    __m256i sa = _mm256_setzero_si256(), ta = _mm256_setzero_si256(), sb = _mm256_setzero_si256(), tb = _mm256_setzero_si256();
    _Pragma("GCC unroll 8")   // fully unrolled: index halves, scale and delta selects become constants (rolled clang keeps cmov + branches: -25%)
    for (int v = 0; v < 8; ++v) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) (y->qs + 32 * v));
        const __m256i ms = _mm256_load_si256((const __m256i *) tables.msel[v]);
        const __m256i dsel = _mm256_load_si256((const __m256i *) tables.dsel[v & 3]);
        const __m256i sca = _mm256_shuffle_epi8(pa.sc, ms);
        sa = _mm256_add_epi16(sa, vdot(pa.ix, v, sca, q));
        ta = _mm256_add_epi16(ta, _mm256_maddubs_epi16(sca, _mm256_sign_epi8(q, _mm256_shuffle_epi8(v < 4 ? pa.dlo : pa.dhi, dsel))));
        if constexpr (two) {
            const __m256i scb = _mm256_shuffle_epi8(pb.sc, ms);
            sb = _mm256_add_epi16(sb, vdot(pb.ix, v, scb, q));
            tb = _mm256_add_epi16(tb, _mm256_maddubs_epi16(scb, _mm256_sign_epi8(q, _mm256_shuffle_epi8(v < 4 ? pb.dlo : pb.dhi, dsel))));
        }
    }
    const __m256i one = _mm256_set1_epi16(1);
    const __m256 da = _mm256_set1_ps(y->d * pa.d);   // ggml: y[i].d * GGML_CPU_FP16_TO_FP32(scale.f16)
    acc[0] = _mm256_fmadd_ps(da, _mm256_cvtepi32_ps(_mm256_madd_epi16(sa, one)), acc[0]);
    acc[1] = _mm256_fmadd_ps(da, _mm256_cvtepi32_ps(_mm256_madd_epi16(ta, one)), acc[1]);
    if constexpr (two) {
        const __m256 db = _mm256_set1_ps(y->d * pb.d);
        acc[2] = _mm256_fmadd_ps(db, _mm256_cvtepi32_ps(_mm256_madd_epi16(sb, one)), acc[2]);
        acc[3] = _mm256_fmadd_ps(db, _mm256_cvtepi32_ps(_mm256_madd_epi16(tb, one)), acc[3]);
    }
}

} // namespace llama_iq1_fork

// ggml_vec_dot_t: s[0] = row vx (n IQ1_S values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq1_s_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq1_fork;
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_iq1_s * x = (const block_iq1_s *) vx;
    const block_q8_K  * y = (const block_q8_K *) vy;
    __m256 acc[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(x + i);
        f::block_s<false>(x + i, nullptr, y + i, acc);
    }
    *s = f::finish(acc[0], acc[1]);
}

// two rows (up and gate) against the same Q8_K input in one pass, each q8 vector and bsums loaded once: bit-identical to two single calls
static inline void llama_dot2_iq1_s_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq1_fork;
    assert(n % QK_K == 0);
    const block_iq1_s * xa = (const block_iq1_s *) vxa;
    const block_iq1_s * xb = (const block_iq1_s *) vxb;
    const block_q8_K  * y  = (const block_q8_K *) vy;
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(xa + i);
        f::prefetch(xb + i);
        f::block_s<true>(xa + i, xb + i, y + i, acc);
    }
    *s_a = f::finish(acc[0], acc[1]);
    *s_b = f::finish(acc[2], acc[3]);
}

// ggml_vec_dot_t: s[0] = row vx (n IQ1_M values) . vy (n Q8_K values); nrc == 1
static inline void llama_dot_iq1_m_q8_K_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    namespace f = llama_iq1_fork;
    assert(n % QK_K == 0 && nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;
    const block_iq1_m * x = (const block_iq1_m *) vx;
    const block_q8_K  * y = (const block_q8_K *) vy;
    __m256 acc[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(x + i);
        f::block_m<false>(x + i, nullptr, y + i, acc);
    }
    *s = f::finish(acc[0], acc[1]);
}

// two IQ1_M rows against the same Q8_K input in one pass, each q8 vector loaded once: bit-identical to two single calls
static inline void llama_dot2_iq1_m_q8_K_avx2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    namespace f = llama_iq1_fork;
    assert(n % QK_K == 0);
    const block_iq1_m * xa = (const block_iq1_m *) vxa;
    const block_iq1_m * xb = (const block_iq1_m *) vxb;
    const block_q8_K  * y  = (const block_q8_K *) vy;
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int i = 0; i < n / QK_K; ++i) {
        f::prefetch(xa + i);
        f::prefetch(xb + i);
        f::block_m<true>(xa + i, xb + i, y + i, acc);
    }
    *s_a = f::finish(acc[0], acc[1]);
    *s_b = f::finish(acc[2], acc[3]);
}

#endif // __AVX2__
