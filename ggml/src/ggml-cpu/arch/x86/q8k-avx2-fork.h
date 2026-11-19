// fork: AVX2 quantize_row_q8_K, byte-identical to quantize_row_q8_K_ref (ggml/src/ggml-quants.c) for finite input.
// x86 quantize_row_q8_K called the scalar ref: 2.5 us for 2560 floats on Zen 2, on the critical path of every decode layer whose experts run on the CPU.
// C and C++; the includer declares block_q8_K and QK_K (ggml-common.h). Test: tests/test-q8k-avx2-fork.cpp.
// NaN/Inf lanes: the max search skips NaN as the ref does, but those lanes' bytes differ (the ref's nearest_int asserts on them in debug builds).
#pragma once

#if defined(__AVX2__)
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <string.h>

static inline void quantize_row_q8_K_avx2_fork(const float * x, block_q8_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const __m256  sign   = _mm256_set1_ps(-0.0f);
    const __m256i perm   = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int64_t i = 0; i < k / QK_K; i++, x += QK_K) {
        // |x| as the first operand: max_ps returns the second one for a NaN, so NaN is skipped like the ref's `ax > amax`
        __m256 m0 = _mm256_setzero_ps(), m1 = m0, m2 = m0, m3 = m0;
        for (int j = 0; j < QK_K; j += 32) {
            m0 = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x + j +  0)), m0);
            m1 = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x + j +  8)), m1);
            m2 = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x + j + 16)), m2);
            m3 = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x + j + 24)), m3);
        }
        const __m256 m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
        __m128 h = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);

        if (amax == 0) {
            y[i].d = 0;
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums)); // the ref leaves them unwritten
            continue;
        }

        // the ref's strict `>` scan keeps the first x with |x| == amax; amax is one of the |x|, so the scan stops inside the block
        const __m256 va = _mm256_set1_ps(amax);
        int j = 0, eq;
        while (!(eq = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)), va, _CMP_EQ_OQ)))) {
            j += 8;
        }
        const float max = (_mm256_movemask_ps(_mm256_loadu_ps(x + j)) & eq & -eq) ? -amax : amax; // sign of the lowest matching lane

        const float iscale = -127.f/max;
        y[i].d = 1/iscale;
        if (isinf(iscale)) {
            // amax < 127/FLT_MAX: the ref's magic add turns inf and inf*0 into ints that truncate to 0, cvtps would give -128
            memset(y[i].qs, 0, sizeof(y[i].qs));
            memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }

        // |iscale*x| <= 127 (+ rounding), so cvtps (round to nearest even) equals the ref's magic add and the packs never saturate
        const __m256 mul = _mm256_set1_ps(iscale);
        __m256i s[QK_K/32];
        for (j = 0; j < QK_K; j += 32) {
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(mul, _mm256_loadu_ps(x + j +  0)));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(mul, _mm256_loadu_ps(x + j +  8)));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(mul, _mm256_loadu_ps(x + j + 16)));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(mul, _mm256_loadu_ps(x + j + 24)));
            // packs work per 128-bit lane; the permute puts the 32 bytes back in order
            const __m256i q = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3)), perm);
            _mm256_storeu_si256((__m256i *)(y[i].qs + j), q);
            s[j/32] = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16); // sums of 4: low lane in group j/16, high lane in j/16+1
        }
        const __m256i b0 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[0], s[1]), _mm256_hadd_epi32(s[2], s[3])); // 0 2 4 6 | 1 3 5 7
        const __m256i b1 = _mm256_hadd_epi32(_mm256_hadd_epi32(s[4], s[5]), _mm256_hadd_epi32(s[6], s[7])); // 8 10 12 14 | 9 11 13 15
        const __m256i b  = _mm256_packs_epi32(b0, b1); // 0 2 .. 14 | 1 3 .. 15, |sum| <= 16*127 fits
        const __m128i lo = _mm256_castsi256_si128(b), hi = _mm256_extracti128_si256(b, 1);
        _mm_storeu_si128((__m128i *)(y[i].bsums + 0), _mm_unpacklo_epi16(lo, hi));
        _mm_storeu_si128((__m128i *)(y[i].bsums + 8), _mm_unpackhi_epi16(lo, hi));
    }
}
#endif // __AVX2__
