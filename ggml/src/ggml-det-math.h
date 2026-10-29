#pragma once

// Deterministic float math shared by the CPU and CUDA backends (GGML_CPU_GPU_SHAPE): only fmaf, single multiplies/adds, IEEE division and integer bit
// tricks, so both sides give the same bits. CUDA's fast-math expf/logf/division and the CPU's vectorized approximations cannot match each other.
// Accuracy is about 1 ulp, enough for the model; results below FLT_MIN flush to 0 on both sides (the GPU flushes denormals anyway).

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define GGML_DET_FN static __host__ __device__ __forceinline__
#else
#define GGML_DET_FN static inline
#endif

// IEEE division: CUDA's fast-math "/" is an approximation, so the device side asks for the rounded one
GGML_DET_FN float ggml_det_div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
#define GGML_DET_DIV(a, b) ggml_det_div((a), (b))

GGML_DET_FN uint32_t ggml_det_f2u(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
GGML_DET_FN float    ggml_det_u2f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

GGML_DET_FN float ggml_det_expf(float x) {
    if (!(x > -87.33f)) return x != x ? x : 0.0f;
    if (x > 88.72f) return INFINITY;
    const float n = rintf(x * 1.44269504088896341f);
    float r = fmaf(n, -0.693145751953125f, x);          // ln2 split in a high part with few bits (exact products) and a low part
    r = fmaf(n, -1.428606765330187e-06f, r);
    float p = 1.0f/5040.0f;
    p = fmaf(p, r, 1.0f/720.0f);
    p = fmaf(p, r, 1.0f/120.0f);
    p = fmaf(p, r, 1.0f/24.0f);
    p = fmaf(p, r, 1.0f/6.0f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    const float res = p * ggml_det_u2f((uint32_t) ((int) n + 127) << 23);
    return res < 1.17549435e-38f ? 0.0f : res;
}

GGML_DET_FN float ggml_det_logf(float x) {
    if (!(x > 0.0f)) return x == 0.0f ? -INFINITY : NAN;
    if (x == INFINITY) return x;
    uint32_t u = ggml_det_f2u(x);
    int e = (int) (u >> 23) - 127;
    if ((u >> 23) == 0) return -87.33f;                  // denormal input: flushed on the GPU, not needed here
    float m = ggml_det_u2f((u & 0x7fffffu) | 0x3f800000u);
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }
    const float f = m - 1.0f;
    const float s = GGML_DET_DIV(f, 2.0f + f);
    const float s2 = s * s;
    float q = 2.0f/11.0f;
    q = fmaf(q, s2, 2.0f/9.0f);
    q = fmaf(q, s2, 2.0f/7.0f);
    q = fmaf(q, s2, 2.0f/5.0f);
    q = fmaf(q, s2, 2.0f/3.0f);
    q = fmaf(q, s2, 2.0f);
    const float lg = s * q;
    const float fe = (float) e;
    return fmaf(fe, 0.693145751953125f, fmaf(fe, 1.428606765330187e-06f, lg));
}

GGML_DET_FN float ggml_det_log1pf(float y) {
    const float u = 1.0f + y;
    const float c = y - (u - 1.0f);
    return ggml_det_logf(u) + GGML_DET_DIV(c, u);
}

GGML_DET_FN float ggml_det_sigmoid(float x) { return GGML_DET_DIV(1.0f, 1.0f + ggml_det_expf(-x)); }
GGML_DET_FN float ggml_det_silu(float x)    { return GGML_DET_DIV(x, 1.0f + ggml_det_expf(-x)); }
GGML_DET_FN float ggml_det_softplus(float x){ return x > 20.0f ? x : ggml_det_log1pf(ggml_det_expf(x)); }
