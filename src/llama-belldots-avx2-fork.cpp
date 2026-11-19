// fork: the bell pool's own dot kernels (llama-iq2s-, -iq2x-, -iq3-, -iq4-, -kq-, -iq1-avx2-fork.h) built with AVX2, FMA and F16C whatever the rest of
// libllama is built for (src/CMakeLists.txt gives this file its own flags). llama_bell_dots_fork hands them out only on a CPU that has all three.
// Each is bit-identical to the AVX2 form of the CPU backend's vec_dot for its type (tests/test-*-avx2-fork.cpp). One thread, cold weights, Zen 2,
// against the backend's own dot (2026-11-19): IQ3_XXS +46%, IQ3_S +44%, IQ2_XXS +47%, IQ2_XS +57%, Q2_K +11%, Q3_K +18%, Q4_K +16%, IQ4_XS +6%;
// two rows in one pass: IQ4_NL +9 to 19%, Q2_0 +19 to 23%, MXFP4 +8 to 20% (their single rows only match the backend's, so those stay the backend's).
// IQ1_S and IQ1_M: 25 to 35 times the backend's default form on Zen 1 / Zen 2 (it builds its grid indices with pdep, microcoded there), 1.5 times its
// form without pdep. Q8_0 has no kernel here (a pair was 5% slower than the backend's two rows).
#include "llama-iq2s-avx2-fork.h"
#include "llama-iq2x-avx2-fork.h"
#include "llama-iq3-avx2-fork.h"
#include "llama-iq4-avx2-fork.h"
#include "llama-kq-avx2-fork.h"
#include "llama-iq1-avx2-fork.h"

#include "ggml.h"

#include <cstddef>

extern "C" {
typedef void (*llama_bell_dot_fn)(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
typedef void (*llama_bell_dot2_fn)(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy);
// the kernels for weight type `type` against activation type *act (the one the caller must have quantized to). *dot: one row (nullptr: the backend's
// stays), *dot2: two rows against one activation row (nullptr: none). false: nothing for this type or this CPU
bool llama_bell_dots_fork(int type, int * act, llama_bell_dot_fn * dot, llama_bell_dot2_fn * dot2);
}

#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define BELL_DOT(name, fn)  static void name(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) { fn(n, s, bs, vx, bx, vy, by, nrc); }
#define BELL_DOT2(name, fn) static void name(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) { fn(n, s_a, s_b, vxa, vxb, vy); }
BELL_DOT (d_iq2_s,    llama_dot_iq2_s_q8_K_avx2)    BELL_DOT2(d2_iq2_s,   llama_dot2_iq2_s_q8_K_avx2)
BELL_DOT (d_iq2_xxs,  llama_dot_iq2_xxs_q8_K_avx2)  BELL_DOT2(d2_iq2_xxs, llama_dot2_iq2_xxs_q8_K_avx2)
BELL_DOT (d_iq2_xs,   llama_dot_iq2_xs_q8_K_avx2)   BELL_DOT2(d2_iq2_xs,  llama_dot2_iq2_xs_q8_K_avx2)
BELL_DOT (d_iq3_xxs,  llama_dot_iq3_xxs_q8_K_avx2)  BELL_DOT2(d2_iq3_xxs, llama_dot2_iq3_xxs_q8_K_avx2)
BELL_DOT (d_iq3_s,    llama_dot_iq3_s_q8_K_avx2)    BELL_DOT2(d2_iq3_s,   llama_dot2_iq3_s_q8_K_avx2)
BELL_DOT (d_iq1_s,    llama_dot_iq1_s_q8_K_avx2)    BELL_DOT2(d2_iq1_s,   llama_dot2_iq1_s_q8_K_avx2)
BELL_DOT (d_iq1_m,    llama_dot_iq1_m_q8_K_avx2)    BELL_DOT2(d2_iq1_m,   llama_dot2_iq1_m_q8_K_avx2)
BELL_DOT (d_iq4_xs,   llama_dot_iq4_xs_q8_K_avx2)   BELL_DOT2(d2_iq4_xs,  llama_dot2_iq4_xs_q8_K_avx2)
BELL_DOT (d_q2_K,     llama_dot_q2_K_q8_K_avx2)
BELL_DOT (d_q3_K,     llama_dot_q3_K_q8_K_avx2)
BELL_DOT (d_q4_K,     llama_dot_q4_K_q8_K_avx2)
BELL_DOT2(d2_iq4_nl,  llama_dot2_iq4_nl_q8_0_avx2)
BELL_DOT2(d2_mxfp4,   llama_dot2_mxfp4_q8_0_avx2)
BELL_DOT2(d2_q2_0,    llama_dot2_q2_0_q8_0_avx2)

bool llama_bell_dots_fork(int type, int * act, llama_bell_dot_fn * dot, llama_bell_dot2_fn * dot2) {
    static const bool cpu = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    *dot = nullptr; *dot2 = nullptr;
    if (!cpu) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ2_S:   *act = GGML_TYPE_Q8_K; *dot = d_iq2_s;   *dot2 = d2_iq2_s;   return true;
        case GGML_TYPE_IQ2_XXS: *act = GGML_TYPE_Q8_K; *dot = d_iq2_xxs; *dot2 = d2_iq2_xxs; return true;
        case GGML_TYPE_IQ2_XS:  *act = GGML_TYPE_Q8_K; *dot = d_iq2_xs;  *dot2 = d2_iq2_xs;  return true;
        case GGML_TYPE_IQ3_XXS: *act = GGML_TYPE_Q8_K; *dot = d_iq3_xxs; *dot2 = d2_iq3_xxs; return true;
        case GGML_TYPE_IQ3_S:   *act = GGML_TYPE_Q8_K; *dot = d_iq3_s;   *dot2 = d2_iq3_s;   return true;
        case GGML_TYPE_IQ1_S:   *act = GGML_TYPE_Q8_K; *dot = d_iq1_s;   *dot2 = d2_iq1_s;   return true;
        case GGML_TYPE_IQ1_M:   *act = GGML_TYPE_Q8_K; *dot = d_iq1_m;   *dot2 = d2_iq1_m;   return true;
        case GGML_TYPE_IQ4_XS:  *act = GGML_TYPE_Q8_K; *dot = d_iq4_xs;  *dot2 = d2_iq4_xs;  return true;
        case GGML_TYPE_Q2_K:    *act = GGML_TYPE_Q8_K; *dot = d_q2_K;    return true;   // a pair gains nothing over two rows (Q2_K, Q3_K, Q4_K)
        case GGML_TYPE_Q3_K:    *act = GGML_TYPE_Q8_K; *dot = d_q3_K;    return true;
        case GGML_TYPE_Q4_K:    *act = GGML_TYPE_Q8_K; *dot = d_q4_K;    return true;
        case GGML_TYPE_IQ4_NL:  *act = GGML_TYPE_Q8_0; *dot2 = d2_iq4_nl; return true;
        case GGML_TYPE_MXFP4:   *act = GGML_TYPE_Q8_0; *dot2 = d2_mxfp4;  return true;
        case GGML_TYPE_Q2_0:    *act = GGML_TYPE_Q8_0; *dot2 = d2_q2_0;   return true;
        default:                return false;
    }
}
#else
// shortcut: no CPU check without the GCC/clang builtin (MSVC): the CPU backend's own dots stay; add a cpuid call when a Windows build shows the phase
bool llama_bell_dots_fork(int, int *, llama_bell_dot_fn * dot, llama_bell_dot2_fn * dot2) {
    *dot = nullptr; *dot2 = nullptr;
    return false;
}
#endif
