// fork: the bell pool's IQ2_S dots (llama-iq2s-avx2-fork.h) built with AVX2 and FMA whatever the rest of libllama is built for
// (src/CMakeLists.txt gives this file its own flags). llama_bell_iq2s_fork hands them out only on a CPU that has both.
#include "llama-iq2s-avx2-fork.h"

#include <cstddef>

extern "C" {
typedef void (*llama_bell_dot_fn)(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
typedef void (*llama_bell_dot2_fn)(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy);
bool llama_bell_iq2s_fork(llama_bell_dot_fn * dot, llama_bell_dot2_fn * dot2);
}

#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
static void bell_dot(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    llama_dot_iq2_s_q8_K_avx2(n, s, bs, vx, bx, vy, by, nrc);
}
static void bell_dot2(int n, float * s_a, float * s_b, const void * vxa, const void * vxb, const void * vy) {
    llama_dot2_iq2_s_q8_K_avx2(n, s_a, s_b, vxa, vxb, vy);
}
bool llama_bell_iq2s_fork(llama_bell_dot_fn * dot, llama_bell_dot2_fn * dot2) {
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        return false;
    }
    *dot  = bell_dot;
    *dot2 = bell_dot2;
    return true;
}
#else
// shortcut: no CPU check without the GCC/clang builtin (MSVC): the CPU backend's own dot stays; add a cpuid call when a Windows build shows the phase
bool llama_bell_iq2s_fork(llama_bell_dot_fn *, llama_bell_dot2_fn *) {
    return false;
}
#endif
