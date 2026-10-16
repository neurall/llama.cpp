// The AVX2 form of the x86 dot products (AVX-512 and VNNI code generation switched off) next to the default form, in the same library: quants.c compiled a second time
// with every exported function renamed *_256 (quants-avx2-form-rename.h) and the flags of this one source file lowered (CMakeLists.txt). A start-up timing per quant type
// picks the faster of the two for each type (ggml_cpu_tune_vec_dot in ggml-cpu.c). Measured on Zen 4 (ns per 256 weights in cache, default / AVX2): Q4_K 9.3 / 5.4,
// Q2_K 4.5 / 4.3, Q3_K 6.8 / 6.3, but Q6_K 5.3 / 5.9, Q8_0 5.7 / 6.4, Q4_0 7.4 / 8.3, IQ2_XXS 17.6 / 21.4: neither form wins everywhere.
#if defined(GGML_CPU_AVX2_FORM)
#include "quants-avx2-form-rename.h"
#include "quants.c"
#endif
