// accuracy of every quant type's vec_dot against a double-precision reference (dequantized weights x the Q8 activations as the kernel sees them),
// built against two ggml-cpu variants to compare: checksum over all rows (identical = same sums to the last bit) and worst relative difference to the reference
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    ggml_cpu_init();
    const int n = 4096, rows = 512;
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> act(n), w((size_t) rows * n), imp(n, 1.0f);
    for (auto & v : act) v = nd(rng);
    for (auto & v : w) v = nd(rng) * 0.05f;
    const ggml_type types[] = { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL,
                                GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_MXFP4 };
    for (ggml_type t : types) {
        const auto * tr = ggml_get_type_traits(t); const auto * trc = ggml_get_type_traits_cpu(t); const auto * q8 = ggml_get_type_traits_cpu(trc->vec_dot_type);
        ggml_quantize_init(t);
        const size_t rb = ggml_row_size(t, n);
        std::vector<uint8_t> q(rb * rows); ggml_quantize_chunk(t, w.data(), q.data(), 0, rows, n, imp.data());
        std::vector<uint8_t> a8(ggml_row_size(trc->vec_dot_type, n)); q8->from_float(act.data(), a8.data(), n);
        std::vector<float> deq(n);
        const std::vector<float> & aq = act;   // the float activations: the Q8 rounding is the same in every build compared, the checksum shows it
        double maxrel = 0, cs = 0; unsigned long long bits = 0;
        for (int r = 0; r < rows; ++r) {
            float s = 0; trc->vec_dot(n, &s, 0, q.data() + r * rb, 0, a8.data(), 0, 1);
            tr->to_float(q.data() + r * rb, deq.data(), n);
            double ref = 0; for (int i = 0; i < n; ++i) ref += (double) deq[i] * aq[i];
            const double rel = std::fabs(s - ref) / (std::fabs(ref) + 1e-2); if (rel > maxrel) maxrel = rel;
            cs += s; uint32_t u; memcpy(&u, &s, 4); bits = bits * 1099511628211ULL + u;
        }
        printf("%-8s max rel to reference %.2e  sum %.6f  bits %016llx\n", ggml_type_name(t), maxrel, cs, bits);
    }
}
