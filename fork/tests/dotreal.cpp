// Dot products on REAL quantized weights: rows of a tensor of the given type from a GGUF file, against the plain C reference (ggml_vec_dot_*_generic) and the form chosen by
// GGML_CPU_KERNEL_FORM (default: the tuner's pick). Shows the largest relative difference, so a kernel that is exact on random data but not on real weights is caught.
// Usage: dotreal model.gguf TYPE   (TYPE: iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s)
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "../../ggml/src/ggml-cpu/quants.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: dotreal model.gguf TYPE\n"); return 1; }
    ggml_cpu_init();
    struct { const char * name; ggml_type t; ggml_vec_dot_t ref; } T[] = {
        { "iq1_s", GGML_TYPE_IQ1_S, ggml_vec_dot_iq1_s_q8_K_generic }, { "iq1_m", GGML_TYPE_IQ1_M, ggml_vec_dot_iq1_m_q8_K_generic },
        { "iq2_xxs", GGML_TYPE_IQ2_XXS, ggml_vec_dot_iq2_xxs_q8_K_generic }, { "iq2_xs", GGML_TYPE_IQ2_XS, ggml_vec_dot_iq2_xs_q8_K_generic },
        { "iq2_s", GGML_TYPE_IQ2_S, ggml_vec_dot_iq2_s_q8_K_generic }, { "iq3_xxs", GGML_TYPE_IQ3_XXS, ggml_vec_dot_iq3_xxs_q8_K_generic },
        { "iq3_s", GGML_TYPE_IQ3_S, ggml_vec_dot_iq3_s_q8_K_generic } };
    ggml_type type = GGML_TYPE_COUNT; ggml_vec_dot_t ref = nullptr;
    for (auto & e : T) { if (!strcmp(e.name, argv[2])) { type = e.t; ref = e.ref; } }
    if (!ref) { fprintf(stderr, "unknown type\n"); return 1; }
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    FILE * f = fopen(argv[1], "rb");
    const auto * tr = ggml_get_type_traits_cpu(type); const auto * q8 = ggml_get_type_traits_cpu(tr->vec_dot_type);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    double worst = 0; long rows = 0, diffrows = 0; int used = 0;
    for (int64_t id = 0; id < gguf_get_n_tensors(g) && used < 6; ++id) {
        if (gguf_get_tensor_type(g, id) != type) { continue; }
        const char * name = gguf_get_tensor_name(g, id);
        const int64_t n = 2048;   // row length (OLMoE: 2048 or 1024 columns; use a multiple of 256 that every tensor has)
        const size_t rb = ggml_row_size(type, n);
        const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
        std::vector<uint8_t> w(rb * 256); fseek(f, (long) off, SEEK_SET);
        if (fread(w.data(), 1, w.size(), f) != w.size()) { continue; }
        std::vector<float> act(n); for (auto & v : act) v = nd(rng);
        std::vector<uint8_t> a8(ggml_row_size(tr->vec_dot_type, n)); q8->from_float(act.data(), a8.data(), n);
        double tw = 0; long td = 0;
        for (int r = 0; r < 256; ++r) {
            float s = 0, s0 = 0;
            tr->vec_dot(n, &s, 0, w.data() + r * rb, 0, a8.data(), 0, 1);
            ref(n, &s0, 0, w.data() + r * rb, 0, a8.data(), 0, 1);
            const double rel = fabs((double) s - s0) / (fabs((double) s0) + 1e-3);
            if (rel > tw) tw = rel;
            if (rel > 1e-6) ++td;
        }
        printf("%-28s max rel diff %.2e, %ld of 256 rows differ\n", name, tw, td);
        worst = std::max(worst, tw); rows += 256; diffrows += td; ++used;
    }
    printf("%s: %d tensors, worst %.2e, %ld of %ld rows differ from the C reference\n", argv[2], used, worst, diffrows, rows);
    return 0;
}
