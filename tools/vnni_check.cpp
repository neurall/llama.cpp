// vnni_check LIB_A LIB_B [types...]: loads two ggml CPU backend libraries (e.g. libggml-cpu-haswell and libggml-cpu-zen4) in one process, quantizes random rows and
// runs each library's vec_dot on identical inputs: results must match bit for bit, then both are timed. Needs libggml-base next to the libraries.
//   Linux:   g++ -O2 -std=c++17 tools/vnni_check.cpp -Iggml/include -Iggml/src -ldl -o vnni_check
//   Windows: clang++ -O2 -std=c++17 tools/vnni_check.cpp -Iggml/include -Iggml/src -o vnni_check.exe
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
static void * load_lib(const char * p) { return (void *) LoadLibraryA(p); }
static void * get_sym(void * h, const char * n) { return (void *) GetProcAddress((HMODULE) h, n); }
#else
#include <dlfcn.h>
static void * load_lib(const char * p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
static void * get_sym(void * h, const char * n) { return dlsym(h, n); }
#endif

typedef const struct ggml_type_traits_cpu * (*traits_fn)(enum ggml_type);
typedef void (*void_fn)(void);
typedef void (*qinit_fn)(enum ggml_type);
typedef size_t (*qchunk_fn)(enum ggml_type, const float *, void *, int64_t, int64_t, int64_t, const float *);
typedef bool (*qreq_fn)(enum ggml_type);
typedef size_t (*rowsize_fn)(enum ggml_type, int64_t);

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s LIB_A LIB_B [type names...]\n", argv[0]); return 2; }
    void * la = load_lib(argv[1]); void * lb = load_lib(argv[2]);
    if (!la || !lb) { fprintf(stderr, "cannot load the libraries\n"); return 2; }
    auto ta = (traits_fn) get_sym(la, "ggml_get_type_traits_cpu"); auto tb = (traits_fn) get_sym(lb, "ggml_get_type_traits_cpu");
    auto ia = (void_fn) get_sym(la, "ggml_cpu_init");              auto ib = (void_fn) get_sym(lb, "ggml_cpu_init");
    if (!ta || !tb) { fprintf(stderr, "ggml_get_type_traits_cpu not exported\n"); return 2; }
    if (ia) ia();
    if (ib) ib();
    // quantization from ggml-base (loaded through the library's dependency)
#ifdef _WIN32
    void * base = (void *) GetModuleHandleA("ggml-base.dll");
#else
    void * base = load_lib("libggml-base.so");
#endif
    auto qinit = (qinit_fn) get_sym(base, "ggml_quantize_init");  auto qchunk = (qchunk_fn) get_sym(base, "ggml_quantize_chunk");
    auto qreq = (qreq_fn) get_sym(base, "ggml_quantize_requires_imatrix"); auto rowsize = (rowsize_fn) get_sym(base, "ggml_row_size");
    if (!qinit || !qchunk || !qreq || !rowsize) { fprintf(stderr, "ggml-base symbols missing\n"); return 2; }

    struct T { const char * name; enum ggml_type t; };
    std::vector<T> all = { {"q4_0",GGML_TYPE_Q4_0},{"q5_0",GGML_TYPE_Q5_0},{"q8_0",GGML_TYPE_Q8_0},{"q2_K",GGML_TYPE_Q2_K},{"q3_K",GGML_TYPE_Q3_K},{"q4_K",GGML_TYPE_Q4_K},
        {"q5_K",GGML_TYPE_Q5_K},{"q6_K",GGML_TYPE_Q6_K},{"iq2_xxs",GGML_TYPE_IQ2_XXS},{"iq2_xs",GGML_TYPE_IQ2_XS},{"iq2_s",GGML_TYPE_IQ2_S},{"iq3_xxs",GGML_TYPE_IQ3_XXS},
        {"iq3_s",GGML_TYPE_IQ3_S},{"iq1_s",GGML_TYPE_IQ1_S},{"iq1_m",GGML_TYPE_IQ1_M},{"iq4_nl",GGML_TYPE_IQ4_NL},{"iq4_xs",GGML_TYPE_IQ4_XS},{"q2_0",GGML_TYPE_Q2_0} };
    std::vector<T> sel;
    for (int i = 3; i < argc; ++i) for (auto & x : all) if (!strcmp(argv[i], x.name)) sel.push_back(x);
    if (sel.empty()) sel = all;

    const int64_t n = 8192, nrows = 256;
    std::mt19937 rng(12345); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src((size_t) n * nrows), imat(n, 1.0f), ysrc(n);
    for (auto & v : src) v = nd(rng);
    for (auto & v : ysrc) v = nd(rng);
    int bad = 0;
    printf("%-8s %-9s %12s %12s %8s\n", "type", "bitexact", "A GB/s", "B GB/s", "B/A");
    for (auto & ty : sel) {
        qinit(ty.t);
        const auto * tra = ta(ty.t); const auto * trb = tb(ty.t);
        if (!tra || !trb || !tra->vec_dot || !trb->vec_dot) { printf("%-8s no vec_dot\n", ty.name); continue; }
        const enum ggml_type yt = tra->vec_dot_type;
        qinit(yt);
        std::vector<uint8_t> xq(rowsize(ty.t, n) * nrows), yq(rowsize(yt, n));
        qchunk(ty.t, src.data(), xq.data(), 0, nrows, n, qreq(ty.t) ? imat.data() : nullptr);
        ta(yt)->from_float(ysrc.data(), yq.data(), n);
        const size_t rs = rowsize(ty.t, n);
        bool exact = true;
        std::vector<float> ra(nrows), rb(nrows);
        for (int r = 0; r < nrows; ++r) {
            tra->vec_dot((int) n, &ra[r], 0, xq.data() + r*rs, 0, yq.data(), 0, 1);
            trb->vec_dot((int) n, &rb[r], 0, xq.data() + r*rs, 0, yq.data(), 0, 1);
            if (memcmp(&ra[r], &rb[r], sizeof(float)) != 0) exact = false;
        }
        auto bench = [&](const struct ggml_type_traits_cpu * tr) {
            float s; const int reps = 200;
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < reps; ++k) for (int r = 0; r < nrows; ++r) tr->vec_dot((int) n, &s, 0, xq.data() + r*rs, 0, yq.data(), 0, 1);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return (double) reps * nrows * rs / sec / 1e9;
        };
        bench(tra); bench(trb);                       // warm
        // best of several interleaved blocks: other load on the machine (a browser) only ever slows a block down
        double ga = 0, gb = 0;
        const int blocks = getenv("VNNI_BLOCKS") ? atoi(getenv("VNNI_BLOCKS")) : 7;
        for (int k = 0; k < blocks; ++k) { ga = std::max(ga, bench(tra)); gb = std::max(gb, bench(trb)); }
        printf("%-8s %-9s %12.2f %12.2f %8.3f\n", ty.name, exact ? "yes" : "NO", ga, gb, gb/ga);
        if (!exact) bad++;
    }
    // the activation quantization (float -> Q8_0) that runs once per token before every dot of prompt processing: speed in GB/s of float input
    {
        const auto * qa = ta(GGML_TYPE_Q8_0); const auto * qb = tb(GGML_TYPE_Q8_0);
        if (qa && qb && qa->from_float && qb->from_float) {
            qinit(GGML_TYPE_Q8_0);
            std::vector<uint8_t> oa(rowsize(GGML_TYPE_Q8_0, n)), ob(rowsize(GGML_TYPE_Q8_0, n));
            bool exact = true;
            for (int r = 0; r < nrows; ++r) {
                qa->from_float(src.data() + (size_t) r*n, oa.data(), n); qb->from_float(src.data() + (size_t) r*n, ob.data(), n);
                if (memcmp(oa.data(), ob.data(), oa.size()) != 0) exact = false;
            }
            auto benchq = [&](const struct ggml_type_traits_cpu * tr) {
                const int reps = 200;
                auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < reps; ++k) for (int r = 0; r < nrows; ++r) tr->from_float(src.data() + (size_t) r*n, oa.data(), n);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                return (double) reps * nrows * n * sizeof(float) / sec / 1e9;
            };
            benchq(qa); benchq(qb);
            double ga = 0, gb = 0;
            const int blocks = getenv("VNNI_BLOCKS") ? atoi(getenv("VNNI_BLOCKS")) : 7;
            for (int k = 0; k < blocks; ++k) { ga = std::max(ga, benchq(qa)); gb = std::max(gb, benchq(qb)); }
            printf("%-8s %-9s %12.2f %12.2f %8.3f   (float -> Q8_0, GB/s of float input)\n", "quantq8", exact ? "yes" : "NO", ga, gb, gb/ga);
            if (!exact) bad++;
        }
    }
    printf(bad ? "MISMATCH in %d type(s)\n" : "all bit-exact\n", bad);
    return bad ? 1 : 0;
}
