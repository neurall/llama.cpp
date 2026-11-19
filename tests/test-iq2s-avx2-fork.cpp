// src/llama-iq2s-avx2-fork.h against ggml's own IQ2_S x Q8_K dot product (the CPU backend's vec_dot), bit for bit (memcmp of the float).
// Loads only the CPU backend variant (ggml_backend_load_all_from_path would start CUDA too). One thread: OMP_NUM_THREADS is forced to 1
// (ggml_quantize_init(IQ2_S) runs an OpenMP search over every core otherwise). From the repo root:
//   B=/p/bw/rels/devbuilds/bell_one_fence OUT=/tmp/test-iq2s-avx2-fork
//   clang++ -std=c++17 -O2 -mavx2 -mfma -mf16c -I src -I ggml/include -I ggml/src tests/test-iq2s-avx2-fork.cpp -o $OUT -L $B -lggml -lggml-base -Wl,-rpath,$B
//   $OUT [build dir, default $B] [cpu variant, default haswell]
#include "llama-iq2s-avx2-fork.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(__AVX2__)
int main() { printf("skip: built without AVX2\n"); return 0; }
#else

static ggml_vec_dot_t    ref_dot;
static ggml_from_float_t to_q8k;

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint16_t finite_half(std::mt19937 & rng) {   // any fp16 bit pattern but Inf/NaN (zeros and subnormals included)
    for (;;) { const uint16_t h = (uint16_t) rng(); if (((h >> 10) & 31) != 31) { return h; } }
}

// one case: rows a and b against q; compares single and dual calls with ggml, returns false on the first mismatch
static bool check(const char * what, int n, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const std::vector<uint8_t> & q) {
    float ra, rb, sa, sb, da, db;
    ref_dot(n, &ra, 0, a.data(), 0, q.data(), 0, 1);
    ref_dot(n, &rb, 0, b.data(), 0, q.data(), 0, 1);
    llama_dot_iq2_s_q8_K_avx2(n, &sa, 0, a.data(), 0, q.data(), 0, 1);
    llama_dot_iq2_s_q8_K_avx2(n, &sb, 0, b.data(), 0, q.data(), 0, 1);
    llama_dot2_iq2_s_q8_K_avx2(n, &da, &db, a.data(), b.data(), q.data());
    const float got[4] = { sa, sb, da, db }, want[4] = { ra, rb, ra, rb };
    static const char * names[4] = { "single a", "single b", "dual a", "dual b" };
    for (int k = 0; k < 4; ++k) {
        if (memcmp(&got[k], &want[k], 4) != 0) {
            printf("MISMATCH %s n=%d %s: ggml %.9g (0x%08x) fork %.9g (0x%08x)\n", what, n, names[k], want[k], bits(want[k]), got[k], bits(got[k]));
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    setenv("OMP_NUM_THREADS", "1", 1);   // before the first OpenMP region (libomp reads it then)
    const std::string dir = argc > 1 ? argv[1] : "/p/bw/rels/devbuilds/bell_one_fence";
    const std::string lib = dir + "/libggml-cpu-" + (argc > 2 ? argv[2] : "haswell") + ".so";
    ggml_backend_reg_t reg = ggml_backend_load(lib.c_str());
    auto traits = reg ? (const ggml_type_traits_cpu * (*)(ggml_type)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_type_traits") : nullptr;
    if (!traits) { printf("FAIL: no CPU backend type traits from %s\n", lib.c_str()); return 1; }
    ref_dot = traits(GGML_TYPE_IQ2_S)->vec_dot;
    to_q8k  = traits(GGML_TYPE_Q8_K)->from_float;
    if (!ref_dot || !to_q8k || traits(GGML_TYPE_IQ2_S)->vec_dot_type != GGML_TYPE_Q8_K) { printf("FAIL: IQ2_S traits\n"); return 1; }
    ggml_quantize_init(GGML_TYPE_IQ2_S);

    int cases = 0;
    for (const int n : { 256, 2560, 4096 }) {
        const size_t wb = ggml_row_size(GGML_TYPE_IQ2_S, n), qb = ggml_row_size(GGML_TYPE_Q8_K, n);
        const int nb = n / QK_K;
        std::vector<uint8_t> a(wb), b(wb), q(qb);
        std::vector<float> f(n);
        for (uint32_t seed = 1; seed <= 4; ++seed) {
            std::mt19937 rng(seed * 7919 + n);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            const float wscale[4] = { 1.0f, 1e-3f, 30.0f, 0.02f };
            auto quant_w = [&](std::vector<uint8_t> & w) {   // normal weights through ggml's own IQ2_S quantizer
                for (float & v : f) { v = wscale[seed - 1] * nd(rng); }
                ggml_quantize_chunk(GGML_TYPE_IQ2_S, f.data(), w.data(), 0, 1, n, nullptr);
            };
            auto raw_w = [&](std::vector<uint8_t> & w) {     // random bytes: every grid index, sign and scale, any finite d
                for (uint8_t & v : w) { v = (uint8_t) rng(); }
                for (int i = 0; i < nb; ++i) { ((block_iq2_s *) w.data())[i].d = finite_half(rng); }
            };
            auto normal_q = [&](float sd) { for (float & v : f) { v = sd * nd(rng); } to_q8k(f.data(), q.data(), n); };
            auto raw_q = [&]() {                              // q8 bytes incl. -128 (from_float never makes it), d of any size
                for (uint8_t & v : q) { v = (uint8_t) rng(); }
                for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = std::ldexp(nd(rng), (int) (rng() % 120) - 60); }
            };

            quant_w(a); quant_w(b);
            normal_q(1.0f);  if (!check("quantized w, normal x", n, a, b, q)) { return 1; }
            raw_q();         if (!check("quantized w, raw q8", n, a, b, q)) { return 1; }
            raw_w(a); raw_w(b);
            normal_q(3.0f);  if (!check("raw w, normal x", n, a, b, q)) { return 1; }
            raw_q();         if (!check("raw w, raw q8", n, a, b, q)) { return 1; }
            for (float & v : f) { v = nd(rng) * std::ldexp(1.0f, (int) (rng() % 200) - 100); }   // extreme spread within each block
            to_q8k(f.data(), q.data(), n);
            if (!check("raw w, wide-range x", n, a, b, q)) { return 1; }
            cases += 5;
        }
        // edge rows: all-zero weights / input, all-0xFF blocks with the largest d (inf), the smallest subnormal d
        std::vector<uint8_t> zw(wb, 0), ff(wb, 0xFF);
        for (int i = 0; i < nb; ++i) { ((block_iq2_s *) ff.data())[i].d = 0x7BFF; }   // 65504
        std::mt19937 rng(n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::fill(f.begin(), f.end(), 0.0f);
        ggml_quantize_chunk(GGML_TYPE_IQ2_S, f.data(), a.data(), 0, 1, n, nullptr);   // quantized zero row
        to_q8k(f.data(), q.data(), n);
        if (!check("zero w, zero x", n, zw, a, q)) { return 1; }
        for (float & v : f) { v = nd(rng); }
        to_q8k(f.data(), q.data(), n);
        if (!check("zero w, normal x", n, zw, a, q)) { return 1; }
        if (!check("0xFF w, normal x", n, ff, zw, q)) { return 1; }
        for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = 3e38f; }
        if (!check("0xFF w, huge d (inf)", n, ff, a, q)) { return 1; }
        memset(q.data(), 0x80, qb);   // every q8 byte -128
        for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = 1.0f; ((block_iq2_s *) ff.data())[i].d = 0x0001; }
        if (!check("0xFF w subnormal d, q8 -128", n, ff, zw, q)) { return 1; }
        cases += 5;
    }
    printf("ok: %d cases x (2 single + 1 dual) bit-identical to ggml's IQ2_S vec_dot (%s), n = 256, 2560, 4096\n", cases, lib.c_str());
    return 0;
}

#endif
