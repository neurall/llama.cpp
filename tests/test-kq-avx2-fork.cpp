// src/llama-kq-avx2-fork.h against ggml's own Q2_K, Q3_K and Q4_K x Q8_K dot products (the CPU backend's vec_dot), bit for bit (memcmp of the float).
// Loads only the CPU backend variant (ggml_backend_load_all_from_path would start CUDA too). Single-threaded. From the repo root:
//   B=/p/bw/rels/devbuilds/defaults_on OUT=/tmp/test-kq-avx2-fork
//   clang++ -std=c++17 -O2 -mavx2 -mfma -mf16c -I src -I ggml/include -I ggml/src tests/test-kq-avx2-fork.cpp -o $OUT -L $B -lggml -lggml-base -Wl,-rpath,$B
//   $OUT [build dir, default $B] [cpu variant, default haswell]
#include "llama-kq-avx2-fork.h"

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

typedef void (*dot2_t)(int, float *, float *, const void *, const void *, const void *);

struct kq_type {
    const char * name;
    ggml_type    type;
    ggml_vec_dot_t fork;
    dot2_t       fork2;
    int          halves[2];   // byte offsets of the fp16 fields (d, and dmin where the type has one; -1 = none)
};
static const kq_type types[] = {
    { "Q2_K", GGML_TYPE_Q2_K, llama_dot_q2_K_q8_K_avx2, llama_dot2_q2_K_q8_K_avx2, { QK_K / 16 + QK_K / 4, QK_K / 16 + QK_K / 4 + 2 } },
    { "Q3_K", GGML_TYPE_Q3_K, llama_dot_q3_K_q8_K_avx2, llama_dot2_q3_K_q8_K_avx2, { QK_K / 8 + QK_K / 4 + 12, -1 } },
    { "Q4_K", GGML_TYPE_Q4_K, llama_dot_q4_K_q8_K_avx2, llama_dot2_q4_K_q8_K_avx2, { 0, 2 } },
};
static_assert(sizeof(block_q2_K) == QK_K / 16 + QK_K / 4 + 4 && sizeof(block_q3_K) == QK_K / 8 + QK_K / 4 + 12 + 2, "half offsets");

static ggml_vec_dot_t    ref_dot;
static ggml_from_float_t to_q8k;
static const kq_type *   cur;

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint16_t finite_half(std::mt19937 & rng) {   // any fp16 bit pattern but Inf/NaN (zeros and subnormals included)
    for (;;) { const uint16_t h = (uint16_t) rng(); if (((h >> 10) & 31) != 31) { return h; } }
}

// every fp16 field of every block of row w set by f()
template <typename F> static void set_halves(std::vector<uint8_t> & w, F f) {
    const size_t bsz = ggml_type_size(cur->type);
    for (size_t o = 0; o < w.size(); o += bsz) {
        for (int h : cur->halves) { if (h >= 0) { const uint16_t v = f(); memcpy(&w[o + h], &v, 2); } }
    }
}

// one case: rows a and b against q; compares single and dual calls with ggml, returns false on the first mismatch
static bool check(const char * what, int n, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const std::vector<uint8_t> & q) {
    float ra, rb, sa, sb, da, db;
    ref_dot(n, &ra, 0, a.data(), 0, q.data(), 0, 1);
    ref_dot(n, &rb, 0, b.data(), 0, q.data(), 0, 1);
    cur->fork(n, &sa, 0, a.data(), 0, q.data(), 0, 1);
    cur->fork(n, &sb, 0, b.data(), 0, q.data(), 0, 1);
    cur->fork2(n, &da, &db, a.data(), b.data(), q.data());
    const float got[4] = { sa, sb, da, db }, want[4] = { ra, rb, ra, rb };
    static const char * names[4] = { "single a", "single b", "dual a", "dual b" };
    for (int k = 0; k < 4; ++k) {
        if (memcmp(&got[k], &want[k], 4) != 0) {
            printf("MISMATCH %s %s n=%d %s: ggml %.9g (0x%08x) fork %.9g (0x%08x)\n", cur->name, what, n, names[k], want[k], bits(want[k]), got[k], bits(got[k]));
            return false;
        }
    }
    return true;
}

static int run_type() {
    const ggml_type t = cur->type;
    int cases = 0;
    for (const int n : { 256, 640, 2560, 4096 }) {
        if (n % QK_K != 0) { continue; }   // 640: not a whole number of 256-value blocks
        const size_t wb = ggml_row_size(t, n), qb = ggml_row_size(GGML_TYPE_Q8_K, n);
        const int nb = n / QK_K;
        std::vector<uint8_t> a(wb), b(wb), q(qb);
        std::vector<float> f(n);
        for (uint32_t seed = 1; seed <= 4; ++seed) {
            std::mt19937 rng(seed * 7919 + n + 31 * t);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            const float wscale[4] = { 1.0f, 1e-3f, 30.0f, 0.02f };
            auto quant_w = [&](std::vector<uint8_t> & w) {   // normal weights (with an offset: nonzero mins) through ggml's own quantizer
                for (float & v : f) { v = wscale[seed - 1] * (nd(rng) + 0.3f); }
                ggml_quantize_chunk(t, f.data(), w.data(), 0, 1, n, nullptr);
            };
            auto raw_w = [&](std::vector<uint8_t> & w) {     // random bytes: every q, high bit, scale and min, any finite d / dmin
                for (uint8_t & v : w) { v = (uint8_t) rng(); }
                set_halves(w, [&] { return finite_half(rng); });
            };
            auto normal_q = [&](float sd) { for (float & v : f) { v = sd * nd(rng); } to_q8k(f.data(), q.data(), n); };
            auto raw_q = [&]() {                              // q8 bytes incl. -128 and any bsums (from_float never makes those), d of any size
                for (uint8_t & v : q) { v = (uint8_t) rng(); }
                for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = std::ldexp(nd(rng), (int) (rng() % 120) - 60); }
            };

            quant_w(a); quant_w(b);
            normal_q(1.0f);  if (!check("quantized w, normal x", n, a, b, q)) { return -1; }
            raw_q();         if (!check("quantized w, raw q8", n, a, b, q)) { return -1; }
            raw_w(a); raw_w(b);
            normal_q(3.0f);  if (!check("raw w, normal x", n, a, b, q)) { return -1; }
            raw_q();         if (!check("raw w, raw q8", n, a, b, q)) { return -1; }
            for (float & v : f) { v = nd(rng) * std::ldexp(1.0f, (int) (rng() % 200) - 100); }   // extreme spread within each block
            to_q8k(f.data(), q.data(), n);
            if (!check("raw w, wide-range x", n, a, b, q)) { return -1; }
            cases += 5;
        }
        // edge rows: all-zero weights / input, all-0xFF blocks with the largest d (inf), the smallest subnormal d, every q8 byte -128
        std::vector<uint8_t> zw(wb, 0), ff(wb, 0xFF);
        set_halves(ff, [] { return (uint16_t) 0x7BFF; });   // 65504
        std::mt19937 rng(n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::fill(f.begin(), f.end(), 0.0f);
        ggml_quantize_chunk(t, f.data(), a.data(), 0, 1, n, nullptr);   // quantized zero row
        to_q8k(f.data(), q.data(), n);
        if (!check("zero w, zero x", n, zw, a, q)) { return -1; }
        for (float & v : f) { v = nd(rng); }
        to_q8k(f.data(), q.data(), n);
        if (!check("zero w, normal x", n, zw, a, q)) { return -1; }
        if (!check("0xFF w, normal x", n, ff, zw, q)) { return -1; }
        for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = 3e38f; }
        if (!check("0xFF w, huge d (inf)", n, ff, a, q)) { return -1; }
        memset(q.data(), 0x80, qb);   // every q8 byte -128 (bsums 0x8080 too)
        for (int i = 0; i < nb; ++i) { ((block_q8_K *) q.data())[i].d = 1.0f; }
        set_halves(ff, [] { return (uint16_t) 0x0001; });
        if (!check("0xFF w subnormal d, q8 -128", n, ff, zw, q)) { return -1; }
        cases += 5;
    }
    return cases;
}

int main(int argc, char ** argv) {
    setenv("OMP_NUM_THREADS", "1", 1);
    const std::string dir = argc > 1 ? argv[1] : "/p/bw/rels/devbuilds/defaults_on";
    const std::string lib = dir + "/libggml-cpu-" + (argc > 2 ? argv[2] : "haswell") + ".so";
    ggml_backend_reg_t reg = ggml_backend_load(lib.c_str());
    auto traits = reg ? (const ggml_type_traits_cpu * (*)(ggml_type)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_type_traits") : nullptr;
    if (!traits) { printf("FAIL: no CPU backend type traits from %s\n", lib.c_str()); return 1; }
    to_q8k = traits(GGML_TYPE_Q8_K)->from_float;
    int fails = 0;
    for (const kq_type & kt : types) {
        cur = &kt;
        ref_dot = traits(kt.type)->vec_dot;
        if (!ref_dot || !to_q8k || traits(kt.type)->vec_dot_type != GGML_TYPE_Q8_K) { printf("FAIL: %s traits\n", kt.name); ++fails; continue; }
        const int cases = run_type();
        if (cases < 0) { ++fails; continue; }
        printf("ok: %s %d cases x (2 single + 1 dual) bit-identical to ggml's vec_dot (%s), n = 256, 2560, 4096\n", kt.name, cases, lib.c_str());
    }
    return fails ? 1 : 0;
}

#endif
