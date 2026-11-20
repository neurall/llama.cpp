// AVX2 Q8_K quantizer (ggml/src/ggml-cpu/arch/x86/q8k-avx2-fork.h) against quantize_row_q8_K_ref from libggml-base: d, qs and bsums byte for byte.
//   B=/p/bw/rels/devbuilds/bell_one_fence
//   clang++ -std=c++17 -O2 -mavx2 -mfma -mf16c -I ggml/include -I ggml/src -I ggml/src/ggml-cpu/arch/x86 tests/test-q8k-avx2-fork.cpp \
//       -L $B -lggml-base -Wl,-rpath,$B -o test-q8k-avx2-fork && ./test-q8k-avx2-fork
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-quants.h"
#include "q8k-avx2-fork.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_blocks = 0;

static unsigned bits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }

// one row through both quantizers; prints the first difference and returns false
static bool check(const char * name, const std::vector<float> & x) {
    const int64_t nb = (int64_t) x.size() / QK_K;
    std::vector<block_q8_K> r(nb), a(nb);
    memset(r.data(), 0xab, nb * sizeof(block_q8_K));
    memset(a.data(), 0xab, nb * sizeof(block_q8_K));
    quantize_row_q8_K_ref(x.data(), r.data(), (int64_t) x.size());
    quantize_row_q8_K_avx2_fork(x.data(), a.data(), (int64_t) x.size());
    for (int64_t i = 0; i < nb; i++) {
        if (bits(r[i].d) != bits(a[i].d)) {
            printf("FAIL %s: block %lld d ref %08x avx2 %08x\n", name, (long long) i, bits(r[i].d), bits(a[i].d));
            return false;
        }
        for (int j = 0; j < QK_K; j++) {
            if (r[i].qs[j] != a[i].qs[j]) {
                printf("FAIL %s: block %lld qs[%d] ref %d avx2 %d (x %a)\n", name, (long long) i, j, r[i].qs[j], a[i].qs[j], x[i*QK_K + j]);
                return false;
            }
        }
        float amax = 0;
        for (int j = 0; j < QK_K; j++) { amax = std::max(amax, std::fabs(x[i*QK_K + j])); }
        for (int j = 0; amax != 0 && j < QK_K/16; j++) { // the ref leaves bsums of an all-zero block unwritten
            if (r[i].bsums[j] != a[i].bsums[j]) {
                printf("FAIL %s: block %lld bsums[%d] ref %d avx2 %d\n", name, (long long) i, j, r[i].bsums[j], a[i].bsums[j]);
                return false;
            }
        }
    }
    n_blocks += (int) nb;
    return true;
}

int main() {
    std::vector<float> x;

    for (int seed = 1; seed <= 5; seed++) {
        for (int k : {256, 2560, 4096}) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd;
            x.resize(k);
            for (float & v : x) { v = nd(rng); }
            if (!check("normal", x)) { return 1; }
        }
    }

    // magnitudes: below 127/FLT_MAX = 3.7e-37 the ref's iscale is inf; 1e-45 rounds to the smallest denormal
    for (float s : {1e-45f, 1e-41f, 1e-38f, 1e-37f, 1.3e-37f, 1e-36f, 1e-30f, 1e-10f, 1e10f, 1e30f, 1e37f}) {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd;
        x.resize(2560);
        for (float & v : x) { v = s * nd(rng); }
        if (!check("scaled", x)) { return 1; }
    }

    // amax on both sides of the smallest amax whose iscale is finite, both signs
    float t = 127.f / FLT_MAX;
    while (std::isinf(-127.f / t)) { t = std::nextafter(t, 1.f); }
    while (!std::isinf(-127.f / std::nextafter(t, 0.f))) { t = std::nextafter(t, 0.f); }
    for (float a : {std::nextafter(t, 0.f), t, std::nextafter(t, 1.f), FLT_MAX}) {
        for (float sg : {1.f, -1.f}) {
            std::mt19937 rng(8);
            std::uniform_real_distribution<float> ud(-1.f, 1.f);
            x.assign(QK_K, 0.f);
            for (float & v : x) { v = a * ud(rng); }
            x[77] = sg * a;
            if (!check("threshold", x)) { return 1; }
        }
    }

    // ties of |max|: x and -x in either order, in one 8-lane vector and across vectors, two equal maxima
    const int pos[][2] = {{26, 29}, {24, 31}, {7, 200}, {0, 255}, {0, 1}, {254, 255}};
    const float sgn[][2] = {{1, -1}, {-1, 1}, {1, 1}, {-1, -1}};
    for (auto & p : pos) {
        for (auto & sg : sgn) {
            std::mt19937 rng(p[0] * 256 + p[1]);
            std::uniform_real_distribution<float> ud(-1.f, 1.f);
            x.resize(QK_K);
            for (float & v : x) { v = ud(rng); }
            x[p[0]] = sg[0] * 1.75f;
            x[p[1]] = sg[1] * 1.75f;
            if (!check("tie", x)) { return 1; }
        }
    }

    // small integers: many ties of both signs at random positions
    {
        std::mt19937 rng(9);
        std::uniform_int_distribution<int> ui(-3, 3);
        x.resize(QK_K * 400);
        for (float & v : x) { v = (float) ui(rng); }
        if (!check("ints", x)) { return 1; }
    }

    // round half to even: max = +-127 gives iscale -+1, max 254 gives iscale -0.5; the products land on .5
    for (float mx : {127.f, -127.f, 254.f}) {
        x.resize(QK_K);
        for (int j = 0; j < QK_K; j++) { x[j] = mx == 254.f ? (float) (j - 128) : (j - 128) * 0.5f; }
        x[3] = mx;
        if (!check("half", x)) { return 1; }
    }

    // all-zero blocks (+0, -0, mixed) and one non-zero value in a zero block
    for (int z = 0; z < 3; z++) {
        x.resize(QK_K);
        for (int j = 0; j < QK_K; j++) { x[j] = (z == 1 || (z == 2 && (j & 1))) ? -0.f : 0.f; }
        if (!check("zero", x)) { return 1; }
    }
    for (int p : {0, 1, 7, 8, 131, 255}) {
        for (float v : {0.37f, -0.37f, 1e-20f, -3e5f}) {
            x.assign(QK_K, 0.f);
            x[p] = v;
            if (!check("one", x)) { return 1; }
        }
    }

    printf("ok: %d blocks identical to quantize_row_q8_K_ref (d, qs, bsums): normal k=256/2560/4096 x 5 seeds, magnitudes 1e-45..FLT_MAX, "
           "inf-iscale threshold, ties, ints, half-even, zero, one non-zero\n", n_blocks);
    return 0;
}
