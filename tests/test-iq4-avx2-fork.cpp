// src/llama-iq4-avx2-fork.h against ggml's own IQ4_NL / MXFP4 / Q8_0 / Q2_0 x Q8_0 and IQ4_XS x Q8_K dot products (the CPU backend's vec_dot from its
// type traits, and its AVX2 form ggml_vec_dot_<f>_avx2 that the per-type kernel tuner may pick), bit for bit (memcmp of the float).
// Loads only the CPU backend variant (ggml_backend_load_all_from_path would start CUDA too). One thread; GGML_CPU_GPU_SHAPE forced off (with it on,
// ggml's Q8_0 / Q2_0 dots switch to the CUDA-order sums these kernels do not reproduce). From the repo root:
//   B=/p/bw/rels/devbuilds/defaults_on OUT=/tmp/test-iq4-avx2-fork
//   clang++ -std=c++17 -O2 -mavx2 -mfma -mf16c -I src -I ggml/include -I ggml/src tests/test-iq4-avx2-fork.cpp -o $OUT -L $B -lggml -lggml-base -Wl,-rpath,$B
//   $OUT [build dir, default $B] [cpu variant, default haswell] [bench | hot]
// bench: single-thread rows per second, weights cold (each timing walks rows of a 64 MiB buffer not touched for the last 30 timings: past L2 and
// this CCX's 16 MiB L3), the q8 row hot as in the pool; hot: the same 128 rows (in L2) every time, for the instruction cost alone; under 1 s in all.
#include "llama-iq4-avx2-fork.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
int main() { printf("skip: built without AVX2 + FMA + F16C\n"); return 0; }
#else

typedef void (*dot2_t)(int, float *, float *, const void *, const void *, const void *);

struct kind {
    const char * name;
    ggml_type    type, act;
    int          scale_bytes;   // the block's scale field at offset 0: 2 = fp16 d, 1 = MXFP4's e8m0
    ggml_vec_dot_t mine;        // nullptr: dual only
    dot2_t       mine2;
    const char * form;          // ggml_vec_dot_<form>_avx2
};

static const kind kinds[] = {
    { "IQ4_NL", GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, 2, llama_dot_iq4_nl_q8_0_avx2, llama_dot2_iq4_nl_q8_0_avx2, "iq4_nl_q8_0" },
    { "IQ4_XS", GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K, 2, llama_dot_iq4_xs_q8_K_avx2, llama_dot2_iq4_xs_q8_K_avx2, "iq4_xs_q8_K" },
    { "Q2_0",   GGML_TYPE_Q2_0,   GGML_TYPE_Q8_0, 2, nullptr,                    llama_dot2_q2_0_q8_0_avx2,   "q2_0_q8_0"   },
    { "Q8_0",   GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0, 2, nullptr,                    llama_dot2_q8_0_q8_0_avx2,   "q8_0_q8_0"   },
    { "MXFP4",  GGML_TYPE_MXFP4,  GGML_TYPE_Q8_0, 1, llama_dot_mxfp4_q8_0_avx2,  llama_dot2_mxfp4_q8_0_avx2,  "mxfp4_q8_0"  },
};

static const ggml_type_traits_cpu * (*traits)(ggml_type);
static ggml_vec_dot_t ref_dot, ref_form;

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint16_t finite_half(std::mt19937 & rng) {   // any fp16 bit pattern but Inf/NaN (zeros and subnormals included)
    for (;;) { const uint16_t h = (uint16_t) rng(); if (((h >> 10) & 31) != 31) { return h; } }
}

// one case: rows a and b against q; ggml's default and AVX2-form dots against single and dual calls, false on the first mismatch
static bool check(const kind & k, const char * what, int n, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, const std::vector<uint8_t> & q) {
    float ra, rb, fa, fb, got[4], want[4];
    ref_dot(n, &ra, 0, a.data(), 0, q.data(), 0, 1);
    ref_dot(n, &rb, 0, b.data(), 0, q.data(), 0, 1);
    if (ref_form) {
        ref_form(n, &fa, 0, a.data(), 0, q.data(), 0, 1);
        ref_form(n, &fb, 0, b.data(), 0, q.data(), 0, 1);
        if (memcmp(&fa, &ra, 4) != 0 || memcmp(&fb, &rb, 4) != 0) { printf("NOTE %s %s n=%d: ggml's default and AVX2 forms differ\n", k.name, what, n); }
    }
    int m = 0;
    static const char * names[4] = { "single a", "single b", "dual a", "dual b" };
    if (k.mine) {
        k.mine(n, &got[0], 0, a.data(), 0, q.data(), 0, 1); want[0] = ra;
        k.mine(n, &got[1], 0, b.data(), 0, q.data(), 0, 1); want[1] = rb;
    }
    k.mine2(n, &got[2], &got[3], a.data(), b.data(), q.data()); want[2] = ra; want[3] = rb;
    for (m = k.mine ? 0 : 2; m < 4; ++m) {
        if (memcmp(&got[m], &want[m], 4) != 0) {
            printf("MISMATCH %s %s n=%d %s: ggml %.9g (0x%08x) fork %.9g (0x%08x)\n", k.name, what, n, names[m], want[m], bits(want[m]), got[m], bits(got[m]));
            return false;
        }
    }
    return true;
}

static void set_scale(const kind & k, uint8_t * blk, std::mt19937 & rng) {
    if (k.scale_bytes == 2) { const uint16_t h = finite_half(rng); memcpy(blk, &h, 2); } else { blk[0] = (uint8_t) rng(); }
}

static int test_kind(const kind & k) {
    const auto * tt = traits(k.type);
    ref_dot = tt->vec_dot;
    const auto to_q = traits(k.act)->from_float;
    if (!ref_dot || !to_q || tt->vec_dot_type != k.act) { printf("FAIL: %s traits\n", k.name); return -1; }
    const size_t wblk = ggml_type_size(k.type), qblk = ggml_type_size(k.act);
    const int    wqk = (int) ggml_blck_size(k.type), qqk = (int) ggml_blck_size(k.act);

    int cases = 0;
    for (const int n : { 192, 224, 256, 640, 2560, 4096 }) {
        if (n % wqk != 0 || n % qqk != 0) { continue; }   // 192 / 224: odd block counts (ggml's tails); 640 not for IQ4_XS
        const size_t wb = ggml_row_size(k.type, n), qb = ggml_row_size(k.act, n);
        const int nb = n / wqk, nq = n / qqk;
        std::vector<uint8_t> a(wb), b(wb), q(qb);
        std::vector<float> f(n);
        auto raw_q = [&](std::mt19937 & rng, std::normal_distribution<float> & nd) {   // q8 bytes incl. -128 (from_float never makes it), any scale
            for (uint8_t & v : q) { v = (uint8_t) rng(); }
            for (int i = 0; i < nq; ++i) {
                uint8_t * blk = q.data() + i * qblk;
                if (k.act == GGML_TYPE_Q8_K) { const float d = std::ldexp(nd(rng), (int) (rng() % 120) - 60); memcpy(blk, &d, 4); }
                else { const uint16_t h = finite_half(rng); memcpy(blk, &h, 2); }
            }
        };
        for (uint32_t seed = 1; seed <= 4; ++seed) {
            std::mt19937 rng(seed * 7919 + n);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            const float wscale[4] = { 1.0f, 1e-3f, 30.0f, 0.02f };
            auto quant_w = [&](std::vector<uint8_t> & w) {   // normal weights through ggml's own quantizer
                for (float & v : f) { v = wscale[seed - 1] * nd(rng); }
                ggml_quantize_chunk(k.type, f.data(), w.data(), 0, 1, n, nullptr);
            };
            auto raw_w = [&](std::vector<uint8_t> & w) {     // random bytes: every index, scale and sign, any finite scale
                for (uint8_t & v : w) { v = (uint8_t) rng(); }
                for (int i = 0; i < nb; ++i) { set_scale(k, w.data() + i * wblk, rng); }
            };
            auto normal_q = [&](float sd) { for (float & v : f) { v = sd * nd(rng); } to_q(f.data(), q.data(), n); };

            quant_w(a); quant_w(b);
            normal_q(1.0f);     if (!check(k, "quantized w, normal x", n, a, b, q)) { return -1; }
            raw_q(rng, nd);     if (!check(k, "quantized w, raw q8", n, a, b, q)) { return -1; }
            raw_w(a); raw_w(b);
            normal_q(3.0f);     if (!check(k, "raw w, normal x", n, a, b, q)) { return -1; }
            raw_q(rng, nd);     if (!check(k, "raw w, raw q8", n, a, b, q)) { return -1; }
            for (float & v : f) { v = nd(rng) * std::ldexp(1.0f, (int) (rng() % 200) - 100); }   // extreme spread within each block
            to_q(f.data(), q.data(), n);
            if (!check(k, "raw w, wide-range x", n, a, b, q)) { return -1; }
            cases += 5;
        }
        // edge rows: all-zero weights / input, all-0xFF blocks with the largest scale (inf), the smallest subnormal scale with q8 -128
        std::vector<uint8_t> zw(wb, 0), ff(wb, 0xFF);
        for (int i = 0; i < nb; ++i) { if (k.scale_bytes == 2) { const uint16_t h = 0x7BFF; memcpy(ff.data() + i * wblk, &h, 2); } }   // 65504 (e8m0 0xFF: 2^127)
        std::mt19937 rng(n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::fill(f.begin(), f.end(), 0.0f);
        ggml_quantize_chunk(k.type, f.data(), a.data(), 0, 1, n, nullptr);   // quantized zero row
        to_q(f.data(), q.data(), n);
        if (!check(k, "zero w, zero x", n, zw, a, q)) { return -1; }
        for (float & v : f) { v = nd(rng); }
        to_q(f.data(), q.data(), n);
        if (!check(k, "zero w, normal x", n, zw, a, q)) { return -1; }
        if (!check(k, "0xFF w, normal x", n, ff, zw, q)) { return -1; }
        raw_q(rng, nd);
        for (int i = 0; i < nq; ++i) {   // huge q8 scales: inf products
            uint8_t * blk = q.data() + i * qblk;
            if (k.act == GGML_TYPE_Q8_K) { const float d = 3e38f; memcpy(blk, &d, 4); } else { const uint16_t h = 0x7BFF; memcpy(blk, &h, 2); }
        }
        if (!check(k, "0xFF w, huge d (inf)", n, ff, a, q)) { return -1; }
        memset(q.data(), 0x80, qb);   // every q8 byte -128
        for (int i = 0; i < nq; ++i) {
            uint8_t * blk = q.data() + i * qblk;
            if (k.act == GGML_TYPE_Q8_K) { const float d = 1.0f; memcpy(blk, &d, 4); } else { const uint16_t h = 0x3C00; memcpy(blk, &h, 2); }
        }
        for (int i = 0; i < nb; ++i) { uint8_t * blk = ff.data() + i * wblk; if (k.scale_bytes == 2) { blk[0] = 1; blk[1] = 0; } else { blk[0] = 0; } }
        if (!check(k, "0xFF w subnormal d, q8 -128", n, ff, zw, q)) { return -1; }
        for (uint8_t & v : a) { v = (uint8_t) rng(); }
        for (int i = 0; i < nb; ++i) { set_scale(k, a.data() + i * wblk, rng); }
        if (!check(k, "raw w, q8 -128", n, a, ff, q)) { return -1; }
        cases += 6;
    }
    return cases;
}

// ---- throughput: rows per second on one thread, weights cold
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void bench(const kind & k, std::vector<uint8_t> & buf, bool hot) {
    const auto to_q = traits(k.act)->from_float;
    const size_t wblk = ggml_type_size(k.type);
    for (const int n : { 640, 2560 }) {
        if (n % ggml_blck_size(k.type) != 0) { continue; }
        const size_t rb = ggml_row_size(k.type, n);
        const int rows = hot ? 128 : n == 640 ? 4096 : 1536;                    // cold: >= 1300 rows, > 1 MiB: past L2; hot: <= 184 KiB, in L2
        const size_t slice = rows * rb, nslice = hot ? 1 : buf.size() / slice;  // cold: >= 30 slices, a slice comes back after 30 timings, long out of L3
        // sane scales in every block (1.0 / e8m0 127): no denormal float math in the timing
        for (size_t o = 0; o + wblk <= nslice * slice; o += wblk) { if (k.scale_bytes == 2) { buf[o] = 0x00; buf[o + 1] = 0x3C; } else { buf[o] = 127; } }
        std::vector<float> f(n), out(rows);
        std::vector<uint8_t> q(ggml_row_size(k.act, n));
        std::mt19937 rng(n);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float & v : f) { v = nd(rng); }
        to_q(f.data(), q.data(), n);
        size_t next = 0;
        const int reps = hot ? 31 : 9;
        std::vector<double> t[3];
        for (int r = 0; r < reps; ++r) {
            for (int v = 0; v < 3; ++v) {
                if (v == 1 && !k.mine) { continue; }
                const uint8_t * w = buf.data() + (next++ % nslice) * slice;
                const double t0 = now();
                if (v == 0)      { for (int i = 0; i < rows; ++i) { ref_dot(n, &out[i], 0, w + i * rb, 0, q.data(), 0, 1); } }
                else if (v == 1) { for (int i = 0; i < rows; ++i) { k.mine(n, &out[i], 0, w + i * rb, 0, q.data(), 0, 1); } }
                else             { for (int i = 0; i < rows; i += 2) { k.mine2(n, &out[i], &out[i + 1], w + i * rb, w + (i + 1) * rb, q.data()); } }
                t[v].push_back(now() - t0);
            }
        }
        double med[3] = { 0, 0, 0 };
        for (int v = 0; v < 3; ++v) { if (!t[v].empty()) { std::sort(t[v].begin(), t[v].end()); med[v] = t[v][t[v].size() / 2]; } }
        auto mrs = [&](double s) { return s > 0 ? rows / s / 1e6 : 0.0; };
        printf("bench %-6s n=%4d %4d rows %s, M rows/s (median of %d): ggml %.2f  single %s%.2f  dual %.2f  (single %+.0f%%, dual %+.0f%%)\n",
               k.name, n, rows, hot ? "hot " : "cold", reps, mrs(med[0]), k.mine ? "" : "-", mrs(med[1]), mrs(med[2]),
               k.mine ? 100.0 * (med[0] / med[1] - 1) : 0.0, 100.0 * (med[0] / med[2] - 1));
    }
}

int main(int argc, char ** argv) {
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("GGML_CPU_GPU_SHAPE", "0", 1);
    setenv("GGML_CPU_KERNEL_TUNE", "0", 1);
    const std::string dir = argc > 1 ? argv[1] : "/p/bw/rels/devbuilds/defaults_on";
    const std::string lib = dir + "/libggml-cpu-" + (argc > 2 ? argv[2] : "haswell") + ".so";
    ggml_backend_reg_t reg = ggml_backend_load(lib.c_str());
    traits = reg ? (const ggml_type_traits_cpu * (*)(ggml_type)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_type_traits") : nullptr;
    if (!traits) { printf("FAIL: no CPU backend type traits from %s\n", lib.c_str()); return 1; }
    void * h = dlopen(lib.c_str(), RTLD_NOW | RTLD_NOLOAD);
    const bool do_bench = argc > 3 && (std::string(argv[3]) == "bench" || std::string(argv[3]) == "hot");
    const bool hot = do_bench && std::string(argv[3]) == "hot";

    std::vector<uint8_t> buf;
    if (do_bench) {   // 64 MiB of random bytes (xorshift: fast)
        buf.resize(64u << 20);
        uint64_t x = 88172645463325252ull;
        for (size_t i = 0; i + 8 <= buf.size(); i += 8) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; memcpy(&buf[i], &x, 8); }
    }
    int rc = 0;
    for (const kind & k : kinds) {
        ref_form = h ? (ggml_vec_dot_t) dlsym(h, (std::string("ggml_vec_dot_") + k.form + "_avx2").c_str()) : nullptr;
        const int cases = test_kind(k);
        if (cases < 0) { rc = 1; continue; }
        printf("ok: %s %d cases x (%s) bit-identical to ggml's vec_dot%s (%s), n = 192/224/256/640/2560/4096 where the block divides\n",
               k.name, cases, k.mine ? "2 single + 1 dual" : "1 dual", ref_form ? " and its AVX2 form" : "", lib.c_str());
        if (do_bench) { bench(k, buf, hot); }
    }
    return rc;
}

#endif
