// bell CPU expert pool (src/llama-bellpool-fork.h): pure, float weights against a naive reference.
//   clang++ -std=c++17 -O2 -pthread -I src tests/test-bellpool-fork.cpp -o test-bellpool-fork && ./test-bellpool-fork
#include "llama-bellpool-fork.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <random>

static void dot_f32(int n, float * s, size_t, const void * x, size_t, const void * y, size_t, int) {
    const float * a = (const float *) x, * b = (const float *) y;
    double acc = 0;
    for (int i = 0; i < n; ++i) { acc += (double) a[i] * b[i]; }
    *s = (float) acc;
}

// a "quantized" input that is a scaled float copy: proves the pool quantizes once and the dot reads the quantized form
static void quant_x2(const float * x, void * y, int64_t n) { float * o = (float *) y; for (int64_t i = 0; i < n; ++i) { o[i] = 2.0f * x[i]; } }

int main() {
    // split covers every row once, sizes differ by at most one
    for (int total = 0; total < 200; total += 7) {
        for (int n = 1; n < 9; ++n) {
            int64_t covered = 0, prev_hi = 0, mn = 1 << 30, mx = 0;
            for (int w = 0; w < n; ++w) { int64_t lo, hi; bellpool_split(total, n, w, lo, hi); assert(lo == prev_hi); prev_hi = hi; covered += hi - lo; mn = std::min(mn, hi - lo); mx = std::max(mx, hi - lo); }
            assert(covered == total && prev_hi == total && mx - mn <= 1);
        }
    }
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    const int n_embd = 96, n_ff = 40;
    for (int workers : {1, 3, 8}) {
        bellpool pool(workers);
        for (int round = 0; round < 50; ++round) {
            const int n = 1 + (int) (rng() % 4);
            std::vector<std::vector<float>> up(n), gate(n), down(n), out(n, std::vector<float>(n_embd, -1.0f));
            std::vector<float> x(n_embd);
            for (auto & v : x) { v = nd(rng); }
            bellpool_layer L;
            L.n_embd = n_embd; L.n_ff = n_ff; L.n = n; L.x = x.data();
            const bool q = round % 2 == 1;
            L.up   = { dot_f32, q ? quant_x2 : nullptr, (size_t) n_embd * 4, (size_t) n_embd * 4 };
            L.gate = { dot_f32, q ? quant_x2 : nullptr, (size_t) n_embd * 4, (size_t) n_embd * 4 };
            L.down = { dot_f32, q ? quant_x2 : nullptr, (size_t) n_ff * 4,   (size_t) n_ff * 4 };
            for (int e = 0; e < n; ++e) {
                up[e].resize((size_t) n_ff * n_embd); gate[e].resize((size_t) n_ff * n_embd); down[e].resize((size_t) n_embd * n_ff);
                for (auto & v : up[e]) { v = nd(rng); } for (auto & v : gate[e]) { v = nd(rng); } for (auto & v : down[e]) { v = nd(rng); }
                L.w_up[e] = (const char *) up[e].data(); L.w_gate[e] = (const char *) gate[e].data(); L.w_down[e] = (const char *) down[e].data();
                L.out[e] = out[e].data();
            }
            pool.run(L);
            const float k = q ? 2.0f : 1.0f;   // the quantized input is twice the float one
            for (int e = 0; e < n; ++e) {
                std::vector<float> h(n_ff);
                for (int r = 0; r < n_ff; ++r) {
                    double u = 0, g = 0;
                    for (int i = 0; i < n_embd; ++i) { u += (double) up[e][(size_t) r * n_embd + i] * k * x[i]; g += (double) gate[e][(size_t) r * n_embd + i] * k * x[i]; }
                    h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
                }
                for (int r = 0; r < n_embd; ++r) {
                    double d = 0;
                    for (int i = 0; i < n_ff; ++i) { d += (double) down[e][(size_t) r * n_ff + i] * k * h[i]; }
                    assert(std::fabs(out[e][r] - (float) d) < 1e-3f * (1.0f + std::fabs((float) d)));
                }
            }
        }
        // idle past the spin: the workers sleep and still wake for the next layer
        pool.spin_us = 1000;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bellpool_layer L; std::vector<float> x(n_embd, 0.1f), w((size_t) n_ff * n_embd, 0.01f), wd((size_t) n_embd * n_ff, 0.01f), o(n_embd, -1.0f);
        L.n_embd = n_embd; L.n_ff = n_ff; L.n = 1; L.x = x.data();
        L.up = L.gate = { dot_f32, nullptr, (size_t) n_embd * 4, (size_t) n_embd * 4 }; L.down = { dot_f32, nullptr, (size_t) n_ff * 4, (size_t) n_ff * 4 };
        L.w_up[0] = L.w_gate[0] = (const char *) w.data(); L.w_down[0] = (const char *) wd.data(); L.out[0] = o.data();
        pool.run(L);
        assert(o[0] != -1.0f);
    }
    {   // up and gate rows in one pass give what two dots give (float input shared, and a quantized input shared)
        const int n_embd = 64, n_ff = 48;
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> U(-1.0f, 1.0f);
        std::vector<float> x(n_embd), wu((size_t) n_ff * n_embd), wg((size_t) n_ff * n_embd), wd((size_t) n_embd * n_ff), a(n_embd), b(n_embd);
        for (auto * v : { &x, &wu, &wg, &wd }) { for (float & f : *v) { f = U(rng); } }
        for (int quantized = 0; quantized < 2; ++quantized) {
            for (int two = 0; two < 2; ++two) {
                bellpool pool(3);
                bellpool_layer L;
                L.n_embd = n_embd; L.n_ff = n_ff; L.n = 1; L.x = x.data();
                L.up = L.gate = { dot_f32, quantized ? quant_x2 : nullptr, (size_t) n_embd * 4, (size_t) n_embd * 4 };
                L.down = { dot_f32, nullptr, (size_t) n_ff * 4, (size_t) n_ff * 4 };
                if (two) { L.upgate2 = [](int n, float * s_a, float * s_b, const void * xa, const void * xb, const void * y) { dot_f32(n, s_a, 0, xa, 0, y, 0, 1); dot_f32(n, s_b, 0, xb, 0, y, 0, 1); }; }
                L.w_up[0] = (const char *) wu.data(); L.w_gate[0] = (const char *) wg.data(); L.w_down[0] = (const char *) wd.data();
                L.out[0] = two ? b.data() : a.data();
                pool.run(L);
            }
            for (int i = 0; i < n_embd; ++i) { assert(a[i] == b[i]); }
        }
    }
    {   // two neighbouring down rows in one pass give what two dots give: odd row count, two experts, shares that end inside a pair
        const int n_embd = 63, n_ff = 48;
        std::mt19937 rng(12);
        std::uniform_real_distribution<float> U(-1.0f, 1.0f);
        std::vector<float> x(n_embd), wu((size_t) n_ff * n_embd), wg((size_t) n_ff * n_embd), wd[2], a[2], b[2];
        for (auto * v : { &x, &wu, &wg }) { for (float & f : *v) { f = U(rng); } }
        for (auto & v : wd) { v.resize((size_t) n_embd * n_ff); for (float & f : v) { f = U(rng); } }
        static std::atomic<int> pairs; static size_t row_bytes; row_bytes = (size_t) n_ff * 4;
        for (int workers : { 1, 3, 4 }) {
            for (int two = 0; two < 2; ++two) {
                bellpool pool(workers);
                bellpool_layer L;
                L.n_embd = n_embd; L.n_ff = n_ff; L.n = 2; L.x = x.data();
                L.up = L.gate = { dot_f32, nullptr, (size_t) n_embd * 4, (size_t) n_embd * 4 };
                L.down = { dot_f32, quant_x2, (size_t) n_ff * 4, (size_t) n_ff * 4 };
                if (two) {
                    pairs = 0;
                    L.down2 = [](int n, float * s_a, float * s_b, const void * xa, const void * xb, const void * y) {
                        assert((const char *) xb == (const char *) xa + row_bytes && s_b == s_a + 1);   // the next row of the same matrix, the next output
                        pairs++; dot_f32(n, s_a, 0, xa, 0, y, 0, 1); dot_f32(n, s_b, 0, xb, 0, y, 0, 1);
                    };
                }
                for (int e = 0; e < 2; ++e) {
                    (two ? b : a)[e].assign(n_embd, -7.0f);
                    L.w_up[e] = (const char *) wu.data(); L.w_gate[e] = (const char *) wg.data(); L.w_down[e] = (const char *) wd[e].data(); L.out[e] = (two ? b : a)[e].data();
                }
                pool.run(L);
                if (two) { assert(pairs.load() >= 2 * (n_embd / 2) - workers); }   // at most one unpaired row a share beyond the experts' odd ends
            }
            for (int e = 0; e < 2; ++e) { for (int i = 0; i < n_embd; ++i) { assert(a[e][i] == b[e][i] && a[e][i] != -7.0f); } }
        }
    }
    printf("ok: bell pool, 1/3/8 workers, 50 layers each, float and quantized input, wake after sleep, up and gate in one pass, down rows in pairs\n");
    return 0;
}
