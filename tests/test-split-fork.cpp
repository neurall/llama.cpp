// Unit tests of the dense split planner's layer model and search (src/llama-splitplan-fork.h): how many tokens the helper card takes (s), how many of each layer's experts
// move to it (f; 0 = the experts are skipped on the helper, the offload card computes them all), and whether splitting beats one card. Per-layer numbers in, plan out;
// no GPU and no model.
//
// build: cmake --build <dir> --target test-split-fork;  run: ctest -R test-split-fork   or   test-split-fork [iterations] [seed]
#include "../src/llama-splitplan-fork.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int  n_failed = 0;
static long g_iters  = 300;
static unsigned long g_seed = 1;

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-62s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}

// a model of L layers, the first n_dense_first of them dense (no experts); GLM-like sizes: 800 MB of experts per MoE layer, 5 ms of compute per 1k tokens per layer
static dsx_in model(int64_t T, int L = 40, int n_dense_first = 3, double Bm = 24.0, double Bh = 6.0, double rel = 1.0) {
    dsx_in in;
    in.T = T; in.L = L; in.Bm = Bm; in.Bh = Bh; in.rel = rel;
    in.w.assign(L, 1.0); in.xf.assign(L, 0.6); in.E.assign(L, 800e6); in.kv_b.assign(L, 0.0); in.att_w.assign(L, 1.0 / L);
    in.held_m.assign(L, 0.0); in.held_h.assign(L, 0.0);
    for (int il = 0; il < n_dense_first; ++il) { in.E[il] = 0.0; in.xf[il] = 0.0; }
    in.w_sum = L; in.c_base = 5.0; in.att_tok = 0.0;   // us per token for the whole model share of a layer = c_base * w / w_sum
    in.c_base = 5.0 * L;
    in.b_in = in.b_out = 6144.0 * 2.0;
    return in;
}

static void test_plan_shape() {
    printf("Testing the dense split plan:\n");
    std::mt19937_64 rng(g_seed);
    long bad = 0, n_f = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        auto in = model(256 + (int64_t) (rng() % 30000), 8 + (int) (rng() % 60), (int) (rng() % 6), 4.0 + (double) (rng() % 300) / 10.0, 2.0 + (double) (rng() % 300) / 10.0, 0.3 + (double) (rng() % 200) / 100.0);
        for (int il = 0; il < in.L; ++il) { in.held_m[il] = (double) (rng() % 100) / 100.0; in.held_h[il] = (double) (rng() % 100) / 100.0; in.kv_b[il] = (double) (rng() % 50) * 1e6; }
        const dsx_out o = dsx_plan(in);
        if (!(o.t_one > 0.0 && o.best_split > 0.0)) { fail("non-positive time"); continue; }
        if (o.best_s < 0.2 - 1e-9 || o.best_s > 0.8 + 1e-9) { fail(fmt("helper token share %.3f outside 0.2..0.8", o.best_s)); }
        if ((int) o.best_f.size() != in.L) { fail("f per layer"); continue; }
        for (int il = 0; il < in.L; ++il) {
            const double f = o.best_f[il];
            if (f < 0.0 || f > 1.0) { fail(fmt("layer %d: f %.2f", il, f)); }
            if (std::fabs(f * 20.0 - std::round(f * 20.0)) > 1e-9) { fail(fmt("layer %d: f %.3f is not a step of 5%%", il, f)); }
            if (in.E[il] <= 0.0 && f != 0.0) { fail(fmt("dense layer %d: the helper was given %.2f of its (no) experts", il, f)); }
            if (f > 0.0) {   // experts move only for a clear win (10% of the layer)
                n_f++;
                if (!(dsx_layer_us(in, il, o.best_s, f) < 0.90 * dsx_layer_us(in, il, o.best_s, 0.0))) { fail(fmt("layer %d: f %.2f is not a 10%% win", il, f)); }
            }
        }
    }
    report("s in 0.2..0.8; f in steps of 5%; dense layers f = 0; f > 0 only for a 10% win", bad == 0, first);
    report("the run had layers that took experts (not vacuous)", n_f > 100, fmt("%ld layer plans", n_f));
}

static void test_behaviour() {
    printf("Testing what the plan does when the cards differ (the experts are skipped on a slow helper):\n");
    {   // a helper behind a very slow link takes no experts, and splitting is no gain
        auto in = model(8000, 40, 3, 24.0, 0.05);
        const dsx_out o = dsx_plan(in);
        double fsum = 0.0; for (double f : o.best_f) { fsum += f; }
        report("helper link 0.05 GB/s: no experts on the helper", fsum == 0.0, fmt("sum f %.2f", fsum));
        report("helper link 0.05 GB/s: the split is slower than one card", o.best_split >= o.t_one, fmt("%.0f vs %.0f ms", o.best_split / 1e3, o.t_one / 1e3));
    }
    {   // experts already resident on a fast helper, compute-bound pass: it takes a share of them
        auto in = model(16000, 40, 3, 24.0, 24.0);
        in.c_base *= 6.0;
        for (int il = 0; il < in.L; ++il) { in.held_h[il] = 1.0; }
        const dsx_out o = dsx_plan(in);
        double fsum = 0.0; for (double f : o.best_f) { fsum += f; }
        report("fast helper that holds the experts: it computes some of them", fsum > 0.0, fmt("sum f %.2f", fsum));
    }
    {   // equal cards, no hops, no experts: the tokens split in half
        auto in = model(4000, 20, 20, 24.0, 24.0);
        in.b_in = in.b_out = 0.0;
        const dsx_out o = dsx_plan(in);
        report("equal cards, dense work only: half the tokens each", std::fabs(o.best_s - 0.5) < 0.021, fmt("s %.2f", o.best_s));
        report("equal cards, dense work only: the split beats one card by ~2x", o.best_split < 0.6 * o.t_one, fmt("%.0f vs %.0f ms", o.best_split / 1e3, o.t_one / 1e3));
    }
    {   // a faster helper takes more tokens
        auto a = dsx_plan(model(8000, 40, 40, 24.0, 24.0, 1.0)), b = dsx_plan(model(8000, 40, 40, 24.0, 24.0, 2.0));
        report("a helper twice as fast takes a larger share of the tokens", b.best_s >= a.best_s, fmt("%.2f vs %.2f", b.best_s, a.best_s));
    }
    {   // [the first model: 80-85% of the tokens on the helper, measured best 77%/less] the hand-in keeps the helper's share moderate for equal cards
        auto in = model(8800, 40, 3, 24.0, 24.0);
        const dsx_out o = dsx_plan(in);
        report("equal cards with experts: the helper never holds most of the tokens", o.best_s <= 0.6, fmt("s %.2f", o.best_s));
    }
    {   // time grows with the pass length
        bool mono = true; double p1 = 0, p2 = 0;
        for (int64_t T : { 512, 1024, 4096, 16384 }) { const dsx_out o = dsx_plan(model(T)); mono &= o.t_one > p1 && o.best_split > p2; p1 = o.t_one; p2 = o.best_split; }
        report("one-card and split times grow with the tokens", mono);
    }
    {   // the KV of a long context read over the helper's link makes the helper's range dearer
        auto a = model(8000, 40, 40, 24.0, 6.0), b = a;
        for (auto & x : b.kv_b) { x = 2e9; }
        report("a KV cache the helper must read over its link raises the split's time", dsx_plan(b).best_split > dsx_plan(a).best_split);
    }
}

// ---- any number of cards: the balance's shares, the token counts, the live loop and the N-card planner ----------------------------------------------------

static double card_time(const dsb_card & c, double s, double T) { return c.k + s * T * c.a; }

static void test_nc_target() {
    printf("Testing the N-card shares (equal time per card):\n");
    for (int n = 2; n <= 6; ++n) {   // equal cards: an equal share each
        std::vector<dsb_card> c(n, dsb_card{ 10.0, 0.0 });
        const auto s = dsb_target(c, 8000, 0.02, n - 1);
        bool ok = true; for (double v : s) { ok &= std::fabs(v - 1.0 / n) < 1e-9; }
        report(fmt("%d equal cards: 1/%d each", n, n).c_str(), ok);
    }
    {   // a card half as fast: half the share of each of the others
        const auto s = dsb_target({ { 20.0, 0.0 }, { 10.0, 0.0 }, { 10.0, 0.0 } }, 8000, 0.02, 2);
        report("3 cards, one twice as slow: 20/40/40%", std::fabs(s[0] - 0.2) < 1e-9 && std::fabs(s[1] - 0.4) < 1e-9 && std::fabs(s[2] - 0.4) < 1e-9,
               fmt("%.3f %.3f %.3f", s[0], s[1], s[2]));
    }
    {   // hand-offs over a slow link make a card of the same compute take fewer tokens (the x4 card of 2x RTX 3090: 45%, not 50%)
        const auto s = dsb_target({ { 30.0 + 7.6, 0.0 }, { 30.0, 0.0 } }, 8000, 0.02, 1);
        report("2 equal GPUs, the helper's hand-offs 25% of its compute: about 44% for it", std::fabs(s[0] - 30.0 / 67.6) < 1e-9, fmt("%.3f", s[0]));
    }
    std::mt19937_64 rng(g_seed);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters; ++it) {   // random cards: every card with a share takes the same time, the shares add up to 1, the expert card keeps one
        const int n = 2 + (int) (rng() % 7), x = (int) (rng() % n);
        const double T = 256 + (double) (rng() % 60000);
        std::vector<dsb_card> c(n);
        for (auto & ci : c) { ci.a = 1.0 + (double) (rng() % 2000) / 10.0; ci.k = (double) (rng() % 4) == 0 ? (double) (rng() % 500000) : 0.0; }
        const double min_share = 0.02;
        const auto s = dsb_target(c, T, min_share, x);
        // the expert card held at its minimum share (its fixed cost alone is longer than the others' time) is outside the comparison: the others end together
        const bool x_min = s[x] <= min_share * 1.0001;
        double sum = 0.0, tmax = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += s[i];
            if (s[i] < 0.0) { if (bad++ == 0) { first = fmt("negative share %.3f", s[i]); } }
            if (s[i] > 0.0 && !(i == x && x_min)) { tmax = std::max(tmax, card_time(c[i], s[i], T)); }
        }
        if (std::fabs(sum - 1.0) > 1e-9) { if (bad++ == 0) { first = fmt("shares add up to %.6f", sum); } }
        if (s[x] <= 0.0) { if (bad++ == 0) { first = "the expert card lost its range"; } }
        // every card with a share ends at the same time, except: the expert card held at its minimum, and at most one card that joined at the jump (it takes
        // what is left, at most min_share, and ends early)
        int early = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] <= 0.0 || card_time(c[i], s[i], T) >= tmax * (1.0 - 1e-6)) { continue; }
            if (i == x && x_min) { continue; }
            if (i != x && s[i] <= min_share * 1.0001) { ++early; continue; }
            if (bad++ == 0) { first = fmt("card %d ends at %.1f us, the others at %.1f", i, card_time(c[i], s[i], T), tmax); }
        }
        if (early > 1) { if (bad++ == 0) { first = fmt("%d cards end early", early); } }
        for (int i = 0; i < n; ++i) {   // a card left out would be the slowest even with the minimum share
            if (i != x && s[i] == 0.0 && c[i].a > 0.0 && card_time(c[i], min_share, T) < tmax * (1.0 - 1e-6)) {
                if (bad++ == 0) { first = fmt("card %d left out although %.0f%% of the tokens would take it %.1f us < %.1f", i, 100 * min_share, card_time(c[i], min_share, T), tmax); }
            }
        }
    }
    report("random cards: equal ends, shares add up to 1, nobody left out who would help", bad == 0, first);
    {   // a very slow card is left out; a card whose fixed cost alone is longer than the rest is left out
        const auto s1 = dsb_target({ { 1000.0, 0.0 }, { 10.0, 0.0 }, { 10.0, 0.0 } }, 8000, 0.02, 2);
        report("a card 100x slower than the others gets no tokens", s1[0] == 0.0 && std::fabs(s1[1] - 0.5) < 1e-9, fmt("%.3f %.3f %.3f", s1[0], s1[1], s1[2]));
        const auto s2 = dsb_target({ { 10.0, 1e9 }, { 10.0, 0.0 } }, 8000, 0.02, 1);
        report("a card whose KV reads alone take longer than the whole pass gets no tokens", s2[0] == 0.0 && s2[1] == 1.0, fmt("%.3f %.3f", s2[0], s2[1]));
    }
}

static void test_nc_counts() {
    printf("Testing the token counts per card:\n");
    std::mt19937_64 rng(g_seed + 7);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters * 10; ++it) {
        const int n = 1 + (int) (rng() % 8), x = (int) (rng() % n);
        const int64_t T = 1 + (int64_t) (rng() % 40000), min_tok = 1 + (int64_t) (rng() % 600);
        std::vector<double> s(n);
        for (auto & v : s) { v = (rng() % 5) == 0 ? 0.0 : (double) (rng() % 1000) / 1000.0; }
        const auto c = dsb_counts(T, s, min_tok, x);
        int64_t sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += c[i];
            if (c[i] < 0) { if (bad++ == 0) { first = "negative count"; } }
            if (i != x && c[i] > 0 && c[i] < min_tok) { if (bad++ == 0) { first = fmt("card %d: %lld tokens < %lld", i, (long long) c[i], (long long) min_tok); } }
        }
        if (sum != T) { if (bad++ == 0) { first = fmt("counts add up to %lld of %lld", (long long) sum, (long long) T); } }
        if (c[x] < std::min<int64_t>(min_tok, T)) { if (bad++ == 0) { first = fmt("expert card %lld tokens", (long long) c[x]); } }
    }
    report("counts add up to T, every range at least min_tok, the expert card one", bad == 0, first);
    const auto c = dsb_counts(10000, { 0.45, 0.55 }, 512, 1);
    report("45/55% of 10000 tokens: 4500 + 5500", c[0] == 4500 && c[1] == 5500, fmt("%lld %lld", (long long) c[0], (long long) c[1]));
}

// The live loop against made-up cards whose true per-token costs the balance does not know: it sees each pass's times (with noise) and moves the shares.
static void test_nc_loop() {
    printf("Testing the live balance loop (unknown cards, noisy times):\n");
    std::mt19937_64 rng(g_seed + 11);
    std::normal_distribution<double> noise(0.0, 0.01);
    long bad = 0, n_runs = 0; std::string first;
    for (long it = 0; it < g_iters; ++it) {
        const int n = 2 + (int) (rng() % 5), x = n - 1;
        const double T = 2000 + (double) (rng() % 30000);
        std::vector<double> c(n), hop(n, 0.0), k(n, 0.0);
        for (int i = 0; i < n; ++i) {
            c[i] = 400.0 + (double) (rng() % 1200);                                   // dense us per token, any card any speed
            if (i != x) { hop[i] = (double) (rng() % 300); k[i] = (double) (rng() % 3) == 0 ? (double) (rng() % 200000) : 0.0; }   // its link
        }
        std::vector<double> s(n, 1.0 / n);   // the start: an equal share per card
        double gap = 1.0;
        int passes = 0;
        for (int p = 0; p < 8; ++p) {
            const auto cnt = dsb_counts((int64_t) T, s, 64, x);
            std::vector<dsb_card> meas(n);
            std::vector<double> t(n, 0.0), sr(n, 0.0);
            for (int i = 0; i < n; ++i) {
                sr[i] = (double) cnt[i] / T;
                const double ci = c[i] * (1.0 + noise(rng));                          // what the stamps say: dense time over its tokens
                meas[i] = { ci + hop[i], k[i] };                                          // + the link costs the engine models from the probed rates
                t[i] = cnt[i] > 0 ? k[i] + (double) cnt[i] * (c[i] + hop[i]) : 0.0;    // the true time of the card in this pass
            }
            const auto target = dsb_target(meas, T, 0.02, x);
            s = sr;
            gap = dsb_step(s, target, t, p == 0, 0.03);
            ++passes;
            if (p >= 1 && gap < 0.03) { break; }
        }
        ++n_runs;
        if (gap >= 0.05) { if (bad++ == 0) { first = fmt("%d cards: still %.1f%% apart after %d passes", n, 100 * gap, passes); } }
    }
    report("any 2..6 cards: within 5% after at most 8 passes (the first jumps)", bad == 0, first);
    (void) n_runs;
}

static dsxn_in model_n(int64_t T, int n, int L = 40, int n_dense_first = 3) {
    dsxn_in in;
    in.T = T; in.L = L; in.x = n - 1;
    in.w.assign(L, 1.0); in.xf.assign(L, 0.6); in.E.assign(L, 800e6); in.kv_b.assign(L, 0.0); in.att_w.assign(L, 1.0 / L);
    in.lcard.assign(L, n - 1); in.lB.assign(L, 24.0);
    for (int il = 0; il < n_dense_first; ++il) { in.E[il] = 0.0; in.xf[il] = 0.0; }
    in.w_sum = L; in.c_base = 5.0 * L; in.att_tok = 0.0;
    in.b_in = in.b_out = 6144.0 * 4.0;
    in.rel.assign(n, 1.0); in.B.assign(n, 24.0);
    in.held.assign(n, std::vector<double>(L, 0.0));
    return in;
}

static void test_nc_plan() {
    printf("Testing the N-card planner:\n");
    {   // dense work only, no hand-offs: n equal cards take 1/n of one card's time
        bool ok = true; std::string why;
        for (int n = 2; n <= 6; ++n) {
            auto in = model_n(8000, n, 20, 20);
            in.b_in = in.b_out = 0.0;
            const auto o = dsxn_plan(in);
            ok &= std::fabs(o.t_split * n / o.t_one - 1.0) < 1e-6;
            if (!ok && why.empty()) { why = fmt("%d cards: %.0f vs %.0f / %d", n, o.t_split, o.t_one, n); }
        }
        report("n equal cards, dense work only: 1/n of one card's time", ok, why);
    }
    {   // more equal cards never make the split slower
        bool mono = true; double prev = 1e300;
        for (int n = 2; n <= 6; ++n) { const auto o = dsxn_plan(model_n(8000, n)); mono &= o.t_split <= prev * (1.0 + 1e-9); prev = o.t_split; }
        report("one more equal card never makes the split slower", mono);
    }
    {   // a card on a slow link takes fewer tokens than one of the same compute on a fast link; the expert card (no hand-offs) the most
        auto in = model_n(8000, 3);
        in.B = { 6.0, 24.0, 24.0 };
        const auto o = dsxn_plan(in);
        report("3 cards, one on a 4x slower link: fewer tokens than its twin", o.s[0] < o.s[1] && o.s[1] < o.s[2], fmt("%.3f %.3f %.3f", o.s[0], o.s[1], o.s[2]));
    }
    {   // a faster card takes more tokens than its twin on the same link
        auto in = model_n(8000, 3);
        in.rel = { 2.0, 1.0, 1.0 };
        const auto o = dsxn_plan(in);
        report("3 cards, one twice as fast: more tokens than its twin", o.s[0] > o.s[1], fmt("%.3f %.3f %.3f", o.s[0], o.s[1], o.s[2]));
    }
    {   // a card with almost no compute gets no tokens and the split is no slower than without it
        auto in = model_n(8000, 3);
        in.rel = { 0.001, 1.0, 1.0 };
        const auto o = dsxn_plan(in), o2 = dsxn_plan(model_n(8000, 2));
        report("a card 1000x slower is left out: same time as the other two alone", o.s[0] == 0.0 && std::fabs(o.t_split / o2.t_split - 1.0) < 1e-6,
               fmt("%.3f, %.0f vs %.0f", o.s[0], o.t_split, o2.t_split));
    }
    {   // the KV of a long context read over the links raises the split's time
        auto a = model_n(8000, 2), b = a;
        b.lcard.assign(b.L, 0);
        for (auto & v : b.kv_b) { v = 2e9; }
        report("KV caches read over the links raise the split's time", dsxn_plan(b).t_split > dsxn_plan(a).t_split);
    }
    {   // two cards: the same shape as the two-card search (a helper on a 4x slower link takes under half)
        auto in = model_n(8000, 2);
        in.B = { 6.0, 24.0 };
        const auto o = dsxn_plan(in);
        report("2 cards, the helper on a 4x slower link: under half of the tokens", o.s[0] < 0.5 && o.s[0] > 0.0, fmt("%.3f", o.s[0]));
    }
    {   // hand-offs that block the host (ggml's fallback copy drains both cards, Vulkan between two devices): each crossing costs its measured time on top,
        // nothing overlaps it. MiMo 11k on two Vulkan cards (2026-11-17): split 13.5-16.5 s against one card 11.0 s; priced in stream order the split was 0.91 x one
        auto a = model_n(11011, 2, 48);
        const auto oa = dsxn_plan(a);
        auto b = a;
        b.xb_a = 20000.0; b.xb_b = 1.0 / 4e3;   // 20 ms a crossing, 4 GB/s through the host
        const auto ob = dsxn_plan(b);
        report("hand-offs in stream order: the split beats one card", oa.t_split < oa.t_one, fmt("%.0f vs %.0f", oa.t_split, oa.t_one));
        report("blocking hand-offs: the split is slower than one card", ob.t_split > ob.t_one, fmt("%.0f vs %.0f", ob.t_split, ob.t_one));
        report("blocking hand-offs leave one card's time alone", ob.t_one == oa.t_one);
        auto c = b;
        c.xb_a = 0.0; c.xb_b = 0.0;
        report("no blocking measured: the plan of a pair in stream order", dsxn_plan(c).t_split == oa.t_split && dsxn_plan(c).s == oa.s);
    }
}

// The expert shares of a pass-size bucket with nothing stored yet (2026-11-14, qni4-i -19%): the four readers of the per-bucket shares each had their own
// default for a missing bucket; three read 0 (every expert on the expert card, the helper idle) while a new bucket starts from the expert trial's shares in
// force. A read before the bucket exists must give what the bucket is created with: one function for both (llama_dsplit_f_fresh_fork), one reader in the
// engine (dsplit_f_of); the source check below keeps new per-pass lookups from adding a default of their own.
static void test_f_fresh() {
    printf("expert shares of a bucket with nothing stored:\n");
    const auto eq = [](const std::vector<double> & a, const std::vector<double> & b) {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) { if (std::fabs(a[i] - b[i]) > 1e-12) { return false; } }
        return true;
    };
    report("3 cards, trial shares 0.2 and 0.1: the expert card takes the rest", eq(llama_dsplit_f_fresh_fork(3, { 0.2, 0.1 }), { 0.2, 0.1, 0.7 }));
    report("2 cards, no trial yet: every expert on the expert card", eq(llama_dsplit_f_fresh_fork(2, {}), { 0.0, 1.0 }));
    report("2 cards, trial shares for 3 cards: not used", eq(llama_dsplit_f_fresh_fork(2, { 0.3, 0.1 }), { 0.0, 1.0 }));
    report("no cards: nothing", llama_dsplit_f_fresh_fork(0, { 0.3 }).empty());
    std::mt19937 rng(g_seed);
    bool ok = true;
    for (long it = 0; it < g_iters && ok; ++it) {   // any trial: shares sum to 1, each in [0, 1], the helpers' as the trial's
        const size_t n = 2 + rng() % 4;
        std::vector<double> t(n - 1);
        double sum = 0.0;
        for (auto & x : t) { x = (rng() % 1000) / 1000.0 / (double) n; sum += x; }
        const auto f = llama_dsplit_f_fresh_fork(n, t);
        double fs = 0.0;
        for (size_t i = 0; i < n; ++i) { fs += f[i]; ok = ok && f[i] >= 0.0 && f[i] <= 1.0 && (i + 1 == n || f[i] == t[i]); }
        ok = ok && std::fabs(fs - 1.0) < 1e-9;
    }
    report("any trial: shares sum to 1, helpers keep the trial's", ok);
    // the engine: every per-pass lookup of the shares goes through dsplit_f_of (a lookup of its own brings a default of its own)
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-dsplit-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    int bad = 0;
    std::string fn, where;
    for (size_t pos = 0, nl; pos < src.size(); pos = nl + 1) {
        nl = src.find('\n', pos);
        if (nl == std::string::npos) { nl = src.size(); }
        const std::string line = src.substr(pos, nl - pos);
        if (!line.empty() && line[0] != ' ' && line[0] != '}' && line[0] != '/' && line.find('(') != std::string::npos) { fn = line; }
        if (line.find("g_bal.find(dsplit_bucket_of(n_tokens))") != std::string::npos && fn.find("dsplit_f_of(") == std::string::npos) {
            bad++; where += "\n      in " + fn.substr(0, 90);
        }
    }
    report("source read", !src.empty());
    report("no per-pass share lookup outside dsplit_f_of", bad == 0, fmt("%d%s", bad, where.c_str()));
}

// --moe staging=0: the dense split takes a prompt call from its one-card path only when the call streams into the cache's blocks (its experts come from that stream).
// Qwen3.8 IQ3_S, 2442 tokens, 2 x 3090, 2026-11-14: it took a prompt chain call, its experts became plain graph inputs, every expert of every layer was copied when its
// layer came up (50 GB at 15.9 GB/s): 3.0 s against the chain's 2.45 (742 against 997 prompt t/s).
static void test_may_take_call() {
    printf("the split takes a call from its one-card path:\n");
    report("staged: any call", llama_dsplit_may_take_call_fork(false, false) && llama_dsplit_may_take_call_fork(false, true));
    report("staging=0, a stream call: yes", llama_dsplit_may_take_call_fork(true, true));
    report("staging=0, a chain or decode chain call: no (the failure)", !llama_dsplit_may_take_call_fork(true, false));
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t a = src.find("\nbool llama_moe_cache_pp_yield_fork(");
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    report("the engine's yield asks it", b != std::string::npos && src.substr(a, b - a).find("llama_dsplit_may_take_call_fork(") != std::string::npos);
}

// The zero-wait shares start at factor 1 only when the cards take turns at whole layers' experts (staging=0, every layer on its home card: --moe moe-split=home).
// staging=0 with the expert card computing every expert (the default since the 11-13 placement fix) is the staged case: factor 0.75. Qwen3.8 IQ4_XS 12k, 2 x 3090,
// 2026-11-14: factor 1 gave the helper 49% of the tokens (staged 41%), pass 5452 ms against 4903 on the same build.
static void test_turns() {
    printf("the cards take turns at the experts:\n");
    report("staging=0, experts on the expert card (moe-split unset): no (the failure)", !llama_dsplit_turns_fork(true, 0, -1));
    report("staging=0, moe-split=expert: no", !llama_dsplit_turns_fork(true, 0, 0));
    report("staging=0, moe-split=home: yes", llama_dsplit_turns_fork(true, 0, 1));
    report("staged, moe-split=home: no", !llama_dsplit_turns_fork(false, 0, 1));
    report("experts shared out by the split: no", !llama_dsplit_turns_fork(true, 1, 1));
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-dsplit-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t a = src.find("\nstatic bool dsplit_zw_turns(");
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    report("the split's start factor asks it", b != std::string::npos && src.substr(a, b - a).find("llama_dsplit_turns_fork(") != std::string::npos);
}

// staging=0's uploads into the cache's blocks (cs_copy) count in the pass statistics like the staged uploads: Qwen3.8 IQ4_XS 12k, 2026-11-14, the split balance line
// said "link GB: experts 0.00, link busy 21 ms" on the expert card while about 48 GB went up, so a slow upload and an upload started late looked alike.
static void test_cs_bytes_counted() {
    printf("staging=0 uploads in the pass statistics:\n");
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t a = src.find("\nstatic void cs_copy(");
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    report("cs_copy adds its bytes to the card's uploads", b != std::string::npos && src.substr(a, b - a).find("ggml_backend_sched_upload_add_fork(") != std::string::npos);
}

// The split's hand-off blocks (pinned host memory between two cards, ggml_cuda_xstage_copy) grow inside a pass when a bigger tensor crosses. Freeing the old block there
// (cudaHostUnregister) drains the cards' queues: Qwen3.8 IQ4_XS 12k, request 2 (the expert trial), nsys 2026-11-14: 9 calls of 72-150 ms on the scheduler thread, both
// cards idle 56 and 72 ms. The old block is retired and freed at a card's synchronize instead.
static void test_xstage_no_free_in_pass() {
    printf("hand-off blocks are not freed inside a pass:\n");
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../ggml/src/ggml-cuda/ggml-cuda.cu";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t a = src.find("\nstatic bool ggml_cuda_xstage_copy(");
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    const std::string body = b != std::string::npos ? src.substr(a, b - a) : std::string();
    report("the copy never unregisters or frees a block", !body.empty() && body.find("cudaHostUnregister") == std::string::npos && body.find("cudaFreeHost") == std::string::npos);
    const size_t c = src.find("\nstatic void ggml_backend_cuda_synchronize(");
    const size_t d = c == std::string::npos ? c : src.find("\n}\n", c);
    report("a card's synchronize frees the retired blocks", d != std::string::npos && src.substr(c, d - c).find("ggml_cuda_xstage_reap(") != std::string::npos);
}

// one card under two backends: the stored pass times are per card key, and with one key a Vulkan run read the split time CUDA measured on the same cards
// (MiMo 11k, 2026-11-17: split 7.7 s planned, 15.5 s on Vulkan, where one Vulkan card ran the pass in 11.0 s)
static void test_card_key_backend() {
    const std::string cu = dsplit_card_key("CUDA", "10de2204", "41:00.0", 4, "16", true);
    const std::string vk = dsplit_card_key("Vulkan", "10de2204", "41:00.0", 4, "16", true);
    report("one card has a key per backend", cu != vk, cu + " / " + vk);
    report("a card's key is the same at every start", cu == dsplit_card_key("CUDA", "10de2204", "41:00.0", 4, "16", true));
    report("the run-time check leaves out the link generation (an idle link drops it)",
           dsplit_card_key("CUDA", "10de2204", "41:00.0", 4, "16", false) == dsplit_card_key("CUDA", "10de2204", "41:00.0", 1, "16", false));
    report("two slots are two keys", cu != dsplit_card_key("CUDA", "10de2204", "61:00.0", 4, "16", true));
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_plan_shape();
    test_behaviour();
    test_nc_target();
    test_nc_counts();
    test_nc_loop();
    test_nc_plan();
    test_f_fresh();
    test_may_take_call();
    test_turns();
    test_cs_bytes_counted();
    test_xstage_no_free_in_pass();
    test_card_key_backend();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
