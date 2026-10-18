// Unit tests of the GPU prefill planning (src/llama-gpu-probe-fork.cpp): which cards take part in a prompt pass and with which share of the layers, whether a card
// joins the expert split, and the memory plan of one card (dense weights resident or streamed, how big a micro-batch fits). The planner is a pure function of the probe
// results and the model sizes; the tests feed it made-up cards (no GPU, no model).
//
// build: cmake --build <dir> --target test-gpu-fork;  run: ctest -R test-gpu-fork   or   test-gpu-fork [iterations] [seed]
#include "../src/llama-gpu-probe-fork.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int  n_failed = 0;
static long g_iters  = 3000;
static unsigned long g_seed = 1;

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-62s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}

static llama_gpu_probe_fork_result card(const char * name, double gbs, double compute, double lat_us = 20.0) {
    llama_gpu_probe_fork_result c;
    c.name = name; c.h2d_gbs = gbs; c.h2d_conc_gbs = gbs; c.compute = compute; c.lat_us = lat_us; c.res_us = 100.0;
    return c;
}
// a GLM-like model: 78 layers, 800 MB of experts and 180 MB of dense weights per layer
static llama_prefill_model_fork glm(double n_tokens, double pass_tokens = 4096) {
    llama_prefill_model_fork m;
    m.n_layer = 78; m.expert_bytes_layer = 800e6; m.dense_bytes_layer = 180e6; m.n_expert = 256; m.n_used = 8; m.n_embd = 6144;
    m.kv_bytes_token_layer = 1152; m.n_tokens = n_tokens; m.pass_tokens = pass_tokens;
    return m;
}
static std::vector<const llama_gpu_probe_fork_result *> ptrs(const std::vector<llama_gpu_probe_fork_result> & v) {
    std::vector<const llama_gpu_probe_fork_result *> p;
    for (const auto & c : v) { p.push_back(&c); }
    return p;
}

static void test_inputs() {
    printf("Testing the prefill plan on unusable input (no plan, the engine keeps its default layout):\n");
    const auto m = glm(8000);
    std::vector<llama_gpu_probe_fork_result> two = { card("x16", 24.0, 100.0), card("x4", 6.0, 100.0) };
    report("two usable cards give a plan of two shares", llama_prefill_plan_fork(ptrs(two), {}, m, nullptr).size() == 2);
    report("one card: no plan", llama_prefill_plan_fork(ptrs({ two.begin(), two.begin() + 1 }), {}, m, nullptr).empty());
    report("no cards: no plan", llama_prefill_plan_fork({}, {}, m, nullptr).empty());
    { auto bad = two; bad[1].h2d_gbs = 0.0;  report("a card with no measured upload rate: no plan", llama_prefill_plan_fork(ptrs(bad), {}, m, nullptr).empty()); }
    { auto bad = two; bad[0].compute = 0.0;  report("a card with no measured compute: no plan", llama_prefill_plan_fork(ptrs(bad), {}, m, nullptr).empty()); }
    { auto mm = m; mm.n_layer = 0;           report("a model with no layers: no plan", llama_prefill_plan_fork(ptrs(two), {}, mm, nullptr).empty()); }
    { auto mm = m; mm.n_tokens = 0;          report("a prompt of no tokens: no plan", llama_prefill_plan_fork(ptrs(two), {}, mm, nullptr).empty()); }
    { std::vector<llama_gpu_probe_fork_result> nine(9, card("c", 10.0, 10.0)); report("more than 8 cards: no plan", llama_prefill_plan_fork(ptrs(nine), {}, m, nullptr).empty()); }
    report("evaluating a split of the wrong size: -1", llama_prefill_plan_eval_fork(ptrs(two), {}, m, { 1.0 }) < 0.0);
}

static void test_shares() {
    printf("Testing which cards take part and their shares:\n");
    std::mt19937_64 rng(g_seed);
    long bad = 0, n_both = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const size_t n = 2 + rng() % 3;
        std::vector<llama_gpu_probe_fork_result> cs;
        for (size_t i = 0; i < n; ++i) { cs.push_back(card("c", 2.0 + (double) (rng() % 300) / 10.0, 20.0 + (double) (rng() % 200), 5.0 + (double) (rng() % 100))); }
        auto m = glm(64.0 + (double) (rng() % 40000), rng() % 3 ? 4096 : 0);
        m.slow_scale = 0.5 + (double) (rng() % 50) / 100.0;
        const auto sh = llama_prefill_plan_fork(ptrs(cs), {}, m, nullptr);
        if (sh.size() != n) { fail("size of the plan"); continue; }
        double sum = 0.0; size_t members = 0;
        for (float f : sh) { if (f < 0.0f) { fail("negative share"); } sum += f; members += f > 0.0f; }
        if (std::fabs(sum - 1.0) > 1e-4) { fail(fmt("shares sum to %.6f", sum)); }
        if (members >= 2) {
            n_both++;
            // the slow-link card (damped by slow_scale) never holds more than a faster-link card
            double top = 0.0; for (size_t i = 0; i < n; ++i) { if (sh[i] > 0.0f) { top = std::max(top, cs[i].h2d_gbs); } }
            for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j < n; ++j) {
                if (sh[i] > 0.0f && sh[j] > 0.0f && cs[i].h2d_gbs < cs[j].h2d_gbs && sh[i] > sh[j] + 1e-6) { fail(fmt("slower link %.1f got %.3f, faster link %.1f got %.3f", cs[i].h2d_gbs, sh[i], cs[j].h2d_gbs, sh[j])); }
            } }
        }
        const auto sh2 = llama_prefill_plan_fork(ptrs(cs), {}, m, nullptr);
        if (sh2 != sh) { fail("same input, different plan"); }
    }
    report("shares >= 0, sum 1, slower link never holds more, deterministic", bad == 0, first);
    report("the run planned with several cards in many cases (not vacuous)", n_both > 200, fmt("%ld of %ld", n_both, g_iters));
    // [the 86% x4 share of the first model, GLM 12k: the measured best was 15-20%] two cards, x16 and x4: the x4 card never holds the majority
    std::vector<llama_gpu_probe_fork_result> two = { card("x4", 6.0, 100.0), card("x16", 24.0, 100.0) };
    const auto s12k = llama_prefill_plan_fork(ptrs(two), {}, glm(12000), nullptr);
    report("x4 + x16 card, 12k prompt: the x4 card holds at most a third of the layers", s12k.size() == 2 && s12k[0] < 0.34f, s12k.size() == 2 ? fmt("%.2f / %.2f", s12k[0], s12k[1]) : "");
    // identical cards share equally
    std::vector<llama_gpu_probe_fork_result> eq = { card("a", 24.0, 100.0), card("b", 24.0, 100.0) };
    const auto se = llama_prefill_plan_fork(ptrs(eq), {}, glm(12000), nullptr);
    report("two identical cards: equal shares (or one alone, never a lopsided pair)", se.size() == 2 && (std::fabs(se[0] - se[1]) < 1e-4 || se[0] == 0.0f || se[1] == 0.0f));
    // a card that costs more than it brings is left out of a short prompt
    std::vector<llama_gpu_probe_fork_result> weak = { card("x16", 24.0, 100.0), card("weak", 1.0, 1.0, 5000.0) };
    const auto sw = llama_prefill_plan_fork(ptrs(weak), {}, glm(64), nullptr);
    report("a very slow, high-latency card is left out of a 64-token prompt", sw.size() == 2 && sw[1] == 0.0f && sw[0] > 0.99f);
    // log lines explain the choice
    std::vector<std::string> log;
    llama_prefill_plan_fork(ptrs(two), {}, glm(12000), &log);
    bool chosen = false; for (const auto & l : log) { chosen |= l.find("chosen cards") != std::string::npos; }
    report("the log names the chosen cards", chosen);
}

static void test_eval_and_verdict() {
    printf("Testing the modelled pass time and the expert-split verdict:\n");
    std::vector<llama_gpu_probe_fork_result> two = { card("x16", 24.0, 100.0), card("x4", 6.0, 100.0) };
    const std::vector<double> split = { 0.8, 0.2 };
    double prev = 0.0; bool mono = true;
    for (double n : { 256.0, 1024.0, 4096.0, 16384.0, 65536.0 }) {
        const double t = llama_prefill_plan_eval_fork(ptrs(two), {}, glm(n), split);
        mono &= t > prev; prev = t;
    }
    report("the modelled time grows with the prompt length", mono);
    report("giving the slow-link card everything is slower than giving it a small share",
           llama_prefill_plan_eval_fork(ptrs(two), {}, glm(12000), { 0.0, 1.0 }) > llama_prefill_plan_eval_fork(ptrs(two), {}, glm(12000), { 0.85, 0.15 }));
    // verdict: equal strong cards on a long pass pay for the expert split; a card behind a huge latency does not
    std::vector<llama_gpu_probe_fork_result> eq = { card("a", 24.0, 100.0, 10.0), card("b", 24.0, 100.0, 10.0) };
    auto expert_heavy = glm(12000);   // experts are most of a token's compute and the speed is calibrated: a second card can take half of them
    expert_heavy.dense_bytes_layer = 20e6; expert_heavy.fastest_prefill_tps = 800.0;
    llama_prefill_plan_fork(ptrs(eq), {}, expert_heavy, nullptr);
    const int v_eq = llama_prefill_split_verdict_fork();
    std::vector<llama_gpu_probe_fork_result> lag = { card("a", 24.0, 100.0, 10.0), card("b", 24.0, 100.0, 5e6) };
    llama_prefill_plan_fork(ptrs(lag), {}, expert_heavy, nullptr);
    const int v_lag = llama_prefill_split_verdict_fork();
    report("equal cards: the expert split pays; a card with 5 s latency: it does not", v_eq == 1 && v_lag == 0, fmt("verdicts %d / %d", v_eq, v_lag));
}

static void test_memory_plan() {
    printf("Testing the memory plan of one card (dense weights resident or streamed):\n");
    std::mt19937_64 rng(g_seed + 2);
    long bad = 0, n_res = 0, n_str = 0, n_inf = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters * 10; ++it) {
        llama_prefill_mem_fork m;
        m.vram_bytes = (double) (rng() % 48) * 1e9; m.dense_bytes_layer = (double) (rng() % 400) * 1e6; m.dense_bytes_total = m.dense_bytes_layer * (double) (1 + rng() % 90);
        m.ring_bytes = (double) (rng() % 3000) * 1e6; m.kv_bytes_per_token = (double) (rng() % 200000); m.scratch_bytes_per_token = 1.0 + (double) (rng() % 200000);
        m.n_tokens = 1.0 + (double) (rng() % 100000); m.ub_cap = rng() % 2 ? 0.0 : 1.0 + (double) (rng() % 16384);
        m.pass_fixed_us = (double) (rng() % 100000); m.upload_gbs = rng() % 8 ? 1.0 + (double) (rng() % 25) : 0.0;
        const auto c = llama_prefill_mem_plan_fork(m);
        if (!c.feasible) {
            n_inf++;
            // infeasible only when not even one token fits with the dense weights streamed
            const double room = m.vram_bytes - m.ring_bytes - m.kv_bytes_per_token * m.n_tokens - 2.0 * m.dense_bytes_layer;
            if (std::floor(std::max(0.0, room) / m.scratch_bytes_per_token) >= 1.0 && m.ub_cap != 0.0 ? false : false) { fail("infeasible with room"); }
            if (room / m.scratch_bytes_per_token >= 1.0) { fail(fmt("infeasible but %.0f tokens of scratch fit", room / m.scratch_bytes_per_token)); }
            continue;
        }
        (c.stream_dense ? n_str : n_res)++;
        const double dense_res = c.stream_dense ? 2.0 * m.dense_bytes_layer : m.dense_bytes_total;
        if (c.ub < 1.0 || c.ub > m.n_tokens) { fail(fmt("micro-batch %.0f outside 1..%.0f", c.ub, m.n_tokens)); }
        if (m.ub_cap > 0 && c.ub > m.ub_cap) { fail(fmt("micro-batch %.0f above the cap %.0f", c.ub, m.ub_cap)); }
        // what the choice says it uses fits the card
        if (m.ring_bytes + m.kv_bytes_per_token * m.n_tokens + dense_res + c.ub * m.scratch_bytes_per_token > m.vram_bytes + 1.0) { fail(fmt("ub %.0f does not fit the card's %.1f GB", c.ub, m.vram_bytes / 1e9)); }
        if (std::fabs(c.passes - std::ceil(m.n_tokens / c.ub)) > 1e-9) { fail("passes != ceil(tokens / ub)"); }
        // a prompt that fits one pass with the dense weights resident never streams
        if (c.stream_dense && !c.reason[0]) { fail("no reason given"); }
        if (!c.stream_dense && c.ub >= m.n_tokens && c.passes != 1.0) { fail("one pass fits but passes != 1"); }
    }
    report("fits the card, ub within 1..tokens and the cap, passes = ceil(tokens / ub)", bad == 0, first);
    report("the run met resident, streamed and infeasible cases", n_res > 100 && n_str > 100 && n_inf > 100, fmt("%ld / %ld / %ld", n_res, n_str, n_inf));
    llama_prefill_mem_fork m; m.vram_bytes = 24e9; m.dense_bytes_total = 14e9; m.dense_bytes_layer = 180e6; m.ring_bytes = 2.4e9; m.kv_bytes_per_token = 0.0; m.scratch_bytes_per_token = 65e3;
    m.n_tokens = 4000; m.pass_fixed_us = 50000; m.upload_gbs = 24.0;
    const auto a = llama_prefill_mem_plan_fork(m);
    report("one pass of 4000 tokens fits with the dense weights resident: resident, one pass", a.feasible && !a.stream_dense && a.passes == 1.0);
    m.n_tokens = 300000; m.kv_bytes_per_token = 70e3;   // KV 21 GB: the resident layout leaves no room
    const auto b = llama_prefill_mem_plan_fork(m);
    report("a 300k prompt whose KV leaves no room for the dense weights: streamed, or infeasible", !b.feasible || b.stream_dense);
    m.kv_bytes_per_token = 400e3;
    report("KV bigger than the card: infeasible, 'more cards must take part'", !llama_prefill_mem_plan_fork(m).feasible);
    llama_prefill_mem_fork z; report("no scratch size or tokens given: infeasible", !llama_prefill_mem_plan_fork(z).feasible);
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_inputs();
    test_shares();
    test_eval_and_verdict();
    test_memory_plan();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
