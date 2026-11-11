// Unit tests of the KV fallback ladder (qkv-ram-fallback): f16 -> q8_0 -> q4_0 in the pool -> RAM. Which tier a request gets, that only the tiers the user allowed
// (--moe kv-fallback=f16,k8v8,k4v4,ram | none; a tier not listed is off) are ever used, that every tier is reachable, that a KV that fills up moves up and never down, the capacity formula
// and the host buffer of the RAM tier. Pure functions of the tier capacities (src/llama-kvtier-fork.h); no GPU, no model.
//
// build: cmake --build <dir> --target test-kvtier-fork;  run: ctest -R test-kvtier-fork   or   test-kvtier-fork [iterations] [seed]
#include "../src/llama-kvtier-fork.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int  n_failed = 0;
static long g_iters  = 20000;
static unsigned long g_seed = 1;

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-62s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}

struct caps_t { size_t f16, k8v8, k4v4, ram; };
static size_t cap_of(const caps_t & c, int t) { return t == 0 ? c.f16 : t == 1 ? c.k8v8 : t == 2 ? c.k4v4 : c.ram; }

static void test_list() {
    printf("Testing the kv-fallback list:\n");
    auto eq = [](std::vector<int> a, std::vector<int> b) { return a == b; };
    report("default f16,k8v8,k4v4 allows f16, q8_0, q4_0 and RAM last", eq(kvx_allowed(KVX_DEFAULT), { 0, 1, 2, 3 }));
    report("none is f16 only, no RAM after it", eq(kvx_allowed("none"), { 0 }));
    report("each alone, RAM always last: f16 -> 0,3, k8v8 -> 1,3, k4v4 -> 2,3, ram -> 3", eq(kvx_allowed("f16"), { 0, 3 }) && eq(kvx_allowed("k8v8"), { 1, 3 }) && eq(kvx_allowed("k4v4"), { 2, 3 }) && eq(kvx_allowed("ram"), { 3 }));
    report("unknown words and an empty list leave f16 and RAM", eq(kvx_allowed("q5_0,foo"), { 0, 3 }) && eq(kvx_allowed(""), { 0, 3 }));
    {   // a word that is no tier is reported, not silently read as "no fallback"; case and blanks do not matter
        std::vector<std::string> u;
        const auto a = kvx_allowed("RAM, k8v8,q8", &u);
        report("unknown words are reported (q8), case and blanks ignored (RAM, k8v8 -> tiers 1 and 3)", eq(a, { 1, 3 }) && u.size() == 1 && u[0] == "q8");
        std::vector<std::string> e;
        kvx_allowed("", &e);
        report("an empty list is reported", e.size() == 1);
        std::vector<std::string> n;
        kvx_allowed("none", &n);
        report("none is a list, not an error", n.empty());
    }
    report("the order of the list does not matter, a repeat counts once", eq(kvx_allowed("ram,k8v8,ram"), { 1, 3 }));
    report("a wrong word between right ones is skipped", eq(kvx_allowed("k8v8,xx,ram"), { 1, 3 }));
}

static void test_select_properties() {
    printf("Testing the tier choice (only allowed tiers, the lowest that holds the request):\n");
    std::mt19937_64 rng(g_seed);
    long bad = 0, hit[4] = { 0, 0, 0, 0 }, none = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        caps_t c; c.f16 = rng() % 5000; c.k8v8 = c.f16 + rng() % 5000; c.k4v4 = c.k8v8 + rng() % 8000; c.ram = c.k4v4 + rng() % 100000;
        if (rng() % 8 == 0) { c.ram = rng() % 1000; }   // a RAM tier that holds less than the pool (little free RAM)
        const unsigned mask = (unsigned) (rng() % 16);   // which of f16, k8v8, k4v4, ram the user allowed
        std::string list; for (int t = 0; t < 4; ++t) { if (mask >> t & 1) { list += (list.empty() ? "" : ","); list += t == 0 ? "f16" : t == 1 ? "k8v8" : t == 2 ? "k4v4" : "ram"; } }
        const auto allowed = kvx_allowed(mask ? list : "none");
        const int kv_tier = (int) (rng() % 4), min_tier = (int) (rng() % 5) - 1;
        const size_t n = rng() % 120000;
        const int t = kvx_select(c, allowed, kv_tier, min_tier, n);
        const int lo = min_tier < 0 ? std::min(kv_tier + 1, 3) : min_tier;
        auto is_allowed = [&](int x) { return std::find(allowed.begin(), allowed.end(), x) != allowed.end(); };
        // eligible tiers: the allowed tiers at or above lo (f16 only when listed)
        int expect = -1;
        for (int x = 0; x < 4 && expect < 0; ++x) { if (is_allowed(x) && x >= lo && n <= cap_of(c, x)) { expect = x; } }
        if (t != expect) { fail(fmt("n %zu caps %zu/%zu/%zu/%zu list '%s' tier %d min %d: got %d, expected %d", n, c.f16, c.k8v8, c.k4v4, c.ram, list.c_str(), kv_tier, min_tier, t, expect)); }
        if (t >= 0) {
            if (!is_allowed(t)) { fail(fmt("tier %d chosen but not allowed (list '%s')", t, list.c_str())); }
            if (n > cap_of(c, t)) { fail("the chosen tier does not hold the request"); }
            if (t < lo) { fail(fmt("tier %d below the lowest allowed %d", t, lo)); }
            hit[t]++;
        } else { none++; }
    }
    report("the chosen tier is allowed, holds the request, is the lowest that does", bad == 0, first);
    report("the run chose every tier and refused some requests", hit[0] > 50 && hit[1] > 50 && hit[2] > 50 && hit[3] > 50 && none > 100, fmt("%ld/%ld/%ld/%ld, refused %ld", hit[0], hit[1], hit[2], hit[3], none));
}

static void test_ladder() {
    printf("Testing the ladder end to end (a request grows past every tier, and what each list allows):\n");
    const caps_t c = { 1000, 1900, 3500, 20000 };   // f16 holds 1000 tokens, q8_0 1900, q4_0 3500, RAM 20000
    struct step { size_t n; int expect_all, expect_k8only, expect_k4only, expect_ramonly, expect_none; };
    const step steps[] = {
        {   500,  0,  0,  0,  0,  0 },
        {  1000,  0,  0,  0,  0,  0 },
        {  1500,  1,  1,  2,  3, -1 },
        {  1900,  1,  1,  2,  3, -1 },
        {  3000,  2,  3,  2,  3, -1 },   // RAM is the last resort of every list but none
        {  3500,  2,  3,  2,  3, -1 },
        { 10000,  3,  3,  3,  3, -1 },
        { 20000,  3,  3,  3,  3, -1 },
        { 20001, -1, -1, -1, -1, -1 },
    };
    bool ok_all = true; std::string why;
    // a new request (the server asks with min_tier 0): the lowest allowed tier that holds it
    for (const auto & s : steps) {
        const int got[5] = { kvx_select(c, kvx_allowed("f16,k8v8,k4v4,ram"), 0, 0, s.n), kvx_select(c, kvx_allowed("f16,k8v8"), 0, 0, s.n), kvx_select(c, kvx_allowed("f16,k4v4"), 0, 0, s.n),
                             kvx_select(c, kvx_allowed("f16,ram"), 0, 0, s.n), kvx_select(c, kvx_allowed("none"), 0, 0, s.n) };
        const int want[5] = { s.expect_all, s.expect_k8only, s.expect_k4only, s.expect_ramonly, s.expect_none };
        for (int i = 0; i < 5; ++i) { if (got[i] != want[i]) { ok_all = false; why = fmt("%zu tokens, list %d: got %d expected %d", s.n, i, got[i], want[i]); } }
    }
    report("a new request lands on the lowest allowed tier that holds it, every list", ok_all, why);
    report("a list without f16 starts at its first tier (k8v8,ram: q8_0 for a short request)", kvx_select(c, kvx_allowed("k8v8,ram"), 0, 0, 500) == 1);
    report("the list 'ram' alone: RAM from the first token, every layer, refused above it",
           kvx_select(c, kvx_allowed("ram"), 0, 0, 500) == 3 && kvx_select(c, kvx_allowed("ram"), 0, 0, 20001) == -1 &&
           kvx_ram_only(kvx_allowed("ram")) && !kvx_ram_only(kvx_allowed("f16,ram")) && !kvx_ram_only(kvx_allowed(KVX_DEFAULT)));
    // the KV fills up mid-answer (min_tier -1): the cache goes above the tier it has, step by step, to RAM, then stays in RAM
    {
        const auto all = kvx_allowed("f16,k8v8,k4v4,ram");
        int tier = 0; std::vector<int> path;
        for (size_t n : { (size_t) 1100, (size_t) 2000, (size_t) 4000, (size_t) 6000 }) { const int t = kvx_select(c, all, tier, -1, n); if (t < 0) { break; } tier = t; path.push_back(t); }
        report("a KV that fills up climbs f16 -> q8_0 -> q4_0 -> RAM", path == std::vector<int>({ 1, 2, 3, 3 }), path.size() ? fmt("path %d %d %d %d", path[0], path.size() > 1 ? path[1] : -1, path.size() > 2 ? path[2] : -1, path.size() > 3 ? path[3] : -1) : "no path");
        report("full in RAM and still too small: refused, not looped", kvx_select(c, all, 3, -1, 30000) == -1);
    }
    {   // fill-up with q8_0 not allowed skips it; with nothing allowed it is refused
        report("fill-up from f16 with q8_0 off goes straight to q4_0", kvx_select(c, kvx_allowed("f16,k4v4,ram"), 0, -1, 1100) == 2);
        report("fill-up from f16 with only RAM allowed goes to RAM", kvx_select(c, kvx_allowed("f16,ram"), 0, -1, 1100) == 3);
        report("fill-up with the list 'none': refused", kvx_select(c, kvx_allowed("none"), 0, -1, 1100) == -1);
    }
    report("RAM takes q4_0 first when the list has it and the cache is below q4_0",
           kvx_ram_needs_q4(kvx_allowed("k8v8,k4v4,ram"), 0) && kvx_ram_needs_q4(kvx_allowed("k4v4,ram"), 1) && !kvx_ram_needs_q4(kvx_allowed("k4v4,ram"), 2) &&
           !kvx_ram_needs_q4(kvx_allowed("k8v8,ram"), 0) && !kvx_ram_needs_q4(kvx_allowed("ram"), 0));
}

static void test_zone_first() {
    printf("Testing quant first (strict, user 2026-11-11): no expert block while a listed tier fits the zone, then the most compact at the floor, then RAM:\n");
    const caps_t c = { 1000, 1900, 3500, 20000 };   // at the experts' floor
    const size_t z[3] = { 300, 560, 1050 };          // in the zone's start size
    struct one { const char * list; int kv_tier, min_tier; size_t n; int want; const char * what; };
    const one cases[] = {
        { KVX_DEFAULT, 0, 0,   200,  0, "f16 inside the zone" },
        { KVX_DEFAULT, 0, 0,   500,  1, "q8_0 inside the zone before f16 takes blocks" },
        { KVX_DEFAULT, 0, 0,  1000,  2, "q4_0 inside the zone although f16 holds it at the floor" },
        { KVX_DEFAULT, 0, 0,  1500,  2, "no tier in the zone: the most compact takes blocks" },
        { KVX_DEFAULT, 0, 0,  3500,  2, "q4_0 down to the floor" },
        { KVX_DEFAULT, 0, 0,  3600,  3, "then RAM to keep running" },
        { KVX_DEFAULT, 0, 0, 20001, -1, "nothing holds it" },
        { "f16,k8v8",  0, 0,  1000,  1, "no q4_0 listed: q8_0 takes blocks" },
        { "f16,k8v8",  0, 0,  2000,  3, "no q4_0 listed: RAM after q8_0's floor" },
        { "f16",       0, 0,   500,  0, "f16 alone takes blocks" },
        { "f16",       0, 0,  1500,  3, "f16 alone, then RAM" },
        { "none",      0, 0,   500,  0, "none: f16 takes blocks" },
        { "none",      0, 0,  1500, -1, "none: no RAM" },
        { "ram",       0, 0,   500,  3, "ram alone: RAM from the first token" },
        { KVX_DEFAULT, 1, -1,  1200,  2, "filled up at q8_0: q4_0, at the floor" },
        { KVX_DEFAULT, 3, -1,  5000,  3, "filled up in RAM: more layers to RAM" },
    };
    bool ok = true; std::string why;
    for (const auto & k : cases) {
        const int got = kvx_select_zone_first(c, z, kvx_allowed(k.list), k.kv_tier, k.min_tier, k.n);
        if (got != k.want) { ok = false; why = fmt("%s (%s, %zu tokens): got %d, expected %d", k.what, k.list, k.n, got, k.want); }
    }
    report("each case lands where the order says", ok, why);
    std::mt19937_64 rng(g_seed + 2);
    long bad = 0, took = 0, zoned = 0; std::string first;
    for (long it = 0; it < g_iters; ++it) {
        caps_t f; f.f16 = rng() % 5000; f.k8v8 = f.f16 + rng() % 5000; f.k4v4 = f.k8v8 + rng() % 8000; f.ram = rng() % 2 ? f.k4v4 + rng() % 100000 : rng() % 1000;
        size_t zz[3]; zz[0] = rng() % (f.f16 + 1); zz[1] = std::min(f.k8v8, zz[0] * 2 + rng() % 50); zz[2] = std::min(f.k4v4, zz[1] * 2 + rng() % 50);
        const unsigned mask = (unsigned) (rng() % 16);
        std::string list; for (int t = 0; t < 4; ++t) { if (mask >> t & 1) { list += (list.empty() ? "" : ","); list += t == 0 ? "f16" : t == 1 ? "k8v8" : t == 2 ? "k4v4" : "ram"; } }
        const auto al = kvx_allowed(mask ? list : "none");
        const int kv_tier = (int) (rng() % 4), min_tier = (int) (rng() % 5) - 1, lo = min_tier < 0 ? std::min(kv_tier + 1, 3) : min_tier;
        const size_t n = rng() % 30000;
        const int t = kvx_select_zone_first(f, zz, al, kv_tier, min_tier, n);
        auto listed = [&](int x) { return std::find(al.begin(), al.end(), x) != al.end(); };
        int in_zone = -1, top = -1;
        for (int x = 2; x >= 0; --x) { if (listed(x) && x >= lo) { if (top < 0) { top = x; } if (n <= zz[x]) { in_zone = x; } } }
        int want = in_zone >= 0 ? in_zone : (top >= 0 && n <= cap_of(f, top)) ? top : (listed(3) && n <= f.ram) ? 3 : -1;
        if (t != want) { if (bad++ == 0) { first = fmt("n %zu zone %zu/%zu/%zu floor %zu/%zu/%zu/%zu list '%s' tier %d min %d: got %d want %d", n, zz[0], zz[1], zz[2], f.f16, f.k8v8, f.k4v4, f.ram, list.c_str(), kv_tier, min_tier, t, want); } }
        if (t >= 0 && t < 3 && n > zz[t]) { took++; } else if (t >= 0 && t < 3) { zoned++; }
    }
    report("random caps and lists: the zone first, the most compact at the floor next, RAM last", bad == 0, first);
    report("the run took blocks and stayed in the zone both", took > 50 && zoned > 50, fmt("%ld took blocks, %ld in the zone", took, zoned));
}

static void test_capacity_and_host() {
    printf("Testing capacities and the host buffer of the RAM tier:\n");
    // the cells grow as the row shrinks: q8_0 rows are 34/64 of f16, q4_0 18/64 (per value: 2 bytes, 34/32, 18/32 -> 1.0 : 0.53 : 0.28)
    const size_t tail = 10000000, lend0 = 400000, F = 50000, R16 = 4000, pc = 0;
    const size_t c16 = kvx_cells(tail, lend0, F, R16, pc), c8 = kvx_cells(tail, lend0, F, R16 * 34 / 64, pc), c4 = kvx_cells(tail, lend0, F, R16 * 18 / 64, pc);
    report("cells grow down the ladder: f16 < q8_0 < q4_0 (about 1.9x and 3.5x)", c16 < c8 && c8 < c4 && c8 * 10 > c16 * 17 && c4 * 10 > c16 * 33, fmt("%zu / %zu / %zu", c16, c8, c4));
    report("a lend that grows per cell (the attention's f16 copies of quantized K/V) takes cells off", kvx_cells(tail, lend0, F, R16, 2000) < c16);
    report("no room behind the lend and the fixed part: 0 cells", kvx_cells(1000, 5000, 0, 100, 0) == 0 && kvx_cells(1000, 500, 600, 100, 0) == 0 && kvx_cells(1000, 0, 0, 0, 0) == 0);
    std::mt19937_64 rng(g_seed + 1);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters; ++it) {
        const size_t n_seg = rng() % 200, r_all = 1 + rng() % 400000, budget = rng() % ((size_t) 64 << 30), cmax = 1 + rng() % 2000000;
        const size_t hc = kvx_host_cells(budget, n_seg, r_all, cmax);
        if (hc > cmax) { bad++; first = "more cells than the model's context"; }
        if (hc > 0 && kvx_host_size(hc, r_all, n_seg) > budget) { bad++; first = fmt("buffer %zu over the budget %zu", kvx_host_size(hc, r_all, n_seg), budget); }
        if (hc > 0 && hc < cmax && kvx_host_size(hc + 1, r_all, n_seg) <= budget) { bad++; first = "room for one more cell"; }
        // the arrays: n_seg strides summing to r_all, each padded to 256: they fit the buffer (GGML_ASSERT(hused <= hsize) in the engine)
        std::vector<size_t> st; size_t left = r_all;
        for (size_t i = 0; i < n_seg && left > 0; ++i) { const size_t x = i + 1 == n_seg ? left : 1 + rng() % left; st.push_back(x); left -= x; }
        if (left > 0 && n_seg == 0) { continue; }
        if (n_seg > 0 && kvx_host_used(hc, st) > kvx_host_size(hc, r_all, n_seg)) { bad++; first = fmt("arrays use %zu of a %zu buffer", kvx_host_used(hc, st), kvx_host_size(hc, r_all, n_seg)); }
    }
    report("host buffer: inside the budget, at most the context, all arrays fit it", bad == 0, first);
    report("a budget below the padding slack holds no cells", kvx_host_cells(100, 10, 50, 1000) == 0);
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_list();
    test_select_properties();
    test_ladder();
    test_zone_first();
    test_capacity_and_host();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
