// Unit tests of the decisions around the expert cache that can break and have no GPU in them (header-only, the engine's own code: src/llama-poolalloc-fork.h and
// src/llama-passplan-fork.h):
//   hot cache      poolx_pick_slot  : which block an expert takes (free first, else the coldest evictable; never a lent, zone, busy or streaming block)
//   prefetch       ppx_stage_slots  : staging slots behind the graph (at least 3 or none, inside the lendable bytes, at most the configured number)
//   prompt passes  ppx_equal_pass   : equal passes, never a nearly empty last pass
//   dense split    ppx_ds_edge      : the token ranges of the cards (both non-empty, each at least the minimum when the pass allows)
//
// build: cmake --build <dir> --target test-pass-fork;  run: ctest -R test-pass-fork   or   test-pass-fork [iterations] [seed]
#include "../src/llama-passplan-fork.h"
#include "../src/llama-poolalloc-fork.h"

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

// ---- hot cache ----------------------------------------------------------------------------------------------------------------------------------
struct hps { std::vector<int32_t> owner, free; };
struct hot_state { hps ps; std::vector<char> busy; std::vector<int32_t> expert; std::vector<double> rank; };   // per block (the owner's layer is in ps.owner)

static void test_hot_cache() {
    printf("Testing the hot cache victim choice:\n");
    std::mt19937_64 rng(g_seed);
    long bad = 0, n_free = 0, n_victim = 0, n_none = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const int N = 1 + (int) (rng() % 40), NL = 1 + (int) (rng() % 4);
        hot_state h; h.ps.owner.resize(N); h.busy.resize(N); h.expert.resize(N); h.rank.resize(N);
        const bool with_free = rng() % 4 == 0;
        for (int b = 0; b < N; ++b) {
            const int r = (int) (rng() % 10);
            h.ps.owner[b] = r == 0 ? -2 : r == 1 ? -3 : r == 2 && with_free ? -1 : (int32_t) (rng() % NL);
            if (h.ps.owner[b] == -1) { h.ps.free.push_back(b); }
            h.busy[b]   = rng() % 6 == 0;
            h.expert[b] = rng() % 8 == 0 ? -1 : (int32_t) (rng() % 64);
            h.rank[b]   = rng() % 5 == 0 ? -1.0 : (double) (rng() % 1000);   // < 0: may not be evicted
        }
        const int32_t n_active = (int32_t) (rng() % (N + 1));
        double best = 0;
        const poolx_pick p = poolx_pick_slot(h.ps, n_active,
                [&](int32_t, int32_t b) { return h.busy[b] != 0; }, [&](int32_t, int32_t b) { return h.expert[b]; },
                [&](int32_t, int32_t e) { double r = -1; for (int b = 0; b < N; ++b) { if (h.expert[b] == e) { r = h.rank[b]; break; } } return r; }, best);
        // the oracle: what may be handed out
        if (!h.ps.free.empty()) {
            n_free++;
            const int32_t hi = *std::max_element(h.ps.free.begin(), h.ps.free.end());
            if (p.slot != hi || best != 0) { fail(fmt("free blocks exist: expected the highest (%d), got %d", hi, p.slot)); }
            continue;
        }
        if (p.slot >= 0) {
            n_victim++;
            if (p.slot >= n_active) { fail(fmt("block %d is past the active blocks %d", p.slot, n_active)); }
            const int32_t o = h.ps.owner[p.slot];
            if (o == -2 || o == -3) { fail(fmt("handed out block %d with owner %d (lent / under the zone)", p.slot, o)); }
            if (o >= 0 && h.busy[p.slot]) { fail(fmt("handed out a busy block %d", p.slot)); }
            if (o >= 0 && best != 0 && h.expert[p.slot] >= 0) {
                double rv = -1; for (int c = 0; c < N; ++c) { if (h.expert[c] == h.expert[p.slot]) { rv = h.rank[c]; break; } }
                if (rv < 0) { fail(fmt("evicted block %d whose rank %.0f says it may not be evicted", p.slot, rv)); }
                // nothing evictable before it has a lower (>= 0) rank; ranks here are looked up by expert id, so recompute the same way
                for (int b = 0; b < n_active; ++b) {
                    const int32_t ob = h.ps.owner[b];
                    if (ob < 0 || h.busy[b] || h.expert[b] < 0) { continue; }
                    double r = -1; for (int c = 0; c < N; ++c) { if (h.expert[c] == h.expert[b]) { r = h.rank[c]; break; } }
                    if (r >= 0 && r < best) { fail(fmt("block %d has rank %.0f, below the victim's %.0f", b, r, best)); }
                }
            }
        } else {
            n_none++;
            for (int b = 0; b < n_active; ++b) {
                const int32_t o = h.ps.owner[b];
                if (o == -1) { fail("a free block was there but none was handed out"); }
                if (o >= 0 && !h.busy[b] && h.expert[b] < 0) { fail("an empty owned block was there but none was handed out"); }
            }
        }
    }
    report("free block first (the highest); else the coldest evictable; never lent / zone / busy", bad == 0, first);
    report("the run met all three cases (free, victim, nothing evictable)", n_free > 100 && n_victim > 100 && n_none > 10, fmt("%ld free, %ld victim, %ld none", n_free, n_victim, n_none));
    // fixed cases
    hot_state h; h.ps.owner = { -2, -3, 0, 1 }; h.busy = { 0, 0, 1, 0 }; h.expert = { 5, 5, 6, 7 }; h.rank = { 1, 1, 1, 9 };
    double best = 0;
    const poolx_pick p = poolx_pick_slot(h.ps, 4, [&](int32_t, int32_t b) { return h.busy[b] != 0; }, [&](int32_t, int32_t b) { return h.expert[b]; },
            [&](int32_t, int32_t e) { return e == 7 ? 9.0 : 1.0; }, best);
    report("lent, zone and busy blocks skipped: the only candidate is block 3", p.slot == 3 && p.owner == 1 && best == 9.0);
    {   // regression: a free-list entry whose block was lent or taken by the zone since is never picked (the highest entry used to win without a look at the owner)
        hot_state st; st.ps.owner = { -1, -3, -2, 0 }; st.ps.free = { 0, 1, 2 }; st.busy = { 0, 0, 0, 0 }; st.expert = { -1, -1, -1, 4 }; st.rank = { 0, 0, 0, 5 };
        double b3 = 0;
        const poolx_pick q = poolx_pick_slot(st.ps, 4, [&](int32_t, int32_t) { return false; }, [&](int32_t, int32_t b) { return st.expert[b]; }, [&](int32_t, int32_t) { return 5.0; }, b3);
        report("stale free entries (blocks lent or under the zone) are skipped: the free block 0 is picked", q.slot == 0 && b3 == 0);
        hot_state su; su.ps.owner = { -3, -2 }; su.ps.free = { 0, 1 }; su.busy = { 0, 0 }; su.expert = { -1, -1 }; su.rank = { 0, 0 };
        const poolx_pick q2 = poolx_pick_slot(su.ps, 2, [&](int32_t, int32_t) { return false; }, [&](int32_t, int32_t b) { return su.expert[b]; }, [&](int32_t, int32_t) { return 5.0; }, b3);
        report("only stale free entries: no pick", q2.slot < 0);
    }
    report("no pick when every block is lent, zone or busy",
            [&]() { hot_state g; g.ps.owner = { -2, -3, 0 }; g.busy = { 0, 0, 1 }; g.expert = { 1, 1, 1 }; double b2 = 0;
                    return poolx_pick_slot(g.ps, 3, [&](int32_t, int32_t b) { return g.busy[b] != 0; }, [&](int32_t, int32_t) { return 1; }, [&](int32_t, int32_t) { return 1.0; }, b2).slot == -1; }());
}

// ---- prefetch staging ---------------------------------------------------------------------------------------------------------------------------
static void test_prefetch() {
    printf("Testing prefetch staging slots:\n");
    std::mt19937_64 rng(g_seed + 1);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const size_t slot_b = rng() % 5 == 0 ? 0 : 1 + rng() % (64 << 20), stage_off = rng() % (1 << 30), cap_b = rng() % (3u << 30);
        const int n_cfg = (int) (rng() % 12) - 1, plan_k = rng() % 3 ? -1 : (int) (rng() % 12);
        const int n = ppx_stage_slots(slot_b, cap_b, stage_off, n_cfg, plan_k);
        if (n != 0 && n < 3) { fail(fmt("%d slots: fewer than 3 must be none", n)); }
        if (n > std::max(0, n_cfg)) { fail(fmt("%d slots, only %d configured", n, n_cfg)); }
        if (plan_k >= 0 && n > plan_k) { fail(fmt("%d slots, the planner chose %d", n, plan_k)); }
        if (n > 0 && stage_off + (size_t) n * slot_b > cap_b) { fail(fmt("%d slots of %zu from %zu leave the %zu lendable bytes", n, slot_b, stage_off, cap_b)); }
        if (n > 0 && slot_b == 0) { fail("slots of 0 bytes"); }
        // not too few: one more slot would still have been allowed
        if (n >= 3 && n < n_cfg && (plan_k < 0 || n < plan_k) && stage_off + (size_t) (n + 1) * slot_b <= cap_b) { fail(fmt("%d slots, room and permission for one more", n)); }
    }
    report("3 or none; at most configured / planned; inside the lendable bytes; no room wasted", bad == 0, first);
    report("room for exactly 2 slots gives none, for 3 gives 3", ppx_stage_slots(100, 250, 0, 8, -1) == 0 && ppx_stage_slots(100, 300, 0, 8, -1) == 3);
}

// ---- prompt passes ------------------------------------------------------------------------------------------------------------------------------
static void test_passes() {
    printf("Testing equal prompt passes:\n");
    std::mt19937_64 rng(g_seed + 2);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const uint32_t n = 1 + (uint32_t) (rng() % 200000), ub = 1 + (uint32_t) (rng() % 16384), mb = 1 + (uint32_t) (rng() % 600);
        const uint32_t u = ppx_equal_pass(n, ub, mb);
        if (u == 0 || u > ub) { fail(fmt("pass size %u outside 1..%u", u, ub)); continue; }
        const uint32_t passes = (n + u - 1) / u, last = n - (passes - 1) * u;
        if (u == ub) {
            if (n > ub && ub > mb && n % ub > mb && (n + (n + ub - 1) / ub - 1) / ((n + ub - 1) / ub) != ub) { fail(fmt("n %u ub %u mb %u: a tail of %u that the cache path does not take was left", n, ub, mb, n % ub)); }
        } else {
            if (passes != (n + ub - 1) / ub) { fail(fmt("equal passes %u more than the %u full passes needed", passes, (n + ub - 1) / ub)); }
            if (last + passes < u) { fail(fmt("last pass %u of size %u: nearly empty", last, u)); }   // equal passes differ by less than one per pass
        }
    }
    report("one pass size for the prompt: <= ub, as few passes as full ones, no stranded tail", bad == 0, first);
    report("a prompt of 1500 with ub 1024 and max batch 31 runs two equal passes (750)", ppx_equal_pass(1500, 1024, 31) == 750);
    report("a tail the cache path takes (1030 with ub 1024, mb 31) keeps full passes", ppx_equal_pass(1030, 1024, 31) == 1024);
}

// ---- dense split --------------------------------------------------------------------------------------------------------------------------------
static void test_dense_split() {
    printf("Testing dense split token ranges:\n");
    std::mt19937_64 rng(g_seed + 3);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const int64_t min_t = 1 + (int64_t) (rng() % 1024), n = min_t + (int64_t) (rng() % 40000);   // the planner runs a pass of at least the minimum
        const double share = (double) (rng() % 1001) / 1000.0;
        const int64_t e = ppx_ds_edge(n, min_t, share);
        if (n >= 2) {
            if (e < 1 || e > n - 1) { fail(fmt("n %lld min %lld share %.3f: edge %lld leaves a range empty", (long long) n, (long long) min_t, share, (long long) e)); continue; }
            if (n >= 2 * min_t && (e < min_t || n - e < min_t)) { fail(fmt("n %lld min %lld: ranges %lld / %lld under the minimum", (long long) n, (long long) min_t, (long long) e, (long long) (n - e))); }
        }
    }
    report("both ranges non-empty; each at least the minimum when the pass has room", bad == 0, first);
    report("a pass of exactly the minimum (512 tokens) still gives two ranges", ppx_ds_edge(512, 512, 0.55) >= 1 && ppx_ds_edge(512, 512, 0.55) <= 511);
    report("one token is one range (edge = n), not an empty second range", ppx_ds_edge(1, 512, 0.55) == 1 && ppx_ds_edge(0, 512, 0.55) == 0);
    report("share 0.55 of 4000 tokens: first range 2200", ppx_ds_edge(4000, 512, 0.55) == 2200);
    report("share is monotone: more share, no smaller first range", [&]() { int64_t prev = 0; for (int i = 0; i <= 100; ++i) { const int64_t e = ppx_ds_edge(5000, 512, i / 100.0); if (e < prev) { return false; } prev = e; } return true; }());
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_hot_cache();
    test_prefetch();
    test_passes();
    test_dense_split();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
