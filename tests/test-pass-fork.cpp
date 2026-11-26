// Unit tests of the decisions around the expert cache that can break and have no GPU in them (header-only, the engine's own code: src/llama-poolalloc-fork.h and
// src/llama-passplan-fork.h):
//   hot cache      poolx_pick_slot  : which block an expert takes (free first, else the coldest evictable; never a lent, zone, busy or streaming block)
//   prefetch       ppx_stage_slots  : staging slots behind the graph (at least 2 or none, inside the lendable bytes, at most the configured number)
//   prompt passes  ppx_equal_pass   : equal passes, never a nearly empty last pass
//   dense split    ppx_ds_edge      : the token ranges of the cards (both non-empty, each at least the minimum when the pass allows)
//   stream chunks  ppx_pick_chunks  : the chunk count of a staging=0 stream call (ring reserve against passes against entries; a pass costs its link upload or
//                                     its SSD reads, ppx_stream_pass_us)
//   checkpoint cut ppx_cut_last     : the last pass of a request cut off as its own call (fewest passes, the cut on a 256-cell pad, the last pass never longer
//                                     than the equal ones when the passes before it can take the pad)
//
// build: cmake --build <dir> --target test-pass-fork;  run: ctest -R test-pass-fork   or   test-pass-fork [iterations] [seed]
#include "../src/llama-passplan-fork.h"
#include "../src/llama-poolalloc-fork.h"
#include "../ggml/src/ggml-sched-fork.h"

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
    {   // regression: an empty block found after a candidate: the candidate's owner must not stay in the answer, a caller evicts by owner and slot (2026-11-02: an adoption took expert -1 of that owner)
        hot_state se; se.ps.owner = { 0, 1 }; se.busy = { 0, 0 }; se.expert = { 4, -1 }; se.rank = { 5, 0 };
        double b4 = 0;
        const poolx_pick q = poolx_pick_slot(se.ps, 2, [&](int32_t, int32_t) { return false; }, [&](int32_t, int32_t b) { return se.expert[b]; }, [&](int32_t, int32_t) { return 5.0; }, b4);
        report("an owned empty block after a candidate: picked, with no owner", q.slot == 1 && q.owner == -1 && b4 == 0);
        hot_state sf; sf.ps.owner = { 0, -1 }; sf.busy = { 0, 0 }; sf.expert = { 4, -1 }; sf.rank = { 5, 0 };
        const poolx_pick q3 = poolx_pick_slot(sf.ps, 2, [&](int32_t, int32_t) { return false; }, [&](int32_t, int32_t b) { return sf.expert[b]; }, [&](int32_t, int32_t) { return 5.0; }, b4);
        report("a free block after a candidate, not on the free list: picked, with no owner", q3.slot == 1 && q3.owner == -1);
    }
    report("no pick when every block is lent, zone or busy",
            [&]() { hot_state g; g.ps.owner = { -2, -3, 0 }; g.busy = { 0, 0, 1 }; g.expert = { 1, 1, 1 }; double b2 = 0;
                    return poolx_pick_slot(g.ps, 3, [&](int32_t, int32_t b) { return g.busy[b] != 0; }, [&](int32_t, int32_t) { return 1; }, [&](int32_t, int32_t) { return 1.0; }, b2).slot == -1; }());
}

// ---- prefetch staging ---------------------------------------------------------------------------------------------------------------------------
// the picks of a step from one scan (poolx_pick_order) against a scan a pick (poolx_pick_slot), while picked blocks are taken as the cache step takes them
static void test_hot_order() {
    printf("Testing the hot cache's picks of a step from one scan:\n");
    std::mt19937_64 rng(g_seed + 17);
    long bad = 0, n_pick = 0, n_take = 0, n_rescan = 0; std::string first;
    for (long it = 0; it < g_iters; ++it) {
        const int N = 1 + (int) (rng() % 60), NL = 1 + (int) (rng() % 4);
        hot_state h; h.ps.owner.resize(N); h.busy.resize(N); h.expert.resize(N); h.rank.resize(N);
        const bool with_free = rng() % 4 == 0;
        for (int b = 0; b < N; ++b) {
            const int r = (int) (rng() % 10);
            h.ps.owner[b] = r == 0 ? -2 : r == 1 ? -3 : r == 2 && with_free ? -1 : (int32_t) (rng() % NL);
            if (h.ps.owner[b] == -1 && rng() % 4 != 0) { h.ps.free.push_back(b); }   // one in four free blocks is not in the stack
            h.busy[b]   = rng() % 6 == 0;
            h.expert[b] = rng() % 8 == 0 ? -1 : (int32_t) (1000 * b + (int32_t) (rng() % 64));   // an expert id of its own a block
            h.rank[b]   = rng() % 5 == 0 ? -1.0 : (double) (rng() % 12);                          // few values: equal ranks are common
        }
        const int32_t n_active = (int32_t) (rng() % (N + 1));
        const auto busy   = [&](int32_t, int32_t b) { return h.busy[b] != 0; };
        const auto expert = [&](int32_t, int32_t b) { return h.expert[b]; };
        const auto rank   = [&](int32_t, int32_t e) { return h.rank[e / 1000]; };
        poolx_pick_order ord;
        const size_t cap = (size_t) (rng() % 6);
        poolx_pick_order_make(ord, h.ps, n_active, busy, expert, rank, cap);
        for (int k = 0; k < 12; ++k) {
            double b0 = 0, b1 = 0;
            const poolx_pick want = poolx_pick_slot(h.ps, n_active, busy, expert, rank, b0);
            const size_t before = ord.i_ranked;
            const poolx_pick got  = poolx_pick_order_next(ord, h.ps, n_active, busy, expert, rank, b1);
            n_pick++;
            n_rescan += ord.cut && ord.i_ranked == ord.ranked.size() && before <= ord.ranked.size();
            if (want.slot != got.slot || want.owner != got.owner || b0 != b1) {
                if (bad++ == 0) { first = fmt("pick %d of a step: a scan gives block %d (owner %d, rank %g), the order block %d (owner %d, rank %g)", k, want.slot, want.owner, b0, got.slot, got.owner, b1); }
                break;
            }
            if (want.slot < 0) { break; }
            if (rng() % 4 != 0) {   // taken, as the step does: out of the free stack, another layer's now, its upload in flight
                n_take++;
                const int32_t b = want.slot;
                h.ps.free.erase(std::remove(h.ps.free.begin(), h.ps.free.end(), b), h.ps.free.end());
                h.ps.owner[b] = (int32_t) (rng() % NL);
                h.expert[b]   = -1;
                h.busy[b]     = 1;
            }
        }
    }
    report("the order of one scan gives every pick a scan of its own gives", bad == 0, first);
    printf("  (%ld picks, %ld blocks taken, %ld picks past a cut list)\n", n_pick, n_take, n_rescan);
}

static void test_prefetch() {
    printf("Testing prefetch staging slots:\n");
    std::mt19937_64 rng(g_seed + 1);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const size_t slot_b = rng() % 5 == 0 ? 0 : 1 + rng() % (64 << 20), stage_off = rng() % (1 << 30), cap_b = rng() % (3u << 30);
        const int n_cfg = (int) (rng() % 12) - 1, plan_k = rng() % 3 ? -1 : (int) (rng() % 12);
        const int n = ppx_stage_slots(slot_b, cap_b, stage_off, n_cfg, plan_k);
        if (n != 0 && n < 2) { fail(fmt("%d slots: fewer than 2 must be none", n)); }
        if (n > std::max(0, n_cfg)) { fail(fmt("%d slots, only %d configured", n, n_cfg)); }
        if (plan_k >= 0 && n > plan_k) { fail(fmt("%d slots, the planner chose %d", n, plan_k)); }
        if (n > 0 && stage_off + (size_t) n * slot_b > cap_b) { fail(fmt("%d slots of %zu from %zu leave the %zu lendable bytes", n, slot_b, stage_off, cap_b)); }
        if (n > 0 && slot_b == 0) { fail("slots of 0 bytes"); }
        // not too few: one more slot would still have been allowed
        if (n >= 2 && n < n_cfg && (plan_k < 0 || n < plan_k) && stage_off + (size_t) (n + 1) * slot_b <= cap_b) { fail(fmt("%d slots, room and permission for one more", n)); }
    }
    report("2 or none; at most configured / planned; inside the lendable bytes; no room wasted", bad == 0, first);
    report("room for exactly 1 slot gives none, for 2 gives 2", ppx_stage_slots(100, 150, 0, 8, -1) == 0 && ppx_stage_slots(100, 200, 0, 8, -1) == 2);
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
    report("Qwen3.6 split call of 6912 with ub 2048, no cache path: 4 equal passes (1728), not 3x2048 + 768", ppx_equal_pass(6912, 2048, 0) == 1728);
    report("a call that is a multiple of ub keeps full passes", ppx_equal_pass(4096, 2048, 0) == 2048);
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

// ---- stream into the cache's blocks ------------------------------------------------------------------------------------------------------------
static void test_stream_chunks() {
    printf("Testing the stream's chunk count (--moe staging=0):\n");
    std::mt19937_64 rng(g_seed + 4);
    long bad = 0, n_first = 0, n_other = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const int nc = 1 + (int) (rng() % 6);
        uint32_t cap[6]; int extra[6]; double us[6];
        const uint32_t n = 1 + (uint32_t) (rng() % 100000);
        uint32_t c = 64 + (uint32_t) (rng() % 8000);
        for (int i = 0; i < nc; ++i) {   // caps rise with the chunks (a smaller ring), entries too; now and then a candidate cannot run
            cap[i]   = rng() % 7 == 0 ? 0 : c;
            extra[i] = i == 0 ? 0 : extra[i - 1] + (int) (rng() % 2);
            c += (uint32_t) (rng() % 3000);
        }
        const double up = (double) (rng() % 5000000), comp = (double) (rng() % 2000), ent = (double) (rng() % 200);
        const int b = ppx_pick_chunks(n, cap, extra, nc, up, comp, ent, us);
        bool any = false;
        for (int i = 0; i < nc; ++i) { any = any || cap[i] > 0; }
        if (!any) { if (b != -1) { fail("a pick though no candidate can run"); } continue; }
        if (b < 0 || b >= nc || cap[b] == 0) { fail(fmt("pick %d cannot run", b)); continue; }
        (b == 0 ? n_first : n_other)++;
        double t_min = -1.0;
        for (int i = 0; i < nc; ++i) { if (cap[i] > 0 && (t_min < 0.0 || us[i] < t_min)) { t_min = us[i]; } }
        if (us[b] > 1.01 * t_min) { fail(fmt("the pick %d takes %.0f us, the fastest %.0f", b, us[b], t_min)); }
        const uint32_t kb = (n + cap[b] - 1) / cap[b];
        for (int i = 0; i < nc; ++i) {
            if (cap[i] == 0) { if (us[i] >= 0.0) { fail("a time for a candidate that cannot run"); } continue; }
            if (us[i] > 1.01 * t_min) { continue; }
            const uint32_t k = (n + cap[i] - 1) / cap[i];
            if (k < kb) { fail(fmt("candidate %d is as fast with %u passes, the pick %d runs %u", i, k, b, kb)); }
            if (k == kb && i < b) { fail(fmt("candidate %d is as fast with as few passes and fewer chunks than the pick %d", i, b)); }
        }
        uint32_t p = 0;
        const double t = ppx_stream_us(n, cap[b], up, comp + ent * extra[b], &p);
        if (p == 0 || p > cap[b] || t != us[b]) { fail(fmt("pass %u of cap %u", p, cap[b])); }
        if ((uint64_t) p * ((n + cap[b] - 1) / cap[b]) < n) { fail("the passes do not cover the prompt"); }
    }
    report("runnable; within 1% of the fastest; of those the fewest passes, then the fewest chunks; passes cover the prompt", bad == 0, first);
    report("the run met both outcomes (fewest chunks, more chunks)", n_first > 100 && n_other > 100, fmt("%ld first, %ld other", n_first, n_other));
    report("ring reserve: two common entries + two blocks; one of the largest when that is more",
            ppx_ring_reserve(100, 100, 3) == 206 && ppx_ring_reserve(500, 100, 7) == 514 && ppx_ring_reserve(0, 0, 0) == 0);
    report("entries beyond the first: whole layer 1/2/4 chunks -> 0/1/3; half a layer 1/2/3/4/8 -> 0/0/1/1/3",
            ppx_extra_entries(1, 1.0) == 0 && ppx_extra_entries(2, 1.0) == 1 && ppx_extra_entries(4, 1.0) == 3 && ppx_extra_entries(1, 0.5) == 0 &&
            ppx_extra_entries(2, 0.5) == 0 && ppx_extra_entries(3, 0.5) == 1 && ppx_extra_entries(4, 0.5) == 1 && ppx_extra_entries(8, 0.5) == 3);
    {   // Qwen3.8 IQ4_XS, 13.7k prompt, 2 x 8 GiB, layers on one card each (rows 2026-11-12): upload 2.5 s a pass, 470 us a token, 47 us a further entry
        const uint32_t cap[4] = { 4400, 6470, 7100, 7500 }; const int extra[4] = { 0, 1, 2, 3 }; double us[4];
        const int b = ppx_pick_chunks(13720, cap, extra, 4, 2.5e6, 470.0, 47.0, us);
        report("13.7k tokens, upload-bound at 1 chunk (4 passes): 2 chunks (3 passes) picked", b == 1, fmt("pick %d: %.1f %.1f %.1f %.1f s", b, us[0] / 1e6, us[1] / 1e6, us[2] / 1e6, us[3] / 1e6));
        const uint32_t cap2[3] = { 3300, 5300, 6300 }; const int extra2[3] = { 0, 0, 1 };   // the expert split: half layers, 2 chunks cost no entry
        report("expert split: 2 chunks (no further entry, 3 passes for 5) picked", ppx_pick_chunks(13720, cap2, extra2, 3, 1.3e6, 425.0, 85.0) == 1);
        report("a prompt that fits one pass at 1 chunk keeps 1 chunk", ppx_pick_chunks(3000, cap, extra, 4, 2.5e6, 470.0, 47.0) == 0);
        report("compute not measured yet (0): the fewest passes, then the fewest chunks", ppx_pick_chunks(13720, cap, extra, 4, 2.5e6, 0.0, 0.0) == 2);
        {   // a compute read too high makes 5 small passes and 3 big ones look alike: the 3 win (the links never beat their rates, a pass under the upload bound costs more than planned)
            const uint32_t cap3[2] = { 3300, 5300 }; const int extra3[2] = { 0, 0 }; double us3[2];
            const int b3 = ppx_pick_chunks(13720, cap3, extra3, 2, 1.2e6, 503.0, 50.3, us3);
            report("equal times (within 1%): the fewer passes", b3 == 1, fmt("pick %d: %.2f %.2f s", b3, us3[0] / 1e6, us3[1] / 1e6));
        }
    }
    {   // Kimi K3.1 IQ1_S, 13k prompt (12558 tokens), two cards, rows k31-i-mir 2026-11-20. Chunks 1/2/3/4/6/8, a layer on one card (share 1). The drives stream the
        // experts the RAM pool does not hold, 290 GiB a pass, at about 6.5 GB/s together (passes took 46-50 s); the link upload is 20.5 s a pass.
        const double up = 20495e3, ssd_b = 290.0 * 1073741824.0;
        const int extra[6] = { 0, 1, 2, 3, 5, 7 };
        double us[6];
        // request 2, caps measured, compute measured 5814 us a token (581 a further entry)
        const uint32_t cap[6] = { 0, 2272, 3264, 3776, 4160, 4352 };
        const int b0 = ppx_pick_chunks(12558, cap, extra, 6, ppx_stream_pass_us(up, 0.0, 0.0), 5814.0, 581.4, us);
        report("K3.1 priced by the link alone: 3 chunks, 4 passes (what ran)", b0 == 2, fmt("pick %d", b0));
        const int b1 = ppx_pick_chunks(12558, cap, extra, 6, ppx_stream_pass_us(up, ssd_b, 6.5), 5814.0, 581.4, us);
        report("K3.1 with the drives' 290 GiB a pass: 8 chunks, 3 passes", b1 == 5,
                fmt("pick %d: %.0f %.0f %.0f %.0f %.0f s", b1, us[1] / 1e6, us[2] / 1e6, us[3] / 1e6, us[4] / 1e6, us[5] / 1e6));
        // request 1, first call: no pass measured yet, caps estimated, compute calculated 1651 us a token
        const uint32_t cap_e[6] = { 0, 2656, 3616, 4064, 4416, 4544 };
        const int b2 = ppx_pick_chunks(12558, cap_e, extra, 6, ppx_stream_pass_us(up, ssd_b, 6.82), 1651.0, 165.1, us);
        report("K3.1 caps estimated, no error known: 6 chunks (3 passes planned; measured 4160 < 4186 ran 4)", b2 == 4, fmt("pick %d", b2));
        double err[6];
        ppx_cap_err(cap_e, cap, 6, err);
        report("K3.1 estimate error per chunk count from the next pick: 6 chunks 6.2% (4416 / 4160), 8 chunks 4.4%",
                err[0] == 0.0 && err[4] > 0.061 && err[4] < 0.063 && err[5] > 0.043 && err[5] < 0.045, fmt("%.4f %.4f %.4f", err[1], err[4], err[5]));
        const int b3 = ppx_pick_chunks(12558, cap_e, extra, 6, ppx_stream_pass_us(up, ssd_b, 6.82), 1651.0, 165.1, us, err);
        report("K3.1 caps estimated, error known: 8 chunks, 3 passes", b3 == 5, fmt("pick %d", b3));
        const uint32_t est3[3] = { 100, 0, 300 }, now3[3] = { 80, 40, 330 };
        double e3[3];
        ppx_cap_err(est3, now3, 3, e3);
        report("estimate error: a cap that fell 100 -> 80 is 25% high; none for one that rose or could not run", std::abs(e3[0] - 0.25) < 1e-12 && e3[1] == 0.0 && e3[2] == 0.0);
        report("the SSD part: the slower of link and drives; none without a rate",
                ppx_stream_pass_us(20e6, 650e9, 6.5) == 100e6 && ppx_stream_pass_us(20e6, 65e9, 6.5) == 20e6 && ppx_stream_pass_us(20e6, 650e9, 0.0) == 20e6);
    }
}

// ---- the server's checkpoint cut before the last pass ------------------------------------------------------------------------------------------
static void test_pass_cut() {
    printf("Testing the checkpoint cut before the last pass:\n");
    std::mt19937_64 rng(g_seed + 6);
    long bad = 0, n_up = 0, n_down = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        const uint32_t cap = 64 + (uint32_t) (rng() % 16000);
        uint32_t n = 1 + (uint32_t) (rng() % 120000);
        if (it % 2) { n = (2 + (uint32_t) (rng() % 8)) * cap - (uint32_t) (rng() % std::min<uint32_t>(cap, 300)); }   // passes nearly full: little room before the cut or in the last pass
        const uint32_t end = n + (uint32_t) (rng() % 60000);
        auto passes = [&](uint32_t m) { return (m + cap - 1) / cap; };
        const uint32_t k = passes(n), p = (n + k - 1) / k;
        const uint32_t q = ppx_cut_last(n, end, 256, passes);
        if (q == 0 || q > n || q > cap) { fail(fmt("n %u cap %u: last pass %u", n, cap, q)); continue; }
        if (k == 1) { if (q != n) { fail(fmt("n %u cap %u: one pass cut to %u", n, cap, q)); } continue; }
        if (passes(n - q) != k - 1) { fail(fmt("n %u cap %u: %u passes before the last, %u planned", n, cap, passes(n - q), k - 1)); continue; }
        const uint32_t r = (end - p) % 256;
        const bool up_ok = r > 0 && p > 256 - r && passes(n - (p - (256 - r))) == k - 1;
        const bool down_ok = r > 0 && p + r < n && passes(p + r) == 1;
        if ((r == 0 || up_ok || down_ok) && (end - q) % 256 != 0) { fail(fmt("n %u end %u cap %u: cut at %u off its pad though one was in reach", n, end, cap, end - q)); }
        if (up_ok && q > p) { fail(fmt("n %u end %u cap %u: last pass %u grew though the passes before it take the pad", n, end, cap, q)); }
        n_up += q < p; n_down += q > p;
    }
    report("the last pass fits; no pass more than the equal ones; on a pad when one is in reach; shorter before longer", bad == 0, first);
    report("the run met both moves (shorter, longer)", n_up > 100 && n_down > 100, fmt("%ld shorter, %ld longer", n_up, n_down));
    {   // Kimi K3.1 2026-11-20 (rows k31-i): 12558 tokens before the 4-token tail; cap at the request's end estimated 4416 (6 chunks), measured 4160 after the first passes
        auto est = [](uint32_t m) { return (m + 4416 - 1) / 4416; };
        const uint32_t q = ppx_cut_last(12558, 12558, 256, est);
        report("K3.1 12558 tokens: the last pass fits the cap measured later (4160), the cut on a pad", q <= 4160 && (12558 - q) % 256 == 0, fmt("last pass %u, cut at %u", q, 12558 - q));
        report("K3.1: the two passes before the cut stay two", est(12558 - q) == 2, fmt("%u", est(12558 - q)));
    }
    report("one pass: the call is the pass", ppx_cut_last(3000, 3000, 256, [](uint32_t m) { return (m + 4095) / 4096; }) == 3000);
    report("no plan: 0", ppx_cut_last(3000, 3000, 256, [](uint32_t) { return 0u; }) == 0);
    report("passes before full, no room in the last: the equal cut, off a pad", ppx_cut_last(8000, 8000, 256, [](uint32_t m) { return (m + 3999) / 4000; }) == 4000);
    report("passes before full, room in the last: the cut moves down to the pad", ppx_cut_last(8000, 8106, 256, [](uint32_t m) { return (m + 4049) / 4050; }) == 4010);
}

// ---- a graph's bytes from the sizes measured ---------------------------------------------------------------------------------------------------
static void test_lend_line() {
    printf("Testing the graph size line (pass cap, find_slot's limit, the least lend):\n");
    typedef std::vector<std::pair<int32_t, size_t>> pts_t;
    const double MiB = 1048576.0;
    double a = 0.0, b = 0.0;
    report("no size: no line", !ppx_lend_line(pts_t(), a, b));
    report("one size: bytes per token, no fixed part", ppx_lend_line(pts_t{ { 512, 1000 * 512 } }, a, b) && a == 0.0 && b == 1000.0);
    {   // sizes on a line come back as that line
        const bool ok = ppx_lend_line(pts_t{ { 64, 5000 + 64 * 300 }, { 512, 5000 + 512 * 300 }, { 4096, 5000 + 4096 * 300 } }, a, b);
        report("sizes on a line: that line", ok && std::fabs(a - 5000.0) < 1e-6 && std::fabs(b - 300.0) < 1e-9, fmt("a %.3f b %.6f", a, b));
    }
    {   // Qwen3.8 IQ4_XS --moe staging=0, 2026-11-12: 6656 and 6860 tokens ran (0.41 MiB per token each), the old two-point line sized 7064 tokens at 4.2 GiB
        const pts_t p{ { 6656, (size_t) (2734 * MiB) }, { 6860, (size_t) (2862 * MiB) } };
        const bool ok = ppx_lend_line(p, a, b);
        const double y = (a + b * 7064.0) / MiB;
        report("two passes of nearly one size: the next size 204 tokens up is within 3% of the largest measured", ok && y >= 2862.0 && y <= 2862.0 * 1.03, fmt("7064 tokens: %.1f MiB", y));
    }
    {   // GLM 3.0: the larger pass measured less (a slope of noise, negative)
        const bool ok = ppx_lend_line(pts_t{ { 8973, (size_t) (4500 * MiB) }, { 8995, (size_t) (4490 * MiB) } }, a, b);
        const double y = (a + b * 9100.0) / MiB;
        report("the larger size measured less: still a rising line near the sizes", ok && b > 0.0 && y > 4400.0 && y < 4650.0, fmt("9100 tokens: %.1f MiB", y));
    }
    std::mt19937_64 rng(g_seed + 5);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {   // any sizes: a fixed part never negative, a slope never above the least bytes per token, the line through the mean
        pts_t p;
        const int n = 1 + (int) (rng() % 6);
        int32_t x = 64 + (int32_t) (rng() % 512);
        for (int i = 0; i < n; ++i) { p.push_back({ x, (size_t) (1 + rng() % 4000000) * 1024 }); x += 1 + (int32_t) (rng() % 4000); }
        if (!ppx_lend_line(p, a, b)) { fail("no line through sizes"); continue; }
        double rmin = 0.0, mx = 0.0, my = 0.0;
        for (const auto & q : p) { const double r = (double) q.second / q.first; rmin = rmin > 0.0 ? std::min(rmin, r) : r; mx += q.first; my += (double) q.second; }
        mx /= n; my /= n;
        if (a < 0.0 || b <= 0.0) { fail(fmt("a %.1f b %.3f", a, b)); }
        if (b > rmin * (1.0 + 1e-12)) { fail(fmt("slope %.3f above the least bytes per token %.3f", b, rmin)); }
        if (std::fabs(a + b * mx - my) > 1e-6 * my + 1.0) { fail(fmt("the line misses the mean: %.1f against %.1f", a + b * mx, my)); }
    }
    report("any sizes: fixed part >= 0, 0 < slope <= least bytes per token, through the mean", bad == 0, first);
}

// ---- card stamps: which splits are expert time ---------------------------------------------------------------------------------------------------
static void test_split_class() {
    printf("Testing the card stamps' expert/dense class of a split (by its first node):\n");
    // Qwen3.8 IQ4_XS 12k staging=0, 2026-11-14: the stream part's reduce over all tokens (ffn_moe_cs_rel) read as dense on the expert card, 324 us/token against staged's 270
    report("a stream part's reduce (ffn_moe_cs_rel) is expert time", ggml_sched_split_moe(false, true, false, "ffn_moe_cs_rel-12"));
    report("a stream part's start (ffn_moe_cs_slots) is expert time", ggml_sched_split_moe(false, true, false, "ffn_moe_cs_slots-3"));
    report("an expert matmul first in a stream pass is expert time", ggml_sched_split_moe(true, true, false, "ffn_moe_up-3"));
    report("an expert matmul over an expert range is expert time", ggml_sched_split_moe(true, false, true, "ffn_moe_up-3"));
    report("an expert matmul first in a cache pass without a range is dense", !ggml_sched_split_moe(true, false, false, "ffn_moe_up-3"));
    report("an attention node is dense", !ggml_sched_split_moe(false, true, false, "attn_norm-3"));
    report("a stream split is named by its slots, not by its reduce", ggml_sched_cs_split_name("ffn_moe_cs_slots-0") && !ggml_sched_cs_split_name("ffn_moe_cs_rel-0"));
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_hot_cache();
    test_hot_order();
    test_prefetch();
    test_passes();
    test_dense_split();
    test_stream_chunks();
    test_pass_cut();
    test_lend_line();
    test_split_class();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
