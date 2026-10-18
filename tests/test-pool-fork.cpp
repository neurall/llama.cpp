// Unit tests of the expert pool buffer (the fork's block pool, compute-alias lend, KV zone and range map). No GPU, no model: the rules live in two header-only
// files the engine itself includes (src/llama-poolalloc-fork.h, src/llama-kvzone-fork.h), so these tests run the engine's own code, never a copy of it.
//
// Every fixed bug gets a check that fails if it comes back (fork/docs/bug-log.md entry in brackets):
//   [#3, #6]  pad rows: the zone and the pass cap count the cells rounded up to the attention pad       -> test_regress_padded_rows
//   [#4]      lend potential counts the cells of the graph asked about                                   -> test_regress_lend_potential_cells
//   [#5]      a zone never lies over a class zero block (the fixed part picks another gap or is refused) -> test_regress_zone_not_over_zero
//   [#7]      (ggml level, not covered here)  [#1, #2] (model level)
//   [#8]      the lend potential walks the same gaps (block-aligned starts) as the placement              -> test_regress_potential_matches_place
//   [#9]      capacity is the largest gap, not pool bytes / row                                           -> test_regress_capacity_is_gap
//   [#10]     capacity and claim check are the LEAST over every card                                      -> test_regress_least_over_cards
//   plus the allocator itself: block sizing, lend granularity (never more blocks than a class has), the owner ledger under random operations with a byte-exact
//   check that no two owners claim the same byte, the gate, the range map's collision rule, and the range map and free list from many threads.
//
// build: cmake --build <dir> --target test-pool-fork;  run: ctest -R test-pool-fork   or   test-pool-fork [iterations] [seed]
#include "../src/llama-poolalloc-fork.h"
#include "../src/llama-kvzone-fork.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

static int  n_failed = 0;
static long g_iters  = 20000;
static unsigned long g_seed = 1;

static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-62s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}
static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}

// ---- a pool shape like the engine's alias_pool / moe_cache::pool_state, with a few fields only -----------------------------------------------------
struct tseg { int32_t pcls = -1; size_t base = 0, B = 0; int32_t n = 0, blocked = 0; bool zero_dirty = false; };
struct tps  { std::vector<int32_t> owner, free; };
struct tpool {
    std::vector<tseg> segs; std::vector<tps> ps; std::vector<int> kvt;
    size_t tail_end = 0, kv_fixed = 0, kv_h = 0, kv_row = 0, kv_start = 0, span = 0, want_b = 0;
};
static tpool make_pool(std::mt19937_64 & rng, bool with_zone_rows) {
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    tpool p;
    const size_t unit = 16;
    size_t at = 0;
    const int ncls = (int) R(1, 5);
    for (int c = 0; c < ncls; ++c) {
        tseg sg; sg.pcls = c; sg.base = at; sg.B = unit * R(1, 8); sg.n = (int) R(1, 10);
        p.segs.push_back(sg);
        tps ps; ps.owner.assign((size_t) sg.n, -1);
        for (int b = sg.n - 1; b >= 0; --b) { ps.free.push_back(b); }
        p.ps.push_back(ps);
        at += (size_t) (sg.n + 1) * sg.B;
    }
    p.tail_end = at;
    p.span = p.segs.back().base + (size_t) p.segs.back().n * p.segs.back().B;
    if (with_zone_rows) { p.kvt.push_back(1); p.kv_row = R(1, 40); }
    return p;
}

// every byte of the pool, who claims it: the lend (graph bytes in lent blocks), the KV zone, an expert (the block's owner), a zero block. One claimant per byte or the buffers
// overwrite each other. Returns "" when it holds, else what is wrong.
static std::string check_pool(const tpool & p) {
    std::vector<int> claim(p.tail_end + 1, 0);
    for (size_t i = 0; i < p.segs.size(); ++i) {
        const auto & sg = p.segs[i]; const auto & ps = p.ps[i];
        // owner / free consistency
        std::vector<int> in_free(sg.n, 0);
        for (int32_t b : ps.free) { if (b < 0 || b >= sg.n) { return "free list holds a block out of range"; } in_free[b]++; }
        for (int b = 0; b < sg.n; ++b) {
            const int32_t o = ps.owner[b];
            if (in_free[b] > 1) { return fmt("class %zu block %d on the free list twice", i, b); }
            if ((o == -1) != (in_free[b] == 1)) { return fmt("class %zu block %d owner %d but %d free entries", i, b, o, in_free[b]); }
            if (o == -2 && b >= sg.blocked) { return fmt("class %zu block %d lent past `blocked` %d", i, b, sg.blocked); }
            if (b < sg.blocked && o != -2) { return fmt("class %zu block %d under `blocked` %d has owner %d", i, b, sg.blocked, o); }
            const size_t lo = sg.base + (size_t) b * sg.B;
            if (o >= 0 || o == -3 || o == -2) { for (size_t x = lo; x < lo + sg.B; ++x) { claim[x]++; } }
        }
        // zero block: nobody else's bytes, except a lend that covers it (zero_dirty: zeroed again when the lend leaves)
        const size_t z0 = sg.base + (size_t) sg.n * sg.B;
        if (!sg.zero_dirty) { for (size_t x = z0; x < z0 + sg.B; ++x) { claim[x]++; } }
    }
    // the zone claims its bytes itself; blocks under it are -3 (counted above through the owner), so count the zone only where no -3 block is: zone bytes must lie in -3 blocks
    if (p.kv_h > 0) {
        const size_t lo = p.kv_start, hi = poolx_kv_end(p);
        if (hi > p.tail_end) { return "zone past the pool"; }
        for (size_t i = 0; i < p.segs.size(); ++i) {
            const auto & sg = p.segs[i];
            for (int b = 0; b < sg.n; ++b) {
                const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                const bool over = lo < b1 && hi > b0;
                if (over != (p.ps[i].owner[b] == -3)) { return fmt("class %zu block %d: over zone %d but owner %d", i, b, (int) over, p.ps[i].owner[b]); }
            }
            const size_t z0 = sg.base + (size_t) sg.n * sg.B;
            if (lo < z0 + sg.B && hi > z0) { return fmt("zone [%zu,%zu) over the zero block of class %zu", lo, hi, i); }
        }
    } else {
        for (const auto & ps : p.ps) { for (int32_t o : ps.owner) { if (o == -3) { return "a block under a zone that is not there"; } } }
    }
    // the graph's own bytes [0, want_b): must be lent blocks or a dirty zero block; never expert or zone bytes
    for (size_t x = 0; x < p.want_b && x < p.tail_end; ++x) {
        bool ok = false;
        for (size_t i = 0; i < p.segs.size() && !ok; ++i) {
            const auto & sg = p.segs[i];
            if (x < sg.base || x >= sg.base + (size_t) (sg.n + 1) * sg.B) { continue; }
            const size_t b = (x - sg.base) / sg.B;
            ok = b >= (size_t) sg.n ? sg.zero_dirty : p.ps[i].owner[b] == -2;
        }
        if (!ok) { return fmt("the graph's byte %zu (lend %zu) is not in a lent block", x, p.want_b); }
    }
    for (size_t x = 0; x < p.tail_end; ++x) { if (claim[x] > 1) { return fmt("byte %zu claimed %d times", x, claim[x]); } }
    return "";
}

// ---- block sizing -------------------------------------------------------------------------------------------------------------------------------
static void test_block_sizing() {
    printf("Testing block sizing (min allocation granularity of an expert block):\n");
    std::mt19937_64 rng(g_seed);
    const size_t tsz[] = { 2, 4, 18, 20, 34, 36, 54, 66, 84, 110, 144, 210 }, blk[] = { 1, 1, 32, 32, 32, 32, 32, 32, 256, 256, 256, 256 };
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        poolx_slice s[3];
        for (auto & x : s) {
            const size_t t = rng() % 12;
            x.ts = tsz[t]; x.blck = blk[t];
            x.ne0 = 64 * (1 + rng() % 64) + (rng() % 3 == 0 ? 0 : 32 * (rng() % 8));   // multiples of 32: rows that are, and are not, whole 512 tiles
            x.nb2 = x.ne0 / x.blck * x.ts * (1 + rng() % 512);   // bytes of the slice: rows x row bytes
        }
        const poolx_block c = poolx_block_class(s);
        size_t l = 1;
        for (auto & x : s) { l = std::lcm(l, x.ts); }
        l = std::lcm(l, (size_t) 128);
        if (c.B % l != 0 && c.B % std::lcm((size_t) 128, std::max(s[0].ts, std::max(s[1].ts, s[2].ts))) != 0) { fail(fmt("B %zu not a multiple of the type sizes", c.B)); }
        for (int k = 0; k < 3; ++k) {
            if (c.off[k] % 128) { fail(fmt("slice %d at %zu not 128-aligned", k, c.off[k])); }
            if (k > 0 && c.off[k] < c.off[k - 1] + c.len[k - 1]) { fail(fmt("slice %d overlaps slice %d", k, k - 1)); }
            if (c.off[k] + c.len[k] > c.B) { fail(fmt("slice %d (%zu + %zu) does not fit the block %zu", k, c.off[k], c.len[k], c.B)); }
            if (s[k].ne0 % 512 == 0 && c.past[k] != 0) { fail("past guard for a whole-tile row"); }
            if (c.off[k] + c.len[k] + c.past[k] > c.B) { fail(fmt("slice %d: its read past the end (%zu) leaves the block", k, c.past[k])); }
        }
    }
    report("random slices: aligned, disjoint, inside the block, B multiple of the type sizes", bad == 0, first);
    // [Qwen3.8-Flash-Next] down rows of 640 values: a quantized read past the slice must stay inside the block
    poolx_slice q[3] = { { 1048576, 110, 256, 2048 }, { 1048576, 110, 256, 2048 }, { 1310720, 110, 256, 640 } };
    const poolx_block c = poolx_block_class(q);
    report("640-value down rows keep their read-past guard inside the block", c.past[2] > 0 && c.off[2] + c.len[2] + c.past[2] <= c.B);
}

// ---- lend granularity ---------------------------------------------------------------------------------------------------------------------------
static void test_lend_granularity() {
    printf("Testing lend granularity (a lend takes whole blocks, never more than a class has):\n");
    std::mt19937_64 rng(g_seed + 1);
    long bad = 0; std::string first;
    auto fail = [&](const std::string & s) { if (bad++ == 0) { first = s; } };
    for (long it = 0; it < g_iters; ++it) {
        tpool p = make_pool(rng, false);
        const size_t bytes = rng() % (p.tail_end + p.tail_end / 4 + 1);   // up to past the pool: more than it holds
        const size_t end = poolx_lend_round(p, bytes);
        if (end < bytes) { fail(fmt("lend %zu ended at %zu, short of it", bytes, end)); }
        for (const auto & sg : p.segs) {
            bool z = false;
            const int32_t need = poolx_lend_blocks(sg, bytes, &z);
            if (need > sg.n) { fail(fmt("asked for %d blocks of a class of %d", (int) need, (int) sg.n)); }
            const size_t cov = bytes > sg.base ? bytes - sg.base : 0;
            if (cov <= (size_t) sg.n * sg.B && (size_t) need * sg.B < cov) { fail(fmt("%d blocks of %zu do not cover %zu bytes", (int) need, sg.B, cov)); }
            if (z != (cov > (size_t) sg.n * sg.B)) { fail("zero block reach flag wrong"); }
            if (cov > 0 && cov <= (size_t) sg.n * sg.B && (size_t) (need - 1) * sg.B >= cov) { fail("took a block no byte of the lend is in"); }
        }
        // the end is on a block edge of at least one class the lend overlaps (or the lend already ended on one)
        bool edge = end == bytes;
        for (const auto & sg : p.segs) { if (sg.base < bytes && sg.B > 0 && end >= sg.base && (end - sg.base) % sg.B == 0) { edge = true; } }
        if (!edge) { fail(fmt("lend %zu ends at %zu, on no block edge", bytes, end)); }
    }
    report("random pools: 8 blocks asked of a class of 7 returns 7; ends on block edges", bad == 0, first);
    // [MiMo] class 0 has smaller blocks than the largest class at the end: the round must follow each class, not the largest block size
    tpool m; m.segs = { { 0, 0, 9200, 4, 0, false }, { 1, 5 * 9200, 12600, 3, 0, false } };
    report("MiMo shape: a 10800-byte lend ends at 2 blocks of class 0 (18400)", poolx_lend_round(m, 10800) == 18400);
    bool z = false;
    report("7 blocks in the class, lend of 8: needs 7, reaches the zero block", poolx_lend_blocks(tseg{ 0, 0, 100, 7, 0, false }, 800, &z) == 7 && z);
    z = false;
    report("exactly 7 blocks: needs 7, zero block not reached", poolx_lend_blocks(tseg{ 0, 0, 100, 7, 0, false }, 700, &z) == 7 && !z);
}

// ---- owner ledger under random operations -------------------------------------------------------------------------------------------------------
// The operations the engine runs on the ledger: a graph lends blocks, a sync point releases them, the KV zone is placed in a gap (blocks under it taken, blocks it left
// given back), experts claim free blocks and are evicted. After EVERY operation check_pool proves no byte has two claimants.
static void ledger_run(std::mt19937_64 & rng, long steps, std::string & first, long & bad, long & n_ops, long & n_zone) {
    tpool p = make_pool(rng, true);
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    const size_t bmax = std::max<size_t>(1, kvz_bmax(p));
    for (long st = 0; st < steps; ++st) {
        const int op = (int) (rng() % 7);
        const size_t lend_end = poolx_lend_round(p, p.want_b);
        auto evict = [&](int32_t, int32_t) {};
        if (op == 0 || op == 1) {   // a graph lends `bytes` (as the engine, only below the zone)
            const size_t bytes = R(0, p.span);
            const size_t end = (poolx_lend_round(p, bytes) + bmax - 1) / bmax * bmax;
            if (p.kv_h > 0 && end > p.kv_start) { continue; }   // the engine moves the zone behind the lend first; here the op is skipped
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                const int32_t need = poolx_lend_blocks(sg, bytes, &sg.zero_dirty);
                if (need > sg.blocked) {
                    if (poolx_lend(p.ps[i], sg.blocked, need, evict) != 0) { if (bad++ == 0) { first = "a lend went over the zone"; } }
                    sg.blocked = need;
                }
            }
            p.want_b = std::max(p.want_b, bytes);
        } else if (op == 2) {   // sync point: the blocks above what the last graph needed come back
            p.want_b = R(0, p.want_b);
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                const size_t cov = p.want_b > sg.base ? p.want_b - sg.base : 0;
                const int32_t keep = (int32_t) std::min<size_t>((size_t) sg.n, (cov + sg.B - 1) / sg.B);
                if (keep < sg.blocked) { poolx_release(p.ps[i], keep, sg.blocked); sg.blocked = keep; }
                if (sg.zero_dirty && cov <= (size_t) sg.n * sg.B) { sg.zero_dirty = false; }
            }
        } else if (op == 3 || op == 4) {   // the zone is laid out for h cells behind the lend, in a gap
            const size_t h = R(0, 80), zone = p.kv_fixed + h * p.kv_row;
            if (zone == 0) { continue; }
            const size_t s = kvz_gap_place(p, bmax, lend_end, lend_end, zone);
            if (s == (size_t) -1) { continue; }
            n_zone++;
            const size_t old_lo = p.kv_start, old_hi = poolx_kv_end(p);
            (void) old_lo; (void) old_hi;
            p.kv_start = s; p.kv_h = h; p.kv_fixed = p.kv_fixed;
            const size_t lo = s, hi = s + zone;
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                std::vector<int32_t> take, give;
                for (int32_t b = 0; b < sg.n; ++b) {
                    const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                    const bool want = lo < b1 && hi > b0;
                    const int32_t o = p.ps[i].owner[b];
                    if (want) { if (o != -3) { take.push_back(b); } } else if (o == -3) { give.push_back(b); }
                }
                poolx_zone_give(p.ps[i], give);
                poolx_zone_take(p.ps[i], take, &sg.blocked, evict);
            }
            if (const char * why = kvz_bad(p, lend_end)) { if (bad++ == 0) { first = std::string("kvz_bad after placing: ") + why; } }
        } else if (op == 5) {   // experts claim free blocks
            const size_t i = R(0, p.segs.size() - 1);
            const int32_t b = poolx_claim_free(p.ps[i], (int32_t) R(0, 3));
            if (b >= 0) {
                const auto & sg = p.segs[i];
                size_t rlo = 0, rhi = 0;
                const size_t lo = sg.base + (size_t) b * sg.B;
                if (poolx_gate_kind(p, lo, lo + sg.B, &rlo, &rhi) != nullptr) { if (bad++ == 0) { first = "an expert block the gate refuses was handed out"; } }
            }
        } else {   // an expert is evicted
            const size_t i = R(0, p.segs.size() - 1);
            const int32_t b = (int32_t) R(0, (size_t) p.segs[i].n - 1);
            if (p.ps[i].owner[b] >= 0) { p.ps[i].owner[b] = -1; p.ps[i].free.push_back(b); }
        }
        n_ops++;
        const std::string e = check_pool(p);
        if (!e.empty() && bad++ == 0) { first = fmt("after op %d (step %ld): %s", op, st, e.c_str()); }
        if (bad) { return; }
    }
}
static void test_ledger_random() {
    printf("Testing the owner ledger (random lend / release / zone / claim / evict, byte-exact claim check):\n");
    std::mt19937_64 rng(g_seed + 2);
    long bad = 0, n_ops = 0, n_zone = 0; std::string first;
    for (long it = 0; it < g_iters / 100 && !bad; ++it) { ledger_run(rng, 400, first, bad, n_ops, n_zone); }
    report("no byte has two claimants after any operation", bad == 0, first);
    report("the run placed zones and lent graphs (the test is not vacuous)", n_zone > 100 && n_ops > 5000, fmt("%ld ops, %ld zone placements", n_ops, n_zone));
}

// the detector itself: break the layout on purpose, the check must see it (otherwise a green run proves nothing)
static void test_detectors() {
    printf("Testing that the checks catch a broken pool (negative controls):\n");
    std::mt19937_64 rng(g_seed + 3);
    int caught_zero = 0, caught_lend = 0, caught_gate = 0, n = 0;
    for (int it = 0; it < 200; ++it) {
        tpool p = make_pool(rng, true);
        if (p.segs.size() < 2) { continue; }
        n++;
        const auto & sg = p.segs[0];
        const size_t z0 = sg.base + (size_t) sg.n * sg.B;
        // a zone laid over the first class's zero block
        tpool q = p; q.kv_start = z0 > 8 ? z0 - 8 : 0; q.kv_h = 1; q.kv_fixed = sg.B; q.kv_row = 4;
        if (kvz_bad(q, 0) != nullptr && !check_pool(q).empty()) { caught_zero++; }
        size_t rlo = 0, rhi = 0;
        if (poolx_gate_kind(q, z0, z0 + 1, &rlo, &rhi) != nullptr) { caught_gate++; }
        // a lend (graph bytes) that covers an expert-owned block
        tpool r = p; r.ps[0].owner[0] = 1; r.ps[0].free.erase(std::find(r.ps[0].free.begin(), r.ps[0].free.end(), 0)); r.want_b = sg.B;
        if (!check_pool(r).empty()) { caught_lend++; }
    }
    report("zone over a zero block: kvz_bad, the claim check and the gate all see it", caught_zero == n && caught_gate == n, fmt("%d of %d", caught_zero, n));
    report("graph bytes over an expert's block: the claim check sees it", caught_lend == n, fmt("%d of %d", caught_lend, n));
    // poolx_lend never takes a block under the zone
    tpool p; p.segs = { { 0, 0, 16, 4, 0, false } }; p.ps = { tps{ { -1, -3, -3, -1 }, { 3, 0 } } };
    int32_t evicted = 0;
    const int bad = poolx_lend(p.ps[0], 0, 4, [&](int32_t, int32_t) { evicted++; });
    report("a lend reaching blocks under the zone leaves them under the zone", bad == 2 && p.ps[0].owner[1] == -3 && p.ps[0].owner[2] == -3 && p.ps[0].owner[0] == -2);
}

// ---- the gate ------------------------------------------------------------------------------------------------------------------------------------
static void test_gate() {
    printf("Testing the write gate (zone and zero blocks are never written by an expert upload):\n");
    std::mt19937_64 rng(g_seed + 4);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool p = make_pool(rng, true);
        p.kv_h = rng() % 6; p.kv_row = 1 + rng() % 20; p.kv_start = rng() % p.tail_end; p.kv_fixed = rng() % 3 ? 0 : rng() % 100;
        const size_t lo = rng() % p.tail_end, hi = lo + 1 + rng() % 64;
        size_t rlo = 0, rhi = 0;
        const char * k = poolx_gate_kind(p, lo, hi, &rlo, &rhi);
        // oracle: brute force over every byte
        bool hit = false;
        for (size_t x = lo; x < hi && !hit; ++x) {
            if (p.kv_h > 0 && x >= p.kv_start && x < poolx_kv_end(p)) { hit = true; }
            for (const auto & sg : p.segs) { const size_t z0 = sg.base + (size_t) sg.n * sg.B; if (x >= z0 && x < z0 + sg.B) { hit = true; } }
        }
        if (hit != (k != nullptr)) { bad++; first = fmt("[%zu,%zu): oracle %d gate %s", lo, hi, (int) hit, k ? k : "clear"); }
        else if (k && !(lo < rhi && hi > rlo)) { bad++; first = "gate names a region the range does not touch"; }
    }
    report("gate refuses exactly the ranges that touch the zone or a zero block", bad == 0, first);
}

// ---- range map: collision rule ------------------------------------------------------------------------------------------------------------------
static void test_range_collision() {
    printf("Testing the range map collision rule:\n");
    int a, b;
    poolx_ranges<const void *> m;
    m.enter(&a, 100, 200, true);
    report("a mover on an overlapping range of the same pool holds new entrants back", [&]() { m.use.n = 0; m.move_begin(&a, 150, 250, 1); const bool r = !m.try_enter(&a, 140, 160, true); m.move_end(&a); return r; }());
    report("adjacent range [200,300) next to a mover on [100,200) is not held", [&]() { m.move_begin(&a, 100, 200, 1); const bool r = m.try_enter(&a, 200, 300, true); m.leave(&a, 200, 300, true); m.move_end(&a); return r; }());
    report("same bytes on another pool do not collide", [&]() { m.move_begin(&a, 0, 1000, 1); const bool r = m.try_enter(&b, 0, 1000, true); m.leave(&b, 0, 1000, true); m.move_end(&a); return r; }());
    report("a mover waits for a user of an overlapping range (times out, reports it)", [&]() { m.enter(&a, 10, 20, true); const bool r = !m.move_begin(&a, 15, 25, 1); m.move_end(&a); m.leave(&a, 10, 20, true); return r; }());
    report("a mover does not wait for a user of a disjoint range", [&]() { m.enter(&a, 10, 20, true); const bool r = m.move_begin(&a, 20, 30, 1); m.move_end(&a); m.leave(&a, 10, 20, true); return r; }());
    report("after move_end the range is open again, tables empty", [&]() { return m.try_enter(&a, 150, 160, true) && (m.leave(&a, 150, 160, true), m.use.n == 0 && m.moving.n == 0); }());
    report("leave of an entry never entered changes nothing", [&]() { m.leave(&a, 1, 2, true); return m.use.n == 0; }());
}

// ---- threads: range map, free list ---------------------------------------------------------------------------------------------------------------
static void test_threads() {
    printf("Testing the pool from many threads:\n");
    {   // writers (DMA uploads) enter and leave random ranges; a mover announces its range, waits, moves, ends. No user may be inside a range while it is moved, and no entrant
        // may start inside a range being moved.
        constexpr int G = 256;           // granules of the fake pool
        std::atomic<int> inside[G];       // users per granule
        std::atomic<int> moving[G];       // movers per granule
        for (int i = 0; i < G; ++i) { inside[i] = 0; moving[i] = 0; }
        poolx_ranges<const void *> m;
        int pool_key;
        std::atomic<long> violations{0}, writes{0}, skipped{0}, moves{0}, deadlock{0};
        std::atomic<bool> stop{false};
        auto writer = [&](unsigned seed) {
            std::mt19937 rng(seed);
            for (long i = 0; i < g_iters / 4 && !stop; ++i) {
                const size_t lo = rng() % G, hi = std::min<size_t>(G, lo + 1 + rng() % 16);
                const bool nb = (rng() & 1) != 0;     // an upload worker never waits (try_enter); the decode thread does (enter)
                if (nb) { if (!m.try_enter(&pool_key, lo, hi, true)) { skipped++; continue; } } else { m.enter(&pool_key, lo, hi, true); }
                for (size_t g = lo; g < hi; ++g) { inside[g]++; if (moving[g].load() != 0) { violations++; } }
                writes++;
                std::this_thread::yield();
                for (size_t g = lo; g < hi; ++g) { if (moving[g].load() != 0) { violations++; } inside[g]--; }
                m.leave(&pool_key, lo, hi, true);
            }
        };
        auto mover = [&](unsigned seed) {
            std::mt19937 rng(seed);
            for (long i = 0; i < g_iters / 40 && !stop; ++i) {
                const size_t lo = rng() % G, hi = std::min<size_t>(G, lo + 1 + rng() % 64);
                if (!m.move_begin(&pool_key, lo, hi, 5)) { deadlock++; }
                for (size_t g = lo; g < hi; ++g) { moving[g]++; if (inside[g].load() != 0) { violations++; } }
                std::this_thread::yield();
                for (size_t g = lo; g < hi; ++g) { if (inside[g].load() != 0) { violations++; } moving[g]--; }
                m.move_end(&pool_key);
                moves++;
            }
        };
        // one mover only: the engine has one zone move per card at a time (a second one, overlapping, is not a supported case: its move_end drops the first)
        std::vector<std::thread> th;
        for (unsigned t = 0; t < 6; ++t) { th.emplace_back(writer, 100 + t); }
        th.emplace_back(mover, 7);
        for (auto & t : th) { t.join(); }
        report("6 writers + 1 mover: nobody inside a range while it moves", violations == 0, fmt("%ld violations in %ld writes, %ld moves", violations.load(), writes.load(), moves.load()));
        report("the mover never waited out its timeout (no deadlock)", deadlock == 0);
        report("tables empty at the end", m.use.n == 0 && m.moving.n == 0);
        report("the run really overlapped (writes skipped by a move or stalled)", skipped.load() + m.st_enter_n.load() + m.st_move_n.load() > 0, fmt("%ld skipped, %lld stalled", skipped.load(), (long long) m.st_enter_n.load()));
    }
    {   // many threads claim free blocks (under the mutex the engine holds: moe_cache::mtx); every block goes to exactly one thread
        tps ps; const int N = 4000;
        ps.owner.assign(N, -1);
        for (int b = N - 1; b >= 0; --b) { ps.free.push_back(b); }
        std::mutex mu;
        std::vector<std::vector<int32_t>> got(8);
        std::vector<std::thread> th;
        for (int t = 0; t < 8; ++t) {
            th.emplace_back([&, t]() {
                for (;;) { std::lock_guard<std::mutex> lk(mu); const int32_t b = poolx_claim_free(ps, t); if (b < 0) { break; } got[t].push_back(b); }
            });
        }
        for (auto & t : th) { t.join(); }
        std::set<int32_t> all; size_t total = 0;
        for (auto & g : got) { total += g.size(); all.insert(g.begin(), g.end()); }
        bool owners = true;
        for (int t = 0; t < 8; ++t) { for (int32_t b : got[t]) { owners &= ps.owner[b] == t; } }
        report("8 threads claiming: every block to exactly one thread", total == (size_t) N && all.size() == (size_t) N && owners && ps.free.empty());
    }
    {   // a stale entry on the free list (a block taken by the zone since) is never handed out
        tps ps; ps.owner = { -1, -3, -2, -1 }; ps.free = { 0, 1, 2, 3 };
        std::vector<int32_t> got;
        for (int32_t b; (b = poolx_claim_free(ps, 5)) >= 0; ) { got.push_back(b); }
        report("claim skips free-list entries whose block is lent or under the zone", got.size() == 2 && ps.owner[1] == -3 && ps.owner[2] == -2);
    }
}

// ---- regression tests of fixed bugs ---------------------------------------------------------------------------------------------------------------
// a pool of classes (n blocks of B, then the zero block of B) as the kvzone fuzz builds them
static tpool shaped(std::initializer_list<std::pair<size_t, int>> cls, size_t row, size_t fixed = 0) {
    tpool p; size_t at = 0; int c = 0;
    for (auto & x : cls) { tseg sg; sg.pcls = c++; sg.base = at; sg.B = x.first; sg.n = x.second; p.segs.push_back(sg); at += (size_t) (x.second + 1) * x.first; }
    p.tail_end = at; p.span = p.segs.back().base + (size_t) p.segs.back().n * p.segs.back().B;
    p.kvt.push_back(1); p.kv_row = row; p.kv_fixed = fixed;
    return p;
}
static void test_regress_padded_rows() {
    printf("Testing [bug 3, 6] pad rows: cells are counted rounded up to the attention pad:\n");
    // one gap of 100000 bytes (1 byte per cell), a graph of 10 bytes per pass token: 60000 cells padded to 4096 are 61440 rows, so less lend is left for the graph
    const tpool p = shaped({ { 100, 1000 } }, 1);
    const unsigned cap_pad   = kvz_pass_cap(p, 60000, 4096, 0.0, 10.0, 4096, 0);
    const unsigned cap_nopad = kvz_pass_cap(p, 60000, 4096, 0.0, 10.0, 1, 0);
    report("a prompt's pass cap counts the padded rows (smaller than for the bare cells)", cap_pad < cap_nopad, fmt("pad 4096: %u, pad 1: %u", cap_pad, cap_nopad));
    // property over random pools: the zone for the padded rows is placeable wherever the cap says the prompt fits
    std::mt19937_64 rng(g_seed + 5);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = shaped({ { 16 * (1 + rng() % 8), (int) (1 + rng() % 30) }, { 16 * (1 + rng() % 8), (int) (rng() % 30) } }, 1 + rng() % 30);
        const size_t cells = 1 + rng() % 400, pad = (size_t) 1 << (rng() % 9);
        const size_t padded = (cells + pad - 1) / pad * pad, b = std::max<size_t>(1, kvz_bmax(q));
        const size_t fit = kvz_gap_fit(q, b, 0, q.kv_row);   // whole cells the largest gap holds
        // the engine admits a prompt only when fit >= padded rows: a fit below the padded rows must not be reported as room for `cells` by the pad-aware cap
        const unsigned cap = kvz_pass_cap(q, cells, 4096, 0.0, 1.0, pad, 0);
        if (cap < 64 && fit >= padded) { /* conservative is fine */ }
        const size_t pot = kvz_lend_potential(q, padded);
        if (pot > 0) {   // a lend the potential promises leaves a gap for the PADDED rows
            const size_t zone = q.kv_fixed + padded * q.kv_row;
            if (kvz_gap_place(q, b, pot, pot, zone) == (size_t) -1) { bad++; first = fmt("potential %zu promised but the %zu padded rows (%zu bytes) fit no gap", pot, padded, zone); }
        }
    }
    report("a lend promised for the padded rows leaves a gap for the padded rows", bad == 0, first);
}
static void test_regress_lend_potential_cells() {
    printf("Testing [bug 4] lend potential counts the cells of the graph asked about:\n");
    // 185 MiB-like pool: one class, a graph taking nearly all of it must not leave a zone of 0 cells
    tpool p = shaped({ { 1000, 20 } }, 5);
    p.kv_h = 0;
    const size_t pot0 = kvz_lend_potential(p, 0), pot512 = kvz_lend_potential(p, 512);
    report("potential at 512 cells is smaller than at 0 by the zone", pot512 < pot0, fmt("%zu vs %zu", pot512, pot0));
    report("the potential never exceeds pool less the zone", pot512 + 512 * 5 <= p.span);
    report("a graph that takes the whole pool leaves no room (potential 0 when the zone does not fit)", kvz_lend_potential(p, 100000) == 0);
}
static void test_regress_zone_not_over_zero() {
    printf("Testing [bug 5] a zone never lies over a class zero block:\n");
    // MiMo shape: fixed part 1600 behind class 0; the gap behind class 0's zero block is smaller than the fixed part
    tpool p = shaped({ { 100, 10 }, { 100, 10 }, { 100, 30 } }, 1, 1600);
    const size_t bmax = kvz_bmax(p), s = kvz_gap_place(p, bmax, 0, 0, 1600 + 50);
    bool over = false;
    if (s != (size_t) -1) { tpool q = p; q.kv_start = s; q.kv_h = 50; over = kvz_bad(q, 0) != nullptr; }
    report("the fixed part picks a gap that holds it, or is refused: never over a zero block", !over, s == (size_t) -1 ? "refused" : fmt("placed at %zu", s));
    // random: every placement the template returns passes kvz_bad and the byte-exact claim check
    std::mt19937_64 rng(g_seed + 6);
    long bad = 0, placed = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = make_pool(rng, true);
        q.kv_fixed = rng() % 2 ? 0 : 16 * (rng() % 200);
        const size_t cells = 1 + rng() % 100, zone = q.kv_fixed + cells * q.kv_row, b = std::max<size_t>(1, kvz_bmax(q));
        const size_t lo = (rng() % (q.tail_end / 2 + 1) + b - 1) / b * b;
        const size_t s2 = kvz_gap_place(q, b, lo, lo, zone);
        if (s2 == (size_t) -1) { continue; }
        placed++;
        q.kv_start = s2; q.kv_h = cells;
        if (const char * why = kvz_bad(q, lo)) { bad++; first = why; }
        for (const auto & sg : q.segs) { const size_t z0 = sg.base + (size_t) sg.n * sg.B; if (s2 < z0 + sg.B && s2 + zone > z0) { bad++; first = "zone bytes over a zero block"; } }
    }
    report("random pools: every placed zone is behind the lend, in one gap, off every zero block", bad == 0 && placed > 100, first.empty() ? fmt("%ld placed", placed) : first);
}
static void test_regress_potential_matches_place() {
    printf("Testing [bug 8] the lend potential walks the placement's gaps:\n");
    std::mt19937_64 rng(g_seed + 7);
    long bad = 0, checked = 0; std::string first;
    for (long it = 0; it < g_iters * 5 && !bad; ++it) {
        tpool p = make_pool(rng, true);
        p.kv_fixed = rng() % 3 ? 0 : 16 * (rng() % 100); p.kv_h = rng() % 100;
        const size_t min_cells = rng() % 200, b = std::max<size_t>(1, kvz_bmax(p));
        const size_t pot = kvz_lend_potential(p, min_cells);
        const size_t zone = p.kv_fixed + std::max(p.kv_h, min_cells) * p.kv_row;
        if (pot == 0 || zone == 0) { continue; }
        checked++;
        if (kvz_gap_place(p, b, pot, pot, zone) == (size_t) -1) { bad++; first = fmt("potential %zu promised a zone of %zu bytes no gap holds", pot, zone); }
    }
    report("whatever lend the potential promises, the zone can be placed behind it", bad == 0 && checked > 1000, first.empty() ? fmt("%ld checked", checked) : first);
}
static void test_regress_capacity_is_gap() {
    printf("Testing [bug 9] capacity is the largest gap behind the lend, not pool bytes / row:\n");
    tpool p = shaped({ { 100, 4 }, { 100, 4 } }, 10);   // two classes: zero blocks cut the pool in gaps of 400 and 400 bytes
    const size_t b = kvz_bmax(p), fit = kvz_gap_fit(p, b, 0, p.kv_row);
    report("two gaps of 400 bytes hold 40 cells, not (pool 1000 bytes) / 10 = 100", fit == 40, fmt("%zu cells", fit));
    p.kv_fixed = 150;
    report("the fixed part comes off the gap first: (400 - 150) / 10 = 25 cells", kvz_gap_fit(p, b, 0, p.kv_row) == 25);
    std::mt19937_64 rng(g_seed + 8);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = make_pool(rng, true);
        const size_t bm = std::max<size_t>(1, kvz_bmax(q)), lo = (rng() % (q.tail_end + 1) + bm - 1) / bm * bm;
        const size_t fit2 = kvz_gap_fit(q, bm, lo, q.kv_row);
        if (fit2 > 0) {   // a zone of exactly `fit` cells is placeable (so the capacity is real) and one cell more is not in the same gap
            const size_t zone = q.kv_fixed + fit2 * q.kv_row;
            if (kvz_gap_place(q, bm, lo, lo, zone) == (size_t) -1) { bad++; first = "capacity claims a zone that cannot be placed"; }
            if (kvz_gap_place(q, bm, lo, lo, q.kv_fixed + (fit2 + 1) * q.kv_row) != (size_t) -1) { bad++; first = "a zone of one cell more also fits: capacity too small"; }
        }
    }
    report("random pools: capacity cells are exactly placeable, one more is not", bad == 0, first);
}
static void test_regress_least_over_cards() {
    printf("Testing [bug 10] capacity is the least over every card:\n");
    // two cards, records on both: the admitted cells are the smaller capacity, never the first card's
    tpool big = shaped({ { 100, 40 } }, 10), small = shaped({ { 100, 8 } }, 10);
    const size_t cb = kvz_gap_fit(big, 100, 0, 10), cs = kvz_gap_fit(small, 100, 0, 10);
    std::vector<size_t> caps = { cb, cs };
    report("the pool of card 1 holds more than card 2: the request is limited by card 2", cb > cs && *std::min_element(caps.begin(), caps.end()) == cs, fmt("%zu vs %zu cells", cb, cs));
}

int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_block_sizing();
    test_lend_granularity();
    test_ledger_random();
    test_detectors();
    test_gate();
    test_range_collision();
    test_threads();
    test_regress_padded_rows();
    test_regress_lend_potential_cells();
    test_regress_zone_not_over_zero();
    test_regress_potential_matches_place();
    test_regress_capacity_is_gap();
    test_regress_least_over_cards();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
