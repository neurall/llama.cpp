// Unit tests of the expert pool buffer (the fork's block pool, compute-alias lend, KV zone and range map). No GPU, no model: the rules live in two header-only
// files the engine itself includes (src/llama-poolalloc-fork.h, src/llama-kvzone-fork.h), so these tests run the engine's own code, never a copy of it.
//
// Every fixed bug gets a check that fails if it comes back (fork/docs/bug-log.md entry in brackets):
//   [#3, #6]  pad rows: the zone and the pass cap count the cells rounded up to the attention pad       -> test_regress_padded_rows
//   [#4]      lend potential counts the cells of the graph asked about                                   -> test_regress_lend_potential_cells
//   [#5]      a zone lies behind the lend inside the expert blocks, across class boundaries (no zero blocks since 2026-10-24) -> test_regress_zone_across_classes
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
struct tseg { int32_t pcls = -1; size_t base = 0, B = 0; int32_t n = 0, blocked = 0; };
struct tps  { std::vector<int32_t> owner, free; int32_t home_lo = 0, home_hi = INT32_MAX; };   // home: the class's own blocks; owner past them: the grid over the ring reserve (-4)
struct tpool {
    std::vector<tseg> segs; std::vector<tps> ps; std::vector<int> kvt;
    size_t tail_end = 0, floor_end = 0, kv_fixed = 0, kv_h = 0, kv_row = 0, kv_start = 0, span = 0, want_b = 0;
    size_t room_max = 0, room_lo = 0, room_lo_want = 0, tail_real = 0;   // CUDA's room at the tail: [min(room_lo, room_lo_want), tail_real) when room_max > 0
};
// grids (--moe staging=0): a class's block grid runs on from its home to the pool's end; blocks past its home are -4 until the ring claims them
static tpool make_pool(std::mt19937_64 & rng, bool with_zone_rows, bool grids = false) {
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
        ps.home_lo = 0; ps.home_hi = sg.n;
        p.ps.push_back(ps);
        at += (size_t) sg.n * sg.B;   // the classes back to back (no zero block between them)
    }
    p.tail_end = at; p.floor_end = at;
    if (grids) {   // the class's grid runs on from its home to the pool's end (the ring reserve); the lend and the zone walk the home only (segs)
        for (size_t c = 0; c < p.segs.size(); ++c) {
            auto & sg = p.segs[c]; auto & ps = p.ps[c];
            const int32_t fwd = (int32_t) ((at - sg.base - (size_t) sg.n * sg.B) / sg.B);
            ps.owner.resize((size_t) (sg.n + fwd), -4);
        }
    }
    p.span = p.segs.back().base + (size_t) p.segs.back().n * p.segs.back().B;
    if (with_zone_rows) { p.kvt.push_back(1); p.kv_row = R(1, 40); }
    return p;
}

// every byte of the pool, who claims it: the lend (graph bytes in lent blocks), the KV zone, an expert (the block's owner). One claimant per byte or the buffers
// overwrite each other. Returns "" when it holds, else what is wrong.
static std::string check_pool(const tpool & p) {
    std::vector<int> claim(p.tail_end + 1, 0);
    for (size_t i = 0; i < p.segs.size(); ++i) {
        const auto & sg = p.segs[i]; const auto & ps = p.ps[i];
        // owner / free consistency
        std::vector<int> in_free(sg.n, 0);
        for (int32_t b : ps.free) { if (b < 0 || b >= sg.n) { return "free list holds a block out of range"; } in_free[b]++; }
        for (size_t b = (size_t) sg.n; b < ps.owner.size(); ++b) { if (ps.owner[b] != -4) { return fmt("class %zu grid block %zu past its home has owner %d", i, b, ps.owner[b]); } }
        for (int b = 0; b < sg.n; ++b) {
            const int32_t o = ps.owner[b];
            if (in_free[b] > 1) { return fmt("class %zu block %d on the free list twice", i, b); }
            if ((o == -1) != (in_free[b] == 1)) { return fmt("class %zu block %d owner %d but %d free entries", i, b, o, in_free[b]); }
            if (o == -2 && b >= sg.blocked) { return fmt("class %zu block %d lent past `blocked` %d", i, b, sg.blocked); }
            if (b < sg.blocked && o != -2) { return fmt("class %zu block %d under `blocked` %d has owner %d", i, b, sg.blocked, o); }
            const size_t lo = sg.base + (size_t) b * sg.B;
            // a byte is an expert's or free in exactly one class; lent and zone blocks of several classes may cover the same graph or zone bytes (grids)
            if (o >= 0 || o == -1) { for (size_t x = lo; x < lo + sg.B; ++x) { claim[x]++; } }
            if (o == -4) { return fmt("class %zu block %d in its home is -4", i, b); }
        }
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
        }
    } else {
        for (const auto & ps : p.ps) { for (int32_t o : ps.owner) { if (o == -3) { return "a block under a zone that is not there"; } } }
    }
    // the graph's own bytes [0, want_b): must be lent blocks; never expert or zone bytes
    for (size_t x = 0; x < p.want_b && x < p.tail_end; ++x) {
        bool ok = false, bad_cover = false;
        for (size_t i = 0; i < p.segs.size(); ++i) {
            const auto & sg = p.segs[i];
            if (x < sg.base || x >= sg.base + (size_t) sg.n * sg.B) { continue; }
            const int32_t o = p.ps[i].owner[(x - sg.base) / sg.B];
            ok |= o == -2;
            bad_cover |= o >= 0 || o == -1;   // an expert or a free block over graph bytes
        }
        ok &= !bad_cover;
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
            const int32_t need = poolx_lend_blocks(sg, bytes);
            if (need > sg.n) { fail(fmt("asked for %d blocks of a class of %d", (int) need, (int) sg.n)); }
            const size_t cov = bytes > sg.base ? bytes - sg.base : 0;
            if (cov <= (size_t) sg.n * sg.B && (size_t) need * sg.B < cov) { fail(fmt("%d blocks of %zu do not cover %zu bytes", (int) need, sg.B, cov)); }
            if (cov > 0 && cov <= (size_t) sg.n * sg.B && (size_t) (need - 1) * sg.B >= cov) { fail("took a block no byte of the lend is in"); }
        }
        // the end is the block edge of the class the lend ends in (the next edge at or after its last byte), or the lend itself on an edge or past the blocks
        bool edge = end == bytes;
        for (const auto & sg : p.segs) { if (bytes > sg.base && bytes < sg.base + (size_t) sg.n * sg.B && sg.B > 0 && (end - sg.base) % sg.B == 0 && end - bytes < sg.B) { edge = true; } }
        if (!edge) { fail(fmt("lend %zu ends at %zu, not at the next block edge of its class", bytes, end)); }
    }
    report("random pools: 8 blocks asked of a class of 7 returns 7; ends on block edges", bad == 0, first);
    // [MiMo] class 0 has smaller blocks than the largest class at the end: the round must follow each class, not the largest block size
    tpool m; m.segs = { { 0, 0, 9200, 4, 0 }, { 1, 4 * 9200, 12600, 3, 0 } };
    report("MiMo shape: a 10800-byte lend ends at 2 blocks of class 0 (18400)", poolx_lend_round(m, 10800) == 18400);
    report("7 blocks in the class, lend of 8: needs 7 (the rest lies in the next class)", poolx_lend_blocks(tseg{ 0, 0, 100, 7, 0 }, 800) == 7);
    report("exactly 7 blocks: needs 7", poolx_lend_blocks(tseg{ 0, 0, 100, 7, 0 }, 700) == 7);
    report("a lend ending inside the second class rounds to that class's block (not the first's)", poolx_lend_round(m, 4 * 9200 + 100) == 4 * 9200 + 12600);
}

// ---- owner ledger under random operations -------------------------------------------------------------------------------------------------------
// The operations the engine runs on the ledger: a graph lends blocks, a sync point releases them, the KV zone is placed in a gap (blocks under it taken, blocks it left
// given back), experts claim free blocks and are evicted. After EVERY operation check_pool proves no byte has two claimants.
static void ledger_run(std::mt19937_64 & rng, long steps, std::string & first, long & bad, long & n_ops, long & n_zone, bool grids = false) {
    tpool p = make_pool(rng, true, grids);
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    const size_t bmax = std::max<size_t>(1, kvz_bmax(p));
    for (long st = 0; st < steps; ++st) {
        const int op = (int) (rng() % 7);
        const size_t lend_end = poolx_lend_round(p, p.want_b);
        auto evict = [&](int32_t, int32_t) {};
        if (op == 0 || op == 1) {   // a graph lends `bytes` (as the engine, only below the zone)
            const size_t bytes = R(0, p.span);
            const size_t end = poolx_lend_round(p, bytes);
            if (p.kv_h > 0 && end > p.kv_start) { continue; }   // the engine moves the zone behind the lend first; here the op is skipped
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                const int32_t need = poolx_lend_blocks(sg, bytes);
                if (need > sg.blocked) {
                    if (poolx_lend(p.ps[i], sg.blocked, need, evict, [&](int32_t b) { poolx_take_bytes(p.segs, p.ps, (int32_t) i, b, evict); }) != 0) { if (bad++ == 0) { first = "a lend went over the zone"; } }
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
                poolx_zone_take(p.ps[i], take, &sg.blocked, evict, [&](int32_t b) { poolx_take_bytes(p.segs, p.ps, (int32_t) i, b, evict); });
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
            if (p.ps[i].owner[b] >= 0) { poolx_give_back(p.ps[i], b); }
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
    // grids (--moe staging=0): the lend and the zone walk the homes only; the grid blocks past them stay -4
    bad = 0; n_ops = 0; n_zone = 0; first.clear();
    for (long it = 0; it < g_iters / 100 && !bad; ++it) { ledger_run(rng, 400, first, bad, n_ops, n_zone, true); }
    report("grids: no byte free or an expert's in two classes after any operation", bad == 0, first);
    report("grids: the run placed zones and lent graphs", n_zone > 100 && n_ops > 5000, fmt("%ld ops, %ld zone placements", n_ops, n_zone));
}

// ---- grids with claims (--moe staging=0) ---------------------------------------------------------------------------------------------------------
// A class's grid block takes other classes' bytes (the ring's cls_claim, the chain's misses), the lend and the KV zone walk the homes, an expert leaves a block,
// the end of a prompt call returns every class home. Checked after every operation, byte by byte: at most one block (an expert's or a free one, any class,
// home or grid) holds a byte, and none lies over the graph's bytes or the zone's. Negative control: the same runs with the lend and the zone NOT taking claimed
// bytes back (the code before 2026-11-14) must break that, or the run proves nothing.
static std::string check_grid_pool(const tpool & p) {
    std::vector<int> held(p.tail_end, 0);
    const size_t z0 = p.kv_h > 0 ? p.kv_start : 0, z1 = p.kv_h > 0 ? poolx_kv_end(p) : 0;
    for (size_t i = 0; i < p.segs.size(); ++i) {
        const auto & sg = p.segs[i]; const auto & ps = p.ps[i];
        std::vector<int> in_free(ps.owner.size(), 0);
        for (int32_t b : ps.free) { if (b < 0 || b >= (int32_t) ps.owner.size()) { return "free list holds a block out of range"; } in_free[b]++; }
        for (size_t b = 0; b < ps.owner.size(); ++b) {
            const int32_t o = ps.owner[b];
            if (in_free[b] > 1) { return fmt("class %zu block %zu on the free list twice", i, b); }
            if ((o == -1) != (in_free[b] == 1)) { return fmt("class %zu block %zu owner %d but %d free entries", i, b, o, in_free[b]); }
            if ((int32_t) b >= sg.n && (o == -2 || o == -3)) { return fmt("class %zu grid block %zu past its home is lent or under the zone", i, b); }
            if (o < -1) { continue; }
            const size_t lo = sg.base + b * sg.B, hi = std::min(lo + sg.B, p.tail_end);
            for (size_t x = lo; x < hi; ++x) {
                if (++held[x] > 1) { return fmt("byte %zu held by two blocks (class %zu block %zu owner %d is the second)", x, i, b, o); }
                if (x < p.want_b) { return fmt("class %zu block %zu (owner %d) over the graph's byte %zu", i, b, o, x); }
                if (x >= z0 && x < z1) { return fmt("class %zu block %zu (owner %d) over the zone's byte %zu", i, b, o, x); }
            }
        }
    }
    return "";
}
static void grid_run(std::mt19937_64 & rng, long steps, bool take_back, std::string & first, long & bad, long & n_claim, long & n_over) {
    tpool p = make_pool(rng, true, true);
    if (p.segs.size() < 2) { return; }
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    const size_t bmax = std::max<size_t>(1, kvz_bmax(p));
    auto gn = [&](int32_t q) { return (int32_t) p.ps[q].owner.size(); };
    auto evict = [&](int32_t, int32_t) {};
    auto take_for = [&](size_t i) {
        return [&, i](int32_t b) { n_over++; if (take_back) { poolx_take_bytes(p.segs, p.ps, (int32_t) i, b, evict); } };
    };
    int32_t li = 0;
    for (long st = 0; st < steps; ++st) {
        const int op = (int) (rng() % 10);
        const size_t lend_end = poolx_lend_round(p, p.want_b);
        if (op <= 1) {   // a graph lends `bytes` (below the zone, as the engine)
            const size_t bytes = R(0, p.span);
            if (p.kv_h > 0 && poolx_lend_round(p, bytes) > p.kv_start) { continue; }
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                const int32_t need = poolx_lend_blocks(sg, bytes);
                if (need > sg.blocked) { poolx_lend(p.ps[i], sg.blocked, need, evict, take_for(i)); sg.blocked = need; }
            }
            p.want_b = std::max(p.want_b, bytes);
        } else if (op == 2) {   // a sync point: the blocks above what the last graph needed come back
            p.want_b = R(0, p.want_b);
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                const size_t cov = p.want_b > sg.base ? p.want_b - sg.base : 0;
                const int32_t keep = (int32_t) std::min<size_t>((size_t) sg.n, (cov + sg.B - 1) / sg.B);
                if (keep < sg.blocked) { poolx_release(p.ps[i], keep, sg.blocked); sg.blocked = keep; }
            }
        } else if (op == 3) {   // the zone for h cells behind the lend
            const size_t h = R(0, 80), zone = p.kv_fixed + h * p.kv_row;
            if (zone == 0) { continue; }
            const size_t s = kvz_gap_place(p, bmax, lend_end, lend_end, zone);
            if (s == (size_t) -1) { continue; }
            p.kv_start = s; p.kv_h = h;
            for (size_t i = 0; i < p.segs.size(); ++i) {
                auto & sg = p.segs[i];
                std::vector<int32_t> take, give;
                for (int32_t b = 0; b < sg.n; ++b) {
                    const size_t b0 = sg.base + (size_t) b * sg.B, b1 = b0 + sg.B;
                    const bool want = s < b1 && s + zone > b0;
                    const int32_t o = p.ps[i].owner[b];
                    if (want) { if (o != -3) { take.push_back(b); } } else if (o == -3) { give.push_back(b); }
                }
                poolx_zone_give(p.ps[i], give);
                poolx_zone_take(p.ps[i], take, &sg.blocked, evict, take_for(i));
            }
        } else if (op <= 6) {   // a grid block claims other classes' bytes: none of them lent or under the zone (the ring's and the chain's rule)
            const size_t i = R(0, p.segs.size() - 1);
            auto & ps = p.ps[i];
            const int32_t b = (int32_t) R(0, ps.owner.size() - 1);
            if (ps.owner[b] != -4) { continue; }
            if (!poolx_over(p.segs, gn, (int32_t) i, b, [&](int32_t q, int32_t x) { const int32_t o = p.ps[q].owner[x]; return o != -2 && o != -3; })) { continue; }
            poolx_take_bytes(p.segs, p.ps, (int32_t) i, b, evict);
            ps.owner[b] = -1; ps.free.push_back(b);
            n_claim++;
        } else if (op == 7) {   // experts take free blocks (home or claimed)
            poolx_claim_free(p.ps[R(0, p.segs.size() - 1)], li++ % 4);
        } else if (op == 8) {   // an expert leaves its block
            const size_t i = R(0, p.segs.size() - 1);
            const int32_t b = (int32_t) R(0, p.ps[i].owner.size() - 1);
            if (p.ps[i].owner[b] >= 0) { poolx_give_back(p.ps[i], b); }
        } else {   // the end of a prompt call (cls_home): every class back home
            if (rng() % 4) { continue; }
            for (int pass = 0; pass < 2; ++pass) { for (auto & ps : p.ps) { poolx_home_pass(ps, pass, [](int32_t, int32_t) {}); } }
        }
        const std::string e = check_grid_pool(p);
        if (!e.empty() && bad++ == 0) { first = fmt("after op %d (step %ld): %s", op, st, e.c_str()); }
        if (bad) { return; }
    }
}
static void test_grid_claims() {
    printf("Testing grids with claims (staging=0: a class's blocks over other classes' bytes; lend, zone, evict, home):\n");
    std::mt19937_64 rng(g_seed + 9);
    long bad = 0, n_claim = 0, n_over = 0; std::string first;
    for (long it = 0; it < g_iters / 50 && !bad; ++it) { grid_run(rng, 300, true, first, bad, n_claim, n_over); }
    report("no byte held twice, none over the graph or the zone", bad == 0, first);
    report("the runs claimed bytes and lent or zoned over claimed bytes (not vacuous)", n_claim > 1000 && n_over > 100, fmt("%ld claims, %ld lends/zones over claimed bytes", n_claim, n_over));
    long bad_neg = 0, c2 = 0, o2 = 0; std::string first_neg;
    for (long it = 0; it < g_iters / 50 && !bad_neg; ++it) { grid_run(rng, 300, false, first_neg, bad_neg, c2, o2); }
    report("negative control: a lend or zone that keeps the claimant breaks the check", bad_neg > 0, first_neg.empty() ? "no violation found" : first_neg.substr(0, 90));
}

// --moe ring-early (2026-11-21): the last pass of a prompt call sends its computed entries home while the others still hold claimed bytes (poolx_home_early). After
// it no byte is held twice, a home block stays -4 while a claim of an entry still running lies over it, and cls_home at the call's end (poolx_home_pass) leaves the
// same owners and free blocks as without the early step (no block freed twice, none lost). Negative control: freeing every home -4 block at once (poolx_home_pass's
// pass 1 while claims stand) must break the byte check, or the check proves nothing.
static void test_regress_ring_early() {
    printf("Testing [ring-early] computed entries go home before the call's end, the rest keep their claims, cls_home after it changes nothing:\n");
    std::mt19937_64 rng(g_seed + 21);
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    long bad = 0, n_freed = 0, n_kept = 0, n_neg = 0; std::string first, first_neg;
    for (long it = 0; it < g_iters / 20 && !bad; ++it) {
        tpool p = make_pool(rng, false, true);
        if (p.segs.size() < 2) { continue; }
        auto gn = [&](int32_t q) { return (int32_t) p.ps[q].owner.size(); };
        auto evict = [](int32_t, int32_t) {};
        for (int st = 0; st < 60; ++st) {   // the pass so far: claims of grid blocks, entries 0..3 taking free blocks, an expert leaving
            const size_t i = R(0, p.segs.size() - 1);
            auto & ps = p.ps[i];
            const int32_t b = (int32_t) R(0, ps.owner.size() - 1);
            const int op = (int) R(0, 2);
            if (op == 0 && ps.owner[b] == -4 && poolx_over(p.segs, gn, (int32_t) i, b, [](int32_t, int32_t) { return true; })) {
                poolx_take_bytes(p.segs, p.ps, (int32_t) i, b, evict);
                ps.owner[b] = -1; ps.free.push_back(b);
            } else if (op == 1) {
                poolx_claim_free(ps, (int32_t) R(0, 3));
            } else if (op == 2 && ps.owner[b] >= 0 && R(0, 3) == 0) {
                poolx_give_back(ps, b);
            }
        }
        const unsigned done = (unsigned) R(0, 15);   // the entries computed so far (a bit per entry)
        tpool a = p;
        n_freed += (long) poolx_home_early(a.segs, a.ps, [&](int32_t, int32_t, int32_t o) { return o >= 0 && ((done >> o) & 1); }, [](int32_t, int32_t, int32_t) {});
        std::string e = check_grid_pool(a);
        for (size_t i = 0; e.empty() && i < a.segs.size(); ++i) {
            const auto & ps = a.ps[i];
            for (int32_t b = 0; e.empty() && b < (int32_t) ps.owner.size(); ++b) {
                const bool home = b >= ps.home_lo && b < ps.home_hi;
                const int32_t o = ps.owner[b];
                if (!home && (o == -1 || (o >= 0 && ((done >> o) & 1)))) { e = fmt("class %zu block %d outside its home still owner %d", i, b, o); }
                if (home && o == -4) {
                    if (poolx_over(a.segs, [&](int32_t q) { return (int32_t) a.ps[q].owner.size(); }, (int32_t) i, b, [&](int32_t q, int32_t x) { return a.ps[q].owner[x] == -4; })) {
                        e = fmt("class %zu home block %d left -4 with nothing over it", i, b);
                    }
                    n_kept++;
                }
            }
        }
        tpool c = p;
        for (tpool * t : { &a, &c }) { for (int pass = 0; pass < 2; ++pass) { for (auto & ps : t->ps) { poolx_home_pass(ps, pass, evict); } } }
        for (size_t i = 0; e.empty() && i < a.ps.size(); ++i) {
            std::multiset<int32_t> fa(a.ps[i].free.begin(), a.ps[i].free.end()), fc(c.ps[i].free.begin(), c.ps[i].free.end());
            if (a.ps[i].owner != c.ps[i].owner || fa != fc) { e = fmt("class %zu: cls_home after the early step differs from cls_home alone", i); }
        }
        if (e.empty()) { e = check_grid_pool(a); }
        if (!e.empty() && bad++ == 0) { first = e; }
        tpool n = p;   // negative control
        for (auto & ps : n.ps) { poolx_home_pass(ps, 1, evict); }
        const std::string en = check_grid_pool(n);
        if (!en.empty() && n_neg++ == 0) { first_neg = en; }
    }
    report("no byte held twice, running claims kept, cls_home after it unchanged", bad == 0, first);
    report("the runs freed home blocks early and kept some under running claims", n_freed > 100 && n_kept > 10, fmt("%ld freed, %ld kept", n_freed, n_kept));
    report("negative control: home blocks freed under standing claims break the check", n_neg > 0, first_neg.empty() ? "no violation found" : first_neg.substr(0, 90));
}

// the detector itself: break the layout on purpose, the check must see it (otherwise a green run proves nothing)
static void test_detectors() {
    printf("Testing that the checks catch a broken pool (negative controls):\n");
    std::mt19937_64 rng(g_seed + 3);
    int caught_zone = 0, caught_lend = 0, caught_gate = 0, n = 0;
    for (int it = 0; it < 200; ++it) {
        tpool p = make_pool(rng, true);
        if (p.segs.size() < 2) { continue; }
        n++;
        const auto & sg = p.segs[0];
        // a zone that starts inside the lend: kvz_bad and the gate see it
        tpool q = p; q.want_b = sg.B; q.kv_start = 0; q.kv_h = 1; q.kv_fixed = 0; q.kv_row = 4;
        if (kvz_bad(q, sg.B) != nullptr) { caught_zone++; }
        size_t rlo = 0, rhi = 0;
        if (poolx_gate_kind(q, 0, 1, &rlo, &rhi) != nullptr) { caught_gate++; }
        // a lend (graph bytes) that covers an expert-owned block
        tpool r = p; r.ps[0].owner[0] = 1; r.ps[0].free.erase(std::find(r.ps[0].free.begin(), r.ps[0].free.end(), 0)); r.want_b = sg.B;
        if (!check_pool(r).empty()) { caught_lend++; }
    }
    report("zone inside the lend: kvz_bad and the gate see it", caught_zone == n && caught_gate == n, fmt("%d of %d", caught_zone, n));
    report("graph bytes over an expert's block: the claim check sees it", caught_lend == n, fmt("%d of %d", caught_lend, n));
    // poolx_lend never takes a block under the zone
    tpool p; p.segs = { { 0, 0, 16, 4, 0 } }; p.ps = { tps{ { -1, -3, -3, -1 }, { 3, 0 } } };
    int32_t evicted = 0;
    const int bad = poolx_lend(p.ps[0], 0, 4, [&](int32_t, int32_t) { evicted++; }, [](int32_t) {});
    report("a lend reaching blocks under the zone leaves them under the zone", bad == 2 && p.ps[0].owner[1] == -3 && p.ps[0].owner[2] == -3 && p.ps[0].owner[0] == -2);
}

// ---- the gate ------------------------------------------------------------------------------------------------------------------------------------
static void test_gate() {
    printf("Testing the write gate (the zone is never written by an expert upload):\n");
    std::mt19937_64 rng(g_seed + 4);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool p = make_pool(rng, true);
        p.kv_h = rng() % 6; p.kv_row = 1 + rng() % 20; p.kv_start = rng() % p.tail_end; p.kv_fixed = rng() % 3 ? 0 : rng() % 100;
        // CUDA's room (bug 2026-11-16: uploads into its unmapped chunks crashed inside libcuda): sometimes none, sometimes a growth asked for (room_lo_want)
        p.tail_real = p.tail_end; p.room_max = rng() % 2 ? p.tail_end / 4 : 0;
        p.room_lo = p.tail_real - (p.room_max ? rng() % (p.room_max + 1) : 0); p.room_lo_want = rng() % 3 ? p.room_lo : p.tail_real - (p.room_max ? rng() % (p.room_max + 1) : 0);
        const size_t lo = rng() % p.tail_end, hi = lo + 1 + rng() % 64;
        size_t rlo = 0, rhi = 0;
        const char * k = poolx_gate_kind(p, lo, hi, &rlo, &rhi);
        // oracle: brute force over every byte
        bool hit = false;
        for (size_t x = lo; x < hi && !hit; ++x) {
            if (p.kv_h > 0 && x >= p.kv_start && x < poolx_kv_end(p)) { hit = true; }
            if (p.room_max > 0 && x >= std::min(p.room_lo, p.room_lo_want) && x < p.tail_real) { hit = true; }
        }
        if (hit != (k != nullptr)) { bad++; first = fmt("[%zu,%zu): oracle %d gate %s", lo, hi, (int) hit, k ? k : "clear"); }
        else if (k && !(lo < rhi && hi > rlo)) { bad++; first = "gate names a region the range does not touch"; }
    }
    report("gate refuses exactly the ranges that touch the zone or CUDA's room", bad == 0, first);
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
    {   // the zone gives back only blocks that are under it: a lent or owned block is never put on the free list
        tps ps; ps.owner = { -3, -2, 4, -3 };
        const size_t n = poolx_zone_give(ps, { 0, 1, 2, 3 });
        report("zone give frees only blocks under the zone (-3); lent and owned blocks stay", n == 2 && ps.owner[0] == -1 && ps.owner[1] == -2 && ps.owner[2] == 4 && ps.owner[3] == -1 && ps.free.size() == 2);
    }
}

// ---- regression tests of fixed bugs ---------------------------------------------------------------------------------------------------------------
// a pool of classes (n blocks of B each, back to back) as the kvzone fuzz builds them
static tpool shaped(std::initializer_list<std::pair<size_t, int>> cls, size_t row, size_t fixed = 0) {
    tpool p; size_t at = 0; int c = 0;
    for (auto & x : cls) { tseg sg; sg.pcls = c++; sg.base = at; sg.B = x.first; sg.n = x.second; p.segs.push_back(sg); at += (size_t) x.second * x.first; }
    p.tail_end = at; p.floor_end = at; p.span = p.segs.back().base + (size_t) p.segs.back().n * p.segs.back().B;
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
// the experts' floor (2026-11-11): the zone and the lend stay in front of floor_end, so each size class keeps one layer's experts for the prompt chain
static void test_experts_floor() {
    printf("Testing the experts' floor (floor_end) bounds the zone and the lend:\n");
    tpool p = shaped({ { 100, 50 }, { 60, 20 } }, 10);   // class 0: 50 blocks of 100 B, class 1: 20 of 60 B right behind it
    const int E = 8;                                     // experts per layer: class 0 keeps its last 8 blocks
    p.floor_end = p.segs[0].base + (size_t) (p.segs[0].n - E) * p.segs[0].B;
    const size_t fit = kvz_gap_fit(p, kvz_bmax(p), 0, p.kv_row);
    report("gap fit stops at the floor", fit == p.floor_end / p.kv_row, fmt("fit %zu cells, floor at %zu B", fit, p.floor_end));
    const size_t pot = kvz_lend_potential(p, 100);
    report("lend potential leaves the zone in front of the floor", pot + 100 * p.kv_row <= p.floor_end, fmt("potential %zu, zone %zu, floor %zu", pot, (size_t) 100 * p.kv_row, p.floor_end));
    p.kv_start = pot; p.kv_h = (p.floor_end - pot) / p.kv_row + 1;   // one cell past the floor
    const char * bad = kvz_bad(p, pot);
    report("a zone into the floor is an invariant break", bad != nullptr, bad ? bad : "kvz_bad accepted it");
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
static void test_regress_zone_across_classes() {
    printf("Testing [bug 5] a zone lies behind the lend, inside the expert blocks, across class boundaries:\n");
    // MiMo shape (9 classes on the x4 card, 2026-10-24): a fixed part bigger than any one class, behind a lend: it fits across the classes
    tpool p = shaped({ { 100, 10 }, { 100, 10 }, { 100, 30 } }, 1, 1600);
    const size_t bmax = kvz_bmax(p), s = kvz_gap_place(p, bmax, 1000, 1000, 1600 + 50);
    bool bad0 = s == (size_t) -1;
    if (!bad0) { tpool q = p; q.kv_start = s; q.kv_h = 50; bad0 = kvz_bad(q, 1000) != nullptr; }
    report("a 1650-byte zone behind a 1000-byte lend fits across the classes (largest class 3000, the others 1000)", !bad0, s == (size_t) -1 ? "refused" : fmt("placed at %zu", s));
    // random: every placement the template returns passes kvz_bad
    std::mt19937_64 rng(g_seed + 6);
    long bad = 0, placed = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = make_pool(rng, true);
        q.kv_fixed = rng() % 2 ? 0 : 16 * (rng() % 200);
        const size_t cells = 1 + rng() % 100, zone = q.kv_fixed + cells * q.kv_row, b = std::max<size_t>(1, kvz_bmax(q));
        const size_t lo = kvz_block_ceil(q, rng() % (q.tail_end / 2 + 1));
        const size_t s2 = kvz_gap_place(q, b, lo, lo, zone);
        if (s2 == (size_t) -1) { if (lo + zone <= kvz_block_floor(q, q.tail_end)) { bad++; first = fmt("a zone of %zu bytes behind %zu was refused in a pool of %zu", zone, lo, q.tail_end); } continue; }
        placed++;
        q.kv_start = s2; q.kv_h = cells;
        if (const char * why = kvz_bad(q, lo)) { bad++; first = why; }
        if (kvz_block_ceil(q, s2) != s2) { bad++; first = "zone start off a block edge"; }
    }
    report("random pools: every zone that fits behind the lend is placed, on a block edge, and passes kvz_bad", bad == 0 && placed > 100, first.empty() ? fmt("%ld placed", placed) : first);
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
    tpool p = shaped({ { 100, 4 }, { 100, 4 } }, 10);   // two classes back to back: one range of 800 bytes
    const size_t b = kvz_bmax(p), fit = kvz_gap_fit(p, b, 0, p.kv_row);
    report("two classes of 400 bytes back to back hold 80 cells (the zone crosses the class edge)", fit == 80, fmt("%zu cells", fit));
    p.kv_fixed = 150;
    report("the fixed part comes off first: (800 - 150) / 10 = 65 cells", kvz_gap_fit(p, b, 0, p.kv_row) == 65);
    std::mt19937_64 rng(g_seed + 8);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = make_pool(rng, true);
        const size_t bm = std::max<size_t>(1, kvz_bmax(q)), lo = kvz_block_ceil(q, rng() % (q.tail_end + 1));
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

// The bytes a matmul reads past each slice (Qwen3.8's down rows of 640) for a run of n experts e..e+n-1 copied into consecutive blocks: the next expert's first bytes,
// so one strided copy covers the rows whose next expert exists (poolx_past_rows), and only the tensor's last expert takes zeros on its own. Qwen3.8 IQ4_XS 12k, 2026-11-14:
// one copy per expert per tensor, 12461 copies before a split pass's first kernel (650 ms of the scheduler thread, both cards idle 1.29 s against 0.62 staged).
static void test_regress_past_rows() {
    printf("Testing the bytes past a run's slices in one copy:\n");
    report("a run inside the tensor: every row from the next expert", poolx_past_rows(0, 4, 512) == 4 && poolx_past_rows(100, 1, 512) == 1);
    report("a run ending at the last expert: that one row takes zeros", poolx_past_rows(508, 4, 512) == 3 && poolx_past_rows(511, 1, 512) == 0);
    report("an empty run: none", poolx_past_rows(0, 0, 512) == 0);
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t a = src.find("\nstatic void cs_copy(");
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    report("cs_copy writes a run's past bytes with one strided copy", b != std::string::npos && src.substr(a, b - a).find("poolx_past_rows(") != std::string::npos);
}

// staging=0, a layer's uploads coalesced (user 2026-11-14: "sort and coalesce and upload moes needed for compute as one n size class coalesced upload, after which upload
// rest to overlap with compute"). Qwen3.8 IQ4_XS 12k on 2 x 3090 (nsys): the tiered block pick scattered a layer over the class (8323 runs for 22894 experts a pass), the
// scheduler thread issued every layer its blocks allowed before the pass's first kernel and sat 3.4 s of a request in copy calls with the queues full.
static void test_regress_coalesced_ring() {
    printf("Testing the coalesced ring (one run of blocks a layer, copies on the card's upload worker):\n");
    const std::vector<uint8_t> all(10, 1), holes = { 1, 1, 0, 1, 1, 1, 1, 0, 1, 1 };
    report("free class: the window starts at the cursor", poolx_first_window(all, 3, 4) == 3);
    report("a window never wraps the class end", poolx_first_window(all, 8, 4) == 0);
    report("holes: the first run long enough from the cursor", poolx_first_window(holes, 0, 4) == 3 && poolx_first_window(holes, 8, 2) == 8 && poolx_first_window(holes, 8, 3) == 3);
    report("no run long enough: none (the tiered pick takes over)", poolx_first_window(holes, 0, 5) == -1 && poolx_first_window(all, 0, 11) == -1);
    // ring_window_worker, Qwen3.8 12k pass 1: every block held a layer to come (rank 6) or another class's claimable bytes (rank 3), interleaved; 0 of 123 entries found a run
    std::vector<int8_t> start(40);
    for (size_t j = 0; j < start.size(); ++j) { start[j] = j % 2 ? 6 : 3; }
    report("a pass's start (layers to come between other classes' bytes): the window still forms at the cursor", poolx_ring_window(start, 7, 10) == 7);
    std::vector<int8_t> mixed(40, 6);
    for (int j = 20; j < 32; ++j) { mixed[(size_t) j] = 2; }
    report("a run of the best ranks wins over the cursor's layers to come", poolx_ring_window(mixed, 0, 10) == 20);
    mixed[25] = -1;
    report("a block in flight breaks a run", poolx_ring_window(mixed, 0, 10) == 0 && poolx_ring_window(mixed, 26, 6) == 26);
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    auto body = [&](const char * head) { const size_t a = src.find(head); const size_t b = a == std::string::npos ? a : src.find("\n}\n", a); return b == std::string::npos ? std::string() : src.substr(a, b - a); };
    report("cs_plan places a layer in one window", body("\nstatic bool cs_plan(").find("poolx_ring_window(") != std::string::npos);
    const std::string lay = body("\nstatic void cs_layer("), rel = body("\nstatic void cs_release(");
    report("the hooks hand their copies to the cards' workers", lay.find("cs_issue(") != std::string::npos && rel.find("cs_issue(") != std::string::npos
           && lay.find("cs_copy(") == std::string::npos && rel.find("cs_copy(") == std::string::npos && body("\nstatic void cs_issue(").find("g_cs_q[") != std::string::npos);
    report("a layer's kernels wait until its own copies are queued", lay.find("cs_wait_issued(") != std::string::npos && lay.find("cs_wait_issued(") < lay.find("ggml_backend_event_wait(cbe, ls.ring_ev)"));
    report("the upload worker takes ring jobs first", src.find("if (cs_w && cs_worker_has(cs_link)) { cs_worker_run(") != std::string::npos);
    report("pass start, pass end and a lend wait until the workers copied everything", body("\nstatic void cs_begin(").find("cs_drain(") != std::string::npos
           && body("\nstatic void cs_end(").find("cs_drain(") != std::string::npos && body("\nbool llama_moe_cache_alias_fork(").find("cs_drain(") != std::string::npos);
    report("the JIT path's pinned slots are taken under one lock", body("\nstatic void cs_copy(").find("jit_mu") != std::string::npos);
}

// staging=0 ring on the upload worker: how many layers' copies a card's ring stream holds queued. nsys, Qwen3.8 IQ4_XS 12k, 2 x 3090, build ring_diag, the expert
// split's first pass: CUDA1's worker queued 23 layers (14 GB) at once; the driver's queue filled, the worker's copy calls blocked 155 + 48 + 140 ms, and the scheduler
// thread launched nothing meanwhile (CUDA0's worker idled 344 ms with no job, both cards waited).
static void test_regress_ring_depth() {
    printf("Testing the ring's depth on the upload worker (one job on the link, the next queued):\n");
    report("nothing in flight: nothing to wait for", poolx_ring_retire(0, 2) == 0);
    report("one in flight: the next is queued behind it", poolx_ring_retire(1, 2) == 0);
    report("two in flight: the oldest lands before a third is queued", poolx_ring_retire(2, 2) == 1);
    report("more in flight (a depth lowered meanwhile): down to one below the depth", poolx_ring_retire(5, 2) == 4);
    report("depth 0: no bound", poolx_ring_retire(30, 0) == 0);
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    size_t a = src.find("\nstatic void cs_worker_run(");   // the definition, not the forward declaration
    while (a != std::string::npos && src.substr(a + 1, src.find('\n', a + 1) - a - 1).find(") {") == std::string::npos) { a = src.find("\nstatic void cs_worker_run(", a + 1); }
    const size_t b = a == std::string::npos ? a : src.find("\n}\n", a);
    const std::string run = b == std::string::npos ? std::string() : src.substr(a, b - a);
    const size_t w = run.find("ggml_backend_event_synchronize("), c = run.find("cs_copy(");
    report("the worker waits for its oldest job before it queues another", run.find("poolx_ring_retire(") != std::string::npos && w != std::string::npos && c != std::string::npos && w < c);
}

// staging=0 ring: which blocks a stream entry takes first. Qwen3.8 IQ4_XS 12k, 2 x 3090, build rel_moe_class, pass 1: 2825 of 23013 blocks held cached experts of
// layers still to come and 4510 were other classes' bytes that could hold them too; each such expert went up twice (55.9 GB against staged's 47.9)
// A ring pass as cs_ahead and cs_plan run it, on the engine's own rules (poolx_ring_cheap, poolx_ring_tier, poolx_ring_victim, poolx_ring_before), one card: n layers of
// e experts, d of each cached before the first pass (the decode set), blocks for cap experts; every hook plans as far ahead as the blocks allow. old: the rule before
// ring_cheap_ahead (every entry may take a layer to come's expert). up: uploads per pass; churn: experts of layers to come taken (each goes up again at its turn);
// bad: such a take by an entry planned beyond the one needed now and the card's next; starved: the entry needed now found too few blocks (the engine aborts)
struct ring_sim { std::vector<int> up; int churn = 0, bad = 0, starved = 0, busy = 0, stale = 0; };   // stale: a taken block whose expert would stay in its table
static ring_sim ring_sim_run(int n, int e, int d, int cap, int passes, bool old) {
    struct blk { int lay = -1, ex = -1, pass = -1; };
    std::vector<blk> bl((size_t) cap);
    std::vector<std::vector<int>> slot((size_t) n, std::vector<int>((size_t) e, -1));   // expert -> block
    for (int l = 0, b = 0; l < n; ++l) { for (int x = 0; x < d && b < cap; ++x, ++b) { bl[(size_t) b] = { l, x, -1 }; slot[(size_t) l][(size_t) x] = b; } }
    ring_sim r;
    for (int ps = 0; ps < passes; ++ps) {
        int ups = 0;
        size_t nx = 0;
        for (size_t p = 0; p < (size_t) n; ++p) {   // entry p's hook: the entries before it are queued (done = p), p is needed now, p + 1 is the card's next
            const size_t done = p, lead = p + 1 < (size_t) n ? p + 1 : SIZE_MAX;
            for (; nx < (size_t) n; ++nx) {
                const bool cheap = !old && poolx_ring_cheap(p, nx, lead);
                std::vector<int> want;
                for (int x = 0; x < e; ++x) { if (slot[nx][(size_t) x] < 0) { want.push_back(x); } }
                struct cand { int tier; int b; double rk; };
                std::vector<cand> c;
                for (int b = 0; b < cap; ++b) {
                    const blk & k = bl[(size_t) b];
                    if (k.lay < 0) { c.push_back({ 0, b, 0.0 }); continue; }
                    const int t = poolx_ring_tier((size_t) k.lay, done, nx, k.pass == ps, false, cheap);
                    if (t >= 0) { c.push_back({ t, b, poolx_ring_victim((size_t) k.lay, nx, (size_t) n, (double) (e - k.ex), k.pass == ps) }); }   // expert 0 the hottest
                }
                std::stable_sort(c.begin(), c.end(), [](const cand & x, const cand & y) { return poolx_ring_before(x.tier, x.rk, y.tier, y.rk); });
                if (c.size() < want.size()) { r.starved += nx == p; break; }
                for (size_t i = 0; i < want.size(); ++i) {
                    blk & k = bl[(size_t) c[i].b];
                    if (poolx_ring_costly(c[i].tier)) { r.churn++; r.bad += nx != p && nx != lead; }
                    r.busy += c[i].tier == 8;
                    if (k.lay >= 0) {
                        if (poolx_ring_take(c[i].tier) != POOLX_TAKE_EVICT) { r.stale++; continue; }   // the engine would leave its table entry on the new expert's bytes
                        slot[(size_t) k.lay][(size_t) k.ex] = -1;
                    }
                    k = { (int) nx, want[i], ps };
                    slot[nx][(size_t) want[i]] = c[i].b;
                    ups++;
                }
            }
        }
        r.up.push_back(ups);
    }
    return r;
}

static void test_regress_ring_costly() {
    printf("Testing the ring's block order (an expert of a layer still to come goes up twice):\n");
    bool order = true;
    for (int t = 0; t < 9; ++t) {
        for (int u = 0; u < 9; ++u) {
            if (t != 7 && !poolx_ring_costly(t) && poolx_ring_costly(u) && poolx_ring_rank(t) >= poolx_ring_rank(u)) { order = false; }
            if (t != 7 && poolx_ring_rank(t) >= poolx_ring_rank(7)) { order = false; }
        }
    }
    report("every cheap block ranks before every costly one, a prediction giving way last", order);
    report("costly: a layer ahead's cached expert, another class's bytes over one", poolx_ring_costly(4) && poolx_ring_costly(6));
    report("cheap: free, empty, computed in this pass, the layer still computing, another class's bytes without one",
           !poolx_ring_costly(0) && !poolx_ring_costly(1) && !poolx_ring_costly(2) && !poolx_ring_costly(3) && !poolx_ring_costly(5) && !poolx_ring_costly(8));
    // user 2026-11-14: "remember to not evict any moe selected [by] this processed layer": the card's last queued entry may still compute; its experts go after
    // every other computed block, before any layer to come's (that one goes up twice, this one only waits for its compute)
    report("the layer still computing after every other computed block and claim", poolx_ring_rank(8) > poolx_ring_rank(2) && poolx_ring_rank(8) > poolx_ring_rank(3) && poolx_ring_rank(8) > poolx_ring_rank(5));
    report("the layer still computing before any layer to come", poolx_ring_rank(8) < poolx_ring_rank(4) && poolx_ring_rank(8) < poolx_ring_rank(6));
    {   // two-part refill after a prompt (2026-11-15): part 1 = the missed experts among a layer's top k by decode score, part 2 = every other missed one with a score
        const std::vector<double> sc = { 5, 4, 3, 2, 1, 0 };
        const std::vector<bool>   ca = { true, false, false, false, false, false };
        const std::vector<int8_t> pt = poolx_refill_parts(sc, ca, 2);
        report("refill parts: a missed expert in the layer's top k goes first", pt[1] == 1);
        report("refill parts: a cached top expert is not refilled, it still counts toward k", pt[0] == 0 && pt[2] == 2);
        report("refill parts: the rest with a score in the background, none without one", pt[3] == 2 && pt[4] == 2 && pt[5] == 0);
        const std::vector<double> tie = { 1, 1, 1 };
        const std::vector<bool>   none = { false, false, false };
        const std::vector<int8_t> pt2 = poolx_refill_parts(tie, none, 1);
        report("refill parts: equal scores keep the expert order (exactly k first)", pt2[0] == 1 && pt2[1] == 2 && pt2[2] == 2);
    }
    {   // ring keep-hot (Qwen3.8 IQ4_XS 12k 2026-11-15: the stream took decode-hot experts, staging=0 refilled 24-26 GB against staged's 14-15)
        const std::vector<uint64_t> cnt = { 0, 0, 10, 1, 1 };
        const double cut = poolx_hot_cut(cnt);
        report("decode-hot: at or above the layer's mean visits, never an unvisited one", poolx_is_hot(10, cut) && !poolx_is_hot(1, cut) && !poolx_is_hot(0, cut) && !poolx_is_hot(0, 0.0));
        report("hot cached experts after every cheap block, before layers to come", poolx_ring_rank(9) > poolx_ring_rank(8) && poolx_ring_rank(9) > poolx_ring_rank(3) && poolx_ring_rank(9) > poolx_ring_rank(5) && poolx_ring_rank(9) < poolx_ring_rank(4) && poolx_ring_rank(10) < poolx_ring_rank(4));
        report("hot blocks costly: only the entry needed now takes them", poolx_ring_costly(9) && poolx_ring_costly(10));
        report("a hot claim claims, a hot cached expert is evicted", poolx_ring_take(10) == POOLX_TAKE_CLAIM && poolx_ring_take(9) == POOLX_TAKE_EVICT);
    }
    report("tiers: computed 2/3, still computing 8, to come 4, queued or own -1",
           poolx_ring_tier(3, 9, 12, true, false, false) == 2 && poolx_ring_tier(3, 9, 12, false, false, false) == 3 && poolx_ring_tier(8, 9, 12, true, false, false) == 8 &&
           poolx_ring_tier(20, 9, 12, false, false, false) == 4 && poolx_ring_tier(10, 9, 12, true, false, false) == -1 && poolx_ring_tier(12, 9, 12, false, false, false) == -1);
    report("an entry planned ahead of its need takes no layer to come's expert, nor a mix home's", poolx_ring_tier(20, 9, 12, false, false, true) == -1 && poolx_ring_tier(20, 9, 12, false, true, false) == -1);
    report("planned ahead: beyond the entry needed now and the card's next", !poolx_ring_cheap(5, 5, 6) && !poolx_ring_cheap(5, 6, 6) && poolx_ring_cheap(5, 7, 6) && poolx_ring_cheap(SIZE_MAX, 0, SIZE_MAX));
    // the shared parts (user 2026-11-14: "have adapted prefill copy sharing just same parts so both benefit"): next use, and the decode cache's score within a layer
    report("next use: later in this pass, else at its turn in the next one", poolx_next_use(30, 20, 48) == 10 && poolx_next_use(5, 20, 48) == 33 && poolx_next_use(20, 20, 48) == 48);
    bool same = true;   // the decode cache's all-visited prompt rank is the shared next-use rank
    for (size_t o = 0; o < 48; ++o) { if (poolx_victim_rank(true, true, o, 20, 48, 7.0) != poolx_rank_next_use(o, 20, 48, 7.0, 2)) { same = false; } }
    report("decode's prompt rank (all visited) is the shared next-use rank", same);
    // decode's own weighting (default): the coldest first wherever, layers to come furthest first
    const auto vr = [](size_t o, double s) { return poolx_victim_rank(false, true, o, 20, 48, s); };
    report("decode: computed layers' coldest expert first, whichever layer", vr(3, 5.0) < vr(17, 50.0) && vr(17, 5.0) < vr(3, 50.0));
    report("decode: every computed layer's expert before any layer to come's", vr(19, 9e5) < vr(47, 0.0));
    // prefill's weighting (user 2026-11-14: "prioritize moes that are to be used soon first", "the load and priorities are way different in prefill"): next use first
    const auto rv = [](size_t o, double s, bool st) { return poolx_ring_victim(o, 20, 48, s, st); };
    report("prefill: the most recent computed layer first (its next use is the latest)", rv(17, 1.0, false) < rv(3, 800.0, false));
    report("prefill: layers to come furthest first, the next two last", rv(40, 800.0, false) < rv(25, 1.0, false) && rv(25, 800.0, false) < rv(21, 1.0, false));
    report("prefill: one layer's cached experts coldest first", rv(17, 1.0, false) < rv(17, 800.0, false));
    report("prefill: one layer's streamed blocks keep the cursor's order (copies in runs)", rv(17, 1.0, true) == rv(17, 800.0, true) && rv(17, 0.0, true) <= rv(17, 1.0, false));
    report("same tier: the lower rank first; equal ranks keep the cursor's order", poolx_ring_before(3, 1.0, 3, 9.0) && !poolx_ring_before(3, 9.0, 3, 1.0) && !poolx_ring_before(2, 0.0, 2, 0.0));
    report("cheap before pending, whatever the rank", poolx_ring_before(3, 9e12, 4, -1e12) && !poolx_ring_before(4, -1e12, 3, 9e12));
    // the churn (Qwen3.8 IQ4_XS 12k staging=0, 2026-11-14: entries planned far ahead took layers to come's experts, 55.9 GB a pass against staged 47.9)
    const ring_sim nw = ring_sim_run(8, 16, 4, 64, 3, false), od = ring_sim_run(8, 16, 4, 64, 3, true);
    printf("  sim: uploads a pass %d %d %d (old rule %d %d %d), churn %d (old %d), still-computing blocks taken %d\n", nw.up[0], nw.up[1], nw.up[2], od.up[0], od.up[1], od.up[2], nw.churn, od.churn, nw.busy);
    report("sim: no entry planned ahead takes a layer to come's expert", nw.bad == 0);
    report("sim: the entry needed now always finds its blocks", nw.starved == 0);
    report("sim: the old rule churns, the rule does not in the first pass", od.churn > nw.churn && nw.up[0] == 8 * 12);
    report("sim: fewer uploads than the old rule", nw.up[0] + nw.up[1] + nw.up[2] < od.up[0] + od.up[1] + od.up[2]);
    report("sim: every taken block's expert leaves its table", nw.stale == 0 && od.stale == 0);
    // what a take does to the block (vf-qni3 on ring_cover_all: tier 8 went down the claim branch, its expert stayed in the table over another expert's bytes: 1830 slices bad)
    bool kinds = poolx_ring_take(0) == POOLX_TAKE_NONE && poolx_ring_take(1) == POOLX_TAKE_NONE && poolx_ring_take(5) == POOLX_TAKE_CLAIM && poolx_ring_take(6) == POOLX_TAKE_CLAIM && poolx_ring_take(7) == POOLX_TAKE_YIELD;
    for (int t : { 2, 3, 4, 8 }) { kinds = kinds && poolx_ring_take(t) == POOLX_TAKE_EVICT; }
    report("take: a cached expert's block evicts it (2, 3, 4, 8), claims (5, 6), a prediction yields (7)", kinds);
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const auto body = [&](const char * head) { const size_t x = src.find(head), y = x == std::string::npos ? x : src.find("\n}\n", x); return y == std::string::npos ? std::string() : src.substr(x, y - x); };
    const std::string ah = body("\nstatic void cs_ahead("), pl = body("\nstatic bool cs_plan("), pv = body("inline double prompt_victim_rank(");
    report("the engine: cs_ahead plans by poolx_ring_cheap", ah.find("poolx_ring_cheap(") != std::string::npos && ah.find("cs_plan(") != std::string::npos);
    report("the engine: cs_plan ranks by poolx_ring_tier and poolx_ring_victim", pl.find("poolx_ring_tier(") != std::string::npos && pl.find("poolx_ring_victim(") != std::string::npos);
    report("the engine: cs_plan takes by poolx_ring_take, no tier comparison picks the branch", pl.find("poolx_ring_take(") != std::string::npos && pl.find("tier >= 5") == std::string::npos && pl.find("tier >= 2") == std::string::npos);
    // a shadow (one pass's part of another card's layer) has no decode heat (vf-ds on ring_cover_all: segfault in score() from the ring's rank)
    report("the engine: cs_plan scores real layers only", pl.find("victim_score(") != std::string::npos && pl.find("mc->layers.size() ? victim_score(") != std::string::npos);
    report("the engine: decode's prompt rank is poolx_victim_rank", pv.find("poolx_victim_rank(") != std::string::npos);
    // a copy over a computed entry's block waits for the done event recorded over it (cs_cov), so cs_done moves only where that event is recorded: cs_mark_done
    // (vf-ds and vf-qni3 on ring_next_use: the reduce split's hook moved it on its own, "GGML_ASSERT ... cs_cov[wq] >= wq")
    const std::string md = body("\nstatic void cs_mark_done(");
    int n_move = 0;
    for (size_t at = src.find("cs_done["); at != std::string::npos; at = src.find("cs_done[", at + 1)) {
        size_t k = src.find(']', at) + 1;
        while (k < src.size() && src[k] == ' ') { ++k; }
        if (k + 1 < src.size() && src[k] == '=' && src[k + 1] != '=') {
            size_t v = k + 1;
            while (v < src.size() && src[v] == ' ') { ++v; }
            const size_t in = src.find("\nstatic void cs_mark_done(");
            const bool inside = in != std::string::npos && at > in && at < in + md.size();
            n_move += src[v] != '0' && src[v] != '{' && !inside;   // a pass start resets it to 0, the declaration to {}
        }
    }
    report("the engine: cs_done moves only in cs_mark_done, which records the event and its cover", !md.empty() && md.find("cs_cov[") != std::string::npos && n_move == 0);
}

// K3 4-token call after a 16k prompt (2026-11-14): CUDA1's home blocks were lent, the chain claimed grid blocks layer by layer, and a claimed block lies past
// n_active: the old scan (owned blocks below n_active, grid only at -4) never offered it again, the 26th layer found 38 for 50 misses (GGML_ABORT).
static void test_regress_chain_reuses_claimed_grid() {
    printf("Testing the prompt chain's blocks over a pass (a claimed grid block stays the class's):\n");
    struct PS { std::vector<int32_t> owner; };
    const int32_t home_hi = 20, miss = 50;
    const auto pass = [&](bool old_scan) {   // 30 layers of one class, each takes 50 blocks (claim or evict, then pool_take); the fewest any layer found
        PS ps; ps.owner.assign(120, -4);
        for (int32_t b = 0; b < home_hi; ++b) { ps.owner[b] = -2; }   // the home lent to the graph
        ps.owner[119] = -3; ps.owner[118] = -7;                       // KV rows, a stream shadow
        int32_t worst = INT32_MAX; bool bad_kind = false;
        for (int32_t layer = 0; layer < 30; ++layer) {
            std::vector<int32_t> c;
            const auto take = [&](int32_t b, int32_t o) { bad_kind |= o == -2 || o == -3 || o == POOLX_ROOM || o <= POOLX_SHADOW0; c.push_back(b); };
            if (old_scan) { for (int32_t b = 0; b < (int32_t) ps.owner.size(); ++b) { const int32_t o = ps.owner[b]; if ((b < home_hi && o >= -1) || o == -4) { take(b, o); } } }
            else          { poolx_chain_scan(ps, true, take); }
            worst = std::min(worst, (int32_t) c.size());
            for (int32_t i = 0; i < miss && i < (int32_t) c.size(); ++i) { ps.owner[c[i]] = layer; }
        }
        return std::make_pair(worst, bad_kind);
    };
    const auto now = pass(false), old = pass(true);
    report("every layer of a 30-layer pass finds blocks for its 50 misses among 98 grid blocks", now.first >= miss, fmt("fewest %d", now.first));
    report("lent, KV and shadow blocks are never offered", !now.second);
    report("negative control: the old scan runs out (as the K3 abort did)", old.first < miss, fmt("fewest %d", old.first));
    PS q; q.owner.assign(10, -4); int n0 = 0; poolx_chain_scan(q, false, [&](int32_t, int32_t) { ++n0; });
    report("without staging=0 no grid block is offered", n0 == 0);
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * fp = fopen(path.c_str(), "r");
        if (fp) { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, fp)) > 0) { src.append(buf, n); } fclose(fp); }
    }
    const size_t x = src.find("\nvoid llama_moe_cache_pp_upload_fork("), y = x == std::string::npos ? x : src.find("missed experts (the check at the graph's build", x);
    report("the engine: the chain's misses pick from poolx_chain_scan", y != std::string::npos && src.substr(x, y - x).find("poolx_chain_scan(") != std::string::npos);
}

// Kimi K3.1 2026-11-15, the first build with its prompt rankings: the k arm planned entries ahead while cheap blocks lasted; their blocks stay until their kernels
// ran, and layer 11's late fix found none of its class's 831 (340 in flight, 376 uploaded and not computed, 115 lent): abort. A class of 831 ring blocks, 115 lent,
// a layer of 896 experts in 8 entries of 112: every entry ahead holds its whole range (a 995-token call ranks every expert), the entry being computed holds its
// range, its late fix needs up to a range more.
static void test_regress_k_ahead_room() {
    const int32_t blocks = 831, lent = 115, entry = 112;
    auto late_fix_room = [&](int ahead) { return blocks - lent - entry * (ahead + 1); };   // what the late fix of the entry being computed may still take
    int unbounded = 0;   // as before: entries ahead while a whole range of blocks is left
    while (late_fix_room(unbounded) >= entry) { unbounded++; }
    report("k arm, unbounded plans ahead: the late fix finds less than a range (the K3 abort)", late_fix_room(unbounded) < entry);
    const int m = poolx_k_ahead_max(blocks, lent, entry);
    report("k arm, bounded by poolx_k_ahead_max: the late fix keeps a whole range", m >= 1 && late_fix_room(m) >= entry);
    report("k arm: at least the lead entry ahead", poolx_k_ahead_max(300, 100, 112) == 1 && poolx_k_ahead_max(831, 115, 0) == 1);
    // the engine: cs_ahead asks for it under the k arm
    std::string src;
    {
        std::string path = __FILE__;
        path = path.substr(0, path.rfind('/') + 1) + "../src/llama-moecache-fork.cpp";
        FILE * f = fopen(path.c_str(), "rb");
        if (f) { char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) { src.append(b, r); } fclose(f); }
    }
    const size_t x = src.find("static void cs_ahead("), y = x == std::string::npos ? x : src.find("\n}\n", x);
    report("the engine: cs_ahead bounds the k arm by cs_k_ahead_max", y != std::string::npos && src.substr(x, y - x).find("cs_k_ahead_max(") != std::string::npos);
}

// [bug 317] a cold start's zone sits ~10 MiB from the pool start: no whole block in front of it, the retype's staging must come from past the zone
static void test_regress_stage_past_zone() {
    printf("Testing [bug 317] KV retype staging range past the zone:\n");
    tpool p = shaped({ { 100, 50 } }, 10);
    p.kv_start = 10; p.kv_h = 30;                       // zone [10, 310): inside the first blocks, nothing whole in front
    const size_t zend = p.kv_start + kvz_bytes(p, p.kv_h);
    size_t lo = 0, hi = 0;
    const bool ok = kvz_stage_range(p, 0, zend, 1000, lo, hi);
    report("a range is found past the zone", ok && lo >= zend && hi <= p.floor_end && lo % 100 == 0 && hi - lo == 1000, fmt("ok %d [%zu, %zu) zone end %zu", ok, lo, hi, zend));
    std::mt19937_64 rng(g_seed + 317);
    long bad = 0; std::string first;
    for (long it = 0; it < g_iters && !bad; ++it) {
        tpool q = shaped({ { 16 * (1 + rng() % 8), (int) (1 + rng() % 30) }, { 16 * (1 + rng() % 8), (int) (rng() % 30) } }, 1 + rng() % 30);
        q.floor_end = q.tail_end - rng() % (q.tail_end / 4 + 1);
        const size_t from = rng() % (q.tail_end / 3 + 1);
        q.kv_start = from + rng() % (q.tail_end / 3 + 1); q.kv_h = rng() % 20;
        const size_t ze = q.kv_start + kvz_bytes(q, q.kv_h + rng() % 10);
        size_t a = 0, b = 0;
        if (!kvz_stage_range(q, from, ze, 1 + rng() % 2000, a, b)) { continue; }
        bool in_seg = false;
        for (const auto & sg : q.segs) { if (a >= sg.base && b <= sg.base + (size_t) sg.n * sg.B && (a - sg.base) % sg.B == 0 && (b - a) % sg.B == 0) { in_seg = true; } }
        if (a < from || (a < ze && b > q.kv_start) || b > std::max(q.floor_end, q.kv_start) || !in_seg) { bad++; first = fmt("[%zu, %zu) from %zu zone [%zu, %zu) floor %zu", a, b, from, q.kv_start, ze, q.floor_end); }
    }
    report("a staging range is whole blocks of one class, off the zone, behind the lend, before the floor", bad == 0, first);
}
static void test_regress_cuda_room() {
    printf("Testing [bug: second request dies of CUDA OOM, 2026-11-16] CUDA's room at the pool's tail, whole blocks, no gap:\n");
    const size_t M = (size_t) 1 << 20, B = 8 * M;
    // a new shape class may load new kernels: mmq/mmvq/flash attention variants are picked by the token count
    const int64_t n[]  = { 1, 4, 8, 9, 17, 24, 25, 128, 129, 300, 4096, 4097 };
    const int64_t cl[] = { 1, 4, 8, 16, 24, 24, 32, 128, 256, 512, 4096, 8192 };
    bool ok = true;
    std::string first;
    for (size_t i = 0; i < sizeof(n) / sizeof(n[0]); ++i) {
        const int64_t c = poolx_room_class(n[i]);
        if (c != cl[i] && ok) { ok = false; first = fmt("%lld tokens: class %lld, want %lld", (long long) n[i], (long long) c, (long long) cl[i]); }
    }
    report("shape class: tokens up to 8 alone, then steps of 8 to 128, then powers of two", ok, first);
    // glm-i call 2 (build state_split_heat): CUDA's own use grew 94 MiB with 1 MiB free; it gets whole blocks from the tail when it runs out
    size_t r = poolx_room_grow(0, 480 * M, B, 0);
    report("out of memory with no room: one block", r == B, fmt("room %zu MiB", r / M));
    r = poolx_room_grow(16 * M, 480 * M, B, 0);
    report("out of memory again: twice the room", r == 32 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_grow(0, 480 * M, B, 94 * M);
    report("the most it grew between two looks is given at once, whole blocks", r == 96 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_grow(400 * M, 480 * M, B, 0);
    report("never past the tail area", r == 480 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_grow(16 * M, 480 * M, 0, 0);
    report("no block size (a plain buffer): unchanged", r == 16 * M, fmt("room %zu MiB", r / M));
    // after a call: whole blocks of what is free go back to the cache at once (no reserve kept free); VRAM that belongs to others stays
    r = poolx_room_give_back(96 * M, 50 * M, 0, B);
    report("50 MiB free of a 96 MiB room: 48 MiB of blocks go back", r == 48 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_give_back(96 * M, 5 * M, 0, B);
    report("less than a block free: nothing moves", r == 96 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_give_back(96 * M, 300 * M, 0, B);
    report("more free than the room: all of it back", r == 0, fmt("room %zu MiB", r / M));
    r = poolx_room_give_back(96 * M, 50 * M, 20 * M, B);
    report("free VRAM that was left outside the pool is not the room's", r == 72 * M, fmt("room %zu MiB", r / M));
    r = poolx_room_give_back(96 * M, 50 * M, 0, 0);
    report("no block size: unchanged", r == 96 * M, fmt("room %zu MiB", r / M));
}
static void test_regress_cuda_kernel_oom_trace() {
    printf("Testing [bug: second request dies of CUDA OOM, 2026-11-16] kernels loaded on first use, the old margin against the tail room:\n");
    const size_t M = (size_t) 1 << 20, B = 8 * M;
    // glm-i, build state_split_heat (bug log): the stored need was 24 MiB, so 24 + max(64, 24 / 8) = 88 MiB were kept free beside the pool (vram_margin).
    // CUDA's own use since the pool was sized, after each call: the prompt left 1 MiB of the margin free, a 4-token call loaded new kernels (mmvq for 4
    // columns) and died "CUDA error: out of memory"; a decode token later wants a bigger cuBLAS workspace in a shape it ran before.
    struct call_t { int64_t n_tokens; size_t use; };
    const call_t trace[] = { { 12438, 87 * M }, { 1, 87 * M }, { 4, 94 * M }, { 1, 110 * M }, { 12438, 110 * M } };
    const size_t n = sizeof(trace) / sizeof(trace[0]);
    // old: the margin is kept free and a call that needs more dies (nothing hands CUDA memory, nothing retries)
    const size_t margin = (24 + 64) * M;
    int old_fail = -1;
    for (size_t i = 0; i < n && old_fail < 0; ++i) { if (trace[i].use > margin) { old_fail = (int) i; } }
    report("old: the margin runs out at the 4-token call (call 3), as in the log", old_fail == 2, fmt("first failing call %d", old_fail + 1));
    // new: nothing kept free; the tail area (at least the cold figure, 484 MiB: it holds cached experts, so its size costs no VRAM) gives CUDA whole blocks.
    // A shape class not seen yet gets the whole area before it runs, what is left free goes back after the call; an OOM in a seen class grows the room and
    // the call runs again
    const size_t area = poolx_up(std::max<size_t>(margin, 484 * M), B);
    size_t room = 0, reruns = 0, fails = 0;
    std::vector<int64_t> seen;
    for (size_t i = 0; i < n; ++i) {
        const int64_t c = poolx_room_class(trace[i].n_tokens);
        if (std::find(seen.begin(), seen.end(), c) == seen.end()) { room = area; seen.push_back(c); }
        while (trace[i].use > room) {   // out of memory inside the graph: the call is taken back
            const size_t next = poolx_room_grow(room, area, B, 0);
            if (next <= room) { fails++; break; }
            room = next;
            reruns++;
        }
        room = poolx_room_give_back(room, room > trace[i].use ? room - trace[i].use : 0, 0, B);   // the card's free VRAM after the call is the room's unused part
    }
    report("new: no call fails", fails == 0, fmt("%zu failed", fails));
    report("new: one rerun, the seen shape that wanted more; new shapes never run twice", reruns == 1, fmt("%zu reruns", reruns));
    report("new: CUDA ends with the whole blocks it uses and no more (110 MiB in 112)", room == 112 * M, fmt("room %zu MiB", room / M));
}
static void test_regress_pool_start_pins() {
    printf("Testing [bug: the pool fails at start, build cuda_room_tail 2026-11-16] free VRAM read before the probe, pins after it:\n");
    const size_t M = (size_t) 1 << 20, G = (size_t) 1 << 30;
    // glm-i on CUDA0: 19153 MiB free at the sizing; the kernel probe then ran 12.7 s while the loader pinned 13 buffers of 3.7 GiB (8 B of page tables per
    // 4 KiB page on every card), and the pool asked for all 19153 MiB with no margin: "cudaMalloc failed: out of memory", the cache was disabled
    const size_t free_before = 19153 * M, pt_probe = 48 * G / 4096 * 8;
    const size_t free_after_probe = free_before - pt_probe;
    report("old: the pool sized before the probe does not fit after it", free_before > free_after_probe, fmt("%zu > %zu MiB", free_before / M, free_after_probe / M));
    // new: free read after the probe; the loader's pins still to come (101.7 GiB planned, 64 GiB pinned by then) are left free beside the pool
    const size_t pin_total = (size_t) (101.7 * G), pinned = 64 * G;
    const size_t pool = free_after_probe - poolx_pins_vram(pin_total, pinned);
    const size_t pt_rest = (pin_total - pinned) / 4096 * 8;   // what the rest takes at 8 B per 4 KiB page, the most measured
    report("new: the pool and the rest of the pins fit the card", pool + pt_rest <= free_after_probe,
           fmt("pool %zu + pins %zu <= %zu MiB", pool / M, pt_rest / M, free_after_probe / M));
    report("new: nothing left to pin keeps nothing", poolx_pins_vram(pin_total, pin_total) == 0 && poolx_pins_vram(0, 0) == 0, "");
    // the pins' reserve inside the tail area (CUDA's room from the start, not VRAM kept beside the pool): once they took their page tables (1.47 MiB per GiB
    // measured on GLM 3.5) the rest goes back to the cache in whole blocks
    const size_t Bk = (size_t) (7.88 * M), reserve = poolx_up(poolx_pins_vram(pin_total, pinned), Bk);
    const size_t took = (size_t) ((double) (pin_total - pinned) / (double) G * 1.47 * (double) M);
    const size_t kept = poolx_room_give_back(reserve, reserve - took, 0, Bk);
    report("new: the pins' reserve left over goes back to the cache", kept >= took && kept - took < Bk,
           fmt("room %zu MiB for %zu MiB of page tables, %zu MiB back", kept / M, took / M, (reserve - kept) / M));
    // a chunk the driver could not back at creation is CUDA's from the start, as long as it stays inside the tail area
    const size_t B = 50, vsz = 1000, tail = 900, vol = 700;
    report("room start: nothing unmapped", poolx_room_start(vsz, 0, vol, tail, vol, B) == tail, "");
    report("room start: only the slack past the last block", poolx_room_start(vsz, 50, vol, tail, vol, B) == tail, "");
    report("room start: the block edge at or below the first unmapped byte", poolx_room_start(vsz, 170, vol, tail, vol, B) == 800,
           fmt("%zu", poolx_room_start(vsz, 170, vol, tail, vol, B)));
    report("room start: below the tail area the pool is refused", poolx_room_start(vsz, 310, vol, tail, vol, B) == SIZE_MAX, "");
}
static void test_regress_room_shadow_tags() {
    printf("Testing [bug: CUDA's room blocks swept back as free by the stream shadows' reset, glm-i 2026-11-16] owner tags of the room and the shadows:\n");
    // the room took owner -5, which was also the first stream shadow's (cs_owner: -5 - k): cs_shadow_drop reset every owner <= -5 to free at the next pass, so
    // unmapped room blocks went back to the cache, the refill and the prompt chain wrote there (segfault in libcuda, guard: "moe_cache_up.9 into CUDA's room")
    report("the room's tag is no shadow's", POOLX_ROOM > POOLX_SHADOW0 && POOLX_ROOM < -4, fmt("room %d, first shadow %d", POOLX_ROOM, POOLX_SHADOW0));
    // the bytes of a grid block taken back from the classes under it: a free block and an expert become -4, CUDA's room is left as a lend or the KV zone is
    std::vector<tseg> segs(2);
    segs[0].pcls = 0; segs[0].base = 0; segs[0].B = 10; segs[0].n = 10;
    segs[1].pcls = 1; segs[1].base = 0; segs[1].B = 25; segs[1].n = 4;
    std::vector<tps> ps(2);
    ps[0].owner.assign(10, -1); ps[0].free = { 7 };
    ps[0].owner[8] = 2; ps[0].owner[9] = POOLX_ROOM;
    ps[1].owner.assign(4, -4);
    int evicted = 0;
    poolx_take_bytes(segs, ps, 1, 3, [&](int32_t, int32_t) { evicted++; });
    report("take-back: free and expert blocks become another class's bytes", ps[0].owner[7] == -4 && ps[0].owner[8] == -4 && ps[0].free.empty(),
           fmt("owners %d %d, free %zu", ps[0].owner[7], ps[0].owner[8], ps[0].free.size()));
    report("take-back: CUDA's room is left as it is, never evicted", ps[0].owner[9] == POOLX_ROOM && evicted == 1, fmt("owner %d, %d evicted", ps[0].owner[9], evicted));
}
static void test_regress_room_graph_warmup() {
    printf("Testing [bug: decode CUDA graphs out of VRAM after the room's give-back, GLM 3.0 room_tag_fix 2026-11-16] a new shape keeps the room through its second call:\n");
    // CUDA's use by call (MiB): page tables of the pinned RAM, a shape's kernels at its first call, its CUDA graphs at its second (ggml instantiates a graph's
    // CUDA graph at the graph's second call, its warmup). A shape not seen yet gets the whole tail area (room_max) before it runs; after a call the room gives
    // back its free whole blocks. The log: room 346 -> 189 after the first decode token, 1 MiB free, the 42 decode graphs failed at the second.
    const int64_t step = 8, room_max = 344, tables = 180;
    struct shape { int64_t cls, kern, graphs; };
    const std::vector<shape> calls = { { 16384, 8, 0 }, { 4, 0, 0 }, { 1, 2, 40 }, { 1, 2, 40 }, { 1, 2, 40 }, { 1, 2, 40 }, { 1, 2, 40 } };
    for (int pol = 0; pol < 2; ++pol) {   // 0: the old rule (seen after one call, a give-back after every call), 1: poolx_room_after_call
        std::vector<int64_t> seen; std::vector<poolx_room_lent> pend; std::map<int64_t, int> ran;
        int64_t room = 184, oom_at = -1, use = tables;
        for (size_t i = 0; i < calls.size(); ++i) {
            const shape & s = calls[i];
            if (std::find(seen.begin(), seen.end(), s.cls) == seen.end()) {
                room = room_max;
                bool in = false; for (auto & p : pend) { in = in || p.cls == s.cls; }
                if (!in) { pend.push_back({ s.cls, 0 }); }
            }
            const int n = ++ran[s.cls];
            use += (n == 1 ? s.kern : 0) + (n == 2 ? s.graphs : 0);
            if (use > room && oom_at < 0) { oom_at = (int64_t) i; }
            bool give;
            if (pol == 0) { for (auto & p : pend) { seen.push_back(p.cls); } pend.clear(); give = true; }
            else          { (void) poolx_room_after_call(pend, seen, { s.cls }); give = true; }   // a class not seen yet gets the whole area again at its next lend
            if (give) { room = (int64_t) poolx_room_give_back((size_t) room, (size_t) std::max<int64_t>(0, room - use), 0, (size_t) step); }
        }
        if (pol == 0) { report("old rule: the second decode token runs out (the regression)", oom_at == 3, fmt("first out-of-memory at call %lld", (long long) oom_at)); }
        else {
            report("new rule: no call runs out", oom_at < 0, fmt("first out-of-memory at call %lld", (long long) oom_at));
            report("new rule: the room ends at CUDA's use in whole blocks, nothing kept beyond", room >= use && room - use < step,
                   fmt("room %lld MiB for a use of %lld", (long long) room, (long long) use));
        }
    }
}
static void test_regress_pool_buffer_limits() {
    printf("Testing [bug: Vulkan pool cut to 1 GiB buffers, the ring off, PC1 RTX 3090 2026-11-16] the pool's buffers from the backend's real limits:\n");
    // ggml_backend_buft_get_max_size on Vulkan is its 1 GiB suballocation block (what the graph allocator should ask for), not the device's limit: the pool split
    // every class into buffers under 1 GiB and ran without the ring. The 3090 takes 1 TiB in one buffer and its shaders index with 64 bits.
    const size_t G = (size_t) 1 << 30, T = (size_t) 1 << 40, total = 40 * G, longest = 20 * G;
    const poolx_buf_plan old = poolx_buffer_plan(total, longest, G, G);
    report("old limit (get_max_size 1 GiB): per-class buffers, no ring (the regression)", old.split && !old.ring_ok && old.class_cap == G, "");
    const poolx_buf_plan vk = poolx_buffer_plan(total, longest, T, T);
    report("3090 limits (1 TiB, 64-bit indexing): one buffer, ring on, as on CUDA", !vk.split && vk.ring_ok, "");
    const poolx_buf_plan v4 = poolx_buffer_plan(total, longest, T, 4 * G);
    report("no 64-bit indexing, a 20 GiB class: per-class buffers of at most 4 GiB", v4.split && !v4.ring_ok && v4.class_cap == 4 * G, "");
    const poolx_buf_plan v4s = poolx_buffer_plan(10 * G, 3 * G, T, 4 * G);
    report("no 64-bit indexing, classes under 4 GiB: one buffer, no ring (its grids would span the pool)", !v4s.split && !v4s.ring_ok, "");
    const poolx_buf_plan v4t = poolx_buffer_plan(3 * G, 2 * G, T, 4 * G);
    report("no 64-bit indexing, the whole pool under 4 GiB: one buffer and the ring", !v4t.split && v4t.ring_ok, "");
    const poolx_buf_plan cu = poolx_buffer_plan(total, longest, SIZE_MAX, SIZE_MAX);
    report("CUDA (no limit): one buffer, ring on", !cu.split && cu.ring_ok, "");
}

static void test_regress_room_own_calls() {
    printf("Testing [bug: Qwen3.8 on two cards, verify list 2026-11-19: every second request failed] a pending shape counts its own calls only:\n");
    // The same CUDA use model as test_regress_room_graph_warmup (kernels at a shape's first call, its CUDA graphs at its second). Two requests: a 142-token
    // prompt (class 256), its last 4 tokens (class 4), 7 decode tokens. The prompt shapes' second calls come with the next request. When every call counted for
    // every pending class, both prompt shapes were seen after the first two decode tokens, the room went back to CUDA's use during the decode, and the
    // next request's prompt captures found no room: in the engine the kernel-load guard tripped mid-prompt (15 MiB free) and the recurrent model's call failed.
    const int64_t step = 8, room_max = 176, tables = 4;
    struct shape { int64_t cls, kern, graphs; };
    std::vector<shape> calls;
    for (int r = 0; r < 2; ++r) {
        calls.push_back({ 256, 8, 16 });
        calls.push_back({ 4, 4, 16 });
        for (int t = 0; t < 7; ++t) { calls.push_back({ 1, 2, 40 }); }
    }
    for (int pol = 0; pol < 2; ++pol) {   // 0: every call counts for every pending class (the bug), 1: poolx_room_after_call with the classes the call ran
        std::vector<int64_t> seen; std::vector<poolx_room_lent> pend; std::map<int64_t, int> ran;
        int64_t room = 8, oom_at = -1, use = tables;
        for (size_t i = 0; i < calls.size(); ++i) {
            const shape & s = calls[i];
            if (std::find(seen.begin(), seen.end(), s.cls) == seen.end()) {
                room = room_max;
                bool in = false; for (auto & p : pend) { in = in || p.cls == s.cls; }
                if (!in) { pend.push_back({ s.cls, 0 }); }
            }
            const int n = ++ran[s.cls];
            use += (n == 1 ? s.kern : 0) + (n == 2 ? s.graphs : 0);
            if (use > room && oom_at < 0) { oom_at = (int64_t) i; }
            std::vector<int64_t> this_call = { s.cls };
            if (pol == 0) { this_call.clear(); for (auto & p : pend) { this_call.push_back(p.cls); } }
            const bool settled = poolx_room_after_call(pend, seen, this_call);
            const bool give = pol == 0 ? settled : true;   // the old engine gave back only with nothing pending; now a class not seen yet gets the whole area at its lend
            if (give) { room = (int64_t) poolx_room_give_back((size_t) room, (size_t) std::max<int64_t>(0, room - use), 0, (size_t) step); }
        }
        if (pol == 0) { report("every call counted for every pending class: the next request's prompt runs out (the regression)", oom_at == 9, fmt("first out-of-memory at call %lld", (long long) oom_at)); }
        else {
            report("a class counts its own calls: no call runs out", oom_at < 0, fmt("first out-of-memory at call %lld", (long long) oom_at));
            report("a class counts its own calls: the room ends at CUDA's use once every shape has its CUDA graphs", room >= use && room - use < step && pend.empty(),
                   fmt("room %lld MiB for a use of %lld", (long long) room, (long long) use));
        }
    }
}
static void test_regress_ring_cold_order() {
    printf("Testing [Qwen3.8 IQ3_S one card 4k, 2026-11-20: the stream took 248 of the layers' top-10 decode experts] the cold order takes the coldest displaced expert first:\n");
    // candidates as cs_plan builds them: tier 3 = a cached expert of a computed layer (rank: its decode score with --moe ring-cold, else next use), tier 5 = a claim
    // over another class's bytes (rank: the hottest expert under them with ring-cold, else 1, so the cursor's order decides); 0 free, 2 the stream's own, 8 the layer computing
    struct cand { int tier; double rk; const char * what; };
    const std::vector<cand> c = { { 5, 900.0, "claim over a hot expert" }, { 3, 10.0, "cold cached expert" }, { 8, 0.0, "the layer still computing" },
                                  { 5, 5.0, "claim over a cold expert" }, { 3, 800.0, "hot cached expert" }, { 2, 0.0, "own computed write" }, { 0, 0.0, "free" } };
    auto order = [&](bool cold) {
        std::vector<cand> v = c;
        std::stable_sort(v.begin(), v.end(), [cold](const cand & x, const cand & y) { return poolx_ring_before(x.tier, x.rk, y.tier, y.rk, cold); });
        std::string o;
        for (const auto & x : v) { o += std::string(x.what) + "; "; }
        return v;
    };
    const auto old_o = order(false), new_o = order(true);
    // the old order: every claim before every cached expert, so the claim over the hot expert goes before the cold cached one (the regression)
    size_t i_hot_claim = 0, i_cold_cached = 0;
    for (size_t i = 0; i < old_o.size(); ++i) { if (old_o[i].tier == 5 && old_o[i].rk == 900.0) { i_hot_claim = i; } if (old_o[i].tier == 3 && old_o[i].rk == 10.0) { i_cold_cached = i; } }
    report("old order: a claim over a hot expert goes before a cold cached expert (the regression)", i_hot_claim < i_cold_cached, "");
    report("cold order: free, the stream's own write, then by decode score: cold claim, cold cached, hot cached, hot claim, then the layer computing",
           new_o[0].tier == 0 && new_o[1].tier == 2 && new_o[2].rk == 5.0 && new_o[3].rk == 10.0 && new_o[4].rk == 800.0 && new_o[5].rk == 900.0 && new_o[6].tier == 8, "");
    // a run (one copy per tensor) over blocks of rank 0-2 only in the cold order: a run over cached experts would take them by position again
    const std::vector<int8_t> rk = { 4, 4, 4, 4, 3, 3, 3, 3, 0, 2, -1, 1, 2, 2 };
    report("old run: four claims over other classes' experts in a row from the cursor (rank 3, whatever their heat)", poolx_ring_window(rk, 0, 4, false) == 4, "");
    report("cold run: no run over cached experts, one over free, empty and own blocks", poolx_ring_window(rk, 0, 4, true) == -1 && poolx_ring_window(rk, 0, 3, true) == 11, "");
    // what a block loses: only an expert the cache held before the pass; free, empty and the stream's own writes (tier 2, tier 8 streamed this pass) nothing
    report("lost value: free, empty, own computed write, own write of the layer computing: 0; cached before the pass: its score",
           poolx_ring_lost(0, false, 7.0) == 0.0 && poolx_ring_lost(1, false, 7.0) == 0.0 && poolx_ring_lost(2, true, 7.0) == 0.0 && poolx_ring_lost(8, true, 7.0) == 0.0 &&
           poolx_ring_lost(8, false, 7.0) == 7.0 && poolx_ring_lost(3, false, 7.0) == 7.0 && poolx_ring_lost(4, false, 7.0) == 7.0, "");
    // the look-ahead: an entry beyond the one needed now and the card's next (cheap) took computed layers' cached experts; level 2 leaves them to the two
    report("levels 0 and 1: an entry planned ahead takes a cached expert with decode value (the regression)", poolx_ring_cold_ok(0, true, 800.0) && poolx_ring_cold_ok(1, true, 800.0), "");
    report("level 2: an entry planned ahead takes only blocks that lose nothing; the entries needed now and next take the coldest",
           !poolx_ring_cold_ok(2, true, 800.0) && !poolx_ring_cold_ok(2, true, 1e-9) && poolx_ring_cold_ok(2, true, 0.0) && poolx_ring_cold_ok(2, false, 800.0), "");
    // --moe cs-tail: the upload area's blocks first, never the last queued entry's (its compute may still read them) nor a layer's to come
    report("tail first: free, empty, own computed, cached computed, a claim; not the layer computing, a layer to come, a claim over one",
           poolx_tail_first(0) && poolx_tail_first(1) && poolx_tail_first(2) && poolx_tail_first(3) && poolx_tail_first(5) &&
           !poolx_tail_first(8) && !poolx_tail_first(4) && !poolx_tail_first(6) && !poolx_tail_first(7), "");
    report("tail order: a tail block before a free block elsewhere, the rest as before",
           poolx_ring_before_tail(true, 3, 800.0, false, 0, 0.0, true) && !poolx_ring_before_tail(false, 0, 0.0, true, 3, 800.0, true) &&
           poolx_ring_before_tail(false, 0, 0.0, false, 3, 5.0, true) == poolx_ring_before(0, 0.0, 3, 5.0, true), "");
    // the moves: hottest into coldest while hotter; free/empty (-1) take anything ("don't drop anything" while there are gaps)
    {
        const std::vector<double> mv = { 5.0, 900.0, 0.0, 40.0 }, ds = { 30.0, -1.0, 2000.0, 1.0 };
        const auto pr = poolx_tail_moves(mv, ds);
        report("moves: 900 into the free block, 40 into the 1.0, 5 into the 30.0 refused (colder), 0 stays", pr.size() == 2 &&
               pr[0].first == 1 && pr[0].second == 1 && pr[1].first == 3 && pr[1].second == 3, "");
        const auto pf = poolx_tail_moves(std::vector<double>{ 0.0, 0.0, 3.0 }, std::vector<double>{ -1.0, -1.0, -1.0, -1.0 });
        report("moves: with gaps every mover goes, even a never-used one", pf.size() == 3, "");
        report("moves: no destination, nothing moves", poolx_tail_moves(mv, std::vector<double>{}).empty(), "");
    }
}
static void test_regress_room_give_back_chunks() {
    printf("Testing [bug: a block given back with a chunk the driver could not back, Qwen3.8 IQ4_XS room_graphs_clear 2026-11-16] the room's give-back in whole chunks:\n");
    // blocks of 2.24 MiB, chunks of 2 MiB; the room's first unmapped byte 1000 bytes above its first block: giving that block back needs two chunks. The log:
    // 3 MiB free, one block went back by its byte count, the second chunk failed to map, the shrink kept max(wanted edge, first unmapped byte) and the
    // block entered the cache half unmapped: "a copy of 921600 bytes (moe_cache_down.24) into CUDA's room"
    const size_t M = 1048576, g = 2 * M, B = 2348810;
    const size_t room_lo = 112 * g - 1000, first = 112 * g, tail = room_lo + 52 * B, base = room_lo;   // the room's blocks: room_lo + k * B
    const size_t old_room = poolx_room_give_back(tail - room_lo, 3 * M, 0, B);
    report("old rule: 3 MiB free gives a block back by its bytes (the regression)", old_room == tail - room_lo - B, fmt("room %zu -> %zu blocks", (tail - room_lo) / B, old_room / B));
    size_t e = poolx_room_give_back_chunks(room_lo, first, tail, 3 * M, 0, B, g);
    report("new rule: 3 MiB free (one chunk) keeps a block that needs two", e == room_lo, fmt("%zu blocks back", (e - room_lo) / B));
    e = poolx_room_give_back_chunks(room_lo, first, tail, 4 * M, 0, B, g);
    report("new rule: 4 MiB free (two chunks) gives it back", e == room_lo + B, fmt("%zu blocks back", (e - room_lo) / B));
    // the shrink after a map that failed: the driver backed one chunk, the second stays unmapped below the wanted edge
    const size_t want = room_lo + B, first_after = first + g;   // first unmapped byte after one chunk mapped: inside the block wanted back
    const size_t old_lo = poolx_room_edge_down(std::max(want, first_after), room_lo, tail, base, B), new_lo = poolx_room_edge_down(first_after, room_lo, tail, base, B);
    report("old shrink edge: above an unmapped byte (the regression)", old_lo > first_after, fmt("edge %zu, first unmapped %zu", old_lo, first_after));
    report("new shrink edge: every unmapped byte stays in the room", new_lo <= first_after && new_lo == room_lo, fmt("edge %zu, first unmapped %zu", new_lo, first_after));
    // random rooms: the edge is a block edge, its chunks fit the free whole chunks, one block more would not
    std::mt19937 rng(7);
    int bad = 0;
    for (int it = 0; it < 20000; ++it) {
        const size_t Bk = (size_t) (rng() % (8 * M)) + 4096, gk = (size_t) 1 << (20 + rng() % 3), nb = 1 + rng() % 200;
        const size_t lo0 = (size_t) (rng() % 1000) * Bk, tl = lo0 + nb * Bk;
        const size_t fu = std::min(tl + gk, lo0 + (size_t) (rng() % (gk + 1)));   // a first unmapped byte at or above the room's first block edge
        const size_t fr = (size_t) (rng() % (64 * M)), kp = (size_t) (rng() % (4 * M));
        const size_t r = poolx_room_give_back_chunks(lo0, fu, tl, fr, kp, Bk, gk);
        const size_t can = fr > kp ? (fr - kp) / gk * gk : 0;
        auto need = [&](size_t x) { return x > fu ? (x - fu + gk - 1) / gk * gk : 0; };
        bad += r < lo0 || r > tl || (r - lo0) % Bk != 0 || need(r) > can || (r + Bk <= tl && need(r + Bk) <= can);
    }
    report("random rooms: the largest block edge whose chunks fit, never more", bad == 0, fmt("%d bad of 20000", bad));
}
int main(int argc, char ** argv) {
    if (argc > 1) { g_iters = atol(argv[1]); }
    if (argc > 2) { g_seed  = (unsigned long) atol(argv[2]); }
    test_regress_past_rows();
    test_regress_coalesced_ring();
    test_regress_ring_depth();
    test_regress_ring_costly();
    test_block_sizing();
    test_lend_granularity();
    test_ledger_random();
    test_grid_claims();
    test_detectors();
    test_gate();
    test_range_collision();
    test_threads();
    test_regress_padded_rows();
    test_experts_floor();
    test_regress_lend_potential_cells();
    test_regress_zone_across_classes();
    test_regress_potential_matches_place();
    test_regress_capacity_is_gap();
    test_regress_least_over_cards();
    test_regress_chain_reuses_claimed_grid();
    test_regress_k_ahead_room();
    test_regress_stage_past_zone();
    test_regress_cuda_room();
    test_regress_cuda_kernel_oom_trace();
    test_regress_pool_start_pins();
    test_regress_room_shadow_tags();
    test_regress_room_graph_warmup();
    test_regress_room_own_calls();
    test_regress_ring_cold_order();
    test_regress_ring_early();
    test_regress_room_give_back_chunks();
    test_regress_pool_buffer_limits();
    if (n_failed > 0) {
        printf("%d tests failed\n", n_failed);
        return 1;
    }
    printf("All tests passed (seed %lu, %ld iterations)\n", g_seed, g_iters);
    return 0;
}
