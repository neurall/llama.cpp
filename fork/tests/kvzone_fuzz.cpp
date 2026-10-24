// KV zone placement fuzz: random pool shapes (size classes back to back, a fixed KV part, a row size, a lend) driven through the
// templates of src/llama-kvzone-fork.h; no GPU, no model. Checks the relations kv_layout and the compute-alias lend rely on:
//   1. the gaps lie inside [0, tail_end)
//   2. gap_fit(h cells) >= h  <=>  gap_place(zone of h cells) finds a place
//   3. a zone the lend potential promised (lend_potential(min_cells)) can be placed behind a lend of that size
//   4. a placed zone passes the layout's own invariant check (kvz_bad)
//   5. lend_potential never grows with the cells asked for
//   6. the pass cap (kvz_pass_cap): a graph of the capped size (a + b * tokens, rounded up to whole blocks) leaves a gap for the zone of the padded prompt rows
// build: g++ -O1 -std=c++17 -I src fork/tests/kvzone_fuzz.cpp -o /tmp/kvzone_fuzz && /tmp/kvzone_fuzz [iterations] [seed]
#include "llama-kvzone-fork.h"
#include <cstdio>
#include <cstdlib>
#include <random>

struct seg_t { size_t base; int n; size_t B; };
struct pool_t {
    std::vector<seg_t> segs; std::vector<int> kvt;
    size_t tail_end = 0, kv_fixed = 0, kv_h = 0, kv_row = 0, kv_start = 0, span = 0;
};

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails++ < 12) { printf("FAIL %s:%d: " #c "\n  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

int main(int argc, char ** argv) {
    const long iters = argc > 1 ? atol(argv[1]) : 200000;
    std::mt19937_64 rng(argc > 2 ? atol(argv[2]) : 1);
    auto R = [&](size_t lo, size_t hi) { return lo + (size_t) (rng() % (hi - lo + 1)); };
    for (long it = 0; it < iters; ++it) {
        pool_t ap;
        const size_t unit = 4096 * R(1, 8);
        size_t at = 0;
        const int ncls = (int) R(1, 6);
        for (int c = 0; c < ncls; ++c) {   // a class: n blocks of B, the next class right behind
            seg_t sg; sg.base = at; sg.B = unit * R(1, 64); sg.n = (int) R(0, 40);
            ap.segs.push_back(sg);
            at += (size_t) sg.n * sg.B;
        }
        ap.tail_end = at;
        ap.kvt.push_back(1);
        ap.kv_row   = R(1, 4096) * 16;
        ap.kv_fixed = rng() % 3 == 0 ? unit * R(0, 100) : 0;
        ap.kv_h     = R(0, 3000);
        const size_t bmax = std::max<size_t>(1, kvz_bmax(ap));

        // 1. gaps lie inside the pool
        const auto gaps = kvz_gaps(ap, R(0, ap.tail_end / 2 + 1), bmax);
        for (const auto & g : gaps) {
            CHECK(g.first < g.second && g.second <= ap.tail_end, "gap [%zu,%zu) outside [0,%zu)", g.first, g.second, ap.tail_end);
        }
        // 5. lend potential never grows with the cells
        size_t prev = (size_t) -1;
        for (size_t mc : { (size_t) 0, (size_t) 64, (size_t) 512, (size_t) 4096, (size_t) 100000 }) {
            const size_t lp = kvz_lend_potential(ap, mc);
            CHECK(lp <= prev, "lend potential grew: %zu after %zu at %zu cells", lp, prev, mc);
            prev = lp;
            // 3. the zone it promised fits behind a lend of that size
            const size_t zone = ap.kv_fixed + std::max(ap.kv_h, mc) * ap.kv_row;
            if (lp > 0 && zone > 0) {   // a zone of 0 bytes is no layout
                const size_t s = kvz_gap_place(ap, bmax, lp, lp, zone);
                CHECK(s != (size_t) -1, "lend potential %zu promised a zone of %zu B (%zu cells) but no gap holds it", lp, zone, std::max(ap.kv_h, mc));
            }
        }
        // 6. the pass cap leaves the zone a place (unless it is the 64-token floor: nothing smaller exists)
        {
            const size_t pad = (size_t) 1 << R(0, 9), n_prompt = R(1, 6000), mirror = rng() % 4 == 0 ? unit * R(0, 8) : 0;
            const double a = (double) (unit * R(0, 200)), b = (double) R(1, 4000) * 16;
            const unsigned cap = kvz_pass_cap(ap, n_prompt, 1u << 30, a, b, pad, mirror);
            if (cap > 64) {
                const size_t lend = kvz_block_ceil(ap, mirror + (size_t) (a + b * cap));   // the lend ends on the block edge at or after its bytes
                const size_t rows = (std::max<size_t>(n_prompt, ap.kv_h) + pad - 1) / pad * pad, zone = ap.kv_fixed + rows * ap.kv_row;
                CHECK(kvz_gap_place(ap, bmax, lend, lend, zone) != (size_t) -1, "pass cap %u (lend %zu, a %.0f b %.0f) leaves no gap for %zu padded rows (%zu B)", cap, lend, a, b, rows, zone);
            }
        }
        // 2 + 4. fit and place agree, the placed zone passes the invariant
        const size_t lend = (R(0, ap.tail_end / 2 + 1) + bmax - 1) / bmax * bmax, want = (R(0, ap.tail_end + 1) + bmax - 1) / bmax * bmax;
        const size_t fit = kvz_gap_fit(ap, bmax, lend, ap.kv_row);
        for (size_t h : { (size_t) 1, fit / 2, fit, fit + 1 }) {
            if (h == 0) { continue; }
            const size_t size = ap.kv_fixed + h * ap.kv_row, s = kvz_gap_place(ap, bmax, lend, want, size);
            CHECK((h <= fit) == (s != (size_t) -1), "fit %zu cells, asked %zu: place %s (lend %zu want %zu)", fit, h, s == (size_t) -1 ? "none" : "found", lend, want);
            if (s != (size_t) -1) {
                pool_t z = ap; z.kv_h = h; z.kv_start = s;
                const char * bad = kvz_bad(z, lend);
                CHECK(!bad, "%s: zone [%zu,%zu) lend %zu", bad ? bad : "", s, s + size, lend);
            }
        }
    }
    printf("%s: %ld pools, %d failures\n", fails ? "FAIL" : "ok", iters, fails);
    return fails != 0;
}
