// fork: the three last-10 histories of a cached element and the admission rule built on them (ggml/src/ggml-hist-fork.h).
// g++ -std=c++17 -O1 tests/test-hist-fork.cpp -o test-hist-fork && ./test-hist-fork
#include "../ggml/src/ggml-hist-fork.h"

#include <cassert>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

// one recall of element e at `clock`: a hit when resident, else a miss that is admitted when the rule says so (a free block, or the newcomer's expected reads per stay
// beat the victim's: the resident with the fewest). Returns true on a hit.
struct sim {
    size_t cap;
    std::map<int, hist10> h;
    std::set<int> in;
    size_t hits = 0, misses = 0, admitted = 0, declined = 0;
    double turnover = 1.0;   // clock units a resident stays (measured: capacity / admissions per unit)
    bool rule = true;        // false: the control, every miss admitted over the least recently recalled resident
    std::map<int, uint32_t> last;
    explicit sim(size_t c, bool r = true) : cap(c), rule(r) {}
    bool recall(int e, uint32_t clock) {
        hist10 & x = h[e];
        const bool hit = x.in;
        hist_recall(x, clock);
        last[e] = clock;
        if (hit) { hits++; return true; }
        misses++;
        if (in.size() < cap) { hist_admit(x); in.insert(e); admitted++; return false; }
        int v = -1; double ev = 1e300;
        for (int r : in) {
            const double q = rule ? hist_expect(h[r], clock, turnover) : (double) last[r];
            if (q < ev) { ev = q; v = r; }
        }
        if (!rule || hist_admit_ok(x, clock, turnover, ev)) {
            hist_evict(h[v], clock); in.erase(v);
            hist_admit(x); in.insert(e); admitted++;
        } else {
            declined++;
        }
        return false;
    }
};

int main() {
    // one recall and nothing else known: no expected reads, a full tier declines it (it is read from the SSD), its recall is remembered
    {
        hist10 x;
        hist_recall(x, 5);
        assert(hist_expect(x, 5, 10.0) == 0.0);
        assert(!hist_admit_ok(x, 5, 10.0, 0.5));
        assert(hist_admit_ok(x, 5, 10.0, -1.0));   // a free block (no victim): admitted
        assert(x.n_recall == 1);
    }
    // two recalls 2 units apart, residents stay 10 units: about 5 reads expected per stay
    {
        hist10 x;
        hist_recall(x, 3);
        hist_recall(x, 5);
        const double q = hist_expect(x, 5, 10.0);
        assert(q > 4.0 && q < 6.0);
    }
    // stays that served nothing: not worth admitting, whatever its recall rate says
    {
        hist10 x;
        for (uint32_t c = 0; c < 6; ++c) { hist_recall(x, c); hist_admit(x); hist_evict(x, c); }
        assert(x.n_evict == 6);
        assert(hist_expect(x, 6, 10.0) == 0.0);
        assert(!hist_admit_ok(x, 6, 10.0, 0.5));
    }
    // churns (evicted 8 times) but each stay served 3 reads: it pays back, admitted over a victim worth 1
    {
        hist10 x;
        uint32_t c = 0;
        for (int k = 0; k < 8; ++k) {
            hist_recall(x, c++); hist_admit(x);
            for (int r = 0; r < 3; ++r) { hist_recall(x, c++); }
            hist_evict(x, c++);
        }
        const double q = hist_expect(x, c, 10.0);
        assert(q > 2.9 && q < 3.1);
        assert(hist_admit_ok(x, c, 10.0, 1.0));
    }
    // the rings keep the last 10: an old good stay drops out
    {
        hist10 x;
        hist_admit(x); for (int r = 0; r < 50; ++r) { hist_recall(x, 1); } hist_evict(x, 2);
        for (uint32_t k = 0; k < 10; ++k) { hist_recall(x, 3 + k); hist_admit(x); hist_evict(x, 3 + k); }
        assert(x.n_evict == 10);
        assert(hist_expect(x, 20, 10.0) == 0.0);
    }
    // thrash: a hot set of 50 recalled every unit and 200 one-off rows a unit, 100 blocks. Without the rule (admit all, evict the least
    // recalled) the one-offs flush the hot set; with it the hot set stays and the one-offs go to the SSD.
    {
        size_t hits_of[2] = {};
        for (int rule = 1; rule >= 0; --rule) {
            sim s(100, rule != 0);
            int next_cold = 1000000;
            for (uint32_t clock = 0; clock < 200; ++clock) {
                for (int e = 0; e < 50; ++e) { s.recall(e, clock); }
                for (int k = 0; k < 200; ++k) { s.recall(next_cold++, clock); }
            }
            size_t hot_in = 0;
            for (int e = 0; e < 50; ++e) { hot_in += s.in.count(e); }
            printf("thrash sim, %s: hot in RAM %zu of 50, hits %zu, misses %zu, admitted %zu, declined %zu\n", rule ? "history rule" : "admit all, LRU",
                   hot_in, s.hits, s.misses, s.admitted, s.declined);
            hits_of[rule] = s.hits;
            if (rule) {
                assert(hot_in == 50);
                assert(s.hits >= 50 * 198);   // after the first unit every hot recall hits
            }
        }
        assert(hits_of[0] * 10 < hits_of[1]);   // the control: one-offs flush the hot set
    }
    printf("test-hist-fork: all passed\n");
    return 0;
}
