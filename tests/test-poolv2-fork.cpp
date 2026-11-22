// test-poolv2-fork: pool v2's placement rules (src/llama-poolv2-fork.h): one byte heap in heat order, no size classes. No GPU, no model.
//   build: g++ -std=c++17 -O2 -I src tests/test-poolv2-fork.cpp -o test-poolv2-fork
//   run:   ./test-poolv2-fork                                  the rules against a brute-force search, plus a negative control
//          ./test-poolv2-fork --sim [HOTDIR] [SERVERLOG] [SEED] [--heat=own|prev|global]  the rules replayed on recorded decode heat (Qwen3.8 IQ3_S, the
//          pool of one RTX 3090); --heat: what drives the placement during a prompt: the request's own decode heat (default), the previous request's, or
//          the map a release ships (the engine cannot know the request's own until the switch, where the refill uses it)
#include "../src/llama-poolv2-fork.h"
#include "../src/llama-poolalloc-fork.h"   // poolx_block_class: today's block bytes, to name the types behind the log's size classes

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <string>

static int g_fail = 0;
static void check(const char * what, bool ok) {
    if (!ok) { printf("FAIL: %s\n", what); g_fail++; }
}

// residents and free extents tile [lo, hi) exactly (no overlap, nothing lost), two free extents never touch, every start is a multiple of its unit
static bool heap_ok(const pv2_heap & h, const std::map<int32_t, size_t> & unit_of) {
    size_t at = h.lo;
    auto r = h.res.begin();
    auto g = h.gaps.begin();
    bool gap_last = false;
    while (at < h.hi) {
        if (g != h.gaps.end() && g->first == at) {
            if (g->second == 0 || gap_last) { return false; }
            at += g->second; ++g; gap_last = true;
        } else if (r != h.res.end() && r->first == at) {
            const auto u = unit_of.find(r->second.id);
            if (u == unit_of.end() || r->first % u->second != 0 || r->second.n == 0) { return false; }
            at += r->second.n; ++r; gap_last = false;
        } else {
            return false;
        }
    }
    return at == h.hi && r == h.res.end() && g == h.gaps.end();
}

// the windows in reach, by brute force, nearest distance of each kind: free at an edge of its free extent (fe), free anywhere (fa), over colder residents
// not in flight at an edge of their run or of a resident in it (re), anywhere (ra); and the best one-victim replacement: (victim heat, bytes left over,
// distance) of a window at an edge of a victim's hole (one)
struct vkey_t { int64_t heat; size_t left, d; bool operator<(const vkey_t & o) const { return heat != o.heat ? heat < o.heat : left != o.left ? left < o.left : d < o.d; } };
struct want_t { size_t fe = PV2_NONE, fa = PV2_NONE, re = PV2_NONE, ra = PV2_NONE; bool has_one = false; vkey_t one{}; };
// the hole a resident leaves: from the end of the one before it (or lo) to the start of the one after it (or hi)
static std::pair<size_t, size_t> hole_of(const pv2_heap & h, size_t s) {
    auto it = h.res.find(s);
    return { it == h.res.begin() ? h.lo : std::prev(it)->first + std::prev(it)->second.n, std::next(it) == h.res.end() ? h.hi : std::next(it)->first };
}
static want_t brute(const pv2_heap & h, size_t t, size_t n, size_t u, int64_t heat, size_t reach) {
    want_t w;
    auto blocks = [&](const pv2_res & r) { return r.pin || r.heat >= heat; };
    auto near = [&](size_t & d, size_t A, size_t B, size_t s) { if (s >= A && s + n <= B && pv2_dist(s, t) <= reach) { d = std::min(d, pv2_dist(s, t)); } };
    auto edge = [&](size_t & d, size_t A, size_t B, size_t x) { near(d, A, B, (x + u - 1) / u * u); if (x >= n) { near(d, A, B, (x - n) / u * u); } };
    for (const auto & [rs, r] : h.res) {
        if (blocks(r)) { continue; }
        const auto [A, B] = hole_of(h, rs);
        for (size_t s : { (A + u - 1) / u * u, B >= n ? (B - n) / u * u : PV2_NONE }) {
            if (s == PV2_NONE || s < A || s + n > B || pv2_dist(s, t) > reach) { continue; }
            const vkey_t k{ r.heat, B - A - n, pv2_dist(s, t) };
            if (!w.has_one || k < w.one) { w.one = k; w.has_one = true; }
        }
    }
    for (size_t s = (h.lo + u - 1) / u * u; s + n <= h.hi; s += u) {
        bool any = false, blocked = false;
        for (const auto & [rs, r] : h.res) {
            if (rs < s + n && rs + r.n > s) { any = true; blocked = blocked || blocks(r); }
        }
        if (!blocked) { near(any ? w.ra : w.fa, h.lo, h.hi, s); }
    }
    for (const auto & [a, len] : h.gaps) { edge(w.fe, a, a + len, a); edge(w.fe, a, a + len, a + len); }
    std::vector<std::pair<size_t, size_t>> runs;   // [A, B) between blockers
    size_t A = h.lo;
    for (const auto & [rs, r] : h.res) { if (blocks(r)) { runs.push_back({ A, rs }); A = rs + r.n; } }
    runs.push_back({ A, h.hi });
    for (const auto & [ra, rb] : runs) {
        edge(w.re, ra, rb, ra); edge(w.re, ra, rb, rb);
        for (const auto & [rs, r] : h.res) { if (!blocks(r) && rs >= ra && rs < rb) { edge(w.re, ra, rb, rs); edge(w.re, ra, rb, rs + r.n); } }
    }
    return w;
}
// a pick is right when it is the brute-force answer: a free window (edge first) if one is in reach, else (one) the best one-victim replacement, else a
// window over colder residents (edge first), exactly those evicted, else none (the expert lands: nothing in reach fits)
static bool pick_ok(const pv2_heap & h, const pv2_pick & p, size_t t, size_t n, size_t u, int64_t heat, size_t reach, bool one) {
    const want_t w = brute(h, t, n, u, heat, reach);
    const bool   fr = w.fe != PV2_NONE || w.fa != PV2_NONE;
    const size_t d  = w.fe != PV2_NONE ? w.fe : w.fa != PV2_NONE ? w.fa : w.re != PV2_NONE ? w.re : w.ra;
    if (d == PV2_NONE && !(one && w.has_one)) { return p.s == PV2_NONE; }
    if (p.s == PV2_NONE || p.s % u != 0 || p.s < h.lo || p.s + n > h.hi) { return false; }
    size_t k = 0;
    for (const auto & [rs, r] : h.res) {
        if (rs < p.s + n && rs + r.n > p.s) {
            if (k >= p.evict.size() || p.evict[k] != rs || r.pin || r.heat >= heat) { return false; }
            ++k;
        }
    }
    if (k != p.evict.size() || (k == 0) != fr) { return false; }
    if (!fr && one && w.has_one) {
        if (k != 1) { return false; }
        const auto [A, B] = hole_of(h, p.evict[0]);
        const vkey_t got{ h.res.at(p.evict[0]).heat, B - A - n, pv2_dist(p.s, t) };
        return p.s >= A && p.s + n <= B && !(w.one < got) && !(got < w.one);
    }
    return pv2_dist(p.s, t) == d;
}

// random newcomers, drops and pins on a small heap; with `oracle` every pick is held against the brute force and the heap against its invariants
static uint64_t stress(uint64_t seed, int ops, bool oracle) {
    static const size_t units[] = { 8, 24, 40, 64 };
    std::mt19937_64 rng(seed);
    pv2_heap h = pv2_make(1000, 7000);   // lo on no unit: alignment is to the address, not to the heap's start
    std::map<int32_t, size_t> unit_of;
    int32_t next = 0;
    int seen[3] = { 0, 0, 0 };   // picks into a free window, over evicted residents, none
    for (int op = 0; op < ops; ++op) {
        const int k = (int) (rng() % 10);
        if (k < 7) {
            const size_t  u     = units[rng() % 4];
            const size_t  n     = 40 + rng() % 400;
            const size_t  t     = h.lo + rng() % (h.hi - h.lo);
            const int64_t heat  = (int64_t) (rng() % 13) - 1;   // -1: a landing leftover's heat
            const size_t  reach = rng() % 2 ? h.hi - h.lo : rng() % 1500;
            const bool    one   = rng() % 4 != 0;   // false: the landing ring's rule
            const pv2_pick p = pv2_place(h, t, n, u, heat, reach, one);
            seen[p.s == PV2_NONE ? 2 : p.evict.empty() ? 0 : 1]++;
            if (oracle) { check("place: the brute-force answer (free window, else one victim, else several, else lands)", pick_ok(h, p, t, n, u, heat, reach, one)); }
            if (p.s != PV2_NONE) {
                std::vector<pv2_res> out;
                check("commit: the pick's window takes the newcomer", pv2_commit(h, p, { next, n, heat, false }, &out));
                for (const auto & v : out) { check("evict: never a hotter, equally hot or in-flight resident", !v.pin && v.heat < heat); }
                unit_of[next++] = u;
            }
        } else if (!h.res.empty()) {
            auto it = h.res.begin();
            std::advance(it, rng() % h.res.size());
            if (k < 9) { pv2_drop(h, it->first); } else { it->second.pin = !it->second.pin; }
        }
        if (oracle) { check("heap: no overlap, free map exact and merged, starts aligned", heap_ok(h, unit_of)); }
    }
    if (oracle) {
        printf("stress: %d picks into free windows, %d over evicted residents, %d with no place\n", seen[0], seen[1], seen[2]);
        check("stress: every outcome is exercised", seen[0] > 100 && seen[1] > 100 && seen[2] > 100);
    }
    uint64_t x = 1469598103934665603ull;
    for (const auto & [s, r] : h.res) { x = (x ^ s) * 1099511628211ull; x = (x ^ (uint64_t) r.id) * 1099511628211ull; }
    return x;
}

// ---- pv2_plan_move: the brute-force oracle. Every order-preserving layout of the residents left after the k coldest eligible ones go, plus the newcomer at
// a slot with the fewest heat inversions, in a small heap; pinned ones stay; the cost is the bytes of the residents whose start changes. ----
struct mv_key { int64_t cost = INT64_MAX, d = INT64_MAX, s = INT64_MAX; bool ok = false; size_t k = 0; };
static mv_key brute_move(const pv2_heap & h, size_t t, size_t n, size_t u, int64_t heat, size_t budget) {
    mv_key best;
    // a free window anywhere: no eviction, no move
    for (size_t s = (h.lo + u - 1) / u * u; s + n <= h.hi; s += u) {
        bool free = true;
        for (const auto & [rs, r] : h.res) { if (rs < s + n && s < rs + r.n) { free = false; } }
        if (free && (!best.ok || (int64_t) pv2_dist(s, t) < best.d)) { best.ok = true; best.d = pv2_dist(s, t); best.s = s; best.cost = 0; }
    }
    if (best.ok) { return best; }
    std::vector<std::pair<size_t, pv2_res>> vic;
    for (const auto & [s, r] : h.res) { if (!r.pin && r.id != PV2_HOLD && r.heat < heat) { vic.push_back({ s, r }); } }
    std::sort(vic.begin(), vic.end(), [](const auto & a, const auto & b) {
        return a.second.heat != b.second.heat ? a.second.heat < b.second.heat : a.second.id != b.second.id ? a.second.id < b.second.id : a.first < b.first; });
    for (size_t k = 0; k <= std::min<size_t>(vic.size(), PV2_MOVE_EVICT); ++k) {
        std::vector<std::pair<size_t, pv2_res>> L;
        for (const auto & [s, r] : h.res) {
            bool gone = false;
            for (size_t i = 0; i < k; ++i) { gone = gone || vic[i].first == s; }
            if (!gone) { L.push_back({ s, r }); }
        }
        const size_t m = L.size();
        std::vector<int> inv(m + 1, 0);
        int mi = INT32_MAX;
        for (size_t j = 0; j <= m; ++j) {
            for (size_t i = 0; i < m; ++i) { inv[j] += i < j ? L[i].second.heat > heat : L[i].second.heat < heat; }
            mi = std::min(mi, inv[j]);
        }
        mv_key kb;
        for (size_t j = 0; j <= m; ++j) {
            if (inv[j] != mi) { continue; }
            // elements in order: L[0..j), the newcomer, L[j..m)
            std::vector<int> seq;
            for (size_t i = 0; i < j; ++i) { seq.push_back((int) i); }
            seq.push_back(-1);
            for (size_t i = j; i < m; ++i) { seq.push_back((int) i); }
            std::function<void(size_t, size_t, int64_t, int64_t, int64_t)> go = [&](size_t q, size_t at, int64_t cost, int64_t ds, int64_t sn) {
                if (q == seq.size()) {
                    if (!kb.ok || cost < kb.cost || (cost == kb.cost && (ds < kb.d || (ds == kb.d && sn < kb.s)))) { kb.ok = true; kb.cost = cost; kb.d = ds; kb.s = sn; }
                    return;
                }
                const int e = seq[q];
                const size_t en = e < 0 ? n : L[e].second.n, eu = e < 0 ? u : L[e].second.u;
                for (size_t s = (at + eu - 1) / eu * eu; s + en <= h.hi; s += eu) {
                    if (e >= 0 && L[e].second.pin && s != L[e].first) { continue; }
                    go(q + 1, s + en, cost + (e >= 0 && s != L[e].first ? en : 0), e < 0 ? (int64_t) pv2_dist(s, t) : ds, e < 0 ? (int64_t) s : sn);
                }
            };
            go(0, h.lo, 0, 0, 0);
        }
        if (kb.ok && (size_t) kb.cost <= budget) { kb.k = k; return kb; }
    }
    return best;
}

static pv2_heap rand_move_heap(std::mt19937 & g, std::map<int32_t, size_t> & uo) {
    const size_t H = 12 + g() % 14;
    pv2_heap h = pv2_make(0, H);
    static const size_t us[3] = { 1, 2, 4 };
    size_t at = 0;
    int32_t id = 0;
    for (int i = 0; i < 5; ++i) {
        at += g() % 4 == 0 ? g() % 3 : 0;
        const size_t u = us[g() % 3], n = 1 + g() % 5;
        at = (at + u - 1) / u * u;
        if (at + n > H) { break; }
        pv2_res r; r.id = id; r.n = n; r.u = u; r.heat = g() % 10; r.pin = g() % 7 == 0;
        pv2_add(h, at, r);
        uo[id++] = u;
        at += n;
    }
    return h;
}

static void move_tests() {
    const auto ids_at = [](const pv2_heap & h) { std::vector<int32_t> v; for (const auto & [s, r] : h.res) { v.push_back(r.id); } return v; };
    {   // a free window anywhere: the one nearest the heat target, nothing moves, nothing leaves
        pv2_heap h = pv2_make(0, 100);
        pv2_add(h, 10, { 1, 20, 5, false, 1 }); pv2_add(h, 40, { 2, 20, 5, false, 1 }); pv2_add(h, 70, { 3, 10, 5, false, 1 });
        // free: [0,10) [30,40) [60,70) [80,100)
        pv2_move m = pv2_plan_move(h, 85, 15, 1, 7);
        check("move: a free window takes the newcomer nearest its target, no step, no eviction", m.ok && m.s == 85 && m.steps.empty() && m.evict.empty() && m.bytes == 0);
        m = pv2_plan_move(h, 0, 9, 1, 7);
        check("move: a window at the cold end when the target is there", m.ok && m.s == 0 && m.steps.empty());
        m = pv2_plan_move(h, 50, 10, 1, 7);
        check("move: the target is inside a resident: the nearest window that fits, ties to the lower start", m.ok && (m.s == 30 || m.s == 60) && m.steps.empty() && m.evict.empty());
    }
    {   // a full heap in heat order: the coldest leaves, the chain below the newcomer's place shifts one step to the cold end, the newcomer goes last
        pv2_heap h = pv2_make(0, 100);
        for (int i = 0; i < 10; ++i) { pv2_add(h, 10 * i, { i, 10, 10 * (i + 1), false, 1 }); }
        pv2_move m = pv2_plan_move(h, 40, 10, 1, 55);
        check("move: full heap: the coldest resident (id 0) is the one evicted", m.ok && m.evict.size() == 1 && m.evict[0] == 0);
        check("move: full heap: ids 1..4 each step 10 toward the cold end, in dependency order", m.steps.size() == 4 && m.steps[0].id == 1 && m.steps[0].from == 10 && m.steps[0].to == 0 &&
              m.steps[3].id == 4 && m.steps[3].from == 40 && m.steps[3].to == 30 && m.s == 40 && m.bytes == 40);
        pv2_heap g = h;
        check("move: the plan applies step by step (every destination free when its step runs)", pv2_apply_move(g, m, { 100, 10, 55, false, 1 }));
        std::vector<int32_t> want = { 1, 2, 3, 4, 100, 5, 6, 7, 8, 9 };
        check("move: heat order kept after the insert", ids_at(g) == want && g.gaps.empty());
        // the newcomer is the hottest: it goes to the top, the coldest still the only one to leave
        m = pv2_plan_move(h, 90, 10, 1, 999);
        check("move: hottest newcomer: coldest leaves, everything shifts down, newcomer on top", m.ok && m.evict.size() == 1 && m.evict[0] == 0 && m.steps.size() == 9 && m.s == 90 && m.bytes == 90);
        // colder than everything: nothing may be evicted
        m = pv2_plan_move(h, 0, 10, 1, 5);
        check("move: a newcomer colder than every resident finds nothing to replace", !m.ok);
        m = pv2_plan_move(h, 0, 10, 1, 10);
        check("move: equal heat never replaces", !m.ok);
        // budget: the chain costs 40 bytes
        m = pv2_plan_move(h, 40, 10, 1, 55, 30);
        check("move: a budget under the chain's bytes takes more of the coldest instead (two leave, 30 bytes move)", m.ok && m.evict.size() == 2 && m.bytes == 30 && pv2_plan_move(h, 40, 10, 1, 55, 40).evict.size() == 1);
        check("move: no budget for any chain but evicting the coldest five: they leave and the newcomer lands free", pv2_plan_move(h, 40, 10, 1, 55, 0).ok && pv2_plan_move(h, 40, 10, 1, 55, 0).bytes == 0);
        // pinned coldest: the next coldest goes, the pinned one stays where it is
        pv2_heap p = h;
        p.res.at(0).pin = true;
        m = pv2_plan_move(p, 40, 10, 1, 55);
        check("move: a pinned coldest is neither evicted nor moved; the next coldest leaves", m.ok && m.evict.size() == 1 && m.evict[0] == 10 && m.steps.size() == 3 && m.steps[0].id == 2 && m.s == 40);
        // a pinned resident in the chain's way: only the stretch above it is shifted
        p = h;
        p.res.at(20).pin = true;   // id 2 at 20
        m = pv2_plan_move(p, 40, 10, 1, 55);
        pv2_heap g2 = p;
        check("move: a pinned resident in the way blocks the chain: the plan is sound, never moves it, evicts only a colder one",
              !m.ok || (pv2_apply_move(g2, m, { 100, 10, 55, false, 1 }) && g2.res.at(20).id == 2 && g2.res.count(40) + g2.res.count(m.s) > 0));
    }
    {   // seven byte sizes: the chain steps must fit the gap with the unit; two victims when one hole is too small
        pv2_heap h = pv2_make(0, 60);
        pv2_add(h, 0,  { 1, 8, 1, false, 4 });    // coldest, small
        pv2_add(h, 8,  { 2, 8, 2, false, 4 });
        pv2_add(h, 16, { 3, 12, 8, false, 4 });
        pv2_add(h, 28, { 4, 32, 9, false, 4 });
        pv2_move m = pv2_plan_move(h, 50, 12, 4, 5);   // 12 bytes needed, the coldest holds 8: a second victim or a chain through the free gap
        pv2_heap g = h;
        check("move: sizes: the newcomer is bigger than the coldest, still a sound plan", m.ok && pv2_apply_move(g, m, { 100, 12, 5, false, 4 }));
        check("move: sizes: never a victim hotter than the newcomer", [&] { for (size_t s : m.evict) { if (h.res.at(s).heat >= 5) { return false; } } return true; }());
    }
    {   // the property test against the oracle
        std::mt19937 g(20261121);
        size_t ok_n = 0, chain_n = 0, evict_n = 0;
        bool all = true;
        for (int it = 0; it < 400; ++it) {
            std::map<int32_t, size_t> uo;
            pv2_heap h = rand_move_heap(g, uo);
            static const size_t us[3] = { 1, 2, 4 };
            const size_t n = 1 + g() % 6, u = us[g() % 3], t = g() % (h.hi + 1);
            const int64_t heat = g() % 11;
            const size_t budget = g() % 3 == 0 ? g() % 10 : SIZE_MAX;
            const mv_key bk = brute_move(h, t, n, u, heat, budget);
            const pv2_move m = pv2_plan_move(h, t, n, u, heat, budget);
            bool good = m.ok == bk.ok;
            if (good && m.ok) {
                pv2_heap a = h;
                uo[1000] = u;
                size_t moved = 0;
                for (const auto & s : m.steps) { moved += h.res.at(s.from).n; }
                bool evict_ok = m.evict.size() == bk.k;
                int64_t eh = 0;
                for (size_t s : m.evict) { evict_ok = evict_ok && !h.res.at(s).pin && h.res.at(s).heat < heat; eh += h.res.at(s).heat; }
                for (const auto & s : m.steps) { evict_ok = evict_ok && !h.res.at(s.from).pin; }
                good = evict_ok && pv2_apply_move(a, m, { 1000, n, heat, false, u }) && heap_ok(a, uo) && a.res.count(m.s) && a.res.at(m.s).id == 1000 &&
                       (int64_t) moved == bk.cost && (int64_t) m.bytes == bk.cost && (int64_t) pv2_dist(m.s, t) == bk.d && (int64_t) m.s == bk.s;
                ++ok_n; chain_n += !m.steps.empty(); evict_n += !m.evict.empty();
                // the evicted ones are the k coldest eligible: no resident left that is colder and eligible than one that left
                for (size_t s : m.evict) { for (const auto & [rs, r] : h.res) {
                    if (!r.pin && r.id != PV2_HOLD && r.heat < heat && r.heat < h.res.at(s).heat && std::find(m.evict.begin(), m.evict.end(), rs) == m.evict.end()) { good = false; } } }
            }
            if (!good) { all = false; printf("  case %d: planner ok=%d brute ok=%d (k %zu cost %lld d %lld s %lld) planner steps %zu evict %zu s %zu bytes %zu\n", it, (int) m.ok, (int) bk.ok, bk.k, (long long) bk.cost, (long long) bk.d, (long long) bk.s, m.steps.size(), m.evict.size(), m.s, m.bytes); }
        }
        check("move: 400 random heaps agree with the brute-force oracle (ok, k, moved bytes, place; every plan applies and keeps the heap sound)", all);
        check("move: the random heaps exercise chains and evictions", ok_n > 100 && chain_n > 20 && evict_n > 20);
        printf("move oracle: %zu plans, %zu with a chain, %zu with an eviction\n", ok_n, chain_n, evict_n);
    }
    {   // a stream far longer than the heap: nothing hotter than the newcomer is ever lost; a full sorted heap stays sorted
        std::mt19937 g(77);
        pv2_heap h = pv2_make(0, 100);
        for (int i = 0; i < 10; ++i) { pv2_add(h, 10 * i, { i, 10, 1000 * (i + 1), false, 1 }); }
        bool good = true;
        int32_t next = 100;
        size_t kept = 0, refused = 0;
        for (int i = 0; i < 400; ++i) {
            const int64_t heat = g() % 11000;
            size_t colder = 0;
            for (const auto & [s, r] : h.res) { colder += r.heat < heat || (r.heat == heat); }
            int64_t lo = INT64_MAX;
            for (const auto & [s, r] : h.res) { lo = std::min(lo, r.heat); }
            const pv2_move m = pv2_plan_move(h, 10 * (colder > 9 ? 9 : colder), 10, 1, heat);
            if (!m.ok) { good = good && heat <= lo; ++refused; continue; }
            good = good && m.evict.size() == 1 && h.res.at(m.evict[0]).heat == lo && lo < heat;
            good = good && pv2_apply_move(h, m, { next++, 10, heat, false, 1 });
            int64_t prev = -1;
            for (const auto & [s, r] : h.res) { good = good && r.heat >= prev; prev = r.heat; }
            good = good && h.gaps.empty() && h.res.size() == 10;
            ++kept;
        }
        printf("stream: kept %zu refused %zu good %d\n", kept, refused, (int) good);
        check("move: a 400-expert stream through a 10-expert heap: only the coldest ever leaves, the heap stays full and heat-sorted", good && kept > 30 && refused > 5);   // the heap warms as the stream runs: most late newcomers are colder than all it holds and are refused
    }
}

// pv2_sort_step's oracle: the best single move on this heap state (any resident not pinned, any aligned start free but for its own bytes), by distance gained
// toward its target, then the lower id, then the lower start; only moves within `left` bytes and with a gain
static bool brute_sort_best(const pv2_heap & h, const std::map<int32_t, size_t> & tgt, size_t left, pv2_step & best) {
    size_t bg = 0;
    bool found = false;
    for (const auto & [s, r] : h.res) {
        const auto it = tgt.find(r.id);
        if (r.pin || it == tgt.end() || r.n > left) { continue; }
        for (size_t x = (h.lo + r.u - 1) / r.u * r.u; x + r.n <= h.hi; x += r.u) {
            bool free = true;
            for (const auto & [s2, r2] : h.res) { if (s2 != s && s2 < x + r.n && x < s2 + r2.n) { free = false; } }
            const size_t d0 = pv2_dist(s, it->second), d1 = pv2_dist(x, it->second);
            if (!free || d1 >= d0) { continue; }
            const size_t g = d0 - d1;
            if (!found || g > bg || (g == bg && (r.id < best.id || (r.id == best.id && x < best.to)))) { found = true; bg = g; best = { r.id, s, x }; }
        }
    }
    return found;
}
static void sort_tests() {
    {   // a hot expert left at the cold end moves next to its target; a sorted heap needs nothing
        pv2_heap h = pv2_make(0, 100);
        pv2_add(h, 0,  { 1, 10, 9, false, 1 });   // hot, target 90
        pv2_add(h, 10, { 2, 10, 1, false, 1 });   // cold, target 0
        pv2_add(h, 80, { 3, 10, 5, false, 1 });   // target 80
        std::map<int32_t, size_t> tgt = { { 1, 90 }, { 2, 0 }, { 3, 80 } };
        auto st = pv2_sort_step(h, tgt, SIZE_MAX);
        check("sort: the misplaced hot expert moves to its target first", !st.empty() && st[0].id == 1 && st[0].from == 0 && st[0].to == 90);
        check("sort: then the cold one slides to the cold end, then nothing is left to gain", st.size() == 2 && st[1].id == 2 && st[1].to == 0);
        pv2_heap g = h;
        for (const auto & s : st) { const pv2_res v = pv2_drop(g, s.from); check("sort: each step applies", pv2_add(g, s.to, v)); }
        check("sort: a heap in place needs no move", pv2_sort_step(g, tgt, SIZE_MAX).empty());
        check("sort: the budget caps the bytes moved", pv2_sort_step(h, tgt, 15).size() == 1 && pv2_sort_step(h, tgt, 9).empty());
        h.res.at(0).pin = true;
        st = pv2_sort_step(h, tgt, SIZE_MAX);
        check("sort: a pinned expert never moves", std::none_of(st.begin(), st.end(), [](const pv2_step & s) { return s.id == 1; }));
    }
    {   // the property test: every step is the oracle's best single move on the state before it, the budget holds, misplacement only falls
        std::mt19937 g(1121);
        bool all = true;
        size_t steps = 0;
        for (int it = 0; it < 300; ++it) {
            std::map<int32_t, size_t> uo, tgt;
            pv2_heap h = rand_move_heap(g, uo);
            for (const auto & [s, r] : h.res) { tgt[r.id] = g() % h.hi; }
            const size_t budget = g() % 2 ? g() % 12 : SIZE_MAX;
            const auto st = pv2_sort_step(h, tgt, budget);
            size_t left = budget;
            pv2_heap a = h;
            for (const auto & s : st) {
                pv2_step b;
                const bool f = brute_sort_best(a, tgt, left, b);
                const size_t n = a.res.count(s.from) ? a.res.at(s.from).n : 0;
                if (!f || b.id != s.id || b.from != s.from || b.to != s.to) { all = false; break; }
                const pv2_res v = pv2_drop(a, s.from);
                if (!pv2_add(a, s.to, v)) { all = false; break; }
                left -= n; ++steps;
            }
            pv2_step b;
            all = all && heap_ok(a, uo) && !brute_sort_best(a, tgt, left, b);   // stops only when nothing fits the budget and gains
        }
        check("sort: 300 random heaps: each step is the best single move, the budget holds, it stops only when no move gains", all && steps > 100);
        printf("sort oracle: %zu steps\n", steps);
    }
}

static int unit_tests() {
    {   // the free map: take splits an extent, give merges with both neighbours
        pv2_heap h = pv2_make(0, 1000);
        check("take: inside one free extent", pv2_take(h, 100, 100) && pv2_take(h, 200, 100));
        check("take: not twice", !pv2_take(h, 150, 10));
        check("take: split", h.gaps.size() == 2 && h.gaps.at(0) == 100 && h.gaps.at(300) == 700);
        pv2_give(h, 100, 100);
        check("give: merged left", h.gaps.size() == 2 && h.gaps.at(0) == 200);
        pv2_give(h, 200, 100);
        check("give: merged both sides", h.gaps.size() == 1 && h.gaps.at(0) == 1000);
    }
    {   // fit: the aligned start nearest the target inside an extent
        check("fit: target inside", pv2_fit(100, 1000, 100, 64, 500) == 512);
        check("fit: round down when nearer", pv2_fit(100, 1000, 100, 64, 520) == 512);
        check("fit: clamped to the last start", pv2_fit(100, 1000, 100, 64, 990) == 896);
        check("fit: clamped to the first start", pv2_fit(100, 1000, 100, 64, 0) == 128);
        check("fit: too small after alignment", pv2_fit(100, 200, 100, 64, 0) == PV2_NONE);
    }
    {   // the real expert (Qwen3.8 IQ3_S up/gate, IQ4_NL down rows of 640 values): one table for its three slices, so one unit for all three views
        const poolx_slice sl[3] = { { 704000, 110, 256, 2560 }, { 704000, 110, 256, 2560 }, { 921600, 18, 32, 640 } };
        const pv2_shape sh = pv2_shape_of(sl);
        check("shape: unit lcm(16, 110, 110, 18)", sh.u == 7920);
        check("shape: slices at today's block offsets", sh.off[0] == 0 && sh.off[1] == 704000 && sh.off[2] == 1408000);
        check("shape: extent = down's end + the 216 bytes read past its last row", sh.n == 1408000 + 921600 + 216);
        check("shape: no more than today's block", sh.n <= poolx_block_class(sl).B);
        // the kernels step a view by nb[2] / block bytes: exact for every slice type; a slice starts on 16 bytes wherever its expert starts on the unit
        bool al = sh.u % 16 == 0;
        for (int k = 0; k < 3; ++k) { al = al && sh.u % sl[k].ts == 0 && sh.off[k] % 16 == 0; }
        check("shape: the unit steps every slice exactly, slices on 16 bytes", al);
        const poolx_slice ol[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1720320, 210, 256, 1024 } };   // OLMoE Q4_K up/gate, Q6_K down
        const pv2_shape so = pv2_shape_of(ol);
        check("shape: OLMoE Q6_K down: unit 5040, nothing read past", so.u == 5040 && so.n == 2 * 1179648 + 1720320);
    }
    {   // stage 1, unit views over today's classes: every real class (Qwen3.8 IQ3_S's 7, OLMoE's 2) seen through views that step by its unit
        struct ty { size_t ts, blck; };
        const ty ug[] = { { 82, 256 }, { 98, 256 }, { 110, 256 }, { 136, 256 } }, dn[] = { { 18, 32 }, { 18, 64 } };   // iq2_s iq3_xxs iq3_s iq4_xs / iq4_nl q2_0
        std::vector<std::array<poolx_slice, 3>> cls;
        for (const auto & a : ug) {
            for (const auto & d : dn) {
                const size_t ugb = 640 * (2560 / a.blck) * a.ts, dnb = 2560 * (640 / d.blck) * d.ts;
                cls.push_back({ poolx_slice{ ugb, a.ts, a.blck, 2560 }, poolx_slice{ ugb, a.ts, a.blck, 2560 }, poolx_slice{ dnb, d.ts, d.blck, 640 } });
            }
        }
        cls.push_back({ poolx_slice{ 1179648, 144, 256, 2048 }, poolx_slice{ 1179648, 144, 256, 2048 }, poolx_slice{ 1179648, 144, 256, 1024 } });
        cls.push_back({ poolx_slice{ 1179648, 144, 256, 2048 }, poolx_slice{ 1179648, 144, 256, 2048 }, poolx_slice{ 1720320, 210, 256, 1024 } });
        bool ok = true;
        for (const auto & c : cls) {
            const size_t B = poolx_block_class(c.data()).B, u = pv2_shape_of(c.data()).u;
            ok = ok && B % u == 0;
            for (int64_t n : { 1, 2, 7, 5733 }) {
                const int64_t ne2 = pv2_view_ne2(n, B, u);
                for (int32_t b : { 0, 1, (int32_t) n / 2, (int32_t) n - 1 }) {
                    if (b < 0 || b >= n) { continue; }
                    const int32_t e = pv2_block_entry(b, B, u);
                    // the kernels read block b at e x nb[2] = e x u, its last slice ends inside the view, the entry is one of the view's
                    ok = ok && (size_t) e * u == (size_t) b * B && e < ne2 && (size_t) (ne2 - 1) * u + c[2].nb2 >= (size_t) b * B + c[2].nb2;
                }
                ok = ok && pv2_view_ne2(n, B, B) == n && pv2_block_entry((int32_t) n - 1, B, B) == (int32_t) n - 1;   // u = B: today's views and tables
            }
        }
        check("stage 1: a block of every real class is whole units, its entry x unit is its offset, inside the view; u = B is today", ok);
        check("stage 1: an empty class has an empty view", pv2_view_ne2(0, 2344320, 7920) == 0);
    }
    {   // free first: a colder resident at the target, a free window farther: the free one, nothing evicted
        pv2_heap h = pv2_make(0, 1000);
        pv2_add(h, 400, { 1, 100, 3, false });
        pv2_add(h, 500, { 2, 100, 9, false });
        pv2_add(h, 600, { 3, 300, 9, false });
        pv2_add(h, 0,   { 4, 400, 9, false });
        const pv2_pick p = pv2_place(h, 400, 100, 1, 5, 1000, true);
        check("place: a free window before an eviction", p.s == 900 && p.evict.empty());
        pv2_add(h, 900, { 5, 100, 9, false });
        const pv2_pick q = pv2_place(h, 450, 100, 1, 5, 1000, true);
        check("place: no free window: the colder resident's", q.s == 400 && q.evict.size() == 1 && q.evict[0] == 400);
        check("place: never an equally hot one", pv2_place(h, 450, 100, 1, 3, 1000, true).s == PV2_NONE);
        h.res.at(400).pin = true;
        check("place: never one in flight", pv2_place(h, 450, 100, 1, 5, 1000, true).s == PV2_NONE);
        h.res.at(400).pin = false;
        check("place: out of reach is no place", pv2_place(h, 950, 100, 1, 5, 400, true).s == PV2_NONE);
        check("place: two colder residents make one window", pv2_place(h, 0, 450, 1, 10, 1000, true).evict.size() == 2);
        h.res.at(500).heat = 1;   // a second victim, colder than resident 1 and farther from the target
        const pv2_pick c = pv2_place(h, 450, 100, 1, 5, 1000, true);
        check("place: the coldest victim, not the nearest", c.s == 500 && c.evict.size() == 1 && c.evict[0] == 500);
        const pv2_pick r = pv2_place(h, 450, 100, 1, 5, 1000, false);
        check("place: the ring takes the nearest victim", r.s == 400 && r.evict.size() == 1 && r.evict[0] == 400);
    }
    {   // heat order: distances from the hot end, ties by id; past the heap the target is its cold end
        const std::vector<int64_t> heat = { 5, 9, 5, 0 };
        const std::vector<size_t>  n    = { 10, 20, 30, 40 };
        const auto d = pv2_heat_dist(heat, n);
        check("heat order: hottest at 0, ties by id", d[1] == 0 && d[0] == 20 && d[2] == 30 && d[3] == 60);
        check("target: below the hotter ones", pv2_target(0, 100, 20, 10) == 70);
        check("target: past the heap: its cold end", pv2_target(0, 100, 95, 10) == 0);
    }
    {   // c1 boot fill: OLMoE's shape (16 layers x 64 experts, 8 used): one heat scale over all layers (a count's share of its layer x 1e9), hottest first,
        // ties to the lower id (id = layer x 64 + expert), the unpicked (heat < 0) left out; along the order the heap packs down from the top
        const int L = 16, E = 64;
        std::vector<int64_t> heat(L * E, -1);
        for (int l = 0; l < L; ++l) {
            for (int e = 0; e < 10; ++e) { heat[l * E + e] = pv2_share_heat(100 - 10 * e + (l % 2), 1000); }   // 10 picked per layer
        }
        check("share heat: a count's share x 1e9", pv2_share_heat(1, 4) == 250000000 && pv2_share_heat(0, 0) == 0 && pv2_share_heat(3, 0) == 3000000000LL);
        const auto o = pv2_boot_order(heat);
        bool desc = o.size() == (size_t) L * 10, ties = true;
        for (size_t i = 1; desc && i < o.size(); ++i) {
            desc = heat[o[i - 1]] >= heat[o[i]];
            if (heat[o[i - 1]] == heat[o[i]]) { ties = ties && o[i - 1] < o[i]; }
        }
        check("boot order: every picked expert once, hottest first", desc);
        check("boot order: ties to the lower id", ties);
        check("boot order: an odd layer's hottest before layer 0's", o[0] == 1 * E + 0 && o[7] == 15 * E + 0 && o[8] == 0 * E + 0);
        std::vector<size_t> n(L * E, 0);
        for (int l = 0; l < L; ++l) { for (int e = 0; e < E; ++e) { n[l * E + e] = 1179648 + 4096 * (l % 3); } }   // OLMoE Q4 expert bytes, three units
        const auto d = pv2_heat_dist(heat, n);
        bool packs = d[o[0]] == 0;
        for (size_t i = 1; i < o.size(); ++i) { packs = packs && d[o[i]] == d[o[i - 1]] + n[o[i - 1]]; }
        check("boot order: the heap packs down from the top along it", packs);
    }
    {   // c2 target: packed down from the top of the main heap [0, 1000) with the landing hole [600, 800) skipped
        check("target main: the hottest just under the top", pv2_target_main(0, 1000, 600, 800, 0, 100) == 900);
        check("target main: below the hole as it is", pv2_target_main(0, 1000, 600, 800, 150, 100) == 550);
        check("target main: a target at the hole's start goes above it", pv2_target_main(0, 1000, 600, 800, 100, 100) == 800);
        check("target main: past the heap: its cold end", pv2_target_main(0, 1000, 600, 800, 750, 100) == 0);
        check("target main: no hole is pv2_target", pv2_target_main(0, 1000, 1000, 1000, 150, 100) == pv2_target(0, 1000, 150, 100));
    }
    {   // c2 decode heat: xscore x 1e6; a victim must be colder than the newcomer by the pay-back margin; the dist refresh every 64 steps
        check("decode heat: xscore x 1e6", pv2_decode_heat(0.25) == 250000 && pv2_decode_heat(0.0) == 0);
        check("victim heat: less the margin", pv2_victim_heat(0.25, 0.05) == 200000);
        {   // zone sort: the KV zone [0, Z) must only take the coldest: hotter residents move out above Z (free window first, else over a colder one there)
            pv2_heap h = pv2_make(0, 1000);
            pv2_add(h, 0,   { 1, 100, 100, false, 1 });   // hot, in the zone
            pv2_add(h, 100, { 2, 100, 1,   false, 1 });   // cold, in the zone
            pv2_add(h, 200, { 3, 100, 50,  false, 1 });   // above the zone (Z = 200)
            const pv2_zone zp = pv2_sort_range(h, 200, SIZE_MAX, 1);   // heat above 1 moves; the cold one stays
            check("zone sort: the hot one leaves the zone into the free window above it", zp.steps.size() == 1 && zp.steps[0].id == 1 && zp.steps[0].to >= 200 && zp.evicted.empty());
            pv2_heap g = h;   // apply and check: the zone now holds only the cold one
            for (const auto & st : zp.steps) { const pv2_res v = pv2_drop(g, st.from); if (st.to != PV2_NONE) { pv2_add(g, st.to, v); } }
            bool hot_in_zone = false;
            for (const auto & [x, r] : g.res) { hot_in_zone = hot_in_zone || (x < 200 && r.heat > 1); }
            check("zone sort: no hot resident left in the zone", !hot_in_zone);
            // no free window above: a colder resident there is evicted, never a hotter one
            pv2_heap f = pv2_make(0, 400);
            pv2_add(f, 0,   { 1, 100, 100, false, 1 });
            pv2_add(f, 100, { 2, 100, 1,   false, 1 });
            pv2_add(f, 200, { 3, 100, 5,   false, 1 });   // colder than the hot one
            pv2_add(f, 300, { 4, 100, 500, false, 1 });   // hotter
            const pv2_zone fz = pv2_sort_range(f, 200, SIZE_MAX, 0);
            check("zone sort: the colder resident above the zone is evicted for the hot one", fz.steps.size() == 2 && fz.steps[0].id == 3 && fz.steps[0].to == PV2_NONE && fz.steps[1].id == 1 && fz.evicted.size() == 1 && fz.evicted[0] == 3);
            const pv2_zone bz = pv2_sort_range(f, 200, 50, 0);   // a budget below one expert: nothing moves
            check("zone sort: the byte budget stops the moves", bz.steps.empty() && bz.evicted.empty());
            // the landing hole (pv2_hold, no resident entry) is never a destination
            pv2_heap q = pv2_make(0, 400);
            pv2_add(q, 0, { 1, 100, 100, false, 1 });
            pv2_add(q, 100, { 2, 100, 5, false, 1 });
            pv2_hold(q, 200, 400);
            const pv2_zone qz = pv2_sort_range(q, 100, SIZE_MAX, 0);
            bool in_hole = false;
            for (const auto & st : qz.steps) { in_hole = in_hole || (st.to != PV2_NONE && st.to + 100 > 200); }
            check("zone sort: the landing hole is no destination", !in_hole);
            // pinned residents and holds stay
            pv2_heap p = pv2_make(0, 400);
            pv2_add(p, 0, { 1, 100, 100, true, 1 });
            pv2_add(p, 100, { PV2_HOLD, 100, 0, true, 1 });
            const pv2_zone pz = pv2_sort_range(p, 200, SIZE_MAX, 0);
            check("zone sort: a pinned resident and a hold do not move", pz.steps.empty());
        }
        {   // property: random heaps, the zone's total heat never rises, the heap stays sound, evictions are colder than what moved
            std::mt19937 g(4242);
            bool good = true; size_t plans = 0, moved = 0;
            for (int it = 0; it < 300; ++it) {
                pv2_heap h = pv2_make(0, 2000);
                std::vector<size_t> starts;
                size_t at = 0;
                while (at + 40 < 2000) {
                    const size_t n = 20 + (g() % 3) * 20;
                    if (g() % 4 == 0) { at += n; continue; }
                    pv2_add(h, at, { (int32_t) starts.size() + 1, n, (int64_t) (g() % 1000), g() % 7 == 0, 1 });
                    starts.push_back(at); at += n;
                }
                const size_t Z = 400 + (g() % 8) * 100;
                auto zone_heat = [&](const pv2_heap & x) { int64_t t = 0; for (const auto & [s2, r] : x.res) { if (s2 < Z && r.id > 0) { t += r.heat * (int64_t) r.n; } } return t; };
                const int64_t before = zone_heat(h);
                const pv2_zone zp = pv2_sort_range(h, Z, SIZE_MAX, 0);
                pv2_heap a = h;
                bool applied = true;
                for (const auto & st : zp.steps) { if (!a.res.count(st.from)) { applied = false; break; } const pv2_res v = pv2_drop(a, st.from); if (st.to != PV2_NONE && !pv2_add(a, st.to, v)) { applied = false; break; } }
                if (!applied && good) { printf("  case %d: a step did not apply (%zu steps, %zu evicted)\n", it, zp.steps.size(), zp.evicted.size()); }
                good = good && applied;
                int64_t prev_end = 0; bool sound = true;
                for (const auto & [s2, r] : a.res) { sound = sound && s2 >= (size_t) prev_end && s2 + r.n <= a.hi; prev_end = s2 + r.n; }
                const bool colder = zone_heat(a) <= before;
                if ((!sound || !colder) && good) { printf("  case %d: sound %d zone heat %lld -> %lld, %zu steps %zu evicted\n", it, (int) sound, (long long) before, (long long) zone_heat(a), zp.steps.size(), zp.evicted.size()); }
                good = good && sound && colder;
                int64_t max_moved = -1, max_ev = -1;
                for (const auto & st : zp.steps) { for (const auto & [s2, r] : h.res) { if (r.id == st.id) { max_moved = std::max(max_moved, r.heat); } } }
                for (int32_t id : zp.evicted) { for (const auto & [s2, r] : h.res) { if (r.id == id) { max_ev = std::max(max_ev, r.heat); } } }
                if (!(zp.evicted.empty() || max_ev < max_moved) && good) { printf("  case %d: evicted heat %lld vs hottest mover %lld\n", it, (long long) max_ev, (long long) max_moved); }
                good = good && (zp.evicted.empty() || max_ev < max_moved);   // every eviction is colder than the hottest mover (each is colder than its own mover)
                plans += !zp.steps.empty(); moved += zp.steps.size();
            }
            printf("zone sort property: %zu heaps changed, %zu moves\n", plans, moved);
            check("zone sort: 300 random heaps stay sound, the zone only gets colder, evictions are colder than the movers", good && plans > 100);
        }
        // move budget: 1 MiB newcomer, 25 GB/s link, 450 GB/s copies: the pay-back cap is 18 MiB; a 10 us window caps it at 4.5 MB; no window, no moves
        check("move budget: pay-back cap", pv2_move_budget(1 << 20, 25.0, 450.0, 1e9) == (size_t) (18.0 * (1 << 20)));
        check("move budget: idle window cap", pv2_move_budget(1 << 20, 25.0, 450.0, 10.0) == 4500000);
        check("move budget: no window or rate, nothing moves", pv2_move_budget(1 << 20, 25.0, 450.0, 0.0) == 0 && pv2_move_budget(1 << 20, 0.0, 450.0, 10.0) == 0 &&
              pv2_move_budget(1 << 20, 25.0, 0.0, 10.0) == 0);
        check("dist due: at the first step and every 64", pv2_dist_due(0, PV2_NONE) && !pv2_dist_due(63, 0) && pv2_dist_due(64, 0) && pv2_dist_due(130, 64));
        pv2_heap h = pv2_make(0, 1000);
        pv2_add(h, 400, { 1, 100, pv2_decode_heat(0.22), false });    // colder than 0.25, not by the margin 0.05
        pv2_add(h, 500, { 2, 100, pv2_decode_heat(0.10), false });    // colder by more than the margin
        pv2_hold(h, 0, 400); pv2_hold(h, 600, 1000);                  // no free byte left
        const pv2_pick p = pv2_pick_main(h, 400, 100, 100, pv2_victim_heat(0.25, 0.05), 1000);
        check("pick main: only a victim colder by the margin leaves", p.s == 500 && p.evict.size() == 1 && p.evict[0] == 500);
    }
    {   // c2 pick main: a free window anywhere in the heap first (the target's reach does not limit it), else colder victims within reach; brute force
        std::mt19937_64 rng(7);
        int seen[3] = { 0, 0, 0 };   // free window, over victims, none
        for (int it = 0; it < 3000; ++it) {
            pv2_heap h = pv2_make(0, 4000);
            if (rng() % 2) { pv2_hold(h, 2400, 3200); }   // a landing hole
            for (int k = 0; k < 40; ++k) {
                const size_t s = rng() % 4000 / 8 * 8, n = 40 + rng() % 300;
                if (s + n <= 4000) { pv2_add(h, s, { k, n, (int64_t) (rng() % 10), rng() % 8 == 0 }); }
            }
            const size_t  u = 8, n = 40 + rng() % 300, t = rng() % 4000, reach = rng() % 1200;
            const int64_t heat = rng() % 4 ? (int64_t) (rng() % 12) : std::numeric_limits<int64_t>::min();
            const pv2_pick p = pv2_pick_main(h, t, n, u, heat, reach);
            const want_t w = brute(h, t, n, u, std::numeric_limits<int64_t>::min(), h.hi - h.lo);
            const bool free_any = w.fe != PV2_NONE || w.fa != PV2_NONE;
            const bool ok = free_any || heat == std::numeric_limits<int64_t>::min() ? pick_ok(h, p, t, n, u, std::numeric_limits<int64_t>::min(), h.hi - h.lo, true)
                                                                                    : pick_ok(h, p, t, n, u, heat, reach, true);
            check("pick main: free window anywhere first, else colder victims within reach (brute force)", ok);
            seen[p.s == PV2_NONE ? 2 : p.evict.empty() ? 0 : 1]++;
        }
        printf("pick main: %d free windows, %d over victims, %d none\n", seen[0], seen[1], seen[2]);
        check("pick main: every outcome is exercised", seen[0] > 100 && seen[1] > 100 && seen[2] > 100);
    }
    {   // c2 views: every layer's views reach every start the heap can give an expert (a multiple of the unit up to span - n); the backend checks that the last
        // view's slice (ne[2] - 1) x u + bytes ends inside the buffer
        const poolx_slice c0[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 1024 } };   // OLMoE: Q4_K down
        const poolx_slice c1[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1720320, 210, 256, 1024 } };   // Q6_K down
        for (const poolx_slice * sl : { c0, c1 }) {
            const pv2_shape sh = pv2_shape_of(sl);
            const size_t B = poolx_block_class(sl).B, nblk = 700, span = nblk * B;   // the pool's experts: 700 blocks of the class
            const int64_t ne2 = pv2_heap_ne2(span, sh.n, sh.u);
            check("heap views: the last start's slices end inside the pool", (size_t) (ne2 - 1) * sh.u + sh.n <= span);
            check("heap views: no later start fits", (size_t) ne2 * sh.u + sh.n > span);
            check("heap views: at least as far as the class's own views", ne2 >= pv2_view_ne2((int64_t) nblk, B, sh.u));
        }
        check("heap views: a pool smaller than one expert has no view", pv2_heap_ne2(100, 101, 16) == 0 && pv2_heap_ne2(100, 100, 16) == 1);
    }
    {   // c2 boot fill (profile_preload on the heap), one shape and no hole: the hottest experts pack down from the heap's top, each start a multiple of its unit
        const poolx_slice c0[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 1024 } };   // OLMoE Q4_K down
        const pv2_shape sh = pv2_shape_of(c0);
        const size_t lo = 1000, hi = lo + 10 * sh.n + 77;   // room for 9 or 10 (the starts are aligned to the unit)
        std::vector<int64_t> heat(30);
        std::vector<size_t>  nb(30, sh.n);
        for (int i = 0; i < 30; ++i) { heat[i] = 100 - i; }
        pv2_heap h = pv2_make(lo, hi);
        const std::vector<size_t> d = pv2_heat_dist(heat, nb);
        std::vector<pv2_boot_t> ex;
        for (int32_t id : pv2_heat_order(heat)) { ex.push_back({ id, sh.n, sh.u, d[id], heat[id] }); }
        const std::vector<pv2_put_t> put = pv2_boot_fill(h, ex, hi, hi);
        bool packs = put.size() >= 9 && put.size() <= 10, hottest = !put.empty() && put[0].id == 0 && hi - (put[0].s + sh.n) < sh.u;
        for (size_t i = 0; i < put.size(); ++i) {
            packs = packs && put[i].id == (int32_t) i && put[i].s % sh.u == 0 && (i == 0 || (put[i].s < put[i - 1].s && put[i - 1].s - put[i].s - sh.n < sh.u));
        }
        check("boot fill: the hottest first, packed down from the top, nothing between them but the unit's slack", packs && hottest);
        check("boot fill: the colder ones that no longer fit are left out", put.size() < 30 && pv2_pick_main(h, 0, sh.n, sh.u, std::numeric_limits<int64_t>::min(), 0).s == PV2_NONE);
    }
    {   // c2 boot fill, OLMoE's two shapes (even layers Q4_K down, odd Q6_K down; 16 layers x 64 experts) in a heap that holds about 60% of them with the landing
        // hole in it: every expert goes, hottest first, into the free window nearest its heat target, until nothing fits
        const poolx_slice c0[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 1024 } };
        const poolx_slice c1[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1720320, 210, 256, 1024 } };
        const pv2_shape sh[2] = { pv2_shape_of(c0), pv2_shape_of(c1) };
        const int L = 16, E = 64;
        std::mt19937_64 rng(11);
        std::vector<int64_t> heat((size_t) L * E);
        std::vector<size_t>  nb((size_t) L * E), ub((size_t) L * E), shape((size_t) L * E);
        size_t total = 0;
        for (int i = 0; i < L * E; ++i) {
            shape[i] = (size_t) (i / E) % 2;
            nb[i] = sh[shape[i]].n; ub[i] = sh[shape[i]].u; total += nb[i];
            heat[i] = (int64_t) (1e9 / (1 + rng() % 200));   // a few hot experts, many cold, ties among them
        }
        const size_t lo = 1000, hi = lo + total * 6 / 10, M = lo + (hi - lo) / 2, H = M + (hi - lo) / 10;   // lo on no unit; the hole in the middle
        pv2_heap h = pv2_make(lo, hi);
        pv2_hold(h, M, H);
        const std::vector<size_t> d = pv2_heat_dist(heat, nb);
        std::vector<pv2_boot_t> ex;
        for (int32_t id : pv2_heat_order(heat)) { ex.push_back({ id, nb[id], ub[id], d[id], heat[id] }); }
        const std::vector<pv2_put_t> put = pv2_boot_fill(h, ex, M, H);
        std::map<int32_t, size_t> uh = { { PV2_HOLD, 1 } };
        for (int i = 0; i < L * E; ++i) { uh[i] = ub[i]; }
        check("boot fill: the heap is sound (no overlap, units, nothing lost)", heap_ok(h, uh));
        check("boot fill: a good part of the experts is placed, not all", put.size() > (size_t) L * E * 4 / 10 && put.size() < (size_t) L * E);
        bool clear = true, pinned = true;
        for (const auto & p : put) {
            clear  = clear && (p.s + nb[p.id] <= M || p.s >= H);
            pinned = pinned && h.res.count(p.s) && h.res.at(p.s).id == p.id && h.res.at(p.s).pin && h.res.at(p.s).heat == heat[p.id];
        }
        check("boot fill: nothing in the landing hole, and the hole is still held whole", clear && h.res.count(M) && h.res.at(M).id == PV2_HOLD && h.res.at(M).n == H - M);
        check("boot fill: each is a pinned resident (its upload is on its way) with its heat", pinned);
        check("boot fill: the hottest sits at the heap's top", h.hi - (put[0].s + nb[put[0].id]) < ub[put[0].id]);
        // nothing is left out that still fits, and per shape the placed ones are the hottest: a prefix of the shape's heat order
        std::set<int32_t> placed;
        for (const auto & p : put) { placed.insert(p.id); }
        bool fits = true, prefix = true, gone[2] = { false, false };
        for (const auto & x : ex) {
            if (placed.count(x.id)) { prefix = prefix && !gone[shape[x.id]]; continue; }
            gone[shape[x.id]] = true;
            fits = fits && pv2_pick_main(h, 0, x.n, x.u, std::numeric_limits<int64_t>::min(), 0).s == PV2_NONE;
        }
        check("boot fill: no expert left out would still fit a free window", fits);
        check("boot fill: per shape the placed experts are the hottest (a prefix of its heat order)", prefix);
    }
    {   // c2/c4: a stretch is taken from the heap (a graph's lend, the KV zone, a retype's staging): what each resident that reaches into it needs first. A queued
        // upload is dropped, a finished one not yet published is discarded, a published expert is evicted, a running one is awaited (one copy)
        std::map<int32_t, size_t> uh = { { PV2_HOLD, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 }, { 4, 1 }, { 5, 1 }, { 6, 1 }, { 7, 1 }, { 8, 1 } };
        std::map<int32_t, pv2_job_t> st = { { 1, PV2_CACHED }, { 2, PV2_CACHED }, { 3, PV2_QUEUED }, { 4, PV2_DONE }, { 5, PV2_RUNNING }, { 6, PV2_QUEUED }, { 7, PV2_CACHED }, { 8, PV2_CACHED } };
        pv2_heap h = pv2_make(0, 1500);
        pv2_add(h, 100, { 1, 100, 5, false });   // [100, 200) before the stretch [300, 800)
        pv2_add(h, 200, { 7, 100, 5, false });   // [200, 300) ends exactly where the stretch starts
        pv2_add(h, 300, { 2, 100, 5, false });   // [300, 400) from its first byte
        pv2_add(h, 420, { 3, 60, 5, true });     // [420, 480) queued, inside
        pv2_add(h, 500, { 4, 100, 5, true });    // [500, 600) finished, not published, inside
        pv2_add(h, 750, { 5, 100, 5, true });    // [750, 850) running, over the stretch's end
        pv2_add(h, 870, { 6, 30, 5, true });     // [870, 900) queued, after it
        pv2_add(h, 900, { 8, 100, 5, false });   // [900, 1000) after it
        const auto plan = [&](size_t a, size_t b) { return pv2_range_plan(h, a, b, [&](int32_t id) { return st.at(id); }); };
        auto r = plan(300, 800);
        check("range plan: a queued job is dropped", r.drop.size() == 1 && r.drop[0] == 3);
        check("range plan: a finished job is discarded", r.discard.size() == 1 && r.discard[0] == 4);
        check("range plan: a published expert is evicted", r.evict.size() == 1 && r.evict[0] == 2);
        check("range plan: a running upload is awaited, so the stretch is not ready", r.wait.size() == 1 && r.wait[0] == 5 && !r.ready());
        check("range plan: residents that only touch the stretch from outside are no part of it", plan(0, 300).wait.empty() && plan(0, 300).evict.size() == 2 && plan(800, 900).wait.size() == 1);
        auto leave = [&](const std::vector<int32_t> & ids) { for (int32_t id : ids) { for (const auto & [s, v] : h.res) { if (v.id == id) { pv2_drop(h, s); break; } } } };
        leave(r.drop); leave(r.discard); leave(r.evict);   // what the engine does before it waits
        check("range plan: with one still running the stretch cannot be held, and nothing changes", !pv2_hold(h, 300, 800) && heap_ok(h, uh) && plan(300, 800).wait.size() == 1);
        st[5] = PV2_CACHED;   // its copy landed and was published: an expert like the others
        r = plan(300, 800);
        check("range plan: after the copy only that expert is left to evict, nothing is awaited", r.ready() && r.evict.size() == 1 && r.evict[0] == 5 && r.drop.empty() && r.discard.empty());
        leave(r.evict);
        bool full = pv2_hold(h, 300, 800) && heap_ok(h, uh);
        for (const auto & [g, n] : h.gaps) { full = full && (g + n <= 300 || g >= 800); }
        check("range plan: the stretch is held whole: no resident and no free byte in it", full && pv2_overlap(h, 300, 800).empty());
        r = plan(0, 1500);   // the held stand-ins are no resident of a plan (the state function is never asked for them: st has no entry for them)
        check("range plan: the stand-ins of a held stretch are left alone", r.evict.size() == 3 && r.drop.size() == 1 && r.drop[0] == 6 && r.discard.empty() && r.ready());
        check("range plan: an empty stretch needs nothing", plan(1000, 1000).ready() && plan(1000, 1000).evict.empty() && plan(1000, 1000).drop.empty());
    }
    {   // the switch to decode: the hottest of residents and missing experts that fill the area stay or come in, cold leftovers leave
        pv2_heap h = pv2_make(0, 1000);
        pv2_add(h, 0,   { 1, 200, 5, false });
        pv2_add(h, 300, { 2, 200, 1, false });
        const std::vector<pv2_cand> miss = { { 3, 300, 1, 9 }, { 4, 300, 1, 3 }, { 5, 300, 1, 2 } };
        std::vector<pv2_res> left;
        const auto up = pv2_refill(h, miss, &left);
        check("refill: the hottest that fit come in", up.size() == 2 && up[0].id == 3 && up[1].id == 4);
        check("refill: a hot resident stays in place, a cold one leaves", h.res.count(0) && h.res.at(0).id == 1 && h.res.size() == 3);
        check("refill: the one that left is reported", left.size() == 1 && left[0].id == 2);
        std::map<int32_t, size_t> u1 = { { 1, 1 }, { 3, 1 }, { 4, 1 } };
        check("refill: heap invariants", heap_ok(h, u1));
        // an upload on its way (pinned) stays whatever its heat, and its bytes count against the area
        pv2_heap g = pv2_make(0, 1000);
        pv2_add(g, 0,   { 1, 400, 0, true });
        pv2_add(g, 500, { 2, 100, 0, false });
        left.clear();
        const auto up2 = pv2_refill(g, { { 3, 400, 1, 9 }, { 4, 400, 1, 8 } }, &left);
        check("refill: a resident in flight stays, the area is what it leaves", g.res.count(0) && g.res.at(0).id == 1 && up2.size() == 1 && up2[0].id == 3);
        check("refill: a cold resident not in flight leaves for a hotter newcomer", left.size() == 1 && left[0].id == 2);
    }
    {   // held stretches (a graph's lend, the KV zone): no placement crosses one, no refill drops one, they come back free
        std::map<int32_t, size_t> uh = { { PV2_HOLD, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } };
        pv2_heap h = pv2_make(0, 1000);
        pv2_add(h, 100, { 1, 100, 1, false });
        pv2_add(h, 400, { 2, 100, 1, false });
        const auto ov = pv2_overlap(h, 150, 450);
        check("overlap: the residents that reach into a stretch", ov.size() == 2 && ov[0] == 100 && ov[1] == 400);
        check("overlap: none outside", pv2_overlap(h, 200, 400).empty() && pv2_overlap(h, 500, 1000).empty());
        check("hold: refused while a resident is in it", !pv2_hold(h, 150, 450));
        check("hold: the heap is unchanged after a refusal", heap_ok(h, uh) && h.gaps.size() == 3);
        pv2_drop(h, 100);
        check("hold: a stretch with no resident", pv2_hold(h, 0, 300) && heap_ok(h, uh));
        check("hold: nothing free in it", h.gaps.lower_bound(0) == h.gaps.end() || h.gaps.begin()->first >= 300);
        check("place: never over a held stretch, however hot", pv2_place(h, 0, 250, 1, std::numeric_limits<int64_t>::max(), 1000, true).s >= 300 ||
                                                               pv2_place(h, 0, 250, 1, std::numeric_limits<int64_t>::max(), 1000, true).s == PV2_NONE);
        check("hold: a second stretch over the first holds the rest", pv2_hold(h, 200, 350) && heap_ok(h, uh) && (h.gaps.empty() || h.gaps.begin()->first >= 350));
        pv2_unhold(h, 100, 250);
        check("unhold: the middle comes back free, the ends stay held", heap_ok(h, uh) && h.gaps.count(100) && h.gaps.at(100) == 150 &&
                                                                       h.res.count(0) && h.res.at(0).id == PV2_HOLD && h.res.at(0).n == 100);
        std::vector<pv2_res> left;
        pv2_heap land = pv2_make(0, 300);
        pv2_hold(land, 0, 100);
        pv2_refill(land, { { 3, 300, 1, 9 } }, &left);
        check("refill: a held stretch stays", land.res.count(0) && land.res.at(0).id == PV2_HOLD && left.empty());
        pv2_unhold(h, 0, 1000);
        check("unhold: everything held comes back", heap_ok(h, uh) && h.res.size() == 1 && h.res.count(400));
    }
    {   // stage 2b, the heap as a shadow of the size classes: OLMoE's two classes (Q4_K down, Q6_K down) back to back, a recorded sequence of what the engine
        // does to their blocks (boot fill, a decode swap, a lend, the KV zone, a stream pass, the lend back); the shadow hears each expert's new place (the
        // host table entry) and each change of the held blocks, and must equal what the classes hold after every step
        const poolx_slice c0[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 1024 } };
        const poolx_slice c1[3] = { { 1179648, 144, 256, 2048 }, { 1179648, 144, 256, 2048 }, { 1720320, 210, 256, 1024 } };
        const size_t B[2] = { poolx_block_class(c0).B, poolx_block_class(c1).B }, n[2] = { pv2_shape_of(c0).n, pv2_shape_of(c1).n };
        const int32_t N[2] = { 6, 4 };
        const size_t base[2] = { 0, (size_t) N[0] * B[0] }, top = base[1] + (size_t) N[1] * B[1];
        std::vector<int32_t> o[2] = { std::vector<int32_t>(N[0], -1), std::vector<int32_t>(N[1], -1) };   // the classes: per block its expert, -1 free, -2 lent, -3 KV
        pv2_heap h = pv2_make(0, top);
        std::map<int32_t, size_t> where;
        size_t refused = 0;
        auto blocks = [&]() {   // what the engine hands the check: every block of every class
            std::vector<pv2_blk> v;
            for (int c = 0; c < 2; ++c) {
                for (int32_t b = 0; b < N[c]; ++b) { v.push_back({ base[c] + (size_t) b * B[c], B[c], o[c][(size_t) b] >= 0 ? o[c][(size_t) b] : -1, n[c], o[c][(size_t) b] <= -2 }); }
            }
            return v;
        };
        auto check_now = [&]() {
            std::vector<pv2_at> res;
            std::vector<std::pair<size_t, size_t>> u;
            pv2_expect(blocks(), res, u);
            return pv2_check(h, res, u);
        };
        auto held_now = [&]() {   // the engine at a lend, a release, a KV change: the held union from the blocks
            std::vector<pv2_at> res;
            std::vector<std::pair<size_t, size_t>> u;
            pv2_expect(blocks(), res, u);
            return pv2_held_set(h, u);
        };
        auto entry = [&](int c, int32_t b, int32_t id) {   // set_table_entry: expert id now in block b of class c (b -1: uncached)
            if (b >= 0) { o[c][(size_t) b] = id; }
            refused += !pv2_mirror(h, where, id, b >= 0 ? base[c] + (size_t) b * B[c] : PV2_NONE, n[c]);
        };
        auto evict = [&](int c, int32_t b) { const int32_t id = o[c][(size_t) b]; o[c][(size_t) b] = -1; entry(c, -1, id); };
        // boot fill (the hottest into the highest free blocks), each publish one entry
        entry(0, 5, 3); entry(0, 4, 7); entry(1, 3, 64 + 2); entry(1, 2, 64 + 9);
        check("shadow: boot fill", check_now() == 0 && refused == 0);
        evict(0, 4); entry(0, 4, 12);   // a decode swap: the victim's entry first, then the newcomer's publish
        check("shadow: a decode swap", check_now() == 0 && refused == 0);
        o[0][0] = o[0][1] = -2;        // a lend of two blocks (they were free), the KV zone over a block of the other class
        o[1][0] = -3;
        check("shadow: a lend not yet heard is a mismatch", check_now() > 0);
        check("shadow: the lend and the zone heard, no resident in them", held_now() == 0 && check_now() == 0);
        entry(0, 2, 5); entry(0, 3, 6);   // a stream pass: two experts into free blocks, one cached expert replaced
        evict(0, 5); entry(0, 5, 40);
        entry(1, 1, 64 + 33);
        check("shadow: a stream pass", check_now() == 0 && refused == 0);
        o[0][0] = o[0][1] = -1;        // the lend back (alias_release)
        check("shadow: the lend back", held_now() == 0 && check_now() == 0 && refused == 0);
        std::map<int32_t, size_t> uo = { { PV2_HOLD, 1 } };
        for (const auto & [id, s] : where) { uo[id] = 1; }
        check("shadow: heap invariants after the sequence", heap_ok(h, uo) && where.size() == 7);
        // what the check catches: a victim the shadow never heard of (the newcomer's place is not free), a lend over a cached expert, a lost evict
        const int32_t was = o[0][4];
        o[0][4] = 13;
        check("shadow: a missed evict refuses the newcomer's place", !pv2_mirror(h, where, 13, base[0] + 4 * B[0], n[0]) && check_now() > 0);
        o[0][4] = was;
        check("shadow: the state is unchanged by the refusal", check_now() == 0);
        o[1][3] = -2;   // a lend over block 3 of class 1, whose expert the shadow still holds
        check("shadow: a lend over a cached expert is counted", held_now() == 1 && check_now() > 0);
        o[1][3] = 64 + 2;
        check("shadow: back to the same", held_now() == 0 && check_now() == 0);
        o[0][2] = -1;   // the classes dropped an expert the shadow was not told of
        check("shadow: a lost evict is a mismatch", check_now() > 0);
    }
    stress(1, 4000, true);
    check("deterministic: one seed, one result", stress(7, 3000, false) == stress(7, 3000, false));
    check("deterministic: another seed, another result", stress(7, 3000, false) != stress(8, 3000, false));
    {   // negative control: the checks must fail on a broken heap and a broken pick, or they prove nothing
        pv2_heap h = pv2_make(0, 1000);
        std::map<int32_t, size_t> u1 = { { 1, 1 }, { 2, 1 }, { 3, 64 } };
        pv2_add(h, 100, { 1, 200, 9, false });
        pv2_add(h, 300, { 2, 100, 1, false });
        const bool good = heap_ok(h, u1);
        pv2_heap bad = h;
        bad.res[250] = { 3, 100, 9, false };                 // overlaps resident 1 and is not taken from the free map
        pv2_heap misaligned = h;
        pv2_add(misaligned, 500, { 3, 10, 9, false });       // 500 is no multiple of its unit 64
        pv2_pick wrong;                                      // evicts the hotter resident 1 for a newcomer of heat 5
        wrong.s = 100; wrong.evict = { 100 };
        const pv2_pick right = pv2_place(h, 100, 200, 1, 5, 1000, true);
        const bool caught = !heap_ok(bad, u1) && !heap_ok(misaligned, u1) && !pick_ok(h, wrong, 100, 200, 1, 5, 1000, true);
        check("negative control: the checks pass a sound heap and pick", good && pick_ok(h, right, 100, 200, 1, 5, 1000, true));
        check("negative control: the checks catch an overlap, a misaligned start and a hotter victim", caught);
        printf("negative control: overlap %s, misaligned start %s, hotter victim %s\n", heap_ok(bad, u1) ? "MISSED" : "caught",
               heap_ok(misaligned, u1) ? "MISSED" : "caught", pick_ok(h, wrong, 100, 200, 1, 5, 1000, true) ? "MISSED" : "caught");
    }
    move_tests();
    sort_tests();
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}

// ---- the simulator --------------------------------------------------------------------------------------------------------------------------------------
static const int    NL = 48, NE = 512;
static const size_t MiB = 1048576;

static bool read_heat(const std::string & path, std::vector<int64_t> & h, bool * settled) {
    std::ifstream f(path);
    std::string line;
    int got = 0;
    h.assign((size_t) NL * NE, 0);
    *settled = false;
    while (std::getline(f, line)) {
        if (line.rfind("# meta ", 0) == 0) { *settled = line.find("\"settled\": true") != std::string::npos; continue; }
        if (line.rfind("hot.", 0) != 0) { continue; }
        const int l = atoi(line.c_str() + 4);
        const size_t eq = line.find('=');
        if (l < 0 || l >= NL || eq == std::string::npos) { return false; }
        std::istringstream is(line.substr(eq + 1));
        for (int e = 0; e < NE; ++e) { if (!(is >> h[(size_t) l * NE + e])) { return false; } }
        ++got;
    }
    return got == NL;
}

// the map a release ships for the model (fork/run.py _hot_seeds): the settled prompt maps (all of them when none is), each layer's counts as shares of
// the layer, averaged over the maps, times 1e5
static std::vector<int64_t> shipped_map(const std::vector<int64_t> * maps, const bool * settled, int n) {
    std::vector<int> use;
    for (int k = 0; k < n; ++k) { if (settled[k]) { use.push_back(k); } }
    if (use.empty()) { for (int k = 0; k < n; ++k) { use.push_back(k); } }
    std::vector<int64_t> g((size_t) NL * NE, 0);
    for (int l = 0; l < NL; ++l) {
        std::vector<double> acc(NE, 0.0);
        for (int k : use) {
            double t = 0;
            for (int e = 0; e < NE; ++e) { t += (double) maps[k][(size_t) l * NE + e]; }
            for (int e = 0; e < NE; ++e) { acc[e] += (double) maps[k][(size_t) l * NE + e] / (t > 0 ? t : 1); }
        }
        for (int e = 0; e < NE; ++e) { g[(size_t) l * NE + e] = std::llround(acc[e] * 1e5 / (double) use.size()); }
    }
    return g;
}

struct cls_t { int blocks, layers; double mib; };
// "moe-cache: block pool on CUDA0: 48 layers, 20244.7 MiB, size classes: 2982 x 2.10 MiB (14 layers, grid of 5733), ..."
static bool read_pool(const std::string & path, std::vector<cls_t> & cls, double & pool_mib) {
    std::ifstream f(path);
    std::string line;
    const char * key = "moe-cache: block pool on CUDA0: ";
    while (std::getline(f, line)) {
        const size_t at = line.find(key);
        int nl = 0;
        if (at == std::string::npos || sscanf(line.c_str() + at + strlen(key), "%d layers, %lf MiB", &nl, &pool_mib) != 2) { continue; }
        const char * q = strstr(line.c_str(), "size classes: ");
        if (!q) { return false; }
        for (q += strlen("size classes: "); ; q += 3) {
            cls_t c;
            if (sscanf(q, "%d x %lf MiB (%d layers", &c.blocks, &c.mib, &c.layers) != 3) { break; }
            cls.push_back(c);
            q = strchr(q, ')');
            if (!q || strncmp(q, "), ", 3) != 0) { break; }
        }
        return !cls.empty();
    }
    return false;
}

struct lay_t { size_t n, u, today; const char * ug, * dn; };   // v2 bytes of an expert, its start unit, today's block bytes, its types

struct sim_t {
    std::vector<lay_t>  L;
    size_t H = 0, M = 0;                   // pool bytes; the main heap is [0, M), the landing area [M, H)
    size_t reach = 256 * MiB;              // a probe walks this far each way: ~120 experts, the heat order holds within ~1.3% of the heap
    int    ahead = 1;                      // layers the stream runs ahead of the one computing
    pv2_heap main, land;
    std::vector<int8_t> where;             // 0 nowhere, 1 main heap, 2 landing area
    std::vector<size_t> at;
    std::mt19937_64 rng;
    size_t n_of(int32_t id) const { return L[id / NE].n; }
    size_t u_of(int32_t id) const { return L[id / NE].u; }
};

struct row_t {
    size_t streamed[2] = {}, sbytes[2] = {}, inplace[2] = {};
    size_t s0 = 0, survive = 0, overflow = 0, peak_land = 0, refill = 0, rbytes = 0, free_b = 0, largest = 0, unusable = 0;
    double cov_before = 0, cov_after = 0, cov_ideal = 0;
};

static double coverage(const sim_t & S, const std::vector<int64_t> & h) {
    double in = 0, all = 0;
    for (size_t i = 0; i < h.size(); ++i) { all += (double) h[i]; if (S.where[i]) { in += (double) h[i]; } }
    return all > 0 ? in / all : 0;
}
static std::vector<pv2_cand> missing(const sim_t & S, const std::vector<int64_t> & h) {
    std::vector<pv2_cand> m;
    for (int32_t i = 0; i < (int32_t) h.size(); ++i) { if (!S.where[i] && h[i] > 0) { m.push_back({ i, S.n_of(i), S.u_of(i), h[i] }); } }
    std::sort(m.begin(), m.end(), [](const pv2_cand & a, const pv2_cand & b) { return a.heat != b.heat ? a.heat > b.heat : a.id < b.id; });
    return m;
}
// the landing area after a refill: where its residents are
static void land_index(sim_t & S) {
    for (auto & w : S.where) { if (w == 2) { w = 0; } }
    for (const auto & [s, r] : S.land.res) { S.where[r.id] = 2; S.at[r.id] = s; }
}
static std::vector<size_t> sizes(const sim_t & S) {
    std::vector<size_t> n((size_t) NL * NE);
    for (size_t i = 0; i < n.size(); ++i) { n[i] = S.n_of((int32_t) i); }
    return n;
}

// the decode state before the first prompt: the main heap packed in heat order, then the landing area refilled as at every switch to decode
static void fill(sim_t & S, const std::vector<int64_t> & h) {
    S.main = pv2_make(0, S.M);
    S.land = pv2_make(S.M, S.H);
    S.where.assign((size_t) NL * NE, 0);
    S.at.assign((size_t) NL * NE, 0);
    const auto dist = pv2_heat_dist(h, sizes(S));
    for (const auto & c : missing(S, h)) {
        const pv2_pick p = pv2_place(S.main, pv2_target(0, S.M, dist[c.id], c.n), c.n, c.u, c.heat, S.reach, true);
        if (p.s != PV2_NONE && pv2_commit(S.main, p, { c.id, c.n, c.heat, false }, nullptr)) { S.where[c.id] = 1; S.at[c.id] = p.s; }
    }
    pv2_refill(S.land, missing(S, h));
    land_index(S);
}

// one prompt of `passes` passes over all layers, every layer routing all but `skip` of its experts, then the switch to decode. hp: the heat the placement
// knows during the prompt; hd: the request's own decode heat, known at the switch: the refill and the coverage use it
static row_t prompt(sim_t & S, const std::vector<int64_t> & hp, const std::vector<int64_t> & hd, int passes = 2, int skip = 24) {
    row_t row;
    const auto dist = pv2_heat_dist(hp, sizes(S));
    for (auto & [s, r] : S.main.res) { r.heat = hp[r.id]; }
    for (auto & [s, r] : S.land.res) { r.heat = -1; r.pin = false; }   // the landing area stays free for the stream: what it holds is a leftover
    std::vector<char> was(S.where.size());
    for (size_t i = 0; i < was.size(); ++i) { was[i] = S.where[i] != 0; row.s0 += was[i]; }
    const auto dd = pv2_heat_dist(hd, sizes(S));
    for (size_t i = 0; i < hd.size(); ++i) { if (hd[i] > 0 && dd[i] + S.n_of((int32_t) i) <= S.H) { row.cov_ideal += (double) hd[i]; } }
    double all = 0;
    for (int64_t v : hd) { all += (double) v; }
    row.cov_ideal /= all;
    size_t cur = S.land.lo, land_live = 0;
    auto res_of = [&](int32_t id) -> pv2_res & { return (S.where[id] == 1 ? S.main : S.land).res.at(S.at[id]); };
    auto gone = [&](const std::vector<pv2_res> & out) { for (const auto & v : out) { S.where[v.id] = 0; } };
    for (int pass = 0; pass < passes; ++pass) {
        std::vector<std::vector<int32_t>> flight(NL);
        auto unpin = [&](int l) {
            for (int32_t id : flight[l]) {
                pv2_res & r = res_of(id);
                r.pin = false;
                if (S.where[id] == 2) { land_live -= r.n; }
            }
            flight[l].clear();
        };
        for (int l = 0; l < NL; ++l) {
            std::vector<int> ids(NE);
            std::iota(ids.begin(), ids.end(), 0);
            std::shuffle(ids.begin(), ids.end(), S.rng);
            std::vector<char> routed(NE, 1);
            for (int k = 0; k < skip; ++k) { routed[ids[k]] = 0; }
            for (int e = 0; e < NE; ++e) {   // residents first: used in place, held for the layer's flight
                const int32_t id = l * NE + e;
                if (!routed[e] || !S.where[id]) { continue; }
                pv2_res & r = res_of(id);
                r.pin = true;
                if (S.where[id] == 2) { land_live += r.n; }
                flight[l].push_back(id);
                row.inplace[pass]++;
            }
            for (int e = 0; e < NE; ++e) {
                const int32_t id = l * NE + e;
                if (!routed[e] || S.where[id]) { continue; }
                const size_t n = S.n_of(id), u = S.u_of(id);
                row.streamed[pass]++;
                row.sbytes[pass] += n;
                std::vector<pv2_res> out;
                pv2_pick p = pv2_place(S.main, pv2_target(0, S.M, dist[id], n), n, u, hp[id], S.reach, true);
                if (p.s != PV2_NONE) {
                    pv2_commit(S.main, p, { id, n, hp[id], true }, &out);
                    gone(out);
                    S.where[id] = 1; S.at[id] = p.s;
                } else {
                    p = pv2_place(S.land, cur, n, u, hp[id], S.H - S.M, false);   // the ring: nearest its cursor
                    if (p.s == PV2_NONE) { row.overflow++; continue; }   // the stream would wait for the layer computing
                    pv2_commit(S.land, p, { id, n, -1, true }, &out);
                    gone(out);
                    S.where[id] = 2; S.at[id] = p.s;
                    land_live += n;
                    cur = p.s + n < S.H ? p.s + n : S.land.lo;
                }
                flight[l].push_back(id);
            }
            row.peak_land = std::max(row.peak_land, land_live);
            if (l >= S.ahead) { unpin(l - S.ahead); }   // the layer computing finishes as the next one is placed
        }
        for (int l = 0; l < NL; ++l) { unpin(l); }
    }
    for (size_t i = 0; i < was.size(); ++i) { row.survive += was[i] && S.where[i]; }
    row.cov_before = coverage(S, hd);
    for (auto & [s, r] : S.land.res) { r.heat = hd[r.id]; }
    for (const auto & c : pv2_refill(S.land, missing(S, hd))) { row.refill++; row.rbytes += c.n; }
    land_index(S);
    row.cov_after = coverage(S, hd);
    for (const auto & [s, g] : S.main.gaps) {
        row.free_b += g;
        row.largest = std::max(row.largest, g);
        bool fits = false;
        for (const auto & ly : S.L) { fits = fits || pv2_fit(s, s + g, ly.n, ly.u, s) != PV2_NONE; }
        if (!fits) { row.unusable += g; }
    }
    std::map<int32_t, size_t> unit_of;
    for (int32_t i = 0; i < NL * NE; ++i) { unit_of[i] = S.u_of(i); }
    check("sim: main heap invariants", heap_ok(S.main, unit_of));
    check("sim: landing area invariants", heap_ok(S.land, unit_of));
    return row;
}

static void print_head() {
    printf("%-3s %-5s %6s %7s %6s %7s %6s %6s %6s %5s %7s %6s %7s %7s %7s %6s %6s %6s %7s %7s %7s\n", "#", "heat", "p1_exp", "p1_GB", "p2_exp", "p2_GB",
           "pre_exp", "kept", "lost", "ovfl", "land_GB", "refill", "ref_GB", "ms@7", "ms@15", "cov0_%", "cov1_%", "ideal%", "free_MiB", "big_MiB", "unus_MiB");
}
static void print_row(int i, const char * name, const row_t & r) {
    printf("%-3d %-5s %6zu %7.2f %6zu %7.2f %7zu %6zu %6zu %5zu %7.2f %6zu %6.2f %7.0f %7.0f %6.1f %6.1f %6.1f %8.0f %7.1f %8.1f\n", i, name,
           r.streamed[0], r.sbytes[0] / 1e9, r.streamed[1], r.sbytes[1] / 1e9, r.s0, r.survive, r.s0 - r.survive, r.overflow, r.peak_land / 1e9,
           r.refill, r.rbytes / 1e9, r.rbytes / 7e9 * 1e3, r.rbytes / 15e9 * 1e3, 100 * r.cov_before, 100 * r.cov_after, 100 * r.cov_ideal,
           r.free_b / (double) MiB, r.largest / (double) MiB, r.unusable / (double) MiB);
}

static int run_sim(const std::string & dir, const std::string & log, uint64_t seed, const std::string & mode) {
    const char * names[4] = { "pf1k", "pf4k", "pf12k", "pf14k" };
    std::vector<int64_t> heat[4];
    bool settled[4];
    for (int k = 0; k < 4; ++k) {
        if (!read_heat(dir + "/" + names[k] + ".ini", heat[k], &settled[k])) { fprintf(stderr, "sim: cannot read %s/%s.ini\n", dir.c_str(), names[k]); return 1; }
    }
    if (mode != "own" && mode != "prev" && mode != "global") { fprintf(stderr, "sim: --heat=%s: own, prev or global\n", mode.c_str()); return 1; }
    const std::vector<int64_t> shipped = shipped_map(heat, settled, 4);
    // the heat the placement knows during prompt i of the 50 (the first one's previous request made the pf14k start state)
    auto known = [&](int i) -> const std::vector<int64_t> & { return mode == "global" ? shipped : mode == "prev" ? heat[i == 0 ? 3 : (i - 1) % 4] : heat[i % 4]; };
    std::string used;
    for (int k = 0; k < 4; ++k) { if (settled[k]) { used += (used.empty() ? "" : ", ") + std::string(names[k]); } }
    printf("placement heat during a prompt: %s%s\n", mode.c_str(), mode == "global" ? (" (the shipped map: the settled " + used + " merged)").c_str() : "");
    std::vector<cls_t> cls;
    double pool_mib = 0;
    if (!read_pool(log, cls, pool_mib)) { fprintf(stderr, "sim: no block pool line in %s\n", log.c_str()); return 1; }
    // a class's block (MiB in the log) names its types: the model's types (the log's bell-pool lines: iq2_s, iq3_xxs, iq3_s, iq4_xs rows of 2560 for up and
    // gate, iq4_nl and q2_0 rows of 640 for down) through today's block rule; exactly one pair must make each class
    struct qt { const char * name; size_t ts, blck; };
    static const qt ug[] = { { "iq2_s", 82, 256 }, { "iq3_xxs", 98, 256 }, { "iq3_s", 110, 256 }, { "iq4_xs", 136, 256 } };
    static const qt dn[] = { { "iq4_nl", 18, 32 }, { "q2_0", 18, 64 } };
    const size_t n_embd = 2560, n_ff = 640;
    std::vector<lay_t> per_cls;
    size_t today_pool = 0;
    for (const auto & c : cls) {
        int hits = 0;
        lay_t ly{};
        for (const auto & a : ug) {
            for (const auto & d : dn) {
                const size_t ugb = n_ff * (n_embd / a.blck) * a.ts, dnb = n_embd * (n_ff / d.blck) * d.ts;
                const poolx_slice sl[3] = { { ugb, a.ts, a.blck, n_embd }, { ugb, a.ts, a.blck, n_embd }, { dnb, d.ts, d.blck, n_ff } };
                const poolx_block pb = poolx_block_class(sl);
                if (std::llround(pb.B / (double) MiB * 100) != std::llround(c.mib * 100)) { continue; }
                const pv2_shape sh = pv2_shape_of(sl);
                ly.n = sh.n; ly.u = sh.u;
                ly.today = pb.B; ly.ug = a.name; ly.dn = d.name;
                hits++;
            }
        }
        if (hits != 1) { fprintf(stderr, "sim: class of %.2f MiB matches %d type pairs\n", c.mib, hits); return 1; }
        per_cls.push_back(ly);
        today_pool += (size_t) c.blocks * ly.today;
    }
    // layer -> class: the pairs the q5h logs of this pool name ("cstream: pass 1 entry 1 (layer 1, card 0): no run of 512 blocks in class 1"); the other
    // layers, ascending, take the classes' remaining layer counts in class order (a stated guess: the line counts layers per class, not which ones)
    static const int seen[][2] = { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 4, 2 }, { 5, 2 }, { 6, 2 }, { 8, 1 }, { 9, 2 }, { 11, 2 } };
    std::vector<int> lc(NL, -1), left(cls.size());
    for (size_t c = 0; c < cls.size(); ++c) { left[c] = cls[c].layers; }
    for (const auto & s : seen) { lc[s[0]] = s[1]; left[s[1]]--; }
    for (int l = 0, c = 0; l < NL; ++l) {
        if (lc[l] >= 0) { continue; }
        while (c < (int) cls.size() && left[c] == 0) { ++c; }
        if (c == (int) cls.size()) { fprintf(stderr, "sim: the classes count fewer than %d layers\n", NL); return 1; }
        lc[l] = c; left[c]--;
    }
    sim_t S;
    for (int l = 0; l < NL; ++l) { S.L.push_back(per_cls[lc[l]]); }
    S.H = today_pool;
    printf("pool: %.1f MiB in the log, %.1f MiB from the classes' types; %zu classes:\n", pool_mib, today_pool / (double) MiB, cls.size());
    for (size_t c = 0; c < cls.size(); ++c) {
        std::string ls;
        for (int l = 0; l < NL; ++l) { if (lc[l] == (int) c) { ls += (ls.empty() ? "" : ",") + std::to_string(l); } }
        printf("  %4d x %.2f MiB  %-7s/%-6s today %8zu B, v2 %8zu B (%+.2f%%), unit %6zu B, layers %s\n", cls[c].blocks, cls[c].mib, per_cls[c].ug, per_cls[c].dn,
               per_cls[c].today, per_cls[c].n, 100.0 * ((double) per_cls[c].n / per_cls[c].today - 1), per_cls[c].u, ls.c_str());
    }
    printf("today (same prompt, class pool): 23421 experts streamed a pass, ~6300 experts (~9 GB) refilled after the prompt\n");
    for (int ahead = 1; ahead <= 2; ++ahead) {
        // the landing area: what the first prompt has in flight there when the main heap leaves room for (ahead + 1) whole layers, plus one expert
        size_t full = 0, big = 0;
        for (const auto & ly : S.L) { full = std::max(full, NE * ly.n + ly.u); big = std::max(big, ly.n + ly.u); }
        S.ahead = ahead;
        S.M = (S.H - (ahead + 1) * full) / MiB * MiB;
        S.rng.seed(seed);
        fill(S, heat[3]);
        const size_t need = prompt(S, heat[3], heat[3]).peak_land;   // sized on the request's own heat in every mode
        S.M = (S.H - need - big) / MiB * MiB;
        printf("\nstream %d layer(s) ahead: landing area %.0f MiB (in flight at most %.0f MiB with %.0f MiB for it), main heap %.0f MiB\n", ahead,
               (S.H - S.M) / (double) MiB, need / (double) MiB, (ahead + 1) * full / (double) MiB, S.M / (double) MiB);
        S.rng.seed(seed);
        fill(S, heat[3]);
        printf("decode state before the prompt: %zu experts resident, coverage of pf14k %.1f%%\n", (size_t) std::count_if(S.where.begin(), S.where.end(),
               [](int8_t w) { return w != 0; }), 100 * coverage(S, heat[3]));
        print_head();
        print_row(0, "pf14k", prompt(S, mode == "global" ? shipped : heat[3], heat[3]));
        printf("50 prompts, each with the next request's decode heat (pf1k, pf4k, pf12k, pf14k in turn), from the pf14k state:\n");
        S.rng.seed(seed);
        fill(S, heat[3]);
        print_head();
        for (int i = 0; i < 50; ++i) { print_row(i + 1, names[i % 4], prompt(S, known(i), heat[i % 4])); }
    }
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail ? 1 : 0;
}

static void test_upchunks() {
    std::vector<pv2_up_t> ex; for (int i = 0; i < 50; ++i) { ex.push_back({ i, (size_t) (3 + i % 4) * 100, (size_t) (50 - i) * 1000 }); }
    auto ch = pv2_upchunks(ex, 2000);
    size_t n = 0, tot = 0, want = 0; bool ok = true;
    for (const auto & x : ex) { want += x.bytes; }
    for (const auto & c : ch) {
        ok = ok && c.bytes <= 2000 && !c.mv.empty(); size_t o = 0;
        for (const auto & m : c.mv) { ok = ok && m.src == o && m.dst == (size_t) (50 - m.id) * 1000; o += m.bytes; ++n; tot += m.bytes; }
        ok = ok && o == c.bytes;
    }
    check("upchunks: every expert once, packed, within chunk", ok && n == ex.size() && tot == want);
    check("upchunks: chunks are full (next expert did not fit)", ch.size() > 1 && ch[0].bytes + ex[ch[0].mv.size()].bytes > 2000);
    check("upchunks: empty and oversize skipped", pv2_upchunks({ {1, 0, 0}, {2, 5000, 0} }, 2000).empty());
}
static void test_upring() {
    auto r = pv2_upring(20, 8); bool ok = r.size() == 20;
    for (const auto & o : r) { ok = ok && o.slot == o.chunk % 8 && o.wait_scatter == (o.chunk >= 8 ? (int64_t) o.chunk - 8 : -1); }
    check("upring: slot = chunk mod slots, waits only on the scatter 'slots' chunks back", ok);
    check("upring: the first 'slots' uploads wait for nothing", pv2_upring(3, 8)[2].wait_scatter == -1);
    check("upring: no slots, no ops", pv2_upring(5, 0).empty());
}
int main(int argc, char ** argv) {
    test_upchunks();
    test_upring();
    if (argc > 1 && strcmp(argv[1], "--sim") == 0) {
        std::vector<std::string> pos;
        std::string mode = "own";
        for (int i = 2; i < argc; ++i) { if (strncmp(argv[i], "--heat=", 7) == 0) { mode = argv[i] + 7; } else { pos.push_back(argv[i]); } }
        const std::string dir = pos.size() > 0 ? pos[0] : "/p/bw/data/hot/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf";
        const std::string log = pos.size() > 1 ? pos[1] : "/p/bw/data/q5h/ring_early_switch-q1-p14k-re0.server.log";
        return run_sim(dir, log, pos.size() > 2 ? strtoull(pos[2].c_str(), nullptr, 10) : 1, mode);
    }
    return unit_tests();
}
