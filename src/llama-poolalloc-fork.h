// The pure part of the expert pool buffer, as templates over the pool types (anything with the fields named below) so that tests/test-pool-fork.cpp can drive them with
// random pools, many threads and a shadow memory, with no GPU and no model. llama-moecache-fork.cpp includes this and calls these (it keeps no second copy of any rule).
//   block sizing        poolx_block_class     : bytes of one expert block, slice offsets, the read-past guard
//   lend granularity    poolx_lend_round      : where a lend of n bytes really ends (whole blocks of every class it overlaps)
//                       poolx_lend_blocks     : blocks of one class a lend takes (never more than the class has; flags the zero block)
//   owner ledger        poolx_lend / poolx_release / poolx_zone_take / poolx_zone_give / poolx_claim_free: the block owner table + free stack
//                       owner: >= 0 layer, -1 free, -2 lent to a graph, -3 under the KV zone
//   region map          poolx_regions / poolx_gate_kind : where each region lies now, and the gate every expert write passes
//   range use counts    poolx_ranges          : who touches which bytes (DMA, graph, zone move), movers wait for users, entrants wait for movers
// Bugs these rules came from: fork/docs/bug-log.md (the tests name the entry they guard).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <vector>

static inline size_t poolx_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// ---- block sizing ----------------------------------------------------------------------------------------------------------------------------
// One slice (up, gate or down) of an expert: its bytes (nb2), the type's block bytes (ts) and values per block (blck), values per row (ne0).
struct poolx_slice { size_t nb2, ts, blck, ne0; };
struct poolx_block { size_t B = 0, off[3] = { 0, 0, 0 }, len[3] = { 0, 0, 0 }, past[3] = { 0, 0, 0 }; };
// A block holds the three slices one after the other, each at a multiple of 128 (the CUDA buffer alignment); a quantized matmul reads whole 512-wide tiles of a row, so for rows
// that are not a multiple of 512 values it reads past the slice end: the block keeps that read (the largest) behind the down slice. B is a multiple of the least common
// multiple of the type sizes (the kernels take the expert stride as nb[2] / type size).
static inline poolx_block poolx_block_class(const poolx_slice s[3]) {
    poolx_block c;
    size_t off = 0, q = 128, guard = 0;
    for (int k = 0; k < 3; ++k) {
        c.off[k] = off;
        c.len[k] = s[k].nb2;
        off += poolx_up(s[k].nb2, 128);
        q = q / std::gcd(q, s[k].ts) * s[k].ts;
        if (s[k].ne0 % 512 != 0) {
            c.past[k] = (512 - s[k].ne0 % 512) / s[k].blck * s[k].ts;   // ggml_row_size(type, 512 - ne0 % 512)
            guard = std::max(guard, c.past[k]);
        }
    }
    c.B = poolx_up(off + guard, q);
    return c;
}

// ---- lend granularity -------------------------------------------------------------------------------------------------------------------------
// where a lend of `bytes` from the start of the buffer really ends: it takes whole blocks of every size class its range overlaps (the classes are stacked in the buffer, each
// with its own block size), so the end is the furthest of base + ceil((bytes - base) / B) * B over those classes. Rounding by the LARGEST block size is wrong when a smaller-
// block class sits at the start (MiMo: class 0 has 9.2 MB blocks, the largest class 12.6 MB sits at the end: a 10.8 MB lend ended at 18.3 MB, not 12.6 MB)
template <class P> static size_t poolx_lend_round(const P & ap, size_t bytes) {
    size_t end = bytes;
    for (const auto & sg : ap.segs) {
        if (sg.base >= bytes || sg.B == 0) { continue; }
        end = std::max(end, (size_t) sg.base + (bytes - sg.base + sg.B - 1) / sg.B * sg.B);
    }
    return end;
}
// the blocks of one class a lend of need_b bytes covers (never more than the class's n expert blocks: the class's zero block is no lendable block), and whether the range
// reaches the zero block (it must be zeroed again when the lend leaves it)
template <class S> static int32_t poolx_lend_blocks(const S & sg, size_t need_b, bool * reaches_zero) {
    const size_t cov = need_b > sg.base ? need_b - sg.base : 0;
    if (reaches_zero && cov > (size_t) sg.n * sg.B) { *reaches_zero = true; }
    return sg.B == 0 ? 0 : (int32_t) std::min<size_t>((size_t) sg.n, (cov + sg.B - 1) / sg.B);
}

// ---- owner ledger -----------------------------------------------------------------------------------------------------------------------------
// ps: { std::vector<int32_t> owner, free; }. The caller holds the cache's mutex. evict(layer, block) takes an expert out of a block it owned (the caller's bookkeeping).
// Lend blocks [from, to) of a class to a graph. Returns the blocks that were under the KV zone (-3): they are NOT taken (a lend never goes over the zone; the caller reports).
template <class PS, class F> static int poolx_lend(PS & ps, int32_t from, int32_t to, F evict) {
    int bad = 0;
    for (int32_t b = from; b < to; ++b) {
        const int32_t o = ps.owner[b];
        if (o == -3) { bad++; continue; }
        if (o >= 0) { evict(o, b); }
        else if (o == -1) {
            auto it = std::find(ps.free.begin(), ps.free.end(), b);
            if (it != ps.free.end()) { ps.free.erase(it); }
        }
        ps.owner[b] = -2;
    }
    return bad;
}
// Blocks [keep, blocked) go back to the experts (only the lent ones). Returns how many.
template <class PS> static size_t poolx_release(PS & ps, int32_t keep, int32_t blocked) {
    size_t n = 0;
    for (int32_t b = keep; b < blocked; ++b) {
        if (ps.owner[b] == -2) { ps.owner[b] = -1; ps.free.push_back(b); n++; }
    }
    return n;
}
// The KV zone takes `take` (ascending block numbers): experts out, free blocks off the free list, lent blocks marked (*blocked drops to the lowest: the lend of this graph starts
// over below the KV). The owner becomes -3.
template <class PS, class F> static void poolx_zone_take(PS & ps, const std::vector<int32_t> & take, int32_t * blocked, F evict) {
    std::vector<char> free_mark;
    for (int32_t b : take) {
        const int32_t o = ps.owner[b];
        if (o >= 0) { evict(o, b); }
        else if (o == -1) {
            if (free_mark.empty()) { free_mark.assign(ps.owner.size(), 0); }
            free_mark[b] = 1;
        } else if (o == -2 && blocked) { *blocked = std::min(*blocked, b); }
        ps.owner[b] = -3;
    }
    if (!free_mark.empty()) {
        ps.free.erase(std::remove_if(ps.free.begin(), ps.free.end(), [&](int32_t b) { return free_mark[b] != 0; }), ps.free.end());
    }
}
// Blocks the zone left go back to the free list.
template <class PS> static size_t poolx_zone_give(PS & ps, const std::vector<int32_t> & give) {
    size_t n = 0;
    for (int32_t b : give) { if (ps.owner[b] != -3) { continue; } ps.owner[b] = -1; ps.free.push_back(b); n++; }   // only a block under the zone goes back: a lent or owned one is not free
    return n;
}
// An expert takes a free block for layer li: the block popped off the free stack, or -1. A block that is not free (stale entry) is dropped, never handed out.
template <class PS> static int32_t poolx_claim_free(PS & ps, int32_t li) {
    while (!ps.free.empty()) {
        const int32_t b = ps.free.back(); ps.free.pop_back();
        if (ps.owner[b] == -1) { ps.owner[b] = li; return b; }
    }
    return -1;
}

// ---- region map and gate ----------------------------------------------------------------------------------------------------------------------
template <class P> static size_t poolx_kv_end(const P & ap) { return ap.kv_h > 0 ? ap.kv_start + ap.kv_fixed + ap.kv_h * ap.kv_row : ap.kv_start; }
struct poolx_region { const char * kind; int32_t pcls; size_t lo, hi; };
template <class P> static std::vector<poolx_region> poolx_regions(const P & ap) {
    std::vector<poolx_region> m;
    if (ap.want_b > 0) { m.push_back({ "lend", -1, 0, ap.want_b }); }
    if (ap.kv_h > 0)   { m.push_back({ "zone", -1, ap.kv_start, poolx_kv_end(ap) }); }
    for (const auto & sg : ap.segs) {
        m.push_back({ "blocks", sg.pcls, sg.base, sg.base + (size_t) sg.n * sg.B });
        m.push_back({ "zero",   sg.pcls, sg.base + (size_t) sg.n * sg.B, sg.base + (size_t) (sg.n + 1) * sg.B });
    }
    return m;
}
// The gate every write into an expert block consults first: nullptr when [lo, hi) is clear, else "zone" or "zero" (and the region in *rlo / *rhi). No allocation.
template <class P> static const char * poolx_gate_kind(const P & ap, size_t lo, size_t hi, size_t * rlo, size_t * rhi) {
    if (ap.kv_h > 0 && lo < poolx_kv_end(ap) && hi > ap.kv_start) { *rlo = ap.kv_start; *rhi = poolx_kv_end(ap); return "zone"; }
    for (const auto & sg : ap.segs) {
        const size_t z0 = sg.base + (size_t) sg.n * sg.B, z1 = z0 + sg.B;
        if (lo < z1 && hi > z0) { *rlo = z0; *rhi = z1; return "zero"; }
    }
    return nullptr;
}

// ---- range use counts -------------------------------------------------------------------------------------------------------------------------
// Use counts per byte range (replaces a lock per buffer): everyone who touches a range of a card's pool - an upload DMA (w), a keep-hot copy (w), a graph reading the lend -
// enters it first and leaves it when its DMA has finished. A mover (zone resize + row move) first announces its range, which holds new entrants back, then waits until no
// entered user overlaps it, moves, and ends the move. So a move never starts under a running DMA and no write starts under a running move. K identifies the pool (its address).
// Fixed tables, no heap at run time: at most one entry per upload worker plus the decode thread (use), one per mover (moving).
template <class K> struct poolx_ranges {
    static constexpr int USE_MAX = 256, MOVE_MAX = 8;
    struct ent { K ap; size_t lo, hi; bool w; };
    struct tab { ent e[USE_MAX]; int n = 0; void drop(int i) { e[i] = e[--n]; } };
    std::mutex mu; std::condition_variable cv;
    tab use, moving;
    // a stall is a wait that really blocked (the predicate was false at the first look)
    std::atomic<int64_t> st_enter_n{0}, st_enter_us{0}, st_enter_max{0}, st_move_n{0}, st_move_us{0}, st_calls{0};
    static bool ovl(const ent & u, K ap, size_t lo, size_t hi) { return u.ap == ap && lo < u.hi && hi > u.lo; }
    static int64_t now_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    bool clear_of_moves(K ap, size_t lo, size_t hi) const { for (int i = 0; i < moving.n; ++i) { if (ovl(moving.e[i], ap, lo, hi)) { return false; } } return true; }
    bool clear_of_use(K ap, size_t lo, size_t hi) const { for (int i = 0; i < use.n; ++i) { if (ovl(use.e[i], ap, lo, hi)) { return false; } } return true; }
    void enter(K ap, size_t lo, size_t hi, bool w) {
        std::unique_lock<std::mutex> lk(mu);
        st_calls++;
        if (!clear_of_moves(ap, lo, hi)) {
            const int64_t t0 = now_us();
            cv.wait(lk, [&]() { return clear_of_moves(ap, lo, hi); });
            const int64_t d = now_us() - t0;
            st_enter_n++; st_enter_us += d;
            int64_t m = st_enter_max.load(); while (d > m && !st_enter_max.compare_exchange_weak(m, d)) {}
        }
        add(use, { ap, lo, hi, w }, USE_MAX);
    }
    // never waits: a worker blocked here could be the very upload a mover waits for (deadlock). false: inside a move range, the job is skipped.
    bool try_enter(K ap, size_t lo, size_t hi, bool w) {
        std::lock_guard<std::mutex> lk(mu);
        st_calls++;
        if (!clear_of_moves(ap, lo, hi)) { st_enter_n++; return false; }
        add(use, { ap, lo, hi, w }, USE_MAX);
        return true;
    }
    void leave(K ap, size_t lo, size_t hi, bool w) {
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int i = 0; i < use.n; ++i) { if (use.e[i].ap == ap && use.e[i].lo == lo && use.e[i].hi == hi && use.e[i].w == w) { use.drop(i); break; } }
        }
        cv.notify_all();
    }
    // false: users of the range were still inside after timeout_s (the move goes on regardless, the caller logs it)
    bool move_begin(K ap, size_t lo, size_t hi, int timeout_s = 10) {
        std::unique_lock<std::mutex> lk(mu);
        add(moving, { ap, lo, hi, true }, MOVE_MAX);
        const int64_t t0 = now_us();
        auto idle = [&]() { return clear_of_use(ap, lo, hi); };
        const bool ok = idle() || cv.wait_for(lk, std::chrono::seconds(timeout_s), idle);
        if (now_us() - t0 > 50) { st_move_n++; st_move_us += now_us() - t0; }
        return ok;
    }
    void move_end(K ap) {
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int i = 0; i < moving.n; ++i) { if (moving.e[i].ap == ap) { moving.drop(i); break; } }
        }
        cv.notify_all();
    }
private:
    static void add(tab & t, const ent & u, int cap) {
        if (t.n >= cap) { std::abort(); }   // "pool range table full"
        t.e[t.n++] = u;
    }
};

// ---- the hot cache: which block an expert may take ----------------------------------------------------------------------------------------------
// A free block (the highest one: prompts borrow the pool from its start, so the experts placed best end at the back), else a block whose cached expert has the lowest
// rank >= 0 (< 0: may not be evicted). Never a block lent to a graph (-2), under the KV zone (-3), with an upload in flight or a streamed expert (busy). Scans only the
// first n_active blocks (the rest lies under the KV rows). best: the rank of the victim, 0 for a free or empty block.
// expert(o, b): the expert layer o caches in block b (< 0: owned and empty); rank(o, e).
struct poolx_pick { int32_t slot = -1, owner = -1; };
template <class PS, class Busy, class Expert, class Rank>
static poolx_pick poolx_pick_slot(const PS & ps, int32_t n_active, Busy busy, Expert expert, Rank rank, double & best) {
    best = 1e300;
    poolx_pick p;
    int32_t top = -1;   // the highest block that is really free: an entry of the free stack can be stale (the block was lent or went under the zone since), never picked
    for (int32_t b : ps.free) { if (b > top && b < (int32_t) ps.owner.size() && ps.owner[b] == -1) { top = b; } }
    if (top >= 0) { p.slot = top; best = 0; return p; }
    for (int32_t b = 0; b < n_active; ++b) {
        const int32_t o = ps.owner[b];
        if (o == -2 || o == -3) { continue; }
        if (o < 0) { p.slot = b; best = 0; return p; }
        if (busy(o, b)) { continue; }
        const int32_t e = expert(o, b);
        if (e < 0) { p.slot = b; best = 0; return p; }
        const double r = rank(o, e);
        if (r >= 0 && r < best) { best = r; p.slot = b; p.owner = o; }
    }
    return p;
}
