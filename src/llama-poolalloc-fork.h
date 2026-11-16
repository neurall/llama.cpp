// The pure part of the expert pool buffer, as templates over the pool types (anything with the fields named below) so that tests/test-pool-fork.cpp can drive them with
// random pools, many threads and a shadow memory, with no GPU and no model. llama-moecache-fork.cpp includes this and calls these (it keeps no second copy of any rule).
//   block sizing        poolx_block_class     : bytes of one expert block, slice offsets, the read-past guard
//   lend granularity    poolx_lend_round      : where a lend of n bytes really ends (whole blocks of every class it overlaps)
//                       poolx_lend_blocks     : blocks of one class a lend takes (never more than the class has)
//   owner ledger        poolx_lend / poolx_release / poolx_zone_take / poolx_zone_give / poolx_claim_free: the block owner table + free stack
//                       owner: >= 0 layer, -1 free, -2 lent to a graph, -3 under the KV zone, -4 another class's bytes (outside [home_lo, home_hi)),
//                       -5 CUDA's room (the tail area's blocks handed to CUDA's own memory, poolx_room_*)
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
#include <thread>
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
// where a lend of `bytes` from the start of the buffer really ends: at the end of the block it ends in, in the class that block belongs to (the classes lie back to
// back, each with its own block size; the ones before it are covered whole); a lend that ends on a block edge or past the blocks ends where it ends. Every edge of
// the pool layout (the lend's end, the KV zone's start) uses this rule. Rounding by the LARGEST block size is wrong when a smaller-block class sits at the start
// (MiMo: class 0 has 9.2 MB blocks, the largest class 12.6 MB sits at the end: a 10.8 MB lend ended at 18.3 MB, not 12.6 MB)
// A run of n experts e..e+n-1 of a tensor with ne2 experts, copied into consecutive blocks: the bytes past each slice are the next expert's first bytes, so the rows whose
// next expert exists go up as one strided copy (source pitch the slice, destination pitch the block); the tensor's last expert, when in the run, reads zeros past it.
static inline size_t poolx_past_rows(int64_t e, size_t n, int64_t ne2) { return n == 0 ? 0 : e + (int64_t) n < ne2 ? n : n - 1; }

// staging=0 ring: the first block i, searched from the cursor `from` on and wrapping to the class start, that begins n usable blocks in a row (ok[i..i+n-1]); a run
// never wraps the class end (one copy per tensor needs one run of addresses). -1: no run that long, the caller picks blocks one by one.
static inline int32_t poolx_first_window(const std::vector<uint8_t> & ok, int32_t from, int32_t n) {
    const int32_t N = (int32_t) ok.size();
    if (n <= 0 || n > N) { return -1; }
    std::vector<int32_t> run(N + 1, 0);
    for (int32_t i = N - 1; i >= 0; --i) { run[i] = ok[i] ? run[i + 1] + 1 : 0; }
    for (int32_t k = 0; k < N; ++k) { const int32_t i = (from + k) % N; if (run[i] >= n) { return i; } }
    return -1;
}

// staging=0 ring: where a stream entry's n experts go, given each block's rank (-1 unusable: in flight, pinned, lent; else cs_plan's tier rank, 6 = a layer still to
// come). The best ranks first (free, empty, own computed, other class), then cached experts already computed, then a layer to come: at a pass's start every block
// holds one, and the window must still form there (2026-11-14, Qwen3.8 12k: gated on "the tiered pick would take them too", 0 of 123 entries found a run, the
// scattered picks kept the class scattered). -1: no run that long, the caller picks blocks one by one.
static inline int32_t poolx_ring_window(const std::vector<int8_t> & rank, int32_t from, int32_t n) {
    for (int lim : { 3, 5, 6 }) {
        std::vector<uint8_t> ok(rank.size());
        for (size_t j = 0; j < rank.size(); ++j) { ok[j] = rank[j] >= 0 && rank[j] <= lim; }
        const int32_t j0 = poolx_first_window(ok, from, n);
        if (j0 >= 0) { return j0; }
    }
    return -1;
}

// staging=0 ring on a card's upload worker: how many of its oldest queued jobs must land before one more is queued, with depth jobs allowed in flight (0: no bound).
// Depth 2 is double buffering: one job on the link, the next queued behind it, so the link never idles; a deeper queue only fills the driver's, whose blocked
// submissions then hold everyone's calls (2026-11-14, Qwen3.8 12k: 14 GB queued on one ring, the scheduler thread's launches sat 343 ms behind it).
static inline size_t poolx_ring_retire(size_t in_flight, size_t depth) { return depth > 0 && in_flight >= depth ? in_flight - depth + 1 : 0; }

// staging=0 ring: what a block a stream entry may take holds (cs_plan's tiers): 0 free, 1 a layer's empty block, 2 an expert streamed this pass whose layer is computed,
// 3 an expert cached before the pass whose layer is computed, 4 a cached expert of a layer still to come in this pass, 5 another class's bytes over nothing of a layer
// to come, 6 another class's bytes over an expert of a layer to come, 7 a prediction giving way (the k arm's late fix), 8 an expert of the card's last queued entry
// (it may still compute: its copy waits for that compute; user 2026-11-14 "not evict any moe selected [by] this processed layer"). Costly (4, 6): that expert goes up
// again when its layer comes (Qwen3.8 IQ4_XS 12k staging=0, 2026-11-14: 2825 of pass 1's 23013 blocks were layers ahead and 4510 were claims ranked before computed
// experts; 55.9 GB uploaded against staged's 47.9). Rank: the order taken, cheap before costly, the layer still computing after every other cheap block; a costly
// block only for an entry needed now (poolx_ring_cheap).
// 9 a decode-hot cached expert of a computed layer, 10 another class's bytes over one (--moe ring-keep-hot=1, default off): the stream takes them after every
// other cheap block and only for the entry needed now, so a prompt leaves decode its hot experts (Qwen3.8 IQ4_XS 12k, 2026-11-15: staging=0 refilled 24-26 GB
// after the prompt against staged's 14-15 GB, decode 45.8 against 49.95 t/s)
static inline bool poolx_ring_costly(int tier) { return tier == 4 || tier == 6 || tier == 9 || tier == 10; }
static inline int  poolx_ring_rank(int tier) { static const int r[11] = { 0, 1, 2, 4, 8, 3, 9, 10, 5, 6, 7 }; return tier >= 0 && tier < 11 ? r[tier] : 11; }
// The refill after a prompt in two parts (user 2026-11-15): part 1 = the experts the next tokens most likely use, the ones the cache misses among a layer's
// top k by decode score (k: the model's experts per token), at normal priority, so the step after waits for them; part 2 = every other missed expert with a
// score, in the background behind decode's own misses. All at normal priority made a 121-token answer wait 0.9 s for 24 GB (Qwen3.8 IQ4_XS 11k: 34.0
// against 45.8 t/s). score: the layer's experts' decode scores; cached: held, queued or adopting already. part[e]: 0 none, 1 first, 2 rest.
template <class S, class C> static std::vector<int8_t> poolx_refill_parts(const S & score, const C & cached, int32_t k) {
    const size_t n = score.size();
    std::vector<int32_t> ord(n);
    for (size_t i = 0; i < n; ++i) { ord[i] = (int32_t) i; }
    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return score[(size_t) a] > score[(size_t) b]; });
    std::vector<int8_t> part(n, 0);
    for (size_t r = 0; r < n; ++r) {
        const size_t e = (size_t) ord[r];
        if (cached[e] || !(score[e] > 0)) { continue; }
        part[e] = (int64_t) r < (int64_t) k ? 1 : 2;
    }
    return part;
}
// decode-hot: decode visits at or above the layer's mean over all its experts
template <class V> static double poolx_hot_cut(const V & cnt) { double s = 0; for (const auto v : cnt) { s += (double) v; } return cnt.empty() ? 0.0 : s / (double) cnt.size(); }
static inline bool poolx_is_hot(double visits, double cut) { return visits > 0 && visits >= cut; }
// what taking a block of a tier does to what lies there: nothing (free, empty), its cached expert leaves the table (cs_evict: 2, 3, 4, 8), the other classes'
// experts over its bytes leave (cls_claim: 5, 6), a prediction gives way (7). Never by comparing tiers: tier 8 once went down the claim branch and its expert stayed
// in the table over the new expert's bytes (vf-qni3 on ring_cover_all, 1830 slices bad)
enum poolx_take_kind { POOLX_TAKE_NONE, POOLX_TAKE_EVICT, POOLX_TAKE_CLAIM, POOLX_TAKE_YIELD };
static inline poolx_take_kind poolx_ring_take(int tier) {
    switch (tier) {
        case 0: case 1:         return POOLX_TAKE_NONE;
        case 5: case 6: case 10: return POOLX_TAKE_CLAIM;
        case 7:                 return POOLX_TAKE_YIELD;
        default:                return POOLX_TAKE_EVICT;
    }
}
// an entry planned beyond the one needed now (must) and the card's next (lead) takes cheap blocks only: it waits for computed layers' blocks or its own turn
// (SIZE_MAX: none needed now, as at a pass edge)
static inline bool poolx_ring_cheap(size_t must, size_t nx, size_t lead) { return must == SIZE_MAX || (nx > must && nx != lead); }
// The k arm's entries a card may plan ahead of the one it computes now. Its lists upload only predicted experts, and each planned entry keeps up to its range of
// the class's blocks until its kernels ran; the entry being computed and its late fix (what its real ids use beyond the list) need up to two ranges more.
// Kimi K3.1 2026-11-15, the first build with its prompt rankings: the plans ahead held 716 of a class's 831 blocks (340 in flight, 376 uploaded and not
// computed, 115 lent), layer 11's late fix found no block and the server aborted. held: the blocks lent to graphs and under KV rows.
static inline int poolx_k_ahead_max(int32_t class_blocks, int32_t held, int32_t entry_max) {
    if (entry_max <= 0) { return 1; }
    return std::max<int32_t>(1, (class_blocks - held - 2 * entry_max) / entry_max);
}
// the tier of a cached expert's block for the entry at pass position p, -1 when the entry may not take it. q: the position of the layer (or shadow) holding it;
// done: the card's entries queued; up_now: streamed in this pass; mix_home: a mix home's (its shadow left those out); cheap: poolx_ring_cheap
static inline int poolx_ring_tier(size_t q, size_t done, size_t p, bool up_now, bool mix_home, bool cheap) {
    if (q < done) { return q + 1 == done ? 8 : up_now ? 2 : 3; }
    if (q > p && !mix_home) { return cheap ? -1 : 4; }
    return -1;   // uploaded for in this pass and not queued yet, or the entry's own
}
// The parts a cached expert's victim rank is built from, shared by the decode cache's prompt misses (prompt_victim_rank) and the staging=0 ring (cs_plan), each
// weighing them by its own load. A prompt's passes walk the layers in order: a layer's experts are used again at its turn, later in this pass or in the next one.
// layer steps until layer (or pass position) o is used again, the pass being at jli of n
static inline size_t poolx_next_use(size_t o, size_t jli, size_t n) { return o > jli ? o - jli : n - jli + o; }
// by next use: the furthest first, one layer's coldest first (s: the decode cache's score, bounded under 9e5); the next `soon` layers last of all
static inline double poolx_rank_next_use(size_t o, size_t jli, size_t n, double s, size_t soon) {
    const size_t d = poolx_next_use(o, jli, n);
    return (d <= soon ? 2e12 : 1e12) - (double) d * 1e6 + s;
}
// The decode cache's prompt victim rank (prompt_victim_rank), lowest taken first; jli: the layer the pass is at. off (default): the layers done coldest first
// wherever they lie, then the layers to come, the furthest first and the coldest within. on (--moe prompt-victim=1), all (the pass visits nearly every expert): by
// next use, the next two layers last; on, a smaller pass: the coldest wherever (equal: the later layer), the next two layers last.
static inline double poolx_victim_rank(bool on, bool all, size_t o, size_t jli, size_t n_layers, double s) {
    if (!on) { return o > jli ? 1e12 - (double) o * 1e6 + s : s; }
    if (all) { return poolx_rank_next_use(o, jli, n_layers, s, 2); }
    if (o > jli && o <= jli + 2) { return 1e12 - (double) o * 1e6 + s; }
    return s + 1e-3 * (1.0 - (double) o / (double) n_layers);
}
// The staging=0 ring's rank (prefill: every expert of every layer streams each pass, so when a block's expert is used again decides, not how hot it is; user
// 2026-11-14 "prioritize moes that are to be used soon first", "the load and priorities are way different in prefill"): by next use over pass positions (n: the
// pass's entries), the next two entries last; a block streamed in this pass ranks by its layer alone (s ignored: one layer's blocks go in the cursor's order, so an
// entry's copies stay in runs), a cached one coldest first within its layer.
static inline double poolx_ring_victim(size_t o, size_t jli, size_t n, double s, bool streamed) { return poolx_rank_next_use(o, jli, n, streamed ? 0.0 : s, 2); }
// the order taken: by tier rank, then the lower rank (poolx_ring_victim; a claim over a pending expert by its nearest pending layer, the furthest first); equal
// ranks keep the cursor's order (stable sort)
static inline bool poolx_ring_before(int tier_x, double rank_x, int tier_y, double rank_y) {
    const int rx = poolx_ring_rank(tier_x), ry = poolx_ring_rank(tier_y);
    if (rx != ry) { return rx < ry; }
    return rank_x < rank_y;
}

template <class P> static size_t poolx_lend_round(const P & ap, size_t bytes) {
    for (const auto & sg : ap.segs) {
        const size_t end = sg.base + (size_t) sg.n * sg.B;
        if (sg.B > 0 && bytes > sg.base && bytes < end) { return sg.base + (bytes - sg.base + sg.B - 1) / sg.B * sg.B; }
    }
    return bytes;
}
// the blocks of one class a lend of need_b bytes covers (never more than the class's n blocks: past them lies the next class)
template <class S> static int32_t poolx_lend_blocks(const S & sg, size_t need_b) {
    const size_t cov = need_b > sg.base ? need_b - sg.base : 0;
    return sg.B == 0 ? 0 : (int32_t) std::min<size_t>((size_t) sg.n, (cov + sg.B - 1) / sg.B);
}

// ---- owner ledger -----------------------------------------------------------------------------------------------------------------------------
// ps: { std::vector<int32_t> owner, free; int32_t home_lo, home_hi; }. The caller holds the cache's mutex. evict(layer, block) takes an expert out of a block it owned (the caller's bookkeeping).
// Lend blocks [from, to) of a class to a graph. Returns the blocks that were under the KV zone (-3): they are NOT taken (a lend never goes over the zone; the caller reports).
template <class PS, class F, class G> static int poolx_lend(PS & ps, int32_t from, int32_t to, F evict, G take_bytes) {
    int bad = 0;
    for (int32_t b = from; b < to; ++b) {
        const int32_t o = ps.owner[b];
        if (o == -3) { bad++; continue; }
        if (o == -4) { take_bytes(b); }   // another class's grid holds these bytes (staging=0): its block goes first (poolx_take_bytes)
        if (o >= 0) { evict(o, b); }
        else if (o == -1) {
            auto it = std::find(ps.free.begin(), ps.free.end(), b);
            if (it != ps.free.end()) { ps.free.erase(it); }
        }
        ps.owner[b] = -2;
    }
    return bad;
}
// A block a lend or the zone gives back: free in the class's home, another class's bytes (-4) outside it (a grid over a neighbour's bytes, --moe staging=0:
// freed there, two classes would own the same bytes and place two experts on them). Returns true when it went on the free list.
template <class PS> static bool poolx_give_back(PS & ps, int32_t b) {
    if (b >= ps.home_lo && b < ps.home_hi) { ps.owner[b] = -1; ps.free.push_back(b); return true; }
    ps.owner[b] = -4;
    return false;
}
// Blocks [keep, blocked) go back to the experts (only the lent ones). Returns how many went on the free list.
template <class PS> static size_t poolx_release(PS & ps, int32_t keep, int32_t blocked) {
    size_t n = 0;
    for (int32_t b = keep; b < blocked; ++b) {
        if (ps.owner[b] == -2) { n += poolx_give_back(ps, b); }
    }
    return n;
}
// The KV zone takes `take` (ascending block numbers): experts out, free blocks off the free list, lent blocks marked (*blocked drops to the lowest: the lend of this graph starts
// over below the KV). The owner becomes `tag` (-3 the KV zone, -5 CUDA's room).
template <class PS, class F, class G> static void poolx_zone_take(PS & ps, const std::vector<int32_t> & take, int32_t * blocked, F evict, G take_bytes, int32_t tag = -3) {
    std::vector<char> free_mark;
    for (int32_t b : take) {
        const int32_t o = ps.owner[b];
        if (o == -4) { take_bytes(b); }   // another class's grid holds these bytes (staging=0): its block goes first (poolx_take_bytes)
        if (o >= 0) { evict(o, b); }
        else if (o == -1) {
            if (free_mark.empty()) { free_mark.assign(ps.owner.size(), 0); }
            free_mark[b] = 1;
        } else if (o == -2 && blocked) { *blocked = std::min(*blocked, b); }
        ps.owner[b] = tag;
    }
    if (!free_mark.empty()) {
        ps.free.erase(std::remove_if(ps.free.begin(), ps.free.end(), [&](int32_t b) { return free_mark[b] != 0; }), ps.free.end());
    }
}
// Blocks the zone (or CUDA's room, tag -5) left go back to the free list.
template <class PS> static size_t poolx_zone_give(PS & ps, const std::vector<int32_t> & give, int32_t tag = -3) {
    size_t n = 0;
    for (int32_t b : give) { if (ps.owner[b] != tag) { continue; } n += poolx_give_back(ps, b); }   // only a block under the zone goes back: a lent or owned one is not free
    return n;
}

// ---- CUDA's room at the pool's tail ------------------------------------------------------------------------------------------------------------
// (user 2026-11-16: "use tail", "give them to cuda as needed and reclaim when not", no gap kept free.) The tail of the pool holds cached experts only (no
// stream, lend or KV there); CUDA's own memory (kernels loaded on first use, cuBLAS, CUDA graphs, page tables of pinned RAM) takes whole blocks from its top
// when it runs out and gives them back as soon as its use drops. A call of a shape class not seen yet may load new kernels (mmq, mmvq and flash attention
// variants are picked by the token count): tokens up to 8 alone, then steps of 8 up to 128, then powers of two.
static inline int64_t poolx_room_class(int64_t n_tokens) {
    if (n_tokens <= 8) { return n_tokens; }
    if (n_tokens <= 128) { return (n_tokens + 7) / 8 * 8; }
    int64_t c = 256;
    while (c < n_tokens) { c *= 2; }
    return c;
}
// the room after CUDA ran out of its own memory: one block more, twice what it had or `hint` (the most its use grew between two looks), whichever is
// largest, in whole blocks of `step`, never past the tail area (room_max)
static inline size_t poolx_room_grow(size_t room, size_t room_max, size_t step, size_t hint) {
    if (step == 0) { return room; }
    const size_t r = poolx_up(std::max({ room + step, 2 * room, hint }), step);
    return std::min(r, room_max);
}
// the room after a call: whole blocks of the card's free bytes go back to the cache at once; `keep` is free VRAM that is not the room's (left outside the pool
// for others when it was sized)
static inline size_t poolx_room_give_back(size_t room, size_t free_b, size_t keep, size_t step) {
    if (step == 0 || free_b <= keep) { return room; }
    const size_t give = (free_b - keep) / step * step;
    return room - std::min(room, give);
}
// The room a pool starts with when the driver could not back all of its chunked tail at creation (VRAM taken between the sizing and the allocation): the
// block edge (base + k * step) at or below the first unmapped byte, tail_real when only the slack past the last block is unmapped, SIZE_MAX when it reaches
// below the tail area (vol_lo): the stream, the lend and the KV live there, such a pool cannot be used.
static inline size_t poolx_room_start(size_t vsz, size_t unmapped, size_t vol_lo, size_t tail_real, size_t base, size_t step) {
    const size_t first = vsz - std::min(vsz, unmapped);   // the first unmapped byte
    if (first >= tail_real) { return tail_real; }
    if (first < vol_lo || step == 0 || first < base) { return SIZE_MAX; }
    return base + (first - base) / step * step;
}
// The page tables the loader's pins still to come take on every card (it reads into pinned buffers while the context is made): one 8 B GPU page table entry
// per 4 KiB page (PC1 measured 1.4-2 MiB per GiB) and one 2 MiB step of the driver's page-table allocation; 0 when nothing is left to pin.
static inline size_t poolx_pins_vram(size_t pin_total, size_t pinned) {
    return pin_total > pinned ? (pin_total - pinned) / 4096 * 8 + ((size_t) 2 << 20) : 0;
}
// An expert takes a free block for layer li: the block popped off the free stack, or -1. A block that is not free (stale entry) is dropped, never handed out.
template <class PS> static int32_t poolx_claim_free(PS & ps, int32_t li) {
    while (!ps.free.empty()) {
        const int32_t b = ps.free.back(); ps.free.pop_back();
        if (ps.owner[b] == -1) { ps.owner[b] = li; return b; }
    }
    return -1;
}

// ---- grids (--moe staging=0): a class's blocks over other classes' bytes ----------------------------------------------------------------------
// Every class's block grid runs on from its home over the pool's other bytes (the layer ring). A block past a class's home is -4 (another class's bytes) until
// it claims them; a home block whose bytes another class claimed is -4 too. A byte has one owner at a time: an expert or a free block of one class, a lent
// block, or the KV zone. tests/test-pool-fork.cpp (test_grid_claims) checks that byte by byte under random lends, zones, claims, evictions and returns home.
//
// every block of another class over the bytes of class pc's block b: f(class, block), false from f stops it (and the call returns false). segs: one per
// class with its pcls, base and block size B; grid_n(class): the blocks of that class's grid (its home and the blocks past it).
template <class SEGS, class GN, class F> static bool poolx_over(const SEGS & segs, GN grid_n, int32_t pc, int32_t b, F f) {
    const typename SEGS::value_type * me = nullptr;
    for (const auto & sg : segs) { if (sg.pcls == pc) { me = &sg; } }
    if (!me) { return false; }
    const size_t lo = me->base + (size_t) b * me->B, hi = lo + me->B;
    for (const auto & sg : segs) {
        if (sg.pcls == pc || sg.B == 0) { continue; }
        const size_t gnb = (size_t) grid_n(sg.pcls);
        const size_t end = sg.base + gnb * sg.B;
        if (hi <= sg.base || lo >= end) { continue; }
        const int32_t x0 = lo > sg.base ? (int32_t) ((lo - sg.base) / sg.B) : 0;
        const int32_t x1 = (int32_t) std::min<size_t>(gnb, (hi - sg.base + sg.B - 1) / sg.B);
        for (int32_t x = x0; x < x1; ++x) { if (!f(sg.pcls, x)) { return false; } }
    }
    return true;
}
// The bytes of class pc's block b back from every other class: an expert there is evicted (evict(class, block); the caller drops it from its layer's table),
// a free block leaves its free list; each becomes -4. Lent or zone blocks of another class are left (a lend or zone of its own over the same bytes). Called
// before a claim (pc's block then free), a lend (-2) or the KV zone (-3) takes block b: without it a lend or zone over bytes another class's grid held an
// expert in wrote over that expert while its table still pointed there (found 2026-11-14 reading the code; latent in the ring's calls).
template <class SEGS, class PSV, class F> static void poolx_take_bytes(const SEGS & segs, PSV & ps, int32_t pc, int32_t b, F evict) {
    poolx_over(segs, [&](int32_t q) { return (int32_t) ps[q].owner.size(); }, pc, b, [&](int32_t q, int32_t x) {
        auto & qs = ps[q];
        const int32_t o = qs.owner[x];
        if (o == -4 || o == -2 || o == -3) { return true; }
        if (o == -1) { auto it = std::find(qs.free.begin(), qs.free.end(), x); if (it != qs.free.end()) { qs.free.erase(it); } }
        else { evict(q, x); }   // an expert (o >= 0) or a one-pass shadow's (o <= -5)
        qs.owner[x] = -4;
        return true;
    });
}
// The end of a prompt call: every class back to its home. Pass 0 (over every class first): a block outside its home that holds bytes goes (an expert evicted:
// evict(block, owner), a free one off the list) and is -4 again; lent or zone blocks are never outside a home. Pass 1: a home block another class held is free.
template <class PS, class F> static void poolx_home_pass(PS & ps, int pass, F evict) {
    for (int32_t b = 0; b < (int32_t) ps.owner.size(); ++b) {
        const bool home = b >= ps.home_lo && b < ps.home_hi;
        const int32_t o = ps.owner[b];
        if (pass == 0 && !home && o != -4) {
            if (o < -1) { continue; }
            if (o >= 0) { evict(b, o); }
            else { auto it = std::find(ps.free.begin(), ps.free.end(), b); if (it != ps.free.end()) { ps.free.erase(it); } }
            ps.owner[b] = -4;
        } else if (pass == 1 && home && o == -4) {
            ps.owner[b] = -1;
            ps.free.push_back(b);
        }
    }
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
    }
    return m;
}
// The gate every write into an expert block consults first: nullptr when [lo, hi) is clear, else "zone" (and the region in *rlo / *rhi). No allocation.
template <class P> static const char * poolx_gate_kind(const P & ap, size_t lo, size_t hi, size_t * rlo, size_t * rhi) {
    if (ap.kv_h > 0 && lo < poolx_kv_end(ap) && hi > ap.kv_start) { *rlo = ap.kv_start; *rhi = poolx_kv_end(ap); return "zone"; }
    // CUDA's room, a growth asked for included: its chunks are unmapped, and a copy there is no GPU fault but a CPU write inside libcuda (the driver takes an
    // unknown address for host memory: segfault, glm-i and mimo-i 2026-11-16)
    const size_t r0 = std::min(ap.room_lo, ap.room_lo_want);
    if (ap.room_max > 0 && r0 < ap.tail_real && lo < ap.tail_real && hi > r0) { *rlo = r0; *rhi = ap.tail_real; return "room"; }
    return nullptr;
}

// ---- range use counts -------------------------------------------------------------------------------------------------------------------------
// Use counts per byte range (replaces a lock per buffer): everyone who touches a range of a card's pool - an upload DMA (w), a keep-hot copy (w), a graph reading the lend -
// enters it first and leaves it when its DMA has finished. A mover (zone resize + row move) first announces its range, which holds new entrants back, then waits until no
// entered user overlaps it, moves, and ends the move. So a move never starts under a running DMA and no write starts under a running move. K identifies the pool (its address).
// Fixed tables, no heap at run time: at most one entry per upload worker plus the decode thread (use), one per mover (moving).
template <class K> struct poolx_ranges {
    static constexpr int USE_MAX = 256, MOVE_MAX = 8;
    struct ent { K ap; size_t lo, hi; bool w; int64_t t0 = 0; std::thread::id tid; };   // t0: when it entered (a DMA's start; it leaves when the copy is done); tid: who entered
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
    // the writes in flight (upload DMAs, keep-hot copies) that overlap [lo, hi): up to max of them into out, returns how many there are (the pool guard names them)
    int writers(K ap, size_t lo, size_t hi, ent * out, int max) {
        std::lock_guard<std::mutex> lk(mu);
        int n = 0;
        const std::thread::id me = std::this_thread::get_id();   // the caller's own entry is the copy it is making now (one thread, one stream: ordered), not a race
        for (int i = 0; i < use.n; ++i) { if (use.e[i].w && use.e[i].tid != me && ovl(use.e[i], ap, lo, hi)) { if (n < max) { out[n] = use.e[i]; } n++; } }
        return n;
    }
private:
    static void add(tab & t, const ent & u, int cap) {
        if (t.n >= cap) { std::abort(); }   // "pool range table full"
        t.e[t.n] = u;
        t.e[t.n].tid = std::this_thread::get_id();
        t.e[t.n++].t0 = now_us();
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
        if (o < -1) { continue; }   // lent, under the KV zone, another class's bytes (-4: a grid block not claimed, llama-moecache-fork.cpp)
        // an empty block has no victim: the owner of an earlier candidate must not stay in the answer. A caller evicts by owner and slot, and an adoption took expert -1 of that
        // owner: a write before its table, and a general protection fault at the next step (Qwen Next, forced prompt mix, 2026-11-02)
        if (o < 0) { p.slot = b; p.owner = -1; best = 0; return p; }
        if (busy(o, b)) { continue; }
        const int32_t e = expert(o, b);
        if (e < 0) { p.slot = b; p.owner = -1; best = 0; return p; }
        const double r = rank(o, e);
        if (r >= 0 && r < best) { best = r; p.slot = b; p.owner = o; }
    }
    return p;
}

// ---- the prompt chain: which blocks of a class a layer's misses may take -------------------------------------------------------------------------
// fn(b, o) for every block that is free (-1), holds a layer's expert (>= 0) or, with grid (staging=0), is another class's bytes the chain may claim (-4). Every block
// of the class, not only the first n_active: a grid block a layer of this pass claimed sits past the home and stays this class's until cls_home, so the layers after
// it must see it as theirs to evict (a K3 4-token call after a 16k prompt: CUDA1's layers each claimed fresh grid blocks and the 26th found 38 for 50 misses, abort,
// 2026-11-14). Lent (-2), under the KV rows (-3) and a stream shadow's (<= -5) are never passed.
template <class PS, class Fn>
static void poolx_chain_scan(const PS & ps, bool grid, Fn fn) {
    for (int32_t b = 0; b < (int32_t) ps.owner.size(); ++b) {
        const int32_t o = ps.owner[b];
        if (o >= -1 || (grid && o == -4)) { fn(b, o); }
    }
}
