// Unit test of CUDA's room before anything can load a kernel (src/llama-room-fork.h room_lo_before_plan). No GPU, no model.
// The numbers are CUDA0's of the Hy4 start that failed on 2026-11-26 (release b16903, RTX 3090): blocks of 2.43 MiB, a tail area of 132 blocks (every
// kernel of the build, 202 MiB, measured by the probe processes, plus the page tables of the pins still to come, 117 MiB), 4 MiB free beside the pool.
// The order that failed: CUDA's room from the start held the pins' share only (49 blocks), and the first call's scratch plan (the 2-token warmup's flash
// attention) ran before the lend grew the room: it loaded its kernel with what the pins left and aborted, out of memory. The rule: the whole tail area is
// CUDA's before anything can load a kernel (the pool's start, a class not seen yet), from the pool, whatever is free beside it (also 0 MiB).
// The binary runs the checks on the rule and on the old order: the rule must pass all, the old order must fail the kernel checks.
// build: cmake --build <dir> --target test-room-fork;  run: ctest -R test-room-fork
#include "../src/llama-room-fork.h"

#include <cstdio>

static const size_t MiB       = (size_t) 1 << 20;
static const size_t B         = (size_t) (2.43 * MiB);
static const size_t vol_lo    = 18730 * MiB;           // the tail area's first byte, a block edge
static const size_t tail_real = vol_lo + 132 * B;      // the pool's end: 132 blocks of tail area
static const size_t room_max  = tail_real - vol_lo;    // the whole tail area (sized from the kernels and the pins)
static const size_t pins      = 117 * MiB;             // page tables of the pins the loader made after the sizing
static const size_t kernels   = 202 * MiB;             // every kernel of this build loaded (the probe processes)

static size_t lo_of(size_t want, bool seen, size_t rmax = room_max) { return room_lo_before_plan(want, seen, vol_lo, tail_real, rmax, vol_lo, B); }

// the room's first byte at the pool's start and at the first call's plan, by the rule and by the old order
static size_t start_rule()        { return lo_of(tail_real, false); }                 // nothing is seen at the start
static size_t plan_rule(size_t s) { return lo_of(s, false); }                         // the warmup's class is not seen yet
static size_t start_old()         { return tail_real - (pins + B - 1) / B * B; }      // the pins' share (poolx_up(sz_pins, step))
static size_t plan_old(size_t s)  { return s; }                                       // the room grew after the plan

static int failed(size_t (*start)(), size_t (*plan)(size_t)) {
    int n = 0;
    auto check = [&](bool c, const char * what) { if (!c) { n++; fprintf(stderr, "  fails: %s\n", what); } };
    const size_t s = start(), p = plan(s);
    // at the plan CUDA holds the pins' page tables and may load every kernel; the free VRAM beside the pool is not counted on (4 MiB that day, or none)
    for (size_t beside : { 4 * MiB, (size_t) 0 }) {
        check(beside + (tail_real - p) >= pins + kernels, beside ? "first call's plan, 4 MiB beside the pool: every kernel fits" : "first call's plan, 0 MiB beside the pool: every kernel fits");
    }
    check(tail_real - s >= room_max, "the pool's start: the room is the whole tail area");
    check(p == s || tail_real - s < room_max, "the room is enough already at the plan: no extra block");
    check(p >= vol_lo && p <= tail_real && (p - vol_lo) % B == 0, "the room's edge is a block edge in the tail area");
    return n;
}

int main() {
    int n_fn = 0;
    auto check = [&](bool c, const char * what) { if (!c) { n_fn++; fprintf(stderr, "  fails: %s\n", what); } };
    const size_t want = tail_real - 49 * B;   // a room of 49 blocks
    check(lo_of(want, true) == want, "class seen: the room stays (nothing evicted)");
    check(lo_of(want, false) == vol_lo, "class not seen yet: the whole tail area");
    check(lo_of(vol_lo, false) == vol_lo, "the room is the whole area already: no extra block");
    check(lo_of(want, false, 100 * B + B / 2) == tail_real - 101 * B, "room limit 100.5 blocks (the driver's chunks): 101 blocks");
    check(lo_of(vol_lo, false, room_max / 2) == vol_lo, "a larger room than asked stays (the room only grows here)");
    check(lo_of(tail_real, false, 0) == tail_real, "no room on the card (no VMM, --moe room=0): nothing changes");

    fprintf(stderr, "rule:\n");
    const int n_new = failed(start_rule, plan_rule);
    fprintf(stderr, "old order (the pins' share at the start, the room grew after the plan):\n");
    const int n_old = failed(start_old, plan_old);
    printf("test-room-fork: the rule fails %d checks (function %d), the old order %d\n", n_new, n_fn, n_old);
    if (n_new != 0 || n_fn != 0 || n_old == 0) {
        fprintf(stderr, "test-room-fork: FAILED\n");
        return 1;
    }
    printf("test-room-fork: ok\n");
    return 0;
}
