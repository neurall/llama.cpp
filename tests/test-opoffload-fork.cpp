// Unit test of the op offload decision (src/llama-opoffload-fork.h). No GPU, no model.
// build: cmake --build <dir> --target test-opoffload-fork;  run: ctest -R test-opoffload-fork
#include "../src/llama-opoffload-fork.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    const size_t MiB = 1ull << 20;
    // RTX 4060 8 GB on W10, Inkling 2 Mini 12k (2026-11-26): "graph 10147 MiB at 8192 tokens, 8356 at 4096: 6566 MiB fixed ... + 0.437 MiB a token;
    // free 5950 MiB < 8737 MiB for a 512-token pass"; Windows margin 484 MiB, KV arena fixed part 1440 MiB, 32768 B a cell
    const opoff_plan_fork p = opoff_plan_fork_calc(10147 * MiB, 8356 * MiB, 8192, 5950 * MiB, 484 * MiB, 0, 1440 * MiB, 32768);
    CHECK(p.off);
    CHECK(p.fixed / MiB > 6560 && p.fixed / MiB < 6570);
    CHECK(p.per_tok / MiB > 0.43 && p.per_tok / MiB < 0.44);
    CHECK(p.need / MiB > 8700 && p.need / MiB < 8760);
    // without op offload the pass takes at most half of what the margins leave: at 8192 tokens the KV got 10240 cells and the 18642-token prompt failed
    CHECK(p.ubatch >= 512 && p.ubatch < 8192 && p.ubatch % 32 == 0);
    CHECK((double) p.ubatch * p.per_tok <= p.left / 2 + p.per_tok * 32);
    // the same graph on a 24 GB card keeps op offload and its micro-batch
    const opoff_plan_fork big = opoff_plan_fork_calc(10147 * MiB, 8356 * MiB, 8192, 22000 * MiB, 384 * MiB, 0, 1440 * MiB, 32768);
    CHECK(!big.off && big.ubatch == 0);
    // a graph that grows with tokens alone (no fixed part) on a card it fits: kept
    const opoff_plan_fork lin = opoff_plan_fork_calc(4096 * MiB, 2048 * MiB, 8192, 5950 * MiB, 484 * MiB, 0, 1440 * MiB, 32768);
    CHECK(!lin.off && lin.fixed / MiB < 1);
    // nothing left beside the margins: the smallest pass, never 0
    const opoff_plan_fork none = opoff_plan_fork_calc(10147 * MiB, 8356 * MiB, 8192, 1500 * MiB, 484 * MiB, 0, 1440 * MiB, 32768);
    CHECK(none.off && none.ubatch == 512);
    // a second measure that did not shrink (lt2 >= lt): no decision
    CHECK(!opoff_plan_fork_calc(4096 * MiB, 4096 * MiB, 8192, 100 * MiB, 484 * MiB, 0, 0, 0).off);
    // W10 retry 2026-11-27: the 512-token try ("graph 7081 MiB at 544 tokens, 6887 at 272: 6692 MiB fixed") must still be checked and refused
    CHECK(opoff_check_due_fork(true, 48, 512));          // the old rule (n_tokens > 512) skipped it: every request then failed with "Compute error"
    CHECK(!opoff_check_due_fork(true, 0, 512));          // a CPU run keeps the old way
    CHECK(!opoff_check_due_fork(false, 48, 512));        // no op offload: nothing to check
    CHECK(opoff_check_due_fork(false, 48, 8192));
    CHECK(opoff_plan_fork_calc(7040 * MiB, 6886 * MiB, 512, 5950 * MiB, 484 * MiB, 0, 1440 * MiB, 32768).off);
    // a 256-token pass that fits is not refused for what a 512-token pass would need
    CHECK(!opoff_plan_fork_calc(4000 * MiB, 3944 * MiB, 256, 5990 * MiB, 484 * MiB, 0, 1440 * MiB, 32768).off);
    // W10 --no-op-offload 2026-11-27: "free 5950 MiB, margin 3212 (KV fixed 2728, 0.031 MiB a cell), graph 7205 MiB at 16384 -> 5792": KV got 4608 cells
    const double g = 7205.0 * MiB / 16384;
    const uint32_t full = ub_cut_fork(5950 * MiB, 3212 * MiB, g, 32768, 16384, false);
    CHECK(full >= 5700 && full <= 5800);                  // the old cut: the 12k prompt did not fit the KV
    const uint32_t half = ub_cut_fork(5950 * MiB, 3212 * MiB, g, 32768, 16384, true);
    CHECK(half >= 512 && half < full);
    CHECK((5950.0 - 3212) * MiB - half * g >= 12846.0 * 32768);   // the KV holds the 12846-token prompt beside that pass
    CHECK(ub_cut_fork(3000 * MiB, 3212 * MiB, g, 32768, 16384, true) == 0);
    if (g_fail) { fprintf(stderr, "test-opoffload-fork: %d failed\n", g_fail); return 1; }
    printf("test-opoffload-fork: ok\n");
    return 0;
}
