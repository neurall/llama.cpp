// Unit test of the decode bell's pool size (src/llama-bellsize-fork.h bell_pool_size). No GPU, no model.
// The machines: PC1 (Threadripper PRO 3975WX: 64 CPUs, 32 cores, 8 L3 domains; Hy4-Flash's decode team of 32 computes 12 lightning indexers a token on
// the CPU and spins), the W10 laptop (Ryzen 9 8945HS: 16 CPUs, 8 cores, 1 L3 domain), a CPU without SMT (24 cores, 2 L3 domains). On 2026-11-26 the
// pool had a worker a core: on PC1 with Hy4 that made 64 spinning threads on 64 CPUs, 7-24% of the tokens over 1.5x the median (p99 71-107 ms), and
// 5 of 11 cells fell under stock in a run (24.2/24.4, 23.3, 21.2 o/s against 28.95/30.08, runs.db 36045/36046, 36056, 36016); 24 workers: 34.1-35.2 o/s
// in 4 answers, 1% over 1.5x the median, p99 38-39 ms.
// The binary runs the checks on the rule and on the old one (a worker a core): the rule must pass all, the old one must fail the spinning-team checks.
// build: cmake --build <dir> --target test-bellsize-fork;  run: ctest -R test-bellsize-fork
#include "../src/llama-bellsize-fork.h"

#include <cstdio>

struct box { const char * name; int n_allowed, n_physical, n_l3; };
static const box pc1  = { "PC1",           64, 32, 8 };
static const box w10  = { "W10 laptop",    16,  8, 1 };
static const box nsmt = { "CPU, no SMT",   24, 24, 2 };

static int rule(const box & b, int team) { return bell_pool_size(b.n_allowed, b.n_physical, team, b.n_l3); }
static int old_rule(const box & b, int team) { (void) team; return b.n_physical; }

static int failed(int (*size)(const box &, int)) {
    int n = 0;
    auto check = [&](bool c, const box & b, const char * what) { if (!c) { n++; fprintf(stderr, "  fails: %s: %s\n", b.name, what); } };
    // a spinning team: the pool's workers, the team and a free CPU an L3 domain fit the CPUs (or the pool is the bell thread alone)
    for (const box * b : { &pc1, &w10, &nsmt }) {
        const int team = b->n_physical;   // the decode threads start at the true cores (common_thread_start_fork; every CPU without SMT)
        const int k = size(*b, team);
        check(k == 1 || team + k + b->n_l3 <= b->n_allowed, *b, "a spinning decode team: the pool fits beside it, a free CPU in every L3 domain");
        check(k >= 1 && k <= b->n_physical, *b, "a spinning decode team: from the bell thread alone to a worker a core");
    }
    check(size(pc1, 32) >= 16, pc1, "Hy4 (decode team 32): the pool keeps at least the 16 workers measured at 34.9/35.3 o/s");
    check(size(pc1, 16) + 16 + pc1.n_l3 <= pc1.n_allowed, pc1, "a decode team of 16 (a stored walk): the pool fits beside it");
    // a decode team that waits (its graph has no CPU work): a worker a core, as before, where the cores leave a CPU in every L3 domain
    check(size(pc1, 0) == 32, pc1, "a waiting decode team: a worker a core");
    check(size(w10, 0) == 8, w10, "a waiting decode team: a worker a core");
    return n;
}

int main() {
    int n_fn = 0;
    auto check = [&](bool c, const char * what) { if (!c) { n_fn++; fprintf(stderr, "  fails: %s\n", what); } };
    check(bell_pool_size(64, 32, 32, 8) == 24, "PC1, Hy4's decode team of 32: 24 workers (measured: 34.6/35.2 and 34.1/35.1 o/s)");
    check(bell_pool_size(16, 8, 8, 1) == 7, "W10 laptop, a spinning decode team of 8: 7 workers");
    check(bell_pool_size(24, 24, 24, 2) == 1, "no SMT, the decode team on every CPU: the bell thread alone");
    check(bell_pool_size(24, 24, 0, 2) == 22, "no SMT, a waiting decode team: the cores less a CPU an L3 domain");
    check(bell_pool_size(2, 1, 2, 1) == 1, "never under the bell thread itself");

    fprintf(stderr, "rule:\n");
    const int n_new = failed(rule);
    fprintf(stderr, "old rule (a worker a core):\n");
    const int n_old = failed(old_rule);
    printf("test-bellsize-fork: the rule fails %d checks (function %d), the old rule %d\n", n_new, n_fn, n_old);
    if (n_new != 0 || n_fn != 0 || n_old == 0) {
        fprintf(stderr, "test-bellsize-fork: FAILED\n");
        return 1;
    }
    printf("test-bellsize-fork: ok\n");
    return 0;
}
