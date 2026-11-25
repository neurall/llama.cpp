// Unit test of the rule that refuses a separate draft beside a target bigger than the free VRAM (common/spec-guard-fork.h). No GPU, no model.
// Cases are the rows of 2026-11-25 (2x RTX 3090, 23.6 GiB free each; Qwen3.8 Next IQ4_XS 87 GiB, Inkling 2 Mini 15 GiB).
// build: cmake --build <dir> --target test-spec-guard-fork;  run: ctest -R test-spec-guard-fork
#include "../common/spec-guard-fork.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    const double free1 = 23.6, free2 = 2 * 23.6, qwen = 87.0, ink = 15.0;
    CHECK( spec_draft_refused_fork(qwen / free2, true, false));    // two cards, 1.84x: the row that aborted at load
    CHECK( spec_draft_refused_fork(qwen / free1, true, false));    // one card, 3.7x: refused before too
    CHECK(!spec_draft_refused_fork(ink / free1, true, false));     // Inkling + DSpark fits one card: the draft loads
    CHECK(!spec_draft_refused_fork(ink / free2, true, false));
    CHECK(!spec_draft_refused_fork(qwen / free2, true, true));     // --spec-draft-n-max N forces it
    CHECK(!spec_draft_refused_fork(qwen / free2, false, false));   // built-in MTP, no -md: not this rule
    if (g_fail) { fprintf(stderr, "test-spec-guard-fork: %d failed\n", g_fail); return 1; }
    printf("test-spec-guard-fork: ok\n");
    return 0;
}
