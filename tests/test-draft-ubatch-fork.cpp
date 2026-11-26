// Unit test of the draft context's ubatch rule (common/draft-ubatch-fork.h). No GPU, no model.
// build: cmake --build <dir> --target test-draft-ubatch-fork;  run: ctest -R test-draft-ubatch-fork
#include "../common/draft-ubatch-fork.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    const uint64_t MiB = 1ull << 20;
    // Inkling 2 Mini DSpark draft as the server logs it: "draft ubatch 256 (target 4096): vocab-wide rows 1.5 MiB a token, draft weights 469 MiB"
    CHECK(draft_ubatch_fork(196608, 469 * MiB, 4096, 0) == 256);
    // a target ubatch below the rule's result is kept
    CHECK(draft_ubatch_fork(196608, 469 * MiB, 128, 0) == 128);
    CHECK(draft_ubatch_fork(196608, 469 * MiB, 32, 0) == 32);
    // a draft too small for 128 rows still gets a draft block batch (64)
    CHECK(draft_ubatch_fork(196608, 10 * MiB, 4096, 0) == 64);
    // no vocab size known: the target's ubatch (a power of two) as is
    CHECK(draft_ubatch_fork(0, 469 * MiB, 4096, 0) == 4096);
    // a big draft is held to the target's ubatch
    CHECK(draft_ubatch_fork(131072, 64000 * MiB, 2048, 0) == 2048);
    // DSV4 draft with a prompt ring: window 128 + a 64-token draft block = 192; 1 MiB of vocab rows a token, 3 GiB of weights (2048 by the vocab rule)
    CHECK(draft_ubatch_fork(131072, 3072 * MiB, 4096, 192) == 256);
    // the ring never cuts below 64
    CHECK(draft_ubatch_fork(131072, 3072 * MiB, 4096, 1) == 64);
    if (g_fail) { fprintf(stderr, "test-draft-ubatch-fork: %d failed\n", g_fail); return 1; }
    printf("test-draft-ubatch-fork: ok\n");
    return 0;
}
