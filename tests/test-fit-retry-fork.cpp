// Unit test of fit's retry rule for an extra model's memory measure (common/fit-retry-fork.h). No GPU, no model.
// The texts are the ones fit and llama_init_from_model really produce: fit throws FIT_ERR_NO_CTX when the context is null.
// build: cmake --build <dir> --target test-fit-retry-fork;  run: ctest -R test-fit-retry-fork
#include "../common/fit-retry-fork.h"

#include <cstdio>
#include <stdexcept>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    // what fit's measure throws for a draft that cannot make a context alone (the case of 2026-11-25)
    try { throw std::runtime_error(FIT_ERR_NO_CTX); }
    catch (const std::runtime_error & e) { CHECK(fit_retry_with_dry_target_fork(e.what())); }
    // the error text itself, should it ever reach fit
    CHECK(fit_retry_with_dry_target_fork("qwen4exp requires ctx_other to be set (this warning is normal during memory fitting)"));
    CHECK(fit_retry_with_dry_target_fork("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)"));
    // other failures are not retried
    CHECK(!fit_retry_with_dry_target_fork("failed to load model"));
    CHECK(!fit_retry_with_dry_target_fork(""));
    CHECK(!fit_retry_with_dry_target_fork(nullptr));
    if (g_fail) { fprintf(stderr, "test-fit-retry-fork: %d failed\n", g_fail); return 1; }
    printf("test-fit-retry-fork: ok\n");
    return 0;
}
