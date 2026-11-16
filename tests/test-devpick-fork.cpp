// Self-check of the GPUs a model runs on when no -dev is given (src/llama-devpick-fork.h): llama.cpp's device list takes a discrete card once, on the first
// backend that lists it; the later twins (same PCI id) hold nothing, so a feature decided from the GPUs in use must not count them.
// build: cmake --build <dir> --target test-devpick-fork;  run: ctest -R test-devpick-fork
#include "../src/llama-devpick-fork.h"

#include <cstdio>

static int n_failed = 0;
static void check(const char * what, bool ok) {
    printf("  %-90s %s\n", what, ok ? "OK" : "FAILED");
    if (!ok) { n_failed++; }
}

int main() {
    // PC1, the CUDA + Vulkan build (2026-11-16): CUDA0, CUDA1, then Vulkan0, Vulkan1 for the same two RTX 3090s; the Vulkan twins turned the KV tail and the
    // compute alias off for a CUDA run ("the expert cache runs without the KV tail and the compute alias (CUDA-only so far)")
    const std::vector<std::string> pc1 = { "0000:41:00.0", "0000:61:00.0", "0000:41:00.0", "0000:61:00.0" };
    const std::vector<bool> t = llama_dev_twins(pc1);
    check("two cards on CUDA then Vulkan: the CUDA devices hold the model", !t[0] && !t[1]);
    check("two cards on CUDA then Vulkan: both Vulkan devices are twins", t[2] && t[3]);
    // a driver without a PCI id is never a twin (llama.cpp compares ids only when both have one)
    const std::vector<bool> u = llama_dev_twins({ "", "", "0000:41:00.0" });
    check("devices without an id are never twins", !u[0] && !u[1] && !u[2]);
    // a Vulkan-only build of one card plus another card: nothing to skip
    const std::vector<bool> v = llama_dev_twins({ "0000:03:00.0", "0000:04:00.0" });
    check("two different cards on one backend: no twins", !v[0] && !v[1]);
    printf("%s\n", n_failed ? "FAILED" : "all passed");
    return n_failed ? 1 : 0;
}
