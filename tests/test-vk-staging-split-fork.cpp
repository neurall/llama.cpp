// Unit test of the Vulkan staging split (ggml/src/ggml-vulkan/vk-staging-split-fork.h). No GPU.
// build: cmake --build <dir> --target test-vk-staging-split-fork;  run: ctest -R test-vk-staging-split-fork
#include "../ggml/src/ggml-vulkan/vk-staging-split-fork.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

struct region { int buf; size_t offset; void * host; size_t size; };

int main() {
    const size_t MiB = 1ull << 20;
    const size_t cap = vk_staging_cap_fork();
    CHECK(cap == 32 * MiB);
    // the 780M prompt save: one KV region of 195.47 MiB asked a staging buffer of its whole size (204963840 bytes, ErrorOutOfDeviceMemory)
    static uint8_t host[1];
    std::vector<region> in = { { 7, 4096, host, 204963840 }, { 7, 0, host + 0, 0 }, { 9, 100, host, 5 * MiB } };
    std::vector<region> out = vk_staging_split_fork(in, cap);
    size_t total = 0, n7 = 0, next_off = 4096;
    uint8_t * next_host = host;
    for (const auto & r : out) {
        CHECK(r.size > 0 && r.size <= cap);
        total += r.size;
        if (r.buf == 7) {
            CHECK(r.offset == next_off && r.host == next_host);
            next_off += r.size; next_host += r.size; n7++;
        }
    }
    CHECK(total == 204963840 + 5 * MiB);
    CHECK(n7 == 7);                                  // 6 x 32 MiB + 3.47 MiB
    CHECK(out.back().buf == 9 && out.back().offset == 100 && out.back().size == 5 * MiB);
    if (g_fail) { fprintf(stderr, "test-vk-staging-split-fork: %d failed\n", g_fail); return 1; }
    printf("test-vk-staging-split-fork: ok\n");
    return 0;
}
