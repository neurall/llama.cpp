// Unit test of the host room rule (src/llama-hostroom-fork.h). No GPU, no model.
// build: cmake --build <dir> --target test-hostroom-fork;  run: ctest -R test-hostroom-fork
#include "../src/llama-hostroom-fork.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    const double GiB = 1073741824.0;
    const auto gib = [&](double g) { return (size_t) (g * GiB); };
    // W10 2026-11-26 (RTX 4060 8 GB, Inkling 2 Mini 14.7 GiB): 22.2 GiB RAM free, 21.8 GiB of commit left (25.5 of 47.3 in use, the page file unable to grow),
    // 6.9 GiB of free VRAM. Free RAM alone pinned all 14.7 GiB (14.7 + the 4.4 GiB margin < 22.2) and left 6.3 GiB of commit for the 3.3 GiB KV arena, the
    // graphs and CUDA: "bad allocation" at the first decode. With the devices' commit taken off, the room no longer holds the whole model pinned.
    const size_t room = host_room_fork_calc(gib(22.2), gib(21.8), SIZE_MAX, gib(6.9));
    CHECK(room > gib(14.8) && room < gib(15.0));
    const size_t margin = gib(4.4), model = gib(14.7);
    CHECK(model + margin > room);              // not pinned whole: the rest goes to the RAM pool / mapping
    CHECK(room - margin < model);              // the pin budget (room less the margin) is below the model
    CHECK(gib(21.8) - (room - margin) > gib(6.9));   // the commit left after the pin holds what the devices still take
    // free RAM alone, the old reading: the whole model pinned
    CHECK(host_room_fork_calc(gib(22.2), SIZE_MAX, SIZE_MAX) == gib(22.2));
    // a job's limit is a ceiling as well, and the devices' share comes off it too
    CHECK(host_room_fork_calc(gib(22.2), gib(21.8), gib(10.0), gib(2.0)) == gib(8.0));
    // no device term and plenty of commit: free RAM rules (Linux readers never pass a commit)
    CHECK(host_room_fork_calc(gib(5.0), gib(40.0), SIZE_MAX) == gib(5.0));
    // devices that take more than the commit left: no room, never a wrapped huge number
    CHECK(host_room_fork_calc(gib(22.2), gib(3.0), SIZE_MAX, gib(6.9)) == 0);
    CHECK(host_room_fork_calc(gib(22.2), gib(21.8), gib(1.0), gib(6.9)) == 0);
    if (g_fail) { fprintf(stderr, "test-hostroom-fork: %d failed\n", g_fail); return 1; }
    printf("test-hostroom-fork: ok\n");
    return 0;
}
