// Unit test of the RAM pool's resize (ggml/src/ggml-rampool-fork.cpp: ggml_rampool_grow_fork, ggml_rampool_shrink_fork). No GPU, no model: a file of N elements whose
// bytes name their element, a pool of a quarter of them, then
//   - the misses after a grow land in the new blocks before anything is evicted;
//   - a shrink gives back the last chunk whole, its elements go out, every element still in keeps its own bytes;
//   - an element a reader holds keeps its chunk until it is dropped.
// build: cmake --build <dir> --target test-rampool-fork;  run: ctest -R test-rampool-fork
#include "../ggml/src/ggml-rampool-fork.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

extern "C" size_t ggml_rampool_grow_fork(size_t bytes, ggml_rampool_alloc_t alloc);   // declared where used, as in llama-moecache-fork.cpp
extern "C" size_t ggml_rampool_shrink_fork(size_t bytes, void (*free_chunk)(void * chunk, void * ud));

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

#ifndef _WIN32
static std::vector<void *> g_live;   // chunks handed out and not given back
static void * t_alloc(size_t n, void *) {
    void * p = nullptr;
    if (posix_memalign(&p, 4096, n) != 0) { return nullptr; }
    g_live.push_back(p);
    return p;
}
static void t_free(void * c, void *) {
    for (size_t i = 0; i < g_live.size(); ++i) {
        if (g_live[i] == c) { free(c); g_live.erase(g_live.begin() + (long) i); return; }
    }
    CHECK(!"a chunk given back that was never handed out");
}
static double t_heat(const void *, int64_t e, void *) { return (double) e; }   // element e is expected back at time e: the fill takes the lowest

static const int64_t N   = 64;
static const size_t  ESZ = 3 * 4096 + 100;   // no sector multiple: the reader's edge path too
static int g_key = 0;                        // its address is the tensor's key

static bool holds_own_bytes(int64_t e) {
    const char * p = ggml_rampool_table_fork(&g_key)[e];
    if (p == nullptr) { return false; }
    for (size_t i = 0; i < ESZ; ++i) { if ((uint8_t) p[i] != (uint8_t) (e * 7 + 1)) { return false; } }
    return true;
}
static void need(int64_t a, int64_t b) {   // elements [a, b) for one graph
    std::vector<int32_t> ids;
    for (int64_t e = a; e < b; ++e) { ids.push_back((int32_t) e); }
    const void * keys[1] = { &g_key };
    ggml_rampool_need_fork(keys, 1, ids.data(), (int) ids.size());
}
static void next_graphs() { ggml_rampool_graph_fork(); ggml_rampool_graph_fork(); }   // what the graphs before asked for may go now
static ggml_rampool_stats stats() { ggml_rampool_stats s; ggml_rampool_stats_fork(&s, false); return s; }

int main() {
    char path[] = "/tmp/test-rampool-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) { fprintf(stderr, "no temp file\n"); return 1; }
    unlink(path);
    std::vector<char> el(ESZ);
    for (int64_t e = 0; e < N; ++e) {
        memset(el.data(), (int) (uint8_t) (e * 7 + 1), ESZ);
        if (write(fd, el.data(), ESZ) != (ssize_t) ESZ) { fprintf(stderr, "write failed\n"); return 1; }
    }

    ggml_rampool_add_fork(&g_key, fd, 0, N, ESZ, 0, 0);
    CHECK(ggml_rampool_open_fork((size_t) N / 4 * ESZ, t_alloc, t_heat, nullptr, 2));
    ggml_rampool_evict_fork(true);
    ggml_rampool_fill_fork();
    const ggml_rampool_stats s0 = stats();
    const int64_t n0 = (int64_t) (s0.bytes / ESZ);   // the fill's blocks (a quarter, less the extra block a chunk keeps)
    CHECK(n0 > 0 && n0 < N / 2);
    for (int64_t e = 0; e < n0; ++e) { CHECK(holds_own_bytes(e)); }

    // misses with the pool full: they evict, and give the class its pressure
    next_graphs();
    need(40, 45);
    CHECK(stats().evicted - s0.evicted == 5);
    for (int64_t e = 40; e < 45; ++e) { CHECK(holds_own_bytes(e)); }

    // grow: empty blocks; the next misses take them, nothing is evicted
    const size_t add = ggml_rampool_grow_fork((size_t) 16 * ESZ, t_alloc);
    CHECK(add == (size_t) 16 * ESZ);
    CHECK(stats().bytes == s0.bytes + add);
    CHECK(g_live.size() == 2);
    const uint64_t ev1 = stats().evicted;
    next_graphs();
    need(20, 36);
    CHECK(stats().evicted == ev1);
    for (int64_t e = 20; e < 36; ++e) { CHECK(holds_own_bytes(e)); }
    CHECK(ggml_rampool_grow_fork((size_t) 16 * ESZ, t_alloc) == (size_t) 16 * ESZ);   // pressure again (the 16 reads): one more chunk
    CHECK(g_live.size() == 3);

    // a held element keeps its chunk: the last chunk is empty but for it
    next_graphs();
    const char * held = ggml_rampool_take_fork(&g_key, 50);
    CHECK(held != nullptr);
    next_graphs();
    CHECK(ggml_rampool_shrink_fork(1, t_free) == 0);
    CHECK(g_live.size() == 3);
    ggml_rampool_drop_fork(&g_key, 50);

    // shrink: the last chunk goes whole, its element out; the rest keep their bytes
    const size_t before = (size_t) stats().bytes;
    const size_t got = ggml_rampool_shrink_fork(1, t_free);
    CHECK(got == (size_t) 16 * ESZ);
    CHECK(stats().bytes == before - got);
    CHECK(g_live.size() == 2);
    CHECK(ggml_rampool_table_fork(&g_key)[50] == nullptr);
    for (int64_t e = 20; e < 36; ++e) { CHECK(holds_own_bytes(e)); }
    // and the chunk before it, holding 20..35: recently read elements keep it until two graphs passed
    need(20, 21);
    CHECK(ggml_rampool_shrink_fork(1, t_free) == 0);
    next_graphs();
    CHECK(ggml_rampool_shrink_fork(1, t_free) == (size_t) 16 * ESZ);
    for (int64_t e = 20; e < 36; ++e) { CHECK(ggml_rampool_table_fork(&g_key)[e] == nullptr); }
    // the fill's chunk is left: its elements still read right, a miss evicts again
    int64_t in = 0;
    for (int64_t e = 0; e < N; ++e) { if (ggml_rampool_table_fork(&g_key)[e]) { in++; CHECK(holds_own_bytes(e)); } }
    CHECK(in > 0 && in <= n0);

    ggml_rampool_close_fork();
    for (void * p : g_live) { free(p); }
    close(fd);
    if (g_fail) { fprintf(stderr, "%d checks failed\n", g_fail); return 1; }
    printf("test-rampool-fork: all checks passed\n");
    return 0;
}
#else
int main() { printf("test-rampool-fork: skipped on Windows (POSIX temp file)\n"); return 0; }
#endif
