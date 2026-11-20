// Unit test of the RAM pool's resize (ggml/src/ggml-rampool-fork.cpp: ggml_rampool_grow_fork, ggml_rampool_shrink_fork). No GPU, no model: a file of N elements whose
// bytes name their element, a pool of a quarter of them, then
//   - the misses after a grow land in the new blocks before anything is evicted;
//   - a shrink gives back the last chunk whole, its elements go out, every element still in keeps its own bytes;
//   - an element a reader holds keeps its chunk until it is dropped.
// build: cmake --build <dir> --target test-rampool-fork;  run: ctest -R test-rampool-fork
#include "../ggml/src/ggml-rampool-fork.h"
#include "../ggml/src/ggml-rampool-tune-fork.h"

#include <algorithm>
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

// rows of lazy row tables (PLE, Engram) as pool entries with their histories (ggml-rampool-fork.cpp, rows part)
extern "C" void   ggml_rampool_rows_add_fork(const void * key, size_t rs);
extern "C" size_t ggml_rampool_rows_get_fork(const void * key, const int32_t * rows, size_t n, uint8_t * dst, uint8_t * hit);
extern "C" void   ggml_rampool_rows_put_fork(const void * key, const int32_t * rows, size_t n, const uint8_t * src);
extern "C" void   ggml_rampool_rows_has_fork(const void * key, const int32_t * rows, size_t n, uint8_t * in);
extern "C" void   ggml_rampool_rows_stats_fork(const void * key, uint64_t * v, double * turnover);

static uint8_t row_byte(int32_t r, size_t i) { return (uint8_t) (r * 7 + (int32_t) i); }

// one call of the fill: get, read the misses "from the file" (bytes named by row), put them. Returns the hits; *bad counts hit rows with wrong bytes.
static size_t rows_call(const void * key, size_t rs, std::vector<int32_t> rows, int * bad) {
    std::sort(rows.begin(), rows.end());
    std::vector<uint8_t> dst(rows.size() * rs), hit(rows.size());
    const size_t nh = ggml_rampool_rows_get_fork(key, rows.data(), rows.size(), dst.data(), hit.data());
    std::vector<int32_t> mr;
    std::vector<uint8_t> mb;
    for (size_t k = 0; k < rows.size(); ++k) {
        if (hit[k]) {
            for (size_t i = 0; i < rs; ++i) { if (dst[k * rs + i] != row_byte(rows[k], i)) { (*bad)++; break; } }
            continue;
        }
        mr.push_back(rows[k]);
        for (size_t i = 0; i < rs; ++i) { mb.push_back(row_byte(rows[k], i)); }
    }
    ggml_rampool_rows_put_fork(key, mr.data(), mr.size(), mb.data());
    return nh;
}

// the fill's thread count from its windows (rp_tune_step): Kimi K3.1's drive on PC1, 2026-11-20, as a rate by threads that keeps rising past the probe's pick:
// 4 threads 4.58 GB/s (the dev fill), 16 threads 5.34 (b15843's fill), flat after that
static double k3_fill_gbs(int n) { return n <= 4 ? 4.58 : n <= 8 ? 5.05 : n <= 16 ? 5.34 : 5.36; }
static void test_tune() {
    rp_thread_tune t; t.n = 4;
    for (int w = 0; w < 10 && !t.settled; ++w) { t = rp_tune_step(t, k3_fill_gbs(t.n), 64); }
    CHECK(t.settled && t.n == 16);   // 4 -> 8 (+10%) -> 16 (+6%) -> 32 (+0.4%: back to 16, b15843's count)
    rp_thread_tune f; f.n = 4;   // a drive that is full at 4 threads: one doubling tried, taken back
    for (int w = 0; w < 10 && !f.settled; ++w) { f = rp_tune_step(f, f.n <= 4 ? 6.27 : 6.30, 64); }
    CHECK(f.settled && f.n == 4);
    rp_thread_tune b; b.n = 4;   // more threads read slower (contention): back to the count before the doubling
    for (int w = 0; w < 10 && !b.settled; ++w) { b = rp_tune_step(b, b.n <= 4 ? 6.0 : 5.0, 64); }
    CHECK(b.settled && b.n == 4);
    rp_thread_tune c; c.n = 32;   // the cap stops it
    for (int w = 0; w < 10 && !c.settled; ++w) { c = rp_tune_step(c, (double) c.n, 64); }
    CHECK(c.settled && c.n == 64);
}

// the pool's start and grows against the RAM a request needed beside it (rp_start_pick, rp_grow_cap, rp_need_now): Kimi K3.1 on PC1, 2026-11-20 (b16041 gainers)
static size_t gib(double g) { return (size_t) (g * 1073741824.0); }
static bool near(size_t a, double g) { const double d = (double) a / 1073741824.0 - g; return d > -0.01 && d < 0.01; }
static void test_start() {
    // the failure: short requests had grown the pool and stored 12.05 GiB kept; the next start took 229.6 GiB (the equation gave 217.0), its 12.5k-token
    // prompt needed 13.3 GiB beside the pool and RAM available fell to 1.5 GiB. No need measured yet: the start stays at the equation
    rp_size_rule r { gib(217.0), gib(241.7), gib(12.05), 0, gib(5.16) };
    CHECK(near(rp_start_pick(r), 217.0));
    // the need of the dev rows' 12.5k prompt at the equation's pool: 241.7 available, pool 217.7, at least 10.73 left
    CHECK(near(rp_need_now(gib(241.7), gib(217.7), gib(10.73)), 13.27));
    CHECK(rp_need_now(gib(241.7), gib(217.7), gib(30.0)) == 0);   // other processes gave RAM back: no need below zero
    // measured: the start keeps it and the grow floor free, above the equation where that leaves room
    r.need = gib(13.27);
    CHECK(near(rp_start_pick(r), 241.7 - 13.27 - 5.16));
    r.kept = gib(25.0);   // the last session left more beside its pool (it gave chunks back): that start
    CHECK(near(rp_start_pick(r), 241.7 - 25.0));
    r.kept = 0;           // no session stored one: the equation
    CHECK(near(rp_start_pick(r), 217.0));
    r.kept = gib(250.0);  // less RAM available than the room to keep: the equation, as before
    CHECK(near(rp_start_pick(r), 217.0));
    // a grow after short requests stops where the measured need and the floor stay free; with no need measured it is the grow's own
    CHECK(near(rp_grow_cap(gib(2.0), gib(241.7), gib(241.7 - 13.27 - 5.16), gib(13.27), gib(5.16)), 0.0));
    CHECK(near(rp_grow_cap(gib(2.0), gib(241.7), gib(217.0), gib(13.27), gib(5.16)), 2.0));
    CHECK(near(rp_grow_cap(gib(9.0), gib(241.7), gib(217.0), gib(13.27), gib(5.16)), 241.7 - 13.27 - 5.16 - 217.0));
    CHECK(rp_grow_cap(gib(2.0), gib(241.7), gib(230.0), 0, gib(5.16)) == gib(2.0));
}

static void test_rows() {
    // a pass's rows: none in RAM the first time, all the second time (free blocks take anyone), each with its own bytes
    {
        static int key = 0;
        const size_t rs = 90;
        ggml_rampool_rows_add_fork(&key, rs);
        std::vector<int32_t> rows;
        for (int32_t r = 0; r < 100; ++r) { rows.push_back(r * 1000003 % 320000000); }
        int bad = 0;
        CHECK(rows_call(&key, rs, rows, &bad) == 0);
        CHECK(rows_call(&key, rs, rows, &bad) == 100);
        CHECK(bad == 0);
        std::vector<uint8_t> in(rows.size());
        std::sort(rows.begin(), rows.end());
        ggml_rampool_rows_has_fork(&key, rows.data(), rows.size(), in.data());
        for (uint8_t x : in) { CHECK(x == 1); }
        uint64_t v[10]; double turn = 0;
        ggml_rampool_rows_stats_fork(&key, v, &turn);
        CHECK(v[0] == 100 && v[2] == 1000 && v[3] == 100 && v[4] == 100 && v[5] == 100);
        // admissions per row recalled, by rows: 100 of 100, then 100 hits (the first faded by e^-0.1): a stay lasts about 2105 rows recalled
        CHECK(turn > 2000 && turn < 2200);
        // 50 decode-sized calls of 2 rows that all hit weigh 100 rows, not 50 calls: the stay grows only to about 3000
        std::vector<int32_t> two(rows.begin(), rows.begin() + 2);
        for (int c = 0; c < 50; ++c) { rows_call(&key, rs, two, &bad); }
        ggml_rampool_rows_stats_fork(&key, v, &turn);
        CHECK(turn > 2500 && turn < 3500);
    }
    // amnesia: the tier is full of rows read once. A hot set that starts later is declined at its first recall (nothing known) but remembered, so its second
    // recall gets it in over the one-offs (they served nothing), and from then on it hits. One-offs never hit. Calls of 300 rows: capacity 3000.
    {
        static int key = 0;
        const size_t rs = 144;
        ggml_rampool_rows_add_fork(&key, rs);
        int32_t next = 100000000;
        size_t hot_hits = 0, other_hits = 0;
        int bad = 0;
        for (int c = 0; c < 40; ++c) {
            std::vector<int32_t> rows;
            const bool hot = c >= 20;
            if (hot) { for (int32_t r = 0; r < 50; ++r) { rows.push_back(r); } }
            while (rows.size() < 300) { rows.push_back(next++); }
            std::vector<int32_t> h(rows.begin(), rows.begin() + (hot ? 50 : 0));
            std::sort(h.begin(), h.end());
            std::vector<uint8_t> in_before(h.size());
            ggml_rampool_rows_has_fork(&key, h.data(), h.size(), in_before.data());
            const size_t nh = rows_call(&key, rs, rows, &bad);
            size_t hh = 0;
            for (uint8_t x : in_before) { hh += x; }
            hot_hits += hh;
            other_hits += nh - hh;
            if (c == 20 || c == 21) { CHECK(hh == 0); }   // first recall declined, second admitted (a miss that is read and kept)
        }
        CHECK(bad == 0);
        CHECK(hot_hits == 50 * 18);   // calls 22..39
        CHECK(other_hits == 0);
        uint64_t v[10]; double turn = 0;
        ggml_rampool_rows_stats_fork(&key, v, &turn);
        CHECK(v[2] == 3000 && v[0] == 3000);
        printf("rows, amnesia check: hot hits %zu of %d, one-off hits %zu, in RAM %llu of %llu, entries %llu, admitted %llu, declined %llu, evicted %llu, forgotten %llu, %.1f KiB\n",
               hot_hits, 50 * 18, other_hits, (unsigned long long) v[0], (unsigned long long) v[2], (unsigned long long) v[1], (unsigned long long) v[5],
               (unsigned long long) v[6], (unsigned long long) v[7], (unsigned long long) v[8], v[9] / 1024.0);
    }
}

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

    test_rows();
    test_tune();
    test_start();
    if (g_fail) { fprintf(stderr, "%d checks failed\n", g_fail); return 1; }
    printf("test-rampool-fork: all checks passed\n");
    return 0;
}
#else
int main() { printf("test-rampool-fork: skipped on Windows (POSIX temp file)\n"); return 0; }
#endif
