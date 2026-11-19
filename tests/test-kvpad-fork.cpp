// How many cells a decode graph reads (src/llama-kvpad-fork.h). Standalone: clang++ -std=c++17 -O1 -I src tests/test-kvpad-fork.cpp -o tkp && ./tkp
#include "llama-kvpad-fork.h"

#include <cstdio>
#include <cstdlib>
#include <set>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t pad256(uint32_t n) { return (n + 255) / 256 * 256; }

int main() {
    // the pad from what a rebuild costs against a token: rebuilds stay under half a percent of the decode time
    CHECK(llama_kv_decode_pad_fork(0.0, 12900.0) == 256);            // nothing measured yet: the old rule
    CHECK(llama_kv_decode_pad_fork(51000.0, 0.0) == 256);
    CHECK(llama_kv_decode_pad_fork(51000.0, 12900.0) == 1024);       // Qwen3.8 one card, 2026-11-19: 51 ms a rebuild, 12.9 ms a token: one per 791 tokens at most
    CHECK(llama_kv_decode_pad_fork(62000.0, 12900.0) == 1024);
    CHECK(llama_kv_decode_pad_fork(100000.0, 55000.0) == 512);       // a slow model: a rebuild is a small share already
    CHECK(llama_kv_decode_pad_fork(300000.0, 166000.0) == 512);
    CHECK(llama_kv_decode_pad_fork(15000.0, 3000.0) == 1024);
    CHECK(llama_kv_decode_pad_fork(900.0, 12900.0) == 256);          // a cheap rebuild: no wider
    CHECK(llama_kv_decode_pad_fork(5e6, 3000.0) == 2048);            // never past the cap

    const uint64_t all = (uint64_t) -1;
    // pad 256 (or nothing measured) is the old rule exactly
    for (uint32_t base = 256; base <= 8192; base += 256) {
        CHECK(llama_kv_n_kv_decode_fork(base, 65536, 256, all) == base);
        CHECK(llama_kv_n_kv_decode_fork(base, 65536, 0, all) == base);
    }
    // a walk of 6000 decode tokens from 12300 cells, every row backed: the graph's cells change once per pad, never below the cells in use, never past the cache
    for (uint32_t P : { 512u, 1024u, 2048u }) {
        std::set<uint32_t> seen;
        uint32_t prev = 0;
        for (uint32_t used = 12301; used <= 18300; ++used) {
            const uint32_t n = llama_kv_n_kv_decode_fork(pad256(used), 65536, P, all);
            CHECK(n >= used && n >= pad256(used) && n % 256 == 0 && n < used + P + 256);
            CHECK(n >= prev);   // it only grows as cells are added
            prev = n;
            seen.insert(n);
            // the mask: from a multiple of 256 at or before the token's cell, the same for every token of a window, P cells wide
            const uint32_t first = used - 1, k0 = llama_kv_decode_k0_fork(first / 256 * 256, n, P);
            CHECK(k0 % 256 == 0 && k0 <= first && n - k0 == P);
        }
        CHECK(seen.size() <= 6000 / P + 2);
    }
    // only rows the zone backs: the horizon stops the wide count, the old rule is the floor (the zone grows before the pass, as before)
    CHECK(llama_kv_n_kv_decode_fork(12544, 65536, 1024, 13000) == 12800);   // 13312 wanted, 13000 rows backed: whole pads of 256 below it
    CHECK(llama_kv_n_kv_decode_fork(12544, 65536, 1024, 12544) == 12544);
    CHECK(llama_kv_n_kv_decode_fork(12544, 65536, 1024, 12000) == 12544);   // fewer rows backed than the old rule reads: the old rule, the zone code grows it
    CHECK(llama_kv_n_kv_decode_fork(12544, 13000, 1024, all) == 13000);     // the cache's size is the last limit
    CHECK(llama_kv_n_kv_decode_fork(4096, 4096, 2048, all) == 4096);
    // the mask of a horizon-bound or short count still starts at or before the token's cell
    CHECK(llama_kv_decode_k0_fork(12288, 12800, 1024) == 11776);
    CHECK(llama_kv_decode_k0_fork(0, 256, 1024) == 0);
    CHECK(llama_kv_decode_k0_fork(768, 1024, 1024) == 0);
    CHECK(llama_kv_decode_k0_fork(12288, 12544, 256) == 12288);             // pad 256: the old start
    CHECK(llama_kv_decode_k0_fork(12032, 13312, 1024) == 12032);            // a tie moved the first cell back: never past it
    printf("ok: decode pad, cells of a decode graph, mask start\n");
    return 0;
}
