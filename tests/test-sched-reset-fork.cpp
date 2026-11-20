// The scheduler's sparse table reset (ggml/src/ggml-sched-reset-fork.h) against the whole-table clear it replaces.
// Standalone: clang++ -std=c++17 -O1 -I ggml/src tests/test-sched-reset-fork.cpp -o tsr && ./tsr
#include "ggml-sched-reset-fork.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

int main() {
    std::mt19937 rng(7);
    for (size_t size : { (size_t) 1, (size_t) 31, (size_t) 32, (size_t) 33, (size_t) 1021, (size_t) 65537 }) {
        for (size_t per : { (size_t) 1, (size_t) 3, (size_t) 8 }) {
            for (int fill : { 0, 1, 7, 100 }) {   // percent of the slots a graph marked
                std::vector<uint32_t> used((size + 31) / 32, 0);
                std::vector<int>      ids(size, -1);
                std::vector<void *>   copies(size * per, nullptr);
                std::vector<char>     flags_c(size, 0);
                bool * flags = (bool *) flags_c.data();
                size_t n_marked = 0;
                for (size_t i = 0; i < size; ++i) {
                    if ((int) (rng() % 100) >= fill) { continue; }
                    used[i / 32] |= 1u << (i % 32);   // as hash_id does, then the tables are written
                    n_marked++;
                    ids[i] = (int) (rng() % 4);
                    for (size_t k = 0; k < per; ++k) { if (rng() % 2) { copies[i * per + k] = (void *) (uintptr_t) (rng() | 1u); } }
                    flags[i] = rng() % 2;
                }
                CHECK(fill != 100 || n_marked == size);
                ggml_sched_reset_used_fork(used.data(), size, ids.data(), copies.data(), per, flags);
                for (uint32_t w : used) { CHECK(w == 0); }
                for (size_t i = 0; i < size; ++i) {
                    CHECK(ids[i] == -1 && !flags[i]);
                    for (size_t k = 0; k < per; ++k) { CHECK(copies[i * per + k] == nullptr); }
                }
            }
        }
    }
    printf("ok: sparse scheduler reset equals the whole clear\n");
    return 0;
}
