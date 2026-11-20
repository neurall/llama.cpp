// q8 reuse cache (ggml/src/ggml-cuda/q8reuse-fork.h): pure, no CUDA.
//   clang++ -std=c++17 -I ggml/src/ggml-cuda tests/test-q8reuse-fork.cpp -o test-q8reuse-fork && ./test-q8reuse-fork
#include "q8reuse-fork.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

static q8_reuse_key key(uintptr_t src, int64_t n) {
    q8_reuse_key k; k.src = (const void *) src; k.ne[0] = n; k.ne[1] = k.ne[2] = k.ne[3] = 1; k.s[0] = n; k.s[1] = k.s[2] = n;
    return k;
}

int main() {
    q8_reuse_cache<2, 4> c;
    c.reset();
    bool hit = false;
    // the layer input read by five matvecs: quantized once, then four hits
    const int s0 = c.lookup_or_claim(0, key(1000, 2560), 1000, 1000 + 2560 * 4, hit);
    assert(!hit);
    for (int i = 0; i < 4; ++i) { assert(c.lookup_or_claim(0, key(1000, 2560), 1000, 1000 + 2560 * 4, hit) == s0 && hit); }
    // another shape of the same address is another input
    c.lookup_or_claim(0, key(1000, 512), 1000, 1000 + 512 * 4, hit);
    assert(!hit);
    // a write beside the input keeps it, a write into it drops it
    c.written(1000 + 2560 * 4, 1000 + 2560 * 4 + 64);
    assert(c.lookup_or_claim(0, key(1000, 2560), 1000, 1000 + 2560 * 4, hit) == s0 && hit);
    c.written(1000 + 100, 1000 + 104);
    c.lookup_or_claim(0, key(1000, 2560), 1000, 1000 + 2560 * 4, hit);
    assert(!hit);
    // the other stream lane never sees lane 0's slots
    c.lookup_or_claim(1, key(1000, 2560), 1000, 1000 + 2560 * 4, hit);
    assert(!hit);
    assert(c.lookup_or_claim(1, key(1000, 2560), 1000, 1000 + 2560 * 4, hit) >= 0 && hit);
    // full lane: the least recently used slot is replaced, the hot one stays
    c.reset();
    c.lookup_or_claim(0, key(1, 32), 1, 129, hit);
    for (uintptr_t a = 2; a < 12; ++a) {
        c.lookup_or_claim(0, key(a * 1000, 32), a * 1000, a * 1000 + 128, hit);
        assert(c.lookup_or_claim(0, key(1, 32), 1, 129, hit) >= 0 && hit);   // used between every new input
    }
    // reset drops everything
    c.reset();
    c.lookup_or_claim(0, key(1, 32), 1, 129, hit);
    assert(!hit);
    printf("ok: q8 reuse cache\n");
    return 0;
}
