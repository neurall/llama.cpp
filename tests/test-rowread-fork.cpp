// fork: the pure parts of the lazy row tables (src/llama-rowread-fork.h): a pass's rows taken from a read-ahead, the rest named for a read of their own.
// g++ -std=c++17 -O1 tests/test-rowread-fork.cpp -o test-rowread-fork && ./test-rowread-fork
#include "../src/llama-rowread-fork.h"

#include <cassert>
#include <cstdio>
#include <vector>

static std::vector<uint8_t> rows_of(const std::vector<int32_t> & ids, size_t rs) {   // row r's bytes: r's low byte + the byte's index
    std::vector<uint8_t> d(ids.size()*rs);
    for (size_t k = 0; k < ids.size(); ++k) { for (size_t b = 0; b < rs; ++b) { d[k*rs + b] = (uint8_t) (ids[k] + (int32_t) b); } }
    return d;
}

static void check(const std::vector<int32_t> & ahead, const std::vector<int32_t> & need, const std::vector<size_t> & want_miss) {
    const size_t rs = 7;
    const std::vector<uint8_t> jd = rows_of(ahead, rs);
    const std::vector<uint8_t> full = rows_of(need, rs);
    std::vector<uint8_t> dst(need.size()*rs, 0xEE);
    const std::vector<size_t> miss = rowread_take(ahead, jd.data(), need, rs, dst.data());
    assert(miss == want_miss);
    size_t m = 0;
    for (size_t k = 0; k < need.size(); ++k) {
        const bool missing = m < miss.size() && miss[m] == k;
        for (size_t b = 0; b < rs; ++b) { assert(dst[k*rs + b] == (missing ? 0xEE : full[k*rs + b])); }   // a missing row is left for its own read
        m += missing;
    }
}

int main() {
    check({ 1, 5, 9 }, { 1, 5, 9 }, {});                          // the read-ahead holds the pass
    check({}, { 2, 3 }, { 0, 1 });                                // none read ahead
    check({ 1, 2, 3, 4, 5, 6 }, { 2, 4, 6 }, {});                 // more read ahead than the pass takes
    check({ 1, 4, 8, 20 }, { 0, 4, 5, 8, 30 }, { 0, 2, 4 });      // interleaved: before, between and after its rows
    check({ 10, 11 }, { 3, 4 }, { 0, 1 });                        // disjoint
    check({ 3 }, {}, {});                                         // an empty pass
    printf("test-rowread-fork: all passed\n");
    return 0;
}
