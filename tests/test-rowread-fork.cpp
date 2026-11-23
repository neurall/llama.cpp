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

// two read-aheads: the newer one first, the older one fills only what the newer lacks
static void check2(const std::vector<int32_t> & a, const std::vector<int32_t> & b, const std::vector<int32_t> & need, const std::vector<size_t> & want_miss) {
    const size_t rs = 5;
    const std::vector<uint8_t> da = rows_of(a, rs), db = rows_of(b, rs), full = rows_of(need, rs);
    std::vector<uint8_t> dst(need.size()*rs, 0xEE);
    std::vector<size_t> rest(need.size());
    for (size_t k = 0; k < rest.size(); ++k) { rest[k] = k; }
    rest = rowread_take_rest(a, da.data(), need, rest, rs, dst.data());
    rest = rowread_take_rest(b, db.data(), need, rest, rs, dst.data());
    assert(rest == want_miss);
    size_t m = 0;
    for (size_t k = 0; k < need.size(); ++k) {
        const bool missing = m < rest.size() && rest[m] == k;
        for (size_t c = 0; c < rs; ++c) { assert(dst[k*rs + c] == (missing ? 0xEE : full[k*rs + c])); }
        m += missing;
    }
}

int main() {
    check2({ 4, 9 }, { 1, 4, 7 }, { 1, 4, 7, 9 }, {});           // the next-token job holds some rows, the pass's own job the rest
    check2({ 2 }, { 3 }, { 1, 2, 3, 8 }, { 0, 3 });              // neither holds 1 or 8: read now
    check2({}, {}, { 5 }, { 0 });
    assert(rowread_overlap({ 1, 5, 9 }, { 2, 9 }) && !rowread_overlap({ 1, 5 }, { 2, 6 }) && !rowread_overlap({}, { 1 }));
    check({ 1, 5, 9 }, { 1, 5, 9 }, {});                          // the read-ahead holds the pass
    check({}, { 2, 3 }, { 0, 1 });                                // none read ahead
    check({ 1, 2, 3, 4, 5, 6 }, { 2, 4, 6 }, {});                 // more read ahead than the pass takes
    check({ 1, 4, 8, 20 }, { 0, 4, 5, 8, 30 }, { 0, 2, 4 });      // interleaved: before, between and after its rows
    check({ 10, 11 }, { 3, 4 }, { 0, 1 });                        // disjoint
    check({ 3 }, {}, {});                                         // an empty pass
    printf("test-rowread-fork: all passed\n");
    return 0;
}
