#pragma once
// fork: the pure parts of the lazy row tables (llama-rowread-fork.cpp), here so tests/test-rowread-fork.cpp checks them without the engine.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// rows u (sorted, distinct) into dst (row k at dst + k*rs) from a read-ahead: ju sorted and distinct, row j's bytes at jd + j*rs. Copies every row the read-ahead
// holds and returns the indexes k into u of the rows it lacks, ascending (the caller reads those itself).
static inline std::vector<size_t> rowread_take(const std::vector<int32_t> & ju, const uint8_t * jd, const std::vector<int32_t> & u, size_t rs, uint8_t * dst) {
    std::vector<size_t> miss;
    size_t j = 0;
    for (size_t k = 0; k < u.size(); ++k) {
        while (j < ju.size() && ju[j] < u[k]) { ++j; }
        if (j < ju.size() && ju[j] == u[k]) { memcpy(dst + k*rs, jd + j*rs, rs); } else { miss.push_back(k); }
    }
    return miss;
}
