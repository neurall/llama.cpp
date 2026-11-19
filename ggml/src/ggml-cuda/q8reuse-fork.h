// fork: reuse of a quantized activation (q8_1) between the matvecs of one graph that read the same f32 input.
//
// Every MUL_MAT / MUL_MAT_ID on the matvec path (mmvq.cu) quantizes its f32 input before its kernel. A decode layer of Qwen3.8 reads the same input
// (hc_mixed) in five quantized matvecs, so the same bytes are written five times (trace 2026-11-19: 381 quantize kernels a token, Strata 145).
// The quantized copy of an input stays valid until a node writes memory that overlaps the input: then its slot is dropped.
//
// Pure, no CUDA: tests/test-q8reuse-fork.cpp. Slots are per stream lane (a two-stream graph, ggml-streams-fork.h): a slot written on one stream
// is never read or overwritten from the other, so no wait between the streams is needed.
#pragma once

#include <cstdint>
#include <cstring>

struct q8_reuse_key {
    const void * src = nullptr;
    int64_t ne[4] = {0, 0, 0, 0};
    int64_t s[3]  = {0, 0, 0};   // strides of the f32 input in elements
    bool operator==(const q8_reuse_key & o) const {
        return src == o.src && memcmp(ne, o.ne, sizeof ne) == 0 && memcmp(s, o.s, sizeof s) == 0;
    }
};

template <int N_LANES, int N_SLOTS>
struct q8_reuse_cache {
    struct slot { q8_reuse_key k; uintptr_t lo = 0, hi = 0; uint32_t age = 0; bool valid = false; };
    slot     slots[N_LANES][N_SLOTS];
    uint32_t clock = 0;

    void reset() {
        for (auto & l : slots) { for (auto & s : l) { s.valid = false; } }
        clock = 0;
    }

    // returns the slot index; hit = the slot holds the quantized copy of exactly this input
    int lookup_or_claim(int lane, const q8_reuse_key & k, uintptr_t lo, uintptr_t hi, bool & hit) {
        slot * l = slots[lane];
        int victim = 0;
        for (int i = 0; i < N_SLOTS; ++i) {
            if (l[i].valid && l[i].k == k) { l[i].age = ++clock; hit = true; return i; }
        }
        for (int i = 0; i < N_SLOTS; ++i) {
            if (!l[i].valid) { victim = i; break; }
            if (l[i].age < l[victim].age) { victim = i; }
        }
        l[victim].k = k; l[victim].lo = lo; l[victim].hi = hi; l[victim].age = ++clock; l[victim].valid = true;
        hit = false;
        return victim;
    }

    // a node wrote [lo, hi): every slot whose input overlaps it is stale
    void written(uintptr_t lo, uintptr_t hi) {
        for (auto & l : slots) {
            for (auto & s : l) {
                if (s.valid && s.lo < hi && lo < s.hi) { s.valid = false; }
            }
        }
    }
};
