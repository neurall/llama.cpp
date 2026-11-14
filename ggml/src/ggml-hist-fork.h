#pragma once
// fork: the three last-10 histories of a cached element and the admission rule built on them (user 2026-11-14), one copy for every tier: the RAM pool's experts and the
// rows of lazy row tables (ggml-rampool-fork.cpp), the VRAM expert cache (llama-moecache-fork.cpp). Kept also while the element is out, served by seek reads: an element
// that is forgotten when it leaves looks new at every recall, and admitted blindly it pushes out one that pays ("persistent amnesia" thrashes the tier).
//   recall: when it was needed (the tier's clock), its last 10
//   evict:  when it left the tier, its last 10
//   served: the reads it served from the tier in each of its last 10 stays
// Admission: a free block takes anyone. Else the newcomer must expect more reads per stay than the victim (the resident expected to serve fewest): from its stays when
// it had any (one that churns but is read while in pays back), else from its recall rate times the time a resident stays (the tier's turnover). Never seen before:
// expects 0, so it is read from the SSD and only its recall is kept; a second recall soon enough lets it in.

#include <algorithm>
#include <cstdint>

struct hist10 {
    static constexpr int N = 10;
    uint32_t recall[N] = {};   // clocks of its last N recalls; slots [0, n_recall) until the ring is full
    uint32_t evict[N]  = {};   // clocks of its last N evictions; same slots as served
    uint16_t served[N] = {};   // reads served in each of its last N stays
    uint16_t cur       = 0;    // reads served in the stay now
    uint8_t  n_recall  = 0, h_recall = 0;   // entries held (up to N), the next slot
    uint8_t  n_evict   = 0, h_evict  = 0;
    bool     in        = false;
};

// it is needed now: a read served from the tier while in
static inline void hist_recall(hist10 & h, uint32_t clock) {
    h.recall[h.h_recall] = clock;
    h.h_recall = (uint8_t) ((h.h_recall + 1) % hist10::N);
    if (h.n_recall < hist10::N) { h.n_recall++; }
    if (h.in && h.cur < UINT16_MAX) { h.cur++; }
}

// it got a block (after the recall that missed)
static inline void hist_admit(hist10 & h) {
    h.in  = true;
    h.cur = 0;
}

// it left the tier: the stay ends with the reads it served
static inline void hist_evict(hist10 & h, uint32_t clock) {
    h.evict[h.h_evict]  = clock;
    h.served[h.h_evict] = h.cur;
    h.h_evict = (uint8_t) ((h.h_evict + 1) % hist10::N);
    if (h.n_evict < hist10::N) { h.n_evict++; }
    h.in  = false;
    h.cur = 0;
}

// recalls per clock unit over its ring, measured up to now (a burst long ago fades): 0 with fewer than two recalls
static inline double hist_rate(const hist10 & h, uint32_t clock) {
    if (h.n_recall < 2) { return 0.0; }
    const uint32_t oldest = h.n_recall < hist10::N ? h.recall[0] : h.recall[h.h_recall];
    return (double) (h.n_recall - 1) / (double) std::max<uint32_t>(clock - oldest, 1u);
}

// the reads one stay is expected to serve. Out: the mean of its stays when it had any, else its recall rate times the turnover. In: the larger of the two, so a
// resident read now is not judged by poor old stays.
static inline double hist_expect(const hist10 & h, uint32_t clock, double turnover) {
    const double r = hist_rate(h, clock) * turnover;
    if (h.n_evict == 0) { return r; }
    double s = 0.0;
    for (int i = 0; i < h.n_evict; ++i) { s += h.served[i]; }
    const double m = s / h.n_evict;
    return h.in ? std::max(m, r) : m;
}

// may newcomer h take the victim's block: victim_expect < 0 means a free block
static inline bool hist_admit_ok(const hist10 & h, uint32_t clock, double turnover, double victim_expect) {
    return victim_expect < 0.0 || hist_expect(h, clock, turnover) > victim_expect;
}
