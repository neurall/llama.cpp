#pragma once
// fork (pool v2, user 2026-11-22: "make map sync cheap as our bell mech was"): the expert tables (expert -> pool offset) reach the card without a copy call.
// The host keeps what it last sent per table and appends only the entries that changed to a ring in pinned, device-mapped memory; one small kernel on the
// card's stream applies the ring up to the host's head and records how far it got (the tail lives in the same mapped page, so the host sees it without a
// sync). A ring too full for the change list falls back to the whole-table copy the engine does today. Pure host logic here; the kernel is in ggml-cuda.

#include <cstdint>
#include <cstddef>
#include <vector>

struct mapsync_entry { uint32_t tbl; int32_t idx; int32_t val; uint32_t pad; };   // 16 bytes: tbl = the table's slot in the card's table list

struct mapsync_ring {
    mapsync_entry * e    = nullptr;   // cap entries, device-mapped
    uint64_t      * head = nullptr;   // written by the host (entries appended so far), read by the kernel
    uint64_t      * tail = nullptr;   // written by the kernel (entries applied so far), read by the host
    uint64_t        cap  = 0;
};

// entries of cur that differ from sent (both n long); sent becomes cur
static inline void mapsync_diff(uint32_t tbl, const int32_t * cur, int32_t * sent, size_t n, std::vector<mapsync_entry> & out) {
    for (size_t i = 0; i < n; ++i) {
        if (cur[i] != sent[i]) { out.push_back({ tbl, (int32_t) i, cur[i], 0 }); sent[i] = cur[i]; }
    }
}

// append es to the ring; false: not enough room (the caller copies the tables whole, sets *head = *tail after the copy so the ring restarts empty)
static inline bool mapsync_push(mapsync_ring & r, const std::vector<mapsync_entry> & es) {
    const uint64_t h = *r.head, t = *(volatile uint64_t *) r.tail;
    if (h - t + es.size() > r.cap) { return false; }
    for (size_t i = 0; i < es.size(); ++i) { r.e[(h + i) % r.cap] = es[i]; }
    __atomic_store_n(r.head, h + es.size(), __ATOMIC_RELEASE);   // entries before the head
    return true;
}

// the kernel's work on the CPU (the reference and the test): apply [tail, h) to the tables. h is the head when the kernel was QUEUED (a launch argument):
// read when the kernel runs, a kernel queued before graph N applied graph N+1's entries too, whose experts were still uploading (GLM 3.5 NaN, 2026-11-22)
static inline void mapsync_apply_ref(mapsync_ring & r, int32_t * const * tables, uint64_t h) {
    for (uint64_t i = *r.tail; i < h; ++i) { const mapsync_entry & m = r.e[i % r.cap]; tables[m.tbl][m.idx] = m.val; }
    __atomic_store_n(r.tail, h, __ATOMIC_RELEASE);
}
