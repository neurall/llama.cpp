// The decode bell: a card and the host meet once per MoE layer without a driver call (fork/docs/decode-doorbell-plan-2026-11-18.md).
//
// A decode layer of the MoE cache needs the host for two things: to see the router's picks, and to compute the picks that have no block on the card. Through the
// scheduler that was two graph launches and two stream waits a layer (nsys, Qwen3.8 IQ3_S, build b15843: 104 launches and 122 waits a token, 5.1 ms of a 19.2 ms
// token with no card computing). With a bell the token is one launch per card:
//   card   ggml_bell_publish    the picks, and the block each has on the card (< 0: none), go into the layer's bell; the layer input too when a pick has no
//                               block; then seq counts up
//   host   ggml_bell_host_wait  spins until seq moved; ggml_bell_host_misses names the picks to compute (exactly those the card called missing, so the two
//                               never disagree, whatever a table update is doing); the host writes their rows and calls ggml_bell_host_done
//   card   ggml_bell_rows_take  nothing missing: returns at once, the host is not waited for. Else spins until the flag reaches seq and takes the rows
// A bell lives in host memory the card reads and writes by pointer (CUDA: mapped pinned memory); on the CPU backend it is plain memory. This header is the
// protocol and its CPU reference (tests/test-bell-fork.cpp); a backend's kernels do the card's two steps the same way.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define GGML_BELL_PAUSE() _mm_pause()
#else
#define GGML_BELL_PAUSE() ((void) 0)
#endif

#define GGML_BELL_MAX_USED 32   // picks per token (Qwen3.8 Next 10, Kimi K3 16)

struct ggml_bell {
    volatile uint32_t seq;      // the card: counted up once the picks of a token are written
    uint32_t          pad0[15]; // seq and flag are written by different sides: a cache line each
    volatile uint32_t flag;     // the host: the seq whose rows are ready
    uint32_t          pad1[15];
    int32_t           n_used;   // set once (ggml_bell_init)
    int32_t           n_embd;
    volatile int32_t  n_miss;   // the card: picks of this token without a block
    int32_t           pad2;
    int32_t           ids[GGML_BELL_MAX_USED];     // the card: the picks
    int32_t           slots[GGML_BELL_MAX_USED];   // the card: the block of each pick, < 0 when it has none
    // float x[n_embd]            the card: the layer input, written when n_miss > 0
    // float rows[n_used*n_embd]  the host: row i is pick i's output, written for the missing picks
};

static inline size_t ggml_bell_size(int32_t n_used, int32_t n_embd) {
    return sizeof(struct ggml_bell) + sizeof(float) * ((size_t) n_embd + (size_t) n_used * (size_t) n_embd);
}
static inline float * ggml_bell_x(struct ggml_bell * b)    { return (float *) (b + 1); }
static inline float * ggml_bell_rows(struct ggml_bell * b) { return (float *) (b + 1) + b->n_embd; }

static inline struct ggml_bell * ggml_bell_init(void * mem, int32_t n_used, int32_t n_embd) {
    struct ggml_bell * b = (struct ggml_bell *) mem;
    memset(mem, 0, ggml_bell_size(n_used, n_embd));
    b->n_used = n_used;
    b->n_embd = n_embd;
    return b;
}

// --- the card's side (CPU reference; a backend's kernel does the same) ---

// table[e]: expert e's block on this card, < 0 when it has none. A negative pick (deferred, skipped) is nobody's: no block, and not missing.
// slots_out gets each pick's block. Returns the number of missing picks.
static inline int ggml_bell_publish(struct ggml_bell * b, const int32_t * ids, const int32_t * table, const float * x, int32_t * slots_out) {
    int n_miss = 0;
    for (int32_t i = 0; i < b->n_used; ++i) {
        const int32_t s = ids[i] >= 0 ? table[ids[i]] : -1;
        slots_out[i] = s < 0 ? -1 : s;
        b->ids[i]    = ids[i];
        b->slots[i]  = slots_out[i];
        n_miss += ids[i] >= 0 && s < 0;
    }
    if (n_miss > 0) { memcpy(ggml_bell_x(b), x, sizeof(float) * (size_t) b->n_embd); }
    b->n_miss = n_miss;
    b->seq    = b->seq + 1u;   // last: everything above is in place when the host sees it
    return n_miss;
}

static inline int ggml_bell_rows_ready(const struct ggml_bell * b) { return b->n_miss == 0 || b->flag == b->seq; }

// down: [n_embd, n_used], the card's own output; the rows of the missing picks are taken from the host. Returns how many.
static inline int ggml_bell_rows_take(struct ggml_bell * b, float * down) {
    if (b->n_miss == 0) { return 0; }
    while (b->flag != b->seq) { GGML_BELL_PAUSE(); }
    int n = 0;
    for (int32_t i = 0; i < b->n_used; ++i) {
        if (b->ids[i] >= 0 && b->slots[i] < 0) {
            memcpy(down + (size_t) i * (size_t) b->n_embd, ggml_bell_rows(b) + (size_t) i * (size_t) b->n_embd, sizeof(float) * (size_t) b->n_embd);
            n++;
        }
    }
    return n;
}

// --- the host's side ---

// spins until the card has written the next token's picks (seq differs from seen). 0: not within timeout_us (the card's graph died or never ran).
static inline int ggml_bell_host_wait(const struct ggml_bell * b, uint32_t seen, int64_t timeout_us) {
    struct timespec t0; int have_t0 = 0;
    for (uint32_t spins = 0; b->seq == seen; ++spins) {
        GGML_BELL_PAUSE();
        if ((spins & 0xffffu) == 0xffffu) {   // the clock only now and then: the wait is a few microseconds when all is well
            struct timespec t; timespec_get(&t, TIME_UTC);
            if (!have_t0) { t0 = t; have_t0 = 1; }
            if ((int64_t) (t.tv_sec - t0.tv_sec) * 1000000 + (t.tv_nsec - t0.tv_nsec) / 1000 > timeout_us) { return 0; }
        }
    }
    return 1;
}

// miss_ids[i]: pick i when the host has to compute it, else -1. Returns how many.
static inline int ggml_bell_host_misses(const struct ggml_bell * b, int32_t * miss_ids) {
    int n = 0;
    for (int32_t i = 0; i < b->n_used; ++i) {
        const int miss = b->ids[i] >= 0 && b->slots[i] < 0;
        miss_ids[i] = miss ? b->ids[i] : -1;
        n += miss;
    }
    return n;
}

// the rows of the token with this seq are written
static inline void ggml_bell_host_done(struct ggml_bell * b, uint32_t seq) { b->flag = seq; }
