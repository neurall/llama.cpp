// fork: the dev event log. One in-memory ring of timestamped spans for the whole process (scheduler, backends, cards, loader, pools, planner, server), so a dev
// run shows where time goes without nsys: host spans (start, end, name, two numbers such as bytes or tokens), GPU spans (a card's graph from CUDA events,
// mapped onto the host clock) and system samples (MemAvailable, RSS, major faults, process CPU) on one timeline.
//
// Switch: GGML_EVLOG=all or a list of categories (sched,graph,copy,sync,gpu,upload,rampool,loader,planner,pool,decode,server,sys); unset = off, and every
// trace point then costs one load of ggml_evlog_mask. run.py sets it for dev runs, never for benchmarks or users.
// GGML_EVLOG_FILE: where the flusher thread appends the records every 500 ms (default /dev/shm/evlog-<pid>.tsv; RAM, not the model drive).
// GGML_EVLOG_N: ring size in records (default 1M = 48 MB). A record lapped before the flusher wrote it is counted as lost in the file.
// File: a header line with the clock pair (mono_ns real_ns) so other samplers (run.py's GPU/PCIe log) line up, then one line a record:
//   t0_ns  t1_ns  cat  name  tid  a0  a1        (mono ns; t1 == t0 for an instant record)
// Names must be string literals (the ring stores the pointer). No trace point in a per-node or per-kernel path: per graph, per split, per copy, per request.
#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_evlog_cat {
    GGML_EV_SCHED = 0, GGML_EV_GRAPH, GGML_EV_COPY, GGML_EV_SYNC, GGML_EV_GPU, GGML_EV_UPLOAD, GGML_EV_RAMPOOL, GGML_EV_LOADER, GGML_EV_PLANNER,
    GGML_EV_POOL, GGML_EV_DECODE, GGML_EV_SERVER, GGML_EV_SYS, GGML_EV_N_CAT
};

GGML_API volatile uint32_t ggml_evlog_mask;   // bit per category; 0 = off

GGML_API uint64_t ggml_evlog_now(void);              // CLOCK_MONOTONIC ns
GGML_API void     ggml_evlog_rec(int cat, const char * name, uint64_t t0, uint64_t t1, int64_t a0, int64_t a1);
GGML_API void     ggml_evlog_flush(void);            // write what is new now (also done by the flusher thread and at exit)
// a copy of the ring's records from index `since` on (oldest kept when lapped), as text in the file's line format, malloc'd (free with ggml_evlog_free);
// *next = the index to ask for next time. NULL when the log is off. Reading takes no record away from the file (the server's GET /evlog?since=N).
GGML_API char *   ggml_evlog_snapshot(uint64_t since, uint64_t * next);
GGML_API void     ggml_evlog_free(char * text);

#define GGML_EV_ON(cat) (ggml_evlog_mask & (1u << (cat)))

#ifdef __cplusplus
}

// a span from construction to destruction; a0/a1 may be set before it ends
struct ggml_ev_scope {
    int cat; const char * name; uint64_t t0; int64_t a0, a1;
    ggml_ev_scope(int c, const char * n, int64_t x0 = 0, int64_t x1 = 0) : cat(c), name(n), t0(GGML_EV_ON(c) ? ggml_evlog_now() : 0), a0(x0), a1(x1) {}
    ~ggml_ev_scope() { if (t0) { ggml_evlog_rec(cat, name, t0, ggml_evlog_now(), a0, a1); } }
};
#define GGML_EV_CAT2(a, b) a##b
#define GGML_EV_CAT1(a, b) GGML_EV_CAT2(a, b)
#define GGML_EV_SCOPE(cat, name, ...) ggml_ev_scope GGML_EV_CAT1(ev_scope_, __LINE__)(cat, name, ##__VA_ARGS__)
#endif
