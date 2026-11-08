// The RAM pool (--moe rampool=1): expert slices in pinned host memory chosen by heat, read past the page cache. See ggml-rampool-fork.cpp.
// Plain C so the CPU backend (ggml-cpu.c) can call it; tensors are keys here, nothing in this API reads a tensor.
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// a chunk of pinned memory that stays until the pool closes; NULL: no more
typedef void * (*ggml_rampool_alloc_t)(size_t bytes, void * ud);
// when element e is expected to be used next, on the caller's clock (smaller = sooner). HUGE_VAL: a RAM copy pays nothing (a card holds it): never filled, evicted first
typedef double (*ggml_rampool_heat_t)(const void * key, int64_t e, void * ud);

// ---- set-up, in this order ----
// `key` holds n elements of esz bytes each, at file_off of the file behind fd. kind: elements of two kinds never share a block (the ggml type: what lies behind an element is read as this type).
// pad: bytes a kernel reads past an element (0: none); its block is esz + pad and carries them, the next element's bytes read along, zeros behind the last element
GGML_API void ggml_rampool_add_fork(const void * key, int fd, size_t file_off, int64_t n, size_t esz, size_t pad, int32_t kind);
// at most `bytes` of pool, every class the same share of its elements, n_threads readers. false: no pool, every call below is a no-op and every table NULL
GGML_API bool ggml_rampool_open_fork(size_t bytes, ggml_rampool_alloc_t alloc, ggml_rampool_heat_t heat, void * ud, int n_threads);
GGML_API void ggml_rampool_fill_fork(void);    // reads as many elements as the pool holds, the soonest expected first, laid out in file order; then the readers start
GGML_API void ggml_rampool_close_fork(void);   // the readers stop; the chunks belong to whoever allocated them
// true: a block may change owner during the run (eviction by heat). false (the default): the fill's set stays; a slice out of it is read the old way (mapping, page cache)
GGML_API void ggml_rampool_evict_fork(bool on);

// ---- use ----
// element -> pointer into the pool, NULL: not there. NULL table: this tensor is not in the pool. A pointer is good from a need until the graph after next (see need).
GGML_API const char * const * ggml_rampool_table_fork(const void * key);
// the next graph starts: what the one before it asked for is left alone a little longer (its copies may still run), what is older may be evicted
GGML_API void ggml_rampool_graph_fork(void);
// CHECK AND LOAD BEFORE THE KERNEL: these elements of each of the nk tensors are read now if they are out (all reads start first, then one wait), returns when they are
// in. An element that stays out (no block: it is expected later than every resident) has no table entry: the caller reads it the old way, the stats count those as refused.
GGML_API void ggml_rampool_need_fork(const void * const * keys, int nk, const int32_t * ids, int n);
// a kernel's look-up (no lock, no read): the element's block when it is in (with eviction on: only when this graph or the one before it asked for it, it cannot be evicted before the graph after next), else NULL
GGML_API const char * ggml_rampool_get_fork(const void * key, int64_t e);
// a prompt pass's copy: the block when the element is in (it is kept until the graph after next), NULL: out, read it the old way; never reads, counted as pp_hit / pp_miss
GGML_API const char * ggml_rampool_use_fork(const void * key, int64_t e);
// one element for a reader outside any graph (an upload worker): read now if it is out, never evicted until the drop. NULL: stays out, read it the old way
GGML_API const char * ggml_rampool_take_fork(const void * key, int64_t e);
GGML_API void ggml_rampool_drop_fork(const void * key, int64_t e);

struct ggml_rampool_stats {
    uint64_t asked, hit, cold, refused, evicted;   // elements asked for; found in the pool; read from the file; left out (no block); blocks taken from another element
    uint64_t cold_bytes, wait_us;                  // bytes read from the file on demand; time callers waited for reads
    uint64_t pp_hit, pp_miss;                      // prompt-pass copies found in the pool / left to the SSD path
    uint64_t bytes, used;                          // pool bytes, bytes holding an element
    int      classes;
};
GGML_API void ggml_rampool_stats_fork(struct ggml_rampool_stats * s, bool reset);
// time callers waited for demand reads since the pool opened, never reset (the decode thread tuner takes it out of the token time); 0 without a pool
GGML_API uint64_t ggml_rampool_wait_us_fork(void);

#ifdef __cplusplus
}
#endif
