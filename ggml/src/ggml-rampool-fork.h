// The RAM pool: expert weights in pinned host memory without mmap and without the page cache. See ggml-rampool-fork.cpp.
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
// true: this element needs no RAM copy now (its expert sits on a card), it goes first when a block is needed
typedef bool   (*ggml_rampool_low_t)(const void * key, int64_t e, void * ud);

// ---- set-up, in this order ----
// `key` holds n elements of esz bytes each, at file_off of the file behind fd. kind: elements of two kinds never share a block (the ggml type: what lies behind an element is read as this type, see the class comment)
GGML_API void ggml_rampool_add_fork(const void * key, int fd, size_t file_off, int64_t n, size_t esz, int32_t kind);
// at most `bytes` of pool in chunks of about chunk_bytes, n_threads readers. false: no pool, every call below is a no-op and every table NULL
GGML_API bool ggml_rampool_open_fork(size_t bytes, size_t chunk_bytes, ggml_rampool_alloc_t alloc, ggml_rampool_low_t low, void * ud, int n_threads);
GGML_API void ggml_rampool_seed_fork(const void * key, int64_t e, uint32_t uses);   // how often this element was used in earlier runs: the fill takes the most used
GGML_API void ggml_rampool_fill_fork(void);    // reads as many elements as the pool holds, most used first, laid out in file order
GGML_API void ggml_rampool_close_fork(void);   // the readers stop; the chunks belong to whoever allocated them

// ---- use ----
// element -> pointer into the pool, NULL: not there. NULL table: this tensor is not in the pool. A pointer is good from a need until the graph after next (see need).
GGML_API const char * const * ggml_rampool_table_fork(const void * key);
// the next graph starts: what the one before it asked for is left alone a little longer (its copies may still run), what is older may be evicted
GGML_API void ggml_rampool_graph_fork(void);
// CHECK AND LOAD BEFORE THE KERNEL: these elements are read now if they are out, returns when they are in. ids NULL: all of them. An element that stays out (no block to take) has
// no table entry: the caller reads it the old way, ggml_rampool_stats_fork counts those.
GGML_API void ggml_rampool_need_fork(const void * key, const int32_t * ids, int n, bool count);
// the same for an op that has per-element row counts: the elements with a count > 0
GGML_API void ggml_rampool_need_counts_fork(const void * key, const int64_t * counts, int n);
// the same without waiting: the readers fetch ahead of the compute that will ask. ids NULL: all of them. Blocks are taken only from elements the compute will not read (low) when only_low.
GGML_API void ggml_rampool_ahead_fork(const void * key, const int32_t * ids, int n, bool only_low);
// how many elements from e (at most up to end) lie directly behind each other in one chunk: one copy reads them all
GGML_API int64_t ggml_rampool_run_fork(const void * key, int64_t e, int64_t end);
// one element for a reader outside any graph (an upload worker): read now if it is out, never evicted until the drop. NULL: stays out, read it the old way
GGML_API const char * ggml_rampool_take_fork(const void * key, int64_t e);
GGML_API void ggml_rampool_drop_fork(const void * key, int64_t e);

struct ggml_rampool_stats {
    uint64_t asked, cold, ahead, evicted, lost;   // elements asked for; read on demand; read ahead; blocks taken from another element; stayed out (read the old way)
    uint64_t cold_bytes, wait_us;                 // bytes read from the file; time callers waited for reads
    uint64_t bytes, used;                         // pool bytes, bytes holding an element
};
GGML_API void ggml_rampool_stats_fork(struct ggml_rampool_stats * s, bool reset);

#ifdef __cplusplus
}
#endif
