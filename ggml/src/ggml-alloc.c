#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;
    // fork: the planner takes each tensor's bytes from the caller's own allocator (the expert pool's heap: free gaps or windows over cold experts) and gives
    // them back after the tensor's last reader; offsets are the external buffer's own (ggml_gallocr_set_external_heap). NULL: stock's free-block planner
    size_t (*h_alloc)(void * ud, size_t size, const struct ggml_tensor * tensor);
    void   (*h_free)(void * ud, size_t offset, size_t size);
    void   (*h_reset)(void * ud);
    void * h_ud;

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(struct tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(struct tallocr_chunk * chunk, int idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    for (int i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static int ggml_dyn_tallocr_new_chunk(struct ggml_dyn_tallocr * alloc, size_t min_size) {
    if (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS) {
        return -1;
    }
    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static int galloc_hw_on(void) {   // fork: GGML_GALLOC_HW=1, allocator diagnostics (high water, alive at the end of a graph)
    static int hw = -1;
    if (hw < 0) { const char * e = getenv("GGML_GALLOC_HW"); hw = e && atoi(e) != 0; }
    return hw;
}

static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);

    if (alloc->h_alloc) {
        if (alloc->n_chunks == 0) {
            alloc->chunks[0] = calloc(1, sizeof(struct tallocr_chunk));
            GGML_ASSERT(alloc->chunks[0] != NULL);
            alloc->n_chunks = 1;
        }
        const size_t off = alloc->h_alloc(alloc->h_ud, size, tensor);
        if (off == SIZE_MAX) {
            GGML_ABORT("ggml-alloc: the external heap has no room for %s (%.1f MiB)", tensor ? tensor->name : "?", size / 1048576.0);
        }
        alloc->chunks[0]->max_size = MAX(alloc->chunks[0]->max_size, off + size);
        struct buffer_address a = { .chunk = 0, .offset = off };
        return a;
    }

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    if (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    struct buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    // fork: GGML_GALLOC_HW=1 names each tensor that raises a buffer's high-water mark by 64 MiB or more (the pass-size probe of 2026-11-14 reserved 58 GiB
    // against a 5.4 GiB peak of the graph's own tensors)
    if (tensor && addr.offset + size > chunk->max_size + ((size_t) 64 << 20)) {
        if (galloc_hw_on()) {
            GGML_LOG_WARN("galloc: high water %.0f -> %.0f MiB (chunk %d) by %s %s %.0f MiB at offset %.0f MiB\n", chunk->max_size / 1048576.0,
                (addr.offset + size) / 1048576.0, addr.chunk, tensor->name, ggml_op_desc(tensor), size / 1048576.0, addr.offset / 1048576.0);
        }
    }
    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);

    if (alloc->h_free) {
        alloc->h_free(alloc->h_ud, addr.offset, size);
        return;
    }

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    if (alloc->h_reset) {
        alloc->h_reset(alloc->h_ud);   // the last plan's bytes go back to the heap; the new plan takes its own
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc, int chunk) {
    return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
    // fork: an external buffer (ggml_gallocr_set_external_buffer): one chunk inside a buffer the allocator does not own, at this offset
    size_t offset;
    size_t capacity;   // > 0: external
};

static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) {
        return;
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->capacity == 0; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(struct vbuffer * buf, int chunk) {
    if (buf->capacity > 0) {
        return chunk == 0 ? buf->capacity : 0;
    }
    return buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(struct vbuffer * buf) {
    if (buf->capacity > 0) {
        return buf->capacity;
    }
    size_t size = 0;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    return size;
}

static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}

static void ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (char *)base + buf->offset + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(struct vbuffer * buf) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i] && buf->capacity == 0; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    // fork: per buffer id, an external buffer the graphs are placed in (NULL: the allocator owns its buffers)
    ggml_backend_buffer_t * ext_buf;    // [n_buffers]
    size_t * ext_off;                   // [n_buffers]
    size_t * ext_cap;                   // [n_buffers]
    struct ggml_backend_buffer * ext_twin;   // [n_buffers] the external buffer as compute memory (ggml_gallocr_set_external_buffer)

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;

    // fork: ggml_gallocr_get_graph_peak's answer per buffer for the plan in node_allocs and leaf_allocs (SIZE_MAX: not computed since the last reserve).
    // The expert cache asks before every graph: 0.17 ms per decode token over a 2,500-node plan that had not changed (Qwen3.8, perf record).
    size_t * peak; // [n_buffers]

    // fork: frees held back (ggml_gallocr_set_spread_fork): a tensor's bytes go back to the allocator `spread` nodes after the node that read it last
    struct galloc_held * held;
    int n_held, held_head, held_cap;
    int spread;    // the window of the graph being planned, in nodes (0: bytes go back at once, as stock)
    int node_i;    // the node being planned
};

// fork: a decode graph runs on several streams of a card at once (ggml-streams-fork.h), and two launches that touch the same memory keep their graph
// order whether or not one needs the other's result. Stock hands a freed buffer to the next node that fits, so independent branches of a layer came to
// share memory and waited for each other (plan dump 2026-11-19, Qwen3.8: the delta net's beta projection behind alpha's add, the shared expert's down
// projection behind an HC product). A backend that runs a graph that way says over how many nodes its streams stay apart (n_nodes: launches further
// apart are joined in between) and which graphs it takes (every node within max_node_bytes). In such a graph, on device buffers, a freed tensor's bytes
// go back n_nodes later, and only when the plan then still fits the memory the buffers have (never a bigger buffer for it; a size estimate plans as
// stock). GGML_GALLOC_SPREAD=0 turns it off.
struct galloc_held { struct ggml_dyn_tallocr * alloc; struct buffer_address addr; size_t size; int at; };
static int    g_galloc_spread_nodes = 0;
static size_t g_galloc_spread_bytes = 0;
static bool   g_galloc_spread_host  = false;   // host buffers too (tests)
GGML_API void ggml_gallocr_set_spread_fork(int n_nodes, size_t max_node_bytes, bool host_too);
void ggml_gallocr_set_spread_fork(int n_nodes, size_t max_node_bytes, bool host_too) {
    const char * e = getenv("GGML_GALLOC_SPREAD");
    if (e != NULL && atoi(e) == 0) {
        return;
    }
    g_galloc_spread_nodes = n_nodes;
    g_galloc_spread_bytes = max_node_bytes;
    g_galloc_spread_host  = host_too;
}
static int ggml_gallocr_spread_for(const struct ggml_cgraph * graph) {
    if (g_galloc_spread_nodes <= 0 || graph->n_nodes < 32) {
        return 0;
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        const struct ggml_tensor * t = graph->nodes[i];
        if (t->view_src == NULL && ggml_nbytes(t) > g_galloc_spread_bytes) {
            return 0;
        }
    }
    return g_galloc_spread_nodes;
}
// the bytes freed at node `upto` or before go back to their allocators
static void ggml_gallocr_held_release(ggml_gallocr_t galloc, int upto) {
    while (galloc->held_head < galloc->n_held && galloc->held[galloc->held_head].at <= upto) {
        const struct galloc_held * h = &galloc->held[galloc->held_head++];
        ggml_dyn_tallocr_free_bytes(h->alloc, h->addr, h->size);
    }
}
// does the plan in the allocators fit the memory the device buffers have now?
static bool ggml_gallocr_plan_fits(ggml_gallocr_t galloc) {
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (!g_galloc_spread_host && ggml_backend_buft_is_host(galloc->bufts[i])) {
            continue;   // planned as stock: whatever it needs, it needed before
        }
        struct ggml_dyn_tallocr * a = galloc->buf_tallocs[i];
        size_t sum = 0;
        for (int c = 0; c < a->n_chunks; c++) {
            sum += ggml_dyn_tallocr_max_size(a, c);
        }
        if (galloc->ext_buf[i] != NULL) {
            if (a->n_chunks > 1 || sum > galloc->ext_cap[i]) { return false; }
        } else if (galloc->buffers[i] == NULL) {
            if (sum > 0) { return false; }
        } else {
            for (int c = 0; c < a->n_chunks; c++) {
                if (ggml_dyn_tallocr_max_size(a, c) > ggml_vbuffer_chunk_size(galloc->buffers[i], c)) { return false; }
            }
        }
    }
    return true;
}

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->peak = malloc(n_bufs * sizeof(size_t));
    GGML_ASSERT(galloc->peak != NULL);
    for (int i = 0; i < n_bufs; i++) { galloc->peak[i] = SIZE_MAX; }

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    galloc->ext_buf = calloc(n_bufs, sizeof(ggml_backend_buffer_t));
    galloc->ext_off = calloc(n_bufs, sizeof(size_t));
    galloc->ext_cap = calloc(n_bufs, sizeof(size_t));
    galloc->ext_twin = calloc(n_bufs, sizeof(struct ggml_backend_buffer));
    GGML_ASSERT(galloc->ext_buf != NULL && galloc->ext_off != NULL && galloc->ext_cap != NULL && galloc->ext_twin != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

size_t ggml_gallocr_get_graph_peak(ggml_gallocr_t galloc, int buffer_id) {
    // the end of the highest tensor the allocator placed for the current graph in this buffer (its own allocation records, so tensors it
    // did not place, e.g. weights living in an external buffer, do not count)
    if (buffer_id >= 0 && buffer_id < galloc->n_buffers && galloc->peak[buffer_id] != SIZE_MAX) {
        return galloc->peak[buffer_id];
    }
    size_t peak = 0;
    for (int i = 0; i < galloc->n_nodes; i++) {
        const struct node_alloc * na = &galloc->node_allocs[i];
        if (na->dst.buffer_id == buffer_id && na->dst.size_max > 0) {
            peak = MAX(peak, na->dst.addr.offset + na->dst.size_max);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (na->src[j].buffer_id == buffer_id && na->src[j].size_max > 0) {
                peak = MAX(peak, na->src[j].addr.offset + na->src[j].size_max);
            }
        }
    }
    for (int i = 0; i < galloc->n_leafs; i++) {
        const struct tensor_alloc * la = &galloc->leaf_allocs[i].leaf;
        if (la->buffer_id == buffer_id && la->size_max > 0) {
            peak = MAX(peak, la->addr.offset + la->size_max);
        }
    }
    if (buffer_id >= 0 && buffer_id < galloc->n_buffers) {
        galloc->peak[buffer_id] = peak;
    }
    return peak;
}

void ggml_gallocr_set_external_buffer(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t buffer, size_t offset, size_t capacity) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);
    // the allocator's own buffer for this id is released now (its VRAM goes back to the device); graphs are placed in the external one from the next reserve
    if (galloc->buffers[buffer_id] != NULL) {
        bool shared = false;
        for (int j = 0; j < galloc->n_buffers; j++) {
            shared = shared || (j != buffer_id && galloc->buffers[j] == galloc->buffers[buffer_id]);
        }
        if (!shared) {
            ggml_vbuffer_free(galloc->buffers[buffer_id]);
        }
        galloc->buffers[buffer_id] = NULL;
    }
    // The graph's tensors are compute memory, as in the allocator's own buffers (usage COMPUTE): they are placed through a copy of the buffer object (same memory,
    // same interface) that says so. Placed through the expert pool's buffer itself (WEIGHTS), CUDA's init_tensor zeroed the row padding behind every quantized
    // graph tensor when the graph was placed, before the expert cache lent the blocks: keep-hot then carried those zeros into the experts it moved out (qni4's
    // streamed down tensors, ne0 640: 216 and 408 bytes; DeepSeek; verify -r 2, 2026-11-10). As compute memory the matmuls clear it when they run (mmq.cu, mmvq.cu).
    if (buffer != NULL) {
        galloc->ext_twin[buffer_id] = *buffer;
        galloc->ext_twin[buffer_id].usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    }
    // GGML_GALLOC_TWIN=0 places through the pool buffer itself, as before: only to test that the pool guard (--moe guard) finds that bug
    static int twin = -1;
    if (twin < 0) { const char * e = getenv("GGML_GALLOC_TWIN"); twin = !(e != NULL && atoi(e) == 0); }
    galloc->ext_buf[buffer_id] = buffer ? (twin ? &galloc->ext_twin[buffer_id] : buffer) : NULL;
    galloc->ext_off[buffer_id] = offset;
    galloc->ext_cap[buffer_id] = buffer ? capacity : 0;
    // The external buffer is one chunk: the graph is planned as one up to its capacity. The buffer type's own limit (Vulkan: its 1 GiB suballocation block)
    // cut a 1.6 GiB graph in two chunks, refused at reserve (OLMoE at 32k cells, 2026-11-17); cleared, the type's own limit applies again.
    // shortcut: a tallocr shared by two buffer ids of one type gets the larger limit for both; split them if a type ever mixes external and own buffers
    const size_t own = MIN(ggml_backend_buft_get_max_size(galloc->bufts[buffer_id]), SIZE_MAX/2);
    galloc->buf_tallocs[buffer_id]->max_chunk_size = buffer ? MAX(own, capacity) : own;
}

// fork (no header edit: declared where used): buffer_id's graphs are planned in the external buffer's own address space by the caller's allocator (see
// ggml_dyn_tallocr.h_alloc); set the external buffer first (ggml_gallocr_set_external_buffer, offset 0, the whole buffer). NULL alloc: stock's planner again
void ggml_gallocr_set_external_heap(ggml_gallocr_t galloc, int buffer_id, size_t (*alloc)(void *, size_t, const struct ggml_tensor *), void (*freef)(void *, size_t, size_t),
                                    void (*reset)(void *), void * ud) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);
    struct ggml_dyn_tallocr * t = galloc->buf_tallocs[buffer_id];
    t->h_alloc = alloc; t->h_free = freef; t->h_reset = reset; t->h_ud = ud;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    free(galloc->peak);
    free(galloc->held);

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->ext_buf);
    free(galloc->ext_off);
    free(galloc->ext_cap);
    free(galloc->ext_twin);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    struct ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    if (galloc->spread > 0 && (g_galloc_spread_host || !ggml_backend_buft_is_host(buft))) {
        if (galloc->n_held == galloc->held_cap) {
            galloc->held_cap = galloc->held_cap ? 2 * galloc->held_cap : 1024;
            galloc->held = realloc(galloc->held, sizeof(struct galloc_held) * (size_t) galloc->held_cap);
            GGML_ASSERT(galloc->held != NULL);
        }
        galloc->held[galloc->n_held++] = (struct galloc_held) { alloc, hn->addr, size, galloc->node_i };
    } else {
        ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    }
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);
    galloc->n_held = galloc->held_head = 0;   // fork: see ggml_gallocr_set_spread_fork
    galloc->node_i = -1;

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);
        galloc->node_i = i;
        if (galloc->spread > 0) { ggml_gallocr_held_release(galloc, i - galloc->spread); }   // fork: see ggml_gallocr_set_spread_fork

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_impl_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }

    ggml_gallocr_held_release(galloc, INT_MAX);   // fork: the allocators hold again what stock would hold here

    // fork: what is still allocated at the graph's end, always on (one pass over the nodes, at a re-plan only): a tensor no node reads is computed for nothing and
    // never freed, a view no node reads keeps its source to the end. The pass-size probe of 2026-11-14 reserved 58 GiB on CUDA0 for a 7-8 GiB graph that way and
    // capped every prompt pass at 3776 tokens; run.py fails a row on these lines (tests/test-graphcheck-fork.cpp). Under 16 MiB a tensor is left alone: the fork's
    // hook nodes (ids a split callback reads back, not a graph source) reach a MiB on big passes.
    {
        enum { TOP = 8 };
        size_t sum = 0, top_b[TOP] = { 0 };
        int n_unread = 0;
        const struct ggml_tensor * top_t[TOP] = { NULL };
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * t = graph->nodes[i];
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, t);
            const size_t b = ggml_nbytes(t);
            // (a branch ggml_build_forward_select leaves in the graph uncomputed, so the graph keeps one layout for every input type, is no leak: llama-graph.cpp build_inp_embd)
            if (ggml_impl_is_view(t) || !hn->allocated || (t->flags & GGML_TENSOR_FLAG_OUTPUT) || !(t->flags & GGML_TENSOR_FLAG_COMPUTE) || b < ((size_t) 16 << 20)) {
                continue;
            }
            sum += b;
            n_unread++;
            for (int k = 0; k < TOP; k++) {
                if (b > top_b[k]) {
                    for (int m = TOP - 1; m > k; m--) { top_b[m] = top_b[m - 1]; top_t[m] = top_t[m - 1]; }
                    top_b[k] = b; top_t[k] = t;
                    break;
                }
            }
        }
        static size_t last_sum = 0;   // a graph re-planned as it was says it once
        static int last_nodes = 0;
        if (n_unread > 0 && (sum != last_sum || graph->n_nodes != last_nodes)) {
            last_sum = sum;
            last_nodes = graph->n_nodes;
            GGML_LOG_WARN("galloc: UNREAD %.0f MiB in %d tensors still allocated at the end of a graph of %d nodes (computed for nothing, held to the end), largest:\n",
                sum / 1048576.0, n_unread, graph->n_nodes);
            for (int k = 0; k < TOP && top_t[k]; k++) {
                const struct ggml_tensor * pin = NULL;   // the unread view that holds it
                for (int j = 0; ggml_gallocr_hash_get(galloc, top_t[k])->n_views > 0 && j < graph->n_nodes && !pin; j++) {
                    if (graph->nodes[j]->view_src == top_t[k]) { pin = graph->nodes[j]; }
                }
                GGML_LOG_WARN("galloc: UNREAD   %s %s %.0f MiB (buffer %d): %s%s%s\n", top_t[k]->name, ggml_op_desc(top_t[k]), top_b[k] / 1048576.0,
                    ggml_gallocr_hash_get(galloc, top_t[k])->buffer_id, pin ? "kept by the unread view " : "no node reads it",
                    pin ? pin->name : "", pin ? ggml_op_desc(pin) : "");
            }
        }
    }
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, bool no_alloc) {
    for (int i = 0; i < galloc->n_buffers; i++) { galloc->peak[i] = SIZE_MAX; }   // fork: the plan is rewritten below
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    galloc->spread = no_alloc ? 0 : ggml_gallocr_spread_for(graph);   // fork: see ggml_gallocr_set_spread_fork
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    if (galloc->spread > 0 && !ggml_gallocr_plan_fits(galloc)) {   // not within the memory at hand: the plan as stock makes it
        galloc->spread = 0;
        for (int i = 0; i < galloc->n_buffers; i++) {
            ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
        }
        ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    }
    galloc->spread = 0;

    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        if (realloc) {
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                if (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            if (no_alloc) {
                galloc->buffers[i] = NULL;
            } else if (galloc->ext_buf[i] != NULL) {
                // fork: the graph goes into the external buffer; it never grows: a graph that does not fit is an error, not a reallocation
                galloc->buffers[i] = NULL;
                if (galloc->buf_tallocs[i]->n_chunks > 1 || new_size > galloc->ext_cap[i]) {
                    GGML_LOG_ERROR("%s: graph needs %.1f MiB in %d chunk(s), the external %s buffer holds %.1f MiB\n", __func__,
                        new_size / 1048576.0, galloc->buf_tallocs[i]->n_chunks, ggml_backend_buft_name(galloc->bufts[i]), galloc->ext_cap[i] / 1048576.0);
                    return false;
                }
                struct vbuffer * vb = (struct vbuffer *) calloc(1, sizeof(struct vbuffer));
                GGML_ASSERT(vb != NULL);
                vb->chunks[0] = galloc->ext_buf[i];
                vb->offset    = galloc->ext_off[i];
                vb->capacity  = galloc->ext_cap[i];
                galloc->buffers[i] = vb;
            } else {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    return false;
                }
            }
        }
    }

    return true;
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    GGML_ASSERT(ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true));
    for (int i = 0; i < galloc->n_buffers; i++) {
        sizes[i] = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    size_t node_size = 0;
    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}

static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    // fork: a buffer that was released (ggml_gallocr_set_external_buffer) and not reserved again holds nothing: a graph of the same shape still has to be
    // reserved before it is placed in it (placing it dereferenced the missing buffer)
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] == NULL && galloc->buf_tallocs[i] != NULL && ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], 0) > 0) {
            return true;
        }
    }

    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }

    // fork: a graph of the same shape but much smaller is planned again. It fits the old plan, so stock keeps it, and its tensors then lie at the offsets of the biggest graph so far:
    // on the pool (the compute alias lends the buffer's start, blocks evicted) a 4704-token prompt pass reached 2611 MiB where it needs 1400 (Qwen3.6, after 9792-token passes). The
    // graph does not use the room, the experts lose it, the pass planner reads the reach as the graph's size and cuts later passes to 64 tokens.
    // Only on an external buffer (the lent pool): a buffer the allocator owns keeps stock's plan, the worst case reserved at load. Re-planning there helps nothing and costs a lot:
    // every pass ended under 3/4 of the 262144-cell reserve, the next pass's KV outgrew the tight plan, and a multi-buffer graph re-plans through the scheduler, which synchronises
    // every backend first: each prompt pass of Qwen3.6-35B on two cards waited (prompt 6974 -> 5062 t/s, b14244 against b14717, 2026-11-06).
    size_t size_plan = 0, size_now = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }
        if (!node->data && !node->view_src && node_alloc->dst.buffer_id >= 0 && galloc->ext_buf[node_alloc->dst.buffer_id] != NULL) {
            size_plan += node_alloc->dst.size_max;
            size_now  += ggml_backend_buft_get_alloc_size(galloc->bufts[node_alloc->dst.buffer_id], node);
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
                return true;
            }
        }
    }

    if (size_plan > 64*1024*1024 && size_now < size_plan / 4 * 3) {
        return true;
    }

    return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// utils

static struct ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        struct ggml_context * ctx, int * n_tensors) {
    int n = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    if (n == 0) {
        return NULL;
    }

    struct ggml_tensor ** tensors = (struct ggml_tensor **) malloc(n * sizeof(struct ggml_tensor *));
    if (tensors == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * sizeof(struct ggml_tensor *));
        return NULL;
    }
    int i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    return tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    int n_tensors = 0;
    struct ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    if (tensors == NULL) {
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    free(tensors);
    return buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    int n_tensors = 0;
    struct ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    if (tensors == NULL) {
        return 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    free(tensors);
    return nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
