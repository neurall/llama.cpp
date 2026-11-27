#pragma once

// fork: the op_params slots the fork gives a meaning of its own, one constant per slot, and the only setters and readers of them.
//
// Upstream gave slots the fork used a meaning of its own three times: MUL_MAT slot 1 (the Hadamard hint, which a MUL_MAT slice of our
// MUL_MAT_ID inherits: segfault, 2026-11-24), MUL_MAT_ID slot 3 (the src1 precision, where our expert range was) and FLASH_ATTN_EXT slot 5
// (the indexed bias flag, where our sparse gate was: no CUDA kernel, abort at load, 2026-11-26). So:
// - fork/op-slot-check (run it against the upstream commit before merging it) fails when upstream code reads or writes a slot listed here for
//   the same op, or when fork code writes a slot of an op that is not listed here;
// - tests/test-op-slots-fork.cpp sets every slot listed here through the setter below and checks that upstream's accessors of the op still
//   read their defaults and that the fork's reader gets the value back.
// A MUL_MAT_ID's op_params reach the MUL_MAT slices a backend cuts from it (ggml_cuda_mul_mat_id copies the tensor): its slots count for
// MUL_MAT too. A new fork slot: a constant, an entry in ggml_fork_op_slots, a setter and a reader here, a case in the test.

#include "ggml.h"

#define GGML_FORK_SLOT_FA_SPARSE_MIN     6  // FLASH_ATTN_EXT: the K/V length from which a sparse attention gathers (0: the backend's own rule)
#define GGML_FORK_SLOT_MMID_NEG_IDS      1  // MUL_MAT_ID: ids may be -1 (an expert this card does not compute): the output is zeroed first
#define GGML_FORK_SLOT_MMID_RANGE_LO     8  // MUL_MAT_ID: the card's expert range [lo, hi) of the multi-GPU prompt split (hi <= lo: all)
#define GGML_FORK_SLOT_MMID_RANGE_HI     9
#define GGML_FORK_SLOT_MMID_CACHE_ABSENT 10 // MUL_MAT_ID with src[3] (the MoE cache's host table): the table's value of an expert not on the card
#define GGML_FORK_SLOT_MASK_N_KV         15 // the KQ mask input (GGML_OP_NONE): the cells its graph's K/V views cover (can_reuse_kq_mask); not 0, which
                                            // upstream's own NONE nodes use (hexagon fences)
#define GGML_FORK_SLOT_BELL_PTR          0  // MOE_BELL, MOE_ROWS (fork ops): the struct ggml_bell pointer, two slots

// Reviewed overlaps, printed by fork/op-slot-check instead of failed (fork op, slot, upstream op whose accesses are reviewed, why):
// op-slot-reviewed: MUL_MAT_ID 1 MUL_MAT the hint of a MUL_MAT reaches our flag only through a slice of the MUL_MAT_ID, which ggml_fork_mm_slice_params clears, and the fused gate/up check, which reads it on a MUL_MAT only (574af86934)

struct ggml_fork_op_slot {
    enum ggml_op op;
    int          slot;
    int          width; // slots taken
    const char * name;
};

// every slot above, for fork/op-slot-check and the test (a function, not a table in the header: no unused-variable warnings)
static inline const struct ggml_fork_op_slot * ggml_fork_op_slots(int * n) {
    static const struct ggml_fork_op_slot slots[] = {
        { GGML_OP_FLASH_ATTN_EXT, GGML_FORK_SLOT_FA_SPARSE_MIN,     1, "fa_sparse_min"     },
        { GGML_OP_MUL_MAT_ID,     GGML_FORK_SLOT_MMID_NEG_IDS,      1, "mmid_neg_ids"      },
        { GGML_OP_MUL_MAT_ID,     GGML_FORK_SLOT_MMID_RANGE_LO,     1, "mmid_range_lo"     },
        { GGML_OP_MUL_MAT_ID,     GGML_FORK_SLOT_MMID_RANGE_HI,     1, "mmid_range_hi"     },
        { GGML_OP_MUL_MAT_ID,     GGML_FORK_SLOT_MMID_CACHE_ABSENT, 1, "mmid_cache_absent" },
        { GGML_OP_NONE,           GGML_FORK_SLOT_MASK_N_KV,         1, "mask_n_kv"         },
        { GGML_OP_MOE_BELL,       GGML_FORK_SLOT_BELL_PTR,          2, "bell_ptr"          },
        { GGML_OP_MOE_ROWS,       GGML_FORK_SLOT_BELL_PTR,          2, "bell_ptr"          },
    };
    *n = (int) (sizeof(slots)/sizeof(slots[0]));
    return slots;
}

// FLASH_ATTN_EXT (the setter is ggml_flash_attn_ext_set_sparse_min)
static inline int32_t ggml_fork_fa_sparse_min(const struct ggml_tensor * t) { return t->op_params[GGML_FORK_SLOT_FA_SPARSE_MIN]; }

// MUL_MAT_ID
static inline void    ggml_fork_mmid_set_neg_ids(struct ggml_tensor * t)       { t->op_params[GGML_FORK_SLOT_MMID_NEG_IDS] = 1; }
static inline int32_t ggml_fork_mmid_neg_ids(const struct ggml_tensor * t)     { return t->op_params[GGML_FORK_SLOT_MMID_NEG_IDS]; }
static inline void    ggml_fork_mmid_set_range(struct ggml_tensor * t, int32_t lo, int32_t hi) {
    t->op_params[GGML_FORK_SLOT_MMID_RANGE_LO] = lo;
    t->op_params[GGML_FORK_SLOT_MMID_RANGE_HI] = hi;
}
static inline int32_t ggml_fork_mmid_range_lo(const struct ggml_tensor * t)    { return t->op_params[GGML_FORK_SLOT_MMID_RANGE_LO]; }
static inline int32_t ggml_fork_mmid_range_hi(const struct ggml_tensor * t)    { return t->op_params[GGML_FORK_SLOT_MMID_RANGE_HI]; }
static inline void    ggml_fork_mmid_set_cache_absent(struct ggml_tensor * t, int32_t v) { t->op_params[GGML_FORK_SLOT_MMID_CACHE_ABSENT] = v; }
static inline int32_t ggml_fork_mmid_cache_absent(const struct ggml_tensor * t) { return t->op_params[GGML_FORK_SLOT_MMID_CACHE_ABSENT]; }

// a MUL_MAT slice of a MUL_MAT_ID (its op_params copied, op set to MUL_MAT): slot 1 is upstream's hint on a MUL_MAT, our flag must not reach it
static inline void ggml_fork_mm_slice_params(struct ggml_tensor * slice) {
    slice->op_params[1] = GGML_HINT_NONE; // op-slot: upstream (clears upstream's MUL_MAT hint slot)
}

// the KQ mask input
static inline void    ggml_fork_mask_set_n_kv(struct ggml_tensor * t, int32_t n_kv) { t->op_params[GGML_FORK_SLOT_MASK_N_KV] = n_kv; } // op-slot: NONE
static inline int32_t ggml_fork_mask_n_kv(const struct ggml_tensor * t)              { return t->op_params[GGML_FORK_SLOT_MASK_N_KV]; } // op-slot: NONE
