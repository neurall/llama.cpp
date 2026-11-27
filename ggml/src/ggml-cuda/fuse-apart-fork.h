#pragma once
// test: tests/test-fuse-apart-fork.cpp (no GPU)
#include "ggml.h"
#include "../ggml-impl.h"   // ggml_are_same_layout, GGML_LOG_WARN

#include <initializer_list>

// fork: a fusion that skips nodes runs at its first node and reads the inputs of the nodes it skips, but the allocator laid the buffers out for the
// unfused order: an input whose last reader is a skipped node is free before the group's last node is allocated, so the group's output may sit on
// it, and a kernel that reads that input while it writes the output reads its own writes. Inkling 2's depth mix with experts on the CPU: the layer
// state is the scheduler's copy of the CPU split's output, freed after the concat; decode PPL 1.60-1.63, different every run, against 1.582 with the
// mix fusion off (2026-11-27). The upstream check (ggml_cuda_check_fusion_memory_ranges) skips inputs whose op is NONE, and the scheduler's copies
// are op NONE. So: the fusion runs only when the bytes it writes share none with the bytes it reads. same_ok allows an input at exactly the output's
// address and layout, for kernels that read each output element's own inputs before they write it (the loop controller's in-place add).
static inline bool ggml_cuda_fuse_apart_fork(const ggml_tensor * out, std::initializer_list<const ggml_tensor *> ins, bool same_ok) {
    const char * o0 = (const char *) out->data;
    const char * o1 = o0 + ggml_nbytes(out);
    for (const ggml_tensor * in : ins) {
        if (in == nullptr || in->data == nullptr) {
            continue;
        }
        const char * i0 = (const char *) in->data;
        const char * i1 = i0 + ggml_nbytes(in);
        if (o0 >= i1 || i0 >= o1) {
            continue;   // apart
        }
        if (same_ok && i0 == o0 && ggml_are_same_layout(in, out)) {
            continue;
        }
        return false;
    }
    return true;
}

// the refusal's line in the log, a few times a fusion (the fix's own line to look for in a run's log)
static inline void ggml_cuda_fuse_apart_warn_fork(const char * fusion, const ggml_tensor * out, int & said) {
    if (said < 4) {
        ++said;
        GGML_LOG_WARN("%s fusion not used: its output %s shares bytes with an input it reads (the nodes run one by one)\n", fusion, out->name);
    }
}
