// fork: optional backend hooks, looked up by neutral name from a device's own backend registry.
// Core code never names a backend: a backend that implements a hook exports it as ggml_backend_fork_<name> through its
// registry's get_proc_address; one that does not returns nullptr and the core takes its generic path.
// The CUDA backend (and HIP, built from the same sources) answers the neutral names; other backends may add theirs.
#pragma once

#include "ggml-backend.h"

// the hook from the backend of one device (nullptr: the backend has none, or dev is nullptr)
static inline void * llama_fork_proc_dev(ggml_backend_dev_t dev, const char * name) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// the hook from any loaded backend that has it: for process-wide hooks (tracing, graph hooks, the OOM report), not for a copy
// into a given buffer; with several GPU backends loaded the first one wins, so a per-device hook must use llama_fork_proc_dev
static inline void * llama_fork_proc(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        if (void * p = ggml_backend_reg_get_proc_address(ggml_backend_reg_get(i), name)) { return p; }
    }
    return nullptr;
}

// the index of a device inside its own backend (CUDA2 -> 2, Vulkan0 -> 0): what a backend hook that takes a device number expects; -1 when unknown
static inline int llama_fork_dev_index(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    for (size_t i = 0; reg && i < ggml_backend_reg_dev_count(reg); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) { return (int) i; }
    }
    return -1;
}
