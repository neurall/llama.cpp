#pragma once

#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <string>

// Optional functions of a GPU backend that the MoE expert cache uses (see fork/docs/backend-interface.md).
// A backend exports them with ggml_backend_reg_get_proc_address as "ggml_backend_moe_<fn>" ("ggml_backend_cuda_<fn>" is also accepted).
// A missing function is not an error: the caller has a generic fallback.

// registry of the first GPU device, nullptr when there is none
static inline ggml_backend_reg_t llama_moe_backend_get_reg() {
    static ggml_backend_reg_t reg = [] {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                return ggml_backend_dev_backend_reg(dev);
            }
        }
        return (ggml_backend_reg_t) nullptr;
    }();
    return reg;
}

// function `fn` of the backend `reg`, nullptr when it does not have it
static inline void * llama_moe_backend_get_proc(ggml_backend_reg_t reg, const char * fn) {
    if (reg == nullptr) {
        return nullptr;
    }
    void * proc = ggml_backend_reg_get_proc_address(reg, (std::string("ggml_backend_moe_") + fn).c_str());
    if (proc == nullptr) {
        proc = ggml_backend_reg_get_proc_address(reg, (std::string("ggml_backend_cuda_") + fn).c_str());
    }
    return proc;
}

// function `fn` of the backend of the first GPU device
static inline void * llama_moe_backend_get_proc(const char * fn) {
    return llama_moe_backend_get_proc(llama_moe_backend_get_reg(), fn);
}

// trailing number of the device name ("CUDA1" -> 1), -1 when there is none
static inline int llama_moe_backend_get_dev_index(ggml_backend_dev_t dev) {
    const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
    if (name == nullptr) {
        return -1;
    }
    const char * end = name + strlen(name);
    const char * p   = end;
    while (p > name && p[-1] >= '0' && p[-1] <= '9') {
        --p;
    }
    return p == end ? -1 : atoi(p);
}
