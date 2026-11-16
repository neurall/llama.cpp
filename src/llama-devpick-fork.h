#pragma once

#include "ggml-backend.h"

#include <string>
#include <vector>

// The discrete GPUs a model runs on when no -dev is given: llama.cpp's device list (llama_model_load_from_file_impl) takes a card once, on the first backend
// that lists it; a later device with the same PCI id is skipped, an unknown id never matches. ids: the discrete GPUs' PCI ids in registry order ("" unknown);
// returns which entries are such later twins. A build with CUDA and Vulkan lists every NVIDIA card twice: a feature decided from "every GPU" counted the
// Vulkan twins and turned the KV tail and the compute alias off for a CUDA run (common_moe_cache_auto_impl, 2026-11-16)
static inline std::vector<bool> llama_dev_twins(const std::vector<std::string> & ids) {
    std::vector<bool> twin(ids.size(), false);
    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = 0; j < i && !ids[i].empty(); ++j) { if (ids[j] == ids[i]) { twin[i] = true; break; } }
    }
    return twin;
}

// One physical GPU seen by two backends (an integrated Radeon via HIP and Vulkan): the backend that is faster on this GPU, measured once per build and
// kept in the engine's state file: [gpu <pci id>] 1tok_p_time_<backend>_b<build> (ms per prompt token of a 512-token matrix pass), 1tok_g_time_<backend>_b<build>
// (ms of a one-token matrix-vector pass), pick_b<build>. LLAMA_DEVICE_PICK=0: no measurement, a is kept; =remeasure: ignore the stored values.
ggml_backend_dev_t llama_device_pick(ggml_backend_dev_t a, ggml_backend_dev_t b);
