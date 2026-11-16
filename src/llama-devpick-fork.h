#pragma once

#include "ggml-backend.h"

// One physical GPU seen by two backends (an integrated Radeon via HIP and Vulkan): the backend that is faster on this GPU, measured once per build and
// kept in the engine's state file: [gpu <pci id>] 1tok_p_time_<backend>_b<build> (ms per prompt token of a 512-token matrix pass), 1tok_g_time_<backend>_b<build>
// (ms of a one-token matrix-vector pass), pick_b<build>. LLAMA_DEVICE_PICK=0: no measurement, a is kept; =remeasure: ignore the stored values.
ggml_backend_dev_t llama_device_pick(ggml_backend_dev_t a, ggml_backend_dev_t b);
