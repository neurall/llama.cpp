#pragma once
// fork: backend-neutral access to the few optional functions the expert cache and the loader want from a GPU backend.
//
// Until now the engine asked the registry named "CUDA" for functions named "ggml_backend_cuda_*" in about a dozen places, which tied the cache to one backend by
// name. A backend (CUDA today, Vulkan or HIP later) offers any subset of these through ggml_backend_reg_get_proc_address, under the neutral name
// "ggml_backend_moe_<fn>" (the legacy "ggml_backend_cuda_<fn>" is still found, so the CUDA backend needs no change), and the engine falls back when one is missing:
//
//   fn                              what                                                       fallback when missing
//   obs_enable / obs_read           routing observer counters on the device (decode heat)       the host callback of the CPU path counts instead
//   obs_prompt_init / _read_fork    per-layer expert-use histograms of a prompt, on the card    the prompt's use is not counted (the cache starts colder)
//   copy_slices_fork                many expert slices copied device-to-device in one launch    one generic tensor copy per slice (correct, slower)
//   set_tensor_nosync_fork          host-to-device copy without a stream sync per slice         the generic async set plus one sync at the end
//   thread_sync_fork                the sync that pairs with the nosync copy                    ggml_backend_synchronize on the backend
//   scratch_fork                    the backend's scratch region placed in lent pool blocks     the scratch stays in the backend's own pool (the lend is smaller)
//   host_cache                      weights cache file for the pinned host buffer (huge pages)  weights are read from the model file (slower load)
//   nvtx_push_fork / nvtx_pop_fork  profiler ranges                                             none
//
// The kernels (MUL_MAT_ID with the cache table and skipped ids) are ggml graph ops: a backend that cannot run them says so through supports_op and the cache is
// not enabled on its devices (capability flags in common.cpp).
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <string>

namespace moe_backend {

// the registry of the first GPU device: the backend that owns the cache's device memory
inline ggml_backend_reg_t gpu_reg() {
    static ggml_backend_reg_t reg = [] {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { return ggml_backend_dev_backend_reg(d); }
        }
        return (ggml_backend_reg_t) nullptr;
    }();
    return reg;
}

// the function `fn` ("obs_enable", "copy_slices_fork", ...) of a backend: the neutral name first, then the legacy CUDA name; nullptr when the backend has none
inline void * proc(ggml_backend_reg_t reg, const char * fn) {
    if (!reg) { return nullptr; }
    void * p = ggml_backend_reg_get_proc_address(reg, (std::string("ggml_backend_moe_") + fn).c_str());
    if (!p) { p = ggml_backend_reg_get_proc_address(reg, (std::string("ggml_backend_cuda_") + fn).c_str()); }
    return p;
}

inline void * proc(const char * fn) { return proc(gpu_reg(), fn); }

// the index at the end of a device name ("CUDA1" -> 1, "Vulkan0" -> 0); -1 for none. The backend's scratch and observer calls take it.
inline int dev_index(ggml_backend_dev_t dev) {
    const char * nm = dev ? ggml_backend_dev_name(dev) : nullptr;
    if (!nm) { return -1; }
    const char * p = nm + strlen(nm);
    while (p > nm && p[-1] >= '0' && p[-1] <= '9') { --p; }
    return *p ? atoi(p) : -1;
}

} // namespace moe_backend
