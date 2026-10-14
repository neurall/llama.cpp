# Backend interface of the expert cache

The cache engine (`src/llama-moecache-fork.cpp`, `llama-moestate-fork.cpp`, `llama-dsplit-fork.cpp`) talks to a GPU backend only through the generic ggml
backend API plus a small set of **optional functions** found with `ggml_backend_reg_get_proc_address`. They are looked up by device (the registry of the first GPU
device) under the neutral name `ggml_backend_moe_<fn>`; the legacy name `ggml_backend_cuda_<fn>` is still accepted, so the CUDA backend needed no change.
The lookup is `llama_moe_backend_get_proc("<fn>")` in `src/llama-moe-backend.h`.

| fn | signature | purpose | when the backend has none |
|---|---|---|---|
| `obs_enable`, `obs_read` | `void(bool)`, `bool(int,int,uint64_t*)` | routing counters on the device (decode heat) | the CPU callback counts |
| `obs_prompt_init_fork`, `obs_prompt_read_fork` | `bool(int dev,int layers,int experts)`, `bool(int,int,uint8_t*)` | per-layer expert use of a prompt, counted on the card | the prompt's use is not counted |
| `copy_slices_fork` | `bool(backend,const void*const*src,void*const*dst,size_t n,size_t bytes)` | many device-to-device slice copies in one launch | one generic tensor copy per slice |
| `set_tensor_nosync_fork`, `thread_sync_fork` | | host-to-device copy without a sync per slice, and its sync | the generic async set and one sync |
| `scratch_fork` | `size_t(int dev,void*,size_t)` | the backend's scratch placed inside lent pool blocks | the scratch stays in the backend pool |
| `host_cache` | `int(int,void*)` | weights cache file for the pinned host buffer | the weights are read from the model file |
| `nvtx_push_fork`, `nvtx_pop_fork` | | profiler ranges | none |

The kernels are ggml graph ops: `MUL_MAT_ID` with the cache table (`src[3]`, op_params[0] = the "not cached" dummy) and ids below zero skipped (output rows zeroed).
A backend that does not implement them reports it in `supports_op` and the cache is not enabled on its devices.

A new backend (Vulkan, HIP) therefore needs: the two kernel behaviours above for the quantizations it wants, a pinned or host-visible buffer type
(`ggml_backend_dev_host_buffer_type`), and, for speed, `copy_slices_fork`. Everything else has a correct fallback.
