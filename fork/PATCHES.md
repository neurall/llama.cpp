# What this fork changes, area by area

The fork is upstream llama.cpp plus the patches below. Each area is one commit of the branch `series` (`python3 fork/series.py` rebuilds it from the tip), so every change can be read, reviewed or sent upstream on its own. The areas share a few files, so only the last commit of the series is guaranteed to build.

Status words: **candidate** = general enough for an upstream pull request, **local** = specific to this fork's design or to Linux/NVIDIA hardware, **overlaps** = upstream has its own implementation.

| # | area | main files | status |
|---|---|---|---|
| 1 | Pinned weight loading | `ggml/src/ggml-cuda/ggml-cuda.cu`, `src/llama-model-loader.cpp` | local (Linux huge pages), parts candidate |
| 2 | Scheduler | `ggml/src/ggml-backend.cpp`, `ggml/include/ggml-backend.h`, `ggml/src/ggml.c` | **candidate**: host-barrier refinement |
| 3 | Kernels | `ggml/src/ggml-cpu/`, `ggml/src/ggml-cuda/` (mmq, mmvq, cpy, concat) | candidate: Q2_0 AVX2 dot, MMQ tail padding |
| 4 | Expert cache and tuner | `src/llama-moecache-fork.*`, `src/llama-moestate-fork.*`, hooks in `src/llama-graph.*`, `src/llama-context.*` | overlaps upstream's `llama_moe_cache`, coexists with it |
| 5 | Placement and flags | `common/`, `tools/cli/`, `tools/server/` | local |
| 6 | MTP and models | `src/models/glm5-next.cpp`, `src/llama-memory-hybrid-idx.*`, `conversion/` | candidate (draft head support) |
| 7 | Tooling | `fork/` | local |

## 1. Pinned weight loading

`--load-mode pin|mmap|dio`. When the model fits in RAM the weights are loaded into pinned (page-locked) memory, which makes expert uploads and prompt processing faster. Pinned pages come from, in order: an administrator-reserved 1 GiB huge-page pool, a file cache in hugetlbfs that outlives the process (`GGML_CUDA_HUGEFS`), transparent huge pages, or registered ordinary pages. The loader reads with several threads. A pin that stalls or fails exits with code 75 and says that a reboot defragments memory, instead of silently falling back to mmap.
Evidence: reload of a 109 GiB model 43 s instead of 88-95 s with the pool; GLM prompt processing +45% against mmap (`fork/docs/fork-knobs.md`, loading section). Setup: `fork/resident_pin.sh`.

## 2. Scheduler

- **Host barrier refinement.** The scheduler waits for the previous split before a CPU split that reads host memory, but not when both splits run on GPUs or the previous split reads no host data. Upstream's first version (#26040) waited for every CPU split after a GPU split; the merge of #29943 (selective expert copying moved into a user callback) lost that refinement. Cost of the loss: 6.6 ms per GLM decode graph (-9.5% decode), -15 to -17% on a restricted-cache OLMoE cell on two machines. Repro: `python3 fork/run.py regress` (restricted OLMoE, 500 tokens). **Upstream candidate.**
- `sched_copy_experts` hook: the selective expert copy calls the cache's observer, and a split timer (`GGML_SCHED_PROF=1`) prints per graph where the time goes (barrier, input copy, CPU compute, GPU launch).
- Asynchronous device-to-host copies for CPU splits (`GGML_SCHED_D2H_ASYNC=0` turns it off).

## 3. Kernels

- Q2_0 AVX2 dot product and repacking (upstream has no AVX2 kernel for Q2_0): Qwen3.6-35B Q2_0 on a CPU-only Ryzen 5 3600, 1.8x. Written so that clang and gcc generate the same fast code (a rolled loop and constant shuffle indices cost clang 25% on Zen 2).
- tinyBLAS Q8_0 `gemm4xN`: an opaque delta vector so clang does not fold the permutes.
- CUDA MMQ ids path: the tail padding is sized from the flattened row count (`ne12*n_expert_used`) instead of `ne12`. Bug fix.
- Fused gate kernel for the shared-expert gate of GLM (mul, add, sigmoid, scale), +1.0% decode on GLM.

## 4. Expert cache and self-tuner

For MoE models bigger than VRAM: experts stay in RAM, the free VRAM becomes a cache of the experts in use, the GPUs compute the cached ones while the CPU computes the rest. A tuner adjusts the knobs on measured token times, per phase (prompt processing and generation behave differently). Upstream added its own `llama_moe_cache` (#29887) after this work started. The two coexist: upstream's switch (`--moe-cache-mib`) selects theirs, otherwise this one runs; our symbols carry the `_fork` suffix (`llama_moe_cache_step_fork`, files `llama-moecache-fork.*`), so an upstream merge never collides with them. Every knob: `fork/docs/fork-knobs.md`.

## 5. Placement and flags

`--moe KEY=VAL,...`, `--fork on|off`, `-at on|off`, `--load-mode`. With no switch the fork decides per model: the cache where it measures faster than stock placement, stock placement otherwise (models that fit VRAM are placed exactly like upstream). A decode/prompt thread tuner and a draft-depth tuner for the server's speculative decoding. The decision and the tuner state are kept in `~/.cache/llama.cpp/moe-state.ini`.

## 6. MTP and models

MTP draft head for GLM-5.3-Flash and Qwen3.8-Flash-Next (`-md FILE --spec-type draft-mtp`), hybrid-index memory for the GLM-5 indexer cache, converter fixes.

## 7. Tooling (`fork/`)

`run.py` (benchmarks, builds, regression check), the run log (`run-history.csv`, `runlog.py`), simulators, experiment records. See `fork/README.md`.

## Upstream plan

1. Scheduler host-barrier refinement, with the restricted-OLMoE repro (smallest, clearest win).
2. MMQ tail-padding fix.
3. Q2_0 AVX2 kernels.
4. Pinned loading (the portable parts), after upstream's loader settles.
5. Cache: only as small improvements to upstream's `llama_moe_cache` where this one measured better.
