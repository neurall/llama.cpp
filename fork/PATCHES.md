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

## Upstream regressions and bugs found, and how they were fixed

Every change of this fork is measured against the build before it, and so is every upstream merge. These came out of that, in the order they were found. Numbers are decode tokens/s on the RTX 3090 machine unless stated.

| # | upstream change | what it broke | how it was found | fix in this fork | status |
|---|---|---|---|---|---|
| 1 | #29943 (selective expert copy moved into a user callback), on top of the #26040 host barrier | every CPU split after a GPU split waited for the GPU: +6.6 ms per GLM decode graph (-9.5% decode), -15 to -17% on a restricted-cache OLMoE cell on both test machines (PC1 124.3 -> 145.9 t/s once fixed, W10 laptop 105 -> 123) | A/B of the pre-merge and post-merge builds on the small OLMoE cell, then `GGML_SCHED_PROF=1` (barrier 0.01 ms against 6.66 ms per graph) | the barrier is skipped when the previous split ran on a GPU and either side is a GPU or the previous split read no host memory (`is_gpu`, `prev_reads_host` in `ggml_backend_sched_compute_splits`) | fixed, verified; **upstream candidate** |
| 2 | #29184 (shared experts fused into MMVQ) | the fused `mul_mat_vec_q` kernels carry a runtime shared-expert branch and reassignable pointers that every launch pays, also for models without a shared expert: Q4_K kernel +7%, Q6_K +5%, whole decode -2.2% on OLMoE (nsys: same 163,456 launches, GPU busy 759.5 -> 778.9 ms; stock b11552 has it too) | decode of a VRAM-resident OLMoE slower than the b11509 release in the same session; a compiler test (b11509 source with today's recipe = b11509 speed) ruled out the toolchain; nsys kernel table pointed at the kernel | `has_shared` template parameter: only launches with `fusion.shared_up` use the instantiation with the branch | fix committed (`20c5d02b1`), measurement across models pending; **upstream candidate** |
| 3 | MMQ ids path (`ggml-cuda/mmq.cu`) | the padding of the tail was sized from `ne12` and not from the flattened row count (`ne12*n_expert_used`) | wrong results on a batch tail with a MoE model | padding from the flattened row count | fixed; **upstream candidate** |
| 4 | Q2_0 on x86 (a missing optimisation, not a regression) | upstream has only the generic scalar `Q2_0 x Q8_0` dot product on x86, so Q2 models run far below the memory-bandwidth limit on the CPU | Qwen3.6-35B-A3B Q2_0 CPU only (Ryzen 5 3600): 6.3 t/s | AVX2 kernel and an AVX-512 VNNI path (`vpdpbusd`); two code-generation traps found and removed (clang 21 turned constant shuffle indices into `vpermb` and kept a rolled loop: 25% slower on Zen 2 until the halves were written out), the same for the tinyBLAS Q8_0 `gemm4xN` | 11.3 t/s (1.8x); **upstream candidate** |
| 5 | GLM-5 graph (k-pool models) | the graph shape changed with the token position, so the graph was rebuilt and re-allocated on every decode token and graph reuse never happened | GLM-5.3-Flash decode 21.45 t/s | fixed k-pool graph shape (fork, 28 Sep: 23.2 vs 21.45 t/s, 4 rounds on 2 servers, +8%); upstream fixed the same problem separately on 5 Oct (#29958, "unexpected graph reallocation in the k-pool models") and it is merged here | fixed upstream too |

Two of the "regressions" turned out not to be one, and are listed so the numbers are not misread: an observer that counts expert use on stock placement costs nothing measurable (`observe=0` gives +0.3%), and the clang build recipe is as fast as the gcc one on the same source.

## Upstream plan

1. Scheduler host-barrier refinement, with the restricted-OLMoE repro (smallest, clearest win).
2. MMQ tail-padding fix.
3. Q2_0 AVX2 kernels.
4. Pinned loading (the portable parts), after upstream's loader settles.
5. Cache: only as small improvements to upstream's `llama_moe_cache` where this one measured better.
