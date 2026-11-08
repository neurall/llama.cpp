# Load-time 29.7 GiB reserve: compute the first micro-batch instead of failing into it

2026-11-07. Read-only analysis of the `allocating 29753.66 MiB on device 1 ... graph_reserve: failed to allocate compute buffers`
lines in the GLM-5.3-Flash GSQ 3.0/3.5bit pf12k cells (runs.db rows 23715, 23716, 23740, 23770, 23786, 23788, 23802).

## What it is

Not a crash. A benign load-time step-down that every build does, rel_b14244 (29f224aa4) included: the first context is asked for
ubatch 16384, its prompt reserve wants 1.816 MiB/token on each card (29.7 GiB), fails, and `common_init_result` bisects
16384 x, 8192 ok, 12288 x, 10240 ok, 11264 x, 10752 x, 10496 x, 10368 ok, 10432 x, 10400 ok: 10 context creations. Each one
that fits allocates and pins the CUDA_Host compute buffer (4-5 GiB), freed again by the next try. The real crash of those rows
is the illegal memory access (tile_spec), fixed in 06963e086d.

| build, GLM log | first fail | load done (threadpool init) | lost | host pins per fitting try |
|---|---|---|---|---|
| rel_b14244 glm35 | 0.43 | 1.00 | 17 s | 1.3, 2.1, 3.4, 4.7 s (weights mmap, 100 GiB page cache) |
| rel_b14244 glm30 | 0.45 | 0.57 | 12 s | similar |
| rampool11 glm35 .prev | 0.44 | 1.00 | 16 s | 1.1, 1.6, 1.7, 1.6 s at free RAM 1.3-9 GiB, 1-3 GiB of each pin landing in CMA blocks and re-faulted |
| rampool11 glm35 | 0.45 | 1.04 | 18 s | same |

rel_b14244 does not start faster: same ubatch, same step-down, same count of contexts (its final size is 10336 vs 10400 because its
reserve graph is 320 MiB bigger). What differs in the candidates: the weights are O_DIRECT-pinned (100.5 GiB, `pin-mlock`), so the
4-5 GiB host compute pins of the retries happen at 1-9 GiB free RAM, inside CMA blocks, under the zram pressure that caused the
GLM/MiMo pin stall (fork/docs on lock_before_pin). The release pins the same buffers against 100 GiB of reclaimable page cache.
Both waste ~15 s per GLM load and eight pins; the candidates do it in the one window where a pin can stall.

## Where (file:line, branch prerelease at 2dcbce609c)

- `common/common.cpp:2251-2255` picks ubatch = min(ctx, 16384) with no look at VRAM ("lowered to what the pool can lend"), but
  the pool does not exist yet at the first reserve.
- `src/llama-context.cpp:1712-1713` already runs a measure-only reserve of exactly that graph before the real one and stores
  the per-backend bytes in `lt[]` (used only for the "largest tensors" log line). For GLM at 16384: CUDA1 29753 MiB, CUDA0 the same
  slope; linear through the origin (29753/16384 = 22318/12288 = 18943/10432 = 1.816 MiB/token). The size is the
  `indexer_score-N LIGHTNING_INDEXER [262208, ubatch] f32` tensors, 16 MiB/token each: the reserve graph is built before the KV
  pool is bound, so `llama_kv_cache::get_size_reserve()` (`src/llama-kv-cache.cpp:3714`) returns the full 1M-cell context, not the
  10240-cell tail step. After the pool binds, the same graph measures 11865 MiB at 10400 tokens (1.14 MiB/token).
- `src/llama-context.cpp:1789` gates the computed fit (`fit_ub`) on `!moe_alias_bufs.empty()`: skipped at load. Line 1802 then does
  the real allocating reserve of 16384 and throws at 1826.
- `common/common.cpp:2650-2670` catches the throw and bisects, a new `llama_init_from_model` per step.
- `ggml_backend_dev_memory` is already used in this file for free VRAM (`src/llama-context.cpp:370`); the pool margin is
  `vram_margin()` in `src/llama-moecache-fork.cpp:7721` (484 MiB floor, user `MARGIN_MB`, stored need).

Note `cparams.n_ubatch` after load is the ceiling for every prompt pass of the run: `fit_ub` at 1791 only lowers it
("micro-batch from the pools: 10400 tokens (the configured size fits)"), and the pool could lend more (room 14520 MiB vs
11865 used on CUDA1). The load-time bisection, not the pool, decides the pass size today.

## Spec (one place, no new constants)

In `llama_context::sched_reserve`, `src/llama-context.cpp`, directly after the `list_top(...)` call at 1712-1713 and before the
`ub_fitted` block at 1788, when `cparams.moe_cache && moe_alias_bufs.empty() && !model.no_alloc() && n_tokens > 512`:

1. For each GPU backend `i` with `lt[i] > 0`: `free_i` from `ggml_backend_dev_memory`, `margin_i` from a one-line exported wrapper
   `size_t llama_moe_cache_vram_margin_fork(ggml_backend_dev_t dev)` around `vram_margin(dev, "", false, nullptr, nullptr)`
   (returns the existing 484 MiB floor or the user's `MARGIN_MB`; add the prototype next to the other `_fork` externs at the top
   of llama-context.cpp, ~line 69).
2. `cap_i = (free_i > margin_i) ? (free_i - margin_i) * n_tokens / lt[i] / 32 * 32 : 0` (bytes are linear in tokens through the
   origin, see the three measured points above; no fitted intercept).
3. `u = min_i cap_i`. If `u >= 512 && u < n_tokens`: log one line
   `micro-batch from free VRAM: %u tokens (configured %u; CUDA1 free F MiB, margin M, graph G MiB at %u)`, call
   `llama_moe_decision_fork("ubatch", ...)` with that reason, set `cparams.n_ubatch = u`, `sched_need_reserve = true`,
   `sched_reserve(); return;` (the same pattern as 1792-1797; the re-entry computes `lt` for `u` and the real reserve fits).
   Guard the re-entry with a `static const llama_context * ub_capped` like `ub_fitted` so it runs once per context.
4. Leave `common/common.cpp:2650-2670` as the fallback; with the cap it does not fire (expected: 18902 MiB free on CUDA0 at load,
   (18902 - 484) / 1.816 = 10142 -> 10112; without the margin 10400, the bisection's own answer). The expected log for GLM
   3.0/3.5: no `cudaMalloc failed`, one host pin instead of four, load ~15 s shorter.

Cost of the margin: 10400 -> ~10112 pass ceiling on GLM (-2.8 %; GLM is flat from 8192 to 12416 per the comment at
common.cpp:2240). If the ceiling matters, the better follow-up is to let `fit_ub` at 1791 raise the ubatch back to the
common.cpp maximum once the pool lends (it measures 11865 of 14520 MiB room today), not to drop the margin.

Option B (bigger lever, not for this release): bind the KV reserve size before the first reserve so `get_size_reserve()` returns the
10240-cell step at load; the graph shrinks to ~1.14 MiB/token and 16384 nearly fits on one card. Changes the reserve graph of
every model and the DSpark draft case noted at llama-kv-cache.cpp:3712; needs its own gate run.

## 06963e086d (margin 384 -> 484) and these lines

Irrelevant to them: `vram_margin` is read when the expert pool is sized (`llama-moecache-fork.cpp:7911`), after the context exists;
the failing reserve runs before the pool, against raw free VRAM, and in the release at margin 90 MiB (stored need) the same
step-down happened with the same sizes. The bump only shrinks the pool by 100 MiB per card.

## Checked against

`/p/bw/data/q5h/rel_b14244-glm35.server.log`, `rel_b14244-glm30.server.log`, `rampool11-planner-simple-glm35.server.prev.log`,
`rampool11-planner-simple-glm35.server.log`, `rampool11-planner-simple-glm30.server.log`; runs.db rows above (all `ok=0` by the
tile_spec fault, several with completed cells before it).
