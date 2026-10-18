# Bug log: found, solved, open

One entry per bug: when it was found, how to repro it on the command line, where in the source it lives (file, function, line), the cause and how it was solved, and what verified the fix. Newest first. Append an entry with every fix; a bug is closed only when its repro cell passes.

Lines are those of the engine tree `/p/bw/wt-kvgrow` (branch `kv-grow`) at the commit "KV zone and pool: lend counts the graph's own KV cells ..." plus the uncommitted kv-arena refusal; they move with edits. Repro cells run from `/p/bw/wt-main` with `export PERF_BUILDS=/p/bw/rels/devbuilds PERF_MIN_AVAIL_GB=4 PERF_WATCH=1`, prompt `/p/bw/data/q5h/prompt_real_262k.txt`, outputs in `/p/bw/data/q5h/`. Tools: `--moe poison=3|4`, `GGML_SCHED_NAN_CHECK=1`, `--moe ledger=1` with `fork/pool_ledger.py --map|--blame`, `--hog-gpu GiB --hog-ram GiB` (memory left free per card), the squeeze driver `descend.py`.

All times are 2026-10-18, PC1 (RTX 3090 x16 + x4), local time.

---

## 1. KV arena: no room for the KV, the graph is built over unattached records (abort) , FIXED (refusal)

| | |
|---|---|
| found | 16:23 (squeeze bisection cell, OLMoE, 0.471 GiB VRAM / 0.942 GiB RAM left; on the backlog as "kv-arena abort") |
| repro (CLI) | `python3 fork/run.py cell olmo-ref OLMoE-1B-7B-0924-Q4_K_M.gguf kvg74 1 pf128k "--moe poison=3,ledger=1,mode=cache,prefill-split=1,dense-split=1,state=0 -at off" --prompt @prompt_real_262k.txt --ctx -1 --clamp 1 --tokens 1 --hog-gpu 0.471 --hog-ram 0.942 --keep-loops --env GGML_SCHED_NAN_CHECK=1` (model path `/m/o/4/`) |
| symptom | `ggml-backend.cpp: GGML_ASSERT(tensor->buffer == NULL)`, `tensor 'cache_kv_record' (op NONE, data (nil)) already has a buffer (CUDA1)`, from `ggml_gallocr_alloc_graph` <- `process_ubatch` |
| where | `src/llama-moecache-fork.cpp`: `llama_moe_cache_kv_arena_start_fork` (l. 3735); the three early `continue` after `kv-arena: ... no room for the KV` / `could not allocate` / `fixed records do not fit` left `g_kvt` records unattached. Called from `llama_context::decode` (`src/llama-context.cpp` l. 2954) |
| cause | with no room for MoE cache slots the card has no pool, the KV goes to a separate arena; below the 384 MiB safety margin plus 64 MiB the arena is skipped, `decode` went on and the graph used KV records that have a buffer but no memory |
| solved | each failing branch stores the reason (`g_kv_arena_err`); new `llama_moe_cache_kv_arena_unplaced_fork` (l. 3791) says so while a record stays unattached; `decode` logs `decode: kv-arena: CUDA1 has 184.0 MiB of VRAM free, the KV cache needs more than the 384 MiB safety margin and 64 MiB: leave more VRAM free, or use a smaller model or context` and returns -2 (also frees the buffer in the "fixed part" branch, which leaked it) |
| verified | 16:33 same cell on `kvg75`: the message twice, no assert |
| note | the first attempt (a check after `kv_tail_prepare` in `process_ubatch`) did not trigger: no pool existed at all, so that code never ran. The check is still in as a second guard (`llama_moe_cache_kv_tail_unplaced_fork`) |

## 2. Dense split on Qwen3.5 MoE (Qwen3.6) and Qwen3-Next (Qwen3.8): `out_ids` never read , FIXED

| | |
|---|---|
| found | 14:28 (first Qwen3.6 squeeze cell); diagnosed 15:55 to 16:08 |
| repro (CLI) | `python3 fork/run.py cell q36-repro Qwen3.6-35B-A3B-GSQ-hybrid.gguf kvg60 1 pf128k "--moe poison=2,ledger=1,mode=cache,prefill-split=1,dense-split=1,state=0 -at off" --prompt @prompt_real_262k.txt --chat "Summarize the text below in a few sentences." --ctx -1 --clamp 1 --tokens 1 --hog-gpu 4.687 --hog-ram 11.862 --keep-loops` (model path `/m/q/36/`); ~30 s |
| symptom | `GGML_ASSERT(buffer) failed` at `ggml-backend.cpp:347` (`ggml_backend_buffer_get_usage`) via `ggml_backend_buffer_is_host` <- `llm_graph_input_out_ids::set_input` <- `process_ubatch` <- `moe_probe_costs_fork` (the load-time cost probe: a 512-token graph, 1 output row) |
| where | `src/models/qwen35moe.cpp`: dense-split branch (`ds_plan`, l. 186 to 339) ends with `inpL = ds_join(...)` (l. 337) and the layer loop that crops to the output rows (`for (... && ds_r.empty())`, l. 346, `ggml_get_rows(cur, inp_out_ids)` at its last layer) is skipped; same in `src/models/qwen3next.cpp` (`ds_join` l. 315, loop l. 324). Consumer: `src/llama-graph.cpp` `llm_graph_input_out_ids::set_input` (l. 624, assert at l. 633) |
| cause | nothing read `out_ids`, so the scheduler never allocated that input and `set_input` asserted on its null buffer; the logits would also have been those of the first rows, not the output rows |
| how it was traced | a diagnostic in `set_input` (tensor `out_ids`, ne0 1, flag input, data nil); `LLAMA_REBUILD_DEBUG=1` (new, `process_ubatch`) showing two allocations of the same 8472-node graph (the zone moved) and no reader of `out_ids`; reading the model files: `mimo2.cpp` crops right after `ds_join`, these two did not |
| solved | after `ds_join`: `if (crop_before_nextn(inp_out_ids)) inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);` in both files (commit "dense split (Qwen3.5 MoE, Qwen3-Next): take the output rows after the ranges are joined") |
| verified | 16:08 same cell on `kvg67`: answer "Here", descent 4.687 and 2.343 GiB good. `qwen3next` changed the same way, not run yet; `deepseek4`, `qwen4exp`, `mimo2` already crop |

## 3. KV pad rows never zeroed when the alias is dropped: NaN in the first graph , FIXED

| | |
|---|---|
| found | 14:43 (squeeze cell 5, OLMoE, 0.786 GiB VRAM / 1.57 GiB RAM left, `poison=2` + NaN check); diagnosed 16:12; fixed 16:21 |
| repro (CLI) | `python3 fork/run.py cell olmo-diag OLMoE-1B-7B-0924-Q4_K_M.gguf kvg70 1 pf128k "--moe poison=2,ledger=1,mode=cache,prefill-split=1,dense-split=1,state=0 -at off" --prompt @prompt_real_262k.txt --ctx -1 --clamp 1 --tokens 1 --hog-gpu 0.786 --hog-ram 1.57 --keep-loops --env GGML_SCHED_NAN_CHECK=1`; ~15 s (model path `/m/o/4/`). No hog and `--cpu-moe --moe-expert-cache 6` does not show it: it needs the alias dropped and little free VRAM |
| symptom | `nan-check: graph 1, split 1 of 66 on CUDA1, node 31 of 41: ADD ffn_inp-0 [2048,4] f32 holds 8192 bad values of 8192`; the process exits (`pool-dump: first NaN`). Without the check the answer was fine (the warm-up graph's output is discarded) |
| where | `src/llama-moecache-fork.cpp`: `kv_tail_pad_zero` (l. 3930) zeroes the K/V rows the attention reads past the cells in use, and is called only inside `llama_moe_cache_alias_fork` (l. 3990), which returns at l. 3998 (`!alias_on()`) once `llama_moe_cache_alias_drop_fork` has run ("the pool cannot lend the graph even at micro-batch 512: compute buffers stay separate") |
| cause | with separate compute buffers the KV zone still lives in the pool but its pad rows were never zeroed: NaN under poison, stale expert bytes otherwise (the MiMo `!!!!` mechanism, entry 6, on a path it did not cover). Cells with a pool that still lends the graph (0.825 GiB and up) were clean |
| solved | `llama_moe_cache_alias_fork` calls `kv_tail_pad_zero(mc)` when the pool has KV zones but the alias is off (commit "KV zone and pool: ...") |
| verified | 16:21 same cell on `kvg71`: 0 NaN lines, answer normal; OLMoE descent on that build |
| why it took long | `poison=1` (finite ids) passed the repro; the NaN report was never printed (the pool-dump hook exits first); `GGML_SCHED_NAN_CHECK=2` runs out of VRAM here. All three fixed (tooling section) |

## 4. Lend potential ignored the KV cells of the graph asked about , FIXED

| | |
|---|---|
| found | ~16:00 (OLMoE with a 185 MiB pool on the gap-aware build: abort `kv-tail: a 512-token graph needs 179.9 MiB lent ... 512 cells of KV (28.0 MiB) do not fit behind it in the pool of 184.6 MiB`; on the build before it the lend overlapped the KV zone and gave the NaN of entry 3) |
| repro (CLI) | entry 3, with `kvg62` |
| where | `src/llama-moecache-fork.cpp`: `kv_lend_potential` (l. 3078), used by `llama_moe_cache_alias_fit_fork` (graph fit) and the staging slot count |
| cause | the zone counted at its current size (0 cells at the first graph), so a graph that takes nearly the whole pool "fitted" and the KV then had no room behind it |
| solved | the zone counts at least `min_cells` = the graph's token count; the lend reaches only the start of the last gap that holds the zone (commit "KV zone and pool: ...") |
| verified | the graph no longer fits at 512, the alias drops (entry 3), the cell runs |

## 5. Zone over a class zero block / lend too big for the fixed KV part (MiMo 12k, 15 GiB held) , FIXED

| | |
|---|---|
| found | 14:40 (`cmp-kvg60`, `cmp-kvg61`: 2 to 6 pool-ledger overlaps per run, assert `llama_moe_cache_kv_tail_covers_fork` at cell 11672) |
| repro (CLI) | `python3 fork/run.py cell cmp-kvgN MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf kvgN 1 pf128k "--moe dense-split=one,ledger=1" --prompt @prompt_real_262k.txt --chat "Summarize the text below in a few sentences." --ctx -1 --clamp 12000 --tokens 60 --hog-gpu 15 --keep-loops --env GGML_SCHED_NAN_CHECK=1` (model path `/m/m/3/`); ~2 min; then `fork/pool_ledger.py cmp-kvgN.ledger.csv.gz --map` |
| where | `src/llama-moecache-fork.cpp`: `kv_gap_fit` (l. 3524), `kv_layout` (l. 3556), `kv_lend_potential` (l. 3078) |
| cause | the shrink-to-fit used "pool end less the zone" and let the 12k graph lend 10 GiB of an 11.6 GiB pool; the 1.6 GiB fixed sliding-window part then fitted no gap between the class zero blocks and lay on two of them |
| solved | zone placed in a gap between zero blocks, never over one (commit "kv zone: placed in a gap between class zero blocks ..."), and the lend limited to what leaves such a gap (entry 4) |
| verified | 15:41 `cmp-kvg62`: 0 overlaps, no assert, clean answer, 5536 uploads none crossed; decode 11.88 t/s, prompt 222.8 t/s (277.2 before, one run, unexplained) |

## 6. MiMo `!!!!` from ~9k tokens with 15 GiB held per card: zone horizon , FIXED

| | |
|---|---|
| found | ~12:00 (`pf128k` rows kvg46 to kvg51 garbled until 12:07) |
| repro (CLI) | entry 5 with a 9k to 12k clamp on `kvg51` or earlier |
| where | `src/llama-moecache-fork.cpp` `kv_layout` (horizon `h`, `h_min` at l. 3566) and `llama_moe_cache_kv_tail_covers_fork` (l. 3893) |
| cause | the horizon was clamped to the bare cell count (11673) while the attention reads cells rounded up to the pad (256): 103 pad rows lay in expert blocks behind the zone; NaN-looking expert bytes x masked softmax weight 0 = NaN in layer 0 of the last query chunk |
| solved | horizon at least the padded planned cells; `covers` checks the padded read (commit "zone horizon vs padded attention read") |
| verified | 9k, 12k and 35k prompts ok |

## 7. Qwen3.8 `!!!!` on long prompts: stale padding behind quantized weights copied into the pool , FIXED

| | |
|---|---|
| found | 10:00 (root cause by `GGML_SCHED_NAN_CHECK=1`; first garbled `pf128k` rows 01:25) |
| repro (CLI) | `fork/run.py cell` on `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (model path `/m/q/1/`), one 40k chat run with the learned state `/p/bw/data/q5h/qwen38-split-garbage-repro.ini` (needs the learned split shares) |
| where | `ggml/src/ggml-backend.cpp`: scheduler input copies (`ggml_backend_tensor_zero_tail_fork` declared l. 2075, called l. 2130); CUDA MMQ/MMVQ clear padding only for compute buffers |
| cause | the compute alias puts graph memory in the pool (usage WEIGHTS), so the CUDA kernels skip the clear of the padding behind a quantized tensor; a cross-card copy or dense-split fill of `blk.5.ffn_down_shexp` [640, 2560] IQ4_NL (rows not a multiple of 512) had expert bytes behind its last row: NaN in the last element of each row |
| solved | `ggml_backend_tensor_zero_tail_fork` after those copies (commit "stale padding behind quantized copies"); -1 expert ids in split gathers clamped |
| verified | the reproducer, 40k, ok |

---

## Tooling bugs (they cost time, fixed)

| tool | problem | fix |
|---|---|---|
| `--moe poison=1` | finite ids do not trip the NaN check: the OLMoE repro passed (14:57) | any `--moe poison=N` is now poison=3: NaN fill with the filling code's 4-letter name and fill serial in a few words; `poison=4` stores the 8-byte call-site address instead (`pool_fill`, `src/llama-moecache-fork.cpp` l. 1560); the NaN report prints the writers found inside the bad tensor |
| NaN check | `ggml_backend_sched_nan_check` (`ggml/src/ggml-backend.cpp` l. 2468) called the pool-dump hook, which exits the process, before printing the tensor | report first, then the hook; also prints each bad input's buffer size and offset |
| `descend.py` | a hog of 0 means no hog: the "0 / 0" cell ran with full memory and broke the bisection; out-of-memory cells were labelled `refused/ram`; the descent stopped at the first failure | floor 0.02 GiB, alloc failures are `SKIP`, halving goes on to the floor, then a three-way bisection (GOOD / OOM / BROKEN) between the lowest good and the highest OOM below it; `ONLY=L,R` pushes one config with longer prompts |
| `run.py build` | ran the full regress (24+ min) | fix builds use `--no-regress` |
| `run.py post` | a crashed run kept no answer text | with `PERF_WATCH=1` `/completion` is streamed, the text so far goes to `PARTIAL` and `q5h/<tag>-<test>.answer.txt`, a failed row stores it |
| Windows builds | need a committed tree (`--commit`), not an uncommitted one | commit first |
