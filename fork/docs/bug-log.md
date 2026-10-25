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

## 8. Lend potential promised a lend whose zone no gap could hold , FIXED

| | |
|---|---|
| found | 17:50, by the new fuzz test (no model run hit it yet) |
| repro (CLI) | `g++ -O1 -std=c++17 -I src fork/tests/kvzone_fuzz.cpp -o /tmp/kvzone_fuzz && /tmp/kvzone_fuzz 200000 1`: 5340 of 200000 random pools fail |
| where | `src/llama-kvzone-fork.h` `kvz_lend_potential` (was `kv_lend_potential`, `src/llama-moecache-fork.cpp`) |
| cause | the lend was `(gap end - zone) / bmax * bmax` over gaps that start unrounded; the placement (`kvz_gap_place`) rounds a gap start up to whole blocks. A gap that held the zone only unaligned counted for the lend and for no zone: the zone then got no place (h clamped to 0 or the layout fell back) |
| solved | the potential walks the same gaps as the placement (starts rounded up to `bmax`) |
| verified | 4 seeds x 1,000,000 random pools, 0 failures; model cells pending (kvg77) |

## 9. OLMoE long prompt: admission advertised room the pool's gaps could not hold , FIXED

| | |
|---|---|
| found | 17:20, answer-equality cells at 3500-token prompt, squeezed 0.8/1.6 |
| repro (CLI) | `eq.sh` cell c: OLMoE, `--clamp 3500 --tokens 40`, `sim-vram=819,sim-ram=1638`, poison=4: `set_input_k_idxs` assert |
| where | `llama_moe_cache_kv_tail_cap_fork`, `pp_cap_to_pool`, `find_slot` |
| cause | (a) capacity was pool bytes / row size, ignoring that zero blocks cut the pool into gaps, the zone must lie in one gap; (b) the pass cap ignored the padded read (rows rounded up to the pad); (c) `find_slot` placed cells outside the zone with no check |
| solved | capacity = largest gap behind the least lend, whole pads; `kvz_pass_cap` counts padded rows (shared template, fuzzed); `find_slot` refuses when a cell is not covered |
| verified | kvg80, cells 1.0/2.5, 0.9/2.0, 0.8/1.6: clean "request needs 3440 tokens ... can hold 2816 / 3072 / 2048", no assert; fuzz 3 seeds x 500k clean |

---

## 10. Prompt pass: capacity and claim check looked at the first card only, and used a fitted lend , FIXED

| | |
|---|---|
| found | 19:00, OLMoE 1900-token prompt, squeezed 0.8/1.6 (kvg80-kvg83) |
| repro (CLI) | `eq4.sh BUILD 1900:c:0.8:1.6`: `set_input_k_idxs` assert (cell 1343, zone 1481 under a read of 1536), later NaN in `ffn_inp-0` on CUDA1 |
| where | `llama_moe_cache_kv_tail_cap_fork`, `llama_moe_cache_kv_tail_cap_n_fork`, `llama_moe_cache_kv_tail_covers_fork` |
| cause | (a) the cap for a prompt pass used a line through the two largest measured lends, less than the 121.6 MiB the 512-token graph really borrows; (b) all three functions returned at the first card with a record of the cache: CUDA0 held 2079 cells, CUDA1 only 1752, cells 1792+ went through and card 1 read bad data |
| solved | exact measured lend when the pass size was measured; bisection over pads of lend and gap; least over every card in the pool; `covers` needs every card |
| verified | kvg84: 1900 c and 1900 d refuse cleanly ("request needs 1898 tokens ... can hold 1792 / 1024"), no assert, no NaN |

## 11. Not a bug: 1200-token answers differ from the roomy reference

The squeezed answers (cells a, c) equal the roomy run at micro-batch 256 (same md5); the roomy run at 512 differs from the one at 256. Different pass sizes change the numbers in the last bits and the greedy text diverges. Equality cells must compare at the same pass size.

## 12. KV arena (cards without an expert pool): greed, no retry, no pad rounding , FIXED (hardware run pending)

| | |
|---|---|
| found | W10 Qwen3.6, 8 GB card: pool alloc 4590 MiB failed, then the arena tried 4124 MiB and failed |
| where | `llama_moe_cache_kv_arena_start_fork` |
| cause | arena took all free VRAM, no retry smaller, capacity not in whole pads |
| solved | sized to the need, retries 25% smaller down to fixed part plus one pad, cells in whole pads (commit 91ddb88b3) |
| verified | not yet on hardware (needs a Windows build) |

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

## 2026-10-18 silent-failure review (pr-review-toolkit silent-failure-hunter, read-only) of the pool / KV / tier code

Findings read from the code, fixed on prerelease4 with a unit test each where the rule lives in a header (build pending).

| where | what failed silently | fix |
|---|---|---|
| upload worker, repacked tensor with no file location or a failed read | only an error line; the slot was then published as holding the expert (the old occupant's bytes) | the job is dropped (`skipped`), never published |
| `llama_moe_cache_alias_fork`, graph and KV do not fit | `GGML_ABORT` killed the server mid-request | returns the reason (`llama_moe_cache_alias_err_fork`), the decode fails, the request gets an error |
| same function, a lend reaching blocks under the KV zone | a warning, then the graph was placed over live KV rows | the same refusal |
| `llama_moe_cache_kv_retype_fork`, converting up | the new zone extent was never checked against the gap (could cover a class zero block) | refused in the planning step when it leaves the zone's gap; records that cannot be retyped refuse a type change |
| `pool_move_begin` timeout | a WARN, then the zone move copied under a DMA or graph still using the range | aborts with the range (a stuck user is a bug, not a state to copy under) |
| `poolx_pick_slot`, `poolx_zone_give` | a stale free-list entry could be picked; a lent or owned block could be put on the free list | picker takes the highest block that is really free; give frees only `-3` blocks |
| `llama_moe_cache_kv_ram_fork` | a failure on card 1 left card 0's host places and layers moved; success was returned when every layer moved and the request still did not fit | rollback on every failure return, and a card must meet the request |
| `--moe` values (`common/arg.cpp`) | `atoi`: `autotune=on` read as off, `slots=auto` as 0 (cache off) | numbers or on/off/true/false, anything else is an error |
| `--fork off` | prefetch staging (default auto = 3) and the predictor stayed armed | `common_fork_off_apply` zeroes them too |
| `kv-fallback` list | a typo (`RAM`, `q8`) read as "no fallback", the refusal then printed the list as typed | case and blanks ignored, unknown words and an empty list warn once |
| `kv_fit_select` caller | tier sized from sequence 0 only (`-np > 1`) | longest sequence |
| `ppx_ds_edge` | n = 1 gave an empty second dense-split range | returns n (one range) |

Left as they are, on purpose: the 10 s upload waits in the zone take and the lend (`GGML_ASSERT(spin < 200000 ...)`) and `pool_take` stay aborts, since going on would overwrite a block mid-copy. The dense-split mirror `valid` flag set before the caller's fill is by design (the copy API returns nothing to check). A failed RAM step after the q4_0 retype keeps the cells (converted in place, nothing lost).

## 2026-10-19 W10 / PC1 KV ladder runs

| where | symptom | cause | fix |
|---|---|---|---|
| W10, OLMoE, default list, 40000 tokens; `k8v8` 20000 | server abort `kv-tail: the cache writes a cell the pool has not claimed` | the zone moved behind a bigger lend into a gap smaller than the cells in use (19712 in use, zone 19608), the attention reads the cells rounded up to the pad | the layout is refused and the zone stays (`kv_layout` returns before moving); the planner keeps one pad of margin |
| PC1, Qwen3.6 with 8 GiB free, 13720 cells | the same abort (zone 13780, padded read 13824) | the lend of the real graph came 0.3% above the pass planner's fit | same fix, three repeats clean |
| W10, 40000 tokens | HTTP 500 `Compute error` after 24 s | the refused graph failed the decode (-3) | the decode returns 1 (no room for this many tokens) and the server retries a smaller batch |
| W10, RAM tier | `could not allocate 9-11 GiB of host memory` although the free RAM said it fits | free physical RAM only; W10 has no page file, the commit limit is the physical RAM and the pinned weights are committed | the estimate is the smaller of free physical RAM and free commit |
| `--moe kv-fallback=k8v8,k4v4,ram` | `--moe: expected KEY=VAL, got 'k4v4'`, the server did not start | the option string is split on commas, a word without `=` was read as a key | a word without `=` continues the previous value; unit tests |
| `--moe kv-fallback=RAM,q8` on W10 | the engine read `RAM` (the list cut at the comma) | open: the parser gives the whole list in the unit test, the W10 server log shows the first word only | `pre4o` logs the list the engine received (checking) |
| staging | `--moe pf-slots=2` turned prefetch off (up to 40% slower prompt) | the scheduler needs at least 3 slots (it keeps n_slots - 1 uploads in flight) | not changed; 3 is the floor, measured with 1, 2, 3, 4 and 6 slots on GLM |
| `prefill-own=K` (experiment, branch prefillv2) | slower with K (15.7 / 16.8 / 18.2 s for K 0 / 4 / 8) | the x4 card's link serialises at its layers' turn | experiment only, not in the release |

## 2026-10-20 f16 hand-offs between the cards cost perplexity

| where | symptom | cause | fix |
|---|---|---|---|
| dense / expert split, `src/llama-dsplit-fork.cpp` `llama_dsplit_hop_mode_fork` | forced split on Qwen3.8-Flash-Next: perplexity +0.42 +- 0.23% (IQ3_S) and +0.18 +- 0.14% (IQ1_M) against stock, 10 chunks paired; one card equal to stock to four decimals | the hidden states that cross between the cards travelled as f16 since 2026-10-13 (`409c58942`); with f32 hand-offs the same runs read +0.09 +- 0.20% and +0.09 +- 0.14%, f16 minus f32 on IQ3_S +0.33 +- 0.17% (8 of 10 chunks worse) | default back to f32; `--moe split-f16=1` keeps f16 (half the link bytes, about 2% faster at 3k); open: pf12k speed of the new default on GLM and Next at 8k-12k (gate) |

## 2026-10-22 release build died on a stale build dir, then on a missing -pthread

| where | symptom | cause | fix |
|---|---|---|---|
| `<worktree>/build-linux` of the release recipe | `No rule to make target '/usr/local/cuda-13.2/.../libcudart.so'` at 43% of the CUDA objects | the build dir is reused for incremental builds and CMake keeps absolute toolkit paths; it was configured when the image had CUDA 13.2 and the image is CUDA 13.4 now | `run.py build` writes `.image-id` into the build dir and configures afresh when the image differs |
| old `hcf/lrel/build.sh` | `undefined reference to 'pthread_join'` linking `test-backend-ops` (el8, glibc 2.28) | the old script lacked `-pthread`; `run.py build --release` has it | the old script is retired (stub, text in `hcf/lrel/old/`); releases are built with `run.py build NAME --release` |
| Windows DLL build | `undefined symbol` for `llama_moe_decision_fork`, `llama_moe_stats_json_fork`, `llama_moe_cache_prefill_ubatch_fork` | fork functions that common/ and the server call across the DLL were declared locally, not exported | declared `LLAMA_API` in `include/llama.h` (the pass planner through the C wrapper `llama_moe_prefill_pass_fork`) |

Found while checking: the `b13600` release was configured against CUDA 13.2 (its build dir cache had 54 references to 13.2, none to 13.4; its `libcublasLt` is 505 MB against 545 MB in b13742) although its archive is named `cuda-13.4`. The archive name is fixed in the script. Not verified beyond the cache and the library sizes.

## 2026-10-22 first-prompt pass size and test-harness findings (release candidate rc1/rc2)

| where | symptom | cause | fix / state |
|---|---|---|---|
| pass planner, cold start (`src/llama-moecache-fork.cpp`, `graph_tok`) | first 12k prompt after an empty state 933 prompt t/s, the same prompt with a state file 1491 (Qwen3.6-35B, cache path) | the planner's graph bytes per token took the largest ratio of any graph; the max-batch calibration's 64-token graphs (2.4 MB/token against 0.3 at 512) made every pass look 2.6x heavier, so 512-token passes "did not fit" the 1 GiB pool and ran as 320/256 | fixed (branch `cold-plan`, in rc): `graph_tok` from the biggest graph seen; cold 1493 |
| pass planner, warm start on a small card (`pp_cap_to_pool`) | rc1 restricted Qwen3.6 (`--moe sim-vram=3000`): cold 3706 but warm 1860 prompt t/s; log `kv-tail: 4608 cells of KV ... do not fit in any gap behind a lend of 1654 MiB`, server `retrying with smaller batch size` | a warm start takes the max-batch probe from the state file and lends no prompt-sized graph before its first prompt: no measured sizes, so the cap that keeps the KV zone in the pool returned the ceiling; the old inflated `graph_tok` had hidden it | fixed in rc2: without measured sizes the cap uses the seeded estimate (`graph_tok + tok_bytes`) |
| server checkpoint cut (`tools/server/server-context.cpp`) | 13.7k planner calls per 13.7k-token prompt (13-18 ms per 2048-token batch); `prefill decision` says passes=1; compute per prompt token never measured ("not measured yet" in every log) | the pass size for the checkpoint offset was asked inside the per-token loop, and every call added to the planner's totals | partly fixed in rc: asked once per batch; the totals are still polluted (the measurement stays dead): open, see T9 in the plan |
| `--moe dense-split=one,one-range=1` (research knob) | `CUDA error: invalid argument` in `ggml_cuda_xstage_read` (ggml-cuda.cu), Qwen3.8 12k prompt, both runs | not investigated | open; repro: `run.py run t23 -b coldplan2 -cs -onl q38dsone-` |
| `decode: host buffer check` log line | "MORE than the fit: an input the calibration misses" on every GLM/Qwen3.8 call | compares this pass's fit with the buffer the largest graph so far (the startup reserve) left: a false alarm, the fit itself matches (GLM 598,200 B per pass token + 2 B per token per cell) | open: compare at the largest graph's own size |
| run.py `cell` pool check (`pool_sanity`) | the 108g pool vanished in the middle of a `-nm` list; GLM then loaded unpinned and every later model without a pool | `pool_sanity` unmounts a pool smaller than the model (GLM 3.0 is 109.4 GiB); after uptime the pool cannot be mounted again (49-58 of 108 pages) | fixed in run.py: with `run -nm` the pool is never unmounted, a model bigger than the pool runs that cell without it |
| restricted test cells | `--moe-expert-cache N` / `--cpu-moe` cells logged "expert cache set by user, no auto setup": stock micro-batch 512, no automatic context, ~19 GiB idle per card | user placement skips the automatic setup by design | test rule: restricted runs use `--moe sim-vram=MiB` (goes through the automatic path) |

## 2026-10-22 pass count above b13600's: pinned host-buffer cap and the pass-count trial (release candidate inifix6-9)

| where | symptom | cause | fix / state |
|---|---|---|---|
| pinned host-buffer budget (`src/llama-context.cpp`, `mask-mb`) | GLM 3.0 12k prompt, warm: 5 passes (2 x 2117 + 3 x 2752) instead of b13600's 3 (1 x 4233 + 2 x 4096), 520 against 641 prompt t/s; with the RAM reserve added first (inifix7) 7 passes, 413 t/s | the budget was the RAM left after pinning (MemAvailable - 4 GiB): GLM pinned to the last GiB left 1.0-1.9 GiB, and the cap followed whatever RAM a run happened to have free | inifix8: the budget no longer caps the pass (`--moe mask-cap=1` restores it); before pinning the loader keeps the host buffer of the largest prompt pass free (no-alloc probe, `common_pass_buffer_reserve`). GLM 3 passes cold and warm, 645 / 671 t/s. Stopgap until the pinned-memory rewrite |
| pass-cap headroom (`pp_cap_to_pool`) | the 1.35x headroom of inifix2-6 cost passes where they fit (user rule: no more passes than before) | added against KV misfits on small cards | removed in inifix8; restricted Qwen3.6 (`sim-vram=3000`) cold now retries 2 calls smaller (`failed to find free space in the KV cache`), still 3654 against b13600's 2272 prompt t/s (accepted small-card cost) |
| pass-count trial (`llama_moe_ubadapt_cache_fork`, `src/llama-moestate-fork.cpp`) | restricted Qwen3.6, warm: second call of the 12k prompt (6080 tokens) ran 3 x 2048 while the planner said 2 x 3040 (`pass plan input: n 6080 ... ub_old 6080 -> 3040`), 3550 against 3990 t/s on that call, prompt 3807 against b13600's 4024 | the trial (not in b13600) treats calls of the same power of two as one experiment and tries the planner's count + 1 once the count is measured: the 7660-token first call measured 2 passes, the 6080-token second call was the trial | inifix9: the trial of more passes only with `--moe ubadapt-more=1`, also the half-size step of the fits-VRAM hill-climb; repro `run.py run t4 -b inifix8 -cs -onl q36r-i` |

## 2026-10-23 one-card chunked expert staging gives '!!!!' (half-slots branch)

| where | symptom | cause | fix / state |
|---|---|---|---|
| `--moe pf-chunks=2` (`src/llama-graph.cpp`, one-card parts through the expert-split path) | GLM 3.0 12k prompt: the answer is all `!`, decode 31.9 t/s (normal 17-18), 4 of 5 runs fail the answer check; the 8192-token call ran as one pass with 3 half-size slots (656-683 prompt t/s) | not found yet: the same range machinery is correct across two cards; one card holding both ranges is new (compact staging of each range, id masks, partial sums on one card) | open; repro `run.py run t4h2 -b halfslots2 -cs -onl glmc2-i`; check perplexity at 2 chunks paired against one card |
| `--moe pf-min-slots=2` (double buffering) | works: the 8192-token call ran as one pass with 2 slots (12k prompt in 2 passes instead of 3) | – | prompt 644 against 671 t/s warm (the cap still reserved all slots; halfslots3 removes that) |

## 2026-10-23 GLM-5.3-Flash crash on split prompt passes (released b13838)

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/models/glm5-next.cpp`, dense-split attention per token range | every GLM 3.5-bit request: `segfault at 38` / `at 10` in `llama_model_glm5_next::graph::graph` right after the dense-split copies are placed; GLM 3.0 only when its split engages (it chose one card, so it ran) | upstream #30640 (fused top-k mask op, `cparams.fused_top_k_mask`) stopped creating `inp_kpool->sel_mask`; the fork's per-range attention dereferenced it (`sel_mask->ne[0]`) | each range builds its mask with `build_fused_top_k_mask(kq_mask_r, top_k, score, pool_idxs, tail_r)` like the one-card path; slices of `sel_mask` only when it exists (commit 6cdedcb5c, release b13842). Repro: GLM 3.5-bit 12k prompt on b13838 |
| `src/llama-dsplit-fork.cpp`, split choice | Qwen3.8-Flash-Next: the second prompt of every server start ran the slower dense split (3k 1421 -> 1091, 12k 1535 -> 1459 prompt t/s) | the planner predicted the split ahead (model 7-27% short), the measurement was kept per session only (ini-fix 2026-10-22), and the 3k one-card prediction was 31-36% high | measured arms and calibration stored per model, cards and build; calibration prior 1.25; split weighed against the measured one-card time; every stored size loaded for the nearest-size fallback (a0cf49845, b13842) |
| `fork/run.py` prefill decisions per run | runs.db `pf_mode` of run 2 showed run 1's decision | the row read the cell's server-log copy (made at cell end) and the whole log | rows read the live server log from the previous row's position (per inode, `q5h/pf-pos.json`); new `par_gpus` column (56f8608aa) |

## 2026-10-25 dense split on two cards: helper idle, 96-token passes, auto-mode crash, live display (prerelease5 / pre5-pre10)

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-graph.cpp` `ds_plan` | `cuda0x4` 0 tokens in every split pass at a 36% share, waiting 53% of the pass (GLM 60k, auto and `split-pass=0`) | the expert card took min(card limit, pass); only the forced pass loop makes passes bigger than the limit, so every other pass went to the expert card alone | fixed (b440b137b): passes no bigger than the limit are cut by the shares; repro `run.py run -c gainers -pt i -onl glm- -ps 60k1k -a "moe:dense-split=1 moe:split-pass=0"` |
| `src/llama-moecache-fork.cpp` `pp_cap_to_pool` | after the first 32768-token call, the next call ran in passes of 96 tokens (239 passes on 60k, the cell ran 20+ min, GPUs idle) | the line through two graph sizes 2 tokens apart (6552, 6554) gave a slope of 124.8 MB per token (real 1.4 MB) | fixed: the slope is clamped to the smallest measured bytes per token (a graph's fixed part is never negative); same repro |
| `src/llama-dsplit-fork.cpp` share rule | helper 24% of the tokens, idle 70-80% of each pass | the zero-wait factor fell from 0.75 to 0.33 on expert-card waits the helper did not cause | fixed: equal dense time per card (measured dense us/token + hand-off), plus the share climb; pre7 Qwen 60k 1910 vs 1708 best forced split |
| `src/llama-dsplit-fork.cpp` planner | every auto prompt on one card ("close call, one card") although the split was 15-25% faster | the one-card estimate was replaced by its measurement, the split estimate stayed modelled (both off by ~1.33x) | fixed: the split estimate is scaled by the same correction (ratio trusted, not size); pre6 GLM 698 vs 566, MiMo 985 vs 795 |
| `src/llama-dsplit-fork.cpp` load-time timing (pre7/pre8) | auto mode, MiMo 60k: `CUDA error: out of memory` in `cuMemCreate` at the first prompt (96 MiB free per card); forced dense split did not crash | the matmul/attention timing ran at the first prompt decision, after the pool had taken the VRAM; forced mode never reaches the planner | fixed (4ccdb2ca5): timed once when the first context is created, before the KV and pool; repro `run.py run -c gainers -onl mimo- -ps 60k1k -pt i -b pre7` |
| `fork/run.py` live view | stale `engine`/`cpu` rows, 1-row jumps, whole-screen repaint on long prompts | the engine line grew past the terminal width (two pass sizes, pcie rx) and the pass table (12 passes) outgrew the screen | fixed: transfer rates on their own line, the last 6 passes in the table |
| `src/llama-context.cpp` / `common` `-ub` | `-ub N` set the micro-batch exactly and skipped the automatic sizing | user wanted a clamp for sweeps | `-ub` now clamps the automatic maximum (never raises it) |
