# Bug log: found, solved, open

One entry per bug: when it was found, how to repro it on the command line, where in the source it lives (file, function, line), the cause and how it was solved, and what verified the fix. Chronological: new sections are added at the end. Append an entry with every fix; a bug is closed only when its repro cell passes.

Lines are those of the engine tree `/p/bw/wt-kvgrow` (branch `kv-grow`) at the commit "KV zone and pool: lend counts the graph's own KV cells ..." plus the uncommitted kv-arena refusal; they move with edits. Repro cells run from `/p/bw/wt-main` with `export PERF_BUILDS=/p/bw/rels/devbuilds PERF_MIN_AVAIL_GB=4 PERF_WATCH=1`, prompt `/p/bw/data/q5h/prompt_real_262k.txt`, outputs in `/p/bw/data/q5h/`. Tools: `--moe poison=3|4`, `GGML_SCHED_NAN_CHECK=1`, `--moe ledger=1` with `fork/pool_ledger.py --map|--blame`, `--hog-gpu GiB --hog-ram GiB` (memory left free per card), the squeeze driver `descend.py`.

All times are 2026-10-18, PC1 (RTX 3090 x16 + x4), local time.

---

## Classes: the root cause of each entry

Added 2026-11-12 (user: "keep storing all the found bugs in bug list so later we can identify major design flaw causing them and fix it more systematically"). Every entry from the section "2026-11-10 evening" on carries one class: the root cause, not the symptom. The tally is brought up to date with every batch of entries. The entries before that section are not classified yet.

| class | what it is | entries, 2026-11-10 evening to 2026-11-12 |
|---|---|---|
| stale-number | a capacity, estimate, key or stored/learned value computed at another moment, state or path than where it is used | 12 |
| flag-coupling | a global switch or mode set in one layer whose side effect another layer relies on, so a path changes silently | 10 |
| harness | the test tool or a debug switch changes or misreports what it measures | 10 |
| pool-owner | two users of the same pool bytes with no single owner or ledger (lend, KV zone, expert blocks, ring, class grid) | 7 |
| card-asymmetry | code written for one card or the first card meeting another layout (home card, expert card, links) | 6 |
| past-end | a read or write past an allocation's used end at the access granularity (row padding, tiles, pad rows, a view cleared at its nominal size) | 3 |
| silent-fallback | a refusal or failure that degrades the path without a log line or row marker | 2 |
| request-carryover | state left by one request, call or pass that the next one reads | 0 |
| kernel | a GPU kernel's own arithmetic or bounds | 0 |
| other | named in the entry | 5 |

First reading of these 55 entries (three days, to be redone once the older entries are classified): three classes hold 29 of them, each with one shared cause.

- stale-number: one quantity is computed in several places from different inputs (start-up fit, pass cap, lend line, stored needs, learned times). One function per quantity, fed the live state where it is used; a stored value keyed by everything it depends on.
- flag-coupling: the path of a prompt call lives in global flags (`g_pp_mb`, `g_pp_dec`, the staging mode) that other layers read a meaning into. One plan object per call, handed to the graph builder, which logs what it built against it.
- pool-owner: the pool has several users and no single table of who owns which bytes (planned in `fork/docs/pool-manager-plan.md`).

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
| `llama_context::decode` forced pass loop (`--moe dense-split=1`, `split-pass=1`) | GLM 3.0 60k1k run 2: the third call (10336 tokens at 62.5k cells) failed: `init_batch: failed to prepare attention ubatches` even after the KV went to q8_0 and q4_0 | the loop's passes of up to 15588 tokens lent ~10 GiB each and left the pools short ("the card limit 10336 of the context does not fit the pools now"); the one-pass call after them found no KV slot | open for the loop; the loop is off by default (pre11): the share-cut split was faster on Qwen3.8 and GLM. Repro: `run.py run -c gainers -pt i -onl glm- -ps 60k1k -b pre10 -ds on` (run 2) |

## 2026-10-26 DeepSeek 4.1 without -c: every generated token is id 0 (empty answers, fake decode speed)

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-kv-cache.cpp` kv-tail record `rows`, `src/llama-kv-cache-dsv4.cpp` | every DeepSeek 4.1 cell without `-c` (pf1k, pf2k, pf16k, chat endpoint): content empty or only `<think>`, all 1024 tokens id 0, decode 40-44 t/s on every build ds41i2..deep-seek; cells with `-c 4096` (edit4c, tetrisc) gave real text | the DSV4 compressed caches are written by the graph at row pos / ratio, never cells marked used: `used_max_p1()` = 0. Without `-c` (context 1M) the per-token compressed cache is the zone's only growing record, so the zone counted 0 rows in use: a zone move copied none of its rows (the first server call's rows stayed in blocks given back to the experts) and its pad rows were never zeroed; the attention read expert bytes as K (NaN), argmax token 0 | fixed (9a2786b5c, build dsfix1): the cache owner registers its rows in use (`llama_kv_cache_tail_rows_fork`), DSV4 reports (last position + 1) / ratio. Repro `run.py run dscmp -b deep-seek`; dsfix1: real code review, uniq4 0.96 / 0.92, decode 20-23 t/s. Every DeepSeek decode number before dsfix1 is invalid (prompt speeds stand) |
| `src/llama-kv-cache-dsv4.cpp` dummy compressor block | DeepSeek 4.1 16k on one GPU (`CUDA_VISIBLE_DEVICES=1`): `CUDA error: an illegal memory access` at the first 1-token decode after the 16384-token pass, every run (graphs off + blocking launches: in a kernel launch of the decode graph); two cards: no crash | the dummy block of a ubatch with no completed block writes row kv_size - 1. Since "a compressed cache grows with the zone" (ds41) the V4.1 ratio-2 cache is a growing record too: its row 524287 lies ~2.5 GiB past the zone start (forward rows, 5120 B combined row). One card: zone at 14.8 GiB, the row past the pool end. Two cards: zone at ~6.3 GiB of a 20.3 GiB pool, the row inside an expert block: 3840 B written into a cached expert every other decode token (silent) | fixed (dde3b025e, build fix_dsv4_dummy_block_row_inside_zone): the dummy writes the sequence's next row to complete (not visible yet, inside the zone, overwritten by its block). State: one-GPU 16k cell queued; every two-card DeepSeek decode number and answer since the growing compressed cache may carry the corrupted expert (ppl pair pending) |
| DeepSeek 4.1 first request, `-c 4096` (edit4c run 1) | HTTP 500 `Compute error`: CUDA out of memory on pcuda1x16 (386 MiB outside the pool) in the first 3392-token pass; run 2 passed with the stored need | the pool lends a graph the CUDA scratch its expert matmuls ask for, but the start value was a guess: 61,440 B per batch token (6 used experts x 5120 x 2), against 193,072 B per token on the sorted-rows path the 3389-token pass takes (48 B ids + 178,176 B rows as f32 in and out + 14,848 B f16 per-expert copies, plus a fixed 22.5 MiB converted-weights buffer). Measured on pcuda1x16 with `--moe log=a`: the backend asked for 638.8 MiB, the computed lend is 648.0 MiB, the old guess lent 198.6 MiB (the scratch depends on the matmul path: MMVQ, MMQ, MMF or the sorting fallback). The graph got too little lend, the rest was cudaMalloc'd outside the pool on a card with 386 MiB free. Run 2 passed only because the stored rate had been learned from run 1's failure | fixed (72a1712dc, build fix_first_pass_scratch_computed): the CUDA backend computes the scratch per path from the real shapes (`ggml_backend_cuda_mmid_scratch_fork`, type, row size, experts, tokens); the pool asks it before the first pass, the learned rate starts at 0. Verified: edit4c passes run 1 (decode 21.0 t/s, prompt 124.8 t/s) and run 2 (23.1, 162.9), no HTTP 500. Still open: DSV4 cannot take back a failed prompt call, so any future out-of-memory in a first pass still ends in HTTP 500 (follow-up: state snapshot before the call); the error text "graph was not run" from `ggml_cuda_oom_report` is misleading |
| `src/llama-moecache-fork.cpp` scratch lend, `ggml/src/ggml-backend.cpp` scheduler, `ggml/src/ggml-cuda/*` scratch mirrors | DeepSeek 4.1 after the first-request fix: every card was lent 3416 MiB of CUDA scratch for a 3389-token pass whose nodes took at most 678 MB (`GGML_CUDA_SCRATCH_CHECK=1`, the per-node took-vs-computed check): about 2.8 GB of expert blocks per card idle in every prompt pass | two guesses left after the first fix: a learned per-token floor (a 2-token decode graph's flash attention took 2.02 MiB against a 2 MiB lend, so the rate became 1 MB per token for every later graph), and the whole batch's scratch lent on each card although each card runs only its own splits' nodes | fixed (branch fix_scratch_per_card_from_graph, build of the same name): per split, the scheduler asks the backend for the largest scratch any node of that split takes on that card, through exact mirrors of each op's own pool allocations (MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT plan mode, TOP_K, ARGSORT, SOFT_MAX, SUM, MEAN, CUMSUM, SOLVE_TRI, SSM_SCAN, and the gather of a gappy split input copied from another card), cached per split graph uid; the floor is gone. 3389-token pass: cuda0x4 3416 -> 119.2 MiB (real 119.2), pcuda1x16 3416 -> 646.5 MiB (real 638.9); decode 10-12 -> 2.5 MiB; lend cost per decode step 0.31 -> 0.34 ms. `tests/test-scratch-fork.cpp` runs each real op against its mirror: 5039 cases, 0 failed. `GGML_CUDA_SCRATCH_CHECK=1` cells on DeepSeek 4.1, MiMo, GLM 3.0, OLMoE, Qwen3.6, Qwen3.8 IQ1_M and IQ3_S: no node took other than computed. Correction 2026-10-27 (release candidate rc_deepseek_fixes, DeepSeek edit4c, 3385-token pass): 29 MUL_MAT_ID nodes on pcuda1x16 took 7.9-43.9 MB (1-6%) LESS than computed (636-669 MB against 677 MB), none took more: on the sorted-rows path the real scratch depends on how the tokens spread over the experts, so the mirror is an upper bound there (safe: a lend a little too big, never too small). Unmirrored pool users left (no model here runs them): OUT_PROD, CROSS_ENTROPY_LOSS, CONV_2D/3D, allreduce, copy_slices pointer arrays |

## 2026-10-26 DSpark draft on DeepSeek 4.1: VRAM taken from the target, aborts, crash at the first request

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-moecache-fork.cpp` kv-tail pool-expected flag | every DSpark cell: expert cache 28.6 instead of 41.3 GiB, prompt micro-batch 16384 -> 640, a 7.05 GiB KV arena, prompt graph reserved at 13.5 GiB | the flag was one global set by every context at its creation; the draft (created after the target) turned it off: the target's reserves sized its KV at the full 1M cells, and the draft's load-time decode took the target's KV records into a fixed full-context arena | fixed (cc4bfa6e1, build fix_dspark_draft_keeps_kv_in_pool_and_runs_on_one_card): records remember their context's flag; only that context hands its records to an arena. No-draft bar unchanged (pools 20.3 GiB each, micro-batch 16384) |
| `src/llama-context.cpp` draft backends | `-devd CUDA1` (draft weights on one card): abort on the borrowed output.weight | the draft context had backends only for its own devices; the target's output.weight lives on the other card | fixed (11126c1c8): a backend for every card holding target tensors the draft shares |
| `fork/run.py` cell args | `-otd "blk\..*=CUDA1"` arms: `unknown buffer type` | `args.split()` kept the quote characters | fixed (`argv_of`, shlex) |
| `common/common.cpp` `common_context_can_seq_rm`, `src/llama-kv-cache-dsv4.cpp` | after the first fix, every DSpark cell: `GGML_ASSERT(tensor->data != NULL && "tensor not allocated")` at the first request (twice per cell) | the draft gives the target rollback slots, so the load probe returned before its decode: the target's pool (and its KV records) started only in the first request, whose sequence removal first cleared DSV4 records without memory; before the first fix the draft's decode had taken them | fixed (b05ee8132, build fix_draft_crash_at_first_request): contexts with rollback slots run the probe decode too; the DSV4 clear returns on a tensor without memory |
| `fork/run.py` `cmd_cell` | `PERF_LOG=rp run dspark8 ...`: every server still got `--moe log=r`, no periodic cache stats in any depth cell | `cmd_cell` set `PERF_LOG="r"` in the cell's environment over the caller's value | fixed: `PERF_LOG=LOG_LETTERS` (the caller's letters, r by default); quick, bisect, plcal and baseline keep r on purpose (clean speed runs) |
| `common/common.cpp` `common_hog` (`--moe hog-<card>=MB`) | the VRAM control cell `--moe hog-CUDA0=7943,hog-CUDA1=253`: no `hog:` line, pools 20374.6 / 20327.1 MiB as without it (stopped as a trash run) | the `--moe` parser stores every key in lower case (`hog-cuda0`, `common/arg.cpp`); the lookup used the backend's spelling `hog-CUDA0` | fixed (build fix_hog_knob_lowercase_card_name): the card name is lowercased for the lookup |
| `ggml/src/ggml-cuda/argmax.cu` `argmax_f32` (used by the DSpark markov head, `src/models/dflash.cpp` `build_dspark_markov_head`) | DeepSeek 4.1 + Lucebox draft at spec depth 2 and 3, 3D tetris prompt, chat, reasoning off (`--nothink`), `-c 8192`: `CUDA error: an illegal memory access` (`ggml_backend_cuda_thread_sync_fork` / `set_tensor_nosync_fork`) at the third verify step after the prompt, every run (3 of 3); depth 1 passes. Passes with `CUDA_LAUNCH_BLOCKING=1`, with JIT uploads off (`--moe upload-now=0`); the same depth on the tetris prompt with reasoning on, the 1k prompt with `--nothink`, and the 1k prompt with `-c 8192` all pass. Repro: `run.py run dsbug15 -b fix_cuda_argmax_returns_zero_for_nonfinite_row -nm -nodl` at night power (200/200 W; fork/runs/dsbug15.csv, 64 tokens, 3 runs per depth). At 200/200 W the plain cell crashes only some of the time (3 of 3 passed once, 4 of 5 cells crashed in the matrix); the logging build (`LLAMA_PROC_DBG=1`) crashed 2 of 2. Matrix dsbug13 (night power): draft on CPU passes, draft on CUDA0 or CUDA1 crashes, kv-tail off crashes, so the draft's GPU graphs are involved | the CUDA `argmax` starts at `argmax = -1` and takes only values `> -FLT_MAX`: a row with no such value (all NaN or all -inf) answers -1, where the CPU op answers 0. The markov head feeds that result to `ggml_get_rows(markov_w1, prev)` for block position 2 onward, so it reads one row BEFORE the weight tensor; whether that faults depends on what memory lies before `markov_w1` (hence intermittent, hence only depth >= 2: depth 1 has no argmax chain). Located by `LLAMA_PROC_DBG=2` (sync after each draft decode): the fault is inside the draft's 2-token block graph, after the 4-token inject graph synced clean. Not a race between graphs: level 4 (sync of both contexts before and after every decode) still crashed. Ruled out earlier: draft lending from the pool, after-prompt refill, pool overlaps (ledger), bad host-set indices (idx-check, whose graph-130 abort was a false positive: it read a computed split-copy index before the graph ran), kv-tail, `-c` | FIXED (6faa18876, build fix_cuda_argmax_returns_zero_for_nonfinite_row): the kernel answers 0 for a row with no finite value, like the CPU. Verified at 200/200 W: the level-1 cell (crashed 2 of 2 before) 3 of 3 pass; plain depth 2 and depth 3, 3 runs each, all pass (19.64 and 19.34 t/s). Open question: why the draft logits row is non-finite at all on some steps (not found; a bad draft token costs one round of acceptance, no crash now). Earlier guard 135bb6e46 (draft context never lends from the pool) stays: a real defect, not this crash

## 2026-10-27 release candidate repros (rc_deepseek_fixes, branch fix_scratch_per_card_from_graph)

Every bug fixed since release b14020 rerun on one clean build of the candidate, at day power (220/280 W): DeepSeek without `-c` (2 of 2 real answers, uniq4 0.94/0.88), edit4c first request with an empty state (no HTTP 500), 16k on one GPU (2 of 2, no illegal access), DSpark depth 2 and 3 on the tetris prompt (4 of 4, acceptance 89% / 88%), `GGML_CUDA_SCRATCH_CHECK=1` (no node took more than computed). Lists `rc_fix_repro1`, `dsbug15`.

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` scratch region (`g_cuda_scratch`), `src/llama-moecache-fork.cpp` lend | DSpark depth 3 (4-token verify graphs): `SCRATCH SHORT: a 4-token graph asked for 5283840 bytes, 2642176 were computed and lent` on CUDA0, 16 times; `GGML_CUDA_SCRATCH_CHECK=1` named no node of CUDA0 (every node took exactly its computed scratch) | the scratch region is one per card and stays set from one lend of the target to the next; the draft context's graphs, run between two target steps on their own CUDA stream, took their scratch from it too (from blocks lent to the target's graph: the two contexts could be handed the same offsets for kernels on two streams), and their asks went into the high-water mark compared against the target's lend: the 5.28 MB was a draft node | fixed, commit "the CUDA scratch region serves only the context it is lent to" (builds fix_scratch_region_serves_only_its_context, then the candidate rc_deepseek_dspark_fixes; unit test 5043 cases, 0 failed; DSpark depth 3: no SCRATCH SHORT, acceptance 88% as before): the region serves only the backend context it is lent to (the lend passes its backend); another context's scratch comes from the device pool as before the region existed; the scratch check skips other contexts' nodes; `tests/test-scratch-fork.cpp` checks that a region lent to another context on the card takes nothing. Repro `run.py run rc_fix_repro2 -b rc_deepseek_fixes -day -nodl` |
| `GGML_CUDA_SCRATCH_CHECK=1`, MUL_MAT_ID sorted-rows path | 29 nodes on pcuda1x16 took 1-6% less than computed (edit4c 3385-token pass), 26 on a 268-token pass | the real scratch depends on how the tokens spread over the experts; the mirror is the bound for any spread (the unit test already treats spread routing as `asked <= computed`) | not a bug: safe direction (lend a little too big) |
| `src/llama-moecache-fork.cpp` `vram_margin` / `vram_key` (stored VRAM need outside the pool) | DSpark depth 3 after plain runs, no clean state: the server died at load at every start, `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 796.79 MiB on device 0: cudaMalloc failed: out of memory` (then 725.50 MiB without pipeline parallelism); the sizing line said `margin 102 MiB (stored need 38 MiB + safety, warm start)`, free after the pool 104 MiB | the pool keeps a margin from the need an earlier start measured on the card, build and driver; plain runs stored 38 MiB. The draft's compute buffer grows by 256 MiB after the pool is sized (the draft's first decode after the speculative setup turned on its extra outputs, `common_context_can_seq_rm(ctx_dft)`), so a draft start needs ~295 MiB outside the pool. A draft run never stored its own figure (its target calls are all verify batches of several tokens, the store waits for a one-token decode sample), so the plain figure stayed and every draft start failed: a stuck state. The release candidate rc_deepseek_fixes has the same code | fixed, commit "a run with a draft keeps its own stored VRAM need outside the pool" (build fix_draft_keeps_its_own_vram_need, candidate rc_deepseek_dspark_fixes): `llama_init_from_model` / `llama_free` register the live contexts' models; the stored need is keyed by the other contexts' model bytes (`...with4695.warm`), a plain run keeps the key it had. Verified in that order without clean state: plain (stored 34 MiB), DSpark depth 3 (margin 384, loads), DSpark depth 1 with a 1k answer (loads, stored 295 MiB), plain again (margin 99, its own figure). List `rc_fix_draftneed1`. Open: a reserve that fails at load is still not counted as need (only decode-time CUDA out-of-memory errors are), so a draft needing more than the 384 MiB default would fail every start until a run stored more |
| `fork/run.py` kept-server key (`server_run`) | the cell `ds-d3-owner-check` (`--env GGML_CUDA_SCRATCH_CHECK=1`) printed no check line and the same draft counts as the cell before it | a cell's `--env` values become the bench process's own environment, and the key only took env entries that differ from that environment: a cell that differed from the one before only by `--env` reused the kept server, the values never reached a server | fixed: the cell passes its `--env` list as `PERF_CELL_ENV`, which is part of the key |

## 2026-10-28 DSpark on DeepSeek 4.1: prompt passes ran the draft, load crash after a stored VRAM need, a crash's backtrace doubling the pinned RAM

| where | symptom | cause | fix / state |
|---|---|---|---|
| `common/speculative.cpp` (DFlash / DSpark `process`, `begin`) | ds-i-sp (16k prompt): every prompt pass copied all its target features to the host and ran the draft over all rows; with the old answer-start injection ds-i-sp and ds-io-sp failed with HTTP 500 `inconsistent sequence positions: last position stored X=26, starting position Y=3287` | the draft (DSV4 backbone, sliding window of 128 in every stage) only ever reads the last 129 positions, but every prompt row was gathered and injected; the injection at the answer start then met stale cells of an earlier request in the draft's cache | fixed (branch fix_dspark_prompt_runs_no_draft, build fix_dspark_window_card_need): a prompt pass keeps only its last 129 rows in a ring, no draft call; `begin()` trims the draft's cache and injects them. Log proof per request: `begin: prompt of 17575 tokens: 4 passes (17575 rows) ran no draft (draft calls in that time: 0)`. ds-i-sp 2 of 2 (decode 19.95 / 18.90 t/s, prompt 254 / 273), ds-io-sp 2 of 2 (20.77 / 22.04, acceptance 79%), ds-o-sp passes |
| `src/models/deepseek4.cpp` layer inputs, `src/llama-graph.cpp` `set_outputs`, `src/llama-context.cpp` `output_reserve` | the target held the 3 extracted layer inputs of a 16384-row pass (3 x 320 MiB of VRAM) to the end of the graph and a 0.9 GiB pinned, cleared host buffer, for the 129 rows the draft reads | the layer-input mean was taken over the whole pass and copied out whole | fixed (same build): `llama_set_embeddings_layer_inp_last(n)`: DeepSeek takes the mean of the last n rows only (8 MiB), the copy goes to lazily paged memory, not the pinned output buffer |
| `src/llama-moecache-fork.cpp` `vram_key` (stored VRAM need outside the pool) | build fix_dspark_prompt_no_draft_work, ds-i-sp: server died at load, `allocating 796.79 MiB on device 0: cudaMalloc failed: out of memory`; sizing line `margin 99 MiB (stored need 35 MiB + safety)` | the key held the GPU model name only: PC1's two RTX 3090 shared it, CUDA1's 35 MiB overwrote CUDA0's 290 MiB, so cuda0x4 got a 99 MiB margin where the draft's setup re-reserve needs ~260 MiB | fixed (same build): the key carries the card's PCI address. Audit of every stored per-card value: link speed `link<k>` was keyed by link index (link 0 = the prompt card), fixed in source (built later); `pp_c0_us` / `pp_a1_us` one value per card set, read-only legacy: open |
| `ggml/src/ggml.c` `ggml_print_backtrace` | the crash above (`failed to allocate compute pp buffers`) made MemAvailable fall 19.7 -> 4.3 GiB in 8 s, the memory guard stopped the run; likely also the 2026-10-27 22:29 freeze (power cycle) | the crash handler forks to start gdb; Linux copies DMA-pinned pages at fork (no copy-on-write for pinned anonymous pages), so a process with 90-110 GiB pinned (cudaHostRegister) needs that much again. Measured: 4 GiB registered, `ggml_abort`: lowest MemAvailable 3.75 GiB below the start | fixed (same build): gdb (or lldb) starts through `posix_spawn` (vfork: nothing copied) behind a shell that waits for `PR_SET_PTRACER`. Same test: 0.01 GiB, gdb stack still printed |

## 2026-10-28 DSpark on DeepSeek 4.1: every repeat request's answer stopped at ~717 characters

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_graph_update_required` (CUDA graph reuse) with the expert pool's CUDA scratch region (`ggml_backend_cuda_scratch_fork`) | DeepSeek 4.1 + Lucebox DSpark draft, depth 1, tetris 4k, `-c 8192`: the first request fine (11.7k characters for 4096 tokens, 92% drafts accepted), every repeat request stopped printing at ~717 characters (~256 tokens), acceptance 3%; all 15 draft repeat runs since 2026-10-27 00:44, including the published b14064; depth 3/5, no draft and `cache_prompt` off were fine | a captured CUDA graph keeps the scratch addresses of its capture. The pool moves the scratch region whenever a graph's lend changes (the region starts where that graph's tensors end): in the repeat request CUDA1's lend sat at one class-7 block and grew to two when the cells crossed 512, so the region moved 5.16 -> 7.09 MB. A reused ggml graph (same uid) replays its CUDA graph with no check and the node properties hold no scratch address, so the replay wrote temporaries over live tensors. `--moe poison=3` + `GGML_SCHED_NAN_CHECK=1` found the first NaN in layer 14's MoE on CUDA1 at that step; `GGML_SCHED_NAN_CHECK=2` (no graphs) and `GGML_CUDA_DISABLE_GRAPHS=1` hid it (300 tokens: 717 characters and 116/182 drafts with graphs, 887 and 145/153 without) | forgotten-fixes, build fix_cuda_graph_scratch_moved: the region each CUDA graph was captured with is kept per graph and a replay with another region captures again (`GGML_CUDA_SCRATCH_DEBUG=1` logs it). 300 tokens: 887 characters, 145/153 drafts on the repeat, decode 22.1 / 25.1 t/s against 21.0 / 24.2 before. General form for the pool manager: a layout serial on every cached address (fork/docs/pool-manager-plan.md on pool_refactor). Repro lists fork/runs/dswarm1-4.csv. The earlier forgotten-fixes commit (draft checkpoint whole state) changed nothing here. |

## 2026-10-28 CUDA out-of-memory take-back on sliding-window models and DeepSeek 4.1 (#155, #168)

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-context.cpp` `llama_context::decode`, CUDA out-of-memory take-back (g_oom_retry), with `src/llama-kv-cache-dsv4.cpp` | DeepSeek 4.1: a prompt call that ran out of VRAM was not retried (seq_rm cannot take back a call that starts inside the committed range of the compressed caches) | the compressed caches keep a compressor state per sequence that seq_rm cannot roll back | #155 (bugfix-7): the compressor state is kept at the call's start (`comp_snapshot`) and put back with the take-back (`comp_take_back`); test switch `--moe fake-oom=N`. Built `fix_dsv4_oom_takeback`; superseded by the #168 build before its test ran |
| same, `src/llama-kv-cache.cpp` `find_slot` (stock: a ring cell masked for the pass's first position is reused) | read from the code, not seen in a run: a failure in pass 2 or later of one call on MiMo (39 of 48 layers sliding-window) or DeepSeek 4.1 (raw cache is iswa) leaves the retry without the window before the call's start: refused (the ring is empty for the sequence) or a window with holes (answer quietly off). Single-pass calls are not affected; multi-pass calls come from `-b` > `-ub`, the planner's pass cut at long context, and the lowered ceiling after a first out-of-memory | the ring holds the window plus one pass, so a later pass reuses the cells of the window the call's first pass reads; taking the whole call back cannot bring them back | #168 (bugfix-7, user picked option A over a stock `find_slot` change and a window copy): a later pass of a one-sequence call whose earlier passes gave no output takes back only itself and the call goes on from it (`g_oom_resume`); DSV4 keeps its compressor state at the start of every later pass. Also found while reading: the state write-back ran before the other cards finished the failed graph, and the call-start snapshot did not wait for the call before; both now wait. Build `fix_oom_takeback_failed_pass`; test list fork/runs/takeback.csv (DeepSeek 16k with `-b 4096` and `-b 8192 -ub 4096`, MiMo 12k with `-b 8192 -ub 4096`, each with and without `--moe fake-oom`): the answers must be equal. Pending |

## 2026-10-28 decode thread tuner: a second search in an SSD stall left 1 thread for the next request

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-context.cpp` `llama_context::thread_tune_feed` (settled state, drift watch); batch tuner `thread_tune_feed_batch` | DeepSeek 4.1 16k1k, dspark_fix3, clean state: request 1 settled on 11 decode threads, searched again at token 954 and walked 16 -> 8 -> 4 -> 2 -> 1 in 70 tokens; request 2 decoded on 1 thread, 9.67 against 15.15 t/s (run log `dspark_fix3-um7-ds-nodraft.server.log`, 2026-10-28 16:38) | the drift watch searched again from all threads when the mean token time of two 64-token windows was 10% above the best; on a model read partly from the SSD the token time drifts with the page cache, and in a stall a lower count wins every pair | user rule: one cheap search per request, the count stays to the end of that request and is never stored or carried over (a first fix kept it for the server's life: rejected, a bad kept value is the harm). The batch tuner (only with `--moe batch-tune=1`) does the same; `--moe thread-drift=1` restores the old re-search. bugfix-7, build `fix_takeback_pass_threads_per_req`; check: one `threads_decode` search at the start of each request in the DeepSeek clean-state cell, none later in an answer. Pending |

## 2026-10-30 CPU and GPU gave different float bits (branch cpu-gpu-exact), found with the layer dumps of llama-eval-callback (ROUTER_DUMP_ALL), Qwen3.6, batch 1

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` BF16 trait (`ggml_vec_dot_bf16`) | alpha/beta matmul 33% bit-identical to the GPU, 3 digits lost | the CPU rounded the f32 activation to bf16 before the dot; CUDA keeps it f32 | `ggml_vec_dot_bf16_f32` (f32 activation); a real accuracy gain on the CPU, not only a match |
| `ggml/src/ggml-cuda/norm.cu` `rms_norm_f32`, `ggml-cpu/ops.cpp` rms_norm | attn_norm 69% bit-identical | squares added in a different order, fast-math division/rsqrt on the GPU, f64 sum on the CPU | both add in the CUDA order (explicit `__fmaf_rn`, `__fdiv_rn`, `__fsqrt_rn`) |
| `ggml/src/ggml-cuda/quantize.cu` `quantize_q8_1` (mmvq activation) | one token in 26: ssm_out differs everywhere (1e-6 relative) | `roundf(x / d)` with the fast-math division: an exact tie x/d = -27.5 came out -27.499998 and rounded to -27, the CPU gives -28 | `__fdiv_rn(x, d)`; CPU quantizer already exact |
| `ggml/src/ggml-cuda/{unary,softmax,gated_delta_net}.cu*`, `ggml-det-math.h` | silu/sigmoid/softplus 40-75% identical, softmax probs 28% | `-use_fast_math` makes `expf`/`log1pf`/`/` approximations nobody can reproduce on the CPU | one shared deterministic `ggml_det_expf/logf/log1pf/div` (fmaf only) used by the CUDA kernels and by the CPU ops |
| `ggml/src/ggml-cuda/binbcast.cu` `op_div` | MoE weights normalization (div by the row sum) 55% identical | fast-math division | `__fdiv_rn` |
| CPU copies of the CUDA reduction shapes (`GGML_CPU_GPU_SHAPE=1`): Q8_0 and Q2_0 dots, f32/bf16 matvec, gated delta net, softmax, sum_rows, activation quantizer | matmul outputs 34% identical | float addition order (lanes, warps, xor trees) differs | copied from the sm_86 SASS and sources; Q2_0 expert kernel rounds product and add separately (no FMA), found by testing variants against the GPU output |
| result | layers 0 and 1: every tensor 100% bit-identical (attn_norm .. l_out); layer 2 breaks at token 0 (open) | | check: `cmp9.py r12_gpu.bin r12_cpu.bin` in the session tmp, builds exact_det4 |
## 2026-10-30 CPU expert misses not bit-identical to the GPU: greedy answers forked with the cache size (branch cpu-gpu-exact)

Goal: a missed expert computed on the CPU must give the GPU's bits, so a greedy run never depends on what the cache holds (dynhot replay at 100%). Scope: the expert chain only (activation quantization, up/gate dot, swiglu, down dot); attention, delta net, norms and router stay on the GPU in production. The switch: `GGML_CPU_GPU_SHAPE=1`, `--moe gpu-shape=1`, or `GPU_SHAPE=1` in the cache's control file while the server runs.

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cpu/quants.c` GPU-shaped dots, `ggml/src/ggml-cuda/quantize.cu`, `ggml/src/ggml-cuda/vecdotq.cuh` | hybrid dump (experts on the CPU against all on the GPU, one token per pass): router top-8 differed in 66-84 of 1040 token-layers on Qwen3.6; K-quant dots 27-47% bit-identical | the CPU adds a row's block terms in another order than the CUDA kernel (4 warps over K, lanes by an xor tree); the CUDA Q8_1 quantizer divided with fast math and the CPU multiplied by 127/amax; Q2_K/Q4_K combine `fma(d, sumf_d, -(dmin*sumf_m))` on the GPU | commits "CPU/GPU bit-exact decode", "CUDA/CPU: IEEE amax/127 in the Q8_1 quantizer, deterministic RoPE ...", "CPU GPU-shaped dots for Q2_K, Q3_K, Q4_K, ...": CPU copies of the sm_86 decode kernel's order for Q8_0, Q2_0, Q2_K, Q3_K, Q4_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 (harness against the real kernel: 100% for K 256-7168); IEEE amax/127 in the CUDA quantizer; deterministic expf/logf/powf/sincos/division (`ggml-det-math.h`); RoPE, top-k MoE softmax, silu, softplus, rms norm, delta net. Check: hybrid dump 100% in all 40 Qwen3.6 layers, router mismatches 0 of 1040 |
| `src/llama-moecache-fork.cpp` `llama_moe_cache_max_batch_fork`, `ggml/src/ggml-cuda/mmvq.cu` kernel choice by batch | Qwen3.6, llama-server, switch on, cache mode at `--moe sim-vram=3000` against `4000`: token 0's logprob differed (-0.0842 against -0.0806), 391 of 400 logprobs differed, the text forked at token 183, although every dump said 100% | the GPU computes a cached expert with `mul_mat_vec_q` (4 warps over K) for one token, `mul_mat_vec_q_moe` (one warp per token) for 2..8 tokens (5..7 for some types) and MMQ above; the CPU copies the one-token kernel. max-batch (auto, 35 here) sends passes up to that size through the cache, and the server cut the 36-token chat prompt into 32 + 4 tokens (checkpoint before the last tokens), so those passes mixed GPU hits and CPU misses. The dumps could not see it: `llama-eval-callback` never preheats the cache, so each of its prompt passes had 0 hits (CPU against CPU) | v1: with the switch on max-batch is 1, only one-token passes go through the cache; a larger pass takes the prompt path (all its experts on the GPU, the same bits at any cache size). Check (server, chat request, 400 greedy tokens, logprobs at two cache sizes): six models, 400 greedy tokens, logprobs at a larger and a smaller cache (Qwen3.6 misses 3513 / 12777; qni3 hits 98.8% / 94.2%; qni4 96.8% / 91.4%; GLM 3.0 99.9% / 69.6%; GLM 3.5 99.7% / 40.1%; MiMo 95.5% / 63.3%): differing logprobs 0/400, 0/378, 0/400, 0/400, 0/400, 0/400. Open: a 4-token cache pass with misses still differs from one without (not the Q2_0 dot order, which is the same in both kernels for K <= 4096); v2 would keep short passes (speculative verify, the 4-token prompt tail, several slots) on the cache path |
| `ggml/src/ggml-cpu/tiled/tiled.cpp` `ggml_tiled_supported` | (found reading) a K-quant expert with 8 or more rows in one CPU pass took the tiled kernel, not the GPU-shaped dot | the tiled matmul takes over by row count | the switch keeps a type with a GPU-shaped dot off the tiled path |

## 2026-10-30 llama-server does not exit after a second identical request in cache mode (open)

Repro: GLM 3.5 with `--moe sim-vram=6000 wait-swaps=0`, two identical chat requests, then SIGTERM. The log ends with "cleaning up before exit" and the process stays; SIGKILL needed. Thread states read from /proc (ptrace denied). Cause not found; seen while running the exact-mode checks above, not tied to them.

## 2026-11-01 Every answer above the sparse gate was `!!!!` on GLM and Qwen Next, and a day of rows reported it as a gain

Repro: `python3 fork/run.py jobs -c up8kv -b up8_gather_rows -r 0 -nm -onl kvb-glm60,kvb-qn60` (build before the fix): answer `!!!!...`, decode and hit rate look excellent. Below the gate (GLM 10196 cells, Qwen Next 21784) nothing is wrong, so short rows and PPL at 16k on Qwen Next passed.

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` `flash_attn_ext_f16`, the mask stride | NaN logits, argmax token 0, one repeated token; GLM decode 27-28 t/s and Qwen Next 44 t/s with hit 99.9%, all void | commit "Attention mask only for the pass's own cells" derives the pass's first cell as `ne11 - stride`. In a gather `ne11` is the list length, not the KV length, so the mask was read far off and every cell was masked | `mask && !use_sparse` in the stride: a gather's mask covers all of K, there is no first cell to derive |
| `fork/run.py` `is_loop`, `--keep-loops`, how results were read | the rows carried `ok=1`, and speeds were reported before answers | `--keep-loops` skipped the loop test, which is what caught `!!!!` | `is_garbage` fails such a row always. Every answer now goes to a judge model on W10 (`judge_row`, column `judge`), and a nonsense row has no accepted speed |

Verified: build `up8_gather_mask_fix`, GLM 62k and Qwen Next 68k answers are real text (rows `kvb-glm60`, `kvb-qn60`). Long prompt rows for every engine build: `fork/runs/sane.csv`.

## 2026-11-02 Second 128k request `!!!!` on Qwen3.6: a GLU fused into MMQ left the rows of skipped experts unwritten

Repro: `python3 fork/run.py jobs -c up8kv -b up8_kv_auto10 -r 0 -nm -onl kvb-m90-128k` (build before the fix, 2 requests, about 3 min). The bug is on a prompt mix's first pass: the planner first tries the mix on request 2's last call, so request 1 is always fine.

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_try_fuse` (GLU into MMQ, upstream #30489, merged by hand) against `ggml_cuda_mul_mat_id` | Qwen3.6, `--hog-gpu 4.5`, 137014-token prompt: request 2 answers `!!!!`, fake 77 t/s. Without NaN bits in the old bytes, a coherent answer that parts from request 1 at character 5 (row 21866, `ok=1`) | the fused call went straight to `ggml_cuda_mul_mat_q` and skipped what `ggml_cuda_mul_mat_id` does before its kernel: zero the output (`op_params[1]`: the kernels leave rows of skipped experts unwritten), the expert ids and table (src[3], src[4]), the heat count. Only a pass of several tokens with skipped experts shows it, which is the mix's home card: those rows kept what the lent memory held and went into the experts' sum | `ggml_cuda_mul_mat_id_begin` shared by both paths; the fused call passes ids and table and its overlap check covers them. Build `fix_glu_fuse_zero`, rows `kvb-m384-128k`, `kvb-m90-128k`, one mix call each: request 2 coherent, parts from request 1 at character 342 as with `pp-mix=0`, decode 71-72 t/s |

| false lead | why it looked right | why it was wrong |
|---|---|---|
| #31391 batched reads (reverted, then relanded) | `GGML_CUDA_NO_PINNED=1` cleared the NaN and switches that path off | the NaN stayed with the PR reverted; that switch also removed the mix call from the row |
| `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_DISABLE_FUSION=1` | both clean | the first row's answer still parted at character 5; the second row ran no mix call at all |
| pool size, CUDA0's KV zone moving | every NaN row so far had the cold 384 MiB margin and a zone move | margin 90 with no zone move was NaN too |

What found it: the NaN check named the first tensor (experts' sum on the mix's first layer), `pp-mix=0` proved the path, comparing the answer text of request 1 and 2 showed damage without a NaN, and the poison ledger showed nobody had written the bad row since the graph's lend. A switch that clears a bug proves nothing until the row shows the same path ran (mix calls, pool, margin).

Rows with a mix call on builds `up8_kv_auto10` and `up8_no_gather` are void. Not run yet: `test-backend-ops` GLU fusion test on a GPU, a true PPL row through the fused path.

## 2026-11-02 Qwen Next: an expert uploaded during decode left stale bytes behind its slice

Repro: `python3 fork/run.py jobs -c paths -b up8_ledger2_verify -r 0 -nm -onl dbg-qn-past` (build before the fix, about 6 min). Found by the first run of the forced path list with `--moe verify=1`, not by any answer.

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-moecache-fork.cpp`, the JIT upload in decode (and the move of experts between cards) | `pool-verify: BAD moe_cache_down.30 ...: 406 of 1741208 bytes differ, the first at +1740800 (only past the slice)` from request 2 on. Under poison, NaN in the last value of an output row (element 325119 of a 2560-wide tensor) on the mix's pass. Without poison the last value of each row reads stale bytes: wrong, or NaN when they were a graph's or the KV's | a matmul reads past the last row of a slice when rows are not a multiple of 512 values (Qwen Next and Qwen3.8 down rows of 640). The upload workers and the prompt chain wrote those bytes, the JIT upload wrote only the slice. Ledger: the last writer of all 130 flagged blocks was `jit-up` | one rule, `expert_past`, for every writer and for the byte check; the prompt chain now also writes zeros behind the last expert |

Only models with such rows are touched; GLM, MiMo and Qwen3.6 showed 0 bad bytes on every path.

The first fix had its own bug: it clamped those bytes to the class's view, which ends with the last block's slice, so that block's bytes were no longer written and the byte check no longer read them. `fix_jit_past_bytes` gave NaN under poison from request 1 with a clean byte check. Found with `GGML_SCHED_NAN_CHECK=2 GGML_SCHED_NAN_CHECK_ONLY=ffn_moe_cache`: all bad values in column 2559 of `ffn_moe_cache_down-30`, the table reaching the class's last block. Those bytes now go through `expert_past_view`, a tensor of their own. Verified on build `fix_jit_past_last_block`: `pa-qn_iq4xs-mix` under poison, NaN check and byte check, both requests: no NaN, 0 bad bytes in 5 byte checks.

## 2026-11-02 Found by the forced path list and the long prompt list on the same night

| where | symptom | cause | fix / state |
|---|---|---|---|
| `src/llama-poolalloc-fork.h` `poolx_pick_slot` | Qwen Next, `pp-path=mix`, RUN_TRUE_PPL row: `general protection fault` in `llama_moe_cache_step_fork`, on the loop over `adopt_victims`, right after a 95-token pass | the picker went on to an empty block with the owner of an earlier candidate still in its answer. An adoption then took expert -1 of that owner: a write before its table, and a bad index at the next step | the owner is reset on both early returns; `tests/test-pass-fork.cpp` has the two cases, which fail on the old header. Build `fix_pick_owner_kv_flip`: the row runs, PPL 1.05099 |
| `llama_moe_cache_kv_retype_fork` with the q8_0 policy | Qwen3.6 128k with the dense split on (`sl-q36`): HTTP 400 `kv-fit: CUDA0: no lend in front of the KV to stage the conversion` | the conversion asked for staging blocks with no rows to convert, and the dense split's copies left none in front of the zone. The policy's optional pick failed the request | no rows, no staging; a pick of the policy that cannot run stays on f16 with a logged decision. `sl-q36`: answer coherent, decode 70.9 t/s, prompt 2221 t/s |
| `moe_fill_cb` | the adoption path wrote the slice only, like the JIT upload above | same as the JIT entry | `expert_past` there too |

Verified on `fix_pick_owner_kv_flip`, `kvb-m90-128k`, one mix call on request 2: the two answers are identical over all 802 characters, decode 71.6 and 71.4 t/s.


## 2026-11-02 MiMo off topic on the decode chain: the fused GLU's zeroing wiped its own inputs

| where | symptom | cause | fix / state |
|---|---|---|---|
| `ggml_cuda_try_fuse`, GLU into MMQ, with `ggml_cuda_mul_mat_id_begin` | MiMo, `pp-path=dec`, 121 tokens: the answer is off topic from its first word and loops, at speeds that look like gains. Clean with `upload-now=0` | The fix of the `!!!!` entry above zeroed the output before the quantize. The GLU is gate's and up's last reader, so the allocator lays the output (`ffn_moe_cache_down`) exactly on their bytes (all 46 expert layers, `ledger=2`). The zeroing wiped gate and up, every GPU expert of the pass came out as zero and only the CPU's picks reached the answer | the output is zeroed inside `ggml_cuda_mul_mat_q`, behind the quantize |

It looked like an upload race because of how much was lost: with `upload-now=0` the CPU kept about 40% of the picks and the answer stayed on topic, with mid-pass uploads less stayed on the CPU, and `jit-all=1` gave the worst text. What did not find it: poison, `verify`, `--races`, `--map`, and the new `--dma` windows (32,720 copies, nothing under a running copy), because every byte was right and nothing was NaN. `GGML_SCHED_NAN_CHECK=1` kept the bug and `=2` cleared it (no fusion there), `GGML_CUDA_DISABLE_FUSION=1` cleared it, `fuse-off=GLU` named it and `ledger=2` showed the overlap.

Reached by passes that run MMQ with ids that may be -1: the decode chain above the vector kernels' batch, the prompt mix, the 2-card prefill. No release has it. Rows of those paths since that fix are suspect even when on topic: the answers lost GPU experts.

Verified on `glu_zero_late`, 100-token prompt, two requests each: MiMo, GLM and DeepSeek on the decode chain with uploads, MiMo with uploads off and on the mix, all answers on task.

## 2026-11-08 the 64k gate night: three long-context faults

| where | symptom | cause | fix / state |
|---|---|---|---|
| `tools/server/server-context.cpp`, the equal call cut | Qwen3.6-35B (hybrid, gated delta net) at 72k tokens: the answer degenerates into one repeated identifier in every run; rel_b14244 coherent; 12k fine | calls of 24161 tokens end 97 cells past a 256-cell pad; passes of 2014 and of exactly 2048 both fail, calls of 32768 pass: the boundary, not the pass size. The KV and its mask are laid out in 256-cell pads (`GGML_PAD(kv_end, 256)`); a call boundary off a pad corrupts something the recurrent layers read (the exact op is open) | the cut rounds the call size up to a multiple of 256 (022064c06d): coherent again (q35-calls256-64k). `--moe equal-calls=0` switches the cut off. The latent pad bug stays open |
| `ggml/src/ggml-cuda/concat.cu`, the non-contiguous kernel | DeepSeek 64k: `CONCAT failed: CUDA error: invalid argument` on CUDA1 at pass 2/2 of the second call, every run | the kv-tail's K is `concat(cache view 16640 cells, computed rows 49152 cells)` along the cells; grid y = 65792 > 65535 | grid y and z capped at 65535, the blocks stride (b37b600705). Any model's kv-tail concat past 65535 cells hit it; rel_b14244 has the same kernel and never ran DS 64k (the cell froze PC1 at 101 GiB pinned) |
| `llama_context::process_ubatch` / `decode` | DeepSeek 64k after the concat fix: a 16384-token pass at 57k -> 73k cells needs a 25.4 GiB graph (the indexer's cells x tokens term) against a 20.2 GiB lend: `ggml_gallocr_reserve: graph needs ... the external buffer holds ...`, HTTP 500 | the failure returned ALLOC_FAILED without the pool's refusal flag, so decode's smaller-pass retry never ran | the failure records a refusal and decode accepts ALLOC_FAILED there (c7c30dc30a); first try still died without a retry line, diagnostics added (f177f4a9aa), build retry-diag |

## 2026-11-10 Cached experts zeroed before a server's 2nd request (qni4, DeepSeek): CUDA init_tensor cleared the row padding of graph tensors placed in the pool before the lend , FIXED

| | |
|---|---|
| found | 15:35, by the first `verify -r 2` (two requests per server; every verify before ran one request and could not see it) |
| symptom | qni4: 2-4 bad expert slices per server, DeepSeek 13-18, on the x16 card (07:00.0) only, all before request 2's first graph. GLM 3.0 and MiMo clean, release oneexe b15065 clean. The answers looked fine |
| repro (CLI) | `python3 fork/run.py run corr -b verify_dumps_bad_bytes -r 2` (build a83449c926 dumps the bad bytes: `pool-verify: BAD ... card 0000...`) |
| where | `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_backend_cuda_buffer_init_tensor` (~l. 1032: cudaMemsetAsync of alloc_size - nbytes behind a quantized tensor in a non-COMPUTE buffer); `ggml/src/ggml-alloc.c` `ggml_gallocr_set_external_buffer`, `ggml_vbuffer_tensor_alloc`; `src/llama-context.cpp` ~l. 3559 places the graph, ~l. 3562 lends after it |
| cause | the bad bytes are zeros, in windows of exactly one padded row: qni4's down tensors (ne0 640) 408 B Q8_0 and 216 B IQ4_NL, DeepSeek 66 B IQ2_XXS and 74 B IQ2_XS. The compute alias placed each graph through the pool's own buffer (usage WEIGHTS), so CUDA's init_tensor zeroed the padding behind every quantized graph tensor (the streamed expert copies) at placement, before `llama_moe_cache_alias_fork` lent those blocks. The blocks still held cached experts, and keep-hot then copied them, zeros included, into cold blocks. Same root as entry 7 (graph memory marked as weights), write side this time. Discriminators: CUDA graphs off and the row reader off were still bad; ledger=2 was clean (its syncs change the timing) |
| solved | f6dbde5665: gallocr places graphs through a twin of the pool buffer object with usage COMPUTE (same memory, same interface). Nothing is written at placement; MMQ/MMVQ clear the padding when they run, inside the lend. Staging slots keep the pool buffer (their shifted data pointers get no run-time clear) |
| verified | list `vffix` on fix_graph_pad_zero_before_lend: qni4 x2, DS x2, GLM, MiMo, -r 2: 0 bad slices, answers sane (before: qni4 bad in 3 of 3 servers, DS 18 slices) |
| guard | list `guard` (e24af9c9d2, `--moe guard=1`, -r 2): with the old placement put back (`GGML_GALLOC_TWIN=0`) the pool guard logged 64 violations, each `init_tensor padding clear` of a `ffn_down_exps` graph tensor (216 B IQ4_NL, 408 B Q8_0) into expert blocks outside the lend, and verify found 2 bad slices; the fixed placement logged 0 on qni4, GLM, MiMo, q35. DS and GLM 3.5 logged the guard's own false positive (an upload's tensor_set into the range its own worker had entered), fixed in 098ae6ebaa |
| follow-up | this class keeps coming back: a read or write past an allocation's end at the access granularity (row padding, tiles, pad rows): entries 3 and 7, both 2026-11-02 entries. Pool guard planned in `fork/docs/pool-manager-plan.md`: one access check for every write including ggml's own, DMA locks with start and stop, a debug probe of every kernel's ranges |

## 2026-11-10 evening

| where | symptom | cause | fix / state | class |
|---|---|---|---|---|
| pool guard (`pool_guard_write_hooks`, writers()) | DS and GLM 3.5 logged 64 violations each; verify read back 0 bad | a range entry did not carry the thread that entered it, so an expert upload's tensor_set into its own worker's block (0.004-2.1 ms earlier, one thread, one stream) was reported as a race | 098ae6ebaa FIXED | harness |
| `run.py` memwatch, queue-daemon check, run/attach/status checks | not stated (messages give no symptom): a zombie server or run counted as alive; the daemon check hung | a zombie pid was not treated as gone; systemctl hangs while PID 1 is stuck | 742c545154, 9cb04207b5, 3c75258c34 FIXED (5 s timeout; zombie = gone) | harness |
| `run.py` speed parser | "0.62 t/s pp 10.1 t/s" read as a prompt speed of 0.62 | number taken before its label | 61284d9c89 FIXED (each number after its label) | harness |
| `llama_context` micro-batch fit (sched_reserve step-down) | Kimi K3.1: 7936 fit, 8064/8000/7968 did not, 7936 again did not; op-offload retry failed too; server started with the fork off: prompt 0.92 t/s, decode 0.5 t/s through the mapping | the largest size that fit failed on its second try and the code turned the fork off instead of stepping down | 71fd617edf FIXED (step down) | stale-number |

## 2026-11-11

| where | symptom | cause | fix / state | class |
|---|---|---|---|---|
| `llama_context` sched_reserve once-per-context guards | Kimi K3.1: one session ended at micro-batch 6272 although 8192 fit; another came up without op offload, prompt 7x slower through the mapping | guards (free-VRAM cut, pools' fit, asked size) keyed by context pointer; a freed context's address is reused by the next one, so from the 3rd-4th try the cut was skipped | 54f1f809b4 FIXED (constructor clears a guard naming its own address) | stale-number |
| `common.cpp` chooser, moe-cache setup, `--moe pred-top` | Kimi K3.1 pred-top: ubatch 1024 and a 4.6 GiB reserve on CUDA1 failed, uncaught exception at load; then 10.7 GiB cache beside a 7.2 GiB compute buffer on CUDA1 crashed at load; later the same exception again | pred-top switched the cache to per-layer tensors (no block pool, nothing lends graphs back) while the alias still gave compute buffers back before the cache was sized; the cache took that memory | ebb10b7db9, a6e2ab5c11, aa6fac3570 FIXED (alias needs block pool and no predictor); block pool with predictor reverted: OLMoE pred-top + rampool gave CUDA illegal memory access at first graph | flag-coupling |
| moe-cache router prediction setup, latent MoE | Kimi K3.1 with `--moe pred-top` printed nothing and read nothing ahead (OLMoE set up 15 layers); "router of 48 has another shape" | experts run in a 3584-wide latent space, routers score the 7168-wide hidden state; the shape check compared the wrong input and no layer got a predictor; no log line said why | de4c29e39a, 47e8984685 FIXED (reason logged; predictor takes the router's width and input) | silent-fallback |
| `fit_ub` / pass-size probes (CPU compute buffer) | Kimi K3.1: sessions with micro-batch 6560-6944 held 4.4 GiB CPU compute buffer (5280: 0.7 GiB), no expert uploads, experts read through the mapping at 0.8 GB/s with swapping: 557 s instead of 82 s | not stated beyond the nodes put on the CPU per probe | aeddb66055 diagnostic only (one log line per probe); cause OPEN in message | other: cpu-offload fit |
| sched copy, run tail upload | upload read and wrote past the run's last expert, past the tensor's end when the run ended at the last expert (expert cache on) | tail upload started at the first uncopied expert but kept the whole run's length | 35bd40433a FIXED (last + 1 - next, as stock master) | past-end |
| moe-cache `fill=layers` ranking | Kimi K3.1: RAM pool held 3-4% of layers 1-14; the fill went to the other card's layers | `mc->layers` is per card group (46 layers per card), its index is not the model layer | 09173c1c4f FIXED (rank by pub.il) | card-asymmetry |
| `kv_tail_pad_zero` hook (alias hook gating) | OLMoE nostage: 6 of 8 rows NaN at the first graph (attention read poison NaN pad rows) | since the DSpark fix (2026-10-27) the hook ran only while moe_alias_bufs was non-empty; with the lend dropped (cold start, small pool, floor) it never ran | 903908c2e3, 4f9de4893f, e0f7cc5580 FIXED (zone code zeroes every row past the cells where the zone changes) | flag-coupling |
| KV tier pick, `kv_lend_potential` | OLMoE alias=0 at 1280 MiB: 1317 tokens passed against room for 1506 cells, growth asked 1536, cache refused the prompt (3 of 6 rows); Qwen3.6 iq4_xs at simulated 8 GiB: fit picked 1568 tokens, zone held 1536 rows, find_slot refused, prompt ran on the decode chain | tier capacity and lend reserved unpadded rows; attention reads them rounded up to the 256 pad | 4f9de4893f, 1f7d2a4de1 FIXED (whole pads) | past-end |
| staging=0 forced stream (`pp-path=stream`) | KV write asserted (nostage ns1-stream, OLMoE): the zone was cut under cells find_slot had placed | forced stream still ran the staging stream; its 137 MiB staging lend (128-token pass) took the KV's room in front of the experts' floor | 9e5863be7b FIXED (forced stream = chain with every expert marked used) | pool-owner |
| staging=0 path selection | a refused prompt chain fell to the stream; learned planner kept stream arms; table check hit preheat victims reset in the same step | the mode did not apply to every path that stages | cfd4783797, 399298b22f FIXED (refused chain becomes decode chain; stream arms dropped) | flag-coupling |
| pool-floor under staging=0 | Qwen3.8-Flash-Next, 2 cards at 8 GiB: two one-layer classes of 99 blocks against 512 experts, every call on the decode chain: prompt 307 t/s against 643 staged | a class short of one layer sent its size to the decode chain | d4a49eef54 FIXED (staging=0 implies pool-floor); superseded next day by b62bcbec14 | flag-coupling |
| prompt chain decision (`pp_settle`) | one-layer class of exactly its experts with one refill on the way refused the chain for the whole call (dense split, staging=0: every call on the decode chain) | the decision ran before pp_settle; uploads in flight not counted as landed | 8eb3b20e02 FIXED | stale-number |
| pp-ahead v1 (prompt prefetch flag) | pah-8g: decode 27.9 t/s against 32.7 with pp-ahead=0, hit rate equal (77.3 / 77.1%), decode uploads 284/s against 110/s | prefetched experts were flagged as predictor prefetches, so their decode hits counted as used predictions | 4b4162136e FIXED; later be5822c384 sets pp-ahead off by default (decode -26% at 8 GiB, one hung call: OPEN) | flag-coupling |
| cstream layer blocks (OLMoE, 64 slots) | abort: 0 free blocks | pp_settle dropped the queued preload but its blocks stayed the layer's; not counted free | c389b79ed7 FIXED | pool-owner |
| cstream split hook | hook never ran | split name sat on the get_rows reshape; the scheduler skips view ops | 640a5b73e4 FIXED (name on the get_rows node) | other: view op |
| `run.py` live view | uploads strip drew an x16-only row on the x4 card | `--env CUDA_VISIBLE_DEVICES` not mapped to the card | 844d9a4745 FIXED | harness |
| RAM available accounting (auto pin, pass cap, mask budget, pool, memwatch), predictor copies | PC1 freeze; PC1 shows 9.4 GiB CmaFree with CmaTotal 0 | CMA's free pages counted as available RAM; learned predictor copies took all VRAM | 641606fa4a, e241f928ae, c75b449642 FIXED (CmaFree left out; 1 GiB VRAM kept; memwatch kills on page-cache thrash) | stale-number |

## 2026-11-12

| where | symptom | cause | fix / state | class |
|---|---|---|---|---|
| planner bulk upload estimates, live view link max | not stated (message gives no numbers) | estimates used the per-expert upload average, not the probed link maximum | 0e3a0769c4, e3f7b5b173 FIXED | stale-number |
| per-layer cache (router prediction), predictor copies | not stated | cache did not leave the KV arena its context and margin | e21c8c1dd6 FIXED | pool-owner |
| cstream `cs_layer` done events | two cards: `cudaEventRecord: invalid resource handle` | recorded the previous cstream layer's done event on the current compute stream; on two cards that layer's event lives on the other device | 30b3590b47 FIXED (done chain per card) | card-asymmetry |
| nan-check | Qwen3.8 verify rows failed on GET_ROWS of inp_mixed_tokens in a token-only decode graph, staged and cstream | checked nodes the graph does not compute (unpicked ggml_build_forward_select branch holds stale bytes) | 01e79c3b4d FIXED | harness |
| cstream block choice | 14744 copies in a pass where the staged stream makes 3 per layer | ranked choice scattered a layer over the pool and took blocks of layers ahead | a939c97e7c FIXED (ring from a write cursor) | other: ring layout |
| staging slots lend | Qwen3.8 12k: KV fit refused a 5056-token pass at 13720 cells (3776 + 1280 instead of one pass) | lend gave every prompt graph staging slots whatever its path; cached experts evicted for unused slots, slots counted in measured need | 05e0c02a2e FIXED | pool-owner |
| staging=0 pool layout and chain rule | Qwen3.8 12k prompt 158 t/s against 867 staged; lend 1363 MiB against 3356 (passes 1216 against 5056) | implied pool floor and chain candidate while uploads run | b62bcbec14 FIXED | flag-coupling |
| staging=0 stream gate | Qwen3.8 at 8 GiB: 12k prompt ran as decode chain, 363 t/s against 867 staged | stream required every layer to fit the ring, not some | 9897ba2611 FIXED | flag-coupling |
| size-class grids, experts' floor, ring (class grid, owner -4) | Qwen3.8 at 8 GiB: classes of 92 and 279 blocks against 512 experts kept out of the ring; 12k passes cut from 5632 to 2016 tokens, KV refused 1504-token graphs; ring abort naming layers; layer 4 aborted | floor of two largest layers; main-class block across the floor edge marked as KV; last layer's output rows kept experts in grid class | c0e03f1570, 6d522a6758, 28e38edf60, 35bba7d6f1, 23e47fdb60 (floor part) FIXED | pool-owner |
| `moe-cache` slow-layer rule, ring pins | PC1 both x16: 30 sticky blocks of a 512-block grid kept layer 30 out of the ring; second 12k request aborted (Qwen3.8) | every layer off the prompt card counted as slow (old x4 slot) so hot experts were sticky | 1c997dca2a FIXED (slow = link measured below 0.6 of the fastest; second scan ignores pins) | card-asymmetry |
| `pp_cap_to_pool`, `kvz_pass_cap`, decode card order | Qwen3.8 staging=0: a 5056-token call ran as one pass at 8664 cells, KV did not fit, find_slot refused; two x16 cards (26.4 against 26.3 GB/s) swapped order run to run (pool 5810 or 5510 MiB, ceiling 5632 or 5056) | cap used the call's own tokens as zone rows, not the cells at call end; no tolerance in link ordering | e799018208 FIXED (links within 10% equal) | stale-number |
| pass cap line | cap's line said 5408 tokens fit at 13720 cells, measured graph bigger, find_slot refused the KV | least-squares line not shifted through measured size; dense split cap double-counted rows | 23e47fdb60, cf56ec11ae FIXED | stale-number |
| cstream refused chain, ring cursor | Qwen3.8 12k, 2 x 3090: layer 2 refused the chain (99 blocks, pass wanted 512), rest ran as decode chain: 13.1 s against 5.1 s streamed; card 1's link started 0.5 s before its turn | refused graph not re-decided; one ring cursor for both cards in graph order | 8710ee597d FIXED | card-asymmetry |
| cstream expert split (S3') | `--moe staging=0` with expert split ran as a plain layer split: qni4 12k run 2: 691 t/s against 1012 staged | cache path built no parts, one card computed while the other waited | a09be61031, 61d0063ff0, 153dd373dc FIXED (shadow layers; every op of a part on its card) | card-asymmetry |
| experts' floor in staged mode | GLM 3.0 lendable 18.1 GiB not given back, 10144-token passes lost (as b15065) | staging=0 floor applied to staged passes | e81c1263f9 FIXED | flag-coupling |
| `run.py relgate` | not stated | pool ledger on in fix_ builds, never paid by oneexe | e2093076f2 FIXED (off on both sides) | harness |
| `poolx_give_back` | two classes could own the same bytes | a grid block given back by a lend or the KV zone went on the free list whatever its place | e2be8585e5 FIXED (only inside [home_lo, home_hi)) | pool-owner |
| class layout, layer ring reserve | 12k passes GLM 3.0 4608 vs 10144, MiMo 8288 vs 16384, DeepSeek 4576 vs 16384; assert on Qwen3.8 at 8 GiB (3.32 and 3.08 MiB classes) | floor kept one layer per class, the first class that held a layer ended the graph's reach; single-layer classes at the tail had no bytes past home | 63e6e3a266, 3ca68f615d FIXED (class-blind tail reserve; most-layers class last) | pool-owner |
| `run.py` stock pairing, ring rows | MiMo edit and short rows showed 2.71x and 2.95x decode against stock b11866 (13.99 t/s), where same-machine stock runs about 30 t/s | machine filter matched host name only, not the changed PC1 hardware; also ring OLMoE row ran without the cache | 9ee6d838b6, 06fee207dc FIXED | harness |
| cstream chunks | Qwen3.8 expert split, 4 chunks: layer 2 ran whole, ring held 174 blocks for 512 experts, abort | a layer a card could not hold its range of left its home card | fd8bd78fb7 FIXED | card-asymmetry |
| staging=0 start-up micro-batch fit and pp reserve | graph measured 3426 MiB for 4704 tokens where the pass lends 1.9 GiB and capped every pass; ring at rest held 131 blocks for 512 experts; 32-token step-downs, 12 host pins at load; start-up estimates 0.61 MiB per token against 0.42 lent | fit measured a plain call's graph (host experts as inputs), not the stream into blocks | f669c6179d, c6167326f0, 6dc8d1be95, e3d6fde8a2 FIXED | stale-number |
| dense split copies judged by room | placed copies came off every pass's room: 54 passes of 64 tokens | judged against the ring at rest, not a whole-layer ring | 5193440e55 FIXED | stale-number |
| graph size line (`ppx_lend_line`), start-up probes (kpool) | Qwen3.8 `--moe staging=0`, 2nd request: find_slot refused a 7064-token pass ("n_tokens = 7064 > size = 0"), ran as 2 passes: 1544 t/s against 2079; slope 0.60 vs 0.42 MiB per token; k-pool indexer models (Qwen3.8, GLM-5.3) probes 1.5x too big (0.585 vs 0.389) | line through two close measured sizes with a negative fixed part cut to 0; full-context kpool state took the whole context's pools after kpool_build_sizes sized to the reserve's cells | 253289d479, cbb2abca63, 7aee3b1a0d FIXED | stale-number |
| call-end chain ban | every streamed staging=0 call banned the prompt chain at its size and up | g_pp_mb > 0 on a non-chain path read as a refused chain; the stream sets it too | b792b9517f FIXED | flag-coupling |
| `run.py` calib | calib exited with "no models match" for k31 | probed only models onepass.csv lists | 827a476f05 FIXED | harness |
| cuBLAS handle creation | Qwen3.8 IQ3_S, 2 x 3090, default: `cublasCreate_v2` resource allocation failed in a parallel branch's stream; server restarted at micro-batch 8192 | handles made in the first prompt pass, after the pool took the free VRAM (~95 MiB margin gone) | 44c3a1871b FIXED (handles before the pool reads free VRAM) | other: init order |
| `ggml-backend-sched` CPU-split to card copy | Qwen3.8 IQ3_S, 2 x 3090: 2.67 ms of a 15.1 ms decode graph in 'wait before GPU' (98 GPU splits), stock 0.40 ms | drain held the host until the layer's cache chain ended; card idled per layer | e5cf654bc7 FIXED (async upload on the card's stream; `GGML_SCHED_H2D_ASYNC=0` keeps the drain) | other: sync drain |
| link byte counters (STA/UPL, `/moe/stats`, viewer) | Qwen3.8 IQ3_S: split pass sent 32.3 GB in 2 s, status line 0.43 GB/s, NVML 18-28 GB/s; about 17 of 53 GB into the expert card in no counter (NVML rx 54.0 GB, engine 35.5 GB) | counters took only the cache's own uploads; `tensor_set_2d_async` neither counted nor paced | 9ffc2b0fff, a49293890c FIXED | harness |
| DeepSeek KV clear (kv-tail contiguous record) | DS without -c on two cards: second request computed with zeroed experts (verify: 99 experts of layer 27 on CUDA1, staged and staging=0); in every release since 2026-10-25 | single-member per-cell KV record gives a contiguous view; the clear took it for a packed one and zeroed 1048576 cells = 1280 MiB over the expert blocks behind the zone | 5198efc364 FIXED | past-end |
| stored VRAM need outside the pool | staged start after a staging=0 one sized its pool to the smaller figure (DeepSeek 233 against 163, GLM 3.5 216/149, MiMo 195/131 MiB) and died at first prompt: CUDA1 free 1 MiB, OOM in mul_mat_q/flash attention/cuBLAS, restart, request lost; later `cublasGemmEx` with 1 MiB free in DS and GLM 3.5 | first guess: kernel code loaded at first launch per mode; real cause: a `--moe poison` start measured the need when most of the model was pinned (149 against 220 MiB) and stored it | f91215c7d6 (staging=0 key), adbca23937 FIXED (poisoned start stores nothing; key removed) | stale-number |
| planner router-or-stream switch under staging=0 | GLM 3.0 12k: router won the first big call at 2.7 s against 3.5 s on paper; chain took 16.3 s, forced stream 12.1-12.7 s, staged chain 11.8-12.6 s | measured compute per token is 0 until a pass ran, both priced as uploads alone | 086447e995 FIXED (calculated compute until a pass is timed) | stale-number |
| dense split under staging=0, planner posteriors | GLM 3.0 12k, 2 x 3090: expert card waited 4.2 s of a 15.4 s pass; 760 prompt t/s against 1030 staged; stored staged split 11.6 s stood as prediction of a 15.5 s staging=0 pass | cache path switched off when the dense split took the call, experts became plain graph inputs; posterior regime key lacked the staging mode | 8dfc92d0c1 FIXED | flag-coupling |
| `ggml-backend.cpp` scheduler card stamps, `dsplit_bal_step` | staging=0 dense split pass over the stream into the blocks (GLM 3.0 12k, build ns_split_keeps_stream): no `split balance` line, token shares stay 50/50, nothing says why | card stamps are taken only for passes whose expert splits start with a MUL_MAT_ID that carries an id range (the staged split's shape); the stream's expert splits start with the table read, so the pass is never measured and `dsplit_bal_step` returns on "this pass was read already" | OPEN (found 22:55) | silent-fallback |
| `run.py build --commit HEAD` | built a 9-day-old commit (27 min full compile instead of 25 s); the fix row would have run the old code | `--commit` is resolved in run.py's build repo, not in the tree the command is run from | WORKAROUND (explicit hash, then `head -1 BUILD_INFO`); run.py unchanged | harness |

Commits of these three days whose message did not say whether they fix a fault (not entered): 2b93165564, 5430bf5cf6, 34fb708aae, 045d09a0e2, bd94e23ce6, 08758800de.
