# Pool v2: the expert pool without size classes (design)

Behind `--moe pool-v2=1`, default off until measured. CUDA first, one card first. Line numbers are those of `src/llama-moecache-fork.cpp` (M) when this note
was added; G = `src/llama-graph.cpp`, P = `src/llama-poolalloc-fork.h`, V2 = `src/llama-poolv2-fork.h`.

## Why

Qwen3.8 IQ3_S on one RTX 3090, 15.5k prompt / 1k answer: decode 78.0-79.1 against 80.9 t/s (steady decode is equal; the gap is the start of decode). The
prompt's expert stream (`--moe staging=0`, the cstream ring) put 23,421 experts a pass into blocks, 14,106 of them (60%) claimed from other size classes
("no run of 512 blocks in class 1"). Those claims evict decode-hot experts; about 6,300 experts (about 9 GB) are refilled after the prompt while about 70
tokens run at 16-17 ms instead of 12.2 ms. Qwen3.8 IQ3_S has 7 expert byte sizes over 48 layers: 7 classes, interleaved.

Classes exist because an expert's address is `block x nb[2]` in its layer's view: same-size experts share one block grid. They buy O(1) fixed-size allocation and
nothing else.

## The design (user 2026-11-14, 2026-11-21)

* One byte heap per card. Tables hold unit offsets, no classes.
* The heap is laid out by decode heat: coldest at its low end (the KV and lend edge), hottest at its top. A newcomer's target is where it would sit if every hotter
  expert were packed above it (V2 `pv2_heat_dist`, `pv2_target`). It takes the nearest free window within reach, else the coldest colder resident not in
  flight whose hole holds it, else a window over several colder residents (V2 `pv2_place`). A hotter or equally hot resident is never evicted.
* A landing area [M, H) at the tail takes streamed experts with no place during a prompt, reused layer after layer (leftovers have heat -1 once their layer is
  computed). At the switch to decode it is refilled with the hottest experts no heap holds (V2 `pv2_refill`): the only extra uploads.
* Heat: placement during a prompt by the shipped/global map; the switch refill by the request's own decode heat (`tree-key=last`).
* Sizes, replay (`test-poolv2-fork --sim --heat=global`, 50 prompts of changing heat, the shipped map for the placement): landing area 1648 MiB at one layer
  ahead (the first prompt's in-flight peak 1644 MiB + one expert), main heap 18597 MiB; a pass streams 14.3-15.1k experts (today 23.4k); 770-800 experts lost
  per prompt; refill at the switch 760-800 experts, 1.57-1.65 GB (today about 6.3k experts, 9 GB); coverage of the request's decode heat 92.6-93.9% after the
  refill (ideal 94.2-98.2%); free bytes 213 MiB, 200 MiB of them in gaps no expert fits (flat over the 50 prompts).

## Addressing

One table per layer, so up, gate and down share one `nb[2]`, the unit `u = lcm(16, block bytes of the three slice types)` (V2 `pv2_shape_of`). The kernels step
a view by `nb[2] / block bytes` (integer division: the unit must be a multiple of every slice type's block bytes; 16 for vector loads). The slices keep today's
offsets in a block (each at a multiple of 128 from the expert's start). An expert's extent ends behind down's slice plus the largest read past a row; today's
block rounds that up to `lcm(128, block bytes)`. Units, Qwen3.8 IQ3_S: 2448-7920 B (iq4_xs/iq4_nl 2448, iq2_s/* 5904, iq3_xxs/* 7056, iq3_s/* 7920); an
expert is 0.03-1.98% smaller than today's block. OLMoE: Q4_K/Q4_K/Q4_K unit 144, Q4_K/Q4_K/Q6_K unit 5040.

The host table keeps the slot space and its sentinels (`n_slots` uncached, `n_slots + 1` JIT pool): the CPU chain computes a pick whose host entry equals
`op_params[0] = n_slots` (G:3481-3515, ggml-cpu.c:1705-1723), the defer mask and miss lists compare with `n_slots` (G:3206-3233, 3285-3316). Only the device
tables change: they hold the expert's start in units of its layer, -1 uncached (every kernel skips negative ids).

### How a token reaches an expert (kernels)

1. `dev_table[e]` gives the expert's index in the view, -1 a miss (M:4917 `flush_tables` writes it; the stream writes its own snapshot, M:13489-13500).
2. One token: `ggml_moe_bell` (G:3355, the decode bell) or `get_rows(dev_table, ids)` (G:3360) turns the picks into those indices; they are `src[2]` of
   `mul_mat_id` with no `src[3]`/`src[4]` (G:3382-3386). MMVQ reads `x + id * (nb[2] / block bytes)` (mmvq.cu 674, 966-979, 1740).
3. Several tokens: the same indices, plus `src[3]` = the logical expert ids and `src[4]` = the table (G:3385). CUDA MMQ sizes its grid by the table
   (`n_grid = src[4]->ne[1]`, mmq.cu:245-353) and reads `expert_map[zt] * stride_channel_x` (mmq.cuh 1115, 1204, 1279); the sorted path loops over the
   table's experts (ggml-cuda.cu 4304-4371, slice at `expert_block[i02] * nb02`). Neither sizes anything by `ne02` except thresholds (should_use_mmq).
4. Float experts (F16/BF16/F32) take mmf: with more than 16 columns it sizes its tiles and its id helper's grid by `ne02` (mmf.cu 40, 76-105; mmid.cu:187).
   With unit views `ne02` is millions: v2 keeps `u = B` for such layers until mmf reads the table.

Checked by test-backend-ops `MUL_MAT_ID ... unit` (tests/test-backend-ops.cpp `test_mul_mat_id_unit`): a view with `nb[2]` one unit of about 4 KiB, experts at
unit offsets with gaps, 1/8/512 tokens, with and without `src[3]`/`src[4]`; IQ3_S, Q4_K, IQ4_XS, Q8_0, F16; CUDA0 50/50 (build poolv2_unit_stride).

### Vulkan (checked, not ported)

Vulkan0 (RTX 3090, driver 595.104, `VK_ICD_FILENAMES=/p/bw/p2p/userspace-595-gl/nvidia_icd_595.json`, the 595 GL userspace on `LD_LIBRARY_PATH`) passes the same
50 cases. They are small (138 units): correct, not at scale. What fails at a real pool:

* `n_as = ne02` (ggml-vulkan.cpp:8048): with unit views that is the unit count. `count_experts` (count_experts.comp:108-175, one workgroup) walks the experts in
  chunks of 1024 and reads every id twice per chunk: a 20 GB pool at 8 KiB units is about 2,500 chunks per matmul. Port: with `src[4]`, `n_as` = the table's
  experts and the ids = `src[3]`, the A offset of expert e = `table[e] * batch_stride_a` (mul_mm.comp:307 and the mat-vec shader), as CUDA's `expert_map`.
* Expert chunks for a view past `maxStorageBufferRange` (ggml-vulkan.cpp:8134-8155, 8378-8395): a chunk's A subbuffer is `ne_c * nb[2]` bytes, so with
  `nb[2]` smaller than an expert the last experts of a chunk read past it. Port: the chunk covers its units plus one expert's bytes (and the chunk count follows).
* `batch_stride_a / LOAD_VEC_A_EFF` (mul_mm.comp:307) is an integer division: the unit must be a multiple of the load width in elements.
* Until then pool v2 asks the backend (`ggml_backend_fork_unit_views_fork`, answered by CUDA/HIP only) and stays off elsewhere.

### CPU reference

`ggml_compute_forward_mul_mat_id` (ggml-cpu.c:1636-1652) sizes by `n_as = ne02` and reads `src[3]` as a host table ("a pick runs where its entry equals
`op_params[0]`"), not as the GPU's logical ids with `src[4]` a table. The engine never gives it a pool view (the CPU chain reads the host weights), so it
matters for test-backend-ops only: the unit cases build `src[3]` so that both readings compute every pick. Port: with `src[4]`, the GPU semantics.

## Every user of slots and classes

"addr" = `class base + slot * B + off[k]`, computed as `slot * view->nb[2]` on a class view. "per block" = an array with one entry per block of a class grid.

### Layout and creation

| site | what | v2 |
|---|---|---|
| M:10026-10064 `pool_class`, `pool_layout` | classes by (3 types, 3 nb[2]); `poolx_block_class` (P:36) B, off, len, past | layer shape = `pv2_shape_of`; no grouping |
| M:10566-10589 `slots_for` | pool sized in rounds of one block per layer | bytes; rounds stay the sizing unit |
| M:10596-10615 | `dtbl` I32 [1,E] per layer; `stbl` shadow tables carved before the guard arms | unchanged (entries are units) |
| M:10695-10746 | class order, `nblk`, leftover to the first class, pool-floor, bases back to back | one heap |
| M:10752-10838 | `poolx_buffer_plan`, split buffers (a backend with a buffer limit), VMM room buffer, `area_b` = last-class blocks | one buffer; the tail area in bytes |
| M:10881-10911 | grids (`gn`, `gbase`), `pool_state` per class: owner, free stack, home, ring_lo/hi | the heap |
| M:10917-10970 | foreign classes (another card's shapes) and their views | multi-card, later |
| M:10972-10991 | the views: `ne[2] = gn`, `nb[2] = B`, `data = base + gbase + off[k]`, `n_slots = gn`, `pcls` | `nb[2] = u`, `data = base + off[k]`, `ne[2]` units |
| M:10996-11114 | `alias_pool`: segs per class, span, tail area, room start, rcls (ring), floor_end, cs_byte | byte edges |
| M:11223-11238, 11270-11273 | per-slot arrays sized `gn`; host tables `n_slots`, device -1 | per expert |

### Block runtime and its callers

| site | what | v2 |
|---|---|---|
| M:2373 `pick_slot` (P:559 `poolx_pick_slot`) | free block of the class (stack), else the lowest-rank victim over `[0, n_active)` | `pv2_place`: a window and 0..n victims, all colder |
| M:2398 `pool_take`, M:2423 `pool_release` | owner, free stack, `n_own` | heap add / drop |
| M:2434 `evict_slot` | slot arrays, table entry -> `n_slots` | also the heap drop |
| callers of pick_slot | M:2588 profile_preload, M:4788 moe_fill_cb, M:7916 reheat_free_blocks, M:8398 replay_prefetch, M:12341 jit_fork, M:15159 swap_layers, M:16042 step | each takes a victim list |
| other pool_take | M:5734 keep_hot_move, M:13153 cs_tail_moves, M:13343 cs_plan, M:14278 / 14341 pp_upload_fork, M:17290 preheat_swap | heap add |
| other evict_slot | M:5731, 5894 (zone), 7792 (lend), 12851 cs_evict, 12869 cls_take_bytes, 12906 cls_home, 13148, 13660 cs_begin, 14276, 14340, 17289 | heap drop |
| `n_own` | M:2416-2430, 5895, 7793, 12869, 12906, 15140, 16131, 16151, 16273 | residents per layer |

### Uploads

| site | what | v2 |
|---|---|---|
| M:307 `upload_job` | (layer, expert, slot) | slot = the expert; the start is the layer's `at[e]` |
| M:4855 `upload_slice` | bounds and offset `slot * nb[2]` | through the helper |
| M:11474-11741 the upload worker | every branch (rampool, pin_ptr, O_DIRECT, staging, bounce, nosync, chunked, past bytes) writes at `slot * dsts[k]->nb[2]`; the gate and the ledger take `led_block(pcls, slot)` | one helper for the offset; byte ranges from the start |
| M:3685 `publish_job` | slot arrays, table entry | per expert |
| M:4751 `moe_fill_cb` (adopt from a staged pass) | compat test `c->nb[2] >= weight->nb[2]` (4761); returns `c->data + slot * c->nb[2]` (4807) | the block bytes, not nb[2]; the helper |
| M:5274 `moe_where_cb`, M:5317 `moe_src_cb` | the same test and offset (5292/5308, 5343/5353) | the same |
| M:12111 `jit_fork` | the JIT pool has own tensors (12314, unchanged); its pick_slot branch writes pool views (12383) | the helper |
| M:15059 `swap_layers_fork` | reads and writes at `slot * t3[k]->nb[2]` (15118, 15163) | multi-card, later |
| M:4730 `expert_past` | the bytes read past a slice, per class `past[k]` | per layer shape |

### Decode step, refill, preheat

| site | what | v2 |
|---|---|---|
| M:15447 `llama_moe_cache_step_fork` | misses by score; per class a cached victim order (cls_order/cls_min, 15984-16068); pay-back margin; budget; evict, take, upload | `pv2_place` with the score as heat |
| M:15398 `tables_check` | host entry s < n_slots needs owner[s] == layer | per expert |
| M:2491 `profile_preload` (boot fill) | per layer its share, hottest into the highest free block | packed in heat order (`pv2_heat_dist`) |
| M:7872 `reheat_free_blocks` | per class its free blocks to the hottest missing of its layers (`poolx_refill_parts`) | free windows of the heap only, hottest missing first |
| M:17256 `preheat_swap` | per class hot (uncached) for cold (cached) | none: the switch refill |
| M:8606 `topic_drop_foreign`, M:8478 `topic_hottest` | per-layer scans; capacity per class in blocks | per expert; bytes |
| M:5679 `keep_hot_move` | the lend moves hot experts of its blocks into colder blocks of the same class | none: the low end holds the coldest |

### The prompt stream (cstream ring, `--moe staging=0`)

| site | what | v2 |
|---|---|---|
| M:1320-1360 `cs_ring_floor`, `cs_ring_apply` | the ring reserve behind `floor_end`: an entry's experts x class B, snapped to class blocks | the landing area in bytes |
| M:13166 `cs_plan` | the entry's uncached experts; tiers 0-10 over blocks (free, empty, computed, claims of other classes, layers to come); a window of consecutive blocks from the cursor (`poolx_ring_window`), else tiered picks; take kinds evict / claim / yield; `wait_at` from the victim's entry | per expert: resident in place (pinned), else `pv2_place` by the placement heat, else the landing area near its cursor; `wait_at` from the victim's layer |
| M:12814-12909 `cls_claimable`, `cs_evict`, `cls_claim`, `cls_take_bytes`, `cls_home` | claims of other classes' bytes and their return at the call's end | none |
| M:13088 `cs_tail_moves` | `--moe cs-tail` moves within a class | none |
| M:13400 `cs_copy` | runs of consecutive experts into consecutive blocks as one 2D copy (dst pitch `nb[2]`, 13445-13456), past bytes, the table snapshot (13489-13500), `ring_ev` | runs of adjacent placements; offsets from the start; snapshot entries in units |
| M:13570 `cs_begin` | first pass: evicts the ring range `[ring_lo, ring_hi)` of every class (13654-13667) | the landing area's leftovers |
| M:13711 `cs_ahead`, M:13763 `cs_layer`, M:13852 `cs_split_cb`, M:13525 worker | order, depth, events (`cs_done_ev`, `ring_ev`, `cs_start_ev`) | unchanged |
| M:13863 `cs_end` | drain, `cls_home`, tables | + `pv2_refill` of the landing area (the switch) |
| M:12428 `pp_blocks`, M:12671 `pp_chain_fork`, M:12485 `cs_shadow_fork` | usable blocks against an entry's experts; shadows' views = a twin of the same class | bytes; shadows later |
| M:14149 `pp_upload_fork` (chain-up, ahead-up) | grid scans incl. -4, claims, writes at `slot * nb[2]` | the heap |

### Lend, KV zone, room

| site | what | v2 |
|---|---|---|
| M:7539 `llama_moe_cache_alias_fork` | the graph's lend: per class the prefix `[0, need)` of its blocks (`poolx_lend_blocks`, `poolx_lend`), drops jobs in it, `keep_hot_move` | a byte range from the heap's low end: residents in it leave |
| M:7832 `alias_release` | keeps `ceil((want_b - base) / B)` blocks per class | the range back to the heap |
| M:5829 `kv_blocks_apply` | a byte range -> whole blocks of every class it overlaps, owner -3 / ROOM | a byte range of the heap |
| M:6032 `kv_layout`, M:6139 `kv_tail_plan`, K `kvz_*` | zone edges on class block edges (`kv_lend_round`, `kvz_block_floor`), starved = no block left | edges on 256 bytes |
| M:6613-6911 retype / RAM part | the same take/give | the same |
| M:7472 `room_apply`, M:7456 `room_edge_down` | CUDA's room: whole blocks of the last class at the tail, unmapped | the tail area outside the heap: map/unmap only |
| M:6002 `poison_free_blocks` | runs of free blocks per class | free extents |

### Verify, ledger, guard

| site | what | v2 |
|---|---|---|
| M:13923 `llama_moe_cache_verify_fork` | reads each cached expert at `slot * nb[2]` + past, compares with the host | the start |
| M:2756 `led_block` | the one block -> bytes map (seg base + b x B) | the start and the extent |
| M:2766-2996 ledger, M:2867-3253 guard | owner per block, block ends from a uniform B, `g_gtbl` copies of the device tables indexed as slots | byte ranges; table copies kept in slot space |
| M:5619 `pool_fill_log` | per class owner counts x B | heap counts |

## Heat sources (what v2 can read)

| source | where | when | scale |
|---|---|---|---|
| `glob_count` (state ini `hot.<layer>`, else the GGUF's `moe_cache.expert_usage`, else the shipped `hotmaps/<section>.ini`) | M:243, loaded by `profile_preload` M:2497-2560 | start; +1 per decode token (and prompt token with `count-prompt`) M:3514 | raw; the shipped map 1e5 per layer |
| `topic_heat` | M:249 | set at the prompt's end by `topic_steer` (M:9680) or `tree_read` (M:9849, the tree's tg leaf) | 0..1, one scale over all layers |
| `heat_cnt` / `vis` (`globalhot.ini`) | M:262, 261 | start (`rp_heat_load`), per decode token | raw; only with `--moe rampool` |
| `score` / `xscore` | M:1499, 2338 | composites of `win_count` + `glob_count / gmax_all` | one scale over the pool |
| tree pp/tg leaves | M:8013-8312 | pp stored at the prompt's end, tg at the answer's end; a key at the prompt's start only with `tree-key=emb` | raw |

Before a prompt every cache layer has `glob_count` (the placement heat; the shipped map is the same scale for every layer). At the switch `topic_heat` holds the
predicted decode heat of this request when `dynhot` is on (`tree-key=last` keys it by the prompt's final hidden state); otherwise `glob_count` again. The
request's own decode counts exist only after its answer.

## Stages

Each stage builds, keeps `pool-v2=0` the same bytes and the same picks, and has a unit test or a test-backend-ops check before the engine rows.

1. **Unit views, classes kept.** `--moe pool-v2=1` on one card whose backend answers `ggml_backend_fork_unit_views_fork`: every class's views get
   `nb[2]` = the class unit, the device tables hold `block x B / u`, every slot offset goes through one helper (`slot x B`). Classes, owners, ring, lend, KV
   and room are unchanged, so the placement is today's: answers and speed must equal `pool-v2=0`. It proves the engine paths take unit views (MMVQ decode,
   the bell, MMQ with the table, the stream's 2D copies, verify, adoption). Float-expert classes keep `u = B`.
2. **Handles and the heap, decode side.** A layer's slot becomes the expert id (`n_slots = E`), its start lives per expert; one `pv2_heap` per card over the
   pool minus the tail area. pick_slot and its callers, pool_take/release, evict_slot, the step's victim choice (`pv2_place`, heat = `xscore`), the boot fill
   in heat order, `reheat_free_blocks` (free windows, hottest missing first), verify, ledger, guard, stats. Lend and KV zone: byte ranges from the heap's low
   end (new pure functions: take a range, give it back; tests with the brute-force checker). Room: map/unmap of the tail area only.
3. **The stream on the heap.** `cs_plan` per expert (resident in place, else heat placement, else the landing area), copies per run of adjacent placements,
   waits per victim layer, no claims; `cs_end` refills the landing area (`pv2_refill`) by `topic_heat` (or `glob_count`) and logs one line: experts placed
   by heat / landed / evicted colder, refill experts and bytes. Landing size from the first prompt's in-flight peak + one expert.
4. **The prompt chains and Qwen3.8.** `pp_upload_fork` (chain-up, ahead-up), `pp_blocks`/`pp_chain_fork` and the pass planner's reserve in bytes. Then Qwen3.8
   IQ3_S one card 15.5k/1k against `pool-v2=0`.

Later: Vulkan (above), the CPU reference's `src[3]`/`src[4]`, mmf's table grid (float experts), several cards (shadows, expert split, JIT pool, layer swaps),
compaction (move residents when stranded gaps pass a bound; the replay shows 200 MiB flat), the eager pre-fill of the prompt's first layers before a prompt.

## Risks

* Uploads in flight against running graphs: today a block's owner and `slot_in_flight` keep a running graph's experts and a pending upload apart; the stream
  waits on the victim entry's `cs_done_ev`. In v2 a window can cover several residents of several layers: every one must be out of flight and, during a
  stream, its layer's done event must be behind the copy. `pv2_place` never takes a pinned resident; the engine must pin every resident a queued or running
  graph reads, including the decode graph's experts while the step runs beside it (`--moe sync=0`, `graph_ev`).
* The read past an expert's last row lands in the next resident's bytes or a free gap (today it stays inside the block). The values are the next expert's
  quant data or whatever a free gap held, read by a tile whose activations past the row are zero, as in a normal expert tensor; NaN bit patterns in a gap
  that the lend or the KV wrote are the open question. v2 keeps the read-past bytes inside the extent, as today's block does.
* Table entries are int32 units: 24 GB / 2448 B is 10M, fine. MMQ's `expert_map[zt] * stride_channel_x` is an int in blocks: the same values as today.
* Fragmentation: the replay holds 200 MiB stranded flat; a long session with other heat may not. Compaction is the answer if it grows.
