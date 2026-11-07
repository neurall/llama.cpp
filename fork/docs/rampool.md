# RAM pool: expert tiers by heat

Written 2026-11-07. Goal (user): no dead weight in RAM. Today a model bigger than pinned RAM is pinned in tensor order and the
rest is mapped, so cold experts of the first layers hold RAM while hot experts of the last layers come from the SSD through
the page cache. With the pool the tiers follow heat: the hottest experts sit in VRAM (the GPU expert cache, unchanged), the
hot ones in a pinned RAM pool, the cold ones only on the SSD.

## v1 (this release)

Switch: `--moe rampool=1` (default 0 until the cells below prove it; then on). Off: nothing changes.

Where it hooks:

| place | what |
|---|---|
| loader (`llama-model-loader.cpp`) | with rampool on and the model over the pin budget, no expert tensor (`_exps`) takes pinned host memory: dense host weights are pinned first, the experts stay mapped and are never read through the mapping. The loader reports what it pinned (`pin-used`). |
| pool budget (`llama-moecache-fork.cpp`, `rp_setup`) | pool bytes = planner's pin budget (`pin-budget`, the equation in `common_auto_pin`) minus what the loader pinned (`pin-used`); every 512 MiB chunk only while MemAvailable stays 6 GiB above it (pin_alloc's floor). So the pool pins no more than the tensor-order pin did. Every number is logged. Size classes: one per (slice bytes, type); each class gets the same share of its elements. Chunks are pinned and mlocked at init (no pinning during compute: it cut uploads 7x). |
| fill (at load, right after the usage profile) | the pool reads the experts with the earliest expected next use first (ring below; without a ring the lifetime count orders them), O_DIRECT straight into the blocks, `rd_threads` readers, requests in spaced order (even, then odd). Experts the VRAM cache holds or has queued are not copied. Synchronous: it replaces the loader's pinned read of the same bytes, so load time should stay. |
| decode JIT (`llama_moe_cache_jit_fork`) | first thing, before the cache lock, the layer's misses (not on the card) are asked for (`ggml_rampool_need_fork`): what is in the pool is stamped for this graph, what is out is read now by the readers in parallel (no page faults), what finds no block stays on the SSD. The CPU op reads the block through `ggml_cpu_expert_ptr_hook` -> `ggml_rampool_get_fork` (only a block stamped by this or the last graph; else the weight's own mapping). |
| prompt pass (`ggml_backend_sched_upload_weight`, ggml-backend.cpp) | the experts of an upload range that the pool holds are copied straight from their blocks (runs that are neighbours in the pool as one copy, stamped so they stay until the graph after next, `ggml_rampool_use_fork`); the rest and the bytes past the last whole expert go the old O_DIRECT way. A pass admits nothing into the pool. |
| upload workers (step swaps, JIT uploads) | `ggml_rampool_take_fork`: pool hit = DMA from pinned RAM, held until the copy ends; pool miss = the old O_DIRECT staging read (an expert on its way to a card ranks HUGE_VAL, so the pool does not admit it: no RAM copy of what VRAM holds). A JIT miss was already asked for by the need above, so its upload finds it in the pool. |
| eviction and promotion | the victim is the resident with the longest expected time to its next use among 64 sampled blocks of the class (experts the VRAM cache holds first: a RAM copy of them pays nothing). A newcomer is admitted only when its expected next use beats that victim's, so a cold streamed expert does not churn the pool. Nothing asked by this step or the one before is a victim (its DMA may still run). |
| heat measure | per (layer, expert) a ring of the last 4 decode-step stamps (16 bytes per expert, fixed) and its decode uses over all runs. Expected next use = last stamp + mean gap; one visit: now + its age; past due: now + how long overdue; never seen: after all seen ones, by the use count. Persisted in its own file `<state dir>/<model>/globalhot.ini` (`clock=`, `hot.<layer>=` uses, `vis.<layer>=` 4 stamps per expert, oldest first), written beside it and renamed under `<file>.lock`, read at pool setup; not in the state ini, whose tuning values get reset or rolled back by policy. The first run without the file seeds the uses from the state's lifetime counts (`hot.<layer>`). |

What it replaces when on: `pinr`, `preheat` and `pin-lazy` (the three earlier pinned-copy paths; kept, off). The page cache no
longer caches experts: every expert read is O_DIRECT (pool reads, prompt stream, upload staging), except a slice the pool refused
that the CPU kernel reads (the mapping, as before). Expert tensors whose matmul reads past the slice (Qwen3.8/Next down rows of 640,
`expert_past`) are not given to the pool: the bytes behind a block are another expert's, not the next one's.

Measure (per run, `rampool:` lines every stats period and at exit): experts asked, served from RAM, read from the SSD, refused
(no block), evicted, wait time. The yardstick is the share of expert misses served from RAM and decode t/s on a model larger
than RAM+VRAM (GLM 3.5 at `-simram`, the 35B at `-simram 16g/32g`), against direct_load2 and against the same build with
`rampool=0`. Gate: not slower than direct_load2 on the cells, `jobs -c sane` passes.

Dense weights: the loader pins host-placed dense tensors before any expert, so with rampool on they are resident as long as
the pin budget covers them; a pool class for them would be a pinned buffer without eviction, so none is added in v1.

## v2 (plan only, after the release)

| item | design | measurement it needs first |
|---|---|---|
| dense RAM to VRAM per prompt pass | at extreme context a pass's compute time grows with tokens while the dense bytes' upload time is fixed: upload = dense bytes / measured link GB/s; when it is under a share of the measured pass time the pass streams the dense weights from the pool and their VRAM holds KV/compute. The crossover context is solved from those two measured numbers, logged with them, never a constant. | MiMo at long context: pass time per size (stats.log PAS lines), link GB/s (probe), dense bytes per layer; against the resident layout. |
| richer per-expert stats | fixed-size per expert, persisted with the ring: visits per phase (prompt / decode), measured read cost per tier (the slice's O_DIRECT time differs by size and file locality), co-activation with the layer's other experts (keep or fetch groups). Each stays only if it changes a pick that measures better. | A/B of the eviction rank with and without each term on the same cells; drop what does not move the RAM share. |
| cold path without the page cache | an expert refused by the pool is read O_DIRECT into the worker's staging slot and uploaded from there; today it falls back to the mapping (page faults, a page-cache copy). | share of refused experts per run; if it is small the mapping path can stay. |
| ahead reads | the next layer's router (early router) asks the pool ahead of the kernel so the wait before the CPU op vanishes. | wait time per step from the `rampool:` line; worth it only if it is a visible share of the token time. |

## Gate failure analysis (GLM3.5, rampool3)

Written 2026-11-07. PC1, -pl 220/280, pool unmounted, GLM 3.5 (128 GiB) pf12k1k 1024 out, row 1 cold, row 2 warm, one power session.
Baseline: up9_merged2 (90d772e026) pp 336/585, decode 15.3/14.9. Pool-fitting models (OLMoE, Qwen3.6 35B, Qwen3.8 IQ4_XS) pass: rampool is off
there (the model fits the pin budget), rp=1 = rp=0 within noise.

| build | arm | load s | pp t/s cold/warm | decode t/s cold/warm | what the logs show |
|---|---|---|---|---|---|
| rampool1 9badbdddb4 | rp=0 | 68 | 340/574 | 15.0/15.4 | reference |
| rampool1 | rp=1 | - | fail | fail | cuBLAS create OOM on CUDA1 at the first prompt, both attempts: the pool pinned 101.9 GiB after the VRAM expert pool was sized; pinned host memory costs each card ~150 MiB VRAM for its mapping, the 156 MiB margin was gone |
| rampool2 5bd93e0257 (pin before the VRAM sizing) | rp=0 | 67 | 361/584 | 13.3/15.0 | |
| rampool2 | rp=1 | 159 | 117/455 | 11.2/13.0 | VRAM fixed (outside the pool 28 MiB again). Pool 1.8 GiB over the budget (last chunk rounded up + an extra block per chunk): mlock 103.5 vs 101.5 GiB, memory PSI 26-77% at load and first pass (rp=0 under 2); the 5.1 GiB pass host buffer pinned in 45 s (1.2 s in rp=0); prepin through cudaMallocHost (512 MiB pieces) 57 s; the fill left 20 GiB empty (skipped card-held slices); warm request read 31.7 GiB from the SSD O_DIRECT, waited 13.8 s |
| rampool2 3k | rp=0 / rp=1 | | 152/391 vs 62/100 | 16.7/15.8 vs 12.5/13.4 | same causes |
| rampool3 3dc06a46d4 (budget kept, 1 GiB pieces, card-held fill, demand reads buffered) | rp=0 | 84 | 350/559 | 15.8/15.8 | |
| rampool3 | rp=1 | 85 | 199/466 | 10.3/6.5 | load fixed (prepin 25.7 s, mlock 101.3 = budget, PSI under 10 except 2 s spikes). Fill 100.7 GiB (5752 of 11082 card-held slices in RAM). First 1938-token pass still 46.8 s (rp=0 16.7). Request: misses from RAM 95.6/96.1%, 50.2/44.7 GiB read from the SSD into the pool per request, evicted 15.3k/13.5k, waited 17.9/16.0 s. Warm decode 157 s vs 68 s in rp=0 |

Remaining warm-decode gap (rampool3): 89 s per 1024 tokens (~87 ms per token). The pool waits explain 16 s (~16 ms per token). PSI is low and
mlock equals the budget, so the rest is engine time, not RAM pressure. Major faults 50-250/s (rp=0 400-1000/s), so the mapping is not the cause either.
Open: the log says "103.4 of 118.1 GiB in RAM" while 101.0 GiB was prepinned and mlock shows 101.3: check what the 103.4 counts.

Hypotheses, ranked:
1. Demand reads through the page cache (rampool3) made decode worse: warm decode was 13.0 in rampool2 (O_DIRECT) and 6.5 in rampool3; the other
   rampool3 changes touch load and fill only. A buffered read is one more full copy per slice (page cache -> block, single thread under the reader)
   and it evicts the page cache that the refused (mapped) slices live on. Test first: revert only that line (rp_reader back to t.dfd).
2. Pool churn: ~14k reads + 14k evictions (~45-50 GiB) per request, the warm request no better than the cold one. The victim rule takes card-held
   copies first, but prompt passes churn the cards, so those copies are needed again shortly after. Evidence: SSD bytes per request do not fall from
   cold to warm. Test: count re-reads of slices evicted in the same request; try no eviction during a prompt pass (pass churn is not decode heat).
3. The CPU kernel does not get the block: ggml_rampool_get_fork returns null when the slice was not stamped by this or the last graph, and the
   kernel then reads the mapping. If the graph counter moves per sub-graph or per card (2-card dense split) rather than per token, stamps go stale
   and hits become mapping reads. Evidence for: decode slower than rampool2 even with fewer SSD bytes. Test: count get_fork nulls per token.

Next: per-token breakdown first (stats.log PAS/STA, -lv 3) of rp=0 vs rp=1 on the same pass, then 1 (revert one line), then 3 (counter), then 2.
The first-pass slowness (46.8 s vs 16.7 s for 1938 tokens) shows in every rp=1 build: read that pass's breakdown beside rp=0's first.
Until this passes, rampool stays off by default (it is: rampool=0) and leaves v1.

## Root cause found (2026-11-07, rampool3 logs, q5h/rampool3-glm35-12k-rp{0,1}.{server,stats}.log)

Not DMA from pageable memory: every pool piece is anonymous THP memory registered with `cudaHostRegister(Portable|Mapped)`
(server log: 101 x "pinned 1.0 GiB in 0.10 s (10.7 GB/s) ... populated first", no PIN_FAIL, mlock = budget), and decode uploads
from pool blocks run at the x16 link's usual 17-21 GB/s (SWT lines). Not the eviction policy either (rampool4 keeps the fill's set).

The cause is the **decode thread tuner** (`llama_context::thread_tune_feed`, src/llama-context.cpp): it alternates the best thread
count with a lower one token by token and adopts the lower one when it wins every pair. With rampool the token time holds the
pool's demand reads (`ggml_rampool_need_fork` waits on the SSD before the kernel, cold tokens 10-50 ms), which the thread count
does not change but which spinning compute threads lengthen (they starve the 8 reader threads on an 8-core box). So every lower
count won:

| arm | DEC threads_decode (stats.log) | CPU expert rate (SWT "all on the CPU") | host cpu busy (STA) | decode |
|---|---|---|---|---|
| rampool3 rp=0 | 16 vs 8, 8 vs 4, 8 vs 6, 8 vs 7 -> 7 or 8, every request | 26-31 GB/s | 45-55% | 15.8/15.8 t/s |
| rampool3 rp=1 | cold 16 -> 8 -> 4 -> 3 -> settled 4; warm 16 -> 8 -> 4 -> 2 -> 1 | 10.6 GB/s | 7-8% | 10.3/6.5 t/s |

At 0.84-0.97 GB of CPU-computed expert bytes per token, 10.6 instead of 27.7 GB/s is 45 ms per token, plus the 16 ms of real
SSD waits and the starved readers: the whole 87 ms warm gap. rp=0 shows no bias because its stalls (page faults of the mapped
tail) happen inside the kernel on every thread alike.

Fix (commit 380889d64f, build rampool5): the pool keeps a never-reset wait total (`ggml_rampool_wait_us_fork`), and
`thread_tune_feed` subtracts the wait that accrued during the token before judging the thread count. ~10 lines, harmless with
rampool off (returns 0).

Result, build rampool5 (q5h/rampool5-glm35-12k-rp1.*, rows 23730/23731, answers read: coherent code review): decode 15.8 cold /
15.3 warm against rp=0 15.4-15.8 and up9_merged2 15.3/14.9: the decode gate passes. pp 196/487 against 338/559: the prompt gap stays.

### Second cause: miss uploads DMA from the mapping (the user's hypothesis), fixed in the tree, NOT yet built

The 1938-token first call (decode chain path) took 43.8 s against rp=0's 17.6 s with the same bytes per token (cpu 12.7 MiB, upload
32.5 MiB). During it the SSD read 78 GB through 800-1200 major faults/s of 1.5-2.6 MB, page scan 1 M pages/s, swap in/out thousands
of pages/s, link0 0.82 GB/s against 2.10 in rp=0 (STA/DON lines 105-130 s). 78 GB = the call's upload bytes: the JIT uploads of
`llama_moe_cache_jit_fork` (src/llama-moecache-fork.cpp, both the `jit_n` pool path and the slot path) copy with
`ggml_backend_tensor_set_async(be, dst, srcs[k]->data + id*sz, ...)`, the tensor's own host pointer. With rampool on that is the
never-populated mmap (the loader pins no expert, every pool read is O_DIRECT), so each upload page-faults the SSD and then goes
through the driver's pageable bounce; with rp=0 the same pointer is the tensor-order pinned copy. Decode's per-layer JIT uploads
("1 of 4 uploaded") took the same path, which is also why the thread tuner saw spiky tokens.

Fix in the tree (commit after 72f9a91a07): both sites take `ggml_rampool_use_fork(srcs[k], id)` first (the pinned block, stamped so it
stays until the graph after next, counted in the RPL `prompt copies from RAM` share) and fall back to the mapping only when the
pool does not hold the slice (rampool=1: refused slices, ~4% of asks). Expected: the first call back near 17 s (pp cold ~330), warm
pp near 560; decode unchanged or better (fewer faults). Next for Opus: build rampool6, run the acceptance cells below.

Still open after it (in this order; each one measured on GLM 3.5 12k cold/warm before the next):
1. Refused slices (rampool=1 refuses everything out of the fill set: `read into the pool 0, refused 15605` per request) still go
   through the mapping on the CPU op and on uploads. Cheapest: the upload fallback reads O_DIRECT into the worker's staging slot (the
   path at `read_direct`, line ~8621) instead of the mapping; the CPU op's fallback stays (4% of asks).
2. Demand reads on the decode thread (rampool=2 only): `need_fork` blocks the graph while 15 threads spin. Ask for the NEXT layer's
   misses after this layer's router (early router hook) so the reads overlap the GPU layer; measure `waited` in the RPL line.
3. `ggml_compute_forward_mul_mat_id_tiled` (ggml-cpu.c, before the hook) reads `src0->data`: check whether a prompt pass's CPU op
   takes it (PAS line "cpu MiB" of pass 1 with faults); if so the hook goes before it.
4. The user's tier ladder and transfer-class / hit-waste stats lines (coordinator 2026-11-07 14:30): not started; design after the gate.

Acceptance cells (all `-nm -pl 220/280 -r 2`, one power session, interleaved rp=1 / rp=0 / up9_merged2, answers read first):
GLM 3.5 12k and 3k (fork/runs/rp2.csv glm35-*), then MiMo 12k, DS 12k, then `-simram 32g` small-RAM. Pass = rp=1 >= rp=0 and
>= up9_merged2 within noise on pp and decode, cold and warm.
