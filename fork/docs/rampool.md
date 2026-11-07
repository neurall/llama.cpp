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

## Fill cost spec (2026-11-07, rampool7 rp=1 warm pp 496 against rp=0 583 / up9_merged2 585, rows 23743 / 23741)

Logs: q5h/rampool7-glm35-12k-rp1.server.log lines 552-618 (cold) and 671-697 (warm), q5h/rampool6-glm35-12k-rp0.server.log lines 309-441.

**What happened.** The cold rp=1 request took the dense split on both calls (1938 tokens, then 10496: 828 prompt t/s, a win). Its first split
pass paid the copy fill, 8.5 GiB of the other card's dense weights card to card through pinned host staging: `moe_mirror_fill ... in 8904 ms`
(rp=0 warm, the same bytes: 1492 ms). `llama_dsplit_note_fill_fork` learned 8904 ms as the split's per-prompt cost (`g_dsf_fill`, one sample).
On the warm request both calls then went to one card: `split factors ... one card 15049 ms, split 12624 ms + copy fill 8904 ms: one card`
(line 686), although the size bucket had both arms measured a minute earlier (`this size: one card 581, split 828`, line 696) and the measured
branch in `llama_dsplit_decide_fork` had picked the split. The factors (`dsf_decide`) override the measured verdict unconditionally
(src/llama-dsplit-fork.cpp:2771-2777). Warm: 6.3 s (one card, 1938) + 17.9 s (one card, 10496) = 24.5 s = 508 t/s; rp=0 warm: 7.4 + 1.5 fill
+ 13.1 split = 20.8 s = 598.

**(1) Is the fill cost real on a warm request?** The copies live in expert-pool blocks lent only while a prompt splits ("expert blocks again in
decode", `llama_dsplit_mirror_place_fork`), so a request that splits pays one fill (the copies stay across the calls of one request: nothing is
lost between the 1938 and the 10496 pass when both split). The fill is not double-counted in the pass: `dsplit_update` subtracts `p.fill_ms`
from the pass before it becomes a per-token sample (line 2587), and the per-size t/s (828) excludes it too. The number is the problem, not the
charging: of the 8904 ms, about 7.7 s were `ggml_cuda_xstage_copy` pinning new staging blocks in the middle of the fill (lines 560-588: ~30
"new N MiB staging block ... pinned in 0.45 s", the 4 big slots of each direction growing 2 -> 34 -> 52 -> 68 MiB, each growth a free + a new
pin; rp=0 pins the same blocks in 0.01-0.03 s, rampool's 101 GiB of locked THP memory leaves 11 GiB available and every pin reclaims first).
Those pins happen once per process (the blocks stay; the warm fill would have needed one, slot 2 of CUDA0->CUDA1 at 16 MiB). The steady cost
of the fill is the copy alone, ~1.2-1.5 s (rp=0 measured, 8.5 GiB over x16 + x4), and that is what a warm request pays.

**(2) Minimal rules.** Three changes, no header edits:

a. `src/llama-context.cpp:1461` `moe_mirror_fill`: issue `todo` largest tensor first (`std::sort` by `ggml_nbytes(pr.first)` descending, one line
   before the loop). The first four big copies of each direction grow the four ring slots to the largest size once; every later copy fits: 8 pins
   per process instead of ~30 (`ggml_cuda_xstage_copy`'s "oldest block big enough" rule then always finds one). Rule "no CUDA free/realloc".

b. The learned fill excludes staging pins. `ggml/src/ggml-cuda/ggml-cuda.cu:4469` `ggml_cuda_xstage_copy`: add a never-reset
   `static std::atomic<uint64_t> g_xstage_pin_us` and add the big block's pin time to it (the `t0` already taken before the `mmap` at ~4515; also
   the `cudaHostAlloc` fallback), exported as `extern "C" uint64_t ggml_cuda_xstage_pin_us_fork(void)` (defined in ggml-cuda.cu, declared locally
   in llama-context.cpp next to the `llama_moe_cache_ledger_fork` declaration at line 1455, "no header edit"; 0 when the CUDA backend is not
   linked: weak default or `#ifdef GGML_USE_CUDA`). In `moe_mirror_fill` read it before and after; `llama_dsplit_note_fill_fork((elapsed - pins) / 1e3)`
   and log both: `copies filled ... in 8904 ms (7700 ms of it staging-block pins, once per process)`. Same pattern as `ggml_rampool_wait_us_fork`
   for the thread tuner (rampool5).

c. Crossover safety, `src/llama-dsplit-fork.cpp:2771-2777`: the factors (`dsf_decide`) decide only when this size bucket does NOT have both
   arms measured. Where it does (`B.n[0] > 0 && B.n[1] > 0`, the "measured: faster at this size" branch at 2746), the measured verdict stands,
   with the fill this request still owes added to the split side: `B.us_tok[1] * n_tokens / 1e3 + fill_ms <= (1 - split_min_gain) * B.us_tok[0] * n_tokens / 1e3`,
   `fill_ms` = `g_dsf_fill.us` when the copies are not in place (`llama_dsplit_mirror_take_upload_fork` would return something; cheapest: a
   `g_mirror_lost` flag set where the blocks are given back at the end of the prompt, cleared in `moe_mirror_fill`), else 0. Change: wrap the
   `fc >= 0` override in `if (!(B.n[0] > 0 && B.n[1] > 0))`, and add `+ fill` to the measured branch's comparison. A measured pair at this size is the
   one thing the factors are calibrated from; a curve must never overrule it (the user's "proper crossovers": a wrong pick here cost 1.45x).
   Measured arms are per pass size; `dsf_decide` keeps the sizes not measured.

Expected after a+b+c (GLM 3.5 12k, -pl 220/280, pool unmounted): fill noted ~1.3 s (the pins reported apart, ~3.6 s on the cold request only);
warm `split factors ... one card 15049 ms, split 12624 ms + copy fill ~1300 ms` and the bucket verdict `measured: faster at this size` at 10496:
split; warm = 6.3 (1938 one card: its own bucket says one card 306 against split 121-with-fill, a correct pick) + 1.3 + 12.7 = 20.3 s, pp ~610
(rp=0 598, gate >= 583 within 5%: >= 554). Cold: 15.9 -> ~10 s on the 1938 call (pins 3.6 s instead of 7.7), 12.2 s on the 10496: pp ~560 (was 434).
Decode unchanged (15.3-15.8).

**(3) rp=0.** The same two faults exist there (the factors override a measured pair; the fill sample carries the one-time pins) but its pins take
0.01-0.03 s, so the learned fill is ~1.5 s and the picks come out right; under `-simram 32g` (small-RAM acceptance cell) rp=0 pins slowly too and
would show it. a-c are generic and fix both arms.

**(4) rp=0 GUARD kill at load (14:57:53, watch-2026-11-07.log 8203-8221).** Environment, not a rampool7 change: rp=0 runs no pool code; its loader
pinned the tensor-order copy (anon 101.8 GiB) as in up9_merged2, then the last pin, the 4 GiB pass host buffer, came at MemAvailable 16 GiB with
10 GiB of page cache from the just-read model file (pool unmounted); 2.76 GiB of that buffer landed in CMA blocks, were given back and faulted
in again (9.2 s, server.prev.log line 141), and the kernel swapped anonymous pages to zram at 87k pages/s (PSI 31) instead of dropping the cache,
which is the guard's kill rule. The retry 12 s later with the same binary loaded clean (swap-out 0, PSI 12, row 23741 ran). A load-order race
with reclaim; if it recurs: `posix_fadvise(DONTNEED)` on the file after the loader's reads when the pool is unmounted, or pin the pass host
buffer before the weights. Not in this fix.

Acceptance cells (one power session, answers read first): GLM 3.5 12k `-nm -pl 220/280 -r 2`, rp=1 and rp=0 interleaved, then up9_merged2:
pass = warm pp >= 583 (rp=1 and rp=0), cold pp >= rp=0's, decode within noise; the server log of each warm request shows `measured: faster at
this size` -> split at 10496 and `copy fill` <= 2000 ms; then the 3k cell and `-simram 32g` for rp=0.

## Robust planner design (2026-11-07, design only; Opus implements)

Logs read (copies in the job tmp dir): q5h/{up9_merged2,rampool1,2,3,5,6,7,8}-glm35-12k-rp{0,1}.server.log, rampool8 rp0 repeat (rows 23752 cold pp 341,
23753 warm pp 481) included. GLM 3.5 12k, -pl 220/280, x16 (CUDA1, experts) + x4 (CUDA0). Each request is two prompt calls: rp=0 2034 + 10400 tokens, rp=1
1938 + 10496. Build ids: up9_merged2 = before 5757dad57d; rampool6 = 5757dad57d; rampool7 = 5757dad57d + b (pins logged, no planner change); rampool8 =
7d5bf6e72.

### 1. Every prompt-pass decision, in hindsight

Measured truths on this machine (the other path at the same size, from the runs where it ran; every number is a whole call, wall):

| call | one card | split (pass only) | copy fill owed when the copies are not in place |
|---|---|---|---|
| cold 2k (first call of the process, experts not on the card) | decode chain 18.6-24.9 s (calculated 11.3-13.5) | 6.7 s (rampool7 rp1: 15.9 s call = 6.7 pass + 8.9 fill of which 7.7 pins) | 1.4-1.5 s copy; pins once per process 0.03-2.0 s (rampool8, largest first), 7.7 s before |
| warm 2k | stream 5.9-7.7 s (263-330 t/s) | 6.7 s (one sample) | same |
| 10k (cold or warm) | stream 16.8-18.4 s (566-626 t/s) | 11.6-12.5 s pass (split balance "pass" line); as a call 12.7-13.5 s warm | same |

Decisions (time = the chosen path's call; loss = against the faster path's call at that size, fill included where it was owed; ok = right pick):

| build / arm | call | pick | by | predicted (one card / split + fill) | took | loss | wrong input |
|---|---|---|---|---|---|---|---|
| up9_merged2 rp0 cold | 2034 | one card (decode chain) | moe-cache: decode chain faster | dec 11.3 s calc / split never weighed | 18.6 s | 10 s, 2.2x | A, C |
| up9_merged2 rp0 cold | 10400 | one card | the call's path keeps the pass (split 13.8 measured prior vs planner one card 11.7) | 11.7 / 13.8 | 17.6 s | 4.4 s, 1.33x | C (planner one-card time has no upload; the stream calc said 17.5) |
| up9_merged2 rp0 warm | 2034 | one card | no split arm here | - / - | 7.5 s | ok (1 s better than split+fill) | - |
| up9_merged2 rp0 warm | 10400 | split | planner / factors | 17.8 / 14.2 | 13.2 s | ok | - |
| rampool1,2,3,5,6 rp0 (and rampool5 rp1) cold | 2k | one card (decode chain) | moe-cache | dec 11.3-13.5 calc | 18.6-23 s | ~10 s, 2x | A, C |
| same builds cold | 10k | one card | planner: >15% apart | 5.8 / 7.0 (compute only) | 17.2-17.4 s | 4 s, 1.3x | C (planner one card 5.8 s, took 17.3: the upload is missing) |
| same builds warm | 2k | one card | nothing measured for the split | - | 7.4-7.6 s | ok | - |
| same builds warm | 10k | split | planner (one card 17.6 incl. the measured stream) | 17.6 / 7.1 | 13.2-14.1 s | ok | - |
| rampool2,3,6 rp1 cold | 1938 | one card (decode chain) | factors: 5.5 = 5.5 upload-bound + fill | 5.5 / 5.5+0.7 | 23 s | ~13 s, 2.5x | A, C, D (the factors' one-card time is max(U, compute) = U 5.5 s; the decode chain took 23 s) |
| rampool2,3,6 rp1 cold | 10496 | split | factors (state file) | 15.3 / 11.7+0.7 | 14.3-17.3 s | ok | - |
| rampool2,3,6 rp1 warm | 1938 | one card | factors (fill 1.5 owed) | 5.3-5.7 / 5.3-5.7+1.5 | 5.9-6.4 s | ok | - |
| rampool2,3,6 rp1 warm | 10496 | **one card** | g_pp_no_dense veto (rampool6) / "the call's path keeps the pass" | measured 603 vs 681-736 t/s split | 16.8-17.4 s | 2.5-3.9 s, 1.2x | B (exact-size veto: the one-card stream had not run at exactly 10496, only at 10400) |
| rampool7 rp1 cold | 1938 | split | planner 12.3 / 6.2 | 12.3 / 6.2+0.7 | 15.9 s (8.9 fill, 7.7 pins) | ok against 23 s, pins made it look lost | E (one-time pins inside the fill sample) |
| rampool7 rp1 cold | 10496 | split | planner | 18.2 / 15.6 | 12.7 s | ok | - |
| rampool7 rp1 warm | 1938 | one card | factors, fill 8.9 | 5.1 / 5.1+8.9 | 6.3 s | ok (by luck: fill 8.9 is 6x the truth) | E |
| rampool7 rp1 warm | 10496 | **one card** | factors override: split 12.6 + fill 8.9 > one 15.0 | 15.0 / 21.5 | 18.1 s | 5.4 s, 1.45x | E, F (one-sample fill with pins; the curve overruled the measured pair 581 vs 828) |
| rampool8 rp0 cold (repeat, row 23752) | 2034 | one card (decode chain) | moe-cache | dec calc 11.5 | 19.5 s | ~10 s, 2x | A, C |
| rampool8 rp0 cold | 10400 | split | factors (state file): 16.6 / 13.2+0.7 | 16.6 / 13.9 | 16.3 s call = 12.1 pass + 3.1 fill (2.0 pins) + 1.1 | ok (one card 17.5) | - |
| rampool8 rp0 warm (row 23753) | 2034 | one card | factors | 4.3 / 4.3+1.1 | 7.7 s | ok | - |
| rampool8 rp0 warm | 10400 | **one card** | measured branch: split 640 t/s x n + fill 1.1 > 0.92 x one 597 | 17.4 / 16.3+1.1 | 17.5 s | 4.3 s, 1.33x (pp 534 -> 481) | G (fill double count: the bucket arm 640 t/s is the whole call incl. 3.1 s fill+pins; 7d5bf6e72 c adds the fill again) |
| rampool8 rp1 cold | 1938 | one card (decode chain) | factors: 6.1 = 6.1 + 0.7 | 6.1 / 6.8 | 24.9 s | ~15 s, 2.6x | A, C, D |
| rampool8 rp1 cold | 10496 | split | factors | 17.0 / 13.3+0.7 | 14.8 s | ok | - |
| rampool8 rp1 warm | 1938 | one card | factors, fill 1.2 | 5.9 / 5.9+1.2 | 6.5 s | ok | - |
| rampool8 rp1 warm | 10496 | split | factors | 16.8 / 13.5+1.2 | 13.9 s | ok | - |

Of 36 logged decisions 15 picked the slower path; the losses are 1.2x to 2.6x of the call. The cold 2k call is wrong in every run but one
(rampool7 rp1, which paid 7.7 s of pins for it): the decode chain is picked by the moe-cache planner against its own stream calculation and the split is
never weighed there; the split with its copies (7 s + 1.5 fill) is 2x faster than the decode chain (19-25 s). The cold 10k call is wrong in every rp=0 run
before rampool8 (the planner's compute-only one-card time, 5.8 s, against a stream that takes 17.3 s).

**Failure taxonomy (which input was wrong):**

- A. **The one-card path's own prediction was not calibrated**: the moe-cache planner's "decode chain 11.3 s calculated" took 19-25 s every time (the
  misses upload serially behind the router); nothing learns that ratio per path (`g_pp_took` keeps the last time per power-of-two size, never the
  calculated-to-measured ratio, and the first call of a size has no entry).
- B. **Exact-size rules**: `g_pp_no_dense` and `pp_took_us` are keyed to `pp_bucket(n)` (powers of two) and `dsplit_bucket_of` (sqrt(2) steps x context
  band); 10400 and 10496 fall into different buckets of one of them, so a measurement at 10400 said nothing at 10496 (rampool6 rp1 warm, 1.2x).
- C. **Two one-card times that disagree**: the dense-split planner's `plan.t_one_ms` is compute only (5.8 s at 10400) while the moe-cache's stream
  calculation (17.5 s) is near the truth (17.3); the 2026-11-06 bridge takes `plan.t_one_ms - saving` where the path was not measured at that exact size.
- D. **max(U, compute) hides the path**: `dsf_decide` models one card as max(upload, compute); at 2k the upload (5.5 s) wins and the decode chain's 23 s
  never enters, so the factors say "one card 5.5 s" for a call that takes 23 s.
- E. **One-sample learned cost with a one-time part inside**: `g_dsf_fill` took 8904 ms from one sample (7.7 s pins once per process) and charged it to
  every later request (rampool7: 1.45x). Pins are now logged apart (7d5bf6e72 b) but the estimator still takes any single sample as the truth.
- F. **A model overruling a measured pair** at the same size (`dsf_decide` override, fixed for the exact bucket only by 7d5bf6e72 c).
- G. **Fill double count**: `llama_dsplit_report_fork(n, us)` gets `ggml_time_us() - t_compute_start_us`, which contains `moe_mirror_fill` (3.1 s with pins
  in rampool8 rp0 cold); the bucket arm `us_tok[1]` therefore includes the fill, and 7d5bf6e72 c adds `fill_ms` on top. This is row 23753's 481 pp: the warm
  10400 call went to one card, 17.5 s against 13.3 + 1.1. **rp=0's warm loss is this planner change, not the pass**: the split balance line shows the same
  pass in every build (pass 11601 ms up9_merged2, 11695 rampool6, 12109 rampool8 rp0 cold with the pins' page reclaim behind it; "wait for other cards"
  7.7-7.9 s in all of them is CUDA0 waiting for the expert card and is not new). The 711 t/s split call of the first rampool8 rp0 run (14.6 s) against 782
  (13.3 s) is 0.4 s more fill (1928 ms: its 709 ms of pins happened at the warm request because the cold one did not split, a (a)-ordering effect:
  the biggest blocks are pinned first) and 0.9 s of pass; of 5757dad57d / 7d5bf6e72 only the largest-first sort (a) touches the copy path, so the one-line
  A/B is the sort reverted (acceptance cell 5). Neither commit touches the pass kernels.
- H. Not seen in these logs but in the code: `dsplit_pass_end_locked` drops passes under a quarter of the card limit, the split planner's calibration
  averages "half and half" from the first sample, `split_min_gain` 8% and the 15% "plan_sure" margin are fixed thresholds.

### 2. The learning design (no tuned constants)

One estimator shape for every path, fed by every measured pass, predicting the actual call sizes from measured quantities. Nothing is keyed to a size.

**Model.** For a call of n tokens starting at c cells on path p (the moe-cache's PP_RING, PP_MIX, PP_CHAIN, PP_DEC, PP_K+i, and the dense split), the
call's time is linear in measured features:

    t_p(n, c) = theta_p . phi,   phi = (1, n, n*c, U_p(n))

1: the fixed part of a pass; n: compute per token; n*c: attention per token per cell (the planner's fitted attention curve says it exists); U_p(n): the
path's own calculated upload for this call, in us (the moe-cache computes it per call from measured link rates and the bytes not on the card: "streaming
upload", "misses uploaded", the decode chain's chunks). The coefficient on U says how much of the upload the path hides behind compute (1 serial, 0
overlapped, 2 when the uploads also stall the compute, as the decode chain does): learned, not assumed; max(U, compute) is gone (fault D). The split adds
the copy fill when the copies are not in place: `fill = theta_fill * bytes` (one coefficient, us per byte of copies), plus nothing for pins (they are
measured apart by `ggml_cuda_xstage_pin_us_fork` and are not part of any request's cost; fault E).

**Estimator: recursive least squares with its own covariance**, per path, dimension k = 4 (fill: k = 1):

    e = y - theta.phi                       (residual of the measured call time)
    g = P phi / (1 + phi' P phi)            (gain)
    theta += g e;  P -= g phi' P
    s2 = ((n-1) s2 + e^2 / (1 + phi' P phi)) / n     (residual variance, running mean, n = samples so far)
    NIS = e^2 / (s2 (1 + phi' P phi))       (normalised innovation squared; 1 is what a correct model gives on average)

Prediction for a call: mean `mu = theta.phi`, variance `var = s2 (1 + phi' P phi)`. One sample moves theta by the gain the covariance allows and no more:
an 8904 ms fill against a prior of 741 +- 741 moves the mean to ~4.8 s and leaves the variance large, and the next 1.4 s sample pulls it back; a single
sample never becomes "the truth" (fault E). O(k^2) = 16 multiplies per update and per prediction: O(1).

**Drift from residuals, not timers.** Keep `nis` as a running mean (same 1/n weight as s2). When a new sample's NIS > 1 (the residual exceeded what the
model's own variance allowed), inflate: `P *= NIS` and `n = k + 1` (the running means restart at the identifiability minimum). A changed machine (power
cap, contention, another process on a card, a new pin cost) then re-opens the variance by exactly the measured surprise, exploration resumes, and
confidence regrows with the next samples. No forgetting factor, no window.

**Prior (cold start).** theta_0 from the quantities the planner already measures at start: compute per token from the matmul bench
(`split planner: matmul latency`), attention per token per cell from the attention fit, fixed part 0, upload coefficient 1, fill us/byte from the
link probe (`g_mirror_fill_us` / bytes). P_0 = diag(theta_0^2) (each prior parameter is uncertain by its own size: in every log here the planner's
compute-only time was 2x off and the bench's "matmul latency" and "probe" rates differ 3.5x; a prior that claims less would veto a measurement) with a
floor of the squared smallest feature scale for a zero prior (the fixed part: one pass of 64 tokens at the bench rate). s2_0 = (theta_0.phi)^2 at the
first call's phi for the same reason. rp=0 and rp=1 see the same code and the same priors; they differ only in the features (1938/10496 against
2034/10400 tokens, the bytes behind U) and in pin time, which is excluded. A state file prior (below) replaces theta_0 / P_0 when present.

**Decision per request, with exploration from the posterior (Thompson).** At the first prompt pass of a request (a pos0 below the last seen one, the
existing `g_dsf_seen_pos0` rule) draw one parameter vector per path from N(theta_p, s2_p P_p) and keep the draw for the request (one consistent view
across its calls: no flip-flop inside a request, and no hysteresis constant). The planned calls are known (the moe-cache's `pp_cells_get` end and the
ubatch, as `dsf_decide` already loops over them). For each remaining call k: `one_k` = the drawn time of the one-card path the moe-cache would take for
that call (it picks among its paths by the same draws: the lowest drawn time), `split_k` = the drawn split time. Gains `g_k = one_k - split_k`; the
fill is a request cost, paid once if any call splits: split the calls with `g_k > 0` when `sum(g_k > 0) > fill_drawn` (fill 0 when the copies are in
place), else none. The probability of picking the slower path equals the posterior probability that it is faster; when the two are within noise this is
near 1/2 and the regret of either pick is within that noise; when the posterior is sure it is ~0. Exploration shrinks with confidence and regrows on
drift by itself; there is no X%, no N requests, no explore flag, no `split_min_gain`, no 15%, and no veto (faults B, C, F). A measured path is never
overruled by a model: the model is the measurements. The draws are redone after any pass whose NIS > 1 (the posterior changed materially).

**Keep as they are:** the measured-pass hygiene in `dsplit_pass_end_locked` (up-wait subtraction, the `g_dsf_urat` upload correction becomes
unnecessary once U is a learned feature and goes), `g_veto_tokens` (the split did not fit: a fact), the fill measured without pins (7d5bf6e72 b),
largest-first (a, pending cell 5).

**Logging (feedback_planner_explains_picks).** One WARN + one `stats.log` `PLN` line per decision:
`plan: request of 12434 tokens, calls 1938+10496 at 0 cells: one card [stream] 6.9 +- 2.1 s / 17.1 +- 0.8 s (n=3, nis 0.9), split 6.8 +- 2.4 / 12.2 +- 0.6
(n=2, nis 1.3) + fill 1.4 +- 0.3 (owed; n=4); draws one 7.4/17.0, split 6.5/12.4: split both (p_split 0.52 / 0.99)`. After the pass (`DON` line):
`predicted 12.2 +- 0.6, took 12.1, nis 0.03, regret 0.0 s` (regret = took - min over paths of the predicted mean, 0 when the pick was the predicted
best; a sum per server in `/moe/stats`: `plan_regret_s`, `plan_hits`, `plan_picks`).

**Persistence (feedback_stored_values_policy).** Store theta, diag(P)/s2 and s2 per path, under the existing calib section and `@hw#build` key,
only when the estimator is determined and quiet: n >= k + 1 and running NIS <= 1. Loaded as the prior (theta_0, P_0). Never the samples, never a
per-size arm: `dsplit_arm4_b*`, `dsf.one.k*`, `dsf.split.k*`, `dsf.fill`, `dsplit_calib5/7_b*` are no longer written (left in the file, ignored).

### 3. Replay of the logged decisions through the estimator (by hand, from the logged inputs)

Inputs each decision would have had: the moe-cache's calculated uploads (logged per call), the planner's prior (logged), the passes measured before it
in the same process (the state file carries the previous run's theta where the build matches). The draws are taken at the posterior mean where the
posterior separates the paths by more than one sigma, and reported as "either" where it does not.

| decision | posterior at the call (one card / split+fill) | pick | measured best | result |
|---|---|---|---|---|
| cold 2k, first start of a build (no state) | priors: decode chain 11-13 s +- 12 (U-coefficient 1, prior), stream 10 +- 10, split 7 +- 7 + 0.7 +- 0.7 | split with p ~0.6, else the stream | split 8.2 (6.7 + 1.5) vs 19-25 | a draw that takes the decode chain loses once; its 20 s residual (NIS >> 1) sets the chain's U-coefficient to ~2 and the next start never does |
| cold 2k, state file from one run | decode chain 20 +- 3, stream 7.4 +- 1, split 6.7 +- 1 + 1.4 +- 0.3; 10k: 17.3 +- 1 / 12.1 +- 0.7 | request: g_2k = +0.7, g_10k = +5.2, sum > fill: split both | split both (6.7+1.4+12.1 = 20.2 vs 7.4+1.4+12.1 = 20.9 vs 7.4+17.3 = 24.7) | ok, +4.5 s per request against the logged picks |
| cold 10k after a cold 2k split | copies in place, fill 0: 17.3 +- 1 / 12.1 +- 0.7 (or priors 10 +- 10 / 7 +- 7 on a first start) | split | split 12.1 vs 17.3 | ok in every log (the logged picks lost 4 s in 9 of 12 runs) |
| warm 2k | stream 7.4 +- 1 (measured at 2034 or 1938: the same point to the regression) vs 6.7 +- 1 + fill 1.4 | g_2k = -0.7, g_10k = +5.2: split 10k only, 2k one card; or both when the 10k is split anyway and the 2k draw favours it | either (one card 7.4 vs split 6.7 pass, fill sunk on the 10k call) | ok, within noise (0.7 s) |
| warm 10k, one card measured at 10400 and the split at 10496 (rampool6 rp1) | 17.3 +- 1 / 12.1 +- 0.7 + 1.4 | split, p 0.99 | split 15.4 vs 17.4 | ok (logged: one card, 1.2x) |
| warm 10k after a fill sample of 8904 ms with pins (rampool7 rp1) | pins excluded by 7d5bf6e72 b: fill 1.3 +- 0.3; even with the pins inside, one sample against the prior 0.74 +- 0.74 gives fill ~4.8 +- 3: split 12.1 + 4.8 = 16.9 +- 3 vs 15.0 +- 1 | split (p ~0.7 with the pins inside, 0.99 without) | split 12.7 vs 18.1 | ok (logged: one card, 1.45x) |
| warm 10k after a cold split call timed with its fill (rampool8 rp0 repeat) | the split's sample is the pass without the fill (step 4 below): 12.1 +- 0.7 + 1.1 vs 17.4 | split | split 13.3 vs 17.5 | ok (logged: one card, 1.33x, row 23753) |
| every warm 10k in the table above | both measured within one sigma of the truth | split | split | ok, as logged |

36 of 36 right or within noise; the 15 logged losses are all removed by (1) the one-card path predicted from its own calibrated model instead of a
compute-only plan or an upload bound, (2) one regression per path instead of exact-size arms, (3) the fill estimated with covariance and without the
one-time pins, and not counted twice. The remaining exposure is the first start of a new build with no state file: the cold 2k pick is a draw at p ~0.6,
one loss of ~10 s possible once per build, after which the state file carries the lesson.

### 4. Implementation spec (ordered; file/function level)

Step 1. **`src/llama-dsplit-fork.cpp` near line 673 (`DSF_KEEP`/`dsf_pt`)**: add

    struct dsf_rls { int k; double th[4], P[16], s2, nis; int n; bool det() const { return n >= k + 1; } };
    static void rls_init(dsf_rls & r, int k, const double * th0, const double * floor);     // P = diag(max(th0^2, floor^2)), s2 = 0, n = 0
    static void rls_update(dsf_rls & r, const double * phi, double y);                      // the five lines of section 2; NIS > 1 -> P *= NIS, n = k + 1
    static double rls_pred(const dsf_rls & r, const double * phi, double * var);             // mu, s2 (1 + phi' P phi)
    static void rls_draw(const dsf_rls & r, const double * phi, uint64_t seed, double * t);  // one Gaussian draw of theta (Cholesky of s2 P, 4x4) applied to phi
    static std::string rls_str(const dsf_rls & r);                                           // "th0 th1 th2 th3 | Pdiag | s2 n nis" for the log and the state file

Keep `dsf_pt`/`dsf_upd` only for `g_dsf_dense` and `g_dsf_hop` (the balance still reads them); delete `g_dsf_one`, `g_dsf_split`, `g_dsf_fill`,
`g_dsf_urat`, `dsf_fit_pass`, `dsf_fit_curve`, `dsf_fit_str`, `dsf_fill_owed_us` (line 790). New state: `static dsf_rls g_rls_split; static dsf_rls g_rls_fill;`
(k = 4 and k = 1). `g_dsf_ban` goes: a slow split pass is a residual, the redraw handles it.

Step 2. **`src/llama-moecache-fork.cpp` line 11447 `g_pp_took[PP_PATHS][32]`**: replace by `static dsf_rls g_pp_rls[PP_PATHS]` (move `dsf_rls` and its
four functions into a small shared internal header `src/llama-rls-fork.h`, new file, static inline: both .cpp files include it; no public header edit).
`pp_took_us(path, n)` becomes `pp_pred_us(path, n, c, U, &var)`; `pp_scaled_us`, `pp_other_us`, `pp_bucket`, `g_pp_no_chain[32]` -> `g_pp_no_chain` keyed by
nothing (the graph refusal is a fact per ubatch: keep as a single `uint32_t` "refused at or above this ubatch"), `pp_try_ok`, `g_pp_lost_us` go. The
per-call feature U per path: `U_stream` = the call's "streaming upload" (already computed as `U`), `U_mix` = `Um`, `U_dec` = `up` (misses uploaded) + the
decode chain's chunk term, `U_chain` = `up`, `U_k` = the prediction's staged upload. Store the call's `phi` per path in `g_pp_phi_call[PP_PATHS][4]` at
decision time (line ~12173) so the report can update the path that ran with the same features.

Step 3. **`llama_moe_cache_pp_decide_fork` (line ~12105-12192)**: the path pick becomes: for each candidate path (the existing eligibility: `has_slower`
for the mix, `chain_ok`/`dec_ok`, k steps while `n_used * k < n_expert`) `t = rls_draw(g_pp_rls[p], phi_p, g_pp_seed)`; take the lowest draw. `g_pp_seed`
is set at the first call of a request (`llama_moe_prompt_begin_fork`) and after any pass whose NIS > 1. Delete `g_pp_no_dense`, `llama_moe_cache_pp_no_dense_fork`
(line 12197), `g_pp_save_us`, `llama_moe_cache_pp_saving_us_fork`, `pp_mix_pick`, `pp_k_pick`, `pp_chain_pays` (lines 11505-11570). New export
`double llama_moe_cache_pp_one_draw_fork(int64_t n, int64_t c, double * mu, double * var)`: the lowest drawn one-card time for a call of n tokens at c cells
with its posterior mean and variance (the split planner's `one_k`). The log line at 12187 prints every candidate's `mu +- sqrt(var)`, its draw and n/nis.

Step 4. **`llama_moe_cache_pp_report` (line ~11917)**: `rls_update(g_pp_rls[path], g_pp_phi_call[path], us / P)` for the path that ran (a split call
updates nothing here: `llama_dsplit_cur_fork() == 1`). The `DON` line adds `predicted mu +- sigma, nis, regret`. The cold-pass rule at 11938 stays as a
feature, not a skip: add nothing; the first pass's residual is handled by the covariance.
**`llama_dsplit_report_fork` (line 2990)** and **`dsplit_pass_end_locked` (2579)**: the split sample is `us - fill_wall_ms` where `fill_wall_ms` is the
whole `moe_mirror_fill` wall (pins included) of this call, recorded in `g_dsf_ps.fill_wall_ms` by `llama_dsplit_note_fill_fork(ms, pin_ms)` (new second
argument from `moe_mirror_fill`, `src/llama-context.cpp:1516`): fault G. Then `rls_update(g_rls_split, phi, us_pass)` with phi = (1, T, T*cm, U_split) where
U_split = `llama_moe_cache_pass_upload_ms_fork(T, k_used, n_exp) * 1e3` (already called in `dsf_decide`); `rls_update(g_rls_fill, {bytes}, ms - pin_ms)`
with bytes = the copies' bytes of this fill (`moe_mirror_fill` knows `total`; pass it as a third argument). Drop the quarter-of-card-limit skip (2591) and
the upload-bound skip (2597-2600): the model has the fixed part and the upload as features. Keep the up-wait subtraction (2603-2611). Delete the
bucket update at 3007-3046 (`B.us_tok`, `g_split_calib`, `g_calib_split_b`, `g_calib_one_b`, their saves) except the `/moe/stats` numbers the pass table shows:
keep `dsplit_bucket` only as a display of the last measured t/s per size ("this size: one card X, split Y" stays in the log), never read by a decision.

Step 5. **`dsf_decide` (2453-2564)** becomes `dsf_plan(model, n_tokens, dsf_plan_out &)`: the loop over the calls left (2531-2538) stays; per call
`one_k = llama_moe_cache_pp_one_draw_fork(T, c, &mu1, &v1)`, `split_k = rls_draw(g_rls_split, phi_k, g_dsf_seed)`, `fill = copies in place ? 0 :
rls_draw(g_rls_fill, {bytes}, g_dsf_seed)`; `g_k = one_k - split_k`; choice for THIS call = (g_this > 0 && sum of positive g_k > fill) ? 1 : 0. The
seed: new request (the `g_dsf_seen_pos0` rule at 2457) or NIS > 1 on the last pass. The log line (2558) prints per call `mu +- sigma` of both, the draws,
the fill, `p_split` (the normal CDF of the mean gain over the combined sigma, for the log only) and the choice. `sigma`, `k_sig`, `spread`, `margin`,
`covered`, `one_pass`, `split_pass` go.

Step 6. **`llama_dsplit_decide_fork` (2731-2820)**: replace the cascade (`B.n` branches, `plan_sure`, `explore`, `split_min_gain`, the nearest-size search,
the `fc` override, the `chain_takes` bridge with `g_pp_no_dense`) by `choice = dsf_plan(...)`, `why = its log line`. Keep: `dsplit_opt_one/auto` forced
modes, `dsplit_link_model`, `dsplit_calib_load` (the section key), `g_veto_tokens`, the pass timing start (2835-2842). `llama_dsplit_plan_get_fork` is still
called for the balance's shares (it is not a decision input any more): keep the call, drop the `plan.t_one_ms` rescaling at 2742-2746.
`llama_moe_cache_pp_yield_fork` stays as the mechanism by which the split takes the pass from a one-card path, called when `choice == 1`.

Step 7. **State file** (`dsf_load` 766, `dsf_flush` 759, `dsplit_bucket_save` 862): keys `dsf.rls.split`, `dsf.rls.fill` (dsplit side) and
`pp.rls.<path>` (moe-cache side, same section/key helper via the existing `moe_state_set/get`) holding `rls_str`; written at the end of a call (the
existing `dsf_flush` point) only when `det() && nis <= 1`; read once into the prior. Stop writing `dsplit_arm4_b*`, `dsf.one.k*`, `dsf.split.k*`, `dsf.fill`,
`dsplit_calib5_b*`, `dsplit_calib7_b*`, `dsplit_calib4`.

Step 8. **Switch** (feedback_disable_means_disable): `--moe split-model=0` keeps the old cascade compiled and selectable for one release (the old
functions under `#if` is not wanted: an `if (!split_model_on())` at the top of step 6's block and step 3's pick). Default on. Removed in the release after.

Step 9. **Stats**: `stats.log` lines `PLN` (decision) and the `DON` additions of step 4; `/moe/stats` fields `plan_picks`, `plan_hits` (the chosen path's
measured time was within one predicted sigma of the lowest predicted mean), `plan_regret_s`.

Constants that remain, each from a measured quantity or the model's dimension: k = 4 features (physics of a pass), the identifiability minimum n >= k+1,
the NIS threshold 1 (the expectation of a correctly modelled residual), P_0 = diag(theta_0^2) (the logged 2x planner errors), the 64-token minimum
for a managed pass (existing: below it no split graph exists). No percentages, no sizes, no periods.

### 5. Acceptance cells (one power session, answers read first, pool unmounted unless stated)

1. GLM 3.5 12k `-nm -pl 220/280 -r 2`, rp=1 and rp=0 interleaved: pass = cold request splits the 2k call (log: `plan: ... split both`), cold pp >= 450
   (now 341-434), warm pp >= 583 on both arms (rp=0 rows 23741/23743 were 583/585 on the old builds); every `DON` line's `regret` 0 on the warm request;
   decode within noise of 15.3-16.0.
2. GLM 3.5 3k (`-pl 220/280`, the short-prompt cell): one call; pass = the pick equals the measured-faster path after two requests, pp not below the
   up9_merged2 row of the same cell; the `PLN` line shows the fill in the split's sum.
3. MiMo 12k, x16 + x4: pass = no `regret` above one predicted sigma on request 2 and 3; the 60k context band (the attention feature) picks what the
   measured pair of request 1 shows.
4. Mid-size crossover, Qwen3.8-Flash-Next (the 35B stand-in) 2k and 12k: at 2k the two paths are within noise (one card 2.2-2.5 s, split within 20%):
   pass = `p_split` between 0.3 and 0.7 on the log and the pick's `regret` within one sigma on every request of the cell; at 12k the split (measured
   1873-2008 t/s against a 1537 t/s prediction try on b14244) is picked from request 2 on with `p_split` >= 0.9.
5. rp=0 fill order A/B (fault G's sibling): rampool8 build with the `std::sort` of `moe_mirror_fill` reverted against as is, GLM 12k rp=0 warm: the split
   balance's `pass` ms and the `copies filled ... ms` line side by side; keep the order whose warm call is shorter.
6. `-simram 32g` rp=0 GLM 12k: the pins are slow there too; pass = the learned fill (`dsf.rls.fill` in the log) within 2 sigma of the rp=1 value.
7. Cold start of a new build name (no state): the first cold 2k pick is logged with `p_split`; whichever path it takes, the second start of the same
   build picks the split at 2k with `p_split` >= 0.9.
