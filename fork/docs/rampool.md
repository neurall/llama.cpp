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
