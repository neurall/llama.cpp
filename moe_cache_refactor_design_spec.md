# MoE cache refactor: design spec (draft)

Status: design only, nothing implemented. Written 2026-10-03 from the discussion with the owner and a read of colibri (github.com/JustVugg/colibri, `c/tier.h`, `c/qwen36_tier.c`, `docs/qwen36-cuda-tier.md`).  
Rule for everything below: **the fork is never slower than stock.** Only the measurements that decide between stock and cache, and reconfirmations of the cache from a saved profile, may cost speed. Every change is an experiment: paired A/B through `tools/run.py exp`, kept only if it wins.

## 1. Principles

- **Decide by measuring.** No size thresholds. Every placement and every swap policy knob is chosen from measured token times, bandwidth and hit rates.
- **Decouple swaps from routing.** In the routing path it is already too late to swap anything for the current token, and an upload there steals compute and bandwidth at the wrong time. Swaps are decided and executed outside the per-layer path.
- **Global heat.** One heat table over all (layer, expert) pairs, not slots per layer.
- **No change of model semantics.** Placement and caching never change which experts run or at what precision. Routing-changing options (as colibri's `CACHE_ROUTE`) stay opt-in and off by default.

## 2. Global heat and layer classes

- **Heat:** one counter per (layer, expert), incremented on each routed use, halved every `D` tokens (colibri: 1024) so an old workload can be replaced. Counters saturate, they never wrap.
- **Budget:** one VRAM pool per GPU, filled with the hottest (layer, expert) pairs of the whole model. A layer's share follows from its heat, not from a fixed per-layer slot count.
- **Layer classes**, decided from the heat distribution and measured speeds:
  - *pure CPU*: a layer with flat (trashy) routing. VRAM spent there buys almost no hits. Served entirely by the CPU, no GPU partial result, so **no hybrid merge and no sync** for that layer. Often faster than a GPU/CPU hybrid whose merge costs more than it saves.
  - *pure GPU*: all experts of the layer resident.
  - *hybrid*: a skewed layer with a resident hot set, CPU computes the rest, results merged.
- **Layer value** = expected hits gained per VRAM byte (from the heat curve), compared with the cost of the merge/sync of a hybrid layer. Layers with value below the cost go pure CPU.
- **Persistence:** the heat table is saved in the state file and used to order the first fill, so a second run starts placed before the first token (colibri `HEAT_FILE`, our `moe-state.ini`).

## 3. Eviction: who, when, how

Taken from colibri, to be verified against our own data:
- **Who:** the coldest resident expert of the same GPU, searched over all layers. Residents already queued are neither victims nor candidates. Ties: first found.
- **When:** the analyzer runs every `N` tokens, once per GPU. It compares the coldest resident with the hottest non-resident (bytes available in RAM) and swaps only if `hot > cold + cold/4 + 4` (25% relative plus a small absolute margin; relative so already-hot experts need a larger lead). Colibri: `N = 16`, one swap per GPU per window.
- **How:** the victim is marked non-resident at once (its next uses take the CPU path), the upload is queued to an uploader that overwrites the victim's slot, so the pool size never changes. If the queue is full the victim is marked resident again.
- **Frequency first, recency second:** the score is heat; recency only breaks near ties (it may add at most one use's worth).
- **No swaps in the routing path.** The per-layer, per-step inserts (`up-max`) and the just-in-time upload (`upload-now`) are removed once the analyzer is at least as fast.

## 4. Scheduling: fill the gaps

The number of swaps per token is not limited by the design; it is limited by the idle time and bandwidth the uploads may use. The budget is in **bytes per second**, not swaps per step.
- **Stage 1 (colibri style):** execute swaps only in idle gaps in the code: after a token ends (output head and sampling, RAM idle), between layers where the CPU is waiting.
- **Stage 2 (better):** a monitor measures PCIe use, DDR bandwidth (the CPU's reads), CPU busy and GPU busy, and issues uploads in small chunks so they fill the bandwidth and compute gaps evenly, backing off when the CPU's reads are at the RAM limit. This generalises our `upload-wait` gate (proven: +6% on a short GLM chat).
- **Cadence and cap** are knobs: window `N`, swaps or bytes per window. Eager against lazy is decided by an experiment, not by belief (on GLM, eager swapping won; colibri is lazy).

## 5. Live status counters

Updated every token, cheap, readable from the log, the server and `run.py exp`:
- hit rate (global and per layer), misses served by CPU, resident bytes per GPU;
- uploads, evictions, uploaded MiB per token (**churn**);
- PCIe GB/s and DDR GB/s with their probed peaks as a percentage, CPU busy, GPU busy;
- queue depth, swaps in flight, swaps refused by the margin.

At exit a one-line summary `moe-summary: hit=.. uploads=.. evictions=.. up_mib=.. ddr_gbs=.. pcie_gbs=.. ddr_peak=.. pcie_peak=..` (parsed by `run.py exp`).

## 6. Multi-GPU

- Home device by expert id (`eid % n_gpus`) as the simplest striping: no duplicates, both links and both GPUs work on every token. Per-GPU budgets from free VRAM and link speed.
- Profile-driven placement (a static split from a heat file) is an alternative to compare, not a replacement.

## 7. Prediction (optional, later)

- The analyzer already swaps ahead of need from global heat. A router-lookahead predictor (colibri `PILOT`: the next layer's router on the current state, 72% top-8 recall on GLM) may feed it later.
- Prefetched experts are protected from eviction until used. Our own measurements so far: the predictor raised the hit rate but not t/s because its uploads compete with CPU reads; keep it only if the gap-filling scheduler makes it win.

## 8. Measurement protocol

- Every change is an arm in a `run.py exp` spec (`tools/experiments/<name>/spec.ini`), n at least 3, learned state off or per arm, cache state noted, paired against stock (`/p/bw/rels/stock-bed0a8566`) and against the current policy.
- Metrics: t/s (prompt and generation), hit rate, uploaded MiB per token, DDR and PCIe utilisation. Hit rate is reported, never the criterion alone.
- Keep only winners. Decision rule written before the run.

## 9. Phases

1. Engine summary line and live counters (needed to measure churn).
2. Global heat table with decay, persisted, and the first-fill order. A/B against per-layer slots.
3. Analyzer with the relative margin and a swap cap, replacing the per-step inserts. A/B on cadence and cap.
4. Idle-gap execution, then the bandwidth monitor.
5. Layer classes (pure CPU / hybrid / pure GPU) from heat and measured costs.
6. `eid % n_gpus` striping for two GPUs.
7. Optional predictor feed.

## 10. Reference measurements (this repo)

- `upload-wait` (gate 3): +6% on a short GLM chat (22.10 against 20.87 t/s).
- `swap-lead=0`: hit rate 71.2% against about 65%, +6.7% t/s on GLM chat; a high lead capped the hit rate.
- `upload-now` (JIT): +4.5% on one run, noise on a 6-round rerun. Candidate for removal.
- Predictor: hit rate up (chat 46.6 to 56.1%), t/s not; lost 3 to 10% on PC1 because uploads compete with CPU reads.
- Colibri: heat halved every 1024 ticks, swap check every 16 tokens per GPU, admission `hot > cold + cold/4 + 4`, home GPU `eid % n_gpus`.

## 11. Open questions

- How much does a flat layer gain from pure CPU against the hybrid merge on our hardware? (needs a per-layer timing probe)
- Is one global pool better than per-layer slots when the layers' skew differs a lot? (A/B in phase 2)
- Does idle-gap execution leave enough bandwidth for a useful swap rate on a 2-GPU box with an x4 link?
