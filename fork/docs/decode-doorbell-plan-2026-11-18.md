# Decode: the host out of the per-layer loop (plan, 2026-11-18)

User, 2026-11-18: learn the principle behind Strata's 2x decode, build our own variant (inspired, not copied; its source is MIT, current source
extracted to `/p/bw/strata-head`), lowest-hanging fruit first.

## Measured

Qwen3.8-Flash-Next IQ3_S, 2 x RTX 3090 (both x16), 13,786-token prompt, 1,024 tokens out, no drafting, same machine and hour.

| engine | decode (t/s) | ms per token |
|---|---|---|
| stock b12499 | 58.2 | 17.2 |
| ours, release b15843 `rel_pool_copy_own_card` | 56.2-58.4 | 17.1-17.8 |
| ours, `host_step_one_thread` b15862 | 60.0-60.4 | 16.6 |
| Strata 0.2.15, no drafting, every expert computed | 89.5 | 11.2 |

Our token under nsys (b15843, 19.2 ms): card 0 computes 7.0 ms, card 1 computes 7.1 ms, neither computes 5.1 ms.
- 104 CUDA graph launches, 122 stream waits, 187 small copies per token. A layer is two GPU graphs and two host steps.
- Per layer: stream wait returns, about 24 us of host work, graph launch 10 us; then the next (large) graph's launch takes 22-48 us of host time.
- Kernels, per token, both cards: quantised matrix-vector 6.0 ms (374 calls), float matrix-vector 2.95 ms (476 calls), about 2,900 kernels in all
  (61 per layer). The expert chain is about 48 us of a layer, the dense half 190-360 us.
- Traces: `/p/bw/data/nsys/qni3-1137178` (graph level, CPU samples), `qni3k-1153876` (per kernel). Scripts: job tmp `prof/dec.py`.

Handshake cost on these cards (`fork/bench/doorbell.cu`), a layer with all experts on the card: stream wait and graph launch 13.7 us, flags in mapped
host memory 5.2 us. Returning 100 KB of rows through a kernel that reads mapped host memory costs 88 us against 31 us for an ordinary copy: mapped
memory is for ids and flags, rows of one or two experts at most.

## How Strata does it (principle)

- A token is one captured CUDA graph per card, launched once. Values that change per token are read by kernels from mapped pinned host memory.
- After each layer's router a kernel writes the 10 picks into mapped host memory and counts up a word. The host spins on that word, plans, computes the
  missing experts on its CPU pool and raises flags; kernels of the same graph spin on the flags. No driver call inside a token.
- Its expert arithmetic is llama.cpp's quantised dot products, read in place.
- Tail skip (misses ranked 7th or lower dropped) is a quality trade, measured at 0-5% speed on this machine. Not adopted.

## Our variant

Decode only (one token per call), a plain cache layer; every other layout keeps today's path. Switch `--moe bell=1`, off until the rows say it wins;
then a planner pick.

Two ops, generic names, CUDA first, CPU reference, Vulkan port after (task #358 rule):

1. `ggml_moe_bell(ids, table, x)` -> slot ids (what `get_rows(dev_table, ids)` gives today). Side effect: writes the picks and their slots into the
   layer's bell block in mapped host memory, the layer input too when a pick is not on the card, then counts the block's `seq` up.
2. `ggml_moe_rows(down, slots)` -> `down` in place. When every pick is on the card it does nothing and waits for nobody. Otherwise it spins until the
   block's `flag` reaches `seq` and copies the host's rows over the rows of the missing picks.

The card decides hit or miss from its own table and publishes the verdict; the host computes exactly the picks the card called missing. So the two can
never disagree, whatever the timing of a table update.

Host side, on the thread that launched the card's graph (the scheduler calls a hook after the launch, before its wait): for each bell of that graph, in
order: spin on `seq`; record the picks (observation callback, counts); if a pick is missing, run that layer's small CPU graph (the same `mul_mat_id`
chain as today, picks on the card masked out) into the block's rows and raise `flag`.

With no host op left in the layer the scheduler makes one split per card, and the backend's graph capture makes it one CUDA graph per card per token.

Cache changes (uploads into blocks, evictions, table flushes) stay at the token boundary: nothing changes a block while a token's graph is in flight.
The mid-layer upload of a miss (`llama_moe_cache_jit_fork`) has no place in this path; the bell records what it would have wanted.

## Steps (each: failing test first, one commit)

| step | what | test |
|---|---|---|
| D0 done | host steps with nothing to compute plan one thread (+3.3-4% decode) | `tests/test-cpuplan-fork.cpp` |
| D1 done | the handshake as a pure header (`ggml/src/ggml-bell-fork.h`): block layout, publish, serve, wait; a CPU thread plays the card | standalone test: random misses over many tokens, the host computes exactly the published misses, an all-hit layer never waits |
| D2 done | the two ops: ggml.h/ggml.c, CPU reference, CUDA kernels | `test-backend-ops` cases |
| D3 done | graph branch `--moe bell=1`, the scheduler hook, per-layer CPU graphs | OLMoE: same greedy tokens as the default path |
| D4 done | Qwen cells against b15862 and b15843, nsys: launches and idle time per token | rows in runs.db |
| D5 | kernel side: fewer small kernels in the dense half, the two matrix-vector kernels (task #292) | per-kernel trace |

## After D3: where a token's time goes (2026-11-18 evening)

D1-D3 are in (`--moe bell=1`, build `bell_served_before_launch`). Two bugs on the way, both in the bug log: the host chains ran on scheduler memory another
card's inputs lived in; the bells were served behind the launch, and a CUDA stream queues only 1021 calls behind a kernel that waits (`fork/bench/launchblock.cu`),
so a graph's first, kernel-by-kernel call stood on Qwen (1500 kernels a card). The scheduler now serves a split's bells on a thread of its own from before the launch.

Decode, 2 x RTX 3090, 14k prompt, 1k out, same build: Qwen3.8 IQ3_S 60.2 -> 65.5 t/s, UD-IQ4_XS 56.2 -> 60.4, GLM 5.3 3.0-bit 28.9 -> 29.1 (nothing).

Qwen3.8 IQ3_S, per token, nsys with every kernel traced (it slows each engine, most where launches are many; the unprofiled token is the first row):

| ms per token | ours, default path | ours, bell | Strata 0.2.15, every expert computed |
|---|---|---|---|
| token, no profiler | 16.6 | 15.3 | 11.2 |
| token, under the profiler | 21.1 | 16.5 | 13.5 |
| some card computes | 14.0 | 13.6 | 10.9 |
| a kernel waits for the host | 0 | 0.2 | 0.3 |
| no kernel on any card | 7.1 | 2.8 | 2.3 |
| kernels | 2996 | 2996 | 1993 |
| sum of the kernels' own times | 14.2 | 13.8 | 13.0 |
| graph launches / stream waits | 99 / 115 | 15 / 18 | 2 / 2.4 |

- With the low-overhead trace (graph level) the bell token is 15.15 ms: a card runs a graph 13.49 ms, none 1.66 ms (11%), of that 1.4 ms the gap between
  tokens (sampling, graph build and allocation, the first launch) and 0.2 ms the hand-off between the cards.
- The sync delay was about 2.6 ms of the default path's 16.6 ms on this machine, not half of it. What is left to Strata is 2.7 ms of GPU work and about 1.3 ms
  of host work between tokens.
- Strata's GPU work: 2000 kernels a token, fused per step (`native_gu_swiglu_kernel`, `gr_up_multi_kernel`, `gdn_*_multi_kernel`), and branches of its one graph
  run at the same time on a card: its kernels' times add up to 13.0 ms but fill only 10.9 ms. Its matrix-vector time (9.3 ms) is no lower than ours (9.2 ms);
  the rest of our token is 4.4 ms in about 2100 small kernels (norms, adds, activations, row gathers, copies) against its 3.6 ms that overlap. Float
  matrix-vector products: ours 484 calls, 2.9 ms; Strata 122 calls, 0.6 ms plus its fused delta-net kernels.
- GLM 5.3 3.0-bit with the bell (36.4 ms a token under the profiler): a card computes 22.3 ms, a kernel waits for the host 9.4 ms (34 of 42 layers wait, median
  202 us, p90 505 us: the CPU computes about 2 experts of 8.3 MB a layer), no kernel 4.6 ms (the gap between tokens is 3.8 ms). 41 experts a token are uploaded
  and 46 evicted (452 MB a token) while 86 a token still miss: the cache churns. One link moves an expert in 0.38 ms, the CPU delivers a layer's two in about
  0.35 ms, so an upload for the layer that needs it now loses; uploads ahead of the layer (a predictor) are what the link's idle 30 GB/s could be used for.

Next, by size of the gain on Qwen: the small kernels of the dense half fused and its independent branches on streams of their own (D5, task #292); the gap
between tokens (graph rebuilt and reallocated every token, 15 launches where 2 would do); then uploads during a token into spare blocks (D3b) with the predictor.

## Prefill (after decode)

Strata's prompt speed barely drops with small VRAM (IQ3_S: 2,635 t/s free, 2,525 t/s at 8 GB a card; ours 2,300-2,600 and 1,344-1,655). From its
source (old checkout read; re-check in `/p/bw/strata-head`): from 1,024 tokens a chunk every expert the card lacks is streamed once per chunk in a fixed
order; the chunk buffers and the upload ring are lent from the expert cache's tail slots and refilled after the prompt; experts are multiplied quantised.
