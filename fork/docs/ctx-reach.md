# Context reach: stock against us (README comparison)

Question the README table answers: how long a prompt can each build take on a given card, and how fast. Stock keeps every
weight, the whole KV for `-c` and the micro-batch compute buffer on the GPU, so it has a hard limit: the longest `-c` that
fits. We have no such limit (experts in RAM behind a cache, the KV grows in the expert pool, quantised-KV and RAM fallbacks
are designed in kv-fit.md), and above stock's limit stock cannot run at all.

## Static limit (calculated, then confirmed by a run)

Per card: `free = weights_on_card + kv_per_token * ctx + compute(micro_batch, ctx)`. The one-pass micro-batch is the prompt
length rounded to the pass planner's equal chunk (12k prompt: about 12.4k). `compute` and `kv_per_token` come from the
no_alloc probe (`common_auto_ctx`: two contexts at 8192 and 32768 cells give bytes per cell per card). The limit `ctx*` is
where the line meets the card's free memory; stock is tested at `0.9 ctx*` (runs), `1.1 ctx*` (out of memory expected) and
at 2x and 4x (ours only).

## Cells (per model and card size; card size is simulated with `run.py cell --hold`)

| build | `-c` | prompt | what it shows |
|---|---|---|---|
| stock | 0.9 ctx* | pf12k, pf48k clamp | stock speed where it still fits |
| ours default (no -c, growth) | none | same | our speed at the same prompt |
| ours `kv-tail=0` | none | same | the old calculated layout, for the gain of growth alone |
| stock | 1.1 ctx* | same | expect out of memory: the limit |
| ours default | none | 2x, 4x of ctx* | reach beyond stock's limit, and its speed |

Rules: same-session runs, each build on its own defaults (no `-ub`, no pins), stock built fresh from master, chat or raw
prompt as the model needs, answers checked for loops (`run.py` flags them), n >= 2 where a gap is under 5%.

## Status

2026-10-17: Qwen3.6 35B fits whole on 24 GB cards with its full 262144 context (11.4 GiB weights, 2.5 GiB KV per card), so
its limit sits on smaller cards: cells `q36-h*` (held-back memory) find it. GLM-5.3 and Qwen3.8 Next do not fit whole.

## Small cards are the headline

The comparison matters most where stock cannot start: an 8 GB, 12 GB or 16 GB card (simulated with `--hold`, the real
2080 Ti 11 GB report is task 53). Stock needs the model, the KV and the compute buffer resident, so on a small card it either
refuses to load, or loads only with a short `-c` and a small micro-batch. We load with the experts in RAM, a cache in what is
left, and a KV that starts at the horizon and grows. Card sizes to run: 5, 8, 12, 16 and 24 GB; per size the table gives
stock's maximum `-c` (or "does not load"), its speeds, and ours with no `-c`.

## Reach beyond stock's limit (the claim the table has to prove)

Columns by multiple of stock's out-of-memory context `ctx*`: 1x, 2x, 3x, 5x, 10x, each with prompt and decode speed and a
note on where the KV sits. Rungs, from kv-fit.md: f16 KV in VRAM (growth), then quantised KV (`k8v8`, `k8v4`, `k4v4`: about
2x to 4x the cells in the same bytes; K is the sensitive tensor, see the OLMoE ladder), then KV in host RAM
(`--moe kv-fallback=...,ram`; the user is told upfront which rung a context needs). Status: growth in VRAM works (GLM 125k
prompt, 133k cells, no -c); the quantised rungs run only as a fixed `-ctk/-ctv` today, and the automatic ladder and the RAM
rung are not built. The 3x and 10x columns need them.

## Qwen3.6 35B on a simulated 8 GB card (2026-10-17, 12k prompt, 64 tokens, hold 16 GiB per card)

Stock never hits an out-of-memory here: its fit lowers the offloaded layers as `-c` grows, so its speed collapses instead (stock b11639, `-c` = column 1).

| -c | stock decode t/s | stock prompt t/s | ours static -c decode | ours static -c prompt | ours growth (kv-tail) decode | ours growth prompt |
|---|---|---|---|---|---|---|
| 32k | 135.5 | 4133 | | | 140.9 (no -c) | 4310 (no -c) |
| 128k | 99.7 | 1166 | | | | |
| 256k | 75.8 | 593 | 40.4 | 1929 | 141.5 | 4387 |
| 512k | 23.0 | 346 | 8.1 | 613 | 141.9 | 4404 |

Findings: with a fixed large `-c` our build turns the expert cache off (the KV takes the card), decode is 0.53x / 0.35x of stock at 256k / 512k although prompt speed is 3.3x / 1.8x. The growth path reserves only what the prompt uses, so it stays at the unconstrained speed. Open: growth rows used a 12k prompt; a prompt of 100k+ tokens must be run to show the reach claim. Fixed `-c` must go through the same growth / quantised-KV / RAM-KV ladder (branch qkv-ram-fallback).

## Qwen3.6 35B, one card with 12 GiB free (2026-10-17, build kvg14, fair comparison)

GPU0 hidden (`CUDA_VISIBLE_DEVICES=1`), 12 GiB of GPU1 held. Stock b11639 runs with its own defaults and `-c` 500 / 1000 / 2000; ours runs with no `-c` (auto context and KV). Prompts of 400 / 900 / 1900 tokens, 64 generated.

| prompt tokens | stock decode t/s | ours decode t/s | decode gain | stock prompt t/s | ours prompt t/s | prompt gain |
|---|---|---|---|---|---|---|
| 400 | 114.0 | 119.8 | +5% | 1243 | 1431 | +15% |
| 900 | 115.3 | 118.0 | +2% | 1590 | 1830 | +15% |
| 1900 | 113.6 | 117.4 | +3% | 1846 | 2242 | +21% |

Measurement pitfalls found on the way (earlier tables in this file that used a hold on one card only, or different `-dev` / `-ub` per build, are not comparable):
- the `cell` harness starts a stock build with only `-m ... -c N`: variant args and `-dev` are ignored, so stock used both cards; hide a card with `--env CUDA_VISIBLE_DEVICES=1` for both builds;
- direct same-flag runs on one card give parity (stock 120-121, ours 120-122 t/s at 12 GiB, fork on or off);
- a saved decode thread count of 2 (tuned with all experts on the GPUs) cost 30% decode with experts on the CPU: now keyed by the place and the CPU share of the experts, and the tuner is off when no expert work runs on the CPU.

## Qwen3.6 35B, one card with 8 GiB free, fixed `-c` (2026-10-17, build kvg16)

GPU0 hidden (`CUDA_VISIBLE_DEVICES=1`), 16 GiB of GPU1 held. 12k-token prompt, 64 generated tokens. Stock b11639 at its defaults, ours at its defaults (no `-ub`). About 35% of the experts fit beside the dense weights here (model 11.4 GiB, 7.0 GiB free).

| -c | stock decode t/s | ours decode t/s | decode gain | stock prompt t/s | ours prompt t/s | prompt gain |
|---|---|---|---|---|---|---|
| 32768 | 71.5 | 100.7 | 1.41x | 964 | 3942 | 4.1x |
| 131072 | 62.4 | 100.0 | 1.60x | 776 | 3925 | 5.1x |
| 262144 | 18.8 | 100.9 | 5.4x | 644 | 3938 | 6.1x |
| none (ours auto) | n/a | 100.4 | | n/a | 3922 | |

One run per cell. Ours is flat in `-c` (the KV is claimed per request, the rest of the card keeps the experts); stock falls as the reserved KV pushes layers to the CPU. The first prompt on this card used to abort (one 13720-token pass needing 5.1 GiB of lent graph in a 4.6 GiB pool); a pass is now capped by the exact lend inequality (`pp_cap_to_pool`: lend rounded to whole blocks + KV zone + two blocks <= pool end, graph bytes linear in tokens from the two largest measured sizes). The earlier kvg15 row at 262144 (130 t/s, 2647 prompt t/s) came from the 90%-margin version and is superseded.
