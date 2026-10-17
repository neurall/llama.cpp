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
