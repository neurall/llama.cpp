# Prompt pass: which expert path wins at which size (report, 2026-10-31)

Branch `predictor-upload-bw-reducer`, PC1 (2x RTX 3090, CUDA1 x16 about 25 GB/s, CUDA0 x4 about 6 GB/s). All times are **pure prompt time** (`--moe pp-only=1`,
one token generated, no decode, no refill). "Prompt tokens" are real-text tokens as the server tokenises them (not words).

## Files

| file | what is in it |
|---|---|
| `2026-10-31_pp-sweep_qwen36-35b-a3b.csv`, `..._qwen38-next-iq4xs.csv`, `..._glm53-flash-3bit.csv` | THE sweep: 6 configs x 11 sizes (3 to 20000 prompt tokens), median of 2-7 runs after one discarded run (written while it runs; see README.md for the configs) |
| `2026-10-30_pp-by-size_qwen36-paths-3to200-tokens.csv` | earlier Qwen3.6 sweep 3-200 tokens, mean of 6 runs: upload (stage), K=8, forced cache, CPU only, everything in VRAM (floor) |
| `2026-10-31_pp-by-size_qwen36-paths-1k-10k-tokens.csv` | Qwen3.6 at 1000 and 10000 tokens (mean of 3), stage / K=8 / cache |
| `2026-10-31_pp-by-size_big-models-first-sweep-3-runs.csv` | first qni4 (Qwen3.8 Next IQ4_XS) and GLM 3.0-bit sweep, mean of 3 runs, stage / K=8 / K=32 / cache. Noisy (first run of an arm is cold); superseded by the median sweep above |
| `2026-10-31_expert-prediction_router-vs-token-heatmap_qwen36.csv` | recall of the experts a layer uses, early router (K=8/16/32, layer lag 1/3/8) against token heat maps (1-3 tokens of context) and mixes, at equal staged-expert budget |
| `2026-10-31_expert-costs_qwen36.csv` | measured costs: upload per expert per link, CPU per expert per token, GPU compute fit, experts used per layer by pass size |

## Conclusions so far

1. **CPU compute of cache misses never pays in a prompt except in tiny passes.** A missed expert gets (tokens x 8 / experts used) tokens, so CPU time grows
   with the pass; an upload costs the same at any size (35 us over x16, 147 us over x4). Qwen3.6, pass time (ms): cache path 6109 at 1000 tokens and 97895 at
   10000 against 379 and 2553 uploaded. The cache path wins only up to about 35 tokens on Qwen3.6, about 100-150 on GLM, about 200 on qni4 (first sweep).
   Rule implemented: a prompt call never computes misses on the CPU (`--moe prefill-cpu=1` restores the old rule).
2. **Decide by measured budgets, not token counts:** per missed expert, CPU time = cpu_us_per_token x tokens_on_that_expert against upload time = bytes / link
   speed. The crossover moves with the model (expert size, quant) and the CPU; the budgets do not.
3. **K=8 early router helps only in a window.** Qwen3.6: up to 250 words / 100-1000 tokens it saves 30-40% of the pass; at 1000+ tokens it costs 5-8%
   (ranking cost and late fixes with nothing left to hide). K=32 never beats K=8 in pass time although it covers more experts.
4. **Router beats token heat maps for predicting the next layer's experts.** Recall at the same staged-expert budget (CSV): router K=8 82% at 1 token (lag 1)
   against 48% for the best heat map; K=8 + heat fill 89% against router K=16 96%. At layer lag 8 the heat map holds near 47% while the router falls to 57%,
   and from 50 tokens up the heat map at the K=8 budget equals the router. A mix of 90% router + 10% heat helps only at 1-3 tokens and lag 8.
5. **Traces:** a first version of the expert-input trace was misaligned (each layer's block had one extra row at its end); fixed, own-layer router now
   reproduces the picks 100%. The router columns in any earlier heat-map table are void.

## Not done yet / open

- The full six-config sweep (upload, K=8, K=32, forced cache, dense split only, dense + expert split) is running for Qwen3.6, then qni4, then GLM.
- Dense split adds about 290 ms at 4500 MB simulated VRAM on Qwen3.6 (earlier measurement); not traced.
- The 10000-token GLM staged run had one 36 s cold first run; the median sweep discards the first run.
