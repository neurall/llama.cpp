# Swap timing: why the predictor lost on PC1, and the plan to fix lookahead and admission (2026-10-02)

Machine: PC1, 2 x RTX 3090 (GPU 1 on PCIe 4.0 x16, GPU 0 on the chipset x4 slot), Ryzen 7 3700X, 2-channel DDR4-3200, 280 W per card. Model: GLM-5.3-Flash 3.0-bit (106 GB, 8.26 MB per expert,
42 MoE layers, 288 experts, top-8). Test: `tools/run.py bench -t ppl` (identical routing input, ABBA, fresh state). The engine's own log lines are in `logs/`.

## What we measured (diagnostic runs, `LLAMA_MOE_CACHE_STATS=1`)

Upload time per expert on each link (the engine's running average under real load), token and layer time:

| weights | run | x16: ms per expert (GB/s) | x4: ms per expert (GB/s) | token ms | ms per layer | uploads/token | s/pass |
|---|---|---|---|---|---|---|---|
| mmap | predictor off | 1.82 (5.5) | 5.85 (2.2) | 42.6 | 1.01 | 4.3 | 51.12 |
| mmap | learned predictor | 1.53 (5.9) | 4.07 (2.7) | 48.0 | 1.14 | 10.3 | 55.87 |
| mmap | learned, 2 candidates, auto off | 2.06 (5.0) | 6.85 (2.2) | 47.7 | 1.14 | 8.7 | 55.00 |
| pinned | predictor off | 0.59 (16.9) | 4.04 (3.5) | 35.8 | 0.85 | 6.3 | 44.93 |
| pinned | learned predictor | 0.89 (13.7) | 4.39 (3.4) | 41.2 | 0.98 | 9.1 | 48.41 |

Predictor accuracy and timing (mmap, learned): top-8 overlap L+1 86.0%, L+2 82.4%; the best 2 candidates cover 61% of the real misses (1.84 misses per layer); of the predicted uploads
136,506 were rejected as too late to start (timing filter OFFSET), 10,786 started, 7,327 were dropped in the queue because the layer began, 3,431 were published in time, 1,489 used.
The CPU still computed 721 MB per token against 747 MB without the predictor.

Conclusions so far
1. The predictor knows the experts; the uploads do not arrive in time. A layer lasts about 1 ms; one expert takes 0.6-0.9 ms on the x16 link and about 4 ms on the x4 link, so the x16 card
   needs a lookahead of 1 layer and the x4 card 4-5 layers. The default lookahead is 2 (`PREDICT_AHEAD`), so every prediction for the x4 card's layers was rejected as too late.
2. Per-layer time grew 13% with the predictor (1.01 to 1.14 ms mmap, 0.85 to 0.98 ms pinned): gate waits +1.3 ms per token and more contention.
3. Memory bandwidth was not saturated (DDR 17-19 GB/s of ~44); the limit is the links, above all the x4 card (2.2-3.5 GB/s under load, 6 GB/s alone).
4. The `ppl` comparator loads weights by mmap (pageable uploads, 3x slower on the x16 link); the server pins them. Tests must pass `--load-mode pin` to match server use.
5. An evicted expert paid back only 14-17% of its upload cost on average (4.9-6.7 hits each): most uploads never pay off, more so on the x4 link: motivates a stricter admission there.

## Plan (status in brackets)
1. [done] Engine, dev build: `LEAD` (dynamic lookahead per link: start = ceil(upload time per expert / layer time), width `LEAD_SPAN`+1; predictor lookahead count derived from the probed link speeds),
   `ADMIT_SLOW` (admission rule of the slower link), `ADMIT` (2 by default, see lru-vs-ours), `PREDICT_FROM`, `ADMIT_JIT`; stats log columns `layer_ms_avg`, `layer_ms_min`, `up_ms_l0`, `up_ms_l1`, `pred_up`,
   `pred_pub`, `pred_used`, `pred_late` every 32 tokens (tools/bench/stats.ini) and a per-expert file at shutdown (`<LLAMA_MOE_STATSLOG>.experts`: uses, uploads, evictions, tokens cached, returns within 64 tokens).
2. [running] Campaign `lead-ab-pin`: GLM 3.0, pinned weights, 3 runs ABBA: predictor off | old lookahead 2 | dynamic lookahead | dynamic + `ADMIT_SLOW=4` | early router.
   Judge by hit rate, CPU-served misses (1 - hit), published/used/late counters; speed second.
3. [next] Pick the lookahead window and admission per link from step 2; try `ADMIT_SLOW` 4 and 6; per-expert statistics on the x4 card (which experts stay, which churn; does keeping the hottest on the x4 card pay off).
4. [next] Re-run the policy comparison (`lru-vs-ours`, stage 2: GLM 3.5 and MiMo) with pinned weights; the first comparison used mmap uploads.
5. [next] Decide defaults (admit 2, recent-use window score, BIG off, lookahead) only from protocol v2 results; commit winners only.

Results of step 2 onward are appended below.
