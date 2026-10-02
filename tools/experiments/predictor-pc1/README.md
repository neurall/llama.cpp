# Router predictor on PC1 (2026-10-02): no speed gain once the cache is warm

Question: does prefetching the experts the router will pick (early router `--moe predict=8`, learned online predictor `--moe train=1`) raise decode speed on GLM-5.3-Flash 3.0-bit, 2 x RTX 3090
at 280 W? Protocol v2: `tools/run.py bench -t ppl -n 3` (identical routing input, ABBA, fresh state), dev build. Rows: campaign `pred-ab-ppl2`; per-window logs: `tools/bench/stats.ini`.

Means over 3 runs (stats are per 32 tokens):

| variant | t/s | hit | uploads | evictions | uploaded MiB | PCIe GB/s | DDR GB/s |
|---|---|---|---|---|---|---|---|
| predictor off (default) | 20.46 | 76.2% | 154 | 129 | 1,328 | 0.97 | 18.65 |
| early router (predict=8) | 19.70 | 76.6% | 298 | 253 | 2,574 | 1.85 | 18.58 |
| learned (train=1) | 18.36 | 76.9% | 334 | 287 | 2,899 | 1.90 | 16.92 |
| learned, stream_m=4 | 18.49 | 77.0% | 322 | 274 | 2,854 | 1.90 | 16.80 |
| learned, stream_m=2 | 18.47 | 76.7% | 295 | 250 | 2,623 | 1.76 | 16.90 |
| stock def4d406a | 14.53 | | | | | | |

Conclusions
- With a warm cache the predictor raises the hit rate by 0.4-0.7 points only and costs 3% (early router) to 10% (learned) of the speed.
- Uploads, evictions and PCIe traffic double. DDR stays at 17-19 GB/s, far below the ~44 GB/s budget: the loss is not DDR starvation. Cutting the candidates from 12 to 4 or 2 removes
  only 4-12% of the uploads and none of the loss, so the learned predictor's own graph work and upload copies are the cost.
- The earlier cold-start chat test (hit 46.6% -> 56.1%, t/s 14.21 -> 13.94) is consistent: prediction fills a cold cache faster but does not pay for itself in speed.

Offline (tools/sim, GLM dumps, 1,790 tokens, top-8 overlap with the real set): early router 0.630 / 0.526 / 0.462 / 0.413 and learned online (NLMS) 0.725 / 0.649 / 0.603 / 0.568 at 1 / 2 / 3 / 4 layers
ahead; routing history alone reaches 0.38. Recall of experts that are NOT cached (34% LRU cache, best 2 candidates one layer ahead): 56% at 59% precision; the default 12 candidates waste ~80% of uploads.
The laptop (RTX 4060) has no recorded predictor win: its Qwen3.6-35B gain is the AVX Q2_0 CPU kernels. Laptop runs are in `pc3-pred-*` campaigns once rerun cleanly.
