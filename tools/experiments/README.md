# Experiments

One folder per question: the plan, the data behind it, the conclusions. Raw runs are in `tools/bench/run-history.csv` (campaign names), per-window logs in `tools/bench/stats.ini`, conclusions per campaign in
`tools/bench/campaigns.csv`.

| folder | question | status |
|---|---|---|
| `vram-sim` | fork against stock on simulated smaller cards (VRAM holder, clock locks) | 1 x 16 GB 1.5x, 2 x 16 GB 0.9x, 2 x 24 GB 0.78x (Qwen Next IQ1_M); GLM 3.0 on 2 x 16 GB 1.35x |
| `lru-vs-ours` | our eviction against plain LRU and other strategies | ours level with LRU; admit 2, margin 0, aging LFU, LRU-2 ahead on GLM 3.0 (mmap uploads: re-check pinned) |
| `act-sparsity` | does a constant vector make expert inputs sparse? | negative |
| `predictor-pc1` | router predictor on PC1, warm cache | loses 3-10%; hit +0.4-0.7 points |
| `swap-timing` | why the predictor lost (uploads too late), dynamic lookahead, per-link admission | in progress |
