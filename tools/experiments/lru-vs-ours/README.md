# Our eviction policy against plain LRU and other strategies (2026-10-02)

Question: is our eviction (window count + lifetime share, pay-back margin) worse than the plain LRU of the upstream MoE-cache PRs, and do strategies from the
other forks beat it? Protocol: `tools/run.py bench -t ppl -n 3` (teacher-forced decode of one fixed text, one token per call, so every cell sees identical routing
input; cells interleaved ABBA, one discarded run per cell, fresh state `LLAMA_MOE_STATE=0`), dev build, GLM-5.3-Flash 3.0-bit (106 GB), 2 x RTX 3090 at 280 W,
clocks unlocked. Rows: `tools/bench/run-history.csv`, campaign `lru-ab-ppl1`. Variants are engine knobs (`LLAMA_MOE_CACHE_POLICY=lru|lruk|slru|lfua`,
`LLAMA_MOE_CACHE_ADMIT=N`, `--moe margin=0`); the recency variants set `--moe big=0` so the lifetime rule does not mix in.

Stage 1 (median [min-max] t/s over 3 runs):

| variant | t/s | vs ours | vs stock |
|---|---|---|---|
| ours (default) | 18.61 [18.41-18.68] | | 1.46x |
| plain LRU | 18.71 [18.65-18.77] | +0.5%, ranges overlap | 1.47x |
| LRU-2 | 20.36 [19.95-21.68] | +9.4% | 1.59x |
| SLRU | 19.66 [17.76-19.68] | +5.6%, ranges overlap | 1.54x |
| aging LFU (halve at 128) | 20.71 [18.75-20.87] | +11.3% | 1.62x |
| ours, margin 0 | 20.75 [19.07-20.77] | +11.5% | 1.63x |
| ours, admit 2 (a miss needs 2 uses in 64 tokens) | 21.62 [19.50-21.73] | +16.2% | 1.69x |
| stock def4d406a | 12.77 [12.68-14.53] | | 1.00x |

Reading: ours is level with plain LRU (no reason to revert); four variants beat ours by more than 2% with non-overlapping ranges on this one workload. Rule for adopting one
as the default: it must beat the default by more than 2% with non-overlapping ranges on at least two workloads and lose by more than 2% on none (stage 2: GLM 3.5 and
MiMo, queued). Hit rate is shown in the csv for diagnosis only.
