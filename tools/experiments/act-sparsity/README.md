# Does subtracting a constant vector make an expert's input sparse? (2026-10-02, negative)

Idea: the residual stream has a large near-constant part; subtract a per-layer mean vector `mu` and the remainder should be sparse or low rank, so
a bandwidth-bound expert matvec could skip the weight columns of near-zero components (and the CPU, which computes cache misses, would read fewer bytes).

Method: `tools/sim/act_sparsity.py` on a router dump (`llama-eval-callback` with `ROUTER_DUMP`, GLM-5.3-Flash 3.0-bit, one 604-token prompt of text and code,
4096-wide inputs). `mu` = mean of the first half of the tokens; test tokens from the second half; the experts the router really picks (top-8) are dequantized and
one expert's output is compared with the output when only a fraction f of the input components (largest magnitude per token) is kept.

Result (relative error of one expert's output; layers 5 / 15 / 25 / 35):

| f (weight bytes read) | raw input | centred input | down-projection input |
|---|---|---|---|
| 0.75 | 0.119 / 0.115 / 0.113 / 0.127 | 0.114 / 0.108 / 0.108 / 0.117 | 0.030 / 0.031 / 0.031 / 0.036 |
| 0.50 | 0.345 / 0.337 / 0.331 / 0.360 | 0.335 / 0.315 / 0.315 / 0.341 | 0.117 / 0.121 / 0.122 / 0.136 |
| 0.35 | 0.526 / 0.518 / 0.506 / 0.547 | 0.511 / 0.485 / 0.483 / 0.512 | 0.209 / 0.215 / 0.216 / 0.238 |
| 0.25 | 0.665 / 0.657 / 0.645 / 0.685 | 0.648 / 0.618 / 0.618 / 0.641 | 0.298 / 0.303 / 0.305 / 0.332 |

The mean is only 15-28% of the input energy (`mu share`); the centred remainder needs 240-300 principal components for 90% of its variance (of at most 604).
Centring lowers the truncation error by 0.5-2 points. Gate/up inputs are not sparse (skipping 25% of the columns costs ~11% error). Only the down-projection
input tolerates it (skipping 25% costs ~3%) which saves ~8% of an expert's bytes, under ~2 ms of a ~45 ms GLM token, with a quality cost not measured.

Verdict: not a speed lever. What survives: the mean as the predictor's per-expert bias and a reduced-rank regression on the remainder (see the predictor work).
Limits: one prompt, one model, per-expert error only (no perplexity).
