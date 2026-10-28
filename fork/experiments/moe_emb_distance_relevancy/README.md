# Can activation / embedding distance tell which cached MoE experts are still relevant?

**Question.** When the topic of the context changes (prose to code, code to Latin, ...), do the experts that were hot before keep
newcomers out of the cache? Can the distance between the current layer activation and what activated an expert measure that
the context has drifted, so that eviction (and admission) can be decided from it instead of from hit counters?

**Short answer.** Distance is an excellent *topic-change detector* and a strong "this expert fits the current token" signal (useful for
prefetch), but a poor *eviction* signal. What actually starves the new topic is the lifetime-share term of the default score together
with the "newcomer must beat the victim" comparison. Plain recency (LRU) or the window count, with no comparison and the lifetime share removed,
evicts the old topic fastest and gives the best hit rate on every stream tested.

## Setup

* Models: Qwen3.6-35B-A3B (256 experts, 8 per token, 40 layers; 8 layers recorded: the layer output feeding layers 5, 10, ..., 39) and OLMoE-1B-7B (64 experts, 16 layers; not trained on code).
* Logger: the router input (OLMoE, fp16 3 hex digits) or the layer output (Qwen3.6, int8 per token) plus the selected expert ids for every token, `capture/logger_patch.diff` (branch `admit-pred`; `LLAMA_MOE_ACT_F16=1`, `LLAMA_MOE_LOG_PROMPT=1` with `-b 1 -ub 1` to log prompt tokens).
* Samples (`samples/`): hand-written, no model output, no loops. Dev set (tuned on): minified JS, Python, Latin, hard English, HTML+CSS, 2048 tokens each. Held-out set (written afterwards, not looked at while tuning): JSON records, Spanish prose, SQL, a legal contract, web-server log lines, a chat dialogue, CSV sensor data, a shell script, math/physics text. Short segment sets (`samples/short/`) for per-token tests.
* Streams: all samples concatenated into ONE context and prefilled as one recording (`samples/dev_stream.txt`, `held_stream.txt`) so the activations carry the context bleed from the previous topic; the topic starts are found from the verbose token list (`stream_convert.py`).
* Captures are big (npz 300-400 MB); they stay out of git, see `DATA.md`.

## Findings (tables)

### 1. Return gaps: most experts come back fast
Share of repeat picks by gap since the previous pick of the same expert in the same layer (`sim/gaps.py`).

| capture | gap <= 8 | <= 32 | <= 64 | <= 256 | first ever |
|---|---|---|---|---|---|
| Qwen Latin | 83% | 93% | 96% | 99% | 1.1% |
| Qwen hard English | 75% | 91% | 95% | 99% | 1.3% |
| Qwen HTML | 67% | 86% | 92% | 99% | 1.5% |
| OLMoE prose | 77% | 96% | 99% | 100% | 0.4% |
| OLMoE tetris | 82% | 98% | 99% | 100% | 0.4% |

Experts with at most one earlier pick in the last 64 tokens are 3-17% of repeat picks; about half of those return after more than 64 tokens.

### 2. One token against stored activations (clean single-topic recordings)
AUC of "other topic vs same topic" for the distance of one token to the stored last-activation keys of one segment (OLMoE, `sim/pick_*.py`, scripts in the `short` set): Latin vs JS 0.98-1.00 in all layers, sci-fi vs history 0.64-0.97 (best layer 12: nearest key 0.88-0.90, median 0.96-0.97), C++ vs JS 0.52-0.90 (only layers 12-15 work). Raw (uncentered, unnormalized) distances are worse than centered cosine by 0.1-0.5 AUC; L1, square-root and geometric-mean aggregations are not better than cosine.
With context bleed (the same text in one context) the per-token jump at a boundary is only 1.0-1.2 times the within-topic level; averaging the last 8-10 tokens gives a 1.3-1.7 times peak exactly at the boundary in every layer.

### 3. Per-expert signature
Per expert, mean and variance (per dimension) of the activations that picked it; distance over its most consistent dimensions (`sim/sig_dist.py`, `sig_para.py`). AUC for "picked at this token" (OLMoE): signature 0.87-0.96 vs window count 0.72-0.88 vs recency 0.68-0.84. For "picked in the next 32 tokens" (what an eviction needs) it is 0.55-0.84 vs 0.84-0.96 for window/recency. Strong for prefetch, weak for eviction.

### 4. Relevance for the next H tokens (OLMoE, `sim/pick_auc.py --horizon`)
Key distance 0.64-0.81, recency 0.70-0.95, window64 0.76-0.94, window256 0.78-0.96; key distance + window is worse than the window alone. After a topic switch every single signal is weak for the first 64 tokens and recency is the most robust (Qwen3.6, 7 switch pairs, `sim/sig_switch.py`): AUC 0.58-0.76 in the first 16 tokens, recency best in 25 of 35 cells. A logistic model trained on one topic is no better than the best single signal and worse right after a switch (`sim/alive_model.py`).

### 5. Per-expert ring detector (topic-change trigger)
Each expert keeps the last R activations that picked it; at a pick, novelty = mean cosine distance of the new activation to its ring; per token the mean over the picked experts; alarm when its z-score vs the previous 256 tokens exceeds 3 (`sim/ring_novelty.py`, `ring_sweep.py`). Qwen3.6, 6 topic switches x 3 layers:

| ring size | detected within 8 tokens (of 18) | mean delay | false alarms (512 tokens before) |
|---|---|---|---|
| 4 | 14 | 1.1 | 3 |
| 8 | 14 | 1.1 | 2 |
| 10 | 15 | 1.1 | 3 |
| 16 | 14 | 1.1 | 4 |
| 32 | 13 | 1.3 | 3 |

0-3 tokens after the switch in 15 of 18 cases (the pooled 8-token window needs 3-7 tokens and gives ~2.5 false alarms per 512 tokens vs ~0.2 for the ring). Two layers, first alarm wins, covers the misses of a single layer.

### 6. Eviction replays (hit rate; `sim/*_sim.py`, `ring_replay.py`)
* Eviction by activation distance, router score of the current token, or `hits x relevancy`: no gain over LRU/window on Qwen3.6 (router score alone loses 8-12 points; hits x relevancy equals the window count when there is no gate).
* Mean over six switch pairs (Qwen3.6, 35% cache), hit % of the first 320 tokens / the next 320, uploads per 32-token window:

| policy | admit 2 | admit 0 |
|---|---|---|
| LRU | 91.0 / 93.6, 56 | 92.6 / 93.5, 105 |
| window | 85.9 / 87.7, 53 | 87.7 / 89.0, 104 |
| fork (window + 16 x lifetime share, today's default) | 86.7 / 88.5, 56 | 88.5 / 89.8, 111 |
| fork + reset/halve/decay of the lifetime counts on the ring alarm | 86.3-86.8 / 88.5-88.7 | 88.3-88.7 / 89.9-90.2 |
| LRU + admit relaxed to 0 for 64 tokens after the ring alarm | 91.9 / 93.4, 65 | |

Resetting or halving the counts on an alarm changes the hit rate by at most 0.6 points; the policy is the lever, not the counters. The window count loses 4-5 points to LRU on Qwen3.6 (many experts have 0-2 hits per window) and ties it on OLMoE.

### 7. The blocking, measured on hardware
Engine refusal counters from the baseline `switch` sessions (GLM 3.0, code and prose alternating, 512 tokens each), share of the unmet demand refused, by 32-token window after a request start:

| window | hit % | margin | budget | admit |
|---|---|---|---|---|
| 1 | 47.0 | 45% | 15% | 3% |
| 2 | 64.2 | 61% | 18% | 8% |
| 3 | 60.8 | 88% | 0% | 5% |
| 4 | 64.5 | 85% | 0% | 7% |
| 7 | 67.7 | 63% | 18% | 10% |
| 13 | 70.7 | 10% | 57% | 16% |

The margin gate (newcomer score must beat the victim's) refuses 45-88% of the unmet demand right after a topic change; admit refuses 3-10%.

### 8. Boundary replay on the concatenated dev stream (context bleed)
`sim/boundary_replay.py`, Qwen3.6, 10,445 tokens, 4 boundaries, 90 slots/layer. Mean over the boundaries, bins of 32 tokens after the boundary, no beat-the-victim gate, admit 2:

| policy | hit % bins 1/2/3/4/6/8/12 | gap vs a fresh cache, bins 4/8/12 | stale share (old topic still cached), bins 4/8/16 |
|---|---|---|---|
| LRU | 83.2 / 88.2 / 91.4 / 92.8 / 93.8 / 93.3 / 93.3 | +0.9 / +0.5 / +1.4 | 8.9 / 2.6 / 1.1 % |
| window | 83.2 / 88.0 / 91.4 / 92.8 / 93.8 / 93.3 / 93.3 | +0.9 / +0.5 / +1.4 | 9.1 / 2.6 / 1.1 % |
| fork (today) | 83.0 / 87.9 / 90.3 / 90.8 / 91.4 / 91.1 / 90.6 | -1.1 / -1.8 / -1.6 | 20.2 / 15.2 / 11.7 % |

With the gate on and admit 0, `window` has 42% of the misses refused in the first bin and trails LRU by 4.7 points; `fork` keeps 24-42% refused and stays 2.3-2.6 points below a fresh cache. Admit 0 under LRU: +6.1 points in bin 1, +3.5 in bin 2, nothing after bin 3, at 335 vs 218 uploads in bin 1 and about 2.4x the uploads in the steady bins.

Full outputs: `results/`.

### 9. Held-out stream (design frozen on the dev samples, nothing retuned)
`sim/boundary_replay.py` on the concatenation of nine unseen samples (JSON, Spanish, SQL, legal text, log lines, dialogue, CSV, shell script, math/physics), Qwen3.6, 12,990 tokens, 8 boundaries. No gate, admit 2, mean over the boundaries:

| policy | hit % bins 1/2/3/4/6/8/12 | gap vs a fresh cache, bins 4/8/12 | stale share, bins 4/8/16 |
|---|---|---|---|
| LRU | 83.2 / 88.9 / 89.7 / 90.6 / 91.6 / 91.9 / 90.7 | +2.0 / +0.9 / -0.2 | 7.5 / 0.8 / 0.5 % |
| window | 83.0 / 88.8 / 89.6 / 90.6 / 91.6 / 91.9 / 90.8 | +2.0 / +0.8 / -0.2 | 7.9 / 0.8 / 0.5 % |
| fork (today) | 81.2 / 87.5 / 88.2 / 88.4 / 88.8 / 88.4 / 87.7 | -0.2 / -3.0 / -3.6 | 22.1 / 13.3 / 8.8 % |
| LRU + ring relax (admit 0 for 64 tokens after an alarm) | 88.8 / 91.6 / 90.2 / 90.7 / 91.6 / 91.9 / 90.3 | +2.1 / +0.8 / -0.6 | 4.1 / 0.7 / 0.3 % |

On the dev stream the same relax gives 87.8 / 90.3 / 91.9 in bins 1-3 against 83.2 / 88.2 / 91.4 for plain LRU (308 vs 218 uploads in bin 1, identical uploads from bin 3 on). The held-out numbers match the dev pattern, so the result is not an artifact of tuning.

### 10. Tiering by stickiness for the slow (x4) card (`sim/tier_sim.py`)
Which experts to keep on the slow card? Rank the experts of a layer by a stickiness estimated from the first half of the stream; the slow set is the top 40%; measure the share of the second half's uploads that land on it (lower is better, random = 40%).

| ranking | dev stream | held-out stream |
|---|---|---|
| random | 40.0% | 39.5% |
| pick frequency | 44.8% | 48.1% |
| ring stability (oldest 4 vs newest 4 activations) | 43.9% | 41.4% |
| frequency + ring stability | 44.4% | 47.0% |
| past churn (fewest inserts = stickiest) | **29.8%** | **23.3%** |

The observed swap history is the right stickiness signal (25-42% fewer uploads on the slow link than random placement; a slow card with about a fifth of the fast card's bandwidth wants at most about 20% of the uploads). The activation ring says whether an expert's context is persistent, not how often it is evicted. Pick frequency is worse than random: the most-picked experts are swapped the most.

### 11. Two timescales in the ring, detector fusion, and a split by dimension
Each expert keeps 16 activations: newest 4 = short, oldest 4 = long (`sim/ring2.py`, `ring2b.py`). Qwen3.6, 6 switch pairs x 3 layers:

| detector | detected within 8 tokens (of 18) | mean delay | false-alarm tokens (512 before) | 3 layers fused (of 6) | false alarms |
|---|---|---|---|---|---|
| short (newest 4) | 16 | 0.5 | 5 | 6 | 5 |
| long (oldest 4) | 13 | 1.8 | 0 | 5 | 0 |
| AND (both within 4 tokens) | 13 | 2.0 | 0 | 5 | 0 |
| SUM (z_s + z_l) / sqrt 2 > 3.5 | 15 | 0.7 | 4 | 6 | 4 |
| single ring of 8, all dims | 15 | 1.0 | 3 | 6 | 3 |

AND equals the long ring (silent but slow, misses one pair); SUM or the single ring of 8 is the best compromise. As a relevance score the oldest 4 is poor (AUC 0.54-0.76 for "picked in the next 32 tokens") and mixing it with the newest 4 is worse than recency.
Splitting by dimension instead (`sim/ring3.py`; per expert the half of the dimensions where the newest-4 and oldest-4 means differ least = stable, the rest = drift): the stable-dimension novelty detects 18/18 switches (mean delay 0.6) but with 21 false-alarm tokens; the drift dimensions 16/18 with 8; stable + drift similarity (0.5/0.5) reaches AUC 0.95-0.96 for "picked at this token" (recency 0.91-0.92, newest 4 all dims 0.94-0.95), and nothing better than recency for "picked in the next 32 tokens" (0.68-0.86 vs 0.69-0.87).

### 12. Held-out stream, all settings (hit % bins 1 / 2 / 4 / 8)

| setting | LRU | window | fork (today) | score gate refuses, bin 1 |
|---|---|---|---|---|
| no gate, admit 2 | 83.2 / 88.9 / 90.6 / 91.9 | 83.0 / 88.8 / 90.6 / 91.9 | 81.2 / 87.5 / 88.4 / 88.4 | 0% |
| gate on, admit 2 | 83.2 / 88.9 / 90.6 / 91.9 | 81.9 / 88.6 / 90.6 / 91.9 | 79.5 / 86.9 / 88.1 / 87.9 | window 8%, fork 26% |
| no gate, admit 0 | 89.0 / 92.2 / 91.5 / 90.1 | 86.6 / 91.3 / 91.8 / 91.5 | 82.7 / 88.1 / 89.3 / 88.9 | 0% |
| gate on, admit 0 | 89.0 / 92.2 / 91.5 / 90.1 | 82.8 / 90.1 / 91.4 / 91.6 | 79.6 / 87.2 / 88.3 / 87.8 | window 54%, fork 64% |

### 13. Learned expert predictor from the activation vectors (OLMoE, live-generated data)

Data: OLMoE-1B-7B on PC2 (`capture/gen3.sh`, `--moe log=xa`: per decoded token the ids, router probabilities and the fp16 router input of all 16 layers plus the
last layer's output, `.act` next to the `.routing` file), 600 prompts (300 prompts x 2 seeds, 50 topics x 6 styles), ~260k tokens. Validation: 8 topics never trained on
(104 prompts, 47k tokens). Trainer `sim/live_train3.py` consumed the data live while it was generated. Model: logit = a·window + b·prev + c + MLP(input), BatchNorm on
the input, 1 hidden x 1024, last layer zero-initialised. "Uncached" = not among the 22 most recently used experts of the layer (LRU 35%); 34% of the picks are uncached.
newprec = of the 2 best-ranked uncached experts per layer, the share actually picked (prefetch precision).

| input (mean over layers, %) | rec8 | newrec | newprec |
|---|---|---|---|
| router probabilities of layer 0 (64 numbers, `pred_ablate.py`) | 59.9 | | |
| **act0**: router input of layer 0, same token | **79.4** | **70.9** | **66.7** |
| next1: all 17 vectors of token t-1 | 68.1 | 56.2 | 54.3 |
| next1hl: next1 + every layer's router on it (skip kept) | 68.2 | 56.4 | 54.5 |
| next2: all vectors of t-2 and t-1 | 66.8 | 54.8 | 53.1 |
| lead1o: layer-0 input of t-1 + last-layer output of t-2 (one token of lead, `train_lead.py`) | 61.3 | 46.5 | 48.1 |
| linear priors (prev + window) | 38.3 | 4.5 | 16.8 |
| previous token's picks | 36.6 | 0 | 6.7 |

The full activation carries the context; the router probabilities do not (they only say what layer 0 wants). act0 and next1 are available at the same moment
(start of token t vs end of token t-1), so act0 wins. Per layer act0 newprec runs from 93% (layer 0) down to 54-60% (layers 5-7, 13-15).

Timeline replay (`sim/olmoe_pf_sim.py`, all 104 validation prompts concatenated, cache 22/64): at slice 0 of token t the predictor ranks every layer; uncached experts
with p >= 0.6 are queued with deadline = their layer's slice, served earliest-deadline-first from the leftover of both links; demand uploads only from free PCIe
(no debt), admit 0.

| PCIe uploads / RAM reads per token | none | oracle | **act0** | gain | prefetches used |
|---|---|---|---|---|---|
| 8 / 80 | 63.35 | 66.94 | **65.90** | +2.6 | 95% |
| 16 / 80 | 63.44 | 71.55 | **68.94** | +5.5 | 93% |
| 32 / 120 | 63.46 | 73.93 | **70.73** | +7.3 | 91% |

How the design was found (12 validation prompts, PCIe 16 / RAM 80, act0 snapshot):

| step | hit |
|---|---|
| no predictor, admit 2 | 64.30 |
| act0, top-4 per layer, deadline order, demand may run into debt | 67.87 |
| stricter admission (admit 3/5/6/8) | 67.83 / 67.73 / 67.66 / 67.56 (worse) |
| demand only every 2nd / 4th token | 68.12 / 67.75 |
| **demand only from free PCIe** (prefetch 5.4 -> 10.7 per token) | 68.53 |
| + newer model (cycle 12) | 69.01 |
| + queue only p >= 0.5 / 0.6 / 0.7 | 70.03 / **70.06** / 69.99 |
| confidence order instead of deadline order | 69.71 (worse) |
| admit 8 or every 2 with the threshold | 69.37 / 69.62 (worse) |

* Bandwidth, not precision, is the limit: even the oracle drops 25-37 requests per token. The gain scales with the PCIe headroom the evictor leaves.
* Strict admission only pays with a near-perfect predictor (oracle: admit 8 +1.0); with the real one every demand upload (an expert just used) beats a prefetch.
* Lead: lead1o (queued one token earlier) alone gives 67.4-67.8 (+4.3), but on top of act0 it adds nothing: act0 + lead1o 68.58 / 68.86 / 69.06 at lead threshold
  0.5 / 0.6 / 0.7 vs act0 alone 68.94 (PCIe 16), 70.50 vs 70.73 at PCIe 32. Its less precise requests take the bandwidth of better ones; lead time is not what binds.
* Design: act0 predictor (one 2048 -> 1024 -> 16x64 MLP per token, ~3M MACs), queue uncached p >= 0.6 at the start of the token, deadline order, both links' leftover
  only; demand uploads only from free PCIe, admit 0.

### 14. On the fly: the predictor and a learned reuse score, learned from zero while decoding

`sim/online_pred.py`: the act0 model walks the validation stream (104 prompts, 54k tokens, 8 unseen topics); token j is scored with the model as it is before
j, its picks are the free label afterwards. Update schedule: a step every token for the first 2k tokens, every 4 until 8k, then every 16 (batch 256 from the
last 2048 tokens, AdamW lr 1e-3). `--horizon 16`: a second model with the target "picked within the next 16 tokens" (label known 16 tokens later).
Convergence (`sim/curve.py`, newprec % by scored tokens from a cold start):

| learner | 0-500 | 500-1k | 1-2k | 2-4k | 4-8k | 8-16k | 16-32k | 32k+ |
|---|---|---|---|---|---|---|---|---|
| offline pretrained, frozen | 60.9 | 64.2 | 65.5 | 67.0 | 63.3 | 69.5 | 65.6 | 68.1 |
| scratch, step every token | 42.2 | 52.5 | 59.5 | 58.4 | 58.3 | 62.5 | 60.6 | 63.0 |
| scratch, every 16 | 19.3 | 17.7 | 32.9 | 48.6 | 53.4 | 62.0 | 62.0 | 65.5 |
| **scratch, schedule, lr 1e-3** | 47.0 | 53.2 | 59.6 | 57.8 | 58.2 | 62.9 | 62.7 | 65.8 |

A step every token overfits (pretrained + every token: 64.9 overall vs 66.9 frozen); every 16 is best once warm (pretrained + every 16: 69.5 in the last quarter).

Timeline replay, PCIe 16 / RAM 80, everything learned from zero on the validation stream (scored from token 1 of every prompt; the window prior
is shorter at a prompt's start):

| design | hit | uploads/token | prefetch/token |
|---|---|---|---|
| today (fork score + newcomer gate), no predictor | 63.44 | 6.5 | 0 |
| LRU without gate, no predictor | 63.75 | 15.8 | 0 |
| learned reuse eviction (victim = lowest p(reuse in 16)), no predictor | 67.23 | 15.8 | 0 |
| online act0 prefetch, fork eviction | 69.20 | 15.6 | 10.0 |
| **online act0 prefetch + learned reuse eviction** | **69.69** | 15.9 | 9.4 |
| + learned newcomer gate (p(reuse) newcomer > victim) | 69.67 | 15.8 | 9.4 |
| oracle prefetch | 71.55 | | |

By PCIe budget (all learned from zero, RAM 80 / 80 / 120):

| PCIe uploads/token | today | learned eviction | online prefetch | **both** |
|---|---|---|---|---|
| 8 | 63.35 | 65.49 | 65.80 | **66.50** (+3.2) |
| 16 | 63.44 | 67.23 | 69.20 | **69.69** (+6.3) |
| 32 | 63.46 | 68.79 | 71.39 | **71.60** (+8.1) |

Serve order of the prefetch queue (PCIe 16 / RAM 80, online act0 + learned eviction; `--order jsd` = job sequencing with deadlines as in  
SeqMoE, arXiv 2609.12978: at slice 0 the most probable jobs first, each kept only if every kept deadline still fits the capacity from now, the kept  
set served earliest deadline first; `results/online/jsd/`):

| order | p min | hit | uploads/token | prefetch/token | used | expired/token |
|---|---|---|---|---|---|---|
| earliest deadline first | 0.6 | **69.69** | 15.93 | 9.41 | 91% | 17.61 |
| job sequencing | 0.6 | **69.69** | 15.93 | 7.97 | 91% | 18.26 |
| job sequencing | 0.3 | 69.56 | 15.96 | 8.67 | 88% | 30.70 |

No gain: the probability threshold already keeps the weak candidates out, and the bandwidth the scheduler declines goes to demand uploads.

Negative: throttling demand per layer by measured hits per upload (demand earns ~2.5 hits in 16 tokens, prefetch ~3.8) loses at every setting
(kappa 0.7/1/1.5: 68.70/67.89/67.58 vs 68.92): the freed PCIe is not turned into prefetch (the predictor runs out of p >= 0.6 candidates); with p >= 0.4 still
68.52 vs 68.68. Predictor-only uploads (no demand): 62.58, demand on layer 0 only: 65.73 (demand uploads carry the "just used, back soon" signal).

### 15. On hardware: `policy=learn` in the engine (branch `learn-cache`, live soak A/B)

Engine prototype on branch `learn-cache` (worktree `/p/bw/wt-learn`, from the admit-pin release). Workload: `tools/soak.py` there, an endless  
loop: Qwen3.6-35B on PC2 writes short programs with one subtle bug (python, javascript, c++, html/css/js, rust, go, sql, bash), the model under  
test is asked through its chat endpoint to find it (768 tokens, temperature 0.7, same seed per prompt for every arm). One `llama-server` per  
arm on the same GPU, requests one at a time, arm order alternating; OLMoE-1B-7B Q4_K_M, `--cpu-moe --moe-expert-cache 22` (22 of 64 experts per  
layer in VRAM), pinned weights, decode-window metrics only (prefill streams every expert over PCIe). Paired = per prompt, arm t/s over base t/s,  
geometric mean with a 95% interval. Data: `/p/bw/data/soak/OLMoE-1B-7B-0924-Instruct-Q4_K_M/old*`.

Hardware facts found on the way:

| fact | value |
|---|---|
| GPU0 PCIe link | gen4 x4 (`nvidia-smi`: width 4 of 16): upload 0.72 ms per OLMoE expert, 4.6-5.3 GB/s |
| GPU1 PCIe link | gen4 x16: probe 23.5 GB/s, 0.16 ms per expert (all runs below) |
| DDR4 (probe) | CPU alone 41 GB/s, CPU + one upload together 44 GB/s: while an upload runs the CPU experts get about half |
| OLMoE expert | 3.8 MB (3 x 2048 x 1024, Q4_K_M) |

Runs (round 1 = the learners start cold; n = prompts):

| run | arm | t/s vs base | 95% interval | n | hit % (base) | uploads/token | CPU experts GB/s in phase (base) |
|---|---|---|---|---|---|---|---|
| x4 GPU, CPU learner, `admit=0`, no margin | learn | 0.873 | 0.834-0.914 | 6 | 78.9 (72.6) | 5.6 | |
| | learnevict | 0.901 | 0.847-0.958 | 6 | 78.5 | 4.2 | |
| x16, CPU learner, no margin, `upload-now` on | learn | 0.914 | 0.871-0.960 | 12 | 81.3 (75.2) | 7.1 | |
| | learnevict | 0.903 | 0.874-0.933 | 12 | 80.7 | 4.9 | |
| | learnwide (share 1, p >= 0.4) | 0.928 | 0.896-0.962 | 11 | | | |
| x16, CPU learner, no margin, `upload-now=0` | learn | 0.845 | 0.795-0.898 | 7 | | | |
| | learnevict | 0.794 | 0.737-0.855 | 7 | | | |
| | learnwide | 0.805 | 0.751-0.863 | 7 | | | |
| x16, CPU learner per sample, pay-back margin | learnoff (learner runs, changes nothing) | 0.905 | 0.830-0.986 | 6 | 74.0 (73.2) | 2.1 | 31.0 (38.3) |
| | addlearn (`add` + learned prefetch) | 0.861 | 0.805-0.922 | 6 | 76.0 | 5.4 | 29.6 |
| | learn2 (learned rate score + margin + prefetch) | 0.889 | 0.855-0.925 | 6 | 80.8 | 7.1 | 27.2 |
| | learn2evict | 0.876 | 0.835-0.918 | 6 | 76.7 | 3.4 | 29.5 |
| x16, CPU learner batched | learnoff | 0.893 | 0.818-0.976 | 6 | 72.1 (72.9) | 2.3 | 31.8 (38.5) |
| | addlearn | 0.952 | 0.891-1.016 | 6 | 77.2 | 6.1 | 32.7 |
| | learn2 | 0.970 | 0.891-1.057 | 6 | 79.0 | 6.4 | 33.0 |
| | learn2evict | 0.952 | 0.878-1.032 | 6 | 76.8 | 3.2 | 35.6 |
| x16, upload window cap (`upload-wait-max-us=3000`, `state=0`) | basew | 1.009 | 0.961-1.060 | 12 | | | |
| | learn2 (cap one layer) | 0.955 | 0.922-0.989 | 12 | 79.8 | 7.7 | |
| | learn2w | 0.959 | 0.914-1.006 | 12 | | | |
| x16, GPU learner, backward bug (hidden layer not trained) | learnoff | 0.981 | 0.916-1.051 | 6 | 73.4 (72.3) | 2.0 | 33.7 (34.1) |
| **x16, GPU learner** (`state=0`, 3 rounds) | learnoff | 0.984 | 0.952-1.017 | 18 | 70.6 (71.2) | 0.9 | |
| | addlearn | 1.015 | 0.986-1.044 | 18 | 80.0 | 10.4 | |
| | **learn2** | **1.057** | **1.029-1.086** | 18 | **84.1** | 10.4 | |
| | **learn2evict** | **1.042** | **1.016-1.069** | 18 | | | |

Learner cost:

| learner | training step (OLMoE sizes) | inference | runs on |
|---|---|---|---|
| CPU, per sample | 21-28 ms | 0.2-0.5 ms | a CPU core, its L3 and DDR4 |
| CPU, batched | 9.1-9.8 ms | 0.1-0.25 ms | the same |
| GPU side-car (`learn-gpu`, default) | 0.25-0.3 ms | 0.07-0.17 ms | two streams of its own on the cache GPU, ~50 MB of VRAM |

What the runs say:
* Without the pay-back margin the learned score uploads far more experts than the default, many never used; an upload reads as much DDR4
  as computing the expert once on the CPU, so the extra hit rate does not pay. The margin (score in picks per 64 tokens, as `add`) stays.  
* The CPU learner itself slows decode (`learnoff` rows): it shares cores, L3 and DDR4 with the CPU expert matmuls; batching the training
  step made it 2-3x faster but not cheaper for decode. On the GPU it costs nothing measurable.  
* Uploads already wait for the gaps between CPU expert phases (`upload-wait=3`, chunked copies); a longer wait cap changed nothing.
* With the learner on the GPU the learned cache wins: `learn2` (learned eviction score with the margin, learned prefetch) and `learn2evict`
  (the score alone) decode faster than `add` with intervals above 1; the hit rates are round 3 medians, still rising as the  
  learners warm up.  

### 16. Can the experts be made smaller? (compression measurements)

Why it matters: a slot is one expert in VRAM, so experts of half the size are twice the slots (the doubled-slots rows of section 15 are the  
speed that buys). Measured on OLMoE Q4_K_M (bartowski, built with his imatrix), GLM Q2_K and Qwen3.6 Q2_0. Scripts: `sim/shape_poc.py`,  
`sim/imat_poc.py`, `sim/imat_only.py`.

Lossless and reordering:

| measurement | result |
|---|---|
| lossless ratio of the expert data, OLMoE Q4_K / Q6_K and GLM Q2_K (zstd -19, lzma) | 1.00-1.02x |
| lossless ratio, Qwen3.6 Q2_0 | 1.10-1.46x |
| PCIe use on the x16 card during decode | OLMoE 8-13 of 23.5 GB/s, Qwen3.6 1-2 of 17.8 GB/s |
| free reordering (hidden neurons, residual dimensions): variance left after predicting a weight from its neighbour | 0.97-1.00 |
| nearest-row cosine inside an expert (random pairs 0.07-0.10) | 0.16-0.20 |
| whole-expert cosine across experts, mean / max; variance left after the mean expert | 0.007 / 0.013; 0.93-0.94 |
| weight histograms: kurtosis (normal = 3) | 3.0-3.8, last layer 8.6 |

Sorted rows (`tools/quant` of the `quant` worktree: every row sorted, its curve stored as a shape id from a 64-entry vocabulary plus offset,  
scale and tail points; OLMoE `up` tensors, numbers of that tool):

| stored | whole model | per expert tensor | bits/weight |
|---|---|---|---|
| Q4_K source | 1152 MiB | 1.125 MiB | 4.5 |
| `.vq`, K points per sorted row | 253 MiB (4.55x) | 247 KiB | 0.99 |
| `.sd` SDF3, shape id + offset + scale + tail points | 115.84 MiB (9.94x) | 113 KiB | 0.45 |
| `.pm`, the order of each row (needed to run the model) | 2816 MiB (2.44x the source) | 2.75 MiB | 11 |

The sorted values are the row's histogram and every row has nearly the same one; the information is the order. Ways to avoid storing it:

| way | measurement | result |
|---|---|---|
| store only a range of the curve per weight | 4 / 8 / 16 unequal ranges: bits/weight, rel. RMSE | 2.02, 0.345 / 3.03, 0.188 / 4.06, 0.107 |
| | equal ranges with the SDF3 curve, 4 / 8 / 16: bits/weight, rel. RMSE (range mean) | 2.45, 0.377 / 3.45, 0.240 / 4.45, 0.156 |
| | GGUF quantizers on the same rows: IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS | 2.06, 0.349 / 2.56, 0.254 / 3.06, 0.185 / 3.44, 0.139 / 4.25, 0.071 |
| one order for all rows: reorder or rotate the residual basis upstream | ideal bits/weight saved by the best shared rotation, per layer and tensor kind | 0.03-0.12 |
| | same with one basis for the whole model (what the residual stream allows) | 0.00-0.04 |
| | energy in the top 64 of 2048 directions (flat: 3%) | 7-14% |
| | `mag-permute-error` of that worktree: error change after a shared column reorder | -3.2% to +0.9% |
| learn the order (`sort_test.py` of the `undim` worktree) | holdout run: train / held-out rel. RMSE | 0.0033 / 1.2085 |
| | one expert (1024 rows), 9M-parameter net: train rel. RMSE, exact ranks | 0.55, 0.18% |

The imatrix (bartowski's, 128 entries): one value per input column of a tensor, per expert for the expert tensors (2048 per expert for up  
and gate, which share their input; 1024 per expert for down, one per hidden neuron). It holds nothing per weight.

| tensor (what a column is) | energy in the lowest 50% / 75% / 90% of columns | columns that hold 50% / 90% of the energy | ideal bits/weight saved by a bit count per column | columns that can go at 1% / 5% / 10% / 25% energy lost |
|---|---|---|---|---|
| expert up and gate (residual dimension) | 41% / 63% / 78% | 39% / 87% | 0.06 | 1% / 7% / 13% / 32% |
| expert down (hidden neuron, the axis `llmsort` cuts) | 31% / 54% / 73% | 29% / 80% | 0.11 | 3% / 11% / 20% / 43% |
| attention q, k, v | 26% / 41% / 52% | 15% / 75% | 0.35 | |
| flat (every column equal) | 50% / 75% / 90% | 50% / 90% | 0 | |

Per weight (column energy times w squared, 2048 rows of up/gate): the rank correlation of abs(w) with the column energy is -0.07 to 0.00, and  
the lowest weights hold the same share of the energy that reaches the output as of the plain energy:

| lowest weights of each row | plain energy | energy that reaches the output | same, ranked by abs(w) x sqrt(column energy) |
|---|---|---|---|
| 50% | 7.1% | 7.2% | 6.6% |
| 75% | 27.4% | 27.5% | 25.6% |
| 90% | 55.7% | 55.2% | 52.1% |

| scheme (imatrix-weighted error, same rows) | bits/weight | weighted rel. RMSE |
|---|---|---|
| lowest 90% set to 0, the rest at Q4_K | 0.92 | 0.722 |
| IQ1_M with the imatrix | 1.75 | 0.440 |
| lowest 90% keep only the sign | 1.82 | 0.409 |
| lowest 75% set to 0 | 1.94 | 0.506 |
| IQ2_XXS with the imatrix | 2.06 | 0.338 |
| IQ2_S with the imatrix | 2.56 | 0.246 |
| lowest 75% keep only the sign | 2.69 | 0.275 |
| IQ3_XXS with the imatrix | 3.06 | 0.178 |
| lowest 50% set to 0 | 3.25 | 0.256 |
| lowest 50% shuffled among themselves (values kept, order dropped) | 3.25 | 0.379 |
| IQ3_S with the imatrix | 3.44 | 0.135 |

What this says:
* The sorted curve is small because it is the same for every row; the order carries the information and is not shared between rows,
  experts or layers, not learnable, and not removable by any upstream reorder or rotation.  
* A range id per weight is a scalar quantizer: equal to IQ2_XXS / IQ3_XXS at the same bits, behind at 4 bits.
* The imatrix is nearly flat for the expert inputs and does not single out the small weights; dropping them or keeping only their sign
  lands on the curve of the IQ quantizers, not below it.  
* The usable form of "smaller experts" is a lower-bit quant of the experts (existing types, GPU kernels exist), or cutting hidden neurons
  along the `llmsort` axis (about a fifth of the neurons at a tenth of the activation energy).  

### 17. Regression hunt, 8 Oct 2026 (builds b11214 ... b12364, branch `learn-cache`)

Quick cell: OLMoE Q4_K_M, chat endpoint, 500 tokens, temperature 0, GPU1 (x16); "19 slots" = `--cpu-moe --moe-expert-cache 19` (30% of the experts in VRAM). PC2 = Ryzen 5 3600, CPU only.

| cell | b11214 | b11399 | b11509 | b11707 | b12040-b12209 | b12316 | b12353 | branch |
|---|---|---|---|---|---|---|---|---|
| PC1 19 slots, default settings, t/s | 152.0 | 165.7 | 168.9 | 161.1 | | | 158.6 | 163.5 |
| PC1 19 slots, tuning fixed, t/s | 150.7 | 168.4 | 170.4 | 154.5 (1 run) | | 151.1 (1 run) | 152.4 (1 run) | |
| PC1 whole model on GPU (cache off), t/s | 508.5 | 517.0 | 516.9 | 513.8 | | 503.5 | 503.9 | |
| PC1 cache off, `--fork off` | | | | | | 503.8 | 504.3 | |
| PC2 CPU only, tuning fixed, t/s | 42.3 | 42.6 | 43.2 | 42.3 | 45.8 | 45.8 | 45.8 | 45.85 |

| PC2 thread count (b12353, CPU only) | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| t/s | 45.5 | 45.4 | 45.7 | 44.3 | 43.6 | 42.3 |

What it says:
* The cache path lost 4.6% (default settings) to 9% (tuning fixed) between b11509 and b11707, so it is code: the 30 Sep upstream merge replaced our GLM code, dropped the fork's batched IQ `mul_mat` and added upstream tiled k-quant CPU kernels.
* The shared path lost another 2% after b11707 (cache off and `--fork off` give the same number): upstream merges or common code, not the cache.
* The CPU path gained 6-8% from b12040 (b11707 is 1.5% below b11509 there).
* The thread autotune picks the right count on PC2 (6).
* The quick cell's own spread is large (CV 2.6%, worst 6.7%): about 35 runs per build are needed to see 2%.

| learner, W10 laptop (RTX 4060 8 GB), 1024 tokens, 3 runs | add | learn | learn, fresh |
|---|---|---|---|
| Qwen3.6 | 55.5 | 51.4 | 51.0 |
| Qwen3.8 IQ1_M | 15.8 | 14.5 | 14.8 |

| old predictor build (`N:\pred`), W10 | off | predict=8 | train=1 | pred-top=8,train-every=64 |
|---|---|---|---|---|
| Qwen3.6 | 56.8 | 56.2 | 54.8 | 57.2 |
| IQ1_M | 15.0 | 14.4 | 12.9 | 13.8 |

| async host-to-GPU hand-off (same binary) | on | off |
|---|---|---|
| W10 Qwen3.6 | 55.9 | 57.3 |
| PC1 Qwen Next IQ4_XS | 50.45 | 49.88 |

| TF32 router matmul (same binary, OLMoE 19 slots, 16 runs) | median t/s | range | CV | distinct outputs |
|---|---|---|---|---|
| on | 163.7 | 145.6 - 176.0 | 4.5% | 13 |
| off (`GGML_CUDA_MMF_F32_DISABLE=1`) | 163.9 | 148.2 - 166.8 | 3.8% | 14 |

The text of a greedy run differs between runs whether TF32 is on or off, so TF32 is not the source of the run-to-run noise.

### What the embedding distance buys, summarised
* Eviction: nothing over recency / the window count.
* Topic-change alarm: 0-3 tokens after the switch (counts are near chance for the first 32 tokens). Worth +5.6 / +2.7 hit points in the first two bins when it opens admit to 0 for 64 tokens (about +0.9 averaged over the first 320 tokens), ~+0.15 to +0.6 over a session; the activation access per token is the cost to watch (the old predictor lost 3-6% to its synchronisation).
* Prefetch: the signature is the strongest "fits the current token" signal (AUC 0.87-0.96 vs 0.72-0.88 for counts); not exploited.
* The large effect comes without embeddings: dropping the lifetime share and the newcomer-must-beat-victim gate (LRU vs today's policy: +2.0 to +3.6 points in the plateau).

## Decisions (what to take into the engine)

1. Remove the lifetime-share term from the eviction score; evict the least recently used (or lowest window count) expert, **without** the "newcomer must beat the victim" comparison.
2. Keep admit 2 in steady state; open admit to 0 for ~64 tokens when the ring detector fires.
3. Ring detector (R = 8, two layers, z > 3) as the topic-change trigger; its action matters little once the score has no lifetime term.
4. Do not pursue activation / router-score / signature eviction; keep the signature idea for prefetch.
5. For a two-GPU split, place experts on the slow card by their observed swap count (fewest inserts), not by frequency or by activation similarity.
6. Hardware tests (`policy=lru`, `policy=window`, admit A/B, refusal shares at boundaries) decide; the simulator has no upload cost.

## Status

* Predictor (section 13): sim result only. Next: the same pipeline on Qwen3.6-35B-A3B (logger `log=xao`, built in /p/bw/pred-src), then the engine (predictor MLP on the GPU, queue in the cache's upload thread).

* Dev stream (section 8) and held-out stream (section 9) replays done with the design frozen; gate and admit-0 variants of the held-out stream in `results/boundary_held_*.txt`.
* Hardware A/B of `policy=lru` and the refusal shares: queued in the reset-agent worktree, not run yet.

## Reproduce

```
# captures (PC2, Qwen3.6): capture/capture_olmoe.sh, capture/q5_convert2.py, sim/stream_convert.py (see DATA.md for the paths)
python3 sim/boundary_replay.py STREAM.npz STREAM.bounds.json --policies lru,window,fork --admit 2 [--gate] --span 512
python3 sim/ring_sweep.py DIR --rings 4,8,10,16,32
python3 sim/ring_replay.py DIR --admit 2
```

## Data

Scripts, samples and results (199 files) stay in the tag `archive/fork-layout` (`git show archive/fork-layout:fork/experiments/moe_emb_distance_relevancy/...`).
