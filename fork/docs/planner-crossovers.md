# Planner crossovers: triggers from the logs, the grid, the design, the campaign, work packages (2026-11-07, tree 0678d96b44 + Opus edits in wt-up9)

Read first: fork/docs/rampool.md "Robust planner design", fork/docs/planner-rules.md (C1-C22, resolutions, invariants I1-I12), /p/bw/data/planner_dataset.md.
Lines are of wt-up9 at 0678d96b44 (MC = src/llama-moecache-fork.cpp, DS = src/llama-dsplit-fork.cpp, CTX = src/llama-context.cpp). Numbers in sections 1-3
are medians of forced-arm and auto pass rows on the x16 + x4 link from /p/bw/data/planner_dataset.csv (bins +-33%, an arm needs n >= 3; cold/warm not
recorded there; full per-flip signatures in /tmp/xo_sig/full.txt, script /tmp/xo_sig/b.py). "k" in the logs is always k x1: k = 2, 4, 8 was never run.

Hard rules: never slower than rel_b14244 on any cell, never slower than the previous best of our builds (runs.db, same cell); a wrong pick never costs
more than 5% against the measured-best forced arm from request 2 on; answers read before speeds (fork/outputs/<rowid>.txt, judge); nothing hand-tuned.

## 0. The decision (what is built, what was dropped and why)

Built: ONE cost predictor per strategy (the RLS per arm that exists at MC:12232-12270 and DS:2639-2720), the lowest predicted time wins, three additions:
(a) every arm carries its switch cost as a learned one-time term (the fill is the model: DS:2670 `g_rls_fill`), charged only when the arm is not the
request's current state and compared against the gain over the calls left; (b) the full-stream and mix arms' times are COMPUTED per call from live bytes
and the measured link, SSD and compute rates (per layer max(upload, compute), summed) and that number is the arm's feature with prior coefficient 1,
so the stream is right before any sample and ties within the uncertainty go to it (100% hit rate, no miss variance); (c) a paired ABAB trial runs
only when the two best arms lie within each other's uncertainty and the stake pays (section 4.3), every switch is verified or sure, a loss reverts
at once. Utilization of the binding resource (link, SSD, CPU, kernel share) is a FEATURE of the predictors (section 4.2), nothing else.

Dropped: a runtime grid and a transition graph with per-edge conditions (sections 2-3 exist for the campaign and the replay tests only: edges are
implicit in the predictors; maintaining 20 hand conditions is how the cascade of C1-C15 came to be); the pure hill climber (its bottleneck
classifier + neighbour map IS a hand-made transition graph, and it cannot see a non-adjacent optimum: one card -> dense split when the real winner is
mix; its two good ideas stay: verify every switch by a paired trial, remember "did not pay" per state, which the posterior does keyed by features
instead of buckets); a separate saturation trigger mechanism (a utilization near its ceiling already moves the prediction through its coefficient;
one mechanism, not two); per-level margins, vetoes, try budgets, the 15% rule (planner-rules res. 8). What the simple triggers of section 1 are for:
they are the PRIORS and computed features of the predictors, so a cold start picks right; they are not a second decision path.

## 1. Trigger point per transition, found in the logs (the centre of the plan)

| transition | observable trigger (A > B, live quantities) | fit over the logged crossovers | verdict |
|---|---|---|---|
| partial path (chain, k, CPU misses) -> full stream, FITTING model (every expert in VRAM or the pinned RAM pool) | computed full-stream time T_s = sum over layers max(bytes from the pool / link, compute) <= the partial path's measured or predicted time x (1 + its own relative noise); the stream's hit rate is 100%, its time has no miss variance | stream beat 'k x1' in 10 of 11 fitting bins (GLM3.0 4/4, MiMo 4/4, Qwen3.6 3/4: 4096 lost with n = 3; Qwen3.8 3/3 counted here, it fits at PC1's RAM); beat the chain in 4 of 5 (lost Qwen3.8 16384 1670 vs 1790 = 7%, within the bin's spread); beat the decode chain 3 of 3 | CLEAN: full stream is the default above ~1k; "close" was enough everywhere a partial path lost; the partial won by 7% once (Qwen3.8 16k) |
| the same, NON-FITTING model (experts come partly from the SSD: DS at PC1, GLM 3.5 over the pool margin, every model at the small-RAM budgets of section 1a) | T_s per source tier: VRAM-resident 0, pool bytes / link, SSD bytes / min(ssd_rate, link) + reads x per-read latency (both live-measured, contended by the pool's own reads), per layer max(sum, compute); the SSD term is the ceiling, not the link | DS: stream beat 'k x1' 3 of 4 (lost 16384 359 vs 492), lost to the chain and the decode chain at 512 (49.5 vs 67.5): the misses' SSD reads are fewer bytes than the full stream's; GLM3.5 rp1 cold 2k: decode chain 43-86 s against stream 7-11 s (the misses from the SSD serialised) | CLEAN only with the tier split: a partial path pays the SSD for its misses alone, the full stream for every non-resident expert; which is less is the computed comparison, "close" is NOT enough here (the SSD term's variance is the stream's own) |
| stream -> stream + mix (x4 helps) | the stream's binding resource is the x16 link, not the SSD: T_s(link) > compute AND SSD bytes / ssd_rate < T_s | mix beat plain in 9 of 10 bins (GLM3.0 2/2, MiMo 1/1, Qwen3.6 3/3, Qwen3.8 3/3); lost DS 16384 (130 vs 364) where every expert comes from the SSD (`-nm 0`) | CLEAN: mix whenever a second link exists and the SSD is not the ceiling; the DS loss is the SSD rule |
| upload misses -> CPU misses (decode chain) | tokens per missed expert x CPU us per token-expert < upload us per expert (two measured rates; project_no_cpu_experts_in_prefill: ~35 tokens on Qwen3.6 over x16, past 1k on DeepSeek) | 3 of 3: DS 512 wins (67.5 vs 49.5, predicted past 1k), Qwen3.6 6144/10240 loses (769 vs 4508), GLM3.5 2048 loses (105 vs 274) | CLEAN but only 3 bins; the campaign adds the band 64-2k |
| one card -> dense split | the one card is upload-bound: its predicted upload U_1 > its compute C_1 (the logs show it as the one-card pass jumping superlinearly while its cache hit falls to ~0: GLM3.0 6144 -> 8192 6.3 -> 16.1 s, hit 9 -> 1%), and the split's fill is paid within the request | 4 of 6 models: GLM3.0 at 8192 (510 -> 690), GLM3.5 at 10240 (602 -> 766), Qwen3.8 from its lowest both-arm bin 10240 (1512 -> 1705), Qwen3.6 at 6144 (4260 -> 4490, hit 14% vs 23% at 4096); NOT MiMo (flips 4k/6k/8k/12k follow cells_end 8k vs 126k, a context mix, not size) and NOT DS (split wins only at 1024, 66 -> 74.5 with helper share 0.55; one card from 3072) | PARTIAL: the trigger plus the attention term `T x a x cells` (MiMo) and the x4 share's cost (DS) = the predictor; at the split's crossover wait-for-other is 0.57-0.70 of the pass on every model (the x4 card is the wall), x4 link busy 4.4-7.3 s per pass on GLM/MiMo |
| dense split -> dense + expert split | none found: wins GLM3.0 10240 only (768 -> 864) then loses 12288/16384; Qwen3.6 from 8192 (3936 -> 4823); Qwen3.8 10240 and 16384 (+7-10%); MiMo never (907/897, 995/774) | 2 of 4 models, 5-12% gains, no quantity crosses: the x4 share's link busy and stall are the same on both sides | NO TRIGGER: predictor + one confirmation per model; off by prediction where it never paid |
| stream plain -> stream k = 2, 4, 8 | unmeasured (only k x1 in every log) | - | UNKNOWN: campaign cells |
| chain -> stream at small passes (< 1k) | no bins with n >= 3 on both arms below 1k except DS 512 | - | UNKNOWN: campaign cells (the pf64-pf512 tests) |

### 1a. Which models fit (every expert in VRAM + the pinned RAM pool), by budget

File sizes (fork/models.ini paths, GiB): olmoe 3.9, qwen36 11.4, qwen38 77.9, glm 109.4, glm35 127.7, mimo 107.7, ds 231.7. PC1: 2 x 24 GiB VRAM,
125 GiB RAM (pin share computed, feedback_pin_share_computed: about 100 GiB pinnable with Engram mapped). Fitting = file + KV + copies <= VRAM + pin.

| budget (VRAM + pin) | fitting | non-fitting (SSD tier in T_s) |
|---|---|---|
| PC1: 48 + ~100 | olmoe, qwen36 (VRAM alone), qwen38, mimo, glm 3.0 (needs the pool near its max: the rampool margin) | glm35 (128 > the margin: the rp1 regime of section 5), ds (232: always SSD, `-nm 0`) |
| 48 + 64 (`-simram 64g`) | olmoe, qwen36, qwen38 (78 < 112) | glm 3.0, glm35, mimo, ds |
| 48 + 32 | olmoe, qwen36 | qwen38 (78 > 80 less KV), glm, glm35, mimo, ds |
| 48 + 16, 48 + 8 | olmoe, qwen36 | all others |
| W10 4060 8 GiB + its RAM | olmoe (VRAM), qwen36 (RAM pool) | all others |
| `-sim 4.5g` (Qwen3.6 small-VRAM stand-in) | qwen36 still fits (RAM pool), it only stops being VRAM-resident | - |

So the non-fitting rows of section 1 are DS always, GLM 3.5 at PC1, and every big model at the simulated small-RAM budgets (user_typical_hw_small_ram):
the campaign's `-simram 32g/64g` cells (section 5, GLM 3.0 and Qwen3.8) measure the SSD-tier stream against the chain there.

Signature at the one-card -> split crossover (split rows): bal_wait_other_c0 / pass 0.57-0.70, bal_upload_c1 / pass 0.33-0.59, every expert upload over
the x16 card (bal_upload_c0 = 0), x4 link busy GLM3.0 4390 ms, GLM3.5 7280, MiMo 4680 per pass, stall 218-1330 ms. The per-card upload GB/s fields
exist in 192 rows only (utilization not computable from the dataset: a reason for the RES line of section 4.2).

## 2. The grid (campaign and replay use only; not code)

Axes that index it: pass size (the only axis the logs cover well), context at the pass (cells: MiMo's flips), link per card (only x16 + x4 is measured;
x16 + x16 and gen3 are "predicted" by the computed arms), VRAM free (Qwen3.6 at real VRAM and `-sim 4.5g`), regime (experts on the cards / pool / SSD).
Cell = the best strategy; m = measured (both arms, n >= 3), p = predicted by section 1's triggers, u = unknown (a campaign cell). Strategies: 1 = one
card stream (+mix where measured), S = dense split (stream on both), SE = dense + expert split, C = prompt chain, D = decode chain (CPU misses), K = stream k.

| model (regime) | 64-512 | 1k | 2k | 3k | 4k | 6k | 8k | 10k | 12k | 16k | context axis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM 3.0 (cards) | C/D vs 1: u | 1 p | 1 p (D measured losing: 18-25 s) | 1 m | 1 m | 1 m | S m | S m (SE m +12%, one bin) | S m | S m | 126k cells: S m at 8k |
| GLM 3.5 (cards, pool rp1) | u | u | 1 m (D loses) | 1 m | u | 1 m | 1 m | S m | S p | S p | rp1: u at every size (the release question) |
| MiMo (cards) | C vs 1: u | 1 p | u | 1 p | 1 m (8k cells) | S m (126k cells) / 1 p (8k cells) | 1 m (8k) / S p (126k) | S p | S m | S m | the axis that decides: u at 6k/8k with the context fixed; 60k-245k u |
| DS (`-nm 0`, SSD) | D m at 512 | S m (+13%) | 1 p | 1 m | 1 m | u | u | u | 1 p | 1 m (K wins 492 vs 359: u why) | SSD-bound: mix loses (m) |
| Qwen3.6 (real VRAM) | 1 p | 1 m | 1 m | 1 m | 1 m (K 4197 vs 3061 n=3: u) | S m | SE m (+22%) | S p | S m | S p | `-sim 4.5g`: C vs 1 u below 1k; D loses m |
| Qwen3.8 (cards) | u | u | u | u | u | u | S p | SE m (+7%) | S m (C 1790 vs 1670 at 16k: u) | SE m (+10%) | - |
| OLMoE (PC1 resident) | 1 p (nothing streams) | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p | 1 p (confirm u) | 1 p | W10 8 GB: C/D/1 u at 128-4k |

## 3. Transitions (edge table; the conditions are what the predictors encode, listed here for the replay test and the campaign)

| edge | condition (learned/computed quantities) | one-time switch cost | per-pass cost | verification | feasibility | never pays (model, why) |
|---|---|---|---|---|---|---|
| any partial -> full stream | T_s <= T_partial (1 + noise) | none (`in place`) | the full upload | none needed: computed, ties go here | link rate measured | OLMoE on PC1 (resident) |
| stream -> mix | x16 link is the ceiling and SSD is not | none | x4 share of the bytes | none: computed | a slower card with free slots | DS `-nm 0` (SSD-bound), OLMoE |
| stream -> chain | misses' serial upload (behind the router) + compute < T_s; small passes | the chain drops nothing; a chain pass after a stream re-uses the cache | misses serial | ABAB when within uncertainty (expected below 1k) | no uploads under way (`busy`), graph fits (`g_pp_chain_refused`) | Qwen3.6 above 1k (m), GLM above 2k (m) |
| upload misses -> CPU misses | tokens per missed expert x CPU us < upload us per expert | the dropped queued uploads (MC:12151) re-uploaded next pass | CPU time, 4-token chunks | ABAB at the band's edge | `cpu_known` | all models above ~1k (m: Qwen3.6, GLM3.5), DS above ~2k (p) |
| stream -> stream k | staged bytes hide behind compute: the link idle share x pass > the k staging's bytes / link | the slots k evicted (later layers) re-uploaded | staging | ABAB (unknown) | ranks short of all experts | unknown everywhere (never run with k > 1) |
| one card -> dense split | U_1 > C_1 and sum of gains over calls left > fill | the copy fill (GLM 1.1-1.5 s), lent blocks | wait-for-other on the x4 wall (0.6-0.7 of the pass) | ABAB across requests (copies stay in place) | two cards, `managed`, `g_veto_tokens` | OLMoE (dense part tiny), DS above 2k (m: one card) |
| dense split -> dense + expert | predictor only (no trigger found); gains 5-12% | the re-plan sync (project_alloc_forced_replan: 27% of a prompt once) + first slices | slices over the x4 link per pass | ABAB only when predicted gain > re-plan cost x calls left | x4 free VRAM for slices | MiMo (m), GLM3.0 above 10k (m); off by prediction elsewhere until a confirmation says otherwise |
| split -> one card (back) | the reverse of its edge; the copies stay (no cost back) | none | - | same trial | - | - |

## 4. The runtime design (fewest moving parts, O(1) per decision)

### 4.1 One predictor per strategy, one pick

Arms at the one-card level (MC:12248-12260): PP_RING (full stream), PP_MIX, PP_CHAIN, PP_DEC, PP_K+i; at the card level (DS:2639-2720): ONE (the lowest
one-card draw passed down), DENSE, DENSE_EXPERT, EXPERT (the last two new). Each arm: t = theta . (1, n, n a c, U_arm) by its own RLS (llama-rls-fork.h),
plus a switch RLS of k = 1 (the fill's shape, DS:2670) on the bytes or re-plan it moves when the arm is not the request's current state. Pick: lowest
draw (Thompson, one seed per request for both levels), the switch charged against the gain over the calls left (`pp_cells_get`): switch to a when
sum_k (t_stay_k - t_a_k) > s_a. 16 estimators of k <= 4 and 16 of k = 1; 32 draws per request.

Computed anchors (not learned), per source tier of every expert the pass touches (`ggml_rampool_get_fork` says VRAM / pool / SSD): U_RING = sum over
layers of max(pool bytes / link + SSD bytes / min(ssd_rate, link) + reads x ssd_latency, compute_layer) (Ts, MC:~12172, made per layer and per tier; the
SSD rate and latency live-measured by the stream's own reads, contended by the pool's); U_MIX the same with the x4 share of the pool bytes (the SSD
term does not split: section 1 row 2's DS loss); U_DEC = tokens per missed expert x cpu_us_tok_expert + chunk reads + the misses' SSD term; U_CHAIN =
the misses' serial upload + their SSD term. Their coefficient's prior is 1 with P = 1 (the design's prior rule); they learn only from residuals. Ties:
when two arms' means lie within the larger sigma, the pick is the arm with the lower variance: on a fitting model that is the full stream (100% hits,
no miss variance; section 1 row 1); on a non-fitting model the stream carries the SSD term's own variance and wins a tie only when it is the lower one
(section 1 row 1b). Fitting is a per-pass fact (the tiers of its experts), not a model flag.

### 4.2 Features that carry saturation

Per pass the RES line: utilization of each resource over the pass = bytes (or kernel us) / wall / ceiling, ceilings = running max and p90 per `@hw` key
(stored, rechecked each process; stored values policy). Resources: link per card, SSD read, CPU read GB/s, DDR (run.py's `ddr_util` per run until the
engine reads it), GPU kernel share per card, wait-for-other and stall per card (the `split balance` numbers, logged as BAL for both arms). They enter the
predictors as phi[3]'s composition (which part of U is at its ceiling) and the switch estimates; no separate trigger path.

### 4.3 The paired trial (only when it can pay)

Trigger: the two best arms' means within each other's sigma (the posterior can flip the sign) AND the predicted gain x calls left exceeds one pass's
expected loss (else `stay: trial cannot pay`). Form: A B A B on consecutive comparable passes (phi within the RLS's 10% tolerance; same request for the
one-card level, consecutive requests for the card level with the copies kept in place so the fill is paid once). Each pair d_i = t_A - t_B. Sequential
stop, no fixed N: switch when mean(d) - z sd(d) / sqrt(n) > s_a / calls_left, stay when mean(d) + z sd(d) / sqrt(n) < s_a / calls_left; z is the
Gaussian quantile for a miss probability of one pass's share of the request (2-call requests stop at ~1 sigma, 40-call ones at ~2.5). Budget: the
trial's booked regret (the losing arm's measured excess) never exceeds the predicted gain x calls left; the first pair losing more than the whole gain
reverts there. Every pass of the trial updates its arm's RLS (drift hits both within the minute: the pairing is the scheduling, the difference is
what the test reads). TRY line per pair and verdict. A NIS > 1 on the stay arm re-draws and restarts the count.

### 4.4 Logging (feedback_planner_explains_picks) and invariants

PLN `x= arm: mu sd n nis draw switch(owed|in place)` per arm + `gain G sd mode(stay|switch|trial|tie->stream)`; DON `x= arm pred took nis regret`; TRY `x=
pair i t_A t_B mean sd n verdict budget`; RES and BAL per pass; INV `x=`. Invariants I1-I12 of planner-rules.md hold; I2 extends to every switch RLS
(`INV switch twice`); new I13: a switch inside a request is preceded by a TRY verdict or by `G > 2 sd` (checked by run.py).

## 5. The campaign: cells from the unknown and uncertain grid cells only

One cell = one ABAB burst: the two forced arms alternate on one live server (run.py T7: `POST /moe/plan force=<x>:<arm>` per request), 2 pairs (cold,
warm); until T7: one server per arm, warm pair only (`seq`). The auto arm joins as a third rotation on the merged builds (`-r 6`). Prompts the planner
never saw (feedback_similar_never_same). Switch forms per arm: `--moe pp-path=stream|mix|chain|dec`, `--moe pp-k=0|1|4`, `-ds one|on`, `-a moe:prefill-split=0|0.25|1`,
regimes `-nm 0` (DS), `--moe rampool=1` (GLM3.5, MiMo), `-sim 4.5g` (Qwen3.6). Sizes below 1k need tests pf64-pf512 (T1).

| model | cells (from the u cells of section 2) | n |
|---|---|---|
| GLM 3.0 | X2 chain vs stream at 256, 512, 1k; stream vs k=4 at 512, 1k, 4k; X1 dec vs chain at 128, 512; X4 one vs on at 1k, 2k; X5 dense vs prefill-split=0.25 at 10k (confirm); `-simram 64g` (non-fitting): stream vs chain at 2k, 10k | 14 |
| GLM 3.5 | X3 stream vs chain and stream vs mix at 2k, 4k, 10k under rp0 and rp1; X4 one vs on at 4k, 12k; X1 dec vs stream at 2k cold rp1; X5 at 16k (confirm) | 10 |
| MiMo | X4 one vs on at 6k and 8k with the context fixed (8k cells and 126k cells: `-ps` ctx variants) and at 60k; X2 chain vs stream at 256, 1k; k=4 at 4k; X1 dec at 256; X5 at 16k (confirm) | 10 |
| DS (`-nm 0`) | X1 dec vs chain at 512, 1k, 2k; X4 one vs on at 1k, 2k, 8k; X3 stream vs k=4 at 16k and stream vs mix at 4k; X2 chain at 2k | 9 |
| Qwen3.6 | X4 one vs on at 4k, 6k with the context fixed; stream vs k=4 at 4k; `-sim 4.5g`: chain vs stream at 256, 1k; dec at 64; k=4 at 1k; X5 at 8k (confirm the +22%) | 9 |
| Qwen3.8 | X4 one vs on at 2k, 4k, 6k; X2 chain vs stream at 512, 1k, 16k; X5 at 10k (confirm); `-simram 32g` (non-fitting): stream vs chain at 2k, 10k | 9 |
| OLMoE | PC1: X4 at 12k (confirm one card); W10: dec vs chain at 128, 512; chain vs stream at 1k, 4k | 1 + 4 W10 |

PC1 62 cells x ~8 min (two arms x two runs + load; DS 14, MiMo 60k 20) = about 8 h of forced bursts (the full 7 x 5 x 3-4 matrix was 39 h). Auto rotation
on the merged build: the same cells at 1.5x = 12 h, two rounds (after P4/P2/P1; final). Order on PC1: GLM 3.0 -> GLM 3.5 (rp1: the release question)
-> MiMo -> Qwen3.8 -> Qwen3.6 -> DS (slow, unmounted) -> OLMoE. W10: OLMoE 4 cells (40 min). PC2: CPU rates (section 6, P9) and builds.

run.py tasks (before the cells, never while jobs run): T1 pf64/pf128/pf256/pf512 tests (first N tokens of long1k, 64 out). T2 `run.py crossover list` writes
fork/runs/xo-<model>.csv from the table above, skipping pairs runs.db holds for the model, size band, link key and build (both arms), never the auto rotation.
T3 `state_snap(run_rowid, when, zlib)` before and after each run (`-cs` cells). T4 `decisions` table from PLN/DON/TRY/RES/BAL/INV/`prefill decision`/
`prompt pass`/`decided kv/ubatch` lines, and `run.py crossover report`: per model x transition x size the best forced arm, auto's pick, loss %, pass/fail at
5%, the measured crossover (where the paired difference changes sign) against auto's switch size (its TRY verdict or `G > 2 sd`); a replay test
`run.py crossover replay` that feeds the dataset's pass rows through the predictors (llama-rls-fork.h's self-check hook) and prints the fit of section 1's
triggers (hit/miss %) per transition. T5 the line formats of 4.4; a PLN without DON is refused. T6 judge before report. T7 the arms rotation on one server.

## 6. Work packages (P0 first; P1-P5 disjoint, parallel; Sonnet runs cells)

| pkg | owner | effort | what | owns (disjoint) | depends | acceptance |
|---|---|---|---|---|---|---|
| P0 framework | opus | high | `src/llama-plan-fork.h`: `plan_arm {id, name, phi[4], mu, var, draw, n, nis, feas, why_not, sw (dsf_rls*), sw_est, sw_mu, sw_var, sw_draw, in_place}`; `plan_decide(arms, n, stay, seed, calls_left)` -> {pick, mode stay/switch/trial/tie, G, V} (4.1 pick, switch vs gain, tie -> lower variance, 4.3 trigger + budget); `plan_trial` + `plan_trial_pair(dA, dB)` -> verdict (4.3 sequential test); `plan_res` (ceilings max/p90 per `@hw`, stored, rechecked; RES line); `plan_pln/don/try/inv`; `plan_request_seed()` (replaces g_pp_seed, g_dsf_seed); `plan_state_put/get` (moves `rls_str`/`rls_parse` DS:805-840); `enum dsf_choice {ONE, DENSE, DENSE_EXPERT, EXPERT}`, `struct dsf_pick {choice, plan_arm arms[4], exp_share}`; `llama_moe_plan_force_fork(x, arm)` + `/moe/plan` endpoint (T7). Rewire MC:12262-12270 and DS:2676-2720 with no behaviour change (only the split arm has a switch RLS = `g_rls_fill`; trials off until arms carry one) | the header, the two call sites, the endpoint | - | builds; GLM 3.0 pf12k warm 2 runs: picks and PLN/DON content unchanged; rls self-check extended with a paired-trial replay (synthetic d_i: stops at the sign, never at a fixed N) |
| P1 CPU misses | opus | medium | PP_DEC: computed U_DEC (tokens per missed expert x cpu us + chunks + SSD + pool wait CTX:2657), `cpu_known` from the first decode call or PC2's stored rate, every pass a sample (res. 7), switch = the queued uploads a decode-chain pass drops (MC:12151), RES for CPU/DDR/link-in/SSD | MC:12251-12255, 978, 12210-12230, 12151, the PP_DEC branch of 11960-12009 | P0 | xo cells X1: auto within 5% of best of {dec, chain, stream} at GLM3.0 128/512, GLM3.5 2k rp1, DS 512/1k/2k, Qwen3.6 64; `stay` with no trial above the band |
| P2 chain and k | opus | medium | PP_CHAIN, PP_K+i: computed U_CHAIN (serial misses), `U_staged_i` logged against the measured staged bytes (I6), C7 sample fix (11942), `g_pp_chain_refused` as `feas:`, switch = the slots a K pass evicts (counter at the prompt `pick_slot`), RES link utilization + overlap | MC:12250, 12256-12260, 11486-11489, 11942, 9421, the evict counter | P0 | X2 cells: auto within 5% of best of {chain, k=0, k=4} at every point (GLM3.0 256/512/1k/4k, MiMo, Qwen3.8 512/1k/16k, Qwen3.6 sim); at most one trial per size band per process |
| P3 full stream and mix | opus | medium | PP_RING, PP_MIX: the per-layer computed anchor U_RING/U_MIX with the SSD term from the pool tier (`ggml_rampool_get_fork`) and the ring's stall, the tie rule (lower variance wins), `pp_mix_opt` as `feas:` with the SSD-ceiling rule, the stream sample's SSD share (11966), runs.db `rp_hit/rp_ssd/rp_refused` per decode call (RP:607), RES SSD/ring/links | MC:12248-12249, ~12172, 11966, pp_mix_*, rampool read side, the stats line | P0 | X3 cells: GLM3.5 rp0/rp1 2k/4k/10k and DS 4k/16k: auto within 5% of best of {stream, mix, chain, k=4}; a stream DON shows ssd GB/rate; the replay test's section 1 rows 1-2 fit >= the logged 13/15 and 9/10 |
| P4 dense split | opus | high | `dsf_plan` ONE/DENSE through `plan_decide` with the fill as DENSE's switch RLS (res. 1: prior t_raw_ms, no refill; I2 once per request), res. 2 regret over both levels (`llama_dsplit_split_pred_us_fork`), res. 3 `g_rls_one`, res. 7; trials across consecutive requests with copies in place; BAL line for both arms; `dsf_pick.arms[4]` with two empty slots | DS:2639-2800, 1077-1092, `g_rls_split/fill/one`, the split balance writer | P0 | X4 cells: GLM3.0 1k/2k, GLM3.5 4k/12k, MiMo 6k/8k x context + 60k, DS 1k/2k/8k, Qwen3.8 2k/4k/6k, Qwen3.6 4k/6k: auto within 5% of best of {one, on} from request 2; p_split >= 0.9 at 12k GLM; INV overrule 0; DS's split at 1k and one card at 2k found by trial or sure |
| P5 expert share | opus | high | arms DENSE_EXPERT, EXPERT at the card level (features T, T a c, slice upload us at share F, x4 wait from BAL), the probe MC:10159-10190 as their prior only, `exp_share` per request from the pick, modes DS:3329-3337 from `dsf_choice`, switch = the measured re-plan wall + first slices; expect `stay: no gain` on most models | MC:9570, 10159-10190; DS:3288-3340; `g_rls_xs[2]`, `g_rls_xs_sw[2]`; the re-plan timer | P0; P4 merged | X5 confirmations: GLM3.0 10k, Qwen3.6 8k, Qwen3.8 10k, MiMo/GLM3.5 16k: auto within 5% of best of {0, 0.25, 1, expert-only}, never slower than `-ds on prefill-split=0`; where it never pays, no trial starts and the PLN shows G < 0 |
| P6 run.py | opus | medium | T1-T7 incl. the replay test | fork/run.py, fork/judge_w10.py read side | P0 | `crossover list` emits section 5; `report` on today's rel_b14244 rows; `replay` prints section 1's fit; PLN without DON refused; ABAB cell = four runs on one server |
| P7 forced bursts | sonnet | medium | section 5's pairs on rel_b14244 and rampool9, PC1 + W10 (`seq` until T7) | fork/runs/xo-*.csv | hand csv until P6 | tables with paired differences and the measured crossover per model |
| P8 auto rounds | sonnet | medium | after each merge: the same cells with the auto rotation; 5% rule, never-slower rules, no INV, I13 | - | P7 + package | table; stop at first failure |
| P9 PC2 | sonnet | low | CPU us per token-expert and host read ceiling per model (`-ng 0`), builds of plan-pN | - | P1 branch | rows in runs.db; builds in devbuilds |

Within 5% in P8 = the warm pair's auto time against the best forced arm's warm time of the same burst. A failing cell stops the package
(feedback_stop_fix_repeat); the owning package fixes, rebuilds, P8 repeats that transition's cells.

## 7. Schedule

| when | PC1 | PC2 | W10 | agents |
|---|---|---|---|---|
| day 1 h0-2 | P7 GLM 3.0 pairs on rel_b14244 (`seq`) | P9 CPU rates (olmoe, qwen36, glm) | P7 OLMoE 4 cells | P0 (opus high) |
| day 1 h2-10 | P7 GLM 3.5 rp0/rp1, MiMo, Qwen3.8, Qwen3.6 | builds of plan-p1..p5 as they commit | judge rows (T6) | P1-P5 (5 x opus), P6 (opus) |
| day 1 night | P7 DS (2 h); P8 round 1 on the first merged build (P4, P2, P1) with the auto rotation | | | |
| day 2 | P8 round 1 continues (12 h); fixes | regression builds | P8 OLMoE auto | opus fixes |
| day 3 | P8 round 2 on the final build (P3, P5) + the gate vs rel_b14244 and the previous best | | | |

Total about 8 h forced + 2 x 12 h auto + the gate = 2.5 days wall clock, PC1 never idle; Opus work fits day 1 (P0 3 h, P1-P5 4-8 h in parallel, P6 5 h).

## Agent briefs (Opus; paste one per package)

P0 (opus, high): In /p/bw/wt-up9 (branch prerelease) add src/llama-plan-fork.h with exactly the API of fork/docs/planner-crossovers.md section 6 row P0 and
the rules of section 4 (4.1 pick with the switch cost against the gain over the calls left and the tie rule to the lower-variance arm; 4.3 trial trigger,
sequential paired test with z from the stake, budget, revert; 4.2 RES ceilings max/p90 per `@hw` key stored and rechecked; 4.4 line formats). Move
`rls_str`/`rls_parse` from src/llama-dsplit-fork.cpp:805-840 into it. Rewire src/llama-moecache-fork.cpp:12262-12270 and src/llama-dsplit-fork.cpp:2676-2720
to `plan_decide` with NO behaviour change (only the split arm has a switch RLS = `g_rls_fill`; trials stay off until arms carry one), one seed per request
for both levels (replaces g_pp_seed, g_dsf_seed; redraw on NIS > 1). Add `llama_moe_plan_force_fork(x, arm)` and the `/moe/plan` endpoint (T7). Extend the
llama-rls-fork.h self-check with a paired-trial replay on synthetic d_i (must stop at the sign, never at a fixed N). No tuned constants; comment the why.
Build with the release recipe, run GLM 3.0 pf12k warm 2 runs through run.py, diff PLN/DON content against the tree before. Branch plan-p0, push the fork,
report the API and the two call-site diffs. Do not touch the arms' feature code or dsf_plan's arms beyond the rewiring.

P1 (opus, medium): After plan-p0 merges, own only PP_DEC: src/llama-moecache-fork.cpp 12251-12255 (computed U_DEC = tokens per missed expert x CPU us per
token-expert + chunk reads + SSD term + the pool wait read at src/llama-context.cpp:2657, each logged apart), 978 `cpu_gbs_now`, 12210-12230 (`v[]`, `chunk_at`),
12151 (count the bytes of the queued uploads a decode-chain pass drops: its switch-cost sample), the PP_DEC branch of 11960-12009 (every pass a sample,
planner-rules res. 7). `cpu_known` also from the stored PC2 rate (`@hw#build`) before the first decode call. RES: CPU GB/s, DDR, link-in, SSD utilizations.
No token threshold (project_no_cpu_experts_in_prefill: the two rates decide). Acceptance: section 6 row P1; cells by Sonnet. Branch plan-p1, push, report
U_DEC's terms and the PLN/TRY/DON of a GLM 3.0 512-token call.

P2 (opus, medium): After plan-p0 merges, own PP_CHAIN and PP_K+i only: src/llama-moecache-fork.cpp 12250 (computed U_CHAIN = the misses' serial upload),
12256-12260 (`U_staged_i`, logged next to the measured staged bytes of a K pass, I6), 11486-11489, 11942 (C7), 9421, `g_pp_chain_refused` as `feas:`, and a
counter at the prompt `pick_slot` for the slots a K pass evicts (switch-cost sample). RES: link utilization and overlap = 1 - up-wait / upload. At most one
trial per size band per process, from the posterior and the budget (I8: no counter). Acceptance: section 6 row P2. Branch plan-p2, push, report the arms'
features and one PLN with the K arms and the switch column.

P3 (opus, medium): After plan-p0 merges, own PP_RING and PP_MIX only: src/llama-moecache-fork.cpp 12248-12249 and the Ts computation (~12172) made per layer:
U_RING = sum max(bytes not resident / link, compute_layer) with the SSD term for mapped or pool-missing experts (`ggml_rampool_get_fork` read side) and the
ring's stall; U_MIX with the x4 share; `pp_mix_opt` as `feas:` with the SSD-ceiling rule (section 1 row 2); the tie rule (lower variance wins); the stream
sample's SSD share (11966); runs.db `rp_hit/rp_ssd/rp_refused` per decode call from ggml-rampool-fork.cpp:607 through the stats line. Pins and the pool fill
never in a sample (I2). RES: SSD, ring, links. Regimes: `-nm 0` (DS), `--moe rampool=1` (GLM 3.5, MiMo), `-sim 4.5g` (Qwen). Acceptance: section 6 row P3.
Branch plan-p3, push, report U_RING's terms per layer and a DON + RES of a stream pass.

P4 (opus, high): After plan-p0 merges, own src/llama-dsplit-fork.cpp 2639-2800 (`dsf_plan`, the learned branch of `dsplit_pass_end_locked`), 1077-1092,
`g_rls_split/fill/one`, and the `split balance` writer (BAL line in stats.log for both arms). Apply planner-rules.md resolutions 1 (prior t_raw_ms, no refill,
the fill as DENSE's switch RLS charged once per request with the I2 counter), 2 (regret once per pass over both levels: export `llama_dsplit_split_pred_us_fork`),
3 (`g_rls_one` for every managed model), 7. Trials across consecutive requests with the copies kept in place (4.3). Build `dsf_pick.arms[4]` with ONE and DENSE,
two slots empty for P5. Keep DS:2718-2733 hygiene. Acceptance: section 6 row P4. Branch plan-p4, push, report the arm array layout and the PLN/TRY/DON of
GLM 3.0 12k requests 1-3.

P5 (opus, high): After plan-p0 and plan-p4 merge, own src/llama-moecache-fork.cpp 9570 and 10159-10190 (the probe becomes the two new arms' prior;
`--moe prefill-split=F` stays the forced switch), src/llama-dsplit-fork.cpp 3288-3340 (modes and `exp_share` from `dsf_choice`), new `g_rls_xs[2]` +
`g_rls_xs_sw[2]` into P4's `arms[2..3]`, the BAL reader for their features, and a timer on the forced re-plan (project_alloc_forced_replan) as the switch
sample. F per arm = the measured link share and the probe's pick. Section 1 says no trigger exists and the gain is 5-12% on two models only: expect
`stay: no gain` elsewhere with G < 0 in the PLN and no trial (the budget rule). Acceptance: section 6 row P5; never slower than `-ds on prefill-split=0`.
Branch plan-p5, push, report the arms' features and the PLN of a 10k call on GLM 3.0 and Qwen3.8.

P6 (opus, medium): In /p/bw/wt-up9 fork/run.py do T1-T7 of section 5 (tests pf64-pf512; `crossover list` from section 5's table with the skip rule; `state_snap`
and `decisions` tables; `crossover report` with the measured crossover vs auto's switch size; `crossover replay` of the dataset through the predictors printing
section 1's fit per transition; the line formats of 4.4 and the PLN-without-DON refusal; judge before report; the `arms` rotation on one server via `/moe/plan`).
Read `decisions()` at run.py:1609 and the stats blob writer near 925 first; keep existing columns. No cell may be running while run.py is edited.
Branch plan-p6, push, report the subcommands and one `crossover report` on today's rel_b14244 rows.

## Sonnet test briefs

P7 forced bursts (sonnet, medium): PC1, user o, run.py only. Lists fork/runs/xo-<model>.csv from section 5 (hand-written until P6: `name,model,n,test,tokens,args,extra`,
one row per arm, paired names: `glm-x4-2k-one,glm,2,pf2k1k,1024,,` with `-ds one` and `glm-x4-2k-on,...` with `-ds on`; `glm-x2-512-chain,glm,2,pf512,64,--moe pp-path=chain,`
/ `glm-x2-512-stream,glm,2,pf512,64,--moe pp-path=stream,` / `glm-x2-512-k4,glm,2,pf512,64,--moe pp-k=4,`; `q36-x1-64-dec,qwen36,2,pf64,64,--moe pp-path=dec,--hog-gpu 4.5`;
DS rows extra `-nm 0`; GLM 3.5 rp1 rows args `--moe rampool=1`; MiMo context variants through `-ps`). Commands: `run.py jobs xo-glm -b rel_b14244 -nm -pl 220/280 -r 2 -cs`
(and `-ds one` / `-ds on` as two invocations for the X4 rows), then `-b rampool9`. The two arms of a pair run back to back in one session (`seq`). Check runs.db
first (feedback_check_runs_db_before_queue), skip a pair whose both arms exist on that build at that size. One queue instance. Read every answer in
fork/outputs/<rowid>.txt before its speed; nonsense = ok 0, excluded. Stop at the first failing cell (server death, GUARD in q5h/watch-DATE.log, refused
pass, ok 0): hand back the cause, rowid, log path; do not continue past it. Return one table per transition: `model | size | arm A | arm B | build | rowids
| judge | cold pp A/B | warm pp A/B | paired diff % | vs rel_b14244 (same arm) | vs previous best (runs.db)` and per model the measured crossover size (where
the paired difference changes sign). Hand back when every pair has both arms or a named failure.

P8 auto rounds (sonnet, medium): after Opus reports "plan-pN merged, built as <build>": `run.py jobs xo-<model> -b <build> -nm -pl 220/280 -r 6 -cs` with the
`arms` column `A|B|auto` (T7; before T7: `-r 2 -onl auto`), same session power as P7. Then `run.py crossover report --build <build>` (after P6) or by hand: per
point the best forced arm's warm time, auto's warm time, loss %. Fail: loss > 5% from request 2; auto slower than rel_b14244 or the previous best on the same
cell; any INV line; a PLN without DON; a switch with neither a TRY verdict nor `G > 2 sd` (I13). Stop at the first failure; hand back rowids, the PLN/TRY/DON
of the failing call and both arms' numbers; do not re-run or change switches. Return `model | size | best arm | best warm pp | auto pick | auto warm pp | loss %
| mode (stay/switch/trial/tie) | TRY pairs | INV | ok` and auto's switch size next to P7's measured crossover. Hand back when all pass or at the first failure.

P9 PC2 (sonnet, low): `run.py jobs xo-cpu -b <plan-p1 build> -pc pc2 -ng 0 -r 2` with fork/runs/xo-cpu.csv rows `cpu-<model>,<model>,2,pf512,64,,` for olmoe,
qwen36, glm (models present on PC2 only; no downloads). Return per model the CPU us per token-expert and host read GB/s lines with the rowids; hand back
when the three rows exist or a model is missing (say which; do not fetch it). Builds: when Opus asks, build plan-pN on PC2 with the release recipe
(memory project_build_recipe_clang_cuda134) into devbuilds/<name> and report the path.
