# Planner crossovers: triggers from the logs, the grid, the design, the campaign, work packages (2026-11-07, tree 0678d96b44 + Opus edits in wt-up9)

Read first: fork/docs/rampool.md "Robust planner design", fork/docs/planner-rules.md (C1-C22, resolutions, invariants I1-I12), /p/bw/data/planner_dataset.md.
Lines are of wt-up9 at 0678d96b44 (MC = src/llama-moecache-fork.cpp, DS = src/llama-dsplit-fork.cpp, CTX = src/llama-context.cpp). Numbers in sections 1-3
are medians of forced-arm and auto pass rows on the x16 + x4 link from /p/bw/data/planner_dataset.csv (bins +-33%, an arm needs n >= 3; cold/warm not
recorded there; full per-flip signatures in /tmp/xo_sig/full.txt, script /tmp/xo_sig/b.py). "k" in the logs is always k x1: k = 2, 4, 8 was never run.

Hard rules: never slower than rel_b14244 on any cell, never slower than the previous best of our builds (runs.db, same cell); a wrong pick never costs
more than 5% against the measured-best forced arm from request 2 on; answers read before speeds (fork/outputs/<rowid>.txt, judge); nothing hand-tuned.

## Transition tables (five models; from = rows, to = columns; measured status from /tmp/xo_score/matrix_compact.txt, rule ids from section 0)

Reading a cell: `net +X s (+Y% of the from-arm's time) over n size classes with both arms measured (c cold + w warm); wins at [sizes], losses at [sizes]; worst single
loss`. A size tag is the pass-size class, c/w = cold/warm request. `unknown` = no class has both arms in the stats.logs (runs.db-only pairs are named).
The lower triangle is the upper cell with the sign flipped (wins and losses swapped). Rule id = what decides the switch in the spec (section 0):
R1 cold start = stream (+mix with a slower link, not SSD-bound), chains only below the measured CPU crossover; R2 one card -> split by the learned
ratio r on T_one / Ts (compute-bound) and the gain over the fill; R3 = lowest measured mean at comparable features, otherwise a VOI trial; R5 = k only as a
warm trial; R6 = expert split only as a trial from the split. The dense medians exclude the fill (lower bound for its wins). `today` = the LOGGED
pick's loss against the class best on the same classes, per build era (rel_b14244 logs = rel_stable_auto*; up9_merged2; rampool9-10; others = the
2026-10-31 .. 11-06 builds).

### qwen36 (Qwen3.6-35B, 363 calls; fits VRAM, the miss paths only under `-sim 4.5g`)

| from \ to | mix | k | chain | dec | dense |
|---|---|---|---|---|---|
| stream | R1: -8.7 s (-222%) n=2c, loses 1kc,12kc, worst 6.2 s | R5: -0.2 s (-6%) n=1c (12kc) | R1/R3: -8.8 s (-97%) n=4c+1w, loses 512c..12kw, worst 4.0 s | R1: -28.5 s (-308%) n=3c+2w, loses all, worst 11.9 s | R2/R3: +0.9 s (+14%) n=2c+2w, wins 8kc,12kw, loses 1kc,1kw, worst 0.3 s |
| mix | - | R3: +6.0 s (+63%) n=1c | R3: +4.0 s (+32%) n=2c | R3: -4.6 s (-36%) n=2c, worst 5.7 s | R3: +2.3 s (+74%) n=1c |
| k | | - | R3: -3.8 s n=1c | R3: -11.7 s n=1c | R3: +3.1 s (+54%) n=1c |
| chain | | | - | R3: -18.5 s (-108%) n=3c+1w, loses all, worst 8.2 s | R3: +5.1 s (+60%) n=1c+1w |
| dec | | | | - | R3: +19.6 s (+76%) n=2c+2w, wins all |
| dense -> expert | unknown in the stats.logs (runs.db: no forced expert row on this model) | | | | |
| today | others n=344: loss 555 s (83%), worst 127.7 s (the decode-chain and chain calls of the predictor sweeps) | | | | |

### qni4 (Qwen3.8-Flash-Next IQ4_XS, 62 calls)

| from \ to | mix | k | chain | dec | dense |
|---|---|---|---|---|---|
| stream | R1: -0.5 s (-2%) n=2c+2w, wins 16kc,16kw, loses 6kc,6kw, worst 0.4 s | R5: -5.1 s (-14%) n=3c+3w, loses all six, worst 1.8 s | unknown (16k both in runs.db: chain 1790 vs stream 1670 t/s warm, 7%) | R1: -49.9 s (-266%) n=3c+1w, loses all, worst 26.8 s | R2/R3: +5.5 s (+21%) n=1c+2w, wins 12kc,12kw,16kw, no loss |
| mix | - | R3: -3.7 s (-16%) n=1c+2w, worst 1.8 s | unknown | R3: -37.8 s n=2c, worst 26.8 s | R3: +2.3 s (+24%) n=1w |
| k | | - | unknown | R3: -36.4 s n=2c+1w, worst 25.4 s | R3: +5.5 s (+27%) n=1c+1w |
| dense -> expert | unknown (no forced expert row) | | | | |
| today | others n=56: loss 198 s (56%), worst 26.8 s; rel_b14244 n=4: 1 s (6%), worst 1.2 s; up9_merged2 n=2: 1 s (11%), worst 1.5 s | | | | |

### glm (GLM-5.3-Flash 3.0bit = /30 and 3.5bit = /35, 170 calls; 3.5 non-fitting at PC1)

| from \ to | mix | k | chain | dec | dense |
|---|---|---|---|---|---|
| stream | R1: +1.4 s (+6%) n=1c (12kc/30) | R5: -0.1 s (0%) n=7c+5w, wins 512c/30,3kc/35,8kc/35, loses 9 classes, worst 0.7 s | R1/R3: -10.8 s (-49%) n=1c (12kc/30) | R1: -32.7 s (-107%) n=3c+2w, loses all, worst 13.4 s (2kw/35) | R2/R3: +15.0 s (+23%) n=2c+3w, wins 6kw/30,8kc/35,10kc/35,10kw/35, loses 2kw/35, worst 0.6 s |
| mix | - | R3: -2.6 s (-6%) n=1c+1w, worst 1.7 s | R3: -61.5 s (-153%) n=1c+1w, worst 49.4 s (12kw/30) | unknown | unknown |
| k | | - | R3: -59.0 s n=1c+1w, worst 48.5 s | R3: -44.7 s (-117%) n=4c+2w, loses all, worst 14.2 s | R3: +10.4 s (+9%) n=4c+4w, wins 2kc/35,8kc/35,10kc/35,10kw/35,16kc/30, loses 2kw/35,3kw/35,8kw/35, worst 4.2 s |
| dec | | | | - | R3: +25.3 s (+64%) n=1c+1w (2k/35), wins both |
| dense -> expert | unknown in the stats.logs; runs.db: 8k warm one row (3.0); planner_dataset: 10240 +12% then losses at 12k/16k | | | | |
| today | others n=135: loss 692 s (54%), worst 79.4 s; up9_merged2 n=7: 43 s (63%), worst 13.5 s; rel_b14244 n=7: 14 s (25%), worst 5.5 s; rampool9-10 n=14: 28 s (21%), worst 6.6 s | | | | |

### mimo (MiMo-V2.6-Flash 3bit, 90 calls)

| from \ to | mix | k | chain | dec | dense |
|---|---|---|---|---|---|
| stream | R1: +1.1 s (+4%) n=1c+1w (12k), wins both | R5: +13.3 s (+10%) n=5c+5w, wins 512c,8kw,12kc,12kw,16kc,16kw, loses 512w,1kc,1kw,8kc, worst 1.0 s | unknown (runs.db 12k/16k rows: chain cells exist, no stats.log pair) | R1: -8.3 s (-72%) n=1c+2w, loses 512w,1kc,1kw, worst 4.1 s | R2/R3: +35.6 s (+38%) n=1c+3w, wins 8kw,12kw,16kc,16kw, no loss |
| mix | - | R3: +0.8 s (+3%) n=1c+1w | unknown | unknown | R3: +4.2 s (+31%) n=1w |
| k | | - | unknown | R3: -7.6 s (-63%) n=1c+2w, worst 3.8 s | R3: +22.4 s (+28%) n=1c+3w, wins all |
| dense -> expert | unknown in the stats.logs; runs.db warm rows at 256, 8k, 12k; planner_dataset 12k/16k: dense wins (907/897, 995/774) | | | | |
| today | others n=84: loss 353 s (39%), worst 36.9 s; rel_b14244 n=2: 0 s | | | | |

### ds (DeepSeek-V4.1-Flash IQ2_XXS, 226 calls; never fits: `-nm 0 -lm mmap`, SSD tier)

| from \ to | mix | k | chain | dec | dense |
|---|---|---|---|---|---|
| stream | R1 (SSD-bound: no mix): -3.8 s (-6%) n=1c+1w, loses 12kc,12kw | R5: +24.6 s (+10%) n=4c+5w, wins 1kc,1kw,16kc,16kw, loses 512c,512w,3kw,12kc,12kw, worst 5.9 s | unknown in stats.logs (runs.db warm 256/16k pairs) | R1 (the one model where the chain pays): dec wins 512c,512w,1kc,1kw,3kw,16kc, loses 16kw by 283 s (one 328 s outlier call); net -237 s with it, +46 s without | R2/R3: +22.7 s (+49%) n=1c+1w (1k), wins both; split loses from 3k (dataset) |
| mix | - | R3: -6.3 s (-9%) n=1c+1w, worst 4.0 s | unknown | unknown | unknown |
| k | | - | unknown | R3: wins 512-3k, loses 16kc,16kw (worst 296.9 s) | R3: +10.6 s (+31%) n=1c+1w |
| dec | | | | - | R3: +5.6 s (+20%) n=1c+1w (1k) |
| dense -> expert | unknown (no forced expert row) | | | | |
| today | others n=214: loss 3305 s (98%), worst 296.9 s (decode-chain calls at 16k); rel_b14244 n=9: 55 s (31%), worst 10.5 s; up9_merged2 n=2: 0 s | | | | |

What the five tables say together: (1) stream is the right incumbent at every size on four models and at >= 3k on DS; (2) the chains lose on every
class of qwen36, qni4, glm, mimo (-97% to -308% of the stream's time) and win on DS below 3k; (3) the split pays from 8-12k on glm/mimo/qni4 and at
1k on DS (the only sub-3k split win), always compute-bound; (4) k is a coin flip except MiMo/DS at 8-16k (+10%), hence a warm trial only; (5) mix is
+4-6% where a slower link helps (glm30 12k, mimo 12k, qni4 16k) and -6% on the SSD-bound DS; (6) "today": the release loses 6-31% on these classes
(GLM 25%, DS 31%), up9_merged2 11-63%, rampool9-10 21% on GLM; the predictor-era builds ("others") 39-98%: the decode chain at large passes is the
single largest loss class in every model (worst 127.7 / 26.8 / 79.4 / 36.9 / 296.9 s).

## Strategy decision (replay of the candidates on the same 917 logged calls; /tmp/xo_score/score2.py, lomo*.txt)

Verdict: keep the learned means with persistence (rampool13's R1-R7 with R4), not "calibrate the physics at load + one scalar per arm". On the same
data the scalar strategy (h) is -8% warm and -34% cold against +26% / -27% for the measured means (c), and carrying the scalars across processes
(h+) is worse still (-34% warm: one bad decode-chain scalar carried from DS's 328 s call costs 297 s), while carrying the means (c+) is the best
of everything scored: +48% warm, +25% cold, worst loss 11 s. The physics anchor is not useless: it is the prior of every arm (rule b, -1% warm
by itself), and the scalar strategy fails for a stated reason: a single scalar must carry what the anchor misses (overlap, wait-for-other,
contention) across sizes, and those are not proportional to the computed time (the chain's serial misses at 512 vs 16k, the split's x4 wall).
A per-arm RLS with k = 4 is the smallest model that is right at two sizes at once; with the state file it needs no more samples than the scalar
(c+ scores from the first call of a process). Hybrid where the scalar fails: none needed; the anchor already enters as the RLS prior (coefficient 1 on U).
Beyond this and the oracle fallback no better strategy is in the data: every event/utilization trigger tested so far (section 1: T_partial/T_uploadAll,
T_one/Ts on the SSD-bound Ts) separated the crossovers worse than the measured means; the leave-one-model-out table below is the selection
criterion for the generic triggers (fit on four models, applied to the fifth).

| rule (fresh memory per process unless +) | cold net s (%) | warm net s (%) | worst single loss | warm per model ds / glm30 / glm35 / mimo / qni4 / qwen36 |
|---|---|---|---|---|
| c measured means | -882 (-27%) | +557 (+26%) | 15 s (73%) mimo 16k cold stream | +403 / +8 / +62 / +44 / +19 / +21 |
| d c + ABAB on overlap | -882 (-27%) | +538 (+32% of scored) | 11 s (136%) ds 3k warm k | +348 / +8 / +67 / +72 / +22 / +21 |
| h physics x scalar | -988 (-34%) | -128 (-8%) | 14 s (68%) mimo 16k warm | -85 / +6 / -57 / +16 / -18 / +10 |
| h2 h + ABAB | -1249 (-43%) | +194 (+17%) | 11 s (136%) | +97 / +6 / +15 / +44 / +4 / +28 |
| c+ means carried across processes | **+426 (+25%)** | **+862 (+48%)** | 11 s (136%) | +596 / +8 / +95 / +99 / +13 / +50 |
| h+ scalars carried | +201 (+11%) | -436 (-34%) | 297 s (944%) ds 16k warm dec | -373 / +8 / -17 / +26 / -25 / -55 |

Per transition c+ (cold | warm, net s): one/split +452 | +471, chain/dec +265 | +569, k -3 | +16, mix -4 | +4. Caveats: carried memory picks arms
without a truth cell (c+ scores 199 cold calls of 534), the truth medians come from the same calls, n per class 1-3.

Leave-one-model-out (constants fit on four models, applied to the fifth unchanged; 938 calls, unmeasured picks scored as the stream; /tmp/xo_score/lomo.py,
lomo_out.txt). Strategies: A release factors (no fit), C measured means (split prior ratio fit), H physics x scalar (same fit), E utilization/event
triggers (cu = C1/U for the split, miss_share and n for the chains, Um/U and ssd_share for mix, u_link for k; 5 thresholds fit), HY = E cold + H warm.

| held-out | A cold / warm | C cold / warm | H cold / warm | E cold / warm | HY cold / warm |
|---|---|---|---|---|---|
| qwen36 | 0% / +2% | 0% / +2% | -67% / 0% | -148% / +20% | -140% / 0% |
| qni4 | -4% / -2% | -4% / -2% | -4% / -6% | -5% / +3% | -8% / -6% |
| glm | -10% / +13% | -8% / +14% | -8% / -1% | -1% / +13% | -1% / -1% |
| mimo | -28% / -4% | -28% / -3% | -28% / -3% | -23% / +25% | +22% / -3% |
| ds | -45% / -40% | -45% / -24% | -45% / -38% | -17% / -35% | -17% / -38% |
| mean / min (pooled) | -13% / -43% | **-10% / -33%** | -23% / -49% | -26% / -101% | -25% / -101% |

Reading: no strategy generalises in net terms; C (measured means) loses least and A is close; the event triggers E/HY win on one model (mimo cold
+132 s, ds cold -275 vs -744 s) and lose 650 s on qwen36 cold with the same thresholds, and their fitted thresholds sit on the grid edges in every
fold (t1 = 0.5: split almost always allowed; t2/t3: the chains never fire), so their "stability across folds" is degenerate, not evidence. What no
strategy gets right on a held-out model: the decode chain as the best arm (mis-picked 41-166 times), k as the best arm, mix as the best arm (never
found, only mis-picked), one card vs split on ds and mimo cold. So the generic-trigger criterion does not select a trigger set from this data; it
confirms the rule set of section 0: measured means (R3) with persistence (R4: c+ is the only variant positive cold, +25%), stream as the cold
start (R1), and the remaining per-model crossovers (DS's chain, the k arms, mix) found by the sweep of section 5 and the VOI trial, not by a constant.

## 0. The final rule set (spec for Opus; scored on 917 logged prompt calls of 384 q5h stats.logs, section 1)

Scoring (section 1): every candidate rule replayed in log order with a fresh memory per process; outcome of a pick = the measured median of that arm
at (model, size class, cold/warm); gain = time saved against always-stream, loss = time lost against the measured best, cold and warm apart.
Result in one line: cold requests lose 27-31% against the best arm under EVERY rule because a fresh process knows nothing; warm requests gain
+26% with "lowest measured mean" (rule c) and lose 15% under the release's factors (rule a). So the planner's job is (1) a safe cold start and (2)
measured means kept across processes. The rules:

R1 cold start = full stream (+ mix where a second link exists and the SSD is not the ceiling) at every call above the model's measured CPU crossover
(the release's `max-batch` measurement, 62-128 tokens here: MC:8092-8250 at 29f224aa4, `g_mb_auto`); below it the cache path (misses on the CPU).
Evidence: no rule gains cold on chain/dec vs stream (-399 s for all, the chains lose cold on every model but DeepSeek at <= 3k); mix +15-23 s warm,
-12 cold, never worse than stream where a slower link exists; the "T_partial ~ T_uploadAll" hypothesis is REJECTED: the chain loses at
T_chain / T_uploadAll 1.17-2.2 (GLM 3.0/3.5, MiMo) and wins at 1.26-3.56 (DS): no single r separates them (best r = 0 = always stream, 12/15 cold,
8/11 warm). The chains are therefore never a computed candidate above `g_mb_auto`; they can only be measured into the memory (R3).

R2 one card vs dense split, no sample at comparable features: split iff the one-card time predicted from measured rates, max(Ts, C1), exceeds
r x Ts (the pass is compute-bound by the margin the x4 card's wall and the fill cost) AND the gain over the calls left in the request exceeds the fill
owed; r is a learned scalar (its own 1-dim RLS on the logged T_one/Ts at the split's wins and losses), prior 1.83 from the logs (warm: 13/13 cells
right; cold 9/10). The direction is settled: compute-bound -> split (rule e, warm +213 s) against upload-bound -> split (rule f, -276 s). The
release's 8% margin and 15% "sure" rule (R7/R10) go: they lost GLM 3.5 warm 10k (rows 23772/73: one card 362 t/s picked against the split's measured
497) and score -15% warm overall.

R3 with samples at comparable features (`rls_known`): the lowest measured mean among the arms known here, the split's fill added when owed; an arm not
known here is a trial only by value of information (EI x calls-of-this-size-so-far > EL, Opus's `rls_voi_pick`); the trial alternates with the
incumbent (A B A B) and stops at the sign of the paired differences (section 4.3). Evidence: rule c +557 s warm (+26%), with the paired trial (d)
+538 s overall but +35 s more on one/split and +94 vs +13 on k (thin cells); worst single loss 14 s (68%) in both, one call.

R4 the memory crosses processes: the state file stores every arm's RLS (theta, P/s2, n) under the `@hw#build` key when determined and quiet (the design's
persistence rule, rampool.md) and loads it as the prior. This is the largest lever in the data: the cold gap (27-31%) is exactly what a process
without the last process's means loses. The cells are run with `-cs` (empty state) only for the cold-start rows; every other row keeps its state.

R5 the k arms (stream with k x picks ahead): each competes on its calculated upload: the experts the call's tokens rank in their first 2^i x picks, by the lifetime counts,
x 1.3 (PP_K_LATE: the forced sweep's break-even at 77% of a layer's experts, 2026-10-31), so a prompt that touches most experts is the stream's and a short one is the arm's: a mean pick, cold too,
and against a known stream a VOI trial. Lowest k first (a higher k only after it ran). Earlier revision: never a cold candidate and the upload feature upside down (the stream's x (1 - 2^i x picks / experts), so k x32
looked free and was tried first). Measured wins stand (DS 16k 31.4 vs 45.0 s, MiMo 16k 32.3 vs 35.0; GLM 3.5 10k lost 17.4 vs 16.9).

R1b R1's bound is max(g_mb_auto, min(dec_max, 256)): g_mb_auto (experts / picks) is 32 on the big models while the chains win to about 200 tokens on MiMo and GLM and to about 1000 on DeepSeek (forced sweep 2026-10-31).
The calculated crossover dec_max (in every CAL line) says thousands before a stream pass measured the compute, so it is capped at 256; a chain that ran competes on its mean at any size.

R3b the mix competes on its mean up to 4096 tokens (PP_MIX_COLD): MiMo 418 tokens 3018 ms against 3834, 1016 tokens 3187 against 3907 (2026-11-08); Qwen3.8 IQ3 +18-47% at 1.5-4.5k; GLM 3.5 2.3k +10%. Past 4096 it stays a VOI trial: MiMo 12k 495 against 933 t/s, DS 12k 304 against 530.
A mix not known here is no candidate while uploads are under way: its freeze drops the queued ones (pp_settle), right after a start the hot set (forced mix cells: 24% found on the fast card against 44%).

R5b a call under GGML_OP_OFFLOAD_MIN_BATCH (32 tokens) streams nothing: its expert matmuls run on the CPU, so the stream family (stream, mix, k) is no candidate when a chain can run, and such a call never
teaches the stream arms or the SSD rate. Measured 2026-11-08: the prompt's 4-token tail 1.1 s (GLM 3.5) and 1.7 s (DS) on that path, 0.5 and 0.3 s on the decode chain; the 2-token start-up call, priced against the
link rate before any stream ran (1.3 GB/s), left the stream's upload coefficient 17x low, and the first short prompt (15-33 tokens, GLM 3.5 and DS) then streamed cold: 12.5 s, against 2.4 s on ram-pool.

R6 dense + expert split and expert-only: no cold candidate (section 1 of the previous revision: 5-12% on two models, no trigger); VOI trial only
when the split is already the incumbent and a slower card has free VRAM; off everywhere the trial lost.

R7 every one-time cost (fill, re-plan, the uploads a chain pass drops) is a learned term charged once per request against the gain over the calls
left; every pick logged with both sides' numbers (PLN/DON/TRY, I12); the 5% loss bound is the replay gate's and the cells' yardstick, never a run-time
constant.

Dropped from the earlier revision of this file: the computed per-tier stream anchor as a DECISION (it stays as the stream's prior U feature; as a
tie-breaker it scored no better than the means), the saturation trigger (no utilization field separates the logged crossovers; the per-card GB/s
exists in 192 rows), the "close is enough -> stream" tie rule (it never changed a scored pick), the compute/upload ratio for the chains (rejected
above). Kept from Opus's planner-simple (e42a95af09, 374c6ac772, a448602c58): deterministic means, VOI trials, the stake instead of a 5% cap.
To change there (message sent to the agent): restore the chains' eligibility above the MEASURED `g_mb_auto` crossover as "trial only, never a cold
pick" (a448602c58 removed `dec_max` gating as a ballpark: the measurement is not a ballpark, the faulty calculated crossover was); replace the split's
prior ratio RLS_SPLIT_ONE_RATIO 0.96 by R2's learned r on T_one/Ts (prior 1.83); make the state file carry the arms' RLS across processes (R4) and
have the replay gate run with the previous process's state, not only fresh.

## 1. Scoreboard (python replay over /p/bw/data/q5h stats.logs; files in /tmp/xo_score: events.csv, truth.txt, scoreboard_full.txt, extra_out.txt)

Events: ds 226, glm30 66, glm35 104, mimo 90, qni4 (Qwen3.8-Flash-Next IQ4_XS) 62, qni3 6, qwen36 363; qwen38 (27B) and olmoe none. Cells are thin
(n = 1-3 for most model x size x cold/warm x arm); the truth medians come from the same calls, so the numbers are indicative, not a gate.

Rules: a = release b14244 factors (max(U, C), 8% margin, chains only <= mb_auto); b = computed anchor per arm, lowest; c = lowest measured mean,
unmeasured arms priced by b; d = c + ABAB on overlap; e = one/split by "compute-bound -> split" (C1 > Ts and gain > fill), stream otherwise;
f = e reversed (upload-bound -> split); g = always stream, never split.

| rule | cold: net s (net %) / worst loss | warm: net s (net %) / worst loss | one/split warm | chain-dec vs stream warm | k vs stream warm | mix vs stream warm |
|---|---|---|---|---|---|---|
| a release | -949 (-27%) / 15 s | -342 (-15%) / 14 s | +193 | -195 | -191 | +20 |
| b anchor | -882 (-27%) / 15 s | -22 (-1%) / 14 s | +246 | -22 | -84 | +17 |
| c means | -882 (-27%) / 15 s | **+557 (+26%)** / 14 s | +337 | +240 | **+94** | +23 |
| d means + ABAB | -882 (-27%) / 15 s | +538 (+32% of scored) / 11 s | **+372** | **+275** | +13 | +15 |
| e compute-bound -> split | -930 (-27%) / 15 s | -258 (-12%) / 14 s | +213 | -119 | -193 | +20 |
| f upload-bound -> split | -1070 (-30%) / 15 s | -1224 (-42%) / 14 s | -276 | -577 | -381 | +2 |
| g always stream | -1134 (-31%) / 15 s | -1477 (-43%) / 14 s | -323 | -674 | -475 | -5 |

Cold: one/split -146 s for b/c/d (empty memory), chain/dec -399 s for every rule. Per model warm net (c): ds +403, glm30 +8, glm35 +62, mimo +44,
qni4 +19, qwen36 +21 s.

Ratio tests (extra_out.txt): chain/dec vs stream: stream wins from 512 on GLM 3.0 (T_chain/Ts 1.24), MiMo (1.28), Qwen3.6 (2.12, rising to 19-21 at
12k), 2k on GLM 3.5 (1.17), 3k on Qwen3.8 IQ4 (3.7-5.0); the chain wins only on DS at 512-16k cold and 512-3k warm (1.26-3.56), and DS's decode chain
at 16k warm takes 328 s against 45 s streamed. One card -> split: T_one/Ts best threshold 1.83 warm (13/13), cold 0 (9/10, misses GLM 3.5 2k cold);
the upload-bound cases (GLM 3.5 cold, Ts 17 s at 2k-10k) are exactly where the split fails: the direction is compute-bound.

Release baseline with the same switches (rows 23772-23777, rel_b14244, pf12k, cold/warm pp): GLM 3.5 rp0 273/386 (one card at 10336 against its own
measured split 497 vs 362 t/s: the C3 loss), MiMo 917/978 (one card), DS 336/373 (split, "measured: the nearest size").

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

## 5. The campaign: forced-pure sweep of the cells still missing (inventory of 2026-11-07, /tmp/xo_inv: tables.txt, pairs.txt, missing.csv)

Models: exactly qwen36, qni4 (Qwen3.8 IQ4_XS, models.ini `qn_iq4xs`), glm (3.0 and 3.5), mimo, ds. Inventory of forced pure strategies in runs.db
(ok = 1, builds from 2026-10-14 on; the `pp-path`/`pp-k` switches exist since 2026-11-02 builds and only in the "mid" era, no cur-era row has them;
395 of the 2117 forced rows have a stats.log in q5h, all 40 cur-era ones do). Sizes where BOTH arms of a pair are measured (cold | warm):

| pair | glm30 | glm35 | mimo | ds | qwen36 | qni4 |
|---|---|---|---|---|---|---|
| one / split | 256, 2k, 4k, 8k, 12k \| 4k, 8k | 4k, 8k \| 2k, 4k, 8k | 4k, 12k, 16k \| 256, 4k, 8k, 12k | 16k \| 1k, 16k | 8k \| 256-16k | 4k, 8k, 16k \| 2k-16k |
| stream / chain | 64, 16k \| 64, 12k | none | 12k, 16k \| 12k | - \| 256, 16k | 12k \| 1k, 16k | 16k \| 16k |
| stream / dec | 64, 256, 512 \| 64, 256 | none | 64 \| 64, 256, 512 | 256 \| 256, 1k | 1k, 8k \| 256, 1k | none |
| stream / k | none | none | none | none | - \| 256, 1k (k x1) | none |
| stream / mix | 16k \| 64, 12k | none | 64, 12k, 16k \| 64 | - \| 16k | 8k \| 1k, 8k | 16k \| 16k |
| split / expert | - \| 8k | none | - \| 256, 8k, 12k | none | none | none |

Regimes measured: rampool=1 only glm35 (21 rows), `-lm mmap` ds 529 / glm35 195 / qni4 36; sim-vram qwen36 4.5 GiB x3316 (the small-VRAM stand-in), mimo,
glm30 a few; sim-ram glm30/mimo 32-124 GiB. Of a 5-model x 8-strategy x 9-size pure matrix 300 cells, 229 have no row; the sweep below runs only the
cells a rule of section 0 needs and the existing pairs do not answer (reuse rule: a pair measured cold and warm at a size is not rerun).

Cell = one pure strategy at one size, `n=2` (run 1 cold with `-cs`, run 2 warm), tokens 64 out, stats.log captured (`--moe log=esrgp`, PERF_LOG as
run.py sets for fork builds). Strategy switches (args column): stream `--moe pp-path=stream,pp-k=0`; mix `pp-path=mix`; chain `pp-path=chain`; dec
`pp-path=dec`; k2 `pp-path=stream,pp-k=2` (k4 likewise); one `dense-split=0`; split `dense-split=1` (with the one-card path left to stream); expert
`dense-split=0,prefill-split=1`; regimes on the command line: `-nm` (never mount; ds and glm35 `-lm mmap` as their rows have), `-sim 4.5g` (qwen36),
`--moe rampool=1` (glm35 rp1 rows). Sizes below 1k: the new tests pf64/pf256/pf512 (T1) or `pf128k` with `--clamp N --vary` in extra as the old sweeps did.

| model (csv) | cells to run (strategy at sizes) | why (rule) | n |
|---|---|---|---|
| glm (3.0), fork/runs/xo-glm30.csv | one, split at 1k, 2k (cold+warm; 2k warm missing); dec, chain, stream at 1k, 2k (R1's crossover band: dec measured losing at 2k only on the decode chain of auto rows); k2 at 2k, 8k, 16k; mix at 4k; expert at 12k | R2 cold r at 1-2k; R1 band; R5; R6 | 16 |
| glm35, fork/runs/xo-glm35.csv | stream, chain, dec at 512, 2k (none measured on this model); one, split at 2k, 12k cold; k2 at 10k; mix at 10k; all of stream/chain/split at 2k and 10k again under `--moe rampool=1` | R1 on the non-fitting model; R2 cold; the rp1 release question | 17 |
| mimo, fork/runs/xo-mimo.csv | dec, chain, stream at 256, 512, 1k cold (warm exists); one, split at 2k, 8k cold and at 6k with the context fixed (8k cells and 126k cells via `-ps` ctx variants); k2 at 8k, 16k; expert at 8k cold | R1 band; R2 (the context axis); R5; R6 | 17 |
| ds, fork/runs/xo-ds.csv (`-nm 0 -lm mmap`) | dec, chain, stream at 512, 1k, 2k, 4k cold+warm (the one model whose chain wins: the crossover sits in 3k-16k warm); one, split at 1k, 2k, 4k, 8k cold; k2 at 16k (k x1 won 31 vs 45 s), 8k; mix at 4k | R1's only exception; R2; R5 | 21 |
| qwen36, fork/runs/xo-qwen36.csv (`-sim 4.5g` for stream/chain/dec/k; real VRAM for one/split) | dec, chain, stream at 64, 256, 512 cold (warm exists); k2, k4 at 1k, 8k; one, split at 2k, 4k, 12k cold; expert at 8k | R1 band (35-token crossover); R5; R2 cold | 16 |
| qn_iq4xs, fork/runs/xo-qni4.csv | dec, chain, stream at 256, 1k, 4k (chain/dec unmeasured below 16k); one, split at 2k, 12k cold; k2 at 16k, 8k; expert at 8k, 12k | R1; R2 cold; R5; R6 | 16 |

Total 103 cells x about 6 min (one strategy, two runs, the load; ds 10, mimo 60k-context 15) = about 11 h on PC1; W10 none (OLMoE is a smoke test
only); PC2: the CPU us per token-expert per model (`-ng 0`, R1's measured crossover input) 1 h. Order by value: ds (the one chain exception and
the largest losses: 328 s decode-chain calls) -> glm35 (rp1 question, non-fitting) -> glm30 -> mimo -> qni4 -> qwen36.

Commands (Sonnet, one list per model): `run.py jobs xo-ds -b <build> -nm -pl 220/280 -r 2 -cs` and so on; the build is rel_b14244 for the pure
strategies (the forced paths do not depend on the planner) and the merged planner build for the auto rows of section 6's P8.

### Analysis recipe (how a rule is read off the sweep)

Per cell and pass, from stats.log and server.log: CAL `prompt call of n` (streaming upload ms = Ts, mix ms = Um, misses uploaded ms = up, 4-token chunks
ms = ch, stream reads GB / SSD GB/s), DON `took ms`, PAS (up GB per card and GB/s, cpu MiB/token), SPN `copy fill dur` (the fill, pins apart), `split
balance` (bal_kern, bal_upwait, bal_wait_other, bal_stall, bal_linkbusy, bal_exp_gb per card), cells at the pass, cache hit %; fork/planner_events.py
extended by T4 writes them per call (model, size class, cold/warm, arm). Then per model and adjacent pair (stream|chain, stream|dec, stream|k2,
stream|mix, one|split, split|expert): (1) the paired difference per size, cold and warm apart, and the size where its sign changes = the measured
crossover; (2) on both sides the candidate ratios: T_one / Ts, C1 (= stream took - its upload wait) / Ts, T_chain / Ts, wait_other / pass,
linkbusy_x4 / pass, hit %, SSD GB / took; (3) a ratio qualifies as the rule's quantity when ONE threshold separates win from loss on every model and
both temperatures (hit rate >= the paired noise allows: the threshold's uncertainty = the ratio's spread across the two sides); its value is the prior
of the learned scalar (R2's r), never a constant; (4) a pair with no qualifying ratio stays "measured means + VOI trial" (R3) and the sweep's medians
become the state-file prior per `@hw#build`; (5) the scoreboard of section 1 is rerun with the new events (rules a-g) and the rule set of section 0
is amended only where a rule's net falls below the measured-means rule.

### Briefs for the sweep

Sonnet (medium), run the forced-pure sweep: PC1, user o, run.py only. Write fork/runs/xo-<model>.csv from section 5's table (columns
`name,model,n,test,tokens,args,extra`; one row per strategy x size, n=2, tokens 64; args = the strategy switches of section 5; ds rows extra `-lm mmap`;
qwen36 miss-path rows extra `--hog-gpu 4.5`; glm35 rp1 rows args `--moe rampool=1,<strategy>`; MiMo context variants via `-ps`). Start from
/tmp/xo_inv/missing_5models.csv and keep only the rows named in section 5. Run `run.py jobs xo-ds -b rel_b14244 -nm -pl 220/280 -r 2 -cs`, then
xo-glm35, xo-glm30, xo-mimo, xo-qni4, xo-qwen36 (that order). Check runs.db first (feedback_check_runs_db_before_queue): skip a cell that has a cold and
a warm ok row of that strategy at that size on a build from 2026-11-02 on. One queue instance. Read every answer in fork/outputs/<rowid>.txt before its
speed (nonsense = ok 0). Stop at the first failing cell (server death, GUARD in q5h/watch-DATE.log, refused pass): hand back the cause, rowid and log
path; do not continue past it. Return per model a table `strategy | size | rowids | judge | cold took s | warm took s | cold pp | warm pp | stats.log
stem` and the list of cells skipped as already measured. Hand back when all lists are done or at the first failure.

Opus (high), turn the tables into rules: inputs = /tmp/xo_score (events.csv, truth.txt, score.py, matrix.py), /tmp/xo_inv, the new sweep's stats.logs
in /p/bw/data/q5h, fork/planner_events.py, tests/test-planner-replay-fork.cpp. (1) Extend planner_events.py to every model and the fields of the
analysis recipe (section 5) and rerun score.py's rules a-g and matrix.py on old + new events; (2) per adjacent pair apply the recipe's steps 1-5
and report the qualifying ratio or "means + trial" per pair, cold and warm apart, with the fit; (3) encode the result as section 0's R1-R7 in the
planner (the chains' measured crossover as eligibility, R2's learned r with the sweep's prior, the state-file persistence of every arm's RLS, the VOI
trial with the paired stop), each pick logged with its numbers; (4) the replay gate runs twice, from a fresh state and from the previous process's
state, and passes when no pick loses > 5% against the measured best; (5) commit on branch plan-rules, push, report the scoreboard before and after
with the worst loss per model. Nothing hand-tuned; a number the data does not give is a learned prior with its sd logged.

## Oracle-trace procedure (one model and prompt that visits most crossovers; the best run's decisions become the rules)

FALLBACK, not started (user 2026-11-07): the current plan is rampool13's rules R1-R7 under the release gate on pf12k4k. Start the oracle trace only
when: (a) the release gate has a failing model or cell after two fix rounds (feedback_stop_fix_repeat), or (b) on any gated cell the auto run's total
prompt time is slower than the best forced arm of the same cell by more than X, where X = the paired noise of that cell = the relative half-range of
the cell's repeated warm runs (runs.db, same build and arm), with a floor of the measured run-to-run spread of the model's pf12k rows (GLM 3.5 rp0:
481-600 pp across rampool8-10 warm runs, about 10%; the 5% yardstick is below that spread, so it cannot be the trigger). Until then no oracle cell
runs and PC1 is not occupied by this section.

### 1. Which crossovers one request visits (from the logs' pass structure)

A request = the server's first call (~2k tokens: 1938/2034/2089 by build) + the rest in passes cut by the pool cap (GLM 10336-10496 in one pass; MiMo
9568-token passes, DS 13502 at `-nm 0`) + the decode. Per pass the planner meets: X2/X3 (stream / mix / chain / k) and X4 (one / split) at every pass,
X5 at passes >= 8k, X1 only where the pass is small (the chains' band is 64-512 tokens on four models, <= 3k on DS), the ctx axis only when passes
reach 60k+ cells, the SSD tier only on non-fitting models.

| candidate (test) | passes | X1 CPU misses | X2/X3 chain, k, mix | X4 split | X5 expert | ctx axis | SSD / pool tier | decode decisions | cell min |
|---|---|---|---|---|---|---|---|---|---|
| GLM 3.5 pf12k4k, rp0 and rp1 | 2k + 10k | the cold 2k call only (the 43-86 s decode-chain loss lives here) | both passes | both passes (one at 2k, split at 10k: the release's 6/7) | 10k (confirm) | no | yes: non-fitting at PC1, pool rp1 | 4k tokens: cache / pool / SSD misses | 7 (rp1 10) |
| MiMo pf60k4k | 2k + 6 x 9.5k | no | every pass | every pass, across 2k -> 60k cells (the attention term, the 245k ban's origin) | every big pass | yes | no (fits) | 4k | 20 |
| DS pf12k4k (`-nm 0`) | 2k + 13.5k | the 2k call (the chain WINS here below 3k) | both | 2k (split won at 1k, lost from 3k) and 13.5k | 13.5k | no | yes: always SSD | 4k, 328 s decode-chain class | 14 |
| qni4 pf16k4k | 2k + 14k | no | both | both (split from 10k) | 14k (+7-10% measured) | no | no at PC1 (78 GiB fits) | 4k | 8 |
| qwen36 pf4k1k `-sim 4.5g` | 2k + 2k | no | both | 2k | no | no | no | 1k | 4 |
| any model pf512 (64 out) | 1 x 512 | yes: the band itself | yes | no | no | no | per model | - | 3 |

Recommendation: first GLM 3.5 pf12k4k (rp0, then rp1): it visits X1-X5 and the SSD/pool tier in two passes at 7-10 min per cell, and it is the release
question (rows 23772/73 one card 273/386 pp against the split's measured 497). Then MiMo pf60k4k (the ctx axis), DS pf12k4k (the chain's one
winning regime and the SSD tier), qni4 pf16k4k (X5's only measured gain), qwen36 `-sim 4.5g` (small VRAM). Every model also gets one pf512 request
(X1's band is not reachable by a 12k request). 12k is enough for every crossover but the ctx axis.

### 2. The procedure

Hysteresis: a pass's time depends on the passes before it (a chain pass leaves misses, k staging evicts later layers, the copies stay in place after a
split, the pool's set after the first request). So the oracle is over whole-request ARM SEQUENCES, never per-pass arms from different histories:
a per-pass "best" is defined only among runs that share the same decisions up to that pass.

1. Sequences. For the 2-pass request: pass 1 in {stream, mix, chain, dec, k2}, pass 2 in {one card stream, split, split + mix}: 15 sequences; forced
   per pass through the per-request force (`/moe/plan force=` of T7; until it exists `--moe pp-path=`/`pp-k=` for pass 1 and `-ds` for pass 2 apply to
   both passes, so the split arm is `-ds on` with pass 1's path forced and the one-card arm `-ds one`: 10 sequences). For MiMo's 7 passes: greedy by
   position: fix the best prefix found, vary only the next pass (3 arms), 2 repeats each, then the next position (the attention term makes the later
   passes' winner shift to one card at some cell count: that cell count is the crossover).
2. Runs. Each sequence as one cell, `-r 2 -cs` (run 1 cold, run 2 warm: two oracle states); the sequences of one position run interleaved A B A B
   (one queue, same power session) so drift hits all alike. The noise = the paired half-range of the 2 repeats of the same sequence (per pass and per
   request). The oracle at a position is stable when the best sequence beats the runner-up by more than that noise in both repeats; otherwise one more
   repeat of the top two (never more than 4 repeats: beyond that the two arms are a tie and either is the oracle, logged as such).
3. Archive the best run: its runs.db row (`oracle=1` column via T4's decisions table: the sequence, per pass took, fill, the paired noise), its stats.log
   (already stored zlib in `stats`), the server log, and the state file after the run (T3). The oracle for the model = the best sequence cold and the best
   sequence warm, with per-pass times.
4. State features at each pass boundary of the best run, from its stats.log: CAL (U link-bound, SSD GB and rate, Um, up, ch), PAS (per-card GB/s =
   utilization against the link ceiling, cpu MiB/token), DON took, SPN fill (pins apart), split balance (wait_other, linkbusy, stall, kern per card), hit %,
   cells, C1 = stream took - its upload wait, the pool counters (rp_hit/rp_ssd). The rule for a crossover = the quantity whose value at the oracle's switch
   separates the sequences that lost from the one that won across the repeats (the recipe of section 5: a single threshold with the paired noise as its
   uncertainty; else "measured means + trial"). Fitted on the first model only.
5. Acceptance on the same model: the auto run (the rule-driven planner, `-cs` for the cold request) reproduces the oracle's sequence cold and warm and
   its total time within the paired noise; any disagreement is reported as the pass, both arms' times and the loss in seconds. Then the NEXT model runs
   the same auto build WITHOUT retuning against its own oracle (found by steps 1-3): the generalisation check; a rule that needs a retune per model is
   not a rule but a prior and goes into the state file instead.

Cost of the first model (GLM 3.5 pf12k4k): 10 sequences x 7 min x rp0 + the same at 10 min rp1 = 170 min, + the stability repeats (top two, both
regimes, ~35 min) + the pf512 band (4 arms, 12 min) + the auto acceptance runs (rp0 and rp1, `-r 2`, 20 min) = about 4 h PC1. MiMo pf60k4k greedy:
7 positions x 3 arms x 2 repeats x 20 min = 14 h (run at night); DS 10 x 14 x 2 = 4.7 h; qni4 10 x 8 = 1.3 h; qwen36 10 x 4 = 0.7 h.

### 3. Briefs

Sonnet (medium), oracle runs: PC1, user o, run.py only, one queue. Write fork/runs/oracle-glm35.csv with the 10 sequences of step 1 (name
`or-glm35-<p1>-<p2>`, model glm35, n=2, test pf12k4k, tokens 4096, args `--moe pp-path=<p1>` or `pp-k=2`, extra empty), run
`run.py jobs oracle-glm35 -b rel_b14244 -nm -pl 220/280 -r 2 -cs -ds one` and again with `-ds on`, then the same list with `-a moe:rampool=1`
(rp1). Interleave by running the list twice (A B A B over the sequences). Read every answer in fork/outputs/<rowid>.txt before its speed (ok 0 = out).
Stop at the first failing cell (server death, GUARD in q5h/watch-DATE.log, refused pass) and hand back the cause with rowid and log path. Return per
regime a table `sequence | rowids | cold: pass1 s, pass2 s, fill s, total s | warm: the same | paired noise s` sorted by warm total, mark the best
cold and the best warm sequence and whether they are stable (step 2); add the pf512 band (4 arms: stream, chain, dec, k2) as rows. Hand back when
the list is done or at the first failure. Then the same for oracle-mimo (greedy, step 1), oracle-ds, oracle-qni4, oracle-qwen36 when asked.

Opus (high), rule extraction: inputs = the oracle tables and the archived best runs' stats.logs (q5h), fork/planner_events.py, /tmp/xo_score.
(1) Extract the step-4 features at every pass boundary of every sequence of the first model, cold and warm; (2) for each crossover the best run
visited, find the quantity and threshold that separate the losing sequences from the winner across the repeats, with the paired noise as its
uncertainty; where none separates, record "means + trial"; (3) encode: the thresholds as PRIORS of the learned scalars (section 0's R2 form), the
means into the state file (R4), the chains' eligibility (R1), the VOI trial (R3); (4) replay gate from fresh and carried state on the oracle's calls:
the auto pick must equal the oracle's sequence cold and warm within the noise; (5) build, Sonnet runs the acceptance of step 5 on the same model,
then the next model without any change; (6) commit on branch plan-oracle, push, report per model the oracle sequence, the auto sequence, the loss in
seconds of each disagreement and the rule each pass boundary was decided by. Nothing hand-tuned; every threshold carries its noise.

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
