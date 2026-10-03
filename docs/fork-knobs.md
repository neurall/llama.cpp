# Fork options: flags, `--moe` settings, tuning knobs, environment variables, state file

Everything this fork adds to stock llama.cpp. Settings you give are used as given and are never auto-tuned. Defaults are those of  
the release this file ships with.

## What the fork does, in one minute

A mixture-of-experts (MoE) model has many small "experts" in every layer and uses only a few of them (for example 8 of 256) for each token. When the model is  
bigger than the GPU's memory, upstream llama.cpp ("stock") keeps some layers on the GPU and computes the others on the CPU. This fork instead keeps the  
experts in RAM and turns the free VRAM into a **cache**: in each layer a number of **slots** on the GPU hold the experts used lately, the GPU computes the  
cached ones and the CPU the others at the same time, and an expert that keeps being used replaces one that is not (it is uploaded over PCIe). An expert that is  
needed and cached is a **hit**; the **hit rate** is the share of such experts.

With no switches at all: the cache is on only when a MoE model does not fit in VRAM, the slots are sized from the free VRAM, and the fork measures token  
speeds and tunes its own settings (**autotune**) and picks the cache layout or the stock layout (**placement**) for the model and the GPUs. What it learns is kept in the  
state file. Everything below is for overriding that, and a setting you give is never touched by the tuner.

### Which switch do I want?

| I want | type |
|---|---|
| exactly upstream behaviour, nothing of the fork | `--fork off` |
| the cache, but fixed settings and no self-tuning | `-at off` (and `--moe slots=N` for a fixed slot count) |
| a reproducible benchmark | `LLAMA_MOE_STATE=0` so nothing learned earlier is used, plus `-at off` or explicit `--moe` settings; change one thing per run |
| the cache even where the fork would choose stock (or the other way) | `LLAMA_MOE_AUTO_MODE=cache` (or `stock`) |
| the fork to forget what it learned | delete `~/.cache/llama.cpp/moe-state.ini`, or run once with `LLAMA_MOE_AUTO_MODE=retest` |
| to check whether the cache is what slows my machine down | run the same command with `--fork off` and compare |
| big pinned models to start faster | the section on loading pinned weights, below |

## Command-line flags

| flag | environment form | what it does |
|---|---|---|
| `--moe KEY=VAL,...` | `LLAMA_ARG_MOE` | the expert cache settings and knobs, below |
| `--moe-expert-cache N` | `LLAMA_ARG_MOE_EXPERT_CACHE` | expert slots per layer in VRAM; `-1` sizes them from free VRAM, `0` = no cache (the rest of the fork stays on; use `--fork off` for stock behaviour) |
| `--prefetch-experts-slots N` | | staging slots for host-to-GPU prefetch of big batches (same as `--moe pf-slots=N`) |
| `-at on\|off`, `--autotune` | `LLAMA_ARG_AUTOTUNE`, `LLAMA_AUTOTUNE=0` | self-tuning of cache knobs, thread counts and the cache-or-stock placement; `off` = fixed defaults, nothing measured or saved (same as `--moe autotune=0`) |
| `-lm pin\|mmap\|dio`, `--load-mode` | `LLAMA_ARG_LOAD_MODE` | `pin`: weights in pinned RAM (the default of every tool when part of the model stays in host RAM and the model fits in the RAM available now; faster prompts and uploads, the load takes longer; `-lm mmap` opts out); `mmap`: memory-mapped, for models bigger than RAM; `dio`: direct IO |
| `-md FILE --spec-type draft-mtp` | | MTP draft head (Qwen3.8-Flash-Next, GLM-5.3-Flash), see the README |
| `-t N`, `-tb N` | | fixed decode / prompt thread counts (otherwise tuned) |

## Turn the fork off and run stock

| you want | pass | effect |
|---|---|---|
| plain stock behaviour | `--fork off` (or `LLAMA_ARG_FORK=off`) | no expert cache, nothing tuned, measured or saved; the same placement and kernels as upstream llama.cpp |
| the cache, but no self-tuning | `-at off` (or `--moe autotune=0`, `LLAMA_AUTOTUNE=0`) | the cache works with fixed defaults, placement uses a static rule, nothing is measured or saved |
| stock placement chosen by the fork | `LLAMA_MOE_AUTO_MODE=stock` | skips the cache-or-stock measurement and uses stock |
| the cache forced on | `LLAMA_MOE_AUTO_MODE=cache` | skips the measurement and uses the cache |
| a fresh start | `LLAMA_MOE_STATE=0`, or delete `~/.cache/llama.cpp/moe-state.ini` | ignore the learned state, write nothing |
| the hand-tuned stock baseline | `--fork off --n-cpu-moe N` (or `-ot`) | keep the first layers' experts in VRAM, the rest on the CPU, as in upstream; the fair comparison for the cache |

With `--fork off` the fork behaves as upstream apart from unrelated changes (the loader, MTP, merged upstream commits).

## `--moe` settings (command line)

`--moe` takes comma separated `name=value` pairs, names are case-insensitive and `_` equals `-`; the same string works as  
`LLAMA_ARG_MOE=...`. A setting you give is used as given and never self-tuned.

| key | default | meaning |
|---|---|---|
| `slots=N` | automatic: on when a MoE model does not fit in VRAM | expert slots per layer in VRAM; `-1` sizes them from the free VRAM, a number fixes them, `0` = no cache (the rest of the fork stays on; `--fork off` is stock behaviour) |
| `pf-slots=N` | `0` (off) | staging slots for host-to-GPU prefetch of whole expert tensors for big batches (3 is recommended when used, at most 4) |
| `up-max=N` | `2` | most expert uploads per layer and decode step; fewer means less PCIe and RAM traffic and a slower-changing cache |
| `recent=N` | `64` | how many recent tokens the cache looks at when it decides which experts to keep |
| `pred-top=M` | `0` (off) | guess which experts the next layers will need and upload the likely ones early (the top M of the guessed ranking); experimental, see the evidence below |
| `train-every=N` | `0` (the predictor is not trained while running) | train the predictor every N decoded tokens; implies `pred-top=8` if `pred-top` is not given |
| `autotune=0` | autotune on | no self-tuning (same as `-at off`) |

**Names.** The settings were renamed to say what they do. The old names (`cache`, `inserts`, `window`, `prefetch-slots`, `predict`, `train`, the knobs `MARGIN`, `GATE`, `SWAP_FRAC`, `BIG`, ... and their `LLAMA_MOE_CACHE_<OLD>` variables) no longer exist: an old `--moe` name is ignored, and an old state file entry is skipped, so the tuner relearns it. Case and `-` / `_` do not matter in the new names. `--moe-expert-cache N` (`LLAMA_ARG_MOE_EXPERT_CACHE`) is unchanged and works on every build.  

| new name | old name | | new name | old name |
|---|---|---|---|---|
| `slots` | `cache` | | `pred` | `PREDICT` |
| `up-max` | `inserts` | | `pred-iso` | `STREAM` |
| `recent` | `window` | | `pred-n` | `STREAM_M` |
| `pf-slots` | `prefetch-slots` | | `pred-lead` | `OFFSET` |
| `pred-top` | `predict` | | `pred-slow` | `STREAM_SLOW` |
| `train-every` | `train` | | `pred-tune` | `SELF_TUNE` |
| `swap-lead` | `MARGIN` | | `pred-auto` | `AUTO` |
| `swap-lead-per-link` | `LINK` | | `pred-keep` | `SLOTKEEP` |
| `swaps-per-step` | `BUDGET` | | `pred-slots` | `STREAM_SLOTS` (env only) |
| `vram-reserve-mb` | `MARGIN_MB` | | `idle-up`, `idle-up-n` | `TBP`, `TBP_LAYERS` |
| `upload-wait` | `GATE` | | `ev-cld` | `BIG` |
| `upload-wait-max-us` | `GATE_MAX_US` | | `pin-hot` | `HOT_FRAC` |
| `upload-chunk-kb` | `CHUNK_KB` | | `slow-min-stay` | `SLOW_STAY` |
| `upload-share` | `SWAP_FRAC` | | `stay-bonus` | `STICKY` |
| `upload-now` | `JIT` | | `l3-pf`, `l3-auto` | `L3PF`, `AUTO_L3` |
| `cpu-ram-gbs`, `ram-ceiling-gbs` | `CPU_GBS`, `DDR_GBS` | | `trace-n`, `trace-skip` | `TRACE`, `TRACE_AFTER` |
| `wait-swaps` | `WAIT` | | | |

Any other name is a tuning knob of the cache engine (`--moe upload-wait=3,swap-lead=0`). A knob you set is never self-tuned; a name the  
engine does not know is logged as `unknown key` and ignored. Values are numbers; `0`/`1` are off/on.

The cache keeps some experts in VRAM (the GPU computes those); any other expert is computed by the CPU or uploaded over PCIe first.  
The knobs trade CPU work against uploads. Every example below is a complete option: `llama-server -m model.gguf --moe <example>`.

Evidence labels: **proven** = a paired measurement showed a gain (where and how much is given); **unproven, on the way out** = no  
measured gain, will be removed or folded into the tuner unless a test shows one; **no A/B** = never tested on its own, kept as a  
measurement input or a safety limit. Numbers are decode tokens/s unless stated; "chat" = short chat prompts.

### The three knobs that matter most, in plain words

Each decode step a layer needs a few experts. The cached ones run on the GPU. For a missed one the engine can let the CPU compute it, or **upload** it into a  
cache slot so it is a hit next time. An upload costs PCIe time, RAM bandwidth (it reads the same RAM the CPU is reading) and an eviction (a cached expert  
has to make room). The three knobs answer three questions about uploads:

- **`swap-lead` (which swaps are worth it).** A missed expert only gets a slot if it beats the weakest cached expert by this many recent uses (over the last 64 tokens).  
  `0`: any expert used more than the weakest cached one gets in; the cache adapts fast and churns more. A high value: a stable cache that can go stale.  
  Default `-1` is 0; `-2` computes it from speeds (about the CPU read rate over the link rate: around 2 on an x16 link, 9 to 11 on a x4 link).  
  Measured on GLM (2 x 3090): the speed-derived margin capped the hit rate; margin 0 gave 71.2% against about 65% and 21.3 against 19.9 t/s (+6.7%).  
  Short chats like margin 0, long prompts did better with the speed rule, so the tuner chooses.
- **`upload-wait` (when an upload may run).** The CPU computing its experts and an upload both read system RAM and together reach its limit (on machine A about 38 GB/s for  
  the CPU alone, 44 with a DMA running). An upload that overlaps the CPU's reading slows the CPU, which is the critical path. `upload-wait=3` makes uploads wait, up to a limit,  
  while the CPU computes, so they run in the gaps; `upload-wait=0` uploads any time. Measured: +6% on a short GLM chat (22.10 against 20.87 t/s). `upload-wait=2` starved the uploads: never use it.
- **`upload-share` (how much).** The most time uploads may take per step, as a share of the token time (0.25 is a quarter). More uploads help the hit rate but risk stalling the step.  
  Measured: no benefit as a user setting (5 against the default gave the same hit rate, 64.6% against 64.7%; 0.5 with margin 0 was slower), so it is labelled "unproven, on the way out".

The tuner sets all three while it runs. If you change one, change one at a time and use `LLAMA_MOE_STATE=0`, or the learned settings will mix into the comparison.

**Swap decisions**

| knob | default | what it does | example | evidence |
|---|---|---|---|---|
| `swap-lead` | `-1` | An upload costs link time and only pays back if the expert is used again, so a missed expert must beat the one it evicts by this many recent uses. `-1` = 0 (swap whenever it scores higher), `-2` = from speeds (CPU read rate / link rate is about the uses an upload needs to pay for itself), `n` = fixed | `swap-lead=0` swaps eagerly; `swap-lead=2` swaps rarely | **Proven** on GLM-5.3-Flash 3.0-bit, 2x3090: the speed-derived margin (2 on x16, 9-11 on x4) capped the hit rate; `swap-lead=0` gave 71.2% hit and 21.26 vs 19.93 t/s (+6.7%, 8 topic chats). Not universal: margin 0 wins short chats, the speed rule wins a long prompt, so the tuner picks |
| `swaps-per-step` | `-1` | Fixed swaps per step; `-1` derives it from the measured upload time | `swaps-per-step=4` | No A/B |
| `upload-share` | `0.25`, tuned live | Uploads may take at most this share of a token's time; uploads that outlast the token stall it | `upload-share=0.5` | **Unproven, on the way out as a user knob**: GLM 2x3090, 5 vs default gave the same hit rate (64.6% vs 64.7%); `swap-lead=0` with `upload-share=0.5` was slower (20.62 vs 21.26 t/s) |
| `swap-lead-per-link` | `1` | Margin per upload link, so a GPU on a slow x4 slot gets a higher margin than one on x16; `0` uses one margin from the average upload | `swap-lead-per-link=0` | No A/B |
| `upload-now` | `1` | When a layer's router ids are known, upload some of the missed experts right then, so the GPU computes them this token while the CPU does the rest | `upload-now=0` turns it off | **Unproven, on the way out**: GLM 2x3090, a first +4.5% was noise on a 6-round rerun (off 21.80, on 22.03 / 22.25, sd about 1); without admission it churns a cold cache (18.99 vs 20.70). W10 Qwen3.6: no change |
| `lead` | `1` | The predicted layers per link are the nearest ones whose upload still lands in time: the window starts at the upload time per expert on that link over the layer time. `0`: every lookahead up to the predictor's count (the old behaviour) | `lead=0` |
| `lead-span` | `1` | Layers beyond the nearest feasible one that are also used | `lead-span=2` |
| `admit` | `2` | A missed expert may take a cache slot only after this many uses in the last 64 tokens (0: any miss). 2 beat 0 by 16% decode on GLM 3.0-bit: one recent use of a historically popular expert no longer evicts one that is hot now | `admit=0` |
| `admit-slow` | `0` | The same rule for layers on the slower link, where an upload costs 4-7x more (0: same as `admit`) | `admit-slow=4` |
| `admit-jit` | `0` | The same rule for `upload-now` uploads (0: any miss of this token may be uploaded) | `admit-jit=1` |

**Not fighting for RAM bandwidth**

| knob | default | what it does | example | evidence |
|---|---|---|---|---|
| `upload-wait` | `3` | The CPU reading experts and the upload DMA both read system RAM and together hit its limit. `3`: an upload waits while the CPU computes uncached experts; `0` off | `upload-wait=0` uploads at any time | **Proven** on GLM 3.0-bit 2x3090 short chat: `upload-wait=3` 22.10 vs 20.87 t/s (+6%, n=2); ties on long prompts. `upload-wait=2` starved uploads (19.24 t/s): never use it |
| `upload-wait-max-us` | `-1` | Longest wait per tensor copy, microseconds; `-1` = the measured mean layer time | `upload-wait-max-us=800` | No A/B |
| `upload-chunk-kb` | `-1` | With `upload-wait=3`: copy in chunks of this size and recheck the RAM budget before each; `0` = the whole tensor | `upload-chunk-kb=512` | No A/B |
| `cpu-ram-gbs`, `ram-ceiling-gbs` | `38`, `44` | Assumed RAM read rate in GB/s of the CPU alone, and of CPU plus uploads together; starting values until the start-up probe measures them, `ram-ceiling-gbs` is the ceiling for `upload-wait=3` | `cpu-ram-gbs=22,ram-ceiling-gbs=28` for a slower RAM kit | Measurement inputs, not tunables (PC1 DDR4-3200: CPU alone 37-38 GB/s, with both DMAs 44-45) |
| `wait-swaps` | `1` | The step waits for queued swaps to finish; `0` never waits and finished uploads appear at the next split | `wait-swaps=0` | No A/B recorded |

**Predictor streaming (uploads ahead of need)**. The whole group is **unproven, on the way out unless the pending corrected A/B shows a  
win**: on GLM 2x3090 the predictor raised the hit rate (chat 46.6 to 56.1%) but not tokens/s (14.2 vs 13.9), and our PC1 experiment  
lost 3-10% (hit rate +0.4-0.7 points) because uploads arrived too late and compete with the CPU for RAM bandwidth. On the W10 laptop  
(RTX 4060, Qwen3.6-35B) an earlier +27-34% shrank to 0-3% once the AVX2 CPU kernels landed. Off by default (needs `pred-top=M` or `train-every=N`).

| knob | default | what it does | example | evidence |
|---|---|---|---|---|
| `pred` | `1` | The learned predictors run at all (they exist only with `--moe pred-top=M` or `train-every=N`) | `pred-top=8,train-every=64` enables them | see above |
| `pred-iso` | on | Predicted uploads go into separate stream slots, so a wrong guess evicts nothing useful; `0` = they evict cache slots (old path) | `pred-iso=0` | see above |
| `pred-n` | `12` | Predicted candidates uploaded per target layer (over-guesses on purpose, no confidence cut) | `pred-n=6` | W10: 6 was best; offline about 80% of the uploads at 12 are wasted |
| `pred-lead` | `1` | Predict only layers far enough ahead that the upload lands in time on their link; a slow link needs more lead | `pred-lead=0` | see above; the dynamic-lead version is still being tested |
| `pred-slow` | `1` | Also stream onto layers served by a slower link | `pred-slow=0` | see above |
| `pred-tune` | `1` | The tuner tries one streaming knob at a time on real token times and keeps the faster setting | `pred-tune=0` | It turns the predictor off where it does not pay (PC1 GLM) |
| `pred-auto` | `1` | `1`: adjust lead and `pred-n` per link from the measured late share and precision; `2`: uploads per target layer = time until that layer / measured upload time per expert; `0` off | `pred-auto=2` | `pred-auto=1` +22% on W10, -3% on PC1 (before the AVX kernels) |

**Sizing**

| knob | default | what it does | example | evidence |
|---|---|---|---|---|
| `vram-reserve-mb` | `0` (= 384, or from the batch size by autotune) | VRAM in MiB kept free per GPU after the cache takes its slots, so a longer context or batch does not run out of memory (not the same as `swap-lead`) | `vram-reserve-mb=1024` | Safety limit: measured post-load growth was 54-170 MiB per GPU |

**Experimental knobs** (off by default; **unproven, on the way out**: none beat the default in our A/B tests, kept only for  
experiments and probably removed in a later release; do not rely on them):

| knob | default | what it does | example | evidence |
|---|---|---|---|---|
| `ev-cld` | `0` | An expert may only evict one with at most its own lifetime use count | `ev-cld=1` | Never won; off and not tuned |
| `pin-hot` | `0` | Pin the always-hot set of a layer on slower links | `pin-hot=0.5` | Never won in A/B |
| `slow-min-stay` | `0` | Minimum stay in steps of an expert in a slow-link layer's tier | `slow-min-stay=64` | Never won in A/B |
| `stay-bonus` | `0` | Bonus for staying in the cache | `stay-bonus=1` | +1.6% on one GLM chat run, within noise |
| `pred-keep` | `0` | A prediction may only replace a stream slot holding a lower-scored expert of this step | `pred-keep=1` | No data |
| `idle-up`, `idle-up-n` | `0`, `6` | After a token ends, stream up to `idle-up` of its misses into the first `idle-up-n` layers while the output head and sampling keep RAM idle | `idle-up=2,idle-up-n=6` | No data |
| `l3-pf`, `l3-auto` | `0`, `0` | Prefetch `l3-pf` experts per layer into the CPU's L3 cache (needs `GGML_MOE_CCX_SPLIT` and pinned threads); `l3-auto=1` tests it live | `l3-pf=2,l3-auto=1` | The static per-L3 row split it needs alone lost 19% on GLM (Ryzen 3700X); pinning itself was neutral (21.73 vs 21.53) |
| `trace-n`, `trace-skip` | `0`, `0` | Debug: record routing for this many steps, starting after this many | `trace-n=2000,trace-skip=500` | Debug only |

## Environment variables

Every knob above can also be given as `LLAMA_MOE_CACHE_<NAME>=value` (for example `LLAMA_MOE_CACHE_SWAP_LEAD=0`); the `--moe` form wins.  
These are the other fork switches. Most exist for experiments; the default is what we ship.

| variable | default | what it does |
|---|---|---|
| `LLAMA_MOE_STATE` | `~/.cache/llama.cpp/moe-state.ini` | path of the state file; `0` ignores it and never writes it (`LLAMA_MOE_CACHE_PROFILE=0` is the older switch) |
| `LLAMA_MOE_AUTO_MODE` | measured | `stock` or `cache` forces the placement; `retest` forgets the saved decision and measures again |
| `LLAMA_MOE_AUTO_PREFILL_SLOWDOWN` | `2.0` | the placement measurement drops the cache if it makes prompt processing more than this many times slower |
| `LLAMA_MOE_CACHE_POLICY` | `add` | eviction score: `add` (recent use plus lifetime share), `window` (recent use only), `hybrid`, `halve` |
| `LLAMA_MOE_CACHE_GLOBAL_WEIGHT` | `16` | weight of the lifetime share in policy `add` |
| `LLAMA_MOE_CACHE_HALVE_EVERY` | `64` | counts halve every N steps, so recent use dominates old use |
| `LLAMA_MOE_CACHE_ADOPT` | `1` | share of each layer's cache slots that may keep experts the prompt pass already put on the GPU (`1` all, `0.0625` a sixteenth; measured on GLM 12k prompt: 1/16 +0.3 hit points, 1/4 +3.8, all +11.4) |
| `LLAMA_MOE_CACHE_PROBE` | on | `0` skips the start-up probe that measures CPU read rate and each upload link |
| `LLAMA_MOE_CACHE_CPU_RAM_GBS`, `..._RAM_CEILING_GBS` | probe | override the probed RAM rates in GB/s (same as the knobs) |
| `LLAMA_MOE_CACHE_DETERMINISTIC` | `0` | `1` fixes every knob and disables tuning, for repeatable measurements |
| `LLAMA_MOE_CACHE_TUNED` | state file | start from this tuner result instead of the saved one |
| `LLAMA_MOE_CACHE_MAX_BATCH` | `31` | largest batch that still uses the cache path; bigger batches take the prompt path |
| `LLAMA_MOE_CACHE_PREFILL_D2D` | on | `0` stops prompt processing from copying experts between GPUs |
| `LLAMA_PREFILL_SPLIT` | tuned | how prompt batches spread experts over GPUs: `1` by link bandwidth, `0` the fastest GPU only, between = a mix; unset lets the tuner measure it |
| `LLAMA_MOE_CACHE_SYNC` | `1` | `0` stops the cache from waiting for the GPU between graph runs (unsafe, for experiments) |
| `LLAMA_MOE_CACHE_UPLOAD_THREADS`, `..._LINK_WORKERS` | auto, on | upload worker threads; one worker set per link so a slow x4 copy does not block the fast link (`0` = one shared set) |
| `LLAMA_MOE_CACHE_PREAD` | `0` | `1` uploads with `pread` through a pinned buffer instead of copying from the mapped memory (copying from the mapping measured faster, so off) |
| `LLAMA_MOE_CACHE_DROP`, `..._DROP_STAY` | `0`, `1024` | for models bigger than RAM: drop the RAM pages of VRAM-cached experts after they stayed cached this many steps (**unproven, on the way out**: -9% on MiMo with one GPU, off) |
| `LLAMA_MOE_CACHE_JIT_POOL`, `..._JIT_POOL_ALL` | `0`, `0` | extra slots for just-in-time uploads of slower-link layers; **unproven, on the way out**: costs 10-12% on GLM with two 3090s even when empty, off |
| `LLAMA_MOE_CACHE_STREAM_SLOTS` | `4` | stream slots per layer for predicted uploads (only with the predictor) |
| `LLAMA_MOE_CACHE_STATS` | off | set it: print the cache counters every 64 steps |
| `LLAMA_MOE_CACHE_TRACE`, `..._CTL` | off | record routing to a file; a control file whose settings are re-read while running |
| `LLAMA_THREAD_AUTOTUNE` | on | `0` turns the thread-count tuner off |
| `LLAMA_AUTO_PLACE` | `0` | `1` places threads on physical cores per L3 domain |
| `GGML_MOE_CCX_SPLIT`, `LLAMA_MOE_L3PF_CPUS` | off | split the expert rows per L3 domain; cores for the L3 prefetch (experimental) |
| `LLAMA_MOE_DEFER`, `..._DEFER_MASS` | `0`, `1.0` | experimental expert deferral (compute N selected experts during the next layer's attention). Faster (+8 to 12% on GLM) but changes the output (KL cost), so opt-in; **not proven acceptable, on the way out** |
| `GGML_CUDA_NO_PINNED` | off | never pin host memory |
| `GGML_CUDA_REGISTER_HOST` | off | register mapped host buffers with CUDA |
| `GGML_CUDA_P2P` | off | enable GPU-to-GPU peer access |

### Predictor variables

With `--moe pred-top=M` (or `train-every=N`) a small predictor guesses which experts the next layers will need.

| variable | default | what it does |
|---|---|---|
| `LLAMA_MOE_CACHE_PREDICT_AHEAD` | `2` (0..8) | how many layers ahead it predicts |
| `LLAMA_MOE_CACHE_PREDICT_MAX` | `2` | most predicted uploads queued per layer on the non-stream path |
| `LLAMA_MOE_CACHE_PREDICT_MARGIN` | `0.1` | a candidate is uploaded only if its score leads the M-th prediction by this much |
| `LLAMA_MOE_CACHE_PREDICT_MU` | `0.5` | step size of the online predictor |
| `LLAMA_MOE_CACHE_PREDICT_STRIDE` | `1` | share one predictor over this many consecutive layers |
| `LLAMA_MOE_CACHE_PREDICT_Q8` | on | `0` keeps predictor weights in fp16 instead of 8 bit |
| `LLAMA_MOE_CACHE_PREDICT_MISS` | `0.05` | predictors whose miss share stays above this are switched off |
| `LLAMA_MOE_CACHE_PREDICT_FILE` | auto | where the learned predictor is saved between runs; `0` = not saved |
| `LLAMA_MOE_CACHE_PREDICT_RA`, `..._NOHANDOFF` | | diagnostics |

## Faster loading of pinned weights: huge pages, the pool and the cache

With `--load-mode pin` (the `llama-server` default when the model fits in free RAM) the weights are read into pinned host memory so the GPU  
can copy them directly. Pinning 100 GiB is slow with ordinary 4 KiB pages. The loader has three levels, tried in this order; each falls through  
to the next without an error, so nothing here is required.

| level | needs | what it does |
|---|---|---|
| **1. Cache file** (`GGML_CUDA_HUGEFS`) | the pool and the mount below, set up once with root | the weights of a model are kept in a file in RAM that outlives the process; a later start maps it and skips the disk read |
| **2. Reserved pool pages** | the pool below | the buffer is placed on free 1 GiB (then 2 MiB) huge pages of the pool, no page-by-page work |
| **3. Hybrid** (default, no setup) | nothing | the part of the buffer the kernel can back with 2 MiB transparent huge pages (70% of the free 2 MiB blocks in `/proc/buddyinfo`) uses them, the rest 4 KiB pages; one `cudaHostRegister` call over everything |

### Measured (GLM-5.3-Flash 3.0-bit, 109.4 GiB, `--load-mode pin`, cold page cache, machine A: Ryzen 7 3700X, 125 GB DDR4-3200, 2x RTX 3090, kernel 7.1.5, driver 580.173.02)

| comparison | time to ready | runs |
|---|---|---|
| plain 4 KiB pinning (16 read threads) | 91.3 s median (87.8 to 92.7) | ABBA, n=4 each |
| hybrid (level 3) | 72.8 s (69.4 to 75.9), the pin step 60.5 s down to 44.1 s | same runs |
| warm cache file (level 1) against the pool path (level 2), one-chunk perplexity run, `LLAMA_MOE_STATE=0` | 34.5 s (34.3 to 35.7) against 56.6 s (55.6 to 60.8); the whole run 41.1 s against 62.9 s | ABBA, n=4 each |

The perplexity was identical in every run of the cache comparison (3.7227), and a cold fill, two warm loads and a no-cache load on a  
longer text all gave the same value (6.6026): the cache returns the same weights. One model, one machine, one kernel and driver: how much you gain elsewhere  
depends on how fragmented your free memory is and how fast your RAM is. The first load of a model still reads the disk (about 47 s here); only  
later starts are fast. Loading the pinned weights also holds that much RAM for as long as the process runs.

### One-time setup: `pool.sh`

Linux only (Windows uses the old path). The kernel needs 1 GiB huge pages (`CONFIG_CONTIG_ALLOC`, the CPU flag `pdpe1gb`); tested on kernel 7.1.5 only.

```sh
sudo ./pool.sh mount              # the most the machine can spare: total RAM minus a margin (default max(24 GiB, 20% of RAM), HUGEFS_MARGIN_GIB=N to change)
sudo ./pool.sh mount 100G         # exactly that many GiB (a number with G), or --pages N, or the path of a model file to size it for that model
sudo ./pool.sh unmount            # delete the cached models, unmount, give the pages back
```

It reserves the 1 GiB pages (it drops the page cache and compacts memory first so nothing has to be migrated), and mounts a hugetlbfs at  
`/mnt/huge1g` owned by you. Root is only needed to change the pool or the mount; with a big enough pool and the mount in place it runs as a normal user.  
The pool stays reserved until `unmount` (or a reboot), so that RAM is not available to other programs meanwhile: this is a setup for a machine  
that serves one model, not for a desktop. Reserve once; the loader never grows or shrinks it. For a model that only partly sits in host memory  
(layers on the GPUs) the cache holds just that part, so give the size by hand if the model's file size exceeds the limit.

Then start the fork with the cache on:

```sh
GGML_CUDA_HUGEFS=/mnt/huge1g llama-server -m model.gguf --load-mode pin
```

The first load of a model creates `<key>.w.part`, fills it while reading the weights and renames it to `<key>.w` when complete. The key is made  
from the model file (device, inode, size, modification time) and the layout of the tensors in the buffer, so a changed file or a different  
split of layers between GPU and RAM gets its own cache file. When the pool is full, the next fill deletes the least recently used cache file  
that no process has mapped (a running process holds a lock on its file, so a model in use is never evicted); if there is still no room it  
silently uses level 2 or 3.

### Do not boot with `hugetlb_cma=`

Huge pages taken from a CMA area cannot be pinned for the GPU: the NVIDIA driver pins with `pin_user_pages(FOLL_LONGTERM)`, the kernel refuses  
that for CMA pages unless it can migrate them, and a 1 GiB page inside the area has nowhere to go. `cudaHostRegister` then fails with  
`invalid argument` (we measured this for every flag and chunk size). `pool.sh mount` refuses to run if that option is on the kernel command  
line. Allocate the pool at run time, as the script does, from ordinary memory.

### Environment variables for loading

| variable | default | what it does |
|---|---|---|
| `GGML_CUDA_HUGEFS` | off | mount point of the cache (`/mnt/huge1g`); unset or without a mount the loader uses level 2 or 3 |
| `GGML_CUDA_THP_SHARE` | `0.7` | share of the free 2 MiB blocks used for transparent huge pages in level 3. `0.9` was faster (70 s) in five runs but once froze a load for more than 13 minutes while memory was heavily used; `0` is plain 4 KiB pages |
| `GGML_CUDA_REG_CHUNK_MB` | whole buffer | register the buffer in chunks of this many MiB. **Do not use:** a tensor that straddles two registrations makes `cudaMemcpyAsync` fail with `invalid argument` at the first prompt batch (measured: perplexity crashed with 4 GiB chunks, ran with one registration) |
| `GGML_CUDA_NO_PINNED` | off | never pin the weights |
| `LLAMA_LOAD_THREADS` | half the CPU threads, at most 16 | threads that read the model file into the buffer; `1` keeps the sequential reader |

### If something goes wrong

- `only N of M pages could be allocated`: memory is too fragmented or too full; stop other programs, run `pool.sh unmount`, mount again with a smaller size, or reboot.
- No `hugetlbfs cache` line in the log: the mount is missing, the pool is too small for the file, or another process is filling the same model; the load used level 2 or 3.
- A load that stalls for minutes while the machine is swapping: lower `GGML_CUDA_THP_SHARE` (or set it to `0`), and do not load a second large model while a cache file holds most of the RAM.

## The state file

`~/.cache/llama.cpp/moe-state.ini` (`LLAMA_MOE_STATE=PATH` moves it, `LLAMA_MOE_STATE=0` ignores it and never writes) holds one  
`[model name + size]` section per model:

| line | content |
|---|---|
| `hot.N = c0 c1 ...` | lifetime use count of every expert of layer N. At start the most used experts are loaded into the cache first, so the first prompt is already warm |
| `tuned.lK = NAME=value ...` | what the self-tuner settled on, for K upload links (GPUs) |
| `place.g<gpus>.<GPU hash>.v3.cache`, `.stock`, `.decided` | measured prompt and decode speed of cache and stock placement for this GPU set, and which one won |
| `place.g<gpus>.<GPU hash>.v3.link<k>` | the link bandwidth (GB/s, rounded to half octaves) seen when the placement was measured; a later probe that differs by a factor of 2 or more drops this hardware's placement records |

Counts only seed the start; during a run the cache scores by recent use. It is plain text, safe to edit or delete.

## Several GPUs and MTP

- The cache is sized per GPU from its free VRAM and the experts spread over all GPUs; each GPU keeps its own slots and upload  
  link. A card on a slow slot (x4) helps less, its uploads take 4 to 7 times longer, and prompt processing goes to the fastest  
  link. `-ts`, `-dev` and `-sm` work as in stock; see [multi-GPU usage](docs/multi-gpu.md).
- MTP works together with the cache: pass `-md` and `--spec-type draft-mtp` as in the MTP section above. The draft head is  
  small and stays in VRAM, the cache serves the main model.

