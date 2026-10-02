# Fork options: flags, `--moe` settings, tuning knobs, environment variables, state file

Everything this fork adds to stock llama.cpp. Settings you give are used as given and are never auto-tuned. Defaults are those of
the release this file ships with.

## Command-line flags

| flag | environment form | what it does |
|---|---|---|
| `--moe KEY=VAL,...` | `LLAMA_ARG_MOE` | the expert cache settings and knobs, below |
| `--moe-expert-cache N` | `LLAMA_ARG_MOE_EXPERT_CACHE` | expert slots per layer in VRAM; `-1` sizes them from free VRAM, `0` turns the whole fork off (same as `--moe cache=0`) |
| `--prefetch-experts-slots N` | | staging slots for host-to-GPU prefetch of big batches (same as `--moe prefetch-slots=N`) |
| `-at on\|off`, `--autotune` | `LLAMA_ARG_AUTOTUNE`, `LLAMA_AUTOTUNE=0` | self-tuning of cache knobs, thread counts and the cache-or-stock placement; `off` = fixed defaults, nothing measured or saved (same as `--moe autotune=0`) |
| `-lm pin\|mmap\|dio`, `--load-mode` | `LLAMA_ARG_LOAD_MODE` | `pin`: weights in pinned RAM (server default when the model fits in RAM, faster prompts); `mmap`: memory-mapped, for models bigger than RAM; `dio`: direct IO |
| `-md FILE --spec-type draft-mtp` | | MTP draft head (Qwen3.8-Flash-Next, GLM-5.3-Flash), see the README |
| `-t N`, `-tb N` | | fixed decode / prompt thread counts (otherwise tuned) |

## `--moe` settings (command line)

`--moe` takes comma separated `name=value` pairs, names are case-insensitive and `_` equals `-`; the same string works as
`LLAMA_ARG_MOE=...`. A setting you give is used as given and never self-tuned.

| key | meaning |
|---|---|
| `cache=N` | expert slots per layer in VRAM; `-1` sizes them from free VRAM, unset is automatic, `0` is the whole fork off (stock behaviour) |
| `prefetch-slots=N` | staging slots for host-to-GPU prefetch |
| `inserts=N` | most expert uploads per layer and decode step |
| `window=N` | tokens of recent use the cache scores experts by (default 64) |
| `predict=M`, `train=N` | prefetch the experts the router is likely to pick in the next layers (top M); `train=N` also trains the learned predictor every N tokens |
| `autotune=0` | no self-tuning (same as `-at off`) |

Any other name is a tuning knob of the cache engine (`--moe gate=3,margin=0`). A knob you set is never self-tuned; a name the
engine does not know is logged as `unknown key` and ignored. Values are numbers; `0`/`1` are off/on.

The cache keeps some experts in VRAM (the GPU computes those); any other expert is computed by the CPU or uploaded over PCIe first.
The knobs trade CPU work against uploads. Every example below is a complete option: `llama-server -m model.gguf --moe <example>`.

**Swap decisions**

| knob | default | what it does | example |
|---|---|---|---|
| `MARGIN` | `-1` | An upload costs link time and only pays back if the expert is used again, so a missed expert must beat the one it evicts by this many recent uses. `-1` = 0 (swap whenever it scores higher), `-2` = from speeds (CPU read rate / link rate is about the uses an upload needs to pay for itself), `n` = fixed | `margin=0` swaps eagerly; `margin=2` swaps rarely |
| `BUDGET` | `-1` | Fixed swaps per step; `-1` derives it from the measured upload time | `budget=4` |
| `SWAP_FRAC` | `0.25`, tuned live | Uploads may take at most this share of a token's time; uploads that outlast the token stall it | `swap_frac=0.5` lets uploads use half the token time |
| `LINK` | `1` | Margin per upload link, so a GPU on a slow x4 slot gets a higher margin than one on x16; `0` uses one margin from the average upload | `link=0` |
| `JIT` | `1` | When a layer's router ids are known, upload some of the missed experts right then, so the GPU computes them this token while the CPU does the rest; the number is chosen so CPU and link finish together | `jit=0` turns it off |

**Not fighting for RAM bandwidth**

| knob | default | what it does | example |
|---|---|---|---|
| `GATE` | `3` | The CPU reading experts and the upload DMA both read system RAM and together hit its limit. `3`: an upload waits while the CPU computes uncached experts; `0` off | `gate=0` uploads at any time |
| `GATE_MAX_US` | `-1` | Longest wait per tensor copy, microseconds; `-1` = the measured mean layer time | `gate_max_us=800` |
| `CHUNK_KB` | `-1` | With `GATE=3`: copy in chunks of this size and recheck the RAM budget before each; `0` = the whole tensor | `chunk_kb=512` |
| `CPU_GBS`, `DDR_GBS` | `38`, `44` | Assumed RAM read rate in GB/s of the CPU alone, and of CPU plus uploads together; starting values until measured, `DDR_GBS` is the ceiling for `GATE=3` | `cpu_gbs=22,ddr_gbs=28` for a slower RAM kit |
| `WAIT` | `1` | The step waits for queued swaps to finish; `0` never waits and finished uploads appear at the next split | `wait=0` |

**Predictor streaming (uploads ahead of need)**

| knob | default | what it does | example |
|---|---|---|---|
| `STREAM` | on | Predicted uploads go into separate stream slots, so a wrong guess evicts nothing useful; `0` = they evict cache slots (old path) | `stream=0` |
| `STREAM_M` | `12` | Predicted candidates uploaded per target layer (over-guesses on purpose, no confidence cut) | `stream_m=6` |
| `OFFSET` | `1` | Predict only layers far enough ahead that the upload lands in time on their link; a slow link needs more lead | `offset=0` |
| `STREAM_SLOW` | `1` | Also stream onto layers served by a slower link | `stream_slow=0` |
| `PREDICT` | `1` | The learned predictors run at all (they exist only with `--moe predict=M` or `train=N`) | `predict=8,train=64` enables them |
| `SELF_TUNE` | `1` | The tuner tries one streaming knob at a time on real token times and keeps the faster setting | `self_tune=0` |
| `AUTO` | `1` | `1`: adjust lead and `STREAM_M` per link from the measured late share and precision; `2`: uploads per target layer = time until that layer / measured upload time per expert; `0` off | `auto=2` |

**Sizing**

| knob | default | what it does | example |
|---|---|---|---|
| `MARGIN_MB` | `0` (= 384, or from the batch size by autotune) | VRAM in MiB kept free per GPU after the cache takes its slots, so a longer context or batch does not run out of memory (not the same as `MARGIN`) | `margin_mb=1024` |

**Experimental knobs** (off by default; none beat the default in our A/B tests, kept only for experiments, and probably removed
in a later release; do not rely on them):

| knob | default | what it does | example |
|---|---|---|---|
| `BIG` | `0` | An expert may only evict one with at most its own lifetime use count; never won, off and not tuned | `big=1` |
| `HOT_FRAC` | `0` | Pin the always-hot set of a layer on slower links | `hot_frac=0.5` |
| `SLOW_STAY` | `0` | Minimum stay in steps of an expert in a slow-link layer's tier | `slow_stay=64` |
| `STICKY` | `0` | Bonus for staying in the cache; +1.6% on one chat run, within noise | `sticky=1` |
| `SLOTKEEP` | `0` | A prediction may only replace a stream slot holding a lower-scored expert of this step | `slotkeep=1` |
| `TBP`, `TBP_LAYERS` | `0`, `6` | After a token ends, stream up to `TBP` of its misses into the first `TBP_LAYERS` layers while the output head and sampling keep RAM idle | `tbp=2,tbp_layers=6` |
| `L3PF`, `AUTO_L3` | `0`, `0` | Prefetch `L3PF` experts per layer into the CPU's L3 cache (needs `GGML_MOE_CCX_SPLIT` and pinned threads); `AUTO_L3=1` tests it live | `l3pf=2,auto_l3=1` |
| `TRACE`, `TRACE_AFTER` | `0`, `0` | Debug: record routing for this many steps, starting after this many | `trace=2000,trace_after=500` |


## Environment variables

Every knob above can also be given as `LLAMA_MOE_CACHE_<NAME>=value` (for example `LLAMA_MOE_CACHE_MARGIN=0`); the `--moe` form wins.
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
| `LLAMA_MOE_CACHE_CPU_GBS`, `..._DDR_GBS` | probe | override the probed RAM rates in GB/s (same as the knobs) |
| `LLAMA_MOE_CACHE_DETERMINISTIC` | `0` | `1` fixes every knob and disables tuning, for repeatable measurements |
| `LLAMA_MOE_CACHE_TUNED` | state file | start from this tuner result instead of the saved one |
| `LLAMA_MOE_CACHE_MAX_BATCH` | `31` | largest batch that still uses the cache path; bigger batches take the prompt path |
| `LLAMA_MOE_CACHE_PREFILL_D2D` | on | `0` stops prompt processing from copying experts between GPUs |
| `LLAMA_PREFILL_SPLIT` | tuned | how prompt batches spread experts over GPUs: `1` by link bandwidth, `0` the fastest GPU only, between = a mix; unset lets the tuner measure it |
| `LLAMA_MOE_CACHE_SYNC` | `1` | `0` stops the cache from waiting for the GPU between graph runs (unsafe, for experiments) |
| `LLAMA_MOE_CACHE_UPLOAD_THREADS`, `..._LINK_WORKERS` | auto, on | upload worker threads; one worker set per link so a slow x4 copy does not block the fast link (`0` = one shared set) |
| `LLAMA_MOE_CACHE_PREAD` | `0` | `1` uploads with `pread` through a pinned buffer instead of copying from the mapped memory |
| `LLAMA_MOE_CACHE_DROP`, `..._DROP_STAY` | `0`, `1024` | for models bigger than RAM: drop the RAM pages of VRAM-cached experts after they stayed cached this many steps (-9% on MiMo with one GPU, off) |
| `LLAMA_MOE_CACHE_JIT_POOL`, `..._JIT_POOL_ALL` | `0`, `0` | extra slots for just-in-time uploads of slower-link layers; costs about 10% on GLM with two 3090s even when empty, off |
| `LLAMA_MOE_CACHE_STREAM_SLOTS` | `4` | stream slots per layer for predicted uploads (only with the predictor) |
| `LLAMA_MOE_CACHE_STATS` | off | set it: print the cache counters every 64 steps |
| `LLAMA_MOE_CACHE_TRACE`, `..._CTL` | off | record routing to a file; a control file whose settings are re-read while running |
| `LLAMA_THREAD_AUTOTUNE` | on | `0` turns the thread-count tuner off |
| `LLAMA_AUTO_PLACE` | `0` | `1` places threads on physical cores per L3 domain |
| `GGML_MOE_CCX_SPLIT`, `LLAMA_MOE_L3PF_CPUS` | off | split the expert rows per L3 domain; cores for the L3 prefetch (experimental) |
| `LLAMA_MOE_DEFER`, `..._DEFER_MASS` | `0`, `1.0` | experimental expert deferral (compute N selected experts during the next layer's attention) |
| `GGML_CUDA_NO_PINNED` | off | never pin host memory |
| `GGML_CUDA_REGISTER_HOST` | off | register mapped host buffers with CUDA |
| `GGML_CUDA_P2P` | off | enable GPU-to-GPU peer access |

### Predictor variables

With `--moe predict=M` (or `train=N`) a small predictor guesses which experts the next layers will need.

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

## The state file

`~/.cache/llama.cpp/moe-state.ini` (`LLAMA_MOE_STATE=PATH` moves it, `LLAMA_MOE_STATE=0` ignores it and never writes) holds one
`[model name + size]` section per model:

| line | content |
|---|---|
| `hot.N = c0 c1 ...` | lifetime use count of every expert of layer N. At start the most used experts are loaded into the cache first, so the first prompt is already warm |
| `tuned.lK = NAME=value ...` | what the self-tuner settled on, for K upload links (GPUs) |
| `place.g<gpus>x<MiB>.v1.cache`, `.stock`, `.decided` | measured prompt and decode speed of cache and stock placement for this GPU set, and which one won |

Counts only seed the start; during a run the cache scores by recent use. It is plain text, safe to edit or delete.

## Several GPUs and MTP

- The cache is sized per GPU from its free VRAM and the experts spread over all GPUs; each GPU keeps its own slots and upload
  link. A card on a slow slot (x4) helps less, its uploads take 4 to 7 times longer, and prompt processing goes to the fastest
  link. `-ts`, `-dev` and `-sm` work as in stock; see [multi-GPU usage](docs/multi-gpu.md).
- MTP works together with the cache: pass `-md` and `--spec-type draft-mtp` as in the MTP section above. The draft head is
  small and stays in VRAM, the cache serves the main model.

