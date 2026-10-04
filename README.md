# llama.cpp fork: 1.7x to 2.4x faster decode on MoE models bigger than your VRAM

The newest mixture-of-experts models (GLM-5.3-Flash, MiMo-V2.6-Flash, Qwen3.8-Flash-Next, Qwen3.6) are far bigger than a gaming GPU.  
This fork keeps their experts in RAM and turns the free VRAM into a live cache of the experts the model is using; the GPUs compute
the cached ones and the CPU the rest, at the same time. No special switches needed.  

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
llama-cli    -m model.gguf -p "hello"
```

The decode speedup is largest when a third to a half of a MoE model fits in VRAM (1.9x to 2.4x), fades above about 55% and is 1.0x once it fits; dense models gain nothing.  
**Highlights (all measured, all in the table below):**

- **Up to 2.4x faster generation** on models bigger than your VRAM: MiMo-V2.6-Flash 132 GB on two 24 GB cards, 4.6 to 10.9 tokens/s.  
- **1.9x generation and 2.4x prompt processing on a laptop**: Qwen3.6-35B on an 8 GB RTX 4060, 31.6 to 60.0 and 41.2 to 100.9 tokens/s.  
- **3x faster prompt processing with no GPU at all**: Qwen3.6-35B on a Ryzen 5 3600, 10.1 to 31.5 tokens/s (1.8x generation).  
- **Reloads of a 100 GB model take half the time** with the optional huge-page pool: 43 s instead of 88 to 95 s.  
- **When the cache cannot help, it falls back to stock speed by measurement**, so a model that fits or a machine that gains nothing runs at stock speed (1.0x rows below).  

Gain over stock llama.cpp, best first (t/s = generated tokens per second on top, pp = prompt processing tokens per second below, in every cell with two numbers; four gains per model: short prompt t/s and pp, long prompt t/s and pp; both are stock's best of 4 runs against ours best of 4, no warm-up (short: four different 'smallest html game' prompts; long: four different edit instructions over one long source file); stock and ours are the numbers of the earlier same-prompt test until they are re-measured; *italic* gains are from the earlier tests (best run of ours against best run of stock from the run log: short = one short prompt repeated, long = the 12k-token prompt, or 2.2k for the MTP row); they are placeholders until the new tests (the plain numbers) replace them, `-` = not measured yet; the build column is the build the number was taken on):

| model | hardware | in VRAM | stock<br>t/s pp | ours<br>t/s pp | short gain<br>t/s pp | long gain<br>t/s pp | build |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Flash-<br>RL-IQ3_XXS | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 36% | 4.6<br>156 | **10.9**<br>112 | *2.6x*<br>*1.0x* | *2.3x*<br>*0.8x↓* | earlier |
| GLM-5.3-Flash-<br>GSQ-RCO-3.5bit | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 35% | 6.9<br>- | **15.1**<br>- | *2.7x*<br>*2.4x* | -<br>- | b11707 |
| qwen36 | 4060 8G 4/8<br>8945HS 32G 48G/s | 73%<br>slow CPU | 31.6<br>41.2 | **60.0**<br>100.9 | *1.9x*<br>*2.6x* | -<br>- | b12030 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 45% | 11.9<br>- | **22.4**<br>- | *2.2x*<br>*1.4x* | -<br>- | earlier |
| qwen36 | 3600 A520 64G | 0%<br>CPU only | 6.3<br>10.1 | 11.1<br>34.1 | *1.8x*<br>*3.7x* | -<br>- | b12040 |
| Qwen3.8-Flash-<br>Next-UD-IQ4_XS | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 55% | 27.7<br>500 | 46.5<br>538 | *2.0x*<br>*1.5x* | *1.6x*<br>*1.1x* | earlier |
| Qwen3.8-27B-MTP-Q5_K_M | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | fits<br>2.2k | 31.3<br>599 | 51.1<br>792 | *1.0x*<br>*1.0x* | *1.6x*<br>*1.3x* | earlier |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 66%<br>3 of 4 | 20.1<br>- | 32.6<br>- | *1.6x*<br>- | -<br>- | b11707+ |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ3_S | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 58% | 43.9<br>- | 57.1<br>- | 1.3x<br>1.1x | -<br>- | b11707 |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ1_M | 4060 8G 4/8<br>8945HS 32G 48G/s | 13% | 13.5<br>18.3 | 17.3<br>21.6 | *1.3x*<br>*1.2x* | -<br>- | b12030 |
| GLM-5.3-Flash-<br>GSQ-RCO-3.0bit | 4x3090 4/16<br>7B12 256G 74G/s | 88%<br>4 GPUs | 24.7<br>- | 31.5<br>- | *1.3x*<br>- | -<br>- | b11707+ |
| Qwen3.8-27B-IQ4_NL | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | fits<br>dense | 45.1<br>- | 45.0<br>- | 1.0x<br>1.1x | *1.0x*<br>*1.0x* | earlier |
| Qwen3.8-27B-MTP-Q5_K_M | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | fits<br>short | 78.3<br>- | 77.0<br>- | *1.0x*<br>*1.0x* | -<br>- | earlier |
| Qwen3.8-Flash-Next-<br>GSQ-RCO-IQ1_M | 2x3090 4/16+4 X570<br>3700X 125G 44G/s | 87% | 69.1<br>- | 68.3<br>40 | 1.0x<br>1.0x | -<br>- | b12040 |
| Qwen3.8-27B-GSQ-<br>RCO-IQ3_S-mtp | 3600 A520 64G | 0%<br>CPU only | 1.7<br>5.6 | 1.6<br>5.8 | *0.9x↓*<br>*1.0x* | -<br>- | b12030 |
| any model that fits | any | 100% | same<br>same | same<br>same | 1.0x<br>1.0x | -<br>- | any |

Two cells are slower than stock and say so: prompt processing of MiMo on long prompts (0.7x) and generation of the CPU-only 27B IQ3_S (0.9x, 1.6 against 1.7 tokens/s).  
Hardware, in this order: GPUs with PCIe generation x lanes (4/16 = PCIe 4.0 x16, 4/16+4 = a second card on x4, same generation), CPU, RAM (G = GB, channels, type and speed), measured RAM bandwidth.  

## What you get

Single stream, temperature 0, model in RAM. "Stock" and "upstream" mean stock llama.cpp.  

Rows marked `earlier` or `b11707` keep their old numbers and are being re-measured on the current release; upstream stock is the release b11379 on the laptop and upstream 836d57176 built the same way as the fork on the Ryzen 5 3600.  

The 4x RTX 3090 EPYC rows: upstream is the downloaded release b11323 (a source build of def4d406a gave 24.8 and 20.1).  
the fork is b11707 with the placement change in this branch (any model that does not fit takes the cache);  
the model is 85% in VRAM on 4 GPUs, so the first run only matches stock placement.  
Those numbers are from one session on a rented 4 gpu box (raw logs not kept, not in run-history.csv).  

Older rows: GLM and MiMo were measured on release-candidate builds (MiMo also on b11509) before the last placement and thread commits, Qwen3.6 on the release binary. Every run behind these numbers (commit, build, machine, settings) is in
[`tools/bench/run-history.csv`](tools/bench/run-history.csv).  

### All measured rows (kept)

The rows behind the table, with the machine letters A to D, as measured earlier (a number is only replaced by a better measured one):  

Decode tokens/s, single stream, temperature 0, model in RAM. "Upstream" is stock llama.cpp.  

Machine A: 2x RTX 3090 (PCIe 4.0 x16 + chipset x4), Ryzen 7 3700X, 125 GB DDR4-3200.  
Machine B, a laptop: RTX 4060 8 GB, Ryzen 9 8945HS, 32 GB LPDDR5X-6400.  
Machine C: no GPU, Ryzen 5 3600, 64 GB DDR4-3200.  
Machine D, rented: 4x RTX 3090 (PCIe 4.0 x16 each), EPYC 7B12 (64 cores), 256 GB DDR4 on 4 of 8 memory channels (74 GB/s read measured).  

GLM 3.5-bit, IQ3_S and IQ1_M are the first run of build b11707 against fresh upstream def4d406a 
(no discarded run before it); the other rows are hot runs of earlier builds.  
Rows marked **b11988** are re-measured on this release (commit 3db71c4ee): one discarded run, then the 4th start; upstream is the release b11379 on B and upstream 836d57176 built the same way as the fork on C;  
the other rows keep their old numbers and are being re-measured, the build column says on which build each was taken.  

| model (size) | machine | test | upstream | this fork | | build |
|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** 3.0-bit (106 GB) | A | short chat, decode | 11.9 | **22.4** | **1.9x** | earlier |
| **GLM-5.3-Flash** 3.0-bit (117.5 GB file) | D | short chat, decode, first run | 24.7 | 25.2 | 1.0x | b11707+ |
| | | same, second run (saved state) | 24.7 | **31.5** | **1.3x** | b11707+ |
| | | same, second run, `-t 16` | 24.7 | **33.8** | **1.4x** | b11707+ |
| | | same, 3 of the 4 GPUs, second run (best ratio) | 20.1 | **32.6** | **1.6x** | b11707+ |
| **MiMo-V2.6-Flash** IQ3_XXS (132 GB, bigger than RAM) | A | short chat, decode | 4.6 | **10.9** | **2.4x** | earlier |
| | | 12k-token prompt, decode | 4.2 | **9.1** | **2.2x** | earlier |
| | | 12k-token prompt, processing | 156 | 112 | 0.7x | earlier |
| **Qwen3.8-Flash-Next** UD-IQ4_XS (88 GB) * | A | short chat, decode | 27.7 | **46.5** | **1.7x** | earlier |
| | | 12k-token prompt, decode / processing | 25.3 / 500 | **42.3 / 538** | 1.7x / 1.1x | earlier |
| **GLM-5.3-Flash** 3.5-bit (137 GB, bigger than RAM) | A | short tetris prompt, 100 tokens, decode (text-dependent) | 6.9 | **15.1** | **2.2x** | b11707 |
| **Qwen3.8-Flash-Next** GSQ IQ3_S (83 GB) | A | short tetris prompt, 100 tokens, decode | 43.9 | **57.1** | **1.3x** | b11707 |
| **Qwen3.8-Flash-Next** GSQ IQ1_M (55 GB, barely over 48 GB VRAM) | A | same | 69.1 | 67.5 (picks stock) | 1.0x | b11707 |
| | B | same | 12.0 | **16.0** | **1.3x** | **b11988** |
| | B | same, prompt processing | 18.1 | 19.6 | 1.1x | **b11988** |
| **Qwen3.6-35B-A3B** Q2_0 (11 GB, on an 8 GB GPU) | B | short tetris prompt, 100 tokens, decode | 29.2 | **59.3** | **2.0x** | earlier |
| | C | same, CPU only (AVX Q2_0 kernels) | 6.3 | **11.3** | **1.8x** | earlier |
| Qwen3.8-27B IQ4_NL, dense (fits VRAM) | A | same | 45.1 | 45.0 | 1.0x | earlier |
| Qwen3.8-27B IQ3_S, dense, CPU only | C | same | 1.7 | 1.6 | 1.0x | earlier |
| Qwen3.8-27B Q5_K_M with MTP (`--spec-type draft-mtp`), fits VRAM | A | same | 78.3 | 77.0 | 1.0x (38.6 without MTP) | earlier |
| | | 2.2k-token prompt, decode / processing | 31.3 / 599 | **51.1 / 792** | 1.6x / 1.3x | earlier |
| Models that fit in VRAM | any | anything | same | same | 1.0x (cache off) | any |

D: upstream is the downloaded release b11323 (a source build of def4d406a gave 24.8 and 20.1).  
the fork is b11707 with the placement change in this branch (any model that does not fit takes the cache);  
the model is 85% in VRAM on 4 GPUs, so the first run only matches stock placement.  
D numbers are from one session on a claud rented 4 gpu box (raw logs not kept, not in run-history.csv).  

\* from the previous release. GLM and MiMo were measured on release-candidate builds (MiMo also on b11509) before the last placement and thread commits, Qwen3.6 on the release binary. Every run behind these numbers (commit, build, machine, settings) is in
[`tools/bench/run-history.csv`](tools/bench/run-history.csv).  

### Why the gain depends on how much of the model fits

**Methodology: a coding session, not one repeated prompt.** A model run on the same prompt over and over at temperature 0 routes to the same experts every time, which flatters any expert cache. So the stricter test (`python3 tools/run.py bench -t edit4`) feeds one long source file (about 2700 tokens, `tools/bench/src_3k.cpp`) with four different edit instructions, one per run:  
"add a function that counts the lines", "delete the code that is never used", "add a short comment above every function", "rename the longest function and update its callers" (128 tokens generated, temperature 0).  
The short test is the same idea at CLI size (`-t game4`): "write smallest html tetris game", then racing, shooter and snake, 100 tokens each, so the task, the language and the stack stay the same (as in a real session) and only the game changes.  
Every build gets the same instruction at the same repetition, builds alternate in order, each build has one discarded warm-up run with its own separate instruction, and the table reports **stock's best of the four runs against ours best of the four**.  
Introduced 2026-10-04 (commit `11d9fd2b7`; `game4` the same day). The tests are the `edit4` and `game4` branches and the `EDIT_INSTR` and `GAME_WORDS` lists in [`tools/run.py`](tools/run.py), the snippet is [`tools/bench/src_3k.cpp`](tools/bench/src_3k.cpp), every run is a row (column `test` = `edit4`) in [`tools/bench/run-history.csv`](tools/bench/run-history.csv), and the raw logs of the runs behind this release's checks (regression tests, the IQ3_S diagnosis, the model load-time test) are in [`tools/bench/logs/`](tools/bench/logs/), one folder per experiment named with its date and time.  
**The learned state is never deleted, not between runs and not at the start** (`~/.cache/llama.cpp/moe-state.ini` and the profile files next to it): in normal use it keeps accumulating, which is where the cache is strongest, so every rerun starts from what the earlier runs learned. An old or wrong state that leaves a model stuck on low numbers is a bug in the fork, and is fixed in the code (the placement records are versioned, so records of an older release are ignored and measured again); only the regression tests give every arm a state file of its own, so that one arm cannot learn from another.  
So the table compares each side at its best in normal use: stock's best run against ours with the state it has accumulated (the preheated cache and profiles of a model you have used a few times). A model's first starts are slower while the state learns: IQ3_S on the 3090s ran 40, 43 and 41 to 53 tokens/s on its first three starts and 53 to 57 from the fourth on, against 44 for stock ([run logs](tools/bench/logs/iq3s-diag/)).  
The cells of the table above were taken with the simpler repeated-prompt test (the 4th run of the same prompt) and are being repeated with these two; a row says which test it used once it has been re-measured. The repeated-prompt test is still what catches a stuck placement decision fastest, so both stay.  

**Reproduce and check these numbers.** Every run behind the table (date, commit, build, machine, model, command line, speed) is a row in [`tools/bench/run-history.csv`](tools/bench/run-history.csv), failed and cold runs included.  
`tools/experiments/release-gate/readme-rerun.sh <fork build> <stock build>` repeats the cells (one discarded run, then the recorded runs, builds in alternating order), `tools/experiments/release-gate/readme-compare.py` prints them next to the numbers above, and `python3 tools/run.py bench --help` runs any single cell. The table is sorted by gain, not by how good a row looks, and the slow rows stay in.  

### Why the gain depends on how much of the model fits

Time per token is a fixed floor plus the cost of what is served from slow memory: `T = T0 + m x (fraction served from the CPU)`. Stock serves from the CPU every weight
that does not fit in VRAM; the fork serves only its cache misses, and its hit rate is far above the cached fraction. Fitted on machine D, GLM 3.0-bit (ms per token):

| GPUs (VRAM / model) | weights off VRAM | stock | fork: hit, model `28.3 + 46 x miss` | fork over stock |
|---|---|---|---|---|
| 4 (88%) | 15% | 40.5 ms (24.7 t/s) | 93.8%: 31.7 ms measured (31.5 t/s), 31.1 modelled | 1.3x |
| 3 (66%) | 36% | 49.8 ms (20.1 t/s) | 94.9%: 30.6 ms measured (32.6 t/s) | 1.6x |
| 2 (44%) | 57% | about 59 ms (about 17 t/s), modelled `34.0 + 44 x fraction`, not run | 87.9%: 32.9 ms measured (30.4 t/s), 33.9 modelled | about 1.7x |
| 1 (22%) | 79% | about 68 ms (about 15 t/s), modelled, not run | 66.9%: 43.7 ms measured (22.9 t/s) | about 1.6x |

- **The gain is the CPU reads the cache removes.** It is largest when stock leaves a large share of the model on a slow CPU: on machine A (2 GPUs, 44 GB/s RAM) a model 2 to 3 times the VRAM gives 1.9x to 2.4x.  
- **More fast memory has diminishing returns.** The slow-memory cost per unit is about the same for both (44 to 46 ms), so the fork's lead is only the difference between the share stock leaves on the CPU and its own miss rate. As the VRAM share grows both approach their floors (fork about 28 ms, stock about 34 ms, ratio about 1.2x). On 4 GPUs the fork is at 94% of its floor, and the 2, 3 and 4 GPU runs differ by 7%.  
- **A faster CPU lowers the gain,** because the slow-memory term shrinks (the 4-GPU EPYC box reads 74 GB/s against 44 GB/s on the 3700X box).  
- **A model that fits in VRAM gains nothing** (the cache is off, 1.0x).  

Four points from one session with different output texts: a good explanation, not a proof. The 2 and 1 GPU stock figures are model predictions, not measurements.  

## Too many knobs? Smart autotune picks them

Running a model bigger than your VRAM well means choosing placement, cache size, upload schedule, batch size, thread counts, the VRAM margin. Instead of a page of flags, the
fork measures your machine and tunes most of those while you use it. The defaults are chosen on your machine, not hard-coded:

- **Cache or stock.** If the model fits in VRAM it is placed exactly like stock llama.cpp. A model that does not fit is measured: the first start
  runs stock placement, the next one tries the cache, and from then on the faster of the two is used. Which one wins depends on your machine and your requests,
  nothing is decided from the model size. Every few starts the stock placement is measured again.  
- **Self-tuning on real token times.** The cache policy and the upload schedule are adjusted while you use it. The decode and prompt thread counts start from
  a formula (cores minus one per GPU) and the tuner moves them a few threads at a time; on a many-core host `-t 16` can be a better start (on a 64-core machine the
  difference was under 2%). A setting that does not help is dropped, a setting you fix yourself is never touched.  
- **It remembers.** What it learned per model (hot experts, tuned settings, cache-or-stock) is kept in one file,
  `~/.cache/llama.cpp/moe-state.ini`, so the next start, even a one-shot short, begins from it. Delete the file to start over.  
- **Research: hot experts by topic (`--moe emb=1`).** At exit one line is appended to `embhot.csv` in the working directory: `model,embhex,hots`, the last layer embedding of the last token (1 byte per float as hex) and the hot experts of the run (counts per layer). Nothing is read or written while running, so there is no speed cost, and the file only ever grows by appending. Collect it over time and compare how the hot experts diverge between topics; share it if you do. This is the groundwork for topic-specific hot caches: a start could preload the hot experts of the closest earlier topic instead of one average map (not done yet; whether this embedding is a good topic key is still to be measured).  
- **Hot start from ours.** [`moe-state.ini`](moe-state.ini) in the root of this repo is our learned state (hot experts of GLM 3.0/3.5-bit, MiMo, Qwen3.8 IQ1_M/IQ3_S/IQ4_XS, Qwen3.6, OLMoE). If you run one of these models and have no state file yet, copy it to `~/.cache/llama.cpp/moe-state.ini` for a fast hot start from the first run. Do not overwrite your own file: yours holds what your hardware learned.It holds only the hardware-independent hot experts (tagged with arch and expert shape); stock-or-cache placement and tuning are measured on your machine. A model file with no record of its own is seeded from a same-name or same-family record.  
- **The first run of a model is slower, but just once.** The first start of a model runs like stock while the fork captures which experts are hot into the state file
  (`moe-state.ini`). If the cache is faster than stock on your machine, you will not see the maximum gain on the first run but on all the later ones, once the file exists
  and the fork has measured both placements. Do the first runs with a short, simple prompt and let the answer run to its natural length, then judge the speed after them.  
- **It tells you what it does.** `llama-server` logs, and `llama-cli -lv 3` prints after each reply, whether the cache is on,
  the hit rate, the tuned settings, threads and batch sizes.  

## MTP speculative decoding

Qwen3.8-Flash-Next MTP from PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
([@danielhanchen](https://github.com/danielhanchen)), GLM-5.3-Flash MTP from PR
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos), both in this build. Load a model's MTP draft head
with `-md`:

```sh
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS.gguf \
    -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -md GLM-5.3-Flash-MTP-Q4_K.gguf --spec-type draft-mtp --spec-draft-n-max 3
```

- MTP heads: Qwen3.8-Flash-Next from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  (`MTP/`, the `shared` files reuse the main model's embeddings); GLM-5.3-Flash from
  [neuralll/GLM-5.3-Flash-MTP-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF)
  (4.3 GiB, works with any `glm5-next` GLM-5.3-Flash GGUF). We made it: no GLM MTP GGUF existed, so
  `tools/bench/glm_splice_mtp.py` pulls just the MTP tensors out of unsloth's UD-Q4_K_XL GGUF with HTTP range requests
  (~4.3 GiB instead of the whole model) and writes them as a draft file; see [`tools/bench/`](tools/bench/) to rebuild it.  
- GLM MTP: 89% of drafts accepted in our test, output identical to plain decoding. For a model far bigger than VRAM the draft
  is not loaded unless you pass `--spec-draft-n-max`.  

## Your settings win

Anything you pass is used as given and is never auto-tuned:

| you pass | effect |
|---|---|
| `-t N`, `-tb N` | fixed thread counts |
| `--moe-expert-cache N` | cache slots per layer; `0` turns the cache off, `-1` sizes it from free VRAM |
| `--moe KEY=VAL,...` | `slots`, `pf-slots`, `up-max`, `recent`, `pred-top`, `train-every`, or any tuning knob (`swap-lead`, `upload-wait`, `wait-swaps`, `ev-cld`, `upload-share`, ...), for example `--moe upload-wait=3,swap-lead=0` |
| `--load-mode pin\|mmap` | pinned weights (the default for every tool when part of the model stays in host RAM and fits in available RAM, faster prompts; `-lm mmap` opts out) or mmap |
| `LLAMA_MOE_AUTO_MODE=stock\|cache` | force the placement; `LLAMA_MOE_STATE=0` ignores and never writes the state file |

## Turn it off

| you pass | effect |
|---|---|
| `--fork off` | the whole fork off: no expert cache, nothing tuned or measured, no auto-pinned weights, no thread tuner: plain stock behaviour (`LLAMA_ARG_FORK=off`) |
| `-at off` | only the self-tuning off (`--autotune off`, same as `--moe autotune=0`): the cache keeps working with fixed defaults, placement uses a static rule, nothing is measured or saved |

Environment forms: `LLAMA_AUTOTUNE=0`, `LLAMA_ARG_AUTOTUNE=off`, `LLAMA_ARG_FORK=off`.  

## Options

All fork options (flags, every `--moe` setting and tuning knob with defaults and examples, experimental ones marked, environment  
variables, the state file, multi-GPU and MTP use) are in [docs/fork-knobs.md](docs/fork-knobs.md). The key ones:

| option | what it does |
|---|---|
| `--moe slots=N` | expert slots per layer in VRAM (`-1` from free VRAM, `0` no cache) |
| `--moe swap-lead=N` | uses an expert needs over the one it evicts before it is swapped in; `0` swaps eagerly, higher swaps less |
| `--moe upload-wait=3` | uploads wait while the CPU reads experts, so both do not fight for RAM bandwidth (`0` off) |
| `--moe upload-now=0` | stop uploading this token's missed experts on the fly |
| `--moe upload-share=F` | share of the token time uploads may take (tuned live, default 0.25) |
| `--moe pred-top=M,train-every=N` | prefetch the experts the next layers will probably need (top M); the predictor learns every N tokens |
| `--moe recent=N` | tokens of recent use the cache scores by (default 64) |
| `-at off` | no self-tuning, fixed defaults |
| `-lm pin\|mmap` | pinned or memory-mapped weights |
| `LLAMA_MOE_STATE=0` | ignore and never write the state file |

Experimental knobs (`ev-cld`, `pin-hot`, `idle-up`, `l3-pf`, ...) are off by default, never beat the default in our tests and may be removed;  
they are listed in the document, not here.

## Resident pinned models (Optional)

Everything that stays in host RAM is pinned now, because pinned weights are far faster than mmap: GLM-5.3-Flash 3.0-bit (109 GiB) on machine A  
decodes at 12 to 15 tokens/s pinned and at 2 to 2.5 tokens/s when the load falls back to mmap, and a 12k-token prompt goes from 127 to 285 tokens/s once the state is learned.  
That is why auto-pin is the default, and it needs nothing from you. **Add `-lm mmap` to any command to fall back to mmap should you prefer to trade start speed for performance.**  

The optional part is a pool of 1 GiB huge pages that keeps the pinned models resident in RAM between runs: the first load fills a cache file in the pool,  
every later load maps that file instead of reading the model again, which more than halves the load time (measured on GLM-5.3-Flash 3.0-bit, 109 GiB: 43 s instead of 88 to 95 s).  
It is worth the hassle below only if you reload models of 100 GB and up often, which is mostly the command line (every `llama-cli` run is a new process that loads the model again);  
a server loads once and keeps running, and on a smaller model waiting 10 s instead of 5 s does not matter, so neither needs the pool.  

```
sudo ./pool.sh 100g                       # reserve a 100 GiB pool (any model under 100 GB) and mount the weights cache at /mnt/huge1g
GGML_CUDA_HUGEFS=/mnt/huge1g llama-cli -m model.gguf ...
sudo ./pool.sh unmount                    # give the memory back
```

- `sudo ./pool.sh model.gguf` reserves what that model needs; `sudo ./pool.sh mount` reserves what the machine can spare (RAM minus a margin of the larger of 24 GiB and 20%).  
- The first load of a model fills a cache file in the pool; later loads map it. Several models share the pool, the least recently used one goes when it is full.  
- Without `GGML_CUDA_HUGEFS`, or without a pool, nothing changes.  
- **Needs Linux 4.11 or newer, built with `CONFIG_CONTIG_ALLOC`, on a CPU with 1 GiB pages (`pdpe1gb`)**: reserving 1 GiB pages at run time needs both. That is what makes the pool optional and cheap: the RAM is reserved only while you want it (`mount`), and `unmount` gives it back, whereas a boot-time reservation (`hugepages=N` on the kernel command line) holds it for good and starves every other program. Check with `grep CONFIG_CONTIG_ALLOC /boot/config-$(uname -r)` (it must say `=y`) and `ls /sys/kernel/mm/hugepages/hugepages-1048576kB` (it must exist). Tested on Linux 7.1. Do not boot with `hugetlb_cma=`: pages inside a CMA area cannot be pinned for the GPU. Linux only.  

The disadvantages:  
- **The first start is slow.** The first load of a model reads it from disk, pins it and fills the pool's cache file; only the later loads are fast.  
- **It needs sudo and a reboot.** 1 GiB pages can only be reserved from memory that is not fragmented yet. Reboot, then run `sudo ./pool.sh 100g` before anything else uses RAM, to get the biggest pool; after days of uptime it is no longer possible to reserve a big one.  
- **The pool is RAM nobody else can use** while it is mounted (a 100 GiB pool leaves about 25 GiB for everything else).  

## Good to know

- **RAM is the limit.** Decode speed is bound by how fast the CPU reads the experts that are not in VRAM. More or faster RAM,
  more VRAM or a faster GPU link all raise it.  
- Output can differ slightly from stock at temperature 0: a cached expert runs on the GPU, a missed one on the CPU.  
- Prompt processing of models bigger than RAM (mmap) is 20-30% below stock: the experts stream over PCIe.  
- A second GPU on a slow slot helps less; prompt processing goes to the fastest link.  

## Credits and contact

The expert cache builds on [@csantiago78](https://github.com/csantiago78)'s llama.cpp PR
[#27861](https://github.com/ggml-org/llama.cpp/pull/27861); GLM-5.3-Flash support is upstream
([#27773](https://github.com/ggml-org/llama.cpp/pull/27773)).  

**About the author of this fork**: I'm actively looking for an AI engineering/research
role and open to relocating out of Eastern Europe. If this work is useful to you or
your team, reach out: [linkedin.com/in/neuralll](https://www.linkedin.com/in/neuralll/)

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.  

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.  

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
