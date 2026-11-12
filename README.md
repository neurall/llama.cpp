# llama.cpp MoE fork: large mixture-of-experts models on consumer GPUs

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) runs mixture-of-experts (MoE) models that do not fit in GPU memory, and models larger than system RAM, at many times the speed of upstream llama.cpp on the same hardware. On two RTX 3090 cards with dual-channel RAM: prompt processing up to **23x** faster, generation up to **2.5x** faster (MiMo-V2.6-Flash, GLM-5.3-Flash, Qwen3.8-Next, DeepSeek-V4.1-Flash); on an 8-channel workstation, where upstream itself runs twice as fast, MiMo's 12k prompt still runs **12.6x** faster.

It is a drop-in replacement for `llama-server` and `llama-cli`. The context, the micro-batch, the expert cache and the GPU split are all sized automatically per request, so no `-c`, `-ub`, `-ngl` or `--moe` flags are needed:

```sh
llama-server -m MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf
```

**Download:** the [latest release](https://github.com/neurall/llama.cpp/releases/latest) has one package per OS with CPU, CUDA, Vulkan and HIP backends (Linux `tar.gz`, Windows `zip`). NVIDIA cards also need the `cudart-` archive if CUDA 13 is not installed.

## Results

Each row: the official upstream binary (stock, its own defaults, `-np 1`) against this fork, same machine, same request, same day. **12k/1k** = 12.4k-13.8k-token code-review prompt, 1024 tokens out; **3k/1k** = about 3k-token source file plus an edit instruction, answer up to 1024 tokens; **short/1k** = one-line prompt, 1024 tokens out (too short to time the prompt). Decode = generated tokens/s, prompt = prompt tokens/s. Every answer is stored in [fork/outputs](fork/outputs) (`<id>.txt` ours, `<id>s.txt` stock).

### Current machine (from 2026-11-12)

Lenovo P620: Threadripper PRO 3975WX (32 cores), 256 GB DDR4-3200 in 8 channels, 2x RTX 3090 on PCIe 4.0 x16, power-limited to 280 W each. Against the previous machine this has 4x the memory bandwidth and the full PCIe link on the second card.

| model | test | stock decode (t/s) | ours decode (t/s) | stock prompt (t/s) | ours prompt (t/s) | decode gain | prompt gain | stock build | ours build | answers (ours, stock) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash 3-bit | 12k/1k | 30.01 | **36.65** | 115.6 | **1458.0** | 1.22x | **12.6x** | b12317 | b15065 | [26477](fork/outputs/26477.txt), [26475s](fork/outputs/26475s.txt) |
| MiMo-V2.6-Flash 3-bit | 3k/1k | - | 37.96 | - | 660.7 | - | - | - | b15065 | [26479](fork/outputs/26479.txt) |
| MiMo-V2.6-Flash 3-bit | short/1k | - | 41.19 | - | - | - | - | - | b15065 | [26481](fork/outputs/26481.txt) |

Measured 2026-11-12, two runs per cell, the faster one shown; `-` = not measured on this machine yet (the other models follow). Stock gained the most from this machine: it computes the experts on the CPU, so the 8-channel memory made it 2.1x faster in decode and 2.3x in prompt processing than on the previous machine (MiMo 12k/1k: 13.99 to 30.01 t/s, 49.4 to 115.6 prompt t/s); ours gained 1.38x and 1.42x (26.6 to 36.65 t/s, 1025 to 1458 prompt t/s).

### Previous machine (until 2026-11-11)

Ryzen 7 3700X (8 cores), 128 GB DDR4-3200 in 2 channels, 2x RTX 3090 on PCIe x16 + x4, power-limited to 280 W (x16 card) and 220 W (x4 card).

<img width="1591" height="1090" alt="Best gainers on the previous machine" src="https://github.com/user-attachments/assets/7968c9de-766b-4d23-9fa2-2d0673d19c16" />

Upper number stock, lower ours; ppl = perplexity (see [Quality](#quality)). Stock is the official b11829 (rows of builds b13838 and b13842) or b11742 (older rows); each cell shows the build with the best run. Rows where neither gain reaches +10% are left out: models that fit in VRAM, and dense models, run at stock speed.

| model | test | t/s | pp | gain t | gain pp | ppl | build | output |
|---|---|---|---|---|---|---|---|---|
| MiMo 2.6F 3-bit | 16k/64† | 13.73<br>26.67 | 48.6<br>1123.9 | +94% 1.94x† | +2212% 23.12x | -<br>- | b13954 | [19204](fork/outputs/19204.txt)<br>[19571s](fork/outputs/19571s.txt) |
| MiMo 2.6F 3-bit | 12k/1k | 14.0<br>26.6 | 49<br>1025 | +90% | +1975% 20.8x | 1.8758<br>1.8758 | b13842 | [18373](fork/outputs/18373.txt)<br>[18221s](fork/outputs/18221s.txt) |
| GLM 5.3 3-bit | 16k/64† | 12.53<br>18.58 | 41.7<br>821.2 | +48% 1.48x† | +1869% 19.69x | -<br>- | b13959 | [19212](fork/outputs/19212.txt)<br>[19572s](fork/outputs/19572s.txt) |
| GLM 5.3 3-bit | 12k/1k | 12.8<br>19.8 | 43<br>689 | +55% | +1518% 16.2x | 1.7626<br>1.7578 | b13838 | [18209](fork/outputs/18209.txt)<br>[18232s](fork/outputs/18232s.txt) |
| MiMo 2.6F 3-bit | 3k/1k | 14.1<br>31.6 | 48<br>754 | +124% 2.2x | +1460% 15.6x | 1.8758<br>1.8758 | b13838 | [18199](fork/outputs/18199.txt)<br>[18223s](fork/outputs/18223s.txt) |
| GLM 5.3 3-bit | 3k/1k | 12.7<br>20.2 | 40<br>617 | +59% | +1443% 15.4x | 1.7626<br>1.7578 | b13838 | [18211](fork/outputs/18211.txt)<br>[18235s](fork/outputs/18235s.txt) |
| Qwen 3.8 Next IQ4_XS | 4k/64† | 30.22<br>53.92 | 209.5<br>1705.0 | +78% 1.78x† | +714% 8.14x | -<br>- | b13964 | [19270](fork/outputs/19270.txt)<br>[19567s](fork/outputs/19567s.txt) |
| Qwen 3.8 Next IQ4_XS | 16k/64† | 31.98<br>50.24 | 207.9<br>1659.8 | +57% 1.57x† | +698% 7.98x | -<br>- | b13959 | [19208](fork/outputs/19208.txt)<br>[19568s](fork/outputs/19568s.txt) |
| GLM 5.3 3.5-bit* | 3k/1k | 10.7<br>16.1 | 33<br>241 | +50% | +640% 7.4x | 1.6326<br>1.6265 | b13842 | [18429](fork/outputs/18429.txt)<br>[18277s](fork/outputs/18277s.txt) |
| Qwen 3.8 Next IQ4_XS | 12k/1k | 32.6<br>52.7 | 213<br>1902 | +62% | +793% 8.9x | 1.4101<br>1.4101 | b13838, b14020 | [18203](fork/outputs/18203.txt), [19588](fork/outputs/19588.txt)<br>[18227s](fork/outputs/18227s.txt) |
| GLM 5.3 3.5-bit* | 12k/1k | 10.8<br>11.0 | 34<br>230 | +2% | +572% 6.7x | 1.6326<br>1.6265 | b13563 | [16759](fork/outputs/16759.txt)<br>[17143s](fork/outputs/17143s.txt) |
| Qwen 3.8 Next IQ4_XS | 3k/1k | 32.1<br>54.9 | 209<br>1339 | +71% | +542% 6.4x | 1.4101<br>1.4101 | b13842 | [18381](fork/outputs/18381.txt)<br>[18228s](fork/outputs/18228s.txt) |
| Qwen 3.8 Next IQ3_S | 12k/1k | 43.8<br>53.6 | 372<br>1477 | +22% | +297% 4.0x | 1.3927<br>1.3927 | b13586 | [16719](fork/outputs/16719.txt)<br>[17094s](fork/outputs/17094s.txt) |
| Qwen 3.8 Next IQ3_S | 3k/1k | 44.6<br>57.3 | 387<br>1339 | +28% | +246% 3.5x | 1.3927<br>1.3927 | b13842 | [18423](fork/outputs/18423.txt)<br>[18270s](fork/outputs/18270s.txt) |
| MiMo 2.6F 3-bit | short/1k | 14.1<br>34.8 | -<br>- | +147% 2.5x | - | 1.8758<br>1.8758 | b13838 | [18201](fork/outputs/18201.txt)<br>[18225s](fork/outputs/18225s.txt) |
| Qwen 3.8 Next IQ4_XS | short/1k | 30.9<br>55.1 | -<br>- | +78% | - | 1.4101<br>1.4101 | b13586 | [16788](fork/outputs/16788.txt)<br>[17114s](fork/outputs/17114s.txt) |
| GLM 5.3 3-bit | short/1k | 12.7<br>21.6 | -<br>- | +69% | - | 1.7626<br>1.7578 | b13838 | [18213](fork/outputs/18213.txt)<br>[18236s](fork/outputs/18236s.txt) |
| GLM 5.3 3.5-bit* | short/1k | 10.7<br>17.1 | -<br>- | +59% | - | 1.6326<br>1.6265 | b13842 | [18431](fork/outputs/18431.txt)<br>[18279s](fork/outputs/18279s.txt) |
| Qwen 3.8 Next IQ3_S | short/1k | 43.5<br>56.9 | -<br>- | +31% | - | 1.3927<br>1.3927 | b13563 | [16766](fork/outputs/16766.txt)<br>[17102s](fork/outputs/17102s.txt) |

† Rows with a 64-token answer (/64): stock was asked for 64 tokens, below the 256 tokens a decode speed is taken from, so the decode gain of these rows is for information; the prompt gain stands. The new rows show the best decode speed and the best prompt speed among the runs of the cell (what the run list prints, two decimals); the linked answer is one of those runs, so its header can show a slower speed than the table.  
\* GLM 3.5-bit (127.7 GiB) is larger than this machine's 128 GB RAM, so part of it runs from the SSD and the system swaps (up to about 4 GiB of swap in use during runs). Ours is faster than stock despite that.  

## Where the speed comes from

Stock llama.cpp keeps whole layers on the GPU and computes every expert that does not fit on the CPU, for prompt processing as well as generation. This fork treats VRAM, RAM and the SSD as one memory hierarchy for the experts, and keeps the GPUs computing. Each change below was measured as a paired A/B on the same machine when it was added; the effects do not add up linearly.

| change | what it does | measured effect |
|---|---|---|
| **Prompt passes on the GPUs** | Experts the GPU does not hold are streamed to it during the prompt, in passes of up to 16k tokens, so every upload serves many tokens; stock computes them on the CPU | 16x-23x prompt speed on GLM and MiMo on the previous machine (stock 42-49 prompt t/s, ours 700-1150), 12.6x on the current one (MiMo 12k: 115.6 to 1458 prompt t/s); micro-batch 1024 to 4096 alone: 2.1x (GLM 12k: 176.6 to 370.1 prompt t/s) |
| **Expert cache in free VRAM (VRAM hot map)** | All free VRAM becomes one pool of expert blocks shared by all layers; the experts the tokens use most stay there, misses run on the CPU in parallel | decode +20% to +150% against stock on models that do not fit (table above); global pool against fixed slots per layer +2-4% |
| **Learned hot map kept between runs** | Each expert's lifetime use is saved per model (`moe-state.ini`) and preloaded at the next start; after a prompt, the blocks it borrowed are refilled with the hottest experts | first request after a restart +4-6% decode; decode after a 12k prompt +12.7% (cache hits 43% to 55%) |
| **RAM pool with direct SSD reads (RAM hot map)** | Experts live in a pinned RAM pool sized from free memory; experts outside it are read from the SSD with direct I/O, never through the page cache or a file mapping | DeepSeek generation +29-31%, first short prompt after a start 3-5x, MiMo edit prompt +28% (b14965 against b14929); models larger than RAM run at all |
| **Dynamic context and KV in the same pool** | No context is reserved: each request claims the KV it needs from the same VRAM as the experts and gives it back; if it does not fit, the KV converts in place to q8_0, then q4_0, then moves layers to RAM | no `-c` to guess; prompts that stock refuses at its automatic context (30k, 60k) run at full speed; MiMo 200k-token prompt 794 prompt t/s, 400k 363 prompt t/s; Qwen3.6-35B 200k 3259 prompt t/s; q8_0 KV perplexity within noise of f16 |
| **Compute buffer lent per pass** | The large prompt micro-batch borrows expert-cache blocks only for the pass and returns them for generation | +6.6% decode after a long prompt at the same prompt speed |
| **Dense split over several GPUs** | Tensor-parallel split of the dense (attention, shared) layers over the cards for long prompts, experts streamed to each card, chosen per request by a planner from measured link and compute speed | prompt speed at 60k tokens: Qwen3.6-35B +44% (2890 to 4169 prompt t/s), Qwen3.8 IQ3_S +25%, GLM-5.3 3.0-bit +23-25%; GLM 12k +35-70% (b14965); MiMo 200k +39% |
| **Expert split over several GPUs** | Each card computes its share of every layer's experts (MoE tensor parallelism), for models that do not fit the cards' memory | Qwen3.8 IQ4_XS 12k on 8 GiB cards: +19-23% prompt speed over one card (development build, next release) |
| **Next-layer prediction** | The next layer's router runs early on the current layer's input, so the upload of its experts starts while the current layer computes | 79% of the next layer's experts found in its predicted top 8, 94% in its top 16 |
| **Faster start** | Weights read with direct I/O at the drive's measured queue depth while earlier buffers are pinned | GLM-5.3 3.0-bit 69 s to 37 s until the server answers, DeepSeek-V4.1 71 s to 49.5 s (b15065) |

To our knowledge no other engine splits the dense layers and the experts of a model over several GPUs when the model does not fit in their memory: upstream llama.cpp splits only the layers that fit, and vLLM and SGLang expect the model in VRAM.

<img src="fork/docs/img/vram-layout.svg" alt="One VRAM buffer per GPU: compute graph, KV and expert cache share it" width="100%">

Older measurements and the per-feature A/B details: [fork/docs/features.md](fork/docs/features.md), [fork/docs/perf-history.md](fork/docs/perf-history.md).

## What gain to expect

The gain depends on how much of the model fits in VRAM. Measured on the previous machine (2x RTX 3090, 48 GiB of VRAM, 128 GB RAM); *VRAM share* is 48 GiB divided by the model file. Prompt speed from the 12k prompt, decode speed from the short prompt, against stock b11742.

| model | file | VRAM share | prompt speed | decode speed |
|---|---|---|---|---|
| OLMoE-1B-7B Q4_K_M | 3.9 GiB | fits | - | +1% |
| Qwen3.6-35B-A3B | 11.4 GiB | fits | 1.0x | +1% |
| Qwen3.8 27B IQ4_NL (dense) | 15.2 GiB | fits | 1.0x | +0.3% |
| Qwen3.8 Next IQ1_M | 54.4 GiB | 88% | 1.04x | -0.5% |
| Qwen3.8 Next IQ3_S | 77.8 GiB | 62% | 3.9x | +31% |
| Qwen3.8 Next IQ4_XS | 87.2 GiB | 55% | 7.1x | +78% |
| MiMo-V2.6-Flash 3-bit | 107.7 GiB | 45% | 19.4x | +131% (2.3x) |
| GLM-5.3-Flash 3.0-bit | 109.4 GiB | 44% | 14.8x | +62% |
| GLM-5.3-Flash 3.5-bit* | 127.7 GiB | 38% | 6.7x | +37% |

- **Models that fit in VRAM, and dense models, run at stock speed.** The gains come from caching and streaming experts.
- **The less of the model fits, the more there is to gain**, down to models larger than RAM, which run partly from the SSD: GLM-5.3 3.5-bit (127.7 GiB) on a 128 GB machine still ran 7x faster prompts and +21% decode on the 3k prompt.
- **On your hardware:** divide total VRAM by the model file for the share. Reports from other machines are welcome (issues on this repository, with the startup log of `-lv 5`).

## Usage

```sh
llama-server -m MODEL.gguf                 # OpenAI-compatible server, everything automatic
llama-cli    -m MODEL.gguf -p "hello"
```

The fork's defaults apply only when a MoE model is larger than the free VRAM of all GPUs; other models, and every setting you pass yourself, behave as in upstream. The chosen values are printed at startup (`-lv 5` prints all of them).

| setting | upstream default | fork default |
|---|---|---|
| expert placement | whole layers on the GPU, the rest on the CPU | all experts in RAM, free VRAM as the expert cache |
| `-ub` (tokens per prompt pass) | 512 | sized per request and card, up to 16384 |
| `-c` (context) | model maximum, fitted to VRAM once | claimed per request; KV falls back f16, q8_0, q4_0, RAM |
| `-t` (threads) | physical cores | tuned on measured decode time; prompt threads tuned separately |
| GPU split | layers by memory | per request: one card, dense split or expert split, from measured speeds |

Useful switches:

| switch | effect |
|---|---|
| `--fork off` | upstream behaviour for everything |
| `-at off` | no automatic tuning (the fork's other defaults stay) |
| `--moe fixed=1` | repeatable output for benchmarks (a cached expert runs on the GPU, a missed one on the CPU, and they round slightly differently) |
| `--moe dense-split=0` / `=1` | force one card / the dense split |
| `--moe kv-fallback=k8v8,ram` | skip the q4_0 KV step (`none` turns the fallback off) |
| `--moe-expert-cache 0` | expert cache off |
| `GET /status` | server phase (loading, prefill, generation) with live ms per token |

Every `--moe` key and its default: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md).

## Quality

Perplexity of the same text, same chunks, official stock `llama-perplexity` against ours; in brackets the per-chunk difference to stock (mean ± standard error). Differences under 0.2% are inside the noise; where the prompt is cut into passes of the same size, ours matches stock to four decimals. Build b13600 against stock b11742 (DeepSeek: b11887).

| model | stock ppl | ours default | ours, one card | ours, dense split only | ours, two cards (f32 hand-offs) | chunks |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash-IQ2_XXS | 1.5665 | 1.5662 (-0.02 ± 0.13%) | - | - | - | 10 |
| GLM-5.3-Flash-GSQ-RCO-3.0bit | 1.7626 | 1.7578 (-0.27 ± 0.18%) | 1.7599 (-0.15 ± 0.27%) | 1.7584 (-0.24 ± 0.53%) | 1.7631 (+0.03 ± 0.39%) | 9 |
| GLM-5.3-Flash-GSQ-RCO-3.5bit | 1.6326 | 1.6265 (-0.37 ± 0.20%) | 1.6260 (-0.41 ± 0.19%) | 1.6349 (+0.14 ± 0.51%) | 1.6288 (-0.23 ± 0.28%) | 9 |
| MiMo-V2.6-Flash-RL-GSQ-RCO-3bit | 1.8758 | 1.8758 (+0.00) | 1.8758 (+0.00) | 1.8736 (-0.12 ± 0.11%) | 1.8725 (-0.18 ± 0.25%) | 9 |
| OLMoE-1B-7B-0924-Q4_K_M | 2.0696 | 2.0696 (+0.00) | 2.0696 (+0.00) | 2.0688 (-0.04 ± 0.07%) | 2.0694 (-0.01 ± 0.08%) | 12 |
| Qwen3.6-35B-A3B-GSQ-hybrid | 1.7843 | 1.7843 (+0.00) | 1.7843 (+0.00) | 1.7842 (-0.01 ± 0.08%) | 1.7851 (+0.04 ± 0.07%) | 10 |
| Qwen3.8-27B-IQ4_NL | 1.6360 | 1.6360 (+0.00) | 1.6360 (+0.00) | - | - | 10 |
| Qwen3.8-27B-MTP-Q5_K_M | 1.6382 | 1.6382 (+0.00) | 1.6382 (+0.00) | - | - | 10 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002 | 1.4756 | 1.4756 (+0.00) | 1.4756 (+0.00) | 1.4758 (+0.01 ± 0.17%) | 1.4745 (-0.07 ± 0.13%) | 10 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002 | 1.3927 | 1.3927 (+0.00) | 1.3927 (+0.00) | 1.3969 (+0.30 ± 0.14%) | 1.3975 (+0.34 ± 0.16%) | 10 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003 | 1.4101 | 1.4101 (+0.00) | 1.4101 (+0.00) | 1.4114 (+0.09 ± 0.18%) | 1.4111 (+0.07 ± 0.14%) | 10 |

The two-card paths cut a prompt into token ranges per card, which moves perplexity like a smaller `-ub` does on stock itself (Qwen3.8 IQ3_S: stock 1.3927 at `-ub 1024`, 1.3946 at 512, 1.3996 at 256). Method, decode-path numbers and how to check it on any model: [fork/docs/perplexity.md](fork/docs/perplexity.md).

## Reproduce

Ours from the [releases page](https://github.com/neurall/llama.cpp/releases), stock from [ggml-org releases](https://github.com/ggml-org/llama.cpp/releases) (the build named in the table). The exact requests are in [fork/bench](fork/bench) (`req-12k-1k.json`, `req-3k-1k-a.json` / `-b`, `req-short-1k.json`):

```sh
llama-server -m MODEL -np 1 --port 9931
curl -s localhost:9931/v1/chat/completions -H 'Content-Type: application/json' -d @fork/bench/req-12k-1k.json | grep -o '"timings":{[^}]*}'
```

`prompt_per_second` is the prompt speed, `predicted_per_second` the decode speed. The whole table with the harness (stock first, then ours, two runs per cell, results into `fork/runs.db` and `fork/outputs/`): set the paths in [fork/local.toml.example](fork/local.toml.example) and the models in [fork/models.ini](fork/models.ini), then `python3 fork/run.py run gainers -sc -r 2`. Perplexity: `python3 fork/run.py ppl -b BUILD`.

## Known limits

- **Measured on two machines with 2x RTX 3090**, plus a Radeon 780M laptop for HIP and Vulkan. Windows builds are smoke-tested only. 3+ GPU setups are untested.
- **The first runs of a new model are slower:** the hot maps are learned from use. A pre-warmed state for the models above is in [fork/moe-state.ini](fork/moe-state.ini) (copy to `~/.cache/llama.cpp/`, Windows `%LOCALAPPDATA%\llama.cpp\`).
- **Start-up takes longer on big models** while the RAM pool fills (GLM-5.3 3.0-bit: 37 s until the server answers).
- **Output can differ between runs at temperature 0** (GPU and CPU round differently); `--moe fixed=1` makes it repeat.

## Roadmap

1. **Prompt passes without staging copies:** uploads land directly in expert-cache blocks, so the VRAM that staging held goes to larger passes (in progress; expert split +9-16% over staged on the development build).
2. **Predicted expert fetch for very large models** (1 TB class, e.g. Kimi K3.1): the predictor's list is read from the SSD into RAM and VRAM ahead of the layer that needs it.
3. **More hardware:** results from other GPUs, 3+ cards and AMD/Intel systems are welcome.

## Credits and thanks

- **Georgi Gerganov ([ggerganov](https://github.com/ggerganov)) and every contributor to [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml**: this fork is a small layer on a fantastic project that has given so much to so many people for so many years. Thank you.
- **[@csantiago78](https://github.com/csantiago78)** and their pull request [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) ("GPU-resident LRU cache for host-offloaded MoE expert weights"), the starting point of this fork's expert cache. It has since diverged (window and admission instead of LRU, one global pool instead of fixed slots per layer), but the idea and the first working version are theirs.
- **People who tried this fork and reported back**: [@MichaelDietzel](https://github.com/MichaelDietzel) (tested on a 2080 Ti and asked for documentation of the `--moe` keys and of multi-GPU use), [@Volunteer-1](https://github.com/Volunteer-1) (Windows build report and the same documentation questions; every flag is now documented in [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md)), [@homeofe](https://github.com/homeofe) (the small-VRAM data point where the cache was slower than stock, and the AVX2 dot products for IQ3_S and IQ3_XXS in `ggml-cpu`, merged into this fork, MIT license) and [@yuffx](https://github.com/yuffx) (the PCIe-bound explanation of that result).

---

*The original llama.cpp README follows below.*

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

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

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
| [CANN](docs/build.md#cann) | Huawei Ascend NPU |
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
