# README of b16046 (moved here 2026-11-26)

The fork part of README.md as published with release b16046, unchanged.

## Rocket LLama 


## up to 20x faster prefill and 2.4x decode on big models 

Fork of llama.cpp with multi gpu acceleration even for models bigger than total gpu mem. 
But even on single 5090 20x pp and 2x t gains 



In other words tha gains you see bellow are not from multigpu yet just from more efficient vram mmgmt
ie the same 20x speedup in mimo is happening on single 5090 too.

```sh
llama-server -m MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf
```

**Download:** the [latest release](https://github.com/neurall/llama.cpp/releases/latest) has one package per OS with CPU, CUDA, Vulkan and HIP backends (Linux `tar.gz`, Windows `zip`). NVIDIA cards also need the `cudart-` archive if CUDA 13 is not installed.

### Results

Each row: the official upstream binary (stock, its own defaults, `-np 1`) against this fork, same machine, same request, same day. **12k/1k** = 12.4k-13.8k-token code-review prompt, 1024 tokens out; **3k/1k** = about 3k-token source file plus an edit instruction, answer up to 1024 tokens; **short/1k** = one-line prompt, 1024 tokens out (too short to time the prompt). Decode = generated tokens/s, prompt = prompt tokens/s. Every answer is stored in [fork/outputs](fork/outputs) (`<id>.txt` ours, `<id>s.txt` stock).

One run of ours (the best valid run among our builds; the build column names it) against the official stock run of the same cell closest to stock's median (the stock column names its version; stock binaries are pulled, never built). Every number is from those two runs, whose answers are in the output column (ours on top, stock below, so you can open both). Sorted by the bigger gain; rows where neither gain reaches +10% are left out (models that fit in VRAM and the dense 27B models run at stock speed). Upper number stock, lower ours. **12k/1k** = 12.4k-13.8k-token code-review prompt, 1024 tokens out; **15k/1k** = the same review at 15.5k tokens; **16k/1k** = the same review at 16k tokens; **3k/1k** = ~3k-token source file plus an edit instruction, answer up to 1024 tokens; **short/1k** = one-line prompt, 1024 tokens out (no prompt speed: too short to time). t/s = decode, pp = prompt processing, ppl = perplexity on b13600 (details below; - = not measured for that build). Valid = same power limits as stock (x4 card 220 W, x16 card 280 W), answer not looped, decode only from answers of 256+ tokens. 2x RTX 3090 (x16 + x4), Ryzen 7 3700X, 128 GB DDR4-3200.

| model | test | t/s | pp | gain t | gain pp | ppl | build | stock | output |
|---|---|---|---|---|---|---|---|---|---|
| MiMo 2.6F 3-bit | 12k/1k | 13.99<br>25.71 | 49.4<br>1241.1 | +84% | +2412% 25.1x | -<br>- | **b14965** | b11829 | [24369](fork/outputs/24369.txt)<br>[18221s](fork/outputs/18221s.txt) |
| MiMo 2.6F 3-bit | 16k/64† | 13.73<br>26.67 | 48.6<br>1123.9 | +94% 1.94x† | +2212% 23.12x | -<br>- | b13954 | b11887 | [19204](fork/outputs/19204.txt)<br>[19571s](fork/outputs/19571s.txt) |
| DeepSeek 4.1 IQ2_XXS* | short/1k | 1.47<br>32.95 | -<br>- | +2141% 22.4x | - | -<br>- | b14939 | b11887 | [24094](fork/outputs/24094.txt)<br>[19818s](fork/outputs/19818s.txt) |
| MiMo 2.6F 3-bit, 1× RTX 5090 | 12k/1k | 6.83<br>36.13 | 141.3<br>3016.3 | +429% 5.3x | +2035% 21.3x | -<br>- | b13348 | b11799 | [14701](fork/outputs/14701.txt)<br>[14700s](fork/outputs/14700s.txt) |
| GLM 5.3 3-bit | 16k/1k† | 12.53<br>19.56 | 41.7<br>869.6 | +56%† | +1985% 20.9x | -<br>- | b14939 | b11887 | [24065](fork/outputs/24065.txt)<br>[19572s](fork/outputs/19572s.txt) |
| DeepSeek 4.1 IQ2_XXS* | 3k/1k | 1.46<br>26.25 | 39.1<br>336.4 | +1698% 18.0x | +760% 8.6x | -<br>- | b14939 | b11887 | [24092](fork/outputs/24092.txt)<br>[19814s](fork/outputs/19814s.txt) |
| GLM 5.3 3.5-bit* | 12k/1k | 10.72<br>15.18 | 34.3<br>614.9 | +42% | +1693% 17.9x | -<br>- | b14954 | b11829 | [24296](fork/outputs/24296.txt)<br>[18275s](fork/outputs/18275s.txt) |
| GLM 5.3 3-bit | 12k/1k | 12.8<br>19.8 | 43<br>689 | +55% | +1518% 16.2x | 1.7626<br>1.7578 | b13838 | b11829 | [18209](fork/outputs/18209.txt)<br>[18232s](fork/outputs/18232s.txt) |
| MiMo 2.6F 3-bit | 3k/1k | 14.1<br>31.6 | 48<br>754 | +124% 2.2x | +1460% 15.6x | 1.8758<br>1.8758 | b13838 | b11829 | [18199](fork/outputs/18199.txt)<br>[18223s](fork/outputs/18223s.txt) |
| GLM 5.3 3-bit | 3k/1k | 12.7<br>20.2 | 40<br>617 | +59% | +1443% 15.4x | 1.7626<br>1.7578 | b13838 | b11829 | [18211](fork/outputs/18211.txt)<br>[18235s](fork/outputs/18235s.txt) |
| GLM 5.3 3.5-bit* | 3k/1k | 10.70<br>17.05 | 32.5<br>465.7 | +59% | +1333% 14.3x | -<br>- | b14940 | b11829 | [24158](fork/outputs/24158.txt)<br>[18277s](fork/outputs/18277s.txt) |
| DeepSeek 4.1 IQ2_XXS* | 16k/1k | 1.46<br>18.60 | 42.5<br>347.9 | +1174% 12.7x | +719% 8.2x | -<br>- | b14563 | b11887 | [22095](fork/outputs/22095.txt)<br>[19809s](fork/outputs/19809s.txt) |
| Qwen 3.8 Next IQ4_XS | 15k/1k | 29.14<br>52.31 | 176.8<br>1934.1 | +80% | +994% 10.9x | -<br>- | b13845, b14705 | b12131 | [23477](fork/outputs/23477.txt), [23510](fork/outputs/23510.txt)<br>[23452s](fork/outputs/23452s.txt) |
| Qwen 3.8 Next IQ4_XS | 16k/1k† | 31.98<br>50.98 | 207.9<br>1907.5 | +59%† | +818% 9.2x | -<br>- | b14939 | b11887 | [24059](fork/outputs/24059.txt)<br>[19568s](fork/outputs/19568s.txt) |
| Qwen 3.8 Next IQ4_XS | 12k/1k | 32.6<br>52.7 | 213<br>1902 | +62% | +793% 8.9x | 1.4101<br>1.4101 | b13838, b14020 | b11829 | [18203](fork/outputs/18203.txt), [19588](fork/outputs/19588.txt)<br>[18227s](fork/outputs/18227s.txt) |
| Qwen 3.8 Next IQ4_XS | 4k/64† | 30.22<br>53.92 | 209.5<br>1705.0 | +78% 1.78x† | +714% 8.14x | -<br>- | b13964 | b11887 | [19270](fork/outputs/19270.txt)<br>[19567s](fork/outputs/19567s.txt) |
| Qwen 3.8 Next IQ4_XS | 3k/1k | 32.1<br>54.9 | 209<br>1339 | +71% | +542% 6.4x | 1.4101<br>1.4101 | b13842 | b11829 | [18381](fork/outputs/18381.txt)<br>[18228s](fork/outputs/18228s.txt) |
| Qwen 3.8 Next IQ3_S | 3k/1k | 44.59<br>56.18 | 387.2<br>1777.2 | +26% | +359% 4.6x | -<br>- | b14939 | b11829 | [24122](fork/outputs/24122.txt)<br>[18270s](fork/outputs/18270s.txt) |
| Qwen 3.8 Next IQ3_S | 12k/1k | 43.8<br>53.6 | 372<br>1477 | +22% | +297% 4.0x | 1.3927<br>1.3927 | b13586 | b11742 | [16719](fork/outputs/16719.txt)<br>[17094s](fork/outputs/17094s.txt) |
| MiMo 2.6F 3-bit | short/1k | 14.1<br>34.8 | -<br>- | +147% 2.5x | - | 1.8758<br>1.8758 | b13838 | b11829 | [18201](fork/outputs/18201.txt)<br>[18225s](fork/outputs/18225s.txt) |
| Qwen 3.8 Next IQ4_XS | short/1k | 30.9<br>55.1 | -<br>- | +78% | - | 1.4101<br>1.4101 | b13586 | b11742 | [16788](fork/outputs/16788.txt)<br>[17114s](fork/outputs/17114s.txt) |
| GLM 5.3 3-bit | short/1k | 12.7<br>21.6 | -<br>- | +69% | - | 1.7626<br>1.7578 | b13838 | b11829 | [18213](fork/outputs/18213.txt)<br>[18236s](fork/outputs/18236s.txt) |
| GLM 5.3 3.5-bit* | short/1k | 10.7<br>17.1 | -<br>- | +59% | - | 1.6326<br>1.6265 | b13842 | b11829 | [18431](fork/outputs/18431.txt)<br>[18279s](fork/outputs/18279s.txt) |
| Qwen 3.8 Next IQ3_S | short/1k | 43.5<br>56.9 | -<br>- | +31% | - | 1.3927<br>1.3927 | b13563 | b11742 | [16766](fork/outputs/16766.txt)<br>[17102s](fork/outputs/17102s.txt) |

† Rows with a 64-token answer (/64): stock was asked for 64 tokens, below the 256 tokens a decode speed is taken from, so the decode gain of these rows is for information; the prompt gain stands. The new rows show the best decode speed and the best prompt speed among the runs of the cell (what the run list prints, two decimals); the linked answer is one of those runs, so its header can show a slower speed than the table.  
16k/1k†: ours answered 1024 tokens, stock was asked for 64 there, so the decode gain of those rows is for information too.  
Rows marked **1× RTX 5090**: a rented cloud server (2× EPYC 9J14, 760 GB RAM, one RTX 5090 at PCIe 5 x16), 2026-10-21, stock and ours minutes apart. All other rows: Ryzen 7 3700X, 128 GB DDR4, 2× RTX 3090 (the second at x4).  
\* GLM 3.5-bit (127.7 GiB) is larger than this machine's 128 GB RAM, so part of it runs from the SSD and the system swaps (up to about 4 GiB of swap in use during runs). Ours is faster than stock despite that.  
\* DeepSeek 4.1 IQ2_XXS (232 GiB) is far larger than the 128 GB RAM: stock reads it through the page cache at about 1.5 t/s; ours keeps the hot experts in VRAM and a pinned RAM pool and reads the rest from the SSD (O_DIRECT).  

### Where the speed comes from

Our builds in the table: b14965 (this release, no-mmap), b14939-b14954 (its candidate builds of 2026-11-08, same engine line), b14563 (2026-11-02), and older rows from b13563-b14020 that newer builds did not beat by 5% or more. b13348 (2026-10-21) is the 1× RTX 5090 row, against stock b11799 of the same day; b13845 and b14705 (2026-11-05/06) are the 15k/1k row, against stock b12131 (2026-11-05). Stock versions differ per row (stock column): b11742 (2026-10-20), b11829 (2026-10-23), b11866 and b11887 (2026-10-24/26), each the newest official stock on the day of its row. Older tables (earlier releases, laptop and CPU rows): [fork/docs/perf-history.md](fork/docs/perf-history.md).

#### Reproduce a row

1. Binaries: ours (Windows zip on the add the `cudart-` archive if CUDA 13 is not installed); stock: `llama-b11829-bin-ubuntu-cuda-13.4-x64.tar.gz` from [ggml-org b11829](https://github.com/ggml-org/llama.cpp/releases/tag/b11829) for the b13838/b13842 rows, `llama-b11742-bin-ubuntu-cuda-13.4-x64.tar.gz` from [ggml-org b11742](https://github.com/ggml-org/llama.cpp/releases/tag/b11742) for the older rows.
2. Start the server (port 9931). The `-c` values are the ones the table's runs used:

| test | ours | stock |
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

### What gain to expect

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

### Usage

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

### Quality

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

### Reproduce

Ours from the [releases page](https://github.com/neurall/llama.cpp/releases), stock from [ggml-org releases](https://github.com/ggml-org/llama.cpp/releases) (the build named in the table). The exact requests are in [fork/bench](fork/bench) (`req-12k-1k.json`, `req-3k-1k-a.json` / `-b`, `req-short-1k.json`):

```sh
llama-server -m MODEL -np 1 --port 9931
curl -s localhost:9931/v1/chat/completions -H 'Content-Type: application/json' -d @fork/bench/req-12k-1k.json | grep -o '"timings":{[^}]*}'
```

`prompt_per_second` is the prompt speed, `predicted_per_second` the decode speed. The whole table with the harness (stock first, then ours, two runs per cell, results into `fork/runs.db` and `fork/outputs/`): set the paths in [fork/local.toml.example](fork/local.toml.example) and the models in [fork/models.ini](fork/models.ini), then `python3 fork/run.py run gainers -sc -r 2`. Perplexity: `python3 fork/run.py ppl -b BUILD`.

### Known limits

- **One machine measured:** 2x RTX 3090 (x16 + x4 slots), Ryzen 7 3700X, 128 GB DDR4. Other hardware, AMD/Intel GPUs and 3+ GPUs are untested; the release archives carry CUDA and Vulkan, and the Windows archive also HIP (ROCm 6.4) for AMD cards. Vulkan is measured on an RTX 4060 and a Radeon 780M (Windows); on Linux it is only load-tested (no GPU run yet). Windows binaries are smoke-tested only.
- **The first runs of a model are slower.** The hot-expert state in `moe-state.ini` is learned from use (copy the pre-warmed [fork/moe-state.ini](fork/moe-state.ini) to start from it).
- **Models larger than free RAM load lazily.** The server answers while the expert weights are still being pinned in the background (about 80 s for 100 GiB), so the first big prompt after start can be several times slower than the next ones (GLM 3.5-bit, 3k prompt: 26-147 prompt t/s on the first request, 225-267 on the second request on the same server). The table shows the steady-state speed.
- **Perplexity:** equal to stock where the pass shape is the same; on the two-card paths Qwen 3.8 Next IQ3_S reads +0.3%, which is the size of stock's own change between `-ub 512` and `-ub 256` (see [Perplexity moves with the micro-batch size](#perplexity-moves-with-the-micro-batch-size-in-stock-too)).
- **Output can differ between runs** even at temperature 0 (a cached expert runs on the GPU, a missed one on the CPU, and they round slightly differently).
- **The gainers table takes each row from the best valid run among our builds** (build column), not all from this release: a row moves to a newer build only when that build's run beats it by 5% or more and has its answer in fork/outputs.

### Roadmap

1. **AMD tuning and Linux HIP.** Vulkan ships in the release archives and HIP in the Windows archive since b15021. HIP decode on the Radeon 780M still trails Vulkan (11.9 against 16.5 t/s); a kernel choice measured per card at start is next, then a Linux HIP build. Work is allocated by each card's compute speed and PCIe link, so cards of different vendors should be able to share a model; a mixed-vendor model split is untested.
2. **One pool for the expert cache and the staging buffers.** Merging the two is expected to improve VRAM efficiency further.
3. **Expert-cache lookup per topic and initialisation per request**, planned for later versions.
4. **Testing on more hardware.** Everything here was measured on one X570 board (x16 + x4 slots) with two RTX 3090s, plus a notebook with an RTX 4060. Multi-GPU setups with fast slots on every card are the most wanted: send your numbers and the startup log (`-lv 5`).

### Credits and thanks

- **Georgi Gerganov ([ggerganov](https://github.com/ggerganov)) and every contributor to [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml**: this fork is a small layer on a fantastic project that has given so much to so many people for so many years. Thank you.
- **[@csantiago78](https://github.com/csantiago78)** and their pull request [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) ("GPU-resident LRU cache for host-offloaded MoE expert weights"), the starting point of this fork's expert cache. It has since diverged (window and admission instead of LRU, one global pool instead of fixed slots per layer), but the idea and the first working version are theirs.
- **People who tried this fork and reported back**: [@MichaelDietzel](https://github.com/MichaelDietzel) (tested on a 2080 Ti and asked for documentation of the `--moe` keys and of multi-GPU use), [@Volunteer-1](https://github.com/Volunteer-1) (Windows build report and the same documentation questions; every flag is now documented in [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md)), [@homeofe](https://github.com/homeofe) (the small-VRAM data point where the cache was slower than stock, and the AVX2 dot products for IQ3_S and IQ3_XXS in `ggml-cpu`, merged into this fork, MIT license) and [@yuffx](https://github.com/yuffx) (the PCIe-bound explanation of that result).
- **[Strata](https://github.com/Niko1221/Strata) by [@Niko1221](https://github.com/Niko1221) and the Strata contributors**: ideas for a few of this fork's later decode tweaks, worth up to about a fifth more decode speed on some models, on top of the lead over stock shown in the table above: a dedicated per-core thread pool for the experts the CPU computes (the fork computed them through the CPU backend before), split kernels for the hyper-connection mix, and overlapping the parts of a decode token on the card. We reimplemented these ideas in our own code and carried some of them over to models Strata does not support yet, GLM 5.3 and MiMo 2.6, with Kimi K3.1 next; the gains there are still small. Strata's speeds are a benchmark target we measure against.

# Performance history

Tables moved out of the README when the newest run table replaced them (nothing deleted). Newest first as they were in the README; the README keeps only the latest runs.

### Best gainers so far (measured 2026-10-14; 2x RTX 3090 24 GB with one x16 and one x4 link, and an 8 GB RTX 4060 laptop where marked)

`pp` = prompt processing tokens/s, `t/s` = generated tokens/s. Perplexity: lower is better; `missing` is a run that is not finished or was never made. A blue (linked) gain is a fork result that beats stock, and in the `ppl` column the lower of the two numbers is linked; click it for the exact command that reproduces the row (`fork/bench/repro.py ID`, listed in [REPRO.md](fork/bench/REPRO.md)). In every `pp`, `t/s` and `ppl` cell the stock value is on top and the fork value below it, with the gain in brackets. Every measured run of our builds passes `--load-mode pin` (pinned weights) from now on. Sorted by the larger of the two gains. Same prompts as below, each build on its own defaults (only `-np 1`, `-c` for stock). "stock" is the newest upstream build that was measured on that cell (build in the last column); the fork rows are the best measured build of this release line (builds are given by build number, the commit is in the release notes and `fork/bench`; the published b12844 is being re-measured the same way, see the known-issue note).

<!--latest-->
## Latest measurements: build b13180 against latest upstream master (2026-10-15)

Each build on its own defaults (auto against auto), `-np 1` only, 3 runs per cell, no warm-up run. Stock over ours, gain in brackets. **i** = long prompt, prompt speed; **io** = 1k in and 1k out, prompt speed and token speed; **o** = short prompt, long answer, token speed. Older rows below are earlier measurements and stay as they were.  

| Model | i: prompt t/s | io: prompt t/s | io: tokens t/s | o: tokens t/s |
|---|---|---|---|---|
| MiMo-V2.6-Flash 3-bit | 54<br>**1074**&nbsp;<sub>(19.81x)</sub> | 48<br>**587**&nbsp;<sub>(12.29x)</sub> | 14.1<br>**33.7**&nbsp;<sub>(2.39x)</sub> | 14.1<br>**36.6**&nbsp;<sub>(2.60x)</sub> |
| GLM-5.3-Flash 3.0-bit | 43<br>**660**&nbsp;<sub>(15.43x)</sub> | 40<br>**393**&nbsp;<sub>(9.94x)</sub> | 12.5<br>**22.9**&nbsp;<sub>(1.83x)</sub> | 12.6<br>**22.5**&nbsp;<sub>(1.78x)</sub> |
| Qwen3.8-Flash-Next UD-IQ4_XS | 212<br>**1376**&nbsp;<sub>(6.50x)</sub> | 201<br>**801**&nbsp;<sub>(3.99x)</sub> | 30.7<br>**28.7**&nbsp;<sub>(0.93x)</sub> | missing |
| Qwen3.8-Flash-Next IQ3_S (78 GB) | 359<br>**1410**&nbsp;<sub>(3.92x)</sub> | 359<br>**643**&nbsp;<sub>(1.79x)</sub> | 42.5<br>**30.7**&nbsp;<sub>(0.72x)</sub> | 44.4<br>**40.4**&nbsp;<sub>(0.91x)</sub> |
| Qwen3.8-27B MTP Q5_K_M (both with MTP) | 1573<br>**1131**&nbsp;<sub>(0.72x)</sub> | 1378<br>**1040**&nbsp;<sub>(0.75x)</sub> | 34.5<br>**45.5**&nbsp;<sub>(1.32x)</sub> | 35.2<br>**32.3**&nbsp;<sub>(0.92x)</sub> |
| OLMoE-1B-7B Q4_K_M (fits VRAM) | missing | 13653<br>**15097**&nbsp;<sub>(1.11x)</sub> | 359.9<br>**359.1**&nbsp;<sub>(1.00x)</sub> | 486.6<br>**498.2**&nbsp;<sub>(1.02x)</sub> |
| Qwen3.6-35B-A3B hybrid (fits VRAM) | 5257<br>**4727**&nbsp;<sub>(0.90x)</sub> | 3762<br>**3445**&nbsp;<sub>(0.92x)</sub> | 146.6<br>**148.6**&nbsp;<sub>(1.01x)</sub> | 150.4<br>**152.4**&nbsp;<sub>(1.01x)</sub> |
| Qwen3.8-Flash-Next IQ1_M | 1247<br>**915**&nbsp;<sub>(0.73x)</sub> | 1212<br>**618**&nbsp;<sub>(0.51x)</sub> | 61.2<br>**60.9**&nbsp;<sub>(1.00x)</sub> | 62.3<br>**62.7**&nbsp;<sub>(1.01x)</sub> |
| Qwen3.8-27B IQ4_NL (dense) | 1748<br>**1562**&nbsp;<sub>(0.89x)</sub> | 1482<br>**1340**&nbsp;<sub>(0.90x)</sub> | 42.4<br>**42.5**&nbsp;<sub>(1.00x)</sub> | 43.3<br>**43.3**&nbsp;<sub>(1.00x)</sub> |
| GLM-5.3-Flash 3.5-bit (137 GB) | missing | missing | missing | missing |

<!--/latest-->
| model | cell | builds stock fork | pp: stock / **fork** (gain) | t/s: stock / **fork** (gain) | ppl: stock / fork |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash 3.0-bit (106 GB) | <sub>12k prompt, `-c 16384` (best pp)</sub> | <sub>b11381 b12719</sub> | 42.8<br>**756**&nbsp;<sub>[(17.7x)](fork/bench/REPRO.md#glm-12k)</sub> | 11.5<br>**15.9**&nbsp;<sub>[(1.38x)](fork/bench/REPRO.md#glm-12k)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| GLM-5.3-Flash 3.0-bit | <sub>1k edit, 128 out</sub> | <sub>b11452 b12825</sub> | 41.5<br>**495**&nbsp;<sub>[(11.9x)](fork/bench/REPRO.md#glm-edit)</sub> | 12.4<br>**19.0**&nbsp;<sub>[(1.53x)](fork/bench/REPRO.md#glm-edit)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| Qwen3.6-35B-A3B hybrid (11.4 GiB) on an 8 GB laptop (4060) | <sub>17-token prompt, 100 out</sub> | <sub>b11514 b12843</sub> | 20.2<br>**98.0**&nbsp;<sub>[(4.9x)](fork/bench/REPRO.md#laptop-17)</sub> | 15.6<br>**56.5**&nbsp;<sub>[(3.6x)](fork/bench/REPRO.md#laptop-17)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.8-Flash-Next UD-IQ4_XS (88 GB) | <sub>1k edit (best pp)</sub> | <sub>b11452 b12364</sub> | 160<br>**599**&nbsp;<sub>[(3.7x)](fork/bench/REPRO.md#qni4-edit)</sub> | 30.6<br>**45.8**&nbsp;<sub>[(1.50x)](fork/bench/REPRO.md#qni4-edit)</sub> | - |
| Qwen3.6-35B-A3B Q2_0 (CPU only, Ryzen 5 3600) | <sub>short prompt, 100 out (CLI)</sub> | <sub>b11302 b11648</sub> | 10.2<br>**37.6**&nbsp;<sub>[(3.7x)](fork/bench/REPRO.md#q2-cpu)</sub> | 6.3<br>**11.3**&nbsp;<sub>[(1.79x)](fork/bench/REPRO.md#q2-cpu)</sub> | - |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>12k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 687<br>**918**&nbsp;<sub>[(1.34x)](fork/bench/REPRO.md#laptop-12k)</sub> | 13.6<br>**45.6**&nbsp;<sub>[(3.4x)](fork/bench/REPRO.md#laptop-12k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>40k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 584<br>**726**&nbsp;<sub>[(1.24x)](fork/bench/REPRO.md#laptop-40k)</sub> | 11.7<br>**35.9**&nbsp;<sub>[(3.1x)](fork/bench/REPRO.md#laptop-40k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.8-Flash-Next UD-IQ4_XS | <sub>12k prompt</sub> | <sub>b11381 b11707</sub> | 201<br>**553**&nbsp;<sub>[(2.8x)](fork/bench/REPRO.md#qni4-12k)</sub> | 29.6<br>**45.9**&nbsp;<sub>[(1.55x)](fork/bench/REPRO.md#qni4-12k)</sub> | - |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>125k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 448<br>**557**&nbsp;<sub>[(1.24x)](fork/bench/REPRO.md#laptop-125k)</sub> | 9.2<br>**25.2**&nbsp;<sub>[(2.7x)](fork/bench/REPRO.md#laptop-125k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| MiMo-V2.6-Flash IQ3_XXS (132 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12250</sub> | 30.0<br>**73.6**&nbsp;<sub>[(2.5x)](fork/bench/REPRO.md#mimo-edit)</sub> | 4.8<br>**9.0**&nbsp;<sub>[(1.88x)](fork/bench/REPRO.md#mimo-edit)</sub> | 1.5717<br>missing |
| Qwen3.6-35B-A3B Q2_0 on an 8 GB laptop (4060) | <sub>short prompt, 100 out (CLI)</sub> | <sub>b11261 b11648</sub> | 41.5<br>**101.9**&nbsp;<sub>[(2.5x)](fork/bench/REPRO.md#q2-laptop)</sub> | 28.9<br>**58.6**&nbsp;<sub>[(2.0x)](fork/bench/REPRO.md#q2-laptop)</sub> | - |
| GLM-5.3-Flash 3.0-bit | <sub>game, 600 out (best t/s)</sub> | <sub>b11381 b12209</sub> | - | 11.7<br>**27.8**&nbsp;<sub>[(2.4x)](fork/bench/REPRO.md#glm-game)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| MiMo-V2.6-Flash IQ3_XXS | <sub>chat, short prompt (best t/s)</sub> | <sub>b11381 b11707</sub> | - | 4.9<br>**10.5**&nbsp;<sub>[(2.1x)](fork/bench/REPRO.md#mimo-chat)</sub> | 1.5717<br>missing |
| Qwen3.8-Flash-Next UD-IQ4_XS | <sub>chat, short prompt (best t/s)</sub> | <sub>b11381 b11365</sub> | 41.8<br>**63.3**&nbsp;<sub>[(1.51x)](fork/bench/REPRO.md#qni4-chat)</sub> | 30.9<br>**60.9**&nbsp;<sub>[(1.97x)](fork/bench/REPRO.md#qni4-chat)</sub> | - |
| GLM-5.3-Flash 3.5-bit (137 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12250</sub> | 26.2<br>**49.0**&nbsp;<sub>[(1.87x)](fork/bench/REPRO.md#glm35-edit)</sub> | 9.7<br>**9.9**&nbsp;<sub>[(1.02x)](fork/bench/REPRO.md#glm35-edit)</sub> | - |
| Qwen3.8-Flash-Next IQ3_S (78 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12128</sub> | 376<br>**683**&nbsp;<sub>[(1.82x)](fork/bench/REPRO.md#qni3-edit)</sub> | 42.7<br>**54.2**&nbsp;<sub>[(1.27x)](fork/bench/REPRO.md#qni3-edit)</sub> | [**1.1113**](fork/bench/REPRO.md#qni3-edit)<br>1.1151 |
| Qwen3.6-35B-A3B hybrid (fits VRAM) | <sub>game, 600 out</sub> | <sub>b11381 b12824</sub> | 85.0<br>**145**&nbsp;<sub>[(1.71x)](fork/bench/REPRO.md#q36-game)</sub> | 153<br>**155**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#q36-game)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| GLM-5.3-Flash 3.5-bit | <sub>game (best t/s)</sub> | <sub>b11381 b11707</sub> | 3.7<br>**5.4**&nbsp;<sub>[(1.46x)](fork/bench/REPRO.md#glm35-game)</sub> | 6.9<br>**10.5**&nbsp;<sub>[(1.52x)](fork/bench/REPRO.md#glm35-game)</sub> | - |
| OLMoE-1B-7B Q4_K_M (fits VRAM) | <sub>game, 600 out</sub> | <sub>b11574 b12824</sub> | 213<br>**315**&nbsp;<sub>[(1.48x)](fork/bench/REPRO.md#olmoe-game)</sub> | 466<br>**472**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#olmoe-game)</sub> | 2.1489<br>[**2.1482**](fork/bench/REPRO.md#olmoe-game) |
| Qwen3.8-Flash-Next IQ3_S | <sub>game (best t/s)</sub> | <sub>b11381 b11707</sub> | 26.5<br>**29.9**&nbsp;<sub>[(1.13x)](fork/bench/REPRO.md#qni3-game)</sub> | 42.1<br>**56.2**&nbsp;<sub>[(1.33x)](fork/bench/REPRO.md#qni3-game)</sub> | - |
| Qwen3.8-Flash-Next IQ1_M | <sub>1k edit</sub> | <sub>b11381 b12128</sub> | 1165<br>**1409**&nbsp;<sub>[(1.21x)](fork/bench/REPRO.md#qn1-edit)</sub> | 65.0<br>**65.9**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#qn1-edit)</sub> | 1.1741<br>missing |
| Qwen3.8-27B MTP Q5_K_M (dense, fits 24 GB), both with MTP | <sub>1k edit</sub> | <sub>b11381 b12193</sub> | 1531<br>**1616**&nbsp;<sub>[(1.06x)](fork/bench/REPRO.md#q27mtp-edit)</sub> | 62.5<br>**64.4**&nbsp;<sub>[(1.03x)](fork/bench/REPRO.md#q27mtp-edit)</sub> | - |
| Qwen3.8-27B MTP Q5_K_M, both with MTP | <sub>game, 600 out</sub> | <sub>b11381 b12193</sub> | 32.4<br>**34.1**&nbsp;<sub>[(1.05x)](fork/bench/REPRO.md#q27mtp-game)</sub> | 74.9<br>**65.7**&nbsp;<sub>(0.88x)</sub> | - |
| Qwen3.8-27B IQ4_NL (dense, fits 24 GB) | <sub>1k edit</sub> | <sub>b11381 b11707</sub> | 1714<br>**1730**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#q27nl-edit)</sub> | 44.3<br>**44.3**&nbsp;<sub>(1.00x)</sub> | - |

\* `pplb` (batched perplexity, 1024-token batches over the same text) because the 1-token `ppl` took hours on this model.

On smaller cards the gain grows, see the next section (an 8 GB laptop: decode 2.7-3.6x; a simulated 16 GB card: GLM 6.5x decode, Qwen3.8-Flash-Next 3.4x on two cards).

**Q2_0 rows:** the 2-bit Qwen3.6 file (`qwen36.gguf`) is no longer on any of our machines, so these two rows are the last measurement of 2026-09-30 (build b11648); the AVX2 Q2_0 dot product is in b12844 and in the merged tree, but it has not been re-measured since.

**Decode right after a long prompt (GLM-5.3-Flash 3.0-bit, 12k prompt):** the decode rates in the table above are the average of the first 32 generated tokens, while the pool is still refilling the blocks the prompt borrowed. How the rate develops over a long answer after a long prompt has not been measured yet (a run with 128, 512 and 2,000 output tokens on b12686, b12825 and b12844 is queued). What is measured: with a short prompt and a 600-token answer (cache warm, nothing to refill) b12825 decodes at 15.6 t/s against 18.5 t/s for b12686, so part of the slowdown is not the refill.

**Known issue in b12844 (found while updating this table):** after a 12k prompt, Qwen3.8-Flash-Next decodes at 35 t/s where b12686 gives 50 t/s, and GLM-5.3-Flash at 14.3 vs 16.2 t/s (the prompt itself is 1.3-1.4x faster). Not fixed yet; the prefill changes of this release are the suspected cause, a fix is next.

**Results, 2x RTX 3090 (one AM4 CPU PCIe 4.0 x16, one X570 chipset x4 slot) + Ryzen 7 3700X + 125 GB DDR4,
CPU frequency governor `performance`, single stream, temp 0.** Short = 1500-token chat reply to "write smallest html tetris game". Long =
12k-token code prompt (llama.cpp sources): prompt processing, then decode. The prompts and a script to
reproduce these tests are in [`fork/bench/`](fork/bench/).

## Release "parallel-prefill": small cards first (2026-10-14)

Each build runs on **its own defaults**: only `-np 1` (and `-c` for stock, whose 4096 default rejects long prompts). No thread, batch or `--moe` settings on either side.
`prompt` = prompt processing t/s, `gen` = generation t/s. Single stream, temperature 0. Multipliers are fork / stock.

**An 8 GB laptop** (RTX 4060 Laptop 8 GB, 31 GB RAM, Windows 10, `llama-cli`). Stock = upstream b11514 (`de7fa0a3c`), the newest upstream binary tried on this machine.
Qwen3.6-35B-A3B-GSQ-hybrid is 11.4 GiB, so it does not fit the card; stock crashes or crawls here, the fork picks its own placement and micro-batch.

| model, prompt | stock prompt / gen | fork prompt / gen | prompt | gen |
|---|---|---|---|---|
| Qwen3.6-35B-A3B 11.4 GiB, 17 tokens | 20.2 / 15.6 | 98.0 / 56.5 | **4.9x** | **3.6x** |
| Qwen3.6-35B-A3B, 12k tokens | 687 / 13.6 | 918 / 45.6 | 1.34x | **3.4x** |
| Qwen3.6-35B-A3B, 40k tokens | 584 / 11.7 | 726 / 35.9 | 1.24x | **3.1x** |
| Qwen3.6-35B-A3B, 125k tokens | 448 / 9.2 | 557 / 25.2 | 1.24x | **2.7x** |
| Qwen3.8-27B IQ3_S 12.1 GB (dense), 10 tokens | 19.0 / 3.0 | 17.4 / 3.1 | 0.92x ‡ | 1.03x |
| Qwen3.8-27B IQ3_S (dense), 12k tokens | 270 / 2.2 | 258 / 2.2 | 0.96x ‡ | 1.00x |

‡ Dense models get no gain from the fork on a small card (no experts to cache; the layers that do not fit go through the CPU as in stock).
The 4-8% are inside this laptop's run-to-run noise (stock itself measured 270, 312, 313 and 337 on the 12k cell), the 27B is being re-measured with 3+ runs per build.

**A 16 GB card** (RTX 3090 limited to 16 GB with a memory holder, `-dev CUDA1`, x16 link) and **two 16 GB cards** (x16 + x4), Ryzen 7 3700X, 125 GB RAM. Stock = upstream b11574.

| model (size) | cell | cards | stock gen | fork gen | gen | stock prompt | fork prompt | prompt |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit (106 GB) | game: 600 tokens | 1 x 16 GB | 1.7 | 11.2 | **6.5x** | 1.2 | 13.8 | **11x** |
| Qwen3.8-Flash-Next | 4k prompt | 1 x 16 GB | 24.8 | 33.5 | 1.35x | 159 | 1214 | **7.6x** |
| Qwen3.8-Flash-Next | game | 2 x 16 GB | 15.0 | 51.1 | **3.4x** | 8.4 | 78.6 | **9.4x** |
| Qwen3.8-Flash-Next | 4k prompt | 2 x 16 GB | 16.5 | 29.1 | 1.76x | 72 | 1067 | **15x** |
| Qwen3.6-35B-A3B hybrid (fits) | game | 1 x 16 GB | 150.0 | 144.1 | 0.96x ¶ | 121 | 161 | 1.33x |
| Qwen3.6-35B-A3B hybrid (fits) | 4k prompt | 1 x 16 GB | 141.1 | 135.3 | 0.96x ¶ | 3009 | 2622 | 0.87x ¶ |
| Qwen3.6-35B-A3B hybrid (fits) | 4k prompt | 2 x 16 GB | 142.3 | 142.2 | 1.00x | 3370 | 3137 | 0.93x ¶ |
| OLMoE-1B-7B Q4_K_M (fits) | game | 1 x 16 GB | 460.9 | 454.1 | 0.99x | 213 | 322 | 1.5x |

¶ **Known gap:** a model that fits entirely in VRAM runs 0-4% slower on decode and 7-13% slower on prompt than stock (fork overhead on a path that has nothing to cache or to split).
The fix is queued: models that fit leave every fork hook, so the path is stock's.

**CPU only** (Ryzen, 12 cores, 60 GB RAM, clang, no GPU, `-dev none`, 3 runs each, each build on its defaults). Stock = upstream master 37d6d653e (b11631).
There is nothing to cache or to split without a GPU. What the CPU-only build adds is the kernels: **a 2-bit Q2_0 model runs 1.8x faster on decode and 3.7x faster on prompt than upstream (top table), because upstream has only a scalar dot product for that type on x86**; for the other quantizations the gain is the thread and batch tuning (+13 to +18% prompt, +5% decode on the 35B hybrid below). the CPU archives of the release are the same binaries without the CUDA backend.

| model, cell | builds stock fork | pp: stock / **fork** | t/s: stock / **fork** |
|---|---|---|---|
| Qwen3.6-35B-A3B hybrid (12 GB), short prompt | <sub>b11631 b12844</sub> | 32.1<br>**36.2**&nbsp;<sub>[(1.13x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.7<br>**11.3**&nbsp;<sub>[(1.06x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid (12 GB), 1k edit | <sub>b11631 b12844</sub> | 41.9<br>**49.3**&nbsp;<sub>[(1.18x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.4<br>**10.9**&nbsp;<sub>[(1.05x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid, short prompt, merged tree | <sub>b11631 b13086</sub> | 32.1<br>31.6&nbsp;<sub>(0.98x)</sub> | 10.7<br>**11.1**&nbsp;<sub>[(1.04x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid, 1k edit, merged tree | <sub>b11631 b13086</sub> | 41.9<br>**46.0**&nbsp;<sub>[(1.10x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.4<br>**10.8**&nbsp;<sub>[(1.04x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| OLMoE Q4_K_M, short prompt | <sub>b11631 b13086 (merged tree)</sub> | 144<br>**145** | 47.6<br>**48.0** |
| OLMoE Q4_K_M, 1k edit | <sub>b11631 b12844 (release)</sub> | 198<br>**184**&nbsp;<sub>(0.93x)</sub> | 36.7<br>**36.1**&nbsp;<sub>(0.98x)</sub> |
| OLMoE Q4_K_M, 1k edit | <sub>b11631 b13086 (merged tree)</sub> | 198<br>**196** | 36.7<br>**33.5**&nbsp;<sub>(0.91x)</sub> |
| Qwen3.8-27B IQ3_S, short prompt | <sub>b11631 b13086 (merged tree)</sub> | 5.8<br>**6.0** | 1.7<br>**1.8** |
| Qwen3.8-27B IQ3_S, 1k edit | <sub>b11631 b13086 (merged tree)</sub> | 10.2<br>**10.3** | 1.7<br>**1.8** |

Two open items from this table: the release is 7% slower on prompt on CPU, and the merged tree is 9% slower on decode after a 1k prompt (neither `autotune=0` nor `--fork off` changes it).

**Fixed in this release:** a clean start on small cards. The previous release crashed with "CUDA error: out of memory" at the first kernel on an 8 GB laptop for models larger than VRAM (the cache pool was sized
from the whole model, then the compute buffer of a context-sized micro-batch did not fit). Now the pool is counted once, a 512 MiB / 8% margin is kept on cards of 12 GiB and less, the scheduler no longer reserves 12 GB of
split bookkeeping at a 14k-token micro-batch (which Windows refused), and a start that still runs out of memory restarts itself with half the micro-batch.

**Stock llama.cpp vs this fork**, tokens/s, model already in RAM (*). "mmap": weights
memory-mapped (models bigger than RAM, `llama-cli`); "pinned": weights in pinned RAM, what
`llama-server` does by itself when the model fits in RAM; "+ MTP": plus the model's MTP draft head
(`-md`, automatic draft depth). Multipliers vs stock; best number per row in bold (links to how to reproduce it).

| model | test | stock (b11041) | fork, mmap (b11327-b11367) | fork, mmap + MTP (b11367) | fork, pinned (b11341) | fork, pinned + MTP (b11367) |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit [Q4_K attn](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF), 106 GB; `llama-server` default: pinned | short: decode | 13.8 | 20.4 (1.48x) | 18.2 (1.32x) ‡ | [**21.1 (1.53x)**](fork/bench/) | 18.2 (1.32x) ‡ |
| | long: prompt processing † | 217 | 180 (0.83x) | | [**262 (1.21x)**](fork/bench/) ||
| | long: decode | 12.3 | 15.6 (1.27x) | | [**16.3 (1.33x)**](fork/bench/) ||
| MiMo-V2.6-Flash-RL IQ3_XXS, 132 GB; `llama-server` default: mmap (bigger than RAM) | short: decode | 4.0 | [**10.1 (2.54x)**](fork/bench/) | no gain ¶ | - (bigger than RAM) | - |
| | long: prompt processing † | [**136**](fork/bench/) | 133 (0.98x) | | - | - |
| | long: decode | 4.2 | [**8.3 (1.98x)**](fork/bench/) | | - | - |
| Qwen3.8-Flash-Next UD-IQ4_XS, 88 GB; `llama-server` default: pinned | short: decode | 27.7 | 47.1 (1.70x) | 58.3 (2.11x) | 46.5 (1.68x) | [**60.4 (2.18x)**](fork/bench/) |
| | long: prompt processing † | 500 | 372 (0.74x) | 353 (0.71x) | [**538 (1.08x)**](fork/bench/) | 500 (1.00x) |
| | long: decode | 25.3 | 38.4 (1.52x) | 40.1 (1.59x) § | [**42.3 (1.67x)**](fork/bench/) | 28.0 (1.11x) § |
| Qwen3.8-27B IQ4_NL (dense, fits VRAM), 16 GB | short / long prompt | 44.7 / 1726 | 44.8 / 1811 (no cache needed) | | ||
| OLMoE-1B-7B Q4_K_M (fits VRAM), 4 GB | short: decode | 504 | 504 (no cache needed) | | ||

¶ MiMo's built-in MTP (3 dense layers, no experts): the depth tuner keeps depth 0 (drafting was
not 5% faster in its probes), so no measurable gain; MiMo's numbers vary between sessions (8.9-10.1)
because the model is bigger than RAM.
§ The 12k test generates only 32 tokens: too few for the MTP depth tuner to finish a comparison,
so these decode numbers are mostly probing overhead and vary a lot (28-40 t/s); MTP pays off on
longer replies (short-prompt row).
\* Model already in RAM (OS page cache), as on a server after its first request. The
first run after switching to another large model is slower, once, while the file is
read from disk. MiMo (132 GB) can't fully stay in 125 GB RAM, so it always reads part of
the model from disk. Multipliers are vs stock.
† Prompt processing streams the experts over PCIe to one GPU, so the slot limits it (here a
PCIe 4.0 x16 CPU slot; the second card sits in an X570 chipset x4 slot). A board with more
x16 slots, or another GPU, should raise it; splitting prompt processing across both GPUs'
links is the next milestone.
‡ MTP on GLM-5.3-Flash is slower on this box (pinned: 18.2 vs 21.1 t/s without; mmap: 18.2 vs 20.4; tuner picks depth 1; the model is 2.3x
the VRAM, so verifying drafts adds CPU work and the draft takes cache VRAM). The fork
doesn't load the draft here by default. Another GPU (more VRAM) should make it pay off,
as it does for Qwen (1.9x VRAM).

stock llama.cpp -> this fork. The fork numbers match a plain `llama-server -m model` within ~4%
(auto mode, see Run). Stock can't load GLM-5.3-Flash, so its stock column is
this fork without the cache. Models that fit in VRAM don't use the cache and run the
same (identical output). With the default `schedutil`/`powersave` governor, decode can
be lower, mostly where the CPU computes missed experts. Prompt processing: stock keeps whole layers in VRAM and
never uploads them, this fork uploads experts, so it depends on upload speed. With pinned
weights (the server default when the model fits in RAM) GLM-5.3-Flash is 1.21x stock, Qwen
1.08x; with mmap both are 17-26% below stock.
Decode speed depends on how often generated tokens reuse cached experts: ~75% of
experts are hits on GLM chat, ~95% on Qwen.

## Moved from the README on 2026-11-17 (the first rows on the current machine, measured 2026-11-12; a full gainers table for this machine follows in the README)

Lenovo P620: Threadripper PRO 3975WX (32 cores), 256 GB DDR4-3200 in 8 channels, 2x RTX 3090 on PCIe 4.0 x16, power-limited to 280 W each. Against the previous machine this has 4x the memory bandwidth and the full PCIe link on the second card.

| model | test | stock decode (t/s) | ours decode (t/s) | stock prompt (t/s) | ours prompt (t/s) | decode gain | prompt gain | stock build | ours build | answers (ours, stock) |
|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash 3-bit | 12k/1k | 30.01 | **36.65** | 115.6 | **1458.0** | 1.22x | **12.6x** | b12317 | b15065 | [26477](../outputs/26477.txt), [26475s](../outputs/26475s.txt) |
| MiMo-V2.6-Flash 3-bit | 3k/1k | - | 37.96 | - | 660.7 | - | - | - | b15065 | [26479](../outputs/26479.txt) |
| MiMo-V2.6-Flash 3-bit | short/1k | - | 41.19 | - | - | - | - | - | b15065 | [26481](../outputs/26481.txt) |

Measured 2026-11-12, two runs per cell, the faster one shown; `-` = not measured on this machine yet (the other models follow). Stock gained the most from this machine: it computes the experts on the CPU, so the 8-channel memory made it 2.1x faster in decode and 2.3x in prompt processing than on the previous machine (MiMo 12k/1k: 13.99 to 30.01 t/s, 49.4 to 115.6 prompt t/s); ours gained 1.38x and 1.42x (26.6 to 36.65 t/s, 1025 to 1458 prompt t/s).

## Moved from the README on 2026-11-12 (measured on the previous machine: Ryzen 7 3700X, 128 GB DDR4-3200 dual channel, RTX 3090 on x16 + RTX 3090 on x4)

### Status note of 2026-10-23

Status Update:23:10:2026. Right now multi gpu decode is supported all models multigpu dense split 
prefill is implemented for glm and gives gains only from prompts 7k up on 2x3090 at least on my rig.

In other words tha gains you see bellow are not from multigpu yet just from more efficient vram mmgmt
ie the same 20x speedup in mimo is happening on single 5090 too. In a sense its good news so people with one gpu will have fun too ;)
And that multigpu fun is just starting too. For glm 2nd gpu is adding +30%  on pp so far

**Same VRAM, different use.** Each GLM token uses 8 of 288 experts per layer. Stock
fills VRAM with whole layers (~14 of 42), mostly experts the current token doesn't
touch, so the CPU does most expert work and the GPUs wait. This fork fills the same
VRAM with the ~100 most-used experts of every layer, so most expert work runs on the
GPUs, at the same time as the CPU handles the misses.

### Other notable changes: the two-card prompt split


**Where the two-card dense split starts to pay.** Per prompt, the planner measures one card and two cards and takes the faster; it does not use a fixed size. The medians of all runs since 2026-10-20 (220 + 280 W, answers not looped; each cell has 2 to 50 runs from several builds, so differences under 10% are noise) show where it wins:

| model | prompt | one card (pp t/s) | two cards (pp t/s) | official stock (pp t/s, build) | one card vs stock | two cards vs stock | best of our builds |
|---|---|---|---|---|---|---|---|
| Qwen3.8 Next IQ4_XS | 4k | 1684 | 1350 | 209, b11887 | 8.1x | 6.5x | pre7 |
|  | 8k | 1663 | 1513 | 213 (9k prompt), b11829 | 7.8x | 7.1x | pre9 |
|  | 11k | 1581 | 1824 | 211 (10k prompt), b11829 | 7.5x | 8.6x | rc5 |
|  | 12k | 1498 | 1648 | 213, b11829 | 7.0x | 7.7x | spass15 |
|  | 16k | 1543 | 1115 | 208, b11887 | 7.4x | 5.4x | pre9 |
|  | 30k | 1553 | 1554 | fail: context too small | - | - | ncard6 |
|  | 60k | 1600 | 1668 | fail: context too small | - | - | pre10 |
| Qwen3.8 Next IQ3_S | 12k | 1443 | - | 379, b11829 | 3.8x | - | relfx1 |
|  | 60k | 1546 | 1864 | fail: context too small | - | - | um6g |
| GLM-5.3-Flash 3.0-bit | 4k | 703 | 639 | 42 (5k prompt), b11829 | 16.7x | 15.2x | pre9 |
|  | 12k | 658 | 609 | 43, b11829 | 15.3x | 14.2x | ceil1 |
|  | 16k | 660 | 775 | 42, b11887 | 15.7x | 18.4x | pre10 |
|  | 30k | 557 | 686 | 42, b11887 | 13.3x | 16.3x | ncard6 |
|  | 60k | 566 | 694 | - | - | - | pre10 |
| MiMo-V2.6-Flash 3-bit | 4k | 905 | 903 | 48 (3.4k prompt), b11866 | 18.9x | 18.8x | pre13 |
|  | 8k | 1062 | 999 | 50, b11866 | 21.2x | 20.0x | relmf1 |
|  | 12k | 988 | 1150 (1 run) | 49, b11829 | 20.2x | 23.5x | rc5 |
|  | 16k | 906 | 1094 | 49, b11887 | 18.5x | 22.3x | pre9 |
|  | 60k | 846 | 841 | - | - | - | pre12 |

Stock cells: official ggml-org binaries with their own defaults (`-np 1` only, micro-batch and context auto); where the stock prompt size differs from the row it is named. **fail: context too small** = stock sizes its context itself (32768 tokens at 30k, 63488 at 60k on Qwen) and refuses a prompt that does not fit (a 30k prompt is 33.6k tokens, a 60k prompt 68.3k): `request (68337 tokens) exceeds the available context size (63488 tokens)`. It needs `-c` set by hand, which was not done here, so stock cannot serve these prompts as shipped; ours sizes the context per request. Stock is CPU-bound, so its prompt speed is nearly the same at every size (GLM 42-43, MiMo 48-50, Qwen IQ4_XS 208-213). "-" = not measured (stock at 60k would take 20 minutes or more per cell on GLM and MiMo; GLM 30k took 12 minutes).

- **The split starts to pay at about 11k-16k prompt tokens**: Qwen3.8 Next from 11k, GLM and MiMo from 16k (MiMo possibly 12k, one run). Below about 8k one card is as fast or faster (the second card sits in a x4 slot and the hand-offs between cards cost more than the shorter pass saves). The auto setup takes one card there.
- **Largest gains: GLM +18% to +26% (16k to 60k), MiMo +21% at 16k, Qwen3.8 Next +15% at 11k.** Qwen3.6 and OLMoE fit in VRAM and show no split gain.
- **Big contexts at full speed are one of the main benefits of this fork.** Stock sizes the context once at start from the free memory (32768 and 63488 tokens on the Qwen IQ4_XS and IQ3_S cells above); a longer prompt is refused. Ours claims the KV it needs per request from the same buffer as the experts and gives it back at the end of the request, so no `-c` has to be guessed and a big context costs nothing while it is not in use. Measured prompts, prompt speed (pp t/s), on this machine: Qwen3.8 Next IQ4_XS 60k 1600-1668 (stock: refused with its own context), Qwen3.6-35B 200k 3259 (decode 72.2 t/s), MiMo 200k 794 (decode 17.3 t/s) and 400k 363; the 200k cells with the antiloop sampler (repeat penalty 1.15) so the answers are real text, the speed is not touched by it. Details of the KV handling are in [fork/docs/features.md](fork/docs/features.md).
- The split is chosen by the planner and needs no switch; `--moe dense-split=0` forces one card, `=1` forces the split.


### What each feature contributes


Each row is one paired A/B on the same binary and settings, measured when the feature was added (2x RTX 3090 unless noted; older builds, one A/B each, so read them as the size of the effect, not as additive shares of the table above). Method and run notes per feature: [fork/docs/features.md](fork/docs/features.md).

| feature | test | without | with | change |
|---|---|---|---|---|
| Expert cache in free VRAM | decode, models that do not fit | stock (no expert cache) | the gainers table above | +20% to +130% decode |
| Pinned weights (server default) | GLM 12k prompt, prompt speed | 180 pp t/s | 262 pp t/s | +45% |
| | Qwen 3.8 Next 12k prompt, prompt speed | 372 pp t/s | 538 pp t/s | +45% |
| Big micro-batch (`-ub` 1024 to 4096) | GLM 12k prompt, prompt speed | 176.6 pp t/s | 370.1 pp t/s | 2.1x |
| Micro-batch buffer lent to the prompt pass only | GLM 12k prompt then decode, `-ub` 4096 | 17.81 t/s (static buffer) | 18.99 t/s | +6.6% decode, same prompt speed (368.9 vs 367.6) |
| Prefill preheat | GLM 12k prompt then 256 tokens, 1 GPU, decode | 13.5 t/s (cache hits 43%) | 15.3 t/s (55%) | +12.7% |
| Learned hot experts kept between runs | GLM first request after restart, decode | 20.5-20.9 t/s | 21.8 t/s | +4-6% |
| Global expert pool (vs fixed slots per layer) | decode with 20-35% of the experts in VRAM | 132.5-177.5 t/s | 137.3-181.7 t/s | +2.1% to +3.6% (no change when most of the model fits) |
| Dense split over two cards (forced path) | Qwen 3.8 Next 4k prompt, prompt speed, 180+220 W | 1065 and 1113 pp t/s (run ids 12190, 12191) | 1274 and 1275 pp t/s (12193, 12194) | +17% (held back by the second card sitting in a slow x4 slot, 6 GB/s against 13 GB/s on the x16 card; cards on equal links gain more) |

Which features drive which number: the **expert cache** is what lifts decode over stock; **pinned weights and the big micro-batch** (with its buffer lent only to the prompt pass, so decode keeps its cache) are what lift prompt speed. The rest are smaller, single-digit effects.

