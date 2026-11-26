# Rocket LLama 

# up to 59x faster input tokens\s and 22x output t\s especially on best big models 

<img width="1894" height="621" alt="image" src="https://github.com/user-attachments/assets/a6be5bb1-7bb7-4b63-b25b-5c4112d7dc64" />

**Mixture-of-experts models larger than your GPUs, at GPU speed.** A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp): the experts the tokens use most stay in VRAM, the rest stream from a pinned RAM pool and the SSD, and long prompts run in large passes on every GPU. On the same machine it runs Kimi K3.1 Pro (580 GB) 59x faster on prompts and 8.9x faster on answers than stock llama.cpp.

```sh
llama-server -m MODEL.gguf      # everything automatic; models that fit in VRAM run as in upstream
```

**Support:** if this fork helps you, please support my open-source development (nothing fancy, just rent, food and hardware): [paypal.me/neuralll](https://paypal.me/neuralll)

**Download:** [latest release](https://github.com/neurall/llama.cpp/releases/latest). Linux `tar.gz` and Windows `zip`, each with the CPU, CUDA, Vulkan and HIP backends (the Windows zip carries the ROCm runtime). NVIDIA cards also need the `cudart-` archive when CUDA 13 is not installed.

## Biggest gains against stock llama.cpp

Each row: the official stock binary (its own defaults, `-np 1`) against this fork, same machine, same request; the best valid run of ours among our builds (the build column names it). In the t/s and pp columns the first line is stock, the second ours. **12k/1k** = a 12.4k-13.8k-token code-review prompt with 1024 tokens out, **short/1k** = a short coding request.

| model | test | t/s | pp | gain t | gain pp | ppl | build | stock | output |
|---|---|---|---|---|---|---|---|---|---|
| Kimi K3.1 Pro IQ1_S (580 GB)‡ | 13k/1k | 0.31<br>2.74 | 1.46<br>86.0 | +784% 8.9x | +5791% 58.9x | -<br>- | b16028 | b12426 | [29437s](fork/outputs/29437s.txt)<br>[31951](fork/outputs/31951.txt) |
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
| Inkling 2 Mini IQ4_XS (fits one card)§ | short/1k | 18.39<br>80.62 | 43.9<br>106.0 | +338% 4.4x | +141% 2.4x | -<br>- | b16902 | b12670 | [34081s](fork/outputs/34081s.txt)<br>[35613](fork/outputs/35613.txt) |
| Inkling 2 Mini IQ4_XS + DSpark draft§ | short/1k | 18.39<br>87.10 | 43.9<br>77.4 | +374% 4.7x | +76% 1.8x | -<br>- | b16903 | b12670 | [34081s](fork/outputs/34081s.txt)<br>[35698](fork/outputs/35698.txt) |
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
§ Inkling 2 Mini rows: 1x RTX 3090 of the Threadripper machine, 2026-11-24/26; stock b12670 at its defaults (`-c 32768 -b 512`, answer stopped at 263 tokens). Stock rows at 12k and edit follow.  
‡ Kimi K3.1 Pro row: Threadripper PRO 3975WX, 256 GB DDR4-3200, the same two RTX 3090 (2026-11-20; stock 2026-11-16, its answer stopped at 357 tokens). Column order as above: first line stock, second line ours.

All rows, all builds: [fork/docs/perf-history.md](fork/docs/perf-history.md).

## This release against the previous one and stock

2x RTX 3090, Threadripper PRO 3975WX, 256 GB DDR4-3200 (2 GPUs; Qwen 3.6 fits one), same session, 12k/1k, run 1 / run 2, stock = the last official stock run of the same cell on this machine.

| model | stock i/s, o/s | b16046 i/s | this release i/s | b16046 o/s | this release o/s |
|---|---|---|---|---|---|
| MiMo 2.6F 3-bit | 115.6, 30.01 (b12317) | 1417.0 / 1456.6 | 1365.8 / 1367.2 | 42.93 / 43.92 | 45.68 / 45.13 |
| Qwen 3.8 Next IQ4_XS | 511.1, 51.42 (b12669) | 2268.6 / 2669.6 | 2296.2 / 2575.6 | 77.31 / 78.23 | 77.52 / 78.58 |
| GLM 5.3 3.5-bit | - | 968.5 / 1043.0 | 968.3 / 1009.3 | 31.66 / 30.98 | 30.05 / 31.04 |
| GLM 5.3 3-bit | - | 915.7 / 989.4 | 912.1 / 954.8 | 32.85 / 33.04 | 33.58 / 34.23 |
| Qwen 3.6 35B-A3B (fits VRAM) | - | 8302.0 / 8947.5 | 8209.2 / 8802.7 | 160.25 / 160.90 | 161.01 / 161.11 |
| DeepSeek 4.1 IQ2_XXS | - | 750.8 / 779.6 | 694.9 / 710.4 | 37.34 / 37.37 | 37.05 / 37.50 |

## Against Strata

Where both run the same model, this fork is comparable with [Strata](https://github.com/Niko1221/Strata) and [Adamant](https://github.com/nixcat-dev/adamant) (a [Basalt](https://github.com/jesdga95/basalt) fork). The models with the biggest gains above are in none of their model lists (their READMEs, 2026-11-26): Strata 0.2.22 lists Qwen3.8-Flash-Next, Hy4-Flash, Qwen3.6-35B-A3B and Inkling 2 Mini; Basalt lists Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B, Hy4-Flash and Inkling 2 Mini, on Blackwell GPUs only; Adamant, its port to Ada and Ampere cards, lists Qwen3.8-Flash-Next. Of these engines, Kimi K3.1, GLM 5.3 and MiMo 2.6 run only here.

The same machine, one session (2026-11-26), 1024 tokens out, the better of two runs. This fork: release b16903. Strata: 0.2.21 with engine 0.1.60 (Qwen), 0.2.20 (Inkling), its own packs. **Cards** = the cards the engine used: with both cards visible, Strata ran on one.

| model | engine | cards | 12k/1k i/s | 12k/1k o/s | short/1k o/s |
|---|---|---|---|---|---|
| Qwen 3.8 Next IQ4_XS (87 GiB) | this fork | 2 | **2603.7** | 78.72 | 80.15 |
| | this fork + MTP draft | 2 | 2038.8 | 77.19 | 74.03 |
| | Strata | 1 | 2533.8 | 84.10 | 85.80 |
| Qwen 3.8 Next IQ4_XS, one card visible | this fork | 1 | **2399.2** | 69.38 | 68.21 |
| | this fork + MTP draft | 1 | 2073.1 | 55.55 | 56.30 |
| | Strata | 1 | 2106.6 | 75.20 | 77.30 |
| Inkling 2 Mini IQ4_XS (fits one card) | this fork | 1 | **1357.4** | 69.22 | - |
| | this fork + DSpark draft | 1 | 1296.7 | **86.35** | - |
| | Strata + draft | 1 | 918.9 | 69.50 | - |
| | Strata, no draft | 1 | 918.7 | 69.50 | - |

This fork reads prompts fastest in every row and leads Inkling 2 Mini with its DSpark draft. On Qwen 3.8 Next, Strata decodes 7-13% faster than this fork without drafts; this fork's own MTP draft is slower than no draft there (the next release turns a slower draft off by itself).

## Where it wins, where it does not

- **Wins:** MoE models larger than the free VRAM. The less of the model fits, the bigger the gain, down to models larger than RAM that run partly from the SSD (Kimi K3.1: 580 GB on 256 GB of RAM). Long prompts. Two or more GPUs without NVLink.
- **Stock speed:** models that fit in VRAM, and dense models: the fork's defaults only apply when a MoE model is larger than the free VRAM.
- **Drafts** (MTP, DSpark): Inkling 2 Mini with its DSpark draft decodes 25% faster (86.4 against 69.1 o/s, 12k/1k, 1x RTX 3090). On Qwen 3.8 Next the MTP and DSpark drafts are not faster than no draft yet.
- **8 GB cards:** Inkling 2 Mini runs its experts on the CPU there for now (RTX 4060 laptop: 12k/1k 30 i/s, 12.9 o/s); its fused experts enter the expert cache in the next release.
- **Vulkan** runs at about half the CUDA speed on the same NVIDIA card. HIP and Vulkan on AMD iGPUs work (Radeon 780M, Windows: Inkling 2 Mini 12k/1k 96.4 i/s and 7.8 o/s on Vulkan).

## Quick start

```sh
llama-server -m MODEL.gguf                                            # OpenAI-compatible server on :8080
llama-server -m MODEL.gguf -md DRAFT.gguf --spec-type draft-dspark    # with a draft model
```

| switch | effect |
|---|---|
| `--fork off` | upstream behaviour for everything |
| `--moe fixed=1` | repeatable output for benchmarks |
| `--moe dense-split=0` / `=1` | force one card / the dense split over the cards |
| `--moe-expert-cache 0` | expert cache off |
| `GET /status` | server phase with live ms per token |

Every `--moe` key: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md). How it works and the per-feature measurements: [fork/docs/features.md](fork/docs/features.md). Perplexity against stock (equal where the pass shape is the same): [fork/docs/perf-history.md](fork/docs/perf-history.md).

## Known limits

- Measured on 2x RTX 3090 (Ryzen 7 3700X with 128 GB, later Threadripper PRO 3975WX with 256 GB), one rented RTX 5090 and one notebook (ASUS GA403UV: RTX 4060 laptop 8 GB, Radeon 780M, 32 GB RAM); other hardware is untested.
- The first runs of a model are slower: the hot-expert state is learned from use and kept in `moe-state.ini`.
- Output can differ between runs even at temperature 0 (a cached expert runs on the GPU, a missed one on the CPU).

## Credits and thanks

- **Georgi Gerganov ([ggerganov](https://github.com/ggerganov)) and every contributor to [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml**: this fork is a small layer on a fantastic project that has given so much to so many people for so many years. Thank you.
- **[@csantiago78](https://github.com/csantiago78)** and their pull request [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) ("GPU-resident LRU cache for host-offloaded MoE expert weights"), the starting point of this fork's expert cache. It has since diverged (window and admission instead of LRU, one global pool instead of fixed slots per layer), but the idea and the first working version are theirs.
- **People who tried this fork and reported back**: [@MichaelDietzel](https://github.com/MichaelDietzel) (tested on a 2080 Ti and asked for documentation of the `--moe` keys and of multi-GPU use), [@Volunteer-1](https://github.com/Volunteer-1) (Windows build report and the same documentation questions; every flag is now documented in [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md)), [@homeofe](https://github.com/homeofe) (the small-VRAM data point where the cache was slower than stock, and the AVX2 dot products for IQ3_S and IQ3_XXS in `ggml-cpu`, merged into this fork, MIT license) and [@yuffx](https://github.com/yuffx) (the PCIe-bound explanation of that result).
- **[Strata](https://github.com/Niko1221/Strata) by [@Niko1221](https://github.com/Niko1221) and the Strata contributors**: ideas for a few of this fork's later decode tweaks, worth up to about a fifth more decode speed on some models, on top of the lead over stock shown in the table above: a dedicated per-core thread pool for the experts the CPU computes (the fork computed them through the CPU backend before), split kernels for the hyper-connection mix, and overlapping the parts of a decode token on the card. We reimplemented these ideas in our own code and carried some of them over to models Strata does not support yet, GLM 5.3 and MiMo 2.6, with Kimi K3.1 next; the gains there are still small. Strata's speeds are a benchmark target we measure against.

---

*The original llama.cpp README follows below.*

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

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
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
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
