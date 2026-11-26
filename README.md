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

The best measured run of ours per model against the official stock binary (its own defaults, `-np 1`), same machine, same request. **i/s** = prompt tokens per second, **o/s** = generated tokens per second; **12k/1k** = a 12.4k-13.8k-token code-review prompt with 1024 tokens out, **short/1k** = a short coding request.

| model (file) | test | stock i/s | ours i/s | stock o/s | ours o/s | prompt | decode | ours / stock | machine | answers |
|---|---|---|---|---|---|---|---|---|---|---|
| Kimi K3.1 Pro IQ1_S (580 GB) | 13k/1k | 1.46 | 86.0 | 0.31 | 2.74 | 59x | 8.9x | b16028 / b12426 | B | [ours](fork/outputs/31951.txt), [stock](fork/outputs/29437s.txt) |
| MiMo 2.6F 3-bit (108 GiB) | 12k/1k | 49.4 | 1241.1 | 13.99 | 25.71 | 25.1x | 1.84x | b14965 / b11829 | A | [ours](fork/outputs/24369.txt), [stock](fork/outputs/18221s.txt) |
| DeepSeek 4.1 IQ2_XXS (232 GiB) | short/1k | - | - | 1.47 | 32.95 | - | 22.4x | b14939 / b11887 | A | [ours](fork/outputs/24094.txt), [stock](fork/outputs/19818s.txt) |
| MiMo 2.6F 3-bit | 12k/1k | 141.3 | 3016.3 | 6.83 | 36.13 | 21.3x | 5.3x | b13348 / b11799 | C | [ours](fork/outputs/14701.txt), [stock](fork/outputs/14700s.txt) |
| GLM 5.3 3-bit (109 GiB) | 16k/1k | 41.7 | 869.6 | 12.53 | 19.56 | 20.9x | 1.56x† | b14939 / b11887 | A | [ours](fork/outputs/24065.txt), [stock](fork/outputs/19572s.txt) |
| GLM 5.3 3.5-bit (128 GiB) | 12k/1k | 34.3 | 614.9 | 10.72 | 15.18 | 17.9x | 1.42x | b14954 / b11829 | A | [ours](fork/outputs/24296.txt), [stock](fork/outputs/18275s.txt) |
| Qwen 3.8 Next IQ4_XS (87 GiB) | 15k/1k | 176.8 | 1934.1 | 29.14 | 52.31 | 10.9x | 1.80x | b13845 / b12131 | A | [ours](fork/outputs/23477.txt), [stock](fork/outputs/23452s.txt) |
| Qwen 3.8 Next IQ3_S (78 GiB) | 3k/1k | 387.2 | 1777.2 | 44.59 | 56.18 | 4.6x | 1.26x | b14939 / b11829 | A | [ours](fork/outputs/24122.txt), [stock](fork/outputs/18270s.txt) |

Machines: **A** = 2x RTX 3090, Ryzen 7 3700X, 128 GB DDR4 (until 2026-11-11). **B** = the same two RTX 3090 (PCIe 4.0 x16 each), Threadripper PRO 3975WX, 256 GB DDR4-3200 (since 2026-11-12). **C** = 1x RTX 5090, a rented server, stock and ours minutes apart. With 256 GB of RAM stock keeps more of a model in the page cache, so the same models gain less on B than on A (below). † stock was asked for 64 tokens there, so that decode gain is for information. All rows, all builds: [fork/docs/perf-history.md](fork/docs/perf-history.md).

## This release against the previous one and stock

Machine B (2 GPUs; Qwen 3.6 fits one), same session, 12k/1k, run 1 / run 2, stock = the last official stock run of the same cell on B.

| model | stock i/s, o/s | b16046 i/s | this release i/s | b16046 o/s | this release o/s |
|---|---|---|---|---|---|
| MiMo 2.6F 3-bit | 115.6, 30.01 (b12317) | 1417.0 / 1456.6 | 1365.8 / 1367.2 | 42.93 / 43.92 | 45.68 / 45.13 |
| Qwen 3.8 Next IQ4_XS | 511.1, 51.42 (b12669) | 2268.6 / 2669.6 | 2296.2 / 2575.6 | 77.31 / 78.23 | 77.52 / 78.58 |
| GLM 5.3 3.5-bit | - | 968.5 / 1043.0 | 968.3 / 1009.3 | 31.66 / 30.98 | 30.05 / 31.04 |
| GLM 5.3 3-bit | - | 915.7 / 989.4 | 912.1 / 954.8 | 32.85 / 33.04 | 33.58 / 34.23 |
| Qwen 3.6 35B-A3B (fits VRAM) | - | 8302.0 / 8947.5 | 8209.2 / 8802.7 | 160.25 / 160.90 | 161.01 / 161.11 |
| DeepSeek 4.1 IQ2_XXS | - | 750.8 / 779.6 | 694.9 / 710.4 | 37.34 / 37.37 | 37.05 / 37.50 |

## Against Strata and Adamant

Where both run the same model, this fork is comparable with [Strata](https://github.com/Niko1221/Strata) and [Adamant](https://github.com/nixcat-dev/adamant) (a [Basalt](https://github.com/jesdga95/basalt) fork). The models with the biggest gains above are in none of their model lists (their READMEs, 2026-11-26): Strata 0.2.22 lists Qwen3.8-Flash-Next, Hy4-Flash, Qwen3.6-35B-A3B and Inkling 2 Mini; Basalt lists Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.8-27B, Hy4-Flash and Inkling 2 Mini, on Blackwell GPUs only; Adamant, its port to Ada and Ampere cards, lists Qwen3.8-Flash-Next. Of these engines, Kimi K3.1, GLM 5.3 and MiMo 2.6 run only here.

A same-session table against both engines on machine B (Strata 0.2.21 with engine 0.1.60 and 0.2.20, Adamant at its defaults with MTP and without) follows in this README after the release.

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

- Measured on machines A, B and C above and one notebook (ASUS GA403UV: RTX 4060 laptop 8 GB, Radeon 780M, 32 GB RAM); other hardware is untested.
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
