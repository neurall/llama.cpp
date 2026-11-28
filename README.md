# Rocket LLama

A fork of llama.cpp for MoE models larger than your VRAM. The free VRAM of every GPU becomes a cache of the experts that are in use, the CPU computes the rest in parallel, and long prompts stream the missing experts to the GPUs in large passes. On the same machine with the same request: up to **45x the prompt speed** and **7.8x the decode speed** of stock llama.cpp.

```sh
llama-server -m MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf
```

No `-c`, `-ub`, `-ngl` or `--moe` flags: the context, the micro-batch, the expert placement and the threads are sized per request.

**Download:** the [latest release](https://github.com/neurall/llama.cpp/releases/latest) has one package per OS with CPU, CUDA, Vulkan and HIP backends (Linux `tar.gz`, Windows `zip`). NVIDIA cards also need the `cudart-` archive if CUDA 13 is not installed.

## Best gains against stock llama.cpp

One row per model and machine: our best valid run against the official stock binary, on the same machine and with the same request, each build on its own defaults (`-np 1` only). **i/s** = prompt tokens per second, **o/s** = generated tokens per second. Both answers are linked (ours, then stock). Sorted by the bigger of the two gains.

| model | machine | test | prompt i/s: stock → ours | decode o/s: stock → ours | ours | stock | answers |
|---|---|---|---|---|---|---|---|
| Kimi K3.1 Pro IQ1_S | TR | 13k/1k | 1.5 → 67.9 (**45x**) | 0.31 → 2.41 (**7.8x**) | b15843 | b12426 | [31918](fork/outputs/31918.txt), [29437s](fork/outputs/29437s.txt) |
| MiMo 2.6 Flash 3-bit | R7 | 12k/1k | 49.4 → 1241.1 (**25.1x**) | 13.99 → 25.71 (1.8x) | b14965 | b11829 | [24369](fork/outputs/24369.txt), [18221s](fork/outputs/18221s.txt) |
| DeepSeek 4.1 Flash IQ2_XXS | R7 | short/1k | - | 1.47 → 32.95 (**22.4x**) | b14939 | b11887 | [24094](fork/outputs/24094.txt), [19818s](fork/outputs/19818s.txt) |
| MiMo 2.6 Flash 3-bit | 5090 | 12k/1k | 141.3 → 3016.3 (**21.3x**) | 6.83 → 36.13 (5.3x) | b13348 | b11799 | [14701](fork/outputs/14701.txt), [14700s](fork/outputs/14700s.txt) |
| GLM 5.3 Flash 3-bit | R7 | 16k/1k† | 41.7 → 869.6 (**20.9x**) | 12.53 → 19.56 (1.6x†) | b14939 | b11887 | [24065](fork/outputs/24065.txt), [19572s](fork/outputs/19572s.txt) |
| GLM 5.3 Flash 3.5-bit | R7 | 12k/1k | 34.3 → 614.9 (**17.9x**) | 10.72 → 15.18 (1.4x) | b14954 | b11829 | [24296](fork/outputs/24296.txt), [18275s](fork/outputs/18275s.txt) |
| MiMo 2.6 Flash 3-bit | TR | 12k/1k | 115.6 → 1699.9 (**14.7x**) | 30.01 → 33.58 (1.1x) | b15463 | b12317 | [28397](fork/outputs/28397.txt), [26475s](fork/outputs/26475s.txt) |
| Qwen3.8 Flash Next IQ4_XS | R7 | 15k/1k | 176.8 → 1934.1 (**10.9x**) | 29.14 → 52.31 (1.8x) | b13845, b14705 | b12131 | [23477](fork/outputs/23477.txt), [23510](fork/outputs/23510.txt), [23452s](fork/outputs/23452s.txt) |
| DeepSeek 4.1 Flash IQ2_XXS | TR | 2k/1k | 81.0 → 371.5 (4.6x) | 5.02 → 38.97 (**7.8x**) | b15065 | b12317 | [26851](fork/outputs/26851.txt), [26797s](fork/outputs/26797s.txt) |
| Qwen3.8 Flash Next IQ4_XS | TR | 14k/1k | 443.2 → 2714.9 (**6.1x**) | 50.42 → 55.71 (1.1x) | b15862 | b12499 | [30944](fork/outputs/30944.txt), [30903s](fork/outputs/30903s.txt) |
| Qwen3.8 Flash Next IQ3_S | R7 | 3k/1k | 387.2 → 1777.2 (**4.6x**) | 44.59 → 56.18 (1.3x) | b14939 | b11829 | [24122](fork/outputs/24122.txt), [18270s](fork/outputs/18270s.txt) |
| ERNIE 5.5 Flash IQ3_XXS/IQ4_XS | TR | 12k/1k | 116.3 → 453.7 (**3.9x**) | 18.05 → 20.18 (1.1x) | b16934 | stock + PR‡ | [36667](fork/outputs/36667.txt), [36687s](fork/outputs/36687s.txt) |
| Qwen3.8 Flash Next IQ3_S | TR | 14k/1k | 754.5 → 2631.2 (**3.5x**) | 58.22 → 57.05 (1.0x) | b15781 | b12499 | [30408](fork/outputs/30408.txt), [30905s](fork/outputs/30905s.txt) |
| Hy4 Flash IQ2_AH | TR | 4k edit | 329.4 → 1055.9 (**3.2x**) | 28.2 → 39.37 (1.4x) | b16923 | b12804 | [36465](fork/outputs/36465.txt), [36475s](fork/outputs/36475s.txt) |

**Tests:** 12k/1k = 12.4k-13.8k-token code-review prompt, 1024 tokens out (13k-16k: the same review, longer); 2k/1k and 3k/1k = a 2k-3k-token source file to review or edit; 4k edit = a 4k-token source file plus an edit instruction; short/1k = one-line prompt, 1024 tokens out (too short to time the prompt).
**Machines:** **TR** = Threadripper PRO 3975WX (32 cores), 256 GB DDR4-3200 in 8 channels, 2x RTX 3090 (both PCIe 4.0 x16); **R7** = Ryzen 7 3700X, 128 GB DDR4-3200, 2x RTX 3090 (x16 + x4); **5090** = a rented server with one RTX 5090 (PCIe 5.0 x16).
† stock was asked for 64 tokens there, so that decode gain is for information. ‡ stock has no ERNIE 5.5 support yet: stock b12220 plus upstream PR #31925 and our model loader. Models that fit in VRAM, and dense models, run at stock speed and are left out. Older tables, the per-feature gains and perplexity: [fork/docs/perf-history.md](fork/docs/perf-history.md), [fork/docs/features.md](fork/docs/features.md).

## Where it wins

- **Wins:** MoE models that do not fit in VRAM. The less of the model fits, the bigger the gain, down to models larger than RAM (Kimi K3.1, DeepSeek), which run partly from the SSD.
- **Same as stock:** models that fit in VRAM, and dense models.

## Quick start

```sh
llama-server -m MODEL.gguf                 # OpenAI-compatible server, everything automatic
llama-cli    -m MODEL.gguf -p "hello"
```

The fork's defaults apply only when a MoE model is larger than the free VRAM of all GPUs; other models, and every setting you pass yourself, behave as in upstream. The chosen values are printed at startup (`-lv 5` prints all of them).

| switch | effect |
|---|---|
| `--fork off` | upstream behaviour for everything |
| `-at off` | no automatic tuning (the fork's other defaults stay) |
| `--moe fixed=1` | repeatable output for benchmarks |
| `--moe dense-split=0` / `=1` | force one card / the dense split over the cards |
| `--moe-expert-cache 0` | expert cache off |
| `GET /status` | server phase (loading, prefill, generation) with live ms per token |

Every `--moe` key and its default: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md). How a row is reproduced: [fork/docs/perf-history.md](fork/docs/perf-history.md#reproduce).

## Known limits

- **Measured hardware:** two machines with 2x RTX 3090, a rented RTX 5090, an RTX 4060 laptop (8 GB) and a Radeon 780M (Vulkan, HIP). 3+ GPUs and mixed vendors are untested.
- **The first runs of a model are slower:** the hot-expert map is learned from use and kept in `moe-state.ini`; the release ships pre-learned maps in `hotmaps/`.
- **Models larger than free RAM load lazily:** the server answers while the experts are still being pinned, so the first big prompt after a start can be slower.
- **Output can differ between runs** even at temperature 0: a cached expert runs on the GPU, a missed one on the CPU, and they round slightly differently (`--moe fixed=1` fixes the placement). Perplexity matches stock within noise ([table](fork/docs/perf-history.md#quality)).

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
