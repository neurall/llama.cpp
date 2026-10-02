# llama.cpp fork: 1.7x to 2.4x faster decode on MoE models bigger than your VRAM

The newest mixture-of-experts models (GLM-5.3-Flash, MiMo-V2.6-Flash, Qwen3.8-Flash-Next, Qwen3.6) are far bigger than a gaming GPU.
This fork keeps their experts in RAM and turns the free VRAM into a live cache of the experts the model is using; the GPUs compute
the cached ones and the CPU the rest, at the same time. No special switches needed.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
llama-cli    -m model.gguf -p "hello"
```

## What you get

Decode tokens/s, single stream, temperature 0, model in RAM. "Upstream" is stock llama.cpp.
Machine A: 2x RTX 3090 (PCIe 4.0 x16 + chipset x4), Ryzen 7 3700X, 125 GB DDR4-3200.
Machine B, a laptop: RTX 4060 8 GB, Ryzen 9 8945HS, 32 GB LPDDR5X-6400.
Machine C: no GPU, Ryzen 5 3600, 64 GB DDR4-3200.
Machine D, rented: 4x RTX 3090 (PCIe 4.0 x16 each), EPYC 7B12 (64 cores), 256 GB DDR4 on 4 of 8 memory channels (74 GB/s read measured).
GLM 3.5-bit, IQ3_S and IQ1_M are the first run of build b11707 against fresh upstream def4d406a (no discarded run before it); the other rows are hot runs of earlier builds.

| model (size) | machine | test | upstream | this fork | |
|---|---|---|---|---|---|
| **GLM-5.3-Flash** 3.0-bit (106 GB) | A | short chat, decode | 11.9 | **22.4** | **1.9x** |
| **GLM-5.3-Flash** 3.0-bit (117.5 GB file) | D | short chat, decode, first run | 24.7 | 25.2 | 1.0x |
| | | same, second run (saved state) | 24.7 | **31.5** | **1.3x** |
| | | same, second run, `-t 16` | 24.7 | **33.8** | **1.4x** |
| | | same, 3 of the 4 GPUs, second run (best ratio) | 20.1 | **32.6** | **1.6x** |
| **MiMo-V2.6-Flash** IQ3_XXS (132 GB, bigger than RAM) | A | short chat, decode | 4.6 | **10.9** | **2.4x** |
| | | 12k-token prompt, decode | 4.2 | **9.1** | **2.2x** |
| | | 12k-token prompt, processing | 156 | 112 | 0.7x |
| **Qwen3.8-Flash-Next** UD-IQ4_XS (88 GB) * | A | short chat, decode | 27.7 | **46.5** | **1.7x** |
| | | 12k-token prompt, decode / processing | 25.3 / 500 | **42.3 / 538** | 1.7x / 1.1x |
| **GLM-5.3-Flash** 3.5-bit (137 GB, bigger than RAM) | A | short tetris prompt, 100 tokens, decode (text-dependent) | 6.9 | **15.1** | **2.2x** |
| **Qwen3.8-Flash-Next** GSQ IQ3_S (83 GB) | A | short tetris prompt, 100 tokens, decode | 43.9 | **57.1** | **1.3x** |
| **Qwen3.8-Flash-Next** GSQ IQ1_M (55 GB, barely over 48 GB VRAM) | A | same | 69.1 | 67.5 (picks stock) | 1.0x |
| | B | same | 11.4 | **13.5** | **1.2x** |
| | B | same, prompt processing | 15.5 | 5.9 | 0.4x |
| **Qwen3.6-35B-A3B** Q2_0 (11 GB, on an 8 GB GPU) | B | short tetris prompt, 100 tokens, decode | 29.2 | **59.3** | **2.0x** |
| | C | same, CPU only (AVX Q2_0 kernels) | 6.3 | **11.3** | **1.8x** |
| Qwen3.8-27B IQ4_NL, dense (fits VRAM) | A | same | 45.1 | 45.0 | 1.0x |
| Qwen3.8-27B IQ3_S, dense, CPU only | C | same | 1.7 | 1.6 | 1.0x |
| Qwen3.8-27B Q5_K_M with MTP (`--spec-type draft-mtp`), fits VRAM | A | same | 78.3 | 77.0 | 1.0x (38.6 without MTP) |
| | | 2.2k-token prompt, decode / processing | 31.3 / 599 | **51.1 / 792** | 1.6x / 1.3x |
| Models that fit in VRAM | any | anything | same | same | 1.0x (cache off) |

D: upstream is the downloaded release b11323 (a source build of def4d406a gave 24.8 and 20.1); the fork is b11707 with the placement change in this branch (any model that does not fit takes the cache);
the model is 85% in VRAM on 4 GPUs, so the first run only matches stock placement. D numbers are from one session on a rented box (raw logs not kept, not in run-history.csv).

\* from the previous release. GLM and MiMo were measured on release-candidate builds (MiMo also on b11509) before the last placement and thread commits, Qwen3.6 on the release binary. Every run behind these numbers (commit, build, machine, settings) is in
[`tools/bench/run-history.csv`](tools/bench/run-history.csv).

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
- **A faster CPU lowers the gain,** because the slow-memory term shrinks (machine D reads 74 GB/s against 44 GB/s on machine A).
- **A model that fits in VRAM gains nothing** (the cache is off, 1.0x).

Four points from one session with different output texts: a good explanation, not a proof. The 2 and 1 GPU stock figures are model predictions, not measurements.

## Too many knobs? Smart autotune picks them

Running a model bigger than your VRAM well means choosing placement, cache size, upload schedule, batch size, thread counts, the VRAM margin. Instead of a page of flags, the
fork measures your machine and tunes most of those while you use it. The defaults are chosen on your machine, not hard-coded:

- **Cache or stock.** If the model fits in VRAM it is placed exactly like stock llama.cpp. A model that does not fit takes
  the cache right away. When what you run is mostly long prompts (processing them would cost more than the faster generation
  gains), the first two runs compare both and keep the faster one.
- **Self-tuning on real token times.** The cache policy and the upload schedule are adjusted while you use it. The decode and prompt thread counts start from
  a formula (cores minus one per GPU) and the tuner moves them a few threads at a time; on a many-core host `-t 16` can be a better start (on a 64-core machine the
  difference was under 2%). A setting that does not help is dropped, a setting you fix yourself is never touched.
- **It remembers.** What it learned per model (hot experts, tuned settings, cache-or-stock) is kept in one file,
  `~/.cache/llama.cpp/moe-state.ini`, so the next start, even a one-shot short prompt, begins from it. Delete the file to start over.
- **It tells you what it does.** `llama-server` logs, and `llama-cli -lv 3` prints after each reply, whether the cache is on,
  the hit rate, the tuned settings, threads and batch sizes.

## MTP speculative decoding

Qwen3.8-Flash-Next MTP from PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
([@danielhanchen](https://github.com/danielhanchen)), GLM-5.3-Flash MTP from PR
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos), both in this build. Load a model's MTP draft head
with `-md`:

```sh
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
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
| `--moe KEY=VAL,...` | `cache`, `prefetch-slots`, `inserts`, `window`, `predict`, `train`, or any tuning knob (`MARGIN`, `GATE`, `WAIT`, `BIG`, `SWAP_FRAC`, ...), for example `--moe gate=3,margin=0` |
| `--load-mode pin\|mmap` | pinned weights (the server default when the model fits in RAM, faster prompts) or mmap |
| `LLAMA_MOE_AUTO_MODE=stock\|cache` | force the placement; `LLAMA_MOE_STATE=0` ignores and never writes the state file |

## Turn it off

| you pass | effect |
|---|---|
| `--moe cache=0` | the whole fork off: no expert cache, nothing tuned or measured, plain stock behaviour (same as `--moe-expert-cache 0`) |
| `-at off` | only the self-tuning off (`--autotune off`, same as `--moe autotune=0`): the cache keeps working with fixed defaults, placement uses a static rule, nothing is measured or saved |

Environment forms: `LLAMA_AUTOTUNE=0`, `LLAMA_ARG_AUTOTUNE=off`, `LLAMA_ARG_MOE=cache=0`.

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
