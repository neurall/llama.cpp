# llama.cpp: fork with multi gpu acceleration even for models bigger than total gpu mem

> **Big thanks to [@csantiago78](https://github.com/csantiago78)**: the expert cache
> here builds on their implementation in llama.cpp PR
> [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) ("GPU-resident LRU cache
> for host-offloaded MoE expert weights"), the first working hot-expert cache in
> llama.cpp. This fork takes it further: filling all free VRAM, adaptive eviction,
> CPU/GPU overlap, small-batch support and fused kernels.

For MoE models larger than VRAM: every expert stays in system RAM, and all VRAM left
after the KV cache becomes a live cache of the experts actually being used. The GPUs
compute cached experts while the CPU computes the rest, in parallel.

llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf

**Results, 2x RTX 3090 (one AM4 CPU PCIe 4.0 x16, one X570 chipset x4 slot) + Ryzen 7 3700X + 125 GB DDR4,
CPU frequency governor `performance`, single stream, temp 0.** Short = 1500-token chat reply to "write smallest html tetris game". Long =
12k-token code prompt (llama.cpp sources): prompt processing, then decode. The prompts and a script to
reproduce these tests are in [`tools/moe-bench/`](tools/moe-bench/).

| model | size | short: decode t/s | short: decode with MTP | long: prefill t/s † | long: decode t/s |
|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit [Q4_K attn](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) | 106 GB | 13.8 -> **21.5 (1.56x)*** | 17.6, slower ‡ | 217 -> 181* | 12.3 -> **15.7 (1.28x)*** |
| MiMo-V2.6-Flash-RL IQ3_XXS | 132 GB | 4.0 -> **10.1 (2.54x)**** | no MTP head | 136 -> 133** | 4.2 -> **8.3 (1.98x)**** |
| Qwen3.8-Flash-Next UD-IQ4_XS | 88 GB | 27.7 -> **46.4 (1.68x)*** | **57.0 (2.06x vs stock)*** | 500 -> 390* | 25.3 -> **38.7 (1.53x)*** |
| Qwen3.8-27B IQ4_NL (dense, fits VRAM) | 16 GB | 44.7 -> 44.8 | | 1726 -> 1811 | |
| OLMoE-1B-7B Q4_K_M (fits VRAM) | 4 GB | 504 -> 504 | | | |

\* Model already in RAM (OS page cache), as on a server after its first request. The
first run after switching to another large model is slower, once, while the file is
read from disk. \*\* MiMo (132 GB) can't fully stay cached in 125 GB RAM, so it always
reads part of the model from disk.
† Prompt processing streams the experts over PCIe to one GPU, so the slot limits it (here a
PCIe 4.0 x16 CPU slot; the second card sits in an X570 chipset x4 slot). A board with more
x16 slots, or another GPU, should raise it; splitting prompt processing across both GPUs'
links is the next milestone.
‡ MTP on GLM-5.3-Flash is slower on this box (17.6 vs 20.2 t/s without, same session; the model is 2.3x
the VRAM, so verifying drafts adds CPU work and the draft takes cache VRAM). The fork
doesn't load the draft here by default. Another GPU (more VRAM) should make it pay off,
as it does for Qwen (1.9x VRAM).

stock llama.cpp -> this fork. The fork numbers match a plain `llama-server -m model` within ~4%
(auto mode, see Run). Stock can't load GLM-5.3-Flash, so its stock column is
this fork without the cache. Models that fit in VRAM don't use the cache and run the
same (identical output). With the default `schedutil`/`powersave` governor, decode can
be lower, mostly where the CPU computes missed experts. Prompt processing is slower than stock: stock keeps whole
layers in VRAM and never uploads them, this fork trades that VRAM for the decode gain.
Decode speed depends on how often generated tokens reuse cached experts: ~75% of
experts are hits on GLM chat, ~95% on Qwen.

Models: GLM original 3.0-bit GGUF [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
works as-is (reference quality); the Q4_K attention variant above (same experts,
non-expert Q8_0 weights at Q4_K, +0.95% perplexity) decodes ~10% faster.

**Same VRAM, different use.** Each GLM token uses 8 of 288 experts per layer. Stock
fills VRAM with whole layers (~14 of 42), mostly experts the current token doesn't
touch, so the CPU does most expert work and the GPUs wait. This fork fills the same
VRAM with the ~100 most-used experts of every layer, so most expert work runs on the
GPUs, at the same time as the CPU handles the misses.

## Run

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
llama-cli    -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf -p "hello"
```

**Automatic defaults that differ from stock llama.cpp.** They apply only when a MoE
model's weights are larger than the free VRAM of all GPUs. Other models, and every
setting you pass yourself, behave as in stock llama.cpp. The chosen values are
printed at startup (`-lv 5` prints all of them).

| setting | stock default | fork default | why |
|---|---|---|---|
| expert placement | autofit: whole layers on GPU, rest on CPU | all experts in RAM (`--cpu-moe`) | the VRAM is used for the expert cache instead of fixed layers |
| expert cache | off | fills each GPU's free VRAM (`--moe-expert-cache -1`) | holds the experts tokens actually use, so most expert work runs on the GPUs |
| CPU weight repacking | on | off (`-nr`) | repacked experts can't be copied to the GPU cache |
| `-ub` (tokens per prompt step) | 512 | 2048 if the largest GPU has 20+ GiB free, 1024 at 10+ GiB, else 512 | every expert upload serves more prompt tokens: ~2x faster long prompts; costs ~0.7 GiB cache VRAM (~1% decode) on GLM-5.3-Flash |
| `-c` (context) | model maximum, fitted to VRAM | 32768 | a bigger KV cache would take VRAM from the expert cache; `-c 65536` if you need more |
| `-t` (threads) | all physical cores | cores minus one per GPU (6 of 8 here) | a free core per GPU keeps kernel launches and cache uploads fast; measured faster |

Settings you pass always win, e.g. `--moe-expert-cache 0` turns the cache off. With your
own placement (`-ngl`, `-ot`, `--cpu-moe`) the cache stays off unless you also pass
`--moe-expert-cache -1`.
- **Prompt processing runs on one GPU, so its PCIe bandwidth sets the speed.** The
  fork measures host->GPU upload speed per GPU at startup and sends prompt processing
  to the fastest (here x16 13.2 GB/s vs chipset x4 6.1 GB/s); layers stay in bus order.
- Output can differ between runs even at temperature 0: a cached expert runs on the
  GPU, a missed one on the CPU, and they round slightly differently. For benchmarks,
  `LLAMA_MOE_CACHE_DETERMINISTIC=1` makes a build repeat its output.
- Optional tuning (defaults are measured best): `LLAMA_MOE_CACHE_MARGIN_MB` (VRAM left
  free, 384), `LLAMA_MOE_CACHE_STATS=1` (hit rate), `LLAMA_MOE_CACHE_WAIT=0`,
  `LLAMA_MOE_CACHE_BUDGET` / `_MARGIN` / `_SWAP_FRAC` / `_POLICY`.

**More GPUs**: each GPU gets its own cache from its free VRAM, so more GPUs mean more
cached experts, fewer CPU misses and faster decode, up to the point where every
expert fits. Reports from 3+ GPU setups are welcome.

### MTP (multi-token prediction)

Added in release b11327: Qwen3.8-Flash-Next MTP from PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
([@danielhanchen](https://github.com/danielhanchen)), GLM-5.3-Flash MTP from PR
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos).

Load a model's MTP draft head with `-md` and the fork picks the draft depth itself:

```sh
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -md GLM-5.3-Flash-MTP-Q4_K.gguf --spec-type draft-mtp
```

- MTP heads: Qwen3.8-Flash-Next from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  (`MTP/`, the `shared` files reuse the main model's embeddings); GLM-5.3-Flash from
  [neuralll/GLM-5.3-Flash-MTP-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF)
  (4.3 GiB, works with any `glm5-next` GLM-5.3-Flash GGUF).
- **Why depth is automatic:** every drafted token has to be verified, and each token picks
  its own experts. Experts that miss the VRAM cache run on the CPU at the same cost per
  drafted token as per generated one, so the best depth depends on how much of the model the
  GPUs hold, not only on how often drafts are accepted. The fork starts from a guess (model
  size vs free VRAM: up to 1x → 3, up to 2x → 2, more → no drafting), then measures real
  generation speed and moves the depth up or down in doubles, then by 1, re-checking every
  4096 tokens. The max depth is 2 when the model is bigger than VRAM (the rollback buffers of
  hybrid models are sized by it and cost cache VRAM), 5 when it fits.
- Measured here: Qwen3.8-Flash-Next (1.9x VRAM, 95% cache hits) 47.1 -> **57.0 t/s (1.21x)**
  at depth 2. GLM-5.3-Flash (2.3x VRAM, ~70% hits) is slower with MTP (17.6 vs 20.2 t/s), so
  the fork doesn't load the draft there (a warning says so; `--spec-draft-n-max N` forces it).
  More VRAM should make GLM MTP worth it: with 3x 24 GB, GLM is ~1.5x VRAM (less than
  Qwen's 1.9x here), cache hits should reach 90%+ and verifying drafts would cost little CPU
  work (estimate ~1.2-1.3x from MTP); with 4 cards nearly all experts fit. A third card on
  a chipset x4 slot slows cache uploads and prompt processing, not decode.
- `LLAMA_SPEC_DEPTH=N` pins the depth for benchmarks.

## What's in it

- The GPU expert cache from PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
  (csantiago78), extended with: VRAM-filling auto-size; usage tracking with decay;
  swaps only when the upload pays back; an adaptive swap budget from measured upload
  and token times; the cache published every step (+16% decode on long prompts);
  CPU/GPU overlap within a layer; small batches up to 31 tokens (short prompts,
  speculative verify) through the cache.
- Prefill: experts the prompt selects warm the cache before the first token
  ([@sdroege](https://github.com/sdroege) explored the same idea in the PR thread);
  in big prefill batches, cached experts are copied GPU-to-GPU instead of over PCIe.
- Automatic defaults for MoE models bigger than free VRAM (see Run): experts in RAM,
  cache, no repack, `-ub` by VRAM, 32k context, one core per GPU left free.
- Startup upload-bandwidth probe that sends prompt processing to the fastest-link GPU.
- Lookahead expert prefetch from PR [#28414](https://github.com/ggml-org/llama.cpp/pull/28414)
  ([@leshchukandrej](https://github.com/leshchukandrej)), `--prefetch-experts-slots N`,
  off by default (slower with the cache on this box; may help without it).
- Scheduler fixes (no host barrier between GPU splits that don't read host memory,
  splits inserted exactly where another GPU's result is needed), fused CUDA gate
  kernels, repacked CPU experts stay cacheable, `GGML_SCHED_PROF=1` profiling.
- MTP: Qwen3.8-Flash-Next MTP from PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243)
  ([@danielhanchen](https://github.com/danielhanchen)); GLM-5.3-Flash MTP from PR
  [#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos), plus loading GLM's
  MTP head as a separate small file; the expert cache stays with the main model (the draft
  never builds or frees it, and the cache sizes itself after the draft is loaded); automatic
  draft depth from measured speed.
- GLM-5.3-Flash support from PRs [#27773](https://github.com/ggml-org/llama.cpp/pull/27773)
  and [#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos).
- Every change is benchmarked against the previous release binary on fixed inputs
  before it ships.

## Known limits and next steps

- Prompt processing is up to ~22% slower than stock on these models. Next milestone: use
  both GPUs' PCIe links for it (split each layer's experts by measured bandwidth).
- GPU order, ongoing research. On this box decode is ~7% faster (tetris 28.6 vs
  26.7 t/s, cache hits 90% vs 87%, same cache size) when the layers stay in bus order
  (x4 GPU first) than when the x16 GPU takes the first layers, while prompt processing
  wants the x16 GPU. So since this release prompt processing goes to the fastest-link
  GPU and layers keep bus order, which gets both. Why the layer order matters is not
  known yet: the two cards differ (x16: Gainward 3-slot 370 W with partly blocked
  airflow, x4: Dell OEM 2-slot 350 W; a budget build, these were the cards available
  at a good price), so clocks or memory hotspot throttling may count
  as much as the PCIe link. Next: log clocks and throttle reasons per GPU during
  decode, and in auto mode choose the order per request (prompt length, measured
  speeds).
- MTP on models much bigger than VRAM (GLM-5.3-Flash here) is slower: verifying drafts
  multiplies the CPU's expert work. Next: let the draft use the expert cache, and verify
  drafts with the experts the main token already selected where possible.

**About the author of this fork**: I'm actively looking for an AI engineering/research
role and open to relocating out of Eastern Europe. If this work is useful to you or
your team, reach out: [linkedin.com/in/neuralll](https://www.linkedin.com/in/neuralll/)

---

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
