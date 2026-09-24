# llama.cpp: fork with multi gpu acceleration even for models bigger than total gpu mem

For MoE models much larger than VRAM: every expert stays in system RAM, and all
VRAM left after the KV cache becomes a live cache of the experts actually being
used. The GPUs compute cached experts while the CPU computes the rest, in parallel.

**GLM-5.3-Flash 3.0-bit (117 GB) on 2x RTX 3090 (48 GB) + 125 GB RAM, single stream:**

| | decode t/s | wikitext-2 PPL |
|---|---|---|
| stock llama.cpp (autofit) | 12.3 | 3.5534 |
| this fork, original GGUF | ~25 | 3.5534 |
| this fork, [Q4_K attention GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) (faster) | **27.72** | 3.5871 (+0.95%) |

Models:
- Original GGUF (tested): [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF),
  the 3.0-bit file. It works as-is with this fork, no conversion needed, and it's the
  quality reference (unchanged perplexity). Most of the speedup comes from the fork,
  not the requantization: ~25 t/s with this file vs ~28 t/s with the Q4_K attention
  variant below.
- Q4_K attention variant (same experts, non-expert Q8_0 weights requantized to Q4_K):
  [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF).

Decode: prompt "generate smallest html tetris game.", 1024 context, temperature 0.
Perplexity: 40 x 512-token chunks.

Run (with the faster Q4_K attention GGUF from [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)):

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -np 1 -c 1024 -t 6 --cpu-moe -nr --moe-expert-cache -1
```

- `--cpu-moe` keeps all experts in RAM; `--moe-expert-cache -1` sizes the cache
  per GPU from the VRAM free after KV and compute buffers. Set `-c` explicitly:
  without it autofit grows the context and takes the VRAM the cache needs.
- `-t 6` suits an 8-core CPU (leave cores to drive the GPUs).
- Cache hit rate is ~85% after warm-up. `LLAMA_MOE_CACHE_STATS=1` logs it.
- Tuning: `LLAMA_MOE_CACHE_POLICY` (`add` default, `halve`, `window`),
  `LLAMA_MOE_CACHE_MARGIN_MB` (VRAM left free, default 1024),
  `LLAMA_MOE_CACHE_SWAP_FRAC` (share of token time for uploads, default 0.25).
- `GGML_SCHED_PROF=1` prints where each token's host time goes.

**More GPUs (estimate, only 2 tested).** Nothing assumes two GPUs: each GPU gets
its own cache, sized from its free VRAM. Each extra GPU adds cache room, so more
of every token's experts are hits and less work falls to the CPU. For this model,
per token today: ~20 ms GPU work on non-expert layers, ~10 ms CPU on missed experts.

| GPUs (24 GB each) | cache room | slots/layer (of 288) | hit rate | CPU miss time | decode t/s |
|---|---|---|---|---|---|
| 2 (measured) | ~34 GB | ~100 | 85-88% | ~10 ms | 26-28 |
| 3 | ~58 GB | ~170 | ~95% | ~3-4 ms | ~33-35 |
| 4 | ~82 GB | ~240 | ~99% | ~1 ms | ~38-42 |
| 5+ | whole model | 288 | 100% | 0 | ~40-45 (plateau) |

The plateau is the ~20 ms GPU part: with the default layer split each layer runs on
one GPU at a time, so extra GPUs add cache room, not speed on that part. System RAM
must still hold all experts. Cards in x4 PCIe slots upload experts slower, so the
cache warms up slower. Reports from 3+ GPU setups are welcome.

What's in it: the GPU expert cache from PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
(csantiago78), extended with VRAM-filling auto-sizing, prefill warm start,
usage-driven eviction that only swaps when the upload pays back, CPU/GPU overlap
per layer, scheduler barrier fixes and fused gate kernels. GLM-5.3-Flash support
comes from PRs [#27773](https://github.com/ggml-org/llama.cpp/pull/27773) and
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos); stock
llama.cpp can't load GLM-5.3-Flash yet.

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
