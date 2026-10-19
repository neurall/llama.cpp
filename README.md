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

## Best gainers

Newest run of each cell against the newest official stock binary, sorted by gain. Each build on its own defaults (`-np 1` only). Medians of the valid runs, looped answers excluded. in_heavy = long prompt, prompt tokens/s; in_out_heavy and out_heavy = decode tokens/s. 2x RTX 3090 (x16 and x4 link), both cards limited to 200 W, no huge page pool; stock and ours back to back.

| model | test | stock | ours | gain | stock build | our build |
|---|---|---|---|---|---|---|
| MiMo-V2.6-Flash 3-bit | in_heavy | 49.2 prompt t/s | 986.9 prompt t/s | 20.05x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| GLM-5.3-Flash 3.0-bit | in_heavy | 42.4 prompt t/s | 627.2 prompt t/s | 14.78x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ4_XS | in_heavy | 208.7 prompt t/s | 1392.9 prompt t/s | 6.67x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| GLM-5.3-Flash 3.5-bit | in_heavy | 33.4 prompt t/s | 151.0 prompt t/s | 4.52x | b11734 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ3_S | in_heavy | 373.2 prompt t/s | 1477.3 prompt t/s | 3.96x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| MiMo-V2.6-Flash 3-bit | out_heavy | 14.1 decode t/s | 32.8 decode t/s | 2.32x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| GLM-5.3-Flash 3.5-bit | out_heavy | 7.6 decode t/s | 15.4 decode t/s | 2.03x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| MiMo-V2.6-Flash 3-bit | in_out_heavy | 14.2 decode t/s | 26.1 decode t/s | 1.84x | b11729 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ4_XS | out_heavy | 30.8 decode t/s | 55.1 decode t/s | 1.79x | b11729 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ4_XS | in_out_heavy | 32.0 decode t/s | 51.4 decode t/s | 1.60x | b11729 (2026-10-19) | b13586 (2026-10-19) |
| GLM-5.3-Flash 3.0-bit | in_out_heavy | 12.6 decode t/s | 20.1 decode t/s | 1.59x | b11729 (2026-10-19) | b13586 (2026-10-19) |
| GLM-5.3-Flash 3.0-bit | out_heavy | 12.7 decode t/s | 20.1 decode t/s | 1.59x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| GLM-5.3-Flash 3.5-bit | in_out_heavy | 10.7 decode t/s | 13.7 decode t/s | 1.29x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ3_S | in_out_heavy | 44.9 decode t/s | 55.6 decode t/s | 1.24x | b11729 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ3_S | out_heavy | 45.7 decode t/s | 56.4 decode t/s | 1.23x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-Flash-Next IQ1_M | in_heavy | 1432.5 prompt t/s | 1476.0 prompt t/s | 1.03x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| OLMoE-1B-7B Q4_K_M | in_out_heavy | 350.8 decode t/s | 354.6 decode t/s | 1.01x | b11739 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-27B IQ4_NL | out_heavy | 45.3 decode t/s | 45.5 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.6-35B-A3B | out_heavy | 158.4 decode t/s | 158.9 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-27B MTP Q5_K_M | out_heavy | 38.7 decode t/s | 38.8 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-27B IQ4_NL | in_out_heavy | 44.8 decode t/s | 44.9 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-27B MTP Q5_K_M | in_out_heavy | 38.3 decode t/s | 38.4 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.8-Flash-Next IQ1_M | out_heavy | 69.5 decode t/s | 69.6 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.6-35B-A3B | in_out_heavy | 156.3 decode t/s | 156.0 decode t/s | 1.00x | b11729 (2026-10-19) | b13563 (2026-10-19) |
| Qwen3.6-35B-A3B | in_heavy | 5041.0 prompt t/s | 5022.3 prompt t/s | 1.00x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-27B IQ4_NL | in_heavy | 2053.0 prompt t/s | 2033.4 prompt t/s | 0.99x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-27B MTP Q5_K_M | in_heavy | 1860.9 prompt t/s | 1840.3 prompt t/s | 0.99x | b11739 (2026-10-19) | b13586 (2026-10-19) |
| Qwen3.8-Flash-Next IQ1_M | in_out_heavy | 68.7 decode t/s | 67.7 decode t/s | 0.99x | b11729 (2026-10-19) | b13563 (2026-10-19) |

Our builds here are b13563 and b13586 (earlier builds of this release line); b13600 has no runs yet. Older tables (earlier releases, laptop and CPU rows): [fork/docs/perf-history.md](fork/docs/perf-history.md).


Models: GLM original 3.0-bit GGUF [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
works as-is (reference quality); the Q4_K attention variant above (same experts,
non-expert Q8_0 weights at Q4_K, +0.95% perplexity) decodes ~10% faster.

**Same VRAM, different use.** Each GLM token uses 8 of 288 experts per layer. Stock
fills VRAM with whole layers (~14 of 42), mostly experts the current token doesn't
touch, so the CPU does most expert work and the GPUs wait. This fork fills the same
VRAM with the ~100 most-used experts of every layer, so most expert work runs on the
GPUs, at the same time as the CPU handles the misses.

## Perplexity against stock

Same text, same chunks, official stock `llama-perplexity` against ours. The number in brackets is the per-chunk difference to stock in percent of perplexity (mean ± standard error); 0.00 means identical to four decimals. *default* is what you get without settings (the planner decides per prompt size whether to use one card or both), *one card* and *two cards* force the path (`dense-split=0` / `dense-split=1`), *dense split only* is `split-experts=0` (see below). The plain Qwen3.8 27B models have no dense split, so for them *two cards* splits the experts only. About 10 chunks of 1024 tokens per model: differences under 0.2% are inside the noise.

| model | stock ppl | ours default | ours, one card | ours, dense split only | ours, two cards | chunks |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash-GSQ-RCO-3.0bit | 1.7626 | 1.7580 (-0.26 ± 0.42%) | 1.7608 (-0.10 ± 0.25%) | - | 1.7540 (-0.49 ± 0.44%) | 9 |
| MiMo-V2.6-Flash-RL-GSQ-RCO-3bit | 1.8758 | 1.8698 (-0.32 ± 0.18%) | 1.8758 (+0.00 ± 0.00%) | - | 1.8737 (-0.11 ± 0.22%) | 9 |
| OLMoE-1B-7B-0924-Q4_K_M | 2.0696 | 2.0696 (+0.00 ± 0.00%) | 2.0696 (+0.00 ± 0.00%) | - | 2.0692 (-0.02 ± 0.08%) | 12 |
| Qwen3.6-35B-A3B-GSQ-hybrid | 1.7843 | 1.7843 (+0.00 ± 0.00%) | 1.7843 (+0.00 ± 0.00%) | 1.7860 (+0.10 ± 0.05%) | 1.7837 (-0.03 ± 0.06%) | 10 |
| Qwen3.8-27B-IQ4_NL | 1.6360 | 1.6360 (+0.00 ± 0.00%) | 1.6360 (+0.00 ± 0.00%) | - | 1.6360 (+0.00 ± 0.00%) | 10 |
| Qwen3.8-27B-MTP-Q5_K_M | 1.6382 | 1.6382 (+0.00 ± 0.00%) | 1.6382 (+0.00 ± 0.00%) | - | 1.6382 (+0.00 ± 0.00%) | 10 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002 | 1.4756 | 1.4782 (+0.18 ± 0.07%) | 1.4756 (+0.00 ± 0.00%) | 1.4732 (-0.16 ± 0.15%) | 1.4770 (+0.09 ± 0.14%) | 10 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002 | 1.3927 | 1.3927 (+0.00 ± 0.00%) | 1.3927 (+0.00 ± 0.00%) | 1.3958 (+0.22 ± 0.17%) | 1.3940 (+0.09 ± 0.20%) | 10 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003 | 1.4101 | 1.4101 (+0.00 ± 0.00%) | 1.4101 (+0.00 ± 0.00%) | - | 1.4093 (-0.06 ± 0.18%) | 10 |

Measured on build b13586 (12 chunks asked, 9 to 12 used where a text ended earlier), stock is the official b11739 binary.

Reproduce one row (stock and ours, 12 chunks, the perplexity of each is stored in `runs.db` and printed by `run ppl-table`):  
```
PERF_PPL_CHUNKS=12 run bench --models qwen36 --builds stock-latest,b13586 --variants default -t pplb -n 1
```

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
| `-ub` (tokens per prompt step) | 512 | calculated per request and card: candidates up to 12288, each one's compute buffer is probed and weighed against the prompt time it saves (the buffer costs expert cache VRAM); a long prompt runs as equal passes | every expert upload serves more prompt tokens: ~2x faster long prompts at a small decode cost (about -3% per doubling on GLM-5.3-Flash) |
| `-c` (context) | model maximum, fitted to VRAM | sized automatically: the largest context whose KV and context-scaled buffers take at most 30% of the expert cache VRAM, never above the model maximum; the KV moves down f16, q8_0, q4_0, host RAM if a request needs more | a bigger KV cache takes VRAM from the expert cache; `-c N` sets it yourself |
| `-t` (threads) | all physical cores | starts at physical cores minus one per GPU (6 of 8 here), then tuned on measured decode time; prompt batches tuned separately, up to the logical cores | a free core per GPU keeps kernel launches and cache uploads fast; the tuner moves the count when another one measures faster (`-at off` keeps the start value) |

Settings you pass always win, e.g. `--moe-expert-cache 0` turns the cache off. With your
own placement (`-ngl`, `-ot`, `--cpu-moe`) the cache stays off unless you also pass
`--moe-expert-cache -1`.
- **Prompt processing is placed by upload speed.** The fork measures host->GPU upload speed per GPU at startup (here x16 13.2 GB/s vs chipset x4 6.1 GB/s). For each prompt a planner picks the fastest card alone or both cards (dense layers split between them, experts streamed to one or both); layers stay in bus order.
- Output can differ between runs even at temperature 0: a cached expert runs on the
  GPU, a missed one on the CPU, and they round slightly differently. For benchmarks,
  `--moe fixed=1` makes a build repeat its output.
- Optional tuning (defaults are measured best): every setting is a `--moe` key, there are no
  environment variables for them, e.g. `vram-reserve-mb` (VRAM left free, 384), `wait-swaps`,
  `swaps-per-step`, `swap-lead`, `policy`. All keys: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md).

**More GPUs**: each GPU gets its own cache from its free VRAM, so more GPUs mean more
cached experts, fewer CPU misses and faster decode, up to the point where every
expert fits. Reports from 3+ GPU setups are welcome.

### More

Details moved to [fork/docs/features.md](fork/docs/features.md): pinned weights, persistent hot experts, prefill preheat, global expert pool, hot layers on the slower link, MTP, what's in it, known limits and next steps. Every `--moe` key: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md). Older measurements: [fork/docs/perf-history.md](fork/docs/perf-history.md).


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
- [homeofe](https://github.com/homeofe) - AVX2 dot products for IQ3_S and IQ3_XXS on x86 CPUs (`ggml-cpu`), merged into this fork (in the release after b12844) - MIT license
