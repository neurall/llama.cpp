# llama.cpp: fork with multi gpu acceleration even for models bigger than total gpu mem

> **Big thanks to [@csantiago78](https://github.com/csantiago78)**: the expert cache
> here builds on their implementation in llama.cpp PR
> [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) ("GPU-resident LRU cache
> for host-offloaded MoE expert weights"), the first to get a working hot-expert
> cache into llama.cpp. This fork takes it further: filling all free VRAM, smarter
> eviction, CPU/GPU overlap and fused kernels.

For MoE models much larger than VRAM: every expert stays in system RAM, and all
VRAM left after the KV cache becomes a live cache of the experts actually being
used. The GPUs compute cached experts while the CPU computes the rest, in parallel.

**Results, 2x RTX 3090 (48 GB) + 125 GB RAM, single stream, temp 0:**

| model | size | stock t/s | fork t/s, short prompt + chat reply | gain | fork t/s, short prompt + raw completion | PPL (stock -> fork) |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit, original GGUF | 117 GB | 12.4 | **17.2** | 1.4x | ~25 | 3.5534 -> 3.5534 |
| GLM-5.3-Flash 3.0-bit, [Q4_K attention GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) | 106 GB | 12.4 | **18.5** | 1.5x | 27.7 | 3.5534 -> 3.5871 (+0.95%) |
| MiMo-2.6-Flash-RL IQ3_XXS | 132 GB | 4.7 | not measured | | 9.9 | unchanged (no requant) |
| Qwen3.8-Flash-Next UD-IQ4_XS | 88 GB | 29.7 | not measured | | 32.1 | unchanged (no requant) |

Decode speed depends on how often the generated tokens reuse cached experts (hit
rate), i.e. on the input and on what the model writes. Short prompt + chat reply:
"write smallest html tetris game" via /v1/chat/completions, 1500-token answer
(reasoning + code), ~66% hit rate: the most common interactive case. Short prompt +
raw completion: "generate smallest html tetris game." as a plain completion, where
the model keeps repeating itself, 79-87% hit rate: the best case (23-27.5 t/s
depending on the exact text). Long input (12k-token document): the first ~100
generated tokens run at ~14 t/s (~44% hit) while the cache adapts to the new text.
MiMo and Qwen were only measured with raw completion, so their chat-reply gain is
likely lower (stock speed doesn't depend on the text). PPL: wikitext-2, 40 x 512-token chunks
(GLM only).
MiMo needed `-fitt 8000` on both stock and fork to avoid autofit OOM-ing on this
arch/quant combo; the others loaded fine with default fit.

Qwen's gain is small because stock's autofit already placed most of its experts on
GPU by default here — little room left for the cache to improve on. The big wins
(GLM, MiMo) are on models where default placement leaves most expert work on the CPU.

Models:
- GLM original GGUF (tested): [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF),
  the 3.0-bit file. It works as-is with this fork, no conversion needed, and it's the
  quality reference (unchanged perplexity). Most of the speedup comes from the fork,
  not the requantization.
- GLM Q4_K attention variant (same experts, non-expert Q8_0 weights requantized to Q4_K):
  [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF).

**Same VRAM, different use.** Stock llama.cpp and this fork get the same 48 GB; what
differs is what it holds. Each token uses only 8 of the 288 experts in each layer.

- Stock places experts statically, whole layers at a time: ~36 GB fits all 288
  experts of ~14 of the 42 MoE layers. Most of that VRAM holds experts the current
  token doesn't touch, so only ~33% of each token's expert work runs on GPU and the
  CPU does ~67%, one after the other.
- This fork fills the same VRAM with the ~100 most-used experts of every layer.
  Usage is skewed, so those cover ~66% of what tokens actually pick in chat replies
  (up to ~87% when the output repeats itself): ~66% of expert work runs on GPU and the CPU
  does ~34%, at the same time as the GPUs.

| | expert work on GPU | expert work on CPU | decode t/s |
|---|---|---|---|
| stock (static whole layers) | ~33% | ~67% | 12.4 |
| this fork (cache of hot experts) | ~66% | ~34%, in parallel | 18.5 |

So stock can't reach this on the same hardware: without an expert cache, extra VRAM
mostly holds experts that aren't being used.

Run (with the faster Q4_K attention GGUF from [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)):

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -np 1 -c 4096 -t 6 --cpu-moe -nr --moe-expert-cache -1 -ub 2048 -b 2048
```

- `--cpu-moe` keeps all experts in RAM; `--moe-expert-cache -1` sizes the cache
  per GPU from the VRAM free after KV and compute buffers. Set `-c` explicitly:
  without it autofit grows the context and takes the VRAM the cache needs.
- `-t 6` suits an 8-core CPU (leave cores to drive the GPUs).
- `-ub 2048 -b 2048` speeds up prompt processing ~2.2x at ~4% decode cost; drop
  it if you only send short prompts.
- **Put the GPU in the fastest PCIe slot first, or prefill collapses.** Prompt
  processing uploads the experts to the **first GPU**, so that GPU's PCIe
  bandwidth sets the prefill speed, and GPUs are numbered by PCI bus order, not
  slot speed: the first one is often a card in a chipset x4 slot. Building from
  source on Linux, the fork puts the widest-link GPU first automatically. **With
  the current release binary (b11214), on Linux too, and on Windows,** list the
  fastest-link card first yourself, e.g. if GPU 1 is in the CPU x16 slot and
  GPU 0 in a chipset x4 slot: `CUDA_VISIBLE_DEVICES=1,0 llama-server ...` (Linux),
  `set CUDA_VISIBLE_DEVICES=1,0` (Windows cmd) or
  `$env:CUDA_VISIBLE_DEVICES="1,0"` (PowerShell). Check link widths with
  `nvidia-smi --query-gpu=index,pcie.link.width.max,pcie.link.width.current --format=csv`
  (read the current width under load, idle cards can downshift).

  12k-token prompt on 2x 3090 (one x4, one x16 slot), prefill t/s. Both tips also
  apply to stock llama.cpp:

  | | x4 card first, default batch | x16 card first, default batch | x16 first, `-ub 2048 -b 2048` |
  |---|---|---|---|
  | stock | 45.9 (265 s) | 84.1 (145 s) | **227.7 (53 s)** |
  | this fork | | | 188.1 (65 s) |

  At equal settings the fork's prefill is ~20% slower than stock today: stock keeps
  ~14 whole MoE layers in VRAM so they need no upload, while the fork keeps all
  experts in RAM and serves only ~26% of prefill expert copies from its cache. The
  fork's gain is decode (12.4 to 18.5 t/s); for long prompts the batch and slot
  tips matter far more than either build.
- Cache hit rate is ~66% on chat replies, 79-87% when the output repeats itself,
  ~44% right after a long unrelated input.
- **Output varies between runs even at temperature 0.** Each expert is computed
  on the GPU when it is cached and on the CPU when it is not, and which experts
  are cached at a given token depends on when the asynchronous PCIe uploads
  finish. GPU and CPU round slightly differently, so near-tied tokens can flip
  and the text diverges from there (stock llama.cpp places experts the same way
  every run, so it repeats itself). For benchmarks and regression tests set
  `LLAMA_MOE_CACHE_DETERMINISTIC=1`: each step publishes exactly the uploads of
  the previous step with a fixed swap budget (`LLAMA_MOE_CACHE_BUDGET`,
  default 8) and margin (`LLAMA_MOE_CACHE_MARGIN`, default 4), so a given
  build repeats its output. It is slightly slower and its hit rate differs from
  the normal adaptive mode, so compare builds within the same mode.
  `LLAMA_MOE_CACHE_STATS=1` logs it.
- The defaults need no environment variables. Optional tuning:
  `LLAMA_MOE_CACHE_POLICY` (`add` default, `halve`, `window`),
  `LLAMA_MOE_CACHE_MARGIN_MB` (VRAM left free, default 1024),
  `LLAMA_MOE_CACHE_SWAP_FRAC` (share of token time for uploads, default 0.25),
  `LLAMA_MOE_CACHE_WAIT=0` (don't wait for the previous step's uploads; the wait is
  on by default and keeps the cache current: +16% decode on long code prompts),
  `LLAMA_MOE_CACHE_BUDGET` / `LLAMA_MOE_CACHE_MARGIN` (fixed swaps per step and
  pay-back margin instead of the adaptive ones; adaptive was faster on varied text).
- `GGML_SCHED_PROF=1` prints where each token's host time goes.

**More GPUs (estimate, only 2 tested).** Nothing assumes two GPUs: each GPU gets
its own cache, sized from its free VRAM. Each extra GPU adds cache room, so more
of every token's experts are hits and less work falls to the CPU. For this model,
per token today on chat replies: ~20 ms GPU work on non-expert layers, ~23 ms CPU on
missed experts, ~7 ms other. Hit rates for 3+ GPUs are rough guesses.

| GPUs (24 GB each) | cache room | slots/layer (of 288) | hit rate (chat replies) | CPU miss time | decode t/s |
|---|---|---|---|---|---|
| 2 (measured) | ~34 GB | ~100 | ~66% | ~23 ms | 18.5 |
| 3 | ~58 GB | ~170 | ~80% | ~13 ms | ~25 |
| 4 | ~82 GB | ~240 | ~92% | ~5 ms | ~30 |
| 5+ | whole model | 288 | 100% | 0 | ~35-37 (plateau) |

The plateau is the ~20 ms GPU part: with the default layer split each layer runs on
one GPU at a time, so extra GPUs add cache room, not speed on that part. System RAM
must still hold all experts. Cards in x4 PCIe slots upload experts slower, so the
cache warms up slower. Reports from 3+ GPU setups are welcome.

What's in it: the GPU expert cache from PR [#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
(csantiago78), extended with VRAM-filling auto-sizing, usage-driven eviction that
only swaps when the upload pays back, CPU/GPU overlap per layer, scheduler barrier
fixes and fused gate kernels. GLM-5.3-Flash support comes from PRs
[#27773](https://github.com/ggml-org/llama.cpp/pull/27773) and
[#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos); stock
llama.cpp can't load GLM-5.3-Flash yet.

**Prefill warm start**, implemented independently for this fork: the cache observes
which experts the prompt itself selects during prefill and preloads them before the
first generated token, instead of starting cold and only learning from decode.
[@sdroege](https://github.com/sdroege) explored the same idea independently
in the [PR #27861 discussion](https://github.com/ggml-org/llama.cpp/pull/27861)
with their own patch; worth checking out too.

**The cache also serves prefill**: small batches (up to 31 tokens, e.g. server
requests with short prompts) run through the cache chain like decode, and in large
prefill batches experts already in the cache are copied GPU-to-GPU instead of over
PCIe (warm 1.1k-token prefill 27.9 to 33.6 t/s). For long prompts add
`-ub 2048 -b 2048`: each expert upload is shared by 4x more tokens (1.1k-token
prefill 33 to 73 t/s) and decode stays within ~4%.

**Lookahead expert prefetch**, thanks to
[@leshchukandrej](https://github.com/leshchukandrej) (llama.cpp PR
[#28414](https://github.com/ggml-org/llama.cpp/pull/28414)), is included as
`--prefetch-experts-slots N` (off by default, 3 recommended). While one layer
computes, a second CUDA stream uploads the next layer's host-resident experts into
rotating staging buffers, overlapping PCIe transfers with compute during prefill.
This fork reserves the staging VRAM up front so the expert cache doesn't crowd it
out. On 2x RTX 3090 with the expert cache it is slower (73 to 58 t/s prefill):
uploads there are PCIe-bound and full-tensor prefetch bypasses the GPU-to-GPU cache
copies. It is expected to help on setups without the cache or with less VRAM
headroom; measure both on yours.

**Other notable tweaks:**
- **Swap budget from measured cost, not a guess**: each step measures real upload
  time (ms/expert) and real token time, then computes how many swaps fit in
  `LLAMA_MOE_CACHE_SWAP_FRAC` of a token (default 25%) — instead of a fixed
  swaps-per-step constant.
- **Pay-back filter on every eviction**: a swap only happens if the candidate's
  measured usage beats the cached victim's by more than what the upload itself
  costs in CPU-equivalent time, so churn can't cost more than it saves.
- **Usage tracking with decay** (`LLAMA_MOE_CACHE_POLICY`: `add` default, `halve`,
  `window`): recent use counts more than old use, so the cache follows shifts in
  which experts are hot instead of freezing on early-token bias.
- **CPU/GPU overlap inside a layer**: the GPU cache chain is queued and its inputs
  copied *before* the CPU miss chain runs, so both compute at the same time instead
  of the scheduler serializing them.
- **Scheduler fixes upstream benefits from too**: no host barrier between two GPU
  splits when neither reads host memory, and a new split is inserted exactly when
  another GPU's result is needed mid-split — both apply to any multi-GPU llama.cpp
  workload, not just this cache.
- **Fused CUDA kernel** for the hyper-connection/KDA gate chain
  (`MUL -> ADD|SCALE -> SIGMOID -> SCALE`, one kernel instead of four), and GLM5-Next's
  KDA Q/K norm collapsed from `rms_norm`+`scale` into one `l2_norm` op.
- **Repacked CPU experts stay cacheable**: uploads read raw bytes straight from the
  GGUF file (recorded per-tensor file offsets) when host memory holds a
  repack-transformed layout instead of the on-disk one.
- `GGML_SCHED_PROF=1` and `LLAMA_MOE_CACHE_STATS=1` for live profiling: barrier vs.
  copy wait time, fill %, in-flight uploads, queue depth, hit rate.

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
