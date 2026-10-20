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

Best valid run of ours against the median of the newest official stock (b11742, 2026-10-20), sorted by the bigger gain; rows where neither gain reaches +10% are left out (models that fit in VRAM and the dense 27B models run at stock speed). Upper number stock, lower ours. **12k/1k** = 12.4k-13.8k-token code-review prompt, 1024 tokens out; **3k/1k** = ~3k-token source file plus an edit instruction, answer up to 1024 tokens; **short/1k** = one-line prompt, 1024 tokens out (no prompt speed: too short to time). t/s = decode, pp = prompt processing, ppl = perplexity on b13600 (details below). Valid = same power limits as stock (x4 card 220 W, x16 card 280 W), answer not looped, decode only from answers of 256+ tokens. 2x RTX 3090 (x16 + x4), Ryzen 7 3700X, 128 GB DDR4-3200.

| model | test | t/s | pp | gain t | gain pp | ppl | build | output |
|---|---|---|---|---|---|---|---|---|
| MiMo 2.6F 3-bit | 12k/1k | 14.2<br>25.8 | 49<br>1022 | +82% | +1971% 20.7x | 1.8758<br>1.8758 | b13600 | [17120](fork/outputs/17120.txt) [17121](fork/outputs/17121.txt) |
| MiMo 2.6F 3-bit | 3k/1k | 14.3<br>26.7 | 48<br>738 | +86% | +1427% 15.3x | 1.8758<br>1.8758 | b13563 | [16753](fork/outputs/16753.txt) [16591](fork/outputs/16591.txt) |
| GLM 5.3 3-bit | 12k/1k | 12.9<br>19.0 | 42<br>627 | +47% | +1378% 14.8x | 1.7626<br>1.7578 | b13586 | [16725](fork/outputs/16725.txt) |
| GLM 5.3 3-bit | 3k/1k | 12.7<br>20.3 | 40<br>585 | +60% | +1365% 14.7x | 1.7626<br>1.7578 | b13600 | [17137](fork/outputs/17137.txt) |
| GLM 5.3 3.5-bit | 3k/1k | 10.7<br>12.9 | 31<br>224 | +21% | +617% 7.2x | 1.6326<br>1.6265 | b13563 | [16756](fork/outputs/16756.txt) |
| Qwen 3.8 Next IQ4_XS | 12k/1k | 32.0<br>51.4 | 210<br>1483 | +61% | +606% 7.1x | 1.4101<br>1.4101 | b13600 | [17108](fork/outputs/17108.txt) [17109](fork/outputs/17109.txt) |
| GLM 5.3 3.5-bit | 12k/1k | 10.8<br>11.0 | 34<br>230 | +2% | +579% 6.8x | 1.6326<br>1.6265 | b13563 | [16759](fork/outputs/16759.txt) |
| Qwen 3.8 Next IQ4_XS | 3k/1k | 30.9<br>53.6 | 207<br>1240 | +74% | +499% 6.0x | 1.4101<br>1.4101 | b13600 | [17113](fork/outputs/17113.txt) |
| Qwen 3.8 Next IQ3_S | 12k/1k | 44.2<br>54.1 | 376<br>1469 | +22% | +291% 3.9x | 1.3927<br>1.3927 | b13600 | [17096](fork/outputs/17096.txt) [17097](fork/outputs/17097.txt) |
| Qwen 3.8 Next IQ3_S | 3k/1k | 43.8<br>55.8 | 384<br>1254 | +27% | +227% 3.3x | 1.3927<br>1.3927 | b13600 | [17101](fork/outputs/17101.txt) |
| MiMo 2.6F 3-bit | short/1k | 14.3<br>33.1 | -<br>- | +131% 2.3x | - | 1.8758<br>1.8758 | b13600 | [17128](fork/outputs/17128.txt) |
| Qwen 3.8 Next IQ4_XS | short/1k | 31.4<br>55.1 | -<br>- | +75% | - | 1.4101<br>1.4101 | b13586 | [16788](fork/outputs/16788.txt) |
| GLM 5.3 3-bit | short/1k | 12.8<br>20.8 | -<br>- | +62% | - | 1.7626<br>1.7578 | b13600 | [17141](fork/outputs/17141.txt) |
| GLM 5.3 3.5-bit | short/1k | 9.8<br>14.9 | -<br>- | +52% | - | 1.6326<br>1.6265 | b13563 | [16687](fork/outputs/16687.txt) |
| Qwen 3.8 Next IQ3_S | short/1k | 44.2<br>56.9 | -<br>- | +29% | - | 1.3927<br>1.3927 | b13563 | [16766](fork/outputs/16766.txt) |
| Qwen 3.8 Next IQ1_M | 3k/1k | 68.3<br>67.7 | 1446<br>1751 | -1% | +21% | 1.4756<br>1.4756 | b13600 | [17088](fork/outputs/17088.txt) [17089](fork/outputs/17089.txt) |

Every answer behind the table is in [fork/outputs](fork/outputs) as `<id>.txt` (header: build, test, speeds, arguments, power); stock's answers too. With the run database: `python3 fork/run.py out ID` (get it with `git fetch https://github.com/neurall/llama.cpp runs-db && git show FETCH_HEAD:runs.db > fork/runs.db`).

Our builds in the table are b13600 and two earlier builds of this release line (b13563, b13586); each cell shows the build with the best run. Older tables (earlier releases, laptop and CPU rows): [fork/docs/perf-history.md](fork/docs/perf-history.md).

### Reproduce a row

1. Binaries: ours [`llama-release-b13600-c1d4fe980-bin-ubuntu-cuda-13.4-x64.tar.gz`](https://github.com/neurall/llama.cpp/releases/download/release-b13600-c1d4fe980-dynamic-ctx/llama-release-b13600-c1d4fe980-bin-ubuntu-cuda-13.4-x64.tar.gz) (Windows zip on the [release page](https://github.com/neurall/llama.cpp/releases/tag/release-b13600-c1d4fe980-dynamic-ctx); add the `cudart-` archive if CUDA 13 is not installed); stock: `llama-b11742-bin-ubuntu-cuda-13.4-x64.tar.gz` from [ggml-org b11742](https://github.com/ggml-org/llama.cpp/releases/tag/b11742).
2. Start the server (port 9931). The `-c` values are the ones the table's runs used:

| test | ours | stock |
|---|---|---|
| 12k/1k | `llama-server -m MODEL -np 1` | `llama-server -m MODEL -np 1 -c 16384` |
| 3k/1k | `llama-server -m MODEL -np 1 -c 4096` | `llama-server -m MODEL -c 4096` |
| short/1k | `llama-server -m MODEL -np 1 -c 1024` | `llama-server -m MODEL -c 1024` |

   GLM 5.3 3.5-bit (137 GB) is larger than the 128 GB RAM here; our rows ran with `-lm mmap --moe pin-budget=108447924224` added (pin 101 GiB, map the rest).
3. Send the exact request the harness sends ([fork/bench/req-*.json](fork/bench): 12k-1k, 3k-1k-a / -b (the two edit instructions), short-1k), twice on the same server like the harness:
```sh
curl -sL https://raw.githubusercontent.com/neurall/llama.cpp/release/fork/bench/req-12k-1k.json | curl -s localhost:9931/v1/chat/completions -H 'Content-Type: application/json' -d @- | grep -o '"timings":{[^}]*}'
```
   `prompt_per_second` is pp, `predicted_per_second` is t/s.
4. Perplexity (same text and chunks as the table below; ours forces the expert-cache path even where the model fits VRAM):
```sh
llama-perplexity -m MODEL -f fork/bench/longsrc.cpp -c 1024 --chunks 12 -b 1024 -ub 1024 -nr                             # stock
llama-perplexity -m MODEL -f fork/bench/longsrc.cpp -c 1024 --chunks 12 -b 1024 -ub 1024 -nr --cpu-moe --moe-expert-cache -1   # ours
```

The whole table with the harness (stock and ours, 2 runs per cell, results into `fork/runs.db` and `fork/outputs/`): set the paths in [fork/local.toml.example](fork/local.toml.example) and the model files in [fork/models.ini](fork/models.ini), then `python3 fork/run.py run gainers -n` (lists the cells) and the same without `-n`; perplexity: `python3 fork/run.py ppl -b b13600`.



Models: GLM original 3.0-bit GGUF [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
works as-is (reference quality); the Q4_K attention variant above (same experts,
non-expert Q8_0 weights at Q4_K, +0.95% perplexity) decodes ~10% faster.

**Same VRAM, different use.** Each GLM token uses 8 of 288 experts per layer. Stock
fills VRAM with whole layers (~14 of 42), mostly experts the current token doesn't
touch, so the CPU does most expert work and the GPUs wait. This fork fills the same
VRAM with the ~100 most-used experts of every layer, so most expert work runs on the
GPUs, at the same time as the CPU handles the misses.

## How it works

<img src="fork/docs/img/vram-layout.svg" alt="One VRAM buffer per GPU: compute graph, KV and expert cache share it" width="100%">

- **Expert cache.** All experts stay in RAM; each GPU's free VRAM is one buffer of expert blocks. The GPUs compute the cached (hot) experts while the CPU computes the misses, in parallel. A missed expert that keeps being used is uploaded and replaces a colder one.
- **Learned hot experts, kept between runs.** Per model, the lifetime use of every expert is saved in `moe-state.ini` (`~/.cache/llama.cpp/`, Windows `%LOCALAPPDATA%\llama.cpp\`) and updated by every run; the next start preloads the hottest experts. The first run or two of a new model are slower (it is a cache). A pre-warmed state for the models in the table is in the repo: copy [fork/moe-state.ini](fork/moe-state.ini) there if you have none.
- **Prompts in few big passes.** Every prompt pass streams the experts the GPU does not hold, so the micro-batch (`-ub`) is sized per request (up to 12288 tokens) and the graph borrows expert blocks only for that pass; decode gets them back.
- **KV inside the same buffer.** With `-np 1` no context is reserved: each request claims the KV it needs and gives it back at its end, so a big `-c` costs nothing. A request whose KV does not fit in f16 converts it in place to q8_0, then q4_0, then moves whole layers' KV to host RAM. On by default (`--moe kv-fallback=k8v8,k4v4,ram`); skip q4_0 with `--moe kv-fallback=k8v8,ram` (q8_0 KV measured within noise of f16: OLMoE perplexity 2.1926 vs 2.1946; q4_0 +0.58%), turn it off with `--moe kv-fallback=none`.
- **Two GPUs.** A planner picks per prompt size one card or a split of the dense layers over both, from measured link speed and compute.

Every setting and its default: [fork/docs/fork-knobs.md](fork/docs/fork-knobs.md). Internals and measurements per feature: [fork/docs/features.md](fork/docs/features.md).

## Perplexity against stock

Same text, same chunks, official stock `llama-perplexity` against ours. The number in brackets is the per-chunk difference to stock in percent of perplexity (mean ± standard error); 0.00 means identical to four decimals. *default* is what you get without settings (the planner decides per prompt size whether to use one card or both), *one card* and *two cards* force the path (`dense-split=0` / `dense-split=1`), *dense split only* is `split-experts=0` (see below). The plain Qwen3.8 27B models are dense with no split path, so their split columns are empty. The GLM two-card cell is not measured yet. About 10 chunks of 1024 tokens per model: differences under 0.2% are inside the noise.

| model | stock ppl | ours default | ours, one card | ours, dense split only | ours, two cards (f32 hand-offs) | chunks |
| --- | --- | --- | --- | --- | --- | --- |
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

Measured on build b13600 (12 chunks asked, 9 to 12 used where a text ended earlier), stock is the official b11742 binary.

Reproduce: see [Reproduce a row](#reproduce-a-row), step 4.

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
- [homeofe](https://github.com/homeofe) - AVX2 dot products for IQ3_S and IQ3_XXS on x86 CPUs (`ggml-cpu`), merged into this fork (in this release) - MIT license
