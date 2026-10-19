# Features, internals and known limits

Moved out of the README to keep it short; nothing deleted.

### Pinned weights (automatic when the model fits in RAM)

Added in release b11341. When `llama-server` runs a MoE model bigger than VRAM that fits in
the RAM available at startup, the fork loads its weights into pinned (page-locked) memory instead of
memory-mapping the file (`--load-mode pin` does it by hand, `--load-mode mmap` turns it off).
The GPUs then read experts straight from RAM by DMA, for prompt processing and for the cache's
uploads during decode. Only the server does this by default: it starts once and serves many
requests, so the longer startup pays off. `llama-cli` and the other tools keep mmap (fast
startup for one-off runs); pass `--load-mode pin` to pin there too.

| 2x RTX 3090, model in RAM | mmap | pinned |
|---|---|---|
| GLM-5.3-Flash: 12k prompt processing t/s | 180 | **262 (+45%)** |
| GLM-5.3-Flash: 12k decode t/s | 15.6 | 16.3 (+5%) |
| GLM-5.3-Flash: chat decode t/s | 20.4 | same (decode isn't limited by uploads) |
| Qwen3.8-Flash-Next: 12k prompt processing t/s | 372 | **538 (+45%)** |
| Qwen3.8-Flash-Next: 12k decode t/s | 38.4 | 42.3 (+10%) |
| GLM-5.3-Flash: startup until the server answers (model in page cache) | 38 s | ~100 s |

**Why the server pins by default:** the extra startup is paid once, every request after it is
faster. On GLM-5.3-Flash it costs ~60 s more at startup and saves ~22 s on every 12k-token
prompt (12,302 tokens at 179 vs 261 t/s) plus a few seconds per chat reply, so it pays back
after about 3 long prompts; a server usually runs for hours. To keep mmap on the server
(faster startup, RAM stays free for other programs):

```sh
llama-server -m model.gguf --load-mode mmap
```

To pin with `llama-cli` (or another tool), e.g. for a long session with big prompts:

```sh
llama-cli -m model.gguf --load-mode pin
```

(an explicit `--load-mode` is used as given: there is no automatic fallback to mmap if the
model doesn't fit in RAM.)

Pros: much faster prompt processing, slightly faster decode, and no page-fault stalls on the
first requests (the whole model is read once at startup).
Cons: startup takes longer (pinning ~100 GB runs at the NVIDIA driver's speed, ~2.5 GB/s),
the model's RAM stays locked while the server runs (other programs can't use it), and it only
works when the model fits in RAM. Models bigger than RAM (MiMo-V2.6 here) keep mmap, which pages
experts in from disk on demand. If pinning makes the system swap anyway, the fork reloads with
mmap and says so.

### Persistent hot experts

Added in release b11391. When the server stops, the expert cache saves how often each expert was
used (lifetime activation counts, ~50 KB per model) to `~/.cache/llama.cpp/`; the next start
preloads each layer's most-used experts into VRAM before the first request, instead of warming up
over the first answers. Measured on GLM-5.3-Flash, first request after a restart: decode +4-6%
(20.5-20.9 -> 21.8 t/s), cache hits +2-4 points, short-prompt processing ~3x (5.6 -> 17 t/s, its
experts are already on the GPU). Steady-state speed is unchanged.

- `LLAMA_MOE_CACHE_PROFILE=<file>` uses another profile file, `=0` turns it off.
- A GGUF can carry the counts itself (u32 array `moe_cache.expert_usage`, n_layer x n_expert);
  it is used when there is no local profile yet. The model file is never written to.
- Also new: `--moe-cache-window N` (tokens of recent usage the cache scores by, default 64) and
  `LLAMA_MOE_CACHE_STICKY=0.3` (never evict the most-used 30% of each layer's slots; off by
  default, +1.6% measured, within noise).
- Tried and dropped: uploading a long prompt's hot experts right after it. It raised the hit
  rate (67% -> 72%) but the extra uploads compete with decode (32-token answers -6% to -20%);
  the prefill preheat below does it device-to-device instead.

### Prefill preheat

Added in release b11397. While a long prompt is processed, the experts it uses are copied to the
GPU anyway (for the offloaded matmuls). The expert cache now keeps the prompt's most-used ones
from those copies, device-to-device (no extra PCIe traffic), in place of cached experts the prompt
used less. Decode after the prompt then starts with the prompt's experts in VRAM. GLM-5.3-Flash,
12k-token code prompt then 256 tokens, 1 GPU: decode 13.5 -> **15.3 t/s (+12.7%)**, cache hits
43% -> 55%; prompt processing -1%; short answers and chat unchanged.
`LLAMA_MOE_CACHE_ADOPT` = share of a layer's slots that one prompt batch may replace (default 1.0, 0 = off).

### Global expert pool

The expert cache used to give every layer the same fixed number of slots. Some layers spread their traffic over many experts, others use a few very often, so part of every layer's share held cold experts while busier layers missed. Now the cache of a GPU is one pool of blocks per expert size (one block = one expert's up, gate and down weights). No block belongs to a layer: a missed expert takes a free block, or the block of the lowest-scored cached expert of any layer with the same expert size. Layers end up with as many experts as earn their place (OLMoE, 35% of the experts in VRAM: 17 to 26 per layer instead of 22 each). Scores of different layers are compared on one scale (the busiest expert of any layer); per-layer scales starved layers with a few hot experts.

It helps most where VRAM is tight, which is where 8 and 16 GB cards are. Same binary, `--moe pool=0` (fixed slots per layer) vs the default pool, 400-500 generated tokens, n=3-4:

| model, share of experts that fit in VRAM | t/s fixed slots | t/s pool | cache hits fixed slots | cache hits pool |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (12-layer test copy), 20% | 132.5 | 137.3 (+3.6%) | 80.2% | 83.1% |
| Qwen3.8-Flash-Next (12-layer test copy), 25% | 138.6 | 141.5 (+2.1%) | 83.6% | 85.5% |
| MiMo-V2.6-Flash 3-bit (12-layer test copy), 25% | 94.9 | 97.8 (+3.1%) | 71.7% | 73.4% |
| OLMoE-1B-7B, 35% | 177.5 | 181.7 (+2.4%) | 78.3% | 79.3% |
| GLM-5.3-Flash 3.0-bit, all free VRAM (2x RTX 3090), n=2 | 20.31 | 20.23 (same) | 72.9% | 73.3% |
| MiMo-V2.6-Flash 3-bit, all free VRAM (2x RTX 3090), n=2 | 32.70 | 32.44 (same) | 83.6% | 83.8% |

With most of the model in VRAM there is little to gain and nothing is lost within noise. The test copies keep the first 12 layers of the real model so a run takes minutes. `--moe pool=0` turns the pool off; with the predictor on (`--moe predict=1`) the fixed slots are used.

### Hot layers on the GPU with the slower link

With two GPUs where one sits in a slower slot (here x4 at 6.4 GB/s next to x16 at 25.5 GB/s), every cache miss on the slow card costs four times as long to upload. `--moe layer-heat=1` puts the layers whose experts are used most unevenly (their caches hit most) on the slower card and the rest on the faster one, with the same number of layers per card. It reads the usage counts saved by earlier runs (`~/.cache/llama.cpp/moe-state.ini`), so the first run of a model is placed as before. GLM-5.3-Flash 3.0-bit, 2x RTX 3090, n=2: 20.44 -> 20.62 t/s, cache hits 74.1% -> 75.6%. Off by default until a longer test confirms it.

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
  (4.3 GiB, works with any `glm5-next` GLM-5.3-Flash GGUF). We made it: no GLM MTP GGUF
  existed, so `fork/bench/glm_splice_mtp.py` pulls just the MTP tensors out of unsloth's
  UD-Q4_K_XL GGUF with HTTP range requests (~4.3 GiB instead of the whole model) and writes
  them as a draft file; see [`fork/bench/`](fork/bench/) to rebuild it.
- **Why depth is automatic:** every drafted token has to be verified, and each token picks
  its own experts. Experts that miss the VRAM cache run on the CPU at the same cost per
  drafted token as per generated one, so the best depth depends on how much of the model the
  GPUs hold, not only on how often drafts are accepted. The fork starts from a guess (model
  size vs free VRAM: up to 1x → 3, up to 2x → 2, more → 0; a separate `-md` draft is then not
  loaded at all, built-in MTP layers are kept and measured from depth 0), then measures real
  generation speed and moves the depth up or down in doubles, then by 1, re-checking every
  4096 tokens. The max depth is 2 when the model is bigger than VRAM (the rollback buffers of
  hybrid models are sized by it and cost cache VRAM), 5 when it fits.
- Measured here: Qwen3.8-Flash-Next (1.9x VRAM, 95% cache hits) **57.0 t/s with MTP vs 47.1 without (1.21x)**
  at depth 2. GLM-5.3-Flash (2.3x VRAM, ~70% hits) is slower with MTP (17.6 vs 20.2 t/s), so
  the fork doesn't load the draft there (a warning says so; `--spec-draft-n-max N` forces it).
  More VRAM should make GLM MTP worth it: with 3x 24 GB, GLM is ~1.5x VRAM (less than
  Qwen's 1.9x here), cache hits should reach 90%+ and verifying drafts would cost little CPU
  work (estimate ~1.2-1.3x from MTP); with 4 cards nearly all experts fit. A third card on
  a chipset x4 slot slows cache uploads and prompt processing, not decode.
- Since release b11367: built-in MTP (MiMo-V2.6) is loaded on models bigger than VRAM too, and the
  tuner decides; on MiMo here it keeps depth 0 (no measurable gain).
- The depth tuner keeps its current depth (at first the model-size guess) unless another depth is
  at least 5% faster: early in a reply (the model's reasoning) depths measure near-tied, and
  without that margin noise sometimes picked depth 1 on Qwen (~51 instead of ~60 t/s).
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
- Pinned weights in `llama-server` when the model fits in RAM: experts are uploaded by direct
  DMA (prompt processing GLM +45%, Qwen +45%); `--load-mode pin` / `mmap` to choose.
- Persistent hot experts: the cache's usage is saved at shutdown and preloaded at the next start.
- Prefill preheat: a long prompt's hot experts stay in the cache (device-to-device, no extra PCIe).
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
- Regressions and missing optimisations found in upstream along the way, and how they were
  solved (the host barrier of #29943 that cost GLM 9.5% decode, the shared-expert fusion of
  #29184 that slows the fused MMVQ kernel for models without a shared expert, an AVX2/AVX-512
  Q2_0 kernel, a GLM-5 graph that was rebuilt every token): the table in
  [`fork/PATCHES.md`](fork/PATCHES.md#upstream-regressions-and-bugs-found-and-how-they-were-fixed).

## Known limits and next steps

- Prompt processing with mmap (models bigger than RAM, or `--load-mode mmap`) is ~20% below
  stock. Next milestone: use both GPUs' PCIe links for it (split each layer's experts by
  measured bandwidth), and pin what fits for models bigger than RAM.
- GPU order, ongoing research. On this box decode is ~7% faster (tetris 28.6 vs
  26.7 t/s, cache hits 90% vs 87%, same cache size) when the layers stay in bus order
  (x4 GPU first) than when the x16 GPU takes the first layers, while prompt processing
  wants the x16 GPU. So since this release prompt processing goes to the fastest-link
  GPU and layers keep bus order, which gets both. Why the layer order matters is not
  known yet: the two cards differ (x16: Gainward 3-slot 370 W with partly blocked
  airflow, x4: Dell OEM 2-slot 350 W; a budget build, these were the cards available
  at a good price). Thermals are ruled out: logged every 5 s through a 12-minute
  128k-token run and the pinned tests (30+ minutes), neither GPU ever hit a thermal or
  hardware slowdown (x16 card peaked at ~79 °C, clocks steady at 1860-1890 MHz; CPU and
  DIMMs also stayed below throttling). So the difference comes from where the layers sit
  relative to the links and cards, not from heat. Next: per-GPU timing of each layer's
  split during decode, and in auto mode choose the order per request.
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
