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
- [homeofe](https://github.com/homeofe) - AVX2 dot products for IQ3_S and IQ3_XXS on x86 CPUs (`ggml-cpu`), merged into this fork (in the release after b12844) - MIT license
