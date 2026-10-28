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

    llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf

No `-c`, no `-ub`, no `--moe` flags: the context, the micro-batch and the cache are sized per request.

## Best gainers

`pp` = prompt processing tokens/s, `t/s` = generated tokens/s. Stock over ours, gain in brackets, each build on its own defaults (`-np 1` only; stock also gets `-c 16384` so its default does not reject the 12k prompt), warm runs, median of 3 (2 for the two biggest models). **i** = 12k-token prompt, prompt speed; **io** = about 1k in and 1k out, prompt speed and token speed; **o** = short prompt, 1k-token answer. Stock is the official upstream release (`llama-b11721`, CUDA 13.4); ours is the release `dynamic-ctx` (b13510).

{{BEST}}

Older measurements (earlier releases, laptop and CPU rows, perplexity): [fork/docs/perf-history.md](fork/docs/perf-history.md).

## Dynamic context: you do not set `-c`

We all know a smaller context is faster. If you ever wished the context was dynamic per request, now you have it.

What the flags mean: `-c` is the context size the server reserves up front (prompt plus answer tokens, shared by the slots); `-n` is how many tokens one request may generate. With stock llama.cpp you pick `-c` once, for the biggest request you will ever send, and its KV cache is allocated at start whether you use it or not. On a model that does not fit VRAM every GiB of that KV is a GiB less of the expert cache, so a generous `-c` makes every request slower.

Here there is no `-c` to choose. Each request gets exactly the KV it needs, its prompt plus the answer it may produce (`-n` or the request's `max_tokens`), claimed from the end of the GPU pool in steps of 2048 cells and given back when the request ends. The KV takes the coldest experts' blocks first and never more VRAM than it needs; everything else stays what it should be: the weights cache.

The same holds for the prompt itself. Stock sizes its compute buffers for the biggest micro-batch and keeps them (and the prefetch staging) reserved while it generates, so half the GPU can be dead weight during decode. Here the graph and the staging slots are lent from the pool only for the length of a prompt pass and go back to the expert cache for decode.

Qwen3.6-35B-A3B on the same prompt, stock with a static `-c` against ours with a static `-c` and with none:

{{CTX}}

Stock's decode speed falls as `-c` grows (the KV eats the cache); ours stays flat, and with no `-c` at all it matches our best static value.

## Dense parallel split for multi-GPU prompts

With two or more GPUs the prompt is split between the cards: each card computes the dense part (attention and the shared layers) of its own range of the prompt tokens, in parallel. A planner prices one card against the split from measured link speeds and compute, and, when a card sits on a slow link, it can leave the expert distribution out and keep the dense gain, which is where most of the prompt time goes. The prompt speed with the split off (one card) and on:

{{DS}}

`--moe split-experts=0` splits the dense part between the GPUs only: no MoE experts are streamed to the second GPU, the card on the fastest link computes and holds every expert, and the other cards compute just their token ranges of the dense part. The other cards hold copies of the dense weights in their expert-cache blocks, filled at the start of a split prompt (GLM-5.3-Flash 3.0-bit: 4.5 GB over the x4 link, ~1.4 s) and given back to the cache for decoding; the experts never cross the narrow link. It is the right setting for a fast GPU in a narrow slot (x1 or x4 risers, mining-style boards): that card would spend the prompt time waiting for expert uploads over its link, and this way the extra GPU still gives you fast prompt processing. This is the default; `--moe split-experts=auto` lets the planner also put experts on the other cards per layer (worth it only on fast links: x16 PCIe 4/5).

### When the split pays, and how the server decides

A prompt pass costs the longer of two things: streaming the experts it needs that are not cached on the card, and computing its tokens. The split shortens only the computing, so it pays once a pass is compute-bound, and only by more than the dense-weight copies it fills on the other card first. On 2x RTX 3090 (x16 + x4) that is from about:

| model | split pays from | max micro-batch |
|---|---|---|
| GLM-5.3-Flash 3.0-bit | ~5-7k prompt tokens | 10368 |
| Qwen3.8-Flash-Next IQ4_XS | ~8k | 16384 |
| MiMo-V2.6-Flash 3-bit | ~10-12k | 16384 |

You do not set this. The server times every prompt pass, keeps the compute per token per card at each context length (the attention makes a token cost more the more context it sees), computes each pass's expert upload from what the cache holds, and picks one card or the split per request from those numbers, starting from timings taken when the model loads. The timings are kept per model, cards and build in the state file, so a restart does not start over. `--moe dense-split=1` forces the split, `dense-split=0` one card.

Models whose attention selects cells (GLM-5.3, Qwen3.8: the indexer's top 2048) now read only the selected cells from the context length where that is faster on your card, measured at load (~20k cells on an RTX 3090) instead of a fixed ~33k.

## Perplexity against stock

Same text, same chunks, official stock `llama-perplexity` against ours. The number in brackets is the per-chunk difference to stock in percent of perplexity (mean ± standard error); 0.00 means identical to four decimals. *default* is what you get without settings (the planner decides per prompt size whether to use one card or both), *one card* and *two cards* force the path (`dense-split=0` / `dense-split=1`), *dense split only* is `split-experts=0` (see below). The plain Qwen3.8 27B models have no dense split, so for them *two cards* splits the experts only. About 10 chunks of 1024 tokens per model: differences under 0.2% are inside the noise.

{{PPL}}

Reproduce one row (stock and ours, 12 chunks, the perplexity of each is stored in `runs.db` and printed by `run ppl-table`):  
```
PERF_PPL_CHUNKS=12 run bench --models qwen36 --builds stock-latest,b13586 --variants default -t pplb -n 1
```

## KV tiers (experimental)

A request whose KV does not fit in f16 moves to q8_0, then q4_0 (both in place in the pool), then whole layers' KV to host memory, so the same small GPU can take much larger documents and only the huge requests pay the quantisation. It is on by default; `--moe kv-fallback=k8v8,k4v4,ram` lists the tiers (for example `k8v8,ram` skips q4_0, `none` keeps f16 only and turns the ladder off). Perplexity of a quantised KV on Qwen3.6-35B (CPU, fixed text): f16 1.5244, q8_0 1.5257, q4_0 1.5217: no measurable loss. Tokens a card holds per tier, OLMoE with 3.5 GB of VRAM (a small-GPU stand-in):

{{TIERS}}

