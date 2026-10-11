# Draft pull request: skip the split barrier when it protects nothing

Status: **draft, not opened.** The patch is `0001-ggml-backend-skip-the-split-barrier.patch` (against upstream `23b0202a1`). The measurements on upstream master itself are queued (see "Evidence"); the numbers from the fork use a slightly broader form of the same rule and are given as such.

## Title

`ggml-backend: skip the split barrier when the previous split read nothing this split can overwrite`

## What

`ggml_backend_sched_compute_splits` waits for the previous split's backend before it starts a split that has no inputs ("the allocator may have reused buffer regions across splits"). That wait is needed only if the previous split's asynchronous work can still be *reading* a region this split is about to *write*.

The regions a split can write are those of its own backend's compute buffer (`sched->bufts[split_backend_id]`). The patch skips the wait when none of the previous split's inputs lives in a buffer of that type.

## Why

With `--cpu-moe` (experts on the CPU, everything else on the GPU) a decode graph alternates GPU and CPU splits in every MoE layer. Each CPU split that has no inputs waits for the GPU although the GPU split before it only read GPU memory and the activations copied to the host.

Cost on the fork (same rule, see below): 6.6 ms per decode graph on GLM-5.3-Flash with two RTX 3090 (-9.5% decode), -15 to -17% on a small MoE model with a restricted expert cache on an RTX 3090 and on an RTX 4060 laptop.

## Why it is safe (please check this argument)

- The barrier is only evaluated for splits with `n_inputs == 0`; a split with inputs already synchronises through the input copies.
- Memory reuse by the allocator (`ggml_gallocr`) happens inside one buffer. A split writes only tensors the allocator placed in its own backend's compute buffer (`sched->bufts[split_backend_id]`) and its outputs into the same.
- The previous split can still be reading such a region only through an input that was copied asynchronously *from* it, and every input of a split is listed in `prev->inputs[]`. If none of them lives in a buffer of the current backend's compute buffer type, there is no aliasing to protect.
- Weights, the KV cache and other model buffers are not subject to allocator reuse.

What I could not rule out by reading: a backend whose compute buffer type is shared with another backend's (same `ggml_backend_buffer_type_t` pointer) is handled correctly by the rule (it compares the types and keeps the barrier); a buffer-less input (`buffer == nullptr`) keeps the barrier too.

## Evidence

On the fork (a broader form of this rule: it also skips when both splits run on GPUs, and when the previous split read no host memory), `GGML_SCHED_PROF=1` timer per decode graph, GLM-5.3-Flash:

| build | barrier ms per graph | decode t/s |
|---|---|---|
| before the merge of #29943 (fast) | 0.01 | 23.45 |
| after the merge (unconditional barrier) | 6.66 | 21.34 |
| rule restored | 0.01 | 22.87 - 23.39 |

Restricted-cache OLMoE (small, minutes per run), t/s: RTX 3090 machine 124.3 -> 145.9; RTX 4060 laptop 113.3 -> 134.6.

**Not measured yet on upstream master:** the exact patch in this PR (the buffer-type rule). `stock-master` (upstream master) and `stock-barrier` (master plus this patch) are being built with the same recipe; the table above will be replaced by their numbers (`llama-bench` / `llama-cli`, `-ngl 99 --cpu-moe`, OLMoE-1B-7B Q4_K_M) before the PR is opened.

## How to reproduce

```sh
# any MoE model; experts on the CPU, the rest on the GPU
llama-bench -m OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -ngl 99 --cpu-moe -n 128 -p 0 -r 5
GGML_SCHED_PROF=1 llama-cli ...   # only in the fork: per-graph barrier time
```

## Testing

- `test-backend-ops` and `test-llama-archs` are unaffected (no behaviour change on a single backend).
- Perplexity of the same model with and without the patch is identical (the patch changes only when the host waits, not what is computed): to be attached with the measurements.

## AI assistance

*[To be completed by the author before opening, as AGENTS.md asks: whether and how AI tools contributed, and that the author understands and can defend every line.]*
