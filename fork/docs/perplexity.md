# Perplexity against stock: method and notes

Moved from the README on 2026-11-12 (text and numbers unchanged).

Same text, same chunks, official stock `llama-perplexity` against ours. The number in brackets is the per-chunk difference to stock in percent of perplexity (mean ± standard error); 0.00 means identical to four decimals. *default* is what you get without settings (the planner decides per prompt size whether to use one card or both), *one card* and *two cards* force the path (`dense-split=0` / `dense-split=1`), *dense split only* is `split-experts=0` (see below). The plain Qwen3.8 27B models are dense with no split path, so their split columns are empty. The GLM two-card cell is not measured yet. About 10 chunks of 1024 tokens per model: differences under 0.2% are inside the noise.

**What this measures.** `llama-perplexity` with 1024-token passes, so the prompt path: stock and ours both run the experts on the GPU with the same kernels there, which is why many cells are identical to four decimals (checked chunk by chunk). It does not test the decode path, where a cached expert runs on the GPU and a missed one on the CPU and the rounding differs slightly. Perplexity with one token per call exercises that path (`run ppl` can be pointed at it); between two of our own builds it differed in the 4th digit (Qwen3.6-35B: 1.5219 against 1.5209), and against stock (10 chunks for Qwen Next IQ3_S, 9 for GLM 3.5-bit, per-chunk difference, mean ± standard error): Qwen Next IQ3_S 1.3954 stock, 1.3975 ours (+0.15 ± 0.08%, 7 of 10 chunks worse); GLM 3.5-bit 1.6270 stock, 1.6335 ours (+0.40 ± 0.35%, 7 of 9 worse). Both are inside the spread that stock itself shows between micro-batch sizes (next section); only these two files were repeated at 10 chunks, the other models have 2-chunk decode-path numbers only, which cannot decide anything.


| model | stock ppl | ours default | ours, one card | ours, dense split only | ours, two cards (f32 hand-offs) | chunks |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash-IQ2_XXS | 1.5665 | 1.5662 (-0.02 ± 0.13%) | - | - | - | 10 |
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

DeepSeek 4.1 row: build dsfix1 (the fix for the all-zero token bug without `-c`), stock is the official b11887 binary, default path only, 10 chunks of 1024 tokens; 4 of 10 chunks are worse than stock, the cumulative difference shrinks from -0.31% after chunk 1 to -0.02% after chunk 10. Its decode speed rows are not in the tables yet: they come back after the gainers are rerun on this build.

### Perplexity moves with the micro-batch size, in stock too

The few cells above that differ from stock (Qwen 3.8 Next IQ3_S on the two-card paths, +0.3%) are not a loss of quality of the split. A prompt cut into smaller passes gives a slightly different perplexity on **official stock** as well, in either direction: the same text, same model, same binary, only `-ub` changed (10 chunks of 1024 tokens, one RTX 3090 or CPU):

| model | -ub 1024 | 512 | 256 | 128 | 64 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next IQ3_S (stock) | 1.3927 | 1.3946 (+0.14%) | 1.3996 (+0.50%) | 1.3983 (+0.40%) | 1.4005 (+0.56%) |
| Qwen3.6-35B-A3B hybrid (stock) | 1.7843 | - | 1.7842 | 1.7855 | - |
| OLMoE-1B-7B Q4_K_M (stock) | 2.1618 | - | 2.1627 | - | - |

The two-card dense split cuts every prompt pass into token ranges, one per card, so it behaves like a smaller `-ub`: Qwen Next IQ3_S lands at 1.3969 / 1.3975, between stock's own 512 and 256 rows. With the same pass shape (one card, any `-ub`) ours is identical to stock to four decimals.

**Why it moves.** Floating-point sums depend on the order of the additions, and the order depends on how many tokens a pass has: the quant dot-product kernels themselves (IQ3_S and the others) are the same at every pass size above 8 tokens (below that the CUDA backend switches to a matrix-vector kernel, `MMVQ_MAX_BATCH_SIZE`), but the tile shape and the split of the reduction over thread blocks follow the number of tokens; flash attention picks its kernel and tile width from the number of query tokens; the dense BF16/Q4_K weights, the router and the recurrent layers sum in an order that depends on the pass; the CPU backend blocks its matrix products by the batch size. Which of these dominates for a given model has not been traced op by op. The last bits differ, and normally nobody sees them. A mixture-of-experts model amplifies them: Qwen Next picks 10 of 512 experts per token (GLM and the sparse-attention models also pick top-k tokens), and a near-tie that flips changes the token's output, and from there every later token that sees it. Models without such picks hardly move (Qwen3.6 above, +0.00 to +0.07%), and the quant mix matters (IQ3_S is the most sensitive file measured, IQ1_M moves by 0.01%).

Consequences: compare perplexities only at the same `-ub` (and the same number of chunks); a single chunk moves by up to ±1.2% on its own, so use 10 or more chunks and a paired per-chunk difference (`python3 fork/run.py ppl-table`); repeating the same run does not average anything out, it returns the same number.

**Check it on any model** with official stock, no fork code involved. Only `-ub` changes (`-b` stays 1024 so the passes can be cut):
```sh
for ub in 1024 512 256 128 64; do
  echo "ub=$ub"; llama-perplexity -m MODEL -f fork/bench/longsrc.cpp -c 1024 --chunks 10 -b 1024 -ub $ub -nr 2>&1 | grep "Final estimate"
done
```
Add `-ngl 0` (CPU only) for a run that needs no GPU: OLMoE-1B-7B IQ3_S with one 256-token chunk gave 2.2241 (`-ub 256`), 2.2292 (64), 2.2140 (16), 2.2284 (1), and `-nr` (no weight repacking) made no difference, so the repacked CPU kernels are not the cause. A model that does not move (Qwen3.6 on CPU: 1.7850 at every `-ub`) is the control.
