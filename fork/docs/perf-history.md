# Older measurements

Moved out of the README on 2026-10-19 so the front page shows only the newest table. Every number is kept as it was published; the build column of each row says which build it is from.

### Best gainers so far (measured 2026-10-14; 2x RTX 3090 24 GB with one x16 and one x4 link, and an 8 GB RTX 4060 laptop where marked)


| model | cell | builds stock fork | pp: stock / **fork** (gain) | t/s: stock / **fork** (gain) | ppl: stock / fork |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash 3.0-bit (106 GB) | <sub>12k prompt, `-c 16384` (best pp)</sub> | <sub>b11381 b12719</sub> | 42.8<br>**756**&nbsp;<sub>[(17.7x)](fork/bench/REPRO.md#glm-12k)</sub> | 11.5<br>**15.9**&nbsp;<sub>[(1.38x)](fork/bench/REPRO.md#glm-12k)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| GLM-5.3-Flash 3.0-bit | <sub>1k edit, 128 out</sub> | <sub>b11452 b12825</sub> | 41.5<br>**495**&nbsp;<sub>[(11.9x)](fork/bench/REPRO.md#glm-edit)</sub> | 12.4<br>**19.0**&nbsp;<sub>[(1.53x)](fork/bench/REPRO.md#glm-edit)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| Qwen3.6-35B-A3B hybrid (11.4 GiB) on an 8 GB laptop (4060) | <sub>17-token prompt, 100 out</sub> | <sub>b11514 b12843</sub> | 20.2<br>**98.0**&nbsp;<sub>[(4.9x)](fork/bench/REPRO.md#laptop-17)</sub> | 15.6<br>**56.5**&nbsp;<sub>[(3.6x)](fork/bench/REPRO.md#laptop-17)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.8-Flash-Next UD-IQ4_XS (88 GB) | <sub>1k edit (best pp)</sub> | <sub>b11452 b12364</sub> | 160<br>**599**&nbsp;<sub>[(3.7x)](fork/bench/REPRO.md#qni4-edit)</sub> | 30.6<br>**45.8**&nbsp;<sub>[(1.50x)](fork/bench/REPRO.md#qni4-edit)</sub> | - |
| Qwen3.6-35B-A3B Q2_0 (CPU only, Ryzen 5 3600) | <sub>short prompt, 100 out (CLI)</sub> | <sub>b11302 b11648</sub> | 10.2<br>**37.6**&nbsp;<sub>[(3.7x)](fork/bench/REPRO.md#q2-cpu)</sub> | 6.3<br>**11.3**&nbsp;<sub>[(1.79x)](fork/bench/REPRO.md#q2-cpu)</sub> | - |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>12k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 687<br>**918**&nbsp;<sub>[(1.34x)](fork/bench/REPRO.md#laptop-12k)</sub> | 13.6<br>**45.6**&nbsp;<sub>[(3.4x)](fork/bench/REPRO.md#laptop-12k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>40k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 584<br>**726**&nbsp;<sub>[(1.24x)](fork/bench/REPRO.md#laptop-40k)</sub> | 11.7<br>**35.9**&nbsp;<sub>[(3.1x)](fork/bench/REPRO.md#laptop-40k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| Qwen3.8-Flash-Next UD-IQ4_XS | <sub>12k prompt</sub> | <sub>b11381 b11707</sub> | 201<br>**553**&nbsp;<sub>[(2.8x)](fork/bench/REPRO.md#qni4-12k)</sub> | 29.6<br>**45.9**&nbsp;<sub>[(1.55x)](fork/bench/REPRO.md#qni4-12k)</sub> | - |
| Qwen3.6-35B-A3B hybrid on an 8 GB laptop | <sub>125k prompt, 32 out</sub> | <sub>b11514 b12843</sub> | 448<br>**557**&nbsp;<sub>[(1.24x)](fork/bench/REPRO.md#laptop-125k)</sub> | 9.2<br>**25.2**&nbsp;<sub>[(2.7x)](fork/bench/REPRO.md#laptop-125k)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| MiMo-V2.6-Flash IQ3_XXS (132 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12250</sub> | 30.0<br>**73.6**&nbsp;<sub>[(2.5x)](fork/bench/REPRO.md#mimo-edit)</sub> | 4.8<br>**9.0**&nbsp;<sub>[(1.88x)](fork/bench/REPRO.md#mimo-edit)</sub> | 1.5717<br>missing |
| Qwen3.6-35B-A3B Q2_0 on an 8 GB laptop (4060) | <sub>short prompt, 100 out (CLI)</sub> | <sub>b11261 b11648</sub> | 41.5<br>**101.9**&nbsp;<sub>[(2.5x)](fork/bench/REPRO.md#q2-laptop)</sub> | 28.9<br>**58.6**&nbsp;<sub>[(2.0x)](fork/bench/REPRO.md#q2-laptop)</sub> | - |
| GLM-5.3-Flash 3.0-bit | <sub>game, 600 out (best t/s)</sub> | <sub>b11381 b12209</sub> | - | 11.7<br>**27.8**&nbsp;<sub>[(2.4x)](fork/bench/REPRO.md#glm-game)</sub> | [**1.4191**](fork/bench/REPRO.md#glm-12k)<br>1.4234\* |
| MiMo-V2.6-Flash IQ3_XXS | <sub>chat, short prompt (best t/s)</sub> | <sub>b11381 b11707</sub> | - | 4.9<br>**10.5**&nbsp;<sub>[(2.1x)](fork/bench/REPRO.md#mimo-chat)</sub> | 1.5717<br>missing |
| Qwen3.8-Flash-Next UD-IQ4_XS | <sub>chat, short prompt (best t/s)</sub> | <sub>b11381 b11365</sub> | 41.8<br>**63.3**&nbsp;<sub>[(1.51x)](fork/bench/REPRO.md#qni4-chat)</sub> | 30.9<br>**60.9**&nbsp;<sub>[(1.97x)](fork/bench/REPRO.md#qni4-chat)</sub> | - |
| GLM-5.3-Flash 3.5-bit (137 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12250</sub> | 26.2<br>**49.0**&nbsp;<sub>[(1.87x)](fork/bench/REPRO.md#glm35-edit)</sub> | 9.7<br>**9.9**&nbsp;<sub>[(1.02x)](fork/bench/REPRO.md#glm35-edit)</sub> | - |
| Qwen3.8-Flash-Next IQ3_S (78 GB) | <sub>1k edit (best pp)</sub> | <sub>b11381 b12128</sub> | 376<br>**683**&nbsp;<sub>[(1.82x)](fork/bench/REPRO.md#qni3-edit)</sub> | 42.7<br>**54.2**&nbsp;<sub>[(1.27x)](fork/bench/REPRO.md#qni3-edit)</sub> | [**1.1113**](fork/bench/REPRO.md#qni3-edit)<br>1.1151 |
| Qwen3.6-35B-A3B hybrid (fits VRAM) | <sub>game, 600 out</sub> | <sub>b11381 b12824</sub> | 85.0<br>**145**&nbsp;<sub>[(1.71x)](fork/bench/REPRO.md#q36-game)</sub> | 153<br>**155**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#q36-game)</sub> | [**1.5242**](fork/bench/REPRO.md#q36-game)<br>1.5252 |
| GLM-5.3-Flash 3.5-bit | <sub>game (best t/s)</sub> | <sub>b11381 b11707</sub> | 3.7<br>**5.4**&nbsp;<sub>[(1.46x)](fork/bench/REPRO.md#glm35-game)</sub> | 6.9<br>**10.5**&nbsp;<sub>[(1.52x)](fork/bench/REPRO.md#glm35-game)</sub> | - |
| OLMoE-1B-7B Q4_K_M (fits VRAM) | <sub>game, 600 out</sub> | <sub>b11574 b12824</sub> | 213<br>**315**&nbsp;<sub>[(1.48x)](fork/bench/REPRO.md#olmoe-game)</sub> | 466<br>**472**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#olmoe-game)</sub> | 2.1489<br>[**2.1482**](fork/bench/REPRO.md#olmoe-game) |
| Qwen3.8-Flash-Next IQ3_S | <sub>game (best t/s)</sub> | <sub>b11381 b11707</sub> | 26.5<br>**29.9**&nbsp;<sub>[(1.13x)](fork/bench/REPRO.md#qni3-game)</sub> | 42.1<br>**56.2**&nbsp;<sub>[(1.33x)](fork/bench/REPRO.md#qni3-game)</sub> | - |
| Qwen3.8-Flash-Next IQ1_M | <sub>1k edit</sub> | <sub>b11381 b12128</sub> | 1165<br>**1409**&nbsp;<sub>[(1.21x)](fork/bench/REPRO.md#qn1-edit)</sub> | 65.0<br>**65.9**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#qn1-edit)</sub> | 1.1741<br>missing |
| Qwen3.8-27B MTP Q5_K_M (dense, fits 24 GB), both with MTP | <sub>1k edit</sub> | <sub>b11381 b12193</sub> | 1531<br>**1616**&nbsp;<sub>[(1.06x)](fork/bench/REPRO.md#q27mtp-edit)</sub> | 62.5<br>**64.4**&nbsp;<sub>[(1.03x)](fork/bench/REPRO.md#q27mtp-edit)</sub> | - |
| Qwen3.8-27B MTP Q5_K_M, both with MTP | <sub>game, 600 out</sub> | <sub>b11381 b12193</sub> | 32.4<br>**34.1**&nbsp;<sub>[(1.05x)](fork/bench/REPRO.md#q27mtp-game)</sub> | 74.9<br>**65.7**&nbsp;<sub>(0.88x)</sub> | - |
| Qwen3.8-27B IQ4_NL (dense, fits 24 GB) | <sub>1k edit</sub> | <sub>b11381 b11707</sub> | 1714<br>**1730**&nbsp;<sub>[(1.01x)](fork/bench/REPRO.md#q27nl-edit)</sub> | 44.3<br>**44.3**&nbsp;<sub>(1.00x)</sub> | - |

\* `pplb` (batched perplexity, 1024-token batches over the same text) because the 1-token `ppl` took hours on this model.

On smaller cards the gain grows, see the next section (an 8 GB laptop: decode 2.7-3.6x; a simulated 16 GB card: GLM 6.5x decode, Qwen3.8-Flash-Next 3.4x on two cards).

**Q2_0 rows:** the 2-bit Qwen3.6 file (`qwen36.gguf`) is no longer on any of our machines, so these two rows are the last measurement of 2026-09-30 (build b11648); the AVX2 Q2_0 dot product is in b12844 and in the merged tree, but it has not been re-measured since.

**Decode right after a long prompt (GLM-5.3-Flash 3.0-bit, 12k prompt):** the decode rates in the table above are the average of the first 32 generated tokens, while the pool is still refilling the blocks the prompt borrowed. How the rate develops over a long answer after a long prompt has not been measured yet (a run with 128, 512 and 2,000 output tokens on b12686, b12825 and b12844 is queued). What is measured: with a short prompt and a 600-token answer (cache warm, nothing to refill) b12825 decodes at 15.6 t/s against 18.5 t/s for b12686, so part of the slowdown is not the refill.

**Known issue in b12844 (found while updating this table):** after a 12k prompt, Qwen3.8-Flash-Next decodes at 35 t/s where b12686 gives 50 t/s, and GLM-5.3-Flash at 14.3 vs 16.2 t/s (the prompt itself is 1.3-1.4x faster). Not fixed yet; the prefill changes of this release are the suspected cause, a fix is next.

**Results, 2x RTX 3090 (one AM4 CPU PCIe 4.0 x16, one X570 chipset x4 slot) + Ryzen 7 3700X + 125 GB DDR4,
CPU frequency governor `performance`, single stream, temp 0.** Short = 1500-token chat reply to "write smallest html tetris game". Long =
12k-token code prompt (llama.cpp sources): prompt processing, then decode. The prompts and a script to
reproduce these tests are in [`fork/bench/`](fork/bench/).

## Release "parallel-prefill": small cards first (2026-10-14)

Each build runs on **its own defaults**: only `-np 1` (and `-c` for stock, whose 4096 default rejects long prompts). No thread, batch or `--moe` settings on either side.
`prompt` = prompt processing t/s, `gen` = generation t/s. Single stream, temperature 0. Multipliers are fork / stock.

**An 8 GB laptop** (RTX 4060 Laptop 8 GB, 31 GB RAM, Windows 10, `llama-cli`). Stock = upstream b11514 (`de7fa0a3c`), the newest upstream binary tried on this machine.
Qwen3.6-35B-A3B-GSQ-hybrid is 11.4 GiB, so it does not fit the card; stock crashes or crawls here, the fork picks its own placement and micro-batch.

| model, prompt | stock prompt / gen | fork prompt / gen | prompt | gen |
|---|---|---|---|---|
| Qwen3.6-35B-A3B 11.4 GiB, 17 tokens | 20.2 / 15.6 | 98.0 / 56.5 | **4.9x** | **3.6x** |
| Qwen3.6-35B-A3B, 12k tokens | 687 / 13.6 | 918 / 45.6 | 1.34x | **3.4x** |
| Qwen3.6-35B-A3B, 40k tokens | 584 / 11.7 | 726 / 35.9 | 1.24x | **3.1x** |
| Qwen3.6-35B-A3B, 125k tokens | 448 / 9.2 | 557 / 25.2 | 1.24x | **2.7x** |
| Qwen3.8-27B IQ3_S 12.1 GB (dense), 10 tokens | 19.0 / 3.0 | 17.4 / 3.1 | 0.92x ‡ | 1.03x |
| Qwen3.8-27B IQ3_S (dense), 12k tokens | 270 / 2.2 | 258 / 2.2 | 0.96x ‡ | 1.00x |

‡ Dense models get no gain from the fork on a small card (no experts to cache; the layers that do not fit go through the CPU as in stock).
The 4-8% are inside this laptop's run-to-run noise (stock itself measured 270, 312, 313 and 337 on the 12k cell), the 27B is being re-measured with 3+ runs per build.

**A 16 GB card** (RTX 3090 limited to 16 GB with a memory holder, `-dev CUDA1`, x16 link) and **two 16 GB cards** (x16 + x4), Ryzen 7 3700X, 125 GB RAM. Stock = upstream b11574.

| model (size) | cell | cards | stock gen | fork gen | gen | stock prompt | fork prompt | prompt |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit (106 GB) | game: 600 tokens | 1 x 16 GB | 1.7 | 11.2 | **6.5x** | 1.2 | 13.8 | **11x** |
| Qwen3.8-Flash-Next | 4k prompt | 1 x 16 GB | 24.8 | 33.5 | 1.35x | 159 | 1214 | **7.6x** |
| Qwen3.8-Flash-Next | game | 2 x 16 GB | 15.0 | 51.1 | **3.4x** | 8.4 | 78.6 | **9.4x** |
| Qwen3.8-Flash-Next | 4k prompt | 2 x 16 GB | 16.5 | 29.1 | 1.76x | 72 | 1067 | **15x** |
| Qwen3.6-35B-A3B hybrid (fits) | game | 1 x 16 GB | 150.0 | 144.1 | 0.96x ¶ | 121 | 161 | 1.33x |
| Qwen3.6-35B-A3B hybrid (fits) | 4k prompt | 1 x 16 GB | 141.1 | 135.3 | 0.96x ¶ | 3009 | 2622 | 0.87x ¶ |
| Qwen3.6-35B-A3B hybrid (fits) | 4k prompt | 2 x 16 GB | 142.3 | 142.2 | 1.00x | 3370 | 3137 | 0.93x ¶ |
| OLMoE-1B-7B Q4_K_M (fits) | game | 1 x 16 GB | 460.9 | 454.1 | 0.99x | 213 | 322 | 1.5x |

¶ **Known gap:** a model that fits entirely in VRAM runs 0-4% slower on decode and 7-13% slower on prompt than stock (fork overhead on a path that has nothing to cache or to split).
The fix is queued: models that fit leave every fork hook, so the path is stock's.

**CPU only** (Ryzen, 12 cores, 60 GB RAM, clang, no GPU, `-dev none`, 3 runs each, each build on its defaults). Stock = upstream master 37d6d653e (b11631).
There is nothing to cache or to split without a GPU. What the CPU-only build adds is the kernels: **a 2-bit Q2_0 model runs 1.8x faster on decode and 3.7x faster on prompt than upstream (top table), because upstream has only a scalar dot product for that type on x86**; for the other quantizations the gain is the thread and batch tuning (+13 to +18% prompt, +5% decode on the 35B hybrid below). the CPU archives of the release are the same binaries without the CUDA backend.

| model, cell | builds stock fork | pp: stock / **fork** | t/s: stock / **fork** |
|---|---|---|---|
| Qwen3.6-35B-A3B hybrid (12 GB), short prompt | <sub>b11631 b12844</sub> | 32.1<br>**36.2**&nbsp;<sub>[(1.13x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.7<br>**11.3**&nbsp;<sub>[(1.06x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid (12 GB), 1k edit | <sub>b11631 b12844</sub> | 41.9<br>**49.3**&nbsp;<sub>[(1.18x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.4<br>**10.9**&nbsp;<sub>[(1.05x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid, short prompt, merged tree | <sub>b11631 b13086</sub> | 32.1<br>31.6&nbsp;<sub>(0.98x)</sub> | 10.7<br>**11.1**&nbsp;<sub>[(1.04x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| Qwen3.6-35B-A3B hybrid, 1k edit, merged tree | <sub>b11631 b13086</sub> | 41.9<br>**46.0**&nbsp;<sub>[(1.10x)](fork/bench/REPRO.md#q36-cpu)</sub> | 10.4<br>**10.8**&nbsp;<sub>[(1.04x)](fork/bench/REPRO.md#q36-cpu)</sub> |
| OLMoE Q4_K_M, short prompt | <sub>b11631 b13086 (merged tree)</sub> | 144<br>**145** | 47.6<br>**48.0** |
| OLMoE Q4_K_M, 1k edit | <sub>b11631 b12844 (release)</sub> | 198<br>**184**&nbsp;<sub>(0.93x)</sub> | 36.7<br>**36.1**&nbsp;<sub>(0.98x)</sub> |
| OLMoE Q4_K_M, 1k edit | <sub>b11631 b13086 (merged tree)</sub> | 198<br>**196** | 36.7<br>**33.5**&nbsp;<sub>(0.91x)</sub> |
| Qwen3.8-27B IQ3_S, short prompt | <sub>b11631 b13086 (merged tree)</sub> | 5.8<br>**6.0** | 1.7<br>**1.8** |
| Qwen3.8-27B IQ3_S, 1k edit | <sub>b11631 b13086 (merged tree)</sub> | 10.2<br>**10.3** | 1.7<br>**1.8** |

Two open items from this table: the release is 7% slower on prompt on CPU, and the merged tree is 9% slower on decode after a 1k prompt (neither `autotune=0` nor `--fork off` changes it).

**Fixed in this release:** a clean start on small cards. The previous release crashed with "CUDA error: out of memory" at the first kernel on an 8 GB laptop for models larger than VRAM (the cache pool was sized
from the whole model, then the compute buffer of a context-sized micro-batch did not fit). Now the pool is counted once, a 512 MiB / 8% margin is kept on cards of 12 GiB and less, the scheduler no longer reserves 12 GB of
split bookkeeping at a 14k-token micro-batch (which Windows refused), and a start that still runs out of memory restarts itself with half the micro-batch.

**Stock llama.cpp vs this fork**, tokens/s, model already in RAM (*). "mmap": weights
memory-mapped (models bigger than RAM, `llama-cli`); "pinned": weights in pinned RAM, what
`llama-server` does by itself when the model fits in RAM; "+ MTP": plus the model's MTP draft head
(`-md`, automatic draft depth). Multipliers vs stock; best number per row in bold (links to how to reproduce it).

| model | test | stock (b11041) | fork, mmap (b11327-b11367) | fork, mmap + MTP (b11367) | fork, pinned (b11341) | fork, pinned + MTP (b11367) |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash 3.0-bit [Q4_K attn](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF), 106 GB; `llama-server` default: pinned | short: decode | 13.8 | 20.4 (1.48x) | 18.2 (1.32x) ‡ | [**21.1 (1.53x)**](fork/bench/) | 18.2 (1.32x) ‡ |
| | long: prompt processing † | 217 | 180 (0.83x) | | [**262 (1.21x)**](fork/bench/) ||
| | long: decode | 12.3 | 15.6 (1.27x) | | [**16.3 (1.33x)**](fork/bench/) ||
| MiMo-V2.6-Flash-RL IQ3_XXS, 132 GB; `llama-server` default: mmap (bigger than RAM) | short: decode | 4.0 | [**10.1 (2.54x)**](fork/bench/) | no gain ¶ | - (bigger than RAM) | - |
| | long: prompt processing † | [**136**](fork/bench/) | 133 (0.98x) | | - | - |
| | long: decode | 4.2 | [**8.3 (1.98x)**](fork/bench/) | | - | - |
| Qwen3.8-Flash-Next UD-IQ4_XS, 88 GB; `llama-server` default: pinned | short: decode | 27.7 | 47.1 (1.70x) | 58.3 (2.11x) | 46.5 (1.68x) | [**60.4 (2.18x)**](fork/bench/) |
| | long: prompt processing † | 500 | 372 (0.74x) | 353 (0.71x) | [**538 (1.08x)**](fork/bench/) | 500 (1.00x) |
| | long: decode | 25.3 | 38.4 (1.52x) | 40.1 (1.59x) § | [**42.3 (1.67x)**](fork/bench/) | 28.0 (1.11x) § |
| Qwen3.8-27B IQ4_NL (dense, fits VRAM), 16 GB | short / long prompt | 44.7 / 1726 | 44.8 / 1811 (no cache needed) | | ||
| OLMoE-1B-7B Q4_K_M (fits VRAM), 4 GB | short: decode | 504 | 504 (no cache needed) | | ||

¶ MiMo's built-in MTP (3 dense layers, no experts): the depth tuner keeps depth 0 (drafting was
not 5% faster in its probes), so no measurable gain; MiMo's numbers vary between sessions (8.9-10.1)
because the model is bigger than RAM.
§ The 12k test generates only 32 tokens: too few for the MTP depth tuner to finish a comparison,
so these decode numbers are mostly probing overhead and vary a lot (28-40 t/s); MTP pays off on
longer replies (short-prompt row).
\* Model already in RAM (OS page cache), as on a server after its first request. The
first run after switching to another large model is slower, once, while the file is
read from disk. MiMo (132 GB) can't fully stay in 125 GB RAM, so it always reads part of
the model from disk. Multipliers are vs stock.
† Prompt processing streams the experts over PCIe to one GPU, so the slot limits it (here a
PCIe 4.0 x16 CPU slot; the second card sits in an X570 chipset x4 slot). A board with more
x16 slots, or another GPU, should raise it; splitting prompt processing across both GPUs'
links is the next milestone.
‡ MTP on GLM-5.3-Flash is slower on this box (pinned: 18.2 vs 21.1 t/s without; mmap: 18.2 vs 20.4; tuner picks depth 1; the model is 2.3x
the VRAM, so verifying drafts adds CPU work and the draft takes cache VRAM). The fork
doesn't load the draft here by default. Another GPU (more VRAM) should make it pay off,
as it does for Qwen (1.9x VRAM).

stock llama.cpp -> this fork. The fork numbers match a plain `llama-server -m model` within ~4%
(auto mode, see Run). Stock can't load GLM-5.3-Flash, so its stock column is
this fork without the cache. Models that fit in VRAM don't use the cache and run the
same (identical output). With the default `schedutil`/`powersave` governor, decode can
be lower, mostly where the CPU computes missed experts. Prompt processing: stock keeps whole layers in VRAM and
never uploads them, this fork uploads experts, so it depends on upload speed. With pinned
weights (the server default when the model fits in RAM) GLM-5.3-Flash is 1.21x stock, Qwen
1.08x; with mmap both are 17-26% below stock.
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

