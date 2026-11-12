# Performance history

Tables moved out of the README when the newest run table replaced them (nothing deleted). Newest first as they were in the README; the README keeps only the latest runs.

### Best gainers so far (measured 2026-10-14; 2x RTX 3090 24 GB with one x16 and one x4 link, and an 8 GB RTX 4060 laptop where marked)

`pp` = prompt processing tokens/s, `t/s` = generated tokens/s. Perplexity: lower is better; `missing` is a run that is not finished or was never made. A blue (linked) gain is a fork result that beats stock, and in the `ppl` column the lower of the two numbers is linked; click it for the exact command that reproduces the row (`fork/bench/repro.py ID`, listed in [REPRO.md](fork/bench/REPRO.md)). In every `pp`, `t/s` and `ppl` cell the stock value is on top and the fork value below it, with the gain in brackets. Every measured run of our builds passes `--load-mode pin` (pinned weights) from now on. Sorted by the larger of the two gains. Same prompts as below, each build on its own defaults (only `-np 1`, `-c` for stock). "stock" is the newest upstream build that was measured on that cell (build in the last column); the fork rows are the best measured build of this release line (builds are given by build number, the commit is in the release notes and `fork/bench`; the published b12844 is being re-measured the same way, see the known-issue note).

<!--latest-->
## Latest measurements: build b13180 against latest upstream master (2026-10-15)

Each build on its own defaults (auto against auto), `-np 1` only, 3 runs per cell, no warm-up run. Stock over ours, gain in brackets. **i** = long prompt, prompt speed; **io** = 1k in and 1k out, prompt speed and token speed; **o** = short prompt, long answer, token speed. Older rows below are earlier measurements and stay as they were.  

| Model | i: prompt t/s | io: prompt t/s | io: tokens t/s | o: tokens t/s |
|---|---|---|---|---|
| MiMo-V2.6-Flash 3-bit | 54<br>**1074**&nbsp;<sub>(19.81x)</sub> | 48<br>**587**&nbsp;<sub>(12.29x)</sub> | 14.1<br>**33.7**&nbsp;<sub>(2.39x)</sub> | 14.1<br>**36.6**&nbsp;<sub>(2.60x)</sub> |
| GLM-5.3-Flash 3.0-bit | 43<br>**660**&nbsp;<sub>(15.43x)</sub> | 40<br>**393**&nbsp;<sub>(9.94x)</sub> | 12.5<br>**22.9**&nbsp;<sub>(1.83x)</sub> | 12.6<br>**22.5**&nbsp;<sub>(1.78x)</sub> |
| Qwen3.8-Flash-Next UD-IQ4_XS | 212<br>**1376**&nbsp;<sub>(6.50x)</sub> | 201<br>**801**&nbsp;<sub>(3.99x)</sub> | 30.7<br>**28.7**&nbsp;<sub>(0.93x)</sub> | missing |
| Qwen3.8-Flash-Next IQ3_S (78 GB) | 359<br>**1410**&nbsp;<sub>(3.92x)</sub> | 359<br>**643**&nbsp;<sub>(1.79x)</sub> | 42.5<br>**30.7**&nbsp;<sub>(0.72x)</sub> | 44.4<br>**40.4**&nbsp;<sub>(0.91x)</sub> |
| Qwen3.8-27B MTP Q5_K_M (both with MTP) | 1573<br>**1131**&nbsp;<sub>(0.72x)</sub> | 1378<br>**1040**&nbsp;<sub>(0.75x)</sub> | 34.5<br>**45.5**&nbsp;<sub>(1.32x)</sub> | 35.2<br>**32.3**&nbsp;<sub>(0.92x)</sub> |
| OLMoE-1B-7B Q4_K_M (fits VRAM) | missing | 13653<br>**15097**&nbsp;<sub>(1.11x)</sub> | 359.9<br>**359.1**&nbsp;<sub>(1.00x)</sub> | 486.6<br>**498.2**&nbsp;<sub>(1.02x)</sub> |
| Qwen3.6-35B-A3B hybrid (fits VRAM) | 5257<br>**4727**&nbsp;<sub>(0.90x)</sub> | 3762<br>**3445**&nbsp;<sub>(0.92x)</sub> | 146.6<br>**148.6**&nbsp;<sub>(1.01x)</sub> | 150.4<br>**152.4**&nbsp;<sub>(1.01x)</sub> |
| Qwen3.8-Flash-Next IQ1_M | 1247<br>**915**&nbsp;<sub>(0.73x)</sub> | 1212<br>**618**&nbsp;<sub>(0.51x)</sub> | 61.2<br>**60.9**&nbsp;<sub>(1.00x)</sub> | 62.3<br>**62.7**&nbsp;<sub>(1.01x)</sub> |
| Qwen3.8-27B IQ4_NL (dense) | 1748<br>**1562**&nbsp;<sub>(0.89x)</sub> | 1482<br>**1340**&nbsp;<sub>(0.90x)</sub> | 42.4<br>**42.5**&nbsp;<sub>(1.00x)</sub> | 43.3<br>**43.3**&nbsp;<sub>(1.00x)</sub> |
| GLM-5.3-Flash 3.5-bit (137 GB) | missing | missing | missing | missing |

<!--/latest-->
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

## Moved from the README on 2026-11-12 (measured on the previous machine: Ryzen 7 3700X, 128 GB DDR4-3200 dual channel, RTX 3090 on x16 + RTX 3090 on x4)

### Status note of 2026-10-23

Status Update:23:10:2026. Right now multi gpu decode is supported all models multigpu dense split 
prefill is implemented for glm and gives gains only from prompts 7k up on 2x3090 at least on my rig.

In other words tha gains you see bellow are not from multigpu yet just from more efficient vram mmgmt
ie the same 20x speedup in mimo is happening on single 5090 too. In a sense its good news so people with one gpu will have fun too ;)
And that multigpu fun is just starting too. For glm 2nd gpu is adding +30%  on pp so far

**Same VRAM, different use.** Each GLM token uses 8 of 288 experts per layer. Stock
fills VRAM with whole layers (~14 of 42), mostly experts the current token doesn't
touch, so the CPU does most expert work and the GPUs wait. This fork fills the same
VRAM with the ~100 most-used experts of every layer, so most expert work runs on the
GPUs, at the same time as the CPU handles the misses.

### Other notable changes: the two-card prompt split


**Where the two-card dense split starts to pay.** Per prompt, the planner measures one card and two cards and takes the faster; it does not use a fixed size. The medians of all runs since 2026-10-20 (220 + 280 W, answers not looped; each cell has 2 to 50 runs from several builds, so differences under 10% are noise) show where it wins:

| model | prompt | one card (pp t/s) | two cards (pp t/s) | official stock (pp t/s, build) | one card vs stock | two cards vs stock | best of our builds |
|---|---|---|---|---|---|---|---|
| Qwen3.8 Next IQ4_XS | 4k | 1684 | 1350 | 209, b11887 | 8.1x | 6.5x | pre7 |
|  | 8k | 1663 | 1513 | 213 (9k prompt), b11829 | 7.8x | 7.1x | pre9 |
|  | 11k | 1581 | 1824 | 211 (10k prompt), b11829 | 7.5x | 8.6x | rc5 |
|  | 12k | 1498 | 1648 | 213, b11829 | 7.0x | 7.7x | spass15 |
|  | 16k | 1543 | 1115 | 208, b11887 | 7.4x | 5.4x | pre9 |
|  | 30k | 1553 | 1554 | fail: context too small | - | - | ncard6 |
|  | 60k | 1600 | 1668 | fail: context too small | - | - | pre10 |
| Qwen3.8 Next IQ3_S | 12k | 1443 | - | 379, b11829 | 3.8x | - | relfx1 |
|  | 60k | 1546 | 1864 | fail: context too small | - | - | um6g |
| GLM-5.3-Flash 3.0-bit | 4k | 703 | 639 | 42 (5k prompt), b11829 | 16.7x | 15.2x | pre9 |
|  | 12k | 658 | 609 | 43, b11829 | 15.3x | 14.2x | ceil1 |
|  | 16k | 660 | 775 | 42, b11887 | 15.7x | 18.4x | pre10 |
|  | 30k | 557 | 686 | 42, b11887 | 13.3x | 16.3x | ncard6 |
|  | 60k | 566 | 694 | - | - | - | pre10 |
| MiMo-V2.6-Flash 3-bit | 4k | 905 | 903 | 48 (3.4k prompt), b11866 | 18.9x | 18.8x | pre13 |
|  | 8k | 1062 | 999 | 50, b11866 | 21.2x | 20.0x | relmf1 |
|  | 12k | 988 | 1150 (1 run) | 49, b11829 | 20.2x | 23.5x | rc5 |
|  | 16k | 906 | 1094 | 49, b11887 | 18.5x | 22.3x | pre9 |
|  | 60k | 846 | 841 | - | - | - | pre12 |

Stock cells: official ggml-org binaries with their own defaults (`-np 1` only, micro-batch and context auto); where the stock prompt size differs from the row it is named. **fail: context too small** = stock sizes its context itself (32768 tokens at 30k, 63488 at 60k on Qwen) and refuses a prompt that does not fit (a 30k prompt is 33.6k tokens, a 60k prompt 68.3k): `request (68337 tokens) exceeds the available context size (63488 tokens)`. It needs `-c` set by hand, which was not done here, so stock cannot serve these prompts as shipped; ours sizes the context per request. Stock is CPU-bound, so its prompt speed is nearly the same at every size (GLM 42-43, MiMo 48-50, Qwen IQ4_XS 208-213). "-" = not measured (stock at 60k would take 20 minutes or more per cell on GLM and MiMo; GLM 30k took 12 minutes).

- **The split starts to pay at about 11k-16k prompt tokens**: Qwen3.8 Next from 11k, GLM and MiMo from 16k (MiMo possibly 12k, one run). Below about 8k one card is as fast or faster (the second card sits in a x4 slot and the hand-offs between cards cost more than the shorter pass saves). The auto setup takes one card there.
- **Largest gains: GLM +18% to +26% (16k to 60k), MiMo +21% at 16k, Qwen3.8 Next +15% at 11k.** Qwen3.6 and OLMoE fit in VRAM and show no split gain.
- **Big contexts at full speed are one of the main benefits of this fork.** Stock sizes the context once at start from the free memory (32768 and 63488 tokens on the Qwen IQ4_XS and IQ3_S cells above); a longer prompt is refused. Ours claims the KV it needs per request from the same buffer as the experts and gives it back at the end of the request, so no `-c` has to be guessed and a big context costs nothing while it is not in use. Measured prompts, prompt speed (pp t/s), on this machine: Qwen3.8 Next IQ4_XS 60k 1600-1668 (stock: refused with its own context), Qwen3.6-35B 200k 3259 (decode 72.2 t/s), MiMo 200k 794 (decode 17.3 t/s) and 400k 363; the 200k cells with the antiloop sampler (repeat penalty 1.15) so the answers are real text, the speed is not touched by it. Details of the KV handling are in [fork/docs/features.md](fork/docs/features.md).
- The split is chosen by the planner and needs no switch; `--moe dense-split=0` forces one card, `=1` forces the split.


### What each feature contributes


Each row is one paired A/B on the same binary and settings, measured when the feature was added (2x RTX 3090 unless noted; older builds, one A/B each, so read them as the size of the effect, not as additive shares of the table above). Method and run notes per feature: [fork/docs/features.md](fork/docs/features.md).

| feature | test | without | with | change |
|---|---|---|---|---|
| Expert cache in free VRAM | decode, models that do not fit | stock (no expert cache) | the gainers table above | +20% to +130% decode |
| Pinned weights (server default) | GLM 12k prompt, prompt speed | 180 pp t/s | 262 pp t/s | +45% |
| | Qwen 3.8 Next 12k prompt, prompt speed | 372 pp t/s | 538 pp t/s | +45% |
| Big micro-batch (`-ub` 1024 to 4096) | GLM 12k prompt, prompt speed | 176.6 pp t/s | 370.1 pp t/s | 2.1x |
| Micro-batch buffer lent to the prompt pass only | GLM 12k prompt then decode, `-ub` 4096 | 17.81 t/s (static buffer) | 18.99 t/s | +6.6% decode, same prompt speed (368.9 vs 367.6) |
| Prefill preheat | GLM 12k prompt then 256 tokens, 1 GPU, decode | 13.5 t/s (cache hits 43%) | 15.3 t/s (55%) | +12.7% |
| Learned hot experts kept between runs | GLM first request after restart, decode | 20.5-20.9 t/s | 21.8 t/s | +4-6% |
| Global expert pool (vs fixed slots per layer) | decode with 20-35% of the experts in VRAM | 132.5-177.5 t/s | 137.3-181.7 t/s | +2.1% to +3.6% (no change when most of the model fits) |
| Dense split over two cards (forced path) | Qwen 3.8 Next 4k prompt, prompt speed, 180+220 W | 1065 and 1113 pp t/s (run ids 12190, 12191) | 1274 and 1275 pp t/s (12193, 12194) | +17% (held back by the second card sitting in a slow x4 slot, 6 GB/s against 13 GB/s on the x16 card; cards on equal links gain more) |

Which features drive which number: the **expert cache** is what lifts decode over stock; **pinned weights and the big micro-batch** (with its buffer lent only to the prompt pass, so decode keeps its cache) are what lift prompt speed. The rest are smaller, single-digit effects.

