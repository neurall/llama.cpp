# Kernel select grid (GGML_KERNEL_SELECT, CPU only, --fork off)

Value 0 = upstream dot product, 1 = ours. Text: longsrc.cpp, 1024-token passes, 2 chunks. Cell = perplexity (blank = not run yet).

Combos (U = upstream 0, O = ours 1), order: IQ3_S, IQ3_XXS, IQ1_S, IQ1_M

| combo | IQ3_S | IQ3_XXS | IQ1_S | IQ1_M | PC1 Ryzen 7 3700X | PC2 Ryzen 5 3600 | W10 |
|---|---|---|---|---|---|---|---|
| stock official binary | - | - | - | - | | | |
| all upstream | 0 | 0 | 0 | 0 | | | |
| IQ3_S ours | 1 | 0 | 0 | 0 | | | |
| IQ3_S + IQ3_XXS ours | 1 | 1 | 0 | 0 | | | |
| IQ1_S ours | 0 | 0 | 1 | 0 | | | |
| IQ1_M ours | 0 | 0 | 0 | 1 | | | |
| all ours (default) | 1 | 1 | 1 | 1 | | | |

Models per machine: PC1 Qwen3.8 Flash Next IQ3_S (84 GB) and OLMoE IQ3_S / IQ1_M / IQ1_S; PC2 and W10 OLMoE.
