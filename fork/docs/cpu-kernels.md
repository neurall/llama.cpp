# CPU dot-product kernels: where the speed is

Measured 2026-10-16 on an AMD Ryzen 5 3600 (Zen 2, AVX2, clang 21, dual-channel DDR4-3200), one thread unless noted. Tools: `fork/tests/dotbench.cpp`
(speed), `dotcheck.cpp` (accuracy against a double-precision reference), `tilebench.cpp` (weight reuse across tokens), `dotmt.cpp` (threads).  
The figure that matters for a kernel is **in-cache nanoseconds per 256 weights** (working set 0.25 MB: no memory limit); the RAM-bound figure is a property of the machine.

| Type | ns / 256 w in cache | ns / 256 w from RAM (96 MB) | Note |
|---|---|---|---|
| Q2_K | 6.4 | 7.7 | |
| Q4_K | 7.7 | 10.7 | |
| Q6_K | 7.8 | 14.4 | |
| Q3_K | 9.0 | 10.6 | |
| Q5_K | 9.7 | 13.2 | |
| Q8_0 | 10.1 -> **8.4** | 17.7 | four independent float accumulators (this branch): 19% faster in cache, sums differ in the last float bits only |
| IQ4_XS | 10.1 | 12.3 | |
| MXFP4 | 10.4 | 12.5 | |
| IQ4_NL | 10.5 | 13.0 | one 256-bit table lookup + four accumulators: 10.70 -> 10.66, **no gain, reverted** |
| Q4_0 | 11.0 | 13.8 | |
| IQ2_S | 17.5 | 18.7 | scalar grid loads |
| IQ2_XXS | 19.1 | 19.6 | scalar grid + sign loads |
| IQ2_XS | 23.6 | 24.0 | |
| IQ3_XXS | 25.6 | 26.6 | |
| IQ3_S | 40.0 -> **29.2** | 30.3 | contributor's index computation (on by default since 2026-10-16) |

* Aggregate throughput with 4-6 threads hits about 40 GB/s on this machine (its RAM limit) for every common type; on a machine with more bandwidth
  (DDR5, workstation, server) the per-thread compute figure above is what limits, so kernel speed matters there.
* Batches already reuse weights: the MUL_MAT_ID tile (16 weight rows x 16 tokens, `ggml-cpu.c`) keeps the weight rows in L1/L2 across the tokens.
  Q4_K: 169 ns per (row x token) at 1 token (RAM-bound), 134 at 4, 126 at 16: with 4+ tokens the kernel is compute-bound again, so a faster kernel
  helps most for CPU prompt processing and parallel requests. The tile size (4..32 rows) makes no difference.
* The IQ2/IQ3 family is limited by scalar table loads (about 24 loads per 64 weights for IQ2_XXS against two load ports: a floor near 13 ns per 256 weights,
  1.4x better than now). More needs gather instructions (cheap on recent Intel / Zen 4) or AVX-512 permutes in the ISA-specific backends.
* Not worth it: compiler unrolling (`-funroll-loops`: no change for any type).

## Zen 4 (Ryzen 9 8945HS laptop, Windows, clang 20, native build; 2026-10-16)

In cache, ns per 256 weights, Zen 2 -> Zen 4: Q2_K 6.4 -> 4.5, Q3_K 9.0 -> 6.8, **Q4_K 7.7 -> 9.3 (slower)**, Q5_K 9.6 -> 6.7, Q6_K 7.8 -> 5.3, Q8_0 8.4 -> 5.7, Q4_0 11.0 -> 7.4,
IQ4_NL 10.5 -> 8.4, IQ4_XS 10.1 -> 6.9, **MXFP4 10.4 -> 12.3 (slower)**, IQ3_S 29 -> 21, IQ3_XXS 25.6 -> 16.7, IQ2_S 17.5 -> 13.5, IQ2_XS 23.6 -> 18.5, IQ2_XXS 19.1 -> 17.6, Q2_0 12.7 -> 8.0.
Single-thread memory read 40.6 GB/s (Zen 2 box: 24). No x86 dot product has an AVX-512 path; VNNI (`dpbusd`) is reached only through the shared helper used by Q4_0/Q4_1/Q5_x/Q8_0/MXFP4/NVFP4/IQ4_NL/IQ4_XS/IQ1_x;
the K-quants and the IQ2/IQ3 kernels call `maddubs` + `madd` directly. Q4_K and MXFP4 are slower on Zen 4 than on Zen 2: the first targets (compiler version differs, to be checked).

## IQ3_XXS ablation (Zen 2, 2026-10-16; fork/tests/iq3xxs_ablation.cpp)

Where the time of the AVX2 IQ3_XXS dot product goes, ns per 256 weights in cache (same on a Ryzen 7 3700X and a Ryzen 5 3600):

| Variant | ns | Share |
|---|---|---|
| original (set_epi32 lookups + 8-byte sign table) | 25.2-26.6 | 100% |
| lookups through a stack array | 25.0-26.3 | no change |
| sign step removed | 18.2-18.9 | sign = 28% |
| lookups removed (constant grid) | 13.7-14.4 | lookups = 46% |
| sign by byte table + broadcast + shuffle (IQ3_S style) | 27.2-27.8 | slower, same sums |

Lookups already cost about 0.66 cycle each (about one insert per cycle), so AVX2 has little left there. Two AVX-512 attempts that cut the number of loads (IQ3_S gather, IQ2_XXS scalar lookups + mask sign + vpdpbusd) were slower on Zen 4. A real gain needs a different data flow (vpermb register lookups on AVX-512 VBMI CPUs, or decoding the 3-bit grid fields), not fewer loads.
