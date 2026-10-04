
GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf  chat  pc1
  stock-docker-836d57176                               n= 1    12.50 t/s [12.50-12.50]  pp    15.6  x1.00 stock  n<3
  release-ac2e8344b default                            n= 4    20.41 t/s [20.16-20.87]  pp    15.2  x1.63 stock  md5 same 1/4

GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf  t100  pc1
  release-ac2e8344b default                            n= 3    24.94 t/s [12.88-26.62]  pp    12.2    md5 same 1/3

GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf  t100  pc1
  release-ac2e8344b default                            n= 3    10.57 t/s [4.16-11.11]  pp     3.6    md5 same 1/3

MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf  chat  pc1
  b11707-49fe4b756 default                             n= 3    10.26 t/s [9.81-11.44]  pp     3.4    n_gen differs, md5 same 1/3
  release-ac2e8344b default                            n= 3    10.60 t/s [10.53-10.61]  pp     2.6    n_gen differs, md5 same 1/3

MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf  pf12k  pc1
  b11707-49fe4b756 default                             n= 3     8.96 t/s [8.95-8.97]  pp   117.8  
  release-ac2e8344b default                            n= 3     8.90 t/s [8.80-8.91]  pp    57.0  

Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-0000  t100  pc1
  release-ac2e8344b default                            n= 3    68.00 t/s [38.56-68.04]  pp    39.1    md5 same 2/3

Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-0000  t100  pc1
  release-ac2e8344b default                            n= 3    41.73 t/s [32.91-41.92]  pp    29.0    md5 same 2/3

Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gg  chat  pc1
  stock-docker-836d57176                               n= 2    30.61 t/s [30.14-31.08]  pp    42.1  x1.00 stock  n<3
  b11707-49fe4b756 default                             n= 2    41.14 t/s [27.35-54.94]  pp    61.0  x1.34 stock  n<3, md5 same 1/2
  release-ac2e8344b default                            n= 2    52.79 t/s [52.55-53.03]  pp    63.3  x1.72 stock  n<3, md5 same 1/2

Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gg  pf12k  pc1
  stock-docker-836d57176                               n= 3    29.27 t/s [28.93-29.29]  pp   209.8  x1.00 stock
  b11707-49fe4b756 default                             n= 3    45.82 t/s [45.65-46.09]  pp   556.5  x1.57 stock
  release-ac2e8344b default                            n= 3    47.86 t/s [47.17-47.90]  pp   435.8  x1.64 stock

| model | test | machine | build and settings | n | median [min-max] t/s | vs stock | vs tuned stock | protocol, first run |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf | chat | pc1 | stock-docker-836d57176 | 1 | 12.50 [12.50-12.50] |  |  | v1 2026-10-04 01:36 +0 |
| GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf | chat | pc1 | release-ac2e8344b default | 4 | 20.41 [20.16-20.87] | 1.63x |  | v1 2026-10-04 02:03 +3 |
| GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf | t100 | pc1 | release-ac2e8344b default | 3 | 24.94 [12.88-26.62] |  |  | v1 2026-10-04 04:06 +2 |
| GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf | t100 | pc1 | release-ac2e8344b default | 3 | 10.57 [4.16-11.11] |  |  | v1 2026-10-04 03:47 +2 |
| MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-0000 | chat | pc1 | b11707-49fe4b756 default | 3 | 10.26 [9.81-11.44] |  |  | v1 2026-10-04 02:56 +2 |
| MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-0000 | chat | pc1 | release-ac2e8344b default | 3 | 10.60 [10.53-10.61] |  |  | v1 2026-10-04 03:00 +2 |
| MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-0000 | pf12k | pc1 | b11707-49fe4b756 default | 3 | 8.96 [8.95-8.97] |  |  | v1 2026-10-04 03:19 +2 |
| MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-0000 | pf12k | pc1 | release-ac2e8344b default | 3 | 8.90 [8.80-8.91] |  |  | v1 2026-10-04 03:28 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | t100 | pc1 | release-ac2e8344b default | 3 | 68.00 [38.56-68.04] |  |  | v1 2026-10-04 02:05 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-o | t100 | pc1 | release-ac2e8344b default | 3 | 41.73 [32.91-41.92] |  |  | v1 2026-10-04 02:06 +2 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | chat | pc1 | stock-docker-836d57176 | 2 | 30.61 [30.14-31.08] |  |  | v1 2026-10-04 02:10 +1 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | chat | pc1 | b11707-49fe4b756 default | 2 | 41.14 [27.35-54.94] | 1.34x |  | v1 2026-10-04 02:13 +1 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | chat | pc1 | release-ac2e8344b default | 2 | 52.79 [52.55-53.03] | 1.72x |  | v1 2026-10-04 02:15 +1 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | pf12k | pc1 | stock-docker-836d57176 | 3 | 29.27 [28.93-29.29] |  |  | v1 2026-10-04 02:35 +2 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | pf12k | pc1 | b11707-49fe4b756 default | 3 | 45.82 [45.65-46.09] | 1.57x |  | v1 2026-10-04 02:38 +2 |
| Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00 | pf12k | pc1 | release-ac2e8344b default | 3 | 47.86 [47.17-47.90] | 1.64x |  | v1 2026-10-04 02:41 +2 |
