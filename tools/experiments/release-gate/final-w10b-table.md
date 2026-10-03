
Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-0000  cli-t100  pc3
  stock-b11379                                         n= 3    12.30 t/s [12.00-12.90]  pp    17.7  x1.00 stock
  b11707-49fe4b756 default                             n= 3    16.10 t/s [15.10-16.10]  pp    19.0  x1.31 stock
  final-win default                                    n= 3    16.00 t/s [15.80-16.80]  pp    21.1  x1.30 stock

qwen36.gguf  cli-t100  pc3
  stock-b11379                                         n= 3    30.80 t/s [30.70-30.90]  pp    41.0  x1.00 stock
  b11707-49fe4b756 default                             n= 3    57.00 t/s [50.30-57.90]  pp   101.3  x1.85 stock
  final-win default                                    n= 3    59.00 t/s [58.70-59.10]  pp   108.4  x1.92 stock

| model | test | machine | build and settings | n | median [min-max] t/s | vs stock | vs tuned stock | protocol, first run |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | stock-b11379 | 3 | 12.30 [12.00-12.90] |  |  | v1 2026-10-04 00:23 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | b11707-49fe4b756 default | 3 | 16.10 [15.10-16.10] | 1.31x |  | v1 2026-10-04 00:24 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | final-win default | 3 | 16.00 [15.80-16.80] | 1.30x |  | v1 2026-10-04 00:25 +2 |
| qwen36.gguf | cli-t100 | pc3 | stock-b11379 | 3 | 30.80 [30.70-30.90] |  |  | v1 2026-10-04 00:20 +2 |
| qwen36.gguf | cli-t100 | pc3 | b11707-49fe4b756 default | 3 | 57.00 [50.30-57.90] | 1.85x |  | v1 2026-10-04 00:20 +2 |
| qwen36.gguf | cli-t100 | pc3 | final-win default | 3 | 59.00 [58.70-59.10] | 1.92x |  | v1 2026-10-04 00:21 +2 |
