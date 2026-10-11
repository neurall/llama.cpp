
Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-0000  cli-t100  pc3
  stock-b11379                                         n= 3    13.40 t/s [13.40-13.60]  pp    18.0  x1.00 stock
  b11707-49fe4b756 default                             n= 3    16.50 t/s [16.40-16.80]  pp    20.3  x1.23 stock
  final-win default                                    n= 3    14.70 t/s [12.60-14.70]  pp    17.9  x1.10 stock

qwen36.gguf  cli-t100  pc3
  stock-b11379                                         n= 4    30.95 t/s [30.60-31.60]  pp    41.4  x1.00 stock
  b11707-49fe4b756 default                             n= 5    58.70 t/s [57.50-59.10]  pp   103.0  x1.90 stock
  final-win default                                    n= 5    59.00 t/s [58.70-61.20]  pp   108.5  x1.91 stock

| model | test | machine | build and settings | n | median [min-max] t/s | vs stock | vs tuned stock | protocol, first run |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | stock-b11379 | 3 | 13.40 [13.40-13.60] |  |  | v1 2026-10-04 00:08 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | b11707-49fe4b756 default | 3 | 16.50 [16.40-16.80] | 1.23x |  | v1 2026-10-04 00:08 +2 |
| Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-o | cli-t100 | pc3 | final-win default | 3 | 14.70 [12.60-14.70] | 1.10x |  | v1 2026-10-04 00:09 +2 |
| qwen36.gguf | cli-t100 | pc3 | stock-b11379 | 4 | 30.95 [30.60-31.60] |  |  | v1 2026-10-03 23:06 +3 |
| qwen36.gguf | cli-t100 | pc3 | b11707-49fe4b756 default | 5 | 58.70 [57.50-59.10] | 1.90x |  | v1 2026-10-03 23:06 +4 |
| qwen36.gguf | cli-t100 | pc3 | final-win default | 5 | 59.00 [58.70-61.20] | 1.91x |  | v1 2026-10-03 23:06 +4 |
