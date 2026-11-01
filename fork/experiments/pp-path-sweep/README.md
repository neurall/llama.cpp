# Prompt pass time by pass size and expert path (2026-10-31)

Pure prompt time (`--moe pp-only=1`, one generated token, nothing of the generation part runs) of one prompt of N tokens (real text, tokenised by the
server, `cache_prompt` off), median of several runs after one discarded run. Branch `predictor-upload-bw-reducer` (early router, `prefill-cpu`, `tokheat`).
Two RTX 3090 (CUDA1 x16 about 25 GB/s, CUDA0 x4 about 6 GB/s), `-c 20992`, 12 threads, `--ctx-checkpoints 0`.

| config | `--moe` options | env |
|---|---|---|
| upload | `max-batch=1` | |
| k8 | `max-batch=1` | `GGML_PRED_K=8` |
| k32 | `max-batch=1` | `GGML_PRED_K=32` |
| cache | `max-batch=32768,prefill-cpu=1` (forced cache path, misses computed on the CPU at every size) | |
| dense | `max-batch=1,dense-split=1,dense-split-min=1` | |
| dense_moe | `max-batch=1,dense-split=1,dense-split-min=1,prefill-split=1` | |

Qwen3.6 runs also pass `sim-vram=4500` (restricted card memory, experts in RAM + cache); the larger models run unrestricted. Runs per size: 7 up to 200
tokens, 5 up to 1000, 3 up to 10000, 2 at 20000. Columns: see the header of each CSV (units in the names).
