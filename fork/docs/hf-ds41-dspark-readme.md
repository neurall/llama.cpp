---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
license: mit
tags:
  - gguf
  - llama.cpp
  - deepseek
  - moe
  - gsq
  - rco
  - dspark
  - speculative-decoding
---

# DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XXS for llama.cpp, with a DSpark draft

ISTA-DASLab's **GSQ-RCO IQ2_XXS** quantization of DeepSeek-V4.1-Flash-1017 (2.00 bits per weight, 248.8 GB), with the file layout and names changed so that **llama.cpp loads it**. The original files were made for the [Strata](https://github.com/Niko1221/Strata) engine and do not load in llama.cpp as they are.

Also here: the **DSpark draft** from Lucebox, changed the same way, for speculative decoding in [our llama.cpp fork](https://github.com/neurall/llama.cpp).

- Runs on **stock llama.cpp**, but slowly: about 1.5 tokens/s of answer on our test machine.
- Whole requests take **9.7x to 19x less time** in our fork, which keeps the experts in RAM and uses the free VRAM as a live expert cache: the least with a long prompt (16k prompt, 1k answer: 1097 s against 114 s), the most with a short one (27-token prompt, 1k answer: 689 s against 36 s).
- The DSpark draft adds about **8-10%** decode speed on top in our fork (measured, details below).

## Credits

All weights are the work of others. This repo changes only file headers, tensor names and the file layout.

| what | by | link |
|---|---|---|
| base model | DeepSeek | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| GSQ-RCO quantization (IQ2_XXS) | IST Austria, DASLab: [GSQ](https://arxiv.org/abs/2604.18556) ([code](https://github.com/IST-DASLab/GSQ)), [RCO](https://arxiv.org/abs/2605.00649) ([code](https://github.com/IST-DASLab/RCO)) | [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) |
| engine the quantization was made for | Strata | [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) |
| DSpark draft (Q2_K / Q4_K) | Lucebox | [Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF) |

For quality numbers (AIME25, GPQA-Diamond, LiveCodeBench v6) see ISTA-DASLab's card: by their evaluation IQ2_XXS keeps 100.0% of the base model's task average. Our check: perplexity over 10 chunks equals stock llama.cpp's on the same files (-0.02% ± 0.13%).

## Files

| file | size | changed against the original |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00005.gguf` | 39.1 GB | metadata keys renamed to llama.cpp's `deepseek41` names; the small Engram helper tensors (token map, primes, offsets, multipliers) moved into metadata keys; 2 Engram gate tensors added, computed from the Engram q and k weights (their product) |
| `...-00002-of-00005.gguf` | 39.4 GB | header only (split count 4 to 5) |
| `...-00003-of-00005.gguf` | 39.2 GB | header only |
| `...-00004-of-00005.gguf` | 20.5 GB | Engram q and k tensors removed (replaced by the gate above), `engram_wkv` renamed to `engram_kv` |
| `...-00005-of-00005.gguf` | 110.6 GB | the original `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`: header only (architecture `deepseek41`, the two Engram tables named as one `per_layer_token_embd.weight`); the table data is byte for byte the original |
| `Lucebox-DSpark-draft-Q2K-Q4K.dflash.sp.gguf` | 4.9 GB | Lucebox's `DeepSeek-V4.1-Flash-DSpark-draft-Q2K-Q4K.gguf`: architecture renamed to llama.cpp's `dflash` draft with its key and tensor names, tokenizer and chat template added, DSV4.1 draft settings (target layers 37-39, block size 5), single-pass hyper-connection flag |

Keep all five shards in one directory and point llama.cpp at shard 1.

## Speed

Test machine: AMD Ryzen 7 3700X (8 cores), 128 GB RAM, 2x RTX 3090 24 GB (PCIe x16 and x4), Linux. The model (248.8 GB) is about twice the RAM, so part of it is read from the SSD.  
Same machine, same day, power limits 220/280 W, one request at a time, temperature 0. Each row is one run per build: the one with the shortest request of 2 (the fork's 16k: 1 run). Request time = the measured prompt time + the answer at the measured decode speed (1024 tokens; stock stopped the 16k answer at 857, its time is for 1024 at the same speed).

| test | stock request s | fork request s | stock decode t/s | fork decode t/s | stock prompt t/s | fork prompt t/s | request gain | decode gain | prompt gain |
|---|---|---|---|---|---|---|---|---|---|
| 27-token prompt (tetris), 987-token answer | 689 | 36 | 1.47 | 29.11 | 1.5 | 14.6 | 19.3x | 19.8x | 9.7x |
| 1k prompt (1099 tokens), 1k answer | 742 | 58 | 1.46 | 25.73 | 27.1 | 62.2 | 12.9x | 17.6x | 2.3x |
| 16k prompt (17575 tokens), 1k answer | 1097 | 114 | 1.47 | 18.16 | 43.9 | 307.4 | 9.7x | 12.4x | 7.0x |

The longer the prompt against the answer, the lower the gain: prompt processing gains less than answering.

Stock is the official llama.cpp release b11887 (`-np 1 -fitt 2048`). The fork ran with `-np 1` plus our test harness's load settings (memory-mapped load, at most 101 GiB of the model pinned in RAM).

### DSpark (our fork)

Tetris prompt, power limits 200/200 W, best of 2 runs:

| answer length | no draft t/s | DSpark depth 1 t/s | gain |
|---|---|---|---|
| 1k tokens | 28.32 | 31.29 | +10% |
| 4k tokens | 25.34 | 27.26 | +8% |

Depth 1 (one drafted token per step) was the most consistent. Deeper drafts varied: depth 3 reached 29.40 t/s once on the 4k answer, depth 2 was slower than no draft. Very short answers (256 tokens) do not gain.

## Usage

Our fork ([releases](https://github.com/neurall/llama.cpp/releases)), no tuning flags needed. Use a release **newer than b14020**: b14020 answers this model with empty text when run without `-c`, and has no DSpark fixes yet.

```
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00005.gguf -np 1
```

With the DSpark draft:

```
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00005.gguf -np 1 \
  -md Lucebox-DSpark-draft-Q2K-Q4K.dflash.sp.gguf \
  --spec-type draft-dspark --spec-draft-n-max 1 --moe spec-depth=1
```

Stock llama.cpp:

```
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00005.gguf -np 1 -fitt 2048
```

The model is chat-tuned: use the chat endpoint (`/v1/chat/completions`).

## License

MIT, as the base model, the GSQ-RCO quantization and the DSpark draft.
