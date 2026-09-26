# Benchmark prompts for the MoE expert cache

The inputs behind the numbers in the main README, frozen so anyone can reproduce them.

- `src_12k.cpp`: 12,302 tokens of llama.cpp source (the "long" 12k test).
- `src_128k.cpp`: 125,257 tokens of llama.cpp source (the 128k test).
- `run.py`: sends a test to a running `llama-server` and prints prompt and decode speed.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf -c 131072   # -c only needed for 128k
python3 tools/moe-bench/run.py short    # 1500-token chat reply
python3 tools/moe-bench/run.py 12k
python3 tools/moe-bench/run.py 128k
```

Run each test twice and keep the second: the first request after starting the server also
reads the model from disk. With the cache off (`--moe-expert-cache 0`) the same binary
behaves like stock llama.cpp, for comparison. Token counts are for the GLM-5.3-Flash tokenizer; other models tokenize them differently.

## Regression testing with perf.py

`perf.py` runs the same tests against several builds and stores every run (speed, cache hit
rate, output text and its md5) in a sqlite database, so a new build can be compared with the
previous ones at any time.

Keep each build in its own directory under one folder (copy `build/bin/*` there), point
`PERF_BUILDS` at that folder, and run from the repo root:

```sh
mkdir -p ../rels/b11327-79e9090 && cp build/bin/* ../rels/b11327-79e9090/
export PERF_BUILDS=../rels                        # builds + perf.db (default: current directory)
export MODEL=../models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf   # first split for split models
export PERF_MODELS_DIR=../models                  # optional, see below
python3 tools/moe-bench/perf.py run b11297-fc23325 b11327-79e9090 -t chat --bare -n 2
python3 tools/moe-bench/perf.py show -t chat      # mean/stdev per build
```

Tests (`-t`): `chat` (1500-token chat reply), `tetris` (raw completion, repetitive output),
`pf12k` (12k-token prompt, then 32 tokens), `ppl` / `ppl3` (llama-perplexity on
`longsrc.cpp`, 1 or 3 tokens per decode call: the most precise decode comparison),
`agent` (compile-and-fix loop, needs miniagentic in `$PERF_AGENT_DIR`).

Options: `--bare` runs `llama-server -m MODEL` with nothing else (the fork's automatic
defaults, what a new user gets); `--plain` turns the cache off (stock-like); `--args="..."`
adds server arguments; `-e KEY=VALUE` sets environment variables (e.g.
`-e LLAMA_SPEC_DEPTH=2`); `-n N` repeats, alternating builds; `--note` labels runs.

Measurements are made with the model in RAM, as on a server after its first request:
before measuring, perf.py drops other models under `$PERF_MODELS_DIR` from the OS page cache,
reads the model file once if it fits in RAM, and runs every test once, discarded, right before
the measured run with the same build and settings. Compare decode speed only between runs of the same test; chat and tetris vary by a
few percent between runs, so use `-n 2` or more.

## GLM-5.3-Flash MTP head

`glm_splice_mtp.py --mtp-only <any glm5-next GLM-5.3-Flash.gguf> GLM-5.3-Flash-MTP-Q4_K.gguf`
rebuilds [neuralll/GLM-5.3-Flash-MTP-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-MTP-GGUF):
it reads the header of unsloth's UD-Q4_K_XL GLM-5.3-Flash GGUF over HTTP range requests
(`gguf_remote.py`), downloads only the 29 NextN (MTP) tensors (~4.3 GiB instead of ~200 GB),
and writes them with your model's metadata as a small GGUF for `-md`. Without `--mtp-only`
it writes a full model with the MTP block added.
