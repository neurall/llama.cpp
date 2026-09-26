# Benchmark prompts for the MoE expert cache

The inputs behind the numbers in the main README, frozen so anyone can reproduce them.

- `src_12k.cpp`: 12,302 tokens of llama.cpp source (the "long" 12k test).
- `src_128k.cpp`: 125,257 tokens of llama.cpp source (the 128k test).
- `run.py`: sends a test to a running `llama-server` and prints prompt and decode speed.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf -c 131072   # -c only needed for 128k
python3 bench/moe-cache/run.py short    # 1500-token chat reply
python3 bench/moe-cache/run.py 12k
python3 bench/moe-cache/run.py 128k
```

Run each test twice and keep the second: the first request after starting the server also
reads the model from disk. With the cache off (`--moe-expert-cache 0`) the same binary
behaves like stock llama.cpp, for comparison. Token counts are for the GLM-5.3-Flash tokenizer; other models tokenize them differently.

## Regression testing with perf.py

`perf.py` runs the same tests against several builds and stores every run (speed, cache hit
rate, output text and its md5) in a sqlite database, so a new build can be compared with the
previous ones at any time.

1. Keep each build in its own directory, e.g. `~/rels/b11327-79e9090/` (copy `build/bin/*` there).
2. Run tests from that parent directory:

```sh
cd ~/rels
export MODEL=/models/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf   # first split for split models
export PERF_MODELS_DIR=/models                                  # optional, see below
python3 /path/to/llama.cpp/bench/moe-cache/perf.py run b11297-fc23325 b11327-79e9090 -t chat --bare -n 2
python3 /path/to/llama.cpp/bench/moe-cache/perf.py show -t chat          # mean/stdev per build
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
reads the model file once if it fits in RAM, and on every model switch does one discarded
run. Compare decode speed only between runs of the same test; chat and tetris vary by a
few percent between runs, so use `-n 2` or more.
