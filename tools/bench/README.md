# Benchmarks for the MoE expert cache

`tools/run.py` is the one entry point: it runs the fork's tests against any number of builds, models and settings, on this host or
on other machines over ssh, and appends every run to `run-history.csv`. This directory holds what it needs and the analysis tools.

- `src_12k.cpp` (12,302 tokens of llama.cpp source), `src_128k.cpp` (125,257 tokens), `longsrc.cpp` (perplexity text): frozen inputs
  so anyone can reproduce the numbers in the main README. Token counts are for the GLM-5.3-Flash tokenizer.
- `run-history.csv`, `campaigns.csv`, `machines.csv`, `tune.csv`: every dev and test run and what is needed to read it (plain csv, appended by
  `run.py`, committed).
- Cache policy and predictor simulations on router traces, and the GPU / cache trace summaries are in `tools/sim/`.
- `ddrbw.cu`, `membw.cpp`, `memlat.c`, `l3pref.cpp`, `sync_lat.cu`, `uva_read.cu`: memory bandwidth, latency and sync micro-benchmarks.
- `glm_splice_mtp.py`, `gguf_remote.py`, `gen_session.py`: GGUF and session helpers.

## Reproduce a README number against a running server

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf -c 131072   # -c only needed for 128k
python3 tools/run.py prompt short     # 1500-token chat reply; also: 12k, 128k, chatv (varied prompts), code
```

Run each twice and keep the second: the first request after starting the server also reads the model from disk. `--moe cache=0`
turns the whole fork off (stock behaviour), `-at off` only the self-tuning. `--prompt TEXT|@file`, `--tokens N`, `--ctx N` replace
a test's own prompt, length and context.

## Compare builds: `run.py bench`

Each build lives in its own directory (copy `build/bin/*` there); `PERF_BUILDS` points at the folder holding them (the csv files are written to `tools/bench/`, or to `$RUN_DATA`):

```sh
export PERF_BUILDS=../rels
python3 tools/run.py bench --models ../models/a.gguf,../models/b.gguf --builds mybuild,stock-abc1234 -n 2
```

Protocol v2 (what a number in the main README should come from): `-t fix` (short prompt, `-c 2048`) or `-t fix12k` (12k-token prompt, 256 generated), each with a
fixed seed (424242), `ignore_eos` and an exact token count, so every build generates the same number of tokens and equal outputs show equal `md5`; `-n 3`
(the default for these tests) runs the cells interleaved ABBA (odd repetitions run backwards) so a drift cannot favour one cell, with one discarded
run per cell before its first repetition. `--stock-variant tuned='--n-cpu-moe 30'` also runs a hand-placed stock (the baseline a cache has to beat at
the same VRAM); a variant can set environment variables with `env:` words, e.g. `--variant lru='env:LLAMA_MOE_CACHE_POLICY=lru'`. `PERF_HOLD_MB=N` tells
the harness N MiB of VRAM are held on purpose (a simulated smaller card, see `tools/experiments/vram-sim`).

Per model it runs every build with the variants `default`, `cache0` (`--moe cache=0`) and `atoff` (`-at off`) (add or replace with
`--variant gate3='--moe gate=3'`; builds named `stock*` run the default only), then prints the report: n, median and min-max per cell, the ratio to the stock median (and to the best hand-placed stock),
and flags a cell with n < 3, different token counts or different outputs; `report --md` prints README table rows with the first run of each cell. Every measured run is preceded by a discarded run with the same settings, so the numbers are hot runs. `-t`
picks the test: `t100` (default: one short prompt, 100 tokens, `-c 1024`, temperature 0, what most people run), `tetris`, `chat`,
`chatv`, `pf12k`, `pf128k`; `run.py run` has the rest (`ppl`, `ppl3`, `agent`). `--deadline HH:MM` skips cells that would start
later, `--tol 0.95` flags cells below 95% of stock, `--last N` reports only the last N runs of a cell (an autotune that is still learning).

Other machines: `run.py machine add pc2 --ssh user@host --root DIR --os linux` (Windows: `--os windows`, a W10 box must be on AC power),
then `--machine pc2`; the builds are subdirectories of that root and the test runs through `llama-cli`.

## run-history.csv

One line per run, appended; the key columns come first so a line reads at a glance (`column -s, -t < run-history.csv | less -S` for a table):
`ts` (date and time), `commit` and `bno` (commit and build number from the binary's own `--version`), `tps` (decode t/s), `pp` (prompt t/s),
`hit` (expert cache hit %), `model`, `args` (the variant's command line, e.g. `--moe cache=0`), `test`, `hw` (machine), `ok` (failures are
kept, error in `note`), `build`, `md5` (of the output text: compare t/s only between equal md5), `campaign`, `note`, then `ppl`, `spp`
(seconds per pass), `n_gen`, `origin` (`stock` or `fork`), `env`, `cmd` (the full server arguments), then `pl` (GPU power limits in W, e.g. `280+280`, local
machine only), `proto` (`v2` = fixed workload above, `v1` = earlier runs), `rep` (repetition within a bench) and `seed`. Rows from before these columns
existed have them empty.

- `stats.ini`: the expert cache's own log of a run, one `[ts build]` section per run (the same key as its row): `every` tokens per value, then one line per
  metric with a value per window: `tok_s`, `hit_pct`, `uploads`, `evictions`, `up_mib` (uploaded MiB) `ddr_gbs`, `pcie_gbs`. Written by builds that have
  `--moe statslog=N` (a line every N steps, 1 means 32; `LLAMA_MOE_STATSLOG=path` for a text file of one process).

- `machines.csv`: the machines (`pc1` 2x RTX 3090 on PCIe 4.0 x16 + chipset x4, Ryzen 7 3700X, 125 GB DDR4-3200; `pc2` CPU-only Ryzen 5 3600,
  64 GB DDR4-3200, PCIe 3.0; `pc3` RTX 4060 laptop 8 GB, Ryzen 9 8945HS, 32 GB LPDDR5X-6400) and how to reach them.
- `campaigns.csv`: a named comparison with its start time, note, conclusion and why. `tune.csv`: the self-tuner's decisions per run.

```sh
python3 tools/run.py show -t t100          # mean/stdev per build   |   python3 tools/run.py report --campaign NAME
```

`run.py ctl BUILD name:KEY=V,KEY=V ...` switches settings at runtime on one server (`LLAMA_MOE_CACHE_CTL`) with rotating prompts and
run order: the A/B for knobs that need a warm cache.

## Shared slot pool (tried, not kept)

One slot pool shared by the layers of a quant type, with a victim chosen over the whole pool, was tested on GLM 3.5 (2 x RTX 3090), OLMoE and Qwen3.6-35B on the
8 GB laptop. `pool-grid.csv` has every finished run (one row each: ratio, window, decay, learned weights, hit rate, t/s; append-only). No variant beat the per-layer
cache: the best tied it within noise (GLM steady hit 63.6% vs 62.4-63.7% for pool off), the shared structure with the old score was 0.2-1.4 points lower, and scores that
rely on recent use alone (window-only, exponential decay, hidden Markov model) were lower still. The simulator (`tools/sim/pool_sim.py`) had predicted gains of 1.7-2.5 points,
so it is not a guide for tuning without async upload timing and stream slots. The engine code lives on the private branch `pool-experiment` of neurall/moe-net.

