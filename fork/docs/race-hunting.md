# Hunting races, overwrites and stale reads

Written 2026-10-28 after the DSpark repeat-request bug (answers stopped at ~717 characters, a CUDA graph replayed with the scratch addresses of its capture). The switches below, in the order that narrows such a bug fastest. Every run goes through run.py; put the switches in a list row (`--moe ...` in args, `--env K=V` in extra).

## 1. See it in the numbers

| check | where | what it catches |
|---|---|---|
| ANSWER STOPPED | run.py prints it after a run; `run.py bad-runs` lists it as "answer stopped" | an answer under half a character per generated token: the model emits tokens that print nothing |
| REPEAT DIFFERS | run.py, the runs of one cell (greedy) | a later run whose text parts from the first run's: cache, replay or state bugs that need a warm request |
| draft acceptance per request | the server log (`draft acceptance = ...`) | a draft whose target went wrong collapses to a few percent |

Run the cell with `-r 0` and n = 2 (first and repeat request) and a long enough answer (the DSpark bug needed 256 tokens).

## 2. Race or layout? (minutes each, no build)

| switch | if the bug disappears |
|---|---|
| `--env GGML_CUDA_DISABLE_GRAPHS=1` | a replayed CUDA graph uses something that changed since its capture (an address, a size). The 2026-10-28 bug. |
| `--env GGML_SCHED_NAN_CHECK=2` (every node alone, very slow) | a race between streams or threads: draining after every node closes the window |
| `--env GGML_SCHED_NAN_CHECK=2 --env GGML_SCHED_NAN_CHECK_ONLY=word` | only the splits whose first node's name holds the word run node by node, so a report goes to the node you are after. A bad MUL_MAT_ID also names its bad columns and the experts and blocks behind them. No fusion in this mode |
| `--env GGML_SCHED_NAN_CHECK=1` (every split drained) | still there: the fault is inside one split; it names the first tensor with a NaN |
| `--moe sync=1` (default) / a run with uploads off | uploads racing the graph |
| `--env GGML_CUDA_DISABLE_FUSION=1`, then `--moe fuse-off=GLU+MUL_MAT_ID` (a `+` list of the ops a fusion starts at) | a fused kernel: the first switch parts all of them and changes the allocation, the second names the one |

## 3. Name the bytes and the writer

| switch | gives |
|---|---|
| `--moe poison=3,ledger=1` + `--env GGML_SCHED_NAN_CHECK=1 --env PERF_LEDGER=1` | unwritten bytes are NaN with the writer's tag; the first NaN dumps every pool (`fork/bench/pool-dump-<card>-*.txt`: per MiB the byte that fills it and the writers named in it) and stops |
| `--moe ledger=1` (`PERF_LEDGER=1`; on by default for builds named `fix_*`) | `q5h/TAG.ledger.csv.gz`: one line per pool access: t_us, thread, card, kind, rw, state, lo, hi, bytes, who; flushed every 2 s |
| `fork/pool_ledger.py LEDGER --races` | another thread's write inside a graph's window on its bytes; an upload overlapping a graph window |
| `fork/pool_ledger.py LEDGER --stale` | a CUDA graph replay whose scratch range is outside the current lend |
| `fork/pool_ledger.py LEDGER --blame N`, `--addr CARD:MIB`, `--at T` | who touched those bytes last, everything at one moment |
| `--env GGML_CUDA_SCRATCH_DEBUG=1` | every scratch region set, and every CUDA graph captured again because the region moved |
| `--moe ledger=2` | on top of the writers: what the graphs touch. `node-w` and `node-r` per pool range a node writes or reads (split, node, op, tensor names), `mix-read` per expert a home card reads by its table, `chain-up` and `jit-up` for the uploads inside a graph. `pool_ledger.py LEDGER --addr CARD:MIB` then lists everyone who touched a byte, in order. 5-10 thousand lines per new graph |
| `--moe ledger=1`, `fork/pool_ledger.py LEDGER --dma` | every copy into a pool with the times its stream really ran it, its pointer and length, against the pool map: a lend, a KV zone, KV rows moved or zeroed, a fill or blocks given back on its bytes while it ran, two copies on one range at once, a copy that never ended, and how far the streams run behind the host |
| `--moe verify=1` (2: every graph of 64+ tokens, 3: every graph) | every cached expert read back and compared with the host's copy, before a graph of the prompt mix or chain. A bad block is named and goes into the ledger. Clean means the experts are right and the fault is in a kernel or its inputs |
| table check (always every 64th step, every step with ledger or poison) | an expert table entry pointing at a block its layer does not own |

A switch that clears a bug proves nothing until the row shows the same path ran: compare `pool`, `margin` and `paths` in the two rows' notes, and the answer text of request 1 and 2 (`repeat parts at char N`). The 2026-11-02 NaN cleared under `GGML_CUDA_NO_PINNED=1` and `GGML_CUDA_DISABLE_FUSION=1` only because those rows never ran the prompt mix.

## 4. Make a rare race happen every time

| switch | does |
|---|---|
| `--moe race-stress=N` | upload workers sleep 0..N ms before each copy and before reporting it done |
| `--moe pp-path=stream\|mix\|chain\|dec` | every prompt call takes this path: the planner picks by time, so a path may run once in a long request or never. List `fork/runs/paths.csv`: each path, 16k prompt, poison, NaN check and byte check |
| a shorter `-c` (8192) and a prompt whose length puts a pad boundary (256 cells) inside the answer | the lend and the zone change mid-answer |

## Hotspots (bug-log rows by 2026-10-28)

The lend, compute alias and CUDA scratch (26), the pass planner (23), the KV zone (23), expert uploads and tables (21), the dense split and the DSpark draft (12 each). New code there gets a ledger kind and, where cheap, a check that runs always.
