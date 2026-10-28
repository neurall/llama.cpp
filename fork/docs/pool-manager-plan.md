# Pool debugger and pool manager: plan

Written 2026-10-27. Rules from the user: keep it simple, keep speed, be pragmatic. Every buffer on a card is
allocated, grown and accessed through checked, logged code, and each kind has a test that what is used equals what is
asked.

## What changes the order

The ledger (`--moe ledger=1`, `led_line` in `src/llama-moecache-fork.cpp`) already logs at every writer of the pool:
- the lend and its release
- the zone layout, row moves and pad zeroing
- expert uploads (start and end)
- keep-hot copies
- the dense split's mirror fills
- poison fills
- an OVERLAP check at each upload

So the live writer on today's code is small: it sends those events to a JSONL file and adds snapshots. The pool
manager refactor is the bigger job. The rule "refactor first only if it is the smaller effort" therefore puts the writer
first. The refactor later moves the same calls inside the manager; the file format and the page stay as they are.

## Part 1: live debugger on today's code

| step | what | cost when off |
|---|---|---|
| 1 | `--moe statsjson=1`: the ledger's events as JSONL in `/dev/shm/pool-events-<pid>.jsonl` (time, card, kind, start, end, who). A snapshot line in `pool-snaps-<pid>.jsonl` at each layout change and once per second: regions, block owner, layer, expert, hot counter, lifetime uses | none: the writers already test one flag |
| 2 | the lend split into its parts (graph, CUDA scratch, staging slots), which `llama_moe_cache_alias_fork` already knows | none |
| 3 | access check: once per split, the scheduler hands the nodes to one function, which checks every tensor in a pool buffer against the region map. Graph tensors must lie in the lend's graph part, KV views in the zone, expert weight views in their class. A miss is a VIOLATION event (red flash on the page); `statsjson=2` aborts at the first | one branch per split |
| 4 | run.py: a cell flag passes `statsjson=1`; the page on port 7000 already reads the files | none |

About 300 lines, one build.

## Part 2: pool manager (task 145)

| step | what | check |
|---|---|---|
| 1 | one pure layout function in `llama-kvzone-fork.h`. Each member declares its shape (row bytes, ratio, maximum rows, unit, read padding) and its rows in use; the function returns the offsets | unit test: one-lane shapes give byte-identical offsets to today; fuzz: no two members overlap |
| 2 | the ~70 hand copies of `fixed + cells x row` call that function | regress every model: the same pool lines in the log |
| 3 | lanes, one per ratio. Ordered moves: members moving down go first in rising order, then members moving up in falling order | fuzz with a shadow memory. DeepSeek 4.1's ratio-2 cache gives about 1.9 GiB back to the experts |
| 4 | every layout change and grow goes through the manager. It checks the new layout first (inside the pool, no overlap, blocks under new rows owned by the zone before the rows land), then emits the ledger event | the Part 1 page shows it unchanged |

## Speed rules

- Nothing added per token when the debugger is off.
- Layout work happens once per pass or growth step, never per token.
- Moves copy only the rows in use.

## Not doing (simple first)

- A registry for host-side buffers.
- Computed scratch for small ops (argsort, top-k, softmax): the shortfall alarm covers them.
- Byte-level shadow memory on the GPU: unit tests only.
