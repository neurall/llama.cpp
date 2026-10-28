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

## Part 2 design rules (user, 2026-10-28)

Trigger: DeepSeek 4.1 + DSpark draft, every repeat request broke after ~256 tokens (answer stops printing, draft acceptance 3%). With `--moe poison=3` and the NaN check, the first NaN is layer 14's MoE output on CUDA1 for the draft token of a 2-token verify batch, in the graph whose lend grew from one block of class 7 to two (7.44 -> 9.28 MiB) when the cells crossed 512. Bounds and owners are worked out in many places, so such a case has no single place to check. The user's rules for the manager:

| rule | what it means here |
|---|---|
| one manager per card owns every byte | the lend parts (graph, CUDA scratch, staging slots), the dense split's copies (today placed from llama-context.cpp and only reported to the ledger), the KV zone (fixed part, growing rows, every record), the expert blocks of every size class, and the guards. No other code computes an offset or changes an owner. |
| one function changes the map | a commit takes the new layout, checks it (inside the pool, no two regions overlap, KV rows and their padded read inside the zone, lend parts behind each other in front of the zone), then runs the transitions in order: drop or await uploads into blocks being taken, evict, mark owners, publish the expert tables, move rows, zero the pad, give blocks back last. Ledger events and poison fills happen only inside it. |
| a request states its access chunk | an allocation asks for (len, chunk): the tile or pad its kernels read and write in. The manager returns [ptr, ptr + len) and at least one more chunk owned by the same region, so a whole-chunk read or write behind ptr + len never reaches another owner. The guard holds safe values for reads (zero for masked attention rows: 0 x NaN is NaN), also in poison mode. Cost: one chunk per region, well under 1% of the pool. |
| shared functions, copies only when measured slower | every bound (zone end for N cells, padded read cells, lend end, blocks under a range) comes from one function; the hand copies of `kv_start + kv_fixed + cells x kv_row` (about 15 in llama-moecache-fork.cpp) call it. |
| fast reuse, autogrow, automove, tight fit stay | layout work only per pass or growth step, never per token; moves copy only rows in use; no free and realloc of CUDA memory (one buffer per card, owner swaps). |
| one cheap check per graph, always on | before a graph runs: every pool region it touches is owned by its request at that moment (KV rows up to the padded read inside the zone and owned by the KV, lend parts not overlapping, every expert table entry pointing at a block owned by that layer). A comparison of a few numbers per region, not per byte. |
| n pools for n cards | one map per card (`mc->alias`, one pool buffer per GPU), keyed per card, nothing assumes one or two cards. A commit may change several cards' maps at once (a graph lends on every card it runs on; a cache's KV records span every card): it checks every card's new layout first and applies only when all pass. The per-graph check runs on every card the graph touches. Each card's manager holds all of its own state: no process-wide flag or counter speaks for every card (today `kv_gave_back` and `starved` in `moe_cache`, `g_alias_err` and the `g_kvs` counters are shared by all cards). |
| layout serial on every cached address | each card's manager bumps a serial whenever an address a graph can bake in changes (lend base, graph region, CUDA scratch region, staging slots, dense split copies). Whatever keeps such an address stores the serial it saw and compares before use: llama's ggml-graph reuse (can_reuse), the CUDA graph cache (re-capture on a mismatch), any later pool pointer. One integer compare per use, always on. Found 2026-10-28: a replayed CUDA graph kept the scratch addresses of its capture after the pool moved the region (DeepSeek 4.1 + DSpark repeat requests broke ~256 tokens in; fine with CUDA graphs off). The immediate fix compares the scratch base and size at replay; the serial is the general form. |
| access log for debug and fix-validation runs | one global log, one line per access: time, host thread, GPU stream, who (upload worker, graph and node, zone move, pad zero, keep-hot, draft graph), what (expert L/e, KV rows, graph tensor, scratch, staging), GPU, pool, offset, bytes, start or done, read or write. GPU work carries GPU event times (a host start/done is only when it was queued). Replays log the address ranges their capture recorded, so stale addresses show. Written to a file and flushed every few seconds (default 2) so a crash or kill keeps it; on for debug cells and fix-validation cells, off for speed A/B cells. A checker flags any write that overlaps another actor's read or write of the same bytes in time, and any access outside the owner's region at that moment. |
| easy to trace and to find the culprit | every region carries its owner, the call site that asked for it and the serial of the commit that last changed it; poison fills carry the same serial, so bad bytes lead back to a commit. A ring of the last few hundred commits is always kept in memory (cheap) and dumped with the region map on any failed check or abort, so the dump already holds the culprit. A failed check prints the region, the owner it expected, the owner it found, the graph, and the last commits that touched that range. The ledger, the pool dump, the live pool page and the fuzz test all read the manager's own map. |

## Speed rules

- Nothing added per token when the debugger is off.
- Layout work happens once per pass or growth step, never per token.
- Moves copy only the rows in use.

## Not doing (simple first)

- A registry for host-side buffers.
- Computed scratch for small ops (argsort, top-k, softmax): the shortfall alarm covers them.
- Byte-level shadow memory on the GPU: unit tests only.
