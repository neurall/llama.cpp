# KV cache at the tail of the expert pool (design)

Goal: a context only pays VRAM for the cells it uses. Today `-c N` reserves N cells of K and V for every layer at start; with a large `-c` and a short prompt
that reserve is expert cache lost (the expert cache hit rate is what the decode speed follows).

## Layout

* `--moe kv-token-major=1`: K and V of all layers of one device are one record per cell (a layer's K / V is a strided view into it). The cell count is the only
  growing dimension. Costs nothing against the normal layout (MiMo 1k edit: 537/532 vs 540/540 t/s prompt, decode equal).
* `--moe kv-token-major=2`: the same, cell `c` in record row `kv_size-1-c` (a ggml view cannot have a negative stride). The used cells are the LAST rows of
  the record, so the record's free part is at its low addresses.

## Where it lives (step B, not written)

The record is a range at the end of the pool buffer, ending where the last size class's expert blocks end (before its zero block). Only the used rows
`[end - w*row, end)` are claimed; the rest of the record's address range is expert blocks that the cache may keep using. Nothing ever reads or writes a row of an
unclaimed cell.

* claim: before a graph that writes cell `c`, the blocks covering rows up to `c` are taken from the experts: their experts are evicted, the owner becomes -3
  (the compute alias uses -2 at the FRONT of the buffer; the two ranges may not meet: the lend limit is span minus the KV claim).
  The eviction, the in-flight upload handling and the table flush are the ones of `llama_moe_cache_alias_fork`; the claim runs at the same point, from the
  watermark the cache published when it allocated the cells (`find_slot`).
* release: when the highest used cell drops (sequence removed, context shifted), blocks above the new watermark become free expert blocks (metadata only; they
  refill by uploads like blocks after a prompt, `reheat_free_blocks`).
* sizing: the pool is sized without the KV reserve; a context that needs more than the pool can give (keeping a minimum of experts) is refused with the numbers.

## Limits

* one slot (`-np 1`): the used cells must be a prefix. With several slots cells scatter; compaction (moving rows) is a later step.
* K and V of equal float type, flash attention, no shared cells (the same as kv-token-major=1).
* cross-card: a strided view is gathered on the source card (rows, one plane per KV head) and crosses as one block (`strided-copy`).

## What it can win

The difference in decode speed and hit rate between `--ctx 4096` and `--ctx 131072` on the same prompt (queue cells `ctxcost2-*`) is the most a tail watermark
can gain on a model. If it is inside noise, step B is not worth its complexity.
