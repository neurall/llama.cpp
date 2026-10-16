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
* sizing (rule): nothing is reserved at start, the pool is sized as if the model had no KV. A request claims what it needs before its first pass (the server knows the
  prompt length), generation claims a few more blocks before the graph that writes a cell past the claim, and the end of the request drops the claim to what the
  next request keeps (reused prefix cells). The KV may take the whole expert cache (starved experts are streamed experts, still faster than a starved KV). What it leaves is the part of the pool a decode graph borrows from the front (measured per graph, at least two blocks): a request that needs more is refused with the numbers, and long prompts run in smaller passes (the pass planner reads what the pool can still lend).
  `-c` is ignored in this mode: the context of a request is calculated from the request and what the pool can give (upper bound: the model's trained context),
  a static reserve only hurts. A `-c` given by hand is still a cap.

## No allocation, no free

The record has no memory of its own: its tensors sit in a zero-size buffer until the pool exists, and the pool places the record in its own buffer (an offset, a
pointer record). Claiming and giving back blocks is an owner change in the pool's block table, never a CUDA allocation or free. Because the pool starts at the
first real decode, a warm-up decode would run on a KV without memory: with `kv-tail=1` there is no warm-up run.

## Switches

* `--moe kv-tail=1` (needs `kv-token-major=2`, flash attention): the tail layout; no memory is reserved for the context. Without `-c` a request may use as many tokens as the buffer holds (up to the model's trained maximum); an explicit `-c N` is the cap of a request, prompt plus answer (it stops an answer like `-n`), and costs nothing in VRAM. If `-n` is also given, `-n` bounds the answer and `-c` is not applied.
* `--moe kv-auto-ctx=0` (with kv-tail=1): the context is always `-c` (or the default 13312), as the size of the virtual record.
* `--moe kv-tail=0` (default): the normal KV cache, `-c` as always.
* Quantized K and V (`-ctk q8_0 -ctv q8_0`, ...) work in the record: rows of whole blocks, one extra alignment unit per cell in the record; the gather between cards uses the row size of the type.
* Flash attention is required: without it V is stored transposed (one row per dimension, not per token), which cannot be a record per cell; the cache then keeps the old layout and says so.
* Dense models (no expert pool on a card): one buffer of the card's free VRAM less the margin is allocated once at the first real decode and the record sits at its tail (kv-arena);
  a request may use as many tokens as the buffer holds.

## Layout of the one buffer per card (high addresses first)

1. the fixed part (hybrid recurrent state, static copies): at the very end, never moves
2. the KV record: its used rows grow downward from just below the fixed part
3. expert blocks: all the rest below, what the KV takes from
4. compute lend: at the front, grows up

The pool buffer gets the fixed part's bytes after its last block; the record's end offset is unchanged. In the dense arena the record ends the fixed part's size before the
buffer end and capacity is (buffer - fixed) / bytes per token. Until the recurrent memory takes its state from the buffer (a pointer instead of its own allocation) the state
is outside and the capacity reported for a hybrid model is slightly optimistic (the log line says so).

## What keeps the KV from overwriting experts it did not free

* the claim covers the tokens the graph will write (the cache has placed them before the graph runs), plus one block ahead, and a block the KV range touches is taken whole;
* uploads into taken blocks are dropped or awaited, the slot picker / refill / keep-hot paths skip KV blocks (owner -3);
* the prompt's tokens are claimed when its passes are planned (before the micro-batch is sized), so the planner sees the lendable bytes that are really left; a decode graph ends that plan;
* the compute graph borrows from the front: a graph that would reach into the KV rows is refused with the numbers (never silently).

## Why the padding rows must be cleaned

`llama_kv_cache::get_n_kv` rounds the number of cells the attention reads up to a multiple of 256 (the flash-attention tile; it also keeps the graph the same size between
steps so it can be reused). The rows between the used cells and that boundary are masked but read. The normal cache zero-fills its buffer at allocation, so they are always
finite numbers. In the tail they are memory that held experts: arbitrary bytes, and 0 x NaN is NaN in the softmax. So the invariant is "every row the attention can read is
finite". It is kept by zeroing the pad window only (the rows from the used count to the next multiple of 256), once per 256 generated tokens and at the start of a request;
no claimed block is cleared. The claim covers the padded count, so these rows are always the KV's.

## Length of a request

The memory of a request is its prompt plus its answer; nothing is reserved at start.
* `-c N` only: the answer is what N leaves after the prompt (N is in + out, like the end of the old context).
* `-n M` only (or `n_predict` in the request): the context of this request is prompt + M, calculated at its start; no `-c` is needed.
* both: `-n` bounds the answer, `-c` is not applied.
* neither: llama.cpp's own defaults (`-c 0` = the model's trained context, `-n -1` = until the end token); the answer also ends when the machine's capacity is reached. The fork's default context of 13312 is not injected in this mode (it only existed to keep the KV reserve from starving the cache).
A request that needs more than the machine holds (the smaller of the KV capacity and the model's trained context) is refused at its start with the numbers.

## Limits

* one slot (`-np 1`): the used cells must be a prefix. With several slots cells scatter; compaction (moving rows) is a later step.
* K and V of equal float type, flash attention, no shared cells (the same as kv-token-major=1).
* cross-card: a strided view is gathered on the source card (rows, one plane per KV head) and crosses as one block (`strided-copy`).

## What it can win

The difference in decode speed and hit rate between `--ctx 4096` and `--ctx 131072` on the same prompt (queue cells `ctxcost2-*`) is the most a tail watermark
can gain on a model. If it is inside noise, step B is not worth its complexity.
