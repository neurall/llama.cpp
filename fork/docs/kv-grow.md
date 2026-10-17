# kv-grow: forward KV records behind the compute lend, grown by a horizon

Design: fork/docs/kv-fit.md (branch auto-kv-ram-quant-fallback), sections "First version", "Each growth step", "Measured: the KV move is negligible".

## Layout per card

    [ compute lend: 0 .. L ][ record 1: H cells ][ record 2: H cells ] ... [ experts: coldest .. hottest ]

* H = horizon in cells, the same for every record of the card (GLM has two: the latent-K attention cache and the indexer cache). Rows are forward:
  cell c is row c (no reversed rows), so the indexer's cell addressing is right.
* L = lend capacity: what the biggest graph of the current horizon borrows (graph tensors, CUDA scratch, staging slots).
* Growth of H (cells needed > H): H += step (1000 up to 10000, then 10000; step later from measured copy bandwidth + re-reserve time), the records are
  moved back to front (device copy, chunked, overlapping) and re-attached, the new blocks are claimed from the coldest experts.
* Growth of L (a graph needs more than L): all records shift forward by the difference, same move.
* Request end: H and L go back to their start values, records move down, the experts get the blocks back.

## Touch points (current code)

* llama-kv-cache.cpp: `can` (tm==2 only) and `kv_tail = kv_rev && kt` -> tail with forward rows (tm>=1, kv_rev off); the indexer guard on tm stays only
  for kv_rev; reg.attach is the re-bind used by a move.
* llama-moecache-fork.cpp: placement at load (~3909: stacked downward from tail_end -> upward from L); kvt_run.end_off -> start_off + H;
  kv_tail_claim ranges [end_off - w, end_off) -> [start_off, start_off + H*row); ap.span = L; kv_tail_pad_zero (~3213), kv_tail_covers (~3180),
  kv_tail_cap (~3155) count from end_off; the arena path (kv_arena_run, dense models) has the same end_off logic; moe_alias_plan's GGML_ABORT when the
  graph does not fit in front of the KV -> grow L (move) instead.
* kv_tail_cap -> "cells the horizon can still grow to" (the experts down to their floor); find_slot asks for growth instead of refusing.
* llama-context / common: no -c -> start horizon 2000 (or measured), micro-batch from the start horizon (no 16384 start + failed reserves).
