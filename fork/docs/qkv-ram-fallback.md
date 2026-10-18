# qkv-ram-fallback: KV tiers by context length (branch qkv-ram-fallback, from kv-grow)

User design 2026-10-17. For every model, calculate at start the largest context whose f16 KV still leaves the expert cache at its
floor ("max real ctx where no space is left for moe", example Qwen Next about 256k). A request is then served by the tier its
length needs:

| request length (prompt + answer) | KV | where |
|---|---|---|
| up to `T0` (f16 cap, calculated) | f16 | GPU, growing in the pool (kv-grow) |
| `T0` .. `T1` (about 1.9 x T0) | q8_0 K, q8_0 V (`k8v8`), flash attention | GPU pool |
| `T1` .. `T2` (about 3.5 x T0) | q4_0 K, q4_0 V (`k4v4`) | GPU pool |
| above `T2` | RAM, no KV in the GPU | host; attention on the CPU for this request |

When the KV leaves the GPU (RAM tier) the pool space it held goes back to the expert cache, which compensates the slower
attention. Anything above the RAM capacity (free RAM minus a margin) is refused with the numbers. `--moe kv-fallback=LIST`
(default `k8v8,k4v4,ram`) selects and orders the tiers; `none` refuses as today. The user is told upfront (log at warning
level before the prefill, and a `kv_fit` object in the first SSE event). Details of the list syntax and the quality notes:
branch auto-kv-ram-quant-fallback, fork/docs/kv-fit.md.

## Thresholds

Per card the pool room for the KV is `room = tail_end - lend - kv_fixed - floor` (the existing `kv_tail_cap`); cells of a tier
= `room / (kv_row * bpv(tier) / bpv(current))` with bytes per value f16 2.0, q8_0 1.0625, q5_0 0.6875, q4_0 0.5625. The three
thresholds are the minimum over the cards (a record sits on every card that has layers). Logged once when the pool exists
(first real decode), and offered in /props.

## Mechanism (phases)

1. Capacities and the log line (no behaviour change).
2. Tier switch at a request start (cache empty): `llama_kv_refit(ctx, type_k, type_v, offload_kv, n_cells)` rebuilds the
   memory object with the new types, the pool re-places the record, the graph is reserved again; the expert pool stays.
3. RAM tier: the cache is built with `offload_kqv=false` for the KV (host buffer sized by the request), attention on the CPU;
   the pool zone shrinks to nothing and the experts take the space back.
4. Context default without `-c` becomes the maximum over the enabled tiers (trained maximum, bounded by RAM), not the f16 cap.
5. README table: stock's maximum -c per card against us at 1x, 2x, 3x, 5x, 10x (fork/docs/ctx-reach.md).

## Open checks

flash attention with quantised K/V at each model's head size (MiMo 192/128, GLM 512 MLA, Qwen Next 256); perplexity per
rung (OLMoE ladder: K is the sensitive tensor); CPU attention speed on the RAM layout at 128k+; the first request, when the
pool does not exist yet (the pool starts at the first real decode): start it before serving or decide the tier mid-request.
