# Prompt mix crash (task P8): stale MMQ tile specs below the `-1` ids

2026-11-07, GLM-5.3-Flash 3.5bit (`glm35`) 12k, rows 23770 / 23786 / 23788, builds rampool10, rampool12. Server dies with
`CUDA error: an illegal memory access` at `ggml/src/ggml-cuda/ggml-cuda.cu:135`; run.py sees `Remote end closed connection`.

## What the logs say

The error is sticky: the site that reports it is only the next CUDA call, which is why the three crashes name three places
(`ggml_backend_cuda_device_event_synchronize` on device 1, `ggml_cuda_kernel_launch` on device 1, `ggml_cuda_xstage_copy` on device 0).
None of them is the faulting kernel.

What every crash shares: the split planner chose **one card** for a pass whose moe-cache pick was `stream, mix`, so the mix graph really
ran (`llama_moe_cache_pp_mix_fork` returns the x16 card, `mc_mix_b` set in `llm_graph_context::build_moe_ffn`):

| row | log (`/p/bw/data/q5h/`) | pass | planner line | outcome |
|---|---|---|---|---|
| 23770 | `rampool10-glm35-12k-rp1.server.prev.log:550` | 1938 tokens, 1st request | `one card [stream, mix] ... : one card (p_split 0.28)` | crash at `:558`, 30 ms after the first two staging blocks of the pass were pinned (`:556-557`, 122 MiB CUDA1->CUDA0, 32 MiB back) = inside the first mix layer |
| 23786 | `rampool12-planner-learned-glm35-12k-rp0.server.log:502` | 2034 tokens, 3rd request | `one card [stream, mix] ... trial of the one card` | crash at `:505`, 2.2 s into the pass |
| 23788 | `rampool12-planner-learned-glm35-12k-rp1.server.prev.log:665` | 10496 tokens, 1st request | `one card [stream, mix] ... : one card (p_split 0.50)` | crash at `:668`, 7 s into the pass |

The five "good mix" calls of rampool11 (`rampool11-planner-simple-glm35-12k-rp0.server.log:341,420,440,510,529`) all end in
`: split (p_split ...)` and the next `prefill decision:` says `mode=dense-split`. A dense split sets `moe_ranges`, and
`build_moe_ffn` builds the mix only when `moe_ranges.empty()` (`src/llama-graph.cpp:2999`). So the mix graph never ran in those calls;
the `stream, mix` label is the moe-cache pick, the dense split overrode it. Across all of `/p/bw/data/q5h/*.server*.log` there is
**no one-card mix pass that finished**: 3 ran, 3 crashed. The bug is not intermittent, the planner's choice is.

## Root cause

`ggml/src/ggml-cuda/mmq.cu:338-346` (`ggml_cuda_mul_mat_q`, the ids branch):

```
if (dst->op_params[1] != 0) {
    // ids may be -1 (skipped experts): the helper writes no entries for them; row 0 is a safe target ...
    CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows*sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ids_dst.get(),  0, ne_get_rows*sizeof(int32_t), stream));
}
ggml_cuda_launch_mm_ids_helper(..., tile_spec.get(), n_grid, ne12, n_expert_used, ne11, J_best, ...);
```

`tile_spec` (`ntj = ceil(ne_get_rows / J_best) + n_grid` entries from the pool, `mmq.cu:326-327`) is **not** cleared. The helper
(`ggml/src/ggml-cuda/mmid.cu`, `mm_ids_helper`) counts every id `< expert` into `nex_prev`, and `-1 < expert` for every expert, so
the compact columns `[0, n_neg)` (one per `-1` id) belong to no expert and expert 0 starts at `n_neg`. The seam loop at the end of
the helper writes `tile_spec[k]` only for seams `k*J` inside some expert's range, so the seams `k < ceil(n_neg / J)` are never written
and keep whatever the pool memory held. `mul_mat_q` (`mmq.cuh:1079-1083,1162-1166,1242-1246`) reads `tile_spec[jt]` for every
`jt < ntj`: a stale `expert` indexes `expert_map[]` / `x + expert*stride_channel_x`, stale `col_low/col_high` index `ids_dst[]` and `y`,
all out of bounds -> illegal memory access. When the stale bytes are the previous layer's tile specs of the same shape the reads stay
in bounds and the tile silently writes a wrong row (the same op, the layer before), which is the "'!!!!' or a wrong answer" the comment
at `ggml-cuda.cu:3461-3462` describes for the mix.

Who has `-1` ids with a big batch on the MMQ path:
* the mix's helper card (x16): `ids[i]` is `-1` for every expert the home card holds (`src/llama-graph.cpp:3618-3622`), about half
  of them on these runs ("52% belong to the other card's cache"), `op_params[1] = 1` set by the scheduler's 2-GPU prefill rule;
* the dense+expert split (`helper_expert_share > 0`), the same mask by id range (`src/llama-graph.cpp:3615-3617`); it did not run on
  these cells (`cards=CUDA0:0.46:0.00,CUDA1:0.54:1.00`: all experts on one card), so it did not crash, but it is the same hole.
The home card's chain is not affected: its ids are the raw expert ids (`src[3] = sel_dev`, `mc_mm` at `src/llama-graph.cpp:3183`)
and the `-1` lives in `expert_map`, which the kernel checks. The decode chain's `-1` ids go through `mmvq` (batch <= 8), not MMQ.

Why now: upstream `69ee2ecc01 CUDA: linearize MoE data layout for MMQ/MMF (#31047)` (2026-11-05), merged in
`45a4d6fc56 Merge upstream into up-merge-9` (2026-11-07 06:44), replaced `expert_bounds` (n_grid + 1 prefix sums, every entry written
for every expert, `git show 45a4d6fc56^1:ggml/src/ggml-cuda/mmq.cu:319,336`) with the per-seam `tile_spec`. The fork's `-1` handling
(the two memsets) was kept but covers only the two id arrays. rampool6 (`863d6a8e6b`, 14:35) already carries the merge; its mix calls
were dense splits too. `0bce6ce660 scratch: MMQ tile width and grid over the experts of a table` adapted the scratch mirror, not the hole.
up9_merged2 (`90d772e026`) has the merge as well; no one-card mix is logged for it.

`ggml/src/ggml-cuda/mmf.cu:90-96` has the identical pattern (f16/bf16 experts, `tile_spec_dev` not cleared, same helper); its kernel
returns on an empty tile (`mmf.cuh:357-366`), so the same fix applies there.

## Fix (two lines, no header edit, no scratch-size change)

A zeroed `ggml_cuda_mm_id_tile_spec` is `{expert 0, col_low 0, col_high 0}`: an empty tile, which every `mul_mat_q` branch and the
stream-k fixup return on (`col_low == col_high`). Clear the specs in the same block that clears the id arrays:

`ggml/src/ggml-cuda/mmq.cu`, inside `if (dst->op_params[1] != 0) {` after the `ids_dst` memset (line 342):
```
            CUDA_CHECK(cudaMemsetAsync(tile_spec.get(), 0, ntj*sizeof(ggml_cuda_mm_id_tile_spec), stream));   // the seams below expert 0 (the -1 columns) are never written by the helper
```
`ggml/src/ggml-cuda/mmf.cu`, inside `if (dst->op_params[1] != 0) {` after the `ids_dst_compact_dev` memset (line 93):
```
            CUDA_CHECK(cudaMemsetAsync(tile_spec_dev.get(), 0, max_tiles*sizeof(ggml_cuda_mm_id_tile_spec), ctx.stream()));
```
Both buffers come from the pool, the memset adds no allocation, so `ggml_cuda_mmq_scratch_fork` and `tests/test-scratch-fork.cpp` stay
as they are. The fused MMQ+GLU path (`ggml-cuda.cu:6038,6623`) goes through the same function. Then `pp_mix_opt()`
(`src/llama-moecache-fork.cpp:9320`, off since `75eaa6761b`) can go back to on by default.

## Reproducer (Sonnet, PC1 when the gate cells are done)

The dense split must lose, so force one card; the mix must be picked, so force the path and the switch:

```
python3 fork/run.py cell mixfix-glm35-12k glm35 <build> 2 pf12k1k "--moe dense-split=0,pp-path=mix,pp-mix=1"
```
Before the fix: every start dies in the first mix pass (the 1938/2034-token call of request 1; rampool10 died inside its first mix layer)
with `illegal memory access`, row ok=0. After the fix: both requests finish, `prefill decision: ... mode=one-card` with
`chosen_by="... one card [stream, mix] ..."`, the answers read (judge), pp and decode near the one-card `stream, k x8` rows of the same size
(`rampool12-planner-learned-glm35-12k-rp1.server.prev.log:641-658`, 300 t/s at 1938 tokens).

Exact-kernel confirmation (no model needed, one card): run `compute-sanitizer --tool memcheck` on the server for the same cell with
`-c 4096` and the 1938-token request only; before the fix memcheck names `mul_mat_q<...>` with an out-of-bounds global read of `tile_spec`
or `ids_dst`. Alternatively `--moe verify=1` on the cell: with the fix the mix pass reads back clean.

Also worth one run after the fix: a forced dense+expert split (`--moe dense-split=1,prefill-split=1` on `glm35` 12k), the other user
of `-1` ids on MMQ since the merge.
