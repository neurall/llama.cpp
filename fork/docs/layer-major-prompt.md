# Layer-major prompt (design, gain plan item 2)

Goal: a long prompt reads the experts the RAM pool does not hold from the drives once, not once per pass. Today a 12558-token prompt of Kimi K3.1 runs as 3
passes of 4186 tokens (4 on 2026-11-20, 270 s), and every pass reads the same 290 GiB again (57.7 s at 5.4 GB/s). Layer-major: the call's tokens are cut into
k chunks, and each group of layers runs all k chunks before the next group starts. The plan and its time are pure functions in `src/llama-layermajor-fork.h`
(tests: `tests/test-layermajor-fork.cpp`); this file is the engine design. Switch for phase 2: `--moe layer-major=1` (forced; off by default until measured).

Terms. A **chunk** is a slice of the call's tokens (what a pass is today). An **entry** is the stream's cut of a layer's experts into parts that fit the ring
(the code calls them chunks: `cs_chunks`, `--moe cs-chunks`); here they stay entries. A **group** is a run of consecutive layers on one card.

## The numbers it is built on

| model | layers | expert GiB a layer | in RAM pool | from the drives a pass | hidden state a token (f32) | recurrent layers | sliding window |
|---|---|---|---|---|---|---|---|
| Kimi K3.1 IQ1_S | 93 (1 dense, 46 + 46 a card) | 5.52 | 217.9 of 507.9 GiB | 290 GiB | 28 KiB and 28 KiB a residual checkpoint (1-8): 56-252 KiB | 69 (KDA) | no |
| OLMoE-1B-7B Q4_K_M | 16 | 0.21-0.24 | (row setting) | (row setting) | 8 KiB | no | no |
| Qwen3.8 Flash Next | 48 | - | - | - | 40 KiB (4 hyper-connection copies of 2560) | yes | no |
| GLM 5.3 Flash | 45 (3 dense) | - | - | - | 64 KiB (4 x 4096) | yes | no |
| DeepSeek V4.1 Flash | 40 | - | - | - | 80 KiB (4 x 5120) | yes | 128 |
| MiMo V2.6 Flash | 48 | - | - | - | 16 KiB | no | 128, most layers |

K3.1 card numbers (server logs of 2026-11-20): pool 1056 blocks of 6.15 MiB a card; a graph lends 1.1248 MiB a token on CUDA0 and 1.0597 on CUDA1 at 12558
cells; the stream's caps measured at 12558 cells: 4160 tokens with a layer in 6 entries, 4352 with 8; links 26.5 and 26.6 GB/s; compute 5814 us a token
measured, 1651 calculated (task #303: the measurement is in doubt). The pool beside the KV zone and the reserves holds 5614 MiB a card (from the 6-entry cap:
4160 x 1.1248 MiB and one entry of 150 experts and two blocks): **less than one layer (5653 MiB) even with no graph at all**, so a K3.1 layer can never stay on
the card between chunks.

## Design

### Layer groups (what fits the ring)

At chunk size C a card's ring is `free_b - lend(C)` (`lm_ring_b`: the pool bytes left by the KV zone at the call's end cells and the fixed reserves, less the
lend line of a C-token graph, `ppx_lend_line`). `lm_groups` walks the layers in order; a group never spans two cards, and a layer without bytes to stream
(dense, or all its experts on the card) joins the group it falls in. The first layer with experts decides the kind:

* **held, next beside** (`LM_HELD2`): the group's uploads fit half the ring, so it stays on the card for all k chunks (uploaded once) and the next group's
  uploads land beside it while it computes. It grows while twice its bytes fit.
* **held** (`LM_HELD`): it fits the ring but not twice: uploaded once, the next group's uploads wait for its last chunk (time: uploads + compute).
* **streamed** (`LM_STREAMED`): one layer is larger than the ring. Every chunk streams it again in entries (ceil(layer / ring), at most the stream's 8),
  from RAM: its drive bytes stay in the RAM pool until its last chunk. It grows while twice its drive bytes fit the hold (its own reads and the next group's
  read ahead); one whose drive bytes alone do not fit is read again by every chunk (`reread`: no gain from the drives for it).

K3.1 at every chunk size: 92 streamed groups of one layer (2 x 3.15 GiB of drive bytes in a 7.5 GiB hold). OLMoE on the test machine (1.2 GiB pool, 0.3 MiB a
token): 16 groups of one layer held-next-beside at 2048-token chunks, 8 groups of two at 512.

### Chunk size

One k for the call: k equal chunks of ceil(n / k) tokens, every group the same (`lm_pick`). Larger chunks mean fewer graphs and fewer re-uploads of streamed
groups; smaller chunks a larger ring (held groups, fewer entries). `lm_pick` prices every k from the fewest the cap allows to one chunk per 32 tokens (the pass
granule) and takes the fewest chunks within 1% of the lowest time (the rule of `ppx_pick_chunks`). The cards' own limits come out of `lm_groups` (a chunk
whose lend leaves no ring, or a layer in more entries than the stream cuts, cannot run): for K3.1 the 8-entry cap 4352 follows from the 6-entry one, as measured.

A chunk count per group (each group its own k) is possible: the KV and the recurrent state are per layer, so a layer only needs its own chunks in order. It pays
only where layers differ (dense lead layers, two size classes, two cards with different lend lines); it is a DP over the same per-group times. Not in v1.

### Hidden state between groups

Between two groups the call's whole hidden state is parked: n tokens x the state a group starting there reads. For most models that is the residual stream
(n_embd floats), for hyper-connection models n_hc copies of it (Qwen3.8, GLM 5.3, DeepSeek V4.1), for K3.1 the residual stream plus the residual checkpoints
banked so far (`attn_res.block_size` 12: one at layers 0, 12, ..., 84). K3.1 at 12558 tokens: 0.66 GiB from layer 1 on (one checkpoint), 3.0 GiB from
layer 85 on (eight).

v1 keeps it in pinned host RAM (one buffer per call, the size of the largest boundary; for K3.1 3.0 GiB taken from the RAM pool's budget): a group graph reads
its chunk's rows as an input (the pattern of `llm_graph_input_embd` / `embd_h`: rows set before the graph) and its output rows are copied back after it (the
pattern of the layer inputs of draft models, `build_layer_inp_dev`). The read crosses the link beside the experts and is priced (K3.1, 3 chunks: 6.6 s of the
68.3 s link part); the write goes the other direction and is free. No new VRAM tenant (the rule: no VRAM outside the pool). Keeping it in VRAM (a region of
the pool for the call, beside the KV zone) saves the link part but costs ring: an option once the estimator shows the link part matters.

### Attention, KV and recurrent state, in causal order

The call's memory context is made once (`init_batch`: every chunk's cells found up front, as today). Group g, chunk c:

* attention layers write K/V of chunk c's cells and read cells [0, end of chunk c): chunks before c were written by the earlier graphs of the same group, so
  every layer sees exactly what a pass sees. With chunks of the pass size the results should be bit for bit those of the passes (same batches, same KV, f32
  state, the same entries a layer); a kernel fusion across a group's edge is the one thing that may differ, and the first engine check names it.
* recurrent layers (KDA, DeltaNet) read the state the layer's chunk c-1 left and write its own: each layer's state is its own tensor, so the order across layers
  does not matter, only the chunk order within a layer.
* the first group applies the memory context chunk by chunk as today. Later groups must see chunk c's view again without applying it: `apply()` (its
  `find_slot`) moves the recurrent memory's head and copy sources, reading the copy sources resets its rollback index, and the KV's cell count per chunk comes
  from the cells used at apply time. So the
  memory contexts get a replay: the KV context keeps each chunk's `n_kv` and `n_kv_rows` and goes back to chunk c (`i_cur`); the recurrent context keeps each
  chunk's head, zeroed cell and copy sources from its first apply; the hybrid contexts pass it on.
* sliding-window layers reuse cells inside one call when the call is longer than their ring: a replayed chunk would find its cells' positions overwritten by
  later chunks. v1 refuses layer-major for such calls (MiMo, DeepSeek V4.1); later: a chunk's mask from its first apply, or a ring that holds the call.

### KV tail and the lend

* KV tail: the zone grows to the call's end cells before `init_batch` (`llama_moe_cache_kv_tail_grow_fork`), as for any call. With layer-major the call is the
  whole prompt, so the zone holds all n cells from the first graph (today it grows call by call). It does not move during the call, so no group graph is built
  again for a moved zone. The chunk cap is computed at the call's end cells, as the passes' cap already is (`ppx_cut_last`).
* lend: every chunk graph of a call has the same size and lends the same bytes, so the lend's reach is fixed for the call and a held group's blocks lie behind
  it (`lm_ring_b` is exactly that room). A group graph lends about what a full graph does per token (one layer's temporaries are the peak), plus its chunk's
  hidden-state input and output.

### One stream per call (the stream and the RAM pool readers)

Today `cs_begin` runs at every graph (cursors to 0, statistics), `cs_end` at the call's end; the order of entries is the graph's. With group graphs:

* held groups upload once with the stream as it is: the call keeps it live between its graphs (`g_cs_live`), so chunk 1's plan finds the group's experts in
  their blocks (`cs_plan` plans only experts without a block) and uploads nothing. What they lack is the overlap: the stream runs ahead only inside a graph, so
  the next group's uploads start with its first graph (`LM_HELD2` then costs what `LM_HELD` does).
* the order the stream runs ahead in (`cs_ahead`) and the read-ahead walks (`cs_read_ahead`), both over `cs_order`, becomes the call's: during group g's chunks
  the ring takes group g+1's uploads where they fit and the readers read its drive bytes, up to the hold. Without it the drives idle during chunks 1..k-1 of
  every group. Every rule that asks whether a layer is in the pass (`cs_pos`, the first graph's evictions in `cs_begin`) asks the call.
* a streamed group's RAM blocks go back to the pool (`cs_release_pool` -> `ggml_rampool_done_fork`) only after its last chunk; every chunk's `need` stamps
  them again, so the pool's "until the graph after next" rule keeps them between its chunk graphs.
* evict processed only: a layer is processed after its last chunk. Until then a held group's blocks are pending; the ring takes blocks of processed groups first
  (the rule as today, per call instead of per pass). The pass line (`cstream: pass N: ... of layers ahead`) stays the check, one line per group.
* one SSD stream: the drive reads 290 GiB once for K3.1 (the estimate's drive part equals one pass's).

### When it does not pay (fallback)

The planner prices layer-major beside today's passes for every prompt call and takes it only when it is faster (decide by total time). It never pays for a call
of one pass (same streams, more graphs), nor when nothing streams (every expert on the cards). Refused outright, with the reason in the decision line: an arch
without the layer window (v1: OLMoE only), sliding-window calls longer than the window's ring, a call with more than one sequence or with embeddings or several
outputs, a forced dense split or mix. The old path stays as it is (keep winning modes).

The server's checkpoint cut: for recurrent and sliding-window models it cuts the last pass off as its own call so a checkpoint lands before it (`ppx_cut_last`).
That call is a second stream: layer-major then gives two streams instead of k. K3.1 13k: 8372 tokens layer-major (61-72 s) and 4186 as one pass (57.7 s),
119-130 s against 173 s. The whole gain (74-97 s) needs the checkpoint inside the call: a chunk edge at the checkpoint position, and each layer's recurrent
state copied out when that chunk has passed it. The server sends the request as one call (only the 4-token tail cut off) when the plan is layer-major and
n_batch holds it; a longer request is one stream per n_batch call.

### The estimator (seconds for the call)

`estimate_layermajor_time(in, k)`: per group the longest of its drive reads (its drive bytes once; once per chunk if reread), its link part (its uploads once if
held, once per chunk if streamed, plus the chunks' hidden state in) and its compute (n tokens x its layers' share of a token, plus a streamed layer's further
entries as the passes price them), uploads and compute added for `LM_HELD`; summed over the groups; plus every graph's host cost (k x groups). Against it
`lm_passes_time(in, cap, extra)`: today's equal passes from the same inputs through `ppx_stream_us` and `ppx_stream_pass_us` (no second copy).

Inputs, all measured or computed by the engine: per layer its card, the bytes without a block on it and the part only the drives have (`pp_expert_bytes` per
layer), the hidden state a token at its start (from the arch); per card the pool room at the call's end cells, the lend line, the link rate; the drives' rate
(`pp_ssd_rate_gbs`), the compute per token at the call's cells (`g_pp_c_us + a x cells`), the further entry's cost (`CS_ENTRY_COST`), one graph's host cost
(measured on group graphs; before the first one the prompt graph's build and input time per layer), the RAM pool's hold (its free and probation bytes), the
stream's most entries (`cs_chunks_max`). Every call logs the pick and the refused option with these numbers (`llama_moe_decision_fork`), and the measured call
time goes to the estimator log beside the estimate.

| case (estimator, the test's numbers) | today's passes | layer-major | |
|---|---|---|---|
| K3.1 12558 tokens, one drive, compute 5814 us a token | 3 passes, 173.0 s | 4 chunks of 3140, 97.3 s (drives 57.7, link 88.8, compute 87.5, graphs 7.4) | 1.78x |
| K3.1 12558 tokens, one drive, compute 1651 us a token | 3 passes, 173.0 s | 3 chunks of 4186, 73.8 s (drives 57.7, link 68.3, compute 31.0, graphs 5.5) | 2.34x |
| K3.1 12558 tokens, two drives (10.8 GB/s), 5814 us | 3 passes, 124.1 s | 4 chunks, 97.3 s | 1.28x |
| K3.1 1000 tokens | 1 pass, 57.7 s | 59.5 s | passes |
| OLMoE 8192 tokens, 2.6 of 3.6 GiB from the drive | 4 passes, 2094 ms | 4 chunks, 16 held groups, 654 ms | 3.2x |
| OLMoE 8192 tokens, every expert in RAM | 4 passes, 587 ms | 620 ms | passes |

Against the 4 passes K3.1 ran on 2026-11-20 (270 s) the same estimates are 2.8x and 3.7x; against the 3 passes of the fixed cut 1.8-2.3x, not 3x: a K3.1 layer
does not fit a card's pool, so every chunk uploads it again (20.5 s a chunk over one link), and once the drive is read once the link and the compute are the
bounds. What lifts them: both links (the expert split inside layer-major: each card streams half of every layer), the attention side file (item 6: 3-4 GiB more
pool a card, k 2 instead of 3), and the second drive helps only together with those.

## Pure functions (phase 1, committed)

`src/llama-layermajor-fork.h` (new; nothing includes it yet, so no rebuild beyond its test): `lm_ring_b`, `lm_groups`, `estimate_layermajor_time`, `lm_pick`,
`lm_passes_time`. `tests/test-layermajor-fork.cpp`: K3.1 and OLMoE as above, the group kinds, the drive part once, the link part per chunk, the 8-entry cap,
the refusals (one pass, nothing to stream, every expert in RAM).

## What the scheduler and graph code need (phase 2)

No existing way builds a graph for a layer range. The nearest: the dense split cuts the tokens of a pass between cards (all layers, `llama-dsplit-fork.cpp`),
layer parallel lets the cards take turns inside one graph, the prompt chain steps per layer through the scheduler's split hooks (`cs_split_cb`), draft
models copy chosen layers' inputs into a device buffer (`build_layer_inp_dev`, `set_rows`) and DFlash reads another context's buffer as its input
(`embd_rows`, `get_rows`), `llm_graph_input_embd_h` takes a hidden state per token from the batch, and control vectors have a layer range (values only).

* graph: a layer window [a, b) and the hidden-state rows in `llm_graph_params`; `llm_graph_context` helpers for the window and for the state's input and
  output; each arch opts in (loop bounds, state in at a > 0, state out at b < n_layer, the output head only at b = n_layer). OLMoE: about 5 lines. K3.1: the
  checkpoints as part of the state. Hyper-connection models: n_hc x n_embd. A graph test on the tiny model: a call as groups and chunks gives the logits of the
  same call as passes, bit for bit, on the CPU backend.
* scheduler: nothing. A group graph is an ordinary graph; its expert splits carry the names the stream hooks read.
* memory: the replay of a chunk's view (KV context, recurrent context, hybrid contexts), see above.
* context: the loop in `llama_context::decode` runs groups outer, chunks inner over the call's ubatches, copies the hidden state rows, takes the outputs
  after the last group only. A refused graph in the middle fails the call (a recurrent memory cannot take back half a call): the plan only picks sizes its
  own cap admits.
* server: when the plan is layer-major the request goes as one call (only the 4-token tail cut off; see the checkpoint cut above), so the prompt is one stream.
* header edits: `llama-graph.h`, `llama-memory.h`, `llama-kv-cache.h`, `llama-memory-recurrent.h` and the hybrid memories' headers: about 180 libllama files
  recompile (the model files include `llama-graph.h`); no ggml file, no nvcc.

Order (each step a commit with its test, OLMoE first; rows go through run.py after each, answers before speeds):

1. memory replay, KV context; test of the replayed inputs.
2. layer window and hidden-state input/output for OLMoE; the bit-for-bit graph test.
3. the decode loop behind `--moe layer-major=1`, state in RAM, the stream per graph as today (held groups already upload once).
4. one stream per call: the call's order for the read-ahead, the RAM blocks of a streamed group kept to its last chunk, held blocks pending to the last chunk.
5. the planner's candidate: the estimator wired in, the decision line, the estimator log.
6. K3.1: checkpoints in the state, the KDA replay, the checkpoint inside the call; then the hyper-connection models; then sliding windows.

First rows: OLMoE restricted so its prompt streams (sim-vram, sim-ram), `layer-major=1` against `=0` on one build: answers identical, the drive bytes of a
prompt once, the call time against the estimate. Then K3.1 13k against the passes.

## Not checked

* The compute per token of K3.1 in a stream pass (5814 measured against 1651 calculated) decides between 1.8x and 2.3x; layer-major shows the compute that the
  passes hide under their drive reads.
* One group graph's host cost (20 ms assumed for K3.1, 2 ms for OLMoE): the mask of an attention layer is filled for every group graph that has one; measure on
  the first rows.
* The hold (7.5 GiB, the pool's probation blocks of 2026-11-13) is not today's number; the engine reads it from the pool.
* Weight reads of small chunks (fewer tokens than a layer has experts per pick) are not in the estimate (shortcut in the header).
* The checkpoint inside the call, the sliding-window replay and the hyper-connection state are designed here, not prototyped.
