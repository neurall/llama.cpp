# One-card decode against Strata: state and work list (2026-11-19, evening)

Model Qwen3.8-Flash-Next IQ3_S, one RTX 3090, PC1. Speed row `q1-hcload`, trace row `p1-allk`, both in `fork/runs/onecard.csv`.
Readers of the traces: `fork/nsys/` (README there).

## Where the token's time is (nsys, microseconds per token)

| | `rs_inplace` | `gdn_z_bus` | Strata |
|---|---|---|---|
| wall | 15,637 | 15,512 | 14,878 |
| work kernels per token | 2,647 | 2,538 | 1,837 |
| work kernels, summed | 13,991 | 13,092 | 12,944 |
| card busy with work | 12,530 | 12,247 | 11,207 |
| work running in parallel | 1,461 | 845 | 1,737 |
| no work running (host, idle, waits for the host's rows) | 3,107 | 3,265 | 3,671 |

"Work" leaves out the kernels that only wait (`k_moe_rows`, Strata's `wait_flag_ge`).

- The kernel work is the same as Strata's (13,092 against 12,944), in 700 more kernels. Fusion is not where Strata leads on this model.
- Strata's lead is work in parallel: 1,737 against 845. That is 892 of the 1,040 it is ahead in "card busy".
- We lead by 400 to 560 in the time with no work running (token boundary).
- The bus rule of build `gdn_z_bus` cut our own parallel work from 1,461 to 845: it moved the shared expert behind the routed chain. It is out of the
  planner again (HEAD).

## Why z (the delta net's gate projection, 21 us, 36 layers) does not overlap yet

1. `rs_inplace`: the graph's output pulls z in last, behind the delta net; the gated norm waited 18.4 us a layer.
2. `gdn_z_bus`: z entered the graph behind the conv. The wait is gone (2.3 us), but z ran in the main chain between the conv and the small kernels:
   delta-net layer median 235.4 -> 229.6 us only. z took no quantize of its own, so it was captured on the main stream (the Q8 reuse slots are per stream).
   Not confirmed by a plan dump yet; the likely cause is a memory conflict: the allocator gave z's output a buffer the conv chain had used, and two
   launches that touch the same memory keep their graph order.
3. Kernels of two streams did not interleave: the side stream's small kernels started 2 us before z's end although their input was ready 15 us
   earlier. A kernel with more blocks than the card holds keeps every SM until its last wave is handed out. HEAD gives the main stream the highest
   priority and instantiates the CUDA graphs with `cudaGraphInstantiateFlagUseNodePriority`. Untested until the `side_prio` trace is read.
4. The plan has one side stream. A unit planned behind z on the side waits for z's real end. The plan's estimate lets z end 12 us after the layer
   input; in reality z cannot start before the input projection's last wave. In HEAD the gate chain (alpha, beta, sigmoid, softplus) is expected to
   land behind z. Check in the `p1-plan` dump.

## Work list, biggest measured piece first

### 1. z and the gate chain beside the main chain (0.5 to 0.85 ms a token, 4 to 6%)

Files: `ggml/src/ggml-streams-fork.h` (`ggml_stream_plan_make`, pure, test `tests/test-streams-fork.cpp`), `ggml/src/ggml-cuda/ggml-cuda.cu`
(`ggml_cuda_stream_unit_fork` cost and ranges, `ggml_cuda_stream_movable_fork`, `sp_chunk` / `sp_before` / `sp_join` in
`ggml_cuda_graph_evaluate_and_capture`, `ggml_cuda_graph_instantiate_fork`, the priority in `ggml_backend_cuda_init`),
`src/models/qwen4exp.cpp` (`build_layer_attn_linear`: where z enters the graph).

Steps:
1. Read the `side_prio` results (chain started 18:20): plan dump (`p1-plan`, lines `CUDA0 streams:` of the server log in `/p/bw/data/q5h`), trace
   (`fork/nsys/layer_one.py`, `overlap.py`), speed pairs against `gdn_z_bus` and against `q1-noprio`.
2. If the main chain's small kernels now run while z has waves left: priority works. Keep it, default on.
3. Planner, as a pure function with a failing test first:
   - more than one side stream (Strata uses three streams): a unit that fills the card goes to a stream of its own, nothing queues behind it;
   - place by slack, not by "starts earlier": one pass for earliest starts, one from the output back for latest ends; the chain without slack stays
     on the main stream, the rest goes beside it. Greedy graph order cannot see that z is needed only at the gated norm.
4. Memory conflicts that are no data dependencies: count them in the dump (`after N (name)` names the launch a unit waits for). If they keep
   launches in line, decode-sized graphs get buffers that are not handed out twice (60 MB at most), or the conflicting node gets its own buffer.

Done when: trace shows work in parallel at or above 1,737 us a token and the delta-net layer median at or below 215 us; `q1-hcload` 3% or more
above `gdn_z_bus` in two pairs; `pd-hcload` perplexity unchanged (paired per chunk, as `reference_ppl_paired_check`).

### 2. The row's switches are not defaults (users do not get the row's speed)

`--moe bell=1,bell-pool=32,sync=0,hc-q8=1` and `GGML_CUDA_STREAMS=2` are still arguments of the row. Checked so far: Q8 reuse costs no quality
(-0.06 +- 0.02%), stays on; the HC fuse costs +0.20 +- 0.08% and gives no speed, off again (bug-log row). Open: `sync=0` (the cache step beside the
graph's tail: read `llama_moe_cache_step_overlap_fork`, race rows), the pool size computed from the cores (rows `q1-bp16/24/31` exist, not run),
then the defaults flipped and one row without any switch against the row with all of them.

### 3. The card waits for the host's experts (967 us a token in `k_moe_rows`)

`src/llama-bellpool-fork.h` (pool, phases under `LLAMA_BELL_TIME=1`), `bell_traits` in `src/llama-moecache-fork.cpp`. The AVX2 IQ2_S dot
(`src/llama-iq2s-avx2-fork.h`, test `tests/test-iq2s-avx2-fork.cpp`, bit-identical, not committed) is not wired in. Wire it with target attributes
and a runtime AVX2 check, A/B by `LLAMA_BELL_TIME` first, then the row.

### 4. Token boundary (1.54 ms of host time a token outside CUDA calls)

Rows `q1-steptime` (`LLAMA_STEP_TIME=1`, sections of the host time) and `p1-perf` (perf record of the server). `cudaMemGetInfo` runs every token
(56 us), caller not found yet. About 20 blocking input copies a token.

### 5. Launch geometry from the card's properties

Today: fixed tables per GPU family (`calc_nwarps`, `calc_rows_per_block` in `ggml/src/ggml-cuda/mmvq.cu`), and a tuner that is off unless
`--moe mmvq-tune=1` (`ggml_cuda_mmvq_tune_fork`: times 1/2/4/8 warps per matvec shape at load). Measured: the warp count moves a matvec 5 to 10% at
most (Q4_K 10240 rows: 22.7 -> 21.5 us). Rates in the decode trace: Q6_K 21.5 MB 863 GB/s (near the card's limit), Q4_K 8.85 MB 418 to 511,
IQ4_XS 580 to 710, BF16 2.6 MB 470, matrices under 1 MB 150 to 200 (launch latency, not the bus).

To do: one function from `cudaDeviceProp` (SM count, threads and blocks an SM holds, L2 size) and the shape to threads per block, rows per block and
blocks in flight, so enough loads are outstanding to cover memory latency; the tuner measures the candidates around that value and stores the result
per card and shape (hw key ini); cover the float matvec, the routed matvec, the quantize and the HC kernels, not only mmvq; on by default once it
wins. Small matrices gain nothing from geometry: they need fewer launches (item 6).

### 6. Fusion for every model (the user's goal: dynamic, no per-architecture hand work)

Measured basis on this model: about 1,400 kernels a token are small (norm, quantize, add/mul, unary, copy), 2.6 ms summed, 1.5 to 2.5 us each,
mostly launch latency. Expect 0.5 to 1.0 ms a token, measured by trace, not by t/s (the HC fuse took 485 kernels out and the token time did not
move; it also changed the numbers).

Levels, each generic (the matcher works on the ggml graph of any model):
1. Scheduling (exists): the planner works from launches and memory ranges, it knows no model. Items 1.3 and 1.4 make it complete.
2. One kernel that runs a short program per element (a list of ops with operand slots): chains of add, mul, scale, unary, clamp with broadcast.
   Matcher in `ggml_cuda_try_fuse`. No compiler needed.
3. Matvec kernels that take such a program for their input (norm scale, the quantize inside the kernel) and for their output (activation, residual
   add, gate multiply). Most small kernels sit right before or behind a matvec.
4. Runtime code generation (NVRTC) of the specialized kernel per chain signature, cached on disk per card; the interpreter of level 2 stays as the
   fallback. Check the size of the NVRTC library for the release package first.
5. One kernel per sublayer (several dependent stages with a grid-wide sync) only if the traces still show launch latency on the critical path.

Rule for every fused kernel: a `test-backend-ops` case, fused against the unfused graph, bit-equal where the op order is kept; the decode perplexity
pair before it becomes a default. Fusion decisions by measured time per launch (the backend's `op_prof`), not by op names.

## Traps met today

- An edit of `ggml/src/ggml-cuda/common.cuh` rebuilds every CUDA file (35 min). The priority code sits in `ggml-cuda.cu`.
- Stream ids of graph-launched kernels in nsys are the graph executor's. Use the plan dump to see the capture stream.
- A new build's prompt planner tries `k x1` .. `k x32` once each (3 to 6% slower prompt on those calls). Compare prompt speed on `stream` calls only.
- Decode speed of one cell varies about 1% run to run: decide by trace plus two pairs.
- The trace cell's server exits late: the list's pool unmount met "target is busy" (run.py now waits for the server).

## Results, 18:45 (build `side_prio`: main stream at the highest priority, graphs with node priority, z right behind qkv, no bus rule, HC fuse off)

| per token, us | `gdn_z_bus` | `side_prio` | Strata |
|---|---|---|---|
| wall | 15,370 | 15,084 | 14,878 |
| work kernels, summed | 12,956 | 13,756 | 12,944 |
| card busy with work | 12,119 | 11,903 | 11,207 |
| work running in parallel | 836 | 1,853 | 1,737 |
| no work running | 3,251 | 3,181 | 3,671 |

(Token window by the top-k kernel here; the first table's window was keyed on an HC fuse kernel.)

| decode, run 2, t/s | pair 1 | pair 2 |
|---|---|---|
| `gdn_z_bus` q1-hcload | 72.48 | 72.55 |
| `side_prio` q1-hcload | 71.40 | 71.94 |
| `side_prio` q1-noprio (`GGML_CUDA_STREAMS_PRIO=0`) | 70.09 | 70.53 |

- Priority works: +1.9% on the same build. In the trace z starts when the input projection hands out its last wave and the main chain's small kernels
  run during z, each about one wave late (concat 5.5 us instead of 2.4). The delta net starts 12.5 us earlier in a layer.
- The plan dump confirmed item 1's fourth point: with one side stream the k norm and the beta projection were planned behind z, and the gate chain
  stayed in the main chain. Also seen there: waits that are no data dependency (beta's projection waits for alpha's add, the shared expert's down
  projection for an HC product) because the allocator handed a buffer out twice. Item 1.4 stands.
- The HC fuse is worth 9 us a layer (unfused: two quantizes, scale, unary and two products, 25.1 us a stage against 20.6): 0.43 ms a token. My note
  "no speed gain" was wrong, it came from t/s while the token was host-bound. Rows `pd-bf16` (HC weights unquantized, the reference) and `pd-fuse`
  decide whether the fused kernel or the unfused graph is the one that is off.
- Done after this: planner with up to three side streams and the hold rule (`ggml-streams-fork.h`, test passes for 2, 3 and 4 streams), the scratch
  region safe for several side streams, a q8 reuse lane per stream. Build `multi_side`, rows `p1-plan4`, `p1-allk4`, `q1-s3`, `q1-s4`, `pd-s4`.

## Correction, 19:05: traced token times do not compare the two engines

| decode, t/s | without tracer | under nsys | tracer cost, ms a token |
|---|---|---|---|
| Strata 0.2.15, exact mode | 82.2, 81.6 | 67.2 in the traced window (cell average 71) | about 2.7 |
| Strata 0.2.15, default mode | 83.5 | 76.1 | |
| ours, `multi_side`, 4 streams | 74.0 (one cell) | 67.4 | about 1.3 |

Strata launches about 2,000 kernels a token through the API and nsys hooks every call; we replay captured graphs. So the "wall" and "no work running"
rows of the tables above flatter us: under the tracer the `multi_side` trace reads 14,814 us against Strata's 14,878, without it the tokens take
13.5 ms against 12.2 ms. The kernels' own times on the card and what runs in parallel inside one engine's trace stay usable; a comparison of the two
engines' walls or host times from traces does not. The real gap is 1.3 ms a token: the HC fuse (0.45 ms, off in `multi_side`) and host time between
tokens plus waits for the CPU experts (item 4 measured 1.54 ms of host work a token on our side). Measure those with the engine's own timers
(`LLAMA_STEP_TIME`, `LLAMA_BELL_TIME`), not with nsys.

Also in the run history of this row (4096-token prompt): Strata's prompt speed 1,970 to 1,990 t/s, ours 1,240 to 1,320. Not checked whether both
are measured the same way; the claim "we lead in prefill on one card" needs this row looked at.
