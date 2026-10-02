# Next release: what goes in, in what order (2026-10-03)

State: remote `release` is `f4d1d219e` (PR #1 bf16 fix, and a README edit made on GitHub). Local `release` is `59f5a6bc2`: 3 commits not pushed (protocol-v2  
harness, experiment records) and 2 behind the remote.

## What is worth announcing

| order of value | feature | evidence | branch |
|---|---|---|---|
| 1 | faster loading of pinned weights: hybrid huge pages (no setup), the 1 GiB pool and a cache file that outlives the process (`pool.sh`, `GGML_CUDA_HUGEFS`) | GLM 109 GB: 91 s to 73 s to ready with the hybrid alone (ABBA n=4); warm cache 34.5 s against 56.6 s for the pool path, same perplexity (ABBA n=4, state off) | `load-pinned-multithread` |
| 2 | every flag, `--moe` setting, knob and variable documented with an evidence label; `--moe name=value` for any setting; how to run stock | written; the `--moe` generalisation compiles (pc2) but has no GPU run | `docs-moe-switches`, `moe-switch-any-setting` |
| 3 | the cache is not tried when it cannot fit, a placement that could not start is remembered | compiles (pc2); GPU runs of the auto placement at 10% and 20% share are in `tools/bench` when finished | `placement-low-share-guard` |
| 4 | upstream sync (41 commits) | compiles on pc2 only; no CUDA build, no speed claim | `merge-upstream-2026-10-02` |

Not announced: the predictor and cache-policy work (hit rate up, no speed gain), `pread` uploads (16% slower on prompts in our test), the shared slot pool  
(dropped), the chunked register overlap (crashes at the first prompt batch, withdrawn), P2P (not needed: the host-staged hop of 39 us is 40-65% of the per-layer  
expert time, a 5-9% ceiling even with P2P).

## Merge order and the check for each step

The first four branches touch disjoint files, so they merge without conflicts among themselves; the upstream merge touches `common/common.cpp`,  
`common/common.h` and `README.md` as well, so it goes last and is redone on top of the others.

0. **Sync.** Fetch the remote; merge it into local `release` (the README edit); push the 3 local commits (a decision for the owner, nothing is pushed by the plan).
1. `moe-switch-any-setting` (`common/arg.cpp`, `src/llama-moecache.cpp`). Gate: dev build on PC1; `--moe policy=lru` logs the policy; default run unchanged (same perplexity).
2. `load-pinned-multithread` (`ggml-cuda.cu`, `llama-model-loader.cpp`, `llama-model.cpp`, `pool.sh`). Gates: **a Windows build on W10** (the POSIX includes are guarded for Linux  
   only; unverified until it builds), perplexity equal with and without the cache (done on PC1: 6.6026 and 3.7227), a run without any pool or mount (the hybrid fallback), `pool.sh mount` then `unmount` leaves no reserved pages.
3. `docs-moe-switches`. Rebase it on the remote README first (both edit `README.md`); the pool section describes step 2, so it goes after it.
4. `placement-low-share-guard`. Gates: GPU runs of the auto placement at about 10% share (expected: stock chosen without a trial, no out-of-memory) and at about 20% (cache), and the laptop.
5. `merge-upstream-2026-10-02`, redone on the result. Gates: CUDA build on PC1 and W10, perplexity equal to the previous release on GLM and Qwen, decode against the previous release and stock (protocol v2, the stock baseline also with `--no-op-offload`, see below).

Before announcing: run the regression checks against the previous release and stock on PC1 and the laptop (every comparison includes upstream stock).

## Open findings that may change the story

- `--no-op-offload` changes stock a lot at low share: at about 10% share (GLM, one 3090, 10.9 GiB free) real stock gave 4.4 t/s by default and 9.5 with the flag in the first round (one run); the fork's stock mode went from 4.3 to 5.0.  
  If it holds over the remaining rounds, the stock baseline for every comparison must include it, or the headline gains are against a handicapped baseline.
- The cache runs out of memory at 10.9 GiB free on GLM and MiMo; the viability check in the guard branch avoids it, but a cache that can start with less VRAM would extend the range.
- On the laptop the same forced-cache configuration drifted from 16.1 to 10.6 t/s within a session; no laptop number goes in the announcement until it is measured interleaved.
