# Branches: rules and audit (2026-10-22)

Local, untracked note. Written after the merge of `fast-up` and `neurall/release` into `release` and the upstream catch-up `up-merge-4`. Nothing here deletes or renames a branch.  

## Why

Two lines grew apart from `cdebac405` (2026-10-16): `fast-up` (`tokms31`'s base) and the published release line (`b13600`, `dynamic-ctx`, `neurall/release`). A comparison between builds of the two lines compared different code. Copies of `run.py` between the lines gave 51 conflicts, data files merged as text left conflict markers in `run-history.csv`, and a merge was built before it was compiled.  

## Rules

- **One trunk: `release`**, equal to `neurall/release` plus unpublished work. `git fetch neurall` before any merge or build; compare `release..neurall/release` first.
- **Features branch from trunk** and merge back small and often. No file copies between lines: `fork/run.py` and the fork tools live only on trunk.
- **Upstream catch-up** is a branch `up-merge-N` off trunk, merged from `origin/master` (small slices). It gets a compile check (PC2 dev build) and a smoke start before trunk moves to it.
- **Data files** (`run-history.csv`, `stats.ini`, benchmark CSVs, `runs.db`) live in the `data` and `runs-db` branches, not in merges of trunk.
- **Publishing order:** merge compiles, both OS builds, smoke start, then tag `release-b<build>-<hash>-<name>`; push to `neurall` only. The build number is the commit count of the line, so numbers of different lines are not comparable.
- **Dev builds** (`tokmsN`) come from trunk or a branch rooted on it; `BUILD_INFO` carries commit, branch and patch hash, and every `runs.db` row keeps them.

## State

| ref | commit |
|---|---|
| `release` (local) | `04acb1a89` |
| `neurall/release` | `27cc499ee` (contained in local `release`) |
| `up-merge-4` | upstream `7170cad4d` merged on top of `release`; ahead of release by 72 commits |

117 local branches: 62 are already contained in `release`, 55 are not.

## Already contained in `release` (safe to delete, nothing is lost)

`2stagingbuffs`, `bench-data`, `bug-fixes`, `catchup`, `chunk-pin`, `clean-509`, `cpu-topology`, `dynamic-ctx`, `dynkv`, `expert-parallel`, `fast-up`, `gpu-select`, `inplace_quants`, `iq3-fast-dots`, `knob-log`, `kv-grow`, `kv-tail`, `live-params`, `low-bw-cachemaxed-kernels`, `main`, `master`, `merge-test2`, `merge-up`, `merge-up-loader`, `mirror-pool`, `moe-cache-fork-rename`, `next-lr-predictor`, `no-mmap`, `no-mmap-fallback`, `noiq3-test`, `par-bench`, `pin-exit75`, `pin-load-fix`, `pin-tier`, `planner-mirror-fill`, `pr-28243`, `prefill-select`, `prerelease`, `prerelease2`, `prerelease3`, `prompt-cut`, `q4-k-accumulators`, `q8-0-accumulators`, `rc-fix`, `rc-merge`, `rc-release`, `rel-front`, `release`, `release-noup`, `run-build-smallcard3`, `sched-prof`, `split-calib-learn`, `stock-cuda`, `stock-master-620d2225c`, `stock-rebase`, `strided-copy`, `switches`, `tune-fixes`, `tuned-quants`, `up-8`, `up-catchup`, `up-merge-3`

Branches with a worktree among them cannot be deleted until the worktree is removed: `bench-data` (/p/bw/wt-release-next), `rc-fix` (/p/bw/wt-switches), `rc-release` (/p/bw/wt-rc), `mirror-pool` (/p/bw/wt-mir), `cpu-topology` (/p/bw/wt-topo), `inplace_quants` (/p/bw/wt-inplace), `low-bw-cachemaxed-kernels` (/p/bw/wt-lowbw), `split-calib-learn` (/p/bw/wt-calib), `tuned-quants` (/p/bw/wt-tuned-quants), `kv-tail` (/p/bw/wt-dynkv), `prerelease3` (/p/bw/wt-prerelease), `2stagingbuffs` (/p/bw/wt-2sb), `bug-fixes` (/p/bw/wt-kvgrow), `no-mmap` (/p/bw/wt-nommap), `pin-load-fix` (/p/bw/wt-pinfix), `up-merge-3` (/p/bw/wt-up3), `dynamic-ctx` (/p/bw/wt-dynctx), `noiq3-test` (/p/bw/wt-noiq3), `fast-up` (/p/bw/wt-main), `release` (/p/bw/wt-rel)

## Not contained in `release`

`ahead` = commits that `release` lacks. `pushed` = a branch of the same name exists on `neurall`. The last column is a suggestion from the numbers, not a decision.

| branch | ahead | last commit | pushed | worktree | suggestion | last subject |
|---|---|---|---|---|---|---|
| `data` | 1 | 2026-10-11 | yes | - | data branch, keep | data branch: run history as published on release (49 columns, append o |
| `kv-tail-glm-fix` | 1 | 2026-10-17 | no | /p/bw/wt-kvfix | has a worktree: owner decides | auto context: log the non-expert memory per card at 2k and 10k |
| `moe-reset` | 1 | 2026-10-11 | no | - | small old prototype, probably dead | wip: moe-reset agent state as left in the worktree (cache/state source |
| `no-predictor` | 1 | 2026-10-20 | no | /p/bw/wt-nopred | has a worktree: owner decides | Remove the router predictor: learned/early prediction, its graph nodes |
| `prerelease4` | 1 | 2026-10-19 | yes | /p/bw/wt-pre4 | effectively merged (its one commit was already in release); can go | dense split: not for a model that sits entirely in VRAM (token_embd st |
| `proto-ovl` | 1 | 2026-10-09 | no | - | small old prototype, probably dead | sched: GGML_SCHED_PROF=3 prints one line per graph that keeps the host |
| `readme-top` | 1 | 2026-10-17 | no | - | unmerged, no worktree: owner decides | README: the measurement tables move unchanged to fork/docs/perf-histor |
| `runs-db` | 1 | 2026-10-20 | yes | - | data branch, keep | runs.db: 15187 runs, 2026-10-20 11:44 (only commit of this branch, rep |
| `up-merge-1` | 1 | 2026-10-10 | no | /p/bw/wt-upmerge | has a worktree: owner decides | merge upstream: the 40 commits before the MoE cache (up to d6cf9acb2^, |
| `up-release` | 1 | 2026-10-14 | yes | /p/bw/wt-up | has a worktree: owner decides | run.py: measured runs of our builds always pass --load-mode pin (serve |
| `upstream-sched-barrier` | 1 | 2026-10-11 | no | /p/bw/build-tmp/upstream-pr | has a worktree: owner decides | ggml-backend: skip the split barrier when the previous split read noth |
| `dense-ring` | 2 | 2026-10-17 | no | /p/bw/wt-dense-ring | has a worktree: owner decides | log the measured compute scratch per token at the graph reserve (input |
| `h-iq3` | 2 | 2026-10-13 | no | - | small old prototype, probably dead | ggml-cpu : cast the IQ3_S sign shift in the AVX branch too, align spre |
| `make-api-agnostic` | 2 | 2026-10-14 | yes | /p/bw/wt-agn | has a worktree: owner decides | llama-moe-backend.h: llama.cpp naming (llama_moe_backend_<action>_<nou |
| `noup-tunefix` | 2 | 2026-10-09 | no | - | small old prototype, probably dead | context: dynamic ubatch (LLAMA_UBATCH_DYN=<generation ubatch>, LLAMA_U |
| `proto-chunk` | 2 | 2026-10-09 | no | - | small old prototype, probably dead | moe: --moe chunk=N runs the whole expert block (chain, bias, weighting |
| `release-pfslots` | 2 | 2026-10-10 | no | - | small old prototype, probably dead | pf-slots default auto: --prefetch-experts-slots / --moe pf-slots defau |
| `release-pipe` | 2 | 2026-10-10 | no | - | small old prototype, probably dead | pf-slots default auto: --prefetch-experts-slots / --moe pf-slots defau |
| `moe-phase-buffers` | 3 | 2026-10-10 | no | - | small old prototype, probably dead | moe-cache: borrowable tail. The cache buffer of the last layers of the |
| `moe-tail-borrow` | 3 | 2026-10-10 | no | /p/bw/wt-relpfslots | has a worktree: owner decides | moe-cache: borrowable tail. The cache buffer of the last layers of the |
| `prefillv2` | 3 | 2026-10-19 | no | /p/bw/wt-pv2 | has a worktree: owner decides | pass planner: two staging slots are a plan (cost max + half the smalle |
| `prerelease-tune` | 3 | 2026-10-09 | no | - | small old prototype, probably dead | self-tune: the cycle starts at once when the window hit rate (256 step |
| `proto-ovl2` | 3 | 2026-10-09 | no | - | small old prototype, probably dead | sched: the prefetch uploader thread is for pageable (mmap) weights onl |
| `review/engine` | 3 | 2026-10-11 | no | - | small old prototype, probably dead | wip: engine review notes REVIEW1-3 |
| `sched-serial` | 3 | 2026-10-10 | no | - | small old prototype, probably dead | sched: GGML_SCHED_SERIAL=1 diagnostic, a split is waited for before th |
| `509-clang-fixed` | 4 | 2026-10-08 | no | - | unmerged, no worktree: owner decides | tinyBLAS Q8_0 gemm4xN: opaque delta vector (clang folded permute2f128  |
| `noup-phases2` | 4 | 2026-10-09 | no | - | unmerged, no worktree: owner decides | self-tune: the cycle starts at once when the window hit rate (256 step |
| `noup-phases` | 7 | 2026-10-09 | no | - | unmerged, no worktree: owner decides | dynamic ubatch (opt-in): the compute buffer is predicted from two prob |
| `pr-29761` | 7 | 2026-10-01 | no | - | unmerged, no worktree: owner decides | cont : clean-up comments |
| `proto-pipe-v1` | 7 | 2026-10-09 | no | - | unmerged, no worktree: owner decides | sched: ggml_backend_sched_prefetch_hint, resident slots filled ahead f |
| `series` | 7 | 2026-10-11 | no | - | unmerged, no worktree: owner decides | fork/: benchmark harness, run log, docs, experiments, build recipe; RE |
| `prefill-pipe` | 8 | 2026-10-12 | no | /p/bw/wt-pipe | has a worktree: owner decides | prefill plan: card set from the search, layer shares from the upload r |
| `proto-passes` | 8 | 2026-10-09 | no | - | unmerged, no worktree: owner decides | n passes: no pass mode when layer inputs are read out (a range graph h |
| `moe-chunk` | 9 | 2026-10-09 | no | - | unmerged, no worktree: owner decides | moe: --moe chunk=N runs the whole expert block (chain, bias, weighting |
| `base-509` | 10 | 2026-10-08 | no | - | unmerged, no worktree: owner decides | moe-cache: dynamic lookahead (LEAD, LEAD_SPAN, PREDICT_FROM), hand-por |
| `fork-layout` | 10 | 2026-10-08 | no | - | unmerged, no worktree: owner decides | experiments: section 17, regression hunt of 8 Oct (cache path, shared  |
| `backup-before-trailer-strip-prefill-select` | 11 | 2026-10-12 | no | - | unmerged, no worktree: owner decides | bench: run history, campaigns and gpu logs of the prefill-select sweep |
| `dyn-ubatch` | 11 | 2026-10-08 | no | - | unmerged, no worktree: owner decides | context: the dynamic ubatch is calculated, not picked from two numbers |
| `learn-cache` | 13 | 2026-10-11 | no | - | unmerged, no worktree: owner decides | wip: soak.py change left in the worktree |
| `admit-pred` | 19 | 2026-10-11 | no | - | unmerged, no worktree: owner decides | wip: llama-context.cpp change left in the worktree (6 lines) |
| `nilo85-b70` | 26 | 2026-10-02 | no | - | unmerged, no worktree: owner decides | ... |
| `backup-before-trailer-strip-expert-parallel` | 29 | 2026-10-13 | no | - | unmerged, no worktree: owner decides | prefill plan: expert-split verdict per card (compute taken over vs act |
| `qkv-ram-fallback` | 29 | 2026-10-18 | no | /p/bw/wt-qkv | has a worktree: owner decides | kv-tail: the zone always covers the attention's padded read (cells rou |
| `auto-kv-ram-quant-fallback` | 30 | 2026-10-17 | no | /p/bw/wt-kv-fit | has a worktree: owner decides | kv-fit: preallocate the quadratic lend, monotone with headroom, linear |
| `fix-pred-stock` | 37 | 2026-09-27 | no | - | unmerged, no worktree: owner decides | pred-algos: low-rank NLMS (shared projection per source layer, NLMS he |
| `pred-stock` | 37 | 2026-09-27 | yes | - | unmerged, no worktree: owner decides | pred-algos: low-rank NLMS (shared projection per source layer, NLMS he |
| `proto-pipe` | 56 | 2026-10-10 | no | - | unmerged, no worktree: owner decides | n passes: --moe p2p=1: the residual stash goes card to card with a pee |
| `proto-pipen` | 57 | 2026-10-10 | no | - | unmerged, no worktree: owner decides | bench: xgpu_lat.cu peer-access option (XGPU_PEER=1) for the cross-card |
| `pipe-qwen-next` | 58 | 2026-10-10 | no | - | unmerged, no worktree: owner decides | sched prefetch slots: size by the device buffer type's alloc size (row |
| `pf-slots-auto` | 59 | 2026-10-10 | no | - | unmerged, no worktree: owner decides | pf-slots default auto: --prefetch-experts-slots / --moe pf-slots defau |
| `up-merge-4` | 72 | 2026-10-22 | no | /p/bw/wt-up4 | ALIVE: upstream catch-up, compile check on PC2 running | Merge remote-tracking branch 'origin/master' into up-merge-4 |
| `tune-fix` | 364 | 2026-10-08 | no | - | old line (diverged before 10-08), probably dead | self-tune: the swap margin default is the pay-back margin again (-2, b |
| `h-mmvf` | 398 | 2026-10-02 | no | - | old line (diverged before 10-08), probably dead | cuda: mmvf v16 bf16 unpack builds on pre-Ampere / CUDA 12.0 |
| `predictor-auto-layer-offset` | 402 | 2026-10-02 | no | /p/bw/llama.cpp | old line (diverged before 10-08), probably dead | bench: backfill.py (stats.csv from stats.ini, branch/patch for earlier |
| `x-release` | 439 | 2026-10-06 | no | - | old line (diverged before 10-08), probably dead | docs: 4090D GSQ-RCO-Q4 权重预取测试组录入 README 与双语网页（16 槽+2 预取/ub32768/1M，16k |

## Next

1. You mark each unmerged branch keep / dead.  
2. Dead and merged branches get a tag `archive/<name>` on the tip, then the branch is deleted (the commits stay reachable by the tag).  
3. `fast-up` is retired as a name after the release is published; new work branches from `release`.  

