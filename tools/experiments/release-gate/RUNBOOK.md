# Release runbook (final rerun of the README numbers)

Rule set (memory): never slower than stock, not consistently slower than the previous release (b11707), release recipe builds only, clang for the CPU backend on Linux and Windows if it is faster (it is: OLMoE CPU-only +8..16%), no OpenBLAS.

## Before the reboot (builds, nothing needs the pool)
1. Final tree: `cd /p/bw/wt-rc-up && git merge rc-merge` (rc-merge-up = rc-merge + upstream master), commit hash = FINAL.
2. Linux clang CUDA build: `W=/p/bw/wt-rc-up $CLAUDE_JOB_DIR/tmp/clang-build.sh`, then `cp -r build-clang/bin /p/bw/rels/final-$FINAL; cp -L cpu-clang libomp.so into it`.
3. Stock built the same way at the upstream commit the final tree merged (836d57176): docker release recipe (dev-build.sh with W=a worktree of origin/master) -> `/p/bw/rels/stock-docker-836d57176`.
4. Perplexity equality clang against gcc and stock (llama-perplexity, OLMoE or Qwen small), same text: PPL must agree to ~1e-3.
5. W10: build the final tree with clang CPU (`wllvm.bat` recipe, VS clang, `-Wno-error=incompatible-pointer-types`) + MSVC CUDA; run the laptop rows (Qwen3.6 Q2_0 t100, long) against the official win zips.
6. pc2: CPU-only rows (Qwen3.6 CPU, OLMoE) with the final clang CPU build.

## After the reboot (PC1)
1. `sudo /p/bw/wt-release-next/pool.sh mount 100G` FIRST, before any CUDA load; verify `HugePages_Free` ~100.
2. `python3 tools/reg_tests.py all /p/bw/rels/final-$FINAL` (regressions 1-4; 3 and 4 need the pool).
3. `tools/experiments/release-gate/readme-rerun.sh final-$FINAL stock-docker-836d57176` (README protocol, n=3 ABBA, ~2-3 h): the table lands in `readme-rerun-table.md`.
4. Gate: fork >= 0.98 x b11707 and >= stock on every cell, md5-equal text for cache cells (`LLAMA_MOE_CACHE_DETERMINISTIC=1`) when comparing against b11707.
5. Fill README (numbers from the table, protocol note: stock = release recipe), release notes (delete `~/.cache/llama.cpp/moe-state.ini` once), merge into the release branch, `RELEASE=1` packaging, then ask before pushing.
