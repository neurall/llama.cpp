#!/bin/bash
# README table rerun on machine A (PC1), README protocol (run.py bench: t100 / chat / pf12k, n=3 ABBA, one discarded warm-up per cell), after the reboot and `sudo pool.sh mount 100G`.
# usage: readme-rerun.sh FINAL_BUILD_DIRNAME STOCK_BUILD_DIRNAME     (both subdirectories of /p/bw/rels)
#   FINAL = the release-recipe build of the final tree (clang), STOCK = upstream built the same way (release recipe, same upstream commit), b11707-49fe4b756 = the previous release
set -u
F=${1:?final build dir name}; S=${2:?stock build dir name}
cd /p/bw/wt-release-next; export PERF_BUILDS=/p/bw/rels
B=$S,b11707-49fe4b756,$F
C=readme-rerun
run() { python3 tools/run.py bench --models "$1" --builds ${3:-$B} --variants default -t "$2" -n 4 --campaign $C; }   # third argument: the builds (default all three)
R=$F   # cells that already have stock and b11707 numbers on the same prompts and machine (earlier today): the release only
GLM3=/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf
GLM35=/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf
MIMO=/m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf
IQ4=/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
IQ3=/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
IQ1=/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
run $IQ1 t100 $R      # picks stock: must equal stock
run $IQ3 t100 $R
run $IQ4 chat         # Qwen IQ4_XS short chat decode
run $IQ4 pf12k        # 12k prompt: processing and decode
run $MIMO chat        # MiMo short chat decode
run $MIMO pf12k
run $GLM35 t100 $R    # the slowest cells last: 109 and 137 GB loads
run $GLM3 chat $R
run $GLM3 t100 $R
python3 tools/run.py report --campaign $C --md > tools/experiments/release-gate/readme-rerun-table.md
echo README_RERUN_DONE
