#!/bin/bash
# release gate: the final build against stock on every model class, README protocol (run.py bench t100 / chat / pf12k), same prompts, ABBA, logged in run-history.csv.
# usage: gate.sh FINAL_BUILD_DIRNAME   (a subdirectory of /p/bw/rels, built from the final tree) ; stock = stock-def4d406a (README's upstream), README's fork build = b11707-49fe4b756
# PASS rule (checked by hand from `run.py report --campaign release-gate`): fork >= 0.98 x stock on every cell where it picked stock, >= README claim -5% where it picked cache.
set -u
F=${1:?final build dir name}
cd /p/bw/wt-release-next; export PERF_BUILDS=/p/bw/rels
B=stock-def4d406a,b11707-49fe4b756,$F
C=release-gate
run() { python3 tools/run.py bench --models "$1" --builds $B --variants default -t "$2" -n 2 --campaign $C; }
python3 tools/reg_tests.py all /p/bw/rels/$F                                  # numbered regressions 1-4 against the last good builds
run /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf t100                            # README rows: GLM 3.0 A, short
run /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf chat
run /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf t100              # Qwen IQ4_XS A
run /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf pf12k             # 12k prompt: prompt processing and decode
run /m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf t100          # picks stock: must equal stock
run /m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf chat                # MiMo, bigger than RAM
run /m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf pf12k               # README: processing 0.7x before
echo GATE_DONE
