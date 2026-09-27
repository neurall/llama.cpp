#!/bin/bash
# parallel runner: par.sh "gpu|data|algo args" ... ; each job in its own process, results appended to results.txt
# pauses before each launch if a GPU is >= 80C, resuming once all GPUs are <= 70C (gpu_cooldown.sh)
cd "$(dirname "$0")"
for job in "$@"; do
  IFS='|' read -r gpu data cmd <<< "$job"
  ./gpu_cooldown.sh
  ( out=$(CUDA_VISIBLE_DEVICES=$gpu python3 $cmd --data $data 2>&1 | grep -E "generated only|Error|error")
    printf '## %s: %s\n%s\n' "$data" "$cmd" "$out" >> results.txt ) &
done
wait
