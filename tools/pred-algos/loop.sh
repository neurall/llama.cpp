#!/bin/bash
# Repeat an algo in --time-limit runs (each resumes its checkpoint), appending results to loop_<name>.log.
#   loop.sh <name> <gpu> <start offset s> <algo.py args...>
# Stop: touch stop_<name> (the current run finishes and saves first).
name=$1 gpu=$2 offset=$3; shift 3
cd "$(dirname "$0")"
rm -f "stop_$name"
sleep "$offset"
r=0
until [ -e "stop_$name" ]; do
  ./gpu_cooldown.sh
  { echo "== $name round $r $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$@" 2>&1 | grep -E "generated only\]|this run\]|same tokens|recall rate|Error|Traceback"
  } >> "loop_$name.log"
  r=$((r + 1))
done
