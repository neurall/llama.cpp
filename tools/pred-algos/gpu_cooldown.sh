#!/bin/bash
# GPU thermal guard: call before launching GPU work.
#   gpu_cooldown.sh [hot_threshold] [cool_target]
# If any GPU is >= hot_threshold (default 80C), waits (polling every 5s, logging every 30s)
# until ALL GPUs are <= cool_target (default 70C) before returning. Does not kill anything itself
# (callers should avoid launching new GPU jobs while this blocks); safe to call in a loop before
# each new job.
HOT=${1:-80}
COOL=${2:-70}
temps() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null; }
max_temp() { temps | sort -n | tail -1; }
t=$(max_temp)
[ -z "$t" ] && exit 0   # nvidia-smi unavailable, don't block
if [ "$t" -lt "$HOT" ]; then
  exit 0
fi
echo "gpu_cooldown: max temp ${t}C >= ${HOT}C, waiting for <= ${COOL}C on all GPUs" >&2
i=0
while true; do
  t=$(max_temp)
  [ -z "$t" ] && break
  [ "$t" -le "$COOL" ] && break
  if [ $((i % 6)) -eq 0 ]; then
    echo "gpu_cooldown: $(date +%T) still ${t}C, waiting..." >&2
  fi
  i=$((i + 1))
  sleep 5
done
echo "gpu_cooldown: cooled to ${t}C, resuming" >&2
