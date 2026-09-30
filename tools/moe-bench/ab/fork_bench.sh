#!/bin/bash
# fork_bench.sh tag [server args...]: Qwen3.6-35B-A3B, experts in RAM + small cache on GPU1; warm-up + 2 timed answers
tag=$1; shift
R=${R:-/tmp/moe-ab}; mkdir -p $R
cd /p/bw/llama.cpp/.claude/worktrees/partial-pin
LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_PREDICT_FILE=0 CUDA_VISIBLE_DEVICES=1 ./build-cuda/bin/llama-server \
  -m /m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf -ngl 99 --cpu-moe -nr --moe-expert-cache 64 -c 4096 --port 8231 "$@" > $R/fb_$tag.log 2>&1 &
for i in $(seq 180); do curl -sf localhost:8231/health >/dev/null && break; sleep 1; done
req() { curl -s localhost:8231/v1/chat/completions -d '{"messages":[{"role":"user","content":"Write a Python function that parses a CSV file and returns the average of each column."}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 "$(dirname "$0")/tps.py"; }
req > /dev/null
echo "$tag: $(req) | $(req)"
for i in 1 2 3 4 5; do pkill -x llama-server; sleep 1; pgrep -x llama-server >/dev/null || break; done
grep -E "hit rate|predicted top-k|router predict|learned router" $R/fb_$tag.log | tail -4 | cut -c1-220
