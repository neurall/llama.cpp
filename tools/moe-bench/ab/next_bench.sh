#!/bin/bash
# next_bench.sh tag [server args...]: Qwen3.8-Flash-Next, fork auto mode, both GPUs; warm-up + 2 timed 400-token answers
tag=$1; shift
R=${R:-/tmp/moe-ab}; mkdir -p $R
cd /p/bw/llama.cpp/.claude/worktrees/partial-pin
LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_PREDICT_FILE=0 ./build-cuda/bin/llama-server \
  -m /m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --port 8232 "$@" > $R/nb_$tag.log 2>&1 &
for i in $(seq 600); do curl -sf localhost:8232/health >/dev/null && break; sleep 1; done
req() { curl -s localhost:8232/v1/chat/completions -d '{"messages":[{"role":"user","content":"Write a Python function that parses a CSV file and returns the average of each column."}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 "$(dirname "$0")/tps.py"; }
req > /dev/null
echo "$tag: $(req) | $(req)"
for i in 1 2 3 4 5; do pkill -x llama-server; sleep 2; pgrep -x llama-server >/dev/null || break; done
grep -E "hit rate|predicted top-k|router prediction:|learned router" $R/nb_$tag.log | tail -3 | cut -c1-200
