#!/bin/bash
# glm_bench.sh tag [server args...]: GLM-5.3-Flash q4kattn, fork auto mode, both GPUs; warm-up (WARM tokens) + 2 timed answers (N tokens)
tag=$1; shift
R=${R:-/tmp/moe-ab}; mkdir -p $R
cd /p/bw/llama.cpp/.claude/worktrees/partial-pin
LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_PREDICT_FILE=0 ./${BIN:-build-cuda}/bin/llama-server \
  -m /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf --port 8234 "$@" > $R/gb_$tag.log 2>&1 &
for i in $(seq 600); do curl -sf localhost:8234/health >/dev/null && break; sleep 1; done
req() { curl -s localhost:8234/v1/chat/completions -d '{"messages":[{"role":"user","content":"Write a Python function that parses a CSV file and returns the average of each column."}],"max_tokens":'${1:-${N:-400}}',"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 "$(dirname "$0")/tps.py"; }
req ${WARM:-400} > /dev/null
echo "$tag: $(req) | $(req)"
for i in 1 2 3 4 5; do pkill -x llama-server; sleep 2; pgrep -x llama-server >/dev/null || break; done
grep -E "hit rate|predicted top-k|router prediction:|learned router" $R/gb_$tag.log | tail -3 | cut -c1-200
