#!/bin/bash
# ctl_bench.sh ROUNDS name:KEY=V,KEY=V ... : one GLM server, variants switched at runtime through LLAMA_MOE_CACHE_CTL,
# each round runs every variant once (warm-up WARM tokens after the switch, then one timed N-token answer); prints mean t/s.
# Prompts vary (a repeated temp-0 answer would reuse exactly the cached experts): round r, variant i answers prompt (r+i) mod V
# of the current block of V prompts, so after V rounds every variant saw every prompt once, in every position
rounds=$1; shift
R=${R:-/tmp/moe-ab}; mkdir -p $R; D=$(dirname "$(readlink -f "$0")")
cd /p/bw/llama.cpp/.claude/worktrees/partial-pin
CTL=$R/ctl.txt; : > $CTL
# a previous server's pinned weights take a while to be released: starting before that makes auto mode fall back
# to pageable (mmap) uploads and the run doesn't compare; wait until MemAvailable is back (MIN_AVAIL_GB, default 110)
for i in $(seq 120); do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge "${MIN_AVAIL_GB:-110}" ] && break
  sleep 2
done
LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_PREDICT_FILE=0 LLAMA_MOE_CACHE_CTL=$CTL ${SERVER:-./${BIN:-build-link}/bin/llama-server} \
  -m ${MODEL:-/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf} --port 8234 $ARGS > $R/ctl.log 2>&1 &
for i in $(seq 600); do curl -sf localhost:8234/health >/dev/null && break; sleep 1; done
if grep -q "from pageable host memory" $R/ctl.log; then
  echo "ctl_bench: weights not pinned (pageable uploads), results would not compare -- aborting" >&2
  for i in 1 2 3 4 5; do pkill -x llama-server; sleep 2; pgrep -x llama-server >/dev/null || break; done
  exit 1
fi
P=(
"Write a Python function that parses a CSV file and returns the average of each column."
"Write a short story about a lighthouse keeper who finds a message in a bottle."
"Explain how a CPU cache hierarchy works and why cache misses are expensive."
"Write a SQL query that finds the top 5 customers by total order value, with the schema."
"Translate into German and French: The meeting is moved to Thursday because the room is booked."
"Solve step by step: a train leaves at 9:40 at 84 km/h, a second at 10:10 at 105 km/h. When does it catch up?"
"Write a bash script that backs up a directory to a dated tar.gz and keeps only the last 7 backups."
"Summarize the causes and consequences of the French Revolution."
"Write a Rust function that reverses the words in a string without allocating per word."
"Give a recipe for a vegetarian lasagna with a shopping list."
"Write a JavaScript debounce function and explain when to use it instead of throttle."
"Explain the difference between TCP and UDP with examples of when to use each."
)
req() { python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}))' "$2" "$1" \
  | curl -s localhost:8234/v1/chat/completions -d @- | python3 "$D/tps.py"; }
req 64 "Hello, who are you?" > /dev/null
V=$#
declare -A sum
VS=("$@")
declare -A last
for r in $(seq $rounds); do
  # run order rotates each round (variant i runs at position (i - r + 1) mod V): no variant always runs first or last,
  # which matters on machines that heat up during a round; each variant keeps its own prompt schedule
  for j in $(seq 0 $((V-1))); do
    i=$(( (j + r - 1) % V )); v=${VS[$i]}
    name=${v%%:*}; tr ',' '\n' <<< "${v#*:}" > $CTL
    p=${P[$(( ( (r-1)/V*V + (r-1+i)%V ) % ${#P[@]} ))]}
    req ${WARM:-32} "Say hi in one short sentence." > /dev/null
    t=$(req ${N:-200} "$p" | grep -o '[0-9.]* t/s' | cut -d' ' -f1)
    sum[$name]=$(python3 -c "print(${sum[$name]:-0} + ${t:-0})")
    last[$name]=$t
  done
  line="round $r:"
  for v in "$@"; do name=${v%%:*}; line="$line $name ${last[$name]}"; done
  echo "$line"
done
for v in "$@"; do name=${v%%:*}; echo "$name mean $(python3 -c "print(round(${sum[$name]} / $rounds, 2))") t/s"; done
for i in 1 2 3 4 5; do pkill -x llama-server; sleep 2; pgrep -x llama-server >/dev/null || break; done
grep -E "cached on|ctl: segment|phases:" $R/ctl.log | cut -c1-250
