#!/bin/bash
# w10_reg.sh NEW_BIN OLD_BIN [rounds] [extra server args for both]
# One-GPU regression on the W10 box (RTX 4060 8 GB, Ryzen 9 8945HS, LPDDR5X, Windows): run this for every release build next to
# the PC1 (2-GPU) perf.py regression, to see that a new build didn't break single-GPU / Windows / Zen 4 performance.
#   NEW_BIN / OLD_BIN: directories on W10 holding llama-server.exe, e.g. N:\llama.cpp\build-cuda-dl\bin and N:\rel\b11399
#   env: W10=user@host (o@192.168.1.2)  MODEL=N:\m\qwen36.gguf  PORT=18080  TOL=0.95 (new must reach 95% of old)
# Per variant and round (alternating order to cancel thermal/position bias): server start, 500-token warmup, README "short"
# test (1500 tokens), 2 chats (300 tokens), and a LONG prompt (12k tokens: prefill t/s is the metric). Prints HB heartbeat lines, a summary table and REGRESSION / OK, appends
# tools/moe-bench/w10_reg_history.csv, and ends with W10REG_DONE. Needs AC power (battery caps the GPU at ~600 MHz).
NEW=${1:?NEW_BIN}; OLD=${2:?OLD_BIN}; shift 2; ROUNDS=2; if [[ "${1:-}" =~ ^[0-9]+$ ]]; then ROUNDS=$1; shift; fi; XARGS="$*"
W10=${W10:-o@192.168.1.2}; MODEL=${MODEL:-N:\\m\\qwen36.gguf}; PORT=${PORT:-18080}; TOL=${TOL:-0.95}
HERE=$(cd "$(dirname "$0")" && pwd); CLIENT="$HERE/overnight/client.py"; OUT=${OUT:-$(mktemp -d)}
hb() { echo "HB $(date +%H:%M:%S) $*"; }

ac=$(ssh "$W10" 'powershell -c "(Get-CimInstance Win32_Battery).BatteryStatus"' 2>/dev/null | tr -d '\r ')
[ "$ac" = "2" ] || { echo "ABORT: W10 not on AC power (BatteryStatus='$ac'); results would be invalid"; exit 2; }
ssh "$W10" "taskkill /F /IM llama-server.exe" > /dev/null 2>&1

# generic server launcher on W10: %1 = bin dir, %2 = log name (generated here, copied over: no nested ssh quoting)
BAT=$(mktemp)
printf '@echo off\r\nset PATH=N:\\nv\\bin;N:\\nv\\bin\\x64;%%1;%%PATH%%\r\nset LLAMA_MOE_CACHE_PREDICT_FILE=0\r\n%%1\\llama-server.exe -m %s --host 127.0.0.1 --port 8080 %s > N:\\tmp\\w10reg_%%2.log 2>&1\r\n' "$MODEL" "$XARGS" > "$BAT"
scp -q "$BAT" "$W10:N:/tmp/w10reg_srv.bat" || { echo "ABORT: cannot copy launcher to W10"; exit 2; }
rm -f "$BAT"

run_variant() { # tag dir round
  local tag=$1 dir=$2 round=$3 t0 ok=0 sp tp
  hb "round $round $tag: start"
  ssh "$W10" "N:\\tmp\\w10reg_srv.bat $dir $tag" > /dev/null 2>&1 & sp=$!
  ssh -N -L "$PORT":127.0.0.1:8080 "$W10" & tp=$!
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 300 ]; do curl -sf -m 2 "localhost:$PORT/health" > /dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 2; done
  if [ $ok = 1 ]; then
    PIDX=0 python3 "$CLIENT" "http://127.0.0.1:$PORT" short 500 > /dev/null
    echo "$round $tag short $(PIDX=1 timeout 600 python3 "$CLIENT" "http://127.0.0.1:$PORT" short 1500)" >> "$OUT/raw.txt"
    for i in 2 3; do echo "$round $tag chat $(PIDX=$i timeout 300 python3 "$CLIENT" "http://127.0.0.1:$PORT" chat 300)" >> "$OUT/raw.txt"; done
    echo "$round $tag long $(timeout 900 python3 "$CLIENT" "http://127.0.0.1:$PORT" 12k 32)" >> "$OUT/raw.txt"   # 12k-token prompt: pp = prefill t/s
    hb "round $round $tag: $(tail -3 "$OUT/raw.txt" | sed -E 's/.*"tg": ([0-9.]+).*/\1/' | tr '\n' ' ') t/s"
  else
    echo "$round $tag FAIL" >> "$OUT/raw.txt"; hb "round $round $tag: server failed to start"
  fi
  ssh "$W10" "taskkill /F /IM llama-server.exe" > /dev/null 2>&1
  kill $tp $sp 2>/dev/null; wait $sp 2>/dev/null; sleep 3
}

for round in $(seq "$ROUNDS"); do
  if [ $((round % 2)) = 1 ]; then run_variant old "$OLD" "$round"; run_variant new "$NEW" "$round"
  else run_variant new "$NEW" "$round"; run_variant old "$OLD" "$round"; fi
done

python3 - "$OUT/raw.txt" "$TOL" "$HERE/w10_reg_history.csv" "$NEW" "$OLD" <<'EOF'
import sys, json, statistics as st, re, time, os
raw, tol, hist, new, old = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
d = {}
for l in open(raw):
    m = re.match(r"(\d+) (\w+) (\w+) (\{.*\})", l)
    if m:
        try:
            j = json.loads(m.group(4)); d.setdefault((m.group(2), m.group(3)), []).append(j["pp"] if m.group(3) == "long" else j["tg"])
        except Exception: pass
print("\nW10 (RTX 4060, Windows), mean of runs (n); short/chat = decode t/s, long = 12k-prompt prefill t/s      old        new      new/old")
bad = False
for test in ("short", "chat", "long"):
    o, n = d.get(("old", test), []), d.get(("new", test), [])
    if not o or not n: print(f"  {test:6s} missing data (old {len(o)}, new {len(n)})"); bad = True; continue
    ro = st.mean(o); rn = st.mean(n)
    flag = "" if rn >= tol*ro else "   <-- REGRESSION"
    bad |= bool(flag)
    print(f"  {test:6s} {ro:8.1f} ({len(o)})  {rn:8.1f} ({len(n)})   x{rn/ro:.2f}{flag}")
    with open(hist, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{test},{old},{new},{ro:.2f},{rn:.2f}\n")
print("RESULT:", "REGRESSION" if bad else "OK")
EOF
echo W10REG_DONE
