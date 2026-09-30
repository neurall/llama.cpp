#!/bin/bash
# Shared helpers for the overnight runners. Needs RES (results dir), DEADLINE (epoch seconds), MACHINE, LIBDIR.
# Rules: every wait also ends on failure (dead process, error text in the log) or timeout; every experiment is skipped
# when it can't finish before DEADLINE; nothing here aborts the whole run.
: "${RES:?}"; : "${DEADLINE:?}"; : "${MACHINE:?}"; : "${LIBDIR:?}"
mkdir -p "$RES/logs"
SRVPID=

log()    { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$RES/log.txt"; }
result() { printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MACHINE" "$1" "$2" "$3" "$4" "${5:-}" >> "$RES/results.tsv"; }
time_left() { echo $(( DEADLINE - $(date +%s) )); }
have_time() { [ "$(time_left)" -gt "${1:-600}" ]; }
jget() { python3 -c "
import json,sys
try: d=json.loads(sys.stdin.read())
except Exception: d={}
print(d.get('$1',''))"; }

exp() { # name budget_seconds function [args]: run one experiment if it fits
    local name=$1 budget=$2 fn=$3; shift 3
    if ! have_time "$budget"; then log "EXP_SKIP $name (needs ~${budget}s, ${DEADLINE:+$(time_left)}s left)"; return 0; fi
    log "EXP_START $name (budget ${budget}s, $(time_left)s left)"
    local t0; t0=$(date +%s)
    "$fn" "$@" || log "EXP_ERROR $name returned $?"
    log "EXP_END $name after $(( $(date +%s) - t0 ))s"
}

wait_ram() { # GB, max_seconds
    local gb=$1 max=${2:-300} i
    for i in $(seq $((max / 2))); do
        [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge "$gb" ] && return 0
        sleep 2
    done
    log "WARN: MemAvailable < ${gb} GB after ${max}s, continuing"; return 1
}

srv_stop() {
    [ -n "$SRVPID" ] || return 0
    kill "$SRVPID" 2>/dev/null
    local i; for i in $(seq 90); do kill -0 "$SRVPID" 2>/dev/null || break; sleep 2; done
    kill -9 "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=
}

# srv_start TAG PORT BIN LOGFILE ARGS... (env vars for the server in $SRV_ENV): 0 when /health answers
srv_start() {
    local tag=$1 port=$2 bin=$3 slog=$4; shift 4
    env $SRV_ENV "$bin" --port "$port" "$@" > "$slog" 2>&1 &
    SRVPID=$!
    local t0; t0=$(date +%s)
    while true; do
        curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null && return 0
        if ! kill -0 "$SRVPID" 2>/dev/null; then log "FAIL $tag: server exited: $(tail -2 "$slog" | tr '\n' ' ' | cut -c1-200)"; SRVPID=; return 1; fi
        if grep -qE "CUDA error|out of memory|Aborted|failed to load|Compute error" "$slog"; then
            log "FAIL $tag: $(grep -m1 -E 'CUDA error|out of memory|Aborted|failed to load|Compute error' "$slog" | cut -c1-150)"; srv_stop; return 1
        fi
        if [ $(( $(date +%s) - t0 )) -gt "${SRV_TIMEOUT:-900}" ]; then log "FAIL $tag: no /health after ${SRV_TIMEOUT:-900}s"; srv_stop; return 1; fi
        sleep 2
    done
}

req() { # port kind n -> JSON line
    timeout "${REQ_TIMEOUT:-1800}" python3 "$LIBDIR/client.py" "http://127.0.0.1:$1" "$2" "$3" 2>&1 | tail -1
}

# series TEST VARIANT PORT SPEC: SPEC like "short:3,12k:2,code:3,chat:3" (kind:repetitions); PREWARM=n long answers first
series() {
    local test=$1 variant=$2 port=$3 spec=$4 item kind cnt r n out e dr w
    for w in $(seq "${PREWARM:-0}"); do PIDX=$w req "$port" short 1500 > /dev/null; done
    for item in ${spec//,/ }; do
        kind=${item%%:*}; cnt=${item##*:}
        case $kind in short) n=1500;; 12k) n=32;; code) n=300;; chat) n=400;; *) n=200;; esac
        for r in $(seq "$cnt"); do
            out=$(PIDX=$r req "$port" "$kind" "$n")
            e=$(echo "$out" | jget err)
            if [ -n "$e" ] || [ -z "$out" ]; then log "ERR $test/$variant $kind rep $r: ${e:-no output}"; break; fi
            result "$test" "$variant" "${kind}_tg" "$(echo "$out" | jget tg)" "rep=$r"
            [ "$kind" = 12k ] && result "$test" "$variant" "12k_pp" "$(echo "$out" | jget pp)" "rep=$r"
            dr=$(echo "$out" | jget draft); [ -n "$dr" ] && result "$test" "$variant" "${kind}_draft" "$dr" "rep=$r"
        done
    done
    log "  $test/$variant done"
}

grab() { # TAG SERVERLOG: the interesting server lines into serverlines.txt
    grep -hE "probe:|prefill links|prefill split|self-tune:|cached on|upload workers|auto_impl|weights pinned|from pinned|failed|CUDA error" "$2" 2>/dev/null \
        | cut -c1-280 | sed "s|^|[$1] |" >> "$RES/serverlines.txt"
}

# run_variant TEST VARIANT MODEL BIN "ENV=.." SPEC [server args...]
run_variant() {
    local test=$1 variant=$2 model=$3 bin=$4 envs=$5 spec=$6; shift 6
    wait_ram "${WAITGB:-110}" 300
    local slog="$RES/logs/${test}_${variant//\//_}.srv.log"
    log "  $test/$variant: starting server"
    SRV_ENV="$envs" srv_start "$test/$variant" "${PORT:-8250}" "$bin" "$slog" -m "$model" "$@" || return 1
    series "$test" "$variant" "${PORT:-8250}" "$spec"
    grab "$test/$variant" "$slog"
    srv_stop
}

lb_csv() { # llama-bench command...: lines "test avg_ts"
    "$@" -o csv 2>/dev/null | python3 -c "
import csv,sys
for r in csv.DictReader(sys.stdin):
    print(r.get('test',''), r.get('avg_ts',''))"
}
