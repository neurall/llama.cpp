#!/bin/bash
# PC1 (2x RTX 3090, Ryzen 3700X, DDR4-3200 ECC) overnight experiments. Run via run_all.sh.
export MACHINE=pc1; LIBDIR=$(dirname "$(readlink -f "$0")"); export LIBDIR
source "$LIBDIR/lib.sh"
WT=/p/bw/llama.cpp/.claude/worktrees/partial-pin
BIN=/p/bw/rels/satur-final/llama-server
REL=/p/bw/rels/b11399-5c739a3/llama-server
GLM=/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf
QN=/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
MIMO=/m/m/3/MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf
export PORT=8250
export LLAMA_MOE_CACHE_PREDICT_FILE=0

busy() { pgrep -x llama-server > /dev/null || pgrep -x llama-perplexity > /dev/null || pgrep -f "[p]c1_spec.sh|[p]c1_upstream.sh|[p]c1_reg.sh|[c]tl_bench.sh|[p]erf.py run" > /dev/null; }

wait_old_chain() { # earlier interactive queue (regression, upstream comparison, speculation): let it finish, max 150 min
    local t0 quiet=0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt 9000 ]; do
        if busy; then quiet=0; else quiet=$((quiet + 1)); [ $quiet -ge 6 ] && return 0; fi
        sleep 10
    done
    log "WARN: old queue still busy after 150 min, continuing anyway"
}

prepare() {
    cd "$WT" || return 1
    log "HEAD $(git log --oneline -1 | cut -c1-90)"
    cmake --build build-link --target llama-server llama-perplexity llama-bench -j12 > "$RES/logs/build.txt" 2>&1 || { log "FAIL build-link"; tail -3 "$RES/logs/build.txt"; return 1; }
    rm -rf /p/bw/rels/satur-final; mkdir -p /p/bw/rels/satur-final && cp -a build-link/bin/. /p/bw/rels/satur-final/
    log "built and archived /p/bw/rels/satur-final ($($BIN --version 2>&1 | grep -m1 version))"
    nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,memory.used --format=csv,noheader | tee -a "$RES/log.txt"
    { lscpu | grep -E "Model name|L3"; grep -E "MemTotal|MemAvailable" /proc/meminfo; } >> "$RES/logs/hw.txt"
}

exp_selftune() { # self-tuning on (default) vs off, GLM, bare and with the predictor; alternating servers, tuner pre-converged
    local order cfg ex a
    for order in 1 2; do
        for cfg in bare pred; do
            a=""; [ $cfg = pred ] && a="--moe-predict 4 --lrn-prd 32"
            for v in on off; do
                ex=""; [ $v = off ] && ex="LLAMA_MOE_CACHE_SELF_TUNE=0"
                PREWARM=3 run_variant "selftune_$cfg" "$v#$order" "$GLM" "$BIN" "$ex" "short:3,code:3" $a
            done
        done
    done
}

exp_prefill() { # multi-GPU prefill: auto decision vs one GPU vs fixed alphas
    local v ex
    for v in auto 0 1 0.5 0.25; do
        ex="GGML_SCHED_EVENTS=1"; [ $v != auto ] && ex="$ex LLAMA_PREFILL_SPLIT=$v"
        run_variant prefill "alpha_$v" "$GLM" "$BIN" "$ex" "12k:3"
    done
    # the released build as reference
    run_variant prefill "b11399" "$GLM" "$REL" "" "12k:3"
}

exp_qwen_next() { # Qwen3.8-Flash-Next (88 GB): release vs fork default vs fork without self-tune vs fork + predictor
    export WAITGB=100
    run_variant qwen_next "b11399" "$QN" "$REL" "" "short:3,12k:2"
    run_variant qwen_next "fork" "$QN" "$BIN" "" "short:3,12k:2"
    run_variant qwen_next "fork_notune" "$QN" "$BIN" "LLAMA_MOE_CACHE_SELF_TUNE=0" "short:3,12k:2"
    PREWARM=2 run_variant qwen_next "fork_pred" "$QN" "$BIN" "" "short:3,12k:2" --moe-predict 4 --lrn-prd 32
    unset WAITGB
}

exp_mimo() { # MiMo (132 GB, bigger than RAM, mmap)
    export WAITGB=20 SRV_TIMEOUT=2400
    run_variant mimo "b11399" "$MIMO" "$REL" "" "short:2,12k:1"
    run_variant mimo "fork" "$MIMO" "$BIN" "" "short:2,12k:1"
    unset WAITGB SRV_TIMEOUT
}

exp_perfpy() { # the established regression harness, stored in its own db
    cd "$WT" || return 1
    export PERF_BUILDS=/p/bw/rels MODEL=$GLM PERF_MODELS_DIR=/m PERF_DB="$RES/perf.db"
    timeout 3600 python3 tools/moe-bench/perf.py run -t chat -n 3 --bare --note overnight b11399-5c739a3 satur-final > "$RES/logs/perfpy_chat.txt" 2>&1
    timeout 1800 python3 tools/moe-bench/perf.py run -t ppl -n 2 --note overnight b11399-5c739a3 satur-final > "$RES/logs/perfpy_ppl.txt" 2>&1
    python3 tools/moe-bench/perf.py show -t chat > "$RES/logs/perfpy_show_chat.txt" 2>&1
    python3 tools/moe-bench/perf.py show -t ppl  > "$RES/logs/perfpy_show_ppl.txt" 2>&1
    tail -5 "$RES/logs/perfpy_show_chat.txt" | tee -a "$RES/log.txt"
}

exp_mem() {
    mkdir -p "$RES/bin"
    g++ -O2 -mavx2 -pthread "$WT/tools/moe-bench/membw.cpp" -o "$RES/bin/membw" 2>&1 | head -3
    cc -O2 -pthread "$WT/tools/moe-bench/memlat.c" -o "$RES/bin/memlat" 2>&1 | head -3
    local i out
    for i in 1 2 3; do
        "$RES/bin/membw" 2048 > "$RES/logs/membw_$i.txt" 2>&1
        grep "^full" "$RES/logs/membw_$i.txt" | while read -r _ n _ g _; do result mem "run$i" "membw_full_${n}t_GBs" "$g"; done
        "$RES/bin/memlat" 1024 4 > "$RES/logs/memlat_$i.txt" 2>&1
        out=$(grep -oE "idle latency: [0-9.]+" "$RES/logs/memlat_$i.txt" | grep -oE "[0-9.]+$"); [ -n "$out" ] && result mem "run$i" "idle_latency_ns" "$out"
    done
}

exp_repeat_glm() { # more statistics: release vs fork default, chat and 12k, alternating
    local o
    for o in 1 2 3; do
        run_variant repeat_glm "b11399#$o" "$GLM" "$REL" "" "short:2,12k:1"
        run_variant repeat_glm "fork#$o" "$GLM" "$BIN" "" "short:2,12k:1"
    done
}

log "pc1 runner start, deadline in $(time_left)s"
wait_old_chain
exp prepare 900 prepare || { log "prepare failed, aborting pc1"; exit 1; }
exp selftune   3000 exp_selftune
exp prefill    2400 exp_prefill
exp qwen_next  3000 exp_qwen_next
exp perfpy     4200 exp_perfpy
exp mem         300 exp_mem
exp mimo       3600 exp_mimo
exp repeat_glm 3600 exp_repeat_glm
log "pc1 runner finished"
echo PC1_OVERNIGHT_DONE >> "$RES/log.txt"
