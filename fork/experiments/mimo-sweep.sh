#!/bin/bash
# MiMo best-gain prompt length: the key points of the curve, then the rest of the release gate under the same conditions (user 2026-10-25).
# 1 ours, automatic mode, over prompt sizes (about 1 min a cell), 2 stock (official binary) only at 8k, 16k, 32k - it is flat, the chart interpolates between,
# 3 forced one card / forced split only at the crossover (12k), 4 the README cells of both builds (a gate needs both halves under the same pool state), 5 gate_total.
# usage: experiments/mimo-sweep.sh [OURS_BUILD=rc5] [LAST_RELEASE_BUILD=relmf1] [STOCK=stock-b11866-official]      run it after the pool is mounted; as user o
# then: python3 fork/readme_chart.py --ours OURS --stock STOCK     (SVG for the README + the table)
OURS=${1:-rc5}; REL=${2:-relmf1}; STOCK=${3:-stock-b11866-official}
cd "$(dirname "$0")/.." || exit 1
LOG=${MIMO_SWEEP_LOG:-/p/bw/data/mimo-sweep}; mkdir -p "$LOG"; M=$LOG/progress
waitfree() { while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | grep -qv '^0$'; do sleep 10; done; }
ps_args() { for s in "$@"; do printf -- '-ps %s ' "$s"; done; }
run() { waitfree; env COLUMNS=200 python3 run.py run -c gainers "$@" -pl 220/280 -nm; }
run -onl 'mimo-i$' $(ps_args 6k1k 8k1k 10k1k 12k1k 14k1k 16k1k 20k1k 24k1k 32k1k) -b "$OURS" > "$LOG/ours-auto.out" 2>&1; echo "ours auto done" >> "$M"
run -onl 'mimo-i$' $(ps_args 8k1k 16k1k 32k1k) -r 1 -b "$STOCK" > "$LOG/stock.out" 2>&1; echo "stock 8k 16k 32k done" >> "$M"
run -onl 'mimo-i$' $(ps_args 12k1k) -b "$OURS" -a "moe:dense-split=0" > "$LOG/one-12k.out" 2>&1; echo "forced one card 12k done" >> "$M"
run -onl 'mimo-i$' $(ps_args 12k1k) -b "$OURS" -a "moe:dense-split=1" > "$LOG/split-12k.out" 2>&1; echo "forced split 12k done" >> "$M"
run -b "$OURS" > "$LOG/gate-ours.out" 2>&1; echo "gainers $OURS done" >> "$M"
run -b "$REL" > "$LOG/gate-rel.out" 2>&1; echo "gainers $REL done" >> "$M"
python3 gate_total.py "" "$OURS" "$REL" > "$LOG/gate.out" 2>&1; echo "gate exit $?" >> "$M"
echo "ALLDONE" >> "$M"
