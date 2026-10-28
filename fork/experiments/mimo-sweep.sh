#!/bin/bash
# MiMo best-gain prompt length: the key points of the curve, then the README cells of both builds, each model in the memory state that suits it (user 2026-10-25).
# The huge page pool (101 GiB here) is used only for models that fit it; the others (MiMo 107.7, GLM 3.0 109.4, GLM 3.5 127.7 GiB) run with the pool unmounted, and so does stock:
# stock reads the model through the page cache and thrashes, or stalls ssh, with ~9-17 GiB of RAM left beside a pool. Never mount/unmount the pool while a run is active.
#  P (pool mounted):  the README cells of the models that fit the pool, ours then the last release (same pool state for both); smaller models first (user 2026-10-25)
#  N (pool unmounted): MiMo ours-auto sweep over prompt sizes 6k-32k, forced one card / split at 12k, then the README cells of the models that do not fit, ours then the last release, 5 stock MiMo at 8k, 16k, 32k (it is flat, the chart interpolates between), 6 gate_total
# usage: experiments/mimo-sweep.sh P|N [OURS_BUILD=rc5] [LAST_RELEASE_BUILD=relmf1] [STOCK=stock-b11866-official]      as user o
# then: python3 fork/readme_chart.py --ours OURS --stock STOCK     (SVG for the README + the table)
PHASE=${1:?P (pool mounted) or N (pool unmounted)}; OURS=${2:-rc5}; REL=${3:-relmf1}; STOCK=${4:-stock-b11866-official}
cd "$(dirname "$0")/.." || exit 1
LOG=${MIMO_SWEEP_LOG:-/p/bw/data/mimo-sweep}; mkdir -p "$LOG"; M=$LOG/progress
FIT='qni4-,qni3-,qn1-,olmoei-,q36-,q27nl-,q27mtp-'; NOFIT='mimo-,glm-,glm35-'   # by the model sizes in the README (GiB): 87, 78, 54, 4, 11, 15, 15 against 108, 109, 128
waitfree() { while pgrep -f "[r]un.py run gainers" > /dev/null || nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | grep -qv '^0$'; do sleep 10; done; }   # a run already in flight finishes first
ps_args() { for s in "$@"; do printf -- '-ps %s ' "$s"; done; }
run() { waitfree; env COLUMNS=200 python3 run.py run -c gainers "$@" -pl 220/280 ${NMARG:--nm} $PINARG; }   # PINARG=-pin (phase P): load into the pool pages; the engine default pins too but from ordinary RAM, the mounted pool stays unused (101 of 101 free, 2026-10-25)   # NMARG="-nm 108": MiMo pins its 107.7 GiB straight from ordinary RAM (pool unmounted)
if [ "$PHASE" = P ]; then
  PINARG=-pin
  run -onl "$FIT" -b "$OURS" > "$LOG/fit-ours.out" 2>&1; echo "fitting models $OURS done" >> "$M"
  run -onl "$FIT" -b "$REL" > "$LOG/fit-rel.out" 2>&1; echo "fitting models $REL done" >> "$M"
else
  [ -n "$SKIP_SWEEP" ] || { NMARG="-nm 108" run -onl 'mimo-i$' $(ps_args 6k1k 8k1k 10k1k 12k1k 14k1k 16k1k 20k1k 24k1k 32k1k) -b "$OURS" > "$LOG/ours-auto.out" 2>&1; }; echo "ours auto done" >> "$M"
  NMARG="-nm 108" run -onl 'mimo-i$' $(ps_args 12k1k) -b "$OURS" -a "moe:dense-split=0" > "$LOG/one-12k.out" 2>&1; echo "forced one card 12k done" >> "$M"
  NMARG="-nm 108" run -onl 'mimo-i$' $(ps_args 12k1k) -b "$OURS" -a "moe:dense-split=1" > "$LOG/split-12k.out" 2>&1; echo "forced split 12k done" >> "$M"
  run -onl "$NOFIT" -b "$OURS" > "$LOG/nofit-ours.out" 2>&1; echo "non-fitting models $OURS done" >> "$M"
  run -onl "$NOFIT" -b "$REL" > "$LOG/nofit-rel.out" 2>&1; echo "non-fitting models $REL done" >> "$M"
  run -onl 'mimo-i$' $(ps_args 8k1k 16k1k 32k1k) -r 1 -b "$STOCK" > "$LOG/stock.out" 2>&1; echo "stock 8k 16k 32k done" >> "$M"
  python3 gate_total.py "" "$OURS" "$REL" > "$LOG/gate.out" 2>&1; echo "gate exit $?" >> "$M"
fi
echo "PHASE $PHASE ALLDONE" >> "$M"
