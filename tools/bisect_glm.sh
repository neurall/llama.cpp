#!/bin/bash
# git bisect run helper: builds HEAD of the bisect worktree, runs the GLM 3.0-bit README t100 prompt through llama-cli (run.py exp, deterministic cache,
# fresh state, 3 starts) and compares the settled third start with THRESH (generation t/s, midway between the good and the bad build).
# exit 0: good (>= THRESH), 1: bad, 125: cannot build or run (skip).
# cache-off variant: BISPEC=bisect-off.ini NEED=2 (one measured start after a warm-up, no state, no cache noise)
# usage (in the bisect worktree):  THRESH=24.5 git bisect run /p/bw/wt-release-next/tools/bisect_glm.sh
set -u
R=/p/bw/wt-release-next
C=$(git rev-parse --short HEAD)
echo "== bisect test $C"
W=$PWD /p/bw/data/hcf/lrel/dev-build.sh > /tmp/bisect-build-$C.log 2>&1
tail -1 /tmp/bisect-build-$C.log
[ -x $PWD/build-dev/bin/llama-cli ] || { echo "no llama-cli"; exit 125; }
grep -q "DEVBUILD_EXIT 0" /tmp/bisect-build-$C.log || { echo "build failed"; exit 125; }
S=/p/bw/wt-release-next/tools/experiments/glm-regress/${BISPEC:-bisect-one.ini}
SPEC=$CLAUDE_JOB_DIR/tmp/bisect-$C.ini
sed -e "s#@BIN@#$PWD/build-dev/bin#" -e "s#@NAME@#glm-bis-$C#" -e "s#@STATE@#$CLAUDE_JOB_DIR/tmp/bis-state-$C.ini#" $S > $SPEC
(cd $R && python3 tools/run.py exp $SPEC) 2>&1 | tail -3
G=$(python3 - "$C" "${NEED:-3}" <<'PY'
import csv,sys
rows=[r for r in csv.DictReader(open('/p/bw/wt-release-next/tools/bench/run-history.csv')) if (r.get('exp') or '').startswith('glm-bis-'+sys.argv[1])]
print(rows[-1]['tps'] if len(rows)>=int(sys.argv[2]) and rows[-1]['tps'] else '')
PY
)
echo "settled generation $G (threshold $THRESH)"
[ -n "$G" ] || exit 125
python3 -c "import sys; sys.exit(0 if float('$G') >= float('$THRESH') else 1)"
