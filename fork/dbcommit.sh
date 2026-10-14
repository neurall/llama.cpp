#!/bin/bash
# publish fork/runs.db as the ONLY commit of the branch runs-db: every call replaces that commit (no parent), so the history of the binary never accumulates.
# usage: fork/dbcommit.sh [--push]     --push: force-with-lease to the fork (neurall, branch runs-db only, never anything else)
# get it: git fetch neurall runs-db && git show FETCH_HEAD:runs.db > fork/runs.db      (or rebuild: python3 fork/run.py runlog load; python3 fork/run.py runlog-all --fast)
set -e
R=$(git rev-parse --show-toplevel); DB=$R/fork/runs.db; T=$(mktemp -d)
python3 - "$DB" "$T/runs.db" <<'PY'
import sqlite3, sys
DBP, OUT = sys.argv[1], sys.argv[2]
c = sqlite3.connect(DBP, timeout=60)
# run.py carries runstats as one of its merged sections (registered as module `runstats`): load run.py as a module, not as a program
import importlib.util
sys.argv = ['run.py']
sp = importlib.util.spec_from_file_location("run", DBP.rsplit("/", 1)[0] + "/run.py")
runmod = importlib.util.module_from_spec(sp); sp.loader.exec_module(runmod)
sys.modules["runstats"].refresh(c)                       # the statistics of records added since the last query are current in the published file
c.execute("VACUUM INTO ?", (OUT,))   # compact copy: no free pages
PY
N=$(python3 -c "import sqlite3,sys; print(sqlite3.connect('$T/runs.db').execute('select count(*) from runs').fetchone()[0])")
B=$(git -C "$R" hash-object -w "$T/runs.db")
TREE=$(printf '100644 blob %s\truns.db\n' "$B" | git -C "$R" mktree)
C=$(git -C "$R" commit-tree "$TREE" -m "runs.db: $N runs, $(date +%F' '%H:%M) (only commit of this branch, replaced on every publish)")
git -C "$R" update-ref refs/heads/runs-db "$C"
git -C "$R" reflog expire --expire=now refs/heads/runs-db 2>/dev/null || true
echo "runs-db = $C ($N runs, $(du -h "$T/runs.db" | cut -f1))"
if [ "$1" = "--push" ]; then
  git -C "$R" push --force-with-lease=refs/heads/runs-db neurall refs/heads/runs-db:refs/heads/runs-db 2>&1 | tail -2
fi
rm -rf "$T"
