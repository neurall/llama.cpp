#!/usr/bin/env python3
"""merge_runs.py [--apply]: run-history.csv rows of the other machines (ssh) that this machine's tools/bench/run-history.csv lacks are appended to it
(append-only; a row is the same run when ts, hw, build, test and model agree), and inserted into runs.db (never rebuilt: load would drop the outputs table).
Without --apply only counts. Rows keep their own columns (missing ones stay empty); a copy of each fetched file is kept in /tmp/merge-runs/."""
import csv, io, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "bench", "run-history.csv")
SRC = [("o@192.168.1.12", "/home/o/w/tools/bench/run-history.csv"), ("o@192.168.1.12", "/home/o/wt-merge/tools/bench/run-history.csv"),
       ("o@192.168.1.12", "/home/o/wt-pin/tools/bench/run-history.csv"), ("o@192.168.1.2", "N:/llama.cpp/tools/bench/run-history.csv"),
       ("o@192.168.1.2", "N:/llama-pred/tools/bench/run-history.csv"), ("o@192.168.1.2", "N:/wt-next/tools/bench/run-history.csv")]
key = lambda r: (r.get("ts"), r.get("hw"), r.get("build"), r.get("test"), r.get("model"))
with open(CSV, newline="") as f:
    rd = csv.DictReader(f); cols = rd.fieldnames; have = {key(r) for r in rd}
new = []
os.makedirs("/tmp/merge-runs", exist_ok=True)
for host, path in SRC:
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", host, ("type " + path.replace("/", "\\")) if host.endswith(".2") else "cat " + path], capture_output=True, text=True)
    if p.returncode or "ts" not in p.stdout[:400]:
        print(f"{host}:{path}: not readable"); continue
    open(f"/tmp/merge-runs/{host.split('@')[1]}-{path.replace('/', '_').replace(':', '')}", "w").write(p.stdout)
    rows = [r for r in csv.DictReader(io.StringIO(p.stdout)) if r.get("ts")]
    add = [r for r in rows if key(r) not in have]
    for r in add: have.add(key(r))
    new += add
    print(f"{host}:{path}: {len(rows)} rows, {len(add)} new")
print(f"{len(new)} new rows in total")
if new and "--apply" in sys.argv:
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval=""); [w.writerow(r) for r in new]
    sys.path.insert(0, HERE); import runlog   # rows are inserted into runs.db one by one: "runlog.py load" would delete the database and with it the outputs table (texts exist only there)
    for r in new: runlog.insert(r, cols)
    print("inserted into", runlog.DB)
