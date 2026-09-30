#!/usr/bin/env python3
"""tunelog.py TAG: store the self-tune decisions of the last perf.py server log (/tmp/perf-<build>.log) in perf.db table tune_log, tied to the newest run."""
import sqlite3, sys, re, shutil, glob, os
c = sqlite3.connect(os.environ.get("PERF_DB", "perf.db"))
c.execute("create table if not exists tune_log (id integer primary key, run_id integer, tag text, line text)")
run = c.execute("select max(id) from runs").fetchone()[0]
log = max(glob.glob("/tmp/perf-*.log"), key=os.path.getmtime)
shutil.copy(log, f"logs/run{run}_{sys.argv[1]}.log")
n = 0
for l in open(log, errors="replace"):
    if "self-tune" in l or "thread autotune" in l:
        c.execute("insert into tune_log (run_id,tag,line) values (?,?,?)", (run, sys.argv[1], re.sub(r"^\S+ [A-Z] ", "", l.strip())[:300])); n += 1
c.commit(); print(f"run {run} {sys.argv[1]}: {n} tune lines")
