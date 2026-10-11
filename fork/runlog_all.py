#!/usr/bin/env python3
"""runlog_all.py: every run we ever recorded in one table of tools/runs.db (all_runs), plus views that make a regression visible.

usage: runlog_all.py [--db tools/runs.db] [--readme-source]   (rebuilds all_runs and its views, leaves the other tables alone)

Sources, de-duplicated on (ts to the second, model, test, build, tps):
  csv     fork/bench/run-history.csv of this tree and of every worktree next to it (the same file at other branches)
  gitcsv  every committed version of fork/bench/run-history.csv (tools/bench/ before 2026-10-11) in git history (all branches), rows later dropped from the file included
  db      the old SQLite files (tools/moe-bench/perf.db from git history, fork/bench/archive/perf-*.db, /p/bw/rels/runs.db, /p/bw/data/**/perf*.db, rc_perf*.db)
  readme  --readme-source: the speed tables of old README versions in git (numbers from before the run log, only where no logged run of that model and test exists)
Normalised columns: fam (model family label), tgroup (test without the cli- prefix and -lv3 suffix), bno (build number from the build name or bno column),
bseq (rank of the build by the time of its first run: orders dev and release-candidate builds that have no number), side (stock / ours), tps, pp, hit, ok.
Views:
  cell_builds   per family, test, machine, side and build: best, median and number of runs, in build order
  regress       for each build of the fork: its best against the best of ALL earlier builds of that cell (ratio), flagged when below 0.9
  best_ever     the build with the best number of each cell, and the newest build's number next to it"""
import argparse, csv, glob, io, os, re, sqlite3, statistics, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
ap.add_argument("--readme-source", action="store_true")
ap.add_argument("--extra", nargs="*", default=[], help="more run-history.csv and old perf db files, e.g. copies of the logs of pc2 and the laptop")
ap.add_argument("--fast", action="store_true", help="only the current run-history.csv files (no git history, no old db files): seconds, for the rebuild after new records")
ap.add_argument("--write-csv", action="store_true", help="merge the rows missing from fork/bench/run-history.csv (git history, old db files, other worktrees) into it: existing rows stay byte-identical, the missing ones are appended in its columns")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from runnorm import FAMS, fam, bno_of, tgroup, prompt_of


rows, seen = [], set()
raw = []   # (src, dict in run-history.csv columns) of every kept row, for --write-csv


CUR = {}


def add(src, ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note="", rep="", tag=""):
    try:
        tps = float(tps)
    except (TypeError, ValueError):
        return
    if not tps:
        return
    ts = (ts or "")[:19]
    key = (ts, fam(model, args), test, build, round(tps, 2))
    if key in seen:
        return
    seen.add(key)
    try: pp = float(pp)
    except (TypeError, ValueError): pp = None
    try: hit = float(hit)
    except (TypeError, ValueError): hit = None
    raw.append((src, CUR))
    rows.append((src, ts, hw or "pc1", fam(model, args), tgroup(test), test, build, bno_of(build, bno), "stock" if (build or "").startswith("stock") else "ours", tps, pp, hit, 0 if str(ok) == "0" else 1, (args or "")[:60], (note or "")[:80], prompt_of(test, rep, tag)))


def read_csv_text(text, src):
    global CUR
    for r in csv.DictReader(io.StringIO(text)):
        CUR = dict(r)
        add(src, r.get("ts"), r.get("hw"), r.get("model"), r.get("args"), r.get("test"), r.get("build"), r.get("bno"), r.get("tps"), r.get("pp"), r.get("hit"), r.get("ok", 1), r.get("note"), r.get("rep"), r.get("prompt"))


# csv: every worktree
for f in sorted(set(glob.glob("/p/bw/*/tools/bench/run-history.csv") + glob.glob("/p/bw/*/fork/bench/run-history.csv") + glob.glob("/p/bw/llama.cpp/.claude/worktrees/*/tools/bench/run-history.csv") + [os.path.join(ROOT, "fork/bench/run-history.csv")])):
    read_csv_text(open(f, errors="replace").read(), "csv")
for f in a.extra:
    if f.endswith(".csv"):
        read_csv_text(open(f, errors="replace").read(), "extra")
n_csv = len(rows)
n_git = n_db = 0

# gitcsv: every committed version
seen_blobs = set()
log = [] if a.fast else subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "fork/bench/run-history.csv", "tools/bench/run-history.csv", "tools/moe-bench/perf.csv"], capture_output=True, text=True).stdout.split()
for h in log:
    for path in ("fork/bench/run-history.csv", "tools/bench/run-history.csv", "tools/moe-bench/perf.csv"):   # fork/ since 2026-10-11, tools/ before
        b = subprocess.run(["git", "-C", ROOT, "rev-parse", f"{h}:{path}"], capture_output=True, text=True).stdout.strip()
        if b and b not in seen_blobs:
            seen_blobs.add(b)
            read_csv_text(subprocess.run(["git", "-C", ROOT, "cat-file", "-p", b], capture_output=True).stdout.decode(errors="replace"), "gitcsv")
n_git = len(rows) - n_csv


def read_db(path, src):
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        cols = [x[1] for x in c.execute("pragma table_info(runs)")]
        if not cols:
            return
        for r in c.execute("select * from runs"):
            d = dict(zip(cols, r))
            global CUR
            CUR = {"ts": d.get("ts"), "commit": d.get("commit_sha"), "bno": d.get("build_no"), "tps": d.get("tps"), "pp": d.get("pp_tps"), "hit": d.get("hit_rate"), "model": d.get("model"),
                   "args": d.get("args"), "test": d.get("test"), "hw": d.get("hw") or "pc1", "ok": d.get("ok", 1), "build": d.get("build"), "md5": d.get("md5"), "campaign": "import-old-db",
                   "note": d.get("note"), "ppl": d.get("ppl"), "n_gen": d.get("n_gen"), "origin": d.get("origin"), "env": d.get("env")}
            add(src, d.get("ts"), d.get("hw"), d.get("model"), d.get("args"), d.get("test"), d.get("build"), d.get("build_no"), d.get("tps"), d.get("pp_tps"), d.get("hit_rate"), d.get("ok", 1), d.get("note"))
    except sqlite3.Error:
        pass


dbs = glob.glob(os.path.join(ROOT, "fork/bench/archive/*.db")) + glob.glob("/p/bw/rels/runs.db") + glob.glob("/p/bw/data/**/perf*.db", recursive=True) + glob.glob("/p/bw/data/**/rc_perf*.db", recursive=True)
for f in ([] if a.fast else sorted(set(dbs))):
    read_db(f, "db")
for f in a.extra:
    if f.endswith(".db"):
        read_db(f, "db")
for h in ([] if a.fast else subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "tools/moe-bench/perf.db"], capture_output=True, text=True).stdout.split()):
    blob = subprocess.run(["git", "-C", ROOT, "cat-file", "-p", f"{h}:tools/moe-bench/perf.db"], capture_output=True).stdout
    if blob:
        with tempfile.NamedTemporaryFile(suffix=".db") as t:
            t.write(blob); t.flush(); read_db(t.name, "db")
n_db = len(rows) - n_csv - n_git

con = sqlite3.connect(a.db)
con.execute("DROP TABLE IF EXISTS all_runs")
con.execute("""CREATE TABLE all_runs (src TEXT, ts TEXT, hw TEXT, fam TEXT, tgroup TEXT, test TEXT, build TEXT, bno INTEGER, side TEXT, tps REAL, pp REAL, hit REAL, ok INTEGER, args TEXT, note TEXT, prompt TEXT, bseq INTEGER)""")
con.executemany("INSERT INTO all_runs (src,ts,hw,fam,tgroup,test,build,bno,side,tps,pp,hit,ok,args,note,prompt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
# bseq: rank of the build by the time of its first run
con.execute("""UPDATE all_runs SET bseq = (SELECT COUNT(*) FROM (SELECT build b, MIN(ts) t FROM all_runs GROUP BY build) x WHERE x.t < (SELECT MIN(ts) FROM all_runs y WHERE y.build = all_runs.build))""")
con.execute("CREATE INDEX IF NOT EXISTS i_all ON all_runs (fam, tgroup, hw, side, build)")
for v in ("cell_builds", "regress", "best_ever", "coverage"):
    con.execute(f"DROP VIEW IF EXISTS {v}")
con.execute("""CREATE VIEW cell_builds AS SELECT fam, tgroup, hw, side, build, bseq, MIN(ts) AS first, COUNT(*) AS n, ROUND(MAX(tps), 2) AS best, ROUND(AVG(tps), 2) AS mean,
               ROUND(MAX(pp), 1) AS best_pp, ROUND(MAX(hit), 1) AS best_hit FROM all_runs WHERE ok = 1 GROUP BY fam, tgroup, hw, side, build""")
con.execute("""CREATE VIEW regress AS SELECT fam, tgroup, hw, build, first, n, best,
               ROUND((SELECT MAX(p.best) FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq), 2) AS prior_best,
               ROUND(best / (SELECT MAX(p.best) FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq), 2) AS ratio,
               (SELECT p.build FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq ORDER BY p.best DESC LIMIT 1) AS prior_build
               FROM cell_builds c WHERE side = 'ours' AND n >= 2""")
con.execute("""CREATE VIEW best_ever AS SELECT fam, tgroup, hw, build AS best_build, best, first AS best_first,
               (SELECT build FROM cell_builds l WHERE l.fam = b.fam AND l.tgroup = b.tgroup AND l.hw = b.hw AND l.side = 'ours' ORDER BY l.bseq DESC LIMIT 1) AS newest_build,
               (SELECT best FROM cell_builds l WHERE l.fam = b.fam AND l.tgroup = b.tgroup AND l.hw = b.hw AND l.side = 'ours' ORDER BY l.bseq DESC LIMIT 1) AS newest_best
               FROM cell_builds b WHERE side = 'ours' AND best = (SELECT MAX(best) FROM cell_builds m WHERE m.fam = b.fam AND m.tgroup = b.tgroup AND m.hw = b.hw AND m.side = 'ours')""")
con.execute("DROP TABLE IF EXISTS cell_stats")
con.execute("""CREATE TABLE cell_stats (fam TEXT, tgroup TEXT, prompt TEXT, hw TEXT, side TEXT, build TEXT, n INTEGER, t_min REAL, t_med REAL, t_max REAL, t_sd REAL, spread REAL, pp_min REAL, pp_med REAL, pp_max REAL, hit REAL, first TEXT, bseq INTEGER,
               PRIMARY KEY (fam, tgroup, prompt, hw, side, build))""")
import runstats
runstats.recompute(con, None)
if con.execute("SELECT name FROM sqlite_master WHERE name = 'runs'").fetchone():
    if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
        con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")
    con.execute("UPDATE runs SET stats_ok = 1")
con.execute("""CREATE VIEW coverage AS SELECT fam, prompt, hw, side, SUM(n) AS runs, COUNT(*) AS builds, MAX(build) AS newest FROM cell_stats GROUP BY fam, prompt, hw, side""")   # runs and builds per model and prompt: the gaps
con.commit()
tot = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT build) FROM all_runs").fetchone()
print(f"all_runs: {tot[0]} runs, {tot[1]} to {tot[2]}, {tot[3]} builds (csv {n_csv}, git history +{n_git}, old db +{n_db}) in {a.db}")


if a.write_csv:
    path = os.path.join(ROOT, "fork/bench/run-history.csv")
    snap = open(path, "rb").read()
    cur_rows = list(csv.DictReader(io.StringIO(snap.decode(errors="replace"))))
    header = list(cur_rows[0].keys())
    have = set()
    for r in cur_rows:
        try:
            have.add(((r.get("ts") or "")[:19], fam(r.get("model"), r.get("args")), r.get("test"), r.get("build"), round(float(r.get("tps") or 0), 2)))
        except ValueError:
            pass
    missing = []
    for src, d in raw:
        try:
            key = ((d.get("ts") or "")[:19], fam(d.get("model"), d.get("args")), d.get("test"), d.get("build"), round(float(d.get("tps") or 0), 2))
        except ValueError:
            continue
        if key not in have:
            have.add(key); missing.append(d)
    missing.sort(key=lambda d: d.get("ts") or "")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    for d in missing:
        w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in header})
    now = open(path, "rb").read()   # rows a running benchmark appended meanwhile stay (they follow the snapshot)
    assert now.startswith(snap)
    tmp = path + ".new"
    open(tmp, "wb").write(snap + buf.getvalue().encode() + now[len(snap):])
    os.replace(tmp, path)
    print(f"run-history.csv: {len(cur_rows)} rows kept byte-identical, {len(missing)} missing rows appended")
