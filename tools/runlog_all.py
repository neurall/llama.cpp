#!/usr/bin/env python3
"""runlog_all.py: every run we ever recorded in one table of tools/runs.db (all_runs), plus views that make a regression visible.

usage: runlog_all.py [--db tools/runs.db] [--readme-source]   (rebuilds all_runs and its views, leaves the other tables alone)

Sources, de-duplicated on (ts to the second, model, test, build, tps):
  csv     tools/bench/run-history.csv of this tree and of every worktree next to it (the same file at other branches)
  gitcsv  every committed version of tools/bench/run-history.csv in git history (all branches), rows later dropped from the file included
  db      the old SQLite files (tools/moe-bench/perf.db from git history, tools/bench/archive/perf-*.db, /p/bw/rels/runs.db, /p/bw/data/**/perf*.db, rc_perf*.db)
  readme  --readme-source: the speed tables of old README versions in git (numbers from before the run log, only where no logged run of that model and test exists)
Normalised columns: fam (model family label), tgroup (test without the cli- prefix and -lv3 suffix), bno (build number from the build name or bno column),
bseq (rank of the build by the time of its first run: orders dev and release-candidate builds that have no number), side (stock / ours), tps, pp, hit, ok.
Views:
  cell_builds   per family, test, machine, side and build: best, median and number of runs, in build order
  regress       for each build of the fork: its best against the best of ALL earlier builds of that cell (ratio), flagged when below 0.9
  best_ever     the build with the best number of each cell, and the newest build's number next to it"""
import argparse, csv, glob, io, os, re, sqlite3, statistics, subprocess, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
ap.add_argument("--readme-source", action="store_true")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = [("GLM-5.3-Flash-GSQ-RCO-3.5bit", "GLM 3.5-bit"), ("GLM-5.3-Flash-GSQ-RCO-3.0bit", "GLM 3.0-bit"), ("MiMo", "MiMo IQ3_XXS"), ("Qwen3.8-Flash-Next-UD-IQ4_XS", "Next IQ4_XS"),
        ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "Next IQ3_S"), ("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", "Next IQ1_M"), ("Qwen3.6-35B-A3B-GSQ", "Qwen3.6 Q2_0"), ("qwen36", "Qwen3.6 Q2_0"),
        ("Qwen3.8-27B-IQ4_NL", "27B IQ4_NL"), ("Qwen3.8-27B-GSQ-RCO-IQ3_S", "27B IQ3_S"), ("Qwen3.8-27B-MTP", "27B Q5_K_M MTP"), ("Qwen3.8-27B", "27B"), ("OLMoE", "OLMoE")]


def fam(model, args=""):
    m = (model or "").strip()
    if not m:
        m = (re.search(r"(?:-m|--model)\s+(\S+)", args or "") or [None, ""])[1]
    m = os.path.basename(m.split(" ")[0])
    for k, v in FAMS:
        if k.lower() in m.lower():
            return v
    return re.sub(r"-0000\d-of-0000\d|\.gguf", "", m)[:28] or "?"


def bno_of(build, bno=""):
    m = re.search(r"b(\d{5})", build or "") or re.search(r"(\d{5})", str(bno or ""))
    return int(m.group(1)) if m else None


def tgroup(t):
    return re.sub(r"-lv3$", "", re.sub(r"^cli-", "", t or ""))


rows, seen = [], set()


def add(src, ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note=""):
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
    rows.append((src, ts, hw or "pc1", fam(model, args), tgroup(test), test, build, bno_of(build, bno), "stock" if (build or "").startswith("stock") else "ours", tps, pp, hit, 0 if str(ok) == "0" else 1, (args or "")[:60], (note or "")[:80]))


def read_csv_text(text, src):
    for r in csv.DictReader(io.StringIO(text)):
        add(src, r.get("ts"), r.get("hw"), r.get("model"), r.get("args"), r.get("test"), r.get("build"), r.get("bno"), r.get("tps"), r.get("pp"), r.get("hit"), r.get("ok", 1), r.get("note"))


# csv: every worktree
for f in sorted(set(glob.glob("/p/bw/*/tools/bench/run-history.csv") + glob.glob("/p/bw/llama.cpp/.claude/worktrees/*/tools/bench/run-history.csv") + [os.path.join(ROOT, "tools/bench/run-history.csv")])):
    read_csv_text(open(f, errors="replace").read(), "csv")
n_csv = len(rows)

# gitcsv: every committed version
seen_blobs = set()
log = subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "tools/bench/run-history.csv", "tools/moe-bench/perf.csv"], capture_output=True, text=True).stdout.split()
for h in log:
    for path in ("tools/bench/run-history.csv", "tools/moe-bench/perf.csv"):
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
            add(src, d.get("ts"), d.get("hw"), d.get("model"), d.get("args"), d.get("test"), d.get("build"), d.get("build_no"), d.get("tps"), d.get("pp_tps"), d.get("hit_rate"), d.get("ok", 1), d.get("note"))
    except sqlite3.Error:
        pass


dbs = glob.glob(os.path.join(ROOT, "tools/bench/archive/*.db")) + glob.glob("/p/bw/rels/runs.db") + glob.glob("/p/bw/data/**/perf*.db", recursive=True) + glob.glob("/p/bw/data/**/rc_perf*.db", recursive=True)
for f in sorted(set(dbs)):
    read_db(f, "db")
for h in subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "tools/moe-bench/perf.db"], capture_output=True, text=True).stdout.split():
    blob = subprocess.run(["git", "-C", ROOT, "cat-file", "-p", f"{h}:tools/moe-bench/perf.db"], capture_output=True).stdout
    if blob:
        with tempfile.NamedTemporaryFile(suffix=".db") as t:
            t.write(blob); t.flush(); read_db(t.name, "db")
n_db = len(rows) - n_csv - n_git

con = sqlite3.connect(a.db)
con.execute("DROP TABLE IF EXISTS all_runs")
con.execute("""CREATE TABLE all_runs (src TEXT, ts TEXT, hw TEXT, fam TEXT, tgroup TEXT, test TEXT, build TEXT, bno INTEGER, side TEXT, tps REAL, pp REAL, hit REAL, ok INTEGER, args TEXT, note TEXT, bseq INTEGER)""")
con.executemany("INSERT INTO all_runs (src,ts,hw,fam,tgroup,test,build,bno,side,tps,pp,hit,ok,args,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
# bseq: rank of the build by the time of its first run
con.execute("""UPDATE all_runs SET bseq = (SELECT COUNT(*) FROM (SELECT build b, MIN(ts) t FROM all_runs GROUP BY build) x WHERE x.t < (SELECT MIN(ts) FROM all_runs y WHERE y.build = all_runs.build))""")
con.execute("CREATE INDEX IF NOT EXISTS i_all ON all_runs (fam, tgroup, hw, side, build)")
for v in ("cell_builds", "regress", "best_ever"):
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
con.commit()
tot = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT build) FROM all_runs").fetchone()
print(f"all_runs: {tot[0]} runs, {tot[1]} to {tot[2]}, {tot[3]} builds (csv {n_csv}, git history +{n_git}, old db +{n_db}) in {a.db}")
