#!/usr/bin/env python3
"""SQLite view of the run log: what model, prompt test and build gave what speed, and when.
Every run is stored twice: as a row of tools/bench/run-history.csv (append-only, the source of truth, in git) and, written by run.py at the same moment, in tools/runs.db (this database, queryable, not in git).
`load` rebuilds the database from the CSV any time.

  python3 tools/runlog.py load                     rebuild the database from the CSV
  python3 tools/runlog.py best MODEL [TEST]        per build: runs, consistent best t/s and pp, first and last date (MODEL is a substring of the file name)
  python3 tools/runlog.py runs MODEL [TEST] [BUILD] every run, newest first
  python3 tools/runlog.py import-old [apply]       runs of the old perf.db (archive) that the CSV lacks, appended to the CSV
  python3 tools/runlog.py q "SELECT ..."           any SQL; tables: runs, tune_log, campaign_notes, view cells

consistent best = the highest value that at least two runs agree on within 5% (a lone outlier does not count), else the median."""
import csv, os, re, sqlite3, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "bench", "run-history.csv")
DB = os.path.join(HERE, "runs.db")
class Consistent:
    def __init__(self): self.v = []
    def step(self, x):
        if x is not None: self.v.append(x)
    def finalize(self):
        xs = sorted(self.v, reverse=True)
        if not xs: return None
        if len(xs) < 2: return xs[0]
        for x in xs:
            if sum(1 for y in xs if y >= 0.95 * x) >= 2: return x
        return statistics.median(xs)
def connect():
    con = sqlite3.connect(DB)
    con.create_aggregate("consistent", 1, Consistent)
    return con
NUMERIC = {"p_met", "p_unmet", "p_pending", "p_admit_free", "p_admit_evict", "p_d_admit", "p_d_margin", "p_d_budget", "p_d_victim", "p_d_big", "p_d_dup", "p_regret", "p_ev_unpaid", "p_ev_zero", "n_cell", "tps_min", "tps_med", "tps_max", "pp_min", "pp_med", "pp_max", "tps", "pp", "hit", "ppl", "spp", "n_gen", "bno", "pl", "rep", "seed", "ready_s", "wall_s", "rss_kb", "major_faults", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs",
           "ddr_util", "pcie_util", "cpu_util", "gpu_util", "held_gib", "mem_limit", "pred_hit", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late",
           "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3", "mem_gb", "mem_peak_gbs", "pcie_peak_gbs"}
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def q(c): return '"' + c + '"'
def ensure(con, cols):
    """the runs table has every CSV column (numeric ones REAL), plus stem (file name without split suffix and .gguf) and okf (1 unless the run failed); new columns are added as they appear"""
    have = [r[1] for r in con.execute("PRAGMA table_info(runs)")]
    if not have:
        con.execute("CREATE TABLE runs (" + ", ".join(f"{q(c)} {'REAL' if c in NUMERIC else 'TEXT'}" for c in cols) + ", stem TEXT, okf INTEGER)")
        con.execute("CREATE INDEX IF NOT EXISTS i_runs ON runs (stem, test, build)")
    else:
        for c in cols:
            if c not in have: con.execute(f"ALTER TABLE runs ADD COLUMN {q(c)} {'REAL' if c in NUMERIC else 'TEXT'}")
    if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
        con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")   # NULL = a new record: the cell statistics (runlog_all.py) are stale and rebuilt on the next query (runq.py); the rebuild sets 1
    con.execute("DROP VIEW IF EXISTS cells")
    con.execute("""CREATE VIEW cells AS SELECT stem, hw, test, build, MAX(bno) AS bno, COUNT(*) AS n, ROUND(consistent(tps), 2) AS tps, ROUND(consistent(pp), 1) AS pp,
                   ROUND(MAX(tps), 2) AS tps_max, MIN(ts) AS first, MAX(ts) AS last FROM runs WHERE okf = 1 AND tps IS NOT NULL GROUP BY stem, hw, test, build""")
def ins(cols):
    """named-column INSERT: ALTER TABLE appends new columns after stem/okf, so position is not safe"""
    return "INSERT INTO runs (" + ", ".join(q(c) for c in cols) + ", stem, okf) VALUES (" + ",".join("?" * (len(cols) + 2)) + ")"
def row_values(d, cols):
    stem = re.sub(r"-0000\d-of-0000\d", "", (str(d.get("model") or "")).split(" ")[0]).removesuffix(".gguf")
    vals = [(num(d.get(c)) if c in NUMERIC else (None if d.get(c) is None else str(d.get(c)))) for c in cols]
    return vals + [stem, 0 if str(d.get("ok")) == "0" else 1]
def insert(d, cols):
    """called by run.py with every run (a dict by column name); never raises into a run"""
    try:
        con = connect(); ensure(con, cols)
        con.execute(ins(cols), row_values(d, cols))
        con.commit(); con.close()
    except Exception as e:
        print("runs.db:", e, file=sys.stderr)
EMB = os.path.join(HERE, "bench", "embhot.csv")
def ensure_emb(con):
    con.execute("CREATE TABLE IF NOT EXISTS embhot (ts TEXT, hw TEXT, build TEXT, test TEXT, model TEXT, embhex TEXT, hots TEXT)")
def insert_emb(since, ts, hw, build, test):
    """called by run.py after every run: the embhot.csv lines the run appended (from byte offset since) go into the table embhot; returns the new offset"""
    try:
        size = os.path.getsize(EMB) if os.path.exists(EMB) else 0
        if size <= since: return size
        with open(EMB, newline="") as f:
            f.seek(since); rows = [l for l in f.read().split("\n") if l and not l.startswith("model,")]
        con = connect(); ensure_emb(con)
        for l in rows:
            m = l.split(",", 2)
            if len(m) == 3: con.execute("INSERT INTO embhot VALUES (?,?,?,?,?,?,?)", (ts, hw, build, test, m[0], m[1], m[2]))
        con.commit(); con.close()
        return size
    except Exception as e:
        print("embhot:", e, file=sys.stderr); return since
def load():
    if os.path.exists(DB): os.remove(DB)
    con = connect()
    n = 0
    with open(CSV, newline="") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames
        ensure(con, cols)
        for r in rd:
            con.execute(ins(cols), row_values(r, cols)); n += 1
    ensure_emb(con)
    if os.path.exists(EMB):
        with open(EMB, newline="") as f:
            for l in f.read().split("\n"):
                m = l.split(",", 2)
                if len(m) == 3 and m[0] != "model": con.execute("INSERT INTO embhot (model, embhex, hots) VALUES (?,?,?)", tuple(m))
    archive = os.path.join(HERE, "bench", "archive", "perf-2026-09-30.db")
    if os.path.exists(archive): attach_old(con, archive)
    con.commit(); print(f"{n} runs loaded into {DB}")
def show(cur):
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(str("" if v is None else v).ljust(w[i]) for i, v in enumerate(r)))
def import_old(path, apply=False):
    """one-shot: runs of the old perf.db (tools/bench/archive/perf-2026-09-30.db, the database before the CSV) that the CSV does not have yet are appended to the CSV (campaign 'perf-db'); the old tune_log lines and campaign notes go into tables of runs.db"""
    old = sqlite3.connect(path); old.row_factory = sqlite3.Row
    with open(CSV, newline="") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; have = list(rd)
    def key(ts, b, t, tps):
        try: return ((ts or "")[:16], str(b)[:12], t, round(float(tps), 2))
        except (TypeError, ValueError): return ((ts or "")[:16], str(b)[:12], t, None)
    inc = {key(r["ts"], r["build"], r["test"], r["tps"]) for r in have}
    out = []
    for r in old.execute("SELECT * FROM runs ORDER BY ts"):
        if key(r["ts"], r["build"], r["test"], r["tps"]) in inc: continue
        d = {c: "" for c in cols}
        d.update(ts=r["ts"], commit=r["commit_sha"], bno=r["build_no"], tps=r["tps"], pp=r["pp_tps"], hit=r["hit_rate"], model=r["model"], args=r["args"], test=r["test"], hw=r["hw"],
                 ok=r["ok"], build=r["build"], md5=r["md5"], campaign="perf-db", note=r["note"], ppl=r["ppl"], spp=r["s_per_pass"], n_gen=r["n_gen"], origin=r["origin"], env=r["env"], cmd=r["args"], proto="v1")
        out.append({k: ("" if v in (None, "None") else v) for k, v in d.items()})
    print(len(out), "runs of the old perf.db are not in the CSV yet")
    if apply and out:
        with open(CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writerows(out)
    return out

def attach_old(con, path):
    old = sqlite3.connect(path)
    con.execute("DROP TABLE IF EXISTS tune_log"); con.execute("CREATE TABLE tune_log (run_ts TEXT, run_build TEXT, tag TEXT, line TEXT)")
    for run_id, tag, line in old.execute("SELECT run_id, tag, line FROM tune_log"):
        r = old.execute("SELECT ts, build FROM runs WHERE id = ?", (run_id,)).fetchone() or (None, None)
        con.execute("INSERT INTO tune_log VALUES (?,?,?,?)", (r[0], r[1], tag, line))
    con.execute("DROP TABLE IF EXISTS campaign_notes"); con.execute("CREATE TABLE campaign_notes (name TEXT, start_ts TEXT, note TEXT, conclusion TEXT, why TEXT)")
    for row in old.execute("SELECT name, start_ts, note, conclusion, why FROM campaigns"):
        con.execute("INSERT INTO campaign_notes VALUES (?,?,?,?,?)", row)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "load":
        load()
    else:
        if not os.path.exists(DB): load()
        con = connect()
        if a[0] == "best":
            q = "SELECT test, build, bno, hw, n, tps, pp, tps_max, substr(first,1,10) AS first, substr(last,1,10) AS last FROM cells WHERE stem LIKE ?" + (" AND test = ?" if len(a) > 2 else "") + " ORDER BY test, last DESC"
            show(con.execute(q, ("%" + a[1] + "%",) + ((a[2],) if len(a) > 2 else ())))
        elif a[0] == "runs":
            q = "SELECT substr(ts,1,16) AS ts, hw, test, build, rep, tps, pp, hit, campaign FROM runs WHERE stem LIKE ? AND okf = 1" + (" AND test = ?" if len(a) > 2 else "") + (" AND build LIKE ?" if len(a) > 3 else "") + " ORDER BY ts DESC"
            show(con.execute(q, ("%" + a[1] + "%",) + tuple(("%" + x + "%" if i == 1 else x) for i, x in enumerate(a[2:]))))
        elif a[0] == "import-old":
            import_old(os.path.join(HERE, "bench", "archive", "perf-2026-09-30.db"), apply=len(a) > 1 and a[1] == "apply")
        elif a[0] == "q":
            show(con.execute(a[1]))
        else:
            print(__doc__)
