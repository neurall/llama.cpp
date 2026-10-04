#!/usr/bin/env python3
"""SQLite view of the run log: what model, prompt test and build gave what speed, and when.
Every run is stored twice: as a row of tools/bench/run-history.csv (append-only, the source of truth, in git) and, written by run.py at the same moment, in tools/runs.db (this database, queryable, not in git).
`load` rebuilds the database from the CSV any time.

  python3 tools/runlog.py load                     rebuild the database from the CSV
  python3 tools/runlog.py best MODEL [TEST]        per build: runs, consistent best t/s and pp, first and last date (MODEL is a substring of the file name)
  python3 tools/runlog.py runs MODEL [TEST] [BUILD] every run, newest first
  python3 tools/runlog.py q "SELECT ..."           any SQL; tables: runs, view cells

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
NUMERIC = {"tps", "pp", "hit", "ppl", "spp", "n_gen", "bno", "pl", "rep", "seed", "ready_s", "wall_s", "rss_kb", "major_faults", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs",
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
    con.execute("DROP VIEW IF EXISTS cells")
    con.execute("""CREATE VIEW cells AS SELECT stem, hw, test, build, MAX(bno) AS bno, COUNT(*) AS n, ROUND(consistent(tps), 2) AS tps, ROUND(consistent(pp), 1) AS pp,
                   ROUND(MAX(tps), 2) AS tps_max, MIN(ts) AS first, MAX(ts) AS last FROM runs WHERE okf = 1 AND tps IS NOT NULL GROUP BY stem, hw, test, build""")
def row_values(d, cols):
    stem = re.sub(r"-0000\d-of-0000\d", "", (str(d.get("model") or "")).split(" ")[0]).removesuffix(".gguf")
    vals = [(num(d.get(c)) if c in NUMERIC else (None if d.get(c) is None else str(d.get(c)))) for c in cols]
    return vals + [stem, 0 if str(d.get("ok")) == "0" else 1]
def insert(d, cols):
    """called by run.py with every run (a dict by column name); never raises into a run"""
    try:
        con = connect(); ensure(con, cols)
        con.execute("INSERT INTO runs VALUES (" + ",".join("?" * (len(cols) + 2)) + ")", row_values(d, cols))
        con.commit(); con.close()
    except Exception as e:
        print("runs.db:", e, file=sys.stderr)
def load():
    if os.path.exists(DB): os.remove(DB)
    con = connect()
    n = 0
    with open(CSV, newline="") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames
        ensure(con, cols)
        for r in rd:
            con.execute("INSERT INTO runs VALUES (" + ",".join("?" * (len(cols) + 2)) + ")", row_values(r, cols)); n += 1
    con.commit(); print(f"{n} runs loaded into {DB}")
def show(cur):
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(str("" if v is None else v).ljust(w[i]) for i, v in enumerate(r)))
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
        elif a[0] == "q":
            show(con.execute(a[1]))
        else:
            print(__doc__)
