"""special experiments: one python file per experiment in this directory (run: python3 run.py experiment NAME [its own options]; no NAME lists them). Results go to bench/experiments.csv (append-only, in git) and the runs.db table experiments,
one row per run, named experiment_<exp>_<chunk>k_<qd>qd_<backend>_<threads>t_<order>[_pick<N>]. This file holds what all of them share."""
import csv, os, re, subprocess, sys, time

ROW = re.compile(r"stream(?:\[(\w+)\])? chunk\s+(\d+)K (?:pick\s+(\d+)% mode (\d) )?(\S+)\s+thr (\d+) qd\s+(\d+) \| (\d+) layers ([\d.]+) GB in ([\d.]+) s = ([\d.]+) GB/s \| layer time first ([\d.]+) s, middle ([\d.]+) s, last ([\d.]+) s \| content (\w+)")
COLS = ["ts", "kind", "name", "backend", "chunk_kb", "pick_pct", "mode", "read_order", "threads", "qd", "layers", "gb", "secs", "gbps", "layer_first_s", "layer_mid_s", "layer_last_s", "content", "commit", "note", "raw"]


def store(lines, ts, a, kind="iobench-stream"):
    """append the parsed streambench lines (skipping ones already stored) to the CSV and the database; returns the number of new rows"""
    import sqlite3
    fork = os.path.dirname(os.path.dirname(os.path.realpath(__file__))); path = os.path.join(fork, "bench", "experiments.csv")
    commit = getattr(a, "commit", None) or subprocess.run(["git", "-C", fork, "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True).stdout.strip()
    have = set()
    if os.path.exists(path):
        with open(path, newline="") as f: have = {(r["kind"], r["ts"], r["raw"]) for r in csv.DictReader(f)}
    rows = []
    for line in lines:
        m = ROW.match(line)
        if not m or (kind, ts, line.strip()) in have: continue
        g = m.groups(); be = g[0] or "uring"
        rows.append({"ts": ts, "kind": kind, "name": f"experiment_{a.exp}_{g[1]}k_{g[6]}qd_{'io_uring' if be == 'uring' else be}_{g[5]}t_{'file' if g[4].startswith('file') else 'rand' if g[4].startswith('random') else g[4][:6]}" + (f"_pick{g[2]}" if g[2] else ""),
                     "backend": be, "chunk_kb": g[1], "pick_pct": g[2] or "100", "mode": g[3] or "2", "read_order": g[4], "threads": g[5], "qd": g[6], "layers": g[7], "gb": g[8], "secs": g[9], "gbps": g[10],
                     "layer_first_s": g[11], "layer_mid_s": g[12], "layer_last_s": g[13], "content": g[14], "commit": commit, "note": a.note or "", "raw": line.strip()})
    if not rows: return 0
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, COLS)
        if new: w.writeheader()
        w.writerows(rows)
    if not os.environ.get("FORK_NO_RUNS_DB"):
        con = sqlite3.connect(os.path.join(fork, "runs.db")); con.execute("CREATE TABLE IF NOT EXISTS experiments (" + ", ".join('"' + c + '"' + (" REAL" if c in ("gb", "secs", "gbps") else "") for c in COLS) + ")")
        con.executemany("INSERT INTO experiments VALUES (" + ",".join("?" * len(COLS)) + ")", [[r[c] for c in COLS] for r in rows]); con.commit(); con.close()
    return len(rows)


def stream_run(a, p, be, ch, o, qd, th):
    """one streambench run, stored the moment it is printed"""
    fork = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    r = subprocess.run([a.bin or os.path.join(fork, "iobench", "streambench"), a.file, a.list, a.dev, ch, o, th, qd, a.layers, p, a.mode], env={**os.environ, "IOB": be}, capture_output=True, text=True)
    n = store(r.stdout.splitlines(), time.strftime("%Y-%m-%d %H:%M:%S"), a)
    print(r.stdout.strip() or r.stderr.strip(), flush=True)
    return n


def db():
    """the runs.db connection (rows as sqlite3.Row)"""
    import sqlite3
    con = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "runs.db")); con.row_factory = sqlite3.Row
    return con


def show(like, cols="name,gbps,secs,layer_first_s,layer_last_s,content"):
    """markdown table of the stored experiment rows whose name matches the SQL LIKE pattern (--show 'experiment_1_maxreads_128k%')"""
    c = cols.split(","); rs = db().execute(f"SELECT {cols} FROM experiments WHERE name LIKE ? ORDER BY name", (like,)).fetchall()
    print("| " + " | ".join(c) + " |\n|" + "---|" * len(c)); [print("| " + " | ".join(str(r[k]) for k in c) + " |") for r in rs]


def import_file(a):
    """import a result file of streambench lines (--import FILE)"""
    n = store(open(a.imp).read().splitlines(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(a.imp))), a)
    print(f"imported {n} rows", file=sys.stderr)


def main(argv):
    """python3 run.py experiment NAME [options]: NAME is a file of this directory with run(argv); no NAME lists them"""
    import importlib
    names = sorted(f[:-3] for f in os.listdir(os.path.dirname(os.path.realpath(__file__))) if re.match(r"exp\d+_.*\.py$", f))   # the other entries of this directory are older stand-alone scripts
    if not argv or argv[0] not in names:
        print("experiments:", ", ".join(names)); return
    importlib.import_module("experiments." + argv[0]).run(argv[1:])
