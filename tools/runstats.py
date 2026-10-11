"""runstats.py: cell statistics (min, median and max of t/s and pp per model, test, machine, side and build) kept in runs.db table cell_stats.

Records added since the last refresh are marked (runs.stats_ok IS NULL, set by runlog.insert); refresh(con) takes ONLY those: they go into all_runs
(if not there yet), only the cells they belong to are recomputed, then the records are marked done. Called at the start of every runq.py query."""
import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runnorm import fam, bno_of, tgroup, prompt_of


def recompute(con, cells):
    """cells: list of (fam, tgroup, prompt, hw, side, build) to recompute, None = all"""
    where, args = "", []
    if cells is not None:
        where = " AND (fam, tgroup, prompt, hw, side, build) IN (" + ",".join("(?,?,?,?,?,?)" for _ in cells) + ")"
        args = [x for c in cells for x in c]
    groups = {}
    for r in con.execute(f"SELECT fam, tgroup, COALESCE(prompt, tgroup), hw, side, build, tps, pp, hit, ts FROM all_runs WHERE ok = 1{where}", args):
        g = groups.setdefault(r[:6], ([], [], [], []))
        g[0].append(r[6])
        if r[7]:
            g[1].append(r[7])
        if r[8] is not None:
            g[2].append(r[8])
        g[3].append(r[9])
    mm = lambda v: (min(v), statistics.median(v), max(v)) if v else (None, None, None)
    for k, (t, pp, hit, ts) in groups.items():
        sd = statistics.pstdev(t) if len(t) > 1 else None   # observed spread of this model, prompt and build: standard deviation and max/min
        con.execute("INSERT OR REPLACE INTO cell_stats (fam, tgroup, prompt, hw, side, build, n, t_min, t_med, t_max, t_sd, spread, pp_min, pp_med, pp_max, hit, first) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    k + (len(t),) + mm(t) + (sd, max(t) / min(t) if len(t) > 1 and min(t) else None) + mm(pp) + (max(hit) if hit else None, min(ts)))
    # build order: by the time of the first run of the build, over all sides
    firsts = sorted(con.execute("SELECT build, MIN(ts) FROM all_runs GROUP BY build ORDER BY MIN(ts)"))
    for i, (b, _) in enumerate(firsts):
        con.execute("UPDATE cell_stats SET bseq = ? WHERE build = ?", (i, b))
        con.execute("UPDATE all_runs SET bseq = ? WHERE build = ?", (i, b))


def refresh(con):
    """returns the number of new records taken in (0: statistics were current)"""
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    if "all_runs" not in have or "cell_stats" not in have or "runs" not in have:
        return -1   # no full build yet: python3 tools/runlog_all.py
    if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
        con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")   # every record counts as new: the dedupe in all_runs keeps the rows already there
    for col in ("prompt", "rep"):   # a db made before these columns existed
        if col not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
            con.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
    new = con.execute("SELECT ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note, rep, prompt FROM runs WHERE stats_ok IS NULL").fetchall()
    if not new:
        return 0
    cells = set()
    for ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note, rep, prompt in new:
        if not tps:
            continue
        f, tg, pr = fam(model, args), tgroup(test), prompt_of(test, rep, prompt)
        side = "stock" if (build or "").startswith("stock") else "ours"
        if not con.execute("SELECT 1 FROM all_runs WHERE ts = ? AND fam = ? AND test = ? AND build = ? AND ROUND(tps, 2) = ROUND(?, 2)", ((ts or "")[:19], f, test, build, tps)).fetchone():
            con.execute("INSERT INTO all_runs (src, ts, hw, fam, tgroup, test, build, bno, side, tps, pp, hit, ok, args, note, prompt) VALUES ('csv',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        ((ts or "")[:19], hw or "pc1", f, tg, test, build, bno_of(build, bno), side, tps, pp, hit, 0 if str(ok) == "0" else 1, (args or "")[:60], (note or "")[:80], pr))
        cells.add((f, tg, pr, hw or "pc1", side, build))
    recompute(con, sorted(cells))
    con.execute("UPDATE runs SET stats_ok = 1 WHERE stats_ok IS NULL")
    con.commit()
    return len(new)
