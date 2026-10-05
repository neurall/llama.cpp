#!/usr/bin/env python3
"""runq.py: query every run we ever recorded (tools/runs.db table all_runs, built by tools/runlog_all.py).

usage: runq.py MODEL [TEST] [BUILD ...] [--hw pc1] [--side ours|stock|both] [--sort oldest|newest|best] [--regress] [--best]
  MODEL   part of the family label or model name, case-insensitive: mimo, "glm 3.5", iq4, 27b, olmoe
  TEST    chat, t100, tetris, game4, edit4, pf12k ... (omitted: every test of the model)
  BUILD   build names or numbers as patterns (b11707, 12209, rc-final, release-b12): only these builds
  --regress   the builds of the fork whose best is below 0.9 of the best of all earlier builds (the regression list), for the model (and test)
  --best      the best build of each cell and the newest build's number next to it
Default: one line per build (--sort oldest|newest|best), min / median / max over the runs of the build (one run: just the number), runs, pp, hit rate, stock lines marked."""
import argparse, os, sqlite3, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("what", nargs="+")
ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
ap.add_argument("--hw", default="pc1")
ap.add_argument("--side", default="both")
ap.add_argument("--sort", choices=["oldest", "newest", "best"], default="oldest", help="order of the builds within each model and test: oldest first (default), newest first, or best number first")
ap.add_argument("--regress", action="store_true")
ap.add_argument("--best", action="store_true")
a = ap.parse_args()
con = sqlite3.connect(a.db)
model, rest = a.what[0], a.what[1:]
tests = {r[0] for r in con.execute("SELECT DISTINCT tgroup FROM all_runs")}
test = next((x for x in rest if x in tests), None)
builds = [x for x in rest if x != test]
w, p = ["LOWER(fam) LIKE ?", "hw = ?"], [f"%{model.lower()}%", a.hw]
if test:
    w.append("tgroup = ?"); p.append(test)
if a.side != "both":
    w.append("side = ?"); p.append(a.side)
if builds:
    w.append("(" + " OR ".join("build LIKE ?" for _ in builds) + ")"); p += [f"%{b}%" for b in builds]
if a.regress:
    q = f"SELECT fam, tgroup, build, first, n, best, prior_best, ratio, prior_build FROM regress WHERE ratio < 0.9 AND LOWER(fam) LIKE ? AND hw = ?" + (" AND tgroup = ?" if test else "") + " ORDER BY fam, tgroup, first"
    pr = [f"%{model.lower()}%", a.hw] + ([test] if test else [])
    print(f"{'model':16} {'test':8} {'build':28} {'first':10} {'n':>3} {'best':>7} {'prior best':>10} {'ratio':>6}  vs build")
    for r in con.execute(q, pr):
        print(f"{r[0]:16} {r[1]:8} {r[2][:28]:28} {r[3][:10]:10} {r[4]:3d} {r[5]:7.1f} {r[6]:10.1f} {r[7]:6.2f}  {r[8][:28]}")
elif a.best:
    q = "SELECT fam, tgroup, best_build, best, best_first, newest_build, newest_best FROM best_ever WHERE LOWER(fam) LIKE ? AND hw = ?" + (" AND tgroup = ?" if test else "") + " ORDER BY fam, tgroup"
    pr = [f"%{model.lower()}%", a.hw] + ([test] if test else [])
    print(f"{'model':16} {'test':8} {'best build':28} {'best':>7} {'date':10}  {'newest build':28} {'newest':>7}")
    for r in con.execute(q, pr):
        print(f"{r[0]:16} {r[1]:8} {r[2][:28]:28} {r[3]:7.1f} {r[4][:10]:10}  {r[5][:28]:28} {r[6]:7.1f}")
else:
    q = f"SELECT fam, tgroup, side, build, MIN(ts), COUNT(*), MAX(tps), AVG(tps), MAX(pp), MAX(hit), bseq, GROUP_CONCAT(tps) FROM all_runs WHERE ok = 1 AND {' AND '.join(w)} GROUP BY fam, tgroup, side, build ORDER BY fam, tgroup, bseq"
    rows_ = con.execute(q, p).fetchall()
    key = {"oldest": lambda r: (r[0], r[1], r[10]), "newest": lambda r: (r[0], r[1], -r[10]), "best": lambda r: (r[0], r[1], -r[6])}[a.sort]
    cur, last = sorted(rows_, key=key), None
    print(f"{'model':16} {'test':8} {'build':30} {'date':10} {'n':>3} {'min':>6} {'median':>6} {'max':>6} {'pp':>7} {'hit%':>5}")
    for r in cur:
        if (r[0], r[1]) != last:
            last = (r[0], r[1]); print()
        v = sorted(float(x) for x in r[11].split(","))
        lo, med, hi = (f"{v[0]:6.1f}", f"{statistics.median(v):6.1f}", f"{v[-1]:6.1f}") if len(v) >= 2 else ("", f"{v[0]:6.1f}", "")   # one run: just the number
        print(f"{r[0]:16} {r[1]:8} {('[stock] ' if r[2] == 'stock' else '') + r[3][:22]:30} {r[4][:10]:10} {r[5]:3d} {lo:>6} {med:>6} {hi:>6} {(r[8] or 0):7.1f} {(r[9] or 0):5.1f}")
