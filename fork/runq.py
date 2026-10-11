#!/usr/bin/env python3
"""runq.py: query every run we ever recorded (fork/runs.db table all_runs, built by tools/runlog_all.py).

usage: runq.py MODEL [TEST] [BUILD ...] [--hw pc1] [--side ours|stock|both] [--sort oldest|newest|best] [--regress] [--best]
  MODEL   part of the family label or model name, case-insensitive: mimo, "glm 3.5", iq4, 27b, olmoe
  TEST    chat, t100, tetris, game4, edit4, pf12k ... (omitted: every test of the model)
  BUILD   build names or numbers as patterns (b11707, 12209, rc-final, release-b12): only these builds
  --regress   the builds of the fork whose best is below 0.9 of the best of all earlier builds (the regression list), for the model (and test)
  --best      the best build of each cell and the newest build's number next to it
Default: one line per build (--sort oldest|newest|best), min / median / max of t/s and of pp over the runs of the build (one run: just the number), runs, best hit rate, stock lines marked."""
import argparse, os, sqlite3, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runstats

ap = argparse.ArgumentParser()
ap.add_argument("what", nargs="+")
ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
ap.add_argument("--hw", default="pc1")
ap.add_argument("--side", default="both")
ap.add_argument("--sort", choices=["oldest", "newest", "best"], default="oldest", help="order of the builds within each model and test: oldest first (default), newest first, or best number first")
ap.add_argument("--prompts", action="store_true", help="one line per prompt (game4-racing, edit4-2 ...) instead of per test: the spread of each prompt on each build")
ap.add_argument("--gaps", action="store_true", help="runs and builds per model and prompt (ours and stock) on the machine: where data is missing")
ap.add_argument("--regress", action="store_true")
ap.add_argument("--best", action="store_true")
a = ap.parse_args()
con = sqlite3.connect(a.db, timeout=60)
new = runstats.refresh(con)   # records added since the last query: only their cells are recomputed
if new < 0:
    sys.exit("no statistics yet: run  python3 tools/runlog_all.py  once (a few minutes: it also reads the git history and the old db files)")
if new:
    print(f"(statistics refreshed for {new} new records)", file=sys.stderr)
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
if a.gaps:
    print(f"{'model':16} {'prompt':18} {'side':5} {'runs':>5} {'builds':>6}")
    for r in con.execute("SELECT fam, prompt, side, runs, builds FROM coverage WHERE hw = ? AND prompt NOT LIKE 'cli:%' AND LOWER(fam) LIKE ? ORDER BY fam, prompt, side", (a.hw, f"%{model.lower()}%")):
        print(f"{r[0]:16} {(r[1] or '')[:18]:18} {r[2]:5} {r[3]:5d} {r[4]:6d}")
elif a.regress:
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
    tg = "COALESCE(prompt, tgroup)" if a.prompts else "tgroup"
    q = f"SELECT fam, {tg}, side, build, MIN(ts), COUNT(*), MAX(tps), AVG(tps), MAX(pp), MAX(hit), bseq, GROUP_CONCAT(tps), GROUP_CONCAT(COALESCE(pp, -1)) FROM all_runs WHERE ok = 1 AND {' AND '.join(w)} GROUP BY fam, {tg}, side, build ORDER BY fam, {tg}, bseq"
    rows_ = con.execute(q, p).fetchall()
    key = {"oldest": lambda r: (r[0], r[1], r[10]), "newest": lambda r: (r[0], r[1], -r[10]), "best": lambda r: (r[0], r[1], -r[6])}[a.sort]
    cur, last = sorted(rows_, key=key), None
    def mmm(txt, w=6):   # min / median / max of a comma separated list (-1 = no value); one run: just the number
        v = sorted(float(x) for x in txt.split(",") if float(x) >= 0)
        if not v:
            return f"{'':>{w}} {'':>{w}} {'':>{w}}"
        if len(v) == 1:
            return f"{'':>{w}} {v[0]:{w}.1f} {'':>{w}}"
        return f"{v[0]:{w}.1f} {statistics.median(v):{w}.1f} {v[-1]:{w}.1f}"
    print(f"{'model':16} {'test/prompt':16} {'build':30} {'date':10} {'n':>3} {'t/s min':>6} {'median':>6} {'max':>6}   {'pp min':>6} {'median':>6} {'max':>6} {'hit%':>5} {'sd':>5} {'max/min':>7}")
    for r in cur:
        if (r[0], r[1]) != last:
            last = (r[0], r[1]); print()
        tv = [float(x) for x in r[11].split(",")]
        sd = f"{statistics.pstdev(tv):5.1f}" if len(tv) > 1 else ""
        sp = f"{max(tv) / min(tv):7.2f}" if len(tv) > 1 and min(tv) else ""
        print(f"{r[0]:16} {r[1][:16]:16} {('[stock] ' if r[2] == 'stock' else '') + r[3][:22]:30} {r[4][:10]:10} {r[5]:3d} {mmm(r[11])}   {mmm(r[12])} {(r[9] or 0):5.1f} {sd:>5} {sp:>7}")
