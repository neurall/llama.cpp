#!/usr/bin/env python3
"""Rows marked 'earlier' (and old b11xxx labels) get their numbers and build number from the run log.
usage: readme-rows-from-log.py README HISTORY.csv
Per row (PC1, 2x3090): test class from the row (short: chat/t100, 12k rows: pf12k); stock and ours = consistent best of the logged runs (two runs within 5%, else the median),
ours over builds with a build number (cache on); the build column = the build of the best run that agrees with that value. Gains are recomputed from the two."""
import csv, re, sys, statistics
NB = " "
readme, hist = sys.argv[1], sys.argv[2]
runs = []
for r in csv.DictReader(open(hist)):
    try:
        tps, pp = float(r["tps"]), float(r["pp"] or 0)
    except ValueError:
        continue
    if not tps or r["ok"] == "0" or r["hw"] != "pc1":
        continue
    a = r["args"] + " " + r["env"]
    if any(x in a for x in ("cache 0", "--fork off", "-at off", "cache0", "slots=0")):
        continue
    stem = re.sub(r"-0000\d-of-0000\d", "", r["model"].split(" ")[0]).removesuffix(".gguf")
    runs.append(dict(stem=stem, test=r["test"], build=r["build"], bno=r["bno"], tps=tps, pp=pp, stock=r["build"].startswith("stock")))
def consistent(xs):
    xs = sorted(xs, reverse=True)
    if len(xs) < 2:
        return xs[0]
    for x in xs:
        if sum(1 for y in xs if y >= 0.95 * x) >= 2:
            return x
    return statistics.median(xs)
def side(stem, tests, stock):
    """per build the consistent best; stock = the best stock build; ours = the best of the builds from b11700 on (b11707 and newer, plus the release candidates of 10-03): newest builds first, b11707 is the reference"""
    def collect(newest_only):
        groups = {}
        for r in runs:
            if r["stem"] != stem or r["test"] not in tests or r["stock"] != stock:
                continue
            num = int(r["bno"]) if r["bno"].isdigit() else 0
            if not stock and newest_only and not (num >= 11700 or r["build"].startswith("rc-final")):
                continue
            if not stock and num == 0 and not r["build"].startswith("rc-final"):
                continue
            label = f"b{num}" if num > 0 else r["build"].replace("rc-final-", "")[:9]
            groups.setdefault(label, []).append(r)
        return groups
    groups = collect(True)
    if not groups and not stock:   # no run from b11700 on: the newest build that has the test
        groups = collect(False)
        if not groups:
            return None
        newest = max(groups, key=lambda k: int(k[1:]) if k[1:].isdigit() else 0)
        groups = {newest: groups[newest]}
    if not groups:
        return None
    best = None
    for label, rs in groups.items():
        t = consistent([r["tps"] for r in rs]); p = consistent([r["pp"] for r in rs])
        if best is None or t > best[0]:
            best = (t, p, label[1:] if label.startswith("b") else label, len(rs))
    return best
def two(a, b):
    return " " + (a + "<br>" + b).replace(" ", NB) + " "
def gain(o, s):
    return f"{o/s:.1f}x" + ("↓" if o/s < 0.95 else "")
out = []
for l in open(readme).read().split("\n"):
    if l.startswith("| ") and l.count("|") == 9 and not l.startswith(("| ---", "| model")):
        c = l.split("|")   # '' model hardware vram stock ours short long build ''
        stem = c[1].strip().replace(NB, " ").replace("<br>", "").replace(" ", "")
        q = c[3].strip().replace(NB, " ")
        build = c[8].strip()
        if c[2].strip().startswith("2x3090") and "2.2k" not in q and build in ("earlier",):
            tests = ("pf12k",) if "12k" in q else ("chat", "t100")
            s, o = side(stem, tests, True), side(stem, tests, False)
            col = 7 if "12k" in q else 6
            col = 7 if "12k" in q else 6
            if s and o:
                c[4] = two(f"{s[0]:.1f}", f"{s[1]:.0f}")
                c[5] = two(f"**{o[0]:.1f}**" if o[0] >= 1.1 * s[0] else f"{o[0]:.1f}", f"{o[1]:.0f}")
                c[col] = two(f"*{gain(o[0], s[0])}*", f"*{gain(o[1], s[1])}*")
                c[8] = (f" b{o[2]} " if o[2].isdigit() else f" {o[2]} ")
                print(stem[:26].ljust(26), q[:14].ljust(14), "stock %.1f/%.0f ours %.1f/%.0f build b%s (n %d/%d)" % (s[0], s[1], o[0], o[1], o[2], s[3], o[3]))
        l = "|".join(c)
    out.append(l)
open(readme, "w").write("\n".join(out))
