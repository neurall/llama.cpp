#!/usr/bin/env python3
"""Update the README table from the new tests (campaign newtests in run-history.csv): short4 = game4, long4 = edit4, PC1 rows (2x3090) only.
usage: readme-update.py README HISTORY.csv OURS_BUILD
Each cell shows the best run of each side over all its runs in the log (t/s and pp separately): stock, or the same build with --fork off (campaign newtests-forkoff) when stock cannot load the model, and ours (the published build).
A cell takes the new results only when ours is at least as fast as the number already in it; otherwise it keeps its number and build (every run is in the run log).
A row that gets results also takes the build."""
import csv, re, sys
NB = " "
readme, hist, OURS = sys.argv[1], sys.argv[2], sys.argv[3]
latest = {}
for r in csv.DictReader(open(hist)):
    try:
        tps, pp = float(r["tps"]), float(r["pp"] or 0)
    except ValueError:
        continue
    if not tps or r["ok"] == "0" or r["test"] not in ("game4", "edit4") or r["hw"] != "pc1":
        continue
    stem = re.sub(r"-0000\d-of-0000\d", "", r["model"].split(" ")[0]).removesuffix(".gguf")
    b = r["build"]
    if r["campaign"] == "newtests-forkoff" and b.startswith(OURS):
        side = "standin"
    elif r["campaign"] != "newtests":
        continue
    elif b.startswith("stock"):
        side = "stock"
    elif b.startswith(OURS):
        side = "ours"
    else:
        continue
    try:
        rep = int(float(r["rep"]))
    except ValueError:
        continue   # no repetition number: the prompt of the run is unknown, it cannot be paired
    latest[(stem, r["test"], side, rep)] = (r["ts"], tps, pp)   # the CSV is chronological: the newest run of this build at this prompt wins
def result(stem, test):
    """the best run of each side (t/s and pp separately), over every run of the cell in the log: stock (or the --fork off stand-in) and ours"""
    def best(sides):
        rs = [v for k, v in latest.items() if k[0] == stem and k[1] == test and k[2] in sides]
        return [max(v[1] for v in rs), max(v[2] for v in rs), len(rs)] if rs else None
    s, o = best(("stock",)) or best(("standin",)), best(("ours",))
    if s and o and s[0] and o[0]:
        return s, o
def g(o, s):
    return "-" if not (o and s) else f"{o/s:.1f}x" + ("↓" if o/s < 0.95 else "")
def num(x):
    m = re.search(r"([\d.]+)x", x.replace("*", ""))
    return float(m.group(1)) if m else None
def two(a, b):
    return " " + (a + "<br>" + b).replace(" ", NB) + " "
def cells(s, o):
    """four cells: t/s (stock over ours), pp (stock over ours), and the gain cell (t/s over pp)"""
    ours_t = f"**{o[0]:.1f}**" if o[0] >= 1.1*s[0] else f"{o[0]:.1f}"
    return (f" {s[0]:.1f}<br>{ours_t} ", f" {s[1]:.0f}<br>{o[1]:.0f} ", f" {g(o[0], s[0])}<br>{g(o[1], s[1])} ")
out, changed = [], 0
# columns: '' model hardware vram | short4 t/s, pp | long4 t/s, pp | short4 gain, long4 gain | build ''
for l in open(readme).read().split("\n"):
    if l.startswith("| ") and l.count("|") == 11 and not l.startswith(("| ---", "| model")):
        c = l.split("|")
        stem = c[1].strip().replace(NB, " ").replace("<br>", "").replace(" ", "")
        q = c[3].strip().replace(NB, " ")
        # the 27B MTP row waits for its own runs with --spec-type draft-mtp (the plain cells are not the MTP test)
        if c[2].strip().startswith("2x3090") and "MTP" not in stem and not any(k in q for k in ("run", "of 4")):
            hit = False
            for col, gcol, test in ((4, 8, "game4"), (6, 9, "edit4")):
                res = result(stem, test)
                if res:
                    m = re.findall(r"([\d.]+)", c[col].replace(NB, " ").split("<br>")[-1]) if "<br>" in c[col] else []
                    cur_ours = float(m[0]) if m else 0.0   # ours t/s already in the cell
                    if res[1][0] < cur_ours:
                        continue   # not better: the cell keeps its number and build
                    c[col], c[col + 1], c[gcol] = cells(*res); changed += 1; hit = True
            if hit:
                c[10] = " " + OURS.removeprefix("release-") + " "
        l = "|".join(c)
    out.append(l)
open(readme, "w").write("\n".join(out))
print("cells updated:", changed)
