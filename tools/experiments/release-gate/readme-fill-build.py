#!/usr/bin/env python3
"""Fill the EMPTY cells of the README table (PC1 rows) from the run log with the numbers of one earlier build, shown italic.
usage: readme-fill-build.py README HISTORY.csv BUILD_PREFIX     e.g. b11707
Best run of that build against best stock run, per model: short = game4, t100, chat; long = edit4, pf12k (the source with the best gain wins);
only '-' cells are filled: gains, and the prompt-processing values of stock and ours."""
import csv, re, sys
NB = " "
readme, hist, BP = sys.argv[1], sys.argv[2], sys.argv[3]
SRC = {"short": ("game4", "t100", "chat"), "long": ("edit4", "pf12k")}
best = {}
vals = {}
import statistics
def consistent(xs):
    """the best value that at least two runs agree on (within 5%); a lone outlier does not count, without agreement the median"""
    xs = sorted(xs, reverse=True)
    if len(xs) < 2:
        return xs[0] if xs else 0.0
    for x in xs:
        if sum(1 for y in xs if y >= 0.95 * x) >= 2:
            return x
    return statistics.median(xs)
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
    b = r["build"]
    side = "stock" if b.startswith("stock") else "ours" if b.startswith(BP) else None
    if side:
        vals.setdefault((stem, r["test"], side), []).append((tps, pp))
for k, xs in vals.items():
    best[k] = [consistent([x[0] for x in xs]), consistent([x[1] for x in xs]), len(xs)]
def pick(stem, kind):
    cand = []
    for t in SRC[kind]:
        o, s = best.get((stem, t, "ours")), best.get((stem, t, "stock"))
        if o and s and o[0] and s[0] and o[2] >= 2 and s[2] >= 2:
            cand.append((o[0] / s[0], t, s, o))
    return max(cand) if cand else None
def fmt(x, y):
    return "-" if not (x and y) else f"{x/y:.1f}x" + ("↓" if x/y < 0.95 else "")
def two(a, b):
    return " " + (a + "<br>" + b).replace(" ", NB) + " "
n = 0
out = []
for l in open(readme).read().split("\n"):
    if l.startswith("| ") and l.count("|") == 9 and not l.startswith(("| ---", "| model")):
        c = l.split("|")   # '' model hardware vram stock ours short long build ''
        stem = c[1].strip().replace(NB, " ").replace("<br>", "").replace(" ", "")
        q = c[3].strip().replace(NB, " ")
        if c[2].strip().startswith("2x3090") and "run" not in q:
            longrow = "12k" in q or "2.2k" in q
            for col, kind in (((7, "long"),) if longrow else ((6, "short"), (7, "long"))):
                cur = c[col].strip().replace(NB, "").split("<br>")
                p = pick(stem, kind)
                if not p:
                    continue
                _, _, s, o = p
                new = [fmt(o[0], s[0]), fmt(o[1], s[1])]
                new = [f"*{x}*" if x != "-" else x for x in new]
                res = [cur[0] if cur[0] != "-" else new[0], cur[1] if cur[1] != "-" else new[1]]
                if res != cur:
                    c[col] = two(*res); n += 1
        l = "|".join(c)
    out.append(l)
open(readme, "w").write("\n".join(out))
print("cells filled:", n)
