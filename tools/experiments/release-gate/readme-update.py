#!/usr/bin/env python3
"""Update the README table from the new tests (campaign newtests in run-history.csv): short = game4, long = edit4, PC1 rows (2x3090) only.
usage: readme-update.py README HISTORY.csv OURS_BUILD
Best run of ours (the published build) and of stock, 4 runs each; stock = the stock build, or the same build with --fork off (campaign newtests-forkoff) when stock cannot load the model.
A cell with a result of the new tests (best of 4 runs, same prompts for every build) is replaced by it outright, whatever it replaces: the new methodology is the better one. Old values stay in git history and in the run log. Cells without new results keep what they have.
A short row that adopts a game4 result also takes its stock and ours numbers and the build."""
import csv, re, sys
MIN_RUNS = 3   # valid runs per side: a repetition whose instruction makes a model answer empty (IQ1_M, long test, 2nd instruction) fails on every build, so 3 is all there is
NB = " "
readme, hist, OURS = sys.argv[1], sys.argv[2], sys.argv[3]
bestv = {}
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
    vals.setdefault((stem, r["test"], side), []).append((tps, pp))
for k, xs in vals.items():
    bestv[k] = [max(x[0] for x in xs), max(x[1] for x in xs), len(xs)]   # the best run of each side
def result(stem, test):
    o = bestv.get((stem, test, "ours"))
    s = bestv.get((stem, test, "stock")) or bestv.get((stem, test, "standin"))
    if o and s and o[2] >= MIN_RUNS and s[2] >= MIN_RUNS and o[0] and s[0]:
        return s, o
def g(o, s):
    return "-" if not (o and s) else f"{o/s:.1f}x" + ("↓" if o/s < 0.95 else "")
def num(x):
    m = re.search(r"([\d.]+)x", x.replace("*", ""))
    return float(m.group(1)) if m else None
def two(a, b):
    return " " + (a + "<br>" + b).replace(" ", NB) + " "
out, changed = [], 0
for l in open(readme).read().split("\n"):
    if l.startswith("| ") and l.count("|") == 9 and not l.startswith(("| ---", "| model")):
        c = l.split("|")   # '' model hardware vram stock ours short long build ''
        stem = c[1].strip().replace(NB, " ").replace("<br>", "").replace(" ", "")
        q = c[3].strip().replace(NB, " ")
        # the 27B MTP row waits for its own run with --spec-type draft-mtp (the plain cells are not the MTP test)
        if c[2].strip().startswith("2x3090") and "MTP" not in stem and not any(k in q for k in ("12k", "2.2k", "run")):
            for col, test in ((6, "game4"), (7, "edit4")):
                res = result(stem, test)
                if not res:
                    continue
                s, o = res
                new = [g(o[0], s[0]), g(o[1], s[1])]
                c[col] = two(*new); changed += 1
                if test == "game4":
                    c[4] = two(f"{s[0]:.1f}", f"{s[1]:.0f}")
                    c[5] = two((f"**{o[0]:.1f}**" if o[0] >= 1.1*s[0] else f"{o[0]:.1f}"), f"{o[1]:.0f}")
                    c[8] = " " + OURS.removeprefix("release-") + " "
        l = "|".join(c)
    out.append(l)
open(readme, "w").write("\n".join(out))
print("cells updated:", changed)
# empty cells that remain take the numbers of b11707 from the run log (italic): that build is hard to beat so far
import os, subprocess
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "readme-fill-build.py"), readme, hist, "b11707"])
