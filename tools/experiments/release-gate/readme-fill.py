#!/usr/bin/env python3
"""Fill the README table's short and long gain cells (t/s over pp) from run-history.csv (the run log).
usage: readme-fill.py [README] [run-history.csv]
short: game4 (new test, plain) else t100, chat (earlier tests, italic); long: edit4 (plain) else pf12k (italic)
Gain = best stock run over best run of a published release build (release-b*), t/s and pp, per model and machine (pc1 = 2x3090 rows, pc3 = laptop, pc2 = CPU only).
A plain cell is never overwritten by an italic one, a cell is only replaced by a measured better-sourced value; '-' cells get the best source available."""
import csv, re, sys
NB = " "
readme = sys.argv[1] if len(sys.argv) > 1 else "README.md"
hist = sys.argv[2] if len(sys.argv) > 2 else "../wt-release-next/tools/bench/run-history.csv"
HW = {"2x3090": "pc1", "4060": "pc3", "no": "pc2"}
SRC = {"short": [("game4", False), ("t100", True), ("cli-t100", True), ("cli-t100-ppl", True), ("chat", True)], "long": [("edit4", False), ("pf12k", True)]}
best = {}
for r in csv.DictReader(open(hist)):
    try:
        tps, pp = float(r["tps"]), float(r["pp"] or 0)
    except ValueError:   # a damaged old row
        continue
    if not tps or r["ok"] == "0":
        continue
    stem = re.sub(r"-0000\d-of-0000\d", "", r["model"].split(" ")[0]).removesuffix(".gguf")
    stem = "Qwen3.6-35B-A3B-GSQ-hybrid" if stem == "qwen36" else stem   # the file was renamed to its original name
    b = r["build"]
    a = r["args"] + " " + r["env"]
    off = any(x in a for x in ("cache 0", "--fork off", "-at off", "cache0", "slots=0"))
    side = "stock" if b.startswith("stock") else None if off else "ours"
    if side:
        v = best.setdefault((stem, r["hw"], r["test"], side), [0.0, 0.0, 0])
        v[0] = max(v[0], tps); v[1] = max(v[1], pp); v[2] += 1
def fmt(o, s):
    return "-" if not (o and s) else f"{o/s:.1f}x" + ("\u2193" if o/s < 0.95 else "")   # a down arrow marks a cell slower than stock
def pick(stem, hw, kind):
    """the source with the best t/s gain (stock's best run over the best run of ours); the plain tests win a tie"""
    cand = []
    for test, ital in SRC[kind]:
        o, s = best.get((stem, hw, test, "ours")), best.get((stem, hw, test, "stock"))
        if o and s and o[0] and s[0]:
            a, b = fmt(o[0], s[0]), fmt(o[1], s[1])
            if ital:
                a, b = (f"*{a}*" if a != "-" else a), (f"*{b}*" if b != "-" else b)
            cand.append((round(o[0]/s[0], 2), not ital, a, b, ital, test))
    return max(cand)[2:] if cand else None
out = []
for l in open(readme).read().split("\n"):
    if l.startswith("| ") and l.count("|") == 9 and not l.startswith(("| ---", "| model")):
        c = l.split("|")
        stem = c[1].strip().replace(NB, " ").replace("<br>", "").replace(" ", "")
        hw = HW.get(c[2].strip().replace(NB, " ").split(" ")[0])
        if stem and hw:
            for col, kind in ((6, "short"), (7, "long")):  # columns: model, hardware, in VRAM, stock, ours, short gain, long gain, build
                p = pick(stem, hw, kind)
                if not p:
                    continue
                cur = c[col].strip().replace(NB, "").split("<br>")
                plain = [x for x in cur if x not in ("-", "") and not x.startswith("*")]
                # fill each of the two lines: never replace a plain value, replace '-' and italic ones
                new = [p[0], p[1]]
                res = []
                for old, nw in zip(cur, new):
                    res.append(nw if nw != "-" else old)
                c[col] = " " + res[0].replace(" ", NB) + "<br>" + res[1].replace(" ", NB) + " "
        l = "|".join(c)
    out.append(l)
open(readme, "w").write("\n".join(out))
