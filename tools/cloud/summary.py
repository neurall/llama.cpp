#!/usr/bin/env python3
"""summary.py [RESULTS_DIR]: the pod matrix as one table per model and test: rows = GPU count, columns = fork cold / prewarm / hot and stock
(decode t/s, prompt t/s, cache hit %), the fork-over-stock ratio of the hot run, and whether the outputs agree (md5)."""
import csv, os, sys, collections
d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = list(csv.DictReader(open(os.path.join(d, "run-history.csv"))))
cell = collections.defaultdict(dict)  # (model, test) -> {(gpus, tag): row}  (the last row of a tag wins)
for r in rows:
    if r.get("campaign") != "pod" or r.get("ok") == "0" or not r.get("tps"):
        continue
    n = int("".join(c for c in r["hw"].replace("pod-", "").split("gpu")[0] if c.isdigit()) or 0)
    tag = (r.get("note") or "").split(" | ")[0] or ("stock" if r["origin"] == "stock" else "fork")
    cell[(r["model"], r["test"])][(n, tag)] = r
f = lambda r, k, fmt="%.1f": (fmt % float(r[k])) if r and r.get(k) not in (None, "") else "-"
for (model, test), c in sorted(cell.items()):
    print(f"\n{model}   test {test}   (decode t/s | prompt t/s | hit %)")
    print(f"{'GPUs':>4}  {'fork cold':>24} {'fork prewarm':>24} {'fork hot':>24} {'stock':>24}  {'hot/stock':>9}  md5")
    for n in sorted({k[0] for k in c}):
        cs = [c.get((n, t)) for t in ("cold", "prewarm", "hot", "stock")]
        txt = [(f"{f(r,'tps')} | {f(r,'pp')} | {f(r,'hit')}" if r else "-") for r in cs]
        ratio = f"{float(cs[2]['tps']) / float(cs[3]['tps']):.2f}x" if cs[2] and cs[3] and float(cs[3]['tps']) > 0 else "-"
        md5 = {r["md5"] for r in cs if r and r.get("md5")}
        print(f"{n:>4}  " + " ".join(f"{t:>24}" for t in txt) + f"  {ratio:>9}  {'same' if len(md5) == 1 else ('differs: ' + ','.join(sorted(md5)) if md5 else '-')}")
