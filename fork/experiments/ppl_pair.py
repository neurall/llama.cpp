#!/usr/bin/env python3
"""paired per-chunk perplexity comparison. usage: ppl_pair.py REF_NAME FILE.out [FILE2.out ...]
Each file holds cells '== name: ...' followed by the line '[1]c1,[2]c2,...' of llama-perplexity (cumulative perplexity after chunk k).
Per chunk mean negative log likelihood: nll_k = k ln c_k - (k-1) ln c_(k-1). For every cell against REF: mean and standard error of the per-chunk
difference in percent of perplexity, and how many chunks came out worse."""
import math, re, sys
ref, cells = sys.argv[1], {}
for f in sys.argv[2:]:
    name = None
    for line in open(f, errors="replace"):
        m = re.match(r"== ([^:]+):", line)
        if m: name = m.group(1).strip(); continue
        if name and line.startswith("[1]"):
            c = [float(x) for x in re.findall(r"\[\d+\]([0-9.]+|nan)", line) if x != "nan"]
            cells[name] = [k * math.log(c[k - 1]) - ((k - 1) * math.log(c[k - 2]) if k > 1 else 0.0) for k in range(1, len(c) + 1)]
            cells[name + "/final"] = c[-1]
            name = None
r = cells[ref]
print(f"| cell | chunks | final perplexity | per-chunk difference to {ref}, % (mean +- s.e.) | chunks worse | largest chunk shift, % |")
print("|---|---|---|---|---|---|")
for n, v in cells.items():
    if n.endswith("/final"): continue
    k = min(len(v), len(r))
    d = [100.0 * (v[i] - r[i]) for i in range(k)]   # d(nll) = relative change of the chunk's perplexity
    mean = sum(d) / k
    se = (sum((x - mean) ** 2 for x in d) / (k - 1) / k) ** 0.5 if k > 1 else float("nan")
    print(f"| {n} | {k} | {cells[n + '/final']:.4f} | {mean:+.3f} +- {se:.3f} | {sum(1 for x in d if x > 1e-9)}/{k} | {max(d, key=abs):+.2f} |")
