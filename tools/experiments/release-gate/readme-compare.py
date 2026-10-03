#!/usr/bin/env python3
"""Side by side: the numbers in README.md (old) against the readme-rerun campaign (new, hot runs: median of n), with % gain.
usage: readme-compare.py [FINAL_BUILD] [STOCK_BUILD]   (build directory names as in run-history.csv)"""
import csv, os, statistics as st, sys
HERE = os.path.dirname(os.path.realpath(__file__))
HIST = os.path.join(HERE, "..", "..", "bench", "run-history.csv")
FINAL = sys.argv[1] if len(sys.argv) > 1 else "release-63bde81ac"
STOCK = sys.argv[2] if len(sys.argv) > 2 else "stock-docker-836d57176"
PREV = "b11707-49fe4b756"
# cell: (model file, match substring, test label, test, metric, README upstream, README fork)
GLM3 = "GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
GLM35 = "GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf"
MIMO = "MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf"
IQ4 = "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
IQ3 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
IQ1 = "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
CELLS = [
    (GLM3, "short chat, decode", "chat", "tps", 11.9, 22.4),
    (MIMO, "short chat, decode", "chat", "tps", 4.6, 10.9),
    (MIMO, "12k prompt, decode", "pf12k", "tps", 4.2, 9.1),
    (MIMO, "12k prompt, processing", "pf12k", "pp", 156, 112),
    (IQ4, "short chat, decode", "chat", "tps", 27.7, 46.5),
    (IQ4, "12k prompt, decode", "pf12k", "tps", 25.3, 42.3),
    (IQ4, "12k prompt, processing", "pf12k", "pp", 500, 538),
    (GLM35, "t100, decode", "t100", "tps", 6.9, 15.1),
    (IQ3, "t100, decode", "t100", "tps", 43.9, 57.1),
    (IQ1, "t100, decode", "t100", "tps", 69.1, 67.5),
]
rows = [r for r in csv.DictReader(open(HIST)) if r.get("campaign") == "readme-rerun"]
SETTLED = int(os.environ.get("SETTLED", "1"))   # the last N runs of a cell (default 1: the 4th start = the last of n=3 after the discarded warm-up, where the state is settled); SETTLED=0: all runs
def med(model, test, build, metric):
    v = []
    for r in sorted(rows, key=lambda r: r.get("ts") or ""):
        if model.split(".gguf")[0] in (r.get("model") or "") and r.get("test") == test and (r.get("build") or "") == build and r.get(metric):
            try: v.append(float(r[metric]))
            except ValueError: pass
    if SETTLED and len(v) > SETTLED:
        v = v[-SETTLED:]
    return (st.median(v), len(v)) if v else (None, 0)
f = lambda x: "-" if x is None else f"{x:.1f}"
pct = lambda a, b: "-" if a is None or not b else f"{(a / b - 1) * 100:+.0f}%"
print("| model | test | README upstream | README fork (b11707) | new stock | new b11707 | **new release** (n) | vs new stock | vs new b11707 | vs README fork |")
print("|---|---|---|---|---|---|---|---|---|---|")
for model, label, test, metric, ru, rf in CELLS:
    s, _ = med(model, test, STOCK, metric); p, _ = med(model, test, PREV, metric); n, k = med(model, test, FINAL, metric)
    print(f"| {model} | {label} | {ru} | {rf} | {f(s)} | {f(p)} | **{f(n)}** ({k}) | {pct(n, s)} | {pct(n, p)} | {pct(n, rf)} |")
