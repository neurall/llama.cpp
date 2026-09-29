#!/usr/bin/env python3
"""tl_layers.py server.log [graph#]: per-layer records from a GGML_SCHED_PROF=2 split timeline (decode graph).

Each MoE layer = [GPU main split (merge + attention + router)] -> [CPU tiny split (input copy = waits for the GPU)]
-> [GPU chain launch] -> [CPU expert split]. Prints per layer: GPU wait (~GPU compute of the main split), transition
overheads, CPU expert time; and totals. Output JSON lines (one per layer) with -j for the replay simulator."""
import re, sys, json
lines = open(sys.argv[1]).read().splitlines()
want = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-j" else 1
js = "-j" in sys.argv
graphs, cur = [], None
for l in lines:
    if "sched-timeline:" in l:
        cur = []; graphs.append(cur); continue
    m = re.match(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(.*)", l)
    if m and cur is not None:
        i, t0, tb, ti, tc, be, nn, ni, names = m.groups()
        cur.append(dict(t0=int(t0), bar=int(tb), inp=int(ti), run=int(tc), be=be, nn=int(nn), ni=int(ni), names=names))
g = graphs[min(want, len(graphs) - 1)]
layers = []
for k, s in enumerate(g):
    m = re.search(r"ffn_moe_gate-(\d+)", s["names"])
    if s["be"] == "CPU" and m and k >= 3:
        tiny, chain, main = g[k - 2], g[k - 1], g[k - 3]
        layers.append(dict(il=int(m.group(1)), gpu_dev=main["be"],
            main_inputs=main["bar"] + main["inp"], main_launch=main["run"],
            gpu_wait=tiny["bar"] + tiny["inp"], tiny_run=tiny["run"],
            chain_launch=chain["bar"] + chain["inp"] + chain["run"],
            cpu=s["bar"] + s["inp"] + s["run"]))
if js:
    for L in layers: print(json.dumps(L))
    sys.exit()
tot = g[-1]["t0"] + g[-1]["run"]
keys = ["main_inputs", "main_launch", "gpu_wait", "tiny_run", "chain_launch", "cpu"]
print(f"graph {want}: {len(g)} splits, {tot/1e3:.2f} ms, {len(layers)} MoE layers")
print("sum per token (ms): " + ", ".join(f"{k} {sum(L[k] for L in layers)/1e3:.2f}" for k in keys))
print("median per layer (us): " + ", ".join(f"{k} {sorted(L[k] for L in layers)[len(layers)//2]}" for k in keys))
rest = tot - sum(sum(L[k] for k in keys) for L in layers)
print(f"outside the MoE layer pattern (dense lead layers, output head, sampling side): {rest/1e3:.2f} ms")
