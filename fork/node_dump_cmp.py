#!/usr/bin/env python3
"""Compare two node dumps of one graph (GGML_SCHED_DUMP=DIR with GGML_SCHED_NAN_CHECK=2 writes DIR/<backend>-g<graph>.bin): node by node, in B's order, how far
A's output is from B's. The first node past --tol is where the two backends part: its kernel, or an input of it that no earlier node wrote (a cache, a weight).

usage: node_dump_cmp.py A.bin B.bin [--tol 0.05] [--show 25]"""
import argparse
import numpy as np

def load(path):
    out, seen = [], {}
    with open(path, "rb") as f:
        while True:
            h = f.read(64 + 4 + 32 + 8)
            if len(h) < 108:
                break
            name = h[:64].split(b"\0")[0].decode(errors="replace")
            typ = int(np.frombuffer(h[64:68], np.int32)[0])
            ne = np.frombuffer(h[68:100], np.int64).tolist()
            nb = int(np.frombuffer(h[100:108], np.uint64)[0])
            data = np.frombuffer(f.read(nb), np.float32 if typ == 0 else np.float16).astype(np.float32)
            k = seen.get(name, 0); seen[name] = k + 1
            out.append(((name, k), ne, data))
    return out

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--tol", type=float, default=0.05)
p.add_argument("--show", type=int, default=25)
a = p.parse_args()
A = {k: (ne, d) for k, ne, d in load(a.a)}
B = load(a.b)
print(f"A {len(A)} nodes, B {len(B)} nodes")
shown = 0
for k, ne, d in B:
    if k not in A:
        continue
    ne_a, x = A[k]
    if ne_a != ne or x.size != d.size:
        print(f"{k[0]}#{k[1]}: shapes differ {ne_a} vs {ne}")
        continue
    fin = np.isfinite(x) & np.isfinite(d)
    rel = float(np.abs(x[fin] - d[fin]).mean() / (np.abs(d[fin]).mean() + 1e-9)) if fin.any() else 0.0
    nonfin = int((~np.isfinite(x)).sum() - (~np.isfinite(d)).sum())
    if rel > a.tol or nonfin != 0:
        print(f"{k[0]:>32}#{k[1]} {str(ne):>26} rel {rel:8.4f}  non-finite A-B {nonfin:+d}  |B| {np.abs(d[fin]).mean():.4g}")
        shown += 1
        if shown >= a.show:
            break
if shown == 0:
    print(f"no node past rel {a.tol}")
