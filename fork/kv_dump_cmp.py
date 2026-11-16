#!/usr/bin/env python3
"""Compare two KV dumps of one prompt (--moe kv-dump=N writes kv-dump-<card>-<k>.bin): per layer and K/V, how far the rows of A are from those of B.

usage: kv_dump_cmp.py A.bin B.bin --row 131072 --seg 4096 [--bands 8]

A combined row holds each layer's K and V back to back (L0 K, L0 V, L1 K, ...), --seg bytes of f16 each. A row counts as bad when its mean absolute
difference is above 5% of B's mean absolute value. The bands split the rows in use into equal runs, so a fault that starts at some cell shows where."""
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--row", type=int, required=True, help="bytes per combined row (the kv-dump log line)")
p.add_argument("--seg", type=int, required=True, help="bytes of one layer's K (or V) in a row")
p.add_argument("--bands", type=int, default=8)
a = p.parse_args()

x = np.fromfile(a.a, dtype=np.uint8)
y = np.fromfile(a.b, dtype=np.uint8)
cells = min(len(x), len(y)) // a.row
print(f"A {len(x) // a.row} cells, B {len(y) // a.row} cells: comparing {cells}")
x = x[:cells * a.row].reshape(cells, a.row)
y = y[:cells * a.row].reshape(cells, a.row)
edges = np.linspace(0, cells, a.bands + 1).astype(int)
print(f"{'seg':>8} {'mean rel':>9} {'bad rows':>9} {'first bad':>9}  rel per band of {cells // a.bands} rows")
for s in range(a.row // a.seg):
    u = np.ascontiguousarray(x[:, s * a.seg:(s + 1) * a.seg]).view(np.float16).astype(np.float32)
    v = np.ascontiguousarray(y[:, s * a.seg:(s + 1) * a.seg]).view(np.float16).astype(np.float32)
    rel = np.abs(u - v).mean(axis=1) / (np.abs(v).mean(axis=1) + 1e-6)
    rel = np.where(np.isfinite(rel), rel, 1e9)
    bad = np.nonzero(rel > 0.05)[0]
    bands = " ".join(f"{rel[edges[i]:edges[i + 1]].mean():.3f}" if edges[i + 1] > edges[i] else "-" for i in range(a.bands))
    print(f"L{s // 2:>3} {'KV'[s % 2]}  {rel.mean():9.4f} {len(bad):9d} {bad[0] if len(bad) else '-':>9}  {bands}")
