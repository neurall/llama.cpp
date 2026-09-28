#!/usr/bin/env python3
"""Per-GPU busy time and cross-GPU overlap from an nsys report (kernels + memcpys).

usage: nsys_overlap.py report.nsys-rep [t0_s t1_s | -W 0]  (-W: auto-pick the W-second window with the most H2D)
Prints, per device: kernel time, H2D / D2H / D2D copy time and bytes, busy time (union of all its activity);
then the time both devices are busy at once and the longest stretches where exactly one of them is idle.
Only the window [t0, t1] (seconds from the first GPU event) is counted when given.
"""
import csv, io, subprocess, sys
from collections import defaultdict

rep = sys.argv[1]
win = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else None
out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv", "--output", "-", rep],
                     capture_output=True, text=True).stdout
lines = out.splitlines()
hdr = next(i for i, l in enumerate(lines) if l.lstrip('"').startswith("Start (ns)"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[hdr:]))))
ev = []
for r in rows:
    try:
        t0 = int(r["Start (ns)"]); d = int(r["Duration (ns)"])
    except (KeyError, ValueError):
        continue
    dev = r.get("Device") or r.get("GpuId") or "?"
    name = r.get("Name", "")
    kind = "kernel"
    if "HtoD" in name or "Host-to-Device" in name: kind = "H2D"
    elif "DtoH" in name or "Device-to-Host" in name: kind = "D2H"
    elif "DtoD" in name or "PtoP" in name or "Device-to-Device" in name or "Peer" in name: kind = "D2D"
    elif "memset" in name.lower(): kind = "memset"
    try:
        nbytes = float(r.get("Bytes (MB)") or 0) * 1e6
    except ValueError:
        nbytes = 0
    ev.append((t0, t0 + d, dev, kind, nbytes))
if not ev:
    sys.exit("no GPU events")
T0 = min(e[0] for e in ev)
if win and win[0] < 0:
    # auto: the window of length -win[0] s with the most H2D bytes (prefill streams the experts)
    W = -win[0] * 1e9
    h2d = sorted((a, n) for a, b, dv, k, n in ev if k == "H2D")
    best, j, acc, bi = 0.0, 0, 0.0, 0
    for i in range(len(h2d)):
        acc += h2d[i][1]
        while h2d[i][0] - h2d[j][0] > W:
            acc -= h2d[j][1]; j += 1
        if acc > best: best, bi = acc, j
    win = ((h2d[bi][0] - T0) / 1e9, (h2d[bi][0] - T0 + W) / 1e9)
    print(f"auto window {win[0]:.1f}-{win[1]:.1f} s ({best/1e9:.1f} GB H2D)")
if win:
    lo, hi = T0 + win[0] * 1e9, T0 + win[1] * 1e9
    ev = [(max(a, lo), min(b, hi), dv, k, n) for a, b, dv, k, n in ev if b > lo and a < hi]

def union(iv):
    iv = sorted(iv); res = []
    for a, b in iv:
        if res and a <= res[-1][1]: res[-1][1] = max(res[-1][1], b)
        else: res.append([a, b])
    return res

per = defaultdict(list); stats = defaultdict(lambda: defaultdict(float))
for a, b, dv, k, n in ev:
    per[dv].append((a, b)); stats[dv][k] += (b - a) / 1e6; stats[dv][k + "_MB"] += n / 1e6
span = (max(e[1] for e in ev) - min(e[0] for e in ev)) / 1e6
print(f"window {span:.0f} ms")
busy = {}
for dv in sorted(per):
    u = union(per[dv]); busy[dv] = u
    tot = sum(b - a for a, b in u) / 1e6
    s = stats[dv]
    print(f"device {dv}: busy {tot:.0f} ms ({100*tot/span:.0f}%) | kernels {s['kernel']:.0f} ms | H2D {s['H2D']:.0f} ms "
          f"{s['H2D_MB']/1e3:.1f} GB ({s['H2D_MB']/max(s['H2D'],1e-9):.1f} GB/s) | D2H {s['D2H']:.0f} ms | D2D {s['D2D']:.0f} ms | memset {s['memset']:.0f} ms")
devs = sorted(busy)
if len(devs) >= 2:
    a, b = busy[devs[0]], busy[devs[1]]
    i = j = 0; both = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo: both += hi - lo
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    print(f"both devices busy at once: {both/1e6:.0f} ms ({100*both/1e6/span:.0f}% of the window)")
