#!/usr/bin/env python3
"""When does each card's expert upload start and end? usage: nsys_uploads.py TRACE.sqlite [FIRST_LAYER=20] [N_LAYERS=4] [MIN_KERNEL_MS=10]

Time 0 = the first expert upload (host-to-device copy of 1 MB or more) of the prompt pass. Per layer (the window between the first heavy expert matmuls of
consecutive layers on the x16 card), for every card: start, end and duration of its uploads, the bytes, and the start of its first expert matmul, all in ms.
The question it answers: do the two cards upload at the same time, and does a card compute only after its own upload?"""
import sqlite3, sys
path = sys.argv[1]; first = int(sys.argv[2]) if len(sys.argv) > 2 else 20; nl = int(sys.argv[3]) if len(sys.argv) > 3 else 4
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
c = sqlite3.connect(path)
ks = c.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
bursts, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur)
b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
mm = c.execute("select k.start,k.end,k.deviceId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName where n.value like 'mul_mat_q%' and k.start>=? and k.start<=? order by k.start", (s, e)).fetchall()
cp = c.execute("select start,end,deviceId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and bytes>=1048576 and end>=? and start<=? order by start", (s, e)).fetchall()
t0 = min(x[0] for x in cp)
big = [k for k in mm if k[2] == 1 and k[1] - k[0] > thr * 1e6]
starts = [big[i][0] for i in range(0, len(big) - 2, 3)] + [e]
r = lambda x: (x - t0) / 1e6
print("ms from the first expert upload of the pass (t=0)")
print("%-5s %-4s %9s %9s %9s %8s %14s" % ("layer", "card", "up start", "up end", "up dur", "GB", "matmul start"))
for li in range(first, min(first + nl, len(starts) - 1)):
    a, z = starts[li], starts[li + 1]
    for d, nm in ((1, "x16"), (0, "x4")):
        up = [x for x in cp if x[2] == d and x[1] > a and x[0] < z]
        kn = [k for k in mm if k[2] == d and k[1] > a and k[0] < z]
        if up:
            u0, u1 = min(x[0] for x in up), max(x[1] for x in up)
            print("%-5d %-4s %9.0f %9.0f %9.0f %8.2f %14s" % (li, nm, r(u0), r(u1), r(u1) - r(u0), sum(x[3] for x in up) / 1e9, "%.0f" % r(min(k[0] for k in kn)) if kn else "-"))
        else:
            print("%-5d %-4s %9s %9s %9s %8s %14s" % (li, nm, "-", "-", "-", "-", "%.0f" % r(min(k[0] for k in kn)) if kn else "-"))

# raw list: every upload of 1 MB or more in time order for SLICE_MS after SLICE_FROM_MS (env), no layer attribution
import os
f0 = float(os.environ.get("SLICE_FROM_MS", "6000")); sl = float(os.environ.get("SLICE_MS", "400"))
print("\nraw uploads >= 1 MB, %.0f..%.0f ms:" % (f0, f0 + sl))
print("%-5s %9s %9s %8s %7s" % ("card", "start", "end", "dur", "MB"))
for st, en, d, by in cp:
    if f0 <= r(st) < f0 + sl:
        print("%-5s %9.1f %9.1f %8.1f %7.0f" % ("x16" if d == 1 else "x4", r(st), r(en), r(en) - r(st), by / 1e6))
