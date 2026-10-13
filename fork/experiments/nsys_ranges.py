#!/usr/bin/env python3
"""Start-stop ranges per lane and layer. usage: nsys_ranges.py TRACE.sqlite [FIRST_LAYER=20] [N_LAYERS=3] [MIN_KERNEL_MS=10] [MERGE_GAP_MS=3]
Same lanes and time base as nsys_timeline.py (ms from the first expert upload of the pass); neighbouring events closer than MERGE_GAP_MS are one range."""
import sqlite3, sys
path = sys.argv[1]; first = int(sys.argv[2]) if len(sys.argv) > 2 else 20; nl = int(sys.argv[3]) if len(sys.argv) > 3 else 3
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0; gap = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
c = sqlite3.connect(path)
ks = c.execute("select k.start,k.end,k.deviceId,n.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName order by k.start").fetchall()
bursts, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur); b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (s, e)).fetchall()
t0 = min(x[0] for x in cps if x[3] == 1 and x[4] >= 1 << 20)
big = [k for k in b if k[3].startswith("mul_mat_q") and k[2] == 1 and k[1] - k[0] > thr * 1e6]
starts = [big[i][0] for i in range(0, len(big) - 2, 3)] + [e]
r = lambda x: (x - t0) / 1e6
def merged(iv, a, z):
    iv = sorted([x, y] for x, y in iv if y > a and x < z); out = []
    for x, y in iv:
        if out and (x - out[-1][1]) / 1e6 <= gap: out[-1][1] = max(out[-1][1], y)
        else: out.append([x, y])
    return out
lane = {
 "x16 expert upload": [(x[0], x[1]) for x in cps if x[2] == 1 and x[3] == 1 and x[4] >= 1 << 20],
 "x16 quantized matmuls (experts+dense)": [(k[0], k[1]) for k in b if k[2] == 1 and k[3].startswith("mul_mat_q")],
 "x16 other kernels": [(k[0], k[1]) for k in b if k[2] == 1 and not k[3].startswith("mul_mat_q")],
 "x4  expert upload": [(x[0], x[1]) for x in cps if x[2] == 0 and x[3] == 1 and x[4] >= 1 << 20],
 "x4  quantized matmuls (experts+dense)": [(k[0], k[1]) for k in b if k[2] == 0 and k[3].startswith("mul_mat_q")],
 "x4  other kernels": [(k[0], k[1]) for k in b if k[2] == 0 and not k[3].startswith("mul_mat_q")],
 "hand-off/merge copies": [(x[0], x[1]) for x in cps if 120 << 20 <= x[4] <= 140 << 20 and x[3] in (1, 2)],
}
for li in range(first, first + nl):
    a, z = starts[li], starts[li + 1]
    print("layer %d: %.0f - %.0f ms, total %.0f ms" % (li, r(a), r(z), r(z) - r(a)))
    for nm, iv in lane.items():
        m = merged(iv, a, z)
        busy = sum(min(y, z) - max(x, a) for x, y in m) / 1e6
        txt = ", ".join("%.0f-%.0f" % (max(r(x), r(a)), min(r(y), r(z))) for x, y in m[:8]) + (" ..." if len(m) > 8 else "")
        print("   %-22s busy %5.0f ms  %s" % (nm, busy, txt if m else "-"))
