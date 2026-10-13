#!/usr/bin/env python3
"""Why is the x16 card idle? usage: nsys_gaps.py TRACE.sqlite   (nsys export --type sqlite of a run with the multi-GPU prompt split)

Takes the prompt pass (the longest burst of kernels), lists every period of more than 2 ms in which the x16 card (device 1) runs no kernel, and labels it by what the
GPUs are doing in it: the x4 card's expert upload, the x16's own expert upload, kernels on the x4 card, other copies (the hidden-state hand-off), or nothing on any GPU
(host or launch gaps). Prints the pass time, the card totals and one line per label."""
import sqlite3, sys, collections
def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def ov(iv, a, z):
    t = 0
    for x, y in iv:
        if y <= a: continue
        if x >= z: break
        t += min(y, z) - max(x, a)
    return t
c = sqlite3.connect(sys.argv[1])
ks = c.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
bursts, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur); b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
k16 = merge([[k[0], k[1]] for k in b if k[2] == 1]); k4 = merge([[k[0], k[1]] for k in b if k[2] == 0])
cps = c.execute("select deviceId,copyKind,bytes,start,end from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (s, e)).fetchall()
up16 = merge([[x[3], x[4]] for x in cps if x[0] == 1 and x[1] == 1 and x[2] >= 1 << 20])
up4 = merge([[x[3], x[4]] for x in cps if x[0] == 0 and x[1] == 1 and x[2] >= 1 << 20])
x4o = merge([[x[3], x[4]] for x in cps if x[0] == 0 and not (x[1] == 1 and x[2] >= 1 << 20)])
x16o = merge([[x[3], x[4]] for x in cps if x[0] == 1 and not (x[1] == 1 and x[2] >= 1 << 20)])
gaps, end = [], s
for x, y in k16:
    if x - end > 2e6: gaps.append((end, x))
    end = max(end, y)
cat, cnt = collections.defaultdict(float), collections.Counter()
for a, z in gaps:
    d = z - a
    parts = [("x4 expert upload", ov(up4, a, z)), ("x16 expert upload", ov(up16, a, z)), ("x4 kernels", ov(k4, a, z)),
             ("other copies (hand-off)", ov(x4o, a, z) + ov(x16o, a, z))]
    lab, v = max(parts, key=lambda p: p[1])
    if v < 0.2 * d: lab = "nothing on any GPU (host/launch)"
    cat[lab] += d / 1e6; cnt[lab] += 1
w = (e - s) / 1e9
bk = sum(y - x for x, y in k16) / 1e9; b4 = sum(y - x for x, y in k4) / 1e9
up16g = sum(x[2] for x in cps if x[0] == 1 and x[1] == 1) / 1e9; up4g = sum(x[2] for x in cps if x[0] == 0 and x[1] == 1) / 1e9
print("pass %.2f s | x16 kernels %.2f s, idle %.2f s | x4 kernels %.2f s | uploaded: x16 %.1f GB, x4 %.1f GB" % (w, bk, w - bk, b4, up16g, up4g))
print("x16 idle gaps > 2 ms: %d, %.2f s" % (len(gaps), sum(z - a for a, z in gaps) / 1e9))
for k, v in sorted(cat.items(), key=lambda x: -x[1]): print("   %-34s %3d gaps %6.2f s" % (k, cnt[k], v / 1e3))
