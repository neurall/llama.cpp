#!/usr/bin/env python3
"""Every millisecond of a prompt pass, per layer and per card, in four parts that add up to the layer time (DEVICE time, nsys trace exported to sqlite):

    compute   at least one kernel of the card runs
    upload    no kernel, but a host -> card copy of the card runs (expert uploads, the write half of a hand-off)
    download  no kernel and no upload, but a card -> host copy runs (the read half of a hand-off)
    wait      nothing of the card runs at all

An upload that runs while the card computes is hidden: it counts as compute, not as upload. A layer is the span between the starts of the expert `gate` matmul
of two consecutive layers on the card that computes the most (kernels joined to the scheduler split that launched them, NVTX "SPLIT n CARD OP name").

usage: nsys_layer4.py TRACE.sqlite [PASS=longest]   (trace: nsys profile --trace=cuda,nvtx --cuda-graph-trace=node, GGML_NVTX=1)"""
import sqlite3, sys, bisect, re

db = sys.argv[1]
c = sqlite3.connect(db)
p0 = [r[0] for r in c.execute("select start from NVTX_EVENTS where text like 'SPLIT 0 %' order by start")]
passes = []
for i, s in enumerate(p0):
    e = p0[i + 1] if i + 1 < len(p0) else s + 9e10
    n, last = c.execute("select count(*), max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'", (s, e)).fetchone()
    if n >= 20 and last - s > 100e6:
        passes.append((s, e, last - s))
if not passes:
    sys.exit("no prompt pass in the trace")
pi = int(sys.argv[2]) if len(sys.argv) > 2 else max(range(len(passes)), key=lambda i: passes[i][2])
S, E_, _ = passes[pi]

sp = c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start", (S, E_)).fetchall()
starts = [x[0] for x in sp]
def split_of(t):
    i = bisect.bisect_right(starts, t) - 1
    return i if i >= 0 else None
rt = {cid: t for cid, t in c.execute("select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?", (S - 20e6, E_))}
K = []   # start, end, device, split
for st, en, d, cid in c.execute("select start,end,deviceId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<? order by start", (S, E_ + 3e9)):
    if cid in rt:
        K.append((st, en, d, split_of(rt[cid])))
M = []   # start, end, device, kind (1 = H2D, 2 = D2H)
for st, en, d, ck in c.execute("select start,end,deviceId,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and copyKind in (1,2)", (S - 1e9, E_ + 3e9)):
    M.append((st, en, d, ck))
if not K:
    sys.exit("no kernels joined to this pass (CUDA graph replay: trace with --cuda-graph-trace=node)")
devs = sorted(set(k[2] for k in K))
name = {}
for d in devs:
    cnt = {}
    for k in K:
        if k[2] == d and k[3] is not None:
            m = re.match(r"SPLIT \d+ (\S+) ", sp[k[3]][2])
            if m: cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
    name[d] = max(cnt, key=cnt.get) if cnt else "dev%d" % d
main = max(devs, key=lambda d: sum(k[1] - k[0] for k in K if k[2] == d))

def measure(iv, a, b):
    iv = sorted((max(x, a), min(y, b)) for x, y in iv if y > a and x < b)
    tot = 0; cs = ce = None
    for x, y in iv:
        if cs is None: cs, ce = x, y
        elif x <= ce: ce = max(ce, y)
        else: tot += ce - cs; cs, ce = x, y
    if cs is not None: tot += ce - cs
    return tot

first_k = {}
for k in K:
    if k[3] is not None and (k[3] not in first_k or k[0] < first_k[k[3]]): first_k[k[3]] = k[0]
lay = []
for i, (st, en, tx) in enumerate(sp):
    m = re.search(r"SPLIT \d+ (\S+) MUL_MAT_ID ffn_moe_gate-(\d+)$", tx)
    if m and m.group(1) == name[main] and i in first_k:
        lay.append((int(m.group(2)), first_k[i]))
lay = sorted(dict(lay).items())   # one start per layer (the first gate split of the layer on the main card)
lay_start = {}
for L, t in lay:
    lay_start[L] = min(t, lay_start.get(L, t))
lay = sorted(lay_start.items())
if len(lay) < 2:
    sys.exit("fewer than 2 layers found")

kk = {d: [(k[0], k[1]) for k in K if k[2] == d] for d in devs}
hh = {d: [(m[0], m[1]) for m in M if m[2] == d and m[3] == 1] for d in devs}
dd = {d: [(m[0], m[1]) for m in M if m[2] == d and m[3] == 2] for d in devs}
ms = lambda v: v / 1e6
order = [main] + [d for d in devs if d != main]
print("%s  pass %d of %d (longest by default)  %d layers  cards: %s" % (db.split("/")[-1], pi, len(passes), len(lay) - 1,
      ", ".join("%s%s" % (name[d], " (most compute)" if d == main else "") for d in order)))
hdr = "layer  period"
for d in order: hdr += "  | %-6s compute  upload download   wait" % name[d]
print(hdr)
tot = {d: [0.0, 0.0, 0.0, 0.0] for d in order}; tot_p = 0.0
for j in range(len(lay) - 1):
    L, a = lay[j]; b = lay[j + 1][1]
    line = "%5d  %6.1f" % (L, ms(b - a))
    for d in order:
        u_k   = measure(kk[d], a, b)
        u_kh  = measure(kk[d] + hh[d], a, b)
        u_khd = measure(kk[d] + hh[d] + dd[d], a, b)
        part = [u_k, u_kh - u_k, u_khd - u_kh, (b - a) - u_khd]
        for i in range(4): tot[d][i] += part[i]
        line += "  |        %7.1f %7.1f  %7.1f %6.1f" % tuple(ms(v) for v in part)
    tot_p += b - a
    print(line)
n = len(lay) - 1
for label, div in (("mean", n), ("total", 1)):
    line = "%5s  %6.1f" % (label, ms(tot_p / div))
    for d in order: line += "  |        %7.1f %7.1f  %7.1f %6.1f" % tuple(ms(v / div) for v in tot[d])
    print(line)
for d in order:
    t = tot[d]
    print("%-6s share of the layers' time: compute %4.1f%%, upload %4.1f%%, download %4.1f%%, wait %4.1f%%" % (name[d], *(100.0 * v / tot_p for v in t)))
