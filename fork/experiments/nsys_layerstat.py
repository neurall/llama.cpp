#!/usr/bin/env python3
# Per layer and per card, in DEVICE time: how long the card computed, copied, and sat idle, for one prefill pass of an nsys trace
# (nsys export --type sqlite). Kernels are joined to the scheduler split that issued them (NVTX "SPLIT n ..." range holding the launch call),
# so a layer is the span between the owner card's `gate` matmuls of two consecutive layers. Any number of cards.
# usage: layerstat.py trace.sqlite [pass_index=0] [detail_layer=-1]
import sqlite3, sys, bisect, re

db = sys.argv[1]
pass_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
detail = int(sys.argv[3]) if len(sys.argv) > 3 else -1
c = sqlite3.connect(db)

p0 = [r[0] for r in c.execute("select start from NVTX_EVENTS where text like 'SPLIT 0 %' order by start")]
# prefill passes only: the ones with many splits
passes = []
for i, s in enumerate(p0):
    e = p0[i + 1] if i + 1 < len(p0) else s + 5e9
    n, last = c.execute("select count(*), max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'", (s, e)).fetchone()
    if n >= 20 and last - s > 100e6:   # a prompt pass: many splits and a long launch span (decode graphs have as many splits and take a few ms)
        passes.append((s, e, n))
if not passes:
    sys.exit('no prefill pass found')
s, E_, nsp = passes[min(pass_i, len(passes) - 1)]
sp = c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start", (s, E_)).fetchall()
starts = [x[0] for x in sp]
def split_of(t):
    i = bisect.bisect_right(starts, t) - 1
    return i if i >= 0 else None

rt = {}
for cid, st in c.execute("select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?", (s - 1e6, E_)):
    rt[cid] = st
K = []   # start, end, dev, stream, split
for st, en, d, strm, cid in c.execute("select start,end,deviceId,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?", (s, E_ + 3e9)):
    if cid in rt:
        K.append((st, en, d, strm, split_of(rt[cid])))
M = []   # start, end, dev, stream, kind, bytes, split
for st, en, d, strm, cid, ck, b in c.execute("select start,end,deviceId,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?", (s, E_ + 3e9)):
    if cid in rt:
        M.append((st, en, d, strm, ck, b, split_of(rt[cid])))
if not K:
    sys.exit('pass %d: no kernels joined to its splits (CUDA graph replay? trace with --cuda-graph-trace=node)' % pass_i)

devs = sorted(set(k[2] for k in K))
busy_dev = {d: sum(k[1] - k[0] for k in K if k[2] == d) for d in devs}
owner = max(devs, key=lambda d: busy_dev[d])

def union(iv, a, b):
    iv = sorted((max(x, a), min(y, b)) for x, y in iv if y > a and x < b)
    tot = 0; cs = ce = None; out = []
    for x, y in iv:
        if cs is None: cs, ce = x, y
        elif x <= ce: ce = max(ce, y)
        else: out.append((cs, ce)); tot += ce - cs; cs, ce = x, y
    if cs is not None: out.append((cs, ce)); tot += ce - cs
    return tot, out

# layer boundaries: first kernel of the owner's gate split of each layer
first_k = {}
for k in K:
    if k[4] is not None and (k[4] not in first_k or k[0] < first_k[k[4]]): first_k[k[4]] = k[0]
bnd = []
names = {0: 'x4', 1: 'x16'} if False else {}
for i, (st, en, tx) in enumerate(sp):
    m = re.search(r'SPLIT \d+ (\S+) MUL_MAT_ID ffn_moe_gate-(\d+)', tx)
    if m and i in first_k:
        bnd.append((int(m.group(2)), m.group(1), i))
card_of_dev = {}
for i, (st, en, tx) in enumerate(sp):
    m = re.match(r'SPLIT \d+ (\S+) ', tx)
    if m:
        ds = set(k[2] for k in K if k[4] == i)
        if len(ds) == 1: card_of_dev[ds.pop()] = m.group(1)
own_name = card_of_dev.get(owner, 'dev%d' % owner)
lay = sorted((L, i) for L, nm, i in bnd if nm == own_name)
if len(lay) < 3:
    # single card or no range split: use any gate split
    lay = sorted((L, i) for L, nm, i in bnd)
print('trace %s  pass %d of %d: %d splits, host %.1f ms, owner card %s, cards %s' % (db.split('/')[-1], pass_i, len(passes), nsp, (sp[-1][1] - s) / 1e6, own_name,
      ', '.join('%s=dev%d' % (card_of_dev.get(d, '?'), d) for d in devs)))
hdr = 'layer  period'
for d in devs:
    n = card_of_dev.get(d, 'dev%d' % d)
    hdr += '  | %s compute  idle   h2d   d2h' % n
print(hdr)
tot = {d: [0, 0, 0, 0] for d in devs}; totp = 0; nrow = 0
rows = []
for j in range(len(lay) - 1):
    L, i0 = lay[j]; a = first_k[i0]; b = first_k[lay[j + 1][1]]
    line = '%5d  %6.2f' % (L, (b - a) / 1e6)
    for d in devs:
        cb, _ = union([(k[0], k[1]) for k in K if k[2] == d], a, b)
        h2d, _ = union([(m[0], m[1]) for m in M if m[2] == d and m[4] == 1 and m[5] > 65536], a, b)
        d2h, _ = union([(m[0], m[1]) for m in M if m[2] == d and m[4] == 2 and m[5] > 65536], a, b)
        line += '  |   %7.2f %6.2f %5.2f %5.2f' % (cb / 1e6, (b - a - cb) / 1e6, h2d / 1e6, d2h / 1e6)
        if 1 <= j:   # skip the first layer (warm-up of the ring)
            t = tot[d]; t[0] += cb; t[1] += b - a - cb; t[2] += h2d; t[3] += d2h
    if 1 <= j: totp += b - a; nrow += 1
    rows.append(line)
for r in rows: print(r)
if nrow:
    line = ' mean  %6.2f' % (totp / nrow / 1e6)
    for d in devs:
        t = tot[d]
        line += '  |   %7.2f %6.2f %5.2f %5.2f' % (t[0] / nrow / 1e6, t[1] / nrow / 1e6, t[2] / nrow / 1e6, t[3] / nrow / 1e6)
    print(line)
    kern_span = (max(k[1] for k in K) - min(k[0] for k in K)) / 1e6
    print('pass device span %.1f ms (first to last kernel); mean layer %.2f ms x %d layers' % (kern_span, totp / nrow / 1e6, len(lay)))

if detail >= 0:
    idx = [j for j in range(len(lay) - 1) if lay[j][0] == detail]
    if idx:
        j = idx[0]; a = first_k[lay[j][1]]; b = first_k[lay[j + 1][1]]
        print('\nlayer %d in device time, %.2f ms (0 = owner gate start):' % (detail, (b - a) / 1e6))
        ev = []
        # kernel runs per (dev, split)
        runs = {}
        for k in K:
            if k[1] > a and k[0] < b:
                key = (k[2], k[4]); r = runs.get(key)
                if r is None: runs[key] = [k[0], k[1], k[1] - k[0], 1]
                else: r[0] = min(r[0], k[0]); r[1] = max(r[1], k[1]); r[2] += k[1] - k[0]; r[3] += 1
        for (d, si), r in runs.items():
            nm = sp[si][2] if si is not None else '?'
            nm = re.sub(r'^SPLIT (\d+) \S+ ', r'split \1 ', nm)
            ev.append((r[0], r[1], '%-5s COMPUTE %6.2f ms busy %6.2f (%3d kernels)  %s' % (card_of_dev.get(d, 'dev%d' % d), (r[1] - r[0]) / 1e6, r[2] / 1e6, r[3], nm[:44])))
        kinds = {1: 'H2D', 2: 'D2H', 8: 'D2D', 10: 'P2P'}
        for m in M:
            if m[1] > a and m[0] < b and m[5] > 65536:
                si = m[6]; nm = re.sub(r'^SPLIT (\d+) \S+ ', r'split \1 ', sp[si][2]) if si is not None else '?'
                ev.append((m[0], m[1], '%-5s %s     %6.2f ms %7.1f MB %5.1f GB/s stream %-3d issued in %s' % (card_of_dev.get(m[2], 'dev%d' % m[2]), kinds.get(m[4], str(m[4])),
                           (m[1] - m[0]) / 1e6, m[5] / 1e6, m[5] / 1e3 / max(1, m[1] - m[0]), m[3], nm[:40])))
        for e in sorted(ev):
            print('  %7.2f - %7.2f  %s' % ((e[0] - a) / 1e6, (e[1] - a) / 1e6, e[2]))
        for d in devs:
            _, iv = union([(k[0], k[1]) for k in K if k[2] == d], a, b)
            gaps = []; prev = a
            for x, y in iv:
                if x - prev > 0.3e6: gaps.append((prev, x))
                prev = y
            if b - prev > 0.3e6: gaps.append((prev, b))
            print('  %s idle gaps over 0.3 ms: %s' % (card_of_dev.get(d, 'dev%d' % d), ', '.join('%.2f-%.2f (%.2f)' % ((x - a) / 1e6, (y - a) / 1e6, (y - x) / 1e6) for x, y in gaps)))
