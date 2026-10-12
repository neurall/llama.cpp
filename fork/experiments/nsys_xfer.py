#!/usr/bin/env python3
"""Transfers of one prompt pass from an nsys sqlite export: how many, how big, how fast.

usage: nsys_xfer.py TRACE.sqlite [GAP_S=0.3]

The prompt pass is the longest burst of GPU kernels (bursts are split at gaps longer than GAP_S). For every card and direction (H2D host to device, D2H, D2D inside
a card, P2P card to card) it prints the number of copies, the bytes, the average size, a size histogram, the time the copy engine was busy (union of the copy
intervals) and two bandwidths: bytes / busy time (what the engine achieved while copying) and bytes / pass window (what the pass got out of it). Then the host
side: how many cudaMemcpy*/cudaStreamSynchronize/cudaEventSynchronize calls the pass made and how long the host sat in them.
"""
import sqlite3, sys, collections

KIND = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
BINS = [(0, 64e3, "<64KB"), (64e3, 1e6, "64KB-1MB"), (1e6, 8e6, "1-8MB"), (8e6, 64e6, "8-64MB"), (64e6, 1e18, ">=64MB")]

def merge(iv):
    iv.sort(); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def main():
    path = sys.argv[1]; gap = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 0.3e9
    c = sqlite3.connect(path)
    ks = c.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    bursts, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - cur[-1][1] > gap: bursts.append(cur); cur = [k]
        else: cur.append(k)
    bursts.append(cur)
    b = max(bursts, key=lambda x: x[-1][1] - x[0][0])
    s, e = b[0][0], max(k[1] for k in b)
    win = (e - s) / 1e9
    print("pass window %.2f s" % win)
    rows = c.execute("select deviceId,copyKind,bytes,start,end from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<=?", (s, e)).fetchall()
    groups = collections.defaultdict(list)
    for d, k, by, st, en in rows: groups[(d, KIND.get(k, str(k)))].append((by, st, en))
    print("%-4s %-4s %8s %10s %9s %9s %8s %9s %9s  size histogram (count)" % ("card", "dir", "copies", "GB", "avg MB", "busy s", "busy %", "GB/s busy", "GB/s win"))
    for (d, k), v in sorted(groups.items()):
        tot = sum(x[0] for x in v); n = len(v)
        busy = sum(z - a for a, z in merge([[x[1], x[2]] for x in v])) / 1e9
        hist = [sum(1 for x in v if lo <= x[0] < hi) for lo, hi, _ in BINS]
        print("%-4d %-4s %8d %10.2f %9.2f %9.2f %7.0f%% %9.1f %9.1f  %s" % (d, k, n, tot / 1e9, tot / n / 1e6, busy, 100 * busy / win, tot / 1e9 / max(busy, 1e-9), tot / 1e9 / win,
              " ".join("%s:%d" % (lab, h) for (lo, hi, lab), h in zip(BINS, hist) if h)))
    print("host API calls in the window (all threads):")
    q = ("select st.value,count(*),round(sum(r.end-r.start)/1e6,1) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds st on st.id=r.nameId "
         "where r.start>=? and r.start<=? and (st.value like 'cudaMemcpy%' or st.value like '%Synchronize%' or st.value like 'cudaMemset%' or st.value like '%StreamWaitEvent%') "
         "group by st.value order by 3 desc")
    for name, n, ms in c.execute(q, (s, e)):
        print("  %-30s %7d calls  %9.1f ms  (%.0f us per call)" % (name.split("_v")[0], n, ms, ms * 1e3 / max(n, 1)))

main()
