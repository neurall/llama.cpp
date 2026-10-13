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
    # When does each card's expert upload run? Host-to-device copies of 1 MB or more (expert tensors; the small ones are inputs and ids), per card: first start, last end,
    # bytes, and the share of the time of the shorter of the two uploads during which BOTH cards were copying (100% = fully parallel, 0% = one after the other).
    big = {d: sorted((st, en, by) for (by, st, en) in v if by >= 1 << 20) for (d, k), v in groups.items() if k == "H2D"}
    print("expert uploads (H2D >= 1 MB), time from the start of the pass window:")
    for d, v in sorted(big.items()):
        if v:
            print("  card %d: %5d copies, %8.2f GB, first start %6.2f s, last end %6.2f s, span %6.2f s, copying %6.2f s" %
                  (d, len(v), sum(x[2] for x in v) / 1e9, (v[0][0] - s) / 1e9, (max(x[1] for x in v) - s) / 1e9, (max(x[1] for x in v) - v[0][0]) / 1e9,
                   sum(z - a for a, z in merge([[x[0], x[1]] for x in v])) / 1e9))
    if len(big) >= 2 and all(big.values()):
        ds = sorted(big)
        iv = {d: merge([[x[0], x[1]] for x in big[d]]) for d in ds}
        a_, b_ = iv[ds[0]], iv[ds[1]]
        ov = 0
        for x0, x1 in a_:
            for y0, y1 in b_:
                ov += max(0, min(x1, y1) - max(x0, y0))
        short = min(sum(z - a for a, z in a_), sum(z - a for a, z in b_))
        print("  both cards copying at once: %.2f s = %.0f%% of the shorter card's copying time" % (ov / 1e9, 100 * ov / max(short, 1)))
        # per second of the pass: bytes each card uploaded (a timeline of 1 s buckets)
        nb = int(win) + 1
        print("  GB per second of the pass:    " + " ".join("%5d" % i for i in range(nb)))
        for d in ds:
            bk = [0.0] * nb
            for st, en, by in big[d]:
                bk[min(nb - 1, int((st - s) / 1e9))] += by / 1e9
            print("    card %d                      " % d + " ".join("%5.1f" % x for x in bk))
    print("host API calls in the window (all threads):")
    q = ("select st.value,count(*),round(sum(r.end-r.start)/1e6,1) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds st on st.id=r.nameId "
         "where r.start>=? and r.start<=? and (st.value like 'cudaMemcpy%' or st.value like '%Synchronize%' or st.value like 'cudaMemset%' or st.value like '%StreamWaitEvent%') "
         "group by st.value order by 3 desc")
    for name, n, ms in c.execute(q, (s, e)):
        print("  %-30s %7d calls  %9.1f ms  (%.0f us per call)" % (name.split("_v")[0], n, ms, ms * 1e3 / max(n, 1)))

main()
