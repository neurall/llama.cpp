#!/usr/bin/env python3
"""Every millisecond accounted for. usage: nsys_states.py TRACE.sqlite [LAYER|pass] [MIN_KERNEL_MS=10]   (needs the nsys sqlite export of a run)

For each card the time of the window (one layer, or the whole prompt pass) is split into states, one per 0.25 ms, first match wins:
  COMPUTE            a kernel of this card is running
  UPLOAD             an expert upload (copy of 1 MB or more) of this card is running
  OTHER COPY         some other copy of this card (hand-off, ids, small inputs)
and when none of those, IDLE, named by what is going on elsewhere at that moment (first match wins):
  waits for the other card's upload / the other card's compute / the host is blocked in a sync call / the host is launching / the host is in a memcpy call / nothing found
The host = the thread with the most time in cudaStreamSynchronize (the scheduler thread). Prints the table and returns the state arrays (used by nsys_layer_view.py)."""
import sqlite3, sys, bisect
import numpy as np
STATES = ["COMPUTE", "UPLOAD", "OTHER COPY", "idle: other card uploads", "idle: other card computes", "idle: host blocked in sync", "idle: host launching", "idle: host in memcpy call", "idle: nothing found"]
def load(path, thr):
    c = sqlite3.connect(path)
    ks = c.execute("select k.start,k.end,k.deviceId,n.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName order by k.start").fetchall()
    bursts, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
        else: cur.append(k)
    bursts.append(cur); b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
    cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (s, e)).fetchall()
    rt = c.execute("select r.start,r.end,r.globalTid,st.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds st on st.id=r.nameId where r.end>=? and r.start<=?", (s, e)).fetchall()
    sync = {}
    for st, en, tid, nm in rt:
        if nm.startswith("cudaStreamSynchronize"): sync[tid] = sync.get(tid, 0) + en - st
    host = max(sync, key=sync.get) if sync else None
    return dict(b=b, s=s, e=e, cps=cps, rt=[r for r in rt if r[2] == host], thr=thr)
def layers(d):
    b, e = d["b"], d["e"]; thr = d["thr"]
    heavy = {1: [k for k in b if k[3].startswith("mul_mat_q") and k[2] == 1 and k[1] - k[0] > thr * 1e6], 0: [k for k in b if k[3].startswith("mul_mat_q") and k[2] == 0 and k[1] - k[0] > 1.5e6]}
    trip = [heavy[1][i:i + 3] for i in range(0, len(heavy[1]) - 2, 3)]
    def end(i):
        a, z = trip[i][0][0], (trip[i + 1][0][0] if i + 1 < len(trip) else e)
        return max([trip[i][2][1]] + [k[1] for k in heavy[0] if a <= k[0] < z])
    return [end(i) for i in range(len(trip))], heavy
def classify(d, t0, t1, res=0.25e6):
    n = int((t1 - t0) // res) + 1
    def mark(iv):
        m = np.zeros(n, bool)
        for st, en in iv:
            if en < t0 or st > t1: continue
            m[max(0, int((st - t0) // res)):min(n, int((en - t0) // res) + 1)] = True
        return m
    k = {dv: mark([(x[0], x[1]) for x in d["b"] if x[2] == dv]) for dv in (0, 1)}
    up = {dv: mark([(x[0], x[1]) for x in d["cps"] if x[2] == dv and x[3] == 1 and x[4] >= 1 << 20]) for dv in (0, 1)}
    oc = {dv: mark([(x[0], x[1]) for x in d["cps"] if x[2] == dv and not (x[3] == 1 and x[4] >= 1 << 20)]) for dv in (0, 1)}
    hs = mark([(r[0], r[1]) for r in d["rt"] if r[3].startswith(("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize"))])
    hl = mark([(r[0], r[1]) for r in d["rt"] if "Launch" in r[3]])
    hm = mark([(r[0], r[1]) for r in d["rt"] if r[3].startswith("cudaMemcpy")])
    out = {}
    for dv in (0, 1):
        o = 1 - dv; st = np.full(n, 8, np.int8)
        idle = ~(k[dv] | up[dv] | oc[dv])
        st[idle & hm] = 7; st[idle & hl] = 6; st[idle & hs] = 5; st[idle & k[o]] = 4; st[idle & up[o]] = 3
        st[oc[dv] & ~k[dv] & ~up[dv]] = 2; st[up[dv] & ~k[dv]] = 1; st[k[dv]] = 0
        out[dv] = st
    return out, n * res / 1e6
def table(out, T, label):
    print("%s: %.0f ms" % (label, T))
    print("  %-30s %10s %10s" % ("state", "x16 ms", "x4 ms"))
    for i, nm in enumerate(STATES):
        a, b = (out[1] == i).sum() * 0.25, (out[0] == i).sum() * 0.25
        print("  %-30s %10.1f %10.1f" % (nm, a, b))
    print("  %-30s %10.1f %10.1f" % ("total", len(out[1]) * 0.25, len(out[0]) * 0.25))
if __name__ == "__main__":
    path = sys.argv[1]; what = sys.argv[2] if len(sys.argv) > 2 else "pass"; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    d = load(path, thr)
    if what == "pass": table(*classify(d, d["s"], d["e"]), "whole pass")
    else:
        ends, _ = layers(d); L = int(what); table(*classify(d, ends[L - 1], ends[L]), "layer %d" % L)
