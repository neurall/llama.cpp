#!/usr/bin/env python3
"""nsys_layer.py report.sqlite: per-layer GPU window anatomy (decode). Kernels of one GPU are clustered into windows
split at idle gaps > GAP us; windows longer than MAIN us are the per-layer attention/merge splits. Prints their mean
length, kernel busy time, kernel count, idle inside the window, and the kernels that dominate them."""
import collections, sqlite3, sys, statistics as st
GAP, MAIN = 40e3, 120e3  # ns
db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("select id, value from StringIds"))
kern = list(db.execute("select deviceId, start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
mem = list(db.execute("select deviceId, start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start")) \
    if db.execute("select count(*) from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMCPY'").fetchone()[0] else []
for dev in sorted({k[0] for k in kern}):
    kk = [k for k in kern if k[0] == dev]
    wins, cur = [], [kk[0]]
    for k in kk[1:]:
        if k[1] - max(x[2] for x in cur[-3:]) > GAP:
            wins.append(cur); cur = [k]
        else:
            cur.append(k)
    wins.append(cur)
    main = [w for w in wins if w[-1][2] - w[0][1] > MAIN]
    other = [w for w in wins if w[-1][2] - w[0][1] <= MAIN]
    if not main:
        continue
    L = [w[-1][2] - w[0][1] for w in main]
    B = [sum(k[2] - k[1] for k in w) for w in main]
    N = [len(w) for w in main]
    print(f"GPU{dev}: {len(main)} main windows: length median {st.median(L)/1e3:.0f} us, kernel busy {st.median(B)/1e3:.0f} us, "
          f"{st.median(N):.0f} kernels, idle inside {st.median([l-b for l,b in zip(L,B)])/1e3:.0f} us "
          f"(~{st.median([ (l-b)/n for l,b,n in zip(L,B,N)])/1e3:.1f} us per kernel boundary); {len(other)} short windows median "
          f"{st.median([w[-1][2]-w[0][1] for w in other])/1e3 if other else 0:.0f} us")
    top = collections.defaultdict(lambda: [0, 0])
    for w in main:
        for k in w:
            nm = S.get(k[4], S.get(k[3], "?"))[:100]
            top[nm][0] += 1; top[nm][1] += k[2] - k[1]
    tot = sum(v[1] for v in top.values())
    for nm, (n, t) in sorted(top.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"   {100*t/tot:5.1f}%  {t/len(main)/1e3:6.1f} us/window  {n/len(main):5.1f}/window  avg {t/n/1e3:6.1f} us  {nm}")
if mem:
    c = collections.defaultdict(lambda: [0, 0, 0])
    for d, a, b, by, kind in mem:
        x = c[(d, kind)]; x[0] += 1; x[1] += b - a; x[2] += by
    print("memcpy (dev, kind): count, total ms, avg us, avg bytes")
    for (d, kind), (n, t, by) in sorted(c.items()):
        print(f"   {d} {kind}: {n}  {t/1e6:.1f} ms  {t/n/1e3:.1f} us  {by/n:.0f} B")
