#!/usr/bin/env python3
"""tr_summary.py trace.json: per-token summary of a LLAMA_MOE_CACHE_TRACE file (tokens, CPU expert time,
per-worker busy time, uploads, late/dropped, queue waits, how far ahead uploads landed)."""
import json, sys, statistics as st, re
from collections import defaultdict
ev = json.load(open(sys.argv[1]))["traceEvents"]
names = {e["tid"]: e["args"]["name"] for e in ev if e.get("ph") == "M"}
ev = [e for e in ev if e.get("ph") != "M"]
evx = [e for e in ev if e.get("ph") != "C"]
tok = [e for e in ev if e["name"].startswith("token ")]
n = max(1, len(tok))
T = sum(e["dur"] for e in tok)
print(f"{len(tok)} tokens, {T/n/1e3:.1f} ms/token")
cpu = sum(e["dur"] for e in ev if e["name"] == "cpu experts")
print(f"CPU experts {cpu/n/1e3:.1f} ms/token ({100*cpu/max(T,1):.0f}%)")
for tid in sorted(t for t in names if t >= 10):
    up = [e for e in ev if e["tid"] == tid and e.get("dur") is not None and e["name"].startswith("L")]
    busy = sum(e["dur"] for e in up)
    kinds = defaultdict(int)
    for e in up:
        kinds[e["name"].split()[2]] += 1
    late = sum(1 for e in up if "LATE" in e["name"])
    q = [e["args"]["queued us"] for e in up]
    ah = [e["args"]["ahead at finish"] for e in up if e["name"].split()[2] in ("stream", "predicted")]
    drop = sum(1 for e in ev if e["tid"] == tid and e["name"].startswith("drop"))
    gate = sum(e["dur"] for e in ev if e["tid"] == tid and e["name"] == "gate wait")
    print(f"{names[tid]}: busy {busy/n/1e3:.1f} ms/token ({100*busy/max(T,1):.0f}%), {len(up)/n:.1f} uploads/token {dict(kinds)}, "
          f"late {late}, dropped {drop/n:.1f}/token, queue wait median {st.median(q)/1e3 if q else 0:.2f} ms, "
          f"ahead at finish {'median %d' % st.median(ah) if ah else '-'}, gate {gate/n/1e3:.2f} ms/token")
r = [e for e in ev if e["name"].startswith("router L")]
miss = [int(re.search(r": (\d+) miss", e["name"]).group(1)) for e in r]
sh = sum(e["args"].get("streamed hits", 0) for e in r)
print(f"router: {sum(miss)/n:.1f} misses/token, {sh/n:.1f} streamed hits/token")
pub = [e for e in ev if e["name"].startswith("publish")]
print(f"publishes {len(pub)/n:.1f}/token, LATE {sum('LATE' in e['name'] for e in pub)/n:.1f}/token")
# DDR demand counter: time-weighted average and share of time near the limit
c = sorted((e["ts"], e["args"]["GB/s"]) for e in ev if e.get("ph") == "C")
if len(c) > 1:
    tot = sum(v * (c[i + 1][0] - t) for i, (t, v) in enumerate(c[:-1]))
    span = c[-1][0] - c[0][0]
    hi = sum(c[i + 1][0] - t for i, (t, v) in enumerate(c[:-1]) if v > 30)
    print(f"DDR demand: avg {tot/max(span,1):.1f} GB/s, peak {max(v for _, v in c):.1f}, >30 GB/s {100*hi/max(span,1):.0f}% of the time")

# idle-gap report: inside the traced tokens, when is DDR idle (no CPU expert phase, no upload running) and
# which link sits idle while the CPU doesn't read either (a missed chance to prefetch)
def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
def length(iv):
    return sum(b - a for a, b in iv)
def minus(span, iv):
    a0, b0 = span
    gaps, cur = [], a0
    for a, b in iv:
        if b <= a0 or a >= b0:
            continue
        if a > cur:
            gaps.append([cur, a])
        cur = max(cur, b)
    if cur < b0:
        gaps.append([cur, b0])
    return gaps
if tok:
    span = (min(e["ts"] for e in tok), max(e["ts"] + e["dur"] for e in tok))
    cpu_iv = union([(e["ts"], e["ts"] + e["dur"]) for e in ev if e["name"] == "cpu experts"])
    up_iv = {tid: union([(e["ts"], e["ts"] + e["dur"]) for e in ev if e["tid"] == tid and e.get("dur") is not None and e["name"].startswith("L")])
             for tid in names if tid >= 10}
    busy = union(cpu_iv + [x for v in up_iv.values() for x in v])
    idle = minus(span, busy)
    n = max(1, len(tok))
    big = sorted((b - a for a, b in idle), reverse=True)
    print(f"DDR idle (no CPU phase, no upload): {length(idle)/n/1e3:.1f} ms/token ({100*length(idle)/(span[1]-span[0]):.0f}%), "
          f"{len(idle)/n:.0f} gaps/token, largest {', '.join(f'{g/1e3:.2f}' for g in big[:5])} ms")
    cpu_idle = minus(span, cpu_iv)
    for tid, iv in up_iv.items():
        free = minus(span, union(cpu_iv + iv))       # neither the CPU nor this link busy
        print(f"  {names[tid]}: idle while the CPU is idle too {length(free)/n/1e3:.1f} ms/token "
              f"= room for ~{length(free)/n/(370 if 'fast' in names[tid] else 1480):.0f} more uploads/token")
