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
