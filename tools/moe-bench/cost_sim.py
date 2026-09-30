#!/usr/bin/env python3
"""cost_sim.py trace.txt POLICY RATE_FAST RATE_SLOW [MARGIN] [SLOTS] [NSLOW]: cache hit rate AND estimated token time on PC1.

Per-layer caches of SLOTS experts. Uploads are rate-limited per layer by a token bucket: RATE_* uploads per layer and
token for layers on the fast (x16) / slow (x4) link (the first NSLOW layers are on the slow link). An upload started at
token t is usable at t+1 (the victim stays mapped until then). Policies:
  lru     every miss is a candidate; victim = least recently used (not used this token)
  fork    score = uses in last 64 tokens + 16 * lifetime/lifetime_max; insert if score > victim's + MARGIN
  slru    segmented LRU (probation / protected 50%), victim from probation first
  lrum    LRU victim, but admit only experts used >= MARGIN times in the last 64 tokens
Token time model, calibrated on glm_pol_default (64.7% hit, 118 misses and 7 uploads per token, 50.1 ms):
  ms = G + misses * CPU_MS + uploads * UP_MS  (G = 21.2 ms rest of the token, CPU_MS = 0.245 ms per missed expert,
  UP_MS = DDR contention per upload, env UP_MS, default 0.06 = 25% of uploads overlap a CPU phase)."""
import sys, re, collections, os
fn, pol, rf, rs = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
margin = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
C = int(sys.argv[6]) if len(sys.argv) > 6 else 99
nslow = int(sys.argv[7]) if len(sys.argv) > 7 else 20
G, CPU_MS, UP_MS = 21.2, 0.245, float(os.environ.get("UP_MS", "0.06"))
per = collections.defaultdict(list)
for l in open(fn):
    p = l.split(); il = int(re.search(r"(\d+)", p[0]).group(1)); per[il].append([int(x) for x in p[1:] if int(x) >= 0])
layers = sorted(per); T = min(len(per[il]) for il in layers)
hit = tot = ups = 0
for li, il in enumerate(layers):
    rate = rs if li < nslow else rf
    seq = per[il]; cache = collections.OrderedDict(); prot = collections.OrderedDict()
    win = collections.Counter(); glob = collections.Counter(); recent = collections.deque(); gmax = 1
    bucket = 0.0; pend = []
    for t in range(T):
        ids = seq[t]
        for e, v in pend:
            if v is not None: cache.pop(v, None); prot.pop(v, None)
            cache[e] = 1
        pend = []
        for e in ids:
            tot += 1
            if e in cache or e in prot: hit += 1
        # bookkeeping
        if len(recent) >= 64:
            for e in recent.popleft(): win[e] -= 1
        recent.append(ids)
        for e in ids:
            win[e] += 1; glob[e] += 1; gmax = max(gmax, glob[e])
            if e in cache:
                cache.move_to_end(e)
                if pol == "slru": del cache[e]; prot[e] = 1
            elif e in prot: prot.move_to_end(e)
        if pol == "slru":
            while len(prot) > C // 2:
                v, _ = prot.popitem(last=False); cache[v] = 1; cache.move_to_end(v, last=False)
        bucket = min(bucket + rate, max(rate, 1.0) * 4)
        sc = lambda x: win[x] + 16.0*glob[x]/gmax
        miss = [e for e in ids if e not in cache and e not in prot]
        if pol == "fork": miss.sort(key=lambda x: -sc(x))
        if pol == "lrum": miss = [e for e in miss if win[e] >= margin]
        taken = set()
        for e in miss:
            if bucket < 1.0: break
            if len(cache) + len(prot) + len(pend) < C:
                pend.append((e, None)); bucket -= 1; ups += 1; continue
            if pol == "fork":
                v = min((x for x in cache if x not in ids and x not in taken), key=sc, default=None)
                if v is None or sc(e) <= sc(v) + margin: continue
            else:
                v = next((x for x in cache if x not in ids and x not in taken), None)
                if v is None: continue
            taken.add(v); pend.append((e, v)); bucket -= 1; ups += 1
misses_tok = (tot - hit) / T; ups_tok = ups / T
print(f"{pol:5s} fast {rf:5.2f} slow {rs:5.2f} margin {margin:3.1f} slots {C:3d} | hit {100.0*hit/tot:5.1f}%  misses/tok {misses_tok:6.1f}  "
      f"uploads/tok {ups_tok:5.1f}  est {G + misses_tok*CPU_MS + ups_tok*UP_MS:5.1f} ms  (UP_MS {UP_MS})", flush=True)
