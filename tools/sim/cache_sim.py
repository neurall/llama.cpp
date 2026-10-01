#!/usr/bin/env python3
"""cache_sim.py trace.txt [...] --frac 0.25,0.35,0.5: hit rates of per-layer expert caches on a GGML_MOE_LOG routing trace.

Trace lines: "<tensor name> id0 id1 ...", one per ffn_gate_exps op (one token, one layer); concatenated in the order given
(topic switches included). Each layer has its own cache of C = frac * n_experts slots (the fork's layout). Policies:
  lru     least recently used
  lfu     least frequently used (lifetime counts, no decay)
  lfuw    LFU with window decay (counts halve every 512 tokens)
  hot     static: the C most used experts of the WHOLE trace (oracle profile, like a perfect persistent-hot profile)
  belady  offline optimal: evict the expert whose next use is farthest (upper bound for any policy, incl. any predictor)
A miss inserts the expert (evicting per policy); a token's experts are all looked up first, then inserted."""
import sys, re, argparse, collections, heapq

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--frac", default="0.25,0.35,0.5")
ap.add_argument("--skip", type=int, default=0, help="ignore the first N tokens of each file (prompt processing)")
ap.add_argument("--ins", action="store_true", help="hit rate vs inserts allowed per layer and token (upload budget), LRU and fork policy, vs Belady")
ap.add_argument("--glob", action="store_true", help="global upload budget model: B inserts per token over all layers, usable D tokens later; hit rate vs (B, D, policy, margin, victim kept mapped)")
ap.add_argument("--sweep", action="store_true", help="sweep the fork policy window / global weight / margin / inserts at the first --frac")
ap.add_argument("--nexp", type=int, default=0, help="experts per layer (default: max id + 1 in the traces)")
ap.add_argument("--warm", type=int, default=0, help="simulate from token 0 but count hits only from token N of the concatenated trace (steady state)")
a = ap.parse_args()
per = collections.defaultdict(list)  # layer -> list of tuples (token order)
n_exp = 0
for fn in a.files:
    lines = open(fn).read().splitlines()
    cnt = collections.Counter()
    for l in lines:
        p = l.split()
        m = re.search(r"(\d+)", p[0]); il = int(m.group(1))
        ids = [int(x) for x in p[1:] if int(x) >= 0]
        if cnt[il] >= a.skip: per[il].append(tuple(ids))
        cnt[il] += 1
        n_exp = max(n_exp, max(ids) + 1 if ids else 0)
n_exp = a.nexp or n_exp
layers = sorted(per)
print(f"{len(layers)} layers, {len(per[layers[0]])} tokens, {n_exp} experts, top-{len(per[layers[0]][0])}")

def sim_lru(seq, C, max_ins=None):
    cache = collections.OrderedDict(); hit = tot = 0; t_ = 0
    for ids in seq:
        for e in ids:
            if t_ >= a.warm:
                tot += 1
                if e in cache: hit += 1
        t_ += 1
        ins_ = 0
        for e in ids:
            if e in cache: cache.move_to_end(e)
            elif max_ins is not None and ins_ >= max_ins: continue
            else:
                ins_ += 1
                if len(cache) >= C:
                    for v in cache:
                        if v not in ids: del cache[v]; break
                    else: continue
                cache[e] = 1
    return hit, tot

def sim_lfu(seq, C, decay=0):
    cnt = collections.Counter(); cache = set(); hit = tot = 0
    for t, ids in enumerate(seq):
        if decay and t % decay == 0 and t:
            for k in list(cnt): cnt[k] //= 2
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        for e in ids: cnt[e] += 1
        for e in ids:
            if e in cache: continue
            if len(cache) >= C:
                v = min((x for x in cache if x not in ids), key=lambda x: cnt[x], default=None)
                if v is None or cnt[v] > cnt[e]: continue
                cache.remove(v)
            cache.add(e)
    return hit, tot


def sim_fork(seq, C, window=64, k=16.0, margin=2.0, max_ins=2):
    """the fork's default policy (LLAMA_MOE_CACHE_POLICY=add): score = uses in the last `window` tokens + k * lifetime/lifetime_max;
    a miss replaces the lowest-scored cached expert not used this token if its score exceeds the victim's + margin; at most
    max_ins inserts per layer and token; new experts are usable from the next token."""
    glob = collections.Counter(); win = collections.Counter(); recent = collections.deque(); cache = set(); hit = tot = 0; gmax = 1
    for t, ids in enumerate(seq):
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        if len(recent) >= window:
            for e in recent.popleft(): win[e] -= 1
        recent.append(ids)
        for e in ids:
            win[e] += 1; glob[e] += 1; gmax = max(gmax, glob[e])
        sc = lambda x: win[x] + k*glob[x]/gmax
        ins = 0
        for e in sorted((x for x in ids if x not in cache), key=lambda x: -sc(x)):
            if ins >= max_ins: break
            if len(cache) < C: cache.add(e); ins += 1; continue
            v = min((x for x in cache if x not in ids), key=sc, default=None)
            if v is None or sc(e) <= sc(v) + margin: continue
            cache.remove(v); cache.add(e); ins += 1
    return hit, tot


def sim_lruk(seq, C, K=2):
    """LRU-K: evict the cached expert whose K-th most recent use is oldest (K=2 ignores one-off bursts)."""
    hist = collections.defaultdict(lambda: collections.deque(maxlen=K)); cache = set(); hit = tot = 0
    for t, ids in enumerate(seq):
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        for e in ids: hist[e].append(t)
        key = lambda x: (hist[x][0] if len(hist[x]) >= K else -1, hist[x][-1])
        for e in ids:
            if e in cache: continue
            if len(cache) >= C:
                v = min((x for x in cache if x not in ids), key=key, default=None)
                if v is None: continue
                cache.remove(v)
            cache.add(e)
    return hit, tot

def sim_slru(seq, C, prot=0.5):
    """segmented LRU: a hit promotes an expert into the protected segment (prot * C); the probationary segment evicts first."""
    P = int(prot*C); probation = collections.OrderedDict(); protected = collections.OrderedDict(); hit = tot = 0
    for t, ids in enumerate(seq):
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in probation or e in protected: hit += 1
        for e in ids:
            if e in protected: protected.move_to_end(e)
            elif e in probation:
                del probation[e]; protected[e] = 1
                if len(protected) > P:
                    old, _ = protected.popitem(last=False); probation[old] = 1
            else:
                if len(probation) + len(protected) >= C:
                    victim = next((v for v in probation if v not in ids), None)
                    if victim is None:
                        victim = next((v for v in protected if v not in ids), None)
                        if victim is None: continue
                        del protected[victim]
                    else: del probation[victim]
                probation[e] = 1
    return hit, tot

def sim_hot(seq, C):
    cnt = collections.Counter(e for ids in seq for e in ids)
    cache = {e for e, _ in cnt.most_common(C)}
    hit = sum(1 for ids in seq[a.warm:] for e in ids if e in cache); return hit, sum(len(i) for i in seq[a.warm:])

def sim_belady(seq, C):
    nxt = collections.defaultdict(collections.deque)
    for t, ids in enumerate(seq):
        for e in ids: nxt[e].append(t)
    cache = {}; hit = tot = 0  # expert -> next use time
    INF = 1 << 60
    for t, ids in enumerate(seq):
        for e in ids:
            nxt[e].popleft()
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        for e in ids:
            n = nxt[e][0] if nxt[e] else INF
            if e in cache: cache[e] = n; continue
            if len(cache) >= C:
                v = max((x for x in cache if x not in ids), key=lambda x: cache[x], default=None)
                if v is None or cache[v] <= n: continue  # bypass: the newcomer is needed later than everything cached
                del cache[v]
            cache[e] = n
    return hit, tot

for f in [float(x) for x in a.frac.split(",")]:
    C = max(1, int(f*n_exp)); res = {}
    for name, fn in [("lru", sim_lru), ("lru2", lambda s, c: sim_lru(s, c, 2)), ("lfu", lambda s, c: sim_lfu(s, c)), ("lfuw", lambda s, c: sim_lfu(s, c, 512)), ("fork", sim_fork), ("lruk", sim_lruk), ("slru", sim_slru), ("hot", sim_hot), ("belady", sim_belady)]:
        h = t = 0
        for il in layers:
            x, y = fn(per[il], C); h += x; t += y
        res[name] = 100.0*h/t
    print(f"cache {int(100*f):3d}% ({C} slots/layer): " + "  ".join(f"{k} {v:5.1f}%" for k, v in res.items()) +
          f"   | misses vs fork: lru {100*(1-res['lru']/100)/(1-res['fork']/100):.0f}%, lru2 {100*(1-res['lru2']/100)/(1-res['fork']/100):.0f}%, belady {100*(1-res['belady']/100)/(1-res['fork']/100):.0f}%")

if a.sweep:
    f = float(a.frac.split(",")[0]); C = max(1, int(f*n_exp)); rows = []
    for w in (4, 8, 16, 32, 64, 128):
        for k in (0.0, 2.0, 8.0, 16.0):
            for mg in (0.0, 2.0):
                for mi in (2, 4):
                    h = t = 0
                    for il in layers:
                        x, y = sim_fork(per[il], C, window=w, k=k, margin=mg, max_ins=mi); h += x; t += y
                    rows.append((100.0*h/t, w, k, mg, mi))
    rows.sort(reverse=True)
    print(f"sweep at {int(100*f)}% cache ({C} slots): best 8 and the default (window 64, k 16, margin 2, inserts 2)")
    for r in rows[:8]: print(f"  {r[0]:5.1f}%  window {r[1]:3d}  k {r[2]:4.1f}  margin {r[3]:.0f}  inserts {r[4]}")
    d = [r for r in rows if r[1:] == (64, 16.0, 2.0, 2)][0]; print(f"  default {d[0]:5.1f}%  (rank {rows.index(d) + 1} of {len(rows)})")

if a.ins:
    f = float(a.frac.split(",")[0]); C = max(1, int(f*n_exp))
    def run(fn):
        h = t = 0
        for il in layers:
            x, y = fn(per[il], C); h += x; t += y
        return 100.0*h/t
    print(f"inserts per layer and token vs hit rate at {int(100*f)}% cache ({C} slots), warm from token {a.warm}")
    print(f"  belady (unbounded inserts) {run(sim_belady):5.1f}%")
    for mi in (0, 1, 2, 3, 4, 8, 99):
        print(f"  max_ins {mi:2d}: lru {run(lambda s, c: sim_lru(s, c, mi)):5.1f}%   fork(margin 2) {run(lambda s, c: sim_fork(s, c, max_ins=mi)):5.1f}%   fork(margin 0) {run(lambda s, c: sim_fork(s, c, margin=0.0, max_ins=mi)):5.1f}%")

def sim_global(B, D, pol, margin, keep_victim, C, window=64, k=16.0):
    """All layers together. Each token: look up (hits count from token a.warm), then start up to B uploads over all layers
    (misses first by score, best first); an upload is usable D tokens later and evicts its victim then (keep_victim) or
    at start (not keep_victim: the slot is unusable while in flight)."""
    st = {}
    for il in layers:
        st[il] = dict(cache=set(), last={}, win=collections.Counter(), glob=collections.Counter(), recent=collections.deque(), gmax=1, pend=[], busy=set())
    T = len(per[layers[0]]); hit = tot = 0
    for t in range(T):
        cands = []
        for il in layers:
            L = st[il]; ids = per[il][t]
            for i in [p for p in L['pend'] if p[0] <= t]:
                L['pend'].remove(i); _, e, v = i
                if v is not None: L['cache'].discard(v)
                L['cache'].add(e); L['busy'].discard(e)
            for e in ids:
                if t >= a.warm:
                    tot += 1
                    if e in L['cache'] and (keep_victim or True): hit += 1
            if len(L['recent']) >= window:
                for e in L['recent'].popleft(): L['win'][e] -= 1
            L['recent'].append(ids)
            for e in ids:
                L['win'][e] += 1; L['glob'][e] += 1; L['gmax'] = max(L['gmax'], L['glob'][e]); L['last'][e] = t
            sc = (lambda x, L=L: L['last'].get(x, -1)) if pol == 'lru' else (lambda x, L=L: L['win'][x] + k*L['glob'][x]/L['gmax'])
            for e in ids:
                if e not in L['cache'] and e not in L['busy']:
                    cands.append((sc(e) if pol != 'lru' else t, il, e, sc))
        cands.sort(key=lambda c: -c[0])
        started = 0
        for _, il, e, sc in cands:
            if started >= B: break
            L = st[il]; ids = per[il][t]
            inflight = len(L['pend'])
            if len(L['cache']) + (0 if keep_victim else -inflight) < C and len(L['cache']) + inflight < C:
                L['pend'].append((t + D, e, None)); L['busy'].add(e); started += 1; continue
            taken = {p[2] for p in L['pend'] if p[2] is not None}
            v = min((x for x in L['cache'] if x not in ids and x not in taken), key=sc, default=None)
            if v is None or (pol != 'lru' and sc(e) <= sc(v) + margin): continue
            if not keep_victim: L['cache'].discard(v)
            L['pend'].append((t + D, e, v if keep_victim else None)); L['busy'].add(e); started += 1
    return 100.0*hit/tot

if a.glob:
    f = float(a.frac.split(",")[0]); C = max(1, int(f*n_exp))
    print(f"global upload budget model at {int(100*f)}% cache ({C} slots/layer), warm {a.warm}; B = uploads started per token over all layers, D = tokens until usable")
    print(f"  belady {sum(sim_belady(per[il], C)[0] for il in layers)*100.0/sum(sim_belady(per[il], C)[1] for il in layers):5.1f}%")
    for pol, mg in (("lru", 0.0), ("fork", 2.0), ("fork", 0.0)):
        for B in (4, 8, 16, 40, 400):
            print(f"  {pol:4s} margin {mg:.0f} B {B:3d}: " + "  ".join(f"D{D} {sim_global(B, D, pol, mg, kv, C):5.1f}%/{sim_global(B, D, pol, mg, not kv, C):5.1f}%" for D in (1, 2, 4) for kv in (False,)))
