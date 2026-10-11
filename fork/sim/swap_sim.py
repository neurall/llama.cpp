#!/usr/bin/env python3
"""swap_sim.py FILE.routing [...] [--frac 0.35] [--budget 4,16,1000] [--decay 0.5] [--delay 1]: eviction scores under a swap budget, replayed on real routing.

The model of the engine you describe: per layer a cache of frac x experts slots. Token t runs against the cache as it is (met = selected and cached, unmet = selected and not
cached: computed on the CPU). After the LAST layer of token t every layer has its scores: each unmet expert is a candidate, its victim is the lowest-scored cached expert of its
layer (candidates and victims are paired best with worst per layer; empty slots first). All pairs of all layers are sorted by gain = score(candidate) - score(victim) and the best
`budget` swaps (swaps per token over all layers, evictions only: filling an empty slot is free) are issued; they land `delay` tokens later (the upload time), so token t+delay
is the first that can hit them. Swaps with gain <= 0 are never made. Scores:
  lru      time of the last use
  fork     the engine's score: uses in the last 64 tokens + 16 x lifetime share
  ema      decay x score + 1 if selected (the indicator version of your relevance idea; no router probabilities needed)
  rel      decay x score + router probability of the expert for the token (needs a version 2 routing file: the router probabilities of ALL experts)
  relmax   like rel but score = max(decay x score, probability): the strongest recent relevance, no summing
  belady   offline ceiling: the expert used soonest wins (no budget limit on its own, same budget here)
--pool global lets the layers share the slots (scores are compared across layers as they are). Several files are replayed one after another (topic switches at the file boundaries). Hit rate over everything and over the first 100 tokens after each boundary."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_csv as rc

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--frac", type=float, default=0.35)
ap.add_argument("--budget", default="4,16,1000", help="swaps per token over all layers, comma separated")
ap.add_argument("--decay", type=float, default=0.5)
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--policies", default="lru,fork,ema,rel,relmax,belady")
ap.add_argument("--layers", type=int, default=0, help="only every Nth layer (speed)")
ap.add_argument("--slots", type=int, default=0, help="slots per layer (the real number of a run, e.g. 57..64 on MiMo with 2 x 24 GB); default frac x experts")
ap.add_argument("--pool", choices=["layer", "global"], default="layer", help="layer: every layer evicts only its own experts (the engine); global: ONE pool of slots x layers, the globally lowest-scored cached expert of any layer is the victim and the best candidate of any layer comes in")
a = ap.parse_args()

segs, prs = [], []
for f in a.files:
    ids, bounds, layers, pr = rc.read_bin(f, probs=True)
    segs.append(ids[: bounds[1]].astype(np.int32))
    prs.append(pr[: bounds[1]].astype(np.float32) if pr is not None else None)
ids = np.concatenate(segs)
probs = np.concatenate(prs) if all(p is not None for p in prs) else None
starts = np.cumsum([0] + [len(s) for s in segs])[:-1]
T, NL, K = ids.shape
if a.layers:
    sel = list(range(0, NL, a.layers)); ids = ids[:, sel]; NL = len(sel)
    if probs is not None: probs = probs[:, sel]
NE = int(ids.max()) + 1 if probs is None else probs.shape[2]
C = a.slots or max(1, round(a.frac * NE))
first = np.zeros(T, bool)
for s in starts[1:]:
    first[s:s + 100] = True


def run(policy, budget):
    if policy in ("rel", "relmax") and probs is None:
        return None
    cache = [set() for _ in range(NL)]
    score = np.zeros((NL, NE))
    glob = np.zeros((NL, NE)); win = np.zeros((NL, NE)); last = np.zeros((NL, NE))
    recent = [[] for _ in range(NL)]
    pend = []   # (land_t, layer, victim, cand)
    hits = np.zeros(T); look = np.zeros(T)
    nxt = None
    if policy == "belady":
        nxt = np.full((T + 2, NL, NE), 10 ** 9, np.int64)
        for t in range(T - 1, -1, -1):
            nxt[t] = nxt[t + 1]
            for l in range(NL):
                for e in ids[t, l]:
                    if e >= 0: nxt[t, l, e] = t
    for t in range(T):
        for p in [p for p in pend if p[0] <= t]:
            _, l, v, c = p
            if isinstance(v, tuple): cache[v[0]].discard(v[1])
            elif v >= 0: cache[l].discard(v)
            cache[l].add(c)
        pend = [p for p in pend if p[0] > t]
        unmet = []
        for l in range(NL):
            used = [int(e) for e in ids[t, l] if e >= 0]
            m = [e for e in used if e not in cache[l]]
            hits[t] += len(used) - len(m); look[t] += len(used)
            unmet.append(m)
            # scores after the token
            if policy == "lru":
                last[l, used] = t + 1
            elif policy == "fork":
                glob[l, used] += 1; win[l, used] += 1; recent[l].append(used)
                if len(recent[l]) > 64:
                    for e in recent[l].pop(0): win[l, e] -= 1
            elif policy == "ema":
                score[l] *= a.decay; score[l, used] += 1
            elif policy == "rel":
                score[l] = a.decay * score[l] + probs[t, l]
            elif policy == "relmax":
                score[l] = np.maximum(a.decay * score[l], probs[t, l])
        # swap decision after the last layer: every layer has its scores
        pairs = []
        inflight = {(p[1], p[3]) for p in pend}
        scs = []
        for l in range(NL):
            if policy == "lru": sc = last[l]
            elif policy == "fork": sc = win[l] + 16.0 * glob[l] / max(glob[l].max(), 1.0)
            elif policy == "belady": sc = -nxt[t + 1, l].astype(np.float64)
            else: sc = score[l]
            scs.append(sc)
        if a.pool == "layer":
            for l in range(NL):
                sc = scs[l]
                cs = sorted({e for e in unmet[l] if (l, e) not in inflight}, key=lambda e: -sc[e])
                incoming = {p[3] for p in pend if p[1] == l}
                vs = sorted((e for e in cache[l] if e not in set(int(x) for x in ids[t, l])), key=lambda e: sc[e])
                free = C - len(cache[l]) - len(incoming)
                for c in cs:
                    if free > 0:
                        pairs.append((1e30, l, -1, c)); free -= 1
                    elif vs:
                        v = vs.pop(0); pairs.append((sc[c] - sc[v], l, v, c))
        else:   # one pool of C x NL slots shared by all layers
            cs = sorted(((scs[l][e], l, e) for l in range(NL) for e in set(unmet[l]) if (l, e) not in inflight), reverse=True)
            vs = sorted((scs[l][e], l, e) for l in range(NL) for e in cache[l] if e not in set(int(x) for x in ids[t, l]))
            free = C * NL - sum(len(c) for c in cache) - len(pend)
            vi = 0
            for sc_c, l, c in cs:
                if free > 0:
                    pairs.append((1e30, l, -1, c)); free -= 1
                elif vi < len(vs):
                    sc_v, lv, v = vs[vi]; vi += 1
                    pairs.append((sc_c - sc_v, l, (lv, v), c))
        pairs.sort(key=lambda x: -x[0])
        b = budget
        for g, l, v, c in pairs:
            if isinstance(v, tuple) or v >= 0:
                if g <= 0 or b <= 0: continue
                b -= 1
            pend.append((t + a.delay, l, v, c))
    return hits, look


print(f"{len(segs)} prompts, {T} tokens, {NL} layers, {NE} experts, {C} slots per layer ({a.pool} pool), delay {a.delay}, decay {a.decay}, router probabilities: {'yes' if probs is not None else 'no'}")
print(f"{'policy':8} " + " ".join(f"{'budget ' + b:>16}" for b in a.budget.split(",")) + "     (hit rate all / first 100 after a switch)")
for pol in a.policies.split(","):
    cells = []
    for b in a.budget.split(","):
        r = run(pol, int(b))
        cells.append("n/a" if r is None else f"{r[0].sum() / r[1].sum():6.1%} /{r[0][first].sum() / max(1, r[1][first].sum()):6.1%}")
    print(f"{pol:8} " + " ".join(f"{c:>16}" for c in cells))
