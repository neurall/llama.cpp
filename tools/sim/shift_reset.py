#!/usr/bin/env python3
"""shift_reset.py FILE.routing [...] [--frac 0.35] [--window 16] [--thr 0.5]: reset the cache when the activations suddenly move.

Replays the prompts of the given per-prompt routing files one after another (a topic switch at every file boundary) on a per-layer expert cache of frac x experts slots
(demand fill: a missed expert is inserted and evicts the lowest score that the token does not use). Policies:
  lru        least recently used
  fork       the engine's score: uses in the last 64 tokens + 16 x lifetime share (lifetime counts start empty)
  fork+soft  fork, and when the activations jump: the lifetime counts and the window are cleared (the cache keeps its experts, they are evicted by the new topic first)
  fork+hard  fork, and when the activations jump: the whole cache is emptied
  fork+keep30  on a jump: all cached experts but the 30% with the highest score are evicted
  fork+decay   on a jump: lifetime counts and window x0.25 (weak experts lose their protection, the hottest still lead)
  ema-sum      score = d*score + activation (1 if the expert was selected for the token), decay d per token (--decay, default 0.95): a slowly fading memory, always-present experts keep a high sum
  ema-mul      score = log of a geometric mean: log m = d*log m + (1-d)*log(activation + 0.05): one absence costs a lot, only experts present in (nearly) every recent token score high
  fork+age     no jump signal: the lifetime counts halve every 256 tokens (plain aging)
  belady     offline best (ceiling)
A jump = the overlap of the experts used in the last W tokens with the experts used in the 4W tokens before them (mean over layers) falls below thr.
Reported per policy: hit rate over everything, over the first 100 tokens after each topic switch (the cost of the switch), and the number of resets fired (a reset
that fires inside a prompt is a false alarm). Data = real routing of the traces in tools/bench/routing/."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_csv as rc

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--frac", type=float, default=0.35)
ap.add_argument("--window", type=int, default=16)
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--nexp", type=int, default=0)
ap.add_argument("--decay", type=float, default=0.95)
a = ap.parse_args()

segs = []
for f in a.files:
    ids, bounds, layers = rc.read_bin(f)
    segs.append(ids[: bounds[1]].astype(np.int32) if len(bounds) > 1 else ids.astype(np.int32))
ids = np.concatenate(segs)
starts = np.cumsum([0] + [len(s) for s in segs])[:-1]
T, NL, K = ids.shape
NE = a.nexp or int(ids.max()) + 1
C = max(1, round(a.frac * NE))
W = a.window


def jump(t):
    """1 - overlap of the last W tokens' experts with those of the 4W tokens before them (mean over layers); None while there is not enough history"""
    if t < 5 * W:
        return None
    ov = 0.0
    for l in range(NL):
        recent = set(ids[t - W:t, l].ravel()) - {-1}
        before = set(ids[t - 5 * W:t - W, l].ravel()) - {-1}
        ov += len(recent & before) / max(1, len(recent))
    return 1 - ov / NL


# the jump signal does not depend on the policy: compute once
J = [jump(t) for t in range(T)]
fire = [False] * T
cool = 0
for t in range(T):
    cool -= 1
    if J[t] is not None and J[t] > 1 - a.thr and cool <= 0:   # overlap below thr
        fire[t] = True; cool = 2 * W


def run(policy):
    hits = np.zeros(T); look = np.zeros(T)
    resets = 0
    for l in range(NL):
        cache = {}   # expert -> last use
        glob = np.zeros(NE); win = np.zeros(NE); recent = []
        esum = np.zeros(NE); emul = np.full(NE, np.log(0.05))
        nxt = None
        if policy == "belady":
            nxt = np.full((T + 1, NE), 10 ** 9, np.int64)
            for t in range(T - 1, -1, -1):
                nxt[t] = nxt[t + 1]
                for e in ids[t, l]:
                    if e >= 0:
                        nxt[t, e] = t
        for t in range(T):
            used = [int(e) for e in ids[t, l] if e >= 0]
            if policy == "fork+age" and t % 256 == 255:
                glob *= 0.5
            if fire[t] and policy in ("fork+soft", "fork+hard", "fork+keep30", "fork+decay"):
                if l == 0:
                    resets += 1
                if policy == "fork+hard":
                    cache.clear()
                if policy == "fork+keep30":
                    gm = max(glob.max(), 1.0)
                    keep = sorted(cache, key=lambda x: win[x] + 16.0 * glob[x] / gm, reverse=True)[: int(0.3 * C)]
                    for x in [x for x in cache if x not in keep]:
                        del cache[x]
                if policy == "fork+decay":
                    glob *= 0.25; win *= 0.25
                else:
                    glob[:] = 0; win[:] = 0; recent.clear()
            if policy in ("ema-sum", "ema-mul"):
                act = np.zeros(NE); act[used] = 1.0
                esum = a.decay * esum + act
                emul = a.decay * emul + (1 - a.decay) * np.log(act + 0.05)
            for e in used:
                look[t] += 1
                if e in cache:
                    hits[t] += 1
            for e in used:
                glob[e] += 1; win[e] += 1
                if e in cache:
                    cache[e] = t
                    continue
                if len(cache) >= C:
                    cand = [x for x in cache if x not in used]
                    if policy == "lru":
                        v = min(cand, key=lambda x: cache[x])
                    elif policy == "ema-sum":
                        v = min(cand, key=lambda x: esum[x])
                    elif policy == "ema-mul":
                        v = min(cand, key=lambda x: emul[x])
                    elif policy == "belady":
                        v = max(cand, key=lambda x: nxt[t + 1][x])
                    else:
                        gm = max(glob.max(), 1.0)
                        v = min(cand, key=lambda x: win[x] + 16.0 * glob[x] / gm)
                    del cache[v]
                cache[e] = t
            recent.append(used)
            if len(recent) > 64:
                for e in recent.pop(0):
                    win[e] -= 1
    return hits, look, resets


print(f"{len(segs)} prompts, {T} tokens, {NL} layers, {NE} experts, {C} slots, window {W}, threshold {a.thr}; jumps fired {sum(fire)} (switches: {len(segs) - 1})")
print(f"{'policy':10} {'hit rate':>9} {'first 100 after a switch':>25} {'resets':>7}")
for pol in ("lru", "fork", "fork+soft", "fork+hard", "fork+keep30", "fork+decay", "fork+age", "ema-sum", "ema-mul", "belady"):
    h, n, r = run(pol)
    first = np.zeros(T, bool)
    for s in starts[1:]:
        first[s:s + 100] = True
    print(f"{pol:10} {h.sum() / n.sum():9.1%} {h[first].sum() / n[first].sum():25.1%} {r:7d}")
