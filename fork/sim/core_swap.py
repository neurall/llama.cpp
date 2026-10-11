#!/usr/bin/env python3
"""core_swap.py FILE.routing [more files] [--frac 0.35] [--core 0,0.25,0.5,0.75]: shared experts as a pinned core, the rest swapped fast.

Question: a stored hot list that keeps only the experts SHARED between prompts (resident, never evicted) with the rest of the slots LRU-swapped,
against the whole hot list as a start state and against a cold LRU. Leave one prompt out: the core and the hot list come from the other prompts
(one sample per prompt: a file, or a run inside a file), the held-out prompt is replayed on a per-layer cache of frac x experts slots; the hit rate is
counted over the whole held-out prompt (the hot list is preloaded, so there is no warm-up to skip).
Arms (slots C per layer):
  cold     empty LRU cache
  hotlist  LRU cache preloaded with the C most used experts of the other prompts (what the stored state does today)
  core c   the c*C experts most shared between the other prompts (by number of prompts using them, then by count) are pinned, the other slots LRU, preloaded with the next most used
Shared = used in many of the other prompts; a topic expert is used in few. Reported: mean hit rate over the held-out prompts, and the share of the held-out
prompt's picks that the core alone covers (the part that never needs a swap)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_csv as rc

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--frac", type=float, default=0.35)
ap.add_argument("--core", default="0,0.25,0.5,0.75")
ap.add_argument("--nexp", type=int, default=0)
a = ap.parse_args()

samples = []   # [T, NL, K] per prompt
for f in a.files:
    ids, bounds, layers = rc.read_bin(f)
    for i in range(len(bounds) - 1):
        s = ids[bounds[i]:bounds[i + 1]]
        if len(s) >= 20:
            samples.append(s.astype(np.int32))
NE = a.nexp or int(max(s.max() for s in samples)) + 1
NL = samples[0].shape[1]
C = max(1, round(a.frac * NE))
cores = [float(x) for x in a.core.split(",")]


def counts(ss):
    cnt = np.zeros((NL, NE)); pres = np.zeros((NL, NE))
    for s in ss:
        for l in range(NL):
            c = np.bincount(s[:, l, :].ravel()[s[:, l, :].ravel() >= 0], minlength=NE)
            cnt[l] += c; pres[l] += c > 0
    return cnt, pres


def replay(s, l, pinned, preload):
    """per-layer LRU with some experts pinned; returns hits, lookups"""
    cache = list(preload[: C - len(pinned)])   # least recent first
    inset = set(cache) | set(pinned)
    hits = lookups = 0
    for t in range(len(s)):
        used = [e for e in s[t, l] if e >= 0]
        for e in used:
            lookups += 1
            if e in inset:
                hits += 1
        for e in used:
            if e in pinned:
                continue
            if e in cache:
                cache.remove(e); cache.append(e)
            else:
                if len(cache) >= C - len(pinned):
                    old = cache.pop(0); inset.discard(old)
                cache.append(e); inset.add(e)
    return hits, lookups


res = {("cold", 0): [], ("hotlist", 0): []}
for c in cores:
    if c > 0:
        res[("core", c)] = []
cover = {c: [] for c in cores if c > 0}
for h, held in enumerate(samples):
    rest = [s for i, s in enumerate(samples) if i != h]
    cnt, pres = counts(rest)
    tot = {k: [0, 0] for k in res}; cov_h = {c: 0 for c in cover}; picks = 0
    for l in range(NL):
        hot = list(np.argsort(-cnt[l])[:C][::-1])   # most used last = most recent
        order_core = np.lexsort((-cnt[l], -pres[l]))   # by presence, then count
        for k in res:
            if k[0] == "cold":
                pin, pre = [], []
            elif k[0] == "hotlist":
                pin, pre = [], hot
            else:
                n = round(k[1] * C); pin = [int(e) for e in order_core[:n]]
                pre = [int(e) for e in np.argsort(-cnt[l]) if int(e) not in pin][: C - n][::-1]
            hh, ll = replay(held, l, set(pin), pre)
            tot[k][0] += hh; tot[k][1] += ll
        flat = held[:, l, :].ravel(); flat = flat[flat >= 0]; picks += len(flat)
        for c in cover:
            cov_h[c] += int(np.isin(flat, order_core[: round(c * C)]).sum())
    for k in res:
        res[k].append(tot[k][0] / tot[k][1])
    for c in cover:
        cover[c].append(cov_h[c] / picks)

print(f"{len(samples)} prompts, {NL} layers, {NE} experts, {C} slots per layer ({a.frac:.0%})")
print(f"{'arm':22} {'hit rate':>9} {'min':>7} {'core covers':>12}")
for k, v in res.items():
    name = k[0] if k[0] != "core" else f"core {k[1]:.0%} pinned"
    cv = f"{np.mean(cover[k[1]]):12.1%}" if k[0] == "core" else ""
    print(f"{name:22} {np.mean(v):9.1%} {np.min(v):7.1%} {cv}")
