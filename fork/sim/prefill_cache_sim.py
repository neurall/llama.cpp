#!/usr/bin/env python3
"""prefill_cache_sim.py: bytes streamed per prompt for policies that keep experts in the VRAM left over after the graph and the KV, per pass size.
Trace: Qwen3.6 routing captures (/p/bw/data/acts/*.npz, ids[token, layer, k]). A pass of P tokens needs, per layer, the union U of its tokens' experts;
the experts of U not on the card are uploaded (a needed expert is uploaded anyway: keeping it costs nothing, so a policy only chooses what to RETAIN
from cache + U after the layer's pass). C slots per layer = frac * n_experts. Policies (retain top-C of cache + U by score):
  none      no cache: every expert of U is uploaded
  lru       score = last pass the expert was used (what decode's LRU does)
  window    score = hits in the last W tokens (your 'local hottest')
  local     score = hits since the start of the prompt
  global    frozen: the C hottest of OTHER prompts (leave-one-out), never changes
  glob+loc  start from global, then retain by local count (your plan: global before the first token, local hottest from then on)
  oracle    frozen: the C hottest of this very prompt (upper bound for any frozen set)"""
import numpy as np, glob, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--frac", default="0.1,0.25"); ap.add_argument("--pass", dest="P", default="64,128,256,512"); ap.add_argument("--window", type=int, default=256)
a = ap.parse_args()
fs = sorted(glob.glob("/p/bw/data/acts/q36-*.npz"))
T = {f.split("-")[-1][:-4]: np.load(f)["ids"].astype(np.int64) for f in fs}
NE = 256
L = next(iter(T.values())).shape[1]
def heat(ids):   # [layer, expert] counts
    h = np.zeros((ids.shape[1], NE), np.int64)
    for l in range(ids.shape[1]): h[l] = np.bincount(ids[:, l, :].ravel(), minlength=NE)
    return h
def run(name, ids, P, C, glob_h, W):
    n = ids.shape[0]; up = 0; asked = 0
    for l in range(ids.shape[1]):
        cache = np.zeros(NE, bool); last = np.full(NE, -1.0); loc = np.zeros(NE); win = []   # win: per-token id arrays
        if name in ("global", "glob+loc"): cache[np.argsort(-glob_h[l])[:C]] = True
        if name == "oracle": cache[np.argsort(-np.bincount(ids[:, l, :].ravel(), minlength=NE))[:C]] = True
        for pi, s in enumerate(range(0, n, P)):
            blk = ids[s:s + P, l, :]
            U = np.zeros(NE, bool); U[np.unique(blk)] = True
            asked += U.sum(); up += (U & ~cache).sum()
            cnt = np.bincount(blk.ravel(), minlength=NE); loc += cnt; last[U] = pi
            win.append(cnt)
            if name in ("none", "global", "oracle"):
                if name == "none": cache[:] = False
                continue
            if sum(1 for _ in win) * P > W: win.pop(0)
            sc = {"lru": last, "window": np.sum(win, 0), "local": loc, "glob+loc": loc}[name].astype(float)
            cand = cache | U
            sc = np.where(cand, sc + (1e-3 * glob_h[l] / max(1, glob_h[l].max()) if name == "glob+loc" else 0), -1)
            keep = np.argsort(-sc)[:C]; cache[:] = False; cache[keep[sc[keep] >= 0]] = True
    return up, asked
print(f"{'prompt':8} {'tokens':>6} {'frac':>5} {'pass':>5} | " + " ".join(f"{p:>9}" for p in ("none", "lru", "window", "local", "global", "glob+loc", "oracle")) + "   (experts uploaded per layer-pass, % of none)")
for key, ids in T.items():
    others = np.concatenate([v for k, v in T.items() if k != key]); gh = heat(others)
    for fr in map(float, a.frac.split(",")):
        C = int(fr * NE)
        for P in map(int, a.P.split(",")):
            r = {p: run(p, ids, P, C, gh, a.window) for p in ("none", "lru", "window", "local", "global", "glob+loc", "oracle")}
            base = r["none"][0]
            print(f"{key:8} {ids.shape[0]:6} {fr:5.2f} {P:5} | " + " ".join(f"{100 * r[p][0] / base:8.1f}%" for p in r))
