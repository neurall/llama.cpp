#!/usr/bin/env python3
"""A learned eviction score against LRU, aging LFU, the fork's policy and Belady, on routing traces (no hidden state).

usage: route_evict.py routes.npz [test_sample=3] [W=16]
Per layer a cache of C slots (34% / 25% / 50% of the experts) with demand fill: every missed expert is inserted (unbounded inserts, as in
cache_sim.py --ins) and evicts the cached expert with the lowest score that the current token does not use. Hits are counted after 200 warm-up tokens.
Learned score: P(the expert is used again within the next W tokens at this layer), gradient-boosted trees on features known when the eviction happens:
age, previous gap, use rates (EMA 8 / 32 / 128), total uses, used at t-1, and the co-use score from the sets of layers j-1 and j-2 for this token.
Trained on every sample except the test sample (a different prompt); counts and rates start at zero at each sample's start (online, as in a session).
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

z = np.load(sys.argv[1])
ids, bounds = z['ids'].astype(np.int64), z['bounds']
T, NL, K = ids.shape
NE = int(z['n_expert'])
test = int(sys.argv[2]) if len(sys.argv) > 2 else len(bounds) - 2
W = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ns = len(bounds) - 1
LAYERS = list(range(0, NL, 3))
WARM = 200
FEAT = ['age', 'prev_gap', 'ema8', 'ema32', 'ema128', 'used_t1', 'total', 'co_L1', 'co_L2', 'layer']

class State:
    def __init__(self, sel):
        self.sel = sel
        self.last = np.full((NL, NE), -1, np.int64); self.gap = np.full((NL, NE), -1, np.int64)
        self.ema = np.zeros((3, NL, NE), np.float32); self.total = np.zeros((NL, NE), np.float32)
        self.L1 = np.zeros((NL, NE, NE), np.float32); self.L2 = np.zeros((NL, NE, NE), np.float32)
        self.al = np.array([2 / 9, 2 / 33, 2 / 129], np.float32)
        self.win = np.zeros((NL, NE), np.int32)      # uses in the last 64 tokens (the fork's window)
        self.glob = np.zeros((NL, NE), np.float32)   # lifetime uses
    def feats(self, t, j, cross=True):
        f = np.zeros((NE, len(FEAT)), np.float32)
        f[:, 0] = np.log1p(np.where(self.last[j] < 0, t + 1, t - self.last[j]))
        f[:, 1] = np.log1p(np.maximum(self.gap[j], 0))
        f[:, 2], f[:, 3], f[:, 4] = self.ema[0, j], self.ema[1, j], self.ema[2, j]
        if t > 0:
            f[self.sel[t - 1, j], 5] = 1
        f[:, 6] = np.log1p(self.total[j])
        norm = 1.0 / (max(t, 1) * K)
        if cross and j >= 1:
            f[:, 7] = self.L1[j][self.sel[t, j - 1]].sum(0) * norm
        if cross and j >= 2:
            f[:, 8] = self.L2[j][self.sel[t, j - 2]].sum(0) * norm
        f[:, 9] = j / NL
        return f
    def update(self, t, j):
        S = self.sel[t, j]
        u = np.zeros(NE, np.float32); u[S] = 1
        used = u > 0
        m = used & (self.last[j] >= 0)
        self.gap[j][m] = (t - self.last[j])[m]
        self.last[j][used] = t
        for q in range(3):
            self.ema[q, j] += self.al[q] * (u - self.ema[q, j])
        self.total[j] += u
        self.glob[j] += u
        self.win[j][S] += 1
        if t >= 64:
            self.win[j][self.sel[t - 64, j]] -= 1
        if j >= 1: self.L1[j][np.repeat(self.sel[t, j - 1], K), np.tile(S, K)] += 1
        if j >= 2: self.L2[j][np.repeat(self.sel[t, j - 2], K), np.tile(S, K)] += 1

def future_use(sel, j, n):
    """y[t, e] = e is used at layer j in the next W tokens (t+1 .. t+W)"""
    u = np.zeros((n + W + 1, NE), np.int8)
    for t in range(n):
        u[t, sel[t, j]] = 1
    c = np.concatenate([np.zeros((1, NE), np.int32), np.cumsum(u, 0)])
    return np.stack([(c[min(t + W + 1, n + W + 1)] - c[t + 1]) > 0 for t in range(n)])

def train_rows(s, step=3):
    a, b = bounds[s], bounds[s + 1]
    sel = ids[a:b]; n = b - a
    st = State(sel)
    fu = {j: future_use(sel, j, n) for j in LAYERS}
    X, Y = [], []
    for t in range(n):
        for j in range(NL):
            if t % step == 0 and j in LAYERS and t > 8 and t < n - W:
                X.append(st.feats(t, j)); Y.append(fu[j][t])
            st.update(t, j)
    return np.concatenate(X), np.concatenate(Y).reshape(-1).astype(np.int8)

print(f"training the scorer on samples {[i for i in range(ns) if i != test]} (W={W}) ...", flush=True)
parts = [train_rows(s) for s in range(ns) if s != test]
Xtr = np.concatenate([p[0] for p in parts]); Ytr = np.concatenate([p[1] for p in parts])
rng = np.random.default_rng(0)
sub = rng.choice(len(Ytr), size=min(len(Ytr), 600000), replace=False)
models = {}
for name, cols in (('learned', list(range(len(FEAT)))), ('learned no cross-layer', [c for c in range(len(FEAT)) if c not in (7, 8)])):
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    m.fit(Xtr[sub][:, cols], Ytr[sub])
    models[name] = (m, cols)
print(f"base rate of 'used in the next {W} tokens': {Ytr.mean():.3f}", flush=True)

a, b = bounds[test], bounds[test + 1]
sel = ids[a:b]; n = b - a

def simulate(policy, C):
    st = State(sel)
    hit = tot = 0
    cache = {j: np.zeros(NE, bool) for j in LAYERS}
    nxt = None
    if policy == 'belady':
        nxt = {}
        for j in LAYERS:
            nu = np.full((n + 1, NE), 10 ** 9, np.int64)       # next use of e at or after token t
            for t in range(n - 1, -1, -1):
                nu[t] = nu[t + 1]; nu[t, sel[t, j]] = t
            nxt[j] = nu
    for t in range(n):
        for j in range(NL):
            if j in LAYERS:
                S = sel[t, j]
                cached = cache[j]
                miss = [e for e in S if not cached[e]]
                if t >= WARM:
                    hit += K - len(miss); tot += K
                if miss:
                    free = C - int(cached.sum())
                    need = len(miss) - max(free, 0)
                    protect = np.zeros(NE, bool); protect[S] = True
                    if need > 0:
                        if policy == 'lru':
                            sc = st.last[j].astype(np.float64)
                        elif policy == 'lfua':
                            sc = st.lfu[j] if hasattr(st, 'lfu') else st.total[j]
                        elif policy == 'ours':
                            sc = st.win[j] + 16.0 * st.glob[j] / max(st.glob[j].max(), 1.0)
                        elif policy == 'belady':
                            sc = -nxt[j][t + 1].astype(np.float64) if t + 1 <= n else np.zeros(NE)
                        else:
                            mdl, cols = models[policy]
                            sc = mdl.predict_proba(st.feats(t, j)[:, cols])[:, 1]
                        sc = np.where(cached & ~protect, sc, np.inf)
                        for e in np.argsort(sc)[:need]:
                            if np.isinf(sc[e]): break
                            cached[e] = False
                    for e in miss:
                        if cached.sum() < C:
                            cached[e] = True
            st.update(t, j)
    return hit / tot

print(f"\nhit rate on the held-out prompt (sample {test}, {n} tokens, layers {LAYERS[0]}..{LAYERS[-1]} step 3, {WARM} warm-up tokens)")
print(f"{'slots':>10} " + " ".join(f"{p:>22}" for p in ('lru', 'ours', 'learned no cross-layer', 'learned', 'belady')))
for C in (int(0.25 * NE), int(0.34 * NE), int(0.5 * NE)):
    r = [simulate(p, C) for p in ('lru', 'ours', 'learned no cross-layer', 'learned', 'belady')]
    print(f"{C:4d} ({100 * C / NE:2.0f}%) " + " ".join(f"{100 * x:21.1f}%" for x in r), flush=True)
