#!/usr/bin/env python3
"""Learn which experts a layer will select from routing history alone (no hidden state), and which signals matter.

usage: route_learn.py routes.npz [test_sample=3]
For every (token t, layer j, expert e) the features are computed online, the way a running engine could (counts start at zero at each sample):
  same layer, earlier tokens   age since last use, previous gap, use rates (EMA 8 / 32 / 128 tokens), used at t-1, co-use score from the set of t-1 at layer j
  cross-layer, this token      co-use scores from the sets selected at layers j-1 and j-2 for token t (known before layer j runs)
  cross-token, other layers    co-use scores from the set selected at t-1 at layer j+1 and at layer j-1
  layer index, total uses
A co-use score is sum over s in the source set of count[s -> e] / (tokens seen * 8): how often e was selected in layer j when s was selected there.
Target: e is among the 8 selected at (t, j). Train on all samples but one (gradient-boosted trees), test on the held-out sample (a different prompt),
report top-8 overlap with the real set for: the full model, each feature family removed, and simple baselines.
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

z = np.load(sys.argv[1])
ids, bounds = z['ids'].astype(np.int64), z['bounds']
T, NL, K = ids.shape
NE = int(z['n_expert'])
test = int(sys.argv[2]) if len(sys.argv) > 2 else len(bounds) - 2
ns = len(bounds) - 1

FEAT = ['age', 'prev_gap', 'ema8', 'ema32', 'ema128', 'used_t1', 'co_same_t1', 'co_L1', 'co_L2', 'co_up_t1', 'co_dn_t1', 'layer', 'total']
FAM = {'same-layer history': [0, 1, 2, 3, 4, 5, 6, 12], 'cross-layer (this token)': [7, 8], 'cross-token other layers': [9, 10]}
LAYERS = list(range(0, NL, 3))      # layers used for training/testing rows (every 3rd)
STEP = 3                            # token subsampling for rows

def sample_rows(s):
    """sequential pass over one sample; returns X [rows, F], y [rows], and (token, layer) of each group of NE rows"""
    a, b = bounds[s], bounds[s + 1]
    sel = ids[a:b]                                              # [n, NL, K]
    n = b - a
    last = np.full((NL, NE), -1, np.int64); gap = np.full((NL, NE), -1, np.int64)
    ema = np.zeros((3, NL, NE), np.float32); total = np.zeros((NL, NE), np.float32)
    C = {k: np.zeros((NL, NE, NE), np.float32) for k in ('same', 'L1', 'L2', 'up', 'dn')}
    al = np.array([2 / 9, 2 / 33, 2 / 129], np.float32)
    X, Y = [], []
    for t in range(n):
        for j in range(NL):
            S = sel[t, j]
            if t % STEP == 0 and j in LAYERS and t > 8:
                f = np.zeros((NE, len(FEAT)), np.float32)
                f[:, 0] = np.log1p(np.where(last[j] < 0, t + 1, t - last[j]))
                f[:, 1] = np.log1p(np.maximum(gap[j], 0))
                f[:, 2], f[:, 3], f[:, 4] = ema[0, j], ema[1, j], ema[2, j]
                norm = 1.0 / (max(t, 1) * K)
                if t > 0:
                    f[sel[t - 1, j], 5] = 1
                    f[:, 6] = C['same'][j][sel[t - 1, j]].sum(0) * norm
                    if j + 1 < NL:
                        f[:, 9] = C['up'][j][sel[t - 1, j + 1]].sum(0) * norm
                    if j > 0:
                        f[:, 10] = C['dn'][j][sel[t - 1, j - 1]].sum(0) * norm
                if j >= 1:
                    f[:, 7] = C['L1'][j][sel[t, j - 1]].sum(0) * norm
                if j >= 2:
                    f[:, 8] = C['L2'][j][sel[t, j - 2]].sum(0) * norm
                f[:, 11] = j / NL
                f[:, 12] = np.log1p(total[j])
                y = np.zeros(NE, np.int8); y[S] = 1
                X.append(f); Y.append(y)
            # update state with what was really selected
            u = np.zeros(NE, np.float32); u[S] = 1
            used = u > 0
            gap[j][used & (last[j] >= 0)] = (t - last[j])[used & (last[j] >= 0)]
            last[j][used] = t
            for q in range(3):
                ema[q, j] += al[q] * (u - ema[q, j])
            total[j] += u
            if t > 0:
                C['same'][j][np.repeat(sel[t - 1, j], K), np.tile(S, K)] += 1
                if j + 1 < NL: C['up'][j][np.repeat(sel[t - 1, j + 1], K), np.tile(S, K)] += 1
                if j > 0: C['dn'][j][np.repeat(sel[t - 1, j - 1], K), np.tile(S, K)] += 1
            if j >= 1: C['L1'][j][np.repeat(sel[t, j - 1], K), np.tile(S, K)] += 1
            if j >= 2: C['L2'][j][np.repeat(sel[t, j - 2], K), np.tile(S, K)] += 1
    return np.concatenate(X), np.concatenate(Y)

data = []
for s in range(ns):
    X, Y = sample_rows(s)
    data.append((X, Y))
    print(f"sample {s}: {len(Y) // NE} (token, layer) groups", flush=True)

def top8_overlap(score, y):
    sc, yy = score.reshape(-1, NE), y.reshape(-1, NE)
    top = np.argsort(-sc, axis=1)[:, :K]
    return float(np.take_along_axis(yy, top, axis=1).sum(1).mean() / K)

Xtr = np.concatenate([d[0] for i, d in enumerate(data) if i != test]); Ytr = np.concatenate([d[1] for i, d in enumerate(data) if i != test])
Xte, Yte = data[test]
print(f"\ntrain {len(Ytr) // NE} groups (samples except {test}), test {len(Yte) // NE} groups (sample {test}, a different prompt)\n")

def fit_eval(cols, label):
    sub = np.random.default_rng(0).choice(len(Ytr), size=min(len(Ytr), 600000), replace=False)
    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    m.fit(Xtr[sub][:, cols], Ytr[sub])
    ov = top8_overlap(m.predict_proba(Xte[:, cols])[:, 1], Yte)
    print(f"{label:46s} top-8 overlap {ov:.3f}", flush=True)
    return ov

print("baselines (no learning), top-8 overlap with the real set:")
for name, col in (('frequency prior (EMA 128)', 4), ('recent use rate (EMA 8)', 2), ('same set as the previous token', 5),
                  ('co-use from layer j-1 (this token)', 7), ('co-use from t-1 same layer', 6), ('co-use from t-1 layer above', 9)):
    print(f"  {name:44s} {top8_overlap(Xte[:, col], Yte):.3f}")
print("\nlearned (gradient-boosted trees on all features, then one family removed / one family alone):")
allc = list(range(len(FEAT)))
full = fit_eval(allc, 'all features')
for fam, cols in FAM.items():
    fit_eval([c for c in allc if c not in cols], f'without {fam}')
for fam, cols in FAM.items():
    fit_eval(cols, f'only {fam}')
