#!/usr/bin/env python3
"""Routing structure from a route_trace.py trace: how long experts live, whether a return after a gap predicts reuse, and how well the experts
active in one layer predict the next layer's set (no hidden states, only the ids).

usage: route_stats.py routes.npz
  A  generic vs context-specific: per layer, the share of activations from experts that are active in every sample (generic) vs in one sample only
     (context-specific), and the burstiness (variance / mean of per-64-token use counts: 1 = steady, >>1 = bursts)
  B  gaps between uses of one expert (tokens): median, share longer than 64 / 256 tokens, and the next gap given the previous one:
     P(next gap <= 32 | previous gap <= 32) against P(next gap <= 32 | previous gap > 128)  -- does a quick return mean "keep it"?
  C  same layer, next token: overlap of the selected sets (how much of the set repeats)
  D  next layer, same token: top-8 overlap of a co-activation predictor (counts of expert pairs seen so far, learned online: the sum of
     counts[s -> e] over the experts s active in this layer), a frequency prior (most used experts of the target layer so far), and both
"""
import sys
import numpy as np

z = np.load(sys.argv[1])
ids, bounds, layers, NE = z['ids'].astype(np.int64), z['bounds'], z['layers'], int(z['n_expert'])
T, NL, K = ids.shape
ns = len(bounds) - 1
print(f"{T} tokens, {NL} MoE layers, top-{K} of {NE}, {ns} samples {[int(bounds[i + 1] - bounds[i]) for i in range(ns)]}")

# per layer expert use matrix [T, NE] (bool)
use = np.zeros((NL, T, NE), bool)
for j in range(NL):
    use[j, np.arange(T)[:, None], ids[:, j]] = True

print("\nA  generic vs context-specific, burstiness")
print(f"{'layer':>5} {'generic%':>9} {'one-sample%':>11} {'burst(64)':>10}")
for j in range(0, NL, 6):
    per = np.stack([use[j, bounds[i]:bounds[i + 1]].any(0) for i in range(ns)])   # [ns, NE] expert seen in the sample
    n_act = use[j].sum(0)                                                           # activations per expert
    gen = n_act[per.all(0)].sum() / n_act.sum()
    one = n_act[per.sum(0) == 1].sum() / n_act.sum()
    blk = use[j, :T // 64 * 64].reshape(-1, 64, NE).sum(1)                         # uses per 64-token block
    fano = np.nanmean(blk.var(0) / np.maximum(blk.mean(0), 1e-9)[None][0]) if blk.shape[0] > 1 else float('nan')
    print(f"{int(layers[j]):5d} {100 * gen:9.1f} {100 * one:11.1f} {fano:10.2f}")

print("\nB  gaps between uses of one expert (tokens), pooled over layers")
gaps_all, prev_next = [], []
for j in range(NL):
    for e in range(NE):
        t = np.nonzero(use[j, :, e])[0]
        if len(t) > 2:
            g = np.diff(t)
            gaps_all.append(g)
            prev_next.append(np.stack([g[:-1], g[1:]], 1))
g = np.concatenate(gaps_all)
pn = np.concatenate(prev_next)
print(f"gaps: {len(g)}, median {np.median(g):.0f}, >64: {100 * (g > 64).mean():.1f}%, >256: {100 * (g > 256).mean():.1f}%")
q, l = pn[pn[:, 0] <= 32], pn[pn[:, 0] > 128]
print(f"P(next gap <= 32 | previous gap <= 32)  = {(q[:, 1] <= 32).mean():.3f}   (n={len(q)})")
print(f"P(next gap <= 32 | previous gap  > 128) = {(l[:, 1] <= 32).mean():.3f}   (n={len(l)})")
print(f"P(next gap  > 128 | previous gap  > 128) = {(l[:, 1] > 128).mean():.3f}   -- an expert gone quiet for >128 tokens stays away")

print("\nC  same layer, next token: share of the 8 selected experts that repeat")
rep = [np.mean([len(set(ids[t, j]) & set(ids[t + 1, j])) / K for t in range(T - 1) if True]) for j in range(0, NL, 6)]
print("layers", [int(layers[j]) for j in range(0, NL, 6)], "->", " ".join(f"{r:.3f}" for r in rep))

print("\nD  next layer, same token: top-8 overlap of simple predictors learned online (no hidden state)")
def overlap(pred, real):
    return len(set(pred) & set(real)) / K
res = {'cooc': [], 'freq': [], 'both': []}
for j in range(0, NL - 1, 5):
    C = np.zeros((NE, NE), np.float32)   # C[s, e]: times e was selected in layer j+1 when s was selected in layer j
    F = np.zeros(NE, np.float32)
    ov = {'cooc': [], 'freq': [], 'both': []}
    for t in range(T):
        S, R = ids[t, j], ids[t, j + 1]
        if t > 200:  # score after a warm-up
            sc = C[S].sum(0)
            ov['cooc'].append(overlap(np.argsort(-sc)[:K], R))
            ov['freq'].append(overlap(np.argsort(-F)[:K], R))
            comb = sc / (sc.max() + 1e-9) + F / (F.max() + 1e-9)
            ov['both'].append(overlap(np.argsort(-comb)[:K], R))
        C[np.repeat(S, K), np.tile(R, K)] += 1
        F[R] += 1
    for k in res:
        res[k].append(np.mean(ov[k]))
    print(f"layer {int(layers[j]):2d} -> {int(layers[j + 1]):2d}: co-activation {res['cooc'][-1]:.3f}  frequency prior {res['freq'][-1]:.3f}  both {res['both'][-1]:.3f}")
print("mean:", {k: round(float(np.mean(v)), 3) for k, v in res.items()}, " (reference: the learned hidden-state predictors reach ~0.6-0.7 on GLM)")
