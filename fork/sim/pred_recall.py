#!/usr/bin/env python3
"""What the predictor is worth to a cache: recall of the experts that are NOT cached, by lookahead and number of candidates.

usage: pred_recall.py data.pt [slots=97] [tokens_skipped=200] [layers_step=3]   (data.pt: pred_lab.py build)
For each lookahead k (the prediction is made from layer L's MoE input, for layer L+k) the online-learned predictor of pred_lab.py (NLMS, starting from the
target layer's router) ranks the experts of the target layer. Next to it each target layer has a demand-filled LRU cache of `slots` experts fed with the real
selections. For every token the engine would upload the top-M predicted experts that are not cached (M candidates), so the numbers that matter are:
  miss/tok     real experts of the token that are not cached
  recall(M)    share of those misses among the M top-ranked uncached experts (they could have arrived before the layer runs)
  useful(M)    share of the M uploaded candidates that were real misses (the rest is wasted link bandwidth)
"""
import os, sys
import torch

torch.set_num_threads(1)
d = torch.load(sys.argv[1])
slots = int(sys.argv[2]) if len(sys.argv) > 2 else 97
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 200
step = int(sys.argv[4]) if len(sys.argv) > 4 else 3
X, Y, W, B, K, bounds = d["X"].float(), d["Y"], d["W"], d["B"], d["k"], d["bounds"]
NL, T, D = X.shape
E = W.shape[1]
MU = 0.5
sc = lambda lg, tl: torch.sigmoid(lg) + B[tl] if B is not None else lg
real = torch.stack([torch.topk(sc(Y[l], l), K, dim=-1).indices for l in range(NL)])   # [NL, T, K]
starts = {int(b) for b in bounds[:-1]}
print(f"{T} tokens, {NL} layers, {slots} cache slots of {E}, every {step}th target layer, first {skip} tokens of each sample not counted\n")
print(f"{'lookahead':>9} {'miss/tok':>8}   " + "   ".join(f"M={m}: recall useful" for m in (1, 2, 3, 4, 8)))
for k in (1, 2, 3, 4):
    tl = torch.arange(k, NL, step)
    src = tl - k
    Mx = W[tl].clone()
    xs = X[src]
    cache = torch.zeros(len(tl), E, dtype=torch.bool)
    age = torch.zeros(len(tl), E)
    miss_n = 0; tok_n = 0; ev = 0
    hit = {m: 0.0 for m in (1, 2, 3, 4, 8)}; use = {m: 0.0 for m in (1, 2, 3, 4, 8)}
    rank_ok = [0] * 8
    for t in range(T):
        if t in starts:
            cache.zero_(); age.zero_()                       # a new prompt starts with an empty cache
        sample_t = max(b for b in starts if b <= t)
        xt = xs[:, t]
        p = torch.einsum("led,ld->le", Mx, xt)
        score = sc(p, tl)
        r = real[tl, t]                                     # [n, K] real selection of the target layers
        for i in range(len(tl)):
            rs = r[i]
            unc = rs[~cache[i][rs]]
            if t - sample_t >= skip and len(unc):
                s = score[i].masked_fill(cache[i], -1e9)    # candidates: uncached experts only
                order = torch.topk(s, 16).indices
                for q, e in enumerate(order[:8].tolist()):
                    rank_ok[q] += e in set(unc.tolist())
                mset = set(unc.tolist())
                tok_n += 1; miss_n += len(unc); ev += 1
                for m in hit:
                    top = order[:m].tolist()
                    got = sum(1 for e in top if e in mset)
                    hit[m] += got / len(mset); use[m] += got / m
            elif t - sample_t >= skip:
                tok_n += 1
            # demand fill, LRU
            age[i] += 1
            age[i][rs] = 0
            for e in rs.tolist():
                if not cache[i][e]:
                    if int(cache[i].sum()) >= slots:
                        cand = torch.where(cache[i], age[i], torch.tensor(-1.0))
                        cand[rs] = -1.0
                        cache[i][int(torch.argmax(cand))] = False
                    cache[i][e] = True
        g = (Y[tl, t] - p) * (MU / ((xt * xt).sum(-1, keepdim=True) + 1e-6))
        Mx += g.unsqueeze(-1) * xt.unsqueeze(1)
    print(f"          precision by rank (chance that the candidate of rank r is a real miss): " + " ".join(f"r{q + 1} {100 * rank_ok[q] / max(ev, 1):.0f}%" for q in range(8)))
    print(f"{k:9d} {miss_n / max(tok_n, 1):8.2f}   " + "   ".join(f"{100 * hit[m] / max(ev, 1):13.1f}% {100 * use[m] / max(ev, 1):5.1f}%" for m in hit), flush=True)
