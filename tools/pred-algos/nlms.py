#!/usr/bin/env python3
"""Algo 1: linear transform per (layer L, lookahead k) trained online by NLMS (delta rule).

W starts as layer L+k's router and predicts its logits from layer L's MoE input:
    W += mu / |x|^2 * (real_logits - W x) x^T
hits[L+k, t] = top-k overlap. --every N trains every N tokens. Also reports the untrained router.
"""
import time

import numpy as np
import torch

from common import cli, load, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--save", default=None, help="save per-token hits {name: [NL, T]} to this .npz (reference for other algos)")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X, Y, W = d["X"].to(dev).float(), d["Y"].to(dev), d["W"].to(dev)
    real, K = d["real"].to(dev), d["k"]
    NL, T, _ = X.shape

    def hit(pred_logits, tl):  # [n, T, E], target layers [n] -> [n, T]
        p = torch.topk(pred_logits, K, dim=-1).indices
        return (p.unsqueeze(-1) == real[tl].unsqueeze(-2)).any(-1).float().mean(-1)

    saved = {}
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        t0 = time.time()
        base = torch.einsum("led,ltd->lte", W[tl], X[src])
        M, xs = W[tl].clone(), X[src]
        pred = torch.empty_like(base)
        for t in range(T):
            xt = xs[:, t]
            p = torch.einsum("led,ld->le", M, xt)
            pred[:, t] = p
            if a.every > 0 and t % a.every == 0:
                M += ((Y[tl, t] - p) * (a.mu / ((xt * xt).sum(-1, keepdim=True) + 1e-6))).unsqueeze(-1) * xt.unsqueeze(1)
        for name, lg in (("router", base), (f"nlms mu {a.mu} every {a.every}", pred)):
            h = np.full((NL, T), np.nan)
            h[k:] = hit(lg, tl).cpu().numpy()
            saved[f"{'router' if name == 'router' else 'nlms'}_k{k}"] = h
            report(f"L+{k} {name}", h, d["bounds"], f"  ({time.time() - t0:.1f} s)" if name != "router" else "", d["gen"])

    if a.save:
        np.savez(a.save, **saved)


if __name__ == "__main__":
    main()
