#!/usr/bin/env python3
"""Algo 7: the real router of L+k applied to an extrapolated MoE input.

x_{L+k} ~ x_L + g ⊙ (x_L - x_{L-1}), g a learned diagonal [D] per (L, k). The router logits are linear in g:
    logits = W x_L + (W diag(delta)) g
so g is trained online by NLMS on the feature matrix A = W diag(delta). Few parameters (D per pair) -> fast convergence?
"""
import time

import numpy as np
import torch

from common import cli, load, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--g0", type=float, default=0.0, help="initial gain")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X, Y, W = d["X"].to(dev).float(), d["Y"].to(dev), d["W"].to(dev)
    real, K = d["real"].to(dev), d["k"]
    NL, T, D = X.shape
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(1, NL - k, device=dev), torch.arange(k + 1, NL, device=dev)
        t0 = time.time()
        g = torch.full((len(src), D), a.g0, device=dev)
        pred = torch.empty(len(src), T, Y.shape[2], device=dev)
        for t in range(T):
            x, delta = X[src, t], X[src, t] - X[src - 1, t]
            A = W[tl] * delta.unsqueeze(1)                                        # [n, E, D]
            p = torch.einsum("ned,nd->ne", W[tl], x) + torch.einsum("ned,nd->ne", A, g)
            pred[:, t] = p
            err = Y[tl, t] - p
            g += a.mu * torch.einsum("ned,ne->nd", A, err) / (A.square().sum((1, 2)).unsqueeze(-1) + 1e-6)
        pi = torch.topk(pred, K, dim=-1).indices
        h = np.full((NL, T), np.nan)
        h[k + 1:] = (pi.unsqueeze(-1) == real[tl].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
        report(f"L+{k} extrapolate mu {a.mu}", h, d["bounds"], f"  ({time.time() - t0:.1f} s, mean |g| {g.abs().mean():.2f})", d["gen"])


if __name__ == "__main__":
    main()
