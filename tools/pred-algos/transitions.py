#!/usr/bin/env python3
"""Algo 6: expert transition tables (only the discrete picks, no activations to learn from).

Per (layer L, lookahead k), decayed counts:
  C1[i, j]: expert j picked at L+k when expert i was picked at L (same token)
  C2[i, j]: expert j picked at L+k when expert i was picked at L+k on the previous token
score_j = a * log P1(j | picks at L) + b * log P2(j | previous picks at L+k) [+ router log-softmax of L+k on x_L]
where P = mean over the 8 source experts of the row-normalised counts.
"""
import time

import numpy as np
import torch

from common import cli, load, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=1.0, help="multiply counts by this every token (1: none)")
    ap.add_argument("--no-router", action="store_true")
    a = parse(ap)
    d = load(a.data, need_x=not a.no_router)
    dev = a.device
    real, K, E = d["real"].to(dev), d["k"], d["n_expert"]
    NL, T, _ = real.shape
    oh = torch.zeros(NL, T, E, device=dev).scatter_(2, real, 1.0)            # one-hot picks [NL, T, E]

    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        n = len(src)
        t0 = time.time()
        if not a.no_router:
            X, W = d["X"], d["W"].to(dev)
            base = torch.cat([torch.einsum("led,ltd->lte", W[tl], X[src.cpu(), c:c + 4096].to(dev).float())
                              for c in range(0, T, 4096)], 1).log_softmax(-1)
        C1 = torch.zeros(n, E, E, device=dev)
        C2 = torch.zeros(n, E, E, device=dev)
        score = torch.empty(n, T, E, device=dev)
        eps = 1e-3
        for t in range(T):
            s_src = oh[src, t]                                                  # [n, E]
            s_prev = oh[tl, t - 1] if t else torch.zeros_like(s_src)
            P1 = C1 / (C1.sum(-1, keepdim=True) + eps)
            P2 = C2 / (C2.sum(-1, keepdim=True) + eps)
            p1 = torch.einsum("ni,nij->nj", s_src, P1) / K
            p2 = torch.einsum("ni,nij->nj", s_prev, P2) / K
            sc = a.a * torch.log(p1 + eps) + a.b * torch.log(p2 + eps)
            score[:, t] = sc if a.no_router else sc + base[:, t]
            tgt = oh[tl, t]
            if a.decay != 1.0:
                C1 *= a.decay
                C2 *= a.decay
            C1 += s_src.unsqueeze(-1) * tgt.unsqueeze(1)
            C2 += s_prev.unsqueeze(-1) * tgt.unsqueeze(1)
        p = torch.topk(score, K, dim=-1).indices
        h = np.full((NL, T), np.nan)
        h[k:] = (p.unsqueeze(-1) == real[tl].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
        name = f"L+{k} transitions a {a.a} b {a.b} decay {a.decay}" + (" (no router)" if a.no_router else " + router")
        report(name, h, d["bounds"], f"  ({time.time() - t0:.1f} s)", d["gen"])


if __name__ == "__main__":
    main()
