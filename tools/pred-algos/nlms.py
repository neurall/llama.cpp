#!/usr/bin/env python3
"""Algo 1: linear transform per (layer L, lookahead k) trained online by NLMS (delta rule).

W starts as layer L+k's router and predicts its logits from layer L's MoE input:
    W += mu / |x|^2 * (real_logits - W x) x^T
Also scores the untrained router. Runs --time-limit seconds, then saves and exits (common.Run).
--input wpool: all layers' activations weighted decay^distance instead of x_L alone.
--save ref.npz (with --time-limit 0 --fresh): per-token hits used as the same-token reference by the other algos.
"""
import numpy as np
import torch

from common import Run, cli, load, parse, topk_hit


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--input", choices=["x", "wpool"], default="x")
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--save", default=None, help="save the hits {nlms_k, router_k: [NL, T]} to this .npz")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X, Y, W, real = d["X"].to(dev).float(), d["Y"].to(dev), d["W"].to(dev), d["real"].to(dev)
    NL, T, _ = X.shape
    Xin = X
    if a.input == "wpool":
        # distance in layer steps back through the computation: current token j<=L at L-j, previous token j at L+NL-j
        Ls, js = torch.arange(NL, device=dev)[:, None], torch.arange(NL, device=dev)[None, :]
        A = torch.where(js <= Ls, a.decay ** (Ls - js).float(), torch.zeros((), device=dev))
        B = a.decay ** (Ls + NL - js).float()
        norm = A.sum(1, keepdim=True) + B.sum(1, keepdim=True)
        Xprev = torch.cat([torch.zeros_like(X[:, :1]), X[:, :-1]], 1)
        Xin = torch.einsum("lj,jtd->ltd", A / norm, X) + torch.einsum("lj,jtd->ltd", B / norm, Xprev)
        Xin = Xin * (X.square().mean(-1, keepdim=True) / (Xin.square().mean(-1, keepdim=True) + 1e-12)).sqrt()
        del Xprev
    run = Run(a, f"nlms-{a.input}", T)
    ks = range(1, a.ahead + 1)
    pairs = {k: (torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)) for k in ks}
    M = {k: m.to(dev) for k, m in run.state["M"].items()} if "M" in run.state else {k: W[pairs[k][1]].clone() for k in ks}
    tag = f"nlms {a.input}{' d' + str(a.decay) if a.input == 'wpool' else ''} mu {a.mu}"
    for t in range(run.start, T):
        if run.expired(t):
            break
        for k in ks:
            src, tl = pairs[k]
            x = Xin[src, t]
            p = torch.einsum("ned,nd->ne", M[k], x)
            run.hit(f"L+{k} {tag}", NL)[k:, t] = topk_hit(p, real[tl, t])
            run.hit(f"L+{k} router", NL)[k:, t] = topk_hit(torch.einsum("ned,nd->ne", W[tl], X[src, t]), real[tl, t])
            if a.every > 0 and t % a.every == 0:
                M[k] += ((Y[tl, t] - p) * (a.mu / (x.square().sum(-1, keepdim=True) + 1e-6))).unsqueeze(-1) * x.unsqueeze(1)
    run.save(M={k: m.cpu() for k, m in M.items()})
    run.report(d)
    if a.save:
        np.savez(a.save, **{f"{'router' if 'router' in n else 'nlms'}_k{n[2]}": h for n, h in run.hits.items()})


if __name__ == "__main__":
    main()
