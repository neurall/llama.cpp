#!/usr/bin/env python3
"""Low-rank NLMS: the cheap version of algo 1 (target: 5-10x fewer weight bytes per token than full NLMS).

Per source layer L a shared projection z = V_L x (r dims), per lookahead k a small head p = U_{L,k} z.
V_L starts as the top-r right singular vectors of the stacked routers of L+1..L+ahead, U_{L,k} as router(L+k)
projected onto them; U is trained online by NLMS on z, V optionally by the same error (--mu-v).
Weight bytes per source layer: D*r + ahead*r*E (fp16 x2) instead of ahead*D*E.
Also reports precision@M: the share of the M predicted experts that the layer really used.
"""
import torch

from common import Run, cli, load, parse, topk_hit


def precision(logits, real, m):
    p = torch.topk(logits, m, dim=-1).indices
    return (p.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()


def main():
    ap = cli(__doc__)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--mu-v", type=float, default=0.0)
    ap.add_argument("--m", type=int, default=6, help="precision@M")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X, Y, W, real = d["X"].to(dev).float(), d["Y"].to(dev), d["W"].to(dev).float(), d["real"].to(dev)
    NL, T, D = X.shape
    E = W.shape[1]
    r = a.rank
    run = Run(a, f"lowrank-r{r}", T)
    ks = range(1, a.ahead + 1)
    pairs = {k: (torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)) for k in ks}
    if "V" in run.state:
        V = run.state["V"].to(dev)
        U = {k: u.to(dev) for k, u in run.state["U"].items()}
    else:
        V = torch.zeros(NL, r, D, device=dev)
        for L in range(NL - 1):
            tg = torch.cat([W[L + k] for k in ks if L + k < NL])           # [(E*n), D]
            V[L] = torch.linalg.svd(tg, full_matrices=False).Vh[:r]
        U = {k: torch.einsum("ned,nrd->ner", W[pairs[k][1]], V[pairs[k][0]]) for k in ks}
    full = a.ahead * D * E
    low = D * r + a.ahead * r * E
    tag = f"lowrank r{r} mu {a.mu} mu_v {a.mu_v} ({full / low:.1f}x fewer weights)"
    for t in range(run.start, T):
        if run.expired(t):
            break
        z_all = torch.einsum("nrd,nd->nr", V, X[:, t])                     # [NL, r]
        dV = torch.zeros_like(V) if a.mu_v else None
        for k in ks:
            src, tl = pairs[k]
            z = z_all[src]
            p = torch.einsum("ner,nr->ne", U[k], z)
            run.hit(f"L+{k} {tag}", NL)[k:, t] = topk_hit(p, real[tl, t])
            run.hit(f"L+{k} P@{a.m} {tag}", NL)[k:, t] = precision(p, real[tl, t], a.m)
            err = Y[tl, t] - p
            U[k] += (err * (a.mu / (z.square().sum(-1, keepdim=True) + 1e-6))).unsqueeze(-1) * z.unsqueeze(1)
            if dV is not None:
                dV.index_add_(0, src, torch.einsum("ner,ne->nr", U[k], err).unsqueeze(-1) * X[src, t].unsqueeze(1))
        if dV is not None:
            V += a.mu_v * dV / (X[:, t].square().sum(-1).view(NL, 1, 1) + 1e-6)
    run.save(V=V.cpu(), U={k: u.cpu() for k, u in U.items()})
    run.report(d)


if __name__ == "__main__":
    main()
