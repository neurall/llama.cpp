#!/usr/bin/env python3
"""Algo 5: activation hash memory (random-projection LSH of x_L -> counts of L+k's picks).

Per (layer L, lookahead k): R tables, each hashes sign(P_r x_L) into 2^bits buckets holding counts of the
experts L+k really picked. p = mean over tables of the bucket's normalised counts.
score = a * log(p + eps) [+ router log-softmax]. No gradients: one sighting fills a bucket.
"""
import time

import numpy as np
import torch

from common import cli, load, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--bits", type=int, default=12)
    ap.add_argument("--tables", type=int, default=4)
    ap.add_argument("--a", type=float, default=0.3)
    ap.add_argument("--no-router", action="store_true")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X, W, real, K, E = d["X"], d["W"].to(dev), d["real"].to(dev), d["k"], d["n_expert"]
    NL, T, D = X.shape
    oh = torch.zeros(NL, T, E, device=dev).scatter_(2, real, 1.0)
    g = torch.Generator(device=dev).manual_seed(0)
    P = torch.randn(NL, a.tables, a.bits, D, device=dev, generator=g)
    pow2 = 2 ** torch.arange(a.bits, device=dev)
    codes, base = [], []
    for c in range(0, T, 4096):                                              # [NL, R, T] bucket ids, router logits
        xc = X[:, c:c + 4096].to(dev).float()
        codes.append(((torch.einsum("lrbd,ltd->lrtb", P, xc) > 0).long() * pow2).sum(-1))
        base.append(xc)
    codes = torch.cat(codes, 2)
    eps = 1e-3
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        n = len(src)
        t0 = time.time()
        rb = torch.cat([torch.einsum("led,ltd->lte", W[tl], xc[src]) for xc in base], 1).log_softmax(-1)
        C = torch.zeros(n, a.tables, 2 ** a.bits, E, device=dev)
        score = torch.empty(n, T, E, device=dev)
        ni, ri = torch.arange(n, device=dev)[:, None], torch.arange(a.tables, device=dev)[None, :]
        for t in range(T):
            b = codes[src, :, t]                                              # [n, R]
            cnt = C[ni, ri, b]                                                # [n, R, E]
            p = (cnt / (cnt.sum(-1, keepdim=True) + eps)).mean(1)
            sc = a.a * torch.log(p + eps)
            score[:, t] = sc if a.no_router else sc + rb[:, t]
            C[ni, ri, b] += oh[tl, t].unsqueeze(1)
        pi = torch.topk(score, K, dim=-1).indices
        h = np.full((NL, T), np.nan)
        h[k:] = (pi.unsqueeze(-1) == real[tl].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
        name = f"L+{k} lsh {a.tables}x{a.bits}b a {a.a}" + (" (no router)" if a.no_router else " + router")
        report(name, h, d["bounds"], f"  ({time.time() - t0:.1f} s)", d["gen"])


if __name__ == "__main__":
    main()
