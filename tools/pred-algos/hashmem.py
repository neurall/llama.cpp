#!/usr/bin/env python3
"""Algo 8: hash memory, the simplest learner: key = hash(previous token's activation at the target layer L+k
(--prev target, kept per layer: O(1)) or at its last layer (--prev last), current token's x_L),
value = the experts layer L+k really picked the last time that key was seen. Recalled on a repeat,
otherwise the router (and, in the report, NLMS from the reference run). Runs --time-limit seconds (common.Run).

Hash = sign bits of fixed random projections (--bits per vector, the two codes concatenated into one key).
"""
import numpy as np
import torch

import common
from common import Run, cli, load, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--bits", type=int, default=16, help="sign bits per hashed vector")
    ap.add_argument("--prev", choices=["last", "target"], default="target")
    ap.add_argument("--prev-bits", type=int, default=None, help="bits for the previous token's vector (default --bits, 0: current only)")
    a = parse(ap)
    pb = a.bits if a.prev_bits is None else a.prev_bits
    d = load(a.data, need_x=True)
    X, W, real, K = d["X"].float(), d["W"], d["real"].numpy(), d["k"]
    NL, T, D = X.shape
    g = torch.Generator().manual_seed(0)
    Pc, Pp = torch.randn(a.bits, D, generator=g), torch.randn(max(pb, 1), D, generator=g)

    def codes(v, P, b):  # [..., D] -> int64 codes
        if b == 0:
            return torch.zeros(v.shape[:-1], dtype=torch.long)
        return ((v @ P[:b].T > 0).long() * (2 ** torch.arange(b))).sum(-1)
    cur = codes(X, Pc, a.bits).numpy()                                   # [NL, T]
    pc = codes(X, Pp, pb).numpy()
    prev = np.concatenate([np.full((NL, 1), -1), pc[:, :-1]], 1)         # previous token's codes, every layer
    run = Run(a, f"hashmem-{a.prev}-{pb}+{a.bits}", T)
    ks = range(1, a.ahead + 1)
    mem = run.state.get("mem") or {k: [dict() for _ in range(NL - k)] for k in ks}
    router = {k: torch.topk(torch.einsum("led,ltd->lte", W[k:], X[:NL - k]), K, dim=-1).indices.numpy() for k in ks}
    name = f"hashmem {a.prev} {pb}+{a.bits}b"
    for t in range(run.start, T):
        if run.expired(t):
            break
        for k in ks:
            h_rec, h_all = run.hit(f"L+{k} {name} recalled only", NL), run.hit(f"L+{k} {name} + router fallback", NL)
            for L in range(NL - k):
                tgt = L + k
                key = (int(prev[tgt if a.prev == "target" else NL - 1, t]) << a.bits) | int(cur[L, t])
                got = mem[k][L].get(key)
                h = np.isin(real[tgt, t], got if got is not None else router[k][L, t]).mean()
                h_all[tgt, t] = h
                if got is not None:
                    h_rec[tgt, t] = h
                mem[k][L][key] = real[tgt, t]
    run.save(mem=mem)
    if common.REF is not None:
        # recalled key -> stored experts, otherwise NLMS (reference run)
        for k in ks:
            h_rec = run.hits[f"L+{k} {name} recalled only"]
            seen = np.isfinite(run.hits[f"L+{k} {name} + router fallback"])
            run.hits[f"L+{k} {name} + NLMS fallback"] = np.where(np.isfinite(h_rec), h_rec, np.where(seen, common.REF[f"nlms_k{k}"], np.nan))
    for k in ks:
        n = f"L+{k} {name} recalled only"
        rate = np.isfinite(run.hits[n]).sum() / max(np.isfinite(run.hits[f"L+{k} {name} + router fallback"]).sum(), 1)
        print(f"L+{k} recall rate {rate:.3f}")
    run.report(d)


if __name__ == "__main__":
    main()
