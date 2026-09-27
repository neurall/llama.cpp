#!/usr/bin/env python3
"""Algo 3: stacked skip-connected shadow net, one small block per model layer, trained online with Adam.

State s [H] runs alongside the model. Block L mixes in 1-3 layers of current and past activations:
    u = proj_L([x_L, x_{L-1}, prev-token x_{L+1}, prev-token x_{L+2}])     (RMS-normalised, missing ones = 0)
    s = s + down_L(silu(gate_L([s, u])) * up_L([s, u]))
Head (L, k) predicts a residual on the router of L+k applied to x_L (zero-init). Loss: MSE to the real
router logits of every target, backprop through the whole stack each token.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import cli, load, report, parse


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default=None, help="path to save/resume weights+optimizer across reruns")
    ap.add_argument("--time-limit", type=float, default=None, help="stop this pass after N seconds (mid-stream)")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X = d["X"].to(dev)
    Y, W, real, K = d["Y"].to(dev), d["W"].to(dev), d["real"].to(dev), d["k"]
    NL, T, D = X.shape
    E, H, A = W.shape[1], a.hidden, a.ahead

    def p(*shape, scale):
        return (torch.randn(*shape, device=dev) * scale).requires_grad_()
    if a.ckpt and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev)
        proj, up, gate, down, head = (ck[n].to(dev).requires_grad_() for n in ("proj", "up", "gate", "down", "head"))
        opt = torch.optim.Adam([proj, up, gate, down, head], lr=a.lr)
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["lr"] = a.lr                      # the saved state would override the lr this round asks for
        start = ck.get("tokens_seen", 0) % T    # continue the stream where the last round stopped
        print(f"  resumed from {a.ckpt} at token {start}")
    else:
        proj = p(NL, 4 * D, H, scale=(4 * D) ** -0.5)
        up, gate = p(NL, 2 * H, 2 * H, scale=(2 * H) ** -0.5), p(NL, 2 * H, 2 * H, scale=(2 * H) ** -0.5)
        down = p(NL, 2 * H, H, scale=0.0)
        head = torch.zeros(NL, A, H, E, device=dev, requires_grad=True)
        opt = torch.optim.Adam([proj, up, gate, down, head], lr=a.lr)
        start = 0
    # (source L, lookahead k) pairs with a target inside the model
    pairs = [(L, k) for k in range(1, A + 1) for L in range(NL - k)]
    Ls = torch.tensor([L for L, _ in pairs], device=dev)
    ks = torch.tensor([k for _, k in pairs], device=dev)
    pred = torch.full((len(pairs), T, E), float("nan"), device=dev)
    zero = torch.zeros(D, device=dev)
    t0 = time.time()
    t_stop = T
    for t in range(start, T):
        if a.time_limit and t % 64 == 0 and t > start and time.time() - t0 > a.time_limit:
            t_stop = t
            break
        cur = rms(X[:, t].float())
        prev = rms(X[:, t - 1].float()) if t else torch.zeros(NL, D, device=dev)
        s = torch.zeros(H, device=dev)
        states = []
        for L in range(NL):
            u = torch.cat([cur[L], cur[L - 1] if L else zero,
                           prev[L + 1] if L + 1 < NL else zero, prev[L + 2] if L + 2 < NL else zero]) @ proj[L]
            z = torch.cat([s, u])
            s = s + (F.silu(z @ gate[L]) * (z @ up[L])) @ down[L]
            states.append(s)
        S = torch.stack(states)                                               # [NL, H]
        base = torch.einsum("ped,pd->pe", W[Ls + ks], X[Ls, t].float())
        out = base + torch.einsum("ph,phe->pe", S[Ls], head[Ls, ks - 1])
        pred[:, t] = out.detach()
        loss = (out - Y[Ls + ks, t]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred_v = pred[:, start:t_stop]
    pi = torch.topk(pred_v, K, dim=-1).indices
    hit = (pi.unsqueeze(-1) == real[Ls + ks, start:t_stop].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
    for k in range(1, A + 1):
        hm = np.full((NL, T), np.nan)
        sel = (ks == k).cpu().numpy()
        hm[k:, start:t_stop] = hit[sel]
        report(f"L+{k} stack h{H} lr {a.lr}", hm, d["bounds"], f"  ({time.time() - t0:.1f} s, tokens {start}-{t_stop} of {T})", None if a.time_limit else d["gen"])
    if a.ckpt:
        torch.save({"proj": proj.detach().cpu(), "up": up.detach().cpu(), "gate": gate.detach().cpu(),
                    "down": down.detach().cpu(), "head": head.detach().cpu(),
                    "opt": opt.state_dict(), "tokens_seen": t_stop % T}, a.ckpt)


if __name__ == "__main__":
    main()
