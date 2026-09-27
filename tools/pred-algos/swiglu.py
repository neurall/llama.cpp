#!/usr/bin/env python3
"""Algo 2: pooled activations -> SwiGLU per (layer L, lookahead k), trained online with Adam.

Input: mean of the current token's MoE inputs of layers 0..L and the previous token's L+1..end (RMS-normalised).
Output: residual on the router of L+k applied to x_L (down zero-initialised, so it starts at router quality).
Loss: MSE to L+k's real router logits (--loss mse) or BCE on its real top-k one-hot (--loss bce).
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import cli, load, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--ahead", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", choices=["none", "inv-sqrt"], default="none",
                     help="inv-sqrt: lr / sqrt(1 + t / warmup), warmup = --lr-warmup tokens")
    ap.add_argument("--lr-warmup", type=int, default=512)
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse")
    ap.add_argument("--input", choices=["pool", "x"], default="pool")
    ap.add_argument("--ckpt", default=None, help="path to save/resume weights+optimizer across reruns")
    ap.add_argument("--time-limit", type=float, default=None, help="stop this pass after N seconds (mid-stream), for short tune-and-resume rounds")
    a = parse(ap)
    d = load(a.data, need_x=True)
    dev = a.device
    X = d["X"].to(dev)                                                        # fp16 [NL, T, D]
    Y, W, real, K = d["Y"].to(dev), d["W"].to(dev), d["real"].to(dev), d["k"]
    NL, T, D = X.shape
    E = W.shape[1]
    oh = torch.zeros(NL, T, E, device=dev).scatter_(2, real, 1.0)
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        n = len(src)
        t0 = time.time()
        ckpt_path = f"{a.ckpt}.k{k}.pt" if a.ckpt else None
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=dev)
            up, gate, down = ck["up"].to(dev).requires_grad_(), ck["gate"].to(dev).requires_grad_(), ck["down"].to(dev).requires_grad_()
            opt = torch.optim.Adam([up, gate, down], lr=a.lr)
            opt.load_state_dict(ck["opt"])
            print(f"  resumed L+{k} from {ckpt_path} ({ck.get('tokens_seen', '?')} tokens seen before)")
        else:
            up = (torch.randn(n, D, a.hidden, device=dev) / D ** 0.5).requires_grad_()
            gate = (torch.randn(n, D, a.hidden, device=dev) / D ** 0.5).requires_grad_()
            down = torch.zeros(n, a.hidden, E, device=dev, requires_grad=True)
            opt = torch.optim.Adam([up, gate, down], lr=a.lr)
        Wt = W[tl]
        mask = (torch.arange(NL, device=dev)[None, :] <= src[:, None]).float().unsqueeze(-1)  # [n, NL, 1]
        pred = torch.full((n, T, E), float("nan"), device=dev)
        t_stop = T
        for t in range(T):
            if a.time_limit and t % 64 == 0 and t > 0 and time.time() - t0 > a.time_limit:
                t_stop = t
                break
            xs = X[src, t].float()
            if a.input == "pool":
                cur, prev = X[:, t].float(), (X[:, t - 1].float() if t else torch.zeros(NL, D, device=dev))
                inp = (mask * cur[None] + (1 - mask) * prev[None]).mean(1)
            else:
                inp = xs
            inp = inp * torch.rsqrt(inp.square().mean(-1, keepdim=True) + 1e-6)
            h = F.silu(torch.bmm(inp.unsqueeze(1), gate)) * torch.bmm(inp.unsqueeze(1), up)
            out = torch.einsum("ned,nd->ne", Wt, xs) + torch.bmm(h, down).squeeze(1)
            pred[:, t] = out.detach()
            if a.loss == "mse":
                loss = (out - Y[tl, t]).square().mean()
            else:
                loss = F.binary_cross_entropy_with_logits(out, oh[tl, t])
            if a.lr_decay == "inv-sqrt":
                for g in opt.param_groups:
                    g["lr"] = a.lr / (1 + t / a.lr_warmup) ** 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()
        pred_v = pred[:, :t_stop]
        pi = torch.topk(pred_v, K, dim=-1).indices
        hm = np.full((NL, T), np.nan)
        hm[k:, :t_stop] = (pi.unsqueeze(-1) == real[tl, :t_stop].unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()
        name = f"L+{k} swiglu {a.input} h{a.hidden} {a.loss} lr {a.lr} decay {a.lr_decay}"
        report(name, hm, d["bounds"], f"  ({time.time() - t0:.1f} s, {t_stop}/{T} tokens)", None if a.time_limit else d["gen"])
        if ckpt_path:
            torch.save({"up": up.detach().cpu(), "gate": gate.detach().cpu(), "down": down.detach().cpu(),
                        "opt": opt.state_dict(), "tokens_seen": t_stop}, ckpt_path)


if __name__ == "__main__":
    main()
