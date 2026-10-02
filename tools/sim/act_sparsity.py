#!/usr/bin/env python3
"""Does subtracting a per-layer constant vector make an MoE expert's input sparse enough to skip weight columns?

usage: act_sparsity.py router_dump.bin model.gguf [layers=5,15,25,35] [n_tokens=24]
The dump is llama-eval-callback with ROUTER_DUMP=... (see examples/eval-callback): each MoE layer's expert input (ffn_norm-<il>)
and router logits for a prompt. Per layer, with mu = mean of the first half of the tokens (the calibration vector) and the second
half as test tokens:
  mu share      ||mu||^2 / mean ||x||^2  (how much of the signal is the constant part)
  rank          principal components of delta = x - mu needed for 90 / 95 / 99 % of its variance (limited by the number of tokens)
  error curve   for the experts the router really picks (top-8), the relative error of one expert's output E(x) = D(silu(Gx) * Ux)
                when only the fraction f of the input components (largest |.| per token) is kept:
                  raw     x_s = keep_f(x)
                  centred x_s = mu + keep_f(x - mu)   (G mu, U mu are constants: precomputed, exact)
                and the same keep fraction on the down-projection input h = silu(Gx) * Ux ("down").
Skipping an input component skips one weight column: f is the fraction of weight bytes still read for that matrix.
"""
import struct, sys
import numpy as np
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize

dump, model = sys.argv[1], sys.argv[2]
layers = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "5,15,25,35").split(",")]
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 24
FR = [1.0, 0.75, 0.5, 0.35, 0.25]
K = 8

recs = {}
with open(dump, 'rb') as f:
    while True:
        h = f.read(4)
        if not h:
            break
        n, = struct.unpack('I', h)
        name = f.read(n).decode()
        ne0, ne1 = struct.unpack('qq', f.read(16))
        recs.setdefault(name, np.frombuffer(f.read(ne0 * ne1 * 4), np.float32).reshape(ne1, ne0))

def xin(il):
    for k in (f'ffn_norm-{il}', f'attn_post_norm-{il}'):
        if k in recs:
            return recs[k]
    return None

def logit(il):
    return next((recs[k] for k in recs if k.startswith('ffn_moe_logits') and k.endswith(f'-{il}')), None)

T = {t.name: t for t in GGUFReader(model).tensors}

def deq_expert(il, kind, e):
    t = T[f'blk.{il}.ffn_{kind}_exps.weight']
    ne0, ne1, ne2 = [int(x) for x in t.shape]
    raw = np.ascontiguousarray(t.data.reshape(ne2, -1)[e])
    return dequantize(raw.reshape(1, -1), t.tensor_type).reshape(ne1, ne0).astype(np.float32)

def keep(v, f):
    """zero all but the fraction f of the components of each row of v with the largest magnitude"""
    if f >= 1.0:
        return v
    k = max(1, int(round(f * v.shape[1])))
    thr = -np.partition(-np.abs(v), k - 1, axis=1)[:, k - 1:k]
    return np.where(np.abs(v) >= thr, v, 0.0)

silu = lambda g: g / (1.0 + np.exp(-g))
rng = np.random.default_rng(0)
print(f"{'layer':>5} {'tokens':>6} {'mu share':>8} {'rank90':>6} {'rank95':>6} {'rank99':>6}   relative error of one expert's output, keeping fraction f of the input components")
print(f"{'':>5} {'':>6} {'':>8} {'':>6} {'':>6} {'':>6}   " + "  ".join(f"f={f:<4}" for f in FR) + "   (raw | centred | down-input)")
for il in layers:
    x = xin(il); lg = logit(il)
    if x is None or lg is None:
        print(il, "missing records"); continue
    n = x.shape[0]
    cal, test = x[:n // 2], x[n // 2:]
    mu = cal.mean(0)
    share = float(mu @ mu / np.mean(np.sum(x * x, 1)))
    d = x - x.mean(0)
    s = np.linalg.svd(d, compute_uv=False) ** 2
    c = np.cumsum(s) / s.sum()
    ranks = [int(np.searchsorted(c, q) + 1) for q in (0.90, 0.95, 0.99)]
    # test tokens and the experts the router really picks for them
    ti = rng.choice(len(test), size=min(NT, len(test)), replace=False)
    lt = lg[n // 2:][ti]
    sel = np.argsort(-lt, axis=1)[:, :K]
    err = {m: np.zeros(len(FR)) for m in ("raw", "cen", "down")}
    cnt = 0
    for e in sorted(set(sel.flatten().tolist())):
        rows = [i for i in range(len(ti)) if e in sel[i]]
        G, U, D = deq_expert(il, 'gate', e), deq_expert(il, 'up', e), deq_expert(il, 'down', e)
        gm, um = G @ mu, U @ mu
        for i in rows:
            xt = test[ti[i]]
            g, u = G @ xt, U @ xt
            h = silu(g) * u
            ref = D @ h
            rn = np.linalg.norm(ref) + 1e-9
            for k, f in enumerate(FR):
                xr = keep(xt[None], f)[0]
                err["raw"][k] += np.linalg.norm(D @ (silu(G @ xr) * (U @ xr)) - ref) / rn
                dl = keep((xt - mu)[None], f)[0]
                err["cen"][k] += np.linalg.norm(D @ (silu(gm + G @ dl) * (um + U @ dl)) - ref) / rn
                err["down"][k] += np.linalg.norm(D @ keep(h[None], f)[0] - ref) / rn
            cnt += 1
        del G, U, D
    print(f"{il:5d} {n:6d} {share:8.3f} {ranks[0]:6d} {ranks[1]:6d} {ranks[2]:6d}")
    for m, lab in (("raw", "raw"), ("cen", "centred"), ("down", "down")):
        print(f"{'':>5} {'':>6} {'':>8} {'':>6} {'':>6} {lab:>6}   " + "  ".join(f"{v / cnt:6.3f}" for v in err[m]), flush=True)
