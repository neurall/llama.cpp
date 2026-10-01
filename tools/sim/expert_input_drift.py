#!/usr/bin/env python3
"""How wrong is an expert's output if it is computed on an earlier layer's input?

usage: expert_input_drift.py router_dump.bin model.gguf [layers=5,15,25,35] [n_tokens=64]
The dump (llama-eval-callback with ROUTER_DUMP=..., see examples/eval-callback) holds each MoE layer's expert input
(ffn_norm-<il> or attn_post_norm-<il>) and router logits for a prompt batch. For layer L+1 this script takes the
experts the router really selects, computes their outputs on the real input x_{L+1} and on the proxy x_L (what is known
one layer earlier, the only thing a speculative precompute could use), and prints:
  cos(x_L, x_{L+1}) of the inputs, the relative error of the routed sum sum_e w_e E_e(x)  (||E(x_L) - E(x_{L+1})|| /
  ||E(x_{L+1})||), and the same for the k-layer-ahead proxy x_{L+1-k} (k = 2, 4).
"""
import struct, sys
import numpy as np
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize

dump, model = sys.argv[1], sys.argv[2]
layers = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "5,15,25,35").split(",")]
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 64

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

logit = lambda il: next((recs[k] for k in recs if k.startswith('ffn_moe_logits') and k.endswith(f'-{il}')), None)

r = GGUFReader(model)
T = {t.name: t for t in r.tensors}

def expert_w(il, kind):
    t = T[f'blk.{il}.ffn_{kind}_exps.weight']
    return t

def deq_q2_0(raw, n):
    # Q2_0 blocks of 64 weights: fp16 delta + 16 bytes; byte b holds weights 4b..4b+3 at bits 0/2/4/6, value (q - 1) * d
    blk = raw.reshape(-1, 18)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)          # [nb, 1]
    q = blk[:, 2:]
    v = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32) - 1.0
    return (v * d).reshape(-1)[:n]

def deq_expert(t, e):
    # one expert's matrix, dequantized: shape [rows, cols] with rows = output dim
    ne0, ne1, ne2 = [int(x) for x in t.shape]
    raw = np.ascontiguousarray(t.data.reshape(ne2, -1)[e])
    if t.tensor_type.name == 'Q2_0':
        return deq_q2_0(raw, ne0 * ne1).reshape(ne1, ne0)
    a = dequantize(raw.reshape(1, -1), t.tensor_type)
    return a.reshape(ne1, ne0).astype(np.float32)

K = 8
rng = np.random.default_rng(0)
print("layer  cos(x_L,x_L+1)  relerr(k=1)  relerr(k=2)  relerr(k=4)   (routed-sum relative error, real router weights)")
for L1 in layers:
    x1 = xin(L1); lg = logit(L1)
    if x1 is None or lg is None:
        print(L1, "missing records"); continue
    toks = rng.choice(x1.shape[0], size=min(NT, x1.shape[0]), replace=False)
    # the experts really selected (softmax gating over the top-K logits)
    sel = np.argsort(-lg[toks], axis=1)[:, :K]
    w = np.take_along_axis(lg[toks], sel, axis=1)
    w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    need = sorted(set(sel.flatten().tolist()))
    G = {e: deq_expert(expert_w(L1, 'gate'), e) for e in need}
    U = {e: deq_expert(expert_w(L1, 'up'), e) for e in need}
    D = {e: deq_expert(expert_w(L1, 'down'), e) for e in need}
    def routed(x):  # x [n_tok, n_embd] -> weighted routed sum [n_tok, n_embd]
        out = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(K):
                e = sel[i, j]
                g = G[e] @ x[i]; u = U[e] @ x[i]
                out[i] += w[i, j] * (D[e] @ ((g / (1 + np.exp(-g))) * u))
        return out
    ref = routed(x1[toks])
    row = []
    for k in (1, 2, 4):
        xp = xin(L1 - k)
        if xp is None:
            row.append(float('nan')); continue
        approx = routed(xp[toks])
        row.append(np.linalg.norm(approx - ref) / np.linalg.norm(ref))
    x0 = xin(L1 - 1)
    cos = np.mean(np.sum(x0[toks] * x1[toks], 1) / (np.linalg.norm(x0[toks], axis=1) * np.linalg.norm(x1[toks], axis=1)))
    print(f"{L1:5d}  {cos:14.3f}  {row[0]:11.3f}  {row[1]:11.3f}  {row[2]:11.3f}")
