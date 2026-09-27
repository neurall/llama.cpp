#!/usr/bin/env python3
"""Router-predictor lab: fast offline iteration on dumped MoE routing.

  pred_lab.py build model.gguf out.pt dump0.bin [dump1.bin ...]
      dumps from llama-eval-callback with ROUTER_DUMP=<file> (prompt + generated answer as one
      prefill: a causal model gives each token the same activations as in decode)
  pred_lab.py eval data.pt [--ahead 2] [--mu 0.5] [--every 1]

eval streams the samples in order (one user session), predicting each layer's top-k from an earlier
layer's MoE input and learning online (NLMS on the router logits, starting from the target router),
all layers batched on the GPU.
"""
import argparse, os, re, struct, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))


def read_dump(path):
    recs = {}
    with open(path, "rb") as f:
        while h := f.read(4):
            n, = struct.unpack("I", h)
            name = f.read(n).decode()
            ne0, ne1 = struct.unpack("qq", f.read(16))
            a = np.frombuffer(f.read(ne0 * ne1 * 4), np.float32).reshape(ne1, ne0)
            recs.setdefault(name, []).append(a)
    return {k: np.concatenate(v) for k, v in recs.items()}


def build(a):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(a.model)
    W, B = {}, {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(ffn_gate_inp\.weight|exp_probs_b\.bias)", t.name)
        if m:
            arr = dequantize(t.data, t.tensor_type).reshape(tuple(reversed([int(x) for x in t.shape]))).astype(np.float32)
            (W if "gate_inp" in m.group(2) else B)[int(m.group(1))] = arr
    layers = sorted(W)
    X, Y, bounds = [], [], [0]
    for d in a.dumps:
        recs = read_dump(d)
        X.append(np.stack([recs[f"ffn_norm-{l}"] for l in layers]))        # [NL, T, D]
        Y.append(np.stack([recs[f"ffn_moe_logits-{l}"] for l in layers]))  # [NL, T, E]
        bounds.append(bounds[-1] + X[-1].shape[1])
    out = {
        "layers": layers,
        "X": torch.tensor(np.concatenate(X, 1)).half(),
        "Y": torch.tensor(np.concatenate(Y, 1)),
        "W": torch.tensor(np.stack([W[l] for l in layers])),
        "B": torch.tensor(np.stack([B[l] for l in layers])) if B else None,
        "bounds": bounds,
        "k": int(a.k),
    }
    torch.save(out, a.out)
    print(f"{len(layers)} MoE layers, {bounds[-1]} tokens in {len(a.dumps)} samples, D={out['X'].shape[2]} E={out['W'].shape[1]} -> {a.out}")


def evaluate(a):
    dev = "cuda"
    d = torch.load(a.data)
    X, Y, W, B = d["X"].to(dev).float(), d["Y"].to(dev), d["W"].to(dev), d["B"]
    B = B.to(dev) if B is not None else None
    K, bounds = d["k"], d["bounds"]
    NL, T, D = X.shape

    def score(lg, tl):  # selection score of target layers tl (sigmoid + bias when the model has one)
        return torch.sigmoid(lg) + B[tl] if B is not None else lg

    def topk(lg, tl):
        return torch.topk(score(lg, tl), K, dim=-1).indices

    real = torch.stack([topk(Y[l], l) for l in range(NL)])                 # [NL, T, K]

    def overlap(pred_idx, real_idx):  # [..., K] each -> [...]
        return (pred_idx.unsqueeze(-1) == real_idx.unsqueeze(-2)).any(-1).float().mean(-1)

    t0 = time.time()
    for k in range(1, a.ahead + 1):
        src, tl = torch.arange(NL - k, device=dev), torch.arange(k, NL, device=dev)
        base = torch.einsum("led,ltd->lte", W[tl], X[src])               # router of L+k on x_L
        ov_router = overlap(topk(base, tl[:, None]), real[tl])            # [NL-k, T]
        M = W[tl].clone()                                                 # learned: starts as the router
        xs = X[src]
        nrm = (xs * xs).sum(-1)                                           # [NL-k, T]
        pred = torch.empty_like(base)
        for t in range(T):
            p = torch.einsum("led,ld->le", M, xs[:, t])
            pred[:, t] = p
            if a.every > 0 and t % a.every == 0:
                g = (Y[tl, t] - p) * (a.mu / (nrm[:, t:t + 1] + 1e-6))
                M += g.unsqueeze(-1) * xs[:, t].unsqueeze(1)
        ov_learn = overlap(topk(pred, tl[:, None]), real[tl])
        per = lambda ov: " ".join(f"{ov[:, bounds[i]:bounds[i + 1]].mean().item():.3f}" for i in range(len(bounds) - 1))
        print(f"L+{k}: router {ov_router.mean().item():.3f}  learned {ov_learn.mean().item():.3f}   per sample: router [{per(ov_router)}]  learned [{per(ov_learn)}]")
    print(f"({time.time() - t0:.2f} s, {T} tokens, {NL} layers, mu={a.mu}, every {a.every})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("model"); b.add_argument("out"); b.add_argument("dumps", nargs="+")
    b.add_argument("--k", default=8, help="experts per token")
    e = sp.add_parser("eval")
    e.add_argument("data")
    e.add_argument("--ahead", type=int, default=2)
    e.add_argument("--mu", type=float, default=0.5)
    e.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else evaluate(a)


if __name__ == "__main__":
    main()
