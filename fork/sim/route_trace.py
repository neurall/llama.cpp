#!/usr/bin/env python3
"""Selected experts per layer per token from router dumps -> npz, for mining routing structure offline.

usage: route_trace.py model.gguf out.npz dump1.bin [dump2.bin ...]
The dumps are llama-eval-callback with ROUTER_DUMP=... (examples/eval-callback). A dump is one prompt as a single prefill; a causal
model routes each token the same way in decode. Only the router logits are read (the big expert-input records are skipped),
selection = top-K of sigmoid(logits) + bias (models with exp_probs_b) or of the logits. out.npz: ids [T, NL, K] int16, bounds
[n_samples + 1] (token offsets of each dump), layers [NL] (model layer numbers), n_expert.
"""
import re, struct, sys
import numpy as np
sys.path.insert(0, 'gguf-py')

def read_logits(path):
    out = {}
    with open(path, 'rb') as f:
        while h := f.read(4):
            n, = struct.unpack('I', h)
            name = f.read(n).decode()
            ne0, ne1 = struct.unpack('qq', f.read(16))
            if name.startswith('ffn_moe_logits') and re.search(r'-(\d+)$', name):
                out[int(re.search(r'-(\d+)$', name).group(1))] = np.frombuffer(f.read(ne0 * ne1 * 4), np.float32).reshape(ne1, ne0)
            else:
                f.seek(ne0 * ne1 * 4, 1)
    return out

def main():
    from gguf import GGUFReader
    from gguf.quants import dequantize
    model, out, dumps = sys.argv[1], sys.argv[2], sys.argv[3:]
    K = 8
    bias = {}
    for t in GGUFReader(model).tensors:
        m = re.match(r'blk\.(\d+)\.exp_probs_b\.bias', t.name)
        if m:
            bias[int(m.group(1))] = dequantize(t.data, t.tensor_type).reshape(-1).astype(np.float32)
    ids, bounds, layers, ne = [], [0], None, 0
    for d in dumps:
        lg = read_logits(d)
        layers = sorted(lg)
        ne = lg[layers[0]].shape[1]
        T = lg[layers[0]].shape[0]
        a = np.zeros((T, len(layers), K), np.int16)
        for j, l in enumerate(layers):
            s = 1 / (1 + np.exp(-lg[l])) + bias[l] if l in bias else lg[l]
            a[:, j] = np.argsort(-s, axis=1)[:, :K]
        ids.append(a)
        bounds.append(bounds[-1] + T)
        print(d, T, 'tokens', len(layers), 'layers', flush=True)
    np.savez(out, ids=np.concatenate(ids), bounds=np.array(bounds), layers=np.array(layers), n_expert=ne)

main()
