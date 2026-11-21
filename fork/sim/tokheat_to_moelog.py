#!/usr/bin/env python3
"""tokheat_to_moelog.py TRACE_STEM [--prompt-tokens P --ubatch U] [-o moelog.txt] | --selftest

Turns a `--moe tokheat=DIR,tokheat-trace=1` request trace (DIR/<model>/trace/N.json + N.ids, any backend, GPU runs too) into the
GGML_MOE_LOG text the replay tools read ("blk.<L>.ffn_gate_exps.weight id id ...", one line per token and layer).
The trace keeps the picks per layer in token order, 16 int16 per token, -1 padded; it has no pass boundaries, so the order is rebuilt:
the first P tokens run as prompt passes of U tokens (every layer of a pass, layer by layer, then the next pass), the rest as one-token passes.
"""
import argparse, json, struct, sys, os, tempfile


def load(stem):
    meta = json.load(open(stem + ".json"))
    raw = open(stem + ".ids", "rb").read()
    vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
    layers, off = {}, 0
    for il, rows, _ in meta["layers"]:
        layers[il] = [[v for v in vals[off + r * 16: off + r * 16 + 16] if v >= 0] for r in range(rows)]
        off += rows * 16
    return layers


def lines(layers, prompt_tokens=0, ubatch=512):
    n = max((len(r) for r in layers.values()), default=0)
    t = 0
    while t < n:
        step = max(1, min(ubatch, prompt_tokens - t)) if t < prompt_tokens else 1
        for il in sorted(layers):
            for r in range(t, min(t + step, len(layers[il]))):
                yield "blk.%d.ffn_gate_exps.weight %s" % (il, " ".join(map(str, layers[il][r])))
        t += step


def selftest():
    d = tempfile.mkdtemp(prefix="tokheat_")
    stem = os.path.join(d, "0")
    # 2 layers, 3 tokens, k = 2; layer 1 has only 2 rows (a one-token pass without a named top-k node)
    rows = {0: [[1, 2], [3, 4], [5, 6]], 1: [[7, 8], [9, 10]]}
    ids = []
    for il in (0, 1):
        for r in rows[il]:
            ids += r + [-1] * 14
    open(stem + ".ids", "wb").write(struct.pack("<%dh" % len(ids), *ids))
    json.dump({"n_embd": 4, "layers": [[0, 3, 0], [1, 2, 0]]}, open(stem + ".json", "w"))
    assert load(stem) == rows
    # all three tokens as one prompt pass: layer 0 rows 0-2, then layer 1 rows 0-1
    got = list(lines(load(stem), prompt_tokens=3, ubatch=512))
    assert got == ["blk.0.ffn_gate_exps.weight 1 2", "blk.0.ffn_gate_exps.weight 3 4", "blk.0.ffn_gate_exps.weight 5 6",
                   "blk.1.ffn_gate_exps.weight 7 8", "blk.1.ffn_gate_exps.weight 9 10"], got
    # ubatch 2: pass 1 = tokens 0-1 over both layers, then token 2 as a one-token pass
    got = list(lines(load(stem), prompt_tokens=3, ubatch=2))
    assert got[:2] == ["blk.0.ffn_gate_exps.weight 1 2", "blk.0.ffn_gate_exps.weight 3 4"] and got[2:4] == ["blk.1.ffn_gate_exps.weight 7 8", "blk.1.ffn_gate_exps.weight 9 10"], got
    assert got[4] == "blk.0.ffn_gate_exps.weight 5 6" and len(got) == 5, got
    # decode only: token by token
    got = list(lines(load(stem)))
    assert got[0] == "blk.0.ffn_gate_exps.weight 1 2" and got[1] == "blk.1.ffn_gate_exps.weight 7 8" and got[2] == "blk.0.ffn_gate_exps.weight 3 4", got
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?"); ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--ubatch", type=int, default=512); ap.add_argument("-o"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    out = open(a.o, "w") if a.o else sys.stdout
    for l in lines(load(a.stem), a.prompt_tokens, a.ubatch):
        out.write(l + "\n")
