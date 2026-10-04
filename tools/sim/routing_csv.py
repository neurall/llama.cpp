#!/usr/bin/env python3
"""Routing traces of --moe log=x -> npz for the replay tools (route_evict.py, cache_sim.py): the real decode routing of a run.

usage: routing_csv.py <model>.routing out.npz [n_expert]
       routing_csv.py routing.csv out.npz [n_expert] [model-substring]     (the older text format: model,pos,layer,ids)
       routing_csv.py split routing.csv DIR                                 (the older text file -> DIR/<model>.routing)
<model>.routing, little endian: "MOER", u16 version (1), u16 layers, u16 k, u16 layer numbers; then per sampled generated token an
i32 pos and layers x k u16 expert ids (0xffff = none). Prefill tokens are not logged. A run starts where pos goes down (every process
appends to the same file); each run becomes one sample. out.npz as route_trace.py writes it: ids [T, NL, K] int16, bounds [n_samples + 1]
(token offsets of the runs), layers [NL] (model layer numbers), n_expert (given, else the largest id + 1).
"""
import csv, os, re, struct, sys
import numpy as np


def read_bin(path):
    b = open(path, "rb").read()
    assert b[:4] == b"MOER", path
    ver, nl, k = struct.unpack_from("<HHH", b, 4)
    layers = list(struct.unpack_from(f"<{nl}H", b, 10))
    off = 10 + 2 * nl
    rec = np.dtype([("pos", "<i4"), ("ids", "<u2", (nl, k))])
    r = np.frombuffer(b, rec, offset=off, count=(len(b) - off) // rec.itemsize)
    ids = r["ids"].astype(np.int32)
    ids[ids == 0xffff] = -1
    bounds = [0] + [i for i in range(1, len(r)) if r["pos"][i] < r["pos"][i - 1]] + [len(r)]
    return ids.astype(np.int16), np.array(bounds), np.array(layers)


def read_csv(path, only=""):
    runs, cur, last_pos, layers = [], {}, None, set()
    rows_ = [r for r in csv.DictReader(open(path)) if not only or only in r.get("model", "")]
    from collections import Counter
    kk = Counter(len(r["ids"].split()) for r in rows_).most_common(1)[0][0]   # the real k: older traces also hold warm-up rows with n_tokens x k ids
    for r in rows_:
        if len(r["ids"].split()) != kk:
            continue
        pos, il = int(r["pos"]), int(r["layer"])
        if last_pos is not None and pos < last_pos and cur:   # a new run
            runs.append(cur); cur = {}
        last_pos = pos
        cur.setdefault(pos, {})[il] = [int(x) for x in r["ids"].split()]
        layers.add(il)
    if cur:
        runs.append(cur)
    layers = sorted(layers)
    k = max(len(v) for run in runs for tok in run.values() for v in tok.values())
    rows, bounds, poss = [], [0], []
    for run in runs:
        for pos in sorted(run):
            tok = run[pos]
            if len(tok) != len(layers):   # a token cut off at the end of a run
                continue
            rows.append([tok[l] + [-1] * (k - len(tok[l])) for l in layers]); poss.append(pos)
        bounds.append(len(rows))
    return np.array(rows, np.int16), np.array(bounds), np.array(layers), np.array(poss)


def write_bin(path, ids, layers, poss):
    nl, k = ids.shape[1], ids.shape[2]
    with open(path, "wb") as f:
        f.write(b"MOER" + struct.pack("<HHH", 1, nl, k) + struct.pack(f"<{nl}H", *layers))
        rec = np.zeros(len(ids), np.dtype([("pos", "<i4"), ("ids", "<u2", (nl, k))]))
        rec["pos"] = poss
        rec["ids"] = np.where(ids < 0, 0xffff, ids).astype(np.uint16)
        f.write(rec.tobytes())


def main():
    if sys.argv[1] == "split":
        src, d = sys.argv[2], sys.argv[3]
        models = sorted({r["model"] for r in csv.DictReader(open(src))})
        for m in models:
            name = re.sub(r"[^A-Za-z0-9._-]", "_", re.sub(r"-0000\d-of-0000\d", "", m.split(" ")[0]).removesuffix(".gguf"))
            ids, bounds, layers, poss = read_csv(src, m)
            out = os.path.join(d, f"{name}.routing")
            write_bin(out, ids, layers, poss)
            print(f"{out}: {len(bounds) - 1} runs, {len(ids)} tokens, {os.path.getsize(out) / 1e6:.2f} MB")
        return
    src, out = sys.argv[1], sys.argv[2]
    n_expert = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0
    if src.endswith((".routing", ".bin")):
        ids, bounds, layers = read_bin(src)
    else:
        ids, bounds, layers, _ = read_csv(src, sys.argv[4] if len(sys.argv) > 4 else "")
    ne = n_expert or int(ids.max()) + 1
    np.savez(out, ids=ids, bounds=bounds, layers=layers, n_expert=ne)
    print(f"{out}: {len(bounds) - 1} runs, {ids.shape[0]} tokens, {len(layers)} layers, top-{ids.shape[2]}, {ne} experts")


if __name__ == "__main__":
    main()
