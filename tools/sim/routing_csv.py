#!/usr/bin/env python3
"""routing.csv (--moe log=x) -> npz for the replay tools (route_evict.py, cache_sim.py): the real decode routing of a run.

usage: routing_csv.py routing_<model>.csv out.npz [n_expert] [model-substring]
routing_<model>.csv: pos,layer,ids (older routing.csv: model,pos,layer,ids; model-substring picks one model) (one line per sampled generated token and MoE layer; prefill tokens are not logged). A run starts where
pos goes down (a new process appends to the same file); each run becomes one sample. out.npz as route_trace.py writes it: ids [T, NL, K]
int16, bounds [n_samples + 1] (token offsets of the runs), layers [NL] (model layer numbers), n_expert (given, else the largest id + 1).
"""
import csv, sys
import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    n_expert = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0
    only = sys.argv[4] if len(sys.argv) > 4 else ""
    runs, cur, last_pos, layers = [], {}, None, set()
    for r in csv.DictReader(open(src)):
        if only and only not in r.get("model", ""):
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
    rows, bounds = [], [0]
    for run in runs:
        for pos in sorted(run):
            tok = run[pos]
            if len(tok) != len(layers):   # a token cut off at the end of a run
                continue
            rows.append([tok[l] + [-1] * (k - len(tok[l])) for l in layers])
        bounds.append(len(rows))
    ids = np.array(rows, np.int16)
    ne = n_expert or int(ids.max()) + 1
    np.savez(out, ids=ids, bounds=np.array(bounds), layers=np.array(layers), n_expert=ne)
    print(f"{out}: {len(runs)} runs, {ids.shape[0]} tokens, {len(layers)} layers, top-{k}, {ne} experts")


if __name__ == "__main__":
    main()
