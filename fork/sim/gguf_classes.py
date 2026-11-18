#!/usr/bin/env python3
"""gguf_classes.py MODEL.gguf [...] [--json OUT]: the size classes a model's expert tensors fall into, from the GGUF headers alone (no weights read).

RAM pool (ggml/src/ggml-rampool-fork.cpp): a class per (slice bytes, tensor type); a slice is one expert's part of one tensor (up, gate, down each their own).
VRAM pool (src/llama-moecache-fork.cpp): a class per distinct triple (up + gate + down bytes of one expert of one layer).
Each class holds blocks of one size only, so an expert competes for memory only with experts of its own class: the table shows how the model's expert
bytes spread over the classes, i.e. what a per-class share fixes. A split model: give the first part, the others are found (-00001-of-000NN).
--json writes {layer: {"up": bytes, "gate": bytes, "down": bytes, "types": [...], "n": experts}} for the replays (ram_class_sim.py)."""
import sys, os, re, glob, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader

def parts(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    return [path] if not m else sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))

def layers_of(path):
    """layer -> {kind: (bytes per expert, type name, experts)} for kind in up, gate, down (gate_up: a fused pair)"""
    out = collections.defaultdict(dict)
    for p in parts(path):
        for t in GGUFReader(p).tensors:
            m = re.match(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$", t.name)
            if not m:
                continue
            n = int(t.shape[-1])
            out[int(m.group(1))][m.group(2)] = (int(t.n_bytes) // n, t.tensor_type.name, n)
    return dict(sorted(out.items()))

def report(path):
    L = layers_of(path)
    if not L:
        print(f"{os.path.basename(path)}: no expert tensors"); return L
    tot = sum(b * n for k in L.values() for b, _, n in k.values())
    nexp = sorted({n for k in L.values() for _, _, n in k.values()})
    print(f"\n{os.path.basename(path)}: {len(L)} expert layers, {'/'.join(map(str, nexp))} experts a layer, {tot / 2**30:.1f} GiB of experts")
    ram = collections.defaultdict(lambda: [0, 0, 0, set()])   # (bytes, type) -> slices (tensors), elements, bytes, layers
    for l, k in L.items():
        for kind, (b, ty, n) in k.items():
            r = ram[(b, ty)]; r[0] += 1; r[1] += n; r[2] += b * n; r[3].add(l)
    print(f"  RAM pool classes: {len(ram)}")
    print(f"    {'slice (MiB)':>11}  {'type':8}  {'tensors':>7}  {'layers':>6}  {'elements':>8}  {'GiB':>7}  {'share (%)':>9}")
    for (b, ty), (ns, ne, by, ls) in sorted(ram.items(), key=lambda x: -x[1][2]):
        print(f"    {b / 2**20:11.3f}  {ty:8}  {ns:7d}  {len(ls):6d}  {ne:8d}  {by / 2**30:7.2f}  {100 * by / tot:9.1f}")
    vr = collections.defaultdict(lambda: [0, 0, 0])   # triple bytes -> layers, experts, bytes
    for l, k in L.items():
        b = sum(x[0] for x in k.values()); n = max(x[2] for x in k.values())
        v = vr[b]; v[0] += 1; v[1] += n; v[2] += b * n
    print(f"  VRAM pool classes (expert = up + gate + down): {len(vr)}")
    print(f"    {'expert (MiB)':>12}  {'layers':>6}  {'experts':>8}  {'GiB':>7}  {'share (%)':>9}")
    for b, (nl, ne, by) in sorted(vr.items(), key=lambda x: -x[1][2]):
        print(f"    {b / 2**20:12.3f}  {nl:6d}  {ne:8d}  {by / 2**30:7.2f}  {100 * by / tot:9.1f}")
    big = max(vr); one = sum(big * ne for _, ne, _ in vr.values())
    print(f"  one block size for all ({big / 2**20:.2f} MiB): {100 * (one - tot) / one:.1f}% of the pool would be padding; a class each or a heap of exact sizes: 0%")
    return L

if __name__ == "__main__":
    args = sys.argv[1:]; js = None
    if "--json" in args:
        i = args.index("--json"); js = args[i + 1]; del args[i:i + 2]
    allm = {}
    for p in args:
        L = report(p)
        allm[os.path.basename(p)] = {str(l): {**{kind: v[0] for kind, v in k.items()}, "types": sorted({v[1] for v in k.values()}), "n": max(v[2] for v in k.values())} for l, k in L.items()}
    if js:
        json.dump(allm, open(js, "w"))
