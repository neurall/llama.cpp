#!/usr/bin/env python3
"""ram_class_sim.py MODEL.gguf --hot globalhot.ini [--best] [--frac 0.1,0.2,...] [--vram 0.2] [--trace moelog.txt ...]

What the RAM pool's fixed class shares cost against one heat order over all sizes (task #327, user 2026-11-18: "how unbalanced the RAM classes are ... observe
from gguf"). Sizes come from the GGUF header (gguf_classes.py), heat from a stored globalhot.ini (hot.<layer> = uses per expert) or from routing traces.

The RAM pool today (ggml/src/ggml-rampool-fork.cpp): a class per (slice bytes, type); at open every class gets the same share of its elements
(pool bytes / expert bytes); a class evicts only inside itself; chunks move between classes only when the pool's size changes.

Static estimate (stored counts, a fixed popularity): the hottest experts by count sit in VRAM (--vram: share of the expert bytes), the RAM tier holds
  fixed   per class its hottest share x elements (today's split with the best eviction a class can have)
  chunks  the same exact-size blocks, 1 GiB chunks dealt to the class whose next chunk holds the most use per byte (classes kept, shares by pressure)
  heap    one order over all sizes: the most used first until the bytes are full (no classes)
and the cost is the share of the demanded bytes (uses x slice bytes, VRAM hits left out) that the SSD serves.

Replay (--trace: GGML_MOE_LOG lines "<tensor> id id ..."): LRU with the same three layouts (chunks: every 64 tokens one chunk goes from the class with the least
file bytes read per byte held to the one with the most, the engine's own pressure), cost = file bytes read per token after the first quarter of the trace."""
import sys, os, re, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_classes import layers_of

def load_hot(path, best):
    hot, sec = {}, ""
    for l in open(path, errors="replace"):
        l = l.strip()
        if l.startswith("["):
            sec = l.strip("[]"); continue
        if (sec == "best") != best:
            continue
        m = re.match(r"hot\.(\d+)=(.*)", l)
        if m:
            hot[int(m.group(1))] = np.array(m.group(2).split(), dtype=np.float64)
    return hot

def elements(L, heat):
    """one row per slice: class id, bytes, heat, expert id (layer * 4096 + e); classes: list of (bytes, type)"""
    cls_of, classes, rows = {}, [], []
    for l, kinds in L.items():
        h = heat.get(l)
        if h is None:
            continue
        for kind, (b, ty, n) in kinds.items():
            c = cls_of.setdefault((b, ty), len(cls_of))
            if c == len(classes):
                classes.append((b, ty))
            for e in range(min(n, len(h))):
                rows.append((c, b, h[e], l * 4096 + e))
    a = np.array(rows, dtype=np.float64)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3].astype(np.int64), classes

def static(L, heat, fracs, vram):
    cls, by, h, ex, classes = elements(L, heat)
    total = by.sum()
    # VRAM: whole experts, the most used first
    order = np.argsort(-h, kind="stable")
    in_vram = np.zeros(len(h), bool)
    if vram > 0:
        exb = collections.defaultdict(float); exh = {}
        for i in range(len(h)):
            exb[ex[i]] += by[i]; exh[ex[i]] = h[i]
        got = 0.0; keep = set()
        for e in sorted(exh, key=lambda e: -exh[e]):
            if got + exb[e] > vram * total:
                break
            got += exb[e]; keep.add(e)
        in_vram = np.array([e in keep for e in ex])
    dem = (h * by)[~in_vram].sum()                      # demanded bytes the RAM tier or the SSD serves
    nc = len(classes)
    idx = [np.array([i for i in order if cls[i] == c and not in_vram[i]]) for c in range(nc)]   # per class, hottest first, VRAM's left out
    elems = np.array([(cls == c).sum() for c in range(nc)])
    print(f"  classes: " + "; ".join(f"{classes[c][0] / 2**20:.2f} MiB {classes[c][1]} {100 * by[cls == c].sum() / total:.0f}% of bytes, "
          f"{100 * (h * by)[(cls == c) & ~in_vram].sum() / dem:.0f}% of demand" for c in range(nc)))
    print(f"  {'pool (% of expert bytes)':>24}  {'SSD share fixed (%)':>19}  {'chunks (%)':>10}  {'heap (%)':>8}  {'heap vs fixed (%)':>17}  shares fixed -> heap (% of each class's elements)")
    out = []
    for f in fracs:
        B = f * total
        # fixed: every class the same share of its elements
        n_fix = [int(min(len(idx[c]), elems[c] * f)) for c in range(nc)]
        miss_fix = sum((h * by)[idx[c][n_fix[c]:]].sum() for c in range(nc))
        # heap: one order
        rest = np.array([i for i in order if not in_vram[i]])
        cum = np.cumsum(by[rest]); k = int(np.searchsorted(cum, B, side="right"))
        miss_heap = (h * by)[rest[k:]].sum()
        n_heap = [int((cls[rest[:k]] == c).sum()) for c in range(nc)]
        # chunks: 1 GiB at a time to the class whose next chunk holds the most use per byte
        n_ch = [0] * nc; left = B
        while True:
            best, gain = -1, 0.0
            for c in range(nc):
                per = max(1, int(2**30 // classes[c][0])); nxt = idx[c][n_ch[c]:n_ch[c] + min(per, int(left // classes[c][0]))]   # the last chunk as big as what is left
                if len(nxt) == 0:
                    continue
                g = h[nxt].sum() / len(nxt)
                if g > gain:
                    best, gain = c, g
            if best < 0:
                break
            per = max(1, int(2**30 // classes[best][0])); k2 = len(idx[best][n_ch[best]:n_ch[best] + min(per, int(left // classes[best][0]))])
            n_ch[best] += k2; left -= k2 * classes[best][0]
        miss_ch = sum((h * by)[idx[c][n_ch[c]:]].sum() for c in range(nc))
        sh = "  ".join(f"{100 * n_fix[c] / elems[c]:.0f}->{100 * n_heap[c] / elems[c]:.0f}" for c in range(nc))
        gain = 100 * (miss_fix - miss_heap) / miss_fix if miss_fix > 0 else 0.0
        print(f"  {100 * f:24.0f}  {100 * miss_fix / dem:19.2f}  {100 * miss_ch / dem:10.2f}  {100 * miss_heap / dem:8.2f}  {gain:17.1f}  {sh}")
        out.append((f, miss_fix / dem, miss_ch / dem, miss_heap / dem))
    return out

def load_trace(files):
    toks = []   # per graph (a decode token or a prompt batch): list of (layer, ids)
    for fn in files:
        cur, last = [], -1
        for l in open(fn):
            p = l.split()
            m = re.search(r"blk\.(\d+)\.", p[0]) if p else None
            if not m:
                continue
            layer = int(m.group(1))
            if layer < last:   # back at an earlier layer: the next graph (a decode token; a prompt batch logs one line per token and layer, layer by layer)
                toks.append(cur); cur = []
            last = layer; cur.append((layer, [int(x) for x in p[1:]]))
        if cur:
            toks.append(cur)
    return toks

def replay(L, toks, fracs, cycles):
    kinds = {l: list(k.items()) for l, k in L.items()}
    cls_of, classes = {}, []
    for l, k in L.items():
        for kind, (b, ty, n) in k.items():
            if (b, ty) not in cls_of:
                cls_of[(b, ty)] = len(classes); classes.append((b, ty))
    nc = len(classes); size = [c[0] for c in classes]
    elems = [0] * nc
    for l, k in L.items():
        for kind, (b, ty, n) in k.items():
            elems[cls_of[(b, ty)]] += n
    total = sum(elems[c] * size[c] for c in range(nc))
    stream = toks * cycles; warm = len(stream) // 4
    print(f"  replay: {len(toks)} tokens x {cycles}, LRU; file bytes read per token after token {warm}")
    print(f"  {'pool (% of expert bytes)':>24}  {'fixed (MiB/token)':>17}  {'chunks (MiB/token)':>18}  {'heap (MiB/token)':>16}  {'chunks vs fixed (%)':>19}  {'heap vs fixed (%)':>17}")
    for f in fracs:
        res = []
        for mode in ("fixed", "chunks", "heap"):
            cap = [int(elems[c] * f) for c in range(nc)]            # blocks per class
            lru = [collections.OrderedDict() for _ in range(nc if mode != "heap" else 1)]
            used = 0.0; B = f * total; read = 0.0; cold = [0.0] * nc
            for t, tok in enumerate(stream):
                for layer, ids in tok:
                    if layer not in kinds:
                        continue
                    for kind, (b, ty, n) in kinds[layer]:
                        c = cls_of[(b, ty)]; d = lru[0 if mode == "heap" else c]
                        for e in ids:
                            key = (layer, kind, e)
                            if key in d:
                                d.move_to_end(key); continue
                            if t >= warm:
                                read += b
                            cold[c] += b
                            if mode == "heap":
                                d[key] = b; used += b
                                while used > B:
                                    _, vb = d.popitem(last=False); used -= vb
                            else:
                                d[key] = b
                                while len(d) > cap[c]:
                                    d.popitem(last=False)
                if mode == "chunks" and t % 64 == 63:
                    pr = [cold[c] / max(size[c], cap[c] * size[c]) for c in range(nc)]
                    hi = max(range(nc), key=lambda c: pr[c] if cap[c] < elems[c] else -1.0); lo = min(range(nc), key=lambda c: pr[c] if cap[c] > 0 else 1e300)
                    if hi != lo and pr[hi] > pr[lo]:
                        chunk = min(2**30, cap[lo] * size[lo])
                        k_lo = int(chunk // size[lo]); k_hi = int(chunk // size[hi])
                        cap[lo] -= k_lo; cap[hi] = min(elems[hi], cap[hi] + k_hi)
                        while len(lru[lo]) > cap[lo]:
                            lru[lo].popitem(last=False)
                    cold = [0.0] * nc
            res.append(read / max(1, len(stream) - warm) / 2**20)
        g = lambda x: 100 * (res[0] - x) / res[0] if res[0] > 0 else 0.0
        print(f"  {100 * f:24.0f}  {res[0]:17.2f}  {res[1]:18.2f}  {res[2]:16.2f}  {g(res[1]):19.1f}  {g(res[2]):17.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf"); ap.add_argument("--hot"); ap.add_argument("--best", action="store_true", help="the [best] section of the hot file")
    ap.add_argument("--frac", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"); ap.add_argument("--vram", type=float, default=0.2)
    ap.add_argument("--trace", nargs="*"); ap.add_argument("--cycles", type=int, default=3)
    a = ap.parse_args()
    L = layers_of(a.gguf); fr = [float(x) for x in a.frac.split(",")]
    print(os.path.basename(a.gguf))
    if a.hot:
        heat = load_hot(a.hot, a.best)
        if not set(heat) & set(L):   # the file numbers the expert layers 1..n
            heat = {l: heat[i + 1] for i, l in enumerate(sorted(L)) if i + 1 in heat}
        print(f"  stored counts: {len(heat)} layers, {sum(v.sum() for v in heat.values()):.0f} uses; VRAM holds the hottest {100 * a.vram:.0f}% of the bytes")
        static(L, heat, fr, a.vram)
    if a.trace:
        replay(L, load_trace(a.trace), fr, a.cycles)
