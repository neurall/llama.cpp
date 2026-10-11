#!/usr/bin/env python3
"""alloc_sim.py trace.txt [SLOTS]: uniform vs greedy per-layer slot allocation (same total), LRU all-insert hit rates from
per-layer LRU stack distances (one pass gives the hit rate at every cache size)."""
import sys, re, collections
fn = sys.argv[1]; S = int(sys.argv[2]) if len(sys.argv) > 2 else 99
per = collections.defaultdict(list)
for l in open(fn):
    p = l.split(); per[int(re.search(r"(\d+)", p[0]).group(1))].append([int(x) for x in p[1:]])
layers = sorted(per); curves = {}
for il in layers:
    stack = []; hist = collections.Counter(); tot = 0
    for ids in per[il]:
        for e in ids:
            tot += 1
            if e in stack:
                d = stack.index(e); hist[d] += 1; stack.pop(d)
            stack.insert(0, e)
    # hits at size C = sum hist[d] for d < C
    cum = [0]; 
    for c in range(1, 289): cum.append(cum[-1] + hist[c-1])
    curves[il] = (cum, tot)
tot = sum(t for _, t in curves.values())
uni = sum(curves[il][0][S] for il in layers)
alloc = {il: 8 for il in layers}; budget = S*len(layers) - 8*len(layers)
import heapq
h = [(-(curves[il][0][alloc[il]+1]-curves[il][0][alloc[il]]), il) for il in layers]; heapq.heapify(h)
while budget > 0:
    g, il = heapq.heappop(h); alloc[il] += 1; budget -= 1
    if alloc[il] < 288: heapq.heappush(h, (-(curves[il][0][alloc[il]+1]-curves[il][0][alloc[il]]), il))
gre = sum(curves[il][0][alloc[il]] for il in layers)
print(f"LRU (all-insert, stack distance) {S} slots/layer: uniform {100*uni/tot:.1f}%  greedy per-layer {100*gre/tot:.1f}%")
print("greedy slots:", " ".join(f"{il}:{alloc[il]}" for il in layers))
print("per-layer uniform hit%:", " ".join(f"{il}:{100*curves[il][0][S]/curves[il][1]:.0f}" for il in layers))
