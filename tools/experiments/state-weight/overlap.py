#!/usr/bin/env python3
"""overlap.py [DIR]: hot-expert overlap between the phase-end states of tools/bench/state-phases (olmoe-<ARM>-end<A|B|C>.ini).
The counts are lifetime counts, so a phase's own use is the difference of two snapshots (B-A, C-B). Per layer the hot set of a count
vector is the fewest experts covering 75% of the picks (the engine's always-hot rule). Columns: mean Jaccard overlap of the hot sets over
the layers, the share of the top-16 experts the two vectors have in common, the hot set's share of all experts, and stale = the share of the
earlier vector's hot experts the later use does not need (preloaded from the state, never hit, only holding VRAM)."""
import os, re, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench", "state-phases")


def load(p):
    return {int(m.group(1)): [int(x) for x in m.group(2).split()] for m in re.finditer(r"^hot\.(\d+)\s*=\s*([\d ]+)$", open(p).read(), re.M)}


def hot(c, frac=0.75):
    tot, acc, s = sum(c), 0, set()
    for e in sorted(range(len(c)), key=lambda e: -c[e]):
        if tot == 0 or acc >= frac * tot:
            break
        s.add(e)
        acc += c[e]
    return s


def top(c, k=16):
    return set(sorted(range(len(c)), key=lambda e: -c[e])[:k])


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def sub(x, y):
    return {l: [max(0, a - b) for a, b in zip(x[l], y[l])] for l in x if l in y}


def cmp(x, y):
    ls = sorted(set(x) & set(y))
    n = len(x[ls[0]])
    return (sum(jac(hot(x[l]), hot(y[l])) for l in ls) / len(ls), sum(len(top(x[l]) & top(y[l])) / 16 for l in ls) / len(ls),
            sum(len(hot(x[l])) + len(hot(y[l])) for l in ls) / (2 * len(ls) * n),
            sum(len(hot(x[l]) - hot(y[l])) / max(1, len(hot(x[l]))) for l in ls) / len(ls))


print(f"{'arm':8} {'pair':26} {'hot Jaccard':>11} {'top16 shared':>12} {'hot share':>9} {'stale':>6} {'stale':>6}")
for arm in ("GW16", "GW4", "GW0", "FRESH16"):
    S = {k: load(f) for k in "ABC" if os.path.exists(f := os.path.join(D, f"olmoe-{arm}-end{k}.ini"))}
    if "A" not in S:
        continue
    pairs = []
    if "B" in S:
        pairs.append(("A use vs B use (poetry)", S["A"], sub(S["B"], S["A"])))
    if "B" in S and "C" in S:
        pairs += [("A use vs C use (python)", S["A"], sub(S["C"], S["B"])), ("B use vs C use", sub(S["B"], S["A"]), sub(S["C"], S["B"]))]
    for name, x, y in pairs:
        j, t, h, st = cmp(x, y)
        print(f"{arm:8} {name:26} {j:11.2f} {t:12.2f} {h:9.2f} {st:6.2f}")
