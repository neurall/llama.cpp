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


print(f"{'arm':8} {'pair':26} {'hot Jaccard':>11} {'top16 shared':>12} {'hot share':>9} {'stale':>6}")
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


# stickiness: with the engine's score (window + 16 x lifetime / max, window = the picks of the last 64 tokens, top-8 of every token) and a cache of a third of the experts,
# which experts does a phase need that the score keeps out (ideal = the cache by the phase's own use, actual = the cache by the score with the lifetime counts of the state at the phase end)?
# blocked = share of the ideal slots held by a different expert; stale kept = share of the actual slots held by experts the phase did not use at all.
def sticky(use, life, slots_frac=1 / 3, window=64, top_k=8, k=16.0):
    ls = sorted(set(use) & set(life))
    blk = stale = 0.0
    for l in ls:
        n = len(use[l]); slots = max(1, round(n * slots_frac)); tu = sum(use[l]) or 1; mx = max(life[l]) or 1
        win = [window * top_k * u / tu for u in use[l]]
        ideal = set(sorted(range(n), key=lambda e: -win[e])[:slots])
        actual = set(sorted(range(n), key=lambda e: -(win[e] + k * life[l][e] / mx))[:slots])
        blk += len(ideal - actual) / slots
        stale += sum(1 for e in actual if use[l][e] == 0) / slots
    return blk / len(ls), stale / len(ls)


print(f"\n{'arm':8} {'phase (state kept from before)':32} {'blocked':>8} {'stale kept':>10}")
for arm in ("GW16", "GW4", "GW0"):
    S = {kk: load(f) for kk in "ABC" if os.path.exists(f := os.path.join(D, f"olmoe-{arm}-end{kk}.ini"))}
    for name, a, b in (("B poetry after A", "B", "A"), ("C python after B", "C", "B")):
        if a in S and b in S:
            bl, st = sticky(sub(S[a], S[b]), S[a], k={"GW16": 16.0, "GW4": 4.0, "GW0": 0.0}[arm])
            print(f"{arm:8} {name:32} {bl:8.2f} {st:10.2f}")
