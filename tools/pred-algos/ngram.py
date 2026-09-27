#!/usr/bin/env python3
"""Algo 4: token n-gram hash -> every layer's experts, known before the forward pass starts.

Key = the last n token ids (current token included). Value = per-layer counts of the experts really selected
after that key (optionally decayed). Back-off: the longest order seen before wins. No gradients: one sighting
is enough. Unmatched tokens fall back to the previous token's picks at the same layer (reported separately).
"""
import time

import numpy as np

from common import cli, load, overlap, report, parse


def main():
    ap = cli(__doc__)
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--decay", type=float, default=1.0, help="count decay per sighting of the same key (1: none)")
    a = parse(ap)
    d = load(a.data)
    tokens, real, K, E = d["tokens"].numpy(), d["real"].numpy(), d["k"], d["n_expert"]
    NL, T, _ = real.shape
    orders = range(a.max_n, a.min_n - 1, -1)
    tables = {n: {} for n in orders}
    hits = np.full((NL, T), np.nan)       # n-gram only
    hits_fb = np.empty((NL, T))           # with previous-token fallback
    used = np.zeros(a.max_n + 1)          # tokens predicted by each order
    used_hit = np.zeros(a.max_n + 1)
    t0 = time.time()
    for t in range(T):
        keys = {n: tuple(tokens[t - n + 1:t + 1]) for n in orders if t - n + 1 >= 0}
        for n in orders:
            c = tables[n].get(keys.get(n))
            if c is not None:
                pred = np.argpartition(-c, K, axis=1)[:, :K]
                hits[:, t] = overlap(pred, real[:, t])
                used[n] += 1
                used_hit[n] += hits[:, t].mean()
                break
        hits_fb[:, t] = hits[:, t] if np.isfinite(hits[0, t]) else (overlap(real[:, t - 1], real[:, t]) if t else 0)
        onehot = np.zeros((NL, E), np.float32)
        np.put_along_axis(onehot, real[:, t], 1.0, axis=1)
        for n, key in keys.items():
            c = tables[n].get(key)
            tables[n][key] = onehot if c is None else c * a.decay + onehot
    extra = f"  ({time.time() - t0:.1f} s, {sum(len(v) for v in tables.values())} entries)"
    report(f"ngram {a.min_n}..{a.max_n} matched only", hits, d["bounds"], extra, d["gen"])
    report(f"ngram {a.min_n}..{a.max_n} + prev-token fallback", hits_fb, d["bounds"], gen=d["gen"])
    print("  per order: " + "  ".join(f"n={n}: {used[n] / T:.2f} of tokens, hit {used_hit[n] / max(used[n], 1):.3f}"
                                      for n in orders))


if __name__ == "__main__":
    main()
