#!/usr/bin/env python3
"""Pool state per card at the end of a run, from the pool write ledger (--moe ledger=1): what each buffer holds when memory is tight.
usage: pool_state.py LEDGER.csv[.gz] [--peak]   one line per card: pool size, lend (max / last), KV zone (last / max), expert blocks per class, zero blocks, uploads
(lend = compute graph + CUDA scratch + prefetch staging borrowed from the front of the pool)"""
import sys, gzip, collections
def summarize(path):
    op = gzip.open if path.endswith(".gz") else open
    c = collections.defaultdict(lambda: dict(pool=0, lend_max=0, lend_last=0, zone_last=0, zone_max=0, blocks={}, zero={}, up=0, moves=0, overlaps=0))
    for line in op(path, "rt", errors="replace"):
        f = line.rstrip("\n").split(",", 6)
        if len(f) < 7 or not f[0].isdigit(): continue
        card, kind, lo, hi, who = f[1], f[2], int(f[3]), int(f[4]), f[6]
        d = c[card]; d["pool"] = max(d["pool"], hi)
        if kind == "lend": d["lend_last"] = hi; d["lend_max"] = max(d["lend_max"], hi)
        elif kind == "release": d["lend_last"] = lo
        elif kind == "kv-zone": d["zone_last"] = hi - lo; d["zone_max"] = max(d["zone_max"], hi - lo)
        elif kind == "kv-move": d["moves"] += 1
        elif kind == "up-start": d["up"] += 1
        elif kind == "OVERLAP": d["overlaps"] += 1
        elif kind == "map":
            w = who.split()
            if "zone" in w: d["zone_last"] = hi - lo; d["zone_max"] = max(d["zone_max"], hi - lo)
            elif "blocks" in w: d["blocks"][w[-1]] = hi - lo
            elif "zero" in w: d["zero"][w[-1]] = hi - lo
    mib = lambda b: f"{b / 1048576:.0f}"
    out = []
    for card, d in sorted(c.items()):
        out.append(f"{card}: pool {mib(d['pool'])} MiB | lend max {mib(d['lend_max'])} last {mib(d['lend_last'])} | KV zone last {mib(d['zone_last'])} max {mib(d['zone_max'])} | "
                   f"expert blocks {sum(d['blocks'].values()) / 1048576:.0f} MiB in {len(d['blocks'])} classes | uploads {d['up']}, kv-moves {d['moves']}, overlaps {d['overlaps']}")
    return out
if __name__ == "__main__":
    print("\n".join(summarize(sys.argv[1])))
