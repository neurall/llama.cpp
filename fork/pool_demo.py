#!/usr/bin/env python3
"""pool_demo.py DIR: synthetic pool-snaps / pool-events files in the exact format of sj_snap / sj_event (llama-moecache-fork.cpp): livelist.py writes them when no real trace exists, so the page can be debugged without a run"""
import json, os, random, sys, time
d = sys.argv[1]
os.makedirs(d, exist_ok=True)
pid = 999999
t0 = int(time.monotonic() * 1e6)
random.seed(1)
MiB = 1 << 20

def card(name, label, nblk, B, kv_cells):
    lend = 119 * MiB if name == "CUDA0" else 646 * MiB
    base = ((lend + B - 1) // B) * B
    segs = [{"c": 0, "base": 0, "n": nblk, "B": B}]
    kv_row = 20736
    kv_start = base + 4 * B
    kv_end = kv_start + kv_cells * kv_row
    b = []
    for k in range(nblk):
        lo = k * B
        if lo < lend: b += [-2, -1, -1, 0, 0]
        elif kv_start <= lo < kv_end + B: b += [-3, -1, -1, 0, 0]
        elif random.random() < 0.03: b += [-1, -1, -1, 0, 0]
        else:
            il = random.randrange(43); e = random.randrange(256); rc = int(random.expovariate(1 / 20)); b += [il, il, e, rc, rc * 37]
    segs[0]["b"] = b
    regions = [{"k": "graph", "c": -1, "lo": 0, "hi": lend - 30 * MiB}, {"k": "scratch", "c": -1, "lo": lend - 30 * MiB, "hi": lend - 10 * MiB},
               {"k": "staging", "c": -1, "lo": lend - 10 * MiB, "hi": lend}, {"k": "zone", "c": -1, "lo": kv_start, "hi": kv_end},
               {"k": "blocks", "c": 0, "lo": 0, "hi": nblk * B}]
    return {"name": name, "label": label, "bytes": nblk * B + B, "lend": lend, "mirror": 0, "scr_last": 20 * MiB, "scr_hwm": 640 * MiB, "tok_bytes": 190000,
            "stage": 2, "slot_b": 5 * MiB, "cells": kv_cells // 2, "kv_h": kv_cells, "kv_start": kv_start, "kv_end": kv_end, "kv_row": kv_row, "kv_fixed": 0,
            "regions": regions, "classes": segs}

with open(os.path.join(d, f"pool-snaps-{pid}.jsonl"), "w") as f, open(os.path.join(d, f"pool-events-{pid}.jsonl"), "w") as g:
    n = 0
    for s in range(20):
        t = t0 + s * 250000
        cs = [card("CUDA1", "pcuda1x16", 2654, 8 * MiB, 4096), card("CUDA0", "cuda0x4", 2413, 8 * MiB, 4096)]
        f.write(json.dumps({"t": t, "seq": s + 1, "pid": pid, "cards": cs, "bufs": []}, separators=(",", ":")) + "\n")
        for c in cs:
            for _ in range(5):
                k = random.randrange(200, 2000); n += 1
                g.write(json.dumps({"t": t + 1000 * n % 200000, "n": n, "card": c["name"], "kind": random.choice(["up-start", "up-done", "keep-hot"]),
                                    "lo": k * 8 * MiB, "hi": (k + 1) * 8 * MiB, "who": f"L{k % 43} e{k % 256} block {k} class 0"}, separators=(",", ":")) + "\n")
            n += 1
            g.write(json.dumps({"t": t + 5000, "n": n, "card": c["name"], "kind": "lend", "lo": 0, "hi": c["lend"], "who": "graph 1 tokens"}, separators=(",", ":")) + "\n")
        # one write that crosses its block's end: the page must flash it red
        n += 1
        g.write(json.dumps({"t": t + 9000, "n": n, "card": "CUDA0", "kind": "up-done", "lo": 1500 * 8 * MiB + 4 * MiB, "hi": 1501 * 8 * MiB + 4 * MiB, "who": "crossing test"}, separators=(",", ":")) + "\n")
print("ok", pid)
