#!/usr/bin/env python3
"""tier_sim.py: how big a pinned host tier must be for decode, replayed on real routing traces (--moe log=x).

usage: tier_sim.py --model-gb 137.1 [--vram 0.35] [--pin 0.1,0.2,...] [--base-ms 50] <model>.routing [...]

Three tiers of expert copies (one expert = its up, gate and down slices of one layer):
  1. VRAM cache   the fork's GPU expert pool, a global LRU over (layer, expert) of --vram x all experts
  2. pinned RAM   a global LRU of --pin x all experts, fed by the VRAM misses: a VRAM miss found here uploads at pinned speed
  3. the rest     page cache (pageable upload) or SSD (read, then upload)
Today the whole model is pinned (pin = 1.0): every VRAM miss uploads at pinned speed. The question is how small tier 2 can be
before decode slows, because during decode most pinned bytes are never read (prompt passes stream every expert through a ring
instead, which this does not model).

Cost per token = what today's decode pays (--base-ms, every miss from pinned memory) + for each miss served by tier 3 the extra
time of the slower source: pageable (--pageable GB/s instead of --link GB/s), or SSD (--ssd GB/s + --ssd-lat ms per read).
Uploads overlap with compute in the engine, so this is an upper bound on the slowdown (a miss costs its full transfer time)."""
import argparse, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routing_csv import read_bin   # the trace reader of the other replay tools

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", help=".routing traces of ONE model, replayed in the given order (topic switches included)")
ap.add_argument("--model-gb", type=float, required=True, help="model file size in GB (1e9): the expert size is taken from it")
ap.add_argument("--expert-share", type=float, default=0.95, help="share of the file that is expert tensors (MoE models: ~0.93-0.97)")
ap.add_argument("--vram", type=float, default=0.35, help="VRAM tier as a share of all experts (2x 24 GB cards on GLM: ~0.35)")
ap.add_argument("--pin", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.65,0.8,1.0", help="pinned tier sizes as shares of all experts")
ap.add_argument("--link", type=float, default=24.0, help="upload GB/s from pinned memory (x16 link)")
ap.add_argument("--pageable", type=float, default=12.4, help="upload GB/s from pageable memory (measured 12.4 on the x16 link)")
ap.add_argument("--ssd", type=float, default=5.0, help="SSD read GB/s for an expert's slices")
ap.add_argument("--ssd-lat", type=float, default=0.1, help="ms per SSD read (one expert)")
ap.add_argument("--base-ms", type=float, default=50.0, help="decode ms per token today (every miss from pinned memory)")
ap.add_argument("--warm", type=int, default=64, help="tokens before counting starts (the caches fill)")
a = ap.parse_args()

seqs, nl, ne = [], 0, 0
for fn in a.files:
    ids, bounds, layers = read_bin(fn)
    seqs.append(ids)
    nl = max(nl, ids.shape[1]); ne = max(ne, int(ids.max()) + 1)
n_all = nl * ne
expert_mb = a.model_gb * 1e9 * a.expert_share / n_all / 1e6
tokens = sum(s.shape[0] for s in seqs)
print(f"{len(a.files)} traces, {tokens} decode tokens, {nl} MoE layers x {ne} experts = {n_all} experts of {expert_mb:.2f} MB (model {a.model_gb:.1f} GB)")
print(f"VRAM tier {a.vram:.2f} of the experts; uploads: pinned {a.link} GB/s, pageable {a.pageable} GB/s, SSD {a.ssd} GB/s + {a.ssd_lat} ms; today {a.base_ms:.0f} ms/token")


def replay(pin_share):
    """one pass over all traces: counts of VRAM hits, tier-2 hits and tier-3 reads after the warm-up"""
    c1, c2 = int(a.vram * n_all), int(pin_share * n_all)
    vram, pinned = collections.OrderedDict(), collections.OrderedDict()
    t = h1 = h2 = m3 = 0
    for s in seqs:
        for tok in s:
            count = t >= a.warm
            for il in range(tok.shape[0]):
                for e in tok[il]:
                    if e < 0: continue
                    key = (il, int(e))
                    if key in vram:
                        vram.move_to_end(key); h1 += count; continue
                    if key in pinned:                      # a VRAM miss served from pinned memory
                        pinned.move_to_end(key); h2 += count
                    else:                                  # from the page cache or the SSD; it enters the pinned tier
                        m3 += count
                        if c2 > 0:
                            if len(pinned) >= c2: pinned.popitem(last=False)
                            pinned[key] = 1
                    if len(vram) >= c1: vram.popitem(last=False)
                    vram[key] = 1
            t += 1
    return h1, h2, m3, t - a.warm


rows = []
for p in [float(x) for x in a.pin.split(",")]:
    h1, h2, m3, n = replay(p)
    uses = h1 + h2 + m3
    miss_tok = (h2 + m3) / n                      # uploads per token (VRAM misses)
    slow_tok = m3 / n                             # of them from the slow tier
    mb = expert_mb
    pin_ms = miss_tok * mb / a.link               # what today's decode spends on uploads (MB / (GB/s) = ms)
    extra_page = slow_tok * mb * (1 / a.pageable - 1 / a.link)
    extra_ssd = slow_tok * (mb / a.ssd + a.ssd_lat - mb / a.link)
    rows.append((p, 100 * h1 / uses, 100 * h2 / max(h2 + m3, 1), miss_tok, slow_tok, pin_ms, extra_page, extra_ssd))

print()
print("| pinned tier | pinned GB | VRAM hit % | VRAM misses from pinned % | uploads/token | from slow tier/token | upload ms/token if all pinned | +ms/token, rest in page cache | decode t/s (page cache) | +ms/token, rest on SSD | decode t/s (SSD) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p, v, f2, mt, st, pm, ep, es in rows:
    gb = p * n_all * expert_mb / 1e3
    print(f"| {p:.2f} | {gb:.1f} | {v:.1f} | {f2:.1f} | {mt:.1f} | {st:.2f} | {pm:.1f} | {ep:.2f} | {1000 / (a.base_ms + ep):.1f} | {es:.2f} | {1000 / (a.base_ms + es):.1f} |")
