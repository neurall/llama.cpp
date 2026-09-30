#!/usr/bin/env python3
"""replay.py layers.jsonl [probe numbers]: what-if replay of one decode token from measured per-layer records
(tl_layers.py -j). Rates are the startup probe's (GB/s): --cpu (host expert op rate), --cpu-shared (CPU while a link
copies), --link (per GPU name, e.g. CUDA1=23.8,CUDA0=5.2), --lat (us per upload). Designs:
  base     : measured
  jit      : per layer, a share of the miss bytes goes over the layer GPU's link while the CPU reads the rest; share
             chosen to minimize the layer time (the GPU computes uploaded experts in ~0 time)
  jitfast  : jit, but all layers upload over the fastest link (JIT pool on the fastest GPU)
  shexp    : GPU work after the CPU run moved ahead of it (--moved us per layer off the critical path)
  gapfill  : --prec share of the next layer's miss bytes uploaded during the GPU wait (DDR idle), link-limited"""
import json, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("f")
ap.add_argument("--cpu", type=float, default=37.4); ap.add_argument("--cpu-shared", type=float, default=21.8)
ap.add_argument("--link", default="CUDA1=23.8,CUDA0=5.2"); ap.add_argument("--lat", type=float, default=15)
ap.add_argument("--moved", type=float, default=20); ap.add_argument("--prec", type=float, default=0.5)
a = ap.parse_args()
link = {k: float(v) for k, v in (x.split("=") for x in a.link.split(","))}
fast = max(link.values())
L = [json.loads(l) for l in open(a.f)]
over = lambda x: x["main_inputs"] + x["main_launch"] + x["tiny_run"] + x["chain_launch"]
def jit_cpu(cpu_us, rl):
    by = cpu_us * a.cpu * 1e3  # bytes (us * GB/s * 1e3)
    best = cpu_us
    for s in [i / 100 for i in range(1, 100)]:
        t = max(by * (1 - s) / (a.cpu_shared * 1e3), by * s / (rl * 1e3) + a.lat)
        best = min(best, t)
    return best
def token(design):
    t = 0
    for i, x in enumerate(L):
        cpu = x["cpu"]; wait = x["gpu_wait"]; o = over(x)
        if design in ("jit", "all"):
            cpu = jit_cpu(cpu, link.get(x["gpu_dev"], fast))
        if design == "jitfast":
            cpu = jit_cpu(cpu, fast)
        if design in ("shexp", "all"):
            o -= min(a.moved, o)
        if design in ("gapfill", "all"):
            # uploads during this layer's GPU wait cover a share of the next layer's CPU bytes (link-limited)
            nxt = L[i + 1]["cpu"] if i + 1 < len(L) else 0
            cover = min(a.prec * nxt * a.cpu, wait * link.get(x["gpu_dev"], fast)) / a.cpu  # us of CPU saved next layer
            if i + 1 < len(L): L[i + 1]["_save"] = cover
        cpu -= x.get("_save", 0)
        t += o + wait + max(cpu, 0)
    for x in L: x.pop("_save", None)
    return t / 1e3
b = token("base")
for d in ["base", "jit", "jitfast", "shexp", "gapfill", "all"]:
    ms = token(d); print(f"{d:8s} {ms:6.2f} ms/token  {1000/ms:5.1f} t/s  ({100*(b/ms-1):+5.1f}%)")
