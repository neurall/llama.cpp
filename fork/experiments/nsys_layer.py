#!/usr/bin/env python3
"""Per-layer, per-token timing of one prompt pass from an nsys sqlite export (nsys export --type sqlite).

usage: nsys_layer.py TRACE.sqlite TOKENS [MIN_KERNEL_MS=12]

What it does: takes the longest burst of GPU kernels (the prompt pass), finds the layers by the expert matmuls (three mul_mat_q kernels longer than
MIN_KERNEL_MS per MoE layer: gate, up, down) on the card that runs the most of them, and for every layer reports the wall time, the time per token, and how that wall time splits on each
card into kernel-busy time, idle time, and host-to-device copy time. Totals for the whole pass at the end. Needs only the cuda trace of nsys.
"""
import sqlite3, sys

def main():
    path, tokens = sys.argv[1], int(sys.argv[2])
    min_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    c = sqlite3.connect(path)
    ks = c.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    bursts, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
        else: cur.append(k)
    bursts.append(cur)
    b = max(bursts, key=lambda x: x[-1][1] - x[0][0])
    s, e = b[0][0], max(k[1] for k in b)
    devs = sorted({k[2] for k in b})
    cp = c.execute("select start,end,deviceId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and start>=? and start<=? order by start", (s - 5e8, e)).fetchall()
    # a layer = three consecutive heavy expert matmuls (mul_mat_q longer than MIN_KERNEL_MS: gate, up, down) on the card that has the most of them
    heavy = c.execute("select k.start,k.end,k.deviceId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName where n.value like 'mul_mat_q%' and k.end-k.start>? and k.start>=? and k.start<=? order by k.start",
                      (min_ms * 1e6, s, e)).fetchall()
    cnt = {d: sum(1 for h in heavy if h[2] == d) for d in devs}
    ref = max(cnt, key=cnt.get)
    hv = [h for h in heavy if h[2] == ref]
    if len(hv) < 3:
        print("no heavy expert matmuls found: lower MIN_KERNEL_MS"); return
    layers = [hv[i:i + 3] for i in range(0, len(hv) - len(hv) % 3, 3)]
    starts = [s] + [l[-1][1] for l in layers[:-1]] + [e]   # layer i runs from the end of layer i-1's last expert matmul to the end of its own
    layers_end = starts[1:]
    print("pass window %.2f s, %d tokens, %d layers found by %d heavy expert matmuls (>%.0f ms) on card %d, cards %s" % ((e - s) / 1e9, tokens, len(layers), len(hv), min_ms, ref, devs))
    print("layer  wall_ms  us/token | per card: kernel_busy_ms  copy_ms  idle_ms")
    tot = {d: [0.0, 0.0, 0.0] for d in devs}
    tw = 0.0
    for i, l in enumerate(layers):
        a, z = starts[i], starts[i + 1]
        if i == len(layers) - 1: z = layers_end[i] if layers_end[i] < e else e
        wall = (z - a) / 1e6
        tw += wall
        row = []
        for d in devs:
            kb = sum(max(0, min(k[1], z) - max(k[0], a)) for k in b if k[2] == d and k[1] > a and k[0] < z) / 1e6
            cb = sum(max(0, min(x[1], z) - max(x[0], a)) for x in cp if x[2] == d and x[1] > a and x[0] < z) / 1e6
            idle = max(0.0, wall - kb)
            row.append((kb, cb, idle))
            tot[d][0] += kb; tot[d][1] += cb; tot[d][2] += idle
        print("%4d  %7.1f  %8.1f | %s" % (i, wall, wall * 1e3 / tokens, " | ".join("%6.1f %6.1f %6.1f" % r for r in row)))
    print("total  %7.1f  %8.1f | %s" % (tw, tw * 1e3 / tokens / len(layers), " | ".join("%6.0f %6.0f %6.0f" % tuple(tot[d]) for d in devs)))
    print("(per-layer mean: %.1f ms wall = %.2f us per token per layer)" % (tw / len(layers), tw * 1e3 / tokens / len(layers)))

main()
