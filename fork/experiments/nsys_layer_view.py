#!/usr/bin/env python3
"""One layer, nine lanes. usage: nsys_layer_view.py TRACE.sqlite OUT.png LAYER [MIN_KERNEL_MS=10] [RAM_PEAK_GBS=46] [X16_PEAK=26] [X4_PEAK=6.5]

Layer = its dense phase and its expert phase: it starts (thick line, LAYER START) at the end of the previous layer's last expert matmul and ends (LAYER END) at the end of its own last one.
Lanes, top to bottom (time in ms from the layer start):
  1  UPLOAD x16, the experts this layer uses          2  UPLOAD x4, the same
  3  COMPUTE x16, expert matmuls of this layer         4  COMPUTE x4, expert matmuls of this layer
  5  PREFETCH x16 for layer L+1 (uploads of the next layer's experts, wherever they run)       6  PREFETCH x4, the same
  7  host memory read bandwidth (the sum of both cards' uploads), % of RAM_PEAK_GBS and GB/s
  8  x16 link, % of X16_PEAK and GB/s                 9  x4 link, % of X4_PEAK and GB/s
An upload (a copy of 100 MB or more) belongs to the layer whose expert matmul reads it (copies and matmuls of a card matched in order: the ring is first in, first out).
Expert matmuls = the heavy mul_mat_q kernels (x16: longer than MIN_KERNEL_MS; x4: longer than 1.5 ms); dense attention matmuls are shorter and not drawn."""
import sqlite3, sys, bisect
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
path, out, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
RAM = float(sys.argv[5]) if len(sys.argv) > 5 else 46.0; P16 = float(sys.argv[6]) if len(sys.argv) > 6 else 26.0; P4 = float(sys.argv[7]) if len(sys.argv) > 7 else 6.5
c = sqlite3.connect(path)
ks = c.execute("select k.start,k.end,k.deviceId,n.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName order by k.start").fetchall()
bursts, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur); b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? order by start", (s - 1_000_000_000, e)).fetchall()
mmq = [k for k in b if k[3].startswith("mul_mat_q")]
heavy = {1: [k for k in mmq if k[2] == 1 and k[1] - k[0] > thr * 1e6], 0: [k for k in mmq if k[2] == 0 and k[1] - k[0] > 1.5e6]}
trip = [heavy[1][i:i + 3] for i in range(0, len(heavy[1]) - 2, 3)]
def layer_end(i):
    a, z = trip[i][0][0], (trip[i + 1][0][0] if i + 1 < len(trip) else e)
    return max([trip[i][2][1]] + [k[1] for k in heavy[0] if a <= k[0] < z])
ends = [layer_end(i) for i in range(len(trip))]
layer_of = lambda t: bisect.bisect_left(ends, t)
up = {0: [], 1: []}
for d in (0, 1):
    big = [x for x in cps if x[2] == d and x[3] == 1 and x[4] >= 100 << 20 and not (120 << 20 <= x[4] <= 140 << 20)]
    mats = sorted(heavy[d], key=lambda k: k[0]); j = 0
    for st, en, _, _, by in sorted(big, key=lambda x: x[1]):
        while j < len(mats) and mats[j][0] < en - 2e6: j += 1
        if j < len(mats): up[d].append((st, en, by, layer_of(mats[j][1]))); j += 1
a, z = ends[L - 1], ends[L]; T = (z - a) / 1e6
mine = [x for d in (0, 1) for x in up[d] if x[3] == L]; nxt = [x for d in (0, 1) for x in up[d] if x[3] == L + 1]
x_lo = max(a - 700e6, min([a] + [x[0] for x in mine])); x_hi = z
rel = lambda t: (t - a) / 1e6
fig = plt.figure(figsize=(15, 14.6))
gs = fig.add_gridspec(13, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 0.8, 0.8], hspace=0.28)
axs = [fig.add_subplot(gs[i, 0]) for i in range(13)]
for ax in axs[1:]: ax.sharex(axs[0])
names = ["1  UPLOAD x16: experts used by layer %d" % L, "2  UPLOAD x4: experts used by layer %d" % L, "3  COMPUTE x16: expert matmuls", "4  COMPUTE x16: dense + other kernels",
         "5  COMPUTE x4: expert matmuls", "6  COMPUTE x4: dense + other kernels", "7  PREFETCH x16: experts for layer %d" % (L + 1), "8  PREFETCH x4: experts for layer %d" % (L + 1),
         "9  host memory read", "10  x16 link", "11  x4 link"]
cols = ["#2f5f9e", "#b03a36", "#2a6f2a", "#2a6f2a", "#8fb3dd", "#e3a09c"]
def draw(ax, ivs, color, label_ms=False):
    for st, en in ivs:
        if en < x_lo or st > x_hi: continue
        ax.barh(0, rel(en) - rel(st), left=rel(st), height=0.8, color=color)
        if label_ms and en - st > 14e6: ax.text((rel(st) + rel(en)) / 2, 0, "%.0f ms" % ((en - st) / 1e6), ha="center", va="center", fontsize=8, color="white")
draw(axs[0], [(x[0], x[1]) for x in up[1] if x[3] == L], cols[0], True)
draw(axs[1], [(x[0], x[1]) for x in up[0] if x[3] == L], cols[1], True)
isexp = lambda k, dv: k in heavy_set[dv]
heavy_set = {1: set(map(id, heavy[1])), 0: set(map(id, heavy[0]))}
win = lambda k: a - 1 <= k[0] and k[1] <= z + 1
draw(axs[2], [(k[0], k[1]) for k in heavy[1] if win(k)], "#2a6f2a")
draw(axs[3], [(k[0], k[1]) for k in b if k[2] == 1 and id(k) not in heavy_set[1] and win(k)], "#7a7a7a")
draw(axs[4], [(k[0], k[1]) for k in heavy[0] if win(k)], "#2a6f2a")
draw(axs[5], [(k[0], k[1]) for k in b if k[2] == 0 and id(k) not in heavy_set[0] and win(k)], "#7a7a7a")
draw(axs[6], [(x[0], x[1]) for x in up[1] if x[3] == L + 1], cols[4], False)
draw(axs[7], [(x[0], x[1]) for x in up[0] if x[3] == L + 1], cols[5], False)
for i in range(8):
    axs[i].set_yticks([]); axs[i].set_ylim(-0.5, 0.5); axs[i].set_ylabel(names[i], rotation=0, ha="right", va="center", fontsize=9)
    axs[i].axvspan(rel(x_lo), 0, color="#f2f2f2", zorder=0)           # before the layer starts
# bandwidth lanes: bytes of every host-to-device copy per 2 ms bin
binw = 2e6; edges = np.arange(x_lo, x_hi + binw, binw); nb = len(edges) - 1
def rate(dev):
    by = np.zeros(nb)
    for st, en, d, kind, n in cps:
        if kind != 1 or (dev is not None and d != dev) or en <= x_lo or st >= x_hi or n < 65536: continue
        dur = max(en - st, 1)
        i0 = max(0, int((st - x_lo) // binw)); i1 = min(nb - 1, int((en - x_lo) // binw))
        for i in range(i0, i1 + 1):
            o = min(en, edges[i + 1]) - max(st, edges[i])
            if o > 0: by[i] += n * o / dur
    return by / (binw / 1e9) / 1e9                                      # GB/s per bin
xs = (edges[:-1] - a) / 1e6
import nsys_states as NS
ns_d = NS.load(path, thr); ns_out, ns_T = NS.classify(ns_d, a, z)
SCOL = ["#2a6f2a", "#4f81bd", "#e69f00", "#d9a0a0", "#c0504d", "#7a3f98", "#888888", "#bbbbbb", "#000000"]
for ax, dv, lab in ((axs[11], 1, "12  STATE x16 (every ms)"), (axs[12], 0, "13  STATE x4 (every ms)")):
    st_arr = ns_out[dv]
    for i in range(len(st_arr)):
        ax.barh(0, 0.25, left=i * 0.25, height=0.9, color=SCOL[st_arr[i]], linewidth=0)
    ax.set_yticks([]); ax.set_ylim(-0.5, 0.5); ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
from matplotlib.patches import Patch
axs[12].legend(handles=[Patch(color=SCOL[i], label="%s (x16 %.0f / x4 %.0f ms)" % (NS.STATES[i], (ns_out[1] == i).sum() * 0.25, (ns_out[0] == i).sum() * 0.25)) for i in range(9)], loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=8, frameon=False)
for ax, dev, peak, lab, col in ((axs[8], None, RAM, names[8], "#555555"), (axs[9], 1, P16, names[9], "#2f5f9e"), (axs[10], 0, P4, names[10], "#b03a36")):
    r = rate(dev); ax.fill_between(xs, 0, np.minimum(100 * r / peak, 160), step="post", color=col, alpha=0.85)
    ax.set_ylim(0, 105); ax.axhline(100, color="k", lw=0.6, ls=":")
    ax.set_ylabel(lab + "\n%% of %.1f GB/s" % peak, rotation=0, ha="right", va="center", fontsize=9)
    ax.text(0.995, 0.9, "busy-bin level %.1f GB/s (95th percentile), mean %.1f GB/s over the layer" % (np.percentile(r[r > 0.5], 95) if (r > 0.5).any() else 0, r[(xs >= 0)].mean() if (xs >= 0).any() else 0), transform=ax.transAxes, ha="right", va="top", fontsize=8)
for ax in axs:
    ax.axvline(0, color="k", lw=2.2); ax.axvline(T, color="k", lw=2.2); ax.grid(axis="x", alpha=0.25)
axs[0].text(0, 0.62, " LAYER %d START (0 ms)" % L, fontsize=10, weight="bold", ha="left", va="bottom", transform=axs[0].get_xaxis_transform())
axs[0].text(T, 0.62, "LAYER %d END (%.0f ms) " % (L, T), fontsize=10, weight="bold", ha="right", va="bottom", transform=axs[0].get_xaxis_transform())
axs[0].set_xlim(rel(x_lo) - 2, T + 3); axs[12].set_xlabel("ms from the start of layer %d  (grey area = before the layer starts: prefetch during the layer before)" % L)
for ax in axs[:-1]: plt.setp(ax.get_xticklabels(), visible=False)
fig.suptitle("%s: GLM 8k prompt, one pass, layer %d (%.0f ms)" % (path.split("/")[-1].replace(".sqlite", ""), L, T), fontsize=12, y=0.92)
plt.savefig(out, dpi=100, bbox_inches="tight"); print("wrote", out, "layer %d: %.0f ms" % (L, T))
