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
import sqlite3, sys, bisect, os
GAP = float(os.environ.get("NSYS_GAP_MS", "400")) * 1e6; X4THR = float(os.environ.get("NSYS_X4_MS", "1.5")) * 1e6; UPMB = float(os.environ.get("NSYS_UP_MB", "100")) * (1 << 20)   # burst gap; x4 expert matmul size limit
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
    if k[0] - cur[-1][1] > GAP: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur)
big = [x for x in bursts if len(x) >= 0.8 * max(len(y) for y in bursts)]; b = min(big, key=lambda x: max(k[1] for k in x) - x[0][0]) if os.environ.get("NSYS_PICK", "fastest") == "fastest" else big[-1]
s = b[0][0]; e = max(k[1] for k in b)
cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? order by start", (s - 1_000_000_000, e)).fetchall()
mmq = [k for k in b if k[3].startswith("mul_mat_q")]
heavy = {1: [k for k in mmq if k[2] == 1 and k[1] - k[0] > thr * 1e6], 0: [k for k in mmq if k[2] == 0 and k[1] - k[0] > X4THR]}
trip = [heavy[1][i:i + 3] for i in range(0, len(heavy[1]) - 2, 3)]
def layer_end(i):
    a, z = trip[i][0][0], (trip[i + 1][0][0] if i + 1 < len(trip) else e)
    return max([trip[i][2][1]] + [k[1] for k in heavy[0] if a <= k[0] < z])
ends = [layer_end(i) for i in range(len(trip))]
layer_of = lambda t: bisect.bisect_left(ends, t)
up = {0: [], 1: []}
for d in (0, 1):
    big = [x for x in cps if x[2] == d and x[3] == 1 and x[4] >= UPMB and not (120 << 20 <= x[4] <= 140 << 20)]
    mats = sorted(heavy[d], key=lambda k: k[0]); j = 0
    for st, en, _, _, by in sorted(big, key=lambda x: x[1]):
        while j < len(mats) and mats[j][0] < en - 2e6: j += 1
        if j < len(mats): up[d].append((st, en, by, layer_of(mats[j][1]))); j += 1
a, z = ends[L - 1], ends[L]; T = (z - a) / 1e6
mine = [x for d in (0, 1) for x in up[d] if x[3] == L]; nxt = [x for d in (0, 1) for x in up[d] if x[3] == L + 1]
x_lo = a if os.environ.get("NSYS_FROM_START", "1") == "1" else max(a - 700e6, min([a] + [x[0] for x in mine])); x_hi = z   # default: from this layer's first own upload (its data), only its own activity
rel = lambda t: (t - a) / 1e6
fig = plt.figure(figsize=(15, 16.6))
gs = fig.add_gridspec(15, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 0.8, 0.8, 1.0, 1.0], hspace=0.28)
axs = [fig.add_subplot(gs[i, 0]) for i in range(15)]
for ax in axs[1:]: ax.sharex(axs[0])
names = ["1  UPLOAD x16: experts used by layer %d" % L, "2  UPLOAD x4: experts used by layer %d" % L, "3  COMPUTE x16: expert matmuls", "4  COMPUTE x16: dense + other kernels",
         "5  COMPUTE x4: expert matmuls", "6  COMPUTE x4: dense + other kernels", "7  PREFETCH x16: experts of later layers", "8  PREFETCH x4: experts of later layers",
         "9  host memory read", "10  x16 link", "11  x4 link"]
cols = ["#2f5f9e", "#b03a36", "#2a6f2a", "#2a6f2a", "#8fb3dd", "#e3a09c"]
def draw(ax, ivs, color, label_ms=False):
    for st, en in ivs:
        if en < x_lo or st > x_hi: continue
        ax.barh(0, rel(en) - rel(st), left=rel(st), height=0.8, color=color)
        if label_ms and en - st > 14e6: ax.text((rel(st) + rel(en)) / 2, 0, "%.0f ms" % ((en - st) / 1e6), ha="center", va="center", fontsize=8, color="white")
# this layer's uploads that ran before the layer started (prefetched during the previous one): noted as text, the axis stays on the layer itself
for ax_i, dd in ((0, 1), (1, 0)):
    early = [x for x in up[dd] if x[3] == L and x[1] <= a]
    if early:
        axs[ax_i].text(0.5, 0, "  %d copies done earlier (%.0f ms of copying, finished %.0f ms before the start)" % (len(early), sum(x[1] - x[0] for x in early) / 1e6, (a - max(x[1] for x in early)) / 1e6),
                       ha="left", va="center", fontsize=7, color="#444444")
draw(axs[0], [(x[0], x[1]) for x in up[1] if x[3] == L], cols[0], True)
draw(axs[1], [(x[0], x[1]) for x in up[0] if x[3] == L], cols[1], True)
isexp = lambda k, dv: k in heavy_set[dv]
heavy_set = {1: set(map(id, heavy[1])), 0: set(map(id, heavy[0]))}
win = lambda k: a - 1 <= k[0] and k[1] <= z + 1
draw(axs[2], [(k[0], k[1]) for k in heavy[1] if win(k)], "#2a6f2a")
draw(axs[3], [(k[0], k[1]) for k in b if k[2] == 1 and id(k) not in heavy_set[1] and win(k)], "#7a7a7a")
draw(axs[4], [(k[0], k[1]) for k in heavy[0] if win(k)], "#2a6f2a")
draw(axs[5], [(k[0], k[1]) for k in b if k[2] == 0 and id(k) not in heavy_set[0] and win(k)], "#7a7a7a")
draw(axs[6], [(x[0], x[1]) for x in up[1] if x[3] > L], cols[4], True)
draw(axs[7], [(x[0], x[1]) for x in up[0] if x[3] > L], cols[5], True)
for i in range(8):
    axs[i].set_yticks([]); axs[i].set_ylim(-0.5, 0.5); axs[i].set_ylabel(names[i], rotation=0, ha="right", va="center", fontsize=9)
    axs[i].axvspan(rel(x_lo), 0, color="#f2f2f2", zorder=0)           # before the layer starts
# bandwidth lanes: bytes of every host-to-device copy per 2 ms bin
own_or_next = set((x[0], x[1]) for dd in (0, 1) for x in up[dd] if x[3] >= L)
binw = 2e6; edges = np.arange(x_lo, x_hi + binw, binw); nb = len(edges) - 1
def rate(dev):
    by = np.zeros(nb)
    for st, en, d, kind, n in cps:
        if kind != 1 or (dev is not None and d != dev) or en <= x_lo or st >= x_hi or n < 65536: continue
        if (st, en) not in own_or_next: continue   # only the copies of this layer's experts and of the next layer's (prefetch): no other layer's traffic
        dur = max(en - st, 1)
        i0 = max(0, int((st - x_lo) // binw)); i1 = min(nb - 1, int((en - x_lo) // binw))
        for i in range(i0, i1 + 1):
            o = min(en, edges[i + 1]) - max(st, edges[i])
            if o > 0: by[i] += n * o / dur
    return by / (binw / 1e9) / 1e9                                      # GB/s per bin
xs = (edges[:-1] - a) / 1e6
import nsys_states as NS
ns_d = NS.load(path, thr); ns_out, ns_T = NS.classify(ns_d, a, z)
SCOL = ["#2a6f2a", "#4f81bd", "#e69f00", "#d9a0a0", "#c0504d", "#7a3f98", "#888888", "#bbbbbb", "#000000", "#00a6a6", "#f0d000", "#ff00ff"]
SHORT = ["compute", "upload", "copy", "wait other card upload", "wait other card compute", "host sync", "host launching", "host memcpy", "unexplained", "host CPU in launch", "host CPU in input copy", "host outside loop"]
for ax, dv, lab in ((axs[11], 1, "12  STATE x16 (every ms)"), (axs[12], 0, "13  STATE x4 (every ms)")):
    st_arr = ns_out[dv]; i = 0
    while i < len(st_arr):
        j = i
        while j < len(st_arr) and st_arr[j] == st_arr[i]: j += 1
        x0, w = i * 0.25, (j - i) * 0.25
        ax.barh(0, w, left=x0, height=0.9, color=SCOL[st_arr[i]], linewidth=0)
        if w >= 2.0 * T / 46.0:
            ax.text(x0 + w / 2, 0, SHORT[st_arr[i]], ha="center", va="center", fontsize=6.5, color="white" if st_arr[i] in (0, 1, 4, 5, 8) else "black", clip_on=True)
        i = j
    ax.set_yticks([]); ax.set_ylim(-0.5, 0.5); ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
from matplotlib.patches import Patch
# scheduler thread, from the NVTX ranges of the run (GGML_NVTX=1, nsys --trace=cuda,nvtx): lane 14 = split launches (blue = x16 split, red = x4 split, grey = other), lane 15 = what the thread does inside them
has_nvtx = c.execute("select count(*) from sqlite_master where name='NVTX_EVENTS'").fetchone()[0] > 0
if has_nvtx:
    nv = c.execute("select start,end,text from NVTX_EVENTS where end>=? and start<=?", (a, z)).fetchall()
    for st, en, tx in nv:
        x0, x1 = rel(max(st, a)), rel(min(en, z))
        if tx.startswith("SPLIT"):
            col = "#4f81bd" if "CUDA1" in tx else "#c0504d" if "CUDA0" in tx else "#aaaaaa"
            axs[13].barh(0, x1 - x0, left=x0, height=0.8, color=col, edgecolor="white", linewidth=0.4)
            if x1 - x0 > 1.5: axs[13].text((x0 + x1) / 2, 0, tx.split()[-1].replace("ffn_moe_", "").replace("-" + tx.split()[-1].split("-")[-1], ""), ha="center", va="center", fontsize=6, color="white")
        elif tx.startswith(("COPY inputs", "LAUNCH", "WAIT", "TENSOR_COPY")):
            col = "#e69f00" if tx.startswith("COPY inputs") else "#2a6f2a" if tx.startswith("LAUNCH") else "#7a3f98" if tx.startswith("WAIT") else "#000000"
            hgt = 0.8 if tx.startswith(("COPY inputs", "LAUNCH")) else 0.45
            axs[14].barh(0 if hgt == 0.8 else 0.18, x1 - x0, left=x0, height=hgt, color=col, edgecolor="none")
    axs[13].set_ylabel("14  HOST scheduler thread:\nsplit launches (blue x16, red x4)", rotation=0, ha="right", va="center", fontsize=9)
    axs[14].set_ylabel("15  HOST inside a launch:\norange = COPY inputs (sync), green = LAUNCH,\npurple/black = named WAIT / sync copy", rotation=0, ha="right", va="center", fontsize=8)
    for i in (13, 14): axs[i].set_yticks([]); axs[i].set_ylim(-0.5, 0.5)
else:
    for i in (13, 14): axs[i].axis("off")
fig.legend(handles=[Patch(color=SCOL[i], label="%s (x16 %.0f / x4 %.0f ms)" % (NS.STATES[i], (ns_out[1] == i).sum() * 0.25, (ns_out[0] == i).sum() * 0.25)) for i in range(12) if ((ns_out[1] == i).sum() + (ns_out[0] == i).sum()) > 0], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False, title="STATE lanes 12-13: every ms of each card (ms: x16 / x4)", title_fontsize=9)
for ax, dev, peak, lab, col in ((axs[8], None, RAM, names[8], "#555555"), (axs[9], 1, P16, names[9], "#2f5f9e"), (axs[10], 0, P4, names[10], "#b03a36")):
    r = rate(dev); ax.fill_between(xs, 0, np.minimum(100 * r / peak, 160), step="post", color=col, alpha=0.85)
    ax.set_ylim(0, 105); ax.axhline(100, color="k", lw=0.6, ls=":")
    ax.set_ylabel(lab + "\n%% of %.1f GB/s" % peak, rotation=0, ha="right", va="center", fontsize=9)
    ax.text(0.995, 0.9, "busy-bin level %.1f GB/s (95th percentile), mean %.1f GB/s over the layer" % (np.percentile(r[r > 0.5], 95) if (r > 0.5).any() else 0, r[(xs >= 0)].mean() if (xs >= 0).any() else 0), transform=ax.transAxes, ha="right", va="top", fontsize=8)
for ax in axs:
    ax.axvline(0, color="k", lw=2.2); ax.axvline(T, color="k", lw=2.2); ax.grid(axis="x", alpha=0.25)
axs[0].text(0, 0.62, " LAYER %d START (0 ms)" % L, fontsize=10, weight="bold", ha="left", va="bottom", transform=axs[0].get_xaxis_transform())
axs[0].text(T, 0.62, "LAYER %d END (%.0f ms) " % (L, T), fontsize=10, weight="bold", ha="right", va="bottom", transform=axs[0].get_xaxis_transform())
axs[0].set_xlim(rel(x_lo) - 2, T + 3); axs[14].set_xlabel("ms from the start of layer %d  (grey area = this layer's experts uploading before its compute starts; only this layer's activity is drawn)" % L)
for ax in axs[:-1]: plt.setp(ax.get_xticklabels(), visible=False)
for ax in axs: ax.set_xlim(rel(x_lo) - 2, T + 3)
fig.suptitle("%s: GLM 8k prompt, one pass, layer %d (%.0f ms)" % (path.split("/")[-1].replace(".sqlite", ""), L, T), fontsize=12, y=0.92)
plt.savefig(out, dpi=100, bbox_inches="tight"); print("wrote", out, "layer %d: %.0f ms" % (L, T))
