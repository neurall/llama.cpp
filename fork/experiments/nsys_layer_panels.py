#!/usr/bin/env python3
"""One panel per layer, uploads attributed to the layer that consumes them. usage: nsys_layer_panels.py TRACE.sqlite OUT.png [FIRST_LAYER=20] [N_LAYERS=3] [MIN_KERNEL_MS=10]

A layer = its dense phase and its expert phase: it STARTS at the end of the previous layer's last expert matmul and ENDS at the end of its own last expert matmul (on either
card). Expert matmuls = the heavy mul_mat_q kernels (longer than MIN_KERNEL_MS), three per layer and card (gate, up, down).
Each upload (a copy of 100 MB or more) is given to the layer whose matmul reads it: copies and matmuls of a card are matched in order (the staging ring is first in, first out).
Panel of layer L: COMPUTE of layer L only (cut at its start and end); UPLOAD lanes separate "data of layer L" (dark) from "data of layer L+1" (light, prefetch) and
"data of layer L-1" is not drawn. The x axis starts where layer L's first upload began (it may be before the layer start: that is the prefetch during layer L-1)."""
import sqlite3, sys, bisect
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
path, out = sys.argv[1], sys.argv[2]
first = int(sys.argv[3]) if len(sys.argv) > 3 else 20; nl = int(sys.argv[4]) if len(sys.argv) > 4 else 3
thr = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
c = sqlite3.connect(path)
ks = c.execute("select k.start,k.end,k.deviceId,n.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds n on n.id=k.shortName order by k.start").fetchall()
bursts, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - cur[-1][1] > 400e6: bursts.append(cur); cur = [k]
    else: cur.append(k)
bursts.append(cur); b = max(bursts, key=lambda x: x[-1][1] - x[0][0]); s = b[0][0]; e = max(k[1] for k in b)
cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? order by start", (s, e)).fetchall()
heavy = {d: [k for k in b if k[3].startswith("mul_mat_q") and k[1] - k[0] > thr * 1e6 and k[2] == d] for d in (0, 1)}
trip = [heavy[1][i:i + 3] for i in range(0, len(heavy[1]) - 2, 3)]
def layer_end(i):
    a, z = trip[i][0][0], (trip[i + 1][0][0] if i + 1 < len(trip) else e)
    return max([trip[i][2][1]] + [k[1] for k in heavy[0] if a <= k[0] < z])
ends = [layer_end(i) for i in range(len(trip))]
def layer_of(t):                       # the layer whose window contains time t
    return bisect.bisect_left(ends, t)
# attribute uploads: per card, big copies in order <-> heavy matmuls in order (FIFO)
up = {0: [], 1: []}                    # card -> list of (start, end, bytes, layer of the consumer)
for d in (0, 1):
    big = [x for x in cps if x[2] == d and x[3] == 1 and x[4] >= 100 << 20 and not (120 << 20 <= x[4] <= 140 << 20)]
    # the x4 card does no dense work and its expert matmuls are short: every mul_mat_q kernel above 1.5 ms counts there
    mats = sorted(heavy[d] if d == 1 else [k for k in b if k[3].startswith("mul_mat_q") and k[2] == 0 and k[1] - k[0] > 1.5e6], key=lambda k: k[0]); j = 0
    for st, en, _, _, by in sorted(big, key=lambda x: x[1]):
        while j < len(mats) and mats[j][0] < en - 2e6: j += 1
        if j < len(mats): up[d].append((st, en, by, layer_of(mats[j][1]))); j += 1
lanes = ["MERGE: hidden state (copy)", "x4  COMPUTE: other kernels", "x4  COMPUTE: matmuls", "x4  UPLOAD: data of layer L+1 (prefetch)", "x4  UPLOAD: data of layer L",
         "x16 COMPUTE: other kernels", "x16 COMPUTE: matmuls", "x16 UPLOAD: data of layer L+1 (prefetch)", "x16 UPLOAD: data of layer L"]
fig, axs = plt.subplots(nl, 1, figsize=(14, 4.0 * nl))
if nl == 1: axs = [axs]
tot = []
for ax, li in zip(axs, range(first, first + nl)):
    a, z = ends[li - 1], ends[li]; T = (z - a) / 1e6
    mine = [x for d in (0, 1) for x in up[d] if x[3] == li]
    x0 = max(a - 600e6, min([a] + [x[0] for x in mine])); span0 = (x0 - a) / 1e6
    def bar(lane, st, en, color, clip=False):
        if clip: st, en = max(st, a), min(en, z)
        if en <= st or en < x0 or st > z + 1: return
        ax.barh(lanes.index(lane), max((en - st) / 1e6, 0.3), left=(st - a) / 1e6, height=0.7, color=color, edgecolor="none")
    for k in b:
        nm = "x4 " if k[2] == 0 else "x16"
        bar(f"{nm} COMPUTE: matmuls" if k[3].startswith("mul_mat_q") else f"{nm} COMPUTE: other kernels", k[0], k[1], "#2a6f2a" if k[3].startswith("mul_mat_q") else "#9a9a9a", True)
    for d in (0, 1):
        nm = "x4 " if d == 0 else "x16"
        for st, en, by, ly in up[d]:
            if ly == li: bar(f"{nm} UPLOAD: data of layer L", st, en, "#c0504d" if d == 0 else "#4f81bd")
            elif ly == li + 1: bar(f"{nm} UPLOAD: data of layer L+1 (prefetch)", st, en, "#e8b4b0" if d == 0 else "#b3cbe6")
            else: continue
            if ly in (li, li + 1) and (en - st) / 1e6 >= 18 and st < z and en > x0:
                ax.text(((st + en) / 2 - a) / 1e6, lanes.index(f"{nm} UPLOAD: data of layer L" if ly == li else f"{nm} UPLOAD: data of layer L+1 (prefetch)"), "%.0f" % ((en - st) / 1e6), ha="center", va="center", fontsize=7, color="black" if ly == li + 1 else "white")
    for st, en, d, kind, by in cps:
        if 120 << 20 <= by <= 140 << 20 and kind in (1, 2): bar("MERGE: hidden state (copy)", st, en, "#e69f00", True)
    for i, ln in enumerate(lanes):
        if "UPLOAD" in ln: ax.axhspan(i - 0.5, i + 0.5, color="#dde8f5" if ln.startswith("x16") else "#f5dede", alpha=0.45, zorder=0)
    ax.axvline(0, color="k", lw=2); ax.axvline(T, color="k", lw=2)
    ax.text(0, len(lanes) - 0.3, " LAYER %d START (0 ms)" % li, fontsize=9, va="bottom", ha="left", weight="bold")
    ax.text(T, len(lanes) - 0.3, "LAYER %d END (%.0f ms) " % (li, T), fontsize=9, va="bottom", ha="right", weight="bold")
    ax.set_yticks(range(len(lanes))); ax.set_yticklabels(lanes, fontsize=8); ax.set_xlim(span0 - 3, T + 3); ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("ms from the start of layer %d (the numbers on uploads are their durations in ms)" % li, fontsize=8)
    tot.append((li, T))
axs[0].legend(handles=[Patch(color="#4f81bd", label="UPLOAD x16, data used by this layer"), Patch(color="#b3cbe6", label="UPLOAD x16, data of the NEXT layer"), Patch(color="#c0504d", label="UPLOAD x4, this layer"),
                       Patch(color="#e8b4b0", label="UPLOAD x4, next layer"), Patch(color="#2a6f2a", label="COMPUTE matmuls (experts+dense)"), Patch(color="#9a9a9a", label="COMPUTE other"),
                       Patch(color="#e69f00", label="MERGE copy")], loc="upper center", bbox_to_anchor=(0.5, 1.55), ncol=4, fontsize=8, frameon=False)
fig.suptitle(path.split("/")[-1].replace(".sqlite", "") + ": GLM 8k, one pass", y=1.02, fontsize=11)
plt.tight_layout(); plt.savefig(out, dpi=100, bbox_inches="tight")
print("wrote", out, " layer totals ms:", ", ".join("%d: %.0f" % x for x in tot), " sum %.0f" % sum(x[1] for x in tot))
