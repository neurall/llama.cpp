#!/usr/bin/env python3
"""Timeline of the multi-GPU prompt split from an nsys sqlite export.   usage: nsys_timeline.py TRACE.sqlite OUT.png [FIRST_LAYER=20] [N_LAYERS=3] [MIN_KERNEL_MS=10]

One lane per card and event kind, time in ms from the first expert upload of the pass:
  x16 / x4 expert upload      host-to-device copies of 1 MB or more (the experts of the layer, staged ahead)
  x16 / x4 quantized matmuls (experts+dense)     the mul_mat_q kernels (gate, up, down)
  x16 / x4 other kernels      attention, norms, router, masks, sums
  hand-off / merge            the hidden state going between the cards (copies of 120-140 MB (n_tokens x n_embd x 4 B = 131 MB)), device to host and host to device
Dashed vertical lines are the layer boundaries (first heavy expert matmul of the layer on the x16 card); the layer's result is complete at the next line."""
import sqlite3, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
cps = c.execute("select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (s, e)).fetchall()
t0 = min(x[0] for x in cps if x[3] == 1 and x[4] >= 1 << 20)
mm = [k for k in b if k[3].startswith("mul_mat_q")]
big = [k for k in mm if k[2] == 1 and k[1] - k[0] > thr * 1e6]
starts = [big[i][0] for i in range(0, len(big) - 2, 3)] + [e]
a, z = starts[first], starts[first + nl]
r = lambda x: (x - t0) / 1e6
lanes = ["MERGE: hidden state x4<->x16 (copy)", "x4  COMPUTE: other kernels", "x4  COMPUTE: matmuls", "x4  UPLOAD: experts (copy)", "x16 COMPUTE: other kernels", "x16 COMPUTE: matmuls", "x16 UPLOAD: experts (copy)"]
col = {0: "#c0504d", 1: "#4f81bd"}
fig, ax = plt.subplots(figsize=(15, 6.2))
def bar(lane, x0, x1, color, label=None):
    if x1 < a or x0 > z: return
    ax.barh(lanes.index(lane), max(r(x1) - r(x0), 0.4), left=r(x0), height=0.7, color=color, edgecolor="none")
for k in b:
    d = k[2]; nm = "x4 " if d == 0 else "x16"
    if k[3].startswith("mul_mat_q"): bar(f"{nm} COMPUTE: matmuls", k[0], k[1], "#2a6f2a")
    else: bar(f"{nm} COMPUTE: other kernels", k[0], k[1], "#9a9a9a")
for st, en, d, kind, by in cps:
    nm = "x4 " if d == 0 else "x16"
    if kind == 1 and by >= 1 << 20: bar(f"{nm} UPLOAD: experts (copy)", st, en, col[d])
    elif 120 << 20 <= by <= 140 << 20 and kind in (1, 2): bar("MERGE: hidden state x4<->x16 (copy)", st, en, "#e69f00")
for i in range(first, first + nl + 1):
    ax.axvline(r(starts[i]), color="k", ls="--", lw=0.9)
    if i < first + nl: ax.text(r(starts[i]) + 2, len(lanes) - 0.35, f"layer {i}  ({r(starts[i+1]) - r(starts[i]):.0f} ms)", fontsize=9, va="bottom")
ax.set_yticks(range(len(lanes))); ax.set_yticklabels(lanes, fontsize=9)
ax.set_xlim(r(a) - 5, r(z) + 5); ax.set_xlabel("ms from the first expert upload of the pass"); ax.grid(axis="x", alpha=0.25)
ax.set_title("%s: GLM 8k prompt, one pass; layers %d-%d" % (path.split("/")[-1].replace(".sqlite", ""), first, first + nl - 1), fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4f81bd", label="UPLOAD x16 (host to GPU copy of experts)"), Patch(color="#c0504d", label="UPLOAD x4 (host to GPU copy of experts)"),
                   Patch(color="#2a6f2a", label="COMPUTE: quantized matmuls (experts and dense attention)"), Patch(color="#9a9a9a", label="COMPUTE: all other kernels"),
                   Patch(color="#e69f00", label="MERGE copy of the hidden state between the cards")], loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8, frameon=False)
for st, en, d, kind, by in cps:
    if kind == 1 and by >= 1 << 20 and r(en) - r(st) >= 18 and a <= st <= z:
        ax.text(r(st) + (r(en) - r(st)) / 2, lanes.index(("x4  UPLOAD: experts (copy)" if d == 0 else "x16 UPLOAD: experts (copy)")), "%.0f ms" % (r(en) - r(st)), ha="center", va="center", fontsize=7, color="white")
for i, ln in enumerate(lanes):
    if "UPLOAD" in ln: ax.axhspan(i - 0.5, i + 0.5, color="#dde8f5" if ln.startswith("x16") else "#f5dede", alpha=0.5, zorder=0)
plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
