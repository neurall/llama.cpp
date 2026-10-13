#!/usr/bin/env python3
"""One layer of a prompt pass, every card: what it computes, what its links carry, and for every wait what it waits for and why.

usage: nsys_wait_chart.py TRACE.sqlite OUT.png LAYER [PASS=longest]
       (trace: nsys profile --trace=cuda,nvtx --cuda-graph-trace=node, GGML_NVTX=1; nsys export --type sqlite)

Three lanes per card, DEVICE time, ms from the first kernel of the layer:
  compute   kernel runs, one box per scheduler split (NVTX "SPLIT n CARD OP name": the first node names it); hatched = WAIT n, reason in the table
  H2D       host -> card: expert uploads (purple, the prefetch ring's own stream) and hand-off writes (orange, on the compute stream)
  D2H       card -> host: hand-off reads (orange)
Why a card waited: walk back from the moment it resumed. Whatever ended then is what it waited for: its own hand-off copy, a copy or a kernel of another card.
A copy that started the moment another copy of the same direction on the same card ended was queued behind it (a card's copies of one direction run one
after the other, first in, first out), so a hand-off behind an expert upload reads "queued behind upload ...". The chain continues until the wait's start."""
import sqlite3, sys, bisect, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

db, out, LAYER = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = sqlite3.connect(db)
p0 = [r[0] for r in c.execute("select start from NVTX_EVENTS where text like 'SPLIT 0 %' order by start")]
passes = []
for i, s in enumerate(p0):
    e = p0[i + 1] if i + 1 < len(p0) else s + 9e10
    n, last = c.execute("select count(*), max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'", (s, e)).fetchone()
    if n >= 20 and last - s > 100e6: passes.append((s, e, last - s))
pi = int(sys.argv[4]) if len(sys.argv) > 4 else max(range(len(passes)), key=lambda i: passes[i][2])
S, E_, _ = passes[pi]
sp = c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start", (S, E_)).fetchall()
st = [x[0] for x in sp]
so = lambda t: bisect.bisect_right(st, t) - 1
rt = {cid: t for cid, t in c.execute("select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?", (S - 1e9, E_))}
K = [(a, b, d, strm, so(rt[cid])) for a, b, d, strm, cid in c.execute(
    "select start,end,deviceId,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?", (S, E_ + 3e9)) if cid in rt]
M = [(a, b, d, strm, k, by, so(rt[cid]) if cid in rt else -1) for a, b, d, strm, k, by, cid in c.execute(
    "select start,end,deviceId,streamId,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? and copyKind in (1,2)", (S - 1e9, E_ + 3e9))]
devs = sorted(set(k[2] for k in K))
cname = {}
for d in devs:
    cnt = {}
    for k in K:
        if k[2] == d and k[4] >= 0:
            m = re.match(r"SPLIT \d+ (\S+) ", sp[k[4]][2])
            if m: cnt[m.group(1)] = cnt.get(m.group(1), 0) + 1
    cname[d] = max(cnt, key=cnt.get) if cnt else "dev%d" % d
cstream = {}
for d in devs:
    t = {}
    for k in K:
        if k[2] == d: t[k[3]] = t.get(k[3], 0) + k[1] - k[0]
    cstream[d] = max(t, key=t.get)
first_k = {}
for k in K:
    if k[4] >= 0 and (k[4] not in first_k or k[0] < first_k[k[4]]): first_k[k[4]] = k[0]
def lay_first(L):
    v = [first_k[i] for i, x in enumerate(sp) if i in first_k and re.search(r"-%d$" % L, x[2])]
    return min(v) if v else None
A, B = lay_first(LAYER), lay_first(LAYER + 1)
if A is None or B is None: sys.exit("layer %d not found in pass %d" % (LAYER, pi))

def short(i):
    if i < 0: return "?"
    t = re.sub(r"^SPLIT \d+ \S+ ", "", sp[i][2])
    t = re.sub(r" \((view|copy|cont)\)", "", t)
    return "#%d %s" % (i, t[:34])
runs = {d: {} for d in devs}
for k in K:
    if k[1] <= A or k[0] >= B: continue
    r = runs[k[2]].setdefault(k[4], [k[0], k[1], 0.0])
    r[0] = min(r[0], k[0]); r[1] = max(r[1], k[1]); r[2] += k[1] - k[0]
acts = []   # start, end, dev, kind ('K' kernel run, 'U' expert upload, 'HW' hand-off write, 'HR' hand-off read), label
for d in devs:
    for i, r in runs[d].items(): acts.append((r[0], r[1], d, "K", short(i)))
for m in M:
    if m[1] <= A - 400e6 or m[0] >= B or m[5] < 65536: continue
    if m[4] == 1 and m[3] != cstream[m[2]]: kind = "U"
    elif m[4] == 1: kind = "HW"
    else: kind = "HR"
    acts.append((m[0], m[1], m[2], kind, "%.1f MB, issued in %s" % (m[5] / 1e6, short(m[6]))))
acts.sort(key=lambda a: a[1])
ends = [a[1] for a in acts]
EPS = 0.25e6
desc = {"K": "computing", "U": "expert upload", "HW": "hand-off write", "HR": "hand-off read"}
def why(d, g0, g1):
    """segments (a, b, text) covering [g0, g1] of card d's compute lane"""
    segs = []; t = g1
    for _ in range(40):
        if t <= g0 + 1e4: break
        lo = bisect.bisect_left(ends, t - EPS); hi = bisect.bisect_right(ends, t + EPS)
        cand = [a for a in acts[lo:hi] if a[0] < t - 1e4 and not (a[2] == d and a[3] == "K")]
        if not cand:
            j = lo - 1
            while j >= 0 and acts[j][2] == d and acts[j][3] == "K": j -= 1
            a0 = max(g0, acts[j][1]) if j >= 0 else g0
            segs.append((a0, t, "nothing runs on any card (launch / event latency, or the host has not issued the next work yet)")); t = a0; continue
        prio = lambda a: (0 if a[2] == d and a[3] == "HW" else 1 if a[3] in ("HR", "HW") else 2 if a[3] == "K" else 3, -a[1])
        a = min(cand, key=prio)
        who = cname[a[2]]
        if a[3] == "K": txt = "waits for %s: %s %s" % (who, desc["K"], a[4])
        else:
            txt = "waits for %s's %s (%s)" % (who, desc[a[3]], a[4])
            # was this copy queued behind another copy of the same direction on the same card?
            same = "U" if a[3] in ("U", "HW") else "HR"
            prev = [p for p in acts if p[2] == a[2] and p[3] in (("U", "HW") if same == "U" else ("HR",)) and abs(p[1] - a[0]) < EPS and p is not a]
            if prev:
                p = max(prev, key=lambda x: x[1] - x[0])
                txt += "; it was queued on %s's %s engine behind %s (%s), which ran %.1f ms" % (who, "upload" if same == "U" else "download", desc[p[3]], p[4], (p[1] - p[0]) / 1e6)
        b0 = max(g0, a[0]); segs.append((b0, t, txt)); t = b0
    if t > g0 + 1e4: segs.append((g0, t, "not explained"))
    return segs[::-1]

ms = lambda v: v / 1e6
waits = []
lanes = {}
for d in devs:
    rr = sorted((r[0], r[1], i) for i, r in runs[d].items())
    t = A; lane = []
    for a, b, i in rr + [(B, B, -1)]:
        a = max(a, A)
        if a - t > 0.5e6:
            for x, y, txt in why(d, t, a):
                if y - x > 0.2e6:
                    waits.append((d, x, y, txt)); lane.append((x, y, "W", len(waits)))
        if i < 0: break
        lane.append((max(a, A), min(b, B), "K", short(i)))
        t = max(t, min(b, B))
    lanes[d] = lane

# ---- picture ----
order = sorted(devs, key=lambda d: -sum(r[2] for r in runs[d].values()))
fig, (ax, axt) = plt.subplots(2, 1, figsize=(21, 3.2 * len(devs) + 0.26 * (len(waits) + 6)), gridspec_kw={"height_ratios": [3.2 * len(devs), 0.26 * (len(waits) + 6)]})
def bar(y, a, b, col, txt="", tc="k", hatch=None, ec="k", h=0.74, fs=7):
    ax.broken_barh([(ms(a - A), ms(b - a))], (y + (1 - h) / 2, h), facecolors=col, edgecolors=ec, linewidth=0.5, hatch=hatch)
    if txt and ms(b - a) > 2.5: ax.text(ms((a + b) / 2 - A), y + 0.5, txt, ha="center", va="center", fontsize=fs, color=tc)
yl = []; y = 3 * len(devs)
for d in order:
    y -= 1; yl.append((y, "%s\ncompute" % cname[d]))
    for a, b, k, lab in lanes[d]:
        if k == "W": bar(y, a, b, "#f7c6c6", str(lab), "#900", "///", "#c44", fs=9)
        else:
            col = "#1f77b4" if "MUL_MAT_ID" in lab else "#2ca02c" if not re.search(r"CPY|CONT|CONCAT|GET_ROWS", lab) else "#ff7f0e"
            bar(y, a, b, col, lab.replace(" ", "\n", 1) + "\n%.1f ms" % ms(b - a), "w" if col != "#ff7f0e" else "k")
    y -= 1; yl.append((y, "%s\nH2D (upload)" % cname[d]))
    for a in acts:
        if a[2] == d and a[3] in ("U", "HW") and a[1] > A and a[0] < B:
            bar(y, max(a[0], A), min(a[1], B), "#9467bd" if a[3] == "U" else "#ff7f0e", ("expert upload\n" if a[3] == "U" else "hand-off in\n") + a[4].split(",")[0], "w" if a[3] == "U" else "k", h=0.6)
    y -= 1; yl.append((y, "%s\nD2H (download)" % cname[d]))
    for a in acts:
        if a[2] == d and a[3] == "HR" and a[1] > A and a[0] < B:
            bar(y, max(a[0], A), min(a[1], B), "#ff7f0e", "hand-off out\n" + a[4].split(",")[0], h=0.6)
for i in range(1, len(devs)): ax.axhline(3 * i, color="k", linewidth=1.2)
ax.set_yticks([p + 0.5 for p, _ in yl]); ax.set_yticklabels([t for _, t in yl], fontsize=9)
ax.set_ylim(0, 3 * len(devs)); ax.set_xlim(0, ms(B - A)); ax.grid(axis="x", alpha=0.3)
ax.set_xlabel("ms from the first kernel of layer %d (device time)" % LAYER)
ax.set_title("%s  pass %d  layer %d: %.1f ms   blue = expert matmuls, green = dense, orange = copies between cards / casts, purple = expert uploads, hatched = WAIT n (see table)" % (
    db.split("/")[-1].replace(".sqlite", ""), pi, LAYER, ms(B - A)), fontsize=10)
axt.axis("off")
lines = ["WAITS   n  card    from -   to      ms   what the card waits for, and why"]
for i, (d, x, yv, txt) in enumerate(waits):
    lines.append("      %3d  %-6s %6.1f - %6.1f  %6.1f   %s" % (i + 1, cname[d], ms(x - A), ms(yv - A), ms(yv - x), txt[:190]))
tot = {}
for d, x, yv, txt in waits: tot[d] = tot.get(d, 0) + (yv - x)
lines.append("")
lines.append("waits in this layer: " + ", ".join("%s %.1f ms of %.1f" % (cname[d], ms(tot.get(d, 0)), ms(B - A)) for d in order))
axt.text(0, 1, "\n".join(lines), family="monospace", fontsize=8, va="top", ha="left", transform=axt.transAxes)
plt.tight_layout(); plt.savefig(out, dpi=100)
print("\n".join(lines)); print("wrote", out)
