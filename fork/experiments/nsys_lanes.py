#!/usr/bin/env python3
"""The life of ONE layer N of a multi-GPU (or one-GPU) prompt pass, from its own expert uploads to its end, every millisecond accounted for.

usage: nsys_lanes.py TRACE.sqlite OUT.png [PASS=last] [LAYER=middle]
       (trace: nsys profile --trace=cuda,nvtx --cuda-graph-trace=node, GGML_NVTX=1; nsys export --type sqlite)

All times are DEVICE time (the host thread issues work about four layers ahead of the cards, host time says nothing), in ms from the start of layer N's first upload.
Panel 1, "experts of layer N arrive": per card the uploads RAM -> card of layer N's own expert tensors (gate, up, down; they run ahead of the layer). A gap
          between two of them is a numbered wait: what the next upload waited for (a staging slot still read by an earlier layer's matmul, or the link).
Panel 2, "layer N computes": from the start of its dense part (attention, router) to the moment the next layer's dense part starts. Per card:
  GPU       dense, expert matmuls (moe gate / up / down), the card's half of a copy between cards, or WAIT n (reason in the table under the chart).
  prefetch  uploads of LATER layers' experts that run on this card's link meanwhile (they can stand in front of a copy between cards).
  hand-off  layer N's activations crossing the cards over this card's link: hidden state, partial sum.
A wait is explained by walking back from the moment the card resumes: whatever ended at that moment (a copy on this card, a copy or a kernel on another
card) is what the card waited for; then the same question for that activity's start, until the start of the wait is reached. So a wait of 40 ms reads as
"x16 computing dense 31 ms, x16 receiving the partial sum 1 ms, x16 link busy with its own upload 2 ms, x16 computing moe down 6 ms".
Kernels are joined to the scheduler split that issued them (NVTX range "SPLIT n CARD OP name" holding the launch call), which names the layer and the phase.
The totals table adds up to the layer period for every card."""
import sqlite3, sys, bisect, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
EPS = 0.30e6          # two events closer than this are "the same moment"
MIN_WAIT = 0.30e6     # shorter gaps between kernels are launch gaps, summed as such

# ---- the pass ------------------------------------------------------------------------------------------------------------------------------------------------
p0 = [r[0] for r in c.execute("select start from NVTX_EVENTS where text like 'SPLIT 0 %' order by start")]
passes = []
for i, s in enumerate(p0):
    e = p0[i + 1] if i + 1 < len(p0) else s + 5e9
    n, last = c.execute("select count(*), max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'", (s, e)).fetchone()
    if n >= 20 and last - s > 100e6:
        passes.append((s, e))
if not passes:
    sys.exit("no prompt pass in the trace")
spans = [c.execute("select max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'", (a, b)).fetchone()[0] - a for a, b in passes]
longest = max(range(len(passes)), key=lambda i: spans[i])
pi = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ("last", "auto") else (longest if spans[longest] > 3 * sorted(spans)[len(spans) // 2] else len(passes) - 1)
S, E_ = passes[pi]
sp = c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start", (S, E_)).fetchall()
sp_start = [x[0] for x in sp]
def split_of(t):
    i = bisect.bisect_right(sp_start, t) - 1
    return i if i >= 0 else None
sp_card, sp_op, sp_name = [], [], []
for st, en, tx in sp:
    m = re.match(r"SPLIT \d+ (\S+) (\S+) ?(.*)", tx)
    sp_card.append(m.group(1)); sp_op.append(m.group(2)); sp_name.append(m.group(3))
rt = {cid: t for cid, t in c.execute("select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?", (S - 20e6, E_))}   # the first uploads of a pass are issued just before its first split
K = []
for st, en, d, strm, cid, nm in c.execute("select k.start,k.end,k.deviceId,k.streamId,k.correlationId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<? order by k.start", (S, E_ + 3e9)):
    if cid in rt:
        K.append((st, en, d, strm, split_of(rt[cid]), nm))
M = []
for st, en, d, strm, cid, ck, b in c.execute("select start,end,deviceId,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<? order by start", (S, E_ + 3e9)):
    if cid in rt and ck in (1, 2):
        M.append((st, en, d, strm, ck, b, split_of(rt[cid])))
if not K:
    sys.exit("no kernels joined to this pass (CUDA graph replay: trace with --cuda-graph-trace=node)")
devs = sorted(set(k[2] for k in K))
card = {}
for d in devs:
    cnt = {}
    for k in K:
        if k[2] == d and k[4] is not None: cnt[sp_card[k[4]]] = cnt.get(sp_card[k[4]], 0) + 1
    card[d] = max(cnt, key=cnt.get) if cnt else "dev%d" % d
owner = max(devs, key=lambda d: sum(k[1] - k[0] for k in K if k[2] == d))
cstream = {}
for d in devs:
    t = {}
    for k in K:
        if k[2] == d: t[k[3]] = t.get(k[3], 0) + k[1] - k[0]
    cstream[d] = max(t, key=t.get)

# ---- phases of the compute stream ----------------------------------------------------------------------------------------------------------------------------
def phase_of(si, idx_in_split, n_in_split, names_before):
    op, nm = sp_op[si], sp_name[si]
    m = re.match(r"ffn_moe_(gate|up|down)-(\d+)", nm)
    if op == "MUL_MAT_ID" and m:
        if "moe_weighted_reduction" in names_before or (n_in_split > 24 and idx_in_split >= 13 and "moe_weighted_reduction" not in names_before and m.group(1) == "down" and idx_in_split >= 14):
            return "dense", "dense"
        return "moe", "moe " + m.group(1)
    if nm.startswith("ffn_moe_out") or n_in_split > 24:
        return "dense", "dense"
    if n_in_split <= 14:
        return "mask", "id mask"
    return "dense", "dense"
by_split = {}
for k in K:
    by_split.setdefault((k[2], k[4]), []).append(k)
runs = {d: [] for d in devs}          # (start, end, cat, label, layer)
for (d, si), ks in by_split.items():
    if si is None: continue
    seen = set(); cur = None
    lay = re.search(r"-(\d+)$", sp_name[si])
    for i, k in enumerate(ks):
        cat, lab = phase_of(si, i, len(ks), seen)
        seen.add(k[5])
        if cur and cur[2] == cat and k[0] - cur[1] < MIN_WAIT:
            cur[1] = max(cur[1], k[1]); cur[5] += k[1] - k[0]
        else:
            if cur: runs[d].append(tuple(cur))
            cur = [k[0], k[1], cat, lab, int(lay.group(1)) if lay else -1, k[1] - k[0]]
    if cur: runs[d].append(tuple(cur))
for d in devs:
    runs[d].sort()
    # name the layer in the label: a dense run issued by the split "...-M" is the add of layer M's partial sums and the attention + router of layer M+1
    runs[d] = [(r[0], r[1], r[2], ("dense of layer %d (attention, router; starts with the add of layer %d)" % (r[4] + 1, r[4]) if r[2] == "dense" and r[4] >= 0 else
                                   "%s-%d" % (r[3], r[4]) if r[2] == "moe" else r[3]), r[4] + 1 if r[2] == "dense" else r[4], r[5]) for r in runs[d]]

# ---- copies ----------------------------------------------------------------------------------------------------------------------------------------------------
mm_splits = {d: [i for i in range(len(sp)) if sp_card[i] == card[d] and sp_op[i] == "MUL_MAT_ID"] for d in devs}
ups = {d: [] for d in devs}; hand = {d: [] for d in devs}
raw_up = {d: [] for d in devs}
for st, en, d, strm, ck, b, si in M:
    if d not in devs: continue
    if ck == 1 and strm != cstream[d] and b > (1 << 20):
        raw_up[d].append((st, en, b))
    else:
        nm = sp_name[si] if si is not None else ""
        what = "hidden state" if "ffn_moe_gate" in nm else "partial sum" if nm.startswith("ffn_moe_out") else "ids and router weights" if b < (1 << 20) else "activation"
        way = "RAM -> %s" % card[d] if ck == 1 else "%s -> RAM" % card[d]
        hand[d].append((st, en, "handoff", "%s %s (%.1f MB)" % (what, way, b / 1e6), what, ck, b, strm == cstream[d]))
# An upload belongs to the expert matmul that reads it: the ring is first in, first out, so the uploads of a card and its MUL_MAT_ID splits match in order.
# Matched from the END of the pass: the first uploads are issued before the first split's range opens and may be missing from the join.
N_EXPERT = int(sys.argv[5]) if len(sys.argv) > 5 else 0      # experts per layer, to print id ranges as ids instead of percent
tmp = {d: [] for d in devs}; by_tensor = {}
for d in devs:
    off = len(mm_splits[d]) - len(raw_up[d])
    for i, (st, en, b) in enumerate(raw_up[d]):
        tgt = mm_splits[d][i + off] if 0 <= i + off < len(mm_splits[d]) else None
        lab = sp_name[tgt].replace("ffn_moe_", "") if tgt is not None else "?"
        tmp[d].append((st, en, b, lab)); by_tensor.setdefault(lab, {})[d] = b
slow_first = sorted(devs, key=lambda d: sorted(b / max(1, en - st) for st, en, b in raw_up[d])[len(raw_up[d]) // 2] if raw_up[d] else 1e9)
def id_range(lab, d):
    """the expert ids of tensor `lab` that card d holds, as text"""
    parts = by_tensor.get(lab, {}); tot = float(sum(parts.values())) or 1.0
    lo = 0.0
    for x in slow_first:
        if x not in parts: continue
        hi = lo + parts[x] / tot
        if x == d:
            return ("ids %d-%d" % (round(lo * N_EXPERT), round(hi * N_EXPERT) - 1)) if N_EXPERT else ("experts %.0f-%.0f%%" % (100 * lo, 100 * hi))
        lo = hi
    return ""
for d in devs:
    for st, en, b, lab in tmp[d]:
        ups[d].append((st, en, "upload", "upload %s %s (%.0f MB)" % (lab, id_range(lab, d), b / 1e6), lab, id_range(lab, d), b))

# a readable name per card: role and measured upload rate of its expert uploads (median), e.g. "CUDA1 owner 26 GB/s"
def _rate(d):
    r = sorted(b / max(1, en - st) for st, en, b in raw_up[d])   # bytes per ns = GB/s
    return r[len(r) // 2] if r else 0.0
bare = dict(card)
for d in devs:
    card[d] = "%s %s %.0f GB/s" % (bare[d], "owner" if d == owner else "helper", _rate(d)) if _rate(d) > 0 else "%s %s" % (bare[d], "owner" if d == owner else "helper")

# ---- layer N ---------------------------------------------------------------------------------------------------------------------------------------------------
own_runs = runs[owner]
dense_start = {}                                                                    # layer -> start of its dense part on the owner (the first of its kernel runs)
for r in own_runs:
    if r[2] == "dense" and r[4] >= 0 and (r[4] not in dense_start or r[0] < dense_start[r[4]]): dense_start[r[4]] = r[0]
moe_layers = sorted(set(r[4] for r in own_runs if r[2] == "moe"))
ok = [L for L in moe_layers if L in dense_start and L + 1 in dense_start]
if len(ok) < 2:
    sys.exit("fewer than 2 whole layers found")
N = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "middle" else ok[len(ok) // 2]
if N not in ok:
    sys.exit("layer %d is not a whole layer of this pass (%d..%d)" % (N, ok[0], ok[-1]))
Bs, Be = dense_start[N], dense_start[N + 1]
up_n = {d: sorted(u for u in ups[d] if re.search(r"-%d$" % N, u[4])) for d in devs}
all_up = [u for d in devs for u in up_n[d]]
T0 = min([u[0] for u in all_up] + [Bs])            # time zero: the first upload of layer N
As, Ae = T0, max([u[1] for u in all_up] + [T0])

# ---- what did a card wait for ------------------------------------------------------------------------------------------------------------------------------------
acts = []   # (start, end, dev, kind, label)
for d in devs:
    for r in runs[d]: acts.append((r[0], r[1], d, "K", r[3]))
    for u in ups[d]:  acts.append((u[0], u[1], d, "U", u[3]))
    for h in hand[d]: acts.append((h[0], h[1], d, "H", h[3]))
acts.sort(key=lambda a: a[1])
ends = [a[1] for a in acts]
def explain(d, g0, g1, lane):
    """segments (a, b, text, own_copy) covering [g0, g1]: why card d's compute stream (lane 'K') or upload stream (lane 'U') did not run"""
    segs = []; t = g1; me = card[d]
    for _ in range(60):
        if t <= g0 + 1e3: break
        lo = bisect.bisect_left(ends, t - EPS); hi = bisect.bisect_right(ends, t + EPS)
        cand = [a for a in acts[lo:hi] if a[0] < t - 1e3 and not (a[2] == d and a[3] == lane)]
        if not cand:
            j = lo - 1
            while j >= 0 and (acts[j][2] == d and acts[j][3] == lane): j -= 1
            a = max(g0, acts[j][1]) if j >= 0 else g0
            segs.append((a, t, "nothing running on any card: launch and event latency of the driver", False))
            t = a; continue
        def prio(a):
            if a[2] == d and a[3] == "H": return 0
            if a[2] == d: return 1
            if a[3] == "H": return 2
            if a[3] == "K": return 3
            return 4
        a = min(cand, key=lambda x: (prio(x), -x[1]))
        other = card[a[2]]; own = False
        if lane == "U":
            if a[2] == d and a[3] == "K":   txt = "staging slot not free: %s still reads it (%s); the upload starts when that matmul ends" % (me, a[4])
            elif a[2] == d and a[3] == "H": txt = "%s link busy with a copy between cards: %s" % (me, a[4])
            elif a[3] == "K":               txt = "waits for %s: computing %s" % (other, a[4])
            else:                           txt = "waits for %s: %s" % (other, a[4])
        elif a[2] == d and a[3] == "H":
            txt = "%s: its own half of the copy, %s" % (me, a[4]); own = True
        elif a[2] == d and a[3] == "U": txt = "%s link busy with its own %s (prefetch); the incoming copy is queued behind it: copies of one card run one after the other" % (me, a[4])
        elif a[3] == "H":               txt = "waits for %s: %s over the %s link" % (other, a[4], other)
        elif a[3] == "K":               txt = "waits for %s: computing %s%s" % (other, a[4], "; %s holds no dense weights, only the experts are shared" % me if "dense" in a[4] else "")
        else:                           txt = "waits for %s: its link busy with its own %s (prefetch), a copy between cards is queued behind it" % (other, a[4])
        b0 = max(g0, a[0])
        segs.append((b0, t, txt, own)); t = b0
    if t > g0 + 1e3: segs.append((g0, t, "not explained", False))
    return segs[::-1]

waits = []                                  # [card, from, to, text]
def wait_no(d, x, y, txt):
    waits.append([d, x, y, txt]); return len(waits)

# panel 1: the uploads of layer N
p1 = {}
for d in devs:
    lane = []; t = None
    for u in up_n[d]:
        if t is not None and u[0] - t > MIN_WAIT:
            for (x, y, txt, _) in explain(d, t, u[0], "U"):
                if y - x >= 0.05e6: lane.append((x, y, "wait", wait_no(d, x, y, "upload of layer %d: " % N + txt)))
        lane.append((u[0], u[1], "upload", u[4])); t = u[1]
    p1[d] = lane

# panel 2: layer N computes
p2 = {}; totals = {d: {} for d in devs}
def add_tot(d, key, v): totals[d][key] = totals[d].get(key, 0) + v
for d in devs:
    gpu = []; t = Bs
    rs = [r for r in runs[d] if r[1] > Bs and r[0] < Be]
    for r in rs + [(Be, Be, "end", "", -1, 0)]:
        a = max(r[0], Bs)
        if a - t > MIN_WAIT:
            for (x, y, txt, own) in explain(d, t, a, "K"):
                if y - x < 0.05e6:
                    add_tot(d, "gaps under 0.3 ms between kernels and copies", y - x)
                elif own:
                    gpu.append((x, y, "copy", txt)); add_tot(d, "copy between cards, own half", y - x)
                else:
                    gpu.append((x, y, "wait", wait_no(d, x, y, txt))); add_tot(d, "WAIT " + re.sub(r" \([\d.]+ MB\)", "", txt).split(";")[0], y - x)
        elif a > t:
            add_tot(d, "gaps under 0.3 ms between kernels and copies", a - t)
        if r[2] == "end": break
        b = min(r[1], Be); whole = r[0] >= Bs and r[1] <= Be
        gpu.append((a, b, r[2], r[3]))
        add_tot(d, "compute " + r[3].split(" (")[0], r[5] if whole else b - a)
        if whole: add_tot(d, "gaps under 0.3 ms between kernels and copies", (b - a) - r[5])
        t = b
    p2[d] = gpu

# ---- text ------------------------------------------------------------------------------------------------------------------------------------------------------
ms = lambda v: v / 1e6
head1 = "%s  pass %d of %d  layer %d  owner %s" % (db.split("/")[-1].replace(".sqlite", ""), pi + 1, len(passes), N, card[owner])
print(head1)
print("experts of layer %d arrive %.1f .. %.1f ms; its dense part starts at %.1f ms; the layer ends at %.1f ms (compute phase %.2f ms)" % (N, 0.0, ms(Ae - T0), ms(Bs - T0), ms(Be - T0), ms(Be - Bs)))
for d in devs:
    print("\n%s (%s): uploads of layer %d: %s" % (card[d], "holds the dense weights" if d == owner else "computes its share of the experts", N,
          ", ".join("%s %.2f-%.2f" % (u[4], ms(u[0] - T0), ms(u[1] - T0)) for u in up_n[d]) or "none in this pass"))
    print("  compute phase, %.2f ms accounted of %.2f:" % (ms(sum(totals[d].values())), ms(Be - Bs)))
    for k, v in sorted(totals[d].items(), key=lambda kv: -kv[1]):
        print("  %6.2f ms %5.1f%%  %s" % (ms(v), 100.0 * v / (Be - Bs), k))
    lu = sum(min(u[1], Be) - max(u[0], Bs) for u in ups[d] if u[1] > Bs and u[0] < Be)
    lh = sum(min(h[1], Be) - max(h[0], Bs) for h in hand[d] if h[1] > Bs and h[0] < Be and h[6] >= (1 << 20))
    print("  link in the compute phase: prefetch of later layers %.2f ms, copies between cards %.2f ms, nothing to send %.2f ms" % (ms(lu), ms(lh), ms(Be - Bs - lu - lh)))
print("\nwaits (ms from the first upload of layer %d):" % N)
for i, (d, x, y, txt) in enumerate(waits):
    print("  %2d  %-6s %7.2f - %7.2f  %5.2f ms  %s" % (i + 1, card[d], ms(x - T0), ms(y - T0), ms(y - x), txt))

# ---- picture ---------------------------------------------------------------------------------------------------------------------------------------------------
CARD_COL = ["#17becf", "#e6c229", "#e377c2", "#7f7f7f"]     # one colour per card for its expert uploads: the owner first
COL = {"moe gate": "#1f77b4", "moe up": "#4a9bd9", "moe down": "#8cc3ee", "dense": "#2ca02c", "id mask": "#bcbd22", "copy": "#ff7f0e", "wait": "#f7c6c6",
       "upload": "#9467bd", "prefetch": "#c5b0d5", "handoff": "#ff7f0e"}
order = [owner] + [d for d in devs if d != owner]
n1, n2 = len(devs), 3 * len(devs)
txt_rows = len(waits) + 2 * len(devs) + 5
fig, (ax1, ax2, axt) = plt.subplots(3, 1, figsize=(20, 0.9 * n1 + 1.0 * n2 + 0.17 * txt_rows + 2.2), gridspec_kw={"height_ratios": [0.9 * n1 + 0.4, 1.0 * n2, 0.17 * txt_rows]})
def bar(ax, a, b, y, h, col, text="", tc="k", fs=8, hatch=None, ec="k"):
    ax.broken_barh([(ms(a - T0), ms(b - a))], (y + (1 - h) / 2, h), facecolors=col, edgecolors=ec, linewidth=0.5, hatch=hatch)
    if text: ax.text(ms((a + b) / 2 - T0), y + 0.5, text, ha="center", va="center", fontsize=fs, color=tc)
# panel 1
y = n1; yl = []
for d in order:
    y -= 1; yl.append((y, "%s\nupload of layer %d experts" % (card[d], N)))
    for (a, b, cat, lab) in p1[d]:
        if cat == "wait": bar(ax1, a, b, y, 0.7, COL["wait"], str(lab), "#900", 9, "///", "#c44")
        else:
            u = [x for x in up_n[d] if x[0] == a][0]
            bar(ax1, a, b, y, 0.7, CARD_COL[order.index(d) % 4], "moe %s\n%s\n%.0f MB  %.1f ms" % (lab, u[5], u[6] / 1e6, ms(b - a)), "k", 8)
pad = max(1e6, 0.06 * (Ae - As))
ax1.set_xlim(ms(As - T0 - pad), ms(Ae - T0 + pad)); ax1.set_ylim(0, n1)
ax1.set_yticks([p + 0.5 for p, _ in yl]); ax1.set_yticklabels([t for _, t in yl], fontsize=9); ax1.grid(axis="x", alpha=0.3)
ax1.set_title("%s    1. MoE expert weights of layer %d arrive: each card takes its own id range of the three expert tensors (gate, up, down), RAM -> card, one transfer per tensor; staged %.0f ms before the layer's dense part starts" % (head1, N, ms(Bs - Ae)), fontsize=9.5)
# panel 2
y = n2; yl = []
for d in order:
    for kind in ("GPU", "prefetch\n(later layers)", "hand-off"):
        y -= 1; yl.append((y, "%s\n%s" % (card[d], kind.replace("\n", " "))))
        if kind == "GPU":
            for (a, b, cat, lab) in p2[d]:
                if cat == "wait":   bar(ax2, a, b, y, 0.76, COL["wait"], str(lab) if ms(b - a) > 0.5 else "", "#900", 9, "///", "#c44")
                elif cat == "copy": bar(ax2, a, b, y, 0.76, COL["copy"], "copy\n%.1f" % ms(b - a) if ms(b - a) > 1.5 else "", "k", 7)
                else:
                    short = lab.split(" (")[0]; key = re.sub(r"-\d+$", "", short) if cat == "moe" else cat
                    bar(ax2, a, b, y, 0.76, COL.get(key, "#888"), "%s\n%.1f ms" % (short, ms(b - a)) if ms(b - a) > 2.2 else "", "w" if cat in ("moe", "dense") else "k", 8)
        elif kind.startswith("prefetch"):
            for u in ups[d]:
                if u[1] > Bs and u[0] < Be:
                    a, b = max(u[0], Bs), min(u[1], Be)
                    bar(ax2, a, b, y, 0.6, CARD_COL[order.index(d) % 4], "moe %s" % u[4] if ms(b - a) > 1.6 else "", "k", 7)
        else:
            for h in hand[d]:
                if h[1] > Bs and h[0] < Be and h[6] >= (1 << 20):
                    a, b = max(h[0], Bs), min(h[1], Be)
                    bar(ax2, a, b, y, 0.6, COL["handoff"], (("hidden" if h[4] == "hidden state" else "partial") + (" in" if h[5] == 1 else " out")) if ms(b - a) > 1.0 else "", "k", 7)
ax2.set_xlim(ms(Bs - T0), ms(Be - T0)); ax2.set_ylim(0, n2)
ax2.set_yticks([p + 0.5 for p, _ in yl]); ax2.set_yticklabels([t for _, t in yl], fontsize=9); ax2.grid(axis="x", alpha=0.3)
for i in range(1, len(devs)): ax2.axhline(3 * i, color="k", linewidth=1.0)
ax2.set_xlabel("ms from the start of layer %d's first upload (device time)" % N)
ax2.set_title("2. layer %d computes: %.1f ms   (green = dense: attention + router, on the owner only; blue = MoE expert matmuls on each card's id range; orange = copy between cards; prefetch = MoE weights of later layers; hatched = WAIT n, see table)" % (N, ms(Be - Bs)), fontsize=9.5)
axt.axis("off")
lines = ["WAITS   n  card    from -   to     ms    what it waits for, and why"]
for i, (d, x, yv, txt) in enumerate(waits):
    lines.append("       %2d  %-6s %6.1f - %6.1f  %5.2f  %s" % (i + 1, card[d], ms(x - T0), ms(yv - T0), ms(yv - x), txt[:175]))
lines.append("")
for d in order:
    parts = ["%s %.1f" % (k.replace("compute ", "")[:60], ms(v)) for k, v in sorted(totals[d].items(), key=lambda kv: -kv[1]) if ms(v) >= 0.05]
    lines.append("%s, compute phase %.1f ms = " % (card[d], ms(Be - Bs)) + " + ".join(parts[:4]))
    if len(parts) > 4: lines.append("          + " + " + ".join(parts[4:8]))
axt.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.3, va="top", ha="left", transform=axt.transAxes)
plt.tight_layout()
plt.savefig(out, dpi=105)
print("\nwrote", out)
