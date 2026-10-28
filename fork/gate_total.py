#!/usr/bin/env python3
"""gate_total.py CAMPAIGN_PREFIX CANDIDATE RELEASE: the release gate on TOTAL request time, not on decode or prompt speed alone.
Total time of a run = prompt seconds (the server's own, the note's prompt_s) + generated tokens / decode speed, divided by the tokens it handled (prompt tokens + tokens really generated):
ms per token, so builds whose answers differ in length compare fairly (user, 2026-10-19). A cell passes when the candidate's mean total,
less the usual spread, is not more than 20% above the release's: (cand_mean - spread) <= 1.20 x rel_mean, spread = 2 x the larger standard deviation of the two sets
of runs (the run-to-run noise on that cell). Why: a build that streams a long prompt several times faster loses a little decode speed while the cache refills,
which a 1k-in / short-out cell shows as a drop; what a user feels is how long the whole request takes (user, 2026-10-19).
Runs are compared only under the same power limits (the pl column; day 220/280 W, night 200/200 W). Cells without a prompt_s in the note use n_gen / decode speed only (short prompts: the prompt part is under a second)."""
import re, sqlite3, statistics, sys
pre, cand, rel = sys.argv[1], sys.argv[2], sys.argv[3]
# SINCE (optional 4th argument, 'YYYY-MM-DD HH:MM'): only runs from then on. The release's older runs may come from another hardware state
# (GPU unlock 2026-10-27 11:08: tensor math 1.3-1.5x faster) or carry a since-fixed bug (DeepSeek token-0 answers before dsfix1), so a gate reruns both builds and reads only those rows
since = sys.argv[4] if len(sys.argv) > 4 else ""
c = sqlite3.connect("/p/bw/wt-main/fork/runs.db")
rows = c.execute("select campaign, build, tps, pp, n_gen, note, pl, pw_avg, out_chars, loop from runs where build in (?, ?) and tps > 0 and ts >= ?", (cand, rel, since)).fetchall()
# run.py run names a cell's campaign <build>-<row>-<test>: the build prefix comes off, so the candidate's and the release's runs of a row meet in one cell
# (campaigns named the same for both builds, e.g. gate-..., stay as they are); PREFIX then filters the cell names
rows = [((camp[len(b) + 1:] if camp.startswith(b + "-") else camp), b) + tuple(rest) for camp, b, *rest in rows]
rows = [r for r in rows if r[0].startswith(pre)]
# power limits differ by time of day (day 220/280 W, night 200/200 W): only runs under the SAME limits are compared, per cell the limits of the candidate's runs
cand_pl = {}
for camp, b, tps, pp, n_gen, note, pl, pw, oc, lp in rows:
    if b == cand: cand_pl.setdefault(camp, set()).add(pl)
rows = [r for r in rows if r[0] in cand_pl and r[6] in cand_pl[r[0]]]
pls = {}
outs = {}   # per cell and build: (answer characters, looping flag) of every run: a build whose answers loop or stop short is not timed on the same text
watts = {}   # per cell and build: speed per watt (pp and decode over the average power of all GPUs: pw_avg 'a/b' W), to compare numbers measured under different limits
cells = {}
for camp, b, tps, pp, n_gen, note, pl, pw, oc, lp in rows:
    outs.setdefault(camp, {}).setdefault(b, []).append((float(oc or 0), int(float(lp or 0))))
    pls.setdefault(camp, {}).setdefault(b, set()).add(pl)
    try: w = sum(float(x) for x in (pw or '').split('/'))
    except ValueError: w = 0.0
    if w > 0: watts.setdefault(camp, {}).setdefault(b, []).append((pp / w if pp else 0.0, tps / w))
    m = re.search(r"prompt_s=([0-9.]+)", note or "")
    ps = float(m.group(1)) if m else 0.0
    mn = re.search(r"prompt_n=([0-9]+)", note or "")
    pn = int(mn.group(1)) if mn else 0
    if not n_gen or not tps: continue
    cells.setdefault(camp, {}).setdefault(b, []).append((ps + n_gen / tps) / (pn + n_gen) * 1000.0)   # ms per token over the input plus the tokens really generated
bad = 0
print(f"{'cell':36} {'rel pl':>9} {'cand pl':>9} {'rel ms/tok':>9} {'cand ms/tok':>8} {'ratio':>6} {'spread':>7} {'adj ratio':>9}  verdict")
for camp in sorted(cells):
    d = cells[camp]
    if cand not in d or rel not in d: continue
    r, k = d[rel], d[cand]
    rm, km = statistics.mean(r), statistics.mean(k)
    sd = max(statistics.pstdev(r) if len(r) > 1 else 0.0, statistics.pstdev(k) if len(k) > 1 else 0.0)
    spread = 2 * sd
    adj = (km - spread) / rm
    ok = adj <= 1.20
    bad += not ok
    wr, wk = watts.get(camp, {}).get(rel, []), watts.get(camp, {}).get(cand, [])
    wtxt = ""
    if wr and wk:
        mean = lambda xs, i: statistics.mean(x[i] for x in xs)
        wtxt = f"  per W: pp {mean(wk, 0):.2f}/{mean(wr, 0):.2f}  t/s {mean(wk, 1):.3f}/{mean(wr, 1):.3f} (cand/rel)"
    plr, plk = "/".join(sorted(map(str, pls.get(camp, {}).get(rel, [])))), "/".join(sorted(map(str, pls.get(camp, {}).get(cand, []))))
    orr, okk = outs.get(camp, {}).get(rel, []), outs.get(camp, {}).get(cand, [])
    otxt = (f" out chars {statistics.mean(x[0] for x in okk):.0f}/{statistics.mean(x[0] for x in orr):.0f} loops {sum(x[1] for x in okk)}/{sum(x[1] for x in orr)}" if orr and okk else "")
    print(f"{camp:36} {plr:>9} {plk:>9} {rm:10.2f} {km:11.2f} {km / rm:6.2f} {spread:7.2f} {adj:9.2f}  {'ok' if ok else 'SLOWER'}  (n {len(r)}/{len(k)}){wtxt}{otxt}")
print(f"\n{bad} cell(s) over the 20% total-time line")
sys.exit(1 if bad else 0)
