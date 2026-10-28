#!/usr/bin/env python3
"""readme_table.py: the README tables from the rt-* cells of fork/experiments/release_table.py (runs.db): best gainers (stock over ours per model and test),
the 35B context table (stock static -c against ours static and auto), the dense split on/off. Median of the valid runs; a run whose answer looped is not counted
(a cell where every run looped shows 'loops'), a refused run shows 'refused'.  usage: readme_table.py [--db PATH]"""
import sqlite3, statistics, sys
db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "/p/bw/wt-main/fork/runs.db"
c = sqlite3.connect(db)
NAMES = [("mimo", "MiMo-V2.6-Flash 3-bit"), ("glm", "GLM-5.3-Flash 3.0-bit"), ("qni4", "Qwen3.8-Flash-Next UD-IQ4_XS"), ("qni3", "Qwen3.8-Flash-Next IQ3_S (78 GB)"),
         ("q27mtp", "Qwen3.8-27B MTP Q5_K_M (both with MTP)"), ("olmoe", "OLMoE-1B-7B Q4_K_M (fits VRAM)"), ("q36", "Qwen3.6-35B-A3B hybrid (fits VRAM)"),
         ("qn1", "Qwen3.8-Flash-Next IQ1_M"), ("q27nl", "Qwen3.8-27B IQ4_NL (dense)"), ("glm35", "GLM-5.3-Flash 3.5-bit (137 GB)")]
TESTS = {"i": "pf12k", "io": "edit4c", "o4": "tetrisc"}   # o4: the tetris prompt (~0.6k in) with a 4096-token answer (was 1024: too short to show the decode side)
def cell(camp, test):
    rows = c.execute("select tps, pp, loop, pl from runs where campaign=? and tps is not null and tps > 0", (f"{camp}-{test}",)).fetchall()
    ok = [r for r in rows if not int(float(r[2] or 0))]
    if not rows:
        n = c.execute("select count(*) from runs where campaign=?", (f"{camp}-{test}",)).fetchone()[0]
        return ("refused" if n else None)
    if not ok:
        return "loops"
    return dict(tps=statistics.median(r[0] for r in ok), pp=statistics.median(r[1] or 0 for r in ok), n=len(ok), pl=ok[-1][3])
def fmt(st, ou, key, d):
    if ou is None: return "missing"
    if isinstance(ou, str): return f"{st if isinstance(st, str) else (f'{st[key]:.{d}f}' if st else '-')} / {ou}"
    o = ou[key]
    if st is None: return f"**{o:.{d}f}**"
    if isinstance(st, str): return f"{st}<br>**{o:.{d}f}**"
    g = o / st[key] if st[key] else 0
    return f"{st[key]:.{d}f}<br>**{o:.{d}f}**&nbsp;<sub>({g:.2f}x)</sub>"
print("<<BEST>>")
print("> **Measured at about half power.** Both GPUs (2x RTX 3090) ran with their power limit lowered from 370 W to 200 W each (220/280 W by day): thermal throttling had set in because of old thermal pads and dry core paste, and the limit also keeps the night runs quiet. Absolute speeds are therefore lower than a well-cooled machine gives. Stock and ours always ran back to back under the same limit, so the gains are comparable.\n")
print("| Model | i: prompt t/s | io: prompt t/s | io: tokens t/s | o: tokens t/s (0.6k in / 4k out) |\n|---|---|---|---|---|")
pls = set()
for n, label in NAMES:
    g = {t: (cell(f"rt-{n}-{t}-stock", TESTS[t]), cell(f"rt-{n}-{t}-ours", TESTS[t])) for t in TESTS}
    for t in g:
        for x in g[t]:
            if isinstance(x, dict): pls.add(x["pl"])
    print(f"| {label} | {fmt(*g['i'], 'pp', 0)} | {fmt(*g['io'], 'pp', 0)} | {fmt(*g['io'], 'tps', 1)} | {fmt(*g['o4'], 'tps', 1)} |")
print(f"\npower limits of these runs (x4 card + x16 card, W): {', '.join(sorted(map(str, pls)))}\n")
print("<<CTX>>")
print("| -c | stock decode (t/s) | ours decode (t/s) | decode gain | stock prompt (t/s) | ours prompt (t/s) | prompt gain |\n|---|---|---|---|---|---|---|")
for cx in (32768, 131072, 262144):
    s, o = cell(f"rt-ctx-{cx}-stock", "pf12k"), cell(f"rt-ctx-{cx}-ours", "pf12k")
    if isinstance(s, dict) and isinstance(o, dict):
        print(f"| {cx} | {s['tps']:.1f} | {o['tps']:.1f} | {o['tps'] / s['tps']:.2f}x | {s['pp']:.0f} | {o['pp']:.0f} | {o['pp'] / s['pp']:.1f}x |")
a = cell("rt-ctx-auto-ours", "pf12k")
if isinstance(a, dict): print(f"| none (ours auto) | n/a | {a['tps']:.1f} | | n/a | {a['pp']:.0f} | |")
print("<<DS>>")
print("| model | dense split off: prompt t/s | dense split auto: prompt t/s | gain |\n|---|---|---|---|")
for n in ("glm", "mimo", "qni4"):
    off, on = cell(f"rt-ds-{n}-off", "pf12k"), cell(f"rt-ds-{n}-on", "pf12k")
    if isinstance(off, dict) and isinstance(on, dict):
        print(f"| {dict(NAMES)[n]} | {off['pp']:.0f} | {on['pp']:.0f} | {on['pp'] / off['pp']:.2f}x |")
