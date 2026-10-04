#!/usr/bin/env python3
"""Update the README speed table from the new tests (campaign newtests in run-history.csv).
usage: readme-update.py README HISTORY.csv OURS_BUILD        e.g. readme-update.py README.md tools/bench/run-history.csv release-b12209

Table (one row per test): | model | machine | in VRAM | test | stock t/s | ours t/s | stock pp | ours pp | gain t/s | gain pp | build |
Rows with test short4 (game4) or long4 (edit4) are filled; machine A = pc1 (2x RTX 3090), B = pc3 (laptop), C = pc2 (CPU only).
A row without a model name continues the model above.
Each side's best run over all its runs in the log (t/s and pp separately); stock = the stock build, or the same build with --fork off
(campaign newtests-forkoff) when stock cannot load the model; ours = OURS_BUILD only.
A row is replaced unless ours is clearly slower than the number already in it (below 0.9x): slightly lower numbers of the new build are fine.
A model with results for a test and machine it has no row for gets one, appended to its rows, when its gain is not clearly below the model's best row (0.9x).
Groups of rows are sorted by the t/s gain of their first row, best first ("fits VRAM" last)."""
import csv, re, sys
readme, hist, OURS = sys.argv[1], sys.argv[2], sys.argv[3]
LABELS = {   # README model label -> file name stem in the run log
    "**MiMo** IQ3_XXS 132G": "MiMo-V2.6-Flash-RL-IQ3_XXS",
    "**GLM** 3.5-bit 137G": "GLM-5.3-Flash-GSQ-RCO-3.5bit",
    "**GLM** 3.0-bit 117G": "GLM-5.3-Flash-GSQ-RCO-3.0bit",
    "**Qwen Next** IQ4_XS 88G": "Qwen3.8-Flash-Next-UD-IQ4_XS",
    "**Qwen Next** IQ3_S 83G": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
    "**Qwen Next** IQ1_M 55G": "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M",
    "**Qwen3.6** Q2_0 11G": "Qwen3.6-35B-A3B-GSQ-hybrid",
    "27B IQ4_NL dense": "Qwen3.8-27B-IQ4_NL",
    "27B IQ3_S dense": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp",
    # the 27B with MTP is left out: its stock and ours runs must both use --spec-type draft-mtp
}
MACHINES = {"pc1": "A", "pc3": "B", "pc2": "C"}
FULL = {"A": "2x3090 128G", "B": "4060 32G", "C": "CPU 64G"}   # machine cell text (no letter)
LETTER = {v: k for k, v in FULL.items()}
TESTS = {"short4": "game4", "long4": "edit4"}
SHORT = OURS.removeprefix("release-")

best = {}
for r in csv.DictReader(open(hist)):
    try:
        tps, pp = float(r["tps"]), float(r["pp"] or 0)
    except ValueError:
        continue
    mach = MACHINES.get(r["hw"])
    if not tps or r["ok"] == "0" or not mach or r["test"] not in TESTS.values():
        continue
    stem = re.sub(r"-0000\d-of-0000\d", "", r["model"].split(" ")[0]).removesuffix(".gguf")
    stem = "Qwen3.6-35B-A3B-GSQ-hybrid" if stem == "qwen36" else stem
    b = r["build"]
    if r["campaign"] == "newtests-forkoff" and b.startswith(OURS):
        side = "standin"
    elif r["campaign"] != "newtests":
        continue
    elif b.startswith("stock"):
        side = "stock"
    elif b.startswith(OURS):
        side = "ours"
    else:
        continue
    v = best.setdefault((stem, mach, r["test"], side), [0.0, 0.0])
    v[0] = max(v[0], tps); v[1] = max(v[1], pp)

def pair(stem, mach, test):
    s = best.get((stem, mach, test, "stock")) or best.get((stem, mach, test, "standin"))
    o = best.get((stem, mach, test, "ours"))
    return (s, o) if s and o and s[0] and o[0] else None

def gain(o, s):
    return "-" if not (o and s) else f"{o/s:.1f}x"

def num(x):
    m = re.search(r"([\d.]+)", x.replace("*", ""))
    return float(m.group(1)) if m else None

def row(model, mach, test, s, o, vram="-"):
    short = test == "short4"   # a prompt of a few tokens: pp is measured but not comparable, marked ? instead of bold
    bt = o[0] >= 1.1 * s[0]
    bp = bool(s[1]) and o[1] >= 1.1 * s[1] and not short
    ours_t = f"**{o[0]:.1f}**" if bt else f"{o[0]:.1f}"
    ours_p = ("? " if short else "") + (f"**{o[1]:.0f}**" if bp else f"{o[1]:.0f}")
    gt = gain(o[0], s[0]); gt = f"**{gt}**" if bt else gt
    gp = ("? " if short else "") + gain(o[1], s[1])
    return f"| {model} | {FULL.get(mach, mach)} | {vram} | {test} | {s[0]:.1f} | {ours_t} | {s[1]:.0f} | {ours_p} | {gt} | {gp} | {SHORT} |"

lines = open(readme).read().split("\n")
hi = next(i for i, l in enumerate(lines) if l.startswith("| model | machine | in VRAM | test |"))
j = hi + 2
while j < len(lines) and lines[j].startswith("|"):
    j += 1
groups, cur = [], None
for l in lines[hi + 2:j]:
    model = l.split("|")[1].strip()
    if model or cur is None:
        cur = [model, []]; groups.append(cur)
    cur[1].append(l)

changed = 0
for model, rows in groups:
    stem = LABELS.get(model)
    if not stem:
        continue
    have, mach_above, have_m = {}, "", {}
    for k, l in enumerate(rows):
        c = [x.strip() for x in l.split("|")]
        mach_above = LETTER.get(c[2], mach_above) if c[2] else mach_above   # an empty machine cell continues the machine above
        have_m[l] = mach_above
        if c[4] in TESTS:
            have[(mach_above, c[4])] = k
    best_gain = max((num(l.split("|")[9]) or 0) for l in rows)
    for mach in ("A", "B", "C"):
        for test, t in TESTS.items():
            p = pair(stem, mach, t)
            if not p:
                continue
            s, o = p
            if (mach, test) in have:
                k = have[(mach, test)]
                cur_ours = num(rows[k].split("|")[6]) or 0
                if o[0] < 0.9 * cur_ours:
                    continue   # clearly slower than the number in the row: the row stays
                new = row(model if k == 0 else "", mach if (k == 0 or rows[k].split("|")[2].strip()) else "", test, s, o, rows[k].split("|")[3].strip())
                if new != rows[k]:
                    rows[k] = new; changed += 1
            elif o[0] / s[0] >= 0.9 * best_gain:
                rows.append(row("", mach, test, s, o, next((l.split("|")[3].strip() for l in rows if have_m.get(l) == mach), "-"))); changed += 1

def key(g):
    if g[0].startswith("fits VRAM"):
        return -1
    return num(g[1][0].split("|")[9]) or 0
groups.sort(key=key, reverse=True)
lines[hi + 2:j] = [l for g in groups for l in g[1]]
open(readme, "w").write("\n".join(lines))
print("rows updated:", changed)
