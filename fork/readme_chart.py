#!/usr/bin/env python3
"""readme_chart.py: speed gain over stock by prompt length for one model, as a README line chart (SVG, follows the reader's light/dark setting) plus the same numbers as a table.
Two lines on one axis (both are "x over stock"): prompt speed and decode speed. Cells come from the prompt-length sweep
  run run -c gainers -onl 'mimo-i$' -ps 8k1k -ps 12k1k ... -b <ours>      and the same with -b <stock>
(campaigns <build>-<model>-<N>k1k-pf<N>k). Per size the second (warm) request counts; a run whose answer looped does not; ours = the automatic arm (no forced dense-split).
Stock is measured at fewer sizes than ours: between its sizes it is interpolated (it is flat), the point is drawn hollow and the table marks it with *.
usage: readme_chart.py [--model mimo] [--title TEXT] [--ours rc5] [--stock stock-b11866-official] [--hw TEXT] [--db PATH] [--out PATH]"""
import argparse, math, os, re, sqlite3, sys
here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mimo", help="the row prefix of the sweep campaigns")
ap.add_argument("--title", default="MiMo-V2.6-Flash 3-bit")
ap.add_argument("--ours", default="rc5")
ap.add_argument("--stock", default="stock-b11866-official")
ap.add_argument("--hw", default="2x RTX 3090 (x16 + x4)")
ap.add_argument("--db", default=os.path.join(here, "runs.db"))
ap.add_argument("--note", default="", help="one more line under the chart (what is partial, what state the box was in)")
ap.add_argument("--out", default=os.path.join(here, "docs", "mimo-gain-by-prompt.svg"))
o = ap.parse_args()


def cells(build):
    """{prompt size in k: (prompt t/s, decode t/s)} of the last good run of each sweep cell of this build"""
    rows = sqlite3.connect(o.db).execute(
        "select campaign, pp, tps, coalesce(loop, 0), coalesce(args, '') from runs where build=? and campaign glob ? and cast(ok as integer)=1 and pp is not null and tps > 0 order by ts",
        (build, f"{build}-{o.model}-[0-9]*k1k-pf[0-9]*k")).fetchall()
    out = {}
    for camp, pp, tps, lp, args in rows:
        if int(float(lp or 0)) or "dense-split" in args: continue   # a looped answer, or a forced arm of the sweep (one card / split): not the automatic mode
        out[int(re.search(rf"-{re.escape(o.model)}-(\d+)k1k-pf", camp)[1])] = (pp, tps)   # ordered by time: the last run is the warm one
    return out


ours, stock = cells(o.ours), cells(o.stock)
if not ours or not stock: sys.exit(f"no sweep cells: ours {sorted(ours)} stock {sorted(stock)} (build {o.ours} / {o.stock}, campaigns <build>-{o.model}-<N>k1k-pf<N>k)")


def stock_at(n):
    """stock's (prompt, decode) at n k: measured, else linear between its neighbouring sizes (flat beyond the ends); the flag says interpolated"""
    if n in stock: return stock[n] + (False,)
    lo, hi = [k for k in sorted(stock) if k < n], [k for k in sorted(stock) if k > n]
    if lo and hi:
        a, b = lo[-1], hi[0]; f = (n - a) / (b - a)
        return tuple(stock[a][i] + f * (stock[b][i] - stock[a][i]) for i in (0, 1)) + (True,)
    return stock[(lo or hi)[-1 if lo else 0]] + (True,)


pts = []
for n in sorted(ours):
    sp, st, derived = stock_at(n)
    pts.append(dict(n=n, op=ours[n][0], ot=ours[n][1], sp=sp, st=st, d=derived, gp=ours[n][0] / sp, gt=ours[n][1] / st))


def fx(g):
    """a gain as 21x / 2.1x / 2x"""
    return f"{g:.0f}x" if g >= 9.5 else f"{g:.1f}".rstrip("0").rstrip(".") + "x"


# ---- the table (the chart's data view)
print(f"| prompt (k tokens) | stock prompt t/s | ours prompt t/s | prompt gain | stock decode t/s | ours decode t/s | decode gain |\n|---|---|---|---|---|---|---|")
for p in pts:
    m = "*" if p["d"] else ""
    print(f"| {p['n']}k | {p['sp']:.0f}{m} | {p['op']:.0f} | **{fx(p['gp'])}** | {p['st']:.1f}{m} | {p['ot']:.1f} | {fx(p['gt'])} |")
if any(p["d"] for p in pts): print("\n\\* stock was not run at this size: interpolated between its neighbouring sizes")

# ---- the chart
W, H, L, R, T, B = 760, 440, 58, 132, 96, 84
if o.note: H += 18; B += 18   # room for the note line, the plot keeps its height
pw, ph = W - L - R, H - T - B
x0, x1 = min(p["n"] for p in pts), max(p["n"] for p in pts)
ymax = max(5, math.ceil(max(max(p["gp"], p["gt"]) for p in pts) * 1.12 / 5) * 5)
step = 5 if ymax <= 30 else 10
X = lambda n: L + (n - x0) / max(1, x1 - x0) * pw
Y = lambda g: T + ph - g / ymax * ph
e = []
e.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" class="c" role="img" aria-labelledby="t d">')
e.append(f'<title id="t">{o.title}: prompt and decode speed over stock by prompt length</title>')
peak = max(pts, key=lambda p: p["gp"])
e.append(f'<desc id="d">Prompt speed over stock peaks at {fx(peak["gp"])} with a {peak["n"]}k prompt; decode speed is {fx(pts[-1]["gt"])} at {pts[-1]["n"]}k. The table below the chart has every value.</desc>')
e.append("""<style>
.c{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--axis:#c9c8c3;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme: dark){.c{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2f2f2d;--axis:#4a4a47;--s1:#3987e5;--s2:#d95926}}
.c text{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;fill:var(--ink)}
.t2{fill:var(--ink2)!important}.ti{font-size:16px;font-weight:600}.sm{font-size:12px}.lb{font-size:13px;font-weight:600}
.g{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}
.l1{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.l2{fill:none;stroke:var(--s2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.m1{fill:var(--s1);stroke:var(--surface);stroke-width:2}.m2{fill:var(--s2);stroke:var(--surface);stroke-width:2}
.h1{fill:var(--surface);stroke:var(--s1);stroke-width:2.5}.h2{fill:var(--surface);stroke:var(--s2);stroke-width:2.5}
</style>""")
e.append(f'<rect width="{W}" height="{H}" rx="8" fill="var(--surface)"/>')
e.append(f'<text x="{L}" y="28" class="ti">{o.title}: speed over stock by prompt length</text>')
e.append(f'<text x="{L}" y="48" class="sm t2">{o.hw}, automatic mode against stock, second request of each cell, 1k-token answer</text>')
# legend (two series: always present) - a line key + dot per series, the words in ink
for i, (cls, name) in enumerate((("1", "prompt speed over stock"), ("2", "decode speed over stock"))):
    lx = L + i * 240
    e.append(f'<line x1="{lx}" y1="72" x2="{lx + 22}" y2="72" class="l{cls}"/><circle cx="{lx + 11}" cy="72" r="5" class="m{cls}"/><text x="{lx + 30}" y="76" class="sm">{name}</text>')
for g in range(0, ymax + 1, step):   # hairline grid, tick labels in secondary ink
    e.append(f'<line x1="{L}" y1="{Y(g):.1f}" x2="{L + pw}" y2="{Y(g):.1f}" class="{"ax" if g == 0 else "g"}"/><text x="{L - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" class="sm t2">{g}x</text>')
for p in pts:
    e.append(f'<text x="{X(p["n"]):.1f}" y="{T + ph + 20}" text-anchor="middle" class="sm t2">{p["n"]}k</text>')
e.append(f'<text x="{L + pw / 2:.0f}" y="{T + ph + 44}" text-anchor="middle" class="sm t2">prompt length (thousand tokens)</text>')
for cls, key in (("2", "gt"), ("1", "gp")):   # decode first, prompt on top
    e.append(f'<path class="l{cls}" d="' + " ".join(("M" if i == 0 else "L") + f"{X(p['n']):.1f} {Y(p[key]):.1f}" for i, p in enumerate(pts)) + '"/>')
    for p in pts:
        what = "prompt" if key == "gp" else "decode"
        s, ours_v, stock_v = ("t/s", p["op"] if key == "gp" else p["ot"], p["sp"] if key == "gp" else p["st"])
        tip = f'{p["n"]}k prompt, {what}: ours {ours_v:.1f} {s}, stock {stock_v:.1f} {s}{" (interpolated)" if p["d"] else ""} = {fx(p[key])}'
        e.append(f'<circle cx="{X(p["n"]):.1f}" cy="{Y(p[key]):.1f}" r="5" class="{"h" if p["d"] else "m"}{cls}"><title>{tip}</title></circle>')
last = pts[-1]
if peak is not last: e.append(f'<text x="{X(peak["n"]):.1f}" y="{Y(peak["gp"]) - 12:.1f}" text-anchor="middle" class="lb">{fx(peak["gp"])}</text>')
for key, name in (("gp", "prompt"), ("gt", "decode")):   # direct labels at the line ends: name and value
    e.append(f'<text x="{X(last["n"]) + 12:.1f}" y="{Y(last[key]) + 4:.1f}" class="lb">{name} {fx(last[key])}</text>')
if o.note: e.append(f'<text x="{L}" y="{T + ph + 84}" class="sm t2">{o.note}</text>')
if any(p["d"] for p in pts):
    e.append(f'<text x="{L}" y="{T + ph + 66}" class="sm t2">hollow marker: stock was run at neighbouring sizes only, its speed here is interpolated</text>')
e.append("</svg>")
os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
open(o.out, "w").write("\n".join(e) + "\n")
print(f"\nwrote {o.out}", file=sys.stderr)
