#!/usr/bin/env python3
"""crosstest.py: did the planner detect and take every strategy crossover a request visits, and what did a wrong or missing switch cost?

`run.py test cross [-b BUILD ...] [--since TS] [--models m1,m2] [--all]` (this module, also runnable alone) reads the stats.log of every cell run.py kept
(runs.db table `stats`, one zlib blob per cell) and judges, per model and build, the five crossovers of fork/docs/planner-crossovers.md:

  X1 misses on the CPU vs uploaded (the chains; the band of small calls, 64-512 tokens, and the decode's first tokens)
  X2 stream vs the chains / k picks ahead (predicted oversampling)
  X3 stream alone vs stream + mix (the slower link carries a share)
  X4 one card vs the dense split
  X5 dense split vs dense + expert split

Evidence is what the processes of that build measured themselves: every DON line is one measurement of its arm at its call size (`prompt call of N tokens:
ARM took T ms`; ARM `dense` = the dense split took the call), the PRE line carries the card-level pick with the one-card and split rates the planner
knew, the TOK lines the decode's ms per token. Measurements are compared within one build, one size class (sqrt-2 steps) and one temperature (the first
request of a cell is cold, the rest warm). A crossover is
  PASS       every pick within 5% of the best arm measured there (the 5% yardstick of planner-crossovers.md), or the lost time under 5% of the prompt time,
  FAIL x%    a measured arm was faster: x = prompt seconds lost over the prompt seconds of the calls that met the crossover (trials included, counted),
  unverified no other arm was ever measured at those sizes (the planner never tried it): nothing proves the pick; the forced cell to run is named.
The matrix (models x X1..X5) and one line per model and crossover go to stdout; the exit code is the number of FAIL cells (a release build stops on it).
Example: python3 -I fork/crosstest.py -b ram-pool -b pinpool-default. Numbers are seconds and tokens per second; nothing here is a knob."""
import argparse, collections, math, os, re, sqlite3, statistics, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(os.path.realpath(os.path.join(HERE, "runs.db"))), "stats.db")   # the kept stats.logs (run.py runlog STATS_DB)
YARD = 0.05   # a pick within 5% of the measured best passes (planner-crossovers.md: the replay gate's yardstick)
XO = ["X1 cpu-vs-upload", "X2 stream-vs-chain/k", "X3 stream-vs-mix", "X4 one-vs-split", "X5 split-vs-expert"]
KNOWN = ("glm30", "glm35", "mimo", "ds", "qwen36", "qni4", "qni3", "qn1m", "olmoe")
ONE = ("stream", "mix", "k", "chain", "dec")   # the one-card arms (DON path names map onto them); "dense" is the card-level arm
TO_RUN = {XO[0]: "pf512 with --moe pp-path=dec | chain | stream", XO[1]: "the gate size with --moe pp-path=chain and pp-path=stream,pp-k=1|2",
          XO[2]: "the gate size with --moe pp-path=mix", XO[3]: "the gate size with --moe dense-split=0 and dense-split=1", XO[4]: "the gate size with --moe dense-split=1,prefill-split=1"}

def model_key(name):
    """the short model name of a gguf file name or a cell tag; '' when unknown"""
    s = name.lower()
    if "glm35" in s or ("glm" in s and "3.5" in s): return "glm35"
    if "glm" in s: return "glm30"
    if "mimo" in s: return "mimo"
    if "deepseek" in s or re.search(r"(^|[-_])ds([-_]|$)", s): return "ds"
    if "qwen3.6" in s or "qwen36" in s or re.search(r"(^|[-_])q35([-_]|$)", s): return "qwen36"
    if "iq1_m" in s or re.search(r"(^|[-_])qn1([-_]|$)", s): return "qn1m"
    if "iq3_s" in s or "qni3" in s: return "qni3"
    if "iq4_xs" in s or "qni4" in s or "qwen3.8" in s or "qwen38" in s: return "qni4"
    if "olmoe" in s: return "olmoe"
    return ""

def arm_class(p):
    """the DON/CAL path name -> arm"""
    if p.startswith("dense"): return "dense"
    if p == "decode chain": return "dec"
    if p == "prompt chain": return "chain"
    if ", mix" in p: return "mix"
    if ", k x" in p: return "k"
    return "stream"

def size_class(n):
    """the planner's call size classes are sqrt-2 steps: two classes per octave"""
    return int(round(2 * math.log2(max(1, n))))

num = lambda pat, s, d=0.0: float(m.group(1)) if (m := re.search(pat, s)) else d

def parse(lines):
    """one stats.log -> (requests, warm-up calls). A request: calls (n, arm, path, took_ms, trial), passes (the PRE fields), tok (ms per token per 10), swt (decode miss modes)"""
    reqs, cur, cal, pre_calls = [], None, {}, []
    for l in lines:
        p = l.split(None, 4)
        if len(p) < 5: continue
        t, s = p[3], p[4].rstrip("\n")
        if t == "REQ" and s.startswith("begin"):
            cur = dict(n_prompt=int(num(r"begin: (\d+)", s)), calls=[], passes=[], tok=[], swt=collections.Counter(), prompt_s=0.0)
        elif t == "REQ" and s.startswith("end") and cur:
            reqs.append(cur); cur = None
        elif t == "CAL" and s.startswith("prompt call of"):
            cal[int(num(r"prompt call of (\d+)", s))] = "(trial" in s
        elif t == "DON" and s.startswith("prompt call of"):
            m = re.match(r"prompt call of (\d+) tokens: (.+?) took (\d+) ms", s)
            if not m: continue
            n, path = int(m.group(1)), m.group(2)
            (cur["calls"] if cur else pre_calls).append(dict(n=n, path=path, arm=arm_class(path), took=float(m.group(3)), trial=cal.pop(n, False)))
        elif t == "PRE" and cur is not None:
            f = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', s))
            cur["passes"].append(dict(n=int(f.get("prompt_tokens", "0")), mode=f.get("mode", ""), tps=float(f.get("prompt_tps", "0") or 0), one=f.get("one_card_tps", "-"), split=f.get("split_tps", "-"),
                                      expert=float(f.get("helper_expert_share", "0") or 0), trial="trial" in f.get("chosen_by", "")))
        elif t == "TOK" and cur is not None:
            cur["tok"].append(num(r"^10 tokens: ([\d.]+) ms", s))
        elif t == "SWT" and cur is not None and s.startswith("decode") and "misses go to" in s:
            cur["swt"][re.search(r"misses go to ([A-Za-z ]+?) \(", s).group(1)] += 1
        elif t == "PHA" and cur is not None and "prefill -> decode" in s:
            cur["prompt_s"] = num(r"tokens ([\d.]+) s", s)
    return reqs, pre_calls

def judge(cells):
    """cells of one model and build: list of (tag, reqs, pre_calls) -> {X: (verdict, loss_pct, n_bad, text)}, plus the decode-start line under key 'decode'"""
    out = {}
    meas = collections.defaultdict(list)   # (size class, cold/warm, arm) -> ms per token of every measured call: what the processes of this build knew
    calls = []                              # (call, cw)
    for tag, reqs, pre in cells:
        for c in pre: calls.append((c, "c")); meas[(size_class(c["n"]), "c", c["arm"])].append(c["took"] / c["n"])
        for i, r in enumerate(reqs):
            cw = "c" if i == 0 else "w"
            for c in r["calls"]:
                calls.append((c, cw)); meas[(size_class(c["n"]), cw, c["arm"])].append(c["took"] / c["n"])
    def best(c, cw, arms):
        """the fastest arm among `arms` measured at c's size and temperature: (arm, ms for c's n) or None"""
        b = None
        for a in arms:
            v = meas.get((size_class(c["n"]), cw, a))
            if v and (b is None or statistics.median(v) * c["n"] < b[1]): b = (a, statistics.median(v) * c["n"])
        return b
    def score(X, pick_in, alt_of, min_n, never):
        """every call whose arm is in pick_in against the best measured arm of alt_of(arm); loss over the prompt time of those calls"""
        lost = tot = 0.0; bad = 0; n = 0; ev = []; any_alt = False
        for c, cw in calls:
            if c["n"] < min_n or c["arm"] not in pick_in: continue
            n += 1; tot += c["took"]
            cand = best(c, cw, alt_of(c["arm"]))
            if cand is None: continue
            any_alt = True
            if cand[1] < c["took"] * (1 - YARD):
                bad += 1; lost += c["took"] - cand[1]
                ev.append(f"{c['n']}{cw} {c['path']} {c['took'] / 1e3:.1f} s, {cand[0]} measured {cand[1] / 1e3:.1f}{' (trial)' if c['trial'] else ''}")
        if n == 0: return ("unverified", 0.0, 0, "no call met this crossover; " + never)
        if not any_alt: return ("unverified", 0.0, 0, never + f" ({n} calls, only the picked arm measured)")
        pct = 100 * lost / max(tot, 1e-9)
        if pct > 100 * YARD: return ("FAIL", pct, bad, f"{bad} of {n} calls lost {lost / 1e3:.1f} of {tot / 1e3:.0f} s: " + "; ".join(ev[:4]))
        return ("PASS", pct, bad, f"{n} calls, {bad} lost more than {100 * YARD:.0f}% ({lost / 1e3:.1f} of {tot / 1e3:.0f} s)" + ("; " + "; ".join(ev[:2]) if ev else ""))
    out[XO[1]] = score(XO[1], ONE, lambda a: ("k", "chain", "dec") if a in ("stream", "mix") else ("stream", "mix"), 1000, "no chain or k arm ever measured at these sizes: the planner never tried one (R5: trial only)")
    out[XO[2]] = score(XO[2], ONE, lambda a: ("mix",) if a != "mix" else ("stream",), 1000, "mix never measured at these sizes: the slower link's share was never tried")
    # X4: the dense split's calls against the one-card arms and the other way round (DON), plus the picks the planner made against its own one/split buckets (PRE)
    x4 = score(XO[3], ONE + ("dense",), lambda a: ("dense",) if a != "dense" else ONE, 1000, "the other card arm never ran at these sizes")
    knew = []   # passes that ran the arm the planner's own buckets, as they stood BEFORE the pass (the PRE line is written after it), said was slower
    prior = {}
    for tag, reqs, pre in cells:
        for i, r in enumerate(reqs):
            for p in r["passes"]:
                one = float(p["one"]) if p["one"] not in ("-", "") else None; spl = float(p["split"]) if p["split"] not in ("-", "") else None
                k = (size_class(p["n"]), "c" if i == 0 else "w"); b4 = prior.get(k)
                if b4 and b4[0] and b4[1] and p["n"] >= 1000:
                    mine = b4[1] if p["mode"].startswith("dense") else b4[0]; other = b4[0] if p["mode"].startswith("dense") else b4[1]
                    if other > mine * (1 + YARD): knew.append(f"{p['n']} {p['mode']} {p['tps']:.0f} t/s, known before: one {b4[0]:.0f} / split {b4[1]:.0f}{' (trial)' if p['trial'] else ''}")
                prior[k] = (one or (b4[0] if b4 else None), spl or (b4[1] if b4 else None))
    if knew and x4[0] != "FAIL": x4 = ("FAIL", x4[1], x4[2] + len(knew), f"{len(knew)} passes ran the slower arm the planner's own buckets knew: " + "; ".join(knew[:3]) + ("; " + x4[3] if x4[0] == "PASS" else ""))
    elif knew: x4 = (x4[0], x4[1], x4[2] + len(knew), x4[3] + "; known-slower passes: " + "; ".join(knew[:2]))
    out[XO[3]] = x4
    # X5: dense + expert split against the dense split at the same size (PRE rates)
    modes = collections.defaultdict(list)
    for tag, reqs, pre in cells:
        for i, r in enumerate(reqs):
            for p in r["passes"]:
                if p["n"] >= 1000 and p["tps"] > 0: modes[(size_class(p["n"]), "c" if i == 0 else "w", p["mode"])].append(p["tps"])
    pairs = [(k[0], k[1], statistics.median(modes[k]), statistics.median(modes[(k[0], k[1], "dense+expert-split")])) for k in modes if k[2] == "dense-split" and (k[0], k[1], "dense+expert-split") in modes]
    if not any(k[2] == "dense+expert-split" for k in modes): out[XO[4]] = ("unverified", 0.0, 0, "dense + expert split never ran (R6: a trial only from the split; the split " + ("ran" if any(k[2] == "dense-split" for k in modes) else "never ran") + ")")
    elif not pairs: out[XO[4]] = ("unverified", 0.0, 0, "dense + expert split ran, the plain split never at the same size: no comparison; " + "; ".join(f"{int(2 ** (k[0] / 2))}{k[1]} {k[2]} {statistics.median(v):.0f} t/s" for k, v in list(modes.items())[:3]))
    else:
        worst = max(pairs, key=lambda x: abs(x[2] - x[3]) / max(x[2], x[3]))
        loss = max(0.0, 1 - min(worst[2], worst[3]) / max(worst[2], worst[3])) * 100
        out[XO[4]] = ("PASS" if loss <= 100 * YARD else "FAIL", loss, 0 if loss <= 100 * YARD else 1, "; ".join(f"{int(2 ** (s / 2))}{cw}: split {d:.0f} t/s, with experts {e:.0f}" for s, cw, d, e in pairs[:3]))
    # X1: the decode's start, where the misses' CPU-vs-upload decisions and the cache refill after the prompt live: the decode time lost in the ramp against the
    # steady state (the second half of the decode) is what better first-token decisions could recover at most; the prompt calls of the chains' band (64-512
    # tokens) are judged like X2 when a cell has them
    first, steady, ramp, total, swt = [], [], 0.0, 0.0, collections.Counter()
    for tag, reqs, pre in cells:
        for r in reqs:
            t = [x for x in r["tok"] if x > 0]
            if len(t) >= 30:
                st = statistics.median(t[len(t) // 2:]); first.append(statistics.mean(t[:10])); steady.append(st)
                ramp += sum(max(0.0, x - st) for x in t[:len(t) // 2]) * 10; total += sum(t) * 10
            swt.update(r["swt"])
    sw = ", ".join(f"{k} {100 * v / max(1, sum(swt.values())):.0f}%" for k, v in swt.most_common(3))
    small = [(c, cw) for c, cw in calls if 16 <= c["n"] <= 512 and c["arm"] in ONE]
    lost = tot = 0.0; bad = 0; ev = []
    for c, cw in small:
        tot += c["took"]; cand = best(c, cw, tuple(x for x in ONE if x != c["arm"]))
        if cand and cand[1] < c["took"] * (1 - YARD): bad += 1; lost += c["took"] - cand[1]; ev.append(f"{c['n']}{cw} {c['path']} {c['took']:.0f} ms, {cand[0]} measured {cand[1]:.0f}")
    band = (f"; band 64-512: {len(small)} calls, {bad} lost >5% ({lost / 1e3:.1f} of {tot / 1e3:.1f} s): " + "; ".join(ev[:2])) if small else "; no prompt call in the band 64-512 (pf512 would show the chains' crossover)"
    if not first: out[XO[0]] = ("unverified", 0.0, bad, "no decode window" + band)
    else:
        pct = 100 * ramp / max(total, 1e-9)
        out[XO[0]] = ("FAIL" if pct > 100 * YARD or lost > YARD * max(tot, 1e-9) else "PASS", pct, bad,
                      f"decode ramp: first 100 tokens {statistics.mean(first):.1f} ms per token vs steady {statistics.mean(steady):.1f} ({100 * (statistics.mean(first) / statistics.mean(steady) - 1):+.0f}%), "
                      f"{pct:.1f}% of the decode time above steady; misses went to: {sw}" + band)
    out["decode"] = f"{len(first)} decodes" if first else "no decode window"
    return out

def load(db, builds, since, models):
    """stats rows -> {(model, build): [(ts, tag, reqs, pre_calls)]}"""
    q, args = "select ts, build, tag, log from stats where 1=1", []
    if builds: q += " and build in (%s)" % ",".join("?" * len(builds)); args += builds
    if since: q += " and ts >= ?"; args.append(since)
    groups = collections.defaultdict(list)
    for ts, build, tag, blob in db.execute(q + " order by ts", args):
        m = model_key(tag)
        if not m or (models and m not in models): continue
        try: reqs, pre = parse(zlib.decompress(blob).decode(errors="replace").splitlines())
        except Exception: continue
        if reqs: groups[(m, build)].append((ts, tag, reqs, pre))
    return groups

def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-b", "--build", action="append", default=[], help="builds whose cells are judged (repeatable); default: every model's newest build")
    ap.add_argument("--since", default="", help="only cells from this time on, 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--models", default="", help="comma list of short names: " + ",".join(KNOWN))
    ap.add_argument("--all", action="store_true", help="every (model, build) of the window, not only each model's newest build")
    a = ap.parse_args(argv)
    groups = load(sqlite3.connect(DB), a.build, a.since, {m.strip() for m in a.models.split(",") if m.strip()})
    if not groups:
        print("no stats.log kept for these builds (run.py keeps one per cell in runs.db table stats)"); return 2
    if not a.all and not a.build:   # each model's newest build only
        newest = {}
        for (m, b), lst in groups.items():
            if m not in newest or max(x[0] for x in lst) > newest[m][0]: newest[m] = (max(x[0] for x in lst), b)
        groups = {k: v for k, v in groups.items() if newest[k[0]][1] == k[1]}
    fails = 0; rows = []
    for (m, b) in sorted(groups):
        cells = [(tag, reqs, pre) for _, tag, reqs, pre in groups[(m, b)]]
        v = judge(cells)
        print(f"\n{m} on {b}: {len(cells)} cells, {sum(len(r) for _, r, _ in cells)} requests; decode {v['decode']}")
        for X in XO:
            verdict, loss, bad, text = v[X]
            fails += verdict == "FAIL"
            print(f"  {X:22s} {verdict:10s} {('%.0f%%' % loss) if verdict == 'FAIL' else '':5s} {text}" + (f"\n{'':40s} to verify: {TO_RUN[X]}" if verdict == "unverified" else ""))
        rows.append((m, b, v))
    print("\nmatrix: PASS = every pick within 5% of the best arm measured there; FAIL n% = prompt time lost (bad picks in brackets); unverified = the other arm never measured")
    print(f"{'model':7s} {'build':22s} " + " ".join(f"{x.split()[0]:>14s}" for x in XO))
    for m, b, v in rows:
        print(f"{m:7s} {b[:22]:22s} " + " ".join(f"{(('%s %.0f%% (%d)' % (vd, loss, bad)) if vd == 'FAIL' else ('PASS' if vd == 'PASS' else 'unverified')):>14s}" for vd, loss, bad, _ in (v[X] for X in XO)))
    print(f"\n{fails} FAIL cells. Unverified cells need the forced arm at the same size on the same build (planner-crossovers.md section 5); the lines above name the cell per crossover.")
    return fails

if __name__ == "__main__":
    sys.exit(min(run(), 125))
