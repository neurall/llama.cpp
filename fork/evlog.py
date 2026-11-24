#!/usr/bin/env python3
# Reader of the dev event log (ggml/src/ggml-evlog-fork.h): a file (/dev/shm/evlog-<pid>.tsv) or a live server (GET /evlog).
#   summary: per (category, name): count, total ms, p50/p99 us, share of the wall window; per card: busy share, idle gaps
#   --html OUT: per-token map, one row per decode call (token), one lane per card ('gpu graph' spans) and per host thread, colored by name
import argparse, glob, html, json, os, re, sys, urllib.request
from collections import defaultdict


def load(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src.rstrip("/") + "/evlog?since=0", timeout=10) as r:
            text = r.read().decode()
    else:
        text = open(src).read()
    recs = []
    for ln in text.splitlines():
        if not ln or ln[0] == "#":
            continue
        f = ln.split("\t")
        if len(f) < 7:
            continue
        recs.append((int(f[0]), int(f[1]), f[2], f[3], int(f[4]), int(f[5]), int(f[6])))
    recs.sort()
    return recs


def lane(r):   # a card's GPU spans by device (a1), host spans by thread id
    return f"card{r[6]}" if r[2] in ("gpu", "kern", "memcpy") else f"tid{r[4]}"


def summary(recs):
    spans = [r for r in recs if r[1] > r[0]]
    if not spans:
        print("no spans"); return
    t0, t1 = min(r[0] for r in spans), max(r[1] for r in spans)
    wall = (t1 - t0) / 1e6
    g = defaultdict(list)
    for r in spans:
        g[(r[2], r[3])].append((r[1] - r[0]) / 1e3)
    print(f"window {wall:.1f} ms, {len(recs)} records")
    print(f"{'cat':8} {'name':40} {'count':>7} {'total ms':>10} {'p50 us':>9} {'p99 us':>9} {'% wall':>7}")
    for (c, n), d in sorted(g.items(), key=lambda kv: -sum(kv[1])):
        d.sort()
        print(f"{c:8} {n[:40]:40} {len(d):7d} {sum(d)/1e3:10.2f} {d[len(d)//2]:9.1f} {d[min(len(d)-1, int(len(d)*.99))]:9.1f} {100*sum(d)/1e3/wall:7.1f}")
    cards = defaultdict(list)
    for r in spans:
        if r[2] == "gpu":
            cards[r[6]].append((r[0], r[1]))
    for dev, s in sorted(cards.items()):
        s.sort()
        busy, gaps, end = 0, [], s[0][0]
        for a, b in s:
            if a > end:
                gaps.append((a - end) / 1e3)
            busy += max(0, b - max(a, end)); end = max(end, b)
        gaps.sort()
        span = (s[-1][1] - s[0][0]) / 1e6
        print(f"card {dev}: busy {busy/1e6:.1f} of {span:.1f} ms ({100*busy/1e6/span if span else 0:.0f}%), {len(gaps)} idle gaps, p50 {gaps[len(gaps)//2] if gaps else 0:.1f} us, "
              f"max {gaps[-1] if gaps else 0:.1f} us, sum {sum(gaps)/1e3:.1f} ms")


CK = (10, 100, 500, 1000, 4000)   # user 11-24: windows of the prompt and of the answer, tokens (prev, ck]; a prompt files under the largest ck <= its length
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_times.db")
CARD_CATS = ("gpu", "kern", "memcpy")   # records whose a1 is the device


def card_lane(r):
    return f"card{r[6]}"


def requests(recs, n_prompt=0, n_gen=0, pp=0):
    """[(prompt_t0, prompt_tokens, [start of each answer token's period ..., end])]. Ours: llama_decode calls (a prompt call, tokens > 1, after one-token
    calls starts a new request). Any other engine (stock, Strata: only evkern records): the answer tokens are the device-to-host copies of the size seen
    most often (the token or logits read back once a token), requests split by gaps over 2 s, prompt = first kernel of the request to the first token"""
    dec = [r for r in recs if r[3] == "llama_decode"]
    out = []
    if dec:
        cur = None
        for r in dec:
            if r[5] > 1:
                if cur is None or cur[2]: cur = [r[0], 0, []]; out.append(cur)
                cur[1] += r[5]
            elif cur is not None:
                cur[2].append(r[0])
        return out
    d2h = [r for r in recs if r[2] == "memcpy" and r[3] == "DtoH"]
    if not d2h: return out
    cnt = defaultdict(int)
    for r in d2h: cnt[r[5]] += 1
    big = [b for b in cnt if cnt[b] >= 0.9 * n_gen] if n_gen else []
    size = min(big, key=cnt.get) if big else max(cnt, key=cnt.get)   # the copy made once a token: at least one answer's length (a server may run several)
    ts = sorted(r[1] for r in d2h if r[5] == size)
    if len(ts) > 2:   # copies of that size in a burst (prompt-time reads) are one boundary: merge those closer than a fifth of the median spacing
        med = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))[(len(ts) - 1) // 2]
        ts = [t for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > med / 5]
    kern = sorted(r[0] for r in recs if r[2] == "kern")
    import bisect
    groups = [[ts[0]]]
    for t in ts[1:]:
        (groups[-1].append(t) if t - groups[-1][-1] < 2 * 10**9 else groups.append([t]))
    prev_end = 0
    for g in groups:
        i = bisect.bisect_right(kern, prev_end)
        t0 = kern[i] if i < len(kern) else g[0]
        out.append([t0, n_prompt or (round(pp * (g[0] - t0) / 1e9) if pp else 0), g]); prev_end = g[-1]   # no prompt count from the engine: the row's prompt rate
    return out


def window(recs_sorted, a, b, ntok):
    """slices of [a, b) per token: {(lane, name): [us, calls]}; card lanes: busy/IDLE from kernels (or graph spans when no kernel records), k:<kernel>, c:<copy>"""
    tot = defaultdict(lambda: [0.0, 0]); cards = defaultdict(list); haskern = set()
    for r in recs_sorted:
        if r[0] >= b: break
        x0, x1 = max(r[0], a), min(r[1], b)
        if x1 <= x0 or r[3] == "llama_decode": continue
        if r[2] in CARD_CATS:
            ln = card_lane(r)
            nm = "k:" + r[3] if r[2] == "kern" else "c:" + r[3] if r[2] == "memcpy" else r[3]
            if r[2] == "kern": haskern.add(ln)
            cards[ln].append((x0, x1, r[2]))
        else:
            ln, nm = f"tid{r[4]}", r[3]
        tot[(ln, nm)][0] += x1 - x0; tot[(ln, nm)][1] += 1
    per = (b - a) / ntok / 1e3
    for ln, sp in cards.items():
        src = "kern" if ln in haskern else "gpu"
        sp = sorted((x0, x1) for x0, x1, c in sp if c == src); end = a; u = 0
        for x0, x1 in sp:
            if x1 > end: u += x1 - max(x0, end); end = x1
        tot[(ln, "busy")] = [u, 0]; tot[(ln, "IDLE")] = [(b - a) - u, 0]
    return {k: (v[0] / ntok / 1e3, v[1] / ntok) for k, v in tot.items()}, per


def rle(recs_sorted, a, b):
    """one token's kernel sequence per card, consecutive repeats folded: name[mean us]*n (user 11-24: fusion candidates and what a build removed)"""
    seq = defaultdict(list)
    for r in recs_sorted:
        if r[0] >= b: break
        if r[2] == "kern" and r[0] >= a: seq[card_lane(r)].append((r[3], (r[1] - r[0]) / 1e3))
    out = {}
    for ln, ks in seq.items():
        parts, i = [], 0
        while i < len(ks):
            j = i
            while j < len(ks) and ks[j][0] == ks[i][0]: j += 1
            us = sum(x[1] for x in ks[i:j]) / (j - i)
            parts.append(f"{ks[i][0]}[{us:.1f}us]" + (f"*{j - i}" if j - i > 1 else ""))
            i = j
        out[ln] = f"{len(ks)} kernels: " + " ".join(parts)
    return out


def comp(recs, n_prompt=0, n_gen=0, pp=0):
    """[(req, phase, ck, tokens, period_us, {(lane, name): (us, calls)}, {lane: rle})] for every request: the prompt, then the answer windows of CK"""
    recs = sorted(recs)
    hn = {}
    t = defaultdict(int)
    for r in recs:
        if r[2] not in CARD_CATS: t[r[4]] += r[1] - r[0]
    hn = {f"tid{tid}": f"host{i}" for i, (tid, _) in enumerate(sorted(t.items(), key=lambda kv: -kv[1]))}   # busiest thread first, so runs line up
    out = []
    import bisect
    for k, (t0, npp, tok) in enumerate(requests(recs, n_prompt, n_gen, pp)):
        def add(phase, ck, a, b, n, mid):
            if n <= 0 or b <= a: return
            i = bisect.bisect_left(recs, (a - 10**10,))
            sl, per = window(recs[i:], a, b, n)
            sl = {(hn.get(ln, ln), nm): v for (ln, nm), v in sl.items()}
            seq = rle(recs[bisect.bisect_left(recs, (mid[0],)):], *mid) if mid else {}
            out.append((k, phase, ck, n, per, sl, seq))
        if tok and npp:
            add("in", max([c for c in CK if c <= npp] or [CK[0]]), t0, tok[0], npp, None)
        prev = 0
        for ck in CK:
            if len(tok) < ck + 1: break
            m = (prev + ck) // 2
            add("out", ck, tok[prev], tok[ck], ck - prev, (tok[m], tok[m + 1]))
            prev = ck
    return out


def connect(db=DB):
    import sqlite3
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE IF NOT EXISTS best (id INTEGER PRIMARY KEY, hw TEXT, model TEXT, build TEXT, test TEXT, cats TEXT, phase TEXT, ck INTEGER, ts TEXT, args TEXT,
                                       tokens INTEGER, period_us REAL, commit_ TEXT, improved_ts TEXT, prev_us REAL, arm TEXT DEFAULT '', UNIQUE (hw, model, build, test, arm, cats, phase, ck));
      CREATE TABLE IF NOT EXISTS slice (id INTEGER, lane TEXT, name TEXT, us REAL, calls REAL);
      CREATE TABLE IF NOT EXISTS seq (id INTEGER, lane TEXT, text TEXT);
      CREATE INDEX IF NOT EXISTS i_slice ON slice (id); CREATE INDEX IF NOT EXISTS i_seq ON seq (id);""")
    if "arm" not in [r[1] for r in con.execute("PRAGMA table_info(best)")]:   # 11-24: arm (args + cell --env) joins the key; old rows get their args as arm
        con.executescript("""ALTER TABLE best RENAME TO best_old;
          CREATE TABLE best (id INTEGER PRIMARY KEY, hw TEXT, model TEXT, build TEXT, test TEXT, cats TEXT, phase TEXT, ck INTEGER, ts TEXT, args TEXT,
                             tokens INTEGER, period_us REAL, commit_ TEXT, improved_ts TEXT, prev_us REAL, arm TEXT DEFAULT '', UNIQUE (hw, model, build, test, arm, cats, phase, ck));
          INSERT INTO best SELECT *, coalesce(args, '') FROM best_old; DROP TABLE best_old;""")
    return con


def store(recs, meta, db=DB):
    """per (hw, model, build, test, categories, phase, window) only the best run is kept (user 11-24: a later run of the same build that is faster, after a fix
    elsewhere, replaces it); the first request of the run only (later ones run warm caches). Returns [(phase, ck, period, kept)]"""
    con = connect(db); res = []
    for req, phase, ck, n, per, sl, seq in comp(recs, meta.get("n_prompt") or 0, int(float(meta.get("n_gen") or 0)), float(meta.get("pp") or 0)):
        if req: continue
        key = (meta["hw"], meta["model"], meta["build"], meta["test"], meta.get("arm") or "", meta["cats"], phase, ck)
        old = con.execute("SELECT id, period_us FROM best WHERE hw=? AND model=? AND build=? AND test=? AND arm=? AND cats=? AND phase=? AND ck=?", key).fetchone()
        if old and old[1] <= per:
            res.append((phase, ck, per, False)); continue
        if old:
            con.execute("DELETE FROM slice WHERE id=?", (old[0],)); con.execute("DELETE FROM seq WHERE id=?", (old[0],)); con.execute("DELETE FROM best WHERE id=?", (old[0],))
        import datetime   # user 11-24: when a better token time happened, and on which commit
        cur = con.execute("INSERT INTO best (hw, model, build, test, arm, cats, phase, ck, ts, args, tokens, period_us, commit_, improved_ts, prev_us) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          key + (meta.get("ts"), meta.get("args"), n, per, meta.get("commit"), datetime.datetime.now().isoformat(timespec="seconds"), old[1] if old else None))
        con.executemany("INSERT INTO slice VALUES (?,?,?,?,?)", [(cur.lastrowid, ln, nm, us, calls) for (ln, nm), (us, calls) in sl.items() if us >= 0.05 or calls >= 0.5])
        con.executemany("INSERT INTO seq VALUES (?,?,?)", [(cur.lastrowid, ln, t) for ln, t in seq.items()])
        res.append((phase, ck, per, True))
    con.commit(); con.close()
    return res


def load_pid(pid):
    recs = []
    for f in (f"/dev/shm/evlog-{pid}.tsv", f"/dev/shm/evk-{pid}.tsv"):
        if os.path.exists(f): recs += load(f)
    return recs


def after_run(d):
    """run.py calls this with every stored run while the event log is on (-ev): the biggest process's logs (the server; probes leave small ones) go into
    token_times.db; kept as /dev/shm/evlog-last.tsv + evk-last.tsv for --html, every other log of the run deleted (RAM)"""
    fs = [f for f in glob.glob("/dev/shm/evlog-*.tsv") + glob.glob("/dev/shm/evk-*.tsv") if not f.endswith("-last.tsv")]
    if not fs: return
    size = defaultdict(int)
    for f in fs: size[re.search(r"-(\d+)\.tsv$", f).group(1)] += os.path.getsize(f)
    pid = max(size, key=size.get)
    if size[pid] < 100_000: return   # run.py stores more than one row per cell; only the server's log is a run (probes leave ~100 bytes)
    m = re.search(r"prompt_n=(\d+)", str(d.get("note") or ""))
    meta = {"ts": d.get("ts"), "hw": d.get("hw"), "model": re.sub(r"-0000\d-of-0000\d", "", str(d.get("model") or "").split(" ")[0]).removesuffix(".gguf"),
            "build": d.get("build"), "commit": d.get("commit"), "test": d.get("test"), "args": d.get("args"),
            "arm": " ".join(x for x in ((d.get("args") or "").strip(), os.environ.get("PERF_CELL_ENV", "").strip()) if x),   # 1-card (--env CUDA_VISIBLE_DEVICES=0) and 2-card rows apart
            "cats": os.environ.get("GGML_EVLOG", ""), "n_prompt": int(m.group(1)) if m else 0, "n_gen": d.get("n_gen"), "pp": d.get("pp")}
    try:
        res = store(load_pid(pid), meta)
        print("token_times: " + ", ".join(f"{ph}{ck} {per:.0f}us{' (best)' if kept else ''}" for ph, ck, per, kept in res), file=sys.stderr)
    except Exception as e:
        print("token_times:", e, file=sys.stderr)
    for f in fs:
        if re.search(rf"-{pid}\.tsv$", f): os.replace(f, re.sub(rf"-{pid}\.tsv$", "-last.tsv", f))
        elif os.path.exists(f): os.remove(f)


def show(rows, top=15):
    for req, phase, ck, n, per, sl, seq in rows:
        print(f"\nrequest {req} {phase} {'window to ' + str(ck) if phase == 'out' else 'prompt ' + str(n)} tokens: {per:.1f} us/token = {1e6/per:.1f} {'o/s' if phase == 'out' else 'i/s'}")
        print(f"  {'lane':7} {"name":90} {'us/token':>9} {'calls':>6} {'%':>5}")
        for (ln, nm), (us, calls) in sorted(sl.items(), key=lambda kv: (not kv[0][1] in ("busy", "IDLE"), -kv[1][0]))[:top + 4]:
            print(f"  {ln:7} {nm[:90]:90} {us:9.1f} {calls:6.1f} {100*us/per:5.1f}")
        for ln, t in sorted(seq.items()):
            print(f"  {ln} one token: {t[:600]}{' ...' if len(t) > 600 else ''}")


def diff(model, a, b, db=DB, phase=None, top=15, hw=None):
    """best rows of two builds (or engines: stock, strata) per (test, phase, window), slices sorted by size of change: the regressions and their offenders,
    kernels whose calls a build removed (fused)"""
    con = connect(db)
    q = "SELECT id, build, test, phase, ck, period_us FROM best WHERE model LIKE ? AND build IN (?, ?)" + (" AND phase=?" if phase else "") + (" AND hw=?" if hw else "")
    g = defaultdict(dict)
    for i, bld, test, ph, ck, per in con.execute(q, (f"%{model}%", a, b) + ((phase,) if phase else ()) + ((hw,) if hw else ())):
        g[(test, ph, ck)][bld] = (i, per)
    for key in sorted(g):
        if len(g[key]) < 2: continue
        (ia, pa), (ib, pb) = g[key][a], g[key][b]
        sa = {(ln, nm): (us, c) for ln, nm, us, c in con.execute("SELECT lane, name, us, calls FROM slice WHERE id=?", (ia,))}
        sb = {(ln, nm): (us, c) for ln, nm, us, c in con.execute("SELECT lane, name, us, calls FROM slice WHERE id=?", (ib,))}
        ka = sum(c for (ln, nm), (us, c) in sa.items() if nm.startswith("k:")); kb = sum(c for (ln, nm), (us, c) in sb.items() if nm.startswith("k:"))
        print(f"\n{key[0]} {key[1]} {key[2]}: {a} {pa:.1f} us/token, {b} {pb:.1f} us/token ({100*(pb-pa)/pa:+.1f}%); kernels/token {ka:.0f} -> {kb:.0f}")
        print(f"  {'lane':7} {"name":90} {a[:10]:>10} {b[:10]:>10} {'delta us':>9} {'calls':>11}")
        for k in sorted(set(sa) | set(sb), key=lambda k: -abs(sb.get(k, (0, 0))[0] - sa.get(k, (0, 0))[0]))[:top]:
            ua, ca = sa.get(k, (0, 0)); ub, cb = sb.get(k, (0, 0))
            print(f"  {k[0]:7} {k[1][:60]:60} {ua:10.1f} {ub:10.1f} {ub-ua:+9.1f} {ca:5.0f}>{cb:<5.0f}")


def gaps(db=DB, top=5):
    """session start (user 11-24): per model and window, our best build vs the best other engine, biggest gap first, with the slices that make it"""
    con = connect(db)
    rows = con.execute("SELECT hw, model, test, phase, ck, build, period_us, id FROM best").fetchall()
    g = defaultdict(list)
    for hw, mo, test, ph, ck, bld, per, i in rows: g[(hw, mo, ph, ck)].append((per, bld, i, test))
    out = []
    for key, v in g.items():
        ours = [x for x in v if not re.match(r"(stock|strata)", str(x[1]))]; other = [x for x in v if re.match(r"(stock|strata)", str(x[1]))]
        if ours: out.append((min(ours)[0] / min(other)[0] if other else 0, key, min(ours), min(other) if other else None))
    kpt = lambda i: con.execute("SELECT coalesce(sum(calls), 0) FROM slice WHERE id=? AND name LIKE 'k:%'", (i,)).fetchone()[0]
    for ratio, (hw, mo, ph, ck), o, x in sorted(out, key=lambda t: -t[0]):
        print(f"{hw} {mo} {ph}{ck}: ours {o[1]} {o[0]:.0f} us/token, {kpt(o[2]):.0f} kernels/token" + (f"; {x[1]} {x[0]:.0f} us/token, {kpt(x[2]):.0f} kernels/token ({100*(ratio-1):+.0f}%)" if x else "; no other engine"))
        for ln, nm, us, c in con.execute("SELECT lane, name, us, calls FROM slice WHERE id=? AND (name LIKE 'k:%' OR name = 'IDLE') ORDER BY us DESC LIMIT ?", (o[2], top)):   # card idle + biggest kernels: what to cut or fuse
            print(f"    {ln:7} {nm[:80]:80} {us:9.1f} us {c:6.1f} calls")


def page(recs, out, max_tokens):
    # one row per llama_decode call: every span inside its window, drawn on that row's own time axis (0 = decode start)
    dec = [r for r in recs if r[3] == "llama_decode"][-max_tokens:]
    names = sorted({r[3] for r in recs if r[1] > r[0] and r[3] != "llama_decode"})
    pal = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    col = {n: pal[i % len(pal)] for i, n in enumerate(names)}
    spans = [r for r in recs if r[1] > r[0] and r[3] != "llama_decode"]
    lanes = sorted({lane(r) for r in spans}, key=lambda s: (not s.startswith("card"), s))
    rows = []
    j = 0
    for d in dec:
        while j < len(spans) and spans[j][1] < d[0]:
            j += 1
        k, inside = j, []
        while k < len(spans) and spans[k][0] <= d[1]:
            if spans[k][1] >= d[0]:
                r = spans[k]
                inside.append([lanes.index(lane(r)), (max(r[0], d[0]) - d[0]) / 1e3, (min(r[1], d[1]) - d[0]) / 1e3, r[3], r[5]])
            k += 1
        rows.append({"t": (d[1] - d[0]) / 1e3, "n": d[5], "s": inside})
    data = json.dumps({"lanes": lanes, "col": col, "rows": rows})
    legend = "".join(f'<span style="background:{c}">&nbsp;&nbsp;&nbsp;</span> {html.escape(n)} &nbsp; ' for n, c in col.items())
    with open(out, "w") as f:
        f.write("""<!doctype html><meta charset=utf-8><title>Per-token time map</title>
<style>:root{--bg:#fff;--fg:#222}@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ddd}}body{background:var(--bg);color:var(--fg);font:12px system-ui;margin:16px}
canvas{width:100%;border:1px solid #8884}#tip{position:fixed;pointer-events:none;background:#000c;color:#fff;padding:3px 6px;border-radius:3px;display:none}</style>
<h3>Per-token time map</h3><p>One block per decode call (token), newest last. Inside a block: one lane per card (GPU spans) and per host thread; x = microseconds from the call's start. Hover for names.</p>
<p>""" + legend + """</p><canvas id=c></canvas><div id=tip></div><script>
const D=""" + data + """;const L=D.lanes.length,LH=6,RH=L*LH+4,cv=document.getElementById('c'),W=cv.clientWidth,H=D.rows.length*RH+20;
cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;cv.style.height=H+'px';const g=cv.getContext('2d');g.scale(devicePixelRatio,devicePixelRatio);
const tmax=Math.max(...D.rows.map(r=>r.t)),LW=60,sx=(W-LW-60)/tmax;g.font='10px system-ui';
D.rows.forEach((r,i)=>{const y=i*RH;g.fillStyle='#8882';g.fillRect(LW,y,r.t*sx,RH-2);g.fillStyle=getComputedStyle(document.body).color;g.fillText(r.t.toFixed(0)+' us',LW+r.t*sx+4,y+RH-4);
 r.s.forEach(s=>{g.fillStyle=D.col[s[3]]||'#888';g.fillRect(LW+s[1]*sx,y+s[0]*LH,Math.max(1,(s[2]-s[1])*sx),LH-1);});});
D.lanes.forEach((l,k)=>{g.fillStyle=getComputedStyle(document.body).color;if(D.rows.length)g.fillText(l,2,k*LH+LH);});
const tip=document.getElementById('tip');cv.onmousemove=e=>{const b=cv.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top,i=Math.floor(y/RH),r=D.rows[i];
 if(!r){tip.style.display='none';return}const ln=Math.floor((y-i*RH)/LH),t=(x-LW)/sx;const s=r.s.find(s=>s[0]==ln&&t>=s[1]&&t<=s[2]);
 tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';
 tip.textContent='token '+i+' ('+r.t.toFixed(0)+' us)'+(s?' | '+D.lanes[s[0]]+': '+s[3]+' '+(s[2]-s[1]).toFixed(1)+' us a0='+s[4]:'');};
</script>""")
    print(f"map: {out} ({len(rows)} tokens, {len(lanes)} lanes)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["diff"]:
        ap = argparse.ArgumentParser(prog="run.py evlog diff", description="per-token composition of two builds (or stock/strata) from token_times.db, biggest changes first")
        ap.add_argument("model"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--phase", choices=("in", "out")); ap.add_argument("--top", type=int, default=15); ap.add_argument("--hw")
        x = ap.parse_args(argv[1:]); return diff(x.model, x.a, x.b, phase=x.phase, top=x.top, hw=x.hw)
    if argv[:1] == ["gaps"]:
        return gaps()
    ap = argparse.ArgumentParser(prog="run.py evlog", description="subcommands: diff MODEL A B, gaps")
    ap.add_argument("src", nargs="*", help="evlog/evk files or a server URL (default: /dev/shm/*-last.tsv)")
    ap.add_argument("--html", help="write the per-token map here")
    ap.add_argument("--tokens", type=int, default=300, help="last N decode calls in the map")
    ap.add_argument("--prompt", type=int, default=0, help="prompt tokens (engines without our log)")
    a = ap.parse_args(argv)
    src = a.src or [f for f in ("/dev/shm/evlog-last.tsv", "/dev/shm/evk-last.tsv") if os.path.exists(f)]
    if not src:
        sys.exit("no event log (run with -ev CATS)")
    recs = sorted(sum((load(f) for f in src), []))
    summary(recs)
    show(comp(recs, a.prompt))
    if a.html:
        page(recs, a.html, a.tokens)


if __name__ == "__main__":
    main()
