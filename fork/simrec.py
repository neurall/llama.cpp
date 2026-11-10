# Sim replay data (user 2026-11-10: "record new sim replay ds from mimo glm and qni and 35b so we dont need big runs anymore, except for final confirm runs";
# "runsim mimo 60k4k replay and report planner efficiency ... if crossover opportunities were missed or bad choices picked or too soon or too late without
# running big model"; "make sure simrec can resume and collect just whats missing if interrupted"; "storage and query src is stats.db").
#   run.py simrec [MODEL ...] [SIZE ...] [-b BUILD] [run options]   the rows of fork/runs/simrec.csv (one per model, prompt size and strategy: auto and each forced
#       path) through `run.py run simrec`, so every run is a normal row in runs.db and its stats.log lands in stats.db. Only the rows stats.db does not hold
#       complete for BUILD run: an interrupted recording resumes with what is missing. Each cell starts from an empty learned state (-cs): a forced row never
#       teaches the auto row what it measured.
#   run.py runsim MODEL SIZE [-b BUILD]   replays them from stats.db alone (no model, no card)
#   python3 -I fork/simrec.py test        the verdict rules' self-check
import collections, csv, json, os, re, sqlite3, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(HERE, 'runs', 'simrec.csv')
DB = '/p/bw/data/stats.db'
NOISE = 0.05   # paths closer than 5% are a tie: running either one loses nothing


def list_rows():
    """the truth list's rows, each with its model word, strategy and size from the name (sr-q35-auto-2k: q35, auto, 2k)"""
    out = []
    for r in csv.DictReader(open(LIST, newline='')):
        p = r['name'].split('-')
        out.append(dict(r, m=p[1], strat=p[2], size='-'.join(p[3:])))
    return out


def pick_size(word, sizes):
    """the list's size for a word: the same (2k, chat, 64k), else the nearest token count (60k4k: 64k)"""
    if word in sizes:
        return word
    m = re.match(r'(?:pf)?(\d+)k', word)
    num = {z: int(re.match(r'\d+', z).group()) for z in sizes if re.fullmatch(r'\d+k', z)}
    return min(num, key=lambda z: abs(num[z] - int(m.group(1)))) if m and num else None


def select(rows, words):
    """rows of the models and sizes named in words (none named: all); words that name neither are returned as unknown"""
    sizes = sorted({r['size'] for r in rows})
    ms, zs, bad = set(), set(), []
    for w in words:
        if any(w in (r['m'], r['model']) for r in rows):
            ms.add(w)
        elif pick_size(w, sizes):
            zs.add(pick_size(w, sizes))
        else:
            bad.append(w)
    return [r for r in rows if (not ms or ms & {r['m'], r['model']}) and (not zs or r['size'] in zs)], bad


def complete_cells(con, build, rows):
    """row name -> stat id of its newest cell on BUILD whose every request finished its prompt (PHA prefill rows), and name -> (finished, wanted) of the rest"""
    done, part = {}, {}
    for r in rows:
        want = int(r['n'] or 1)
        best = 0
        for sid, in con.execute('SELECT rowid FROM stats WHERE tag = ? ORDER BY rowid DESC', (f"{build}-{r['name']}",)):
            k = con.execute("SELECT COUNT(DISTINCT req) FROM ppev WHERE stat_id = ? AND ev = 'PHA' AND path = 'prefill' AND req >= 0", (sid,)).fetchone()[0]
            if k >= want:
                done[r['name']] = sid
                break
            best = max(best, k)
        else:
            part[r['name']] = (best, want)
    return done, part


def simrec_cmd(argv, resolve_build, run_list_detached):
    words = []
    while argv and not argv[0].startswith('-'):
        words.append(argv.pop(0))
    build = 'latest'
    if '-b' in argv or '--build' in argv:
        i = argv.index('-b') if '-b' in argv else argv.index('--build')
        build = argv[i + 1]; del argv[i:i + 2]
    build = resolve_build(build)
    rows, bad = select(list_rows(), words)
    if bad:
        sys.exit(f"simrec: {', '.join(bad)} names no model or size of {LIST}")
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    done, part = complete_cells(con, build, rows)
    con.close()
    print(f"simrec on {build}: {len(rows)} rows, {len(done)} complete in stats.db, {len(part)} to run")
    for r in rows:
        print(f"  {r['name']:28} " + (f"complete (stats {done[r['name']]})" if r['name'] in done else f"to run ({part[r['name']][0]} of {part[r['name']][1]} requests recorded)"))
    if not part:
        ms, zs = sorted({r['m'] for r in rows}), sorted({r['size'] for r in rows})
        print(f"nothing missing: run.py runsim {ms[0] if len(ms) == 1 else 'MODEL'} {zs[0] if len(zs) == 1 else 'SIZE'} -b {build}")
        return 0
    # -r 0: each row's own request count (the list's n); -cs: every cell from an empty learned state of its own
    extra = [] if '-cs' in argv or '--clean-state' in argv else ['-cs']
    return run_list_detached('simrec', ['-b', build, '-r', '0', '-o', ','.join(n + '$' for n in part)] + extra + argv)


# ---- the replay ----

def load_cell(con, sid):
    """the prompt calls of one cell in order: dict(req, call, n, path, took, pos, passes [(n, ms)], dec [path per pass], pln {path: mean ms}, why); the request totals"""
    calls, cur, pos, last_req = [], None, 0, None
    tot = collections.defaultdict(dict)
    for req, call, ev, path, n, vals in con.execute("SELECT req, call, ev, path, n, vals FROM ppev WHERE stat_id = ? AND ev IN ('DON', 'DEC', 'PAS', 'PLN', 'CAL', 'PHA') ORDER BY rowid", (sid,)):
        v = json.loads(vals or '{}')
        if ev == 'PHA':
            tot[req][path] = v
            continue
        if req != last_req:
            pos, last_req = 0, req
        if cur is None or (cur['req'], cur['call']) != (req, call):
            cur = dict(req=req, call=call, passes=[], dec=[], pln={}, why='')
        if ev == 'PAS' and v.get('ms_tok') is not None and n:
            cur['passes'].append((n, v['ms_tok'] * n))
        elif ev == 'DEC' and path in ('chain', 'stream'):
            cur['dec'].append(path)
        elif ev == 'PLN' and 'one_s' not in v:
            cur['pln'] = {p: x[0] for p, x in v.items() if isinstance(x, list)}
        elif ev == 'CAL':
            cur['why'] = v.get('why') or ''
        elif ev == 'DON':
            cur.update(n=n, path=path, took=v.get('took_ms'), pos=pos)
            pos += n or 0
            calls.append(cur); cur = None
    return calls, tot


def verdicts(seq):
    """seq: per step in order (call or pass), dict(auto=path, t={path: ms}). Adds best, loss and verdict to each: ok, bad pick, switch too soon (auto left a path that was
    still best), switch too late (auto stayed on a path the crossover had left, and moved later), missed crossover (it never moved). Ties within NOISE are no loss."""
    for i, s in enumerate(seq):
        t = s['t']
        s['best'] = min(t, key=t.get) if t else None
        bset = s['bset'] = {p for p, x in t.items() if x <= t[s['best']] * (1 + NOISE)} if t else set()
        s['loss'] = max(0.0, t[s['auto']] - t[s['best']]) if s['auto'] in t and s['best'] else 0.0
        if s['auto'] not in t or s['auto'] in bset:
            s['verdict'] = 'ok'
            continue
        prev = seq[i - 1] if i else None
        if prev and prev['auto'] == s['auto'] and (prev['auto'] in prev['bset'] or prev['verdict'] in ('switch too late', 'missed crossover')):   # still on the path the crossover left
            later = any(x['auto'] != s['auto'] and x['auto'] in bset for x in seq[i + 1:])
            s['verdict'] = 'switch too late' if later else 'missed crossover'
        elif prev and prev['auto'] != s['auto'] and prev['auto'] in bset:
            s['verdict'] = 'switch too soon'
        else:
            s['verdict'] = 'bad pick'
    return seq


def runsim_cmd(argv):
    words = [a for a in argv if not a.startswith('-')]
    build = argv[argv.index('-b') + 1] if '-b' in argv else None
    if build in words:
        words.remove(build)
    rows, bad = select(list_rows(), words)
    ms, zs = {r['m'] for r in rows}, {r['size'] for r in rows}
    if bad or len(ms) != 1 or len(zs) != 1:
        sys.exit(f"usage: run.py runsim MODEL SIZE [-b BUILD]  (one model of {sorted({r['m'] for r in list_rows()})}, one size of {sorted({r['size'] for r in list_rows()})})")
    m, z = ms.pop(), zs.pop()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    if not build:   # the newest recording of this model and size
        r = con.execute("SELECT build FROM stats WHERE tag LIKE ? ORDER BY rowid DESC LIMIT 1", (f'%-sr-{m}-auto-{z}',)).fetchone()
        if not r:
            sys.exit(f"runsim: no auto row of {m} {z} in {DB}: run.py simrec {m} {z} -b BUILD")
        build = r[0]
    done, part = complete_cells(con, build, rows)
    print(f"runsim {m} {z} on {build}, from {DB} (no model run)")
    if part:
        print(f"  not complete (run.py simrec {m} {z} -b {build} records them): " + ', '.join(f"{n} ({k} of {w})" for n, (k, w) in part.items()))
    if 'auto' not in {r['strat'] for r in rows if r['name'] in done}:
        sys.exit("runsim: the auto row is not complete: nothing to replay")
    cells, tots = {}, {}
    for r in rows:
        if r['name'] in done:
            cells[r['strat']], tots[r['strat']] = load_cell(con, done[r['name']])
    pw = {}
    for r in rows:
        if r['name'] in done:
            c = con.execute("SELECT vals FROM ppev WHERE stat_id = ? AND ev = 'CEL'", (done[r['name']],)).fetchone()
            g = json.loads(c[0]) if c else {}
            pw[r['strat']] = (g.get('g1') or {}).get('busy_w_p50')
    con.close()
    ws = [w for w in pw.values() if w]
    print(f"  rows: " + ', '.join(f"{s} (x16 card {pw[s] or '?'} W busy)" for s in cells))
    if ws and max(ws) > 1.1 * min(ws):
        print(f"  WARNING: busy power differs over 10% between rows ({min(ws)}-{max(ws)} W): the times are not alike")

    # truth per call: (req, call, tokens) -> path -> ms, from every row (forced ones and auto's own), the path its DON names
    truth = collections.defaultdict(lambda: collections.defaultdict(list))
    ptruth = collections.defaultdict(lambda: collections.defaultdict(list))   # (req, call, pass index, tokens) -> path -> ms
    for s, calls in cells.items():
        for c in calls:
            k = (c['req'], c['call'], c['n'])
            truth[k][c['path']].append(c['took'])
            if c['path'] in ('chain', 'stream') and len(c['passes']) > 1:
                for i, (pn, pt) in enumerate(c['passes']):
                    dp = c['dec'][i] if i < len(c['dec']) else c['path']   # what this pass ran: the per-pass decision, else the call's path
                    if s != 'auto' or dp == c['path']:
                        ptruth[(c['req'], c['call'], i, pn)][dp].append(pt)
    med = lambda d: {p: st.median(v) for p, v in d.items()}

    # per request: auto's calls in order with the truth of each; the verdicts over that sequence
    print(f"\nper prompt call: auto against every recorded path (ms; tie = within {NOISE:.0%})")
    print(f"{'req':>4} {'call':>4} {'at token':>8} {'tokens':>6}  {'auto ran':9} {'auto (ms)':>9}  {'best':9} {'best (ms)':>9} {'loss (ms)':>9}  {'verdict':17} why auto picked it / every path (ms)")
    allv = []
    for req in sorted({c['req'] for c in cells['auto']}):
        seq = [dict(c=c, auto=c['path'], t=med(truth[(c['req'], c['call'], c['n'])])) for c in cells['auto'] if c['req'] == req]
        verdicts(seq)
        for s in seq:
            c = s['c']
            print(f"{'start' if req < 0 else req:>4} {c['call']:>4} {c['pos']:>8} {c['n']:>6}  {c['path']:9} {c['took']:>9.0f}  {s['best'] or '-':9} {s['t'].get(s['best'], 0):>9.0f} {s['loss']:>9.0f}  "
                  f"{s['verdict']:17} {c['why'][:28]:28} " + ' '.join(f"{p} {x:.0f}" for p, x in sorted(s['t'].items(), key=lambda x: x[1])))
        allv += seq

    # inside a call: the router / full stream decision per pass
    pv = []
    for c in cells['auto']:
        if c['path'] not in ('chain', 'stream') or len(c['passes']) < 2:
            continue
        seq = [dict(c=c, i=i, auto=(c['dec'][i] if i < len(c['dec']) else c['path']), t=med(ptruth[(c['req'], c['call'], i, pn)])) for i, (pn, _) in enumerate(c['passes'])]
        seq = [s for s in seq if len(s['t']) > 1]
        if seq:
            pv += verdicts(seq)
    if pv:
        print(f"\ninside calls: the router or full stream per pass (ms per pass)")
        print(f"{'req':>4} {'call':>4} {'pass':>4}  {'auto ran':9} {'best':9} {'loss (ms)':>9}  verdict")
        for s in pv:
            if s['verdict'] != 'ok':
                print(f"{s['c']['req']:>4} {s['c']['call']:>4} {s['i']:>4}  {s['auto']:9} {s['best']:9} {s['loss']:>9.0f}  {s['verdict']}")
        print(f"  {sum(s['verdict'] == 'ok' for s in pv)} of {len(pv)} passes ok, {sum(s['loss'] for s in pv):.0f} ms lost")

    # totals: auto, the best path per call (what a perfect planner gets), and each strategy held for the whole prompt
    print(f"\nprompt time per request (ms): auto against a perfect planner (best path per call) and each forced strategy")
    print(f"{'req':>5} {'auto':>9} {'perfect':>9} {'efficiency':>10}  " + ' '.join(f"{s:>9}" for s in cells if s != 'auto'))
    for req in sorted({c['req'] for c in cells['auto']}):
        a = sum(s['c']['took'] for s in allv if s['c']['req'] == req)
        b = sum(s['t'].get(s['best'], s['c']['took']) for s in allv if s['c']['req'] == req)
        per = {s: sum(c['took'] for c in cs if c['req'] == req) for s, cs in cells.items() if s != 'auto'}
        print(f"{'start' if req < 0 else req:>5} {a:>9.0f} {b:>9.0f} {100 * b / a if a else 0:>9.0f}%  " + ' '.join(f"{per[s]:>9.0f}" for s in per))
    print(f"\nprefill and decode per request (t/s, the engine's own totals)")
    for s, t in tots.items():
        print(f"  {s:7} " + '  '.join(f"req {q}: prompt {x.get('prefill', {}).get('tps') or 0:.0f}, answer {x.get('decode', {}).get('tps') or 0:.1f}" for q, x in sorted(t.items()) if q >= 0))

    # verdict totals
    print(f"\nverdicts over {len(allv)} calls" + (f" and {len(pv)} passes" if pv else ""))
    for v in ('ok', 'bad pick', 'switch too soon', 'switch too late', 'missed crossover'):
        xs = [s for s in allv + pv if s['verdict'] == v]
        if xs:
            print(f"  {v:17} {len(xs):>4}  {sum(s['loss'] for s in xs):>9.0f} ms lost")

    # the planner's beliefs: its mean for a path before the call against that path's measured time in the same call
    rat = collections.defaultdict(list)
    for s in allv:
        for p, mean in s['c']['pln'].items():
            if p in s['t'] and s['t'][p] > 0 and mean > 0:
                rat[p].append(mean / s['t'][p])
    if rat:
        print(f"\nthe planner's mean before the call / that path's measured time in the same call (1.00 = it knew)")
        print(f"{'path':8} {'calls':>5} {'median':>7} {'in 1.3x':>8}")
        for p, v in sorted(rat.items()):
            print(f"{p:8} {len(v):>5} {st.median(v):>7.2f} {100 * sum(1 / 1.3 <= x <= 1.3 for x in v) / len(v):>7.0f}%")
    return 0


def selftest():
    T = lambda auto, **t: dict(auto=auto, t=t)
    v = lambda seq: [s['verdict'] for s in verdicts(seq)]
    assert v([T('a', a=10, b=20)]) == ['ok']
    assert v([T('a', a=10, b=10.3)]) == ['ok']                                            # a tie
    assert v([T('a', a=20, b=10)]) == ['bad pick']
    assert v([T('a', a=10, b=20), T('a', a=20, b=10), T('b', a=20, b=10)]) == ['ok', 'switch too late', 'ok']
    assert v([T('a', a=10, b=20), T('a', a=20, b=10), T('a', a=20, b=10)]) == ['ok', 'missed crossover', 'missed crossover']
    assert v([T('a', a=10, b=20), T('b', a=10, b=20), T('b', a=20, b=10)]) == ['ok', 'switch too soon', 'ok']
    s = verdicts([T('a', a=30, b=10)])[0]
    assert s['loss'] == 20 and s['best'] == 'b'
    assert pick_size('60k4k', ['2k', '64k', 'chat']) == '64k' and pick_size('chat', ['2k', 'chat']) == 'chat' and pick_size('x', ['2k']) is None
    print('simrec: verdict rules pass')


if __name__ == '__main__':
    if sys.argv[1:2] == ['test']:
        selftest()
    else:
        print(__doc__ or 'run.py simrec / run.py runsim')
