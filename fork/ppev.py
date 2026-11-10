# The planner's events of a cell as rows (user 2026-11-10: "new table with timestamp pppos ubatch maxubatch promptlen evtype vals"): table ppev in stats.db,
# one row per event of the stats.log (REQ, PLN, CAL, DEC, SPL, PAS, DON, PRE, PHA), stamped with the request, the call within it, the prompt length, the
# pass size and its ceiling in force; vals holds the event's numbers as JSON. run.py fills it with every cell (runlog.insert_stats); the replay and the
# simulator read it instead of the text logs (fork/replay_score.py).
#   python3 -I fork/ppev.py load        every stats row of stats.db not in ppev yet (the model from the cell's server log, else from table runs)
#   python3 -I fork/ppev.py reload      all of them again (after a parser change)
import json, os, re, sqlite3, sys, zlib

DB = '/p/bw/data/stats.db'   # next to runs.db, local only (run.py runlog STATS_DB)
Q5H = '/p/bw/data/q5h'
PATHS = {'stream': 'stream', 'stream, mix': 'mix', 'prompt chain': 'chain', 'decode chain': 'dec', 'dense': 'dense'}
# model: short name (glm30, glm35, mimo, ds, qni4, q35, olmoe; else the file's stem), prompt: the test's short name (pf2k, scific, pf64k4k), forced: the strategy the
# cell's command line forced (pp-path, pp-k, dense-split; '' = auto). Per call (user 2026-11-10): pp_pct the request's prompt done once the row's call ends, moe_used_pct
# the experts its routers picked of all, picked_up_pct of those the ones uploaded, up_ssd_pct the uploaded bytes read from the model file (the rest from the RAM pool or host)
SCHEMA = ('CREATE TABLE IF NOT EXISTS ppev (stat_id INTEGER, build TEXT, model TEXT, prompt TEXT, forced TEXT, model_file TEXT, tag TEXT, t REAL, req INTEGER, call INTEGER, '
          'ev TEXT, path TEXT, n INTEGER, pos INTEGER, ub INTEGER, ub_max INTEGER, prompt_len INTEGER, pp_pct REAL, moe_used_pct REAL, picked_up_pct REAL, up_ssd_pct REAL, vals TEXT)')
NCOL = 22
SHORT = [(r'GLM-5\.3-Flash.*3\.0bit', 'glm30'), (r'GLM-5\.3-Flash.*3\.5bit', 'glm35'), (r'MiMo', 'mimo'), (r'DeepSeek', 'ds'), (r'Qwen3\.8-Flash-Next.*IQ4', 'qni4'),
         (r'Qwen3\.6-35B', 'q35'), (r'OLMoE', 'olmoe')]


def short_model(file):
    return next((k for p, k in SHORT if re.search(p, file)), re.sub(r'(-\d{5}-of-\d{5})?\.gguf$', '', file))


def forced_of(args):
    o = dict(re.findall(r'(pp-path|pp-k|dense-split)=([\w.]+)', args or ''))
    return ','.join(f'{k}={v}' for k, v in sorted(o.items()))

# One row per prompt call (DON), the numbers the simulator and the planner ask for most (user 2026-11-10: "easy fast lookup of info that is key for sim and planner"):
# the call, what ran and what it took, what the planner saw before it (CAL: uploads, misses, CPU, SSD rate, compute; PLN: its mean for the path that ran and the
# path it picked; DEC: router and stream estimates), and the cell's power on the x16 card (CEL). ms_tok = took / tokens.
VIEW = """CREATE VIEW IF NOT EXISTS ppcall AS SELECT d.stat_id, d.build, d.model, d.prompt, d.forced, d.tag, d.req, d.call, d.path, d.n, d.ub, d.ub_max, d.prompt_len,
  d.pp_pct, d.moe_used_pct, d.picked_up_pct, d.up_ssd_pct,
  json_extract(d.vals, '$.took_ms') AS took_ms, json_extract(d.vals, '$.took_ms') / d.n AS ms_tok, json_extract(d.vals, '$.pass_took_ms') AS pass_ms,
  json_extract(d.vals, '$.pred_ms') AS pred_pass_ms, json_extract(d.vals, '$.best_path') AS best_known_path, json_extract(d.vals, '$.best_ms') AS best_known_ms,
  json_extract(d.vals, '$.host_gb') AS host_gb, json_extract(d.vals, '$.file_gb') AS file_gb, json_extract(d.vals, '$.ram_pool_gb') AS ram_pool_gb,
  json_extract(d.vals, '$.sched_read_ms') AS sched_read_ms, json_extract(d.vals, '$.ram_avail_gib') AS ram_avail_gib,
  (SELECT c.path FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'CAL' ORDER BY c.t LIMIT 1) AS cal_pick,
  (SELECT json_extract(c.vals, '$.why') FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'CAL' ORDER BY c.t LIMIT 1) AS why,
  (SELECT c.vals FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'CAL' ORDER BY c.t LIMIT 1) AS cal,
  (SELECT json_extract(c.vals, '$.' || d.path || '[0]') FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'PLN' ORDER BY c.t LIMIT 1) AS plan_mean_ms,
  (SELECT c.vals FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'PLN' ORDER BY c.t LIMIT 1) AS plan,
  (SELECT c.vals FROM ppev c WHERE c.stat_id = d.stat_id AND c.req = d.req AND c.call = d.call AND c.ev = 'DEC' AND c.path != '' ORDER BY c.t LIMIT 1) AS switch,
  (SELECT json_extract(e.vals, '$.g1.busy_w_p50') FROM ppev e WHERE e.stat_id = d.stat_id AND e.ev = 'CEL') AS g1_busy_w,
  (SELECT json_extract(e.vals, '$.g0.busy_w_p50') FROM ppev e WHERE e.stat_id = d.stat_id AND e.ev = 'CEL') AS g0_busy_w
FROM ppev d WHERE d.ev = 'DON'"""


def path_of(s):
    s = s.strip()
    if s.startswith('stream, k x'):
        return 'k' + s[len('stream, k x'):]
    return PATHS.get(s, s)


def f(p, s, g=1):
    m = re.search(p, s)
    return float(m.group(g)) if m else None


def events(text):
    """(t, req, call, ev, path, n, pos, ub, ub_max, prompt_len, pp_pct, moe_used_pct, picked_up_pct, up_ssd_pct, vals) per event line"""
    req, call, plen, ub, ub_max, done = -1, 0, 0, 0, 0, 0
    pct = lambda: round(100.0 * done / plen, 2) if plen else None
    smp, t_end = {}, 0.0   # per card: (watts, MHz) of the busy samples (util > 80%), for the cell's CEL row
    for line in text.splitlines():
        p = line.split(None, 4)
        if len(p) < 5 or p[0].startswith('#'):
            continue
        try:
            t = float(p[2])
        except ValueError:
            continue
        ev, s = p[3], p[4]
        path, n, pos, v = '', None, None, {}
        t_end = t
        if ev == 'SMP':
            m = re.match(r'(g\d+) util (\d+)% .*? (\d+) W (\d+) MHz', s)
            if m and int(m.group(2)) > 80:
                smp.setdefault(m.group(1), []).append((int(m.group(3)), int(m.group(4))))
            continue
        if ev == 'REQ' and s.startswith('begin'):
            req += 1; call = 0; done = 0; plen = int(f(r'begin: (\d+) prompt tokens', s) or 0)
        elif ev == 'DEC' and s.startswith('ubatch = '):
            ub = int(f(r'ubatch = (\d+)', s) or 0); ub_max = int(f(r'ceiling (\d+)', s) or ub_max)
            v = {'passes': f(r'\((\d+) pass', s)}; n = f(r'for (\d+) tokens', s)
        elif ev == 'DEC' and ': router ' in s and 'full stream' in s:
            n = f(r'prompt call of (\d+)', s); path = 'stream' if ': full stream:' in s else 'chain'
            v = {'router_ms_tok': f(r'router ([\d.]+) ms per token', s), 'router_measured': '(measured' in s.split('full stream ')[0].split('router ', 1)[1],
                 'picks_ms': f(r'picks (\d+)', s), 'wait_ms': f(r'wait (\d+)', s), 'compute_ms': f(r'compute (\d+) ms', s),
                 'stream_ms_tok': f(r'full stream ([\d.]+) ms per token', s), 'stream_measured': '(measured' in s.split('full stream ', 2)[-1], 'upload_ms': f(r'upload (\d+) ms', s)}
        elif ev == 'PLN' and s.startswith('plan: prompt call of'):
            n = f(r'prompt call of (\d+)', s); pos = f(r'at (\d+) cells', s)
            v = {path_of(m.group(1)): [float(m.group(2)), float(m.group(3)), int(m.group(4))] for m in re.finditer(r'(?:; |cells: )([a-z][a-z ,0-9x]*?) (-?\d+) \+- (\d+) ms, draw -?\d+ \(n=(\d+)', s)}
            m = re.search(r'picked (.+)$', s); path = path_of(m.group(1)) if m else ''
        elif ev == 'PLN' and s.startswith('plan: pass of'):
            n = f(r'pass of (\d+) tokens', s)
            v = {'one_s': f(r'one card \[[^\]]*\] ([\d.]+)', s), 'split_s': f(r'split ([\d.]+) \+-', s), 'fill_s': f(r'fill ([\d.]+) \+-', s), 'passes_left': f(r'(\d+) passes left', s)}
        elif ev == 'CAL' and s.startswith('prompt call of'):
            n = f(r'prompt call of (\d+)', s); m = re.match(r'prompt call of \d+ tokens: (.+?) \(', s); path = path_of(m.group(1)) if m else ''
            v = {'why': (re.search(r'\(([^;]*)', s) or [None, ''])[1], 'U_ms': f(r'streaming upload (\d+) ms', s), 'Um_ms': f(r'with the mix (\d+) ms', s),
                 'up_ms': f(r'misses uploaded (\d+) ms', s), 'cpu_ms': f(r'on the CPU (\d+) ms', s), 'experts_off': f(r'over (\d+) experts off', s),
                 'link0_gbs': f(r'links ([\d.]+)', s), 'link1_gbs': f(r'links [\d.]+/([\d.]+)', s), 'cpu_gbs': f(r'CPU ([\d.]+) GB/s', s),
                 'ssd_gb': f(r'stream reads ([\d.]+) GB', s), 'ssd_gbs': f(r'from the SSD at ([\d.]+) GB/s', s), 'dec_max': f(r'decode chain to (\d+) tokens', s),
                 'dec_calc_ms': f(r'tokens, (\d+) ms calculated', s), 'dec_ratio': f(r'misses x([\d.]+)', s), 'chunk_ms': f(r'4-token chunks (\d+) ms', s),
                 'chunk_k': f(r'4-token chunks \d+ ms x([\d.]+)', s), 'read_gbs': f(r'at (\d+) GB/s weight read', s), 'dec_f_ms': f(r'fixed (\d+) ms', s),
                 'compute_ms': f(r'compute (\d+) ms\)', s)}
            v['took'] = {path_of(m.group(1)): float(m.group(2)) for m in re.finditer(r'(?:took: |, )(decode chain|chain|stream|mix|dense) (\d+) ms', s.split('took: ', 1)[-1].split(';', 1)[0])} if 'took: ' in s else {}
        elif ev == 'SPL':
            n = f(r'^(\d+) tokens', s); path = 'split' if s.rstrip().endswith(': split') else 'one'
            v = {'one_ms': f(r'one card (\d+) ms', s), 'split_ms': f(r'split (\d+) ms', s), 'shares': re.findall(r'(CUDA\d+) (\d+)%', s)}
        elif ev == 'PAS' and s.startswith('prompt pass'):
            n = f(r': (\d+) tokens in', s)
            v = {'i': f(r'pass (\d+)/', s), 'of': f(r'pass \d+/(\d+)', s), 'sec': f(r'tokens in ([\d.]+) s', s), 'cache_pct': f(r'experts from cache (\d+)%', s),
                 'ms_tok': f(r'([\d.]+) ms per token', s), 'upload_mib_tok': f(r'upload ([\d.]+) MiB', s)}
        elif ev == 'DON' and s.startswith('prompt call of'):
            m = re.match(r'prompt call of (\d+) tokens: (.+?) took (\d+) ms', s)
            if m:
                n = float(m.group(1)); path = path_of(m.group(2))
                v = {'took_ms': float(m.group(3)), 'picked_pct': f(r'routers picked ([\d.]+)%', s), 'uploaded_pct': f(r'([\d.]+)% of those uploaded', s),
                     'pred_ms': f(r'predicted (-?\d+) \+-', s), 'pred_sd_ms': f(r'predicted -?\d+ \+- (\d+)', s), 'pass_took_ms': f(r'per pass, took (\d+)', s),
                     'best_path': path_of(m2.group(1)) if (m2 := re.search(r'best known here (.+?) \d+ ms', s)) else None, 'best_ms': f(r'best known here .+? (\d+) ms', s),
                     'ram_pool_gb': f(r'RAM pool ([\d.]+) GB', s), 'host_gb': f(r'host ([\d.]+) GB', s), 'file_gb': f(r'file ([\d.]+) GB', s),
                     'sched_read_ms': f(r'(\d+) ms of the scheduler thread', s), 'read_wait_ms': f(r'(\d+) ms more waiting', s), 'ram_avail_gib': f(r'avail ([\d.]+) GiB', s)}
                done += int(n)
                tot = sum(v[k] or 0.0 for k in ('ram_pool_gb', 'host_gb', 'file_gb'))
                yield (t, req, call, ev, path, n, pos, ub, ub_max, plen, pct(), v['picked_pct'], v['uploaded_pct'], round(100.0 * (v['file_gb'] or 0.0) / tot, 2) if tot > 0 else None, v)
                call += 1
                continue
        elif ev == 'DON' and s.startswith('split pass of'):
            n = f(r'split pass of (\d+)', s); path = 'dense'; ev = 'DONS'   # the split planner's own report of the pass (the call's DON follows)
            v = {'pred_s': f(r'predicted ([\d.]+) \+-', s), 'took_s': f(r'took ([\d.]+)', s), 'fill_s': f(r'fill wall ([\d.]+) s', s)}
        elif ev == 'PRE':
            v = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', s)); n = f(r'prompt_tokens=(\d+)', s); path = v.get('mode', '')
        elif ev == 'PHA' and ('prefill -> decode' in s or 'decode -> idle' in s):   # the request's prompt totals (path prefill), its answer's (path decode)
            path = 'prefill' if 'prefill -> decode' in s else 'decode'
            v = {'tokens': f(r'(?:decode|idle): (\d+) tokens', s), 'sec': f(r'tokens ([\d.]+) s', s), 'tps': f(r'([\d.]+) t/s', s)}
        else:
            continue
        yield (t, req, call, ev, path, n, pos, ub, ub_max, plen, pct(), None, None, None, v)   # req -1: before the first request (the start-up's warm-up call)
    # the cell's conditions, one row: what every time in it was measured under (2026-11-10: GLM 12k 871 t/s at 268-279 W on the x16 card, 585 at the night's 200 W cap)
    v = {}
    for g, xs in sorted(smp.items()):
        w, mhz = sorted(x[0] for x in xs), sorted(x[1] for x in xs)
        v[g] = {'busy_w_p50': w[len(w) // 2], 'busy_w_max': w[-1], 'busy_mhz_p50': mhz[len(mhz) // 2], 'busy_samples': len(xs)}
    yield (t_end, max(req, 0), call, 'CEL', '', None, None, ub, ub_max, plen, None, None, None, None, v)


def insert(con, stat_id, build, model_file, tag, test, args, text):
    con.execute(SCHEMA)
    con.execute('CREATE INDEX IF NOT EXISTS ppev_cell ON ppev (stat_id)')
    con.execute('CREATE INDEX IF NOT EXISTS ppev_mp ON ppev (model, prompt, ev)')
    con.execute('CREATE INDEX IF NOT EXISTS ppev_call ON ppev (stat_id, req, call, ev)')
    con.execute(VIEW)
    mf = os.path.basename(model_file or '')
    head = (stat_id, build, short_model(mf) if mf else '', test, forced_of(args), mf, tag)
    rows = [head + (t, rq, cl, ev, pa, None if n is None else int(n), None if pos is None else int(pos), ub, ubm, pl, pp, mu, pu, us, json.dumps(v, separators=(',', ':')))
            for (t, rq, cl, ev, pa, n, pos, ub, ubm, pl, pp, mu, pu, us, v) in events(text)]
    con.executemany('INSERT INTO ppev VALUES (' + ','.join('?' * NCOL) + ')', rows)
    return len(rows)


def model_of(con, tag, build, test, ts):
    sl = os.path.join(Q5H, tag + '.server.log')
    if os.path.exists(sl):
        with open(sl, errors='replace') as fh:
            for i, line in enumerate(fh):
                m = re.search(r'([\w.+-]+\.gguf)', line)
                if m:
                    return m.group(1)
                if i > 400:
                    break
    runs = sqlite3.connect(os.path.join(os.path.dirname(os.path.realpath(DB)), 'runs.db'))   # stats.db has no runs table
    r = runs.execute('SELECT model FROM runs WHERE build = ? AND test = ? AND ts >= ? ORDER BY ts LIMIT 1', (build, test, ts)).fetchone()
    runs.close()
    return os.path.basename(r[0]) if r and r[0] else ''


def load(db=DB):
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.execute('DELETE FROM ppev WHERE stat_id NOT IN (SELECT stat_id FROM ppev WHERE ev = \'CEL\')')   # parsed before the CEL row existed: parse again
    done = {r[0] for r in con.execute('SELECT DISTINCT stat_id FROM ppev')}
    k = 0
    for sid, ts, build, test, tag, args, blob in con.execute('SELECT rowid, ts, build, test, tag, args, log FROM stats').fetchall():
        if sid in done:
            continue
        k += insert(con, sid, build, model_of(con, tag, build, test, ts), tag, test, args, zlib.decompress(blob).decode(errors='replace'))
    con.commit()
    print(f'ppev: {k} events added, {con.execute("SELECT COUNT(DISTINCT stat_id) FROM ppev").fetchone()[0]} cells')


if __name__ == '__main__':
    if sys.argv[1:2] in (['load'], ['reload']):
        db = sys.argv[2] if len(sys.argv) > 2 else DB
        if sys.argv[1] == 'reload':   # every cell parsed again (after a parser change)
            c = sqlite3.connect(db); c.execute('DROP TABLE IF EXISTS ppev'); c.commit(); c.close()
        load(db)
    else:
        print(__doc__ if __doc__ else 'usage: ppev.py load [db]')
