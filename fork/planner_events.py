# The planner replay gate's input (tests/test-planner-replay-fork.cpp): per prompt call the CAL line's features, the DON line's outcome and the fill wall, and
# the measured medians per call class (size, cold/warm request, arm).
#   python3 -I fork/planner_events.py [out_dir]        GLM 3.5 12k (q5h *glm35-12k-rp*.stats.log) -> out_dir/glm_events.csv, out_dir/truth.csv (the gate's data)
#   python3 -I fork/planner_events.py --all [out_dir]  every stats.log kept in stats.db (fork/crosstest.py's) -> out_dir/events.csv, out_dir/truth.csv: per model
#                                                       and cell configuration (build and options): measurements compare within one, as crosstest's per build
#   python3 -I fork/planner_events.py --check          the parser on the K3 4588-token call
#   then  test-planner-replay-fork events truth [seeds]
import re, glob, csv, os, sys, statistics as st, collections
num = lambda p, s: float(m.group(1)) if (m := re.search(p, s)) else 0.0

def cal_features(s):
    """a CAL line's features of the call: the stream's upload (link or SSD bound), the mix's, the misses', the decode chain's chunks and reach, the link upload,
    layer parallel's calculated time per pass (0: no candidate)"""
    gb, rate = num(r'stream reads ([\d.]+) GB', s), num(r'from the SSD at ([\d.]+) GB/s', s)
    U = num(r'streaming upload (\d+) ms', s)
    return dict(n=num(r'prompt call of (\d+)', s), Ts=max(U, gb / rate * 1e3 if rate > 0 else 0), Um=num(r'with the mix (\d+) ms', s), up=num(r'misses uploaded (\d+) ms', s),
                ch=num(r'4-token chunks (\d+) ms', s), dm=num(r'decode chain to (\d+) tokens', s), U=U, lp=num(r'calculated per pass: [^;]*layer parallel (\d+) ms', s))

def glm(out):
    w = csv.writer(open(out + '/glm_events.csv', 'w'))
    rows = []
    w.writerow(['log', 'req', 'n', 'cells', 'Ts', 'Um', 'up', 'ch', 'path', 'took_ms', 'fill_ms', 'dec_max', 'U_link'])
    for f in sorted(glob.glob('/p/bw/data/q5h/*glm35-12k-rp*.stats.log')):
        log = os.path.basename(f).replace('.stats.log', '')
        cal = None; fill = 0.0; big = 0; req = -1
        for l in open(f, errors='replace'):
            p = l.split(None, 4)
            if len(p) < 5: continue
            t, s = p[3], p[4]
            if t == 'CAL' and s.startswith('prompt call of'):
                cal = cal_features(s)
                fill = 0.0
            elif t == 'SPN' and 'copy fill' in s:
                fill += num(r'dur ([\d.]+) ms', s)
            elif t == 'DON' and s.startswith('prompt call of') and cal:
                m = re.match(r'prompt call of (\d+) tokens: (.+?) took (\d+) ms', s)
                if not m or float(m.group(1)) != cal['n'] or cal['n'] < 1000: cal = None; continue
                if cal['n'] < 3000: req += 1   # each request: one ~2k call, then one ~10k call
                cells = cal['n'] / 2 if cal['n'] < 3000 else (2034 if cal['n'] == 10400 else 1938) + cal['n'] / 2
                r = [log, req, int(cal['n']), cells, cal['Ts'], cal['Um'], cal['up'], cal['ch'], m.group(2), m.group(3), round(fill), cal['dm'], cal['U']]
                w.writerow(r); rows.append(r)
                cal = None

    # the measured medians per call class: the outcome a replayed pick gets (stream, mix and predictions pooled as 'stream'; dense without its fill)
    T = collections.defaultdict(list)
    for r in rows:
        p = r[8]; arm = 'dense' if p == 'dense' else 'dec' if p == 'decode chain' else 'chain' if p == 'prompt chain' else 'stream'
        # each stream variant on its own too (stream, mix, k1 .. k32): flipping between variants costs what they measured (rampool12: k x32 at 2k)
        var = 'mix' if p == 'stream, mix' else ('k' + p.split(' x')[1]) if ', k x' in p else 'ring' if p == 'stream' else None
        if var: T[('2k' if r[2] < 3000 else '10k', 'c' if r[1] == 0 else 'w', var)].append(float(r[9]))
        # the split's pass without its copy fill: whether a fill is owed is the replayed state's, not the logged start's (the replay adds it)
        T[('2k' if r[2] < 3000 else '10k', 'c' if r[1] == 0 else 'w', arm)].append(float(r[9]) - (float(r[10]) if arm == 'dense' else 0.0))
    t = csv.writer(open(out + '/truth.csv', 'w'))
    for k, v in sorted(T.items()): t.writerow(list(k) + [st.median(v), len(v)])

# experts per layer and picks per token by model key: the GGUF headers (expert_count, expert_used_count; read 2026-11-15), the stats.log states neither
EXPERTS = {'olmoe': (64, 8), 'qwen36': (256, 8), 'qni3': (512, 10), 'qni4': (512, 10), 'qn1m': (512, 10), 'ds': (384, 6), 'glm30': (288, 8), 'glm35': (288, 8),
           'mimo': (256, 8), 'k31': (896, 16)}
FILL = re.compile(r'\+ fill ([\d.]+) \+- [\d.]+ s \((owed|copies in place), n=0\)')

def calls(lines):
    """one stats.log -> (its prompt calls above the chains' band, crosstest X1's 64-512 (K3's 995-token prompts are one card or two), the fill the split planner
    was first told it owes in ms or None). A call: its first CAL line's features (a pass the pool refused is planned again smaller, K3 4588 -> 4128: the first
    CAL is the pick that was made), the PRE line's pass size, the DON outcome, the copy fill wall, the request (0: the first and the calls before it, cold)."""
    out, cal, pre, fill, req, fest = [], None, {}, 0.0, -1, None
    for l in lines:
        p = l.split(None, 4)
        if len(p) < 5: continue
        t, s = p[3], p[4]
        if fest is None and t in ('PLN', 'CAL') and (m := FILL.search(s)): fest = float(m.group(1)) * 1e3 if m.group(2) == 'owed' else 0.0
        if t == 'REQ' and s.startswith('begin'): req += 1; cal = None
        elif t == 'CAL' and s.startswith('prompt call of') and cal is None: cal = cal_features(s); fill = 0.0
        elif t == 'PLN' and cal and (m := re.match(r'plan: prompt call of \d+ tokens at (\d+) cells', s)): cal.setdefault('cells', float(m.group(1)))
        elif t == 'SPN' and 'copy fill' in s: fill += num(r'dur ([\d.]+) ms', s)
        elif t == 'PRE': pre = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', s))
        elif t == 'DON' and (m := re.match(r'prompt call of (\d+) tokens: (.+?) took (\d+) ms', s)):
            n = int(m.group(1))
            if cal and n > 512 and cal['n'] <= n:   # the CAL's size is the call's pass, at most the call
                one = pre.get('prompt_tokens') == str(n)   # the PRE line of this call: its pass size, and whether it ran an arm the replay has not (passes of
                # one card and of the split mixed; experts split under a one-card path name, --moe prefill-split)
                pt = int(pre['pass_tokens']) if one and pre.get('pass_tokens', '').isdigit() else 0
                out.append(dict(cal, req=max(req, 0), n_call=n, n=float(pt) if pt > 0 else min(n, cal['n']), path=m.group(2), took=float(m.group(3)), fill=fill,
                                mixed=one and pre.get('mode') in ('mixed', 'expert-split')))
            cal, pre = None, {}
    return out, fest

FORCED = re.compile(r'(pp-path|pp-k|pp-mix|dense-split|prefill-split|moe-split|state|log)=')

def regime(args):
    """a cell's machine and configuration: its options less the arm they force, the planner's state file and the log level (the rest - simulated memory,
    staging, checks, loaders, drafts - changes what every arm costs: those cells are other machines, user rule 'different paths never compared')"""
    t, out = [x for x in re.split(r'[\s,]+', args or '') if x], []
    for i, x in enumerate(t):
        if x in ('-np', '--moe') or (i and t[i - 1] == '-np') or FORCED.match(x): continue
        out.append(os.path.basename(x) if '/' in x else x)
    return ','.join(out)

def split_prior(tag):
    """the split planner's prior (us per token) from the cell's server log in q5h (a tag run again keeps its last run's: the same plan); 0: none logged"""
    try:
        with open(f'/p/bw/data/q5h/{tag}.server.log', errors='replace') as f:
            for l in f:
                if 'split planner: split prior' in l: return num(r'split prior ([\d.]+) us', l)
    except OSError: pass
    return 0.0

def every(out):
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import crosstest as ct, sqlite3, zlib
    db = sqlite3.connect(f'file:{ct.DB}?mode=ro', uri=True)
    files = dict(db.execute('select stat_id, max(model_file) from ppev group by stat_id'))   # run.py's model file per cell: a tag need not name it (K3's p1k cells)
    starts = []   # (model, machine, log, calls, fill estimate, split prior)
    for rid, ts, build, tag, args, blob in db.execute('select rowid, ts, build, tag, args, log from stats order by ts'):
        m = ct.model_key(files.get(rid) or '') or ct.model_key(tag)
        if m not in EXPERTS: continue
        L = zlib.decompress(blob).decode(errors='replace').splitlines()
        cs, fest = calls(L)
        if not cs: continue
        rg = regime(args)
        # the split planner's split / one-card compute ratio (the plan's bench), printed while the split never ran: "split prior = max(upload U s, one x R)"
        ratio = next((float(x.group(1)) for l in L if ' PLN plan: pass of ' in l and (x := re.search(r'one x ([\d.]+)\)', l))), 0.0)
        starts.append((m, build + (' ' + rg if rg else ''), f'{tag}@{ts[5:16]}', cs, fest, split_prior(tag), ratio))
    # per model and machine: the measured copy fill (its median), the fill estimate and the split prior where a start logged none (the first one logged there,
    # else the model's: a start that loaded a stored split posterior logs no prior)
    fills, fests, priors, ratios = collections.defaultdict(list), {}, {}, {}
    for m, mach, _, cs, fest, sp, ra in starts:
        fills[(m, mach)] += [c['fill'] for c in cs if c['fill'] > 0]
        if fest is not None: fests.setdefault((m, mach), fest)
        if sp > 0: priors.setdefault((m, mach), sp); priors.setdefault(m, sp)
        if ra > 0: ratios.setdefault((m, mach), ra); ratios.setdefault(m, ra)
    w = csv.writer(open(out + '/events.csv', 'w'))
    w.writerow(['log', 'req', 'n', 'cells', 'Ts', 'Um', 'up', 'ch', 'path', 'took_ms', 'fill_ms', 'dec_max', 'U_link', 'model', 'cls', 'n_used', 'n_exp', 'fill_est_ms',
                'fill_wall_ms', 'split_prior_us', 'lp_ms', 'n_call', 'mixed', 'ratio'])
    T = collections.defaultdict(list)   # (class, cold/warm, arm) -> ms per token of every call that ran one replayed arm (a mixed one measures none)
    for m, mach, log, cs, fest, sp, ra in starts:
        k = (m, mach)
        fe = fest if fest is not None else fests.get(k, 0.0)
        fw = (st.median(fills[k]) if fills[k] else fe) if fe > 0 else 0.0   # what this start's first split pays (none where its copies were in place)
        for c in cs:
            cls = f'{m} {mach} {ct.size_class(c["n"])}'
            w.writerow([log, c['req'], int(c['n']), round(c.get('cells', c['n'] / 2)), c['Ts'], c['Um'], c['up'], c['ch'], c['path'], int(c['took']), round(c['fill']), c['dm'], c['U'],
                        m, cls, EXPERTS[m][1], EXPERTS[m][0], round(fe), round(fw), sp or priors.get(k) or priors.get(m, 0.0), c['lp'], c['n_call'], int(c['mixed']),
                        ra or ratios.get(k) or ratios.get(m, 0.0)])
            if c['mixed']: continue
            a, cw, p = ct.arm_class(c['path']), 'c' if c['req'] == 0 else 'w', c['path']
            tok = (c['took'] - (c['fill'] if a == 'dense' else 0.0)) / c['n_call']   # the split's pass without its copy fill (the replay adds it where owed)
            if a in ('mix', 'k', 'stream'):   # one card's stream variants pooled, as the GLM gate's, and each on its own
                T[(cls, cw, 'stream')].append(tok)
                T[(cls, cw, {'mix': 'mix', 'stream': 'ring'}.get(a) or 'k' + p.split(' x')[1])].append(tok)
            else: T[(cls, cw, a)].append(tok)
    t = csv.writer(open(out + '/truth.csv', 'w'))
    t.writerow(['class', 'cw', 'arm', 'ms_per_token', 'n', 'lo', 'hi'])   # the median and its samples' range (two samples far apart make a fragile median)
    for k, v in sorted(T.items()): t.writerow(list(k) + [round(st.median(v), 4), len(v), round(min(v), 4), round(max(v), 4)])
    print(f'{len(starts)} starts, {sum(len(x[3]) for x in starts)} calls of more than 512 tokens, {len(T)} measured classes -> {out}/events.csv, {out}/truth.csv')

def check():
    """--check: the K3 4588-token call (stats.log of 2026-11-15, shortened): a refused pass planned again smaller, run mixed; and the cell key's options"""
    L = ['0 0 0 REQ begin: 4592 prompt tokens',
         '0 0 0 CAL prompt call of 4588 tokens: stream, layer parallel (x; took: dense 0 ms; calculated per pass: stream 46640 ms, mix 45735 ms, layer parallel 45735 ms; '
         'streaming upload 20252 ms, with the mix 20020 ms, misses uploaded 20496 ms or on the CPU 249079 ms; stream reads 305.0 GB from the SSD at 6.54 GB/s; 4-token chunks 0 ms)',
         '0 0 0 PLN plan: prompt call of 4588 tokens at 2294 cells: stream 46640 +- 1 ms',
         '0 0 0 PLN plan: pass of 4588 tokens at 0k cells: one card [x] 45.74 +- 1 s, split 46.64 +- 1 s (n=0, nis 1.00) + fill 0.00 +- 0.00 s (copies in place, n=0); x',
         '0 0 0 CAL prompt call of 4128 tokens: stream, layer parallel (x; calculated per pass: stream 47184 ms, mix 46280 ms, layer parallel 46280 ms)',
         '0 0 0 PRE prompt_tokens=4588 passes=1 pass_tokens=4588 mode=mixed gpus=CUDA0+CUDA1',
         '0 0 0 DON prompt call of 4588 tokens: dense took 142224 ms (32 t/s)']
    (c,), fest = calls(L)
    assert (c['n'], c['n_call'], c['lp'], c['Ts'] // 1, c['cells'], c['path'], c['took'], c['mixed'], c['req'], fest) == (4588, 4588, 45735, 46636, 2294, 'dense', 142224, True, 0, 0.0), c
    assert regime('-np 1 -np 1 --moe sim-vram=8192,dense-split=0,state=/a/s.ini,staging=0 -lm /m/x.gguf') == 'sim-vram=8192,staging=0,-lm,x.gguf'
    print('planner_events: check passed')

if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    out = a[0] if a else '/tmp'
    check() if '--check' in sys.argv else every(out) if '--all' in sys.argv else glm(out)
