# Scores the prompt planner against the truth of forced rows (user 2026-11-10: "sim shall be replaying and comparing how its and planners estimated predicted
# crossover decisions would have fared"). Rows of one group run the same prompts, each row on one strategy; the strategy token in the row name (auto, stream,
# mix, k1, dense, chain, dec) is dropped to get the group (xt-2k-auto -> xt-2k, sr-glm30-auto-2k -> sr-glm30-2k). Every prompt call is keyed by
# (request, call within it) and filed under the path its DON line says it ran on, whatever the row forced. Per call: the best path's time (the truth),
# what auto took and picked, what the planner's means (PLN line) and the estimators (DEC line) said each path would take.
#   python3 -I fork/replay_score.py BUILD [BUILD2 ...] [--rows GLOB]     (q5h stats logs BUILD-<row>.stats.log)
import glob, os, re, sys, collections, statistics as st

Q5H = '/p/bw/data/q5h'
STRATS = ('auto', 'stream', 'mix', 'k1', 'dense', 'chain', 'dec')
# DON path names to short ones
PATHS = {'stream': 'stream', 'stream, mix': 'mix', 'prompt chain': 'chain', 'decode chain': 'dec', 'dense': 'dense', 'stream, k x1': 'k1', 'stream, k x2': 'k2', 'stream, k x4': 'k4', 'stream, k x8': 'k8'}


def group_of(row):
    parts = row.split('-')
    strat = next((p for p in parts if p in STRATS), None)
    return ('-'.join(p for p in parts if p != strat), strat) if strat else (row, None)


def parse(path):
    """prompt calls of one stats log: list of dict(req, idx, n, path, took, means, est)"""
    calls, req, idx, means, est = [], -1, 0, {}, {}
    for line in open(path, errors='replace'):
        p = line.split(None, 4)
        if len(p) < 5:
            continue
        typ, s = p[3], p[4]
        if typ == 'REQ' and s.startswith('begin'):
            req += 1; idx = 0
        elif typ == 'PLN' and s.startswith('plan: prompt call of'):
            means = {PATHS.get(m.group(1), m.group(1)): float(m.group(2)) for m in re.finditer(r'(?:; |: )([a-z ,0-9x]+?) (-?\d+) \+- \d+ ms', s)}
        elif typ == 'DEC' and ': router ' in s and 'full stream' in s:
            m = re.search(r'router ([\d.]+) ms per token \((\w+).*full stream ([\d.]+) ms per token \((\w+)', s)
            n = float(re.search(r'prompt call of (\d+)', s).group(1))
            if m:
                est = {'chain': float(m.group(1)) * n, 'stream': float(m.group(3)) * n, 'n': n}
        elif typ == 'DON' and s.startswith('prompt call of'):
            m = re.match(r'prompt call of (\d+) tokens: (.+?) took (\d+) ms', s)
            if not m:
                continue
            # req -1: a call before the first request (the warm-up), never keyed with request 0's calls
            calls.append(dict(req=req, idx=idx, n=int(m.group(1)), path=PATHS.get(m.group(2), m.group(2)), took=float(m.group(3)), means=means, est=est))
            idx += 1; means = {}; est = {}
    return calls


def main():
    args = sys.argv[1:]
    rows_glob = '*'
    if '--rows' in args:
        i = args.index('--rows'); rows_glob = args[i + 1]; del args[i:i + 2]
    for build in args:
        groups = collections.defaultdict(dict)   # group -> strat -> calls
        for f in sorted(glob.glob(f'{Q5H}/{build}-{rows_glob}.stats.log')):
            row = os.path.basename(f)[len(build) + 1:-len('.stats.log')]
            g, strat = group_of(row)
            if strat:
                groups[g][strat] = parse(f)
        print(f'\n== {build}')
        print(f'{"group":16} {"req":>3} {"call":>4} {"tokens":>6}  {"best path":10} {"best (ms)":>9}  {"auto ran":10} {"auto (ms)":>9} {"auto/best":>9}  {"planner mean of":>15} {"its truth (ms)":>14}  per path (ms)')
        for g, rows in sorted(groups.items()):
            truth = collections.defaultdict(lambda: collections.defaultdict(list))   # (req, idx) -> path -> took
            for strat, calls in rows.items():
                for c in calls:
                    truth[(c['req'], c['idx'], c['n'])][c['path']].append(c['took'])   # tokens in the key: rows of a group may have run another prompt
            tot_auto = tot_best = 0.0
            for c in rows.get('auto', []):
                k = (c['req'], c['idx'], c['n'])
                per = {p: st.median(v) for p, v in truth[k].items()}
                bp = min(per, key=per.get)
                tot_auto += c['took']; tot_best += per[bp]
                pm = c['means'].get(c['path'])
                print(f'{g:16} {c["req"]:>3} {c["idx"]:>4} {c["n"]:>6}  {bp:10} {per[bp]:>9.0f}  {c["path"]:10} {c["took"]:>9.0f} {c["took"] / per[bp]:>9.2f}  '
                      f'{(f"{pm:.0f}" if pm is not None else "-"):>15} {c["took"]:>14.0f}  ' + ' '.join(f'{p} {t:.0f}' for p, t in sorted(per.items(), key=lambda x: x[1])))
            if tot_best > 0:
                print(f'{g:16} all calls: auto {tot_auto:.0f} ms, best path per call {tot_best:.0f} ms, auto/best {tot_auto / tot_best:.2f}')


if __name__ == '__main__':
    main()
