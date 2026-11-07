# The planner replay gate's input (tests/test-planner-replay-fork.cpp): GLM 3.5 12k logs (q5h *.stats.log) -> per-call events (the CAL line's features, the
# DON line's outcome, the fill wall) and the measured medians per call class (size, cold/warm request, arm).
#   python3 -I fork/planner_events.py [out_dir]  ->  out_dir/glm_events.csv, out_dir/truth.csv;  then  test-planner-replay-fork events truth [seeds]
import re, glob, csv, os, sys, statistics as st, collections
out = sys.argv[1] if len(sys.argv) > 1 else '/tmp'
w = csv.writer(open(out + '/glm_events.csv', 'w'))
rows = []
w.writerow(['log', 'req', 'n', 'cells', 'Ts', 'Um', 'up', 'ch', 'path', 'took_ms', 'fill_ms'])
num = lambda p, s: float(m.group(1)) if (m := re.search(p, s)) else 0.0
for f in sorted(glob.glob('/p/bw/data/q5h/*glm35-12k-rp*.stats.log')):
    log = os.path.basename(f).replace('.stats.log', '')
    cal = None; fill = 0.0; big = 0; req = -1
    for l in open(f, errors='replace'):
        p = l.split(None, 4)
        if len(p) < 5: continue
        t, s = p[3], p[4]
        if t == 'CAL' and s.startswith('prompt call of'):
            n = num(r'prompt call of (\d+)', s)
            gb, rate = num(r'stream reads ([\d.]+) GB', s), num(r'from the SSD at ([\d.]+) GB/s', s)
            U = num(r'streaming upload (\d+) ms', s)
            cal = dict(n=n, Ts=max(U, gb / rate * 1e3 if rate > 0 else 0), Um=num(r'with the mix (\d+) ms', s), up=num(r'misses uploaded (\d+) ms', s), ch=num(r'4-token chunks (\d+) ms', s))
            fill = 0.0
        elif t == 'SPN' and 'copy fill' in s:
            fill += num(r'dur ([\d.]+) ms', s)
        elif t == 'DON' and s.startswith('prompt call of') and cal:
            m = re.match(r'prompt call of (\d+) tokens: (.+?) took (\d+) ms', s)
            if not m or float(m.group(1)) != cal['n'] or cal['n'] < 1000: cal = None; continue
            if cal['n'] < 3000: req += 1   # each request: one ~2k call, then one ~10k call
            cells = cal['n'] / 2 if cal['n'] < 3000 else (2034 if cal['n'] == 10400 else 1938) + cal['n'] / 2
            r = [log, req, int(cal['n']), cells, cal['Ts'], cal['Um'], cal['up'], cal['ch'], m.group(2), m.group(3), round(fill)]
            w.writerow(r); rows.append(r)
            cal = None

# the measured medians per call class: the outcome a replayed pick gets (stream, mix and predictions pooled as 'stream')
T = collections.defaultdict(list)
for r in rows:
    p = r[8]; arm = 'dense' if p == 'dense' else 'dec' if p == 'decode chain' else 'chain' if p == 'prompt chain' else 'stream'
    T[('2k' if r[2] < 3000 else '10k', 'c' if r[1] == 0 else 'w', arm)].append(float(r[9]))
t = csv.writer(open(out + '/truth.csv', 'w'))
for k, v in sorted(T.items()): t.writerow(list(k) + [st.median(v), len(v)])
