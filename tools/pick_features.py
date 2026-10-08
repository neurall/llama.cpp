#!/usr/bin/env python3
"""pick_features.py GROUPS.txt [--tree /p/bw/wt-base509]: put fork features on the fast base one group at a time, each group tested before the next.
GROUPS.txt: lines 'name: sha sha sha ...' (commits in order). Per group: git cherry-pick them in the tree (a conflict: the group is skipped, tree reset), dev build,
`run.py regress` (restricted OLMoE, 19 slots, 500 tokens, against the best binary kept in /p/bw/rels/best): REGRESSION (below 0.98 x best) drops the group (tree reset to the
last good state), PASS or NEW BEST keeps it. Result per group in /p/bw/data/pick/<tree name>.log and a line in the final table."""
import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.realpath(__file__)); tree = "/p/bw/wt-base509"; args = sys.argv[1:]
if "--tree" in args: i = args.index("--tree"); tree = args[i + 1]; del args[i:i + 2]
groups = [(l.split(":", 1)[0].strip(), l.split(":", 1)[1].split()) for l in open(args[0]) if ":" in l and not l.startswith("#")]
os.makedirs("/p/bw/data/pick", exist_ok=True); LOG = f"/p/bw/data/pick/{os.path.basename(tree)}.log"
def say(*x):
    s = time.strftime("%F %T ") + " ".join(map(str, x)); print(s, flush=True); open(LOG, "a").write(s + "\n")
def git(*c): return subprocess.run(["git", *c], cwd=tree, capture_output=True, text=True)
res = []
for name, shas in groups:
    base = git("rev-parse", "HEAD").stdout.strip(); ok = True
    for s in shas:
        r = git("cherry-pick", s)
        if r.returncode:
            if "empty" in (r.stdout + r.stderr) or "previous cherry-pick is now empty" in (r.stdout + r.stderr): git("cherry-pick", "--skip"); continue
            say(name, "CONFLICT at", s[:9], (r.stderr or r.stdout).strip().splitlines()[-1][:100]); git("cherry-pick", "--abort"); ok = False; break
    if not ok: git("reset", "--hard", base); res.append((name, "conflict", "")); continue
    import fcntl
    lk = open("/tmp/queue.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)   # a build never overlaps a benchmark
    try:
        b = subprocess.run(["bash", "/p/bw/data/hcf/lrel/dev-build.sh"], env={**os.environ, "W": tree, "REGRESS": "0"}, capture_output=True, text=True)
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    if "DEVBUILD_EXIT 0" not in b.stdout:
        open(f"/p/bw/data/pick/build-{name}.log", "w").write(b.stdout[-4000:] + b.stderr[-2000:]); say(name, "BUILD FAILED (tail in /p/bw/data/pick/build-" + name + ".log)"); git("reset", "--hard", base); res.append((name, "build failed", "")); continue
    r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "regress", os.path.join(tree, "build-dev", "bin"), "--wait"], capture_output=True, text=True)
    if r.returncode == 1:
        say(name, "first verdict", [l for l in r.stdout.splitlines() if l.startswith("regress:")][-1:], "-> again with 30 runs")
        r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "regress", os.path.join(tree, "build-dev", "bin"), "--wait", "-n", "30"], capture_output=True, text=True)
    out = r.stdout.strip().splitlines(); verdict = next((l for l in reversed(out) if l.startswith("regress:")), "regress: no verdict")
    say(name, f"exit {r.returncode}", verdict)
    if r.returncode in (1, 2, 3): git("reset", "--hard", base); res.append((name, "DROPPED", verdict))
    else:
        m = re.search(r"([0-9.]+) x the (?:best|previous best)", verdict)
        gain = f"{(float(m.group(1)) - 1) * 100:+.1f}% against the best so far" if m else "gain: this build is the new reference" if "BEST is now" in verdict or "this build is the best" in verdict else "no ratio"
        if "tf32" in name:   # the line now measures with the TF32 switch: this build is the reference from here on
            subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "regress", os.path.join(tree, "build-dev", "bin"), "--wait", "--rebase"], capture_output=True, text=True)
        say(f"FEATURE ADDED: {name} ({len(shas)} commits), {gain}")
        res.append((name, "kept", gain))
say("SUMMARY"); [say(f"  {n:24s} {s:12s} {v}") for n, s, v in res]
