#!/usr/bin/env python3
"""bisect_perf.py --good REF --bad REF --metric off|c19 [--n 4]: which commit made the build slower? Binary search over `git rev-list --reverse good..bad` (every commit, topological
order), one persistent worktree (incremental dev builds, ccache), one measurement per step with run.py (chat endpoint, OLMoE, 500 tokens, GPU1, learned state off):
  off  the whole model on the GPU, no cache (--moe-expert-cache 0): the shared path (kernels, graph, loader)
  c19  19 of 64 experts per layer in VRAM (--cpu-moe --moe-expert-cache 19): the cache path
good/bad speed = a measurement of each end first; a commit is 'fast' when its median is above the middle. A commit that does not build or run is skipped (its neighbour is tried).
Each step holds /tmp/queue.lock around build AND run (a build never overlaps a benchmark). Log: <build_tmp>/bisect-logs/<name>.log, every run in run-history.csv (campaign bisect-<name>)."""
import argparse, csv, fcntl, os, shutil, statistics, subprocess, sys, time
HERE = os.path.dirname(os.path.realpath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import config as fcfg
devbuilds = fcfg.path("devbuilds")
ap = argparse.ArgumentParser(); ap.add_argument("--good", required=True); ap.add_argument("--bad", required=True); ap.add_argument("--metric", default="off", choices=["off", "c19"])
ap.add_argument("--n", type=int, default=4); ap.add_argument("--name"); ap.add_argument("--tree", default=os.path.join(fcfg.path("build_tmp"), "bisect")); a = ap.parse_args()
name = a.name or f"{a.metric}-{a.good[:7]}-{a.bad[:7]}"; LOGDIR = os.path.join(fcfg.path("build_tmp"), "bisect-logs"); os.makedirs(LOGDIR, exist_ok=True); LOG = os.path.join(LOGDIR, f"{name}.log")
def say(*x):
    s = time.strftime("%F %T ") + " ".join(map(str, x)); print(s, flush=True); open(LOG, "a").write(s + "\n")
def git(*c, cwd=None): return subprocess.run(["git", *c], cwd=cwd or REPO, capture_output=True, text=True).stdout.strip()
commits = git("rev-list", "--reverse", "--topo-order", f"{a.good}..{a.bad}").split()
say(f"{len(commits)} commits between {a.good[:9]} and {a.bad[:9]}, metric {a.metric}")
if not os.path.exists(a.tree):
    os.makedirs(os.path.dirname(a.tree), exist_ok=True); subprocess.run(["git", "worktree", "add", "--detach", "-f", a.tree, a.bad], cwd=REPO, check=True)
ARGS = {"off": "-ngl 99 --n-cpu-moe 0 --moe-expert-cache 0", "c19": "--cpu-moe --moe-expert-cache 19"}[a.metric]
def measure(ref, tag):
    """build `ref` in the persistent worktree (run.py build, the queue lock is held here), measure it, remove the build"""
    lock = open("/tmp/queue.lock", "a"); fcntl.flock(lock, fcntl.LOCK_EX)
    label = f"bis-{ref[:9]}"; dst = os.path.join(devbuilds, label)
    try:
        subprocess.run(["git", "checkout", "-q", "-f", ref], cwd=a.tree, check=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "build", label, "--tree", a.tree, "--no-regress"],
                           env={**os.environ, "QUEUE_HOLDS_LOCK": "1"}, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(os.path.join(dst, "llama-server")):
            say(ref[:9], "BUILD FAILED", (r.stdout + r.stderr)[-300:].replace("\n", " | ")); return None
        camp = f"bisect-{name}-{tag}-{ref[:9]}"
        env = dict(os.environ, PERF_BUILDS=devbuilds, PERF_LOG="r", CUDA_VISIBLE_DEVICES="1", PERF_DEV="CUDA0", CUDA_DEVICE_ORDER="PCI_BUS_ID", PERF_THREADS="6")
        subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "bench", "--models", fcfg.model("olmoe"), "--builds", label, "--variants", "v", "--variant", "v=" + ARGS,
                        "-t", "tetrisc", "--tokens", "500", "--ctx", "2048", "-n", str(a.n), "--clean-state", "--campaign", camp], env=env, capture_output=True, text=True)
        v = [float(x["tps"]) for x in csv.DictReader(open(os.path.join(HERE, "bench", "run-history.csv"))) if x["campaign"] == camp and x["ok"] != "0" and x["tps"]]
        return statistics.median(v) if v else None
    finally:
        shutil.rmtree(dst, ignore_errors=True)   # a bisect would otherwise leave a CUDA build per step
        fcntl.flock(lock, fcntl.LOCK_UN)
g = measure(commits[0] if False else a.good, "good"); b = measure(a.bad, "bad"); say("good end", g, "bad end", b)
if not g or not b or g <= b * 1.005: sys.exit(say("no usable difference between the ends: nothing to bisect") or 1)
mid = (g + b) / 2; lo, hi = -1, len(commits) - 1   # commits[lo] is fast (the good end counts as -1), commits[hi] is slow
while hi - lo > 1:
    i = (lo + hi) // 2; v = None; j = i
    for j in [i] + [i + d for k in range(1, 6) for d in (k, -k) if lo < i + d < hi]:
        v = measure(commits[j], f"s{j}")
        if v is not None: break
    if v is None: say("six commits in a row do not build: stop"); break
    say(f"[{lo}..{hi}] step {j} {commits[j][:9]} {v:.2f} t/s ({'fast' if v > mid else 'slow'}; middle {mid:.2f})  {git('log', '-1', '--format=%s', commits[j])[:90]}")
    if v > mid: lo = j
    else: hi = j
c = commits[hi]; say("FIRST SLOW COMMIT:", c[:9], git("log", "-1", "--format=%cs %an %s", c)[:150]); say(git("show", "--stat", "--format=", c)[:1500])
