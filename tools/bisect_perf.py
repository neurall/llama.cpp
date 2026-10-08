#!/usr/bin/env python3
"""bisect_perf.py --good REF --bad REF --metric off|c19 [--n 4]: which commit made the build slower? Binary search over `git rev-list --reverse good..bad` (every commit, topological
order), one persistent worktree (incremental dev builds, ccache), one measurement per step with run.py (chat endpoint, OLMoE, 500 tokens, GPU1, learned state off):
  off  the whole model on the GPU, no cache (--moe-expert-cache 0): the shared path (kernels, graph, loader)
  c19  19 of 64 experts per layer in VRAM (--cpu-moe --moe-expert-cache 19): the cache path
good/bad speed = a measurement of each end first; a commit is 'fast' when its median is above the middle. A commit that does not build or run is skipped (its neighbour is tried).
Each step holds /tmp/queue.lock around build AND run (a build never overlaps a benchmark). Log: /p/bw/data/bisect/<name>.log, every run in run-history.csv (campaign bisect-<name>)."""
import argparse, csv, fcntl, os, statistics, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.realpath(__file__)); REPO = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--good", required=True); ap.add_argument("--bad", required=True); ap.add_argument("--metric", default="off", choices=["off", "c19"])
ap.add_argument("--n", type=int, default=4); ap.add_argument("--name"); ap.add_argument("--tree", default="/p/bw/build-tmp/bisect"); a = ap.parse_args()
name = a.name or f"{a.metric}-{a.good[:7]}-{a.bad[:7]}"; os.makedirs("/p/bw/data/bisect", exist_ok=True); LOG = f"/p/bw/data/bisect/{name}.log"
def say(*x):
    s = time.strftime("%F %T ") + " ".join(map(str, x)); print(s, flush=True); open(LOG, "a").write(s + "\n")
def git(*c, cwd=None): return subprocess.run(["git", *c], cwd=cwd or REPO, capture_output=True, text=True).stdout.strip()
commits = git("rev-list", "--reverse", "--topo-order", f"{a.good}..{a.bad}").split()
say(f"{len(commits)} commits between {a.good[:9]} and {a.bad[:9]}, metric {a.metric}")
if not os.path.exists(a.tree):
    os.makedirs(os.path.dirname(a.tree), exist_ok=True); subprocess.run(["git", "worktree", "add", "--detach", "-f", a.tree, a.bad], cwd=REPO, check=True)
ARGS = {"off": "-ngl 99 --n-cpu-moe 0 --moe-expert-cache 0", "c19": "--cpu-moe --moe-expert-cache 19"}[a.metric]
def measure(ref, tag):
    lock = open("/tmp/queue.lock", "a"); fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        subprocess.run(["git", "checkout", "-q", "-f", ref], cwd=a.tree, check=True)
        r = subprocess.run(["bash", "/p/bw/data/hcf/lrel/dev-build.sh"], env={**os.environ, "W": a.tree, "REGRESS": "0"}, capture_output=True, text=True)
        exe = os.path.join(a.tree, "build-dev", "bin", "llama-server")
        if "DEVBUILD_EXIT 0" not in r.stdout or not os.path.exists(exe): say(ref[:9], "BUILD FAILED"); return None
        stage = tempfile.mkdtemp(prefix="bis-"); label = f"bis-{ref[:9]}"; os.symlink(os.path.join(a.tree, "build-dev", "bin"), os.path.join(stage, label))
        camp = f"bisect-{name}-{tag}-{ref[:9]}"
        env = dict(os.environ, PERF_BUILDS=stage, PERF_LOG="r", CUDA_VISIBLE_DEVICES="1", PERF_DEV="CUDA0", CUDA_DEVICE_ORDER="PCI_BUS_ID", PERF_THREADS="6")
        subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "bench", "--models", "/m/o/8/4/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "--builds", label, "--variants", "v", "--variant", "v=" + ARGS,
                        "-t", "tetrisc", "--tokens", "500", "--ctx", "2048", "-n", str(a.n), "--clean-state", "--campaign", camp], env=env, capture_output=True, text=True)
        v = [float(x["tps"]) for x in csv.DictReader(open(os.path.join(HERE, "bench", "run-history.csv"))) if x["campaign"] == camp and x["ok"] != "0" and x["tps"]]
        return statistics.median(v) if v else None
    finally:
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
