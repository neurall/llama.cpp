#!/usr/bin/env python3
"""Regression checks, one per regression that was ever spotted, numbered. Is regression N back in a build?

  reg_tests.py list                      the regressions: number, title, what it was, the stored last-good build
  reg_tests.py N CURRENT [GOOD]          run check N on build CURRENT (a bin directory) and on the stored last-good build (or GOOD),
                                         through tools/run.py exp (every run lands in run-history.csv), print PASS or REGRESSION
  reg_tests.py N CURRENT --dry           only print the experiment that would run
  reg_tests.py all CURRENT               every check that can run on this machine now (a check that needs a pool says SKIP without one)

CURRENT is a directory with llama-cli (and libs), e.g. /p/bw/wt-release-next/build-dev/bin. The last-good builds are stored under /p/bw/rels.
A check compares CURRENT with the last-good build on the same machine, same tool (llama-cli on both: llama-cli and llama-completion print different
rates for the same engine), own state file per arm (an arm must never learn from another arm's runs).
"""
import csv, glob, os, re, shlex, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
RUN = os.path.join(HERE, "run.py")
HIST = os.path.join(HERE, "bench", "run-history.csv")
LOGS = os.path.join(HERE, "bench", "logs")
PROMPT = os.path.join(HERE, "bench", "reg-prompt.txt")   # a ~650 token prompt, the one every check uses
RELS = "/p/bw/rels"
GOOD = RELS + "/rc-final-fd4c3d2d3"                      # rc-merge fd4c3d2d3: placement measured, pin decision with pool, no mid-split rule

# the stored builds: directory name under RELS -> the commit it is built from. A build that is missing is rebuilt from its commit in a separate temporary worktree
# (the docker dev build, nothing touches the current tree), the binaries are stored under RELS with a BUILD_INFO, the temporary tree is removed.
BUILDS = {
    "rc-final-fd4c3d2d3":  {"ref": "fd4c3d2d3", "note": "rc-merge: placement measured, pin decision with pool, mid-split rule removed"},
    "b11707-49fe4b756":    {"ref": "49fe4b756", "note": "release b11707 (the README numbers)"},
    "stock-bed0a8566":     {"ref": "bed0a8566", "note": "upstream master, the stock floor"},
    "stock-def4d406a":     {"ref": "def4d406a", "note": "upstream master, the older stock"},
}
BUILD_SCRIPT = "/p/bw/data/hcf/lrel/dev-build.sh"
BUILD_TMP = "/p/bw/build-tmp"


def ensure_build(path, dry=False):
    """path: a build directory. Missing: build it from BUILDS[basename] in a temporary worktree and store it. Returns None or an error text."""
    if os.path.exists(os.path.join(path, "llama-cli")):
        return None
    name = os.path.basename(path.rstrip("/"))
    spec = BUILDS.get(name)
    if not spec:
        return f"no llama-cli in {path} and no recipe for '{name}' in BUILDS (add one: the commit it is built from)"
    if dry:
        print(f"[dry] would build {name} from {spec['ref']} in {BUILD_TMP}/{name} and store it in {path}")
        return None
    tmp = os.path.join(BUILD_TMP, name)
    os.makedirs(BUILD_TMP, exist_ok=True)
    print(f"building {name} from {spec['ref']} in {tmp} (docker dev build, several minutes)", flush=True)
    for cmd in (["git", "-C", REPO, "worktree", "add", "--detach", "-f", tmp, spec["ref"]],):
        r = run(cmd)
        if r.returncode:
            return f"git worktree failed: {r.stderr.strip()[:200]}"
    try:
        r = subprocess.run(["bash", BUILD_SCRIPT], env={**os.environ, "W": tmp}, text=True)
        binp = os.path.join(tmp, "build-dev", "bin")
        if not os.path.exists(os.path.join(binp, "llama-cli")):
            return f"build of {name} failed (no llama-cli in {binp}, exit {r.returncode})"
        os.makedirs(path, exist_ok=True)
        subprocess.run(["cp", "-a", binp + "/.", path], check=True)
        with open(os.path.join(path, "BUILD_INFO"), "w") as f:
            f.write(f"commit={spec['ref']}\nbranch=reg_tests\npatch=0\nnote={spec.get('note', '')}\n")
        print(f"stored {path}", flush=True)
        return None
    finally:
        run(["git", "-C", REPO, "worktree", "remove", "--force", tmp])
        subprocess.run(["rm", "-rf", tmp])


M = {
    "qn_iq1m": "/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
    "glm30":   "/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
}

# each regression: what it was, how it shows, the model, the arms' extra args/env, the metric and the tolerance (CURRENT must reach tol x the good build)
REGS = {
    1: dict(title="near-fit model forced onto the expert cache (first start and every start)",
            was="Qwen3.8-Flash-Next IQ1_M (55 GB, 48 GB VRAM): commit aab03fbe8 took the cache without measuring: 54.7 t/s generation and 469 t/s prompt against 66 and 920 with stock placement",
            model="qn_iq1m", rounds=4, args="", env={}, state=True, judge="settled", tol=0.93,
            good_build=GOOD, how="the last two starts (settled) of a fresh state file: generation and prompt t/s"),
    2: dict(title="extra graph split on a plain two-GPU layer split (mid-split rule)",
            was="the scheduler started a new split when an activation from another GPU was needed: 4 splits where stock has 3, 1.1% generation on Qwen Next (65.9 against 66.6 without it)",
            model="qn_iq1m", rounds=3, args="--fork off", env={}, state=False, judge="splits", tol=0.99,
            good_build=GOOD, how="'splits =' of the graph in the log must not exceed the good build's, generation >= 0.99 x good"),
    3: dict(title="pin decision ignores a reserved huge page pool",
            was="with a 100 GiB pool reserved only ~21 GiB 'RAM available': a 109 GiB model fell back to mmap and ran at 2.5 t/s (GLM 3.0-bit)",
            model="glm30", rounds=2, args="", env={}, state=True, judge="pinned", tol=0.85, needs="pool",
            good_build=GOOD, how="log says 'weights pinned' and generation >= 0.85 x good (first start, stock placement)"),
    4: dict(title="pin decision ignores a warm hugetlbfs weights cache",
            was="with GGML_CUDA_HUGEFS holding the weights (68 GiB of the pool) the decision saw too little RAM: mmap fallback, 2.0 t/s on GLM 3.0-bit",
            model="glm30", rounds=2, args="", env={"GGML_CUDA_HUGEFS": "/mnt/huge1g"}, state=True, judge="pinned", tol=0.85, needs="pool",
            good_build=GOOD, how="the second start (cache file warm) says 'weights pinned' and generation >= 0.85 x good"),
}


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def pool_free_gib():
    try:
        return int(open("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages").read())
    except OSError:
        return 0


def spec_text(n, reg, name, cur, good):
    arms = []
    for arm, b in (("GOOD", good), ("CUR", cur)):
        a = f"[arm {arm}]\nbin = {b}\n"
        if reg["args"]:
            a += f"args = {reg['args']}\n"
        env = " ".join(f"{k}={v}" for k, v in reg["env"].items())
        if env:
            a += f"env = {env}\n"
        if reg["state"]:
            a += f"state = /tmp/reg{n}-state-{arm}.ini\n"
        arms.append(a)
    return f"""[exp]
name = {name}
note = regression {n}: {reg['title']}
rounds = {reg['rounds']}
order = fixed

[models]
m = {M[reg['model']]}

[workload]
tool = cli
prompt_file = {PROMPT}
n = 200
log_verbosity = 4
timeout = 2400

[setup]
guard = min_avail_gb=20 max_swap_mb=8000 swapout_pages_s=400000 psi=60

""" + "\n".join(arms)


def rows(name):
    return [r for r in csv.DictReader(open(HIST)) if r.get("campaign") == name and r.get("tps") and r.get("temp", "") != "cold"]


def log_of(name, arm, rnd):
    f = os.path.join(LOGS, name, f"m-r{rnd}-{arm}.log")
    return open(f, errors="replace").read() if os.path.exists(f) else ""


def judge(n, reg, name):
    r = {a: sorted([x for x in rows(name) if x["arm"] == a], key=lambda x: int(x["rep"])) for a in ("GOOD", "CUR")}
    if not r["GOOD"] or not r["CUR"]:
        return "ERROR", "no rows for one of the arms (the runs failed: see the log)"
    g = lambda a, k, last=None: statistics.median(float(x[k]) for x in (r[a][-last:] if last else r[a]) if x.get(k))
    j, tol = reg["judge"], reg["tol"]
    if j == "settled":   # the last two starts of a fresh state
        gt, ct, gp, cp = g("GOOD", "tps", 2), g("CUR", "tps", 2), g("GOOD", "pp", 2), g("CUR", "pp", 2)
        ok = ct >= tol * gt and cp >= tol * gp
        return ("PASS" if ok else "REGRESSION"), f"settled starts: generation {ct:.1f} against good {gt:.1f} ({100*ct/gt:.0f}%), prompt {cp:.0f} against {gp:.0f} ({100*cp/gp:.0f}%), needs {100*tol:.0f}%"
    if j == "splits":
        sp = lambda a: [int(m.group(1)) for m in (re.search(r"graph: nodes = \d+, splits = (\d+)", log_of(name, a, int(x["rep"]) + 1)) for x in r[a]) if m]
        gs, cs = sp("GOOD"), sp("CUR")
        gt, ct = g("GOOD", "tps"), g("CUR", "tps")
        ok = bool(gs) and bool(cs) and max(cs) <= max(gs) and ct >= tol * gt
        return ("PASS" if ok else "REGRESSION"), f"splits {max(cs) if cs else '?'} against good {max(gs) if gs else '?'}, generation {ct:.1f} against {gt:.1f} ({100*ct/gt:.1f}%), needs {100*tol:.1f}%"
    if j == "pinned":
        idx = len(r["CUR"]) - 1 if n == 4 else 0
        lg = log_of(name, "CUR", int(r["CUR"][idx]["rep"]) + 1)
        pinned = "weights pinned" in lg
        gt, ct = float(r["GOOD"][idx]["tps"]), float(r["CUR"][idx]["tps"])
        ok = pinned and ct >= tol * gt
        return ("PASS" if ok else "REGRESSION"), f"{'pinned' if pinned else 'NOT pinned (mmap fallback)'}, generation {ct:.1f} against good {gt:.1f} ({100*ct/gt:.0f}%), needs {100*tol:.0f}%"
    return "ERROR", f"unknown judge {j}"


def check(n, cur, good=None, dry=False):
    reg = REGS[n]
    good = good or reg["good_build"]
    if reg.get("needs") == "pool" and pool_free_gib() < 100:
        return "SKIP", "needs a free 1 GiB hugepage pool of 100 pages (sudo pool.sh mount 100G right after boot)"
    if not os.path.exists(os.path.join(cur, "llama-cli")):
        return "ERROR", f"no llama-cli in {cur}"
    err = ensure_build(good, dry)   # the stored last-good build, rebuilt from its commit when missing
    if err:
        return "ERROR", err
    name = f"reg{n}-{time.strftime('%m%d-%H%M')}"
    spec = f"/tmp/{name}.ini"
    open(spec, "w").write(spec_text(n, reg, name, cur, good))
    for arm in ("GOOD", "CUR"):
        if reg["state"] and os.path.exists(f"/tmp/reg{n}-state-{arm}.ini"):
            os.remove(f"/tmp/reg{n}-state-{arm}.ini")
    if dry:
        print(open(spec).read())
        return "DRY", spec
    p = subprocess.run([sys.executable, RUN, "exp", spec])
    if p.returncode != 0:
        return "ERROR", f"run.py exp exited {p.returncode}"
    return judge(n, reg, name)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "list":
        for n, r in REGS.items():
            print(f"reg {n}: {r['title']}\n   was:  {r['was']}\n   test: {r['how']}\n   last good: {r['good_build']}  model: {M[r['model']]}")
        return 0
    if len(a) < 2:
        print("usage: reg_tests.py N CURRENT_BIN_DIR [GOOD_BIN_DIR] [--dry]   |   reg_tests.py all CURRENT_BIN_DIR   |   reg_tests.py list")
        return 2
    dry = "--dry" in a
    a = [x for x in a if x != "--dry"]
    ns = list(REGS) if a[0] == "all" else [int(a[0])]
    bad = 0
    for n in ns:
        status, msg = check(n, os.path.abspath(a[1]), a[2] if len(a) > 2 else None, dry)
        print(f"reg {n} [{REGS[n]['title']}]: {status}: {msg}")
        bad += status in ("REGRESSION", "ERROR")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
