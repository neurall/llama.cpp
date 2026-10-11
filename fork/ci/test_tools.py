#!/usr/bin/env python3
"""Checks of the fork's tooling that need no GPU and no model (run by .github/workflows/fork-ci.yml, also locally: python3 fork/ci/test_tools.py).

1. every fork/*.py compiles, run.py and series.py answer --help
2. the cache simulator on a seeded synthetic routing trace keeps its invariants: the offline optimum (belady) is never beaten, counting policies beat plain LRU
   on a skewed trace, and two runs print the same numbers
3. the config module: defaults, a TOML file, and the FORK_* variables in that order"""
import os, py_compile, random, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.dirname(HERE)
PY = sys.executable


def check(cond, msg):
    if not cond:
        sys.exit("FAIL: " + msg)
    print("ok:", msg)


def test_compile_and_help():
    for f in sorted(os.listdir(FORK)):
        if f.endswith(".py"):
            py_compile.compile(os.path.join(FORK, f), doraise=True)
    check(True, "every fork/*.py compiles")
    for tool in ("run.py", "series.py"):
        r = subprocess.run([PY, os.path.join(FORK, tool), "--help"], capture_output=True, text=True, timeout=60)
        check(r.returncode == 0 and "usage" in r.stdout.lower(), f"{tool} --help")


def synthetic_trace(path, n_exp=64, n_layer=4, n_tok=600, k=6, seed=7):
    rnd = random.Random(seed)
    w = [1.0 / (i + 1) for i in range(n_exp)]   # skewed: a few experts do most of the work, like real routers
    with open(path, "w") as f:
        for _ in range(n_tok):
            for il in range(n_layer):
                ids = set()
                while len(ids) < k:
                    ids.add(rnd.choices(range(n_exp), weights=w)[0])
                f.write("blk.%d.ffn_gate_exps %s\n" % (il, " ".join(map(str, sorted(ids)))))


def sim(trace):
    r = subprocess.run([PY, os.path.join(FORK, "sim", "cache_sim.py"), trace, "--frac", "0.25"], capture_output=True, text=True, timeout=300)
    check(r.returncode == 0, "cache_sim.py runs")
    line = next(l for l in r.stdout.splitlines() if l.startswith("cache"))
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)\s+([\d.]+)%", line.split("|")[0])}


def test_cache_sim():
    with tempfile.TemporaryDirectory() as d:
        t = os.path.join(d, "trace.txt")
        synthetic_trace(t)
        a, b = sim(t), sim(t)
    check(a == b, "the simulator is deterministic")
    check(a["belady"] >= max(v for k, v in a.items() if k != "belady"), f"belady is the upper bound ({a['belady']}%)")
    check(a["lfu"] > a["lru"], f"on a skewed trace LFU beats LRU ({a['lfu']}% vs {a['lru']}%)")


def test_config():
    sys.path.insert(0, FORK)
    import importlib, config
    for k in [k for k in os.environ if k.startswith("FORK_")]:
        del os.environ[k]
    with tempfile.TemporaryDirectory() as d:
        toml = os.path.join(d, "c.toml")
        open(toml, "w").write('[paths]\nrels = "/from/file"\n')
        os.environ["FORK_CONFIG"] = toml
        c = importlib.reload(config)
        check(c.path("rels") == "/from/file", "a TOML setting overrides the default")
        check(c.path("best").endswith("/rels/best"), "an unset key keeps its default")
        os.environ["FORK_PATHS_RELS"] = "/from/env"
        check(c.path("rels") == "/from/env", "the FORK_* variable wins over the file")
        del os.environ["FORK_PATHS_RELS"], os.environ["FORK_CONFIG"]


if __name__ == "__main__":
    test_compile_and_help()
    test_cache_sim()
    test_config()
    print("all fork tool checks passed")
