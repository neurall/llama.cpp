#!/usr/bin/env python3
"""Benchmark harness of the llama.cpp expert-cache fork: every run is one line of run-history.csv (append-only, key columns first).

  run.py bench  --models M1,M2 --builds autotune,stock-X [--variants default,cache0,atoff] [-t t100] [-n 2] [--machine pc1]
  run.py report [--campaign NAME] [--last N] [--tol 0.95]      variants against stock, per model / test / machine
  run.py jobs    [-t TEST] [-n N] [-e K=V ...] [--bare] [--plain] [--args '...'] BUILD...   (MODEL=/path/model.gguf)
  run.py machine [add ID --ssh user@host --root DIR --os linux|windows]    machines reached over ssh (llama-cli runs)
  run.py prompt short|12k|128k|chatv|code [--url U]            the benchmark prompts against a running server
  run.py ctl    BUILD name:KEY=V,KEY=V ...                     one server, settings switched at runtime, rotating prompts
  run.py show [-t TEST] [-b BUILD] [--runs]
Everything is a parameter: models, builds, variants (--variant NAME=ARGS), test, --prompt TEXT|@file, --tokens, --ctx, machines.

BUILD is a subdirectory of the current directory (or $PERF_BUILDS) holding a build's bin/
contents, e.g. b11214-a3768a8; on another machine, a subdirectory of that machine's --root. Results go to run-history.csv in
fork/bench ($RUN_DATA). MODEL=/path/model.gguf is required for run/ctl; PERF_MODELS_DIR=/dir drops other *.gguf under it from RAM before
measuring. A measured run can be preceded by a discarded run with the same settings (hot runs; off by default, --warm turns it on); every run gets its own server
or process, the build's own libs (LD_LIBRARY_PATH: native builds bake their build dir into RUNPATH) and a pinned GPU order
(-dev CUDA0,CUDA1). The build number and commit of each row come from the binary's own --version. Self-tune decisions of a
run go to table tune_log.
Tests:
  ppl     fixed text, one token per decode call (llama-perplexity -b 1 -ub 1, longsrc.cpp code,
          2 x 1024 tokens): identical input for every build, the precise decode comparator
  ppl3    same as ppl with -b 3 -ub 3: 3-token decode batches, the path MTP/speculative verify
          uses (small-batch cache chain); compare PPL with --plain for correctness
  tetris  raw completion "generate smallest html tetris game.", -c 1024, until full
  t100    raw completion "write smallest html tetris game", -c 1024, 100 tokens, temperature 0: the short one-shot run most people benchmark
  chat    /v1/chat/completions "write smallest html tetris game", 1500 tokens
  chat2   chat like chat with a ~350-token prompt (the same game with a requirements list), 1500 tokens: a real prompt, so pp counts
  long1k  ~1024-token prompt (src_1k.cpp + a bug-fix request), 1024 tokens out, -c 4096
  chatv   chat with varied prompts (run counter picks the prompt, thinking off), 300 tokens: topic switches the cache has to follow
  pf128k  128k-token prefill: src_128k.cpp, 64 tokens generated, -c 131072 (like pf12k otherwise)
  pf12k   12k-token prefill: src_12k.cpp as a raw completion prompt, 32 tokens generated,
          x16 GPU first (-dev CUDA1,CUDA0), -ub 2048 -b 2048; pf12k-stock: same, no cache flags
  agent   miniagentic ($PERF_AGENT_DIR, agentic loop): model writes a C program, 2 missing ';' and a misspelled printf are
          injected, gcc errors + full source go back for a fix, up to 3 fix rounds; prompt
          caching on (-c 16384 -ub 2048 -b 2048): long code prompts + code decode
Compare decode t/s only between runs with the same output md5 (tetris/chat): the text
decides the cache hit rate. LLAMA_MOE_CACHE_DETERMINISTIC=1 makes a build repeat itself.
"""
import argparse, math, csv, glob, hashlib, io, json, os, re, shlex, shutil, signal, statistics, subprocess, sys, threading, time, urllib.request

import types, collections, tempfile


def _register(name, ns):
    """the namespace returned by a merged library section as module `name`: every `import name` in this file and in the sections keeps working"""
    m = types.ModuleType(name, ns.get('__doc__'))
    m.__dict__.update(ns)
    sys.modules[name] = m
    return m


# ======================================================================================================================
# Libraries of the fork tooling, merged from their former files (config, runnorm, runstats, runexp, runlog).
# ======================================================================================================================

def _lib_config():
    # ---- config.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """Machine-specific settings of the fork tools (paths, model files, W10 build host), kept out of the code.

Defaults suit a plain Linux box (everything under ~/llama-fork, models under ~/models). Override in a TOML file, first found wins:
  $FORK_CONFIG, ./fork/local.toml (ignored by git), ~/.config/llama-fork/config.toml
and per setting with the environment variable FORK_<SECTION>_<KEY> (upper case), e.g. FORK_PATHS_RELS=/data/rels.

  [paths]   rels  gh_rel  devbuilds  release_out  images  best  ccache  ccache_stock  build_repo  build_tmp  work  agent
  [models]  olmoe  qwen36  qwen38  glm  mimo        (the regress cells and run.py baseline skip a model whose file is missing)
  [hosts]   w10_ssh  w10_root  history_sources                         (the Windows laptop used by `run.py build --machine pc3`; the ssh target may be an alias from ~/.ssh/config)
  [gpu]     nvidia_smi  ld_extra  limits_env  limits_service  env_sh   (power-limit tools run.py plcal / run.py baseline: nvidia-smi to call through sudo, the files that keep the limits at boot, an
                                                              optional shell file whose exports every queue task gets; empty = not used)
"""
    import glob, os

    HOME = os.path.expanduser("~")
    # everything lives under one directory: $FORK_HOME, default <repo>/.fork (git-ignored), so a clone needs no setup and no absolute path
    _BASE = os.environ.get("FORK_HOME") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".fork")
    DEFAULTS = {
        "paths": {
            "rels": _BASE + "/rels", "gh_rel": _BASE + "/stock", "devbuilds": _BASE + "/rels/devbuilds", "release_out": _BASE + "/rels/release-out",
            "images": _BASE + "/rels/images", "best": _BASE + "/rels/best", "ccache": _BASE + "/ccache", "ccache_stock": _BASE + "/ccache-stock",
            "build_repo": os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_tmp": _BASE + "/build-tmp", "work": _BASE + "/work", "agent": "",
        },
        "models": {
            "olmoe": _BASE + "/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
            "qwen36": _BASE + "/models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf",
            "qwen38": _BASE + "/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
            "glm": _BASE + "/models/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
            "mimo": _BASE + "/models/MiMo-V2.6-Flash-MOPD-MXFP4.gguf",
        },
        "hosts": {"w10_ssh": "", "w10_root": "N:", "history_sources": ""},
        "gpu": {"nvidia_smi": "nvidia-smi", "ld_extra": "", "limits_env": "", "limits_service": "", "env_sh": ""},
    }


    def _load():
        import tomllib
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.environ.get("FORK_CONFIG"), os.path.join(here, "local.toml"), os.path.join(HOME, ".config", "llama-fork", "config.toml")):
            if p and os.path.isfile(p):
                with open(p, "rb") as f:
                    return tomllib.load(f), p
        return {}, None


    _FILE, SOURCE = _load()


    def get(section, key):
        """environment FORK_<SECTION>_<KEY>, then the TOML file, then the default"""
        env = os.environ.get(f"FORK_{section}_{key}".upper())
        if env is not None:
            return env
        if key in _FILE.get(section, {}):
            return _FILE[section][key]
        return DEFAULTS[section][key]


    def path(key):
        return os.path.expanduser(get("paths", key))


    def models_ini():
        """fork/models.ini: [alias] path = full model file, url = where it came from; {alias: {path, url}}"""
        import configparser
        here = os.path.dirname(os.path.abspath(__file__))
        cp = configparser.ConfigParser(defaults={"dir": os.environ.get("FORK_MODELS_DIR") or _FILE.get("models", {}).get("dir") or _BASE + "/models"})
        cp.read(os.path.join(here, "models.ini"))
        return {k: dict(cp[k]) for k in cp.sections()}


    def model(key):
        """alias -> model path: environment FORK_MODELS_<ALIAS>, then models.ini, then config.toml / defaults; anything else (a path) comes back unchanged"""
        env = os.environ.get(f"FORK_MODELS_{key}".upper())
        if env: return os.path.expanduser(env)
        ini = models_ini().get(key)
        if ini and ini.get("path"): return os.path.expanduser(ini["path"])
        return os.path.expanduser(get("models", key)) if key in _FILE.get("models", {}) or key in DEFAULTS["models"] else key

    def prompts_ini():
        """fork/prompts.ini: [name] sections; path = full path of the input file (default dir = fork/bench, FORK_PROMPTS_DIR moves it), strip / lines / in / alias / tokens / prefix / snippet / test, see the file"""
        import configparser
        here = os.path.dirname(os.path.abspath(__file__))
        cp = configparser.ConfigParser(defaults={"here": os.environ.get("FORK_PROMPTS_DIR") or os.path.join(here, "prompts")})
        cp.read([os.path.join(here, "prompts.ini"), os.path.join(here, "prompts.local.ini")])   # prompts.local.ini: your own prompts and tests, not in git
        out = {k: dict(cp[k]) for k in cp.sections()}
        for n in range(1, 1025):   # pf<N>k1k / pf<N>k: the first N thousand tokens of the 12k / 128k source (cut in run.py; beyond 128k the source repeats), answer 1024 tokens; any size up to 1m (the longest trained context, MiMo), nothing stored per size
            out.setdefault(f"pf{n}k1k", dict(alias=f"pf{n}k", tokens="1024", prefix="code_review", **{"in": f"{n}k"})); out.setdefault(f"test-pf{n}k", {"in": f"{n}k"})
        # a file you drop into the prompts directory is a test of its own, named after the file: prompt-NAME.txt (not claimed by a section above) -> test NAME, the whole file one chat message,
        # 1024 tokens out (tokens = / ctx = need a section). No ini edit: copy the file, `run run ... ` with a cell whose test is NAME.
        pdir = cp.defaults()["here"]
        claimed = {os.path.realpath(v["path"]) for v in out.values() if v.get("path")}
        try: names = sorted(os.listdir(pdir))
        except OSError: names = []
        for fn in names:
            m = re.fullmatch(r"prompt-(.+)\.txt", fn)
            fp = os.path.realpath(os.path.join(pdir, fn))
            if m and fp not in claimed and m.group(1) not in out:
                kb = max(1, os.path.getsize(fp) // 4096)
                out[m.group(1)] = {"path": fp, "strip": "1", "test": "chat", "tokens": "1024", "in": f"~{kb}k"}
        return out


    def prompt_path(name):
        p = prompts_ini().get(name, {}).get("path")
        if not p: raise KeyError(f"prompts.ini: no [{name}] with a path")
        return os.path.expanduser(p)


    def prompt(name):
        """the text of prompt [name]; strip = 1 drops the file's trailing newline"""
        t = open(prompt_path(name), errors="replace").read()
        return t[:-1] if t.endswith("\n") and prompts_ini()[name].get("strip") == "1" else t


    def prompt_lines(name):
        """one prompt per line (lines = 1), blank lines skipped"""
        return [x for x in open(prompt_path(name), errors="replace").read().split("\n") if x.strip()]


    def test_in():
        """{test: prompt size text for the status line}: [name] in = ..., [test-name] in = ..."""
        return {k[5:] if k.startswith("test-") else k: v["in"] for k, v in prompts_ini().items() if v.get("in")}


    def user_tests():
        """tests defined in prompts.ini by `test = chat|completion` (+ path, tokens, ctx): {name: section}; any prompt file becomes a test"""
        return {k: v for k, v in prompts_ini().items() if v.get("test") in ("chat", "completion") and v.get("path")}


    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

config = _register('config', _lib_config())


def _lib_runnorm():
    # ---- the normalisation section of run.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """the normalisation section of run.py: the normalisation shared by run.py runlog-all (full rebuild of all_runs) and the statistics section of run.py (incremental refresh): model family, build number, test group."""
    import os, re

    FAMS = [("GLM-5.3-Flash-GSQ-RCO-3.5bit", "GLM 3.5-bit"), ("GLM-5.3-Flash-GSQ-RCO-3.0bit", "GLM 3.0-bit"), ("MiMo", "MiMo IQ3_XXS"), ("Qwen3.8-Flash-Next-UD-IQ4_XS", "Next IQ4_XS"),
            ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "Next IQ3_S"), ("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", "Next IQ1_M"), ("Qwen3.6-35B-A3B-GSQ", "Qwen3.6 Q2_0"), ("qwen36", "Qwen3.6 Q2_0"),
            ("Qwen3.8-27B-IQ4_NL", "27B IQ4_NL"), ("Qwen3.8-27B-GSQ-RCO-IQ3_S", "27B IQ3_S"), ("Qwen3.8-27B-MTP", "27B Q5_K_M MTP"), ("Qwen3.8-27B", "27B"), ("OLMoE", "OLMoE"),
            ("DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS", "DS4.1 IQ2_XXS")]


    def fam(model, args=""):
        m = (model or "").strip()
        if not m:
            m = (re.search(r"(?:-m|--model)\s+(\S+)", args or "") or [None, ""])[1]
        m = os.path.basename(m.split(" ")[0])
        for k, v in FAMS:
            if k.lower() in m.lower():
                return v
        return re.sub(r"-0000\d-of-0000\d|\.gguf", "", m)[:28] or "?"


    def bno_of(build, bno=""):
        m = re.search(r"b(\d{5})", build or "") or re.search(r"(\d{5})", str(bno or ""))
        return int(m.group(1)) if m else None


    def tgroup(t):
        return re.sub(r"-lv3$", "", re.sub(r"^cli-", "", t or ""))


    GAME_WORDS = config.prompt_lines("game_words")


    def prompt_of(test, rep="", tag=""):
        """the prompt of a run: the logged column, else derived from the test and the repetition (game4 and edit4 rotate through four prompts by repetition)"""
        if tag:
            return tag
        t = tgroup(test)
        try:
            r = int(float(rep))
        except (TypeError, ValueError):
            return t
        if t == "game4":
            return "game4-" + GAME_WORDS[r % 4]
        if t == "edit4":
            return f"edit4-{r % 4}"
        return t

    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

runnorm = _register('runnorm', _lib_runnorm())


def _lib_runstats():
    # ---- the statistics section of run.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """the statistics section of run.py: cell statistics (min, median and max of t/s and pp per model, test, machine, side and build) kept in runs.db table cell_stats.

Records added since the last refresh are marked (runs.stats_ok IS NULL, set by runlog.insert); refresh(con) takes ONLY those: they go into all_runs
(if not there yet), only the cells they belong to are recomputed, then the records are marked done. Called at the start of every run.py runq query."""
    import os, statistics, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from runnorm import fam, bno_of, tgroup, prompt_of


    def recompute(con, cells):
        """cells: list of (fam, tgroup, prompt, hw, side, build) to recompute, None = all"""
        where, args = "", []
        if cells is not None:
            where = " AND (fam, tgroup, prompt, hw, side, build) IN (" + ",".join("(?,?,?,?,?,?)" for _ in cells) + ")"
            args = [x for c in cells for x in c]
        groups = {}
        for r in con.execute(f"SELECT fam, tgroup, COALESCE(prompt, tgroup), hw, side, build, tps, pp, hit, ts FROM all_runs WHERE ok = 1{where}", args):
            g = groups.setdefault(r[:6], ([], [], [], []))
            g[0].append(r[6])
            if r[7]:
                g[1].append(r[7])
            if r[8] is not None:
                g[2].append(r[8])
            g[3].append(r[9])
        mm = lambda v: (min(v), statistics.median(v), max(v)) if v else (None, None, None)
        for k, (t, pp, hit, ts) in groups.items():
            sd = statistics.pstdev(t) if len(t) > 1 else None   # observed spread of this model, prompt and build: standard deviation and max/min
            con.execute("INSERT OR REPLACE INTO cell_stats (fam, tgroup, prompt, hw, side, build, n, t_min, t_med, t_max, t_sd, spread, pp_min, pp_med, pp_max, hit, first) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        k + (len(t),) + mm(t) + (sd, max(t) / min(t) if len(t) > 1 and min(t) else None) + mm(pp) + (max(hit) if hit else None, min(ts)))
        # build order: by the time of the first run of the build, over all sides
        firsts = sorted(con.execute("SELECT build, MIN(ts) FROM all_runs GROUP BY build ORDER BY MIN(ts)"))
        for i, (b, _) in enumerate(firsts):
            con.execute("UPDATE cell_stats SET bseq = ? WHERE build = ?", (i, b))
            con.execute("UPDATE all_runs SET bseq = ? WHERE build = ?", (i, b))


    def refresh(con):
        """returns the number of new records taken in (0: statistics were current)"""
        have = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
        if "all_runs" not in have or "cell_stats" not in have or "runs" not in have:
            return -1   # no full build yet: python3 fork/run.py runlog-all
        if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
            con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")   # every record counts as new: the dedupe in all_runs keeps the rows already there
        for col in ("prompt", "rep"):   # a db made before these columns existed
            if col not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
                con.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")
        new = con.execute("SELECT ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note, rep, prompt FROM runs WHERE stats_ok IS NULL").fetchall()
        if not new:
            return 0
        cells = set()
        for ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note, rep, prompt in new:
            if not tps:
                continue
            f, tg, pr = fam(model, args), tgroup(test), prompt_of(test, rep, prompt)
            side = "stock" if (build or "").startswith("stock") else "ours"
            if not con.execute("SELECT 1 FROM all_runs WHERE ts = ? AND fam = ? AND test = ? AND build = ? AND ROUND(tps, 2) = ROUND(?, 2)", ((ts or "")[:19], f, test, build, tps)).fetchone():
                con.execute("INSERT INTO all_runs (src, ts, hw, fam, tgroup, test, build, bno, side, tps, pp, hit, ok, args, note, prompt) VALUES ('csv',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            ((ts or "")[:19], hw or "pc1", f, tg, test, build, bno_of(build, bno), side, tps, pp, hit, 0 if str(ok) == "0" else 1, (args or "")[:60], (note or "")[:80], pr))
            cells.add((f, tg, pr, hw or "pc1", side, build))
        recompute(con, sorted(cells))
        con.execute("UPDATE runs SET stats_ok = 1 WHERE stats_ok IS NULL")
        con.commit()
        return len(new)

    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

runstats = _register('runstats', _lib_runstats())


def _lib_runexp():
    # ---- runexp.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py exp SPEC.ini [--dry] [--only ARM,ARM] [--rounds N]: an experiment described in a file, every run stored like any other run.

Everything an experiment needs is a field of the spec, so no side script is ever needed (and nothing is lost when a temp folder goes away).
Every run is one row of run-history.csv (same columns as run.py bench, plus the ones below) and the spec itself is kept next to the data.

[exp]
name = pinned-load-abba          ; campaign name of every row
note = what this tests
rounds = 3
order = rotate                   ; rotate | abba | fixed
warm = 0                         ; discarded runs of each arm before its first measured one

[models]                         ; name = path
glm = /m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf

[workload]
tool = cli                       ; cli | perplexity | completion
prompt_file = fork/bench/prompt650.txt    ; or prompt = text
n = 100
ctx = 2048
chunks = 2                       ; perplexity: chunks, text_file
extra = -lm pin                  ; args of every arm
timeout = 1500

[setup]
cuda_visible_devices = 1
hold_vram = 0:12.4               ; GPU:GiB, space separated; held for the whole experiment
mem_limit = 30G                  ; systemd-run scope per run, swap off
drop_cache = 1                   ; model leaves the page cache before every run
state = off                      ; off | keep | FILE (learned MoE state, reset at start)
env = K=V K2=V2
guard = min_avail_gb=40 max_swap_mb=1500 swapout_pages_s=25000 psi=30
sample = 1                       ; CPU/GPU/memory once a second, summarised in the row

[arm N]                          ; one section per arm
bin = <build dir>/bin
args = --moe-expert-cache 0
env = K=V
origin = fork                    ; fork | stock
tool = completion                ; optional, overrides workload tool

Command line: run.py exp SPEC.ini [--dry] [--only N,P] [--rounds N]
Columns added to run-history.csv by this (empty for rows of the older runs): exp (spec name), arm, ready_s (model ready: the perplexity start), wall_s, rss_kb,
major_faults, uploads, evictions, up_mib (churn), ddr_gbs, pcie_gbs, ddr_util, pcie_util (% of the probed peaks), cpu_util, gpu_util (per GPU, %), placement, rungs
(retries that rescued a start), held_gib, mem_limit, cache_state, state, sys (json: clocks, power, temperature, free memory, swap, kernel, driver, huge pages).
The engine's end-of-run line `moe-summary: ...` (when the binary prints it) fills the churn and bandwidth columns.
"""
    import glob, shutil, hashlib, json, os, re, shlex, signal, statistics, subprocess, sys, threading, time

    EXP_COLS = ["exp", "arm", "ready_s", "wall_s", "rss_kb", "major_faults", "uploads", "evictions", "up_mib", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late", "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3", "miss_per_layer", "ddr_gbs", "pcie_gbs", "ddr_util", "pcie_util",
                "cpu_util", "gpu_util", "placement", "decisions", "rungs", "held_gib", "mem_limit", "cache_state", "temp", "state", "sys"]
    TOOLS = {"cli": "llama-cli", "perplexity": "llama-perplexity", "completion": "llama-completion"}


    def _read(path, default=""):
        try:
            return open(path).read()
        except OSError:
            return default


    def meminfo():
        m = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+)", _read("/proc/meminfo"), re.M)}
        return m


    def machine_state():
        m = meminfo()
        st = {"mem_avail_gb": round(m.get("MemAvailable", 0) / 1048576, 1), "swap_mb": round((m.get("SwapTotal", 0) - m.get("SwapFree", 0)) / 1024),
              "kernel": os.uname().release, "huge_1g_free": _read("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages").strip(),
              "huge_1g_total": _read("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages").strip()}
        try:
            st["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.split()[0]
        except Exception:
            pass
        return st


    def cpu_times():
        f = _read("/proc/stat").split("\n")[0].split()[1:]
        v = [int(x) for x in f[:8]]
        return sum(v) - v[3] - v[4], sum(v)   # busy (not idle, not iowait), total


    class Sampler(threading.Thread):
        """CPU, GPU and memory once a second; summary() gives the means and extremes"""
        def __init__(self):
            super().__init__(daemon=True)
            self.stop_ev = threading.Event()
            self.cpu, self.gpu, self.clk, self.pw, self.tmp, self.avail, self.swap, self.thr = [], {}, {}, {}, {}, [], [], set()

        def run(self):
            b0, t0 = cpu_times()
            while not self.stop_ev.wait(1.0):
                b1, t1 = cpu_times()
                if t1 > t0:
                    self.cpu.append(100.0 * (b1 - b0) / (t1 - t0))
                b0, t0 = b1, t1
                m = meminfo()
                self.avail.append(m.get("MemAvailable", 0) / 1048576)
                self.swap.append((m.get("SwapTotal", 0) - m.get("SwapFree", 0)) / 1024)
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                    for line in out.strip().split("\n"):
                        i, u, c, p, t, r = [x.strip() for x in line.split(",")]
                        self.gpu.setdefault(i, []).append(float(u)); self.clk.setdefault(i, []).append(float(c))
                        self.pw.setdefault(i, []).append(float(p)); self.tmp.setdefault(i, []).append(float(t))
                        if r not in ("0x0000000000000000", "0x0000000000000001"):
                            self.thr.add(r)
                except Exception:
                    pass

        def summary(self):
            mean = lambda v: round(statistics.mean(v), 1) if v else None
            return {"cpu_util": mean(self.cpu), "cpu_max": round(max(self.cpu), 1) if self.cpu else None,
                    "gpu_util": "/".join(str(mean(v)) for _, v in sorted(self.gpu.items())) or None,
                    "gpu_clk_mhz": "/".join(str(round(statistics.mean(v))) for _, v in sorted(self.clk.items())) or None,
                    "gpu_power_w": "/".join(str(mean(v)) for _, v in sorted(self.pw.items())) or None,
                    "gpu_temp_max": "/".join(str(round(max(v))) for _, v in sorted(self.tmp.items())) or None,
                    "gpu_throttle": sorted(self.thr) or None,
                    "mem_avail_min_gb": round(min(self.avail), 1) if self.avail else None, "swap_max_mb": round(max(self.swap)) if self.swap else None}


    class Guard(threading.Thread):
        """kills the run's process group on a swap storm or memory pressure"""
        def __init__(self, proc, limits):
            super().__init__(daemon=True)
            self.proc, self.lim, self.stop_ev, self.reason = proc, limits, threading.Event(), ""

        @staticmethod
        def pswpout():
            m = re.search(r"^pswpout (\d+)", _read("/proc/vmstat"), re.M)
            return int(m.group(1)) if m else 0

        def run(self):
            prev = self.pswpout()
            while not self.stop_ev.wait(1.0):
                cur = self.pswpout()
                psi = re.search(r"some avg10=([\d.]+)", _read("/proc/pressure/memory"))
                if cur - prev > self.lim.get("swapout_pages_s", 25000) or (psi and float(psi.group(1)) > self.lim.get("psi", 30)):
                    self.reason = f"guard: swap-out +{cur - prev} pages/s, memory pressure {psi.group(1) if psi else '?'}"
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    return
                prev = cur


    def model_files(path):
        m = re.match(r"(.*)-0*1-of-(\d+)\.gguf$", path)
        return sorted(glob.glob(m.group(1) + "-*-of-" + m.group(2) + ".gguf")) if m else [path]


    def drop_cache(path):
        for f in model_files(path):
            try:
                fd = os.open(f, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
        time.sleep(2)


    def parse_log(text, time_txt):
        """everything a run's output says: speeds, perplexity, ready time, placement, rescues, out of memory, the engine's summary line"""
        t = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
        r = {}
        m = re.search(r"\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]", t)
        if m:
            r["pp"], r["tps"] = float(m.group(1)), float(m.group(2))
        else:   # llama-completion / perf lines
            mp = re.search(r"prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", t)
            mg = re.search(r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+) runs.*?([\d.]+) tokens per second", t)
            if mp:
                r["pp"] = float(mp.group(2))
            if mg:
                r["tps"], r["n_gen"] = float(mg.group(2)), int(mg.group(1))
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", t)
        if m:
            r["ppl"] = float(m.group(1))
        m = re.search(r"^(\d+)\.(\d\d)\.(\d{3})\.\d{3} I perplexity: tokenizing the input", t, re.M)
        if m:
            r["ready_s"] = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
        m = re.search(r"MoE placement: ([^\n]{0,140})", t)
        if m:
            r["placement"] = m.group(1).strip()
        dec = [re.sub(r"^\S+ [IWE] ", "", l).strip()[:200] for l in t.split("\n") if re.search(
            r"ubatch \d+ \(|compute threads|decode threads|prompt threads|tuned|autotune|MoE placement|retrying|context creation failed|n_ubatch|threads:", l)]
        if dec:
            r["decisions"] = " || ".join(dict.fromkeys(dec))[:1500]
        rungs = sorted(set(re.findall(r"retrying with ubatch \d+|retrying without op offload|starting over with the fork off", t)))
        if rungs:
            r["rungs"] = ";".join(rungs)
        if "out of memory" in t:
            r["oom"] = 1
        m = re.search(r"moe-pred: hit=([\d.]+) hit_without_est=([\d.]+) boost_pts=([\d.]+) up=(\d+) pub=(\d+) used=(\d+) late=(\d+) stale=(\d+) cov1=([\d.]+) cov2=([\d.]+) cov3=([\d.]+) miss_per_layer=([\d.]+)", t)
        if m:
            hw_, hwo, bp, up, pub, used, late, stale, c1, c2, c3, mpl = m.groups()
            r.update(pred_hit=float(hw_), pred_hit_without=float(hwo), pred_boost_pts=float(bp), pred_up=int(up), pred_pub=int(pub), pred_used=int(used), pred_late=int(late), pred_stale=int(stale),
                     pred_cov1=float(c1), pred_cov2=float(c2), pred_cov3=float(c3), miss_per_layer=float(mpl))
        m = re.search(r"moe-summary: hit=([\d.]+)% uploads=(\d+) evictions=(\d+) up_mib=(\d+) ddr_gbs=([\d.]+) pcie_gbs=([\d.]+) ddr_peak=([\d.]+) pcie_peak=([\d.]+)", t)
        if m:
            h, up, ev, mib, dd, pc, dpk, ppk = [float(x) for x in m.groups()]
            r.update(hit=h, uploads=int(up), evictions=int(ev), up_mib=int(mib), ddr_gbs=dd, pcie_gbs=pc,
                     ddr_util=round(100 * dd / dpk, 1) if dpk else None, pcie_util=round(100 * pc / ppk, 1) if ppk else None)
        if time_txt:
            m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_txt)
            if m:
                r["rss_kb"] = int(m.group(1))
            m = re.search(r"Major \(requiring I/O\) page faults: (\d+)", time_txt)
            if m:
                r["major_faults"] = int(m.group(1))
        return r


    def order_arms(arms, rnd, how):
        if how == "abba":
            return arms if rnd % 2 == 0 else arms[::-1]
        if how == "rotate":
            k = rnd % len(arms)
            return arms[k:] + arms[:k]
        return arms


    def version_of(bindir):
        for tool in ("llama-cli", "llama-completion", "llama-perplexity"):
            exe = os.path.join(bindir, tool)
            if os.path.exists(exe):
                try:
                    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).stderr
                    mc, mb = re.search(r"commit (\w+)", out), re.search(r"build (\d+)", out)
                    return (mc.group(1) if mc else None), (int(mb.group(1)) if mb else None)
                except Exception:
                    return None, None
        return None, None


    def build_meta(bindir):
        """commit, branch, patch of a build: bin/BUILD_INFO (or its parent), else the binary's --version"""
        for d in (bindir, os.path.dirname(bindir.rstrip("/"))):
            try:
                bi = dict(l.strip().split("=", 1) for l in open(os.path.join(d, "BUILD_INFO")) if "=" in l)
                return bi.get("commit"), bi.get("branch", ""), bi.get("patch", ""), None
            except OSError:
                pass
        c, b = version_of(bindir)
        return c, "", "", b


    def _kv(v):
        return dict(x.split("=", 1) for x in shlex.split(v) if "=" in x)


    def load_spec(path):
        """INI spec (see the top of this file) -> the dict the runner uses"""
        import configparser
        c = configparser.ConfigParser(inline_comment_prefixes=(";",), interpolation=None)
        c.optionxform = str
        c.read(path)
        e, w, s = dict(c["exp"]), dict(c["workload"]) if "workload" in c else {}, dict(c["setup"]) if "setup" in c else {}
        num = lambda v: float(v) if "." in v else int(v)
        for k in ("n", "ctx", "chunks", "timeout", "log_verbosity"):
            if k in w:
                w[k] = int(w[k])
        w["extra"] = shlex.split(w.get("extra", ""))
        st = s.get("state", "keep")
        setup = {"state": st if st in ("off", "keep") else {"file": st, "reset": "start"}, "env": _kv(s.get("env", "")),
                 "guard": {k: num(v) for k, v in _kv(s.get("guard", "")).items()}, "drop_cache": s.get("drop_cache", "0") == "1", "save_logs": s.get("save_logs", "1") == "1",
                 "sample": s.get("sample", "1") == "1"}
        if s.get("cuda_visible_devices"):
            setup["cuda_visible_devices"] = s["cuda_visible_devices"]
        if s.get("mem_limit"):
            setup["mem_limit"] = s["mem_limit"]
        if s.get("hold_vram"):
            setup["hold_vram"] = dict(x.split(":") for x in s["hold_vram"].split())
        arms = []
        for sec in c.sections():
            if sec.startswith("arm "):
                a = dict(c[sec])
                arms.append({**a, "name": sec[4:].strip(), "args": shlex.split(a.get("args", "")), "env": _kv(a.get("env", ""))})
        return {"_path": path, "name": e["name"], "note": e.get("note", ""), "rounds": int(e.get("rounds", 1)), "order": e.get("order", "fixed"), "warm": int(e.get("warm", 0)),
                "models": dict(c["models"]), "workload": w, "setup": setup, "arms": arms}


    def run_exp(spec_path, rs, dry=False, only=None, rounds=None):
        """rs: the run.py module (csv_append, RUN_COLS, power_limits, DATA, hw name)"""
        spec = load_spec(spec_path)
        name = spec["name"]
        wl, setup = spec.get("workload", {}), spec.get("setup", {})
        arms = [a for a in spec["arms"] if not only or a["name"] in only]
        models = spec["models"]
        models = models if isinstance(models, dict) else {os.path.basename(p): p for p in models}
        rounds = rounds or spec.get("rounds", 1)
        guard_lim = setup.get("guard", {})
        hw = os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
        spec_sha = hashlib.sha1(open(spec_path, "rb").read()).hexdigest()[:8]
        base_env = dict(os.environ)
        if setup.get("cuda_visible_devices") is not None:
            base_env["CUDA_VISIBLE_DEVICES"] = str(setup["cuda_visible_devices"])
        base_env.update({k: str(v) for k, v in setup.get("env", {}).items()})
        st = setup.get("state", "keep")
        state_file = None
        if st == "off":
            base_env["LLAMA_MOE_STATE"] = "0"
        elif isinstance(st, dict):
            state_file = st["file"]
            base_env["LLAMA_MOE_STATE"] = state_file
            if st.get("reset", "start") == "start" and os.path.exists(state_file) and not dry:
                os.remove(state_file)
        for a in arms:
            if a.get("state") and os.path.exists(a["state"]) and not dry and a.get("keep_state") != "1":   # keep_state = 1: a second phase continues the arm's learned state
                os.remove(a["state"])
        if setup.get("hold_vram"):
            sys.exit("hold_vram is gone: the server holds memory itself; put --moe sim-vram=MB (MiB left free) in the arm args")
        n_run = 0
        try:
            for mname, mpath in models.items():
                warm_done = set()
                for rnd in range(rounds):
                    for arm in order_arms(arms, rnd, spec.get("order", "fixed")):
                        # a discarded hot-up run (cold file read, never stored) only for arms that read the weights lazily (mmap, stock, fork off):
                        # warm = N on the arm, else on [exp]; a pinned arm reads the whole file while loading and needs none
                        warm_n = int(arm.get("warm", spec.get("warm", 0))) if arm["name"] not in warm_done else 0
                        for attempt in range(1 + warm_n):
                            n_run += 1
                            row = one_run(spec, wl, setup, arm, mname, mpath, rnd, base_env, guard_lim, dry, state_file)
                            if attempt < warm_n:   # the cold run is stored too, marked temp=cold; reports leave temp=cold rows out
                                row["temp"] = "cold"
                                print(f"  ({arm['name']}: cold first run recorded as temp=cold)", flush=True)
                                if not dry:
                                    store_row(rs, spec, name, spec_sha, arm, mname, mpath, rnd, row, hw, setup, guard_lim)
                                continue
                            warm_done.add(arm["name"])
                            if not dry:
                                store_row(rs, spec, name, spec_sha, arm, mname, mpath, rnd, row, hw, setup, guard_lim)
        finally:
            pass
        print(f"{name}: {n_run} runs", flush=True)


    def command_for(wl, setup, arm, mpath):
        tool = arm.get("tool", wl.get("tool", "cli"))
        exe = os.path.join(arm["bin"], TOOLS[tool])
        a = ["-m", mpath]
        here = os.path.dirname(os.path.realpath(__file__))
        resolve = lambda p: p if os.path.isabs(p) else os.path.join(os.path.dirname(here), p)
        if tool == "perplexity":
            a += ["-f", resolve(wl.get("text_file", config.prompt_path("longsrc"))), "-c", str(wl.get("ctx", 512)), "--chunks", str(wl.get("chunks", 2))]
        else:
            if wl.get("prompt_file"):
                a += ["-f", resolve(wl["prompt_file"])]
            elif wl.get("prompt"):
                a += ["-p", wl["prompt"]]
            a += ["-n", str(wl.get("n", 100)), "--temp", "0"]
            if tool == "cli":
                a += ["-st"]
            if wl.get("ctx"):
                a += ["-c", str(wl["ctx"])]
        if wl.get("log_verbosity") and tool in ("cli", "completion"):   # engine decisions (ubatch, threads, placement) are info messages
            a += ["--log-verbosity", str(wl["log_verbosity"])]
        a += [str(x) for x in wl.get("extra", [])] + [str(x) for x in arm.get("args", [])]
        if arm.get("state") and arm.get("origin", "fork") != "stock":   # the fork reads --moe state=FILE (the LLAMA_MOE_STATE variable is gone)
            a += ["--moe", "state=" + arm["state"] + (".fresh" if arm.get("fresh_state") else "")]
        cmd = [exe] + a
        if setup.get("time", True):
            if os.path.exists("/usr/bin/time"):   # not on every box (pc2): then no rss or page-fault columns
                cmd = ["/usr/bin/time", "-v"] + cmd
        cmd = ["timeout", str(wl.get("timeout", 1500))] + cmd
        env = {k: str(v) for k, v in arm.get("env", {}).items()}
        if env:
            cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + cmd
        if setup.get("mem_limit"):
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={setup['mem_limit']}", "-p", "MemorySwapMax=0"] + cmd
        return cmd


    def one_run(spec, wl, setup, arm, mname, mpath, rnd, base_env, lim, dry, state_file):
        if arm.get("fresh_state") and arm.get("state") and not dry and os.path.exists(arm["state"] + ".fresh"):
            os.remove(arm["state"] + ".fresh")   # fresh_state: every start begins with an empty state file of the arm's own (never the user's moe-state.ini)
        cmd = command_for(wl, setup, arm, mpath)
        label = f"{spec['name']} {mname} r{rnd + 1} {arm['name']}"
        if dry:
            print(f"[dry] {label}: {shlex.join(cmd)}", flush=True)
            return {}
        ms = machine_state()
        if ms["mem_avail_gb"] < lim.get("min_avail_gb", 0) or ms["swap_mb"] > lim.get("max_swap_mb", 10 ** 9):
            print(f"{label}: SKIPPED (available {ms['mem_avail_gb']} GiB, swap {ms['swap_mb']} MiB)", flush=True)
            return {"skipped": 1, "sys": ms}
        cache_state = "cold" if setup.get("drop_cache") else "as-is"
        temp = "cold" if setup.get("drop_cache") else "hot"
        if setup.get("drop_cache"):
            drop_cache(mpath)
        elif str(arm.get("free_cache", "0")) == "1":   # pinned arms: the file's page cache is released first (RAM for the pinned copy), the run is not marked cold
            drop_cache(mpath)
        env = dict(base_env)
        t0 = time.time()
        sampler = Sampler() if setup.get("sample", True) else None
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
        guard = Guard(p, lim)
        if sampler:
            sampler.start()
        guard.start()
        out, err = p.communicate()
        wall = time.time() - t0
        guard.stop_ev.set()
        if sampler:
            sampler.stop_ev.set()
        # /usr/bin/time -v writes to stderr after the program's own output
        tm = re.search(r"Command being timed:.*", err)
        time_txt = err[tm.start():] if tm else ""
        r = parse_log(out + "\n" + err, time_txt)
        if setup.get("save_logs", True):   # the whole output of every run, next to the data (a row is a summary, the log is the evidence)
            ld = os.path.join(os.environ.get("RUN_DATA") or os.path.join(os.path.dirname(os.path.realpath(__file__)), "bench"), "logs", spec["name"])
            os.makedirs(ld, exist_ok=True)
            with open(os.path.join(ld, f"{mname}-r{rnd + 1}-{arm['name']}.log"), "w") as lf:
                lf.write(out + "\n=== stderr ===\n" + err)
        r.update(wall_s=round(wall, 1), cache_state=cache_state, temp=temp, sys={**ms, **(sampler.summary() if sampler else {})})
        if guard.reason:
            r["note"] = guard.reason
        if p.returncode not in (0, None) and not r.get("tps") and not r.get("ppl"):
            r.setdefault("note", f"exit {p.returncode}" + (", out of memory" if r.get("oom") else ""))
        print(f"{label}: " + (f"{r['tps']:.2f} t/s" if r.get("tps") else "no t/s") + (f", pp {r['pp']:.1f}" if r.get("pp") else "") + (f", PPL {r['ppl']}" if r.get("ppl") else "")
              + (f", ready {r['ready_s']:.1f}s" if r.get("ready_s") else "") + f", wall {wall:.0f}s" + (f" [{r['note']}]" if r.get("note") else "")
              + (f" [{r['rungs']}]" if r.get("rungs") else ""), flush=True)
        return r


    def store_row(rs, spec, name, spec_sha, arm, mname, mpath, rnd, r, hw, setup, lim):
        if r.get("skipped"):
            return
        commit, branch, patch, bno = build_meta(arm["bin"])
        env = {**{k: str(v) for k, v in setup.get("env", {}).items()}, **{k: str(v) for k, v in arm.get("env", {}).items()}}
        if setup.get("state") == "off":
            env["LLAMA_MOE_STATE"] = "0"
        held = "+".join(f"{g}:{v}" for g, v in setup.get("hold_vram", {}).items())
        sysd = r.get("sys") or {}
        row = dict(ts=time.strftime("%F %T"), commit=(commit or "") + ("+" + patch if patch and patch != "0" else ""), bno=bno, tps=r.get("tps"), pp=r.get("pp"), hit=r.get("hit"),
                   model=os.path.basename(mpath), args=" ".join(map(str, arm.get("args", []))), test=spec.get("workload", {}).get("tool", "cli") + ":" + name, hw=hw,
                   ok=1 if (r.get("tps") or r.get("ppl")) else 0, build=arm.get("build") or arm["name"], campaign=name, note=" | ".join(x for x in (spec.get("note", ""), r.get("note", "")) if x),
                   ppl=r.get("ppl"), n_gen=r.get("n_gen") or spec.get("workload", {}).get("n"), origin=arm.get("origin", "fork"),
                   env=json.dumps(env, sort_keys=True, separators=(",", ":")) if env else "", cmd=shlex.join(command_for(spec.get("workload", {}), setup, arm, mpath)),
                   pl=rs.power_limits(hw), proto="v2" if spec.get("rounds", 1) >= 3 else "v1", rep=rnd, branch=branch, patch=patch,
                   exp=f"{name}@{spec_sha}", arm=arm["name"], ready_s=r.get("ready_s"), wall_s=r.get("wall_s"), rss_kb=r.get("rss_kb"), major_faults=r.get("major_faults"),
                   uploads=r.get("uploads"), evictions=r.get("evictions"), up_mib=r.get("up_mib"), ddr_gbs=r.get("ddr_gbs"), pcie_gbs=r.get("pcie_gbs"),
                   ddr_util=r.get("ddr_util"), pcie_util=r.get("pcie_util"), cpu_util=sysd.get("cpu_util"), gpu_util=sysd.get("gpu_util"), placement=r.get("placement"), decisions=r.get("decisions"),
                   rungs=r.get("rungs"), held_gib=held, mem_limit=setup.get("mem_limit"), cache_state=r.get("cache_state"), temp=r.get("temp"),
                   state=json.dumps(setup.get("state")) if setup.get("state") not in (None, "keep") else "", sys=json.dumps(sysd, sort_keys=True, separators=(",", ":")))
        row.update({c: r.get(c) for c in ["pred_hit"] + EXP_COLS if c in r and c not in row})
        rs.csv_append("run-history.csv", rs.RUN_COLS, row)
        # the spec is kept next to the data, once
        sd = os.path.join(rs.DATA, "specs")
        os.makedirs(sd, exist_ok=True)
        dst = os.path.join(sd, f"{name}@{spec_sha}.ini")
        if not os.path.exists(dst):
            shutil.copy(spec["_path"], dst)

    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

runexp = _register('runexp', _lib_runexp())


def _lib_runlog():
    # ---- run.py runlog (merged into run.py; the code of the file, unchanged)
    __doc__ = """SQLite view of the run log: what model, prompt test and build gave what speed, and when.
Every run is stored twice: as a row of fork/bench/run-history.csv (append-only, the source of truth, in git) and, written by run.py at the same moment, in fork/runs.db (this database, queryable, not in git).
`load` rebuilds the database from the CSV any time.

  python3 fork/run.py runlog load                     rebuild the database from the CSV
  python3 fork/run.py runlog sync                     both ways: CSV rows the database lacks go in, database rows the CSV lacks are appended (bad_runs stay out) (after a git merge of run-history.csv or runs from another worktree)
  python3 fork/run.py runlog bestpl MODEL TEST PL     like best, only the rows measured at one power limit, e.g. 280+280
  python3 fork/run.py runlog best MODEL [TEST]        per build: runs, consistent best t/s and pp, first and last date (MODEL is a substring of the file name)
  python3 fork/run.py runlog runs MODEL [TEST] [BUILD] every run, newest first
  python3 fork/run.py runlog import-old [apply]       runs of the old perf.db (archive) that the CSV lacks, appended to the CSV
  python3 fork/run.py runlog q "SELECT ..."           any SQL; tables: runs, tune_log, campaign_notes, view cells

consistent best = the highest value that at least two runs agree on within 5% (a lone outlier does not count), else the median."""
    import csv, os, re, sqlite3, statistics, subprocess, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(HERE, "bench", "run-history.csv")
    DB = os.path.join(HERE, "runs.db")
    class Consistent:
        def __init__(self): self.v = []
        def step(self, x):
            if x is not None: self.v.append(x)
        def finalize(self):
            xs = sorted(self.v, reverse=True)
            if not xs: return None
            if len(xs) < 2: return xs[0]
            for x in xs:
                if sum(1 for y in xs if y >= 0.95 * x) >= 2: return x
            return statistics.median(xs)
    def connect():
        con = sqlite3.connect(":memory:" if os.environ.get("FORK_NO_RUNS_DB") else DB)   # run.py quick without --record: nothing of the run reaches the database file
        con.create_aggregate("consistent", 1, Consistent)
        return con
    NUMERIC = {"p_met", "p_unmet", "p_pending", "p_admit_free", "p_admit_evict", "p_d_admit", "p_d_margin", "p_d_budget", "p_d_victim", "p_d_big", "p_d_dup", "p_regret", "p_ev_unpaid", "p_ev_zero", "n_cell", "tps_min", "tps_med", "tps_max", "pp_min", "pp_med", "pp_max", "tps", "pp", "hit", "ppl", "spp", "n_gen", "bno", "rep", "seed", "ready_s", "load_s", "wall_s", "ev_mean_hits_fast", "ev_mean_hits_slow", "ev_zero_pct_fast", "ev_zero_pct_slow", "ev_ge4_pct_fast", "ev_ge4_pct_slow", "life_fast", "life_slow", "ddr_util", "up_overlap_pct", "cpu_slow_pct", "lead_end0", "lead_end1", "up_ms_fast", "up_ms_slow", "up_gbs_fast", "up_gbs_slow", "link_gbs_cuda0", "link_gbs_cuda1", "up_fast", "up_slow", "mib_fast", "mib_slow", "busy_fast_ms", "busy_slow_ms", "up_ms_expert", "up_gbs_eff", "pcie_h2d_gb", "pcie_h2d_peak_gbs", "pcie_h2d_avg_gbs", "pcie_d2h_gb", "vram_mib", "pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_n", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms", "admit_end", "rss_kb", "major_faults", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs",
               "ddr_util", "pcie_util", "cpu_util", "gpu_util", "held_gib", "mem_limit", "pred_hit", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late",
               "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3", "mem_gb", "mem_peak_gbs", "pcie_peak_gbs"}
    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return None
    def q(c): return '"' + c + '"'
    def ensure(con, cols):
        """the runs table has every CSV column (numeric ones REAL), plus stem (file name without split suffix and .gguf) and okf (1 unless the run failed); new columns are added as they appear"""
        cols = list(dict.fromkeys(cols))   # a CSV header may repeat a name (wall_s, ddr_util): one column each
        have = [r[1] for r in con.execute("PRAGMA table_info(runs)")]
        if not have:
            con.execute("CREATE TABLE runs (" + ", ".join(f"{q(c)} {'REAL' if c in NUMERIC else 'TEXT'}" for c in cols) + ", stem TEXT, okf INTEGER)")
            con.execute("CREATE INDEX IF NOT EXISTS i_runs ON runs (stem, test, build)")
        else:
            for c in cols:
                if c not in have: con.execute(f"ALTER TABLE runs ADD COLUMN {q(c)} {'REAL' if c in NUMERIC else 'TEXT'}")
        if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
            con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")   # NULL = a new record: the cell statistics (run.py runlog-all) are stale and rebuilt on the next query (run.py runq); the rebuild sets 1
        con.execute("DROP VIEW IF EXISTS cells")
        con.execute("""CREATE VIEW cells AS SELECT stem, hw, test, build, MAX(bno) AS bno, COUNT(*) AS n, ROUND(consistent(tps), 2) AS tps, ROUND(consistent(pp), 1) AS pp,
                   ROUND(MAX(tps), 2) AS tps_max, MIN(ts) AS first, MAX(ts) AS last FROM runs WHERE okf = 1 AND tps IS NOT NULL GROUP BY stem, hw, test, build""")
    def ins(cols):
        """named-column INSERT: ALTER TABLE appends new columns after stem/okf, so position is not safe"""
        return "INSERT INTO runs (" + ", ".join(q(c) for c in cols) + ", stem, okf) VALUES (" + ",".join("?" * (len(cols) + 2)) + ")"
    def run_build_number(sha):
        try:
            o = subprocess.run(["git", "-C", HERE, "rev-list", "--count", sha], capture_output=True, text=True, timeout=20)
            return int(o.stdout) if o.returncode == 0 else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
    def row_values(d, cols):
        stem = re.sub(r"-0000\d-of-0000\d", "", (str(d.get("model") or "")).split(" ")[0]).removesuffix(".gguf")
        if str(d.get("build") or "").startswith("stock") and not num(d.get("bno")) and (d.get("commit") or re.search(r"-([0-9a-f]{7,})$", str(d.get("build")))):   # a docker/source build of stock prints build 0: the upstream number is the commit count
            d = dict(d, bno=run_build_number(d.get("commit") or re.search(r"-([0-9a-f]{7,})$", str(d["build"])).group(1)))
        vals = [(num(d.get(c)) if c in NUMERIC else (None if d.get(c) is None else str(d.get(c)))) for c in cols]
        return vals + [stem, 0 if str(d.get("ok")) == "0" else 1]
    def insert(d, cols):
        """called by run.py with every run (a dict by column name); never raises into a run"""
        try:
            con = connect(); ensure(con, cols)
            con.execute(ins(cols), row_values(d, cols))
            con.commit(); con.close()
        except Exception as e:
            print("runs.db:", e, file=sys.stderr)
    def insert_output(ts, hw, build, test, model, md5, text, uniq4, chars, code):
        """the generated text of a run (zlib) into table outputs, keyed by ts + build + hw (read: zlib.decompress(text))"""
        if not text: return
        import zlib
        con = connect()
        con.execute("CREATE TABLE IF NOT EXISTS outputs (ts TEXT, hw TEXT, build TEXT, test TEXT, model TEXT, md5 TEXT, uniq4 REAL, chars INTEGER, code INTEGER, text BLOB)")
        con.execute("INSERT INTO outputs VALUES (?,?,?,?,?,?,?,?,?,?)", (ts, hw, build, test, model, md5, uniq4, chars, code, zlib.compress(text.encode(), 9)))
        con.commit(); con.close()
    EMB = os.path.join(HERE, "bench", "embhot.csv")
    def ensure_emb(con):
        con.execute("CREATE TABLE IF NOT EXISTS embhot (ts TEXT, hw TEXT, build TEXT, test TEXT, model TEXT, embhex TEXT, hots TEXT)")
    def insert_emb(since, ts, hw, build, test):
        """called by run.py after every run: the embhot.csv lines the run appended (from byte offset since) go into the table embhot; returns the new offset"""
        try:
            size = os.path.getsize(EMB) if os.path.exists(EMB) else 0
            if size <= since: return size
            with open(EMB, newline="") as f:
                f.seek(since); rows = [l for l in f.read().split("\n") if l and not l.startswith("model,")]
            con = connect(); ensure_emb(con)
            for l in rows:
                m = l.split(",", 2)
                if len(m) == 3: con.execute("INSERT INTO embhot VALUES (?,?,?,?,?,?,?)", (ts, hw, build, test, m[0], m[1], m[2]))
            con.commit(); con.close()
            return size
        except Exception as e:
            print("embhot:", e, file=sys.stderr); return since
    def read_csv_rows():
        """(cols, rows as dicts): a duplicate header name (ddr_util, wall_s) keeps its first column, a later non-empty value fills an empty first one; lines whose
    first field is not a timestamp (merge-conflict markers) are skipped"""
        with open(CSV, newline="") as f:
            rd = csv.reader(f); head = next(rd); cols = []
            for c in head:
                if c not in cols: cols.append(c)
            rows = []
            for row in rd:
                if not row or not row[0].startswith("20"): continue
                d = {}
                for c, v in zip(head, row):
                    if d.get(c) in (None, ""): d[c] = v
                rows.append(d)
        return cols, rows
    def load():
        if os.path.exists(DB): os.remove(DB)
        con = connect()
        cols, rows = read_csv_rows()
        ensure(con, cols)
        for r in rows:
            con.execute(ins(cols), row_values(r, cols))
        n = len(rows)
        ensure_emb(con)
        if os.path.exists(EMB):
            with open(EMB, newline="") as f:
                for l in f.read().split("\n"):
                    m = l.split(",", 2)
                    if len(m) == 3 and m[0] != "model": con.execute("INSERT INTO embhot (model, embhex, hots) VALUES (?,?,?)", tuple(m))
        archive = os.path.join(HERE, "bench", "archive", "perf-2026-09-30.db")
        if os.path.exists(archive): attach_old(con, archive)
        con.commit(); print(f"{n} runs loaded into {DB}")
    def sync():
        """both ways, never deletes: the CSV rows the database lacks (runs that came in by a git merge from another machine) go into the database, and the
    database rows the CSV lacks (runs that run.py of another worktree wrote to its own CSV copy) are appended to the CSV; runs moved to bad_runs
    (run.py bad-runs) come back on neither side, also when a merge brings their CSV rows back"""
        con = connect()
        cols, rows = read_csv_rows()
        ensure(con, cols)
        def key(ts, build, test, tps):
            t = num(tps)
            return (str(ts), str(build or ""), str(test or ""), None if t is None else round(t, 3))
        have = {key(*r) for r in con.execute("SELECT ts, build, test, tps FROM runs")}
        bad = {key(*r) for r in con.execute("SELECT ts, build, test, tps FROM bad_runs")} if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'bad_runs'").fetchone() else set()
        n = 0
        for r in rows:
            k = key(r.get("ts"), r.get("build"), r.get("test"), r.get("tps"))
            if k in have or k in bad: continue
            con.execute(ins(cols), row_values(r, cols)); n += 1
        con.commit(); n and print(f"{n} CSV rows added to {DB} ({len(rows)} in the CSV)")
        inc = {key(r.get("ts"), r.get("build"), r.get("test"), r.get("tps")) for r in rows}
        with open(CSV, newline="") as f: head = next(csv.reader(f))   # the file's own header: a repeated name (wall_s, ddr_util) gets the value in its first column only
        cur = con.execute("SELECT * FROM runs ORDER BY ts"); names = [d[0] for d in cur.description]
        def txt(v): return "" if v is None else str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        out = []
        for v in cur:
            d = dict(zip(names, v))
            if key(d.get("ts"), d.get("build"), d.get("test"), d.get("tps")) in inc: continue
            out.append([txt(d.get(c)) if c not in head[:i] else "" for i, c in enumerate(head)])
        if out:
            with open(CSV, "a", newline="") as f: csv.writer(f, lineterminator="\n").writerows(out)
            print(f"{len(out)} database rows appended to {CSV}")
    def show(cur):
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols)))
        for r in rows:
            print("  ".join(str("" if v is None else v).ljust(w[i]) for i, v in enumerate(r)))
    def import_old(path, apply=False):
        """one-shot: runs of the old perf.db (fork/bench/archive/perf-2026-09-30.db, the database before the CSV) that the CSV does not have yet are appended to the CSV (campaign 'perf-db'); the old tune_log lines and campaign notes go into tables of runs.db"""
        old = sqlite3.connect(path); old.row_factory = sqlite3.Row
        with open(CSV, newline="") as f:
            rd = csv.DictReader(f); cols = rd.fieldnames; have = list(rd)
        def key(ts, b, t, tps):
            try: return ((ts or "")[:16], str(b)[:12], t, round(float(tps), 2))
            except (TypeError, ValueError): return ((ts or "")[:16], str(b)[:12], t, None)
        inc = {key(r["ts"], r["build"], r["test"], r["tps"]) for r in have}
        out = []
        for r in old.execute("SELECT * FROM runs ORDER BY ts"):
            if key(r["ts"], r["build"], r["test"], r["tps"]) in inc: continue
            d = {c: "" for c in cols}
            d.update(ts=r["ts"], commit=r["commit_sha"], bno=r["build_no"], tps=r["tps"], pp=r["pp_tps"], hit=r["hit_rate"], model=r["model"], args=r["args"], test=r["test"], hw=r["hw"],
                     ok=r["ok"], build=r["build"], md5=r["md5"], campaign="perf-db", note=r["note"], ppl=r["ppl"], spp=r["s_per_pass"], n_gen=r["n_gen"], origin=r["origin"], env=r["env"], cmd=r["args"], proto="v1")
            out.append({k: ("" if v in (None, "None") else v) for k, v in d.items()})
        print(len(out), "runs of the old perf.db are not in the CSV yet")
        if apply and out:
            with open(CSV, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writerows(out)
        return out

    def attach_old(con, path):
        old = sqlite3.connect(path)
        con.execute("DROP TABLE IF EXISTS tune_log"); con.execute("CREATE TABLE tune_log (run_ts TEXT, run_build TEXT, tag TEXT, line TEXT)")
        for run_id, tag, line in old.execute("SELECT run_id, tag, line FROM tune_log"):
            r = old.execute("SELECT ts, build FROM runs WHERE id = ?", (run_id,)).fetchone() or (None, None)
            con.execute("INSERT INTO tune_log VALUES (?,?,?,?)", (r[0], r[1], tag, line))
        con.execute("DROP TABLE IF EXISTS campaign_notes"); con.execute("CREATE TABLE campaign_notes (name TEXT, start_ts TEXT, note TEXT, conclusion TEXT, why TEXT)")
        for row in old.execute("SELECT name, start_ts, note, conclusion, why FROM campaigns"):
            con.execute("INSERT INTO campaign_notes VALUES (?,?,?,?,?)", row)


    def _cli():
        a = sys.argv[1:]
        if not a or a[0] == "load":
            load()
        elif a[0] == "sync":
            sync()
        else:
            if not os.path.exists(DB): load()
            con = connect()
            if a[0] == "best":
                q = "SELECT test, build, bno, hw, n, tps, pp, tps_max, substr(first,1,10) AS first, substr(last,1,10) AS last FROM cells WHERE stem LIKE ?" + (" AND test = ?" if len(a) > 2 else "") + " ORDER BY test, last DESC"
                show(con.execute(q, ("%" + a[1] + "%",) + ((a[2],) if len(a) > 2 else ())))
            elif a[0] == "bestpl":   # bestpl MODEL TEST PL: like best, but only the rows measured at one power limit (like 280+280); the limit is the live reading at the time of the row
                q = ("SELECT test, build, bno, hw, COUNT(*) AS n, ROUND(consistent(tps), 2) AS tps, ROUND(consistent(pp), 1) AS pp, ROUND(MAX(tps), 2) AS tps_max, substr(MIN(ts),1,10) AS first, substr(MAX(ts),1,10) AS last "
                 "FROM runs WHERE stem LIKE ? AND test = ? AND pl = ? AND okf = 1 AND tps IS NOT NULL GROUP BY test, build, hw ORDER BY last DESC")
                show(con.execute(q, ("%" + a[1] + "%", a[2], a[3])))
            elif a[0] == "runs":
                q = "SELECT substr(ts,1,16) AS ts, hw, test, build, rep, tps, pp, hit, campaign FROM runs WHERE stem LIKE ? AND okf = 1" + (" AND test = ?" if len(a) > 2 else "") + (" AND build LIKE ?" if len(a) > 3 else "") + " ORDER BY ts DESC"
                show(con.execute(q, ("%" + a[1] + "%",) + tuple(("%" + x + "%" if i == 1 else x) for i, x in enumerate(a[2:]))))
            elif a[0] == "import-old":
                import_old(os.path.join(HERE, "bench", "archive", "perf-2026-09-30.db"), apply=len(a) > 1 and a[1] == "apply")
            elif a[0] == "q":
                show(con.execute(a[1]))
            else:
                print(__doc__)

    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

runlog = _register('runlog', _lib_runlog())



sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import config as fcfg   # machine-specific paths and model files: the config section of run.py (defaults, ~/.config/llama-fork/config.toml, FORK_* variables)
if fcfg.get("gpu", "ld_extra"): os.environ.setdefault("PERF_LD_EXTRA", os.path.expanduser(fcfg.get("gpu", "ld_extra")))   # userspace libraries of a hand-loaded driver (the system ones then fail: "unsupported display driver / cuda driver combination"); every tool and cell inherits it
if fcfg.get("gpu", "nvidia_smi") != "nvidia-smi": os.environ.setdefault("NVIDIA_SMI", os.path.expanduser(fcfg.get("gpu", "nvidia_smi")))
# the places everything lives (config.toml [paths], FORK_PATHS_<KEY>, default <clone>/.fork): defined once here, used by name below
RELS = fcfg.path("rels"); WORK = fcfg.path("work"); BUILD_TMP = fcfg.path("build_tmp"); GH_REL = fcfg.path("gh_rel")

PROMPT_DIR = os.environ.get("FORK_PROMPTS_DIR") or os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts")   # the frozen prompts and requests (src_12k.cpp, src_128k.cpp, longsrc.cpp, prompt-*.txt, req-*.json) live in fork/prompts/
PROMPTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bench")   # (the data directory: run-history.csv, stock-fitt.csv ...; the name is older than the split)
HERE = os.path.abspath(os.environ.get("PERF_BUILDS") or fcfg.path("devbuilds"))  # one subdirectory per build (its bin/ contents)
DATA = os.environ.get("RUN_DATA") or PROMPTS   # directory of run-history.csv, campaigns.csv, machines.csv, tune.csv (default: fork/bench, committed)
MODEL = os.environ.get("MODEL", "")
MODELS_DIR = os.environ.get("PERF_MODELS_DIR", "")  # other models' pages are dropped from RAM before a run
PPL_TEXT = os.path.join(PROMPT_DIR, os.environ["PERF_PPL_FILE"]) if os.environ.get("PERF_PPL_FILE") else config.prompt_path("longsrc")  # frozen snapshot of llama.cpp common/json-schema-to-grammar.cpp (code, like agent prompts)
BARE = False
CARD_LIM = [0]   # the card limit the engine reported last (the %limit column)
WRAP = []   # --wrap "CMD ARGS": the server is started as CMD ARGS llama-server ... (a profiler, e.g. nsys profile -o FILE --force-overwrite=true)
THREADS = os.environ.get("PERF_THREADS", "")                   # decode threads: unset = no -t, so the thread autotune (decode and batch) runs; set only to measure a fixed count (fork/cloud/pod.sh sets it for rented boxes)
DEVS = os.environ.get("PERF_DEV", "CUDA0,CUDA1")                # -dev list (the GPU order is pinned)
COMMON_ALL = (["-t", THREADS] if THREADS else []) + ["--cpu-moe", "-nr", "--moe-expert-cache", "-1", "-dev", DEVS]
if "," not in os.environ.get("CUDA_VISIBLE_DEVICES", ","):  # one visible GPU: no device order to pin
    COMMON_ALL = COMMON_ALL[:-2]
COMMON = COMMON_ALL
PORT = int(os.environ.get("PERF_PORT", 8099))
NVSMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")   # a driver that does not match the system libraries (the P2P test module): the nvidia-smi of its own userspace
def local_hw():
    return os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)


def ldpath(build):   # the build directory first; PERF_LD_EXTRA=dir:dir appended (the userspace of a test driver whose version differs from the system libraries)
    return os.path.join(HERE, build) + (":" + os.environ["PERF_LD_EXTRA"] if os.environ.get("PERF_LD_EXTRA") else "")
LOGTAG = "" if PORT == 8099 else f"-p{PORT}"   # a second run.py instance (PERF_PORT) keeps its own server logs
SYNC = os.environ.get("PERF_SYNC", "")   # DIR:N start barrier: N instances fire their timed request together
BAR_N = [0]
OVR = {}   # --prompt TEXT|@file, --tokens N, --ctx N override the test's own prompt, generated tokens and context
CAMPAIGN = ""  # --campaign NAME: column campaign of every row of this invocation
REP = None  # bench repetition (0-based) of the run being stored; None outside bench
WARMING = False  # the discarded warm-up run is on
# test edit4: one long source snippet, a different edit instruction per repetition (what a coding session looks like: a repeated identical temp-0 prompt reuses exactly
# the cached experts and flatters the cache); every build gets the same instruction at the same repetition, the warm-up its own
LONG2K_INSTR = config.prompt("bug_instr")
EDIT_INSTR = config.prompt_lines("edit_instr")   # the fifth is run only with --rep-from 4 (README runs pick one of the first four at random: repetition 0 to 3)
EDIT_WARM = config.prompt("edit_warm")
# test game4: the short CLI test with four subjects, "write smallest html <game> game": same task, same language and stack, only the game changes (a coder stays
# on one goal and one stack, so the experts stay relevant, but no two runs are the same prompt); same game at the same repetition for every build, own warm-up game
GAME_WORDS = config.prompt_lines("game_words")
GAME_WARM = config.prompt("game_warm")
GAME_TEMPLATE = config.prompt("game_template")   # "write smallest html {word} game"
CHAT_LONG = config.prompt("chat_long")
KEEP_SERVER = False   # --keep-server (a run list does this unless -x): the server outlives the run and the next cell with the same model, build and args uses it (no reload); later runs find the cache the earlier ones filled
QDIR = os.environ.get("RUN_QDIR") or os.path.join(WORK, "q5h"); os.makedirs(QDIR, exist_ok=True)   # server logs, queue, kept-server and load files of the machine
KEPT = QDIR + "/kept-server.json"   # the kept server: key (model, build, args), pid, log file
DETACH = [False]      # server_run sets it while it starts a server to keep: no die-with-parent, own session


class PidProc:
    """Handle on a server an earlier run.py process started (the kept server): the parts of Popen that stop() and the samplers use."""
    def __init__(self, pid): self.pid = pid
    def poll(self):
        try: return None if open(f"/proc/{self.pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z" else 0
        except OSError: return 0
    def _sig(self, sig):
        try: os.kill(self.pid, sig)
        except OSError: pass
    def terminate(self): self._sig(signal.SIGTERM)
    def kill(self): self._sig(signal.SIGKILL)
    def wait(self, timeout=None):
        t0 = time.time()
        while self.poll() is None:
            if timeout is not None and time.time() - t0 > timeout: raise subprocess.TimeoutExpired("kept server", timeout)
            time.sleep(0.5)


def kept_get():
    """The kept server's record while it lives, else None."""
    try: d = json.load(open(KEPT))
    except (OSError, ValueError): return None
    return d if PidProc(d["pid"]).poll() is None else None


def kept_kill():
    """Stops the kept server (it holds the VRAM and the pool) and forgets it."""
    d = kept_get()
    if d: kill_tree(d["pid"])   # SIGTERM 8 s, then SIGKILL: the old stop() waited for a server stuck in a batch
    try: os.remove(KEPT)
    except OSError: pass


TAGBOX = {"v": ""}   # the running test and prompt (game4-racing, edit4-2, long1k, ...): --moe tag= puts it into the routing file name
def _ctx_args(default):
    """-c for a test: --ctx N, else the test's own; --ctx -1 passes no -c at all (the engine decides: kv-tail makes it the model's maximum)"""
    c = OVR.get("ctx")
    return [] if c is not None and int(c) < 0 else ["-c", str(c or default)]

IDX = 0    # run counter: test chatv answers prompt IDX mod len(CHAT_PROMPTS) (a repeated temp-0 answer would reuse exactly the cached experts)
CHAT_PROMPTS = config.prompt_lines("chatv")


PROMPT_TWEAK = 0   # set by the loop retry below: the repeated run gets this many newlines appended to its prompt. A greedy run of the same prompt loops again; a changed prompt is cheaper than an anti-repetition sampler (which changes the speed and the cache hit rate being measured)
def ovr_prompt(default):
    p = OVR.get("prompt")
    if p and p.startswith("@"):
        p = open(p[1:]).read()
    return (p or default) + "\n" * PROMPT_TWEAK
GREEDY = {"temperature": 0, "top_p": 1, "top_k": 1}
ANTILOOP = {"repeat_penalty": 1.15, "repeat_last_n": 256}   # --antiloop: the same penalty for every build of a cell; it changes the text, so a loop-prone base model (OLMoE) produces a real answer. Speed and cache hit rate are not touched by it (a sampler step on the logits)
if sys.argv[1:2] == ["run"] and "-al" in sys.argv:   # run ... -al: the same switch on demand for a list run (a cell option of its own, so no other subcommand's -al is touched)
    sys.argv.remove("-al"); sys.argv.insert(1, "--antiloop")
if "--antiloop" in sys.argv:   # a global switch for every subcommand (run, cell, bench, remote, rcell); the env var carries it to the cells started as child processes
    sys.argv.remove("--antiloop"); os.environ["RUN_ANTILOOP"] = "1"
if os.environ.get("RUN_ANTILOOP"): GREEDY.update(ANTILOOP)
SEED = 424242
FIXED = {"seed": SEED, "ignore_eos": True}  # protocol v2 (tests fix, fix12k): same prompt, seed and exact token count for every build


# ---- storage: append-only csv files, key info first so a line reads at a glance ($RUN_DATA, default: the builds folder) ----
RUN_COLS = ["ts", "commit", "bno", "tps", "pp", "hit", "model", "args", "test", "hw", "ok", "build", "md5", "campaign", "note",
            "ppl", "spp", "n_gen", "origin", "env", "cmd",
            "pl", "proto", "rep", "seed", "branch", "patch"]  # pl: GPU power limits in W; proto: v2 = fixed workload (tests fix, fix12k), v1 = earlier; rep: repetition in a bench;
                                          # seed: sampling seed of a v2 run; branch, patch: source branch of the build and the hash of its uncommitted diff (bin/BUILD_INFO,
                                          # written by the build script; "commit" then is that commit, +patch when dirty). A run's statslog: section of stats.ini, row of stats.csv (ts + build)
import runexp
RUN_COLS += runexp.EXP_COLS  # columns of `run.py exp` rows, empty for the other runs
RUN_COLS += ["pred_hit", "chipset", "cpu", "mem_gb", "mem", "mem_peak_gbs", "pcie", "pcie_peak_gbs"]  # platform of the machine (machines.csv: chipset, cpu_short, mem_gb, mem_short, mem_gbs, pcie_short, pcie_gbs); *_peak_gbs are measured GB/s peaks of the machine (ddr_gbs, pcie_gbs are what this run used), empty if not measured
PRESSURE_COLS = ["p_met", "p_unmet", "p_pending", "p_admit_free", "p_admit_evict", "p_d_admit", "p_d_margin", "p_d_budget", "p_d_victim", "p_d_big", "p_d_dup", "p_regret", "p_ev_unpaid", "p_ev_zero"]   # cache pressure of the request (server moe_cache.pressure): met/unmet = selected experts in/not in VRAM; of the unmet that reached the swap decision: admitted into a free slot or by eviction, or declined (admit rule, margin, budget, no victim, lifetime regular, already queued); regret = misses of experts evicted within the last cache turnover; ev_unpaid / ev_zero = evicted uploads that never paid back / were never hit
STAT_COLS = ["prompt", "n_cell", "tps_min", "tps_med", "tps_max", "pp_min", "pp_med", "pp_max"]   # prompt: the test prompt of the run; the others: the runs of this cell so far (same model, test, build, settings and campaign), this run included: the last row of a cell holds its final min, median and max
RUN_COLS += STAT_COLS + PRESSURE_COLS
RUN_COLS += ["pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_n", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms", "admit_end"]   # predictor outcomes of the run, from cache.csv (see cache_csv_totals)
RUN_COLS += ["up_ms_fast", "up_ms_slow", "up_gbs_fast", "up_gbs_slow", "link_gbs_cuda0", "link_gbs_cuda1", "up_fast", "up_slow", "mib_fast", "mib_slow", "busy_fast_ms", "busy_slow_ms"]   # per GPU: mean upload time of one expert on the fast / slow link (engine log) and the effective speed MiB per expert / that time; the startup link speed of each CUDA device; admit-pred builds also: uploads, MiB and copy time per link from cache.csv
RUN_COLS += ["repro"]   # the run.py command line that made the row (the outermost process; cells started by it inherit it): paste it to repeat the run
RUN_COLS += ["bestpp"]   # every run, better or worse (regressions show): the run as a models.ini bestpp tuple (total VRAM GiB, prompt tokens, answer tokens, prompt t/s, decode t/s, prompt s, answer s)
RUN_COLS += ["up_ms_expert", "up_gbs_eff"]   # from the engine log: mean upload time of one expert (all three tensors) over the second half of the run, and the effective speed of one upload = MiB per expert / that time
RUN_COLS += ["pcie_h2d_gb", "pcie_h2d_peak_gbs", "pcie_h2d_avg_gbs", "pcie_d2h_gb"]   # nvidia-smi dmon over the server run: GB host->device during the request (all GPUs; the server is already loaded), the peak of any GPU in a second, the mean of the seconds that moved data, GB device->host
RUN_COLS += ["ddr_util", "up_overlap_pct", "cpu_slow_pct", "lead_end0", "lead_end1"]   # from cache.csv (admit-pred builds): mean RAM traffic share of DDR_GBS, mean share of uploads that ran while the CPU read experts, mean slowdown of CPU expert phases with an upload in flight, the first predicted layer ahead per link at the end
RUN_COLS += ["ev_mean_hits_fast", "ev_mean_hits_slow", "ev_zero_pct_fast", "ev_zero_pct_slow", "ev_ge4_pct_fast", "ev_ge4_pct_slow", "life_fast", "life_slow"]   # hits an uploaded expert actually got before its eviction, per link (cache.csv histogram 0/1/2/3/4-5/6-10/11+; mean with bucket midpoints), the share with no hit and with 4 or more, and the mean steps it stayed cached
RUN_COLS += ["t_full_pw"]   # time to full power, for every server start (a bench run restarts the server): seconds from the start of the GPU log (the server start) until each GPU first draws >= 90% of its own peak (1 s average), one value per GPU in index order; the load phase is the time the cards are not at full power
RUN_COLS += ["pw_avg", "pw_max", "pw_inst_max", "energy_wh", "cap_pct"]   # power of the request window, one value per GPU in index order (nvidia-smi, --gpulog): mean W, peak W (1 s average), peak W (instantaneous), energy of all GPUs in Wh, % of samples at the software power cap
RUN_COLS += ["temp_med", "temp_max"]   # GPU temperature (C) of the request window, one value per GPU in index order: median and maximum
RUN_COLS += ["vram_mib"]   # GPU memory in use at the end of the run, all GPUs together (nvidia-smi memory.used)
RUN_COLS += ["endpoint"]
RUN_COLS += ["knobs_end"]   # what the build actually ran with, from the run.knobs line of the end-of-run state snapshot: key settings (slots, workers, upload path, bounce, policy) | every cache knob incl. tuner decisions (defaults differ between builds; * = set by the user)   # completion (raw prompt), chat (template) or cli (llama-cli)
RUN_COLS += ["load_mode"]   # how the weights were loaded, from the server log: pinned (+ upload path), mmap, mmap after a failed pin ("reloading with mmap"); load time (load_s) tells too: pinned GLM 3.0 about 85-120 s, mmap about 25 s
RUN_COLS += ["prompt_s", "gen_s"]   # the cell as the screen shows it: prompt = end of the load to the first decoded token, gen = first token to the end (the live bar's p and g); written with wall_s by store_cell_time
RUN_COLS += ["start_ts", "end_ts", "load_s", "wall_s"]   # start_ts / end_ts: when the process started and ended (local time); load_s: seconds until the model was loaded and ready (server: ready to answer; llama-cli / perplexity: the "load time" it prints); wall_s: seconds the whole run took, start to end
RUN_COLS += ["uniq4", "out_chars", "code", "loop"]   # text class of the generated output: share of unique word 4-grams (a repetition loop is ~0.1, real text ~0.9+), characters, 1 if it contains code; the text itself is in runs.db table outputs
RUN_COLS += ["thr_dec", "thr_batch", "batch", "ubatch", "ctx", "hwline", "gain_t", "gain_pp", "stock_id", "stock_bno"]   # the server's auto choices (threads for decode / batch, batch, ubatch, context) from its log; the machine line of the run; our decode / prompt speed over the newest stock row (best run of the cell against the stock row it was shown with, set when the cell ends) and that stock row's id and build number
RUN_COLS += ["link_probe", "mem_probe_gbs", "cpu_mhz", "cpu_c", "gpu_mhz", "gpu_c", "pcie_rx_gbs"]   # measured maxima of the server's startup probes (per-card prefill link GB/s as the server printed them, CPU memory bandwidth it saturated at) and min/avg/max over the run of the CPU clock (MHz) and temperature (C), the GPU SM clock and temperature (one "min/avg/max" per card, joined by |), and the PCIe host->device rate per card (peak/mean GB/s): what to read to judge afterwards whether the automatic choices were right; local and remote runs fill the same fields
RUN_COLS += ["gpu_pp", "gpu_tg"]   # the same GPU samples split at the end of the prompt (server timings prompt_ms): prefill phase and generation phase separately, per GPU "util / clock / power / share of samples at the software power cap"; the whole-run gpu_log below blends both and hides that a prefill is compute bound while the decode idles
RUN_COLS += ["gpu_log"]   # per-GPU summary of the 5 Hz nvidia-smi samples of the run (share of busy samples, mean and 5th percentile SM clock and mean power over the busy samples (GPU utilization >= 20%), max temp, throttle reason bits, min PCIe width) and the sample file
CAMP_COLS = ["name", "start", "note", "conclusion", "why"]
MACH_COLS = ["id", "ssh", "root", "kind", "cpu", "ram_gb", "ram_type", "gpus", "links", "os", "hostname", "board", "storage",
             "mem_detail", "gpu_detail", "sw", "note"]
TUNE_COLS = ["ts", "commit", "line"]


def _fmt(v):
    if v is None:
        return ""
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def csv_append(name, cols, d):
    if name == "run-history.csv" and "r" not in LOG_LETTERS:
        return   # PERF_LOG without r: the run is shown but not recorded
    path = os.path.join(DATA, name)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        w.writerow(cols)
    w.writerow([_fmt(d.get(k)) for k in cols])
    if os.path.exists(path) and os.path.getsize(path):
        with open(path) as f:
            head = f.readline().rstrip("\n").split(",")
        if head != cols and cols[:len(head)] == head:  # new columns go at the end: only the header line is rewritten, old rows stay as they are
            lines = open(path).read().split("\n", 1)
            open(path, "w").write(",".join(cols) + "\n" + (lines[1] if len(lines) > 1 else ""))
        elif head != cols:  # columns were inserted in the middle: old rows are remapped by name (a row as wide as cols is already in the new order)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))[1:]
            ob = io.StringIO()
            ow = csv.writer(ob, lineterminator="\n")
            ow.writerow(cols)
            for r in rows:
                ow.writerow(r if len(r) == len(cols) else [dict(zip(head, r)).get(k, "") for k in cols])
            open(path, "w").write(ob.getvalue())
    with open(path, "a") as f:
        f.write(buf.getvalue())  # one write per line: concurrent benches (several machines, one host) do not interleave
    if name == "run-history.csv":   # the same run into fork/runs.db (queryable copy; the CSV stays the source of truth)
        try:
            import runlog
            runlog.insert(d, cols)
        except Exception as e:
            print("runs.db:", e, file=sys.stderr)


def csv_load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r[k]) if r.get(k) not in (None, "") else None
    except ValueError:
        return None


def _die_with_parent():
    # the server dies with run.py (killed, Ctrl-C, crash): no orphan holding the port and VRAM
    import ctypes, signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def kill_leftovers():
    """Kill every llama-* process (server, perplexity, cli, ...) by process name, in a loop until
    none is left: TERM, escalating to KILL after 2 s. Name matching (no -f) never hits
    the calling shell; exiting processes (freeing ~100 GB pinned RAM) are waited for."""
    def left():  # zombies (state Z) are already dead: nothing to kill or wait for
        pids = subprocess.run(["pgrep", "^llama-"], capture_output=True, text=True).stdout.split()
        out = []
        for pid in pids:
            try:
                if open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z":
                    out.append(pid)
            except OSError:
                pass
        return out
    t0 = time.time()
    while left():
        subprocess.run(["pkill", "-KILL" if time.time() - t0 > 2 else "-TERM", "^llama-"])
        time.sleep(1)


STOP_WAIT = int(os.environ.get("PERF_STOP_WAIT", "300"))   # seconds a server gets to exit after SIGTERM before it is killed


def stop(p):
    """The engine writes what it learned (placement records, hot experts) when it exits, and tearing down 100+ GiB of pinned weights takes tens of seconds:
    a kill after 2 s lost all of it for the big models (GLM and MiMo never recorded a placement and ran on stock placement for ever)."""
    p.terminate()
    try:
        p.wait(timeout=STOP_WAIT)
    except subprocess.TimeoutExpired:
        print(f"server did not exit within {STOP_WAIT} s after SIGTERM: killed, its state is lost", flush=True)
        p.kill()
        p.wait()


POWER_DAY, POWER_NIGHT = (220, 280), (200, 200)   # W per GPU (GPU0 = x4 card, GPU1 = x16): 07:00-23:50 and 23:50-07:00 (user 2026-10-18; the x4 card's small fan is loud at night)


def power_schedule(now=None):
    """the power limits for the time of day: day 07:00-23:50 (x4 220 W, x16 280 W), night 23:50-07:00 (200 W both)."""
    if "/" in os.environ.get("PERF_POWER", ""):   # run run -pl 220/280: a forced pair for the whole run, day or night
        x4, x16 = (int(v) for v in os.environ["PERF_POWER"].split("/"))
        return (x4, x16)
    t = time.localtime(now)
    m = t.tm_hour * 60 + t.tm_min
    return POWER_DAY if 7 * 60 <= m < 23 * 60 + 50 else POWER_NIGHT


def _power_targets_raw(widths, spec=None, maxes=None):
    """W per GPU for any number of cards. spec (-pl): `285` every card, `220,280,...` one per card (the last repeats), `220/280` the cards on a narrower link than the widest get the first value, the others the second
    (PC1: the x4 card 220, the x16 card 280). No spec: the day/night schedule, mapped the same way."""
    spec = spec or os.environ.get("PERF_POWER", "")
    if spec == "max" and maxes: return list(maxes)   # `-pl` alone: every card at its own maximum
    if "," in spec:
        v = [int(x) for x in spec.split(",")]
        return [v[min(i, len(v) - 1)] for i in range(len(widths))]
    if spec and "/" not in spec: return [int(spec)] * len(widths)
    narrow, wide = (int(x) for x in spec.split("/")) if spec else power_schedule()
    return [narrow if w < max(widths) else wide for w in widths]


X4_MAX_W = 220   # the x4 card throttles and faults above this (user 2026-10-23): no spec, schedule or `-pl max` may set a narrower-link card higher


def power_targets(widths, spec=None, maxes=None):
    """_power_targets_raw, with every card on a narrower link than the widest held to X4_MAX_W"""
    t = _power_targets_raw(widths, spec, maxes)
    return [min(w, X4_MAX_W) if wd < max(widths) else w for w, wd in zip(t, widths)]


def apply_power_schedule():
    """sets the live limits to the schedule's, where they differ (every run start and `run.py stop` call it, so a chain crossing 23:50 or 07:00 changes at its next cell)."""
    try:
        q = [l.split(",") for l in subprocess.run([NVSMI, "--query-gpu=power.limit,pcie.link.width.current,power.max_limit", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.splitlines() if l.count(",") == 2]
        cur = [int(float(a)) for a, _, _ in q]
        want = power_targets([int(b) for _, b, _ in q], maxes=[int(float(c)) for _, _, c in q])
    except Exception:
        return
    for i, w in enumerate(want):
        if cur[i] != w:
            r = subprocess.run(["sudo", "-n", NVSMI, "-i", str(i), "-pl", str(w)], capture_output=True, text=True)
            print(f"power: GPU {i} {cur[i]} -> {w} W" + ("" if r.returncode == 0 else f"  (failed: {r.stderr.strip()[:80]})"), flush=True)


def wait_vram_free():
    apply_power_schedule()
    kill_leftovers()
    t0 = time.time()
    while True:
        if time.time() - t0 > 60:  # something still holds VRAM: clean up again
            kill_leftovers()
            t0 = time.time()
        out = subprocess.run([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        out = [x for x in out if x.isdigit()]   # no GPU, or an nvidia-smi that fails: nothing to wait for
        if not out or max(int(x) for x in out) < 500:
            wait_host_free()
            return
        time.sleep(1)


def wait_host_free(limit_s=900):
    """the host side of 'the machine is free': a 100 GB model process needs a while to unmap its memory after it was told to stop, and the next load must not start into
    that (swap, a stalled cudaHostRegister: PIN_STALL, ssh crawling). Waits until no llama process is left (they are not killed here: wait_vram_free already did that
    for the stale ones), no process holds more than PERF_BIG_RSS_GB (default 8) of resident memory, and MemAvailable is at least PERF_MIN_AVAIL_GB (default 4: with a 108 GiB pool resident only ~6 GiB is available, and 8 made every run wait the full 900 s).
    The hugepage pool is reserved separately and persists on purpose, it is not counted. Gives up (with a message, the run goes on) after limit_s."""
    big, need = int(os.environ.get("PERF_BIG_RSS_GB", 8)), int(os.environ.get("PERF_MIN_AVAIL_GB", 4))
    t0, why = time.time(), ""
    while time.time() - t0 < limit_s:
        why = ""
        me = os.getpid()
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) == me:
                continue
            try:
                st = open(f"/proc/{d}/stat").read()
                comm = st[st.index("(") + 1:st.rindex(")")]
                state = st[st.rindex(")") + 2:].split()[0]
                if state == "Z":
                    continue
                if comm.startswith("llama-"):
                    why = f"{comm} (pid {d}) is still alive"
                    break
                rss_gb = int(st[st.rindex(")") + 2:].split()[21]) * os.sysconf("SC_PAGE_SIZE") / 2**30
                if rss_gb > big:
                    why = f"{comm} (pid {d}) holds {rss_gb:.0f} GiB"
                    break
            except Exception:
                continue
        if not why:
            avail = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) // 1048576
            if avail < need:
                why = f"only {avail} GiB of RAM available"
        if not why:
            time.sleep(5)   # freed memory comes back to the system a moment after the process is gone
            return
        time.sleep(2)
    print(f"wait_host_free: not free after {limit_s} s ({why}); going on", flush=True)


def fail_reason(build, n=3):
    """why a run failed: the first distinct error lines (' E ' level) of this build's server log, so the history note says 'cudaMalloc failed: out of memory'
    and not just 'HTTP Error 500'"""
    try:
        seen = []
        for l in open(f"/tmp/perf-{build}{LOGTAG}.log", errors="replace"):
            m = re.match(r"^\S+ E (.*)", l.rstrip())
            if m:
                t = re.sub(r"\d+(\.\d+)? MiB|size \d+", "N", m.group(1))[:110]
                if t not in [re.sub(r"\d+(\.\d+)? MiB|size \d+", "N", x)[:110] for x in seen]:
                    seen.append(m.group(1)[:110])
        return " | ".join(seen[:n])
    except Exception:
        return ""


def version(build):
    r = subprocess.run([os.path.join(HERE, build, "llama-server"), "--version"], capture_output=True, text=True,
                       env={**os.environ, "LD_LIBRARY_PATH": ldpath(build)})
    m = re.search(r"version: (.*)", r.stdout + r.stderr)
    return m.group(1).strip() if m else "?"


def cache_stats(log):
    m = re.findall(r"steps=(\d+) hits=(\d+) misses=(\d+) hit-rate=([\d.]+)%", log)
    d = {"steps": int(m[-1][0]), "hits": int(m[-1][1]), "misses": int(m[-1][2]), "hit_rate": float(m[-1][3])} if m else {}
    p = re.findall(r"moe-pred: hit=([\d.]+) hit_without_est=([\d.]+) boost_pts=([\d.]+) up=(\d+) pub=(\d+) used=(\d+) late=(\d+) stale=(\d+) cov1=([\d.]+) cov2=([\d.]+) cov3=([\d.]+)", log)
    if p:  # the predictor's hit rate (share of expert uses already uploaded in time) and its counters, from the last moe-pred line
        d.update(zip(("pred_hit", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late", "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3"), p[-1]))
    d.update(decisions(log))
    return d


DECISION_COLS = ["layout", "placement", "ubatch_max", "pf_mode", "pf_gpus", "pf_skipped", "pf_tok_share", "pf_exp_share", "pf_passes", "pf_pass", "pf_slots",
                 "pf_one_tps", "pf_split_tps", "pf_by", "pf_all", "par_gpus"]


_HWL = {}
_PF_POS = {}   # server log -> how much of it the previous row already read (prompt decisions per run, see the row writer)


def store_cell_time(build, model, test, hw, since, el, prompt_s=None, gen_s=None):
    """the cell's real time as the screen showed it (start of the cell to its end: server start, load, the requests, the teardown) into wall_s of its rows, so -nrt and later tables show the same time (the rows hold what the cell process knew: load + requests)"""
    import sqlite3
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    if prompt_s is not None and "prompt_s" not in [r[1] for r in c.execute("pragma table_info(runs)")]:
        for k in ("prompt_s", "gen_s"): c.execute(f"ALTER TABLE runs ADD COLUMN {k} REAL")
    c.execute("UPDATE runs SET wall_s=?" + (", prompt_s=?, gen_s=?" if prompt_s is not None else "") + " WHERE hw=? AND build=? AND model=? AND test=? AND ts>=?", (round(el, 1), *((round(prompt_s, 1), round(gen_s, 1)) if prompt_s is not None else ()), hw, build, os.path.basename(model), test, time.strftime("%F %T", time.localtime(since - 5))))
    c.commit(); c.close()


def store_gains(rec, build, model, test, hw, since):
    """our decode / prompt speed over the stock row of the cell, into the rows of the cell in runs.db (UPDATE: the rows are written by the cell process before the stock row is known; run-history.csv is append-only and keeps no gains).
    A stock cell stores the gain of our newest run over it, an ours cell its own over the stock reference it was shown with."""
    import sqlite3
    res = rec.get("res") or []
    t = max((x[0] for x in res), default=None); p = max((x[1] for x in res if x[1] is not None), default=None)
    if rec.get("ref"): st = dict(t=t, p=p, sb=rec.get("sbno"), sid=rec.get("rid")); t, p = rec["ref"]["t"], rec["ref"]["p"]
    else: st = rec.get("st")
    if not st or not t: return
    gt = t / st["t"] if st.get("t") else None; gp = p / st["p"] if p and st.get("p") else None
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    c.execute("UPDATE runs SET gain_t=?, gain_pp=?, stock_id=?, stock_bno=? WHERE hw=? AND build=? AND model=? AND test=? AND ts>=?",
              (gt, gp, str(st.get("sid") or ""), str(st.get("sb") or ""), hw, build, os.path.basename(model), test, time.strftime("%F %T", time.localtime(since - 5))))
    c.commit(); c.close()


def server_params(log):
    """the auto choices of a server from its log: decode and batch threads, batch, ubatch (the last 'print_timing' line, the numbers of the request that ran), context (n_ctx_slot) and the weights line"""
    out = {}
    ms = list(re.finditer(r"threads = (\d+) decode / (\d+) batch, batch (\d+), ubatch (\d+)", log)); last = ms[-1] if ms else None
    if last: out.update(thr_dec=int(last.group(1)), thr_batch=int(last.group(2)), batch=int(last.group(3)), ubatch=int(last.group(4)))
    c = re.search(r"n_ctx_slot = (\d+)", log)
    if c: out["ctx"] = int(c.group(1))
    lk = re.search(r"prefill links \(GB/s\): ([^>\n]*?) ->", log)
    if lk: out["link_probe"] = lk.group(1).strip()
    mb = re.search(r"memory is saturated at ([\d.]+) GB/s", log) or re.search(r"-> CPU ([\d.]+) GB/s", log)
    if mb: out["mem_probe_gbs"] = float(mb.group(1))
    w = re.search(r"common_init_result: (weights [^\n]*)", log)
    if w: out["load_mode"] = w.group(1)[:120]
    return out


def decisions(log):
    """the engine's automatic choices of a run, from its server log (builds from 2026-10-14 on; older builds leave them empty):
    layout     "layer layout:" at load: rule (ts, bw, plan, dec, mem) and per card its layer count and ranges
    placement  "MoE placement:" cache or stock, "(re-measure)" when the cache run was swapped for a stock re-measurement
    ubatch_max the prompt micro-batch maximum the auto settings chose
    pf_*       the "prefill decision:" line of the run's largest prompt: mode (one-card, dense-split, dense+expert-split, expert-split, mixed), the cards it ran
               on and the ones skipped, the helpers' token and expert shares, passes, pass size, staging slots, the measured one-card / split prompt t/s of
               that pass size so far, and why (chosen_by); pf_all lists every prompt of the run as tokens:mode/cards"""
    d = {}
    m = re.findall(r"layer layout: rule=(\S+)((?: \S+=\d+\([^)]*\))+) output=(\S+)", log)
    if m:
        d["layout"] = (m[-1][0] + m[-1][1]).strip()
    p = re.findall(r"MoE placement: (cache|stock)\b", log)
    if p:
        d["placement"] = p[-1] + (" (re-measure)" if "re-measuring stock this run" in log else "")
    u = re.findall(r"MoE model .*?ubatch (\d+)", log)
    if u:
        d["ubatch_max"] = int(u[-1])
    # the real maximum: the smaller of the engine's pass ceiling (the size its compute buffer was reserved for) and the pass the RAM allows ("prompt pass N tokens
    # (cut from M to fit)"), the last of each in the log; the auto setup's first pick above (16384 on DeepSeek-V4.1) never ran when either is smaller
    lims = [int(x[-1]) for x in (re.findall(r"limit (\d+) from the pinned-mask budget", log), re.findall(r"fits the ceiling (\d+)", log), re.findall(r"prompt pass (\d+) tokens \(cut from", log)) if x]
    if lims:
        d["ubatch_max"] = min(lims)
    rows = []
    for line in re.findall(r"prefill decision: (.*)", log):
        rows.append({k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', line)})
    if rows:
        top = max(rows, key=lambda r: int(r.get("prompt_tokens", 0) or 0))
        for k, c in (("mode", "pf_mode"), ("gpus", "pf_gpus"), ("skipped", "pf_skipped"), ("helper_token_share", "pf_tok_share"), ("helper_expert_share", "pf_exp_share"),
                     ("passes", "pf_passes"), ("pass_tokens", "pf_pass"), ("staging_slots", "pf_slots"), ("one_card_tps", "pf_one_tps"), ("split_tps", "pf_split_tps"),
                     ("chosen_by", "pf_by")):
            if k in top:
                d[c] = top[k]
        d["pf_all"] = ";".join(f"{r.get('prompt_tokens')}:{r.get('mode')}/{r.get('gpus')}" for r in rows)
    return d


RUN_COLS += DECISION_COLS


def cmd_nsys(a):
    """Where the time of a profiled prompt pass went, per card: an nsys report of a cell run with --wrap "nsys profile ..." (exported to sqlite next to it once).
    The pass is the longest stretch of kernels without a gap over 50 ms (the prompt, its fill copies before it and the few decode tokens after it). Per card: kernel
    time, the gaps of its compute stream (the card waiting: for a hand-off from the other card, an upload, the host), copies by kind, the staged hand-off reads
    (k_xstage_read*, the zero-copy write half of a card-to-card copy) and the kernels that took the most time. --gaps N lists the N longest waits with the kernels
    around them."""
    import sqlite3
    rep = a.rep
    db = re.sub(r"\.nsys-rep$", "", rep) + ".sqlite"
    if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep], check=True, stdout=subprocess.DEVNULL)
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    K = c.execute("select start, end, deviceId, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    M = c.execute("select start, end, deviceId, streamId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    if not K:
        print("no kernels in", db); return
    cl, (cs, ce) = [], K[0][:2]
    for s, e, *_ in K[1:]:
        if s - ce > 50e6: cl.append((cs, ce)); cs, ce = s, e
        else: ce = max(ce, e)
    cl.append((cs, ce))
    w0, w1 = max(cl, key=lambda x: x[1] - x[0])
    W = (w1 - w0) / 1e9
    def union(iv):   # seconds covered by a list of (start, end)
        t, end = 0, None
        for s, e in sorted(iv):
            if end is None or s > end: t += e - s; end = e
            elif e > end: t += e - end; end = e
        return t / 1e9
    kind = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
    print(f"pass window {W:.3f} s (of {len(cl)} kernel stretches; {(K[-1][1] - K[0][0]) / 1e9:.1f} s profiled)")
    print("| card | kernels s | compute-stream gaps s | waits over 1 ms | hand-off reads s | HtoD GB in s | DtoH GB in s | PtoP/DtoD GB |")
    print("|---|---|---|---|---|---|---|---|")
    top = {}
    for d in sorted({k[2] for k in K}):
        kd = [k for k in K if k[2] == d and k[0] >= w0 and k[1] <= w1]
        if not kd: continue
        st = max({k[3] for k in kd}, key=lambda s: sum(1 for k in kd if k[3] == s))   # the compute stream: the one with the most kernels
        ks = [k for k in kd if k[3] == st]
        gaps = [(ks[i + 1][0] - ks[i][1], i) for i in range(len(ks) - 1) if ks[i + 1][0] > ks[i][1]]
        g_all = sum(g for g, _ in gaps) / 1e9
        g_big = [(g, i) for g, i in gaps if g > 1e6]
        xs = union([(k[0], k[1]) for k in kd if names.get(k[4], "").startswith("k_xstage_read")])
        md = [m for m in M if m[2] == d and m[0] >= w0 and m[1] <= w1]
        def cp(ks_):
            sel = [m for m in md if m[4] in ks_]
            return sum(m[5] for m in sel) / 1e9, union([(m[0], m[1]) for m in sel])
        h2d, d2h, p2p = cp((1,)), cp((2,)), cp((8, 10))
        print(f"| {d} | {union([(k[0], k[1]) for k in kd]):.3f} | {g_all:.3f} | {len(g_big)} ({sum(g for g, _ in g_big) / 1e9:.3f} s) | {xs:.3f} | {h2d[0]:.2f} in {h2d[1]:.3f} | {d2h[0]:.2f} in {d2h[1]:.3f} | {p2p[0]:.2f} |")
        tk = {}
        for k in kd: tk[names.get(k[4], "?")] = tk.get(names.get(k[4], "?"), 0) + k[1] - k[0]
        top[d] = (sorted(tk.items(), key=lambda x: -x[1])[:a.top], sorted(g_big, reverse=True)[:a.gaps], ks)
    for d, (tk, gb, ks) in top.items():
        print(f"\ncard {d}: kernels by time: " + ", ".join(f"{n[:40]} {t / 1e9:.3f} s" for n, t in tk))
        for g, i in gb:
            print(f"  wait {g / 1e6:7.1f} ms at {(ks[i][1] - w0) / 1e9:7.3f} s after {names.get(ks[i][4], '?')[:40]} before {names.get(ks[i + 1][4], '?')[:40]}")


def cmd_cells(a):
    """A markdown table of the cells of one campaign prefix (decode, prompt speed, hit rate) with what the server log says about the memory: the context the server
    ran with (n_ctx_slot) and the pool of the first card, so a cell that did not get the setting it was queued for shows at once."""
    import csv as _csv
    rows = list(_csv.reader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "run-history.csv"), newline="")))
    hd = rows[0]; ix = {k: i for i, k in enumerate(hd)}
    print("| cell | decode t/s | prompt t/s | hit rate % | n_ctx_slot | pool card 0 (MiB) |\n|---|---|---|---|---|---|")
    for x in rows[1:]:
        c = x[ix["campaign"]]
        if not c.startswith(a.prefix) or c.endswith("probe") or x[ix["ok"]] != "1":
            continue
        tag = c.rsplit("-", 1)[0]; lg = f"{QDIR}/{tag}.server.log"; pool = ctx = ""
        if os.path.exists(lg):
            t = open(lg, errors="replace").read()
            m = re.search(r"block pool on CUDA0: \d+ layers, ([\d.]+) MiB", t); pool = m.group(1) if m else ""
            m = re.search(r"n_ctx_slot = (\d+)", t); ctx = m.group(1) if m else ""
        print(f"| {tag} | {x[ix['tps']]} | {x[ix['pp']]} | {x[ix['hit']]} | {ctx} | {pool} |")
    return 0
def cmd_misfires(a):
    """runs whose prompt ran in the slower mode: per model, machine and pass size (sqrt(2) steps, as the engine's buckets), the prompt speed of each mode
    the engine chose in any run; a run is listed when another mode of the same key was faster by more than --tol. Also every stock re-measurement."""
    import math
    since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - a.days * 86400))
    rows = [r for r in csv_load("run-history.csv") if r.get("pf_mode") and r.get("ts", "") >= since and (not a.model or a.model.lower() in r.get("model", "").lower())]
    def key(r):
        n = num(r, "pf_pass") or 0
        return (r.get("model", ""), r.get("hw", ""), round(2 * math.log2(n)) if n > 0 else -1)
    best = {}   # key -> mode -> (best prompt t/s, row)
    for r in rows:
        pp = num(r, "pp")
        if pp:
            m = best.setdefault(key(r), {})
            if pp > m.get(r["pf_mode"], (0, None))[0]:
                m[r["pf_mode"]] = (pp, r)
    out = []
    for r in rows:
        pp = num(r, "pp")
        if not pp:
            continue
        others = [(v[0], k) for k, v in best.get(key(r), {}).items() if k != r["pf_mode"]]
        if others:
            bp, bm = max(others)
            if bp > pp * (1 + a.tol):
                out.append((r["ts"], r.get("build", ""), os.path.basename(r.get("model", ""))[:28], r.get("test", ""), r.get("pf_pass", ""), r["pf_mode"],
                            f"{pp:.0f}", bm, f"{bp:.0f}", f"{100 * (bp / pp - 1):.0f}%", r.get("pf_by", "")[:60]))
    print(f"misfires since {since}: {len(out)} of {len(rows)} runs with a prefill decision")
    if out:
        print("| time | build | model | test | pass | chosen | prompt t/s | faster mode | its t/s | lost | why chosen |\n|" + "---|" * 11)
        for o in sorted(out):
            print("| " + " | ".join(o) + " |")
    st = [r for r in csv_load("run-history.csv") if "re-measure" in r.get("placement", "") and r.get("ts", "") >= since]
    if st:
        print(f"\nstock re-measurements of the placement (the cache run before was slower than its median): {len(st)}")
        for r in st:
            print(f"  {r['ts']} {r.get('build', '')} {os.path.basename(r.get('model', ''))[:40]} {r.get('test', '')} campaign {r.get('campaign', '')}")


_EMB_OFF = os.path.getsize(os.path.join(DATA, "embhot.csv")) if os.path.exists(os.path.join(DATA, "embhot.csv")) else 0
EMB_FILE = os.path.join(DATA, "embhot.csv")        # written by --moe log=e (logdir=DATA)
LOG_LETTERS = os.environ.get("PERF_LOG", "r")   # default r only (run history line, no hit rate): measuring max speed logs nothing else; PERF_LOG=esrgp32 when a regression is being diagnosed. --moe log=esrgp for every run of a fork build (stock builds have no --moe): e embedding line, s state snapshot, r run history (this file), g GPU identity, p cache stats; PERF_LOG=e etc. narrows it


READY_STALL = int(os.environ.get("PERF_READY_STALL", "600"))    # seconds without a new log line before a starting server is killed
READY_MAX = int(os.environ.get("PERF_READY_MAX", "1800"))        # seconds from start to "listening" at most


PIN_ATTEMPTS = int(os.environ.get("PERF_PIN_ATTEMPTS", "3"))


def _pin_flag_needed(build, args):
    """Measurements of our builds always run with the weights pinned (user 2026-10-14: "always pinned when measuring our builds, even the CLI"): the fork's own auto rule
    pins only for llama-server (and llama-cli since the CLI sets auto_pin), and only when the model fits the RAM available, so a harness run could silently be an mmap run
    (a different cell: pageable uploads). Unless the cell names a load mode itself or the model is bigger than the available RAM (then the engine's hybrid pin budget
    decides), `--load-mode pin` is added for every non-stock build."""
    if build.startswith("stock") or "--load-mode" in args or "-lm" in args:
        return False
    try:
        import glob as _g
        size = sum(os.path.getsize(f) for f in _g.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", MODEL))) if MODEL and os.path.exists(MODEL) else 0
        avail = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) * 1024
        return 0 < size < avail
    except Exception:
        return False


def server_start(build, env, args):
    """a server that bails with exit code 75 (the engine's [PIN_STALL]: pinning the weights stalled) is started again, three attempts in all; then this run.py exits"""
    if _pin_flag_needed(build, list(args)):
        args = list(args) + ["--load-mode", "pin"]
    for attempt in range(1, PIN_ATTEMPTS + 1):
        try:
            return _server_start_once(build, env, args)
        except RuntimeError as e:
            if "[PIN_FAIL]" in str(e) and attempt < PIN_ATTEMPTS:
                # the engine could not map its cache file from the pool and nothing else holds it: mount the pool again, same size
                print(f"pin pool broken, attempt {attempt}/{PIN_ATTEMPTS}: {str(e)[-200:]}: unmount + mount", flush=True)
                try:
                    n_pool = int(open(HUGE_NR).read().strip())
                    cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
                    cmd_hugepool(argparse.Namespace(action="mount", what=[f"{n_pool}g"] if n_pool > 0 else []))
                except SystemExit as x:
                    sys.exit(f"pin pool remount failed: {x}")
                continue
            if "[PIN_STALL]" not in str(e):
                raise
            print(f"pin stall, attempt {attempt}/{PIN_ATTEMPTS}: {str(e)[-200:]}", flush=True)
            time.sleep(15)   # let the kernel settle before the next try
    sys.exit(f"pinning stalled {PIN_ATTEMPTS} times in a row (exit code 75, [PIN_STALL]): giving up. Memory is too fragmented to pin: REBOOT the machine (it defragments memory) and start again; check kcompactd and /proc/pressure/memory")


_MOE_FLAG = {}


def build_has_moe_flag(build, env):
    if build not in _MOE_FLAG:
        h = subprocess.run([os.path.join(HERE, build, "llama-server"), "--help"], capture_output=True, text=True, env={**os.environ, **env}).stdout
        _MOE_FLAG[build] = "--moe " in h
    return _MOE_FLAG[build]


def drop_moe_flag(build, env, args):
    """a build without --moe (stock, the first releases) would not start with it: drop it; autotune=0 becomes the old environment switch"""
    if build_has_moe_flag(build, env):
        return env, list(args)
    out, i = [], 0
    while i < len(args):
        if args[i] == "--moe" and i + 1 < len(args):
            if "autotune=0" in args[i + 1].split(","):
                env = {**env, "LLAMA_AUTOTUNE": "0"}
            i += 2
        else:
            out.append(args[i]); i += 1
    return env, out


STATE_FILE = {"v": None}   # --clean-state: the learned-state file of the running cell (empty before its first repetition, kept for the next)


def merge_moe(args):
    """the server uses only the LAST of several --moe arguments ("DEPRECATED: argument '--moe' specified multiple times ... only last value will be used"): all of them become one comma separated value, run.py's own keys (log=, state=) first, a variant's keys after them so the variant wins"""
    args, vals, out, i = list(args), [], [], 0
    while i < len(args):
        if args[i] == "--moe" and i + 1 < len(args):
            vals.append(args[i + 1]); i += 2
        else:
            out.append(args[i]); i += 1
    if len(vals) < 2:
        return args
    mine = [v for v in vals if v.startswith("log=")]
    return out + ["--moe", ",".join(mine + [v for v in vals if v not in mine])]


DIAG_ENV = ("POOL_LIVE", "PERF_LEDGER", "GGML_CUDA_SCRATCH_CHECK")   # diagnostics recorded in a row's env (see cmd_cell's rec): ~10% decode for the scratch check on MiMo


def _server_start_once(build, env, args):
    if not build.startswith("stock") and build_has_moe_flag(build, env):   # the fork's research logs and a cold tuner, as --moe settings (older fork builds that do not know a key ignore it; the first releases have no --moe at all)
        args = list(args) + ["--moe", f"log={LOG_LETTERS},logdir={DATA}" + (f",tag={TAGBOX['v']}" if TAGBOX["v"] else "") + (f",state={STATE_FILE['v']}" if STATE_FILE["v"] else "")
                             + (",ledger=1" if os.environ.get("PERF_LEDGER") == "1" else "")   # the pool write ledger (who writes which bytes of a card's pool): cells turn it on, see cmd_cell
                             + (",statsjson=1" if os.environ.get("POOL_LIVE") == "1" else "")   # the live pool view (livelist.py, port 7000): pool maps and ledger writes to /dev/shm
                             + ("" if NOSTATS[0] else ",pass-stats=1")]   # the split pass report per card (extra stream stamps, off in the engine by default): -ns leaves it off
    elif not build.startswith("stock") and STATE_FILE["v"]:
        env = {**env, "LLAMA_MOE_STATE": STATE_FILE["v"]}   # the first releases (no --moe flag) read the environment variable
    env, args = drop_moe_flag(build, env, args)
    args = merge_moe(args)
    logf = f"/tmp/perf-{build}{LOGTAG}.log"
    if os.path.exists(logf): os.replace(logf, logf + ".prev")   # the previous start's log stays readable: in a clean-state cell the cold run is the start before the warm one
    global WATCH_LOG; WATCH_LOG = logf   # post() watches this file
    with open(logf, "w") as lf:
        p = subprocess.Popen(WRAP + [os.path.join(HERE, build, "llama-server"), "-m", MODEL, "--port", str(PORT)] + ([] if BARE else ["-np", "1"])
                             + args, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=None if DETACH[0] else _die_with_parent, start_new_session=DETACH[0])
    t0 = last = time.time()
    size = 0
    while True:
        log = open(logf).read()
        if "listening on" in log:
            return p, logf
        if p.poll() is not None:
            raise RuntimeError(("server bailed, pool broken (exit code 75, [PIN_FAIL]): " if p.returncode == 75 and "[PIN_FAIL]" in log else
                               "server bailed, pinning stalled (exit code 75, [PIN_STALL]): " if p.returncode == 75 else "server exited: ") + log[-300:])
        if "couldn't bind" in log:
            stop(p)
            raise RuntimeError("port busy: " + log[-300:])
        now = time.time()
        if len(log) != size:
            size, last = len(log), now
        if now - last > READY_STALL or now - t0 > READY_MAX:   # a hung start (kernel memory compaction stalled the pinning of the weights for 20 minutes once): the run fails, the next one goes on
            p.kill()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(f"server not ready: no log output for {now - last:.0f} s, {now - t0:.0f} s since start: " + log[-300:])
        time.sleep(1)


PARTIAL = {"text": ""}   # the answer text received so far by the last post(): a failed run stores it (outputs table), so a crash keeps what was generated
ANSWER_FILE = os.environ.get("PERF_ANSWER")   # a file: post() also writes the received text here, flushed per chunk (survives a stopped run.py); the cell subcommand sets it per cell
WATCH_LOG = None   # server log of the running server: post() takes its growth as progress, the server is asked (/slots) only while the log is silent (decode writes nothing per token)


HWF = [""]   # " and hw='pc2'" while a list runs: stock, earlier-build and reference rows of the machine that measured (pc2 numbers must not be divided by pc1 stock)
REMOTE_SLOTS = [None]   # file the progress of a cell on another machine is mirrored to (slots_progress reads it instead of the local /slots)
SLOT_EXACT = [None]   # (prompt total, prompt done, decoded, decode total) from /slots, None when the server does not report them


def slots_progress():
    """(prompt tokens in the slot's KV so far, tokens decoded, last generated bytes) from the server's /slots, None when it does not answer in 5 s (it answers between two batches)"""
    if NOSTATS[0]: return None   # -ns: the server is not asked anything while a cell runs; the bars follow the history estimate and the clock
    if REMOTE_SLOTS[0]:   # a cell on another machine: its poller's last line (time, "n_prompt_tokens":N "n_prompt_tokens_processed":N "n_decoded":N)
        try:
            ln = open(REMOTE_SLOTS[0]).read().strip().split("\n")[-1]
            if time.time() - float(ln.split()[0]) > 6: return None
            g = lambda k: max([int(x) for x in re.findall(r'"%s":(\d+)' % k, ln)] or [0])
            SLOT_EXACT[0] = None
            return max(g("n_prompt_tokens_processed"), LIVE_DONE[0], 0) if g("n_decoded") == 0 else g("n_prompt_tokens"), g("n_decoded"), ""
        except (OSError, ValueError, IndexError): return None
    try:
        sl = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/slots", timeout=5))
        busy = [s for s in sl if s.get("is_processing")] or sl
        SLOT_EXACT[0] = None   # the server's own numbers when it gives them (n_prompt_tokens is the prompt's total there, n_prompt_tokens_processed what is done, n_remain what may still be generated)
        if busy and "n_prompt_tokens_processed" in busy[0]:
            nx = (busy[0].get("next_token") or [{}])[0]
            SLOT_EXACT[0] = (busy[0].get("n_prompt_tokens", 0), busy[0]["n_prompt_tokens_processed"] + busy[0].get("n_prompt_tokens_cache", 0), nx.get("n_decoded", 0), nx.get("n_decoded", 0) + nx.get("n_remain", 0))
        return max((s.get("n_prompt_tokens", 0) for s in sl), default=0), max((s["next_token"][0]["n_decoded"] for s in sl if s.get("next_token")), default=0), "".join(s.get("generated_tail", "") for s in sl)
    except Exception:
        return None


_CPU_LAST = {}
def server_cpu_busy(min_ticks=20):
    """True when a llama-server process used >= min_ticks CPU ticks (1/100 s) since the last call: it is working even though the log is silent and the GPUs are idle"""
    busy = False
    for d in glob.glob("/proc/[0-9]*"):
        try:
            if open(d + "/comm").read().strip() != "llama-server": continue
            f = open(d + "/stat").read().rsplit(")", 1)[1].split(); t = int(f[11]) + int(f[12])   # utime + stime
        except (OSError, IndexError, ValueError): continue
        if t - _CPU_LAST.get(d, t) >= min_ticks: busy = True
        _CPU_LAST[d] = t
    return busy


def gpu_busy():
    """True when any GPU shows utilisation > 0 right now (nvidia-smi, nothing is sent to the server)"""
    try:
        o = subprocess.run([NVSMI, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return any(int(x) > 0 for x in o.split())
    except Exception:
        return True


def post(path, body, timeout=3600, every=10, idle_polls=2):
    """POST and wait for the answer. No time limit: every `every` s the server log's growth is printed; the run counts as stopped only when the log did not grow and every GPU was idle
    for `idle_polls` polls in a row (a slow debug run is waited for, a dead one is not). The server is never asked, so bench numbers stay clean. With PERF_WATCH=1 (debug cells) /slots is
    read as well while the log is silent (ctx length, tokens decoded) and a run whose last generated bytes end in '!!!' is stopped."""
    stream = path == "/completion" and bool(os.environ.get("PERF_WATCH"))   # debug runs only (bench runs stay a plain request): as a stream, so the text written so far survives a server crash or a stopped run (PARTIAL, ANSWER_FILE); the result has the same fields as the plain answer
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps({**body, "stream": True} if stream else body).encode(), {"Content-Type": "application/json"})
    box = {}
    PARTIAL["text"] = ""
    def work():
        try:
            resp = urllib.request.urlopen(req, timeout=None if not stream else timeout)   # a plain request answers once at the end (stock MiMo 200k: ~70 min); the loop below stops a dead server
            if not stream:
                box["r"] = json.load(resp); return
            last = {}
            af = open(ANSWER_FILE, "w") if ANSWER_FILE else None
            try:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]": continue
                    last = json.loads(line[5:])
                    PARTIAL["text"] += last.get("content", "")
                    if af and last.get("content"): af.write(last["content"]); af.flush()
            finally:
                if af: af.close()
            box["r"] = {**last, "content": PARTIAL["text"]}
        except BaseException as e: box["e"] = e
    th = threading.Thread(target=work, daemon=True); th.start()
    t0 = time.time(); lsz = -1; idle = 0; last = None
    while True:
        th.join(every)
        if not th.is_alive(): break
        try: sz = os.path.getsize(WATCH_LOG)
        except (OSError, TypeError): sz = -1
        if sz != lsz:   # the log grew (prefill passes, tuner, uploads): alive
            if lsz >= 0:
                try:
                    with open(WATCH_LOG, "rb") as f: f.seek(max(0, sz - 300)); ln = f.read().decode(errors="replace").rstrip().splitlines()[-1]
                    print(f"progress {time.time() - t0:5.0f} s: log {ln[:110]}", flush=True)
                except (OSError, IndexError): pass
            lsz = sz; idle = 0
            continue
        if os.environ.get("PERF_WATCH"):   # debug cells: ask the server while its log is silent
            pr = slots_progress()
            if pr is not None and pr[2].rstrip().endswith("!!!"):   # the run turned into '!!!!': stop it now (the server is killed with the cell), the dump of a NaN hook is already written
                raise RuntimeError(f"garbled output ('!!!' at the end after {pr[1]} tokens, ctx {pr[0]}): stopped")
            if pr is not None and pr[:2] != (last or (None, None))[:2]:
                print(f"progress {time.time() - t0:5.0f} s: ctx {pr[0]} tokens, decoded {pr[1]}, tail {pr[2][-12:]!r}", flush=True); last = pr
        idle = 0 if (gpu_busy() or server_cpu_busy()) else idle + 1   # a silent log and idle GPUs are no stall while the server burns CPU (lazy pinning of a model bigger than free RAM: 3.5-bit GLM, 2026-10-20, two false stops)
        if idle >= idle_polls:
            raise RuntimeError(f"run stopped: server log silent and GPUs idle for {idle * every} s (after {time.time() - t0:.0f} s)")
    if "e" in box: raise box["e"]
    return box["r"]


def post_loopguard(path, body, timeout=3600, every=128, min_tokens=400, window=150):
    """/completion as a stream: every `every` tokens (from `min_tokens` on) the last `window` words are tested; a repetition loop (unique word 4-gram share < 0.5) closes the connection (the server stops
    generating) and the result is marked _loop: a loop costs about a minute instead of the whole run. Returns the same fields as the plain post (content, timings)."""
    # return_progress: the server streams the prompt's progress after every batch, so a read waits for one batch, not for the whole prompt (stock MiMo takes
    # ~70 min for a 200k prompt at 50 t/s, and the socket timeout below cut such a request at 60 min); `timeout` is the longest silence of a live server
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps({**body, "stream": True, "return_progress": True}).encode(), {"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    text, n, last, prog = "", 0, {}, None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:])
        if "prompt_progress" in j and not j.get("content"):
            prog = j["prompt_progress"]   # the prompt's progress, no token: the last one times the whole prompt (kept for an answer that loops)
            continue
        last = j
        text += j.get("content", ""); n += 1
        if j.get("stop"):
            break
        if n >= min_tokens and n % every == 0:
            w = text.split()[-window:]
            g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
            if len(w) >= window and g and len(set(g)) / len(g) < 0.5:
                resp.close()
                # the prompt ran whatever the answer did: its speed from the last progress (processed less cached tokens, ms since the prompt started)
                pn, pms = (prog["processed"] - prog["cache"], prog["time_ms"]) if prog else (0, 0)
                return {"content": text, "_loop": True, "timings": {"predicted_per_second": 0, "prompt_per_second": pn / pms * 1e3 if pms > 0 else 0, "predicted_n": n, "prompt_n": pn, "prompt_ms": pms}}
    resp.close()
    return {**last, "content": text}


PROCSTAT = {}   # of the last server run: major page faults (reads of mmap'd pages from the disk) and the resident set at the end; page-cache residency of the model files before the start


def model_residency():
    """share of the model file(s) resident in the page cache, in percent (mincore through ctypes: no fincore needed); '' on any failure"""
    try:
        import ctypes, mmap
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        files = sorted(glob.glob(re.sub(r"-0000\d-of-(0000\d)", r"-*-of-\1", MODEL))) or [MODEL]
        res = tot = 0
        page = mmap.PAGESIZE
        for f in files:
            size = os.path.getsize(f)
            with open(f, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                n = (size + page - 1) // page
                vec = ctypes.create_string_buffer(n)
                if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) == 0:
                    res += sum(1 for x in vec.raw if x & 1); tot += n
                del vec
                try:
                    mm.close()
                except BufferError:
                    pass
        return f"pagecache={100.0 * res / tot:.0f}%" if tot else ""
    except Exception:
        return ""


PERFPROF = False   # --perf-prof: perf record (499 Hz, user space) of the server during each run -> fork/bench/perf/<time>-<build>.txt (flat top symbols per library)
GPULOG = False   # --gpulog: 5 Hz nvidia-smi and CPU sampling during the runs (off by default; the slow mode turned out to be real code text, not a power or clock effect)
GPU_Q = "index,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active,power.draw.instant,fan.speed,memory.used"


def gpu_sampler(build, pid=None):
    """5 Hz nvidia-smi samples of every GPU while the server runs -> fork/bench/gpulog/<time>-<build>.csv; stop() returns per-GPU summaries (min SM clock, max temperature, mean power, OR of the throttle reason bits, min PCIe width)"""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "gpulog"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9._-]', '_', build)}{LOGTAG}.csv")
    f = open(path, "w"); f.write("t_ms," + GPU_Q + "\n")
    p = subprocess.Popen([NVSMI, "--query-gpu=" + GPU_Q, "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE, text=True)
    rows, t0 = [], time.time()
    def pump():
        for line in p.stdout:
            v = [x.strip() for x in line.split(",")]
            now = time.time(); f.write(f"{now * 1000:.1f}," + ",".join(v) + "\n"); f.flush(); rows.append(v + [f"{now:.3f}"])   # the last element: the time of the sample
    th = threading.Thread(target=pump, daemon=True); th.start()
    cf = open(path[:-4] + ".cpu.csv", "w"); cf.write("t_ms,sys_busy_pct,proc_cores\n"); cpu, alive = [], [True]
    def cpu_poll():   # whole-machine CPU busy share (/proc/stat) and the server's own CPU use in cores (utime+stime), 5 Hz, same clock as the GPU samples
        tick = os.sysconf("SC_CLK_TCK")
        def rd():
            v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            pt = 0
            try:
                f_ = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); pt = int(f_[11]) + int(f_[12])
            except (OSError, IndexError, ValueError, TypeError):
                pass
            return sum(v), v[3] + v[4], pt, time.time()
        t_, i_, p_, w_ = rd()
        while alive[0]:
            time.sleep(0.2)
            t2, i2, p2, w2 = rd()
            busy = 100.0 * (1 - (i2 - i_) / max(1, t2 - t_)); cores = (p2 - p_) / tick / max(1e-3, w2 - w_)
            cf.write(f"{w2 * 1000:.1f},{busy:.1f},{cores:.2f}\n"); cf.flush(); cpu.append((busy, cores))
            t_, i_, p_, w_ = t2, i2, p2, w2
    cth = threading.Thread(target=cpu_poll, daemon=True); cth.start()
    def stop(t_from=None, t_to=None, t_split=None):
        alive[0] = False; cth.join(3); cf.close()
        p.terminate(); th.join(5); f.close()
        out = {}
        if t_from is not None:   # power over the request only (t_from..t_to, epoch seconds), per GPU in index order: mean, peak, instantaneous peak, energy, share of the samples at the software power cap
            win = {}
            for g in sorted({r[0] for r in rows}):
                r = [x for x in rows if x[0] == g and t_from <= float(x[-1]) <= (t_to or time.time()) and x[1].replace(".", "").isdigit()]
                if len(r) < 2:
                    continue
                pw = [float(x[1]) for x in r]; ts = [float(x[-1]) for x in r]
                ins = [float(x[9]) for x in r if len(x) > 10 and x[9].replace(".", "").isdigit()]
                j = sum(0.5 * (pw[i] + pw[i + 1]) * (ts[i + 1] - ts[i]) for i in range(len(r) - 1))
                cap = 0
                for x in r:
                    try: cap += bool(int(x[8], 16) & 0x4)
                    except ValueError: pass
                tp = [float(x[4]) for x in r if x[4].replace(".", "").isdigit()]
                win[g] = (j / max(1e-6, ts[-1] - ts[0]), max(pw), max(ins) if ins else 0.0, j / 3600.0, 100.0 * cap / len(r), statistics.median(tp) if tp else 0.0, max(tp) if tp else 0.0)
            if win:
                PROCSTAT.update(pw_avg="/".join("%.0f" % v[0] for v in win.values()), pw_max="/".join("%.0f" % v[1] for v in win.values()),
                                pw_inst_max="/".join("%.0f" % v[2] for v in win.values()), energy_wh="%.3f" % sum(v[3] for v in win.values()),
                                cap_pct="/".join("%.0f" % v[4] for v in win.values()),
                                temp_med="/".join("%.0f" % v[5] for v in win.values()), temp_max="/".join("%.0f" % v[6] for v in win.values()))
        tfp = {}
        for g in sorted({r[0] for r in rows}):   # time to full power per GPU, over the whole log
            r = [x for x in rows if x[0] == g and x[1].replace(".", "").isdigit()]
            if len(r) < 2:
                continue
            peak = max(float(x[1]) for x in r)
            if peak < 100:
                continue
            for x in r:
                if float(x[1]) >= 0.9 * peak:
                    tfp[g] = float(x[-1]) - t0
                    break
        if tfp:   # every server start: a bench run restarts the server, each start has its own load
            PROCSTAT.update(t_full_pw="/".join("%.1f" % v for v in tfp.values()))
        for g in sorted({r[0] for r in rows}):
            r = [x for x in rows if x[0] == g and x[2].replace(".", "").isdigit() and x[5].isdigit()]
            b = [x for x in r if int(x[5]) >= 20]   # busy samples: idle and ramp samples would set the minimum
            if not b:
                continue
            clk = sorted(int(x[2]) for x in b); pw = [float(x[1]) for x in b]; thr = 0
            for x in r:
                try: thr |= int(x[8], 16)
                except ValueError: pass
            out[g] = f"util_avg={sum(int(x[5]) for x in r) // len(r)}% busy={100 * len(b) // len(r)}% clk_avg={sum(clk) // len(clk)} clk_p5={clk[len(clk) // 20]} pw_avg={sum(pw) / len(pw):.0f} temp_max={max(int(x[4]) for x in r)} vram_peak_mib={max([int(x[11]) for x in r if len(x) > 12 and x[11].isdigit()] or [0])} thr=0x{thr:x} pcie_w={min(int(x[7]) for x in r)}"
        # prefill and generation separately: t_split = end of the prompt (request start + the server's prompt_ms); stop.phases = {"pp": {gpu: text}, "tg": {gpu: text}}
        stop.phases = {"pp": {}, "tg": {}}
        if t_from is not None and t_split is not None:
            for g in sorted({r[0] for r in rows}):
                for ph, (a, b) in (("pp", (t_from, t_split)), ("tg", (t_split, t_to or time.time()))):
                    r = [x for x in rows if x[0] == g and a <= float(x[-1]) <= b and x[1].replace(".", "").isdigit() and x[2].replace(".", "").isdigit() and x[5].isdigit()]
                    if len(r) < 2:
                        continue
                    cap = sum(1 for x in r if x[8].startswith("0x") and int(x[8], 16) & 0x4)
                    stop.phases[ph][g] = (f"util={sum(int(x[5]) for x in r) // len(r)}% clk={sum(int(x[2]) for x in r) // len(r)} pw={sum(float(x[1]) for x in r) / len(r):.0f}"
                                          f" cap={100 * cap // len(r)}%")
        if cpu:
            out["cpu"] = f"sys_avg={sum(c[0] for c in cpu) / len(cpu):.0f}% proc_cores_avg={sum(c[1] for c in cpu) / len(cpu):.1f}"
        return out, path
    return stop


# one sample a second of CPU clock/temperature and GPU SM clock/temperature; the same shell loop runs on this machine and on a remote one (sys.log), sys_summary() reads both
SYS_SH = (r"""while :; do c=$(awk '/cpu MHz/{s+=$4;n++}END{if(n)printf "%.0f",s/n}' /proc/cpuinfo); t=0; for d in /sys/class/hwmon/hwmon*; do case $(cat $d/name 2>/dev/null) in coretemp|k10temp|zenpower) t=$(( $(cat $d/temp1_input 2>/dev/null || echo 0) / 1000 )); break;; esac; done; """
          r"""g=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ';'); echo "$(date +%s) ${c:-0} $t $g"; sleep 1; done""")


def sys_summary(text, t0=0.0):
    """SYS_SH lines (epoch, cpu MHz, cpu C, 'clk,temp;clk,temp;') from t0 on -> the run columns, each 'min/avg/max' (per GPU joined by |)"""
    cm, ct, gc, gt = [], [], {}, {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 3 or not f[0].isdigit() or int(f[0]) < t0: continue
        try:
            cm.append(float(f[1])); ct.append(float(f[2]))
            for i, g in enumerate(x for x in (f[3].split(";") if len(f) > 3 else []) if "," in x):
                a, b = g.split(","); gc.setdefault(i, []).append(float(a)); gt.setdefault(i, []).append(float(b))
        except ValueError: pass
    mm = lambda v: "%d/%d/%d" % (min(v), sum(v) / len(v), max(v)) if v else ""
    out = {}
    if cm: out["cpu_mhz"] = mm(cm)
    if any(ct): out["cpu_c"] = mm(ct)
    if gc: out["gpu_mhz"] = "|".join(mm(v) for _, v in sorted(gc.items())); out["gpu_c"] = "|".join(mm(v) for _, v in sorted(gt.items()))
    return out


def dmon_summary(text, t0=0.0):
    """remote nvidia-smi dmon -s t lines -> 'peak/mean GB/s' host->device per card (mean of the seconds that moved data), joined by |"""
    rx = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and not line.startswith("#") and f[-3].isdigit():   # "gpu rx tx", with a leading HH:MM:SS when dmon was run with -o T
            try: rx.setdefault(int(f[-3]), []).append(float(f[-2]))
            except ValueError: pass
    def one(v):
        mv = [x for x in v if x > 100]
        return "%.1f/%.1f" % (max(v) / 1024, sum(mv) / len(mv) / 1024 if mv else 0)
    return "|".join(one(v) for _, v in sorted(rx.items()))


def pcie_sampler():
    """host -> device and device -> host PCIe traffic of every GPU while the request runs (nvidia-smi dmon, 1 Hz, MB/s): stop() returns (total GB host->device, peak GB/s of any GPU, mean GB/s of the seconds that moved data, total GB device->host). Cheap enough to be on for every server run."""
    rows = []
    try:
        p = subprocess.Popen([NVSMI, "dmon", "-s", "t", "-d", "1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return lambda: None
    def pump():
        for line in p.stdout:
            f = line.split()   # HH:MM:SS  gpu  rxpci  txpci
            if len(f) >= 4 and not line.startswith("#") and f[1].isdigit():
                try:
                    rows.append((int(f[1]), float(f[2]), float(f[3])))
                except ValueError:
                    pass
    th = threading.Thread(target=pump, daemon=True); th.start()
    def stop():
        p.terminate(); th.join(3)
        if not rows:
            return None
        rx = [r[1] for r in rows]; tx = [r[2] for r in rows]; moving = [x for x in rx if x > 100]
        return round(sum(rx) / 1024, 2), round(max(rx) / 1024, 2), round(sum(moving) / len(moving) / 1024, 2) if moving else 0.0, round(sum(tx) / 1024, 2)
    return stop


def server_run(build, env, args, path, body, render=None):
    PROCSTAT.clear()
    PROCSTAT["cache_state"] = model_residency()
    _t0 = time.time(); PROCSTAT["start_ts"] = time.strftime("%F %T", time.localtime(_t0))
    PROCSTAT["endpoint"] = "completion" if path == "/completion" else "chat"
    key = repr((build, MODEL, tuple(args), sorted((k, v) for k, v in env.items() if os.environ.get(k) != v), os.environ.get("PERF_CELL_ENV", "")))
    d = kept_get() if KEEP_SERVER else None
    if d and d["key"] == key:
        p, logf = PidProc(d["pid"]), d["logf"]   # kept from an earlier run or cell: no start, no load wait
        PROCSTAT["kept"] = 1
        global WATCH_LOG; WATCH_LOG = logf
    else:
        had = kept_get()
        kept_kill()   # another model, build or args (or no keeping): the old server goes first, it holds the VRAM
        if had: wait_vram_free()
        DETACH[0] = KEEP_SERVER
        try: p, logf = server_start(build, env, args)
        finally: DETACH[0] = False
        if KEEP_SERVER: json.dump(dict(key=key, pid=p.pid, logf=logf), open(KEPT, "w"))
    PROCSTAT["load_s"] = round(time.time() - _t0, 1)   # server_start returns when the server answers (about 0 for a kept server)
    if not PROCSTAT.get("kept"):   # the server says when it is done: "srv llama_server: model loaded" with its own clock (min.sec.ms.us since it started), stock and ours alike
        try:
            m = re.search(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) \w+ srv\s+llama_server: model loaded", open(logf, errors="replace").read(), re.M)
            if m: PROCSTAT["load_s"] = round(int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000, 1)
        except (OSError, TypeError): pass
    _ld = PROCSTAT['load_s']
    if _ld < 1:   # a repeat on the kept server loads nothing: the cell's load stays the first run's (the file is rewritten so its mtime still marks the start of this run's prompt phase)
        try:
            _old = float(open(LOAD_FILE).read())
            if _old > 1: _ld = _old
        except (OSError, ValueError): pass
    try: open(LOAD_FILE, "w").write(f"{_ld}")   # the watching run.py shows it in the status (peek_line)
    except OSError: pass
    gstop = gpu_sampler(build, p.pid) if GPULOG else (lambda *a: ({}, ""))
    t_req = [None, None]   # the request window: the power columns cover it only
    perf = None
    if PERFPROF:
        pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "perf"); os.makedirs(pdir, exist_ok=True)
        pbase = os.path.join(pdir, time.strftime("%Y%m%d-%H%M%S") + "-" + build)
        perf = (subprocess.Popen(["perf", "record", "-F", "499", "-o", pbase + ".data", "-p", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), pbase)
    _sys = subprocess.Popen(["bash", "-c", SYS_SH], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True); _sys_t0 = time.time()
    bwstop = pcie_sampler() if GPULOG else (lambda: None)   # the engine's own byte and time counters are the bandwidth record; the nvidia-smi view only with --gpulog
    try:
        def clamp_text(txt, n_cl):   # --clamp N: cut to N tokens (the server counts them, so it is right for every tokenizer)
            for _ in range(4):
                n_tok = len(post("/tokenize", {"content": txt})["tokens"])
                if n_tok <= n_cl:
                    break
                txt = txt[:int(len(txt) * n_cl / n_tok * 0.995)]
            print(f"clamp: prompt cut to {n_tok} tokens (limit {n_cl})", flush=True)
            return txt
        if render:   # the server renders the model's chat template into the prompt text; a clamp cuts the user text first (cutting the rendered text would drop the assistant header)
            if OVR.get("clamp"):
                render = [{**m, "content": clamp_text(m["content"], int(OVR["clamp"]))} if m.get("role") == "user" else m for m in render]
            body = {**body, "prompt": post("/apply-template", {"messages": render})["prompt"]}
        elif OVR.get("clamp") and isinstance(body.get("prompt"), str):
            body = {**body, "prompt": clamp_text(body["prompt"], int(OVR["clamp"]))}
        if SYNC and not WARMING:   # start barrier: drop a ready file, wait until N instances have one
            d, n = SYNC.rsplit(":", 1); os.makedirs(d, exist_ok=True); BAR_N[0] += 1
            open(os.path.join(d, f"b{BAR_N[0]}-{PORT}"), "w").close()
            t_b = time.time()
            while len([f for f in os.listdir(d) if f.startswith(f"b{BAR_N[0]}-")]) < int(n) and time.time() - t_b < 240:
                time.sleep(0.05)
        t_req[0] = time.time()
        npar = int(OVR.get("par") or 1)
        if npar > 1:   # --par N: N requests at once against a server started with -np N (parallel slots); aggregate throughput, every failure counted
            import threading
            outs, errs = [None] * npar, []
            def _one(i):
                try:
                    outs[i] = post(path, {**body, "cache_prompt": False})
                except Exception as e:   # a crash of the server or an HTTP error: the run is a failure, not a missing number
                    errs.append(f"req {i}: {type(e).__name__}: {str(e)[:120]}")
            ths = [threading.Thread(target=_one, args=(i,)) for i in range(npar)]
            t_par = time.time()
            for th in ths: th.start()
            for th in ths: th.join()
            wall = time.time() - t_par
            t_req[1] = time.time()
            ok = [o for o in outs if o]
            PROCSTAT.update(par_n=npar, par_ok=len(ok), par_wall_s=round(wall, 1), par_errors=" | ".join(errs)[:300])
            if errs or not ok:
                raise RuntimeError("parallel requests failed: %d of %d: %s" % (len(errs), npar, " | ".join(errs)[:300]))
            tm = [o.get("timings") or {} for o in ok]
            gen = sum(t.get("predicted_n", 0) for t in tm); pn = sum(t.get("prompt_n", 0) for t in tm)
            pms = max(t.get("prompt_ms", 0) for t in tm) / 1000.0
            PROCSTAT["par_req_tps_mean"] = round(sum(t.get("predicted_per_second", 0) for t in tm) / len(tm), 2)
            res = dict(ok[0]); res["timings"] = {**tm[0], "predicted_n": gen, "prompt_n": pn,
                      "predicted_per_second": gen / max(wall - pms, 1e-3), "prompt_per_second": pn / max(pms, 1e-3), "prompt_ms": pms * 1000}
            return res, logf
        rep = max(1, int(OVR.get("repeat") or 1))   # --repeat N: N requests to this server, the last one is the result (warm)
        if OVR.get("lead"):   # --lead N: a request with a prompt of about N tokens to the same server first (the KV grows for it, the measured request then shrinks it)
            src = open(config.prompt_path("src12k" if int(OVR["lead"]) > 3000 else "longsrc"), errors="replace").read()
            src = (src * (1 + int(OVR["lead"]) * 4 // max(1, len(src))))[: int(OVR["lead"]) * 4]   # ~4 characters per token
            lb = dict(body); lb["n_predict"] = 16
            if "messages" in lb: lb["messages"] = [{"role": "user", "content": "Summarize this code in one sentence:\n" + src}]
            else: lb["prompt"] = "// summary of the code below\n" + src
            r0 = post(path, lb)
            PROCSTAT["lead_pp"] = (r0.get("timings") or {}).get("prompt_per_second")
            PROCSTAT["lead_n"] = (r0.get("timings") or {}).get("prompt_n")
        for i_rep in range(rep):
            try:
                res = (post_loopguard(path, body) if path == "/completion" and not KEEP_LOOPS and int(body.get("n_predict", 0)) >= 800 else post(path, body))
            except Exception as e:   # a server whose slot context is smaller than the prompt (stock splits -c over its slots): cut the prompt to fit instead of failing, and say so (prompt_n and clamped= in the note)
                m = re.search(r"exceeds the available context size \((\d+) tokens\)", str(e))
                if not m or not isinstance(body.get("prompt"), str): raise
                lim = int(m.group(1)) - int(body.get("n_predict") or 0) - 16
                body = {**body, "prompt": clamp_text(body["prompt"], lim)}
                PROCSTAT["clamped"] = lim
                res = (post_loopguard(path, body) if path == "/completion" and not KEEP_LOOPS and int(body.get("n_predict", 0)) >= 800 else post(path, body))
            if i_rep == 0 and rep > 1:
                PROCSTAT["rep_first_pp"] = (res.get("timings") or {}).get("prompt_per_second")
        t_req[1] = time.time()
        return res, logf
    finally:
        try:
            t_split = None
            try:   # end of the prompt phase: request start + prompt_ms of the server's timings
                t_split = t_req[0] + float((res.get("timings") or {}).get("prompt_ms")) / 1000.0
            except (NameError, TypeError, ValueError, AttributeError):
                pass
            gs, gpath = gstop(t_req[0], t_req[1], t_split)
            for ph, col in (("pp", "gpu_pp"), ("tg", "gpu_tg")):
                if getattr(gstop, "phases", None) and gstop.phases[ph]:
                    PROCSTAT[col] = " | ".join(f"gpu{g}: {v}" for g, v in gstop.phases[ph].items())
            if GPULOG:
                PROCSTAT["gpu_log"] = " | ".join((f"gpu{g}: {v}" if g != "cpu" else f"cpu: {v}") for g, v in gs.items()) + " " + os.path.relpath(gpath, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception:
            pass
        try:
            f = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()   # fields after the command name: majflt is the 10th, rss the 22nd
            PROCSTAT.update(major_faults=int(f[9]), rss_kb=int(f[21]) * os.sysconf("SC_PAGE_SIZE") // 1024)
        except (OSError, IndexError, ValueError):
            pass
        try:
            o = subprocess.run([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            PROCSTAT["vram_mib"] = sum(int(x) for x in o.split() if x.isdigit())
        except Exception:
            pass
        try:
            _sys.terminate(); PROCSTAT.update(sys_summary(_sys.stdout.read(), _sys_t0 - 1))
        except Exception:
            pass
        try:
            bw = bwstop()
            if bw:
                PROCSTAT.update(pcie_h2d_gb=bw[0], pcie_h2d_peak_gbs=bw[1], pcie_h2d_avg_gbs=bw[2], pcie_d2h_gb=bw[3])
        except Exception:
            pass
        PROCSTAT["wall_s"] = round(time.time() - _t0, 1)
        PROCSTAT["end_ts"] = time.strftime("%F %T")
        if perf:
            try:
                perf[0].send_signal(signal.SIGINT); perf[0].wait(timeout=120)
                rep = subprocess.run(["perf", "report", "-i", perf[1] + ".data", "--no-children", "--sort", "dso,sym", "--stdio"], capture_output=True, text=True, timeout=600).stdout
                open(perf[1] + ".txt", "w").write("\n".join([l for l in rep.split("\n") if l and not l.startswith("#")][:60]) + "\n")
                os.remove(perf[1] + ".data")
            except Exception as e:
                print("perf-prof failed:", e)
        kept = KEEP_SERVER and p.poll() is None and sys.exc_info()[0] is None   # a failed run stops the kept server too
        if not kept:
            if KEEP_SERVER: kept_kill()
            else: stop(p)
            try:   # the snapshot is written when the server exits
                time.sleep(1)
                PROCSTAT.update(snapshot_knobs(_t0))
                PROCSTAT.update(load_mode_from_log(logf))
                if "load_mode" not in PROCSTAT or ("knobs_end" not in PROCSTAT and not build.startswith("stock")):
                    print(f"note: load_mode/knobs_end not found (log {logf!r}, snapshot since {time.strftime('%T', time.localtime(_t0))}, model {MODEL!r})", flush=True)
            except Exception as e:
                print(f"note: load_mode/knobs_end not recorded: {e!r}", flush=True)
    

def agent_run(build, env, args):
    sys.path.insert(0, os.environ.get("PERF_AGENT_DIR") or fcfg.path("agent"))
    import miniagentic
    p, logf = server_start(build, env, args)
    try:
        return miniagentic.run(f"http://127.0.0.1:{PORT}"), logf
    finally:
        stop(p)


def run_one(build, test, extra_env, plain=False, extra_args=()):
    global COMMON
    if plain and not build.startswith("stock"):
        extra_args = ["--moe-expert-cache", "0"] + list(extra_args)  # fork auto-enables the cache on big MoE models
    COMMON = (COMMON_ALL if not (plain or build.startswith("stock")) else \
        [x for i, x in enumerate(COMMON_ALL) if x not in ("--cpu-moe", "--moe-expert-cache")
         and not (x == "-1" and COMMON_ALL[i - 1] == "--moe-expert-cache")]) + list(extra_args)  # stock/plain: autofit, no cache
    if not kept_get(): wait_vram_free()   # a kept server holds the VRAM itself
    env = {**os.environ, **extra_env, "LD_LIBRARY_PATH": ldpath(build), "LLAMA_MOE_CACHE_STATS": "1",
           "LLAMA_MOE_CACHE_STATSLOG": "32", "LLAMA_MOE_STATSLOG": f"/tmp/perf-{build}{LOGTAG}-stats.txt",  # builds with statslog write a text line per 32 steps (moe-stats.csv)
           "LLAMA_MOE_CACHE_TUNED": os.environ.get("LLAMA_MOE_CACHE_TUNED", "0")}  # no saved tuner decisions: every run starts cold
    row = {}
    TAGBOX["v"] = test   # game4 and edit4 add their prompt below
    if test in ("ppl", "ppl3", "pplb", "pplc", "ppll"):
        nb, nub = {"ppl": ("1", "1"), "ppl3": ("3", "3"), "pplb": ("1024", "1024"), "pplc": ("1024", "256"), "ppll": ("8192", "8192")}[test]   # ppll: long chunks (PERF_PPL_CTX, default 32768) of src_128k.cpp in prompt-sized passes: the attention of a chunk's later tokens reaches past the sparse gate   # pplb: 1024-token batches = the big-batch (prefill) path; pplc: the same batch in 256-token ubatches (4 ubatches per decode call)
        nub = os.environ.get("PERF_PPL_UB") or nub   # PERF_PPL_UB=N: the 1024 tokens of a chunk cut into N-token ubatches (how ppl moves with the number of cuts; stored in the note)
        ctx = os.environ.get("PERF_PPL_CTX", "32768" if test == "ppll" else "1024")   # the chunk length; ppll's own test name keeps its runs out of the README's pplb table
        ppl_text = os.path.join(PROMPT_DIR, os.environ["PERF_PPL_FILE"]) if os.environ.get("PERF_PPL_FILE") else (config.prompt_path("src128k") if test == "ppll" and "src128k" in getattr(config, "PROMPTS", {}) else os.path.join(PROMPT_DIR, "src_128k.cpp") if test == "ppll" else PPL_TEXT)
        args = COMMON + ["-f", ppl_text, "-c", ctx, "--chunks", os.environ.get("PERF_PPL_CHUNKS", "2"), "-b", nb, "-ub", nub]   # PERF_PPL_CHUNKS=12: a paired comparison of two compute paths needs more than 2 chunks (a valid path differs by +-0.5% per chunk)
        args += shlex.split(os.environ.get("PERF_PPL_EXTRA", ""))   # PERF_PPL_EXTRA="--kl-divergence-base F": extra llama-perplexity arguments for stock and ours alike (per-position log-probs, fork/ppl_pos.py)
        env, args = drop_moe_flag(build, env, args)   # the b11509 arm of the regress cell failed on --moe autotune=0,...
        if build.startswith("stock"):   # stock has no --moe-expert-cache
            i = args.index("--moe-expert-cache") if "--moe-expert-cache" in args else -1
            if i >= 0:
                del args[i:i + 2]
        ladder = [None]   # stock: when the fit target leaves too little room the run dies; the target is raised step by step until perplexity runs (last resort, like the server runs)
        if build.startswith("stock"):
            cur = int(args[args.index("-fitt") + 1]) if "-fitt" in args else 0
            ladder += [v for v in FITT_LADDER + [10240, 12288, 14336] if v > cur]
        for v in ladder:
            a2 = list(args)
            if v is not None:
                if "-fitt" in a2:
                    a2[a2.index("-fitt") + 1] = str(v)
                else:
                    a2 += ["-fitt", str(v)]
            _t0 = time.time()
            r = subprocess.run([os.path.join(HERE, build, "llama-perplexity"), "-m", MODEL] + a2,
                               capture_output=True, text=True, env=env, preexec_fn=_die_with_parent)
            log = r.stdout + r.stderr
            _wall = round(time.time() - _t0, 1); _start = time.strftime("%F %T", time.localtime(_t0))
            open(f"/tmp/perf-{build}{LOGTAG}-ppl.log", "w").write(log)   # a failed perplexity run is diagnosable
            if re.findall(r"PPL = ([\d.]+)", log):
                if v is not None:
                    print(f"stock perplexity ran with -fitt {v}", flush=True)
                break
            print(f"perplexity failed (exit {r.returncode}){'' if v is None else f' with -fitt {v}'}: raising -fitt", flush=True)
        s = re.search(r"([\d.]+) seconds per pass", log)
        ppl = re.findall(r"PPL = ([\d.]+)", log)
        row = {"s_per_pass": float(s.group(1)) if s else None, "ppl": float(ppl[-1]) if ppl else None,
               "tps": 1024 / float(s.group(1)) if s else None}
        _ld = re.findall(r"load time =\s*([\d.]+) ms", log)
        _modes = re.findall(r"prefill decision: .*?\bmode=(\S+)", log)   # which path every pass took: the planner may turn the split off after its first measurement
        if _modes:
            _cnt = collections.Counter(_modes)
            row.update({k: v for k, v in decisions(log).items() if k.startswith("pf_")}, pf_mode=_cnt.most_common(1)[0][0], pf_all=";".join(f"{m}x{n}" for m, n in _cnt.most_common()))
        row.update(start_ts=_start, end_ts=time.strftime("%F %T", time.localtime(_t0 + _wall)), wall_s=_wall, **({"load_s": round(float(_ld[-1]) / 1000, 1)} if _ld else {}))
        if not ppl:
            row["note"] = "ppl failed, exit %s: %s" % (r.returncode, " | ".join(l.strip() for l in log.strip().splitlines()[-3:])[:300])
        else:
            _ubk = f" ub={nub}" if os.environ.get("PERF_PPL_UB") else ""
            if _ubk: row["note"] = _ubk.strip()
            ppl_chunks_store(_start, build + _ubk + (" " + " ".join(extra_args) if extra_args else ""), MODEL, test, log)   # the variant's arguments are part of the key: default and forced-split runs of one build stay apart
    else:
        if BARE:
            COMMON = list(extra_args)
            if OVR.get("ctx") and int(OVR["ctx"]) > 0 and "-c" not in COMMON:
                COMMON += ["-c", str(OVR["ctx"])]   # --ctx N in bare mode: the pf tests built their own -c and dropped it (a 'static 20k' run had the calculated 203k)
        _pfn = re.fullmatch(r"pf(\d+)k", test)   # pf4k, pf6k, pf8k ...: the first N thousand tokens of the frozen 12k source (one pass when -ub >= N thousand)
        if test in ("pf12k", "pf12k-stock", "pf128k", "fix12k") or (_pfn and int(_pfn.group(1)) not in (12, 128)):
            pf_ctx = str(OVR.get("ctx") or (131072 if test == "pf128k" else (int(_pfn.group(1)) + 2) * 1024 if _pfn and int(_pfn.group(1)) > 14 else 16384))
            dev = (["-t", THREADS] if THREADS else []) + ["-dev", os.environ.get("PERF_PF_DEV") or ("CUDA1,CUDA0" if DEVS == "CUDA0,CUDA1" else DEVS), "-c", pf_ctx, "-ub", "2048", "-b", "2048"]
            args = list(extra_args) if BARE else dev + ((["--moe-expert-cache", "0"] if test == "pf12k-stock" and not build.startswith("stock") else []) if test == "pf12k-stock" or plain or build.startswith("stock") else ["--cpu-moe", "-nr", "--moe-expert-cache", "-1"]) + list(extra_args)
            if BARE and OVR.get("ctx") and int(OVR["ctx"]) > 0 and "-c" not in args:
                args += ["-c", str(OVR["ctx"])]   # --ctx N in bare mode (see above)
            if BARE and (plain or build.startswith("stock")) and "-c" not in args:
                args += ["-c", pf_ctx]  # stock's default context (4096) rejects the 12k prompt (HTTP 400); the fork sets 32k itself
            if BARE and (plain or build.startswith("stock")) and "-np" not in args and "--parallel" not in args:
                args += ["-np", "1"]    # stock's auto slots (4) split -c: 16384 / 4 = 4096 per slot, the same HTTP 400 ("exceeds the available context size (4096 tokens)", 2026-10-19)
            _pf_big = test == "pf128k" or (_pfn and int(_pfn.group(1)) > 12)   # pf40k, pf64k ...: the first N thousand tokens of the 128k source
            _pf_text = open(config.prompt_path("src128k" if _pf_big else "src12k")).read()
            if _pfn and int(_pfn.group(1)) in PF_FILES:   # a frozen file (prompts/src_Nk.cpp, the cut below written out): the one-pass prompts of `run.py onepass`
                _pf_text = open(config.prompt_path(f"src{_pfn.group(1)}k")).read()
            elif _pfn and int(_pfn.group(1)) not in (12, 128):   # the 12k file is 12294 tokens (GLM tokenizer, 4.4 characters each): cut in proportion
                _need = int(54092 * int(_pfn.group(1)) * 1000 / 12294); _pf_text = (_pf_text * (_need // max(len(_pf_text), 1) + 1))[:_need]   # beyond the source (256k contexts): the source again, the speeds do not depend on the text
            _pf_text = OVR.get("prefix", "") + _pf_text
            if OVR.get("prefix"):   # an instruction (12kin_1kout_code_review): an instruct model needs its chat template or a long answer loops, so it goes through the chat endpoint
                res, logf = server_run(build, env, args, "/v1/chat/completions",
                                       {"messages": [{"role": "user", "content": _pf_text}], "max_tokens": int(OVR.get("tokens") or 1024), **GREEDY, "cache_prompt": False,
                                        "reasoning_format": "none", **({"logprobs": True, "top_logprobs": 1} if os.environ.get("RUN_DBG_TOKENS") else {})})   # DeepSeek V4.1 chat answers came back with content "" (everything generated is thinking, 256 of 256 tokens, finish length, no reasoning_content): the parser dropped it; "none" leaves the raw text in content, so every answer is stored and loop-checked (2026-10-26)
            else:
              _np = int(OVR.get("tokens") or os.environ.get("PERF_NPRED", 64 if test == "pf128k" else 256 if test == "fix12k" else 32))
              if OVR.get("chat"):   # --chat: an instruct model gets its chat template (raw cut text: Qwen3.8 stopped after 1 token at 40k, 2026-10-18); /completion keeps the clamp and the loop guard
                  res, logf = server_run(build, env, args, "/completion", {"n_predict": _np, "cache_prompt": False, **GREEDY},
                                         render=[{"role": "user", "content": OVR["chat"] + "\n\n" + ovr_prompt(_pf_text)}])
              else:
                  res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": ovr_prompt(_pf_text), "n_predict": _np,
                                    "cache_prompt": False, **GREEDY, **(FIXED if test == "fix12k" else {})})
            t = res["timings"]
            row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"], "bestpp": bestpp_run(t),
                   "note": f"prompt_n={t['prompt_n']} prompt_s={t['prompt_ms']/1e3:.1f}" + (f" first_pp={PROCSTAT['rep_first_pp']:.1f}" if PROCSTAT.get("rep_first_pp") else "") + (f" clamped={PROCSTAT['clamped']}" if PROCSTAT.get("clamped") else "")}
            # the answer text into runs.db outputs (every prompt test, 2026-10-18: before that only the md5 of fix12k was kept, so no long-context answer was ever checked);
            # a chat answer keeps its reasoning part too (a thinking model can put everything there and leave the content empty)
            _m = ((res.get("choices") or [{}])[0].get("message") or {})
            row["output"] = res.get("content") or ((_m.get("reasoning_content") + "\n---\n" if _m.get("reasoning_content") else "") + (_m.get("content") or ""))
            if os.environ.get("RUN_DBG_TOKENS"):   # debug: the first generated tokens as the server decoded them (repr), to see what an answer that renders empty is made of
                _lp = (((res.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or [])
                print("DBG TOKENS (first 60 of %d): %s" % (len(_lp), [(x.get("token"), x.get("id")) for x in _lp[:40]]), flush=True)
            if not row["output"]:   # no answer text (DeepSeek pf cells stored 0 characters, 2026-10-26): show what the server did send, so the cause is read from the log
                print("NO ANSWER TEXT: response keys " + str(sorted(res.keys())) + "; message keys " + str(sorted(_m.keys())) + "; finish " + str((res.get("choices") or [{}])[0].get("finish_reason")) + "; raw " + json.dumps(res, default=str)[:600], flush=True)
            if test == "fix12k":
                row["md5"] = hashlib.md5(res["content"].encode()).hexdigest()[:8]
            row.update(PROCSTAT)   # load time, rss, GPU log and power of the request window (--gpulog)
            row.update(cache_stats(open(logf).read()))
            row.update(ok=1 if row.get("pp_tps") else 0, args=" ".join(args))
            return row
        if test == "agent":
            args = COMMON + ["-c", "16384", "-ub", "2048", "-b", "2048"]
            r, logf = agent_run(build, env, args)
            text = json.dumps(r["turns"])
            log = open(logf).read()
            row = {"tps": r["tps"], "pp_tps": r["pp_tps"], "n_gen": r["gen_tokens"],
                   "output": text, "md5": hashlib.md5(text.encode()).hexdigest()[:8],
                   "note": f"turns={len(r['turns'])} compiled={r['compiled']} prompt_tokens={r['prompt_tokens']} "
                           f"prompt_s={r['prompt_s']:.1f} gen_s={r['gen_s']:.1f}"}
            row.update(cache_stats(log))
            row.update(ok=1 if row.get("tps") else 0, args=" ".join(args))
            return row
        chat = test in CHAT_TESTS
        base = CHAT_TESTS.get(test, test)
        if base in ("tetris", "t100", "fix"):
            # t100: what most people run on a CLI: one short prompt, 100 tokens, -c 1024, temperature 0 (GREEDY)
            args = COMMON + _ctx_args((2048 if base == "fix" else 1024))
            res, logf, text = generate(build, env, args, ovr_prompt(config.prompt("tetris") if base == "tetris" else GAME_TEMPLATE.format(word="tetris")),
                                       int(OVR.get("tokens") or (-1 if base == "tetris" else 1024 if base == "fix" else 100)), chat, FIXED if base == "fix" else None)
        elif base == "game4":
            word = GAME_WORDS[(OVR.get('pidx', REP or 0)) % len(GAME_WORDS)]
            TAGBOX["v"] = "game4-" + word.replace(" ", "_") + ("-chat" if chat else "")
            args = COMMON + _ctx_args(1024)
            res, logf, text = generate(build, env, args, ovr_prompt(GAME_TEMPLATE.format(word=word)), int(OVR.get("tokens") or 100), chat, {"cache_prompt": False})
        elif base == "long1k":   # a prompt of about 1024 tokens (src_1k.cpp = the first 4.4k characters of src_3k.cpp) asking for a bug fix, 1024 tokens out, -c 4096: a long prefill and a long decode
            snippet = open(config.prompt_path("src1k")).read()
            args = COMMON + _ctx_args(4096)
            res, logf, text = generate(build, env, args, snippet + "\n\n" + LONG2K_INSTR, int(OVR.get("tokens") or 1024), chat, {"cache_prompt": False})
        elif base == "edit4":
            snippet = open(OVR.get("snippet") or config.prompt_path("src3k")).read()
            ins = EDIT_INSTR[(OVR.get('pidx', REP or 0)) % len(EDIT_INSTR)]
            TAGBOX["v"] = "edit4-%d" % ((REP or 0) % len(EDIT_INSTR)) + ("-chat" if chat else "")
            args = COMMON + _ctx_args(4096)
            res, logf, text = generate(build, env, args, snippet + "\n\n" + ins, int(OVR.get("tokens") or 128), chat, {"cache_prompt": False})
        elif test in config.user_tests():   # a prompt of your own from prompts.ini ([name] path = ..., test = chat|completion, tokens = ..., ctx = ...)
            ut = config.user_tests()[test]
            args = COMMON + _ctx_args(int(ut.get("ctx") or 4096))
            res, logf, text = generate(build, env, args, ovr_prompt(config.prompt(test)), int(OVR.get("tokens") or ut.get("tokens") or 256), ut["test"] == "chat", {"cache_prompt": False})
        else:
            args = COMMON + _ctx_args(4096)
            chatv = test == "chatv"  # varied prompts (index = run counter), thinking off: the topic switches the cache has to follow
            res, logf = server_run(build, env, args, "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": ovr_prompt(CHAT_PROMPTS[OVR.get("pidx", IDX) % len(CHAT_PROMPTS)] if chatv else CHAT_LONG if test == "chat2" else GAME_TEMPLATE.format(word="tetris"))}],
                                    "max_tokens": int(OVR.get("tokens") or (300 if chatv else 1500)), **GREEDY,
                                    **({"chat_template_kwargs": {"enable_thinking": False}} if chatv else {})})
            m = res["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
        t = res["timings"]
        log = open(logf).read()
        row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"], "bestpp": bestpp_run(t),
               "output": text, "md5": hashlib.md5(text.encode()).hexdigest()[:8]}
        row.update(PROCSTAT)
        _rx = r"moe churn = (\d+) uploads, (\d+) evictions, ([\d.]+) MiB uploaded \(this request\)"
        ch = re.findall(_rx, log)
        for _ in range(6):   # the server prints the line just after it answers
            if ch:
                break
            time.sleep(0.5)
            ch = re.findall(_rx, open(logf).read())   # the engine's per-request line: swaps and bytes moved over PCIe
        if ch and row.get("uploads") is None:
            row.update(uploads=int(ch[-1][0]), evictions=int(ch[-1][1]), up_mib=float(ch[-1][2]))
            if row.get("tps") and row.get("n_gen"):
                row["pcie_gbs"] = round(float(ch[-1][2]) / 1024 / (float(row["n_gen"]) / float(row["tps"])), 2)   # average upload rate over the decode time of the request
        row["loop_abort"] = int(bool(res.get("_loop")))
        for nm, key in (("fast", "fast"), ("slow", "slow")):   # per upload link = per GPU: the engine's per-link upload time per expert
            lv = [float(x) for x in re.findall(nm + r" link ([\d.]+) ms", log)]
            lv = [x for x in lv[len(lv) // 2:] if x > 0]
            if lv:
                row["up_ms_" + key] = round(sum(lv) / len(lv), 3)
                if row.get("up_mib") and row.get("uploads"):
                    row["up_gbs_" + key] = round(float(row["up_mib"]) / row["uploads"] / 1024 / (row["up_ms_" + key] / 1000), 2)
        pl = re.search(r"prefill links \(GB/s\):((?: CUDA\d+ [\d.]+)+)", log)   # measured at startup
        if pl:
            for d, v in re.findall(r"CUDA(\d+) ([\d.]+)", pl.group(1)):
                row["link_gbs_cuda" + d] = float(v)
        ums = [float(x) for x in re.findall(r"upload ([\d.]+) ms/expert", log)]
        if ums and row.get("uploads"):
            ums = ums[len(ums) // 2:]
            row["up_ms_expert"] = round(sum(ums) / len(ums), 3)
            if row.get("up_mib"):
                row["up_gbs_eff"] = round(float(row["up_mib"]) / row["uploads"] / 1024 / (row["up_ms_expert"] / 1000), 2)   # GiB per expert / seconds per expert
        mc = t.get("moe_cache") or {}   # the server's per-request cache churn: swaps (uploads of experts into VRAM), evictions, MiB moved
        if mc.get("active"):
            row.update(uploads=mc.get("uploads_request"), evictions=mc.get("evictions_request"), up_mib=mc.get("upload_mib_request"))
            for k, v in (mc.get("pressure") or {}).items():
                row["p_" + k] = v
        if t.get("draft_n"):  # speculative decoding (MTP / draft model): accepted / drafted tokens
            row["note"] = f"draft {t.get('draft_n_accepted')}/{t['draft_n']}"
        if "from pageable host memory" in log:  # weights not pinned: uploads are slower, the run does not compare with pinned ones
            row["note"] = (row.get("note", "") + " pageable uploads").strip()
    row.update(cache_stats(log))
    row.update(ok=1 if row.get("tps") else 0, args=" ".join(args))
    return row


CHAT_TESTS = {"tetrisc": "tetris", "game4c": "game4", "edit4c": "edit4", "long1kc": "long1k"}   # the same prompts through /v1/chat/completions (the chat template, what a user of an instruct model gets); the plain names stay on /completion (raw prompt)


def generate(build, env, args, prompt, n, chat, extra=None, tag=None):
    """one request: chat = /v1/chat/completions with the template, else /completion with the raw prompt. Returns (res, logf, text); the endpoint is recorded in PROCSTAT"""
    templ = chat and TEMPLATED   # --templated: the server's own /apply-template renders the model's chat template, the rendered text goes to /completion (same tokens as the chat endpoint, raw path, loop guard applies)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, **GREEDY} if chat and not templ else {"prompt": prompt, "n_predict": n, **GREEDY}
    body.update(extra or {})
    if chat and not templ and OVR.get("nothink"):
        body["chat_template_kwargs"] = {"enable_thinking": False}   # --nothink: reasoning off, the answer starts at once
    res, logf = server_run(build, env, args, "/completion" if templ else "/v1/chat/completions" if chat else "/completion", body, render=[{"role": "user", "content": prompt}] if templ else None)
    PROCSTAT["endpoint"] = "completion-templated" if templ else "chat" if chat else "completion"
    if chat and not templ:
        m = res["choices"][0]["message"]
        return res, logf, (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    return res, logf, res["content"]


def prep_model():
    """Drop other models' pages from RAM first (fadvise, no root) so they are not evicted during the run, and read this model
    into the page cache when it fits in RAM (bigger than RAM: the throwaway run before each measured run caches the used experts)."""
    import glob
    cur = set(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", MODEL)))
    for f in (glob.glob(os.path.join(MODELS_DIR, "**", "*.gguf"), recursive=True) if MODELS_DIR else []):
        if f not in cur:
            try:
                fd = os.open(f, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
    size = sum(os.path.getsize(f) for f in cur)
    if size < 0.85 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"):
        subprocess.run(["dd", f"if={MODEL}", "of=/dev/null", "bs=16M"], stderr=subprocess.DEVNULL)


_PL = {}


def power_limits(hw):
    """GPU power limits in W of this host, like 350+370 (PERF_PL overrides); only for rows of the local machine, remote ones stay empty.
    Read from the cards for every row (the limits change at the start and the end of a bench call and by hand): a cached first value
    recorded 350+370 for a run that crashed the x4 card at 330 W. A card that does not answer leaves its place out (a fault shows as a shorter value)."""
    if "local" not in _PL:
        _PL["local"] = os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
    if hw != _PL["local"]:
        return ""
    if os.environ.get("PERF_PL") is not None:
        return os.environ["PERF_PL"]
    out = ""
    try:
        out = subprocess.run([NVSMI, "--query-gpu=power.limit", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        pass
    return "+".join(str(round(float(x))) for x in out.split() if x.replace(".", "").isdigit())


def build_info(build):
    """commit / branch / diff hash of a local build, from bin/BUILD_INFO (key=value lines); empty for stock and remote builds"""
    try:
        return dict(l.strip().split("=", 1) for l in open(os.path.join(HERE, build, "BUILD_INFO")) if "=" in l)
    except OSError:
        return {}


STATS_CSV_COLS = ["ts", "build", "commit", "branch", "patch", "hw", "model", "test", "args", "env", "tps", "hit", "windows", "every", "tok_s_mean", "hit_mean", "uploads", "evictions", "up_mib",
                  "ddr_gbs", "pcie_gbs", "pred_up", "pred_pub", "pred_used", "pred_late", "layer_ms_avg", "layer_ms_min", "up_ms_l0", "up_ms_l1"]


def stats_csv_row(w):
    """one run's statslog windows -> totals (uploads, evictions, MiB, predictor counters), means (t/s, hit, DDR, PCIe, layer time), min layer time, median upload time per link"""
    col = lambda i: [x[i] for x in w if len(x) > i]
    mean = lambda v: round(statistics.mean(v), 3) if v else ""
    d = dict(windows=len(w), every=int(w[1][0] - w[0][0]) if len(w) > 1 else 32, tok_s_mean=mean(col(2)), hit_mean=mean(col(3)), uploads=int(sum(col(4))), evictions=int(sum(col(5))),
             up_mib=int(sum(col(6))), ddr_gbs=mean(col(7)) if any(col(7)) else "", pcie_gbs=mean(col(8)))
    if len(w[0]) >= 14:
        d.update(pred_up=int(sum(col(10))), pred_pub=int(sum(col(11))), pred_used=int(sum(col(12))), pred_late=int(sum(col(13))))
    if len(w[0]) >= 18:
        lm = [v for v in col(15) if v > 0]
        d.update(layer_ms_avg=mean(col(14)), layer_ms_min=round(min(lm), 3) if lm else "", up_ms_l0=round(statistics.median(col(16)), 3), up_ms_l1=round(statistics.median(col(17)), 3))
    return d


STATS_METRICS = ["tok_s", "hit_pct", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs", "pred_up", "pred_pub", "pred_used", "pred_late", "layer_ms_avg", "layer_ms_min", "up_ms_l0", "up_ms_l1"]


def stats_section(key, model, test, hw, w):
    """one [section] of stats.ini: the run key (ts + build, as in run-history.csv), then one line per metric with a value per statslog window"""
    cols = [[x[2], x[3], x[4], x[5], x[6], x[7], x[8]] + (x[10:] if len(x) >= 14 else []) for x in w]
    lines = [f"[{key}]", f"model={model}", f"test={test}", f"hw={hw}", f"every={int(w[1][0] - w[0][0]) if len(w) > 1 else 32}  ; tokens per value"]
    lines += [f"{m}=" + ",".join(("%.2f" if m in ("tok_s", "ddr_gbs", "pcie_gbs", "layer_ms_avg", "layer_ms_min", "up_ms_l0", "up_ms_l1") else "%.1f" if m == "hit_pct" else "%.0f") % c[k] for c in cols) for k, m in enumerate(STATS_METRICS[:len(cols[0])])]
    return "\n".join(lines) + "\n\n"


def latest_cache_csv(build):
    """rows of this run's routing/<model>.<test>.<time>.cache.csv (--moe log=pN: a row every N decode tokens), None if there is none (stock builds, no log=p, older than 10 minutes)"""
    tag = TAGBOX["v"]
    if build.startswith("stock") or "p" not in LOG_LETTERS or not MODEL or not tag:
        return None
    fs = sorted(glob.glob(os.path.join(DATA, "routing", f"*.{re.sub('[^A-Za-z0-9_-]', '_', tag)}.*.cache.csv")), key=os.path.getmtime)   # the engine names the file by the model's own name (general.name), which can differ from the file name
    if not fs or time.time() - os.path.getmtime(fs[-1]) > 600:
        return None
    try:
        rows = list(csv.DictReader(open(fs[-1])))
    except OSError:
        return None
    return rows or None


def windows_from_cache_csv(build):
    """statslog-style windows [step, 0, tok/s, hit %, uploads, evictions, MiB, DDR GB/s (not measured: 0), PCIe GB/s, 0, pred_up, pred_pub, pred_used, pred_late] from the cache.csv rows: tokens per second from the t_ms stamps of consecutive rows, so every window is exactly N tokens"""
    rows = latest_cache_csv(build)
    if not rows or "t_ms" not in rows[0]:
        return None
    w = []
    for a, b in zip(rows, rows[1:]):
        dt = (float(b["t_ms"]) - float(a["t_ms"])) / 1000
        n = float(b["step"]) - float(a["step"])
        if dt <= 0 or n <= 0:
            continue
        dem = float(b["demand"])
        mib = float(b.get("pred_mib", 0)) + float(b.get("swap_mib", 0))
        w.append([float(b["step"]), 0.0, n / dt, 100 * float(b["met"]) / dem if dem else 0.0, float(b["uploads"]), float(b["evictions"]), mib, 0.0, mib / 1024 / dt, 0.0,
                  float(b.get("pred_up", 0)), float(b.get("pred_pub", 0)), float(b.get("pred_used", 0)), float(b.get("pred_late", 0))])
    return w or None


def gpu_details():
    """{nvidia-smi index: the card's facts} for the gpuN sections of stats.ini: name, PCI bus id, PCIe link (width now and at most, generation at most: the current generation
    drops when idle), power limit in W, memory in MiB, driver. {} when nvidia-smi is not on this machine (remote and cloud runs)."""
    keys = ["name", "pci_bus", "link_width", "link_width_max", "link_gen_max", "power_limit_w", "mem_mib", "driver"]
    try:
        r = subprocess.run([NVSMI, "--query-gpu=index,name,pci.bus_id,pcie.link.width.current,pcie.link.width.max,pcie.link.gen.max,power.limit,memory.total,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    d = {}
    for l in r.stdout.splitlines():
        v = [x.strip() for x in l.split(",")]
        if len(v) == 9:
            d[v[0]] = dict(zip(keys, [v[1], v[2]] + v[3:6] + [str(round(float(v[6]))) if v[6].replace(".", "").isdigit() else v[6], v[7], v[8]]))
    return d


GPU_DET = []   # gpu_details() of this machine, read once: the live display redraws every tick


def card_label(c):
    """The name a card is shown and called by (user 2026-10-24: "cuda0 cuda1 is confusing"): [p]cuda<n>x<slot width>, p for the primary card (it computes the
    experts of the dense split and takes its full card limit every pass), e.g. pcuda1x16 and cuda0x4 on PC1. The width is the link's CURRENT width from nvidia-smi
    (the max field is the card's own: 16 for the card in the x4 slot too). CUDA<n> is nvidia-smi's index n (run.py starts servers with CUDA_DEVICE_ORDER=PCI_BUS_ID),
    through CUDA_VISIBLE_DEVICES when -ng hides cards. A run on another machine shows no width (this machine's nvidia-smi would describe the wrong cards)."""
    dev = str(c.get("dev", "?"))
    if not GPU_DET: GPU_DET.append({} if HOST[0] else gpu_details())
    k = re.sub(r"\D", "", dev)
    vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    idx = vis[int(k)] if k and vis and int(k) < len(vis) else k
    w = GPU_DET[0].get(idx, {}).get("link_width", "")
    return ("p" if c.get("experts") else "") + dev.lower() + (f"x{w}" if str(w).isdigit() else "")


def split_sections(key, model, test, hw):
    """The dense split's passes of the run (SPLIT_PASSES) -> stats.ini: one [key/global] section (pass speed, phase, tokens) and one [key/gpuN] section per card (CUDA<N>: the
    card's facts from gpu_details, its place in the range order and whether it computes the experts, then per pass its tokens, share, dense, hand-off, KV and expert times and its measured waits), a value per pass. The name has no space before the suffix: backfill.py's
    'ts build' sections skip them."""
    rows = [b for _, b in SPLIT_PASSES["rows"]]
    SPLIT_PASSES["rows"] = []
    if not rows:
        return
    f = lambda xs, fmt: ",".join(fmt % x for x in xs)
    head = lambda: [f"model={model}", f"test={test}", f"hw={hw}", "every=1  ; pass"]
    out = [f"[{key}/global]"] + head()
    out += ["us_tok=" + f([r.get("us_tok", 0) for r in rows], "%.2f"), "tokens=" + f([sum(c.get("tok", 0) for c in r["cards"]) for r in rows], "%d"),
            "phase=" + ",".join(str(r.get("phase", "")).replace(" ", "_") for r in rows), "cards=" + f([len(r["cards"]) for r in rows], "%d"), ""]
    devs = sorted({c.get("dev", "?") for r in rows for c in r["cards"]})
    for dv in devs:
        cs = [next((c for c in r["cards"] if c.get("dev") == dv), {}) for r in rows]
        idx = re.sub(r"\D", "", dv) or dv
        g = gpu_details().get(idx, {})   # the card itself (nvidia-smi index = CUDA index: run.py starts servers with CUDA_DEVICE_ORDER=PCI_BUS_ID)
        last = rows[-1]["cards"]
        out += [f"[{key}/gpu{idx}]", f"dev={dv}"] + [f"{k}={v}" for k, v in g.items()]
        lk = [c.get("link_gbs") for c in last if c.get("dev") == dv and c.get("link_gbs")]
        out += [f"pcie_h2d_probe_gbs={lk[0]:.2f}" if lk else "pcie_h2d_probe_gbs=", f"pcie_h2d_peak_gbs={SPLIT_PASSES.get('peak', {}).get(dv, 0):.2f}"]   # measured at this start / busiest second of the cell
        out += [f"range_pos={next((i for i, c in enumerate(last) if c.get('dev') == dv), -1)}", f"experts_card={int(any(c.get('dev') == dv and c.get('experts') for c in last))}", "every=1  ; pass"]
        for name, k, fmt, mul in (("tokens", "tok", "%d", 1), ("share_pct", "s", "%.1f", 100), ("expert_pct", "f", "%.1f", 100), ("dense_ms", "dense_ms", "%.1f", 1),
                                  ("dense_us_tok", "dense_us_tok", "%.2f", 1), ("hop_us_tok", "hop_us_tok", "%.2f", 1), ("kv_ms", "kv_ms", "%.1f", 1), ("moe_ms", "moe_ms", "%.1f", 1),
                                  ("wait_dense_ms", "wait_dense_ms", "%.1f", 1), ("wait_moe_ms", "wait_moe_ms", "%.1f", 1), ("wait_max_ms", "wait_max_ms", "%.2f", 1)):
            out.append(f"{name}=" + f([c.get(k, 0) * mul for c in cs], fmt))
        out.append("")
    with open(os.path.join(DATA, "stats.ini"), "a") as fh:
        fh.write("\n".join(out) + "\n")


def save_stats(build, ts, test, hw, model):
    """The server's statslog file of the measured run (a value per 32 steps: t/s, hit, uploads, evictions, MiB uploaded, DDR and PCIe GB/s) -> a section of stats.ini."""
    f = f"/tmp/perf-{build}{LOGTAG}-stats.txt"
    if os.path.exists(f):
        w = [[float(x) for x in l.split()] for l in open(f) if l[:1].isdigit() and len(l.split()) in (10, 14, 18)]
        os.remove(f)  # the next run of this build must not re-save it
    else:
        w = windows_from_cache_csv(build)   # release-line builds have no statslog file: the same windows from the engine's --moe log=p series (one row per N decode tokens)
    split_sections(f"{ts} {build}", model, test, hw)
    if w:
        with open(os.path.join(DATA, "stats.ini"), "a") as out:
            sec = stats_section(f"{ts} {build}", model, test, hw, w)
            out.write("\n".join(l for l in sec.split("\n") if not re.fullmatch(r"ddr_gbs=(0\.00,?)+", l)))
    return w


CELLS = {}   # (model, test, build, args, campaign, hw) -> (t/s of its runs, pp of its runs)


# Answer checks of every run, printed at once (2026-10-28: 15 DeepSeek + DSpark repeat requests had stopped printing at ~717 characters for 4096 tokens over
# a day before anyone saw it; their uniq4 and loop looked fine, only the length was wrong):
#  - an answer under half a character per generated token stopped printing (real text is ~3, code ~2.5, CJK ~1): ANSWER STOPPED
#  - a later run of the same greedy cell whose answer is over 15% shorter or longer than the first run's (a repeat request that breaks, a cache or replay bug):
#    REPEAT DIFFERS with the lengths and where they part. Smaller differences are the cache's normal numeric drift. Sampling runs are not compared.
FIRST_ANSWER = {}


def answer_checks(row, key):
    out = row.get("output") or ""
    n_gen = int(float(row.get("n_gen") or 0))
    np_ = re.search(r"(?:^|\s)-np\s+(\d+)", str(key[3]))
    if n_gen >= 256 and len(out) * 2 < n_gen and not (np_ and int(np_.group(1)) > 1):   # -np N > 1: n_gen counts every slot, the text is one slot's
        print(col("1;91", f"    ANSWER STOPPED: {len(out)} characters for {n_gen} tokens (under half a character per token: the rest printed nothing)"), flush=True)
    if not out or "temp" in str(key[3]) or "--sampler" in str(key[3]):
        return
    first = FIRST_ANSWER.setdefault(key, out)
    if first is out or first == out:
        return
    # With the expert cache, upload timing decides which experts run on the GPU and which on the CPU, and that numeric drift changes the text late in a
    # long answer: a later run that parts from the first is normal. A repeat much shorter or longer than the first is not (the DSpark break: 717 against
    # 887 characters): only that is flagged.
    if 0.85 * len(first) <= len(out) <= 1.15 * len(first):
        return
    k = next((i for i, (x, y) in enumerate(zip(first, out)) if x != y), min(len(first), len(out)))
    print(col("1;93", f"    REPEAT DIFFERS from the cell's first run: {len(out)} against {len(first)} characters, parting at character {k}: ...{out[max(0, k - 40):k + 40]!r}"), flush=True)


def text_class(text):
    """(unique word 4-gram share, characters, has code): the speed of an MoE decode depends on what is written (a repetition loop reuses few experts)"""
    w = (text or "").split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    low = (text or "").lower()
    return (round(len(set(g)) / len(g), 3) if g else None), len(text or ""), int("```" in low or "<html" in low or "function " in low or "#include" in low)


def is_loop(text):
    """a repetition loop: the last 30% of the words have a unique word 4-gram share below 0.5 (measured: loops 0.01-0.26, real text 0.83-1.0); needs at least 100 words"""
    t = (text or "").strip()
    if len(t) >= 20 and len(set(t)) <= 3:   # '!!!!!!...': the logits are NaN or broken, argmax is token 0 (a GLM kv-tail run gave 33 t/s and a 99% hit rate with such an answer, 2026-10-17)
        return True
    w = t.split()
    if len(w) < 100:
        return False
    w = w[int(len(w) * 0.7):]
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return bool(g) and len(set(g)) / len(g) < 0.5


TEMPLATED = False   # --templated: chat tests (tetrisc, game4c, edit4c, long1kc, chat) render the template through /apply-template and send the text to /completion
KEEP_LOOPS = False   # --keep-loops: do not discard and repeat looping outputs


def history_line(row, test, hw):
    """one line after a run: t/s, tokens and characters of this run against the earlier good runs (ok=1, no loop) of the same model, test and prompt on this machine, all builds: min / mean / max, and HIGH / LOW outside the range"""
    try:
        import sqlite3
        db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")
        if os.environ.get("FORK_NO_RUNS_DB") or not os.path.exists(db):
            return   # no run log on this machine (or run.py quick): no comparison line, and no empty database file created by a read
        con = sqlite3.connect(db, timeout=10)
        stem = re.sub(r"-0000\d-of-0000\d", "", model_name().split(" ")[0]).removesuffix(".gguf")
        q = "SELECT tps, n_gen, out_chars FROM runs WHERE stem = ? AND test = ? AND hw = ? AND ok = 1 AND COALESCE(loop, 0) = 0 AND tps > 0 AND ts < ?"
        rows = con.execute(q, (stem, test, hw, row["ts"])).fetchall()
        if len(rows) < 3 or not row.get("tps"):
            return
        def mm(i, f="{:.1f}"):
            v = [float(r[i]) for r in rows if r[i] not in (None, "")]
            return "/".join(f.format(x) for x in (min(v), sum(v) / len(v), max(v))) if v else "-"
        tps = float(row["tps"]); lo, hi = min(float(r[0]) for r in rows), max(float(r[0]) for r in rows)
        flag = " HIGH" if tps > hi else " LOW" if tps < lo else ""
        print(f"history ({len(rows)} runs): t/s {mm(0)} min/mean/max, tokens {mm(1, '{:.0f}')}, chars {mm(2, '{:.0f}')} | this run: t/s {tps:.1f}{flag}, tokens {row.get('n_gen')}, chars {row.get('out_chars')}", flush=True)
    except Exception as ex:
        print("history:", ex, flush=True)


def build_number(sha):
    """upstream build number of a commit = its commit count (a docker/source build of stock prints build 0): from the repo this script lives in; None if the commit is not there"""
    if not sha:
        return None
    try:
        r = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-list", "--count", sha], capture_output=True, text=True, timeout=20)
        return int(r.stdout) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def store(row, build, test, rec, note, hw, vstr):
    """One csv row per run: date and commit first (the commit and build number come from the binary's own --version), host, model, settings."""
    mc = re.search(r"commit (\w+)", vstr or "")
    mb = re.search(r"build (\d+)", vstr or "")
    row.update(commit_sha=os.environ.get("PERF_COMMIT") or (mc.group(1) if mc else None),  # PERF_COMMIT: a build of a dirty tree, e.g. <sha>+patch
               build_no=(int(mb.group(1)) if mb else None) or build_number(mc.group(1) if mc else None), origin="stock" if build.startswith("stock") else "fork", hw=hw,
               model=model_name(), ts=time.strftime("%F %T"), build=build, test=test,
               env=json.dumps(rec, sort_keys=True), note=" | ".join(x for x in (note, row.get("note"), "sampler=" + SAMPLER if SAMPLER else None) if x) or None)
    if not row.get("pf_mode") and CAMPAIGN and not build.startswith("stock"):   # which path the prompt passes took (one card, two cards, dense split): from the cell's server log; a run with no decision line says so
        # the LIVE server log (/tmp/perf-<build>.log): the cell's copy in QDIR is made only when the cell ends, so a run's row used to read the copy of the
        # cell before (or of an earlier attempt of this cell)
        lg = next((p for p in (f"/tmp/perf-{build}.log", f"{QDIR}/{CAMPAIGN.rsplit('-', 1)[0]}.server.log") if os.path.exists(p)), None)
        if lg:
            # a kept server logs every run of the cell into one file: the prompt decisions of THIS run are the lines after the previous row's read
            # (reading the whole log gave run 2 the decision of run 1: same prompt size, the first line won). Every run is its own process, so the
            # read position lives in a file, keyed by the log's inode: a server start writes a new log (the old one becomes .prev) and starts again at 0.
            # Load-time keys still come from the whole log.
            txt = open(lg, errors="replace").read()
            pf_pos = os.path.join(QDIR, "pf-pos.json")
            try: P = json.load(open(pf_pos))
            except (OSError, ValueError): P = {}
            ino = os.stat(lg).st_ino
            pos = P.get(lg, [None, 0])[1] if P.get(lg, [None])[0] == ino else 0
            if pos > len(txt): pos = 0
            P[lg] = [ino, len(txt)]
            try: json.dump(P, open(pf_pos, "w"))
            except OSError: pass
            d = {k: v for k, v in decisions(txt).items() if not k.startswith("pf_")}
            d.update(decisions(txt[pos:]))
            row.update({k: v for k, v in d.items() if k.startswith("pf_") or k in ("layout", "placement", "ubatch_max")})
            if not d.get("pf_mode"): row["pf_mode"] = "no-decision"
    if not row.get("par_gpus"):   # every row, also when another path filled the prefill fields first (the bench's own log reading): 1,1 without a decision line
        row["par_gpus"] = par_gpus(row.get("pf_all"), row.get("pf_gpus"), " ".join(str(x) for x in (rec.get("args"), row.get("args")) if x))
    for lg in (f"{QDIR}/{CAMPAIGN}.remote.server.log", f"{QDIR}/{CAMPAIGN.rsplit('-', 1)[0]}.server.log") if CAMPAIGN else ():   # the log of the cell: copied from the remote machine, or the local one
        if os.path.exists(lg):
            for k, v in server_params(open(lg, errors="replace").read()).items(): row.setdefault(k, v)
            break
    if not row.get("hwline"):   # the machine line of this host (a remote cell sets its own), read once per process
        if "hw" not in _HWL: _HWL["hw"] = hw_line()
        row["hwline"] = _HWL["hw"]
    w = save_stats(build, row["ts"], test, hw, row["model"])
    row["uniq4"], row["out_chars"], row["code"] = text_class(row.get("output"))
    row["loop"] = int(bool(row.get("loop_abort")) or is_loop(row.get("output")))
    answer_checks(row, (row["model"], test, build, rec.get("args", ""), CAMPAIGN, hw))
    try:   # the generated text (zlib) into runs.db table outputs, keyed by ts + build
        import runlog
        runlog.insert_output(row["ts"], hw, build, test, row["model"], row.get("md5"), row.get("output"), row["uniq4"], row["out_chars"], row["code"])
    except Exception as ex:
        print("outputs:", ex, file=sys.stderr)
    extra = {k: v for k, v in rec.items() if k != "args"}
    if not row.get("knobs_end") and KNOBS_NOW["v"]:
        row["knobs_end"] = KNOBS_NOW["v"] + " [probe]"   # perplexity and other tools write no snapshot: the cell's probed knobs
    bi = build_info(build)
    if bi.get("commit"):
        row["commit_sha"] = bi["commit"] + ("+" + bi["patch"] if bi.get("patch") and bi.get("patch") != "0" else "")   # the commit the binary was built from (+hash of an uncommitted diff)
    if w:
        csv_append("stats.csv", STATS_CSV_COLS, dict(ts=row["ts"], build=build, commit=row["commit_sha"], branch=bi.get("branch", ""), patch=bi.get("patch", ""), hw=hw, model=row["model"], test=test,
                   args=rec.get("args", ""), env=json.dumps(extra, sort_keys=True, separators=(",", ":")) if extra else "", tps=row.get("tps"), hit=row.get("hit_rate"), **stats_csv_row(w)))
    cell = CELLS.setdefault((row["model"], test, build, rec.get("args", ""), CAMPAIGN, hw), ([], []))
    if row.get("ok", 1) and row.get("tps") and not rec.get("temp"):   # temp=warmpass rows are stored but not part of the cell's statistics
        cell[0].append(float(row["tps"]))
        if row.get("pp_tps"):
            cell[1].append(float(row["pp_tps"]))
    mm = lambda v: (min(v), statistics.median(v), max(v)) if v else (None, None, None)
    st = dict(zip(STAT_COLS, (TAGBOX['v'] or test, len(cell[0]),) + mm(cell[0]) + mm(cell[1])))   # prompt: the test and its prompt (game4-racing, edit4-2, tetris, chat ...)
    csv_append("run-history.csv", RUN_COLS, dict(
        **st, ts=row["ts"], commit=row["commit_sha"], hw=hw, test=test, tps=row.get("tps"), pp=row.get("pp_tps"), hit=row.get("hit_rate"),
        ok=row.get("ok", 1), model=row["model"], build=build, args=rec.get("args", ""), md5=row.get("md5"), campaign=CAMPAIGN,
        note=row["note"], ppl=row.get("ppl"), spp=row.get("s_per_pass"), n_gen=row.get("n_gen"), bno=row["build_no"], origin=row["origin"],
        env=json.dumps(extra, sort_keys=True, separators=(",", ":")) if extra else "", cmd=row.get("args"),
        pl=row.get("pl") or power_limits(hw), **{k: row.get(k) for k in ("thr_dec", "thr_batch", "batch", "ubatch", "ctx", "hwline", "link_probe", "mem_probe_gbs", "cpu_mhz", "cpu_c", "gpu_mhz", "gpu_c", "pcie_rx_gbs")}, proto="v2" if test.startswith("fix") else "v1", rep=REP, seed=SEED if test.startswith("fix") else None,
        branch=bi.get("branch", ""), patch=bi.get("patch", ""), repro=os.environ.get("RUNPY_REPRO"), **{k: row.get(k) for k in tuple(PRESSURE_COLS) + tuple(DECISION_COLS) + ("pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_n", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms", "admit_end", "up_ms_fast", "up_ms_slow", "up_gbs_fast", "up_gbs_slow", "link_gbs_cuda0", "link_gbs_cuda1", "up_fast", "up_slow", "mib_fast", "mib_slow", "busy_fast_ms", "busy_slow_ms", "up_ms_expert", "up_gbs_eff", "pcie_h2d_gb", "pcie_h2d_peak_gbs", "pcie_h2d_avg_gbs", "pcie_d2h_gb", "ddr_util", "up_overlap_pct", "cpu_slow_pct", "lead_end0", "lead_end1", "ev_mean_hits_fast", "ev_mean_hits_slow", "ev_zero_pct_fast", "ev_zero_pct_slow", "ev_ge4_pct_fast", "ev_ge4_pct_slow", "life_fast", "life_slow", "vram_mib", "endpoint", "start_ts", "end_ts", "load_s", "wall_s", "uniq4", "out_chars", "code", "loop", "major_faults", "rss_kb", "cache_state", "gpu_log", "pw_avg", "pw_max", "pw_inst_max", "energy_wh", "cap_pct", "temp_med", "temp_max", "knobs_end", "load_mode", "uploads", "evictions", "up_mib", "pred_hit", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late", "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3")}, **platform(hw)))
    save_output_file(row["ts"], build, test, row["model"])   # fork/outputs/<run id>.txt: the answer, readable on GitHub without runs.db
    global _EMB_OFF
    try:
        import runlog
        _EMB_OFF = runlog.insert_emb(_EMB_OFF, row["ts"], hw, build, test)   # the embhot.csv lines of this run into runs.db
    except Exception as ex:
        print("embhot:", ex, file=sys.stderr)


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def save_output_file(ts, build, test, model, rid=None):
    """the answer of a run as fork/outputs/<runs.db row id>.txt, <id>s.txt for stock builds (header: build, test, model, speeds), so anyone can read it on GitHub next to the README table
    that cites the id; returns the path or None (no stored answer). rid given: that row, else the row of ts + build + test + model."""
    import sqlite3, zlib
    try:
        con = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        q = "select rowid, ts, build, test, model, tps, pp, n_gen, args, pl, start_ts from runs where "
        r = con.execute(q + "rowid=?", (rid,)).fetchone() if rid else con.execute(q + "ts=? and build=? and test=? and model=? order by rowid desc limit 1", (ts, build, test, model)).fetchone()
        if not r: return None
        o = con.execute("select text from outputs where build=? and test=? and model=? and abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) <= 120 order by abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) limit 1",
                        (r[2], r[3], r[4], r[1], r[1])).fetchone()
        if not o: return None
        os.makedirs(OUT_DIR, exist_ok=True)
        f = os.path.join(OUT_DIR, f"{r[0]}{'s' if (r[2] or '').startswith('stock') else ''}.txt")   # stock runs: <id>s, sorts next to the id
        hdr = f"run {r[0]} | {r[2]} | {r[3]} | {os.path.basename(r[4] or '')} | decode {r[5] and round(r[5], 2)} t/s | prompt {r[6] and round(r[6], 1)} t/s | {r[7] and int(r[7])} tokens | args: {r[8] or ''} | power {r[9] or '?'} W | {r[10] or r[1]}\n\n"
        open(f, "w").write(hdr + zlib.decompress(o[0]).decode("utf8", "replace"))
        return f
    except Exception as ex:
        print("outputs file:", ex, file=sys.stderr)
        return None


def show_row(row, build, test):
    print(f"{row['ts']} {build:16s} {test:6s} {row.get('env')} "
          + (f"{row['tps']:6.2f} t/s" if row.get("tps") else "FAIL")
          + (f" {row['s_per_pass']:.2f} s/pass" if row.get("s_per_pass") else "")
          + (f" hit {row['hit_rate']}%" if row.get("hit_rate") is not None else "")
          + (f" pp {row['pp_tps']:.1f} t/s" if row.get("pp_tps") else "")
          + (f" md5={row['md5']}" if row.get("md5") else "") + (f" [{row['note']}]" if row.get("note") and (test == "agent" or "draft" in row["note"] or "pageable" in row["note"]) else "")
          + (f" PPL {row['ppl']}" if row.get("ppl") else ""), flush=True)


def load_mode_from_log(logf):
    """pinned / mmap / mmap-fallback from the start of the server log (the cache's upload-path line and the loader's weights line)"""
    try:
        t = open(logf, errors="replace").read()
    except OSError:
        return {}
    if "reloading with mmap" in t:
        mode = "mmap-after-failed-pin"
    elif "weights pinned" in t or "from pinned host memory" in t:
        mode = "pinned"
    elif "weights NOT pinned" in t or "from pageable host memory" in t or "mmap" in t:
        mode = "mmap"
    else:
        return {}
    for key, name in (("pread -> pinned staging", "+pread"), ("memcpy -> pinned staging (bounce)", "+bounce")):
        if key in t:
            mode += name
    return {"load_mode": mode}


def snapshot_knobs(t0):
    """the run.knobs line of the newest state snapshot written since t0 (fork builds with --moe log=s)"""
    fs = [f for f in glob.glob(os.path.join(DATA, "state-snapshots", "*.ini")) if os.path.getmtime(f) >= t0]
    if not fs or not MODEL:
        return {}
    fs = [f for f in fs if f.endswith(os.path.basename(MODEL) + ".ini")] or fs
    for l in open(max(fs, key=os.path.getmtime), errors="replace"):
        if l.startswith("run.knobs ="):
            return {"knobs_end": l.split("=", 1)[1].strip()}
    return {}


def save_tune_log(row):
    """The self-tune decisions of this run's server log, keyed by the run's date and commit (tune.csv)."""
    try:
        lines = [re.sub(r"^\S+ [A-Z] ", "", l.strip())[:300] for l in open(f"/tmp/perf-{row['build']}{LOGTAG}.log", errors="replace")
                 if "self-tune" in l or "thread autotune" in l]
    except OSError:
        return
    for l in lines:
        csv_append("tune.csv", TUNE_COLS, dict(ts=row["ts"], commit=row["commit_sha"], line=l))


PRED_COLS = ["pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_n", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms", "admit_end"]


def cache_csv_totals(build):
    """sums of the predictor and eviction columns of this run's routing/<model>.<test>.<time>.cache.csv (--moe log=p; admit-pred builds): see totals_of_rows"""
    tag = TAGBOX["v"]
    if build.startswith("stock") or "p" not in LOG_LETTERS or not MODEL or not tag:
        return {}
    fs = sorted(glob.glob(os.path.join(DATA, "routing", f"*.{re.sub('[^A-Za-z0-9_-]', '_', tag)}.*.cache.csv")), key=os.path.getmtime)
    if not fs or time.time() - os.path.getmtime(fs[-1]) > 600:
        return {}
    try:
        return totals_of_rows(list(csv.DictReader(open(fs[-1]))))
    except OSError:
        return {}


def totals_of_rows(rows):
    """the cache.csv rows of one run -> the columns of the run row: predicted experts the router picked before their upload landed (still queued / copy running, and by how many layers ahead they were predicted), predictions made, MiB and link busy time of predicted uploads against cache refills, the admit in force at the end, hits per evicted upload per link, RAM contention, the lead per link"""
    try:
        if not rows or "pm_queue" not in rows[0]:
            return {}
        t = lambda c: sum(float(r[c]) for r in rows)
        out = {c: round(t(c), 1) for c in ("pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms")}
        out["pred_n"] = t("pred_up")
        out.update(pred_pub=t("pred_pub"), pred_used=t("pred_used"), pred_late=t("pred_late"))   # published before their layer, used by a pick, too late to start
        if "mib_l0" in rows[0]:
            out.update(up_fast=t("up_l0"), up_slow=t("up_l1"), mib_fast=round(t("mib_l0"), 1), mib_slow=round(t("mib_l1"), 1), busy_fast_ms=round(t("busy_l0_ms"), 1), busy_slow_ms=round(t("busy_l1_ms"), 1))
        out["admit_end"] = float(rows[-1]["admit"])
        if "evh_l0_0" in rows[0]:
            mid = (0, 1, 2, 3, 4.5, 8, 13)
            names = ("0", "1", "2", "3", "4_5", "6_10", "11p")
            for k, nm in ((0, "fast"), (1, "slow")):
                cnt = [t(f"evh_l{k}_{n}") for n in names]
                n = sum(cnt)
                if n:
                    out["ev_mean_hits_" + nm] = round(sum(c * m for c, m in zip(cnt, mid)) / n, 2)
                    out["ev_zero_pct_" + nm] = round(100 * cnt[0] / n, 1)
                    out["ev_ge4_pct_" + nm] = round(100 * (cnt[4] + cnt[5] + cnt[6]) / n, 1)
                    w = [(float(r[f"life_l{k}_steps"]), sum(float(r[f"evh_l{k}_{x}"]) for x in names)) for r in rows]
                    out["life_" + nm] = round(sum(a * b for a, b in w) / max(1.0, sum(b for _, b in w)), 1)
        if "cpu_slow_pct" in rows[0]:
            mean = lambda c: round(sum(float(r[c]) for r in rows) / len(rows), 3)
            out.update(ddr_util=mean("ddr_util"), up_overlap_pct=mean("up_overlap_pct"), cpu_slow_pct=mean("cpu_slow_pct"), lead_end0=float(rows[-1]["lead_l0"]), lead_end1=float(rows[-1]["lead_l1"]))
        return out
    except (OSError, ValueError, KeyError):
        return {}


def save_routing_output(row, build):
    """the generated text next to the routing trace of the same run: routing/<model>.<test-prompt>.<time>.output.txt (the newest .routing of this model and
    test-prompt tag written in the last 10 minutes; fork builds only, with --moe log=x)"""
    text, tag = row.get("output"), TAGBOX["v"]
    if not text or build.startswith("stock") or "x" not in LOG_LETTERS or not MODEL:
        return
    stem = re.sub(r"-0000\d-of-0000\d", "", os.path.basename(MODEL)).removesuffix(".gguf")
    fs = sorted(glob.glob(os.path.join(DATA, "routing", f"{stem}.{re.sub('[^A-Za-z0-9_-]', '_', tag)}.*.routing")), key=os.path.getmtime)
    if fs and time.time() - os.path.getmtime(fs[-1]) < 600:
        open(fs[-1][:-len(".routing")] + ".output.txt", "w").write(text)


STOCK_FITT = os.path.join(PROMPTS, "stock-fitt.csv")   # model,hw,fitt_mib,ts: the smallest -fitt (fit target, MiB left free per GPU) found to start the stock build on this model and machine
FITT_LADDER = [2048, 3072, 4608, 6144, 8192]            # tried in this order, only when upstream's default fit target cannot start the model (last resort)


def stock_fitt_get(hw):
    try:
        for r in csv.DictReader(open(STOCK_FITT)):
            if r["model"] == os.path.basename(MODEL) and r["hw"] == hw:
                return int(r["fitt_mib"])
    except OSError:
        pass
    return None


def stock_fitt_put(hw, v):
    new = not os.path.exists(STOCK_FITT)
    with open(STOCK_FITT, "a") as f:
        if new:
            f.write("model,hw,fitt_mib,ts\n")
        f.write(f"{os.path.basename(MODEL)},{hw},{v},{time.strftime('%F %T')}\n")


def argv_of(s):
    """Cell and variant arguments as a shell splits them: quotes group a word (-otd "blk\\.0\\..*=CUDA1,blk\\.2\\..*=CUDA0"). str.split() kept the quote characters,
    and the server read CUDA0" as a buffer type ("unknown buffer type", the DSpark placement cells of 2026-10-26). Unbalanced quotes fall back to plain words."""
    try: return shlex.split(s or "")
    except ValueError: return (s or "").split()


def run_cell(build, test, extra, plain, args, bare, note, hw=None, warm=True):
    """One measured run, preceded by a discarded run with the same build and settings: model switches, a previous pinned load
    (drops the page cache) and cache/tuner warm-up all showed 5-12% swings on the first run."""
    global BARE, IDX
    BARE = bare
    xargs = argv_of(args) if args else ()
    hw = hw or os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
    fitt_user = "-fitt" in xargs or "--fit-target" in xargs
    if build.startswith("stock") and not fitt_user:   # the tested value of this model and machine, else upstream's default
        known = stock_fitt_get(hw)
        if known:
            xargs = list(xargs) + ["-fitt", str(known)]
            args = (args + " " if args else "") + f"-fitt {known}"
    if warm:  # --no-warm: the first run as a user sees it
        global WARMING
        WARMING = True
        try:   # same prompt as the measured run; stored with temp=warmpass (a cold first pass on a fresh server, the same prompt again on a kept one), left out of the cell statistics
            wrow = run_one(build, test, extra, plain, xargs)
            if wrow.get("ok", 1):
                wrow.update(cache_csv_totals(build), temp="warmpass")   # also in the runs.db column temp
                store(wrow, build, test, {**extra, **({"bare": "1"} if bare else {}), **({"plain": "1"} if plain else {}), **({"args": args} if args else {}), **{k: str(v) for k, v in OVR.items() if k != "prompt"}, "temp": "warmpass"}, "warm-up pass, same prompt", hw, version(build))
        except Exception as e:
            print(f"warm-up run failed: {e}", flush=True)
        finally:
            WARMING = False
    try:
        row = run_one(build, test, extra, plain, xargs)
    except Exception as e:
        row = {"ok": 0, "note": (str(e)[:80] + " [" + fail_reason(build) + "]" if fail_reason(build) else str(e))[:400]}
        if PARTIAL["text"]: row["output"] = PARTIAL["text"]   # what the server generated before it died or the run was stopped
        if build.startswith("stock") and not fitt_user and "-fitt" not in xargs and ("server exited" in str(e) or "server bailed" in str(e)):
            for v in FITT_LADDER:   # last resort: a larger fit target, upward, the first that starts is remembered for this model and machine
                try:
                    xs = list(xargs) + ["-fitt", str(v)]
                    row = run_one(build, test, extra, plain, xs)
                    args = (args + " " if args else "") + f"-fitt {v}"
                    stock_fitt_put(hw, v)
                    print(f"stock started with -fitt {v} (default and smaller targets failed): stored in {STOCK_FITT}", flush=True)
                    break
                except Exception as e2:
                    row = {"ok": 0, "note": f"also failed with -fitt {v}: " + str(e2)[:250]}
    rec = {**extra, **({"bare": "1"} if bare else {}), **({"plain": "1"} if plain else {}), **({"args": args} if args else {}),
           **{k: str(v) for k, v in OVR.items() if k != "prompt"},
           **{k: os.environ[k] for k in DIAG_ENV if os.environ.get(k) not in (None, "", "0")},
           **({"PERF_LOG": os.environ["PERF_LOG"]} if os.environ.get("PERF_LOG", "r") != "r" else {})}   # research logging beyond the run line (cache stats, traces) marks the row too   # a diagnostic that costs time (live pool view, ledger, scratch check) marks the row: speed comparisons group by env
    want = int(OVR["tokens"]) if OVR.get("tokens") else 0
    for k in range(3):   # a repetition loop is not comparable (a short generation that ended with a real answer is kept): stored as a failed run (ok=0, loop=1 for a loop; excluded from every statistic) and repeated, up to 3 times
        if not row.get("ok", 1):
            break
        loop = not KEEP_LOOPS and (row.get("loop_abort") or is_loop(row.get("output")))
        if not loop:
            break
        h = hw or os.environ.get("PERF_HW") or os.uname().nodename
        why = "loop (stopped early)" if row.get("loop_abort") else "loop"
        store({**row, "ok": 0}, build, test, {**rec, "retry": str(k + 1)}, f"{why}, repeated", h, version(build))
        print(f"{why}: repeating ({k + 1}/3) with {k + 1} newline(s) added to the prompt", flush=True)
        global PROMPT_TWEAK
        PROMPT_TWEAK = k + 1
        rec["tweak"] = str(k + 1)
        try:
            row = run_one(build, test, extra, plain, xargs)
        except Exception as e:
            row = {"ok": 0, "note": str(e)[:300]}
        finally:
            PROMPT_TWEAK = 0
    hw = hw or os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
    time.sleep(0.3)   # the cache log of the last interval is flushed at the next step
    row.update(cache_csv_totals(build))
    store(row, build, test, rec, note, hw, version(build))
    save_routing_output(row, build)
    history_line(row, test, hw)
    try:   # the server log of this run next to its row (slots, GPU split, tuned knobs, placement, threads): fast and slow runs of one build can be compared line by line
        lf = f"/tmp/perf-{build}{LOGTAG}.log"
        if os.path.exists(lf):
            os.makedirs(os.path.join(DATA, "logs", "runs"), exist_ok=True)
            shutil.copyfile(lf, os.path.join(DATA, "logs", "runs", f"{row['ts'].replace(' ', '_')}-{build}-{test}.log"))
    except OSError:
        pass
    row["build"] = build
    save_tune_log(row)
    show_row(row, build, test)
    IDX += 1
    return row


def cmd_run(a):
    if not MODEL:
        sys.exit("set MODEL=/path/to/model.gguf (first split for split models)")
    set_overrides(a)
    extra = dict(kv.split("=", 1) for kv in a.env)
    global CAMPAIGN
    CAMPAIGN = a.campaign or ""
    prep_model()
    for i in range(a.n):
        for build in [resolve_build(x) for x in a.builds]:
            run_cell(build.rstrip("/"), a.test, extra, a.plain, a.args, a.bare, a.note, warm=warm_wanted(a.warm, a.no_warm, MODEL))


def cmd_show(a):
    rows = [r for r in csv_load("run-history.csv") if r["ok"] != "0" and (not a.test or r["test"] == a.test) and (not a.build or r["build"] == a.build)]
    if a.runs:
        for r in rows:
            print(" | ".join(r[k] for k in RUN_COLS if r.get(k) not in (None, "")))
        return
    groups = {}
    for r in rows:
        if num(r, "tps") is not None:
            groups.setdefault(((r["model"] or "?")[:12], r["test"], r["build"], r["args"]), []).append(r)
    print(f"{'model':12s} {'test':11s} {'build':21s} {'args':24s} {'n':>2s} {'t/s mean':>8s} {'sd':>5s} {'s/pass':>7s} {'hit%':>6s} {'pp t/s':>7s}  md5s")
    for (m, t, b, e), g in sorted(groups.items()):
        tps = [num(x, "tps") for x in g]
        mean = lambda k: (lambda v: statistics.mean(v) if v else 0)([num(x, k) for x in g if num(x, k) is not None])
        print(f"{m:12s} {t:11s} {b:21s} {e[:24]:24s} {len(g):2d} {statistics.mean(tps):8.2f} {(statistics.stdev(tps) if len(tps) > 1 else 0):5.2f} "
              f"{mean('spp'):7.2f} {mean('hit'):6.1f} {mean('pp'):7.1f}  {','.join(sorted({x['md5'] for x in g if x['md5']}))}")


def first_shard(path):
    """the name shown for a model: its first file only (a split model lists NAME-00001-of-0000N.gguf, not every shard)"""
    return os.path.basename(re.split(r"[,\s]", str(path).strip())[0])


def model_name():
    return re.split(r"[\\/]", MODEL)[-1]


SAMPLER = ""   # --sampler: extra sampling parameters merged into every request (GREEDY stays the base), recorded in the note of each run


# readable test names <in>_<out>_<what>; they resolve to the tests the history already holds (rows keep the old names, so runs stay comparable)
_PI = config.prompts_ini()   # test aliases from prompts.ini: [name] alias = real test, tokens = answer length, prefix = prompt section prepended, snippet = prompt section sent instead of the default source
def test_name(t):
    """a size is written without the pf (8k, 60k1k, user 2026-10-24: no prefixes); the code's own key still has it"""
    if (m := re.fullmatch(r"(\d+)m(\d+k)?", str(t))): t = f"{int(m[1]) * 1024}k{m[2] or ''}"   # 1m1k = 1024k1k (a million-token prompt: the 128k source repeated, cut on the fly)
    return "pf" + t if re.fullmatch(r"\d+k(\d+k)?", str(t)) else t


class TestChoices(list):
    """The tests a bench / run argument accepts: the listed ones and any pf<N>k (the first N thousand tokens of the source, run -ps)."""
    def __contains__(self, v): return list.__contains__(self, v) or bool(re.fullmatch(r"pf\d+k", str(v)))


ALIASES = {k: v["alias"] for k, v in _PI.items() if v.get("alias")}
ALIAS_TOKENS = {k: int(v["tokens"]) for k, v in _PI.items() if v.get("alias") and v.get("tokens")}
ALIAS_PREFIX = {k: config.prompt(v["prefix"]) for k, v in _PI.items() if v.get("alias") and v.get("prefix")}   # prepended to the 12k prompt (pf12k alone sends the bare file)
ALIAS_SNIPPET = {k: config.prompt_path(v["snippet"]) for k, v in _PI.items() if v.get("alias") and v.get("snippet")}   # ~1024 tokens (edit4c alone sends all of src_3k, ~3k tokens)


def set_overrides(a):
    global SAMPLER
    OVR.clear()
    SAMPLER = getattr(a, "sampler", None) or ""
    for kv in filter(None, SAMPLER.split(",")):
        k, v = kv.split("=", 1)
        try:
            GREEDY[k] = json.loads(v)   # numbers, true / false
        except ValueError:
            GREEDY[k] = v
    for k in ("prompt", "tokens", "ctx", "pidx", "repeat", "clamp", "par", "lead", "chat", "nothink"):
        if getattr(a, k, None) is not None:
            OVR[k] = getattr(a, k)
    if getattr(a, "test", None) in ALIAS_TOKENS and "tokens" not in OVR:
        OVR["tokens"] = ALIAS_TOKENS[a.test]   # alias that fixes the output length (edit4c alone defaults to 128)
    if getattr(a, "test", None) in ALIAS_PREFIX:
        OVR["prefix"] = ALIAS_PREFIX[a.test]
    if getattr(a, "test", None) in ALIAS_SNIPPET:
        OVR["snippet"] = ALIAS_SNIPPET[a.test]
    if getattr(a, "test", None) in ALIASES:
        a.test = ALIASES[a.test]


# ---- machines: machines.csv, one row per machine (id, ssh, root, kind). pc1 is the local host; others run llama-cli over ssh ----
def machines():
    return {r["id"]: r for r in csv_load("machines.csv")}


def platform(hw):
    m = machines().get(hw) or {}
    return {k: m.get(src, "") for k, src in (("chipset", "chipset"), ("cpu", "cpu_short"), ("mem_gb", "mem_gb"), ("mem", "mem_short"), ("mem_peak_gbs", "mem_gbs"), ("pcie", "pcie_short"), ("pcie_peak_gbs", "pcie_gbs"))}


def machine(mid):
    r = machines().get(mid)
    if not r or not r.get("ssh"):
        sys.exit(f"unknown machine {mid}: run.py machine add {mid} --ssh user@host --root DIR --os linux|windows")
    return {"id": r["id"], "ssh": r["ssh"], "root": r["root"], "win": (r.get("kind") or "").lower().startswith("win")}


NUMA_C = r"""#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
/* read rate of every reader node against every buffer node: each node's buffer is first-touched by a thread pinned to that node, readers are pinned one per
   physical core; same stride and accounting as the server's numa_probe, plus a full-line-stream (stride 8) column. argv: reps secs */
#define MAXN 16
#define MAXC 256
static int nn, ncpu[MAXN], cpus[MAXN][MAXC], nid[MAXN]; static size_t BUF = (size_t)256 << 20; static unsigned char *buf[MAXN];
static void pin(int c){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *touch(void *a){ int n = (int)(long)a; pin(cpus[n][0]); buf[n] = malloc(BUF); if (buf[n]) memset(buf[n], 1, BUF); return 0; }
typedef struct { int r, b, t, nt, stride; volatile int *stop; double got; } job;
static void *rd(void *a){ job *j = a; pin(cpus[j->r][j->t]); size_t slice = BUF / j->nt, lo = slice * j->t, o = lo; unsigned long x = 0; double g = 0;
  while (!*j->stop) { size_t end = o + (1 << 20); if (end > lo + slice) end = lo + slice; for (; o < end; o += j->stride) x += *(volatile unsigned long *)(buf[j->b] + o); g += 1 << 20; if (o >= lo + slice) o = lo; }
  j->got = g + (x & 1); return 0; }
static double cell(int r, int b, double secs, int stride){ volatile int stop = 0; int nt = ncpu[r]; pthread_t th[MAXC]; job jb[MAXC];
  for (int t = 0; t < nt; ++t) { jb[t] = (job){ r, b, t, nt, stride, &stop, 0 }; pthread_create(&th[t], 0, rd, &jb[t]); }
  double t0 = now(); while (now() - t0 < secs) { struct timespec d = {0, 2000000}; nanosleep(&d, 0); } stop = 1; for (int t = 0; t < nt; ++t) pthread_join(th[t], 0);
  double dt = now() - t0, g = 0; for (int t = 0; t < nt; ++t) g += jb[t].got / dt / 1e9; return g; }
static int parse(const char *s, int *out){ int n = 0; while (*s) { int a = strtol(s, (char **)&s, 10), b = a; if (*s == '-') b = strtol(s + 1, (char **)&s, 10); for (int k = a; k <= b && n < MAXC; ++k) out[n++] = k; if (*s == ',') ++s; else break; } return n; }
int main(int argc, char **argv){ int reps = argc > 1 ? atoi(argv[1]) : 2; double secs = argc > 2 ? atof(argv[2]) : 0.15;
  for (int n = 0; n < 64 && nn < MAXN; ++n) { char p[96], l[2048]; snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", n); FILE *f = fopen(p, "r"); if (!f) { if (nn) break; continue; }
    if (!fgets(l, sizeof l, f)) { fclose(f); continue; } fclose(f); int all[MAXC], na = parse(l, all), k = 0;
    for (int i = 0; i < na; ++i) { char q[128], sb[64]; snprintf(q, sizeof q, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", all[i]); FILE *g = fopen(q, "r"); int first = 1;
      if (g) { if (fgets(sb, sizeof sb, g)) first = atoi(sb) == all[i]; fclose(g); } if (first) cpus[nn][k++] = all[i]; }
    ncpu[nn] = k; nid[nn] = n; if (k) ++nn; }
  if (nn < 2) { printf("%d NUMA node: nothing to compare\n", nn); return 0; }
  for (int n = 0; n < nn; ++n) { pthread_t t; pthread_create(&t, 0, touch, (void *)(long)n); pthread_join(t, 0); if (!buf[n]) { puts("alloc failed"); return 1; } }
  for (int st = 64; st >= 8; st /= 8) for (int rep = 0; rep < reps; ++rep) { printf("stride %d, run %d: reader node (cores) -> buffer nodes", st, rep + 1);
    for (int b = 0; b < nn; ++b) printf("%8d", nid[b]); puts("");
    for (int r = 0; r < nn; ++r) { printf("  node %d (%2d cores)%*s", nid[r], ncpu[r], 20, ""); for (int b = 0; b < nn; ++b) printf("%8.1f", cell(r, b, secs, st)); puts(""); } }
  return 0; }
"""


NUMA_CU = r"""#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <sched.h>
#include <pthread.h>
#include <cuda_runtime.h>
// host -> device copy rate of every GPU against a pinned buffer on every NUMA node. The buffer is allocated and first-touched by a thread pinned to a core of the node.
static void pin(int c){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
int main(int argc, char ** argv){ int reps = argc > 1 ? atoi(argv[1]) : 2; const size_t N = (size_t)256 << 20;
  std::vector<int> first, nid;   // first physical core of each node that has CPUs
  for (int n = 0; n < 64; ++n) { std::ifstream f("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist"); std::string l; if (!std::getline(f, l)) { if (!first.empty()) break; continue; } first.push_back(atoi(l.c_str())); nid.push_back(n); }
  int ng = 0; cudaError_t ce = cudaGetDeviceCount(&ng); if (ce != cudaSuccess) printf("cudaGetDeviceCount: %s", cudaGetErrorString(ce)); puts(""); std::vector<void *> hb(first.size(), nullptr);
  for (size_t b = 0; b < first.size(); ++b) std::thread([&, b] { pin(first[b]); if (cudaHostAlloc(&hb[b], N, cudaHostAllocDefault) == cudaSuccess) memset(hb[b], 1, N); }).join();
  for (int r = 0; r < reps; ++r) { printf("run %d: GPU (pci, own node) -> pinned buffer on node", r + 1); for (int n : nid) printf("%8d", n); puts("");
    for (int g = 0; g < ng; ++g) { cudaSetDevice(g); char pci[32] = {}; cudaDeviceGetPCIBusId(pci, sizeof pci, g); std::string pp = pci; for (auto & c : pp) c = tolower(c);
      std::ifstream f("/sys/bus/pci/devices/" + pp + "/numa_node"); std::string own; std::getline(f, own); void * d = nullptr; cudaMalloc(&d, N); cudaStream_t st; cudaStreamCreate(&st);
      printf("  GPU %d %s node %-3s%*s", g, pci, own.c_str(), 6, "");
      for (size_t b = 0; b < first.size(); ++b) { if (!hb[b]) { printf("%8s", "-"); continue; } cudaMemcpyAsync(d, hb[b], N, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 8; ++i) cudaMemcpyAsync(d, hb[b], N, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); printf("%8.1f", 8.0 * N / dt / 1e9); }
      puts(""); cudaFree(d); cudaStreamDestroy(st); } }
  return 0; }
"""


def cmd_numa(a):
    """numa [-pc ID] [--reps N]: the NUMA map of a machine (GPU -> node from sysfs) and a node x node RAM read-rate table, from a small C program compiled on the machine (gcc): the same measurement as the server's numa_probe, without loading a model (a few seconds)"""
    if a.machine != "pc1": HOST[0] = dict(machines()[a.machine], id=a.machine)
    print(hsh("for d in /sys/bus/pci/devices/*; do [ \"$(cat $d/vendor)\" = 0x10de ] && [ \"$(cat $d/class)\" = 0x030200 -o \"$(cat $d/class)\" = 0x030000 ] && echo \"GPU $(basename $d) numa_node=$(cat $d/numa_node)\"; done; lscpu | grep -i 'numa node'"))
    if a.gpu:   # host -> device rate of every GPU against pinned memory on every node
        print(hsh("d=$(mktemp -d); cat > $d/g.cu <<'EOF'\n" + NUMA_CU + "\nEOF\n(nvcc -O2 -std=c++17 $d/g.cu -o $d/g && $d/g %d) 2>&1 | tail -30; rm -rf $d" % a.reps)); return
    print(hsh("d=$(mktemp -d); cat > $d/n.c <<'EOF'\n" + NUMA_C + "\nEOF\ngcc -O2 -pthread $d/n.c -o $d/n && $d/n %d %s; rm -rf $d" % (a.reps, a.secs)))


def cmd_machine(a):
    ms = machines()
    if a.action == "add":
        mid = a.id or "cld"   # the rented box of a temporary session: `machine add --ssh "-p 47086 root@93.91.156.104 -L 8080:localhost:8080"` (the line the provider shows) updates cld
        sshv, root = a.ssh, a.root
        if sshv and (" " in sshv.strip() or sshv.startswith("-")):   # a pasted ssh command line: its alias goes into ~/.ssh/config, port forwards are dropped, the machine row is reset (a new box behind an old name)
            tk = shlex.split(sshv); port = ""; ident = ""; dest = ""; i = 0
            while i < len(tk):
                if tk[i] == "-p" and i + 1 < len(tk): port = tk[i + 1]; i += 2
                elif tk[i] == "-i" and i + 1 < len(tk): ident = tk[i + 1]; i += 2
                elif tk[i] in ("-L", "-R", "-D", "-o", "-J") and i + 1 < len(tk): i += 2
                elif tk[i].startswith("-"): i += 1
                else: dest = tk[i]; i += 1
            user, _, host = dest.rpartition("@")
            blk = f"Host {mid}\n    HostName {host}\n" + (f"    Port {port}\n" if port else "") + f"    User {user or 'root'}\n" + (f"    IdentityFile {ident}\n" if ident else "") + "    StrictHostKeyChecking accept-new\n    ServerAliveInterval 30\n    LogLevel QUIET\n"
            cfg = os.path.expanduser("~/.ssh/config")
            try: old = open(cfg).read()
            except OSError: old = ""
            new, n = re.subn(rf"^Host {re.escape(mid)}\n(?:[ \t]+.*\n?)*", blk, old, flags=re.M)
            open(cfg, "w").write(new if n else old.rstrip("\n") + ("\n\n" if old.strip() else "") + blk)
            sshv = mid; root = root or ("/root/rel" if (user or "root") == "root" else "~/rel")
            ms[mid] = dict(id=mid, ssh=mid, root=root, kind=a.os)
            print(f"{mid}: ssh {user or 'root'}@{host}" + (f":{port}" if port else "") + f", root {root}")
        else:
            ms.setdefault(mid, {})
            ms[mid].update(id=mid, ssh=sshv or mid, root=root or ms[mid].get("root") or "/root/rel", kind=a.os)
        with open(os.path.join(DATA, "machines.csv"), "w", newline="") as f:  # tiny file, rewritten whole
            w = csv.DictWriter(f, MACH_COLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(ms.values())
    for r in ms.values():
        if not getattr(a, "quiet", False): print(" | ".join(r.get(k, "") for k in ("id", "ssh", "root", "kind", "cpu", "gpus")))


REMOTE_PORT = 8099


def gguf_ctx_train(path, data=None):
    """the model's trained context (<arch>.context_length) read from the GGUF header; 0 when it cannot be read. The header is read as far as that key, not the tensors."""
    import struct
    try:
        with (io.BytesIO(data) if data else open(path, "rb")) as f:   # data: the first bytes of a file that is on another machine only
            if f.read(4) != b"GGUF": return 0
            f.read(4); f.read(8); nkv = struct.unpack("<Q", f.read(8))[0]
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
            def rd(t):
                if t == 8: return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf8", "replace")
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
                    return [rd(et) for _ in range(n)]
                return struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
            arch = None
            for _ in range(nkv):
                k = rd(8); t = struct.unpack("<I", f.read(4))[0]; v = rd(t)
                if k == "general.architecture": arch = v
                elif arch and k == f"{arch}.context_length": return int(v)
    except (OSError, struct.error, KeyError, ValueError): pass
    return 0


def gguf_meta(path):
    """the scalar metadata of a GGUF header as {key: value} (arrays longer than 64 left out); {} when it cannot be read. For a split model give the first part."""
    import struct
    meta = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF": return {}
            f.read(4); f.read(8); nkv = struct.unpack("<Q", f.read(8))[0]
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
            def rd(t):
                if t == 8: return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf8", "replace")
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
                    if et != 8 and et in fmt and n > 64: f.seek(n * struct.calcsize(fmt[et]), 1); return None
                    return [rd(et) for _ in range(n)]
                return struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
            for _ in range(nkv):
                k = rd(8); t = struct.unpack("<I", f.read(4))[0]; v = rd(t)
                if k.startswith("tokenizer.") and isinstance(v, list): continue
                if v is not None and not (isinstance(v, list) and len(v) > 64): meta[k] = v
    except (OSError, struct.error, KeyError, ValueError): pass
    return meta


def kv_bytes_per_token(meta, kv_bytes=2):
    """bytes of KV cache one token takes in this model (f16 = 2 bytes per element), from its GGUF metadata. MLA (kv_lora_rank): one latent plus the rope part per layer. Standard attention: K and V of
    n_head_kv heads of the key/value length per layer. Hybrid models (full_attention_interval) have attention in every Nth layer only. A sliding-window layer is counted as a full one: the estimate
    errs high, the start prompt errs low and the search corrects it."""
    arch = meta.get("general.architecture", ""); g = lambda k, d=0: meta.get(f"{arch}.{k}", d)
    L = int(g("block_count")); emb = int(g("embedding_length")); heads = g("attention.head_count", 1)
    attn_layers = L // int(g("full_attention_interval", 1) or 1)
    if g("attention.kv_lora_rank"): return attn_layers * (int(g("attention.kv_lora_rank")) + int(g("rope.dimension_count", 64))) * kv_bytes
    hkv = g("attention.head_count_kv", heads); hkv = max(hkv) if isinstance(hkv, list) else hkv
    heads = max(heads) if isinstance(heads, list) else heads
    kl = int(g("attention.key_length", emb // max(heads, 1))); vl = int(g("attention.value_length", kl))
    return int(attn_layers * hkv * (kl + vl) * kv_bytes)


def feasible_ctx(path, hw_gb):
    """tokens of context this machine can hold for the model, by arithmetic: the KV cache of one token (kv_bytes_per_token) against the memory it may take: half of the cards' VRAM (the rest
    holds the weights, the expert cache and the compute buffers) plus a quarter of the available RAM (the KV ladder spills to RAM); never more than the trained context. (tokens, bytes per token, budget GiB)"""
    meta = gguf_meta(path); bpt = kv_bytes_per_token(meta) if meta else 0
    try: ram = next(int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    except (OSError, ValueError, StopIteration): ram = 0
    budget = 0.5 * hw_gb * 2**30 + 0.25 * ram
    train = int(meta.get(f"{meta.get('general.architecture', '')}.context_length", 0) or 0)
    tok = int(budget / bpt) if bpt else train
    return (min(tok, train) if train else tok), bpt, budget / 2**30


CHARS_PER_TOKEN = 3.0   # a prompt cut for a small-context model: at most this many characters per token is assumed (the densest tokenizers: code and English give 3-4.5), so the cut prompt always fits


def clamp_tokens(model, out_tokens, want, ctx=None):
    """tokens the prompt may have for this model: its trained context minus the answer and a margin (stock caps the context to it); None when the model holds `want` anyway or the header cannot be read"""
    ctx = ctx if ctx is not None else gguf_ctx_train(model)
    mx = ctx - int(out_tokens or 0) - 128 if ctx else 0
    return mx if ctx and mx < want else None


def hsize(n):
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


def htime(sec):
    return "--:--" if sec is None or sec > 99 * 3600 else f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}" if sec >= 3600 else f"{int(sec // 60)}:{int(sec % 60):02d}"


def url_size(url):
    """bytes of what a URL serves (the last, real answer after the redirects), 0 when unknown"""
    try: out = subprocess.run(["curl", "-sIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError): return 0
    v = [int(x) for x in re.findall(r"(?im)^content-length:\s*(\d+)", out)]
    return max(v) if v else 0


def dev_asset(b):
    """True when the fork has the dev release dev-BUILD with the asset BUILD.tar.gz (made by `run.py devpub BUILD`)"""
    r = subprocess.run(["gh", "release", "view", f"dev-{b}", "-R", "neurall/llama.cpp", "--json", "assets", "-q", ".assets[].name"], capture_output=True, text=True)
    return r.returncode == 0 and f"{b}.tar.gz" in r.stdout.split()


def cmd_devpub(a):
    """`run.py devpub BUILD`: publish a dev build as the prerelease dev-BUILD on the fork (not marked latest) for machines to download; the CUDA runtime is not in it (they fetch the one of the last release)"""
    b = a.build; src = os.path.realpath(os.path.join(DEVBUILDS, resolve_build(b)))
    tgz = os.path.join(tempfile.gettempdir(), f"{b}.tar.gz")
    subprocess.run(["tar", "--exclude=libcudart*", "--exclude=libcublas*", "-czf", tgz, "-C", src, "."], check=True)
    if dev_asset(b): rc = subprocess.run(["gh", "release", "upload", f"dev-{b}", tgz, "--clobber", "-R", "neurall/llama.cpp"]).returncode
    else: rc = subprocess.run(["gh", "release", "create", f"dev-{b}", tgz, "-R", "neurall/llama.cpp", "--prerelease", "--latest=false", "--title", f"dev {b} (not a release)", "--notes", "Dev build for test machines. Not a release."]).returncode
    os.unlink(tgz); return rc


def build_plan(m, b):
    """what a machine still needs for build b: ([download items], push or None). A download item {id, label, url, kind 'tar'|'file', dest, total}; a push {label, src, files, dst}. Official stock and released builds
    are downloaded by the machine itself (two archives: binaries and the CUDA runtime libraries); an unreleased build sends only its own files and downloads the CUDA libraries."""
    dst = f'{m["root"]}/{b}'
    present = "ok" in remote_sh(m, f"test -x {dst}/llama-server && echo ok", 30)
    items = []
    def tars(base, names, label):
        for n in names: items.append(dict(id=re.sub(r"\W", "_", f"{b}_{n}")[:60], label=f"{label} {n.split('-')[0]}", url=f"{base}/{n}", kind="tar", dest=dst, total=url_size(f"{base}/{n}")))
    st = re.fullmatch(r"stock-(b\d+)-official", b)
    if st and present: return [], None   # a release never changes: what is there is reused
    if st:
        t = st.group(1); tars(f"https://github.com/ggml-org/llama.cpp/releases/download/{t}", [f"llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz", f"cudart-llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz"], b)
        return items, None
    src = os.path.realpath(os.path.join(DEVBUILDS, resolve_build(b)))
    info = dict(l.strip().split("=", 1) for l in open(os.path.join(src, "BUILD_INFO")) if "=" in l) if os.path.exists(os.path.join(src, "BUILD_INFO")) else {}
    rel = info.get("release", "")
    gh = lambda *a: subprocess.run(["gh", *a], capture_output=True, text=True).stdout.split()
    if rel:
        tag = next((t for t in gh("release", "list", "-R", "neurall/llama.cpp", "--limit", "200", "--json", "tagName", "-q", ".[].tagName") if t.startswith("release-" + rel.split("-")[2] + "-" + rel.split("-")[3])), "")
        if tag:
            if present: return [], None
            tars(f"https://github.com/neurall/llama.cpp/releases/download/{tag}", [rel + ".tar.gz", "cudart-" + rel + ".tar.gz"], b)
            return items, None
    tag = (gh("release", "list", "-R", "neurall/llama.cpp", "-L", "1", "--exclude-pre-releases", "--json", "tagName", "-q", ".[0].tagName") or [""])[0]
    cud = next((a for a in gh("release", "view", tag, "-R", "neurall/llama.cpp", "--json", "assets", "-q", ".assets[].name") if a.startswith("cudart-llama-release") and "ubuntu" in a), "") if tag else ""
    # a dev build published with `run.py devpub BUILD` (prerelease dev-BUILD on the fork, the build without the CUDA runtime): the machine downloads it from there, the uplink here is not used
    if tag and cud and dev_asset(b):
        if not present: tars(f"https://github.com/neurall/llama.cpp/releases/download/dev-{b}", [b + ".tar.gz"], b); tars(f"https://github.com/neurall/llama.cpp/releases/download/{tag}", [cud], b)
        return items, None
    have = {}   # an unreleased build may have changed under the same name: only the files that are missing there or differ in size travel again, the CUDA libraries only when missing
    for ln in remote_sh(m, f"cd {dst} 2>/dev/null && stat -L -c '%n %s' * 2>/dev/null", 30).splitlines():
        if ln.count(" ") == 1 and ln.split()[1].isdigit(): have[ln.split()[0]] = int(ln.split()[1])
    if cud and not any(f.startswith("libcudart") for f in have): tars(f"https://github.com/neurall/llama.cpp/releases/download/{tag}", [cud], b)
    if not os.path.isdir(src): sys.exit(f"build {b}: no such build ({src}); builds here: {', '.join(sorted(os.listdir(os.path.dirname(src)))[-5:])}")
    files = sorted(f for f in os.listdir(src) if not (cud and f.startswith(("libcudart", "libcublas"))) and have.get(f) != os.path.getsize(os.path.join(src, f)))
    # a new build name starts with an empty folder, but most files are the same as in a build already there (the 159 MB CUDA library only changes with CUDA code): files that have the same name, size and
    # sha256 in another build folder of the machine are copied there instead of sent (the uplink can be 1 MB/s)
    if files:
        cand = {}
        for ln in remote_sh(m, f"stat -L -c '%n %s' {m['root']}/*/* 2>/dev/null", 60).splitlines():
            if ln.count(" ") == 1 and ln.split()[1].isdigit() and not ln.startswith(dst + "/"): cand.setdefault((os.path.basename(ln.split()[0]), int(ln.split()[1])), ln.split()[0])
        import hashlib
        same = []
        for f in files:
            c = cand.get((f, os.path.getsize(os.path.join(src, f))))
            if c and os.path.getsize(os.path.join(src, f)) > 1 << 20:
                h = hashlib.sha256(open(os.path.join(src, f), "rb").read()).hexdigest()
                if remote_sh(m, f"sha256sum {c} | cut -c1-64", 120).strip() == h: same.append((f, c))
        if same:
            remote_sh(m, f"mkdir -p {dst} && " + " && ".join(f"cp -p {c} {dst}/{f}" for f, c in same), 300)
            files = [f for f in files if f not in {x[0] for x in same}]
            print(f"{b}: {len(same)} files copied on {m['ssh']} from other builds, not sent"); PRELUDE[0] += 1
    if not files: return items, None
    return items, dict(label=f"{b} own files", src=src, files=files, dst=dst, total=sum(os.path.getsize(os.path.join(src, f)) for f in files) or 1)


def model_plan(m, model):
    """(items, path on the machine) for a model file the machine does not have: every shard downloaded there from the models.ini url; never uploaded"""
    base = os.path.basename(model); d = f'{m["root"]}/models'
    found = [l for l in remote_sh(m, f"find /m /home/o/m {d} /home/o /root -maxdepth 5 -name '{base}' 2>/dev/null | head -1", 120).strip().splitlines() if l.startswith("/")]
    if found:   # every shard has to be there, not just the first (a download that was stopped leaves the small first shard complete and the others as NAME.part)
        fd = os.path.dirname(found[0]); miss = [n for n in shard_names(base) if "ok" not in remote_sh(m, f"test -s {fd}/{n} && echo ok", 30)]
        if not miss: return [], found[0]
        d = fd
    sh = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", base)
    names = [f"{sh.group(1)}-{i:05d}-of-{sh.group(3)}.gguf" for i in range(1, int(sh.group(3)) + 1)] if sh else [base]
    url = next((v.get("url") for v in (fcfg.models_ini() or {}).values() if os.path.basename(v.get("path", "")) == base and v.get("url")), "")
    if not url:
        print(f"{base}: not on {m['ssh']} and models.ini has no url for it: add the Hugging Face url. Models are never uploaded from here.", flush=True); return [], ""
    pre = url.rsplit("/", 1)[0] + "/" if "/resolve/" in url else url.rstrip("/") + "/resolve/main/"
    last = url.rsplit("/", 1)[1] if "/resolve/" in url else base   # a direct file link may name the file differently (case); it is saved under our name
    sh2 = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", last)
    srcs = [f"{sh2.group(1)}-{i:05d}-of-{sh2.group(3)}.gguf" for i in range(1, int(sh2.group(3)) + 1)] if sh2 else [last]
    if len(srcs) != len(names): print(f"{base}: the url does not name the same number of shards", flush=True); return [], ""
    have_all = set(names) - set(miss) if found else set()
    return [dict(id=re.sub(r"\W", "_", n)[:60], label=n[:34], url=pre + sn, kind="file", dest=f"{d}/{n}", total=url_size(pre + sn)) for n, sn in zip(names, srcs) if n not in have_all], f"{d}/{base}"


def shard_names(base):
    sh = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", base)
    return [f"{sh.group(1)}-{i:05d}-of-{sh.group(3)}.gguf" for i in range(1, int(sh.group(3)) + 1)] if sh else [base]


def make_room(m, need, keep):
    """disk free on the machine against what the transfers need (+ 5 GB). Only when it is not enough, the biggest model files in <root>/models that this list does not use are deleted, biggest first"""
    gb = lambda n: f"{n / 2**30:.0f} GB"
    free = int(remote_sh(m, f"mkdir -p {m['root']} && df -B1 --output=avail {m['root']} | tail -1", 30).split()[-1])
    print(f"disk on {m['ssh']}: {gb(free)} free, the transfers need {gb(need)}", flush=True); PRELUDE[0] += 1
    if free >= need + 5 * 2**30 or keep is None: return
    for ln in sorted((l.split(" ", 1) for l in remote_sh(m, f"find {m['root']}/models -name '*.gguf' -printf '%s %p\\n' 2>/dev/null", 60).splitlines() if re.match(r"\d+ /", l)), key=lambda x: -int(x[0])):
        if free >= need + 5 * 2**30: break
        if os.path.basename(ln[1]) in keep: continue
        remote_sh(m, f"rm -f {ln[1]}", 60); free += int(ln[0]); print(f"  deleted {ln[1]} (not used by this list, {gb(int(ln[0]))}) -> {gb(free)} free", flush=True); PRELUDE[0] += 1
    if free < need: print(f"  still short of space: {gb(free)} free, {gb(need)} needed", flush=True)


def run_transfers(m, items, pushes, keep=None):
    """All downloads run on the machine at once (curl each, in the background there), the pushes run from here; one ssh streams the machine's file sizes once a second. A line per transfer (bar, done/total,
    speed, time left), the total last. Ctrl-C stops the remote curls."""
    if not items and not pushes: return True
    make_room(m, sum(it["total"] for it in items) + sum(p["total"] for p in pushes), keep)
    ssh = ["ssh", "-o", "ServerAliveInterval=30", m["ssh"]]
    D = f'{m["root"]}/.dl'
    jobs, polls = [], []
    awk_w = "'{a=$1*$2; if (t>0 && a>=t) a=t-1; print a}'"   # bytes written so far: allocated blocks of the sparse file, one short of the total until the transfer is finished
    for it in items:
        i = it["id"]; part = f'{D}/{i}.part' if it["kind"] == "tar" else it["dest"] + ".part"
        if it["kind"] == "tar":
            post = (f'T=$(mktemp -d {D}/x.XXXXXX) && tar xzf {D}/{i}.part -C $T && if [ $(ls $T | wc -l) -eq 1 ] && [ -d $T/$(ls $T) ]; then cp -a $T/$(ls $T)/. {it["dest"]}/; else cp -a $T/. {it["dest"]}/; fi; R=$?; rm -rf $T {D}/{i}.part; [ $R -eq 0 ]')
        else: post = f'mv {part} {it["dest"]}'
        jobs.append(f'( mkdir -p {it["dest"] if it["kind"] == "tar" else os.path.dirname(it["dest"])}; dl {it["url"]} {part} && ' + ("touch " + f"{D}/{i}.dl && " if it["kind"] == "tar" else "") + post + f' && touch {D}/{i}.fin || touch {D}/{i}.err ) &')
        polls.append(f'p={part}; [ -f {D}/{i}.err ] && echo "{i} -1" || {{ [ -f {D}/{i}.fin ] && echo "{i} {max(it["total"], 1)}" || {{ [ -f {D}/{i}.dl ] && echo "{i} {max(it["total"], 1)}" || echo "{i} $(stat -c "%b %B %s" $p 2>/dev/null | awk -v t={it["total"]} {awk_w} | grep . || echo 0)"; }}; }}')
    # dl URL OUT: aria2c with 16 connections per file (a CDN limits each connection: one curl got 46 MB/s, 16 connections ~150+ MB/s per file), resumable through its .aria2 control file;
    # installed on the machine when it is missing (root, or sudo without a password), plain resumable curl when that fails
    dlfn = ('command -v aria2c >/dev/null || { S=""; [ "$(id -u)" = 0 ] || S="sudo -n"; ($S apt-get install -y aria2 || ($S apt-get update -qq && $S apt-get install -y aria2)) >/dev/null 2>&1; }\n'
            'dl(){ if command -v aria2c >/dev/null; then aria2c -x16 -s16 -k16M -q -c --file-allocation=none --allow-overwrite=true -d "$(dirname "$2")" -o "$(basename "$2")" "$1"; '
            'else curl -fL -sS -C - -o "$2" "$1"; fi; }\n')
    sched = dlfn + f'mkdir -p {D}; rm -f {D}/*.fin {D}/*.err {D}/*.dl; ' + "\n".join(jobs) + "\nwait"
    if items: subprocess.run(ssh + [f"cat > {D}.sh"], input=f"mkdir -p {D}\n" + sched, text=True) if False else None
    def start_dl():
        subprocess.run(ssh + [f"mkdir -p {D} && cat > {D}/start.sh"], input=sched + "\n", text=True, check=True)
        remote_sh(m, f"cd {D} && (setsid nohup bash start.sh > start.out 2>&1 < /dev/null &)", 30)
    # downloads and a push into the same machine share its inbound link: a full link drops the push's packets (50 KB/s instead of 400 on a lossy path), so with a push the downloads start after it
    if items and not pushes: start_dl()
    poller = subprocess.Popen(ssh + ["while true; do " + "; ".join(polls) + '; echo "--"; sleep 1; done'], stdout=subprocess.PIPE, text=True) if items else None
    size = {it["id"]: 0 for it in items}; failed = set()
    import threading
    def read():
        cur = {}
        for ln in poller.stdout:
            ln = ln.strip()
            if ln == "--": size.update(cur); cur = {}
            elif ln.count(" ") == 1:
                k, v = ln.split()
                try: cur[k] = int(v)
                except ValueError: pass
    if poller: threading.Thread(target=read, daemon=True).start()
    pstate = {id(p): dict(done=0, sent=0, fin=False, err=None) for p in pushes}
    PUSH_STREAMS, PIECE = 16, 8 << 20
    def do_push(p):
        """files in PIECE-sized pieces over PUSH_STREAMS ssh connections at once (a lossy long path gives each TCP stream ~200 KB/s, the streams add up); every file is written as NAME.part and renamed when all its pieces
        are there, so a push that dies leaves no full-size half file for the next run to take as complete; symlinked / identical names are sent once and linked"""
        import zlib, queue
        st = pstate[id(p)]; lock = threading.Lock()
        try:
            src, dst = p["src"], p["dst"]
            uniq, links = {}, []
            for f in p["files"]:
                rp = os.path.realpath(os.path.join(src, f))
                if rp in uniq: links.append((f, uniq[rp]))
                else: uniq[rp] = f
            prep = [f"mkdir -p {dst}"]
            for rp, f in uniq.items(): prep.append(f"truncate -s {os.path.getsize(rp)} {dst}/{f}.part && chmod {os.stat(rp).st_mode & 0o777:o} {dst}/{f}.part")
            remote_sh(m, " && ".join(prep), 120)
            q = queue.Queue(); left = {}
            for rp, f in uniq.items():
                n = os.path.getsize(rp); left[f] = max(1, -(-n // PIECE))
                for off in range(0, max(n, 1), PIECE): q.put((rp, f, off, min(PIECE, n - off)))
            def worker():
                while True:
                    try: rp, f, off, ln = q.get_nowait()
                    except queue.Empty: return
                    for attempt in range(4):
                        try:
                            with open(rp, "rb") as fh: fh.seek(off); raw = fh.read(ln)
                            z = zlib.compressobj(1, zlib.DEFLATED, 31); out = z.compress(raw) + z.flush()
                            rx = subprocess.run(ssh + [f"gunzip | dd of={dst}/{f}.part bs=4M oflag=seek_bytes seek={off} conv=notrunc status=none"], input=out, capture_output=True)
                            if rx.returncode == 0: break
                        except Exception: pass
                    else:
                        st["err"] = f"piece {f}@{off} failed"; return
                    with lock:
                        st["done"] += ln; st["sent"] += len(out); left[f] -= 1
            ws = [threading.Thread(target=worker, daemon=True) for _ in range(PUSH_STREAMS)]
            for w in ws: w.start()
            for w in ws: w.join()
            if not st["err"]:
                fin = [f"mv {dst}/{f}.part {dst}/{f}" for f in uniq.values()] + [f"ln -sf {t} {dst}/{f}" for f, t in links]
                remote_sh(m, " && ".join(fin), 120)
                st["done"] = p["total"]
        except Exception as e: st["err"] = str(e)
        st["fin"] = True
    push_threads = [threading.Thread(target=do_push, args=(p,), daemon=True) for p in pushes]
    for t in push_threads: t.start()
    if items and pushes:
        def after_push():
            for t in push_threads: t.join()
            start_dl()
        threading.Thread(target=after_push, daemon=True).start()
    rows = [dict(label=it["label"], total=it["total"], get=(lambda it=it: size[it["id"]]), dead=(lambda it=it: size[it["id"]] < 0), fin=(lambda it=it: it["total"] > 0 and size[it["id"]] >= it["total"] or size[it["id"]] >= max(it["total"], 1) and it["total"] == 0)) for it in items]
    for p in pushes:
        st = pstate[id(p)]; rows.append(dict(label=p["label"], total=p["total"], get=(lambda st=st: st["done"]), dead=(lambda st=st: bool(st["err"])), fin=(lambda st=st: st["fin"] and not st["err"]), wire=(lambda st=st: st["sent"])))
    hist = {id(r): [] for r in rows}; t0 = time.time(); last_plain = 0
    def line(r, now):
        d = max(r["get"](), 0); tot = r["total"] or max(d, 1); f = min(d / tot, 1.0)
        h = hist[id(r)]; h.append((now, d)); h[:] = [x for x in h if now - x[0] <= 6]
        sp = (h[-1][1] - h[0][1]) / max(h[-1][0] - h[0][0], 1e-3) if len(h) > 1 else 0
        eta = (tot - d) / sp if sp > 1 else None
        return f, d, tot, sp, eta
    try:
        while True:
            now = time.time(); out = []; sd = st_ = sp_ = 0; allfin = True; anyerr = False; etas = []
            for r in rows:
                f, d, tot, sp, eta = line(r, now); fin = r["fin"](); dead = r["dead"]()
                allfin &= fin or dead; anyerr |= dead
                if not fin and not dead: etas.append(eta)
                sd += d if not fin else tot; st_ += tot; sp_ += sp if not fin else 0
                bar = "█" * int(20 * f) + "░" * (20 - int(20 * f))
                out.append(f"{col(36, r['label'][:34]):<44} [{col(32 if fin else 33, bar)}] {f * 100:3.0f}%  {hsize(d):>9}/{hsize(tot):<9}  " + (col(31, "FAILED") if dead else col(32, "done") if fin else f"{col(94, hsize(sp) + '/s'):<20} eta {htime(eta)}"))
            f = sd / max(st_, 1); eta = None if (not etas or None in etas) else max(etas)   # the transfers run side by side: done when the slowest is, not when the summed rate would empty the summed bytes
            out.append(col(1, f"{'total':<34}") + f"   [{col(32, '█' * int(20 * f) + '░' * (20 - int(20 * f)))}] {f * 100:3.0f}%  {hsize(sd):>9}/{hsize(st_):<9}  {col(94, hsize(sp_) + '/s'):<20} eta {htime(eta)}")
            if live_tty(): live_clear(); print(fit_block("\n".join(out)), flush=True); LIVE_UP[0] = len(out)
            elif now - last_plain > 10: last_plain = now; print(out[-1], flush=True)
            if allfin: break
            time.sleep(1)
    except KeyboardInterrupt:
        remote_sh(m, f'pkill -f "[c]url -fL -sS -C - -o {D}"; pkill -f "[c]url -fL -sS -C - -o {m["root"]}"', 30)
        raise
    finally:
        if poller: poller.terminate()
    PRELUDE[1] = LIVE_UP[0]; LIVE_UP[0] = 0
    err = [r["label"] for r in rows if r["dead"]()] + [p["label"] for p in pushes if pstate[id(p)]["err"]]
    if err: print(f"transfer failed: {', '.join(err)}", flush=True)
    return not err


def push_build(m, b):
    items, push = build_plan(m, b)
    if not run_transfers(m, items, [push] if push else []): sys.exit(f"{b}: the transfer to {m['ssh']} failed")


def push_model(m, model):
    items, path = model_plan(m, model)
    return path if run_transfers(m, items, []) and path else ""


def gpu_speed(token):
    """relative speed of a card (1.0 = the reference), from the optional [gpu_speed] table of the fork config: keys are matched as text inside the card's name ('5090 = 2.0', '5070 = 1.0'); unknown cards count 1.0.
    Used to rank cards for -ng and to scale the expected time of a cell measured on another GPU class."""
    t = (token or "").lower()
    try: tab = fcfg._FILE.get("gpu_speed", {})
    except Exception: tab = {}
    for k, v in sorted(tab.items(), key=lambda kv: -len(kv[0])):   # the longest key that fits wins ('5070 ti' before '5070')
        if k.lower() in t:
            try: return float(v)
            except (TypeError, ValueError): pass
    return 1.0


NGPU = [int(os.environ["RUN_NG"]) if os.environ.get("RUN_NG", "").lstrip("-").isdigit() else None]   # (env RUN_NG / RUN_NS carry the switches to the rcell child processes)
# -ng N: use only N cards (None = all), the best N by link then speed, the fastest last; 0 = CPU only


def pick_gpus(n, smi_out):
    """-ng N: the N best cards from `nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,memory.total,name` (csv, no header): best = the fastest slot (link generation x
    width), then the faster card (max clock, then memory); the chosen ones are ordered slowest first and the FASTEST LAST (it is the card the prompt passes are offloaded to). Returns the
    CUDA_VISIBLE_DEVICES list (nvidia-smi / PCI bus order indices, with CUDA_DEVICE_ORDER=PCI_BUS_ID) - empty for n = 0 (CPU only)."""
    g = []
    for ln in smi_out.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        try: g.append((int(f[0]), int(f[1]) * int(f[2]), int(f[3]), int(f[4]), gpu_speed(f[5]) if len(f) > 5 else 1.0))   # index, slot score, max sm clock, memory MiB, speed factor
        except (ValueError, IndexError): continue
    best = sorted(g, key=lambda x: (x[1], x[4], x[2], x[3]), reverse=True)[:max(0, n)]   # slot first, then the known speed of the card, then its clock
    return ",".join(str(x[0]) for x in sorted(best, key=lambda x: (x[4], x[2], x[3], x[1])))


GPU_Q_NG = "nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,memory.total,name --format=csv,noheader,nounits"


def cmd_remote(a):
    """remote ID push BUILD...   |   remote ID test BUILD --model PATH_ON_THE_MACHINE [--req FILE] [--args '...'] [-n 2]
    A server test on a machine reached over ssh (a rented box). push: the build directories go there (rsync, else tar over ssh). test: the server AND the request both run on
    that machine (curl to its own localhost), so no ssh latency is in the numbers; ssh only starts a detached script and tails what it writes: server.log and run.log (the
    requests) and stats.log (the server's /slots once a second: prompt tokens done, tokens decoded). The speeds come from the timings in the answers (as the harness does:
    the same request twice on one server); the row goes into the run history with hw = the machine id."""
    global MODEL, CAMPAIGN
    m = machine(a.id)
    if m["win"]:
        sys.exit("remote: linux machines only")
    if getattr(a, "campaign", None): CAMPAIGN = a.campaign
    ssh = ["ssh", "-o", "ServerAliveInterval=30", m["ssh"]]
    if a.what == "push":
        for b in a.builds: push_build(m, b)
        return
    build, MODEL = a.builds[0], a.model
    exe = f'{m["root"]}/{build}/llama-server'
    if "ok" not in remote_sh(m, f"test -x {exe} && echo ok", 30):
        sys.exit(f"remote: {exe} not found on {a.id} (run.py remote {a.id} push {build})")
    req = os.path.abspath(a.req or os.path.join(PROMPT_DIR, "req-12k-1k.json"))
    run = f'{m["root"]}/remote-runs/{time.strftime("%Y%m%d-%H%M%S")}-{build}'
    remote_sh(m, f"mkdir -p {run}", 30)
    rq = json.load(open(req))
    if os.environ.get("RUN_ANTILOOP"): rq.update(ANTILOOP)
    if getattr(a, "tweak", 0):   # the loop retry (as locally): newlines added to the prompt, a greedy rerun of the same prompt loops again
        msg = rq["messages"][-1]
        if isinstance(msg["content"], str): msg["content"] += "\n" * a.tweak
        else: msg["content"][0]["text"] += "\n" * a.tweak
    if getattr(a, "clamp", 0):   # a model with a small trained context: the user text is cut in memory before it is sent, no second request file
        msg = rq["messages"][-1]; txt = msg["content"] if isinstance(msg["content"], str) else msg["content"][0]["text"]
        cut = txt[:int(a.clamp * CHARS_PER_TOKEN)]
        if isinstance(msg["content"], str): msg["content"] = cut
        else: msg["content"][0]["text"] = cut
    body = json.dumps(rq); sha = hashlib.sha1(body.encode()).hexdigest()[:16]   # the final request (clamped, tweaked, antiloop): kept on the machine by content, uploaded once, reused by every later cell
    pf = f'{m["root"]}/prompts/{sha}.json'
    if "ok" not in remote_sh(m, f"test -s {pf} && echo ok", 30):
        remote_sh(m, f'mkdir -p {m["root"]}/prompts', 30)
        subprocess.run(ssh + [f"cat > {pf}.part && mv {pf}.part {pf}"], input=body, text=True, check=True)
        print(f"prompt {sha} uploaded to {a.id} ({len(body) // 1024} KB)", flush=True)
    remote_sh(m, f"ln -sf {pf} {run}/req.json", 30)
    ng_env = ""; ng_args = a.args
    if NGPU[0] is not None:   # -ng N: which cards the server may see on this machine (read-only query), the fastest last; 0 = none and no offload
        vis = pick_gpus(NGPU[0], remote_sh(m, GPU_Q_NG, 30)) if NGPU[0] > 0 else ""
        ng_env = f"export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='{vis}'"
        if NGPU[0] == 0: ng_args = f"{a.args} -dev none -ngl 0"
        print(f"-ng {NGPU[0]}: cards {vis or 'none (CPU only)'} on {a.id}" + (" (fastest last)" if vis else ""), flush=True)
    script = rf"""cd {run}; echo $$ > run.pid; echo "start $(date +%s.%N)" >> run.log; export LD_LIBRARY_PATH={m["root"]}/{build}; {ng_env}
command -v supervisorctl >/dev/null && supervisorctl stop llama >/dev/null 2>&1; pkill -x llama-server; for i in $(seq 1 30); do pgrep -x llama-server >/dev/null || break; sleep 1; done; pkill -9 -x llama-server   # every old server on the machine (a stopped or crashed run, or the rented image's own `llama` service, stopped through supervisorctl because supervisord restarts a killed one) goes before this one starts: it holds the GPU memory and the pinned pool; -x is the exact process name, no match on this script
NS={1 if NOSTATS[0] else 0}
[ $NS = 1 ] || ({SYS_SH}) > sys.log 2>/dev/null & SYSM=$!
[ $NS = 1 ] || nvidia-smi dmon -s t -d 1 > dmon.log 2>/dev/null & DMON=$!; echo "dmon $(date +%s.%N)" >> run.log   # PCIe rx/tx MB/s per GPU, one line a second: the link use of the whole run
P=$(( 20000 + RANDOM % 20000 )); while curl -sf localhost:$P/health >/dev/null 2>&1; do P=$(( 20000 + RANDOM % 20000 )); done; echo $P > port   # a free port: an old server (a stopped run) must not answer for this one
trap 'kill $SRV $DMON $SYSM 2>/dev/null' EXIT
nohup {exe} -m {a.model} --port $P {ng_args} > server.log 2>&1 &
SRV=$!; echo $SRV > pid
touch alive; ( while sleep 10 && kill -0 $$ 2>/dev/null; do [ $(( $(date +%s) - $(stat -c %Y alive) )) -gt 45 ] && {{ kill -9 $SRV; kill -9 $$; exit; }}; done ) &   # run.py touches `alive` every poll: if it was killed (no cleanup ran), the machine does not keep a server running
for i in $(seq 1 3600); do curl -sf localhost:$P/health >/dev/null && break; kill -0 $SRV 2>/dev/null || {{ echo SERVER_DIED >> run.log; echo "script done" >> run.log; touch done; exit 1; }}; sleep 1; done
echo "server ready $(date +%s.%N)" >> run.log
[ $NS = 1 ] || ( while kill -0 $SRV 2>/dev/null; do echo "$(date +%s.%N) $(curl -s --max-time 4 localhost:$P/slots | grep -o '"n_prompt_tokens[a-z_]*":[0-9]*\|"n_decoded":[0-9]*' | tr '\n' ' ')" >> stats.log; sleep 1; done ) &
for r in $(seq 1 {a.n}); do
  curl -s --max-time 7200 localhost:$P{a.path} -H 'Content-Type: application/json' -d @req.json > resp-$r.json
  echo "request $r done $(date +%s.%N): $(grep -o '"prompt_per_second":[0-9.]*\|"predicted_per_second":[0-9.]*' resp-$r.json | tr '\n' ' ')" >> run.log
done
curl -s --max-time 3 localhost:$P/moe/stats > final-stats.json 2>/dev/null   # the engine's state at the end of the last request: a short cell is over before a status tick sees it
kill $SRV $DMON $SYSM; sleep 2; echo "script done" >> run.log; touch done
"""
    subprocess.run(ssh + [f"cat > {run}/run.sh"], input=script, text=True, check=True)
    remote_sh(m, f"cd {run} && touch server.log run.log stats.log && (setsid nohup bash run.sh > run.out 2>&1 < /dev/null &)", 30)   # the subshell: ssh returns at once, nothing keeps its channel open
    files = f"{run}/run.log" + ("" if getattr(a, "tail", "all") == "run" else f" {run}/server.log") + (f" {run}/stats.log" if a.stats else "")   # tail="run" (the list runner): the request lines only
    quiet = getattr(a, "tail", "all") == "run"   # a list cell (rcell): run_watched draws it from the two files below, as for a local cell
    if quiet:   # the second tailed file: the server's /slots progress, one line a second, read by slots_progress like the local server's answer
        tail = subprocess.Popen(ssh + [f"tail -n 0 -F {run}/stats.log"], stdout=open(a.slots, "w"))
        tail2 = subprocess.Popen(ssh + [f"tail -n +1 -F {run}/run.log"], stdout=open(a.results + ".log", "w"))   # the second persistent ssh: the requests' lines; no per-poll ssh for data
    else: tail = tail2 = subprocess.Popen(ssh + [f"tail -n +1 -F {files}"])   # what the machine writes, live; the run goes on if this ssh dies
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # a stopped run.py stops the run on the machine too (below)
    done_r = 0; t_start = time.time()
    def feed():   # the finished requests of run.log as the lines a local cell writes to its .run.out (run_watched shows them), the load time to LOAD_FILE
        nonlocal done_r
        log = open(a.results + ".log", errors="replace").read()
        ts = lambda k: (re.search(k + r" ([\d.]+)", log) or [0, None])[1]
        if ts("server ready") and ts("start") and not os.path.exists(LOAD_FILE):   # once: its mtime is the end of the load (the prompt phase is timed from it); rewriting it every poll kept "now" as the load end, so the estimate blew up (1691 min) and the progress ran backwards
            open(LOAD_FILE, "w").write(str(float(ts("server ready")) - float(ts("start"))))
        for mm in list(re.finditer(r"request (\d+) done [\d.]+: .*?\"prompt_per_second\":([\d.]+).*?\"predicted_per_second\":([\d.]+)", log))[done_r:]:
            open(a.results, "a").write(f"{time.strftime('%F %T')} {build} {a.test} {{}} {mm.group(3)} t/s hit 0.0% pp {mm.group(2)} t/s\n"); done_r += 1
    try:
        if quiet:
            n = 0
            while "script done" not in open(a.results + ".log", errors="replace").read():
                feed(); time.sleep(2); n += 1
                if n % 5 == 0: remote_sh(m, f"touch {run}/alive", 30)   # the only polling ssh: the keepalive the machine's watchdog looks for
            feed()
        else:
            while "ok" not in remote_sh(m, f"touch {run}/alive; test -f {run}/done && echo ok", 60):
                time.sleep(5)
            time.sleep(2)
    finally:
        tail.terminate(); tail2.terminate()
        remote_sh(m, f"test -f {run}/done || kill -9 $(cat {run}/run.pid) $(cat {run}/pid) 2>/dev/null", 60)   # the script first (its loop would restart nothing, but its stats poller dies with the server)
    if quiet:   # the server's log next to the other cells' (hardware probe lines, load, errors)
        subprocess.run(["scp", "-q", f'{m["ssh"]}:{run}/server.log', os.path.join(QDIR, f"{a.campaign}.remote.server.log")])
        for _f in ("dmon", "sys"): subprocess.run(["scp", "-q", f'{m["ssh"]}:{run}/{_f}.log', os.path.join(QDIR, f"{a.campaign}.remote.{_f}.log")])
        subprocess.run(["scp", "-q", f'{m["ssh"]}:{run}/final-stats.json', os.path.join(QDIR, "final-stats.json")])
    rows = []
    for r in range(1, a.n + 1):
        try:
            d = json.loads(remote_sh(m, f"cat {run}/resp-{r}.json", 60))
        except ValueError:
            print(f"request {r}: no valid answer, see {m['ssh']}:{run}/", flush=True)
            continue
        t = d.get("timings") or {}
        err = (d.get("error") or {}).get("message") if isinstance(d.get("error"), dict) else None
        _m = (d.get("choices") or [{}])[0].get("message") or {}   # a thinking model cut at the token limit has only reasoning_content: stored and loop-checked like a local run (before, such an answer was empty here and never stored)
        txt = ((_m.get("reasoning_content") + "\n---\n" if _m.get("reasoning_content") else "") + (_m.get("content") or "")) or (d.get("choices") or [{}])[0].get("text") or d.get("content") or ""
        w = txt.split()
        g4 = {tuple(w[i:i + 4]) for i in range(len(w) - 3)}
        loop = len(w) > 100 and len(g4) / max(1, len(w) - 3) < 0.5   # distinct word 4-grams: a looping answer repeats itself, its speed is fake
        _el = (t.get("prompt_ms") or 0) / 1000 + (t.get("predicted_ms") or 0) / 1000; _now = time.time()   # remote rows carry start/end like local ones: -nrt and the stock lookups read start_ts
        rows.append(dict(start_ts=time.strftime("%F %T", time.localtime(_now - _el)), end_ts=time.strftime("%F %T", time.localtime(_now)), wall_s=_el,
                         tps=t.get("predicted_per_second"), pp_tps=t.get("prompt_per_second"), n_gen=t.get("predicted_n"), output=txt, ok=0 if loop or not t else 1, args=a.args,
                         note=f"remote {m['ssh']}:{run}" + (" LOOP" if loop else "") + (f" {err}" if err else "")))
        if err: print(f"request {r}: server error: {err}")
        print(f"request {r}: prompt {t.get('prompt_n')} tokens {t.get('prompt_per_second', 0):.1f} t/s | decode {t.get('predicted_n')} tokens {t.get('predicted_per_second', 0):.2f} t/s"
              f" | {len(w)} words, distinct 4-grams {len(g4)}{' LOOP' if loop else ''}")
    try:   # the cell's load time from the server's own clock (as a local cell) and its total (load + the requests), stored so -nrt and later tables show them again
        _lm = re.search(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) \w+ srv\s+llama_server: model loaded", open(os.path.join(QDIR, f"{a.campaign}.remote.server.log"), errors="replace").read(), re.M)
        if _lm and rows:
            rows[0]["load_s"] = round(int(_lm.group(1)) * 60 + int(_lm.group(2)) + int(_lm.group(3)) / 1000, 1)
            _tot = rows[0]["load_s"] + sum(y["wall_s"] for y in rows)
            for _x in rows: _x["wall_s"] = _tot
    except OSError: pass
    if a.record:
        vstr = remote_sh(m, f"LD_LIBRARY_PATH={m['root']}/{build} {exe} --version 2>&1", 60)
        mv = re.search(r"version: (.*)", vstr)
        loops = any("LOOP" in x["note"] for x in rows)
        HOST[0] = m; hwl = hw_line()   # the machine line and the live power limits of the remote host (read once per cell, they drift on a rented host)
        pls = "+".join(str(int(float(x))) for x in remote_sh(m, "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null", 30).split() if re.fullmatch(r"[\d.]+", x))
        _rd = lambda f: open(os.path.join(QDIR, f"{a.campaign}.remote.{f}.log"), errors="replace").read() if os.path.exists(os.path.join(QDIR, f"{a.campaign}.remote.{f}.log")) else ""
        _rl = _rd("server"); _t0r = float((re.search(r"server ready ([\d.]+)", open(a.results + ".log", errors="replace").read()) or [0, 0])[1])   # the window after the load
        _sysc = sys_summary(_rd("sys"), _t0r); _sysc["pcie_rx_gbs"] = dmon_summary(_rd("dmon"))
        for row in rows:
            row.update(_sysc); row["pl"] = pls; row["hwline"] = hwl
            store(row, build, getattr(a, "test", None) or "srv-" + os.path.basename(req).replace("req-", "").replace(".json", ""), {"bare": "1", **({"args": a.args} if a.args else {})}, row.pop("note"),
                  a.id.split("-")[0], mv.group(1).strip() if mv else "?")
    if quiet: sys.exit(0 if rows and all(x["ok"] for x in rows) else 3 if loops else 1)   # the exit code run_watched reads, as for a local cell
    print(f"files on the machine: {m['ssh']}:{run}")


CLI_EXE = "llama-cli"   # --cli-exe llama-completion: the raw-prompt tool (llama-cli only has conversation mode)


def remote_exe(m, build):
    sep = "\\" if m["win"] else "/"
    return f'{m["root"]}{sep}{build}{sep}{CLI_EXE}' + (".exe" if m["win"] else "")


def remote_sh(m, cmd, timeout=3600):
    argv = ["bash", "-c", cmd] if m["ssh"] == "local" else ["ssh", m["ssh"], cmd]   # ssh=local: llama-cli on this host (the CLI one-shot case)
    kw = {"preexec_fn": _die_with_parent} if m["ssh"] == "local" else {}   # a local child (llama-cli) dies with run.py instead of running on as an orphan
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
    return r.stdout + r.stderr


def push_prompt(m, name, text):
    """a prompt file on the remote machine (scp to a scratch directory), returns its path there"""
    loc = os.path.join("/tmp", name)
    open(loc, "w").write(text)
    if m["win"]:
        subprocess.run(["scp", "-q", loc, f'{m["ssh"]}:C:/Users/o/{name}'], check=True)
        return "C:\\Users\\o\\" + name
    if m["ssh"] != "local":   # ssh=local (pc1-cli, pc1-gh): the file is already where llama-cli runs
        subprocess.run(["scp", "-q", loc, f'{m["ssh"]}:/tmp/{name}'], check=True)
    return "/tmp/" + name


def remote_run(m, build, test, args, env=None):
    """The CLI version of a test (llama-cli, one prompt, greedy): what most people run. Returns a row like run_one."""
    exe, win = remote_exe(m, build), m["win"]
    if not build.startswith("stock") and "--load-mode" not in args and "-lm" not in args:
        args = f"{args} --load-mode pin".strip()   # pinned weights for every measured run of our builds, the CLI too
    envp = "".join((f"set {k}={v}&& " if win else f"{k}={v} ") for k, v in (env or {}).items())  # variant environment (env: words)
    if test in ("ppl", "ppl3", "pplb", "pplc"):  # the precise decode comparator on that machine: llama-perplexity, one token per call, longsrc.cpp (copy it to <root>\\<build>\\longsrc.cpp)
        sep = "\\" if win else "/"
        nb, nub = {"ppl": ("1", "1"), "ppl3": ("3", "3"), "pplb": ("1024", "1024"), "pplc": ("1024", "256")}[test]   # pplb: 1024-token batches = the big-batch (prefill) path; pplc: the same batch in 256-token ubatches (4 ubatches per decode call)
        stf = f'{m["root"]}{sep}perf-stats.txt'
        envs = (f'set LLAMA_MOE_STATE=0&& set LLAMA_MOE_CACHE_TUNED=0&& set LLAMA_MOE_CACHE_STATS=1&& set LLAMA_MOE_CACHE_STATSLOG=32&& set LLAMA_MOE_STATSLOG={stf}&& ' if win else
                f'LD_LIBRARY_PATH={os.path.dirname(exe)} LLAMA_MOE_STATE=0 LLAMA_MOE_CACHE_TUNED=0 LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_STATSLOG=32 LLAMA_MOE_STATSLOG={stf} ')   # native Linux builds bake their build dir into RUNPATH
        cmd = envp + envs + f'{exe.replace("llama-cli", "llama-perplexity")} -m {MODEL} -f {m["root"]}{sep}{build}{sep}{os.environ.get('PERF_PPL_FILE', 'longsrc.cpp')} -c 1024 --chunks {os.environ.get('PERF_PPL_CHUNKS', '2')} -b {nb} -ub {nub} {args} ' + ("< nul 2>&1" if win else "< /dev/null 2>&1")
        out = remote_sh(m, cmd)
        sp_ = re.search(r"([\d.]+) seconds per pass", out)
        pp_ = re.findall(r"PPL = ([\d.]+)", out)
        row = {"s_per_pass": float(sp_.group(1)), "tps": 1024 / float(sp_.group(1)), "ppl": float(pp_[-1]) if pp_ else None, "ok": 1, "args": args} if sp_ else {"ok": 0, "note": out[-200:], "args": args}
        row.update(cache_stats(out))
        st = remote_sh(m, ("type " if win else "cat ") + stf + (" 2>nul" if win else " 2>/dev/null"))
        if st.strip():  # the run's statslog goes to stats.ini like a local run's
            open(f"/tmp/perf-{build}{LOGTAG}-stats.txt", "w").write(st)
        return row
    prompt = ovr_prompt(config.prompt("tetris") if test == "tetris" else GAME_TEMPLATE.format(word="tetris")).replace('"', "'")
    n = int(OVR.get("tokens") or (-1 if test == "tetris" else 100))
    pfile = None   # long1k (1024 in, 1024 out) and pf12k (12k in, --tokens 1 out): the prompt goes to the machine as a file
    if test in ("long1k", "pf12k", "in1k"):   # in1k: the long1k prompt (1k in) with ONE token out = prefill only; the decode-only partner is the plain tetris test with --tokens 1024 (short prompt, 1k out)
        text = (open(config.prompt_path("src1k")).read() + "\n\n" + LONG2K_INSTR) if test in ("long1k", "in1k") else OVR.get("prefix", "") + open(config.prompt_path("src12k")).read()
        pfile = push_prompt(m, "bw-prompt-" + test + ".txt", text)
        n = int(OVR.get("tokens") or (1024 if test == "long1k" else 1))
    _raw = CLI_EXE == "llama-completion" and not win   # text on stdout, logs and timings on stderr
    ctx = OVR.get("ctx") or (4096 if test in ("long1k", "in1k") else 16384 if test == "pf12k" else 1024)
    cmd = envp + f'{exe} -m {MODEL} -c {ctx} --temp 0 -n {n} --no-display-prompt ' + (f'-f {pfile} -b 2048 -ub 2048 ' if pfile else f'-p "{prompt}" ') + f'{args} ' \
          + ("< nul 2>&1" if win else ("< /dev/null 2>/tmp/perf-cli.err" if _raw else "< /dev/null 2>&1"))
    if not win:
        cmd = f'LD_LIBRARY_PATH={os.path.dirname(exe)} ' + cmd
    _t0 = time.time()
    out = remote_sh(m, cmd)
    _wall = round(time.time() - _t0, 1)
    err = ""
    if _raw:
        err = remote_sh(m, "cat /tmp/perf-cli.err")
        pe = re.findall(r"prompt eval time =.*?([\d.]+) tokens per second", err)
        ge = re.findall(r"\n[^\n]*common_perf_print:\s+eval time =.*?([\d.]+) tokens per second", err)
        t = [(pe[-1], ge[-1])] if pe and ge else []
    else:
        t = re.findall(r"Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s", out)
    mh = re.findall(r"hit ([\d.]+)% this reply", out)
    _load = re.findall(r"load time =\s*([\d.]+) ms", err or out)
    bad = re.search(r"(send_error|exceeds the available context size|error: request|failed to load model|out of memory)", out)   # the CLI prints its summary line even for a request it refused
    row = {"tps": float(t[-1][1]), "pp_tps": float(t[-1][0]), "ok": 1, "args": args} if t and not bad else {"ok": 0, "note": ("REFUSED: " + bad.group(0) + " | " if bad else "") + out[-200:], "args": args}
    if t:   # the generated text (everything before the closing "[ Prompt: ... ]" line), kept for the loop and first-token checks
        row["output"] = (out.split("> EOF by user")[0].strip("\n ") if _raw else out.split("[ Prompt:")[0].strip("\n ") if "[ Prompt:" in out else out)
        row["md5"] = hashlib.md5(row["output"].encode()).hexdigest()[:8]
        row["wall_s"] = _wall
        row["endpoint"] = "cli"
        row["start_ts"] = time.strftime("%F %T", time.localtime(_t0))
        row["end_ts"] = time.strftime("%F %T", time.localtime(_t0 + _wall))
        if _load:
            row["load_s"] = round(float(_load[-1]) / 1000, 1)
    if mh:
        row["hit_rate"] = float(mh[-1])
    return row


def remote_version(m, build):
    mv = re.search(r"version: (.*)", remote_sh(m, remote_exe(m, build) + " --version" + (" 2>&1" if m["win"] else " 2>&1"), 60))
    return mv.group(1).strip() if mv else "?"


def env_rec(env):
    return dict(env or {})


def remote_cell(mid, m, build, test, args, note, warm=True, env=None):
    """Throwaway + measured run on another machine, recorded in the same db (hw = the machine's id)."""
    global IDX
    if m["win"] and remote_sh(m, 'powershell -c "(Get-CimInstance Win32_Battery).BatteryStatus"', 60).strip() not in ("2", ""):
        sys.exit(f"{mid} is not on AC power: the battery caps the GPU and results would be invalid")
    if warm:
        remote_run(m, build, test, args, env)
    row = remote_run(m, build, test, args, env)
    rec = {"bare": "1", **env_rec(env), **({"args": args} if args else {}), **{k: str(v) for k, v in OVR.items() if k != "prompt"}}
    store(row, build, "cli-" + test, rec, note, mid.split("-")[0], remote_version(m, build))   # pc1-cli rows are hw pc1
    show_row(row, build, "cli-" + test)
    IDX += 1


def split_variant(build, vargs):
    """A variant has server args, env:KEY=VALUE words, and moe:key=value words (fork settings): newer builds read settings only as --moe keys, the first fork releases only as
    LLAMA_MOE_CACHE_<KEY> environment variables. A moe: word goes the way the build reads it, so one variant pins both kinds of build (autotune=0 -> LLAMA_AUTOTUNE=0)."""
    words = argv_of(vargs)
    args = shlex.join(w for w in words if not (w.startswith("env:") or w.startswith("moe:")))
    env = dict(w[4:].split("=", 1) for w in words if w.startswith("env:"))
    pins = [w[4:] for w in words if w.startswith("moe:")]
    if pins and not build.startswith("stock"):
        if build_has_moe_flag(build, env):
            args = (args + " " if args else "") + "--moe " + ",".join(pins)
        else:
            for pin in pins:
                k, v = pin.split("=", 1)
                env["LLAMA_AUTOTUNE" if k == "autotune" else "LLAMA_MOE_CACHE_" + k.upper().replace("-", "_")] = v
    return args, env


KNOBS_NOW = {"v": None}   # the knobs a build reported for the cell being run (filled by probe_cells); store() puts them in every row of the cell whose test cannot report them itself (perplexity)
CELL_KNOBS = {}


def pin_check(env, knobs):
    """REQUESTED (the env words of the variant: LLAMA_MOE_CACHE_<NAME>=value, LLAMA_AUTOTUNE=0) against USED (the run.knobs line the build wrote): the list of pins the build does not show."""
    miss = []
    for k, v in env.items():
        if k.startswith("LLAMA_MOE_CACHE_"):
            name = k[len("LLAMA_MOE_CACHE_"):]
            if not re.search(r"(^|[\s|])" + re.escape(name) + r"\*?=" + re.escape(str(v)) + r"(\*|\s|\||$)", knobs):
                miss.append(f"{name}={v}")
        elif k == "LLAMA_AUTOTUNE" and str(v) == "0" and not re.search(r"autotune[=: ]*(0|off)", knobs, re.I) and "tuned" in knobs.lower():
            miss.append("AUTOTUNE=0")
    return miss


VARIANTS = {"default": "", "cache0": "--moe-expert-cache 0", "atoff": "-at off"}  # --variant NAME=ARGS adds or overrides


GAIN_COLS = ["campaign", "ts", "model", "test", "pl", "variant", "n", "pp_med", "base_variant", "base_n", "base_pp_med", "gain"]


def gains_write(camp):
    """Gain of every setting of a campaign as a ratio (1.22) to the baseline arm of the SAME campaign (the single-card group with the fewest cards, first run), with the GPU power limits of the rows:
    absolute t/s move with power limit, driver and boot flags, the ratio inside one session does not. Appended to gains.csv (campaign, ts, model, test, pl, variant, n, pp_med, base_variant, base_n, base_pp_med, gain)."""
    rows = [r for r in csv_load("run-history.csv") if r["campaign"] == camp and r["ok"] != "0" and num(r, "pp") not in (None, 0)]
    if not rows:
        return
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["test"], r["args"], r.get("env") or ""), []).append(r)
    def label(k):
        m = re.search(r"--moe (\S+)", k[2]); e = re.findall(r"(?:GGML|LLAMA)_[A-Z_]+=\w+", k[3])
        return ("-dev " + (re.search(r"-dev (\S+)", k[2]) or [0, "?"])[1] + " " + (m.group(1) if m else "") + (" " + " ".join(e) if e else "")).strip()
    for (model, test) in sorted({(k[0], k[1]) for k in groups}):
        cells = {k: v for k, v in groups.items() if k[0] == model and k[1] == test}
        single = [k for k in cells if "," not in (re.search(r"-dev (\S+)", k[2]) or [0, "x,y"])[1]]
        if not single:
            continue
        bk = min(single, key=lambda k: min(r["ts"] for r in cells[k]))
        bmed = statistics.median(num(r, "pp") for r in cells[bk])
        for k, v in cells.items():
            med = statistics.median(num(r, "pp") for r in v)
            csv_append("gains.csv", GAIN_COLS, dict(campaign=camp, ts=time.strftime("%F %T"), model=os.path.basename(model or ""), test=test, pl=v[-1].get("pl", ""), variant=label(k),
                       n=len(v), pp_med=round(med, 1), base_variant=label(bk), base_n=len(cells[bk]), base_pp_med=round(bmed, 1), gain=round(med / bmed, 3)))


def cmd_bench(a):
    global MODEL, PERFPROF, GPULOG, KEEP_LOOPS, CLI_EXE, TEMPLATED, KEEP_SERVER
    KEEP_SERVER = getattr(a, "keep_server", False)
    TEMPLATED = a.templated
    CLI_EXE = a.cli_exe
    PERFPROF = a.perf_prof
    GPULOG = a.gpulog or not getattr(a, "no_gpulog", False)   # on by default: the power numbers (pw_*, energy_wh, t_full_pw) are part of every run; --no-gpulog turns it off
    WRAP[:] = shlex.split(a.wrap) if getattr(a, "wrap", None) else []
    KEEP_LOOPS = a.keep_loops
    set_overrides(a)
    variants = dict(VARIANTS)
    for v in a.variant:
        k, _, x = v.partition("=")
        variants[k] = x
    names = a.variants.split(",") if a.variants else list(variants)
    global CAMPAIGN
    CAMPAIGN = camp = a.campaign or "bench-" + time.strftime("%F-%H%M")
    csv_append("campaigns.csv", CAMP_COLS, dict(name=camp, start=time.strftime("%F %T"),
               note=a.note or f"bench {a.models} builds {a.builds} variants {','.join(names)} test {a.test} n={a.n}"))
    deadline = None
    if a.deadline:
        hh, mm = a.deadline.split(":")
        deadline = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + f"{hh}:{mm}", "%Y-%m-%d %H:%M"))
        deadline += 86400 if deadline < time.time() else 0
    m = machine(a.machine) if a.machine != "pc1" else None
    stock_variants = dict(kv.partition("=")[::2] for kv in a.stock_variant)
    global REP
    n = a.n or (3 if a.test.startswith("fix") else 2)
    for model in [fcfg.model(x) for x in a.models.split(",")]:   # an alias of models.ini (q36) or a path
        MODEL = model
        if not m:
            prep_model()
        cells = [(build, name, (stock_variants.get(name, variants[names[0]] if name == "default" and len(names) == 1 and not stock_variants else "") if build.startswith("stock") else variants[name]))   # a stock cell takes the cell's one --variant text too (run -a "-ub 17600 -b 17600" was dropped: the server ran with the defaults)
                 for build in [resolve_build(x) for x in a.builds.split(",")] for name in (list(dict.fromkeys(["default"] + list(stock_variants))) if build.startswith("stock") else names)]
        # a variant may set environment variables: "env:KEY=VALUE" words (e.g. lru='env:LLAMA_MOE_CACHE_POLICY=lru'), the rest are server args
        P = (4 if a.test in ("game4c", "edit4c", "game4", "edit4") else 1) if a.cold_hot else 1   # --cold-hot: every prompt runs twice, 0 = cold (empty state file), 1 = hot (the state that run wrote)
        for pp, i in ([(p, k) for p in range(P) for k in ((0, 1, 2) if (p == 0 and P > 1) else (0, 1))] if a.cold_hot else [(None, k) for k in range(a.rep_from, n)]):  # ABBA: odd repetitions run the cells backwards, so a drift (thermal, page cache) does not favour one cell
            REP = i
            if a.cold_hot:
                OVR["pidx"] = pp
                CAMPAIGN = camp + ("-cold" if i == 0 else "-hot" if i == 1 else "-hot2")
            for build, name, vargs in (cells if i % 2 == 0 else cells[::-1]):
                if build.startswith("stock") and a.test in ("game4c", "edit4c") and not a.redo_stock and not getattr(a, "par", None):   # a stock cell that is already measured (same build, model and chat test, >= 4 valid runs) is not repeated
                    have = [r for r in csv_load("run-history.csv") if r["build"] == build and r["test"] == a.test and os.path.basename(r["model"] or "") == os.path.basename(model) and r["ok"] != "0" and num(r, "tps") is not None]
                    if len(have) >= 4:
                        if i == (0 if a.cold_hot else a.rep_from) and (pp in (None, 0)):
                            print(f"stock cell skipped: {len(have)} runs of {build} {a.test} on {os.path.basename(model)} already recorded", flush=True)
                        continue
                args, env = split_variant(build, vargs)
                if deadline and time.time() > deadline:
                    print("deadline reached: remaining cells skipped", flush=True)
                    return cmd_report(argparse.Namespace(campaign=camp, since=None, last=None, tol=None, md=False))
                if (a.clean_state or a.cold_hot) and not build.startswith("stock"):
                    sf = "/tmp/runstate/" + re.sub("[^A-Za-z0-9._-]", "_", build + "-" + name + "-" + os.path.basename(model) + (f"-{a.test}-p{pp}" if a.cold_hot else "")) + ".ini"   # one file per build, variant and model (and prompt with --cold-hot)
                    os.makedirs("/tmp/runstate", exist_ok=True)
                    if i == (0 if a.cold_hot else a.rep_from) and os.path.exists(sf):
                        os.remove(sf)   # the run's own file, never ~/.cache/llama.cpp/moe-state.ini
                    STATE_FILE["v"] = sf
                else:
                    STATE_FILE["v"] = None
                warm = warm_wanted(a.warm, a.no_warm, model) and i == 0 and not a.cold_hot  # one discarded run per cell, before its first repetition (never with --cold-hot: it would write the state)
                KNOBS_NOW["v"] = CELL_KNOBS.get((build, name))
                if m:
                    remote_cell(a.machine, m, build, a.test, args, a.note, warm=warm, env=env)
                else:
                    run_cell(build, a.test, env, False, args, True, a.note, warm=warm)
    REP = None
    for cn in ([camp + "-cold", camp + "-hot", camp + "-hot2"] if a.cold_hot else [camp]):
        try:
            gains_write(cn)
        except Exception as e:   # the gain table is a convenience: it never fails a bench call
            print(f"gains: {e}", flush=True)
        cmd_report(argparse.Namespace(campaign=cn, since=None, last=a.last, tol=a.tol, md=False))


# ---- report: variants against stock per model / test / machine ----
def cmd_report(a):
    """Per cell: n, median and min-max of decode t/s, ratio to the stock median and to the best hand-tuned stock (stock builds with args), and the
    flags that make a number unpublishable: n < 3, runs with different token counts, outputs that differ. --md prints README table rows."""
    rows = [r for r in csv_load("run-history.csv") if r["ok"] != "0" and num(r, "tps") is not None]
    if a.campaign:
        rows = [r for r in rows if r["campaign"] == a.campaign]
    elif a.since:
        rows = [r for r in rows if r["ts"] >= a.since]
    groups = {}
    for r in rows:
        stock = r["build"].startswith("stock")
        envs = " ".join(f"{k}={v}" for k, v in sorted(json.loads(r["env"] or "{}").items()) if k.startswith(("LLAMA_", "GGML_")))  # variants set by environment
        label = r["build"] + (f" {r['args']}" if stock and r["args"] else "") if stock else f"{r['build']} {r['args'] or 'default'}{' ' + envs if envs else ''}"
        groups.setdefault((r["model"] or "?", r["test"], r["hw"] or "?"), {}).setdefault(label, []).append(r)
    bad = 0
    md = []
    for (model, test, hw), cells in sorted(groups.items()):
        print(f"\n{model[:46]}  {test}  {hw}")
        stat = {}
        for label, v in cells.items():
            v = v[-a.last:] if a.last else v
            t = [num(x, "tps") for x in v]
            stat[label] = (statistics.median(t), min(t), max(t), len(t), v)
        stock = [(k, st) for k, st in stat.items() if k.startswith("stock")]
        is_tuned = lambda k: re.search(r"n-cpu-moe|-ncmoe|--cpu-moe|-ot\b|--override-tensor", k) is not None  # hand-placed stock: the matched-VRAM baseline
        base = next((st[0] for k, st in stock if not is_tuned(k)), None)
        tuned = max((st[0] for k, st in stock if is_tuned(k)), default=None)
        for label, (med, lo, hi, n, v) in stat.items():
            pp = [num(x, "pp") for x in v if num(x, "pp")]
            ngen = {x["n_gen"] for x in v if x["n_gen"]}
            md5s = [x["md5"] for x in v if x["md5"]]
            same = max((md5s.count(h) for h in set(md5s)), default=0)
            flags = [f for f, c in (("n<3", n < 3), ("n_gen differs", len(ngen) > 1), (f"md5 same {same}/{len(md5s)}", len(set(md5s)) > 1)) if c]
            ratio = f"x{med / base:.2f} stock" if base else ""
            ratio += f" x{med / tuned:.2f} tuned-stock" if tuned and not label.startswith("stock") else ""
            regress = a.tol and base and not label.startswith("stock") and med < a.tol * base
            bad += bool(regress)
            print(f"  {label[:52]:52s} n={n:2d} {med:8.2f} t/s [{lo:.2f}-{hi:.2f}]{(f'  pp {statistics.median(pp):7.1f}' if pp else '')}  {ratio}"
                  f"{'  <-- REGRESSION' if regress else ''}{('  ' + ', '.join(flags)) if flags else ''}")
            md.append(f"| {model[:40]} | {test} | {hw} | {label} | {n} | {med:.2f} [{lo:.2f}-{hi:.2f}] | "
                      f"{f'{med / base:.2f}x' if base and not label.startswith('stock') else ''} | {f'{med / tuned:.2f}x' if tuned and not label.startswith('stock') else ''} | "
                      f"{'v2' if v[0].get('proto') == 'v2' else 'v1'} {v[0]['ts'][:16]} +{n - 1} |")
    if getattr(a, "md", False):
        print("\n| model | test | machine | build and settings | n | median [min-max] t/s | vs stock | vs tuned stock | protocol, first run |\n|---|---|---|---|---|---|---|---|---|")
        print("\n".join(md))
    if a.tol:
        print("\nRESULT:", "REGRESSION" if bad else "OK")


# ---- prompt: the fork's benchmark prompts against a running server (what the README numbers use) ----
def cmd_prompt(a):
    set_overrides(a)
    kinds = {"short": ("/v1/chat/completions", None), "chatv": ("/v1/chat/completions", None),
             "12k": ("/completion", "src12k"), "128k": ("/completion", "src128k"), "code": ("/completion", "src12k")}
    path, f = kinds[a.kind]
    if f:
        text = open(config.prompt_path(f)).read()
        body = {"prompt": ovr_prompt(text[:9000] if a.kind == "code" else text),
                "n_predict": int(OVR.get("tokens") or (64 if a.kind == "128k" else 32)), "cache_prompt": False, **GREEDY}
    else:
        pr = CHAT_PROMPTS[a.idx % len(CHAT_PROMPTS)] if a.kind == "chatv" else GAME_TEMPLATE.format(word="tetris")
        body = {"messages": [{"role": "user", "content": ovr_prompt(pr)}], "max_tokens": int(OVR.get("tokens") or (300 if a.kind == "chatv" else 1500)), **GREEDY,
                **({"chat_template_kwargs": {"enable_thinking": False}} if a.kind == "chatv" else {})}
    base = a.url.rstrip("/")
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=7200))["timings"]
    print(f"{a.kind}: prompt {t['prompt_n']} tokens, {t['prompt_per_second']:.1f} t/s ({t['prompt_ms'] / 1e3:.0f} s) | "
          f"decode {t['predicted_n']} tokens, {t['predicted_per_second']:.2f} t/s"
          + (f" | draft {t.get('draft_n_accepted')}/{t['draft_n']}" if t.get("draft_n") else ""))


# ---- ctl: one server, settings switched at runtime (LLAMA_MOE_CACHE_CTL), rotating prompts and run order ----
def cmd_ctl(a):
    global BARE, IDX
    if not MODEL:
        sys.exit("set MODEL=/path/to/model.gguf")
    set_overrides(a)
    BARE = True
    wait_vram_free()
    ctl = "/tmp/perf-ctl.txt"
    open(ctl, "w").close()
    for _ in range(120):  # a previous server's pinned weights take a while to be released; starting before that makes auto mode fall back to pageable uploads
        avail = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) // 1048576
        if avail >= a.min_avail_gb:
            break
        time.sleep(2)
    p, logf = server_start(a.build, {**os.environ, "LD_LIBRARY_PATH": os.path.join(HERE, a.build), "LLAMA_MOE_CACHE_STATS": "1",
                                     "LLAMA_MOE_CACHE_CTL": ctl}, argv_of(a.args) if a.args else [])
    try:
        if "from pageable host memory" in open(logf).read():
            sys.exit("ctl: weights not pinned (pageable uploads), results would not compare: aborting")

        def ask(n, prompt):
            r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, **GREEDY,
                                               "chat_template_kwargs": {"enable_thinking": False}})
            return r["timings"]
        ask(64, "Hello, who are you?")
        for w in range(a.prewarm):  # a self-tuning server converges before it is measured
            ask(1500, CHAT_PROMPTS[w % len(CHAT_PROMPTS)])
        V = [v.partition(":")[::2] for v in a.variants]   # (name, "K=V,K=V")
        V = [(n, kv) for n, kv in V]
        sums = {n: [] for n, _ in V}
        vstr = version(a.build)
        for r in range(1, a.n + 1):
            for j in range(len(V)):  # the run order rotates each round: no variant always runs first or last (machines heat up during a round)
                i = (j + r - 1) % len(V)
                name, kv = V[i]
                open(ctl, "w").write(kv.replace(",", "\n") + "\n")
                prompt = CHAT_PROMPTS[(((r - 1) // len(V)) * len(V) + (r - 1 + i) % len(V)) % len(CHAT_PROMPTS)]
                ask(a.warm, "Say hi in one short sentence.")
                t = ask(a.tokens, ovr_prompt(prompt))
                row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"], "bestpp": bestpp_run(t), "ok": 1, "args": kv}
                store(row, a.build, "ctl", {"variant": name, "settings": kv, "round": str(r), "bare": "1"}, a.note, os.environ.get("PERF_HW") or "pc1", vstr)
                sums[name].append(row["tps"])
            print(f"round {r}: " + " ".join(f"{n} {sums[n][-1]:.2f}" for n, _ in V), flush=True)
        for n, _ in V:
            print(f"{n} mean {statistics.mean(sums[n]):.2f} t/s (n={len(sums[n])})")
    finally:
        stop(p)
        for l in open(logf, errors="replace"):
            if re.search(r"cached on|ctl: segment|phases:", l):
                print(l.strip()[:250])


def add_overrides(sub):
    sub.add_argument("--prompt", help="prompt text or @file instead of the test's own")
    sub.add_argument("--tokens", type=int, help="generated tokens instead of the test's own")
    sub.add_argument("--ctx", type=int, help="context size instead of the test's own")
    sub.add_argument("--clamp", type=lambda v: int(float(v[:-1]) * 1000) if v.lower().endswith("k") else int(v), metavar="N[k]",
                     help="cut the prompt of the test to N tokens (8k = 8000, so it fits one -ub 8192 pass), measured with the server's own /tokenize, for any model; raw-prompt tests only")
    sub.add_argument("--chat", metavar="INSTR", help="pf tests: send INSTR + the prompt through the model's chat template (instruct models stop or loop on raw cut text); --clamp cuts the text before the template")
    sub.add_argument("--nothink", action="store_const", const=True, help="chat tests (tetrisc, edit4c ...): reasoning off (chat_template_kwargs enable_thinking=false), the answer is the text itself")
    sub.add_argument("--lead", type=int, help="first a request with a prompt of about N tokens to the same server, then the measured one (the KV grows, then shrinks)")
    sub.add_argument("--repeat", type=int, help="send the request this many times to the same server and report the last one (warm: kernels loaded, pinned buffers touched, pages mapped); the first one's speed goes to the note as first_pp")
    sub.add_argument("--par", type=int, help="send N identical requests at once (start the server with -np N too, e.g. --variant 'v=-np 10'); the row is the aggregate: generated tokens of all requests over the wall time, prompt tokens of all over the longest prompt phase; a failed request fails the run")
    sub.add_argument("--pidx", type=int, help="game4 / edit4: use prompt N of the four for every repetition (default: repetition k uses prompt k mod 4)")


# ---- regress: the quick check after every commit: this build against the best binary kept so far ----
BEST = os.environ.get("PERF_BEST") or fcfg.path("best")   # BEST/current -> the frozen copy of the fastest build so far; BEST/history.csv: every change of it
REGRESS_MODEL = os.environ.get("REGRESS_MODEL") or fcfg.model("olmoe")
# 19 of 64 experts per layer (30%) in VRAM: the cache has to work for its speed; no learned state: every run of every build starts the same
REGRESS_ARGS = os.environ.get("REGRESS_ARGS", "--cpu-moe --moe-expert-cache 19 -t 6 -tb 8 --moe autotune=0,predict=0,admit-boot=0")   # knobs pinned: a default change must not mask or fake a speed change; this cell amplifies scheduler regressions (the #29943 merge cost it 15-17%, GLM 9.5%)   # learned state: --clean-state below (a fresh state file per build; builds of every age take it)
# cells of the check: (model, args, visible GPUs or None for both, runs or None for -n, big). Every MoE model we publish numbers for; restricted caches
# (30-35% of the experts) on the two small ones so the cache path is what is timed. The verdict is the worst cell. REGRESS_FAST=1: only the small cells.
_T = " -t 6 -tb 8"
REGRESS_CELLS = [(REGRESS_MODEL, REGRESS_ARGS, "1", None, False),
                 (fcfg.model("qwen36"), "--cpu-moe --moe-expert-cache 90" + _T, "1", None, False),
                 (fcfg.model("qwen38"), "--cpu-moe" + _T, None, 2, True),
                 (fcfg.model("glm"), "--cpu-moe" + _T, None, 2, True),
                 (fcfg.model("mimo"), "--cpu-moe" + _T, None, 2, True),   # the README MiMo (115 GB, fits RAM); the MXFP4 file in the config (169 GB) does not
                 (fcfg.model("q27_iq4nl"), _T.strip(), "1", None, False),   # IQ4_NL: the dense README model (no experts: no cache flags), covers the IQ4 kernels
                 (fcfg.model("qn_iq4xs"), "--cpu-moe" + _T, None, 2, True)]   # IQ4_XS gainer
if os.environ.get("REGRESS_FAST") == "1":
    REGRESS_CELLS = [c for c in REGRESS_CELLS if not c[4]]


def regress_label(d):
    """the name of a build directory in the run log: <branch>-<commit>[-<hash of the uncommitted diff>] from its BUILD_INFO, else the directory name"""
    p = os.path.join(d, "BUILD_INFO")
    bi = dict(l.strip().split("=", 1) for l in open(p) if "=" in l) if os.path.exists(p) else {}
    if not bi.get("commit"):
        return os.path.basename(d.rstrip("/"))
    return f"{bi.get('branch', 'x')}-{bi['commit']}" + ("" if bi.get("patch", "0") == "0" else "-" + bi["patch"])


def regress_promote(name, src, med, n, camp, old, why):
    """src becomes the best build: a frozen copy under BEST, BEST/current points to it, one line in BEST/history.csv"""
    os.makedirs(BEST, exist_ok=True)
    dst = os.path.join(BEST, name)
    if os.path.realpath(src) != os.path.realpath(dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.realpath(src), dst)
    tmp = os.path.join(BEST, ".current.new")
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(name, tmp)
    os.replace(tmp, os.path.join(BEST, "current"))
    new = not os.path.exists(os.path.join(BEST, "history.csv"))
    with open(os.path.join(BEST, "history.csv"), "a") as f:
        if new:
            f.write("ts,best,tps_median,runs,campaign,previous,why\n")
        f.write(f"{time.strftime('%F %T')},{name},{med:.2f},{n},{camp},{old or ''},{why}\n")


def queue_lock(wait=True):
    """/tmp/queue.lock for this process, unless it runs as a queue task (the daemon holds the lock for the task then: QUEUE_HOLDS_LOCK=1).
    Returns the open lock file (keep it referenced), None when the daemon holds it, or raises OSError when wait=False and the lock is taken."""
    if os.environ.get("QUEUE_HOLDS_LOCK"):
        return None
    import fcntl
    lk = open("/tmp/queue.lock", "a")
    fcntl.flock(lk, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    return lk


DEVBUILDS = os.environ.get("PERF_DEVBUILDS") or fcfg.path("devbuilds")
BUILD_REPO = os.environ.get("PERF_BUILD_REPO") or fcfg.path("build_repo")   # the repository whose objects a --commit build checks out


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


REL_OUT = os.environ.get("PERF_RELEASE_OUT") or fcfg.path("release_out")   # release archives (Linux tar.gz, Windows zip copies); the KEEP_BUILDS newest build numbers stay
SMOKE_IMAGES = ["rockylinux:8", "ubuntu:20.04", "ubuntu:22.04", "debian:12", "ubuntu:24.04"]   # the Linux archive must start on all of them (glibc 2.28 floor)


IMAGES = os.environ.get("PERF_IMAGES") or fcfg.path("images")   # docker save of every image a release was built with: an old build can be redone with the same compiler


def _image(name, keep=False):
    """the id of a build image, built from fork/docker/Dockerfile when it is missing; keep: saved once per id to IMAGES (zstd, restore with
    `zstd -dc FILE | docker load`). The ccache is shared and content-addressed (CCACHE_MAXSIZE 100G), so an old commit rebuilt in the same image is fast."""
    insp = lambda: subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", name], capture_output=True, text=True)
    r = insp()
    if r.returncode != 0:
        subprocess.run(["docker", "build", "-t", name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker")], check=True)
        r = insp()
    iid = r.stdout.strip().split(":")[-1][:12]
    f = os.path.join(IMAGES, f"{name}-{iid}.tar.zst")
    if keep and not os.path.exists(f):
        os.makedirs(IMAGES, exist_ok=True)
        print(f"saving build image {name}@{iid} to {f} (once per image)", flush=True)
        subprocess.run(f"docker save {name} | zstd -q -T0 -3 -o {f}.part && mv {f}.part {f}", shell=True, check=True)
    return iid


def _rel_prune(keep=int(os.environ.get("KEEP_BUILDS", "2"))):
    """release-out keeps the archives of the `keep` newest build numbers (published releases live on GitHub); the shared cudart-llama-bin-win zip stays"""
    vers = sorted({int(m.group(1)) for f in os.listdir(REL_OUT) for m in [re.search(r"-b(\d{5})-", f)] if m}, reverse=True)
    for v in vers[keep:]:
        for f in glob.glob(os.path.join(REL_OUT, f"*-b{v}-*")):
            print(f"prune {os.path.basename(f)}", flush=True)
            shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)


def _build_stock(a):
    """upstream llama.cpp built exactly like upstream's release workflow (job ubuntu-cuda, x64, CUDA 12.8: nvidia/cuda:12.8.2-devel-ubuntu24.04, GCC 13,
    the workflow's cmake flags): stock numbers must be what upstream's binaries do, never the fork's recipe. Result: <gh_rel>/stock-b<build>-<hash9>
    (write-protected, with the CUDA 12.8 runtime libs) and a link to it in DEVBUILDS."""
    src = os.path.expanduser(a.src); img = "nvidia/cuda:12.8.2-devel-ubuntu24.04"
    _git(src, "fetch", "-q", "origin"); _git(src, "checkout", "-q", a.ref)
    h = _git(src, "rev-parse", "--short=9", "HEAD"); B = os.path.join(src, "build-stock-ubuntu")
    os.makedirs(B, exist_ok=True); os.makedirs(fcfg.path("ccache_stock"), exist_ok=True)
    inner = ("set -e; apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build libssl-dev git ccache >/dev/null; "
             "git config --global --add safe.directory /src; "
             "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache "
             "-DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF "
             "-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3 -DLLAMA_BUILD_UI=OFF; "
             "cmake --build build --config Release -j $(nproc); "
             "cp -L /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 build/bin/; "
             f"chown -R {os.getuid()}:{os.getgid()} build")
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "logs", f"build-stock-{h}.log")
    print(f"stock build of {h} ({_git(src, 'log', '-1', '--format=%s')[:70]}) in {img}, log {log}", flush=True)
    lk = queue_lock(wait=True)   # no build while a benchmark runs
    with open(log, "w") as lf:
        rc = subprocess.run(["docker", "run", "--rm", "-v", f"{src}:/src", "-v", f"{B}:/src/build", "-v", fcfg.path("ccache_stock") + ":/ccache", "-e", "CCACHE_DIR=/ccache",
                             "-w", "/src", img, "bash", "-c", inner], stdout=lf, stderr=subprocess.STDOUT).returncode
    del lk
    vout = subprocess.run([os.path.join(B, "bin", "llama-server"), "--version"], capture_output=True, text=True, env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(B, "bin")))
    m = re.search(r"build (\d+)", vout.stdout + vout.stderr)
    if rc != 0 or not m:
        sys.exit(f"stock build: FAILED (exit {rc}), log {log}")
    D = os.path.join(GH_REL, f"stock-b{m.group(1)}-{h}")
    if os.path.exists(D):
        subprocess.run(["chmod", "-R", "u+w", D]); shutil.rmtree(D)
    shutil.copytree(os.path.join(B, "bin"), D, symlinks=True); shutil.copy2(os.path.join(src, "LICENSE"), D)
    with open(os.path.join(D, "BUILD_INFO"), "w") as f:
        f.write(f"commit={h}\nbranch=upstream-stock (upstream release recipe: ubuntu 24.04, CUDA 12.8.2, GCC)\n")
    subprocess.run(["chmod", "-R", "a-w", D])
    lnk = os.path.join(DEVBUILDS, os.path.basename(D))
    if not os.path.lexists(lnk):
        os.symlink(D, lnk)
    print(f"stock build: OK -> {D} (also {lnk})", flush=True)


def _package_linux(a, W, binp, log):
    """the release archives of a release build: llama-release-b<build>-<commit>-bin-ubuntu-cuda-<toolkit>-x64.tar.gz (binaries, LICENSE, the libomp.so the clang
    build links dynamically) and the matching cudart archive, smoke-started in SMOKE_IMAGES; DEVBUILDS/NAME = both archives unpacked (what users get)"""
    uid = f"{os.getuid()}:{os.getgid()}"
    dk = lambda *args: subprocess.run(["docker", "run", "--rm", "--user", uid] + list(args), capture_output=True, text=True)
    vout = dk("-e", "LD_LIBRARY_PATH=/b", "-v", f"{binp}:/b:ro", "llama-rel-el8", "/b/llama-server", "--version")   # the host has neither libomp nor the CUDA 13 runtime
    v = re.search(r"build (\d+)", vout.stdout + vout.stderr); c = re.search(r"commit ([0-9a-f]+)", vout.stdout + vout.stderr)
    if not v or not c:
        sys.exit(f"release: no version from llama-server --version: {(vout.stdout + vout.stderr)[-300:]}")
    nv = dk("llama-rel-el8", "nvcc", "--version")   # the toolkit the image really has: the archive name used to say 13.4 whatever built it (b13600 shipped older libraries under that name)
    cv = re.search(r"release (\d+\.\d+)", nv.stdout + nv.stderr)
    if not cv:
        sys.exit(f"release: no CUDA version from nvcc in the image: {(nv.stdout + nv.stderr)[-200:]}")
    N = f"llama-release-b{v.group(1)}-{c.group(1)}-bin-ubuntu-cuda-{cv.group(1)}-x64"; C = "cudart-" + N
    os.makedirs(REL_OUT, exist_ok=True)
    for d in (N, C):
        shutil.rmtree(os.path.join(REL_OUT, d), ignore_errors=True); os.makedirs(os.path.join(REL_OUT, d))
    shutil.copytree(binp, os.path.join(REL_OUT, N), symlinks=True, dirs_exist_ok=True); shutil.copy2(os.path.join(W, "LICENSE"), os.path.join(REL_OUT, N))
    dk("-v", f"{REL_OUT}/{N}:/o", "llama-rel-el8", "bash", "-c", "cp -L /usr/lib64/libomp.so /o/libomp.so")
    dk("-v", f"{REL_OUT}/{C}:/o", "llama-rel-el8", "bash", "-c", "for l in cudart cublas cublasLt; do cp -L /usr/local/cuda/lib64/lib$l.so.13 /o/; done")
    for d in (N, C):
        subprocess.run(["tar", "czf", f"{d}.tar.gz", d], cwd=REL_OUT, check=True)
    with open(log, "a") as lf:
        for img in SMOKE_IMAGES:
            r = subprocess.run(["docker", "run", "--rm", "-e", "LD_LIBRARY_PATH=/c", "-v", f"{REL_OUT}/{N}:/r:ro", "-v", f"{REL_OUT}/{C}:/c:ro", img, "/r/llama-server", "--version"],
                               capture_output=True, text=True)
            line = next((l for l in (r.stdout + r.stderr).splitlines() if "version" in l or "error" in l), "no output")
            lf.write(f"smoke {img}: {line}\n"); print(f"smoke {img}: {line}", flush=True)
    dst = os.path.join(DEVBUILDS, a.name)
    if os.path.islink(dst):
        os.remove(dst)
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for d in (N, C):   # the archives, not the build tree: the numbers are the published artifact's
        subprocess.run(["tar", "xzf", os.path.join(REL_OUT, f"{d}.tar.gz"), "-C", dst, "--strip-components=1"], check=True)
        shutil.rmtree(os.path.join(REL_OUT, d), ignore_errors=True)
    _rel_prune()
    return N, dst


def _publish_db(here):
    """fork/runs.db as the ONLY commit of branch runs-db (no parent: the history of the binary never accumulates), force-with-lease to the fork, that branch only.
    Get it: git fetch neurall runs-db && git show FETCH_HEAD:runs.db > fork/runs.db"""
    import sqlite3, tempfile
    R = _git(here, "rev-parse", "--show-toplevel"); out = os.path.join(tempfile.mkdtemp(), "runs.db")
    c = sqlite3.connect(os.path.join(here, "runs.db"), timeout=60)
    sys.modules["runstats"].refresh(c)   # the statistics of records added since the last query are current in the published file
    c.execute("VACUUM INTO ?", (out,)); c.close()
    n = sqlite3.connect(out).execute("select count(*) from runs").fetchone()[0]
    blob = _git(R, "hash-object", "-w", out)
    tree = subprocess.run(["git", "-C", R, "mktree"], input=f"100644 blob {blob}\truns.db\n", capture_output=True, text=True, check=True).stdout.strip()
    com = _git(R, "commit-tree", tree, "-m", f"runs.db: {n} runs, {time.strftime('%F %H:%M')} (only commit of this branch, replaced on every publish)")
    _git(R, "update-ref", "refs/heads/runs-db", com)
    subprocess.run(["git", "-C", R, "reflog", "expire", "--expire=now", "refs/heads/runs-db"], capture_output=True)
    subprocess.run(["git", "-C", R, "push", "--force-with-lease=refs/heads/runs-db", "neurall", "refs/heads/runs-db:refs/heads/runs-db"], check=True)
    print(f"runs-db = {com[:9]} ({n} runs)", flush=True)

def cmd_publish(a):
    """run.py publish [--tree DIR] [--name NAME --notes FILE [--replace OLDTAG]]: pushes the tree's branch to the fork (remote neurall, explicit refspec, no force), replaces runs.db there (branch runs-db), and with --name tags the head
    release-b<build>-<hash>-<name> and makes the GitHub release (neurall/llama.cpp) from the archives of that build in REL_OUT; --replace OLDTAG deletes that release and its tag first. Never origin."""
    tree = os.path.abspath(a.tree); here = os.path.dirname(os.path.abspath(__file__))
    br = _git(tree, "branch", "--show-current")
    for ref in [f"refs/heads/{br}:refs/heads/{br}"]:
        subprocess.run(["git", "-C", tree, "push", "neurall", ref], check=True)
    _publish_db(here)
    if not a.name:
        return
    rev = _git(tree, "rev-parse", "HEAD"); h = rev[:9]
    # the commit in an archive name is as long as the building git made it (the Linux recipe gave 10 characters on 2026-10-28, the Windows build 9): match the
    # 9-character prefix and any further hex digits
    arch = sorted(f for f in os.listdir(REL_OUT) if "llama-release-b" in f and re.search(rf"-{h}[0-9a-f]*-", f))
    if not arch: sys.exit(f"publish: no archives for {h} in {REL_OUT}")
    b = re.search(r"llama-release-b(\d+)-", arch[0]).group(1); tag = f"release-b{b}-{h}-{a.name}"
    gh = ["gh", "-R", "neurall/llama.cpp"]
    if a.replace:
        subprocess.run(gh + ["release", "delete", a.replace, "--cleanup-tag", "-y"], check=True)
    subprocess.run(["git", "-C", tree, "tag", "-a", tag, "-m", f"b{b} {a.name}", rev], check=True)
    subprocess.run(["git", "-C", tree, "push", "neurall", f"refs/tags/{tag}:refs/tags/{tag}"], check=True)
    subprocess.run(gh + ["release", "create", tag, "--title", f"b{b} {a.name}", "--notes-file", a.notes, "--latest"] + [os.path.join(REL_OUT, f) for f in arch], check=True)
    print(f"published {tag}: {', '.join(arch)}", flush=True)


def cmd_build(a):
    r"""Builds, in containers (pc1) or on the W10 laptop (pc3), with ccache.
    default: dev build (the release container and flags, llama-server / llama-cli / llama-perplexity only) of a worktree as it is (--tree) or of a commit
      (--commit: a temporary worktree, removed after the build) into DEVBUILDS/NAME with BUILD_INFO (commit, branch, patch).
    --release: the release recipe (was hcf/lrel/build.sh): every target incl. tests, the archives in REL_OUT (Linux tar.gz + cudart), smoke starts in
      SMOKE_IMAGES, DEVBUILDS/NAME = the archives unpacked. Image llama-rel-el8 from fork/docker/Dockerfile.
    --stock: upstream at --ref from --src with upstream's release recipe (was hcf/lrel/stock-build.sh) into <gh_rel>/stock-b<build>-<hash> (no NAME).
    After a pc1 build the regress cell runs (restricted OLMoE, pinned knobs, against the best binary) unless --no-regress.
    --machine pc3: the same commit built on the W10 laptop (git bundle over ssh, its release-out\winbuild.bat) into N:\NAME (no regress there);
      with --release also the zip (binaries, VC runtime DLLs, LICENSE) in N:\llama.cpp\release-out and a copy in REL_OUT."""
    if a.stock:
        return _build_stock(a)
    if not a.name:
        sys.exit("build: NAME is needed (the directory under DEVBUILDS)")
    name = a.name
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "logs", f"build-{name}" + ("" if a.machine == "pc1" else f"-{a.machine}") + ".log")
    if a.machine == "pc3" and not a.release:
        sys.exit("build --machine pc3: only the release recipe (clang 20 for the CPU backend, as every published Windows release); the dev bat uses MSVC and fails on the GCC-style kernels. Add --release.")
    if a.machine == "pc3":
        rev = _git(a.repo, "rev-parse", a.commit or "HEAD") if not a.tree else _git(a.tree, "rev-parse", "HEAD")
        short = rev[:9]; ref = f"run-build-{name}"; bundle = f"/tmp/{ref}.bundle"
        repo = a.tree or a.repo
        _git(repo, "branch", "-f", ref, rev)
        r = subprocess.run(["git", "-C", repo, "bundle", "create", bundle, ref, "--not", a.w10_base], capture_output=True, text=True)
        _git(repo, "branch", "-D", ref)
        if r.returncode != 0:
            sys.exit(f"bundle: {r.stderr.strip()}")
        host = fcfg.get("hosts", "w10_ssh") or sys.exit("build --machine pc3: set [hosts] w10_ssh in the fork config (ssh target of the Windows build host)")
        w10 = fcfg.get("hosts", "w10_root")   # drive/root of the build host, e.g. N:
        ssh = ["ssh", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=120", host]
        with open(log, "w") as lf:
            subprocess.run(["scp", "-q", bundle, f"{host}:C:/Users/o/{ref}.bundle"], stdout=lf, stderr=subprocess.STDOUT, check=True)
            cmd = (f"cd /d N:\\llama.cpp && git fetch -q C:\\Users\\o\\{ref}.bundle {ref} && git checkout -q FETCH_HEAD && call N:\\llama.cpp\\release-out\\winbuild-rel.bat"
                   f" && (if not exist N:\\build mkdir N:\\build) && (if not exist N:\\build\\{name} mkdir N:\\build\\{name}) && xcopy /q /y /e N:\\build\\rel-cuda134-clang\\* N:\\build\\{name}\\"
                   f" & (copy /y N:\\build\\admit-pin-b12353-10d6c1e5c\\*140*.dll N:\\build\\{name}\\ >nul 2>&1) & echo PKG_OK {short}")   # the VC runtime DLLs of an old build, when it is still there: the release zip takes them from the b11214 zip anyway
            subprocess.run(ssh + [cmd], stdout=lf, stderr=subprocess.STDOUT)
        os.remove(bundle)
        ok = f"PKG_OK {short}" in open(log).read() and "BUILD_OK" in open(log).read()   # PKG_OK is echoed even when the bat failed
        if ok and a.release:   # the zip the way every published Windows release was made (build-cuda-dl\bin + the VC runtime DLLs + LICENSE)
            vout = subprocess.run(ssh + [f"N:\\build\\{name}\\llama-server.exe --version"], capture_output=True, text=True)
            v = re.search(r"build (\d+)", vout.stdout + vout.stderr); c = re.search(r"commit ([0-9a-f]+)", vout.stdout + vout.stderr)
            if not v or not c:
                sys.exit(f"release zip: no version from llama-server.exe: {(vout.stdout + vout.stderr)[-300:]}")
            nv = subprocess.run(ssh + ["N:\\nv\\bin\\nvcc --version"], capture_output=True, text=True)   # the toolkit that built ggml-cuda.dll, not a fixed 13.4
            cv = re.search(r"release (\d+\.\d+)", nv.stdout + nv.stderr)
            if not cv:
                sys.exit(f"release zip: no CUDA version from nvcc on the laptop: {(nv.stdout + nv.stderr)[-200:]}")
            z = f"llama-release-b{v.group(1)}-{c.group(1)}-bin-win-cuda-{cv.group(1)}-x64.zip"
            pkg = (f"set P=N:\\tmp\\pkg& (if exist N:\\tmp\\pkg rmdir /s /q N:\\tmp\\pkg) & mkdir N:\\tmp\\pkg && xcopy /q /y /e N:\\build\\rel-cuda134-clang\\* N:\\tmp\\pkg\\ >nul"
                   f" && cd /d N:\\tmp\\pkg && tar -xf N:\\llama.cpp\\release-out\\llama-release-b11214-a3768a8-bin-win-cuda-13.4-x64.zip msvcp140.dll vcruntime140.dll vcruntime140_1.dll"
                   f" && copy /y N:\\llama.cpp\\LICENSE N:\\tmp\\pkg\\ >nul && (if exist N:\\llama.cpp\\release-out\\{z} del N:\\llama.cpp\\release-out\\{z})"
                   f" && tar -a -cf N:\\llama.cpp\\release-out\\{z} * && echo ZIP_OK")
            with open(log, "a") as lf:
                r = subprocess.run(ssh + [pkg], stdout=lf, stderr=subprocess.STDOUT)
                ok = "ZIP_OK" in open(log).read() and subprocess.run(["scp", "-q", f"{host}:N:/llama.cpp/release-out/{z}", os.path.join(REL_OUT, z)], stdout=lf, stderr=subprocess.STDOUT).returncode == 0
            print(f"release zip {z}: " + (f"OK, N:\\llama.cpp\\release-out and {REL_OUT}" if ok else f"FAILED, see {log}"), flush=True)
        if ok and a.release:   # N:\rel\latest always points at the newest release build (a junction, no admin rights needed)
            subprocess.run(ssh + [f"if exist N:\\rel\\latest rmdir N:\\rel\\latest & mklink /J N:\\rel\\latest N:\\build\\{name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"build {name} on pc3 ({short}): {'OK, N:\\build\\' + name if ok else 'FAILED, see ' + log}", flush=True)
        sys.exit(0 if ok else 1)
    tmp = None
    if a.commit:
        rev = _git(a.repo, "rev-parse", a.commit)
        tmp = os.path.join(BUILD_TMP, name)
        if os.path.exists(tmp):
            _git(a.repo, "worktree", "remove", "--force", tmp, check=False)
        _git(a.repo, "worktree", "add", "--detach", tmp, rev)
        W = tmp
    else:
        W = os.path.abspath(a.tree)
    B = os.path.join(W, "build-linux" if a.release else "build-dev")
    gitc = _git(W, "rev-parse", "--path-format=absolute", "--git-common-dir")
    os.makedirs(B, exist_ok=True); os.makedirs(fcfg.path("ccache"), exist_ok=True)
    gtd = "--gcc-install-dir=/opt/rh/gcc-toolset-15/root/usr/lib/gcc/x86_64-redhat-linux/15"
    inner = ("source /opt/rh/gcc-toolset-13/enable && git config --global --add safe.directory '*' && SLD='-static-libstdc++ -static-libgcc' && "
             f"cmake -S {W} -B {B} -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS={gtd} -DCMAKE_CXX_FLAGS={gtd} "
             "-DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/gcc -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DBUILD_SHARED_LIBS=ON "
             "-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=ON "
             f"-DOPENSSL_USE_STATIC_LIBS=ON -DOPENSSL_ROOT_DIR=/opt/openssl -DLLAMA_BUILD_TESTS={'ON' if a.release or any(t.startswith('test-') for t in a.targets) else 'OFF'} -DLLAMA_BUILD_UI=OFF "
             "-DCMAKE_EXE_LINKER_FLAGS=\"$SLD -pthread\" -DCMAKE_SHARED_LINKER_FLAGS=\"$SLD\" -DCMAKE_MODULE_LINKER_FLAGS=\"$SLD\" && "
             + (f"cmake --build {B} -j $(nproc)" if a.release else f"cmake --build {B} -j 8 --target llama-server llama-cli llama-perplexity " + " ".join(a.targets)))
    lk = queue_lock(wait=True)   # no build while a benchmark runs (a build takes the CPU the measurements use)
    iid = _image("llama-rel-el8", keep=a.release)
    idf = os.path.join(B, ".image-id")   # a build dir keeps the absolute toolkit paths of the image that configured it (CUDA 13.2 paths broke a 13.4 release build, 2026-10-22): another image, or a dir from before this check, is configured afresh; the objects come back from ccache
    if (open(idf).read().strip() if os.path.exists(idf) else None) != str(iid):
        for stale in ("CMakeCache.txt", "CMakeFiles"):
            sp = os.path.join(B, stale)
            if os.path.isdir(sp): shutil.rmtree(sp)
            elif os.path.exists(sp): os.remove(sp)
        print(f"build: {B} was configured with another image, configuring afresh (ccache serves the objects)", flush=True)
    open(idf, "w").write(str(iid))
    docker = ([] if a.release else ["nice", "-n", "19"]) + ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "CCACHE_DIR=/ccache",
              "-e", "CCACHE_MAXSIZE=100G", "-e", f"CCACHE_BASEDIR={W}", "-e", "CCACHE_NOHASHDIR=1", "-v", fcfg.path("ccache") + ":/ccache", "-v", f"{W}:{W}",
              "-v", f"{gitc}:{gitc}:ro", "llama-rel-el8", "bash", "-c", inner]
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(docker, stdout=lf, stderr=subprocess.STDOUT).returncode
    del lk
    binp = os.path.join(B, "bin")
    if rc != 0 or not os.path.exists(os.path.join(binp, "llama-server")):
        errs = [l for l in open(log, errors="replace").read().splitlines() if " error" in l or "Error " in l or "undefined" in l][:8]
        print(f"build {name}: FAILED (exit {rc}) after {time.time() - t0:.0f} s, log {log}" + "".join("\n  " + e[:220] for e in errs), flush=True)
        if tmp:
            _git(a.repo, "worktree", "remove", "--force", tmp, check=False)
        sys.exit(1)
    if not os.path.exists(os.path.join(binp, "libomp.so")):   # the clang build links the LLVM OpenMP runtime dynamically; the image has it
        subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{binp}:/o", "llama-rel-el8", "bash", "-c", "cp -L /usr/lib64/libomp.so /o/libomp.so"],
                       capture_output=True)
    srcs = ["src", "common", "include", "ggml", "tools/server"]
    patch = subprocess.run(["git", "-C", W, "diff", "HEAD", "--"] + srcs, capture_output=True).stdout
    info = (f"commit={_git(W, 'rev-parse', '--short', 'HEAD')}\nbranch={_git(W, 'rev-parse', '--abbrev-ref', 'HEAD')}\npatch={hashlib.sha1(patch).hexdigest()[:8] if patch else 0}\n"
            f"image=llama-rel-el8@{iid}\n")
    if a.release:
        arch, dst = _package_linux(a, W, binp, log)
        info += f"release={arch}\n"
    else:
        with open(os.path.join(binp, "BUILD_INFO"), "w") as f:
            f.write(info)
        dst = os.path.join(DEVBUILDS, name)
        if os.path.islink(dst):
            os.remove(dst)
        elif os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(binp, dst, symlinks=True)
    with open(os.path.join(dst, "BUILD_INFO"), "w") as f:
        f.write(info)
    if tmp:
        _git(a.repo, "worktree", "remove", "--force", tmp, check=False)
    print(f"build {name}: OK in {time.time() - t0:.0f} s -> {dst} ({open(os.path.join(dst, 'BUILD_INFO')).read().split()[0]})", flush=True)
    if os.path.dirname(os.path.abspath(dst)) == os.path.abspath(DEVBUILDS):   # devbuilds/latest -> this build (atomic): queued cells that name "latest" pick it up without a reschedule
        lnk = os.path.join(DEVBUILDS, ".latest.tmp"); os.path.lexists(lnk) and os.remove(lnk); os.symlink(name, lnk); os.replace(lnk, os.path.join(DEVBUILDS, "latest"))
    if not a.no_regress:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "regress", dst, "--wait"])
        sys.exit(r.returncode)


def cmd_quick(a):
    """The short benchmark for anyone with a machine and a model: the fork build against a stock build, interleaved, on short chat answers (decode) and a
    12k-token prompt (prompt processing). Nothing is written to the run history unless --record. Prints one markdown table to paste into a hardware report."""
    import tempfile
    if not a.record:
        os.environ["FORK_NO_RUNS_DB"] = "1"
    fork = os.path.abspath(os.path.expanduser(a.fork))
    for d, what in ((fork, "--fork"), (os.path.abspath(os.path.expanduser(a.stock)) if a.stock else None, "--stock")):
        if d and not os.path.exists(os.path.join(d, "llama-server")):
            sys.exit(f"quick: no llama-server in {what} {d}")
    model = os.path.abspath(os.path.expanduser(a.model))
    if not os.path.exists(model):
        sys.exit(f"quick: model file not found: {model}")
    stage = tempfile.mkdtemp(prefix="quick-builds-")
    data = os.environ.get("RUN_DATA") if a.record else tempfile.mkdtemp(prefix="quick-data-")
    os.symlink(fork, os.path.join(stage, "fork"))
    builds = ["fork"]
    if a.stock:
        os.symlink(os.path.abspath(os.path.expanduser(a.stock)), os.path.join(stage, "stock-ref"))
        builds = ["stock-ref", "fork"]
    env = dict(os.environ, PERF_BUILDS=stage, PERF_LOG="r", **({"RUN_DATA": data} if data else {}), **({} if a.record else {"FORK_NO_RUNS_DB": "1"}))
    tests = [t for t in (("short chat answers (decode)", "game4c", ["--tokens", str(a.tokens)]), ("12k-token prompt (processing)", "pf12k", []))
             if a.tests in ("both", "decode" if t[1] == "game4c" else "prefill")]
    camp = "quick-" + time.strftime("%F-%H%M")
    for label, test, extra in tests:
        print(f"quick: {label}, {len(builds)} build(s), n={a.n} ...", flush=True)
        subprocess.run([sys.executable, os.path.realpath(__file__), "bench", "--models", model, "--builds", ",".join(builds), "--variants", "default",
                        "-t", test, "-n", str(a.n), "--no-probe", "--campaign", f"{camp}-{test}"] + extra + (["--variant", "default=" + a.args] if a.args else []),
                       env=env, stdout=subprocess.DEVNULL if not a.verbose else None, stderr=subprocess.STDOUT if a.verbose else subprocess.DEVNULL)
    rows = []
    path = os.path.join(data or DATA, "run-history.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("campaign", "").startswith(camp) and r.get("ok") != "0"]
    def med(build, test, col):
        v = [float(r[col]) for r in rows if r["build"] == build and r["test"] == test and r.get(col)]
        return statistics.median(v) if v else None
    def cell(x):
        return "-" if x is None else f"{x:.1f}"
    def gain(f, st):
        return "-" if f is None or not st else f"{f / st:.2f}x"
    def sh(*cmd):
        try:
            r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=20, env=dict(os.environ, LD_LIBRARY_PATH=fork))
            return (r.stdout + (r.stderr if "--version" in cmd else "")).strip() if r.returncode == 0 else ""
        except Exception:
            return ""
    gpus = sh("nvidia-smi", "--query-gpu=name,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader").replace("\n", "; ")
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "") if os.path.exists("/proc/cpuinfo") else ""
    mem = next((int(l.split()[1]) // (1 << 20) for l in open("/proc/meminfo") if l.startswith("MemTotal")), 0) if os.path.exists("/proc/meminfo") else 0
    ver = sh(os.path.join(fork, "llama-server"), "--version")
    print()
    print(f"Model: {os.path.basename(model)}")
    print(f"Fork build: {next((l for l in ver.splitlines() if 'version' in l), '?')}")
    print(f"GPUs: {gpus or 'none found'}")
    print(f"CPU: {cpu}, RAM: {mem} GiB")
    print()
    print("| test | stock t/s | fork t/s | gain | stock pp t/s | fork pp t/s | pp gain |")
    print("|---|---|---|---|---|---|---|")
    for label, test, _ in tests:
        st_t, fk_t = med("stock-ref", test, "tps"), med("fork", test, "tps")
        st_p, fk_p = (med("stock-ref", test, "pp"), med("fork", test, "pp")) if test == "pf12k" else (None, None)
        print(f"| {label} | {cell(st_t)} | {cell(fk_t)} | {gain(fk_t, st_t)} | {cell(st_p)} | {cell(fk_p)} | {gain(fk_p, st_p)} |")
    if not rows:
        print("\nquick: no valid run (start the server by hand to see why: llama-server -m MODEL).")
    shutil.rmtree(stage, ignore_errors=True)
    if not a.record and data:
        shutil.rmtree(data, ignore_errors=True)


HUGE_NR = "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"
HUGE_FREE = "/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages"


def _sudo_write(path, val):
    r = subprocess.run(["sudo", "-n", "tee", path], input=str(val), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"hugepool: writing {path} needs root (sudo -n): {r.stderr.strip()}")


def cmd_branches(a):
    """Local branches not contained in TARGET (commits ahead, last commit date); --merge a,b merges them one by one, only inside a checkout of TARGET."""
    def git(*args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or a.repo, capture_output=True, text=True)
    cur = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if git("rev-parse", "--verify", a.target).returncode:
        sys.exit(f"no branch {a.target}")
    if a.merge:
        if cur != a.target:
            sys.exit(f"{a.repo} has {cur} checked out, not {a.target}: nothing merged (use --repo with the worktree of {a.target})")
        if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
            sys.exit("uncommitted changes in the checkout: nothing merged")
        for b in a.merge.split(","):
            r = git("merge", "--no-ff", "-q", "-m", f"{a.target}: merge {b}", b)
            if r.returncode:
                print(f"CONFLICT {b}: {' '.join(git('diff', '--name-only', '--diff-filter=U').stdout.split())}  (merge left open, resolve then commit)")
                return
            print(f"merged {b}")
        return
    rows = []
    for b in git("for-each-ref", "--format=%(refname:short)", "refs/heads", "--sort=-committerdate").stdout.split():
        if b == a.target:
            continue
        n = int(git("rev-list", "--count", f"{a.target}..{b}").stdout.strip() or 0)
        if n:
            rows.append((n, b, git("log", "-1", "--format=%cs", b).stdout.strip()))
    print(f"{len(rows)} local branches with commits not in {a.target} (checked out here: {cur})")
    for n, b, d in rows[:a.limit]:
        print(f"{n:5d}  {d}  {b}")

def _cell_cli(tok, wi):
    """an m111(log).sh command line as the equivalent run.py cell line (WRAP=... in front becomes --wrap)"""
    wrap = next((x[5:] for x in tok[:wi] if x.startswith("WRAP=")), None)
    rest = tok[wi + 1:wi + 7]
    return shlex.join(["python3", os.path.abspath(__file__), "cell"] + rest + (["--wrap", wrap] if wrap else []))
def cmd_queue(a):
    """Adds tasks to the bench queue the safe way: under the queue's own file lock (queued.py read-modify-writes tasks.csv; an edit without that lock is lost to its next
    rewrite - a hand edit was, once), after a backup of tasks.csv with the last 3 versions kept in queue/backups. `backups` lists them."""
    import fcntl, shutil
    Q = QDIR + "/queue"; INI = os.path.join(Q, "tasks.csv"); BK = os.path.join(Q, "backups")
    COLS = ["id", "order", "status", "campaign", "created", "start", "stop", "exit", "pid", "env", "cli", "log", "note"]
    if a.action in ("pause", "resume", "status"):
        # the daemon starts nothing while queue/STOP exists (a running task finishes first); no restart, no REST call needed
        stop = os.path.join(Q, "STOP")
        if a.action == "pause": open(stop, "a").close()
        if a.action == "resume" and os.path.exists(stop): os.remove(stop)
        rows = list(csv.DictReader(open(INI, newline="")))
        run_ = [r["id"] for r in rows if r["status"] == "running"]; nxt = sorted((r for r in rows if r["status"] == "next"), key=lambda r: float(r["order"] or 0))
        el = ""
        for r_ in rows:   # how long the running task has been going: a cell is normally 1-3 minutes, over 10 is flagged
            if r_["status"] == "running" and r_.get("start"):
                try:
                    sec = time.time() - time.mktime(time.strptime(r_["start"], "%Y-%m-%d %H:%M:%S"))
                    el += f" [{r_['id']} {int(sec // 60)}m{int(sec % 60):02d}s{' - UNUSUALLY LONG' if sec > 600 else ''}]"
                except ValueError:
                    pass
        print(("PAUSED: nothing new starts" if os.path.exists(stop) else "running: the next waiting task starts when the current one ends") + f" | running: {', '.join(run_) or '-'}{el} | waiting: {len(nxt)}" +
              (f" (next: {nxt[0]['id']})" if nxt else ""))
        return 0
    if a.action == "backups":
        for f in sorted(os.listdir(BK)) if os.path.isdir(BK) else []: print(os.path.join(BK, f))
        return 0
    if a.action == "del":
        # queue del TASK_ID | queue del all: remove tasks from tasks.csv (a running task is never removed: stop it first with `run.py stop`); same lock and backup as add
        if not a.task_id: sys.exit("queue del TASK_ID | queue del all")
        with open(INI + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            os.makedirs(BK, exist_ok=True)
            if os.path.exists(INI): shutil.copy2(INI, os.path.join(BK, "tasks-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"))
            for f in sorted(os.listdir(BK))[:-3]: os.remove(os.path.join(BK, f))
            rows = list(csv.DictReader(open(INI, newline="")))
            if a.task_id != "all" and not any(r["id"] == a.task_id for r in rows): sys.exit("no such task: " + a.task_id)
            keep, gone, busy = [], [], []
            for r in rows:
                if a.task_id in ("all", r["id"]):
                    (busy if r["status"] == "running" else gone).append(r["id"])
                    if r["status"] == "running": keep.append(r)
                else: keep.append(r)
            with open(INI + ".tmp", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader(); [w.writerow({c: r.get(c, "") for c in COLS}) for r in keep]
            os.replace(INI + ".tmp", INI)
        print(f"deleted {len(gone)} task(s)" + (": " + ", ".join(gone) if len(gone) <= 8 else "") + (f"; kept running: {', '.join(busy)}" if busy else ""))
        return 0
    if a.action == "add":
        if not a.task_id or not a.cli: sys.exit("queue add TASK_ID --cli '...'")
        new = [{"id": a.task_id, "order": a.order, "cli": a.cli, "note": a.note}]
    else:
        new = json.load(sys.stdin)
    with open(INI + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        os.makedirs(BK, exist_ok=True)
        if os.path.exists(INI): shutil.copy2(INI, os.path.join(BK, "tasks-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"))
        for f in sorted(os.listdir(BK))[:-3]: os.remove(os.path.join(BK, f))
        rows = list(csv.DictReader(open(INI, newline="")))
        have = {r["id"] for r in rows}
        # one task = one test: a queue task from m111.sh / m111log.sh with several tests is split into one task per test (a pause, a stop or a failure then
        # costs one cell, not the bundle); the id gets the test as suffix, order stays in sequence
        split = []
        for t in new:
            tok = shlex.split(t.get("cli", "")) if t.get("cli") else []
            wi = next((i for i, x in enumerate(tok) if os.path.basename(x) in ("m111.sh", "m111log.sh")), None)
            if wi is not None and len(tok) > wi + 5 and len(tok[wi + 5].split()) > 1:
                for k, test in enumerate(tok[wi + 5].split()):
                    u = dict(t); tk = list(tok); tk[wi + 5] = test; tk[wi + 1] = tok[wi + 1] + "-" + test
                    u["id"] = t["id"] + "-" + test; u["cli"] = _cell_cli(tk, wi); u["order"] = float(t.get("order", -100)) + 0.01 * k
                    split.append(u)
            elif wi is not None and len(tok) > wi + 5:
                u = dict(t); u["cli"] = _cell_cli(tok, wi); split.append(u)
            else:
                split.append(t)
        new = split
        for t in new:
            if t["id"] in have: print("exists, skipped:", t["id"]); continue
            r = {c: "" for c in COLS}; r.update(status="next", created=time.strftime("%F %T"), campaign=t["id"], log=os.path.join(Q, "logs", t["id"] + ".log")); r.update({k: str(v) for k, v in t.items()})
            rows.append(r); print("added:", t["id"])
        with open(INI + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader(); [w.writerow({c: r.get(c, "") for c in COLS}) for r in rows]
        os.replace(INI + ".tmp", INI)
    return 0
def run_list_help():
    """`run.py jobs --help`: the list mode (fork/runs/NAME.csv), the lists with their model names (the --only values) and the plain `run BUILD...` usage below it."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    print("run.py jobs -c NAME [-b B] [-onl TEXT[,TEXT]] [-s] [-dsc] [-nm] [-w] [-r N] [-n]   run the cells of fork/runs/NAME.csv (campaign list: gainers, reg, ctx)\n"
          "  -skp TEXT rows to skip, same matching as -onl (comma list)\n"
          "  -sc, -su each cell on the newest stock binary first, then on -b\n"
          "  -al      antiloop: repeat_penalty 1.15 over the last 256 tokens for every cell (same for stock and ours; changes the text, not the speed); also --antiloop before any subcommand\n"
          "  -nrt [DAYS]  skip cells already run in the last DAYS days (default 1 = today; runs.db); a stock pulled within those days is kept, no newer release is looked for\n"
          "  -b B     binary under devbuilds/ (default latest); the campaign of a row is B-<row name>\n"
          "  -onl TEXT rows whose name contains TEXT, comma list ok; the trailing '-' keeps glm- apart from glm35-; add i|io|o for one test\n"
          "  -s       newest official ggml-org release binary instead of -b: pulled once (gh, never built) into devbuilds/stock-bNNNNN-official, symlink stock-latest follows it\n"
          "  -w       warm-up pass first (same prompt again, stored as temp=warmpass); default off\n          -r N     measured runs per cell (default 2; -r 0 = the list's n)\n"
          "  -dsc, -from-biggest  biggest model first, one model's rows together; the pool is rebuilt to ceil(model GiB) before each new model\n"
          "  -nodl   no look for a newer stock release and no download: the stock on disk (stock-latest) is used\n"
          "  -a ARGS  extra cell arguments for our cells only, e.g. -a \"moe:split-f16=1\"; leave -nrt off for an A/B\n"
          "  -pl X4/X16  force the power limits in W for the whole run, e.g. -pl 220/280 (default: day 220/280, night 200/200 by the clock)\n"
          "  -from-smallest  smallest model first, so the first results come quickly\n"
          "  -nm      no mount: leave the pool alone. In a row's args column -nm means -lm mmap and no pool for that cell; `-nm 0.5` also pins that fraction of the model (rest mapped)\n"
          "  -n       print the cells, run nothing\n")
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        rows = list(csv.DictReader(open(f, newline="")))
        print(f"{os.path.basename(f)[:-4]}.csv: {len(rows)} cells; --only values, model file, pool GiB:")
        seen = {}
        for r in rows:
            m = re.sub(r"-(i|io|o)$", "", r["name"]) + "-"
            if m in seen: continue
            sz = sum(os.path.getsize(p) for p in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", r["model"]))) if os.path.exists(r["model"]) else 0
            seen[m] = 1
            print(f"  {m:<14} {os.path.basename(r['model']):<52} {-(-sz >> 30) if sz else '?'}")
    print("\nrun.py jobs BUILD... [-t TEST] [-n N] ...   measured runs of one test (below)\n")


CUR_HWL = [None]   # the machine line of the running list (hw_line()): the GPU class and count of the cells it runs, for the expected time of a cell


def gpu_class(hwl, hw=""):
    """the cards of a machine line ('... 5070,5x16,58g,pl250 5070,5x16,...'): their names in order, so '2 x 5070' differs from '1 x 5070' and from a 5090. Rows without a line (older pc1 rows) fall back to the machine id."""
    names = re.findall(r"(?<![\w.])(\w+),\d+x\d+", hwl or "")
    return tuple(names) if names else (hw or "",)


def cell_expect(model, test, n, build, warm=True):
    """the time of a cell from its own history; a sized prompt (16k1k, 7k1k) without history of its size takes the time of the 12k one of the model (a calib probe has none)"""
    try: test = best_test(model, test)   # pfbest1k -> pf<N>k1k: the model's own best prompt size, as the cell runs it (its history is stored under that test)
    except SystemExit: pass   # no best prompt stored yet: the names below still apply
    r = cell_expect_1(model, test, n, build, warm)
    return r if r is not None or not re.fullmatch(r"pf\d+k1k", test or "") or test == "pf12k1k" else cell_expect_1(model, "pf12k1k", n, build, warm)


def cell_expect_1(model, test, n, build, warm=True):
    """Seconds a cell usually takes: the span of earlier cells of this model and test in runs.db (load wait before the first run + first start to last end), the median of those
    with n runs (else scaled to n; the same build when it has rows, else any build). The discarded warm-up run (--warm) is not in runs.db: one median run is added for it. None without history."""
    try:
        import sqlite3
        from datetime import datetime
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")); n = int(n or 1)
        gen = ALIAS_TOKENS.get(test); test = ALIASES.get(test, test)   # an alias (pf12k1k) is stored under its test (pf12k); only runs with about its answer length count
        gq = " and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""
        f = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp()
        cls = gpu_class(CUR_HWL[0], (HWF[0] or "").split("'")[1] if "'" in (HWF[0] or "") else "") if CUR_HWL[0] else None; cls_off = [False]
        def spans(bq):
            d = {}
            for camp, st, en, rd, wl, hwid, hwl in c.execute("select campaign,start_ts,end_ts,coalesce(ready_s,0),coalesce(wall_s,0),hw,hwline from runs where model=? and test=? and start_ts is not null and end_ts is not null" + gq + bq[0], (os.path.basename(model), test) + bq[1]):
                rc = gpu_class(hwl, hwid)
                if cls and rc != cls and not cls_off[0]: continue   # a cell of another GPU class or count takes another time
                sc = (sum(gpu_speed(x) for x in rc) / sum(gpu_speed(x) for x in cls)) if cls and rc != cls else 1.0   # another class: the time follows the total card speed (rough)
                try: d.setdefault(camp, []).append((f(st), f(en), rd, wl, sc))
                except ValueError: pass
            out = []
            for v in d.values():
                if not v: continue   # every row of that campaign had an unreadable timestamp
                span = max(x[1] for x in v) - min(x[0] for x in v)
                if span > 3600 * 3: continue   # a cell split over hours (a rerun of the same tag) is no estimate
                one = statistics.median(x[3] for x in v) if warm else 0   # the discarded warm-up run takes about one run
                sc = v[0][4]
                out.append((len(v), ((span + min(v)[2]) + one if len(v) == n else (span + min(v)[2]) * n / len(v) + one) * sc))
            return out
        kind = (" and build like 'stock%'", ()) if str(build).startswith("stock") else (" and build not like 'stock%'", ())   # stock loads and runs quite differently from ours: a stock cell is estimated from stock cells (any stock build), ours from ours
        w = spans((" and build=?", (build,))) or spans(kind)
        if not w and cls: cls_off[0] = True; w = spans((" and build=?", (build,))) or spans(kind)   # no history on this class yet: any machine's is better than no bar
        if not w and gq: gq = ""; w = spans((" and build=?", (build,))) or spans(kind)   # no run of that length yet: the shorter runs of the test are better than no bar (the bar passes 100% and re-estimates)
        full = [x for k, x in w if k == n]
        return statistics.median(full or [x for _, x in w]) if w else None
    except Exception: return None


def col(code, text):
    """ANSI colour for the live display (terminal or detached log replay); plain text elsewhere. codes: 1 bold, 2 dim, 31 red, 32 green, 33 yellow, 36 cyan"""
    return f"\033[{code}m{text}\033[0m" if os.environ.get("RUN_DAEMON") or sys.stdout.isatty() else str(text)


def speed_txt(t, p=None, stock=False, width=0):
    """`12.85 t/s pp 169.4`: the number in the bright shade of its colour (decode light blue, prompt bright green), its label in the plain shade; stock's own numbers one shade down"""
    nb, lb = (94, 36) if stock else ("1;94", 96)   # bright and readable on a dark terminal: decode number blue, its label cyan; prompt number green, its label yellow
    nb2, lb2 = (92, 33) if stock else ("1;92", 93)
    out = col(nb, f"{t:{width}.2f}") + " " + col(lb, "t/s")
    if p is not None: out += " " + col(lb2, "pp") + " " + col(nb2, f"{p:{(width + 1) if width else 0}.1f}")
    return out


def ratio_col(x, text):
    """green at or above 1.00 (we are not slower), red below"""
    return col(32 if x >= 1 else 31, text)


def level(x, warn, bad, text):
    """text green below warn, yellow below bad, red from bad"""
    return col(32 if x < warn else 33 if x < bad else 31, text)


HOST = [None]   # machine dict while a list runs on another machine: hsh() goes there over ssh, everything else (hw_line, cell_status) is the same code


def hsh(cmd, timeout=20):
    """stdout of a shell command on the machine the cell runs on: here, or over ssh (HOST) -- the status line and the hardware line read /proc and nvidia-smi through this"""
    argv = ["ssh", "-o", "ConnectTimeout=5", HOST[0]["ssh"], cmd] if HOST[0] else ["bash", "-c", cmd]
    try: return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError): return ""


SYS_LAST = [None]   # the last good answer of the status probe
DEC_SEEN = {}   # decision name -> (value, reason) of the previous status tick
STATS_FINAL = [""]   # the engine JSON of the end of the cell (fetched by the cell itself), replaces the live fetch in the one status build after it
SYS_CMD = ("D=$(mktemp); (nvidia-smi dmon -c 1 -s t > $D 2>/dev/null &); head -1 /proc/stat; sleep 1; head -1 /proc/stat; echo '#'; cat /proc/meminfo; echo '#'; "
           "for i in 1 2 3; do o=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits 2>/dev/null); echo \"$o\" | grep -q 'N/A' || break; sleep 0.3; done; echo \"$o\"; echo '#'; "
           "for p in $(pgrep -x llama-server); do grep -E 'RssAnon|RssFile' /proc/$p/status; done; echo '#'; "
           "df -B1 --output=avail @ROOT 2>/dev/null | tail -1; echo '#'; "
           "P=$(ss -ltnp 2>/dev/null | awk '/llama-server/{n=split($4,a,\":\"); print a[n]; exit}'); [ -n \"$P\" ] && curl -s --max-time 2 localhost:$P/moe/stats; echo; echo '#'; cat $D; rm -f $D")   # the engine's own numbers, from the port its server listens on (local or remote)


def cell_status(t0, expect, label, total=None):
    """One status line: label, elapsed, progress bar against the expected time, CPU %, RAM GiB, pin (anon) and mm (file-backed) GiB of the server, swap MiB, per GPU util % and memory GiB."""
    def cpu():
        a = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return a[3] + a[4], sum(a)
    st = eng = ""   # -ns: no probe of the machine or the server, no status lines: only the bars below (the same code, the history estimate and the clock)
    if not NOSTATS[0]:
        sysi = hsh(SYS_CMD.replace("@ROOT", shlex.quote((HOST[0] or {}).get("root") or MODELS_DIR or QDIR)), 30).split("\n#\n") + ["", "", "", ""]   # this machine, or the one a remote list runs on: two /proc/stat samples a second apart, meminfo, GPUs, the server's memory
        if "MemTotal" not in sysi[1]:   # the probe did not answer (an ssh hiccup on a rented box, 30 s timeout): once more, then the last good answer; the cell must not die of a status tick
            sysi = hsh(SYS_CMD.replace("@ROOT", shlex.quote((HOST[0] or {}).get("root") or MODELS_DIR or QDIR)), 30).split("\n#\n") + ["", "", "", ""]
        if "MemTotal" in sysi[1]: SYS_LAST[0] = sysi
        elif SYS_LAST[0]: sysi = SYS_LAST[0]
        else: sysi = ["cpu 0 0 0 0 0\ncpu 0 0 0 0 0", "MemTotal: 1 kB\nMemAvailable: 1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB", "", "", "", "", ""]
        s0, s1 = [[int(x) for x in l.split()[1:]] for l in sysi[0].strip().split("\n")[:2]] if sysi[0].count("\n") else ([0] * 5, [0] * 5)
        busy = 100 * (1 - ((s1[3] + s1[4]) - (s0[3] + s0[4])) / max(sum(s1) - sum(s0), 1))
        m = {l.split(":")[0]: int(l.split()[1]) for l in sysi[1].split("\n") if ":" in l}
        ram = (m["MemTotal"] - m["MemAvailable"]) / 2**20; swap = (m["SwapTotal"] - m["SwapFree"]) / 1024
        g = sysi[2].split("\n")
        fl = lambda x: float(x) if re.fullmatch(r"-?[\d.]+", x) else 0.0   # nvidia-smi answers [N/A] for a reading it cannot take (power or temperature while the card is busy resetting)
        gpu = " ".join(col(36, f"g{i}") + " " + col(97, f"{v[0]}%" if fl(v[0]) or v[0].isdigit() else "-") + " " + col(34, f"{fl(v[1]) / 1024:.0f}/{fl(v[2]) / 1024:.0f}g") + " " + level(fl(v[3]) / max(fl(v[4]), 1), 0.5, 0.9, f"{fl(v[3]):.0f}/{fl(v[4]):.0f}W" if re.fullmatch(r"[\d.]+", v[3]) else "-") + " " + col(97, f"{fl(v[5]):.0f}C") for i, v in enumerate([[y.strip() for y in x.split(",")] for x in g if x.count(",") == 5]))
    el = time.time() - t0
    sf = server_frac()   # the server's own progress, once reliable, replaces the history's estimate for THIS cell; past the estimate with nothing better the estimate grows with the elapsed time
    def bar(done, ex, own=False, c=32, shown=None):   # done and ex in expected seconds: bar, percent, time left (never below 0:00; "+" once past the expectation)
        if own and sf and done > 10: ex = done / sf
        elif own and ex and done > ex: ex = done * 1.2
        if not ex: return f"[no history] {int(done // 60)}m"
        f = min(done / ex, 0.99); r = max(ex - done, 0)
        mm = lambda x: f"{int(x // 60)}:{int(x % 60):02d}".rjust(5)   # fixed width: the bars of the cell and the whole run line up
        return f"[{col(f'{c};40', '█' * int(f * 20))}{col('2;37;40', '░' * (20 - int(f * 20)))}] " + col(1, f"{f * 100:3.0f}%") + "  " + col(34, mm(ex)) + col(90, " - ") + col(34, mm(done if shown is None else shown)) + col(90, " = ") + col(94, mm(r)) + (col(31, "+") if done > ex else "")   # estimated total - elapsed = left
    if not NOSTATS[0]:
        pin = mm = 0   # the server's anonymous memory (the pinned weights buffer, KV tail, heap) and its file-backed memory (mmap'd weights), GiB; VmPin stays 0 for cudaHostRegister pages
        for l in sysi[3].split("\n"):   # RssAnon / RssFile lines of every llama-server there
            try: k, v = l.split(":")
            except ValueError: continue
            if k == "RssAnon": pin += int(v.split()[0]) / 2**20
            elif k == "RssFile": mm += int(v.split()[0]) / 2**20
        st = col(37, "cpu ") + level(busy, 50, 90, f"{busy:.0f}%") + col(37, " ram ") + level(ram / (m["MemTotal"] / 2**20), 0.7, 0.9, f"{ram:.0f}G") + " " + col(33, f"pin {pin:.0f}G") + " " + col(36, f"mm {mm:.0f}G") + " " + (col(31, f"swap {swap:.0f}M") if swap > 100 else col(37, f"swap {swap:.0f}M")) + (lambda fr: " " + level(1 - min(fr / 2**30 / 50, 1), 0.6, 0.9, f"disk {fr / 2**30:.0f}G free") if fr >= 0 else "")(int(sysi[4].strip() or -1) if sysi[4].strip().isdigit() else -1) + f" | {gpu}"   # stats first, the bars below
        eng = ""   # engine state line (GET /moe/stats): pass size, threads, context, cache hit rate, time per token, uploads against the probed link and RAM maximum
        try:
            es = json.loads(STATS_FINAL[0] or sysi[5]); sv = es.get("server") or {}; pr = es.get("prompt") or {}; tk = es.get("token_ms") or {}; bw = es.get("bandwidth") or {}; ht = es.get("hit") or {}
            if es.get("error") or not es.get("server"): raise ValueError   # loading (503) or a stock / older build (no route): no data, no engine lines
            lv_ = es.get("live") or {}
            if TRACE_T0[0] != t0: TRACE.clear(); TRACE_LAST.clear(); TRACE_H.clear(); TRACE_P.clear(); TRACE_T0[:] = [t0, 0]
            TRACE_T0[1] = time.time() - t0
            _h = ht.get("prompt_recent", -1) if lv_ and ht.get("prompt_recent", -1) >= 0 else ht.get("recent", -1)   # the hit rate of the moment: the prompt pass while one runs
            if _h is not None and _h >= 0: TRACE_H.append((time.time() - t0, _h))
            TRACE_P.append((time.time() - t0, "P" if lv_ else "D"))   # prompt call running, else decode
            cur = {"threads": f"{sv.get('threads')}/{sv.get('threads_batch')}", "ubatch": (es.get("live") or {}).get("ubatch") or pr.get("ubatch") or sv.get("n_ubatch")}
            for d in es.get("decisions") or []: cur["decided " + d["name"]] = d["value"]
            for k_, v_ in cur.items():
                if v_ is not None and TRACE_LAST.get(k_) != v_: TRACE.append((time.time() - t0, k_, TRACE_LAST.get(k_), v_)); TRACE_LAST[k_] = v_
            rx = []   # nvidia-smi dmon, one second sample: host->device MB/s per card (the real link traffic, also while a prompt streams the experts, which the engine counters do not see)
            for ln in sysi[6].splitlines():
                f = ln.split()
                if len(f) >= 3 and f[0].isdigit() and f[1].isdigit(): rx.append(int(f[1]) / 1000)
            lv = es.get("live") or {}
            LIVE_DONE[0] = lv.get("done") or 0   # a prompt call is running: its pass size, not the finished one before it
            LIVE_PASS[:] = [lv.get("pass") or 0, lv.get("passes") or 0] if lv.get("passes", 0) > 1 else [0, 0]   # passes finished / passes of the call (engine builds with the per-pass log): the "now:" line shows "pass 2/3 running"
            ub = lv.get("ubatch") or pr.get("ubatch") or sv.get("n_ubatch")
            parts = []
            io = []   # the transfer rates on a line of their own: on the engine line it wrapped once "pp by size" listed two sizes (user 2026-10-25: stale cpu/engine rows, 1-row jumps)
            if ub:   # max = the real maximum: the engine's pass ceiling (VRAM: what its compute buffer holds) or, when smaller, the pass the RAM allows (decision "prompt pass")
                _mx, _why = (int(pr["ceiling"]), "VRAM") if pr.get("ceiling") else (None, "")
                for _d in es.get("decisions") or []:
                    _m = re.match(r"(\d+) tokens \(cut from", str(_d.get("value") or "")) if _d.get("name") == "prompt pass" else None
                    if _m and (_mx is None or int(_m.group(1)) < _mx): _mx, _why = int(_m.group(1)), "RAM"
                parts.append(col(36, f"ubatch {ub}") + (col(90, f" (max {_mx} {_why})") if (lv.get("ubatch") or pr.get("ubatch")) and _mx else ""))
            if sv.get("threads") is not None: parts.append(col(36, f"thr {sv.get('threads')}/{sv.get('threads_batch')}"))
            if sv.get("n_ctx_slot") is not None: parts.append(col(36, f"ctx {sv.get('n_ctx_slot')}"))
            if ht and lv and ht.get("prompt_recent", -1) >= 0: parts.append(level(1 - ht["prompt_recent"] / 100, 0.15, 0.4, f"prompt hit {ht['prompt_recent']:.1f}%") + col(90, f" (call so far {ht.get('prompt_all', 0):.1f}%)"))   # the experts of the running prompt pass found in the cache
            elif ht: parts.append(level(1 - ht.get("all", 0) / 100, 0.15, 0.4, f"hit {ht.get('all', 0):.1f}%") + col(90, f" (now {ht.get('recent', 0):.1f}%)"))
            if tk.get("tokens"): parts.append(col(94, f"{tk['last64']:.1f} ms/tok") + col(90, f" (last {tk['last']:.1f}, all {tk['all']:.1f})"))
            if lv.get("done") and lv.get("s"): parts.append(col(32, f"pp {lv['done'] / lv['s']:.0f} t/s") + col(90, f" running: {lv['done']}/{lv['tokens']} tok in {lv['s']:.0f} s"))   # the running prompt call, live
            elif pr.get("last_tps"): parts.append(col(32, f"pp {pr['last_tps']:.0f} t/s") + (col(90, " by size " + " ".join(f"{k}:{v:.0f}" for k, v in (pr.get("tps_by_ubatch") or {}).items())) if pr.get("tps_by_ubatch") else ""))
            if rx and lv: io.append(col(37, "pcie rx ") + " ".join(f"g{i} {v:.1f} GB/s" for i, v in enumerate(rx)))
            if bw:   # RAM read rate (CPU experts + uploads) and each PCIe link, in GB/s against the probed maximum, and as a share of it
                pc = lambda v, mx: 100 * v / mx if mx else 0
                rp = pc(bw.get("ram_gbs", 0), bw.get("ram_max_gbs", 0))
                io.append(col(37, "mem bw ") + level(rp / 100, 0.7, 0.9, f"{bw.get('ram_gbs', 0):.1f}/{bw.get('ram_max_gbs', 0):.0f} GB/s {rp:.0f}%"))
                try: io.append(col(37, "ram free ") + level(1 - mem_avail_gib() / max(1, ram_total_gib()), 0.85, 0.95, f"{mem_avail_gib():.0f}G"))
                except (OSError, ValueError): pass
                io.append(col(37, "pcie ") + " ".join(level(pc(l["gbs"], l["max_gbs"]) / 100, 0.7, 0.9, f"l{l['link']} {l['gbs']:.1f}/{l['max_gbs']:.0f} GB/s {pc(l['gbs'], l['max_gbs']):.0f}%") for l in bw.get("links", [])))
            eng = " ".join(parts) + ("\n" + " ".join(io) if io else "")
            sp = es.get("split") or {}; links = bw.get("links", []) if bw else []
            # cards that COMPUTE at the same time (user 2026-10-23): only a split prompt pass runs on several cards at once; every other pass and every decode token
            # runs its layers on one card after another (layer split), however many cards hold cache layers (shown apart as "cache on N cards")
            pgpus = sp.get("prefill_gpus", 0) if sp.get("prefill") == 1 else 1
            ncache = bw.get("gpus") if bw else sp.get("gpus")
            up = sum(1 for l in links if l.get("gbs", 0) > 0.05)
            mode = [col(32, "moe cache") if bw else col(33, "stock placement")]
            mode.append(col(32, f"dense split ON {sp.get('devs', '')}") if sp.get("prefill") == 1 else col(33, "dense split off (one card)") if sp.get("prefill") == 0 else col(90, "dense split n/a"))
            mode.append(col(36, f"par gpus prompt {pgpus or '?'} decode 1") + (col(90, f"  cache on {ncache} cards") if ncache else "") + (col(35, f" uploading on {up}/{len(links)} links") if links else ""))
            eng = eng + "\n" + "  ".join(mode) + "\n" + kv_line()
            if sp.get("why"): eng += "\n" + col(90, sp["why"])   # own line: beside the mode it passed the screen width, the wrapped line broke the redraw (cpu row repeated, user 2026-10-25)
            if CELL_REF[0] and (best := best_ref(*CELL_REF[0])):   # live speeds against the best earlier run of this model, test and power limits (runs.db)
                bpp, bpb, btg, btb = best
                if LAST_PP.get("t0") != t0: LAST_PP.clear(); LAST_PP["t0"] = t0
                if lv.get("done") and lv.get("s") and lv["done"] >= 64: LAST_PP["v"] = lv["done"] / lv["s"]   # a prompt call is running; the 4-token calls of decode say nothing about the prompt speed
                lpp = LAST_PP.get("v")   # (the last prompt speed of this cell stays on the line while it decodes)
                ltg = 1000 / tk["last64"] if tk.get("tokens") and tk.get("last64") else None
                gain = lambda v, b: (col(32 if v >= b else 91, f"{100 * (v / b - 1):+.0f}%") if v and b else col(90, "-"))
                eng += ("\n" + col(37, "vs best ") + col(32, f"pp {lpp:.0f}" if lpp else "pp -") + col(90, f" / {bpp:.0f} ({bpb}) ") + col(90, f"{1000 / lpp:.2f} / {1000 / bpp:.2f} ms/tok " if lpp else "") + gain(lpp, bpp)
                        + "   " + col(94, f"tt {ltg:.1f}" if ltg else "tt -") + col(90, f" / {btg:.1f} ({btb}) ") + col(90, f"{1000 / ltg:.1f} / {1000 / btg:.1f} ms/tok " if ltg else "") + gain(ltg, btg))
            bal = sp.get("bal")
            cur0 = sp.get("cur") or {}
            if not (bal and bal.get("cards")) and cur0.get("cards"):   # the first split pass of a run, before any pass is measured
                cap0 = cur0.get("cap_max") or cur0.get("cap") or 0   # of the context's card limit (one maximum per card), not the pass's cut one
                eng += "\n" + col(37, "split   pass  1 ") + col(93, f"running {cur0.get('age_ms', 0) / 1e3:.0f} s:") + "".join(
                    f"  {card_label(dict(c, experts=i == len(cur0['cards']) - 1))} {c.get('tok', 0)} tok" + (f" ({100 * c.get('tok', 0) / cap0:.0f}% of limit)" if cap0 else "")
                    for i, c in enumerate(cur0["cards"]))
            if bal and bal.get("cards"):   # the live balance, any number of cards (engine 2026-10-23 N-card): one line per card of the last split pass, the expert card last
                cs = bal["cards"]
                sig = (bal.get("n"), bal.get("us_tok"), tuple(c.get("tok", 0) for c in cs))
                # A new cell: the server is kept between cells, so the pass it reports at the cell's start is the LAST CELL's last pass (user 2026-10-24: the 12k
                # cell's 13786-token pass showed as pass 1 of the 60k prompt). That one is remembered and never listed; the rows are this prompt's passes only.
                # The engine stamps each pass with its wall time (bal.t_ms, 2026-10-24): the report is the last cell's only when measured before this cell began.
                # Without the stamp (older builds) the first report of the cell is taken as old, which dropped the first pass of a fresh server (GLM 60k on
                # spass30: 6 of its 7 passes stored).
                if SPLIT_PASSES["t0"] != t0: SPLIT_PASSES["t0"] = t0; SPLIT_PASSES["rows"] = []; SPLIT_PASSES["peak"] = {}; SPLIT_PASSES["old"] = sig if not bal.get("t_ms") or bal["t_ms"] / 1e3 < t0 else None
                for l in bw.get("links") or []: SPLIT_PASSES.setdefault("peak", {})[l.get("dev")] = max(SPLIT_PASSES.get("peak", {}).get(l.get("dev"), 0), l.get("gbs", 0))   # the busiest second of each card's host link in this cell
                if sig != SPLIT_PASSES.get("old") and (not SPLIT_PASSES["rows"] or SPLIT_PASSES["rows"][-1][0] != sig): SPLIT_PASSES["rows"].append((sig, bal)); PASS_KV.setdefault(sig, kv_tier())   # a new measured pass
                eng += ("\n" + col(37, "split ") + col(32, bal.get("phase", "")) + col(90, f"  pass {bal.get('us_tok', 0):.1f} us/token")
                        + (col(90, f", dense alone {bal['ref_us_tok']:.1f}") if bal.get("ref_us_tok") else "") + col(90, f"  ({bal.get('last', '')})"))
                # One row per card per pass of this run as each is measured, then the pass running now (user 2026-10-24: every pass, errors must show while the run
                # goes). Cards by card_label: pcuda1x16 (p: the primary card, the experts and its full card limit every pass), cuda0x4. Every card's pass by cause,
                # measured with stream stamps by the engine (--moe pass-stats, run.py turns it on unless -ns): compute (kernels) + upload (waiting for its uploads
                # from host memory, whatever they carry) + wait (for another card's result) + stall (nothing queued: the host, or before its first / after its last
                # split) = total, the pass. link: its upload stream busy (overlaps compute); GB host / GB card: what crossed its link into it. Without pass-stats
                # (-ns, older builds) the upload waits sit in compute, every gap counts as wait, and upload / link / GB show "-".
                PW = [("tokens", 6), ("%limit", 6), ("compute", 7), ("%", 4), ("upload", 6), ("%", 4), ("wait", 6), ("%", 4), ("stall", 6), ("%", 4),
                      ("t/s", 6), ("link", 6), ("%", 4), ("GB host", 7), ("GB card", 7), ("vs best", 7), ("us/tok", 6), ("card t/s", 8)]   # t/s: the whole pass (first row); us/tok: this card's compute time per token, rising as the context grows; card t/s: this card's tokens over the pass time (the cards add up to t/s)
                # "vs best" (user 2026-10-24): the pass's time per token against the best earlier run of ours of this model and test (any other build; run.py has per-run
                # speeds, not per-pass ones, so the reference is that run's average time per token): above 1 is faster than the best so far, below 1 slower.
                bk = (CELL_REF[0][0], CELL_REF[0][1], CELL_BUILD[0], CELL_ASKED[0]) if CELL_REF[0] else None
                if bk and bk not in BEST_MEMO: BEST_MEMO[bk] = best_prev(*bk)
                bestus = 1e6 / BEST_MEMO[bk][1] if bk and BEST_MEMO.get(bk) and BEST_MEMO[bk][1] else None
                def vsb(bk, pos, T, span):   # the pass's time per token against the best earlier one at the same cells (interpolated between the passes we know): above 1 is faster
                    y = pass_best_at(bk[0], bk[2], pos)
                    return f"{y * T / (span * 1e6):.2f}" if y and span and T else "first"
                def pass_rows(k, b, lw, k_sig=None, pos=0):
                    pc = sorted(b.get("cards") or [], key=lambda c: not c.get("experts"))
                    T = sum(c.get("tok", 0) for c in pc)
                    span = (b.get("span_ms") or b.get("us_tok", 0) * T / 1e3) / 1e3
                    det = bool(b.get("detail"))
                    pcf = lambda v: f"{100 * v / span:.0f}%" if span else "-"
                    out = []
                    for i, c in enumerate(pc):
                        comp = (c.get("dense_ms", 0) + c.get("moe_ms", 0)) / 1e3
                        if det:
                            up, wt, st = c.get("upw_ms", 0) / 1e3, c.get("xw_ms", 0) / 1e3, c.get("stall_ms", 0) / 1e3
                        else:
                            up, wt = 0.0, (c.get("wait_dense_ms", 0) + c.get("wait_moe_ms", 0)) / 1e3
                            st = max(0.0, span - comp - wt)
                        ln = c.get("link_ms", 0) / 1e3
                        limp = c.get("lim_pct") or (100 * c.get("tok", 0) / CARD_LIM[0] if CARD_LIM[0] else 0)   # the engine fills lim_pct only in the forced pass loop: else of the card limit it reports
                        vals = [str(c.get("tok", 0)), f"{limp:.0f}%", f"{comp:.1f}s", pcf(comp),
                                f"{up:.1f}s" if det else "-", pcf(up) if det else "-", f"{wt:.1f}s", pcf(wt), f"{st:.1f}s", pcf(st), f"{T / span:.0f}" if i == 0 and span else "",
                                f"{ln:.1f}s" if det else "-", pcf(ln) if det else "-",
                                f"{c.get('gb_up', 0) + c.get('gb_host', 0):.2f}" if det else "-", f"{c.get('gb_in', 0):.2f}" if det else "-",
                                (vsb(bk, pos, T, span) if bk else "") if i == 0 else "",
                                f"{comp * 1e6 / c['tok']:.0f}" if c.get("tok") else "-", f"{c.get('tok', 0) / span:.0f}" if span else "-"]   # best time per token / this pass's; "first" without an earlier run
                        cells = [v.rjust(w) for v, (_, w) in zip(vals, PW)]   # header labels are right-aligned to the same widths: the two cannot drift
                        hot = lambda v: span and v / span > 0.05
                        colour = {1: 36, 4: 94, 5: 94, 6: (91 if hot(wt) else 92) if c.get("experts") else 33, 7: (91 if hot(wt) else 92) if c.get("experts") else 33,
                                  8: 91 if hot(st) else 92, 9: 91 if hot(st) else 92, 11: 90, 12: 90, 13: 90, 14: 90, 15: (90 if vals[15] in ("", "first") else 91 if float(vals[15]) < 0.99 else 92 if float(vals[15]) > 1.01 else 37)}
                        cells = [col(colour[j], v) if j in colour else v for j, v in enumerate(cells)]
                        head = f"  {k:>4} {T:>6} {f'{span:.1f}s':>6}" if i == 0 else " " * 20
                        out.append(head + "  " + col(1, card_label(c).ljust(lw)) + " " + " ".join(cells))
                    return out
                rows = [r for r in SPLIT_PASSES["rows"] if sum(c.get("tok", 0) for c in (r[1].get("cards") or [])) not in (0, 512)]   # the startup calibration's calls are exactly 512 tokens: not the prompt. Every other pass is listed, however small (user 2026-10-24: 506-token passes were hidden by a "> 512" test)
                lw = max([4] + [len(card_label(c)) for _, b in rows for c in (b.get("cards") or [])])
                if any("sc_compute" in c for c in cs):   # the cards' speeds measured once at server start (probe: a sustained matmul burst, a pinned upload), best card = 1.00
                    eng += ("\n" + col(37, "  card speeds (measured at start, best card = 1.00): ")
                            + col(90, " | ").join(col(1, card_label(c)) + col(36, f" compute {c.get('sc_compute', 0):.2f}") + col(94, f" upload {c.get('sc_upload', 0):.2f}")
                                                  for c in sorted(cs, key=lambda c: not c.get("experts"))))
                if any("zw_k" in c for c in cs):   # the zero-wait rule (engine 2026-10-24): each other card's tokens = k x factor x the primary card's
                    prim = next((card_label(c) for c in cs if c.get("experts")), "the primary card")
                    eng += ("\n" + col(37, "  next pass: ") + col(90, " | ").join(
                                col(1, card_label(c)) + col(93, f" {c.get('zw_k', 0) * c.get('zw_f', 0):.2f} x {prim}'s tokens") + col(90, f" (k {c.get('zw_k', 0):.2f} x factor {c.get('zw_f', 0):.2f})")
                                for c in cs if not c.get("experts")) + col(90, f"  k from {bal.get('zw_src', '?')}"))
                cl, pl, cm = (sp.get("cur") or {}).get("cap") or 0, (sp.get("cur") or {}).get("pass_lim") or 0, (sp.get("cur") or {}).get("cap_max") or 0
                CARD_LIM[0] = cm or cl or CARD_LIM[0]
                if rows:
                    prim = next((card_label(c) for c in rows[-1][1].get("cards") or [] if c.get("experts")), "the primary card")
                    if cl:   # the two limits once (user 2026-10-24): what one card holds (its range, its buffers) and what a pass of all cards takes
                        eng += ("\n" + col(37, "  limits: ") + (col(36, f"card max {cm} tokens") + col(90, " (the context's micro-batch: %limit is of this one number)") + col(37, " | ") if cm else "")
                                + col(36, f"card now {cl} tokens") + col(90, " (what the pools hold beside the KV now)" if cm else " (one card's micro-batch; %limit of each pass is of the limit that pass ran under)")
                                + "\n" + col(37, "  limits: ") + col(36, f"pass {pl} tokens" if pl else "pass -") + col(90, f" (all cards; {prim} takes its card limit)"))   # own line: the one line passed the screen width (user 2026-10-25)
                    # the global wait sum (user 2026-10-25: "judge from the sum of wait times, keep the global sum visible"): every measured pass of this run,
                    # per card: its waiting (for its dense / expert inputs and for another card) and its whole time on the cards; the share of waiting is the
                    # efficiency to compare between builds
                    wsum, tsum, per, wall = {}, 0.0, {}, 0.0
                    for _, bb in rows:
                        for c in (bb.get("cards") or []):
                            wt = (c.get("xw_ms", 0) if c.get("xw_ms") is not None and "xw_ms" in c else (c.get("wait_dense_ms", 0) + c.get("wait_moe_ms", 0))) / 1e3
                            k_ = card_label(c)
                            wsum[k_] = wsum.get(k_, 0.0) + wt
                            per[k_] = per.get(k_, 0) + 1
                        tsum += (bb.get("span_ms") or 0) / 1e3 * max(1, len(bb.get("cards") or []))
                        wall += (bb.get("span_ms") or 0) / 1e3   # wall time of the passes (the cards overlap inside a pass)
                    tot_w = sum(wsum.values())
                    eng += ("\n" + col(37, f"  waits so far: {len(rows)} passes, " + ", ".join(f"{k} {v:.1f} s" for k, v in wsum.items())
                            + f", all cards {tot_w:.1f} s" + (f" = {100 * tot_w / tsum:.0f}% of the cards' time" if tsum > 0 else "")
                            + f"; passes wall {wall:.1f} s"))
                    eng += "\n" + col(37, "  pass tokens   time  " + "card".ljust(lw) + " " + " ".join(n.rjust(w) for n, w in PW))
                cum, run_tok = [], 0   # the cells each pass is centred on: the tokens of the passes before it plus half its own
                for _, b_ in rows:
                    t_ = sum(c_.get("tok", 0) for c_ in (b_.get("cards") or [])); cum.append(run_tok + t_ / 2); run_tok += t_
                npass = len(rows) if not sys.stdout.isatty() else max(6, (shutil.get_terminal_size((100, 40)).lines - 24) // 2)   # every pass (user 2026-10-25: keep all) as far as the terminal is tall: a taller block scrolls it and the whole screen repaints, so the oldest drop off first
                # runs (user 2026-10-25: "add separator, count from 1 again"): a pass whose centre is lower than the one before starts a new request (the same prompt
                # again, or the next cell); each run gets a separator line and its passes are numbered from 1 again
                run_no, k_run, prev_c = 1, 0, None
                numbered = []
                for idx, (sg, b) in enumerate(rows):
                    c_ = cum[idx]
                    if prev_c is not None and c_ < prev_c:
                        run_no += 1
                        k_run = 0
                    prev_c = c_
                    k_run += 1
                    numbered.append((run_no, k_run, sg, b, c_))
                shown = numbered[-npass:]
                last_run = None
                for run_i, k, sg, b, c_ in shown:
                    if run_i != last_run and (last_run is not None or run_no > 1):
                        eng += "\n\n" + col("1;90", f"  ======== run {run_i} " + "=" * 60)
                    last_run = run_i
                    eng += "".join("\n" + l for l in pass_rows(k, b, lw, sg, c_))
                cur = sp.get("cur") or {}
                if cur.get("cards"):   # the pass running now (the ranges of the split graph built last); after the last pass it equals that pass and ages out
                    cap = cur.get("cap_max") or cur.get("cap") or 0   # % of the context's card limit
                    ctok = tuple(c.get("tok", 0) for c in cur["cards"])
                    lb = rows[-1][1] if rows else {}
                    ltok = tuple(c.get("tok", 0) for c in (lb.get("cards") or []))
                    lspan = lb.get("us_tok", 0) * sum(ltok) / 1e3 if lb else 0
                    if not rows or set(ctok) != set(ltok) or cur.get("age_ms", 0) < lspan:
                        last = len(cur["cards"]) - 1   # the expert card is the last range
                        eng += "\n" + f"  {len(rows) + 1:>4} " + col(93, f"running {cur.get('age_ms', 0) / 1e3:.0f}s") + "".join(
                            f" | {card_label(dict(c, experts=i == last))} {c.get('tok', 0)}" + (f" {100 * c.get('tok', 0) / cap:.0f}%" if cap else "") for i, c in enumerate(cur["cards"]))
            elif bal:   # the live balance of the dense split (engine 2026-10-23, two cards): each card's dense and MoE time of the last split pass, its share of the tokens and experts
                h, o = bal.get("helper") or {}, bal.get("owner") or {}
                card = lambda c, ts, fs: (col(1, f"{c.get('dev', '?')} ") + col(36, f"{100 * ts:.0f}% tokens {100 * fs:.0f}% experts") + f"  dense {c.get('dense_ms', 0):.0f} ms "
                                         + col(93, f"{c.get('dense_us_tok', 0):.1f} us/token") + f"  moe {c.get('moe_ms', 0):.0f} ms")
                late = h.get("dense_ms", 0) - o.get("dense_ms", 0)
                eng += ("\n" + col(37, "split ") + col(32, bal.get("phase", "")) + "  " + card(h, bal.get("s", 0), bal.get("f", 0)) + col(90, " | ") + card(o, 1 - bal.get("s", 0), 1 - bal.get("f", 0))
                        + col(90, f" | pass {bal.get('us_tok', 0):.1f} us/token") + (col(90, f", dense alone {bal['ref_us_tok']:.1f}") if bal.get("ref_us_tok") else "")
                        + "\n" + col(37, "      ") + col(91 if abs(late) > 0.03 * max(h.get("dense_ms", 0), o.get("dense_ms", 0), 1) else 90, f"{'helper' if late > 0 else 'owner'} dense late by {abs(late):.0f} ms")
                        + col(90, f"  next: {100 * bal.get('s_next', 0):.0f}% tokens {100 * bal.get('f_next', 0):.0f}% experts  ({bal.get('last', '')})"))
            dec = es.get("decisions") or []
            if dec:   # what the engine chose by itself and why, as it happens: a placement of "stock" for a model bigger than the VRAM stands out in yellow
                # one line per decision with its whole reason (fit_block cuts at the terminal width); a decision that changed since the last tick is bold, so the tuners can be watched working
                for d in dec:
                    chg = DEC_SEEN.get(d["name"]) not in (None, (d["value"], d["why"]))
                    DEC_SEEN[d["name"]] = (d["value"], d["why"])
                    bad = d["name"] == "placement" and "stock" in d["value"] and "fits" not in d["value"]
                    why = "" if d["name"] == "threads_decode" else f"  {d['why']}"   # the decode thread search: its count only, its reason text is noise here (user 2026-10-27)
                    eng += "\n" + col(37, "decided") + " " + col("1;93" if chg else 33 if bad else 32, f"{d['name']} {d['value']}") + col("1;37" if chg else 90, why)
        except (ValueError, KeyError, TypeError, AttributeError): pass
        if eng: st = st + "\n" + eng
        if eng or not LAST_ST[0]: LAST_ST[0] = st   # (the last tick WITH engine data: the tick after the server is gone shows an idle machine and would replace it)
    # the machine / engine lines of the last tick: printed under the finished cell, they stay in the scrollback
    now = time.time(); pk = PEEK[0]   # phase of the cell: loading the model (yellow), prompt (green), generation (blue); each phase's time once it is known
    try: ld_s = float(open(LOAD_FILE).read()); ld_end = os.path.getmtime(LOAD_FILE)
    except (OSError, ValueError): ld_s = ld_end = None
    if PHASE.get("t0") != t0: PHASE.clear(); PHASE["t0"] = t0
    if pk and pk[1] and "gen" not in PHASE: PHASE["gen"] = now   # first tick that sees tokens decoded
    ph = 34 if "gen" in PHASE else 32 if ld_end else 33
    tf = lambda x: f"{x:.0f}s" if x < 60 else f"{int(x // 60)}m{int(x % 60):02d}s"
    pt = ([col(33, "l " + tf(ld_s))] if ld_s is not None and ld_s >= 0 else []) + ([col(32, "p " + tf(PHASE.get("gen", now) - ld_end))] if ld_end else []) + ([col(34, "g " + tf(now - PHASE["gen"]))] if "gen" in PHASE else [])
    LW = max(NAMEW[0], 10)   # label column of the cell and whole-run lines
    l1 = f"{col(1, f'{label[:LW]:<{LW}}')} {bar(el, expect, True, ph)}" + ("  " + " ".join(pt) if pt else "") + ("  " + col(36, CELL_TEST[0]) if CELL_TEST[0] else "")
    ex0 = expect or 0   # the whole-run bar follows the cell's re-estimate
    if sf and el > 10: expect = el / sf
    elif expect and el > expect: expect = el * 1.2
    rp = os.environ.get("RUNPY_REPRO"); rl = (col(90, "repro ".ljust(LW) + " " + rp) + "\n") if rp else ""   # the command line of this run, under every status: copy it to repeat the run
    if not total: return (st + "\n" if st else "") + rl + l1
    done_s, total_s, run_t0, k, n = total   # expected seconds done / in all, start of the whole run, cell number k of n
    total_s += (expect or 0) - ex0
    tel = time.time() - run_t0; part = done_s + min(el, expect or 0)
    return (st + "\n" if st else "") + rl + l1 + "\n" + col(33, "whole run".ljust(LW)) + f" {bar(part, total_s, shown=tel)}  " + col(33, f"cell {k}/{n}")


def cell_results(path, since):
    """(t/s, pp t/s) of every measured run the cell has written to its .run.out since `since` (epoch s): the lines `DATE BUILD TEST {args} 56.48 t/s hit 99.2% pp 132.6 t/s ...`."""
    out = []
    try:
        for l in open(path, errors="replace"):
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ +\S+ +\{.*?\} +([\d.]+) t/s.*?pp ([\d.]+) t/s", l)
            if m and time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) >= since - 2: out.append((float(m.group(2)), float(m.group(3))))
    except OSError: pass
    return out


def kill_tree(pid, grace=8):
    """SIGTERM to pid and every descendant (the cell, its bench, the server: `timeout` in between is a process group of its own, so the terminal's Ctrl-C never reaches them), SIGKILL for what is left after `grace` s."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try: kids.setdefault(int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]), []).append(int(d))
            except (OSError, IndexError, ValueError): pass
    todo, tree = [pid], []
    while todo:
        q = todo.pop(); tree.append(q); todo += kids.get(q, [])
    live = lambda: [q for q in tree if os.path.exists(f"/proc/{q}") and PidProc(q).poll() is None]
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10), (signal.SIGKILL, 30)):   # a server stuck in page faults or a batch ignores SIGTERM: SIGKILL after `grace`, repeated until gone
        for q in reversed(tree):   # the servers first
            try: os.kill(q, sig)
            except OSError: pass
        t0 = time.time()
        while time.time() - t0 < wait and live(): time.sleep(0.5)
        if not live(): return
    print(f"kill_tree: still alive after SIGKILL: {live()} (uninterruptible I/O: a reboot-free fix is not known)", flush=True)


def gen_expect(model, test):
    """(median generated tokens, median prompt tokens) of this model's and test's earlier runs in runs.db, None for what has no history."""
    try:
        import sqlite3
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        gen = ALIAS_TOKENS.get(test); test = ALIASES.get(test, test)
        q = "select n_gen, note from runs where model=? and test=? and ok=1 and n_gen>0%s order by start_ts desc limit 20"
        rows = c.execute(q % (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""), (os.path.basename(model), test)).fetchall() or c.execute(q % "", (os.path.basename(model), test)).fetchall()
        pn = [int(m.group(1)) for _, nt in rows if nt and (m := re.search(r"prompt_n=(\d+)", nt))]
        return (float(gen) if gen else statistics.median(r[0] for r in rows) if rows else None), (statistics.median(pn) if pn else None)
    except Exception: return None, None


PHASE = {}   # per cell: t0 and when the first decoded token was seen (cell_status colours and times the phases)
LAST_ST = [""]
NOSTATS = [os.environ.get("RUN_NS") == "1"]   # -ns: nothing that samples the machine or polls the server while a cell runs (no status probes, no nvidia-smi / sys / slots samplers on a remote box): the timings are the engine's alone
TRACE, TRACE_LAST, TRACE_T0, TRACE_H, TRACE_P = [], {}, [None], [], []   # one cell's tuning timeline: (seconds into the cell, knob, old, new), from the status ticks
LIVE_PASS = [0, 0]   # (passes finished, passes of the call) of the running prompt call, from /moe/stats live; [0, 0] = one pass or an older engine
LIVE_DONE = [0]   # tokens of the running prompt call the engine has finished (its own /moe/stats live block): the server counts only after a whole decode call
PEEK = [None]   # (prompt tokens done, tokens decoded, expected decoded, expected prompt) of the last /slots answer: cell_status re-estimates the cell's end from it


DEC_RATE = [None]   # decode tokens/s earlier runs of this cell's model and test reached (runs.db), set per cell: the share of a cell's time that is prompt depends on the model's own speeds


def dec_rate(model, test):
    try:
        import sqlite3
        gen = ALIAS_TOKENS.get(test); test = ALIASES.get(test, test)
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        q = "select tps from runs where model=? and test=? and ok=1 and tps>0%s order by start_ts desc limit 20"
        rows = c.execute(q % (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""), (os.path.basename(model), test)).fetchall() or c.execute(q % "", (os.path.basename(model), test)).fetchall()
        return statistics.median(r[0] for r in rows) if rows else None
    except Exception: return None


def server_frac():
    """Share of the cell's time the server reports done (0..1), None until it is reliable (5 % of the prompt, or any decoding, with a history to compare to). While the prompt runs its speed is
    measured (tokens so far over the time since the load ended) and the decode time comes from the earlier runs' decode speed (DEC_RATE): a 12k prompt at 33 t/s plus 1k tokens at 10.7 t/s is
    59 % prompt, not the 45 % that a fixed weighting of 1:15 per token gave. Without a decode speed in the history nothing is guessed (see below)."""
    if NOSTATS[0]: return None   # -ns: progress from the history estimate alone
    if not PEEK[0]: return None
    n_p, n_d, gen, pn = PEEK[0]
    gen = gen or 0; pn = pn or 0
    try: t_pr = os.path.getmtime(LOAD_FILE)
    except OSError: t_pr = None
    dr = DEC_RATE[0]
    if dr and gen and pn and t_pr:
        td = gen / dr
        if n_d and "gen" in PHASE:
            tp = max(PHASE["gen"] - t_pr, 1e-3)
            return min((tp + td * min(n_d / gen, 1)) / (tp + td), 0.99)
        if n_p / pn >= 0.05:
            el_p = max(time.time() - t_pr, 1e-3)
            return min(el_p / (el_p * pn / n_p + td), 0.99)
        return None
    # no decode speed in a history (a machine that has run nothing yet): the prompt phase shows the prompt's own progress, the decode phase is timed from the tokens written so far
    if n_d and "gen" in PHASE and gen and t_pr:
        tp, el_d = max(PHASE["gen"] - t_pr, 1e-3), time.time() - PHASE["gen"]
        if n_d >= 8 and el_d > 1: return min((tp + el_d) / (tp + gen / (n_d / el_d)), 0.99)
        return None
    if not n_d and pn and n_p / pn >= 0.05: return min(n_p / pn, 0.99)
    return None


NAMEW = [16]   # name column width of the list lines: longest name + 1 (set when a list starts)
TEST_IN = config.test_in()   # what the status line says about a test's prompt (prompts.ini in =)
def out_k(tout):
    """answer length as written in a size: 1000 and 1024 are 1k, 4000 is 4k, anything else the number"""
    return "?" if not tout else f"{round(tout / 1000)}k" if tout % 1000 == 0 or tout % 1024 == 0 and tout < 4096 else str(tout)


def prompt_txt(test, tokens):
    """the prompt column of the results: tokens in/out, `12k/1k`; the test name when its prompt is not a size"""
    tin = TEST_IN.get(test); tout = int(tokens) if tokens else 0
    if (m := re.fullmatch(r"pf(\d+)k", test)): return f"{m[1]}k/" + out_k(tout)   # a sized prompt (run -ps 60k1k): never shown with the pf of its test name
    if not (tin and tin[0].isdigit()): return test
    return tin + "/" + (f"{tout // 1024}k" if tout and tout % 1024 == 0 else str(tout) if tout else "?")


MEMBW_C = r"""#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(){ size_t n=(size_t)1<<29; double *a=malloc(n*8); /* 4 GiB, first touched by the thread that reads it */
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;i++) a[i]=1.0;
 double best=0; for(int r=0;r<4;r++){ double s=0, t=omp_get_wtime();
  #pragma omp parallel for schedule(static) reduction(+:s)
  for(size_t i=0;i<n;i++) s+=a[i];
  t=omp_get_wtime()-t; double g=n*8/t/1e9; if(g>best&&s>0) best=g; }
 printf("%.1f\n",best); return 0; }"""


def membw():
    """RAM read bandwidth of the machine in GB/s, all threads streaming a 4 GiB array (gcc + OpenMP, ~3 s); measured on every call (nothing cached: a rented box keeps its name while the hardware behind it changes)"""
    try:   # threads: the CPUs the container may use (cgroup quota), not the ones it can see (384 threads on 46 CPUs are throttled and read a fraction); best of that and half
        sh = ("q=$(awk '{ if ($1==\"max\") print 0; else print int(($1+$2-1)/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null); n=$(nproc); "
              "[ \"${q:-0}\" -gt 0 ] && [ \"$q\" -lt \"$n\" ] && n=$q; "
              "d=$(mktemp -d); cat > $d/m.c <<'EOF'\n" + MEMBW_C + "\nEOF\ngcc -O3 -fopenmp $d/m.c -o $d/m && for t in $n $((n/2)); do OMP_NUM_THREADS=$t $d/m; done; rm -rf $d")
        v = max(float(x) for x in hsh(sh, 60).split())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError): return None
    return v


def hw_line():
    """one line about the machine for the results (people post screenshots): [128g ddr4,3200,46g 3700x 3090,4x16,26g,pl280 3090,4x4,6g,pl220 ssd,3g], i.e. RAM size, type, speed (MT/s) and measured
    bandwidth (GB/s, the engine's own probe from the newest server log), the CPU, per GPU: model, PCIe generation x width, measured upload GB/s, power limit, and the SSD (fork/models.ini [hw] ssd=)."""
    out = []
    try:
        mem = int([l for l in hsh("cat /proc/meminfo").splitlines() if l.startswith("MemTotal")][0].split()[1]) / 2**20
        ram = f"{-(-int(mem) // 8) * 8}g"
        try: dm = hsh("sudo -n dmidecode -t memory")   # needs root: without it the type and speed are left out, the measured bandwidth says enough
        except (OSError, subprocess.SubprocessError): dm = ""
        ty = re.search(r"^\s*Type: (DDR\d)", dm, re.M); sp = re.search(r"Configured Memory Speed: (\d+) MT/s", dm)
        if not (ty and sp) and HOST[0]:   # a rented VM/container has no DMI table: type and speed come from the machine's row in machines.csv (ram_type text), where they were written down from its listing
            rt = (machines().get(HOST[0].get("id")) or {}).get("ram_type", "")
            ty = ty or re.search(r"(DDR\d)", rt); sp = sp or re.search(r"(?:DDR\d-|up to )(\d{4})", rt)
        logs = sorted((f for f in glob.glob(QDIR + "/*.server.log") if ".remote." not in f), key=os.path.getmtime) if not HOST[0] else sorted(glob.glob(QDIR + "/*.remote.server.log"), key=os.path.getmtime)   # a remote list: the probe lines of its own servers only
        txt = next((t for t in (open(f, errors="replace").read(200000) for f in reversed(logs[-8:])) if "saturated at" in t or "-> CPU" in t), "")   # the newest log that has the probe lines (a cell still loading has none yet)
        bw = re.search(r"saturated at ([\d.]+) GB/s", txt) or re.search(r"-> CPU ([\d.]+) GB/s", txt); lk = dict(re.findall(r"CUDA(\d) ([\d.]+)", (re.search(r"prefill links \(GB/s\): (.*?) ->", txt) or [0, ""])[1] if re.search(r"prefill links \(GB/s\): (.*?) ->", txt) else ""))
        rt = (machines().get((HOST[0] or {}).get("id") or "pc1") or {}).get("ram_type", "")
        ch = re.search(r"(\d+)[ -]channels?", rt) or re.search(r"(\d+)ch\b", rt)   # channels as written in machines.csv; else the DIMMs dmidecode lists as populated (one per channel on most boards)
        ed = hsh("for r in /sys/devices/system/edac/mc/mc*/rank*; do echo \"$(cat $r/dimm_label 2>/dev/null) $(cat $r/dimm_mem_type 2>/dev/null)\"; done 2>/dev/null")   # the memory controller's own view (works in a container): populated channels of every controller
        ech = {(m_[0], m_[1]) for m_ in re.findall(r"mc#(\d+)csrow#\d+channel#(\d+) (?!Empty)\S+", ed)}
        nch = str(len(ech)) if ech else ch.group(1) if ch else (str(len(re.findall(r"^\s*Size: \d+ [MG]B", dm, re.M))) if dm else "")
        mbw = membw() or (float(bw.group(1)) if bw else 0)
        if not re.search(r"^\s*Size: \d+ [MG]B", dm, re.M) and mbw and sp and nch:   # no DMI (container): EDAC lists 7 channels on blg, 192 GiB does not split over 7 DIMMs and the stream test reaches 70% of 4 channels; what the memory delivers decides, shown as ~Nch (0.7 x MT/s x 8 B per channel)
            eff = round(mbw / (int(sp.group(1)) * 8 / 1000 * 0.7))
            if eff and eff != int(nch): nch = f"~{eff}"
        out.append(" ".join(x for x in (ram, ty.group(1).lower() if ty else "", f"{nch}ch" if nch and nch != "0" else "", sp.group(1) if sp else "", (lambda v: f"{v:.0f}g/s" if v else "")(mbw or None)) if x))   # the bandwidth is the machine's own stream test, else the server probe (two log wordings); 128g ddr4 4ch 3200 46g/s: size, type, channels, MT/s, measured GB/s
        mn = [l for l in hsh("grep -m1 'model name' /proc/cpuinfo").splitlines() if l.startswith("model name")][0]
        cpu = re.search(r"\b(\d{4,5}[A-Za-z0-9]*)\b", mn) or re.search(r"(?:EPYC|Xeon|Threadripper)\s+([A-Za-z0-9]+)", mn)   # 3700x, 7B12 (EPYC part numbers can hold letters first)
        ls = dict(re.findall(r"^([^:\n]+):\s+(.*)$", hsh("lscpu 2>/dev/null"), re.M))
        try:   # cores and threads over all sockets, the top clock, NUMA nodes when more than one: what decides a CPU-bound decode (a 64-core EPYC clocks lower than a desktop part)
            cores = int(ls["Socket(s)"]) * int(ls["Core(s) per socket"]); thr = cores * int(ls["Thread(s) per core"])
            cp = [f"{cores}c{thr}t" if thr != cores else f"{cores}c"] + ([f"{float(ls['CPU max MHz']) / 1000:.1f}ghz"] if ls.get("CPU max MHz") else []) + ([f"numa{ls['NUMA node(s)']}"] if int(ls.get("NUMA node(s)", 1)) > 1 else [])
        except (KeyError, ValueError): cp = []
        bn = hsh("cat /sys/class/dmi/id/board_name 2>/dev/null").strip()   # the board: its chipset (X570, TRX40, WRX80 ...) or, for an EPYC server board with no chipset, the board name itself (H12SSW-NT)
        if bn:
            cs = re.search(r"\b([XBAZH]\d{3}[A-Z]?|TRX40|WRX80)\b", bn)
            if cs or not HOST[0]: out.append((cs.group(1) if cs else bn).lower().replace(" ", "-"))   # a rented VM's board name (T1DMGEP ...) says nothing: only a real chipset is shown there
        if cpu: out.append(",".join([cpu.group(1).lower()] + cp))
        g = hsh(f"{NVSMI if not HOST[0] else 'nvidia-smi'} --query-gpu=index,name,pcie.link.gen.max,pcie.link.width.current,power.limit --format=csv,noheader,nounits 2>/dev/null").strip().splitlines()
        g = [l for l in g if l.count(",") == 4]
        gs = []
        for l in g:
            i, nm, gen, w, pl = [x.strip() for x in l.split(",")]
            gs.append((-int(w), int(i), ",".join([re.sub(r"\D*(\d{3,4}).*", r"\1", nm), f"{gen}x{w}"] + ([f"{float(lk[i]):.0f}g"] if i in lk else []) + [f"pl{float(pl):.0f}"])))
        out += [x[2] for x in sorted(gs)]
        ssd = "" if HOST[0] else (fcfg.models_ini().get("hw") or {}).get("ssd", "")
        if ssd: out.append(f"ssd,{ssd}")
    except Exception: pass
    return "[" + " ".join(out) + "]" if out else ""


CELL_TEST = [""]   # the running cell's test and answer length, shown after its name in the status line (set by run_list)
LOAD_FILE = QDIR + "/load-s.txt"   # written by a cell when its server answers: seconds of load; peek_line shows it; run_watched removes it at each cell start


def peek_line(exp):
    """`now: ...` from the server's /slots (every status tick, about 3 s): prompt tokens done while it prefills, then tokens written, against the earlier runs' medians; '' when it does not answer."""
    try: ld = (lambda v: col(32, f"loaded in {v:.0f} s") + "  " if v >= 0 else "")(float(open(LOAD_FILE).read()))
    except (OSError, ValueError): ld = ""
    sp = slots_progress()
    PEEK[0] = None
    if not sp: return col(33, "    now: ") + ld.rstrip() if ld else ""
    n_prompt, n_dec, _ = sp
    gen, pn = exp
    if SLOT_EXACT[0]:   # the server's totals replace the earlier runs' medians
        pt, pd, dd, dt = SLOT_EXACT[0]
        n_prompt, n_dec = pd, dd
        if pt: pn = pt
        if dt and dt - dd < 10**6: gen = dt
    PEEK[0] = (n_prompt, n_dec, gen, pn)
    if not n_dec and n_prompt: return col(33, "    now: ") + ld + col(32, (f"pass {min(LIVE_PASS[0] + 1, LIVE_PASS[1])}/{LIVE_PASS[1]}  " if LIVE_PASS[1] > 1 else "") + f"prompt {n_prompt}" + (f"/~{int(pn)} tok" + (f" ~{min(100, 100 * n_prompt // int(pn))}%" if pn and n_prompt <= pn else " +") if pn else " tok"))   # the phase colours of the bar: prompt green
    if n_dec: return col(33, "    now: ") + ld + col(34, f"{n_dec}" + (f"/~{int(gen)} tok " + (f"~{100 * n_dec // int(gen)}%" if n_dec <= gen else "+") if gen else " tok"))   # generation blue
    return ""


def trace_lines(width=40):
    """when each tuned value changed in the cell and when it settled: per knob the number of changes, a strip of change counts over the cell time (dense = busy), the last change (= settled) and the value"""
    if not TRACE and not TRACE_H and not TRACE_P: return ""
    end = max(TRACE_T0[1] if len(TRACE_T0) > 1 else 0, max([t for t, *_ in TRACE] or [0])) or 1
    out = [col(37, "tuning timeline") + col(90, f"  (cell time 0 .. {int(end // 60)}:{int(end % 60):02d}; the strip shows how many changes fell into each slice)")]
    for k in dict.fromkeys(x[1] for x in TRACE):
        ev = [x for x in TRACE if x[1] == k and x[2] is not None]   # the first sight of a value is not a change
        if not ev: continue
        bins = [0] * 20
        for t, *_ in ev: bins[min(19, int(20 * t / end))] += 1
        mx = max(bins); strip = "".join(" " if b == 0 else "▁▂▃▄▅▆▇█"[min(7, (b * 8 - 1) // mx)] for b in bins)
        last = ev[-1]
        out.append(f"  {k:<14} {len(ev):>3} changes |{strip}| last at {int(last[0] // 60)}:{int(last[0] % 60):02d} -> {last[3]}" + col(90, f"  (unchanged for {int((end - last[0]) // 60)}:{int((end - last[0]) % 60):02d})"))
    if TRACE_P:   # the phases of the whole run: L = loading (no engine data yet), P = a prompt call runs, D = decoding
        sl = [""] * 20
        for t, ph in TRACE_P:
            i = min(19, int(20 * t / end)); sl[i] = ph if sl[i] in ("", ph) else "P" if "P" in (sl[i], ph) else ph
        first = int(20 * TRACE_P[0][0] / end)
        out.insert(1, f"  {'phase':<14} |" + "".join(("L" if i < first else ".") if not c else c for i, c in enumerate(sl)) + "| L load, P prompt, D decode")
    if TRACE_H:   # hit rate over the cell: mean of the ticks in each slice as a digit 0-9 (tenths of 100 %), '.' = no tick
        sl = [[] for _ in range(20)]
        for t, h in TRACE_H: sl[min(19, int(20 * t / end))].append(h)
        out.append(f"  {'hit rate':<14} |" + "".join("." if not v else str(min(9, int(sum(v) / len(v) / 10))) for v in sl) + "| digit = tenths of 100 %, per slice" + col(90, f"  (first {TRACE_H[0][1]:.0f}%, last {TRACE_H[-1][1]:.0f}%)"))
    return "\n".join(out)


def result_lines(res):
    return "\n".join(f"    run {i}: " + speed_txt(t, p, width=7) + col(92, " t/s") + (col(36, "  " + CELL_PROMPT[0]) if CELL_PROMPT[0] else "") for i, (t, p) in enumerate(res, 1))   # prompt in/out of the cell after each run


PRELUDE = [0, 0]   # lines of a remote list's start (old servers stopped, disk check) and of its transfer bars still on screen: erased when the transfers are done
LIVE_UP = [0]       # lines of the status block on screen: the next cell redraws over it, nothing scrolls
CELL_PG = [None, None]   # seconds of the finished cell's prompt and generation phases (stored with wall_s)
CELL_ASKED = [0]    # tokens asked for in the running cell
CELL_PROMPT = [""]   # the running cell's prompt text (set per row)
CELL_BUILD = [""]   # the running cell's build: the "vs best" columns leave its own earlier runs out
BEST_MEMO = {}   # (model, test, build, asked) -> best_prev(...): the live view asks every second
CELL_REF = [None]   # (model file name, test) of the running cell: its live speeds are held against the best earlier run (best_ref)
_BEST_REF = {}
LAST_PP = {}   # the running cell's last live prompt speed (start of the cell, t/s)
SPLIT_PASSES = {"t0": None, "rows": [], "peak": {}}   # the running cell's split passes as the engine reported them live (/moe/stats split.bal); save_stats writes them to stats.ini


def best_ref(model, test):
    """(pp t/s, build, tg t/s, build) of the best earlier ok, non-looping run of this model and test at the power limits now set (runs.db); None for what has no history.
    Cached per cell: the live line asks every second."""
    hw = os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
    pl = power_limits(hw)
    key = (model, test, pl)
    if key not in _BEST_REF:
        try:
            import sqlite3
            c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
            q = "select coalesce(%s_max, %s), build from runs where model=? and test=? and ok=1 and coalesce(loop,'0') in ('0','') and coalesce(%s_max, %s)>0" + (" and pl=?" if pl else "") + " order by 1 desc limit 1"
            a = (model, ALIASES.get(test, test)) + ((pl,) if pl else "")
            pp = c.execute(q % (("pp",) * 4), a).fetchone(); tg = c.execute(q % (("tps",) * 4), a).fetchone()
            _BEST_REF[key] = (pp[0], pp[1], tg[0], tg[1]) if pp and tg else None
        except Exception: _BEST_REF[key] = None
    return _BEST_REF[key]
CELL_REC = []       # structured twin of CELL_LINES: the final table is built from it
STOCK_REC = [None]  # set by stock_gain: the stock numbers behind its text
CELL_LINES = []     # one summary line per finished cell, printed when the list ends


def live_tty():
    """The status block redraws in place on a terminal; the detached run (RUN_DAEMON) writes the same escape codes into its log for the viewer to replay."""
    return bool(os.environ.get("RUN_DAEMON")) or sys.stdout.isatty()


LIVE_W = int(os.environ.get("RUN_LIVE_W") or 150)   # the widest status line; a detached run does not know the viewer's terminal, so one width for all (RUN_LIVE_W changes it)
ANSI_RE = re.compile(r"\033\[[0-9;]*[A-Za-z]")


KV_RE = (re.compile(rb"kv-fit: KV tiers with the expert cache at its floor: f16 up to (\d+) tokens, q8_0/q8_0 up to (\d+), q4_0/q4_0 up to (\d+)"),
         re.compile(rb"kv-fit: \d+ tokens: KV now (\S+) in place|kv-ram: KV of .* moved to host memory"), re.compile(rb"(\d+) cells (?:in use|at its end)"))


def kv_tier():
    """the KV type now (f16, q8_0, q4_0, RAM) from the newest server log, "" before the plan is printed: stamped on every pass row, so a switch shows as a jump in the pass speed"""
    try:
        f = max(glob.glob("/tmp/perf-*.log"), key=os.path.getmtime)
        with open(f, "rb") as h: h.seek(max(0, os.path.getsize(f) - (4 << 20))); t = h.read()
    except (ValueError, OSError): return ""
    if not KV_RE[0].search(t): return ""
    sw = list(KV_RE[1].finditer(t))
    return "f16" if not sw else (sw[-1][1] or b"RAM").decode().split("/")[0]


PASS_KV = {}   # pass signature -> the KV type when the pass was first seen


def kv_line():
    """The KV against what VRAM can hold (user 2026-10-25): its type now (f16, then q8_0, q4_0, part in RAM as the context grows) and the cells in use as a % of that
    type's VRAM limit, so the switch to the next type can be seen coming at 100%. From the newest server log (/tmp/perf-<build>.log): the tier plan the engine prints
    once (kv-fit: KV tiers ...), its switches (kv-fit: ... KV now ..., kv-ram: ...) and the cells of the last pass (kv-tail / pass size lines). None before the plan is printed."""
    try:
        f = max(glob.glob("/tmp/perf-*.log"), key=os.path.getmtime)
        with open(f, "rb") as h: h.seek(max(0, os.path.getsize(f) - (4 << 20))); t = h.read()
    except (ValueError, OSError): return col(37, "kv ") + col(90, "no server log yet")   # always shown (user 2026-10-25)
    caps = KV_RE[0].findall(t)
    if not caps: return col(37, "kv ") + col(90, "plan not printed yet")
    f16, q8, q4 = (int(x) for x in caps[-1])
    sw = list(KV_RE[1].finditer(t)); cells = KV_RE[2].findall(t)
    row = float(m[1]) * 1024 if (m := re.search(rb"\(([\d.]+) KB of f16 KV per token", t)) else 0   # f16 bytes per token over all cards
    tier = "f16" if not sw else (sw[-1][1] or b"RAM").decode().split("/")[0]
    n = int(cells[-1]) if cells else 0
    cap = {"f16": f16, "q8_0": q8, "q4_0": q4}.get(tier)
    per = row * {"f16": 1, "q8_0": 34 / 64, "q4_0": 18 / 64}.get(tier, 1)   # bytes per token of that type (q8_0: 34 bytes per 32 values, q4_0: 18)
    if not cap: return col(37, "kv ") + col(91, f"part in RAM {n * per / 2**30:.1f} GB, VRAM full")
    # space left in this type and what comes next (user 2026-10-25): the tiers' limits are in cells, f16 < q8_0 < q4_0 < part in RAM
    nxt = {"f16": "q8_0", "q8_0": "q4_0", "q4_0": "RAM"}.get(tier, "")
    # The model's own context caps the tier: MiMo allows 1M tokens and its f16 KV (22.5 KB a token) fits the pools up to 1.58M, so the switch to q8_0 can never come
    # (user 2026-10-25: 34 GB was the pool's limit, not the model's). Shown against what can really happen, and the space the zones hold now (grown on demand).
    mm = re.search(rb"\(model cap (\d+)\)", t); mcap = int(mm[1]) if mm else 0
    zone = {}
    for d, z, r in re.findall(rb"kv-tail (CUDA\d+): zone \d+ -> (\d+) cells at [\d.]+ -> [\d.]+ MiB \((\d+) B per cell", t): zone[d] = int(z) * int(r)
    res = f" reserved {sum(zone.values()) / 2**30:.1f}G" if zone else ""
    if mcap and cap >= mcap: cap, nxt = mcap, ""
    left = max(0, cap - n)
    return (col(37, "kv ") + col(36, tier) + " " + level(n / cap, 0.8, 0.95, f"{100 * n / cap:.0f}% {n * per / 2**30:.1f}/{cap * per / 2**30:.0f}G") + col(90, res) + " "
            + level(n / cap, 0.8, 0.95, f"{left // 1000}k left") + (col(90, f" next {nxt}") if nxt else col(90, " no q8 needed" if tier == "f16" else "")))


def mem_avail_gib():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")) / 2**20


def ram_total_gib():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal:")) / 2**20


def fit_block(txt):
    """The status block, every line whole (nothing cut, user 2026-10-24) but none wider than LIVE_W: a longer line is broken at its last separator
    (" | ", two spaces, " (") before the width and goes on, indented, on the next line (user 2026-10-25: a line the terminal wraps puts the cursor-up
    redraw one row short and the cpu row repeats). The key speed lines are kept short at their source instead (vs best: no explanation)."""
    out = []
    for line in txt.split("\n"):
        while len(ANSI_RE.sub("", line)) > LIVE_W:
            vis = 0; cut = None; i = 0
            while i < len(line) and vis <= LIVE_W:   # the last separator at or before the width, counted in visible characters
                if (m := ANSI_RE.match(line, i)): i = m.end(); continue
                if vis > 20 and any(line.startswith(sep, i) for sep in (" | ", "  ", " (")): cut = i
                vis += 1; i += 1
            if cut is None: break   # no separator: the line stays whole and long
            out.append(line[:cut] + "\033[0m"); line = "    " + line[cut:].lstrip(" |")
        out.append(line)
    return "\n".join(out)


def live_clear():
    """Erase the status block (before anything else prints)."""
    if LIVE_UP[0] and live_tty(): print(f"\033[{LIVE_UP[0]}A\033[J", end="", flush=True)
    LIVE_UP[0] = 0


OLDER_MIN_BNO = 13509   # builds before this are not shown as older gainers (user 2026-10-19)


def older_gainers(model, test, cur_bno, st, gm, limit=2):
    """earlier builds of ours with a bigger overall gain over the same stock numbers than this build: [(build, bno, date, t, pp, gain t/s, gain pp)], best first"""
    import sqlite3
    if not model or not test or not st or not str(cur_bno).isdigit(): return []
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        rows = c.execute("select coalesce(nullif(\"commit\",''), build), bno, tps, pp, start_ts, rowid from runs where model=? and test=? and build not like 'stock%' and cast(ok as integer)=1 and tps is not null" + HWF[0] + " and bno<? and bno>=? and coalesce(env,'') not like '%warmpass%' and n_gen between ? and ?",
                         (os.path.basename(model), ALIASES.get(test, test), int(cur_bno), OLDER_MIN_BNO, (ALIAS_TOKENS.get(test) or 0) // 2, (ALIAS_TOKENS.get(test) or 10**9) * 2)).fetchall()
    except sqlite3.Error: return []
    by = {}
    for b, n, t, p, ts, rid in rows: by.setdefault((str(b)[:9], n), []).append((float(t), float(p) if p else None, str(ts)[:10], rid))
    out = []
    for (b, n), v in by.items():
        t = statistics.median(x[0] for x in v); ps = [x[1] for x in v if x[1]]; p = statistics.median(ps) if ps else None
        gt = t / st["t"]; gp = p / st["p"] if p and st["p"] else None
        gs = [g for g in (gt, gp) if g]; g = math.prod(gs) ** (1 / len(gs))
        if g > gm: out.append((g, b, int(n), max(x[2] for x in v), t, p, gt, gp, max(x[3] for x in v)))
    out.sort(reverse=True)
    return [(b, n, d, t, p, gt, gp, rid) for g, b, n, d, t, p, gt, gp, rid in out[:limit]]


def passes_save(model, test, build):
    """the finished dense-split passes of the cell into runs.db (table passes), one row per pass: the live "vs best" of a pass compares with the best earlier pass of the same
    model at a similar size, because dense and expert cost per token differ per model and per pass size (user 2026-10-24). Called once at the end of a cell."""
    import sqlite3
    rows = [r[1] for r in SPLIT_PASSES.get("rows", []) if r[1].get("cards")]
    if not rows or not model or str(build).startswith("stock"): return
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        c.execute("create table if not exists passes (ts text, hw text, model text, test text, build text, k int, tokens int, expert_tokens int, us_tok real, wait_pct real)")
        hw = (HWF[0].split("'")[1] if "'" in HWF[0] else "")
        for k, b in enumerate(rows, 1):
            cs = b["cards"]; T = sum(x.get("tok", 0) for x in cs)
            span = b.get("span_ms") or b.get("us_tok", 0) * T / 1e3
            ex = next((x for x in cs if x.get("experts")), cs[-1])
            if T <= 512 or not span: continue
            wt = (ex.get("xw_ms") if b.get("detail") else ex.get("wait_dense_ms", 0) + ex.get("wait_moe_ms", 0)) or 0
            c.execute("insert into passes values (?,?,?,?,?,?,?,?,?,?)", (time.strftime("%F %T"), hw, os.path.basename(model), ALIASES.get(test, test), str(build), k, T, ex.get("tok", 0), span * 1e3 / T, 100.0 * wt / span if span else 0.0))
        c.commit()
    except sqlite3.Error as e: print("passes:", e, file=sys.stderr)


def prompt_passes_save(model, test, build):
    """every prompt pass of the cell's server log (engine line `prompt pass k/m: N tokens in X s (Y t/s), experts from cache H%, ...`) into runs.db table prompt_passes: one row per
    pass with its tokens, seconds, speed, the share of its experts that came from the cache (hit_pct = found on the card, copied there with the upload skipped; asked = experts asked for; other_card_pct = the share whose layer cache is on the other card), so the first pass (a cold cache) and the later ones can be compared. Called once at the end of a cell."""
    import sqlite3
    log = f"/tmp/perf-{build}.log"
    if not model or not os.path.exists(log): return
    call = 0; rows = []
    for ln in open(log, errors="replace"):
        m = re.search(r"prompt pass (\d+)/(\d+): (\d+) tokens in ([\d.]+) s \((\d+) t/s\)(?:, experts from cache (\d+)% \((\d+) of (\d+) asked; (\d+)% belong to the other card's cache)?", ln)
        if not m: continue
        k, n, tok, sec, tps, hit, fnd, ask, wr = m.groups()
        if k == "1": call += 1
        rows.append((call, int(k), int(n), int(tok), float(sec), int(tps), None if hit is None else int(hit), None if ask is None else int(ask), None if fnd is None else int(fnd), None if wr is None else int(wr)))
    if not rows: return
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        c.execute("create table if not exists prompt_passes (ts text, hw text, model text, test text, build text, call int, k int, n int, tokens int, secs real, tps int, hit_pct int, asked int, found int, other_card_pct int)")
        hw = (HWF[0].split("'")[1] if "'" in HWF[0] else "")
        ts = time.strftime("%F %T")
        c.executemany("insert into prompt_passes values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(ts, hw, os.path.basename(model), ALIASES.get(test, test), str(build)) + r for r in rows])
        c.commit()
    except sqlite3.Error as e: print("prompt passes:", e, file=sys.stderr)


def pass_best_us(model, test, build, tokens):
    """best (smallest) time per token among the 5 earlier passes of this model and test on this machine, other builds of ours, closest to `tokens` in pass size (no size window to guess); None without history"""
    import sqlite3
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        r = c.execute("select min(us_tok) from (select us_tok from passes where model=? and test=? and build!=?" + HWF[0] + " order by abs(tokens - ?) limit 5)", (os.path.basename(model), ALIASES.get(test, test), str(build), int(tokens))).fetchone()
        return r[0] if r and r[0] else None
    except sqlite3.Error: return None


PASS_CURVES = {}   # (model, build, machine) -> one curve per earlier run: the live view asks every second


def pass_curves(model, build):
    """time per token against the cells a pass is centred on, one curve per earlier run of this model on this machine (any prompt length: the cost per token depends on
    the cells, not on how long the prompt is), other builds of ours. The cells of a pass are the tokens of the passes before it plus half its own."""
    import sqlite3
    key = (os.path.basename(model), str(build), HWF[0])
    if key in PASS_CURVES: return PASS_CURVES[key]
    out = []
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        runs, last = {}, {}
        for b, test, k, tok, us in c.execute("select build, test, k, tokens, us_tok from passes where model=? and build!=? and build not like 'stock%'" + HWF[0] + " order by rowid", key[:2]):
            if (b, test) not in runs or k <= last[(b, test)]: runs[(b, test, len(runs))] = runs.pop((b, test), None) or []   # k restarts at 1: the next run of the cell
            runs.setdefault((b, test), []).append((tok, us)); last[(b, test)] = k
        for pts in runs.values():
            pos, cur = 0, []
            for tok, us in pts: cur.append((pos + tok / 2, us)); pos += tok
            if cur: out.append(cur)
    except sqlite3.Error: pass
    PASS_CURVES[key] = out
    return out


def pass_best_at(model, build, pos):
    """the best earlier time per token (us) at `pos` cells: every earlier run's curve interpolated linearly between its two passes around `pos` (outside its range only
    the nearest pass within 20% of the position, at least 8k cells), the smallest of them; None without any"""
    best = None
    for cur in pass_curves(model, build):
        lo = [q for q in cur if q[0] <= pos]; hi = [q for q in cur if q[0] > pos]
        if lo and hi:
            (x0, y0), (x1, y1) = lo[-1], hi[0]
            y = y0 + (y1 - y0) * (pos - x0) / (x1 - x0)
        else:
            x, y = lo[-1] if lo else hi[0]
            if abs(x - pos) > max(8192, 0.2 * pos): continue
        if best is None or y < best: best = y
    return best


def best_prev(model, test, build, asked=0):
    """the best earlier run of ours of this model and test (about the same answer length, this machine, any build of ours except `build`): (best t/s, best pp, build of the best pp) or None.
    Shown as the "vs best" columns of the results, so a run reads as better or worse than the best so far without looking anything up (user 2026-10-24)"""
    import sqlite3
    if not model or not test or not build: return None
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        gen = asked or ALIAS_TOKENS.get(test) or 0   # the answer length the cell asked for (pf60k1k cells ask 64), else the alias's
        # every earlier run of the prompt: its prompt speed does not depend on the answer length (MiMo's only 60k rows asked 1024 tokens: "vs best pp" stayed empty
        # for a 64-token cell); the decode speed is compared only within about the same answer length (a longer answer decodes at a longer context)
        # an answer of less than half a character per token is no text (the deep-seek release without -c answered token 0: 7 characters for 1024 tokens at 43.7 t/s,
        # and every DeepSeek cell after the fix showed 0.5x "vs best" against it, 2026-10-26); rows without a recorded answer stay in
        rows = c.execute("select tps, pp, build, n_gen from runs where model=? and test=? and build not like 'stock%' and build!=? and cast(ok as integer)=1 and tps is not null" + HWF[0] + " and coalesce(env,'') not like '%warmpass%'"
                         " and not (coalesce(out_chars,'') != '' and coalesce(n_gen,0) > 0 and cast(out_chars as integer) * 2 < n_gen)",
                         (os.path.basename(model), ALIASES.get(test, test), str(build))).fetchall()
    except sqlite3.Error: return None
    if not rows: return None
    # the median of each build's runs, then the best build: one lucky run (decode after a 60k prompt swings +-15% inside one build) is not the bar to beat
    by_t, by_p = {}, {}
    for t_, p_, b_, g_ in rows:
        if p_: by_p.setdefault(b_, []).append(float(p_))
        if not gen or (g_ and gen // 2 <= g_ <= gen * 2): by_t.setdefault(b_, []).append(float(t_))
    med_p = {b_: statistics.median(v) for b_, v in by_p.items()}
    bb = max(med_p, key=med_p.get, default=None)
    bt = max((statistics.median(v) for v in by_t.values()), default=None)
    if bt is None and bb is None: return None
    return (bt, med_p[bb] if bb else None, bb or next(iter(by_t), None))


def ours_reference(model, test):
    """our newest run (build of ours with the newest run of this model and test, about the same answer length): {t, p, id, bno, date} best runs (the result lines show the best run of a cell, not the median) and the id of its last run; None when there is none"""
    import sqlite3
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")); mo = os.path.basename(model); t = ALIASES.get(test, test); gen = ALIAS_TOKENS.get(test)
    w = " and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""
    o = c.execute("select build, max(ts) from runs where model=? and test=? and build not like 'stock%' and cast(ok as integer)=1 and tps is not null" + HWF[0] + "" + w + " group by build order by max(ts) desc limit 1", (mo, t)).fetchone()
    if not o: return None
    v = c.execute("select tps, pp, bno, start_ts, rowid from runs where model=? and test=? and build=? and cast(ok as integer)=1 and tps is not null and ts >= datetime(?, '-30 minutes')" + HWF[0] + "" + w + " order by ts", (mo, t, o[0], o[1])).fetchall()   # the newest session only: an older run of other settings (a longer prompt, a short answer) is not the reference
    return dict(t=max(x[0] for x in v), p=max(x[1] for x in v if x[1]) if any(x[1] for x in v) else None, id=v[-1][4], bno=int(v[-1][2]) if v[-1][2] else "?", date=str(v[-1][3] or o[1])[:10])


def known_gain(model, test):
    """overall gain (geometric mean of decode and prompt gain) of the newest build of ours over the newest official stock row, both from runs.db; 0 when either is missing"""
    import sqlite3
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")); mo = os.path.basename(model); t = ALIASES.get(test, test); gen = ALIAS_TOKENS.get(test)
    ours = c.execute("select build, max(ts) from runs where model=? and test=? and build not like 'stock%' and cast(ok as integer)=1 and tps is not null" + HWF[0] + " group by build order by max(ts) desc limit 1", (mo, t)).fetchone()
    if not ours: return 0
    v = c.execute("select tps, pp from runs where model=? and test=? and build=? and cast(ok as integer)=1 and tps is not null" + HWF[0] + "" + (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""), (mo, t, ours[0])).fetchall() or c.execute("select tps, pp from runs where model=? and test=? and build=? and cast(ok as integer)=1 and tps is not null" + HWF[0] + "", (mo, t, ours[0])).fetchall()
    st = c.execute("select tps, pp from runs where model=? and test=? and build like 'stock-b%-official' and cast(ok as integer)=1 and tps is not null" + HWF[0] + "" + (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else "") + " order by ts desc limit 1", (mo, t)).fetchone()
    if not v or not st or not st[0]: return 0
    ti = TEST_IN.get(test) or TEST_IN.get(t) or ""
    gs = [statistics.median(x[0] for x in v) / st[0]] + ([statistics.median(x[1] for x in v if x[1]) / st[1]] if st[1] and any(x[1] for x in v) else [])
    return math.prod(gs) ** (1 / len(gs))


_PPL_CACHE = {}
def ppl_pair(model, bno):
    """`ours/stock` perplexity of a model: the newest default-settings run of build bno (else the newest build with one) and the newest official stock run, both cut to the same chunk count (runs.db ppl_chunks); "" when either is missing"""
    import sqlite3
    key = (os.path.basename(model or ""), bno)
    if key in _PPL_CACHE: return _PPL_CACHE[key]
    res = ""
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        def newest(where, args):
            r = c.execute(f"select start_ts, build from ppl_chunks where model=? and {where} order by start_ts desc limit 1", (key[0],) + args).fetchone()
            return r and [v for _, v in c.execute("select chunk, cum_ppl from ppl_chunks where model=? and start_ts=? and build=? order by chunk", (key[0], r[0], r[1]))]
        o = newest("build=?", (f"b{int(bno)}",)) if bno else None
        o = o or newest("build glob 'b[0-9]*' and build not like '% %'", ())
        s_ = newest("build like 'stock%'", ())
        if o and s_:
            n = min(len(o), len(s_)); res = f"{o[n - 1]:.4f}/{s_[n - 1]:.4f}"
    except (sqlite3.Error, ValueError): pass
    _PPL_CACHE[key] = res
    return res


def run_times(rid):
    """(seconds the run took, `l 44s` load text) of runs.db row rid, from its wall_s and load_s; (None, "") when not recorded"""
    import sqlite3
    try:
        r = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute("select wall_s, load_s, prompt_s, gen_s from runs where rowid=?", (rid,)).fetchone()
    except sqlite3.Error:
        try: r = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute("select wall_s, load_s, null, null from runs where rowid=?", (rid,)).fetchone()   # a db from before the prompt_s / gen_s columns
        except sqlite3.Error: r = None
    if not r or not r[0]: return None, ""
    lv = r[1]
    if not lv:   # a repeat on a kept server loads nothing: show the load of the cell's first run (same build, model and test, started before it)
        try:
            q = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute(
                "select b.load_s from runs a join runs b on b.build=a.build and b.model=a.model and b.test=a.test and b.rowid<a.rowid and b.load_s>0 "
                "and b.rowid>=a.rowid-3 where a.rowid=? order by b.rowid desc limit 1", (rid,)).fetchone()
            if q: lv = q[0]
        except sqlite3.Error: pass
    tf = lambda x: f"{int(x)}s" if x < 100 else f"{int(x // 60)}m{int(x % 60):02d}s"
    return float(r[0]), ("" if lv is None or lv < 0 else "l " + tf(lv) + (f" p {tf(r[2])} g {tf(r[3])}" if r[2] is not None and r[3] is not None else ""))   # l load, p prompt, g generation


def loop_flag(rid):
    """True when the stored run `rid` (runs.db rowid) has a detected repetition loop (loop=1): the table shows LOOP next to FAIL, and its numbers are not comparable"""
    import sqlite3
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        r = c.execute("select loop from runs where rowid=?", (rid,)).fetchone() if rid else None
        return bool(r and str(r[0]) == "1")
    except sqlite3.Error:
        return False


def out_len(ours_id, stock_id):
    """`843/345`: tokens of the answer of our run / of the stock run (runs.db n_gen); the asked length is in the prompt column"""
    import sqlite3
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    def n(i):
        try: r = c.execute("select n_gen from runs where rowid=?", (i,)).fetchone() if i else None
        except sqlite3.Error: r = None
        return str(int(r[0])) if r and r[0] else ""
    o, s_ = n(ours_id), n(stock_id)
    return f"{o}/{s_}" if (o or s_) else ""


def par_gpus(pf_all, pf_gpus, args=""):
    """`2,1`: the cards that computed at the same time in the prompt and in decode. Prompt: the most cards any prompt pass used (pf_all,
    tokens:mode/CUDA0+CUDA1 per pass); a run without a decision line ran on one card. Decode: 1, the layers of a token run on one card after another
    (layer split: the other card waits, apart from its uploads); n only with -sm row/tensor, where every card computes each layer."""
    passes = [p.rsplit("/", 1)[-1] for p in (pf_all or "").split(";") if "/" in p] or ([pf_gpus] if pf_gpus else [])
    pf = max([len(g.split("+")) for g in passes if g and g != "None"] or [1])
    return f"{pf},{max(pf, 2) if re.search(r'(?:-sm|--split-mode)\s+(row|tensor)', args or '') else 1}"


def gpus_used(i):
    """par_gpus of a runs.db row: the stored column, else worked out from its prefill fields (rows before 2026-10-23)"""
    import sqlite3
    if not i: return ""
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        have = {x[1] for x in c.execute("pragma table_info(runs)")}
        r = c.execute(f"select pf_all, pf_gpus, cmd{', par_gpus' if 'par_gpus' in have else ''} from runs where rowid=?", (i,)).fetchone()
    except sqlite3.Error: r = None
    if not r: return ""
    return r[3] if len(r) > 3 and r[3] else par_gpus(r[0], r[1], r[2])


def run_id(campaign):
    """runs.db rowid of the newest ok run of a campaign (the id of `run out ID`), None when there is none"""
    import sqlite3
    try:
        r = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute("select rowid from runs where campaign=? and cast(ok as integer)=1 order by ts desc, rowid desc limit 1", (campaign,)).fetchone()
        return r[0] if r else None
    except sqlite3.Error: return None


def xfer_of(bno, argtext):
    """hand-off precision between the cards for a cell: asked on the command line (split-f16=1 / 0), else the build's default (f32 from b13600, f16 before)"""
    if "split-f16=1" in argtext: return "f16"
    if "split-f16=0" in argtext: return "f32"
    return "f32" if str(bno).isdigit() and int(bno) >= 13600 else "f16"


def final_table(recs, live=False, hdr_out=None, headers=True):
    """the finished cells as an aligned table without grid lines, as wide as the data needs and no wider than the terminal. t/s and pp are named once in the header; ours, stock, the two gains; the builds below.
    Too wide: the load column goes, then the time column, then the dates of the builds; a row is never split in two lines"""
    if not recs: return []
    # a detached run (no terminal, e.g. started with nohup and watched with `jobs`) does not know the viewer's width: no wrapping there, the viewer's terminal
    # decides (user 2026-10-24: the 100-column fallback split every row in two, unreadable)
    try: cols = int(os.environ.get("COLUMNS") or (shutil.get_terminal_size((100, 24)).columns if sys.stdout.isatty() else 1000))
    except (ValueError, OSError): cols = 100
    f = lambda v, d: "" if v is None else f"{v:.{d}f}"
    body = []
    for r in recs:
        res = r["res"]; t = max(x[0] for x in res) if res else None; p = max(x[1] for x in res) if res and res[0][1] is not None else None
        st = r["st"]; el = r["el"]; rid_show = r.get("rid")
        if r.get("ref"):   # a stock cell: the line is our newest run against this stock run (ours columns = ours, stock columns = what was just measured)
            ref = r["ref"]; st = dict(t=t, p=p, ob=ref["bno"], od=ref["date"], sb=r.get("sbno") or "?", sd=r.get("sdate") or "", sid=r.get("rid")); t, p, rid_show = ref["t"], ref["p"], ref["id"]
        elif r.get("stock_row") and res:   # a stock cell and none of ours yet: the stock numbers in the stock columns, ours empty
            st = dict(t=t, p=p, ob="", od="", sb=r.get("sbno") or "?", sd=r.get("sdate") or "", sid=r.get("rid")); t = p = None; rid_show = None
        gt = t / st["t"] if st and t else None; gp = p / st["p"] if st and p and st["p"] else None
        body.append(dict(name=r["name"], xfer=r.get("xfer") or "", id=str(rid_show or ""), sid=str(st.get("sid") or "") if st else "", prompt=r.get("prompt") or "", out=out_len(rid_show, st.get("sid") if st else (r.get("rid") if r.get("stock_row") else None)), time=f"{int(el // 60)}:{int(el % 60):02d}" + (f" {min(99, int(100 * el / r['exp']))}%" if el is not None and r.get("exp") else "") if el is not None else f"~{int(r['est'] // 60)}:{int(r['est'] % 60):02d}" if r.get("est") else "", load=r["lt"].replace("l ", ""), fail=bool(r["rc"]), loop=loop_flag(rid_show), t=f(t, 2), p=f(p, 1), g=gpus_used(rid_show),
                         st=f(st["t"], 2) if st else "", sp=f(st["p"], 1) if st and st["p"] else "", gt=f(gt, 2), gp=f(gp, 2), gtv=gt, gpv=gp, ppl=(ppl_pair(r.get("model"), st.get("ob")) if st and st.get("ob") else ""), nost=(not st and not r["stock_row"])))
        bb = best_prev(r.get("model"), r.get("test"), r.get("build"), r.get("asked") or 0) if not r.get("ref") and not r.get("stock_row") and res else None   # this run over the best earlier run of ours
        vt = t / bb[0] if bb and t and bb[0] else None; vp = p / bb[1] if bb and p and bb[1] else None
        body[-1].update(vt=f(vt, 2), vp=f(vp, 2), vtv=vt, vpv=vp)
        if not r.get("ref") and not r.get("stock_row") and res and not str(r.get("build") or "").startswith("stock"):   # no earlier run of ours: "first", not an empty cell that looks broken
            if vt is None and t: body[-1].update(vt="first")
            if vp is None and p: body[-1].update(vp="first")
        body[-1]["section"] = r.get("section")   # `run status`: done / running / queued
        body[-1]["bt"] = (f"b{st['ob']} {st['od'][5:]} vs b{st['sb']} {st['sd'][5:]}" if st["ob"] else f"stock b{st['sb']} {st['sd'][5:]}") if st else ""; body[-1]["bts"] = (f"b{st['ob']} vs b{st['sb']}" if st["ob"] else f"stock b{st['sb']}") if st else ""
        gs = [g for g in (gt, gp) if g]; body[-1]["gm"] = math.prod(gs) ** (1 / len(gs)) if gs else 0   # overall gain: geometric mean of decode and prompt gain
        body[-1]["old"] = [dict(name=r["name"], is_old=True, out="", id=str(orid), sid="", prompt=r.get("prompt") or "", time="", load="", fail=False, t=f(t, 2), p=f(p, 1), st="", sp="", gt=f(gt, 2), gp=f(gp, 2), gtv=gt, gpv=gp, nost=False, gm=0, old=[],
                                bt=f"b{n} {d[5:]} {b}", bts=f"b{n} {b}", st_=None, g=gpus_used(orid)) for b, n, d, t, p, gt, gp, orid in older_gainers(r.get("model"), r.get("test"), st["ob"], st, body[-1]["gm"])] if st and not live and not r.get("ref") else []
    SEC = {"done": 0, "running": 1, "queued": 2}   # `run status`: the sections in this order, each sorted by gain / list order
    if not live: body.sort(key=lambda x: (SEC.get(x["section"], 0), -x["gm"] if x["section"] in (None, "done") else 0))   # biggest overall gainers first (cells without stock last, list order kept)
    allrows = [y for x in body for y in [x] + x.get("old", [])]
    for x in allrows: x.setdefault("xfer", ""); x.setdefault("vt", ""); x.setdefault("vp", ""); x.setdefault("vtv", None); x.setdefault("vpv", None)
    show_x = any(x["xfer"] not in ("", "f32", "f16") for x in body)   # the hand-off column only when a cell of ours ran with other than the f32 default
    wd = lambda k, h: max([len(h)] + [len(x.get(k, "")) for x in allrows])
    W = {k: wd(k, h) for k, h in (("name", ""), ("prompt", "prompt"), ("out", "out"), ("xfer", "xfer"), ("time", "time"), ("load", "load"), ("t", "t/s"), ("p", "pp"), ("g", "par gpus"), ("st", "stock t/s"), ("sp", "stock pp"), ("gt", "x t/s"), ("gp", "x pp"), ("vt", "vs best t/s"), ("vp", "vs best pp"), ("ppl", "ppl us/stock"), ("bt", "builds"), ("id", "id"), ("sid", "stock id"))}
    if live:   # one finished cell as it happens: fixed minimum widths so the rows of successive cells line up
        W["name"] = NAMEW[0] - 1   # the longest name of the list, whatever this cell is called
        for k, mw in (("prompt", 7), ("out", 7), ("xfer", 4), ("time", 5), ("load", 6), ("t", 7), ("p", 8), ("g", 4), ("st", 9), ("sp", 8), ("gt", 7), ("gp", 7), ("vt", 11), ("vp", 10), ("ppl", 13), ("bt", 16), ("id", 5), ("sid", 8)): W[k] = max(W[k], mw)
        for x in allrows: x["bt"] = x.get("bts", x["bt"])   # short builds, so every row decides the same columns
        W["bt"] = max([16, len("builds")] + [len(x["bt"]) for x in allrows])
    def width(keys): return sum(W[k] + 2 for k in keys)
    ours = ["name", "prompt", "out"] + (["xfer"] if show_x else []) + ["time", "load", "t", "p", "g"]; stk = ["st", "sp", "gt", "gp", "vt", "vp", "ppl", "bt", "id", "sid"]
    while width(ours + stk) > cols and len(ours) > 3 and ("load" in ours or "time" in ours): ours.remove("load" if "load" in ours else "time")
    if width(ours + stk) > cols:   # next to go: the dates in the builds column
        for x in allrows: x["bt"] = x["bts"]
        W["bt"] = max([len("builds")] + [len(x["bt"]) for x in allrows] + ([16] if live else []))
    wrap = False   # never a row in two lines (user 2026-10-24: the stock part on a second line under the name was unreadable); columns are dropped above instead
    H = {"prompt": "prompt", "out": "out", "xfer": "xfer", "time": "time", "load": "load", "t": "t/s", "p": "pp", "g": "par gpus", "st": "stock t/s", "sp": "stock pp", "gt": "x t/s", "gp": "x pp", "vt": "vs best t/s", "vp": "vs best pp", "ppl": "ppl us/stock", "bt": "builds", "id": "id", "sid": "stock id"}
    HC = {"prompt": 36, "out": 36, "xfer": 35, "t": 96, "g": 35, "st": 96, "gt": 96, "p": 93, "sp": 93, "gp": 93, "vt": 96, "vp": 93, "ppl": 36, "time": 37, "load": 33, "bt": 90, "id": 37, "sid": 36}
    hl = lambda keys: "".join("  " + col(HC[k], f"{H[k]:<{W[k]}}" if k == "bt" else f"{H[k]:>{W[k]}}") for k in keys)
    out = [" " * W["name"] + hl(ours[1:]) + ("" if wrap else hl(stk))]
    if wrap: out.append(" " * W["name"] + hl(stk))
    if not headers: out = []   # `run status`: rows only
    if hdr_out is not None: hdr_out.extend(out); out = []   # live rows: the header goes to the caller (printed once)
    for x in allrows:
        c = {"prompt": ("36", x["prompt"]), "out": ("36", x["out"]), "xfer": ("35", x["xfer"]), "time": ("37", x["time"]), "load": ("33", x["load"]), "t": ("1;94", x["t"]), "p": ("1;92", x["p"]), "g": ("35", x.get("g", "")), "st": ("94", x["st"]), "sp": ("92", x["sp"])}
        cell = lambda k: "  " + col(c[k][0], f"{c[k][1]:>{W[k]}}")
        gcell = lambda k, v: "  " + (col("1;91" if v < 1 else "37" if v < 2 else "1;97" if v < 10 else "1;93" if v < 40 else "1;92", f"{x[k]:>{W[k]}}") if v is not None else " " * W[k])   # below 1 red, 1-2 plain, 2x bold white, 10x yellow, 40x green
        vcell = lambda k: "  " + (col("1;91" if x[k + "v"] < 0.99 else "1;92" if x[k + "v"] > 1.01 else "37", f"{x[k]:>{W[k]}}") if x.get(k + "v") else col(90, f"{x[k]:>{W[k]}}") if x.get(k) == "first" else " " * W[k])   # against the best earlier run of ours: below 0.99 red (worse), above 1.01 green (better), "first" without one
        ln = col(37 if x.get("is_old") else {"done": "1;92", "running": "1;97", "queued": "90"}.get(x.get("section"), 1), f"{x['name']:<{W['name']}}") + "".join(cell(k) for k in ours[1:]) + (col(31, " FAIL") if x["fail"] else "") + (col(33, " LOOP") if x.get("loop") else "")   # name colour = state (`run status`): green done, white running, grey next
        tail = cell("st") + cell("sp") + gcell("gt", x["gtv"]) + gcell("gp", x["gpv"]) + vcell("vt") + vcell("vp") + "  " + col(36, f"{x.get('ppl', ''):>{W['ppl']}}") + "  " + col(37 if x.get("is_old") else 36, f"{x['bt']:<{W['bt']}}") + "".join("  " + col(37 if k == "id" else 36, f"{x[k]:>{W[k]}}") for k in ("id", "sid"))
        if not x["st"] and not x.get("is_old"):   # no stock row to compare with: the stock cells stay empty, "vs best" (against our own earlier builds) is still shown
            tail = " " * sum(W[k] + 2 for k in ("st", "sp", "gt", "gp")) + vcell("vt") + vcell("vp")
            if not (x.get("vtv") or x.get("vpv") or "first" in (x.get("vt"), x.get("vp"))): tail = ""
        if wrap and tail.strip(): out += [ln, col(37, f"{x['name']:<{W['name']}}") + tail]
        else: out.append(ln + tail)
    if not live and headers:   # the models of the table stay on screen with their results (the model line of the run scrolled away)
        def msz(path):   # size on disk in GiB, all shards of a split model; "" when the file is not here
            try:
                d = os.path.dirname(path); n = sum(os.path.getsize(os.path.join(d, x)) for x in shard_names(os.path.basename(path)))
                return f" {n / 2**30:.0f}g" if n else ""
            except OSError: return ""
        ms_ = list(dict.fromkeys(os.path.basename(r["model"]) + msz(r["model"]) for r in recs if r.get("model")))
        if ms_: out = [col("1;94", "  ".join(ms_))] + out
    return out


LIVE_HDR = [False]


def live_rows(rec):
    """the lines of a finished cell in the table format of the final results (header before the first one)"""
    hd = []; ln = final_table([rec], live=True, hdr_out=hd)
    if not LIVE_HDR[0]: ln = hd + ln
    LIVE_HDR[0] = True
    return "\n".join(ln)


def own_build_no(build):
    """build number from `llama-server --version` of devbuilds/BUILD ('?' when it cannot be read: the display never stops a run)"""
    st = re.fullmatch(r"stock-b(\d+)-official", str(build))
    if st and not os.path.exists(os.path.join(DEVBUILDS, build, "llama-server")): return st.group(1)   # a stock that only the remote machine has: the number is in its name
    try:
        d = os.path.join(DEVBUILDS, build)
        o = subprocess.run([os.path.join(d, "llama-server"), "--version"], capture_output=True, text=True, timeout=30, env={**os.environ, "LD_LIBRARY_PATH": d})
        return (re.search(r"build (\d+)", o.stdout + o.stderr) or [0, "?"])[1]
    except Exception: return "?"


def stock_gain(peek, res, build=None):
    """"  | stock 38.20 t/s pp 1900.0  x1.20 t/s  x5.0 pp  (bNNNN 2026-10-18)": the median speeds of the cell over the newest stock row of the same model and test in runs.db (nothing when there is none)"""
    import sqlite3
    try:   # rows are stored under the test the alias resolves to (pf12k1k is pf12k with a 1024-token answer): the newest stock row with about that answer length; a stock run with a short answer (64 tokens) is not a comparison for a 1k answer, so there is none then
        gen = ALIAS_TOKENS.get(peek[1]); tst = ALIASES.get(peek[1], peek[1])
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        q = "select tps, pp, build, bno, start_ts, rowid from runs where model=? and test=? and build like 'stock-b%-official' and cast(ok as integer)=1 and tps is not null" + HWF[0] + " and coalesce(env,'') not like '%warmpass%'{G} order by start_ts desc limit 12"
        qa = (os.path.basename(peek[0]), tst)
        rs = c.execute(q.replace("{G}", " and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else ""), qa).fetchall()   # the same prompt only: pf12k1k (instruction + 12k source, 1k answer) is stored as pf12k with about 1024 tokens; an old 12k run with a 64-token answer was another prompt
        tm = max(x[0] for x in res) if res else 0
        if build and str(build).startswith("stock"): rs = [x for x in rs if abs(float(x[0]) - tm) > 0.002 * tm][:1]   # a stock cell is compared with the previous stock run (this run is already in the table)
        r = rs[0] if rs else None
        if r and not (build and str(build).startswith("stock")):   # ours against stock: the best run of the newest stock build, like the best run of ours
            same = [x for x in rs if x[2] == r[2]]
            r = (max(float(x[0]) for x in same), max((float(x[1]) for x in same if x[1]), default=None), r[2], r[3], r[4], r[5])
    except sqlite3.Error: return ""
    STOCK_REC[0] = None
    if not r or not r[0]: return ""
    t = max(x[0] for x in res); p = max(x[1] for x in res)
    STOCK_REC[0] = dict(t=float(r[0]), p=float(r[1]) if r[1] else None, ob=own_build_no(build), od=time.strftime("%F", time.localtime(os.path.getmtime(os.path.join(DEVBUILDS, build, "llama-server")))), sb=int(r[3]) if r[3] else (re.search(r"b(\d+)", r[2]) or [0, "?"])[1], sd=str(r[4])[:10], sid=r[5])
    gc = lambda x, good, text: col(good if x >= 1 else "1;91", text)   # the ratio itself bold white (bold red below 1.00); its label keeps the metric's colour (decode blue, prompt green)
    return "  " + col(90, "|") + " " + speed_txt(float(r[0]), float(r[1]) if r[1] else None, stock=True) + gc(t / float(r[0]), "1;97", f"  x{t / float(r[0]):.2f}") + col(96, " t/s") + (gc(p / float(r[1]), "1;97", f"  x{p / float(r[1]):.2f}") + col(93, " pp") if r[1] else "") + "  " + col(90, "(") + col(94, f"b{own_build_no(build)}") + " " + col(33, time.strftime("%F", time.localtime(os.path.getmtime(os.path.join(DEVBUILDS, build, "llama-server"))))) + col(90, " vs ") + col(36, f"b{int(r[3]) if r[3] else (re.search(r'b(\d+)', r[2]) or [0, '?'])[1]}") + " " + col(33, str(r[4])[:10]) + col(90, ")")


def run_watched(cmd, env, expect, label, every=3, total=None, log=None, results=None, peek=None):
    """subprocess.run with a status every `every` seconds (cell_status: stats line, a bar for the cell, a bar for the whole run, the finished runs' speeds); returns the exit code.
    The cell's own output goes to `log` (its text would break the in-place redraw). On a terminal the block is redrawn in place and the next cell redraws over the last one;
    each finished cell adds a line to CELL_LINES (run_list prints them at the end); on a failure the log's last lines print."""
    out = open(log, "w") if log else None
    try: os.remove(LOAD_FILE)
    except OSError: pass
    p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None); t0 = time.time()
    tty = live_tty(); pexp = gen_expect(*peek) if peek else None
    DEC_RATE[0] = dec_rate(*peek) if peek else None
    first = True   # draw at once: the previous cell's finished block must not stay up as this cell's (bars at 0 %)
    try:
      while True:
        try:
            if not first: rc = p.wait(timeout=max(every - 1.5, 0.5)); break
            first = False; raise subprocess.TimeoutExpired(cmd, 0)
        except subprocess.TimeoutExpired:
            pl = peek_line(pexp) if pexp and not NOSTATS[0] else ""   # first: cell_status re-estimates from the same /slots answer
            txt = cell_status(t0, expect, label, total)
            if results and (rl := result_lines(cell_results(results, t0))): txt += "\n" + rl   # per-run speeds so far, below the bars
            if pl: txt += "\n" + pl   # the run in progress: tokens so far against the earlier runs
            txt = fit_block(txt) if tty else txt
            live_clear(); print(("\x1e" if os.environ.get("RUN_DAEMON") else "") + txt, flush=True); LIVE_UP[0] = txt.count("\n") + 1 if tty else 0   # \x1e marks the first line of a status frame for the viewer (ViewFilter: it shows only as many frame lines as its screen has rows)
    except KeyboardInterrupt:   # Ctrl-C: the cell, its bench and its server are stopped for real (kill_tree: `timeout` is a process group of its own, the terminal's SIGINT never reaches them), then run_list ends (list line, kept server, pool)
        live_clear(); print(f"{label}: Ctrl-C, stopping the cell ...", flush=True)
        kill_tree(p.pid); kept_kill()
        raise
    res = cell_results(results, t0) if results else []
    el = time.time() - t0
    STOCK_REC[0] = None
    try: gain = stock_gain(peek, res, cmd[5]) if peek and res else ""
    except Exception as e: gain = f"  | gain: {e}"   # a display problem never ends the list
    CELL_PG[:] = [None, None]
    try: lv = float(open(LOAD_FILE).read()); lt = "" if lv < 0 else "l " + (f"{int(lv)}s" if lv < 100 else f"{int(lv // 60)}m{int(lv % 60):02d}s")
    except (OSError, ValueError): lt = ""
    try:   # the finished cell keeps the live bar's phases: l load, p prompt (load end to first decoded token), g generation (to the end), then the total time of the cell
        _tf = lambda x: f"{x:.0f}s" if x < 60 else f"{int(x // 60)}m{int(x % 60):02d}s"
        _le = os.path.getmtime(LOAD_FILE); _g = PHASE.get("gen") if PHASE.get("t0") == t0 else None
        if lt and _g and _g > _le: lt += f" p {_tf(_g - _le)} g {_tf(time.time() - _g)}"; CELL_PG[:] = [_g - _le, time.time() - _g]
    except (OSError, ValueError, NameError): pass
    if peek: passes_save(peek[0], peek[1], cmd[5]); prompt_passes_save(peek[0], peek[1], cmd[5])
    CELL_REC.append(dict(asked=CELL_ASKED[0], ref=ours_reference(peek[0], peek[1]) if peek and str(cmd[5]).startswith("stock") and res else None, sbno=own_build_no(cmd[5]), xfer="" if str(cmd[5]).startswith("stock") else xfer_of(own_build_no(cmd[5]), " ".join(map(str, cmd))), sdate=time.strftime("%F", time.localtime(os.path.getmtime(os.path.join(DEVBUILDS, cmd[5], "llama-server")))) if os.path.exists(os.path.join(DEVBUILDS, cmd[5], "llama-server")) else "", rid=run_id(f"{cmd[3]}-{peek[1]}") if peek else None, build=str(cmd[5]), prompt=CELL_PROMPT[0], model=peek[0] if peek else None, test=peek[1] if peek else None, name=label, el=el, lt=lt, rc=rc or (0 if res else 1), res=res, st=STOCK_REC[0], stock_row=str(cmd[5]).startswith("stock")))
    try:
        if peek: store_gains(CELL_REC[-1], str(cmd[5]), peek[0], ALIASES.get(peek[1], peek[1]), (re.search(r"hw='(\w+)'", HWF[0]) or [0, 'pc1'])[1], t0)
    except Exception as e: print(f"gains not stored: {e}", flush=True)   # a display problem or a locked db never ends the list
    try:
        if peek and not rc: store_cell_time(str(cmd[5]), peek[0], ALIASES.get(peek[1], peek[1]), (re.search(r"hw='(\w+)'", HWF[0]) or [0, 'pc1'])[1], t0, el, *CELL_PG)
    except Exception as e: print(f"time not stored: {e}", flush=True)
    CELL_LINES.append(col(1, f"{label[:NAMEW[0]]:<{NAMEW[0]}}") + f" {int(el // 60):>3}:{int(el % 60):02d} " + col(37, "cell") + (col(37, ",") + " " + col(33, lt) if lt else "") + "  " + (col(31, "FAILED ") if rc or not res else "") + "  ".join(speed_txt(t, p) for t, p in res) + gain )   # decode blue, prompt green, load yellow: the colors of the status bar
    try:   # the end-of-cell engine state, if the cell saved one: one more status build with it (the live ticks of a short cell miss the engine)
        _ff = os.path.join(QDIR, "final-stats.json")
        if os.path.exists(_ff) and os.path.getmtime(_ff) >= t0:
            STATS_FINAL[0] = open(_ff).read().strip(); LAST_ST[0] = ""; cell_status(t0, expect, label, total)
    except Exception: pass
    finally:
        STATS_FINAL[0] = ""
        try: os.remove(os.path.join(QDIR, "final-stats.json"))
        except OSError: pass
    live_clear(); print(live_rows(CELL_REC[-1]) + ("\n" + result_lines(res) if len(res) > 1 else "") + ("\n" + trace_lines() if trace_lines() else "") + ("\n" + LAST_ST[0] if LAST_ST[0] else ""), flush=True); LAST_ST[0] = ""   # the finished cell keeps its line; the last machine and engine state stays under it (speeds of every run); the next cell's live block is drawn below the growing list
    if out:
        out.close()
        if rc: live_clear(); print("".join(open(log, errors="replace").readlines()[-8:]), end="", flush=True)   # the failure reason
    return rc


def resolve_build(name):
    """a build given by its number (b13563 or 13563) becomes the newest build directory in devbuilds measured under that number (stock-b11729-official for a stock number); a name that is a directory, or anything else, comes back unchanged; exits when the number matches no directory"""
    if not re.fullmatch(r"b?\d{5}", name or "") or os.path.isdir(os.path.join(DEVBUILDS, name)): return name
    import sqlite3
    num = int(name.lstrip("b")); st_dir = os.path.join(DEVBUILDS, f"stock-b{num}-official")
    if os.path.isdir(st_dir): return os.path.basename(st_dir)
    for (bn,) in sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute("select build from runs where bno=? and build not like 'stock%' order by ts desc", (num,)):
        if os.path.isdir(os.path.join(DEVBUILDS, bn)): return bn
    sys.exit(f"build {name}: no build directory in {DEVBUILDS} was measured under that number (a build name works, e.g. um3d)")


def stock_latest(dry=False):
    """The newest official ggml-org release binary (Linux CUDA 13.4 x64 + its cudart libraries, pulled with gh, never built) as devbuilds/stock-bNNNNN-official, and the symlink
    devbuilds/stock-latest to it. Returns the real build name, so rows and logs carry the build number. Downloaded once; a newer release adds a directory, older ones stay. A stock pulled today is reused all day (no gh call)."""
    if getattr(stock_latest, "remote", False):   # a list on another machine (-pc): the newest official release is looked up here (gh, no download) and downloaded by that machine itself; -nodl does not hold it back
        for tag in subprocess.run(["gh", "release", "list", "-R", "ggml-org/llama.cpp", "-L", "8", "--json", "tagName", "-q", ".[].tagName"], capture_output=True, text=True).stdout.split():
            if f"llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz" in subprocess.run(["gh", "release", "view", tag, "-R", "ggml-org/llama.cpp", "--json", "assets", "-q", ".assets[].name"], capture_output=True, text=True).stdout.split():
                return f"stock-{tag}-official"   # the newest release whose binaries are uploaded already (a fresh one may have only some of its files)
    if getattr(stock_latest, "skip", False):   # -nodl: no look at the newest release, no download: the stock-latest link as it is, else the newest stock directory on disk
        have = [p for p in glob.glob(os.path.join(DEVBUILDS, "stock-b*-official")) if os.path.exists(os.path.join(p, "llama-server"))]
        if not have: sys.exit("stock: -nodl, but no stock-bNNNNN-official directory under the builds folder")
        link = os.path.join(DEVBUILDS, "stock-latest")
        name = os.path.basename(os.path.realpath(link)) if os.path.exists(os.path.join(link, "llama-server")) else os.path.basename(max(have, key=lambda p: int(re.search(r"stock-b(\d+)", p).group(1))))
        if not dry and not os.path.islink(link): os.symlink(name, link)
        return name
    today = [p for p in glob.glob(os.path.join(DEVBUILDS, "stock-b*-official")) if os.path.exists(os.path.join(p, "llama-server")) and time.strftime("%F", time.localtime(os.path.getmtime(p))) >= time.strftime("%F", time.localtime(time.time() - 86400 * (getattr(stock_latest, "days", 1) - 1)))]
    if today:   # a stock pulled today stays the day's stock: no new pull, no rerun of the cells that already have its numbers (a new release is taken tomorrow)
        name = os.path.basename(max(today, key=lambda p: int(re.search(r"stock-b(\d+)", p).group(1))))
        if not dry:
            link = os.path.join(DEVBUILDS, "stock-latest"); tl = link + ".new"
            try: os.remove(tl)
            except OSError: pass
            os.symlink(name, tl); os.replace(tl, link)
        return name
    tag = subprocess.run(["gh", "release", "list", "-R", "ggml-org/llama.cpp", "-L", "1", "--json", "tagName", "-q", ".[0].tagName"], capture_output=True, text=True).stdout.strip()
    if not re.fullmatch(r"b\d+", tag): sys.exit(f"stock: cannot read the newest ggml-org release (gh said {tag!r})")
    name = f"stock-{tag}-official"; d = os.path.join(DEVBUILDS, name)
    if not os.path.isdir(d) and not dry:
        import tempfile
        print(f"stock: pulling {tag} ...", flush=True)
        tmp = tempfile.mkdtemp(prefix="stock-dl-", dir=DEVBUILDS); ex = os.path.join(tmp, "x")   # downloads are untrusted data: an empty directory of their own
        try:
            r = subprocess.run(["gh", "release", "download", tag, "-R", "ggml-org/llama.cpp", "-D", tmp, "-p", f"llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz", "-p", f"cudart-llama-{tag}-bin-ubuntu-cuda-13.4-x64.tar.gz"])
            if r.returncode: sys.exit("stock: download failed")
            os.makedirs(ex)
            for f in sorted(glob.glob(os.path.join(tmp, "*.tar.gz"))):
                top = {x.split("/")[0] for x in subprocess.run(["tar", "-tzf", f], capture_output=True, text=True).stdout.split() if x}
                if subprocess.run(["tar", "-xzf", f, "-C", ex] + (["--strip-components=1"] if len(top) == 1 and not top & {"llama-server"} else [])).returncode: sys.exit(f"stock: cannot unpack {f}")
            if not os.path.exists(os.path.join(ex, "llama-server")): sys.exit("stock: no llama-server in the download")
            os.rename(ex, d)
        finally: shutil.rmtree(tmp, ignore_errors=True)
    link = os.path.join(DEVBUILDS, "stock-latest")
    if not dry:
        tl = link + ".new"
        try: os.remove(tl)
        except OSError: pass
        os.symlink(name, tl); os.replace(tl, link)   # the symlink moves to the newest pull
    return name


def run_list_pool(model_bytes, mnt="/mnt/huge1g"):
    """Size the huge-page pool for the next model: ceil(model GiB) pages (GLM 109.4 GiB -> 110, MiMo 107.7 -> 108), no more, so the rest of the RAM stays free and nothing swaps.
    A pool that is too small is unmounted (cached files deleted) and mounted again; one the model fits in is kept, so models run back to back on one pool and one load."""
    need = max(-(-model_bytes >> 30), 1)
    have = int(open(HUGE_NR).read()) if os.path.exists(HUGE_NR) else 0
    why = ""
    if have >= need and os.path.ismount(mnt): return   # a pool the model fits in stays as it is: every unmount fragments RAM further (the stock cells unmount in the caller, then the next model mounts its own size)
    while subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0: time.sleep(5)   # nothing may map the pool while it is rebuilt
    import contextlib, io
    quiet = lambda a: contextlib.redirect_stdout(io.StringIO())   # the mount's own output is long: the list prints one line, "pool 101->108g"
    try:
        with quiet(0): cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
        # the old pool is gone: its RAM is back. Mounting more than the machine can spare is what froze PC1 (110g): check first, mount only what leaves headroom
        avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) >> 20
        if need + 8 <= avail:   # 8 GiB stay free beyond MemAvailable (~117 GiB here): MiMo 108 passes, GLM 110 does not (110g froze PC1 and core-dumped other programs, 2026-10-19)
            with quiet(0): cmd_hugepool(argparse.Namespace(action="mount", what=[f"{need}g"]))
    except SystemExit as e:
        why = "memory fragmented (a reboot clears it)" if "fragmented" in str(e) else str(e)[:70]   # the mount's own text is long: one short reason on the pool line
    got = int(open(HUGE_NR).read()) if os.path.exists(HUGE_NR) else 0
    if 0 < got < need:   # a smaller pool plus the rest pinned from fragmented RAM stalls (PIN_STALL): better no pool, the binary then pins from ordinary RAM, slower but it loads
        try:
            with quiet(0): cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
        except SystemExit as e: print(f"pool: unmount failed: {e}", flush=True)
        got = 0
    if have == got == 0 and getattr(run_list_pool, "told", 0): return   # still no pool: said once per list, not before every model
    run_list_pool.told = 1
    print(col(33, f"pool {have}->{got}g") + (col(31, f" (wanted {need}g, running without a pool" + (f": {why}" if why else "") + ")") if got < need else ""), flush=True)


RUN_LIVE = QDIR + "/run-list.json"   # {pid, log} of the detached list run


def last_run_rec(r, HW):
    """the fields of a cell that is running or queued: its newest earlier measurement in runs.db (the runs of that run's campaign, ok ones), for ours the gain over the newest stock run of the
    same prompt and the build and date that made it; for a stock cell the newest stock run. Looser each time nothing matches: any answer length, then any machine, then any build of ours."""
    import sqlite3
    stock = r["_b"].startswith("stock")
    gen = int(r["tokens"]) if r.get("tokens") else None
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        top = None
        for use_gen, use_hw in ((True, True), (False, True), (False, False)):
            q = "select campaign, rowid, bno, start_ts from runs where model=? and test=? and build " + ("like 'stock-b%-official'" if stock else "not like 'stock%'") + " and cast(ok as integer)=1 and tps is not null and coalesce(env,'') not like '%warmpass%'" + (" and hw=?" if use_hw else "") + (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen and use_gen else "") + " order by start_ts desc limit 1"
            top = c.execute(q, (os.path.basename(r["model"]), ALIASES.get(r["test"], r["test"])) + ((HW,) if use_hw else ())).fetchone()
            if top: break
        if not top: return {}
        res = [(float(t), float(p or 0)) for t, p in c.execute("select tps, pp from runs where campaign=? and cast(ok as integer)=1 and tps is not null order by ts", (top[0],))]
    except sqlite3.Error: return {}
    if stock: return dict(res=res, rid=top[1], st=None, lt=run_times(top[1])[1], sbno=str(top[2] or ""), sdate=str(top[3])[:10])   # the stock numbers in the stock columns, ours empty
    STOCK_REC[0] = None
    try: stock_gain((r["model"], r["test"]), res, r["_b"])
    except Exception: STOCK_REC[0] = None
    if STOCK_REC[0] and top[2]: STOCK_REC[0]["ob"] = int(top[2]); STOCK_REC[0]["od"] = str(top[3])[:10]   # the build that made the run, not the one this list will run
    return dict(res=res, rid=top[1], st=STOCK_REC[0], lt=run_times(top[1])[1])


def cmd_onepass(a):
    """`run.py onepass [-b BUILD] [--write]`: the best one-pass input prompt of every model, from the runs of the list `run onepass -b BUILD` (fork/runs/onepass.csv: every model at 12k down to 4k, two
    runs of 64 answer tokens each; the first run is cold, the last warm and decides). Per model: the sizes that ran in ONE pass (pf_passes = 1 in the last run), the biggest of them whose prompt
    speed is within 3% of the best one-pass size is the pick. --write puts it in models.ini as best_prompt.<N>gpu (N = the cards seen now; run.py reads it for the test pfbest1k)."""
    import sqlite3
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    b = resolve_build(a.build) if a.build != "latest" else a.build
    rows = c.execute("select campaign, model, test, pf_passes, pf_pass, pp, note, ts from runs where campaign like ? and cast(ok as integer)=1 and pp is not null order by ts", (f"{b}-%-p%-pf%k1k",)).fetchall()
    cells = {}
    for camp, model, test, ps, pt, pp, note, ts in rows:
        m = re.search(r"-p(\d+)-pf\d+k1k$", camp)
        if not m: continue
        nt = re.search(r"prompt_n=(\d+)", note or "")
        cells[(model, int(m.group(1)))] = dict(passes=int(ps) if str(ps or "").isdigit() else 0, pass_tok=pt, pp=float(pp), tokens=int(nt.group(1)) if nt else 0)   # the last run of the cell wins (warm)
    if not cells: sys.exit(f"onepass: no rows for build {b}; run `run.py jobs onepass -b {b} -day -nodl` first")
    cards = n_cards(); ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.ini"); picks = {}
    for model in sorted({k[0] for k in cells}):
        sizes = sorted((n for (m_, n) in cells if m_ == model), reverse=True)
        one = [n for n in sizes if cells[(model, n)]["passes"] == 1]
        best_pp = max((cells[(model, n)]["pp"] for n in one), default=0)
        pick = next((n for n in one if cells[(model, n)]["pp"] >= 0.97 * best_pp), None)
        print(col("1;94", model))
        print(col(37, "   size   prompt tokens   passes   pass tokens   prompt t/s"))
        for n in sizes:
            x = cells[(model, n)]
            print(f"  {col('1;92' if n == pick else '1', f'{n:>3}k')}   {x['tokens']:>13}   {x['passes']:>6}   {str(x['pass_tok'] or ''):>11}   {x['pp']:>10.1f}" + (col("1;92", "   <- best one-pass prompt") if n == pick else ""))
        if pick is None: print(col(33, "  no size ran in one pass: nothing to pick (the biggest tested size stays the default)"))
        else: picks[model] = pick
    if a.write and picks:
        lines = open(ini).read().split("\n")
        for model, n in picks.items():
            sec = model_alias(model if os.path.isabs(model) else next((v["path"] for v in fcfg.models_ini().values() if os.path.basename(v.get("path", "")) == model), model))
            if not sec: print(col(31, f"{model}: no models.ini section, not written")); continue
            key = f"{hw_key()}bestpp"
            i0 = next(i for i, l in enumerate(lines) if l.strip() == f"[{sec}]"); i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i].startswith("[")), len(lines))
            while i1 > i0 + 1 and not lines[i1 - 1].strip(): i1 -= 1   # before the blank lines that end the section
            j = next((i for i in range(i0 + 1, i1) if lines[i].split("=")[0].strip() == key), None)
            new = f"{key} = {n}k"
            if j is not None: lines[j] = new
            else: lines.insert(i1, new)
        open(ini, "w").write("\n".join(lines)); print(col(92, f"written to models.ini ({cards} card(s))"))


def cmd_calib(a):
    """`run.py calib [-b BUILD] [-ng N] [-m MODELS] [--dry] [--nowrite]`: find every model's biggest ONE-pass input prompt by bisection and store it in models.ini (best_prompt.<N>gpu, N = the cards seen).
    Starts at the estimated maximum (12k), halves while the prompt needs more than one pass (12, 6, 3, 1), then bisects between the biggest one-pass size and the smallest multi-pass one. The pass
    count never grows when the prompt shrinks, so 3-5 requests (64 answer tokens) per model find the biggest one-pass prompt, which is the pick: bigger prompts amortize the uploads of a pass best."""
    import sqlite3
    if getattr(a, "pass_args", None) is None:   # standalone calib: its own -nm / -a go to every probe (run run passes its own through pass_args)
        a.pass_args = (["-nm", a.no_pool] if getattr(a, "no_pool", None) is not None else []) + (["-a", a.cell_args] if getattr(a, "cell_args", "") else [])
    here = os.path.dirname(os.path.abspath(__file__)); b = resolve_build(a.build) if a.build != "latest" else a.build
    src = [r for r in csv.DictReader(open(os.path.join(here, "runs", "onepass.csv"), newline="")) if r["test"] == "pf12k1k"]
    want = [x for x in (a.models or "").split(",") if x]
    rows = [r for r in src if not want or r["model"] in want]
    if not rows: sys.exit(f"calib: no models match {want}")
    sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12]; c = sqlite3.connect(os.path.join(here, "runs.db")); picks = {}
    def probe(r, n):
        """one request of n k: (passes, pass tokens, prompt tokens, prompt t/s) of its row; None when it did not run"""
        base = r["name"].rsplit("-", 1)[0]; name = f"{base}-p{n}"
        with open(os.path.join(here, "runs", "calib.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys())); w.writeheader(); w.writerow(dict(r, name=name, test=f"pf{n}k1k", n="1"))
        cmd = [sys.executable, os.path.join(here, "run.py"), "jobs", "calib", "-b", b, "-day", "-nodl"] + (["-ng", str(a.n_gpus)] if a.n_gpus is not None else []) + list(getattr(a, "pass_args", None) or [])
        print(col(90, " ".join(cmd)), flush=True)
        if a.dry: return None
        st = os.path.join(QDIR, "calib-state.json")   # what `run status` shows for a calib: the model, its build, when it began and the probe now running
        try: t0_ = json.load(open(st)).get("t0") if json.load(open(st)).get("model") == r["model"] else None
        except (OSError, ValueError): t0_ = None
        json.dump({"model": r["model"], "build": b, "base": base, "now": n, "t0": t0_ or time.time()}, open(st, "w"))
        subprocess.call(cmd)
        x = c.execute("select pf_passes, pf_pass, note, pp, tps from runs where campaign like ? and cast(ok as integer)=1 and pp is not null order by ts desc limit 1", (f"{b}-{name}-pf{n}k1k",)).fetchone()
        if not x: return None
        tk = re.search(r"prompt_n=(\d+)", x[2] or "")
        return int(x[0] or 0), int(x[1] or 0), int(tk.group(1)) if tk else 0, float(x[3]), float(x[4] or 0)
    def fits(r, n):
        """dry allocation request: can the server hold a context of n nominal k? A 2k prompt with one answer token and --ctx for n k (about 1.1k tokens per nominal k plus the answer): the server allocates
        the KV and the pool for that context at start and refuses when it is too big, nothing is prefilled. True when the request ran."""
        base = r["name"].rsplit("-", 1)[0]; name = f"{base}-a{n}"; ctx = int(n * 1100 + 2048)
        with open(os.path.join(here, "runs", "calib.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(r.keys())); w.writeheader(); w.writerow(dict(r, name=name, test="pf2k1k", n="1", tokens="1", extra=f"--ctx {ctx}"))
        cmd = [sys.executable, os.path.join(here, "run.py"), "jobs", "calib", "-b", b, "-day", "-nodl"] + (["-ng", str(a.n_gpus)] if a.n_gpus is not None else []) + list(getattr(a, "pass_args", None) or [])
        print(col(90, f"dry allocation of {ctx} tokens of context: " + " ".join(cmd)), flush=True)
        if a.dry: return False
        subprocess.call(cmd)
        return c.execute("select count(*) from runs where campaign like ? and cast(ok as integer)=1", (f"{b}-{name}-pf2k1k",)).fetchone()[0] > 0
    for r in rows:
        seen = {}   # size (k) -> (passes, pass tokens, prompt tokens, prompt t/s, decode t/s); passes 0 = the model's rows carry no pass count
        def ask(n):
            if n not in seen: seen[n] = probe(r, n)
            return seen[n]
        ceil_k = None   # the engine allocates the pass in its pool and knows the free space: its line "this call fits the ceiling N" (tokens of the biggest pass) in the log of a short probe is the one-pass maximum
        y2 = ask(2)
        if y2 and y2[2]:
            try: tail = open(f"/tmp/perf-{b}.log", "rb").read()[-400000:].decode("utf8", "replace")
            except OSError: tail = ""
            cm = re.findall(r"fits the ceiling (\d+)", tail) or re.findall(r"ubatch \d+ \(max (\d+)\)", tail) or re.findall(r"common_moe_cache_auto_impl: ubatch (\d+):", tail) or re.findall(r"n_ubatch (\d+)", tail)   # the pool-computed pass ceiling, from either log line
            if cm: ceil_k = max(1, int(int(cm[-1]) / (y2[2] / 2)) )   # nominal k whose prompt has that many tokens (tokens per nominal k from the probe itself)
            if ceil_k: print(col(90, f"{r['model']}: the engine's pass ceiling is {cm[-1]} tokens = {ceil_k}k of this text"), flush=True)
            if ceil_k and not a.nowrite and not a.dry and re.findall(r"fits the ceiling (\d+)", tail): ini_set_maxpp(r["model"], int(cm[-1]))   # only the engine's own ceiling line; ubatch / n_ubatch lines of a model that fits the VRAM are the default micro-batch, not a limit   # kept per VRAM next to bestpp
            if a.ceil: continue   # --ceil: only the engine's ceiling, no search
        elif a.ceil: print(col(31, f"{r['model']}: the probe did not run, no ceiling read")); continue
        mp_ = config.model(r["model"]); mp_ = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", mp_)) or [mp_])[0]
        fit, bpt, gib = feasible_ctx(mp_, int(hw_key()[:-1]))
        pps = [x[0] for x in c.execute("select pp from runs where model=? and test like 'pf%' and cast(ok as integer)=1 and pp is not null order by ts desc limit 20", (os.path.basename(mp_),))]
        if pps: fit = min(fit, int(statistics.median(pps) * 600))   # time is a resource too: a probe may take 10 minutes of prefill at the speed the model last showed (feedback: no cell over 10-15 min)
        n = ceil_k or max(1, int((fit - 2048) * 0.9 / 1000))   # no static start: the biggest prompt the machine can hold, by arithmetic (feasible_ctx), minus the answer and a margin; ~1.1 tokens per nominal token on the big tokenizers
        print(col(90, f"{r['model']}: KV {bpt} bytes per token, {gib:.0f} GiB for it: {fit} tokens fit (memory, the trained context and 10 minutes of prefill cap it): start at {n}k"), flush=True)
        if ceil_k: n = min(ceil_k, n)   # the engine's pass ceiling is where one pass ends: the search starts there
        while True:   # the maximum that fits: a prompt the server cannot allocate or run halves
            x = ask(n)
            if x is not None or n <= 1: break
            n //= 2
        if x is None: print(col(33, f"{r['model']}: no size ran: nothing stored")); continue
        score = lambda d: {k: v[3] / max(w[3] for w in d.values()) + v[4] / max(w[4] for w in d.values()) for k, v in d.items() if v and v[4]}
        # from the biggest one-pass size (the pool's ubatch): the half of it; while the half scores better keep halving, when it scores worse look between (up) again: the score is prompt speed plus decode speed
        # each over the best seen, a smaller prompt can win and so can more than one pass
        if True:
            cur = n
            while cur > 1:
                y = ask(cur // 2)
                if y is None or not y[4]: break
                sc = score({k: v for k, v in seen.items() if v})
                if sc.get(cur // 2, 0) > sc.get(cur, 0): cur //= 2
                else: break
            for _ in range(3):
                sc = score({k: v for k, v in seen.items() if v}); best = max(sc, key=lambda k: (sc[k], k)); ks = sorted(sc)
                lo_ = max((k for k in ks if k < best), default=best // 2); hi_ = min((k for k in ks if k > best), default=best)   # nothing above the pool's ubatch: a bigger prompt only adds passes, and every pass uploads the experts again
                new = [m for m in ((lo_ + best) // 2, (best + hi_) // 2) if m >= 1 and m not in seen and abs(m - best) >= 1]
                if not new: break
                for m in new: ask(m)
            ok_ = {m: v for m, v in seen.items() if v and v[4]}; lo = max(ok_, default=0)
        if not ok_: print(col(33, f"{r['model']}: no size with a decode speed: nothing stored")); continue
        mp, mt = max(v[3] for v in ok_.values()), max(v[4] for v in ok_.values())
        eff = {n: v[3] / mp + v[4] / mt for n, v in ok_.items()}; k = max(eff, key=lambda n: (round(eff[n], 6), n))
        picks[r["model"]] = (k, seen[k][2]); print(col("1;92", f"{r['model']}: best prompt {k}k ({seen[k][2]} tokens, {seen[k][0]} pass(es), prompt {seen[k][3]:.0f} t/s, decode {seen[k][4]:.1f} t/s); biggest one-pass {lo}k"))
        for n in sorted(eff): print(f"   {n:>3}k  {ok_[n][2]:>6} tokens  {ok_[n][0]} pass(es)  prompt {ok_[n][3]:>6.0f} t/s  decode {ok_[n][4]:>5.1f} t/s  score {eff[n]:.3f}  total {ok_[n][2] / ok_[n][3] + 1024 / ok_[n][4]:>5.1f} s")
    # nothing is written here: every finished sized cell updates models.ini by itself when its request time per token beats the model's best so far (learn_bestpp)


def running_cell():
    """(tag, start time) of the cell a list runs now: the `run.py cell TAG ...` process (tag = BUILD-NAME, what the campaign of its rows starts with); (None, None) when no cell runs"""
    boot = next((int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime")), 0)
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            a = open(f"/proc/{p}/cmdline", "rb").read().decode().split("\0")
            if len(a) > 3 and a[1].endswith("run.py") and a[2] == "cell":
                return a[3], boot + int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError): continue
    return None, None


def queue_waiting():
    """names of the tasks the queue daemon has waiting (tasks.csv status next, in its order); none when the daemon never ran"""
    try: rows = list(csv.DictReader(open(os.path.join(QDIR, "queue", "tasks.csv"), newline="")))
    except OSError: return []
    return [r["campaign"] or r["id"] for r in sorted((r for r in rows if r["status"] == "next"), key=lambda r: float(r["order"] or 0))]


def cmd_status(a):
    """`run.py status [LIST [run options]]`: what the running list is doing, in the table `run LIST -nrt` prints: the cells done today (t/s, pp, stock, gains), the one running now, the
    cells queued next (with the time each is expected to take) and the queue daemon's waiting tasks. With no LIST: the list that runs now, else the last one that ran. Starts and touches nothing."""
    last = os.path.join(QDIR, "run-list-last.json")
    try: j = json.load(open(RUN_LIVE)); os.kill(j["pid"], 0); live = True
    except (OSError, ValueError, KeyError): live = False
    lists = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "*.csv")))
    if a.list and a.list not in lists: sys.exit(f"status: no list {a.list!r}; lists: {' '.join(lists)}")
    if a.list: name, argv = a.list, list(a.rest)
    else:
        try: j = json.load(open(last)); name, argv = j["name"], j["argv"]
        except (OSError, ValueError, KeyError): sys.exit(f"status: no list has run since this command exists: run.py status LIST [-b BUILD]; lists: {' '.join(lists)}")
        print(col(92, f"list {name}: running") if live else col("1;91", f"NOTHING IS RUNNING. The table is the last list that ran ({name}); rows with only a ~time are cells it planned and never ran"), flush=True)
    if name == "calib":   # a calib is one list per probe: the probes done, the one running, what comes next
        try: cs = json.load(open(os.path.join(QDIR, "calib-state.json")))
        except (OSError, ValueError): cs = None
        if cs:
            import sqlite3
            db = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
            rows = db.execute("select campaign, pf_passes, note, pp, tps, ts from runs where campaign like ? and cast(ok as integer)=1 and pp is not null order by ts", (f"{cs['build']}-{cs['base']}-p%-pf%k1k",)).fetchall()
            print(col(1, f"calib {cs['model']} on {cs['build']}: probes done"), flush=True)
            last = {}
            for cp, ps, nt, pp, tps, ts in rows:
                if ts is not None and ts >= cs["t0"] - 5: last[cp] = (ps, nt, pp, tps)
            for cp, (ps, nt, pp, tps) in last.items():
                tk = re.search(r"prompt_n=(\d+)", nt or ""); print(f"  {cp.split('-')[-2] if cp.count('-') > 1 else cp:<8} {tk.group(1) if tk else '?':>6} tokens  {ps or 0} pass(es)  prompt {pp:6.0f} t/s  decode {tps or 0:5.1f} t/s")
            print(col(33, f"  now: probe {cs['base']}-p{cs['now']} ({cs['now']}k)") if live else col(90, f"  last probe: {cs['now']}k; the calib is not running"), flush=True)
            if live: print(col(90, "  next: the search halves while the score rises, then refines around the best (decided from the result of the probe now running)"), flush=True)
            return 0
    return run_list(name, [x for x in argv if x not in ("-n", "--dry", "--status")] + ["--dry", "--status"])


class ViewFilter:
    """The viewer's own redraw (`jobs`, attach). The detached run writes its status block with cursor-up codes counted in its own lines. A terminal narrower than a line wraps it,
    the cursor-up then goes short and the block repeats down the screen; a block taller than the screen scrolls its top out of reach. The detached run cannot know the viewer's
    terminal, so the viewer does the fitting itself: every line is clipped to the terminal width (queried on every chunk, so a resize counts; ANSI codes do not count, nothing wraps,
    one logical line is exactly one row) and a block that is taller than the screen is cleared with the whole screen instead of cursor-up (the rows above are out of reach).
    Old logs replay through the same filter, so a replayed history draws like a live one."""
    CLEAR = re.compile(r"\033\[(\d+)A\033\[J")

    def __init__(self):
        import codecs, collections
        self.dec = codecs.getincrementaldecoder("utf-8")("replace"); self.buf = ""; self.rows = collections.deque(maxlen=600)
        self.frame = False; self.budget = 0; self.hidden = 0

    @staticmethod
    def clip(line, w):
        vis = 0; i = 0
        while i < len(line):
            if (m := ANSI_RE.match(line, i)): i = m.end(); continue
            if vis >= w: return line[:i] + "\033[0m"
            vis += 1; i += 1
        return line

    def feed(self, data):
        cols, rows_h = shutil.get_terminal_size((120, 40)); w = max(20, cols - 1); out = []
        self.buf += self.dec.decode(data)
        while True:
            m = self.CLEAR.search(self.buf); nl = self.buf.find("\n")
            if m and (nl < 0 or m.start() < nl):
                out.append(self.buf[:m.start()]); self.buf = self.buf[m.end():]
                n = int(m.group(1)); took = 0; self.frame = False
                for _ in range(min(n, len(self.rows))): took += self.rows.pop()
                out.append("\033[H\033[2J" if took >= rows_h - 1 else (f"\033[{took}A\033[J" if took else "\033[J"))
            elif nl >= 0:
                line = self.buf[:nl]; self.buf = self.buf[nl + 1:]
                if line.startswith("\x1e"): line = line[1:]; self.frame = True; self.budget = max(8, rows_h - 3)
                if self.frame:   # a status frame: no more rows than the screen has (the rest stays in the log); the last row says how many are hidden
                    if self.budget <= 0: self.rows.append(0); self.hidden += 1; continue
                    if self.budget == 1:
                        self.budget = 0; self.rows.append(1)
                        out.append(self.clip(f"... the rest of the status is hidden: this terminal has {rows_h} rows (the log has it all)", w) + "\n"); continue
                    self.budget -= 1
                out.append(self.clip(line, w) + "\n"); self.rows.append(1)
            else:
                break
        return "".join(out)


def run_list_detached(name, argv):
    """`run NAME ...` runs the list in a detached process (own session, output in a log) and shows that log: Ctrl-C only closes the view, the run goes on.
    `run NAME ...` again while one runs attaches instead of starting a second: the log is replayed from its start (the status block's cursor codes redraw
    the screen as it was) and followed. `run stop` stops it for real. A dry list (-n) and the detached process itself run inline."""
    if os.environ.get("RUN_DAEMON"):
        try: return run_list(name, argv)
        finally:
            try: os.remove(RUN_LIVE)
            except OSError: pass
    if "-n" in argv or "--dry" in argv: return run_list(name, argv)
    def alive():
        try:
            rec = json.load(open(RUN_LIVE)); os.kill(rec["pid"], 0)
            return rec
        except (OSError, ValueError, KeyError): return None
    rec = alive(); child = None
    if name is None and not rec and subprocess.call(["pgrep", "-f", "[r]un.py calib"], stdout=subprocess.DEVNULL) == 0:   # a calib is between two probes (each probe is a list of its own): wait for the next one
        print("calib is between two probes: waiting for the next one ...", flush=True)
        for _ in range(180):
            time.sleep(1); rec = alive()
            if rec or subprocess.call(["pgrep", "-f", "[r]un.py calib"], stdout=subprocess.DEVNULL) != 0: break
    if name is None and not rec:   # plain `jobs`: attach only; with none going, wait for the next one (user 2026-10-24: wait instead of exiting)
        print("no detached run is going: waiting for one to start (Ctrl-C quits) ...", flush=True)
        try:
            while not rec:
                time.sleep(1); rec = alive()
        except KeyboardInterrupt:
            print(); return 1
    if rec: print(f"a run is going (pid {rec['pid']}): attaching, Ctrl-C detaches, `run stop` stops it", flush=True)
    else:
        log = f"{QDIR}/run-list-{time.strftime('%m%d-%H%M%S')}.log"
        with open(log, "ab") as lf:
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "jobs", name] + list(argv), stdin=subprocess.DEVNULL, stdout=lf, stderr=lf,
                                 env=dict(os.environ, RUN_DAEMON="1", PYTHONUNBUFFERED="1", COLUMNS=str(shutil.get_terminal_size((100, 24)).columns)), start_new_session=True)
            child = p
        json.dump({"pid": p.pid, "log": log}, open(RUN_LIVE, "w")); rec = {"pid": p.pid, "log": log}
        json.dump({"name": name, "argv": list(argv)}, open(QDIR + "/run-list-last.json", "w"))   # `run status` without a list shows this one when none runs
    import signal
    def kill_run(*_):   # SIGINT to the detached run: its Ctrl-C path stops the cell, the server, then the list ends and gives the pool back
        print("\nstopping the run ...", flush=True)
        try: os.kill(rec["pid"], signal.SIGINT)
        except OSError: pass
    signal.signal(signal.SIGQUIT, kill_run)   # Ctrl-\ (a terminal's Shift+Ctrl+C is copy, no signal reaches us)
    vf = ViewFilter() if sys.stdout.isatty() else None   # on a terminal the viewer fits the block to its own width and height; into a pipe or a file the log goes through as it is
    def show(chunk):
        if vf: sys.stdout.write(vf.feed(chunk)); sys.stdout.flush()
        else: sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    try:
        again = False
        with open(rec["log"], "rb") as f:
            while True:
                chunk = f.read(65536)
                if chunk: show(chunk); continue
                try:
                    if child and child.poll() is not None: raise OSError   # our own child: a finished one is a zombie that kill(0) still finds
                    os.kill(rec["pid"], 0)
                except OSError:
                    chunk = f.read()   # the last lines written before it ended
                    if chunk: show(chunk)
                    if name is None and not child:   # plain `jobs` is a viewer: the run ended, wait for the next one instead of exiting (user 2026-10-24)
                        print("\nthe run ended.\n", flush=True)
                        again = True
                        break
                    return 0
                time.sleep(0.5)
        # the next run, with the old log closed first (2026-10-24: a viewer that re-attached from inside the open log kept every old log open, one level deeper per run)
        if again: return run_list_detached(None, [])
    except KeyboardInterrupt:
        print("\ndetached: the run goes on (Ctrl-C again within 2 s, or Ctrl-\\, stops it); `run NAME ...` re-attaches, `run stop` stops it", flush=True)
        try: time.sleep(2)
        except KeyboardInterrupt:
            kill_run()
            try:
                for _ in range(240):   # the stop takes a while (cell, server, pool): wait until the run is gone
                    try:
                        if child and child.poll() is not None: break   # our own child: a finished one is a zombie that kill(0) still finds (the wait ran the full 2 min)
                        os.kill(rec["pid"], 0)
                    except OSError: break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nthe stop is going on in the background; `run stop` finishes it", flush=True)
        return 0


def run_list(name, argv):
    """`run.py jobs NAME [-b B] [-o TEXT] [-s] [-dsc] [-nm] [-w] [-r N] [-n]`: run the cells listed in fork/runs/NAME.csv (columns name, model, n, test, tokens, args, extra), one after the other,
    each through `cell` (waits for free GPUs, checks the pool, logs to q5h). The campaign of a row is B-name (B defaults to `latest`, the newest dev build), so a rerun
    with another build never collides. A failed cell is reported and the list goes on. -o keeps the rows whose name contains TEXT; -d runs the biggest models first, a model's rows together. Before each new model the pool is
    rebuilt to its size (run_list_pool; --no-pool skips that). Lists: gainers, reg, ctx."""
    ap = argparse.ArgumentParser(prog=f"run.py jobs {name}")
    ap.add_argument("-b", "--build", default="latest"); ap.add_argument("-onl", "-o", "--only", help="rows whose name contains TEXT; comma list: glm-,mimo-"); ap.add_argument("-t", "--test", help="rows whose test is TEXT (comma list): best1k = only the best-prompt input rows, edit4c, tetrisc"); ap.add_argument("-n", "--dry", action="store_true")
    ap.add_argument("-skp", "--skip", help="skip rows whose name contains TEXT; comma list: glm-,mimo-")
    ap.add_argument("-ng", "--gpus", dest="n_gpus", type=int, metavar="N", help="use only N cards: the best N (fastest slot, then fastest card), the fastest one last; 0 = CPU only. Remote machines (-pc) and this host")
    ap.add_argument("-ns", "--no-stats", dest="no_stats", action="store_true", help="no live stats: nothing samples the machine or polls the server while a cell runs (status probes, nvidia-smi / sys / slots samplers on a remote box); only the clock is shown. For timings the monitor cannot touch")
    ap.add_argument("-sc", "-su", "--stock-first", action="store_true", help="every cell runs on the newest official stock binary first, then on -b (same session, so the gain line compares like with like)")
    ap.add_argument("-day", "--day", action="store_true", help="the day power limits (x4 card 220 W, x16 card 280 W) for the whole run, also at night (the schedule drops to 200/200 W from 23:50): same as -pl 220/280")
    ap.add_argument("--status", action="store_true", help="print the table of the cells run today (the -nrt table: t/s, pp, stock, gains) and the cells still to run; nothing is run, started or touched (the web view calls this)")
    ap.add_argument("-nrt", "--not-today", nargs="?", type=int, const=1, default=0, metavar="DAYS", help="skip cells (and so models) that already have their runs in runs.db from the last DAYS days (default 1 = today); a stock pulled within those days is not replaced by a newer release")
    ap.add_argument("-pin", dest="force_pin", action="store_true", help="force pinned RAM for every cell (-lm pin) instead of the default auto; `-nm` is the mmap counterpart. In a row's args column: -pin")
    ap.add_argument("-nm", "--no-mount", dest="no_pool", nargs="?", const="0", default=None, metavar="GB", help="no mount: leave the huge-page pool as it is, at the start and at the end. With GB (-nm 101) our cells also load mmap and pin at most that many GiB of the model (tensor order), the rest is mapped")
    ap.add_argument("-s", "--stock", action="store_true", help="run on the newest official ggml-org release binary (pulled once into devbuilds/stock-bNNNNN-official; stock-latest points to it)")
    ap.add_argument("-w", "--warm", action="store_true", help="a warm-up pass before the measured runs (same prompt again; stored with temp=warmpass, not in the statistics); default off")
    ap.add_argument("-cs", "--clean-state", action="store_true", help="every cell starts from an EMPTY learned state of its own (bench --clean-state): run 1 is the cold start of a new user, run 2 starts a new server on the state run 1 left (warm). The server is not kept between cells, so no cell inherits another's state (an A/B of two builds read in run order once gave a 37%% gap that was only the shared state file)")
    ap.add_argument("-nk", "-x", "--kill", action="store_true", help="kill the server after every run (default: it stays up and the next run, also of the next cell, uses it when model, build and args are the same; later runs then start with what the earlier ones cached)")
    ap.add_argument("-r", "--runs", default="2", help="measured runs per cell (default 2: one run can be an outlier; 0 = the list's n)")
    ap.add_argument("-fb", "-from-best", "--from-best", dest="from_best", action="store_true", help="cells with the biggest known gain over stock first (from runs.db: newest runs of ours against the newest official stock; cells with no gain known last), so the best gainers run and print first")
    ap.add_argument("-dsc", "-d", "--desc", "-from-biggest", "--from-biggest", dest="desc", action="store_true", help="biggest model files first (same-size rows keep the list order): the pool fragments with each unmount, so the big models go while it is still whole")
    ap.add_argument("-nodl", "--skip-download", dest="nodl", action="store_true", help="no check for a newer official stock release and no download: the stock already on disk (stock-latest) is used")
    ap.add_argument("-draft", "--draft", dest="draft", type=int, metavar="N", help="speculative draft on OUR cells of every model that has a `draft =` file in models.ini (ds: the DSpark draft): depth N (--spec-draft-n-max N and --moe spec-depth=N, the tuner out of the loop); the row name gets -draftN; stock cells and models without a draft file run as before")
    ap.add_argument("-a", "--cell-args", default="", metavar="ARGS", help="extra cell arguments for OUR cells (not stock), e.g. -a \"moe:split-f16=1\" (one moe: word per --moe key); stored in the args column of the runs, so the rows stay apart from the defaults. -nrt matches by cell, not by arguments: leave it off for an A/B")
    ap.add_argument("-pt", "--prompt-type", dest="pt", metavar="LIST", help="only the cells of these prompt types (comma list): i = input-heavy (mimo-i), io = input+output-heavy (mimo-io), o = output-heavy (mimo-o); the last word of the row name")
    ap.add_argument("-ps", "--prompt-size", dest="ps", action="append", metavar="TEST", help="run THESE sizes instead of the list's own, for the models the other filters leave (repeatable): 100k4k = the source cut at 100k tokens behind an instruction, 4000 answer tokens; 100k = prompt only (the list's answer length); tetrisc4k = that test with 4000 answer tokens; any other name is a test as it is")
    ap.add_argument("-ub", "--ubatch", dest="ub", metavar="LIST", help="micro-batch clamps, a comma list (4k,8k,16k; k = 1024): every selected cell once per size with the server argument -ub N, which clamps the largest micro-batch per card (the automatic one is never raised); rows are named <name>-ub<size>")
    ap.add_argument("-sim", dest="hog_gpu", help="VRAM left free on every GPU for OUR cells (the server takes the rest at start: --moe sim-vram): -sim 16g = every card behaves like a 16 GiB card; a bare number is GiB, m = MiB")
    ap.add_argument("-simram", dest="hog_ram", help="RAM left free for OUR cells (--moe sim-ram): -simram 32g; a bare number is GiB, m = MiB")
    ap.add_argument("-ds", "--dense-split", dest="dense_split", choices=["on", "off", "one", "auto"], help="force the dense split for OUR cells: on = --moe dense-split=1, off = 0 (expert split alone), one = copies made but every prompt pass on one card, auto = the engine's own choice (the default); stored in the args column like -a")
    ap.add_argument("-pl", "--power", metavar="W", nargs="?", const="max", help="power limits in W for the whole run instead of the day/night schedule: no value = every card at its maximum; 285 = all cards; 285,285,300,285 = per card; 220/280 = cards on a narrower link than the widest / the widest (PC1: x4 card, x16 card). The max of the x4 card throttled and crashed it once on PC1 (330-350 W): not for PC1's x4 card. A remote host may refuse (blg: Insufficient Permissions)")
    ap.add_argument("-from-smallest", "--from-smallest", dest="asc", action="store_true", help="smallest model files first: the first results come quickly")
    ap.add_argument("-pc", "--machine", default="pc1", help="run the cells on another machine over ssh (`run.py machine`): the builds must be there (`run.py remote ID push BUILD...`), the model is found there by file name, the server and the request run on that machine (`run.py remote`), run.py stays here; -nrt does not apply")
    ap.add_argument("-ssh", "--ssh", metavar="LINE", help="the ssh line of a rented box for this session, in quotes (\" -p 47086 root@93.91.156.104 -L 8080:localhost:8080\"): the machine named by -pc (cld when -pc is not given) is set to it first (~/.ssh/config and machines.csv), see `run.py machine add`")
    o = ap.parse_args([x for x in argv if x != "desc"])   # `run gainers desc` works too
    NOSTATS[0] = o.no_stats; NGPU[0] = o.n_gpus
    for opt, key in ((o.hog_gpu, "sim-vram"), (o.hog_ram, "sim-ram")):   # GiB -> the server's MiB
        if opt:
            v = str(opt).lower().rstrip("ib").rstrip("b")   # 16g, 16gb, 16gib, 16384m, 16
            mib = float(v[:-1]) if v.endswith("m") else float(v[:-1] if v.endswith("g") else v) * 1024
            o.cell_args = (o.cell_args + f" {key}={int(mib)}").strip()
    if o.dense_split: o.cell_args = (o.cell_args + " dense-split=" + {"on": "1", "off": "0"}.get(o.dense_split, o.dense_split)).strip()
    if o.status: o.not_today = o.not_today or 1; o.nodl = True
    if o.day and not o.power: o.power = "220/280"
    if o.no_pool is not None: os.environ["RUN_KEEP_POOL"] = "1"   # -nm: the cells never unmount the pool either (pool_sanity); a model bigger than the pool loads without it, the pool stays for the next ones
    if o.no_stats: os.environ["RUN_NS"] = "1"
    if o.n_gpus is not None: os.environ["RUN_NG"] = str(o.n_gpus)
    if o.n_gpus is not None and o.machine == "pc1":   # this host: the children (cells, servers) see the chosen cards only, numbered in the listed order
        _vis = pick_gpus(o.n_gpus, subprocess.run(GPU_Q_NG, shell=True, capture_output=True, text=True).stdout) if o.n_gpus > 0 else ""
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=_vis, PERF_DEV=",".join(f"CUDA{i}" for i in range(len(_vis.split(",")))) if _vis else "none")
        print(f"-ng {o.n_gpus}: cards {_vis or 'none (CPU only)'} (fastest last)", flush=True)
    if o.ssh:   # a temporary session: update the machine's ssh entry before anything talks to it
        if o.machine == "pc1": o.machine = "cld"
        cmd_machine(argparse.Namespace(action="add", id=o.machine, ssh=o.ssh.strip(), root=None, os="linux", quiet=True))
    if o.power:
        if not re.fullmatch(r"max|\d{2,3}(/\d{2,3}|(,\d{2,3})*)", o.power): sys.exit("-pl: no value = every card at its maximum; W for every card (285), or per card (285,285,300,285), or narrow-link cards/widest-link cards (220/280)")
        if o.machine.split("-")[0] == "pc1": os.environ["PERF_POWER"] = o.power   # the cell processes inherit it; wait_vram_free applies it before every cell   # the cell processes inherit it; wait_vram_free applies it before every cell
    HW = o.machine.split("-")[0]; HWF[0] = f" and hw='{HW}'"   # rows of this machine only count for -nrt and for the stock check (pc1-cli is pc1)
    stock_latest.remote = o.machine.split("-")[0] != "pc1"; stock_latest.skip = o.nodl; stock_latest.days = max(o.not_today, 1)
    if o.stock: o.build = stock_latest(o.dry)
    o.build = resolve_build(o.build)   # -b b13563: a build number
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name + ".csv"), newline="")))
    for r in rows: r["model"] = fcfg.model(r["model"])   # the model column is a models.ini alias or a path
    for r in rows: r["test"] = test_name(r["test"])   # the test column may say 8k / 60k1k
    _picked = set()
    _sel = lambda r: (not o.only or any(r["name"] == x[:-1] if x.endswith("$") else x in r["name"] for x in o.only.split(","))) and (not o.skip or not any(x in r["name"] for x in o.skip.split(",")))   # the -onl and -skp filters below, applied early: a model nobody asked for (or one skipped) is not looked at
    if o.machine.split("-")[0] == "pc1" and not o.status:   # a model whose rows ask for its best prompt (pfbest) and that has none stored: ask the engine once, here, in this window (calib --ceil), not "go and run calib"
        for _m in sorted({r["model"] for r in rows if _sel(r) and r["test"].startswith("pfbest")}):
            if prompt_known(_m): continue
            if o.dry: print(col(33, f"{model_alias(_m)}: no best prompt or pass ceiling stored for {hw_key()} of VRAM: a real run asks the engine first (calib --ceil); its rows are left out of this listing"), flush=True); rows = [r for r in rows if r["model"] != _m]; continue
            print(col("1;94", f"{model_alias(_m)}: no best prompt or pass ceiling stored for {hw_key()} of VRAM: asking the engine once (calib --ceil)"), flush=True)
            # the probe runs with this run's pinning and cell arguments: the ceiling depends on the RAM left beside the pinned weights
            _pa = (["-nm", o.no_pool] if o.no_pool is not None else []) + (["-a", o.cell_args] if o.cell_args else [])
            cmd_calib(argparse.Namespace(build=o.build, n_gpus=o.n_gpus, models=model_alias(_m), dry=False, nowrite=False, ceil=True, pass_args=_pa))
            if not prompt_known(_m): sys.exit(f"{model_alias(_m)}: calib --ceil stored no pass ceiling (the engine gave none): fix that first, nothing is guessed")
    for r in rows:
        if not _sel(r): continue
        was = r["test"]; r["test"] = best_test(r["model"], r["test"], remote=o.machine.split("-")[0] != "pc1")
        if r["test"] != was and was != r["test"] and (r["model"], r["test"]) not in _picked and not o.status:
            _picked.add((r["model"], r["test"]))   # pfbest1k: the model's best one-pass prompt (models.ini best_prompt), stock cells run the same
    if o.test: rows = [r for r in rows if r["test"] in {x if x.startswith("pf") or x in {r2["test"] for r2 in rows} else "pf" + x for x in o.test.split(",")}]   # best1k = pfbest1k (plain names, no pf prefix)
    rows = [r for r in rows if not o.only or any(r["name"] == x[:-1] if x.endswith("$") else x in r["name"] for x in o.only.split(","))]   # a trailing $ means the exact name: mimo-i$ is not mimo-io
    if o.pt: rows = [r for r in rows if r["name"].rsplit("-", 1)[-1] in o.pt.split(",")]   # before -ps renames the rows
    if o.ps:   # one row per model (its first row keeps the args and the extra words) and per asked size
        _first = {}
        for r in rows: _first.setdefault(r["model"], r)
        _known = {r["test"] for r in rows} | set(TESTS) | set(ALIASES) | set(ALIASES.values())
        def _ps(p, r):   # 100k4k = prompt of 100k tokens + 4000 answer tokens; 100k = prompt only (the list's answer length); tetrisc4k = that test, 4000 answer tokens
            if (m := re.fullmatch(r"(\d+)m((?:\d+k)?)", p)): p = f"{int(m[1]) * 1024}k{m[2]}"   # 1m1k = 1024k1k
            if (m := re.fullmatch(r"(\d+)k(?:(\d+)k)?", p)):   # the source cut at that many tokens behind an instruction (PS_INSTR); an instruct model needs its chat template
                # --keep-loops: beyond the source's length the prompt repeats it, and an answer about a repeated text may loop; repeating the request (3 more
                # full prompts: ~15 min ours, ~3.5 h stock at 200k) cannot change that. The row keeps loop=1 (out of every statistic), the prompt speed stands.
                return dict(r, test=f"pf{m[1]}k", **({"tokens": str(int(m[2]) * 1000)} if m[2] else {}), extra=((r.get("extra") or "") + " --keep-loops --chat " + shlex.quote(PS_INSTR)).strip())
            if (m := re.fullmatch(r"([a-z]\w*?)(\d+)k", p)) and m[1] in _known: return dict(r, test=m[1], tokens=str(int(m[2]) * 1000))
            return dict(r, test=p if p.startswith("pf") or p in _known else "pf" + p)
        rows = [dict(_ps(p, r), name=r["name"].split("-")[0] + "-" + p) for r in _first.values() for p in o.ps]
    rows = [r for r in rows if not o.skip or not any(x in r["name"] for x in o.skip.split(","))]
    if o.ub:   # the micro-batch sweep: each row once per size
        _sz = [(x, int(x[:-1]) * 1024 if x.lower().endswith("k") else int(x)) for x in o.ub.split(",") if x]
        rows = [dict(r, name=f"{r['name']}-ub{x.lower()}", args=((r.get("args") or "") + f" -ub {n}").strip()) for r in rows for x, n in _sz]   # -skp: same matching as -onl, rows dropped
    for r in rows:
        if o.runs not in (None, "0"): r["n"] = o.runs
    if o.from_best:   # the rows (a stock cell stays with its row) by the gain known from runs.db
        gk = {}
        for r in rows:
            try: gk[(r["name"], r["test"])] = known_gain(r["model"], r["test"])
            except Exception: gk[(r["name"], r["test"])] = 0
        rows.sort(key=lambda r: -gk[(r["name"], r["test"])])
    sb = stock_latest(o.dry) if o.stock_first and not o.stock else None
    if o.not_today and not o.stock and not sb: sb = stock_latest(o.dry)   # -nrt also runs the stock cell of a row that has no stock run of its answer length yet (gap filling)
    def stock_missing(r):
        import sqlite3
        gen = int(r["tokens"]) if r.get("tokens") else None
        try:
            q = "select 1 from runs where model=? and test=? and build like 'stock-b%-official' and cast(ok as integer)=1 and tps is not null and coalesce(env,'') not like '%warmpass%' and hw='" + HW + "'" + (" and n_gen between %d and %d" % (gen // 2, gen * 2) if gen else "") + " limit 1"
            c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")); a = (os.path.basename(r["model"]), ALIASES.get(r["test"], r["test"]))
            for tb in ("runs", "bad_runs"):   # failed runs are moved to bad_runs (run.py bad-runs)
                try:
                    if c.execute(f"select 1 from {tb} where model=? and test=? and build like 'stock%' and cast(ok as integer)=0 and note like '%exceeds the available context%' limit 1", a).fetchone(): return False   # stock cannot run this prompt at all (a model trained on 4k context: stock caps its slot at that): trying again changes nothing
                except sqlite3.Error: pass
            return not c.execute(q, a).fetchone()
        except sqlite3.Error: return False
    rows = [dict(r, _b=b) for r in rows for b in ((sb, o.build) if sb and (o.stock_first or stock_missing(r)) and not (r.get("args") or "").strip() else (o.build,))]   # a cell with forced server args (dense-split=0/1) is a test of our own settings: stock has none of them, no stock partner   # -sc: the stock cell, then ours; -nrt: the stock cell only where its comparison is missing
    if o.not_today:   # -nrt: a cell with enough ok rows (warm-up passes and loop retries do not count) from today in runs.db is not run again; a model with no cell left is not loaded or mounted
        import sqlite3
        done = {}   # campaign -> [(t/s, pp t/s) of today's ok runs of the build under test (a campaign named `latest-...` holds the runs of every build `latest` ever pointed to: only this build's count)]
        cur = own_build_no(o.build); cur = int(cur) if str(cur).isdigit() else None
        if cur is None and re.fullmatch(r"b\d+", os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, o.build)))): cur = int(os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, o.build)))[1:])   # --version failed: the directory is named by the build number
        try:
            for c, t, p, bn in sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute(
                    "select campaign, tps, pp, bno from runs where cast(ok as integer)=1 and tps is not null and date(coalesce(start_ts, ts))>=date('now','localtime','-%d days') and coalesce(env,'') not like '%%warmpass%%' and hw='%s' order by ts" % (o.not_today - 1, HW)):
                if cur and not str(c).startswith("stock") and int(bn or 0) != cur: continue
                done.setdefault(c, []).append((float(t), float(p or 0)))
        except sqlite3.Error as e: print(f"-nrt: runs.db: {e}", flush=True)
        def today(r):   # today's ok runs of this cell on this build; a campaign named after another label of the same build (um3d-..., latest-...) counts, the build number decides
            k = f"{r['_b']}-{r['name']}-{r['test']}"
            if k in done or r['_b'].startswith("stock"): return done.get(k, [])
            return next((v for c, v in done.items() if c.endswith(f"-{r['name']}-{r['test']}") and not c.startswith("stock")), [])
        keep = [r for r in rows if len(today(r)) < (1 if o.status else int(r["n"] or 1))]   # --status: a cell with any run today has its line
        for r in rows:   # the skipped cells keep their line, with the results of today's runs and the gain over stock, as if they had just run
            if r in keep: continue
            if r['_b'].startswith('stock') and any(q['name'] == r['name'] and q['test'] == r['test'] and not q['_b'].startswith('stock') for q in rows): continue   # our line shows the stock numbers
            res = today(r)[-int(r["n"] or 1):]   # the newest n
            STOCK_REC[0] = None
            try: gain = stock_gain((r["model"], r["test"]), res, r["_b"])
            except Exception as e: gain = f"  | gain: {e}"
            _rid = run_id(f"{r['_b']}-{r['name']}-{r['test']}"); _el, _lt = run_times(_rid)
            if STOCK_REC[0] and _rid:   # the label is the build that made the run, not the one `latest` points to now
                _b = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")).execute("select bno, start_ts from runs where rowid=?", (_rid,)).fetchone()
                if _b and _b[0]: STOCK_REC[0]["ob"] = int(_b[0]); STOCK_REC[0]["od"] = str(_b[1])[:10]
            CELL_REC.append(dict(skipped=True, tag=f"{r['_b']}-{r['name']}", asked=int(r["tokens"]) if r.get("tokens") else 0, ref=ours_reference(r["model"], r["test"]) if r["_b"].startswith("stock") else None, sbno=own_build_no(r["_b"]), xfer="" if r["_b"].startswith("stock") else xfer_of(own_build_no(r["_b"]), r.get("args") or ""), sdate=time.strftime("%F", time.localtime(os.path.getmtime(os.path.join(DEVBUILDS, r["_b"], "llama-server")))) if os.path.exists(os.path.join(DEVBUILDS, r["_b"], "llama-server")) else "", rid=_rid, build=r["_b"], prompt=prompt_txt(r["test"], r.get("tokens")), model=r["model"], test=r["test"], name=("s" if r["_b"].startswith("stock") else "") + r["name"], el=_el, lt=_lt, rc=0, res=res, st=STOCK_REC[0], stock_row=r["_b"].startswith("stock")))
            CELL_LINES.append(col(1, f"{('s' if r['_b'].startswith('stock') else '') + r['name']:<{NAMEW[0]}}") + "  " + "  ".join(speed_txt(t, p) for t, p in res) + gain + (f"  [{col(36, CELL_TEST[0])}]" if CELL_TEST[0] else ""))
        if len(keep) < len(rows): print(f"-nrt: {len(rows) - len(keep)} cells already run in the last {o.not_today} day(s) skipped (they are in the table at the end)", flush=True)
        rows = keep
    if o.status:   # one table in the -nrt columns: the cells done today, the one running now, the cells queued next (the rest of the list, then the queue daemon's waiting tasks); nothing else happens
        run_tag, run_since = running_cell()
        for rec in CELL_REC: rec["section"] = "running" if rec.get("tag") == run_tag else "done"
        for r in rows:
            tag = f"{r['_b']}-{r['name']}"; run_ = tag == run_tag
            ex = cell_expect(r["model"], r["test"], r["n"], r["_b"], o.warm)
            rec = dict(tag=tag, section="running" if run_ else "queued", asked=int(r["tokens"]) if r.get("tokens") else 0, ref=None, sbno="", xfer="", sdate="", rid=None, prompt=prompt_txt(r["test"], r.get("tokens")), model=r["model"], test=r["test"],
                       name=("s" if r["_b"].startswith("stock") else "") + r["name"], el=time.time() - run_since if run_ and run_since else None, est=None if run_ else ex, exp=ex if run_ else None, lt="", rc=0, res=[], st=None, stock_row=r["_b"].startswith("stock"))
            rec.update(last_run_rec(r, HW))   # the values of its last run, with the gain over stock: what it is expected to beat
            if not run_: rec["res"] = []   # a cell that has not run shows no speed: the last run of the same model and test is another cell's number
            CELL_REC.append(rec)
        for rec in CELL_REC:   # the running cell: its earlier rows (done today) and its expected time give the progress in percent
            if rec.get("section") == "running" and run_since: rec["el"] = time.time() - run_since   # elapsed since the cell process started, not the time of an earlier run of it
            if rec.get("section") == "running" and not rec.get("exp"): rec["exp"] = cell_expect(rec["model"], rec["test"], 2, rec["tag"].split("-")[0], o.warm)   # the cell may be one the -nrt skip dropped from rows: its model and test come from its record
        qd = queue_waiting()   # tasks of the queue daemon (tasks.csv): only their name is known; a stopped daemon starts none of them, they are counted, not listed
        qd_on = subprocess.run(["systemctl", "is-active", "--quiet", "queued.service"]).returncode == 0
        for t in (qd if qd_on else []):
            CELL_REC.append(dict(tag=t, section="queued", asked=0, ref=None, sbno="", xfer="", sdate="", rid=None, prompt="", model="", test="", name=t, el=None, lt="", rc=0, res=[], st=None, stock_row=False))
        print(col(1, hw_line()) + "\n" + "\n".join(final_table(CELL_REC)), flush=True)
        if qd and not qd_on: print(col(90, f"queue daemon off: {len(qd)} old tasks wait in tasks.csv (none starts)"), flush=True)
        waiting = []   # waiter shells (`while pgrep ...; do sleep; done; ... python3 run.py jobs LIST`): what starts when the current run ends, in start order
        try:
            ps_ = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout.splitlines()
            started = {int(l.split()[1]) for l in ps_[1:] if "run.py" in l}   # a waiter whose list already runs (its python child exists) is not queued any more
            for ln in (l.split(None, 1)[1] if False else l for l in ps_[1:]):
                if int(ln.split()[0]) in started: continue
                ln = ln.split(None, 1)[0] + " " + ln.split(None, 2)[2]   # pid args
                m = re.search(r"bash -c while .*?done;.*?(python3 run\.py (?:run|calib) [^'\"]*)$", ln)
                if m and "sudo" not in ln.split("bash -c")[0]: waiting.append((int(ln.split()[0]), m.group(1).strip()))
        except Exception: pass
        if waiting:
            print(col(1, "\nqueued next (start by themselves, in this order):"), flush=True)
            for k, (pid_, cmd_) in enumerate(sorted(waiting), 1):
                nm = re.search(r"run\.py (?:run|jobs) (\w+)", cmd_); cells = ""
                if nm and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", nm.group(1) + ".csv")):
                    names = [r["name"] for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", nm.group(1) + ".csv"), newline=""))]
                    cells = f"  ({len(names)} cells: {', '.join(names[:6])}{' ...' if len(names) > 6 else ''})"
                print(f"  {k}. {cmd_}{cells}", flush=True)
        return 0
    for r in rows:   # -a: the same cells with extra arguments on our build
        if o.cell_args:
            stock = r["_b"].startswith("stock")   # stock has no --moe keys: it gets only the plain flags (-a "-ub 4096 -b 4096"), a tuned stock cell, recorded in the args column
            # "--moe K=V" as on a server line is K=V: the bare --moe word left in broke the server line (-np 1 --moe -lm mmap ..., 2026-10-27)
            words = [w for w in argv_of(o.cell_args) if w != "--moe" and not (stock and re.fullmatch(r"[\w-]+=\S*", w))]
            r["args"] = ((r.get("args") or "") + " " + shlex.join("moe:" + w if not stock and re.fullmatch(r"[\w-]+=\S*", w) else w for w in words)).strip()   # a bare key=value is a --moe key (-a "split-experts=0")
    for r in rows:   # -draft N: our cells of a model with a draft file in models.ini
        d = (fcfg.models_ini() or {}).get(model_alias(r["model"]) or "", {}).get("draft") if o.draft and not r["_b"].startswith("stock") and "-md" not in shlex.split(r.get("args") or "") else None   # a row that names its own draft keeps it
        if d:
            r["args"] = ((r.get("args") or "") + f" -md {shlex.quote(os.path.expanduser(d))} --spec-type draft-dspark --spec-draft-n-max {o.draft} --moe spec-depth={o.draft}").strip()
            r["name"] += f"-draft{o.draft}"
    if o.machine != "pc1": return run_list_remote(o, rows, name)
    def size(r):   # bytes of the model; a split model counts all its parts (NAME-00001-of-0000N.gguf)
        return sum(os.path.getsize(f) for f in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", r["model"]))) if os.path.exists(r["model"]) else 0
    if o.desc or "desc" in argv or "-dsc" in argv:
        rows.sort(key=lambda r: (-size(r), r["model"]))   # the same model stays together: one pool, one load
    elif o.asc:
        rows.sort(key=lambda r: (size(r), r["model"]))
    if o.force_pin:   # -pin: every non-stock row not already -pin / -nm
        for r in rows:
            if not r["_b"].startswith("stock") and not {"-pin", "-nm"} & set(shlex.split(r.get("args") or "")): r["args"] = ((r.get("args") or "") + " -pin").strip()
    if o.no_pool and o.no_pool != "0":   # -nm GB: every non-stock row pins at most GB GiB of the model, the rest is mapped (row-level -nm GB, see nomount_args)
        for r in rows:
            if not r["_b"].startswith("stock") and "-nm" not in shlex.split(r.get("args") or ""): r["args"] = ((r.get("args") or "") + f" -nm {o.no_pool}").strip()
    ram_gib = int(open("/proc/meminfo").readline().split()[1]) >> 20
    for r in rows:   # a model that cannot fit RAM beside 16 GiB for the system must not be pinned (128 GiB GLM on 125 GiB froze PC1): -lm mmap, no pool; stock mmaps already
        if not r["_b"].startswith("stock") and (size(r) >> 30) + 16 > ram_gib and not nomount_args(r.get("args") or "")[1]:
            pg = max(1, ram_gib - 24)   # default pin cap: RAM - 24 GiB (125 GiB -> 101); pinning 110 GiB froze PC1; the rest is mapped
            r["args"] = ((r.get("args") or "") + f" -nm {pg}").strip()   # (model > RAM - 16 GiB: pin at most RAM - 24 GiB, map the rest; `run ... -n` shows it as -nm in the cell's arguments)
    bad = []; last_model = shown = None
    for b in dict.fromkeys(r["_b"] for r in rows) or [o.build]:   # which binaries the list runs: name, real path, build date
        d = os.path.realpath(os.path.join(DEVBUILDS, b)); bin_ = os.path.join(d, "llama-server")
        print(f"{col(36, b)}: {bin_}  {col(33, time.strftime('%F %H:%M', time.localtime(os.path.getmtime(bin_))))}  {col(94, 'b' + own_build_no(b))}" if os.path.exists(bin_) else f"{b}: {d} (no llama-server)", flush=True)
    if any(not r["_b"].startswith("stock") for r in rows) and os.path.islink(os.path.join(DEVBUILDS, "stock-latest")):   # the stock the gains compare against
        sl = os.path.realpath(os.path.join(DEVBUILDS, "stock-latest")); print(f"{col(36, 'stock')}: {os.path.basename(sl)}  {col(33, time.strftime('%F %H:%M', time.localtime(os.path.getmtime(os.path.join(sl, 'llama-server')))))}  {col(94, 'b' + own_build_no(os.path.basename(sl)))}", flush=True)
    if not o.dry: CUR_HWL[0] = hw_line(); print(col(1, CUR_HWL[0]), flush=True)
    NAMEW[0] = max((len(("s" if r["_b"].startswith("stock") else "") + r["name"]) for r in rows), default=NAMEW[0] - 1) + 1
    exp_all = [cell_expect(r["model"], r["test"], r["n"], r["_b"], o.warm) for r in rows]
    known = [x for x in exp_all if x]; fill = (sum(known) / len(known)) if known else 600.0   # a cell without history counts the average of the known ones, so the whole-run bar and its time cover every cell of the list
    total_s = sum(x or fill for x in exp_all); done_s = 0; run_t0 = time.time()   # expected seconds of the whole list, from runs.db
    try:
        for r in rows:
            key = (r["model"], r["_b"].startswith("stock"))   # the official stock binary knows no pool: it runs without one (the pool's RAM is its page cache), ours size it per model
            if key != last_model and not o.no_pool and not o.dry and not nomount_args((r.get("args") or "") + " " + (r.get("extra") or ""))[1]:
                last_model = key; live_clear(); kept_kill()
                if key[1]:
                    if os.path.ismount("/mnt/huge1g"):
                        try:
                            import contextlib, io
                            with contextlib.redirect_stdout(io.StringIO()): cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
                        except SystemExit as e: print(f"pool: unmount failed: {e}", flush=True)
                else: run_list_pool(size(r))
            if r["model"] != shown and not o.dry: shown = r["model"]; live_clear(); print(col("1;94", first_shard(shown)), flush=True)   # after the pool line: which model the next cells run
            tag = f"{r['_b']}-{r['name']}"
            cmd = [sys.executable, os.path.abspath(__file__), "cell", tag, r["model"], r["_b"], r["n"] or "1", r["test"], (" " + r["args"]) if r.get("args") else ""] + (["--warm"] if o.warm else []) + ([] if (o.kill or o.clean_state) else ["--keep-server"]) + (["--clean-state"] if o.clean_state else [])
            cmd += (["--tokens", r["tokens"]] if r.get("tokens") else []) + shlex.split(r.get("extra") or "")
            if o.dry: print(f"[{name}] {tag}: {re.sub(r' pf(?=\d+k)', ' ', shlex.join(cmd[3:]))}", flush=True)
            ex = cell_expect(r["model"], r["test"], r["n"], r["_b"], o.warm)
            tin = TEST_IN.get(r["test"]); tout = int(r["tokens"]) if r.get("tokens") else 0
            CELL_BUILD[0] = str(r.get("_b") or ""); CELL_REF[0] = (os.path.basename(r["model"]), r["test"]); CELL_PROMPT[0] = prompt_txt(r["test"], r.get("tokens")); CELL_ASKED[0] = int(r["tokens"]) if r.get("tokens") else 0
            CELL_TEST[0] = (re.sub(r"^(\d+k)/(\w+)$", r"\1\2", CELL_PROMPT[0]) if re.fullmatch(r"pf\d+k", r["test"]) else CELL_PROMPT[0]) if re.fullmatch(r"pf\d+k(1k)?", r["test"]) else r["test"] + "  " + (f"{tin}{'t' if tin[-1].isdigit() else ''} in" if tin and tin[0].isdigit() else tin or "") + (f" {tout // 1024}k out" if tout and tout % 1024 == 0 else f" {tout} out" if tout else "")   # "12k in 1k out", "64t in 1k out"
            CELL_TEST[0] += f"  {r['_b']}" + (f"  {r['args'].strip()}" if r.get("args") and r["args"].strip() else "")   # what runs: build, extra server args
            if not o.dry and run_watched(cmd, dict(os.environ, PERF_BUILDS=DEVBUILDS), ex, ("s" if r["_b"].startswith("stock") else "") + r["name"], total=(done_s, total_s, run_t0, rows.index(r) + 1, len(rows)), log=f"{QDIR}/{tag}.list.log", results=f"{QDIR}/{tag}-{r['test']}.run.out", peek=(r["model"], r["test"])): bad.append(tag)
            if not o.dry and not r["_b"].startswith("stock") and not (r.get("args") or "").strip(): learn_bestpp(r["model"], r["test"], tag)
            if not o.dry and not r["_b"].startswith("stock"): learn_att(r["model"], tag)
            done_s += ex or fill
    finally:   # the cells' lines, then the pool: it holds its RAM whether a model uses it or not: give it back when the list ends, also on Ctrl-C or an error
        if CELL_LINES and not o.dry:   # the final block, complete for a screenshot: the machine, then every cell with what it measured
            live_clear(); print("\n" + (col(1, hw_line()) + "\n" if any(not x.get("skipped") for x in CELL_REC) else "") + "\n".join(final_table(CELL_REC) or CELL_LINES) + ("\n" + col(90, "repro  " + os.environ["RUNPY_REPRO"]) if os.environ.get("RUNPY_REPRO") else ""), flush=True)   # the command line under the table: copy it to repeat the run; the machine line is at the top already when no cell ran
        if CELL_LINES and not o.dry: open(f"{QDIR}/{o.build}-{name}.list.txt", "a").write("\n".join(CELL_LINES) + "\n")   # the screen shows only the current cell; the finished cells' lines go here
        if not o.dry: kept_kill()   # the kept server holds the pool: it goes before the unmount
        if not o.no_pool and not o.dry and os.path.ismount("/mnt/huge1g"):
            try:
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()): cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
                print("pool ->0g", flush=True)
            except SystemExit as e: print(f"pool: unmount failed: {e}", flush=True)
    print(f"[{name}] {len(rows)} cells, failed: {', '.join(bad) or 'none'}")
    return 1 if bad else 0


PS_INSTR = "Return the completed source with the bugs fixed."   # run -ps: the instruction in front of the cut source
PF_FILES = ()   # sizes (k) with a frozen prompt file prompts/src_Nk.cpp and a test alias pfNk1k in prompts.ini


def n_cards():
    """how many cards the cells see: -ng N (RUN_NG), else CUDA_VISIBLE_DEVICES, else the cards nvidia-smi lists"""
    if os.environ.get("RUN_NG", "") != "": return int(os.environ["RUN_NG"])
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv is not None: return len([x for x in cv.split(",") if x.strip()])
    try: return len([l for l in subprocess.run([os.environ.get("NVIDIA_SMI", "nvidia-smi"), "-L"], capture_output=True, text=True, timeout=30).stdout.splitlines() if l.startswith("GPU ")])
    except (OSError, subprocess.SubprocessError): return 0


def hw_key():
    """the VRAM of the cards the cells use, in GiB, for models.ini: 24g for one 24 GiB card, 48g for two (the best prompt depends on the VRAM, so a model holds one per setup:
    bestpp = [48=8,24=7]). -ng N takes the N biggest cards. No card: 0g"""
    try: mem = sorted((round(int(x) / 1024) for x in subprocess.run([os.environ.get("NVIDIA_SMI", "nvidia-smi"), "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.split()), reverse=True)
    except (OSError, ValueError, subprocess.SubprocessError): mem = []
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv is not None and not os.environ.get("RUN_NG"): mem = mem[:len([x for x in cv.split(",") if x.strip()])]
    if os.environ.get("RUN_NG", "") != "": mem = mem[:int(os.environ["RUN_NG"])]
    return f"{sum(mem)}g"


def model_alias(model_path):
    """the models.ini section of a model file (by file name), None when it has none"""
    base = os.path.basename(model_path or "")
    return next((k for k, v in fcfg.models_ini().items() if v.get("path") and os.path.basename(v["path"]) == base), None)


def bestpp_entries(text):
    """models.ini value '[(48,8018,1024,733,18.9,10.9,54.2,1.89),...]' -> {48: (8018, 1024, 733.0, 18.9, 10.9, 54.2, 1.89)}: total VRAM of the cards in GiB -> (prompt tokens, answer tokens, prompt t/s,
    decode t/s, prompt seconds, answer seconds, score); the request takes prompt + answer seconds"""
    num = r"\s*,\s*([\d.]+)"
    return {int(g[0]): (int(g[1]), int(g[2])) + tuple(float(x) for x in g[3:]) for g in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)" + num * 5 + r"\s*\)", str(text))}


def bestpp_run(t):
    """the bestpp tuple of one request from the server timings (no load time in them): (total VRAM GiB, prompt tokens, answer tokens, prompt t/s, decode t/s, prompt s, answer s); stored with
    every run in runs.db, whether the run is better or worse than the models.ini entry"""
    try:
        ans = (t.get("predicted_ms") or t["predicted_n"] / max(t["predicted_per_second"], 1e-9) * 1e3) / 1e3
        return f"({hw_key()[:-1]},{t['prompt_n']},{t['predicted_n']},{t['prompt_per_second']:.0f},{t['predicted_per_second']:.1f},{t['prompt_ms'] / 1e3:.1f},{ans:.1f})"
    except (KeyError, TypeError, ZeroDivisionError): return ""


def ini_set_bestpp(sec, tok, out, pp, tps, pp_s, tg_s, score):
    """models.ini [sec]: the line bestpp = [(VRAM GiB, prompt tokens, answer tokens, prompt t/s, decode t/s, prompt s, answer s, score), ...] with the entry of this VRAM (hw_key) set; other VRAM entries stay"""
    ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.ini"); lines = open(ini).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip() == f"[{sec}]"); i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    while i1 > i0 + 1 and not lines[i1 - 1].strip(): i1 -= 1
    j = next((i for i in range(i0 + 1, i1) if lines[i].split("=")[0].strip() == "bestpp"), None)
    e = bestpp_entries(lines[j].split("=", 1)[1]) if j is not None else {}; e[int(hw_key()[:-1])] = (tok, out, pp, tps, pp_s, tg_s, score)
    new = "bestpp = [" + ",".join(f"({g},{t},{o},{p:.0f},{d:.1f},{ps:.1f},{ts:.1f},{sc:.2f})" for g, (t, o, p, d, ps, ts, sc) in sorted(e.items(), reverse=True)) + "]"
    if j is not None: lines[j] = new
    else: lines.insert(i1, new)
    open(ini, "w").write("\n".join(lines))


def ini_set_maxpp(sec, tokens):
    """models.ini [sec]: the line maxpp = [(VRAM GiB, tokens), ...] with the entry of this VRAM (hw_key) set: the biggest pass the engine says fits ("this call fits the ceiling N" in the log of a probe), the start of the search for the best prompt"""
    ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.ini"); lines = open(ini).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip() == f"[{sec}]"); i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    while i1 > i0 + 1 and not lines[i1 - 1].strip(): i1 -= 1
    j = next((i for i in range(i0 + 1, i1) if lines[i].split("=")[0].strip() == "maxpp"), None)
    e = {int(g): int(t) for g, t in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", lines[j].split("=", 1)[1])} if j is not None else {}; e[int(hw_key()[:-1])] = int(tokens)
    new = "maxpp = [" + ",".join(f"({g},{t})" for g, t in sorted(e.items(), reverse=True)) + "]"
    if j is not None: lines[j] = new
    else: lines.insert(i1, new)
    open(ini, "w").write("\n".join(lines))


BESTPP_MEM = {}   # (alias, VRAM GiB) -> request time per token (ms) of the best prompt so far: read from models.ini once, then only in memory


def scores(seen):
    """{prompt tokens: (answer tokens, prompt t/s, decode t/s)} -> {prompt tokens: score}: prompt speed over the best prompt speed plus decode speed over the best decode speed (best on both = 2)"""
    mp, mt = max(v[1] for v in seen.values()), max(v[2] for v in seen.values())
    return {k: v[1] / mp + v[2] / mt for k, v in seen.items()}


def ini_set_key(sec, key, value):
    """models.ini [sec]: the line `key = value` set (added at the end of the section when missing); other lines untouched"""
    ini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.ini"); lines = open(ini).read().split("\n")
    i0 = next((i for i, l in enumerate(lines) if l.strip() == f"[{sec}]"), None)
    if i0 is None: return False
    i1 = next((i for i in range(i0 + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    while i1 > i0 + 1 and not lines[i1 - 1].strip(): i1 -= 1
    j = next((i for i in range(i0 + 1, i1) if lines[i].split("=")[0].strip() == key), None)
    new = f"{key} = {value}"
    if j is not None and lines[j].strip() == new: return False
    if j is not None: lines[j] = new
    else: lines.insert(i1, new)
    open(ini, "w").write("\n".join(lines)); return True


def learn_att(model, tag):
    """after a cell: the engine's line "attention layers: N full (grow with the context), M sliding-window (window W), K linear (no context term), sparse top-k S" (read from
    the GGUF at load) stored once in models.ini as `att = full N, swa M/W, linear K, topk S`: which layers make the prompt cost grow with the context (user 2026-10-25)"""
    sec = model_alias(model); lg = f"{QDIR}/{tag}.server.log"
    if not sec or not os.path.exists(lg): return
    m = re.search(r"attention layers: (\d+) full \(grow with the context\), (\d+) sliding-window \(window (\d+)\), (\d+) linear \(no context term\), sparse top-k (\d+)", open(lg, errors="replace").read())
    if m and ini_set_key(sec, "att", f"full {m[1]}, swa {m[2]}/{m[3]}, linear {m[4]}, topk {m[5]}"):
        print(col(92, f"{sec}: attention layers full {m[1]}, sliding-window {m[2]} (window {m[3]}), linear {m[4]}, sparse top-k {m[5]}, written to models.ini"), flush=True)


def learn_bestpp(model, test, tag):
    """after a finished cell of a sized prompt (7k1k), at any number of passes (one pass is not always the best): its last (warm) run is compared with the model's best prompt for this VRAM (kept in memory, first read from models.ini
    bestpp). Both prompt speed and decode speed count: each is scaled by the better of the two, the two scaled numbers are added, the higher sum wins, a longer prompt wins a tie. The winner is
    the entry, in memory and in models.ini; nothing is written when the stored one wins. The best prompt is learned from the runs that happen anyway, no extra run. (Time per token was tried and
    dropped: a longer prompt spreads the answer over more tokens and always wins, whatever its speed.)"""
    m = re.fullmatch(r"pf(\d+)k1k", test or ""); sec = model_alias(model)
    if not m or not sec: return
    try:
        import sqlite3
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        x = c.execute("select pp, tps, note, pf_passes, n_gen from runs where campaign like ? and cast(ok as integer)=1 and pp is not null and tps is not null order by ts desc limit 1", (f"{tag}-%",)).fetchone()
        pn = re.search(r"prompt_n=(\d+)", (x[2] if x else "") or "")
        if not x or not pn: return   # any pass count counts: one pass is not always the best
        new = (int(pn.group(1)), int(x[4] or 1024), float(x[0]), float(x[1])); hw = int(hw_key()[:-1]); key = (sec, hw)   # prompt tokens, answer tokens of this run, prompt t/s, decode t/s
        pp_s, tg_s = new[0] / new[2], new[1] / new[3]; total = pp_s + tg_s   # seconds of the request: the prompt, then the answer the run gave at the decode speed that followed (the server's own timings: no load time)
        if key not in BESTPP_MEM:   # every one-pass prompt seen so far for this model and VRAM: {prompt tokens: (answer tokens, prompt t/s, decode t/s)}, starting with the models.ini entry
            e0 = bestpp_entries((fcfg.models_ini().get(sec) or {}).get("bestpp", "")).get(hw); BESTPP_MEM[key] = {e0[0]: e0[1:4]} if e0 else {}
        seen = BESTPP_MEM[key]; was = max(seen, key=lambda k: scores(seen)[k]) if seen else None   # the stored best before this run
        seen[new[0]] = new[1:]
        sc_all = scores(seen); best = max(seen, key=lambda k: (round(sc_all[k], 6), k))   # all seen sizes against the same two best numbers: no order effects
        if best != was or new[0] == best:
            o_, p_, d_ = seen[best]; ini_set_bestpp(sec, best, o_, p_, d_, best / p_, o_ / d_, sc_all[best])
            if best != was: print(col(92, f"best prompt of {sec} at {hw}g: {best} tokens, prompt {p_:.0f} t/s, decode {d_:.1f} t/s, total {best / p_ + o_ / d_:.1f} s, score {sc_all[best]:.2f}, written to models.ini"), flush=True)
    except Exception as e: print(col(90, f"best prompt not updated: {e}"), flush=True)


def vram_known(model_path):
    """True when the model's bestpp has an entry for the VRAM of the cards in use"""
    sec = (fcfg.models_ini() or {}).get(model_alias(model_path) or "", {})
    return int(hw_key()[:-1]) in bestpp_entries(sec.get("bestpp", ""))


def prompt_known(model_path):
    """True when models.ini holds a best prompt or a pass ceiling (maxpp) for the VRAM of the cards in use: best_prompt_k can answer without asking the engine"""
    sec = (fcfg.models_ini() or {}).get(model_alias(model_path) or "", {}); g = int(hw_key()[:-1])
    return g in bestpp_entries(sec.get("bestpp", "")) or g in {int(x) for x, _ in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", sec.get("maxpp", ""))}


def best_prompt_k(model_path):
    """the model's best input prompt in nominal thousands of tokens: models.ini `bestpp = [(48,8018,1024,733,18.9,10.9,54.2,1.89),(24,7248,1024,737,21.7,9.6,47.2,2.00)]` (total VRAM in GiB, prompt tokens, answer tokens, prompt t/s, decode t/s, prompt s, answer s, score) for the cards seen now,
    tokens rounded to thousands (7248 -> 7k); no entry for this VRAM: the maxpp of this VRAM (the biggest pass the engine says fits); neither stored: run.py stops and says to ask the engine (calib --ceil). Filled by learn_bestpp from the runs that happen anyway."""
    sec = (fcfg.models_ini() or {}).get(model_alias(model_path) or "", {})
    t = bestpp_entries(sec.get("bestpp", "")).get(int(hw_key()[:-1]))
    if t: return max(1, round(t[0] / 1000))
    mx = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", sec.get("maxpp", ""))
    mx = {int(g): int(x) for g, x in mx}.get(int(hw_key()[:-1]))
    if mx: return max(1, mx // 1000)   # no best yet: the biggest pass that fits (maxpp)
    sys.exit(f"{model_alias(model_path)}: no best prompt and no pass ceiling stored for {hw_key()} of VRAM; ask the engine once: python3 fork/run.py calib --ceil -m {model_alias(model_path)}")   # no guessed size


def best_test(model_path, test, remote=False):
    """pfbest1k -> pf<N>k1k with N = the model's best one-pass prompt (pfbest -> pf<N>k); any other test comes back unchanged. A remote machine runs the 12k prompt (its requests are frozen files)."""
    test = re.sub(r"^(\d+k\d*k|best\d*k)$", r"pf\1", test)   # the plain names 8k1k, best1k are the tests pf8k1k, pfbest1k
    m = re.fullmatch(r"pfbest(.*)", test)
    return f"pf{12 if remote else best_prompt_k(model_path)}k{m.group(1)}" if m else test


CLAMPED_SHOWN = set()
CLAMPED = {}   # cell tag -> prompt tokens it was cut to (model with a small trained context)
REMOTE_TESTS = {   # list test -> (request file in fork/bench, server arguments of ours, of stock): the README's three tests, same -c values as its table
    "pf12k1k": ("req-12k-1k.json", "-np 1", "-np 1 -c 16384"),
    "edit4c": ("req-3k-1k-a.json", "-np 1 -c 4096", "-c 4096"),
    "tetrisc": ("req-short-1k.json", "-np 1 -c 1024", "-c 1024")}


def remote_test(name):
    """(request file, server arguments of ours, of stock) of a list test on another machine: the README's three tests (REMOTE_TESTS), or a test of your own named after a file in the prompts directory:
    prompt-NAME.txt (one chat message, 1024 tokens out) or req-NAME.json (the request as it is); the request of a prompt file is written next to the run files. None for an unknown test."""
    if name in REMOTE_TESTS: return REMOTE_TESTS[name]
    ut = config.user_tests().get(name)
    if ut and ut.get("test") == "chat":
        text = config.prompt(name); toks = int(ut.get("tokens") or 1024)
        fp = os.path.join(QDIR, f"req-{name}.json")
        json.dump({"messages": [{"role": "user", "content": text}], "max_tokens": toks, "temperature": 0, "top_p": 1, "top_k": 1, "cache_prompt": False}, open(fp, "w"))
        ctx = int(ut.get("ctx") or (len(text) // 3 + toks + 512 + 1023) // 1024 * 1024)   # characters / 3 over-counts tokens a little: room, not a measurement
        return fp, f"-np 1 -c {ctx}", f"-c {ctx}"
    rq = os.path.join(PROMPT_DIR, f"req-{name}.json")
    if os.path.exists(rq):
        d = json.load(open(rq)); ctx = (sum(len(m.get("content", "")) for m in d.get("messages", [])) // 3 + int(d.get("max_tokens") or 1024) + 512 + 1023) // 1024 * 1024
        return rq, f"-np 1 -c {ctx}", f"-c {ctx}"
    return None


def run_list_remote(o, rows, name):
    """`run -c NAME -pc ID`: the list's cells (already filtered and ordered: -onl, -skp, -sc) on another machine over ssh, run.py stays on this host. Per row: the model is looked
    up on that machine by file name, then `run.py remote ID test` runs the server and the request there (stock row first with -sc) and records the rows with hw = ID."""
    m = machine(o.machine)
    here = PROMPT_DIR   # the request files (REMOTE_TESTS)
    bad = []
    for b in dict.fromkeys(r["_b"] for r in rows):   # the same header as a local list: the binaries, the stock the gains compare against, the machine
        bin_ = os.path.join(os.path.realpath(os.path.join(DEVBUILDS, b)), "llama-server")
        if os.path.exists(bin_): print(f"{col(36, b)}: {bin_}  {col(33, time.strftime('%F %H:%M', time.localtime(os.path.getmtime(bin_))))}  {col(94, 'b' + own_build_no(b))}", flush=True)
    HOST[0] = m
    if o.power and not o.dry:   # -pl W,W,..: the limits of the remote GPUs for this run (the driver refuses a value outside the card's range, which is printed)
        qq = [l.split(",") for l in hsh("nvidia-smi --query-gpu=pcie.link.width.max,power.max_limit --format=csv,noheader,nounits", 30).splitlines() if l.count(",") == 1]
        wd = [int(a) for a, _ in qq]
        for i, w in enumerate(power_targets(wd, o.power, [int(float(b)) for _, b in qq])):
            r = hsh(f"nvidia-smi -i {i} -pl {w} 2>&1 | head -1", 60).strip()   # the first line is the result or the reason ("Insufficient Permissions"); the last line is always "Terminating early"
            print(f"power: {o.machine} GPU {i} -> {w} W: {r[:110]}", flush=True)
    if not o.dry: CUR_HWL[0] = hw_line(); print(col(1, CUR_HWL[0]), flush=True)
    NAMEW[0] = max((len(("s" if r["_b"].startswith("stock") else "") + r["name"]) for r in rows), default=NAMEW[0] - 1) + 1
    try: _run_list_remote(o, rows, name, m, here, bad)
    finally:
        pass   # the final block, complete for a screenshot, like a local list
        if CELL_REC and not o.dry: print("\n" + col(1, hw_line()) + "\n" + "\n".join(final_table(CELL_REC)), flush=True)
    print(f"[{name}] {len(rows)} cells on {o.machine}, failed: {', '.join(bad) or 'none'}")
    return 1 if bad else 0


def _run_list_remote(o, rows, name, m, here, bad):
    """the cells of the list: `run.py rcell` (server and requests on the machine, files mirrored here) under run_watched, so the screen, the table and the stored rows are the local ones"""
    REMOTE_SLOTS[0] = f"{QDIR}/remote-slots.txt"
    try:
        if not o.dry:   # the machine must answer before anything else: a rented box that is down, restarting or at a new address ends the list here with the reason, not with a traceback
            try: ok_ssh = subprocess.run(["ssh", "-o", "ConnectTimeout=12", "-o", "BatchMode=yes", m["ssh"], "true"], capture_output=True, timeout=40).returncode == 0
            except subprocess.SubprocessError: ok_ssh = False
            if not ok_ssh:
                hn = subprocess.run(["ssh", "-G", m["ssh"]], capture_output=True, text=True).stdout
                ad = " ".join(f"{k}={v}" for k, v in re.findall(r"^(hostname|port) (\S+)", hn, re.M))
                sys.exit(f"{o.machine} does not answer over ssh ({ad}): the box is down or restarting, or its address changed. New line from the provider: add  -ssh \"-p PORT root@HOST\"  to this command (or `run.py machine add`).")
        if not o.dry and not m["win"]:   # first action on a machine: anything holding the GPU goes (vast.ai's own `llama` service restarts itself, so supervisor first)
            remote_sh(m, 'command -v supervisorctl >/dev/null && supervisorctl stop llama 2>&1; pkill -x llama-server; sleep 2; echo "gpu used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr "\\n" " ")"', 60)
        if not o.dry:   # every build and model of the list at once, before the first cell: downloads run on the machine in parallel, the push from here beside them
            items, pushes, keep = [], [], set()
            for b in dict.fromkeys(os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, r["_b"]))) for r in rows):
                it, pu = build_plan(m, b); items += it; pushes += [pu] if pu else []
            for mod in dict.fromkeys(r["model"] for r in rows):
                keep.update(shard_names(os.path.basename(mod)))
                it, _ = model_plan(m, mod); items += it
            tr_ok = run_transfers(m, items, pushes, keep)
            if tr_ok and live_tty() and PRELUDE[0] + PRELUDE[1]: print(f"\033[{PRELUDE[0] + PRELUDE[1]}A\033[J", end="", flush=True)   # the downloads are done: the start lines and the bars go (free disk stays in the stats line)
            PRELUDE[0] = PRELUDE[1] = 0
        shown = None
        for r in rows:
            key = f"{r['_b']}-{r['name']}-{r['test']}"   # the campaign of the stored rows, what -nrt looks for (before the build name is resolved)
            r["_b"] = os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, r["_b"])))   # `latest` / `stock-latest`: the directory that was pushed
            tag = f"{r['_b']}-{r['name']}"
            rt = remote_test(r["test"])
            if not rt: print(f"[{name}] {tag}: test {r['test']} has no remote request (the README tests {', '.join(REMOTE_TESTS)}, or a file prompt-NAME.txt / req-NAME.json in fork/prompts)", flush=True); bad.append(tag); continue
            req, ours, stock = rt
            base = os.path.basename(r["model"])
            found = remote_sh(m, f"find /m /home/o/m {m['root']}/models /home/o /root -maxdepth 5 -name '{base}' 2>/dev/null | head -1 2>/dev/null", 120).strip().splitlines()
            found = next((l for l in found if l.startswith("/")), "")   # a login banner (stderr is merged into the result) is not a path
            if not found and not o.dry: live_clear(); found = push_model(m, r["model"])   # not on the machine yet: download it there (models.ini url) or rsync it
            if not found: print(f"[{name}] {tag}: {base} not found on {o.machine}", flush=True); bad.append(tag); continue
            words, env = split_variant(r["_b"], r.get("args") or "") if not r["_b"].startswith("stock") else ("", {})
            args = f"{stock if r['_b'].startswith('stock') else ours} {nomount_args(words)[0]}".strip()
            ctx_train = gguf_ctx_train(r["model"]) if os.path.exists(r["model"]) else 0
            if not ctx_train and not o.dry:   # the file is on the machine only: its header (metadata and vocabulary come before the tensors) is read over ssh
                hd = subprocess.run(["ssh", m["ssh"], f"head -c 33554432 '{found}'"], capture_output=True).stdout
                ctx_train = gguf_ctx_train(None, hd)
            mx = clamp_tokens(r["model"], r.get("tokens") or 1024, 12000 if r["test"] == "pf12k1k" else 3000 if r["test"] == "edit4c" else 0, ctx_train)   # the README prompts are about 12k / 3k tokens
            if mx:   # the model holds less than the prompt: both builds get the same shorter prompt, cut in memory by cmd_remote (the table shows its size)
                args = re.sub(r"\s*-c \d+", "", args) + f" -c {ctx_train}"   # -c is a ceiling (ours sizes the context itself below it, stock takes it as is): both stop at the model's trained context
                CLAMPED[tag] = mx
                if (base, mx) not in CLAMPED_SHOWN:   # once per model and size, not per cell
                    CLAMPED_SHOWN.add((base, mx)); print(f"[{name}] {base} holds {ctx_train} tokens of context: the prompt is cut to about {mx} tokens", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "rcell", tag, found, r["_b"], str(r["n"] or 2), r["test"], args, "--machine", o.machine, "--req", os.path.join(here, req), "--clamp", str(mx or 0),
                   "--results", f"{QDIR}/{tag}-{r['test']}.run.out", "--slots", REMOTE_SLOTS[0], "--campaign", key]
            if not o.dry: live_clear(); push_build(m, r["_b"])   # a build the machine does not have yet goes there first (once)
            if o.dry: print(f"[{name}] {tag} on {o.machine}: {found} {r['test']} -n {r['n'] or 2} args: {args}", flush=True); continue
            if r["model"] != shown: shown = r["model"]; live_clear(); print(col("1;94", first_shard(shown)), flush=True)
            try: os.remove(f"{QDIR}/{tag}-{r['test']}.run.out")
            except OSError: pass
            tin = TEST_IN.get(r["test"]); tout = int(r["tokens"]) if r.get("tokens") else 0
            CELL_BUILD[0] = str(r.get("_b") or ""); CELL_REF[0] = (os.path.basename(r["model"]), r["test"])
            CELL_PROMPT[0] = prompt_txt(r["test"], r.get("tokens")) if tag not in CLAMPED else f"{CLAMPED[tag] / 1000:.1f}k/" + prompt_txt(r["test"], r.get("tokens")).split("/")[-1]; CELL_ASKED[0] = tout
            CELL_TEST[0] = (re.sub(r"^(\d+k)/(\w+)$", r"\1\2", CELL_PROMPT[0]) if re.fullmatch(r"pf\d+k", r["test"]) else CELL_PROMPT[0]) if re.fullmatch(r"pf\d+k(1k)?", r["test"]) else r["test"] + "  " + (f"{tin}{'t' if tin[-1].isdigit() else ''} in" if tin and tin[0].isdigit() else tin or "") + (f" {tout // 1024}k out" if tout and tout % 1024 == 0 else f" {tout} out" if tout else "")
            CELL_TEST[0] += f"  {r['_b']}" + (f"  {r['args'].strip()}" if r.get("args") and r["args"].strip() else "")   # what runs: build, extra server args
            try:
                if run_watched(cmd, dict(os.environ, PERF_BUILDS=DEVBUILDS), cell_expect(r["model"], r["test"], r["n"], r["_b"], False), ("s" if r["_b"].startswith("stock") else "") + r["name"],
                               log=f"{QDIR}/{tag}.list.log", results=f"{QDIR}/{tag}-{r['test']}.run.out", peek=(r["model"], r["test"])): bad.append(tag)
            except KeyboardInterrupt: bad.append(tag); break
    finally: REMOTE_SLOTS[0] = None


def cmd_rcell(a):
    """`rcell TAG MODEL BUILD N TEST ARGS --machine ID ...`: one list cell on another machine, the remote counterpart of `cell` (run_list_remote starts it under run_watched)"""
    for k in range(4):   # a looping answer is stored as a failed run and repeated with k newlines added to the prompt (as for a local cell)
        try: cmd_remote(argparse.Namespace(id=a.machine, what="test", builds=[a.build], model=a.model, req=a.req, clamp=int(a.clamp or 0), tweak=k, path="/v1/chat/completions", args=a.args, n=int(a.n), stats=False, record=True,
                                  tail="run", campaign=a.campaign, test=ALIASES.get(a.test, a.test), results=a.results, slots=a.slots))
        except SystemExit as e:
            if e.code != 3 or k == 3: raise   # 3 = loop: repeat; any other code ends the cell
            print(f"loop: repeating ({k + 1}/3) with {k + 1} newline(s) added to the prompt", flush=True)


def nomount_args(args, model=None):
    """(args, nomount): `-nm` in a cell's args is a run.py switch for `-lm mmap` (weights memory-mapped, not pinned); a cell that does not pin has no use for the huge-page pool,
    so no pool is mounted, resized or checked for it. An explicit `-lm`/`--load-mode` of mmap, auto or dio counts as -nm too."""
    tok = shlex.split(args or "")
    no = "-nm" in tok
    out = []
    gb = 0.0
    for i, t in enumerate(tok):
        if t == "-pin": out += ["-lm", "pin"]   # force the weights into pinned RAM (the default `auto` maps them, or pins when the model fits); the pool is wanted, so this is not a no-mount cell
        elif t == "-nm":
            out += ["-lm", "mmap"]
            try: gb = float(tok[i + 1])   # `-nm 101`: pin at most 101 GB of the model (tensor order), map the rest (--moe pin-budget, read by the loader with mmap on)
            except (IndexError, ValueError): pass
        elif gb and i and tok[i - 1] == "-nm": continue   # the GB value itself
        else: out.append(t)
    if gb:
        nb = int(gb * 2**30)
        j = out.index("--moe") + 1 if "--moe" in out else 0
        if j: out[j] += f",pin-budget={nb}"
        else: out += ["--moe", f"pin-budget={nb}"]
    mode = next((out[i + 1] for i, t in enumerate(out[:-1]) if t in ("-lm", "--load-mode")), None)
    return shlex.join(out), no or mode in ("mmap", "auto", "dio")


def pool_sanity(mnt="/mnt/huge1g", model=None):
    """Before a load (no server runs: the GPUs are free): delete what a crashed server left in the huge-page pool. An unfinished fill (.w.part) or an empty file whose
    lock nobody holds is stale and would eat pool pages the next model needs. Then compare the pool's used pages with the files' pages: more used than the files
    hold means pages leaked into an orphaned mapping (only a reboot or unmount gives them back): said loudly instead of a later PIN_STALL."""
    import fcntl
    if not os.path.ismount(mnt) or not os.path.exists(HUGE_NR): return
    held = 0
    for f in sorted(os.listdir(mnt)):
        p = os.path.join(mnt, f)
        try:
            with open(p, "rb") as fh:
                try: fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # a live server holds a lock on its file (ggml-cuda.cu)
                except OSError: held += os.path.getsize(p); continue
                if f.endswith(".w.part") or os.path.getsize(p) == 0:
                    os.remove(p); print(f"pool_sanity: removed stale {f}", flush=True)
                else: held += os.path.getsize(p)
        except OSError as e:
            print(f"pool_sanity: {f}: {e}", flush=True)
    if model and os.path.exists(model):   # a pool smaller than the model: the rest would be pinned from fragmented RAM and stall (PIN_STALL), without a pool it loads, slower
        need = -(-sum(os.path.getsize(p) for p in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model))) >> 30)
        have = int(open(HUGE_NR).read())
        if have < need:
            print(f"pool_sanity: pool {have} GiB < model {need} GiB, unmounting it (the model pins from ordinary RAM)", flush=True)
            try: cmd_hugepool(argparse.Namespace(action="unmount", what=[]))
            except SystemExit as e: print(f"pool_sanity: unmount failed: {e}", flush=True)
            return
    used = int(open(HUGE_NR).read()) - int(open(HUGE_FREE).read())
    if used > -(-held >> 30):
        print(f"pool_sanity: WARNING pool has {used} GiB pages in use but the cache files hold {-(-held >> 30)}: leaked pages, `hugepool unmount` + mount (or reboot) frees them", flush=True)


def cmd_cell(a):
    """One queue cell (what m111.sh / m111log.sh did, for one test): wait until both GPUs are free, run `bench` for the builds side by side with the campaign TAG-TEST,
    keep the run output in q5h/TAG-TEST.run.out, add the result lines to q5h/TAG.out, log a failed cell (no result line) to the shared error log and exit 1 (the queue marks
    the task failed), keep the server log as TAG.server.log when one build ran."""
    Q5 = QDIR; U = os.environ.get("PERF_LD_EXTRA", ""); SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")
    while not kept_get():   # nothing may still hold GPU memory (a killed script leaves its children running); the kept server is meant to
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if r.stdout.split() and max(int(x) for x in r.stdout.split()) < 600: break   # strict: a stray GPU process is never mistaken for a hold (one GPU process at a time)
        time.sleep(5)
    keep_pool = os.environ.get("RUN_KEEP_POOL") == "1"   # run -nm: the pool stays mounted for the whole list (a remount after fragmentation gets only part of it back: 49-58 of 108 pages, 2026-10-22)
    too_big = False
    if keep_pool and os.path.ismount("/mnt/huge1g") and os.path.exists(HUGE_NR):
        need = -(-sum(os.path.getsize(p) for p in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", a.model))) >> 30)
        too_big = int(open(HUGE_NR).read()) < need   # GLM 3.0 (109.4 GiB) in the 108 pool: this cell loads without the pool, the pool is kept
    if not nomount_args(a.args)[1] and not kept_get(): pool_sanity(model=None if keep_pool else a.model)
    # the build name "latest" is the symlink the newest dev build moves (cmd_build): resolved HERE, after the wait, so a queued cell runs the newest binary of the moment, and
    # every row, log and tag carries the real build name
    a.builds = ",".join(os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, "latest"))) if b == "latest" else resolve_build(b) for b in a.builds.split(","))
    open(os.path.join(Q5, a.tag + ".build"), "w").write(a.builds + "\n")   # which build(s) this cell really ran ("latest" resolved)
    cell_args, nomount = nomount_args(a.args, a.model)
    args = "-np 1 " + cell_args
    # a pool that is mounted now (pool_sanity above mounts it when it can) is used when the model fits it: -lm pin loads the weights into its 1 GiB pages (2-3% seen,
    # user 2026-10-25). The engine's automatic mode pins too, but from ordinary RAM, and left a mounted pool unused (101 of 101 pages free, open bug). No pool, a model
    # bigger than it, a cell with its own -lm / -nm, a stock build or RUN_POOL_PIN=0: the automatic mode as before.
    if os.path.ismount("/mnt/huge1g") and os.path.exists(HUGE_NR) and not nomount and not too_big and os.environ.get("RUN_POOL_PIN") != "0" \
            and not any(b.startswith("stock") for b in a.builds.split(",")) and not re.search(r"(^|\s)(-lm|--load-mode)(\s|$)", cell_args):
        need = -(-sum(os.path.getsize(p) for p in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", a.model))) >> 30)
        if int(open(HUGE_NR).read()) >= need + 1: args += " -lm pin"
    hog = ",".join(k + "=" + str(int(float(v) * 1024)) for k, v in (("sim-vram", a.hog_gpu), ("sim-ram", a.hog_ram)) if v is not None)
    if hog:   # --moe sim-vram / sim-ram: the server takes the memory itself (common_hog), freed by the OS with the process, no helper process to leave behind
        args += " --moe " + hog
    out = os.path.join(Q5, f"{a.tag}-{a.test}.run.out")
    cmd = ["timeout", "5400", sys.executable, os.path.abspath(__file__), "bench", "--models", a.model, "--builds", a.builds, "--variant", "v=" + args, "--variants", "v",
           "-t", a.test, "-n", str(a.n), "--no-probe", "--campaign", f"{a.tag}-{a.test}"] + (["--keep-loops"] if a.keep_loops else []) + (["--warm"] if a.warm else []) + (["--keep-server"] if a.keep_server else []) + (["--clean-state"] if a.clean_state else []) + (["--wrap", a.wrap] if a.wrap else []) + (["--ctx", str(a.ctx)] if a.ctx else []) + (["--tokens", str(a.tokens)] if a.tokens else []) + (["--lead", str(a.lead)] if a.lead else []) + (["--clamp", str(a.clamp)] if a.clamp else []) + (["--prompt", a.prompt] if getattr(a, "prompt", None) else []) + (["--chat", a.chat] if getattr(a, "chat", None) else []) + (["--nothink"] if getattr(a, "nothink", None) else [])
    env = dict(os.environ, PERF_LD_EXTRA=U, NVIDIA_SMI=SMI, PERF_LOG=LOG_LETTERS)   # r unless the caller set PERF_LOG (PERF_LOG=rp run ...: the periodic cache stats reach the server's --moe log=)
    if os.path.ismount("/mnt/huge1g") and not nomount and not too_big: env["GGML_CUDA_HUGEFS"] = "/mnt/huge1g"   # no pool mounted: the binary pins from ordinary RAM
    if env.get("PERF_WATCH"): env.setdefault("PERF_ANSWER", os.path.join(Q5, f"{a.tag}-{a.test}.answer.txt"))   # debug cells: the text the server streamed, kept even when it crashed or the cell was stopped
    fix_build = any(os.path.basename(b.rstrip("/")).startswith("fix_") for b in re.split(r"[,\s]+", str(a.builds or "")) if b)
    env.setdefault("PERF_LEDGER", "1" if fix_build else "0")   # fix builds (fix_*) keep the ledger on: a fix-validation run records who wrote what when (user 2026-10-28); the pool write ledger is a debug tool and off in cells (it cost our side a few percent that stock never pays); --env PERF_LEDGER=1 records it: kept as q5h/TAG.ledger.csv.gz with a replay summary in TAG.out
    for kv in (a.env or []):
        k, _, v = kv.partition("="); env[k] = v
    # the cell's --env values are this bench process's own environment, so server_run's key (env that differs from it) never saw them: a cell that
    # differed from the one before only by --env reused its kept server and the values never reached a server (2026-10-27, GGML_CUDA_SCRATCH_CHECK=1)
    env["PERF_CELL_ENV"] = " ".join(sorted(a.env or []))
    t_cell0 = time.time()
    try:
        with open(out, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    finally:
        HOST[0] = None
    lines = [l.rstrip()[:200] for l in open(out, errors="replace") if re.search(r"n= *\d+ .* t/s", l)]
    with open(os.path.join(Q5, a.tag + ".out"), "a") as f:
        f.write(f"== {re.sub(r'^pf(?=\d+k)', '', a.test)} {args}\n" + "".join(l + "\n" for l in lines))
    if a.builds.count(",") == 0:
        try: shutil.copy(f"/tmp/perf-{a.builds}.log", os.path.join(Q5, a.tag + ".server.log"))
        except OSError: pass
        try:   # the start before the last one (the cold run of a clean-state cell), when it belongs to this cell
            if os.path.getmtime(f"/tmp/perf-{a.builds}.log.prev") >= t_cell0: shutil.copy(f"/tmp/perf-{a.builds}.log.prev", os.path.join(Q5, a.tag + ".server.prev.log"))
        except OSError: pass
    # the pool write ledgers this cell's servers wrote (one per server start): gzipped next to the server log, the replay's summary (overlaps, uploads a lend
    # or the KV zone crossed while they ran) appended to TAG.out
    import gzip
    led = sorted((p for p in glob.glob(os.path.join(DATA, "pool-ledger-*.csv")) if os.path.getmtime(p) >= t_cell0), key=os.path.getmtime)
    for i, p in enumerate(led):
        dst = os.path.join(Q5, f"{a.tag}.ledger{'' if i == 0 else '.' + str(i)}.csv.gz")
        with open(p, "rb") as fi, gzip.open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(p)
        rep = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.realpath(__file__)), "pool_ledger.py"), dst, "--max", "8"], capture_output=True, text=True)
        with open(os.path.join(Q5, a.tag + ".out"), "a") as f:
            f.write(f"-- pool ledger {os.path.basename(dst)} (replay: fork/pool_ledger.py {dst})\n" + rep.stdout)
    if not lines:
        errs = [l.rstrip()[:240] for l in open(out, errors="replace") if re.search(r"FAIL|Traceback|Error|error:|PIN_FAIL|out of memory|Compute error|Segmentation|abort|Killed", l)][-4:]
        with open(os.path.join(Q5, "queue", "errors.log"), "a") as f:
            f.write(f"{time.strftime('%F %T')} ERROR {a.tag} {a.test} builds={a.builds} model={os.path.basename(a.model)} run.py exit={rc}\n" + "".join(e + "\n" for e in errs))
        return 1
    return 0
def cmd_hotmapmat(a):
    """Reads a --moe dynhot store (<state dir>/hotmap/<model>.hotmap) and prints, for its records in the order the requests were stored, the key cosine of each
    pooling (mean, max) and the expert overlap of every pair: overlap = per layer the share of its uses each expert got, the smaller of the two shares summed
    over the experts, averaged over the layers either request used (1 = the same experts used alike); shared hot = of the experts used above their layer's
    mean by the row's request, the part the column's request also used above its mean. Last the correlation of each pooling's cosine with the overlap over
    all pairs: the pooling whose key predicts the experts used better has the higher one. Records are one request each when the run used dynhot-merge=2
    (nothing pools); --names labels them (a comma list, in storing order)."""
    import struct
    raw = open(a.file, "rb").read()
    if raw[:8] != b"MOETOPC3":
        sys.exit(f"{a.file}: not a hotmap store of this layout (MOETOPC3)")
    e, l, x, _ = struct.unpack_from("<4I", raw, 8)
    kb = (e + 1) // 2   # keys are 4 bits per value: q = nibble - 8, times qinv (1 / length of q); two keys per record (mean pooling, max pooling)
    rec = 4 + 2 * (4 + kb) + 4 * l * x
    n = (len(raw) - 24) // rec
    keys, cnt = ([], []), []
    for i in range(n):
        o = 24 + i * rec + 4
        for k in range(2):
            qi = struct.unpack_from("<f", raw, o)[0]
            keys[k].append([(((raw[o + 4 + j // 2] >> (4 * (j & 1))) & 15) - 8) * qi for j in range(e)])
            o += 4 + kb
        cnt.append([struct.unpack_from(f"<{x}I", raw, o + 4 * x * j) for j in range(l)])
    names = ((a.names.split(",") if a.names else []) + [str(i + 1) for i in range(n)])[:n]
    def overlap(p, q):
        tot, layers = 0.0, 0
        for j in range(l):
            tp, tq = sum(p[j]), sum(q[j])
            if tp == 0 and tq == 0:
                continue
            layers += 1
            if tp and tq:
                tot += sum(min(u / tp, v / tq) for u, v in zip(p[j], q[j]))
        return tot / layers if layers else 0.0
    hots = [[{k for k, u in enumerate(c[j]) if u > 0 and u * x >= sum(c[j])} for j in range(l)] for c in cnt]
    def shared(i, k):
        den = sum(len(s) for s in hots[i])
        return sum(len(s & t) for s, t in zip(hots[i], hots[k])) / den if den else 0.0
    def cos(p, q):
        return sum(u * v for u, v in zip(p, q))
    ov = [[overlap(cnt[i], cnt[k]) for k in range(n)] for i in range(n)]
    mats = (("key cosine, mean pooling", [[cos(keys[0][i], keys[0][k]) for k in range(n)] for i in range(n)]),
            ("key cosine, max pooling", [[cos(keys[1][i], keys[1][k]) for k in range(n)] for i in range(n)]),
            ("expert overlap", ov),
            ("shared hot experts (row's hot set also hot in column)", [[shared(i, k) for k in range(n)] for i in range(n)]))
    for title, m in mats:
        print(f"\n{title}")
        print(" " * 12 + "".join(f"{nm[:7]:>8}" for nm in names))
        for i in range(n):
            print(f"{names[i][:11]:<12}" + "".join(f"{m[i][k]:8.2f}" for k in range(n)))
    pairs = [(i, k) for i in range(n) for k in range(i + 1, n)]
    def corr(m):
        xs, ys = [m[i][k] for i, k in pairs], [ov[i][k] for i, k in pairs]
        if len(xs) < 3:
            return float("nan")
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = sum((u - mx) ** 2 for u in xs) ** 0.5
        sy = sum((v - my) ** 2 for v in ys) ** 0.5
        return sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")
    print(f"\ncorrelation of key cosine with expert overlap over {len(pairs)} pairs: mean pooling {corr(mats[0][1]):.2f}, max pooling {corr(mats[1][1]):.2f}")

def cmd_hitlog(a):
    """Per request of the cells whose name (runs.db campaign: build-cell-test) starts with PREFIX: the cache hit rate the server printed for that request
    ("moe cache = on, hit X% this request"), the uploads it caused, prompt and decode speed, and the --moe dynhot lines of its prompt (nearest stored
    request and its cosine by both poolings, distinct tokens, key and search time, experts dropped at decode start, time to store the request before it).
    Each request leaves a copy of the server's log (bench/logs/runs/<end time>-<build>-<test>.log, the whole log of a kept server so far): the row's log
    is the one named by its end time, and its request is the last one in it."""
    import datetime, sqlite3
    fork = os.path.dirname(os.path.abspath(__file__))   # runs.db and bench/logs sit beside run.py (HERE is the builds directory)
    db = sqlite3.connect(os.path.join(fork, "runs.db"))
    rows = db.execute("select campaign, end_ts, tps, pp, build, test from runs where campaign like ? order by rowid", (a.prefix + "%",)).fetchall()
    if not rows:
        sys.exit(f"no runs.db rows whose campaign starts with {a.prefix}")
    logs = []
    for f in glob.glob(os.path.join(fork, "bench", "logs", "runs", "*.log")):
        m = re.match(r"(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)-(.*)\.log$", os.path.basename(f))
        if m:
            logs.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S"), m.group(2), f))
    print(f"{'cell':<26}{'hit %':>7}{'uploads':>9}{'pp t/s':>9}{'dec t/s':>9}  dynhot (prompt: nearest stored request; decode start: experts dropped)")
    for camp, en, tps, pp, build, test in rows:
        te = datetime.datetime.strptime(en, "%Y-%m-%d %H:%M:%S")
        cand = [l for l in logs if l[1] == f"{build}-{test}" and abs((l[0] - te).total_seconds()) <= 5]
        hit = up = ""
        dyn = []
        if cand:
            lines = open(min(cand, key=lambda l: abs((l[0] - te).total_seconds()))[2], errors="replace").read().split("\n")
            ends = [i for i, l in enumerate(lines) if "this request /" in l]
            if ends:
                start = ends[-2] + 1 if len(ends) > 1 else 0
                for line in lines[start:ends[-1] + 3]:
                    m = re.search(r"hit ([\d.]+)% this request", line)
                    if m:
                        hit = m.group(1)
                    m = re.search(r"moe churn = (\d+) uploads", line)
                    if m:
                        up = m.group(1)
                    m = re.search(r"dynhot: prompt of (\d+) tokens \((\d+) distinct\), nearest of (\d+) stored requests at cosine ([-\d.]+) by (\w+) pooling \(([-\d.]+) by (\w+)\): (.*?) \(key ([\d.]+) ms, search ([\d.]+) ms\)", line)
                    if m:
                        dyn.append(f"{m.group(1)} tok ({m.group(2)} distinct), {m.group(3)} stored, cos {m.group(5)} {m.group(4)} / {m.group(7)} {m.group(6)}: {m.group(8)}, key {m.group(9)} ms search {m.group(10)} ms")
                    m = re.search(r"dynhot: the prompt is over: (\d+) cached experts .* dropped \(([\d.]+) ms", line)
                    if m:
                        dyn.append(f"dropped {m.group(1)} ({m.group(2)} ms)")
                    m = re.search(r"dynhot: request stored: best overlap with a record ([\d.]+), threshold ([\d.]+).*?: (.*?) \((\d+) records, ([\d.]+) ms\)", line)
                    if m:
                        dyn.append(f"stored the request before ({m.group(3)}, overlap {m.group(1)}, {m.group(5)} ms)")
        name = camp[len(a.prefix):] if camp.startswith(a.prefix) else camp
        print(f"{name[:25]:<26}{hit:>7}{up:>9}{(pp or 0):9.1f}{(tps or 0):9.1f}  {'; '.join(dyn)}")

def cmd_stop(a):
    """Stops a running measurement chain for real: the harness (bench, regress, exp, run, quick, ctl, prompt with their flock wrappers), the servers and tools of our builds and an nsys around them, every descendant of it (the
    servers and tools it started: a killed script does not take its children with it, and a lone kill of the harness lets the parent script start the next cell),
    and the .sh scripts that started it. `stop 2 3` or `stop --last 1` stops only those tasks (a task = a chain script with everything it started, or a lone harness; numbered by --dry).
    Never the shell that runs this command or its ancestors (a plain `bash -c` is left alone, only real script files count).
    SIGTERM first, SIGKILL for what is still there after --grace seconds, then it waits until the GPUs have (almost) no memory in use, and puts the power limits
    to the schedule (power_schedule: day 220/280 W, night 200/200 W) unless --no-power. --dry lists what it would stop and changes nothing.
    A build (docker) is not a measurement and is not touched; stop it with docker."""
    import signal
    me = os.getpid()
    def read_proc(pid):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            return argv, ppid
        except (OSError, ValueError, IndexError):
            return None, None
    procs = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            argv, ppid = read_proc(int(d))
            if argv:
                procs[int(d)] = (argv, ppid)
    children = {}
    for pid, (argv, ppid) in procs.items():
        children.setdefault(ppid, []).append(pid)
    protected = set()   # this process and its ancestors
    pid = me
    while pid in procs and pid > 1:
        protected.add(pid)
        pid = procs[pid][1]
    def is_script(argv):   # `bash ./something.sh ...` or `python3 ./driver.py ...` (adaptive stress drivers), not `bash -c ...`
        b = os.path.basename(argv[0])
        return len(argv) >= 2 and ((b in ("bash", "sh") and argv[1].endswith(".sh")) or (b.startswith("python") and argv[1].endswith(".py") and not argv[1].endswith("run.py")))
    def is_harness(argv):
        py = os.path.basename(argv[0]).startswith("python")
        if not (py and any(x.endswith("run.py") for x in argv[1:4])):
            return False
        # every subcommand that runs a model is a harness (cell, queue, ... were missing: `stop` said "nothing running" while a cell loaded its model, 2026-10-18);
        # only the ones that never start a server are left alone
        i = next(k for k in range(1, min(len(argv), 4)) if argv[k].endswith("run.py"))
        sub = next((x for x in argv[i + 1:] if not x.startswith("-")), "")
        return sub not in ("build", "show", "report", "gains", "machine", "stop", "cells", "branches", "misfires", "hugepool", "")
    def is_tool(argv):   # a server or tool of one of our builds, or an nsys wrapped around one
        exe = argv[0]
        return os.path.basename(exe) == "nsys" or any(exe.startswith(d) for d in (DEVBUILDS + "/", RELS + "/", GH_REL + "/") if d) or any(x.endswith("vramhold.py") for x in argv[1:3])
    def is_driver(pid, argv):   # a test driver script (python/bash) started from a job directory (~/.claude/jobs/*/tmp): ascend.py, descend.py, w10down.py, stress.sh ...
        try: cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError: return False
        return "/.claude/jobs/" in cwd and len(argv) >= 2 and os.path.basename(argv[0]) in ("bash", "sh", "python3", "python") and argv[1].lstrip("./").endswith((".py", ".sh"))
    roots = {p for p, (argv, _) in procs.items() if p not in protected and (is_driver(p, argv) or is_harness(argv) or is_tool(argv) or (os.path.basename(argv[0]) in ("flock", "timeout") and any(x.endswith("run.py") for x in argv)))}
    scripts = set()
    for r in roots:   # the scripts above the harness: they would start the next cell
        pid = procs[r][1]
        while pid in procs and pid > 1 and pid not in protected and is_script(procs[pid][0]):
            scripts.add(pid)
            pid = procs[pid][1]
    targets, stack = set(), list(roots | scripts)
    while stack:
        p = stack.pop()
        if p in targets or p in protected:
            continue
        targets.add(p)
        stack.extend(children.get(p, []))
    if not targets:
        print("stop: nothing of the measurement harness is running")
    # a task = one tree of targets under its top process (a chain script with everything it started, or a lone harness), numbered 1.. in start order
    tops = sorted(p for p in targets if procs[p][1] not in targets)
    def top_of(p):
        while procs[p][1] in targets:
            p = procs[p][1]
        return p
    pick = set(a.tasks or [])
    if a.last:
        pick |= set(range(len(tops) - a.last + 1, len(tops) + 1))
    if pick:
        bad = sorted(n for n in pick if not 1 <= n <= len(tops))
        if bad:
            print(f"stop: no task {bad} (running: 1..{len(tops)}); nothing stopped")
            return
        keep = {tops[n - 1] for n in pick}
        targets = {p for p in targets if top_of(p) in keep}
        scripts &= targets
    for p in sorted(targets, key=lambda q: (tops.index(top_of(q)), q)):
        kind = "script" if p in scripts else "harness" if p in roots else "child"
        print(f"  task {tops.index(top_of(p)) + 1}  {kind:8} {p:>8}  {' '.join(procs[p][0])[:100]}")
    if a.dry:
        return
    def alive(p):
        argv, _ = read_proc(p)
        return argv is not None
    for p in sorted(scripts):   # parents first: nothing may start the next cell while the rest goes down
        try: os.kill(p, signal.SIGTERM)
        except OSError: pass
    for p in sorted(targets - scripts):
        try: os.kill(p, signal.SIGTERM)
        except OSError: pass
    t_end = time.time() + a.grace
    while time.time() < t_end and any(alive(p) for p in targets):
        time.sleep(0.5)
    left = [p for p in targets if alive(p)]
    for p in left:
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
    if left:
        print(f"stop: SIGKILL for {len(left)} that ignored SIGTERM for {a.grace} s")
        time.sleep(1.0)
    # the GPUs: wait until (almost) nothing is in use
    t_end = time.time() + a.wait_gpu
    used = None
    while time.time() < t_end:
        out = subprocess.run([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
        used = max([int(float(x)) for x in out if x.replace(".", "").isdigit()] or [0])
        if used < 800:
            break
        time.sleep(2)
    print(f"stop: {len(targets)} processes ended; GPU memory in use {used} MiB" + ("" if used is None or used < 800 else "  <-- NOT free"))
    if not a.no_power:
        apply_power_schedule()


def warm_wanted(flag, no_flag, model):
    """discarded run before the measured one: only with --warm (user 2026-10-14: no warm-up runs; a bigger-than-pool model gets a second measured pass at
    the end of a campaign instead, and the better number stands)."""
    return bool(flag) and not no_flag


def cmd_hugepool(a):
    """The 1 GiB huge-page pool behind the pinned-weights cache (was resident_pin.sh). mount [SIZE like 100g | MODEL.gguf ...]: reserve the pool
    (drop caches, compact, retry until it stops growing) and mount hugetlbfs at /mnt/huge1g (HUGEFS_MOUNT); with no size the most the machine can
    spare (RAM minus max(24 GiB, 20%), HUGEFS_MARGIN_GIB). unmount: delete the cached models, unmount, give the pages back. status: pool and mount.
    Root only through sudo -n for the pool and the mount. Start the fork with GGML_CUDA_HUGEFS=/mnt/huge1g (queue/env.sh does when it is mounted)."""
    mnt = os.environ.get("HUGEFS_MOUNT", "/mnt/huge1g")
    if not os.path.exists(HUGE_NR):
        sys.exit(f"this kernel has no 1 GiB huge pages ({HUGE_NR} missing)")
    rd = lambda p: int(open(p).read().strip())
    if a.action == "status":
        print(f"pool {rd(HUGE_NR)} x 1 GiB, {rd(HUGE_FREE)} free; mount {mnt}: {'yes' if os.path.ismount(mnt) else 'no'}"
              + (f", cached files: {', '.join(sorted(os.listdir(mnt)))}" if os.path.ismount(mnt) and os.listdir(mnt) else ""))
        return
    if a.action == "unmount":
        if os.path.ismount(mnt):
            for f in os.listdir(mnt):
                try:
                    os.remove(os.path.join(mnt, f))
                except OSError:
                    subprocess.run(["sudo", "-n", "rm", "-f", os.path.join(mnt, f)])
            subprocess.run(["sudo", "-n", "umount", mnt], check=True)
        _sudo_write(HUGE_NR, 0)
        print(f"unmounted: cache files deleted, mount removed, pool {rd(HUGE_NR)} pages")
        return
    if "hugetlb_cma" in open("/proc/cmdline").read():
        sys.exit("the kernel was booted with hugetlb_cma=: those pages cannot be pinned for the GPU, boot without it")
    total = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) >> 20
    margin = int(os.environ.get("HUGEFS_MARGIN_GIB") or max(total // 5, 24))
    most = total - margin
    if most <= 0:
        sys.exit(f"not enough RAM for a pool (total {total} GiB, margin {margin} GiB)")
    targets = a.what or []
    if len(targets) == 1 and re.fullmatch(r"\d+[Gg]", targets[0]):
        want = int(targets[0][:-1])
        if want > most:
            print(f"note: {want} GiB is above the computed limit of {most} GiB (RAM {total} GiB, margin {margin} GiB), using your size")
        most = max(most, want)
    elif targets:
        total_b = 0
        for m in targets:
            if not os.path.isfile(m):
                sys.exit(f"no such file: {m}")
            base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", m)
            parts = sorted(glob.glob(base + "-*-of-*.gguf")) if base != m else [m]
            total_b += sum(os.path.getsize(p) for p in parts)
        want = (total_b >> 30) + 2   # the pinned buffer holds at most the whole model, plus slack for alignment
        if want > most:
            print(f"the model(s) need {want} GiB but at most {most} GiB can be spared (RAM {total} GiB, margin {margin} GiB): only part of it can stay in the pool")
            want = most
    else:
        want = most
    have = rd(HUGE_NR)
    if have < want:
        print(f"reserving {want} 1 GiB pages (now {have}): dropping the page cache and compacting memory first", flush=True)
        pc = "/proc/sys/vm/compaction_proactiveness"
        old_pc = open(pc).read().strip() if os.path.exists(pc) else None
        if old_pc is not None:
            _sudo_write(pc, 100)
        subprocess.run(["sync"]); _sudo_write("/proc/sys/vm/drop_caches", 3)
        last, idle, tries = have, 0, 0
        while rd(HUGE_NR) < want and idle < 3 and tries < 12:   # pages a try gets are kept: repeating with compaction in between gets further
            tries += 1
            _sudo_write("/proc/sys/vm/compact_memory", 1); _sudo_write(HUGE_NR, want)
            now = rd(HUGE_NR)
            print(f"  try {tries}: {now} of {want} pages", flush=True)
            if now > last:
                idle = 0
            else:
                idle += 1; time.sleep(2)
            last = now
        if old_pc is not None:
            _sudo_write(pc, old_pc)
    got = rd(HUGE_NR)
    if got < want:
        sys.exit(f"only {got} of {want} pages could be allocated (memory too fragmented or too small, even after repeated compaction); free memory and retry, a smaller size may work, or reboot")
    if not os.path.ismount(mnt):
        subprocess.run(["sudo", "-n", "mkdir", "-p", mnt], check=True)
        subprocess.run(["sudo", "-n", "mount", "-t", "hugetlbfs", "-o", f"pagesize=1G,uid={os.getuid()},gid={os.getgid()},mode=0775", "none", mnt], check=True)
    print(f"pool: {got} x 1 GiB ({rd(HUGE_FREE)} free of RAM {total} GiB, margin {margin} GiB), mount: {mnt}")
    print(f"start the fork with:  GGML_CUDA_HUGEFS={mnt} llama-server -m ... -lm pin")


def cmd_regress(a):
    """One fast cell (OLMoE with a restricted cache through the chat endpoint, 500 tokens, a few seconds per run): the build in DIR against the best build
    kept in BEST/current, alternating, every run in run-history.csv (campaign regress-<label>-<time>). REGRESSION (exit 1): the median is below --tol x
    the best. NEW BEST: above the best by the same margin and the ranges do not overlap: DIR is copied to BEST/<label> and is the reference from then on."""
    import fcntl, tempfile
    cand = os.path.abspath(a.dir or os.path.join(os.getcwd(), "build-dev", "bin"))
    if not os.path.exists(os.path.join(cand, "llama-server")):
        sys.exit(f"regress: no llama-server in {cand}")
    try:
        lock = queue_lock(wait=a.wait)   # none inside a queue task: the daemon holds the lock for it
    except OSError:
        print(f"regress: SKIPPED, a benchmark queue holds /tmp/queue.lock. Later: run.py regress {cand} --wait", flush=True)
        sys.exit(3)
    label = regress_label(cand)
    cur = os.path.join(BEST, "current")
    best = os.path.realpath(cur) if os.path.exists(cur) else None
    builds = {label: cand}
    if best and os.path.basename(best) != label:
        builds[os.path.basename(best)] = best
    for d in (a.seed.split(",") if a.seed else []):
        builds.setdefault(regress_label(os.path.abspath(d)), os.path.abspath(d))
    stage = tempfile.mkdtemp(prefix="regress-")
    for name, d in builds.items():
        os.symlink(d, os.path.join(stage, name))
    camp = "regress-" + label + time.strftime("-%m%d-%H%M")
    # The fixed-text comparator (test ppl: llama-perplexity, one token per decode call, the same 2048 tokens for every build): the hit rate is the same for
    # every build, so the rate differs only by what the build costs per token. The chat test generated a different text per run (hit rate +-5 points) and
    # needed ~35 runs to see 2%. No GGML_CUDA_MMF_F32_DISABLE here: it switched the TF32 matmul off only in the builds that know the switch.
    env = dict(os.environ, PERF_BUILDS=stage, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=os.environ.get("REGRESS_GPU", "1"))
    env.pop("GGML_CUDA_MMF_F32_DISABLE", None)
    t0 = time.time(); by = {}; worst = {}
    for model, margs, gpus, n_cell, big in REGRESS_CELLS:
        if not os.path.exists(model):
            continue
        n_runs = n_cell or a.n
        if gpus: env["CUDA_VISIBLE_DEVICES"] = os.environ.get("REGRESS_GPU", gpus)
        else: env.pop("CUDA_VISIBLE_DEVICES", None)
        if big:   # a big model starts only with the RAM free (a start into a full page cache or swap thrashes the machine): wait up to 15 minutes
            for _ in range(90):
                mi = dict(l.split(":") for l in open("/proc/meminfo")); av = int(mi["MemAvailable"].split()[0]) >> 20; sw = (int(mi["SwapTotal"].split()[0]) - int(mi["SwapFree"].split()[0])) >> 20
                if av >= 100 and sw < 4: break
                time.sleep(10)
            else:
                print(f"regress: {os.path.basename(model)} SKIPPED, memory not free (available {av} GiB, swap used {sw} GiB)"); continue
        ccamp = camp + "-" + os.path.basename(model)[:24]   # 6 characters made Qwen3.6 and Qwen3.8 one campaign (the second cell printed the first one's rows)
        cmd = [sys.executable, os.path.realpath(__file__), "bench", "--models", model, "--builds", ",".join(builds), "--variants", "c", "--variant", "c=" + margs,
               "-t", "ppl", "-n", str(n_runs), "--clean-state", "--campaign", ccamp, "--note", "quick regression check against the best build (fixed-text comparator)"]
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL if not a.verbose else None, stderr=subprocess.STDOUT if a.verbose else subprocess.DEVNULL)
        cell = {}
        for x in csv_load("run-history.csv"):
            if x["campaign"] == ccamp and x["ok"] != "0" and num(x, "tps") is not None:
                cell.setdefault(x["build"], []).append(num(x, "tps"))
        print(f"regress: {os.path.basename(model)} {margs}, fixed text 2 x 1024 tokens, {n_runs} runs per build, campaign {ccamp}")
        for name in builds:
            v = cell.get(name, [])
            print(f"  {name:44s} " + (f"{statistics.median(v):7.2f} t/s [{min(v):.2f}-{max(v):.2f}] n={len(v)}" if v else "no valid run") +
                  ("   <- best so far" if best and name == os.path.basename(best) else "   <- this build" if name == label else ""), flush=True)
        # the verdict uses the cell where this build is worst against the reference (the first cell when there is no reference yet)
        refn = os.path.basename(best) if best else None
        r = statistics.median(cell[label]) / statistics.median(cell[refn]) if cell.get(label) and refn and cell.get(refn) else None
        if not by or (r is not None and (worst.get("r") is None or r < worst["r"])):
            by = cell; worst = {"r": r, "model": os.path.basename(model)}
        if r is not None and r < a.tol:   # a regression in a fast cell: the slower models are not needed for the verdict
            break
    shutil.rmtree(stage, ignore_errors=True)
    print(f"regress: {time.time() - t0:.0f} s, verdict from {worst.get('model')}")
    if not by.get(label):
        print("regress: FAILED, this build produced no valid run (start it by hand: run.py regress DIR --verbose)")
        sys.exit(2)
    med = {name: statistics.median(v) for name, v in by.items()}
    ref = os.path.basename(best) if best else None
    if a.seed or not ref or ref not in med:
        top = max(med, key=med.get)
        regress_promote(top, builds[top], med[top], len(by[top]), camp, ref, "seed: fastest of " + " ".join(sorted(med)))
        print(f"regress: BEST is now {top} ({med[top]:.2f} t/s), kept in {BEST}/{top}")
        return
    if ref == label:
        print(f"regress: PASS, this build is the best so far ({med[label]:.2f} t/s)")
        return
    if a.rebase:
        regress_promote(label, cand, med[label], len(by[label]), camp, ref, "rebase")
        print(f"regress: BEST is now {label} ({med[label]:.2f} t/s), rebase; kept in {BEST}/{label}")
        return
    ratio = med[label] / med[ref]
    if ratio < a.tol:
        print(f"regress: REGRESSION, {ratio:.3f} x the best ({med[label]:.2f} against {med[ref]:.2f} t/s, limit {a.tol})")
        sys.exit(1)
    if ratio > 1 / a.tol and min(by[label]) > max(by[ref]):
        regress_promote(label, cand, med[label], len(by[label]), camp, ref, f"{ratio:.3f} x {ref}")
        print(f"regress: NEW BEST, {ratio:.3f} x the previous best; kept in {BEST}/{label}")
        return
    print(f"regress: PASS, {ratio:.3f} x the best ({med[label]:.2f} against {med[ref]:.2f} t/s)")


def _sync_runs_db():
    """whatever happened to the per-run insert (a locked database, a crash, rows merged in from another machine): at exit runs.db gets every CSV row it lacks"""
    if os.environ.get("FORK_NO_RUNS_DB"):
        return   # run.py quick without --record: nothing of the run goes into the run log
    try:
        import runlog
        runlog.sync()
    except Exception as e:
        print("runs.db sync:", e, file=sys.stderr)




# ======================================================================================================================
# Tools, merged from their former files; `python3 fork/run.py <tool> ARGS` runs one with its own argument parser.
# ======================================================================================================================

def _tool_runlog_all():
    # ---- run.py runlog-all (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py runlog-all: every run we ever recorded in one table of tools/runs.db (all_runs), plus views that make a regression visible.

usage: run.py runlog-all [--db tools/runs.db] [--readme-source]   (rebuilds all_runs and its views, leaves the other tables alone)

Sources, de-duplicated on (ts to the second, model, test, build, tps):
  csv     fork/bench/run-history.csv of this tree and of every worktree next to it (the same file at other branches)
  gitcsv  every committed version of fork/bench/run-history.csv (tools/bench/ before 2026-10-11) in git history (all branches), rows later dropped from the file included
  db      the old SQLite files (tools/moe-bench/perf.db from git history, fork/bench/archive/perf-*.db, <rels>/runs.db, <work>/**/perf*.db, rc_perf*.db)
  readme  --readme-source: the speed tables of old README versions in git (numbers from before the run log, only where no logged run of that model and test exists)
Normalised columns: fam (model family label), tgroup (test without the cli- prefix and -lv3 suffix), bno (build number from the build name or bno column),
bseq (rank of the build by the time of its first run: orders dev and release-candidate builds that have no number), side (stock / ours), tps, pp, hit, ok.
Views:
  cell_builds   per family, test, machine, side and build: best, median and number of runs, in build order
  regress       for each build of the fork: its best against the best of ALL earlier builds of that cell (ratio), flagged when below 0.9
  best_ever     the build with the best number of each cell, and the newest build's number next to it"""
    import argparse, csv, glob, io, os, re, sqlite3, statistics, subprocess, sys, tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    ap.add_argument("--readme-source", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="more run-history.csv and old perf db files, e.g. copies of the logs of pc2 and the laptop")
    ap.add_argument("--fast", action="store_true", help="only the current run-history.csv files (no git history, no old db files): seconds, for the rebuild after new records")
    ap.add_argument("--write-csv", action="store_true", help="merge the rows missing from fork/bench/run-history.csv (git history, old db files, other worktrees) into it: existing rows stay byte-identical, the missing ones are appended in its columns")
    a = ap.parse_args()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from runnorm import FAMS, fam, bno_of, tgroup, prompt_of


    rows, seen = [], set()
    raw = []   # (src, dict in run-history.csv columns) of every kept row, for --write-csv


    CUR = {}


    def add(src, ts, hw, model, args, test, build, bno, tps, pp, hit, ok, note="", rep="", tag=""):
        try:
            tps = float(tps)
        except (TypeError, ValueError):
            return
        if not tps:
            return
        ts = (ts or "")[:19]
        key = (ts, fam(model, args), test, build, round(tps, 2))
        if key in seen:
            return
        seen.add(key)
        try: pp = float(pp)
        except (TypeError, ValueError): pp = None
        try: hit = float(hit)
        except (TypeError, ValueError): hit = None
        raw.append((src, CUR))
        rows.append((src, ts, hw or "pc1", fam(model, args), tgroup(test), test, build, bno_of(build, bno), "stock" if (build or "").startswith("stock") else "ours", tps, pp, hit, 0 if str(ok) == "0" else 1, (args or "")[:60], (note or "")[:80], prompt_of(test, rep, tag)))


    def read_csv_text(text, src):
        nonlocal CUR
        for r in csv.DictReader(io.StringIO(text)):
            CUR = dict(r)
            add(src, r.get("ts"), r.get("hw"), r.get("model"), r.get("args"), r.get("test"), r.get("build"), r.get("bno"), r.get("tps"), r.get("pp"), r.get("hit"), r.get("ok", 1), r.get("note"), r.get("rep"), r.get("prompt"))


    # csv: every worktree
    for f in sorted(set(glob.glob(os.path.join(os.path.dirname(ROOT), "*/tools/bench/run-history.csv")) + glob.glob(os.path.join(os.path.dirname(ROOT), "*/fork/bench/run-history.csv")) + glob.glob(os.path.join(ROOT, ".claude/worktrees/*/tools/bench/run-history.csv")) + [os.path.join(ROOT, "fork/bench/run-history.csv")])):
        read_csv_text(open(f, errors="replace").read(), "csv")
    for f in a.extra:
        if f.endswith(".csv"):
            read_csv_text(open(f, errors="replace").read(), "extra")
    n_csv = len(rows)
    n_git = n_db = 0

    # gitcsv: every committed version
    seen_blobs = set()
    log = [] if a.fast else subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "fork/bench/run-history.csv", "tools/bench/run-history.csv", "tools/moe-bench/perf.csv"], capture_output=True, text=True).stdout.split()
    for h in log:
        for path in ("fork/bench/run-history.csv", "tools/bench/run-history.csv", "tools/moe-bench/perf.csv"):   # fork/ since 2026-10-11, tools/ before
            b = subprocess.run(["git", "-C", ROOT, "rev-parse", f"{h}:{path}"], capture_output=True, text=True).stdout.strip()
            if b and b not in seen_blobs:
                seen_blobs.add(b)
                read_csv_text(subprocess.run(["git", "-C", ROOT, "cat-file", "-p", b], capture_output=True).stdout.decode(errors="replace"), "gitcsv")
    n_git = len(rows) - n_csv


    def read_db(path, src):
        try:
            c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            cols = [x[1] for x in c.execute("pragma table_info(runs)")]
            if not cols:
                return
            for r in c.execute("select * from runs"):
                d = dict(zip(cols, r))
                nonlocal CUR
                CUR = {"ts": d.get("ts"), "commit": d.get("commit_sha"), "bno": d.get("build_no"), "tps": d.get("tps"), "pp": d.get("pp_tps"), "hit": d.get("hit_rate"), "model": d.get("model"),
                       "args": d.get("args"), "test": d.get("test"), "hw": d.get("hw") or "pc1", "ok": d.get("ok", 1), "build": d.get("build"), "md5": d.get("md5"), "campaign": "import-old-db",
                       "note": d.get("note"), "ppl": d.get("ppl"), "n_gen": d.get("n_gen"), "origin": d.get("origin"), "env": d.get("env")}
                add(src, d.get("ts"), d.get("hw"), d.get("model"), d.get("args"), d.get("test"), d.get("build"), d.get("build_no"), d.get("tps"), d.get("pp_tps"), d.get("hit_rate"), d.get("ok", 1), d.get("note"))
        except sqlite3.Error:
            pass


    dbs = glob.glob(os.path.join(ROOT, "fork/bench/archive/*.db")) + glob.glob(os.path.join(RELS, "runs.db")) + glob.glob(os.path.join(WORK, "**/perf*.db"), recursive=True) + glob.glob(os.path.join(WORK, "**/rc_perf*.db"), recursive=True)
    for f in ([] if a.fast else sorted(set(dbs))):
        read_db(f, "db")
    for f in a.extra:
        if f.endswith(".db"):
            read_db(f, "db")
    for h in ([] if a.fast else subprocess.run(["git", "-C", ROOT, "log", "--all", "--format=%H", "--", "tools/moe-bench/perf.db"], capture_output=True, text=True).stdout.split()):
        blob = subprocess.run(["git", "-C", ROOT, "cat-file", "-p", f"{h}:tools/moe-bench/perf.db"], capture_output=True).stdout
        if blob:
            with tempfile.NamedTemporaryFile(suffix=".db") as t:
                t.write(blob); t.flush(); read_db(t.name, "db")
    n_db = len(rows) - n_csv - n_git

    con = sqlite3.connect(a.db)
    con.execute("DROP TABLE IF EXISTS all_runs")
    con.execute("""CREATE TABLE all_runs (src TEXT, ts TEXT, hw TEXT, fam TEXT, tgroup TEXT, test TEXT, build TEXT, bno INTEGER, side TEXT, tps REAL, pp REAL, hit REAL, ok INTEGER, args TEXT, note TEXT, prompt TEXT, bseq INTEGER)""")
    con.executemany("INSERT INTO all_runs (src,ts,hw,fam,tgroup,test,build,bno,side,tps,pp,hit,ok,args,note,prompt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    # bseq: rank of the build by the time of its first run
    con.execute("""UPDATE all_runs SET bseq = (SELECT COUNT(*) FROM (SELECT build b, MIN(ts) t FROM all_runs GROUP BY build) x WHERE x.t < (SELECT MIN(ts) FROM all_runs y WHERE y.build = all_runs.build))""")
    con.execute("CREATE INDEX IF NOT EXISTS i_all ON all_runs (fam, tgroup, hw, side, build)")
    for v in ("cell_builds", "regress", "best_ever", "coverage"):
        con.execute(f"DROP VIEW IF EXISTS {v}")
    con.execute("""CREATE VIEW cell_builds AS SELECT fam, tgroup, hw, side, build, bseq, MIN(ts) AS first, COUNT(*) AS n, ROUND(MAX(tps), 2) AS best, ROUND(AVG(tps), 2) AS mean,
               ROUND(MAX(pp), 1) AS best_pp, ROUND(MAX(hit), 1) AS best_hit FROM all_runs WHERE ok = 1 GROUP BY fam, tgroup, hw, side, build""")
    con.execute("""CREATE VIEW regress AS SELECT fam, tgroup, hw, build, first, n, best,
               ROUND((SELECT MAX(p.best) FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq), 2) AS prior_best,
               ROUND(best / (SELECT MAX(p.best) FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq), 2) AS ratio,
               (SELECT p.build FROM cell_builds p WHERE p.fam = c.fam AND p.tgroup = c.tgroup AND p.hw = c.hw AND p.side = 'ours' AND p.bseq < c.bseq ORDER BY p.best DESC LIMIT 1) AS prior_build
               FROM cell_builds c WHERE side = 'ours' AND n >= 2""")
    con.execute("""CREATE VIEW best_ever AS SELECT fam, tgroup, hw, build AS best_build, best, first AS best_first,
               (SELECT build FROM cell_builds l WHERE l.fam = b.fam AND l.tgroup = b.tgroup AND l.hw = b.hw AND l.side = 'ours' ORDER BY l.bseq DESC LIMIT 1) AS newest_build,
               (SELECT best FROM cell_builds l WHERE l.fam = b.fam AND l.tgroup = b.tgroup AND l.hw = b.hw AND l.side = 'ours' ORDER BY l.bseq DESC LIMIT 1) AS newest_best
               FROM cell_builds b WHERE side = 'ours' AND best = (SELECT MAX(best) FROM cell_builds m WHERE m.fam = b.fam AND m.tgroup = b.tgroup AND m.hw = b.hw AND m.side = 'ours')""")
    con.execute("DROP TABLE IF EXISTS cell_stats")
    con.execute("""CREATE TABLE cell_stats (fam TEXT, tgroup TEXT, prompt TEXT, hw TEXT, side TEXT, build TEXT, n INTEGER, t_min REAL, t_med REAL, t_max REAL, t_sd REAL, spread REAL, pp_min REAL, pp_med REAL, pp_max REAL, hit REAL, first TEXT, bseq INTEGER,
               PRIMARY KEY (fam, tgroup, prompt, hw, side, build))""")
    import runstats
    runstats.recompute(con, None)
    if con.execute("SELECT name FROM sqlite_master WHERE name = 'runs'").fetchone():
        if "stats_ok" not in [r[1] for r in con.execute("PRAGMA table_info(runs)")]:
            con.execute("ALTER TABLE runs ADD COLUMN stats_ok INTEGER")
        con.execute("UPDATE runs SET stats_ok = 1")
    con.execute("""CREATE VIEW coverage AS SELECT fam, prompt, hw, side, SUM(n) AS runs, COUNT(*) AS builds, MAX(build) AS newest FROM cell_stats GROUP BY fam, prompt, hw, side""")   # runs and builds per model and prompt: the gaps
    con.commit()
    tot = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT build) FROM all_runs").fetchone()
    print(f"all_runs: {tot[0]} runs, {tot[1]} to {tot[2]}, {tot[3]} builds (csv {n_csv}, git history +{n_git}, old db +{n_db}) in {a.db}")


    if a.write_csv:
        path = os.path.join(ROOT, "fork/bench/run-history.csv")
        snap = open(path, "rb").read()
        cur_rows = list(csv.DictReader(io.StringIO(snap.decode(errors="replace"))))
        header = list(cur_rows[0].keys())
        have = set()
        for r in cur_rows:
            try:
                have.add(((r.get("ts") or "")[:19], fam(r.get("model"), r.get("args")), r.get("test"), r.get("build"), round(float(r.get("tps") or 0), 2)))
            except ValueError:
                pass
        missing = []
        for src, d in raw:
            try:
                key = ((d.get("ts") or "")[:19], fam(d.get("model"), d.get("args")), d.get("test"), d.get("build"), round(float(d.get("tps") or 0), 2))
            except ValueError:
                continue
            if key not in have:
                have.add(key); missing.append(d)
        missing.sort(key=lambda d: d.get("ts") or "")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore", lineterminator="\n")
        for d in missing:
            w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in header})
        now = open(path, "rb").read()   # rows a running benchmark appended meanwhile stay (they follow the snapshot)
        assert now.startswith(snap)
        tmp = path + ".new"
        open(tmp, "wb").write(snap + buf.getvalue().encode() + now[len(snap):])
        os.replace(tmp, path)
        print(f"run-history.csv: {len(cur_rows)} rows kept byte-identical, {len(missing)} missing rows appended")



def _tool_runq():
    # ---- run.py runq (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py runq: query every run we ever recorded (fork/runs.db table all_runs, built by run.py runlog-all).

usage: run.py runq MODEL [TEST] [BUILD ...] [--hw pc1] [--side ours|stock|both] [--sort oldest|newest|best] [--regress] [--best]
  MODEL   part of the family label or model name, case-insensitive: mimo, "glm 3.5", iq4, 27b, olmoe
  TEST    chat, t100, tetris, game4, edit4, pf12k ... (omitted: every test of the model)
  BUILD   build names or numbers as patterns (b11707, 12209, rc-final, release-b12): only these builds
  --regress   the builds of the fork whose best is below 0.9 of the best of all earlier builds (the regression list), for the model (and test)
  --best      the best build of each cell and the newest build's number next to it
Default: one line per build (--sort oldest|newest|best), min / median / max of t/s and of pp over the runs of the build (one run: just the number), runs, best hit rate, stock lines marked."""
    import argparse, os, sqlite3, statistics, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import runstats

    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    ap.add_argument("--hw", default="pc1")
    ap.add_argument("--side", default="both")
    ap.add_argument("--sort", choices=["oldest", "newest", "best"], default="oldest", help="order of the builds within each model and test: oldest first (default), newest first, or best number first")
    ap.add_argument("--prompts", action="store_true", help="one line per prompt (game4-racing, edit4-2 ...) instead of per test: the spread of each prompt on each build")
    ap.add_argument("--gaps", action="store_true", help="runs and builds per model and prompt (ours and stock) on the machine: where data is missing")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=60)
    new = runstats.refresh(con)   # records added since the last query: only their cells are recomputed
    if new < 0:
        sys.exit("no statistics yet: run  python3 fork/run.py runlog-all  once (a few minutes: it also reads the git history and the old db files)")
    if new:
        print(f"(statistics refreshed for {new} new records)", file=sys.stderr)
    model, rest = a.what[0], a.what[1:]
    tests = {r[0] for r in con.execute("SELECT DISTINCT tgroup FROM all_runs")}
    test = next((x for x in rest if x in tests), None)
    builds = [x for x in rest if x != test]
    w, p = ["LOWER(fam) LIKE ?", "hw = ?"], [f"%{model.lower()}%", a.hw]
    if test:
        w.append("tgroup = ?"); p.append(test)
    if a.side != "both":
        w.append("side = ?"); p.append(a.side)
    if builds:
        w.append("(" + " OR ".join("build LIKE ?" for _ in builds) + ")"); p += [f"%{b}%" for b in builds]
    if a.gaps:
        print(f"{'model':16} {'prompt':18} {'side':5} {'runs':>5} {'builds':>6}")
        for r in con.execute("SELECT fam, prompt, side, runs, builds FROM coverage WHERE hw = ? AND prompt NOT LIKE 'cli:%' AND LOWER(fam) LIKE ? ORDER BY fam, prompt, side", (a.hw, f"%{model.lower()}%")):
            print(f"{r[0]:16} {(r[1] or '')[:18]:18} {r[2]:5} {r[3]:5d} {r[4]:6d}")
    elif a.regress:
        q = f"SELECT fam, tgroup, build, first, n, best, prior_best, ratio, prior_build FROM regress WHERE ratio < 0.9 AND LOWER(fam) LIKE ? AND hw = ?" + (" AND tgroup = ?" if test else "") + " ORDER BY fam, tgroup, first"
        pr = [f"%{model.lower()}%", a.hw] + ([test] if test else [])
        print(f"{'model':16} {'test':8} {'build':28} {'first':10} {'n':>3} {'best':>7} {'prior best':>10} {'ratio':>6}  vs build")
        for r in con.execute(q, pr):
            print(f"{r[0]:16} {r[1]:8} {r[2][:28]:28} {r[3][:10]:10} {r[4]:3d} {r[5]:7.1f} {r[6]:10.1f} {r[7]:6.2f}  {r[8][:28]}")
    elif a.best:
        q = "SELECT fam, tgroup, best_build, best, best_first, newest_build, newest_best FROM best_ever WHERE LOWER(fam) LIKE ? AND hw = ?" + (" AND tgroup = ?" if test else "") + " ORDER BY fam, tgroup"
        pr = [f"%{model.lower()}%", a.hw] + ([test] if test else [])
        print(f"{'model':16} {'test':8} {'best build':28} {'best':>7} {'date':10}  {'newest build':28} {'newest':>7}")
        for r in con.execute(q, pr):
            print(f"{r[0]:16} {r[1]:8} {r[2][:28]:28} {r[3]:7.1f} {r[4][:10]:10}  {r[5][:28]:28} {r[6]:7.1f}")
    else:
        tg = "COALESCE(prompt, tgroup)" if a.prompts else "tgroup"
        q = f"SELECT fam, {tg}, side, build, MIN(ts), COUNT(*), MAX(tps), AVG(tps), MAX(pp), MAX(hit), bseq, GROUP_CONCAT(tps), GROUP_CONCAT(COALESCE(pp, -1)) FROM all_runs WHERE ok = 1 AND {' AND '.join(w)} GROUP BY fam, {tg}, side, build ORDER BY fam, {tg}, bseq"
        rows_ = con.execute(q, p).fetchall()
        key = {"oldest": lambda r: (r[0], r[1], r[10]), "newest": lambda r: (r[0], r[1], -r[10]), "best": lambda r: (r[0], r[1], -r[6])}[a.sort]
        cur, last = sorted(rows_, key=key), None
        def mmm(txt, w=6):   # min / median / max of a comma separated list (-1 = no value); one run: just the number
            v = sorted(float(x) for x in txt.split(",") if float(x) >= 0)
            if not v:
                return f"{'':>{w}} {'':>{w}} {'':>{w}}"
            if len(v) == 1:
                return f"{'':>{w}} {v[0]:{w}.1f} {'':>{w}}"
            return f"{v[0]:{w}.1f} {statistics.median(v):{w}.1f} {v[-1]:{w}.1f}"
        print(f"{'model':16} {'test/prompt':16} {'build':30} {'date':10} {'n':>3} {'t/s min':>6} {'median':>6} {'max':>6}   {'pp min':>6} {'median':>6} {'max':>6} {'hit%':>5} {'sd':>5} {'max/min':>7}")
        for r in cur:
            if (r[0], r[1]) != last:
                last = (r[0], r[1]); print()
            tv = [float(x) for x in r[11].split(",")]
            sd = f"{statistics.pstdev(tv):5.1f}" if len(tv) > 1 else ""
            sp = f"{max(tv) / min(tv):7.2f}" if len(tv) > 1 and min(tv) else ""
            print(f"{r[0]:16} {r[1][:16]:16} {('[stock] ' if r[2] == 'stock' else '') + r[3][:22]:30} {r[4][:10]:10} {r[5]:3d} {mmm(r[11])}   {mmm(r[12])} {(r[9] or 0):5.1f} {sd:>5} {sp:>7}")



def _tool_merge_runs():
    # ---- run.py merge-runs (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py merge-runs [--apply]: run-history.csv rows of the other machines (ssh) that this machine's fork/bench/run-history.csv lacks are appended to it
(append-only; a row is the same run when ts, hw, build, test and model agree), and inserted into runs.db (never rebuilt: load would drop the outputs table).
Without --apply only counts. Rows keep their own columns (missing ones stay empty); a copy of each fetched file is kept in /tmp/merge-runs/."""
    import csv, io, os, subprocess, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(HERE, "bench", "run-history.csv")
    SRC = [tuple(x.split("=", 1)) for x in fcfg.get("hosts", "history_sources").split(";") if "=" in x]   # [hosts] history_sources = "user@host=path/run-history.csv;..." (the run-history.csv of other machines)
    key = lambda r: (r.get("ts"), r.get("hw"), r.get("build"), r.get("test"), r.get("model"))
    with open(CSV, newline="") as f:
        rd = csv.DictReader(f); cols = rd.fieldnames; have = {key(r) for r in rd}
    new = []
    os.makedirs("/tmp/merge-runs", exist_ok=True)
    for host, path in SRC:
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=6", "-o", "BatchMode=yes", host, ("type " + path.replace("/", "\\")) if host.endswith(".2") else "cat " + path], capture_output=True, text=True)
        if p.returncode or "ts" not in p.stdout[:400]:
            print(f"{host}:{path}: not readable"); continue
        open(f"/tmp/merge-runs/{host.split('@')[1]}-{path.replace('/', '_').replace(':', '')}", "w").write(p.stdout)
        rows = [r for r in csv.DictReader(io.StringIO(p.stdout)) if r.get("ts")]
        add = [r for r in rows if key(r) not in have]
        for r in add: have.add(key(r))
        new += add
        print(f"{host}:{path}: {len(rows)} rows, {len(add)} new")
    print(f"{len(new)} new rows in total")
    if new and "--apply" in sys.argv:
        with open(CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval=""); [w.writerow(r) for r in new]
        sys.path.insert(0, HERE); import runlog   # rows are inserted into runs.db one by one: "run.py runlog load" would delete the database and with it the outputs table (texts exist only there)
        for r in new: runlog.insert(r, cols)
        print("inserted into", runlog.DB)



def _tool_pick_features():
    # ---- run.py pick-features (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py pick-features GROUPS.txt [--tree <build_tmp>/base509]: put fork features on the fast base one group at a time, each group tested before the next.
GROUPS.txt: lines 'name: sha sha sha ...' (commits in order). Per group: git cherry-pick them in the tree (a conflict: the group is skipped, tree reset), dev build,
`run.py regress` (restricted OLMoE, 19 slots, 500 tokens, against the best binary kept in <rels>/best): REGRESSION (below 0.98 x best) drops the group (tree reset to the
last good state), PASS or NEW BEST keeps it. Result per group in <work>/pick/<tree name>.log and a line in the final table."""
    import os, re, subprocess, sys, time
    HERE = os.path.dirname(os.path.realpath(__file__)); tree = os.path.join(BUILD_TMP, "base509"); args = sys.argv[1:]
    if "--tree" in args: i = args.index("--tree"); tree = args[i + 1]; del args[i:i + 2]
    groups = [(l.split(":", 1)[0].strip(), l.split(":", 1)[1].split()) for l in open(args[0]) if ":" in l and not l.startswith("#")]
    PICK = os.path.join(WORK, "pick"); os.makedirs(PICK, exist_ok=True); LOG = f"{PICK}/{os.path.basename(tree)}.log"
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
            b = subprocess.run(["bash", os.path.join(WORK, "hcf/lrel/dev-build.sh")], env={**os.environ, "W": tree, "REGRESS": "0"}, capture_output=True, text=True)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        if "DEVBUILD_EXIT 0" not in b.stdout:
            open(f"{PICK}/build-{name}.log", "w").write(b.stdout[-4000:] + b.stderr[-2000:]); say(name, f"BUILD FAILED (tail in {PICK}/build-" + name + ".log)"); git("reset", "--hard", base); res.append((name, "build failed", "")); continue
        r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "regress", os.path.join(tree, "build-dev", "bin"), "--wait"], capture_output=True, text=True)
        if r.returncode == 1:
            say(name, "first verdict", [l for l in r.stdout.splitlines() if l.startswith("regress:")][-1:], "-> again with 8 runs (fixed-text comparator)")
            r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "regress", os.path.join(tree, "build-dev", "bin"), "--wait", "-n", "8"], capture_output=True, text=True)
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



def _tool_bisect_perf():
    # ---- run.py bisect (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py bisect --good REF --bad REF --metric off|c19 [--n 4]: which commit made the build slower? Binary search over `git rev-list --reverse good..bad` (every commit, topological
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
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--name"); ap.add_argument("--tree", default=os.path.join(BUILD_TMP, "bisect")); a = ap.parse_args()
    name = a.name or f"{a.metric}-{a.good[:7]}-{a.bad[:7]}"; LOGDIR = os.path.join(BUILD_TMP, "bisect-logs"); os.makedirs(LOGDIR, exist_ok=True); LOG = os.path.join(LOGDIR, f"{name}.log")
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



def _tool_plcal():
    # ---- run.py plcal (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py plcal: find the highest power limit per GPU without throttling drops.

Each iteration runs one short, GPU-heavy request through fork/run.py bench (logged like every run, GPU log on): the pf128k test with a cut of its prompt file (about 100k tokens)
prefilled by a model that loads in seconds, both GPUs busy (split mode tensor). While it runs the 5 Hz GPU log is read live. A GPU is throttling when
  - a thermal / hardware slowdown reason (0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal slowdown) is active in >= 2 busy samples in a row, or
  - its SM clock falls below 85% of its own median of the last 10 s in >= 3 busy samples in a row (a transient drop; a steady lower clock from a cap is not a drop).
On the first throttle the run is killed, the throttling GPU's limit goes down by STEP watts, and the next iteration starts. A run that ends without a throttle
ends the search: those limits are the answer. Xid 79 in the kernel log ends everything at once.

usage: run.py plcal [--start 280+280] [--step 10] [--floor 150] [--max-iter 12] [--build stock-b11552-23b0202a1] [--model PATH]
Run with the queue stopped; it takes /tmp/queue.lock for the whole search. Result lines go to <work>/plcal.log."""
    import argparse, collections, csv, glob, os, signal, statistics, subprocess, sys, time

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import config as fcfg
    GPULOG = os.path.join(HERE, "bench", "gpulog")
    os.makedirs(WORK, exist_ok=True)
    LOG = os.path.join(WORK, "plcal.log")
    THERMAL = 0x8 | 0x20 | 0x40


    def queue_env():
        """the environment the queue daemon gives its tasks: ours plus the exports of the optional [gpu] env_sh file"""
        sh = fcfg.get("gpu", "env_sh")
        if sh and os.path.isfile(sh):
            out = subprocess.run(["bash", "-c", f". {sh} 2>/dev/null; env -0"], capture_output=True).stdout.decode(errors="replace")
            return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
        return dict(os.environ)



    def log(msg):
        line = time.strftime("%F %T ") + msg
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")


    def kernel_xid79():
        out = subprocess.run(["journalctl", "-k", "--since", "-3min", "-q"], capture_output=True, text=True).stdout
        return "Xid" in out and "79" in out and "fallen off the bus" in out


    def kill_run(proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # a server left behind by the killed bench: exact path match only
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            p = l.split(None, 1)
            if len(p) == 2 and "/rels/devbuilds/" in p[1] and p[1].split()[0].endswith("llama-server"):
                try:
                    os.kill(int(p[0]), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--start", default="280+280"); ap.add_argument("--step", type=int, default=10); ap.add_argument("--floor", type=int, default=150)
        ap.add_argument("--max-iter", type=int, default=12); ap.add_argument("--ctx", type=int, default=131072)
        ap.add_argument("--prompt", default=os.path.join(WORK, "p100k.txt"))
        ap.add_argument("--build", default="stock-b11552-23b0202a1")
        ap.add_argument("--model", default=fcfg.model("qwen36"))
        ap.add_argument("--timeout", type=int, default=240)
        a = ap.parse_args()
        lim = [int(x) for x in a.start.split("+")]
        lock = open("/tmp/queue.lock", "a")
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        log(f"plcal start: limits {lim[0]}+{lim[1]} W, step {a.step}, model {os.path.basename(a.model)}, build {a.build}, prompt {a.prompt}")
        for it in range(1, a.max_iter + 1):
            spec = f"{lim[0]}+{lim[1]}"
            camp = f"plcal-{spec}-it{it}"
            # the environment the queue daemon gives its tasks (queue/env.sh, PERF_BUILDS, PCI bus device order)
            env = queue_env()
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PERF_BUILDS=fcfg.path("devbuilds"), PERF_LOG="r")
            nvsmi = fcfg.get("gpu", "nvidia_smi")
            for gi, w in enumerate(lim):   # this tool is the only place that changes the limits (the boot defaults are fixed): set them before the run
                subprocess.run(["sudo", "-n", nvsmi, "-i", str(gi), "-pl", str(w)], capture_output=True, timeout=30)
            cmd = ["python3", os.path.join(HERE, "run.py"), "bench", "--models", a.model, "--builds", a.build, "--variants", "v",
                   "--variant=v=-sm tensor -ngl 99 -b 4096 -ub 2048 -np 1", "-t", "pf128k", "--prompt", "@" + a.prompt, "--ctx", str(a.ctx), "--tokens", "1", "-n", "1", "--no-probe", "--campaign", camp]
            t_start = time.time()
            proc = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(HERE), stdout=open(os.path.join(WORK, f"plcal-it{it}.out"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
            gfile, pos, hist, bad, thr_clk, thr_th, busy = None, 0, collections.defaultdict(lambda: collections.deque(maxlen=50)), set(), collections.Counter(), collections.Counter(), collections.Counter()
            reason = {}
            while proc.poll() is None and time.time() - t_start < a.timeout and not bad:
                time.sleep(0.4)
                if gfile is None:
                    cand = [f for f in glob.glob(os.path.join(GPULOG, "*.csv")) if not f.endswith(".cpu.csv") and os.path.getsize(f) > 0
                            and time.mktime(time.strptime(os.path.basename(f)[:15], "%Y%m%d-%H%M%S")) >= int(t_start)]   # the log of THIS run: its name carries the start time
                    if cand:
                        gfile = max(cand, key=os.path.basename)
                if gfile:
                    with open(gfile) as f:
                        f.seek(pos); chunk = f.read(); pos = f.tell()
                    for line in chunk.splitlines():
                        p = line.split(",")
                        if len(p) < 10 or not p[1].isdigit() or p[0] == "t_ms":
                            continue
                        g = int(p[1])
                        try:
                            clk = int(p[3]); util = int(p[6]); thr = int(p[9], 16)
                        except ValueError:
                            continue
                        if util < 80:
                            thr_clk[g] = thr_th[g] = 0
                            continue
                        busy[g] += 1
                        h = hist[g]
                        if thr & THERMAL:
                            thr_th[g] += 1
                            if thr_th[g] >= 2:
                                bad.add(g); reason[g] = f"thermal/HW slowdown flag 0x{thr:x} at {clk} MHz"
                        else:
                            thr_th[g] = 0
                        if len(h) >= 25 and clk < 0.85 * statistics.median(h):
                            thr_clk[g] += 1
                            if thr_clk[g] >= 3:
                                bad.add(g); reason[g] = f"clock drop to {clk} MHz (median of the last 10 s {statistics.median(h):.0f})"
                        else:
                            thr_clk[g] = 0
                            h.append(clk)
                if kernel_xid79():
                    log("XID 79 in the kernel log: stopping everything"); kill_run(proc); sys.exit(2)
            dur = time.time() - t_start
            if bad:
                kill_run(proc)
                log(f"it {it}: limits {spec} W: THROTTLE on GPU {sorted(bad)} after {dur:.0f} s ({'; '.join(f'GPU{g}: {r}' for g, r in sorted(reason.items()))}); killed")
                time.sleep(6)
                for g in bad:
                    lim[g] -= a.step
                    if lim[g] < a.floor:
                        log(f"GPU {g} reached the floor {a.floor} W: stopping"); sys.exit(1)
                continue
            if proc.poll() is None:
                kill_run(proc); log(f"it {it}: limits {spec} W: timeout after {dur:.0f} s without a throttle (killed); busy samples {dict(busy)}")
            elif proc.returncode != 0 or not sum(busy.values()):
                log(f"it {it}: limits {spec} W: RUN FAILED after {dur:.0f} s (exit {proc.returncode}, busy samples {dict(busy)}): see {WORK}/plcal-it{it}.out; not a result")
                sys.exit(3)
            else:
                log(f"it {it}: limits {spec} W: CLEAN run of {dur:.0f} s (exit {proc.returncode}), busy samples {dict(busy)}")
            log(f"RESULT: no throttling drops at {spec} W (x4 + x16)")
            return
        log(f"max iterations reached, last limits {lim[0]}+{lim[1]}")


    if __name__ == "__main__":
        main()



def _tool_baseline():
    # ---- run.py baseline (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py baseline: run our README cells (game4c, edit4c, pf12k) for every model on one build (default: the latest stock) through fork/run.py bench, read the
5 Hz GPU log of every run, and when the x4 card throttles (thermal / HW slowdown flag in >= 2 busy samples in a row, or a clock drop below 85% of its
median of the last 10 s in >= 3 busy samples in a row) lower the x4 power limit by STEP watts EVERYWHERE (live, and in the boot files named by [gpu] limits_service / limits_env
of the fork config) and run that cell again. Every run is logged by run.py as always; the verdict per cell goes to <work>/baseline.log.
usage: run.py baseline [--build stock-b11552-23b0202a1] [--models glm,qn38,q36,olmoe] [--step 10] [--floor 150] [--tag base1]
Takes /tmp/queue.lock for the whole run: stop the queue daemon first."""
    import argparse, collections, fcntl, glob, os, statistics, subprocess, sys, time

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    import config as fcfg
    GPULOG = os.path.join(HERE, "bench", "gpulog")
    os.makedirs(WORK, exist_ok=True)
    LOG = os.path.join(WORK, "baseline.log")
    THERMAL = 0x8 | 0x20 | 0x40
    NVSMI = fcfg.get("gpu", "nvidia_smi")
    MODELS = {"glm": fcfg.model("glm"), "qn38": fcfg.model("qwen38"), "q36": fcfg.model("qwen36"), "olmoe": fcfg.model("olmoe")}
    CELLS = {"glm": [("out_heavy", "game4c", [], 3), ("in_out_heavy", "edit4c", ["--tokens", "1024"], 3), ("in_heavy", "pf12k", [], 3)],
             "qn38": [("out_heavy", "game4c", [], 3), ("in_out_heavy", "edit4c", ["--tokens", "1024"], 3), ("in_heavy", "pf12k", [], 3)],
             "q36": [("out_heavy", "game4c", [], 3), ("in_out_heavy", "edit4c", ["--tokens", "1024"], 3), ("in_heavy", "pf12k", [], 3)],
             "olmoe": [("out_heavy", "tetrisc", [], 5)]}


    def queue_env():
        """the environment the queue daemon gives its tasks: ours plus the exports of the optional [gpu] env_sh file"""
        sh = fcfg.get("gpu", "env_sh")
        if sh and os.path.isfile(sh):
            out = subprocess.run(["bash", "-c", f". {sh} 2>/dev/null; env -0"], capture_output=True).stdout.decode(errors="replace")
            return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
        return dict(os.environ)



    def log(msg):
        line = time.strftime("%F %T ") + msg
        print(line, flush=True)
        open(LOG, "a").write(line + "\n")


    def sh(*cmd):
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=60)


    def live_limits():
        out = sh(NVSMI, "--query-gpu=power.limit", "--format=csv,noheader,nounits").stdout
        return [int(float(x)) for x in out.split() if x.replace(".", "").isdigit()]


    def set_x4(w):
        """x4 card (GPU 0) to w watts: live, boot service, boot defaults"""
        svc, lenv = fcfg.get("gpu", "limits_service"), fcfg.get("gpu", "limits_env")
        sh("sudo", "-n", NVSMI, "-i", "0", "-pl", str(w))
        if svc:
            sh("sudo", "-n", "sed", "-i", f"s/-i 0 -pl [0-9]*/-i 0 -pl {w}/", svc)
            sh("sudo", "-n", "systemctl", "daemon-reload")
        if not lenv:
            return
        lines = open(lenv).read().splitlines()
        open(lenv, "w").write("\n".join((f"LIMIT0={w}   # GPU0 = x4 card, lowered by run.py baseline after a throttle ({time.strftime('%F %T')})" if l.startswith("LIMIT0=") else l) for l in lines) + "\n")


    def throttled(files, t_start):
        """per GPU index: reason of the first throttle seen in the GPU logs of this cell, else {}"""
        found = {}
        for f in sorted(files):
            hist = collections.defaultdict(lambda: collections.deque(maxlen=50)); c_clk = collections.Counter(); c_th = collections.Counter()
            for line in open(f).read().splitlines()[1:]:
                p = line.split(",")
                try:
                    g = int(p[1]); clk = int(p[3]); util = int(p[6]); thr = int(p[9], 16)
                except (ValueError, IndexError):
                    continue
                if util < 80:
                    c_clk[g] = c_th[g] = 0
                    continue
                if thr & THERMAL:
                    c_th[g] += 1
                    if c_th[g] >= 2 and g not in found:
                        found[g] = f"flag 0x{thr:x} at {clk} MHz"
                else:
                    c_th[g] = 0
                h = hist[g]
                if len(h) >= 25 and clk < 0.85 * statistics.median(h):
                    c_clk[g] += 1
                    if c_clk[g] >= 3 and g not in found:
                        found[g] = f"clock drop to {clk} MHz (median of the last 10 s {statistics.median(h):.0f})"
                else:
                    c_clk[g] = 0
                    h.append(clk)
        return found


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--build", default="stock-b11552-23b0202a1"); ap.add_argument("--models", default="glm,qn38,q36,olmoe")
        ap.add_argument("--step", type=int, default=10); ap.add_argument("--floor", type=int, default=150); ap.add_argument("--tag", default="base1")
        ap.add_argument("--retries", type=int, default=4)
        a = ap.parse_args()
        lock = open("/tmp/queue.lock", "a"); fcntl.flock(lock, fcntl.LOCK_EX)
        env = queue_env()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PERF_BUILDS=fcfg.path("devbuilds"), PERF_LOG="r")
        log(f"baseline start: build {a.build}, models {a.models}, limits {live_limits()} W (x4 + x16), tag {a.tag}")
        for key in a.models.split(","):
            for tname, test, extra, n in CELLS[key]:
                for attempt in range(1, a.retries + 1):
                    camp = f"{a.tag}-{key}-{tname}" + ("" if attempt == 1 else f"-try{attempt}")
                    t0 = time.time(); lim = live_limits()
                    cmd = ["python3", os.path.join(HERE, "run.py"), "bench", "--models", MODELS[key], "--builds", a.build, "--variants", "v", "--variant=v=-np 1",
                           "-t", test, "-n", str(n), "--no-probe", "--campaign", camp] + extra
                    r = subprocess.run(cmd, env=env, cwd=os.path.dirname(HERE), stdout=open(os.path.join(WORK, f"baseline-{camp}.out"), "w"), stderr=subprocess.STDOUT)
                    files = [f for f in glob.glob(os.path.join(GPULOG, "*.csv")) if not f.endswith(".cpu.csv")
                             and time.mktime(time.strptime(os.path.basename(f)[:15], "%Y%m%d-%H%M%S")) >= int(t0)]
                    thr = throttled(files, t0)
                    lt = {g: v for g, v in thr.items()}
                    log(f"{camp}: limits {lim[0]}+{lim[1]} W, exit {r.returncode}, {len(files)} server starts, {time.time() - t0:.0f} s, throttle: " + ("; ".join(f"GPU{g}: {v}" for g, v in sorted(lt.items())) if lt else "none"))
                    if r.returncode != 0:
                        log(f"{camp}: run.py failed, see {WORK}/baseline-{camp}.out; continuing with the next cell")
                        break
                    if 1 in lt:
                        log(f"note: x16 card throttled in {camp} (limit {lim[1]} W): logged only")
                    if 0 not in lt:
                        break
                    new = lim[0] - a.step
                    if new < a.floor:
                        log(f"x4 floor {a.floor} W reached: stopping"); return
                    set_x4(new)
                    log(f"x4 throttled at {lim[0]} W: lowered to {new} W everywhere (live, gpu-powerlimit.service, limits.env); running {camp.split('-try')[0]} again")
                    time.sleep(20)
        log(f"baseline done, final limits {live_limits()} W")


    if __name__ == "__main__":
        main()



def _tool_reg_tests():
    # ---- run.py reg-tests (merged into run.py; the code of the file, unchanged)
    __doc__ = """Regression checks, one per regression that was ever spotted, numbered. Is regression N back in a build?

  run.py reg-tests list                      the regressions: number, title, what it was, the stored last-good build
  run.py reg-tests N CURRENT [GOOD]          run check N on build CURRENT (a bin directory) and on the stored last-good build (or GOOD),
                                         through fork/run.py exp (every run lands in run-history.csv), print PASS or REGRESSION
  run.py reg-tests N CURRENT --dry           only print the experiment that would run
  run.py reg-tests all CURRENT               every default check that can run on this machine now (a check that needs a pool says SKIP without one; opt-in checks, reg 2, run by number)

CURRENT is a directory with llama-cli (and libs), e.g. <tree>/build-dev/bin. The last-good builds are stored under <rels>.
A check compares CURRENT with the last-good build on the same machine, same tool (llama-cli on both: llama-cli and llama-completion print different
rates for the same engine), own state file per arm (an arm must never learn from another arm's runs).
"""
    import csv, glob, os, re, shlex, statistics, subprocess, sys, time

    HERE = os.path.dirname(os.path.realpath(__file__))
    REPO = os.path.dirname(HERE)
    RUN = os.path.join(HERE, "run.py")
    HIST = os.path.join(HERE, "bench", "run-history.csv")
    LOGS = os.path.join(HERE, "bench", "logs")
    PROMPT = config.prompt_path("src12k")   # the README pf12k prompt (12k tokens, 100 generated, -c 16384 -ub 2048 -b 2048): prompt processing and decode in one run, the same prompt as the README table
    GOOD = RELS + "/rc-final-fd4c3d2d3"                      # rc-merge fd4c3d2d3: placement measured, pin decision with pool, no mid-split rule

    # the stored builds: directory name under RELS -> the commit it is built from. A build that is missing is rebuilt from its commit in a separate temporary worktree
    # (the docker dev build, nothing touches the current tree), the binaries are stored under RELS with a BUILD_INFO, the temporary tree is removed.
    BUILDS = {
        "rc-final-fd4c3d2d3":  {"ref": "fd4c3d2d3", "note": "rc-merge: placement measured, pin decision with pool, mid-split rule removed"},
        "b11707-49fe4b756":    {"ref": "49fe4b756", "note": "release b11707 (the README numbers)"},
        "stock-bed0a8566":     {"ref": "bed0a8566", "note": "upstream master, the stock floor"},
        "stock-def4d406a":     {"ref": "def4d406a", "note": "upstream master, the older stock"},
    }
    BUILD_SCRIPT = os.path.join(WORK, "hcf/lrel/dev-build.sh")


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
        "qn_iq1m": fcfg.model("qn_iq1m"),
        "glm30":   fcfg.model("glm"),
    }

    # each regression: what it was, how it shows, the model, the arms' extra args/env, the metric and the tolerance (CURRENT must reach tol x the good build)
    REGS = {
        1: dict(title="near-fit model forced onto the expert cache (first start and every start)",
                was="Qwen3.8-Flash-Next IQ1_M (55 GB, 48 GB VRAM): commit aab03fbe8 took the cache without measuring: 54.7 t/s generation and 469 t/s prompt against 66 and 920 with stock placement",
                model="qn_iq1m", rounds=4, args="", env={}, state=True, judge="settled", tol=0.93,
                extra_arms=[("STOCK", RELS + "/stock-docker-836d57176", True), ("B11707", RELS + "/b11707-49fe4b756", False)],
                good_build=GOOD, how="the last two starts (settled) of a fresh state file: generation and prompt t/s"),
        2: dict(title="extra graph split on a plain two-GPU layer split (mid-split rule)",
                was="the scheduler started a new split when an activation from another GPU was needed: 4 splits where stock has 3, 1.1% generation on Qwen Next (65.9 against 66.6 without it)",
                model="qn_iq1m", rounds=1, opt_in=True, args="--fork off", env={}, state=False, judge="splits", tol=0.99,
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
        for arm, b, stock in reg.get("extra_arms", ()):   # README-grade rows: upstream stock (release recipe) and the previous release next to the good/current builds
            a = f"[arm {arm}]\nbin = {b}\n"
            if reg["args"]:
                a += f"args = {reg['args']}\n"
            if stock:
                a += "origin = stock\n"
            elif reg["state"]:
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
n = 100
args = -c 16384 -ub 2048 -b 2048
log_verbosity = 4
timeout = 2400

[setup]
guard = min_avail_gb=10 max_swap_mb=8000 swapout_pages_s=400000 psi=60

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
            return "SKIP", "needs a free 1 GiB hugepage pool of 100 pages (sudo resident_pin.sh 100g right after boot)"
        if not os.path.exists(os.path.join(cur, "llama-cli")):
            return "ERROR", f"no llama-cli in {cur}"
        err = ensure_build(good, dry)   # the stored last-good build, rebuilt from its commit when missing
        if err:
            return "ERROR", err
        name = f"reg{n}-{time.strftime('%m%d-%H%M')}"
        spec = f"/tmp/{name}.ini"
        open(spec, "w").write(spec_text(n, reg, name, cur, good))
        for arm in ("GOOD", "CUR", "B11707"):
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
            print("usage: run.py reg-tests N CURRENT_BIN_DIR [GOOD_BIN_DIR] [--dry]   |   run.py reg-tests all CURRENT_BIN_DIR   |   run.py reg-tests list")
            return 2
        dry = "--dry" in a
        a = [x for x in a if x != "--dry"]
        # reg 2 (--fork off against the good build) is opt-in: --fork off has equalled stock within ~1.5% in every comparison, `all` skips it, `run.py reg-tests 2 CURRENT` runs it
        ns = [n for n, r in REGS.items() if not r.get("opt_in")] if a[0] == "all" else [int(a[0])]
        bad = 0
        for n in ns:
            status, msg = check(n, os.path.abspath(a[1]), a[2] if len(a) > 2 else None, dry)
            print(f"reg {n} [{REGS[n]['title']}]: {status}: {msg}")
            bad += status in ("REGRESSION", "ERROR")
        return 1 if bad else 0


    if __name__ == "__main__":
        sys.exit(main())



def _tool_series():
    # ---- run.py series (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py series [--base 6753a033f] [--tip main] [--branch series] [--tree <build_tmp>/series]

Builds the branch `series` from the fork's tip: starting at the upstream commit the fork is based on, ONE COMMIT PER AREA (see fork/PATCHES.md) carrying the
diff of the paths of that area. A path that belongs to no area goes into the last commit, so `git diff series main` is empty. The commits are a reading aid
(what does the fork change, area by area) and the base of upstream pull requests: only the last commit is guaranteed to build, because areas share files.
The temporary worktree is removed afterwards; nothing is pushed."""
    import argparse, os, subprocess, sys

    # area name, commit title, path prefixes (first match wins; later areas take what is left)
    AREAS = [
        ("cuda-loader", "CUDA: pinned weight loading (huge page pool, THP hybrid, registered pages), multi-threaded loader, load-mode",
         ["ggml/src/ggml-cuda/ggml-cuda.cu", "src/llama-model-loader.cpp"]),
        ("scheduler", "scheduler: host-barrier refinement, selective expert copy hook, split timer (GGML_SCHED_PROF), device-to-host async",
         ["ggml/src/ggml-backend.cpp", "ggml/include/ggml-backend.h", "ggml/include/ggml.h", "ggml/src/ggml.c"]),
        ("cpu-kernels", "CPU/CUDA kernels: Q2_0 AVX2 dot, repack and matmul fixes",
         ["ggml/src/ggml-cpu/", "ggml/src/ggml-cuda/"]),
        ("expert-cache", "MoE expert cache and self-tuner (llama-moecache-fork, llama-moestate-fork) with their graph and context hooks",
         ["src/llama-moecache-fork.", "src/llama-moestate-fork.", "src/llama-graph.", "src/llama-context.", "src/llama-cparams.h", "src/llama-ext.h", "src/llama.cpp",
          "include/llama.h", "src/CMakeLists.txt", "examples/eval-callback/"]),
        ("placement-and-flags", "automatic placement, thread tuner, --moe / --fork / --load-mode flags, server and CLI hooks",
         ["common/", "tools/cli/", "tools/server/"]),
        ("mtp-and-models", "GLM MTP draft head, hybrid-idx memory, conversion fixes",
         ["src/models/", "src/llama-model.", "src/llama-arch.", "src/llama-hparams.h", "src/llama-kv-cache.", "src/llama-memory-", "conversion/", "convert_hf_to_gguf.py"]),
        ("fork-tooling", "fork/: benchmark harness, run log, docs, experiments, build recipe; README and ignore rules",
         ["fork/", "README.md", ".gitignore"]),
    ]


    def git(*a, cwd=None, check=True):
        r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
        if check and r.returncode != 0:
            sys.exit(f"git {' '.join(a)}: {r.stderr.strip()}")
        return r.stdout


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--base", default="6753a033f"); ap.add_argument("--tip", default="main"); ap.add_argument("--branch", default="series")
        ap.add_argument("--tree", default=os.path.join(BUILD_TMP, "series"))
        a = ap.parse_args()
        repo = os.getcwd()
        changes = {}
        for line in git("diff", "--name-status", "--no-renames", a.base, a.tip).splitlines():
            st, path = line.split("\t", 1)
            changes[path] = st
        left = dict(changes)
        git("worktree", "add", "--detach", "-f", a.tree, a.base)
        try:
            git("checkout", "-q", "-B", a.branch, a.base, cwd=a.tree)
            for i, (name, title, prefixes) in enumerate(AREAS):
                last = i == len(AREAS) - 1
                mine = [p for p in left if any(p.startswith(x) for x in prefixes)] if not last else list(left)
                if last:   # fork-tooling takes what no earlier area claimed; engine files among them are listed so an area can be added
                    mine = list(left)
                    stray = [p for p in mine if not any(p.startswith(x) for x in prefixes)]
                    if stray:
                        print("not in any area (put into the last commit):", *stray[:20], sep="\n  ")
                for p in mine:
                    if left[p] == "D":
                        git("rm", "-q", "--ignore-unmatch", p, cwd=a.tree)
                    else:
                        git("checkout", a.tip, "--", p, cwd=a.tree)
                    del left[p]
                if git("status", "--short", cwd=a.tree).strip():
                    git("commit", "-q", "-m", title, cwd=a.tree)
                    print(f"{name}: {len(mine)} files", flush=True)
            left_diff = git("diff", "--stat", a.branch, a.tip, cwd=repo).strip()
            print("git diff series main:", left_diff or "empty")
        finally:
            git("worktree", "remove", "--force", a.tree, check=False)


    if __name__ == "__main__":
        main()



def _tool_selftest():
    # ---- selftest.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """run.py selftest: the checks of the fork tooling that need no GPU and no model (the CI workflow fork-ci.yml runs it).

1. run.py compiles and every tool answers --help
2. the cache simulator on a seeded synthetic routing trace keeps its invariants: the offline optimum (belady) is never beaten, a counting policy beats
   plain LRU on a skewed trace, and two runs print the same numbers
3. the config module: defaults, a TOML file, and the FORK_* variables in that order
4. every relative markdown link of README.md, fork/README.md, fork/PATCHES.md and fork/docs/*.md resolves"""
    import glob, os, py_compile, random, re, runpy, subprocess, sys, tempfile

    os.environ["FORK_NO_RUNS_DB"] = "1"
    FORK = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(FORK)
    RUN = os.path.join(FORK, "run.py")
    PY = sys.executable
    failed = []


    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failed.append(msg)


    # 1. compile, help
    py_compile.compile(RUN, doraise=True)
    check(True, "run.py compiles")
    r = subprocess.run([PY, RUN, "--help"], capture_output=True, text=True, timeout=120)
    check(r.returncode == 0 and "quick" in r.stdout, "run.py --help lists the subcommands")
    for tool in ("quick", "plcal", "baseline", "series", "bisect", "runlog-all", "runq"):
        r = subprocess.run([PY, RUN, tool, "--help"], capture_output=True, text=True, timeout=120)
        check(r.returncode == 0 and "usage" in (r.stdout + r.stderr).lower(), f"run.py {tool} --help")


    # 2. cache simulator
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
        line = next(l for l in r.stdout.splitlines() if l.startswith("cache"))
        return {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)\s+([\d.]+)%", line.split("|")[0])}


    with tempfile.TemporaryDirectory() as d:
        t = os.path.join(d, "trace.txt")
        synthetic_trace(t)
        a, b = sim(t), sim(t)
    check(a == b, "the simulator is deterministic")
    check(a["belady"] >= max(v for k, v in a.items() if k != "belady"), f"belady is the upper bound ({a['belady']}%)")
    check(a["lfu"] > a["lru"], f"on a skewed trace LFU beats LRU ({a['lfu']}% vs {a['lru']}%)")

    # 3. config precedence (run.py loaded as a module: nothing is dispatched)
    for k in [k for k in os.environ if k.startswith("FORK_") and k != "FORK_NO_RUNS_DB"]:
        del os.environ[k]
    with tempfile.TemporaryDirectory() as d:
        toml = os.path.join(d, "c.toml")
        open(toml, "w").write('[paths]\nrels = "/from/file"\n')
        os.environ["FORK_CONFIG"] = toml
        c = runpy.run_path(RUN, run_name="forkselftest")["config"]
        check(c.path("rels") == "/from/file", "a TOML setting overrides the default")
        check(c.path("best").endswith("/rels/best"), "an unset key keeps its default")
        os.environ["FORK_PATHS_RELS"] = "/from/env"
        check(c.path("rels") == "/from/env", "the FORK_* variable wins over the file")
        del os.environ["FORK_PATHS_RELS"], os.environ["FORK_CONFIG"]

    # 4. markdown links
    bad = []
    files = [os.path.join(ROOT, f) for f in ("README.md", "fork/README.md", "fork/PATCHES.md")] + sorted(glob.glob(os.path.join(ROOT, "fork/docs/*.md")))
    for f in files:
        if not os.path.exists(f):
            continue
        text = re.sub(r"```.*?```", "", open(f, encoding="utf-8").read(), flags=re.S)   # links inside code blocks are examples
        for m in re.finditer(r"\]\(([^)\s]+)\)", text):
            target = m.group(1).split("#")[0]
            if not target or re.match(r"[a-z]+:", target):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(f), target))):
                bad.append(f"{os.path.relpath(f, ROOT)}: {m.group(1)}")
    check(not bad, "every relative markdown link resolves" + ("" if not bad else ": " + "; ".join(bad[:5])))

    if failed:
        sys.exit(f"selftest: {len(failed)} check(s) failed")
    print("selftest: all checks passed")



def _tool_standin():
    """run.py standin SRC[,SRC2...] DST N: a layer-truncated copy of a GGUF (the first N blocks and the global tensors, per-layer metadata arrays cut to N).
    Same per-layer sizes, attention mix and expert shapes as the full model, so upload and compute timings per layer are real and a test runs about
    (blocks / N) times faster; the output is nonsense and the cache hit rate is not representative. Used to iterate on prefill and upload paths of models
    that do not fit in VRAM (GLM-5.3-Flash 12 blocks: 32.7 GB)."""
    import re as _re
    ap = argparse.ArgumentParser(prog="run.py standin", description=_tool_standin.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the .gguf (first part of a split model; several parts comma separated)")
    ap.add_argument("dst"); ap.add_argument("n", type=int, help="blocks to keep")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gguf-py"))
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    srcs = a.src.split(",")
    r = GGUFReader(srcs[0]); rs = [r] + [GGUFReader(p) for p in srcs[1:]]
    arch = r.fields["general.architecture"].contents()
    old = int(r.fields[f"{arch}.block_count"].contents())
    w = GGUFWriter(a.dst, arch)
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k.startswith("split.") or k == "general.architecture":
            continue
        v = f.contents(); t = f.types[0]
        if k == f"{arch}.block_count":
            v = a.n
        elif t == GGUFValueType.ARRAY and hasattr(v, "__len__") and len(v) == old:
            v = list(v)[:a.n]
        if t == GGUFValueType.ARRAY:
            w.add_key_value(k, v, t, f.types[1])
        else:
            w.add_key_value(k, v, t)
    for t in [t for rr in rs for t in rr.tensors]:
        m = _re.match(r"blk\.(\d+)\.", t.name)
        if m and int(m.group(1)) >= a.n:
            continue
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=True); w.close()
    print("standin written:", a.dst, f"({old} -> {a.n} blocks)")


def _tool_bad_runs():
    """bad-runs [--apply]: runs with a corrupt answer move out of the statistics: table bad_runs in runs.db and bench/bad_runs.csv (same columns + reason), gone from table runs and run-history.csv.
    Corrupt = a repetition loop (loop=1), a thrashing run (under 3 t/s with over 1M page faults: the page cache was starved), a nonsense answer (uniq4 < 0.4 over 300+ characters: the same few words again and again) or '!!!!' in the answer. Failed runs (ok=0: out of memory, HTTP 400, a crash) move too, with their note as the reason.
    Dry run unless --apply; both files are copied to *.bak first. Idempotent: run it again any time."""
    import sqlite3, zlib, shutil as sh
    ap = argparse.ArgumentParser(prog="run.py bad-runs", description=_tool_bad_runs.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="move the rows (default: only count them)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); dbp = os.path.join(here, "runs.db"); csvp = os.path.join(DATA, "run-history.csv"); badp = os.path.join(DATA, "bad_runs.csv")
    con = sqlite3.connect(dbp)
    bang = set()   # runs whose stored answer has '!!!!': by (ts, build, test, model) and by the answer's md5
    for ts, b, t, m, md5, blob in con.execute("select ts, build, test, model, md5, text from outputs"):
        try:
            if "!!!!" in zlib.decompress(blob).decode("utf8", "replace"): bang.add((ts, b, t, m)); bang.add(md5) if md5 else None
        except Exception: pass
    def reason(loop, uniq4, chars, key, md5=None, tps=None, faults=None, ok=None, note=None, n_gen=None, args=None):
        if str(loop) == "1": return "loop"
        if str(ok) == "0": return "failed: " + re.sub(r"\s+", " ", note or "no answer")[:90]   # out of memory, HTTP 400, a crash: no answer to judge
        try:
            if tps not in (None, "") and faults not in (None, "") and 0 < float(tps) < 3 and float(faults) > 1e6: return "thrashing (%.1f t/s, %.0fM page faults: the page cache was starved, e.g. by the huge-page pool)" % (float(tps), float(faults) / 1e6)
        except ValueError: pass
        try:
            if uniq4 not in (None, "") and float(uniq4) < 0.4 and int(float(chars or 0)) >= 300: return "nonsense (uniq4 %.2f)" % float(uniq4)
        except ValueError: pass
        try:   # under half a character per generated token: the answer stopped printing (2026-10-28, DSpark repeat requests: ~717 characters for 4096 tokens).
            # Only with a stored answer (rows from before answers were stored have 0 characters and no md5: unknown, not empty) and one slot (-np N > 1 counts
            # the tokens of every slot against one slot's text)
            np_ = re.search(r"(?:^|\s)-np\s+(\d+)", args or "")
            if n_gen not in (None, "") and chars not in (None, "") and float(n_gen) >= 256 and int(float(chars)) * 2 < float(n_gen) \
                    and (md5 or int(float(chars)) > 0) and not (np_ and int(np_.group(1)) > 1):
                return "answer stopped (%d characters for %d tokens)" % (int(float(chars)), int(float(n_gen)))
        except ValueError: pass
        return "!!!! in the answer" if key in bang or (md5 and md5 in bang) else ""
    # database
    cols = [r[1] for r in con.execute("pragma table_info(runs)")]
    bad = []
    for r in con.execute("select rowid, ts, build, test, model, \"loop\", uniq4, out_chars, md5, tps, major_faults, ok, note, n_gen, args from runs"):
        why = reason(r[5], r[6], r[7], (r[1], r[2], r[3], r[4]), r[8], r[9], r[10], r[11], r[12], r[13], r[14])
        if why: bad.append((r[0], why))
    # csv
    with open(csvp, newline="") as f: rd = csv.DictReader(f); head = rd.fieldnames; rows = list(rd)
    keep, move = [], []
    for r in rows:
        why = reason(r.get("loop"), r.get("uniq4"), r.get("out_chars"), (r.get("ts"), r.get("build"), r.get("test"), r.get("model")), r.get("md5"), r.get("tps"), r.get("major_faults"), r.get("ok"), r.get("note"),
                     r.get("n_gen"), r.get("args"))
        (move if why else keep).append(dict(r, reason=why) if why else r)
    from collections import Counter
    print(f"runs.db: {len(bad)} of {con.execute('select count(*) from runs').fetchone()[0]} runs corrupt:", dict(Counter(w.split(" (")[0].split(":")[0] for _, w in bad)))
    print(f"{os.path.basename(csvp)}: {len(move)} of {len(rows)} rows corrupt:", dict(Counter(r['reason'].split(' (')[0].split(':')[0] for r in move)))
    if not a.apply: print("dry run: --apply moves them"); return
    sh.copy(dbp, dbp + ".bak"); sh.copy(csvp, csvp + ".bak")
    have = {r[1] for r in con.execute("pragma table_info(bad_runs)")}
    if not have: con.execute("create table bad_runs as select * from runs where 0"); con.execute("alter table bad_runs add column reason text")
    else:
        for c in cols:
            if c not in have: con.execute(f'alter table bad_runs add column "{c}"')
    cl = ",".join(f'"{c}"' for c in cols)
    from runnorm import fam, tgroup, prompt_of
    cells = set()
    for rid, why in bad:
        ts, model, args, test, build, tps, rep, prompt, hw = con.execute("select ts, model, args, test, build, tps, rep, prompt, hw from runs where rowid=?", (rid,)).fetchone()
        con.execute(f'insert into bad_runs ({cl}, reason) select {cl}, ? from runs where rowid=?', (why, rid))
        con.execute("delete from runs where rowid=?", (rid,))
        f_, tg, pr = fam(model, args), tgroup(test), prompt_of(test, rep, prompt)
        if tps and "all_runs" in {r[0] for r in con.execute("select name from sqlite_master")}:
            con.execute("delete from all_runs where ts=? and fam=? and test=? and build=? and round(tps,2)=round(?,2)", ((ts or "")[:19], f_, test, build, tps))
            cells.add((f_, tg, pr, hw or "pc1", "stock" if (build or "").startswith("stock") else "ours", build))
    if cells: runstats.recompute(con, sorted(cells))
    con.commit()
    new = not os.path.exists(badp)
    with open(badp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head + ["reason"], lineterminator="\n")
        if new: w.writeheader()
        w.writerows(move)
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, lineterminator="\n"); w.writeheader(); w.writerows(keep)
    print(f"moved; backups {os.path.basename(dbp)}.bak and {os.path.basename(csvp)}.bak next to the files")


def ppl_chunks_store(start_ts, build, model, test, log):
    """the cumulative perplexity after each chunk of a llama-perplexity run, kept in runs.db (table ppl_chunks) and in bench/ppl-chunks.csv: two builds on the same text are compared chunk by chunk (run.py ppl-table)"""
    import sqlite3
    ch = {int(k): float(v) for k, v in re.findall(r"\[(\d+)\]([0-9.]+)", log)}
    if not ch: return
    mo = os.path.basename(model)
    with open(os.path.join(DATA, "ppl-chunks.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(["start_ts", "build", "model", "test", "chunk", "cum_ppl"])
        for k, v in sorted(ch.items()): w.writerow([start_ts, build, mo, test, k, v])
    try:
        c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
        c.execute("create table if not exists ppl_chunks(start_ts text, build text, model text, test text, chunk integer, cum_ppl real, primary key(start_ts, build, model, chunk))")
        c.executemany("insert or replace into ppl_chunks values(?,?,?,?,?,?)", [(start_ts, build, mo, test, k, v) for k, v in sorted(ch.items())]); c.commit(); c.close()
    except sqlite3.Error as e: print(f"ppl_chunks: {e}", flush=True)


PPL_VARIANTS = {"default": "", "nogate": "moe:sparse-gate=0", "one": "moe:dense-split=0 moe:prefill-split=0", "dense": "moe:dense-split=1 moe:split-experts=0 moe:prefill-split=1", "split": "moe:dense-split=1 moe:split-experts=1 moe:prefill-split=1", "f16": "moe:dense-split=1 moe:split-experts=1 moe:prefill-split=1 moe:split-f16=1", "f32": "moe:dense-split=1 moe:split-experts=1 moe:prefill-split=1 moe:split-f16=0"}


def _tool_ppl():
    """ppl [-onl TEXT] [-b BUILD] [--variants default,one,dense,split] [--chunks N] [-nrt] [-n]: the perplexity tests. For every model of the gainers list (smallest first) the official stock binary and then BUILD (default latest) in each variant:
    default (the planner decides), one (one card forced), dense (dense split only, no experts on the helper), split (dense split with experts forced), f16 and f32 (split with the hand-off between the cards forced to that precision, --moe split-f16=1 / 0; both in the default list, because the default precision changed from f16 to f32 at b13600). The per-chunk values go to runs.db (ppl_chunks); the table is `run.py ppl-table --readme`"""
    import sqlite3
    ap = argparse.ArgumentParser(prog="run.py ppl", description=_tool_ppl.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-onl", default="", help="models whose alias or file name contains TEXT (comma list)")
    ap.add_argument("-b", "--build", default="latest")
    ap.add_argument("--variants", default="default,one,dense,f16,f32")
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--no-stock", action="store_true", help="skip the official stock binary (long-context checks: stock at 32k cells takes ~50 min on GLM)")
    ap.add_argument("--test", default="pplb", choices=["pplb", "pplc", "ppl", "ppll"], help="pplb: 1024-token passes (the prompt path, default); ppl: one token per call (the decode path: cached experts on the GPU, misses on the CPU; slow, use --chunks 2 and --variants default); pplc: 1024 tokens in 256-token micro-batches")
    ap.add_argument("-nrt", nargs="?", type=int, const=1, default=0, metavar="DAYS", help="skip variants already stored in the last DAYS days (default 1 = today)")
    ap.add_argument("-n", "--dry", action="store_true")
    ap.add_argument("--table", action="store_true", help="only print the README table")
    a = ap.parse_args()
    if a.table: return _tool_ppl_table_readme()
    here = os.path.dirname(os.path.abspath(__file__))
    models = []
    for r in csv.DictReader(open(os.path.join(here, "runs", "gainers.csv"), newline="")):
        if r["model"] not in models: models.append(r["model"])
    want = [x for x in a.onl.split(",") if x]
    models = [(fcfg.model(al), al) for al in models if not want or any(w.lower() in al.lower() or w.lower() in os.path.basename(fcfg.model(al)).lower() for w in want)]
    models = [m for m, al in models if os.path.exists(m)]
    models.sort(key=lambda m: sum(os.path.getsize(f) for f in glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", m))))
    ours = os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, resolve_build(a.build))))
    stock = os.path.basename(os.path.realpath(os.path.join(DEVBUILDS, "stock-latest")))
    con = sqlite3.connect(os.path.join(here, "runs.db"))
    def done(m, build, v):
        if not a.nrt: return False
        try: return bool(con.execute("select 1 from ppl_chunks where model=? and build=? and test='%s' and date(start_ts)>=date('now','localtime','-%d days') limit 1" % (a.test, a.nrt - 1,), (os.path.basename(m), build + (" --moe " + ",".join(PPL_VARIANTS[v].replace("moe:", "").split()) if PPL_VARIANTS[v] else ""))).fetchone())
        except sqlite3.Error: return False
    for m in models:
        nosplit = os.path.basename(m).startswith(("Qwen3.8-27B",))   # plain dense models have no experts and no token-range graphs: no two-card path, the split variants would only repeat the one-card run
        jobs = ([] if a.no_stock else [(stock, "default", "")]) + [(ours, v, PPL_VARIANTS[v]) for v in a.variants.split(",") if not (nosplit and v not in ("default", "one"))]
        for build, v, words in jobs:
            if done(m, build, v): print(f"skip {os.path.basename(m)} {build} {v}: stored in the last {a.nrt} day(s)"); continue
            cmd = [sys.executable, os.path.abspath(__file__), "bench", "--models", m, "--builds", build, "-t", a.test, "-n", "1", "--campaign", "ppl-pair", "--variants", v] + (["--variant", f"{v}={words}"] if words else [])
            print(" ".join(shlex.quote(c) for c in cmd), flush=True)
            if a.dry: continue
            wait_vram_free()
            subprocess.run(cmd, env=dict(os.environ, PERF_PPL_CHUNKS=str(a.chunks)), stdout=subprocess.DEVNULL)
    if not a.dry: _tool_ppl_table_readme()
    return 0


def _tool_ppl_table_readme():
    sys.argv = [sys.argv[0], "--readme"]
    return _tool_ppl_table()


def _tool_ppl_table():
    """ppl-table [MODEL]: perplexity of each build against the newest official stock on the same text and chunks: final perplexity, per-chunk difference in % (mean +- s.e.), chunks worse"""
    import sqlite3
    ap = argparse.ArgumentParser(prog="run.py ppl-table", description=_tool_ppl_table.__doc__)
    ap.add_argument("model", nargs="?", default="", help="part of the model file name")
    ap.add_argument("--readme", action="store_true", help="the README table: one row per model, columns default / one card / dense only / two cards f16 / two cards f32, newest run of each")
    ap.add_argument("--build", default="", help="the build under test for --readme (default: the newest build number with perplexity runs)")
    ap.add_argument("--test", default="pplb", help="which perplexity test to tabulate: pplb (1024-token passes, default), ppl (one token per call, the decode path), pplc")
    ap.add_argument("--ref", default="stock", help="the build the others are compared with (name starts with this; default the newest official stock)")
    a = ap.parse_args()
    c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db"))
    runs = {}   # (model, test, start_ts, build) -> [cum ppl per chunk]
    try:
        for ts, b, m, t, k, v in c.execute("select start_ts, build, model, test, chunk, cum_ppl from ppl_chunks where model like ? and test=? order by start_ts, chunk", (f"%{a.model}%", a.test)): runs.setdefault((m, t, ts, b), []).append(v)
    except sqlite3.Error: print("no ppl runs stored yet"); return 0
    nll = lambda cs: [(k + 1) * math.log(x) - (k * math.log(cs[k - 1]) if k else 0.0) for k, x in enumerate(cs)]   # per-chunk mean negative log likelihood from the cumulative perplexities
    path = lambda m, ts: next(iter(c.execute("select pf_all from runs where start_ts=? and model=? and pf_all is not null", (ts, m)).fetchone() or [""]), "")   # the passes of the run by path (from the log)
    if a.readme:
        def kind(b):
            if b.startswith(a.ref): return None
            if "dense-split=0" in b: return "one card"
            if "split-experts=0" in b: return "dense only"
            if "dense-split=1" in b:   # forced two cards: the precision asked, else the build's default (f16 before b13600, f32 from it)
                if "split-f16=1" in b: return "two cards f16"
                if "split-f16=0" in b: return "two cards f32"
                n = re.match(r"b(\d+)", b)
                return "two cards f32" if n and int(n.group(1)) >= 13600 else "two cards f16"
            return "default"
        bt = a.build or max((x[3].split(" ")[0] for x in runs if re.fullmatch(r"b\d+", x[3].split(" ")[0])), key=lambda x: int(x[1:]))   # the build under test
        NOSPLIT = ("Qwen3.8-27B",)   # plain dense models: no experts and no token-range graphs, so no two-card path exists; the split columns would only repeat the one-card run
        cells = {}   # model -> kind -> (ts, text)
        for (m, t, ts, b), cs in sorted(runs.items()):
            k = kind(b)
            if k is None or b.startswith("stock") or b.split(" ")[0] != bt: continue
            if k in ("dense only", "two cards f16", "two cards f32") and m.startswith(NOSPLIT): continue
            st = [(sts, sv) for (sm, stt, sts, sb), sv in runs.items() if sm == m and stt == t and sb.startswith(a.ref) and len(sv) >= len(cs)]
            if not st: continue
            sv = max(st)[1]; d = [100.0 * (x - y) for x, y in zip(nll(cs), nll(sv))]; n = len(d); mean = sum(d) / n
            se = (sum((x - mean) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
            cells.setdefault((m, t), {"stock": f"{sv[n - 1]:.4f}", "n": n})[k] = f"{cs[-1]:.4f} ({mean:+.2f} ± {se:.2f}%)"   # runs are in time order: the newest of a column stays
        print("| model | stock ppl | ours default | ours, one card | ours, dense split only | ours, two cards f16 | ours, two cards f32 | chunks |\n|---|---|---|---|---|---|---|---|")
        for (m, t), v in sorted(cells.items()):
            print(f"| {m.replace('.gguf', '')} | {v['stock']} | " + " | ".join(v.get(k, "-") for k in ("default", "one card", "dense only", "two cards f16", "two cards f32")) + f" | {v['n']} |")
        return 0
    print("| model | test | build | reference | chunks | final ppl | reference ppl | difference per chunk, % (mean +- s.e.) | chunks worse | passes by path |\n|---|---|---|---|---|---|---|---|---|---|")
    for (m, t, ts, b), cs in sorted(runs.items()):
        if b.startswith(a.ref): continue
        st = [(sts, sb, v) for (sm, stt, sts, sb), v in runs.items() if sm == m and stt == t and sb.startswith(a.ref) and len(v) >= len(cs)]
        if not st: continue
        sts, sb, sv = max(st)
        d = [100.0 * (x - y) for x, y in zip(nll(cs), nll(sv))]; n = len(d); mean = sum(d) / n
        se = (sum((x - mean) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else float("nan")
        print(f"| {m[:34]} | {t} | {b} | {sb} | {n} | {cs[-1]:.4f} | {sv[n - 1]:.4f} | {mean:+.2f} +- {se:.2f} | {sum(x > 0 for x in d)}/{n} | {path(m, ts)} |")
    return 0


def _tool_out():
    """out ID [ID2]: the answer of run ID (the id column of the results, runs.db rowid); two ids: both answers and how far they agree (common prefix, similarity)"""
    import sqlite3, zlib, difflib
    ap = argparse.ArgumentParser(prog="run.py out", description=_tool_out.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", type=lambda v: int(v.rstrip("sS")), help="run id; the s suffix of a stock file name is accepted")
    ap.add_argument("-n", type=int, default=0, help="only the first N characters of each answer")
    a = ap.parse_args()
    con = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.db")); texts = []
    for i in a.ids:
        r = con.execute("select ts, build, test, model, md5, tps, pp, n_gen, \"loop\", uniq4 from runs where rowid=?", (i,)).fetchone()
        if not r: print(f"{i}: no such run"); texts.append(None); continue
        o = con.execute("select text from outputs where build=? and test=? and model=? and abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) <= 120 order by abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) limit 1", (r[1], r[2], r[3], r[0], r[0])).fetchone() or (con.execute("select text from outputs where md5=? and md5!=''", (r[4],)).fetchone() if r[4] else None)
        t = zlib.decompress(o[0]).decode("utf8", "replace") if o else None; texts.append(t)
        shade = 97 if len(texts) == 1 else 93   # the first answer white, the second yellow
        print(col(f"1;{shade}", f"=== run {i}: {r[1]} {r[2]} {os.path.basename(r[3])}  {r[5] and round(r[5], 2)} t/s  pp {r[6] and round(r[6], 1)}  answer {r[7] and int(r[7])} tokens  uniq4 {r[9]}  {r[0]}"))
        print(col(shade, (t[:a.n] if a.n and t else t) if t else "(no stored answer for this run)"))
    if len(texts) == 2 and all(texts):
        x, y = texts; p = 0
        while p < min(len(x), len(y)) and x[p] == y[p]: p += 1
        print(f"--- agree for {p} characters of {len(x)} / {len(y)}; similarity {difflib.SequenceMatcher(None, x, y, autojunk=False).ratio():.3f}")


TOOLS = {
    'out': _tool_out,
    'ppl': _tool_ppl,
    'ppl-table': _tool_ppl_table,
    'bad-runs': _tool_bad_runs,
    'standin': _tool_standin,
    'runlog-all': _tool_runlog_all,
    'runq': _tool_runq,
    'merge-runs': _tool_merge_runs,
    'pick-features': _tool_pick_features,
    'bisect': _tool_bisect_perf,
    'plcal': _tool_plcal,
    'baseline': _tool_baseline,
    'reg-tests': _tool_reg_tests,
    'series': _tool_series,
    'selftest': _tool_selftest,
    'runlog': lambda: sys.modules['runlog']._cli(),
}


if __name__ == "__main__":
    import atexit
    atexit.register(_sync_runs_db)
    os.environ.setdefault("RUNPY_REPRO", "python3 fork/run.py " + shlex.join(sys.argv[1:]))   # rows record how they were started; a child run.py keeps the parent's line
    if len(sys.argv) > 1 and sys.argv[1] == "jobs": sys.argv[1] = "run"   # `run.py jobs` is the name of the old `run.py jobs` (both work; the code keeps the old subcommand key)
    if len(sys.argv) > 1 and sys.argv[1] in TOOLS:   # a merged tool: it keeps its own argument parser
        _tool = sys.argv[1]
        sys.argv = [f"run.py {_tool}"] + sys.argv[2:]
        TOOLS[_tool]()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] in ("short", "12k", "128k"):  # the old run.py usage: run.py short|12k|128k [--url U]
        sys.argv.insert(1, "prompt")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="tools (each has its own --help): " + ", ".join(sorted(TOOLS)))
    sp = ap.add_subparsers(dest="cmd", required=True)
    TESTS = ["ppl", "ppl3", "pplb", "pplc", "ppll", "tetris", "t100", "chat", "chat2", "chatv", "long1k", "in1k", "agent", "pf2k", "pf3k", "pf4k", "pf5k", "pf6k", "pf7k", "pf8k", "pf9k", "pf10k", "pf11k", "pf12k", "pf12k-stock", "pf128k", "fix", "fix12k", "edit4", "game4", "tetrisc", "game4c", "edit4c", "long1kc"]
    r = sp.add_parser("run", aliases=["jobs"], help="measured runs of one test for BUILD...")
    r.add_argument("builds", nargs="+")
    r.add_argument("-t", "--test", default="ppl", type=test_name, choices=TestChoices(TESTS + list(ALIASES)))
    r.add_argument("-n", type=int, default=1)
    r.add_argument("-e", "--env", action="append", default=[])
    r.add_argument("--note")
    r.add_argument("--campaign", help="name written to the campaign column of every row")
    r.add_argument("--bare", action="store_true", help="only -m (plus --args): the defaults a new user gets")
    r.add_argument("--plain", action="store_true", help="no cache flags (autofit), like stock")
    r.add_argument("--warm", action="store_true", help="force a discarded run before the measured one (hot run); default off")
    r.add_argument("--no-warm", action="store_true", help="never a discarded run: the first run as a user sees it")
    r.add_argument("--args", help="extra server args, e.g. '-fitt 8000'")
    add_overrides(r)
    b = sp.add_parser("bench", help="MODELS x BUILDS x VARIANTS, every cell recorded, then the report")
    b.add_argument("--models", required=True, help="comma separated model paths (on the machine that runs them)")
    b.add_argument("--builds", required=True, help="comma separated build dirs; names starting with stock run the default variant only")
    b.add_argument("--variants", help="comma separated variant names (default: all of default,cache0,atoff and any --variant)")
    b.add_argument("--variant", action="append", default=[], help="NAME=ARGS: add or override a variant, e.g. gate3='--moe upload-wait=3'")
    b.add_argument("-t", "--test", default="t100", type=test_name, choices=TestChoices(["ppl", "ppl3", "pplb", "pplc", "ppll", "t100", "tetris"] + TESTS[6:] + list(ALIASES) + list(config.user_tests())))
    b.add_argument("--stock-variant", action="append", default=[], help="NAME=ARGS: stock builds also run this placement (e.g. tuned='--n-cpu-moe 30'): the hand-tuned baseline")
    b.add_argument("--rep-from", type=int, default=0, help="first repetition to run (0-based): -n 2 --rep-from 1 repeats only the second repetition, e.g. one that failed")
    b.add_argument("-n", type=int, help="measured runs per cell, interleaved ABBA (default 3 for fix/fix12k, else 2); the first run of a cell is preceded by a discarded one")
    b.add_argument("--machine", default="pc1", help="pc1 = this host (server tests); other ids from 'run.py machine' run llama-cli over ssh")
    b.add_argument("--campaign", help="name in table campaigns (default bench-<date>)")
    b.add_argument("--deadline", help="HH:MM: cells that would start later are skipped")
    b.add_argument("--last", type=int, help="report the last N runs per cell only (hot runs of a learning autotune)")
    b.add_argument("--tol", type=float, help="flag cells below TOL x stock as REGRESSION (e.g. 0.95)")
    b.add_argument("--note")
    b.add_argument("--no-probe", action="store_true", help="no effect (the knob probe, a side server with an injected -c 2048 before the measured run, was removed)")
    b.add_argument("--warm", action="store_true", help="force a discarded run before each measured one (hot run); default off")
    b.add_argument("--no-warm", action="store_true", help="never a discarded run: the first run as a user sees it")
    b.add_argument("--sampler", help="extra sampling parameters for every request, comma separated, e.g. repeat_penalty=1.1,dry_multiplier=0.8 (anti-repetition tests); temperature 0 / top_k 1 stay unless overridden")
    b.add_argument("--keep-server", action="store_true", help="keep the server after the run: the next run or cell with the same model, build and args uses it; the list (run run) stops it at a model change and at the end; default: a fresh server per run")
    b.add_argument("--keep-loops", action="store_true", help="keep repetition-loop outputs as normal runs (default: stored as ok=0, loop=1 and repeated up to 3 times)")
    b.add_argument("--templated", action="store_true", help="chat tests: render the model's chat template with the server's /apply-template and send the text to /completion (the loop guard then applies)")
    b.add_argument("--cli-exe", default="llama-cli", help="tool for --machine runs: llama-cli (conversation) or llama-completion (raw prompt, the same prompt as the server tests)")
    b.add_argument("--perf-prof", action="store_true", help="perf record the server during each run (499 Hz, user space) into fork/bench/perf/<time>-<build>.txt (top symbols)")
    b.add_argument("--gpulog", action="store_true", help="sample GPU power/clocks/temperature and CPU use at 5 Hz during the runs into fork/bench/gpulog/ (default: ON since 2026-10-11; --no-gpulog turns it off)")
    b.add_argument("--wrap", default=None, help="start the server under this command, e.g. 'nsys profile --trace=cuda --sample=none -o /tmp/x --force-overwrite=true' (a profiler; one cell per profile)")
    b.add_argument("--no-gpulog", action="store_true", help="no GPU/CPU sampling (power columns stay empty)")
    b.add_argument("--redo-stock", action="store_true", help="run stock cells even when 4 or more valid runs of that stock build, model and chat test (game4c / edit4c) are already recorded")
    b.add_argument("--cold-hot", action="store_true", help="every prompt (4 for game4c/edit4c) runs twice per cell, cold (a new empty state file) then hot (the state the first wrote); the first prompt also a third run hot2 (the state after that) to check the state has converged; campaigns NAME-cold, NAME-hot, NAME-hot2; no warm-up")
    b.add_argument("--clean-state", action="store_true", help="every cell (build and model) starts from an EMPTY learned state: repetition 0 is the cold start, the next ones continue from its state (a state file of its own under /tmp/runstate, the user's moe-state.ini is never touched)")
    add_overrides(b)
    rp = sp.add_parser("report", help="variants against stock per model / test / machine")
    rp.add_argument("--campaign")
    rp.add_argument("--since", help="'YYYY-MM-DD HH:MM'")
    rp.add_argument("--last", type=int)
    rp.add_argument("--tol", type=float)
    rp.add_argument("--md", action="store_true", help="also print README table rows (median, range, run ids)")
    gp = sp.add_parser("gains", help="append the gain (ratio to the baseline arm of the same campaign) of the given campaigns to gains.csv, with the GPU power limits; bench does it for its own campaign")
    gp.add_argument("campaigns", nargs="+")
    m = sp.add_parser("machine", help="list machines, or add one: machine add [ID] --ssh \"-p PORT root@host\" [--root DIR] [--os linux|windows]")
    m.add_argument("action", nargs="?", default="list", choices=["list", "add"])
    m.add_argument("id", nargs="?", help="machine name (add: default cld)")
    m.add_argument("--ssh", help="ssh alias or user@host, or a whole ssh command line in quotes (-p PORT user@host ...): then ~/.ssh/config is written for the name")
    m.add_argument("--root", help="directory holding the build dirs on that machine")
    m.add_argument("--os", default="linux")
    pu = sp.add_parser("publish", help="push the tree's branch to the fork (neurall) + runs.db; with --name the tag and the GitHub release from the REL_OUT archives")
    pu.add_argument("--tree", default="/p/bw/wt-rel"); pu.add_argument("--name"); pu.add_argument("--notes"); pu.add_argument("--replace", help="old release tag to delete first")
    dp = sp.add_parser("devpub", help="publish a dev build as the prerelease dev-BUILD on the fork (not latest) so remote machines download it instead of receiving a push"); dp.add_argument("build")
    nm = sp.add_parser("numa", help="NUMA map and node x node RAM read-rate table of a machine (-pc ID for a remote one), no model needed")
    nm.add_argument("-pc", "--machine", default="pc1")
    nm.add_argument("--reps", type=int, default=2)
    nm.add_argument("--secs", default="0.15")
    nm.add_argument("--gpu", action="store_true", help="host->device rate of every GPU against pinned buffers on every node (needs nvcc on the machine)")
    nm.set_defaults(fn=cmd_numa)
    rc = sp.add_parser("rcell", help="(run -pc) one list cell on another machine")
    for x in ("tag", "model", "build", "n", "test", "args"): rc.add_argument(x)
    for x in ("--machine", "--req", "--results", "--slots", "--campaign"): rc.add_argument(x)
    rc.add_argument("--clamp", type=int, default=0)
    rm = sp.add_parser("remote", help="a server test on a machine reached over ssh: remote ID push BUILD... | remote ID test BUILD --model PATH (server and request run there, files tailed here)")
    rm.add_argument("id", help="a machine from `run.py machine` (the ssh target may be an alias from ~/.ssh/config)")
    rm.add_argument("what", choices=["push", "test"])
    rm.add_argument("builds", nargs="+", help="build names (push: several; test: one)")
    rm.add_argument("--model", help="test: the .gguf path ON THE MACHINE")
    rm.add_argument("--req", help="test: the request json (default prompts/req-12k-1k.json)")
    rm.add_argument("--path", default="/v1/chat/completions", help="test: the endpoint the request goes to")
    rm.add_argument("--args", default="", help="test: server arguments, e.g. '-np 1 -c 4096'")
    rm.add_argument("-n", type=int, default=2, help="test: the same request this many times on one server (default 2)")
    rm.add_argument("--stats", action="store_true", help="test: also tail stats.log")
    rm.add_argument("--record", action="store_true", help="test: add the rows to the run history (hw = the machine id)")
    rm.add_argument("--campaign", help="test --record: the campaign column (the list runner sets the key -nrt looks for)")
    rm.add_argument("--test", help="test --record: the test name stored (default srv-<request>)")
    pr = sp.add_parser("prompt", help="send a benchmark prompt to a running server and print speeds")
    pr.add_argument("kind", choices=["short", "chatv", "12k", "128k", "code"])
    pr.add_argument("--url", default="http://127.0.0.1:8080")
    pr.add_argument("--idx", type=int, default=0, help="chatv: prompt number")
    add_overrides(pr)
    cl = sp.add_parser("ctl", help="one server, settings switched at runtime (LLAMA_MOE_CACHE_CTL), rotating prompts and order")
    cl.add_argument("build")
    cl.add_argument("variants", nargs="+", help="NAME:KEY=V,KEY=V ...")
    cl.add_argument("-n", type=int, default=6, help="rounds")
    cl.add_argument("--tokens", type=int, default=200)
    cl.add_argument("--warm", type=int, default=32, help="tokens answered after each switch before the timed answer")
    cl.add_argument("--prewarm", type=int, default=0, help="long answers before measuring (a self-tuning server converges)")
    cl.add_argument("--min-avail-gb", type=int, default=0, help="wait until this much RAM is free (previous pinned weights released)")
    cl.add_argument("--args", help="server args")
    cl.add_argument("--note")
    cl.add_argument("--prompt")
    cl.add_argument("--ctx", type=int)
    ex = sp.add_parser("exp", help="experiment from an INI spec (arms, env, held VRAM, memory limit, guard, sampler), every run stored; see runexp.py")
    ex.add_argument("spec")
    ex.add_argument("--dry", action="store_true")
    ex.add_argument("--only")
    ex.add_argument("--rounds", type=int)
    rg = sp.add_parser("regress", help="quick check after a commit: the build in DIR (default ./build-dev/bin) against the best build kept in <rels>/best")
    rg.add_argument("dir", nargs="?")
    rg.add_argument("-n", type=int, default=4, help="measured runs per build (one discarded run before)")
    rg.add_argument("--tol", type=float, default=0.98, help="REGRESSION below this share of the best; NEW BEST above its inverse with ranges that do not overlap")
    rg.add_argument("--seed", help="comma separated build directories: they run too and the fastest of all becomes the best (first use, or a new reference set)")
    rg.add_argument("--rebase", action="store_true", help="after the run this build becomes the best whatever the numbers (the reference of a new line)")
    rg.add_argument("--wait", action="store_true", help="wait for a running benchmark queue (/tmp/queue.lock) instead of skipping")
    rg.add_argument("--verbose", action="store_true", help="show the output of the runs")
    bd = sp.add_parser("build", help="dev build (default), --release (archives in REL_OUT, DEVBUILDS/NAME = the archives unpacked) or --stock (upstream recipe) of a worktree (--tree) or a commit (--commit), then the regress cell; --machine pc3 builds on the W10 laptop")
    bd.add_argument("name", nargs="?", help="directory under DEVBUILDS (not for --stock)")
    bd.add_argument("--release", action="store_true", help="the release recipe and packaging (was hcf/lrel/build.sh RELEASE=1); with --machine pc3 the Windows zip")
    bd.add_argument("--stock", action="store_true", help="upstream llama.cpp with upstream's release recipe (was hcf/lrel/stock-build.sh) into <gh_rel>/stock-b<build>-<hash>")
    bd.add_argument("--src", default="~/lc", help="--stock: the upstream clone")
    bd.add_argument("--ref", default="origin/master", help="--stock: the upstream commit to build")
    bd.add_argument("--tree", help="worktree to build as it is (uncommitted changes included, BUILD_INFO patch= marks them)")
    bd.add_argument("--commit", help="commit to build in a temporary worktree of --repo")
    bd.add_argument("--repo", default=BUILD_REPO)
    bd.add_argument("--machine", default="pc1", choices=["pc1", "pc3"])
    bd.add_argument("--w10-base", default="6846d4803", help="a commit the W10 repository has: the bundle carries only what is newer")
    bd.add_argument("--no-regress", action="store_true", help="skip the regress cell after the build")
    bd.add_argument("--targets", nargs="+", default=[], help="dev build: more cmake targets beside server, cli and perplexity (a test-* target turns the tests on), e.g. --targets test-scratch-fork")
    qk = sp.add_parser("quick", help="the short benchmark for any machine: --fork build against --stock, short chat answers and a 12k-token prompt, one markdown table (nothing is recorded unless --record)")
    qk.add_argument("--model", required=True, help="the .gguf file (the first part of a split model)")
    qk.add_argument("--fork", required=True, help="directory with the fork's llama-server")
    qk.add_argument("--stock", help="directory with an upstream llama-server, to compare against")
    qk.add_argument("-n", type=int, default=2, help="measured runs per test and build (default 2)")
    qk.add_argument("--tokens", type=int, default=256, help="tokens per chat answer (default 256)")
    qk.add_argument("--tests", choices=["both", "decode", "prefill"], default="both", help="which tests to run (a slow machine may want only one)")
    qk.add_argument("--args", default="", help="extra server arguments for both builds, e.g. '-c 16384'")
    qk.add_argument("--record", action="store_true", help="also add the runs to fork/bench/run-history.csv")
    qk.add_argument("--verbose", action="store_true", help="show the output of the benchmark runs")
    hp = sp.add_parser("hugepool", help="the 1 GiB huge-page pool for the pinned-weights cache: mount [SIZE|MODEL...] | unmount | status (was resident_pin.sh)")
    hp.add_argument("action", choices=["mount", "unmount", "status"])
    hp.add_argument("what", nargs="*", help="mount: a size like 100g or model file(s); nothing = the most the machine can spare")
    hl = sp.add_parser("hitlog", help="per request: cache hit rate, uploads, speeds and the --moe dynhot lines, for the cells whose runs.db name starts with PREFIX")
    hl.add_argument("prefix", help="the start of the runs.db campaign, e.g. dynhot_two_pool_keys- (build-cell-test)")
    hm = sp.add_parser("hotmapmat", help="print key cosine and expert overlap between the requests in a --moe dynhot store (hotmap/<model>.hotmap)")
    hm.add_argument("file"); hm.add_argument("--names", help="comma list of labels in storing order")
    st = sp.add_parser("stop", help="stop the running measurement chain properly: parent scripts, harness, all children; then wait for free GPU memory and restore the power limits")
    st.add_argument("tasks", nargs="*", type=int, help="only these tasks, by the numbers --dry lists (1 = started first); none = all")
    st.add_argument("--last", type=int, default=0, metavar="N", help="only the N most recently started tasks")
    st.add_argument("--dry", action="store_true", help="list what would be stopped, change nothing")
    st.add_argument("--grace", type=float, default=10.0, help="seconds between SIGTERM and SIGKILL")
    st.add_argument("--wait-gpu", type=float, default=60.0, help="seconds to wait until the GPUs have no memory in use")
    st.add_argument("--no-power", action="store_true", help="leave the power limits alone")
    qq = sp.add_parser("queue", help="the bench queue (queued.service): add TASK_ID --cli '...' [--order N] [--note ..]  |  add-json (a JSON list on stdin)  |  backups  |  pause (no new task starts; the running one finishes) / resume / status; edits take the queue's own file lock and keep the last 3 versions of tasks.csv")
    qq.add_argument("action", choices=["add", "add-json", "del", "backups", "pause", "resume", "status"])
    qq.add_argument("task_id", nargs="?")
    qq.add_argument("--cli", help="the command line the queue runs (bash m111log.sh ... or a run.py line)")
    qq.add_argument("--order", type=float, default=-100, help="run order, ascending (default -100 = ahead of the waiting tasks)")
    qq.add_argument("--note", default="")
    ce = sp.add_parser("cell", help="one queue cell: wait for free GPUs, bench BUILDS (comma separated, side by side) on MODEL for one TEST, results to q5h/TAG.out (replaces m111.sh)")
    ce.add_argument("tag"); ce.add_argument("model"); ce.add_argument("builds"); ce.add_argument("n", type=int); ce.add_argument("test")
    ce.add_argument("args", nargs="?", default="", help="server args after -np 1")
    ce.add_argument("--wrap", help="command prefix for the server (nsys / ncu)")
    ce.add_argument("--keep-server", action="store_true", help="bench --keep-server")
    ce.add_argument("--clean-state", action="store_true", help="bench --clean-state: an empty learned state for this build and cell; run 1 cold, run 2 warm")
    ce.add_argument("--keep-loops", action="store_true", help="do not discard and repeat a looping or garbled answer (bench --keep-loops): one server run per cell")
    ce.add_argument("--warm", action="store_true", help="a discarded run before each measured one: the first run after a server start is cold (nothing learned yet, the planner still guesses) and a gate does not compare it")
    ce.add_argument("--hog-gpu", help="GiB left free on every GPU (the server takes the rest at start: --moe sim-vram=MiB), simulates a smaller card")
    ce.add_argument("--hog-ram", help="GiB of RAM left free (--moe sim-ram=MiB)")
    ce.add_argument("--env", action="append", help="KEY=VALUE for the bench run (e.g. PERF_PPL_CHUNKS=12), repeatable")
    ce.add_argument("--nothink", action="store_const", const=True, help="chat tests: reasoning off (enable_thinking=false)")
    ce.add_argument("--lead", type=int, help="a request of about N prompt tokens first, same server")
    ce.add_argument("--prompt", help="prompt text or @file instead of the test's own")
    ce.add_argument("--tokens", type=int, help="generated tokens instead of the test's own")
    ce.add_argument("--ctx", type=int, help="context size instead of the test's own (a -c in the args loses to the test's own -c)")
    ce.add_argument("--chat", metavar="INSTR", help="pf tests through the chat template with this instruction in front (instruct models)")
    ce.add_argument("--clamp", type=lambda v: int(float(v[:-1]) * 1000) if v.lower().endswith("k") else int(v), help="cut the prompt to N[k] tokens (pf128k clamped to 48k = a 48k prompt)")
    br = sp.add_parser("branches", help="local branches with commits not in TARGET (counts, dates); --merge a,b,c merges them in the checkout of TARGET (refuses on another branch or a dirty tree)")
    br.add_argument("target", nargs="?", default="prerelease2")
    br.add_argument("--repo", default=os.getcwd(), help="the checkout to look at / merge in")
    br.add_argument("--merge", help="comma-separated branch names")
    br.add_argument("--limit", type=int, default=60)
    s_ = sp.add_parser("show")
    s_.add_argument("-t", "--test")
    s_.add_argument("-b", "--build")
    s_.add_argument("--runs", action="store_true")
    cl = sp.add_parser("cells", help="markdown table of the cells of a campaign prefix, with the context and the pool the server really ran with")
    cl.add_argument("prefix")
    mf = sp.add_parser("misfires", help="automatic choices that went against what was measured: a prompt run on one card where the split of the same model and pass "
                                         "size was faster in another run, and the reverse; stock re-measurements of the placement (from the pf_* / placement columns)")
    mf.add_argument("--days", type=float, default=7.0, help="look at runs of the last N days")
    mf.add_argument("--tol", type=float, default=0.05, help="a choice counts as wrong when the other mode was faster by more than this share")
    mf.add_argument("--model", help="part of the model name")
    sx = sp.add_parser("status", help="the running list in the -nrt table: done today, running now, queued next (with the expected time), the queue daemon's waiting tasks; nothing is started")
    sx.add_argument("list", nargs="?", help="a list name (fork/runs/NAME.csv); none = the list that runs now, else the last one")
    sx.add_argument("rest", nargs=argparse.REMAINDER, help="run options of the list (-b BUILD ...)")
    op = sp.add_parser("onepass", help="the best one-pass input prompt of every model from the runs of `run onepass -b BUILD` (pass count and prompt speed per size); --write stores it in models.ini")
    op.add_argument("-b", "--build", default="latest"); op.add_argument("--write", action="store_true")
    cb = sp.add_parser("calib", help="find every model's biggest one-pass input prompt by halving from 12k, then bisecting and store best_prompt.<N>gpu in models.ini")
    cb.add_argument("-b", "--build", default="latest"); cb.add_argument("-ng", "--gpus", dest="n_gpus", type=int); cb.add_argument("-m", "--models", help="comma list of model keys of runs/onepass.csv (default all)")
    cb.add_argument("--dry", action="store_true"); cb.add_argument("--nowrite", action="store_true"); cb.add_argument("--ceil", action="store_true", help="only read the engine's one-pass ceiling (one 2k probe) and store it as maxpp")
    cb.add_argument("-nm", dest="no_pool", nargs="?", const="0", default=None, metavar="GB", help="the probes pin at most GB of the model (as run -nm): the ceiling depends on the RAM beside the pinned weights")
    cb.add_argument("-a", dest="cell_args", default="", metavar="ARGS", help="extra cell arguments for the probes (as run -a)")
    ns = sp.add_parser("nsys", help="per card: kernel time, compute-stream waits, copies and hand-offs of the prompt pass in an nsys report (cell --wrap 'nsys profile ...')")
    ns.add_argument("rep", help="the .nsys-rep file")
    ns.add_argument("--top", type=int, default=8, help="kernels listed per card")
    ns.add_argument("--gaps", type=int, default=0, help="list the N longest compute-stream waits per card with the kernels around them")
    if len(sys.argv) > 1 and (sys.argv[1] in ("-h", "--help") or sys.argv[1] == "run" and len(sys.argv) > 2 and sys.argv[2] in ("-h", "--help")): run_list_help()
    if len(sys.argv) == 2 and sys.argv[1] == "run": sys.exit(run_list_detached(None, []))   # `jobs` alone: attach to the detached run, if any
    if len(sys.argv) > 3 and sys.argv[1] == "run" and "-c" in sys.argv[2:-1]:   # `run -c gainers ...` = `run gainers ...`: -c names the campaign list
        i = sys.argv.index("-c", 2); sys.argv[2:i + 2] = [sys.argv[i + 1]] + sys.argv[2:i]
    if len(sys.argv) > 2 and sys.argv[1] == "run" and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", sys.argv[2] + ".csv")):
        sys.exit(run_list_detached(sys.argv[2], sys.argv[3:]))   # `run gainers [--build B] [--only TEXT] [--dry]`: the cells of fork/runs/gainers.csv
    a = ap.parse_args()
    {"run": cmd_run, "publish": cmd_publish, "numa": cmd_numa, "devpub": cmd_devpub, "bench": cmd_bench, "report": cmd_report, "gains": lambda a: [gains_write(c) for c in a.campaigns], "machine": cmd_machine, "remote": cmd_remote, "rcell": cmd_rcell, "prompt": cmd_prompt, "ctl": cmd_ctl,
     "show": cmd_show, "regress": cmd_regress, "build": cmd_build, "quick": cmd_quick, "hugepool": cmd_hugepool, "stop": cmd_stop, "hotmapmat": cmd_hotmapmat, "hitlog": cmd_hitlog, "queue": cmd_queue, "cell": cmd_cell, "branches": cmd_branches, "misfires": cmd_misfires, "cells": cmd_cells, "nsys": cmd_nsys, "status": cmd_status, "onepass": cmd_onepass, "calib": cmd_calib,
     "exp": lambda a: runexp.run_exp(a.spec, sys.modules[__name__], a.dry, a.only.split(",") if a.only else None, a.rounds)}[a.cmd](a)
