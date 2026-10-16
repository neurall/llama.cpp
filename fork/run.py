#!/usr/bin/env python3
"""Benchmark harness of the llama.cpp expert-cache fork: every run is one line of run-history.csv (append-only, key columns first).

  run.py bench  --models M1,M2 --builds autotune,stock-X [--variants default,cache0,atoff] [-t t100] [-n 2] [--machine pc1]
  run.py report [--campaign NAME] [--last N] [--tol 0.95]      variants against stock, per model / test / machine
  run.py run    [-t TEST] [-n N] [-e K=V ...] [--bare] [--plain] [--args '...'] BUILD...   (MODEL=/path/model.gguf)
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
import argparse, csv, glob, hashlib, io, json, os, re, shlex, shutil, signal, statistics, subprocess, sys, threading, time, urllib.request

import types


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
  [hosts]   w10_ssh  w10_root                         (the Windows laptop used by `run.py build --machine pc3`; the ssh target may be an alias from ~/.ssh/config)
  [gpu]     nvidia_smi  limits_env  limits_service  env_sh   (power-limit tools run.py plcal / run.py baseline: nvidia-smi to call through sudo, the files that keep the limits at boot, an
                                                              optional shell file whose exports every queue task gets; empty = not used)
"""
    import os

    HOME = os.path.expanduser("~")
    _BASE = os.path.join(HOME, "llama-fork")
    DEFAULTS = {
        "paths": {
            "rels": _BASE + "/rels", "gh_rel": _BASE + "/stock", "devbuilds": _BASE + "/rels/devbuilds", "release_out": _BASE + "/rels/release-out",
            "images": _BASE + "/rels/images", "best": _BASE + "/rels/best", "ccache": _BASE + "/ccache", "ccache_stock": _BASE + "/ccache-stock",
            "build_repo": os.getcwd(), "build_tmp": _BASE + "/build-tmp", "work": _BASE + "/work", "agent": "",
        },
        "models": {
            "olmoe": HOME + "/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
            "qwen36": HOME + "/models/Qwen3.6-35B-A3B-GSQ-hybrid.gguf",
            "qwen38": HOME + "/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
            "glm": HOME + "/models/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf",
            "mimo": HOME + "/models/MiMo-V2.6-Flash-MOPD-MXFP4.gguf",
        },
        "hosts": {"w10_ssh": "", "w10_root": "N:"},
        "gpu": {"nvidia_smi": "nvidia-smi", "limits_env": "", "limits_service": "", "env_sh": ""},
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


    def model(key):
        return os.path.expanduser(get("models", key))

    return {k: v for k, v in dict(locals()).items() if not (k.startswith('__') and k != '__doc__')}

config = _register('config', _lib_config())


def _lib_runnorm():
    # ---- the normalisation section of run.py (merged into run.py; the code of the file, unchanged)
    __doc__ = """the normalisation section of run.py: the normalisation shared by run.py runlog-all (full rebuild of all_runs) and the statistics section of run.py (incremental refresh): model family, build number, test group."""
    import os, re

    FAMS = [("GLM-5.3-Flash-GSQ-RCO-3.5bit", "GLM 3.5-bit"), ("GLM-5.3-Flash-GSQ-RCO-3.0bit", "GLM 3.0-bit"), ("MiMo", "MiMo IQ3_XXS"), ("Qwen3.8-Flash-Next-UD-IQ4_XS", "Next IQ4_XS"),
            ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "Next IQ3_S"), ("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", "Next IQ1_M"), ("Qwen3.6-35B-A3B-GSQ", "Qwen3.6 Q2_0"), ("qwen36", "Qwen3.6 Q2_0"),
            ("Qwen3.8-27B-IQ4_NL", "27B IQ4_NL"), ("Qwen3.8-27B-GSQ-RCO-IQ3_S", "27B IQ3_S"), ("Qwen3.8-27B-MTP", "27B Q5_K_M MTP"), ("Qwen3.8-27B", "27B"), ("OLMoE", "OLMoE")]


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


    GAME_WORDS = ["tetris", "racing", "shooter", "snake"]


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
bin = /p/bw/wt-release-next/build-dev/bin
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
    HOLD = os.path.join(os.path.dirname(os.path.realpath(__file__)), "experiments", "vram-sim", "vramhold.py")


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
        holder = None
        if setup.get("hold_vram") and not dry:
            holder = subprocess.Popen([sys.executable, HOLD] + [f"{g}:{gib}" for g, gib in setup["hold_vram"].items()], env=base_env,
                                      stdout=subprocess.PIPE, text=True, start_new_session=True)
            held_ok = False
            for _ in range(120):
                if "holding" in (holder.stdout.readline() or ""):
                    held_ok = True
                    break
                if holder.poll() is not None:
                    break
            if not held_ok or holder.poll() is not None:
                sys.exit(f"hold_vram failed: {HOLD} did not start holding (missing or crashed); rows would claim a constraint that never existed")
            print(f"holding VRAM {setup['hold_vram']}", flush=True)
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
            if holder:
                try:
                    os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
                except Exception:
                    pass
        print(f"{name}: {n_run} runs", flush=True)


    def command_for(wl, setup, arm, mpath):
        tool = arm.get("tool", wl.get("tool", "cli"))
        exe = os.path.join(arm["bin"], TOOLS[tool])
        a = ["-m", mpath]
        here = os.path.dirname(os.path.realpath(__file__))
        resolve = lambda p: p if os.path.isabs(p) else os.path.join(os.path.dirname(here), p)
        if tool == "perplexity":
            a += ["-f", resolve(wl.get("text_file", "fork/bench/longsrc.cpp")), "-c", str(wl.get("ctx", 512)), "--chunks", str(wl.get("chunks", 2))]
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
  python3 fork/run.py runlog sync                     add the CSV rows the database lacks (after a git merge of run-history.csv from another machine)
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
        """append the CSV rows the database lacks (runs that came in by a git merge from another machine); never deletes anything"""
        con = connect()
        cols, rows = read_csv_rows()
        ensure(con, cols)
        def key(ts, build, test, tps):
            t = num(tps)
            return (str(ts), str(build or ""), str(test or ""), None if t is None else round(t, 3))
        have = {key(*r) for r in con.execute("SELECT ts, build, test, tps FROM runs")}
        n = 0
        for r in rows:
            if key(r.get("ts"), r.get("build"), r.get("test"), r.get("tps")) in have: continue
            con.execute(ins(cols), row_values(r, cols)); n += 1
        con.commit(); print(f"{n} CSV rows added to {DB} ({len(rows)} in the CSV)")
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

PROMPTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bench")   # frozen prompts (src_12k.cpp, src_128k.cpp, longsrc.cpp) live in fork/bench/
HERE = os.path.abspath(os.environ.get("PERF_BUILDS", os.getcwd()))  # one subdirectory per build (its bin/ contents)
DATA = os.environ.get("RUN_DATA") or PROMPTS   # directory of run-history.csv, campaigns.csv, machines.csv, tune.csv (default: fork/bench, committed)
MODEL = os.environ.get("MODEL", "")
MODELS_DIR = os.environ.get("PERF_MODELS_DIR", "")  # other models' pages are dropped from RAM before a run
PPL_TEXT = os.path.join(PROMPTS, "longsrc.cpp")  # frozen snapshot of llama.cpp common/json-schema-to-grammar.cpp (code, like agent prompts)
BARE = False
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
LONG2K_INSTR = "There is a bug in the source code above. Find it, explain what goes wrong, and give the corrected code for the affected functions with a short test that would have caught it."
EDIT_INSTR = ["Add a function that counts the lines of this file and prints the count.",
              "Delete the code in this file that is never used.",
              "Add a short comment above every function in this file.",
              "Rename the longest function in this file to a clearer name and update its callers.",
              "Fix the bugs in this code."]   # the fifth is run only with --rep-from 4 (README runs pick one of the first four at random: repetition 0 to 3)
EDIT_WARM = "Explain in two sentences what this file does."
# test game4: the short CLI test with four subjects, "write smallest html <game> game": same task, same language and stack, only the game changes (a coder stays
# on one goal and one stack, so the experts stay relevant, but no two runs are the same prompt); same game at the same repetition for every build, own warm-up game
GAME_WORDS = ["tetris", "racing", "shooter", "snake"]
GAME_WARM = "pong"
CHAT_LONG = ("write smallest html tetris game. Requirements: a single self-contained html file with inline css and javascript, no external libraries or assets. "
             "The board is 10 columns by 20 rows drawn on a canvas, with a one-cell border, and every cell is 24 pixels square. All seven tetrominoes (I, O, T, S, Z, J, L) "
             "have distinct colors and spawn from a shuffled bag so every piece appears once before any repeats. The arrow keys move the active piece left and right and "
             "soft-drop it, the up arrow rotates it clockwise with simple wall kicks, and the space bar hard-drops it. Gravity speeds up every ten cleared lines. Show a ghost "
             "piece where the active piece would land, a preview of the next piece, the score, the level and the number of cleared lines. Scoring is 100, 300, 500 and 800 points "
             "for one, two, three and four lines at once, multiplied by the level, plus one point per soft-dropped cell and two per hard-dropped cell. Pressing P pauses and "
             "resumes the game, and pressing R restarts it. When a new piece cannot spawn, show a game over overlay with the final score and a hint to press R. Keep the code "
             "readable: one function for collision tests, one for locking a piece, one for clearing lines, one for drawing, and a requestAnimationFrame loop with a fixed time "
             "step. Do not use any frameworks. Answer with the complete html file only, in a single code block, without any explanation before or after it.")
TAGBOX = {"v": ""}   # the running test and prompt (game4-racing, edit4-2, long1k, ...): --moe tag= puts it into the routing file name
IDX = 0    # run counter: test chatv answers prompt IDX mod len(CHAT_PROMPTS) (a repeated temp-0 answer would reuse exactly the cached experts)
CHAT_PROMPTS = [
    "Write a Python function that parses a CSV file and returns the average of each column.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a CPU cache hierarchy works and why cache misses are expensive.",
    "Write a SQL query that finds the top 5 customers by total order value, with the schema.",
    "Translate into German and French: The meeting is moved to Thursday because the room is booked.",
    "Solve step by step: a train leaves at 9:40 at 84 km/h, a second at 10:10 at 105 km/h. When does it catch up?",
    "Write a bash script that backs up a directory to a dated tar.gz and keeps only the last 7 backups.",
    "Summarize the causes and consequences of the French Revolution.",
    "Write a Rust function that reverses the words in a string without allocating per word.",
    "Give a recipe for a vegetarian lasagna with a shopping list.",
    "Write a JavaScript debounce function and explain when to use it instead of throttle.",
    "Explain the difference between TCP and UDP with examples of when to use each.",
]


def ovr_prompt(default):
    p = OVR.get("prompt")
    if p and p.startswith("@"):
        return open(p[1:]).read()
    return p or default
GREEDY = {"temperature": 0, "top_p": 1, "top_k": 1}
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
RUN_COLS += ["up_ms_expert", "up_gbs_eff"]   # from the engine log: mean upload time of one expert (all three tensors) over the second half of the run, and the effective speed of one upload = MiB per expert / that time
RUN_COLS += ["pcie_h2d_gb", "pcie_h2d_peak_gbs", "pcie_h2d_avg_gbs", "pcie_d2h_gb"]   # nvidia-smi dmon over the server run: GB host->device during the request (all GPUs; the server is already loaded), the peak of any GPU in a second, the mean of the seconds that moved data, GB device->host
RUN_COLS += ["ddr_util", "up_overlap_pct", "cpu_slow_pct", "lead_end0", "lead_end1"]   # from cache.csv (admit-pred builds): mean RAM traffic share of DDR_GBS, mean share of uploads that ran while the CPU read experts, mean slowdown of CPU expert phases with an upload in flight, the first predicted layer ahead per link at the end
RUN_COLS += ["ev_mean_hits_fast", "ev_mean_hits_slow", "ev_zero_pct_fast", "ev_zero_pct_slow", "ev_ge4_pct_fast", "ev_ge4_pct_slow", "life_fast", "life_slow"]   # hits an uploaded expert actually got before its eviction, per link (cache.csv histogram 0/1/2/3/4-5/6-10/11+; mean with bucket midpoints), the share with no hit and with 4 or more, and the mean steps it stayed cached
RUN_COLS += ["t_full_pw"]   # time to full power, for every server start (a bench run restarts the server): seconds from the start of the GPU log (the server start) until each GPU first draws >= 90% of its own peak (1 s average), one value per GPU in index order; the load phase is the time the cards are not at full power
RUN_COLS += ["pw_avg", "pw_max", "pw_inst_max", "energy_wh", "cap_pct"]   # power of the request window, one value per GPU in index order (nvidia-smi, --gpulog): mean W, peak W (1 s average), peak W (instantaneous), energy of all GPUs in Wh, % of samples at the software power cap
RUN_COLS += ["vram_mib"]   # GPU memory in use at the end of the run, all GPUs together (nvidia-smi memory.used)
RUN_COLS += ["endpoint"]
RUN_COLS += ["knobs_end"]   # what the build actually ran with, from the run.knobs line of the end-of-run state snapshot: key settings (slots, workers, upload path, bounce, policy) | every cache knob incl. tuner decisions (defaults differ between builds; * = set by the user)   # completion (raw prompt), chat (template) or cli (llama-cli)
RUN_COLS += ["load_mode"]   # how the weights were loaded, from the server log: pinned (+ upload path), mmap, mmap after a failed pin ("reloading with mmap"); load time (load_s) tells too: pinned GLM 3.0 about 85-120 s, mmap about 25 s
RUN_COLS += ["start_ts", "end_ts", "load_s", "wall_s"]   # start_ts / end_ts: when the process started and ended (local time); load_s: seconds until the model was loaded and ready (server: ready to answer; llama-cli / perplexity: the "load time" it prints); wall_s: seconds the whole run took, start to end
RUN_COLS += ["uniq4", "out_chars", "code", "loop"]   # text class of the generated output: share of unique word 4-grams (a repetition loop is ~0.1, real text ~0.9+), characters, 1 if it contains code; the text itself is in runs.db table outputs
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


def wait_vram_free():
    kill_leftovers()
    t0 = time.time()
    while True:
        if time.time() - t0 > 60:  # something still holds VRAM: clean up again
            kill_leftovers()
            t0 = time.time()
        out = subprocess.run([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        out = [x for x in out if x.isdigit()]   # no GPU, or an nvidia-smi that fails: nothing to wait for
        if not out or max(int(x) for x in out) < 500 + int(os.environ.get("PERF_HOLD_MB", 0)):  # PERF_HOLD_MB: VRAM a holder process keeps on purpose (simulated smaller card)
            wait_host_free()
            return
        time.sleep(1)


def wait_host_free(limit_s=900):
    """the host side of 'the machine is free': a 100 GB model process needs a while to unmap its memory after it was told to stop, and the next load must not start into
    that (swap, a stalled cudaHostRegister: PIN_STALL, ssh crawling). Waits until no llama process is left (they are not killed here: wait_vram_free already did that
    for the stale ones), no process holds more than PERF_BIG_RSS_GB (default 8) of resident memory, and MemAvailable is at least PERF_MIN_AVAIL_GB (default 8).
    The hugepage pool is reserved separately and persists on purpose, it is not counted. Gives up (with a message, the run goes on) after limit_s."""
    big, need = int(os.environ.get("PERF_BIG_RSS_GB", 8)), int(os.environ.get("PERF_MIN_AVAIL_GB", 8))
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
                 "pf_one_tps", "pf_split_tps", "pf_by", "pf_all"]


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


def _server_start_once(build, env, args):
    if not build.startswith("stock") and build_has_moe_flag(build, env):   # the fork's research logs and a cold tuner, as --moe settings (older fork builds that do not know a key ignore it; the first releases have no --moe at all)
        args = list(args) + ["--moe", f"log={LOG_LETTERS},logdir={DATA}" + (f",tag={TAGBOX['v']}" if TAGBOX["v"] else "") + (f",state={STATE_FILE['v']}" if STATE_FILE["v"] else "")]
    elif not build.startswith("stock") and STATE_FILE["v"]:
        env = {**env, "LLAMA_MOE_STATE": STATE_FILE["v"]}   # the first releases (no --moe flag) read the environment variable
    env, args = drop_moe_flag(build, env, args)
    args = merge_moe(args)
    logf = f"/tmp/perf-{build}{LOGTAG}.log"
    with open(logf, "w") as lf:
        p = subprocess.Popen(WRAP + [os.path.join(HERE, build, "llama-server"), "-m", MODEL, "--port", str(PORT)] + ([] if BARE else ["-np", "1"])
                             + args, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=_die_with_parent)
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


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def post_loopguard(path, body, timeout=3600, every=128, min_tokens=400, window=150):
    """/completion as a stream: every `every` tokens (from `min_tokens` on) the last `window` words are tested; a repetition loop (unique word 4-gram share < 0.5) closes the connection (the server stops
    generating) and the result is marked _loop: a loop costs about a minute instead of the whole run. Returns the same fields as the plain post (content, timings)."""
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps({**body, "stream": True}).encode(), {"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    text, n, last = "", 0, {}
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:]); last = j
        text += j.get("content", ""); n += 1
        if j.get("stop"):
            break
        if n >= min_tokens and n % every == 0:
            w = text.split()[-window:]
            g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
            if len(w) >= window and g and len(set(g)) / len(g) < 0.5:
                resp.close()
                return {"content": text, "_loop": True, "timings": {"predicted_per_second": 0, "prompt_per_second": 0, "predicted_n": n, "prompt_n": 0, "prompt_ms": 0}}
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
                win[g] = (j / max(1e-6, ts[-1] - ts[0]), max(pw), max(ins) if ins else 0.0, j / 3600.0, 100.0 * cap / len(r))
            if win:
                PROCSTAT.update(pw_avg="/".join("%.0f" % v[0] for v in win.values()), pw_max="/".join("%.0f" % v[1] for v in win.values()),
                                pw_inst_max="/".join("%.0f" % v[2] for v in win.values()), energy_wh="%.3f" % sum(v[3] for v in win.values()),
                                cap_pct="/".join("%.0f" % v[4] for v in win.values()))
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
    p, logf = server_start(build, env, args)
    PROCSTAT["load_s"] = round(time.time() - _t0, 1)   # server_start returns when the server answers
    gstop = gpu_sampler(build, p.pid) if GPULOG else (lambda *a: ({}, ""))
    t_req = [None, None]   # the request window: the power columns cover it only
    perf = None
    if PERFPROF:
        pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "perf"); os.makedirs(pdir, exist_ok=True)
        pbase = os.path.join(pdir, time.strftime("%Y%m%d-%H%M%S") + "-" + build)
        perf = (subprocess.Popen(["perf", "record", "-F", "499", "-o", pbase + ".data", "-p", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), pbase)
    bwstop = pcie_sampler() if GPULOG else (lambda: None)   # the engine's own byte and time counters are the bandwidth record; the nvidia-smi view only with --gpulog
    try:
        if render:   # the server renders the model's chat template into the prompt text
            body = {**body, "prompt": post("/apply-template", {"messages": render})["prompt"]}
        if OVR.get("clamp") and isinstance(body.get("prompt"), str):   # --clamp N: the prompt is cut to N tokens (the server counts them, so it is right for every tokenizer)
            txt, n_cl = body["prompt"], int(OVR["clamp"])
            for _ in range(4):
                n_tok = len(post("/tokenize", {"content": txt})["tokens"])
                if n_tok <= n_cl:
                    break
                txt = txt[:int(len(txt) * n_cl / n_tok * 0.995)]
            body = {**body, "prompt": txt}
            print(f"clamp: prompt cut to {n_tok} tokens (limit {n_cl})", flush=True)
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
        for i_rep in range(rep):
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
        stop(p)
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
    wait_vram_free()
    env = {**os.environ, **extra_env, "LD_LIBRARY_PATH": ldpath(build), "LLAMA_MOE_CACHE_STATS": "1",
           "LLAMA_MOE_CACHE_STATSLOG": "32", "LLAMA_MOE_STATSLOG": f"/tmp/perf-{build}{LOGTAG}-stats.txt",  # builds with statslog write a text line per 32 steps (moe-stats.csv)
           "LLAMA_MOE_CACHE_TUNED": os.environ.get("LLAMA_MOE_CACHE_TUNED", "0")}  # no saved tuner decisions: every run starts cold
    row = {}
    TAGBOX["v"] = test   # game4 and edit4 add their prompt below
    if test in ("ppl", "ppl3", "pplb", "pplc"):
        nb, nub = {"ppl": ("1", "1"), "ppl3": ("3", "3"), "pplb": ("1024", "1024"), "pplc": ("1024", "256")}[test]   # pplb: 1024-token batches = the big-batch (prefill) path; pplc: the same batch in 256-token ubatches (4 ubatches per decode call)
        args = COMMON + ["-f", PPL_TEXT, "-c", "1024", "--chunks", os.environ.get("PERF_PPL_CHUNKS", "2"), "-b", nb, "-ub", nub]   # PERF_PPL_CHUNKS=12: a paired comparison of two compute paths needs more than 2 chunks (a valid path differs by +-0.5% per chunk)
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
        row.update(start_ts=_start, end_ts=time.strftime("%F %T", time.localtime(_t0 + _wall)), wall_s=_wall, **({"load_s": round(float(_ld[-1]) / 1000, 1)} if _ld else {}))
        if not ppl:
            row["note"] = "ppl failed, exit %s: %s" % (r.returncode, " | ".join(l.strip() for l in log.strip().splitlines()[-3:])[:300])
    else:
        if BARE:
            COMMON = list(extra_args)
        _pfn = re.fullmatch(r"pf(\d+)k", test)   # pf4k, pf6k, pf8k ...: the first N thousand tokens of the frozen 12k source (one pass when -ub >= N thousand)
        if test in ("pf12k", "pf12k-stock", "pf128k", "fix12k") or (_pfn and int(_pfn.group(1)) not in (12, 128)):
            pf_ctx = str(OVR.get("ctx") or (131072 if test == "pf128k" else (int(_pfn.group(1)) + 2) * 1024 if _pfn and int(_pfn.group(1)) > 14 else 16384))
            dev = (["-t", THREADS] if THREADS else []) + ["-dev", os.environ.get("PERF_PF_DEV") or ("CUDA1,CUDA0" if DEVS == "CUDA0,CUDA1" else DEVS), "-c", pf_ctx, "-ub", "2048", "-b", "2048"]
            args = list(extra_args) if BARE else dev + ((["--moe-expert-cache", "0"] if test == "pf12k-stock" and not build.startswith("stock") else []) if test == "pf12k-stock" or plain or build.startswith("stock") else ["--cpu-moe", "-nr", "--moe-expert-cache", "-1"]) + list(extra_args)
            if BARE and (plain or build.startswith("stock")) and "-c" not in args:
                args += ["-c", pf_ctx]  # stock's default context (4096) rejects the 12k prompt (HTTP 400); the fork sets 32k itself
            _pf_big = test == "pf128k" or (_pfn and int(_pfn.group(1)) > 12)   # pf40k, pf64k ...: the first N thousand tokens of the 128k source
            _pf_text = open(os.path.join(PROMPTS, "src_128k.cpp" if _pf_big else "src_12k.cpp")).read()
            if _pfn and int(_pfn.group(1)) not in (12, 128):   # the 12k file is 12294 tokens (GLM tokenizer, 4.4 characters each): cut in proportion
                _pf_text = _pf_text[:int(54092 * int(_pfn.group(1)) * 1000 / 12294)]
            _pf_text = OVR.get("prefix", "") + _pf_text
            if OVR.get("prefix"):   # an instruction (12kin_1kout_code_review): an instruct model needs its chat template or a long answer loops, so it goes through the chat endpoint
                res, logf = server_run(build, env, args, "/v1/chat/completions",
                                       {"messages": [{"role": "user", "content": _pf_text}], "max_tokens": int(OVR.get("tokens") or 1024), **GREEDY, "cache_prompt": False})
            else:
              res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": ovr_prompt(_pf_text), "n_predict": int(OVR.get("tokens") or os.environ.get("PERF_NPRED", 64 if test == "pf128k" else 256 if test == "fix12k" else 32)),
                                    "cache_prompt": False, **GREEDY, **(FIXED if test == "fix12k" else {})})
            t = res["timings"]
            row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
                   "note": f"prompt_n={t['prompt_n']} prompt_s={t['prompt_ms']/1e3:.1f}" + (f" first_pp={PROCSTAT['rep_first_pp']:.1f}" if PROCSTAT.get("rep_first_pp") else "")}
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
            args = COMMON + ["-c", str(OVR.get("ctx") or (2048 if base == "fix" else 1024))]
            res, logf, text = generate(build, env, args, ovr_prompt("generate smallest html tetris game." if base == "tetris" else "write smallest html tetris game"),
                                       int(OVR.get("tokens") or (-1 if base == "tetris" else 1024 if base == "fix" else 100)), chat, FIXED if base == "fix" else None)
        elif base == "game4":
            word = GAME_WARM if WARMING else GAME_WORDS[(OVR.get('pidx', REP or 0)) % len(GAME_WORDS)]
            TAGBOX["v"] = "game4-" + word.replace(" ", "_") + ("-chat" if chat else "")
            args = COMMON + ["-c", str(OVR.get("ctx") or 1024)]
            res, logf, text = generate(build, env, args, ovr_prompt(f"write smallest html {word} game"), int(OVR.get("tokens") or 100), chat, {"cache_prompt": False})
        elif base == "long1k":   # a prompt of about 1024 tokens (src_1k.cpp = the first 4.4k characters of src_3k.cpp) asking for a bug fix, 1024 tokens out, -c 4096: a long prefill and a long decode
            snippet = open(os.path.join(PROMPTS, "src_1k.cpp")).read()
            args = COMMON + ["-c", str(OVR.get("ctx") or 4096)]
            res, logf, text = generate(build, env, args, snippet + "\n\n" + LONG2K_INSTR, int(OVR.get("tokens") or 1024), chat, {"cache_prompt": False})
        elif base == "edit4":
            snippet = open(os.path.join(PROMPTS, OVR.get("snippet") or "src_3k.cpp")).read()
            ins = EDIT_WARM if WARMING else EDIT_INSTR[(OVR.get('pidx', REP or 0)) % len(EDIT_INSTR)]
            TAGBOX["v"] = "edit4-%d" % ((REP or 0) % len(EDIT_INSTR)) + ("-chat" if chat else "")
            args = COMMON + ["-c", str(OVR.get("ctx") or 4096)]
            res, logf, text = generate(build, env, args, snippet + "\n\n" + ins, int(OVR.get("tokens") or 128), chat, {"cache_prompt": False})
        else:
            args = COMMON + ["-c", str(OVR.get("ctx") or 4096)]
            chatv = test == "chatv"  # varied prompts (index = run counter), thinking off: the topic switches the cache has to follow
            res, logf = server_run(build, env, args, "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": ovr_prompt(CHAT_PROMPTS[OVR.get("pidx", IDX) % len(CHAT_PROMPTS)] if chatv else CHAT_LONG if test == "chat2" else "write smallest html tetris game")}],
                                    "max_tokens": int(OVR.get("tokens") or (300 if chatv else 1500)), **GREEDY,
                                    **({"chat_template_kwargs": {"enable_thinking": False}} if chatv else {})})
            m = res["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
        t = res["timings"]
        log = open(logf).read()
        row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
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


def save_stats(build, ts, test, hw, model):
    """The server's statslog file of the measured run (a value per 32 steps: t/s, hit, uploads, evictions, MiB uploaded, DDR and PCIe GB/s) -> a section of stats.ini."""
    f = f"/tmp/perf-{build}{LOGTAG}-stats.txt"
    if os.path.exists(f):
        w = [[float(x) for x in l.split()] for l in open(f) if l[:1].isdigit() and len(l.split()) in (10, 14, 18)]
        os.remove(f)  # the next run of this build must not re-save it
    else:
        w = windows_from_cache_csv(build)   # release-line builds have no statslog file: the same windows from the engine's --moe log=p series (one row per N decode tokens)
    if w:
        with open(os.path.join(DATA, "stats.ini"), "a") as out:
            sec = stats_section(f"{ts} {build}", model, test, hw, w)
            out.write("\n".join(l for l in sec.split("\n") if not re.fullmatch(r"ddr_gbs=(0\.00,?)+", l)))
    return w


CELLS = {}   # (model, test, build, args, campaign, hw) -> (t/s of its runs, pp of its runs)


def text_class(text):
    """(unique word 4-gram share, characters, has code): the speed of an MoE decode depends on what is written (a repetition loop reuses few experts)"""
    w = (text or "").split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    low = (text or "").lower()
    return (round(len(set(g)) / len(g), 3) if g else None), len(text or ""), int("```" in low or "<html" in low or "function " in low or "#include" in low)


def is_loop(text):
    """a repetition loop: the last 30% of the words have a unique word 4-gram share below 0.5 (measured: loops 0.01-0.26, real text 0.83-1.0); needs at least 100 words"""
    w = (text or "").split()
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
    w = save_stats(build, row["ts"], test, hw, row["model"])
    row["uniq4"], row["out_chars"], row["code"] = text_class(row.get("output"))
    row["loop"] = int(bool(row.get("loop_abort")) or is_loop(row.get("output")))
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
    if row.get("ok", 1) and row.get("tps"):
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
        pl=power_limits(hw), proto="v2" if test.startswith("fix") else "v1", rep=REP, seed=SEED if test.startswith("fix") else None,
        branch=bi.get("branch", ""), patch=bi.get("patch", ""), **{k: row.get(k) for k in tuple(PRESSURE_COLS) + ("pm_queue", "pm_copy", "pm_d1", "pm_d2", "pm_d3", "pm_d4p", "pred_n", "pred_mib", "swap_mib", "pred_busy_ms", "swap_busy_ms", "admit_end", "up_ms_fast", "up_ms_slow", "up_gbs_fast", "up_gbs_slow", "link_gbs_cuda0", "link_gbs_cuda1", "up_fast", "up_slow", "mib_fast", "mib_slow", "busy_fast_ms", "busy_slow_ms", "up_ms_expert", "up_gbs_eff", "pcie_h2d_gb", "pcie_h2d_peak_gbs", "pcie_h2d_avg_gbs", "pcie_d2h_gb", "ddr_util", "up_overlap_pct", "cpu_slow_pct", "lead_end0", "lead_end1", "ev_mean_hits_fast", "ev_mean_hits_slow", "ev_zero_pct_fast", "ev_zero_pct_slow", "ev_ge4_pct_fast", "ev_ge4_pct_slow", "life_fast", "life_slow", "vram_mib", "endpoint", "start_ts", "end_ts", "load_s", "wall_s", "uniq4", "out_chars", "code", "loop", "major_faults", "rss_kb", "cache_state", "gpu_log", "pw_avg", "pw_max", "pw_inst_max", "energy_wh", "cap_pct", "knobs_end", "load_mode", "uploads", "evictions", "up_mib", "pred_hit", "pred_hit_without", "pred_boost_pts", "pred_up", "pred_pub", "pred_used", "pred_late", "pred_stale", "pred_cov1", "pred_cov2", "pred_cov3")}, **platform(hw)))
    global _EMB_OFF
    try:
        import runlog
        _EMB_OFF = runlog.insert_emb(_EMB_OFF, row["ts"], hw, build, test)   # the embhot.csv lines of this run into runs.db
    except Exception as ex:
        print("embhot:", ex, file=sys.stderr)


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


def run_cell(build, test, extra, plain, args, bare, note, hw=None, warm=True):
    """One measured run, preceded by a discarded run with the same build and settings: model switches, a previous pinned load
    (drops the page cache) and cache/tuner warm-up all showed 5-12% swings on the first run."""
    global BARE, IDX
    BARE = bare
    xargs = args.split() if args else ()
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
        try:
            run_one(build, test, extra, plain, xargs)
        except Exception as e:
            print(f"throwaway run failed: {e}", flush=True)
        finally:
            WARMING = False
    try:
        row = run_one(build, test, extra, plain, xargs)
    except Exception as e:
        row = {"ok": 0, "note": (str(e)[:80] + " [" + fail_reason(build) + "]" if fail_reason(build) else str(e))[:400]}
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
           **{k: str(v) for k, v in OVR.items() if k != "prompt"}}
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
        print(f"{why}: repeating ({k + 1}/3)", flush=True)
        try:
            row = run_one(build, test, extra, plain, xargs)
        except Exception as e:
            row = {"ok": 0, "note": str(e)[:300]}
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
        for build in a.builds:
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


def model_name():
    return re.split(r"[\\/]", MODEL)[-1]


SAMPLER = ""   # --sampler: extra sampling parameters merged into every request (GREEDY stays the base), recorded in the note of each run


# readable test names <in>_<out>_<what>; they resolve to the tests the history already holds (rows keep the old names, so runs stay comparable)
ALIASES = {"1kout_make_game": "game4c", "1kin_1kout_edit_src": "edit4c", "12kin_1kout_code_review": "pf12k"}
ALIAS_TOKENS = {"1kout_make_game": 1024, "1kin_1kout_edit_src": 1024, "12kin_1kout_code_review": 1024}
ALIAS_PREFIX = {"12kin_1kout_code_review": "Do a code review of the following source file.\n\n"}   # prepended to the 12k prompt (pf12k alone sends the bare file)
ALIAS_SNIPPET = {"1kin_1kout_edit_src": "src_1k.cpp"}   # ~1024 tokens (edit4c alone sends all of src_3k, ~3k tokens)


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
    for k in ("prompt", "tokens", "ctx", "pidx", "repeat", "clamp", "par"):
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
    return {"ssh": r["ssh"], "root": r["root"], "win": (r.get("kind") or "").lower().startswith("win")}


def cmd_machine(a):
    ms = machines()
    if a.action == "add":
        ms.setdefault(a.id, {})
        ms[a.id].update(id=a.id, ssh=a.ssh, root=a.root, kind=a.os)
        with open(os.path.join(DATA, "machines.csv"), "w", newline="") as f:  # tiny file, rewritten whole
            w = csv.DictWriter(f, MACH_COLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(ms.values())
    for r in ms.values():
        print(" | ".join(r.get(k, "") for k in ("id", "ssh", "root", "kind", "cpu", "gpus")))


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
        cmd = envp + envs + f'{exe.replace("llama-cli", "llama-perplexity")} -m {MODEL} -f {m["root"]}{sep}{build}{sep}longsrc.cpp -c 1024 --chunks 2 -b {nb} -ub {nub} {args} ' + ("< nul 2>&1" if win else "< /dev/null 2>&1")
        out = remote_sh(m, cmd)
        sp_ = re.search(r"([\d.]+) seconds per pass", out)
        pp_ = re.findall(r"PPL = ([\d.]+)", out)
        row = {"s_per_pass": float(sp_.group(1)), "tps": 1024 / float(sp_.group(1)), "ppl": float(pp_[-1]) if pp_ else None, "ok": 1, "args": args} if sp_ else {"ok": 0, "note": out[-200:], "args": args}
        row.update(cache_stats(out))
        st = remote_sh(m, ("type " if win else "cat ") + stf + (" 2>nul" if win else " 2>/dev/null"))
        if st.strip():  # the run's statslog goes to stats.ini like a local run's
            open(f"/tmp/perf-{build}{LOGTAG}-stats.txt", "w").write(st)
        return row
    prompt = ovr_prompt("generate smallest html tetris game." if test == "tetris" else "write smallest html tetris game").replace('"', "'")
    n = int(OVR.get("tokens") or (-1 if test == "tetris" else 100))
    pfile = None   # long1k (1024 in, 1024 out) and pf12k (12k in, --tokens 1 out): the prompt goes to the machine as a file
    if test in ("long1k", "pf12k", "in1k"):   # in1k: the long1k prompt (1k in) with ONE token out = prefill only; the decode-only partner is the plain tetris test with --tokens 1024 (short prompt, 1k out)
        text = (open(os.path.join(PROMPTS, "src_1k.cpp")).read() + "\n\n" + LONG2K_INSTR) if test in ("long1k", "in1k") else OVR.get("prefix", "") + open(os.path.join(PROMPTS, "src_12k.cpp")).read()
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
    words = vargs.split()
    args = " ".join(w for w in words if not (w.startswith("env:") or w.startswith("moe:")))
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


def probe_cells(cells, a, model):
    """Before the first measured run of a bench: every (build, variant) cell of this model is started once, one token through the server, with the cell's own env and args, and the
    run.knobs line the build writes at exit is read back. REQUESTED and USED are printed side by side, written to cell-knobs.csv and put in knobs_end of every row of the cell;
    a pin the build does not show is printed as MISMATCH. Builds that write no knob line (b11509 and older) are recorded as such: their settings are their source defaults."""
    global CAMPAIGN, STATE_FILE, LOG_LETTERS
    camp = CAMPAIGN
    saved_letters = LOG_LETTERS
    if "s" not in LOG_LETTERS:
        LOG_LETTERS = LOG_LETTERS + "s"   # server_start passes --moe log=<letters>: without s the build writes no state snapshot and so no run.knobs line
    for build, name, vargs in cells:
        if build.startswith("stock"):
            continue
        args, env = split_variant(build, vargs)
        sf = "/tmp/runstate/probe-" + re.sub("[^A-Za-z0-9._-]", "_", build + "-" + name) + ".ini"
        os.makedirs("/tmp/runstate", exist_ok=True)
        if os.path.exists(sf):
            os.remove(sf)
        STATE_FILE["v"] = sf
        saved = dict(OVR)
        OVR["tokens"] = "120"; OVR["ctx"] = "2048"   # the build saves its snapshot (run.knobs) only after enough decode steps: a 24-token run writes none
        CAMPAIGN = camp + "-probe"
        PROCSTAT.pop("knobs_end", None)
        try:
            run_cell(build, "tetrisc", env, False, args, True, "knob probe (one token)", warm=False)
        except Exception as e:
            print(f"knob probe failed: {e}", flush=True)
        finally:
            OVR.clear(); OVR.update(saved); CAMPAIGN = camp
        used = PROCSTAT.get("knobs_end") or ""
        CELL_KNOBS[(build, name)] = used or "(no knob line: source defaults of this build)"
        pins = [w[4:] for w in vargs.split() if w.startswith("moe:")]
        miss = pin_check({("LLAMA_MOE_CACHE_" + k.upper().replace("-", "_")): v for k, v in (p.split("=", 1) for p in pins) if k != "autotune"}, used) if used else []
        miss += pin_check(env, used) if used else []
        print(f"KNOBS {build} / {name}\n   requested: {' '.join(f'{k}={v}' for k, v in env.items())} | {args}\n   used:      {used or '(this build writes no knob line)'}"
              + (f"\n   MISMATCH: the build does not show {', '.join(miss)}" if miss else ""), flush=True)
        csv_append("cell-knobs.csv", ["ts", "campaign", "model", "build", "variant", "requested", "used", "mismatch"],
                   dict(ts=time.strftime("%F %T"), campaign=camp, model=os.path.basename(model), build=build, variant=name,
                        requested=json.dumps(env, sort_keys=True) + " | " + args, used=used, mismatch=" ".join(miss)))
    LOG_LETTERS = saved_letters


# ---- bench: models x builds x variants, every cell recorded, then the report ----
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
    global MODEL, PERFPROF, GPULOG, KEEP_LOOPS, CLI_EXE, TEMPLATED
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
    for model in a.models.split(","):
        MODEL = model
        if not m:
            prep_model()
        cells = [(build, name, (stock_variants.get(name, "") if build.startswith("stock") else variants[name]))
                 for build in a.builds.split(",") for name in (list(dict.fromkeys(["default"] + list(stock_variants))) if build.startswith("stock") else names)]
        if not m and not a.no_probe:
            probe_cells(cells, a, model)
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
             "12k": ("/completion", "src_12k.cpp"), "128k": ("/completion", "src_128k.cpp"), "code": ("/completion", "src_12k.cpp")}
    path, f = kinds[a.kind]
    if f:
        text = open(os.path.join(PROMPTS, f)).read()
        body = {"prompt": ovr_prompt(text[:9000] if a.kind == "code" else text),
                "n_predict": int(OVR.get("tokens") or (64 if a.kind == "128k" else 32)), "cache_prompt": False, **GREEDY}
    else:
        pr = CHAT_PROMPTS[a.idx % len(CHAT_PROMPTS)] if a.kind == "chatv" else "write smallest html tetris game"
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
                                     "LLAMA_MOE_CACHE_CTL": ctl}, a.args.split() if a.args else [])
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
                row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"], "ok": 1, "args": kv}
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
                 (fcfg.model("mimo"), "--cpu-moe" + _T, None, 2, True)]
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
    the workflow's cmake flags): stock numbers must be what upstream's binaries do, never the fork's recipe. Result: /p/bw/gh-rel/stock-b<build>-<hash9>
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
    D = os.path.join(fcfg.path("gh_rel"), f"stock-b{m.group(1)}-{h}")
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
    """the release archives of a release build: llama-release-b<build>-<commit>-bin-ubuntu-cuda-13.4-x64.tar.gz (binaries, LICENSE, the libomp.so the clang
    build links dynamically) and the matching cudart archive, smoke-started in SMOKE_IMAGES; DEVBUILDS/NAME = both archives unpacked (what users get)"""
    uid = f"{os.getuid()}:{os.getgid()}"
    dk = lambda *args: subprocess.run(["docker", "run", "--rm", "--user", uid] + list(args), capture_output=True, text=True)
    vout = dk("-e", "LD_LIBRARY_PATH=/b", "-v", f"{binp}:/b:ro", "llama-rel-el8", "/b/llama-server", "--version")   # the host has neither libomp nor the CUDA 13 runtime
    v = re.search(r"build (\d+)", vout.stdout + vout.stderr); c = re.search(r"commit ([0-9a-f]+)", vout.stdout + vout.stderr)
    if not v or not c:
        sys.exit(f"release: no version from llama-server --version: {(vout.stdout + vout.stderr)[-300:]}")
    N = f"llama-release-b{v.group(1)}-{c.group(1)}-bin-ubuntu-cuda-13.4-x64"; C = "cudart-" + N
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


def cmd_build(a):
    r"""Builds, in containers (pc1) or on the W10 laptop (pc3), with ccache.
    default: dev build (the release container and flags, llama-server / llama-cli / llama-perplexity only) of a worktree as it is (--tree) or of a commit
      (--commit: a temporary worktree, removed after the build) into DEVBUILDS/NAME with BUILD_INFO (commit, branch, patch).
    --release: the release recipe (was hcf/lrel/build.sh): every target incl. tests, the archives in REL_OUT (Linux tar.gz + cudart), smoke starts in
      SMOKE_IMAGES, DEVBUILDS/NAME = the archives unpacked. Image llama-rel-el8 from fork/docker/Dockerfile.
    --stock: upstream at --ref from --src with upstream's release recipe (was hcf/lrel/stock-build.sh) into /p/bw/gh-rel/stock-b<build>-<hash> (no NAME).
    After a pc1 build the regress cell runs (restricted OLMoE, pinned knobs, against the best binary) unless --no-regress.
    --machine pc3: the same commit built on the W10 laptop (git bundle over ssh, its release-out\winbuild.bat) into N:\NAME (no regress there);
      with --release also the zip (binaries, VC runtime DLLs, LICENSE) in N:\llama.cpp\release-out and a copy in REL_OUT."""
    if a.stock:
        return _build_stock(a)
    if not a.name:
        sys.exit("build: NAME is needed (the directory under DEVBUILDS)")
    name = a.name
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "logs", f"build-{name}" + ("" if a.machine == "pc1" else f"-{a.machine}") + ".log")
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
            cmd = (f"cd /d N:\\llama.cpp && git fetch -q C:\\Users\\o\\{ref}.bundle {ref} && git checkout -q FETCH_HEAD && call N:\\llama.cpp\\release-out\\winbuild.bat"
                   f" && (if not exist N:\\build mkdir N:\\build) && (if not exist N:\\build\\{name} mkdir N:\\build\\{name}) && xcopy /q /y N:\\llama.cpp\\build-cuda-dl\\bin\\* N:\\build\\{name}\\"
                   f" & (copy /y N:\\build\\admit-pin-b12353-10d6c1e5c\\*140*.dll N:\\build\\{name}\\ >nul 2>&1) & echo PKG_OK {short}")   # the VC runtime DLLs of an old build, when it is still there: the release zip takes them from the b11214 zip anyway
            subprocess.run(ssh + [cmd], stdout=lf, stderr=subprocess.STDOUT)
        os.remove(bundle)
        ok = f"PKG_OK {short}" in open(log).read()
        if ok and a.release:   # the zip the way every published Windows release was made (build-cuda-dl\bin + the VC runtime DLLs + LICENSE)
            vout = subprocess.run(ssh + [f"N:\\build\\{name}\\llama-server.exe --version"], capture_output=True, text=True)
            v = re.search(r"build (\d+)", vout.stdout + vout.stderr); c = re.search(r"commit ([0-9a-f]+)", vout.stdout + vout.stderr)
            if not v or not c:
                sys.exit(f"release zip: no version from llama-server.exe: {(vout.stdout + vout.stderr)[-300:]}")
            z = f"llama-release-b{v.group(1)}-{c.group(1)}-bin-win-cuda-13.4-x64.zip"
            pkg = (f"set P=N:\\tmp\\pkg& (if exist N:\\tmp\\pkg rmdir /s /q N:\\tmp\\pkg) & mkdir N:\\tmp\\pkg && xcopy /q N:\\llama.cpp\\build-cuda-dl\\bin\\* N:\\tmp\\pkg\\ >nul"
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
        tmp = os.path.join(fcfg.path("build_tmp"), name)
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
             f"-DOPENSSL_USE_STATIC_LIBS=ON -DOPENSSL_ROOT_DIR=/opt/openssl -DLLAMA_BUILD_TESTS={'ON' if a.release else 'OFF'} -DLLAMA_BUILD_UI=OFF "
             "-DCMAKE_EXE_LINKER_FLAGS=\"$SLD\" -DCMAKE_SHARED_LINKER_FLAGS=\"$SLD\" -DCMAKE_MODULE_LINKER_FLAGS=\"$SLD\" && "
             + (f"cmake --build {B} -j $(nproc)" if a.release else f"cmake --build {B} -j 8 --target llama-server llama-cli llama-perplexity"))
    lk = queue_lock(wait=True)   # no build while a benchmark runs (a build takes the CPU the measurements use)
    iid = _image("llama-rel-el8", keep=a.release)
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

def cmd_queue(a):
    """Adds tasks to the bench queue the safe way: under the queue's own file lock (queued.py read-modify-writes tasks.csv; an edit without that lock is lost to its next
    rewrite - a hand edit was, once), after a backup of tasks.csv with the last 3 versions kept in queue/backups. `backups` lists them."""
    import fcntl, shutil
    Q = "/p/bw/data/q5h/queue"; INI = os.path.join(Q, "tasks.csv"); BK = os.path.join(Q, "backups")
    COLS = ["id", "order", "status", "campaign", "created", "start", "stop", "exit", "pid", "env", "cli", "log", "note"]
    if a.action == "backups":
        for f in sorted(os.listdir(BK)) if os.path.isdir(BK) else []: print(os.path.join(BK, f))
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
        for t in new:
            if t["id"] in have: print("exists, skipped:", t["id"]); continue
            r = {c: "" for c in COLS}; r.update(status="next", created=time.strftime("%F %T"), campaign=t["id"], log=os.path.join(Q, "logs", t["id"] + ".log")); r.update({k: str(v) for k, v in t.items()})
            rows.append(r); print("added:", t["id"])
        with open(INI + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n"); w.writeheader(); [w.writerow({c: r.get(c, "") for c in COLS}) for r in rows]
        os.replace(INI + ".tmp", INI)
    return 0
def cmd_stop(a):
    """Stops a running measurement chain for real: the harness (bench, regress, exp, run, quick, ctl, prompt with their flock wrappers), the servers and tools of our builds and an nsys around them, every descendant of it (the
    servers and tools it started: a killed script does not take its children with it, and a lone kill of the harness lets the parent script start the next cell),
    and the .sh scripts that started it. Never the shell that runs this command or its ancestors (a plain `bash -c` is left alone, only real script files count).
    SIGTERM first, SIGKILL for what is still there after --grace seconds, then it waits until the GPUs have (almost) no memory in use, and puts the power limits
    back to the values of the [gpu] limits_env file (LIMIT0=, LIMIT1=) unless --no-power. --dry lists what it would stop and changes nothing.
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
    def is_script(argv):   # `bash ./something.sh ...`, not `bash -c ...`
        return len(argv) >= 2 and os.path.basename(argv[0]) in ("bash", "sh") and argv[1].endswith(".sh")
    def is_harness(argv):
        py = os.path.basename(argv[0]).startswith("python")
        return py and any(x.endswith("run.py") for x in argv[1:4]) and any(x in ("bench", "regress", "exp", "run", "quick", "ctl", "prompt", "baseline") for x in argv[1:6])
    def is_tool(argv):   # a server or tool of one of our builds, or an nsys wrapped around one
        exe = argv[0]
        return os.path.basename(exe) == "nsys" or any(exe.startswith(d) for d in (DEVBUILDS + "/", "/p/bw/rels/", "/p/bw/gh-rel/") if d)
    roots = {p for p, (argv, _) in procs.items() if p not in protected and (is_harness(argv) or is_tool(argv) or (os.path.basename(argv[0]) == "flock" and any(x.endswith("run.py") for x in argv)))}
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
    for p in sorted(targets):
        kind = "script" if p in scripts else "harness" if p in roots else "child"
        print(f"  {kind:8} {p:>8}  {' '.join(procs[p][0])[:110]}")
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
    lenv = fcfg.get("gpu", "limits_env")
    if not a.no_power and lenv and os.path.isfile(lenv):
        for line in open(lenv).read().splitlines():
            m = re.match(r"LIMIT(\d+)=(\d+)", line.strip())
            if m:
                r = subprocess.run(["sudo", "-n", NVSMI, "-i", m.group(1), "-pl", m.group(2)], capture_output=True, text=True)
                print(f"stop: GPU {m.group(1)} power limit {m.group(2)} W" + ("" if r.returncode == 0 else f"  (failed: {r.stderr.strip()[:80]})"))


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
        ccamp = camp + "-" + os.path.basename(model)[:6]
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
  db      the old SQLite files (tools/moe-bench/perf.db from git history, fork/bench/archive/perf-*.db, /p/bw/rels/runs.db, /p/bw/data/**/perf*.db, rc_perf*.db)
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
    for f in sorted(set(glob.glob("/p/bw/*/tools/bench/run-history.csv") + glob.glob("/p/bw/*/fork/bench/run-history.csv") + glob.glob("/p/bw/llama.cpp/.claude/worktrees/*/tools/bench/run-history.csv") + [os.path.join(ROOT, "fork/bench/run-history.csv")])):
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


    dbs = glob.glob(os.path.join(ROOT, "fork/bench/archive/*.db")) + glob.glob("/p/bw/rels/runs.db") + glob.glob("/p/bw/data/**/perf*.db", recursive=True) + glob.glob("/p/bw/data/**/rc_perf*.db", recursive=True)
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
    SRC = [("o@192.168.1.12", "/home/o/w/tools/bench/run-history.csv"), ("o@192.168.1.12", "/home/o/wt-merge/tools/bench/run-history.csv"),
           ("o@192.168.1.12", "/home/o/wt-pin/tools/bench/run-history.csv"), ("o@192.168.1.2", "N:/llama.cpp/tools/bench/run-history.csv"),
           ("o@192.168.1.2", "N:/llama-pred/tools/bench/run-history.csv"), ("o@192.168.1.2", "N:/wt-next/tools/bench/run-history.csv")]
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
    __doc__ = """run.py pick-features GROUPS.txt [--tree /p/bw/wt-base509]: put fork features on the fast base one group at a time, each group tested before the next.
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
    WORK = fcfg.path("work"); os.makedirs(WORK, exist_ok=True)
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
    WORK = fcfg.path("work"); os.makedirs(WORK, exist_ok=True)
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
    PROMPT = os.path.join(HERE, "bench", "src_12k.cpp")   # the README pf12k prompt (12k tokens, 100 generated, -c 16384 -ub 2048 -b 2048): prompt processing and decode in one run, the same prompt as the README table
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
    __doc__ = """run.py series [--base 6753a033f] [--tip main] [--branch series] [--tree /p/bw/wt-series]

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
        ap.add_argument("--tree", default="/p/bw/wt-series")
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


TOOLS = {
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
    TESTS = ["ppl", "ppl3", "pplb", "pplc", "tetris", "t100", "chat", "chat2", "chatv", "long1k", "in1k", "agent", "pf2k", "pf3k", "pf4k", "pf6k", "pf8k", "pf10k", "pf12k", "pf12k-stock", "pf128k", "fix", "fix12k", "edit4", "game4", "tetrisc", "game4c", "edit4c", "long1kc"]
    r = sp.add_parser("run", help="measured runs of one test for BUILD...")
    r.add_argument("builds", nargs="+")
    r.add_argument("-t", "--test", default="ppl", choices=TESTS + list(ALIASES))
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
    b.add_argument("-t", "--test", default="t100", choices=["ppl", "ppl3", "pplb", "pplc", "t100", "tetris"] + TESTS[6:] + list(ALIASES))
    b.add_argument("--stock-variant", action="append", default=[], help="NAME=ARGS: stock builds also run this placement (e.g. tuned='--n-cpu-moe 30'): the hand-tuned baseline")
    b.add_argument("--rep-from", type=int, default=0, help="first repetition to run (0-based): -n 2 --rep-from 1 repeats only the second repetition, e.g. one that failed")
    b.add_argument("-n", type=int, help="measured runs per cell, interleaved ABBA (default 3 for fix/fix12k, else 2); the first run of a cell is preceded by a discarded one")
    b.add_argument("--machine", default="pc1", help="pc1 = this host (server tests); other ids from 'run.py machine' run llama-cli over ssh")
    b.add_argument("--campaign", help="name in table campaigns (default bench-<date>)")
    b.add_argument("--deadline", help="HH:MM: cells that would start later are skipped")
    b.add_argument("--last", type=int, help="report the last N runs per cell only (hot runs of a learning autotune)")
    b.add_argument("--tol", type=float, help="flag cells below TOL x stock as REGRESSION (e.g. 0.95)")
    b.add_argument("--note")
    b.add_argument("--no-probe", action="store_true", help="skip the knob probe (one token per cell before the first measured run; prints requested against used settings)")
    b.add_argument("--warm", action="store_true", help="force a discarded run before each measured one (hot run); default off")
    b.add_argument("--no-warm", action="store_true", help="never a discarded run: the first run as a user sees it")
    b.add_argument("--sampler", help="extra sampling parameters for every request, comma separated, e.g. repeat_penalty=1.1,dry_multiplier=0.8 (anti-repetition tests); temperature 0 / top_k 1 stay unless overridden")
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
    m = sp.add_parser("machine", help="list machines, or add one: machine add ID --ssh user@host --root DIR --os linux|windows")
    m.add_argument("action", nargs="?", default="list", choices=["list", "add"])
    m.add_argument("id", nargs="?")
    m.add_argument("--ssh")
    m.add_argument("--root", help="directory holding the build dirs on that machine")
    m.add_argument("--os", default="linux")
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
    rg = sp.add_parser("regress", help="quick check after a commit: the build in DIR (default ./build-dev/bin) against the best build kept in /p/bw/rels/best")
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
    bd.add_argument("--stock", action="store_true", help="upstream llama.cpp with upstream's release recipe (was hcf/lrel/stock-build.sh) into /p/bw/gh-rel/stock-b<build>-<hash>")
    bd.add_argument("--src", default="~/lc", help="--stock: the upstream clone")
    bd.add_argument("--ref", default="origin/master", help="--stock: the upstream commit to build")
    bd.add_argument("--tree", help="worktree to build as it is (uncommitted changes included, BUILD_INFO patch= marks them)")
    bd.add_argument("--commit", help="commit to build in a temporary worktree of --repo")
    bd.add_argument("--repo", default=BUILD_REPO)
    bd.add_argument("--machine", default="pc1", choices=["pc1", "pc3"])
    bd.add_argument("--w10-base", default="6846d4803", help="a commit the W10 repository has: the bundle carries only what is newer")
    bd.add_argument("--no-regress", action="store_true", help="skip the regress cell after the build")
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
    st = sp.add_parser("stop", help="stop the running measurement chain properly: parent scripts, harness, all children; then wait for free GPU memory and restore the power limits")
    st.add_argument("--dry", action="store_true", help="list what would be stopped, change nothing")
    st.add_argument("--grace", type=float, default=10.0, help="seconds between SIGTERM and SIGKILL")
    st.add_argument("--wait-gpu", type=float, default=60.0, help="seconds to wait until the GPUs have no memory in use")
    st.add_argument("--no-power", action="store_true", help="leave the power limits alone")
    qq = sp.add_parser("queue", help="the bench queue (queued.service): add TASK_ID --cli '...' [--order N] [--note ..]  |  add-json (a JSON list on stdin)  |  backups; edits take the queue's own file lock and keep the last 3 versions of tasks.csv")
    qq.add_argument("action", choices=["add", "add-json", "backups"])
    qq.add_argument("task_id", nargs="?")
    qq.add_argument("--cli", help="the command line the queue runs (bash m111log.sh ... or a run.py line)")
    qq.add_argument("--order", type=float, default=-100, help="run order, ascending (default -100 = ahead of the waiting tasks)")
    qq.add_argument("--note", default="")
    br = sp.add_parser("branches", help="local branches with commits not in TARGET (counts, dates); --merge a,b,c merges them in the checkout of TARGET (refuses on another branch or a dirty tree)")
    br.add_argument("target", nargs="?", default="prerelease2")
    br.add_argument("--repo", default=os.getcwd(), help="the checkout to look at / merge in")
    br.add_argument("--merge", help="comma-separated branch names")
    br.add_argument("--limit", type=int, default=60)
    s_ = sp.add_parser("show")
    s_.add_argument("-t", "--test")
    s_.add_argument("-b", "--build")
    s_.add_argument("--runs", action="store_true")
    mf = sp.add_parser("misfires", help="automatic choices that went against what was measured: a prompt run on one card where the split of the same model and pass "
                                         "size was faster in another run, and the reverse; stock re-measurements of the placement (from the pf_* / placement columns)")
    mf.add_argument("--days", type=float, default=7.0, help="look at runs of the last N days")
    mf.add_argument("--tol", type=float, default=0.05, help="a choice counts as wrong when the other mode was faster by more than this share")
    mf.add_argument("--model", help="part of the model name")
    a = ap.parse_args()
    {"run": cmd_run, "bench": cmd_bench, "report": cmd_report, "gains": lambda a: [gains_write(c) for c in a.campaigns], "machine": cmd_machine, "prompt": cmd_prompt, "ctl": cmd_ctl,
     "show": cmd_show, "regress": cmd_regress, "build": cmd_build, "quick": cmd_quick, "hugepool": cmd_hugepool, "stop": cmd_stop, "queue": cmd_queue, "branches": cmd_branches, "misfires": cmd_misfires,
     "exp": lambda a: runexp.run_exp(a.spec, sys.modules[__name__], a.dry, a.only.split(",") if a.only else None, a.rounds)}[a.cmd](a)
