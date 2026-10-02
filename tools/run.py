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
tools/bench ($RUN_DATA). MODEL=/path/model.gguf is required for run/ctl; PERF_MODELS_DIR=/dir drops other *.gguf under it from RAM before
measuring. Every measured run is preceded by a discarded run with the same settings (hot runs; --no-warm skips it); every run gets its own server
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
import argparse, csv, hashlib, io, json, os, re, statistics, subprocess, sys, time, urllib.request

PROMPTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bench")   # frozen prompts (src_12k.cpp, src_128k.cpp, longsrc.cpp) live in tools/bench/
HERE = os.path.abspath(os.environ.get("PERF_BUILDS", os.getcwd()))  # one subdirectory per build (its bin/ contents)
DATA = os.environ.get("RUN_DATA") or PROMPTS   # directory of run-history.csv, campaigns.csv, machines.csv, tune.csv (default: tools/bench, committed)
MODEL = os.environ.get("MODEL", "")
MODELS_DIR = os.environ.get("PERF_MODELS_DIR", "")  # other models' pages are dropped from RAM before a run
PPL_TEXT = os.path.join(PROMPTS, "longsrc.cpp")  # frozen snapshot of llama.cpp common/json-schema-to-grammar.cpp (code, like agent prompts)
BARE = False
THREADS = os.environ.get("PERF_THREADS", "6")                  # decode threads of every run (tools/cloud/pod.sh sets both for rented boxes)
DEVS = os.environ.get("PERF_DEV", "CUDA0,CUDA1")                # -dev list (the GPU order is pinned)
COMMON_ALL = ["-t", THREADS, "--cpu-moe", "-nr", "--moe-expert-cache", "-1", "-dev", DEVS]
if "," not in os.environ.get("CUDA_VISIBLE_DEVICES", ","):  # one visible GPU: no device order to pin
    COMMON_ALL = COMMON_ALL[:-2]
COMMON = COMMON_ALL
PORT = 8099
OVR = {}   # --prompt TEXT|@file, --tokens N, --ctx N override the test's own prompt, generated tokens and context
CAMPAIGN = ""  # --campaign NAME: column campaign of every row of this invocation
REP = None  # bench repetition (0-based) of the run being stored; None outside bench
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
CAMP_COLS = ["name", "start", "note", "conclusion", "why"]
MACH_COLS = ["id", "ssh", "root", "kind", "cpu", "ram_gb", "ram_type", "gpus", "links", "os", "hostname", "board", "storage",
             "mem_detail", "gpu_detail", "sw", "note"]
TUNE_COLS = ["ts", "commit", "line"]


def _fmt(v):
    if v is None:
        return ""
    return f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def csv_append(name, cols, d):
    path = os.path.join(DATA, name)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        w.writerow(cols)
    w.writerow([_fmt(d.get(k)) for k in cols])
    with open(path, "a") as f:
        f.write(buf.getvalue())  # one write per line: concurrent benches (several machines, one host) do not interleave


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


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def wait_vram_free():
    kill_leftovers()
    t0 = time.time()
    while True:
        if time.time() - t0 > 60:  # something still holds VRAM: clean up again
            kill_leftovers()
            t0 = time.time()
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        if out and max(int(x) for x in out) < 500 + int(os.environ.get("PERF_HOLD_MB", 0)):  # PERF_HOLD_MB: VRAM a holder process keeps on purpose (simulated smaller card)
            return
        time.sleep(1)


def version(build):
    r = subprocess.run([os.path.join(HERE, build, "llama-server"), "--version"], capture_output=True, text=True,
                       env={**os.environ, "LD_LIBRARY_PATH": os.path.join(HERE, build)})
    m = re.search(r"version: (.*)", r.stdout + r.stderr)
    return m.group(1).strip() if m else "?"


def cache_stats(log):
    m = re.findall(r"steps=(\d+) hits=(\d+) misses=(\d+) hit-rate=([\d.]+)%", log)
    return {"steps": int(m[-1][0]), "hits": int(m[-1][1]), "misses": int(m[-1][2]), "hit_rate": float(m[-1][3])} if m else {}


def server_start(build, env, args):
    logf = f"/tmp/perf-{build}.log"
    with open(logf, "w") as lf:
        p = subprocess.Popen([os.path.join(HERE, build, "llama-server"), "-m", MODEL, "--port", str(PORT)] + ([] if BARE else ["-np", "1"])
                             + args, stdout=lf, stderr=subprocess.STDOUT, env=env, preexec_fn=_die_with_parent)
    while True:
        log = open(logf).read()
        if "listening on" in log:
            return p, logf
        if p.poll() is not None:
            raise RuntimeError("server exited: " + log[-300:])
        if "couldn't bind" in log:
            stop(p)
            raise RuntimeError("port busy: " + log[-300:])
        time.sleep(1)


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def server_run(build, env, args, path, body):
    p, logf = server_start(build, env, args)
    try:
        return post(path, body), logf
    finally:
        stop(p)


def agent_run(build, env, args):
    sys.path.insert(0, os.environ.get("PERF_AGENT_DIR", "/p/bw/miniagentic"))
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
    env = {**os.environ, **extra_env, "LD_LIBRARY_PATH": os.path.join(HERE, build), "LLAMA_MOE_CACHE_STATS": "1",
           "LLAMA_MOE_CACHE_STATSLOG": "32", "LLAMA_MOE_STATSLOG": f"/tmp/perf-{build}-stats.txt",  # builds with statslog write a text line per 32 steps (moe-stats.csv)
           "LLAMA_MOE_CACHE_TUNED": os.environ.get("LLAMA_MOE_CACHE_TUNED", "0")}  # no saved tuner decisions: every run starts cold
    row = {}
    if test in ("ppl", "ppl3"):
        nb = "1" if test == "ppl" else "3"
        args = COMMON + ["-f", PPL_TEXT, "-c", "1024", "--chunks", "2", "-b", nb, "-ub", nb]
        r = subprocess.run([os.path.join(HERE, build, "llama-perplexity"), "-m", MODEL] + args,
                           capture_output=True, text=True, env=env, preexec_fn=_die_with_parent)
        log = r.stdout + r.stderr
        s = re.search(r"([\d.]+) seconds per pass", log)
        ppl = re.findall(r"PPL = ([\d.]+)", log)
        row = {"s_per_pass": float(s.group(1)) if s else None, "ppl": float(ppl[-1]) if ppl else None,
               "tps": 1024 / float(s.group(1)) if s else None}
    else:
        if BARE:
            COMMON = list(extra_args)
        if test in ("pf12k", "pf12k-stock", "pf128k", "fix12k"):
            pf_ctx = str(OVR.get("ctx") or (131072 if test == "pf128k" else 16384))
            dev = ["-t", THREADS, "-dev", os.environ.get("PERF_PF_DEV") or ("CUDA1,CUDA0" if DEVS == "CUDA0,CUDA1" else DEVS), "-c", pf_ctx, "-ub", "2048", "-b", "2048"]
            args = list(extra_args) if BARE else dev + ((["--moe-expert-cache", "0"] if test == "pf12k-stock" and not build.startswith("stock") else []) if test == "pf12k-stock" or plain or build.startswith("stock") else ["--cpu-moe", "-nr", "--moe-expert-cache", "-1"]) + list(extra_args)
            if BARE and (plain or build.startswith("stock")) and "-c" not in args:
                args += ["-c", pf_ctx]  # stock's default context (4096) rejects the 12k prompt (HTTP 400); the fork sets 32k itself
            res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": ovr_prompt(open(os.path.join(PROMPTS, "src_128k.cpp" if test == "pf128k" else "src_12k.cpp")).read()), "n_predict": int(OVR.get("tokens") or os.environ.get("PERF_NPRED", 64 if test == "pf128k" else 256 if test == "fix12k" else 32)),
                                    "cache_prompt": False, **GREEDY, **(FIXED if test == "fix12k" else {})})
            t = res["timings"]
            row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
                   "note": f"prompt_n={t['prompt_n']} prompt_s={t['prompt_ms']/1e3:.1f}"}
            if test == "fix12k":
                row["md5"] = hashlib.md5(res["content"].encode()).hexdigest()[:8]
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
        if test in ("tetris", "t100", "fix"):
            # t100: what most people run on a CLI: one short prompt, 100 tokens, -c 1024, temperature 0 (GREEDY)
            args = COMMON + ["-c", str(OVR.get("ctx") or (2048 if test == "fix" else 1024))]
            res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": ovr_prompt("generate smallest html tetris game." if test == "tetris" else "write smallest html tetris game"),
                                    "n_predict": int(OVR.get("tokens") or (-1 if test == "tetris" else 1024 if test == "fix" else 100)), **GREEDY, **(FIXED if test == "fix" else {})})
            text = res["content"]
        else:
            args = COMMON + ["-c", str(OVR.get("ctx") or 4096)]
            chatv = test == "chatv"  # varied prompts (index = run counter), thinking off: the topic switches the cache has to follow
            res, logf = server_run(build, env, args, "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": ovr_prompt(CHAT_PROMPTS[IDX % len(CHAT_PROMPTS)] if chatv else "write smallest html tetris game")}],
                                    "max_tokens": int(OVR.get("tokens") or (300 if chatv else 1500)), **GREEDY,
                                    **({"chat_template_kwargs": {"enable_thinking": False}} if chatv else {})})
            m = res["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
        t = res["timings"]
        log = open(logf).read()
        row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
               "output": text, "md5": hashlib.md5(text.encode()).hexdigest()[:8]}
        if t.get("draft_n"):  # speculative decoding (MTP / draft model): accepted / drafted tokens
            row["note"] = f"draft {t.get('draft_n_accepted')}/{t['draft_n']}"
        if "from pageable host memory" in log:  # weights not pinned: uploads are slower, the run does not compare with pinned ones
            row["note"] = (row.get("note", "") + " pageable uploads").strip()
    row.update(cache_stats(log))
    row.update(ok=1 if row.get("tps") else 0, args=" ".join(args))
    return row


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
    """GPU power limits in W of this host, like 350+370 (PERF_PL overrides); only for rows of the local machine, remote ones stay empty."""
    if "pl" not in _PL:
        local = os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
        out = ""
        if os.environ.get("PERF_PL") is None:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            except Exception:
                pass
        _PL["pl"] = os.environ.get("PERF_PL") or "+".join(str(round(float(x))) for x in out.split() if x.replace(".", "").isdigit())
        _PL["local"] = local
    return _PL["pl"] if hw == _PL["local"] else ""


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
             up_mib=int(sum(col(6))), ddr_gbs=mean(col(7)), pcie_gbs=mean(col(8)))
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


def save_stats(build, ts, test, hw, model):
    """The server's statslog file of the measured run (a value per 32 steps: t/s, hit, uploads, evictions, MiB uploaded, DDR and PCIe GB/s) -> a section of stats.ini."""
    f = f"/tmp/perf-{build}-stats.txt"
    if not os.path.exists(f):
        return None
    w = [[float(x) for x in l.split()] for l in open(f) if l[:1].isdigit() and len(l.split()) in (10, 14, 18)]
    os.remove(f)  # the next run of this build must not re-save it
    if w:
        with open(os.path.join(DATA, "stats.ini"), "a") as out:
            out.write(stats_section(f"{ts} {build}", model, test, hw, w))
    return w


def store(row, build, test, rec, note, hw, vstr):
    """One csv row per run: date and commit first (the commit and build number come from the binary's own --version), host, model, settings."""
    mc = re.search(r"commit (\w+)", vstr or "")
    mb = re.search(r"build (\d+)", vstr or "")
    row.update(commit_sha=os.environ.get("PERF_COMMIT") or (mc.group(1) if mc else None),  # PERF_COMMIT: a build of a dirty tree, e.g. <sha>+patch
               build_no=int(mb.group(1)) if mb else None, origin="stock" if build.startswith("stock") else "fork", hw=hw,
               model=model_name(), ts=time.strftime("%F %T"), build=build, test=test,
               env=json.dumps(rec, sort_keys=True), note=" | ".join(x for x in (note, row.get("note")) if x) or None)
    w = save_stats(build, row["ts"], test, hw, row["model"])
    extra = {k: v for k, v in rec.items() if k != "args"}
    bi = build_info(build)
    if bi.get("commit"):
        row["commit_sha"] = bi["commit"] + ("+" + bi["patch"] if bi.get("patch") and bi.get("patch") != "0" else "")   # the commit the binary was built from (+hash of an uncommitted diff)
    if w:
        csv_append("stats.csv", STATS_CSV_COLS, dict(ts=row["ts"], build=build, commit=row["commit_sha"], branch=bi.get("branch", ""), patch=bi.get("patch", ""), hw=hw, model=row["model"], test=test,
                   args=rec.get("args", ""), env=json.dumps(extra, sort_keys=True, separators=(",", ":")) if extra else "", tps=row.get("tps"), hit=row.get("hit_rate"), **stats_csv_row(w)))
    csv_append("run-history.csv", RUN_COLS, dict(
        ts=row["ts"], commit=row["commit_sha"], hw=hw, test=test, tps=row.get("tps"), pp=row.get("pp_tps"), hit=row.get("hit_rate"),
        ok=row.get("ok", 1), model=row["model"], build=build, args=rec.get("args", ""), md5=row.get("md5"), campaign=CAMPAIGN,
        note=row["note"], ppl=row.get("ppl"), spp=row.get("s_per_pass"), n_gen=row.get("n_gen"), bno=row["build_no"], origin=row["origin"],
        env=json.dumps(extra, sort_keys=True, separators=(",", ":")) if extra else "", cmd=row.get("args"),
        pl=power_limits(hw), proto="v2" if test.startswith("fix") else "v1", rep=REP, seed=SEED if test.startswith("fix") else None,
        branch=bi.get("branch", ""), patch=bi.get("patch", "")))


def show_row(row, build, test):
    print(f"{row['ts']} {build:16s} {test:6s} {row.get('env')} "
          + (f"{row['tps']:6.2f} t/s" if row.get("tps") else "FAIL")
          + (f" {row['s_per_pass']:.2f} s/pass" if row.get("s_per_pass") else "")
          + (f" hit {row['hit_rate']}%" if row.get("hit_rate") is not None else "")
          + (f" pp {row['pp_tps']:.1f} t/s" if row.get("pp_tps") else "")
          + (f" md5={row['md5']}" if row.get("md5") else "") + (f" [{row['note']}]" if row.get("note") and (test == "agent" or "draft" in row["note"] or "pageable" in row["note"]) else "")
          + (f" PPL {row['ppl']}" if row.get("ppl") else ""), flush=True)


def save_tune_log(row):
    """The self-tune decisions of this run's server log, keyed by the run's date and commit (tune.csv)."""
    try:
        lines = [re.sub(r"^\S+ [A-Z] ", "", l.strip())[:300] for l in open(f"/tmp/perf-{row['build']}.log", errors="replace")
                 if "self-tune" in l or "thread autotune" in l]
    except OSError:
        return
    for l in lines:
        csv_append("tune.csv", TUNE_COLS, dict(ts=row["ts"], commit=row["commit_sha"], line=l))


def run_cell(build, test, extra, plain, args, bare, note, hw=None, warm=True):
    """One measured run, preceded by a discarded run with the same build and settings: model switches, a previous pinned load
    (drops the page cache) and cache/tuner warm-up all showed 5-12% swings on the first run."""
    global BARE, IDX
    BARE = bare
    xargs = args.split() if args else ()
    if warm:  # --no-warm: the first run as a user sees it
        try:
            run_one(build, test, extra, plain, xargs)
        except Exception as e:
            print(f"throwaway run failed: {e}", flush=True)
    try:
        row = run_one(build, test, extra, plain, xargs)
    except Exception as e:
        row = {"ok": 0, "note": str(e)[:300]}
    rec = {**extra, **({"bare": "1"} if bare else {}), **({"plain": "1"} if plain else {}), **({"args": args} if args else {}),
           **{k: str(v) for k, v in OVR.items() if k != "prompt"}}
    hw = hw or os.environ.get("PERF_HW") or {"1": "pc1", "2": "pc2", "nb": "pc3"}.get(os.uname().nodename, os.uname().nodename)
    store(row, build, test, rec, note, hw, version(build))
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
            run_cell(build.rstrip("/"), a.test, extra, a.plain, a.args, a.bare, a.note, warm=not a.no_warm)


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


def set_overrides(a):
    OVR.clear()
    for k in ("prompt", "tokens", "ctx"):
        if getattr(a, k, None) is not None:
            OVR[k] = getattr(a, k)


# ---- machines: machines.csv, one row per machine (id, ssh, root, kind). pc1 is the local host; others run llama-cli over ssh ----
def machines():
    return {r["id"]: r for r in csv_load("machines.csv")}


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


def remote_exe(m, build):
    sep = "\\" if m["win"] else "/"
    return f'{m["root"]}{sep}{build}{sep}llama-cli' + (".exe" if m["win"] else "")


def remote_sh(m, cmd, timeout=3600):
    argv = ["bash", "-c", cmd] if m["ssh"] == "local" else ["ssh", m["ssh"], cmd]   # ssh=local: llama-cli on this host (the CLI one-shot case)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def remote_run(m, build, test, args, env=None):
    """The CLI version of a test (llama-cli, one prompt, greedy): what most people run. Returns a row like run_one."""
    exe, win = remote_exe(m, build), m["win"]
    envp = "".join((f"set {k}={v}&& " if win else f"{k}={v} ") for k, v in (env or {}).items())  # variant environment (env: words)
    if test in ("ppl", "ppl3"):  # the precise decode comparator on that machine: llama-perplexity, one token per call, longsrc.cpp (copy it to <root>\\<build>\\longsrc.cpp)
        sep = "\\" if win else "/"
        nb = "1" if test == "ppl" else "3"
        stf = f'{m["root"]}{sep}perf-stats.txt'
        envs = (f'set LLAMA_MOE_STATE=0&& set LLAMA_MOE_CACHE_TUNED=0&& set LLAMA_MOE_CACHE_STATS=1&& set LLAMA_MOE_CACHE_STATSLOG=32&& set LLAMA_MOE_STATSLOG={stf}&& ' if win else
                f'LLAMA_MOE_STATE=0 LLAMA_MOE_CACHE_TUNED=0 LLAMA_MOE_CACHE_STATS=1 LLAMA_MOE_CACHE_STATSLOG=32 LLAMA_MOE_STATSLOG={stf} ')
        cmd = envp + envs + f'{exe.replace("llama-cli", "llama-perplexity")} -m {MODEL} -f {m["root"]}{sep}{build}{sep}longsrc.cpp -c 1024 --chunks 2 -b {nb} -ub {nb} {args} ' + ("< nul 2>&1" if win else "< /dev/null 2>&1")
        out = remote_sh(m, cmd)
        sp_ = re.search(r"([\d.]+) seconds per pass", out)
        pp_ = re.findall(r"PPL = ([\d.]+)", out)
        row = {"s_per_pass": float(sp_.group(1)), "tps": 1024 / float(sp_.group(1)), "ppl": float(pp_[-1]) if pp_ else None, "ok": 1, "args": args} if sp_ else {"ok": 0, "note": out[-200:], "args": args}
        row.update(cache_stats(out))
        st = remote_sh(m, ("type " if win else "cat ") + stf + (" 2>nul" if win else " 2>/dev/null"))
        if st.strip():  # the run's statslog goes to stats.ini like a local run's
            open(f"/tmp/perf-{build}-stats.txt", "w").write(st)
        return row
    prompt = ovr_prompt("generate smallest html tetris game." if test == "tetris" else "write smallest html tetris game").replace('"', "'")
    n = int(OVR.get("tokens") or (-1 if test == "tetris" else 100))
    cmd = envp + f'{exe} -m {MODEL} -c {OVR.get("ctx") or 1024} --temp 0 -n {n} --no-display-prompt -p "{prompt}" {args} ' \
          + ("< nul 2>&1" if win else "< /dev/null 2>&1")
    if not win:
        cmd = f'LD_LIBRARY_PATH={os.path.dirname(exe)} ' + cmd
    out = remote_sh(m, cmd)
    t = re.findall(r"Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s", out)
    mh = re.findall(r"hit ([\d.]+)% this reply", out)
    row = {"tps": float(t[-1][1]), "pp_tps": float(t[-1][0]), "ok": 1, "args": args} if t else {"ok": 0, "note": out[-200:], "args": args}
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


# ---- bench: models x builds x variants, every cell recorded, then the report ----
VARIANTS = {"default": "", "cache0": "--moe cache=0", "atoff": "-at off"}  # --variant NAME=ARGS adds or overrides


def cmd_bench(a):
    global MODEL
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
        deadline = time.mktime(time.strptime(time.strftime("%F ") + f"{hh}:{mm}", "%F %H:%M"))
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
                 for build in a.builds.split(",") for name in (["default"] + list(stock_variants) if build.startswith("stock") else names)]
        # a variant may set environment variables: "env:KEY=VALUE" words (e.g. lru='env:LLAMA_MOE_CACHE_POLICY=lru'), the rest are server args
        for i in range(n):  # ABBA: odd repetitions run the cells backwards, so a drift (thermal, page cache) does not favour one cell
            REP = i
            for build, name, vargs in (cells if i % 2 == 0 else cells[::-1]):
                words = vargs.split()
                args = " ".join(w for w in words if not w.startswith("env:"))
                env = dict(w[4:].split("=", 1) for w in words if w.startswith("env:"))
                if deadline and time.time() > deadline:
                    print("deadline reached: remaining cells skipped", flush=True)
                    return cmd_report(argparse.Namespace(campaign=camp, since=None, last=None, tol=None, md=False))
                warm = not a.no_warm and i == 0  # one discarded run per cell, before its first repetition
                if m:
                    remote_cell(a.machine, m, build, a.test, args, a.note, warm=warm, env=env)
                else:
                    run_cell(build, a.test, env, False, args, True, a.note, warm=warm)
    REP = None
    cmd_report(argparse.Namespace(campaign=camp, since=None, last=a.last, tol=a.tol, md=False))


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("short", "12k", "128k"):  # the old run.py usage: run.py short|12k|128k [--url U]
        sys.argv.insert(1, "prompt")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    TESTS = ["ppl", "ppl3", "tetris", "t100", "chat", "chatv", "agent", "pf12k", "pf12k-stock", "pf128k", "fix", "fix12k"]
    r = sp.add_parser("run", help="measured runs of one test for BUILD...")
    r.add_argument("builds", nargs="+")
    r.add_argument("-t", "--test", default="ppl", choices=TESTS)
    r.add_argument("-n", type=int, default=1)
    r.add_argument("-e", "--env", action="append", default=[])
    r.add_argument("--note")
    r.add_argument("--campaign", help="name written to the campaign column of every row")
    r.add_argument("--bare", action="store_true", help="only -m (plus --args): the defaults a new user gets")
    r.add_argument("--plain", action="store_true", help="no cache flags (autofit), like stock")
    r.add_argument("--no-warm", action="store_true", help="no discarded run before the measured one: the first run as a user sees it")
    r.add_argument("--args", help="extra server args, e.g. '-fitt 8000'")
    add_overrides(r)
    b = sp.add_parser("bench", help="MODELS x BUILDS x VARIANTS, every cell recorded, then the report")
    b.add_argument("--models", required=True, help="comma separated model paths (on the machine that runs them)")
    b.add_argument("--builds", required=True, help="comma separated build dirs; names starting with stock run the default variant only")
    b.add_argument("--variants", help="comma separated variant names (default: all of default,cache0,atoff and any --variant)")
    b.add_argument("--variant", action="append", default=[], help="NAME=ARGS: add or override a variant, e.g. gate3='--moe gate=3'")
    b.add_argument("-t", "--test", default="t100", choices=["ppl", "ppl3", "t100", "tetris"] + TESTS[4:])
    b.add_argument("--stock-variant", action="append", default=[], help="NAME=ARGS: stock builds also run this placement (e.g. tuned='--n-cpu-moe 30'): the hand-tuned baseline")
    b.add_argument("-n", type=int, help="measured runs per cell, interleaved ABBA (default 3 for fix/fix12k, else 2); the first run of a cell is preceded by a discarded one")
    b.add_argument("--machine", default="pc1", help="pc1 = this host (server tests); other ids from 'run.py machine' run llama-cli over ssh")
    b.add_argument("--campaign", help="name in table campaigns (default bench-<date>)")
    b.add_argument("--deadline", help="HH:MM: cells that would start later are skipped")
    b.add_argument("--last", type=int, help="report the last N runs per cell only (hot runs of a learning autotune)")
    b.add_argument("--tol", type=float, help="flag cells below TOL x stock as REGRESSION (e.g. 0.95)")
    b.add_argument("--note")
    b.add_argument("--no-warm", action="store_true", help="no discarded run before each measured one: the first run as a user sees it")
    add_overrides(b)
    rp = sp.add_parser("report", help="variants against stock per model / test / machine")
    rp.add_argument("--campaign")
    rp.add_argument("--since", help="'YYYY-MM-DD HH:MM'")
    rp.add_argument("--last", type=int)
    rp.add_argument("--tol", type=float)
    rp.add_argument("--md", action="store_true", help="also print README table rows (median, range, run ids)")
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
    s_ = sp.add_parser("show")
    s_.add_argument("-t", "--test")
    s_.add_argument("-b", "--build")
    s_.add_argument("--runs", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "bench": cmd_bench, "report": cmd_report, "machine": cmd_machine, "prompt": cmd_prompt, "ctl": cmd_ctl,
     "show": cmd_show}[a.cmd](a)
