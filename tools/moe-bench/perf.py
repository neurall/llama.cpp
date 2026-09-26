#!/usr/bin/env python3
"""Perf regression runs for the llama.cpp expert-cache fork, stored in perf.db (sqlite).

  perf.py run [-t tetris|ppl|chat] [-n N] [-e K=V ...] [--note TEXT] BUILD...
  perf.py show [-t TEST] [-b BUILD] [--runs]
  perf.py import results.txt

BUILD is a subdirectory of the current directory (or $PERF_BUILDS) holding a build's bin/
contents, e.g. b11214-a3768a8. Results go to perf.db there ($PERF_DB). MODEL=/path/model.gguf
is required; PERF_MODELS_DIR=/dir drops other *.gguf under it from RAM before measuring. Every run gets its own server or
process, a cold cache, the build's own libs (LD_LIBRARY_PATH: native builds bake their
build dir into RUNPATH) and a pinned GPU order (-dev CUDA0,CUDA1).
Tests:
  ppl     fixed text, one token per decode call (llama-perplexity -b 1 -ub 1, longsrc.cpp code,
          2 x 1024 tokens): identical input for every build, the precise decode comparator
  ppl3    same as ppl with -b 3 -ub 3: 3-token decode batches, the path MTP/speculative verify
          uses (small-batch cache chain); compare PPL with --plain for correctness
  tetris  raw completion "generate smallest html tetris game.", -c 1024, until full
  chat    /v1/chat/completions "write smallest html tetris game", 1500 tokens
  pf12k   12k-token prefill: src_12k.cpp as a raw completion prompt, 32 tokens generated,
          x16 GPU first (-dev CUDA1,CUDA0), -ub 2048 -b 2048; pf12k-stock: same, no cache flags
  agent   miniagentic ($PERF_AGENT_DIR, agentic loop): model writes a C program, 2 missing ';' and a misspelled printf are
          injected, gcc errors + full source go back for a fix, up to 3 fix rounds; prompt
          caching on (-c 16384 -ub 2048 -b 2048): long code prompts + code decode
Compare decode t/s only between runs with the same output md5 (tetris/chat): the text
decides the cache hit rate. LLAMA_MOE_CACHE_DETERMINISTIC=1 makes a build repeat itself.
"""
import argparse, hashlib, json, os, re, sqlite3, statistics, subprocess, sys, time, urllib.request

PROMPTS = os.path.dirname(os.path.realpath(__file__))           # frozen prompts live next to this script
HERE = os.path.abspath(os.environ.get("PERF_BUILDS", os.getcwd()))  # one subdirectory per build (its bin/ contents)
DB = os.environ.get("PERF_DB", os.path.join(HERE, "perf.db"))
MODEL = os.environ.get("MODEL", "")
MODELS_DIR = os.environ.get("PERF_MODELS_DIR", "")  # other models' pages are dropped from RAM before a run
PPL_TEXT = os.path.join(PROMPTS, "longsrc.cpp")  # frozen snapshot of llama.cpp common/json-schema-to-grammar.cpp (code, like agent prompts)
BARE = False
COMMON_ALL = ["-t", "6", "--cpu-moe", "-nr", "--moe-expert-cache", "-1", "-dev", "CUDA0,CUDA1"]
COMMON = COMMON_ALL
PORT = 8099
GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1}


def db():
    c = sqlite3.connect(DB)
    c.execute("""create table if not exists runs (
        id integer primary key, ts text, build text, version text, test text, env text, args text,
        tps real, pp_tps real, n_gen integer, s_per_pass real, ppl real,
        hit_rate real, hits integer, misses integer, steps integer,
        md5 text, output text, note text, ok integer)""")
    if "model" not in [r[1] for r in c.execute("pragma table_info(runs)")]:
        c.execute("alter table runs add column model text")
        c.execute("update runs set model='GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf'")
    return c


def _die_with_parent():
    # the server dies with perf.py (killed, Ctrl-C, crash): no orphan holding the port and VRAM
    import ctypes, signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def kill_leftovers():
    """Kill every llama-* process (server, perplexity, cli, ...) by process name, in a loop until
    none is left: TERM, escalating to KILL after 2 s. Name matching (no -f) never hits
    the calling shell; exiting processes (freeing ~100 GB pinned RAM) are waited for."""
    left = lambda: subprocess.run(["pgrep", "^llama-"], capture_output=True).stdout.split()
    t0 = time.time()
    while left():
        subprocess.run(["pkill", "-KILL" if time.time() - t0 > 2 else "-TERM", "^llama-"])
        time.sleep(1)


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=30)
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
        if out and max(int(x) for x in out) < 500:
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
    env = {**os.environ, **extra_env, "LD_LIBRARY_PATH": os.path.join(HERE, build), "LLAMA_MOE_CACHE_STATS": "1"}
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
        if test in ("pf12k", "pf12k-stock"):
            dev = ["-t", "6", "-dev", "CUDA1,CUDA0", "-c", "16384", "-ub", "2048", "-b", "2048"]
            args = list(extra_args) if BARE else dev + ((["--moe-expert-cache", "0"] if test == "pf12k-stock" and not build.startswith("stock") else []) if test == "pf12k-stock" or plain or build.startswith("stock") else ["--cpu-moe", "-nr", "--moe-expert-cache", "-1"]) + list(extra_args)
            res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": open(os.path.join(PROMPTS, "src_12k.cpp")).read(), "n_predict": 32,
                                    "cache_prompt": False, **GREEDY})
            t = res["timings"]
            row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
                   "note": f"prompt_n={t['prompt_n']} prompt_s={t['prompt_ms']/1e3:.1f}"}
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
        if test == "tetris":
            args = COMMON + ["-c", "1024"]
            res, logf = server_run(build, env, args, "/completion",
                                   {"prompt": "generate smallest html tetris game.", "n_predict": -1, **GREEDY})
            text = res["content"]
        else:
            args = COMMON + ["-c", "4096"]
            res, logf = server_run(build, env, args, "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content": "write smallest html tetris game"}],
                                    "max_tokens": 1500, **GREEDY})
            m = res["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
        t = res["timings"]
        log = open(logf).read()
        row = {"tps": t["predicted_per_second"], "pp_tps": t["prompt_per_second"], "n_gen": t["predicted_n"],
               "output": text, "md5": hashlib.md5(text.encode()).hexdigest()[:8]}
    row.update(cache_stats(log))
    row.update(ok=1 if row.get("tps") else 0, args=" ".join(args))
    return row


def cmd_run(a):
    if not MODEL:
        sys.exit("set MODEL=/path/to/model.gguf (first split for split models)")
    global BARE
    BARE = a.bare
    extra = dict(kv.split("=", 1) for kv in a.env)
    c = db()
    # drop other models' pages first (fadvise, no root) so they aren't evicted during the run
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
    # model fits in RAM: read it all into the page cache (fast when already cached), every run is hot.
    # bigger than RAM (MiMo): dd would leave the file's tail cached, not the used experts, so instead
    # run the test once and discard it: that caches exactly the experts the test uses
    size = sum(os.path.getsize(f) for f in cur)
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    fits = size < 0.85 * ram
    if fits:
        subprocess.run(["dd", f"if={MODEL}", "of=/dev/null", "bs=16M"], stderr=subprocess.DEVNULL)
    last = c.execute("select model from runs order by id desc limit 1").fetchone()
    # every model switch: one discarded run (dd alone left the first Qwen run after GLM 12% low)
    if not last or last[0] != os.path.basename(MODEL):
        print(f"model switch -> {os.path.basename(MODEL)}: throwaway run", flush=True)
        try:
            run_one(a.builds[0].rstrip("/"), a.test, extra, a.plain, a.args.split() if a.args else ())
        except Exception as e:
            print(f"throwaway run failed: {e}", flush=True)
    for i in range(a.n):
        for build in a.builds:
            build = build.rstrip("/")
            try:
                row = run_one(build, a.test, extra, a.plain, a.args.split() if a.args else ())
            except Exception as e:
                row = {"ok": 0, "note": str(e)[:300]}
            rec = {**extra, **({"bare": "1"} if a.bare else {}), **({"plain": "1"} if a.plain else {}), **({"args": a.args} if a.args else {})}
            row.update(model=os.path.basename(MODEL), ts=time.strftime("%F %T"), build=build, version=version(build), test=a.test,
                       env=json.dumps(rec, sort_keys=True),
                       note=" | ".join(x for x in (a.note, row.get("note")) if x) or None)
            cols = ",".join(row)
            c.execute(f"insert into runs ({cols}) values ({','.join('?' * len(row))})", list(row.values()))
            c.commit()
            print(f"{row['ts']} {build:16s} {a.test:6s} {row.get('env')} "
                  + (f"{row['tps']:6.2f} t/s" if row.get("tps") else "FAIL")
                  + (f" {row['s_per_pass']:.2f} s/pass" if row.get("s_per_pass") else "")
                  + (f" hit {row['hit_rate']}%" if row.get("hit_rate") is not None else "")
                  + (f" pp {row['pp_tps']:.1f} t/s" if row.get("pp_tps") else "")
                  + (f" md5={row['md5']}" if row.get("md5") else "") + (f" [{row['note']}]" if a.test == "agent" and row.get("note") else "") + (f" PPL {row['ppl']}" if row.get("ppl") else ""),
                  flush=True)


def cmd_show(a):
    c = db()
    q, p = "select build,test,env,tps,hit_rate,md5,s_per_pass,ts,version,model,pp_tps from runs where ok=1", []
    if a.test:
        q += " and test=?"; p.append(a.test)
    if a.build:
        q += " and build=?"; p.append(a.build)
    rows = c.execute(q + " order by id", p).fetchall()
    if a.runs:
        for r in rows:
            print(" | ".join("" if x is None else str(x) for x in r))
        return
    groups = {}
    for b, t, e, tps, hr, md5, spp, ts, v, mdl, pp in rows:
        groups.setdefault(((mdl or "?")[:12], t, b, e), []).append((tps, hr, md5, spp, pp))
    print(f"{'model':12s} {'test':11s} {'build':21s} {'env':60s} {'n':>2s} {'t/s mean':>8s} {'sd':>5s} {'s/pass':>7s} {'hit%':>6s} {'pp t/s':>7s}  md5s")
    for (m, t, b, e), g in sorted(groups.items()):
        tps = [x[0] for x in g]
        spp = [x[3] for x in g if x[3]]
        hr = [x[1] for x in g if x[1] is not None]
        md5 = sorted({x[2] for x in g if x[2]})
        pp = [x[4] for x in g if x[4]]
        print(f"{m:12s} {t:11s} {b:21s} {e[:60]:60s} {len(g):2d} {statistics.mean(tps):8.2f} "
              f"{(statistics.stdev(tps) if len(tps) > 1 else 0):5.2f} "
              f"{(statistics.mean(spp) if spp else 0):7.2f} {(statistics.mean(hr) if hr else 0):6.1f} {(statistics.mean(pp) if pp else 0):7.1f}  {','.join(md5)}")


def cmd_import(a):
    c = db()
    n = 0
    for line in open(a.file):
        m = re.match(r"(\S+) (\S+) ppl: ([\d.]+) seconds per pass \| PPL = ([\d.]+) \| hit-rate=([\d.]+)%", line)
        if m:
            spp = float(m.group(3))
            c.execute("insert into runs (ts,build,test,env,tps,s_per_pass,ppl,hit_rate,ok,note) values (?,?,?,?,?,?,?,?,1,?)",
                      (m.group(1), m.group(2), "ppl", "{}", 1024 / spp, spp, float(m.group(4)), float(m.group(5)), "imported"))
            n += 1
            continue
        m = re.match(r"(\S+) (\S+) ([\d.]+) t/s '(.*?)'(?: md5=(\w+))? hit-rate=([\d.]+)%(.*)", line)
        if m:
            ok = 0 if "INVALID" in m.group(7) else 1
            c.execute("insert into runs (ts,build,test,env,tps,hit_rate,md5,output,ok,note) values (?,?,?,?,?,?,?,?,?,?)",
                      (m.group(1), m.group(2), "tetris", "{}", float(m.group(3)), float(m.group(6)), m.group(5),
                       m.group(4), ok, ("imported" + m.group(7)).strip()))
            n += 1
    c.commit()
    print(f"imported {n} runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("builds", nargs="+")
    r.add_argument("-t", "--test", default="ppl", choices=["ppl", "ppl3", "tetris", "chat", "agent", "pf12k", "pf12k-stock"])
    r.add_argument("-n", type=int, default=1)
    r.add_argument("-e", "--env", action="append", default=[])
    r.add_argument("--note")
    r.add_argument("--bare", action="store_true", help="only -m (plus --args): the defaults a new user gets")
    r.add_argument("--plain", action="store_true", help="no cache flags (autofit), like stock")
    r.add_argument("--args", help="extra server args, e.g. '-fitt 8000'")
    s = sp.add_parser("show")
    s.add_argument("-t", "--test")
    s.add_argument("-b", "--build")
    s.add_argument("--runs", action="store_true")
    i = sp.add_parser("import")
    i.add_argument("file")
    a = ap.parse_args()
    {"run": cmd_run, "show": cmd_show, "import": cmd_import}[a.cmd](a)
