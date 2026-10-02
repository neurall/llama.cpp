#!/usr/bin/env python3
"""run.py exp SPEC.ini [--dry] [--only ARM,ARM] [--rounds N]: an experiment described in a file, every run stored like any other run.

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
prompt_file = tools/bench/prompt650.txt    ; or prompt = text
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
args = --moe cache=0
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

EXP_COLS = ["exp", "arm", "ready_s", "wall_s", "rss_kb", "major_faults", "uploads", "evictions", "up_mib", "ddr_gbs", "pcie_gbs", "ddr_util", "pcie_util",
            "cpu_util", "gpu_util", "placement", "decisions", "rungs", "held_gib", "mem_limit", "cache_state", "state", "sys"]
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
        if a.get("state") and os.path.exists(a["state"]) and not dry:
            os.remove(a["state"])
    holder = None
    if setup.get("hold_vram") and not dry:
        holder = subprocess.Popen([sys.executable, HOLD] + [f"{g}:{gib}" for g, gib in setup["hold_vram"].items()], env=base_env,
                                  stdout=subprocess.PIPE, text=True, start_new_session=True)
        for _ in range(120):
            if "holding" in (holder.stdout.readline() or ""):
                break
        print(f"holding VRAM {setup['hold_vram']}", flush=True)
    n_run = 0
    try:
        for mname, mpath in models.items():
            warm_done = set()
            for rnd in range(rounds):
                for arm in order_arms(arms, rnd, spec.get("order", "fixed")):
                    for attempt in range(1 + (spec.get("warm", 0) if arm["name"] not in warm_done else 0)):
                        measured = attempt == spec.get("warm", 0) if arm["name"] not in warm_done else True
                        n_run += 1
                        row = one_run(spec, wl, setup, arm, mname, mpath, rnd, base_env, guard_lim, dry, state_file)
                        if arm["name"] not in warm_done and spec.get("warm", 0) and attempt < spec["warm"]:
                            print(f"  (warm-up of {arm['name']} discarded)", flush=True)
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
        a += ["-f", resolve(wl.get("text_file", "tools/bench/longsrc.cpp")), "-c", str(wl.get("ctx", 512)), "--chunks", str(wl.get("chunks", 2))]
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
    cmd = [exe] + a
    if setup.get("time", True):
        cmd = ["/usr/bin/time", "-v"] + cmd
    cmd = ["timeout", str(wl.get("timeout", 1500))] + cmd
    env = {k: str(v) for k, v in arm.get("env", {}).items()}
    if arm.get("state"):   # a state file of its own: arms must not learn from each other's runs
        env["LLAMA_MOE_STATE"] = arm["state"]
    if env:
        cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + cmd
    if setup.get("mem_limit"):
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={setup['mem_limit']}", "-p", "MemorySwapMax=0"] + cmd
    return cmd


def one_run(spec, wl, setup, arm, mname, mpath, rnd, base_env, lim, dry, state_file):
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
    if setup.get("drop_cache"):
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
    r.update(wall_s=round(wall, 1), cache_state=cache_state, sys={**ms, **(sampler.summary() if sampler else {})})
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
               rungs=r.get("rungs"), held_gib=held, mem_limit=setup.get("mem_limit"), cache_state=r.get("cache_state"),
               state=json.dumps(setup.get("state")) if setup.get("state") not in (None, "keep") else "", sys=json.dumps(sysd, sort_keys=True, separators=(",", ":")))
    rs.csv_append("run-history.csv", rs.RUN_COLS, row)
    # the spec is kept next to the data, once
    sd = os.path.join(rs.DATA, "specs")
    os.makedirs(sd, exist_ok=True)
    dst = os.path.join(sd, f"{name}@{spec_sha}.ini")
    if not os.path.exists(dst):
        shutil.copy(spec["_path"], dst)
