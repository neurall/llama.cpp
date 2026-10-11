#!/usr/bin/env python3
"""baseline.py: run our README cells (game4c, edit4c, pf12k) for every model on one build (default: the latest stock) through fork/run.py bench, read the
5 Hz GPU log of every run, and when the x4 card throttles (thermal / HW slowdown flag in >= 2 busy samples in a row, or a clock drop below 85% of its
median of the last 10 s in >= 3 busy samples in a row) lower the x4 power limit by STEP watts EVERYWHERE (live, and in the boot files named by [gpu] limits_service / limits_env
of the fork config) and run that cell again. Every run is logged by run.py as always; the verdict per cell goes to <work>/baseline.log.
usage: baseline.py [--build stock-b11552-23b0202a1] [--models glm,qn38,q36,olmoe] [--step 10] [--floor 150] [--tag base1]
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
    open(lenv, "w").write("\n".join((f"LIMIT0={w}   # GPU0 = x4 card, lowered by baseline.py after a throttle ({time.strftime('%F %T')})" if l.startswith("LIMIT0=") else l) for l in lines) + "\n")


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
