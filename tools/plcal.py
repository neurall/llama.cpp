#!/usr/bin/env python3
"""plcal.py: find the highest power limit per GPU without throttling drops.

Each iteration runs one short, GPU-heavy request through tools/run.py bench (logged like every run, GPU log on): the pf128k test with a cut of its prompt file (about 100k tokens)
prefilled by a model that loads in seconds, both GPUs busy (split mode tensor). While it runs the 5 Hz GPU log is read live. A GPU is throttling when
  - a thermal / hardware slowdown reason (0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal slowdown) is active in >= 2 busy samples in a row, or
  - its SM clock falls below 85% of its own median of the last 10 s in >= 3 busy samples in a row (a transient drop; a steady lower clock from a cap is not a drop).
On the first throttle the run is killed, the throttling GPU's limit goes down by STEP watts, and the next iteration starts. A run that ends without a throttle
ends the search: those limits are the answer. Xid 79 in the kernel log ends everything at once.

usage: plcal.py [--start 280+280] [--step 10] [--floor 150] [--max-iter 12] [--build stock-b11552-23b0202a1] [--model PATH]
Run with the queue stopped; it takes /tmp/queue.lock for the whole search. Result lines go to /p/bw/data/q5h/plcal.log."""
import argparse, collections, csv, glob, os, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
GPULOG = os.path.join(HERE, "bench", "gpulog")
LOG = "/p/bw/data/q5h/plcal.log"
THERMAL = 0x8 | 0x20 | 0x40


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
    ap.add_argument("--prompt", default="/p/bw/data/q5h/p100k.txt")
    ap.add_argument("--build", default="stock-b11552-23b0202a1")
    ap.add_argument("--model", default="/m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf")
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
        envout = subprocess.run(["bash", "-c", ". /p/bw/data/q5h/queue/env.sh 2>/dev/null; env -0"], capture_output=True).stdout.decode(errors="replace")
        env = dict(kv.split("=", 1) for kv in envout.split("\0") if "=" in kv)
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PERF_BUILDS="/p/bw/rels/devbuilds", PERF_LOG="r", PERF_PL_LONG=spec, PERF_PL_SHORT=spec)
        cmd = ["python3", os.path.join(HERE, "run.py"), "bench", "--models", a.model, "--builds", a.build, "--variants", "v",
               "--variant=v=-sm tensor -ngl 99 -b 4096 -ub 2048 -np 1", "-t", "pf128k", "--prompt", "@" + a.prompt, "--ctx", str(a.ctx), "--tokens", "1", "-n", "1", "--no-probe", "--campaign", camp]
        t_start = time.time()
        proc = subprocess.Popen(cmd, env=env, cwd=os.path.dirname(HERE), stdout=open(f"/p/bw/data/q5h/plcal-it{it}.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
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
            log(f"it {it}: limits {spec} W: RUN FAILED after {dur:.0f} s (exit {proc.returncode}, busy samples {dict(busy)}): see /p/bw/data/q5h/plcal-it{it}.out; not a result")
            sys.exit(3)
        else:
            log(f"it {it}: limits {spec} W: CLEAN run of {dur:.0f} s (exit {proc.returncode}), busy samples {dict(busy)}")
        log(f"RESULT: no throttling drops at {spec} W (x4 + x16)")
        return
    log(f"max iterations reached, last limits {lim[0]}+{lim[1]}")


if __name__ == "__main__":
    main()
