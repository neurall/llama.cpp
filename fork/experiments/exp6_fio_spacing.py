"""experiment 6: fio reference for spaced reads: blocks of BS with SKIP bytes skipped after each (fio rw=read:SKIP), io_uring O_DIRECT, queue depth 32, on the model file (whole file range or one region).
python3 run.py experiment exp6_fio_spacing --bs 1M --skips 0,64k,128k,256k,512k,1M,2M --rw read [--offset BYTES --size 1900m]"""
import argparse, json, os, subprocess, time
from experiments import put, show
from experiments.exp1_maxreads import MODEL


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp6_fio_spacing")
    p.add_argument("--bs", default="1M"); p.add_argument("--skips", default="0,64k,128k,256k,512k,1M,2M"); p.add_argument("--rw", default="read", help="read (spaced when skip > 0) or randread"); p.add_argument("--qd", default="32"); p.add_argument("--runtime", default="10")
    p.add_argument("--offset", default="0"); p.add_argument("--size", default="40G"); p.add_argument("--file", default=MODEL); p.add_argument("--exp", default="6_fio_spacing"); p.add_argument("--note", default=""); p.add_argument("--show")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    for sk in a.skips.split(","):
        rw = a.rw if a.rw != "read" or sk == "0" else f"read:{sk}"
        r = subprocess.run(["fio", "--name=t", f"--filename={a.file}", "--readonly", "--ioengine=io_uring", "--direct=1", f"--rw={rw}", f"--bs={a.bs}", f"--iodepth={a.qd}", "--numjobs=1", f"--offset={a.offset}", f"--size={a.size}",
                            f"--runtime={a.runtime}", "--time_based", "--group_reporting", "--output-format=json"], capture_output=True, text=True)
        j = json.loads(r.stdout)["jobs"][0]["read"]; gbps = j["bw_bytes"] / 1e9
        name = f"experiment_{a.exp}_{a.bs}_{a.qd}qd_io_uring_1t_{a.rw}_skip{sk}"
        put([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "fio", "name": name, "backend": "fio_io_uring", "chunk_kb": a.bs, "read_order": f"{a.rw}:{sk}", "threads": "1", "qd": a.qd, "gb": f"{j['io_bytes'] / 1e9:.2f}", "secs": a.runtime,
              "gbps": f"{gbps:.2f}", "content": "n/a", "note": a.note + f" offset={a.offset} size={a.size}", "raw": f"fio rw={rw} bs={a.bs} iodepth={a.qd} offset={a.offset} size={a.size}: {gbps:.2f} GB/s"}])
        print(f"{name}: {gbps:.2f} GB/s", flush=True)
