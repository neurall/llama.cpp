"""experiment 8: separate read ORDER from address RANGE (streambench REGION_MB / TOTAL_GB mode: the first REGION_MB of expert bytes are read again and again as passes until TOTAL_GB were read).
perm: one fixed region, many orders, every order reads exactly the same blocks.   python3 run.py experiment exp8_workingset perm --region 1900 --orders 0,5:2,5:4,5:8,5:16,1,2,4
wset: fixed order and stride, growing region (rows named ..._r<MB>mb).        python3 run.py experiment exp8_workingset wset --regions 256,512,1024,2048,4096,8192,16384,32768 --order 5:6
Each row also stores the read latency (p50/p99 ms) in the note."""
import argparse, copy, re
from experiments import stream_run, show, LAST
from experiments.exp1_maxreads import MODEL


def one(a, region, order, chunk, qd, th):
    o, _, st = order.partition(":")
    b = copy.copy(a); b.exp = f"8_{a.mode_name}_r{region}mb"
    return stream_run(b, "100", "uring", chunk, o, qd, th, {"REGION_MB": str(region), "TOTAL_GB": a.total_gb, **({"STRIDE": st} if st else {})})


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp8_workingset")
    p.add_argument("mode_name", choices=["perm", "wset"]); p.add_argument("--region", default="1900"); p.add_argument("--regions", default="256,512,1024,2048,4096,8192,16384,32768"); p.add_argument("--orders", default="0,5:2,5:4,5:8,5:16,1,2,4"); p.add_argument("--order", default="5:6")
    p.add_argument("--chunk", default="256"); p.add_argument("--qd", default="32"); p.add_argument("--threads", default="1"); p.add_argument("--total-gb", dest="total_gb", default="12"); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--dev", default="1"); p.add_argument("--note"); p.add_argument("--bin"); p.add_argument("--file", default=MODEL); p.add_argument("--list", default="/p/bw/latest/fork/iobench/mimo_experts.txt"); p.add_argument("--layers", default="1000"); p.add_argument("--mode", default="2"); p.add_argument("--commit"); p.add_argument("--show")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    plan = [(a.region, o) for o in a.orders.split(",")] if a.mode_name == "perm" else [(r, a.order) for r in a.regions.split(",")]
    for _ in range(a.reps):   # repetitions interleaved (not one block per setting) so drift hits every setting alike
        for region, order in plan: one(a, region, order, a.chunk, a.qd, a.threads)
    show(f"experiment_8_{a.mode_name}_%")
