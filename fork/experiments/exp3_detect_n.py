"""experiment 3: detect the best read spacing N on this machine. Short probes (a few layers each, repeated) of streambench with stride order, 256K blocks, io_uring: for every candidate stride the median GB/s,
then the best stride and its period (stride x block). The loader can run this once per SSD and store the stride next to the model.
python3 run.py experiment exp3_detect_n [--chunk 256] [--strides 1,2,3,4,6,8,12,16,32] [--reps 3] [--layers 3] [--qd 32] [--threads 1]"""
import argparse, statistics
from experiments import stream_run, show, LAST
from experiments.exp1_maxreads import MODEL


def detect(a):
    """returns {stride: median GB/s} and the best stride (the smallest one within 3% of the best median, so a small gap wins over a huge one)"""
    res = {}
    for s in a.strides.split(","):
        v = []
        for _ in range(a.reps):
            stream_run(a, "100", "uring", a.chunk, "5", a.qd, a.threads, {"STRIDE": s})
            v += [float(r["gbps"]) for r in LAST]
        res[int(s)] = statistics.median(v)
    top = max(res.values()); best = min(s for s, g in res.items() if g >= 0.97 * top)
    return res, best


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp3_detect_n")
    p.add_argument("--chunk", default="256"); p.add_argument("--strides", default="1,2,3,4,6,8,12,16,32"); p.add_argument("--reps", type=int, default=3); p.add_argument("--layers", default="3"); p.add_argument("--qd", default="32"); p.add_argument("--threads", default="1")
    p.add_argument("--dev", default="1"); p.add_argument("--note"); p.add_argument("--exp", default="3_detect_n"); p.add_argument("--bin"); p.add_argument("--file", default=MODEL); p.add_argument("--list", default="/p/bw/latest/fork/iobench/mimo_experts.txt")
    p.add_argument("--mode", default="2"); p.add_argument("--commit"); p.add_argument("--show")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    res, best = detect(a)
    print("stride -> median GB/s:", ", ".join(f"{s}: {g:.2f}" for s, g in res.items()))
    print(f"best stride {best} chunks = period {best * int(a.chunk)}K, gap {(best - 1) * int(a.chunk)}K, {res[best]:.2f} GB/s")
