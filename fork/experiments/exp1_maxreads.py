"""experiment 1: how fast does the SSD stream the MiMo expert tensors layer after layer (up, gate, down) into VRAM when the reads of a layer are cut into chunks and issued in random order
(streambench.cu: O_DIRECT reads into pinned slots, each chunk DMA'd to the GPU). Every combination of the lists is run one after another and stored at once.
python3 run.py experiment exp1_maxreads --chunks 64,128,256 --qds 32,64 --threads 2 --picks 70,100 --backends uring,aio --orders 1"""
import argparse
from itertools import product
from experiments import stream_run, import_file, show

MODEL = "/m/m/3/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf"


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp1_maxreads")
    p.add_argument("--chunks", default="128", help="chunk KB list"); p.add_argument("--orders", default="1", help="0 file, 1 random, 2 window shuffled, 3 stride 64"); p.add_argument("--qds", default="32"); p.add_argument("--threads", default="2")
    p.add_argument("--picks", default="100", help="share of experts read per layer, percent"); p.add_argument("--backends", default="uring", help="uring, aio or pread (pread: threads are the depth, use --qds 2)")
    p.add_argument("--mode", default="2", help="0 sorted merged runs, 1 one read per expert, 2 chunks"); p.add_argument("--layers", default="12"); p.add_argument("--dev", default="1"); p.add_argument("--note"); p.add_argument("--exp", default="1_maxreads")
    p.add_argument("--bin"); p.add_argument("--file", default=MODEL); p.add_argument("--list", default="/p/bw/latest/fork/iobench/mimo_experts.txt"); p.add_argument("--import", dest="imp", help="store the lines of a result file instead of running"); p.add_argument("--commit"); p.add_argument("--show", help="print the stored rows whose name matches this SQL LIKE pattern instead of running")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    if a.imp: return import_file(a)
    n = 0
    for pk, be, ch, o, qd, th in product(a.picks.split(","), a.backends.split(","), a.chunks.split(","), a.orders.split(","), a.qds.split(","), a.threads.split(",")):
        n += stream_run(a, pk, be, ch, o, qd, th)
    print(f"{n} rows stored")
