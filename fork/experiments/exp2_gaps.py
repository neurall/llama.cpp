"""experiment 2: gaps between the reads in flight. Pass r reads chunks r, r+S, r+2S ... of a layer (S = stride, in chunks), so the reads that are in flight together are S-1 chunks apart; the next pass fills the gaps.
S=1 is file order. The sweep shows which gap the SSD likes (flash chip / channel layout).
python3 run.py experiment exp2_gaps --chunks 256 --strides 1,2,4,8,16,32,64,128 --threads 1 --qds 32\nblock width x period grid: --chunks 32,64,128,256,512 --periods 64,128,256,512,1024,2048,4096"""
import argparse
from experiments import stream_run, show
from experiments.exp1_maxreads import MODEL


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp2_gaps")
    p.add_argument("--chunks", default="256"); p.add_argument("--strides", default="1,2,3,4,6,8,12,16,32,64,128"); p.add_argument("--gaps", help="KB list: gap between blocks; stride = 1 + round(gap/chunk), so the real gap is (stride-1) x chunk and is in the row name as stride-N"); p.add_argument("--periods", help="KB list: for every chunk (block width) the stride is period/chunk, so blocks start every PERIOD KB and the later passes fill the gaps (replaces --strides)"); p.add_argument("--qds", default="32"); p.add_argument("--threads", default="1"); p.add_argument("--picks", default="100"); p.add_argument("--layers", default="12")
    p.add_argument("--dev", default="1"); p.add_argument("--note"); p.add_argument("--exp", default="2_gaps"); p.add_argument("--bin"); p.add_argument("--file", default=MODEL); p.add_argument("--list", default="/p/bw/latest/fork/iobench/mimo_experts.txt")
    p.add_argument("--mode", default="2"); p.add_argument("--commit"); p.add_argument("--show", help="print stored rows matching this SQL LIKE pattern instead of running")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    for pk in a.picks.split(","):
        for ch in a.chunks.split(","):
            for qd in a.qds.split(","):
                for th in a.threads.split(","):
                    strides = [str(1 + round(int(g) / int(ch))) for g in a.gaps.split(",")] if a.gaps else [str(int(P) // int(ch)) for P in a.periods.split(",") if int(P) % int(ch) == 0 and int(P) >= int(ch)] if a.periods else a.strides.split(",")
                    for s in strides:
                        stream_run(a, pk, "uring", ch, "5", qd, th, {"STRIDE": s})
    show(f"experiment_{a.exp}_%")
