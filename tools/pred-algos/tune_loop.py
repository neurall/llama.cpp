#!/usr/bin/env python3
"""Tune-and-resume driver: run a FF algo (swiglu.py / stack.py) in short bursts, checkpointing between
rounds, and adapt --lr based on the hit-rate trend (raise while improving, cut on a stall or a drop).

  tune_loop.py swiglu.py --data session --burst 10 --rounds 20 -- --lr 1e-3 --lr-decay inv-sqrt
Everything after `--` is passed through to the algo unchanged except --lr, --ckpt and --time-limit,
which this driver owns.
"""
import argparse
import re
import subprocess
import sys

import torch


def run(algo, data, extra, lr, ckpt, burst, device):
    cmd = [sys.executable, algo, "--data", data, "--device", device, "--ckpt", ckpt,
           "--time-limit", str(burst), "--lr", str(lr)] + extra
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    print(out.strip())
    hits = [float(m) for m in re.findall(r"all (\d\.\d+)", out)]
    return sum(hits) / len(hits) if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("algo")
    ap.add_argument("--data", default="simcity")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--burst", type=float, default=10, help="seconds per round")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--ckpt", default=None)
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    ckpt = a.ckpt or f"{a.algo}.{a.data}.ckpt.pt"
    lr, best, stall = a.lr0, None, 0
    for r in range(a.rounds):
        print(f"-- round {r} lr={lr:.2e} --")
        score = run(a.algo, a.data, extra, lr, ckpt, a.burst, a.device)
        if score is None:
            continue
        if best is None or score > best + 0.002:
            best, stall = score, 0
            lr *= 1.2   # improving: push a bit faster
        else:
            stall += 1
            lr *= 0.6   # stalled or worse: back off
        if stall >= 3:
            print(f"stalled {stall} rounds at best {best:.3f}, stopping")
            break
    print(f"final best avg hit: {best}")


if __name__ == "__main__":
    main()
