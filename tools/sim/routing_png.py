#!/usr/bin/env python3
"""routing_png.py MODEL_STEM TAG [TAG2 ...] [-o out.png]: the routed experts of the first, a middle and the last MoE layer as a picture.

Each layer is a panel: y = expert id (YSCALE = 3 pixel rows per expert), x = generated token (one pixel column per token). A pixel is lit when that expert was
routed for that token, black otherwise. The runs (tags, e.g. edit4-0 edit4-1 edit4-2 edit4-3: the newest routing/<MODEL_STEM>.<TAG>.*.routing of each) are laid
over each other additively in red, green, blue and gray, in the order given: an expert lit in two prompts at the same token shows a mixed colour
(red+green yellow, red+blue magenta, green+blue cyan, all three white), gray adds a neutral lift. The three panels are stacked, first layer on top, separated by a
thin line. PNG, compressed; the layer numbers and prompts are printed.
usage example: routing_png.py Qwen3.6-35B-A3B-GSQ-hybrid edit4-0 edit4-1 edit4-2 edit4-3"""
import glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routing_csv as rc

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "bench", "routing")
YSCALE = 3   # pixel rows per expert: the picture is 3x as tall as the expert count
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 160, 0), (0, 200, 200)]


def main():
    args = sys.argv[1:]
    out = None
    if "-o" in args:
        i = args.index("-o"); out = args[i + 1]; del args[i:i + 2]
    stem, tags = args[0], args[1:]
    runs = []
    for t in tags:
        fs = sorted(glob.glob(os.path.join(D, f"{stem}.{t}.*.routing")), key=os.path.getmtime)
        if not fs:
            sys.exit(f"no routing file for {stem}.{t}")
        ids, bounds, layers = rc.read_bin(fs[-1])
        runs.append((t, ids.astype(np.int32), layers))
    layers = runs[0][2]
    ne = int(max(r[1].max() for r in runs)) + 1
    T = max(len(r[1]) for r in runs)
    pick = [0, len(layers) // 2, len(layers) - 1]
    panels = []
    for pi in pick:
        img = np.zeros((ne, T, 3), np.int32)
        for k, (t, ids, _) in enumerate(runs):
            col = COLORS[k % len(COLORS)]
            for x in range(len(ids)):
                e = ids[x, pi]; e = e[e >= 0]
                img[e, x] += col
        panels.append(np.repeat(np.clip(img, 0, 255).astype(np.uint8), YSCALE, axis=0))   # every expert row YSCALE pixels tall
    sep = np.full((2 * YSCALE, T, 3), 90, np.uint8)
    full = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=0)
    out = out or os.path.join(D, "png", f"{stem}.{tags[0].split('-')[0]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(full).save(out, optimize=True)
    print(f"{out}: {full.shape[1]} x {full.shape[0]} px, {os.path.getsize(out) / 1e3:.0f} KB; layers {[int(layers[p]) for p in pick]} of {len(layers)} MoE layers, {ne} experts; "
          + ", ".join(f"{t}={'red green blue gray orange cyan'.split()[k % 6]} ({len(r)} tokens)" for k, (t, r, _) in enumerate(runs)))


if __name__ == "__main__":
    main()
