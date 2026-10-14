#!/usr/bin/env python3
"""What a prompt leaves of the expert pool, from server logs written with --moe log=f (pool_fill_log in src/llama-moecache-fork.cpp).

usage: pool_fill.py LOG [LOG ...]       (one server log per model / prompt; the name is the file name)

Per log and card: GB of experts held before the prompt, during it (blocks lent to the prompt's graphs), right after it (survivors), and 8 / 32 / 128 decode
steps later (refill). "kept" = held after the prompt / held before it; "refilled" = held 128 steps later / held before it.
"""
import os
import re
import sys

LINE = re.compile(r"moe-cache: pool (.+?):((?: \S+: \d+ of \d+ blocks .*?;)+)\s*$")
CARD = re.compile(r" (\S+): (\d+) of (\d+) blocks hold experts \(([\d.]+) of ([\d.]+) GB\), (\d+) uploading, (\d+) free, (\d+) lent, card ([\d.]+) GB free;")


def parse(path):
    """[(when, {card: dict})] in log order"""
    out = []
    for raw in open(path, errors="replace"):
        m = LINE.search(raw.rstrip("\n"))
        if not m:
            continue
        cards = {}
        for c in CARD.finditer(m.group(2)):
            name, held, n, gb_held, gb_all, up, fr, lent, card_free = c.groups()
            cards[name] = dict(held=int(held), n=int(n), gb_held=float(gb_held), gb_all=float(gb_all), up=int(up), free=int(fr), lent=int(lent),
                               card_free=float(card_free))
        out.append((m.group(1), cards))
    return out


def table(paths):
    rows = []
    for p in paths:
        ev = parse(p)
        before = next((c for w, c in ev if w.startswith("before a prompt")), None)
        after = next((c for w, c in ev if w.startswith("after a prompt")), None)
        later = {k: next((c for w, c in ev if w.startswith(f"{k} decode steps")), None) for k in (8, 32, 128)}
        if before is None:
            rows.append((os.path.basename(p), "-", "no pool lines (log=f off, or no prompt pass)"))
            continue
        for card in sorted(before):
            b = before[card]
            a = (after or {}).get(card)
            cells = [os.path.basename(p), card, f"{b['gb_all']:.1f}", f"{b['gb_held']:.2f}"]
            # the most blocks the prompt borrowed is not logged per pass; right after the prompt the borrowed blocks are back and free
            cells.append(f"{a['gb_held']:.2f} ({100 * a['gb_held'] / b['gb_held']:.0f}%)" if a and b['gb_held'] > 0 else "-")
            for k in (8, 32, 128):
                c = (later[k] or {}).get(card)
                cells.append(f"{c['gb_held']:.2f} ({100 * c['gb_held'] / b['gb_held']:.0f}%)" if c and b['gb_held'] > 0 else "-")
            rows.append(tuple(cells))
    hdr = ("log", "card", "pool GB", "experts before", "after the prompt", "+8 steps", "+32 steps", "+128 steps")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    table(sys.argv[1:])
