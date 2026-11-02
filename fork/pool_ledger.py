#!/usr/bin/env python3
"""Replay a pool write ledger (llama-server --moe ledger=1 writes <logdir>/pool-ledger-<time>.csv).

Every line is one write into a card's expert pool: t_us, card, kind, lo, hi (byte offsets in that card's pool buffer), bytes, who.
Kinds: lend (a graph's bytes from the start of the pool), release (lent blocks back to the experts), zero (a class's zero block restored),
kv-zone (the KV zone laid out), kv-move (KV rows copied to the new zone), kv-pad (pad rows zeroed), up-start / up-done (an expert upload
into one block; "bg" = background refill), keep-hot (an expert moved out of blocks a prompt borrows), mirror (a dense-split copy filled),
OVERLAP (the server's own live check fired).

What this adds to the live check: it pairs every upload's start and end and reports a lend or a KV zone that took the block's bytes
WHILE the copy ran (the live check only sees the two ends), and it answers "who wrote this byte, when" for any address.

  pool_ledger.py LEDGER.csv                    summary, the server's OVERLAP lines, upload windows crossed by a lend or a zone
  pool_ledger.py LEDGER.csv --addr CUDA0:7884.5   every write that covered that MiB offset of CUDA0's pool, in order
  pool_ledger.py LEDGER.csv --map              replay the region map (lend / zone / zero blocks, from the "map", lend and kv-zone lines) and list every
                                               write that lands in a region it does not belong to, per writer kind, with bytes
  pool_ledger.py LEDGER.csv --blame N          the Nth OVERLAP line with the writes before and after it on the same bytes (who touched them last)
  pool_ledger.py LEDGER.csv --at T_US          the lend, the zone and the uploads in flight at that time, per card
  pool_ledger.py LEDGER.csv --races            a graph's window (its lend line to its graph-done line) crossed by another thread's write into its bytes,
                                               and upload windows that overlap a graph window on the same bytes (needs the thread column, 2026-10-28)
  pool_ledger.py LEDGER.csv --dma              every copy's real window on its stream (dma-start, dma-done: pointer, length, when it was queued) against the pool map: a lend, a KV zone, KV rows
                                               moved or zeroed, a fill or blocks given back on its bytes while it ran, two copies on one byte range at once, a copy that never ended, and how far
                                               behind the host the streams run
  pool_ledger.py LEDGER.csv --stale            CUDA graph replays whose scratch range lies outside the lend of that card at that moment (an address kept
                                               from an older layout, the DSpark repeat-request bug of 2026-10-28)

Columns since 2026-10-28: t_us, thread, card, kind, rw (w write, o owner change, rw a graph's read and write, x violation), state (start / done /
once), lo, hi, bytes, who. Older files without thread / rw / state still load; --races then has no threads to tell apart.
"""
import argparse, bisect, collections, csv, sys

MIB = 1048576.0


def load(path):
    rows = []
    import gzip
    with (gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["t_us"]), r["card"], r["kind"], int(r["lo"]), int(r["hi"]), r["who"] or ""))
            except (KeyError, ValueError, TypeError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows


def windows(rows):
    """upload windows: (card, lo, hi, who, t_start, t_done)"""
    open_ = {}
    out = []
    for t, card, kind, lo, hi, who in rows:
        key = (card, lo, who.replace(" bg", ""))
        if kind == "up-start":
            open_[key] = t
        elif kind == "up-done" and key in open_:
            out.append((card, lo, hi, who, open_.pop(key), t))
    return out


def crossed(rows, wins):
    """a lend or a KV zone laid out over an upload's bytes while that upload ran"""
    by_card = collections.defaultdict(list)
    for t, card, kind, lo, hi, who in rows:
        if kind in ("lend", "kv-zone", "kv-move", "kv-pad", "zero", "mirror"):
            by_card[card].append((t, kind, lo, hi, who))
    bad = []
    for card, lo, hi, who, t0, t1 in wins:
        for t, kind, a, b, w in by_card[card]:
            if t0 < t < t1 and a < hi and b > lo:
                bad.append((t, card, kind, a, b, w, who, t0, t1))
    return bad


def load_full(path):
    """rows with the columns of 2026-10-28: (t, thread, card, kind, rw, state, lo, hi, who); thread / rw / state are '' in older files"""
    rows = []
    import gzip
    with (gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["t_us"]), r.get("thread", "") or "", r["card"], r["kind"], r.get("rw", "") or "", r.get("state", "") or "",
                             int(r["lo"]), int(r["hi"]), r["who"] or ""))
            except (KeyError, ValueError, TypeError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows


def dma_windows(full):
    """the copies as their streams ran them: (t_start, t_done, card, lo, hi, who), and the ones with no end in the file"""
    open_, out = {}, []
    for t, th, card, kind, rw, st, lo, hi, who in full:
        if kind == "dma-start":
            open_[(card, lo, hi, who)] = t
        elif kind == "dma-done" and (card, lo, hi, who) in open_:
            out.append((open_.pop((card, lo, hi, who)), t, card, lo, hi, who))
    return sorted(out), sorted((t, *k) for k, t in open_.items())


def dma_check(full, t_first, n_max):
    """what the pool did to bytes a copy was still writing; returns the number of findings"""
    wins, left = dma_windows(full)
    if not wins and not left:
        print("no dma lines: a build without ggml_cuda_dma, or no --moe ledger")
        return 0
    # lines that give bytes to something else, in time order per card; the map's block classes never change, its zone line does
    changes = collections.defaultdict(list)
    for t, th, card, kind, rw, st, lo, hi, who in full:
        if kind in ("lend", "kv-zone", "kv-move", "kv-pad", "zero", "mirror", "fill", "release", "graph-done") or (kind == "map" and "zone" in who):
            changes[card].append((t, kind, lo, hi, who))
    MS = 1e3
    late = sorted(w[0] - int(w[5].split(" queued ")[1].split()[0]) for w in wins if " queued " in w[5])
    run = sorted(w[1] - w[0] for w in wins)
    print(f"{len(wins)} copies, {sum(w[4] - w[3] for w in wins) / MIB:.1f} MiB, {len(left)} never ended")
    if late:
        print(f"queued to started: median {late[len(late) // 2] / MS:.2f} ms, worst {late[-1] / MS:.2f} ms; started to done: median {run[len(run) // 2] / MS:.2f} ms, worst {run[-1] / MS:.2f} ms")
    found = 0
    def show(line):
        nonlocal found
        found += 1
        if found <= n_max:
            print(line)
    for t0, t1, card, lo, hi, who in wins:
        ch = changes[card]
        for t, kind, a, b, w in ch[bisect.bisect_left(ch, (t0,)):bisect.bisect_right(ch, (t1 + 1,))]:
            if a < hi and b > lo and kind != "graph-done":
                show(f"UNDER COPY {(t - t_first) / 1e6:10.3f}s {card}: {kind} [{a / MIB:.3f}, {b / MIB:.3f}) MiB {w[:70]}  x  copy [{lo / MIB:.3f}, {hi / MIB:.3f}) MiB {(t1 - t0) / MS:.2f} ms {who[:64]}")
    by_card = collections.defaultdict(list)
    for w in wins:
        by_card[w[2]].append(w)
    for card, ws in by_card.items():
        live = []
        for w in ws:
            live = [x for x in live if x[1] > w[0]]
            for x in live:
                if x[3] < w[4] and x[4] > w[3]:
                    show(f"TWO COPIES {(w[0] - t_first) / 1e6:10.3f}s {card}: [{w[3] / MIB:.3f}, {w[4] / MIB:.3f}) {w[5][:56]}  x  [{x[3] / MIB:.3f}, {x[4] / MIB:.3f}) {x[5][:56]}")
            live.append(w)
    # the regions at each copy's start and end: an expert block's copy inside the lend or the KV zone of that moment
    state = collections.defaultdict(dict)
    ev = sorted([(w[0], 0, w) for w in wins] + [(w[1], 1, w) for w in wins], key=lambda x: (x[0], x[1]))
    i = 0
    rows = sorted(((t, card, kind, lo, hi) for t, th, card, kind, rw, st, lo, hi, who in full if kind in ("lend", "kv-zone") or (kind == "map" and "zone" in who)), key=lambda x: x[0])
    for t, end, w in ev:
        while i < len(rows) and rows[i][0] <= t:
            st_ = state[rows[i][1]]
            st_["lend" if rows[i][2] == "lend" else "zone"] = (rows[i][3], rows[i][4])
            i += 1
        if "queued" in w[5] and (" cache" in w[5] or w[5].startswith("cache") or "expert" in w[5] or "_c " in w[5]):
            for name, (a, b) in state[w[2]].items():
                if a < w[4] and b > w[3]:
                    show(f"IN {name.upper()} {(t - t_first) / 1e6:10.3f}s {w[2]}: copy [{w[3] / MIB:.3f}, {w[4] / MIB:.3f}) MiB {'ended' if end else 'started'} inside [{a / MIB:.3f}, {b / MIB:.3f}) {w[5][:64]}")
    names = collections.Counter(w[5].split(" queued ")[0].split(".")[0] for w in wins)
    print("copies by tensor:", ", ".join(f"{n} {c}" for n, c in names.most_common(12)))
    print(f"findings {found}")
    return found


def graph_windows(full):
    """per card: (t_start, t_end, lo, hi, thread, who) from a lend line to the next graph-done line of that card"""
    out, open_ = [], {}
    for t, th, card, kind, rw, st, lo, hi, who in full:
        if kind == "lend":
            if card in open_:   # a lend without a graph-done before the next one: close at the next lend
                t0, l0, h0, th0, w0 = open_.pop(card)
                out.append((t0, t, l0, h0, th0, w0, card))
            open_[card] = (t, lo, hi, th, who)
        elif kind == "graph-done" and card in open_:
            t0, l0, h0, th0, w0 = open_.pop(card)
            out.append((t0, t, l0, h0, th0, w0, card))
    return out


def races(full):
    """another thread's write into a graph's bytes inside its window; an upload window overlapping a graph window on the same bytes"""
    gw = graph_windows(full)
    by_card = collections.defaultdict(list)
    for g in gw:
        by_card[g[6]].append(g)
    bad = []
    up_open = {}
    for t, th, card, kind, rw, st, lo, hi, who in full:
        if kind == "up-start":
            up_open[(card, lo, who.replace(" bg", ""))] = (t, th)
            continue
        if kind == "up-done":
            k = (card, lo, who.replace(" bg", ""))
            if k in up_open:
                t0, th0 = up_open.pop(k)
                for g0, g1, a, b, gth, gwho, _ in by_card[card]:
                    if t0 < g1 and t > g0 and lo < b and hi > a:
                        bad.append((max(t0, g0), card, f"upload {who} ({th0}) running {t0}-{t}", f"graph window {g0}-{g1} [{a}, {b}) {gwho}"))
            continue
        if rw != "w" or not th:
            continue
        for g0, g1, a, b, gth, gwho, _ in by_card[card]:
            if g0 < t < g1 and lo < b and hi > a and th != gth:
                bad.append((t, card, f"{kind} by {th} [{lo}, {hi}) {who}", f"graph window {g0}-{g1} of {gth} [{a}, {b}) {gwho}"))
    return bad


def stale(full):
    """cuda-replay lines whose range is not inside the lend of that card at that time"""
    lend, bad = {}, []
    for t, th, card, kind, rw, st, lo, hi, who in full:
        if kind == "lend":
            lend[card] = (lo, hi)
        elif kind == "cuda-replay" and card in lend:
            a, b = lend[card]
            if lo < a or hi > b:
                bad.append((t, card, lo, hi, who, a, b))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--addr", help="CARD:MIB, e.g. CUDA0:7884.5")
    ap.add_argument("--at", type=int, help="a time stamp (t_us of the file)")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--blame", type=int)
    ap.add_argument("--max", type=int, default=40, help="lines per list")
    ap.add_argument("--races", action="store_true")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--dma", action="store_true")
    a = ap.parse_args()
    rows = load(a.ledger)
    if not rows:
        sys.exit("no rows")
    t_first = rows[0][0]
    if a.dma:
        sys.exit(1 if dma_check(load_full(a.ledger), t_first, a.max) else 0)
    if a.races or a.stale:
        full = load_full(a.ledger)
        found = races(full) if a.races else []
        for t, card, what, against in found[: a.max]:
            print(f"RACE {(t - t_first) / 1e6:10.3f}s {card}: {what}  x  {against}")
        st_ = stale(full) if a.stale else []
        for t, card, lo, hi, who, la, lb in st_[: a.max]:
            print(f"STALE {(t - t_first) / 1e6:10.3f}s {card}: replay [{lo / MIB:.3f}, {hi / MIB:.3f}) MiB outside the lend [{la / MIB:.3f}, {lb / MIB:.3f}) MiB  {who}")
        print(f"races {len(found)}, stale replays {len(st_)}")
        sys.exit(1 if found or st_ else 0)
    if a.blame is not None:
        ov = [r for r in rows if r[2] == "OVERLAP"]
        if not 0 < a.blame <= len(ov):
            sys.exit(f"{len(ov)} overlaps")
        t0, card, _, lo, hi, who = ov[a.blame - 1]
        print(f"OVERLAP {a.blame}: {(t0 - t_first) / 1e6:.3f}s {card} [{lo / MIB:.3f}, {hi / MIB:.3f}) MiB  {who}\n")
        hit = [r for r in rows if r[1] == card and r[2] not in ("OVERLAP", "map") and r[3] < hi and r[4] > lo]
        i = max(range(len(hit)), key=lambda k: -abs(hit[k][0] - t0))
        for t, c, kind, l, h, w in [r for r in hit if abs(r[0] - t0) < 2_000_000][-24:]:
            print(f"{(t - t0) / 1e3:+10.2f}ms {kind:9s} [{l / MIB:10.3f}, {h / MIB:10.3f}) MiB {(h - l) / 1024:9.1f} KiB  {w}")
        return
    if a.map:
        state = collections.defaultdict(dict)   # card -> {region name: (lo, hi)}
        bad = collections.Counter()
        first = {}
        mapped = collections.defaultdict(int)
        for t, card, kind, lo, hi, who in rows:
            st = state[card]
            if kind == "map":
                st[who.split(" ", 1)[1]] = (lo, hi)
                mapped[card] += 1
                continue
            if kind == "lend":
                st["lend"] = (lo, hi)
            if kind == "kv-zone":
                st["zone class -1"] = (lo, hi)
            if kind in ("OVERLAP", "lend", "release", "kv-move", "kv-pad", "kv-zone"):
                continue
            for name, (a_, b_) in st.items():
                is_zone, is_zero = name.startswith("zone"), " zero " in " " + name + " " or name.startswith("zero")
                if not (lo < b_ and hi > a_):
                    continue
                # expert uploads / keep-hot belong to blocks only; mirror and graph writes belong to the lend; zero restore to the zero block
                if (kind in ("up-start", "up-done", "keep-hot") and (is_zone or is_zero)) or (kind == "zero" and is_zone):
                    bad[(card, kind, name)] += 1
                    first.setdefault((card, kind, name), (t, lo, hi, who))
        print("map lines replayed:", dict(mapped) or "none (build without the live map?)")
        print(f"writes into a region they do not belong to: {sum(bad.values())}")
        for (card, kind, name), n in bad.most_common(a.max):
            t, lo, hi, who = first[(card, kind, name)]
            print(f"  {card} {kind:9s} into {name:16s} x{n}; first {(t - t_first) / 1e6:.3f}s [{lo / MIB:.3f}, {hi / MIB:.3f}) MiB {who}")
        sys.exit(1 if bad else 0)
    if a.addr:
        card, mib = a.addr.split(":")
        x = int(float(mib) * MIB)
        for t, c, kind, lo, hi, who in rows:
            if c == card and lo <= x < hi:
                print(f"{(t - t_first) / 1e6:10.3f}s {kind:9s} [{lo / MIB:10.3f}, {hi / MIB:10.3f}) MiB  {who}")
        return
    if a.at is not None:
        for card in sorted({r[1] for r in rows}):
            lend = zone = None
            for t, c, kind, lo, hi, who in rows:
                if t > a.at:
                    break
                if c != card:
                    continue
                if kind == "lend":
                    lend = (lo, hi, who)
                if kind == "kv-zone":
                    zone = (lo, hi, who)
            fl = [w for w in windows(rows) if w[0] == card and w[4] <= a.at <= w[5]]
            print(f"{card}: lend {'-' if not lend else f'[{lend[0] / MIB:.1f}, {lend[1] / MIB:.1f}) MiB {lend[2]}'}")
            print(f"{card}: zone {'-' if not zone else f'[{zone[0] / MIB:.1f}, {zone[1] / MIB:.1f}) MiB {zone[2]}'}")
            for w in fl[: a.max]:
                print(f"{card}: upload in flight [{w[1] / MIB:.3f}, {w[2] / MIB:.3f}) MiB {w[3]}")
        return
    kinds = collections.Counter(r[2] for r in rows)
    span = (rows[-1][0] - t_first) / 1e6
    print(f"{len(rows)} writes over {span:.1f} s: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))
    ov = [r for r in rows if r[2] == "OVERLAP"]
    print(f"\nOVERLAP lines from the server: {len(ov)}")
    for t, card, kind, lo, hi, who in ov[: a.max]:
        print(f"  {(t - t_first) / 1e6:10.3f}s {card} [{lo / MIB:.3f}, {hi / MIB:.3f}) MiB  {who}")
    wins = windows(rows)
    bad = crossed(rows, wins)
    print(f"\nuploads: {len(wins)} complete windows; crossed by a lend / zone / move / pad / zero / mirror while running: {len(bad)}")
    for t, card, kind, lo, hi, w, who, t0, t1 in bad[: a.max]:
        print(f"  {(t - t_first) / 1e6:10.3f}s {card} {kind} [{lo / MIB:.1f}, {hi / MIB:.1f}) MiB ({w}) over upload {who} "
              f"running {(t0 - t_first) / 1e6:.3f}-{(t1 - t_first) / 1e6:.3f}s")
    sys.exit(1 if ov or bad else 0)


if __name__ == "__main__":
    main()
