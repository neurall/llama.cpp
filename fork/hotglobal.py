#!/usr/bin/env python3
"""hotglobal.py: one hot-expert map per (model, prompt), shared by every build's dev runs (user 2026-11-18: "to compare apples to apples and spot reg
instead of hot moe drift ... one global hotmoe file loaded and updated per prompt", "option to stop one run each per prompt model when hot moes stop
changing and settle", "this will finally give us if its reg or moe drift").

The engine preloads its expert cache at server start from the lifetime use counts of the model's section in the state file (hot.<layer> = uses per expert).
Those counts differ by build and by run: a clean-state cell starts from none, every other run from whatever the runs before it left. So two builds on the same
prompt start with different experts cached, and a speed difference can be the map, not the code.

run.py -gh (RUN_GLOBAL_HOT=1): before a server starts, the map of its (model, test) is written into the state file that start reads; after each run the
counts the server stored are read back, and while the map is not settled it is replaced by them. Settled = the hottest experts of the map before the run (two cuts:
10% and 35% of a layer) cover the new counts within SETTLE (0.5%) of what the new map's own hottest cover, twice in a row; a settled map is frozen: every later run of that prompt and model, on any
build, starts from the same counts. --hot-settle repeats a cell until its map is settled. Nothing here needs engine support: the hot.<layer> lines are the
state file's own, old builds and releases read them too.

Files: <RUN_HOT_DIR or /p/bw/data/hot>/<model file>/<test>.ini, line 1 "# meta <json>" (runs, changes, settled, the state section's name), then
the hot.<layer> lines. Delete a file to learn its map again.  Self-check: python3 fork/hotglobal.py --selftest"""
import os, re, json, time, math

DIR = os.environ.get("RUN_HOT_DIR", "/p/bw/data/hot")
SETTLE = float(os.environ.get("RUN_HOT_SETTLE_AT", "0.005"))   # share of the uses the last map's hottest experts may miss against the new map's own
CUTS = (0.10, 0.35)

def path(model, test):
    return os.path.join(DIR, os.path.basename(model), re.sub(r"[^A-Za-z0-9._-]", "_", test) + ".ini")

def _sections(lines):
    """[(name, first line of its body, line after its body)]"""
    heads = [(i, l.strip()[1:-1]) for i, l in enumerate(lines) if l.startswith("[") and l.strip().endswith("]")]
    return [(n, i + 1, heads[k + 1][0] if k + 1 < len(heads) else len(lines)) for k, (i, n) in enumerate(heads)]

def _find(lines, model, name=None):
    base = os.path.basename(model)
    secs = _sections(lines)
    for s in secs:
        if name and s[0] == name: return s
    for s in secs:
        if s[0].startswith(base + " "): return s   # "<model file> <bytes>": not "GLM-5.3-Flash 117..." for "GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"
    return None

def read_state(state_file, model, name=None):
    """(section name, {layer: [uses]}) of the model in a state file; (None, {}) when it has none"""
    if not state_file or not os.path.exists(state_file): return None, {}
    lines = open(state_file, errors="replace").read().split("\n")
    s = _find(lines, model, name)
    if not s: return None, {}
    hot = {}
    for l in lines[s[1]:s[2]]:
        m = re.match(r"hot\.(\d+)\s*=\s*(.*)$", l)
        if m: hot[int(m.group(1))] = [int(x) for x in m.group(2).split()]
    return s[0], hot

def write_state(state_file, name, hot):
    """the section's hot.<layer> lines replaced by `hot` (the section appended when the file has none); written beside the file and renamed over it"""
    lines = open(state_file, errors="replace").read().split("\n") if os.path.exists(state_file) else []
    new = [f"hot.{l} = " + " ".join(map(str, hot[l])) for l in sorted(hot)]
    s = next((x for x in _sections(lines) if x[0] == name), None)
    if s:
        body = [l for l in lines[s[1]:s[2]] if not re.match(r"hot\.\d+\s*=", l)]
        while body and body[-1] == "": body.pop()
        lines = lines[:s[1]] + body + new + [""] + lines[s[2]:]
    else:
        while lines and lines[-1] == "": lines.pop()
        lines += ([""] if lines else []) + [f"[{name}]"] + new + [""]
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    tmp = state_file + ".hotglobal.tmp"
    open(tmp, "w").write("\n".join(lines))
    os.replace(tmp, state_file)

def load(model, test):
    p = path(model, test)
    if not os.path.exists(p): return None
    meta, hot = {}, {}
    for l in open(p):
        if l.startswith("# meta "): meta = json.loads(l[7:])
        m = re.match(r"hot\.(\d+)\s*=\s*(.*)$", l)
        if m: hot[int(m.group(1))] = [int(x) for x in m.group(2).split()]
    return {"meta": meta, "hot": hot} if hot else None

def save(model, test, meta, hot):
    p = path(model, test); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p + ".tmp", "w").write("# meta " + json.dumps(meta) + "\n" + "\n".join(f"hot.{l} = " + " ".join(map(str, hot[l])) for l in sorted(hot)) + "\n")
    os.replace(p + ".tmp", p)

def change(a, b):
    """what starting from map a instead of map b costs: over the layers the mean share of b's uses that b's own hottest experts cover and a's hottest do not,
    the larger of the cuts. Experts of about equal use trading places at the cut cost nothing, as in the cache; a layer a does not know costs everything"""
    worst = 0.0
    for cut in CUTS:
        tot, n = 0.0, 0
        for l, cb in b.items():
            ca = a.get(l); n += 1
            if not ca or len(ca) != len(cb) or sum(cb) == 0: tot += 1.0 if sum(cb) else 0.0; continue
            k = max(1, math.ceil(cut * len(cb)))
            top = lambda c: sorted(range(len(c)), key=lambda e: (-c[e], e))[:k]
            own, old = sum(cb[e] for e in top(cb)), sum(cb[e] for e in top(ca))
            tot += (own - old) / own if own else 0.0
        worst = max(worst, tot / max(1, n))
    return worst

def is_settled(model, test):
    g = load(model, test)
    return bool(g and g["meta"].get("settled"))

def note(model, test):
    """what a row records of the map its server started from"""
    g = load(model, test)
    if not g: return "none"
    ch = g["meta"].get("changes") or []
    return f"{os.path.basename(path(model, test))} runs={g['meta'].get('runs', 0)} {'settled' if g['meta'].get('settled') else 'learning'}" + (f" last={100 * ch[-1]:.1f}%" if ch else "")

def before_start(state_file, model, test):
    """puts the (model, test) map into the state file the server is about to read; the line for the run log"""
    g = load(model, test)
    if not g: return f"hot map {test}: none yet, this start reads the state's own counts; its run starts the map"
    name = g["meta"].get("section") or read_state(state_file, model)[0]
    if not name: return f"hot map {test}: no section name known for {os.path.basename(model)}: not written"
    write_state(state_file, name, g["hot"])
    return f"hot map {test}: {len(g['hot'])} layers written into {state_file} ({note(model, test)})"

def after_run(state_file, model, test):
    """reads the counts the server stored and, while the map is not settled, replaces the map by them; the line for the run log"""
    g = load(model, test)
    if g and g["meta"].get("settled"): return f"hot map {test}: settled after {g['meta'].get('runs', 0)} runs, frozen"
    name, hot = read_state(state_file, model, g["meta"].get("section") if g else None)
    if not hot: return f"hot map {test}: the state file holds no counts for {os.path.basename(model)} yet"
    meta = dict(g["meta"]) if g else {"runs": 0, "changes": []}
    ch = change(g["hot"], hot) if g else 1.0
    meta["runs"] = meta.get("runs", 0) + 1
    meta["changes"] = (meta.get("changes") or [])[-15:] + [round(ch, 4)]
    meta["section"] = name
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["settled"] = len(meta["changes"]) >= 3 and all(c < SETTLE for c in meta["changes"][-2:])
    save(model, test, meta, hot)
    hist = " ".join(f"{100 * c:.2f}%" for c in meta["changes"][-4:])
    return f"hot map {test}: run {meta['runs']}, the map before it misses {100 * ch:.2f}% of the uses its update covers (last: {hist}){': settled, frozen from now' if meta['settled'] else ''}"

def _selftest():
    import tempfile, random
    global DIR
    d = tempfile.mkdtemp(); DIR = os.path.join(d, "hot")
    st = os.path.join(d, "state.ini"); model = "/m/x/Model-A.gguf"; sec = "Model-A.gguf 123"
    open(st, "w").write("[Model-A 99]\nhot.1 = 9 9 9 9\n\n[gpu 0]\nx = 1\n")
    assert read_state(st, model) == (None, {}), "the shorter name's section is another model's"
    assert before_start(st, model, "t1").endswith("starts the map")
    rnd = random.Random(1); base = [[1000 - 10 * e + rnd.randint(0, 5) for e in range(64)] for _ in range(3)]
    def server_run(k):   # what a server leaves: its section with the counts so far plus this run's, other keys kept
        name, hot = read_state(st, model, sec)
        hot = {l + 1: [(hot.get(l + 1) or [0] * 64)[e] + base[l][e] + rnd.randint(0, 40 if k == 0 else 3) for e in range(64)] for l in range(3)}
        write_state(st, sec, hot)
        lines = open(st).read().split("\n"); i = lines.index(f"[{sec}]"); lines.insert(i + 1, f"place.k{k} = 1"); open(st, "w").write("\n".join(lines))
    notes = []
    for k in range(8):
        before_start(st, model, "t1"); server_run(k); notes.append(after_run(st, model, "t1"))
        if is_settled(model, "t1"): break
    assert is_settled(model, "t1") and 3 <= len(notes) <= 6, notes
    frozen = load(model, "t1")["hot"]
    server_run(9); assert "frozen" in after_run(st, model, "t1") and load(model, "t1")["hot"] == frozen, "a settled map never changes"
    st2 = os.path.join(d, "clean.ini")   # a clean-state cell: no file yet
    assert "written" in before_start(st2, model, "t1") and read_state(st2, model) == (sec, frozen), "the map reaches an empty state under the section's own name"
    txt = open(st).read()
    assert "[Model-A 99]\nhot.1 = 9 9 9 9" in txt and "[gpu 0]\nx = 1" in txt and "place.k0 = 1" in txt, "other sections and keys stay as they were"
    assert change({1: [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]}, {1: [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]}) == 0.0
    assert change({1: [9, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, {1: [0, 9, 0, 0, 0, 0, 0, 0, 0, 0]}) == 1.0
    assert change({1: [5, 5, 4, 0, 0, 0, 0, 0, 0, 0]}, {1: [5, 4, 5, 0, 0, 0, 0, 0, 0, 0]}) < 0.11, "equal experts trading places at the cut cost next to nothing"
    assert note(model, "t1").startswith("t1.ini runs=") and "settled" in note(model, "t1") and note(model, "none") == "none"
    print("hotglobal: all passed")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: _selftest()
    else:
        for root, _, files in sorted(os.walk(DIR)):
            for f in sorted(files):
                if f.endswith(".ini"):
                    m = json.loads(open(os.path.join(root, f)).readline()[7:])
                    print(f"{os.path.basename(root)[:44]:44}  {f[:-4]:12}  runs {m.get('runs', 0):3}  {'settled ' if m.get('settled') else 'learning'}  changes " + " ".join(f"{100 * c:.1f}%" for c in (m.get('changes') or [])[-5:]))
