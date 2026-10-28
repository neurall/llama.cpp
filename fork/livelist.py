#!/usr/bin/env python3
"""livelist.py [PORT]: web view of the ACTIVE run.py run list (default port 7000, all interfaces).
The first tab (default) is the live pool view (/pool): while the page is open, every running server answers GET /moe/pool on its own port (its newest pool
map and every read and write of pool bytes since the last call, from memory; builds from 2026-10-27 on, nothing to set) and this script appends the answers to
/dev/shm (pool-snaps-<pid>.jsonl, pool-events-<pid>.jsonl, the files a server started with --moe statsjson=1 writes itself, see llama-moecache-fork.cpp); the
page draws both cards' pools and expert blocks, replays them on a time line and flashes red where a range runs past its region. The other four tabs, all with the terminal's own colours: the run-list screen (what run.py prints: per-cell lines with t/s, pp, gains over stock, plans),
the output of the current cell (<cell>.run.out) and the server log of the current cell. The status tab is `run.py status`, the table `run LIST -nrt` prints (-nrt format: every cell run today with t/s, pp, stock and gains, then the cells still to run). The screen is rebuilt like a terminal would show it
(carriage returns and cursor-up redraws of the progress bars are applied), so a finished cell keeps its result lines and the running one shows its live bar.
Reads $RUN_QDIR/run-list.json (written by run.py run) and the logs next to it; read-only, stdlib only."""
import bisect, glob, html, json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

Q = os.environ.get("RUN_QDIR") or "/p/bw/data/q5h"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
FG = {30: "#555", 31: "#e5534b", 32: "#57ab5a", 33: "#c69026", 34: "#539bf5", 35: "#b083f0", 36: "#39c5cf", 37: "#adbac7",
      90: "#768390", 91: "#ff938a", 92: "#6bc46d", 93: "#daaa3f", 94: "#6cb6ff", 95: "#dcbdfb", 96: "#56d4dd", 97: "#f0f6fc"}
BG = {40: "#22272e", 41: "#6e2b2b", 42: "#2b5a2e", 43: "#5c4a14", 44: "#1f3d66", 45: "#4b3478", 46: "#175c63", 47: "#636e7b"}
ANSI = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


def screen(raw):
    """apply \\r, \\n, cursor-up (nA) and clear-below (J) like a terminal: the lines that would be on screen now"""
    lines, row = [""], 0
    pos = 0
    for m in ANSI.finditer(raw):
        row = _put(lines, row, raw[pos:m.start()])
        pos = m.end()
        n = m.group(1)
        if m.group(2) == "A":
            row = max(0, row - int(n or 1))
        elif m.group(2) == "J":
            del lines[row + 1:]
            lines[row] = ""
        elif m.group(2) == "m":
            lines[row] += m.group(0)
        else:
            continue
    _put(lines, row, raw[pos:])
    return lines


def _put(lines, row, text):
    """write text at the row (state kept in the line list; returns the row the cursor is on). A \\r starts the line over."""
    while len(lines) <= row: lines.append("")
    for ch in re.split(r"(\r\n|\r|\n)", text):
        if ch in ("\n", "\r\n"):
            row += 1
            while len(lines) <= row: lines.append("")
        elif ch == "\r":
            lines[row] = ""
        elif ch:
            lines[row] += ch
    return row


def to_html(lines):
    out = []
    for ln in lines:
        s, pos, st = [], 0, {}
        def span(): return f'<span style="{";".join(f"{k}:{v}" for k, v in st.items())}">' if st else ""
        for m in ANSI.finditer(ln):
            seg = html.escape(ln[pos:m.start()])
            if seg: s.append(span() + seg + ("</span>" if st else ""))
            pos = m.end()
            if m.group(2) != "m": continue
            for c in (int(x) for x in (m.group(1) or "0").split(";") if x != ""):
                if c == 0: st = {}
                elif c == 1: st["font-weight"] = "bold"
                elif c == 2: st["opacity"] = ".6"
                elif c in FG: st["color"] = FG[c]
                elif c in BG: st["background"] = BG[c]
        seg = html.escape(ln[pos:])
        if seg: s.append(span() + seg + ("</span>" if st else ""))
        out.append("".join(s))
    return "\n".join(out)


def active():
    try: j = json.load(open(os.path.join(Q, "run-list.json")))
    except Exception: j = {}
    return j.get("log"), j.get("pid")


def latest(pattern):
    fs = [os.path.join(Q, f) for f in os.listdir(Q) if re.fullmatch(pattern, f)]
    return max(fs, key=os.path.getmtime) if fs else None


def tail(path, nbytes):
    with open(path, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - nbytes))
        return f.read().decode("utf-8", "replace")


_st = [0, ""]


def status():
    """`run.py status`: the active list in the -nrt table (done today, running now, queued next); cached 15 s: the call reads runs.db and the stock rows"""
    if time.time() - _st[0] < 15 and _st[1]: return _st[1]
    try:
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), "status"],
                           capture_output=True, text=True, timeout=60, env=dict(os.environ, COLUMNS="230"))
        _st[:] = [time.time(), to_html((r.stdout or r.stderr).split("\n"))]
    except Exception as e:
        _st[:] = [time.time(), "no status (" + html.escape(str(e)) + ")"]
    return _st[1]


def content(tab):
    log, pid = active()
    if tab == "status": return status()
    if tab == "auto": return json.dumps(auto_state())
    if tab == "screen":
        return to_html(screen(tail(log, 400000))[-400:]) if log and os.path.exists(log) else "no active run list"
    if tab == "out":
        f = latest(r".*\.run\.out")
        return (os.path.basename(f) + "\n\n" if f else "") + (html.escape(tail(f, 100000)) if f else "no cell output yet")
    f = latest(r".*\.server\.log")
    return (os.path.basename(f) + "\n\n" if f else "") + (to_html(tail(f, 300000).split("\n")[-1500:]) if f else "no server log")


def procs(pattern):
    """running processes whose argv contains `pattern` (a plain string): (pid, argv, stdout file or None). The `timeout` wrapper and the shell are skipped."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit(): continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
            line = " ".join(argv).strip()
            if pattern not in line or "livelist" in line or os.path.basename(argv[0]) in ("timeout", "bash", "sh"): continue
            if not any(a.endswith("run.py") for a in argv[:3]): continue
            try: fd = os.readlink(f"/proc/{d}/fd/1")
            except OSError: fd = None
            c = line.split('run.py', 1)[1].strip()
            if any(o[1] == c for o in out):   # the launcher and the detached list run share a command: keep the one that writes the run-list log
                if fd and fd.startswith(Q): out = [o for o in out if o[1] != c]
                else: continue
            out.append((int(d), line.split("run.py", 1)[1].strip(), fd if fd and fd.startswith("/") and os.path.isfile(fd) else None))
        except OSError: pass
    return out


def auto_state():
    """what is happening now: the running builds (their last progress lines), the running run lists (their screens, several can run side by side) and whether a pool trace is
 being written; the page shows the pool view as soon as one is, the activity list before"""
    parts = []
    for pid, cmd, log in procs("run.py build"):
        txt = [l for l in screen(tail(log, 60000)) if l.strip()][-6:] if log else []
        parts.append(f'<b style="color:#6cb6ff">BUILD</b> {html.escape(cmd)} (pid {pid})\n' + to_html(txt) + "\n")
    runs = procs("run.py run")
    for pid, cmd, log in runs:
        txt = [l for l in screen(tail(log, 400000)) if l.strip()][-25:] if log else []
        parts.append(f'<b style="color:#57ab5a">RUN</b> {html.escape(cmd)} (pid {pid})\n' + to_html(txt) + "\n")
    # a server that writes a pool trace and still runs counts as live even when it is silent for a while (one long prompt pass writes no snapshot): the page must not flip
    # back to the activity list in between; an ended server counts for 20 s more, so its last map stays on screen before the page goes back
    live = any(os.path.exists("/proc/" + re.search(r"snaps-(\d+)", f).group(1)) or time.time() - os.path.getmtime(f) < 20 for f in glob.glob(os.path.join(SHM, "pool-snaps-*.jsonl")))
    live = live or bool(live_servers())   # a running server that answers GET /moe/pool
    if not parts and glob.glob(os.path.join(SHM, "pool-snaps-*.jsonl")): live = True   # nothing running: the last trace stays on screen
    return {"pool": live, "html": "\n".join(parts) or "nothing running"}


# ---- live pool view: files of a server run with --moe statsjson=1 ----------------------------------------------------------------------------------
SHM = os.environ.get("POOL_LIVE_DIR") or "/dev/shm"
LINE_TN = re.compile(rb'\{"t":(\d+),"(?:n|seq)":(\d+)')


class Idx:
    """index of a growing JSON-lines file (one root per line, each starting {"t":..,"n" or "seq":..}): time, number and byte offset of every k-th line;
    extended as the file grows, started again when it shrinks (the server rotates it at 256 MB)"""
    def __init__(self, path, k):
        self.path, self.k, self.mu = path, k, threading.Lock()
        self.reset()

    def reset(self):
        self.t, self.n, self.off, self.pos, self.cnt, self.last_t, self.last_n = [], [], [], 0, 0, 0, 0

    def update(self):
        with self.mu:
            try: size = os.path.getsize(self.path)
            except OSError: return
            if size < self.pos: self.reset()
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                while self.pos < size:
                    data = f.read(min(16 << 20, size - self.pos))
                    end = data.rfind(b"\n")
                    if end < 0: break
                    off = self.pos
                    for ln in data[:end].split(b"\n"):
                        m = LINE_TN.match(ln)
                        if m:
                            self.last_t, self.last_n = int(m.group(1)), int(m.group(2))
                            if self.cnt % self.k == 0: self.t.append(self.last_t); self.n.append(self.last_n); self.off.append(off)
                        self.cnt += 1; off += len(ln) + 1
                    self.pos += end + 1
                    f.seek(self.pos)

    def scan(self, off, take, stop, cap):
        out = []
        with open(self.path, "rb") as f:
            f.seek(off)
            for ln in f:
                if not ln.endswith(b"\n"): break
                try: e = json.loads(ln)
                except ValueError: continue
                if stop(e): break
                if take(e):
                    out.append(e)
                    if len(out) >= cap: return out, True
        return out, False


_idx, _idx_mu = {}, threading.Lock()


def pool_idx(pid, kind):
    with _idx_mu:
        if (pid, kind) not in _idx: _idx[(pid, kind)] = Idx(os.path.join(SHM, f"pool-{kind}-{pid}.jsonl"), 1 if kind == "snaps" else 256)
        i = _idx[(pid, kind)]
    i.update()
    return i


def pool_runs():
    out = [{"pid": pid, "mtime": time.time(), "alive": True, "port": port} for pid, port, _ in live_servers()]   # running servers first
    seen = {r["pid"] for r in out}
    for f in os.listdir(SHM):
        m = re.fullmatch(r"pool-snaps-(\d+)\.jsonl", f)
        if m and int(m.group(1)) not in seen: out.append({"pid": int(m.group(1)), "mtime": os.path.getmtime(os.path.join(SHM, f)), "alive": os.path.exists(f"/proc/{m.group(1)}")})
    return sorted(out, key=lambda r: -r["mtime"])


# ---- the same view from a running server: GET /moe/pool -----------------------------------------------------------------------------------------------
_srv, _srv_mu, _ing = {"t": 0.0, "v": []}, threading.Lock(), {}


def http_pool(port, since):
    """GET /moe/pool?since=N of the server on that port: {"pid", "snap": newest map or None, "ev": events numbered above N, "more"}; None when it does not answer
    (stock builds, builds before 2026-10-27, a server still loading)"""
    import urllib.request
    try: return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/moe/pool?since={since}", timeout=3))
    except Exception: return None


def live_servers():
    """running llama-server processes that answer GET /moe/pool: [(pid, port, writes its own files)], looked up at most every 2 s. A call keeps the server's
    maps and events coming for 5 s (it builds them only while someone asks), so this runs only while the page is open"""
    with _srv_mu:
        if time.time() - _srv["t"] < 2: return _srv["v"]
        out = []
        for d in os.listdir("/proc"):
            if not d.isdigit(): continue
            try: argv = open(f"/proc/{d}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
            except OSError: continue
            if not argv[0].endswith("llama-server"): continue
            port = server_port(int(d))
            if port and http_pool(port, 1 << 62) is not None: out.append((int(d), port, "statsjson=1" in " ".join(argv)))
        _srv["t"], _srv["v"] = time.time(), out
        return out


def prune_dead():
    """the trace files of servers that ended, all but the newest (as the server's own sj_prune does)"""
    dead = sorted((os.path.getmtime(f), m.group(1)) for f in glob.glob(os.path.join(SHM, "pool-snaps-*.jsonl"))
                  if (m := re.search(r"snaps-(\d+)\.jsonl$", f)) and not os.path.exists("/proc/" + m.group(1)))
    for _, pid in dead[:-1]:
        for k in ("snaps", "events"):
            try: os.remove(os.path.join(SHM, f"pool-{k}-{pid}.jsonl"))
            except OSError: pass


def ingest(pid, port):
    """GET /moe/pool of a running server appended to the files a --moe statsjson=1 server writes itself, so the live view and the time line work alike for both;
    at most every 200 ms and only while the page asks"""
    st = _ing.setdefault(pid, {"t": 0.0, "n": 0, "seq": None, "mu": threading.Lock()})
    with st["mu"]:
        if time.time() - st["t"] < 0.2: return
        st["t"] = time.time()
        r = http_pool(port, st["n"])
        if not r: return
        s, ev = r.get("snap"), r.get("ev") or []
        if s and s.get("seq") != st["seq"]:
            if st["seq"] is None: prune_dead()
            st["seq"] = s.get("seq")
            with open(os.path.join(SHM, f"pool-snaps-{pid}.jsonl"), "a") as f: f.write(json.dumps(s, separators=(",", ":")) + "\n")
        if ev:
            with open(os.path.join(SHM, f"pool-events-{pid}.jsonl"), "a") as f: f.write("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in ev))
            st["n"] = ev[-1]["n"]


def server_port(pid):
    """the TCP port a server process listens on (ss -ltnp)"""
    try:
        for ln in subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=5).stdout.split("\n"):
            if f"pid={pid}," in ln: return int(ln.split()[3].rsplit(":", 1)[1])
    except Exception: pass


def progress(pid):
    """the request of that server: prompt tokens done of the whole prompt, tokens generated so far and the limit, from its /slots (None while it does not answer)"""
    import urllib.request
    port = server_port(pid)
    if not port: return {}
    try: sl = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=2))
    except Exception: return {}
    sl = sl[0] if isinstance(sl, list) and sl else {}
    nt = sl.get("next_token") or [{}]
    nt = nt[0] if isinstance(nt, list) else nt
    return {"total": sl.get("n_prompt_tokens") or 0, "done": sl.get("n_prompt_tokens_processed") or 0, "gen": nt.get("n_decoded") or sl.get("n_decoded") or 0,
            "limit": (sl.get("params") or {}).get("n_predict") or nt.get("n_remain") or 0, "busy": bool(sl.get("is_processing"))}


def pool_api(what, q):
    if what == "runs": return pool_runs()
    if what == "progress": return progress(int(q["pid"])) if q.get("pid") else {}
    runs = pool_runs()
    pid = int(q["pid"]) if q.get("pid") else (runs[0]["pid"] if runs else 0)
    if not pid: return {}
    for p, port, own in live_servers():
        if p == pid and not own: ingest(pid, port)   # a server that writes its own files (--moe statsjson=1) is read from them
    s, e = pool_idx(pid, "snaps"), pool_idx(pid, "events")
    if what == "span":
        t0 = min([x for x in (s.t[:1] + e.t[:1]) if x] or [0])
        return {"pid": pid, "t0": t0, "t1": max(s.last_t, e.last_t), "snaps": s.cnt, "events": e.cnt, "alive": os.path.exists(f"/proc/{pid}")}
    if what == "snap":   # the last snapshot at or before t (the newest without t)
        if not s.off: return {}
        i = len(s.off) - 1 if not q.get("t") else max(0, bisect.bisect_right(s.t, int(float(q["t"]))) - 1)
        with open(s.path, "rb") as f:
            f.seek(s.off[i])
            ln = f.readline()
        try: return json.loads(ln)
        except ValueError: return {}
    if what == "events":   # since=N: every event numbered above N (live); t0, t1: the events of that time window (replay)
        cap = int(q.get("cap", 30000))
        if not e.off: return {"ev": [], "more": False}
        if "since" in q:
            since = int(q["since"])
            i = max(0, bisect.bisect_right(e.n, since) - 1)
            ev, more = e.scan(e.off[i], lambda x: x.get("n", 0) > since, lambda x: False, cap)
        else:
            t0, t1 = int(float(q.get("t0", 0))), int(float(q.get("t1", 1 << 62)))
            i = max(0, bisect.bisect_right(e.t, t0) - 1)
            ev, more = e.scan(e.off[i], lambda x: t0 <= x.get("t", 0) <= t1, lambda x: x.get("t", 0) > t1, cap)
        return {"ev": ev, "more": more}
    return {}


POOL_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>pool live</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0;height:100%;background:#15181d;color:#c9d1d9;font:12px ui-monospace,Menlo,Consolas,monospace;overflow:hidden}canvas{display:block}
#tip{position:fixed;pointer-events:none;background:#000e;border:1px solid #556;padding:5px 7px;white-space:pre;display:none;z-index:2;max-width:640px;line-height:1.35}
#tl{position:fixed;left:0;right:0;bottom:0;height:36px;display:flex;align-items:center;gap:10px;padding:0 10px;background:#1c2128;border-top:1px solid #333}
#tl input{flex:1;accent-color:#e5534b}#tl button{width:70px;height:26px;background:#316dca;color:#fff;border:0;border-radius:4px;cursor:pointer;font:inherit}
#pg{position:relative;height:22px;background:#20242b;display:flex;overflow:hidden}#pgp{background:#316dca;height:100%}#pgg{background:#57ab5a;height:100%}
#pgt{position:absolute;left:8px;top:4px;color:#fff;text-shadow:0 0 3px #000;white-space:nowrap}#tlab{min-width:300px;text-align:right;white-space:nowrap}#run{white-space:nowrap;opacity:.8}</style></head><body>
<div id=pg><div id=pgp></div><div id=pgg></div><span id=pgt></span></div><canvas id=cv></canvas><div id=tip></div>
<div id=tl><span id=run>no run</span><input id=sl type=range min=0 max=10000 value=10000><span id=tlab></span><button id=pp>pause</button></div>
<script>
// Live pool view. Top: one bar per card (the pool buffer to scale: lend, KV zone, expert classes, zero blocks; under it one tick per expert block by owner,
// under that the card's other buffers, e.g. a draft's). Reads blue, writes red dotted, both fading; a range that runs past the region it starts in flashes
// red over the part past the boundary (the server's own check sends VIOLATION / OVERLAP events, flashed the same way). Below: one map per card, every expert
// block in pool order, column by column, so the pool's tail is the right edge; heat black -> yellow -> red by the expert's recent uses, white flash on eviction.
// Bottom: the time line; dragging it replays the map and the events of that moment; play runs it forward, into live at the end.
const $=id=>document.getElementById(id),cv=$('cv'),cx=cv.getContext('2d'),tip=$('tip'),sl=$('sl'),pp=$('pp'),tlab=$('tlab'),runEl=$('run');
const PG=22,TL=36,GAP=20,KEEP=4e6,FADE=1.5e6,VIOL=6e6;
let pid=0,snap=null,prev=null,span={t0:0,t1:0},live=true,playing=true,tv=0,tvAt=performance.now(),lastN=0,ev=[],drag=false,mouse=null,blk={},maxRc=1;
const cellCol=new Map(),flash=new Map();
const MiB=x=>(x/1048576).toFixed(1),GiB=x=>(x/1073741824).toFixed(2);
const SZ=(n,c)=>(n/1073741824).toFixed(n<1073741824?3:2)+' GiB = '+(100*n/c.bytes).toFixed(n/c.bytes<.1?2:1)+'%';   // size and share of the card's pool
async function J(u){try{const r=await fetch(u,{cache:'no-store'});return r.ok?await r.json():null}catch(e){return null}}
const heat=f=>{f=Math.max(0,Math.min(1,f));return f<.5?[510*f,440*f,0]:[255,220*(2-2*f),0]};
// expert blocks: each size class its own pastel hue (golden-angle steps, so neighbours differ), heat = saturation: the class's hottest block is the most saturated,
// its coldest a washed pastel (rank inside the class, so every class shows where its own hottest sit). Lent = near white, KV = green stripes, free = near black.
const hsl=(h,s,l)=>{const k=n=>(n+h/30)%12,a=s*Math.min(l,1-l),f=n=>l-a*Math.max(-1,Math.min(k(n)-3,9-k(n),1));return[255*f(0),255*f(8),255*f(4)]};
const HUE=c=>(c*137.508)%360;
const clsCol=(c,f)=>hsl(HUE(c),.12+.83*f,.83-.31*f);
const blkCol=x=>x.o===-2?[236,240,244]:x.o===-3?[46,140,70]:x.o===-1?[22,25,30]:x.e<0?[62,66,74]:clsCol(x.c,x.hc||0);
const RGN={lend:'rgba(120,170,255,.42)',copies:'rgba(170,170,200,.45)',graph:'rgba(120,170,255,.42)',scratch:'rgba(90,210,230,.5)',staging:'rgba(190,140,250,.45)',zone:'rgba(130,215,140,.42)'};
const CLS=['rgba(250,220,140,.35)','rgba(250,170,120,.35)','rgba(200,160,250,.35)','rgba(140,210,230,.35)','rgba(240,150,190,.35)'];
const READ=new Set(['read','read-wide','copy-r']),BAD=new Set(['VIOLATION','OVERLAP']);
function cards(){return snap?[...snap.cards].sort((a,b)=>((b.label||'')[0]=='p')-((a.label||'')[0]=='p')||a.name.localeCompare(b.name)):[]}
function blocksOf(c){const out=[];for(const k of [...c.classes].sort((a,b)=>a.base-b.base)){const v=k.b;for(let i=0;i<k.n;i++)out.push({c:k.c,i,lo:k.base+i*k.B,hi:k.base+(i+1)*k.B,o:v[5*i],il:v[5*i+1],e:v[5*i+2],rc:v[5*i+3],gc:v[5*i+4]})}return out}
function setSnap(s){if(!s||!s.cards)return;if(snap&&s.seq===snap.seq&&s.pid===snap.pid)return;prev=snap;snap=s;const nb={};maxRc=1;const now=performance.now();
 for(const c of s.cards){const b=blocksOf(c);nb[c.name]=b;for(const x of b)if(x.o>=0&&x.rc>maxRc)maxRc=x.rc;
  {const by={};for(const x of b)if(x.o>=0&&x.e>=0&&x.rc>0)(by[x.c]=by[x.c]||[]).push(x.rc);for(const k in by)by[k].sort((p,q)=>p-q);   // heat = rank among the used blocks of the same class
   for(const x of b){const u=by[x.c];x.hc=x.o>=0&&x.rc>0&&u&&u.length?(u.indexOf(x.rc)+u.lastIndexOf(x.rc))/2/Math.max(1,u.length-1):0}}
  const ob=prev&&prev.pid===s.pid?blk[c.name]:null;if(ob&&ob.length===b.length)for(let i=0;i<b.length;i++){const p=ob[i],q=b[i];if(p.o!==q.o||p.e!==q.e||p.il!==q.il)flash.set(c.name+':'+i,{until:now+900,evict:q.o<0||q.e<0})}}
 blk=nb}
function region(c,off){   // the region a byte is in: a part of the lend (copies, graph, scratch, staging), the KV zone, one expert block (the classes lie back to back)
 for(const r of c.regions)if(r.k!=='blocks'&&off>=r.lo&&off<r.hi)return{k:r.k,lo:r.lo,hi:r.hi};
 for(const k of c.classes){const end=k.base+k.n*k.B;if(off>=k.base&&off<end){const i=Math.floor((off-k.base)/k.B);return{k:'block',c:k.c,i,lo:k.base+i*k.B,hi:k.base+(i+1)*k.B}}}
 return{k:'none',lo:off,hi:off}}
function past(c,e){   // the part of a write outside the region it must stay in (null: inside): a lend in the lend, KV rows in the zone, an upload or a keep-hot copy in its
 // one block; the other kinds span regions by design (release, fill). The server's own checks send OVERLAP events, flashed the same way.
 if(e.buf)return null;let lo,hi;
 if(e.kind==='lend'){lo=0;hi=c.lend}else if(e.kind.startsWith('kv-')){if(!(c.kv_h>0))return null;lo=c.kv_start;hi=c.kv_end}
 else if(e.kind==='up-start'||e.kind==='up-done'||e.kind==='keep-hot'){const r=region(c,e.lo);if(r.k!=='block')return[e.lo,e.hi];lo=r.lo;hi=r.hi}else return null;
 return e.lo<lo?[e.lo,Math.min(e.hi,lo)]:e.hi>hi?[Math.max(e.lo,hi),e.hi]:null}
function layout(){const W=innerWidth,H=innerHeight-TL-PG;cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;cv.style.width=W+'px';cv.style.height=H+'px';cx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);return{W,H}}
let G={W:0,H:0};addEventListener('resize',()=>{G=layout()});G=layout();
function bar(c,y0,HB,W,t){
 const X=o=>o/c.bytes*W,l0=y0+16,l1=y0+HB-50,s0=l1,s1=y0+HB-34;
 cx.fillStyle='#20242b';cx.fillRect(0,y0,W,HB);
 const b=blk[c.name]||[];let used=0,free=0,lent=0,kv=0;for(const x of b){if(x.o>=0)used++;else if(x.o===-1)free++;else if(x.o===-2)lent++;else kv++}
 cx.fillStyle='#e6edf3';cx.font='bold 12px ui-monospace,monospace';cx.fillText(c.label||c.name,4,y0+12);
 cx.font='11px ui-monospace,monospace';cx.fillStyle='#adbac7';
 cx.fillText(`pool ${GiB(c.bytes)} GiB | lend ${MiB(c.lend)} MiB (scratch ${MiB(c.scr_last)} MiB, ${c.tok_bytes} B/token, ${c.stage} staging) | KV ${c.cells} cells, zone ${MiB(c.kv_end-c.kv_start)} MiB | blocks ${used} used ${free} free ${lent} lent ${kv} KV`,(c.label||c.name).length*8+14,y0+12);
 {// one chip per area outside the expert blocks, equal widths (they are far too small to see in the bar: graph and scratch are MiB of a 20 GiB pool): size, share of the pool and, where it has one, how full it is
  const rg=k=>c.regions.find(r=>r.k===k),ch=[];
  const add=(n,r,fill)=>{const sz=r?r.hi-r.lo:0;ch.push({n,sz,fill,col:RGN[r&&r.k]||'rgba(160,170,180,.5)'})};
  add('graph',rg('graph'),c.lend?1:0);add('scratch (now/peak '+MiB(c.scr_last)+'/'+MiB(c.scr_hwm)+' MiB)',rg('scratch'),c.scr_hwm?c.scr_last/c.scr_hwm:0);
  add('staging '+c.stage+' x '+MiB(c.slot_b)+' MiB',rg('staging'),c.stage?1:0);add('KV '+c.cells+' of '+c.kv_h+' cells',rg('zone'),c.kv_h?c.cells/c.kv_h:0);
  const cw=W/(ch.length+1);ch.forEach((q,i)=>{const x=i*cw;cx.fillStyle='#2d333b';cx.fillRect(x+1,y0+HB-18,cw-2,16);cx.fillStyle=q.col;cx.fillRect(x+1,y0+HB-18,(cw-2)*Math.min(1,q.fill),16);cx.fillStyle='#e6edf3';
   cx.save();cx.beginPath();cx.rect(x+1,y0+HB-18,cw-4,16);cx.clip();cx.fillText(q.n+(q.sz?' '+SZ(q.sz,c):' none'),x+4,y0+HB-6);cx.restore()});
  const x=ch.length*cw;cx.fillStyle='#2d333b';cx.fillRect(x+1,y0+HB-18,cw-2,16);cx.fillStyle='#8b98a5';cx.save();cx.beginPath();cx.rect(x+1,y0+HB-18,cw-4,16);cx.clip();cx.fillText('draft, dense, compute outside pool: not tracked',x+4,y0+HB-6);cx.restore()}
 for(const r of [...c.regions].sort((p,q)=>(p.k!=='blocks')-(q.k!=='blocks'))){const x1=X(r.lo),x2=Math.max(x1+(r.k==='blocks'?1:5),X(r.hi));cx.fillStyle=r.k==='blocks'?CLS[(r.c||0)%CLS.length]:(RGN[r.k]||'rgba(200,200,200,.3)');cx.fillRect(x1,l0,x2-x1,l1-l0);
  if(x2-x1>150){cx.fillStyle='#e6edf3';cx.fillText((r.k==='blocks'?'moe c'+r.c:r.k==='zone'?'kv':r.k)+' '+SZ(r.hi-r.lo,c),x1+3,l0+12)}}
 for(const x of b){const x1=X(x.lo),w=Math.max(1,X(x.hi)-x1);cx.fillStyle='rgb('+blkCol(x).map(Math.round).join(',')+')';cx.fillRect(x1,s0,w,s1-s0)}
 const ob=(snap.bufs||[]).filter(o=>o.card===c.name);let bx=0;const bw=ob.length?W/ob.length:W;
 for(const o of ob){cx.fillStyle='#2d333b';cx.fillRect(bx,s1+1,bw-2,12);cx.fillStyle='rgba(220,180,255,.6)';cx.fillRect(bx,s1+1,(bw-2)*Math.min(1,o.hi_seen/o.bytes),12);cx.fillStyle='#e6edf3';cx.fillText(o.name+' '+SZ(o.hi_seen,c)+' of '+SZ(o.bytes,c),bx+3,s1+11);bx+=bw}
 const blink=(performance.now()/150|0)%2;
 for(const e of ev){if(e.card!==c.name||e.buf)continue;const age=t-e.t;if(age<0)continue;const bad=BAD.has(e.kind),pr=past(c,e);
  if(bad||pr){if(age>VIOL)continue;const lo=bad?e.lo:pr[0],hi=bad?e.hi:pr[1],x1=X(lo),x2=Math.max(x1+3,X(hi));if(blink){cx.fillStyle='rgba(255,30,30,.85)';cx.fillRect(x1,l0,x2-x1,s1-l0)}cx.strokeStyle='#ff2020';cx.setLineDash([]);cx.lineWidth=2;cx.strokeRect(x1,l0,x2-x1,s1-l0);continue}
  if(age>FADE)continue;const a=1-age/FADE,x1=X(e.lo),x2=Math.max(x1+1,X(e.hi));cx.lineWidth=1;
  if(READ.has(e.kind)){cx.setLineDash([]);cx.strokeStyle=`rgba(90,170,255,${a})`}else{cx.setLineDash([3,2]);cx.strokeStyle=`rgba(255,70,70,${a})`}cx.strokeRect(x1+.5,l0+.5,x2-x1,s1-l0-1)}
 cx.setLineDash([]);return{y0,HB,X,c,l0,s1}}
function map(c,x0,y0,W,H,t){
 const b=blk[c.name]||[],N=b.length;cx.fillStyle='#101317';cx.fillRect(x0,y0,W,H);if(!N)return null;
 // the grid fills the whole width and height under its title line: columns of `rows` cells, each cw x ch (as square as the area allows, not whole pixels)
 const TT=16,gy=y0+TT,GH=Math.max(1,H-TT);let rows=Math.max(1,Math.round(Math.sqrt(N*GH/W)));const cols=Math.ceil(N/rows);rows=Math.ceil(N/cols);
 const cw=W/cols,ch=GH/rows,s=Math.min(cw,ch),now=performance.now();
 const P=i=>{const k=Math.floor(i/rows),r=i%rows,xa=x0+Math.floor(k*cw),ya=gy+Math.floor(r*ch);return[xa,ya,Math.max(1,x0+Math.floor((k+1)*cw)-xa-1),Math.max(1,gy+Math.floor((r+1)*ch)-ya-1)]};
 const lab=s>=16,fs=Math.max(6,Math.floor(s*.36));cx.font=fs+'px ui-monospace,monospace';cx.textAlign='center';
 for(let i=0;i<N;i++){const x=b[i],[cxp,cyp,w,h]=P(i),key=c.name+':'+i;
  const tg=blkCol(x);
  let cc=cellCol.get(key);if(!cc){cc=tg.slice();cellCol.set(key,cc)}for(let k=0;k<3;k++)cc[k]+=(tg[k]-cc[k])*.18;
  cx.fillStyle=`rgb(${cc[0]|0},${cc[1]|0},${cc[2]|0})`;cx.fillRect(cxp,cyp,w,h);if(x.o===-3&&s>=4){cx.save();cx.beginPath();cx.rect(cxp,cyp,w,h);cx.clip();cx.strokeStyle='#0d2614';cx.lineWidth=1;cx.beginPath();for(let d=-h;d<w;d+=4){cx.moveTo(cxp+d,cyp);cx.lineTo(cxp+d+h,cyp+h)}cx.stroke();cx.restore()}
  const f=flash.get(key);if(f){if(f.until<now)flash.delete(key);else{cx.strokeStyle=f.evict?'#fff':'#9cf';cx.lineWidth=2;cx.strokeRect(cxp+1,cyp+1,w-2,h-2)}}
  if(lab&&x.o>=0&&x.e>=0){cx.fillStyle=(cc[0]+cc[1]>300)?'#000':'#cfd6de';cx.fillText(x.il,cxp+w/2,cyp+h/2-1);cx.fillText(x.e,cxp+w/2,cyp+h/2+fs)}}
 cx.textAlign='left';const blink=(now/150|0)%2;
 for(const e of ev){if(e.card!==c.name||e.buf||e.kind==='read-wide')continue;const age=t-e.t;if(age<0)continue;const bad=BAD.has(e.kind),pr=past(c,e);
  if(!bad&&!pr&&age>FADE)continue;if((bad||pr)&&age>VIOL)continue;
  let lo=0,hi=N;while(lo<hi){const m=(lo+hi)>>1;if(b[m].hi<=e.lo)lo=m+1;else hi=m}
  for(let i=lo;i<N&&b[i].lo<e.hi&&i<lo+64;i++){const [cxp,cyp,w,h]=P(i);
   if(bad||pr){if(blink){cx.fillStyle='rgba(255,30,30,.8)';cx.fillRect(cxp,cyp,w,h)}continue}
   const a=1-age/FADE;cx.lineWidth=s>8?2:1;if(READ.has(e.kind)){cx.setLineDash([]);cx.strokeStyle=`rgba(90,170,255,${a})`}else{cx.setLineDash([3,2]);cx.strokeStyle=`rgba(255,70,70,${a})`}cx.strokeRect(cxp+.5,cyp+.5,w-1,h-1)}}
 cx.setLineDash([]);cx.fillStyle='#e6edf3';cx.font='bold 12px ui-monospace,monospace';{const tt=(c.label||c.name)+': '+N+' blocks, start top left, tail right';cx.fillText(tt,x0+4,y0+13);let lx=x0+12+cx.measureText(tt).width;cx.font='11px ui-monospace,monospace';
  for(const k of [...c.classes].sort((p,q)=>p.base-q.base)){for(let q=0;q<5;q++){cx.fillStyle='rgb('+clsCol(k.c,q/4).map(Math.round).join(',')+')';cx.fillRect(lx+q*7,y0+3,7,11)}cx.fillStyle='#e6edf3';cx.fillText('c'+k.c,lx+38,y0+13);lx+=38+cx.measureText('c'+k.c).width+10}
  cx.fillStyle='rgb(236,240,244)';cx.fillRect(lx,y0+3,11,11);cx.fillStyle='#e6edf3';cx.fillText('lent',lx+14,y0+13);lx+=14+cx.measureText('lent').width+10;cx.fillStyle='rgb(46,140,70)';cx.fillRect(lx,y0+3,11,11);cx.fillStyle='#e6edf3';cx.fillText('KV',lx+14,y0+13);
  lx+=36;cx.fillStyle='#8b98a5';cx.fillText('pale = cold, full colour = hottest of its class',lx,y0+13)}
 return{x0,y0:gy,cw,ch,rows,N,c}}
let hit={bars:[],maps:[]};
function draw(){requestAnimationFrame(draw);const now=performance.now();
 if(playing){tv+= (now-tvAt)*1000;if(live)tv=Math.min(tv,span.t1+5e5);else if(tv>=span.t1){live=true}}tvAt=now;
 const{W,H}=G;cx.clearRect(0,0,W,H);
 if(!snap){cx.fillStyle='#adbac7';cx.font='14px ui-monospace,monospace';cx.fillText('no pool trace yet: a server started with --moe statsjson=1 writes /dev/shm/pool-*-<pid>.jsonl (run.py: POOL_LIVE=1 adds it to every cell)',20,40);return}
 const cs=cards(),HB=20+Math.max(100,Math.min(160,Math.round(W*100/1024)));hit={bars:[],maps:[]};let y=0;
 for(const c of cs){hit.bars.push(bar(c,y,HB,W,tv));y+=HB+GAP}
 const mh=Math.floor((H-y)/Math.max(1,cs.length));for(const c of cs){hit.maps.push(map(c,0,y,W,mh-4,tv));y+=mh}
 if(!drag){sl.value=span.t1>span.t0?Math.round((tv-span.t0)/(span.t1-span.t0)*10000):10000}
 tlab.textContent=`t ${((tv-span.t0)/1e6).toFixed(3)} s of ${((span.t1-span.t0)/1e6).toFixed(1)} s | ${live?'LIVE':playing?'replay':'paused'} | events ${ev.length}`;
 if(mouse)hover()}
function hover(){const{x,y}=mouse;let s=null;const t=tv;
 for(const b of hit.bars){if(b&&y>=b.y0&&y<b.y0+b.HB){const c=b.c,off=x/G.W*c.bytes,r=region(c,off),bl=(blk[c.name]||[]).find(q=>off>=q.lo&&off<q.hi);
  s=`${c.label} (${c.name}) at ${MiB(off)} MiB\nregion ${r.k}${r.c!==undefined?' class '+r.c:''}${r.i!==undefined?' block '+r.i:''} [${MiB(r.lo)}, ${MiB(r.hi)}) MiB, ${MiB(r.hi-r.lo)} MiB`;
  if(bl)s+=`\nblock owner ${bl.o===-2?'LENT to the graph':bl.o===-3?'KV zone':bl.o===-1?'free':'layer '+bl.il+(bl.e>=0?' expert '+bl.e+'  recent uses '+bl.rc+', lifetime '+bl.gc:' (empty: upload in flight)')}`;
  s+=`\nlend ${MiB(c.lend)} MiB (mirror ${MiB(c.mirror)}, scratch last ${MiB(c.scr_last)}, high ${MiB(c.scr_hwm)}, ${c.tok_bytes} B/token, staging ${c.stage} x ${MiB(c.slot_b)} MiB)\nKV zone [${MiB(c.kv_start)}, ${MiB(c.kv_end)}) MiB: ${c.cells} cells used of ${c.kv_h}, row ${c.kv_row} B, fixed ${MiB(c.kv_fixed)} MiB`;
  const near=ev.filter(e=>e.card===c.name&&!e.buf&&e.lo<=off&&e.hi>off&&t-e.t<VIOL&&t>=e.t).slice(-8);for(const e of near)s+=`\n${((t-e.t)/1e3).toFixed(0)} ms ago ${e.kind} [${MiB(e.lo)}, ${MiB(e.hi)}) ${e.who}`}}
 for(const m of hit.maps){if(m&&y>=m.y0&&x>=m.x0){const i=Math.floor((x-m.x0)/m.cw)*m.rows+Math.floor((y-m.y0)/m.ch);const b=(blk[m.c.name]||[])[i];
  if(b&&Math.floor((y-m.y0)/m.ch)<m.rows)s=`${m.c.label} block ${b.i} of class ${b.c} at ${MiB(b.lo)} MiB\n${b.o===-2?'LENT to the graph':b.o===-3?'under the KV zone':b.o===-1?'free':'layer '+b.il+' expert '+(b.e>=0?b.e:'(empty: upload in flight)')}\nrecent uses (hot counter) ${b.rc}\nlifetime uses ${b.gc}`}}
 if(s){tip.textContent=s;tip.style.display='block';tip.style.left=Math.min(x+14,innerWidth-660)+'px';tip.style.top=Math.min(y+14,innerHeight-200)+'px'}else tip.style.display='none'}
cv.onmousemove=e=>{mouse={x:e.clientX,y:e.clientY-PG}};cv.onmouseleave=()=>{mouse=null;tip.style.display='none'};
let seekT=0;async function seek(){const t=tv;seekT=t;const[s,e]=await Promise.all([J(`/pool/snap?pid=${pid}&t=${t}`),J(`/pool/events?pid=${pid}&t0=${t-KEEP}&t1=${t+1e6}`)]);if(seekT!==t)return;setSnap(s);if(e)ev=e.ev;lastFetch=t+1e6}
sl.oninput=()=>{drag=true;live=false;playing=false;pp.textContent='play';tv=span.t0+(span.t1-span.t0)*sl.value/10000;clearTimeout(sl.tm);sl.tm=setTimeout(seek,80)};sl.onchange=()=>{drag=false};
pp.onclick=()=>{playing=!playing;pp.textContent=playing?'pause':'play';if(playing&&tv>=span.t1-2e5)live=true;if(!playing)live=false};
let lastFetch=0,progBusy=false;
async function poll(){
 const runs=await J('/pool/runs');if(runs&&runs.length&&(!pid||(live&&runs[0].pid!==pid))){pid=runs[0].pid;lastN=0;ev=[];snap=null;prev=null;blk={};cellCol.clear()}
 if(!pid){setTimeout(poll,1000);return}
 if(!progBusy){progBusy=true;(async()=>{const g=await J(`/pool/progress?pid=${pid}`);progBusy=false;   // not awaited: /slots waits for a prompt pass to end (2 s timeout here) and the maps must not wait with it; the server's /slots: while generating its n_prompt_tokens counts the generated tokens too, so the prompt is the processed count then
  if(g&&g.busy){const P=Math.max(1,g.gen>0?g.done:g.total),pd=g.gen>0?P:Math.min(g.done,P),all=P+g.gen;$('pgp').style.width=(100*pd/all)+'%';$('pgg').style.width=(100*g.gen/all)+'%';
   $('pgt').textContent=`prompt ${pd}/${P} tokens (${(100*pd/P).toFixed(0)}%)`+(g.gen?`  |  generated ${g.gen}`+(g.limit>0?` of ${g.limit} (${(100*g.gen/g.limit).toFixed(0)}%)`:'')+`  |  bar = prompt + generated, ${all} tokens`:'  |  (the server answers between passes: one big pass jumps from its start to 100%)')}
  else{$('pgp').style.width='0';$('pgg').style.width='0';$('pgt').textContent=g&&g.total?`server idle (${g.total} tokens of the last request in its cache)`:'no request'}})()}
 const sp=await J(`/pool/span?pid=${pid}`);if(sp&&sp.t1){span={t0:sp.t0,t1:sp.t1};runEl.textContent=`server pid ${pid}${sp.alive?'':' (ended)'} | ${sp.snaps} maps, ${sp.events} events`}
 if(live){const[s,e]=await Promise.all([J(`/pool/snap?pid=${pid}`),J(`/pool/events?pid=${pid}&since=${lastN}`)]);setSnap(s);
  if(e&&e.ev.length){ev.push(...e.ev);lastN=e.ev[e.ev.length-1].n}if(playing){tv=Math.max(tv,span.t1)}}
 else if(playing){const[s,e]=await Promise.all([J(`/pool/snap?pid=${pid}&t=${tv}`),J(`/pool/events?pid=${pid}&t0=${lastFetch}&t1=${tv+1e6}`)]);setSnap(s);if(e){ev.push(...e.ev);lastFetch=tv+1e6}}
 const cut=tv-Math.max(KEEP,VIOL);let k=0;while(k<ev.length&&ev[k].t<cut)k++;if(k)ev.splice(0,k);if(ev.length>60000)ev.splice(0,ev.length-60000);
 setTimeout(poll,live?300:250)}
poll();draw();
</script></body></html>"""


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>run list</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{--bg:#1c2128;--fg:#adbac7;--line:#444c56}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.25 ui-monospace,Menlo,Consolas,monospace}
nav{position:sticky;top:0;background:#22272e;border-bottom:1px solid var(--line);padding:6px 16px;display:flex;gap:14px;align-items:center}
nav a{color:var(--fg);cursor:pointer;padding:2px 8px;border-radius:4px}nav a.on{background:#316dca;color:#fff}nav span{margin-left:auto;opacity:.6}
pre{margin:0;padding:10px 16px;white-space:pre;overflow-x:auto}</style></head><body>
<nav><a id=auto>auto</a><a id=pool>pool</a><a id=status>status</a><a id=screen>run list</a><a id=out>cell output</a><a id=server>server log</a><span id=st></span></nav><pre id=t></pre>
<iframe id=pf src="about:blank" style="display:none;width:100%;height:calc(100vh - 36px);border:0"></iframe>
<script>let tab='auto';const t=document.getElementById('t'),pf=document.getElementById('pf');
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{tab=a.id;go()});
async function go(){document.querySelectorAll('nav a').forEach(a=>a.className=a.id==tab?'on':'');
 if(tab=='auto'){let a={};try{a=JSON.parse(await (await fetch('/c/auto')).text())}catch(e){}
  if(a.pool){t.style.display='none';pf.style.display='block';if(pf.src.indexOf('/pool')<0)pf.src='/pool';document.getElementById('st').textContent='auto: live pool view';return}
  t.style.display='block';pf.style.display='none';if(pf.src.indexOf('/pool')>=0)pf.src='about:blank';t.innerHTML=a.html||'';document.getElementById('st').textContent='auto: activity '+new Date().toLocaleTimeString();return}
 if(tab=='pool'){t.style.display='none';pf.style.display='block';if(pf.src.indexOf('/pool')<0)pf.src='/pool';document.getElementById('st').textContent='live pool view';return}
 t.style.display='block';pf.style.display='none';
 try{const r=await fetch('/c/'+tab);const x=await r.text();const bot=innerHeight+scrollY>=document.body.scrollHeight-40;t.innerHTML=x;if(bot)scrollTo(0,document.body.scrollHeight);
 document.getElementById('st').textContent='updated '+new Date().toLocaleTimeString()}catch(e){document.getElementById('st').textContent='offline'}}
go();setInterval(go,3000)</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/pool":
            body, ct = POOL_PAGE.encode(), "text/html; charset=utf-8"
        elif u.path.startswith("/pool/"):
            body, ct = json.dumps(pool_api(u.path[6:], {k: v[0] for k, v in parse_qs(u.query).items()})).encode(), "application/json"
        elif self.path.startswith("/c/"):
            body, ct = content(self.path[3:]).encode(), "text/html; charset=utf-8"
        else:
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    if not glob.glob(os.path.join(SHM, "pool-snaps-*.jsonl")):   # no trace at all: a demo one (pool_demo.py, the exact file format) so the page can be debugged without a run
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_demo.py"), SHM])
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
