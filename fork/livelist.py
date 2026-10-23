#!/usr/bin/env python3
"""livelist.py [PORT]: web view of the ACTIVE run.py run list (default port 7001, all interfaces).
Four tabs, all with the terminal's own colours: the run-list screen (what run.py prints: per-cell lines with t/s, pp, gains over stock, plans),
the output of the current cell (<cell>.run.out) and the server log of the current cell. The status tab is the table `run.py run LIST -b BUILD --dry --status` prints (-nrt format: every cell run today with t/s, pp, stock and gains, then the cells still to run). The screen is rebuilt like a terminal would show it
(carriage returns and cursor-up redraws of the progress bars are applied), so a finished cell keeps its result lines and the running one shows its live bar.
Reads $RUN_QDIR/run-list.json (written by run.py run) and the logs next to it; read-only, stdlib only."""
import html, json, os, re, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    """the -nrt table of the active list (cached 15 s: the call reads runs.db and the stock rows)"""
    log, pid = active()
    if time.time() - _st[0] < 15 and _st[1]: return _st[1]
    try:
        a = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
        i = a.index("run"); name = a[i + 1]; b = a[a.index("-b") + 1] if "-b" in a else a[a.index("--build") + 1]
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), "run", name, "-b", b, "--dry", "--status"],
                           capture_output=True, text=True, timeout=60, env=dict(os.environ, COLUMNS="230"))
        _st[:] = [time.time(), to_html(r.stdout.split("\n"))]
    except Exception as e:
        _st[:] = [time.time(), "no active run list (" + html.escape(str(e)) + ")"]
    return _st[1]


def content(tab):
    log, pid = active()
    if tab == "status": return status()
    if tab == "screen":
        return to_html(screen(tail(log, 400000))[-400:]) if log and os.path.exists(log) else "no active run list"
    if tab == "out":
        f = latest(r".*\.run\.out")
        return (os.path.basename(f) + "\n\n" if f else "") + (html.escape(tail(f, 100000)) if f else "no cell output yet")
    f = latest(r".*\.server\.log")
    return (os.path.basename(f) + "\n\n" if f else "") + (to_html(tail(f, 300000).split("\n")[-1500:]) if f else "no server log")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>run list</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{--bg:#1c2128;--fg:#adbac7;--line:#444c56}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.25 ui-monospace,Menlo,Consolas,monospace}
nav{position:sticky;top:0;background:#22272e;border-bottom:1px solid var(--line);padding:6px 16px;display:flex;gap:14px;align-items:center}
nav a{color:var(--fg);cursor:pointer;padding:2px 8px;border-radius:4px}nav a.on{background:#316dca;color:#fff}nav span{margin-left:auto;opacity:.6}
pre{margin:0;padding:10px 16px;white-space:pre;overflow-x:auto}</style></head><body>
<nav><a id=status>status</a><a id=screen>run list</a><a id=out>cell output</a><a id=server>server log</a><span id=st></span></nav><pre id=t></pre>
<script>let tab=localStorage.tab||'status';const t=document.getElementById('t');
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{tab=localStorage.tab=a.id;go()});
async function go(){document.querySelectorAll('nav a').forEach(a=>a.className=a.id==tab?'on':'');
 try{const r=await fetch('/c/'+tab);const x=await r.text();const bot=innerHeight+scrollY>=document.body.scrollHeight-40;t.innerHTML=x;if(bot)scrollTo(0,document.body.scrollHeight);
 document.getElementById('st').textContent='updated '+new Date().toLocaleTimeString()}catch(e){document.getElementById('st').textContent='offline'}}
go();setInterval(go,3000)</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/c/"):
            body, ct = content(self.path[3:]).encode(), "text/html; charset=utf-8"
        else:
            body, ct = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
