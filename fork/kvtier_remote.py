#!/usr/bin/env python3
"""KV tier ladder (--moe kv-fallback) on a remote GPU host: one server per list, requests of growing length, what each did.
usage: kvtier_remote.py w10 BUILD [sizes ...]
  w10: Windows laptop (RTX 4060), build N:\\build\\BUILD, OLMoE Q4_K_M with the context-length override so the model's cap does not hide the ladder.
Per list (default, k8v8, k4v4, ram, none, and the typo RAM,q8): a fresh server, requests of the given token counts in ascending order, then a short one (back down).
Checks per request: a refusal must be a "fit no enabled tier" refusal naming the list; an accepted one must have been served by an allowed tier (the server's kv-fit line);
the answer must be text (no '!!!!', no single repeated character, mostly printable). Output: /p/bw/data/q5h/kvtier-<host>.txt (one block per list, a summary table at the end)."""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error

HOSTS = {"pc2": dict(ssh="o@192.168.1.12", ip="192.168.1.12", root="/home/o", model="/home/o/m/OLMoE-1B-7B-0924-Q4_K_M.gguf", tmp="/home/o/kvt", exe="llama-server", linux=True),
         "w10": dict(ssh="o@192.168.1.2", ip="192.168.1.2", root="N:\\build", model="N:\\m\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", tmp="N:\\tmp", exe="llama-server.exe")}
Q = "/p/bw/data/q5h"
PROMPT = open(f"{Q}/prompt_real_262k.txt", errors="replace").read()
LISTS = [("default", None), ("k8v8", "k8v8"), ("k4v4", "k4v4"), ("ram", "ram"), ("none", "none"), ("typo", "RAM,q8")]
TIER_OF = {"f16": 0, "q8_0/q8_0": 1, "q4_0/q4_0": 2}


def sh(h, cmd, timeout=120):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", h["ssh"], cmd], capture_output=True, text=True, errors="replace", timeout=timeout).stdout


def killall(h):
    sh(h, "pkill -x llama-server; true" if h.get("linux") else "taskkill /f /im llama-server.exe >nul 2>&1")


def start(h, build, lst, port=8099):
    killall(h)
    time.sleep(3)
    if h.get("linux"):   # CPU only: no pool, so no tiers to climb: this checks that the list parses, the server serves long requests and nothing breaks
        moe = "state=0" + (f",kv-fallback={lst}" if lst else "")
        cmd = (f"mkdir -p {h['tmp']}; cd {h['tmp']}; nohup env LD_LIBRARY_PATH={h['root']}/{build}/build/bin {h['root']}/{build}/build/bin/{h['exe']} -m {h['model']} --host 0.0.0.0 --port {port} -np 1 "
               f"--override-kv olmoe.context_length=int:262144 --moe {moe} > {h['tmp']}/kvt.log 2>&1 < /dev/null & echo started")
        sh(h, cmd)
        for _ in range(120):
            try:
                if urllib.request.urlopen(f"http://{h['ip']}:{port}/health", timeout=3).status == 200:
                    return True
            except Exception:
                time.sleep(2)
        return False
    moe = "state=0,mode=cache,sim-vram=3500" + (f",kv-fallback={lst}" if lst else "")
    bat = (f"@echo off\r\ncd /d {h['tmp']}\r\n{h['root']}\\{build}\\{h['exe']} -m {h['model']} --host 0.0.0.0 --port {port} -np 1 "
           f"--override-kv olmoe.context_length=int:262144 --moe {moe} > {h['tmp']}\\kvt.log 2>&1\r\n")
    open("/tmp/kvt-start.bat", "w", newline="").write(bat)
    subprocess.run(["scp", "-q", "/tmp/kvt-start.bat", f"{h['ssh']}:/{h['tmp'].replace(chr(92), '/')}/kvt-start.bat"], check=True)
    sh(h, f'wmic process call create "cmd /c {h["tmp"]}\\kvt-start.bat"')
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"http://{h['ip']}:{port}/health", timeout=3).status == 200:
                return True
        except Exception:
            time.sleep(2)
    return False


def log(h):
    return sh(h, f"cat {h['tmp']}/kvt.log" if h.get("linux") else f"type {h['tmp']}\\kvt.log")


def request(h, n_tok, port=8099):
    text = PROMPT[: max(16, int(n_tok * 3.2))]
    body = json.dumps({"prompt": text, "n_predict": 24, "temperature": 0, "cache_prompt": False}).encode()
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(f"http://{h['ip']}:{port}/completion", body, {"Content-Type": "application/json"}), timeout=900)
        d = json.loads(r.read())
        return r.status, d.get("content", ""), d.get("tokens_evaluated", 0), time.time() - t0, ""
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        return e.code, "", 0, time.time() - t0, msg[:300]
    except Exception as e:
        return 0, "", 0, time.time() - t0, f"{type(e).__name__}: {e}"


def sane(txt):
    if not txt.strip():
        return False, "empty"
    if "!!!!" in txt:
        return False, "!!!!"
    if len(set(txt.strip())) <= 2 and len(txt) > 8:
        return False, "one repeated character"
    pr = sum(c.isprintable() or c in "\n\t" for c in txt) / len(txt)
    return (pr > 0.95, f"printable {pr:.2f}")


def allowed_tiers(lst):
    if lst is None:
        return {1, 2, 3}
    if lst == "none":
        return set()
    m = {"k8v8": 1, "k4v4": 2, "ram": 3}
    return {m[w] for w in lst.split(",") if w in m}   # a word that is no tier (typo) allows nothing, and must be warned about


def run(host, build, sizes):
    h = HOSTS[host]
    out = open(f"{Q}/kvtier-{host}.txt", "w")
    rows = []
    def w(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    w(f"kvtier {host} build {build} sizes {sizes} {time.strftime('%F %T')}")
    only = os.environ.get("KVT_LISTS")
    for name, lst in LISTS:
        if only and name not in only.split(","): continue
        ok = start(h, build, lst)
        w(f"\n== list {name} ({lst}): server {'up' if ok else 'NOT UP'}")
        if not ok:
            rows.append((name, "-", "server did not start", "FAIL")); continue
        seen = len(log(h).splitlines())
        for n in list(sizes) + [500]:
            code, txt, ne, dt, err = request(h, n)
            lines = log(h).splitlines()
            new = [l for l in lines[seen:] if re.search(r"kv-fit|kv-ram|kv-tail|ERROR|GGML_ASSERT|ABORT", l)]
            seen = len(lines)
            kvline = " | ".join(re.sub(r"^\S+ [IWE] ", "", l)[:220] for l in new)
            tier = None
            for l in new:
                m = re.search(r"KV now (\S+) in place", l)
                if m: tier = TIER_OF[m.group(1)]
                if "layers to host memory" in l or "kv-ram:" in l and "moved to host" in l: tier = 3
            verdict, why = "OK", ""
            al = allowed_tiers(lst)
            if code == 200:
                s, sw = sane(txt)
                if not s: verdict, why = "FAIL", f"answer not text ({sw}): {txt[:60]!r}"
                if tier is not None and tier > 0 and tier not in al: verdict, why = "FAIL", f"served by tier {tier}, list allows {sorted(al)}"
            elif code in (400, 500, 503) and ("fit no enabled tier" in (err + kvline) or "but this machine can hold" in err):
                if "fit no enabled tier" in (err + kvline) and lst is not None and lst not in (err + kvline): verdict, why = "FAIL", "refusal does not name the list"
                if "but this machine can hold" in err and tier is not None and tier != 3: verdict, why = "FAIL", "refused by the current-tier capacity check although a tier took it: " + err[60:200]
                why = why or "refused (no enabled tier holds it)"
            else:
                verdict, why = "FAIL", f"code {code} {err[:120]}"
            if re.search(r"GGML_ASSERT|ABORT|segmentation", " ".join(new)): verdict, why = "FAIL", "server abort: " + kvline[:160]
            rows.append((name, ne or n, f"HTTP {code} tier {tier} {dt:.0f}s {why}", verdict))
            if verdict == "FAIL":
                w("  --- server log tail:\n" + "\n".join(l[:260] for l in lines[-14:]))
            w(f"  {n:>6} asked, {ne:>6} tokens: HTTP {code} {dt:5.0f}s tier {tier} {verdict} {why}\n         {kvline[:400]}\n         answer: {txt[:70]!r}")
        if "typo" in name:
            warn = [l for l in log(h).splitlines() if "not a tier" in l]
            rows.append((name, "-", f"typo warning {'seen' if warn else 'MISSING'}", "OK" if warn else "FAIL"))
            w(f"  typo warning: {warn[:1]}")
        up = sh(h, "pgrep -x llama-server" if h.get("linux") else "tasklist | findstr /i llama-server")
        if not up.strip() or (not h.get("linux") and "llama-server" not in up):
            rows.append((name, "-", "server died", "FAIL")); w("  SERVER DIED")
    killall(h)
    w("\n| list | tokens | result | verdict |\n|---|---|---|---|")
    for r in rows: w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    bad = sum(r[3] == "FAIL" for r in rows)
    w(f"\n{bad} failed of {len(rows)}")
    return bad


if __name__ == "__main__":
    host, build = sys.argv[1], sys.argv[2]
    sizes = [int(x) for x in sys.argv[3:]] or [9000, 20000, 40000, 70000]
    sys.exit(1 if run(host, build, sizes) else 0)
