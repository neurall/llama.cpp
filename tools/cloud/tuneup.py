#!/usr/bin/env python3
"""tuneup.py BUILD_DIR MODEL STATE_FILE [TOKENS=20000] [PORT=8101]: one server, rotating chat prompts (tools/run.py CHAT_PROMPTS), until TOKENS tokens were generated,
so the self-tuner gets through at least one cycle (a short run never does: its first knob needs ~2000 tokens). The server is stopped with SIGTERM, which saves the tuner's
decisions to STATE_FILE. Prints per-request decode speed and hit rate, then the self-tune lines of the server log (also saved next to STATE_FILE as .log)."""
import importlib.util, json, os, re, signal, subprocess, sys, time, urllib.request
b, model, state = sys.argv[1:4]; target = int(sys.argv[4]) if len(sys.argv) > 4 else 20000; port = sys.argv[5] if len(sys.argv) > 5 else "8101"
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("runpy", os.path.join(here, "..", "run.py")); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
log = state + ".log"
env = dict(os.environ, LD_LIBRARY_PATH=b, LLAMA_MOE_STATE=state)
srv = subprocess.Popen([b + "/llama-server", "-m", model, "-c", "4096", "-np", "1", "--port", port, "-lv", "3"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(900):
        time.sleep(2)
        if srv.poll() is not None: sys.exit("server died: " + open(log).read()[-300:])
        if "listening on" in open(log, errors="replace").read(): break
    done, i, t0 = 0, 0, time.time()
    while done < target:
        body = {"messages": [{"role": "user", "content": rp.CHAT_PROMPTS[i % len(rp.CHAT_PROMPTS)]}], "max_tokens": 400, "temperature": 0.7, "seed": i, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(r, timeout=1800))["timings"]
        done += int(t["predicted_n"]); i += 1
        txt = open(log, errors="replace").read(); h = re.findall(r"hit ([\d.]+)% this request", txt)
        print(f"request {i:3d}: {t['predicted_per_second']:5.1f} t/s, hit {h[-1] if h else '?'}%, {done} tokens, {time.time() - t0:.0f} s", flush=True)
finally:
    srv.send_signal(signal.SIGTERM)
    try: srv.wait(60)
    except Exception: srv.kill()
txt = open(log, errors="replace").read()
print("\nself-tune lines:")
for l in re.findall(r"[^\n]*moe-cache: (?:self-tune|auto)[^\n]*", txt): print(" ", l[:220])
print("\nsaved tuner state:", [l for l in open(state).read().splitlines() if "tuned" in l] if os.path.exists(state) else "none")
