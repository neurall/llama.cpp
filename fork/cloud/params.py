#!/usr/bin/env python3
"""params.py LOG GPUS TEST TAG BUILD RESULTS_DIR [TS]: append one row to RESULTS_DIR/server-params.csv for a finished run: date and time, what the server chose
(placement, load mode, threads, batch, ubatch, context, cache slots and device memory = the cache fill, tuned knobs, link and CPU bandwidth it measured),
and what it got (decode and prompt t/s, hit rate, churn). Decode speed and hit rate come from the run's row in run-history.csv, the rest from the server log."""
import csv, os, re, sys, time
log, gpus, test, tag, build, res = sys.argv[1:7]
txt = open(log, errors="replace").read() if os.path.exists(log) else ""
row = {}
hist = os.path.join(res, "run-history.csv")
if os.path.exists(hist):
    rows = [r for r in csv.DictReader(open(hist)) if r.get("hw") == f"pod-{gpus}gpu" and r.get("test") == test and (r.get("note") or "").split(" | ")[0] == tag]
    if rows:
        row = rows[-1]
g = lambda pat, i=1, flags=0: (lambda m: m.group(i) if m else "")(re.search(pat, txt, flags))
ts = row.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(log))) if row.get("ts") or os.path.exists(log) else time.strftime("%Y-%m-%d %H:%M:%S")
d = {
    "date": ts.split(" ")[0], "time": ts.split(" ")[1], "gpus": gpus, "test": test, "tag": tag, "build": build,
    "decode_tps": row.get("tps", ""), "prompt_tps": row.get("pp", ""), "hit_pct": row.get("hit", ""),
    "placement": g(r"MoE placement: ([^(\n]*)").strip()[:70],
    "model_gib": g(r"MoE model \(([\d.]+) GiB\)"), "free_vram_gib": g(r"free VRAM \(([\d.]+) GiB"),
    "load": "pinned" if "weights pinned" in txt else "mmap",
    "ctx": g(r"MoE model .*?ctx (\d+)", 1, re.S), "threads_auto": g(r"MoE model .*?threads (\d+)", 1, re.S),
    "threads_decode": g(r"threads = (\d+) decode"), "threads_batch": g(r"decode / (\d+) batch"), "batch": g(r"decode / \d+ batch, batch (\d+)"), "ubatch": g(r"batch \d+, ubatch (\d+)"),
    "cache_slots_min": g(r"slots/layer (\d+)\.\.\d+, \d+ inserts"), "cache_slots_max": g(r"slots/layer \d+\.\.(\d+), \d+ inserts"), "cache_layers": g(r"cache enabled: (\d+) layers"),
    "cache_mib": g(r"([\d.]+) MiB device memory"),
    "hit_request": g(r"hit ([\d.]+)% this request"), "hit_overall": g(r"this request / ([\d.]+)% overall"),
    "tuned": g(r"margin (-?\d+) gate (\d+) wait (\d+) big (\d+) predict (\d+)", 0).replace("margin ", "m").replace(" gate ", " g").replace(" wait ", " w").replace(" big ", " b").replace(" predict ", " p"),
    "prefill_links_gbs": g(r"prefill links \(GB/s\): ([^>\n]*?) ->").strip(), "probe": g(r"probe: ([^\n]{0,160})").strip(),
    "churn": g(r"moe churn = ([^\n]*?) \(this request\)"),
    "selftune": "; ".join(re.findall(r"self-tune: ([^\n]{0,80})", txt)[-2:]),
    "state": (row.get("env") or "")[:60],
}
out = os.path.join(res, "server-params.csv")
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(d))
    if new:
        w.writeheader()
    w.writerow(d)
