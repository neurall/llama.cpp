#!/usr/bin/env python3
"""client.py URL KIND N: one request to a llama-server, one JSON line {pp, tg, pn, tn[, draft]} or {"err": ...}.

KIND: short (README 'short' test: tetris chat, N tokens), chat (varied chat prompt chosen by $PIDX), code (first 9000
chars of src_12k.cpp as a completion), 12k (all of src_12k.cpp). Greedy. pp = prompt tokens/s, tg = decode tokens/s.
"""
import json, os, sys, urllib.request

here = os.path.dirname(os.path.abspath(__file__))
bench = os.path.dirname(here)
url, kind, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
PROMPTS = [
    "Write a Python function that parses a CSV file and returns the average of each column.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a CPU cache hierarchy works and why cache misses are expensive.",
    "Write a SQL query that finds the top 5 customers by total order value, with the schema.",
    "Translate into German and French: The meeting is moved to Thursday because the room is booked.",
    "Solve step by step: a train leaves at 9:40 at 84 km/h, a second at 10:10 at 105 km/h. When does it catch up?",
    "Write a bash script that backs up a directory to a dated tar.gz and keeps only the last 7 backups.",
    "Summarize the causes and consequences of the French Revolution.",
]
try:
    if kind == "chat":
        i = int(os.environ.get("PIDX", "0")) % len(PROMPTS)
        body = {"messages": [{"role": "user", "content": PROMPTS[i]}], "max_tokens": n, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        path = "/v1/chat/completions"
    elif kind == "short":
        body = {"messages": [{"role": "user", "content": "write smallest html tetris game"}], "max_tokens": n,
                "temperature": 0, "top_k": 1, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}}
        path = "/v1/chat/completions"
    elif kind in ("code", "12k"):
        text = open(os.path.join(bench, "src_12k.cpp")).read()
        body = {"prompt": text if kind == "12k" else text[:9000], "n_predict": n, "cache_prompt": False,
                "temperature": 0, "top_k": 1}
        path = "/completion"
    else:
        raise ValueError("unknown kind " + kind)
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=float(os.environ.get("REQ_TIMEOUT", "3600"))))["timings"]
    out = {"pp": round(t.get("prompt_per_second", 0), 2), "tg": round(t.get("predicted_per_second", 0), 2),
           "pn": t.get("prompt_n"), "tn": t.get("predicted_n")}
    if t.get("draft_n"):
        out["draft"] = "%s/%s" % (t.get("draft_n_accepted"), t["draft_n"])
    print(json.dumps(out))
except Exception as e:  # never raise: the runner logs the error and moves on
    print(json.dumps({"err": str(e)[:200]}))
