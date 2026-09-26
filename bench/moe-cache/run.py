#!/usr/bin/env python3
"""Send one of the fork's benchmark prompts to a running llama-server and print speeds.

usage: run.py short|12k|128k [--url http://127.0.0.1:8080]
  short  1500-token chat reply to "write smallest html tetris game" (/v1/chat/completions)
  12k    src_12k.cpp as a raw completion prompt (12,302 tokens), 32 tokens generated
  128k   src_128k.cpp (125,257 tokens), 64 tokens generated; needs -c 131072 on the server
All greedy (temperature 0). Run each twice and keep the second: the first run after starting
the server also reads the model from disk.
"""
import json, os, sys, urllib.request

GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1}
here = os.path.dirname(os.path.abspath(__file__))
test = sys.argv[1] if len(sys.argv) > 1 else "short"
url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8080"

if test == "short":
    path, body = "/v1/chat/completions", {"messages": [{"role": "user", "content": "write smallest html tetris game"}],
                                          "max_tokens": 1500, **GREEDY}
else:
    f = {"12k": "src_12k.cpp", "128k": "src_128k.cpp"}[test]
    path, body = "/completion", {"prompt": open(os.path.join(here, f)).read(), "n_predict": 32 if test == "12k" else 64,
                                 "cache_prompt": False, **GREEDY}

req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=7200))["timings"]
print(f"{test}: prompt {t['prompt_n']} tokens, {t['prompt_per_second']:.1f} t/s ({t['prompt_ms'] / 1e3:.0f} s) | "
      f"decode {t['predicted_n']} tokens, {t['predicted_per_second']:.2f} t/s")
