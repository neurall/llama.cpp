#!/usr/bin/env python3
"""The overnight run: the best gainers again, the candidate build against the last release (no stock), one task per cell, fastest model first, each cell short.
usage: night_plan.py CANDIDATE_BUILD [RELEASE_BUILD] | python3 fork/run.py queue add-json
Cells per model: 1k edit (prompt + decode), short chat (decode), 12k prompt, and the KV-tail checks on MiMo (static context vs tail, 12k request then a short one).
Nothing here is queued by hand: run.py queue add-json takes the queue lock and keeps the backups."""
import json, sys
cand = sys.argv[1]; rel = sys.argv[2] if len(sys.argv) > 2 else "rel134-el8"
models = [("qni3", "/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"), ("qni4", "/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"),
          ("glm", "/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf"), ("mimo", "/m/m/3/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf")]
R = "python3 /p/bw/wt-main/fork/run.py cell"
t, k = [], 0
def add(id_, cli, note):
    global k
    t.append(dict(id=id_, order=-5000 + k, cli=cli, note=note)); k += 1
for n, m in models:
    for test, tok in (("edit4c", 300), ("tetrisc", 300), ("pf12k", 64)):
        add(f"night-{n}-{test}", f"{R} night-{n}-{test} {m} {cand},{rel} 2 {test} '' --tokens {tok}", f"{n} {test}: candidate vs last release")
M = dict(models)["mimo"]
for card, hold in (("24", ""), ("16", "0:8 1:8"), ("8", "0:16 1:16")):   # 4 GB cards cannot hold MiMo at all (7.4 GiB of non-expert weights): the 4 GB rows use Qwen3.6-35B below
    for arm, args, ctx in (("tail", "--moe kv-token-major=2,kv-tail=1,log=wk", -1), ("c32k", "", 32768)):
        h = f" --hold '{hold}'" if hold else ""
        add(f"night-kv-mimo-{card}g-{arm}", f"{R} night-kv-mimo-{card}g-{arm} {M} {cand} 2 edit4c '-lv 3 {args}' --tokens 300 --ctx {ctx}{h}", f"MiMo on {card} GB cards: kv-tail vs a static 32k context")
for arm, args, ctx in (("tail", "--moe kv-token-major=2,kv-tail=1,log=wk", -1), ("static", "", 32768)):
    add(f"night-kv-mimo-lead-{arm}", f"{R} night-kv-mimo-lead-{arm} {M} {cand} 1 edit4c '-lv 3 {args}' --tokens 300 --ctx {ctx} --lead 12000", "a 12k request, then the measured 1k one on the same server")
Q36 = "/m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf"
for card, hold in (("8", "0:16 1:16"), ("4", "0:20 1:20")):
    for arm, args, ctx in (("tail", "--moe kv-token-major=2,kv-tail=1,log=wk", -1), ("c32k", "", 32768)):
        add(f"night-kv-q36-{card}g-{arm}", f"{R} night-kv-q36-{card}g-{arm} {Q36} {cand} 2 edit4c '-lv 3 --load-mode mmap {args}' --tokens 300 --ctx {ctx} --hold '{hold}'", f"Qwen3.6-35B on {card} GB cards: kv-tail vs a static 32k context")
print(json.dumps(t))
