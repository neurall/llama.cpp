#!/usr/bin/env python3
"""release_table.py OURS STOCK | python3 fork/run.py queue add-json
The README table: every model, three tests (i = long prompt, io = 1k in / 1k out, o = short prompt, long answer), our build and stock side by side, each on its own defaults
(-np 1; stock gets -c 16384 so its 4096 default does not reject the 12k prompt), warm runs, 3 runs per cell (2 for the two biggest models). One cell per build so each starts
from its own defaults. Then the 35B context table (stock with a static -c against ours with and without) and the dense split on/off cells."""
import json, sys
ours, stock = sys.argv[1], sys.argv[2]
M = [("olmoe", "/m/o/4/OLMoE-1B-7B-0924-Q4_K_M.gguf", 3), ("q36", "/m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf", 3), ("qn1", "/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", 3),
     ("q27nl", "/m/q/4/Qwen3.8-27B-IQ4_NL.gguf", 3), ("q27mtp", "/m/q/5/Qwen3.8-27B-MTP-Q5_K_M.gguf", 3), ("qni3", "/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", 3),
     ("qni4", "/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", 3), ("glm", "/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf", 3),
     ("mimo", "/m/m/3/MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf", 2), ("glm35", "/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf", 2)]
T = [("i", "pf12k", 64), ("io", "edit4c", 1024), ("o4", "tetrisc", 4096)]   # o4 (user 2026-10-25): a 4k answer, 1k was too short for the decode side to show (the cache warms over the answer)
R = "python3 /p/bw/wt-main/fork/run.py cell"
out, k = [], 0
def add(id_, cli, note):
    global k
    out.append(dict(id=id_, order=-4000 + k, cli=cli, note=note)); k += 1
for n, path, runs in M:
    for tag, test, tok in T:
        add(f"rt-{n}-{tag}-ours", f"{R} rt-{n}-{tag}-ours {path} {ours} {runs} {test} '' --tokens {tok} --warm", f"{n} {tag}: ours")
        add(f"rt-{n}-{tag}-stock", f"{R} rt-{n}-{tag}-stock {path} {stock} {runs} {test} '-c 16384' --tokens {tok} --warm", f"{n} {tag}: stock")
Q36 = dict(M[1][:2] and [(a, b) for a, b, _ in M])["q36"]
# the 35B context table is the small-VRAM regime: one card with 8 GiB free (the model does not fit; with both 24 GB cards free it fits and stock needs no help)
SMALL = "--hog-gpu 8 --env CUDA_VISIBLE_DEVICES=1"
for c in (32768, 131072, 262144):   # static -c on both, then ours with no -c
    add(f"rt-ctx-{c}-stock", f"{R} rt-ctx-{c}-stock {Q36} {stock} 3 pf12k '' --tokens 64 --ctx {c} --warm {SMALL}", f"35B -c {c}: stock")
    add(f"rt-ctx-{c}-ours", f"{R} rt-ctx-{c}-ours {Q36} {ours} 3 pf12k '' --tokens 64 --ctx {c} --warm {SMALL}", f"35B -c {c}: ours")
add("rt-ctx-auto-ours", f"{R} rt-ctx-auto-ours {Q36} {ours} 3 pf12k '' --tokens 64 --warm {SMALL}", "35B no -c: ours")
for n in ("glm", "mimo", "qni4"):   # dense parallel split on and off (one card): the prompt
    path = dict((a, b) for a, b, _ in M)[n]
    add(f"rt-ds-{n}-on", f"{R} rt-ds-{n}-on {path} {ours} 3 pf12k '' --tokens 32 --warm", f"{n} dense split auto")
    add(f"rt-ds-{n}-off", f"{R} rt-ds-{n}-off {path} {ours} 3 pf12k '--moe dense-split=one' --tokens 32 --warm", f"{n} dense split off")
print(json.dumps(out))
