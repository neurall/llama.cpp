#!/usr/bin/env python3
"""Reproduce one row of the README top table: `repro.py ID` prints the exact command (add `--run` to execute it), `repro.py --md` writes REPRO.md.
Every row is: model file, README cell (run.py test alias), arguments (always only -np 1 and, where it is named, -c), the fork build and the stock build.
Build names are directories under the builds directory of run.py (PERF_BUILDS); fork builds are the dev builds of the commit given in the README, stock builds
come from `run.py build --stock --ref <commit>` (llama.cpp upstream at that build number). The load mode of our builds is --load-mode pin (run.py adds it).
"""
import sys, os, subprocess, shlex
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "run.py")
# id: (model, test, extra args, fork build, stock build, note)
ROWS = {
 "glm-12k":     ("/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf", "12kin_1kout_code_review", "-c 16384", "plan16", "stock-b11381-836d57176", ""),
 "glm-edit":    ("/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf", "1kin_1kout_edit_src", "", "plan100", "stock-b11452-4fbc76dec", ""),
 "glm-game":    ("/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf", "1kout_make_game", "", "release-b12209", "stock-b11381-836d57176", "completion endpoint, 600 tokens"),
 "glm35-edit":  ("/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf", "1kin_1kout_edit_src", "", "release-b12250", "stock-b11381-836d57176", ""),
 "glm35-game":  ("/m/gl/3/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf", "1kout_make_game", "", "b11707-49fe4b756", "stock-b11381-836d57176", ""),
 "mimo-edit":   ("MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf (132 GB, no longer on disk)", "1kin_1kout_edit_src", "", "release-b12250", "stock-b11381-836d57176", "file removed; the 3-bit file replaces it"),
 "mimo-chat":   ("MiMo-V2.6-Flash-RL-IQ3_XXS-00001-of-00008.gguf (132 GB, no longer on disk)", "chat", "", "b11707-49fe4b756", "stock-b11381-836d57176", "chat endpoint, 1500 tokens"),
 "qni4-edit":   ("/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "1kin_1kout_edit_src", "", "release-b12364-6846d4803", "stock-b11452-4fbc76dec", ""),
 "qni4-12k":    ("/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "12kin_1kout_code_review", "", "b11707-49fe4b756", "stock-b11381-836d57176", ""),
 "qni4-chat":   ("/m/q/4/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "chat", "", "b11365-49db9ee", "stock-b11381-836d57176", "chat endpoint, 1500 tokens"),
 "qni3-edit":   ("/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "1kin_1kout_edit_src", "", "release-b12128", "stock-b11381-836d57176", "ppl: run.py bench -t ppl on the same builds (stock b11574, fork b12844)"),
 "qni3-game":   ("/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "1kout_make_game", "", "b11707-49fe4b756", "stock-b11381-836d57176", ""),
 "qn1-edit":    ("/m/q/1/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", "1kin_1kout_edit_src", "", "release-b12128", "stock-b11381-836d57176", ""),
 "q27mtp-edit": ("/m/q/5/Qwen3.8-27B-MTP-Q5_K_M.gguf", "1kin_1kout_edit_src", "--spec-type draft-mtp", "release-b12193", "stock-b11381-836d57176", "both sides with MTP"),
 "q27mtp-game": ("/m/q/5/Qwen3.8-27B-MTP-Q5_K_M.gguf", "1kout_make_game", "--spec-type draft-mtp", "release-b12193", "stock-b11381-836d57176", "both sides with MTP"),
 "q27nl-edit":  ("/m/q/4/Qwen3.8-27B-IQ4_NL.gguf", "1kin_1kout_edit_src", "", "b11707-49fe4b756", "stock-b11381-836d57176", ""),
 "q36-game":    ("/m/q/36/Qwen3.6-35B-A3B-GSQ-hybrid.gguf", "1kout_make_game", "", "plan99", "stock-b11381-836d57176", ""),
 "olmoe-game":  ("/m/o/8/4/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", "1kout_make_game", "", "plan99", "stock-b11574-8bb3cf0d2", ""),
}
# the 8 GB laptop rows are llama-cli on Windows (N:\\smallcard8 = b12843, N:\\stock-de7fa0a3c = b11514), prompts p10.txt / p12k.txt / p40k.txt / p125k.txt are the first N tokens of fork/bench/src_128k.cpp
LAPTOP = {"laptop-17": ("p10.txt", 2048), "laptop-12k": ("p12k.txt", 14336), "laptop-40k": ("p40k.txt", 46000), "laptop-125k": ("p125k.txt", 131072)}
def command(i):
    if i in LAPTOP:
        f, c = LAPTOP[i]
        return [f'{b}\\llama-cli.exe -m N:\\m\\Qwen3.6-35B-A3B-GSQ-hybrid.gguf -st --temp 0 -lv 3 -n 32 -c {c} -f N:\\{f}' for b in ("N:\\stock-de7fa0a3c", "N:\\smallcard8")]
    m, t, a, fk, st, note = ROWS[i]
    var = ("-np 1 " + a).strip()
    return [f'python3 fork/run.py bench --models {m} --builds {fk},{st} --variant "v={var}" --variants v -t {t} -n 3']
def md():
    out = ["# Reproducing the README rows", "", __doc__.strip().split("\n\n")[0], ""]
    for i in list(ROWS) + list(LAPTOP):
        out += [f'<a id="{i}"></a>', f"## {i}", ""]
        if i in ROWS and ROWS[i][5]: out += [ROWS[i][5], ""]
        out += ["```"] + command(i) + ["```", ""]
    return "\n".join(out)
if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    if sys.argv[1] == "--md":
        open(os.path.join(HERE, "REPRO.md"), "w").write(md()); print("REPRO.md written"); sys.exit(0)
    c = command(sys.argv[1])
    print("\n".join(c))
    if "--run" in sys.argv and sys.argv[1] in ROWS:
        subprocess.run(shlex.split(c[0]))
