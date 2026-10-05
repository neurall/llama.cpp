"""runnorm.py: the normalisation shared by runlog_all.py (full rebuild of all_runs) and runstats.py (incremental refresh): model family, build number, test group."""
import os, re

FAMS = [("GLM-5.3-Flash-GSQ-RCO-3.5bit", "GLM 3.5-bit"), ("GLM-5.3-Flash-GSQ-RCO-3.0bit", "GLM 3.0-bit"), ("MiMo", "MiMo IQ3_XXS"), ("Qwen3.8-Flash-Next-UD-IQ4_XS", "Next IQ4_XS"),
        ("Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", "Next IQ3_S"), ("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", "Next IQ1_M"), ("Qwen3.6-35B-A3B-GSQ", "Qwen3.6 Q2_0"), ("qwen36", "Qwen3.6 Q2_0"),
        ("Qwen3.8-27B-IQ4_NL", "27B IQ4_NL"), ("Qwen3.8-27B-GSQ-RCO-IQ3_S", "27B IQ3_S"), ("Qwen3.8-27B-MTP", "27B Q5_K_M MTP"), ("Qwen3.8-27B", "27B"), ("OLMoE", "OLMoE")]


def fam(model, args=""):
    m = (model or "").strip()
    if not m:
        m = (re.search(r"(?:-m|--model)\s+(\S+)", args or "") or [None, ""])[1]
    m = os.path.basename(m.split(" ")[0])
    for k, v in FAMS:
        if k.lower() in m.lower():
            return v
    return re.sub(r"-0000\d-of-0000\d|\.gguf", "", m)[:28] or "?"


def bno_of(build, bno=""):
    m = re.search(r"b(\d{5})", build or "") or re.search(r"(\d{5})", str(bno or ""))
    return int(m.group(1)) if m else None


def tgroup(t):
    return re.sub(r"-lv3$", "", re.sub(r"^cli-", "", t or ""))


GAME_WORDS = ["tetris", "racing", "shooter", "snake"]


def prompt_of(test, rep="", tag=""):
    """the prompt of a run: the logged column, else derived from the test and the repetition (game4 and edit4 rotate through four prompts by repetition)"""
    if tag:
        return tag
    t = tgroup(test)
    try:
        r = int(float(rep))
    except (TypeError, ValueError):
        return t
    if t == "game4":
        return "game4-" + GAME_WORDS[r % 4]
    if t == "edit4":
        return f"edit4-{r % 4}"
    return t
