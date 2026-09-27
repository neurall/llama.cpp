#!/usr/bin/env python3
"""Generate a long single-topic session from a running llama-server, for pred_lab.py datasets.

Each part asks the prompt again with the tail of the previous part and continues it, filling the
context; looping text is cut (a window of lines with too few unique ones) before it is kept.

  gen_session.py OUT_DIR "prompt" [--url http://127.0.0.1:8080] [--tokens 20000] [--ctx 4096]
      [--temp 0.7] [--loop-window 30] [--loop-min-unique 0.4]

Writes OUT_DIR/S00.txt, S01.txt, ... (chat-templated prompt + answer), ready for
ROUTER_DUMP=... llama-eval-callback -f.
"""
import argparse, json, os, time, urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("prompt")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tokens", type=int, default=20000, help="stop after this many kept tokens")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--loop-window", type=int, default=30, help="lines per loop-detection window")
    ap.add_argument("--loop-min-unique", type=float, default=0.4, help="cut where a window has fewer unique lines")
    ap.add_argument("--continue-with", default="Continue the code exactly where this previous part ends, adding more features:")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def post(path, body):
        r = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=900))

    for _ in range(120):
        try:
            urllib.request.urlopen(a.url + "/health")
            break
        except Exception:
            time.sleep(1)

    def cut_loop(text):
        lines = text.split("\n")
        for i in range(a.loop_window, len(lines) + 1):
            w = [l.strip() for l in lines[i - a.loop_window:i] if l.strip()]
            if w and len(set(w)) / len(w) < a.loop_min_unique:
                return "\n".join(lines[:i - a.loop_window])
        return text

    prev, total, i, seed = "", 0, 0, 0
    while total < a.tokens and seed < 200:
        msg = a.prompt if not prev else a.prompt + "\n\n" + a.continue_with + "\n" + prev[-3000:]
        tpl = post("/apply-template", {"messages": [{"role": "user", "content": msg}]})["prompt"]
        n_prompt = len(post("/tokenize", {"content": tpl})["tokens"])
        out = post("/completion", {"prompt": tpl, "n_predict": a.ctx - 96 - n_prompt, "temperature": a.temp,
                                   "seed": seed, "cache_prompt": False})
        seed += 1
        text = cut_loop(out["content"])
        n = len(post("/tokenize", {"content": text})["tokens"])
        if n < 200:
            print(f"part too short after the loop cut ({n} tokens), next seed", flush=True)
            continue
        prev = text
        total += n
        with open(os.path.join(a.out, f"S{i:02d}.txt"), "w") as f:
            f.write(tpl + text)
        print(f"part {i}: {n_prompt} prompt, {out['tokens_predicted']} generated, {n} kept", flush=True)
        i += 1
    print(f"kept {total} tokens in {i} parts")


if __name__ == "__main__":
    main()
