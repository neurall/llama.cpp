#!/usr/bin/env python3
"""The judge: a healthy model reads the answers our runs stored, because a broken engine reports fine speeds and a high hit rate while it writes nonsense.
The gather mask of 2026-11-01 fooled a day of rows. A row the judge calls nonsense has no accepted prompt or decode speed.
Qwen3.6 reads everything: about 6 s an answer, Qwen3.8-27B takes 19 s on the same card. A row Qwen3.6 calls nonsense goes to Qwen3.8-27B on the CPU, and its word stands.

Runs on W10 next to the stock llama-server with Qwen3.6 (port 8190, scheduled task gensrv), on its own copy of runs.db, so PC1 neither waits nor coordinates:
    python judge_w10.py DIR      DIR holds runs.db (a copy), README.md, verdicts.jsonl (appended, one row per line; the last line of a row counts); see docs/w10-builds.md
`run.py judge --sync` copies the first two in and takes the third back. run.py imports this file too, to read a row on demand: same prompt, same scale."""
import hashlib, json, os, sqlite3, subprocess, sys, time, urllib.request, re

PARTS = (300, 1000)   # tokens of an answer's end, read from the end backward, each piece on its own. The next piece only if this one seems fine: the first piece called nonsense ends the reading
PARTS_DEEP = (1000, 4000, 8000)   # the second opinion reads further; the server's context is 16384
SECOND_PORT = 8191   # Qwen3.8-27B, started by the first flagged row. On the CPU: Qwen3.6 keeps the card
SECOND_CMD = [r"N:\build\stock-de7fa0a3c\llama-server.exe", "-m", r"N:\m\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", "--host", "0.0.0.0", "--port", str(SECOND_PORT), "-np", "1", "-c", "16384", "-ngl", "0", "--no-webui"]
BAD = 2   # quality at or below this is nonsense. Measured 2026-11-01: damaged texts, loops and `!!!!` 1-2, real answers 3-5. A separate true/false field came back false for every text, so the scale alone decides
MIN_CHARS = 16   # a shorter answer is too little to tell (prompt rows ask for one token)
ASK_CHARS = 3000   # of the request, shown with each piece: long prompts carry their task at the start or at the end, so both ends
SCHEMA = {"type": "object", "properties": {"why": {"type": "string", "maxLength": 80}, "quality": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}, "required": ["why", "quality"]}
TEXT = """You check the output of a language model for signs of a broken inference engine.

Broken engine output: random tokens or symbols, word salad, sentences that fall apart into nonsense, words from many languages mixed for no reason, one phrase repeating without end, code that dissolves into unrelated fragments.
Healthy output: coherent text or code, even if simple, wrong, unfinished, or not what was asked. Code and data often repeat similar blocks with small changes: that is healthy.

The output below is a slice cut from a longer output. Its first and last lines may be cut mid-word, and it may stop at a token limit: never count that against it. Do not judge facts or style, or whether code would compile.

Request the model was given (may be cut, may be missing):
<<<
{ask}
>>>

Slice of its output:
<<<
{part}
>>>

Reply as JSON. why: what you saw, at most 10 words. quality: 5 coherent, on the task and well made; 4 coherent and on the task; 3 coherent but weak or drifting off the task; 2 partly broken as described above; 1 broken as described above."""


def is_garbage(text):
    """'!!!!!!...': the logits are NaN or broken, argmax is token 0. Such a run reports fine speeds and a 99% hit rate (the same token uses the same experts): GLM kv-tail 33 t/s 2026-10-17; the gather mask
    2026-11-01 fooled a day of rows on GLM and Qwen Next. Never a valid row, --keep-loops or not; a repeat cannot help."""
    t = (text or "").strip()
    return len(t) >= 20 and len(set(t)) <= 3


def cut_ask(ask):
    """what is kept of a request: a prompt of 128k tokens would fill runs.db"""
    ask = ask or ""
    return ask if len(ask) <= ASK_CHARS else ask[:ASK_CHARS // 4] + "\n[...]\n" + ask[-ASK_CHARS * 3 // 4:]


def post(url, path, body):
    q = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(q, timeout=1800))


def judge_text(url, text, ask="", parts=PARTS):
    """the reading of one answer by one model: {verdict: good | nonsense | none, quality, why, parts, of, tokens, s}; none: it could not read it, down=True when the server is the reason"""
    t0, text = time.time(), (text or "").strip()
    if is_garbage(text):
        return {"verdict": "nonsense", "quality": 1, "why": "one repeated token, NaN logits", "parts": 0, "of": 0, "tokens": 0, "s": 0}
    try:
        ids = post(url, "/tokenize", {"content": text})["tokens"]
        sizes, left = [], len(ids)
        for n in parts:
            if left > 0: sizes.append(n); left -= n
        end, res = len(ids), {"verdict": "good", "quality": 6, "why": "", "parts": 0, "of": len(sizes), "tokens": 0}
        for n in sizes:
            part = post(url, "/detokenize", {"tokens": ids[max(0, end - n):end]})["content"]
            r = post(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": TEXT.format(ask=cut_ask(ask) or "(not available)", part=part)}], "max_tokens": 64, "temperature": 0, "seed": 1, "cache_prompt": False,
                                                   "chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
            d = json.loads(r["choices"][0]["message"]["content"])
            res["parts"] += 1; res["tokens"] += min(n, end); end -= n
            if d["quality"] < res["quality"]: res.update(quality=d["quality"], why=d["why"])   # the worst piece speaks for the answer
            if d["quality"] <= BAD:
                res["verdict"] = "nonsense"
                break
    except OSError as ex:
        return {"verdict": "none", "why": f"{type(ex).__name__}: {ex}"[:120], "down": True}
    except (ValueError, KeyError, IndexError, TypeError) as ex:
        return {"verdict": "none", "why": f"{type(ex).__name__}: {ex}"[:120]}
    res["s"] = round(time.time() - t0)
    return res


def judge_answer(url, text, ask="", start=False):
    """the verdict on an answer: Qwen3.6 reads it, what it calls nonsense goes to Qwen3.8-27B, whose word stands (first: what Qwen3.6 said). Second model down and start not set: Qwen3.6's word stands, second: none"""
    v = judge_text(url, text, ask)
    if v["verdict"] != "nonsense" or not v["parts"]:   # no parts: one repeated token, no opinion needed
        return v
    second = url.rsplit(":", 1)[0] + f":{SECOND_PORT}"
    if not server_up(second) and not (start and second_start(second)):
        v["second"] = "none"
        return v
    w = judge_text(second, text, ask, PARTS_DEEP)
    if w["verdict"] == "none":
        v["second"] = "none"
        return v
    w["first"] = f"{v['quality']}: {v['why']}"
    return w


def second_start(url):
    """Qwen3.8-27B for the flagged rows: on the first of them, it stays for the next. False when it does not come up in 3 minutes"""
    print("starting the second model", flush=True)
    subprocess.Popen(SECOND_CMD, stdin=subprocess.DEVNULL, stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "second.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(36):
        time.sleep(5)
        if server_up(url): return True
    return False


def verdict_str(v):
    if v["verdict"] == "none": return f"none: {v['why']}"
    return f"{v['verdict']} {v['quality']}: {v['why']} [{v['parts']} of {v['of']} parts, {v['tokens']} tokens, {v['s']} s]" + (f" (Qwen3.6 said {v['first']})" if v.get("first") else "") + (" (no second opinion)" if v.get("second") else "")


def stored_answer(con, build, test, model, ts):
    """(text, request) a run stored in table outputs: same build, test and model, the nearest time within 2 minutes; (None, "") when there is none. Older rows have no request."""
    has_ask = "ask" in [r[1] for r in con.execute("PRAGMA table_info(outputs)")]
    import zlib
    o = con.execute(f"select text, {'ask' if has_ask else 'NULL'} from outputs where build=? and test=? and model=? and abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) <= 120 order by abs(strftime('%s', ts) - strftime('%s', substr(?, 1, 19))) limit 1",
                    (build, test, model, ts, ts)).fetchone()
    return (zlib.decompress(o[0]).decode("utf8", "replace"), o[1] or "") if o else (None, "")


def key_rows(con, readme):
    """what our reports stand on: the rows the README cites, and per model and test the two best by decode, prompt and both gains"""
    ids = {int(x) for x in re.findall(r"outputs/(\d+)s?\.txt", open(readme, errors="replace").read())} if os.path.exists(readme) else set()
    have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for m in ("tps", "pp", "gain_t", "gain_pp"):
        if m in have:
            ids |= {r[0] for r in con.execute(f"select id from (select rowid id, row_number() over (partition by stem, test order by cast({m} as real) desc) n from runs where coalesce(ok, '1') != '0' and cast({m} as real) > 0 and cast(out_chars as int) >= ?) where n <= 2", (MIN_CHARS,))}
    return ids


def queue(con, done, readme):
    """rows to read in order: those of the last 3 days newest first, then the key rows, then the rest newest first"""
    rows = [r for r in con.execute("select rowid, ts, build, test, model from runs where coalesce(ok, '1') != '0' and cast(out_chars as int) >= ? order by rowid desc", (MIN_CHARS,)) if r[0] not in done]
    recent = time.strftime("%F", time.localtime(time.time() - 3 * 86400))
    key = key_rows(con, readme)
    return sorted(rows, key=lambda r: 0 if r[1] >= recent else 1 if r[0] in key else 2)   # stable: newest first inside each group


def server_up(url):
    try:
        return json.load(urllib.request.urlopen(url + "/health", timeout=5)).get("status") == "ok"
    except (OSError, ValueError):
        return False


def main(d):
    """reads forever: a new runs.db.new from PC1 takes over between batches, so new rows go first; with nothing to read it sleeps"""
    url = "http://127.0.0.1:8190"
    db, new, vf, readme = (os.path.join(d, x) for x in ("runs.db", "runs.db.new", "verdicts.jsonl", "README.md"))
    done, by_text = set(), {}
    if os.path.exists(vf):
        for line in open(vf, encoding="utf8"):
            try: v = json.loads(line)
            except ValueError: continue   # a line cut by a crash: that row is read again
            done.add(v["id"])
            if v.get("sha"): by_text[v["sha"]] = v
    out = open(vf, "a", encoding="utf8")
    while True:
        if not server_up(url):
            print("server down, starting task gensrv", flush=True)
            subprocess.run(["schtasks", "/run", "/tn", "gensrv"], capture_output=True)
            time.sleep(60)
            continue
        if os.path.exists(new): os.replace(new, db)
        if not os.path.exists(db):
            time.sleep(60)
            continue
        con = sqlite3.connect(db)
        con.execute("CREATE INDEX IF NOT EXISTS i_outputs_judge ON outputs (build, test, model)")
        todo = queue(con, done, readme)
        print(f"{time.strftime('%F %T')} {len(todo)} rows to read, {len(done)} read", flush=True)
        for r in todo[:20]:   # a small batch: a new copy from PC1 is seen soon, and nothing holds the file while it comes
            text, ask = stored_answer(con, r[2], r[3], r[4], r[1])
            sha = hashlib.sha1((text or "").encode("utf8", "replace")).hexdigest()[:16]
            if len((text or "").strip()) < MIN_CHARS:
                v = {"verdict": "none", "why": "no stored answer"}
            elif sha in by_text:   # the same text again (a greedy test repeats it): the same verdict, no request
                v = {k: by_text[sha].get(k) for k in ("verdict", "quality", "why", "parts", "of", "tokens", "first", "second")}; v["s"] = 0
            else:
                v = judge_answer(url, text, ask, start=True)
            if v.get("down"):
                print("server stopped answering:", v["why"], flush=True)
                break
            v.update(id=r[0], ts=r[1], build=r[2], test=r[3], model=r[4], sha=sha, read=time.strftime("%F %T"))
            out.write(json.dumps(v, ensure_ascii=False) + "\n"); out.flush()
            done.add(r[0])
            if v["verdict"] == "good" or v["verdict"] == "nonsense": by_text.setdefault(sha, v)
            print(r[0], r[2][:26], r[3][:10], verdict_str(v) if v["verdict"] != "none" else "none: " + v["why"], flush=True)
        con.close()
        if not todo: time.sleep(60)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
