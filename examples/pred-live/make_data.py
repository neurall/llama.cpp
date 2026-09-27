#!/usr/bin/env python3
"""Rebuild the pretraining / test texts used with llama-pred-live (documents separated by '<|endoftext|>').

  make_data.py <out_dir> [tinystories] [cosmopedia] [tiny-codes] [starcoder]

Needs the `hf` CLI logged in (run as `env -u HF_TOKEN python3 make_data.py ...`); tiny-codes and
starcoderdata are gated: accept their terms on huggingface.co first.

  tinystories : roneneldan/TinyStories, TinyStoriesV2-GPT4-valid.txt as is (27,630 stories)
  cosmopedia  : HuggingFaceTB/cosmopedia-20k, all 20,000 'text' docs, shuffled (seed 0)
  tiny-codes  : nampdn-ai/tiny-codes, all 9 parquet files (1,632,309 rows), shuffled (seed 0),
                first 20,000 'prompt' + blank line + 'response'
  starcoder   : bigcode/starcoderdata, shard train-00000 of html, css, python, javascript,
                c, cpp, shell; per language shuffled (seed 0), first 3,000 files with > 200 chars,
                each cut to 6,000 chars; all 21,000 shuffled together (seed 1)
  mix         : starcoder-mix + cosmopedia-20k interleaved so both have used the same number of characters at
                every point (the first 50k tokens are ~25k code + ~25k cosmopedia); needs both built first
"""
import glob, os, random, subprocess, sys

STAR_SHARDS = {"html": 29, "css": 12, "python": 59, "javascript": 65, "c": 53, "cpp": 48, "shell": 4}


def hf(repo, files, local):
    subprocess.run(["hf", "download", repo, *files, "--repo-type", "dataset", "--local-dir", local], check=True)


def write(out, name, docs):
    txt = "\n<|endoftext|>\n".join(docs)
    with open(os.path.join(out, name), "w") as f:
        f.write(txt)
    print(f"{name}: {len(docs)} docs, {len(txt) / 1e6:.1f} M chars")


def tinystories(out):
    hf("roneneldan/TinyStories", ["TinyStoriesV2-GPT4-valid.txt"], out)


def cosmopedia(out):
    import pyarrow.parquet as pq
    d = os.path.join(out, "raw-cosmopedia")
    hf("HuggingFaceTB/cosmopedia-20k", [], d)
    docs = []
    for f in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True)):
        docs += pq.read_table(f, columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(docs)
    write(out, "cosmopedia-20k.txt", [x.strip() for x in docs])


def tiny_codes(out):
    import pyarrow.parquet as pq
    d = os.path.join(out, "raw-tiny-codes")
    hf("nampdn-ai/tiny-codes", [], d)
    rows = []
    for f in sorted(glob.glob(f"{d}/*.parquet")):
        rows += pq.read_table(f, columns=["prompt", "response"]).to_pylist()
    random.Random(0).shuffle(rows)
    write(out, "tiny-codes-20k.txt", [(r["prompt"] or "").strip() + "\n\n" + (r["response"] or "").strip() for r in rows[:20000]])


def starcoder(out):
    import pyarrow.parquet as pq
    d = os.path.join(out, "raw-starcoder")
    hf("bigcode/starcoderdata", [f"{l}/train-00000-of-{n:05d}.parquet" for l, n in STAR_SHARDS.items()], d)
    docs = []
    for lang in sorted(STAR_SHARDS):
        rows = pq.read_table(f"{d}/{lang}/train-00000-of-{STAR_SHARDS[lang]:05d}.parquet", columns=["content"]).column("content").to_pylist()
        random.Random(0).shuffle(rows)
        docs += [r[:6000] for r in rows if r and len(r) > 200][:3000]
    random.Random(1).shuffle(docs)
    write(out, "starcoder-mix.txt", docs)


def mix(out):
    def docs(name):
        return open(os.path.join(out, name)).read().split("\n<|endoftext|>\n")
    a, b = docs("starcoder-mix.txt"), docs("cosmopedia-20k.txt")
    res, ca, cb, ia, ib = [], 0, 0, 0, 0
    while ia < len(a) or ib < len(b):
        if ib >= len(b) or (ia < len(a) and ca <= cb):
            res.append(a[ia]); ca += len(a[ia]); ia += 1
        else:
            res.append(b[ib]); cb += len(b[ib]); ib += 1
    write(out, "code-cosmo-mix.txt", res)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    steps = {"tinystories": tinystories, "cosmopedia": cosmopedia, "tiny-codes": tiny_codes, "starcoder": starcoder, "mix": mix}
    for a in sys.argv[2:]:
        if a in steps:
            steps[a](out)
