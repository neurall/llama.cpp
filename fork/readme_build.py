#!/usr/bin/env python3
"""readme_build.py: README.md = fork/docs/readme-top.md (prose) with the tables of readme_table.py filled in + the rest of the current README from '## Run' on.
usage: readme_build.py [--write]   (without --write: prints the new top only)"""
import subprocess, sys, re, os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
out = subprocess.run([sys.executable, os.path.join(here, "readme_table.py")], capture_output=True, text=True).stdout
parts = dict(re.findall(r"<<(\w+)>>\n(.*?)(?=<<\w+>>\n|\Z)", out, re.S))
ppl = subprocess.run([sys.executable, os.path.join(here, "run.py"), "ppl-table", "--readme"], capture_output=True, text=True).stdout
tiers = """| tier | tokens the card holds |
|---|---|
| f16 (default) | 18,871 |
| q8_0 in the pool | 31,782 |
| q4_0 in the pool | 54,897 |
| host memory (up to the free RAM; 21,953 on this 32 GB laptop at the time) | RAM-bound |"""
top = open(os.path.join(here, "docs", "readme-top.md")).read()
for k, v in (("BEST", parts.get("BEST", "")), ("CTX", parts.get("CTX", "")), ("DS", parts.get("DS", "")), ("TIERS", tiers), ("PPL", ppl)):
    top = top.replace("{{" + k + "}}", v.strip())
if "--write" not in sys.argv:
    print(top); sys.exit(0)
cur = open(os.path.join(root, "README.md")).read()
i = cur.index("\n## Run\n")
open(os.path.join(root, "README.md"), "w").write(top.rstrip() + "\n" + cur[i:])
print("README.md written:", top.count("\n"), "lines of new top")
