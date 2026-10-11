#!/usr/bin/env python3
"""check_links.py: every relative markdown link in README.md, fork/README.md, fork/PATCHES.md and fork/docs/*.md points to a file or directory that exists
(anchors and http(s) links are not checked)."""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
files = [os.path.join(ROOT, f) for f in ("README.md", "fork/README.md", "fork/PATCHES.md")] + sorted(glob.glob(os.path.join(ROOT, "fork/docs/*.md")))
bad = []
for f in files:
    if not os.path.exists(f):
        continue
    text = re.sub(r"```.*?```", "", open(f, encoding="utf-8").read(), flags=re.S)   # links inside code blocks are examples
    for m in re.finditer(r"\]\(([^)\s]+)\)", text):
        target = m.group(1).split("#")[0]
        if not target or re.match(r"[a-z]+:", target):
            continue
        if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(f), target))):
            bad.append(f"{os.path.relpath(f, ROOT)}: {m.group(1)}")
print("\n".join(bad) if bad else "all relative links resolve")
sys.exit(1 if bad else 0)
