#!/usr/bin/env python3
"""Render the README "Best gainers" table as a picture (default fork/docs/img/gainers.png): python3 fork/gainers_png.py [OUT.png]
Reads the table from README.md, writes an HTML page and takes a screenshot with headless Firefox (needs `firefox` on the PATH)."""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "docs", "img", "gainers.png"))
lines = open(os.path.join(HERE, "..", "README.md")).read().split("\n")
a = next(k for k, l in enumerate(lines) if l.startswith("| model | test | t/s"))
rows = []
for l in lines[a + 2:]:
    if not l.startswith("|"): break
    rows.append([re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", c.strip()) for c in l.strip().strip("|").split("|")])
hdr = ["model", "test", "decode t/s<br>stock / ours", "prompt pp t/s<br>stock / ours", "gain decode", "gain prompt", "ppl<br>stock / ours", "build"]
h = "<html><body style='margin:0;background:#fff;font-family:Segoe UI,Helvetica,Arial,sans-serif'><div style='padding:18px 22px;width:1180px'>"
h += "<div style='font-size:22px;font-weight:600;margin-bottom:4px'>llama.cpp fork vs official stock: best gainers</div><div style='font-size:13px;color:#57606a;margin-bottom:12px'>Upper number stock, lower ours, same power limits. * larger than RAM, partly from SSD.</div>"
h += "<table style='border-collapse:collapse;font-size:14px;width:100%'><tr>" + "".join(f"<th style='text-align:left;border-bottom:2px solid #d0d7de;padding:6px 8px;font-size:12px;color:#57606a'>{x}</th>" for x in hdr) + "</tr>"
for i, row in enumerate(rows):
    h += f"<tr style='background:{'#f6f8fa' if i % 2 else '#fff'}'>"
    for j, c in enumerate(row[:8]):
        h += f"<td style='padding:5px 8px;border-bottom:1px solid #eaeef2;{'font-weight:700;color:#1a7f37;' if j in (4, 5) and c != '-' else ''}'>{c}</td>"
    h += "</tr>"
h += "</table></div></body></html>"
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "g.html"); open(p, "w").write(h)
    subprocess.run(["firefox", "--headless", "--screenshot", out, f"--window-size=1240,{150 + 53 * len(rows)}", "file://" + p], check=True, capture_output=True)
print(out)
