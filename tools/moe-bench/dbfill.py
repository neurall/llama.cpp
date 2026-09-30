#!/usr/bin/env python3
"""Idempotent perf.db backfill: commit_sha / build_no from the recorded version (or the build directory name), hw from host."""
import sqlite3, re, subprocess, os
c = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf.db"))
def git(*a): return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), *a], capture_output=True, text=True)
for id_, build, ver in c.execute("select id,build,version from runs where commit_sha is null").fetchall():
    m = re.search(r"commit (\w+)", ver or "") or re.match(r"(?:b\d+-|bis-)([0-9a-f]{7,})$", build)
    if m: c.execute("update runs set commit_sha=? where id=?", (m.group(1), id_))
for id_, sha in c.execute("select id,commit_sha from runs where build_no is null and commit_sha is not null").fetchall():
    r = git("rev-list", "--count", sha)
    if r.returncode == 0 and r.stdout.strip(): c.execute("update runs set build_no=? where id=?", (int(r.stdout), id_))
c.execute("update runs set hw='pc1' where hw is null")
c.commit()
print(c.execute("select count(*), sum(commit_sha is null), sum(build_no is null), sum(hw is null) from runs").fetchone(), "rows, no commit, no build no, no hw")
