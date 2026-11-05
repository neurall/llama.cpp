#!/usr/bin/env python3
"""No commit of this fork names Claude, Anthropic or a co-author unless the owner orders it (owner 2026-11-05: "not supposed to be there in any capacity without my explicit
approval or order", "i have no coauthors and if i had, the decision to add them lies solely on me"). A harness reminder asks for such trailers; the owner's rule wins.
The check is case-insensitive and looks for the words, not for one trailer's spelling, because they vary.

  check_attribution.py --install              copy this file into the repo's hooks as commit-msg and pre-push (all worktrees share them)
  check_attribution.py --message FILE         the commit-msg hook: a message with such a line is refused
  check_attribution.py --push TREE BRANCH     what `run.py publish` pushes: the owner's commits of BRANCH the fork's remote does not have yet
  as pre-push hook (stdin: local ref, local sha, remote ref, remote sha)

Only the owner's own commits count (author or committer e-mail = git config user.email): commits that came in with an upstream merge belong to other people and stay as they are.
ATTRIBUTION_OK_BY_OWNER=1 lets one command through; only the owner sets it.
"""
import os, re, subprocess, sys

BAD = re.compile(r"claude|anthropic|co-?author", re.I)
ZERO = "0" * 40
FORK_REMOTE = "neurall"


def git(*a, cwd=None):
    return subprocess.run(("git",) + a, capture_output=True, text=True, cwd=cwd).stdout


def bad_lines(msg):
    return [l.strip() for l in msg.splitlines() if BAD.search(l)]


def owner_commits_bad(revs, cwd=None):
    """[(short sha, line)] of the owner's commits among `git rev-list revs...` whose message has such a line"""
    me = git("config", "user.email", cwd=cwd).strip().lower()
    out = []
    for h in git("rev-list", *revs, cwd=cwd).split():
        ae, ce, msg = git("show", "-s", "--format=%ae%x00%ce%x00%B", h, cwd=cwd).split("\0", 2)
        if me and me not in (ae.strip().lower(), ce.strip().lower()):
            continue
        out += [(h[:10], l) for l in bad_lines(msg)]
    return out


def revs_not_on(remote_sha, tip, cwd=None):
    """the commits of tip a remote ref (at remote_sha) does not have; a sha this clone does not know: everything no remote ref has"""
    if remote_sha and remote_sha != ZERO and subprocess.run(["git", "cat-file", "-e", remote_sha + "^{commit}"], cwd=cwd, capture_output=True).returncode == 0:
        return [tip, "--not", remote_sha]
    return [tip, "--not", "--remotes"]


def refuse(found, what):
    sys.stderr.write(f"\nREFUSED: {what} names Claude, Anthropic or a co-author. That is not allowed without the owner's explicit order.\n")
    for h, l in found[:20]:
        sys.stderr.write(f"  {h}  {l}\n")
    sys.stderr.write("Take the line out of the message (git commit --amend / git rebase), or the owner sets ATTRIBUTION_OK_BY_OWNER=1 for this one command.\n\n")
    sys.exit(1)


def main():
    me = os.path.basename(sys.argv[0])
    if os.environ.get("ATTRIBUTION_OK_BY_OWNER") == "1":
        sys.exit(0)
    a = sys.argv[1:]
    if a[:1] == ["--install"]:
        hooks = os.path.join(os.path.abspath(git("rev-parse", "--git-common-dir").strip()), "hooks")
        src = open(os.path.abspath(__file__)).read()
        for name in ("commit-msg", "pre-push"):
            p = os.path.join(hooks, name)
            with open(p + ".tmp", "w") as f:
                f.write(src)
            os.chmod(p + ".tmp", 0o755)
            os.replace(p + ".tmp", p)
            print("installed", p)
        return
    if me == "commit-msg" or a[:1] == ["--message"]:
        msg = open(a[-1]).read()
        # lines git adds itself (# comments) are not part of the message
        found = [("message", l) for l in bad_lines("\n".join(x for x in msg.splitlines() if not x.startswith("#")))]
        if found:
            refuse(found, "this commit message")
        return
    if a[:1] == ["--push"]:
        tree, br = a[1], a[2]
        line = git("ls-remote", FORK_REMOTE, f"refs/heads/{br}", cwd=tree).split()
        found = owner_commits_bad(revs_not_on(line[0] if line else ZERO, br, cwd=tree), cwd=tree)
        if found:
            refuse(found, f"the commits of {br} that the fork does not have yet")
        print(f"attribution check: the new commits of {br} name no Claude, Anthropic or co-author", flush=True)
        return
    if me == "pre-push":
        found = []
        for line in sys.stdin:
            lref, lsha, rref, rsha = (line.split() + ["", "", "", ""])[:4]
            if not lsha or lsha == ZERO:
                continue
            found += owner_commits_bad(revs_not_on(rsha, lsha))
        if found:
            refuse(found, "this push")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
