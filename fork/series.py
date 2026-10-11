#!/usr/bin/env python3
"""series.py [--base 6753a033f] [--tip main] [--branch series] [--tree /p/bw/wt-series]

Builds the branch `series` from the fork's tip: starting at the upstream commit the fork is based on, ONE COMMIT PER AREA (see fork/PATCHES.md) carrying the
diff of the paths of that area. A path that belongs to no area goes into the last commit, so `git diff series main` is empty. The commits are a reading aid
(what does the fork change, area by area) and the base of upstream pull requests: only the last commit is guaranteed to build, because areas share files.
The temporary worktree is removed afterwards; nothing is pushed."""
import argparse, os, subprocess, sys

# area name, commit title, path prefixes (first match wins; later areas take what is left)
AREAS = [
    ("cuda-loader", "CUDA: pinned weight loading (huge page pool, THP hybrid, registered pages), multi-threaded loader, load-mode",
     ["ggml/src/ggml-cuda/ggml-cuda.cu", "src/llama-model-loader.cpp"]),
    ("scheduler", "scheduler: host-barrier refinement, selective expert copy hook, split timer (GGML_SCHED_PROF), device-to-host async",
     ["ggml/src/ggml-backend.cpp", "ggml/include/ggml-backend.h", "ggml/include/ggml.h", "ggml/src/ggml.c"]),
    ("cpu-kernels", "CPU/CUDA kernels: Q2_0 AVX2 dot, repack and matmul fixes",
     ["ggml/src/ggml-cpu/", "ggml/src/ggml-cuda/"]),
    ("expert-cache", "MoE expert cache and self-tuner (llama-moecache-fork, llama-moestate-fork) with their graph and context hooks",
     ["src/llama-moecache-fork.", "src/llama-moestate-fork.", "src/llama-graph.", "src/llama-context.", "src/llama-cparams.h", "src/llama-ext.h", "src/llama.cpp",
      "include/llama.h", "src/CMakeLists.txt", "examples/eval-callback/"]),
    ("placement-and-flags", "automatic placement, thread tuner, --moe / --fork / --load-mode flags, server and CLI hooks",
     ["common/", "tools/cli/", "tools/server/"]),
    ("mtp-and-models", "GLM MTP draft head, hybrid-idx memory, conversion fixes",
     ["src/models/", "src/llama-model.", "src/llama-arch.", "src/llama-hparams.h", "src/llama-kv-cache.", "src/llama-memory-", "conversion/", "convert_hf_to_gguf.py"]),
    ("fork-tooling", "fork/: benchmark harness, run log, docs, experiments, build recipe; README and ignore rules",
     ["fork/", "README.md", ".gitignore"]),
]


def git(*a, cwd=None, check=True):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="6753a033f"); ap.add_argument("--tip", default="main"); ap.add_argument("--branch", default="series")
    ap.add_argument("--tree", default="/p/bw/wt-series")
    a = ap.parse_args()
    repo = os.getcwd()
    changes = {}
    for line in git("diff", "--name-status", "--no-renames", a.base, a.tip).splitlines():
        st, path = line.split("\t", 1)
        changes[path] = st
    left = dict(changes)
    git("worktree", "add", "--detach", "-f", a.tree, a.base)
    try:
        git("checkout", "-q", "-B", a.branch, a.base, cwd=a.tree)
        for i, (name, title, prefixes) in enumerate(AREAS):
            last = i == len(AREAS) - 1
            mine = [p for p in left if any(p.startswith(x) for x in prefixes)] if not last else list(left)
            if last:   # fork-tooling takes what no earlier area claimed; engine files among them are listed so an area can be added
                mine = list(left)
                stray = [p for p in mine if not any(p.startswith(x) for x in prefixes)]
                if stray:
                    print("not in any area (put into the last commit):", *stray[:20], sep="\n  ")
            for p in mine:
                if left[p] == "D":
                    git("rm", "-q", "--ignore-unmatch", p, cwd=a.tree)
                else:
                    git("checkout", a.tip, "--", p, cwd=a.tree)
                del left[p]
            if git("status", "--short", cwd=a.tree).strip():
                git("commit", "-q", "-m", title, cwd=a.tree)
                print(f"{name}: {len(mine)} files", flush=True)
        left_diff = git("diff", "--stat", a.branch, a.tip, cwd=repo).strip()
        print("git diff series main:", left_diff or "empty")
    finally:
        git("worktree", "remove", "--force", a.tree, check=False)


if __name__ == "__main__":
    main()
