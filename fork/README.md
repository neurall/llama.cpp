# fork/

Everything of this fork that is not engine code lives here, so the engine diff against upstream stays small and a PR can be cut without it.
Engine code stays where the build needs it: `src/llama-moecache-fork.*`, `src/llama-moestate-fork.*` and our edits to upstream files.

| path | contents |
|---|---|
| `run.py` | the single Python file of the fork: every benchmark run, build, regression check and the huge-page pool (`run.py bench`, `build`, `quick`, `regress`, `hugepool`, `exp`), the run log (`runlog`, `runq`, `runlog-all`, `merge-runs`), power-limit and regression tools (`plcal`, `baseline`, `reg-tests`, `bisect`, `pick-features`), `series` and `selftest`; `run.py --help` lists them, each has its own `--help`. Machine paths come from its config section (`~/.config/llama-fork/config.toml`) |
| `bisect_glm.sh` | `git bisect run` helper for a GLM regression (`run.py publish` pushes the branch and replaces runs.db) |
| `resident_pin.sh` | one-time setup of the 1 GiB huge-page pool for fast pinned loads (users); `run.py hugepool` does the same for our runs |
| `moe-state.ini` | seed state file (hot experts per model): copy to `~/.cache/llama.cpp/moe-state.ini` if you have none |
| `vnni-check.cpp` | CPU feature check |
| `bench/` | frozen prompts, run history (`run-history.csv`), stats, campaigns, GPU logs |
| `sim/` | offline cache simulators and trace tools |
| `cloud/`, `iobench/`, `charts/` | rented-box helpers, disk I/O micro-benchmarks, chart scripts |
| `docs/` | fork documentation: [`fork-knobs.md`](docs/fork-knobs.md) (every flag, `--moe` setting and environment variable), the cache refactor design spec |
| `experiments/` | one directory per experiment: README (question, method, tables, decision), scripts, small sample data |

Run everything through `python3 fork/run.py ...` from the repository root.

## Branches

| branch | role |
|---|---|
| `master` | exact mirror of upstream `ggml-org/llama.cpp` master, fast-forward only |
| `main` | upstream plus the fork's patches: the product; releases are tags `release-b<build>-<hash>` on it |
| `data` | run history (`run-history.csv`, append only) behind the numbers in the README; no code |

Older experiment branches are kept as `archive/<name>` tags only.
