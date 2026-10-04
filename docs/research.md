# Research logs: how the hot experts of a MoE model behave

This fork can record what a MoE model does while you use it, to study which experts stay hot, how that depends on the topic, and how well a router's choice can be predicted.
Everything here is **off by default**, writes only when you ask, and costs no speed while the model runs. One switch controls it, `--moe log=LETTERS` (or `LLAMA_MOE_LOG=LETTERS`);
the letters, with the other fork options, are listed in [fork-knobs.md](fork-knobs.md). Please share what you collect.

| letter | file | one line per | what it holds |
|---|---|---|---|
| `e` | `embhot.csv` | run | `model,embhex,hots`: the last layer embedding of the last token of the last graph (1 byte per float as hex, scaled by the largest value) and this run's hot experts per layer (`layer:count count ...;layer:...`) |
| `s` (or `i`) | `state-snapshots/<date>_<time>-<model>.ini` | run (one file) | the model's own section of the learned state: lifetime hot-expert counts, tuner and placement records |
| `l64` | `layertrace.csv` | layer of every 64th generated token | `model,pos,layer,embhex,ids`: the layer's output embedding and the expert ids its router selected for that token |
| `g`, `p`, `t`, `c` | log, `moe-route.txt`, `moe-trace*` | | GPU identity of the placement record, cache stats every 64 steps, routed expert ids of every MoE layer, decode-step trace (`t` and `c` slow the decode path; use them for diagnosis only) |
| `r` | `tools/bench/run-history.csv`, `tools/runs.db` | run | read by `tools/run.py` (the benchmark harness), not by the engine |

`log=esl32` combines letters; a number right after a letter is its argument (`l32`: every 32nd token). File locations can be moved with `LLAMA_MOE_EMBHOT=FILE`, `LLAMA_MOE_SNAP_DIR=DIR`, `LLAMA_MOE_LAYERTRACE=FILE`, `GGML_MOE_LOG=FILE`.
Files are appended to and never rewritten, so a server that runs for weeks grows them: leave the switch off unless you are collecting data.

## Questions the logs can answer

1. **Do similar prompts use the same experts?** For every pair of runs, compare the cosine similarity of their `embhex` vectors with the overlap of their `hots` (top experts per layer, Jaccard, averaged over layers). Near pairs should overlap more than far pairs (code against prose, maths, translation); if they do, a start can preload the hot experts of the closest earlier topic instead of one average map.
2. **Which experts settle?** The state snapshots show, run after run, which experts stay hot and which drift.
3. **Can the router be predicted?** `layertrace.csv` pairs a layer's input state with the experts chosen at that layer and the next ones: data to train an expert predictor outside the engine.

The embedding saved by `e` is the last token of the last graph, so it is the model's state after its own answer, not the prompt alone; it carries the whole context. Whether it is the best topic key is one of the things to measure.
`python3 tools/runlog.py` (SQLite copy of the run history, table `embhot` for the `e` lines) and numpy are enough to analyse it; no vector database is needed at this scale.

## Status

`e`, `s`, `g`, `p`, `t`, `c` ship in the current release. `l` (layer trace) is in the source and not yet in a release build. The near-versus-far study and a topic prompt set are planned, not done.
