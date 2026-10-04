# Research logs: how the hot experts of a MoE model behave

This fork can record what a MoE model does while you use it, to study which experts stay hot, how that depends on the topic, and how well a router's choice can be predicted.
Everything here is **off by default**, writes only when you ask, and costs no speed while the model runs. One switch controls it, `--moe log=LETTERS`;
the letters, with the other fork options, are listed in [fork-knobs.md](../../../docs/fork-knobs.md). Please share what you collect.

| letter | file | one line per | what it holds |
|---|---|---|---|
| `e` | `embhot.csv` | run | `model,embhex,hots`: the last layer embedding of the last token of the last graph (1 byte per float as hex, scaled by the largest value) and this run's hot experts per layer (`layer:count count ...;layer:...`) |
| `s` (or `i`) | `state-snapshots/<date>_<time>-<model>.ini` | run (one file) | the model's own section of the learned state: lifetime hot-expert counts, tuner and placement records |
| `l` | `layertrace.csv` | layer of every Nth generated token (`l`: every token, `l64`: every 64th) | `model,pos,layer,embhex,xhex,ids`: the layer's output embedding (scaled by its largest value), the router input (the normalised residual the experts see; fixed scale, clipped at ±16, 1 byte per dim) and the expert ids its router selected for that token |
| `x` | `routing_<model>.csv` | like `l`, ids only | `pos,layer,ids` (one file per model): the experts the router chose at every layer of every sampled token, the real routing flow to replay eviction policies on (`tools/sim/route_evict.py`) |
| `g`, `p`, `c` | log, log, `moe-trace*` | | GPU identity of the placement record, cache stats every 64 steps, decode-step trace (`c` slows the decode path; use it for diagnosis only). The routed expert ids of every MoE layer are a backend diagnostic: `GGML_MOE_LOG=FILE` (big) |
| `r` | `tools/bench/run-history.csv`, `tools/runs.db` | run | read by `tools/run.py` (the benchmark harness), not by the engine |

`log=esl32` combines letters; a number right after a letter is passed to that letter's handler as its argument (`l32`: every 32nd token, plain `l`: every token; `p500`: cache stats every 500 steps, plain `p`: every 64). All files go to the directory given with `--moe logdir=DIR` (default: the working directory).
Files are appended to and never rewritten, so a server that runs for weeks grows them: leave the switch off unless you are collecting data.

## Questions the logs can answer

1. **Do similar prompts use the same experts?** For every pair of runs, compare the cosine similarity of their `embhex` vectors with the overlap of their `hots` (top experts per layer, Jaccard, averaged over layers). Near pairs should overlap more than far pairs (code against prose, maths, translation); if they do, a start can preload the hot experts of the closest earlier topic instead of one average map.
2. **Which experts settle?** The state snapshots show, run after run, which experts stay hot and which drift.
3. **Can the router be predicted?** `layertrace.csv` pairs a layer's input state with the experts chosen at that layer and the next ones: data to train an expert predictor outside the engine.

The embedding saved by `e` is the last token of the last graph, so it is the model's state after its own answer, not the prompt alone; it carries the whole context. Whether it is the best topic key is one of the things to measure.
`python3 tools/runlog.py` (SQLite copy of the run history, table `embhot` for the `e` lines) and numpy are enough to analyse it; no vector database is needed at this scale.

## Levers for repeatable experiments

These do not log anything; they hold or change the engine so that runs can be compared. All are `--moe` keys, the defaults are what ships (details and evidence in [fork-knobs.md](../../../docs/fork-knobs.md)).

| lever | what it does for a study |
|---|---|
| `--moe fixed=1` | fixes every knob and turns tuning off, so repeated runs are the same configuration |
| `--moe mode=stock` or `cache` | forces the placement; `retest` forgets the saved decision and measures again |
| `--moe state=PATH` or `0` | uses another state file, or none: each arm of an experiment can have its own state (the regression tests do this) |
| `--moe tuned=...` | starts from a given tuner result instead of the saved one |
| `--moe policy=...` | eviction score: `add`, `window`, `hybrid`, `halve`, to compare policies on the same trace |
| `--moe ctl=FILE` | a control file whose settings are re-read while the model runs, to change a knob mid-run |
| `--moe pred-top=M,train-every=N` and `predict-*` | the learned expert predictor: depth, margin, step size, where it is saved (`..._FILE`), see the predictor table in the knobs document |
| `--fork off` | the whole fork off, plain upstream behaviour: the control arm of any comparison |
| `GGML_SCHED_DEBUG`, `GGML_SCHED_DEBUG_REALLOC`, `LLAMA_BATCH_DEBUG`, `LLAMA_KV_CACHE_DEBUG`, `LLAMA_GRAPH_INPUT_DEBUG`, `LLAMA_GRAPH_RESULT_DEBUG` (environment variables) | upstream's dump switches for the graph scheduler, batches, KV cache and graph inputs and results (verbose, for diagnosis) |
| `LLAMA_TRACE` | upstream's trace output |

The experimental, unproven knobs (`ev-cld`, `pin-hot`, `idle-up`, `l3-pf`, expert deferral `defer`) are listed with their measured effect in the knobs document; none beat the default in our tests.

## Status

`e`, `s`, `g`, `p`, `t`, `c` ship in the current release. `l` (layer trace) is in the source and not yet in a release build. The near-versus-far study and a topic prompt set are planned, not done.
