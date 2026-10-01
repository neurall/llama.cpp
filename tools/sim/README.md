# Simulations and trace analysis

Offline tools for the expert cache, no GPU needed.

- `cache_sim.py`, `cost_sim.py`, `alloc_sim.py`: per-layer cache policies (LRU, LFU, decayed counts, Belady ...), a link-rate-limited token
  cost model and slot allocation, replayed on `GGML_MOE_LOG` router traces (`--nexp` = the model's expert count).
- `pred_lab.py`, `router_predict.py`, `expert_input_drift.py`: the expert predictor lab and its inputs.
- `replay.py`, `tl_layers.py`: trace replay and per-layer timelines.
- `pool_sim.py`: policies for one slot pool shared by all layers (window / decayed use, lifetime share, churn stay, hidden Markov model, co-occurrence) against the
  per-layer score, replayed on router traces. It predicted gains that did not appear on hardware (see `tools/bench/README.md`, "Shared slot pool").
- `nsys_breakdown.py`, `nsys_layer.py`, `nsys_overlap.py`, `tr_summary.py`: summaries of nsys captures and `LLAMA_MOE_CACHE_TRACE` files.

The benchmark harness is `tools/run.py`; its prompts and the run database are in `tools/bench/`.
