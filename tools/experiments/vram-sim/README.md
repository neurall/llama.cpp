# Simulated smaller cards (2026-10-02, PC1: 2 x RTX 3090)

`vramhold.py GPU:GB ...` holds VRAM so fork and stock both see a smaller card; `run.sh NAME DEVS "HOLD" MODEL` runs fork (with `--moe statslog=32`) and stock twice each (llama-cli, temp 0).
Clocks (needs root, undo with `-rgc` / `-rmc`): `nvidia-smi -i 0,1 -lgc 900,900` and `-lmc 4700,4700` (about 0.6x compute, about half the memory bandwidth). GPU 0 is on a x4 chipset link (about 5 GB/s), GPU 1 on x16.
`logs/`: stats files (`moe-stats` columns in the header), generation lines trimmed from the run logs, `summary.txt`.

Qwen3.8-Flash-Next IQ1_M (55 GB), decode t/s, text differs between runs (not bit-identical):

| setup | share in VRAM | fork | stock |
|---|---|---|---|
| 1 x 16 GB (8 GB held) | about 24% | 28.5-28.8 | 18.3-19.5 |
| 2 x 16 GB | about 58% | 30.4-30.8 | 33.2-34.9 |

2 x 16 GB: `--moe cache=0` gives 33.9 (stock 34.7), the default cache 30.9, so the cache costs about 10% there (hit 94-99%, PCIe about 0.7 GB/s). The first-run placement rule has no stock branch above about 50% in VRAM.

2 x 24 GB (nothing held, full clocks), same model: fork 52.3-53.2 t/s, stock 66.4-69.5 t/s (0.78x) at 280 W per card and at the default 350+370 W, so the power cap changed nothing here.
Peak per card with the default limits: 218 W / 234 W, 69 / 65 C (`logs/mon-2x24-pl350-370.csv`, `mon.sh`; a gap of about 75 s in it, between the fork and stock runs, is a monitor parsing bug that was fixed).
Every run is also in tools/bench/run-history.csv (campaign `vram-sim`, `pl` = power limits in env). `tools/run.py` now records `pl` for local runs.
