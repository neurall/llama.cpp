# Try the fork on a rented multi-GPU box (RunPod, vast.ai, ...)

`pod.sh` does the whole thing on one Linux + NVIDIA machine: describe the box, get this fork's binaries and build upstream llama.cpp,
run the same short tests on both with 1, 2 and 4 GPUs, put everything in one tarball.

## Pick the box

- **GPUs:** the interesting case is a model bigger than the VRAM (the cache only matters then). On 4 x RTX 3090 (96 GB) that means 100 GB+ weights.
- **Host:** at least 256 GB RAM and 32 vCPUs. Missed experts are computed on the CPU, so the host's memory bandwidth and cores decide as much as the GPUs.
- **PCIe:** a listing rarely says how the GPUs are connected. `pod.sh report` finds out in two minutes (below), before you pay for downloads.
- **Image:** one that has `nvcc` (a CUDA `-devel` image). The release binary needs a recent driver (about 580 or newer, `nvidia-smi`); on an older one `setup` builds the fork
  from source, which needs `nvcc` too.
- **Disk:** a volume at `/workspace` big enough for the model plus about 15 GB for the two builds.

## Run it

```bash
# 1. what is this box? (topology, live PCIe links, host -> GPU bandwidth alone and all GPUs at once)
bash <(curl -fsSL https://raw.githubusercontent.com/neurall/llama.cpp/release/tools/cloud/pod.sh) report

# 2. get a model (wget / hf download into /workspace/models), then everything else:
bash <(curl -fsSL https://raw.githubusercontent.com/neurall/llama.cpp/release/tools/cloud/pod.sh) all /workspace/models/MODEL.gguf
```

Single steps: `report`, `setup` (fork release binary or source build, upstream built from source at a pinned commit, `STOCK_REF`), `bench MODEL.gguf ...`, `pack`.
Useful env: `GPUS="1 2 4"`, `TESTS="t100 pf12k"` (short decode, 12k-token prompt), `RUNS=2`, `THREADS`, `FORK_BUILD=1`, `POD_DIR` (default `/workspace/pod`).
Both builds run the way a user would start them (no flags), cold, `-n` runs each; rows go to `results/run-history.csv` (`hw=pod-<n>gpu`, campaign `pod`), the table to `results/summary.txt`.

## Reading `report`

- `nvidia-smi topo -m`: `PIX`/`PXB` between GPUs means they sit behind a PCIe switch and share its uplink to the CPU; `PHB`/`NODE` means a host bridge; `SYS` means across CPU sockets.
- GPU to GPU: "peer access 0" (GeForce) means copies go through host memory and share its bandwidth. `memsat` shows the thread count where host memory bandwidth saturates, and how a CPU-and-GPU load splits it.
- The bandwidth table: a GPU on a good x16 gen4 link copies at about 25 GB/s (a x4 link about 6). "All at once" is what the cache does when GPUs upload misses together;
  a large drop from "alone" means a shared link, and more GPUs then add little for the cache.
- Link gen and width are read right after the copies (an idle GPU drops to a lower generation).
- `nproc` and the cgroup lines show how much CPU and RAM the container really gets, which can be less than the host has.
