#!/bin/bash
# CPU kernel report for one machine: builds the CPU backend of this tree (no CUDA) and runs the dot-product benchmarks. Needs cmake, ninja, clang or gcc.
# Usage:  fork/tests/cpu-kernel-report.sh [build-dir]        Output: a markdown report on stdout (also saved as <build-dir>/report.md); paste it into an issue.
# What it shows:  CPU and cache sizes, single-thread memory read speed, ns per 256 weights of every quant type in cache and from RAM, the thread count at which each
# type hits the memory wall, and how the MUL_MAT_ID tile (weight rows x tokens) behaves for short and long rows. ~5 minutes.
set -eu
SRC=$(cd "$(dirname "$0")/../.." && pwd); B=${1:-$SRC/build-cpu-report}
CC=${CC:-$(command -v clang || command -v gcc)}; CXX=${CXX:-$(command -v clang++ || command -v g++)}
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" >/dev/null
ninja -C "$B" ggml-cpu ggml-base ggml >/dev/null
for t in dotbench dotcheck dotmt tilebench; do
  $CXX -O2 -march=native -pthread -I "$SRC/ggml/include" -I "$SRC/ggml/src" "$SRC/fork/tests/$t.cpp" -o "$B/$t" -L"$B/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"$B/bin"
done
{
echo "# CPU kernel report"
echo; echo "commit: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null)  compiler: $($CC --version | head -1)"
echo; echo '```'; lscpu | grep -E "Model name|Socket|Thread|Core|L1d|L2|L3|NUMA node\(s\)|MHz" ; echo '```'
echo; echo "## per quant type, one thread (GB/s of weights read; ns per 256 weights)"
echo; echo "In cache (0.25 MB working set: the kernel's own compute speed):"; echo '```'; "$B/dotbench" 0.25; echo '```'
echo; echo "From RAM (96 MB; for machines with a bigger L3 pass a bigger size by running dotbench by hand):"; echo '```'; "$B/dotbench" 96; echo '```'
echo; echo "From 1 GB (past any cache):"; echo '```'; "$B/dotbench" 1024 | head -20; echo '```'
echo; echo "## threads: aggregate GB/s (where each type hits the memory wall)"; echo '```'; "$B/dotmt"; echo '```'
echo; echo "## MUL_MAT_ID tile: ns per (weight row x token), rows per tile across, tokens down"
for n in 4096 16384; do echo; echo "row length $n:"; echo '```'; "$B/tilebench" $n; echo '```'; done
echo; echo "## accuracy (sums must match between builds; compare the 'bits' column)"; echo '```'; "$B/dotcheck"; echo '```'
} | tee "$B/report.md"
