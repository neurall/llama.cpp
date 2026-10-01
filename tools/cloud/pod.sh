#!/usr/bin/env bash
# Rented Linux + NVIDIA box (RunPod, vast.ai, ...): what the box is, this fork vs upstream on it, results in one tarball.
#   bash pod.sh report                  system, PCIe topology and live links, host -> GPU bandwidth alone and all GPUs at once
#   bash pod.sh setup                   fork binaries (latest GitHub release, or built from source) and upstream llama.cpp (built from source)
#   bash pod.sh bench MODEL.gguf ...    short decode and 12k-prompt tests, stock vs fork, on 1, 2, 4 GPUs (as many as the box has)
#   bash pod.sh pack                    one tarball of everything under $POD_DIR/results
#   bash pod.sh all MODEL.gguf          report, setup, bench, pack
# Env: POD_DIR (/workspace/pod), STOCK_REF (upstream commit), GPUS ("1 2 4"), TESTS ("t100 pf12k"), RUNS (2), THREADS (half the cores, at most 24), FORK_BUILD=1 (never use the release binary), SKIP_STOCK=1 (no upstream build)
set -uo pipefail
POD=${POD_DIR:-/workspace/pod}; FORK=neurall/llama.cpp; STOCK=ggml-org/llama.cpp; STOCK_REF=${STOCK_REF:-def4d406ae2c2f39573120d68730fbb7760b24bf}
mkdir -p "$POD/builds" "$POD/results"
log() { echo "[pod] $*"; }
die() { echo "[pod] $*" >&2; exit 1; }
CORES=$(nproc); THREADS=${THREADS:-$(( CORES / 2 > 24 ? 24 : (CORES / 2 < 4 ? 4 : CORES / 2) ))}

fork_src() {   # the fork's tools (run.py, prompts, h2d.cu) from GitHub
  if [ -n "${FORK_SRC:-}" ]; then ln -sfn "$FORK_SRC" "$POD/fork-src"; return; fi   # a local checkout instead (testing)
  if [ -d "$POD/fork-src/.git" ]; then git -C "$POD/fork-src" pull -q --ff-only || true
  else git clone -q --depth 1 -b release "https://github.com/$FORK" "$POD/fork-src" || die "cannot clone $FORK"; fi
}

report() {
  fork_src
  local out="$POD/results/system.txt"
  {
    date; hostname; echo
    nvidia-smi -L; echo
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv; echo
    nvidia-smi topo -m; echo
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)|L3'; echo "nproc (cores this container may use): $CORES"
    [ -r /sys/fs/cgroup/cpu.max ] && echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
    [ -r /sys/fs/cgroup/memory.max ] && echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
    free -g | head -2; echo
    command -v nvcc > /dev/null && nvcc --version | tail -2
    df -h "$POD" | tail -1; echo
  } > "$out" 2>&1
  if command -v nvcc > /dev/null; then
    nvcc -O2 "$POD/fork-src/tools/cloud/h2d.cu" -o "$POD/h2d" >> "$out" 2>&1 && { echo "host -> GPU copy bandwidth (pinned):"; "$POD/h2d"; } >> "$out" 2>&1
    nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv >> "$out"   # right after the copies: the links are at full speed
  else echo "no nvcc: pick a -devel CUDA image to get the bandwidth numbers" >> "$out"; fi
  cat "$out"
}

deps() {
  local miss=""; for t in git cmake curl python3; do command -v $t > /dev/null || miss="$miss $t"; done
  [ -z "$miss" ] && return 0
  command -v apt-get > /dev/null && [ "$(id -u)" = 0 ] && { apt-get update -qq && apt-get install -y -qq git cmake build-essential curl python3 ccache; return; }
  die "missing:$miss"
}

cuda_build() {   # cuda_build SRC_DIR NAME: cmake build of llama-server + llama-cli, binaries and libs into builds/NAME
  command -v nvcc > /dev/null || die "nvcc not found: use a CUDA -devel image (or the fork's release binary: no source build needed)"
  cmake -S "$1" -B "$1/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native > "$POD/results/cmake-$2.log" 2>&1 \
    && cmake --build "$1/build" -j "$CORES" --target llama-server llama-cli >> "$POD/results/cmake-$2.log" 2>&1 || die "build of $2 failed, see $POD/results/cmake-$2.log"
  rm -rf "$POD/builds/$2"; mkdir -p "$POD/builds/$2"; cp -a "$1/build/bin/." "$POD/builds/$2/"
}

works() { LD_LIBRARY_PATH="$POD/builds/$1" "$POD/builds/$1/llama-server" --list-devices 2>&1 | grep -q 'CUDA0'; }

setup() {
  deps; fork_src
  if [ "${FORK_BUILD:-0}" != 1 ] && [ ! -x "$POD/builds/fork/llama-server" ]; then
    log "fork: latest release binary"
    local urls; urls=$(python3 - <<PY
import json, re, urllib.request
r = json.load(urllib.request.urlopen("https://api.github.com/repos/$FORK/releases/latest"))
a = {x["name"]: x["browser_download_url"] for x in r["assets"]}
for pat in (r"^llama-release-.*ubuntu-cuda.*\.tar\.gz$", r"^cudart-llama-release-.*ubuntu-cuda.*\.tar\.gz$"):
    print(next((u for n, u in a.items() if re.match(pat, n)), ""))
PY
)
    mkdir -p "$POD/builds/fork"
    for u in $urls; do [ -n "$u" ] && { curl -fsSL --retry 3 "$u" | tar xz -C "$POD/builds/fork" --strip-components=1; }; done
    works fork || { log "release binary cannot start CUDA here (driver too old for its CUDA runtime?): building the fork from source"; rm -rf "$POD/builds/fork"; }
  fi
  [ -x "$POD/builds/fork/llama-server" ] || cuda_build "$POD/fork-src" fork
  works fork || die "fork build does not see a GPU"
  local sn="stock-${STOCK_REF:0:9}"
  [ "${SKIP_STOCK:-0}" = 1 ] && { log "builds: $(ls "$POD/builds" | tr '\n' ' ')"; return; }
  if [ ! -x "$POD/builds/$sn/llama-server" ]; then
    log "upstream $STOCK ${STOCK_REF:0:9}: build from source"
    rm -rf "$POD/stock-src"; mkdir -p "$POD/stock-src"; git -C "$POD/stock-src" init -q
    git -C "$POD/stock-src" fetch -q --depth 1 "https://github.com/$STOCK" "$STOCK_REF" && git -C "$POD/stock-src" checkout -q FETCH_HEAD || die "cannot fetch upstream $STOCK_REF"
    cuda_build "$POD/stock-src" "$sn"
  fi
  log "builds: $(ls "$POD/builds" | tr '\n' ' ')"
}

bench() {
  [ $# -ge 1 ] || die "usage: pod.sh bench MODEL.gguf ..."
  fork_src
  local ngpu; ngpu=$(nvidia-smi -L | wc -l); local sn; sn=$(ls "$POD/builds" | grep '^stock-' | head -1)
  [ -n "$sn" ] && [ -x "$POD/builds/fork/llama-server" ] || die "run setup first"
  for m in "$@"; do
    for k in ${GPUS:-1 2 4}; do
      [ "$k" -le "$ngpu" ] || continue
      local vis; vis=$(seq -s, 0 $((k - 1))); local devs; devs=$(seq -s, 0 $((k - 1)) | sed 's/[0-9][0-9]*/CUDA&/g')
      for t in ${TESTS:-t100 pf12k}; do
        log "$(basename "$m"): $k GPU(s), test $t"
        CUDA_VISIBLE_DEVICES=$vis PERF_DEV=$devs PERF_THREADS=$THREADS PERF_BUILDS="$POD/builds" RUN_DATA="$POD/results" PERF_HW="pod-${k}gpu" MODEL="$m" \
          python3 "$POD/fork-src/tools/run.py" run -t "$t" -n "${RUNS:-2}" --no-warm --bare --campaign pod fork "$sn" 2>&1 | tee -a "$POD/results/bench.log" | tail -4
      done
    done
  done
  RUN_DATA="$POD/results" python3 "$POD/fork-src/tools/run.py" show 2>&1 | tee "$POD/results/summary.txt"
}

pack() {
  local f="$POD/pod-results-$(date +%Y%m%d-%H%M).tar.gz"
  tar czf "$f" -C "$POD" results && log "results: $f (runpodctl send $f, or scp)"
}

case "${1:-help}" in
  report) report ;;
  setup) setup ;;
  bench) shift; bench "$@" ;;
  pack) pack ;;
  all) shift; report; setup; bench "$@"; pack ;;
  *) sed -n '2,10p' "$0" ;;
esac
