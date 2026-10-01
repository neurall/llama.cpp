#!/usr/bin/env bash
# Rented Linux + NVIDIA box (RunPod, vast.ai, ...): what the box is, this fork vs upstream on it, results in one tarball.
#   bash pod.sh report                  system, PCIe topology and live links, host -> GPU bandwidth alone and all GPUs at once
#   bash pod.sh setup                   fork binaries (latest GitHub release, or built from source) and upstream llama.cpp (built from source)
#   bash pod.sh bench MODEL.gguf ...    short decode and 12k-prompt tests, stock vs fork, on 1, 2, 4 GPUs (as many as the box has)
#   bash pod.sh matrix MODEL.gguf       fork cold / hot (saved state) and every stock build, short and long prompts, with the server's autotune lines and hit rate
#   bash pod.sh rebuild [NAME] [REF]    the fork from source at a branch or commit into builds/NAME (ccache: quick), bench it with FORK_NAME=NAME
#   bash pod.sh pack                    one tarball of everything under $POD_DIR/results
#   bash pod.sh all MODEL.gguf          report, setup, bench, pack
# Env: POD_DIR (/workspace/pod), STOCK_REF (upstream commit), GPUS ("1 2 4"), TESTS ("t100 pf12k"), RUNS (2), SEED=/path/state.ini (start the fork from a saved state, no cold run), NOSTOCK=1 (no upstream cells), CACHE=force (fork cells with --cpu-moe --moe-expert-cache -1), RESULTS_DIR, THREADS (half the cores, at most 24), FORK_BUILD=1 (never use the release binary), STOCK_BUILD=stock-NAME (which upstream build bench uses), SKIP_STOCK=1 (no upstream build), STOCK_TAG=b11323 (a prebuilt upstream release instead of a source build)
set -uo pipefail
POD=${POD_DIR:-/workspace/pod}; FORK=neurall/llama.cpp; STOCK=ggml-org/llama.cpp; STOCK_REF=${STOCK_REF:-def4d406ae2c2f39573120d68730fbb7760b24bf}
RES=${RESULTS_DIR:-$POD/results}
mkdir -p "$POD/builds" "$RES"
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
  local out="$RES/system.txt"
  {
    date; hostname; echo
    nvidia-smi -L; echo
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv; echo
    nvidia-smi topo -m; echo
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)|L3'; echo "nproc (cores this container may use): $CORES"
    echo "board: $(cat /sys/class/dmi/id/board_vendor 2>/dev/null) $(cat /sys/class/dmi/id/board_name 2>/dev/null)"
    e=/sys/devices/system/edac/mc/mc0   # populated memory channels, when the kernel exposes them (a container may not)
    [ -d $e ] && echo "memory (EDAC): $(ls -d $e/rank* | wc -l) ranks, channels $(cat $e/rank*/dimm_location | grep -o 'channel [0-9]*' | sort -u | tr -d '\n'), controller max: $(cat $e/max_location)"
    [ -r /sys/fs/cgroup/cpu.max ] && echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
    [ -r /sys/fs/cgroup/memory.max ] && echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
    free -g | head -2; echo
    command -v nvcc > /dev/null && nvcc --version | tail -2
    df -h "$POD" | tail -1; echo
  } > "$out" 2>&1
  if command -v nvcc > /dev/null; then
    nvcc -O2 "$POD/fork-src/tools/cloud/h2d.cu" -o "$POD/h2d" >> "$out" 2>&1 && { echo "host -> GPU copy bandwidth (pinned):"; "$POD/h2d"; } >> "$out" 2>&1
    nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv >> "$out"   # right after the copies: the links are at full speed
    { echo; echo "GPU <-> GPU:"; nvcc -O2 "$POD/fork-src/tools/cloud/p2p.cu" -o "$POD/p2p" 2>&1 && "$POD/p2p" 256 1
      echo; echo "host memory: where it saturates, and how much is left for the GPUs:"
      nvcc -O3 -Xcompiler "-O3 -mavx2 -pthread" "$POD/fork-src/tools/cloud/memsat.cu" -o "$POD/memsat" 2>&1 && "$POD/memsat" 16; } >> "$out" 2>&1
  else echo "no nvcc: pick a -devel CUDA image to get the bandwidth numbers" >> "$out"; fi
  cat "$out"
}

deps() {
  local miss=""; for t in git cmake curl python3; do command -v $t > /dev/null || miss="$miss $t"; done
  command -v ccache > /dev/null || miss="$miss ccache"
  [ -z "$miss" ] && return 0
  command -v apt-get > /dev/null && [ "$(id -u)" = 0 ] && { apt-get update -qq; apt-get install -y -qq git cmake build-essential curl python3 ccache; return 0; }
  [ "$miss" = " ccache" ] && return 0   # only ccache missing and no way to install it: builds just take longer
  die "missing:$miss"
}

# ccache for every source build: one cache under POD_DIR, keyed by file content (CCACHE_BASEDIR / NOHASHDIR: the same sources in another directory still hit)
export CCACHE_DIR="$POD/ccache" CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-30G}" CCACHE_BASEDIR="$POD" CCACHE_NOHASHDIR=1 CCACHE_COMPILERCHECK=content

cuda_build() {   # cuda_build SRC_DIR NAME: cmake build of llama-server + llama-cli, binaries and libs into builds/NAME
  command -v nvcc > /dev/null || die "nvcc not found: use a CUDA -devel image (or the fork's release binary: no source build needed)"
  # an old cmake (Ubuntu 24.04: 3.28) cannot find the CUDA 13 libraries; the toolkit image may lack the cuBLAS headers and link
  local cm; cm=$(cmake --version | head -1 | grep -o '[0-9]*\.[0-9]*' | head -1)
  [ "$(printf '%s\n3.31\n' "$cm" | sort -V | head -1)" = 3.31 ] || { pip install -q --break-system-packages cmake 2>/dev/null || pip install -q cmake; hash -r; }
  [ -e "$(dirname "$(dirname "$(command -v nvcc)")")/include/cublas_v2.h" ] || { v=$(nvcc --version | grep -o 'release [0-9]*\.[0-9]*' | grep -o '[0-9]*\.[0-9]*' | tr . -)
    apt-get install -y -qq "libcublas-dev-$v" "cuda-cudart-dev-$v" > /dev/null 2>&1 || log "no cuBLAS dev package for CUDA $v: the build may fail"; }
  cmake -S "$1" -B "$1/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=native \
    $(command -v ccache > /dev/null && echo "-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache") > "$RES/cmake-$2.log" 2>&1 \
    && cmake --build "$1/build" -j "$CORES" --target llama-server llama-cli >> "$RES/cmake-$2.log" 2>&1 || die "build of $2 failed, see $RES/cmake-$2.log"
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
  if [ -n "${STOCK_TAG:-}" ] && [ ! -x "$POD/builds/stock-$STOCK_TAG/llama-server" ]; then
    log "upstream $STOCK $STOCK_TAG: release binary"
    local su; su=$(python3 - <<PY
import json, re, urllib.request
r = json.load(urllib.request.urlopen("https://api.github.com/repos/$STOCK/releases/tags/$STOCK_TAG"))
a = {x["name"]: x["browser_download_url"] for x in r["assets"]}
for pat in (r"^llama-.*bin-ubuntu-cuda.*-x64\.tar\.gz$", r"^cudart-llama-.*bin-ubuntu-cuda.*-x64\.tar\.gz$"):
    m = [(tuple(int(v) for v in re.search(r"cuda-(\d+)\.(\d+)", n).groups()), u) for n, u in a.items() if re.match(pat, n)]
    print(max(m)[1] if m else "")   # the newest CUDA version
PY
)
    mkdir -p "$POD/builds/stock-$STOCK_TAG"
    for u in $su; do [ -n "$u" ] && curl -fsSL --retry 3 "$u" | tar xz -C "$POD/builds/stock-$STOCK_TAG" --strip-components=1; done
    works "stock-$STOCK_TAG" && sn="stock-$STOCK_TAG" || { log "upstream release binary cannot start CUDA here: source build"; rm -rf "$POD/builds/stock-$STOCK_TAG"; }
  fi
  if [ ! -x "$POD/builds/$sn/llama-server" ] && [ -z "$(ls "$POD/builds" | grep '^stock-')" ]; then
    log "upstream $STOCK ${STOCK_REF:0:9}: build from source"
    rm -rf "$POD/stock-src"; mkdir -p "$POD/stock-src"; git -C "$POD/stock-src" init -q
    git -C "$POD/stock-src" fetch -q --depth 1 "https://github.com/$STOCK" "$STOCK_REF" && git -C "$POD/stock-src" checkout -q FETCH_HEAD || die "cannot fetch upstream $STOCK_REF"
    cuda_build "$POD/stock-src" "$sn"
  fi
  log "builds: $(ls "$POD/builds" | tr '\n' ' ')"
}

rebuild() {   # rebuild [NAME=fork-new] [REF=release]: the fork from source at a branch or commit into builds/NAME (ccache makes it quick after the first time)
  deps; fork_src
  [ -n "${FORK_SRC:-}" ] || { git -C "$POD/fork-src" fetch -q --depth 1 origin "${2:-release}" && git -C "$POD/fork-src" checkout -q FETCH_HEAD || die "cannot fetch ${2:-release}"; }
  cuda_build "$POD/fork-src" "${1:-fork-new}" && works "${1:-fork-new}" && log "built builds/${1:-fork-new}; bench it with FORK_NAME=${1:-fork-new}"
}

bench() {
  [ $# -ge 1 ] || die "usage: pod.sh bench MODEL.gguf ..."
  fork_src
  local ngpu; ngpu=$(nvidia-smi -L | wc -l); local sn; sn=${STOCK_BUILD:-$(ls "$POD/builds" | grep "^stock-" | head -1)}
  local fk=${FORK_NAME:-fork}; [ -n "$sn" ] && [ -x "$POD/builds/$fk/llama-server" ] || die "run setup first"
  for m in "$@"; do
    for k in ${GPUS:-1 2 4}; do
      [ "$k" -le "$ngpu" ] || continue
      local vis; vis=$(seq -s, 0 $((k - 1))); local devs; devs=$(seq -s, 0 $((k - 1)) | sed 's/[0-9][0-9]*/CUDA&/g')
      for t in ${TESTS:-t100 pf12k}; do
        log "$(basename "$m"): $k GPU(s), test $t"
        CUDA_VISIBLE_DEVICES=$vis PERF_DEV=$devs PERF_THREADS=$THREADS PERF_BUILDS="$POD/builds" RUN_DATA="$RES" PERF_HW="pod-${k}gpu" MODEL="$m" \
          python3 "$POD/fork-src/tools/run.py" run -t "$t" -n "${RUNS:-2}" --no-warm --bare --campaign pod "$fk" "$sn" 2>&1 | tee -a "$RES/bench.log" | tail -4
      done
    done
  done
  RUN_DATA="$RES" python3 "$POD/fork-src/tools/run.py" show 2>&1 | tee "$RES/summary.txt"
}

matrix() {   # matrix MODEL: per GPU count and test: fork cold / hot and every stock build; server logs + the autotune parameters and hit rate from each
  [ $# -ge 1 ] || die "usage: pod.sh matrix MODEL.gguf"
  fork_src
  local m=$1 ngpu; ngpu=$(nvidia-smi -L | wc -l); local sn=${STOCK_BUILD:-$(ls "$POD/builds" | grep '^stock-' | head -1)} fk=${FORK_NAME:-fork}
  [ -n "$sn" ] && [ -x "$POD/builds/$fk/llama-server" ] || die "run setup first"
  mkdir -p "$RES/logs"
  local dl bt same=0 decided=""
  dl=$(ls "$POD/builds" | grep -E '^stock-b[0-9]{5}$' | head -1); bt=$(ls "$POD/builds" | grep '^stock-' | grep -v -E '^stock-b[0-9]{5}$' | head -1)
  for k in ${GPUS:-4}; do
    [ "$k" -le "$ngpu" ] || continue
    local vis devs; vis=$(seq -s, 0 $((k - 1))); devs=$(seq -s, 0 $((k - 1)) | sed 's/[0-9][0-9]*/CUDA&/g')
    for t in ${TESTS:-t100 chatv pf12k}; do
      local st="$POD/state-$k-$t.ini"; rm -f "$st"
      cell() {   # cell TAG BUILD [run.py flags]: one measured run; keeps its server log and pulls the cache and tuning lines out of it
        local tag=$1 b=$2; shift 2
        local bf=--bare; [ "$b" = "$fk" ] && [ "${CACHE:-auto}" = force ] && bf=""   # CACHE=force: the fork with the harness flags (--cpu-moe --moe-expert-cache -1): the cache on, whatever its own placement decides
        log "$(basename "$m") $k GPU(s) $t: $tag"
        CUDA_VISIBLE_DEVICES=$vis PERF_DEV=$devs PERF_THREADS=$THREADS PERF_BUILDS="$POD/builds" RUN_DATA="$RES" PERF_HW="pod-${k}gpu" MODEL="$m" \
          python3 "$POD/fork-src/tools/run.py" run -t "$t" -n 1 ${bf:+$bf} --campaign pod --note "$tag" -e "LLAMA_MOE_STATE=$st" "$@" "$b" 2>&1 | tee -a "$RES/matrix.log" | tail -2
        cp "/tmp/perf-$b.log" "$RES/logs/$k-$t-$tag.log" 2>/dev/null
        { echo "== $k GPU(s) $t $tag"; grep -h -E 'moe cache = on|moe-cache: (MoE|auto|self-tune|placement)|common_moe_cache_auto|tuned|self-tune|threads  *=|ubatch|n_ubatch|expert cache enabled|prefill links' "/tmp/perf-$b.log" 2>/dev/null | tail -14 | cut -c1-230; } >> "$RES/params.txt"
      }
      if [ -n "${SEED:-}" ] && [ -f "$SEED" ]; then   # HOT_ONLY: no cold run, the saved state of an earlier run (hot experts are per model, so any GPU count can use it)
        cp "$SEED" "$st"
      else
        cell cold "$fk" --no-warm    # the first run: no saved state
      fi
      cell hot "$fk" --no-warm       # the second run: starts from the state the first one saved
      # upstream: the downloaded release binary every time; the source build only while it differs from it (the first pair decides: within 5% decode speed = same, skipped from then on)
      [ "${NOSTOCK:-0}" = 1 ] && continue   # NOSTOCK=1: no upstream cells
      [ -n "$dl" ] && cell "$dl" "$dl" --no-warm
      if [ -n "$bt" ] && [ "$same" != 1 ]; then
        cell "$bt" "$bt" --no-warm
        if [ -n "$dl" ] && [ -z "$decided" ]; then
          decided=1
          same=$(RUN_DATA="$RES" python3 - "$dl" "$bt" "$t" <<'PY'
import csv, sys
dl, bt, t = sys.argv[1:4]
rows = [r for r in csv.DictReader(open(__import__("os").environ["RUN_DATA"] + "/run-history.csv")) if r["test"] == t and r.get("tps")]
a = [float(r["tps"]) for r in rows if r["note"] == dl]; b = [float(r["tps"]) for r in rows if r["note"] == bt]
print(1 if a and b and abs(a[-1] - b[-1]) / max(a[-1], b[-1]) < 0.05 else 0)
PY
)
          log "upstream source build vs downloaded binary: $([ "$same" = 1 ] && echo "same (within 5%), not repeated" || echo "different, kept in the matrix")"
        fi
      fi
    done
  done
  RUN_DATA="$RES" python3 "$POD/fork-src/tools/run.py" show 2>&1 | tee "$RES/summary.txt"
}

pack() {
  local f="$POD/pod-results-$(date +%Y%m%d-%H%M).tar.gz"
  tar czf "$f" -C "$POD" results && log "results: $f (runpodctl send $f, or scp)"
}

case "${1:-help}" in
  report) report ;;
  setup) setup ;;
  rebuild) shift; rebuild "$@" ;;
  bench) shift; bench "$@" ;;
  matrix) shift; matrix "$@" ;;
  pack) pack ;;
  all) shift; report; setup; bench "$@"; pack ;;
  *) sed -n '2,10p' "$0" ;;
esac
