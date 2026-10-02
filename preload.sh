#!/bin/bash
# Prepare 1 GiB huge pages for the pinned-weights cache. Root (sudo) is needed only to change the pool or the mount (once per boot); with a big
# enough pool and the mount in place it runs as a normal user.
#
#   ./preload.sh model.gguf [more.gguf ...]   reserve enough 1 GiB pages for the model(s) if the pool is too small, mount the cache, and load the
#                                                  model(s) once with the fork's binary so the cache is warm (set LLAMA_BIN to choose it)
#   ./preload.sh                                   reserve the most the machine can spare: total RAM minus a safety margin (default max(24 GiB, 20% of RAM),
#                                                  HUGEFS_MARGIN_GIB=N to change), and mount the cache
#   ./preload.sh 100G                              a number with G: reserve exactly that many GiB (manual pool size, no model; --pages N is the same)
#   ./preload.sh --pages N                         reserve N pages
#   sudo ./preload.sh --release                    delete the cached models, unmount, give the pages back
#
# Afterwards start the fork with   GGML_CUDA_HUGEFS=/mnt/huge1g   (the first load fills the cache, later loads map it and skip the disk read).
# Without the variable, or without a pool, the loader works as before. Needs a kernel with 1 GiB huge pages (CONFIG_CONTIG_ALLOC); do NOT
# boot with hugetlb_cma= (pages inside a CMA area cannot be pinned for the GPU).
set -eu
MNT=${HUGEFS_MOUNT:-/mnt/huge1g}
NR=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
FREE=/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages
[ -e "$NR" ] || { echo "this kernel has no 1 GiB huge pages ($NR missing)"; exit 1; }
ARGS="$*"
need_root() { [ "$(id -u)" = 0 ] || { echo "$1: the pool or the mount has to change, which needs root. Run:  sudo $0 $ARGS"; echo "(with a big enough pool and the mount in place no sudo is needed)"; exit 1; }; }

if [ "${1:-}" = --release ]; then
  need_root "release"
  if mountpoint -q "$MNT"; then rm -f "$MNT"/* 2>/dev/null || true; umount "$MNT"; fi
  echo 0 > "$NR"; echo "released: mount removed, 1 GiB pool $(cat $NR) pages"; exit 0
fi
grep -q hugetlb_cma /proc/cmdline && { echo "the kernel was booted with hugetlb_cma=: those pages cannot be pinned for the GPU, boot without it"; exit 1; }

total_gib=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) >> 20 ))
margin=${HUGEFS_MARGIN_GIB:-$(( total_gib / 5 > 24 ? total_gib / 5 : 24 ))}
max=$(( total_gib - margin ))
[ "$max" -gt 0 ] || { echo "not enough RAM for a pool (total $total_gib GiB, margin $margin GiB)"; exit 1; }
models=()
want=0
if [ "${1:-}" = --pages ]; then
  want=${2:?pages}
elif [[ "${1:-}" =~ ^[0-9]+[Gg]$ ]]; then
  want=${1%[Gg]}                    # a size, not a model file: the pool size chosen by hand
  [ "$want" -le "$max" ] || echo "note: $want GiB is above the computed limit of $max GiB (RAM $total_gib GiB, margin $margin GiB), using your size"
  max=$want
else
  bytes=0
  [ $# -ge 1 ] || bytes=$(( max << 30 ))     # no model given: the most that can be spared
  for m in "$@"; do
    models+=("$m")
    [ -f "$m" ] || { echo "no such file: $m"; exit 1; }
    base=${m%-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf}   # a split model: add every part
    if [ "$base" != "$m" ]; then parts=$(ls "$base"-*-of-*.gguf); else parts=$m; fi
    for p in $parts; do bytes=$((bytes + $(stat -c %s "$p"))); done
  done
  want=$(( (bytes >> 30) + 2 ))     # the pinned buffer holds at most the whole model, plus a page of slack for alignment
fi
if [ "$want" -gt "$max" ]; then
  if [ ${#models[@]} -gt 0 ]; then echo "the model(s) need $want GiB but at most $max GiB can be spared (RAM $total_gib GiB, margin $margin GiB): the cache cannot hold them (lower HUGEFS_MARGIN_GIB at your own risk)"; exit 1; fi
  want=$max
fi
have=$(cat "$NR")
if [ "$have" -lt "$want" ]; then
  need_root "reserving $want 1 GiB pages (now $have)"
  echo "reserving $want 1 GiB pages (now $have): dropping the page cache and compacting memory first"
  sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
  echo "$want" > "$NR"
fi
got=$(cat "$NR")
[ "$got" -ge "$want" ] || { echo "only $got of $want pages could be allocated (memory too fragmented or too small); free memory and retry, or reboot"; exit 1; }

if ! mountpoint -q "$MNT"; then
  need_root "mounting $MNT"
  mkdir -p "$MNT"
  mount -t hugetlbfs -o pagesize=1G,uid=${SUDO_UID:-0},gid=${SUDO_GID:-0},mode=0775 none "$MNT"
fi
echo "pool: $got x 1 GiB ($(cat $FREE) free of RAM $total_gib GiB, margin $margin GiB), mount: $MNT"
echo "start the fork with:  GGML_CUDA_HUGEFS=$MNT llama-server -m ... -lm pin"

# load each given model once (as the invoking user) so the cache is filled; a model whose cache file exists is only mapped
if [ ${#models[@]} -gt 0 ]; then
  here=$(cd "$(dirname "$0")" && pwd)
  bin=${LLAMA_BIN:-$(ls "$here"/build*/bin/llama-cli "$here"/bin/llama-cli 2>/dev/null | head -1)}
  [ -n "$bin" ] || bin=$(command -v llama-cli || true)
  [ -x "${bin:-/nonexistent}" ] || { echo "no llama-cli found: set LLAMA_BIN=/path/to/llama-cli and run again to fill the cache"; exit 0; }
  for m in "${models[@]}"; do
    echo "loading $m once to fill the cache ($bin)"
    asuser=(); [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && asuser=(sudo -u "$SUDO_USER")
    "${asuser[@]}" env GGML_CUDA_HUGEFS="$MNT" "$bin" -m "$m" -lm pin -c 64 -n 1 -p "hi" -st --no-warmup </dev/null >/dev/null 2>"${TMPDIR:-/tmp}/preload-$$.log" || true
    grep -h "hugetlbfs cache" "${TMPDIR:-/tmp}/preload-$$.log" 2>/dev/null | tail -1 || true
    rm -f "${TMPDIR:-/tmp}/preload-$$.log"
  done
  ls -la "$MNT"
fi
