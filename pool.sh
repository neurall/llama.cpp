#!/bin/bash
# The 1 GiB huge-page pool behind the pinned-weights cache. Allocate it once; the loader does the rest (it fills a cache file on the first load of
# a model, maps it afterwards, and evicts the least recently used model that nobody has mapped when the pool is full).
#
#   sudo ./pool.sh mount              reserve the most the machine can spare (total RAM minus a margin: max(24 GiB, 20% of RAM), HUGEFS_MARGIN_GIB=N)
#                                     and mount the cache at /mnt/huge1g
#   sudo ./pool.sh mount 100G         reserve exactly that many GiB (--pages N is the same in pages)
#   sudo ./pool.sh mount model.gguf   reserve what that model needs (split models: pass the first part)
#   sudo ./pool.sh unmount            delete the cached models, unmount, give the pages back
#
# Root (sudo) is only needed to change the pool or the mount. Afterwards start the fork with   GGML_CUDA_HUGEFS=/mnt/huge1g   (the first load of a
# model fills the cache, later loads map it and skip the disk read). Without the variable, or without a pool, the loader works as before.
# Needs a kernel with 1 GiB huge pages (CONFIG_CONTIG_ALLOC); do NOT boot with hugetlb_cma= (pages inside a CMA area cannot be pinned for the GPU).
set -eu
MNT=${HUGEFS_MOUNT:-/mnt/huge1g}
NR=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
FREE=/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages
[ -e "$NR" ] || { echo "this kernel has no 1 GiB huge pages ($NR missing)"; exit 1; }
ARGS=""
need_root() { [ "$(id -u)" = 0 ] || { echo "$1: the pool or the mount has to change, which needs root. Run:  sudo $0 $ARGS"; echo "(with a big enough pool and the mount in place no sudo is needed)"; exit 1; }; }

cmd=${1:-}; [ $# -ge 1 ] && shift
case "$cmd" in mount|unmount) ;; *) sed -n '2,14p' "$0"; exit 1;; esac
ARGS="$cmd $*"
if [ "$cmd" = unmount ]; then
  need_root "unmount"
  if mountpoint -q "$MNT"; then rm -f "$MNT"/* 2>/dev/null || true; umount "$MNT"; fi
  echo 0 > "$NR"; echo "unmounted: cache files deleted, mount removed, 1 GiB pool $(cat $NR) pages"; exit 0
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
  if [ ${#models[@]} -gt 0 ]; then echo "the model(s) need $want GiB but at most $max GiB can be spared (RAM $total_gib GiB, margin $margin GiB): the cache cannot hold them. Only the part of the model that stays in host memory is cached (less than the file when layers go to the GPUs): give the size by hand, e.g. sudo $0 ${max}G, or lower HUGEFS_MARGIN_GIB at your own risk"; exit 1; fi
  want=$max
fi
have=$(cat "$NR")
if [ "$have" -lt "$want" ]; then
  need_root "reserving $want 1 GiB pages (now $have)"
  echo "reserving $want 1 GiB pages (now $have): dropping the page cache and compacting memory first"
  # defragment, then allocate; pages a try gets are kept, so repeating the same request with compaction in between gets further each time (it stops when
  # three tries in a row add nothing). Proactive compaction is switched to its most aggressive setting for the duration and restored afterwards.
  pc=/proc/sys/vm/compaction_proactiveness
  old_pc=""; [ -w "$pc" ] && { old_pc=$(cat "$pc"); echo 100 > "$pc"; }
  sync; echo 3 > /proc/sys/vm/drop_caches
  last=$have; idle=0; try=0
  while [ "$(cat "$NR")" -lt "$want" ] && [ "$idle" -lt 3 ] && [ "$try" -lt 12 ]; do
    try=$((try + 1))
    echo 1 > /proc/sys/vm/compact_memory
    echo "$want" > "$NR"
    now=$(cat "$NR")
    echo "  try $try: $now of $want pages"
    if [ "$now" -gt "$last" ]; then idle=0; else idle=$((idle + 1)); sleep 2; fi
    last=$now
  done
  [ -n "$old_pc" ] && echo "$old_pc" > "$pc"
fi
got=$(cat "$NR")
[ "$got" -ge "$want" ] || { echo "only $got of $want pages could be allocated (memory too fragmented or too small, even after repeated compaction); free memory and retry, a smaller size may work, or reboot"; exit 1; }

if ! mountpoint -q "$MNT"; then
  need_root "mounting $MNT"
  mkdir -p "$MNT"
  mount -t hugetlbfs -o pagesize=1G,uid=${SUDO_UID:-0},gid=${SUDO_GID:-0},mode=0775 none "$MNT"
fi
echo "pool: $got x 1 GiB ($(cat $FREE) free of RAM $total_gib GiB, margin $margin GiB), mount: $MNT"
echo "start the fork with:  GGML_CUDA_HUGEFS=$MNT llama-server -m ... -lm pin"
