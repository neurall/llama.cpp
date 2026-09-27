#!/bin/bash
# last finished round of each loop: gain over NLMS on the same tokens at L+1/4/8
cd "$(dirname "$0")"
for f in loop_*.log; do
  last=$(grep -n "^== " "$f" | tail -2 | head -1 | cut -d: -f1)
  [ -z "$last" ] && continue
  tail -n +"$last" "$f" | awk -v f="${f#loop_}" '
    /^== /{r=$3" "$4; next}
    /\[this run\]/ && /(nlms x|NLMS fallback)/ {split($0,a," "); l=a[1]; name=$0; sub(/ \[this run\].*/,"",name); sub(/^L\+[0-9] /,"",name); next}
    /vs nlms/ && l ~ /L\+(1|4|8)$/ {printf "%-10s r%-3s %-45s %s %s\n", f, r, substr(name,1,45), l, $NF}'
done | sort -u
