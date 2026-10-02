#!/bin/bash
# run.sh NAME DEVS "HOLD" MODEL [N]   e.g. run.sh 1x16 1 "0:8" model.gguf
NAME=$1 DEVS=$2 HOLD=$3 MODEL=$4 N=${5:-300}; D=$(dirname $0); OUT=$D/out; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$DEVS
python3 $D/vramhold.py $HOLD > $OUT/hold.log 2>&1 & HP=$!
until grep -q holding $OUT/hold.log; do sleep 1; done
P="Write a long story about a dragon."
for B in fork stock; do
  if [ $B = fork ]; then BIN=/p/bw/llama.cpp/build-dev/bin/llama-cli; EXTRA="--moe statslog=32"; else BIN=$HOME/lc/build/bin/llama-cli; EXTRA=""; fi
  for R in 1 2; do
    LLAMA_MOE_STATE=0 LLAMA_MOE_STATSLOG=$OUT/$NAME-$B-$R.txt timeout 900 $BIN -m $MODEL $EXTRA -n $N --temp 0 -st -p "$P" </dev/null > $OUT/$NAME-$B-$R.log 2>&1
    echo "$NAME $B run$R: $(grep -ao 'Generation: [0-9.]* t/s' $OUT/$NAME-$B-$R.log) hit $(tail -n 1 $OUT/$NAME-$B-$R.txt 2>/dev/null | awk '{print $4}')" | tee -a $OUT/summary.txt
  done
done
kill $HP
