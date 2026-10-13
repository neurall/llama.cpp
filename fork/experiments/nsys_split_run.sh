#!/bin/bash
# t.sh NAME BUILD "<llama args>" ["extra env for the wrapped run"] : one GLM-like OLMoE prefill run (3 passes) under nsys, through run.py; prints prompt t/s and the per-layer device table
NAME=$1; BUILD=$2; ARGS=$3; XENV=$4
U=/p/bw/p2p/userspace-595/usr/lib/x86_64-linux-gnu; S=/p/bw/p2p/bin595/nvidia-smi; M=/m/o/8/4/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
D=/p/bw/data/q5h/nsys
for i in $(seq 1 60); do u=$($S --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); [ "$(echo $u | awk '{print ($1<60 && $2<60)}')" = 1 ] && break; sleep 2; done
cd /p/bw/rels/devbuilds
env PERF_LD_EXTRA=$U NVIDIA_SMI=$S python3 /p/bw/wt-main/fork/run.py bench --models $M --builds $BUILD --variant "v=$ARGS" --variants v -t pf4k --clamp 3500 -n 1 --repeat 3 --ctx 4096 --tokens 4 \
  --wrap "env GGML_NVTX=1 GGML_PF_STATS=1 $XENV nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none -o $D/$NAME --force-overwrite=true" > $D/$NAME.run.out 2>&1
echo "== $NAME ($BUILD) exit $?"
grep -a 'prompt eval time' /tmp/perf-$BUILD.log | tail -3 | sed 's/.*(//; s/)//'
grep -a 'pf-stats sync' /tmp/perf-$BUILD.log | tail -1 | cut -c1-260
grep -aiE 'error|abort|assert|nan|CUDA error' /tmp/perf-$BUILD.log | head -5
rm -f $D/$NAME.sqlite; nsys export --type sqlite -o $D/$NAME.sqlite $D/$NAME.nsys-rep > /dev/null 2>&1
ls -la $D/$NAME.sqlite 2>/dev/null | awk '{print $5, $9}'
