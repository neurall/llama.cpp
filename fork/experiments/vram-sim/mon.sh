#!/bin/bash
# mon.sh OUTFILE: 1 Hz temp/power/clock log; HOT at 87 C, LOST when a GPU leaves the PCI bus or nvidia-smi fails; never stops the run
echo "t,gpu,temp,power,sm_mhz" >> $1
while true; do
  n=$(lspci -d 10de: 2>/dev/null | grep -c "VGA\|3D")
  [ "$n" -lt 2 ] && echo "$(date +%s) LOST: lspci sees $n NVIDIA GPUs" >> $1
  out=$(nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>&1) || echo "$(date +%s) LOST: nvidia-smi failed: $out" >> $1
  echo "$out" | while IFS=, read i t p c; do t=${t# }; p=${p# }; c=${c# }; [[ "$t" =~ ^[0-9]+$ ]] || continue; echo "$(date +%s),$i,$t,$p,$c" >> $1; [ "$t" -ge 87 ] && echo "$(date +%s) HOT $t on GPU $i" >> $1; done
  sleep 1
done
