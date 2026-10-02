# usage: vramhold.py GPU:GB [GPU:GB ...]  -- holds that much VRAM per GPU until killed (simulates a smaller card)
import sys, time, torch
keep = [torch.empty(int(float(g) * 2**30), dtype=torch.uint8, device=int(d)) for d, g in (a.split(":") for a in sys.argv[1:])]
print("holding", sys.argv[1:], flush=True)
while True: time.sleep(3600)
