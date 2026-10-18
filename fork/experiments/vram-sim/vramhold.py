# usage: vramhold.py GPU:GB [GPU:GB ...]  -- holds that much VRAM per GPU until killed (simulates a smaller card)
# It ends by itself when the process that started it is gone (a killed `run.py cell` used to leave it holding 8 GiB per card for hours, which blocked the next
# cell's wait for free GPU memory and shrank every later run: 2026-10-18). There is at most one: a second copy ends the older ones first.
import ctypes, os, signal, sys, time
parent = os.getppid()
try:
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG: the kernel kills this process when its parent exits
except OSError:
    pass
if os.getppid() != parent:
    sys.exit(0)
me = os.getpid()
for d in os.listdir("/proc"):
    if d.isdigit() and int(d) != me:
        try:
            if b"vramhold.py" in open(f"/proc/{d}/cmdline", "rb").read():
                os.kill(int(d), signal.SIGKILL)
        except OSError:
            pass
time.sleep(1.0)
import torch
keep = [torch.empty(int(float(g) * 2**30), dtype=torch.uint8, device=int(d)) for d, g in (a.split(":") for a in sys.argv[1:])]
print("holding", sys.argv[1:], flush=True)
while os.getppid() == parent:   # also covers a parent that was replaced by init
    time.sleep(1)
