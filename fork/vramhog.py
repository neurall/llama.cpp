#!/usr/bin/env python3
"""vramhog.py FREE_MIB [LIBDIR]: hold VRAM on every GPU until the process ends (SIGTERM) so that FREE_MIB is left on each card: a REAL hold, for the binaries that cannot fake free memory (stock has no --moe sim-vram). The fork's own server fakes it (run.py -sim); stock cells under -sim get this."""
import ctypes, glob, os, signal, sys, time


def main():
    free_mib = int(sys.argv[1]); libdir = sys.argv[2] if len(sys.argv) > 2 else "."
    lib = ctypes.CDLL(next(iter(sorted(glob.glob(os.path.join(libdir, "libcudart.so*")))), "libcudart.so"))
    n = ctypes.c_int(); lib.cudaGetDeviceCount(ctypes.byref(n)); held = []
    for d in range(n.value):
        lib.cudaSetDevice(d); fr = ctypes.c_size_t(); tot = ctypes.c_size_t(); lib.cudaMemGetInfo(ctypes.byref(fr), ctypes.byref(tot))
        take = fr.value - free_mib * 2**20
        if take > 0:
            p = ctypes.c_void_p()
            if lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(take)) == 0: held.append((d, take, p))
        print(f"vramhog: card {d}: {max(take, 0) >> 20} MiB held, {free_mib} MiB left free", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print("vramhog: ready", flush=True)
    while True: time.sleep(3600)


if __name__ == "__main__": main()
