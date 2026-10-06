#!/usr/bin/env python3
"""iobench_win.py FILE [--all] [--grid] [--sec 0.25]: iobench.c for Windows (ctypes, no compiler needed).
Unbuffered (FILE_FLAG_NO_BUFFERING) random reads of aligned blocks: engine "iocp" = overlapped ReadFile with `depth` reads in flight on one I/O completion port,
engine "threads" = `depth` threads, each with its own handle and synchronous ReadFile. Hill climb from 1 MiB x 8 (one parameter one step while the gain is >= 3%),
choice = the setting with the fewest bytes in flight that reaches 95% of the best bandwidth seen. CPU ms/GB = process CPU time (user + kernel) per GB."""
import ctypes, ctypes.wintypes as W, random, statistics, sys, threading, time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
HANDLE = ctypes.c_void_p
k32.CreateFileW.restype = HANDLE
k32.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, HANDLE]
k32.CreateIoCompletionPort.restype = HANDLE
k32.CreateIoCompletionPort.argtypes = [HANDLE, HANDLE, ctypes.c_size_t, W.DWORD]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, W.DWORD, W.DWORD]
k32.ReadFile.argtypes = [HANDLE, ctypes.c_void_p, W.DWORD, ctypes.c_void_p, ctypes.c_void_p]
k32.GetQueuedCompletionStatus.argtypes = [HANDLE, ctypes.POINTER(W.DWORD), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p), W.DWORD]
k32.CloseHandle.argtypes = [HANDLE]
k32.GetFileSizeEx.argtypes = [HANDLE, ctypes.POINTER(ctypes.c_longlong)]
INVALID = HANDLE(-1).value
BS = [128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20]
QD = [1, 2, 4, 8, 16, 32, 64]


class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", W.DWORD), ("OffsetHigh", W.DWORD), ("hEvent", HANDLE)]


def opn(path, overlapped):
    h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | (0x40000000 if overlapped else 0), None)   # GENERIC_READ, SHARE_READ, OPEN_EXISTING, NO_BUFFERING [| OVERLAPPED]
    if h in (None, INVALID):
        raise OSError(ctypes.get_last_error(), "CreateFileW " + path)
    return h


def alloc(n):
    return k32.VirtualAlloc(None, n, 0x3000, 4)   # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE (page aligned)


def run_iocp(path, size, bs, qd, sec):
    h = opn(path, True)
    port = k32.CreateIoCompletionPort(h, None, 0, 0)
    buf = alloc(bs * qd); ovs = [OVERLAPPED() for _ in range(qd)]; t0 = [0.0] * qd; lat = []; nblk = size // bs; bytes_ = 0
    addr = {ctypes.addressof(o): i for i, o in enumerate(ovs)}
    rng = random.Random(1)

    def submit(i):
        off = rng.randrange(nblk) * bs
        ovs[i].Offset = off & 0xFFFFFFFF; ovs[i].OffsetHigh = off >> 32; ovs[i].Internal = 0; ovs[i].InternalHigh = 0
        t0[i] = time.perf_counter()
        ok = k32.ReadFile(h, buf + i * bs, bs, None, ctypes.byref(ovs[i]))
        if not ok and ctypes.get_last_error() != 997:   # ERROR_IO_PENDING
            raise OSError(ctypes.get_last_error(), "ReadFile")

    start = time.perf_counter(); c0 = time.process_time()
    for i in range(qd):
        submit(i)
    n = W.DWORD(); key = ctypes.c_size_t(); po = ctypes.c_void_p(); inflight = qd
    while time.perf_counter() - start < sec:
        if not k32.GetQueuedCompletionStatus(port, ctypes.byref(n), ctypes.byref(key), ctypes.byref(po), 5000):
            break
        i = addr[po.value]; t = time.perf_counter(); lat.append((t - t0[i]) * 1e6); bytes_ += bs; submit(i)
    el = time.perf_counter() - start; c1 = time.process_time()
    for _ in range(inflight):   # drain what is still in flight
        if not k32.GetQueuedCompletionStatus(port, ctypes.byref(n), ctypes.byref(key), ctypes.byref(po), 5000):
            break
    k32.CloseHandle(h); k32.CloseHandle(port)
    gbs = bytes_ / el / 1e9
    return gbs, statistics.median(lat) if lat else 0, (c1 - c0) * 1e3 / max(gbs * el, 1e-9)


def run_threads(path, size, bs, qd, sec):
    done = [0] * qd; stop = time.perf_counter() + sec; lat = []; nblk = size // bs

    def work(k):
        h = opn(path, False); buf = alloc(bs); rng = random.Random(k + 1); got = W.DWORD(); ov = OVERLAPPED()
        while time.perf_counter() < stop:
            off = rng.randrange(nblk) * bs; ov.Offset = off & 0xFFFFFFFF; ov.OffsetHigh = off >> 32; t = time.perf_counter()
            if not k32.ReadFile(h, buf, bs, ctypes.byref(got), ctypes.byref(ov)):
                break
            if k == 0:
                lat.append((time.perf_counter() - t) * 1e6)
            done[k] += bs
        k32.CloseHandle(h)

    start = time.perf_counter(); c0 = time.process_time()
    ts = [threading.Thread(target=work, args=(k,)) for k in range(qd)]
    [t.start() for t in ts]; [t.join() for t in ts]
    el = time.perf_counter() - start; c1 = time.process_time(); gbs = sum(done) / el / 1e9
    return gbs, statistics.median(lat) if lat else 0, (c1 - c0) * 1e3 / max(gbs * el, 1e-9)


def search(path, size, engine, sec, grid, quiet):
    fn = run_iocp if engine == "iocp" else run_threads
    R = {}
    t0 = time.perf_counter()

    def M(b, q):
        if (b, q) not in R:
            R[(b, q)] = fn(path, size, BS[b], QD[q], sec)
            if not quiet:
                print(f"{BS[b] >> 10:5d}k {QD[q]:4d} {R[(b, q)][0]:7.2f} {R[(b, q)][1]:8.0f} {R[(b, q)][2]:9.1f}", flush=True)
        return R[(b, q)]

    if not quiet:
        print(f"engine {engine}, {sec:.2f} s per point\n{'bs':>6} {'qd':>4} {'GB/s':>7} {'p50 us':>8} {'cpu ms/GB':>9}")
    b, q = 3, 3
    if grid:
        for i in range(len(BS)):
            for j in range(len(QD)):
                M(i, j)
    else:
        while True:
            best = M(b, q)[0]; nb, nq = b, q
            for db, dq in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ib, iq = b + db, q + dq
                if 0 <= ib < len(BS) and 0 <= iq < len(QD) and M(ib, iq)[0] > best * 1.03:
                    best = M(ib, iq)[0]; nb, nq = ib, iq
            if (nb, nq) == (b, q):
                break
            b, q = nb, nq
    mx = max(v[0] for v in R.values())
    cb, cq = min((k for k, v in R.items() if v[0] >= 0.95 * mx), key=lambda k: BS[k[0]] * QD[k[1]])
    r = R[(cb, cq)]
    print(f"{engine:<10} best {mx:.2f} GB/s; choice: {BS[cb] >> 10:5d} KiB x depth {QD[cq]:2d} = {r[0]:.2f} GB/s (p50 {r[1]:6.0f} us, {r[2]:5.1f} cpu ms/GB), {(BS[cb] * QD[cq]) >> 10} KiB in flight; {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    path = a[0]; sec = float(a[a.index("--sec") + 1]) if "--sec" in a else 0.25
    h = opn(path, False); sz = ctypes.c_longlong(); k32.GetFileSizeEx(h, ctypes.byref(sz)); k32.CloseHandle(h)
    for eng in (["iocp", "threads"] if "--all" in a else ["iocp"]):
        search(path, sz.value, eng, sec, "--grid" in a, "--all" in a)
