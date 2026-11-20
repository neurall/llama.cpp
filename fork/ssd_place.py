#!/usr/bin/env python3
"""Place a model's experts on more drives for llama-server --multissd (ggml/src/ggml-read-fork.h reads each block from the least busy drive that holds it).

  python3 fork/ssd_place.py MODEL_FILE DIR[:SHARE],DIR[:SHARE],... [--trim] [--dry]
  python3 fork/run.py ssd MODEL_FILE DIR[:SHARE],...            (the same)

MODEL_FILE: one file of the model; every shard of it (NAME-000NN-of-000MM.gguf beside it) is placed. The original files stay whole (they serve any block).
Each DIR gets a file of each shard's name and size that holds its share of every layer's experts and nothing else (a sparse file: the rest are holes).
  SHARE: optional; by default each drive's share of the summed read speed of all drives (fio, 256 KiB reads, 32 in flight, on a large file of that drive),
         the original's drive counted too: with 6.1 and 4.3 GB/s the copy gets 0.41. Two drives of one speed: 0.5.
  Expert e of a tensor goes to copy c when frac((e + 0.5) * 0.618..) lies in c's stripe of [0, sum of shares): every copy holds its share of each tensor's
  experts spread over the whole index range, so reads in any order (pool fill, stream, decode misses) find work for every drive. Ranges on the 4 KiB grid.
  The reader spreads what two drives hold by how fast each empties its queue: a copy needs at least its drive's share of the speed, more costs only space.
  --trim: a DIR that already holds whole copies gets the rest punched out (fallocate punch-hole) instead of new reads. Without it whole copies are left as they are.
Every range is checked on sampled blocks against the original. Reads and writes go past the page cache (O_DIRECT)."""
import argparse, ctypes, ctypes.util, json, mmap, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gguf-py"))
ALIGN = 4096             # placed ranges start and end on the file system's and O_DIRECT's 4 KiB grid (a sector two experts share goes with both)
GOLD = 0.6180339887498949
CHUNK = 64 << 20


def parse_dirs(text):
    """'DIR[:SHARE],...' -> [(dir, share or None)]: the share is the text after the last ':' when it is a number in (0, 1] (ggml-read-fork.h sr_parse_ssd)"""
    out = []
    for e in (x.strip() for x in text.split(",")):
        if not e: continue
        share = None
        k = e.rfind(":")
        if 0 <= k < len(e) - 1:
            try:
                v = float(e[k + 1:])
                if 0.0 < v <= 1.0: share, e = v, e[:k]
            except ValueError: pass
        out.append((e.rstrip("/\\") or e, share))
    return out


def shares_from_speed(rates):
    """each drive's share of the summed rate (the original's drive first); the copies' shares only"""
    tot = sum(rates)
    return [r / tot for r in rates[1:]] if tot > 0 else [1.0 / len(rates)] * (len(rates) - 1)


def stripes(shares):
    """each copy's part [lo, hi) of [0, 1): [0, s1), [s1, s1 + s2), ..."""
    out, lo = [], 0.0
    for x in shares:
        out.append((lo, min(1.0, lo + x)))
        lo = min(1.0, lo + x)
    return out


def tensor_ranges(off, n_exp, esz, lo, hi, size):
    """the 4 KiB-grid byte ranges of the experts of a tensor at off whose frac((e + 0.5) * GOLD) lies in [lo, hi), merged"""
    out = []
    for e in range(n_exp):
        u = ((e + 0.5) * GOLD) % 1.0
        if not (lo <= u < hi): continue
        a, b = (off + e * esz) // ALIGN * ALIGN, min(size, -(-(off + (e + 1) * esz) // ALIGN) * ALIGN)
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def merge(ranges):
    out = []
    for a, b in sorted(r for r in ranges if r):
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def expert_tensors(path):
    """(data offset, experts, bytes an expert) of every expert tensor of a GGUF file (name *_exps*, three dimensions)"""
    from gguf import GGUFReader
    r = GGUFReader(path, "r")
    out = []
    for t in r.tensors:
        if "_exps" not in t.name or len(t.shape) != 3: continue
        n_exp = int(t.shape[-1])
        if n_exp > 1 and t.n_bytes % n_exp == 0: out.append((int(t.data_offset), n_exp, int(t.n_bytes) // n_exp))
    return out


def shards(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m: return [path]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def fio_rate(path):
    """GB/s of O_DIRECT reads of path (256 KiB, 32 in flight, 2 s), None without fio"""
    if not shutil.which("fio"): return None
    r = subprocess.run(["fio", "--name=p", f"--filename={path}", "--readonly", "--rw=read", "--direct=1", "--ioengine=io_uring", "--bs=256k", "--iodepth=32",
                        "--runtime=2", "--size=2G", "--output-format=json"], capture_output=True, text=True)
    try: return json.loads(r.stdout)["jobs"][0]["read"]["bw_bytes"] / 1e9
    except (ValueError, KeyError, IndexError): return None


def probe_file(d, names):
    """a large file on the drive of d: a shard copy already there, else the largest file of at least 1 GiB under d"""
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p) and os.path.getsize(p) >= 1 << 30: return p
    best = None
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try: sz = os.path.getsize(p)
            except OSError: continue
            if sz >= 1 << 30 and (best is None or sz > best[0]): best = (sz, p)
    return best[1] if best else None


class Direct:
    """aligned O_DIRECT reads and writes of one file; the unaligned tail of a file through a plain descriptor"""
    def __init__(self, path, write=False):
        self.path, self.write = path, write
        flags = (os.O_RDWR if write else os.O_RDONLY)
        self.fd = os.open(path, flags | os.O_DIRECT)
        self.plain = os.open(path, flags)
        self.buf = mmap.mmap(-1, CHUNK)
    def read(self, off, n):
        if off % 4096 == 0 and n % 4096 == 0:
            mv = memoryview(self.buf)[:n]
            got = os.preadv(self.fd, [mv], off)
            return bytes(mv[:got])
        return os.pread(self.plain, n, off)
    def put(self, off, data):
        if off % 4096 == 0 and len(data) % 4096 == 0:
            self.buf[:len(data)] = data
            return os.pwritev(self.fd, [memoryview(self.buf)[:len(data)]], off)
        return os.pwrite(self.plain, data, off)
    def close(self):
        os.close(self.fd); os.close(self.plain); self.buf.close()


def copy_ranges(src, dst, ranges, size):
    if not os.path.exists(dst) or os.path.getsize(dst) != size:
        with open(dst, "wb") as f: f.truncate(size)   # sparse: what is not written stays a hole
    s, d = Direct(src), Direct(dst, write=True)
    try:
        for a, b in ranges:
            for o in range(a, b, CHUNK):
                n = min(CHUNK, b - o)
                d.put(o, s.read(o, n))
        os.fsync(d.plain)
    finally:
        s.close(); d.close()


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def punch_rest(dst, keep, size):
    """holes everywhere but keep (fallocate PUNCH_HOLE | KEEP_SIZE, on the 4 KiB grid)"""
    fd = os.open(dst, os.O_RDWR)
    try:
        pos = 0
        for a, b in keep + [(size, size)]:
            lo, hi = -(-pos // 4096) * 4096, a // 4096 * 4096
            if hi > lo and _libc.fallocate(fd, 3, ctypes.c_int64(lo), ctypes.c_int64(hi - lo)) != 0:
                raise OSError(ctypes.get_errno(), f"fallocate punch-hole {dst} {lo}+{hi - lo}")
            pos = b
        os.fsync(fd)
    finally:
        os.close(fd)


def check(src, dst, ranges, samples=3):
    s, d = Direct(src), Direct(dst)
    try:
        for a, b in ranges:
            for k in range(samples):
                o = a + (b - a - min(ALIGN, b - a)) * k // max(1, samples - 1) // 4096 * 4096
                n = min(ALIGN, b - o)
                if s.read(o, n) != d.read(o, n): return f"bytes differ at {o} of {dst}"
    finally:
        s.close(); d.close()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model"); ap.add_argument("dirs"); ap.add_argument("--trim", action="store_true"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    files = [f for f in shards(os.path.realpath(a.model)) if os.path.exists(f)]
    if not files: sys.exit(f"{a.model}: no model file")
    dirs = parse_dirs(a.dirs)
    src_dev = os.stat(files[0]).st_dev
    dirs = [(d, s) for d, s in dirs if os.path.realpath(d) != os.path.dirname(files[0]) and os.path.isdir(d)]
    if not dirs: sys.exit("no copy directory (the original's own directory is not one)")
    for d, _ in dirs:
        if os.stat(d).st_dev == src_dev: print(f"{d}: on the original's drive: a copy there adds no read speed")
    names = [os.path.basename(f) for f in files]
    shares = [s for _, s in dirs]
    if any(s is None for s in shares):
        rates = [fio_rate(files[0])] + [fio_rate(p) if (p := probe_file(d, names)) else None for d, _ in dirs]
        if any(r is None for r in rates): sys.exit(f"read speeds {rates}: give the shares (DIR:SHARE) or put a large file on each drive (fio needed)")
        auto = shares_from_speed(rates)
        shares = [s if s is not None else x for s, x in zip(shares, auto)]
        print("read speeds GB/s: " + ", ".join(f"{p} {r:.2f}" for p, r in zip([os.path.dirname(files[0])] + [d for d, _ in dirs], rates)))
    if sum(shares) > 1.0 + 1e-9: sys.exit(f"shares {shares} sum to more than 1")
    st = stripes(shares)
    tot = [0] * len(dirs)
    for f in files:
        size = os.path.getsize(f)
        ten = expert_tensors(f)
        for i, (d, _) in enumerate(dirs):
            rg = merge(r for o, n, e in ten for r in tensor_ranges(o, n, e, st[i][0], st[i][1], size))
            tot[i] += sum(b - x for x, b in rg)
            dst = os.path.join(d, os.path.basename(f))
            whole = os.path.exists(dst) and os.path.getsize(dst) == size and os.stat(dst).st_blocks * 512 >= size - (1 << 20)
            act = "keep (whole)" if whole and not a.trim else "trim" if whole else "write"
            print(f"{dst}: experts {st[i][0]:.3f}-{st[i][1]:.3f} of each tensor, {sum(b - x for x, b in rg) / 2**30:.1f} GiB, {act}", flush=True)
            if a.dry or act == "keep (whole)": continue
            if act == "trim": punch_rest(dst, rg, size)
            else: copy_ranges(f, dst, rg, size)
            if (err := check(f, dst, rg)): sys.exit(err)
    for i, (d, _) in enumerate(dirs): print(f"{d}: share {shares[i]:.2f}, {tot[i] / 2**30:.1f} GiB of experts")
    return 0


def selfcheck():
    assert parse_dirs("/m:0.5,/m2/m:0.5") == [("/m", 0.5), ("/m2/m", 0.5)]
    assert parse_dirs("C:\\m,D:\\m:0.41,/m2/k/") == [("C:\\m", None), ("D:\\m", 0.41), ("/m2/k", None)]
    assert abs(shares_from_speed([6.1, 4.3])[0] - 0.4135) < 1e-3 and shares_from_speed([5.0, 5.0]) == [0.5]
    st = stripes([0.25, 0.25])
    assert abs(st[0][0]) < 1e-12 and abs(st[0][1] - 0.25) < 1e-12 and abs(st[1][0] - 0.25) < 1e-12 and abs(st[1][1] - 0.5) < 1e-12
    n, esz, off = 384, 5 * 2**20 + 12345, 10 * 4096 + 100
    r = tensor_ranges(off, n, esz, 0.0, 0.41, 10**13)
    held = sum(b - a for a, b in r)
    assert all(a % 4096 == 0 and b % 4096 == 0 for a, b in r) and abs(held / (n * esz) - 0.41) < 0.02, held / (n * esz)
    mid = off + n * esz // 2   # spread: the copy holds about as much of the tensor's first half as of its second
    low = sum(max(0, min(b, mid) - a) for a, b in r)
    assert abs(low - (held - low)) < 0.1 * held, (low, held - low)
    assert merge([(0, 5), (3, 9), (20, 30), None]) == [(0, 9), (20, 30)]
    return True


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]: print("ssd_place selfcheck:", selfcheck()); sys.exit(0)
    sys.exit(main())
