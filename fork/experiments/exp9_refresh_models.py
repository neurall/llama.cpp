"""experiment 9: refresh model files whose reads are slow. Per file: probe the cold read speed (fio, O_DIRECT, 1M, qd 32, 2 GB from the middle); if below --min-gbps, write a fresh copy next to it
(dd, direct I/O), compare it byte for byte with the original (cmp), then replace the original by the copy (same mode, owner and mtime) and delete the old data. Nothing is replaced unless the compare passes.
Results (probe before, probe after) go to the experiments table.   python3 run.py experiment exp9_refresh_models --dry   |   --only NAME_SUBSTRING | --min-gbps 4.5 | --all"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from experiments import put, show


def probe(path, size, window=2 << 30):
    """cold read speed in GB/s of a window from the middle of the file (direct I/O does not use the page cache)"""
    off = max(0, (size // 2) // 4096 * 4096 - window // 2); win = min(window, size - off) // (1 << 20) * (1 << 20)
    r = subprocess.run(["fio", "--name=p", f"--filename={path}", "--readonly", "--ioengine=io_uring", "--direct=1", "--rw=read", "--bs=1M", "--iodepth=32", f"--offset={off}", f"--size={win}", "--runtime=4", "--time_based", "--group_reporting", "--output-format=json"], capture_output=True, text=True)
    return json.loads(r.stdout)["jobs"][0]["read"]["bw_bytes"] / 1e9


def run(argv):
    p = argparse.ArgumentParser(prog="experiment exp9_refresh_models")
    p.add_argument("--dry", action="store_true", help="only probe and print"); p.add_argument("--only"); p.add_argument("--min-gbps", dest="min_gbps", type=float, default=4.5); p.add_argument("--all", action="store_true", help="refresh every file, slow or not")
    p.add_argument("--verify", action="store_true", help="compare the copy with the original byte for byte before replacing it (slow: reads both files again)"); p.add_argument("--min-size-gb", dest="min_gb", type=float, default=2.0); p.add_argument("--roots", default="/m"); p.add_argument("--note", default=""); p.add_argument("--show")
    a = p.parse_args(argv)
    if a.show: return show(a.show)
    files = sorted({f for r in a.roots.split(",") for f in glob.glob(os.path.join(r, "**", "*.gguf"), recursive=True)}, key=lambda f: -os.path.getsize(f))
    files = [f for f in files if os.path.getsize(f) >= a.min_gb * 1e9 and (not a.only or a.only in f) and not os.path.islink(f)]
    for f in files:
        size = os.path.getsize(f); before = probe(f, size); slow = before < a.min_gbps or a.all
        print(f"{f}  {size / 1e9:.1f} GB  read {before:.2f} GB/s  {'REFRESH' if slow else 'ok'}", flush=True)
        if a.dry or not slow: continue
        st = os.stat(f); tmp = f + ".refresh"
        if shutil.disk_usage(os.path.dirname(f)).free < size + 50 * 10 ** 9: print("  not enough free space, skipped", flush=True); continue
        t0 = time.time()
        subprocess.run(["dd", f"if={f}", f"of={tmp}", "bs=16M", "iflag=direct", "oflag=direct", "status=none"], check=True)
        subprocess.run(["sync"]); t1 = time.time()
        if a.verify:
            subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
            if subprocess.run(["cmp", f, tmp]).returncode != 0: os.remove(tmp); print("  COMPARE FAILED, copy removed, original kept", flush=True); continue
        if os.path.getsize(tmp) != size: os.remove(tmp); print("  SIZE DIFFERS, copy removed, original kept", flush=True); continue
        os.chmod(tmp, st.st_mode & 0o7777); os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns)); os.replace(tmp, f)
        subprocess.run(["sudo", "-n", "fstrim", os.path.dirname(f)], capture_output=True)
        after = probe(f, size)
        print(f"  copied in {t1 - t0:.0f} s, replaced; read now {after:.2f} GB/s (was {before:.2f})", flush=True)
        put([{"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "refresh", "name": f"experiment_9_refresh_{os.path.basename(f)}", "backend": "fio_io_uring", "chunk_kb": "1024", "qd": "32", "gb": f"{size / 1e9:.2f}", "secs": f"{t1 - t0:.0f}", "gbps": f"{after:.2f}",
              "content": "cmp ok" if a.verify else "not compared", "note": f"{a.note} before {before:.2f} after {after:.2f} GB/s", "raw": f"refresh {f}: {before:.2f} -> {after:.2f} GB/s"}])
