#!/usr/bin/env python3
"""Side GGUFs for --moe tensor-file: some of a model's tensors at another quant (fork/docs/fork-knobs.md, tensor-file).

  attn_side.py copy MODEL.gguf OUT.gguf [--tensors REGEX] [--check]
      the model's own tensors whose names match REGEX (default: Kimi K3.1 attention + ssm_g), byte for byte, with the
      model's metadata (llama-quantize needs it). Loaded with --moe tensor-file=OUT the model answers token for token as
      without it: the check of the loader switch.
  attn_side.py quant IN.gguf OUT.gguf FTYPE --bin PATH/llama-quantize [--threads N]
      llama-quantize --allow-requantize --pure IN OUT FTYPE: every tensor becomes FTYPE's base type (Q3_K_M -> Q3_K), on the
      CPU backend only (a copy of the binary beside the CPU libraries, so no GPU library is loaded). From a copy this
      requantizes: a speed test only, never a default (user 2026-11-13); from a full-resolution side file it quantizes once.

Every byte range read and every file written is dropped from the page cache afterwards: the rows after a run start cold.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
import gguf  # noqa: E402

# K3.1: the attention matrices the model holds as Q4_K (its README's type map) and the KDA gate ssm_g; not the indexer, norms, attn_k_b/v_b
K31_TENSORS = r'^blk\.\d+\.(attn_(q|k|v|output|gate|q_a|q_b|kv_a_mqa)|ssm_g)\.weight$'


def shard_paths(first):
    r = gguf.GGUFReader(first)
    f = r.fields.get('split.count')
    n = int(f.contents()) if f else 1
    if n == 1:
        return [(first, r)]
    m = re.match(r'(.*)-00001-of-(\d{5})\.gguf$', first)
    if not m or int(m.group(2)) != n:
        sys.exit(f'{first}: split.count = {n}, but the name is not the first of a -00001-of-{n:05d}.gguf set')
    return [(first, r)] + [(p, gguf.GGUFReader(p)) for p in (f'{m.group(1)}-{i:05d}-of-{n:05d}.gguf' for i in range(2, n + 1))]


def pread_all(fd, n, off):
    parts = []
    while n > 0:
        b = os.pread(fd, min(n, 1 << 30), off)
        if not b:
            raise OSError(f'short read at {off}')
        parts.append(b)
        n -= len(b)
        off += len(b)
    return b''.join(parts)


def drop(path):
    """written pages to the drive, then out of the page cache (a later row reads the file cold)"""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def copy_kv(reader, writer):
    for f in reader.fields.values():
        # GGUF.*: the header; split.*: one file; alignment: the writer pads with its own
        if f.name.startswith(('GGUF.', 'split.')) or f.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.ALIGNMENT):
            continue
        vt = f.types[0]
        writer.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)


def cmd_copy(a):
    rx = re.compile(a.tensors)
    shards = shard_paths(a.model)
    picked = [(p, t) for p, r in shards for t in r.tensors if rx.search(t.name)]
    if not picked:
        sys.exit(f'no tensor of {a.model} matches {a.tensors}')
    first = shards[0][1]
    total = sum(t.n_bytes for _, t in picked)
    types = sorted({t.tensor_type.name for _, t in picked})
    print(f'{len(picked)} tensors ({", ".join(types)}), {total / 2**30:.3f} GiB -> {a.out}', flush=True)

    w = gguf.GGUFWriter(a.out, arch=first.fields['general.architecture'].contents(), endianess=first.endianess)
    copy_kv(first, w)
    for _, t in picked:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    fds, sums = {}, {}
    try:
        for p, t in picked:
            fd = fds.setdefault(p, os.open(p, os.O_RDONLY))
            b = pread_all(fd, t.n_bytes, t.data_offset)   # not through the reader's mapping: these pages are dropped right after
            os.posix_fadvise(fd, t.data_offset, t.n_bytes, os.POSIX_FADV_DONTNEED)
            if a.check:
                sums[t.name] = hashlib.sha256(b).hexdigest()
            w.write_tensor_data(np.frombuffer(b, dtype=np.uint8))
    finally:
        for fd in fds.values():
            os.close(fd)
    w.close()
    drop(a.out)

    # what the loader will see: the same names, types, shapes and sizes (and with --check the same bytes)
    out = {t.name: t for t in gguf.GGUFReader(a.out).tensors}
    for _, t in picked:
        o = out.get(t.name)
        assert o is not None and o.tensor_type == t.tensor_type and list(o.shape) == list(t.shape) and o.n_bytes == t.n_bytes, t.name
        if a.check:
            fd = os.open(a.out, os.O_RDONLY)
            try:
                assert hashlib.sha256(pread_all(fd, o.n_bytes, o.data_offset)).hexdigest() == sums[t.name], f'{t.name}: bytes differ'
            finally:
                os.close(fd)
    if a.check:
        drop(a.out)
    print(f'ok: {len(out)} tensors in {a.out}, {os.path.getsize(a.out) / 2**30:.3f} GiB{", bytes checked" if a.check else ""}')


def cmd_quant(a):
    exe = os.path.realpath(a.bin)
    src = os.path.dirname(exe)
    n_in = len(gguf.GGUFReader(a.inp).tensors)
    with tempfile.TemporaryDirectory(prefix='attn_side-') as d:
        # the backend loader looks beside the executable's real path and in the working directory: a copy of the binary with the
        # CPU libraries only, run from there, cannot load the GPU backends (no CUDA or Vulkan context on cards that run rows)
        shutil.copy2(exe, d)
        for f in os.listdir(src):
            gpu_backend = f.startswith('libggml-') and not f.startswith(('libggml-base', 'libggml-cpu'))   # cuda, vulkan, hip, ...
            if f.startswith('lib') and '.so' in f and not gpu_backend:
                os.symlink(os.path.join(src, f), os.path.join(d, f))
        cmd = [os.path.join(d, os.path.basename(exe)), '--allow-requantize', '--pure', os.path.abspath(a.inp), os.path.abspath(a.out), a.ftype]
        cmd += [str(a.threads)] if a.threads else []
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=d, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    drop(a.inp)
    drop(a.out)
    out = gguf.GGUFReader(a.out).tensors
    assert len(out) == n_in, f'{len(out)} tensors out, {n_in} in'
    hist = {}
    for t in out:
        hist[t.tensor_type.name] = hist.get(t.tensor_type.name, 0) + 1
    print(f'ok: {len(out)} tensors in {a.out} ({", ".join(f"{k} {v}" for k, v in sorted(hist.items()))}), {os.path.getsize(a.out) / 2**30:.3f} GiB')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('copy', help="the model's own tensors, byte for byte")
    c.add_argument('model', help='the model (first shard of a split set)')
    c.add_argument('out')
    c.add_argument('--tensors', default=K31_TENSORS, help='regex on tensor names (default: K3.1 attention + ssm_g)')
    c.add_argument('--check', action='store_true', help='read the output back and compare every byte (reads it twice)')
    q = sub.add_parser('quant', help='llama-quantize --allow-requantize --pure, CPU only')
    q.add_argument('inp')
    q.add_argument('out')
    q.add_argument('ftype', help="llama-quantize's type name, e.g. Q3_K_M (every tensor Q3_K with --pure)")
    q.add_argument('--bin', required=True, help='llama-quantize of a build that knows the model arch')
    q.add_argument('--threads', type=int, default=0, help="llama-quantize's thread count (default: all)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists: not overwritten')
    {'copy': cmd_copy, 'quant': cmd_quant}[a.cmd](a)


if __name__ == '__main__':
    main()
