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
  attn_side.py hf MODEL.gguf OUT.gguf [--repo ORG/NAME] [--revision SHA] [--tensors REGEX] [--dry-run] [--jobs N] [--cache DIR]
      the same tensors at full resolution from the model's source repo on Hugging Face (from the GGUF's metadata, else the
      README.md beside it): only their byte ranges are fetched, straight into their place in OUT (no second copy on the disk;
      a stopped run goes on where it stopped, OUT.done). Names as fork/conversion/kimi_k3.py writes them; a tensor the
      converter changes (kv_b split, A_log, conv1d, ...) or another shape than the model's is refused before any download.
      --dry-run: sizes, shards and the disk only (reads the index and the shards' headers). Then `quant` per type.

Every byte range read and every file written is dropped from the page cache afterwards: the rows after a run start cold.
A download is paused by SIGSTOP of this process and goes on with SIGCONT (no child process).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

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


HF = 'https://huggingface.co'
# conversion/kimi_k3.py changes these values or splits them (-exp(A_log), dt_bias renamed, conv1d reshaped, kv_b into k_b and v_b,
# res norm x proj fused, experts stacked): never a byte copy
NOT_PLAIN = ('.A_log', '.dt_bias', 'conv1d.weight', 'kv_b_proj.weight')
HF_TYPES = {'BF16': gguf.GGMLQuantizationType.BF16, 'F16': gguf.GGMLQuantizationType.F16, 'F32': gguf.GGMLQuantizationType.F32}


def http_range(url, start, n):
    """n bytes of url from start; a server that ignores the range is refused before its body is read (a whole shard would follow)"""
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + n - 1}'})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 206:
            raise OSError(f'{url}: HTTP {r.status} to a range request')
        b = r.read(n)
    if len(b) != n:
        raise OSError(f'{url}: {len(b)} of {n} bytes')
    return b


def cached(cache, name, fetch):
    p = os.path.join(cache, name)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return f.read()
    b = fetch()
    os.makedirs(cache, exist_ok=True)
    with open(p + '.tmp', 'wb') as f:
        f.write(b)
    os.replace(p + '.tmp', p)
    return b


def hf_source(model, first, a):
    """(repo, revision, where they came from): --repo/--revision, the GGUF's metadata, else the README.md beside the model"""
    repo, rev, how = a.repo, a.revision, '--repo'
    if not repo:
        for k in ('general.base_model.0.repo_url', 'general.source.repo_url', 'general.source.huggingface.repository'):
            if k in first.fields:
                repo, how = re.sub(r'^https?://huggingface\.co/', '', first.fields[k].contents()).strip('/'), f'GGUF {k}'
                break
    readme = os.path.join(os.path.dirname(os.path.abspath(model)), 'README.md')
    txt = open(readme, errors='replace').read() if os.path.exists(readme) else ''
    if not repo:
        m = re.search(r'^base_model:\s*(?:\n\s*-\s*)?([\w.-]+/[\w.-]+)\s*$', txt, re.M)
        if m:
            repo, how = m.group(1), f'{readme} (base_model)'
    if not repo:
        sys.exit(f'no source repo in the GGUF metadata or {readme}: give --repo ORG/NAME')
    if not rev:
        m = re.search(r'revision\W{0,3}([0-9a-f]{40})', txt)
        rev = m.group(1) if m else None
    if not rev:
        print('warning: no revision recorded, taking main: it may hold other weights than the model was made from')
        rev = 'main'
    return repo, rev, how


def hf_plan(a):
    """every picked model tensor with its one HF source (url, byte range, type, shape); refuses before any tensor byte is fetched"""
    shards = shard_paths(a.model)
    first = shards[0][1]
    arch = first.fields['general.architecture'].contents()
    model = {t.name: t for _, r in shards for t in r.tensors}
    rx = re.compile(a.tensors)
    want = [n for n in model if rx.search(n)]
    if not want:
        sys.exit(f'no tensor of {a.model} matches {a.tensors}')
    repo, rev, how = hf_source(a.model, first, a)
    base = f'{HF}/{repo}/resolve/{rev}/'
    print(f'source: {repo} at {rev} ({how})', flush=True)
    index = json.loads(cached(a.cache, 'index.json', lambda: urllib.request.urlopen(base + 'model.safetensors.index.json', timeout=600).read()))['weight_map']
    tmap = gguf.get_tensor_name_map(next(k for k, v in gguf.MODEL_ARCH_NAMES.items() if v == arch), int(first.fields[f'{arch}.block_count'].contents()))
    picked, src = set(want), {}
    for hf_name, shard in index.items():
        n = hf_name.removeprefix('language_model.')   # K3: the text model under a prefix (conversion/kimi_k3.py get_tensors)
        if n.endswith(NOT_PLAIN) or '.experts.' in n or '_res_' in n:
            continue
        m = re.match(r'model\.layers\.(\d+)\.self_attn\.g_proj\.weight$', n)
        if m:   # one HF name for the KDA gate and the MLA output gate: whichever the model holds in that block (kimi_k3.py decides by layer type)
            g = next((x for x in (f'blk.{m[1]}.ssm_g.weight', f'blk.{m[1]}.attn_gate.weight') if x in model), None)
        else:
            g = tmap.get_name(n, try_suffixes=('.weight', '.bias'))
        if g in picked:
            if g in src:
                sys.exit(f'{g}: two HF sources, {src[g][0]} and {hf_name}')
            src[g] = (hf_name, shard)
    missing = [n for n in want if n not in src]
    if missing:
        sys.exit(f'{len(missing)} picked tensors have no HF tensor to copy byte for byte (the converter transforms them, or {repo} lacks them), e.g. {missing[:4]}')

    def header(shard):   # safetensors: 8-byte length, then the JSON with each tensor's dtype, shape and offsets after it
        hb = cached(a.cache, shard + '.header.json', lambda: http_range(base + shard, 8, int.from_bytes(http_range(base + shard, 0, 8), 'little')))
        return shard, (json.loads(hb), 8 + len(hb))
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        headers = dict(ex.map(header, sorted({s for _, s in src.values()})))
    plan = []
    for g in want:
        hf_name, shard = src[g]
        h, data_start = headers[shard]
        e = h[hf_name]
        if e['dtype'] not in HF_TYPES:
            sys.exit(f'{hf_name}: {e["dtype"]}; only BF16, F16 and F32 are copied (a packed or FP8 tensor needs its scales)')
        ne = [int(x) for x in model[g].shape]
        if list(reversed(e['shape'])) != ne:
            sys.exit(f'{g}: {hf_name} has the shape {e["shape"]}, the model {list(reversed(ne))} (numpy order)')
        o0, o1 = e['data_offsets']
        plan.append({'name': g, 'url': base + shard, 'start': data_start + o0, 'size': o1 - o0, 'type': HF_TYPES[e['dtype']], 'shape': e['shape']})
    return first, arch, model, plan, len(set(index.values())), len(headers)


def cmd_hf(a):
    first, arch, model, plan, n_shards, n_used = hf_plan(a)
    total = sum(p['size'] for p in plan)
    weights = sum(int(np.prod(p['shape'])) for p in plan)
    now = sum(model[p['name']].n_bytes for p in plan)
    types = sorted({model[p['name']].tensor_type.name for p in plan})
    d = os.path.dirname(os.path.abspath(a.out))
    while not os.path.isdir(d):   # a dry run before the directory is made: the drive it will be on
        d = os.path.dirname(d)
    free = shutil.disk_usage(d).free
    groups = {}
    for p in plan:
        k = re.sub(r'^blk\.\d+\.', '', p['name'])
        groups[k] = groups.get(k, 0) + p['size']
    print(f'{len(plan)} tensors from {n_used} of {n_shards} shards: {total / 2**30:.2f} GiB ({total / 1e9:.1f} GB) to fetch, '
          f'{", ".join(sorted({p["type"].name for p in plan}))}; the model holds them as {", ".join(types)} in {now / 2**30:.2f} GiB')
    print('by tensor (GiB as fetched): ' + ', '.join(f'{k} {v / 2**30:.2f}' for k, v in sorted(groups.items(), key=lambda kv: -kv[1])))
    print('as one type (GiB): ' + ', '.join(f'{t} {weights * b / 2**30:.2f}' for t, b in (('Q8_0', 34 / 32), ('Q4_K', 144 / 256), ('Q3_K', 110 / 256))))
    print(f'{a.out}: about {total / 2**30:.2f} GiB, {free / 1e9:.1f} GB free there{"" if free > total + (1 << 30) else ": DOES NOT FIT"}')
    if a.dry_run:
        print('dry run: no tensor fetched')
        return
    done_path = a.out + '.done'
    if not os.path.exists(a.out):
        if free < total + (1 << 30):
            sys.exit(f'{a.out}: {total / 1e9:.1f} GB do not fit in {free / 1e9:.1f} GB free')
        w = gguf.GGUFWriter(a.out, arch=arch, endianess=first.endianess)
        copy_kv(first, w)
        for p in plan:
            w.add_tensor_info(p['name'], p['shape'], np.float32 if p['type'] == gguf.GGMLQuantizationType.F32 else np.uint16, p['size'], p['type'])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        w.close()
        pad = lambda n: (n + w.data_alignment - 1) // w.data_alignment * w.data_alignment   # noqa: E731  the writer's own padding rule
        os.truncate(a.out, pad(os.path.getsize(a.out)) + sum(pad(p['size']) for p in plan))   # the data region, filled in place below
        open(done_path, 'w').close()
    offs = {t.name: t for t in gguf.GGUFReader(a.out).tensors}
    for p in plan:   # the header on disk (new or of a stopped run) is the one this plan writes into
        t = offs.get(p['name'])
        assert t is not None and t.tensor_type == p['type'] and t.n_bytes == p['size'] and list(t.shape) == list(reversed(p['shape'])), p['name']
    done = set(open(done_path).read().split())
    todo = [p for p in plan if p['name'] not in done]
    got, t0, lock = [total - sum(p['size'] for p in todo)], time.time(), threading.Lock()
    fd = os.open(a.out, os.O_WRONLY)

    def fetch(p):
        pos, left = offs[p['name']].data_offset, p['size']
        req = urllib.request.Request(p['url'], headers={'Range': f'bytes={p["start"]}-{p["start"] + p["size"] - 1}'})
        with urllib.request.urlopen(req, timeout=600) as r:
            if r.status != 206:
                raise OSError(f'{p["url"]}: HTTP {r.status} to a range request')
            while left:
                b = r.read(min(left, 16 << 20))
                if not b:
                    raise OSError(f'{p["name"]}: the connection ended {left} bytes early')
                os.pwrite(fd, b, pos)
                pos += len(b)
                left -= len(b)
        os.fdatasync(fd)   # on the drive before it is marked done
        os.posix_fadvise(fd, offs[p['name']].data_offset, p['size'], os.POSIX_FADV_DONTNEED)
        with lock:
            with open(done_path, 'a') as f:
                f.write(p['name'] + '\n')
            got[0] += p['size']
            print(f'{p["name"]}: {got[0] / 2**30:.2f} of {total / 2**30:.2f} GiB, {got[0] / 1e6 / max(1e-3, time.time() - t0):.0f} MB/s', flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(fetch, todo))
    finally:
        os.close(fd)
    drop(a.out)
    os.remove(done_path)
    print(f'ok: {len(plan)} tensors in {a.out}, {os.path.getsize(a.out) / 2**30:.2f} GiB. Next, per type: '
          f'attn_side.py quant {a.out} OUT-q3k.gguf Q3_K_M --bin .../llama-quantize')


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
    h = sub.add_parser('hf', help='the same tensors at full resolution from the source repo on Hugging Face')
    h.add_argument('model', help='the model (first shard of a split set)')
    h.add_argument('out')
    h.add_argument('--tensors', default=K31_TENSORS, help='regex on tensor names (default: K3.1 attention + ssm_g)')
    h.add_argument('--repo', help='ORG/NAME (default: from the GGUF metadata, else the README.md beside the model)')
    h.add_argument('--revision', help='commit (default: the one the README names, else main)')
    h.add_argument('--dry-run', action='store_true', help='sizes, shards and the disk only: fetches the index and the shard headers, no tensor')
    h.add_argument('--jobs', type=int, default=4, help='tensors fetched at once')
    h.add_argument('--cache', help='where the index and the shard headers are kept (default: OUT.hf)')
    a = ap.parse_args()
    if a.cmd == 'hf':
        a.cache = a.cache or a.out + '.hf'
    resume = a.cmd == 'hf' and os.path.exists(a.out + '.done')   # a stopped download goes on
    if os.path.exists(a.out) and not resume and not getattr(a, 'dry_run', False):
        sys.exit(f'{a.out} exists: not overwritten')
    {'copy': cmd_copy, 'quant': cmd_quant, 'hf': cmd_hf}[a.cmd](a)


if __name__ == '__main__':
    main()
