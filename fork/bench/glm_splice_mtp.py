#!/usr/bin/env python3
"""Add unsloth's GLM-5.3-Flash MTP (NextN) block to a local GLM-5.3-Flash GGUF.

Downloads only the blk.<n_layer>.* tensors via HTTP range requests (~4.3 GB), copies every
local tensor unchanged (e.g. the GSQ-RCO trunk) and sets block_count/nextn_predict_layers.
The MTP block only drafts tokens; the trunk verifies them, so its quant affects acceptance only.

usage: glm_splice_mtp.py [--mtp-only] <local.gguf> <out.gguf> [unsloth quant dir, default UD-Q4_K_XL]
--mtp-only: write only the NextN block + metadata (a -md draft for any GLM-5.3-Flash trunk)
"""
import os, sys, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'gguf-py'))  # repo's gguf-py
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gguf
import gguf_remote

args = [a for a in sys.argv[1:] if a != '--mtp-only']
if len(args) < 2:
    sys.exit(__doc__)
mtp_only = '--mtp-only' in sys.argv  # write just the NextN block (+ metadata), loaded with -md
src, dst = args[0], args[1]
quant = args[2] if len(args) > 2 else 'UD-Q4_K_XL'
repo = f'https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/{quant}/GLM-5.3-Flash-{quant}'

reader = gguf.GGUFReader(src)
arch = 'glm5-next'
n_layer = int(reader.fields[f'{arch}.block_count'].contents())
mtp = f'blk.{n_layer}.'

# find the MTP tensors in the remote shards (headers only)
n_split = None
for n in range(1, 20):  # shard count is in the file names; find it
    try:
        n_split = gguf_remote.header(f'{repo}-00001-of-{n:05d}.gguf')[0]['split.count']
        break
    except Exception:
        continue
kv_remote = gguf_remote.header(f'{repo}-00001-of-{n_split:05d}.gguf')[0]
new = []
for i in range(1, n_split + 1):
    url = f'{repo}-{i:05d}-of-{n_split:05d}.gguf'
    kv, tensors, ds = gguf_remote.header(url)
    new += [dict(t, url=url, start=ds + t['offset']) for t in tensors if t['name'].startswith(mtp)]
assert new, f'no {mtp}* tensors found in {repo}'
print(f'{len(new)} MTP tensors, {sum(t["size"] for t in new) / 2**30:.2f} GiB')

# download them (raw quantized bytes, cached next to the output)
cache = dst + '.mtp-parts'
os.makedirs(cache, exist_ok=True)
for t in new:
    f = os.path.join(cache, t['name'])
    if os.path.exists(f) and os.path.getsize(f) == t['size']:
        continue
    req = urllib.request.Request(t['url'], headers={'Range': f"bytes={t['start']}-{t['start'] + t['size'] - 1}"})
    with urllib.request.urlopen(req) as r, open(f, 'wb') as o:
        while chunk := r.read(1 << 24):
            o.write(chunk)
    assert os.path.getsize(f) == t['size'], t['name']
    print('got', t['name'], f"{t['size'] / 2**20:.1f} MiB", flush=True)

writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == f'{arch}.block_count':
        val = n_layer + 1
    elif vt == gguf.GGUFValueType.ARRAY and isinstance(val, list) and len(val) == n_layer:
        # per-layer array: append the MTP layer's value from the remote file
        rv = kv_remote.get(field.name.replace(f'{arch}.', 'glm5next.', 1))
        if rv is not None:  # keys the remote file lacks are trunk-only (e.g. indexer.types): keep
            assert len(rv) == n_layer + 1, f'no MTP value for {field.name}'
            val = val + [rv[n_layer]]
    writer.add_key_value(field.name, val, vt, sub_type=st)
writer.add_key_value(f'{arch}.nextn_predict_layers', 1, gguf.GGUFValueType.UINT32)
if f'{arch}.expert_shared_feed_forward_length' not in reader.fields:
    writer.add_key_value(f'{arch}.expert_shared_feed_forward_length', 2048, gguf.GGUFValueType.UINT32)

trunk = [] if mtp_only else reader.tensors
for t in trunk:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
arrays = []
for t in new:
    qt = gguf.GGMLQuantizationType(t['type'])
    bs, ts = gguf.GGML_QUANT_SIZES[qt]
    d = t['dims']
    a = np.memmap(os.path.join(cache, t['name']), dtype=np.uint8, mode='r')
    a = a.reshape(*reversed(d[1:]), d[0] // bs * ts) if len(d) > 1 else a.reshape(d[0] // bs * ts)
    writer.add_tensor_info(t['name'], a.shape, a.dtype, a.nbytes, qt)
    arrays.append(a)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in trunk:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
for a in arrays:
    writer.write_tensor_data(a)
writer.close()
print('wrote', dst)
