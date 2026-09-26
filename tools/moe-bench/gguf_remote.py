#!/usr/bin/env python3
"""Read a remote GGUF's header over HTTP range requests (no full download).

usage: gguf_remote.py <url> [name-filter]   -> prints kv keys and matching tensors with byte ranges
As a module: header(url) -> (kv dict, tensors list, data_start)
"""
import struct, sys, urllib.request

T = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}  # 8 str, 9 arr
# bytes per block and block size for the ggml types used here (others: extend as needed)
GGML = {0: (4, 1), 1: (2, 1), 8: (34, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256),
        14: (210, 256), 16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (36, 32), 21: (110, 256),
        22: (82, 256), 23: (136, 256), 29: (56, 256), 30: (2, 1), 39: (17, 32)}


class Remote:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b'', 0

    def need(self, n):
        while len(self.buf) < self.pos + n:  # fetch in growing chunks
            want = max(1 << 22, len(self.buf))
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={len(self.buf)}-{len(self.buf) + want - 1}'})
            self.buf += urllib.request.urlopen(req).read()

    def read(self, n):
        self.need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def s(self):
        return self.read(self.u('Q')).decode('utf-8', 'replace')

    def val(self, t):
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.u('I'), self.u('Q')
            return [self.val(et) for _ in range(n)]
        return self.u(T[t])


def header(url):
    r = Remote(url)
    assert r.read(4) == b'GGUF'
    r.u('I')
    nt, nkv = r.u('Q'), r.u('Q')
    kv = {}
    for _ in range(nkv):
        k = r.s()
        kv[k] = r.val(r.u('I'))
    tensors = []
    for _ in range(nt):
        name = r.s()
        dims = [r.u('Q') for _ in range(r.u('I'))]
        typ, off = r.u('I'), r.u('Q')
        n = 1
        for d in dims:
            n *= d
        bs, blk = GGML[typ]
        tensors.append({'name': name, 'dims': dims, 'type': typ, 'offset': off, 'size': n // blk * bs})
    align = kv.get('general.alignment', 32)
    data_start = (r.pos + align - 1) // align * align
    return kv, tensors, data_start


if __name__ == '__main__':
    kv, tensors, ds = header(sys.argv[1])
    flt = sys.argv[2] if len(sys.argv) > 2 else None
    for k, v in kv.items():
        if not isinstance(v, list):
            print('kv', k, v)
    for t in tensors:
        if not flt or flt in t['name']:
            print(t['name'], t['dims'], 'type', t['type'], f"bytes {ds + t['offset']}-{ds + t['offset'] + t['size'] - 1}", f"{t['size'] / 2**20:.1f} MiB")
