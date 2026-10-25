#!/usr/bin/env python3
"""Make ISTA-DASLab's DeepSeek-V4.1-Flash GGUF set (4 shards + Engram file) load as 5 shards of the deepseek41 arch.

Five files in, five files out, in place, nothing is copied.  Only header bytes (and 2 small tensors written into
dead space of shard 1) change; every change is logged and can be undone:

  ista_to_ds41.py apply  DIR   # writes DIR/ista2ds41.log (human readable) + DIR/ista2ds41.undo/ (original bytes)
  ista_to_ds41.py revert DIR   # restores every byte range and file name from the undo records
  ista_to_ds41.py dry    DIR   # prints the plan and the log, writes nothing

What changes (ref: PR #30817 key/tensor names, Strata tools/dsv41/pack_gguf.py for the ISTA side):
 - shard 1: kv_source_layer_ids, index_source_layer_ids, attention.candidate_* -> attention.kv_source_layers,
   attention.indexer.source_layers, attention.indexer.candidate_*; the integer tensors engram_token_map/primes/offsets/
   multipliers become metadata ple.* / engram.token_map; engram gate (q*k, F32) written into the freed tensor bytes.
 - shard 4: blk.N.engram_wkv -> blk.N.engram_kv; engram_q / engram_k dropped from the header (bytes stay, unused).
 - Engram file: one tensor per_layer_token_embd.weight [256, sum rows] (its two layer tables are contiguous), arch
   deepseek41, renamed ...-00005-of-00005.gguf.  All five get split.count=5 and the new split.tensors.count.
A header is rewritten to the same data start: a filler string key pads it to the exact old size.
"""
import json, os, struct, sys, glob, re

ALIGN = 32
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SZ = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4, T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}
FMT = {T_U8: "B", T_I8: "b", T_U16: "H", T_I16: "h", T_U32: "I", T_I32: "i", T_F32: "f", T_BOOL: "?", T_U64: "Q", T_I64: "q", T_F64: "d"}
P = "deepseek41."


class Reader:
    def __init__(self, b): self.b, self.o = b, 0
    def u(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)[0]; self.o += struct.calcsize(fmt); return v
    def s(self):
        n = self.u("Q"); v = self.b[self.o:self.o + n].decode(); self.o += n; return v
    def skip_val(self, t):
        if t == T_STR: self.s()
        elif t == T_ARR:
            et = self.u("I"); n = self.u("Q")
            if et in SZ: self.o += n * SZ[et]
            else:
                for _ in range(n): self.skip_val(et)
        else: self.o += SZ[t]


def pstr(s): b = s.encode(); return struct.pack("<Q", len(b)) + b


def parse(path):
    """-> dict(kvs=[(key, type, raw value bytes)], tensors=[[name, dims, type, off]], data_start, hdr_end)"""
    with open(path, "rb") as f: head = f.read(64 << 20)
    r = Reader(head)
    assert head[:4] == b"GGUF"; r.o = 4; assert r.u("I") == 3
    nt, nkv = r.u("Q"), r.u("Q")
    kvs = []
    for _ in range(nkv):
        k = r.s(); t = r.u("I"); st = r.o; r.skip_val(t); kvs.append((k, t, head[st:r.o]))
    tens = []
    for _ in range(nt):
        n = r.s(); nd = r.u("I"); dims = [r.u("Q") for _ in range(nd)]; ty = r.u("I"); off = r.u("Q")
        tens.append([n, dims, ty, off])
    end = r.o
    ds = (end + ALIGN - 1) // ALIGN * ALIGN
    return dict(kvs=kvs, tensors=tens, data_start=ds, hdr_end=end)


def val(t, raw):
    if t == T_STR: return raw[8:].decode()
    if t == T_ARR:
        et = struct.unpack_from("<I", raw)[0]; n = struct.unpack_from("<Q", raw, 4)[0]
        if et not in FMT: return None   # string array (tokenizer): not needed
        return list(struct.unpack_from("<%d%s" % (n, FMT[et]), raw, 12))
    return struct.unpack("<" + FMT[t], raw)[0]


def mk(t, v):
    if t == T_STR: return pstr(v)
    if t == T_ARR: raise ValueError
    return struct.pack("<" + FMT[t], v)


def mkarr(et, vs): return struct.pack("<IQ", et, len(vs)) + struct.pack("<%d%s" % (len(vs), FMT[et]), *vs)


def build(kvs, tens, data_start):
    """header bytes padded with a filler key to exactly data_start"""
    def body(extra):
        b = b"GGUF" + struct.pack("<IQQ", 3, len(tens), len(kvs) + (1 if extra is not None else 0))
        for k, t, raw in kvs: b += pstr(k) + struct.pack("<I", t) + raw
        if extra is not None: b += pstr("general.padding") + struct.pack("<I", T_STR) + pstr(" " * extra)
        for n, dims, ty, off in tens:
            b += pstr(n) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        return b
    b = body(None)
    if data_start is None: return b + b"\0" * (-len(b) % ALIGN)
    if (len(b) + ALIGN - 1) // ALIGN * ALIGN == data_start: return b + b"\0" * (data_start - len(b))
    over = len(body(0))
    if over > data_start: raise SystemExit("header needs %d bytes, only %d available" % (over, data_start))
    b = body(data_start - over)
    assert len(b) == data_start
    return b


def setkv(kvs, key, t, v, arr=None):
    raw = mkarr(arr, v) if arr is not None else mk(t, v)
    for i, (k, _, _) in enumerate(kvs):
        if k == key: kvs[i] = (key, t, raw); return
    kvs.append((key, t, raw))


def plan(d):
    sh = sorted(glob.glob(os.path.join(d, "*-0000[1-4]-of-00004.gguf")))
    eg = glob.glob(os.path.join(d, "*Engram*.gguf"))
    assert len(sh) == 4 and len(eg) == 1, (sh, eg)
    S = [parse(p) for p in sh]; E = parse(eg[0])
    log, patches, renames = [], [], []   # patch = (file, offset, new bytes)

    kv1 = {k: val(t, raw) for k, t, raw in S[0]["kvs"]}
    ren = {"kv_source_layer_ids": ("attention.kv_source_layers", "u32arr"),
           "index_source_layer_ids": ("attention.indexer.source_layers", "u32arr"),
           "attention.candidate_source_layer_id": ("attention.indexer.candidate_source_layer", "i32"),
           "attention.candidate_topk_blocks": ("attention.indexer.candidate_top_k_blocks", "u32"),
           "attention.candidate_block_size": ("attention.indexer.candidate_block_size", "u32")}
    drop_old = {P + k for k in ren}

    # integer tensors of shard 1
    tdat = {}
    with open(sh[0], "rb") as f:
        for n, dims, ty, off in S[0]["tensors"]:
            if n in ("engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight"):
                cnt = 1
                for x in dims: cnt *= x
                fmt, sz = ("i", 4) if ty == 26 else ("q", 8)
                f.seek(S[0]["data_start"] + off); tdat[n] = list(struct.unpack("<%d%s" % (cnt, fmt), f.read(cnt * sz)))
    tok, primes, offs, mult = (tdat[k] for k in ("engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight"))
    layers = kv1[P + "engram.layer_ids"]; nrows = kv1[P + "engram.num_embeddings"]
    nh = len(primes) // len(layers)
    log.append("check: engram layers %s rows %s, %d heads/layer, offsets[0:3]=%s primes[0:3]=%s, layer-2 first offset %d (rows of layer 1 = %d)"
               % (layers, nrows, nh, offs[:3], primes[:3], offs[nh], nrows[0]))
    for c in range(len(primes) - 1):
        if (c + 1) % nh and offs[c + 1] != offs[c] + primes[c]: log.append("WARN: offsets not cumulative at head %d" % c)

    # ---- shard 1 kvs
    kvs = [(k, t, raw) for k, t, raw in S[0]["kvs"] if k not in drop_old]
    for old, (new, kind) in ren.items():
        if P + old not in kv1: continue
        v = kv1[P + old]
        if kind == "u32arr": setkv(kvs, P + new, T_ARR, v, T_U32)
        elif kind == "i32": setkv(kvs, P + new, T_I32, v)
        else: setkv(kvs, P + new, T_U32, v)
        log.append("shard1 kv  %s%s -> %s%s  = %s" % (P, old, P, new, v if not isinstance(v, list) else v))
    # ISTA offsets restart at 0 in every layer's table; the concatenated table needs absolute rows
    base = 0; offs = list(offs)
    for e in range(len(layers)):
        for c in range(e * nh, (e + 1) * nh): offs[c] += base
        base += nrows[e]
    log.append("ple.head_offsets: layer tables concatenated, layer-2 offsets shifted by %d rows (last head ends at row %d, tensor has %d)"
               % (nrows[0], offs[-1] + primes[-1], sum(nrows)))
    assert offs[-1] + primes[-1] <= sum(nrows)
    setkv(kvs, P + "ple.layers", T_ARR, layers, T_U32)
    setkv(kvs, P + "ple.ngram_size", T_U32, kv1[P + "engram_max_ngram_size"])
    setkv(kvs, P + "ple.heads_per_ngram", T_U32, kv1[P + "engram_n_heads"])
    setkv(kvs, P + "embedding_length_per_layer_input", T_U32, kv1[P + "engram_head_dim"])
    setkv(kvs, P + "ple.layer_multipliers", T_ARR, mult, T_U64)
    setkv(kvs, P + "ple.head_offsets", T_ARR, offs, T_U64)
    setkv(kvs, P + "ple.head_vocab_sizes", T_ARR, primes, T_U64)
    setkv(kvs, P + "engram.pad_token_id", T_U32, kv1[P + "engram_pad_token_id"])
    setkv(kvs, P + "engram.token_map", T_ARR, tok, T_I32)
    log.append("shard1 kv  + ple.layers %s ngram %d heads/ngram %d head_dim %d; ple.layer_multipliers/head_offsets/head_vocab_sizes "
               "from tensors (%d/%d/%d values); engram.token_map %d ids (from engram_token_map.weight)"
               % (layers, kv1[P + "engram_max_ngram_size"], kv1[P + "engram_n_heads"], kv1[P + "engram_head_dim"], len(mult), len(offs), len(primes), len(tok)))
    log.append("shard1 tensors dropped (their bytes are not copied; every other offset moves down by their size): engram_token_map/primes/offsets/multipliers.weight")

    # ---- gate = q*k (bf16 -> f32) from shard 4
    s4 = S[3]
    gate_tens, gate_bytes = [], []
    dead = S[0]["tensors"][4][3]      # the 4 integer tensors come first: their bytes are dropped (the reader wants tensor data packed from 0)
    gend = (os.path.getsize(sh[0]) - S[0]["data_start"] - dead + ALIGN - 1) // ALIGN * ALIGN
    with open(sh[3], "rb") as f:
        by = {t[0]: t for t in s4["tensors"]}
        off = 0
        for il in layers:
            qk = []
            for w in ("q", "k"):
                n, dims, ty, o = by["blk.%d.engram_%s.weight" % (il, w)]; assert ty == 30
                cnt = dims[0] * dims[1]
                f.seek(s4["data_start"] + o); raw = f.read(cnt * 2)
                qk.append(struct.unpack("<%df" % cnt, b"".join(b"\0\0" + raw[i:i + 2] for i in range(0, len(raw), 2))))
            g = struct.pack("<%df" % cnt, *[a * b for a, b in zip(*qk)])
            goff = gend + off
            gate_tens.append(["blk.%d.engram_gate.weight" % il, list(dims), 0, goff]); gate_bytes.append((goff, g))
            log.append("shard1 tensor + blk.%d.engram_gate.weight F32 %s appended at data offset %d (%d bytes) = engram_q * engram_k"
                       % (il, dims, goff, len(g)))
            off += (len(g) + ALIGN - 1) // ALIGN * ALIGN
    keep = [[n, d_, ty, o - dead] for n, d_, ty, o in S[0]["tensors"] if not n.startswith("engram_")] + gate_tens
    assert keep[0][3] == 0

    # ---- tensor count: 1008 - 4 - 2*layers*? (q,k) + gate + 1 per_layer
    n_total = len(keep) + (len(S[1]["tensors"]) + len(S[2]["tensors"])) + (len(s4["tensors"]) - 2 * len(layers)) + 1
    log.append("split.tensors.count %d -> %d, split.count 4 -> 5" % (kv1["split.tensors.count"], n_total))

    def fix_split(kvs, no):
        setkv(kvs, "split.no", T_U16, no); setkv(kvs, "split.count", T_U16, 5); setkv(kvs, "split.tensors.count", T_I32, n_total)
        return kvs

    hdr = {}
    hdr1 = build(fix_split(kvs, 0), keep, None)   # grows (token_map in metadata): shard 1 is rewritten as a new file
    for i in (1, 2):
        k2 = list(S[i]["kvs"]); fix_split(k2, i)
        hdr[sh[i]] = build(k2, S[i]["tensors"], S[i]["data_start"])
    t4 = []
    for n, dims, ty, o in s4["tensors"]:
        if re.match(r"blk\.\d+\.engram_[qk]\.weight$", n): log.append("shard4 tensor - %s dropped from header (bytes stay)" % n); continue
        if re.match(r"blk\.\d+\.engram_wkv\.weight$", n):
            log.append("shard4 tensor %s -> %s" % (n, n.replace("engram_wkv", "engram_kv"))); n = n.replace("engram_wkv", "engram_kv")
        t4.append([n, dims, ty, o])
    k4 = list(s4["kvs"]); fix_split(k4, 3)
    # the reader wants tensor data packed from 0 in header order: dropping q/k leaves holes, so shard 4 is repacked too
    end4 = os.path.getsize(sh[3]) - s4["data_start"]
    order = sorted(s4["tensors"], key=lambda t: t[3]); size4 = {}
    for i, t in enumerate(order): size4[t[0]] = (order[i + 1][3] if i + 1 < len(order) else end4) - t[3]
    old_off = {t[0]: t[3] for t in s4["tensors"]}
    ranges4, pos = [], 0
    for t in t4:
        src_name = t[0].replace("engram_kv", "engram_wkv")
        ranges4.append((old_off[src_name], pos, size4[src_name])); t[3] = pos; pos += size4[src_name]
    hdr4 = build(k4, t4, None)

    # ---- engram file
    tot = sum(t[1][1] for t in E["tensors"]); contig = 0
    for n, dims, ty, o in E["tensors"]:
        assert o == contig and ty == 20, "engram tables not contiguous IQ4_NL"
        contig += dims[0] // 32 * 18 * dims[1]
    ke = [("general.architecture", T_STR, pstr("deepseek41"))]
    ke += [(k, t, raw) for k, t, raw in E["kvs"] if k != "general.architecture"]
    fix_split(ke, 4)
    pt = [["per_layer_token_embd.weight", [E["tensors"][0][1][0], tot], 20, 0]]
    hdr[eg[0]] = build(ke, pt, E["data_start"])
    log.append("engram file: arch deepseek41-engram -> deepseek41, tensors %s -> per_layer_token_embd.weight [%d, %d] IQ4_NL (header only, data untouched), split.no 4"
               % ([t[0] for t in E["tensors"]], pt[0][1][0], tot))
    new_eg = os.path.join(d, re.sub(r"-00004\.gguf$", "-00005-of-00005.gguf", os.path.basename(sh[3])).replace("00004-of", "00005-of")
                          if False else os.path.basename(sh[0]).replace("00001-of-00004", "00005-of-00005"))
    renames.append((eg[0], new_eg))
    for p in sh:   # file names say of-00004: the split loader derives the others from the name, so rename all
        renames.append((p, p.replace("-of-00004", "-of-00005")))

    for p, h in hdr.items(): patches.append((p, 0, h))
    new1 = sh[0].replace("-of-00004", "-of-00005")
    log.append("shard1: new file %s = new header (%d bytes, was %d) + copy of old data from byte %d; original kept as %s.orig-ista"
               % (os.path.basename(new1), len(hdr1), S[0]["data_start"], S[0]["data_start"], os.path.basename(sh[0])))
    shard1 = dict(src=sh[0], dst=new1, hdr=hdr1, old_ds=S[0]["data_start"] + dead, gates=gate_bytes)
    new4 = sh[3].replace("-of-00004", "-of-00005")
    log.append("shard4: new file %s = new header + %d tensor ranges repacked (q/k dropped); original kept as .orig-ista" % (os.path.basename(new4), len(ranges4)))
    shard4 = dict(src=sh[3], dst=new4, hdr=hdr4, old_ds=s4["data_start"], ranges=ranges4)
    patches = [x for x in patches if x[0] != sh[3]]
    return log, patches, [r for r in renames if r[0] not in (sh[0], sh[3])], shard1, shard4


def main():
    cmd, d = sys.argv[1], sys.argv[2]
    ud, lg = os.path.join(d, "ista2ds41.undo"), os.path.join(d, "ista2ds41.log")
    if cmd == "revert":
        rec = json.load(open(os.path.join(ud, "undo.json")))
        for nf in rec.get("new_files", []):
            if os.path.exists(nf): os.remove(nf); print("removed", os.path.basename(nf))
        for a, b in reversed(rec["renames"]):      # names first: the patches below use the original names
            if os.path.exists(b) and not os.path.exists(a): os.rename(b, a); print("renamed back", os.path.basename(b), "->", os.path.basename(a))
        for i, (p, off, n) in enumerate(rec["patches"]):
            with open(os.path.join(ud, "%d.bin" % i), "rb") as bf: old = bf.read()
            with open(p, "r+b") as f: f.seek(off); f.write(old)
            print("restored %s @%d %d bytes" % (os.path.basename(p), off, n))
        return
    log, patches, renames, s1, s4 = plan(d)
    if cmd == "dry":
        print("\n".join(log)); [print("patch", os.path.basename(p), off, len(b)) for p, off, b in patches]; print("renames", renames); return
    assert cmd == "apply" and not os.path.exists(ud), "undo records exist: revert first"
    os.makedirs(ud); rec = {"patches": [], "renames": list(renames)}
    for i, (p, off, new) in enumerate(patches):
        with open(p, "rb") as f: f.seek(off); old = f.read(len(new))
        open(os.path.join(ud, "%d.bin" % i), "wb").write(old)
        rec["patches"].append((p, off, len(new)))
        log.append("patch %d: %s offset %d, %d bytes (old bytes saved in undo/%d.bin)" % (i, os.path.basename(p), off, len(new), i))
    json.dump(rec, open(os.path.join(ud, "undo.json"), "w"))
    src, dst = s1["src"], s1["dst"]
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(s1["hdr"]); fi.seek(s1["old_ds"])
        while True:
            buf = fi.read(64 << 20)
            if not buf: break
            fo.write(buf)
        for off, g in s1["gates"]: fo.seek(len(s1["hdr"]) + off); fo.write(g)
    os.rename(src, src + ".orig-ista"); rec["renames"].append((src, src + ".orig-ista")); rec["new_files"] = [dst]
    src4, dst4 = s4["src"], s4["dst"]
    with open(src4, "rb") as fi, open(dst4, "wb") as fo:
        fo.write(s4["hdr"])
        for so, do, n in s4["ranges"]:
            fi.seek(s4["old_ds"] + so); fo.seek(len(s4["hdr"]) + do)
            while n:
                buf = fi.read(min(n, 64 << 20)); fo.write(buf); n -= len(buf)
    os.rename(src4, src4 + ".orig-ista"); rec["renames"].append((src4, src4 + ".orig-ista")); rec["new_files"].append(dst4)
    json.dump(rec, open(os.path.join(ud, "undo.json"), "w"))
    log.append("shard1 written: %s; original renamed to %s" % (os.path.basename(dst), os.path.basename(src) + ".orig-ista"))
    for p, off, new in patches:
        with open(p, "r+b") as f: f.seek(off); f.write(new)
    for a, b in renames: os.rename(a, b); log.append("rename %s -> %s" % (os.path.basename(a), os.path.basename(b)))
    open(lg, "w").write("\n".join(log) + "\n"); print("\n".join(log))


main()
