#!/usr/bin/env python3
"""nsys_breakdown.py report.sqlite [tokens_per_s]: where a decode token's time goes, from an nsys profile taken with
--cuda-graph-trace=node (kernels inside CUDA graphs listed one by one) and CPU sampling.

Prints per GPU: busy ms/token, kernels grouped by op, idle gaps; CUDA API time of the host threads; CPU samples by
function (top) per thread role. Tokens are estimated from the capture span and the measured t/s (default 20)."""
import collections, re, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
tps = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
S = dict(db.execute("select id, value from StringIds"))

kern = list(db.execute("select deviceId, start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"))
t0 = min(k[1] for k in kern); t1 = max(k[2] for k in kern)
span_ms = (t1 - t0) / 1e6
ntok = span_ms / 1000 * tps
print(f"capture {span_ms:.0f} ms, ~{ntok:.0f} tokens at {tps} t/s, {len(kern)} kernels ({len(kern)/ntok:.0f}/token)")

GROUPS = [
    ("moe cache chain (mmvq ids)", r"mul_mat_vec_q_moe|mmvq.*ids|mul_mat_vec_q.*\(bool\)1, \(bool\)1"),
    ("matvec quantized (attn/shexp/dense/out)", r"mul_mat_vec_q"),
    ("matvec f32/bf16/f16", r"mul_mat_vec_f"),
    ("matmul (mmq/cublas/mul_mat_f)", r"mul_mat_q|mul_mat_f|gemm|cutlass|ampere|sm80"),
    ("gated delta net", r"gated_delta"),
    ("flash attn / softmax", r"flash_attn|soft_max|fattn"),
    ("norms", r"norm"),
    ("hyper-connections", r"hc_|dsv4"),
    ("rope", r"rope"),
    ("quantize activations", r"quantize"),
    ("topk / argsort / router", r"topk|argsort|sigmoid|argmax"),
    ("copy/concat/cont/getrows/set", r"cpy|copy|concat|get_rows|set_rows|cont|dup"),
    ("elementwise (add/mul/scale/glu/unary)", r"add|mul|scale|glu|silu|swiglu|unary|k_bin|op_"),
]
def group(name):
    for g, rx in GROUPS:
        if re.search(rx, name):
            return g
    return "other"

def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

for dev in sorted({k[0] for k in kern}):
    kk = [k for k in kern if k[0] == dev]
    busy = union([(k[1], k[2]) for k in kk])
    bt = sum(b - a for a, b in busy) / 1e6
    g = collections.defaultdict(lambda: [0, 0.0])
    top = collections.defaultdict(lambda: [0, 0.0])
    for _, a, b, sn, dn in kk:
        name = S.get(dn, S.get(sn, "?"))
        x = g[group(name)]; x[0] += 1; x[1] += (b - a) / 1e6
        y = top[name[:110]]; y[0] += 1; y[1] += (b - a) / 1e6
    gaps = [busy[i + 1][0] - busy[i][1] for i in range(len(busy) - 1)]
    small = sum(x for x in gaps if x < 20e3) / 1e6
    mid = sum(x for x in gaps if 20e3 <= x < 500e3) / 1e6
    big = sum(x for x in gaps if x >= 500e3) / 1e6
    print(f"\nGPU{dev}: busy {bt/ntok:.2f} ms/token ({100*bt/span_ms:.0f}%), {len(kk)/ntok:.0f} kernels/token, "
          f"idle gaps/token: <20us {small/ntok:.2f} ms, 20-500us {mid/ntok:.2f} ms, >500us {big/ntok:.2f} ms")
    for name, (n, t) in sorted(g.items(), key=lambda kv: -kv[1][1]):
        print(f"   {name:42s} {t/ntok:6.2f} ms/token  {n/ntok:6.1f} kernels/token  avg {1e3*t/max(n,1):6.1f} us")
    print("   top kernels:")
    for name, (n, t) in sorted(top.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"     {t/ntok:5.2f} ms/token {n/ntok:5.1f}/token avg {1e3*t/max(n,1):5.1f} us  {name}")

rt = collections.defaultdict(lambda: [0, 0.0])
for nid, a, b in db.execute("select nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
    x = rt[S[nid].split("_v")[0]]; x[0] += 1; x[1] += (b - a) / 1e6
print("\nCUDA API time (all host threads), per token:")
for name, (n, t) in sorted(rt.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"   {name:28s} {n/ntok:6.1f} calls  {t/ntok:6.2f} ms")

if "COMPOSITE_EVENTS" in tables and "SAMPLING_CALLCHAINS" in tables:
    # leaf function of every CPU sample, per thread
    leaf = {}
    for cid, sym in db.execute("select id, symbol from SAMPLING_CALLCHAINS where stackDepth = 0"):
        leaf[cid] = S.get(sym, "?")
    per_thread = collections.defaultdict(collections.Counter)
    for cid, tid in db.execute("select id, globalTid from COMPOSITE_EVENTS"):
        per_thread[tid][leaf.get(cid, "?")] += 1
    tot = sum(sum(c.values()) for c in per_thread.values())
    print(f"\nCPU samples: {tot} over {len(per_thread)} threads; busiest threads and their top functions:")
    for tid, c in sorted(per_thread.items(), key=lambda kv: -sum(kv[1].values()))[:9]:
        n = sum(c.values())
        tops = ", ".join(f"{f[:38]} {100*k/n:.0f}%" for f, k in c.most_common(4))
        print(f"   tid {tid & 0xffffff}: {n} samples | {tops}")
