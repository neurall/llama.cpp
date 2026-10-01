import struct, sys, numpy as np
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize
recs = {}
with open(sys.argv[1], 'rb') as f:
    while True:
        h = f.read(4)
        if not h: break
        n, = struct.unpack('I', h); name = f.read(n).decode()
        ne0, ne1 = struct.unpack('qq', f.read(16))
        recs.setdefault(name, np.frombuffer(f.read(ne0*ne1*4), np.float32).reshape(ne1, ne0))
print(sorted(recs)[:6])
r = GGUFReader(sys.argv[2])
W = {}; B = {}
for t in r.tensors:
    if t.name.endswith('ffn_gate_inp.weight') or t.name.endswith('exp_probs_b.bias'):
        il = int(t.name.split('.')[1])
        a = dequantize(t.data, t.tensor_type).reshape(tuple(reversed([int(x) for x in t.shape])))
        (W if 'gate_inp' in t.name else B)[il] = a.astype(np.float32)
K = 8
def topk(x): return np.argsort(-x, axis=1)[:, :K]
layers = sorted(il for il in W)
for L in layers:
    nx = L + 1
    xin = recs.get(f'ffn_norm-{L}'); xin1 = recs.get(f'ffn_norm-{nx}')
    lg = recs.get(f'ffn_moe_logits_biased-{nx}')
    if xin is None or xin1 is None or nx not in W: continue
    Wn = W[nx]  # [n_expert, n_embd]
    bias = B.get(nx, 0)
    def sel(x): return topk(1/(1+np.exp(-(x @ Wn.T))) + bias)
    real = sel(xin1); pred = sel(xin)
    hit = np.mean([len(set(a) & set(b)) / K for a, b in zip(real, pred)])
    # same-layer reuse baseline: layer L's own choice predicting layer L+1 (meaningless ids, skip); temporal baseline: previous token's L+1 choice
    tmp = np.mean([len(set(a) & set(b)) / K for a, b in zip(real[1:], real[:-1])])
    print(f'L{L}->L{nx}: early-router top{K} overlap {hit:.3f}  prev-token overlap {tmp:.3f}')
#print('keys', [k for k in recs if k.endswith('-20')])
ok=[]; rc={12:[],16:[],24:[]}
for L in layers:
    nx=L+1
    if f'ffn_norm-{L}' not in recs or nx not in W: continue
    Wn=W[nx]; bias=B.get(nx,0)
    s=lambda x: 1/(1+np.exp(-(x@Wn.T)))+bias
    real=np.argsort(-s(recs[f'ffn_norm-{nx}']),1)[:,:K]
    for key in (f'ffn_moe_logits_biased-{nx}', f'ffn_moe_logits-{nx}'):
        if key in recs:
            d=recs[key]; ok.append(np.mean([len(set(a)&set(b))/K for a,b in zip(real,np.argsort(-d,1)[:,:K])])); break
    pe=s(recs[f'ffn_norm-{L}'])
    for m in rc:
        p=np.argsort(-pe,1)[:,:m]; rc[m].append(np.mean([len(set(a)&set(b))/K for a,b in zip(real,p)]))
print('offline-vs-dumped top8 agreement mean', np.mean(ok), 'min', np.min(ok))
for m in rc: print(f'recall of real top8 within predicted top{m}: mean {np.mean(rc[m]):.3f}')
for nx in ():
    raw=recs[f'ffn_norm-{nx}']@W[nx].T; d=recs[f'ffn_moe_logits-{nx}']
    print(nx, raw.shape, d.shape, 'maxabs diff', np.abs(raw-d).max(), 'corr', np.corrcoef(raw.ravel(), d.ravel())[0,1], 'bias?', nx in B)
def ov(a,b): return np.mean([len(set(x)&set(y))/K for x,y in zip(a,b)])
res={}
for L in layers:
    nx=L+1
    if f'ffn_norm-{L}' not in recs or nx not in W: continue
    X=recs[f'ffn_norm-{L}'].astype(np.float64); Y=recs[f'ffn_moe_logits-{nx}'].astype(np.float64)
    Wn=W[nx].astype(np.float64); b=B[nx]
    sel=lambda lg: np.argsort(-(1/(1+np.exp(-lg))+b),1)[:,:K]
    tr,te=slice(0,256),slice(256,None); real=sel(Y[te])
    res.setdefault('base',[]).append(ov(real, sel(X[te]@Wn.T)))
    # mean residual shift learned on train
    d=(recs[f'ffn_norm-{nx}'][tr]-X[tr]).mean(0)
    res.setdefault('shift',[]).append(ov(real, sel((X[te]+d)@Wn.T)))
    for lam in (1e1,1e2,1e3):
        A=X[tr].T@X[tr]+lam*np.eye(X.shape[1]); Wt=np.linalg.solve(A, X[tr].T@Y[tr]+lam*Wn.T)
        res.setdefault(f'ridge{lam:g}',[]).append(ov(real, sel(X[te]@Wt)))
print('n_embd', X.shape[1])
for k,v in res.items(): print(f'{k:10s} held-out top8 overlap mean {np.mean(v):.3f}')
from collections import OrderedDict
for C in (48, 72, 96):
  for M in (8, 12, 16):
    tot=miss=cov=0
    for L in layers:
        nx=L+1
        if f'ffn_norm-{L}' not in recs or nx not in W: continue
        b=B[nx]; sg=lambda lg: 1/(1+np.exp(-lg))+b
        real=np.argsort(-sg(recs[f'ffn_moe_logits-{nx}']),1)[:,:K]
        pred=np.argsort(-sg(recs[f'ffn_norm-{L}']@W[nx].T),1)[:,:M]
        c=OrderedDict()
        for t,(r,p) in enumerate(zip(real,pred)):
            ps=set(p)
            for e in r:
                tot+=1
                if e in c: c.move_to_end(e)
                else:
                    miss+=1; cov+= e in ps
                    c[e]=1
                    if len(c)>C: c.popitem(last=False)
    print(f'slots {C} hit {1-miss/tot:.3f}  predicted top{M} covers {cov/miss:.3f} of misses')
