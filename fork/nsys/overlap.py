# real-work overlap: kernels that only wait (spin for the host or for the other stream) left out
import sqlite3,sys,collections
SPIN=('k_moe_rows','wait_flag_ge')
def run(f, marker, per_tok, A, B, top):
    c=sqlite3.connect(f)
    rows=c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
    m=[r[0] for r in rows if marker in r[3]]
    t0,t1=m[per_tok*A],m[per_tok*B]; ntok=B-A
    w=[r for r in rows if t0<=r[0]<t1]
    spin=[r for r in w if any(x in r[3] for x in SPIN)]; work=[r for r in w if not any(x in r[3] for x in SPIN)]
    ev=[]
    for i,(s,e,st,n) in enumerate(work): ev.append((s,1,i)); ev.append((e,0,i))
    ev.sort()
    act=set(); last=None; pair=collections.Counter(); multi=0; busy=0
    for t,kind,i in ev:
        if last is not None:
            d=t-last
            if len(act)>=1: busy+=d
            if len(act)>=2:
                multi+=d; names=sorted(work[j][3][:26] for j in act); pair[' + '.join(names[:3])]+=d
        last=t
        if kind: act.add(i)
        else: act.discard(i)
    ksum=sum(e-s for s,e,_,_ in work); ssum=sum(e-s for s,e,_,_ in spin); wall=t1-t0
    k=1e3*ntok
    print(f"== {f.split('/')[-1]}: wall {wall/k:.0f} | work kernels: {len(work)/ntok:.0f}/token, summed {ksum/k:.0f}, card busy with work {busy/k:.0f}, work in parallel {(ksum-busy)/k:.0f} | no work running {(wall-busy)/k:.0f} (spin kernels summed {ssum/k:.0f})  [us/token]")
    for kk,v in pair.most_common(top): print(f"   {v/k:7.1f}  {kk}")
for a in sys.argv[1:]:
    if 'strata' in a: run(a,"gr_norm_split",97,10,105,9)
    else: run(a,"topk_moe",48,60,300,9)   # one per layer in every build of ours (the HC fuse kernels are not)
