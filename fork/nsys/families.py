# decode-only comparison: kernels between mix marker #97*A and #97*B (whole tokens), per token: wall, busy (union), idle, per-family time and calls
import sqlite3,sys,collections
def load(f, marker, A=60, B=360):
    c=sqlite3.connect(f)
    rows=c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
    m=[r[0] for r in rows if marker in r[2]]
    t0,t1=m[97*A],m[97*B]; ntok=B-A
    w=[r for r in rows if t0<=r[0]<t1]
    busy=0; cur_s=cur_e=None
    for s,e,_ in w:
        if cur_e is None or s>cur_e:
            if cur_e is not None: busy+=cur_e-cur_s
            cur_s,cur_e=s,e
        else: cur_e=max(cur_e,e)
    busy+=cur_e-cur_s
    agg=collections.defaultdict(float); cnt=collections.defaultdict(int)
    for s,e,n in w: agg[n[:40]]+=(e-s)/1e3; cnt[n[:40]]+=1
    return dict(wall=(t1-t0)/1e3/ntok, busy=busy/1e3/ntok, n=len(w)/ntok, fam={k:(v/ntok,cnt[k]/ntok) for k,v in agg.items()})
o=load(sys.argv[1],"hcf_up",60,300); s=load(sys.argv[2],"gr_norm_split",10,105)
for name,d in (('OURS',o),('STRATA',s)):
    print(f"== {name}: wall {d['wall']:.0f} us/token, GPU busy {d['busy']:.0f}, idle {d['wall']-d['busy']:.0f}, kernels {d['n']:.0f}/token, kernel sum {sum(v[0] for v in d['fam'].values()):.0f}")
    for k,v in sorted(d['fam'].items(), key=lambda x:-x[1][0])[:int(sys.argv[3]) if len(sys.argv)>3 else 25]: print(f"  {k:40} {v[0]:8.1f} us/tok  {v[1]:6.1f} calls/tok  {v[0]/v[1]:6.2f} us/call")
