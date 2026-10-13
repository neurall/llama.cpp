#!/usr/bin/env python3
# kernel time by name inside the splits of one layer (device time), per card: ksplit.py trace.sqlite pass layer
import sqlite3, sys, bisect, re
db=sys.argv[1]; pi=int(sys.argv[2]); L=sys.argv[3]
c=sqlite3.connect(db)
p0=[r[0] for r in c.execute("select start from NVTX_EVENTS where text like 'SPLIT 0 %' order by start")]
ps=[]
for i,s in enumerate(p0):
    e=p0[i+1] if i+1<len(p0) else s+5e9
    n,last=c.execute("select count(*),max(end) from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %'",(s,e)).fetchone()
    if n>=20 and last-s>100e6: ps.append((s,e))
s,E_=ps[pi]
sp=c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start",(s,E_)).fetchall()
st=[x[0] for x in sp]
rt={cid:t for cid,t in c.execute("select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?",(s-1e6,E_))}
agg={}
for a,b,cid,nm in c.execute("select k.start,k.end,k.correlationId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<?",(s,E_+3e9)):
    if cid not in rt: continue
    i=bisect.bisect_right(st,rt[cid])-1
    tx=sp[i][2]
    if not re.search(r'(gate|up|down|out)-%s$'%L,tx) : continue
    key=re.sub(r'^SPLIT \d+ ','',tx)
    d=agg.setdefault(key,{})
    v=d.setdefault(nm,[0,0]); v[0]+=b-a; v[1]+=1
for key,d in agg.items():
    tot=sum(v[0] for v in d.values())/1e6
    print('%-44s %6.2f ms: %s'%(key,tot,', '.join('%s %.2f/%d'%(n[:22],v[0]/1e6,v[1]) for n,v in sorted(d.items(),key=lambda x:-x[1][0])[:5])))
