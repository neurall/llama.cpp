#!/usr/bin/env python3
# Per scheduler split: when the host issued it and when the card really ran it (kernels joined to their launch call by correlation id),
# plus every upload and every stream-wait-on-event with the stream the event was recorded on.
import sqlite3, sys, bisect
MINB=int(sys.argv[5]) if len(sys.argv)>5 else 400000
db=sys.argv[1]; pass_ms=float(sys.argv[2]); lo=int(sys.argv[3]); hi=int(sys.argv[4])
c=sqlite3.connect(db)
s=c.execute("select start from NVTX_EVENTS where start>=? and text like 'SPLIT 0 %' order by start limit 1",(pass_ms*1e6,)).fetchone()[0]
nxt=c.execute("select start from NVTX_EVENTS where start>? and text like 'SPLIT 0 %' order by start limit 1",(s,)).fetchone()
E_=nxt[0] if nxt else s+2000e6
sp=c.execute("select start,end,text from NVTX_EVENTS where start>=? and start<? and text like 'SPLIT %' order by start",(s,E_)).fetchall()
starts=[x[0] for x in sp]
def split_of(t):
    i=bisect.bisect_right(starts,t)-1
    if i<0: return None
    return i if t<=sp[i][1] else -i-1000   # negative = between splits (after split i)
rt={}
for cid,st,en,nm in c.execute("select r.correlationId,r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<?",(s-50e6,E_)):
    rt[cid]=(st,en,nm)
dev={}
def add(i,devid,stream,st,en,kind,extra=''):
    dev.setdefault(i,[]).append((st,en,devid,stream,kind,extra))
for st,en,d,strm,cid,nm in c.execute("select k.start,k.end,k.deviceId,k.streamId,k.correlationId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<?",(s-50e6,E_+400e6)):
    if cid in rt: add(split_of(rt[cid][0]),d,strm,st,en,'K',nm)
kinds={1:'H2D',2:'D2H',8:'D2D',10:'P2P'}
for st,en,d,strm,cid,ck,b in c.execute("select start,end,deviceId,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",(s-50e6,E_+400e6)):
    if cid in rt and b>MINB: add(split_of(rt[cid][0]),d,strm,st,en,kinds.get(ck,str(ck)),'%.1fMB issued@%.2f'%(b/1e6,(rt[cid][0]-s)/1e6))
evs={}
for d,strm,eid,esid in c.execute("select deviceId,streamId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT"):
    evs[(eid,esid)]=(d,strm)
waits={}
for st,en,d,strm,cid,eid,esid in c.execute("select start,end,deviceId,streamId,correlationId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2 and start>=? and start<?",(s,E_)):
    src=evs.get((eid,esid))
    waits.setdefault(split_of(st),[]).append((st,d,strm,src,en-st))
nm={0:'x4 ',1:'x16'}
for i in range(lo,hi+1):
    st,en,tx=sp[i]
    print('\n== %s   host %.2f-%.2f (%.2f ms)'%(tx[:52],(st-s)/1e6,(en-s)/1e6,(en-st)/1e6))
    for w in waits.get(i,[]):
        print('     wait   issued %8.2f: %s stream %d waits event recorded on %s  (host call %.2f ms)'%((w[0]-s)/1e6,nm[w[1]],w[2],('%s stream %d'%(nm[w[3][0]],w[3][1])) if w[3] else '?',w[4]/1e6))
    items=sorted(dev.get(i,[]))
    # group kernels per (dev,stream)
    grp={}
    for it in items:
        if it[4]=='K': grp.setdefault((it[2],it[3]),[]).append(it)
        else: print('     %s %s %8.2f-%8.2f (%5.2f ms) stream %d  %s'%(nm[it[2]],it[4],(it[0]-s)/1e6,(it[1]-s)/1e6,(it[1]-it[0])/1e6,it[3],it[5]))
    for (d,strm),ks in grp.items():
        busy=sum(k[1]-k[0] for k in ks)/1e6
        print('     %s RUN %8.2f-%8.2f  busy %6.2f ms  %3d kernels stream %d   (first kernel %.2f ms after host issue)'%(nm[d],(ks[0][0]-s)/1e6,(max(k[1] for k in ks)-s)/1e6,busy,len(ks),strm,(ks[0][0]-st)/1e6))
    for key in ( -i-1000,):
        for it in sorted(dev.get(key,[])):
            if it[4]!='K': print('   (after split) %s %s %8.2f-%8.2f (%5.2f ms) stream %d  %s'%(nm[it[2]],it[4],(it[0]-s)/1e6,(it[1]-s)/1e6,(it[1]-it[0])/1e6,it[3],it[5]))
