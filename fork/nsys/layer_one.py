# one delta-net layer interval (topk_moe -> topk_moe) near the median wall time, from the middle of decode
import sqlite3,sys,statistics as st
f=sys.argv[1]; c=sqlite3.connect(f)
rows=c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
tk=[i for i,r in enumerate(rows) if 'topk_moe' in r[3]]
L=[(a,b,(rows[b][0]-rows[a][0])/1e3) for a,b in zip(tk,tk[1:]) if any('gated_delta_net' in r[3] for r in rows[a:b]) and (rows[b][0]-rows[a][0])<1e6]
med=st.median(w for _,_,w in L)
a,b,w=min(L[len(L)//2:], key=lambda t: abs(t[2]-med))
print(f"{b-a} kernels, {w:.1f} us wall (median {med:.1f})")
for i in range(a,b):
    s,e,sid,n=rows[i]; print(f"  {(s-rows[a][0])/1e3:7.1f} +{(e-s)/1e3:5.2f} s{sid} {n[:50]}")
