# per-layer wall time by layer kind (interval topk_moe -> next topk_moe), token time, and in delta-net layers the wait of the gated norm
import sqlite3,sys,statistics as st
for f in sys.argv[1:]:
    c=sqlite3.connect(f)
    rows=c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
    tk=[i for i,r in enumerate(rows) if 'topk_moe' in r[3]]
    # decode region: topk_moe spacing < 1 ms
    kinds={'gdn':[],'att':[],'other':[]}; waits=[]; zpos=[]
    for a,b in zip(tk,tk[1:]):
        w=(rows[b][0]-rows[a][0])/1e3
        if w>1000: continue
        names=[r[3] for r in rows[a:b]]
        k='gdn' if any('gated_delta_net' in n for n in names) else 'att' if any('flash_attn' in n for n in names) else 'other'
        kinds[k].append(w)
        if k=='gdn':
            g=[i for i in range(a,b) if 'gated_delta_net' in rows[i][3]][0]
            n=[i for i in range(g+1,b) if 'unary_gated_op' in rows[i][3]]
            if n: waits.append((rows[n[0]][0]-rows[g][1])/1e3)
    # tokens: gaps between topk_moe runs > 1 ms are not decode; count 48 layers a token
    nl=sum(len(v) for v in kinds.values())
    print(f"{f.split('/')[-1]}: layers {nl}, ", ", ".join(f"{k} {len(v)} x mean {st.mean(v):.1f} median {st.median(v):.1f} us" for k,v in kinds.items() if v))
    if waits: print(f"   delta-net end to gated norm start: mean {st.mean(waits):.1f} median {st.median(waits):.1f} us")
    est=(36*st.median(kinds['gdn'])+12*st.median(kinds['att']))/1e3 if kinds['att'] else 0
    print(f"   48 layers at the medians: {est:.2f} ms")
