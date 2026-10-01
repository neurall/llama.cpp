#!/usr/bin/env python3
"""pool_sim.py [--slots S] [--cycles C] [policy ...]: replays the GLM router traces (code, explain, sql, story; 500 tokens x 42 layers x 8 of 256 experts)
as one mixed stream, C times round, against cache policies on one pool of S x layers slots (or S slots per layer for 'perlayer').
Per step (token): misses are candidates (best first), at most INS inserts per layer and BUDGET evictions in all; a newcomer must beat the victim.
Policies (name:key=val,...): perlayer / poolold (old score: window + 16 x lifetime share per layer; per layer / pooled), pool (wg, ww, n weights of the
lifetime share of the pool's busiest expert and the window use), churn (pool + cs, gx: minimum stay of an expert that was evicted and needed again),
coact (pool + wc: context score, the sum over the experts active in the previous layer of a learned co-occurrence row)."""
import sys, re, collections, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("policies", nargs="*"); ap.add_argument("--slots", type=int, default=85); ap.add_argument("--cycles", type=int, default=3)
ap.add_argument("--traces", default="/p/bw/data/sess-0929/traces"); ap.add_argument("--ins", type=int, default=2); ap.add_argument("--budget", type=int, default=24)
a = ap.parse_args()
E = 256; topics = ["code", "explain", "sql", "story"]
def load(t):
    per = collections.defaultdict(list)
    for l in open(f"{a.traces}/moelog_{t}.txt"):
        p = l.split(); per[int(re.search(r"blk\.(\d+)\.", p[0]).group(1))].append([int(x) for x in p[1:]])
    L = len(per); T = min(len(v) for v in per.values())
    return np.array([[per[l][t] for l in range(L)] for t in range(T)])  # [token, layer, 8]
streams = [load(t) for t in topics]; L = streams[0].shape[1]
def run(spec):
    name, _, kv = spec.partition(":"); P = dict(x.split("=") for x in kv.split(",") if x); P = {k: float(v) for k, v in P.items()}
    wg, ww, n = P.get("wg", .3), P.get("ww", .7), int(P.get("n", 64)); cs, gx, wc = P.get("cs", 0), P.get("gx", 1), P.get("wc", 0)
    pooled = name != "perlayer"; T = a.slots * L
    cached = np.zeros((L, E), bool); glob = np.zeros((L, E)); win = np.zeros((L, E)); churn = np.zeros((L, E)); stay = np.zeros((L, E))
    M = np.zeros((L, E, E), np.float32) if wc else None
    # hmm: K latent states with per-(layer, expert) usage C[s]; a sticky transition; belief b filtered online from each token's active experts
    # adapt: exponentially decayed use counts whose time constant follows the cosine between this token's expert pattern and the counts (topic similarity):
    # tau = tmin + (tmax - tmin) * sim^pw (smoothed by sm); fixed tau (tmin == tmax) is the plain decayed count
    tmin, tmax, pw, sm = P.get("tmin", 16), P.get("tmax", 256), P.get("pw", 1.0), P.get("sm", .7)
    if name == "adapt": cnt = np.zeros((L, E)); simv = 0.5
    K = int(P.get("K", 0)); wh, alpha, stick, hdec = P.get("wh", 1), P.get("alpha", .02), P.get("stick", .9), P.get("hdec", .999)
    if name == "hmm":
        rng = np.random.default_rng(1); C = rng.random((K, L, E)).astype(np.float64) * .5; b = np.ones(K) / K; pred = np.zeros((L, E))
    ghost = collections.deque(); ghost_set = np.zeros((L, E), bool); recent = collections.deque(); occ = np.zeros(L, int)
    hits = [0, 0]; ev = 0; step = 0; lastact = None; seg = []; seg100 = []
    for cyc in range(a.cycles):
        for si, S in enumerate(streams):
            h0 = [0, 0]; first = [0, 0]
            if P.get("iso", 0):
                cached[:] = False; glob[:] = 0; win[:] = 0; churn[:] = 0; stay[:] = 0; occ[:] = 0; recent.clear(); ghost.clear(); ghost_set[:] = False
                if P.get("prof", 0):
                    prof = np.zeros((L, E))
                    for sj, S2 in enumerate(streams):
                        if sj != si:
                            for l in range(L): np.add.at(prof[l], S2[:, l].ravel(), 1)
                    top = np.argsort(-prof.ravel())[:T]
                    cached.ravel()[top[prof.ravel()[top] > 0]] = True; occ[:] = cached.sum(1); glob[:] = prof * 0 + prof  # lifetime counts seeded as the engine does
            for tok in range(S.shape[0]):
                step += 1; act = S[tok]; miss = []
                for l in range(L):
                    for e in act[l]:
                        glob[l, e] += 1; win[l, e] += 1
                        if cached[l, e]: hits[0] += 1; h0[0] += 1; first[0] += tok < 100
                        else:
                            hits[1] += 1; h0[1] += 1; first[1] += tok < 100; miss.append((l, e))
                            if ghost_set[l, e]: churn[l, e] += 1; ghost_set[l, e] = False
                recent.append(act)
                if len(recent) > n:
                    old = recent.popleft()
                    for l in range(L):
                        for e in old[l]: win[l, e] -= 1
                if name == "adapt":
                    oh = np.zeros((L, E)); 
                    for l in range(L): oh[l, act[l]] = 1.0
                    nc = np.linalg.norm(cnt)
                    cosv = float((oh * cnt).sum() / (np.linalg.norm(oh) * nc)) if nc > 0 else 0.5
                    simv = sm * simv + (1 - sm) * cosv
                    tau = tmin + (tmax - tmin) * max(0.0, min(1.0, simv)) ** pw
                    cnt = cnt * np.exp(-1.0 / tau) + oh
                if M is not None and lastact is not None:
                    for l in range(1, L):
                        for p in act[l - 1]: M[l, p, act[l]] += 1
                lastact = act
                if name == "hmm":
                    pi = (C + alpha) / (C + alpha).sum(2, keepdims=True)
                    lp = np.log(pi)
                    ll = np.array([sum(lp[s_, l, act[l]].sum() for l in range(L)) for s_ in range(K)])
                    bp = stick * b + (1 - stick) / K
                    b = bp * np.exp(ll - ll.max()); b /= b.sum()            # the posterior of this token's state
                    for s_ in range(K):
                        C[s_] *= hdec
                        for l in range(L): C[s_, l, act[l]] += b[s_]
                    pi = (C + alpha) / (C + alpha).sum(2, keepdims=True)
                    bn = stick * b + (1 - stick) / K                        # the belief for the next token
                    pred = 8.0 * np.tensordot(bn, pi, axes=1)
                gmax = max(1.0, glob.max()); gl = np.maximum(1.0, glob.max(axis=1, keepdims=True))
                if name in ("perlayer", "poolold"): val = win + 16.0 * glob / gl
                elif name == "hmm": val = ww * win / n + wh * pred
                elif name == "adapt": val = cnt
                else: val = wg * glob / gmax + ww * win / n
                if wc and M is not None:
                    ctx = np.zeros((L, E))
                    for l in range(1, L):
                        ctx[l] = M[l, act[l - 1]].sum(0)
                    ctx /= max(1.0, ctx.max()); val = val + wc * ctx
                budget = a.budget; ins = np.zeros(L, int)
                for l, e in sorted(miss, key=lambda x: -val[x]):
                    if cached[l, e] or ins[l] >= a.ins or val[l, e] <= 0: continue
                    free = (T - cached.sum()) > 0 if pooled else occ[l] < a.slots
                    if not free:
                        if budget <= 0: break
                        cv = np.where(cached & (stay <= step), val, np.inf) if pooled else np.where(cached[l] & (stay[l] <= step), val[l], np.inf)
                        k = int(np.argmin(cv)); vl, ve = divmod(k, E) if pooled else (l, k)
                        if cv.flat[k] == np.inf or val[l, e] <= cv.flat[k]: continue
                        cached[vl, ve] = False; occ[vl] -= 1; budget -= 1; ev += 1
                        ghost.append((vl, ve)); ghost_set[vl, ve] = True
                        while len(ghost) > gx * T: gl_, ge_ = ghost.popleft(); ghost_set[gl_, ge_] = False
                    cached[l, e] = True; occ[l] += 1; ins[l] += 1
                    if cs: stay[l, e] = step + cs * (1 + min(churn[l, e], 8))
            if cyc == a.cycles - 1: seg.append(100.0 * h0[0] / max(1, sum(h0))); seg100.append(100.0 * first[0] / max(1, sum(first)))
    last_n = 0
    return spec, 100.0 * hits[0] / sum(hits), seg, ev / step, seg100
pols = a.policies or ["perlayer", "poolold", "pool:wg=.3,ww=.7,n=64", "pool:wg=0,ww=1,n=64", "pool:wg=0,ww=1,n=128", "pool:wg=0,ww=1,n=256"]
print(f"slots/layer {a.slots} (pool {a.slots * L}), cycles {a.cycles}; hit % overall | last cycle by topic {topics} | evictions per token")
for p in pols:
    s, h, seg, e, s100 = run(p); print(f"{s:40} {h:5.1f}% | " + " ".join(f"{x:5.1f}" for x in seg) + f" | {e:5.1f} | first 100 tokens: " + " ".join(f"{x:5.1f}" for x in s100), flush=True)
