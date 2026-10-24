#!/usr/bin/env python3
"""Per-position comparison of two llama-perplexity runs over the same text.

Each run is started with --kl-divergence-base FILE (run.py: PERF_PPL_EXTRA="--kl-divergence-base FILE"); the file holds, for every scored
position of every chunk (the second half of the context), the log-probabilities of the whole vocabulary (16-bit, 16 nats below the top).
This script reads two such files (A = reference, B = the variant) and prints, per position bin (averaged over the chunks):
  KLD(A||B)       how far B's next-token distribution is from A's
  dNLL            B's negative log-likelihood of the real next token minus A's (positive = B is worse)
A cut that loses information shows as a jump in both right after the cut position; rounding noise spreads evenly over all positions.

  python3 fork/ppl_pos.py A.bin B.bin [--bin 16] [--cuts 768] [--chunks N]
File layout (tools/perplexity/perplexity.cpp, log_softmax): "_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk, int32 tokens[n_chunk*n_ctx],
then per chunk (n_ctx - 1 - n_ctx/2) records of nv = 2*((n_vocab+1)/2)+4 uint16: float scale, float min_log_prob, uint16 q[n_vocab]
(log p = min_log_prob + scale*q; q = 0 means at or below min_log_prob).
"""
import argparse, sys
import numpy as np


def open_base(path):
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            sys.exit(f"{path}: not a --kl-divergence-base file")
        n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
    nv = 2 * ((int(n_vocab) + 1) // 2) + 4
    first = int(n_ctx) // 2
    n_tok = int(n_ctx) - 1 - first
    off_tokens = 8 + 12
    tokens = np.memmap(path, dtype=np.int32, mode="r", offset=off_tokens, shape=(int(n_chunk), int(n_ctx)))
    off = off_tokens + 4 * int(n_chunk) * int(n_ctx)
    recs = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(int(n_chunk), n_tok, nv))
    return dict(n_ctx=int(n_ctx), n_vocab=int(n_vocab), n_chunk=int(n_chunk), first=first, n_tok=n_tok, tokens=tokens, recs=recs)


def logprobs(rec, n_vocab):
    """[n_tok, nv] uint16 records -> [n_tok, n_vocab] float32 log-probabilities"""
    hdr = rec[:, :4].copy().view(np.float32)          # [n_tok, 2]: scale, min_log_prob
    q = rec[:, 4:4 + n_vocab].astype(np.float32)
    return hdr[:, 1:2] + hdr[:, 0:1] * q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--bin", type=int, default=16, help="positions per output row")
    ap.add_argument("--cuts", default="", help="comma list of absolute positions where B's ubatches start (marked in the table)")
    ap.add_argument("--chunks", type=int, default=0, help="use only the first N chunks")
    ap.add_argument("--prefix", action="store_true",
                    help="causality test: the texts agree only up to some token of chunk 1; compare chunk 1 only, up to the first differing token "
                         "(a causal model must give the same predictions there: any difference is information flowing back from later tokens)")
    a = ap.parse_args()
    A, B = open_base(a.a), open_base(a.b)
    for k in ("n_ctx", "n_vocab") + (() if a.prefix or a.chunks else ("n_chunk",)):
        if A[k] != B[k]:
            sys.exit(f"{k} differs: {A[k]} vs {B[k]}")
    nch = min(A["n_chunk"], B["n_chunk"]) if not a.chunks else min(a.chunks, A["n_chunk"], B["n_chunk"])
    nt, nvoc, first = A["n_tok"], A["n_vocab"], A["first"]
    if a.prefix:
        ta, tb = np.asarray(A["tokens"][0]), np.asarray(B["tokens"][0])
        diff = np.nonzero(ta != tb)[0]
        d0 = int(diff[0]) if len(diff) else A["n_ctx"]
        print(f"prefix mode: chunk 1 tokens agree up to index {d0 - 1}; positions predicting tokens {first + 1}..{d0 - 1} are compared")
        nch, nt = 1, max(0, d0 - 1 - first)
        if nt == 0:
            sys.exit("the texts differ before the scored half")
    elif not np.array_equal(A["tokens"][:nch], B["tokens"][:nch]):
        sys.exit("the two runs read different tokens")
    kld = np.zeros((nch, nt)); dnll = np.zeros((nch, nt)); top = np.zeros((nch, nt))
    for c in range(nch):
        la = logprobs(A["recs"][c][:nt], nvoc); lb = logprobs(B["recs"][c][:nt], nvoc)
        nxt = A["tokens"][c, first + 1:first + 1 + nt]
        pa = np.exp(la)
        kld[c] = np.sum(pa * (la - lb), axis=1)
        idx = np.arange(nt)
        dnll[c] = -(lb[idx, nxt] - la[idx, nxt])
        top[c] = (np.argmax(la, axis=1) == np.argmax(lb, axis=1))
    cuts = {int(x) for x in a.cuts.split(",") if x.strip()}
    print(f"{nch} chunks, n_ctx {A['n_ctx']}, scored positions {first + 1}..{first + nt} (position = index of the predicted token)")
    print(f"overall: mean KLD {kld.mean():.6f}, mean dNLL {dnll.mean():+.6f} (ppl ratio {np.exp(dnll.mean()):.5f}), same top token {100 * top.mean():.2f}%")
    print(f"{'positions':>12} {'mean KLD':>10} {'dNLL':>10} {'same top':>9}  cut")
    for s in range(0, nt, a.bin):
        e = min(nt, s + a.bin)
        p0, p1 = first + 1 + s, first + e
        mark = ",".join(str(x) for x in sorted(cuts) if p0 - 1 <= x <= p1) or ""
        print(f"{p0:>5}-{p1:<6} {kld[:, s:e].mean():>10.6f} {dnll[:, s:e].mean():>+10.6f} {100 * top[:, s:e].mean():>8.2f}%  {mark}")


if __name__ == "__main__":
    main()
