"""Shared harness for the MoE expert-predictor algorithm search.

Data: a .pt stream built by pred_lab.py (X [NL,T,D] MoE inputs, Y [NL,T,E] router logits, W routers, bounds)
plus the token ids of each sample, re-tokenized from its source text and cached next to the .pt.
Every algorithm streams the tokens in order (one user session, online learning) and reports a hit matrix
hits[l, t] = fraction of layer l's real top-k it predicted for token t (NaN where it made no prediction).
"""
import ast, glob, os, re, subprocess, sys, warnings
warnings.filterwarnings("ignore", "Mean of empty slice")

import numpy as np
import torch

MODEL = "/m/o/8/4/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
TOKENIZE = os.path.join(os.path.dirname(__file__), "..", "..", "build-cuda", "bin", "llama-tokenize")
DATA = os.path.expandvars("$CLAUDE_JOB_DIR/tmp/olmo")
REF = None  # per-token hits of the reference algos (ref_<data>.npz from nlms.py --save), set by load()
SETS = {"simcity": ("simcity.pt", "S*.txt"), "session": ("session.pt", "L*.txt"),
        "tiny": ("tiny.pt", "T*.txt"),    # TinyStories prefill (no chat prompt: every token counts)
        "story": ("story.pt", "Y*.txt")}  # OLMoE's own "Write a short story." answers, temp 1


def tokenize(path):
    out = subprocess.run([TOKENIZE, "-m", MODEL, "-f", path, "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True).stdout
    return ast.literal_eval(out.strip())


def load(name, need_x=False):
    """-> dict(tokens [T] int64, real [NL,T,k] int64, bounds, and X/W when need_x)"""
    global REF
    pt, pattern = SETS[name]
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ref_{name}.npz")
    REF = dict(np.load(ref)) if os.path.exists(ref) else None
    d = torch.load(os.path.join(DATA, pt), mmap=True)
    bounds, k = d["bounds"], d["k"]
    cache = os.path.join(DATA, name + ".tokens.pt")
    if os.path.exists(cache):
        tokens = torch.load(cache)
    else:
        # match source texts to samples by token count (a sample whose dump failed was dropped)
        lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        toks, gen, i = [], [], 0
        for f in sorted(glob.glob(os.path.join(DATA, pattern))):
            if i < len(lens) and len(t := tokenize(f)) == lens[i]:
                # the prompt ends with the assistant tag: tokens after it were generated
                text = open(f).read()
                n_prompt = 0
                if "<|assistant|>\n" in text:
                    head = text[:text.index("<|assistant|>\n") + len("<|assistant|>\n")]
                    with open(os.path.join(DATA, "_head.txt"), "w") as h:
                        h.write(head)
                    n_prompt = len(tokenize(os.path.join(DATA, "_head.txt")))
                toks += t
                gen += [False] * n_prompt + [True] * (len(t) - n_prompt)
                i += 1
        assert i == len(lens), f"only matched {i} of {len(lens)} samples"
        tokens = {"ids": torch.tensor(toks), "gen": torch.tensor(gen)}
        torch.save(tokens, cache)
    out = {"tokens": tokens["ids"], "gen": tokens["gen"].numpy(), "real": torch.topk(d["Y"], k, dim=-1).indices, "bounds": bounds, "k": k,
           "n_expert": d["Y"].shape[2]}
    if need_x:
        out["X"], out["W"], out["Y"] = d["X"], d["W"], d["Y"]
    return out


def report(name, hits, bounds, extra="", gen=None, cols=None):
    """hits [NL, T] (NaN = no prediction). Prints avg over windows used to rank convergence.
    With gen (bool [T]) a second report covers only the generated tokens (prompts often repeat earlier text)."""
    if gen is not None:
        report(name, hits, bounds, extra)
        report(name + " [generated only]", np.where(gen[None, :], hits, np.nan), bounds, cols=gen)
        return
    h = np.asarray(hits, dtype=np.float64)
    T = h.shape[1]
    tok = np.nanmean(h, 0)                                  # per token over layers
    def avg(a, b):
        s = tok[a:min(b, T)]
        return np.nanmean(s) if np.isfinite(s).any() else float("nan")
    cov = np.isfinite(h[:, cols] if cols is not None else h).mean()  # over the reported tokens
    print(f"{name}: all {avg(0, T):.3f}  0-256 {avg(0, 256):.3f}  0-1024 {avg(0, 1024):.3f}  "
          f"last4k {avg(T - 4096, T):.3f}  coverage {cov:.3f}{extra}")
    m = re.match(r"L\+(\d)", name)
    if REF is not None and m and f"nlms_k{m.group(1)}" in REF:
        # the reference algos on exactly the tokens this run predicted (segments differ in difficulty)
        mask = np.isfinite(h)
        same = {r: np.nanmean(np.where(mask, REF[f"{r}_k{m.group(1)}"], np.nan)) for r in ("nlms", "router")}
        print(f"  same tokens: nlms {same['nlms']:.3f}  router {same['router']:.3f}  -> vs nlms {np.nanmean(h) - same['nlms']:+.3f}")
    win = [avg(a, a + 1024) for a in range(0, T, 1024)]
    print("  per 1024 tok: " + " ".join(f"{w:.2f}" for w in win))
    lay = np.nanmean(h, 1)
    print("  per layer:    " + " ".join(f"{v:.2f}" for v in lay))


def overlap(pred, real):
    """pred, real: [..., k] ids -> fraction of real found in pred"""
    return (np.asarray(real)[..., :, None] == np.asarray(pred)[..., None, :]).any(-1).mean(-1)


def topk_hit(logits, real):
    """logits [..., E], real [..., k] ids (torch) -> fraction of real in the predicted top-k, as numpy [...]"""
    p = torch.topk(logits, real.shape[-1], dim=-1).indices
    return (p.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().mean(-1).cpu().numpy()


class Run:
    """Built-in timer + checkpoint shared by every algo: resume the saved state and stream position, stop after
    --time-limit seconds, save state, position and the per-token hits accumulated over all runs (so each report
    shows the whole convergence curve so far). --fresh starts over."""

    def __init__(self, a, algo, T):
        import time
        self.a, self.T, self._time = a, T, time.time
        os.makedirs(a.ckpt_dir, exist_ok=True)
        self.path = os.path.join(a.ckpt_dir, f"{algo}{a.tag}.{a.data}.pt")
        self.state, self.start, self.hits, self.touched = {}, 0, {}, set()
        if a.init_from and not os.path.exists(self.path):
            self.state = torch.load(a.init_from, weights_only=False)["state"]   # learned state only: new stream, new hits
        if not a.fresh and os.path.exists(self.path):
            ck = torch.load(self.path, weights_only=False)
            self.state, self.start, self.hits = ck["state"], ck["pos"] % T, ck["hits"]
        self.stop = T
        self.t0 = self._time()

    def expired(self, t):
        """call at the top of the token loop; True (and records where it stopped) once the time is up"""
        if self.a.time_limit and t > self.start and t % 16 == 0 and self._time() - self.t0 > self.a.time_limit:
            self.stop = t
            return True
        return False

    def hit(self, name, NL):
        """the cumulative hit matrix [NL, T] for this name"""
        self.touched.add(name)
        if name not in self.hits:
            self.hits[name] = np.full((NL, self.T), np.nan)
        return self.hits[name]

    def save(self, **state):
        torch.save({"state": state, "pos": self.stop % self.T, "hits": self.hits}, self.path)

    def report(self, d):
        """every hit matrix over all tokens so far, then this run's window alone"""
        span = f"  (this run: tokens {self.start}-{self.stop} of {self.T}, {self._time() - self.t0:.1f} s)"
        for name, h in self.hits.items():
            if name not in self.touched:
                continue                            # a variant that did not run this time
            report(name, h, d["bounds"], span, d["gen"])
            w = np.full_like(h, np.nan)
            w[:, self.start:self.stop] = h[:, self.start:self.stop]
            report(name + " [this run]", w, d["bounds"])


def cli(desc):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data", default="simcity", choices=list(SETS))
    ap.add_argument("--device", default="cuda", help="cuda (pick the GPU with CUDA_VISIBLE_DEVICES) or cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (several runs share the 16)")
    ap.add_argument("--time-limit", type=float, default=20, help="save and exit after N seconds (0: run the whole stream)")
    ap.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt"))
    ap.add_argument("--tag", default="", help="checkpoint name suffix, to keep variants apart")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint, start over")
    ap.add_argument("--init-from", default=None, help="start from another run's learned state (e.g. pretrained on tiny)")
    return ap


def parse(ap):
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    return a
