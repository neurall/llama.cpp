// pred-live: live next-layer MoE expert prediction lab on a running model (no dumps).
//
// A graph callback collects every MoE layer's input (ffn_norm-L), router logits (ffn_moe_logits-L) and real
// top-k (ffn_moe_topk-L); after each decode the tokens are processed in order by three predictors of layer
// L+k's experts, for k = 1..--ahead:
//   router : layer L+k's router applied to x_L (untrained baseline)
//   nlms   : linear map started as that router, trained online (delta rule), plus the previous token's
//            activation at L+k as a second input with its own step
//   hash   : key = sign-hash of the current x_L + the last n (6..1) tokens' activations at layer L+k;
//            the longest key seen before recalls the experts stored for it (stored the first time only,
//            with a hit counter); no match -> nlms
// Phases: --pretrain <text> (TinyStories, '<|endoftext|>'-separated) is prefilled to fill/train, then
// --n-stories answers to -p are generated at the sampling settings given, scored on generated tokens.
//
//   llama-pred-live -m olmoe.gguf -ngl 99 --pretrain TinyStories-valid.txt --pretrain-tokens 50000 \
//                   --n-stories 20 -n 400 --temp 1 -p "<|endoftext|><|user|>\nWrite a short story.\n<|assistant|>\n"

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

static constexpr int HIST = 6;     // key ladder length
static constexpr int BITS = 16;    // sign bits per hashed vector

struct entry {
    std::array<int32_t, 8> ids;
    uint32_t hits_x8 = 0;          // hit counter (overlap * 8, summed over recalls)
    uint32_t uses = 0;
};

struct pair_state {                // predictor of layer L+k from layer L
    std::vector<float> M, M2;      // [E][D] linear map on x_L (plain NLMS, also the fixed baseline), and on the previous token's x_{L+k}
    std::vector<float> M3;         // [E][E] on the previous token's real picks at L+k (one-hot): expert persistence
    std::vector<float> MF;         // [E][D] same as M with a faster step (mu_fast): step size picked per pair by results
    float ema[3] = {};             // running hit rate of the candidates: main, main + extras, fast; the best one predicts
    std::unordered_map<uint64_t, entry> mem;
};

struct counters { double tok = 0, router = 0, base = 0, nlms = 0, hash_rec = 0, hash_rec_hit = 0, nlms_on_rec = 0, combo = 0, coarse = 0; };

static int n_layer = 0, n_expert = 0, n_embd = 0, top_k = 0, ahead = 8;
static float mu = 0.5f, mu2 = 0.1f, mu3 = 0.1f, mu_fast = 1.0f;

// per-ubatch capture
static std::vector<std::vector<float>>   cap_x, cap_logits;
static std::vector<std::vector<int32_t>> cap_topk;
static int cap_n = 0;

// state
static std::vector<std::vector<float>> routers;             // [L][E*D]
static std::vector<std::vector<float>> proj_cur, proj_prev;   // [L][BITS*D]
static std::vector<pair_state> pairs;                       // [k-1][L]
static std::vector<std::vector<uint16_t>> ring;             // [L][HIST] previous tokens' codes, newest first
static std::vector<std::vector<float>> prev_x;              // [L][D] previous token's x
static std::vector<std::vector<int32_t>> prev_ids;          // [L][k] previous token's real picks
static std::vector<counters> cnt_run;                       // [k-1], this run only (reported at a pause)
static bool have_prev = false;
static std::vector<counters> cnt;                           // [k-1], current phase
static double scored = 0;
static int64_t last_live = 0;                               // live hit-rate line every 20 s (log + --report-file)
static FILE * report_file = nullptr;
static const char * phase_name = "";

static int layer_of(const char * name, const char * prefix) {
    // exactly "<prefix>-<layer>": views like "ffn_norm-3 (reshaped)" ([D, 1, n]) must not overwrite the capture
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    char * end = nullptr;
    const long il = strtol(name + n + 1, &end, 10);
    return end != name + n + 1 && *end == '\0' ? (int) il : -1;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void *) {
    int il;
    const bool want = (il = layer_of(t->name, "ffn_norm")) >= 0 || (il = layer_of(t->name, "ffn_moe_logits")) >= 0 ||
                      (il = layer_of(t->name, "ffn_moe_topk")) >= 0;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int64_t n = t->ne[1];
    // row by row: ffn_moe_topk is a strided view (first k columns of the argsort), a flat copy scrambles rows
    auto get = [&](auto & dst) {
        using T = typename std::decay_t<decltype(dst)>::value_type;
        dst.resize((size_t) n * t->ne[0]);
        for (int64_t i = 0; i < n; ++i) {
            ggml_backend_tensor_get(t, dst.data() + i * t->ne[0], i * t->nb[1], t->ne[0] * sizeof(T));
        }
    };
    if (strncmp(t->name, "ffn_norm", 8) == 0) {
        get(cap_x[il]);
        cap_n = (int) n;
    } else if (strncmp(t->name, "ffn_moe_logits", 14) == 0) {
        get(cap_logits[il]);
    } else {
        get(cap_topk[il]);
    }
    return true;
}

static uint16_t sign_code(const float * P, const float * x) {
    uint16_t c = 0;
    for (int b = 0; b < BITS; ++b) {
        float s = 0;
        for (int d = 0; d < n_embd; ++d) {
            s += P[b * n_embd + d] * x[d];
        }
        c |= (s > 0) << b;
    }
    return c;
}

static uint64_t mix(uint64_t h, uint64_t v) {       // splitmix-style combine
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 27;
    return h;
}

static void topk_of(const float * v, int32_t * out) {
    std::vector<int> idx(n_expert);
    for (int e = 0; e < n_expert; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    for (int i = 0; i < top_k; ++i) out[i] = idx[i];
}

static double overlap(const int32_t * pred, const int32_t * real) {
    int h = 0;
    for (int i = 0; i < top_k; ++i) for (int j = 0; j < top_k; ++j) h += pred[i] == real[j];
    return (double) h / top_k;
}

static void matvec(const float * M, const float * x, float * out) {
    for (int e = 0; e < n_expert; ++e) {
        float s = 0;
        const float * r = M + (size_t) e * n_embd;
        for (int d = 0; d < n_embd; ++d) s += r[d] * x[d];
        out[e] = s;
    }
}

// process the captured ubatch token by token, in order
static void process(bool score) {
    std::vector<std::vector<uint16_t>> cur_code(n_layer, std::vector<uint16_t>(cap_n)), tgt_code(n_layer, std::vector<uint16_t>(cap_n));
    #pragma omp parallel for collapse(2)
    for (int L = 0; L < n_layer; ++L) {
        for (int j = 0; j < cap_n; ++j) {
            const float * x = cap_x[L].data() + (size_t) j * n_embd;
            cur_code[L][j] = sign_code(proj_cur[L].data(), x);
            tgt_code[L][j] = sign_code(proj_prev[L].data(), x);
        }
    }
    for (int j = 0; j < cap_n; ++j) {
        std::vector<counters> part(ahead), part_run(ahead);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int k = 1; k <= ahead; ++k) {
            for (int L = 0; L < n_layer; ++L) {
                const int T = L + k;
                if (T >= n_layer) continue;
                pair_state & ps = pairs[(size_t) (k - 1) * n_layer + L];
                const float * x = cap_x[L].data() + (size_t) j * n_embd;
                const float * xp = prev_x[T].data();
                const int32_t * real = cap_topk[T].data() + (size_t) j * top_k;
                const float * y = cap_logits[T].data() + (size_t) j * n_expert;

                std::vector<float> p(n_expert), p2(n_expert), r(n_expert);
                int32_t pred[8];
                matvec(routers[T].data(), x, r.data());
                topk_of(r.data(), pred);
                const double h_router = overlap(pred, real);
                const int32_t * pids = prev_ids[T].data();
                matvec(ps.M.data(), x, p.data());                          // main map: trained exactly like the baseline
                std::vector<float> px(p);                                   // + extra inputs (previous token at L+k, its picks)
                if (have_prev) {
                    matvec(ps.M2.data(), xp, p2.data());
                    for (int e = 0; e < n_expert; ++e) {
                        px[e] += p2[e];
                        for (int i = 0; i < top_k; ++i) px[e] += ps.M3[(size_t) e * n_expert + pids[i]];
                    }
                }
                std::vector<float> pf(n_expert);
                matvec(ps.MF.data(), x, pf.data());
                double hcand[3];
                topk_of(p.data(), pred);  hcand[0] = overlap(pred, real);
                const double h_base = hcand[0];                             // plain NLMS = the fixed baseline
                topk_of(px.data(), pred); hcand[1] = overlap(pred, real);
                topk_of(pf.data(), pred); hcand[2] = overlap(pred, real);
                // the candidate with the best running hit rate predicts (decided before seeing this token's picks)
                const int best = ps.ema[1] > ps.ema[0] && ps.ema[1] >= ps.ema[2] ? 1 : ps.ema[2] > ps.ema[0] ? 2 : 0;
                const double h_nlms = hcand[best];
                for (int c = 0; c < 3; ++c) ps.ema[c] = 0.99f * ps.ema[c] + 0.01f * (float) hcand[c];

                // hash ladder: longest key first
                uint64_t keys[HIST];
                uint64_t h = mix(0, cur_code[L][j]);
                for (int n = 0; n < HIST; ++n) {
                    h = mix(h, ring[T][n]);
                    keys[n] = mix(h, n);
                }
                // coarse rung: the same ladder on 8-bit codes (current and previous), trusted only once proven (hit >= 7/8 on average)
                uint64_t ckeys[HIST];
                uint64_t hc = mix(0x5bd1e995ull, cur_code[L][j] >> 8);
                for (int n = 0; n < HIST; ++n) {
                    hc = mix(hc, ring[T][n] >> 8);                          // previous tokens' codes coarse too: more repeats across texts
                    ckeys[n] = mix(hc, n);
                }
                entry * got = nullptr;
                bool coarse = false;
                if (have_prev) {
                    for (int n = HIST - 1; n >= 0 && !got; --n) {
                        auto it = ps.mem.find(keys[n]);
                        if (it != ps.mem.end()) got = &it->second;
                    }
                    for (int n = HIST - 1; n >= 0 && !got; --n) {
                        auto it = ps.mem.find(ckeys[n]);
                        if (it == ps.mem.end()) continue;
                        entry & ce = it->second;
                        if (ce.uses > 0 && ce.hits_x8 >= 7 * ce.uses) {
                            got = &ce;
                            coarse = true;
                        } else {                                            // unproven: score silently so it can earn trust
                            ce.hits_x8 += (uint32_t) std::lround(overlap(ce.ids.data(), real) * top_k);
                            ce.uses++;
                            break;
                        }
                    }
                }
                double h_hash = h_nlms;
                if (got) {
                    h_hash = overlap(got->ids.data(), real);
                    got->hits_x8 += (uint32_t) std::lround(h_hash * top_k);
                    got->uses++;
                }
                if (have_prev) {
                    for (int n = 0; n < HIST; ++n) {
                        for (uint64_t key : { keys[n], ckeys[n] }) {
                            if (ps.mem.find(key) == ps.mem.end()) {       // store only the first time
                                entry e;
                                std::copy(real, real + top_k, e.ids.begin());
                                ps.mem.emplace(key, e);
                            }
                        }
                    }
                }

                // NLMS updates (own normaliser per input)
                float nx = 1e-6f, np = 1e-6f;
                for (int d = 0; d < n_embd; ++d) { nx += x[d] * x[d]; np += xp[d] * xp[d]; }
                for (int e = 0; e < n_expert; ++e) {
                    const float err = y[e] - p[e];                          // main map: its own error, as the baseline
                    float * m = ps.M.data() + (size_t) e * n_embd;
                    const float g = mu * err / nx;
                    for (int d = 0; d < n_embd; ++d) m[d] += g * x[d];
                    if (have_prev) {
                        const float err_x = y[e] - px[e];                   // extras: what main + extras still miss
                        float * m2 = ps.M2.data() + (size_t) e * n_embd;
                        const float g2 = mu2 * err_x / np;
                        for (int d = 0; d < n_embd; ++d) m2[d] += g2 * xp[d];
                        const float g3 = mu3 * err_x / top_k;                 // one-hot input: |x|^2 = k
                        for (int i = 0; i < top_k; ++i) ps.M3[(size_t) e * n_expert + pids[i]] += g3;
                    }
                    float * mf = ps.MF.data() + (size_t) e * n_embd;          // faster-step candidate
                    const float gf = mu_fast * (y[e] - pf[e]) / nx;
                    for (int d = 0; d < n_embd; ++d) mf[d] += gf * x[d];
                }

                if (score) {
                    #pragma omp critical
                    {
                        for (counters * cp : { &part[k - 1], &part_run[k - 1] }) {
                            counters & c = *cp;
                            c.tok++; c.router += h_router; c.base += h_base; c.nlms += h_nlms; c.combo += h_hash;
                            if (got) { c.hash_rec++; c.hash_rec_hit += h_hash; c.nlms_on_rec += h_nlms; c.coarse += coarse; }
                        }
                    }
                }
            }
        }
        for (int k = 0; k < ahead; ++k) {
            for (auto [a, b] : { std::pair<counters *, counters *>{ &cnt[k], &part[k] }, { &cnt_run[k], &part_run[k] } }) {
                a->tok += b->tok; a->router += b->router; a->base += b->base; a->nlms += b->nlms; a->hash_rec += b->hash_rec;
                a->hash_rec_hit += b->hash_rec_hit; a->nlms_on_rec += b->nlms_on_rec; a->combo += b->combo; a->coarse += b->coarse;
            }
        }
        // this token becomes the previous one
        for (int L = 0; L < n_layer; ++L) {
            std::copy(cap_x[L].begin() + (size_t) j * n_embd, cap_x[L].begin() + (size_t) (j + 1) * n_embd, prev_x[L].begin());
            std::rotate(ring[L].rbegin(), ring[L].rbegin() + 1, ring[L].rend());
            ring[L][0] = tgt_code[L][j];
            std::copy(cap_topk[L].begin() + (size_t) j * top_k, cap_topk[L].begin() + (size_t) (j + 1) * top_k, prev_ids[L].begin());
        }
        have_prev = true;
        if (score) scored++;
        if (score && ggml_time_us() - last_live > 20000000) {
            last_live = ggml_time_us();
            std::string line;
            for (int k : { 1, 4, 8 }) {
                if (k > ahead || cnt[k - 1].tok == 0) continue;
                const counters & c = cnt[k - 1];
                char b[128];
                snprintf(b, sizeof(b), "  L+%d base %.3f nlms %+.3f comb %+.3f hash %4.1f%%", k, c.base / c.tok, (c.nlms - c.base) / c.tok,
                         (c.combo - c.base) / c.tok, 100 * c.hash_rec / c.tok);
                line += b;
            }
            LOG("[%s %6.0f tok]%s\n", phase_name, scored, line.c_str());
            if (report_file) { fprintf(report_file, "[%s %6.0f tok]%s\n", phase_name, scored, line.c_str()); fflush(report_file); }
        }
    }
}

static void print_table(const char * phase, const std::vector<counters> & cs) {
    // gains are against the fixed baseline "base" (plain NLMS) on the same tokens
    LOG("\n%s: top-%d hit rate per lookahead (scored tokens x layers)\n", phase, top_k);
    if (report_file) fprintf(report_file, "\n%s\n", phase);
    LOG("  %-5s %8s %6s %6s %6s %6s %8s %6s %6s %6s %6s\n", "k", "n", "router", "base", "nlms", "+base", "hash rec", "coarse", "hit", "comb", "+base");
    for (int k = 0; k < ahead; ++k) {
        const counters & c = cs[k];
        if (c.tok == 0) continue;
        char line[256];
        snprintf(line, sizeof(line), "  L+%-3d %8.0f %6.3f %6.3f %6.3f %+6.3f %7.1f%% %5.1f%% %6.3f %6.3f %+6.3f\n", k + 1, c.tok,
                 c.router / c.tok, c.base / c.tok, c.nlms / c.tok, (c.nlms - c.base) / c.tok, 100 * c.hash_rec / c.tok,
                 c.hash_rec ? 100 * c.coarse / c.hash_rec : 0.0, c.hash_rec ? c.hash_rec_hit / c.hash_rec : 0.0, c.combo / c.tok, (c.combo - c.base) / c.tok);
        LOG("%s", line);
        if (report_file) fputs(line, report_file);
    }
    if (report_file) fflush(report_file);
}

static void report(const char * phase) {
    print_table(phase, cnt);
    cnt.assign(ahead, counters());
    scored = 0;
}

static void report_so_far(const char * phase) {
    print_table("this run", cnt_run);
    print_table(phase, cnt);
}

static std::function<bool()> g_stop = [] { return false; };

// decodes toks[from..]; returns the index reached (toks.size() when done, earlier when g_stop fired), -1 on error
static int decode_from(llama_context * ctx, const std::vector<llama_token> & toks, size_t from, int n_batch, bool score) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (size_t i = from; i < toks.size(); i += n_batch) {
        if (i > from && g_stop()) {
            llama_batch_free(batch);
            return (int) i;
        }
        common_batch_clear(batch);
        const size_t n = std::min((size_t) n_batch, toks.size() - i);
        for (size_t j = 0; j < n; ++j) {
            common_batch_add(batch, toks[i + j], (llama_pos) (i + j), { 0 }, true);   // all outputs: the last layer runs for every token
        }
        if (llama_decode(ctx, batch)) {
            llama_batch_free(batch);
            return -1;
        }
        if (getenv("PRED_DEBUG")) LOG("decode %zu tokens, captured %d rows (x0 %zu, topk15 %zu)\n", n, cap_n, cap_x[0].size() / n_embd, cap_topk[n_layer - 1].size() / top_k);
        process(score);
    }
    llama_batch_free(batch);
    return (int) toks.size();
}

static bool decode_all(llama_context * ctx, const std::vector<llama_token> & toks, int n_batch, bool score) {
    return decode_from(ctx, toks, 0, n_batch, score) == (int) toks.size();
}

static bool load_routers(const std::string & path) {
    ggml_context * gctx = nullptr;
    gguf_init_params gp = { false, &gctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return false;
    routers.assign(n_layer, {});
    for (int L = 0; L < n_layer; ++L) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", L);
        ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (!t) return false;
        routers[L].resize((size_t) n_expert * n_embd);
        if (t->type == GGML_TYPE_F32) {
            memcpy(routers[L].data(), t->data, routers[L].size() * sizeof(float));
        } else {
            ggml_get_type_traits(t->type)->to_float(t->data, routers[L].data(), (int64_t) routers[L].size());
        }
    }
    gguf_free(g);
    ggml_free(gctx);
    return true;
}

// ---- resumable state: run --time-limit seconds, save everything, exit; the next run continues ----
static constexpr uint32_t STATE_VERSION = 4;   // bump when the saved layout changes

struct progress {                 // phase 0 pretrain, 1 stories, 2 finished
    int32_t phase = 0; uint64_t at = 0; int32_t done = 0, story = 0;
    uint64_t chunk_end = 0; int32_t chunk_pos = 0;   // mid-chunk pause: chunk text [at, chunk_end), tokens decoded, KV in <state>.kv
};

template <typename T> static void wr(FILE * f, const T & v) { fwrite(&v, sizeof(T), 1, f); }
template <typename T> static void rd(FILE * f, T & v) { if (fread(&v, sizeof(T), 1, f) != 1) throw std::runtime_error("short state file"); }
template <typename T> static void wrv(FILE * f, const std::vector<T> & v) { uint64_t n = v.size(); wr(f, n); fwrite(v.data(), sizeof(T), n, f); }
template <typename T> static void rdv(FILE * f, std::vector<T> & v) {
    uint64_t n; rd(f, n); v.resize(n);
    if (n && fread(v.data(), sizeof(T), n, f) != n) throw std::runtime_error("short state file");
}

static void save_state(const std::string & path, const progress & pr) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    wr(f, STATE_VERSION); wr(f, pr); wr(f, ahead); wr(f, have_prev); wr(f, scored);
    for (auto & c : cnt) wr(f, c);
    for (auto & r : ring) wrv(f, r);
    for (auto & x : prev_x) wrv(f, x);
    for (auto & x : prev_ids) wrv(f, x);
    for (auto & ps : pairs) {
        wrv(f, ps.M); wrv(f, ps.M2); wrv(f, ps.M3); wrv(f, ps.MF); wr(f, ps.ema);
        uint64_t n = ps.mem.size(); wr(f, n);
        for (auto & kv : ps.mem) { wr(f, kv.first); wr(f, kv.second); }
    }
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

static bool load_state(const std::string & path, progress & pr) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    try {
        uint32_t ver; rd(f, ver);
        if (ver != STATE_VERSION) throw std::runtime_error("state from another version, start with a fresh one");
        int a; rd(f, pr); rd(f, a);
        if (a != ahead) throw std::runtime_error("--ahead differs from the saved state");
        rd(f, have_prev); rd(f, scored);
        for (auto & c : cnt) rd(f, c);
        for (auto & r : ring) rdv(f, r);
        for (auto & x : prev_x) rdv(f, x);
        for (auto & x : prev_ids) rdv(f, x);
        for (auto & ps : pairs) {
            rdv(f, ps.M); rdv(f, ps.M2); rdv(f, ps.M3); rdv(f, ps.MF); rd(f, ps.ema);
            uint64_t n; rd(f, n); ps.mem.clear(); ps.mem.reserve(n);
            for (uint64_t i = 0; i < n; ++i) { uint64_t k; entry e; rd(f, k); rd(f, e); ps.mem.emplace(k, e); }
        }
    } catch (const std::exception & e) {
        LOG_ERR("state %s: %s\n", path.c_str(), e.what());
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    // our own flags first, the rest goes to common_params
    std::string pretrain, report_path = "pred-live.log", state_path;
    int pretrain_tokens = 50000, n_stories = 20, chunk_chars = 2000;
    double time_limit = 0;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pretrain" && i + 1 < argc) pretrain = argv[++i];
        else if (a == "--pretrain-tokens" && i + 1 < argc) pretrain_tokens = atoi(argv[++i]);
        else if (a == "--n-stories" && i + 1 < argc) n_stories = atoi(argv[++i]);
        else if (a == "--ahead" && i + 1 < argc) ahead = atoi(argv[++i]);
        else if (a == "--mu" && i + 1 < argc) mu = (float) atof(argv[++i]);
        else if (a == "--mu2" && i + 1 < argc) mu2 = (float) atof(argv[++i]);
        else if (a == "--mu3" && i + 1 < argc) mu3 = (float) atof(argv[++i]);
        else if (a == "--mu-fast" && i + 1 < argc) mu_fast = (float) atof(argv[++i]);
        else if (a == "--report-file" && i + 1 < argc) report_path = argv[++i];
        else if (a == "--state" && i + 1 < argc) state_path = argv[++i];
        else if (a == "--time-limit" && i + 1 < argc) time_limit = atof(argv[++i]);
        else if (a == "--chunk-chars" && i + 1 < argc) chunk_chars = atoi(argv[++i]);
        else rest.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    report_file = fopen(report_path.c_str(), "a");
    llama_backend_init();
    params.cb_eval = capture_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    n_layer = llama_model_n_layer(model);
    n_embd = llama_model_n_embd(model);
    char buf[64];
    llama_model_meta_val_str(model, "olmoe.expert_count", buf, sizeof(buf));
    n_expert = atoi(buf);
    llama_model_meta_val_str(model, "olmoe.expert_used_count", buf, sizeof(buf));
    top_k = atoi(buf);
    if (n_expert <= 0 || top_k != 8 || !load_routers(params.model.path)) {
        LOG_ERR("needs an OLMoE-style model with top-8 routing (experts %d, used %d)\n", n_expert, top_k);
        return 1;
    }
    cap_x.assign(n_layer, {}); cap_logits.assign(n_layer, {}); cap_topk.assign(n_layer, {});
    std::mt19937 rng(0);
    std::normal_distribution<float> nd;
    proj_cur.assign(n_layer, std::vector<float>((size_t) BITS * n_embd));
    proj_prev.assign(n_layer, std::vector<float>((size_t) BITS * n_embd));
    for (int L = 0; L < n_layer; ++L) {
        for (auto & v : proj_cur[L]) v = nd(rng);
        for (auto & v : proj_prev[L]) v = nd(rng);
    }
    pairs.assign((size_t) ahead * n_layer, {});
    for (int k = 1; k <= ahead; ++k) {
        for (int L = 0; L + k < n_layer; ++L) {
            pair_state & ps = pairs[(size_t) (k - 1) * n_layer + L];
            ps.M = routers[L + k];
            ps.M2.assign((size_t) n_expert * n_embd, 0.0f);
            ps.M3.assign((size_t) n_expert * n_expert, 0.0f);
            ps.MF = routers[L + k];
        }
    }
    ring.assign(n_layer, std::vector<uint16_t>(HIST, 0));
    prev_x.assign(n_layer, std::vector<float>(n_embd, 0.0f));
    prev_ids.assign(n_layer, std::vector<int32_t>(top_k, 0));
    cnt_run.assign(ahead, counters());
    cnt.assign(ahead, counters());
    const int n_batch = std::min(params.n_batch, params.n_ubatch);
    const int n_ctx = llama_n_ctx(ctx);
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    progress pr;
    if (!state_path.empty() && load_state(state_path, pr)) {
        LOG("resumed %s: phase %d, pretrain %d tokens, story %d\n", state_path.c_str(), pr.phase, pr.done, pr.story);
    }
    if (pretrain.empty() && pr.phase == 0) pr.phase = 1;
    const int64_t t_start = ggml_time_us();
    auto out_of_time = [&]() { return time_limit > 0 && (ggml_time_us() - t_start) / 1e6 > time_limit; };
    auto pause = [&]() {
        if (!state_path.empty()) save_state(state_path, pr);
        LOG("\npaused after %.1f s (phase %d, pretrain %d tokens, story %d), state saved\n", (ggml_time_us() - t_start) / 1e6, pr.phase, pr.done, pr.story);
        report_so_far(pr.phase == 0 ? "pretrain so far" : "stories so far");
        llama_backend_free();
        return 3;
    };

    if (pr.phase == 0) {
        std::ifstream f(pretrain);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();
        phase_name = "pretrain";
        g_stop = out_of_time;
        while ((pr.chunk_pos > 0 || pr.done < pretrain_tokens) && pr.at < text.size()) {   // a paused chunk is always finished
            if (pr.chunk_pos == 0 && out_of_time()) return pause();
            // one chunk of whole stories; a pause inside it saves the KV cache and resumes at the exact token
            size_t at = pr.at;
            std::string chunk;
            while (at < text.size() && (pr.chunk_pos ? at < pr.chunk_end : chunk.size() < (size_t) chunk_chars)) {
                size_t end = text.find("<|endoftext|>", at);
                if (end == std::string::npos) end = text.size();
                chunk += text.substr(at, end - at) + "\n";
                at = end + 13;
            }
            auto toks = common_tokenize(ctx, chunk, true, false);
            if ((int) toks.size() > n_ctx) toks.resize(n_ctx);
            llama_memory_clear(mem, true);
            size_t from = 0;
            if (pr.chunk_pos > 0) {
                std::vector<llama_token> kv_toks(toks.size());
                size_t n_kv = 0;
                if (!llama_state_load_file(ctx, (state_path + ".kv").c_str(), kv_toks.data(), kv_toks.size(), &n_kv) || (int) n_kv != pr.chunk_pos) {
                    LOG_ERR("could not restore the KV cache of the paused chunk\n");
                    return 1;
                }
                from = pr.chunk_pos;
            }
            const int reached = decode_from(ctx, toks, from, n_batch, true);
            if (reached < 0) return 1;
            if (reached < (int) toks.size()) {                       // paused mid-chunk
                pr.done += reached - (int) from;
                pr.chunk_pos = reached;
                pr.chunk_end = at;
                llama_state_save_file(ctx, (state_path + ".kv").c_str(), toks.data(), reached);
                return pause();
            }
            pr.done += (int) toks.size() - (int) from;
            pr.chunk_pos = 0;
            pr.at = at;
        }
        g_stop = [] { return false; };
        LOG("\npretrain: %d tokens\n", pr.done);
        report("pretrain (TinyStories prefill, learning as it goes)");
        pr.phase = 1;
    }

    phase_name = "stories";
    for (; pr.story < n_stories; ++pr.story) {
        if (out_of_time()) return pause();
        llama_memory_clear(mem, true);
        auto prompt = common_tokenize(ctx, params.prompt, false, true);
        if (!decode_all(ctx, prompt, n_batch, false)) return 1;
        params.sampling.seed = pr.story;
        common_sampler * smpl = common_sampler_init(model, params.sampling);
        int pos = (int) prompt.size();
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (int i = 0; i < params.n_predict; ++i) {
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            if (llama_vocab_is_eog(vocab, id)) break;
            common_batch_clear(batch);
            common_batch_add(batch, id, pos++, { 0 }, true);
            if (llama_decode(ctx, batch)) return 1;
            process(true);
        }
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }
    pr.phase = 2;
    report(pretrain.empty() ? "stories, cold start" : "stories, after TinyStories pretrain");
    if (!state_path.empty()) save_state(state_path, pr);
    llama_backend_free();
    return 0;
}
