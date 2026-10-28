// Hot expert maps per topic (task 158; user idea 2026-10-27). The one hot map in moe-state.ini sums every request the model ever served, so a prompt about one
// thing finds the experts of all the others cached: DeepSeek-V4.1 on the tetris test ran its prompt at 6.8 t/s from a state diluted by prose, JS and Python
// requests and at 18.9 t/s after a few tetris runs had made the state tetris-like again (same build, same day). Here every request leaves its own map: the
// key is the pooled input embedding of its prompt's tokens (centered on the vocabulary's mean, unit length), the value the experts it used per layer (prompt
// and answer). Before a prompt is read, its key finds the nearest stored maps by cosine similarity and the cache fills from their merge
// (llama-moecache-fork.cpp, topic_begin). People ask similar things about similar code: the nearest map is often a request on the same code.
//
// Two keys per record, one per pooling (KEY_MEAN: the mean of the token embeddings, KEY_MAX: per value the largest); --moe dynhot-pool picks the one the
// lookup uses, and since both are stored, one run shows which of the two predicts the experts a request uses (run.py hotmapmat).
//
// The store is one flat file per model (<user cache dir>/hotmaps/<model>.hotmap), fixed-size records searched by brute force: 1024 records of a 5120-wide key
// take 2.2 ms (PC2), against seconds of expert uploads the answer saves (a vector index - HNSW as Chroma uses, or product quantization - pays only from about
// a million records). A request whose key is near a stored record's (4-bit cosine >= merge_cos) joins that record: one record per topic, not per request
// (add); merge_cos > 1 keeps every request as its own record.
// No ggml in here: tests/test-moetopic-fork.cpp drives it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// A key is stored in 4 bits per value (8x smaller than float: 2.5 KB for a 5120-wide key, so a search reads 2.5 MB for 1024 records instead of 20 MB and
// the file stays small): value = q * qinv with q in -7..7 and qinv = 1 / |q|, so the stored vector has unit length again. The step is 4 times the rms
// of a unit key (4 / sqrt(n)) / 7 and larger values are clipped: on random keys the cosine to the exact key stays above 0.97, far inside the gap between
// requests on one topic and on another.
// Two values per byte (q + 8: the low nibble the even index), n_embd rounded up to even.
static inline void moe_topic_quant(const float * key, uint32_t n, uint8_t * out, float * qinv) {
    const float step = 4.0f / (std::sqrt((float) n) * 7.0f);
    double n2 = 0.0;
    memset(out, 0x88, (n + 1) / 2);   // q = 0 in both nibbles
    for (uint32_t j = 0; j < n; ++j) {
        const int q = std::max(-7, std::min(7, (int) std::lround(key[j] / step)));
        n2 += (double) q * q;
        out[j / 2] = (uint8_t) ((out[j / 2] & ((j & 1) ? 0x0F : 0xF0)) | ((q + 8) << ((j & 1) ? 4 : 0)));
    }
    *qinv = n2 > 0.0 ? (float) (1.0 / std::sqrt(n2)) : 0.0f;
}

// the stored vector (unit length) back as floats
static inline void moe_topic_dequant(const uint8_t * in, uint32_t n, float qinv, float * out) {
    for (uint32_t j = 0; j < n; ++j) { out[j] = (float) (((in[j / 2] >> ((j & 1) ? 4 : 0)) & 15) - 8) * qinv; }
}

// the dot of a float key with a stored key, times the stored key's qinv outside: 8 independent sums, so the compiler vectorizes it without -ffast-math
static inline float moe_topic_qdot(const uint8_t * q, const float * key, uint32_t n) {
    float s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        for (uint32_t k = 0; k < 4; ++k) {
            const uint8_t b = q[j / 2 + k];
            s[2 * k]     += key[j + 2 * k]     * (float) ((b & 15) - 8);
            s[2 * k + 1] += key[j + 2 * k + 1] * (float) ((b >> 4) - 8);
        }
    }
    float t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (; j < n; ++j) { t += key[j] * (float) (((q[j / 2] >> ((j & 1) ? 4 : 0)) & 15) - 8); }
    return t;
}

struct moe_topic_store {
    static constexpr char MAGIC[8] = { 'M', 'O', 'E', 'T', 'O', 'P', 'C', '6' };
    enum { KEY_MEAN = 0, KEY_MAX = 1, N_KEYS = 2 };
    uint32_t n_embd = 0, n_layers = 0, n_exp = 0;
    size_t   max_records = 4096;   // the oldest-used record is replaced past this (0: no limit)
    int      pool = KEY_MAX;       // the key the lookup and the merge candidates use
    std::vector<uint8_t>  qkeys;   // records x N_KEYS x key_bytes(): the keys in 4 bits per value (moe_topic_quant)
    std::vector<float>    fkeys;   // records x N_KEYS x n_embd: the pooled keys as they were (unit length); the file holds both these and the 4-bit copies the search reads
    std::vector<float>    qinv;    // records x N_KEYS: 1 / length of each quantized key
    std::vector<uint32_t> counts;  // records x n_layers x n_exp: uses of each expert
    std::vector<uint32_t> n_req;   // requests merged into each record
    std::vector<float>    hit;     // the best cache hit rate (0..1) a request on this record's topic reached; a request that did worse does not change the record (<0: unknown)
    std::vector<uint32_t> src;     // what steered the cache when that hit rate was measured: 1 = the hot maps (dynhot=full|read), 0 = the ini's hot experts (dynhot=store)
    std::vector<uint64_t> last;    // the request number that last used a record (memory only: picks the record to replace)
    uint64_t clock = 0;
    std::string path;
    float last_cos = 0.0f, last_best = 0.0f;   // the last add: the nearest record's cosine (4-bit, as the lookup reads it), how alike the maps were (the log)
    // the model's global map: every stored request's counts added, none dropped (as the ini's hot map counted them). Its hottest are the experts every request
    // needs whatever the task - language, grammar, syntax (user 2026-10-28) - and it stands in where the nearest topic is far (merge). In the file after the
    // header: requests (uint32), then the counts
    std::vector<uint32_t> glob;
    uint32_t glob_req = 0;
    bool     use_glob = true;    // merge blends it in (--moe dynhot-global=0: the nearest topics only, for the A/B)
    float    last_wg  = 0.0f;    // the last merge: the global map's weight
    bool  last_merged = false, last_kept = false;   // last_merged: the request went to a stored record; last_kept: its prompt, but no better hit rate: nothing changed
    bool  last_same = false;                         // the last add was a prompt already stored (its key as alike as SAME_KEY)
    static constexpr float SAME_KEY = 0.999f;        // keys this alike (cosine of the float keys) are one prompt: one record, its map replaced only by a better hit rate

    size_t size() const { return n_req.size(); }
    size_t rec_cells() const { return (size_t) n_layers * n_exp; }
    size_t key_bytes() const { return (n_embd + 1) / 2; }
    size_t rec_bytes() const { return sizeof(float) + 2 * sizeof(uint32_t) + N_KEYS * (sizeof(float) + key_bytes() + n_embd * sizeof(float)) + rec_cells() * sizeof(uint32_t); }
    static size_t head_bytes() { return sizeof(MAGIC) + 4 * sizeof(uint32_t); }
    size_t glob_bytes() const { return sizeof(uint32_t) + rec_cells() * sizeof(uint32_t); }
    uint8_t *       qk(size_t i, int kind)       { return qkeys.data() + (i * N_KEYS + kind) * key_bytes(); }
    const uint8_t * qk(size_t i, int kind) const { return qkeys.data() + (i * N_KEYS + kind) * key_bytes(); }
    // record i's key of one kind as floats, unit length
    std::vector<float> key_of(size_t i, int kind) const {
        const float * k = fkeys.data() + (i * N_KEYS + kind) * n_embd;
        return std::vector<float>(k, k + n_embd);
    }

    // opens (or starts) the store at path for this shape; a file of another shape (other model or layer split) or another layout is left alone and the store
    // starts empty in memory and is not written. Returns false then.
    bool open(const std::string & p, uint32_t e, uint32_t l, uint32_t x) {
        path = p; n_embd = e; n_layers = l; n_exp = x;
        qkeys.clear(); fkeys.clear(); qinv.clear(); counts.clear(); n_req.clear(); hit.clear(); src.clear(); last.clear();
        glob.assign(rec_cells(), 0);
        glob_req = 0;
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { return true; }   // a new store: the first add creates the file
        char m[8]; uint32_t h[4];
        const bool magic = fread(m, 1, 8, f) == 8 && memcmp(m, MAGIC, 8) == 0;
        if (!magic && memcmp(m, MAGIC, 7) == 0) {   // an older layout of this store: kept beside under its own name, the store starts anew (collecting goes on)
            fclose(f);
            std::string old = path + "." + std::string(m, 8);
            for (int n = 1; FILE * t = fopen(old.c_str(), "rb"); ++n) { fclose(t); old = path + "." + std::string(m, 8) + "." + std::to_string(n); }
            return std::rename(path.c_str(), old.c_str()) == 0 || (path.clear(), false);
        }
        bool ok = magic && fread(h, sizeof(uint32_t), 4, f) == 4 && h[0] == e && h[1] == l && h[2] == x;
        ok = ok && fread(&glob_req, sizeof(uint32_t), 1, f) == 1 && fread(glob.data(), sizeof(uint32_t), rec_cells(), f) == rec_cells();
        if (ok) {
            std::vector<uint8_t> r(rec_bytes());
            while (fread(r.data(), 1, r.size(), f) == r.size()) {
                const uint8_t * at = r.data();
                float hr; memcpy(&hr, at, sizeof hr); at += sizeof hr;   // a record: hit rate, source, requests, per key (1 / length, 4-bit key, float key), counts
                hit.push_back(hr);
                uint32_t so; memcpy(&so, at, sizeof so); at += sizeof so;
                src.push_back(so);
                uint32_t n; memcpy(&n, at, sizeof n); at += sizeof n;
                n_req.push_back(n);
                last.push_back(0);
                for (int k = 0; k < N_KEYS; ++k) {
                    float qi; memcpy(&qi, at, sizeof qi); at += sizeof qi;
                    qinv.push_back(qi);
                    qkeys.insert(qkeys.end(), at, at + key_bytes()); at += key_bytes();
                    const float * fk = (const float *) at;
                    fkeys.insert(fkeys.end(), fk, fk + n_embd); at += n_embd * sizeof(float);
                }
                const uint32_t * c = (const uint32_t *) at;
                counts.insert(counts.end(), c, c + rec_cells());
            }
        }
        fclose(f);
        if (!ok) { path.clear(); }   // never overwrite a file this store cannot read
        return ok;
    }

    // the cosine of record i's float key and a key (unit length, the lookup kind): exact, where the search reads the 4-bit copies
    float key_cos(size_t i, const float * key) const {
        const float * r = fkeys.data() + (i * N_KEYS + pool) * n_embd;
        double d = 0.0;
        for (uint32_t j = 0; j < n_embd; ++j) { d += (double) r[j] * key[j]; }
        return (float) d;
    }

    // the k records nearest to key (unit length, of the kind given) by cosine similarity, best first
    std::vector<std::pair<float, size_t>> nearest(const float * key, size_t k, int kind) const {
        std::vector<std::pair<float, size_t>> s(size());
        for (size_t i = 0; i < size(); ++i) { s[i] = { moe_topic_qdot(qk(i, kind), key, n_embd) * qinv[i * N_KEYS + kind], i }; }
        k = std::min(k, s.size());
        std::partial_sort(s.begin(), s.begin() + (ptrdiff_t) k, s.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        s.resize(k);
        return s;
    }

    // how alike two requests used the experts: per layer the share of its uses each expert got, the smaller of the two shares summed over the experts
    // (1 = the same experts used alike, 0 = none in common), averaged over the layers either request used
    float overlap(const uint32_t * a, const uint32_t * b) const {
        double sum = 0.0;
        size_t layers = 0;
        for (uint32_t l = 0; l < n_layers; ++l) {
            const uint32_t * x = a + (size_t) l * n_exp, * y = b + (size_t) l * n_exp;
            uint64_t tx = 0, ty = 0;
            for (uint32_t e = 0; e < n_exp; ++e) { tx += x[e]; ty += y[e]; }
            if (tx == 0 && ty == 0) { continue; }
            layers++;
            if (tx == 0 || ty == 0) { continue; }
            for (uint32_t e = 0; e < n_exp; ++e) { sum += std::min((double) x[e] / (double) tx, (double) y[e] / (double) ty); }
        }
        return layers > 0 ? (float) (sum / (double) layers) : 0.0f;
    }

    // a finished request (keys: N_KEYS x n_embd floats, each unit length, in KEY_ order), its hit rate and what picked the cache's experts (source). The
    // record nearest by the lookup key (4-bit, as the lookup reads it) is the request's topic when at least merge_cos alike (merge_cos > 1: never, every
    // request a record of its own):
    //   - the same prompt (float keys as alike as SAME_KEY): its map, hit rate and source are replaced when the request hit better, else nothing changes
    //     (user 2026-10-28: "store only if new heatmap better hitrate": repeated runs keep the best map);
    //   - another prompt of the topic: its counts are added and the experts below their layer's mean use dropped, so the record keeps the topic's hottest
    //     experts; the record keeps its first prompt's keys (pooled keys drift: a max-pooled key grows toward a vector near every prompt, and one record
    //     would take every topic). The 4-bit search rounds away the small values of a key, so prompts on one topic come out close: 100k Python requests
    //     make a few records, not 100k, and a query not asked before finds its topic's map (user 2026-10-28).
    // Else a new record (replacing the least recently used one past max_records). Returns the record's index.
    size_t add(const float * keys, const uint32_t * cnt, float merge_cos, float hit_rate = -1.0f, uint32_t source = 0) {
        add_glob(cnt);   // every stored request counts in the global map, whatever its record does
        const float * key = keys + (size_t) pool * n_embd;
        const auto nb = nearest(key, 4, pool);
        size_t at = size();
        last_cos = nb.empty() ? -1.0f : nb[0].first;
        last_same = false;
        for (const auto & c : nb) {
            if (merge_cos <= 1.0f && key_cos(c.second, key) >= SAME_KEY) { at = c.second; last_same = true; break; }
        }
        if (!last_same && !nb.empty() && nb[0].first >= merge_cos) { at = nb[0].second; }
        last_merged = at < size();
        last_best = last_merged ? overlap(counts.data() + at * rec_cells(), cnt) : 0.0f;
        last_kept = false;
        if (last_same) {
            if (hit_rate < 0.0f || (hit[at] >= 0.0f && hit_rate < hit[at])) {   // not measured, or not better: the record stays
                last_kept = true;
                return at;
            }
            std::copy(cnt, cnt + rec_cells(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
            hit[at] = hit_rate;
            src[at] = source;
            n_req[at]++;
        } else if (last_merged) {
            uint32_t * c = counts.data() + at * rec_cells();
            for (uint32_t l = 0; l < n_layers; ++l) {
                uint32_t * x = c + (size_t) l * n_exp;
                const uint32_t * y = cnt + (size_t) l * n_exp;
                uint64_t tot = 0;
                for (uint32_t e = 0; e < n_exp; ++e) {
                    x[e] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) x[e] + y[e]);
                    tot += x[e];
                }
                for (uint32_t e = 0; e < n_exp; ++e) { if ((uint64_t) x[e] * n_exp < tot) { x[e] = 0; } }
            }
            if (hit_rate > hit[at]) { hit[at] = hit_rate; src[at] = source; }   // the best hit rate a request of the topic reached
            n_req[at]++;
        } else {
            if (size() >= max_records && max_records > 0) {
                at = (size_t) (std::min_element(last.begin(), last.end()) - last.begin());
                std::copy(cnt, cnt + rec_cells(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
                n_req[at] = 1;
                hit[at] = hit_rate;
                src[at] = source;
            } else {
                at = size();
                qkeys.resize(qkeys.size() + N_KEYS * key_bytes());
                fkeys.resize(fkeys.size() + N_KEYS * n_embd);
                qinv.resize(qinv.size() + N_KEYS, 0.0f);
                counts.insert(counts.end(), cnt, cnt + rec_cells());
                n_req.push_back(1);
                hit.push_back(hit_rate);
                src.push_back(source);
                last.push_back(0);
            }
            for (int k = 0; k < N_KEYS; ++k) {
                std::copy(keys + (size_t) k * n_embd, keys + (size_t) (k + 1) * n_embd, fkeys.begin() + (ptrdiff_t) ((at * N_KEYS + k) * n_embd));
                moe_topic_quant(keys + (size_t) k * n_embd, n_embd, qk(at, k), &qinv[at * N_KEYS + k]);
            }
        }
        last[at] = ++clock;
        write(at);
        return at;
    }

    // the record at index i to the file (the header first when the file is new); a write error only costs the record
    void write(size_t i) const {
        FILE * f = open_rw();
        if (!f) { return; }
        if (fseek(f, (long) (head_bytes() + glob_bytes() + i * rec_bytes()), SEEK_SET) == 0) {
            fwrite(&hit[i], sizeof(float), 1, f);
            fwrite(&src[i], sizeof(uint32_t), 1, f);
            fwrite(&n_req[i], sizeof(uint32_t), 1, f);
            for (int k = 0; k < N_KEYS; ++k) {
                fwrite(&qinv[i * N_KEYS + k], sizeof(float), 1, f);
                fwrite(qk(i, k), 1, key_bytes(), f);
                fwrite(fkeys.data() + (i * N_KEYS + k) * n_embd, sizeof(float), n_embd, f);
            }
            fwrite(counts.data() + i * rec_cells(), sizeof(uint32_t), rec_cells(), f);
        }
        fclose(f);
    }

    // the file for writing; a new one gets the header and the global map first ("" path or an error: none)
    FILE * open_rw() const {
        if (path.empty()) { return nullptr; }
        FILE * f = fopen(path.c_str(), "r+b");
        if (!f) {
            f = fopen(path.c_str(), "w+b");
            if (!f) { return nullptr; }
            const uint32_t h[4] = { n_embd, n_layers, n_exp, 0 };
            fwrite(MAGIC, 1, 8, f);
            fwrite(h, sizeof(uint32_t), 4, f);
            fwrite(&glob_req, sizeof(uint32_t), 1, f);
            fwrite(glob.data(), sizeof(uint32_t), rec_cells(), f);
        }
        return f;
    }

    // a stored request's counts into the global map, and the map to the file
    void add_glob(const uint32_t * cnt) {
        for (size_t j = 0; j < rec_cells(); ++j) { glob[j] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) glob[j] + cnt[j]); }
        glob_req++;
        FILE * f = open_rw();
        if (!f) { return; }
        if (fseek(f, (long) head_bytes(), SEEK_SET) == 0) {
            fwrite(&glob_req, sizeof(uint32_t), 1, f);
            fwrite(glob.data(), sizeof(uint32_t), rec_cells(), f);
        }
        fclose(f);
    }

    // the merge of the nearest records the cache fills from (key: the lookup key, of kind pool): per layer, each record's counts as shares of that layer's
    // uses, weighted by its similarity (only records at least min_cos alike), scaled so the hottest expert of all layers is 1: one scale, since experts of
    // different layers compete for the blocks of a size class (a scale per layer rates every expert of a layer that spreads its uses near the top). Returns
    // the records used and the best similarity. The global map comes in weighted 1 - the best similarity used: a prompt stored before takes its topic's map
    // (which holds the experts every request needs too), a query far from every topic mostly the global map (last_wg; none stored: the global map alone).
    size_t merge(const float * key, size_t k, float min_cos, std::vector<float> & heat, float * best = nullptr) {
        heat.assign(rec_cells(), 0.0f);
        auto add_shares = [&](const uint32_t * c, float w) {   // per layer, each expert's share of the layer's uses, times w
            for (uint32_t l = 0; l < n_layers; ++l) {
                uint64_t tot = 0;
                for (uint32_t x = 0; x < n_exp; ++x) { tot += c[(size_t) l * n_exp + x]; }
                if (tot == 0) { continue; }
                for (uint32_t x = 0; x < n_exp; ++x) { heat[(size_t) l * n_exp + x] += w * (float) c[(size_t) l * n_exp + x] / (float) tot; }
            }
        };
        size_t used = 0;
        float  bused = 0.0f;
        const auto nb = nearest(key, k, pool);
        if (best) { *best = nb.empty() ? -1.0f : nb[0].first; }
        for (const auto & p : nb) {
            if (p.first < min_cos || p.first <= 0.0f) { break; }   // the weight is the cosine: a record pointing away adds nothing
            add_shares(counts.data() + p.second * rec_cells(), p.first);
            bused = std::max(bused, std::min(1.0f, p.first));
            last[p.second] = ++clock;
            used++;
        }
        last_wg = use_glob && glob_req > 0 ? 1.0f - bused : 0.0f;
        if (last_wg > 0.0f) { add_shares(glob.data(), last_wg); }
        const float mx = heat.empty() ? 0.0f : *std::max_element(heat.begin(), heat.end());
        if (mx > 0.0f) { for (float & h : heat) { h /= mx; } }
        return used;
    }
};
