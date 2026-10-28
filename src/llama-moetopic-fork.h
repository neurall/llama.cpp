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
// The store is one flat file per model (<state dir>/hotmap/<model>.hotmap), fixed-size records searched by brute force: 1024 records of a 5120-wide key take
// 2.2 ms (PC2), against seconds of expert uploads the answer saves (a vector index - HNSW as Chroma uses, or product quantization - pays only from about a
// million records). A request that used nearly the same experts as a stored one (overlap >= merge_overlap) is pooled into that record, keeping the hottest
// experts, instead of becoming a new one; merge_overlap > 1 keeps every request as its own record (the first tests: the store grows unmerged).
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
    static constexpr char MAGIC[8] = { 'M', 'O', 'E', 'T', 'O', 'P', 'C', '4' };
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
    std::vector<uint64_t> last;    // the request number that last used a record (memory only: picks the record to replace)
    uint64_t clock = 0;
    std::string path;
    std::vector<float> seen;       // memory only: per stored request the overlap of its nearest record (the values the automatic threshold splits)
    float last_best = 0.0f, last_thr = 0.0f;   // what the last add compared (for the log)
    bool  last_merged = false, last_kept = false;   // last_kept: merged topic, but the request hit worse than the record: nothing changed

    size_t size() const { return n_req.size(); }
    size_t rec_cells() const { return (size_t) n_layers * n_exp; }
    size_t key_bytes() const { return (n_embd + 1) / 2; }
    size_t rec_bytes() const { return sizeof(uint32_t) + sizeof(float) + N_KEYS * (sizeof(float) + key_bytes() + n_embd * sizeof(float)) + rec_cells() * sizeof(uint32_t); }
    static size_t head_bytes() { return sizeof(MAGIC) + 4 * sizeof(uint32_t); }
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
        qkeys.clear(); fkeys.clear(); qinv.clear(); counts.clear(); n_req.clear(); hit.clear(); last.clear(); seen.clear();
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { return true; }   // a new store: the first add creates the file
        char m[8]; uint32_t h[4];
        bool ok = fread(m, 1, 8, f) == 8 && memcmp(m, MAGIC, 8) == 0 && fread(h, sizeof(uint32_t), 4, f) == 4 && h[0] == e && h[1] == l && h[2] == x;
        if (ok) {
            std::vector<uint8_t> r(rec_bytes());
            while (fread(r.data(), 1, r.size(), f) == r.size()) {
                const uint8_t * at = r.data();
                uint32_t n; memcpy(&n, at, sizeof n); at += sizeof n;
                n_req.push_back(n);
                float hr; memcpy(&hr, at, sizeof hr); at += sizeof hr;
                hit.push_back(hr);
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

    // the overlap at which two requests are one topic, from the requests seen so far: the best overlap of each against the records before it falls in two
    // groups, requests on a topic already stored (high) and requests on a new topic (low); the threshold splits them where the two groups differ most
    // (the split that maximises the variance between them, midway between the neighbouring values; the values start empty with the server). Too high a
    // threshold stores a record per request, too low merges topics; with fewer than 6 values or no clear gap between the groups (under 0.15 apart) the
    // start value 0.90 stands.
    float auto_threshold() const {
        const float start = 0.90f;
        if (seen.size() < 6) { return start; }
        std::vector<float> v(seen);
        std::sort(v.begin(), v.end());
        double tot = 0.0;
        for (float x : v) { tot += x; }
        double lo = 0.0, bestvar = -1.0, thr = start, gap = 0.0;
        for (size_t k = 1; k < v.size(); ++k) {
            lo += v[k - 1];
            const double n0 = (double) k, n1 = (double) (v.size() - k), m0 = lo / n0, m1 = (tot - lo) / n1;
            const double var = n0 * n1 * (m1 - m0) * (m1 - m0);
            if (var > bestvar) { bestvar = var; thr = 0.5 * (v[k - 1] + v[k]); gap = m1 - m0; }
        }
        return gap < 0.15 ? start : std::min(0.98f, std::max(0.5f, (float) thr));
    }

    // a finished request (keys: N_KEYS x n_embd floats, each unit length, in KEY_ order): when one of the records nearest by cosine used the experts nearly as
    // it did (overlap >= merge_overlap, the best such record; a negative merge_overlap = automatic, auto_threshold), the two are one topic: each key is pooled
    // by its own rule (the mean key: a mean weighted by the requests each holds; the max key: per value the larger), the counts added, and the experts below
    // their layer's mean use dropped, so a record keeps only the hottest; the record is replaced in place. Else a new record (replacing the least recently
    // used one past max_records). Returns the record's index.
    size_t add(const float * keys, const uint32_t * cnt, float merge_overlap, float hit_rate = -1.0f) {
        size_t at = size();
        float  best = 0.0f;
        last_thr = merge_overlap >= 0.0f ? merge_overlap : auto_threshold();   // a negative value asks for the automatic threshold
        for (const auto & c : nearest(keys + (size_t) pool * n_embd, 4, pool)) {   // only the records nearest by cosine are compared by experts used
            const float o = overlap(counts.data() + c.second * rec_cells(), cnt);
            if (o > best) { best = o; at = c.second; }
        }
        last_best = best;
        if (size() > 0) {   // the last 256 values: the threshold follows the recent mix of requests and costs the same however long the server runs
            if (seen.size() >= 256) { seen.erase(seen.begin()); }
            seen.push_back(best);
        }
        if (best < last_thr) { at = size(); }
        last_merged = at < size();
        last_kept = false;
        if (at < size() && hit_rate >= 0.0f && hit[at] >= 0.0f && hit_rate < hit[at]) {   // store only a better hit rate: the record already serves this topic better
            last_kept = true;
            return at;
        }
        if (at < size()) {
            hit[at] = std::max(hit[at], hit_rate);
            const float w = (float) n_req[at];
            for (int k = 0; k < N_KEYS; ++k) {
                std::vector<float> r = key_of(at, k);
                const float * key = keys + (size_t) k * n_embd;
                for (uint32_t j = 0; j < n_embd; ++j) { r[j] = k == KEY_MAX ? std::max(r[j], key[j]) : r[j] * w + key[j]; }
                double nn = 0.0;
                for (uint32_t j = 0; j < n_embd; ++j) { nn += (double) r[j] * r[j]; }
                const float inv = nn > 0.0 ? (float) (1.0 / std::sqrt(nn)) : 1.0f;
                for (uint32_t j = 0; j < n_embd; ++j) { r[j] *= inv; }
                std::copy(r.begin(), r.end(), fkeys.begin() + (ptrdiff_t) ((at * N_KEYS + k) * n_embd));   // the pooled key as it is, unit length
                moe_topic_quant(r.data(), n_embd, qk(at, k), &qinv[at * N_KEYS + k]);
            }
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
            n_req[at]++;
        } else {
            if (size() >= max_records && max_records > 0) {
                at = (size_t) (std::min_element(last.begin(), last.end()) - last.begin());
                std::copy(cnt, cnt + rec_cells(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
                n_req[at] = 1;
                hit[at] = hit_rate;
            } else {
                at = size();
                qkeys.resize(qkeys.size() + N_KEYS * key_bytes());
                fkeys.resize(fkeys.size() + N_KEYS * n_embd);
                qinv.resize(qinv.size() + N_KEYS, 0.0f);
                counts.insert(counts.end(), cnt, cnt + rec_cells());
                n_req.push_back(1);
                hit.push_back(hit_rate);
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
        if (path.empty()) { return; }
        FILE * f = fopen(path.c_str(), "r+b");
        if (!f) {
            f = fopen(path.c_str(), "w+b");
            if (!f) { return; }
            const uint32_t h[4] = { n_embd, n_layers, n_exp, 0 };
            fwrite(MAGIC, 1, 8, f);
            fwrite(h, sizeof(uint32_t), 4, f);
        }
        if (fseek(f, (long) (head_bytes() + i * rec_bytes()), SEEK_SET) == 0) {
            fwrite(&n_req[i], sizeof(uint32_t), 1, f);
            fwrite(&hit[i], sizeof(float), 1, f);
            for (int k = 0; k < N_KEYS; ++k) {
                fwrite(&qinv[i * N_KEYS + k], sizeof(float), 1, f);
                fwrite(qk(i, k), 1, key_bytes(), f);
                fwrite(fkeys.data() + (i * N_KEYS + k) * n_embd, sizeof(float), n_embd, f);
            }
            fwrite(counts.data() + i * rec_cells(), sizeof(uint32_t), rec_cells(), f);
        }
        fclose(f);
    }

    // the merge of the nearest records the cache fills from (key: the lookup key, of kind pool): per layer, each record's counts as shares of that layer's
    // uses, weighted by its similarity (only records at least min_cos alike), scaled so the hottest expert of all layers is 1: one scale, since experts of
    // different layers compete for the blocks of a size class (a scale per layer rates every expert of a layer that spreads its uses near the top). Returns
    // the records used and the best similarity.
    size_t merge(const float * key, size_t k, float min_cos, std::vector<float> & heat, float * best = nullptr) {
        heat.assign(rec_cells(), 0.0f);
        size_t used = 0;
        const auto nb = nearest(key, k, pool);
        if (best) { *best = nb.empty() ? -1.0f : nb[0].first; }
        for (const auto & p : nb) {
            if (p.first < min_cos) { break; }
            const uint32_t * c = counts.data() + p.second * rec_cells();
            for (uint32_t l = 0; l < n_layers; ++l) {
                uint64_t tot = 0;
                for (uint32_t x = 0; x < n_exp; ++x) { tot += c[(size_t) l * n_exp + x]; }
                if (tot == 0) { continue; }
                for (uint32_t x = 0; x < n_exp; ++x) { heat[(size_t) l * n_exp + x] += p.first * (float) c[(size_t) l * n_exp + x] / (float) tot; }
            }
            last[p.second] = ++clock;
            used++;
        }
        const float mx = heat.empty() ? 0.0f : *std::max_element(heat.begin(), heat.end());
        if (mx > 0.0f) { for (float & h : heat) { h /= mx; } }
        return used;
    }
};
