// Hot expert maps per topic (task 158; user idea 2026-10-27). The one hot map in moe-state.ini sums every request the model ever served, so a prompt about one
// thing finds the experts of all the others cached: DeepSeek-V4.1 on the tetris test ran its prompt at 6.8 t/s from a state diluted by prose, JS and Python
// requests and at 18.9 t/s after a few tetris runs had made the state tetris-like again (same build, same day). Here every request leaves its own map.
//
// Key: what the model holds at the prompt's last token, the router input of every MoE layer and the final hidden state the logits are made from (user
// 2026-10-28: "that activation has nothing from template in it ... all prev prefill tokens are mixed in"; "store all layer acts ... calc 48 cos distances
// and avg them"). Two keys are compared row by row, each row only with the same row of the other key, and the cosines averaged (a value means something
// else in each layer). On 19 prompts of 7 topics (Qwen3.6, thinking off) that average followed how alike the experts of the answers were at 0.90; the
// pooled input embedding of the prompt's text, the key before, at 0.38. In memory the rows are 4 bits per value (0.901 against 0.903 with floats).
//
// Value: two maps per record (user 2026-10-28: "store 2 maps per emb ... if think tokens detected"): the answer's experts after </think> (or the whole
// answer when the request did not think) and the thinking phase's before it; each cut to what the cache holds (llama-moecache-fork.cpp, topic_hottest)
// and replaced on its own phase's better hit rate. A record is one prompt: the hash of its tokens finds it again (activations of different prompts can be
// alike to 0.99998).
//
// The store is one text file per model, <hot map dir>/<model>.js: plain JS, one assignment per line, the record's number in every name (user 2026-10-29:
// "i like the minimal a=b per line js code ... simpler writer parser than json, fully js compatible and trivial to break to smaller shared subset and
// compress"; a page loads it with <script> as it is, git diffs it by line):
//   hash5="a1b2c3d4e5f60718";  n5=2;  hit5=0.8730;  hitthink5=-1;  src5="ini";  time5="...";  prompt5="...";  thinking5="...";  answer5="...";
//   key5=[[row of the first MoE layer], ..., [final hidden state]];  val5=[[count,layer,expert], ...];  think5=[...];
// (each on its own line; the maps hottest first). Shared expert sets are meant to come as names of their own, written before the names that use them
// (val5=[[52,0,17],set1]; set1=[[...],set2];): a value may hold such a name, and this reader passes over it for now. A changed record is its lines again,
// the last assignment of a name wins (in a page and here), key5=null; drops record 5; opening the store rewrites the file with the last line of each
// name. The global map (every stored request's answer counts) is <model>.global.js (globreq=..., glob=[...]), rewritten whole. Records are searched by
// brute force: 1024 records of 41 rows of 2048 values take a few ms, against seconds of expert uploads.
// No ggml in here: tests/test-moetopic-fork.cpp drives it.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

// A row is kept in 4 bits per value (8x smaller than float): value = q * qinv with q in -7..7 and qinv = 1 / |q|, so the stored vector has unit length
// again. The step is 4 times the rms of a unit row (4 / sqrt(n)) / 7 and larger values are clipped: on random rows the cosine to the exact row stays
// above 0.97; on the router inputs (a few huge values) 0.96, and the averaged similarity follows the experts as well as the float rows do.
// Two values per byte (q + 8: the low nibble the even index), n rounded up to even.
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

// the dot of a float row with a stored row, times the stored row's qinv outside: 8 independent sums, so the compiler vectorizes it without -ffast-math
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

// what a request said and answered, the first few hundred characters of each (the record's line keeps them for the page)
struct moe_topic_text {
    std::string prompt, thinking, answer;
};

// a JS string literal of t (escapes: quote, backslash, control characters)
static inline std::string js_str(const std::string & t) {
    std::string o = "\"";
    for (unsigned char ch : t) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += (char) ch; }
        else if (ch == '\n') { o += "\\n"; }
        else if (ch == '\r') { o += "\\r"; }
        else if (ch == '\t') { o += "\\t"; }
        else if (ch < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); o += b; }
        else { o += (char) ch; }
    }
    return o + "\"";
}

// a value of a store line: a number, a string, null, a name (a set the value refers to), or an array (its numbers in nums, its arrays and names in arr)
struct moe_js {
    enum { NUL, NUM, STR, REF, ARR } t = NUL;
    double num = 0.0;
    std::string str;   // STR: the text; REF: the name
    std::vector<double> nums;
    std::vector<moe_js> arr;
};

static inline void moe_js_ws(const char *& p) { while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') { ++p; } }

// one value at p (p moves past it); false: not a value of the store's lines
static inline bool moe_js_value(const char *& p, moe_js & v) {
    moe_js_ws(p);
    if (*p == '"') {
        v.t = moe_js::STR;
        for (++p; *p && *p != '"'; ++p) {
            if (*p != '\\' || !p[1]) { v.str += *p; continue; }
            const char c = *++p;
            if (c == 'n') { v.str += '\n'; } else if (c == 'r') { v.str += '\r'; } else if (c == 't') { v.str += '\t'; }
            else if (c == 'u' && p[1] && p[2] && p[3] && p[4]) { v.str += (char) std::strtol(std::string(p + 1, 4).c_str(), nullptr, 16); p += 4; }
            else { v.str += c; }
        }
        if (*p != '"') { return false; }
        ++p;
        return true;
    }
    if (*p == '[') {
        v.t = moe_js::ARR;
        ++p;
        moe_js_ws(p);
        if (*p == ']') { ++p; return true; }
        for (;;) {
            moe_js_ws(p);
            if (*p == '[' || *p == '"' || std::isalpha((unsigned char) *p) || *p == '_') {
                v.arr.emplace_back();
                if (!moe_js_value(p, v.arr.back())) { return false; }
            } else {
                char * e = nullptr;
                const double d = std::strtod(p, &e);
                if (e == p) { return false; }
                v.nums.push_back(d);
                p = e;
            }
            moe_js_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; return true; }
            return false;
        }
    }
    if (std::isalpha((unsigned char) *p) || *p == '_') {
        const char * n0 = p;
        while (std::isalnum((unsigned char) *p) || *p == '_') { ++p; }
        v.str.assign(n0, p);
        v.t = v.str == "null" ? moe_js::NUL : moe_js::REF;
        return true;
    }
    char * e = nullptr;
    v.num = std::strtod(p, &e);
    if (e == p) { return false; }
    v.t = moe_js::NUM;
    p = e;
    return true;
}

// a line `name=value;`: its name and value; false: not such a line (a comment, an empty line)
static inline bool moe_js_assign(const std::string & line, std::string & name, moe_js & v) {
    const char * p = line.c_str();
    moe_js_ws(p);
    const char * n0 = p;
    while (std::isalnum((unsigned char) *p) || *p == '_') { ++p; }
    name.assign(n0, p);
    moe_js_ws(p);
    if (name.empty() || *p != '=') { return false; }
    ++p;
    return moe_js_value(p, v);
}

// a name split into its word and its record number ("hit12" -> "hit", 12); -1: no number
static inline std::pair<std::string, long long> moe_js_split(const std::string & name) {
    size_t i = name.size();
    while (i > 0 && std::isdigit((unsigned char) name[i - 1])) { --i; }
    return { name.substr(0, i), i < name.size() ? std::atoll(name.c_str() + i) : -1LL };
}

struct moe_topic_store {
    static constexpr char MAGIC[8] = { 'M', 'O', 'E', 'T', 'O', 'P', 'C', '9' };   // the binary layouts before were MOETOPC3..8 (deleted when found)
    static constexpr const char * FIELDS[12] = { "hash", "n", "hit", "hitthink", "src", "time", "prompt", "thinking", "answer", "key", "val", "think" };   // a record's lines, in order
    uint32_t n_embd = 0, n_layers = 0, n_exp = 0;
    uint32_t n_rows = 0;           // rows of a key (MoE layers and the final state): set by the first key stored or read
    size_t   max_records = 1024;   // the oldest-used record is replaced past this (0: no limit)
    std::vector<uint8_t>  qrows;   // records x n_rows x key_bytes(): the key rows in 4 bits per value (moe_topic_quant)
    std::vector<float>    qinv;    // records x n_rows: 1 / length of each quantized row
    std::vector<uint32_t> counts;  // records x n_layers x n_exp: uses of each expert in the answer (after </think>, or all of it without thinking)
    std::vector<uint32_t> think;   // records x n_layers x n_exp: uses of each expert while the request thought (all 0: it did not think)
    std::vector<uint32_t> n_req;   // requests that changed each record
    std::vector<float>    hit;     // the best answer hit rate (0..1) a request of this prompt reached; one that did worse does not change the map (<0: unknown)
    std::vector<float>    hit_think;   // the same for the thinking phase and its map
    std::vector<uint32_t> src;     // what steered the cache when that hit rate was measured: 1 = the hot maps (dynhot=full|read), 0 = the ini's hot experts
    std::vector<uint64_t> phash;   // a hash of the prompt's tokens (0: none): the same prompt is the same record
    std::vector<uint64_t> seq;     // the record's number in the file (the order records were made in)
    std::vector<moe_topic_text> text;
    std::vector<uint64_t> last;    // the request number that last used a record (memory only: picks the record to replace)
    uint64_t clock = 0, next_seq = 0;
    std::string path;              // <hot map dir>/<model>.js ("": nothing written)
    float last_cos = 0.0f, last_best = 0.0f;   // the last add: the nearest record's similarity, how alike the maps were (the log)
    // the model's global map: every stored request's answer counts added, none dropped. Its hottest are the experts every request needs whatever the
    // task - language, grammar, syntax (user 2026-10-28) - and it stands in where the nearest topic is far (merge). <model>.global.js
    std::vector<uint32_t> glob;
    uint32_t glob_req = 0;
    bool     use_glob = true;      // merge blends it in (--moe dynhot-global=0: the nearest topics only, for the A/B)
    float    last_wg  = 0.0f;      // the last merge: the global map's weight
    bool  last_merged = false, last_kept = false;   // last_merged: the request went to a stored record; last_kept: its prompt, but no better hit rate
    bool  last_same = false;                         // the last add was a prompt already stored (its hash)
    bool  same_pool = false;                         // the same prompt again: its counts added whatever its hit rate (--moe dynhot-same=pool), not the map
                                                     // replaced by a better run's (replace, the default)

    size_t size() const { return n_req.size(); }
    size_t rec_cells() const { return (size_t) n_layers * n_exp; }
    size_t key_bytes() const { return (n_embd + 1) / 2; }
    uint8_t *       qr(size_t i, uint32_t r)       { return qrows.data() + ((size_t) i * n_rows + r) * key_bytes(); }
    const uint8_t * qr(size_t i, uint32_t r) const { return qrows.data() + ((size_t) i * n_rows + r) * key_bytes(); }
    std::string glob_path() const { return path.size() > 3 ? path.substr(0, path.size() - 3) + ".global.js" : ""; }

    // a key's rows at unit length (the rows as the model gave them: n_rows x n_embd); a row of no length stays 0
    std::vector<float> unit_rows(const float * rows) const {
        std::vector<float> u(rows, rows + (size_t) n_rows * n_embd);
        for (uint32_t r = 0; r < n_rows; ++r) {
            float * x = u.data() + (size_t) r * n_embd;
            double nn = 0.0;
            for (uint32_t j = 0; j < n_embd; ++j) { nn += (double) x[j] * x[j]; }
            const float inv = nn > 0.0 ? (float) (1.0 / std::sqrt(nn)) : 0.0f;
            for (uint32_t j = 0; j < n_embd; ++j) { x[j] *= inv; }
        }
        return u;
    }

    // how alike record i and a key are (urows: n_rows unit rows): the cosines of the record's 4-bit rows with the key's rows, averaged
    float score(size_t i, const float * urows) const {
        double s = 0.0;
        for (uint32_t r = 0; r < n_rows; ++r) { s += moe_topic_qdot(qr(i, r), urows + (size_t) r * n_embd, n_embd) * qinv[i * n_rows + r]; }
        return n_rows ? (float) (s / n_rows) : 0.0f;
    }

    // the k records nearest to a key (urows: unit rows), best first
    std::vector<std::pair<float, size_t>> nearest(const float * urows, size_t k) const {
        std::vector<std::pair<float, size_t>> s(size());
        for (size_t i = 0; i < size(); ++i) { s[i] = { score(i, urows), i }; }
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

    // opens (or starts) the store at p (<dir>/<model>.js) for this shape: the last assignment of each name is read, and the file rewritten with those lines
    // only when it held more. A record of another shape (other model or layer split) makes the store refuse the file: empty in memory and nothing written
    // (returns false). The binary file of the layouts before (<dir>/<model>.hotmap) is deleted (user 2026-10-28: old maps are junk, no copies kept)
    bool open(const std::string & p, uint32_t e, uint32_t l, uint32_t x) {
        path = p; n_embd = e; n_layers = l; n_exp = x; n_rows = 0;
        qrows.clear(); qinv.clear(); counts.clear(); think.clear(); n_req.clear(); hit.clear(); hit_think.clear(); src.clear(); phash.clear(); seq.clear();
        text.clear(); last.clear();
        next_seq = 0;
        glob.assign(rec_cells(), 0);
        glob_req = 0;
        if (path.empty()) { return true; }
        if (path.size() > 3 && path.compare(path.size() - 3, 3, ".js") == 0) {
            const std::string old = path.substr(0, path.size() - 3) + ".hotmap";
            if (FILE * f = fopen(old.c_str(), "rb")) {
                char m[8];
                const bool ours = fread(m, 1, 8, f) == 8 && memcmp(m, MAGIC, 7) == 0;
                fclose(f);
                if (ours) { std::remove(old.c_str()); }
            }
        }
        std::map<long long, std::map<std::string, std::string>> lines;   // record number -> word -> its last line
        size_t n_lines = 0;
        for_lines(path, [&](const std::string & line) {
            std::string name;
            moe_js v;
            const size_t eq = line.find('=');
            if (eq == std::string::npos) { return; }
            name = line.substr(0, eq);
            while (!name.empty() && name.back() == ' ') { name.pop_back(); }
            const auto sp = moe_js_split(name);
            if (sp.second < 0) { return; }
            n_lines++;
            lines[sp.second][sp.first] = line;
        });
        size_t kept = 0;
        for (auto it = lines.begin(); it != lines.end(); ) {
            auto k = it->second.find("key");
            if (k == it->second.end() || k->second.find("=null") != std::string::npos) { it = lines.erase(it); continue; }   // dropped (or never whole)
            kept += it->second.size();
            if (!read_record(it->first, it->second)) {
                path.clear();   // another shape or a line this cannot read: the file is never overwritten
                return false;
            }
            ++it;
        }
        if (kept < n_lines) {   // older lines of a name, or dropped records: the file again with the last line of each name
            std::string o;
            for (const auto & rec : lines) {
                for (const char * f : FIELDS) { const auto w = rec.second.find(f); if (w != rec.second.end()) { o += w->second + "\n"; } }
            }
            put_file(path, o, false);
        }
        for_lines(glob_path(), [&](const std::string & line) {
            std::string name;
            moe_js v;
            if (!moe_js_assign(line, name, v)) { return; }
            if (name == "globreq" && v.t == moe_js::NUM) { glob_req = (uint32_t) v.num; }
            if (name == "glob") { map_from(v, glob.data()); }
        });
        return true;
    }

    // fn(line) for each line of a file (none when it cannot be read)
    template <typename F>
    static void for_lines(const std::string & fn_path, F fn) {
        FILE * f = fopen(fn_path.c_str(), "rb");
        if (!f) { return; }
        std::string all;
        char buf[1 << 16];
        for (size_t n; (n = fread(buf, 1, sizeof(buf), f)) > 0; ) { all.append(buf, n); }
        fclose(f);
        for (size_t i = 0; i < all.size(); ) {
            size_t end = all.find('\n', i);
            if (end == std::string::npos) { end = all.size(); }
            fn(all.substr(i, end - i));
            i = end + 1;
        }
    }

    // the counts of a map [[count,layer,expert], ...] into c (n_layers x n_exp; a set's name in it is passed over for now); false: an index outside this shape
    bool map_from(const moe_js & v, uint32_t * c) const {
        for (const auto & t : v.arr) {
            if (t.t == moe_js::REF) { continue; }
            if (t.nums.size() != 3) { return false; }
            const double cnt = t.nums[0], l = t.nums[1], e = t.nums[2];
            if (l < 0 || e < 0 || l >= n_layers || e >= n_exp) { return false; }
            c[(size_t) l * n_exp + (size_t) e] = (uint32_t) cnt;
        }
        return true;
    }

    // one record's lines (word -> line) into memory; false: another shape
    bool read_record(long long id, const std::map<std::string, std::string> & rec) {
        std::vector<uint32_t> c(rec_cells(), 0), t(rec_cells(), 0);
        std::vector<float> rows;
        uint32_t nr = 0;
        float h = -1.0f, ht = -1.0f;
        uint32_t req = 1, so = 0;
        uint64_t ph = 0;
        moe_topic_text tx;
        for (const auto & w : rec) {
            std::string name;
            moe_js v;
            if (!moe_js_assign(w.second, name, v)) { return false; }
            const std::string & word = w.first;
            if (word == "hash") { ph = std::strtoull(v.str.c_str(), nullptr, 16); }
            else if (word == "n") { req = (uint32_t) v.num; }
            else if (word == "hit") { h = (float) v.num; }
            else if (word == "hitthink") { ht = (float) v.num; }
            else if (word == "src") { so = v.str == "maps" ? 1u : 0u; }
            else if (word == "prompt") { tx.prompt = v.str; }
            else if (word == "thinking") { tx.thinking = v.str; }
            else if (word == "answer") { tx.answer = v.str; }
            else if (word == "val") { if (!map_from(v, c.data())) { return false; } }
            else if (word == "think") { if (!map_from(v, t.data())) { return false; } }
            else if (word == "key") {
                nr = (uint32_t) v.arr.size();
                for (const auto & r : v.arr) {
                    if (r.nums.size() != n_embd) { return false; }
                    for (double d : r.nums) { rows.push_back((float) d); }
                }
            }
        }
        if (nr == 0 || (n_rows != 0 && nr != n_rows)) { return false; }
        n_rows = nr;
        const std::vector<float> u = unit_rows(rows.data());
        const size_t i = size();
        qrows.resize(qrows.size() + (size_t) n_rows * key_bytes());
        qinv.resize(qinv.size() + n_rows, 0.0f);
        for (uint32_t r = 0; r < n_rows; ++r) { moe_topic_quant(u.data() + (size_t) r * n_embd, n_embd, qr(i, r), &qinv[i * n_rows + r]); }
        counts.insert(counts.end(), c.begin(), c.end());
        think.insert(think.end(), t.begin(), t.end());
        hit.push_back(h); hit_think.push_back(ht); src.push_back(so); n_req.push_back(req); phash.push_back(ph); seq.push_back((uint64_t) id);
        text.push_back(std::move(tx));
        last.push_back(0);
        next_seq = std::max(next_seq, (uint64_t) id + 1);
        return true;
    }

    // a finished request: rows (n_rows x n_embd, as the model gave them), its answer map cnt and hit rate, what picked the cache's experts (source), its
    // prompt hash, its thinking map and hit rate (nullptr / < 0: it did not think), what it said and answered (tx).
    //   - the same prompt (its hash): each map replaced on its own phase's better hit rate (user 2026-10-28: "store only if new heatmap better hitrate"),
    //     else nothing changes; with same_pool the counts are added instead;
    //   - else the nearest record when at least merge_cos alike (merge_cos > 1: never, the server's default since 2026-10-28: merging is offline):
    //     counts added and the experts below their layer's mean use dropped;
    //   - else a new record (replacing the least recently used one past max_records).
    // The record goes to the file as a new line. Returns its index; SIZE_MAX for a key of another row count.
    size_t add(const float * rows, uint32_t rows_n, const uint32_t * cnt, float merge_cos, float hit_rate = -1.0f, uint32_t source = 0,
               uint64_t prompt_hash = 0, const uint32_t * cnt_think = nullptr, float think_rate = -1.0f, const moe_topic_text * tx = nullptr) {
        if (n_rows == 0) { n_rows = rows_n; }
        if (rows_n != n_rows || rows_n == 0) { return SIZE_MAX; }
        add_glob(cnt);   // every stored request counts in the global map, whatever its record does
        const std::vector<float> u = unit_rows(rows);
        const auto nb = nearest(u.data(), 1);
        size_t at = size();
        last_cos = nb.empty() ? -1.0f : nb[0].first;
        last_same = false;
        for (size_t i = 0; prompt_hash != 0 && i < size(); ++i) {
            if (phash[i] == prompt_hash) { at = i; last_same = true; break; }
        }
        if (!last_same && !nb.empty() && nb[0].first >= merge_cos) { at = nb[0].second; }
        last_merged = at < size();
        last_best = last_merged ? overlap(counts.data() + at * rec_cells(), cnt) : 0.0f;
        last_kept = false;
        if (last_same && !same_pool) {   // each map replaced on its own phase's better hit rate
            const bool ans = hit_rate >= 0.0f && (hit[at] < 0.0f || hit_rate >= hit[at]);
            const bool thk = cnt_think != nullptr && think_rate >= 0.0f && (hit_think[at] < 0.0f || think_rate >= hit_think[at]);
            if (!ans && !thk) {   // not measured, or not better: the record stays
                last_kept = true;
                return at;
            }
            if (ans) {
                std::copy(cnt, cnt + rec_cells(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
                hit[at] = hit_rate;
                src[at] = source;
            }
            if (thk) {
                std::copy(cnt_think, cnt_think + rec_cells(), think.begin() + (ptrdiff_t) (at * rec_cells()));
                hit_think[at] = think_rate;
            }
            if (tx) { text[at] = *tx; }
            n_req[at]++;
        } else if (last_merged) {   // joined (same_pool, or a near prompt with merging on): counts added, the experts below their layer's mean dropped
            auto pool = [&](uint32_t * c, const uint32_t * y) {
                for (uint32_t l = 0; l < n_layers; ++l) {
                    uint32_t * xr = c + (size_t) l * n_exp;
                    const uint32_t * yr = y + (size_t) l * n_exp;
                    uint64_t tot = 0;
                    for (uint32_t e = 0; e < n_exp; ++e) {
                        xr[e] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) xr[e] + yr[e]);
                        tot += xr[e];
                    }
                    for (uint32_t e = 0; e < n_exp; ++e) { if ((uint64_t) xr[e] * n_exp < tot) { xr[e] = 0; } }
                }
            };
            pool(counts.data() + at * rec_cells(), cnt);
            if (hit_rate > hit[at]) { hit[at] = hit_rate; src[at] = source; }
            if (cnt_think != nullptr && think_rate >= 0.0f) {
                pool(think.data() + at * rec_cells(), cnt_think);
                hit_think[at] = std::max(hit_think[at], think_rate);
            }
            n_req[at]++;
        } else {
            if (size() >= max_records && max_records > 0) {
                at = (size_t) (std::min_element(last.begin(), last.end()) - last.begin());
                append("key" + std::to_string(seq[at]) + "=null;\n");   // its record goes from the file
            } else {
                at = size();
                qrows.resize(qrows.size() + (size_t) n_rows * key_bytes());
                qinv.resize(qinv.size() + n_rows, 0.0f);
                counts.resize(counts.size() + rec_cells(), 0);
                think.resize(think.size() + rec_cells(), 0);
                n_req.push_back(0); hit.push_back(-1.0f); hit_think.push_back(-1.0f); src.push_back(0); phash.push_back(0); seq.push_back(0);
                text.emplace_back(); last.push_back(0);
            }
            std::copy(cnt, cnt + rec_cells(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
            if (cnt_think != nullptr && think_rate >= 0.0f) {
                std::copy(cnt_think, cnt_think + rec_cells(), think.begin() + (ptrdiff_t) (at * rec_cells()));
            } else {
                std::fill(think.begin() + (ptrdiff_t) (at * rec_cells()), think.begin() + (ptrdiff_t) ((at + 1) * rec_cells()), 0u);
            }
            n_req[at] = 1;
            hit[at] = hit_rate;
            hit_think[at] = cnt_think != nullptr ? think_rate : -1.0f;
            src[at] = source;
            phash[at] = prompt_hash;
            seq[at] = next_seq++;
            text[at] = tx ? *tx : moe_topic_text();
            for (uint32_t r = 0; r < n_rows; ++r) { moe_topic_quant(u.data() + (size_t) r * n_embd, n_embd, qr(at, r), &qinv[at * n_rows + r]); }
        }
        last[at] = ++clock;
        append(line_of(at, rows));
        return at;
    }

    // a map as its value: [[count,layer,expert], ...], the hottest first
    std::string js_map(const uint32_t * c) const {
        std::vector<size_t> cells;
        for (size_t j = 0; j < rec_cells(); ++j) { if (c[j] > 0) { cells.push_back(j); } }
        std::stable_sort(cells.begin(), cells.end(), [&](size_t u, size_t v) { return c[u] > c[v]; });
        std::string o = "[";
        for (size_t k = 0; k < cells.size(); ++k) {
            o += (k ? ",[" : "[") + std::to_string(c[cells[k]]) + "," + std::to_string(cells[k] / n_exp) + "," + std::to_string(cells[k] % n_exp) + "]";
        }
        return o + "]";
    }

    // record i's lines (rows: its key as the model gave it)
    std::string line_of(size_t i, const float * rows) const {
        const std::string n = std::to_string(seq[i]);
        char b[64];
        const time_t now = time(nullptr);
        char when[32];
        strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&now));
        snprintf(b, sizeof(b), "%016llx", (unsigned long long) phash[i]);
        std::string o = "hash" + n + "=\"" + b + "\";\n";
        o += "n" + n + "=" + std::to_string(n_req[i]) + ";\n";
        snprintf(b, sizeof(b), "%.4f", hit[i]);
        o += "hit" + n + "=" + b + ";\n";
        snprintf(b, sizeof(b), "%.4f", hit_think[i]);
        o += "hitthink" + n + "=" + b + ";\n";
        o += "src" + n + "=\"" + (src[i] ? "maps" : "ini") + "\";\n";
        o += "time" + n + "=" + js_str(when) + ";\n";
        o += "prompt" + n + "=" + js_str(text[i].prompt) + ";\n";
        o += "thinking" + n + "=" + js_str(text[i].thinking) + ";\n";
        o += "answer" + n + "=" + js_str(text[i].answer) + ";\n";
        o += "key" + n + "=[";
        for (uint32_t r = 0; r < n_rows; ++r) {
            o += r ? ",[" : "[";
            for (uint32_t j = 0; j < n_embd; ++j) {
                snprintf(b, sizeof(b), j ? ",%.5g" : "%.5g", rows[(size_t) r * n_embd + j]);
                o += b;
            }
            o += "]";
        }
        o += "];\n";
        o += "val" + n + "=" + js_map(counts.data() + i * rec_cells()) + ";\n";
        o += "think" + n + "=" + js_map(think.data() + i * rec_cells()) + ";\n";
        return o;
    }

    // text at the end of the file (the directory made first); a write error only costs this line
    void append(const std::string & o) const {
        if (path.empty()) { return; }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        FILE * f = fopen(path.c_str(), "ab");
        if (!f) { return; }
        fwrite(o.data(), 1, o.size(), f);
        fclose(f);
    }

    // a file written whole beside the target and renamed over it, so a crash leaves the old file whole
    static bool put_file(const std::string & fn, const std::string & o, bool mkdir = true) {
        std::error_code ec;
        if (mkdir) { std::filesystem::create_directories(std::filesystem::path(fn).parent_path(), ec); }
        const std::string tmp = fn + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) { return false; }
        const bool ok = fwrite(o.data(), 1, o.size(), f) == o.size();
        fclose(f);
        if (ok) { std::filesystem::rename(tmp, fn, ec); }
        if (!ok || ec) { std::remove(tmp.c_str()); return false; }
        return true;
    }

    // a stored request's answer counts into the global map, and the map to <model>.global.js
    void add_glob(const uint32_t * cnt) {
        for (size_t j = 0; j < rec_cells(); ++j) { glob[j] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) glob[j] + cnt[j]); }
        glob_req++;
        if (path.empty()) { return; }
        put_file(glob_path(), "globreq=" + std::to_string(glob_req) + ";\nglob=" + js_map(glob.data()) + ";\n");
    }

    // the merge of the nearest records the cache fills from (urows: the key's unit rows): per layer, each record's counts as shares of that layer's uses,
    // weighted by its similarity (only records at least min_cos alike), scaled so the hottest expert of all layers is 1: one scale, since experts of
    // different layers compete for the blocks of a size class. Returns the records used and the best similarity. The global map comes in weighted 1 - the
    // best similarity used (last_wg; none stored: the global map alone). thinking: the records' thinking maps (no global map for those)
    size_t merge(const float * urows, size_t k, float min_cos, std::vector<float> & heat, float * best = nullptr, bool thinking = false) {
        heat.assign(rec_cells(), 0.0f);
        auto add_shares = [&](const uint32_t * c, float w) {   // per layer, each expert's share of the layer's uses, times w
            for (uint32_t l = 0; l < n_layers; ++l) {
                uint64_t tot = 0;
                for (uint32_t xi = 0; xi < n_exp; ++xi) { tot += c[(size_t) l * n_exp + xi]; }
                if (tot == 0) { continue; }
                for (uint32_t xi = 0; xi < n_exp; ++xi) { heat[(size_t) l * n_exp + xi] += w * (float) c[(size_t) l * n_exp + xi] / (float) tot; }
            }
        };
        size_t used = 0;
        float  bused = 0.0f;
        const auto nb = n_rows ? nearest(urows, k) : std::vector<std::pair<float, size_t>>();
        if (best) { *best = nb.empty() ? -1.0f : nb[0].first; }
        for (const auto & p : nb) {
            if (p.first < min_cos || p.first <= 0.0f) { break; }   // the weight is the similarity: a record pointing away adds nothing
            add_shares((thinking ? think : counts).data() + p.second * rec_cells(), p.first);
            bused = std::max(bused, std::min(1.0f, p.first));
            last[p.second] = ++clock;
            used++;
        }
        last_wg = use_glob && glob_req > 0 && !thinking ? 1.0f - bused : 0.0f;
        if (last_wg > 0.0f) { add_shares(glob.data(), last_wg); }
        const float mx = heat.empty() ? 0.0f : *std::max_element(heat.begin(), heat.end());
        if (mx > 0.0f) { for (float & h : heat) { h /= mx; } }
        return used;
    }
};
