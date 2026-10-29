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
// The store is a directory per model, <hot map dir>/<model>/, of plain JS that a page loads as it is and git diffs by line (user 2026-10-29: "make
// modelname subdir and store hotmoe sets per layer ... in model dir have file named requests and in it line per req in js format too
// k1=[emb1:hex,emb2:hex,etc];v1=[4,3,6,3,etc] ie key val pairs where val is just lineno in that layer file ... just append req. for same req again only
// if hitrate better"):
//   requests.js  one line per stored request, only appended (the same prompt again only with a better hit rate; the last line of a number wins):
//                k5=["<row of the first MoE layer, fp8 E4M3 in hex>", ..., "<final hidden state>"];v5=[17,3,...];vt5=[-1,...];h5=0.8860;ht5=-1.0000;
//                n5=1;src5="ini";hash5="...";time5="...";p5="<prompt>";th5="<thinking>";a5="<answer>";
//                v5 / vt5: per layer, the line of that layer's rows.js holding the answer's / the thinking's row (-1: none); m5 / mt5: per layer, this request's
//                own use of each expert of that row, in ascending expert order (m5=[[234,43,23,4],[17,5,2]]); k5=null; drops request 5.
//   <l>/         layer l's directory (moe_topic_layer), three files of one statement per line:
//     sets.js    s4=[3,34,6,7,s1];  a set: its expert numbers and the sets it holds; only appended, a set later split is redefined s4=[s12,s13];
//     rows.js    c7=[7,s4];         a request's map: sets and a few experts, by line; only appended
//     moes.js    m=[2443,0,17,...]; one line, number k the uses of expert k so far: the one file rewritten (the numbers keep rising)
//   the global map is the layers' moes.js together (the fallback where no stored request is near enough; its 8 hottest experts of every layer rank first).
// A map is the experts its request needed most, as many as the cache held (topic_hottest), with their uses. Records are searched by brute force: 1024
// records of 41 rows of 2048 values take a few ms, against seconds of expert uploads.
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

// a value of a store line: a number, a string, null, a name (a set the value refers to), an array (its numbers in nums, its arrays, objects and names
// in arr) or an object ({e15:52,e40:40}: its keys and numbers in obj)
struct moe_js {
    enum { NUL, NUM, STR, REF, ARR, OBJ } t = NUL;
    double num = 0.0;
    std::string str;   // STR: the text; REF: the name
    std::vector<double> nums;
    std::vector<moe_js> arr;
    std::vector<std::pair<std::string, double>> obj;
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
            if (*p == '[' || *p == '{' || *p == '"' || std::isalpha((unsigned char) *p) || *p == '_') {
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
    if (*p == '{') {   // an object of numbers: {e15:52,e40:40} (keys bare names or strings)
        v.t = moe_js::OBJ;
        ++p;
        for (;;) {
            moe_js_ws(p);
            if (*p == '}') { ++p; return true; }
            std::string key;
            if (*p == '"') {
                for (++p; *p && *p != '"'; ++p) { key += *p; }
                if (*p != '"') { return false; }
                ++p;
            } else {
                while (std::isalnum((unsigned char) *p) || *p == '_') { key += *p++; }
            }
            moe_js_ws(p);
            if (key.empty() || *p != ':') { return false; }
            ++p;
            moe_js_ws(p);
            char * e = nullptr;
            const double d = std::strtod(p, &e);
            if (e == p) { return false; }
            v.obj.emplace_back(std::move(key), d);
            p = e;
            moe_js_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; return true; }
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

// every `name=value;` of a line (a line may hold several): fn(name, value, its text); false: a line that is not only such statements
template <typename F>
static bool moe_js_statements(const std::string & line, F fn) {
    const char * p = line.c_str();
    for (;;) {
        moe_js_ws(p);
        if (!*p) { return true; }
        if (p[0] == '/' && p[1] == '/') { return true; }   // a comment to the line's end
        const char * n0 = p;
        const char * s0 = p;
        while (std::isalnum((unsigned char) *p) || *p == '_') { ++p; }
        const std::string name(n0, p);
        moe_js_ws(p);
        if (name.empty() || *p != '=') { return false; }
        ++p;
        moe_js v;
        if (!moe_js_value(p, v)) { return false; }
        moe_js_ws(p);
        if (*p == ';') { ++p; }
        fn(name, v, std::string(s0, p));
    }
}

// a name split into its word and its record number ("hit12" -> "hit", 12); -1: no number
static inline std::pair<std::string, long long> moe_js_split(const std::string & name) {
    size_t i = name.size();
    while (i > 0 && std::isdigit((unsigned char) name[i - 1])) { --i; }
    return { name.substr(0, i), i < name.size() ? std::atoll(name.c_str() + i) : -1LL };
}

// fn(line) for each line of a file (none when it cannot be read)
template <typename F>
static void moe_topic_store_lines(const std::string & file, F fn) {
    FILE * f = fopen(file.c_str(), "rb");
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

// a float as fp8 E4M3 (bias 7, largest 448, no infinity; the nearest of the 127 codes, larger values clipped) and back: the key rows of requests.js are
// fp8 in hex, 2 digits a value (user 2026-10-29: "fp8 is fine"; the 4-bit rows the lookup reads ranked prompts as well as floats did)
static inline float moe_e2f(uint8_t v) {
    const int e = (v >> 3) & 15, m = v & 7;
    const float x = e == 0 ? (float) m / 8.0f * 0.015625f : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return (v & 0x80) ? -x : x;
}

static inline uint8_t moe_f2e(float f) {
    const float a = std::fabs(f);
    uint8_t lo = 0, hi = 126;   // the positive codes 0..126 grow with their value (127 is NaN)
    if (!(a > 0.0f)) { return 0; }
    while (lo < hi) {   // the first code at least a, then the nearer of it and the one below
        const uint8_t mid = (uint8_t) ((lo + hi) / 2);
        if (moe_e2f(mid) < a) { lo = (uint8_t) (mid + 1); } else { hi = mid; }
    }
    if (lo > 0 && a - moe_e2f((uint8_t) (lo - 1)) <= moe_e2f(lo) - a) { lo--; }
    return (uint8_t) (lo | (f < 0.0f ? 0x80 : 0));
}

// One layer's file of a store, <dir>/<layer>.js, plain JS that a page loads in order (user 2026-10-29: "store in js compat format so line 0 is
// s1={e1:2423,e3:353};s2={e5:34534} and next line c1=[{e7:323},s1,s2] next line c2 etc"), one statement per line, its name's number the line it was
// first written on, so a request names its row by line number ("val is just lineno in that layer file"):
//   s4={e15:52,e40:40};    a leaf set: experts with their uses, the hottest first
//   c7=[{e7:323},s4,s9];   a row (combo): the sets wholly inside it, and its few own experts inline
//   s4=[s12,s13];          on a later line: set 4 split in two (its shared part and its rest), redefined below them as their union
// Lines are only appended ("line appended so the old linenos dont change"), every name defined above its use, the last definition of a name wins (in a
// page and here). Every row written is decomposed against the sets there ("all writes to layer files keep decomposing shared sets to smaller ones and
// that sets are on top"): the sets wholly inside it are referenced, the largest first; a leaf it shares MIN_SET or more experts with is split into the
// shared part and the rest (two new leaves, the leaf redefined as their union) and the shared part referenced; what is left becomes a leaf of its own when
// it has MIN_SET experts (later rows may share it), else stays inline (too few to ever be shared). So each layer's file grows the hierarchy of the expert
// sets its prompts share: over long answers the sets are histogram lines that stop growing in number soon, and a rare prompt is a short row of names; the
// early layers, which depend least on the topic (2026-10-29: code vs prose shared 0.34-0.46 of layers 0-3, 0.10-0.20 deeper), end up as a few shared
// sets. Every expert of a row is in exactly one leaf of its tree or inline, so its uses are defined: a split keeps the leaf's.
struct moe_topic_layer {
    static constexpr uint32_t MIN_SET = 8;   // a shared part smaller than this stays where it is (a set of its own costs more than it saves)
    struct node {
        std::vector<int32_t>  refs;   // the sets it is made of (a split set: its two parts; a row: the sets wholly inside it), by line number in sets.js
        std::vector<uint32_t> own;    // its own experts, the hottest first (a leaf: all of them; a row: the inline ones)
        std::vector<uint64_t> mask;   // every expert it holds (refs expanded)
        uint32_t size = 0;
    };
    uint32_t n_exp = 0;
    std::vector<node> nodes;      // sets.js by line number: the set first written there (its last definition); a line redefining a set holds nothing
    std::vector<char> kind;       // by line number in sets.js: 's' a set, 'r' a set's redefinition
    std::vector<node> rows;       // rows.js by line number: each request's map as sets and a few experts
    std::vector<uint32_t> cnt;    // moes.js: per expert the uses so far (n_exp numbers, one a line); the only file of the three that is rewritten
    std::string path;             // the layer's directory: "": nothing written

    size_t words() const { return (n_exp + 63) / 64; }
    static uint32_t popc(const std::vector<uint64_t> & m) { uint32_t n = 0; for (uint64_t w : m) { n += (uint32_t) __builtin_popcountll(w); } return n; }
    std::string sets_path() const { return path + "/sets.js"; }
    std::string rows_path() const { return path + "/rows.js"; }
    std::string moes_path() const { return path + "/moes.js"; }

    // the statement of node x under the name kind + n: a list of its expert numbers and the sets it refers to, s2=[3,34,6,7,s1];
    std::string text_of(char k, int32_t n, const node & x) const {
        std::string o = std::string(1, k) + std::to_string(n) + "=[";
        for (size_t i = 0; i < x.own.size(); ++i) { o += (i ? "," : "") + std::to_string(x.own[i]); }
        for (size_t i = 0; i < x.refs.size(); ++i) { o += (i || !x.own.empty() ? ",s" : "s") + std::to_string(x.refs[i]); }
        return o + "];\n";
    }

    void fill_mask(node & x) const {
        x.mask.assign(words(), 0);
        for (int32_t r : x.refs) { for (size_t w = 0; w < words(); ++w) { x.mask[w] |= nodes[(size_t) r].mask[w]; } }
        for (uint32_t e : x.own) { x.mask[e / 64] |= 1ull << (e % 64); }
        x.size = popc(x.mask);
    }

    // the experts of row r in ascending order: what a request's counters in requests.js are listed by (a later split of a set moves no expert out of a row)
    std::vector<uint32_t> row_experts(int32_t r) const {
        std::vector<uint32_t> o;
        const node & x = rows[(size_t) r];
        for (uint32_t e = 0; e < n_exp; ++e) { if (x.mask[e / 64] >> (e % 64) & 1) { o.push_back(e); } }
        return o;
    }

    // the experts of mask as own experts of x, the hottest (by use) first
    void own_from(node & x, const std::vector<uint64_t> & mask, const uint32_t * use) const {
        for (uint32_t e = 0; e < n_exp; ++e) { if (mask[e / 64] >> (e % 64) & 1) { x.own.push_back(e); } }
        std::stable_sort(x.own.begin(), x.own.end(), [&](uint32_t a, uint32_t b) { return use[a] > use[b]; });
    }

    // set x on the next line of sets.js; its statement to out. Returns its line number
    int32_t put_set(node x, std::string & out) {
        fill_mask(x);
        const int32_t n = (int32_t) nodes.size();
        out += text_of('s', n, x);
        nodes.push_back(std::move(x));
        kind.push_back('s');
        return n;
    }

    // a new row for the experts with uses in use (n_exp), decomposed against the sets there; the sets made or split for it appended to sets_out, its
    // statement to rows_out, and its uses added to the counts of moes.js (the caller writes that file, see moes_text). Returns the row's line number
    int32_t write_row(const uint32_t * use, std::string & sets_out, std::string & rows_out) {
        if (cnt.size() != n_exp) { cnt.assign(n_exp, 0); }
        for (uint32_t e = 0; e < n_exp; ++e) { cnt[e] += use[e]; }
        std::vector<uint64_t> target(words(), 0);
        for (uint32_t e = 0; e < n_exp; ++e) { if (use[e]) { target[e / 64] |= 1ull << (e % 64); } }
        node row;
        for (;;) {
            if (popc(target) < MIN_SET) { break; }
            int32_t best = -1;
            uint32_t bsz = 0;
            for (size_t n = 0; n < nodes.size(); ++n) {   // the largest set wholly inside what is left
                if (kind[n] != 's') { continue; }
                const node & x = nodes[n];
                if (x.size < MIN_SET || x.size <= bsz) { continue; }
                bool inside = true;
                for (size_t w = 0; w < words() && inside; ++w) { inside = (x.mask[w] & ~target[w]) == 0; }
                if (inside) { best = (int32_t) n; bsz = x.size; }
            }
            if (best < 0) {   // none: the leaf sharing most with it is split, its shared part becomes a leaf of its own
                uint32_t bi = 0;
                for (size_t n = 0; n < nodes.size(); ++n) {
                    if (kind[n] != 's' || !nodes[n].refs.empty()) { continue; }
                    uint32_t i = 0;
                    for (size_t w = 0; w < words(); ++w) { i += (uint32_t) __builtin_popcountll(nodes[n].mask[w] & target[w]); }
                    if (i >= MIN_SET && i < nodes[n].size && i > bi) { best = (int32_t) n; bi = i; }
                }
                if (best < 0) { break; }
                std::vector<uint64_t> shared(words()), rest(words());
                for (size_t w = 0; w < words(); ++w) { shared[w] = nodes[(size_t) best].mask[w] & target[w]; rest[w] = nodes[(size_t) best].mask[w] & ~target[w]; }
                node a, b, un;
                own_from(a, shared, cnt.data());
                own_from(b, rest, cnt.data());
                const int32_t ns = put_set(std::move(a), sets_out);   // both parts above the leaf's new definition
                const int32_t nr = put_set(std::move(b), sets_out);
                un.refs = { ns, nr };
                fill_mask(un);
                sets_out += text_of('s', best, un);   // the redefinition takes the next line; the set keeps its name (its line number)
                nodes[(size_t) best] = std::move(un);
                nodes.emplace_back();
                kind.push_back('r');
                best = ns;
            }
            row.refs.push_back(best);
            for (size_t w = 0; w < words(); ++w) { target[w] &= ~nodes[(size_t) best].mask[w]; }
        }
        if (popc(target) >= MIN_SET) {   // enough left over to be shared later: a leaf of its own
            node leaf;
            own_from(leaf, target, use);
            row.refs.push_back(put_set(std::move(leaf), sets_out));
        } else {
            own_from(row, target, use);
        }
        fill_mask(row);
        const int32_t r = (int32_t) rows.size();
        rows_out += text_of('c', r, row);
        rows.push_back(std::move(row));
        return r;
    }

    // moes.js: one line, m=[2443,0,17,...]; number k the uses of expert k so far
    std::string moes_text() const {
        std::string o = "m=[";
        for (uint32_t e = 0; e < n_exp; ++e) { o += (e ? "," : "") + std::to_string(e < cnt.size() ? cnt[e] : 0u); }
        return o + "];\n";
    }

    // the three files read back (missing: an empty layer); false: a statement this cannot read, a name not defined above its use, or a number that is not
    // the line it stands on (a new set or row) or an earlier set (a redefinition), or a counts file of another width
    bool load(const std::string & dir, uint32_t x) {
        n_exp = x;
        path = dir;
        nodes.clear();
        kind.clear();
        rows.clear();
        cnt.assign(x, 0);
        bool ok = true;
        auto mk = [&](const moe_js & v, node & out) {   // [3,34,6,7,s1]: expert numbers and the sets (lines of sets.js) it refers to
            if (v.t != moe_js::ARR) { ok = false; return; }
            for (double d : v.nums) {
                if (d < 0 || d >= (double) x) { ok = false; return; }
                out.own.push_back((uint32_t) d);
            }
            for (const auto & el : v.arr) {
                const long long r = el.t == moe_js::REF && el.str.size() > 1 && el.str[0] == 's' ? std::atoll(el.str.c_str() + 1) : -1;
                if (r < 0 || r >= (long long) nodes.size() || kind[(size_t) r] != 's') { ok = false; return; }
                out.refs.push_back((int32_t) r);
            }
        };
        moe_topic_store_lines(sets_path(), [&](const std::string & line) {
            if (!ok) { return; }
            ok = moe_js_statements(line, [&](const std::string & name, const moe_js & v, const std::string &) {
                if (!ok || name.size() < 2 || name[0] != 's') { ok = false; return; }
                const long long n = std::atoll(name.c_str() + 1);
                node nd;
                mk(v, nd);
                if (!ok) { return; }
                fill_mask(nd);
                if (n == (long long) nodes.size()) {   // a new set on its own line
                    nodes.push_back(std::move(nd));
                    kind.push_back('s');
                } else if (n >= 0 && n < (long long) nodes.size() && kind[(size_t) n] == 's') {   // a set redefined on this line
                    nodes[(size_t) n] = std::move(nd);
                    nodes.emplace_back();
                    kind.push_back('r');
                } else {
                    ok = false;
                }
            }) && ok;
        });
        moe_topic_store_lines(rows_path(), [&](const std::string & line) {
            if (!ok) { return; }
            ok = moe_js_statements(line, [&](const std::string & name, const moe_js & v, const std::string &) {
                if (!ok || name.size() < 2 || name[0] != 'c' || std::atoll(name.c_str() + 1) != (long long) rows.size()) { ok = false; return; }
                node nd;
                mk(v, nd);
                if (!ok) { return; }
                fill_mask(nd);
                rows.push_back(std::move(nd));
            }) && ok;
        });
        moe_topic_store_lines(moes_path(), [&](const std::string & line) {
            if (!ok) { return; }
            ok = moe_js_statements(line, [&](const std::string & name, const moe_js & v, const std::string &) {
                if (!ok || name != "m" || v.t != moe_js::ARR || !v.arr.empty() || v.nums.size() != x) { ok = false; return; }
                for (uint32_t e = 0; e < x; ++e) { cnt[e] = (uint32_t) v.nums[e]; }
            }) && ok;
        });
        return ok;
    }
};

struct moe_topic_store {
    static constexpr char MAGIC[8] = { 'M', 'O', 'E', 'T', 'O', 'P', 'C', '9' };   // the binary layouts before were MOETOPC3..8 (deleted when found)
    uint32_t n_embd = 0, n_layers = 0, n_exp = 0;
    uint32_t n_rows = 0;           // rows of a key (MoE layers and the final state): set by the first key stored or read
    size_t   max_records = 1024;   // the oldest-used record is replaced past this (0: no limit)
    std::vector<uint8_t>  qrows;   // records x n_rows x key_bytes(): the key rows in 4 bits per value (moe_topic_quant)
    std::vector<float>    qinv;    // records x n_rows: 1 / length of each quantized row
    std::vector<uint32_t> counts;  // records x n_layers x n_exp: uses of each expert in the answer's map (after </think>, or all of it without thinking)
    std::vector<uint32_t> think;   // records x n_layers x n_exp: uses in the thinking phase's map (all 0: it did not think)
    std::vector<int32_t>  rows_ln, trows_ln;   // records x n_layers: the combo of each layer's file holding the answer's / the thinking's map (-1: none)
    std::vector<moe_topic_layer> lay;          // per layer: its file of sets
    std::vector<uint32_t> n_req;   // requests that changed each record
    std::vector<float>    hit;     // the best answer hit rate (0..1) a request of this prompt reached; one that did worse does not change the map (<0: unknown)
    std::vector<float>    hit_think;   // the same for the thinking phase and its map
    std::vector<uint32_t> src;     // what steered the cache when that hit rate was measured: 1 = the hot maps (dynhot=full|read), 0 = the ini's hot experts
    std::vector<uint64_t> phash;   // a hash of the prompt's tokens (0: none): the same prompt is the same record
    std::vector<uint64_t> seq;     // the record's number in keys.js (the order records were made in)
    std::vector<moe_topic_text> text;
    std::vector<uint64_t> last;    // the request number that last used a record (memory only: picks the record to replace)
    uint64_t clock = 0, next_seq = 0;
    std::string path;              // the model's directory, <hot map dir>/<model> ("": nothing written)
    float last_cos = 0.0f, last_best = 0.0f;   // the last add: the nearest record's similarity, how alike the maps were (the log)
    // the model's global map: each layer's moes.js (the uses of every expert in every map written, none dropped). Its hottest are the experts every request
    // needs whatever the task - language, grammar, syntax (user 2026-10-28) - and it stands in where the nearest topic is far (merge): "using global hot moes
    // from per layer moes, not from ini" (user 2026-10-29)
    std::vector<uint32_t> glob_map() const {
        std::vector<uint32_t> g(rec_cells(), 0);
        for (uint32_t li = 0; li < n_layers && li < lay.size(); ++li) {
            for (uint32_t e = 0; e < n_exp && e < lay[li].cnt.size(); ++e) { g[(size_t) li * n_exp + e] = lay[li].cnt[e]; }
        }
        return g;
    }
    uint32_t glob_requests() const { return lay.empty() ? 0u : (uint32_t) lay[0].rows.size(); }   // the maps written (the layers hold one row each per map)
    float    near_cos = 0.80f;     // the best averaged cosine at which the nearest records are used as they are; under it each layer is looked up on its own row, and a
                                   // layer whose best row cosine is under it too takes the global map alone
    uint32_t glob_floor = 8;       // the global map's hottest experts of every layer that always rank above the rest (a layer is never left with fewer)
    bool     use_glob = true;      // merge blends it in (--moe dynhot-global=0: the nearest topics only, for the A/B)
    float    last_wg  = 0.0f;      // the last merge: the global map's weight, averaged over the layers (each layer has its own)
    bool  last_merged = false, last_kept = false;   // last_merged: the request went to a stored record; last_kept: its prompt, but no better hit rate
    bool  last_same = false;                         // the last add was a prompt already stored (its hash)
    // a request whose key is at least this alike to a stored one's (the averaged 4-bit row cosine) is that request again: not appended, its maps replaced
    // only on a better hit rate, as for the same prompt hash (user 2026-10-29: "append only if cos dist too far from all existing"). A prompt scores about
    // 0.96 against its own record (4-bit rows), distinct prompts stayed below 0.92 (95th percentile 0.905 over 19 prompts of 7 topics); > 1: off
    float same_cos = 0.95f;
    bool  same_pool = false;                         // the same prompt again: its map added whatever its hit rate (--moe dynhot-same=pool), not
                                                     // replaced by a better run's (replace, the default)

    size_t size() const { return n_req.size(); }
    size_t rec_cells() const { return (size_t) n_layers * n_exp; }
    size_t key_bytes() const { return (n_embd + 1) / 2; }
    uint8_t *       qr(size_t i, uint32_t r)       { return qrows.data() + ((size_t) i * n_rows + r) * key_bytes(); }
    const uint8_t * qr(size_t i, uint32_t r) const { return qrows.data() + ((size_t) i * n_rows + r) * key_bytes(); }
    std::string req_path() const { return path + "/requests.js"; }
    std::string layer_path(uint32_t l) const { return path + "/" + std::to_string(l); }   // the layer's directory

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

    // how alike two maps are: per layer the share of its uses each expert got, the smaller of the two shares summed over the experts (1 = the same
    // experts used alike, 0 = none in common), averaged over the layers either map used
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

    // opens (or starts) the store in directory p (<dir>/<model>) for this shape: every layer's file of sets and the last assignment of each name of
    // keys.js; keys.js is rewritten with those lines only when it held more (the layer files are never rewritten: their line numbers are what the
    // records keep). A record of another shape (other model or layer split) makes the store refuse the directory: empty in memory and nothing written
    // (returns false). The stores of the layouts before (<dir>/<model>.hotmap, .js, .global.js) are deleted (user 2026-10-28: old maps are junk)
    bool open(const std::string & p, uint32_t e, uint32_t l, uint32_t x) {
        path = p; n_embd = e; n_layers = l; n_exp = x; n_rows = 0;
        qrows.clear(); qinv.clear(); counts.clear(); think.clear(); rows_ln.clear(); trows_ln.clear(); n_req.clear(); hit.clear(); hit_think.clear();
        src.clear(); phash.clear(); seq.clear(); text.clear(); last.clear();
        next_seq = 0;
        lay.assign(n_layers, moe_topic_layer());
        for (auto & ly : lay) { ly.n_exp = n_exp; }
        if (path.empty()) { return true; }
        for (const char * ext : { ".js", ".global.js", ".hotmap" }) {   // the single-file stores before this layout
            const std::string old = path + ext;
            if (FILE * f = fopen(old.c_str(), "rb")) {
                char m[8];
                const bool ours = std::string(ext) != ".hotmap" || (fread(m, 1, 8, f) == 8 && memcmp(m, MAGIC, 7) == 0);
                fclose(f);
                if (ours) { std::remove(old.c_str()); }
            }
        }
        {   // a directory of a layout before (keys.js; or the layers as flat <l>.js files): its files and layer directories go
            std::error_code ec;
            if (std::filesystem::exists(path + "/keys.js", ec) || std::filesystem::exists(path + "/0.js", ec)) {
                for (const auto & de : std::filesystem::directory_iterator(path, ec)) {
                    if (de.is_directory(ec)) { std::filesystem::remove_all(de.path(), ec); }
                    else if (de.path().extension() == ".js") { std::filesystem::remove(de.path(), ec); }
                }
            }
        }
        for (uint32_t li = 0; li < n_layers; ++li) {
            if (!lay[li].load(layer_path(li), n_exp)) { path.clear(); return false; }
                    }
        std::map<long long, std::map<std::string, moe_js>> reqs;   // request number -> word -> its last value (a later line wins)
        bool ok = true;
        moe_topic_store_lines(req_path(), [&](const std::string & line) {
            ok = moe_js_statements(line, [&](const std::string & name, const moe_js & v, const std::string &) {
                const auto sp = moe_js_split(name);
                if (sp.second >= 0) { reqs[sp.second][sp.first] = v; }
            }) && ok;
        });
        if (!ok) { path.clear(); return false; }   // a line this cannot read: requests.js is never written over
        for (const auto & rq : reqs) {
            const auto k = rq.second.find("k");
            if (k == rq.second.end() || k->second.t == moe_js::NUL) { continue; }   // dropped (or never whole)
            if (!read_record(rq.first, rq.second)) { path.clear(); return false; }
        }
        return true;
    }

    // one request's values (word -> value) into memory: its maps are the rows v / t name in each layer's file; false: another shape
    bool read_record(long long id, const std::map<std::string, moe_js> & rq) {
        std::vector<uint32_t> c(rec_cells(), 0), t(rec_cells(), 0);
        std::vector<int32_t> rl(n_layers, -1), tl(n_layers, -1);
        std::vector<float> rows;
        uint32_t nr = 0;
        float h = -1.0f, ht = -1.0f;
        uint32_t req = 1, so = 0;
        uint64_t ph = 0;
        moe_topic_text tx;
        for (const auto & w : rq) {
            const std::string & word = w.first;
            const moe_js & v = w.second;
            if (word == "hash") { ph = std::strtoull(v.str.c_str(), nullptr, 16); }
            else if (word == "n") { req = (uint32_t) v.num; }
            else if (word == "h") { h = (float) v.num; }
            else if (word == "ht") { ht = (float) v.num; }
            else if (word == "src") { so = v.str == "maps" ? 1u : 0u; }
            else if (word == "p") { tx.prompt = v.str; }
            else if (word == "th") { tx.thinking = v.str; }
            else if (word == "a") { tx.answer = v.str; }
            else if (word == "v" || word == "vt") {
                if (v.nums.size() != n_layers) { return false; }
                std::vector<int32_t> & dst = word == "v" ? rl : tl;
                for (uint32_t li = 0; li < n_layers; ++li) {
                    const long long ln = (long long) v.nums[li];
                    if (ln < 0) { continue; }
                    if (ln >= (long long) lay[li].rows.size()) { return false; }
                    dst[li] = (int32_t) ln;
                }
            } else if (word == "k") {   // one hex string per row: fp8, 2 digits a value
                nr = (uint32_t) v.arr.size();
                for (const auto & r : v.arr) {
                    if (r.t != moe_js::STR || r.str.size() != (size_t) n_embd * 2) { return false; }
                    for (uint32_t j = 0; j < n_embd; ++j) { rows.push_back(moe_e2f((uint8_t) std::strtoul(r.str.substr((size_t) j * 2, 2).c_str(), nullptr, 16))); }
                }
            }
        }
        if (nr == 0 || (n_rows != 0 && nr != n_rows)) { return false; }
        for (int m = 0; m < 2; ++m) {   // the maps: the experts of the row each layer names, with this request's own counters (m / mt; one use each when a line has none)
            const auto mv = rq.find(m == 0 ? "m" : "mt");
            const std::vector<int32_t> & ln = m == 0 ? rl : tl;
            for (uint32_t li = 0; li < n_layers; ++li) {
                if (ln[li] < 0) { continue; }
                const std::vector<uint32_t> ex = lay[li].row_experts(ln[li]);
                const moe_js * ct = mv != rq.end() && mv->second.arr.size() == n_layers ? &mv->second.arr[li] : nullptr;
                if (ct && ct->nums.size() != ex.size()) { return false; }
                for (size_t j = 0; j < ex.size(); ++j) { (m == 0 ? c : t)[(size_t) li * n_exp + ex[j]] = ct ? (uint32_t) ct->nums[j] : 1u; }
            }
        }
        n_rows = nr;
        const std::vector<float> u = unit_rows(rows.data());
        const size_t i = size();
        qrows.resize(qrows.size() + (size_t) n_rows * key_bytes());
        qinv.resize(qinv.size() + n_rows, 0.0f);
        for (uint32_t r = 0; r < n_rows; ++r) { moe_topic_quant(u.data() + (size_t) r * n_embd, n_embd, qr(i, r), &qinv[i * n_rows + r]); }
        counts.insert(counts.end(), c.begin(), c.end());
        think.insert(think.end(), t.begin(), t.end());
        rows_ln.insert(rows_ln.end(), rl.begin(), rl.end());
        trows_ln.insert(trows_ln.end(), tl.begin(), tl.end());
        hit.push_back(h); hit_think.push_back(ht); src.push_back(so); n_req.push_back(req); phash.push_back(ph); seq.push_back((uint64_t) id);
        text.push_back(std::move(tx));
        last.push_back(0);
        next_seq = std::max(next_seq, (uint64_t) id + 1);
        return true;
    }

    // a finished request: rows (n_rows x n_embd, as the model gave them), its answer map cnt and hit rate, what picked the cache's experts (source), its
    // prompt hash, its thinking map and hit rate (nullptr / < 0: it did not think), what it said and answered (tx).
    //   - the same prompt (its hash): each map replaced on its own phase's better hit rate (user 2026-10-28: "store only if new heatmap better hitrate"),
    //     else nothing changes; with same_pool the maps are joined instead;
    //   - else the nearest record when at least merge_cos alike (merge_cos > 1: never, the server's default since 2026-10-28: merging is offline):
    //     the maps joined;
    //   - else a new record (replacing the least recently used one past max_records).
    // A map written goes into the layer files as a row of sets (moe_topic_layer::write_row), the request's line to requests.js. Returns its index; SIZE_MAX for a key
    // of another row count.
    size_t add(const float * rows, uint32_t rows_n, const uint32_t * cnt, float merge_cos, float hit_rate = -1.0f, uint32_t source = 0,
               uint64_t prompt_hash = 0, const uint32_t * cnt_think = nullptr, float think_rate = -1.0f, const moe_topic_text * tx = nullptr) {
        if (n_rows == 0) { n_rows = rows_n; }
        if (rows_n != n_rows || rows_n == 0) { return SIZE_MAX; }
        std::vector<uint32_t> mc(cnt, cnt + rec_cells()), mt(rec_cells(), 0);
        if (cnt_think) { std::copy(cnt_think, cnt_think + rec_cells(), mt.begin()); }
        const std::vector<float> u = unit_rows(rows);
        const auto nb = nearest(u.data(), 1);
        size_t at = size();
        last_cos = nb.empty() ? -1.0f : nb[0].first;
        last_same = false;
        for (size_t i = 0; prompt_hash != 0 && i < size(); ++i) {
            if (phash[i] == prompt_hash) { at = i; last_same = true; break; }
        }
        if (!last_same && !nb.empty() && nb[0].first >= same_cos) { at = nb[0].second; last_same = true; }   // as alike as the same request
        if (!last_same && !nb.empty() && nb[0].first >= merge_cos) { at = nb[0].second; }
        last_merged = at < size();
        last_best = last_merged ? overlap(counts.data() + at * rec_cells(), mc.data()) : 0.0f;
        last_kept = false;
        bool ans = true, thk = cnt_think != nullptr && think_rate >= 0.0f;   // the maps that change
        if (last_same && !same_pool) {   // each map replaced on its own phase's better hit rate
            ans = hit_rate >= 0.0f && (hit[at] < 0.0f || hit_rate >= hit[at]);
            thk = thk && (hit_think[at] < 0.0f || think_rate >= hit_think[at]);
            if (!ans && !thk) {   // not measured, or not better: the record stays
                last_kept = true;
                return at;
            }
            if (ans) {
                std::copy(mc.begin(), mc.end(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
                hit[at] = hit_rate;
                src[at] = source;
            }
            if (thk) {
                std::copy(mt.begin(), mt.end(), think.begin() + (ptrdiff_t) (at * rec_cells()));
                hit_think[at] = think_rate;
            }
            if (tx) { text[at] = *tx; }
            n_req[at]++;
        } else if (last_merged) {   // joined (same_pool, or a near prompt with merging on): the uses added
            for (size_t j = 0; j < rec_cells(); ++j) { counts[at * rec_cells() + j] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) counts[at * rec_cells() + j] + mc[j]); }
            if (hit_rate > hit[at]) { hit[at] = hit_rate; src[at] = source; }
            if (thk) {
                for (size_t j = 0; j < rec_cells(); ++j) { think[at * rec_cells() + j] = (uint32_t) std::min<uint64_t>(UINT32_MAX, (uint64_t) think[at * rec_cells() + j] + mt[j]); }
                hit_think[at] = std::max(hit_think[at], think_rate);
            }
            n_req[at]++;
        } else {
            if (size() >= max_records && max_records > 0) {
                at = (size_t) (std::min_element(last.begin(), last.end()) - last.begin());
                append(req_path(), "k" + std::to_string(seq[at]) + "=null;\n");   // its request goes from requests.js (its rows stay: lines never move)
            } else {
                at = size();
                qrows.resize(qrows.size() + (size_t) n_rows * key_bytes());
                qinv.resize(qinv.size() + n_rows, 0.0f);
                counts.resize(counts.size() + rec_cells(), 0);
                think.resize(think.size() + rec_cells(), 0);
                rows_ln.resize(rows_ln.size() + n_layers, -1);
                trows_ln.resize(trows_ln.size() + n_layers, -1);
                n_req.push_back(0); hit.push_back(-1.0f); hit_think.push_back(-1.0f); src.push_back(0); phash.push_back(0); seq.push_back(0);
                text.emplace_back(); last.push_back(0);
            }
            std::copy(mc.begin(), mc.end(), counts.begin() + (ptrdiff_t) (at * rec_cells()));
            std::copy(mt.begin(), mt.end(), think.begin() + (ptrdiff_t) (at * rec_cells()));
            std::fill(trows_ln.begin() + (ptrdiff_t) (at * n_layers), trows_ln.begin() + (ptrdiff_t) ((at + 1) * n_layers), -1);
            n_req[at] = 1;
            hit[at] = hit_rate;
            hit_think[at] = thk ? think_rate : -1.0f;
            src[at] = source;
            phash[at] = prompt_hash;
            seq[at] = next_seq++;
            text[at] = tx ? *tx : moe_topic_text();
            for (uint32_t r = 0; r < n_rows; ++r) { moe_topic_quant(u.data() + (size_t) r * n_embd, n_embd, qr(at, r), &qinv[at * n_rows + r]); }
            ans = true;
        }
        last[at] = ++clock;
        write_sets(at, ans, thk);
        append(req_path(), line_of(at, rows));
        return at;
    }

    // record i's maps that changed into the layer files as sets: rows_ln / trows_ln get the lines that hold them
    void write_sets(size_t i, bool ans, bool thk) {
        for (uint32_t li = 0; li < n_layers; ++li) {
            std::string out, rout;
            for (int m = 0; m < 2; ++m) {
                if (!(m == 0 ? ans : thk)) { continue; }
                const uint32_t * c = (m == 0 ? counts : think).data() + i * rec_cells() + (size_t) li * n_exp;
                bool any = false;
                for (uint32_t e = 0; e < n_exp && !any; ++e) { any = c[e] > 0; }
                const int32_t ln = any ? lay[li].write_row(c, out, rout) : -1;
                (m == 0 ? rows_ln : trows_ln)[i * n_layers + li] = ln;
            }
            if (!out.empty()) { append(lay[li].sets_path(), out); }
            if (!rout.empty()) { append(lay[li].rows_path(), rout); put_file(lay[li].moes_path(), lay[li].moes_text()); }
        }
    }

    // request i's line in requests.js (rows: its key as the model gave it): k the key rows (fp8 in hex), v / t per layer the line of its answer's /
    // thinking's row in that layer's file (-1: none), h / ht the hit rates, then what it said and answered (user 2026-10-29: "k1=[emb1:hex,emb2:hex,
    // etc];v1=[4,3,6,3,etc] ... store k,v,h=0.8 etc as hitrate per line too")
    std::string line_of(size_t i, const float * rows) const {
        const std::string n = std::to_string(seq[i]);
        char b[64];
        std::string o = "k" + n + "=[";
        for (uint32_t r = 0; r < n_rows; ++r) {
            o += r ? ",\"" : "\"";
            for (uint32_t j = 0; j < n_embd; ++j) {
                snprintf(b, sizeof(b), "%02x", moe_f2e(rows[(size_t) r * n_embd + j]));
                o += b;
            }
            o += "\"";
        }
        o += "];";
        for (int m = 0; m < 2; ++m) {
            o += (m == 0 ? "v" : "vt") + n + "=[";
            for (uint32_t li = 0; li < n_layers; ++li) { o += (li ? "," : "") + std::to_string((m == 0 ? rows_ln : trows_ln)[i * n_layers + li]); }
            o += "];";
        }
        for (int m = 0; m < 2; ++m) {   // this request's own counters, per layer, listed in ascending expert order of the row (m1=[[234,43],[17,5,2]];)
            o += (m == 0 ? "m" : "mt") + n + "=[";
            for (uint32_t li = 0; li < n_layers; ++li) {
                const uint32_t * c = (m == 0 ? counts : think).data() + i * rec_cells() + (size_t) li * n_exp;
                o += li ? ",[" : "[";
                bool first = true;
                for (uint32_t e = 0; e < n_exp; ++e) { if (c[e]) { o += (first ? "" : ",") + std::to_string(c[e]); first = false; } }
                o += "]";
            }
            o += "];";
        }
        snprintf(b, sizeof(b), "h%s=%.4f;ht%s=%.4f;", n.c_str(), hit[i], n.c_str(), hit_think[i]);
        o += b;
        o += "n" + n + "=" + std::to_string(n_req[i]) + ";src" + n + "=\"" + (src[i] ? "maps" : "ini") + "\";";
        snprintf(b, sizeof(b), "%016llx", (unsigned long long) phash[i]);
        o += "hash" + n + "=\"" + b + "\";";
        const time_t now = time(nullptr);
        char when[32];
        strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&now));
        o += "time" + n + "=" + js_str(when) + ";p" + n + "=" + js_str(text[i].prompt) + ";th" + n + "=" + js_str(text[i].thinking) + ";a" + n + "=" +
             js_str(text[i].answer) + ";\n";
        return o;
    }

    // text at the end of a file of the store (the directory made first); a write error only costs these lines
    void append(const std::string & file, const std::string & o) const {
        if (path.empty()) { return; }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ec);   // the model's directory, or a layer's inside it
        FILE * f = fopen(file.c_str(), "ab");
        if (!f) { return; }
        fwrite(o.data(), 1, o.size(), f);
        fclose(f);
    }

    // a file written whole beside the target and renamed over it, so a crash leaves the old file whole
    bool put_file(const std::string & fn, const std::string & o) const {
        if (path.empty()) { return false; }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(fn).parent_path(), ec);
        const std::string tmp = fn + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) { return false; }
        const bool ok = fwrite(o.data(), 1, o.size(), f) == o.size();
        fclose(f);
        if (ok) { std::filesystem::rename(tmp, fn, ec); }
        if (!ok || ec) { std::remove(tmp.c_str()); return false; }
        return true;
    }

    // the merge of the nearest records the cache fills from (urows: the key's unit rows): per layer, each record's map as shares of that layer's experts,
    // weighted by its averaged similarity (only records at least min_cos alike), scaled so the hottest expert of all layers is 1: one scale, since experts of
    // different layers compete for the blocks of a size class. Returns the records used and the best similarity. Each layer takes the global map weighted 1 - its OWN row's
    // best cosine among them (last_wg: their mean; none stored: the global map alone). thinking: the records' thinking maps
    size_t merge(const float * urows, size_t k, float min_cos, std::vector<float> & heat, float * best = nullptr, bool thinking = false) {
        heat.assign(rec_cells(), 0.0f);
        auto add_layer = [&](uint32_t l, const uint32_t * c, float w) {   // layer l: each expert's share of the layer's map, times w
            uint64_t tot = 0;
            for (uint32_t xi = 0; xi < n_exp; ++xi) { tot += c[(size_t) l * n_exp + xi]; }
            if (tot == 0) { return; }
            for (uint32_t xi = 0; xi < n_exp; ++xi) { heat[(size_t) l * n_exp + xi] += w * (float) c[(size_t) l * n_exp + xi] / (float) tot; }
        };
        // User 2026-10-29: the records come from the cosine averaged over all rows (the right requests: it followed the experts' overlap at 0.93, a single
        // layer's row 0.72-0.93), weighted by it, and their maps (the answer's m or the thinking's mt counters) give each layer its hottest experts. "Only
        // if avg cos dst too far then per layer cos dist best sets hotsets or fallback per layer global hot": when the best averaged cosine is under
        // near_cos, each layer is looked up with its own row's cosine instead, the records nearest by that row weighting the layer's map, and the layer
        // takes the global map (its moes.js) at weight 1 - its best cosine, with its glob_floor hottest experts first
        const auto nb = n_rows ? nearest(urows, k) : std::vector<std::pair<float, size_t>>();
        if (best) { *best = nb.empty() ? -1.0f : nb[0].first; }
        size_t used = 0;
        std::vector<char> seen(size(), 0);
        auto use_record = [&](size_t i) { if (!seen[i]) { seen[i] = 1; last[i] = ++clock; used++; } };
        last_wg = 0.0f;
        if ((!nb.empty() && nb[0].first >= near_cos) || !n_rows) {
            for (const auto & p : nb) {
                // only records as near as near_cos: a record of another prompt (GLM, 2026-10-29: cosine 0.50 against 0.98 for the same prompt) merged in at
                // weight 0.50 made the cache hold 6428 experts for a map of 4462 and the hit rate no better than the ini's
                if (p.first < min_cos || p.first < near_cos) { break; }
                const uint32_t * c = (thinking ? think : counts).data() + p.second * rec_cells();
                for (uint32_t l = 0; l < n_layers; ++l) { add_layer(l, c, p.first); }
                use_record(p.second);
            }
        } else {
            const std::vector<uint32_t> g = glob_map();
            const bool has_glob = use_glob && glob_requests() > 0;
            std::vector<float> wg(n_layers, 0.0f);
            std::vector<std::pair<float, size_t>> lc(size());
            for (uint32_t l = 0; l < n_layers; ++l) {
                // the layer's row against the same row of every record; the nearest ones set the layer's map, and when even the nearest is under near_cos
                // the layer takes the global map alone: its moes.js, the glob_floor hottest experts first ("if still too far then that layer's moes.js counts")
                float bl = 0.0f;
                std::pair<float, size_t> top1{ -1.0f, 0 };
                if (size() > 0) {
                    const uint32_t r = std::min(l, n_rows - 1);   // the layer's own row of the key (the rows are the MoE layers, then the final state)
                    for (size_t i = 0; i < size(); ++i) { lc[i] = { moe_topic_qdot(qr(i, r), urows + (size_t) r * n_embd, n_embd) * qinv[i * n_rows + r], i }; }
                    const size_t kk = std::min(k, lc.size());
                    std::partial_sort(lc.begin(), lc.begin() + (ptrdiff_t) kk, lc.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                    top1 = lc[0];
                    if (top1.first >= near_cos || !has_glob) {
                        for (size_t j = 0; j < kk; ++j) {
                            if (lc[j].first < min_cos || lc[j].first <= 0.0f) { break; }
                            add_layer(l, (thinking ? think : counts).data() + lc[j].second * rec_cells(), lc[j].first);
                            bl = std::max(bl, std::min(1.0f, lc[j].first));
                            use_record(lc[j].second);
                        }
                    }
                }
                wg[l] = has_glob && bl <= 0.0f ? 1.0f : 0.0f;   // no record near enough in this layer: the global map alone
                last_wg += wg[l] / (float) n_layers;
                if (wg[l] > 0.0f) { add_layer(l, g.data(), wg[l]); }
            }
            // "from all layers pull hottest moes and pick min 8 hottest each, rest based on their global hot numbers": a layer that took the global map has its
            // glob_floor hottest experts (by the global counts) ranked above everything else, in their own order; the rest compete by their shares
            const float top = std::max(1e-6f, *std::max_element(heat.begin(), heat.end()));
            std::vector<uint32_t> ids(n_exp);
            for (uint32_t l = 0; l < n_layers; ++l) {
                if (wg[l] <= 0.0f) { continue; }
                const uint32_t * c = g.data() + (size_t) l * n_exp;
                for (uint32_t xi = 0; xi < n_exp; ++xi) { ids[xi] = xi; }
                std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return c[a] > c[b]; });
                for (uint32_t j = 0; j < glob_floor && j < n_exp && c[ids[j]] > 0; ++j) { heat[(size_t) l * n_exp + ids[j]] += top; }
            }
        }
        const float mx = heat.empty() ? 0.0f : *std::max_element(heat.begin(), heat.end());
        if (mx > 0.0f) { for (float & h : heat) { h /= mx; } }
        return used;
    }
};
