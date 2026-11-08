// Unit test of the hot map store (src/llama-moetopic-fork.h, task 158): the sets of a layer's file (a row decomposed against the sets there: a set
// wholly inside it referenced, a shared part split out as a smaller set written before what uses it, lines only appended), keys of several rows compared
// row by row, one record per prompt (its hash), the two maps (answer, thinking) each replaced on its own better hit rate, the merged heat of the nearest
// records, the global map, and the directory: keys.js (the last assignment of a name wins, rewritten on open), the layer files read back, a key of another
// width refused, the stores of the layouts before deleted. No GPU, no model.
//
// build: cmake --build <dir> --target test-moetopic-fork;  run: ctest -R test-moetopic-fork
#include "../src/llama-moetopic-fork.h"

#include <cmath>
#include <map>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// a check that also runs in release builds (assert is gone under NDEBUG, and several checks below call the store)
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "test-moetopic-fork: FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

// a key as the model would give it: rows of normal values
static std::vector<float> rnd(std::mt19937 & g, size_t n) {
    std::normal_distribution<float> d;
    std::vector<float> v(n);
    for (float & x : v) { x = d(g); }
    return v;
}

static size_t lines_of(const std::string & path) {
    size_t n = 0;
    moe_topic_store_lines(path, [&](const std::string & l) { n += !l.empty(); });
    return n;
}

int main() {
    std::mt19937 g(7);

    // a 4-bit row keeps its direction: on a 5120-wide random unit row the cosine to the exact row stays above 0.97, the quantized row has unit length,
    // and the search kernel agrees with the dequantized vector
    {
        const uint32_t W = 5120;
        std::vector<float> wide = rnd(g, W), back(W);
        double nn = 0.0;
        for (float x : wide) { nn += (double) x * x; }
        for (float & x : wide) { x = (float) (x / std::sqrt(nn)); }
        std::vector<uint8_t> q((W + 1) / 2);
        float qi = 0.0f;
        moe_topic_quant(wide.data(), W, q.data(), &qi);
        moe_topic_dequant(q.data(), W, qi, back.data());
        double d = 0.0, bn = 0.0;
        for (uint32_t j = 0; j < W; ++j) { d += (double) wide[j] * back[j]; bn += (double) back[j] * back[j]; }
        CHECK(d > 0.97 && std::fabs(bn - 1.0) < 1e-4);
        CHECK(std::fabs(moe_topic_qdot(q.data(), wide.data(), W) * qi - d) < 1e-3);
    }

    // a layer's three files: sets.js (s<n>=[3,34,s1]; one set a line), rows.js (c<n>=[...]; one row a line), moes.js (m=[count, ...]; one line, rewritten).
    // row 0 holds experts 0..29 (a leaf of its own: s0), row 1 experts 10..39: the 20 shared (10..29) split out of s0 as a leaf (s2) with its rest (s3),
    // s0 redefined below them (line 4), row 1 = s2 and a leaf of its 10 own (s5); a row of just the shared part names it (row 2); 3 more experts than the
    // shared part stay inline (row 3)
    {
        const std::string dir = std::string(P_tmpdir) + "/test-moetopic-layer";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        moe_topic_layer ly;
        ly.n_exp = 64;
        std::vector<uint32_t> u1(64, 0), u2(64, 0), u3(64, 0), u4(64, 0);
        for (uint32_t e = 0; e < 30; ++e) { u1[e] = 30 - e; }
        for (uint32_t e = 10; e < 40; ++e) { u2[e] = 5; }
        for (uint32_t e = 10; e < 30; ++e) { u3[e] = 1; }
        for (uint32_t e = 10; e < 33; ++e) { u4[e] = 2; }
        std::string sets, rws;
        CHECK(ly.write_row(u1.data(), sets, rws) == 0 && ly.kind[0] == 's' && ly.rows[0].refs == std::vector<int32_t>{ 0 });
        CHECK(ly.write_row(u2.data(), sets, rws) == 1 && std::string(ly.kind.begin(), ly.kind.end()) == "sssrs");
        CHECK(ly.nodes[0].refs == (std::vector<int32_t>{ 1, 2 }) && ly.nodes[1].size == 20 && ly.nodes[2].size == 10 && ly.nodes[4].size == 10);
        CHECK(ly.rows[1].refs == (std::vector<int32_t>{ 1, 4 }) && ly.rows[1].own.empty() && ly.rows[1].size == 30);
        CHECK(ly.write_row(u3.data(), sets, rws) == 2 && ly.rows[2].refs == std::vector<int32_t>{ 1 } && ly.rows[2].own.empty());
        CHECK(ly.write_row(u4.data(), sets, rws) == 3 && ly.rows[3].refs == std::vector<int32_t>{ 1 } && ly.rows[3].own.size() == 3);
        // the counts keep rising, whatever the sets say: expert 10 was used 20 + 5 + 1 + 2 times, expert 35 5 times
        CHECK(ly.cnt[10] == 28 && ly.cnt[35] == 5 && ly.cnt[0] == 30 && ly.cnt[50] == 0);
        const std::vector<uint32_t> y = ly.row_experts(1);   // row 1: experts 10..39 in ascending order
        CHECK(y.size() == 30 && y[0] == 10 && y[29] == 39);
        // the text: sets only appended, in line order, every name above its use, plain numbers; read back, the same sets, rows and counts
        const char * want[] = { "s0=[0,1,", "s1=[", "s2=[", "s0=[s1,s2]", "s4=[" };
        size_t at = 0;
        for (const char * w : want) { const size_t q = sets.find(w, at); CHECK(q != std::string::npos && (q == 0 || sets[q - 1] == '\n')); at = q + 1; }
        CHECK(rws.find("c0=[s0];\n") == 0 && rws.find("c1=[s1,s4];\n") != std::string::npos && rws.find("{") == std::string::npos);
        auto wr = [&](const char * name, const std::string & t) { FILE * f = fopen((dir + "/" + name).c_str(), "wb"); fwrite(t.data(), 1, t.size(), f); fclose(f); };
        wr("sets.js", sets);
        wr("rows.js", rws);
        wr("moes.js", ly.moes_text());
        moe_topic_layer back;
        CHECK(back.load(dir, 64) && back.nodes.size() == 5 && back.kind == ly.kind && back.rows.size() == 4 && back.cnt == ly.cnt);
        for (size_t n = 0; n < 5; ++n) { CHECK(back.nodes[n].mask == ly.nodes[n].mask); }
        CHECK(back.row_experts(1) == y);
        // more rows later: sets.js and rows.js only grow, moes.js stays one line
        const std::string before = sets;
        std::string s2, r2;
        std::vector<uint32_t> u5(64, 0);
        for (uint32_t e = 20; e < 50; ++e) { u5[e] = 1; }
        back.write_row(u5.data(), s2, r2);
        const std::string mt = back.moes_text();
        CHECK(std::count(mt.begin(), mt.end(), '\n') == 1 && mt.compare(0, 3, "m=[") == 0 && back.cnt[45] == 1);
        // several statements on one line: each still counts as a line of its own (s0, then s1)
        wr("sets.js", "s0=[1,2,3];s1=[4]\n");
        wr("rows.js", "c0=[7,s0,s1];\n");
        wr("moes.js", "");
        moe_topic_layer one;
        CHECK(one.load(dir, 64) && one.nodes.size() == 2 && one.rows.size() == 1 && one.rows[0].size == 5);
        // a row naming a line that is not a set: refused
        wr("sets.js", "s0=[1,2]\n");
        wr("rows.js", "c0=[s1];\n");
        moe_topic_layer bad;
        CHECK(!bad.load(dir, 64));
        // an expert number past the layer's width: refused
        wr("sets.js", "s0=[1,70]\n");
        wr("rows.js", "");
        moe_topic_layer wide;
        CHECK(!wide.load(dir, 64));
        std::filesystem::remove_all(dir);
    }

    // fp8 (E4M3): 448 the largest, the nearest code, the sign kept
    CHECK(moe_e2f(moe_f2e(1.0f)) == 1.0f && moe_e2f(moe_f2e(-0.4375f)) == -0.4375f && moe_e2f(moe_f2e(1000.0f)) == 448.0f && moe_e2f(moe_f2e(0.0f)) == 0.0f);
    CHECK(std::fabs(moe_e2f(moe_f2e(27.3f)) - 27.3f) < 2.0f);

    const uint32_t E = 64, L = 3, X = 8, R = 4;   // values per row, layers, experts per layer, rows per key
    const float NEVER = 2.0f;                     // a merge similarity no key reaches: every prompt a record of its own
    const std::string path = std::string(P_tmpdir) + "/test-moetopic-fork";
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    moe_topic_store s;
    CHECK(s.open(path, E, L, X) && s.size() == 0);

    // two prompts far apart, each a record; a key of another row count is refused
    const std::vector<float> a = rnd(g, R * E), b = rnd(g, R * E);
    std::vector<uint32_t> ca(L * X, 0), cb(L * X, 0);
    for (uint32_t l = 0; l < L; ++l) { ca[l * X + 1] = 10; ca[l * X + 2] = l == 2 ? 10 : 5; cb[l * X + 6] = 7; }   // layer 2 spreads its uses
    CHECK(s.add(a.data(), R, ca.data(), NEVER, 0.6f, 0, 11) == 0 && s.n_rows == R);
    CHECK(s.add(b.data(), R, cb.data(), NEVER, 0.7f, 0, 22) == 1 && s.size() == 2);
    CHECK(s.add(a.data(), R + 1, ca.data(), NEVER) == SIZE_MAX && s.size() == 2);

    // a key near a (every row moved a little) finds a first: the averaged row cosine
    std::vector<float> a2 = a;
    const std::vector<float> na = rnd(g, R * E);
    for (size_t j = 0; j < a2.size(); ++j) { a2[j] += 0.1f * na[j]; }
    const std::vector<float> ua2 = s.unit_rows(a2.data());
    const auto nb = s.nearest(ua2.data(), 2);
    CHECK(nb.size() == 2 && nb[0].second == 0 && nb[0].first > 0.9f && nb[1].first < nb[0].first);

    // the same prompt (its hash): the map replaced only by a better hit rate; another prompt with the same key is a record of its own when the key test is
    // off (same_cos > 1), and the same request when it is on (its key as alike as same_cos: not appended, replaced only on a better hit rate)
    CHECK(s.add(a.data(), R, cb.data(), NEVER, 0.5f, 0, 11) == 0 && s.last_same && s.last_kept && s.counts[1] > 0);
    CHECK(s.add(a2.data(), R, cb.data(), NEVER, 0.8f, 1, 11) == 0 && !s.last_kept && s.counts[6] > 0 && s.counts[1] == 0 && s.hit[0] == 0.8f && s.src[0] == 1 &&
          s.n_req[0] == 2);
    CHECK(s.add(a.data(), R, ca.data(), NEVER, 0.7f, 0, 33) == 0 && s.size() == 2 && s.last_same && s.last_kept);   // its key: request 0 again, not better
    s.same_cos = 2.0f;
    CHECK(s.add(a.data(), R, ca.data(), NEVER, 0.9f, 0, 33) == 2 && s.size() == 3 && !s.last_same);

    // two maps per record: the thinking map is replaced on its own better hit rate, the answer's on its own; a request that did not think leaves the
    // thinking map as it was; a new record of a request that did not think has an empty one; the thinking maps merge without the global map
    {
        moe_topic_store tk;
        CHECK(tk.open("", E, L, X));
        CHECK(tk.add(a.data(), R, ca.data(), NEVER, 0.6f, 0, 5, cb.data(), 0.5f) == 0 && tk.hit_think[0] == 0.5f && tk.think[6] > 0 && tk.counts[1] > 0);
        CHECK(tk.add(a.data(), R, cb.data(), NEVER, 0.4f, 0, 5, ca.data(), 0.9f) == 0 && !tk.last_kept && tk.counts[1] > 0 && tk.hit[0] == 0.6f &&
              tk.think[1] > 0 && tk.think[6] == 0 && tk.hit_think[0] == 0.9f);   // a worse answer, a better thinking phase: only the thinking map changes
        CHECK(tk.add(a.data(), R, cb.data(), NEVER, 0.8f, 0, 5) == 0 && tk.counts[6] > 0 && tk.think[1] > 0 && tk.hit_think[0] == 0.9f);
        CHECK(tk.add(b.data(), R, cb.data(), NEVER, 0.8f, 0, 6) == 1 && tk.hit_think[1] < 0.0f && tk.think[X * L + 6] == 0);
        std::vector<float> hk;
        float bk = 0.0f;
        const std::vector<float> ua = tk.unit_rows(a.data());
        CHECK(tk.merge(ua.data(), 1, -1.0f, hk, &bk, true) == 1 && hk[1] == 1.0f && tk.last_wg == 0.0f);
    }

    // the merged heat of a key near a, one scale for all layers: expert 1 of layers 0 and 1 (2/3 of their uses) is 1.0 and expert 2 there 0.5; layer 2
    // splits its uses half and half, so both its experts are 0.75 (a scale per layer would make them 1.0); b's expert 6 is absent (b is below min_cos); b's expert 6 is absent (b is below min_cos)
    {
        moe_topic_store m;
        CHECK(m.open("", E, L, X));
        m.use_glob = false;
        m.add(a.data(), R, ca.data(), NEVER, 0.6f, 0, 1);
        m.add(b.data(), R, cb.data(), NEVER, 0.6f, 0, 2);
        std::vector<float> heat;
        float best = 0.0f;
        CHECK(m.merge(ua2.data(), 4, 0.5f, heat, &best) == 1 && best > 0.9f);
        for (uint32_t l = 0; l < L; ++l) {
            const float h1 = l == 2 ? 0.75f : 1.0f, h2 = l == 2 ? 0.75f : 0.5f;
            CHECK(std::fabs(heat[l * X + 1] - h1) < 1e-6f && std::fabs(heat[l * X + 2] - h2) < 1e-6f && heat[l * X + 6] == 0.0f);
        }
        // a key far from every record takes the global map at weight 1 - its best similarity
        m.use_glob = true;
        std::vector<float> far(a.size());
        for (size_t j = 0; j < far.size(); ++j) { far[j] = -(a[j] + b[j]); }   // away from a and b alike
        const std::vector<float> uf = m.unit_rows(far.data());
        CHECK(m.merge(uf.data(), 4, -1.0f, heat, &best) == 0 && best < 0.0f && m.last_wg == 1.0f && heat[1] > 0.0f && heat[6] > 0.0f);
        // the floor: with one expert per layer guaranteed, layer 0's hottest by the global counts (expert 1: 10 uses against expert 6's 7) ranks first
        m.glob_floor = 1;
        CHECK(m.merge(uf.data(), 4, -1.0f, heat, &best) == 0 && std::fabs(heat[1] - 1.0f) < 1e-6f && heat[6] < heat[1] && heat[6] > 0.0f);
        // under near_cos the layers are looked up on their own rows: a key near a still finds a (its rows are alike), the global map only a little
        m.near_cos = 0.8f;
        CHECK(m.merge(ua2.data(), 4, 0.5f, heat, &best) == 1 && heat[1] > heat[6]);   // (the averaged cosine is near: the records as they are)
        m.near_cos = 2.0f;   // every row 'too far': each layer takes the global map alone (nothing is near enough, no record used)
        CHECK(m.merge(ua2.data(), 4, 0.5f, heat, &best) == 0 && m.last_wg == 1.0f && heat[1] > 0.0f && heat[6] > 0.0f);
    }

    // requests.js, one line per stored request: k the key rows in fp8 hex (2 digits a value), v / t one layer line each, h / ht the hit rates, the hash
    {
        std::vector<std::string> lines;
        moe_topic_store_lines(path + "/requests.js", [&](const std::string & l) { if (!l.empty()) { lines.push_back(l); } });
        CHECK(lines.size() == 4 && lines[0].compare(0, 5, "k0=[\"") == 0);
        std::map<std::string, moe_js> f;
        CHECK(moe_js_statements(lines[0], [&](const std::string & name, const moe_js & v, const std::string &) { f[name] = v; }));
        CHECK(f["k0"].arr.size() == R && f["k0"].arr[0].str.size() == 2 * E && f["v0"].nums.size() == L && f["vt0"].nums.size() == L);
        CHECK(f["h0"].num == 0.6 && f["ht0"].num == -1.0 && f["hash0"].str == "000000000000000b" && f["n0"].num == 1);
        // m: this request's own counters per layer, ascending expert order (experts 1 and 2: 10 and 5, in layer 2 10 and 10); mt: none, it did not think
        CHECK(f["m0"].arr.size() == L && f["m0"].arr[0].nums == (std::vector<double>{ 10, 5 }) && f["m0"].arr[2].nums == (std::vector<double>{ 10, 10 }) && f["mt0"].arr[0].nums.empty());
        for (uint32_t l = 0; l < L; ++l) { CHECK(f["v0"].nums[l] >= 0 && f["vt0"].nums[l] == -1); }   // every layer has an answer row, none thought
        // the key rows read back as fp8: each value the nearest fp8 code of the float given
        for (uint32_t j = 0; j < E; ++j) {
            const uint8_t c = (uint8_t) std::strtoul(f["k0"].arr[0].str.substr((size_t) j * 2, 2).c_str(), nullptr, 16);
            CHECK(c == moe_f2e(a[j]));
        }
    }

    // the layer files are only appended: after more requests every earlier byte is where it was (the line numbers the requests keep never move); a
    // better run of a stored prompt is one more line in requests.js, a worse one none
    {
        std::string before;
        moe_topic_store_lines(path + "/1/rows.js", [&](const std::string & l) { before += l + "\n"; });
        const size_t req_lines = lines_of(path + "/requests.js");
        std::vector<uint32_t> cz(L * X, 0);
        for (uint32_t l = 0; l < L; ++l) { cz[l * X + 1] = 3; cz[l * X + 3] = 9; }
        const std::vector<float> zk = rnd(g, R * E);
        CHECK(s.add(zk.data(), R, cz.data(), NEVER, 0.5f, 0, 77) == 3 && lines_of(path + "/requests.js") == req_lines + 1);
        CHECK(s.add(zk.data(), R, cz.data(), NEVER, 0.4f, 0, 77) == 3 && s.last_kept && lines_of(path + "/requests.js") == req_lines + 1);
        CHECK(s.add(zk.data(), R, ca.data(), NEVER, 0.9f, 0, 77) == 3 && !s.last_kept && lines_of(path + "/requests.js") == req_lines + 2);
        std::string after;
        moe_topic_store_lines(path + "/1/rows.js", [&](const std::string & l) { after += l + "\n"; });
        CHECK(after.size() > before.size() && after.compare(0, before.size(), before) == 0);
    }

    // the global map: every stored request's answer counts (s: ca, cb, cb kept, cb, ca kept, ca, cz, cz kept, ca), the refused key not
    {   // the layers' moes.js together: experts 1 and 6 of layer 0 were used (ca / cz, cb); expert 5 never
        const std::vector<uint32_t> g = s.glob_map();
        CHECK(s.glob_requests() > 0 && g[1] > 0 && g[6] > 0 && g[5] == 0);
    }

    // the directory: requests.js one line per change (4 changes of 3 requests), only appended, the last line of a number wins; the layer files read back
    CHECK(lines_of(path + "/requests.js") == 6 && lines_of(path + "/0/rows.js") > 0);
    moe_topic_store r;
    CHECK(r.open(path, E, L, X) && r.size() == 4 && r.n_rows == R && lines_of(path + "/requests.js") == 6);
    CHECK(r.phash == s.phash && r.counts == s.counts && r.think == s.think && r.hit == s.hit && r.src == s.src && r.n_req == s.n_req && r.seq == s.seq);
    CHECK(r.rows_ln == s.rows_ln && r.glob_requests() == s.glob_requests() && r.glob_map() == s.glob_map());
    const std::vector<float> ub = r.unit_rows(b.data());
    CHECK(r.nearest(ub.data(), 1)[0].second == 1);

    // past the record cap the least recently used record is replaced: its request dropped from requests.js (k<n>=null), the new one with a new number
    {
        moe_topic_store c;
        CHECK(c.open(path, E, L, X) && c.size() == 4);
        c.max_records = 4;
        std::vector<float> hc;
        const std::vector<float> ua = c.unit_rows(a.data());
        c.same_cos = 2.0f;
        c.merge(ua.data(), 2, 0.5f, hc);   // records 0 and 2 (key a) used last
        const std::vector<float> uz = c.unit_rows(c.unit_rows(a.data()).data());
        c.last[3] = c.clock + 100;         // and request 3 just now: record 1 (b) is the one to go
        const std::vector<float> z = rnd(g, R * E);
        CHECK(c.add(z.data(), R, cb.data(), NEVER, 0.5f, 0, 44) == 1 && c.size() == 4 && c.seq[1] == 4 && c.phash[1] == 44);
        moe_topic_store c2;
        CHECK(c2.open(path, E, L, X) && c2.size() == 4 && c2.phash[0] == 11 && c2.phash[1] == 33 && c2.phash[2] == 77 && c2.phash[3] == 44);   // request order: 0, 2, 3, 4
    }

    // a key of another width (another model): the directory is refused and never written over
    {
        moe_topic_store o;
        CHECK(!o.open(path, E + 1, L, X) && o.size() == 0 && o.path.empty());
        moe_topic_store o2;
        CHECK(o2.open(path, E, L, X) && o2.size() == 4);
    }

    // the stores of the layouts before (the binary file, the one-file JS store, a directory with keys.js): deleted at open (old maps are junk)
    {
        FILE * kf = fopen((path + "/keys.js").c_str(), "wb");
        fputs("key0=null;\n", kf);
        fclose(kf);
        moe_topic_store k;
        CHECK(k.open(path, E, L, X) && k.size() == 0 && lines_of(path + "/requests.js") == 0 && lines_of(path + "/0/rows.js") == 0);
        for (const char * ext : { ".hotmap", ".js", ".global.js" }) {
            FILE * f = fopen((path + ext).c_str(), "wb");
            fwrite("MOETOPC8", 1, 8, f);
            fclose(f);
        }
        moe_topic_store o;
        CHECK(o.open(path, E, L, X));
        for (const char * ext : { ".hotmap", ".js", ".global.js" }) {
            FILE * gone = fopen((path + ext).c_str(), "rb");
            CHECK(gone == nullptr);
            if (gone) { fclose(gone); }
        }
    }

    std::filesystem::remove_all(path, ec);
    printf("test-moetopic-fork: OK\n");
    return 0;
}
