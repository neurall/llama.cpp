// Unit test of the hot map store (src/llama-moetopic-fork.h, task 158): keys of several rows compared row by row (the averaged cosine of 4-bit rows),
// one record per prompt (its hash), each of the two maps (answer, thinking) replaced on its own better hit rate, the merged heat of the nearest records,
// the global map, and the JS file: one assignment per line, the last assignment of a name wins, opening rewrites the file with those, a key of another
// width is refused, the binary file of the layouts before is deleted. No GPU, no model.
//
// build: cmake --build <dir> --target test-moetopic-fork;  run: ctest -R test-moetopic-fork
#include "../src/llama-moetopic-fork.h"

#include <cmath>
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
    moe_topic_store::for_lines(path, [&](const std::string & l) { n += !l.empty(); });
    return n;
}

int main() {
    const uint32_t E = 64, L = 3, X = 8, R = 4;   // values per row, layers, experts per layer, rows per key
    const float NEVER = 2.0f;                     // a merge similarity no key reaches: every prompt a record of its own
    const std::string path = std::string(P_tmpdir) + "/test-moetopic-fork.js";
    const std::string gpath = std::string(P_tmpdir) + "/test-moetopic-fork.global.js";
    std::remove(path.c_str());
    std::remove(gpath.c_str());
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

    // the same prompt (its hash): the map replaced only by a better hit rate; another prompt with the same key is a record of its own
    CHECK(s.add(a.data(), R, cb.data(), NEVER, 0.5f, 0, 11) == 0 && s.last_same && s.last_kept && s.counts[1] == 10);
    CHECK(s.add(a2.data(), R, cb.data(), NEVER, 0.8f, 1, 11) == 0 && !s.last_kept && s.counts[6] == 7 && s.counts[1] == 0 && s.hit[0] == 0.8f && s.src[0] == 1 &&
          s.n_req[0] == 2);
    CHECK(s.add(a.data(), R, ca.data(), NEVER, 0.9f, 0, 33) == 2 && s.size() == 3 && !s.last_same);

    // two maps per record: the thinking map is replaced on its own better hit rate, the answer's on its own; a request that did not think leaves the
    // thinking map as it was; a new record of a request that did not think has an empty one; the thinking maps merge without the global map
    {
        moe_topic_store tk;
        CHECK(tk.open("", E, L, X));
        CHECK(tk.add(a.data(), R, ca.data(), NEVER, 0.6f, 0, 5, cb.data(), 0.5f) == 0 && tk.hit_think[0] == 0.5f && tk.think[6] == 7 && tk.counts[1] == 10);
        CHECK(tk.add(a.data(), R, cb.data(), NEVER, 0.4f, 0, 5, ca.data(), 0.9f) == 0 && !tk.last_kept && tk.counts[1] == 10 && tk.hit[0] == 0.6f &&
              tk.think[1] == 10 && tk.think[6] == 0 && tk.hit_think[0] == 0.9f);   // a worse answer, a better thinking phase: only the thinking map changes
        CHECK(tk.add(a.data(), R, cb.data(), NEVER, 0.8f, 0, 5) == 0 && tk.counts[6] == 7 && tk.think[1] == 10 && tk.hit_think[0] == 0.9f);
        CHECK(tk.add(b.data(), R, cb.data(), NEVER, 0.8f, 0, 6) == 1 && tk.hit_think[1] < 0.0f && tk.think[X * L + 6] == 0);
        std::vector<float> hk;
        float bk = 0.0f;
        const std::vector<float> ua = tk.unit_rows(a.data());
        CHECK(tk.merge(ua.data(), 1, -1.0f, hk, &bk, true) == 1 && hk[1] == 1.0f && tk.last_wg == 0.0f);
    }

    // the merged heat of a key near a, one scale for all layers: expert 1 of layers 0 and 1 (2/3 of their uses) is 1.0 and expert 2 there 0.5; layer 2
    // splits its uses half and half, so both its experts are 0.75 (a scale per layer would make them 1.0); b's expert 6 is absent (b is below min_cos)
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
        // a key far from every record takes the global map at weight 1 - its best similarity; with use_glob off none
        m.use_glob = true;
        std::vector<float> far(a.size());
        for (size_t j = 0; j < far.size(); ++j) { far[j] = -(a[j] + b[j]); }   // away from a and b alike
        const std::vector<float> uf = m.unit_rows(far.data());
        CHECK(m.merge(uf.data(), 4, -1.0f, heat, &best) == 0 && best < 0.0f && m.last_wg == 1.0f && heat[1] > 0.0f && heat[6] > 0.0f);
    }

    // the global map: every stored request's answer counts (s: ca, cb, cb kept, cb, ca), the refused key not
    CHECK(s.glob_req == 5 && s.glob[1] == 20 && s.glob[6] == 21);

    // the file: 12 lines per change (4 changes of 3 records); opening keeps the last line of each name and rewrites the file with those
    CHECK(lines_of(path) == 48);
    moe_topic_store r;
    CHECK(r.open(path, E, L, X) && r.size() == 3 && r.n_rows == R && lines_of(path) == 36);
    CHECK(r.phash == s.phash && r.counts == s.counts && r.think == s.think && r.hit == s.hit && r.src == s.src && r.n_req == s.n_req && r.seq == s.seq);
    CHECK(r.glob_req == 5 && r.glob == s.glob);
    const std::vector<float> ub = r.unit_rows(b.data());
    CHECK(r.nearest(ub.data(), 1)[0].second == 1);

    // past the record cap the least recently used record is replaced: its id dropped from the file, the new record with a new id
    {
        moe_topic_store c;
        CHECK(c.open(path, E, L, X) && c.size() == 3);
        c.max_records = 3;
        std::vector<float> hc;
        const std::vector<float> ua = c.unit_rows(a.data());
        c.merge(ua.data(), 2, 0.5f, hc);   // records 0 and 2 (key a) used last: record 1 (b) is the one to go
        const std::vector<float> z = rnd(g, R * E);
        CHECK(c.add(z.data(), R, cb.data(), NEVER, 0.5f, 0, 44) == 1 && c.size() == 3 && c.seq[1] == 3 && c.phash[1] == 44);
        moe_topic_store c2;
        CHECK(c2.open(path, E, L, X) && c2.size() == 3 && c2.phash[0] == 11 && c2.phash[1] == 33 && c2.phash[2] == 44);   // id order: 0, 2, 3
    }

    // a key of another width (another model): the file is refused and never written over
    {
        moe_topic_store o;
        CHECK(!o.open(path, E + 1, L, X) && o.size() == 0 && o.path.empty());
        moe_topic_store o2;
        CHECK(o2.open(path, E, L, X) && o2.size() == 3);
    }

    // the binary file of the layouts before: deleted at open (old maps are junk, no copies kept)
    {
        const std::string old = std::string(P_tmpdir) + "/test-moetopic-fork.hotmap";
        FILE * f = fopen(old.c_str(), "wb");
        fwrite("MOETOPC8", 1, 8, f);
        fclose(f);
        moe_topic_store o;
        CHECK(o.open(path, E, L, X));
        FILE * gone = fopen(old.c_str(), "rb");
        CHECK(gone == nullptr);
        if (gone) { fclose(gone); }
    }

    std::remove(path.c_str());
    std::remove(gpath.c_str());
    printf("test-moetopic-fork: OK\n");
    return 0;
}
