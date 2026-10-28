// Unit test of the per-topic hot map store (src/llama-moetopic-fork.h, task 158): the 4-bit keys, nearest-record search by cosine similarity, merging of
// requests that used the same experts (keys pooled by their own rule, the coldest experts dropped), the automatic merge threshold, replacement past the
// record cap, the merged heat map, and the file round trip (a store of another shape is never read or overwritten). No GPU, no model.
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

static std::vector<float> unit(std::vector<float> v) {
    double n = 0.0;
    for (float x : v) { n += (double) x * x; }
    for (float & x : v) { x = (float) (x / std::sqrt(n)); }
    return v;
}

static std::vector<float> rnd(std::mt19937 & g, uint32_t n) {
    std::normal_distribution<float> d;
    std::vector<float> v(n);
    for (float & x : v) { x = d(g); }
    return unit(v);
}

// the two keys of a request (mean pooling, max pooling) side by side, as topic_key writes them
static std::vector<float> keys2(const std::vector<float> & mean, const std::vector<float> & max) {
    std::vector<float> k(mean);
    k.insert(k.end(), max.begin(), max.end());
    return k;
}

int main() {
    const uint32_t E = 64, L = 3, X = 8;
    const float NEVER = 2.0f;   // a merge threshold no overlap reaches: every request its own record
    const std::string path = std::string(P_tmpdir) + "/test-moetopic-fork.bin";
    std::remove(path.c_str());
    std::mt19937 g(7);

    // a 4-bit key keeps its direction: on a 5120-wide random unit key (a real key's width) the cosine to the exact key stays above 0.97, the quantized key
    // has unit length, and the search kernel agrees with the dequantized vector
    {
        const uint32_t W = 5120;
        std::vector<float> wide = rnd(g, W), back(W);
        std::vector<uint8_t> q((W + 1) / 2);
        float qi = 0.0f;
        moe_topic_quant(wide.data(), W, q.data(), &qi);
        moe_topic_dequant(q.data(), W, qi, back.data());
        double d = 0.0, nn = 0.0;
        for (uint32_t j = 0; j < W; ++j) { d += (double) wide[j] * back[j]; nn += (double) back[j] * back[j]; }
        CHECK(d > 0.97 && std::fabs(nn - 1.0) < 1e-4);
        CHECK(std::fabs(moe_topic_qdot(q.data(), wide.data(), W) * qi - d) < 1e-3);
    }

    moe_topic_store s;
    CHECK(s.open(path, E, L, X) && s.size() == 0);

    // two topics far apart: their own records, each found by a key near it, by either pooling's key
    const std::vector<float> a = rnd(g, E), am = rnd(g, E), b = rnd(g, E), bm = rnd(g, E);
    std::vector<uint32_t> ca(L * X, 0), cb(L * X, 0);
    for (uint32_t l = 0; l < L; ++l) { ca[l * X + 1] = 10; ca[l * X + 2] = l == 2 ? 10 : 5; cb[l * X + 6] = 7; }   // layer 2 spreads its uses
    CHECK(s.add(keys2(a, am).data(), ca.data(), 0.9f) == 0);
    CHECK(s.add(keys2(b, bm).data(), cb.data(), 0.9f) == 1);
    CHECK(s.size() == 2);
    std::vector<float> a2 = a;
    a2[0] += 0.01f;
    a2 = unit(a2);
    auto nb = s.nearest(a2.data(), 2, moe_topic_store::KEY_MEAN);
    CHECK(nb.size() == 2 && nb[0].second == 0 && nb[0].first > 0.95f && nb[1].first < nb[0].first);
    CHECK(s.nearest(bm.data(), 1, moe_topic_store::KEY_MAX)[0].second == 1);

    // a request that used the same experts merges into its record (lookup by the max key, the default): counts add, both keys stay unit length
    CHECK(s.add(keys2(a2, am).data(), ca.data(), 0.9f) == 0);
    CHECK(s.size() == 2 && s.n_req[0] == 2 && s.counts[1] == 20);
    for (int k = 0; k < moe_topic_store::N_KEYS; ++k) {
        const std::vector<float> k0 = s.key_of(0, k);
        double n = 0.0;
        for (uint32_t j = 0; j < E; ++j) { n += (double) k0[j] * k0[j]; }
        CHECK(std::fabs(n - 1.0) < 1e-4);
    }

    // store only a better hit rate: a request of a stored topic that hit worse leaves the record as it was; one that hit better replaces its rate
    {
        moe_topic_store h;
        CHECK(h.open("", E, L, X));
        CHECK(h.add(keys2(a, am).data(), ca.data(), 0.9f, 0.60f) == 0 && h.hit[0] == 0.60f);
        CHECK(h.add(keys2(a, am).data(), ca.data(), 0.9f, 0.50f) == 0 && h.last_kept && h.n_req[0] == 1 && h.counts[1] == 10);
        CHECK(h.add(keys2(a, am).data(), ca.data(), 0.9f, 0.80f) == 0 && !h.last_kept && h.n_req[0] == 2 && h.hit[0] == 0.80f);
    }

    // pooling keeps the hottest: a request on the same experts adds its counts, and an expert below its layer's mean use is dropped; a request that shares
    // only half its experts is a record of its own (the overlap is the sum of the smaller shares: 0.5 here)
    std::vector<uint32_t> cc(L * X, 0), ch(L * X, 0);
    for (uint32_t l = 0; l < L; ++l) { cc[l * X + 1] = 100; cc[l * X + 2] = 60; cc[l * X + 7] = 1; ch[l * X + 1] = 100; ch[l * X + 5] = 100; }
    {
        moe_topic_store p;
        CHECK(p.open("", E, L, X));
        CHECK(std::fabs(p.overlap(ch.data(), cc.data()) - 0.5f) < 1e-5f);
        CHECK(p.add(keys2(a, am).data(), cc.data(), 0.9f) == 0 && p.add(keys2(b, bm).data(), cc.data(), 0.9f) == 0 && p.size() == 1);   // same experts
        CHECK(p.counts[1] == 200 && p.counts[2] == 120 && p.counts[7] == 0 && p.n_req[0] == 2);   // expert 7 (2 of 322 uses) is below the mean of 40
        CHECK(p.add(keys2(a, am).data(), ch.data(), 0.9f) == 1 && p.size() == 2);
        CHECK(p.add(keys2(a, am).data(), cc.data(), NEVER) == 2 && p.size() == 3);   // merging off: the same request again is a record of its own
    }

    // pooled keys, each by its rule: the mean key the mean of the two (weighted by the requests each holds), the max key per value the larger
    {
        moe_topic_store m;
        CHECK(m.open("", E, L, X));
        std::vector<float> u(E, 0.0f), v(E, 0.0f);
        u[0] = 1.0f; v[1] = 1.0f;
        CHECK(m.add(keys2(u, u).data(), cc.data(), 0.9f) == 0 && m.add(keys2(v, v).data(), cc.data(), 0.9f) == 0);   // same experts: one record
        const std::vector<float> kmean = m.key_of(0, moe_topic_store::KEY_MEAN), kmax = m.key_of(0, moe_topic_store::KEY_MAX);
        CHECK(std::fabs(kmean[0] - 0.70710678f) < 1e-4f && std::fabs(kmean[1] - 0.70710678f) < 1e-4f && kmean[2] == 0.0f);   // (u + v) / |u + v|
        CHECK(std::fabs(kmax[0] - 0.70710678f) < 1e-4f && std::fabs(kmax[1] - 0.70710678f) < 1e-4f && kmax[2] == 0.0f);     // max(u, v), unit length
    }

    // the automatic threshold: too few requests or no gap between the two groups keeps the start value, else it splits the groups midway
    {
        moe_topic_store t;
        CHECK(t.open("", E, L, X) && t.auto_threshold() == 0.90f);
        t.seen = { 0.20f, 0.25f, 0.30f, 0.22f, 0.95f, 0.97f };
        CHECK(std::fabs(t.auto_threshold() - 0.625f) < 1e-5f);
        t.seen = { 0.88f, 0.90f, 0.91f, 0.92f, 0.93f, 0.95f };
        CHECK(t.auto_threshold() == 0.90f);
        t.seen.clear();
        CHECK(t.add(keys2(a, am).data(), cc.data(), -1.0f) == 0 && t.add(keys2(a, am).data(), ch.data(), -1.0f) == 1 && !t.last_merged);   // 0.5 < 0.9: new
    }

    // the merged heat of a key near a, one scale for all layers: expert 1 of layers 0 and 1 (2/3 of their uses) is 1.0 and expert 2 there 0.5; layer 2
    // splits its uses half and half, so both its experts are 0.75 (a scale per layer would make them 1.0); b's expert 6 is absent (b is below min_cos)
    std::vector<float> heat;
    float best = 0.0f;
    s.pool = moe_topic_store::KEY_MEAN;
    CHECK(s.merge(a.data(), 4, 0.5f, heat, &best) == 1 && best > 0.95f);
    for (uint32_t l = 0; l < L; ++l) {
        const float h1 = l == 2 ? 0.75f : 1.0f, h2 = l == 2 ? 0.75f : 0.5f;
        CHECK(std::fabs(heat[l * X + 1] - h1) < 1e-6f && std::fabs(heat[l * X + 2] - h2) < 1e-6f && heat[l * X + 6] == 0.0f);
    }

    // the file holds both records with both keys: a second store reads them back
    moe_topic_store r;
    CHECK(r.open(path, E, L, X) && r.size() == 2 && r.fkeys.size() == 2 * moe_topic_store::N_KEYS * E && r.key_of(1, 0) == s.key_of(1, 0) && r.qkeys == s.qkeys && r.n_req[0] == 2 && r.counts[1] == 20 && r.counts[L * X + 6] == 7);
    CHECK(r.nearest(b.data(), 1, moe_topic_store::KEY_MEAN)[0].second == 1 && r.nearest(bm.data(), 1, moe_topic_store::KEY_MAX)[0].second == 1);

    // another shape (another model or layer split): not read, and never written over
    moe_topic_store o;
    CHECK(!o.open(path, E, L + 1, X) && o.size() == 0 && o.path.empty());
    o.add(keys2(a, am).data(), std::vector<uint32_t>((L + 1) * X, 1).data(), 0.9f);
    moe_topic_store r2;
    CHECK(r2.open(path, E, L, X) && r2.size() == 2);

    // past the record cap the least recently used record is replaced; with no cap (0) the store grows
    moe_topic_store c;
    CHECK(c.open("", E, L, X));
    c.max_records = 2;
    c.pool = moe_topic_store::KEY_MEAN;
    c.add(keys2(a, am).data(), ca.data(), 0.9f);
    c.add(keys2(b, bm).data(), cb.data(), 0.9f);
    c.nearest(a.data(), 1, moe_topic_store::KEY_MEAN);
    c.merge(a.data(), 1, 0.5f, heat);   // a is used last: b is the one to go
    const std::vector<float> z = rnd(g, E), zm = rnd(g, E);
    std::vector<uint32_t> cz(L * X, 0);
    for (uint32_t l = 0; l < L; ++l) { cz[l * X + 3] = 9; }
    CHECK(c.add(keys2(z, zm).data(), cz.data(), 0.9f) == 1 && c.size() == 2 && c.nearest(z.data(), 1, moe_topic_store::KEY_MEAN)[0].second == 1);
    c.max_records = 0;
    CHECK(c.add(keys2(b, bm).data(), cb.data(), NEVER) == 2 && c.size() == 3);

    std::remove(path.c_str());
    printf("test-moetopic-fork: OK\n");
    return 0;
}
