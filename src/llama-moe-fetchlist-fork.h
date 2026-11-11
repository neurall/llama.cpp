#pragma once
// The ordered fetch list of the layers ahead, from one layer's predictor logits (user 2026-11-11: at layer L the routers of L+1..L+n read L's
// activation; fetch in the order "L+1 picks, L+1 oversampled, L+2 picks, L+2 oversampled, ..."; only the mispredicted picks wait for the real router).
// Pure: no I/O, no cache state. Decode (one token) and prompt passes rank through this one function (tests/test-fetchlist-fork.cpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct moe_fetch_ent {
    int32_t depth;   // 0 = the next layer
    int32_t e;       // expert
    int32_t rank;    // 0 = a predicted pick (within the top k_pick of some token), 1 = oversampled (the m_over after them)
    int32_t votes;   // tokens that put it there
};

// x: [n_tok][n_blk][ne] logits (the predictor's stacked layout), n_use <= n_blk depths are listed.
// bias: null, or n_use vectors (empty = raw logits; else sigmoid(logit) + bias[e], as the learned predictors score).
// cached(depth, e): true = no fetch needed. budget: the list is cut there (the deepest layers go first, they come last).
// Inside one rank: most tokens first, then the higher score; one token gives plain score order.
template <class Cached>
static inline std::vector<moe_fetch_ent> moe_fetch_list(const float * x, int64_t n_tok, int64_t n_blk, int64_t n_use, int64_t ne,
        const std::vector<float> * bias, int32_t k_pick, int32_t m_over, Cached cached, size_t budget) {
    std::vector<moe_fetch_ent> out;
    const int32_t k = (int32_t) std::min<int64_t>(std::max(k_pick, 0), ne);
    const int32_t m = (int32_t) std::min<int64_t>((int64_t) k + std::max(m_over, 0), ne);
    if (m <= 0) { return out; }
    std::vector<float>   lg(ne), best(ne);
    std::vector<int32_t> idx(ne), vp(ne), vo(ne), ord;
    for (int64_t d = 0; d < std::min(n_use, n_blk) && out.size() < budget; ++d) {
        const std::vector<float> * b = bias && !bias[d].empty() ? &bias[d] : nullptr;
        std::fill(vp.begin(), vp.end(), 0);
        std::fill(vo.begin(), vo.end(), 0);
        std::fill(best.begin(), best.end(), -INFINITY);
        for (int64_t t = 0; t < n_tok; ++t) {
            const float * src = x + (t*n_blk + d)*ne;
            for (int32_t e = 0; e < ne; ++e) { lg[e] = b ? 1.0f/(1.0f + expf(-src[e])) + (*b)[e] : src[e]; idx[e] = e; }
            std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), [&](int32_t a, int32_t c) { return lg[a] > lg[c] || (lg[a] == lg[c] && a < c); });
            for (int32_t j = 0; j < m; ++j) {
                (j < k ? vp : vo)[idx[j]]++;
                best[idx[j]] = std::max(best[idx[j]], lg[idx[j]]);
            }
        }
        for (int32_t rank = 0; rank < 2; ++rank) {
            ord.clear();
            for (int32_t e = 0; e < ne; ++e) {
                if (rank == 0 ? vp[e] > 0 : vp[e] == 0 && vo[e] > 0) { ord.push_back(e); }
            }
            const std::vector<int32_t> & v = rank == 0 ? vp : vo;
            std::sort(ord.begin(), ord.end(), [&](int32_t a, int32_t c) { return v[a] != v[c] ? v[a] > v[c] : best[a] != best[c] ? best[a] > best[c] : a < c; });
            for (int32_t e : ord) {
                if (out.size() >= budget) { break; }
                if (!cached((int32_t) d, e)) { out.push_back({ (int32_t) d, e, rank, v[e] }); }
            }
        }
    }
    return out;
}
