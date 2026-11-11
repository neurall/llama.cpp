// The predictor's ordered fetch list (src/llama-moe-fetchlist-fork.h): rank order, vote order, one entry per expert, cached entries dropped, the budget cut.
// CPU only, no model.
#include "../src/llama-moe-fetchlist-fork.h"

#include <cassert>
#include <cstdio>

int main() {
    // 3 tokens, 2 depths, 6 experts; x[t][d][e]
    const int64_t nt = 3, nb = 2, ne = 6;
    std::vector<float> x(nt*nb*ne, 0.0f);
    auto at = [&](int64_t t, int64_t d, int64_t e) -> float & { return x[(t*nb + d)*ne + e]; };
    // depth 0: tokens 0 and 1 pick expert 4 first, token 2 picks expert 1; second choices 2, 2, 3
    at(0, 0, 4) = 9; at(0, 0, 2) = 5;
    at(1, 0, 4) = 8; at(1, 0, 2) = 4;
    at(2, 0, 1) = 7; at(2, 0, 3) = 6;
    // depth 1: every token picks expert 0, second choice 5
    for (int64_t t = 0; t < nt; ++t) { at(t, 1, 0) = 3; at(t, 1, 5) = 2; }
    auto none = [](int32_t, int32_t) { return false; };

    auto l = moe_fetch_list(x.data(), nt, nb, nb, ne, nullptr, 1, 1, none, 100);
    // depth 0 picks {4 x2, 1 x1}, oversampled {2 x2, 3 x1}; depth 1 picks {0 x3}, oversampled {5 x3}
    const int32_t want[][3] = { {0, 4, 0}, {0, 1, 0}, {0, 2, 1}, {0, 3, 1}, {1, 0, 0}, {1, 5, 1} };
    assert(l.size() == 6);
    for (size_t i = 0; i < l.size(); ++i) {
        assert(l[i].depth == want[i][0] && l[i].e == want[i][1] && l[i].rank == want[i][2]);
    }
    assert(l[0].votes == 2 && l[4].votes == 3);

    // an expert that is a pick of one token and oversampled of another is listed once, as a pick
    at(2, 0, 4) = 6.5f;   // token 2: 1 (7), then 4 (6.5)
    l = moe_fetch_list(x.data(), nt, nb, 1, ne, nullptr, 1, 1, none, 100);
    assert(l.size() == 3 && l[0].e == 4 && l[0].rank == 0 && l[1].e == 1 && l[2].e == 2 && l[2].rank == 1);
    at(2, 0, 4) = 0;

    // cached entries are dropped, the order of the rest stays
    l = moe_fetch_list(x.data(), nt, nb, nb, ne, nullptr, 1, 1, [](int32_t d, int32_t e) { return d == 0 && e == 4; }, 100);
    assert(l.size() == 5 && l[0].e == 1 && l[1].e == 2);

    // the budget cuts the deepest entries
    l = moe_fetch_list(x.data(), nt, nb, nb, ne, nullptr, 1, 1, none, 4);
    assert(l.size() == 4 && l.back().depth == 0);

    // one token (decode): plain score order, picks only
    l = moe_fetch_list(x.data(), 1, nb, 1, ne, nullptr, 2, 0, none, 100);
    assert(l.size() == 2 && l[0].e == 4 && l[1].e == 2 && l[0].rank == 0 && l[1].rank == 0);

    // a bias switches to sigmoid(logit) + bias: expert 3's bias lifts it above expert 4
    std::vector<float> bias[1] = { std::vector<float>(ne, 0.0f) };
    bias[0][3] = 2.0f;
    l = moe_fetch_list(x.data(), 1, nb, 1, ne, bias, 1, 0, none, 100);
    assert(l.size() == 1 && l[0].e == 3);

    // more wanted than experts: no overrun
    l = moe_fetch_list(x.data(), nt, nb, nb, ne, nullptr, 16, 16, none, 100);
    assert(l.size() == 12);

    printf("test-fetchlist-fork: ok\n");
    return 0;
}
