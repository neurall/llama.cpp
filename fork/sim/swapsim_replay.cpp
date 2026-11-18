// swapsim_replay TRACE [--warm TRACE2] [--share 0.2,0.35,0.5] [--budget 4,16,64] [--link 26.7] [--cpu 133.5] [--mib 7.88] [--margin 5] [--settle 200]
//   build: g++ -std=c++17 -O2 -I src fork/sim/swapsim_replay.cpp -o swapsim_replay
// Cache policies on a recorded routing trace (GGML_MOE_LOG lines "<tensor> id id ..."), through the engine's own swap rules (src/llama-swapsim-fork.h):
//   add   the default score, uses in the last 64 tokens + 16 x lifetime share, pay-back margin --margin
//   crf   ClassLRU (upstream llama.cpp's cache, #31233; the engine's --moe policy=crf): 2^class of a use count halving every 128 tokens / age, no margin
//   stay  the three last-10 rings (ggml-hist-fork.h; user 2026-11-14 / 2026-11-18: last 10 recalls, reads served in each of the last 10 stays, last 10 evictions):
//         the reads a stay is expected to serve; the newcomer must expect more than the victim
//   rate  the recall ring alone: uses a token over the span of the last 10 recalls, up to now;  rate+stay: and an expert whose stays did not pay needs more to come back
//   lru, slru   least recently used; upstream's cache before ClassLRU (#30637): a hit protects, half of a layer's blocks protected
//   the budget between tokens spent by layer in turn (the engine); --spend 0,1 adds best first over all layers
// (user 2026-11-18: "read upstreams moe eviction and hot cold strategies ... test if some usable new strategy there").
// The cache: one block pool over all layers, --share of all experts; rates in GB/s, an expert --mib MiB. Decode tokens are replayed, prompt tokens only counted.
// --warm: an earlier session on that trace under the same rules: its lifetime counts and rings are what the state file would hold, and the pool starts filled with
// its most used experts, as the engine preloads (none: an empty pool, no counts).
// The trace given as its own --warm is a known topic, another trace an unknown one. Reported after --settle decode tokens.
#include "llama-swapsim-fork.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

struct tok { bool decode; std::vector<std::vector<int32_t>> picks; };

static std::vector<tok> load(const std::string & path, std::map<int, int> & lay, int & n_exp) {
    std::vector<tok> out;
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    std::vector<std::pair<int, std::vector<int32_t>>> graph;   // (model layer, ids), in file order
    auto flush = [&]() {
        if (graph.empty()) { return; }
        std::map<int, std::vector<std::vector<int32_t>>> by;   // layer -> its lines (one per token of the batch)
        for (auto & g : graph) { by[g.first].push_back(g.second); }
        size_t n = 0;
        for (auto & b : by) { n = std::max(n, b.second.size()); if (!lay.count(b.first)) { const int k = (int) lay.size(); lay[b.first] = k; } }
        for (size_t t = 0; t < n; ++t) {
            tok x; x.decode = n == 1; x.picks.resize(lay.size());
            for (auto & b : by) { if (t < b.second.size()) { x.picks[lay[b.first]] = b.second[t]; } }
            out.push_back(std::move(x));
        }
        graph.clear();
    };
    std::string line;
    int last = -1;
    while (std::getline(f, line)) {
        const size_t p = line.find("blk.");
        if (p == std::string::npos) { continue; }
        const int layer = atoi(line.c_str() + p + 4);
        if (layer < last) { flush(); }   // back at an earlier layer: the next graph
        last = layer;
        std::istringstream is(line);
        std::string name; is >> name;
        std::vector<int32_t> ids; int32_t v;
        while (is >> v) { ids.push_back(v); n_exp = std::max(n_exp, v + 1); }
        graph.push_back({ layer, ids });
    }
    flush();
    return out;
}

static std::vector<double> nums(const char * s) { std::vector<double> v; std::stringstream ss(s); std::string x; while (std::getline(ss, x, ',')) { v.push_back(atof(x.c_str())); } return v; }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: swapsim_replay TRACE [--warm TRACE2] [--share ..] [--budget ..] [--link G] [--cpu G] [--mib M] [--margin N] [--settle N]\n"); return 1; }
    std::string trace = argv[1], warm;
    std::vector<double> shares = { 0.2, 0.35, 0.5 }, budgets = { 4, 16, 64 }, spends = { 0 };
    std::vector<int> policies = { 0, 1, 2, 3, 4, 5, 6 }; double admit = 2;
    double link = 26.7, cpu = 133.5, mib = 7.88; int margin = 5; size_t settle = 200;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--warm") { warm = argv[i + 1]; } else if (a == "--share") { shares = nums(argv[i + 1]); } else if (a == "--budget") { budgets = nums(argv[i + 1]); }
        else if (a == "--link") { link = atof(argv[i + 1]); } else if (a == "--cpu") { cpu = atof(argv[i + 1]); } else if (a == "--mib") { mib = atof(argv[i + 1]); }
        else if (a == "--spend") { spends = nums(argv[i + 1]); }
        else if (a == "--admit") { admit = atof(argv[i + 1]); }
        else if (a == "--policy") { policies.clear(); for (double v : nums(argv[i + 1])) { policies.push_back((int) v); } }
        else if (a == "--margin") { margin = atoi(argv[i + 1]); } else if (a == "--settle") { settle = (size_t) atoi(argv[i + 1]); }
    }
    std::map<int, int> lay; int n_exp = 0;
    std::vector<tok> w; if (!warm.empty()) { w = load(warm, lay, n_exp); }
    const std::vector<tok> toks = load(trace, lay, n_exp);
    size_t n_dec = 0; for (auto & t : toks) { n_dec += t.decode; }
    printf("%s: %zu decode tokens, %zu prompt tokens, %zu layers of %d experts; map: %s; link %.1f GB/s, CPU %.1f GB/s, expert %.2f MiB, margin %d (add), after %zu tokens\n",
           trace.c_str(), n_dec, toks.size() - n_dec, lay.size(), n_exp, warm.empty() ? "none (empty pool)" : warm == trace ? "this trace (known topic)" : "another trace (unknown topic)",
           link, cpu, mib, margin, settle);
    static const char * const names[] = { "add", "crf", "stay", "rate", "rate+stay", "lru", "slru" };
    printf("%-9s %-7s %-9s %-11s %-8s %-18s %-18s %-20s %-22s %s\n", "pool (%)", "budget", "score", "spend", "hit (%)", "CPU (MiB/token)", "upload (MiB/token)", "evictions per token", "miss time (ms/token)", "declined per token: margin / budget / admit");
    for (double share : shares) {
        for (double bud : budgets) {
            for (int policy : policies) {
                for (double spend_d : spends) {
                    const int spend = (int) spend_d;
                    swapsim s;
                    const double S = mib * 1048576.0;
                    for (size_t l = 0; l < lay.size(); ++l) { s.add_layer(n_exp, 0, 0, S); }
                    s.link_bytes_us = { link * 1e3 }; s.cpu_bytes_us = cpu * 1e3; s.margin = { margin };
                    s.k.policy = policy; s.k.spend = spend; s.k.budget = (int) bud; s.k.admit = admit; s.k.payback = std::max(1.0, cpu / link);
                    const int blocks = (int) (share * (double) lay.size() * n_exp);
                    s.free_blocks[0] = blocks;
                    if (!w.empty()) {
                        for (auto & t : w) { if (t.decode) { s.token(t.picks); } else { s.observe(t.picks); } }   // the earlier session
                        // the new start: the pool holds the most used experts of the stored counts; the window and the halving count start empty, the rings stay
                        std::vector<std::pair<uint32_t, std::pair<size_t, int>>> all;
                        for (size_t l = 0; l < s.L.size(); ++l) { for (int e = 0; e < n_exp; ++e) { all.push_back({ s.L[l].glob[e], { l, e } }); } }
                        std::stable_sort(all.begin(), all.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                        std::vector<std::vector<uint8_t>> want(s.L.size(), std::vector<uint8_t>(n_exp, 0));
                        for (int i = 0; i < blocks && i < (int) all.size(); ++i) { if (all[i].first > 0) { want[all[i].second.first][all[i].second.second] = 1; } }
                        int held = 0;
                        for (size_t li = 0; li < s.L.size(); ++li) {
                            swapsim_layer & l = s.L[li];
                            for (int e = 0; e < n_exp; ++e) {
                                const bool in = l.in[e] || l.coming[e];
                                if (in && !want[li][e]) { hist_evict(l.h[e], s.clock); }
                                if (!in && want[li][e]) { hist_admit(l.h[e]); }
                                l.in[e] = want[li][e]; l.coming[e] = 0; held += want[li][e];
                                l.win[e] = 0; l.crf[e] = 0; l.crf_t[e] = 0;
                            }
                            l.recent.clear(); l.tok = 0;
                        }
                        s.free_blocks[0] = blocks - held;
                        s.out = swapsim_out();
                    }
                    swapsim_out o0; size_t d = 0;
                    for (auto & t : toks) {
                        if (!t.decode) { s.observe(t.picks); continue; }
                        if (d++ == settle) { o0 = s.out; }
                        s.token(t.picks);
                    }
                    const swapsim_out & o = s.out;
                    const double n = (double) std::max<uint64_t>(1, o.tokens - o0.tokens);
                    printf("%-9.0f %-7.0f %-9s %-11s %-8.1f %-18.1f %-18.1f %-20.2f %-22.2f %.2f / %.2f / %.2f\n", 100 * share, bud, names[policy], spend ? "best first" : "by layer",
                           100.0 * (double) (o.hits - o0.hits) / (double) std::max<uint64_t>(1, o.picks - o0.picks), (o.cpu_bytes - o0.cpu_bytes) / n / 1048576.0,
                           (o.jit_bytes + o.bg_bytes - o0.jit_bytes - o0.bg_bytes) / n / 1048576.0, (double) (o.evictions - o0.evictions) / n, (o.us - o0.us) / n / 1e3,
                           (double) (o.d_margin - o0.d_margin) / n, (double) (o.d_budget - o0.d_budget) / n, (double) (o.d_admit - o0.d_admit) / n);
                }
            }
        }
    }
    return 0;
}
