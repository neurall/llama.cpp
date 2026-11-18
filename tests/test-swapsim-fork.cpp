// test-swapsim-fork: the decode cache's swap rules as a replay model (src/llama-swapsim-fork.h), standalone.
//   build: g++ -std=c++17 -O1 -I src tests/test-swapsim-fork.cpp -o test-swapsim-fork
// The rules are the engine's (src/llama-moecache-fork.cpp: the step's swap loop and the upload inside the token); each check names the rule it holds.
#include "llama-swapsim-fork.h"

#include <cstdio>
#include <random>
#include <string>

static int g_fail = 0;
static void check(const char * what, bool ok) {
    if (!ok) { printf("FAIL: %s\n", what); g_fail++; }
}

// a cache of `layers` layers of n experts in one class; the first `held` experts of each layer cached
static swapsim make(int layers, int n, int held, double S = 100.0) {
    swapsim s;
    for (int l = 0; l < layers; ++l) {
        s.add_layer(n, 0, 0, S);
        for (int e = 0; e < held; ++e) { s.L.back().in[e] = 1; }
    }
    s.link_bytes_us = { 1.0 }; s.cpu_bytes_us = 1.0; s.margin = { 0 };
    return s;
}

int main() {
    {   // between tokens: a free block is taken without an eviction; an expert seen once does not pass ADMIT 2
        swapsim s = make(1, 8, 0); s.k.jit = 0; s.free_blocks[0] = 2;
        s.token({ { 0, 1 } });
        check("admit: one use in the window takes no block", s.out.d_admit == 2 && s.out.bg_bytes == 0 && s.free_blocks[0] == 2);
        s.token({ { 0, 1 } });
        check("free blocks: taken without evicting", s.out.evictions == 0 && s.out.bg_bytes == 200 && s.free_blocks[0] == 0 && s.L[0].coming[0] && s.L[0].coming[1]);
        const uint64_t hits = s.out.hits;
        s.token({ { 0, 1 } });
        check("an upload queued between tokens serves from the next token", s.out.hits == hits + 2 && s.L[0].in[0] && !s.L[0].coming[0]);
    }
    {   // the pay-back margin between tokens: the newcomer needs the victim's score plus the margin
        for (int margin : { 0, 1 }) {
            swapsim s = make(1, 4, 2); s.k.jit = 0; s.k.admit = 1; s.k.gweight = 0; s.margin = { margin };
            s.token({ { 0 } }); s.token({ { 0 } }); s.token({ { 1 } });   // uses: 0 twice, 1 once
            s.token({ { 2 } });                                          // 2 once: as hot as the coldest cached (1)
            if (margin == 0) { check("margin 0: an expert as hot as the victim swaps in", s.out.evictions == 1 && s.L[0].coming[2] && !s.L[0].in[1] && s.out.bg_bytes == 100); }
            else             { check("margin 1: it does not, and the decline is counted", s.out.evictions == 0 && !s.L[0].coming[2] && s.L[0].in[1] && s.out.d_margin == 1); }
        }
    }
    {   // the budget: evictions a token over all layers
        swapsim s = make(2, 4, 1); s.k.jit = 0; s.k.admit = 1; s.k.gweight = 0; s.k.budget = 1;
        s.token({ { 1 }, { 1 } });   // each layer misses expert 1; the cached expert 0 was never used (score 0)
        check("budget 1: one eviction, the other candidate waits", s.out.evictions == 1 && s.out.d_budget == 1 && s.out.bg_bytes == 100);
    }
    {   // block pool: the victim is the class's coldest, whichever layer holds it
        swapsim s = make(2, 4, 1); s.k.jit = 0; s.k.admit = 1; s.k.gweight = 0;
        s.token({ { 0 }, {} }); s.token({ { 0 }, {} });   // layer 0's cached expert is used, layer 1's is not
        s.token({ { 1 }, {} });                           // layer 0 wants a second expert
        check("pool: a layer takes the block of another layer's colder expert", s.L[0].coming[1] && s.L[0].in[0] && !s.L[1].in[0] && s.out.evictions == 1);
    }
    {   // inside the token: the split by the time model, and the upload's own victim rule
        swapsim s = make(1, 8, 0); s.free_blocks[0] = 8; s.k.admit = 99;
        s.token({ { 0, 1, 2, 3 } });   // link as fast as the CPU: half and half finishes first
        check("time model: equal rates split the misses in two", s.out.jit_bytes == 200 && s.out.cpu_bytes == 200 && s.out.us == 200);
        swapsim f = make(1, 8, 0); f.free_blocks[0] = 8; f.k.admit = 99; f.link_bytes_us = { 100.0 };
        f.token({ { 0, 1, 2, 3 } });
        check("time model: a link 100x the CPU uploads every miss", f.out.jit_bytes == 400 && f.out.cpu_bytes == 0);
        swapsim c = make(1, 8, 0); c.free_blocks[0] = 8; c.k.admit = 99; c.link_bytes_us = { 0.01 };
        c.token({ { 0, 1, 2, 3 } });
        check("time model: a link 100x slower leaves every miss with the CPU", c.out.jit_bytes == 0 && c.out.cpu_bytes == 400);
        swapsim u = make(1, 4, 4); u.link_bytes_us = { 100.0 }; u.k.admit = 99;
        u.token({ { 0, 1, 2, 3 } });   // everything cached is in use by this token... and all hit
        check("a token whose experts are all cached costs nothing", u.out.us == 0 && u.out.hits == 4);
        for (int margin : { 0, 3 }) {
            swapsim m = make(1, 6, 2); m.link_bytes_us = { 100.0 }; m.k.admit = 99; m.k.gweight = 0; m.margin = { margin };
            m.token({ { 0 } }); m.token({ { 0 } });   // 0 used twice, 1 never
            m.token({ { 0, 2 } });                    // 2 misses: its victim can only be 1 (0 is in use)
            if (margin == 0) { check("in-token upload, margin 0: the unused expert gives its block", m.L[0].in[2] && !m.L[0].in[1] && m.out.jit_bytes == 100); }
            else             { check("in-token upload, margin 3: one use does not beat the victim by 3, the CPU computes it", !m.L[0].in[2] && m.L[0].in[1] && m.out.cpu_bytes == 100 && m.out.d_margin == 1); }
        }
        swapsim j2 = make(1, 6, 2); j2.link_bytes_us = { 100.0 }; j2.k.admit = 99; j2.k.gweight = 0; j2.margin = { 3 }; j2.k.jit = 2;
        j2.token({ { 0 } }); j2.token({ { 0, 2 } });
        check("jit 2: any miss takes the coldest block, margin or not", j2.L[0].in[2] && !j2.L[0].in[1]);
    }
    {   // equal margins replay alike; the comparison counts only what follows the settle tokens
        std::mt19937 rng(7);
        std::vector<std::vector<std::vector<int32_t>>> toks;
        for (int t = 0; t < 300; ++t) { toks.push_back({ { (int32_t) (rng() % 16), (int32_t) (rng() % 16) }, { (int32_t) (rng() % 16) } }); }
        swapsim s = make(2, 16, 4);
        const swapsim_verdict same = swapsim_compare(s, toks, { 0 }, { 0 });
        check("equal margins: equal replays", same.us_a == same.us_b && same.a.evictions == same.b.evictions && same.gain == 0.0);
        const swapsim_verdict all = swapsim_compare(s, toks, { 0 }, { 2 }), tail = swapsim_compare(s, toks, { 0 }, { 2 }, swapsim_cost(), 150);
        check("settle: the first tokens count for neither arm", all.a.tokens == 300 && tail.a.tokens == 300 && tail.us_a != all.us_a);
    }
    {   // the cost fit recovers what the bytes cost a token
        std::mt19937 rng(3); std::normal_distribution<double> noise(0.0, 300.0);
        std::vector<double> t, c, j, b;
        for (int i = 0; i < 400; ++i) {
            const double cb = (double) (rng() % 40) * 1e6, jb = (double) (rng() % 30) * 1e6, bb = (double) (rng() % 20) * 1e6;
            c.push_back(cb); j.push_back(jb); b.push_back(bb);
            t.push_back(20000.0 + 2e-4 * cb + 5e-5 * jb + 1e-4 * bb + noise(rng));
        }
        const swapsim_cost w = swapsim_fit(t, c, j, b);
        auto near = [](double x, double want) { return std::fabs(x - want) < 0.1 * want; };
        check("cost fit: the three costs and the base time within 10%", w.fitted && near(w.cpu, 2e-4) && near(w.jit, 5e-5) && near(w.bg, 1e-4) && near(w.base, 20000.0));
        check("cost fit: too few tokens fit nothing", !swapsim_fit({ 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 }).fitted);
        std::vector<double> zero(t.size(), 0.0);
        const swapsim_cost w0 = swapsim_fit(t, c, zero, b);
        check("cost fit: bytes that never occur cost 0, the others still fit", w0.fitted && w0.jit == 0.0 && near(w0.cpu, 2e-4));
    }
    {   // the verdict follows what this machine pays. 4 layers of 64 experts, 24 blocks a layer, 32 experts a layer about equally hot (6 of them a token): more than fit,
        // and none better than another. An eager margin keeps trading equals (every trade an upload, no hit gained); a strict one holds what it has. Which is cheaper
        // is the price of an upload against a miss on the CPU, and nothing else.
        auto picks = [](unsigned seed) {
            std::mt19937 rng(seed);
            std::vector<std::vector<std::vector<int32_t>>> toks;
            for (int t = 0; t < 1200; ++t) {
                std::vector<std::vector<int32_t>> tok(4);
                for (int l = 0; l < 4; ++l) { for (int i = 0; i < 6; ++i) { tok[l].push_back((int32_t) (rng() % 32)); } }
                toks.push_back(tok);
            }
            return toks;
        };
        int eager_cheap = 0, strict_dear = 0, swaps_fewer = 0; double hit_a = 0, hit_b = 0, ev_a = 0, ev_b = 0;
        for (unsigned seed = 1; seed <= 20; ++seed) {
            swapsim s = make(4, 64, 24, 8e6); s.link_bytes_us = { 26.7e3 }; s.cpu_bytes_us = 20e3;
            const auto toks = picks(seed);
            swapsim_cost cheap; cheap.fitted = true; cheap.cpu = 5e-5;  cheap.jit = 1e-5; cheap.bg = 0.0;     // an upload costs a fifth of computing the expert on the CPU
            swapsim_cost dear;  dear.fitted  = true; dear.cpu  = 5e-5;  dear.jit  = 4e-4; dear.bg = 4e-4;    // an upload costs eight times it
            const swapsim_verdict vc = swapsim_compare(s, toks, { 0 }, { 3 }, cheap, 200), vd = swapsim_compare(s, toks, { 0 }, { 3 }, dear, 200);
            eager_cheap += !vc.b_wins; strict_dear += vd.b_wins;
            swaps_fewer += vc.b.evictions < vc.a.evictions;
            hit_a += (double) vc.a.hits / vc.a.picks; hit_b += (double) vc.b.hits / vc.b.picks; ev_a += (double) vc.a.evictions / vc.a.tokens; ev_b += (double) vc.b.evictions / vc.b.tokens;
        }
        printf("swap replay, 32 equally hot experts for 24 blocks, 20 seeds: margin 0 hit %.1f%% with %.2f evictions a token, margin 3 hit %.1f%% with %.2f; "
               "cheaper: margin 0 with cheap uploads in %d, margin 3 with dear uploads in %d\n", 5 * hit_a, ev_a / 20, 5 * hit_b, ev_b / 20, eager_cheap, strict_dear);
        check("verdict: cheap uploads keep the eager margin", eager_cheap >= 19);
        check("verdict: dear uploads take the strict margin", strict_dear >= 19);
        check("a strict margin evicts less on the same picks", swaps_fewer == 20);
    }
    check("trusted: the replay within a quarter of the engine's counters", swapsim_trusted(100, 110, 50, 45, 1000) && !swapsim_trusted(100, 200, 50, 45, 1000) && !swapsim_trusted(100, 110, 50, 20, 1000));
    check("trusted: next to nothing on both sides agrees", swapsim_trusted(100, 110, 2, 9, 1000));
    printf(g_fail ? "swapsim: %d FAILED\n" : "swapsim: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
