// Unit tests of the layer-major prompt plan (src/llama-layermajor-fork.h, fork/docs/layer-major-prompt.md): the layer groups and token chunks of a prompt call and
// its time against today's passes (the formulas of src/llama-passplan-fork.h). CPU only, no model:
//   groups   lm_groups                 : consecutive layers of one card; kept in the ring for all chunks when they fit, else streamed by every chunk
//   time     estimate_layermajor_time  : the call's time with k chunks, its parts (drives, link, compute, graphs)
// Numbers: Kimi K3.1 IQ1_S on PC1 (2 x 24 GiB, one drive; server logs of 2026-11-20), OLMoE-1B-7B Q4_K_M (its tensors) on the restricted machine of the first engine row.
//
// build: cmake --build <dir> --target test-layermajor-fork;  or alone: clang++ -std=c++17 -O1 tests/test-layermajor-fork.cpp -o /tmp/t && /tmp/t
#include "../src/llama-layermajor-fork.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

static int g_fail = 0;
static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[512]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static void check(const char * what, bool ok, const std::string & why = "") {
    printf("  %-96s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    g_fail += ok ? 0 : 1;
}

static const double MiB = 1048576.0, GiB = 1073741824.0;

// Kimi K3.1: 93 layers, layer 0 dense, 1-46 on CUDA0, 47-92 on CUDA1 (46 expert layers a card), 896 experts a layer, 16 picks a token. The expert slices add up to
// 507.9 GiB, 217.9 of them in the RAM pool, so 290 GiB a pass come from the drive (pool line; the 6.15 MiB of the log is the block of the common size class).
// The hidden state a group reads: the residual stream and the residual checkpoints banked every 12 layers (attn_res.block_size), 7168 floats each; layer 0 reads
// the token embeddings (no carry). Pools of 1056 blocks a card; free_b from the cap measured at 12558 cells with 6 entries a layer: 4160 tokens at 1.1248 MiB a token
// (the 10240-cell lend line 1.1115, 7.0764 at the context's end) beside one entry of 150 experts and two blocks (6.15 MiB each). Compute per token as the planner
// had it: measured 5814 us (581.4 a further entry), calculated 1651 us (165.1). One graph's host cost and the RAM the stream may keep its reads in are inputs the
// first rows measure (here 20 ms, 7.5 GiB: the pool's probation blocks of 2026-11-13).
static lm_in k31(uint32_t n, double tok_us, double ssd_gbs = 5.4) {
    lm_in in;
    in.n = n; in.cap = 16384; in.max_entries = 8;   // the stream's chunk counts go up to 8 (cs_cand)
    for (int l = 0; l < 93; ++l) {
        lm_layer x;
        x.card = l <= 46 ? 0 : 1;
        if (l > 0) {
            x.S = 507.9 * GiB / 92.0;
            x.M = (507.9 - 217.9) * GiB / 92.0;
            x.state_b = (1.0 + (double) ((l + 11) / 12)) * 7168.0 * 4.0;
        }
        in.layers.push_back(x);
    }
    const double entry = 150.0 * 6.15 * MiB + 2.0 * 6.15 * MiB;
    lm_card c0, c1;
    c0.lend_b = 1.1248 * MiB; c0.free_b = 4160.0 * c0.lend_b + entry; c0.link_gbs = 26.5;
    c1.lend_b = 1.0597 * MiB; c1.free_b = c0.free_b;                  c1.link_gbs = 26.6;
    in.cards = { c0, c1 };
    in.ssd_gbs = ssd_gbs; in.tok_us = tok_us; in.entry_us_tok = 0.1 * tok_us; in.graph_us = 20e3; in.hold_b = 7.5 * GiB;
    return in;
}

// OLMoE-1B-7B-0924 Q4_K_M: 16 layers, 64 experts of 8 picks; a layer's experts 249 MiB (down in Q6_K: layers 0 1 4 7 10 13 14 15) or 216 MiB, 3.633 GiB in all;
// 2048 floats of residual stream. The machine of the first engine row: one card with 1.2 GiB of pool beside the graph's 0.3 MiB a token, a 1 GiB RAM pool (the
// drive has the rest), 26.6 GB/s link, 5.4 GB/s drive, 60 us a token, 2 ms a graph. Mechanism only: no speed conclusion from OLMoE's dimensions.
static lm_in olmoe(uint32_t n, double ram_gib = 1.0, double free_gib = 1.2) {
    lm_in in;
    in.n = n; in.cap = 2048; in.max_entries = 8;
    const bool q6[16] = { 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1 };
    const double all = 3.633 * GiB, ram = std::min(all, ram_gib * GiB);
    for (int l = 0; l < 16; ++l) {
        lm_layer x;
        x.S = (q6[l] ? 249.0 : 216.0) * MiB;
        x.M = x.S * (1.0 - ram / all);
        x.state_b = l > 0 ? 2048.0 * 4.0 : 0.0;
        in.layers.push_back(x);
    }
    lm_card c;
    c.free_b = free_gib * GiB; c.lend_b = 0.3 * MiB; c.link_gbs = 26.6;
    in.cards = { c };
    in.ssd_gbs = 5.4; in.tok_us = 60.0; in.entry_us_tok = 6.0; in.graph_us = 2e3; in.hold_b = 1.0 * GiB;
    return in;
}

static int count_kind(const std::vector<lm_group> & gs, int kind) { int n = 0; for (const auto & g : gs) { n += g.kind == kind; } return n; }

static void test_groups() {
    printf("Testing the layer groups:\n");
    std::vector<lm_group> gs;
    {   // K3.1 at the 4186-token chunks of a 12558 prompt: a layer (5.52 GiB) is larger than a card's whole pool beside any graph, so every chunk streams it again
        const lm_in in = k31(12558, 5814.0);
        const bool ok = lm_groups(in, 4186, gs);
        bool cards_ok = true, cover = !gs.empty() && gs.front().a == 0 && gs.back().b == 93;
        for (size_t i = 0; i < gs.size(); ++i) {
            for (uint32_t l = gs[i].a; l < gs[i].b; ++l) { cards_ok = cards_ok && in.layers[l].card == gs[i].card; }
            if (i > 0) { cover = cover && gs[i].a == gs[i - 1].b; }
        }
        check("K3.1, 4186-token chunks: 92 groups of one expert layer, the dense layer 0 inside the first", ok && gs.size() == 92 && gs[0].a == 0 && gs[0].b == 2,
              fmt("%zu groups, first [%u, %u)", gs.size(), gs.empty() ? 0 : gs[0].a, gs.empty() ? 0 : gs[0].b));
        check("K3.1: every group streamed by every chunk (a layer > the pool), none read again (2 x 3.15 GiB in a 7.5 GiB hold)",
              count_kind(gs, LM_STREAMED) == 92 && [&] { for (const auto & g : gs) { if (g.reread) { return false; } } return true; }());
        check("K3.1: the groups cover the layers in order and never span the cards (46 | 47)", cover && cards_ok);
        check("K3.1: a streamed layer goes through the ring in entries: 7 on CUDA0 (906 MiB beside 4186 tokens), 5 on CUDA1",
              gs.size() == 92 && gs[1].entries == 7 && gs[50].entries == 5, fmt("%d %d", gs.size() > 1 ? gs[1].entries : -1, gs.size() > 50 ? gs[50].entries : -1));
        lm_in h = in;
        h.hold_b = 6.0 * GiB;
        lm_groups(h, 4186, gs);
        check("K3.1 with a 6 GiB hold: a layer's drive bytes and the next one's read ahead do not fit, every chunk reads them again",
              gs.size() == 92 && [&] { for (const auto & g : gs) { if (!g.reread) { return false; } } return true; }());
        check("K3.1: the most tokens beside a layer in 8 entries is the 8-chunk cap measured (4352), from the 6-chunk one (4160): 4352 run, 4400 do not",
              lm_groups(in, 4352, gs) && !lm_groups(in, 4400, gs));
    }
    {   // OLMoE: the ring beside the graph holds layers; with room for the next group beside it, the uploads of one group overlap the compute of the one before
        const lm_in in = olmoe(8192);
        lm_groups(in, 2048, gs);   // 1228.8 - 614.4 = 614.4 MiB: one layer and the next one's uploads (2 x 249), not two layers (2 x 498)
        check("OLMoE, 2048-token chunks: 16 groups of one layer, each kept on the card with the next one's beside it",
              gs.size() == 16 && count_kind(gs, LM_HELD2) == 16, fmt("%zu groups, %d HELD2", gs.size(), count_kind(gs, LM_HELD2)));
        lm_groups(in, 512, gs);    // 1075 MiB: two layers and the next two (2 x (249 + 249) = 996)
        check("OLMoE, 512-token chunks: 8 groups of two layers", gs.size() == 8 && count_kind(gs, LM_HELD2) == 8 && gs[0].b == 2 && gs[7].a == 14,
              fmt("%zu groups", gs.size()));
        const lm_in tight = olmoe(8192, 1.0, 0.9);   // 921.6 - 614.4 = 307 MiB: one layer fits (249), not with the next one beside it
        lm_groups(tight, 2048, gs);
        check("OLMoE, 0.9 GiB pool (307 MiB ring): every layer kept alone, the next group's uploads wait for its last chunk",
              gs.size() == 16 && count_kind(gs, LM_HELD) == 16, fmt("%zu groups, %d HELD", gs.size(), count_kind(gs, LM_HELD)));
        const lm_in pair = olmoe(8192, 1.0, (470.0 + 614.4) / 1024.0);   // a 470 MiB ring: 249 + 249 do not fit it, 249 + 216 do
        lm_groups(pair, 2048, gs);
        check("OLMoE, 470 MiB ring: a kept group takes the next layer while both fit the ring once (layers 1 and 2: 249 + 216)",
              gs.size() > 1 && gs[0].b == 1 && gs[1].a == 1 && gs[1].b == 3 && gs[1].kind == LM_HELD, fmt("[%u, %u) kind %d", gs.size() > 1 ? gs[1].a : 0, gs.size() > 1 ? gs[1].b : 0, gs.size() > 1 ? gs[1].kind : -1));
        const lm_in tiny = olmoe(8192, 1.0, 0.8);    // 819.2 - 614.4 = 205 MiB: no layer fits, every chunk streams them (two entries each)
        lm_groups(tiny, 2048, gs);
        check("OLMoE, 0.8 GiB pool: no layer fits beside 2048 tokens: streamed in two entries, groups grow while twice their drive bytes fit the 1 GiB hold",
              count_kind(gs, LM_STREAMED) == (int) gs.size() && !gs.empty() && gs[0].entries == 2 && gs.size() == 6,
              fmt("%zu groups, entries %d", gs.size(), gs.empty() ? 0 : gs[0].entries));
        lm_in none = in;
        for (auto & l : none.layers) { l.S = l.M = 0.0; }
        lm_groups(none, 2048, gs);
        check("OLMoE, every expert on the card already: nothing to stream, one group", gs.size() == 1 && gs[0].a == 0 && gs[0].b == 16);
    }
}

static void test_time() {
    printf("Testing the call's time:\n");
    {
        const lm_in in = k31(12558, 5814.0);
        lm_plan p;
        const double t = estimate_layermajor_time(in, 3, &p);
        const double ssd_once = 290.0 * GiB / 5.4e3;   // us
        double S = 0.0, carry = 0.0;
        for (uint32_t l = 1; l < 93; ++l) { S += in.layers[l].S / (in.layers[l].card == 0 ? 26.5e3 : 26.6e3); }
        for (const auto & g : p.groups) { carry += 12558.0 * in.layers[g.a].state_b / (in.cards[g.card].link_gbs * 1e3); }
        check("K3.1, 3 chunks: the drives give the 290 GiB once (57.7 s, a pass's worth)", t > 0.0 && std::fabs(p.ssd_us - ssd_once) < 1.0,
              fmt("%.1f s", p.ssd_us / 1e6));
        check("K3.1, 3 chunks: the link carries every expert once per chunk and each group's hidden state in",
              std::fabs(p.link_us - (3.0 * S + carry)) < 1.0, fmt("%.1f s (experts %.1f, hidden state %.1f)", p.link_us / 1e6, 3.0 * S / 1e6, carry / 1e6));
        check("K3.1, 3 chunks: 276 graphs", std::fabs(p.fixed_us - 276.0 * 20e3) < 1.0 && p.k == 3 && p.chunk == 4186, fmt("%.1f s", p.fixed_us / 1e6));
        check("a chunk past the cap or the pools cannot run", estimate_layermajor_time(in, 2) < 0.0 && estimate_layermajor_time(in, 0) < 0.0);
        lm_in r = in;
        r.hold_b = 6.0 * GiB;
        lm_plan pr;
        estimate_layermajor_time(r, 3, &pr);
        check("K3.1 with a 6 GiB hold: the drives read every layer again in every chunk (no gain from the drives)", std::fabs(pr.ssd_us - 3.0 * ssd_once) < 3.0);
    }
    {   // OLMoE kept on the card: every expert crosses the link once, its drive bytes once
        const lm_in in = olmoe(8192);
        lm_plan p;
        estimate_layermajor_time(in, 4, &p);
        double S = 0.0, M = 0.0;
        for (const auto & l : in.layers) { S += l.S; M += l.M; }
        check("OLMoE, 4 chunks kept on the card: the experts cross the link once and the drives once",
              std::fabs(p.link_us - (S / 26.6e3 + 15.0 * 8192.0 * 8192.0 / 26.6e3)) < 1.0 && std::fabs(p.ssd_us - M / 5.4e3) < 1.0,
              fmt("link %.0f ms, drives %.0f ms", p.link_us / 1e3, p.ssd_us / 1e3));
        const lm_in tight = olmoe(8192, 1.0, 0.9);
        lm_plan pt;
        estimate_layermajor_time(tight, 4, &pt);
        check("OLMoE alone in the ring: a group's uploads add to its compute (they wait for the group before)", pt.us > p.us, fmt("%.0f against %.0f ms", pt.us / 1e3, p.us / 1e3));
    }
}

int main() {
    test_groups();
    test_time();
    if (g_fail > 0) {
        printf("%d tests failed\n", g_fail);
        return 1;
    }
    printf("All tests passed\n");
    return 0;
}
