// Layer-major prompt (fork/docs/layer-major-prompt.md, --moe layer-major): a prompt call's tokens in k equal chunks run group by group, every chunk through a group
// of consecutive layers before the next group starts, so the experts come from the drives once per call instead of once per pass. The plan (the groups, the chunk
// count) and its time, priced beside today's equal passes with their own formulas (llama-passplan-fork.h). Pure functions: tests/test-layermajor-fork.cpp.
#pragma once
#include "llama-passplan-fork.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct lm_layer {           // one layer, in compute order
    int    card    = 0;     // the card that computes it; its link uploads the layer's experts
    double S       = 0.0;   // expert bytes the call uploads for it (no block holds them when the call starts); 0: no experts, or all of them on the card
    double M       = 0.0;   // of S, the bytes only the drives have (not in the RAM pool)
    double state_b = 0.0;   // bytes per token of the hidden state a group starting here reads (residual stream, hyper-connection copies, residual checkpoints)
};
struct lm_card {            // one card's pool at the call's end cells
    double free_b   = 0.0;  // bytes the lend and the ring share once the KV zone and the fixed reserves are out
    double lend_a   = 0.0;  // a graph of t tokens lends lend_a + lend_b x t (ppx_lend_line)
    double lend_b   = 0.0;
    double link_gbs = 0.0;  // host to card; 0: not known, its uploads count nothing
};
struct lm_in {
    uint32_t n = 0;               // tokens of the call
    uint32_t cap = 0;             // the most tokens a graph may have whatever the pools (micro-batch, mask budget); 0: none
    int      max_entries = 1;     // the most entries the stream cuts a layer into (cs_chunks_max)
    std::vector<lm_layer> layers;
    std::vector<lm_card>  cards;
    double ssd_gbs      = 0.0;    // the drives' rate; 0: not known, the reads count nothing
    double tok_us       = 0.0;    // one token's compute over all layers at the call's cells, every layer in one entry
    double entry_us_tok = 0.0;    // what one further entry in every layer adds to a token (ppx_pick_chunks' entry_us_tok)
    double graph_us     = 0.0;    // one graph's host cost: build, lend, inputs (an attention layer's mask), launch
    double hold_b       = 0.0;    // RAM the stream may keep its drive reads in (the RAM pool's free and probation blocks)
};
// LM_HELD2: the group stays in the ring for all chunks (uploaded once) and the next group's uploads fit beside it (they overlap this group's compute);
// LM_HELD: it stays, the next group's uploads wait for its last chunk; LM_STREAMED: larger than the ring, every chunk streams it again in entries
enum { LM_STREAMED = 0, LM_HELD = 1, LM_HELD2 = 2 };
struct lm_group {
    uint32_t a = 0, b = 0;      // layers [a, b), all on one card
    int  card    = 0;
    int  kind    = LM_HELD2;
    int  entries = 1;           // the most entries a layer of it is streamed in
    bool reread  = false;       // streamed, and its drive bytes cannot stay in RAM until its last chunk: every chunk reads them again
};
struct lm_plan {
    uint32_t k = 0, chunk = 0;  // chunks of the call and their size
    std::vector<lm_group> groups;
    double us = -1.0;           // the call; < 0: it cannot run
    double ssd_us = 0.0, link_us = 0.0, comp_us = 0.0, fixed_us = 0.0;   // the parts over all groups (a group takes the longest of its parts) and the graphs' host cost
};

// what a card's ring holds beside a graph of `chunk` tokens (<= 0: the graph does not fit)
static inline double lm_ring_b(const lm_card & c, uint32_t chunk) { return c.free_b - (c.lend_a + c.lend_b * (double) chunk); }

// The groups at chunk size `chunk`, consecutive layers of one card; a layer without bytes to stream joins the group it falls in. The first layer with experts
// decides the kind: held while the group fits the ring (HELD2 while twice it does, so the next group's uploads fit beside it), streamed when one layer is larger
// than the ring; a streamed group grows while twice its drive bytes fit the hold (its own reads stay to its last chunk, the next group's are read ahead beside them),
// and one whose drive bytes alone do not is read again by every chunk. False: a card cannot hold a graph of that size, or a layer needs more entries than the
// stream cuts.
static inline bool lm_groups(const lm_in & in, uint32_t chunk, std::vector<lm_group> & out) {
    out.clear();
    const uint32_t L = (uint32_t) in.layers.size();
    for (uint32_t i = 0; i < L; ) {
        const int card = in.layers[i].card;
        if (card < 0 || (size_t) card >= in.cards.size()) { return false; }
        const double R = lm_ring_b(in.cards[(size_t) card], chunk);
        if (R <= 0.0) { return false; }
        uint32_t j = i;
        while (j < L && in.layers[j].card == card && in.layers[j].S <= 0.0) { ++j; }   // the layers in front of the first one with experts
        lm_group g;
        g.a = i; g.card = card;
        if (j < L && in.layers[j].card == card) {
            const lm_layer & f = in.layers[j];
            g.kind = f.S <= 0.5 * R ? LM_HELD2 : f.S <= R ? LM_HELD : LM_STREAMED;
            if (g.kind == LM_STREAMED) {
                g.entries = (int) std::ceil(f.S / R);
                g.reread  = 2.0 * f.M > in.hold_b;
                if (g.entries > in.max_entries) { return false; }
            }
            double S = f.S, M = f.M;
            for (++j; j < L && in.layers[j].card == card; ++j) {
                const lm_layer & x = in.layers[j];
                if (x.S <= 0.0) { continue; }
                if (g.kind == LM_HELD2 && 2.0 * (S + x.S) > R) { break; }
                if (g.kind == LM_HELD && S + x.S > R) { break; }
                if (g.kind == LM_STREAMED) {
                    if (x.S <= R || g.reread || 2.0 * (M + x.M) > in.hold_b) { break; }
                    const int e = (int) std::ceil(x.S / R);
                    if (e > in.max_entries) { return false; }
                    g.entries = std::max(g.entries, e);
                }
                S += x.S; M += x.M;
            }
        }
        g.b = j;
        out.push_back(g);
        i = j;
    }
    return true;
}
