// Unit tests of the prompt planner's strategy estimators (src/llama-estimate-fork.h): each strategy's time from hand-made metrics. CPU only, no model.
#include "../src/llama-estimate-fork.h"
#include "../src/llama-passplan-fork.h"
#include "../common/cpu-topo-fork.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>

static int g_fail = 0;
static void near(const char * what, double got, double want) {
    if (std::fabs(got - want) > 1e-6 * std::max(1.0, std::fabs(want))) { printf("FAIL %s: got %.6f, want %.6f\n", what, got, want); g_fail++; }
}

int main() {
    // 100 tokens, 10 us compute per token; all times in us, SSD 2 GB/s = 2000 bytes per us
    pp_metrics m;
    m.n = 100.0; m.comp_us_tok = 10.0; m.ssd_gbs = 2.0;
    m.U = 4000.0; m.M = 20e6;      // 4 ms over the link, 20 MB only the SSD has: 10 ms
    m.Um = 3000.0; m.Mm = 2e6;     // the mix: 3 ms over the link, 1 ms from the SSD
    m.up = 500.0; m.cpu = 1500.0; m.ssd_miss = 200.0; m.wait_us = 300.0;
    m.chunk_us = 50.0;
    const double Uk[2] = { 800.0, 1600.0 }, Mk[2] = { 4e6, 1e6 };
    m.Uk = Uk; m.Mk = Mk;

    {   // layer parallel streams its card's layers in bulk: the link's probed rate, not the per-expert average of decode's uploads (Vulkan 2026-11-17: 2 against
        // 22 GB/s, layer parallel priced at 39.6 s for an 11.07 s pass, and its x0.28 miss made the never-run plain stream look like 4.2 s: it took 14.5)
        pp_sums s(0, 2);
        pp_sums_add(s, 100.0, 22e3, 0.0, false, false, 0.5, 22e3, 2e3, 22e3, 1e3, 1);   // 22 kB: bulk 22e3 bytes/us, one expert at a time 2e3
        near("layer parallel upload at the link's bulk rate", s.Ul[1], 1.0);
        near("a miss at the per-expert rate", s.miss.empty() ? -1.0 : s.miss[0].us, 11.0);
    }
    near("ssd", pp_ssd_us(m, 2e6), 1000.0);
    near("miss on gpu: the link is slower", estimate_moe_miss_ongpu_time(m), 500.0);
    near("miss on cpu", estimate_moe_miss_oncpu_time(m), 1500.0);
    near("full stream upload: the SSD is slower", estimate_fullstream_upload_time(m), 10000.0);
    near("full stream per token: the upload hides the compute", estimate_fullstream_time_per_token(m), 100.0);
    near("mix upload: the link is slower", estimate_mix_upload_time(m), 3000.0);
    near("mix per token", estimate_mix_time_per_token(m), 30.0);
    near("router per token: misses + wait + compute", estimate_router_time_per_token(m), (500.0 + 300.0 + 1000.0) / 100.0);
    near("decode chain: side by side + compute + ssd", estimate_decodechain_time(m), 500.0 * 1500.0 / 2000.0 + 1000.0 + 200.0);
    near("decode chain per token", estimate_decodechain_time_per_token(m), (375.0 + 1000.0 + 200.0) / 100.0);
    near("predictor x1: the SSD is slower, late factor", estimate_predictor_upload_time(m, 0, 1.5), 1.5 * 2000.0);
    near("predictor x2: the link is slower", estimate_predictor_upload_time(m, 1, 1.0), 1600.0);
    near("predictor per token: never under the compute", estimate_predictor_time_per_token(m, 1, 0.5), 10.0);

    // a slow SSD turns the misses' time over to it, and with it the router path
    m.ssd_miss = 900.0;
    near("miss on gpu: the SSD is slower", estimate_moe_miss_ongpu_time(m), 900.0);
    near("router per token with it", estimate_router_time_per_token(m), (900.0 + 300.0 + 1000.0) / 100.0);
    // the decode chain's learned parts: fixed, misses ratio, and chunks that pass the compute
    m.dec_f = 100.0; m.dec_r = 2.0; m.dec_kc = 30.0;
    near("decode chain learned", estimate_decodechain_time(m), 100.0 + 375.0 * 2.0 + 1500.0 + 900.0);
    // no SSD rate: the link alone; no compute measured: the upload alone
    m.ssd_gbs = 0.0; m.comp_us_tok = 0.0;
    near("full stream without an SSD rate", estimate_fullstream_time_per_token(m), 40.0);
    // layer parallel: each card's turn is its own uploads (its link or the SSD) or its own compute, whichever is longer, the turns in a row
    {
        pp_metrics l; l.n = 100.0; l.comp_us_tok = 10.0; l.ssd_gbs = 2.0;
        l.Ul = { 3000.0, 1000.0 }; l.Ml = { 8e6, 0.0 }; l.share = { 0.5, 0.5 };   // card 0: 3 ms over its link, 4 ms from the SSD; card 1: 1 ms
        near("layer parallel upload: each card's link or the SSD", estimate_layerparallel_upload_time(l), 4000.0 + 1000.0);
        near("layer parallel: each turn its longest part", estimate_layerparallel_time(l), 4000.0 + 1000.0);
        l.comp_us_tok = 100.0;   // 10 ms of compute, 5 ms a turn
        near("layer parallel compute-bound: the turns add up to the compute", estimate_layerparallel_time(l), 5000.0 + 5000.0);
        near("layer parallel per token", estimate_layerparallel_time_per_token(l), 100.0);
    }
    // the crossover the planner looks for: a short pass favours the router, a long one the stream
    pp_metrics a; a.n = 64.0; a.comp_us_tok = 100.0; a.U = 4e6; a.up = 1e6; a.wait_us = 2e5;
    pp_metrics b = a; b.n = 4096.0; b.up = 3.9e6; b.wait_us = 4e5;
    if (!(estimate_router_time_per_token(a) < estimate_fullstream_time_per_token(a))) { printf("FAIL short pass: router must win\n"); g_fail++; }
    if (!(estimate_fullstream_time_per_token(b) < estimate_router_time_per_token(b))) { printf("FAIL long pass: stream must win\n"); g_fail++; }

    // ---- simulated machines: no model, no card. A DeepSeek-sized model (60 layers x 256 experts of 8.6 MB = 132 GB, 8 picked) on machines made of numbers ----
    pp_model ds; ds.layers = 60; ds.experts = 256; ds.used = 8; ds.expert_bytes = 8.6e6;
    pp_machine pc; pc.ssd_gbs = 3.0; pc.cpu_gbs = 40.0; pc.ram_gb = 101.0; pc.link_gbs = { 26.0, 6.4 }; pc.vram_gb = { 4.0, 4.0 }; pc.comp_us_tok = { 1500.0, 1500.0 }; pc.router_wait_us_layer = 3000.0;
    const double lim = 16384.0;
    auto check = [&](const char * what, bool ok) { if (!ok) { printf("FAIL sim: %s\n", what); g_fail++; } };
    printf("crossover router -> full stream, tokens per pass (0: stream always, %.0f: never)\n", lim + 1.0);
    printf("%-34s %8s\n", "machine", "tokens");
    auto row = [&](const char * name, const pp_machine & hw) { const double x = pp_crossover_tokens(hw, ds, lim); printf("%-34s %8.0f\n", name, x); return x; };
    const double x_pc = row("two cards, 101 GB RAM, SSD 3 GB/s", pc);
    // a slower SSD makes every streamed byte dearer: the router keeps winning up to a larger pass (never the other way)
    double prev = x_pc;
    for (const double ssd : { 2.0, 1.0, 0.5 }) {
        pp_machine hw = pc; hw.ssd_gbs = ssd;
        char nm[64]; snprintf(nm, sizeof nm, "  SSD %.1f GB/s", ssd);
        const double x = row(nm, hw);
        check("a slower SSD never moves the crossover down", x >= prev);
        prev = x;
    }
    // less RAM puts more of the model on the SSD: the same direction (the small-RAM machines users have)
    prev = x_pc;
    for (const double ram : { 64.0, 32.0, 16.0, 8.0 }) {
        pp_machine hw = pc; hw.ram_gb = ram;
        char nm[64]; snprintf(nm, sizeof nm, "  RAM %.0f GB", ram);
        const double x = row(nm, hw);
        check("less RAM never moves the crossover down", x >= prev);
        prev = x;
    }
    // a faster link makes the stream cheaper: the crossover never moves up (all in RAM, so the link is the limit)
    pp_machine ramful = pc; ramful.ram_gb = 200.0;
    prev = row("all experts in RAM, link 6.4 GB/s", [&] { pp_machine hw = ramful; hw.link_gbs = { 6.4 }; hw.vram_gb = { 4.0 }; return hw; }());
    for (const double l : { 13.0, 26.0, 52.0 }) {
        pp_machine hw = ramful; hw.link_gbs = { l }; hw.vram_gb = { 4.0 };
        char nm[64]; snprintf(nm, sizeof nm, "  link %.0f GB/s", l);
        const double x = row(nm, hw);
        check("a faster link never moves the crossover up", x <= prev);
        prev = x;
    }
    // a slower card hides more upload under its compute: the stream is never dearer for it
    {
        pp_machine slow = ramful; slow.comp_us_tok = { 6000.0 };
        check("a slower card never moves the crossover up", row("all in RAM, card 4x slower", slow) <= row("all in RAM", ramful));
    }
    // everything on the cards: nothing to upload, both paths are the compute, plus the router's wait
    {
        pp_machine big = pc; big.vram_gb = { 140.0 }; big.link_gbs = { 26.0 };   // one card: the stream computes on the fastest card, what a second card holds it still uploads
        const pp_metrics mb = pp_metrics_for(big, ds, 512.0);
        near("model on the cards: the stream is the compute", estimate_fullstream_time_per_token(mb), 1500.0);
        near("model on the cards: no misses", estimate_moe_miss_ongpu_time(mb), 0.0);
        check("model on the cards: the stream from the first token", pp_crossover_tokens(big, ds, lim) == 0.0);
    }
    // more card memory never makes any path slower
    for (const double n : { 64.0, 1024.0, 8192.0 }) {
        pp_machine a4 = pc, a16 = pc; a16.vram_gb = { 16.0, 16.0 };
        const pp_metrics m4 = pp_metrics_for(a4, ds, n), m16 = pp_metrics_for(a16, ds, n);
        check("more card memory: stream not slower", estimate_fullstream_time_per_token(m16) <= estimate_fullstream_time_per_token(m4));
        check("more card memory: router not slower", estimate_router_time_per_token(m16) <= estimate_router_time_per_token(m4));
        check("more card memory: decode chain not slower", estimate_decodechain_time_per_token(m16) <= estimate_decodechain_time_per_token(m4));
    }

    // layer parallel on two cards with equal links, everything in RAM: it uploads no more than the stream (each card's cached experts stay where they are) and never
    // beats the compute (the cards take turns)
    {
        pp_machine eq = pc; eq.link_gbs = { 26.0, 26.0 }; eq.ram_gb = 200.0;
        for (const double n : { 512.0, 4096.0, 16384.0 }) {
            const pp_metrics mm = pp_metrics_for(eq, ds, n);
            check("layer parallel uploads no more than the stream", estimate_layerparallel_upload_time(mm) <= estimate_fullstream_upload_time(mm) + 1e-6);
            check("layer parallel never under the compute", estimate_layerparallel_time(mm) >= mm.comp_us_tok * n - 1e-6);
            printf("  equal links, %5.0f tokens: stream %.3f, mix %.3f, layer parallel %.3f ms per token\n", n, estimate_fullstream_time_per_token(mm) / 1e3,
                   estimate_mix_time_per_token(mm) / 1e3, estimate_layerparallel_time_per_token(mm) / 1e3);
        }
    }

    // uneven picks: the cold experts are touched late, so the router keeps winning longer (the real models are far from uniform)
    prev = x_pc;
    for (const double skew : { 0.5, 1.0, 1.5 }) {
        pp_model hot = ds; hot.heat_skew = skew;
        const double x = pp_crossover_tokens(pc, hot, lim);
        printf("  heat skew %.1f                      %8.0f\n", skew, x);
        check("uneven picks never move the crossover down", x >= prev);
        prev = x;
    }

    // ---- estimate against measured (PC1, 2026-11-10 rows; printed, not asserted: where the estimators are off is the planner's weakness list) ----
    {
        pp_machine pc1 = pc; pc1.vram_gb = { 1.2, 1.2 };   // a first request: 6% of the cache blocks hold an expert
        pp_model dsk = ds; dsk.heat_skew = 1.0;
        struct row_t { const char * what; double n; int path; double measured_s; };   // path 0 stream, 1 router, 2 decode chain
        const row_t rows[] = {
            { "DeepSeek 117 tokens, stream",        117.0,   0, 20.0 },
            { "DeepSeek 117 tokens, decode chain",  117.0,   2,  2.2 },
            { "DeepSeek 62 tokens, decode chain",    62.0,   2,  1.6 },
            { "DeepSeek 13489 tokens, stream",    13489.0,   0, 27.5 },
        };
        printf("%-36s %10s %10s %7s\n", "call", "est (s)", "meas (s)", "ratio");
        for (const auto & r : rows) {
            const pp_metrics mm = pp_metrics_for(pc1, dsk, r.n);
            const double est = (r.path == 0 ? estimate_fullstream_time_per_token(mm) : r.path == 1 ? estimate_router_time_per_token(mm) : estimate_decodechain_time_per_token(mm)) * r.n / 1e6;
            printf("%-36s %10.1f %10.1f %7.2f%s\n", r.what, est, r.measured_s, est / r.measured_s, est / r.measured_s > 1.5 || est / r.measured_s < 0.67 ? "  OFF" : "");
        }
    }

    // ---- router or full stream (pp_switch_pick): the arm the planner strikes out. Qwen3.8 IQ3_S, 2442-token calls, 2 x 3090, build chain_grid_claims, 2026-11-14 ----
    {
        pp_metrics q; q.n = 2442.0; q.comp_us_tok = 78.0; q.up = 149e3; q.wait_us = 0.0; q.U = 1334e3;   // first request: compute calculated, no router wait known
        // the router 339 ms against the stream's 1334 struck the stream; the chain took 2449 ms, the release's stream and mix 1767
        check("cold: two calculated estimates strike nothing", pp_switch_pick(q, 0.0, 0.0, false) == PP_SWITCH_NONE);
        // second request: the chain's bound measured 2300 ms of compute; the stream's upload at the per-expert average (10.2 GB/s) is 2863 ms, at the probed maximum 1054
        q.comp_us_tok = 2300e3 / 2442.0; q.up = 436e3;
        q.U = 2863e3; check("per-expert link average: the router wins (the failure)", pp_switch_pick(q, 0.0, 0.0, true) == PP_SWITCH_STREAM_OUT);
        q.U = 1054e3; check("probed link maximum: the stream wins", pp_switch_pick(q, 0.0, 0.0, true) == PP_SWITCH_ROUTER_OUT);
        // a path measured here counts with its real time (Qwen3.8 IQ4_XS 512-token passes: router 2.9-3.0 ms a token, the stream measured 3.6-3.8)
        pp_metrics r; r.n = 512.0; r.comp_us_tok = 2000.0; r.up = 300e3; r.wait_us = 200e3; r.U = 1e6;
        check("measured stream slower than the router's estimate: the stream is struck", pp_switch_pick(r, 0.0, 3.7e3 * 512.0, true) == PP_SWITCH_STREAM_OUT);
        check("measured router slower than the stream's estimate: the router is struck", pp_switch_pick(r, 5e3 * 512.0, 0.0, true) == PP_SWITCH_ROUTER_OUT);
        check("a measured time decides without other measured inputs", pp_switch_pick(r, 5e3 * 512.0, 0.0, false) == PP_SWITCH_ROUTER_OUT);
        check("a tie goes to the stream", pp_switch_pick(r, 1e6, 1e6, true) == PP_SWITCH_ROUTER_OUT);
    }
    // a refused pass (Kimi K3.1 k3onepass, 2026-11-15): layer parallel's 4588-token pass at 4588 cells, its CUDA1 graph 5329.6 MiB beside the KV in a 6206.6 MiB pool
    {
        const uint32_t n = 4588; const uint64_t cells = 4588;
        const pp_refusal lp = pp_refusal_of(true, n, cells, 0.88, 2304);
        check("layer parallel refused: the ceiling is its own", lp.path_only);
        check("layer parallel refused: the call runs again at the same pass size (the one card held 4190-token passes, 57-63 t/s)", lp.retry_pass == n);
        check("layer parallel refused: its ceiling keeps it off a pass of that size", !pp_pass_under(lp.tc, n, (double) cells));
        check("layer parallel refused: a pass of half the tokens and cells stays its own", pp_pass_under(lp.tc, n / 2, cells / 2.0));
        check("no ceiling: every pass", pp_pass_under(0, 1e6, 1e6));
        const pp_refusal one = pp_refusal_of(false, n, cells, 0.88, 2304);
        check("the last resort refused: every path, smaller passes (as before)", !one.path_only && one.retry_pass == 2304);
        // CUDA1's record of graph sizes after that refusal (split_ab-k3-1pass.server.log): the start-up probes, estimates in the stream's layout (graph + CUDA scratch),
        // and the refused graph. A real size of 512 tokens or more takes the estimates out of the fit (need_real_only); the cap is the largest pass whose line leaves the
        // KV of 4608 cells (19968 B each) under the 8-chunk ring floor, 861 MiB short of the 6206.6 MiB pool
        const double MiB = 1048576.0;
        auto record = [&] { return std::map<int32_t, size_t>{ { 2720, (size_t) (2601.3 * MiB) }, { 5120, (size_t) (4891.2 * MiB) }, { 5152, (size_t) (4921.7 * MiB) },
                                                              { 5440, (size_t) (5196.5 * MiB) }, { 4588, (size_t) (5329.6 * MiB) } }; };
        const std::set<int32_t> est = { 2720, 5120, 5152, 5440 };
        auto cap = [&](const std::map<int32_t, size_t> & need) {
            bool real = false;
            for (const auto & kv : need) { real = real || (kv.first >= 512 && !est.count(kv.first)); }
            std::vector<std::pair<int32_t, size_t>> pts;
            for (const auto & kv : need) { if (!real || !est.count(kv.first)) { pts.push_back(kv); } }
            double a = 0.0, b = 0.0;
            return ppx_lend_line(pts, a, b) ? ((6206.6 - 861.0) * MiB - 4608.0 * 19968.0 - a) / b : 0.0;
        };
        std::map<int32_t, size_t> kept = record(), lp_rec = record(), one_rec = record();
        check("kept, the refused size alone is the fit: the stream's cap under the call (2304 + 2284 ran)", cap(kept) < n);
        pp_refusal_drop(lp, n, lp_rec);
        check("layer parallel refused: its graph's size leaves the record", lp_rec.count((int32_t) n) == 0 && lp_rec.size() == 4);
        check("layer parallel refused: the stream's cap holds the 4588-token call in one pass", cap(lp_rec) >= n);
        pp_refusal_drop(one, n, one_rec);
        check("the last resort refused: its size stays (its next pass is that layout, smaller)", one_rec.count((int32_t) n) == 1);
        printf("refused 4588-token pass: the stream's pool cap on CUDA1 %.0f tokens with the refused size kept, %.0f with it dropped\n", cap(kept), cap(lp_rec));
    }

    // the CPU/GPU split's A/B (ab_tune): token times from a made-up cost of the multiplier, the search must land on its optimum
    {
        auto drive = [](ab_tune & a, int n, auto && cost, bool differs = true) { for (int i = 0; i < n; ++i) { a.feed(cost(a.now(), i), differs); } };
        auto vee = [](double opt) { return [opt](double b, int) { return 400.0 + 20.0 * std::fabs(std::log2(b) - std::log2(opt)); }; };
        ab_tune up; drive(up, 60, vee(4.0));
        check("uploads cheaper than priced: the multiplier climbs to 4 and stops", up.base == 4.0 && up.st == 0);
        ab_tune dn; drive(dn, 60, vee(0.25));
        check("the CPU cheaper than priced: x2 loses, then halves to 0.25", dn.base == 0.25 && dn.st == 0);
        ab_tune same; drive(same, 60, vee(4.0), false);
        check("a candidate that splits no layer differently never wins", same.base == 1.0);
        ab_tune warm; drive(warm, 60, [](double, int i) { return 400.0 - 2.0 * i; });
        check("a warming cache alone (later tokens faster) moves nothing", warm.base == 1.0);
        ab_tune top; drive(top, 80, [](double b, int) { return 400.0 - 10.0 * std::log2(b); });
        check("cheaper the higher: stops at the bound 16", top.base == 16.0 && top.st == 0);
        // settled: the next search starts after `every` tokens (the user's 100)
        ab_tune rs; int i = 0, at = -1;
        for (; i < 1000; ++i) { const bool done = rs.feed(vee(4.0)(rs.now(), i), true); if (done && rs.st == 0 && at < 0) { at = i; } if (at >= 0 && rs.st == 1) { break; } }
        check("the search runs again 100 tokens after it settled", at >= 0 && i - at == 100);
        ab_tune ab; drive(ab, 1, vee(4.0)); ab.abort(); check("abort: the samples are dropped, the search restarts on the next token", ab.st == 0 && ab.since == ab.every);
        // noise: +-5% per token around a flat cost, 20 searches; the multiplier may move only when a pair series says so, never past the bounds
        ab_tune nz; std::mt19937 rng(7); std::uniform_real_distribution<double> u(0.95, 1.05);   // not an LCG: its consecutive outputs correlate with the b c alternation
        drive(nz, 2000, [&](double, int) { return 400.0 * u(rng); });
        check("noise: the multiplier stays within one step of 1", nz.base >= 0.5 && nz.base <= 2.0);
        printf("ab_tune noise: 2000 tokens flat cost +-5%%: multiplier %.4g\n", nz.base);
    }

    // the decode thread search (thread_search) on made-up answers of 100 tokens (user 2026-11-18: answers of 64-100 tokens, "we need quick in max 12t solution").
    // A token's time: a base, its upload MiB at 0.012 ms each, its CPU work (half fixed, half by its CPU MiB) at the shape's speed for that thread count, 4%
    // unexplained and a slow token in 30. CPU MiB scatter 35% around 180; uploads mostly about 40 MiB, one token in 7 about 400, and 6x more on an answer's first
    // tokens (the cache refills after the prompt): raw times scatter 10-15% as after the sane cells' long prompts, and the bytes explain most of it as they did in
    // those cells' TOK lines. The search starts at the computed count (GPUs: the true cores, 32 on PC1; CPU only: all 64 threads) and compares the count in use
    // with its half; an answer samples 2 warm-up tokens and at most 5 pairs, a clear win walks on at the next answer. 200 seeds per shape, 30 answers each: 95%
    // must land within the shape's tolerance of the best count's time, and no answer may spend more than 12 tokens searching.
    {
        struct shape { const char * name; double base; double (*cpu)(int); bool cache; double tol; int start; };
        double (*mem)(int) = [](int n) { return 112.0 / std::min(n, 32) + 0.1 * std::max(0, n - 32); };
        const shape shapes[] = {
            { "memory-bound, best at the 32 cores (DeepSeek, MiMo), GPUs", 28.9, mem, true, 1.05, 32 },
            { "memory-bound, CPU only (all 64 threads first)",             28.9, mem, true, 1.05, 64 },
            { "contention, best near 16 (Vulkan), GPUs",                   20.0, [](int n) { return 80.0 / n + 0.3 * n; }, true, 1.05, 32 },
            { "compute-bound, SMT helps, best at 64, CPU only",            20.0, [](int n) { return 400.0 / (std::min(n, 32) + 0.3 * std::max(0, n - 32)); }, true, 1.05, 64 },
            { "no expert cache (no byte counters), best at 32, GPUs",      28.9, mem, false, 1.05, 32 },
            // MiMo V2.6 12k/1k measured at fixed counts (2026-11-15, search off, tokens 200-999 of both runs): 16 threads 26.8 ms a token, 24 25.6, 31 25.8. 16 is 4.7%
            // slower, inside 5%, so this shape asks for 2%
            { "flat top at 24-31, 16 4.7% slower (MiMo measured), GPUs",   23.2, [](int n) { return 57.6 / n + 0.1 * std::max(0, n - 24); }, true, 1.02, 32 },
        };
        for (const shape & sh : shapes) {
            double best = 1e30; int nbest = 0;
            for (int n = 1; n <= 64; ++n) { if (sh.base + sh.cpu(n) < best) { best = sh.base + sh.cpu(n); nbest = n; } }
            int good = 0, low = 0, high = 0, over12 = 0; double searched = 0.0, cost = 0.0;
            for (int seed = 1; seed <= 200; ++seed) {
                std::mt19937 rng(seed); std::normal_distribution<double> g(0.0, 1.0); std::uniform_real_distribution<double> u01(0.0, 1.0);
                thread_search ts; ts.start(sh.start);
                for (int a = 0; a < 30; ++a) {
                    if (ts.settled() && ts.won) { ts.start(ts.b); } else { ts.resume(); }   // the engine at an answer's first token
                    int in_search = 0;
                    for (int i = 0; i < 100; ++i) {
                        in_search += ts.sampling() ? 1 : 0;
                        const int n = ts.now();
                        const double x = 180.0 * std::exp(0.35 * g(rng));
                        const double up = (u01(rng) < 1.0 / 7.0 ? 400.0 : 40.0) * std::exp(0.5 * g(rng)) * (1.0 + 5.0 * std::exp(-i / 30.0));
                        double t = (sh.base + 0.012 * up) * (1.0 + 0.05 * std::exp(-i / 40.0)) + sh.cpu(n) * (0.5 + 0.5 * x / 180.0);
                        t *= std::exp(0.04 * g(rng)) * (u01(rng) < 1.0 / 30.0 ? 1.25 : 1.0);
                        cost += (sh.base + sh.cpu(n)) / best;
                        if (sh.cache) { ts.feed(t, x, up); } else { ts.feed(t); }
                    }
                    searched += in_search; over12 += in_search > 12 ? 1 : 0;
                }
                const double r = (sh.base + sh.cpu(ts.b)) / best;
                if (r <= sh.tol) { ++good; } else if (ts.b < nbest) { ++low; } else { ++high; }
            }
            printf("thread search, %s: %d of 200 within %.0f%% of the best %d (off: %d too low, %d too high); %.1f tokens searched an answer, the counts used %.2f%% over the best\n",
                   sh.name, good, (sh.tol - 1.0) * 100.0, nbest, low, high, searched / 6000.0, (cost / 600000.0 - 1.0) * 100.0);
            check("thread search: within the shape's tolerance of the best count's time in 95% of the seeds", good >= 190);
            check("thread search: no answer spends more than 12 tokens searching", over12 == 0);
        }
    }

    // the drift watch beside the settled thread search (drift_watch). MiMo's decode after a 60k prompt runs in phases of about 320 tokens with and without expert uploads
    // (2026-11-15 TOK lines: 26-28 ms at about 520 MiB of CPU + upload work a token, 30-33 ms at about 850 MiB): the box did not change, so no search again. The old rule
    // (time alone, a 64-token window 10% over the best twice) restarted the search from 64 threads once a cell. A box 20% slower at the same work must still be seen, as must
    // one without byte counters (work 0).
    {
        auto run = [](double comparable, bool phases, bool load, int seed) {
            drift_watch dw; dw.comparable = comparable;
            std::mt19937 rng(seed); std::normal_distribution<double> g(0.0, 1.0);
            for (int i = 0; i < 2000; ++i) {
                const bool up = phases && (i / 320) % 2 == 1;
                double t = (up ? 31.0 : 27.0) * std::exp(0.04 * g(rng));
                const double w = phases ? (up ? 850.0 : 520.0) * std::exp(0.15 * g(rng)) : 0.0;
                if (load && i >= 1000) { t *= 1.2; }
                if (dw.feed(t, w)) { return i; }
            }
            return -1;
        };
        int false_old = 0, false_new = 0, seen = 0, seen_flat = 0, false_flat = 0;
        for (int seed = 1; seed <= 100; ++seed) {
            false_old  += run(1e9, true, false, seed) >= 0;
            false_new  += run(0.25, true, false, seed) >= 0;
            const int at = run(0.25, true, true, seed);
            seen       += at >= 1000 && at < 1600;   // the load starts in an upload phase: seen in the next phase without uploads (1280) plus two windows
            false_flat += run(0.25, false, false, seed) >= 0;
            const int af = run(0.25, false, true, seed);
            seen_flat  += af >= 1000 && af < 1000 + 4 * 64;
        }
        printf("drift watch: upload phases restart the search in %d of 100 seeds by time alone, %d at the same work; a 20%% slower box seen in %d (phases) and %d (no byte counters, %d false)\n",
                false_old, false_new, seen, seen_flat, false_flat);
        check("drift watch: upload phases are no drift", false_new == 0);
        check("drift watch: a slower box at the same work is seen", seen >= 98 && seen_flat >= 98 && false_flat == 0);
        drift_watch dw; for (int i = 0; i < 64; ++i) { dw.feed(27.0, 520.0); }
        dw.reset(); for (int i = 0; i < 64; ++i) { dw.feed(40.0, 520.0); }   // a new request: a 128k context runs slower; its own first window is its baseline
        bool again = false; for (int i = 0; i < 256; ++i) { again |= dw.feed(40.0, 520.0); }
        check("drift watch: a new request starts its own baseline", !again);
        // the refill a prompt leaves behind (the expert cache's background uploads) slows a request's first windows at comparable work: GLM two-card Vulkan
        // 2026-11-17, "64 tokens at 90.58 ms/token against 73.83 settled" restarted the search from 64 threads in every slow run 2 (11-14 t/s against 17-18)
        auto refill_run = [](bool gate) {
            drift_watch w; int fired = -1;
            for (int i = 0; i < 64 * 12; ++i) {
                const bool refill = i < 64 * 6;
                const double t = refill && i >= 64 ? 91.0 : 74.0;
                if (w.feed(t, 520.0, gate && refill) && fired < 0) { fired = i; }
            }
            return fired;
        };
        check("drift watch: the refill's slow windows restart the search when they count (the GLM failure)", refill_run(false) >= 0);
        check("drift watch: no restart while the expert cache refills after a prompt", refill_run(true) < 0);
        drift_watch w2; bool seen2 = false;
        for (int i = 0; i < 64 * 4; ++i) { w2.feed(91.0, 520.0, true); }    // the refill
        for (int i = 0; i < 64 * 2; ++i) { w2.feed(74.0, 520.0); }          // after it: the baseline
        for (int i = 0; i < 64 * 3; ++i) { seen2 |= w2.feed(90.0, 520.0); } // other load on the box
        check("drift watch: a slower box after the refill is still seen", seen2);
    }
    {   // the self-tuner's knob order (self_tune's list without the predictor): GLM two-card Vulkan 2026-11-18, SWAP_LEAD (320-token slices) first left
        // UPLOAD_WAIT untried over both 1k answers in 3 of 6 cells; gate 3 stayed on, 11-14 t/s against 18 with gate 0
        struct knob { const char * name; bool slow; };
        std::vector<knob> k = { { "SWAP_LEAD", true }, { "UPLOAD_WAIT", false }, { "UPLOAD_SHARE", true }, { "WAIT_SWAPS", false } };
        tune_order(k);
        int ahead = 0;   // decode tokens before UPLOAD_WAIT starts: each knob ahead of it at 3 slices of each of its 2 candidates
        size_t i = 0;
        for (; i < k.size() && strcmp(k[i].name, "UPLOAD_WAIT") != 0; ++i) { ahead += 2 * 3 * (k[i].slow ? 320 : 36); }
        check("self-tune: the upload gate is decided within the first 1k-token answer", i < k.size() && ahead + 2 * 3 * 36 <= 1024);
        check("self-tune: transient knobs first, each group in its listed order", strcmp(k[0].name, "UPLOAD_WAIT") == 0 && strcmp(k[1].name, "WAIT_SWAPS") == 0 &&
              strcmp(k[2].name, "SWAP_LEAD") == 0 && strcmp(k[3].name, "UPLOAD_SHARE") == 0);
    }
    {   // the self-tuner's verdict on one knob (tune_decide; slices in ms per token, incumbent 0, 3 to 8 slices each)
        std::vector<std::vector<double>> s = { { 25.0 }, { 28.0 } };
        tune_verdict v = tune_decide(s, 0, 3, 8);
        check("self-tune verdict: an alternative 12% slower after one slice each is dropped", v.done && v.pick == 0);
        s = { { 28.0, 28.1, 27.9 }, { 25.0, 25.1, 24.9 } };
        v = tune_decide(s, 0, 3, 8);
        check("self-tune verdict: a clear 10% win is taken after 3 slices each", v.done && v.sure && v.pick == 1);
        s = { { 28, 31, 26 }, { 27, 24, 29 } };
        v = tune_decide(s, 0, 3, 8);
        check("self-tune verdict: noisy slices are not final before the slice limit", !v.done);
        s = { std::vector<double>(8), std::vector<double>(8) };
        for (int j = 0; j < 8; ++j) { s[0][j] = j % 2 ? 31 : 26; s[1][j] = j % 2 ? 29 : 25; }
        v = tune_decide(s, 0, 3, 8);
        check("self-tune verdict: at the slice limit an unsure alternative leaves the value in use", v.done && !v.sure && v.pick == 0);
    }
    {   // the decode tuners take turns (tune_slice_counts, tune_drift_held). MiMo CUDA 12k/4k 2026-11-18: the self-tuner's slices ran inside the thread search,
        // then a drift restart ran inside SWAP_LEAD's slices; SWAP_LEAD -2 (a clear 10% win on b15807) stayed 0, decode 34.5/33.6 t/s against 37.9/38.5
        thread_search s;
        s.start(64);
        int counted = 0, searched = 0;
        for (int i = 0; i < 2000 && s.sampling(); ++i) {
            searched++;
            if (tune_slice_counts(true, false, !s.sampling())) { counted++; }
            const double t = 25000.0 * (1.0 + 0.15 * std::fabs(std::log2(s.now() / 32.0))) * (1.0 + 0.02 * (double) ((i * 7919) % 11 - 5) / 5.0);
            s.feed(t, 1.0 + 0.01 * (i % 7), 1.0 + 0.01 * (i % 5));
        }
        check("decode tuners take turns: no self-tune slice token while the thread search samples", searched > 0 && counted == 0);
        check("decode tuners take turns: a slice counts a decode token once the search is quiet",
              tune_slice_counts(true, false, true) && !tune_slice_counts(true, true, true) && !tune_slice_counts(false, false, true));
        drift_watch open, closed;
        int fired_open = 0, fired_closed = 0;
        for (int i = 0; i < 64 * 12; ++i) {
            const double t = i < 64 * 4 ? 25000.0 : 28000.0;   // an arm under test, 12% slower at the same work
            fired_open += open.feed(t, 2.0, tune_drift_held(false, true));
            fired_closed += closed.feed(t, 2.0, tune_drift_held(false, false));
        }
        check("decode tuners take turns: a self-tune arm 12% slower does not restart the thread search", fired_open == 0);
        check("decode tuners take turns: with no cycle open a box 12% slower still restarts it", fired_closed > 0);
    }

    {   // the refill after a prompt is a phase of its own (upload_gate_for, user 2026-11-18). The DDR gate (UPLOAD_WAIT 3, on by default) held every upload while
        // the CPU computed uncached experts, the refill's too: after 12k prompts the refill of the blocks the prompt borrowed (8-17 GB on two cards) ran at 15-30
        // GB/s of the links' 53 (MiMo and GLM CUDA logs 2026-11-17/18), and its decode tokens took 1.2-2.6x the steady ones
        check("refill phase: a refill upload waits for no gate", upload_gate_for(3, true) == 0 && upload_gate_for(2, true) == 0 && upload_gate_for(1, true) == 0);
        check("refill phase: a steady upload keeps the gate", upload_gate_for(3, false) == 3 && upload_gate_for(1, false) == 1 && upload_gate_for(0, false) == 0);
    }

    // the CPUs this process may use cut the topology (common_cpu_groups_masked_fork). PC1's 3975WX: 8 L3 groups of 4 cores, CPU c and c+32 are siblings. Under taskset to the
    // first chiplet (0-7,32-39, run.py simjoe: an 8-core desktop) 2 groups of 8 CPUs and 8 cores are left; until 2026-11-15 the probe and the default thread counts took all 64
    {
        std::vector<std::vector<int>> l3, cores;
        for (int g3 = 0; g3 < 8; ++g3) { std::vector<int> m; for (int c = 4 * g3; c < 4 * g3 + 4; ++c) { m.push_back(c); } for (int c = 4 * g3; c < 4 * g3 + 4; ++c) { m.push_back(c + 32); } l3.push_back(m); }
        for (int c = 0; c < 32; ++c) { cores.push_back({ c, c + 32 }); }
        std::vector<int> allowed; for (int c = 0; c < 8; ++c) { allowed.push_back(c); } for (int c = 32; c < 40; ++c) { allowed.push_back(c); }
        const auto d = common_cpu_groups_masked_fork(l3, allowed), k = common_cpu_groups_masked_fork(cores, allowed);
        check("cpu mask: 2 L3 groups left", d.size() == 2);
        check("cpu mask: 8 cores left", k.size() == 8);
        check("cpu mask: the first group is 0-3 and 32-35", !d.empty() && d[0] == std::vector<int>({ 0, 1, 2, 3, 32, 33, 34, 35 }));
        check("cpu mask: no mask keeps every group", common_cpu_groups_masked_fork(l3, {}).size() == 8);
    }

    {   // the decode thread search's start (common_thread_start_fork, user 2026-11-18): with GPUs the true cores (PC1 CUDA: 32 of 64 threads within 1% of the best,
        // 64 22-30% slower), CPU only every thread the process may use; under run.py simjoe (one chiplet: 8 cores, 16 CPUs) the same rule on the mask
        check("thread start: GPUs start on the true cores", common_thread_start_fork(true, 32, 64) == 32);
        check("thread start: CPU only starts on every thread", common_thread_start_fork(false, 32, 64) == 64);
        check("thread start: a masked process counts its own cores", common_thread_start_fork(true, 8, 16) == 8 && common_thread_start_fork(false, 8, 16) == 16);
        check("thread start: no core count found keeps every thread", common_thread_start_fork(true, 0, 64) == 64);
    }

    printf(g_fail ? "%d failed\n" : "estimators: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
