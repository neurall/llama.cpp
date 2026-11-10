// Unit tests of the prompt planner's strategy estimators (src/llama-estimate-fork.h): each strategy's time from hand-made metrics. CPU only, no model.
#include "../src/llama-estimate-fork.h"

#include <cmath>
#include <cstdio>

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

    printf(g_fail ? "%d failed\n" : "estimators: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
