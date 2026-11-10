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

    printf(g_fail ? "%d failed\n" : "estimators: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
