#include "../src/llama-mapsync-fork.h"
#include <cstdio>
#include <random>
static int g_bad = 0;
static void check(const char * n, bool ok) { if (!ok) { printf("FAIL: %s\n", n); g_bad++; } }
int main() {
    const size_t T = 5, E = 64, CAP = 32;
    std::vector<mapsync_entry> ring(CAP); uint64_t head = 0, tail = 0;
    mapsync_ring r{ ring.data(), &head, &tail, CAP };
    std::vector<std::vector<int32_t>> host(T, std::vector<int32_t>(E, -1)), sent = host, dev = host;
    std::vector<int32_t *> dp(T); for (size_t t = 0; t < T; ++t) { dp[t] = dev[t].data(); }
    std::mt19937 rng(7);
    int fallbacks = 0, pushes = 0;
    for (int step = 0; step < 5000; ++step) {
        const int k = (int) (rng() % (step % 97 == 0 ? 80 : 6));   // now and then a burst bigger than the ring
        for (int i = 0; i < k; ++i) { host[rng() % T][rng() % E] = (int32_t) (rng() % 1000) - 1; }
        std::vector<mapsync_entry> es;
        for (size_t t = 0; t < T; ++t) { mapsync_diff((uint32_t) t, host[t].data(), sent[t].data(), E, es); }
        if (!mapsync_push(r, es)) {   // fallback: whole tables, ring restarts empty
            mapsync_apply_ref(r, dp.data());
            for (size_t t = 0; t < T; ++t) { dev[t] = host[t]; }
            *r.tail = *r.head;
            fallbacks++;
        } else { pushes++; }
        if (rng() % 3 == 0) { mapsync_apply_ref(r, dp.data()); }   // the kernel runs when the graph does, not after every change
        if (rng() % 3 == 0) {
            mapsync_apply_ref(r, dp.data());
            bool same = true; for (size_t t = 0; t < T; ++t) { same &= dev[t] == host[t]; }
            check("mapsync: device tables equal host after apply", same);
            if (!same) { break; }
        }
        check("mapsync: ring never overfull", head - tail <= CAP);
    }
    // a diff with no change pushes nothing
    std::vector<mapsync_entry> none; mapsync_diff(0, host[0].data(), sent[0].data(), E, none);
    check("mapsync: no change, no entries", none.empty());
    check("mapsync: both paths seen", fallbacks > 10 && pushes > 1000);
    printf(g_bad ? "FAIL: %d failure(s)\n" : "OK: 0 failure(s)\n", g_bad);
    return g_bad != 0;
}
