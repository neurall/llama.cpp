// Stress test of the pool range-use counts (copy of pool_enter / pool_leave / pool_move_begin / pool_move_end in src/llama-moecache-fork.cpp; keep in step).
// W writer threads enter random block ranges (a "DMA" of ~50-300 us), one mover repeatedly announces a zone range, waits for its users, moves, ends.
// Checks: no writer is inside a range the mover owns, the mover never starts under a writer; no deadlock (watchdog 20 s); prints the stall totals.
// g++ -O2 -std=c++17 -pthread range_use_stress.cpp -o stress && ./stress [writers] [seconds]
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
using clk = std::chrono::steady_clock;
static std::mutex g_map_mu; static std::condition_variable g_map_cv;
struct use_t { const void * ap; size_t lo, hi; bool w; };
static std::vector<use_t> g_use, g_moving;
static bool ovl(const use_t & u, const void * ap, size_t lo, size_t hi) { return u.ap == ap && lo < u.hi && hi > u.lo; }
static std::atomic<long> st_n{0}, st_us{0}, st_calls{0};
static void enter(const void * ap, size_t lo, size_t hi, bool w) {
    std::unique_lock<std::mutex> lk(g_map_mu);
    auto ok = [&]() { for (auto & m : g_moving) { if (ovl(m, ap, lo, hi)) { return false; } } return true; };
    st_calls++;
    if (!ok()) { auto t0 = clk::now(); g_map_cv.wait(lk, ok); st_n++; st_us += std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count(); }
    g_use.push_back({ ap, lo, hi, w });
}
static void leave(const void * ap, size_t lo, size_t hi, bool w) {
    { std::lock_guard<std::mutex> lk(g_map_mu); for (size_t i = 0; i < g_use.size(); ++i) { if (g_use[i].ap == ap && g_use[i].lo == lo && g_use[i].hi == hi && g_use[i].w == w) { g_use.erase(g_use.begin() + i); break; } } }
    g_map_cv.notify_all();
}
static void move_begin(const void * ap, size_t lo, size_t hi) {
    std::unique_lock<std::mutex> lk(g_map_mu);
    g_moving.push_back({ ap, lo, hi, true });
    g_map_cv.wait(lk, [&]() { for (auto & u : g_use) { if (ovl(u, ap, lo, hi)) { return false; } } return true; });
}
static void move_end(const void * ap) {
    { std::lock_guard<std::mutex> lk(g_map_mu); for (size_t i = 0; i < g_moving.size(); ++i) { if (g_moving[i].ap == ap) { g_moving.erase(g_moving.begin() + i); break; } } }
    g_map_cv.notify_all();
}
static const int NB = 256; static const size_t B = 1 << 20;
static std::atomic<int> in_dma[NB];   // writers inside block b
static std::atomic<int> moving_lo{-1}, moving_hi{-1};
static std::atomic<long> bad{0}, dmas{0}, moves{0};
int main(int argc, char ** argv) {
    const int W = argc > 1 ? atoi(argv[1]) : 8; const int secs = argc > 2 ? atoi(argv[2]) : 5;
    int ap_tag = 0; const void * ap = &ap_tag;
    std::atomic<bool> stop{false}; std::atomic<long> progress{0};
    std::vector<std::thread> th;
    for (int w = 0; w < W; ++w) th.emplace_back([&, w]() {
        std::mt19937 rng(w + 1);
        while (!stop) {
            const int b = rng() % NB; enter(ap, b * B, (b + 1) * B, true);
            if (b >= moving_lo.load() && b < moving_hi.load()) { bad++; }   // inside the mover's range while it moves
            in_dma[b]++; std::this_thread::sleep_for(std::chrono::microseconds(50 + rng() % 250)); in_dma[b]--;
            leave(ap, b * B, (b + 1) * B, true); dmas++; progress++;
        }
    });
    std::thread mover([&]() {
        std::mt19937 rng(99);
        while (!stop) {
            const int lo = rng() % (NB - 40), hi = lo + 8 + rng() % 32;
            move_begin(ap, lo * B, hi * B);
            for (int b = lo; b < hi; ++b) { if (in_dma[b].load() != 0) { bad++; } }   // started under a running DMA
            moving_lo = lo; moving_hi = hi; std::this_thread::sleep_for(std::chrono::microseconds(300)); moving_lo = moving_hi = -1;
            move_end(ap); moves++; progress++;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });
    long last = -1;
    for (int s = 0; s < secs * 4 && !stop; ++s) { std::this_thread::sleep_for(std::chrono::milliseconds(250)); }
    stop = true;
    auto t0 = clk::now();
    std::thread wd([&]() { std::this_thread::sleep_for(std::chrono::seconds(20)); fprintf(stderr, "DEADLOCK\n"); _Exit(2); });
    wd.detach();
    for (auto & t : th) t.join(); mover.join(); (void) last; (void) t0;
    printf("writers %d, %lds: %ld dmas, %ld moves, violations %ld; %ld of %ld entries waited for a move, %.1f ms total (%.0f us each)\n", W, (long) secs, dmas.load(), moves.load(), bad.load(),
           st_n.load(), st_calls.load(), st_us.load() / 1e3, st_n ? (double) st_us / st_n : 0.0);
    return bad ? 1 : 0;
}
