// placement: how thread placement changes host memory read bandwidth. L3 groups (a core complex: 4 cores on Zen2) come from sysfs; one hardware thread per core.
//   os     threads left to the scheduler
//   spread thread i goes to L3 group i % groups (one per group first, then a second core in each, ...)
//   pack   fill one L3 group, then the next
// Build: g++ -O3 -mavx2 -pthread placement.cpp -o placement ; run: ./placement [GiB = 8]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sched.h>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<int> parse_list(const std::string & s) {   // "0-3,8-11"
    std::vector<int> r; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) { size_t d = tok.find('-'); int a = atoi(tok.c_str()), b = d == std::string::npos ? a : atoi(tok.c_str() + d + 1); for (int i = a; i <= b; ++i) r.push_back(i); }
    return r;
}
static std::string slurp(const std::string & p) { std::ifstream f(p); std::string s; std::getline(f, s); return s; }
static uint64_t scan(const uint64_t * p, size_t n) { uint64_t a = 0, b = 0, c = 0, d = 0; for (size_t i = 0; i + 4 <= n; i += 4) { a += p[i]; b += p[i + 1]; c += p[i + 2]; d += p[i + 3]; } return a + b + c + d; }

int main(int argc, char ** argv) {
    const size_t n = ((size_t) (argc > 1 ? atoi(argv[1]) : 8) << 30) / 8; const int hw = (int) std::thread::hardware_concurrency();
    // L3 groups, first hardware thread of each core
    std::vector<std::vector<int>> groups; std::set<std::string> seen;
    for (int c = 0; c < hw; ++c) {
        std::string l3 = slurp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cache/index3/shared_cpu_list");
        if (l3.empty() || seen.count(l3)) continue;
        seen.insert(l3); std::vector<int> g;
        for (int x : parse_list(l3)) { std::vector<int> sib = parse_list(slurp("/sys/devices/system/cpu/cpu" + std::to_string(x) + "/topology/thread_siblings_list")); if (!sib.empty() && sib[0] == x) g.push_back(x); }
        if (!g.empty()) groups.push_back(g);
    }
    printf("L3 groups: %zu, cores per group: %zu (hardware threads %d)\n", groups.size(), groups.empty() ? 0 : groups[0].size(), hw);
    uint64_t * buf = (uint64_t *) aligned_alloc(4096, n * 8);
    { std::vector<std::thread> th; for (int t = 0; t < hw; ++t) th.emplace_back([&, t] { for (size_t i = n / hw * t; i < n / hw * (t + 1); ++i) buf[i] = i * 2654435761u; }); for (auto & x : th) x.join(); }
    auto cpu_for = [&](const char * mode, int i) -> int {
        if (groups.empty()) return -1;
        if (mode[0] == 's') { const auto & g = groups[i % groups.size()]; return g[(i / groups.size()) % g.size()]; }
        if (mode[0] == 'p') { size_t k = (size_t) i; for (const auto & g : groups) { if (k < g.size()) return g[k]; k -= g.size(); } return -1; }
        return -1;
    };
    auto run = [&](const char * mode, int threads) {
        std::atomic<bool> stop{false}; std::vector<double> bytes(threads, 0.0); std::vector<std::thread> th; std::atomic<uint64_t> sink{0}; std::atomic<int> ready{0};
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
            int c = mode[0] == 'o' ? -1 : cpu_for(mode, t);
            if (c >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
            ready++; while (ready < threads) {}
            const size_t per = n / threads; const uint64_t * p = buf + per * t; uint64_t s = 0;
            while (!stop) { s += scan(p, per); bytes[t] += (double) per * 8; }
            sink += s;
        });
        const double t0 = now(); std::this_thread::sleep_for(std::chrono::milliseconds(700)); stop = true; for (auto & x : th) x.join();
        double tot = 0; for (double b : bytes) tot += b; return tot / (now() - t0) / 1e9;
    };
    auto run_cpus = [&](const std::vector<int> & cpus) {   // one pinned thread per listed cpu
        const int threads = (int) cpus.size();
        std::atomic<bool> stop{false}; std::vector<double> bytes(threads, 0.0); std::vector<std::thread> th; std::atomic<uint64_t> sink{0}; std::atomic<int> ready{0};
        for (int t = 0; t < threads; ++t) th.emplace_back([&, t] {
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpus[t], &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
            ready++; while (ready < threads) {}
            const size_t per = n / threads; const uint64_t * p = buf + per * t; uint64_t s2 = 0;
            while (!stop) { s2 += scan(p, per); bytes[t] += (double) per * 8; }
            sink += s2;
        });
        const double t0 = now(); std::this_thread::sleep_for(std::chrono::milliseconds(700)); stop = true; for (auto & x : th) x.join();
        double tot = 0; for (double b : bytes) tot += b; return tot / (now() - t0) / 1e9;
    };
    // each L3 group alone (all its cores), then the best and the worst groups against spreading the same number of threads over all groups
    std::vector<std::pair<double, size_t>> rank;
    printf("\neach L3 group alone (all %zu cores pinned in it):\n", groups.empty() ? 0 : groups[0].size());
    for (size_t g = 0; g < groups.size(); ++g) { double b = std::max(run_cpus(groups[g]), run_cpus(groups[g])); rank.push_back({b, g}); printf("  group %2zu (cpu %3d..): %5.1f GB/s\n", g, groups[g][0], b); }
    std::sort(rank.begin(), rank.end(), [](auto & x, auto & y) { return x.first > y.first; });
    printf("\ncores   best groups   worst groups   spread over all groups   (GB/s)\n");
    for (int k = 1; k <= (int) groups.size() && k <= 4; ++k) {
        std::vector<int> best, worst, spread;
        for (int i = 0; i < k; ++i) { for (int c : groups[rank[i].second]) best.push_back(c); for (int c : groups[rank[rank.size() - 1 - i].second]) worst.push_back(c); }
        for (int i = 0; i < (int) best.size(); ++i) spread.push_back(cpu_for("spread", i));
        printf("%5zu   %11.1f   %12.1f   %22.1f\n", best.size(), std::max(run_cpus(best), run_cpus(best)), std::max(run_cpus(worst), run_cpus(worst)), std::max(run_cpus(spread), run_cpus(spread)));
    }
    printf("\nthreads      os   spread     pack   (GB/s read, best of 2)\n");
    std::vector<int> ts; std::vector<double> sp_bw, os_bw; double peak = 0;
    for (int t : {1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32}) {
        if (t > hw) break;
        double r[3];
        const char * m[3] = {"os", "spread", "pack"};
        for (int k = 0; k < 3; ++k) { r[k] = std::max(run(m[k], t), run(m[k], t)); }
        printf("%7d %7.1f %8.1f %8.1f\n", t, r[0], r[1], r[2]);
        ts.push_back(t); os_bw.push_back(r[0]); sp_bw.push_back(r[1]); peak = std::max({peak, r[0], r[1], r[2]});
    }
    // the answer: the fewest threads within 5% of the best bandwidth seen, and the cpu list that spreads them over the L3 groups
    for (size_t i = 0; i < ts.size(); ++i) {
        if (std::max(os_bw[i], sp_bw[i]) >= 0.95 * peak) {
            printf("\npeak %.1f GB/s; %d threads reach %.1f GB/s (%.0f%%): use -t %d; unpinned %.1f, spread over L3 groups %.1f\n", peak, ts[i], std::max(os_bw[i], sp_bw[i]), 100 * std::max(os_bw[i], sp_bw[i]) / peak, ts[i], os_bw[i], sp_bw[i]);
            printf("spread cpu list for taskset -c: ");
            for (int k = 0; k < ts[i]; ++k) printf("%s%d", k ? "," : "", cpu_for("spread", k));
            printf("\n");
            break;
        }
    }
    return 0;
}
