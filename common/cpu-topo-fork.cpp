#include "cpu-topo-fork.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#  include <sched.h>
#endif
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#  include <cpuid.h>
#endif

static std::vector<int> parse_cpu_list(const std::string & s) {   // "0-3,8-11"
    std::vector<int> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) { continue; }
        const size_t d = tok.find('-');
        if (d == std::string::npos) { v.push_back(atoi(tok.c_str())); }
        else { for (int i = atoi(tok.substr(0, d).c_str()); i <= atoi(tok.substr(d + 1).c_str()); ++i) { v.push_back(i); } }
    }
    return v;
}

static std::string cpu_brand() {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    unsigned r[12];
    if (__get_cpuid(0x80000000u, &r[0], &r[1], &r[2], &r[3]) && r[0] >= 0x80000004u) {
        __get_cpuid(0x80000002u, &r[0], &r[1], &r[2], &r[3]); __get_cpuid(0x80000003u, &r[4], &r[5], &r[6], &r[7]); __get_cpuid(0x80000004u, &r[8], &r[9], &r[10], &r[11]);
        char b[49]; memcpy(b, r, 48); b[48] = 0;
        std::string s(b); size_t a = s.find_first_not_of(' '); return a == std::string::npos ? s : s.substr(a);
    }
#endif
    return "unknown CPU";
}

static void detect(common_cpu_topo_fork & t) {
    t.model = cpu_brand();
    t.n_logical = (int) std::thread::hardware_concurrency();
    std::map<int, std::vector<int>> dies;      // first CPU of the group -> members
    std::set<std::vector<int>> cores;          // sets of SMT siblings
    std::set<int> nodes;
#if defined(__linux__)
    t.source = "sysfs";
    for (int c = 0; c < t.n_logical; ++c) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        std::string l;
        { std::ifstream f(base + "/cache/index3/shared_cpu_list"); if (std::getline(f, l)) { auto m = parse_cpu_list(l); if (!m.empty()) { dies[m[0]] = m; } } }
        { std::ifstream f(base + "/topology/thread_siblings_list"); if (std::getline(f, l)) { auto m = parse_cpu_list(l); if (!m.empty()) { cores.insert(m); } } }
    }
    for (int n = 0; n < 64; ++n) { std::ifstream f("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist"); if (!f) { break; } nodes.insert(n); }
#elif defined(_WIN32)
    t.source = "windows";
    DWORD len = 0; GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (len && GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (DWORD off = 0; off < len;) {
            auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
            auto mask_to_list = [](const GROUP_AFFINITY & g) { std::vector<int> v; for (int b = 0; b < 64; ++b) { if ((g.Mask >> b) & 1) { v.push_back((int) g.Group * 64 + b); } } return v; };
            if (p->Relationship == RelationCache && p->Cache.Level == 3) { auto m = mask_to_list(p->Cache.GroupMask); if (!m.empty()) { dies[m[0]] = m; } }
            if (p->Relationship == RelationProcessorCore) { auto m = mask_to_list(p->Processor.GroupMask[0]); if (!m.empty()) { cores.insert(m); } }
            if (p->Relationship == RelationNumaNode) { nodes.insert((int) p->NumaNode.NodeNumber); }
            off += p->Size;
        }
    }
#endif
    if (dies.empty()) {   // cpuid: logical CPUs sharing the L3 (AMD 0x8000001D / Intel 4), consecutive ids in groups of that size
        t.source = "cpuid";
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        unsigned a, b, c, d; int share = 0;
        for (unsigned sub = 0; sub < 8; ++sub) {
            if (!__get_cpuid_count(0x8000001D, sub, &a, &b, &c, &d) && !__get_cpuid_count(4, sub, &a, &b, &c, &d)) { break; }
            if (((a >> 5) & 7) == 3) { share = (int) (((a >> 14) & 0xfff) + 1); break; }
        }
        if (share > 0) { for (int c0 = 0; c0 < t.n_logical; c0 += share) { std::vector<int> m; for (int i = c0; i < c0 + share && i < t.n_logical; ++i) { m.push_back(i); } dies[c0] = m; } }
#endif
    }
    if (dies.empty()) { t.source = "guess"; std::vector<int> m; for (int i = 0; i < t.n_logical; ++i) { m.push_back(i); } dies[0] = m; }
    for (auto & kv : dies) { t.dies.push_back(kv.second); }
    t.n_numa = std::max<int>(1, (int) nodes.size());
    // one CPU per physical core: the lowest id of every sibling set (without sibling info every logical CPU is a core)
    std::set<int> first;
    if (!cores.empty()) { for (auto & m : cores) { first.insert(*std::min_element(m.begin(), m.end())); } }
    else { for (int i = 0; i < t.n_logical; ++i) { first.insert(i); } }
    t.n_physical = (int) first.size();
    // deal the cores round-robin over the dies
    std::vector<std::vector<int>> per_die;
    for (auto & d : t.dies) { std::vector<int> v; for (int c : d) { if (first.count(c)) { v.push_back(c); } } per_die.push_back(v); }
    for (size_t k = 0;; ++k) {
        bool any = false;
        for (auto & v : per_die) { if (k < v.size()) { t.order.push_back(v[k]); any = true; } }
        if (!any) { break; }
    }
}

static void pin_this_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#elif defined(_WIN32)
    GROUP_AFFINITY g = {}; g.Group = (WORD) (cpu / 64); g.Mask = (KAFFINITY) 1 << (cpu % 64); SetThreadGroupAffinity(GetCurrentThread(), &g, nullptr);
#else
    (void) cpu;
#endif
}

// aggregate read bandwidth with threads pinned to `cpus`, each streaming its own 64 MiB slice (first touched by the thread itself, so it sits on its own NUMA node)
static double probe(const std::vector<int> & cpus) {
    const size_t bytes = 64u << 20, n = bytes / 8;
    std::atomic<int> ready { 0 }; std::atomic<bool> go { false };
    std::vector<std::thread> th; std::vector<double> secs(cpus.size(), 0.0); std::atomic<uint64_t> sink { 0 };
    for (size_t i = 0; i < cpus.size(); ++i) {
        th.emplace_back([&, i] {
            pin_this_thread(cpus[i]);
            uint64_t * buf = (uint64_t *) malloc(bytes); for (size_t j = 0; j < n; ++j) { buf[j] = j; }
            ready++; while (!go.load()) { }
            uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0; double best = 1e30;
            for (int rep = 0; rep < 4; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (size_t j = 0; j + 4 <= n; j += 4) { a0 += buf[j]; a1 += buf[j + 1]; a2 += buf[j + 2]; a3 += buf[j + 3]; }
                best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            secs[i] = best; sink += a0 + a1 + a2 + a3; free(buf);
        });
    }
    while (ready.load() < (int) cpus.size()) { }
    go = true;
    for (auto & t : th) { t.join(); }
    double worst = 0; for (double s : secs) { worst = std::max(worst, s); }
    return worst > 0 ? (double) bytes * cpus.size() / worst / 1e9 : 0.0;
}

static std::string cache_path(const common_cpu_topo_fork & t) {
    const char * home = getenv("HOME"); if (!home) { home = getenv("USERPROFILE"); } if (!home) { return ""; }
    std::string key = t.model + "|" + std::to_string(t.n_logical) + "|" + std::to_string(t.dies.size());
    uint64_t h = 1469598103934665603ull; for (char c : key) { h = (h ^ (uint8_t) c) * 1099511628211ull; }
    char b[32]; snprintf(b, sizeof b, "%016llx", (unsigned long long) h);
    return std::string(home) + "/.cache/llama-cpu-bw-" + b + ".txt";
}

static void run_probe(common_cpu_topo_fork & t) {
    const std::string path = cache_path(t);
    if (!path.empty()) {
        std::ifstream f(path); int k; double g;
        while (f >> k >> g) { t.bw.push_back({ k, g }); }
        if (!t.bw.empty()) { t.bw_cached = true; }
    }
    if (t.bw.empty()) {
        std::vector<int> ks; for (int k : { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 }) { if (k <= t.n_physical) { ks.push_back(k); } }
        if (ks.empty() || ks.back() != t.n_physical) { ks.push_back(t.n_physical); }
        for (int k : ks) { std::vector<int> cpus(t.order.begin(), t.order.begin() + std::min<size_t>(k, t.order.size())); t.bw.push_back({ k, probe(cpus) }); }
        if (!path.empty()) {
            const std::string dir = path.substr(0, path.rfind('/'));
#if !defined(_WIN32)
            std::string cmd = "mkdir -p '" + dir + "' 2>/dev/null"; (void) !system(cmd.c_str());
#endif
            std::ofstream f(path); for (auto & p : t.bw) { f << p.first << " " << p.second << "\n"; }
        }
    }
    double best = 0; for (auto & p : t.bw) { best = std::max(best, p.second); }
    t.plateau_gbs = best;
    for (auto & p : t.bw) { if (p.second >= 0.95 * best) { t.plateau_threads = p.first; break; } }
}

const common_cpu_topo_fork & common_cpu_topo_get_fork() {
    static common_cpu_topo_fork t = [] { common_cpu_topo_fork r; detect(r); run_probe(r); return r; }();
    return t;
}

std::vector<int> common_cpu_topo_cpus_fork(int n_threads) {
    const auto & t = common_cpu_topo_get_fork();
    return std::vector<int>(t.order.begin(), t.order.begin() + std::min<size_t>(std::max(n_threads, 0), t.order.size()));
}

void common_cpu_topo_print_fork() {
    const auto & t = common_cpu_topo_get_fork();
    std::string dies;
    for (auto & d : t.dies) { std::set<int> phys; for (int c : d) { if (std::find(t.order.begin(), t.order.end(), c) != t.order.end()) { phys.insert(c); } } dies += (dies.empty() ? "" : "+") + std::to_string(phys.size()); }
    LOG_INF("cpu topology: %s, %d cores / %d threads, %zu chiplets (L3 groups) of %s cores, %d NUMA node(s), detected from %s\n",
            t.model.c_str(), t.n_physical, t.n_logical, t.dies.size(), dies.c_str(), t.n_numa, t.source.c_str());
    std::string bw;
    for (auto & p : t.bw) { char b[48]; snprintf(b, sizeof b, " %d:%.1f", p.first, p.second); bw += b; }
    LOG_INF("cpu memory read probe (threads:GB/s, one thread per core dealt over the chiplets)%s:%s\n", t.bw_cached ? " [cached]" : "", bw.c_str());
    std::string cpus; for (int c : common_cpu_topo_cpus_fork(t.plateau_threads)) { cpus += (cpus.empty() ? "" : ",") + std::to_string(c); }
    LOG_INF("cpu thread strategy: memory is saturated at %.1f GB/s with %d threads (one per core, spread over %zu chiplets, cpus %s); %d cores are compute headroom; the pool is not pinned unless asked\n",
            t.plateau_gbs, t.plateau_threads, t.dies.size(), cpus.c_str(), t.n_physical - t.plateau_threads);
}
