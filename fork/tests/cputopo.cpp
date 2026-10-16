// Which logical CPUs share an L3 (one chiplet / core complex)? Prints the groups the way the thread placement will use them.
// Order of sources: 1) Linux sysfs cache/index3/shared_cpu_list, 2) cpuid (AMD 0x8000001D / Intel 4: logical CPUs sharing the L3, domain = APIC id / count),
// 3) equal groups by a guess. Build: c++ -O2 cputopo.cpp -o cputopo
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <thread>
#include <fstream>
#include <sstream>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

static std::vector<int> parse_list(const std::string & s) {   // "0-3,8-11"
    std::vector<int> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t d = tok.find('-');
        if (d == std::string::npos) { v.push_back(atoi(tok.c_str())); }
        else { for (int i = atoi(tok.substr(0, d).c_str()); i <= atoi(tok.substr(d + 1).c_str()); ++i) v.push_back(i); }
    }
    return v;
}

int main() {
    const int n = (int) std::thread::hardware_concurrency();
    std::map<int, std::vector<int>> dom;   // first CPU of the group -> members
    const char * src = "sysfs";
    for (int c = 0; c < n; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cache/index3/shared_cpu_list");
        std::string l; if (!std::getline(f, l)) { dom.clear(); break; }
        std::vector<int> m = parse_list(l); dom[m[0]] = m;
    }
#if defined(__x86_64__)
    if (dom.empty()) {   // cpuid: logical CPUs sharing the L3, then consecutive ids in groups of that size (APIC ids are consecutive per package on the machines checked)
        src = "cpuid";
        unsigned a, b, c, d; int share = 0;
        for (unsigned sub = 0; sub < 8; ++sub) {
            if (!__get_cpuid_count(0x8000001D, sub, &a, &b, &c, &d) && !__get_cpuid_count(4, sub, &a, &b, &c, &d)) break;
            if (((a >> 5) & 7) == 3) { share = ((a >> 14) & 0xfff) + 1; break; }
        }
        if (share > 0) { for (int c0 = 0; c0 < n; c0 += share) { std::vector<int> m; for (int i = c0; i < c0 + share && i < n; ++i) m.push_back(i); dom[c0] = m; } }
    }
#endif
    if (dom.empty()) { src = "guess: 1 group"; std::vector<int> m; for (int i = 0; i < n; ++i) m.push_back(i); dom[0] = m; }
    printf("source %s, %d logical CPUs, %zu L3 groups\n", src, n, dom.size());
    for (auto & kv : dom) { printf("  group:"); for (int c : kv.second) printf(" %d", c); printf("\n"); }
    return 0;
}
