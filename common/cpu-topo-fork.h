#pragma once
// CPU topology (chiplets / L3 groups, SMT, NUMA), a start-up memory-bandwidth probe over threads spread across the chiplets, and the thread strategy that follows
// from it. Logged at info level (-lv 3). Nothing is applied to the thread pool unless asked (see common_cpu_topo_cpus_fork).
#include <string>
#include <vector>

struct common_cpu_topo_fork {
    std::string model;                         // CPU brand string
    int n_logical  = 0;
    int n_physical = 0;                        // physical cores (SMT siblings counted once)
    int n_numa     = 1;
    std::string source;                        // "sysfs", "windows", "cpuid" or "guess"
    std::vector<std::vector<int>> dies;        // logical CPUs per L3 group (one chiplet / core complex each)
    std::vector<int> order;                    // one logical CPU per physical core, dealt round-robin over the dies: the order threads should be placed in
    // bandwidth probe: GB/s read with k threads placed in `order`
    std::vector<std::pair<int, double>> bw;
    int    plateau_threads = 0;                // smallest thread count within 5% of the best measured bandwidth
    double plateau_gbs     = 0.0;
    bool   bw_cached       = false;
};

const common_cpu_topo_fork & common_cpu_topo_get_fork();          // detects once (the bandwidth probe runs on first use and is cached on disk per CPU)
void common_cpu_topo_print_fork();                                // the info-level report
std::vector<int> common_cpu_topo_cpus_fork(int n_threads);        // the first n_threads of `order` (pin them one per core, spread over the dies)

// The logical CPUs this process may run on: its affinity mask at start-up (taskset, a container's cpuset, run.py simjoe), read before any thread pins itself; every CPU where
// the platform has no mask. The topology, the probe and the default thread counts use these only (until 2026-11-15 they took all 64 CPUs of PC1 under a 16-CPU mask).
const std::vector<int> & common_cpu_allowed_fork();
int common_cpu_allowed_count_fork();
int common_cpu_physical_count_fork();                             // the physical cores among those CPUs (no bandwidth probe)

// The decode thread search's start (user 2026-11-18): with GPUs the true cores (decode reads expert weights and SMT siblings add no bandwidth: on PC1's 3975WX 32 of
// 64 threads ran within 1% of the best on CUDA, 64 22-30% slower), CPU only every thread the process may use; no core count found: every thread
inline int common_thread_start_fork(bool gpus, int n_physical, int n_allowed) { return gpus && n_physical > 0 && n_physical < n_allowed ? n_physical : n_allowed; }

// pure: CPU groups (L3 groups, SMT sibling sets) cut to the allowed CPUs, empty groups dropped, order kept; an empty `allowed` keeps everything
inline std::vector<std::vector<int>> common_cpu_groups_masked_fork(const std::vector<std::vector<int>> & groups, const std::vector<int> & allowed) {
    if (allowed.empty()) { return groups; }
    std::vector<std::vector<int>> out;
    for (const auto & g : groups) {
        std::vector<int> m;
        for (int c : g) { for (int a : allowed) { if (a == c) { m.push_back(c); break; } } }
        if (!m.empty()) { out.push_back(m); }
    }
    return out;
}
