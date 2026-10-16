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
