#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"
#include "cpu-topo-fork.h"

#include "../src/llama-ext.h"

#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#ifdef __linux__
#include <sys/mman.h>
#endif
#include <cinttypes>
#include <climits>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__


// CPUs the container may use: its cgroup CPU quota (v2 cpu.max "QUOTA PERIOD", v1 cpu.cfs_quota_us / cpu.cfs_period_us) as quota / period; 0 when there is none.
// A rented container shows all cores of the host (256) but is throttled to its quota (27.3 CPUs on the 5090 box): 127 threads on it were paused for most of every
// 100 ms period, 96 ms per token at a 95% cache hit rate and 1% of the RAM and PCIe bandwidth in use.
static double common_cpu_quota() {
#if defined(__linux__)
    if (FILE * f = fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char q[32] = {}; long period = 0;
        const bool ok = fscanf(f, "%31s %ld", q, &period) == 2;
        fclose(f);
        if (ok && strcmp(q, "max") != 0 && period > 0) { return atof(q) / (double) period; }
        return 0.0;
    }
    long quota = -1, period = 0;
    for (const char * dir : { "/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct" }) {
        FILE * fq = fopen((std::string(dir) + "/cpu.cfs_quota_us").c_str(), "r");
        FILE * fp = fopen((std::string(dir) + "/cpu.cfs_period_us").c_str(), "r");
        const bool ok = fq && fp && fscanf(fq, "%ld", &quota) == 1 && fscanf(fp, "%ld", &period) == 1;
        if (fq) { fclose(fq); }
        if (fp) { fclose(fp); }
        if (ok) { return quota > 0 && period > 0 ? (double) quota / (double) period : 0.0; }
    }
#endif
    return 0.0;
}

static int32_t common_cpu_get_num_math_raw();

int32_t common_cpu_get_num_math() {
    const int32_t n = common_cpu_get_num_math_raw();
    const double  q = common_cpu_quota();
    if (q > 0.0 && (int32_t) q < n) {
        char v[16], w[120];
        snprintf(v, sizeof v, "%d", std::max(1, (int32_t) q));
        snprintf(w, sizeof w, "container CPU quota %.1f cpus < %d cores seen: threads follow the quota", q, n);
        llama_moe_decision_fork("threads", v, w);
        return std::max(1, (int32_t) q);
    }
    return n;
}

/**
 * Returns number of CPUs on system that are useful for math.
 */
static int32_t common_cpu_get_num_math_raw() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
            cpuparams.auto_threads = true;
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    const int verbosity = common_log_get_verbosity_thold();
    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, verbosity);
    if (verbosity >= LOG_LEVEL_INFO) { common_cpu_topo_print_fork(); }

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices && verbosity >= LOG_LEVEL_TRACE) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
// only wildcards recurse, too many wildcards: no match
static bool glob_match(const char * pattern, const char * str, int depth = 0) {
    if (depth >= COMMON_RECURSIVE_LIMIT) {
        return false;
    }
    while (*pattern != '\0') {
        if (pattern[0] == '*' && pattern[1] == '*') {
            const char * p = pattern + 2;
            for (;; str++) {
                if (glob_match(p, str, depth + 1)) return true;
                if (*str == '\0') return false;
            }
        }
        if (*pattern == '*') {
            const char * p = pattern + 1;
            for (; *str != '\0' && *str != '/'; str++) {
                if (glob_match(p, str, depth + 1)) return true;
            }
            return glob_match(p, str, depth + 1);
        }
        if (*pattern == '?' && *str != '\0' && *str != '/') {
            pattern++;
            str++;
            continue;
        }
        if (*pattern == '[') {
            const char * class_end = pattern + 1;
            // If first character after '[' is ']' or '-', treat it as literal
            if (*class_end == ']' || *class_end == '-') {
                class_end++;
            }
            while (*class_end != '\0' && *class_end != ']') {
                class_end++;
            }
            if (*class_end == ']') {
                if (*str == '\0' || !glob_class_match(*str, pattern + 1, class_end)) return false;
                pattern = class_end + 1;
                str++;
                continue;
            }
            if (*str != '[') return false;
            pattern++;
            str++;
            continue;
        }
        if (*pattern != *str) return false;
        pattern++;
        str++;
    }
    return *str == '\0';
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}

std::string wstring_to_utf8(const std::wstring & str) {
    if (str.empty()) {
        return std::string();
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);

    if (size <= 0) {
        return std::string();
    }

    std::string utf8(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), &utf8[0], size, NULL, NULL);

    return utf8;
}
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fs_path_to_utf8(const std::filesystem::path & path) {
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
}

void fs_write_atomic(const std::filesystem::path & path, const std::string & data) {
    // a random suffix gives every writer its own temporary file, across processes and containers
    std::random_device rd;
    const uint64_t id = ((uint64_t) rd() << 32) | rd();

    std::error_code ec;
    std::filesystem::path path_tmp = path;
    path_tmp += "." + std::to_string(id) + ".tmp";

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }

    std::ofstream file(path_tmp, std::ios::binary);
    file << data;
    file.close();

    if (!file.fail()) {
        std::filesystem::rename(path_tmp, path, ec);
    }

    if (file.fail() || ec) {
        std::filesystem::remove(path_tmp, ec);
        throw std::runtime_error("failed to write file: " + fs_path_to_utf8(path));
    }
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

std::filesystem::path common_get_path_from_env(const std::string & name) {
#if defined(_WIN32)
    const std::wstring wname = utf8_to_wstring(name);
    const wchar_t * wvalue = _wgetenv(wname.c_str());
    return wvalue ? std::filesystem::path(wvalue) : std::filesystem::path();
#else
    const char * value = std::getenv(name.c_str());
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

#if !defined(_WIN32)
static std::filesystem::path get_home_directory() {
    std::filesystem::path home = common_get_path_from_env("HOME");
    if (!home.empty()) {
        return home;
    }
    const struct passwd * pw = getpwuid(getuid());
    if (!pw || !pw->pw_dir || !*pw->pw_dir) {
        throw std::runtime_error("Failed to find $HOME directory");
    }
    return pw->pw_dir;
}
#endif

std::filesystem::path fs_get_cache_directory() {
    std::filesystem::path cache_directory = common_get_path_from_env("LLAMA_CACHE");
    if (!cache_directory.empty()) {
        return cache_directory;
    }
#if defined(_WIN32)
    cache_directory = common_get_path_from_env("LOCALAPPDATA");
    if (cache_directory.empty()) {
        throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
    }
#elif defined(__APPLE__)
    cache_directory = get_home_directory() / "Library/Caches";
#else
    cache_directory = common_get_path_from_env("XDG_CACHE_HOME");
    if (cache_directory.empty()) {
        cache_directory = get_home_directory() / ".cache";
    }
#endif
    return cache_directory / "llama.cpp";
}

std::filesystem::path fs_get_config_directory() {
    std::filesystem::path config_directory;
#if defined(_WIN32)
    config_directory = common_get_path_from_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#else
    config_directory = common_get_path_from_env("XDG_CONFIG_HOME");
    if (config_directory.empty()) {
        config_directory = get_home_directory() / ".config";
    }
#endif
    return config_directory / "llama.cpp";
}

std::filesystem::path fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    const std::filesystem::path cache_directory = fs_get_cache_directory();
    std::error_code ec;
    common_create_directories(cache_directory, ec);
    if (ec) {
        throw std::runtime_error("failed to create cache directory: " + fs_path_to_utf8(cache_directory));
    }
    return cache_directory / std::filesystem::u8path(filename);
}

//
// TTY utils
//

bool common_is_tty(FILE * file) {
#if defined(_WIN32)
    return _isatty(_fileno(file));
#else
    return isatty(fileno(file));
#endif
}

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    return common_is_tty(stdout) || common_is_tty(stderr);
}

bool tty_enable_ansi() {
#if defined(_WIN32)
    // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
    for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
        HANDLE h    = GetStdHandle(id);
        DWORD  mode = 0;
        if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return false;
        }
    }
#endif
    return true;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    common_threadpools threadpools;

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

// --moe-expert-cache default (-2): a MoE model whose weights don't fit in free VRAM gets the
// expert cache setup (experts in RAM, no repack, auto-sized cache, 2048 ubatch) so a plain
// `-m model` runs the fast path; explicit user settings are kept
// model file size, all splits (0 if unknown)

// split.count as stored in the file; a key of another type (a malformed or foreign file) counts as an unsplit file instead of aborting in gguf_get_val_u16
static uint32_t gguf_split_count(const struct gguf_context * gguf, int64_t id) {
    return gguf_get_kv_type(gguf, id) == GGUF_TYPE_UINT16 ? gguf_get_val_u16(gguf, id) : 1;
}

static size_t common_model_file_size(const std::string & path_model) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gguf = gguf_init_from_file(path_model.c_str(), gp);
    if (!gguf) {
        return 0;
    }
    const int64_t id = gguf_find_key(gguf, "split.count");
    const uint32_t n_split = id < 0 ? 1 : std::max<uint32_t>(1, gguf_split_count(gguf, id));
    gguf_free(gguf);
    size_t size = 0;
    char prefix[4096], path[4096];
    const bool split = n_split > 1 && llama_split_prefix(prefix, sizeof(prefix), path_model.c_str(), 0, n_split);
    for (uint32_t i = 0; i < n_split; i++) {
        if (split) {
            llama_split_path(path, sizeof(path), prefix, i, n_split);
        }
        std::error_code ec;
        const size_t s = std::filesystem::file_size(split ? std::string(path) : path_model, ec);
        size += ec ? 0 : s;
    }
    return size;
}

// bytes of the routed-expert tensors (*_exps), over all splits: the rest of the file is what the GPU has to hold before a cache slot fits
// bytes of the tensors that need no VRAM for the expert cache to fit: the experts and the (per-layer) token embeddings
static size_t common_model_expert_bytes(const std::string & path_model) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g0 = gguf_init_from_file(path_model.c_str(), gp);
    if (!g0) {
        return 0;
    }
    const int64_t id = gguf_find_key(g0, "split.count");
    const uint32_t n_split = id < 0 ? 1 : std::max<uint32_t>(1, gguf_split_count(g0, id));
    gguf_free(g0);
    char prefix[4096], path[4096];
    const bool split = n_split > 1 && llama_split_prefix(prefix, sizeof(prefix), path_model.c_str(), 0, n_split);
    size_t bytes = 0;
    for (uint32_t k = 0; k < n_split; k++) {
        if (split) {
            llama_split_path(path, sizeof(path), prefix, k, n_split);
        }
        gguf_context * g = gguf_init_from_file(split ? path : path_model.c_str(), gp);
        if (!g) {
            continue;
        }
        for (int64_t t = 0; t < gguf_get_n_tensors(g); t++) {
            // the experts, and the token embeddings: they stay in host memory (a lookup, one row per token, e.g. the 26.8 GiB per_layer_token_embd of Qwen3.8-Flash-Next),
            // so neither counts against the VRAM the cache needs besides the non-expert weights
            const char * name = gguf_get_tensor_name(g, t);
            if (strstr(name, "_exps") || strstr(name, "token_embd")) {
                bytes += gguf_get_tensor_size(g, t);
            }
        }
        gguf_free(g);
    }
    return bytes;
}

// free memory summed over GPUs, of the largest GPU, and the GPU count
static std::string g_gpu_id;       // the GPUs' names and PCI bus ids, hashed: identifies the hardware (and the slot, so the link) in the placement record
static size_t g_gpu_total_max = 0; // fallback identity when the backend has no PCI id; free VRAM is not used: it moves with every browser tab

// the slot of a GPU, "x16g4": negotiated link width (idle cards keep their width, only the speed drops) and the highest speed the card and its parent port
// both support (the card alone says x16 in an x4 slot). Linux sysfs; empty: unknown
static std::string common_pci_slot(const char * bus_id) {
#ifdef __linux__
    if (!bus_id || strlen(bus_id) < 12) { return ""; }
    std::string id = bus_id + (strlen(bus_id) - 12);   // "0000:01:00.0" whether the backend wrote 4 or 8 domain digits
    for (char & c : id) { c = (char) tolower((unsigned char) c); }
    const std::string dir = "/sys/bus/pci/devices/" + id;
    auto rd = [](const std::string & f) {
        std::string v;
        if (FILE * fp = fopen(f.c_str(), "r")) {
            char b[64] = {0};
            if (fgets(b, sizeof b, fp)) { v = b; }
            fclose(fp);
        }
        while (!v.empty() && isspace((unsigned char) v.back())) { v.pop_back(); }
        return v;
    };
    const std::string w = rd(dir + "/current_link_width");
    if (w.empty() || w == "0") { return ""; }
    double gts = atof(rd(dir + "/max_link_speed").c_str());   // "16.0 GT/s PCIe"
    char real[PATH_MAX];
    if (realpath(dir.c_str(), real)) {
        std::string r = real;
        const size_t cut = r.find_last_of('/');
        if (cut != std::string::npos && cut > 0) {
            const double pg = atof(rd(r.substr(0, cut) + "/max_link_speed").c_str());   // the parent port: the slot
            if (pg > 0 && (gts <= 0 || pg < gts)) { gts = pg; }
        }
    }
    const int gen = gts >= 32 ? 5 : gts >= 16 ? 4 : gts >= 8 ? 3 : gts >= 5 ? 2 : gts > 0 ? 1 : 0;
    return "x" + w + "g" + std::to_string(gen);
#else
    GGML_UNUSED(bus_id);
    return "";
#endif
}

static void common_gpu_free(size_t & total, size_t & max, int & n) {
    total = max = 0;
    n = 0;
    g_gpu_total_max = 0;
    std::vector<std::string> ids;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free, tot;
        ggml_backend_dev_memory(dev, &free, &tot);
        total += free;
        max    = std::max(max, free);
        g_gpu_total_max = std::max(g_gpu_total_max, tot);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        ids.push_back(std::string(ggml_backend_dev_name(dev)) + "|" + (props.device_id ? props.device_id : "") + "|" + common_pci_slot(props.device_id) + "|" + std::to_string(tot >> 30));
        n++;
    }
    std::sort(ids.begin(), ids.end());   // the order the backend lists them in does not matter
    uint32_t h = 2166136261u;            // FNV-1a: the same on every platform, so a state file can move with its machine
    for (const auto & id : ids) { for (char c : id) { h = (h ^ (uint8_t) c) * 16777619u; } h *= 16777619u; }
    g_gpu_id = string_format("%08x", h);
    if (const char * lg = llama_moe_get_opt("log"); lg && strchr(lg, 'g')) {   // --moe log=g
         for (const auto & id : ids) { LOG_INF("%s: GPU identity of the placement record: %s -> %s\n", __func__, id.c_str(), g_gpu_id.c_str()); } }
}

// RAM that can be used without swapping: Linux MemAvailable (free + reclaimable page cache), 0 if unknown
// with_pool: also the free pages of a reserved huge page pool (1 GiB and 2 MiB, Linux): MemAvailable does not count them, but the pinned weights buffer is taken
// from exactly those pages, so for the decision to pin they are available RAM (a 100 GiB pool left 21 GiB "available" and a 109 GiB model fell back to mmap)
static size_t common_ram_available(bool with_pool = false) {
#if defined(_WIN32)
    GGML_UNUSED(with_pool);
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#elif defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemAvailable:") {
            size_t pool = 0;
            if (with_pool) {
                for (const auto & hp : { std::make_pair("/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages", (size_t) 1 << 30),
                                         std::make_pair("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages",    (size_t) 2 << 20) }) {
                    std::ifstream pf(hp.first);
                    size_t n = 0;
                    if (pf >> n) { pool += n * hp.second; }
                }
            }
            if (with_pool) {   // a warm weights cache (GGML_CUDA_HUGEFS) holds pool pages already: its files are the pinned buffer the load maps, so they count too
                if (const char * hd = getenv("GGML_CUDA_HUGEFS")) {
                    std::error_code ec;
                    for (const auto & e : std::filesystem::directory_iterator(hd, ec)) {
                        if (e.is_regular_file(ec) && e.path().extension() == ".w") { pool += (size_t) e.file_size(ec); }
                    }
                }
            }
            size_t mem = kb * 1024;
            // container limit (cgroup v2): MemAvailable is the host's, the container is killed at memory.max (rented box: 85 GiB limit, 89.7 GiB pin budget, OOM kill
            // during the load). Headroom = limit - (usage - page cache, which is reclaimable).
            {
                std::ifstream mx("/sys/fs/cgroup/memory.max"), cu("/sys/fs/cgroup/memory.current"), st("/sys/fs/cgroup/memory.stat");
                size_t lim = 0, cur = 0, file = 0;
                if ((mx >> lim) && (cu >> cur) && lim < ((size_t) 1 << 60)) {
                    std::string k; size_t v = 0;
                    while (st >> k >> v) { if (k == "file") { file = v; break; } }
                    const size_t used = cur > file ? cur - file : 0;
                    mem = std::min(mem, lim > used ? lim - used : (size_t) 0);
                }
            }
            return mem + pool;
        }
        f.ignore(256, '\n');
    }
    return 0;
#else
    return 0;
#endif
}

// bytes swapped out since boot (Linux), to notice when pinning the weights pushed the system into swap
static size_t common_swap_out_bytes() {
#if defined(__linux__)
    std::ifstream f("/proc/vmstat");
    std::string key;
    size_t n = 0;
    while (f >> key >> n) {
        if (key == "pswpout") {
            return n * 4096;
        }
    }
#endif
    return 0;
}

// speculative defaults from model size vs free VRAM: the tuner's starting depth and, unless set,
// the max draft depth (recurrent-state rollback buffers are sized by it, so it costs VRAM)
void common_spec_auto(common_params & params) {
    auto & dft = params.speculative.draft;
    if (std::none_of(params.speculative.types.begin(), params.speculative.types.end(),
                     [](auto t) { return t != COMMON_SPECULATIVE_TYPE_NONE; })) {
        return;
    }
    size_t vram, vram_max;
    int    n_gpu;
    common_gpu_free(vram, vram_max, n_gpu);
    const double ratio = vram ? (double) common_model_file_size(params.model.path) / vram : 1e9;
    // measured: Qwen3.8-Flash-Next (1.9x VRAM, 95% cache hits) is fastest at 2, GLM-5.3-Flash (2.3x) at 0
    dft.n_start = ratio <= 1 ? 3 : ratio <= 2 ? 2 : 0;
    if (!dft.n_max_user) {
        dft.n_max = ratio <= 1 ? 5 : 2; // bigger than VRAM: 2 measured best (Qwen); --spec-draft-n-max raises it
    }
    if (dft.n_start == 0 && !dft.n_max_user && params.speculative.has_dft()) {
        // measured (GLM-5.3-Flash, 2.3x VRAM, separate -md draft with MoE experts): the draft's VRAM
        // costs the expert cache more than drafting gains, even at depth 0; --spec-draft-n-max N
        // loads it anyway. Built-in MTP (no -md) is kept: MiMo-V2.6 (2.8x VRAM, 3 dense MTP layers)
        // measured +9% with it, so the depth tuner starts at 0 and measures instead.
        LOG_WRN("%s: model is %.1fx free VRAM: speculative decoding would be slower here, not loading it "
                "(--spec-draft-n-max N to force)\n", __func__, ratio);
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
        dft.mparams = {};
        return;
    }
    LOG_INF("%s: model %.1fx free VRAM: draft depth starts at %d, max %d\n", __func__, ratio, dft.n_start, dft.n_max);
}

// Measured choice between the MoE expert cache and stock placement (the model fitter's layers, no cache), per model,
// GPU set and build, without loading twice: one start runs stock placement, the next runs the cache, each records its prompt
// and generation time per token at exit; then the faster mode is kept (file in the cache dir). The cache is kept only when
// its generation is not slower than stock and its prompt time per token is below LLAMA_MOE_AUTO_PREFILL_SLOWDOWN (2) x stock.
// LLAMA_MOE_AUTO_MODE=stock|cache forces a mode, =retest forgets the decision.
struct moe_auto_rec { bool have = false; double p_ms = -1, g_ms = 0, g_max = 0, p_max = 0; int n = 0; int failed_mib = 0; std::vector<double> pr, gr; double last_p = -1, last_g = -1; }; // fastest and slowest per-token times seen (the spread is the noise), n: recorded runs; p_ms < 0: no prompt of real size measured yet; failed_mib: free VRAM (MiB) when it could not start (out of memory)
static std::string g_moe_auto_file;  // decision file of this run (empty: auto choice not involved)
static double g_moe_auto_np = 0, g_moe_auto_ng = 600; // the request profile of the decision, for the one made when this run is recorded
static std::string g_moe_auto_mode;  // mode this run explores: "stock" or "cache" (empty: decided already)
static bool g_moe_auto_user_ts = false;   // the user gave a layer split (-ts): a start failure then says nothing about the placement and is not recorded
static size_t      g_moe_auto_vram_free = 0; // free VRAM when this run chose its placement

// The placement records live in the engine's state file (llama_state_*, one INI section per model):
//   [<model file> <bytes>]  place.g<gpus>x<GiB of the largest GPU>g.v2.stock|cache = <ms/prompt token> <ms/generated token> <runs>  and  .decided
// moe_auto_path() returns "<section>\x1f<key prefix>"; the same section is handed to the engine for its own keys (hot experts, tuner)
static std::string moe_auto_path(const common_params & params, size_t model_size, int n_gpu, size_t vram_max) {
    std::string name = params.model.path;
    const size_t sl = name.find_last_of("/\\");
    if (sl != std::string::npos) { name = name.substr(sl + 1); }
    for (auto & c : name) { if (c == '[' || c == ']' || c == '\n' || c == '\r') { c = '_'; } }
    const std::string section = name + " " + std::to_string(model_size);
    llama_state_set_model(section.c_str());
    const std::string place = "place.g" + std::to_string(n_gpu) + "." + (g_gpu_id.empty() ? std::to_string(((g_gpu_total_max ? g_gpu_total_max : vram_max) + ((size_t) 1 << 29)) >> 30) + "g" : g_gpu_id) + ".v3";
    llama_state_set_place(place.c_str());   // the engine keeps the seen link bandwidth under it
    return section + "\x1f" + place; // v1: the placement rule's epoch, not the build, so a new release does not re-measure (LLAMA_MOE_AUTO_MODE=retest forgets)
}

static void moe_auto_split(const std::string & path, std::string & section, std::string & prefix) {
    const size_t p = path.find('\x1f');
    section = path.substr(0, p);
    prefix  = p == std::string::npos ? "" : path.substr(p + 1);
}

// per-token times of the last runs of a placement (oldest first, at most 9) are kept next to the record: the value a decision uses is their median once there are
// three (fewer: the fastest, so a cold first run never decides), the slowest is the noise, the last one shows drift
static void moe_auto_refresh(moe_auto_rec & r) {
    auto med_or_min = [](const std::vector<double> & v) {
        std::vector<double> t = v;
        std::sort(t.begin(), t.end());
        return t.size() >= 3 ? t[t.size()/2] : t.front();
    };
    if (!r.gr.empty()) {
        r.g_ms = med_or_min(r.gr);
        r.g_max = *std::max_element(r.gr.begin(), r.gr.end());
        r.last_g = r.gr.back();
    }
    if (!r.pr.empty()) {
        r.p_ms = med_or_min(r.pr);
        r.p_max = *std::max_element(r.pr.begin(), r.pr.end());
        r.last_p = r.pr.back();
    }
}

static std::vector<double> moe_auto_parse_list(const char * s) {
    std::vector<double> v;
    char * end = nullptr;
    for (double x = strtod(s, &end); end != s; x = strtod(s, &end)) { v.push_back(x); s = end; }
    return v;
}

static std::string moe_auto_list(const std::vector<double> & v) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) { o += string_format(i ? " %.4f" : "%.4f", v[i]); }
    return o;
}

static void moe_auto_read(const std::string & path, moe_auto_rec & st, moe_auto_rec & ca, std::string & decided) {
    std::string section, prefix;
    moe_auto_split(path, section, prefix);
    char buf[128];
    for (moe_auto_rec * r : { &st, &ca }) {
        if (llama_state_get(section.c_str(), (prefix + (r == &st ? ".stock" : ".cache")).c_str(), buf, sizeof buf)) {
            r->have = sscanf(buf, "%lf %lf %d %lf %lf %d", &r->p_ms, &r->g_ms, &r->n, &r->g_max, &r->p_max, &r->failed_mib) >= 3;
            r->failed_mib = 0;   // old marks are ignored (see moe_auto_mark_failed)
            if (r->have && r->g_max < r->g_ms) { r->g_max = r->g_ms; }
            if (r->have && r->p_max < r->p_ms) { r->p_max = r->p_ms; }
            char lb[160];
            const std::string base = prefix + (r == &st ? ".stock" : ".cache");
            if (r->have && llama_state_get(section.c_str(), (base + ".g").c_str(), lb, sizeof lb)) { r->gr = moe_auto_parse_list(lb); }
            if (r->have && llama_state_get(section.c_str(), (base + ".p").c_str(), lb, sizeof lb)) { r->pr = moe_auto_parse_list(lb); }
            moe_auto_refresh(*r);
        }
    }
    if (llama_state_get(section.c_str(), (prefix + ".decided").c_str(), buf, sizeof buf)) { decided = buf; }
}

static void moe_auto_write(const std::string & path, const moe_auto_rec & st, const moe_auto_rec & ca, const std::string & decided) {
    // ini-fix (2026-10-22): placement records are not written: the placement is a static rule (2026-10-21) and nothing decides from them any more
    if (true) { (void) path; (void) st; (void) ca; (void) decided; return; }
    std::string section, prefix;
    moe_auto_split(path, section, prefix);
    auto put = [&](const char * k, const moe_auto_rec & r) {
        if (r.have || r.failed_mib) { llama_state_set(section.c_str(), (prefix + k).c_str(), string_format("%.4f %.4f %d %.4f %.4f %d", r.p_ms, r.g_ms, r.n, r.g_max, r.p_max, r.failed_mib).c_str()); } // p_ms -1: unknown
    };
    put(".stock", st);
    put(".cache", ca);
    for (const auto & kr : { std::make_pair(".stock", &st), std::make_pair(".cache", &ca) }) {
        if (!kr.second->gr.empty()) { llama_state_set(section.c_str(), (prefix + kr.first + ".g").c_str(), moe_auto_list(kr.second->gr).c_str()); }
        if (!kr.second->pr.empty()) { llama_state_set(section.c_str(), (prefix + kr.first + ".p").c_str(), moe_auto_list(kr.second->pr).c_str()); }
    }
    if (!decided.empty()) { llama_state_set(section.c_str(), (prefix + ".decided").c_str(), decided.c_str()); }
}

// the process could not start with the placement it was measuring (out of memory): record it with the free VRAM it saw, the other placement is used from the next start
// on (until that much more VRAM is free than at the failure, e.g. another program that held it has gone)
static void moe_auto_mark_failed() {
    // No stored "this placement failed" verdicts (2026-10-17): a PIN_STALL exit, an assert from a bug under test or a killed run counted as "the cache could not start with this much
    // free VRAM", and the next start of MiMo silently took stock placement. Every start decides from what is true now (the static rule); nothing is written, nothing is read back.
    return;
    if (g_moe_auto_mode.empty()) {
        return;
    }
    if (g_moe_auto_user_ts) {
        // With an explicit -ts the layer fit is skipped ("tensor_split already set by user, abort"), the experts stay on the GPUs and the start fails for lack of
        // VRAM whichever placement was chosen. Recording that made the tuner pick the stock placement for MiMo from then on (decode 14 t/s instead of 26).
        LOG_WRN("%s: the start failed with a layer split given by the user (-ts): not recorded as a failed placement\n", __func__);
        g_moe_auto_mode.clear();
        return;
    }
    moe_auto_rec st, ca;
    std::string decided;
    moe_auto_read(g_moe_auto_file, st, ca, decided);
    (g_moe_auto_mode == "stock" ? st : ca).failed_mib = std::max<int>(1, (int) (g_moe_auto_vram_free >> 20));
    moe_auto_write(g_moe_auto_file, st, ca, decided);
    LOG_ERR("%s: MoE placement: %s could not start with %.1f GiB of free VRAM, the other placement is used from the next start\n", __func__,
        g_moe_auto_mode.c_str(), g_moe_auto_vram_free / 1073741824.0);
    g_moe_auto_mode.clear();
}

// A CUDA out-of-memory is a GGML_ABORT, not an exception: the process dies and none of the failure handling above ran, so the next start chose the same placement and
// died the same way, for ever (RTX 4060 Laptop 8 GB, Qwen3.6-35B, b12686: three starts in a row, "CUDA error: out of memory" right after "MoE expert cache enabled").
// The abort callback records the failed placement with the free VRAM it saw (the other placement is used from the next start on), then the abort goes on.
static uint32_t g_moe_auto_ub_auto = 0;   // the micro-batch maximum we chose ourselves (0: the user's -ub or none): what a startup out-of-memory halves

// a CUDA out-of-memory while the server is still starting (the compute graph of a big micro-batch on a small card) is a GGML_ABORT: no exception, no retry.
// Restart this process with half the micro-batch (LLAMA_FORK_UB_CAP, read where the micro-batch is chosen) instead of dying, down to 512; the placement that
// failed is recorded first, so a later start with a different free VRAM does not repeat it. The user's own -ub is never changed.
static void moe_auto_abort_cb(const char * msg) {
    const bool oom = msg && (strstr(msg, "CUDA error") || strstr(msg, "out of memory"));
    if (msg) { fprintf(stderr, "%s\n", msg); fflush(stderr); }   // the callback replaces ggml's own print: without this a failed start is silent
    if (!oom) { return; }
    moe_auto_mark_failed();
    const char * tries = getenv("LLAMA_FORK_RETRY");
    const int    n_try = tries ? atoi(tries) : 0;
    if (g_moe_auto_ub_auto <= 512 || n_try >= 4) { return; }
    const std::string cap = std::to_string(g_moe_auto_ub_auto / 2), nt = std::to_string(n_try + 1);
    fprintf(stderr, "fork: out of memory while starting with micro-batch %u: restarting with %s\n", g_moe_auto_ub_auto, cap.c_str());
    fflush(stderr);
#if defined(_WIN32)
    _putenv_s("LLAMA_FORK_UB_CAP", cap.c_str());
    _putenv_s("LLAMA_FORK_RETRY", nt.c_str());
    STARTUPINFOW si = {}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    std::wstring cmd = GetCommandLineW();
    if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code);
        ExitProcess(code);
    }
#else
    setenv("LLAMA_FORK_UB_CAP", cap.c_str(), 1);
    setenv("LLAMA_FORK_RETRY", nt.c_str(), 1);
    std::vector<std::string> args;
    { std::ifstream f("/proc/self/cmdline", std::ios::binary); std::string a; while (std::getline(f, a, '\0')) { args.push_back(a); } }
    char exe[4096]; const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0 && !args.empty()) {
        exe[n] = 0;
        std::vector<char *> av; for (auto & a : args) { av.push_back(a.data()); } av.push_back(nullptr);
        execv(exe, av.data());
    }
#endif
}

// typical agentic coding turn when the command line says nothing (server, chat): the system prompt and tool definitions come from
// the KV cache, so a turn adds ~6000 new prompt tokens (tool output, file contents) and generates ~600 (edits, tool calls)
static double moe_auto_est_prompt(const common_params & params) {
    return params.prompt.empty() ? 6000.0 : std::max(1.0, params.prompt.size() / 4.0);
}

static double moe_auto_est_gen(const common_params & params) {
    return params.n_predict > 0 ? params.n_predict : 600.0;
}

// expected time of one request: cache wins when the generation it saves outweighs the prompt time it costs; prompt tokens are
// estimated from the -p text (4 chars/token) and -n, else the agentic coding turn above, so a prompt of several thousand
// tokens with a short answer picks stock and a short prompt with a long answer picks the cache
// the prompt cost per token is only comparable once both placements measured a real prompt: a few tokens time the fixed setup cost,
// and scaled to a server-sized prompt that picked stock for a cache that prefills 10x faster. Until then the prompt term drops out.
static bool moe_auto_p_known(const moe_auto_rec & st, const moe_auto_rec & ca) { return st.p_ms >= 0 && ca.p_ms >= 0; }

// The decision is made again on every start from everything recorded, so it can change its mind. Per placement the fastest per-token times
// seen are kept (a cold run never decides) and the slowest too: their spread is the noise of that placement. A gap smaller than the noise
// (at least 5%, and the 10% a single run is assumed to scatter by) keeps the previous decision, a clear gap changes it.
static double moe_auto_cost(const moe_auto_rec & r, double np, double ng, bool p_known) { return (p_known ? np * r.p_ms : 0.0) + ng * r.g_ms; }

static double moe_auto_noise(const moe_auto_rec & r) {
    return r.n >= 2 && r.g_ms > 0 ? std::min(0.25, (r.g_max - r.g_ms) / r.g_ms) : 0.10;
}

// relative gap between the placements' request costs (positive: the cache is faster) and the noise bar it has to beat
static double moe_auto_gap(const moe_auto_rec & st, const moe_auto_rec & ca, double np, double ng, double & bar) {
    const bool pk = moe_auto_p_known(st, ca);
    const double c = moe_auto_cost(ca, np, ng, pk), s = moe_auto_cost(st, np, ng, pk);
    bar = std::max(0.05, std::max(moe_auto_noise(st), moe_auto_noise(ca)));
    return (s - c) / std::max(1e-9, std::min(c, s));
}

static std::string moe_auto_choose(const moe_auto_rec & st, const moe_auto_rec & ca, double np, double ng, const std::string & prev) {
    double bar = 0;
    const double gap = moe_auto_gap(st, ca, np, ng, bar);
    if ((prev == "cache" || prev == "stock") && std::fabs(gap) <= bar) { return prev; }
    return gap >= 0 ? "cache" : "stock";
}

// the placement that is faster at one phase, "prefill" (p_ms) or generation (g_ms); a gap inside the noise keeps prev
static std::string moe_auto_pick(const moe_auto_rec & st, const moe_auto_rec & ca, bool gen, const std::string & prev) {
    if (!gen && !moe_auto_p_known(st, ca)) { return prev; }
    const double s = gen ? st.g_ms : st.p_ms, c = gen ? ca.g_ms : ca.p_ms;
    const double bar = std::max(0.05, std::max(moe_auto_noise(st), moe_auto_noise(ca)));
    const double gap = (s - c) / std::max(1e-9, std::min(s, c));
    if ((prev == "cache" || prev == "stock") && std::fabs(gap) <= bar) { return prev; }
    return gap >= 0 ? "cache" : "stock";
}

// each recorded run refreshes the two picks (better prefill, better generation); they only set where the next start begins, every run
// then measures and decides again
static void moe_auto_store_picks(const std::string & path, const moe_auto_rec & st, const moe_auto_rec & ca) {
    if (true) { (void) path; (void) st; (void) ca; return; }   // ini-fix (2026-10-22): see moe_auto_write
    std::string sec, pre; moe_auto_split(path, sec, pre);
    char b[16] = {0};
    for (int gen = 0; gen < 2; ++gen) {
        const std::string key = pre + (gen ? ".pick_g" : ".pick_p");
        const std::string prev = llama_state_get(sec.c_str(), key.c_str(), b, sizeof b) ? b : "";
        const std::string pick = moe_auto_pick(st, ca, gen, prev);
        if (!pick.empty()) { llama_state_set(sec.c_str(), key.c_str(), pick.c_str()); }
    }
}

static std::string moe_auto_decide_for(const moe_auto_rec & st, const moe_auto_rec & ca, const common_params & params, const std::string & path, const std::string & prev) {
    const double np = moe_auto_p_known(st, ca) ? moe_auto_est_prompt(params) : 0.0;
    const double ng = moe_auto_est_gen(params);
    std::string mode = moe_auto_choose(st, ca, np, ng, prev);
    {   // the recorded picks: where the two differ, the request decides which one counts (generation-heavy: generation over 10% longer than the prompt)
        std::string sec, pre; moe_auto_split(path, sec, pre);
        char b[16] = {0}, c[16] = {0};
        const bool hp = llama_state_get(sec.c_str(), (pre + ".pick_p").c_str(), b, sizeof b);
        const bool hg = llama_state_get(sec.c_str(), (pre + ".pick_g").c_str(), c, sizeof c);
        if (hp && hg) {
            const bool gen_heavy = ng * st.g_ms > 1.1 * np * std::max(0.0, st.p_ms);
            mode = strcmp(b, c) == 0 ? b : gen_heavy ? c : b;
            LOG_INF("%s: MoE placement: picks: prefill %s, generation %s; this request is %s-heavy -> %s\n", __func__, b, c, gen_heavy ? "generation" : "prefill", mode.c_str());
        }
    }
    // switchover: the prompt length below which the cache wins for this many generated tokens
    const double dp = ca.p_ms - st.p_ms, dg = st.g_ms - ca.g_ms;
    const double breakeven = !moe_auto_p_known(st, ca) || dp <= 0 ? INFINITY : dg <= 0 ? 0.0 : ng * dg / dp;
    LOG_INF("%s: MoE placement: request ~%.0f prompt + %.0f generated tokens; the cache wins for prompts under ~%.0f tokens at that answer length\n",
        __func__, np, ng, breakeven);
    return mode;
}

// one run of each placement decides when the CACHE wins by more than the noise; otherwise a second run of each is taken. The cache is never rejected on one run:
// its first run starts with no learned hot experts (IQ3_S on 2x RTX 3090: 33-40 t/s cold, 53 t/s warm) and a decision "stock" is never revisited, so a cold first
// trial would lock the model to stock speed for good
static bool moe_auto_clear_gap(const moe_auto_rec & st, const moe_auto_rec & ca, const common_params & params) {
    double bar = 0;
    const double np = moe_auto_p_known(st, ca) ? moe_auto_est_prompt(params) : 0.0;
    return moe_auto_gap(st, ca, np, moe_auto_est_gen(params), bar) > bar;
}

// -at off or --moe autotune=0: the kill switch for everything the engine tunes or measures by itself
bool common_autotune_on(const common_params & params) {
    return params.autotune;
}

// The context of a run without -c: the largest one that leaves the expert cache most of its VRAM (fork/docs/kv-fit.md, "What fits means"). The KV and the context-scaled
// graph buffers (attention mask, the indexer's score tensors of the sparse-attention models: GLM 73 KB per context token at a 2048 micro-batch, 7x its KV) are probed as
// ONE slope: two no_alloc contexts at 8192 and 32768 cells (micro-batch 2048, the size a prompt pass takes) give per card the bytes per cell; the cache VRAM left at 8192 cells
// may lose AUTO_CTX_SHARE of itself to a longer context. c = 8192 + share * cache0 / slope, the smallest over the cards, rounded down to 256, between 8192 and the model's maximum.
// Returns 0 when the probe fails (the caller keeps the fork's old default).
static uint32_t common_auto_ctx(common_params & params) {
    const double   AUTO_CTX_SHARE = 0.30;
    const uint32_t C1 = 8192, C2 = 32768;
    std::vector<int64_t> s1, s2, cache0;
    uint32_t trained = 0;
    for (int i : { 0, 1 }) {
        auto mp = common_model_params_to_llama(params);
        auto cp = common_context_params_to_llama(params);
        cp.n_ctx    = i == 0 ? C1 : C2;
        cp.n_ubatch = 2048;
        cp.n_batch  = std::max<uint32_t>(cp.n_batch, 2048);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t ngl = 0, nctx = 0, nexp = 0;
        try {
            const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, nctx, nexp, GGML_LOG_LEVEL_ERROR);
            trained = nctx;
            for (size_t d = 0; d < devs.size() && d < dmd.size(); ++d) {   // the last entry of dmd is the host
                const int64_t target = d < params.fit_params_target.size() ? (int64_t) params.fit_params_target[d] : 0;
                (i == 0 ? s1 : s2).push_back((int64_t) (dmd[d].context + dmd[d].compute));
                if (i == 0) { cache0.push_back(std::max<int64_t>(0, dmd[d].free - (int64_t) (dmd[d].model + dmd[d].context + dmd[d].compute) - target)); }
            }
        } catch (const std::exception & e) {
            LOG_WRN("%s: auto context probe failed (%s), the default context is used\n", __func__, e.what());
            return 0;
        }
    }
    if (s1.empty() || s1.size() != s2.size() || trained == 0) {
        LOG_WRN("%s: auto context probe gave no usable result (cards %zu/%zu, model max %u), the default context is used\n", __func__, s1.size(), s2.size(), trained);
        return 0;
    }
    double c = (double) trained;
    for (size_t d = 0; d < s1.size(); ++d) {
        const double slope = (double) (s2[d] - s1[d]) / (double) (C2 - C1);   // bytes per cell on this card
        if (slope <= 0.0) { continue; }
        c = std::min(c, (double) C1 + AUTO_CTX_SHARE * (double) cache0[d] / slope);
    }
    uint32_t ctx = (uint32_t) std::min<double>(c, (double) trained);
    ctx = std::max<uint32_t>(C1, ctx / 256 * 256);
    LOG_INF("%s: no -c: context %u tokens = min(model max %u, the size where the KV and the context-scaled graph buffers take %.0f%% of the expert cache VRAM)\n", __func__, ctx, trained, 100 * AUTO_CTX_SHARE);
    return std::min(ctx, trained);
}

// -ub from the request and the card, no VRAM tiers: the compute buffer of a ubatch is probed (no_alloc, per candidate) and what it
// takes from the expert cache is weighed against the prefill time it saves.
//   time per request, in decode-token units:  np / (PP_PER_TG * (ub/512)^PP_EXP)  +  ng * (1 + extra / cache_vram)
// extra = compute buffer of this ubatch minus that of 512, cache_vram = VRAM left for cache slots; the decode loss is taken as
// the share of cache slots lost. Constants measured: prefill grows with ub^0.6 (2.3x for 4x on a 16 GB card with experts in RAM,
// 1.8x on 2x RTX 3090 and on OLMoE) and is ~11x the decode rate; np / ng are the request profile (-p / -n, else the agentic estimate).
// What the model below stands for (measured GLM-5.3-Flash 3.0-bit pinned, 12k prompt, 2x RTX 3090, 2026-10-11, -ub 1024/2048/4096 = 177/275/370 t/s):
// one pass over a prompt chunk costs a fixed part a (uploading the experts the chunk uses, 4.1 s here) plus b per token (GPU compute, 1.7 ms here), so
//   pass time = a + ub * b,   prompt speed = ub / (a + ub * b),   upper limit 1/b (about 590 t/s here), 455 t/s at ub 8192, 490 at 12288.
// The ub^0.5-0.6 used below is that curve over 512..2048; it flattens further up. A bigger ubatch costs compute buffer VRAM, which the cache then lacks
// (decode -3% per doubling here), so the best ubatch is the one where the saved upload time per prompt stops paying for it: it depends on the link speed
// (a), the GPU (b) and the prompt length n: with passes = ceil(n / ub_max) equal chunks of n / passes tokens the last pass is never a nearly empty one that
// pays a whole upload. A slow link (PCIe 3.0, x4) raises a and moves the optimum up; a fast GPU lowers b and does the same.
static uint32_t common_auto_ubatch(common_params & params) {
    // pass time a + ub*b gives prompt speed ~ ub / (A + ub) with A = a / b (about 2400 tokens on GLM-5.3-Flash / 2x RTX 3090: 177/275/370 t/s at 1k/2k/4k)
    const double PP_A = 2400.0, PP_PER_TG = 11.0;
    // a candidate whose extra compute buffer would take more than this share of the VRAM the cache has is never offered: it ends in CUDA out of memory
    // (full GLM at -ub 8192 and 12288 on the 24 GB x4 card) or in a cache too small to decode from
    const double MAX_CACHE_SHARE = 0.75;
    // decode time lost per share of the cache VRAM the extra buffer takes: measured far below 1 (Qwen3-235B: ub 256 -> 2048 took 5.6 of 15 GiB and decode stayed 64.0 -> 64.0 t/s;
    // GLM about -3% per doubling), the old 1.0 kept a 13k-token prompt at 53 passes (941 t/s against stock 1206, 1368 at ub 2048)
    const double DEC_LOSS = 0.1;
    // candidates are computed, not a fixed ladder: from the largest batch the cache path takes (max-batch: below it a batch never streams) doubling up to 12288
    const char * mbo = llama_moe_get_opt("max-batch");
    const uint32_t mb = std::max<uint32_t>(8, mbo ? (uint32_t) atoi(mbo) : 31);
    std::vector<uint32_t> cand;
    for (uint32_t ub = mb + 1; ub < 12288; ub *= 2) { cand.push_back(ub); }
    cand.push_back(12288);
    const uint32_t LO = cand[0], MID = 2048;   // probe the smallest candidate and 2048, extrapolate the rest (the buffer is ~linear in ub)
    const uint32_t SAFE = 512;                 // what a failed probe keeps
    const int NC = (int) cand.size();
    std::vector<size_t> compute(2, 0);
    int64_t cache_vram = 0;
    for (int i : { 0, 1 }) {
        auto mp = common_model_params_to_llama(params);
        auto cp = common_context_params_to_llama(params);
        const uint32_t pu = i == 0 ? LO : MID;
        cp.n_ubatch = pu;
        cp.n_batch  = std::max(cp.n_batch, pu);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t ngl = 0, nctx = 0, nexp = 0;
        size_t sum = 0;
        try {
            const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, nctx, nexp, GGML_LOG_LEVEL_ERROR);
            for (size_t d = 0; d < devs.size() && d < dmd.size(); ++d) {   // the last entry of dmd is the host
                sum += dmd[d].compute;
                if (i == 0) {
                    const int64_t target = d < params.fit_params_target.size() ? (int64_t) params.fit_params_target[d] : 0;
                    cache_vram += std::max<int64_t>(0, dmd[d].free - (int64_t) (dmd[d].model + dmd[d].context + dmd[d].compute) - target);
                }
            }
        } catch (const std::exception & e) {
            COM_DBG("ubatch probe failed (%s), keeping %u\n", e.what(), SAFE);
            return SAFE;
        }
        compute[i] = sum;
    }
    if (cache_vram <= 0 || compute[1] <= compute[0]) {
        return SAFE;
    }
    const double np = moe_auto_est_prompt(params), ng = moe_auto_est_gen(params);
    uint32_t best = SAFE;
    double t_best = 1e300;
    for (int ci = 0; ci < NC; ++ci) {
        const uint32_t ub = cand[ci];
        const double extra = (double) (compute[1] - compute[0]) * (ub - LO) / (double) (MID - LO);
        if (extra > MAX_CACHE_SHARE * (double) cache_vram) {
            COM_DBG("ubatch %u: extra buffer %.2f GiB exceeds %.0f%% of the %.1f GiB cache VRAM, not offered\n", ub, extra / 1073741824.0, 100 * MAX_CACHE_SHARE, cache_vram / 1073741824.0);
            continue;
        }
        const double t = np / (PP_PER_TG * (ub / (PP_A + ub)) / (LO / (PP_A + LO))) + ng * (1.0 + DEC_LOSS * extra / (double) cache_vram);
        COM_DBG("ubatch %u: extra buffer %.2f GiB of %.1f GiB cache VRAM, est. %.0f decode-token units\n", ub, extra / 1073741824.0, cache_vram / 1073741824.0, t);
        if (t < t_best) { t_best = t; best = ub; }
    }
    LOG_INF("%s: ubatch %u (request ~%.0f prompt + %.0f generated tokens, compute buffer %.2f -> %.2f GiB at %u -> %u probed, rest extrapolated, %.1f GiB left for the cache)\n", __func__,
        best, np, ng, compute[0] / 1073741824.0, compute[1] / 1073741824.0, LO, MID, cache_vram / 1073741824.0);
    return best;
}

// The micro-batch of a MoE model that fits the free VRAM: the expert placement code returns early for it, so no chooser ran and the stock 512 stayed (Qwen3.6-35B, 12k prompt,
// 2x RTX 3090, 2026-10-21, -ub 512/1024/2048/4096/8192 = 5591/6830/7335/7302/7301 prompt tokens/s, decode unchanged; 128k prompt 512 -> 1024: +17%). There is no cache to compete
// with for VRAM and a short prompt only fills as many tokens as it has, so a bigger maximum costs nothing but the compute buffer. Same rule as the cache path: the maximum is the
// context capped at 16384 (the gains flatten from 8192, and a pass as long as a big context makes some attention masks grow with its square), halved while the compute buffer
// of that size would take more than half of the VRAM the weights and the context leave free on the cards. Never below the stock 512.
static uint32_t common_fits_ubatch(common_params & params) {
    const uint32_t SAFE = 512;
    uint32_t top = 16384;
    if (params.n_ctx > 0) { top = std::min<uint32_t>(top, std::max<uint32_t>(SAFE, params.n_ctx)); }
    for (uint32_t ub = top; ub > SAFE; ub /= 2) {
        auto mp = common_model_params_to_llama(params);
        auto cp = common_context_params_to_llama(params);
        cp.n_ubatch = ub;
        cp.n_batch  = std::max(cp.n_batch, ub);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t ngl = 0, nctx = 0, nexp = 0;
        try {
            const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, nctx, nexp, GGML_LOG_LEVEL_ERROR);
            int64_t left = 0, compute = 0;
            for (size_t d = 0; d < devs.size() && d < dmd.size(); ++d) {   // the last entry of dmd is the host
                left    += (int64_t) dmd[d].free - (int64_t) (dmd[d].model + dmd[d].context);
                compute += (int64_t) dmd[d].compute;
            }
            if (left > 0 && compute * 2 <= left) { return ub; }
        } catch (const std::exception & e) {
            COM_DBG("ubatch probe failed (%s), keeping %u\n", e.what(), SAFE);
            return SAFE;
        }
    }
    return SAFE;
}

static void decide_fork(const char * name, const std::string & value, const std::string & why) { llama_moe_decision_fork(name, value.c_str(), why.c_str()); }
static void common_moe_cache_auto_impl(common_params & params);

// --moe sim-vram=MB / sim-ram=MB: leave exactly MB MiB free on every GPU / in RAM for the rest of the run, by really taking the memory (test runs on a "smaller
// machine"; sim-vram used to only change what the card reported, which let real allocations go past it). Held for the life of the process and freed by the OS when it ends, however it ends: the external
// holder process this replaces left 8 GiB per card taken after its starter was killed and tainted later runs (2026-10-18). Runs before any size is read.
static void common_hog(void) {
    static std::vector<ggml_backend_buffer_t> gpu_hog;
    static void * ram_hog = nullptr;
    const char * hg = llama_moe_get_opt("sim-vram");
    const char * hr = llama_moe_get_opt("sim-ram");
    if (hg && gpu_hog.empty()) {
        const size_t keep = (size_t) (atof(hg) * 1048576.0);
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            size_t free = 0, tot = 0;
            ggml_backend_dev_memory(dev, &free, &tot);
            if (free <= keep) { continue; }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), free - keep);
            if (!buf) { LOG_WRN("sim-vram: could not take %.2f GiB on %s\n", (free - keep) / 1073741824.0, ggml_backend_dev_name(dev)); continue; }
            gpu_hog.push_back(buf);
            size_t f2 = 0; ggml_backend_dev_memory(dev, &f2, &tot);
            LOG_WRN("sim-vram: %s holds %.2f GiB, %.2f GiB left free\n", ggml_backend_dev_name(dev), ggml_backend_buffer_get_size(buf) / 1073741824.0, f2 / 1073741824.0);
        }
    }
    // --moe hog-<card>=MB (hog-CUDA0=7943): take MB MiB on that one card, a control run that holds what something else would (a DSpark draft took 7943 MiB of the
    // x4 card's expert pool: the same cut without the draft separates the smaller cache from the cost of drafting). The --moe parser stores every key in
    // lower case (hog-cuda0), so the card name is looked up in lower case too: with the name as the backend spells it no key ever matched (2026-10-27)
    static bool hog_done = false;
    for (size_t i = 0; i < ggml_backend_dev_count() && !hog_done; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string key = std::string("hog-") + ggml_backend_dev_name(dev);
        for (char & c : key) { c = (char) tolower((unsigned char) c); }
        const char * h = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ? llama_moe_get_opt(key.c_str()) : nullptr;
        if (!h || atof(h) <= 0) { continue; }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), (size_t) (atof(h) * 1048576.0));
        if (!buf) { LOG_WRN("hog: could not take %s MiB on %s\n", h, ggml_backend_dev_name(dev)); continue; }
        gpu_hog.push_back(buf);
        LOG_WRN("hog: %s holds %.0f MiB\n", ggml_backend_dev_name(dev), ggml_backend_buffer_get_size(buf) / 1048576.0);
    }
    hog_done = true;
#ifdef __linux__
    if (hr && !ram_hog) {
        const size_t keep = (size_t) (atof(hr) * 1048576.0);   // stress test: exactly what was asked for, no floor (swapping is part of what is being tested)
        size_t avail = 0;
        if (FILE * f = fopen("/proc/meminfo", "r")) {
            char line[256]; unsigned long long kb = 0;
            while (fgets(line, sizeof line, f)) { if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { avail = (size_t) kb * 1024; break; } }
            fclose(f);
        }
        if (avail > keep) {
            const size_t n = avail - keep;
            void * p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p != MAP_FAILED) {
                for (size_t o = 0; o < n; o += 4096) { ((volatile char *) p)[o] = 1; }
                mlock(p, n);
                ram_hog = p;
                LOG_WRN("sim-ram: holds %.2f GiB, about %.2f GiB left\n", n / 1073741824.0, keep / 1073741824.0);
            }
        }
    }
#elif defined(_WIN32)
    if (hr && !ram_hog) {
        const size_t keep = (size_t) (atof(hr) * 1048576.0);   // same rule as above: exactly what was asked for, no floor
        MEMORYSTATUSEX ms; ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms) && ms.ullAvailPhys > keep) {
            const size_t n = (size_t) ms.ullAvailPhys - keep;
            SetProcessWorkingSetSize(GetCurrentProcess(), n + ((size_t) 256 << 20), (size_t) ms.ullTotalPhys);   // VirtualLock needs the room in the working set; the maximum must stay far above the hog (later pinned host buffers lock pages too)
            void * p = VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (p) {
                for (size_t o = 0; o < n; o += 4096) { ((volatile char *) p)[o] = 1; }
                VirtualLock(p, n);
                ram_hog = p;
                LOG_WRN("sim-ram: holds %.2f GiB, about %.2f GiB left\n", n / 1073741824.0, keep / 1073741824.0);
            }
        }
    }
#endif
}

// pinned weights for every model, whether or not the expert cache is on (OLMoE fits the VRAM, the placement code returns early, and its weights were mapped: 2026-10-21)
// The pinned host buffer of the largest prompt pass (graph inputs: the attention masks and the model's per-token inputs, GLM-5.3-Flash 598,200 bytes per token
// + 2 per token per cell), measured with a no-alloc probe of that pass before any weight is pinned. Pinning big models to the last GiB left GLM 3.0 (109.4 GiB)
// a 1.5 GiB pass budget: its 12k prompt ran as 5 passes instead of 3, 520 instead of 641 prompt t/s (2026-10-22). The pass is the micro-batch ceiling up to
// 8192 (the pass a 24 GB card can lend for the big models) at the context the cells of a long prompt reach (32k at most).
static size_t common_pass_buffer_reserve(common_params & params) {
    auto mp = common_model_params_to_llama(params);
    auto cp = common_context_params_to_llama(params);
    const uint32_t ub = std::min<uint32_t>(params.n_ubatch > 0 ? params.n_ubatch : 512, 8192);
    cp.n_ubatch = ub;
    cp.n_batch  = std::max(cp.n_batch, ub);
    cp.n_ctx    = params.n_ctx > 0 ? std::min<uint32_t>(params.n_ctx, 32768) : 32768;
    std::vector<ggml_backend_dev_t> devs;
    uint32_t ngl = 0, nctx = 0, nexp = 0;
    try {
        const auto dmd = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, nctx, nexp, GGML_LOG_LEVEL_ERROR);
        if (!dmd.empty()) { return dmd.back().compute; }   // the last entry is the host
    } catch (const std::exception & e) {
        COM_DBG("pass buffer probe failed (%s)\n", e.what());
    }
    return 0;
}

static void common_auto_pin(common_params & params) {
    const size_t model_size = common_model_file_size(params.model.path);
    // pinned weights (no mmap, llama-server only: startup takes longer, pays off over many requests):
    // cache uploads and prompt processing read them by direct DMA,
    // measured +46% prompt processing, +3-5% decode on GLM-5.3-Flash; used whenever the model fits
    // in the RAM available now minus the margin below; a model that does not is pinned up to that limit and the rest is mapped
    if (params.auto_pin && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        const size_t avail = common_ram_available(true);
        // RAM kept for the system, the KV cache in RAM and the page cache of the mapped rest: everything else is pinned (2026-10-21: 12 GiB, was 0 when the model fitted and 4 GiB when it did not),
        // plus the largest prompt pass's pinned host buffer (user 2026-10-22: a full-size pass comes before how much of the model is pinned)
        const size_t pass_buf = common_pass_buffer_reserve(params);
        const size_t margin = ((size_t) 12 << 30) + pass_buf;
        LOG_INF("%s: %.1f GiB of RAM kept free for the pinned host buffer of the largest prompt pass\n", __func__, pass_buf / 1073741824.0);
        if (avail && model_size + margin <= avail) {
            params.load_mode        = LLAMA_LOAD_MODE_NONE;
            params.load_pinned_auto = true;
        } else if (avail) {
            // the model does not fit pinned RAM: pin as much as the machine can spare, in tensor order, and map the rest. Never the whole model through mmap behind the
            // user's back. The budget is the reserved huge-page pool (memory nothing else can use) plus the available RAM minus a safety margin for the system.
            size_t pool         = avail - common_ram_available(false);
            const size_t mem    = avail - pool;
            // pages that cache files of earlier loads hold in the pool are not free now but are evicted when this load needs them (ggml_cuda_hugefs_evict):
            // count the whole 1 GiB pool, else a warm cache file shrinks the budget, the next load gets another buffer size and key, and the two files evict each
            // other (GLM 12k prompt cell 2026-10-14: 43.6 GiB pinned, cold fill, rest through the page cache, 24 instead of 700+ prompt t/s)
            if (getenv("GGML_CUDA_HUGEFS")) {
                if (FILE * f = fopen("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages", "r")) {
                    unsigned long long n = 0;
                    if (fscanf(f, "%llu", &n) == 1) { pool = std::max(pool, (size_t) n << 30); }
                    fclose(f);
                }
            }
            params.load_pin_budget = pool + (mem > margin ? mem - margin : 0);
            llama_moe_set_opt("pin-budget", std::to_string(params.load_pin_budget).c_str());
        }
    }
}

static void common_moe_cache_auto(common_params & params) {
    common_hog();
    common_moe_cache_auto_impl(params);
    common_auto_pin(params);
    // The VRAM the expert pool leaves free for CUDA's own use after it is the need an earlier session measured on the same card, driver and build (the
    // engine stores it, vram_margin), else 384 MiB; --moe margin_mb=N sets it by hand. The rule that stood here (0.1 MiB per micro-batch token, 8% on
    // small cards) set margin_mb, which no engine setting read until 2026-10-22: every pool was sized with the 384 MiB default. The build is part of the
    // stored need's key (a new build measures again).
    params.moe_opts += (params.moe_opts.empty() ? "" : ",") + std::string("build-id=") + std::to_string(llama_build_number()) + "-" + llama_commit();
    COM_DBG("moe_cache_slots=%d ctx=%d batch=%d ubatch=%d threads=%d threads_batch=%d repack=%s ngl=%d tensor_overrides=%zu fit=%s\n",
        params.n_moe_cache_slots, params.n_ctx, params.n_batch, params.n_ubatch, params.cpuparams.n_threads,
        params.cpuparams_batch.n_threads, params.no_extra_bufts ? "off" : "on", params.n_gpu_layers,
        (size_t) std::count_if(params.tensor_buft_overrides.begin(), params.tensor_buft_overrides.end(), [](const auto & o) { return o.pattern != nullptr; }),
        params.fit_params ? "on" : "off");
}

static void common_moe_cache_auto_impl(common_params & params) {
    if (params.n_moe_cache_slots != -2) {
        COM_DBG("expert cache set by user (%d), no auto setup\n", params.n_moe_cache_slots);
        return;
    }
    params.n_moe_cache_slots = 0;

    const bool user_placement = params.n_gpu_layers != -1 ||
        (!params.tensor_buft_overrides.empty() && params.tensor_buft_overrides[0].pattern != nullptr);
    if (user_placement) {
        COM_DBG("%s\n", "user tensor placement (-ngl/-ot/--cpu-moe), expert cache stays off unless --moe-expert-cache is set");
        return;
    }

    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gguf = gguf_init_from_file(params.model.path.c_str(), gp);
    if (!gguf) {
        return;
    }
    auto get_uint = [&](const std::string & key) -> uint32_t {
        const int64_t id = gguf_find_key(gguf, key.c_str());
        if (id < 0) {
            return 0;
        }
        switch (gguf_get_kv_type(gguf, id)) {
            case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf, id);
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf, id);
            default:               return 0;
        }
    };
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    const std::string arch = arch_id < 0 ? "" : gguf_get_val_str(gguf, arch_id);
    const uint32_t n_expert = get_uint(arch + ".expert_count");
    gguf_free(gguf);
    if (n_expert == 0) {
        COM_DBG("%s\n", "not a MoE model, no expert cache");
        return;
    }

    const size_t model_size = common_model_file_size(params.model.path);
    size_t vram_free, vram_max;
    int    n_gpu;
    common_gpu_free(vram_free, vram_max, n_gpu);
    // ponytail: weights vs free VRAM minus 1 GiB per GPU for context/compute; fit handles the borderline rest
    const size_t reserve = (size_t) n_gpu << 30;
    // --moe mode=cache is honored even when the model fits: the fit path returns before the placement is decided
    const char * forced_mode = llama_moe_get_opt("mode");
    const bool forced_cache = forced_mode && std::string(forced_mode) == "cache";
    if (n_gpu == 0 || (model_size + reserve <= vram_free && !forced_cache)) {
        COM_DBG("MoE model (%.1f GiB) fits free VRAM (%.1f GiB on %d GPUs), no expert cache\n",
            model_size / 1073741824.0, vram_free / 1073741824.0, n_gpu);
        if (n_gpu > 0 && common_autotune_on(params)) {
            uint32_t ub = common_fits_ubatch(params);
            if (params.ubatch_user) { ub = std::min<uint32_t>(ub, params.n_ubatch); }   // -ub is a clamp on the largest micro-batch, not a fixed size (user 2026-10-25: dense ubatch sweep)
            params.n_batch  = std::max<uint32_t>(params.n_batch, 2 * ub);
            if (params.n_ctx > 0) { params.n_batch = std::min<uint32_t>(params.n_batch, std::max<uint32_t>(params.n_ctx, ub)); }
            params.n_ubatch = ub;
            decide_fork("placement", "fits VRAM (stock)", string_format("model fits the free VRAM (%.1f GiB), no expert cache", vram_free / 1073741824.0));
            decide_fork("ubatch", std::to_string(ub), "model fits the VRAM: largest pass up to min(context, 16384) whose compute buffer is at most half of the free VRAM");
            LOG_INF("%s: ubatch %u: the model fits the VRAM, no cache to weigh (the largest pass up to min(context, 16384) whose compute buffer takes at most half of the free VRAM)\n", __func__, ub);
        }
        return;
    }

    bool use_cache = true;
    g_moe_auto_user_ts = false;
    for (size_t i = 0; i < llama_max_devices(); ++i) { g_moe_auto_user_ts = g_moe_auto_user_ts || params.tensor_split[i] != 0.0f; }
    {
        const std::string path = moe_auto_path(params, model_size, n_gpu, vram_max);
        const char * force = llama_moe_get_opt("mode");
        const std::string f = force ? force : "";
        if (f == "retest") { std::string sec, pre; moe_auto_split(path, sec, pre); llama_state_erase(sec.c_str(), pre.c_str()); }
        moe_auto_rec st, ca;
        std::string decided;
        moe_auto_read(path, st, ca, decided);
        std::string mode;
        g_moe_auto_vram_free = vram_free;
        // can the cache fit at all? Its slots come after the non-expert weights, the KV cache and the compute buffers; when those alone do not fit in the free VRAM
        // there is no cache to measure (GLM 3.0-bit on one 3090 with 10.9 GiB free: out of memory)
        const size_t expert_bytes = common_model_expert_bytes(params.model.path);
        const size_t cache_need   = expert_bytes && model_size > expert_bytes ? model_size - expert_bytes + ((size_t) 3 << 29) + ((size_t) n_gpu << 28) : 0; // + 1.5 GiB + 0.25 GiB per GPU
        if (f == "stock" || f == "cache") {
            mode = f;
            decide_fork("placement", mode, "set by hand: --moe mode=" + f);
        } else if (!common_autotune_on(params)) {
            // no measuring: the static rule, cache whenever the model does not fit (and the prompt is not long); nothing is recorded
            mode = moe_auto_est_prompt(params) < 3000 ? "cache" : "stock";
            decide_fork("placement", mode, string_format("autotune off: static rule, model %.1f GiB, free VRAM %.1f GiB", model_size / 1073741824.0, vram_free / 1073741824.0));
            LOG_INF("%s: MoE placement: %s (autotune off: static rule, model %.1f GiB, free VRAM %.1f GiB)\n", __func__, mode.c_str(),
                model_size / 1073741824.0, vram_free / 1073741824.0);
        } else if (cache_need > vram_free) {
            mode = "stock";
            decide_fork("placement", "stock", string_format("the cache cannot fit: %.1f GiB needed for non-expert weights, KV and compute buffers > free VRAM %.1f GiB", cache_need / 1073741824.0, vram_free / 1073741824.0));
            LOG_INF("%s: MoE placement: stock (the cache cannot fit: about %.1f GiB for the non-expert weights, KV cache and compute buffers > free VRAM %.1f GiB)\n", __func__,
                cache_need / 1073741824.0, vram_free / 1073741824.0);
        } else {
            // 2026-10-15 (user: "stored stock or cache is bad design, remove it"): the placement is a static rule, not a comparison of stored per-token
            // times. Those times moved with the prompt length, the split and the pass plan, so the same build on the same model landed on stock (a third of
            // the speed) on some starts and on the cache on others, and the numbers a first benchmark sees depended on what the state file held.
            // Rule: the cache when the model is clearly bigger than the free VRAM (it won 1.2x to 2.4x on every such model measured, and stock may not even
            // start there), stock when it fits or is near the fit line. 2026-10-21 (user: placement is measured, not read from the state file): a remembered failed start no longer decides either; it was an out of memory caused by another process on the GPU that locked a rented 5090 on stock (8 t/s instead of 40). The failure is still written (failed_mib), nothing reads it.
            // --moe mode=stock|cache decides by hand.
            const bool clearly_bigger = (double) model_size > 1.25 * (double) vram_free;
            mode = clearly_bigger ? "cache" : "stock";
            g_moe_auto_file = path;   // the run is still recorded (for the failed-start memory and for reading by hand), never read to decide
            g_moe_auto_mode = mode;
            decide_fork("placement", mode, string_format("static rule: model %.1f GiB, free VRAM %.1f GiB (%s)", model_size / 1073741824.0, vram_free / 1073741824.0, f.empty() ? "automatic" : "set by hand"));
            LOG_INF("%s: MoE placement: %s (static rule: model %.1f GiB, free VRAM %.1f GiB; --moe mode=stock|cache decides by hand; %s)\n", __func__,
                mode.c_str(), model_size / 1073741824.0, vram_free / 1073741824.0, path.c_str());
        }
        use_cache = mode != "stock";
        if (!g_moe_auto_mode.empty()) {
            ggml_set_abort_callback(moe_auto_abort_cb);   // a decision that is recorded may also be recorded as failed
        }
    }

    // stock placement (the model fitter places layers): only the expert cache and what exists for it stays off (experts on the
    // CPU, repack disabled, cache slots, the big prompt batch that its uploads want); pinned weights and the thread choice stay
    if (use_cache) {
        auto & tbo = params.tensor_buft_overrides;
        tbo.insert(std::find_if(tbo.begin(), tbo.end(), [](const auto & o) { return o.pattern == nullptr; }), llm_ffn_exps_cpu_override());
        params.no_extra_bufts    = true;
        params.n_moe_cache_slots = -1;
        // KV growth (fork/docs/kv-grow.md) is the default whenever the experts stream from RAM: the KV lives in the expert pool behind the compute lend, grows by the
        // model's step, and takes no VRAM that a context nobody asked for would hold. A run that fits in VRAM keeps the calculated static context (growth only adds a fixed
        // ~0.5 s there). --moe kv-tail=0 turns it off.
        if (params.moe_opts.find("kv-tail") == std::string::npos) {
            llama_moe_set_opt("kv-tail", "1");
            if (params.moe_opts.find("kv-token-major") == std::string::npos) { llama_moe_set_opt("kv-token-major", "1"); }
        }
    }
    // the default context is set before the micro-batch below, which takes the context as its maximum. Set after it (as until 2026-10-14), a run without -c
    // got the batch size instead: passes of 2048 tokens, Qwen3.8-Flash-Next 8k prompt in 4 passes at 798 prompt t/s where one pass gives 1201
    llama_moe_set_opt("kv-ctx-given", (params.n_ctx != 0 && params.n_predict < 0) ? "1" : "0");   // --moe kv-tail=1: an explicit -c caps the length of a request (prompt + answer), nothing is reserved for it; with -n present -n bounds the answer and -c is not applied
    if (params.n_ctx == 0) {
        const uint32_t fit_ctx = common_auto_ctx(params);
        llama_moe_set_opt("kv-budget-ctx", std::to_string(fit_ctx).c_str());   // the context the cache gets when the tail layout is declined (llama_context)
        params.n_ctx = fit_ctx ? fit_ctx : 17408; // ponytail: autofit would grow KV to n_ctx_train and starve the cache; -c N for more (17408 = the 16294-token "12k" benchmark prompt + 1k answer; with --moe kv-tail=1 there is no default at all)
    }
    if (common_autotune_on(params)) {
        uint32_t ub;
        // Experts in RAM: every prompt pass uploads all of them (4.1 s on full GLM 3.0-bit, about 2400 tokens' worth of compute), so a bigger pass amortizes the
        // upload until the per-token compute starts to rise (full models, 12k prompt: GLM 4096 -> 476 prompt tokens/s, 8192 -> 572, 12416 -> 573; MiMo 469 -> 599; compute-only
        // models peak at 2048-4096). The compute alias lends the pool's blocks to the graph, so the big maximum costs no cache while decoding; the per-prompt
        // rule (llama_moe_cache_prefill_ubatch_fork) then splits the prompt into equal passes up to this maximum, and the shrink-to-fit in the context lowers it
        // when the pool cannot lend the graph. Without the alias (--moe alias=0, pool=0, router prediction on) the old chooser below weighs the compute buffer against the cache.
        const char * o_alias = llama_moe_get_opt("alias");
        const char * o_pool  = llama_moe_get_opt("pool");
        const bool   alias_ok = use_cache && params.n_moe_predict <= 0 && (!o_alias || atoi(o_alias) != 0) && (!o_pool || atoi(o_pool) != 0);
        if (alias_ok) {
            // no fixed size: the best pass differs per model (12.4k prompt, full models: GLM flat from 8192 to 12416, MiMo 12416 +28% over 8192, Qwen3.8 8192).
            // The maximum is the context; the context's shrink-to-fit halves it until the pool can lend the graph, and the per-prompt rule picks equal passes under it.
            ub = params.n_ctx > 0 ? params.n_ctx : params.n_batch;
            ub = std::max<uint32_t>(512, ub);
            // cap: a pass as long as a big context makes the attention masks of some models grow with its square (GLM-5.3-Flash at -c 140000: the memory fit asked for
            // 432 GB and the server died before loading). Gains flatten from 8192 to 12416 tokens anyway; the per-prompt rule makes equal passes under this maximum.
            ub = std::min<uint32_t>(ub, 16384);
            decide_fork("ubatch", std::to_string(ub), "experts stream from RAM, the compute alias lends the graph (maximum = context, lowered to what the pool can lend)");
            LOG_INF("%s: ubatch %u: experts stream from RAM, the compute alias lends the graph (maximum = context, lowered to what the pool can lend; prompt passes are sized per request)\n", __func__, ub);
        } else {
            ub = common_auto_ubatch(params);
        }
        if (params.ubatch_user && params.n_ubatch < ub) {   // -ub clamps the micro-batch the auto rules chose (it never raises it): a sweep of the largest pass per card (4k, 8k, 16k)
            decide_fork("ubatch", std::to_string(params.n_ubatch), string_format("clamped by -ub from %u", ub));
            LOG_INF("%s: ubatch %u: clamped by -ub (the automatic maximum was %u)\n", __func__, params.n_ubatch, ub);
            ub = params.n_ubatch;
        }
        params.n_batch  = std::max<uint32_t>(params.n_batch, 2 * ub);   // a decode call carries several passes: equal passes are computed from the call's token count
        if (params.n_ctx > 0) {
            params.n_batch = std::min<uint32_t>(params.n_batch, std::max<uint32_t>(params.n_ctx, ub));
        }
        if (const char * cap = getenv("LLAMA_FORK_UB_CAP")) {   // set by moe_auto_abort_cb: the previous start of this process ran out of memory at this size
            ub = std::max<uint32_t>(128, std::min<uint32_t>(ub, (uint32_t) atoi(cap)));
            params.n_batch = std::min<uint32_t>(params.n_batch, std::max<uint32_t>(2 * ub, 512));
            LOG_WRN("%s: micro-batch capped to %u after an out-of-memory start\n", __func__, ub);
        }
        g_moe_auto_ub_auto = ub;
        params.n_ubatch = ub;
    }
    // no core is left per GPU any more (user 2026-10-26: the OS runs thousands of threads anyway; the GPU and upload threads share the cores), and the
    // decode search starts on every hardware thread and steps down if fewer are faster (common_init_result, llama_context::thread_tune_feed)
    // resource saturator: compute threads on physical cores only (SMT siblings add no memory bandwidth), filled L3 domain
    // by domain so each domain's threads compute its rows (GGML_MOE_CCX_SPLIT), leaving each domain's highest core to its
    // L3 prefetch thread. Linux sysfs / Windows GetLogicalProcessorInformationEx; only without a user CPU mask (LLAMA_AUTO_PLACE=1).
    // ponytail: no NUMA-node awareness yet, Windows processor group 0 only
    if (!params.cpuparams.mask_valid && params.cpuparams.auto_threads && llama_moe_get_opt("auto-place") && atoi(llama_moe_get_opt("auto-place")) != 0) {
        std::vector<std::vector<int>> dom; // physical cores (first SMT thread) per L3 domain
#if defined(_WIN32)
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
        std::vector<char> buf(len);
        if (len && GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            std::vector<KAFFINITY> l3, cores; // processor group 0 only (<= 64 logical CPUs)
            for (DWORD off = 0; off < len; ) {
                auto * e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
                if (e->Relationship == RelationCache && e->Cache.Level == 3 && e->Cache.GroupMask.Group == 0) { l3.push_back(e->Cache.GroupMask.Mask); }
                if (e->Relationship == RelationProcessorCore && e->Processor.GroupMask[0].Group == 0) { cores.push_back(e->Processor.GroupMask[0].Mask); }
                off += e->Size;
            }
            for (KAFFINITY m : l3) {
                std::vector<int> d;
                for (KAFFINITY c : cores) {
                    if (c != 0 && (c & m) == c) { int b = 0; while (!((c >> b) & 1)) { ++b; } d.push_back(b); }
                }
                std::sort(d.begin(), d.end());
                if (!d.empty()) { dom.push_back(d); }
            }
            std::sort(dom.begin(), dom.end());
        }
#elif defined(__linux__)
        auto rd = [](const std::string & path) { std::ifstream f(path); std::string v; std::getline(f, v); return v; };
        auto parse = [](const std::string & list) {
            std::vector<int> r;
            std::stringstream ss(list);
            std::string part;
            while (std::getline(ss, part, ',')) {
                const size_t d = part.find('-');
                const int a0 = atoi(part.c_str()), b0 = d == std::string::npos ? a0 : atoi(part.c_str() + d + 1);
                for (int k = a0; k <= b0; ++k) { r.push_back(k); }
            }
            return r;
        };
        std::vector<std::string> seen;
        for (int cpu = 0; cpu < GGML_MAX_N_THREADS; ++cpu) {
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
            const std::string sib = rd(base + "/topology/thread_siblings_list");
            if (sib.empty()) { break; }
            const std::vector<int> sl = parse(sib);
            if (sl.empty() || sl[0] != cpu) { continue; }
            std::string l3 = rd(base + "/cache/index3/shared_cpu_list");
            auto it = std::find(seen.begin(), seen.end(), l3);
            if (it == seen.end()) { seen.push_back(l3); dom.emplace_back(); it = seen.end() - 1; }
            dom[it - seen.begin()].push_back(cpu);
        }
#endif
        const int n_dom = (int) dom.size();
        const int nt = params.cpuparams.n_threads;
        int avail = 0;
        for (auto & d : dom) { avail += std::max<int>(0, (int) d.size() - 1); }
        if (n_dom >= 1 && avail >= nt) {
            // nt threads over the domains, contiguous per domain: domain d gets threads [d*nt/D, (d+1)*nt/D)
            std::vector<int> places;
            for (int d = 0; d < n_dom; ++d) {
                const int want = (d + 1) * nt / n_dom - d * nt / n_dom;
                for (int i = 0; i < want && i < (int) dom[d].size() - 1; ++i) { places.push_back(dom[d][i]); }
            }
            if ((int) places.size() == nt) {
                for (auto * cp : { &params.cpuparams, &params.cpuparams_batch }) {
                    std::fill(std::begin(cp->cpumask), std::end(cp->cpumask), false);
                    for (int c : places) { cp->cpumask[c] = true; }
                    cp->mask_valid = true;
                    cp->strict_cpu = true;
                }
                // the static per-L3 row split only serves the L3 prefetch: alone it lost 19% on GLM (3700X, fixed rows
                // per thread lose the dynamic chunking's load balance), while pinning itself was neutral (21.73 vs 21.53)
                const char * l3pf = llama_moe_get_opt("l3-pf");
                if (!getenv("GGML_MOE_CCX_SPLIT") && l3pf && atof(l3pf) > 0) {
#if defined(_WIN32)
                    _putenv_s("GGML_MOE_CCX_SPLIT", std::to_string(n_dom).c_str());
#else
                    setenv("GGML_MOE_CCX_SPLIT", std::to_string(n_dom).c_str(), 0);
#endif
                }
                std::string pl;
                for (int c : places) { pl += " " + std::to_string(c); }
                LOG_INF("%s: %d compute threads on physical cores%s (%d L3 domains)\n", __func__, nt, pl.c_str(), n_dom);
            }
        }
    }
    {
        // with the KV at the pool's tail and no -c, n_ctx here is only the fallback for when the tail layout is declined: say what the request really gets
        const char * kt = llama_moe_get_opt("kv-tail"), * kg = llama_moe_get_opt("kv-ctx-given");
        const bool tail_ctx = kt && atoi(kt) != 0 && !(kg && atoi(kg) != 0);
        LOG_INF("%s: MoE model (%.1f GiB) exceeds free VRAM (%.1f GiB): experts in RAM + GPU expert cache, no repack, "
            "ctx %s, ubatch %d, threads %d (settings you pass win; the weights are pinned as far as the RAM reaches, see common_init_result)\n", __func__, model_size / 1073741824.0, vram_free / 1073741824.0,
            tail_ctx ? (std::string("per request up to the model maximum (KV grows in the pool; ") + std::to_string(params.n_ctx) + " only if the tail layout is declined)").c_str()
                     : std::to_string(params.n_ctx).c_str(),
            params.n_ubatch, params.cpuparams.n_threads);
    }
}

static const std::map<common_decision_type, std::string> COMMON_DECISION_TYPE_NAMES = {
    { COMMON_DECISION_TYPE_OPENJEV,        "openjev"       },
    { COMMON_DECISION_TYPE_LEV,            "lev"           },
    { COMMON_DECISION_TYPE_KEV,            "kev"           },
    { COMMON_DECISION_TYPE_NIMBLE,         "nimble"        },
    { COMMON_DECISION_TYPE_LAYA,           "laya"          },
    { COMMON_DECISION_TYPE_CLEF,           "clef"          },
    { COMMON_DECISION_TYPE_PPLX_DECIDER,   "pplx-decider"  },
    { COMMON_DECISION_TYPE_VLLM_DECISION2, "vllm-decision2" },
    { COMMON_DECISION_TYPE_VLLM_D3,        "vllm-d3"        },
    { COMMON_DECISION_TYPE_LFM2_D1,        "lfm2-d1"       },
    { COMMON_DECISION_TYPE_LFM2_D1_OMNI,   "lfm2-d1-omni"  },
    { COMMON_DECISION_TYPE_GRANITE_SWITCH, "granite-switch"},
    { COMMON_DECISION_TYPE_H2O_LIGHTNING,  "h2o-lightning" },
    { COMMON_DECISION_TYPE_MINICPM_OCR_JEV, "openbmb-minicpm-ocr-jev" },
    { COMMON_DECISION_TYPE_LFM3_D1,        "lfm3-d1"       },
    { COMMON_DECISION_TYPE_LFM3_D1_OMNI,   "lfm3-d1-omni"  },
    // for future models, ALWAYS add org name prefix to avoid name collisions, e.g. "orgname-modelname"
};

static common_decision_type common_decision_type_from_string(const std::string & str) {
    for (const auto & pair : COMMON_DECISION_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return COMMON_DECISION_TYPE_UNKNOWN;
}

common_decision_type common_get_decision_type(const struct llama_model * model) {
    char buf[64];
    if (llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    const std::string key = std::string(buf) + ".decision.type";
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return COMMON_DECISION_TYPE_NONE;
    }
    return common_decision_type_from_string(buf);
}

common_gguf_info common_get_gguf_info(const std::string & fname) {
    common_gguf_info info;

    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(fname.c_str(), gguf_params));
    if (!gguf_ctx) {
        return info; // missing or unreadable file
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return info; // no architecture in the metadata
    }
    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch.empty()) {
        return info;
    }

    const int64_t type_id = gguf_find_key(gguf_ctx.get(), (arch + ".decision.type").c_str());
    if (type_id < 0) {
        info.decision_type = COMMON_DECISION_TYPE_NONE;
    } else if (gguf_get_kv_type(gguf_ctx.get(), type_id) == GGUF_TYPE_STRING) {
        info.decision_type = common_decision_type_from_string(gguf_get_val_str(gguf_ctx.get(), type_id));
    }

    // same key and type as the model loader
    const int64_t ctx_id = gguf_find_key(gguf_ctx.get(), (arch + ".context_length").c_str());
    if (ctx_id >= 0 && gguf_get_kv_type(gguf_ctx.get(), ctx_id) == GGUF_TYPE_UINT32) {
        info.n_ctx_train = gguf_get_val_u32(gguf_ctx.get(), ctx_id);
    }

    // same key as llama_model_chat_template(model, nullptr)
    info.has_chat_tmpl = gguf_find_key(gguf_ctx.get(), "tokenizer.chat_template") >= 0;

    return info;
}

void common_fit_apply(common_params & params, llama_model_params & mparams, llama_context_params & cparams) {
    if (!params.fit_params) {
        return;
    }

    COM_TRC("%s", "fitting params to device memory ...\n");
    COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");

    // the draft context is created from the same base params and follows the main context, fit both together
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    common_params params_dft = common_base_params_to_speculative(params);

    auto mparams_dft = common_model_params_to_llama(params_dft);
    auto cparams_dft = common_context_params_to_llama(params_dft);
    if (spec_mtp) {
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    cparams_dft.n_rs_seq = 0;

    const common_fit_extra_model extra = {
        /*.path_model   =*/ params_dft.model.path.c_str(),
        /*.mparams      =*/ &mparams_dft,
        /*.cparams      =*/ &cparams_dft,
        /*.shares_model =*/ !has_draft, // an MTP context runs on the weights of the main model
    };

    // --moe kv-tail=1 without -c: the KV memory is not reserved at load (it lives at the tail of the pool or in the arena, grown per request), so the fit must not count it
    // for the whole trained context: it pushed the experts of a 16 GB machine to the CPU (Qwen3.6-35B on 2 x 8 GB: 57 against 150 t/s). A small context here, the real one is set per request.
    const char * kt_fit = llama_moe_get_opt("kv-tail");
    const char * kg_fit = llama_moe_get_opt("kv-ctx-given");
    const uint32_t n_ctx_before_fit = cparams.n_ctx;
    if (kt_fit && atoi(kt_fit) != 0 && !(kg_fit && atoi(kg_fit) != 0)) { cparams.n_ctx = std::max<uint32_t>(params.fit_params_min_ctx, 4096); }
    common_fit_params(params.model.path.c_str(), &mparams, &cparams,
        params.tensor_split,
        params.tensor_buft_overrides.data(),
        params.fit_params_target.data(),
        params.fit_params_min_ctx,
        has_draft || spec_mtp ? &extra : nullptr,
        params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    if (kt_fit && atoi(kt_fit) != 0 && !(kg_fit && atoi(kg_fit) != 0)) { cparams.n_ctx = n_ctx_before_fit; }
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    common_moe_cache_auto(params);
    common_spec_auto(params);
    // the decode thread search starts on every hardware thread, SMT siblings included, and steps down while a step is not slower
    // (llama_context::thread_tune_feed). User 2026-10-26: on an EPYC all threads gained; too many threads costs little, too few a lot. The decode pool is
    // created this big and the search only lowers the count inside it. Only for a count that came from the default; the batch count is not changed here.
    if (params.cpuparams.auto_threads && common_autotune_on(params)) {
        params.cpuparams.n_threads = std::max<int32_t>(params.cpuparams.n_threads, (int32_t) std::thread::hardware_concurrency());
    }
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    common_fit_apply(params, mparams, cparams);

    const size_t swap0 = params.load_pinned_auto ? common_swap_out_bytes() : 0;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    // real memory pressure, not the kernel moving idle pages of other programs to swap (zram)
    const size_t swapped = params.load_pinned_auto ? common_swap_out_bytes() - swap0 : 0;
    const size_t avail   = params.load_pinned_auto ? common_ram_available() : 0;
    // no silent fallback to mmap: a pinned load that swapped or left little RAM is reported and the run goes on pinned (a different load mode behind the user's back
    // gave unexplained t/s); a failed load is an error. A model that does not fit is pinned as far as it reaches and the rest is mapped, and the log says how much of each.
    if (params.load_pinned_auto && model != NULL && (swapped > (2ull << 30) || (avail && avail < (512ull << 20)))) {
        LOG_WRN("%s: pinned weights left little RAM (swapped %.1f GiB, %.1f GiB RAM left); NOT falling back to mmap, expect swapping (use -lm mmap to choose mmap)\n", __func__,
            swapped / 1073741824.0, avail / 1073741824.0);
    }
    if (model == NULL) {
        return;
    }
    if (params.load_pinned_auto) {
        LOG_INF("%s: weights pinned (swapped %.1f GiB while loading, %.1f GiB RAM left)\n", __func__,
            swapped / 1073741824.0, avail / 1073741824.0);
    } else if (params.load_pin_budget > 0) {
        LOG_INF("%s: weights pinned as far as they reach: the model does not fit pinned RAM, up to %.1f GiB pinned (in tensor order), the rest mapped (mmap / page cache)\n", __func__,
            params.load_pin_budget / 1073741824.0);
    } else {
        LOG_INF("%s: weights NOT pinned: mmap / page cache (load mode %s)\n", __func__, llama_load_mode_name(params.load_mode));
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // these decision models return a score for each token via the embeddings output
    // TODO: maybe improve this in the future
    const auto decision_type = common_get_decision_type(model);
    // minicpm-ocr-jev reads its class scores from the pooled output, the pooling type is the one of the GGUF
    if (decision_type == COMMON_DECISION_TYPE_LAYA || decision_type == COMMON_DECISION_TYPE_KEV || decision_type == COMMON_DECISION_TYPE_CLEF ||
        decision_type == COMMON_DECISION_TYPE_LFM2_D1_OMNI || decision_type == COMMON_DECISION_TYPE_VLLM_DECISION2 ||
        decision_type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV) {
        const auto pooling_type = decision_type == COMMON_DECISION_TYPE_MINICPM_OCR_JEV ? LLAMA_POOLING_TYPE_UNSPECIFIED : LLAMA_POOLING_TYPE_NONE;

        params.embedding    = true;
        params.pooling_type = pooling_type;

        cparams.embeddings            = true;
        cparams.pooling_type          = pooling_type;
        cparams.n_outputs_max         = cparams.n_batch;
        cparams.n_outputs_max_per_seq = 1;

        LOG_INF("%s", "decision model reads the embeddings output, enabling embedding mode\n");
    }

    // embeddings need the whole batch in one ubatch, so n_batch must not be larger than n_ubatch
    // (server.cpp does this check for --embedding, but before the model is loaded)
    if (cparams.embeddings && cparams.n_batch > cparams.n_ubatch) {
        LOG_WRN("embeddings enabled: setting n_batch = n_ubatch = %u\n", cparams.n_ubatch);
        cparams.n_batch = cparams.n_ubatch;
        params.n_batch  = params.n_ubatch;
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_moe_set_options((params.moe_opts + (common_autotune_on(params) ? "" : (params.moe_opts.empty() ? "autotune=0" : ",autotune=0"))).c_str());
    llama_context * lctx = llama_init_from_model(model, cparams);
    // the compute buffers did not fit (a card near its limit): the largest ubatch that does, by bisection between the largest that fit and the smallest that
    // did not, in 32-token steps (user 2026-10-24: halving left GLM 3.0 at 8192 for good after one refused 16384, where the size between them fits). Every
    // try below one that fit is a new context: the one that fit is freed first, its buffers would crowd the next. Only an ubatch of ours, the user's -ub is
    // respected. (A GPU fault at run time, Xid 31, kills the CUDA context and cannot be retried.)
    if (lctx == NULL && (!params.ubatch_user || common_autotune_on(params)) && cparams.n_ubatch > 128) {   // a clamp by -ub may still need to go lower
        uint32_t lo = 0, bad = cparams.n_ubatch;   // the largest size that fit (0: none yet), the smallest that did not
        while (bad - lo > 32) {
            const uint32_t m = lo ? (lo + bad) / 2 / 32 * 32 : bad / 2 / 32 * 32;
            if (m < 128 || m <= lo) {
                break;
            }
            if (lctx) {
                llama_free(lctx);
                lctx = NULL;
            }
            cparams.n_ubatch = m;
            LOG_WRN("%s: context creation failed at ubatch %u, trying %u\n", __func__, bad, m);
            lctx = llama_init_from_model(model, cparams);
            if (lctx) { lo = m; } else { bad = m; }
        }
        if (!lctx && lo > 0) {   // the last try was too big: back to the largest that fit
            cparams.n_ubatch = lo;
            lctx = llama_init_from_model(model, cparams);
        }
    }
    // still no room: offloading host-weight ops to the GPU needs a staging buffer the size of a weight tensor on top of the compute buffers, which
    // a card near its limit does not have; without it the same graph needs less (and ran faster at low VRAM share in our tests)
    if (lctx == NULL && cparams.op_offload && !params.no_op_offload) {
        cparams.op_offload = false;
        LOG_WRN("%s: context creation failed, retrying without op offload\n", __func__);
        lctx = llama_init_from_model(model, cparams);
    }
    if (lctx && params.cpuparams.auto_threads && common_autotune_on(params)) {
        llama_set_thread_autotune(lctx, true); // the count came from the default, not from the user: tune it on measured decode time
        if (!params.threads_batch_set) {
            // likewise the batch count, on measured prompt batches, up to the logical cores (the batch pool is created that big)
            llama_set_batch_thread_autotune(lctx, true, (int32_t) std::thread::hardware_concurrency());
        }
    }
    if (lctx == NULL) {
        moe_auto_mark_failed();
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);

    set_process_priority(params.cpuparams.priority);

    pimpl->threadpools.init(lctx, params);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

// only for warmup and probe decodes, fill zeros as dummy input
static void common_batch_set_zero_state(common_batch & batch, const llama_model * model, std::vector<float> & zeros) {
    zeros.assign(llama_model_n_embd_out(model), 0.0f);
    for (int32_t i = 0; i < batch.size(); ++i) {
        batch.set_embd_state(i, { zeros.data(), 1, zeros.size() });
    }
}

void common_fork_off_apply(common_params & params) {
    if (params.fork_off) { // --fork off wins over every other fork setting, whatever the order of the options
        params.n_moe_cache_slots = 0;
        params.autotune          = false;
        params.auto_pin          = false;
        params.prefetch_experts_slots = 0;   // no staging slots (the default -1 is "auto: 3")
        params.n_moe_predict       = 0;     // no router prediction, no predictor training
        params.n_moe_predict_train = 0;
    }
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
#if defined(__linux__)
    // a user CPU mask (-C / -Cb): helper threads (loader, populate, probe, uploads) stay on those cores and their SMT siblings instead of the whole machine
    if (!params.fork_off && (params.cpuparams.mask_valid || params.cpuparams_batch.mask_valid) && !getenv("LLAMA_HELPER_CPUS")) {
        std::set<int> cpus;
        for (const auto * cp : { &params.cpuparams, &params.cpuparams_batch }) {
            if (!cp->mask_valid) { continue; }
            for (int c = 0; c < GGML_MAX_N_THREADS; ++c) {
                if (!cp->cpumask[c]) { continue; }
                cpus.insert(c);
                std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
                std::string l; std::getline(f, l);
                for (size_t pos = 0; pos < l.size(); ) {
                    const int a = atoi(l.c_str() + pos); int b = a;
                    const size_t d = l.find_first_of(",-", pos);
                    if (d != std::string::npos && l[d] == '-') { b = atoi(l.c_str() + d + 1); }
                    for (int i = a; i <= b; ++i) { cpus.insert(i); }
                    const size_t nx = l.find(',', pos);
                    if (nx == std::string::npos) { break; }
                    pos = nx + 1;
                }
            }
        }
        std::string list;
        for (int c : cpus) { list += (list.empty() ? "" : ",") + std::to_string(c); }
        if (!list.empty()) { setenv("LLAMA_HELPER_CPUS", list.c_str(), 1); LOG_INF("%s: helper threads stay on the user CPU mask and its SMT siblings: %s\n", __func__, list.c_str()); }
    }
#endif
    if (params.fork_off || !params.autotune) { // nothing is counted or saved
        llama_moe_set_opt("observe", "0");
    }
    common_fork_off_apply(params);
    const common_params orig = params; // before the placement and tuning below change it
    common_init_result_ptr res(new common_init_result(params, model_only));

    // last resort: the fork could not even load or create a context (out of memory) where upstream stock would run. Free everything and start over
    // with the fork off exactly as --moe cache=0 does (cache off, autotune off, the placement of upstream); not when the user forced a placement
    if ((res->model() == NULL || (!model_only && res->context() == NULL)) && !llama_moe_get_opt("mode") &&
        orig.n_moe_cache_slots != 0 && (orig.n_moe_cache_slots != -2 || common_autotune_on(orig))) {
        moe_auto_mark_failed();
        LOG_WRN("%s: initialisation failed, starting over with the fork off (as --moe cache=0)\n", __func__);
        res.reset();                      // release the model and the memory before loading again
        params = orig;
        params.n_moe_cache_slots = 0;
        params.autotune          = false;
        res.reset(new common_init_result(params, model_only));
    }

    // the stock placement of the fitter does not fit either (MiMo 132 GB on 2 x 24 GB: it plans a 27 GiB buffer for a 24 GiB card with the default 1 GiB fit target; 4.5 GiB starts):
    // once more with a larger fit target (+20% of each GPU's memory, at least 3 GiB: 4.5 and 8 GiB start MiMo, 4 GiB does not) before giving up
    if ((res->model() == NULL || (!model_only && res->context() == NULL)) && params.fit_params && !llama_moe_get_opt("mode")) {
        common_params retry = params;
        size_t g = 0;
        for (size_t i = 0; i < ggml_backend_dev_count() && g < retry.fit_params_target.size(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
            size_t mem_free = 0, mem_total = 0;
            ggml_backend_dev_memory(dev, &mem_free, &mem_total);
            retry.fit_params_target[g++] += std::max<size_t>((size_t) 3 << 30, mem_total / 100 * 20);
        }
        if (g > 0) {
            LOG_WRN("%s: the fitted placement does not fit, trying once more with a larger fit target (%zu MiB on the first GPU instead of %zu)\n", __func__,
                retry.fit_params_target[0] >> 20, params.fit_params_target[0] >> 20);
            res.reset();
            params = retry;
            res.reset(new common_init_result(params, model_only));
        }
    }

    llama_model * model = res->model();
    if (model == NULL) {
        moe_auto_mark_failed();
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        moe_auto_mark_failed();
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    // --moe kv-tail=1: the KV record has no memory until the expert pool exists (it lives at the pool's tail, no allocation of its own), and the pool starts at the first
    // real decode, so there is no warm-up decode before it
    const char * kv_tail_on = llama_moe_get_opt("kv-tail");
    if (params.warmup && kv_tail_on && atoi(kv_tail_on) != 0) {
        COM_TRC("%s", "kv-tail: no warm-up run (the KV cache has no memory before the expert pool starts)\n");
    } else
    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            common_batch batch = common_batch_get_one(lctx, tmp);
            llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get());
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_moe_cache_defer_fork(lctx, true); // start the cache on the first real decode
            tmp.resize(std::min(tmp.size(), (size_t) params.n_batch));
            common_batch batch = common_batch_get_one(lctx, tmp);
            std::vector<float> zeros;
            common_batch_set_zero_state(batch, model, zeros);
            llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            llama_synchronize(lctx);
            llama_memory_clear(llama_get_memory(lctx), true);
            llama_moe_cache_defer_fork(lctx, false);
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() {
    // ini-fix (2026-10-22): nothing of the run is recorded for the placement (static rule), and no stale-state guard erases what the cache learned (strikes):
    // the state file is never deleted from the engine
    if (true || g_moe_auto_mode.empty() || !pimpl || !pimpl->context) {
        return;
    }
    const llama_perf_context_data pd = llama_perf_context(pimpl->context.get());
    // the generation time decides what is recorded; the prompt time is only taken from real prompts (128+ tokens, below), so a short chat prompt
    // (GLM 3.5-bit short game prompt: 6 prompt tokens) still records its run: before, every stock start of such a request was thrown away and measured again
    if (pd.n_eval < 32) {
        LOG_INF("%s: MoE placement: too few tokens this run to record %s (%d prompt, %d generated)\n", __func__,
            g_moe_auto_mode.c_str(), pd.n_p_eval, pd.n_eval);
        return;
    }
    moe_auto_rec st, ca;
    std::string decided;
    moe_auto_read(g_moe_auto_file, st, ca, decided);
    moe_auto_rec & r = g_moe_auto_mode == "stock" ? st : ca;
    // the fastest per-token times seen: a cold run (model not in the page cache yet) never outweighs a warm one
    const double g_ms = pd.t_eval_ms / pd.n_eval;
    // stale state guard: what the cache learned (hot experts, tuner values) can go stale (another workload, state left by another build). A cache run
    // whose generation is more than 15% slower than the best this placement ever did, twice in a row, erases the model's learned state, so the next
    // starts relearn it. The stock record is kept; a single slow run (thermal, other load, a longer context) only counts a strike.
    if (g_moe_auto_mode == "cache" && r.have && r.n >= 2 && pd.n_eval >= 64) {
        std::string sec, pre; moe_auto_split(g_moe_auto_file, sec, pre);
        char b[16] = {0};
        int strikes = llama_state_get(sec.c_str(), (pre + ".strikes").c_str(), b, sizeof b) ? atoi(b) : 0;
        strikes = g_ms > 1.15 * r.g_ms ? strikes + 1 : 0;
        if (strikes >= 2) {
            LOG_INF("%s: MoE state looked stale (generation %.2f ms per token, best %.2f, twice in a row): erasing what was learned for this model, it relearns from the next start\n",
                __func__, g_ms, r.g_ms);
            llama_state_erase(sec.c_str(), "");
            moe_auto_write(g_moe_auto_file, st, moe_auto_rec(), "");
            llama_state_suppress_writes(true);   // marked for deletion: the engine's save when the context is freed (hot experts, tuner values, threads) writes nothing
            return;
        }
        llama_state_set(sec.c_str(), (pre + ".strikes").c_str(), std::to_string(strikes).c_str());
    }
    if (r.gr.empty() && r.have) { r.gr = { r.g_ms, r.g_max }; if (r.g_max == r.g_ms) { r.gr.pop_back(); } }   // a record from before the sample lists: its best and worst seed them
    if (r.pr.empty() && r.have && r.p_ms >= 0) { r.pr = { r.p_ms, r.p_max }; if (r.p_max == r.p_ms) { r.pr.pop_back(); } }
    if (pd.n_p_eval >= 128) { r.pr.push_back(pd.t_p_eval_ms / pd.n_p_eval); }   // a real prompt: per-token prompt cost (a few tokens would only time the fixed setup)
    r.gr.push_back(g_ms);
    if (r.pr.size() > 9) { r.pr.erase(r.pr.begin(), r.pr.end() - 9); }
    if (r.gr.size() > 9) { r.gr.erase(r.gr.begin(), r.gr.end() - 9); }
    r.have = true;
    moe_auto_refresh(r);
    r.n++;
    if (st.have && ca.have) {
        moe_auto_store_picks(g_moe_auto_file, st, ca);
        decided = moe_auto_choose(st, ca, moe_auto_p_known(st, ca) ? g_moe_auto_np : 0.0, g_moe_auto_ng, decided);
        LOG_INF("%s: MoE placement decided: %s (stock %.2f/%.2f, cache %.2f/%.2f ms per prompt/generated token)\n", __func__,
            decided.c_str(), st.p_ms, st.g_ms, ca.p_ms, ca.g_ms);
    }
    moe_auto_write(g_moe_auto_file, st, ca, decided);
}

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    // fork: a context with rollback slots runs the probe decode too and stays RS: the expert pool and the KV records at its tail start at the first decode,
    // which belongs at load. DeepSeek 4.1 with a DSpark draft (rollback slots) skipped it: the pool start fell into the first request's prompt time, and that
    // request's sequence removal cleared KV records that had no memory yet ('tensor not allocated')
    const bool rs = llama_n_rs_seq(ctx) > 0;
    if (rs) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
    }

    common_context_seq_rm_type res = rs ? COMMON_CONTEXT_SEQ_RM_TYPE_RS : COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret;
    {
        common_batch batch = common_batch_get_one(ctx, tmp);
        std::vector<float> zeros;
        common_batch_set_zero_state(batch, llama_get_model(ctx), zeros);
        ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }
    if (ret != 0) {
        COM_ERR("llama_process() failed: %d\n", ret);
        if (!rs) {
            res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        }
        goto done;
    }

    // try to remove the last tokens
    if (!rs && !llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.load_mode       = params.load_mode;
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;
    mparams.load_mtp                    = params.load_mtp || std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_moe_cache_slots   = params.n_moe_cache_slots == -2 ? 0 : params.n_moe_cache_slots;
    cparams.n_moe_cache_inserts = params.n_moe_cache_inserts;
    cparams.n_moe_cache_window  = params.n_moe_cache_window;
    cparams.n_moe_predict       = params.n_moe_predict;
    cparams.n_moe_predict_train = params.n_moe_predict_train;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.prefetch_experts_slots = params.prefetch_experts_slots;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    cparams.moe_cache_size = params.moe_cache_size;

    return cparams;
}

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

common_threadpools::~common_threadpools() {
    if (!free_fn) {
        return;
    }
    free_fn(threadpool);
    free_fn(threadpool_batch);
}

void common_threadpools::init(llama_context * ctx, const common_params & params) {
    GGML_ASSERT(!threadpool);
    GGML_ASSERT(!threadpool_batch);

    COM_INF("llama threadpool init, n_threads = %d\n", (int) params.cpuparams.n_threads);

    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        COM_WRN("%s", "no CPU backend found\n");
        return;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    if (params.cpuparams.auto_threads && !params.threads_batch_set) {
        // the batch tuner may use up to the logical cores: size the batch pool for it, so it differs from the decode pool
        // (graph_compute uses n_threads_batch of them, the rest idle)
        tpp_batch.n_threads = std::max(tpp_batch.n_threads, (int) std::thread::hardware_concurrency());
    }
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    // each pool needs to match the respective n_threads exactly
    // see: https://github.com/ggml-org/llama.cpp/pull/27138#issuecomment-5332307332
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            COM_WRN("batch threadpool create failed : n_threads %d\n", tpp_batch.n_threads);
            return;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        COM_WRN("threadpool create failed : n_threads %d\n", tpp.n_threads);
        free_fn(threadpool_batch);
        threadpool_batch = nullptr;
        return;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    common_batch batch(ctx);
    batch.add(last_token, pos, 0, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

common_batch::common_batch(llama_context * ctx) : batch(llama_batch_ext_init(ctx)) {
    const auto rope_type = llama_model_rope_type(llama_get_model(ctx));
    n_pos = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
    tokens.clear();
}

int32_t common_batch::add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output) {
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, { nullptr, 0, 0 }, {} });
    return size() - 1;
}

int32_t common_batch::add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output) {
    GGML_ASSERT(!seq_ids.empty());

    const int32_t idx = add(id, pos, seq_ids[0], output);
    for (size_t s = 1; s < seq_ids.size(); ++s) {
        add_seq(idx, seq_ids[s]);
    }
    return idx;
}

bool common_batch::add_seq(int32_t idx, llama_seq_id seq_id) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].seq_ids_extra.push_back(seq_id);
    return true;
}

bool common_batch::set_output(int32_t idx, bool value) {
    if (idx < 0 || idx >= size()) {
        return false;
    }
    tokens[idx].output = value;
    return true;
}

bool common_batch::set_embd(int32_t idx, llama_embd embd) {
    if (idx < 0 || idx >= size() || tokens[idx].embd.data != nullptr) {
        return false;
    }
    tokens[idx].embd = embd;
    return true;
}

bool common_batch::set_embd_state(int32_t idx, llama_embd state) {
    if (idx < 0 || idx >= size() || tokens[idx].state.data != nullptr) {
        return false;
    }
    tokens[idx].state = state;
    return true;
}

int32_t common_batch::add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output) {
    token t = { LLAMA_TOKEN_NULL, { 0, 0, 0, 0 }, seq_id, output, embd, { nullptr, 0, 0 }, {} };
    for (int32_t j = 0; j < n_pos; ++j) {
        t.pos[j] = pos[j];
    }
    tokens.push_back(t);
    return size() - 1;
}

llama_batch_ext * common_batch::get_sub_batch(int32_t off, int32_t n) {
    GGML_ASSERT(batch && "common_batch was not initialized with a context");
    GGML_ASSERT(off >= 0 && n >= 0 && off + n <= size());

    llama_batch_ext * res = batch.get();
    llama_batch_ext_clear(res);

    for (int32_t i = off; i < off + n; ++i) {
        const token & t = tokens[i];

        int32_t idx;
        if (t.id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(res, t.seq_id, t.id);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, t.id, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
            if (t.embd.data && !llama_batch_ext_set_embd_token(res, idx, t.embd)) {
                GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, t.id, i);
            }
        } else {
            idx = llama_batch_ext_add_embd(res, t.seq_id, t.embd);
            if (idx < 0) {
                GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, i, idx, n);
            }
            llama_batch_ext_set_pos(res, idx, t.pos.data());
        }
        GGML_ASSERT(idx == i - off);

        for (const llama_seq_id seq_id : t.seq_ids_extra) {
            if (!llama_batch_ext_add_seq(res, idx, seq_id)) {
                GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, seq_id, i);
            }
        }
        if (t.output) {
            llama_batch_ext_set_output_logits(res, idx, true);
        }
        if (t.state.data) {
            llama_batch_ext_set_embd_state(res, idx, t.state); // contexts without a state input ignore it
        }
        if (t.decision_order != 0) {
            llama_batch_ext_set_decision_order(res, idx, (llama_decision_order) t.decision_order);
        }
    }

    return res;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_token * tokens, int32_t n_tokens) {
    common_batch batch(ctx);

    auto mem = llama_get_memory(ctx);
    llama_pos pos = llama_memory_seq_pos_max(mem, 0) + 1; // -1 + 1 == 0 when the memory is empty

    for (int32_t i = 0; i < n_tokens; ++i) {
        const bool output = i == n_tokens - 1;
        batch.add(tokens[i], pos, 0, output);
        pos++;
    }

    return batch;
}

common_batch common_batch_get_one(llama_context * ctx, const llama_tokens & tokens) {
    return common_batch_get_one(ctx, tokens.data(), (int32_t) tokens.size());
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        llama_tokens prefix_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_tokens_before_last);
        common_batch batch_prefix = common_batch_get_one(ctx, prefix_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_prefix.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        common_batch batch_last(ctx);
        batch_last.add(all_tokens.back(), n_past, 0, true);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_last.get())) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        llama_tokens new_tokens(all_tokens.begin() + offset, all_tokens.begin() + offset + n_new);
        common_batch batch = common_batch_get_one(ctx, new_tokens);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

bool common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_tgt.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    return n == data_tgt.size();
}

bool common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return true;
    }

    if (data_dft.empty()) {
        return true;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    return n == data_dft.size();
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
