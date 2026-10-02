#include "ggml.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
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

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
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
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
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
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
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
    std::error_code ec;
    std::filesystem::path path_tmp = path;
    path_tmp += ".tmp";

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
    std::filesystem::create_directories(cache_directory, ec);
    if (ec) {
        throw std::runtime_error("failed to create cache directory: " + fs_path_to_utf8(cache_directory));
    }
    return cache_directory / std::filesystem::u8path(filename);
}

std::vector<common_file_info> fs_list(const std::string & path, bool include_directories) {
    std::vector<common_file_info> files;
    if (path.empty()) return files;

    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        try {
            // Only include regular files (skip directories)
            const auto & p = entry.path();
            if (std::filesystem::is_regular_file(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.is_dir = false;
                try {
                    info.size = static_cast<size_t>(std::filesystem::file_size(p));
                } catch (const std::filesystem::filesystem_error &) {
                    info.size = 0;
                }
                files.push_back(std::move(info));
            } else if (include_directories && std::filesystem::is_directory(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.size   = 0; // Directories have no size
                info.is_dir = true;
                files.push_back(std::move(info));
            }
        } catch (const std::filesystem::filesystem_error &) {
            // skip entries we cannot inspect
            continue;
        }
    }

    return files;
}

std::ifstream fs_open_ifstream(const std::string & fname, std::ios_base::openmode mode) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) { return std::ifstream(); }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), mode);
#else
    return std::ifstream(fname, mode);
#endif
}

//
// TTY utils
//

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
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
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
static size_t common_model_file_size(const std::string & path_model) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gguf = gguf_init_from_file(path_model.c_str(), gp);
    if (!gguf) {
        return 0;
    }
    const int64_t id = gguf_find_key(gguf, "split.count");
    const uint32_t n_split = id < 0 ? 1 : std::max<uint32_t>(1, gguf_get_val_u16(gguf, id));
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
static size_t common_model_expert_bytes(const std::string & path_model) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g0 = gguf_init_from_file(path_model.c_str(), gp);
    if (!g0) {
        return 0;
    }
    const int64_t id = gguf_find_key(g0, "split.count");
    const uint32_t n_split = id < 0 ? 1 : std::max<uint32_t>(1, gguf_get_val_u16(g0, id));
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
            if (strstr(gguf_get_tensor_name(g, t), "_exps")) {
                bytes += gguf_get_tensor_size(g, t);
            }
        }
        gguf_free(g);
    }
    return bytes;
}

// free memory summed over GPUs, of the largest GPU, and the GPU count
static void common_gpu_free(size_t & total, size_t & max, int & n) {
    total = max = 0;
    n = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free, tot;
        ggml_backend_dev_memory(dev, &free, &tot);
        total += free;
        max    = std::max(max, free);
        n++;
    }
}

// RAM that can be used without swapping: Linux MemAvailable (free + reclaimable page cache), 0 if unknown
static size_t common_ram_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#elif defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemAvailable:") {
            return kb * 1024;
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
struct moe_auto_rec { bool have = false; double p_ms = -1, g_ms = 0; int n = 0; int failed_mib = 0; }; // n: recorded runs; p_ms < 0: no prompt of real size measured yet; failed_mib: free VRAM (MiB) when it could not start (out of memory), 0: never
static std::string g_moe_auto_file;  // decision file of this run (empty: auto choice not involved)
static std::string g_moe_auto_mode;  // mode this run explores: "stock" or "cache" (empty: decided already)
static size_t      g_moe_auto_vram_free = 0; // free VRAM when this run chose its placement

// The placement records live in the engine's state file (llama_state_*, one INI section per model):
//   [<model file> <bytes>]  place.g<gpus>x<MiB>.b<build>.stock|cache = <ms/prompt token> <ms/generated token> <runs>  and  .decided
// moe_auto_path() returns "<section>\x1f<key prefix>"; the same section is handed to the engine for its own keys (hot experts, tuner)
static std::string moe_auto_path(const common_params & params, size_t model_size, int n_gpu, size_t vram_max) {
    std::string name = params.model.path;
    const size_t sl = name.find_last_of("/\\");
    if (sl != std::string::npos) { name = name.substr(sl + 1); }
    for (auto & c : name) { if (c == '[' || c == ']' || c == '\n' || c == '\r') { c = '_'; } }
    const std::string section = name + " " + std::to_string(model_size);
    llama_state_set_model(section.c_str());
    return section + "\x1f" + "place.g" + std::to_string(n_gpu) + "x" + std::to_string(vram_max >> 20) + ".v1"; // v1: the placement rule's epoch, not the build, so a new release does not re-measure (LLAMA_MOE_AUTO_MODE=retest forgets)
}

static void moe_auto_split(const std::string & path, std::string & section, std::string & prefix) {
    const size_t p = path.find('\x1f');
    section = path.substr(0, p);
    prefix  = p == std::string::npos ? "" : path.substr(p + 1);
}

static void moe_auto_read(const std::string & path, moe_auto_rec & st, moe_auto_rec & ca, std::string & decided) {
    std::string section, prefix;
    moe_auto_split(path, section, prefix);
    char buf[128];
    for (moe_auto_rec * r : { &st, &ca }) {
        if (llama_state_get(section.c_str(), (prefix + (r == &st ? ".stock" : ".cache")).c_str(), buf, sizeof buf)) {
            r->have = sscanf(buf, "%lf %lf %d %d", &r->p_ms, &r->g_ms, &r->n, &r->failed_mib) >= 3;
        }
    }
    if (llama_state_get(section.c_str(), (prefix + ".decided").c_str(), buf, sizeof buf)) { decided = buf; }
}

static void moe_auto_write(const std::string & path, const moe_auto_rec & st, const moe_auto_rec & ca, const std::string & decided) {
    std::string section, prefix;
    moe_auto_split(path, section, prefix);
    auto put = [&](const char * k, const moe_auto_rec & r) {
        if (r.have || r.failed_mib) { llama_state_set(section.c_str(), (prefix + k).c_str(), string_format("%.4f %.4f %d %d", r.p_ms, r.g_ms, r.n, r.failed_mib).c_str()); } // p_ms -1: unknown
    };
    put(".stock", st);
    put(".cache", ca);
    if (!decided.empty()) { llama_state_set(section.c_str(), (prefix + ".decided").c_str(), decided.c_str()); }
}

// the process could not start with the placement it was measuring (out of memory): record it with the free VRAM it saw, the other placement is used from the next start
// on (until that much more VRAM is free than at the failure, e.g. another program that held it has gone)
static void moe_auto_mark_failed() {
    if (g_moe_auto_mode.empty()) {
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

// typical agentic coding turn when the command line says nothing (server, chat): the system prompt and tool definitions come from
// the KV cache, so a turn adds ~2000 new prompt tokens (tool output, file contents) and generates ~600 (edits, tool calls)
static double moe_auto_est_prompt(const common_params & params) {
    return params.prompt.empty() ? 2000.0 : std::max(1.0, params.prompt.size() / 4.0);
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

static std::string moe_auto_decide_for(const moe_auto_rec & st, const moe_auto_rec & ca, const common_params & params) {
    const double np = moe_auto_p_known(st, ca) ? moe_auto_est_prompt(params) : 0.0;
    const double ng = moe_auto_est_gen(params);
    const bool cache = np * ca.p_ms + ng * ca.g_ms <= np * st.p_ms + ng * st.g_ms;
    // switchover: the prompt length below which the cache wins for this many generated tokens
    const double dp = ca.p_ms - st.p_ms, dg = st.g_ms - ca.g_ms;
    const double breakeven = !moe_auto_p_known(st, ca) || dp <= 0 ? INFINITY : dg <= 0 ? 0.0 : ng * dg / dp;
    LOG_INF("%s: MoE placement: request ~%.0f prompt + %.0f generated tokens; the cache wins for prompts under ~%.0f tokens at that answer length\n",
        __func__, np, ng, breakeven);
    return cache ? "cache" : "stock";
}

// the cache wins the estimated request (same estimate as above) by at least 3% on one run of each placement
static bool moe_auto_cache_clear_win(const moe_auto_rec & st, const moe_auto_rec & ca, const common_params & params) {
    const double np = moe_auto_p_known(st, ca) ? moe_auto_est_prompt(params) : 0.0, ng = moe_auto_est_gen(params);
    return np * ca.p_ms + ng * ca.g_ms <= 0.97 * (np * st.p_ms + ng * st.g_ms);
}

static std::string moe_auto_decide(const moe_auto_rec & st, const moe_auto_rec & ca) {
    const char * e = getenv("LLAMA_MOE_AUTO_PREFILL_SLOWDOWN");
    const double max_p = e ? atof(e) : 2.0;
    return ca.g_ms <= st.g_ms && (!moe_auto_p_known(st, ca) || ca.p_ms < max_p * st.p_ms) ? "cache" : "stock";
}

// --moe autotune=0 or LLAMA_AUTOTUNE=0: the kill switch for everything the engine tunes or measures by itself
bool common_autotune_on(const common_params & params) {
    const char * e = getenv("LLAMA_AUTOTUNE");
    return params.autotune && !(e && atoi(e) == 0);
}

// -ub from the request and the card, no VRAM tiers: the compute buffer of a ubatch is probed (no_alloc, per candidate) and what it
// takes from the expert cache is weighed against the prefill time it saves.
//   time per request, in decode-token units:  np / (PP_PER_TG * (ub/512)^PP_EXP)  +  ng * (1 + extra / cache_vram)
// extra = compute buffer of this ubatch minus that of 512, cache_vram = VRAM left for cache slots; the decode loss is taken as
// the share of cache slots lost. Constants measured: prefill grows with ub^0.6 (2.3x for 4x on a 16 GB card with experts in RAM,
// 1.8x on 2x RTX 3090 and on OLMoE) and is ~11x the decode rate; np / ng are the request profile (-p / -n, else the agentic estimate).
static uint32_t common_auto_ubatch(common_params & params) {
    const double PP_EXP = 0.5, PP_PER_TG = 11.0;
    const uint32_t cand[] = { 512, 1024, 2048 };
    std::vector<size_t> compute(3, 0);
    int64_t cache_vram = 0;
    for (int i : { 0, 2 }) {  // the buffer is ~linear in ub: probe both ends, interpolate the middle
        auto mp = common_model_params_to_llama(params);
        auto cp = common_context_params_to_llama(params);
        cp.n_ubatch = cand[i];
        cp.n_batch  = std::max(cp.n_batch, cand[i]);
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
            COM_DBG("ubatch probe failed (%s), keeping %u\n", e.what(), cand[0]);
            return cand[0];
        }
        compute[i] = sum;
    }
    if (cache_vram <= 0 || compute[2] <= compute[0]) {
        return cand[0];
    }
    const double np = moe_auto_est_prompt(params), ng = moe_auto_est_gen(params);
    uint32_t best = cand[0];
    double t_best = 1e300;
    for (uint32_t ub : cand) {
        const double extra = (double) (compute[2] - compute[0]) * (ub - 512) / 1536.0;
        const double t = np / (PP_PER_TG * std::pow(ub / 512.0, PP_EXP)) + ng * (1.0 + extra / (double) cache_vram);
        COM_DBG("ubatch %u: extra buffer %.2f GiB of %.1f GiB cache VRAM, est. %.0f decode-token units\n", ub, extra / 1073741824.0, cache_vram / 1073741824.0, t);
        if (t < t_best) { t_best = t; best = ub; }
    }
    LOG_INF("%s: ubatch %u (request ~%.0f prompt + %.0f generated tokens, compute buffer %.2f -> %.2f GiB at 512 -> 2048, %.1f GiB left for the cache)\n", __func__,
        best, np, ng, compute[0] / 1073741824.0, compute[2] / 1073741824.0, cache_vram / 1073741824.0);
    return best;
}

static void common_moe_cache_auto_impl(common_params & params);

static void common_moe_cache_auto(common_params & params) {
    common_moe_cache_auto_impl(params);
    // VRAM the cache leaves free for the growth of the CUDA pools after load, which scales with the batch: measured at -ub 2048 (cache
    // sized with a margin, 12k prompt, other margins aborting with CUDA out of memory): 54 MiB OLMoE, 74 IQ3_S, 170 GLM; ~15 at 512.
    // 0.1 MiB per ubatch token is 1.2x the worst of them, floor 200: GLM aborted with out of memory at ub 512 with 51 (-at off keeps the 384 MiB default; --moe margin_mb=N overrides)
    if (params.n_moe_cache_slots != 0 && common_autotune_on(params) && params.moe_opts.find("margin_mb") == std::string::npos &&
        !getenv("LLAMA_MOE_CACHE_MARGIN_MB")) {
        params.moe_opts += (params.moe_opts.empty() ? "" : ",") + std::string("margin_mb=") + std::to_string(std::max(200, (int) std::ceil(0.1 * params.n_ubatch)));
    }
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
    if (n_gpu == 0 || model_size + reserve <= vram_free) {
        COM_DBG("MoE model (%.1f GiB) fits free VRAM (%.1f GiB on %d GPUs), no expert cache\n",
            model_size / 1073741824.0, vram_free / 1073741824.0, n_gpu);
        return;
    }

    bool use_cache = true;
    {
        const std::string path = moe_auto_path(params, model_size, n_gpu, vram_max);
        const char * force = getenv("LLAMA_MOE_AUTO_MODE");
        const std::string f = force ? force : "";
        if (f == "retest") { std::string sec, pre; moe_auto_split(path, sec, pre); llama_state_erase(sec.c_str(), pre.c_str()); }
        moe_auto_rec st, ca;
        std::string decided;
        moe_auto_read(path, st, ca, decided);
        std::string mode;
        g_moe_auto_vram_free = vram_free;
        // a placement that could not start counts as failed until a good deal more VRAM is free than when it failed
        const bool st_failed = st.failed_mib > 0 && (double) vram_free < 1.25 * ((double) st.failed_mib * 1048576.0);
        const bool ca_failed = ca.failed_mib > 0 && (double) vram_free < 1.25 * ((double) ca.failed_mib * 1048576.0);
        // can the cache fit at all? Its slots come after the non-expert weights, the KV cache and the compute buffers; when those alone do not fit in the free VRAM
        // there is no cache to measure (GLM 3.0-bit on one 3090 with 10.9 GiB free: out of memory)
        const size_t expert_bytes = common_model_expert_bytes(params.model.path);
        const size_t cache_need   = expert_bytes && model_size > expert_bytes ? model_size - expert_bytes + ((size_t) 3 << 29) + ((size_t) n_gpu << 28) : 0; // + 1.5 GiB + 0.25 GiB per GPU
        if (f == "stock" || f == "cache") {
            mode = f;
        } else if (!common_autotune_on(params)) {
            // no measuring: the static rule, cache whenever the model does not fit (and the prompt is not long); nothing is recorded
            mode = moe_auto_est_prompt(params) < 3000 ? "cache" : "stock";
            LOG_INF("%s: MoE placement: %s (autotune off: static rule, model %.1f GiB, free VRAM %.1f GiB)\n", __func__, mode.c_str(),
                model_size / 1073741824.0, vram_free / 1073741824.0);
        } else if (st_failed != ca_failed) {
            mode = st_failed ? "cache" : "stock";
            LOG_INF("%s: MoE placement: %s (the other placement could not start with this much free VRAM before; %s)\n", __func__, mode.c_str(), path.c_str());
        } else if (cache_need > vram_free) {
            mode = "stock";
            LOG_INF("%s: MoE placement: stock (the cache cannot fit: about %.1f GiB for the non-expert weights, KV cache and compute buffers > free VRAM %.1f GiB)\n", __func__,
                cache_need / 1073741824.0, vram_free / 1073741824.0);
        } else if ((st.n >= 2 && ca.n >= 2) || (st.n >= 1 && ca.n >= 1 && moe_auto_cache_clear_win(st, ca, params))) {
            // both placements measured on at least two runs (the first of each may be cold), or once each when the cache, whose run came
            // first and may be cold, already wins this request by 10%: decide for this request
            mode = moe_auto_decide_for(st, ca, params);
            // keep recording this placement: a cold first run (page cache, mmap) is replaced by any faster later run
            g_moe_auto_file = path;
            g_moe_auto_mode = mode;
            LOG_INF("%s: MoE placement: %s (measured: stock %.2f/%.2f, cache %.2f/%.2f ms per prompt/generated token; %s)\n", __func__,
                mode.c_str(), st.p_ms, st.g_ms, ca.p_ms, ca.g_ms, path.c_str());
        } else {
            // first run, nothing measured yet (PC1 IQ3_S 83 GB on one 24 GB GPU: stock 23.6 vs cache 42-49 t/s): the cache when the model is clearly bigger than the free VRAM (PC1 IQ1_M 54 GB on 2 x 24 GB: stock 108.9/60.4 vs cache 80.4/54.4 prompt/gen t/s); a prompt of thousands of tokens starts with stock, where the cache's slow prompt
            // processing costs more than its faster generation gains; the measured runs then keep stock only where it is faster)
            const size_t est_prompt = (size_t) moe_auto_est_prompt(params);
            const size_t est_gen = (size_t) moe_auto_est_gen(params);
            const bool prefill_heavy = est_prompt >= 3000 || est_prompt > 8 * est_gen;
            const std::string first = model_size * 10 > vram_free * 13 && !prefill_heavy ? "cache" : "stock";
            // the cache gains most when a fifth to a half of the model fits (2 x RTX 3090: GLM 1.5x-2.2x, MiMo 2.4x, Qwen3.8 1.7x; 4 x RTX 3090 with 22% in VRAM: GLM 1.6x) and can
            // lose on both sides of that band, depending on the machine. Above half (simulated 2 x 16 GB cards, Qwen3.8 IQ1_M 55 GB: 0.9x at 58% in VRAM, 0.78x at 83%, with a 99%
            // hit rate; 4 x RTX 3090, GLM at 88%: 1.3x). Below a fifth (user report, RTX 2080 Ti 11 GB, MiMo IQ2_M 100 GB, ~10% in VRAM: 6.6 vs 15.5 t/s of the static split, hit
            // 21%: a thin cache over every layer loses to whole resident layers; at 15% on an RX 9070 XT the two were equal). Outside the band the two placements are measured,
            // like a request that is mostly prompt.
            const bool uncertain = vram_free * 2 >= model_size || vram_free * 5 < model_size;
            if (!prefill_heavy && !uncertain) {
                // between a fifth and a half of the model fits: the cache, no stock run to spend: nothing is recorded and every start takes the cache.
                mode = "cache";
                LOG_INF("%s: MoE placement: cache (model %.1f GiB does not fit the free VRAM %.1f GiB, nothing to measure)\n", __func__,
                    model_size / 1073741824.0, vram_free / 1073741824.0);
            } else {
                // measuring: the placement with fewer recorded runs (ties: the first), so each gets two runs before it decides
                mode = st.n == ca.n ? first : (st.n < ca.n ? "stock" : "cache");
                g_moe_auto_file = path;
                g_moe_auto_mode = mode;
                LOG_INF("%s: MoE placement: measuring %s this run (stock %d runs, cache %d runs recorded; both need 2 before the faster one is kept; %s)\n", __func__,
                    mode.c_str(), st.n, ca.n, path.c_str());
            }
        }
        use_cache = mode != "stock";
    }

    // stock placement (the model fitter places layers): only the expert cache and what exists for it stays off (experts on the
    // CPU, repack disabled, cache slots, the big prompt batch that its uploads want); pinned weights and the thread choice stay
    if (use_cache) {
        auto & tbo = params.tensor_buft_overrides;
        tbo.insert(std::find_if(tbo.begin(), tbo.end(), [](const auto & o) { return o.pattern == nullptr; }), llm_ffn_exps_cpu_override());
        params.no_extra_bufts    = true;
        params.n_moe_cache_slots = -1;
    }
    // pinned weights (no mmap, llama-server only: startup takes longer, pays off over many requests):
    // cache uploads and prompt processing read them by direct DMA,
    // measured +46% prompt processing, +3-5% decode on GLM-5.3-Flash; used whenever the model fits
    // in RAM available now (no margin); common_init_result reloads with mmap if loading swaps
    if (params.auto_pin && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        const size_t avail = common_ram_available();
        if (avail && model_size <= avail) {
            params.load_mode        = LLAMA_LOAD_MODE_NONE;
            params.load_pinned_auto = true;
        }
    }
    if (!params.ubatch_user && common_autotune_on(params)) {
        params.n_ubatch = std::min<uint32_t>(params.n_batch, common_auto_ubatch(params));
    }
    // leave one core per GPU to drive it (measured on 2 GPUs: 6 of 8 cores beats 8)
    for (auto * cp : { &params.cpuparams, &params.cpuparams_batch }) {
        if (cp->auto_threads) {
            cp->n_threads = std::max(2, cp->n_threads - n_gpu);
        }
    }
    // resource saturator: compute threads on physical cores only (SMT siblings add no memory bandwidth), filled L3 domain
    // by domain so each domain's threads compute its rows (GGML_MOE_CCX_SPLIT), leaving each domain's highest core to its
    // L3 prefetch thread. Linux sysfs / Windows GetLogicalProcessorInformationEx; only without a user CPU mask (LLAMA_AUTO_PLACE=1).
    // ponytail: no NUMA-node awareness yet, Windows processor group 0 only
    if (!params.cpuparams.mask_valid && params.cpuparams.auto_threads && getenv("LLAMA_AUTO_PLACE") && atoi(getenv("LLAMA_AUTO_PLACE")) != 0) {
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
                const char * l3pf = getenv("LLAMA_MOE_CACHE_L3PF");
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
    if (params.n_ctx == 0) {
        params.n_ctx = 32768; // ponytail: autofit would grow KV to n_ctx_train and starve the cache; -c N for more
    }
    LOG_INF("%s: MoE model (%.1f GiB) exceeds free VRAM (%.1f GiB): experts in RAM + GPU expert cache, no repack, "
        "ctx %d, ubatch %d, threads %d, weights %s (settings you pass win)\n", __func__, model_size / 1073741824.0, vram_free / 1073741824.0,
        params.n_ctx, params.n_ubatch, params.cpuparams.n_threads, params.load_pinned_auto ? "pinned" : "mmap");
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    common_moe_cache_auto(params);
    common_spec_auto(params);
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
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

        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            has_draft || spec_mtp ? &extra : nullptr,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    const size_t swap0 = params.load_pinned_auto ? common_swap_out_bytes() : 0;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    // real memory pressure, not the kernel moving idle pages of other programs to swap (zram)
    const size_t swapped = params.load_pinned_auto ? common_swap_out_bytes() - swap0 : 0;
    const size_t avail   = params.load_pinned_auto ? common_ram_available() : 0;
    if (params.load_pinned_auto && (model == NULL || swapped > (2ull << 30) || (avail && avail < (512ull << 20)))) {
        LOG_WRN("%s: pinned weights %s (swapped %.1f GiB, %.1f GiB RAM left), reloading with mmap\n", __func__,
            model ? "left too little RAM" : "failed to load", swapped / 1073741824.0, avail / 1073741824.0);
        llama_model_free(model);
        params.load_mode        = LLAMA_LOAD_MODE_AUTO;
        params.load_pinned_auto = false;
        mparams = common_model_params_to_llama(params);
        model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    }
    if (model == NULL) {
        return;
    }
    if (params.load_pinned_auto) {
        LOG_INF("%s: weights pinned (swapped %.1f GiB while loading, %.1f GiB RAM left)\n", __func__,
            swapped / 1073741824.0, avail / 1073741824.0);
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

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
    // the compute buffers did not fit (a card near its limit): a smaller ubatch needs a smaller buffer, retry halving it; only an
    // ubatch of ours, the user's -ub is respected. (A GPU fault at run time, Xid 31, kills the CUDA context and cannot be retried.)
    while (lctx == NULL && !params.ubatch_user && cparams.n_ubatch > 128) {
        cparams.n_ubatch /= 2;
        LOG_WRN("%s: context creation failed, retrying with ubatch %u\n", __func__, cparams.n_ubatch);
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

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_init_result_ptr res(new common_init_result(params, model_only));

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
            llama_moe_cache_defer(lctx, true); // start the cache on the first real decode
            tmp.resize(std::min(tmp.size(), (size_t) params.n_batch));
            common_batch batch = common_batch_get_one(lctx, tmp);
            llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            llama_synchronize(lctx);
            llama_memory_clear(llama_get_memory(lctx), true);
            llama_moe_cache_defer(lctx, false);
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
    if (g_moe_auto_mode.empty() || !pimpl || !pimpl->context) {
        return;
    }
    const llama_perf_context_data pd = llama_perf_context(pimpl->context.get());
    if (pd.n_p_eval < 8 || pd.n_eval < 32) {
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
    if (pd.n_p_eval >= 128) { // a real prompt: per-token prompt cost (a few tokens would only time the fixed setup)
        const double p_ms = pd.t_p_eval_ms / pd.n_p_eval;
        r.p_ms = r.p_ms >= 0 ? std::min(r.p_ms, p_ms) : p_ms;
    }
    r.g_ms = r.have ? std::min(r.g_ms, g_ms) : g_ms;
    r.have = true;
    r.n++;
    if (st.have && ca.have) {
        decided = moe_auto_decide(st, ca);
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

    if (llama_n_rs_seq(ctx) > 0) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        return COMMON_CONTEXT_SEQ_RM_TYPE_RS;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret;
    {
        common_batch batch = common_batch_get_one(ctx, tmp);
        ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }
    if (ret != 0) {
        COM_ERR("llama_process() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
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
    mparams.load_mtp                    = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

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
    tokens.push_back({ id, { pos, 0, 0, 0 }, seq_id, output, { nullptr, 0, 0 }, {} });
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

int32_t common_batch::add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output) {
    token t = { LLAMA_TOKEN_NULL, { 0, 0, 0, 0 }, seq_id, output, embd, {} };
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

void common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_tgt.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    if (n != data_tgt.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_tgt.size(), n);
    }
}

void common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_dft.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    if (n != data_dft.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_dft.size(), n);
    }
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
