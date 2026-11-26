// fork: the dev event log (ggml-evlog-fork.h)
#include "ggml-evlog-fork.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

volatile uint32_t ggml_evlog_mask = 0;

namespace {

struct ev_rec {
    uint64_t t0, t1;
    const char * name;
    int64_t a0, a1;
    uint32_t tid;
    uint16_t cat;
    uint16_t ok;   // 0 while being written: the flusher skips (and counts) a half-written record
};

static const char * const k_cat_names[GGML_EV_N_CAT] = {
    "sched", "graph", "copy", "sync", "gpu", "upload", "rampool", "loader", "planner", "pool", "decode", "server", "sys"
};

struct ev_log {
    std::vector<ev_rec>   ring;
    uint64_t              n = 0;          // ring size, a power of two
    std::atomic<uint64_t> head{0};        // records claimed so far
    uint64_t              written = 0;    // records the file has (flusher only)
    uint64_t              lost = 0;
    FILE *                fp = nullptr;
    std::mutex            flush_mu;
    std::thread           flusher;
    std::atomic<bool>     stop{false};
};

static ev_log * g_log = nullptr;

static uint32_t ev_tid() {
#if defined(__linux__)
    static thread_local uint32_t t = (uint32_t) syscall(SYS_gettid);
    return t;
#else
    static thread_local uint32_t t = (uint32_t) std::hash<std::thread::id>{}(std::this_thread::get_id());
    return t;
#endif
}

static uint64_t real_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

#if defined(__linux__)
// system state, sampled by the flusher thread: MemAvailable, this process's RSS, major faults and CPU time (user+system), as instant 'sys' records
static void ev_sample_sys() {
    const uint64_t t = ggml_evlog_now();
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256]; long long avail = -1, cma = 0;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %lld kB", &avail) == 1) { continue; }
            sscanf(line, "CmaFree: %lld kB", &cma);
        }
        fclose(f);
        if (avail >= 0) { ggml_evlog_rec(GGML_EV_SYS, "mem_avail_kb", t, t, avail, cma); }
    }
    if (FILE * f = fopen("/proc/self/stat", "r")) {
        char buf[1024]; size_t r = fread(buf, 1, sizeof(buf) - 1, f); fclose(f); buf[r] = 0;
        const char * p = strrchr(buf, ')');   // fields after the command name
        if (p) {
            // after ') ': the state letter, then fields 4.. of proc(5): v[0] ppid ... v[8] majflt, v[10] utime, v[11] stime, v[20] rss (pages)
            unsigned long long v[24] = {0};
            const char * q = p + 2;
            while (*q && *q != ' ') { ++q; }
            for (int k = 0; k < 24 && *q; ++k) { char * end = nullptr; v[k] = strtoull(q, &end, 10); if (end == q) { break; } q = end; }
            const long tck = sysconf(_SC_CLK_TCK);
            ggml_evlog_rec(GGML_EV_SYS, "majflt_cpu_ms", t, t, (int64_t) v[8], (int64_t) ((v[10] + v[11]) * 1000 / (tck > 0 ? tck : 100)));
            ggml_evlog_rec(GGML_EV_SYS, "rss_kb", t, t, (int64_t) v[20] * (sysconf(_SC_PAGESIZE) / 1024), 0);
        }
    }
}
#else
static void ev_sample_sys() {}
#endif

static void ev_flush_locked(ev_log & L) {
    if (!L.fp) { return; }
    const uint64_t h = L.head.load(std::memory_order_acquire);
    if (h - L.written > L.n) { L.lost += h - L.written - L.n; L.written = h - L.n; }
    uint64_t i = L.written;
    for (; i < h; ++i) {
        const ev_rec & r = L.ring[i & (L.n - 1)];
        if (!__atomic_load_n(&r.ok, __ATOMIC_ACQUIRE)) { break; }   // still being written: next flush
        fprintf(L.fp, "%llu\t%llu\t%s\t%s\t%u\t%lld\t%lld\n", (unsigned long long) r.t0, (unsigned long long) r.t1, k_cat_names[r.cat < GGML_EV_N_CAT ? r.cat : 0],
                r.name ? r.name : "?", r.tid, (long long) r.a0, (long long) r.a1);
    }
    L.written = i;
    if (L.lost) { fprintf(L.fp, "# lost %llu records (ring lapped the flusher; raise GGML_EVLOG_N)\n", (unsigned long long) L.lost); L.lost = 0; }
    fflush(L.fp);
}

static void ev_init() {
    const char * e = getenv("GGML_EVLOG");
    if (!e || !*e || strcmp(e, "0") == 0) { return; }
    uint32_t mask = 0;
    if (strcmp(e, "all") == 0 || strcmp(e, "1") == 0) {
        mask = (1u << GGML_EV_N_CAT) - 1;
    } else {
        std::string s(e);
        size_t p = 0;
        while (p <= s.size()) {
            const size_t q = s.find(',', p);
            const std::string w = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
            for (int c = 0; c < GGML_EV_N_CAT; ++c) { if (w == k_cat_names[c]) { mask |= 1u << c; } }
            if (q == std::string::npos) { break; }
            p = q + 1;
        }
    }
    if (!mask) { return; }
    ev_log * L = new ev_log();
    uint64_t n = 1u << 20;
    if (const char * ne = getenv("GGML_EVLOG_N")) { n = strtoull(ne, nullptr, 10); }
    uint64_t p2 = 1024; while (p2 < n) { p2 <<= 1; }
    L->n = p2;
    L->ring.resize(p2);   // zeroed: ok == 0
    std::string path;
    if (const char * f = getenv("GGML_EVLOG_FILE")) { path = f; }
#if defined(__linux__)
    if (path.empty()) { path = "/dev/shm/evlog-" + std::to_string((long) getpid()) + ".tsv"; }
#else
    if (path.empty()) { path = "evlog.tsv"; }
#endif
    L->fp = fopen(path.c_str(), "a");
    if (!L->fp) { fprintf(stderr, "evlog: cannot open %s, event log off\n", path.c_str()); delete L; return; }
    fprintf(L->fp, "# evlog mono_ns %llu real_ns %llu mask %s\n", (unsigned long long) ggml_evlog_now(), (unsigned long long) real_ns(), e);
    fflush(L->fp);
    fprintf(stderr, "evlog: on (%s), records to %s\n", e, path.c_str());
    g_log = L;
    ggml_evlog_mask = mask;
    L->flusher = std::thread([L] {
        int k = 0;
        while (!L->stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (GGML_EV_ON(GGML_EV_SYS)) { ev_sample_sys(); }
            if (++k % 5 == 0) { std::lock_guard<std::mutex> lk(L->flush_mu); ev_flush_locked(*L); }
        }
    });
    L->flusher.detach();   // never joined: at exit the atexit flush writes the rest, the thread dies with the process
    atexit([] { ggml_evlog_flush(); });
}

struct ev_boot { ev_boot() { ev_init(); } };
static ev_boot g_boot;

} // namespace

extern "C" uint64_t ggml_evlog_now(void) {
#if defined(__linux__)
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
#else
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

extern "C" void ggml_evlog_rec(int cat, const char * name, uint64_t t0, uint64_t t1, int64_t a0, int64_t a1) {
    ev_log * L = g_log;
    if (!L) { return; }
    const uint64_t i = L->head.fetch_add(1, std::memory_order_relaxed);
    ev_rec & r = L->ring[i & (L->n - 1)];
    __atomic_store_n(&r.ok, (uint16_t) 0, __ATOMIC_RELAXED);
    r.t0 = t0; r.t1 = t1; r.name = name; r.a0 = a0; r.a1 = a1; r.tid = ev_tid(); r.cat = (uint16_t) cat;
    __atomic_store_n(&r.ok, (uint16_t) 1, __ATOMIC_RELEASE);
}

extern "C" void ggml_evlog_flush(void) {
    ev_log * L = g_log;
    if (!L) { return; }
    std::lock_guard<std::mutex> lk(L->flush_mu);
    ev_flush_locked(*L);
}

extern "C" char * ggml_evlog_snapshot(uint64_t since, uint64_t * next) {
    ev_log * L = g_log;
    if (!L) { if (next) { *next = 0; } return nullptr; }
    const uint64_t h = L->head.load(std::memory_order_acquire);
    if (h > L->n && since < h - L->n) { since = h - L->n; }
    std::string out;
    char line[320];
    snprintf(line, sizeof(line), "# evlog mono_ns %llu real_ns %llu from %llu\n", (unsigned long long) ggml_evlog_now(), (unsigned long long) real_ns(), (unsigned long long) since);
    out += line;
    uint64_t i = since;
    for (; i < h; ++i) {
        const ev_rec r = L->ring[i & (L->n - 1)];   // copy, then check it was whole (a writer may lap it while we read)
        if (!__atomic_load_n(&L->ring[i & (L->n - 1)].ok, __ATOMIC_ACQUIRE) || !r.ok) { break; }
        snprintf(line, sizeof(line), "%llu\t%llu\t%s\t%s\t%u\t%lld\t%lld\n", (unsigned long long) r.t0, (unsigned long long) r.t1, k_cat_names[r.cat < GGML_EV_N_CAT ? r.cat : 0],
                 r.name ? r.name : "?", r.tid, (long long) r.a0, (long long) r.a1);
        out += line;
    }
    if (next) { *next = i; }
    char * t = (char *) malloc(out.size() + 1);
    memcpy(t, out.c_str(), out.size() + 1);
    return t;
}

extern "C" void ggml_evlog_free(char * text) { free(text); }
