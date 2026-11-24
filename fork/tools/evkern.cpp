// evkern: every CUDA kernel and memcpy of any CUDA program (ours, stock, other engines), with no rebuild of that program, in the dev event log's line format.
// Loaded by the CUDA driver through CUDA_INJECTION64_PATH=libevkern.so (run.py -ev ...,kern sets it). CUPTI activity records, timestamps on CLOCK_MONOTONIC
// (same clock as ggml-evlog-fork), written by CUPTI's buffer thread to /dev/shm/evk-<pid>.tsv:
//   t0 t1 kern <demangled name, args cut> <stream> <grid x*y*z> <device>      t0 t1 memcpy HtoD|DtoH|DtoD|PtoP|... <stream> <bytes> <device>
// build: g++ -O2 -shared -fPIC -o libevkern.so evkern.cpp -I$CUPTI/include -L$CUPTI/lib -l:libcupti.so.13 -Wl,-rpath,$CUPTI/lib
#include <cupti.h>
#include <cxxabi.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <mutex>

static FILE * g_fp;
static std::mutex g_mu;
static std::unordered_map<const char *, std::string> g_names;   // CUPTI keeps the name strings alive; demangle once per pointer

static uint64_t CUPTIAPI mono_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; }

static const std::string & short_name(const char * m) {
    auto it = g_names.find(m);
    if (it != g_names.end()) return it->second;
    int st = 0; char * d = abi::__cxa_demangle(m, nullptr, nullptr, &st);
    std::string s = st == 0 && d ? d : m; free(d);
    for (size_t p; (p = s.find("(anonymous namespace)")) != std::string::npos;) s.replace(p, 21, "anon");
    // drop the argument list and a leading 'void ': kernels differ by name + template args, not by parameter types
    int depth = 0; size_t cut = std::string::npos;
    for (size_t i = 0; i < s.size(); ++i) { char c = s[i]; if (c == '<') depth++; else if (c == '>') depth--; else if (c == '(' && depth == 0) { cut = i; break; } }
    if (cut != std::string::npos) s.resize(cut);
    if (s.rfind("void ", 0) == 0) s = s.substr(5);
    for (char & c : s) if (c == '\t' || c == '\n') c = ' ';
    return g_names.emplace(m, s).first->second;
}

static const char * copy_kind(uint8_t k) {
    switch (k) { case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD"; case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH"; case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
                 case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP"; case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH"; default: return "copy"; }
}

static void CUPTIAPI buf_req(uint8_t ** buf, size_t * size, size_t * maxrec) { *size = 8 << 20; *buf = (uint8_t *) aligned_alloc(8, *size); *maxrec = 0; }

static void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t * buf, size_t, size_t valid) {
    std::lock_guard<std::mutex> lk(g_mu);
    CUpti_Activity * r = nullptr;
    while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto * k = (CUpti_ActivityKernel10 *) r;
            fprintf(g_fp, "%llu\t%llu\tkern\t%s\t%u\t%lld\t%u\n", (unsigned long long) k->start, (unsigned long long) k->end, short_name(k->name).c_str(), k->streamId,
                    (long long) k->gridX * k->gridY * k->gridZ, k->deviceId);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto * c = (CUpti_ActivityMemcpy6 *) r;
            fprintf(g_fp, "%llu\t%llu\tmemcpy\t%s\t%u\t%llu\t%u\n", (unsigned long long) c->start, (unsigned long long) c->end, copy_kind(c->copyKind), c->streamId,
                    (unsigned long long) c->bytes, c->deviceId);
        }
    }
    size_t dropped = 0; cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped) fprintf(g_fp, "# dropped %zu records\n", dropped);
    fflush(g_fp);
    free(buf);
}

extern "C" int InitializeInjection(void) {
    std::string path = getenv("EVKERN_FILE") ? getenv("EVKERN_FILE") : "/dev/shm/evk-" + std::to_string((long) getpid()) + ".tsv";
    g_fp = fopen(path.c_str(), "a");
    if (!g_fp) return 0;
    fprintf(g_fp, "# evkern mono_ns %llu\n", (unsigned long long) mono_ns()); fflush(g_fp);
    cuptiActivityRegisterTimestampCallback(mono_ns);
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    std::thread([] { for (;;) { usleep(500000); cuptiActivityFlushAll(0); } }).detach();
    atexit([] { cuptiActivityFlushAll(1); });
    return 1;
}
