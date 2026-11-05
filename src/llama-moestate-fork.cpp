#include "llama-moestate-fork.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "../include/llama.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <cctype>
#include <chrono>
#ifdef __linux__
#include <dlfcn.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>

// the user's cache directory of the fork's files (created): <XDG_CACHE_HOME or ~/.cache>/llama.cpp, %LOCALAPPDATA%\llama.cpp on Windows; "" when there is none
std::string moe_user_dir() {
#ifdef _WIN32
    const char * base = getenv("LOCALAPPDATA");
    const std::string dir = base ? std::string(base) + "\\llama.cpp" : "";
#else
    const char * xdg  = getenv("XDG_CACHE_HOME");
    const char * home = getenv("HOME");
    const std::string dir = xdg ? std::string(xdg) + "/llama.cpp" : home ? std::string(home) + "/.cache/llama.cpp" : "";
#endif
    if (dir.empty()) {
        return "";
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

namespace {

using section_t = std::map<std::string, std::string>;
using file_t    = std::map<std::string, section_t>;

std::mutex g_mtx;
bool g_suppress = false; // the state of this run is marked for deletion: no writes until the process ends
std::string g_model_section;
std::string g_geom;          // arch.experts.experts_used of the loaded model: what its hot map is comparable on
std::string g_geom_section;  // the section this run writes the geom tag into

std::string state_path() {
    if (const char * e = moe_opt("state")) {
        return strcmp(e, "0") == 0 ? "" : e;
    }
    const std::string dir = moe_user_dir();
    return dir.empty() ? "" : dir + "/moe-state.ini";
}

std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char) s[a])) { a++; }
    while (b > a && isspace((unsigned char) s[b - 1])) { b--; }
    return s.substr(a, b - a);
}

file_t load(const std::string & path) {
    file_t f;
    std::ifstream in(path);
    std::string line, cur;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        if (line[0] == '[' && line.back() == ']') {
            cur = line.substr(1, line.size() - 2);
            f[cur];
            continue;
        }
        const size_t eq = line.find('=');
        if (eq != std::string::npos && !cur.empty()) {
            f[cur][trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
        }
    }
    return f;
}

bool save(const std::string & path, const file_t & f) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            return false;
        }
        out << "# llama.cpp MoE state: what the engine learned per model (hot experts, tuner decisions, placement). Safe to delete.\n";
        for (const auto & s : f) {
            if (s.second.empty()) {
                continue;
            }
            out << "\n[" << s.first << "]\n";
            for (const auto & kv : s.second) {
                out << kv.first << " = " << kv.second << "\n";
            }
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

} // namespace

static std::string g_place_prefix;

void moe_state_set_place(const std::string & prefix) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_place_prefix = prefix;
}

const std::string & moe_state_place() {
    return g_place_prefix;
}

void moe_state_set_model(const std::string & section) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_model_section = section;
}

std::string moe_state_section(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_geom = std::string(llm_arch_name(model.arch)) + "." + std::to_string(model.hparams.n_expert) + "." + std::to_string(model.hparams.n_expert_used());
    if (!g_model_section.empty()) {
        return g_geom_section = g_model_section;
    }
    std::string name = model.name;
    for (char & c : name) {
        if (c == '[' || c == ']' || c == '\n' || c == '\r') { c = '_'; }
    }
    return g_geom_section = name + " " + std::to_string(model.size());
}

std::string moe_state_file() {
    return state_path();
}

bool moe_state_enabled() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return !state_path().empty();
}

bool moe_state_get(const std::string & section, const std::string & key, std::string & value) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const std::string path = state_path();
    if (path.empty() || section.empty()) {
        return false;
    }
    const file_t f = load(path);
    const auto s = f.find(section);
    if (s == f.end()) {
        return false;
    }
    const auto k = s->second.find(key);
    if (k == s->second.end()) {
        return false;
    }
    value = k->second;
    return true;
}

bool moe_state_section_kv(const std::string & section, std::vector<std::pair<std::string, std::string>> & kv, bool any_size) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const std::string path = state_path();
    kv.clear();
    if (path.empty() || section.empty()) {
        return false;
    }
    const file_t f = load(path);
    auto s = f.find(section);
    if (s == f.end() && any_size) {   // same file name (any case), other byte size (re-upload, other build of the file): the section with the most keys
        auto lower_name = [](std::string n) {
            n.resize(std::min(n.size(), n.find_last_of(' ')));
            for (char & c : n) { c = (char) tolower((unsigned char) c); }
            return n;
        };
        const std::string name = lower_name(section);
        for (auto it = f.begin(); it != f.end(); ++it) {
            if (lower_name(it->first) == name && (s == f.end() || it->second.size() > s->second.size())) { s = it; }
        }
    }
    if (s == f.end() && any_size && !g_geom.empty()) {   // same family and shape (hot maps of other quants and builds are alike): the section with the most keys
        for (auto it = f.begin(); it != f.end(); ++it) {
            const auto g = it->second.find("geom");
            if (g != it->second.end() && g->second == g_geom && (s == f.end() || it->second.size() > s->second.size())) { s = it; }
        }
    }
    if (s == f.end()) {
        return false;
    }
    kv.assign(s->second.begin(), s->second.end());
    return true;
}

bool moe_state_set(const std::string & section, const std::vector<std::pair<std::string, std::string>> & kv) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const std::string path = state_path();
    if (path.empty() || section.empty() || g_suppress) {
        return false;
    }
    file_t f = load(path);
    for (const auto & p : kv) {
        f[section][p.first] = p.second;
    }
    if (section == g_geom_section && !g_geom.empty()) { f[section]["geom"] = g_geom; }
    return save(path, f);
}

bool moe_state_erase(const std::string & section, const std::string & prefix) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const std::string path = state_path();
    if (path.empty() || section.empty()) {
        return false;
    }
    file_t f = load(path);
    auto s = f.find(section);
    if (s == f.end()) {
        return true;
    }
    for (auto it = s->second.begin(); it != s->second.end();) {
        it = it->first.compare(0, prefix.size(), prefix) == 0 ? s->second.erase(it) : std::next(it);
    }
    return save(path, f);
}

// C API for the tools (common): the same store
extern "C" {
bool llama_state_get(const char * section, const char * key, char * value, size_t n) {
    std::string v;
    if (!moe_state_get(section, key, v) || v.size() + 1 > n) {
        return false;
    }
    memcpy(value, v.c_str(), v.size() + 1);
    return true;
}
bool llama_state_set(const char * section, const char * key, const char * value) {
    return moe_state_set(section, { { key, value } });
}
void llama_state_set_place(const char * prefix) {
    moe_state_set_place(prefix ? prefix : "");
}

bool llama_state_erase(const char * section, const char * key_prefix) {
    return moe_state_erase(section, key_prefix);
}
void llama_state_suppress_writes(bool suppress) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_suppress = suppress;
}
void llama_state_set_model(const char * section) {
    moe_state_set_model(section ? section : "");
}
}

// research: embedding-keyed hot expert snapshots (see llama-moestate-fork.h)
static std::string g_embsnap_hex;

bool moe_log_has(const char * letters) {   // --moe log=LETTERS (LLAMA_MOE_LOG)
    const char * e = moe_opt("log");
    return e && strpbrk(e, letters);
}

static bool log_has(const char * letters) { return moe_log_has(letters) && moe_state_enabled(); }

bool moe_embsnap_enabled() {
    return log_has("e");
}

void moe_embsnap_set(const float * emb, int n) {
    // the whole embedding, 1 byte per float (absmax scaled, 2 hex characters each)
    if (n <= 0) { return; }
    float mx = 1e-20f;
    for (int i = 0; i < n; ++i) { mx = std::max(mx, std::fabs(emb[i])); }
    std::string hex;
    hex.reserve((size_t) n*2);
    char t[4];
    for (int i = 0; i < n; ++i) {
        snprintf(t, sizeof t, "%02x", (unsigned) (uint8_t) (int8_t) std::lround(emb[i]/mx*127.0f));
        hex += t;
    }
    std::lock_guard<std::mutex> lk(g_mtx);
    g_embsnap_hex = hex;
}

void moe_embsnap_write(const std::vector<std::pair<std::string, std::string>> & hot) {
    std::string hex, model;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        hex = g_embsnap_hex;
        model = g_geom_section;
    }
    if (hex.empty() || hot.empty()) { return; }
    const std::filesystem::path file = moe_log_file("embhot.csv");
    std::error_code ec;
    const bool fresh = !std::filesystem::exists(file, ec) || std::filesystem::file_size(file, ec) == 0;
    // one line per run, built whole and appended with one write: model,embhex,hots. hots = layer:count count ...;layer:... (no commas in any field)
    std::string line = fresh ? "model,embhex,hots\n" : "";
    std::string hots;
    for (const auto & p : hot) {
        hots += (hots.empty() ? "" : ";") + p.first.substr(p.first.find('.') + 1) + ":" + p.second;
    }
    for (char & c : model) { if (c == ',' || c == '\n') { c = '_'; } }
    line += model + "," + hex + "," + hots + "\n";
    std::ofstream f(file, std::ios::app | std::ios::binary);
    f.write(line.data(), (std::streamsize) line.size());
}

// research: state snapshot of one model (see llama-moestate-fork.h)
bool moe_snap_enabled() {
    return log_has("si");
}

void moe_snap_save(const std::string & section) {
    std::vector<std::pair<std::string, std::string>> kv;
    if (!moe_snap_enabled() || !moe_state_section_kv(section, kv) || kv.empty()) { return; }
    const std::filesystem::path dir = moe_log_file("state-snapshots");
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    char ts[32];
    const time_t now = time(nullptr);
    strftime(ts, sizeof ts, "%Y-%m-%d_%H%M%S", localtime(&now));
    std::string name = section.substr(0, section.find_last_of(' '));
    for (char & c : name) { if (!isalnum((unsigned char) c) && c != '.' && c != '-' && c != '_') { c = '_'; } }
    std::ofstream f(dir / (std::string(ts) + "-" + name + ".ini"));
    f << "[" << section << "]\n";
    for (const auto & p : kv) { f << p.first << " = " << p.second << "\n"; }
}

int moe_log_num(char letter, int dflt) {
    const char * e = moe_opt("log");
    const char * p = e ? strchr(e, letter) : nullptr;
    return p && isdigit((unsigned char) p[1]) ? atoi(p + 1) : dflt;
}

#ifdef _WIN32
#define setenv(n, v, o) _putenv_s((n), (v))   // MSVC has no setenv
#endif

// --moe settings
static std::map<std::string, std::string> g_moe_opts;   // node based: the returned pointers stay valid
static std::mutex g_opt_mtx;                             // its own lock: the state file code reads options while it holds g_mtx

void llama_moe_set_opt(const char * key, const char * value) {
    std::lock_guard<std::mutex> lk(g_opt_mtx);
    g_moe_opts[key] = value ? value : "";
    // pp-only=1 (measurements): nothing of the generation part happens, so a run times the prompt pass alone. The refill of the blocks a prompt borrowed
    // (reheat) is background upload that would run into the next request's pass, and the dynhot store reads the answer; both are switched off unless set by hand.
    // A request still samples its first token from the prompt pass; ask for n_predict 1 and read timings.prompt_ms.
    if (value && strcmp(key, "pp-only") == 0 && atoi(value) != 0) {
        if (!g_moe_opts.count("reheat")) { g_moe_opts["reheat"] = "0"; }
        if (!g_moe_opts.count("dynhot")) { g_moe_opts["dynhot"] = "off"; }
    }
    // sim-moe-repeat=N / sim-dense-repeat=N: run prompt-sized expert / dense matmuls (and attention) N times in the CUDA backend, to give a small model the compute-to-upload
    // ratio of a big one in scheduler tests (OLMoE standing in for GLM). The backend reads GGML_CUDA_MMID_REPEAT / GGML_CUDA_DENSE_REPEAT at its first prompt graph.
    if (value && strcmp(key, "sim-moe-repeat") == 0)   { setenv("GGML_CUDA_MMID_REPEAT", value, 1); }
    if (value && strcmp(key, "sim-dense-repeat") == 0) { setenv("GGML_CUDA_DENSE_REPEAT", value, 1); }
    // pass-stats=1 (off by default; debugging and measurement runs): every split pass's time per card by cause and its link bytes, measured with extra stream
    // stamps and counters in the scheduler (GGML_SCHED_PASS_STATS, read at the first prompt graph); run.py turns it on unless -ns
    // predicted staging of prompt passes: q1 = a token's picks, q2 = up to pp-k x those picks as far as the link allows; pp-depth: layers ahead (1 = the next layer)
    if (value && strcmp(key, "pp-k") == 0 && strcmp(value, "0") != 0) { setenv("GGML_PRED_K", "1", 1); setenv("GGML_PRED_KM", "1", 1); setenv("GGML_PRED_KM2", strcmp(value, "auto") == 0 ? "4096" : value, 1); }   // auto: every rank, the time left decides how far q2 goes, per layer and per pass
    if (value && strcmp(key, "pp-depth") == 0)         { setenv("GGML_PRED_DEPTH", value, 1); }
    if (value && strcmp(key, "pass-stats") == 0)       { setenv("GGML_SCHED_PASS_STATS", value, 1); }
    // pass-trace=N (debugging, with pass-stats=1): the split-by-split timeline of the first N big passes (GGML_SCHED_PASS_TRACE): are the cards side by side or in turn
    if (value && strcmp(key, "pass-trace") == 0)       { setenv("GGML_SCHED_PASS_TRACE", value, 1); }
    if (value && strcmp(key, "xsplit-dense") == 0)     { setenv("GGML_SCHED_XSPLIT_DENSE", value, 1); }   // research: a node reading another card's result starts a new split in dense splits too
    if (value && strcmp(key, "pass-trace-ops") == 0)   { setenv("GGML_SCHED_PASS_TRACE_OPS", value, 1); }   // with pass-trace: every split's ops under its line
}

const char * llama_moe_get_opt(const char * key) {
    std::lock_guard<std::mutex> lk(g_opt_mtx);
    const auto it = g_moe_opts.find(key);
    return it == g_moe_opts.end() ? nullptr : it->second.c_str();
}

// ---- live parameters ---------------------------------------------------------------------------------------------------------------------------
// Values the user may change while the server runs (POST /moe/params), read at every use. Each has a default, a range and a one-line help.
namespace {
struct live_param {
    const char * name; double def, lo, hi; const char * help; std::atomic<double> v;
    live_param(const char * n, double d, double l, double h, const char * hp) : name(n), def(d), lo(l), hi(h), help(hp), v(d) {}
};
live_param g_live[] = {
    { "min_in_len",     16,   1,    4096,  "tokens of the shortest real prompt: the chat template's own tokens plus one (the server sets it at load from the model's template; \"2+2=\" through the template is 13-16)" },
    { "def_out_len",    256,  1,    65536, "tokens a request is expected to generate when it sets no limit: the room kept for the answer when an over-long prompt is cut" },
    { "prompt_overflow",1,    0,    1,     "1: a prompt longer than the context is cut (start and end kept) and run; 0: refused with an error" },
    { "cut_head_frac",  0.5,  0.05, 0.95,  "share of the kept prompt that is its start when the middle is cut (the rest is the end)" },
    { "split_min_gain", 0.08, 0.0,  0.5,   "the dense split needs to be predicted this much faster than one card (fraction of the one-card time) to be used" },
    { "refill_weight",  1.0,  0.0,  10.0,  "scale of the planner's charge for refilling the cache blocks a split pass lends (0 = not charged)" },
};
}
double llama_moe_live_get(const char * name) {
    for (auto & p : g_live) { if (strcmp(p.name, name) == 0) { return p.v.load(std::memory_order_relaxed); } }
    return std::nan("");
}
bool llama_moe_live_set(const char * name, double value) {
    for (auto & p : g_live) {
        if (strcmp(p.name, name) == 0) { p.v.store(std::min(p.hi, std::max(p.lo, value)), std::memory_order_relaxed); return true; }
    }
    return false;
}
int llama_moe_live_json(char * buf, int cap) {
    std::string o = "{";
    for (size_t i = 0; i < sizeof(g_live) / sizeof(g_live[0]); ++i) {
        const live_param & p = g_live[i];
        char b[640];
        snprintf(b, sizeof b, "%s\"%s\":{\"value\":%g,\"default\":%g,\"min\":%g,\"max\":%g,\"help\":\"%s\"}", i ? "," : "", p.name, p.v.load(), p.def, p.lo, p.hi, p.help);
        o += b;
    }
    o += "}";
    if (cap <= 0) { return (int) o.size(); }
    const int n = std::min<int>((int) o.size(), cap - 1);
    memcpy(buf, o.data(), n); buf[n] = 0;
    return n;
}

std::string llama_moe_bw_json_fork();   // src/llama-moecache-fork.cpp
std::string llama_dsplit_state_json_fork();   // src/llama-dsplit-fork.cpp

// ---- engine state for GET /moe/stats: time per token, prompt passes, auto decisions, adaptive pass size ----
static std::mutex g_tok_mu;
static double g_tok_ring[64];
static unsigned long long g_tok_n = 0;
static double g_tok_sum_ms = 0.0;
struct decision_t { std::string name, value, why; int64_t t; };
static std::vector<decision_t> g_decisions;
// prompt calls: last call, and the adaptive pass size controller (tokens/s per tried size)
static unsigned long long g_pp_calls = 0;
static uint32_t g_pp_tokens = 0, g_pp_ub = 0, g_ub_ceil = 0;
static double g_pp_ms = 0;
static std::map<uint32_t, double> g_ub_tps;   // EMA of prompt tokens/s per pass size
static uint32_t g_ub_cur = 0, g_ub_calls_since = 0;

static void tok_window_note(double ms);
void llama_moe_token_time_fork(double ms) {
    tok_window_note(ms);
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_tok_ring[g_tok_n % 64] = ms;
    g_tok_n++;
    g_tok_sum_ms += ms;
}
// the latest value and reason of each automatic decision; a repeat of the same value only refreshes the time
// --moe stats=PATH: the planner's picks and what they measured afterwards, one line each, with the wall clock and the seconds since the start (close to the server log's clock), apart from
// the server log: a speed drop reads against this file (user 2026-11-02: "log key planner decisions to separate stats.log with timestamps so we can monitor runs and correlate perf drops more simply").
// One event or one status report per line, with a type acronym (user 2026-11-02: "one line per event with timestamp and evt type acronim seems simpler than wastefull csv"; tail -f it to watch a change live, run.py stats does).
// The first line of the file names the types. STA comes once a second from stats_status_run, GPU once a second from run.py (nvidia-smi dmon).
static const auto g_stats_t0 = std::chrono::steady_clock::now();
static const auto g_stats_wall0 = std::chrono::system_clock::now();   // the wall clock of the same instant: a line may carry an earlier time than the moment it is written
// --moe log-profile=NAME[+NAME]: what the engine logs on top of its own server log (user 2026-11-02: "make all the heavy binary logging optional on demand, logging profiles"; "just planner and make stats included
// every profile"). Every profile writes the status lines of stats.log. stats: only those. planner: its event lines too. activations: the per-prompt hot maps too (--moe dynhot=store, until now on by default: it
// records them and nothing reads them back in that mode). all: every one of these. off: nothing. Without the key: planner when --moe stats=PATH is given, else nothing. A key of its own (--moe dynhot=full, ledger=1,
// tokdump=PATH) needs no profile.
bool llama_moe_log_profile_fork(const char * name) {
    const char * o = llama_moe_get_opt("log-profile");
    const std::string v = o ? o : llama_moe_get_opt("stats") ? "planner" : "";
    if (v.empty() || v == "off") { return false; }
    if (strcmp(name, "stats") == 0) { return true; }
    std::istringstream is(v);
    std::string w;
    while (std::getline(is, w, '+')) {
        if (w == name || w == "all") { return true; }
    }
    return false;
}

static void stats_emit(const char * kind, const char * text, double at) {   // at: seconds since the start of the process this line is for, or negative for now
    static std::mutex mu;
    static FILE * f = nullptr;
    static bool opened = false;
    static const bool planner = llama_moe_log_profile_fork("planner");
    if (!planner && strcmp(kind, "STA") != 0) { return; }
    std::lock_guard<std::mutex> lk(mu);
    if (!opened) {
        opened = true;
        const char * p = llama_moe_get_opt("stats");   // a profile without a path: stats.log in the log directory
        const std::string path = p && *p ? p : moe_log_file("stats.log");
        if (!path.empty()) {
            p = path.c_str();
            f = fopen(p, "a");
            if (!f) { LLAMA_LOG_WARN("%s: cannot open %s\n", __func__, p); }
            else if (ftell(f) == 0 && fseek(f, 0, SEEK_END) == 0 && ftell(f) == 0) {
                fprintf(f, "# wall clock, seconds since start, type, text. PHA phase change with what the phase gave, SPN one big operation with its start, stop, bytes and rate (start-up stages, prompt calls, fills), DEC decision, REQ request, CAL prompt call and its path, DON what the call took, PAS prompt pass, SPL split plan, PRE prefill result, CHN chain refused, SWT misses switch between CPU and uploads, KVZ KV zone, DEW decode window, STA engine status, GPU cards (once a 10 s, run.py), SMP fast GPU samples every 100 ms from NVML (load, PCIe rx and tx, power, clock, free VRAM; flushed every 10 s), HST host\n");
            }
        }
    }
    if (!f) { return; }
    const double rel = at >= 0.0 ? at : std::chrono::duration<double>(std::chrono::steady_clock::now() - g_stats_t0).count();
    const auto now = g_stats_wall0 + std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::duration<double>(rel));
    const time_t t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    const int ms = (int) (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
    fprintf(f, "%s.%03d %9.3f %-3s %s\n", ts, ms, rel, kind, text);
    fflush(f);
}

void llama_moe_stats_log_fork(const char * kind, const char * text) { stats_emit(kind, text, -1.0); }
void llama_moe_stats_emit_at_fork(const char * kind, const char * text, double at) { stats_emit(kind, text, at); }

// Spans (user 2026-11-05: "every big load logged in mem and flushed in stats.log, one timestamp start stop events per line ... more info in the debug trace phase, we waste less on nsys,
// we spot deviations on any run, not just on demand rare nsys"): one line per big operation with its start, its stop, its bytes and the rate. The hot path only appends a record
// to memory; a thread writes the records out every second and at exit, sorted by start. Records exist only with the planner log profile on, as every other line of stats.log.
struct moe_span_rec { double t0, t1; uint64_t bytes; char what[72]; char where[16]; unsigned tid; };
static std::mutex g_span_mu;
static std::vector<moe_span_rec> g_span_buf;

double llama_moe_span_now_fork() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_stats_t0).count(); }

// The drive's measured maximum (the read probe, GB/s): a span whose `where` is "SSD..." or "O_DIRECT..." also logs its rate as a share of it, so an inefficient big read stands out.
static std::atomic<double> g_ssd_max_gbs{0.0};
void llama_moe_ssd_max_fork(double gbs) { g_ssd_max_gbs.store(gbs); }

static void span_flush() {
    std::vector<moe_span_rec> v;
    { std::lock_guard<std::mutex> lk(g_span_mu); v.swap(g_span_buf); }
    std::sort(v.begin(), v.end(), [](const moe_span_rec & a, const moe_span_rec & b) { return a.t0 < b.t0; });
    for (const auto & r : v) {
        const double dur = r.t1 - r.t0;
        std::string line = format("%s: start %.3f stop %.3f dur %.1f ms", r.what, r.t0, r.t1, dur * 1e3);
        if (r.bytes) { line += format(", %.1f MiB at %.2f GB/s", (double) r.bytes / 1048576.0, dur > 0.0 ? (double) r.bytes / dur / 1e9 : 0.0); }
        if (r.where[0]) { line += format(", %s", r.where); }
        const double mx = g_ssd_max_gbs.load();
        if (r.bytes && dur > 0.0 && mx > 0.0 && (!strncmp(r.where, "SSD", 3) || !strncmp(r.where, "O_DIRECT", 8))) {
            line += format(", %.0f%% of the measured SSD maximum %.2f GB/s", 100.0 * (double) r.bytes / dur / 1e9 / mx, mx);
        }
        line += format(", thread %u", r.tid);
        stats_emit("SPN", line.c_str(), r.t0);
    }
}

namespace {
struct span_flusher {   // defined after the buffer: at exit it is destroyed first, and writes what is left
    std::thread th;
    std::atomic<bool> stop { false };
    span_flusher() { th = std::thread([this] { while (!stop.load()) { for (int i = 0; i < 50 && !stop.load(); ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); } span_flush(); } span_flush(); }); }
    ~span_flusher() { stop = true; if (th.joinable()) { th.join(); } }
};
}

void llama_moe_span_fork(const char * what, double t0, double t1, uint64_t bytes, const char * where) {
    static const bool on = llama_moe_log_profile_fork("planner");
    if (!on) { return; }
    static span_flusher flusher;   // the thread starts with the first span
    moe_span_rec r = { t0, t1, bytes, {}, {}, (unsigned) (std::hash<std::thread::id>()(std::this_thread::get_id()) % 100000) };
    snprintf(r.what, sizeof r.what, "%s", what);
    snprintf(r.where, sizeof r.where, "%s", where ? where : "");
    std::lock_guard<std::mutex> lk(g_span_mu);
    g_span_buf.push_back(r);
}

// For stats.log: what the disk, the memory and the CPU did since this caller's last call (who: 0 prompt calls, 1 status lines, 2 decode windows). This process's reads from storage and its page faults; the
// kernel's reclaim scans and swapping; page cache, free RAM, swap; CPU load and temperature (user 2026-11-02: "add monitoring and logging of our read speed", "log swapping", "cpu temp util", "add page faults":
// DeepSeek's second request read at 335 MB/s on a drive that does 5.8 GB/s, and nothing said so). Empty without --moe stats, and on a caller's first call.
std::string llama_moe_stats_io_fork(int who) {
#ifdef __linux__
    static const bool on = llama_moe_log_profile_fork("stats") || llama_moe_log_profile_fork("planner");
    if (!on) { return ""; }
    struct sample { int64_t t = 0; unsigned long long rb = 0, mj = 0, mn = 0, sc = 0, si = 0, so = 0, all = 0, idle = 0, iow = 0; };
    static sample last[3];
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    sample c;
    unsigned long long v = 0, cached = 0, free_kb = 0, avail = 0, swap_total = 0, swap_free = 0;
    std::string k, line;
    { std::ifstream f("/proc/self/io"); while (f >> k >> v) { if (k == "read_bytes:") { c.rb = v; } } }
    {   // minflt and majflt: the 8th and 10th fields behind the process name
        std::ifstream f("/proc/self/stat");
        std::getline(f, line);
        const size_t p = line.rfind(')');
        if (p != std::string::npos) {
            std::istringstream is(line.substr(p + 1));
            for (int i = 0; i < 10 && is >> k; ++i) {
                if (i == 7) { c.mn = strtoull(k.c_str(), nullptr, 10); } else if (i == 9) { c.mj = strtoull(k.c_str(), nullptr, 10); }
            }
        }
    }
    {
        std::ifstream f("/proc/vmstat");
        while (f >> k >> v) {
            if (k == "pgscan_kswapd" || k == "pgscan_direct") { c.sc += v; } else if (k == "pswpin") { c.si = v; } else if (k == "pswpout") { c.so = v; }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        while (std::getline(f, line)) {
            std::istringstream is(line);
            if (is >> k >> v) {
                if (k == "Cached:") { cached = v; } else if (k == "MemFree:") { free_kb = v; } else if (k == "MemAvailable:") { avail = v; }
                else if (k == "SwapTotal:") { swap_total = v; } else if (k == "SwapFree:") { swap_free = v; }
            }
        }
    }
    {   // cpu  user nice system idle iowait irq softirq steal
        std::ifstream f("/proc/stat");
        f >> k;
        for (int i = 0; i < 8 && f >> v; ++i) {
            c.all += v;
            if (i == 3) { c.idle = v; } else if (i == 4) { c.iow = v; }
        }
    }
    static const std::string temp_path = [] {   // the CPU's sensor: k10temp on AMD, coretemp on Intel
        std::error_code ec;
        for (const auto & e : std::filesystem::directory_iterator("/sys/class/hwmon", ec)) {
            std::ifstream f(e.path() / "name");
            std::string n;
            if (f >> n && (n == "k10temp" || n == "coretemp" || n == "zenpower")) { return (e.path() / "temp1_input").string(); }
        }
        return std::string();
    }();
    double temp_c = 0.0;
    if (!temp_path.empty()) { std::ifstream f(temp_path); if (f >> v) { temp_c = (double) v / 1e3; } }
    c.t = ggml_time_us();
    sample & l = last[who >= 0 && who < 3 ? who : 0];
    std::string out;
    if (l.t > 0 && c.t > l.t) {
        const double dt = (double) (c.t - l.t) / 1e6, bytes = (double) (c.rb - l.rb), mj = (double) (c.mj - l.mj), all = std::max(1.0, (double) (c.all - l.all));
        out = format("ssd %.0f MB/s (%.1f GB), faults major %.0f/s minor %.0f/s, %.0f KiB per major, scan %.1f M pages/s, swap in %.0f out %.0f pages/s used %.1f GiB, cache %.1f GiB, ram free %.1f avail %.1f GiB, cpu %.0f%% iowait %.0f%% %.0f C",
                bytes / dt / 1e6, bytes / 1e9, mj / dt, (double) (c.mn - l.mn) / dt, mj > 0.0 ? bytes / 1024.0 / mj : 0.0, (double) (c.sc - l.sc) / dt / 1e6, (double) (c.si - l.si) / dt, (double) (c.so - l.so) / dt,
                (double) (swap_total - swap_free) / 1048576.0, (double) cached / 1048576.0, (double) free_kb / 1048576.0, (double) avail / 1048576.0,
                100.0 * (all - (double) (c.idle - l.idle) - (double) (c.iow - l.iow)) / all, 100.0 * (double) (c.iow - l.iow) / all, temp_c);
    }
    l = c;
    return out;
#else
    GGML_UNUSED(who);
    return "";
#endif
}

// Per-token engine counters (user 2026-11-05: "for gpu you cant sample every 10 tokens but you can do prior and after the kernel run and divide by token count"): before and after a run of decode
// calls (10 tokens) or a prompt pass the engine's byte counters (llama_moe_cache_counters_fork: CPU compute, uploads, prompt streaming, every link) and every card's NVML energy counter are read, and the
// difference is divided by the tokens: bytes per token per link, joules and watts per card. TOK lines come every 10 decode tokens, the same numbers close each PAS line.
#ifdef __linux__
static std::vector<unsigned long long> nvml_energy_mj_all() {
    static void * h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    typedef int (*init_t)(); typedef int (*count_t)(unsigned *); typedef int (*handle_t)(unsigned, void **); typedef int (*en_f)(void *, unsigned long long *);
    static std::vector<void *> dev;
    static en_f fe = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (!h) { return; }
        auto init = (init_t) dlsym(h, "nvmlInit_v2"); auto cnt = (count_t) dlsym(h, "nvmlDeviceGetCount_v2"); auto hnd = (handle_t) dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        fe = (en_f) dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
        unsigned n = 0;
        if (!init || !cnt || !hnd || !fe || init() != 0 || cnt(&n) != 0) { fe = nullptr; return; }
        dev.assign(n, nullptr);
        for (unsigned i = 0; i < n; ++i) { if (hnd(i, &dev[i]) != 0) { dev[i] = nullptr; } }
    });
    std::vector<unsigned long long> out;
    if (!fe) { return out; }
    for (void * d : dev) { unsigned long long e = 0; if (d) { fe(d, &e); } out.push_back(e); }
    return out;
}
#else
static std::vector<unsigned long long> nvml_energy_mj_all() { return {}; }
#endif
int llama_moe_cache_counters_fork(int64_t * v, int cap);   // src/llama-moecache-fork.cpp: cpu, upload, stream bytes, then every link's bytes
struct tok_snap { int64_t c[8] = {}; int n = 0; std::vector<unsigned long long> e; double t = 0.0; };
static tok_snap tok_snap_now() { tok_snap s; s.n = llama_moe_cache_counters_fork(s.c, 8); s.e = nvml_energy_mj_all(); s.t = llama_moe_span_now_fork(); return s; }
static std::string tok_per_token(const tok_snap & a, const tok_snap & b, double tokens) {
    const double wall = b.t - a.t;
    if (a.n == 0 || b.n != a.n || tokens <= 0 || wall <= 0) { return ""; }
    auto mib = [&](int i) { return (double) (b.c[i] - a.c[i]) / 1048576.0 / tokens; };
    std::string s = format("per token: cpu %.2f MiB, upload %.2f MiB, stream %.2f MiB", mib(0), mib(1), mib(2));
    for (int k = 3; k < a.n; ++k) { s += format(", link%d %.2f MiB (%.2f GB/s)", k - 3, mib(k), (double) (b.c[k] - a.c[k]) / wall / 1e9); }
    for (size_t g = 0; g < a.e.size() && g < b.e.size(); ++g) { s += format(", g%zu %.2f J (%.0f W)", g, (double) (b.e[g] - a.e[g]) / 1e3 / tokens, (double) (b.e[g] - a.e[g]) / 1e3 / wall); }
    return s;
}
static void tok_window_note(double ms) {   // one decode call of one token took ms
    static std::mutex mu;
    static tok_snap base;
    static int cnt = 0;
    static double sum_ms = 0.0, last_end = 0.0;
    std::lock_guard<std::mutex> lk(mu);
    const double now = llama_moe_span_now_fork();
    if (now - last_end > 0.5 || base.n == 0) { base = tok_snap_now(); cnt = 0; sum_ms = 0.0; last_end = now; return; }   // a new run of tokens starts counting after its first one
    last_end = now; ++cnt; sum_ms += ms;
    if (cnt >= 10) {
        const tok_snap e = tok_snap_now();
        llama_moe_stats_emit_at_fork("TOK", format("10 tokens: %.2f ms per token (engine), %.2f ms per token (wall); %s", sum_ms / cnt, (e.t - base.t) * 1e3 / cnt, tok_per_token(base, e, cnt).c_str()).c_str(), e.t);
        base = e; cnt = 0; sum_ms = 0.0;
    }
}

// Fast GPU samples (user 2026-11-05: "the 13k call is too short for the samples, collect stats every 10 tokens but flush to file every 10 s"): every card's load, PCIe traffic both ways, power, clock and
// free VRAM from NVML (loaded at run time, nothing links to it) every --moe stats-fast-ms (default 100 ms, 0 = off), kept in memory and written as SMP lines of stats.log every 10 s with the time of
// each sample. A 3 s prompt call so shows its own link traffic; the once-a-10-s GPU lines of run.py (nvidia-smi) stay as the overview. Cards are numbered as nvidia-smi does (g0, g1).
#ifdef __linux__
static void fast_gpu_sampler_start() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!llama_moe_log_profile_fork("planner")) { return; }
        const char * o = llama_moe_get_opt("stats-fast-ms");
        const int period_ms = o ? atoi(o) : 100;
        if (period_ms <= 0) { return; }
        std::thread([period_ms] {
            void * h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
            if (!h) { return; }
            typedef int (*init_t)(); typedef int (*count_t)(unsigned *); typedef int (*handle_t)(unsigned, void **);
            typedef struct { unsigned gpu, mem; } util_t; typedef struct { unsigned long long total, free, used; } meminfo_t;
            typedef int (*util_f)(void *, util_t *); typedef int (*pcie_f)(void *, int, unsigned *); typedef int (*power_f)(void *, unsigned *);
            typedef int (*clock_f)(void *, int, unsigned *); typedef int (*mem_f)(void *, meminfo_t *);
            auto init = (init_t) dlsym(h, "nvmlInit_v2"); auto cnt = (count_t) dlsym(h, "nvmlDeviceGetCount_v2"); auto hnd = (handle_t) dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
            auto fu = (util_f) dlsym(h, "nvmlDeviceGetUtilizationRates"); auto fp = (pcie_f) dlsym(h, "nvmlDeviceGetPcieThroughput"); auto fw = (power_f) dlsym(h, "nvmlDeviceGetPowerUsage");
            auto fc = (clock_f) dlsym(h, "nvmlDeviceGetClockInfo"); auto fm = (mem_f) dlsym(h, "nvmlDeviceGetMemoryInfo");
            if (!init || !cnt || !hnd || !fu || !fp || !fw || !fc || !fm || init() != 0) { return; }
            unsigned n = 0; if (cnt(&n) != 0 || n == 0) { return; }
            std::vector<void *> dev(n, nullptr);
            for (unsigned i = 0; i < n; ++i) { if (hnd(i, &dev[i]) != 0) { dev[i] = nullptr; } }
            struct samp { double t; unsigned util, mem, rx, tx, mw, mhz; unsigned long long freeb; };
            std::vector<std::vector<samp>> buf(n);
            double last_flush = llama_moe_span_now_fork();
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
                const double t = llama_moe_span_now_fork();
                for (unsigned i = 0; i < n; ++i) {
                    if (!dev[i]) { continue; }
                    util_t u = {}; unsigned rx = 0, tx = 0, mw = 0, mhz = 0; meminfo_t m = {};
                    fu(dev[i], &u); fp(dev[i], 1, &rx); fp(dev[i], 0, &tx); fw(dev[i], &mw); fc(dev[i], 1, &mhz); fm(dev[i], &m);
                    buf[i].push_back({ t, u.gpu, u.mem, rx, tx, mw, mhz, m.free });
                }
                if (t - last_flush >= 10.0) {
                    last_flush = t;
                    for (unsigned i = 0; i < n; ++i) {
                        for (const auto & s : buf[i]) {
                            llama_moe_stats_emit_at_fork("SMP", format("g%u util %u%% mem %u%% rx %.2f tx %.2f GB/s %u W %u MHz vram free %.1f GiB", i, s.util, s.mem, s.rx / 1e6, s.tx / 1e6, s.mw / 1000, s.mhz, (double) s.freeb / 1073741824.0).c_str(), s.t);
                        }
                        buf[i].clear();
                    }
                }
            }
        }).detach();
    });
}
#else
static void fast_gpu_sampler_start() {}
#endif

void llama_moe_decision_fork(const char * name, const char * value, const char * why) {
    // the server's phase changes (load, idle, prefill, decode) are events of stats.log, not decisions; through here because llama.h stays as it is
    if (strcmp(name, "phase") == 0) { llama_moe_stats_log_fork("PHA", value); return; }
    // start-up stages (user 2026-11-05: "measure and log all stages and delta times to stats.log as they are happening"): a span from the end of the previous stage to this one
    if (strcmp(name, "stage") == 0) {
        static std::mutex mu;
        static double last = 0.0;   // the first stage counts from the start of the process
        const double now = llama_moe_span_now_fork();
        double t0;
        { std::lock_guard<std::mutex> lk(mu); t0 = last; last = now; }
        fast_gpu_sampler_start();
        llama_moe_span_fork((std::string("start-up: ") + value).c_str(), t0, now, 0, "");
        return;
    }
    const std::string line = std::string(name) + " = " + value + " (" + why + ")";
    std::lock_guard<std::mutex> lk(g_tok_mu);
    const int64_t t = (int64_t) time(nullptr);
    // the decision goes to the log when it changes (run.py calib reads the engine's pass ceiling from "this call fits the ceiling N" there; the live endpoint shows it only while the server runs)
    for (auto & d : g_decisions) {
        if (d.name == name) {
            // a changed value, or a new reason of the pass size (once per prompt call); not every reason update: some tuners rewrite theirs with live numbers each step
            if (d.value != value || (d.why != why && strcmp(name, "ubatch") == 0)) { LLAMA_LOG_WARN("%s: decision %s = %s (%s)\n", __func__, name, value, why); llama_moe_stats_log_fork("DEC", line.c_str()); }
            d.value = value; d.why = why; d.t = t; return;   // the reason follows the live numbers too (a tuner updates it as it measures)
        }
    }
    LLAMA_LOG_WARN("%s: decision %s = %s (%s)\n", __func__, name, value, why);
    llama_moe_stats_log_fork("DEC", line.c_str());
    g_decisions.push_back({ name, value, why, t });
}
// Adaptive prompt pass size, one trial per decode call (the passes of a call are split up front, so a call is the finest measurable step).
// Start at min(2048, ceiling) (the ceiling is the pass size the compute buffer was reserved for); every call's tokens/s is kept per size;
// the next call tries the untried neighbour (double first, then half, never below 512); a size is left for a neighbour only when that is
// > 5% faster, so noise does not move it. Every 16 calls the neighbours are forgotten and tried again (the prompt mix may have changed).
// The half (more passes) only with --moe ubadapt-more=1 (user rule 2026-10-22: no call runs more passes than fit; see the cache path below).
uint32_t llama_moe_ubadapt_next_fork(uint32_t n_tokens, uint32_t ceil) {
    static const bool on = [] { const char * e = moe_opt("ubadapt"); return !e || atoi(e) != 0; }();
    static const bool more = [] { const char * e = moe_opt("ubadapt-more"); return e && atoi(e) != 0; }();
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_ub_ceil = ceil;
    if (!on || ceil <= 512 || n_tokens < 1024) { g_ub_cur = 0; return ceil; }   // a short prompt is one pass anyway; fewer than 2 passes of the start size measure nothing
    if (g_ub_cur == 0 || g_ub_cur > ceil) { g_ub_cur = std::min<uint32_t>(2048, ceil); }
    if (++g_ub_calls_since >= 16) { g_ub_calls_since = 0; const double keep = g_ub_tps.count(g_ub_cur) ? g_ub_tps[g_ub_cur] : 0; g_ub_tps.clear(); if (keep > 0) { g_ub_tps[g_ub_cur] = keep; } }
    // best known size, a neighbour must beat the current one by 5%
    uint32_t best = g_ub_cur; double bt = g_ub_tps.count(best) ? g_ub_tps[best] : 0;
    for (auto & kv : g_ub_tps) { if (kv.second > bt * 1.05) { best = kv.first; bt = kv.second; } }
    uint32_t next = best;
    if (g_ub_tps.count(best)) {   // best measured: try an untried neighbour
        if (best * 2 <= ceil && !g_ub_tps.count(best * 2)) { next = best * 2; }
        else if (more && best / 2 >= 512 && !g_ub_tps.count(best / 2)) { next = best / 2; }
    }
    if (next != g_ub_cur) {
        char v[16], w[160];
        snprintf(v, sizeof v, "%u", next);
        snprintf(w, sizeof w, "hill-climb from %u (%.0f t/s) after %llu prompt calls, ceiling %u", g_ub_cur, g_ub_tps.count(g_ub_cur) ? g_ub_tps[g_ub_cur] : 0.0, g_pp_calls, ceil);
        g_ub_cur = next;
        g_tok_mu.unlock();
        llama_moe_decision_fork("ubatch", v, w);
        g_tok_mu.lock();
    }
    return g_ub_cur;
}
// Same idea on the expert-cache path, where the pass planner (llama_moe_cache_prefill_ubatch_fork) already picked a size from its compute/upload model.
// The model is not checked against a measurement, and the biggest pass is not always the fastest (a big pass holds cache VRAM and uploads more at once),
// so prompts of a similar size (same power of two) are one experiment: the planned pass count p0 first, then p0+1, p0+2 ... (smaller passes, never bigger
// than the planner allowed: it knows the memory limits). Each call's tokens/s is kept per pass count; a count is left for the next one only when that is
// > 5% faster. Every 16 calls of a bucket the neighbours are forgotten and tried again.
// The trial of more passes runs only with --moe ubadapt-more=1; by default a call runs the planner's pass count. The trial ran the second call of a 12k
// prompt (6080 tokens, the same power of two as the 7660-token first call) as 3 x 2048 instead of the planner's 2 x 3040: 3550 against 3990 prompt t/s
// on that call (Qwen3.6-35B, 3 GiB free per card, warm start, 2026-10-22; b13600 had no trial). Every pass streams the experts again (user rule: no call
// runs more passes than fit).
struct ubc_t { std::map<uint32_t, double> tps; uint32_t p0 = 0, calls = 0; };
static std::map<int, ubc_t> g_ubc;
static std::string g_one_why;   // the reason of the one-pass rule; the finished call appends its speed to it
static int g_ubc_bucket = -1; static uint32_t g_ubc_p = 0, g_ubc_size = 0;
uint32_t llama_moe_ubadapt_cache_fork(uint32_t n_tokens, uint32_t planned, uint32_t ceil) {
    static const bool on = [] { const char * e = moe_opt("ubadapt"); return !e || atoi(e) != 0; }();
    static const bool more = [] { const char * e = moe_opt("ubadapt-more"); return e && atoi(e) != 0; }();
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_ubc_bucket = -1; g_ub_ceil = ceil;
    if (!on || planned < 512 || n_tokens < 1024) { return planned; }
    // ... unless the pass planner capped the call below its length: the pool cannot hold that one pass beside the KV (a 6080-token call run as one pass
    // anyway did not fit on a 3 GiB card and the server retried it with halved batches 20 times, 2445 prompt t/s against 3950, 2026-10-22)
    if (n_tokens <= ceil && planned + 31 >= n_tokens) {   // always ONE pass when the whole prompt fits under the ceiling: every expert uploads once (user rule; MiMo 12k prompt, one pass of 16384 = 3016 prompt t/s)
        const uint32_t one = std::min<uint32_t>(ceil, (n_tokens + 31) / 32 * 32);
        char v[16], w[120]; snprintf(v, sizeof v, "%u", one);
        snprintf(w, sizeof w, "1 pass for %u tokens (this call fits the ceiling %u; planner: %u)", n_tokens, ceil, (n_tokens + planned - 1) / planned);
        g_tok_mu.unlock(); llama_moe_decision_fork("ubatch", v, w); g_tok_mu.lock();
        g_one_why = w;
        g_ub_cur = one;   // what the finished call reports as its pass size
        return one;
    }
    const int bk = 31 - __builtin_clz(n_tokens);
    ubc_t & b = g_ubc[bk];
    const uint32_t p0 = (n_tokens + planned - 1) / planned;
    if (b.p0 != p0 || ++b.calls >= 16) {   // the planner's answer moved (cache state, memory) or time to look again: start over, keep the best
        double keep = 0; uint32_t kp = 0;
        for (auto & kv : b.tps) { if (kv.second > keep) { keep = kv.second; kp = kv.first; } }
        b.tps.clear(); if (b.p0 == p0 && kp) { b.tps[kp] = keep; }
        b.p0 = p0; b.calls = 0;
    }
    uint32_t best = p0; double bt = b.tps.count(p0) ? b.tps[p0] : 0;
    for (auto & kv : b.tps) { if (kv.second > bt * 1.05) { best = kv.first; bt = kv.second; } }
    uint32_t p = best;
    if (more && b.tps.count(best)) {   // the best is measured: the next pass count up, if its passes are not below 512
        const uint32_t q = best + 1;
        if (!b.tps.count(q) && (n_tokens + q - 1) / q >= 512) { p = q; }
    }
    const uint32_t size = p == p0 ? planned : std::min<uint32_t>(ceil, ((n_tokens + p - 1) / p + 31) / 32 * 32);
    g_ubc_bucket = bk; g_ubc_p = p; g_ubc_size = size;
    {   // said on every call, with the numbers behind it: what was measured per pass count and whether this call is a trial
        char v[16], w[220]; snprintf(v, sizeof v, "%u", size);
        std::string meas;
        for (auto & kv : b.tps) { char t[40]; snprintf(t, sizeof t, "%s%up %.0f t/s", meas.empty() ? "" : ", ", kv.first, kv.second); meas += t; }
        snprintf(w, sizeof w, "%u passes for %u tokens (planner: %u)%s%s", p, n_tokens, p0, p != best || !b.tps.count(p) ? ", trying" : ", best", meas.empty() ? "" : (" | " + meas).c_str());
        g_tok_mu.unlock(); llama_moe_decision_fork("ubatch", v, w); g_tok_mu.lock();
    }
    return size;
}
// a prompt call (more than one token) starts: tokens and the pass size it runs with, so GET /moe/stats shows the running call, not the finished one before it
static uint32_t g_live_tokens = 0, g_live_ub = 0, g_live_done = 0, g_live_pass = 0, g_live_passes = 0;   // pass = passes finished so far, passes = how many the call needs at its pass size (run.py shows "pass 2/3")
static int64_t  g_live_t0 = 0, g_live_tp = 0;
static int64_t g_live_a0 = 0, g_live_f0 = 0, g_live_w0 = 0;
static int64_t g_live_c0[4][3] = {};   // per card (link): asked / found / wrong at the end of the previous pass   // the cache's prompt hit counters at the end of the previous pass: a pass's hit rate is the difference
void llama_moe_cache_fill_fork(uint64_t * filled, uint64_t * total);   // src/llama-moecache-fork.cpp
bool llama_moe_cache_staging_counts_fork(int64_t * asked, int64_t * found, int64_t * wrong);
bool llama_moe_cache_staging_card_fork(int k, const char ** name, int64_t * asked, int64_t * found, int64_t * wrong);   // src/llama-moecache-fork.cpp   // src/llama-moecache-fork.cpp
static tok_snap g_pass_snap;   // the counters at the end of the previous pass of the running prompt call (or at its start)
void llama_moe_prompt_begin_fork(uint32_t n_tokens, uint32_t ubatch) {
    g_pass_snap = tok_snap_now();
    if (n_tokens > 1) {   // (before g_tok_mu: the cache lock is taken inside, the other code takes it first)
        uint64_t filled, total;
        llama_moe_cache_fill_fork(&filled, &total);
        if (total) { LLAMA_LOG_WARN("%s: prompt call of %u tokens starts with %llu of %llu cache blocks holding an expert (%.0f%%)\n", __func__, n_tokens, (unsigned long long) filled, (unsigned long long) total, 100.0 * filled / total); }
    }
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_live_tokens = n_tokens; g_live_ub = ubatch; g_live_done = 0; g_live_t0 = g_live_tp = ggml_time_us();
    g_live_pass = 0; g_live_passes = ubatch > 0 ? (n_tokens + ubatch - 1) / ubatch : 1;
    if (!llama_moe_cache_staging_counts_fork(&g_live_a0, &g_live_f0, &g_live_w0)) { g_live_a0 = g_live_f0 = g_live_w0 = 0; }
    for (int k = 0; k < 4; ++k) {
        const char * nm; int64_t ca, cf, cw;
        if (!llama_moe_cache_staging_card_fork(k, &nm, &ca, &cf, &cw)) { break; }
        g_live_c0[k][0] = ca; g_live_c0[k][1] = cf; g_live_c0[k][2] = cw;
    }
}
// one pass of the running prompt call finished: tokens done so far
// (one log line per pass, so a run is followed pass by pass and the time of every pass is in the server log)
void llama_moe_prompt_progress_fork(uint32_t done) {
    uint32_t pass, passes, prev, tokens;
    double s_all, s_pass;
    int64_t d_asked = 0, d_found = 0, d_wrong = 0;
    std::string per_card;   // " | CUDA1: 12% of 90000 found, 3% other card; CUDA0: ..." the pass's lookups card by card
    {
        std::lock_guard<std::mutex> lk(g_tok_mu);
        const int64_t now = ggml_time_us();
        prev = g_live_done; tokens = g_live_tokens;
        g_live_done = done; ++g_live_pass;
        pass = g_live_pass; passes = std::max(g_live_passes, g_live_pass);
        s_all = (now - g_live_t0) / 1e6; s_pass = (now - g_live_tp) / 1e6; g_live_tp = now;
        int64_t a, f, w;
        if (llama_moe_cache_staging_counts_fork(&a, &f, &w)) {
            d_asked = a - g_live_a0; d_found = f - g_live_f0; d_wrong = w - g_live_w0;
            g_live_a0 = a; g_live_f0 = f; g_live_w0 = w;
        }
        for (int k = 0; k < 4; ++k) {
            const char * nm; int64_t ca, cf, cw;
            if (!llama_moe_cache_staging_card_fork(k, &nm, &ca, &cf, &cw)) { break; }
            const int64_t da = ca - g_live_c0[k][0], df = cf - g_live_c0[k][1], dw = cw - g_live_c0[k][2];
            g_live_c0[k][0] = ca; g_live_c0[k][1] = cf; g_live_c0[k][2] = cw;
            if (da > 0) { char b[120]; snprintf(b, sizeof b, "%s %s: %.0f%% of %lld found, %.0f%% other card", per_card.empty() ? "" : ";", nm, 100.0 * df / da, (long long) da, 100.0 * dw / da); per_card += b; }
        }
    }
    if (tokens) {
        char hs2[400] = "";   // of the experts this pass asked for: found on the card it needs them on (copied there, upload skipped) / asked of the other card (its layer's cache is there) / the rest comes from RAM
        if (d_asked > 0) { snprintf(hs2, sizeof hs2, ", experts from cache %.0f%% (%lld of %lld asked; %.0f%% belong to the other card's cache, the rest uploaded)", 100.0 * d_found / d_asked, (long long) d_found, (long long) d_asked, 100.0 * d_wrong / d_asked); }
        if (!per_card.empty()) { snprintf(hs2 + strlen(hs2), sizeof hs2 - strlen(hs2), " [%s]", per_card.c_str()); }
        const std::string pass_line = format("prompt pass %u/%u: %u tokens in %.1f s (%.0f t/s)%s, %u of %u done after %.1f s", pass, passes,
                done - prev, s_pass, s_pass > 0 ? (done - prev) / s_pass : 0.0, hs2, done, tokens, s_all);
        LLAMA_LOG_WARN("%s: %s\n", __func__, pass_line.c_str());
        const tok_snap pass_end = tok_snap_now();
        llama_moe_stats_log_fork("PAS", (pass_line + format(", %.2f ms per token; ", done > prev ? s_pass * 1e3 / (done - prev) : 0.0) + tok_per_token(g_pass_snap, pass_end, done - prev)).c_str());
        g_pass_snap = pass_end;
    }
}
// a prompt call (more than one token) finished: tokens, milliseconds
void llama_moe_prompt_done_fork(uint32_t n_tokens, double ms) {
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_live_tokens = 0;
    if (g_ubc_bucket < 0 && !g_one_why.empty() && ms > 0) {   // the one-pass decision shows what its last call did, so the line moves with every prompt
        char t[96]; snprintf(t, sizeof t, " | last call %u tokens at %.0f t/s", n_tokens, n_tokens / (ms / 1e3));
        for (auto & d : g_decisions) { if (d.name == "ubatch") { d.why = g_one_why + t; d.t = (int64_t) time(nullptr); } }
    }
    ++g_pp_calls; g_pp_tokens = n_tokens; g_pp_ms = ms; g_pp_ub = g_ubc_bucket >= 0 ? g_ubc_size : g_ub_cur;
    if (g_ubc_bucket >= 0 && ms > 0) {   // cache path: tokens/s of this call for the pass count tried
        const double tps = n_tokens / (ms / 1e3);
        double & e = g_ubc[g_ubc_bucket].tps[g_ubc_p];
        e = e > 0 ? 0.5 * e + 0.5 * tps : tps;
        g_ub_tps[g_ubc_size] = e;
        g_ubc_bucket = -1;
    }
    if (g_ub_cur && ms > 0 && n_tokens >= 2 * g_ub_cur) {   // at least 2 passes of the size: a call with a partial tail is not comparable
        const double tps = n_tokens / (ms / 1e3);
        double & e = g_ub_tps[g_ub_cur];
        e = e > 0 ? 0.5 * e + 0.5 * tps : tps;
    }
}
int llama_moe_stats_json_fork(char * buf, int cap) {
    char b[512];
    std::string o;
    {
        std::lock_guard<std::mutex> lk(g_tok_mu);
        const unsigned long long n = g_tok_n, k = std::min<unsigned long long>(n, 64);
        double w = 0, last = 0;
        for (unsigned long long i = 0; i < k; ++i) { w += g_tok_ring[i]; }
        if (k) { w /= k; last = g_tok_ring[(n - 1) % 64]; }
        snprintf(b, sizeof b, "{\"token_ms\":{\"tokens\":%llu,\"last\":%.3f,\"last64\":%.3f,\"all\":%.3f},", n, last, w, n ? g_tok_sum_ms / n : 0.0);
        o = b;
        if (g_live_tokens) { snprintf(b, sizeof b, "\"live\":{\"tokens\":%u,\"ubatch\":%u,\"done\":%u,\"pass\":%u,\"passes\":%u,\"s\":%.1f},", g_live_tokens, g_live_ub, g_live_done, g_live_pass, g_live_passes, (ggml_time_us() - g_live_t0) / 1e6); o += b; }
        snprintf(b, sizeof b, "\"prompt\":{\"calls\":%llu,\"last_tokens\":%u,\"last_ms\":%.1f,\"last_tps\":%.1f,\"ubatch\":%u,\"ceiling\":%u,\"tps_by_ubatch\":{",
                 g_pp_calls, g_pp_tokens, g_pp_ms, g_pp_ms > 0 ? g_pp_tokens / (g_pp_ms / 1e3) : 0.0, g_pp_ub, g_ub_ceil);
        o += b;
        bool first = true;
        for (auto & kv : g_ub_tps) { snprintf(b, sizeof b, "%s\"%u\":%.1f", first ? "" : ",", kv.first, kv.second); o += b; first = false; }
        o += "}},\"decisions\":[";
        for (size_t i = 0; i < g_decisions.size(); ++i) {
            std::string v = g_decisions[i].value, y = g_decisions[i].why;
            for (char & c : v) { if (c == '"' || c == '\\' || c < ' ') { c = ' '; } }
            for (char & c : y) { if (c == '"' || c == '\\' || c < ' ') { c = ' '; } }
            o += (i ? ",{\"name\":\"" : "{\"name\":\"") + g_decisions[i].name + "\",\"value\":\"" + v + "\",\"why\":\"" + y + "\",\"t\":" + std::to_string(g_decisions[i].t) + "}";
        }
        o += "],";
    }
    o += llama_moe_bw_json_fork() + "," + llama_dsplit_state_json_fork() + "}";
    const int len = (int) o.size();
    if (cap <= 0) { return len; }
    const int m = std::min(len, cap - 1);
    memcpy(buf, o.data(), m); buf[m] = 0;
    return m;
}

const char * moe_opt(const char * key) {
    return llama_moe_get_opt(key);
}

// the stem of the research files of one run: model file name without .gguf and split suffix, --moe tag=NAME, start time; the same for every log of the process
std::string moe_run_stem(const std::string & section) {
    static std::string stem;
    if (!stem.empty()) {
        return stem;
    }
    std::string f = section.substr(0, section.find(' '));   // the section is "<file name> <size>"
    for (char & c : f) { if (c == ',' || c == '\n') { c = '_'; } }
    if (f.size() > 5 && f.compare(f.size() - 5, 5, ".gguf") == 0) { f.resize(f.size() - 5); }
    if (const size_t sp = f.find("-0000"); sp != std::string::npos && f.find("-of-", sp) != std::string::npos) { f.resize(sp); }
    for (char & c : f) { if (!isalnum((unsigned char) c) && c != '.' && c != '-' && c != '_') { c = '_'; } }
    char ts[32];
    const time_t now = time(nullptr);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", &tmv);
    std::string tag = moe_opt("tag") ? moe_opt("tag") : "";
    for (char & c : tag) { if (!isalnum((unsigned char) c) && c != '-' && c != '_') { c = '_'; } }
    stem = f + (tag.empty() ? std::string() : "." + tag) + "." + ts;
    return stem;
}

// where a research log goes: the --moe logdir=DIR directory, else the working directory
std::string moe_log_file(const char * name) {
    const char * d = moe_opt("logdir");
    return d && d[0] ? std::string(d) + "/" + name : std::string(name);
}
