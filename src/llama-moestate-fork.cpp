#include "llama-moestate-fork.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "../include/llama.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>
#include <cctype>
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
    // sim-moe-repeat=N / sim-dense-repeat=N: run prompt-sized expert / dense matmuls (and attention) N times in the CUDA backend, to give a small model the compute-to-upload
    // ratio of a big one in scheduler tests (OLMoE standing in for GLM). The backend reads GGML_CUDA_MMID_REPEAT / GGML_CUDA_DENSE_REPEAT at its first prompt graph.
    if (value && strcmp(key, "sim-moe-repeat") == 0)   { setenv("GGML_CUDA_MMID_REPEAT", value, 1); }
    if (value && strcmp(key, "sim-dense-repeat") == 0) { setenv("GGML_CUDA_DENSE_REPEAT", value, 1); }
    // pass-stats=1 (off by default; debugging and measurement runs): every split pass's time per card by cause and its link bytes, measured with extra stream
    // stamps and counters in the scheduler (GGML_SCHED_PASS_STATS, read at the first prompt graph); run.py turns it on unless -ns
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

void llama_moe_token_time_fork(double ms) {
    std::lock_guard<std::mutex> lk(g_tok_mu);
    g_tok_ring[g_tok_n % 64] = ms;
    g_tok_n++;
    g_tok_sum_ms += ms;
}
// the latest value and reason of each automatic decision; a repeat of the same value only refreshes the time
void llama_moe_decision_fork(const char * name, const char * value, const char * why) {
    std::lock_guard<std::mutex> lk(g_tok_mu);
    const int64_t t = (int64_t) time(nullptr);
    // the decision goes to the log when it changes (run.py calib reads the engine's pass ceiling from "this call fits the ceiling N" there; the live endpoint shows it only while the server runs)
    for (auto & d : g_decisions) {
        if (d.name == name) {
            // a changed value, or a new reason of the pass size (once per prompt call); not every reason update: some tuners rewrite theirs with live numbers each step
            if (d.value != value || (d.why != why && strcmp(name, "ubatch") == 0)) { LLAMA_LOG_WARN("%s: decision %s = %s (%s)\n", __func__, name, value, why); }
            d.value = value; d.why = why; d.t = t; return;   // the reason follows the live numbers too (a tuner updates it as it measures)
        }
    }
    LLAMA_LOG_WARN("%s: decision %s = %s (%s)\n", __func__, name, value, why);
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
void llama_moe_prompt_begin_fork(uint32_t n_tokens, uint32_t ubatch) {
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
        LLAMA_LOG_WARN("%s: prompt pass %u/%u: %u tokens in %.1f s (%.0f t/s)%s, %u of %u done after %.1f s\n", __func__, pass, passes,
                done - prev, s_pass, s_pass > 0 ? (done - prev) / s_pass : 0.0, hs2, done, tokens, s_all);
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
