#include "llama-moestate.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "../include/llama.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

namespace {

using section_t = std::map<std::string, std::string>;
using file_t    = std::map<std::string, section_t>;

std::mutex g_mtx;
bool g_suppress = false; // the state of this run is marked for deletion: no writes until the process ends
std::string g_model_section;

std::string state_path() {
    const char * off = getenv("LLAMA_MOE_CACHE_PROFILE"); // older switch: 0 off (a path there was the old profile file, now unused)
    if (off && strcmp(off, "0") == 0) {
        return "";
    }
    if (const char * e = getenv("LLAMA_MOE_STATE")) {
        return strcmp(e, "0") == 0 ? "" : e;
    }
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
    return dir + "/moe-state.ini";
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
    if (!g_model_section.empty()) {
        return g_model_section;
    }
    std::string name = model.name;
    for (char & c : name) {
        if (c == '[' || c == ']' || c == '\n' || c == '\r') { c = '_'; }
    }
    return name + " " + std::to_string(model.size());
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
