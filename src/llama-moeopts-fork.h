// fork: the --moe value "KEY=VAL,KEY=VAL,..." split into items. A word without '=' continues the value before it, so a list is one value
// (kv-fallback=k8v8,k4v4,ram; tensor-file=A.gguf,B.gguf). The one rule of both parsers: common/arg.cpp (each key set at parse time) and
// llama_moe_set_options (the whole string again before the context is made; with a splitter of its own it cut every list to its first word).
// Pure: tests/test-moeopts-fork.cpp.
#pragma once

#include <string>
#include <vector>

static inline std::vector<std::string> llama_moe_opt_items_fork(const std::string & value) {
    std::vector<std::string> items;
    for (size_t pos = 0; pos < value.size();) {
        const size_t comma = value.find(',', pos);
        const std::string tok = value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? value.size() : comma + 1;
        if (tok.find('=') == std::string::npos && !items.empty()) { items.back() += "," + tok; } else { items.push_back(tok); }
    }
    return items;
}
