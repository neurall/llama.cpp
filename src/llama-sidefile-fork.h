// fork: --moe tensor-file=PATH[,PATH]: the tensors of side GGUFs replace the model's tensors of the same name. Type and bytes come from the side file,
// the shape must be the model's; a tensor the model lacks, another shape, or a tensor two side files hold is refused. Use: K3.1's attention at another
// quant (fork/attn_side.py writes the side files) without rewriting the 580 GB model. Pure: tests/test-sidefile-fork.cpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

static inline std::string llama_side_ne_str_fork(const int64_t * ne) {
    std::string s = "[";
    for (int i = 0; i < 4; ++i) { s += std::to_string(ne[i]) + (i < 3 ? ", " : "]"); }
    return s;
}

// why side tensor `name` (shape side_ne) cannot replace the model's, "" when it can. model_ne: the model's tensor of that name, nullptr when it has none;
// taken: an earlier side file replaced it already
static inline std::string llama_side_tensor_refusal_fork(const char * name, const int64_t * side_ne, const int64_t * model_ne, bool taken) {
    if (!model_ne) { return std::string("tensor '") + name + "' is not in the model"; }
    if (taken) { return std::string("tensor '") + name + "' is in an earlier side file too"; }
    for (int i = 0; i < 4; ++i) {
        if (side_ne[i] != model_ne[i]) {
            return std::string("tensor '") + name + "' has the shape " + llama_side_ne_str_fork(side_ne) + ", the model's is " + llama_side_ne_str_fork(model_ne);
        }
    }
    return "";
}

// the option's value: paths separated by commas (common's --moe parser keeps a list in one value), empty items dropped
static inline std::vector<std::string> llama_side_paths_fork(const std::string & opt) {
    std::vector<std::string> out;
    for (size_t pos = 0; pos <= opt.size();) {
        const size_t comma = std::min(opt.find(',', pos), opt.size());
        if (comma > pos) { out.push_back(opt.substr(pos, comma - pos)); }
        pos = comma + 1;
    }
    return out;
}
