#pragma once

// Routing of the learned state over two files (src/llama-moestate-fork.cpp): the heat of a model (the hot.<layer> keys) depends on the weights and the prompts only, so
// every machine can share one file of it (--moe heat=FILE, LLAMA_MOE_HEAT); everything else the engine learns (placement, tuned values, run records, probes) depends on
// the hardware and stays in the file of this machine (--moe state=FILE, run.py puts it in fork/cfg/<hardware key>/). With no heat file both go to the state file as before.

#include <string>
#include <utility>
#include <vector>

using moe_state_kv = std::vector<std::pair<std::string, std::string>>;

inline bool moe_state_is_heat_key(const std::string & key) {
    return key.compare(0, 4, "hot.") == 0;
}

// one write of mixed keys in two: the heat keys and the rest, each in its order
inline void moe_state_split(const moe_state_kv & in, moe_state_kv & heat, moe_state_kv & rest) {
    heat.clear();
    rest.clear();
    for (const auto & p : in) {
        (moe_state_is_heat_key(p.first) ? heat : rest).push_back(p);
    }
}

// one section read from both files: the keys of the hardware file, with its hot.* keys replaced by those of the heat file; a heat file with no key leaves the hardware
// file's own hot.* in place (the heat of before the split is still read until the first write)
inline moe_state_kv moe_state_merge(const moe_state_kv & hw, const moe_state_kv & heat) {
    moe_state_kv out;
    for (const auto & p : hw) {
        if (heat.empty() || !moe_state_is_heat_key(p.first)) {
            out.push_back(p);
        }
    }
    for (const auto & p : heat) {
        if (moe_state_is_heat_key(p.first)) {
            out.push_back(p);
        }
    }
    return out;
}
