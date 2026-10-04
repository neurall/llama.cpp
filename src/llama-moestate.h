#pragma once

// What the engine learns per model, kept in ONE flat INI file (default <cache dir>/llama.cpp/moe-state.ini):
//   [<model file name> <bytes>]
//   hot.<layer> = <lifetime expert counts, space separated>   expert usage profile (the next start preloads the hottest)
//   tuned.l<links> = NAME=value NAME=value ...                 the live self-tuner's decisions
//   place.g<gpus>x<MiB>.b<build>.stock|cache = <ms/prompt token> <ms/generated token> <runs>, .decided = stock|cache
// LLAMA_MOE_STATE=<file> moves it, =0 (or the older LLAMA_MOE_CACHE_PROFILE=0) turns all of it off.

#include <string>
#include <utility>
#include <vector>

struct llama_model;

// the section of a model: set once by the tool that knows the file (common), else the GGUF name and tensor bytes
void        moe_state_set_model(const std::string & section);
std::string moe_state_section(const llama_model & model);

// the key prefix of this hardware's placement records ("place.g2.<GPU hash>.v3"), set once by the tool that knows the GPUs (common)
void               moe_state_set_place(const std::string & prefix);
const std::string & moe_state_place();

bool moe_state_enabled();

// false: no such key or the state is off
bool moe_state_get(const std::string & section, const std::string & key, std::string & value);
// every key of a section (one file read)
// any_size: no section of this exact name and size, take the one with the same file name (any case, any size) that has the most keys
bool moe_state_section_kv(const std::string & section, std::vector<std::pair<std::string, std::string>> & kv, bool any_size = false);
// read-modify-write of the whole file through a temporary file and rename; several keys in one write
bool moe_state_set(const std::string & section, const std::vector<std::pair<std::string, std::string>> & kv);
// drop every key of the section that starts with prefix
bool moe_state_erase(const std::string & section, const std::string & prefix);

// research (--moe emb=1): at exit one line is appended to ./embhot.csv (LLAMA_MOE_EMBHOT=FILE moves it): model,embhex,hots. embhex = the last layer embedding of the last
// token of the last graph, 1 byte per float (2 hex characters, absmax scaled); hots = this run's expert counts, layer:count count ...;layer:... Nothing is read or written while running
bool moe_embsnap_enabled();
void moe_embsnap_set(const float * emb, int n);
void moe_embsnap_write(const std::vector<std::pair<std::string, std::string>> & hot);
