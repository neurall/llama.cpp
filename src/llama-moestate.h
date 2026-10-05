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

// research (--moe log=e): at exit one line is appended to ./embhot.csv (LLAMA_MOE_EMBHOT=FILE moves it): model,embhex,hots. embhex = the last layer embedding of the last
// token of the last graph, 1 byte per float (2 hex characters, absmax scaled); hots = this run's expert counts, layer:count count ...;layer:... Nothing is read or written while running
bool moe_embsnap_enabled();
void moe_embsnap_set(const float * emb, int n);
void moe_embsnap_write(const std::vector<std::pair<std::string, std::string>> & hot);

// research (--moe log=s or log=i): after the run's last state write, the model's section of the state file is saved as ./state-snapshots/<date time>-<model>.ini (LLAMA_MOE_SNAP_DIR moves the directory);
// the date and time in the name keep the files unique. Nothing is read or written while running
bool moe_snap_enabled();
void moe_snap_save(const std::string & section);

// --moe log=LETTERS (LLAMA_MOE_LOG): true when one of the letters is set. e embedding line (embhot.csv), s or i state snapshot, g GPU identity line, p periodic cache stats, t routing trace, c decode-step trace; r is read by tools/run.py
bool moe_log_has(const char * letters);
// the number written right after a letter in --moe log=LETTERS (log=l64: 64), dflt when there is none; the handler of the letter uses it as its argument (l: sample every Nth token, default 1; p: stats every N steps, default 64)
int  moe_log_num(char letter, int dflt);

// the --moe KEY=VALUE settings (see llama_moe_set_opt in llama.h): NULL when the key was not given
const char * moe_opt(const char * key);
std::string moe_run_stem(const std::string & section);   // the name stem of this run's research files: <model file>[.<tag>].<YYYYmmdd-HHMMSS> (the time of the first call)
std::string moe_log_file(const char * name);   // the --moe logdir=DIR directory (default: the working directory) + name
