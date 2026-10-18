// "--fork off" is the full fallback to stock behaviour: no expert cache, no self-tuning, no auto-pinned weights, no thread tuner, whatever other fork options say and in
// whatever order they come. These tests parse real command lines (the fork's own option parser) and apply the same step common_init_from_params runs.
// A model-level check (the output of --fork off against upstream stock) needs a model and runs as a run.py cell.
//
// build: cmake --build <dir> --target test-fork-off;  run: ctest -R test-fork-off
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_failed = 0;
static void report(const char * what, bool ok) {
    printf("  %-70s %s\n", what, ok ? "OK" : "FAILED");
    if (!ok) { n_failed++; }
}

static bool parse(std::vector<std::string> args, common_params & p) {
    args.insert(args.begin(), "prog");
    args.push_back("-m"); args.push_back("model.gguf");   // the parser requires a model; nothing is loaded
    std::vector<char *> v;
    for (auto & a : args) { v.push_back(a.data()); }
    return common_params_parse((int) v.size(), v.data(), p, LLAMA_EXAMPLE_COMMON);
}

int main() {
    printf("Testing --fork off (full fallback to stock):\n");
    {   // the fork on (default): the other options are kept
        common_params p;
        const bool ok = parse({ "--moe-expert-cache", "64", "-at", "on" }, p);
        common_fork_off_apply(p);
        report("default: the command line parses", ok);
        report("default: cache slots and autotune stay as given", p.n_moe_cache_slots == 64 && p.autotune && p.auto_pin && !p.fork_off);
    }
    const std::vector<std::vector<std::string>> orders = {
        { "--fork", "off", "--moe-expert-cache", "64", "-at", "on" },
        { "--moe-expert-cache", "64", "-at", "on", "--fork", "off" },
        { "-at", "on", "--fork", "off", "--moe-expert-cache", "-1" },
        { "--moe", "slots=128", "--fork", "off" },
        { "--fork", "off", "--moe", "slots=128" },
        { "--fork", "0", "--moe-expert-cache", "32" },
    };
    int i = 0;
    for (const auto & o : orders) {
        common_params p;
        const bool ok = parse(o, p);
        common_fork_off_apply(p);
        std::string name = "order " + std::to_string(i++) + ": cache slots 0, autotune off, auto-pin off";
        report(name.c_str(), ok && p.fork_off && p.n_moe_cache_slots == 0 && !p.autotune && !p.auto_pin);
    }
    {   // the engine-side switches the option sets at parse time
        report("--fork off sets the engine's fork-off and thread-tune options", llama_moe_get_opt("fork-off") && strcmp(llama_moe_get_opt("fork-off"), "1") == 0 &&
                                                                                 llama_moe_get_opt("thread-tune") && strcmp(llama_moe_get_opt("thread-tune"), "0") == 0);
    }
    {   // a bad value is refused, not read as off
        common_params p;
        report("--fork maybe is refused", !parse({ "--fork", "maybe" }, p));
    }
    {   // applying twice changes nothing more
        common_params p;
        parse({ "--fork", "off" }, p);
        common_fork_off_apply(p);
        const common_params q = p;
        common_fork_off_apply(p);
        report("applying it twice is the same as once", p.n_moe_cache_slots == q.n_moe_cache_slots && p.autotune == q.autotune && p.auto_pin == q.auto_pin);
    }
    if (n_failed > 0) { printf("%d tests failed\n", n_failed); return 1; }
    printf("All tests passed\n");
    return 0;
}
