// --moe KEY=VAL,... split into items (src/llama-moeopts-fork.h), the one rule of both parsers (common/arg.cpp, llama_moe_set_options). Pure.
//   clang++ -std=c++17 -I src tests/test-moeopts-fork.cpp -o test-moeopts-fork && ./test-moeopts-fork
#include "llama-moeopts-fork.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>

using v = std::vector<std::string>;

int main() {
    assert(llama_moe_opt_items_fork("").empty());
    assert((llama_moe_opt_items_fork("slots=4") == v{ "slots=4" }));
    assert((llama_moe_opt_items_fork("slots=4,policy=lru") == v{ "slots=4", "policy=lru" }));
    // a word without '=' continues the value before it: a list is one value
    assert((llama_moe_opt_items_fork("kv-fallback=k8v8,k4v4,ram,autotune=0") == v{ "kv-fallback=k8v8,k4v4,ram", "autotune=0" }));
    assert((llama_moe_opt_items_fork("autotune=0,tensor-file=/m2/k/side/a.gguf,/m2/k/side/b.gguf") == v{ "autotune=0", "tensor-file=/m2/k/side/a.gguf,/m2/k/side/b.gguf" }));
    // as common/arg.cpp had it: a first word without '=' stays an item of its own (arg.cpp refuses it, the engine skips it), an empty word continues the value
    assert((llama_moe_opt_items_fork("ram,slots=4") == v{ "ram", "slots=4" }));
    assert((llama_moe_opt_items_fork("slots=4,,recent=8") == v{ "slots=4,", "recent=8" }));
    assert((llama_moe_opt_items_fork(",slots=4") == v{ "", "slots=4" }));
    printf("ok: --moe items\n");
    return 0;
}
