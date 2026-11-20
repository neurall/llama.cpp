// --moe tensor-file (src/llama-sidefile-fork.h): which side tensors may replace the model's, the path list. Pure.
//   clang++ -std=c++17 -I src tests/test-sidefile-fork.cpp -o test-sidefile-fork && ./test-sidefile-fork
#include "llama-sidefile-fork.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>

static bool has(const std::string & s, const char * part) { return s.find(part) != std::string::npos; }

int main() {
    const int64_t q[4]  = { 7168, 12288, 1, 1 };   // K3.1 blk.N.attn_q.weight
    const int64_t qt[4] = { 12288, 7168, 1, 1 };   // the same matrix transposed
    const int64_t e3[4] = { 7168, 2048, 896, 1 };  // an expert tensor: 3 dims count too

    // same name and shape: taken, whatever the type (the side file's type and bytes are the point)
    assert(llama_side_tensor_refusal_fork("blk.0.attn_q.weight", q, q, false).empty());
    assert(llama_side_tensor_refusal_fork("blk.3.ffn_up_exps.weight", e3, e3, false).empty());

    // a name the model does not have: refused, named
    {
        const std::string why = llama_side_tensor_refusal_fork("blk.93.attn_q.weight", q, nullptr, false);
        assert(has(why, "blk.93.attn_q.weight") && has(why, "not in the model"));
    }
    // another shape: refused with both shapes
    {
        const std::string why = llama_side_tensor_refusal_fork("blk.0.attn_q.weight", qt, q, false);
        assert(has(why, "[12288, 7168, 1, 1]") && has(why, "[7168, 12288, 1, 1]"));
    }
    {
        const int64_t e3b[4] = { 7168, 2048, 448, 1 };   // half the experts: the last dimension differs only
        assert(!llama_side_tensor_refusal_fork("blk.3.ffn_up_exps.weight", e3b, e3, false).empty());
    }
    // a tensor an earlier side file replaced already: refused (which one wins would depend on the order)
    assert(has(llama_side_tensor_refusal_fork("blk.0.attn_q.weight", q, q, true), "earlier side file"));

    // the option's list: comma separated, empty items dropped
    assert(llama_side_paths_fork("").empty());
    assert((llama_side_paths_fork("/m2/k3/a.gguf") == std::vector<std::string>{ "/m2/k3/a.gguf" }));
    assert((llama_side_paths_fork("/m2/a.gguf,/m2/b.gguf,") == std::vector<std::string>{ "/m2/a.gguf", "/m2/b.gguf" }));

    printf("ok: side file checks, paths\n");
    return 0;
}
