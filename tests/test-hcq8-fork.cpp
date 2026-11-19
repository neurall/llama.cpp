// --moe hc-q8 decision (src/llama-hcq8-fork.h): pure.
//   clang++ -std=c++17 -I src tests/test-hcq8-fork.cpp -o test-hcq8-fork && ./test-hcq8-fork
#include "llama-hcq8-fork.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
int main() {
    assert(llama_hcq8_wanted_fork(true, "blk.7.hc_attn_up.weight", true, 320, false));
    assert(llama_hcq8_wanted_fork(true, "blk.47.hc_ffn_down.weight", true, 10240, false));
    assert(llama_hcq8_wanted_fork(true, "output_hc_up.weight", true, 320, false));
    assert(!llama_hcq8_wanted_fork(false, "blk.7.hc_attn_up.weight", true, 320, false));   // switch off
    assert(!llama_hcq8_wanted_fork(true, "blk.7.hc_attn_up.weight", false, 320, false));   // not BF16
    assert(!llama_hcq8_wanted_fork(true, "blk.7.hc_attn_up.weight", true, 320, true));     // host buffer
    assert(!llama_hcq8_wanted_fork(true, "blk.7.hc_attn_up.weight", true, 330, false));    // not whole blocks
    assert(!llama_hcq8_wanted_fork(true, "blk.7.hc_attn_inject.weight", true, 10240, false));
    assert(!llama_hcq8_wanted_fork(true, "blk.7.hc_attn_norm.weight", true, 2560, false));
    assert(!llama_hcq8_wanted_fork(true, "blk.7.xhc_attn_up.weight", true, 320, false));   // a longer name ending the same
    assert(!llama_hcq8_wanted_fork(true, "blk.7.ffn_gate_inp.weight", true, 2560, false));
    printf("ok: hc-q8 decision\n");
    return 0;
}
