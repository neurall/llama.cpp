// Checks that name a graph tensor that lives too long, before it costs a night (2026-11-14: a dense split probe reserved 58 GiB on CUDA0 for a 7-8 GiB graph:
// every expert layer's card chain sat in the graph with no reader and was never freed; prompt passes were capped at 3776 tokens).
//  1. ggml's graph planner (ggml-alloc.c) names every non-output tensor still allocated at a graph's end: "galloc: UNREAD" lines, always on, at re-plans only.
//     run.py fails the row on them.
//  2. llama_graph_jump_fork (src/llama-graphcheck-fork.h): a pass-size probe that asks for far more per token than the context's first reserve.
// build: cmake --build <dir> --target test-graphcheck-fork;  run: ctest -R test-graphcheck-fork   (no GPU, no model)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "../src/llama-graphcheck-fork.h"

#include <cstdio>
#include <string>

static std::string g_log;
static void log_cb(ggml_log_level, const char * text, void *) { g_log += text; }

static int g_fails = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum leak { NONE, UNREAD, UNREAD_VIEW, SMALL, SELECT };

// x -> a = x*x -> out = a + x (32 MiB each); UNREAD adds b = x + x that no node reads, UNREAD_VIEW a view of c = x*x that no node reads (it keeps c to the end),
// SMALL an unread tensor of 4 MiB (the fork's hook nodes reach a MiB on big passes: under 16 MiB a tensor is left alone), SELECT the input branch
// ggml_build_forward_select leaves in the graph without computing it (llama-graph.cpp build_inp_embd: the mixed-input copy, 256 MiB at 16384 tokens on MiMo)
static std::string plan(leak k) {
    ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8 << 20);
    ggml_set_input(x);
    ggml_tensor * a = ggml_mul(ctx, x, x);
    ggml_set_name(a, "read_once");
    ggml_tensor * out = ggml_add(ctx, a, x);
    ggml_set_name(out, "the_output");
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    if (k == UNREAD) {
        ggml_tensor * b = ggml_add(ctx, x, x);
        ggml_set_name(b, "never_read");
        ggml_build_forward_expand(gf, b);
    }
    if (k == UNREAD_VIEW) {
        ggml_tensor * c = ggml_mul(ctx, x, x);
        ggml_set_name(c, "kept_by_view");
        ggml_tensor * v = ggml_view_1d(ctx, c, 1024, 0);
        ggml_set_name(v, "unread_view");
        ggml_build_forward_expand(gf, v);
    }
    if (k == SMALL) {
        ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1 << 20);
        ggml_set_input(s);
        ggml_tensor * b = ggml_add(ctx, s, s);
        ggml_set_name(b, "small_unread");
        ggml_build_forward_expand(gf, b);
    }
    if (k == SELECT) {
        ggml_tensor * x2 = ggml_reshape_2d(ctx, x, 4096, 2048);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1), * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_set_input(rows); ggml_set_input(idx);
        ggml_tensor * alt[2] = { ggml_mul(ctx, x2, x2), ggml_set_rows(ctx, ggml_dup(ctx, x2), rows, idx) };
        ggml_set_name(alt[0], "selected");
        ggml_set_name(alt[1]->view_src ? alt[1]->view_src : alt[1], "not_selected");
        ggml_tensor * sel = ggml_build_forward_select(gf, alt, 2, 0);
        ggml_tensor * o2 = ggml_add(ctx, sel, x2);
        ggml_set_output(o2);
        ggml_build_forward_expand(gf, o2);
    }
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    g_log.clear();
    ggml_log_set(log_cb, nullptr);
    const bool ok = ggml_gallocr_reserve(ga, gf);
    ggml_log_set(nullptr, nullptr);
    CHECK(ok, "reserve failed");
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return g_log;
}

static bool has(const std::string & s, const char * t) { return s.find(t) != std::string::npos; }

int main() {
    {
        const std::string l = plan(NONE);
        CHECK(!has(l, "galloc: UNREAD"), "a clean graph reported: %s", l.c_str());
    }
    {
        const std::string l = plan(UNREAD);
        CHECK(has(l, "galloc: UNREAD"), "an unread 32 MiB tensor not reported: [%s]", l.c_str());
        CHECK(has(l, "never_read"), "the unread tensor not named: [%s]", l.c_str());
        CHECK(!has(l, "read_once") && !has(l, "the_output"), "a read tensor or an output named: [%s]", l.c_str());
    }
    {
        const std::string l = plan(UNREAD_VIEW);
        CHECK(has(l, "galloc: UNREAD") && has(l, "kept_by_view") && has(l, "unread_view"), "a source kept by an unread view not named with its view: [%s]", l.c_str());
    }
    {
        const std::string l = plan(SELECT);
        CHECK(!has(l, "galloc: UNREAD"), "a branch ggml_build_forward_select does not compute reported: [%s]", l.c_str());
    }
    {
        const std::string l = plan(SMALL);
        CHECK(!has(l, "galloc: UNREAD"), "a 4 MiB unread tensor reported: [%s]", l.c_str());
    }

    // llama_graph_jump_fork, from the 2026-11-14 GLM two-card log: first reserve 8201 MiB (CUDA0) and 7218 MiB (CUDA1) at 16384 tokens
    const size_t MiB = (size_t) 1 << 20;
    CHECK( llama_graph_jump_fork(8201 * MiB, 16384, 58347 * MiB, 16384), "the 58 GiB probe at 16384 tokens");
    CHECK( llama_graph_jump_fork(7218 * MiB, 16384, 47251 * MiB, 16384), "CUDA1's 47 GiB probe");
    CHECK( llama_graph_jump_fork(8201 * MiB, 16384, 13912 * MiB, 3776),  "the probe at the cap it set, 3776 tokens");
    CHECK(!llama_graph_jump_fork(8201 * MiB, 16384,  7000 * MiB, 16384), "an explicit staging=0 measured 7.0 GiB at 16384");
    CHECK(!llama_graph_jump_fork(8201 * MiB, 16384,  2100 * MiB, 3776),  "a clean pass of 3776 tokens");
    CHECK(!llama_graph_jump_fork(8201 * MiB, 16384,   600 * MiB, 512),   "a small pass with its fixed part");
    CHECK(!llama_graph_jump_fork(0, 0, 58347 * MiB, 16384),              "no first reserve: no verdict");

    if (g_fails) {
        fprintf(stderr, "test-graphcheck-fork: %d failed\n", g_fails);
        return 1;
    }
    printf("test-graphcheck-fork: ok\n");
    return 0;
}
