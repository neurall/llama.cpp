// test-sched-mixed-fork: a graph split between GPUs of two different backends (CUDA and Vulkan in one process) computes the right values.
// The scheduler copies a tensor from one card to the other. With no copy events (one input copy, as with experts in host memory) the fork orders that copy on
// the GPUs: the source card's stream waits for an event at the tail of the destination's stream (ggml_backend_sched_copy_input). An event belongs to its
// backend; a CUDA stream cannot wait on a Vulkan event.
// The failure it guards (fork/docs/bug-log.md, PC1 2026-11-18, b15807): OLMoE perplexity with CUDA_VISIBLE_DEVICES=1 on the one binary got the hidden card
// through Vulkan, split CUDA0 + Vulkan0 and aborted: "CUDA error: invalid device context" in ggml_backend_cuda_event_wait.
//
// usage: test-sched-mixed-fork [BACKEND_DIR]
//   BACKEND_DIR loads the backends of that build directory (one with both the CUDA and the Vulkan module); no two GPUs of different backends: skipped
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static const int64_t N = 1 << 20;

int main(int argc, char ** argv) {
    if (argc > 1) {
        ggml_backend_load_all_from_path(argv[1]);
    } else {
        ggml_backend_load_all();
    }
    ggml_backend_dev_t da = nullptr, db = nullptr;   // the first GPU, then the first GPU of another backend
    for (size_t i = 0; i < ggml_backend_dev_count() && !db; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (!da) {
            da = d;
        } else if (ggml_backend_dev_backend_reg(d) != ggml_backend_dev_backend_reg(da)) {
            db = d;
        }
    }
    if (!db) {
        printf("no two GPUs of different backends: skipped\n");
        return 0;
    }
    printf("split: %s -> %s -> %s\n", ggml_backend_dev_name(da), ggml_backend_dev_name(db), ggml_backend_dev_name(da));

    ggml_backend_t a = ggml_backend_dev_init(da, nullptr), b = ggml_backend_dev_init(db, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[3] = { a, b, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, 64, false, true);   // not parallel: one input copy, no copy events

    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_input(x);
    ggml_tensor * y = ggml_scale(ctx, x, 2.0f);   // on a
    ggml_tensor * z = ggml_scale(ctx, y, 5.0f);   // on b: y crosses a -> b
    ggml_tensor * w = ggml_scale(ctx, z, 3.0f);   // on a: z crosses b -> a
    ggml_set_output(w);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, w);
    ggml_backend_sched_set_tensor_backend(sched, y, a);
    ggml_backend_sched_set_tensor_backend(sched, z, b);
    ggml_backend_sched_set_tensor_backend(sched, w, a);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("FAIL: the graph does not allocate\n");
        return 1;
    }

    std::vector<float> in(N), out(N);
    for (int64_t i = 0; i < N; ++i) {
        in[i] = (float) (i % 1000);
    }
    int bad = 0;
    for (int round = 0; round < 3 && !bad; ++round) {
        ggml_backend_tensor_set(x, in.data(), 0, ggml_nbytes(x));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            printf("FAIL: compute, round %d\n", round);
            return 1;
        }
        ggml_backend_tensor_get(w, out.data(), 0, ggml_nbytes(w));
        for (int64_t i = 0; i < N; ++i) {
            if (out[i] != 30.0f * in[i]) {
                if (bad++ == 0) {
                    printf("FAIL: round %d, element %lld is %g, want %g\n", round, (long long) i, out[i], 30.0f * in[i]);
                }
            }
        }
    }
    printf("%d splits, %s\n", ggml_backend_sched_get_n_splits(sched), bad ? "FAIL" : "ok");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(b);
    ggml_backend_free(a);
    return bad ? 1 : 0;
}
