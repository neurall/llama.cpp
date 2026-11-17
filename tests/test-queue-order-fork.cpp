// test-queue-order-fork: a backend's uploads keep a stream's order with the kernels and copies issued on that backend before and after them, on every GPU
// of every backend in the build. Each check runs a few rounds on a 64 MiB block that about 200 kernels read in turn (tens of ms of GPU time); a race shows
// as a wrong value.
//
// The race it guards (fork/docs/bug-log.md, PC1 2026-11-17): Vulkan with GGML_VK_ASYNC_USE_TRANSFER_QUEUE (default on AMD RDNA dGPUs) ran uploads on
// the transfer queue without waiting for the kernels submitted before them, so an upload overwrote an expert block while the previous pass's kernels
// still read it (garbage answers, failed cells).
//
// usage: test-queue-order-fork [--default-queue] [BACKEND_DIR]
//   without --default-queue the Vulkan transfer queue is forced on (the path that raced); BACKEND_DIR loads the backends of that build directory
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static const int64_t N_FLOATS = 16 << 20;   // the block: 64 MiB, far above any GPU cache
static const int     N_READS  = 200;        // kernels that read the block one after the other
static const int     ROUNDS   = 3;
static const float   OLD = 1.0f, NEW = 3.0f;

struct dev_case {
    ggml_backend_t be = nullptr, be2 = nullptr;   // two backends of one device (a second stream)
    ggml_tensor * blk = nullptr;                  // the block, in device memory
    ggml_tensor * src_old = nullptr, * src_new = nullptr, * host = nullptr;   // pinned host tensors: upload sources, download target
    ggml_tensor * out = nullptr;                  // blk summed N_READS + 1 times by the chain
    ggml_cgraph * chain = nullptr;
    std::vector<float> tmp;
};

static size_t nbytes() { return (size_t) N_FLOATS * sizeof(float); }

// first element of p[0..n) that differs from want: -1 when all match
static int64_t first_bad(const float * p, int64_t n, float want, int64_t * n_bad) {
    int64_t first = -1;
    *n_bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (p[i] != want) { if (first < 0) { first = i; } ++*n_bad; }
    }
    return first;
}

// the block back to OLD, the download target to zero, nothing in flight
static void reset(dev_case & c) {
    ggml_backend_synchronize(c.be);
    ggml_backend_synchronize(c.be2);
    ggml_backend_tensor_set(c.blk, c.src_old->data, 0, nbytes());
    memset(c.host->data, 0, nbytes());
}

static bool check_out(dev_case & c, float want, const char * name, int round) {
    ggml_backend_tensor_get(c.out, c.tmp.data(), 0, nbytes());
    int64_t n_bad = 0;
    const int64_t i = first_bad(c.tmp.data(), N_FLOATS, want, &n_bad);
    if (i >= 0) {
        printf("    FAIL %-46s round %d: %lld of %lld sums wrong (first at %lld: %g, want %g)\n", name, round, (long long) n_bad, (long long) N_FLOATS,
               (long long) i, c.tmp[i], want);
        return false;
    }
    return true;
}

static bool check_host(dev_case & c, float want, const char * name, int round) {
    int64_t n_bad = 0;
    const float * h = (const float *) c.host->data;
    const int64_t i = first_bad(h, N_FLOATS, want, &n_bad);
    if (i >= 0) {
        printf("    FAIL %-46s round %d: %lld of %lld downloaded values wrong (first at %lld: %g, want %g)\n", name, round, (long long) n_bad,
               (long long) N_FLOATS, (long long) i, h[i], want);
        return false;
    }
    return true;
}

static bool run_device(ggml_backend_dev_t dev, bool vulkan) {
    printf("%s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    dev_case c;
    c.be  = ggml_backend_dev_init(dev, nullptr);
    c.be2 = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev);
    if (!c.be || !c.be2 || !hbt) {
        printf("    skipped: no second backend or no pinned host buffer type\n");
        return true;
    }

    ggml_init_params wp = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    c.blk = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, N_FLOATS);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, c.be);

    ggml_init_params hp = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * hctx = ggml_init(hp);
    c.src_old = ggml_new_tensor_1d(hctx, GGML_TYPE_F32, N_FLOATS);
    c.src_new = ggml_new_tensor_1d(hctx, GGML_TYPE_F32, N_FLOATS);
    c.host    = ggml_new_tensor_1d(hctx, GGML_TYPE_F32, N_FLOATS);
    ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors_from_buft(hctx, hbt);
    if (!wbuf || !hbuf) {
        printf("    skipped: allocation failed\n");
        return true;
    }
    for (int64_t i = 0; i < N_FLOATS; ++i) { ((float *) c.src_old->data)[i] = OLD; ((float *) c.src_new->data)[i] = NEW; }
    c.tmp.resize(N_FLOATS);

    ggml_init_params gp = { ggml_tensor_overhead() * (N_READS + 8) + ggml_graph_overhead_custom(N_READS + 8, false), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    c.chain = ggml_new_graph_custom(gctx, N_READS + 8, false);
    ggml_tensor * x = c.blk;
    for (int i = 0; i < N_READS; ++i) { x = ggml_add(gctx, x, c.blk); }
    ggml_set_output(x);
    ggml_build_forward_expand(c.chain, x);
    c.out = x;
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.be));
    if (!ggml_gallocr_alloc_graph(ga, c.chain)) {
        printf("    skipped: graph allocation failed\n");
        return true;
    }

    const float sum_old = OLD * (N_READS + 1), sum_new = NEW * (N_READS + 1);
    ggml_backend_event_t ev = ggml_backend_event_new(dev);

    struct check { const char * name; bool vulkan_only; bool needs_event; std::function<bool(int)> run; };
    const std::vector<check> checks = {
        { "kernels, then upload (write after read)", false, false, [&](int r) {
            ggml_backend_graph_compute_async(c.be, c.chain);
            ggml_backend_tensor_set_async(c.be, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_synchronize(c.be);
            return check_out(c, sum_old, "kernels, then upload (write after read)", r);
        } },
        { "kernels, then copy from host (write after read)", false, false, [&](int r) {
            ggml_backend_graph_compute_async(c.be, c.chain);
            ggml_backend_tensor_copy_async(c.be, c.be, c.src_new, c.blk);
            ggml_backend_synchronize(c.be);
            return check_out(c, sum_old, "kernels, then copy from host (write after read)", r);
        } },
        { "download, then upload (write after read)", false, false, [&](int r) {
            ggml_backend_tensor_get_async(c.be, c.blk, c.host->data, 0, nbytes());
            ggml_backend_tensor_set_async(c.be, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_synchronize(c.be);
            return check_host(c, OLD, "download, then upload (write after read)", r);
        } },
        { "upload, then kernels (read after write)", false, false, [&](int r) {
            ggml_backend_tensor_set_async(c.be, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_graph_compute_async(c.be, c.chain);
            ggml_backend_synchronize(c.be);
            return check_out(c, sum_new, "upload, then kernels (read after write)", r);
        } },
        { "upload, then download (read after write)", false, false, [&](int r) {
            ggml_backend_tensor_set_async(c.be, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_tensor_get_async(c.be, c.blk, c.host->data, 0, nbytes());
            ggml_backend_synchronize(c.be);
            return check_host(c, NEW, "upload, then download (read after write)", r);
        } },
        { "kernels, event, upload on a 2nd backend", false, true, [&](int r) {
            ggml_backend_graph_compute_async(c.be, c.chain);
            ggml_backend_event_record(ev, c.be);
            ggml_backend_event_wait(c.be2, ev);
            ggml_backend_tensor_set_async(c.be2, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_synchronize(c.be2);
            ggml_backend_synchronize(c.be);
            return check_out(c, sum_old, "kernels, event, upload on a 2nd backend", r);
        } },
        // Vulkan's promise beyond a CUDA stream's: an upload never overlaps a kernel of the device submitted before it, whichever backend submitted it
        // (CUDA streams are independent; there the caller orders them with an event, the check above)
        { "kernels, then upload on a 2nd backend, no event", true, false, [&](int r) {
            ggml_backend_graph_compute_async(c.be, c.chain);
            ggml_backend_tensor_set_async(c.be2, c.blk, c.src_new->data, 0, nbytes());
            ggml_backend_synchronize(c.be2);
            ggml_backend_synchronize(c.be);
            return check_out(c, sum_old, "kernels, then upload on a 2nd backend, no event", r);
        } },
    };

    bool ok = true;
    for (const check & ck : checks) {
        if (ck.vulkan_only && !vulkan) { printf("    skip %-46s (not Vulkan)\n", ck.name); continue; }
        if (ck.needs_event && !ev)     { printf("    skip %-46s (no events)\n", ck.name); continue; }
        bool ok_ck = true;
        for (int r = 1; r <= ROUNDS; ++r) {
            reset(c);
            ok_ck = ck.run(r) && ok_ck;
        }
        if (ok_ck) { printf("    ok   %-46s (%d rounds)\n", ck.name, ROUNDS); }
        ok = ok && ok_ck;
    }

    if (ev) { ggml_backend_event_free(ev); }
    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(hbuf);
    ggml_free(hctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(c.be2);
    ggml_backend_free(c.be);
    return ok;
}

int main(int argc, char ** argv) {
    bool default_queue = false;
    const char * dir = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--default-queue") == 0) { default_queue = true; } else { dir = argv[i]; }
    }
    if (!default_queue) {   // read when a Vulkan device starts: before any backend loads
#ifdef _WIN32
        _putenv_s("GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "1");
#else
        setenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "1", 1);
#endif
    }
    if (dir) { ggml_backend_load_all_from_path(dir); } else { ggml_backend_load_all(); }
    printf("Vulkan uploads: %s\n", default_queue ? "default queue choice" : "transfer queue forced (GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1)");

    bool ok = true;
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        ++n_gpu;
        const bool vulkan = strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Vulkan") == 0;
        ok = run_device(dev, vulkan) && ok;
    }
    if (n_gpu == 0) { printf("no GPU: nothing to check\n"); }
    printf("%s\n", ok ? "all ok" : "FAILED");
    return ok ? 0 : 1;
}
