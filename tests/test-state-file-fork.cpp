// fork: a sequence's KV state saved to a file and loaded back costs about what the same state costs in memory (llama_state_seq_get_data / set_data) plus the
// file's bytes. In the token-major KV layout (--moe kv-token-major=1, the default whenever the experts stream from RAM) a state is one entry per cell, layer and
// K/V; the file writer and reader moved each entry with a blocking ggml_backend_tensor_get / set of its own, on every backend (2026-11-17), while the in-memory
// path batches them (ggml_backend_tensor_get_n / set_n). Needs a model, so it is built but not run by ctest:
//   test-state-file-fork MODEL.gguf [DEVICE] [N_TOKENS]      OLMoE-1B-7B: 16 layers, 4096 tokens = 131072 entries; DEVICE e.g. CUDA0 or Vulkan0
#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf [DEVICE] [N_TOKENS]: skipped, no model given\n", argv[0]);
        return 0;
    }
    const char * dev_name = argc > 2 ? argv[2] : nullptr;
    const int    n_tok    = argc > 3 ? atoi(argv[3]) : 4096;

    llama_moe_set_opt("kv-token-major", "1");   // the layout whose state is one entry per cell
    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;   // the KV on the GPU: the case under test
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (dev_name) {
        devs[0] = ggml_backend_dev_by_name(dev_name);
        if (!devs[0]) {
            fprintf(stderr, "no device %s\n", dev_name);
            return 1;
        }
        mp.devices = devs;
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_tok + 256;
    cp.n_batch         = 2048;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // the token-major layout needs V not transposed
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    std::mt19937 rng(1234);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> toks(n_tok);
    for (auto & t : toks) { t = (llama_token) (rng() % n_vocab); }
    for (int i = 0; i < n_tok; i += (int) cp.n_batch) {
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, std::min<int>((int) cp.n_batch, n_tok - i))) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
    }

    const size_t size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> want(size), from_file(size), from_mem(size);
    auto t0 = std::chrono::steady_clock::now();
    const bool got = llama_state_seq_get_data(ctx, want.data(), size, 0) == size;
    const double t_get = ms_since(t0);

    const char * path = "test-state-file-fork.tmp.bin";
    t0 = std::chrono::steady_clock::now();
    const size_t n_saved = llama_state_seq_save_file(ctx, path, 0, toks.data(), toks.size());
    const double t_save = ms_since(t0);

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);
    std::vector<llama_token> toks_out(n_tok);
    size_t n_out = 0;
    t0 = std::chrono::steady_clock::now();
    const size_t n_read = llama_state_seq_load_file(ctx, path, 0, toks_out.data(), toks_out.size(), &n_out);
    const double t_load = ms_since(t0);
    llama_state_seq_get_data(ctx, from_file.data(), size, 0);
    remove(path);

    llama_memory_seq_rm(mem, 0, -1, -1);
    t0 = std::chrono::steady_clock::now();
    const bool set = llama_state_seq_set_data(ctx, want.data(), size, 0) == size;
    const double t_set = ms_since(t0);
    llama_state_seq_get_data(ctx, from_mem.data(), size, 0);
    const bool same_mem_trip = from_mem == want;   // from_mem is reused below

    // the same bytes written and read as one block: what the file adds to the in-memory path
    t0 = std::chrono::steady_clock::now();
    FILE * f = fopen(path, "wb");
    const bool wrote = f && fwrite(want.data(), 1, size, f) == size;
    if (f) { fclose(f); }
    const double t_fwrite = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    f = fopen(path, "rb");
    const bool read = f && fread(from_mem.data(), 1, size, f) == size;
    if (f) { fclose(f); }
    const double t_fread = ms_since(t0);
    remove(path);

    // the device's own rate: one contiguous tensor of the state's size read and written in one piece. The state in memory is the same bytes in strided runs
    // (one per layer and K/V); CUDA moved them one cudaMemcpyAsync per entry (set) or through its gather kernel (get) at 1.8-2.3 GB/s (2026-11-17)
    ggml_backend_dev_t dev = devs[0] ? devs[0] : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_tensor * raw = ggml_new_tensor_1d(gctx, GGML_TYPE_I8, (int64_t) size);
    ggml_backend_buffer_t raw_buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_dev_buffer_type(dev));
    t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(raw, from_mem.data(), 0, size);
    const double t_raw_get = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(raw, want.data(), 0, size);
    const double t_raw_set = ms_since(t0);
    ggml_backend_buffer_free(raw_buf);
    ggml_free(gctx);

    const bool same_file = n_saved > 0 && n_read == n_saved && n_out == (size_t) n_tok && toks_out == toks && from_file == want;
    const bool same_mem  = got && set && wrote && read && same_mem_trip;
    const double lim_save = 1.5*(t_get + t_fwrite) + 100, lim_load = 1.5*(t_fread + t_set) + 100;
    const double lim_get = 2*t_raw_get + 50, lim_set = 2*t_raw_set + 50;
    const bool ok = same_file && same_mem && t_save <= lim_save && t_load <= lim_load && t_get <= lim_get && t_set <= lim_set;
    printf("state of %d cells x %d layers x K/V (%.0f MiB, %s): one contiguous copy get %.1f ms, set %.1f ms (%s); in memory get %.1f ms (limit %.0f), "
           "set %.1f ms (limit %.0f); plain file write %.1f ms, read %.1f ms; file save %.1f ms (limit %.0f), load %.1f ms (limit %.0f)%s%s\n",
           n_tok, llama_model_n_layer(model), size / 1048576.0, dev_name ? dev_name : "default devices", t_raw_get, t_raw_set, ggml_backend_dev_name(dev),
           t_get, lim_get, t_set, lim_set, t_fwrite, t_fread, t_save, lim_save, t_load, lim_load,
           same_file ? "" : "; FILE ROUND TRIP DIFFERS", same_mem ? "" : "; MEMORY ROUND TRIP DIFFERS");
    printf("test-state-file-fork: %s\n", ok ? "OK" : "FAILED");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
