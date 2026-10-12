// Fork: per-GPU probe, and the prefill layer plan computed from it.
//
// Problem. With several GPUs the layers are split between the cards by free memory. That says nothing about how fast a card can take part in a
// prefill pass: a card on a narrow or shared link (x4 through the chipset, an x1 riser), a slower GPU on a fat link, or a fast GPU on x8 all behave
// differently, and a link-width threshold is wrong in both directions (a half-speed card on x16 is a bottleneck that the width does not show, a
// double-speed card on x8 is worth using although its link is half). On the test machine (RTX 3090 x16 + RTX 3090 x4 through the chipset) the default
// split cost MiMo 14% of its prompt speed against the x16 card alone, because the x16 card waited 20 to 40% of the pass for the x4 card's layers.
//
// What the probe measures, per card, before any graph or compute buffer exists (about 1 s per card, no model needed):
//   h2d_gbs       bulk host->device upload, 64 MB from pinned memory, this card alone. The expert upload of a streamed prefill is bound by this.
//   h2d_conc_gbs  the same while every card uploads at once. Cards behind one root complex or the chipset uplink share bandwidth; only this number
//                 shows it, and it is what a real prefill sees because every card streams its own layers' experts at the same time.
//   lat_us        median of 64 small (4 KB) synchronous uploads: the fixed cost of one hand-off between two layers on different cards.
//   wtps          a mul_mat over 64 MB of uploaded quantised weights (Q4_K, the MMQ path the real model uses) with a 512-token batch, in weight bytes x
//                 tokens per second. The dense part of a prefill pass is bound by this (clocks and power capping are included, the kernel runs for ms).
//
// The plan. A prefill pass walks the layers one after the other, each on the card that holds it. The experts of a layer are uploaded by the card that
// computes it, and the upload of later layers runs ahead of the compute (prefetch overlap), so:
//   chain  = sum over cards of (layers of the card x its compute time per layer) + one hand-off per card boundary   (compute is sequential)
//   link_i = layers of the card x its upload time per layer                                                         (every link works in parallel)
//   T      = max(chain, link_1, ..., link_n)
// The plan picks the subset of cards and the shares of layers that minimise T. Streamed models (experts in RAM) are link-bound, so a second link
// helps and the plan spreads the layers; models that compute from resident weights are chain-bound, so the plan keeps everything on the fastest
// card and a card that cannot pay its hand-off gets a share of 0. There is no threshold anywhere: a slow or narrow card gets the share that its
// measured upload and compute earn, a card on a bad riser gets 0. The model is approximate on purpose (uploads are assumed to overlap fully and the
// cache hit rate is ignored); it picks the starting point, a measured prefill speed decides whether it stays (see docs).
//
// Test knob: GGML_CUDA_H2D_CAP=<device>:<GB/s> (ggml-cuda.cu) caps one card's upload rate, the probe sees the cap like a real narrow link.

#include "llama-gpu-probe-fork.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace {

constexpr size_t PROBE_BYTES = 64u << 20;   // size of the bulk upload and of the quantised weights of the compute probe
constexpr int    PROBE_TOKENS = 512;        // tokens of the compute probe (a small prefill batch)
constexpr int    LAT_SAMPLES  = 64;
constexpr size_t LAT_BYTES    = 4096;

std::mutex                                                  g_mu;
std::map<ggml_backend_dev_t, llama_gpu_probe_fork_result>   g_results;

// pinned host memory when the backend offers it (what the real streamed prefill uploads from), plain memory otherwise
struct host_buf {
    ggml_backend_buffer_t  buf = nullptr;
    std::vector<uint8_t>   fallback;
    void *                 ptr = nullptr;
    explicit host_buf(ggml_backend_dev_t dev, size_t n) {
        if (ggml_backend_buffer_type_t bt = ggml_backend_dev_host_buffer_type(dev)) {
            buf = ggml_backend_buft_alloc_buffer(bt, n);
            if (buf) { ptr = ggml_backend_buffer_get_base(buf); }
        }
        if (!ptr) { fallback.resize(n); ptr = fallback.data(); }
        memset(ptr, 1, n);
    }
    ~host_buf() { if (buf) { ggml_backend_buffer_free(buf); } }
    host_buf(const host_buf &) = delete;
    host_buf & operator=(const host_buf &) = delete;
};

// a no_alloc context whose tensors live in one buffer of the device
struct dev_ctx {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    explicit dev_ctx(size_t n_tensors_extra) {
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead() + n_tensors_extra, nullptr, true };
        ctx = ggml_init(ip);
    }
    bool alloc(ggml_backend_dev_t dev) {
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        return buf != nullptr;
    }
    ~dev_ctx() { if (buf) { ggml_backend_buffer_free(buf); } if (ctx) { ggml_free(ctx); } }
    dev_ctx(const dev_ctx &) = delete;
    dev_ctx & operator=(const dev_ctx &) = delete;
};

double median(std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// bytes per microsecond = MB/s; / 1e3 = GB/s. The first copy warms the path up and is not counted.
double bulk_gbs(ggml_tensor * t, const void * src, int reps) {
    double best = 0.0;
    const size_t n = ggml_nbytes(t);
    for (int i = 0; i <= reps; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_tensor_set(t, src, 0, n);
        const double gbs = (double) n / 1e3 / (double) std::max<int64_t>(1, ggml_time_us() - t0);
        if (i > 0) { best = std::max(best, gbs); }
    }
    return best;
}

double latency_us(ggml_tensor * t, const void * src) {
    std::vector<double> v;
    for (int i = 0; i < LAT_SAMPLES + 4; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_tensor_set(t, src, 0, LAT_BYTES);
        if (i >= 4) { v.push_back((double) (ggml_time_us() - t0)); }
    }
    return median(v);
}

// a mul_mat of 64 MB of Q4_K weights with PROBE_TOKENS tokens, in weight bytes x tokens per second
double compute_wtps(ggml_backend_dev_t dev, const void * host_ptr) {
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { return 0.0; }
    const int64_t K = 4096;
    const int64_t rows = (int64_t) (PROBE_BYTES / ((size_t) K / 256 * 144)) / 32 * 32;   // Q4_K: 144 bytes per 256 weights
    double wtps = 0.0;
    {
        dev_ctx c(0);
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q4_K, K, rows);
        ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32,  K, PROBE_TOKENS);
        ggml_tensor * y = ggml_mul_mat(c.ctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(c.ctx);
        ggml_build_forward_expand(gf, y);
        if (c.alloc(dev) && ggml_nbytes(w) <= PROBE_BYTES) {
            ggml_backend_tensor_set(w, host_ptr, 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, host_ptr, 0, ggml_nbytes(x));
            std::vector<double> t_us;
            for (int i = 0; i < 6; ++i) {   // the first run builds kernels and caches: not counted
                ggml_backend_synchronize(be);
                const int64_t t0 = ggml_time_us();
                ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                if (i > 0) { t_us.push_back((double) (ggml_time_us() - t0)); }
            }
            const double t = median(t_us);
            if (t > 0) { wtps = (double) ggml_nbytes(w) * PROBE_TOKENS / (t / 1e6); }
        }
    }
    ggml_backend_free(be);
    return wtps;
}

}  // namespace

std::vector<llama_gpu_probe_fork_result> llama_gpu_probe_fork(const std::vector<ggml_backend_dev_t> & gpus) {
    std::vector<llama_gpu_probe_fork_result> out(gpus.size());
    // per card, alone: upload, latency, compute
    for (size_t i = 0; i < gpus.size(); ++i) {
        out[i].dev  = gpus[i];
        out[i].name = ggml_backend_dev_name(gpus[i]);
        host_buf h(gpus[i], PROBE_BYTES);
        dev_ctx c(0);
        ggml_tensor * big   = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) PROBE_BYTES);
        ggml_tensor * small = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) LAT_BYTES);
        if (c.alloc(gpus[i])) {
            out[i].h2d_gbs = bulk_gbs(big, h.ptr, 3);
            out[i].lat_us  = latency_us(small, h.ptr);
        }
        out[i].wtps = compute_wtps(gpus[i], h.ptr);
    }
    // all cards at once: every thread uploads in a loop after a common start, bandwidth = its bytes over its own time
    if (gpus.size() > 1) {
        std::atomic<bool> go{false};
        std::vector<std::thread> th;
        for (size_t i = 0; i < gpus.size(); ++i) {
            th.emplace_back([&, i] {
                host_buf h(gpus[i], PROBE_BYTES);
                dev_ctx c(0);
                ggml_tensor * big = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) PROBE_BYTES);
                if (!c.alloc(gpus[i])) { return; }
                ggml_backend_tensor_set(big, h.ptr, 0, PROBE_BYTES);   // warm-up
                while (!go.load()) { std::this_thread::yield(); }
                const int reps = 6;
                const int64_t t0 = ggml_time_us();
                for (int r = 0; r < reps; ++r) { ggml_backend_tensor_set(big, h.ptr, 0, PROBE_BYTES); }
                out[i].h2d_conc_gbs = (double) PROBE_BYTES * reps / 1e3 / (double) std::max<int64_t>(1, ggml_time_us() - t0);
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));   // let every thread reach the barrier
        go.store(true);
        for (auto & t : th) { t.join(); }
    } else if (!out.empty()) {
        out[0].h2d_conc_gbs = out[0].h2d_gbs;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (const auto & r : out) { g_results[r.dev] = r; }
    return out;
}

const llama_gpu_probe_fork_result * llama_gpu_probe_find_fork(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_results.find(dev);
    return it == g_results.end() ? nullptr : &it->second;
}

std::vector<float> llama_prefill_plan_fork(const std::vector<const llama_gpu_probe_fork_result *> & cards, const llama_prefill_model_fork & m, std::vector<std::string> * log) {
    const size_t n = cards.size();
    if (n < 2 || n > 8 || m.n_layer <= 0 || m.n_tokens <= 0) { return {}; }
    for (const auto * c : cards) {
        if (!c || c->h2d_gbs <= 0 || c->wtps <= 0) { return {}; }
    }
    const double L   = m.n_layer;
    const double N   = std::max(1.0, m.n_expert);
    const double k   = std::min(N, std::max(0.0, m.n_used));
    // bytes a pass uploads per layer: the experts that at least one of its tokens touches (a dense model has none)
    const double up_b  = m.expert_bytes_layer * (1.0 - std::pow(1.0 - k / N, m.n_tokens));
    // weight bytes a token reads per layer: the dense part and the experts it uses
    const double act_b = m.dense_bytes_layer + m.expert_bytes_layer * k / N;
    auto say = [&](const std::string & s) { if (log) { log->push_back(s); } };
    char buf[320];

    double best_t = 1e300;
    unsigned best_mask = 0;
    std::vector<double> best_share;
    for (size_t size = 1; size <= n; ++size) {              // fewer cards first: a tie keeps the smaller set
        for (unsigned mask = 1; mask < (1u << n); ++mask) {
            size_t bits = 0;
            for (size_t i = 0; i < n; ++i) { bits += mask >> i & 1; }   // (no __builtin_popcount: the MSVC build)
            if (bits != size) { continue; }
            std::vector<size_t> idx;
            for (size_t i = 0; i < n; ++i) { if (mask >> i & 1) { idx.push_back(i); } }
            const bool conc = idx.size() > 1;
            std::vector<double> u(idx.size()), c(idx.size());   // microseconds per layer: upload, compute
            double hand = 0.0;
            for (size_t j = 0; j < idx.size(); ++j) {
                const auto * r = cards[idx[j]];
                const double bw = conc && r->h2d_conc_gbs > 0 ? r->h2d_conc_gbs : r->h2d_gbs;
                u[j] = up_b / (bw * 1e3);
                c[j] = act_b * m.n_tokens / r->wtps * 1e6;
            }
            // one hand-off per boundary: the fixed latency plus the activations of the whole pass crossing the slower link
            for (size_t j = 0; j + 1 < idx.size(); ++j) {
                const auto * a = cards[idx[j]]; const auto * b = cards[idx[j + 1]];
                const double bw = std::max(1e-3, std::min(a->h2d_gbs, b->h2d_gbs));
                hand += std::max(a->lat_us, b->lat_us) + m.n_tokens * m.n_embd * 4.0 / (bw * 1e3);
            }
            // smallest T for which shares exist: every card takes at most T / (layers x its upload time) of the layers (its link finishes in T),
            // the shares add up to 1, and the sequential compute plus hand-offs fits in T; the compute chain is minimised by filling the cheapest
            // card first (fractional knapsack)
            std::vector<size_t> by_c(idx.size());
            for (size_t j = 0; j < by_c.size(); ++j) { by_c[j] = j; }
            std::sort(by_c.begin(), by_c.end(), [&](size_t x, size_t y) { return c[x] < c[y]; });
            auto fill = [&](double T, std::vector<double> * share) -> bool {
                double cap_sum = 0.0, rest = 1.0, chain = hand;
                std::vector<double> cap(idx.size());
                for (size_t j = 0; j < idx.size(); ++j) { cap[j] = u[j] > 0 ? std::min(1.0, T / (L * u[j])) : 1.0; cap_sum += cap[j]; }
                if (cap_sum < 1.0) { return false; }
                if (share) { share->assign(idx.size(), 0.0); }
                for (size_t j : by_c) {
                    const double take = std::min(cap[j], rest);
                    chain += take * L * c[j];
                    rest -= take;
                    if (share) { (*share)[j] = take; }
                }
                return chain <= T;
            };
            double lo = 0.0, hi = 1e15;
            for (int it = 0; it < 90; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (fill(mid, nullptr)) { hi = mid; } else { lo = mid; }
            }
            std::vector<double> share;
            fill(hi, &share);
            snprintf(buf, sizeof buf, "prefill plan: cards %u -> modelled pass %.0f ms", mask, hi / 1e3);
            say(buf);
            if (hi < best_t) {
                best_t = hi; best_mask = mask;
                best_share.assign(n, 0.0);
                for (size_t j = 0; j < idx.size(); ++j) { best_share[idx[j]] = share[j]; }
            }
        }
    }
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) { out[i] = (float) best_share[i]; }
    snprintf(buf, sizeof buf, "prefill plan: chosen cards %u, modelled pass %.0f ms for %.0f tokens", best_mask, best_t / 1e3, m.n_tokens);
    say(buf);
    return out;
}
