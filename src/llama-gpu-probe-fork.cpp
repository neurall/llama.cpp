// Fork: per-GPU probe, and the prefill layer plan computed from it.
//
// Problem. With several GPUs the layers are split between the cards by free memory. That says nothing about how fast a card can take part in a
// prefill pass: a card on a narrow or shared link (x4 through the chipset, an x1 riser), a slower GPU on a fat link, or a fast GPU on x8 all behave
// differently, and a link-width threshold is wrong in both directions (a half-speed card on x16 is a bottleneck that the width does not show, a
// double-speed card on x8 is worth using although its link is half). On the test machine (RTX 3090 x16 + RTX 3090 x4 through the chipset) the default
// split cost MiMo 14% of its prompt speed against the x16 card alone, because the x16 card waited 20 to 40% of the pass for the x4 card's layers.
//
// What the probe measures, per card, before any graph or compute buffer exists (about 1 s per card, no model needed):
//   h2d_gbs       bulk host->device upload, 32 MB from pinned memory, this card alone. The expert upload of a streamed prefill is bound by this.
//   h2d_conc_gbs  the same while every card uploads at once. Cards behind one root complex or the chipset uplink share bandwidth; only this number
//                 shows it, and it is what a real prefill sees because every card streams its own layers' experts at the same time.
//   lat_us        median of 64 small (4 KB) synchronous uploads: the fixed cost of one hand-off between two layers on different cards.
//   compute       the compute part, portable on purpose: one small ggml graph (a mul_mat of Q4_K weights, 4096 wide, against 512 fake tokens) run through the
//                 backend's own matmul kernels on every card, the fastest of three short batches, in weight bytes x tokens per second. It needs no model, no big
//                 allocation and no per-backend code, so it works on CUDA, HIP, Vulkan, Metal, SYCL alike. Only the RATIO between cards is used. The absolute scale
//                 comes from one real measurement (the fastest card's prefill tokens per second on this model, --moe prefill-plan-tps or the stored value of an
//                 earlier run); without it the plan assumes that the fastest card's compute and upload take equally long per layer at the pass size and says so
//                 in its log. A probe this short does not reach the power-capped clocks of a sustained pass (measured: a x4 card limited to 220 W reads 6% slower
//                 here, 20% slower in a long burst); that is left to the measured prefill speed at run time.
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
#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace {

// Size of the bulk upload. The probe ranks cards and sizes shares, it does not need to saturate a link: measured on a x16 + x4 box, 32 MB with 2+4 repetitions
// reads the x16 card within 0.3% of 64 MB (26.0 GB/s) in 0.20 s instead of 0.45 s; at 16 MB the fast link reads 6% low (the transfer ends before the link ramps
// up), at 8 MB 13% low.
constexpr size_t PROBE_BYTES = 32u << 20;
constexpr int    LAT_SAMPLES  = 64;
constexpr size_t LAT_BYTES    = 4096;

std::mutex                                                  g_mu;
std::map<ggml_backend_dev_t, llama_gpu_probe_fork_result>   g_results;
std::vector<std::vector<double>>                            g_conc;

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

// Portable fallback for backends without their own loop probe: one small ggml graph, a mul_mat of Q4_K weights (4096 wide, 1024 rows, 2 MB, the real
// quantised kernels of the backend) against 512 fake tokens, run a few times; the fastest batch of four runs is taken. Result in weight bytes x tokens per second.
double graph_compute(ggml_backend_dev_t dev) {
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { return 0.0; }
    double result = 0.0;
    {
        dev_ctx c(ggml_tensor_overhead() * 8);
        const int64_t K = 4096, rows = 1024, tokens = 512;
        ggml_tensor * w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_Q4_K, K, rows);
        ggml_tensor * x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, tokens);
        ggml_cgraph * gf = ggml_new_graph(c.ctx);
        ggml_build_forward_expand(gf, ggml_mul_mat(c.ctx, w, x));
        if (c.alloc(dev)) {
            std::vector<uint8_t> bytes(ggml_nbytes(w), 1);
            ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());
            std::vector<float> ones((size_t) K * tokens, 1.0f);
            ggml_backend_tensor_set(x, ones.data(), 0, ones.size() * sizeof(float));
            for (int i = 0; i < 3; ++i) { ggml_backend_graph_compute(be, gf); }   // kernels, caches and graph set-up: not counted
            ggml_backend_synchronize(be);
            double best_us = 1e300;
            for (int rep = 0; rep < 3; ++rep) {
                const int64_t t0 = ggml_time_us();
                for (int i = 0; i < 4; ++i) { ggml_backend_graph_compute(be, gf); }
                ggml_backend_synchronize(be);
                best_us = std::min(best_us, (double) (ggml_time_us() - t0) / 4.0);
            }
            result = (double) ggml_nbytes(w) * tokens / (best_us / 1e6);
        }
    }
    ggml_backend_free(be);
    return result;
}

}  // namespace

std::vector<llama_gpu_probe_fork_result> llama_gpu_probe_fork(const std::vector<ggml_backend_dev_t> & gpus, std::vector<std::vector<double>> * conc) {
    std::vector<llama_gpu_probe_fork_result> out(gpus.size());
    // per card, alone: upload, latency, compute
    for (size_t i = 0; i < gpus.size(); ++i) {
        out[i].dev  = gpus[i];
        out[i].index = i;
        out[i].name = ggml_backend_dev_name(gpus[i]);
        host_buf h(gpus[i], PROBE_BYTES);
        dev_ctx c(0);
        ggml_tensor * big   = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) PROBE_BYTES);
        ggml_tensor * small = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) LAT_BYTES);
        if (c.alloc(gpus[i])) {
            out[i].h2d_gbs = bulk_gbs(big, h.ptr, 2);
            out[i].lat_us  = latency_us(small, h.ptr);
        }
    }
    // compute: the portable graph on every card, through the backend's own matmul kernels (see graph_compute); same unit for all cards
    for (size_t i = 0; i < gpus.size(); ++i) { out[i].compute = graph_compute(gpus[i]); }
    // every subset of cards uploading at once (and the singles alone): each member's rate = its bytes over its own time after a common start. Cards behind one
    // root complex or the chipset uplink slow each other down; the subset table is what the plan reads to see what a card steals from the others.
    const size_t n = gpus.size();
    if (conc) { conc->assign((size_t) 1 << n, std::vector<double>(n, 0.0)); }
    for (unsigned mask = 1; mask < ((unsigned) 1 << n); ++mask) {
        std::vector<double> rate(n, 0.0);
        std::atomic<bool> go{false};
        std::atomic<int>  ready{0};   // threads that finished their set-up and warm-up: the start is a real barrier, not a sleep
        int               want = 0;
        std::vector<std::thread> th;
        for (size_t i = 0; i < n; ++i) {
            if (!(mask >> i & 1)) { continue; }
            ++want;
            th.emplace_back([&, i] {
                host_buf h(gpus[i], PROBE_BYTES);
                dev_ctx c(0);
                ggml_tensor * big = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I8, (int64_t) PROBE_BYTES);
                const bool ok = c.alloc(gpus[i]);
                if (ok) { ggml_backend_tensor_set(big, h.ptr, 0, PROBE_BYTES); }   // warm-up
                ready.fetch_add(1);   // (also when the allocation failed, so the barrier cannot hang)
                if (!ok) { return; }
                while (!go.load()) { std::this_thread::yield(); }
                const int reps = 4;
                const int64_t t0 = ggml_time_us();
                for (int r = 0; r < reps; ++r) { ggml_backend_tensor_set(big, h.ptr, 0, PROBE_BYTES); }
                rate[i] = (double) PROBE_BYTES * reps / 1e3 / (double) std::max<int64_t>(1, ggml_time_us() - t0);
            });
        }
        while (ready.load() < want) { std::this_thread::yield(); }
        go.store(true);
        for (auto & t : th) { t.join(); }
        if (conc) { (*conc)[mask] = rate; }
        if (mask == ((unsigned) 1 << n) - 1) { for (size_t i = 0; i < n; ++i) { out[i].h2d_conc_gbs = rate[i]; } }
    }
    if (n == 1) { out[0].h2d_conc_gbs = out[0].h2d_gbs; }
    std::lock_guard<std::mutex> lk(g_mu);
    for (const auto & r : out) { g_results[r.dev] = r; }
    if (conc) { g_conc = *conc; }
    return out;
}

const std::vector<std::vector<double>> & llama_gpu_probe_conc_fork() {
    return g_conc;
}

const llama_gpu_probe_fork_result * llama_gpu_probe_find_fork(ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_results.find(dev);
    return it == g_results.end() ? nullptr : &it->second;
}

static std::vector<float> prefill_plan_impl(const std::vector<const llama_gpu_probe_fork_result *> & cards, const std::vector<std::vector<double>> & conc,
                                            const llama_prefill_model_fork & m, std::vector<std::string> * log, const std::vector<double> * eval_split, double * eval_us) {
    const size_t n = cards.size();
    if (n < 2 || n > 8 || m.n_layer <= 0 || m.n_tokens <= 0) { return {}; }
    for (const auto * c : cards) {
        if (!c || c->h2d_gbs <= 0 || c->compute <= 0) { return {}; }
    }
    const double L = m.n_layer;
    const double N = std::max(1.0, m.n_expert);
    double fastest_compute = 0.0;   // probe units; the card with the most is the reference for the relative speeds
    size_t ref = 0;
    for (size_t i = 0; i < n; ++i) { if (cards[i]->compute > fastest_compute) { fastest_compute = cards[i]->compute; ref = i; } }
    const double k = std::min(N, std::max(0.0, m.n_used));
    auto say = [&](const std::string & s) { if (log) { log->push_back(s); } };
    char buf[400];
    auto has = [](unsigned mask, size_t i) { return (mask >> i & 1u) != 0; };
    auto members = [&](unsigned mask) { std::vector<size_t> v; for (size_t i = 0; i < n; ++i) { if (has(mask, i)) { v.push_back(i); } } return v; };
    // upload rate of card i while exactly the cards of `mask` upload: measured (subset table) when present, else the all-cards rate
    auto rate = [&](unsigned mask, size_t i) {
        if (mask < conc.size() && i < conc[mask].size() && conc[mask][i] > 0) { return conc[mask][i]; }
        return members(mask).size() > 1 && cards[i]->h2d_conc_gbs > 0 ? cards[i]->h2d_conc_gbs : cards[i]->h2d_gbs;
    };
    // the layers each card owns in the decode layout: the layers of a card that is left out of the prefill have to be computed by the others
    std::vector<double> own(n, 1.0 / (double) n);
    if (m.owner_share.size() == n) { own = m.owner_share; }

    // Modelled time in microseconds of a prompt of n_tok tokens on the cards of `mask`. The prompt runs in p = ceil(n_tok / pass) passes and consecutive
    // passes are pipelined over the cards (ggml_backend_sched pipeline parallelism). A card's stage time for one pass is its share of the layers times
    // the larger of its compute and its upload time per layer (uploads run ahead of the compute, so they overlap). With p passes
    //     T = sum of the stage times (the first pass walks through every card)  +  (p - 1) x the slowest stage  +  hand-offs
    // so one pass is sequential and extra cards cannot help beyond their link, while many passes make the slowest stage the limit and the best shares
    // equalise the stage times. On top comes what leaving the other cards out costs: their dense weights have to be on the cards that compute their layers
    // (one copy from host per prompt) and the K/V their layers produce has to reach the owning card afterwards. Shares go to `share` (size n, 0 for non-members).
    const std::vector<double> * fixed_split = nullptr;   // llama_prefill_plan_eval_fork: evaluate this split instead of searching one
    std::function<double(unsigned, double, std::vector<double> *)> solve = [&](unsigned mask, double n_tok, std::vector<double> * share) -> double {
        const auto idx = members(mask);
        const double pass_tok = m.pass_tokens > 0 ? std::min(n_tok, m.pass_tokens) : n_tok;
        const double p        = std::ceil(n_tok / pass_tok);   // number of passes; pipelining between them only counts when m.pipelined (see the header)
        const double up_b  = m.expert_bytes_layer * (1.0 - std::pow(1.0 - k / N, pass_tok));   // experts a pass touches per layer (a dense model has none)
        const double act_b = m.dense_bytes_layer + m.expert_bytes_layer * k / N;               // weight bytes one token reads per layer
        // How this engine runs a prompt pass (measured, nsys + scheduler trace): the experts of EVERY layer are uploaded to and computed on one card, the offload card E
        // (the member with the fastest link), whoever owns the layer. A layer owned by another card j runs its dense part (attention, norms, router, shared expert) on j
        // and costs two hops of the whole batch's activations per pass (E -> j for the attention, j -> E for the experts), each tokens x n_embd x 4 bytes over the
        // slower of the two links plus a stream synchronisation. So per layer:
        //   expert part  x_E = max(upload on E, expert compute on E), always on E (the upload of the next layer hides under the compute of this one)
        //   dense part   d_j = dense compute on the owner;  hops h_j = 2 x (latency + activations / link) for an owner other than E
        // The compute of a layer on a card is the card's relative speed applied to the calibrated time per layer, split into the expert and the dense part by the
        // weight bytes a token reads in each.
        size_t e_j = 0;                                   // index (in idx) of the offload card
        for (size_t j = 1; j < idx.size(); ++j) { if (cards[idx[j]]->h2d_gbs > cards[idx[e_j]]->h2d_gbs) { e_j = j; } }
        const double act_exp = m.expert_bytes_layer * k / N, act_den = m.dense_bytes_layer;
        const double phi = act_exp + act_den > 0 ? act_exp / (act_exp + act_den) : 0.5;   // share of a layer's compute that is experts
        double x_E = 0.0;
        std::vector<double> a(idx.size());                // per unit share of the layers: the dense compute of the layers the card owns, plus the hops when it is not E
        double best_bw = 0.0;
        for (size_t j = 0; j < idx.size(); ++j) {
            const double bw = rate(mask, idx[j]);
            best_bw = std::max(best_bw, bw);
            const double rel = cards[idx[j]]->compute / fastest_compute;
            const double c_ref = m.fastest_prefill_tps > 0 ? pass_tok / (m.fastest_prefill_tps * L) * 1e6 : up_b / (cards[ref]->h2d_gbs * 1e3);
            const double c_layer = c_ref / rel;           // compute of a whole layer on this card
            if (j == e_j) {
                x_E = std::max(up_b / (bw * 1e3), c_layer * phi);
                a[j] = L * c_layer * (1.0 - phi);
            } else {
                const double link = std::max(1e-3, std::min(cards[idx[j]]->h2d_gbs, cards[idx[e_j]]->h2d_gbs));
                // two hops of the batch's activations over the slower link, and the stall of the stream synchronisation at the hop: the host waits until the expert tensor
                // that is uploading on E (one lookahead, a third of a layer's experts) has arrived, and starts no further upload meanwhile (measured: 20 ms per x4 layer on MiMo,
                // the upload of one tensor takes 29 ms there)
                const double stall = up_b / 3.0 / (rate(mask, idx[e_j]) * 1e3);
                const double hop  = 2.0 * (std::max(cards[idx[j]]->lat_us, cards[idx[e_j]]->lat_us) + pass_tok * m.n_embd * 4.0 / (link * 1e3)) + stall;
                a[j] = L * (c_layer * (1.0 - phi) + hop);
            }
        }
        const double hand = 0.0;   // the hops are in a[j] now
        // for a bound M on the slowest stage a card takes at most M / a_j of the layers; the sum of the stage times is smallest when the cheapest card
        // is filled first (fractional knapsack; cards with equal cost share the layers in proportion to their cap); T(M) = chain(M) + (p - 1) M is
        // convex in M, so a ternary search finds the best M
        auto eval = [&](double M, std::vector<double> * sh) -> double {
            if (M < L * x_E) { return 1e300; }              // E's stage holds the experts of every layer
            std::vector<double> cap(idx.size());
            double cap_sum = 0.0;
            for (size_t j = 0; j < idx.size(); ++j) {
                const double room = j == e_j ? M - L * x_E : M;
                cap[j] = a[j] > 0 ? std::min(1.0, room / a[j]) : 1.0;
                cap_sum += cap[j];
            }
            if (cap_sum < 1.0 - 1e-12) { return 1e300; }
            std::vector<size_t> ord(idx.size());
            for (size_t j = 0; j < ord.size(); ++j) { ord[j] = j; }
            std::sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return a[x] < a[y]; });
            if (sh) { sh->assign(idx.size(), 0.0); }
            double rest = 1.0, chain = L * x_E;
            for (size_t g = 0; g < ord.size() && rest > 1e-15; ) {
                size_t e = g;
                double gcap = 0.0;
                while (e < ord.size() && a[ord[e]] <= a[ord[g]] * (1.0 + 1e-9)) { gcap += cap[ord[e]]; ++e; }
                const double take = std::min(gcap, rest);
                for (size_t q = g; q < e; ++q) {
                    const double part = gcap > 0 ? take * cap[ord[q]] / gcap : 0.0;
                    chain += part * a[ord[q]];
                    if (sh) { (*sh)[ord[q]] = part; }
                }
                rest -= take;
                g = e;
            }
            return chain + (p - 1.0) * M;
        };
        if (fixed_split) {   // the modelled time of a given split: sum of the stage times + (passes - 1) x the slowest stage
            double sum = L * x_E, mx = L * x_E;
            for (size_t j = 0; j < idx.size(); ++j) {
                const double f = (*fixed_split)[idx[j]];
                const double st = f * a[j] + (j == e_j ? L * x_E : 0.0);
                sum += f * a[j]; mx = std::max(mx, st);
            }
            if (share) { share->assign(n, 0.0); for (size_t j = 0; j < idx.size(); ++j) { (*share)[idx[j]] = (*fixed_split)[idx[j]]; } }
            return sum + (p - 1.0) * mx;
        }
        // search M between the smallest feasible stage and one card doing everything. Feasible = the caps of the cards add up to the whole model; that is monotone in M,
        // so the smallest feasible M is found by bisection first and the (convex) cost is then searched on the feasible side only
        double hi = L * x_E;
        for (size_t j = 0; j < idx.size(); ++j) { hi = std::max(hi, j == e_j ? L * x_E + a[j] : a[j]); }
        auto feasible = [&](double M) {
            if (M < L * x_E) { return false; }
            double cs = 0.0;
            for (size_t j = 0; j < idx.size(); ++j) { cs += a[j] > 0 ? std::min(1.0, (j == e_j ? M - L * x_E : M) / a[j]) : 1.0; }
            return cs >= 1.0 - 1e-12;
        };
        double lo = L * x_E;
        if (!feasible(lo)) {
            double l2 = lo, h2 = hi;
            for (int it = 0; it < 100; ++it) { const double mid = 0.5 * (l2 + h2); if (feasible(mid)) { h2 = mid; } else { l2 = mid; } }
            lo = h2;
        }
        for (int it = 0; it < 100; ++it) {
            const double m1 = lo + (hi - lo) / 3.0, m2 = hi - (hi - lo) / 3.0;
            if (eval(m1, nullptr) <= eval(m2, nullptr)) { hi = m2; } else { lo = m1; }
        }
        std::vector<double> sh;
        const double t_pass = eval(0.5 * (lo + hi), &sh) + hand;
        if (sh.size() != idx.size()) { sh.assign(idx.size(), 0.0); sh[e_j] = 1.0; }   // (cannot happen: the search stays on the feasible side)
        // a member that ends up with no layers is not taking part: the set without it is the same prefill, and it pays the cost of sitting out (below)
        if (idx.size() > 1) {
            for (size_t j = 0; j < idx.size(); ++j) { if (sh[j] < 1e-9) { return solve(mask & ~(1u << idx[j]), n_tok, share); } }
        }
        double setup = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (has(mask, i)) { continue; }
            setup += own[i] * L * m.dense_bytes_layer / (best_bw * 1e3);                                                              // dense weights to the cards in `mask`
            setup += own[i] * L * m.kv_bytes_token_layer * n_tok / (std::max(1e-3, std::min(cards[i]->h2d_gbs, best_bw)) * 1e3);   // K/V back to the owner
        }
        if (share) { share->assign(n, 0.0); for (size_t j = 0; j < idx.size(); ++j) { (*share)[idx[j]] = sh[j]; } }
        return t_pass + setup;
    };

    if (eval_split) {   // only evaluate: the modelled time of this split over all the cards, no search
        fixed_split = eval_split;
        if (eval_us) { *eval_us = solve((1u << n) - 1, m.n_tokens, nullptr); }
        return {};
    }
    // the best subset for the pass size of the plan; fewer cards first, so a tie keeps the smaller set
    double best_t = 1e300;
    unsigned best_mask = 0;
    std::vector<double> best_share;
    for (size_t size = 1; size <= n; ++size) {
        for (unsigned mask = 1; mask < (1u << n); ++mask) {
            if (members(mask).size() != size) { continue; }
            std::vector<double> sh;
            const double t = solve(mask, m.n_tokens, &sh);
            snprintf(buf, sizeof buf, "prefill plan: cards %u -> modelled pass %.0f ms", mask, t / 1e3);
            say(buf);
            if (t < best_t) { best_t = t; best_mask = mask; best_share = sh; }
        }
    }

    // The accounting per card, at the prompt size of the plan. For a card in the chosen set: the upload bandwidth it brings and the bandwidth it takes away from
    // the others (measured with the subset table), what it saves per prompt token, and the prompt sizes (powers of two, tokens) for which the set with the card
    // is faster than the set without it. For a card left out: the prompt sizes for which including it would be faster. The range has two ends because a card
    // pays back only when the prompt is long enough for its fixed costs and short enough for the pipeline not to be compute-bound.
    auto helps_range = [&](unsigned with, unsigned without) -> std::string {
        double first = -1, last = -1;
        for (double t = 1.0; t <= 16777216.0; t *= 2.0) {
            if (solve(with, t, nullptr) < solve(without, t, nullptr)) { if (first < 0) { first = t; } last = t; }
        }
        if (first < 0) { return "never"; }
        char b2[64];
        snprintf(b2, sizeof b2, "%.0f to %.0f tokens", first, last);
        return b2;
    };
    for (size_t i = 0; i < n; ++i) {
        if (has(best_mask, i)) {
            if (members(best_mask).size() < 2) { continue; }
            const unsigned rest = best_mask & ~(1u << i);
            double stolen = 0.0;
            for (size_t j : members(rest)) { stolen += rate(rest, j) - rate(best_mask, j); }
            const double gain = (solve(rest, m.n_tokens, nullptr) - solve(best_mask, m.n_tokens, nullptr)) / m.n_tokens;
            snprintf(buf, sizeof buf, "prefill plan: %s brings %.1f GB/s, takes %.1f GB/s from the other cards, saves %.1f us per token, helps for prompts of %s",
                     cards[i]->name.c_str(), rate(best_mask, i), stolen, gain, helps_range(best_mask, rest).c_str());
        } else {
            snprintf(buf, sizeof buf, "prefill plan: %s is left out; including it would help for prompts of %s",
                     cards[i]->name.c_str(), helps_range(best_mask | (1u << i), best_mask).c_str());
        }
        say(buf);
    }
    say(m.fastest_prefill_tps > 0 ? "prefill plan: compute scale calibrated to the measured prefill speed of the fastest card" : "prefill plan: UNCALIBRATED, the fastest card's compute assumed equal to its upload per layer; relative speeds from the probe (--moe prefill-plan-tps=N, its prefill tokens/s, calibrates)");
    // The SET of cards comes from the search above (it matched the measurements: the x16 alone at 1.6k tokens, both cards from a few passes up). The SHARES it found
    // do not: the stage balance of a pipelined pass gave the x4 card 76-86% of the layers for a 12k-token GLM prompt, and the measured prompt speed is best at 15%
    // (GLM 765.6 prompt t/s vs 740.5 at 20%; MiMo 711 vs 579 at 20%; 86% was slower than not using the card). So the shares are the upload-rate shares of the chosen
    // cards, damped by `slow_scale` for every card that is not on the fastest link (it also pays for the hand-offs of its layers).
    std::vector<float> out(n, 0.0f);
    if (members(best_mask).size() > 1) {
        double top = 0.0, sum = 0.0;
        for (size_t i : members(best_mask)) { top = std::max(top, rate(best_mask, i)); }
        for (size_t i : members(best_mask)) { sum += rate(best_mask, i) * (rate(best_mask, i) < top ? m.slow_scale : 1.0); }
        for (size_t i : members(best_mask)) { out[i] = (float) (rate(best_mask, i) * (rate(best_mask, i) < top ? m.slow_scale : 1.0) / sum); }
    } else {
        for (size_t i = 0; i < n; ++i) { out[i] = (float) best_share[i]; }
    }
    snprintf(buf, sizeof buf, "prefill plan: chosen cards %u, modelled pass %.0f ms for %.0f tokens", best_mask, best_t / 1e3, m.n_tokens);
    say(buf);
    return out;
}

// Dense weights resident or streamed? The per-token compute is the same either way, so only the pass count and the dense traffic differ:
//     resident: ub_res = (VRAM - ring - K/V(n) - all dense) / scratch per token;       cost = passes_res x a
//     streamed: ub_str = (VRAM - ring - K/V(n) - dense of 2 layers) / scratch per token; cost = passes_str x (a + dense_total / upload rate)
// Streamed also decides alone when the resident layout leaves no room for a micro-batch (a long prompt: its K/V is the big consumer), and neither layout
// is feasible when even the streamed one leaves nothing: then more cards must take part. A prompt that fits one pass in the resident layout never streams.
llama_prefill_mem_choice llama_prefill_mem_plan_fork(const llama_prefill_mem_fork & m) {
    llama_prefill_mem_choice c;
    if (m.scratch_bytes_per_token <= 0 || m.n_tokens <= 0) { return c; }
    const double kv = m.kv_bytes_per_token * m.n_tokens;
    auto ub_for = [&](double dense_resident) {
        const double room = m.vram_bytes - m.ring_bytes - kv - dense_resident;
        double ub = room > 0 ? std::floor(room / m.scratch_bytes_per_token) : 0.0;
        if (m.ub_cap > 0) { ub = std::min(ub, m.ub_cap); }
        return std::min(ub, m.n_tokens);
    };
    const double ub_res = ub_for(m.dense_bytes_total);
    const double ub_str = ub_for(2.0 * m.dense_bytes_layer);
    const double dense_us = m.upload_gbs > 0 ? m.dense_bytes_total / (m.upload_gbs * 1e3) : 0.0;
    if (ub_res < 1.0 && ub_str < 1.0) { c.reason = "does not fit even with the dense weights streamed: more cards must take part"; return c; }
    c.feasible = true;
    const double p_res = ub_res >= 1.0 ? std::ceil(m.n_tokens / ub_res) : 1e30;
    const double p_str = ub_str >= 1.0 ? std::ceil(m.n_tokens / ub_str) : 1e30;
    const double t_res = p_res * m.pass_fixed_us;
    const double t_str = p_str * (m.pass_fixed_us + dense_us);
    if (ub_res < 1.0) {
        c.stream_dense = true; c.ub = ub_str; c.passes = p_str; c.extra_us = t_str; c.reason = "dense resident leaves no room for a micro-batch";
    } else if (t_str < t_res) {
        c.stream_dense = true; c.ub = ub_str; c.passes = p_str; c.extra_us = t_str; c.reason = "fewer passes save more than the dense traffic costs";
    } else {
        c.stream_dense = false; c.ub = ub_res; c.passes = p_res; c.extra_us = t_res; c.reason = ub_res >= m.n_tokens ? "one pass fits with the dense weights resident" : "the extra dense traffic costs more than fewer passes save";
    }
    return c;
}

std::vector<float> llama_prefill_plan_fork(const std::vector<const llama_gpu_probe_fork_result *> & cards, const std::vector<std::vector<double>> & conc,
                                           const llama_prefill_model_fork & m, std::vector<std::string> * log) {
    return prefill_plan_impl(cards, conc, m, log, nullptr, nullptr);
}

// Modelled time in microseconds of the whole prompt for a given split of the layers (one share per card, sum 1), no search: to compare the model with measured runs.
double llama_prefill_plan_eval_fork(const std::vector<const llama_gpu_probe_fork_result *> & cards, const std::vector<std::vector<double>> & conc,
                                    const llama_prefill_model_fork & m, const std::vector<double> & split) {
    double us = -1.0;
    if (split.size() == cards.size()) { prefill_plan_impl(cards, conc, m, nullptr, &split, &us); }
    return us;
}
