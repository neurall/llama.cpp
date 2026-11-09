#include "llama-devpick-fork.h"

#include "llama-impl.h"
#include "llama-moestate-fork.h"
#include "llama-version.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

std::string backend_tag(ggml_backend_dev_t dev) {
    std::string n = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    for (char & c : n) { c = (char) tolower((unsigned char) c); }
    return (n.find("rocm") != std::string::npos || n.find("hip") != std::string::npos) ? "hip" : n;
}

// fastest of n runs of one q4_K 4096x4096 matrix times `cols` columns, in ms; < 0 when the device fails
// the loop stops after budget_ms (one run at least); a first run over 1 s gives up (the default stays): this runs at the start of every first run of a build
double time_matmul_core(ggml_backend_dev_t dev, int cols, int n, double budget_ms) {
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { return -1; }
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 4096, 4096);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, cols);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (!ggml_backend_dev_supports_op(dev, y)) {   // only what the backend lists for this device is timed
        LLAMA_LOG_WARN("%s: %s does not list this matrix op\n", __func__, ggml_backend_dev_name(dev));
        ggml_free(ctx);
        ggml_backend_free(be);
        return -1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    double best = -1;
    if (ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be)) {
        std::vector<uint8_t> wd(ggml_nbytes(w), 0x21);
        std::vector<float>   xd(4096 * (size_t) cols, 0.01f);
        ggml_backend_tensor_set(w, wd.data(), 0, wd.size());
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * sizeof(float));
        double spent = 0;
        for (int i = -2; i < n && spent < budget_ms; ++i) {   // two warm-up runs (pipeline build, kernel tuning)
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(be);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            spent += ms;
            if (!ok || (i == -2 && ms > 1000)) { best = -1; break; }
            if (i >= 0 && (best < 0 || ms < best)) { best = ms; }
        }
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(be);
    return best;
}

double time_matmul(ggml_backend_dev_t dev, int cols, int n, double budget_ms = 150) {
    try {
        return time_matmul_core(dev, cols, n, budget_ms);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: %s: %s\n", __func__, ggml_backend_dev_name(dev), e.what());
    } catch (...) {
        LLAMA_LOG_WARN("%s: %s: unknown error\n", __func__, ggml_backend_dev_name(dev));
    }
    return -1;
}

} // namespace

static ggml_backend_dev_t device_pick(ggml_backend_dev_t a, ggml_backend_dev_t b) {
    const char * env = getenv("LLAMA_DEVICE_PICK");
    if (env && !strcmp(env, "0")) { return a; }
    ggml_backend_dev_props pa;
    ggml_backend_dev_get_props(a, &pa);
    ggml_backend_dev_props pb;
    ggml_backend_dev_get_props(b, &pb);
    const char * gid = pa.device_id ? pa.device_id : pb.device_id;   // else the description (a driver without a PCI id)
    const std::string sec = std::string("gpu ") + (gid ? gid : ggml_backend_dev_description(a));
    const std::string bld = std::string("_b") + std::to_string(LLAMA_BUILD_NUM);
    ggml_backend_dev_t devs[2] = { a, b };
    const int64_t t_start = ggml_time_us();

    std::string v;
    const bool remeasure = env && !strcmp(env, "remeasure");
    if (!remeasure && moe_state_get(sec, "pick" + bld, v)) {
        for (ggml_backend_dev_t d : devs) {
            if (backend_tag(d) == v) {
                LLAMA_LOG_WARN("%s: %s (%s) runs on %s (stored pick, %s)\n", __func__, ggml_backend_dev_name(d), ggml_backend_dev_description(d), v.c_str(), sec.c_str());
                return d;
            }
        }
    }
    bool   failed[2] = { false, false };
    double total[2] = { 0, 0 };   // ms of a typical request: 2000 prompt tokens and 128 generated
    std::vector<std::pair<std::string, std::string>> kv = { { "name", ggml_backend_dev_description(a) } };
    for (int i = 0; i < 2; ++i) {
        const double p = time_matmul(devs[i], 512, 4), g = p < 0 ? -1 : time_matmul(devs[i], 1, 16);
        if (p < 0 || g < 0) {   // this backend fails here: the other one is used (both failing keeps a); nothing is stored
            LLAMA_LOG_WARN("%s: %s failed the timing pass\n", __func__, ggml_backend_dev_name(devs[i]));
            failed[i] = true;
            continue;
        }
        const std::string t = backend_tag(devs[i]);
        kv.push_back({ "1tok_p_time_" + t + bld, std::to_string(p / 512) });
        kv.push_back({ "1tok_g_time_" + t + bld, std::to_string(g) });
        total[i] = p / 512 * 2000 + g * 128;
        LLAMA_LOG_WARN("%s: %s (%s) %s: %.4f ms per prompt token, %.3f ms per generated token\n", __func__, ggml_backend_dev_name(devs[i]),
                ggml_backend_dev_description(devs[i]), t.c_str(), p / 512, g);
    }
    if (failed[0] || failed[1]) {
        return failed[0] && failed[1] ? a : failed[0] ? b : a;
    }
    ggml_backend_dev_t pick = total[1] < total[0] ? b : a;
    kv.push_back({ "pick" + bld, backend_tag(pick) });
    moe_state_set(sec, kv);
    LLAMA_LOG_WARN("%s: %s runs on %s (request of 2000 + 128 tokens: %.0f against %.0f ms)\n", __func__, ggml_backend_dev_description(pick),
            backend_tag(pick).c_str(), std::min(total[0], total[1]), std::max(total[0], total[1]));
    LLAMA_LOG_WARN("%s: measured in %.0f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
    return pick;
}

ggml_backend_dev_t llama_device_pick(ggml_backend_dev_t a, ggml_backend_dev_t b) {
    try {
        return device_pick(a, b);
    } catch (...) {   // a failing pair of backends must not stop the others (nor the load)
        LLAMA_LOG_WARN("%s: %s kept after an error\n", __func__, ggml_backend_dev_name(a));
        return a;
    }
}
