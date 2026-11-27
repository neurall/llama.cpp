// Every weight type x op (MUL_MAT, MUL_MAT_ID, GET_ROWS) x backend (CPU and each GPU) against a reference in double precision: the quantized weights as
// upstream's to_float gives them, times the f32 activations. A kernel or quant change that moves a result shows as a larger normalized error than the
// stored baseline (user 2026-11-27: Inkling 2 Mini PPL 1.61 with experts on the CPU against stock's 1.592; "unit test per quant specific op on cpu and gpu").
// Build against a dev build (no rebuild of the tree needed):
//   B=/p/bw/rels/devbuilds/<build>; clang++ -std=c++17 -O2 -I ggml/include tests/test-quant-ref-fork.cpp -o /tmp/tqr -L $B -lggml -lggml-base -Wl,-rpath,$B
//   /tmp/tqr $B [baseline.txt [--write]]       one line a case: backend type op shape nmse; exit 1 when a case is over its bound
// Bound: 3x the baseline's error for the case (+1e-7), or 1e-3 without a baseline. The activations are quantized by the backends (q8_1 / q8_K), so a few
// 1e-5 is the normal error; a wrong kernel is orders above it.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(42);

static std::vector<float> randv(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto & x : v) { x = d(rng); }
    return v;
}

// rows of n_per_row floats quantized to t, and the floats they stand for (to_float of the quantized rows)
static void quant(ggml_type t, const std::vector<float> & src, int64_t n_per_row, std::vector<uint8_t> & q, std::vector<float> & deq) {
    const int64_t nrows = (int64_t) src.size() / n_per_row;
    q.assign(ggml_row_size(t, n_per_row) * nrows, 0);
    std::vector<float> imat(n_per_row, 1.0f);
    ggml_quantize_chunk(t, src.data(), q.data(), 0, nrows, n_per_row, ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr);
    deq.resize(src.size());
    const auto * tr = ggml_get_type_traits(t);
    if (t == GGML_TYPE_F32) { memcpy(deq.data(), q.data(), q.size()); return; }
    tr->to_float(q.data(), deq.data(), (int64_t) src.size());
}

static double nmse(const std::vector<float> & out, const std::vector<double> & ref) {
    double e = 0, r = 0;
    for (size_t i = 0; i < ref.size(); ++i) { const double d = out[i] - ref[i]; e += d * d; r += ref[i] * ref[i]; }
    return r > 0 ? e / r : e;
}

struct result { std::string key; double err; bool ok; };

// one op on one backend: weights w (type t, [K, N, E]), returns the output as floats; empty when the backend refuses the op
static std::vector<float> run(ggml_backend_t be, ggml_backend_dev_t dev, const char * op, ggml_type t, int64_t K, int64_t N, int64_t E, int64_t T, int64_t U,
                              const std::vector<uint8_t> & wq, const std::vector<float> & act, const std::vector<int32_t> & ids) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, t, K, N, E);
    ggml_tensor * a = nullptr, * id = nullptr, * out = nullptr;
    if (!strcmp(op, "mul_mat")) {
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        out = ggml_mul_mat(ctx, ggml_view_2d(ctx, w, K, N, w->nb[1], 0), a);   // the first expert's matrix
    } else if (!strcmp(op, "mul_mat_id")) {
        a  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
        id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, T);
        out = ggml_mul_mat_id(ctx, w, a, id);
    } else {   // get_rows: T rows of the first expert's matrix
        id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        out = ggml_get_rows(ctx, ggml_view_2d(ctx, w, K, N, w->nb[1], 0), id);
    }
    std::vector<float> res;
    if (!ggml_backend_dev_supports_op(dev, out)) { ggml_free(ctx); return res; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    if (a)  { ggml_backend_tensor_set(a, act.data(), 0, act.size() * sizeof(float)); }
    if (id) { ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t)); }
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);
    if (ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BUILD_DIR [baseline.txt [--write]]\n", argv[0]); return 2; }
    ggml_backend_load_all_from_path(argv[1]);
    const char * base_path = argc > 2 ? argv[2] : nullptr;
    const bool write = argc > 3 && !strcmp(argv[3], "--write");
    std::map<std::string, double> base;
    if (base_path && !write) {
        if (FILE * f = fopen(base_path, "r")) {
            char k[256]; double e;
            while (fscanf(f, "%255s %lf", k, &e) == 2) { base[k] = e; }
            fclose(f);
        }
    }

    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4 };
    const int64_t K = 2304, N = 64, E = 8, U = 4;   // Inkling 2's width (9 x 256: every block size divides it)

    std::vector<result> results;
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        const auto dt = ggml_backend_dev_type(dev);
        if (dt != GGML_BACKEND_DEVICE_TYPE_CPU && dt != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { continue; }
        rng.seed(42);
        for (ggml_type t : types) {
            const auto w_f = randv((size_t) K * N * E, 0.02f);
            std::vector<uint8_t> wq; std::vector<float> wd;
            quant(t, w_f, K, wq, wd);
            for (const char * op : { "mul_mat", "mul_mat_id", "get_rows" }) {
                for (int64_t T : { 1, 7, 64 }) {
                    std::vector<float> act; std::vector<int32_t> ids; std::vector<double> ref;
                    if (!strcmp(op, "mul_mat")) {
                        act = randv((size_t) K * T);
                        ref.assign((size_t) N * T, 0.0);
                        for (int64_t j = 0; j < T; ++j) for (int64_t n = 0; n < N; ++n) {
                            double s = 0; for (int64_t k = 0; k < K; ++k) { s += (double) wd[n * K + k] * act[j * K + k]; }
                            ref[j * N + n] = s;
                        }
                    } else if (!strcmp(op, "mul_mat_id")) {
                        act = randv((size_t) K * T);
                        ids.resize((size_t) U * T);
                        for (int64_t j = 0; j < T; ++j) for (int64_t u = 0; u < U; ++u) { ids[j * U + u] = (int32_t) ((j * 3 + u * 5) % E); }
                        ref.assign((size_t) N * U * T, 0.0);
                        for (int64_t j = 0; j < T; ++j) for (int64_t u = 0; u < U; ++u) for (int64_t n = 0; n < N; ++n) {
                            const float * wr = &wd[((size_t) ids[j * U + u] * N + n) * K];
                            double s = 0; for (int64_t k = 0; k < K; ++k) { s += (double) wr[k] * act[j * K + k]; }
                            ref[(j * U + u) * N + n] = s;
                        }
                    } else {
                        ids.resize(T);
                        for (int64_t j = 0; j < T; ++j) { ids[j] = (int32_t) ((j * 13) % N); }
                        ref.resize((size_t) K * T);
                        for (int64_t j = 0; j < T; ++j) for (int64_t k = 0; k < K; ++k) { ref[j * K + k] = wd[(size_t) ids[j] * K + k]; }
                    }
                    const auto out = run(be, dev, op, t, K, N, E, T, U, wq, act, ids);
                    if (out.empty()) { continue; }
                    char key[256];
                    snprintf(key, sizeof(key), "%s/%s/%s/T%lld", ggml_backend_dev_name(dev), ggml_type_name(t), op, (long long) T);
                    const double e = nmse(out, ref);
                    const auto it = base.find(key);
                    const double bound = it != base.end() ? 3.0 * it->second + 1e-7 : 1e-3;
                    results.push_back({ key, e, e <= bound });
                    printf("%-44s %.3e%s\n", key, e, e <= bound ? "" : it != base.end() ? "   FAIL (baseline " : "   FAIL");
                    if (e > bound && it != base.end()) { printf("%.3e)\n", it->second); }
                }
            }
        }
        ggml_backend_free(be);
    }
    if (write && base_path) {
        FILE * f = fopen(base_path, "w");
        for (const auto & r : results) { fprintf(f, "%s %.6e\n", r.key.c_str(), r.err); }
        fclose(f);
        printf("baseline: %zu cases written to %s\n", results.size(), base_path);
    }
    int bad = 0;
    for (const auto & r : results) { bad += !r.ok; }
    printf("%zu cases, %d over their bound\n", results.size(), bad);
    return bad ? 1 : 0;
}
