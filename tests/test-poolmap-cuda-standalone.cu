// standalone harness: the kernel file + the reference, no ggml
#include "../ggml/src/ggml-cuda/poolmap-fork.cu"   // nvcc -std=c++17 -O2 -arch=sm_86 -I src tests/test-poolmap-cuda-standalone.cu
#include <random>
#include <cstring>
static int g_bad = 0;
static void check(const char * n, bool ok) { if (!ok) { if (g_bad < 20) printf("FAIL: %s\n", n); g_bad++; } }
static pv2_heap random_heap(std::mt19937 & rng, size_t slots) {
    pv2_heap h = pv2_make(0, slots * 10);
    for (size_t i = 0; i < slots; ) {
        const size_t k = 1 + rng() % 3, n = std::min(k, slots - i) * 10, x = i * 10;
        const int c = (int) (rng() % 100);
        pv2_res r; r.n = n; r.heat = (int64_t) (rng() % 1000); r.u = 10;
        if (c < 15) { i += k; continue; }
        if (c < 30) { r.id = PV2_HOLD; r.pin = true; r.kv = true; r.heat = 0; }
        else if (c < 40) { r.id = PV2_HOLD; r.pin = true; r.heat = 0; }
        else { r.id = (int32_t) x; r.pin = (int) (rng() % 100) < 20; }
        pv2_add(h, x, r); i += k;
    }
    return h;
}
int main() {
    const uint32_t cap = 8192;
    pm_entry * v_dev; uint32_t * cnt_dev; pm_result * res, * res_dev;
    CUDA_CHECK(cudaMalloc(&v_dev, cap * sizeof(pm_entry))); CUDA_CHECK(cudaMalloc(&cnt_dev, 4));
    void * scr; CUDA_CHECK(cudaMalloc(&scr, pm_scratch_bytes(cap)));
    const bool g_scr = !getenv("PM_NO_SORT");   // the batch's sorted takeable experts (the server's path); PM_NO_SORT=1: the scan for every request
    CUDA_CHECK(cudaHostAlloc((void **) &res, sizeof(pm_result), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &res_dev, res, 0));
    std::mt19937 rng(31337);
    int runs = 0, moved = 0, freed = 0, none = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float tsum = 0; int tn = 0;
    for (int it = 0; it < 1500; ++it) {
        const size_t slots = it % 10 == 0 ? 2000 : 40;
        pv2_heap h = random_heap(rng, slots);
        std::vector<pm_entry> v = pm_from_heap(h); uint32_t cnt = (uint32_t) v.size();
        cudaMemcpy(v_dev, v.data(), v.size() * sizeof(pm_entry), cudaMemcpyHostToDevice); cudaMemcpy(cnt_dev, &cnt, 4, cudaMemcpyHostToDevice);
        const uint64_t n = (1 + rng() % 4) * 10; const int32_t id = 1000000 + it;
        memset(res, 0, sizeof(pm_result));
        cudaEventRecord(e0);
        ggml_cuda_poolmap_alloc_fork(0, v_dev, cnt_dev, cap, h.lo, h.hi, n, 10, id, 7, 1, res_dev, nullptr, nullptr, 1);
        cudaEventRecord(e1);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (slots == 2000) { float ms; cudaEventElapsedTime(&ms, e0, e1); tsum += ms; tn++; }
        check("done", res->done == 1); check("no overflow", res->overflow == 0);
        pv2_heap ref = h; const pv2_alloc_result rr = pv2_alloc_do(ref, n, 10, id, 7, true); const pv2_alloc_result kr = pm_report(*res);
        runs++; if (rr.s == PV2_NONE) none++; moved += (int) kr.moves.size(); freed += (int) kr.freed.size();
        bool rep = kr.s == rr.s && kr.moves.size() == rr.moves.size() && kr.freed.size() == rr.freed.size();
        for (size_t k = 0; rep && k < kr.moves.size(); ++k) rep = kr.moves[k].from == rr.moves[k].from && kr.moves[k].to == rr.moves[k].to && kr.moves[k].id == rr.moves[k].id;
        for (size_t k = 0; rep && k < kr.freed.size(); ++k) rep = kr.freed[k].s == rr.freed[k].s && kr.freed[k].id == rr.freed[k].id;
        if (!rep && g_bad < 5) printf("  it %d n %zu kernel s %zu mv %zu fr %zu | ref s %zu mv %zu fr %zu\n", it, (size_t) n, kr.s, kr.moves.size(), kr.freed.size(), rr.s, rr.moves.size(), rr.freed.size());
        check("report equals reference", rep);
        cudaMemcpy(&cnt, cnt_dev, 4, cudaMemcpyDeviceToHost); std::vector<pm_entry> back(cnt); cudaMemcpy(back.data(), v_dev, cnt * sizeof(pm_entry), cudaMemcpyDeviceToHost);
        size_t d = 0; const bool sm = pm_same(back, pm_from_heap(ref), &d);
        if (!sm && g_bad < 8) printf("  map differs at %zu (it %d, cnt %u vs %zu)\n", d, it, cnt, pm_from_heap(ref).size());
        check("card map equals reference", sm); check("canonical", pm_canonical(back, h.lo, h.hi));
        pv2_heap mirror = h; check("mirror replays report", pv2_alloc_apply(mirror, kr, n, 10, id, 7, true) && pm_same(pm_from_heap(mirror), back));
    }
    // batches: k requests of different sizes in one launch, against the reference run one after the other
    {
        pm_req * rq, * rq_dev; pm_result * rs, * rs_dev;
        CUDA_CHECK(cudaHostAlloc((void **) &rq, 16 * sizeof(pm_req), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rq_dev, rq, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &rs, 16 * sizeof(pm_result), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rs_dev, rs, 0));
        int bad0 = g_bad, nb = 0; float bt = 0; int btn = 0;
        for (int it = 0; it < 500; ++it) {
            const size_t slots = it % 10 == 0 ? 2000 : 60;
            pv2_heap h = random_heap(rng, slots);
            std::vector<pm_entry> v = pm_from_heap(h); uint32_t cnt = (uint32_t) v.size();
            cudaMemcpy(v_dev, v.data(), v.size() * sizeof(pm_entry), cudaMemcpyHostToDevice); cudaMemcpy(cnt_dev, &cnt, 4, cudaMemcpyHostToDevice);
            const uint32_t k = 1 + rng() % 16;
            for (uint32_t j = 0; j < k; ++j) { rq[j] = { (1 + rng() % 4) * 10, 10, (int64_t) (rng() % 50), (int32_t) (2000000 + it * 16 + j), 1, (int32_t) ((it + j) % 3) }; }   // expert, KV, work
            memset(rs, 0, 16 * sizeof(pm_result));
            cudaEventRecord(e0);
            ggml_cuda_poolmap_alloc_batch_fork(0, v_dev, cnt_dev, cap, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr : nullptr);
            cudaEventRecord(e1);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (slots == 2000) { float ms; cudaEventElapsedTime(&ms, e0, e1); bt += ms / k; btn++; }
            pv2_heap ref = h, mirror = h;
            for (uint32_t j = 0; j < k; ++j) {
                const pv2_alloc_result rr = pv2_alloc_do(ref, rq[j].n, 10, rq[j].id, rq[j].hot, true, true, rq[j].kind);
                const pv2_alloc_result kr = pm_report(rs[j]);
                bool rep = rs[j].done == 1 && kr.s == rr.s && kr.moves.size() == rr.moves.size() && kr.freed.size() == rr.freed.size();
                for (size_t q = 0; rep && q < kr.moves.size(); ++q) rep = kr.moves[q].from == rr.moves[q].from && kr.moves[q].to == rr.moves[q].to;
                check("batch: report equals reference", rep);
                check("batch: mirror replays", pv2_alloc_apply(mirror, kr, rq[j].n, 10, rq[j].id, rq[j].hot, true, rq[j].kind));
            }
            nb += k;
            cudaMemcpy(&cnt, cnt_dev, 4, cudaMemcpyDeviceToHost); std::vector<pm_entry> back(cnt); cudaMemcpy(back.data(), v_dev, cnt * sizeof(pm_entry), cudaMemcpyDeviceToHost);
            check("batch: card map equals reference", pm_same(back, pm_from_heap(ref)));
            check("batch: card map equals mirror", pm_same(back, pm_from_heap(mirror)));
        }
        printf("batch: %d requests, %d failures; per request on ~1000-entry maps %.1f us\n", nb, g_bad - bad0, 1000.0 * bt / std::max(1, btn));
    }
    // the stream's case (Qwen3.8 14k full stream): ~15k same-size experts, no free bytes, every alloc replaces the coldest unpinned one; per request time
    {
        const uint32_t ne = 15000, k = 1000, cap2 = 32768;
        pm_entry * v2; uint32_t * c2; CUDA_CHECK(cudaMalloc(&v2, cap2 * sizeof(pm_entry))); CUDA_CHECK(cudaMalloc(&c2, 4));
        void * scr2; CUDA_CHECK(cudaMalloc(&scr2, pm_scratch_bytes(cap2)));
        pm_req * rq, * rq_dev; pm_result * rs, * rs_dev;
        CUDA_CHECK(cudaHostAlloc((void **) &rq, k * sizeof(pm_req), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rq_dev, rq, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &rs, k * sizeof(pm_result), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rs_dev, rs, 0));
        pv2_heap h = pv2_make(0, (size_t) ne * 10);
        for (uint32_t i = 0; i < ne; ++i) { pv2_res r; r.n = 10; r.u = 10; r.id = (int32_t) i; r.heat = (int64_t) (rng() % 100000); r.pin = rng() % 10 == 0; pv2_add(h, (size_t) i * 10, r); }
        std::vector<pm_entry> v = pm_from_heap(h); uint32_t cnt = (uint32_t) v.size();
        cudaMemcpy(v2, v.data(), v.size() * sizeof(pm_entry), cudaMemcpyHostToDevice); cudaMemcpy(c2, &cnt, 4, cudaMemcpyHostToDevice);
        for (uint32_t j = 0; j < k; ++j) { rq[j] = { 10, 10, (int64_t) (100000 + j), (int32_t) (ne + j), 1, 0 }; }
        memset(rs, 0, k * sizeof(pm_result));
        cudaEventRecord(e0);
        if (getenv("PM_ALLOC_ONLY")) { k_pm_alloc<<<1, PM_T>>>(v2, c2, cap2, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr); } else { ggml_cuda_poolmap_alloc_batch_fork(0, v2, c2, cap2, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr2 : nullptr); }
        cudaEventRecord(e1);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        pv2_heap ref = h; int bad0 = g_bad;
        for (uint32_t j = 0; j < k; ++j) {
            const pv2_alloc_result rr = pv2_alloc_do(ref, 10, 10, rq[j].id, rq[j].hot, true);
            const pv2_alloc_result kr = pm_report(rs[j]);
            check("stream: report equals reference", rs[j].done == 1 && kr.s == rr.s && kr.freed.size() == rr.freed.size() && kr.moves.size() == rr.moves.size());
        }
        cudaMemcpy(&cnt, c2, 4, cudaMemcpyDeviceToHost); std::vector<pm_entry> back(cnt); cudaMemcpy(back.data(), v2, cnt * sizeof(pm_entry), cudaMemcpyDeviceToHost);
        check("stream: card map equals reference", pm_same(back, pm_from_heap(ref)));
        printf("stream: %u requests on a %u-entry map, %d failures; %.1f us a request\n", k, ne, g_bad - bad0, 1000.0 * ms / k);
    }
    printf("runs %d moves %d freed %d none %d; alloc+apply on ~1000-entry maps %.1f us avg\n", runs, moved, freed, none, 1000.0 * tsum / std::max(1, tn));
    printf(g_bad ? "FAIL: %d failure(s)\n" : "OK: 0 failure(s)\n", g_bad);
    return g_bad != 0;
}
