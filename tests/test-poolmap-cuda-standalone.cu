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
    const bool g_scr = !getenv("PM_NO_SORT");   // the size-class lists (the server's path); PM_NO_SORT=1: the scan for every request
    pm_batch g_full; g_full.rebuild = 1;          // every test map below is sent whole
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
            ggml_cuda_poolmap_alloc_batch_fork(0, v_dev, cnt_dev, cap, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr : nullptr, g_full);
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
        if (getenv("PM_ALLOC_ONLY")) { k_pm_alloc<<<1, PM_T>>>(v2, c2, cap2, h.lo, h.hi, rq_dev, k, rs_dev, g_scr ? scr2 : nullptr, g_full); } else { ggml_cuda_poolmap_alloc_batch_fork(0, v2, c2, cap2, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr2 : nullptr, g_full); }
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
    // kv-move (pin bit 3): a resident moving out of a KV stretch takes only free bytes or a colder same-size expert, never a window; else refused
    {
        pm_req * rq, * rq_dev; pm_result * rs, * rs_dev;
        CUDA_CHECK(cudaHostAlloc((void **) &rq, sizeof(pm_req), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rq_dev, rq, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &rs, sizeof(pm_result), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rs_dev, rs, 0));
        int bad0 = g_bad, n_ok = 0, n_ref = 0;
        for (int it = 0; it < 400; ++it) {
            pv2_heap h = random_heap(rng, 60);
            std::vector<pm_entry> v = pm_from_heap(h); uint32_t cnt = (uint32_t) v.size();
            const int64_t hot = (int64_t) (rng() % 1000);
            cudaMemcpy(v_dev, v.data(), v.size() * sizeof(pm_entry), cudaMemcpyHostToDevice); cudaMemcpy(cnt_dev, &cnt, 4, cudaMemcpyHostToDevice);
            rq[0] = { 10, 10, hot, 3000000 + it, 1 | 2 | 8, 0 };
            memset(rs, 0, sizeof(pm_result));
            ggml_cuda_poolmap_alloc_batch_fork(0, v_dev, cnt_dev, cap, h.lo, h.hi, rq_dev, 1, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr : nullptr, g_full);
            CUDA_CHECK(cudaDeviceSynchronize());
            check("kv-move: done", rs[0].done == 1);
            check("kv-move: no window movers", rs[0].n_mv == 0);
            bool colder = true;
            for (uint32_t f = 0; f < rs[0].n_fr; ++f) {
                auto it2 = h.res.find((size_t) rs[0].fr[f].s);
                colder = colder && it2 != h.res.end() && it2->second.heat < hot;
            }
            check("kv-move: victims colder than the mover", colder);
            if (rs[0].s == UINT64_MAX) { n_ref++; } else { n_ok++; }
        }
        printf("kv-move: %d placed, %d refused, %d failures\n", n_ok, n_ref, g_bad - bad0);
    }
    // the card keeps its map and lists between batches: the host's changes go as a diff (pm_diff), layers ahead as busy bits; three rounds a map
    {
        const uint32_t E = 1000, NW = 4096, KQ = 24;
        pm_req * rq, * rq_dev; pm_result * rs, * rs_dev; pm_delta * ops, * ops_dev; pm_entry * ents, * ents_dev; uint32_t * busy, * busy_dev;
        CUDA_CHECK(cudaHostAlloc((void **) &rq, KQ * sizeof(pm_req), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rq_dev, rq, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &rs, KQ * sizeof(pm_result), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &rs_dev, rs, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &ops, cap * sizeof(pm_delta), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &ops_dev, ops, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &ents, cap * sizeof(pm_entry), cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &ents_dev, ents, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &busy, NW * 4, cudaHostAllocMapped)); CUDA_CHECK(cudaHostGetDevicePointer((void **) &busy_dev, busy, 0));
        int bad0 = g_bad, nreq = 0, nops = 0, nid = 0;
        for (int it = 0; it < 300; ++it) {
            pv2_heap h = random_heap(rng, it % 10 == 0 ? 2000 : 200);
            std::vector<pm_entry> sent = pm_from_heap(h); uint32_t cnt = (uint32_t) sent.size();
            cudaMemcpy(v_dev, sent.data(), sent.size() * sizeof(pm_entry), cudaMemcpyHostToDevice); cudaMemcpy(cnt_dev, &cnt, 4, cudaMemcpyHostToDevice);
            pv2_heap card = h;   // what the card holds after each batch
            for (int round = 0; round < 3; ++round) {
                pm_batch bt; bt.E = E; bt.busy = busy_dev; bt.rebuild = round == 0;
                if (round > 0) {   // host changes: heats, pins, experts gone, newcomers in free bytes
                    pv2_heap hh = card;
                    std::vector<size_t> xs; for (const auto & [x, r] : hh.res) { if (r.id != PV2_HOLD) { xs.push_back(x); } }
                    for (size_t x : xs) {
                        const int c = (int) (rng() % 100); pv2_res & r = hh.res.at(x);
                        if (c < 10) { r.heat = (int64_t) (rng() % 1000); } else if (c < 15) { r.pin = !r.pin; } else if (c < 18) { pv2_drop(hh, x); }
                    }
                    std::vector<std::pair<size_t, size_t>> gs(hh.gaps.begin(), hh.gaps.end());
                    for (const auto & [gs0, gn] : gs) { if (rng() % 3 == 0 && gn >= 10) { pv2_res r; r.n = 10; r.u = 10; r.id = 5000000 + nid++; r.heat = (int64_t) (rng() % 1000); pv2_add(hh, (gs0 + 9) / 10 * 10, r); } }
                    std::vector<pm_delta> dv; std::vector<pm_entry> ev;
                    pm_diff(pm_from_heap(card), pm_from_heap(hh), dv, ev);
                    std::copy(dv.begin(), dv.end(), ops); std::copy(ev.begin(), ev.end(), ents);
                    bt.ops = ops_dev; bt.ents = ents_dev; bt.n_ops = (uint32_t) dv.size(); nops += (int) dv.size();
                    card = hh;
                }
                memset(busy, 0, NW * 4);
                for (uint32_t l = 0; l < 30; ++l) { if (rng() % 4 == 0) { busy[l >> 5] |= 1u << (l & 31); } }   // ids < 30000: layers 0..29
                const uint32_t k = 1 + rng() % KQ;
                for (uint32_t j = 0; j < k; ++j) { rq[j] = { (1 + rng() % 2) * 10, 10, (int64_t) (rng() % 1000), (int32_t) (2000000 + it * 64 + round * KQ + j), (int32_t) (rng() % 2 ? 1 : 0), (int32_t) (j % 7 == 6 ? 1 : 0) }; }
                memset(rs, 0, KQ * sizeof(pm_result));
                ggml_cuda_poolmap_alloc_batch_fork(0, v_dev, cnt_dev, cap, h.lo, h.hi, rq_dev, k, rs_dev, nullptr, nullptr, 1, 0, nullptr, g_scr ? scr : nullptr, bt);
                CUDA_CHECK(cudaDeviceSynchronize());
                pv2_heap ref = card;   // the reference sees busy layers as pinned
                for (auto & [x, r] : ref.res) { if (r.id != PV2_HOLD && r.id >= 0 && r.id < 30000 && ((busy[(r.id / E) >> 5] >> ((r.id / E) & 31)) & 1u)) { r.pin = true; } }
                for (uint32_t j = 0; j < k; ++j) {
                    const pv2_alloc_result rr = pv2_alloc_do(ref, rq[j].n, 10, rq[j].id, rq[j].hot, rq[j].pin & 1, true, rq[j].kind);
                    const pv2_alloc_result kr = pm_report(rs[j]);
                    bool rep = rs[j].done == 1 && rs[j].overflow == 0 && kr.s == rr.s && kr.moves.size() == rr.moves.size() && kr.freed.size() == rr.freed.size();
                    for (size_t q = 0; rep && q < kr.freed.size(); ++q) rep = kr.freed[q].s == rr.freed[q].s;
                    for (size_t q = 0; rep && q < kr.moves.size(); ++q) rep = kr.moves[q].from == rr.moves[q].from && kr.moves[q].to == rr.moves[q].to;
                    if (!rep && g_bad - bad0 < 5) printf("  delta it %d round %d req %u n %zu: kernel s %zu fr %zu mv %zu path %u ovf %u | ref s %zu fr %zu mv %zu\n", it, round, j, (size_t) rq[j].n, kr.s, kr.freed.size(), kr.moves.size(), rs[j].path, rs[j].overflow, rr.s, rr.freed.size(), rr.moves.size());
                    check("delta: report equals reference", rep);
                    check("delta: mirror replays", pv2_alloc_apply(card, kr, rq[j].n, 10, rq[j].id, rq[j].hot, rq[j].pin & 1, rq[j].kind));
                    nreq++;
                }
                cudaMemcpy(&cnt, cnt_dev, 4, cudaMemcpyDeviceToHost); std::vector<pm_entry> back(cnt); cudaMemcpy(back.data(), v_dev, cnt * sizeof(pm_entry), cudaMemcpyDeviceToHost);
                size_t d = 0; const bool sm = pm_same(back, pm_from_heap(card), &d);
                bool ex = sm; const std::vector<pm_entry> cv = pm_from_heap(card); for (size_t i = 0; ex && i < cv.size(); ++i) { ex = pm_eq(back[i], cv[i]); }
                if (!ex && g_bad - bad0 < 8) printf("  delta it %d round %d: card map differs at %zu\n", it, round, d);
                check("delta: card map equals the mirror, every field", ex);
            }
        }
        printf("delta: %d requests after %d host changes, %d failures\n", nreq, nops, g_bad - bad0);
    }
    printf("runs %d moves %d freed %d none %d; alloc+apply on ~1000-entry maps %.1f us avg\n", runs, moved, freed, none, 1000.0 * tsum / std::max(1, tn));
    printf(g_bad ? "FAIL: %d failure(s)\n" : "OK: 0 failure(s)\n", g_bad);
    return g_bad != 0;
}
