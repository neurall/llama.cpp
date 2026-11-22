// fork: the card's pool-map alloc kernel against its host reference (pv2_alloc_do) on random maps: same report, same map, canonical; the host mirror
// replays the report to the same map. Skips (passes) without a CUDA device.
#include "../src/llama-poolmap-fork.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <random>

typedef void * (*alloc_t)(size_t, void **);
typedef void (*pm_alloc_t)(ggml_backend_t, pm_entry *, uint32_t *, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, int32_t, int64_t, int, pm_result *, char *, int32_t * const *, int32_t);
typedef void (*pm_io_t)(ggml_backend_t, void *, void *, size_t, int);

static int g_bad = 0;
static void check(const char * n, bool ok) { if (!ok) { printf("FAIL: %s\n", n); g_bad++; } }

static pv2_heap random_heap(std::mt19937 & rng, size_t slots) {
    pv2_heap h = pv2_make(0, slots * 10);
    for (size_t i = 0; i < slots; ) {
        const size_t k = 1 + rng() % 3, n = std::min(k, slots - i) * 10, x = i * 10;
        const int c = (int) (rng() % 100);
        pv2_res r; r.n = n; r.heat = (int64_t) (rng() % 1000); r.u = 10;
        if (c < 15) { i += k; continue; }
        if (c < 30) { r.id = PV2_HOLD; r.pin = true; r.mov = true; r.heat = 0; }
        else if (c < 40) { r.id = PV2_HOLD; r.pin = true; r.heat = 0; }
        else { r.id = (int32_t) x; r.pin = (int) (rng() % 100) < 20; }
        pv2_add(h, x, r);
        i += k;
    }
    return h;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) { printf("no CUDA device: skipped\nOK: 0 failure(s)\n"); return 0; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    auto al = (alloc_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mapped_alloc_fork");
    auto pa = (pm_alloc_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_poolmap_alloc_fork");
    auto io = (pm_io_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_poolmap_io_fork");
    if (!al || !pa || !io) { printf("FAIL: proc addresses missing\n"); return 1; }
    const uint32_t cap = 4096;
    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_type(dev) ? ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), cap * sizeof(pm_entry) + 256) : nullptr;
    char * vbase = (char *) ggml_backend_buffer_get_base(buf);
    pm_entry * v_dev = (pm_entry *) vbase; uint32_t * cnt_dev = (uint32_t *) (vbase + cap * sizeof(pm_entry));
    void * res_dev = nullptr; pm_result * res = (pm_result *) al(sizeof(pm_result), &res_dev);
    std::mt19937 rng(31337);
    int same_rep = 0, runs = 0, moved = 0, freed = 0, none = 0;
    for (int it = 0; it < 1500; ++it) {
        const size_t slots = it % 10 == 0 ? 600 : 40;   // now and then a map bigger than one block's threads
        pv2_heap h = random_heap(rng, slots);
        std::vector<pm_entry> v = pm_from_heap(h);
        uint32_t cnt = (uint32_t) v.size();
        io(be, v_dev, v.data(), v.size() * sizeof(pm_entry), 1);
        io(be, cnt_dev, &cnt, sizeof(cnt), 1);
        const uint64_t n = (1 + rng() % 4) * 10;
        const int32_t id = 1000000 + it;
        memset(res, 0, sizeof(pm_result));
        pa(be, v_dev, cnt_dev, cap, h.lo, h.hi, n, 10, id, 7, 1, (pm_result *) res_dev, nullptr, nullptr, 1);
        ggml_backend_synchronize(be);
        check("kernel reports done", res->done == 1);
        check("kernel no overflow", res->overflow == 0);
        pv2_heap ref = h;
        const pv2_alloc_result rr = pv2_alloc_do(ref, n, 10, id, 7, true);
        const pv2_alloc_result kr = pm_report(*res);
        runs++;
        if (rr.s == PV2_NONE) { none++; }
        moved += (int) kr.moves.size(); freed += (int) kr.freed.size();
        bool rep = kr.s == rr.s && kr.moves.size() == rr.moves.size() && kr.freed.size() == rr.freed.size();
        for (size_t k = 0; rep && k < kr.moves.size(); ++k) { rep = kr.moves[k].from == rr.moves[k].from && kr.moves[k].to == rr.moves[k].to && kr.moves[k].id == rr.moves[k].id; }
        for (size_t k = 0; rep && k < kr.freed.size(); ++k) { rep = kr.freed[k].s == rr.freed[k].s && kr.freed[k].id == rr.freed[k].id; }
        same_rep += rep;
        if (!rep && g_bad < 5) { printf("  report differs: it %d n %zu kernel s %zu mv %zu fr %zu, ref s %zu mv %zu fr %zu\n", it, (size_t) n, kr.s, kr.moves.size(), kr.freed.size(), rr.s, rr.moves.size(), rr.freed.size()); g_bad++; }
        // the card's map read back (the debug check): equals the reference and the host mirror replaying the report
        io(be, cnt_dev, &cnt, sizeof(cnt), 0);
        std::vector<pm_entry> back(cnt);
        io(be, v_dev, back.data(), cnt * sizeof(pm_entry), 0);
        size_t d = 0;
        const bool same_map = pm_same(back, pm_from_heap(ref), &d);
        if (!same_map && g_bad < 8) { printf("  map differs at entry %zu (it %d)\n", d, it); }
        check("card map equals reference map", same_map);
        check("card map canonical", pm_canonical(back, h.lo, h.hi));
        pv2_heap mirror = h;
        check("mirror replays the card's report", pv2_alloc_apply(mirror, kr, n, 10, id, 7, true) && pm_same(pm_from_heap(mirror), back));
    }
    check("every report equals the reference", same_rep == runs);
    check("moves, frees and misses seen", moved > 100 && freed > 100 && none > 10);
    printf("runs %d, moves %d, freed %d, none %d\n", runs, moved, freed, none);
    printf(g_bad ? "FAIL: %d failure(s)\n" : "OK: 0 failure(s)\n", g_bad);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return g_bad != 0;
}
