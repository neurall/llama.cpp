// streambench: simulate streaming prompt processing on a small-VRAM GPU: read every MoE expert tensor of the model (up, gate, down of each layer, in layer order)
// from the file with io_uring O_DIRECT into pinned slots and DMA every chunk to VRAM. order = file order or a random order of the chunks within each layer.
// list file: "layer name offset bytes" per tensor (made from the gguf).     build: nvcc -O2 -std=c++17 -o streambench streambench.cu -lpthread -luring
// run: ./streambench FILE LIST DEV CHUNK_KB ORDER(0 file,1 random per layer) THREADS QD [MAX_LAYERS [PICK_PCT [MODE]]]
// PICK_PCT: only that share of the 256 experts of each layer is read (same random picks for up, gate, down), MODE: 0 picked experts sorted, adjacent ones merged into runs of at most CHUNK_KB,
// 1 each picked expert one read, 2 picked experts cut into CHUNK_KB pieces. ORDER 1 shuffles the reads of a layer. The GB/s printed counts only bytes read.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <mutex>
#include <thread>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Item { size_t foff, n, doff; int layer; };
int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 1; }
    const char * file = argv[1]; const size_t CH = (size_t) atoi(argv[4]) << 10; const int order = atoi(argv[5]), TH = atoi(argv[6]), QD = atoi(argv[7]); const int maxl = argc > 8 ? atoi(argv[8]) : 1000, pick = argc > 9 ? atoi(argv[9]) : 100, mode = argc > 10 ? atoi(argv[10]) : 2;
    CK(cudaSetDevice(atoi(argv[3]))); int fdd = open(file, O_RDONLY | O_DIRECT), fdb = open(file, O_RDONLY);
    std::vector<Item> items; std::vector<size_t> layer_first; size_t dmax = 0, total = 0; int cur = -1, nl = 0;
    const char * ib = getenv("IOB"); const int iob = ib && !strcmp(ib, "pread") ? 1 : ib && !strcmp(ib, "aio") ? 2 : 0;   // IOB=uring (default) | pread (blocking, use many THREADS) | aio (kernel aio syscalls)
    size_t SLOT = CH;
    { FILE * f = fopen(argv[2], "r"); int l; char nm[16]; size_t off, len; size_t doff = 0; std::vector<char> pk(256, 1);
      while (fscanf(f, "%d %15s %zu %zu", &l, nm, &off, &len) == 4) {
          if (l != cur) { if (nl >= maxl) { break; } cur = l; nl++; doff = 0; layer_first.push_back(items.size()); std::mt19937_64 pr(1000 + l); std::vector<int> ix(256); for (int e = 0; e < 256; e++) ix[e] = e; std::shuffle(ix.begin(), ix.end(), pr); for (int e = 0; e < 256; e++) pk[ix[e]] = e < 256 * pick / 100; }
          const size_t ex = len / 256; int e = 0;
          while (e < 256) {
              if (!pk[e]) { e++; continue; }
              int e2 = e; if (mode == 0) { while (e2 + 1 < 256 && pk[e2 + 1]) e2++; }   // merged run e..e2
              const size_t o0 = off + (size_t) e * ex, rl = (size_t) (e2 - e + 1) * ex; const size_t step = mode == 1 ? ex : CH;
              for (size_t o = 0; o < rl; o += step) { const size_t n = std::min(step, rl - o); items.push_back({o0 + o, n, doff, l}); doff += n; total += n; SLOT = std::max(SLOT, n); }
              e = e2 + 1; }
          dmax = std::max(dmax, doff); }
      fclose(f); }
    layer_first.push_back(items.size());
    // REGION_MB (env): instead of the whole model, read only the first REGION_MB of the expert bytes, again and again as passes (a pass plays the part of a layer) until TOTAL_GB (default 12) were read,
    // so every order reads exactly the same blocks and only the order, or the size of the address range, changes
    if (getenv("REGION_MB")) {
        const size_t RB = (size_t) atol(getenv("REGION_MB")) << 20, TB = (size_t) (atof(getenv("TOTAL_GB") ? getenv("TOTAL_GB") : "12") * 1e9); std::vector<Item> reg; size_t acc = 0;
        for (const Item & m : items) { if (acc + m.n > RB) { break; } reg.push_back(m); acc += m.n; }
        std::vector<Item> ni; std::vector<size_t> nf; total = 0; dmax = acc; nl = 0;
        for (size_t p = 0; total < TB; p++) { nf.push_back(ni.size()); size_t d = 0; for (Item m : reg) { m.doff = d; d += m.n; m.layer = (int) p; ni.push_back(m); total += m.n; } nl++; }
        nf.push_back(ni.size()); items = ni; layer_first = nf;
    }
    // ORDER 2: shuffle only inside windows of 64 consecutive reads (keeps the layer roughly sequential); ORDER 3: stride 64 (reads i, i+64, i+128 ... so neighbours are never in flight together)
    // ORDER 4: ping-pong, first read, last read, second, second to last ... (consecutive reads as far apart as possible)
    if (order == 4) { for (int k = 0; k + 1 < (int) layer_first.size(); k++) { const size_t a = layer_first[k], b = layer_first[k + 1]; std::vector<Item> t; for (size_t i = 0, j = b - a; i < j; ) { t.push_back(items[a + i++]); if (i < j) { t.push_back(items[a + --j]); } } std::copy(t.begin(), t.end(), items.begin() + a); } }
    // ORDER 5: stride STRIDE (env, in chunks): pass r reads chunks r, r+STRIDE, r+2*STRIDE ... so reads in flight are STRIDE-1 chunks apart, the next pass fills the gaps
    else if (order == 2 || order == 3 || order == 5) { std::mt19937_64 rng(5); for (int k = 0; k + 1 < (int) layer_first.size(); k++) { const size_t a = layer_first[k], b = layer_first[k + 1];
        if (order == 2) { for (size_t w = a; w < b; w += 64) { std::shuffle(items.begin() + w, items.begin() + std::min(b, w + 64), rng); } }
        else { std::vector<Item> t; const size_t S = order == 5 ? (size_t) atoi(getenv("STRIDE") ? getenv("STRIDE") : "1") : 64; for (size_t r = 0; r < S; r++) for (size_t i = a + r; i < b; i += S) t.push_back(items[i]); std::copy(t.begin(), t.end(), items.begin() + a); } } }
    else if (order) { std::mt19937_64 rng(5); for (int k = 0; k + 1 < (int) layer_first.size(); k++) { std::shuffle(items.begin() + layer_first[k], items.begin() + layer_first[k + 1], rng); } }
    void * dev; CK(cudaMalloc(&dev, dmax));
    { void * w; CK(cudaMalloc(&w, 1 << 20)); uint8_t * wp; CK(cudaHostAlloc((void **) &wp, 1 << 20, 0)); for (int i = 0; i < 8; i++) CK(cudaMemcpy(w, wp, 1 << 20, cudaMemcpyHostToDevice)); }
    if (system("sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'")) { return 2; }
    std::mutex lat_mu; std::vector<double> lat_all;
    std::atomic<size_t> nx{0}; std::atomic<int> ready{0}, bad{0}; std::atomic<bool> go{false}; std::vector<std::thread> th; std::vector<double> layer_done(layer_first.size(), 0.0); double t0 = 0;
    std::vector<std::atomic<size_t>> remaining(layer_first.size()); for (size_t k = 0; k + 1 < layer_first.size(); k++) { remaining[k] = layer_first[k + 1] - layer_first[k]; }
    std::vector<int> layer_idx(items.size()); for (size_t k = 0; k + 1 < layer_first.size(); k++) for (size_t i = layer_first[k]; i < layer_first[k + 1]; i++) layer_idx[i] = (int) k;
    for (int t = 0; t < TH; t++) th.emplace_back([&]() {
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking)); struct io_uring ring; const char * iu = getenv("IOU"); const bool sqp = iu && strstr(iu, "sqpoll"), fixedbuf = iu && strstr(iu, "fixed"), batch = iu && strstr(iu, "batch");   // batch: submit QD reads in one call, wait for all of them, then the next QD   // IOU=sqpoll,fixed: kernel submission thread, registered (pinned) buffers
        { struct io_uring_params pr; memset(&pr, 0, sizeof(pr)); if (sqp) { pr.flags |= IORING_SETUP_SQPOLL; pr.sq_thread_idle = 2000; } if (io_uring_queue_init_params(QD, &ring, &pr) < 0) { perror("io_uring_queue_init_params"); exit(1); } }
        std::vector<uint8_t *> buf(QD); std::vector<double> t_sub(QD, 0.0), lat_local; std::vector<cudaEvent_t> ev(QD); std::vector<size_t> it(QD);
        for (int s = 0; s < QD; s++) { CK(cudaHostAlloc((void **) &buf[s], SLOT + 4096, 0)); CK(cudaEventCreateWithFlags(&ev[s], cudaEventDisableTiming)); CK(cudaEventRecord(ev[s], st)); }
        if (fixedbuf) { std::vector<iovec> iv(QD); for (int s = 0; s < QD; s++) { iv[s] = {buf[s], SLOT + 4096}; } if (io_uring_register_buffers(&ring, iv.data(), QD) < 0) { perror("io_uring_register_buffers"); exit(1); } }
        uint8_t * ref = (uint8_t *) aligned_alloc(4096, SLOT + 4096); ready++; while (!go.load()) { std::this_thread::yield(); }
        auto done_item = [&](int s) { const Item & m = items[it[s]]; if (t_sub[s] > 0) { lat_local.push_back(now() - t_sub[s]); }
            CK(cudaMemcpyAsync((uint8_t *) dev + m.doff, buf[s], m.n, cudaMemcpyHostToDevice, st)); CK(cudaEventRecord(ev[s], st));
            if (it[s] % 2003 == 0 && pread(fdb, ref, m.n, (off_t) m.foff) == (ssize_t) m.n && memcmp(ref, buf[s], m.n)) { bad++; }   // sampled content check against a buffered read
            if (--remaining[layer_idx[it[s]]] == 0) { layer_done[layer_idx[it[s]]] = now() - t0; } };
        // lat_local: submit-to-completion time of every read of this thread (io_uring path), merged at the end
        if (iob == 1) {   // blocking pread, one request per thread at a time (slots only overlap the DMA)
            int s = 0; for (;;) { const size_t i = nx.fetch_add(1); if (i >= items.size()) { break; } CK(cudaEventSynchronize(ev[s])); it[s] = i; const Item & m = items[i];
                if (pread(fdd, buf[s], (m.n + 4095) / 4096 * 4096, (off_t) m.foff) < (ssize_t) m.n) { fprintf(stderr, "short pread\n"); exit(1); } done_item(s); s = (s + 1) % QD; }
        } else if (iob == 2) {   // kernel aio (io_submit / io_getevents)
            aio_context_t ctx = 0; if (syscall(SYS_io_setup, QD, &ctx)) { perror("io_setup"); exit(1); } std::vector<iocb> cb(QD); std::vector<io_event> evs(QD); int infl = 0;
            auto asub = [&](int s) { const size_t i = nx.fetch_add(1); if (i >= items.size()) { return false; } CK(cudaEventSynchronize(ev[s])); it[s] = i; memset(&cb[s], 0, sizeof(iocb));
                cb[s].aio_lio_opcode = IOCB_CMD_PREAD; cb[s].aio_fildes = fdd; cb[s].aio_buf = (uint64_t) buf[s]; cb[s].aio_nbytes = (items[i].n + 4095) / 4096 * 4096; cb[s].aio_offset = (int64_t) items[i].foff; cb[s].aio_data = (uint64_t) s;
                iocb * p = &cb[s]; if (syscall(SYS_io_submit, ctx, 1, &p) != 1) { perror("io_submit"); exit(1); } infl++; return true; };
            for (int s = 0; s < QD; s++) { asub(s); }
            while (infl > 0) { const long n = syscall(SYS_io_getevents, ctx, 1, QD, evs.data(), nullptr); if (n < 0) { perror("io_getevents"); exit(1); }
                for (long k = 0; k < n; k++) { const int s = (int) evs[k].data; if (evs[k].res < (long) items[it[s]].n) { fprintf(stderr, "short aio %ld\n", (long) evs[k].res); exit(1); } infl--; done_item(s); asub(s); } }
            syscall(SYS_io_destroy, ctx);
        } else {
        int infl = 0; std::vector<int> pend;
        auto sub = [&](int s) { const size_t i = nx.fetch_add(1); if (i >= items.size()) { return false; } CK(cudaEventSynchronize(ev[s])); it[s] = i;
            t_sub[s] = now(); struct io_uring_sqe * q = io_uring_get_sqe(&ring); if (fixedbuf) { io_uring_prep_read_fixed(q, fdd, buf[s], (items[i].n + 4095) / 4096 * 4096, items[i].foff, s); } else { io_uring_prep_read(q, fdd, buf[s], (items[i].n + 4095) / 4096 * 4096, items[i].foff); } q->user_data = (unsigned long) s; infl++; return true; };
        for (int s = 0; s < QD; s++) { sub(s); } io_uring_submit(&ring);
        while (infl > 0) {
            struct io_uring_cqe * c; io_uring_wait_cqe(&ring, &c); const int s = (int) c->user_data; const Item & m = items[it[s]]; if (c->res < (int) m.n) { fprintf(stderr, "short read %d of %zu\n", c->res, m.n); exit(1); } io_uring_cqe_seen(&ring, c); infl--;
            done_item(s);
            if (!batch) { if (sub(s)) { io_uring_submit(&ring); } }
            else { pend.push_back(s); if (infl == 0) { bool any = false; for (int q : pend) { any |= sub(q); } pend.clear(); if (any) { io_uring_submit(&ring); } } }
        }
        }
        CK(cudaStreamSynchronize(st)); io_uring_queue_exit(&ring); { std::lock_guard<std::mutex> lk(lat_mu); lat_all.insert(lat_all.end(), lat_local.begin(), lat_local.end()); }
    });
    while (ready.load() < TH) { std::this_thread::yield(); } t0 = now(); go = true;
    for (auto & x : th) { x.join(); } const double dt = now() - t0;
    std::vector<double> lt; for (size_t k = 0; k + 1 < layer_first.size(); k++) { lt.push_back(layer_done[k]); }
    double first = lt.empty() ? 0 : lt[0], mid = lt.size() > 2 ? lt[lt.size() / 2] - lt[lt.size() / 2 - 1] : 0, last = lt.size() > 1 ? lt.back() - lt[lt.size() - 2] : 0;
    printf("%s[%s] chunk %4zuK pick %3d%% mode %d %s thr %d qd %2d | %d layers %.1f GB in %.1f s = %.2f GB/s | layer time first %.2f s, middle %.2f s, last %.2f s | content %s\n", "stream", iob == 1 ? "pread" : iob == 2 ? "aio" : (std::string("uring") + (getenv("IOU") && strstr(getenv("IOU"), "sqpoll") ? "_sqpoll" : "") + (getenv("IOU") && strstr(getenv("IOU"), "fixed") ? "_fixed" : "") + (getenv("IOU") && strstr(getenv("IOU"), "batch") ? "_batch" : "")).c_str(), CH >> 10, pick, mode, order == 1 ? "random-in-layer" : order == 2 ? "window-shuffled" : order == 3 ? "stride-64      " : order == 4 ? "ping-pong      " : order == 5 ? ("stride-" + std::string(getenv("STRIDE") ? getenv("STRIDE") : "1")).c_str() : "file-order     ", TH, QD, nl, total / 1e9, dt, total / dt / 1e9, first, mid, last, bad ? "MISMATCH" : "ok (sampled)");
    if (!lat_all.empty()) { std::sort(lat_all.begin(), lat_all.end()); auto pc = [&](double q) { return lat_all[(size_t) (q * (lat_all.size() - 1))] * 1e3; }; printf("   lat ms p50 %.2f p95 %.2f p99 %.2f\n", pc(0.5), pc(0.95), pc(0.99)); }
    return 0;
}
