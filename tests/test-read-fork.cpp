// Unit tests of the read wrapper's several-drive mode (ggml/src/ggml-read-fork.h, llama-server --multissd): which file holds a block, which drive a block goes
// to, the switch's text, and (scheduling) that the least-bytes-in-flight pick reads drives of different speeds at their summed rate.
// With SR_TEST_DIRS=DIR_A,DIR_B (two directories on different drives that take O_DIRECT) it also reads a file through a trimmed copy and checks every byte.
//
// build: c++ -std=c++17 -O2 -Iggml/include -Iggml/src tests/test-read-fork.cpp -o test-read-fork;  run: test-read-fork
#include "../ggml/src/ggml-read-fork.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;
static void report(const char * what, bool ok, const std::string & why = "") {
    printf("  %-74s %s%s%s\n", what, ok ? "OK" : "FAILED", why.empty() ? "" : "  ", why.c_str());
    if (!ok) { n_failed++; }
}

static void test_holds_pick_parse() {
    printf("Testing which file holds a block, the drive pick and the switch:\n");
    const std::vector<std::pair<uint64_t, uint64_t>> r = { { 0, 100 }, { 200, 300 } };
    report("a block inside a held range is held, across a hole or in one it is not",
           sr_holds(r, 10, 50) && sr_holds(r, 200, 300) && !sr_holds(r, 90, 110) && !sr_holds(r, 150, 160) && !sr_holds(r, 250, 301) && sr_holds({}, 5, 1 << 30));
    const bool h3[3] = { true, true, false }; const int d3[3] = { 0, 1, 2 };
    const uint64_t q1[3] = { 100, 50, 0 }, q2[3] = { 50, 50, 0 };
    const bool none[2] = { false, false };
    report("the least busy drive of those that hold the block; ties: the first; none: -1",
           sr_pick(3, h3, d3, q1) == 1 && sr_pick(3, h3, d3, q2) == 0 && sr_pick(2, none, d3, q1) == -1);
    const int same[2] = { 0, 0 };   // two files on one drive share its queue
    report("two files on one drive: the first (one queue)", sr_pick(2, h3, same, q1) == 0);
    auto v = sr_parse_ssd("/m:0.5,/m2/m:0.5");
    report("--multissd /m:0.5,/m2/m:0.5", v.size() == 2 && v[0].first == "/m" && v[0].second == 0.5 && v[1].first == "/m2/m" && v[1].second == 0.5);
    v = sr_parse_ssd("C:\\m, D:\\m:0.41 ,/m2/k/");
    report("Windows paths keep the drive letter, the share is optional, trailing separators go",
           v.size() == 3 && v[0].first == "C:\\m" && v[0].second == 0.0 && v[1].first == "D:\\m" && v[1].second == 0.41 && v[2].first == "/m2/k" && v[2].second == 0.0);
    v = sr_parse_ssd("/a:2,,/b:0");
    report("a number outside (0, 1] is part of the name; empty entries go", v.size() == 2 && v[0].first == "/a:2" && v[1].first == "/b:0");
}

// Drives serving their reads one after another at their rates; one read of `total` bytes through sr_sched, the engines' issue order, qd of it in flight on each
// drive: the read's rate and each drive's share of the bytes. q = bytes in flight per drive, as g_sr_q.
struct sim_out { double gbs; double share[SR_MAX_SRC]; };
static sim_out simulate(const sr_srcs & S, const std::vector<double> & gbs, size_t total, int qd = 32) {
    const std::vector<sr_piece> pcs = sr_pieces(total, (size_t) 256 << 10, (size_t) 64 << 10);
    sr_sched sc(&S, pcs, 0, qd);
    std::vector<double> busy(SR_MAX_SRC, 0.0);
    uint64_t q[SR_MAX_SRC] = {}, b[SR_MAX_SRC] = {};
    struct fl { double done; int f; size_t n; };
    std::vector<fl> in;
    double now = 0.0;
    while (sc.left > 0 || !in.empty()) {
        int f; size_t j;
        while (sc.next(q, &f, &j)) {
            const int d = S.drive[f];
            busy[d] = std::max(now, busy[d]) + (double) pcs[j].n / (gbs[d] * 1e9);
            in.push_back({ busy[d], f, pcs[j].n });
            q[d] += pcs[j].n; b[d] += pcs[j].n;
        }
        if (in.empty()) { return { -1.0, {} }; }   // pieces no file holds
        size_t k = 0;
        for (size_t i = 1; i < in.size(); ++i) { if (in[i].done < in[k].done) { k = i; } }
        now = in[k].done;
        sc.done(in[k].f);
        q[S.drive[in[k].f]] -= in[k].n;
        in.erase(in.begin() + (long) k);
    }
    sim_out o = { (double) total / now / 1e9, {} };
    for (int d = 0; d < SR_MAX_SRC; ++d) { o.share[d] = (double) b[d] / (double) total; }
    return o;
}

using ranges = std::vector<std::pair<uint64_t, uint64_t>>;
static sr_srcs make_srcs(const std::vector<ranges> & held) {
    sr_srcs S;
    S.n = (int) held.size(); S.n_drives = S.n; S.size = (uint64_t) 1 << 40;
    for (int i = 0; i < S.n; ++i) { S.fd[i] = i; S.drive[i] = i; S.held[i] = held[i]; }
    return S;
}
// a copy of the experts fork/ssd_place.py gives it: expert e of n (esz bytes each, the tensor from 0) when frac((e + 0.5) * 0.618..) lies in [lo, hi), 4 KiB grid
static ranges placed(int n_exp, uint64_t esz, double lo, double hi) {
    ranges r;
    for (int e = 0; e < n_exp; ++e) {
        const double u = std::fmod((e + 0.5) * 0.6180339887498949, 1.0);
        if (u < lo || u >= hi) { continue; }
        const uint64_t a = (uint64_t) e * esz / 4096 * 4096, z = ((uint64_t) (e + 1) * esz + 4095) / 4096 * 4096;
        if (!r.empty() && a <= r.back().second) { r.back().second = z; } else { r.push_back({ a, z }); }
    }
    return r;
}

// R readers, each reading one expert at a time in index order (the pool's fill, a stream's ring of layers ahead), each read through its own sr_sched, all of
// them sharing the drives and the bytes in flight: the rate of the whole tensor
static double simulate_experts(const sr_srcs & S, const std::vector<double> & gbs, int n_exp, uint64_t esz, int readers) {
    struct rd { std::vector<sr_piece> pcs; std::unique_ptr<sr_sched> sc; int infl = 0; };
    std::vector<rd> R(readers);
    int next_e = 0;
    auto start = [&](rd & r) {
        r.sc.reset();
        if (next_e >= n_exp) { return; }
        const uint64_t a = (uint64_t) next_e * esz / 4096 * 4096, z = ((uint64_t) (next_e + 1) * esz + 4095) / 4096 * 4096;
        next_e++;
        r.pcs = sr_pieces(z - a, (size_t) 256 << 10, (size_t) 64 << 10);
        r.sc.reset(new sr_sched(&S, r.pcs, a, 32));
    };
    for (rd & r : R) { start(r); }
    std::vector<double> busy(SR_MAX_SRC, 0.0);
    uint64_t q[SR_MAX_SRC] = {};
    struct fl { double done; int r, f; size_t n; };
    std::vector<fl> in;
    double now = 0.0; uint64_t bytes = 0;
    for (;;) {
        for (int i = 0; i < readers; ++i) {
            int f; size_t j;
            while (R[i].sc && R[i].sc->next(q, &f, &j)) {
                const int d = S.drive[f]; const size_t n = R[i].pcs[j].n;
                busy[d] = std::max(now, busy[d]) + (double) n / (gbs[d] * 1e9);
                in.push_back({ busy[d], i, f, n });
                q[d] += n; R[i].infl++; bytes += n;
            }
        }
        if (in.empty()) { break; }
        size_t k = 0;
        for (size_t i = 1; i < in.size(); ++i) { if (in[i].done < in[k].done) { k = i; } }
        now = in[k].done;
        rd & r = R[in[k].r];
        r.sc->done(in[k].f); q[S.drive[in[k].f]] -= in[k].n; r.infl--;
        if (r.infl == 0 && r.sc->left == 0) { start(r); }
        in.erase(in.begin() + (long) k);
    }
    return (double) bytes / now / 1e9;
}

static void test_schedule() {
    printf("Testing the issue order as a schedule (PC1's drives: 6.1 and 4.3 GB/s, sum 10.4):\n");
    const std::vector<double> two = { 6.1, 4.3 };
    const size_t total = (size_t) 2 << 30;
    const int n_exp = 384; const uint64_t esz = total / n_exp;   // one tensor of experts, 5.3 MiB each
    auto gb = [](double x) { char t[32]; snprintf(t, sizeof t, "%.2f GB/s", x); return std::string(t); };
    const uint64_t cut = (uint64_t) (total * 0.59) / 4096 * 4096, M = (uint64_t) 1 << 20;   // parts that meet: 1 MiB both hold, so a piece across the cut has a file
    printf(" one read of the whole tensor (a range read):\n");
    sim_out o = simulate(make_srcs({ {}, {} }), two, total);
    report("whole copy: the summed rate (within 3%), the slower drive ~41% of the bytes", o.gbs > 0.97 * 10.4 && o.share[1] > 0.38 && o.share[1] < 0.44,
           gb(o.gbs) + ", " + std::to_string(o.share[1]).substr(0, 5));
    o = simulate(make_srcs({ {}, { { cut, total } } }), two, total);
    report("copy of the last 41% of the tensor: the summed rate (within 5%)", o.gbs > 0.95 * 10.4, gb(o.gbs));
    o = simulate(make_srcs({ {}, placed(n_exp, esz, 0.0, 0.41) }), two, total);
    report("copy of 41% of the experts, spread (ssd_place): above 85% of the sum (a piece across experts it lacks: the original)", o.gbs > 0.85 * 10.4, gb(o.gbs));
    o = simulate(make_srcs({ {}, placed(n_exp, esz, 0.0, 0.2) }), two, total);
    report("copy of 20%: faster than one drive, at most 1 / 0.8 of it", o.gbs > 6.1 * 1.05 && o.gbs < 6.1 / 0.8 * 1.02, gb(o.gbs));
    o = simulate(make_srcs({ { { 0, total / 2 + M } }, { { total / 2 - M, total } } }), two, total);
    report("each drive half the tensor (a trimmed original): 2 x 4.3 = 8.6 GB/s, 1.41x", std::abs(o.gbs - 8.6) < 0.3, gb(o.gbs));
    o = simulate(make_srcs({ { { 0, cut + M } }, { { cut - M, total } } }), two, total);
    report("parts by speed, 59 / 41: the summed rate (within 5%), 1.7x", o.gbs > 0.95 * 10.4, gb(o.gbs));
    o = simulate(make_srcs({ {}, {}, {}, {} }), { 6.1, 4.3, 5.0, 3.0 }, total);
    report("four drives, whole copies: the summed rate 18.4 (within 3%)", o.gbs > 0.97 * 18.4, gb(o.gbs));
    printf(" 8 readers, one expert each at a time, in index order (2.9 MB experts, as Kimi K3.1's):\n");
    const int ne = 1024; const uint64_t es = 2900000;
    const double w = simulate_experts(make_srcs({ {}, {} }), two, ne, es, 8);
    const double sp = simulate_experts(make_srcs({ {}, placed(ne, es, 0.0, 0.41) }), two, ne, es, 8);
    const double ct = simulate_experts(make_srcs({ {}, { { (uint64_t) (ne * es * 0.59) / 4096 * 4096, ne * es } } }), two, ne, es, 8);
    report("whole copy: the summed rate (within 5%)", w > 0.95 * 10.4, gb(w));
    report("copy of 41% spread: the summed rate (within 5%)", sp > 0.95 * 10.4, gb(sp));
    report("copy of the last 41%: slower, the copy idles through the first 59% (why ssd_place spreads)", ct < 0.85 * 10.4 && ct < sp, gb(ct));
    {   // one file: the pieces in order, qd in flight, as before the drives
        const std::vector<sr_piece> pcs = sr_pieces((size_t) 64 << 20, (size_t) 256 << 10, (size_t) 64 << 10);
        sr_sched sc(nullptr, pcs, 0, 32);
        int f; size_t j; bool order = true; int n = 0;
        while (sc.next(nullptr, &f, &j)) { order = order && f == 0 && j == (size_t) n; n++; }
        sc.done(0);
        const bool more = sc.next(nullptr, &f, &j) && j == 32;
        report("one file: pieces in order, 32 in flight, the next one when one finished", order && n == 32 && more);
    }
}

#if defined(__linux__)
// two directories on different drives: a 64 MiB file in A, its copy in B trimmed to the second half (a sparse file), both read through sr_read with --multissd B
static void test_files(const char * dirs) {
    const auto d = sr_parse_ssd(dirs);
    if (d.size() < 2) { printf("  (SR_TEST_DIRS needs two directories)\n"); return; }
    printf("Testing a read through a trimmed copy (%s, %s):\n", d[0].first.c_str(), d[1].first.c_str());
    const size_t N = (size_t) 64 << 20;
    const std::string a = d[0].first + "/sr-test.bin", b = d[1].first + "/sr-test.bin";
    std::vector<uint8_t> src(N);
    std::mt19937_64 rng(7);
    for (size_t i = 0; i < N; i += 8) { const uint64_t x = rng(); memcpy(&src[i], &x, 8); }
    FILE * fa = fopen(a.c_str(), "wb"); if (!fa) { report("write the original", false, a); return; }
    fwrite(src.data(), 1, N, fa); fclose(fa);
    const int fb = open(b.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644); if (fb < 0) { report("write the copy", false, b); return; }
    if (ftruncate(fb, (off_t) N) != 0 || pwrite(fb, src.data() + N / 2, N / 2, (off_t) (N / 2)) != (ssize_t) (N / 2)) { close(fb); report("write the copy", false); return; }
    fsync(fb); close(fb);
    const int fh = open(b.c_str(), O_RDONLY);
    const auto held = sr_data_ranges(fh, N);
    close(fh);
    report("the trimmed copy holds its second half only (SEEK_DATA)", held.size() == 1 && held[0].first <= N / 2 && held[0].first >= N / 2 - 65536 && held[0].second == N,
           held.empty() ? "no holes seen" : std::to_string(held[0].first) + ".." + std::to_string(held[0].second));
    setenv("LLAMA_MOE_SSD", d[1].first.c_str(), 1);
    const int fd = open_direct(a.c_str());
    if (fd < 0) { report("O_DIRECT on the original", false); return; }
    report("the copy is found, on another drive", sr_drives(fd) == 2);
    uint8_t * buf = (uint8_t *) aligned_alloc(4096, N);
    bool ok = true;
    for (int rep = 0; rep < 3 && ok; ++rep) { memset(buf, 0, N); ok = sr_read(fd, (char *) buf, 0, N, N) && memcmp(buf, src.data(), N) == 0; }
    report("every byte as in the original, three times over", ok);
    report("both drives read", g_sr_b[0].load() > 0 && g_sr_b[1].load() > 0, sr_drive_bytes());
    free(buf); close(fd);
    unlink(a.c_str()); unlink(b.c_str());
}
#endif

#if defined(__linux__)
// a read that reaches the end of the file returns fewer bytes than asked, off the 4096 grid: complete when it holds every byte needed (W10, 2026-11-27:
// the Windows loader's chunk at the file's end, 37472 of 40960 bytes, counted as failed and the model did not load)
static void test_tail_read() {
    printf("Testing a read up to the end of the file:\n");
    char path[] = "/tmp/test-read-tail-XXXXXX";
    const int fd = mkstemp(path);
    std::vector<char> data(4096 + 904);
    for (size_t i = 0; i < data.size(); ++i) { data[i] = (char) (i * 7 + 3); }
    const bool wrote = fd >= 0 && write(fd, data.data(), data.size()) == (ssize_t) data.size();
    std::vector<char> buf(8192, 0);
    const bool ok = wrote && sr_pread(fd, buf.data(), 4096, 8192, 904) && memcmp(buf.data(), data.data() + 4096, 904) == 0;
    report("the file's last 904 bytes asked as an aligned 8192-byte read: done, bytes right", ok);
    report("more needed than the file holds: failed", wrote && !sr_pread(fd, buf.data(), 4096, 8192, 1000));
    if (fd >= 0) { close(fd); unlink(path); }
}
#endif

int main() {
    test_holds_pick_parse();
    test_schedule();
#if defined(__linux__)
    test_tail_read();
    if (const char * t = getenv("SR_TEST_DIRS")) { test_files(t); }
#endif
    if (n_failed > 0) { printf("%d tests failed\n", n_failed); return 1; }
    printf("All tests passed\n");
    return 0;
}
