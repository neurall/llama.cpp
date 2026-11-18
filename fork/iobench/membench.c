// membench: how much RAM read bandwidth this CPU reaches, by thread count, placement, load kind and page size (user 2026-11-18: "in theory we may be able
// to extract more mem bw? perhaps do simple c test code measuring it first"). Each thread is pinned to one CPU and reads its own contiguous slice of one big
// buffer; all threads start on a barrier, the slowest thread's time counts. Placement "spread" deals one thread per physical core round-robin over the L3
// groups (chiplets), as the engine's probe does; "packed" fills the first group first. Past the physical cores the SMT siblings follow in the same order.
// PC1 (3975WX, 8 x DDR4-3200, one NUMA node): the engine's probe reads 128-133 GB/s; the channels' theoretical peak is 204.8.
// Measured 2026-11-18 09:00 (4 GiB, best of 2): 8 threads packed on one chiplet 40 GB/s, 16 on two 79 (each chiplet's link to the I/O die); 8 spread over the
// four 103 (124 with 4 streams a thread), 16 spread 120-129, 24-64 spread 124-129 whatever the load kind; 2 MiB pages and prefetch change nothing. The ceiling is the
// DRAM at about 128 GB/s, 63% of the peak. Placement matters at low thread counts: fewer threads than cores must sit on different chiplets.
//
// build: cc -O2 -mavx2 -pthread fork/iobench/membench.c -o membench
// usage: membench [GiB=4] [reps=3]
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

enum { K_SCALAR, K_AVX2, K_AVX2_PF, K_AVX2_4S, K_N };
static const char * k_name[K_N] = { "scalar 8 B", "avx2 32 B", "avx2 + prefetch", "avx2, 4 streams" };

typedef struct { const char * p; size_t n; int cpu, kind; double t0, t1; uint64_t sink; } job_t;
static atomic_int g_ready, g_go;

static uint64_t rd_scalar(const char * p, size_t n) {
    const uint64_t * q = (const uint64_t *) p;
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i < n / 8; i += 4) { a += q[i]; b += q[i + 1]; c += q[i + 2]; d += q[i + 3]; }
    return a + b + c + d;
}

static uint64_t hsum(__m256i a) { uint64_t v[4]; _mm256_storeu_si256((__m256i *) v, a); return v[0] + v[1] + v[2] + v[3]; }

static uint64_t rd_avx2(const char * p, size_t n, int pf) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < n; i += 128) {
        if (pf) { _mm_prefetch(p + i + 2048, _MM_HINT_T0); _mm_prefetch(p + i + 2048 + 64, _MM_HINT_T0); }
        a0 = _mm256_add_epi64(a0, _mm256_load_si256((const __m256i *) (p + i)));
        a1 = _mm256_add_epi64(a1, _mm256_load_si256((const __m256i *) (p + i + 32)));
        a2 = _mm256_add_epi64(a2, _mm256_load_si256((const __m256i *) (p + i + 64)));
        a3 = _mm256_add_epi64(a3, _mm256_load_si256((const __m256i *) (p + i + 96)));
    }
    return hsum(_mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3)));
}

// the slice as 4 sub-slices read in lockstep: 4 sequential streams a thread (more DRAM pages open at once)
static uint64_t rd_avx2_4s(const char * p, size_t n) {
    const size_t q = n / 4;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < q; i += 64) {
        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(_mm256_load_si256((const __m256i *) (p + i)),         _mm256_load_si256((const __m256i *) (p + i + 32))));
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(_mm256_load_si256((const __m256i *) (p + q + i)),     _mm256_load_si256((const __m256i *) (p + q + i + 32))));
        a2 = _mm256_add_epi64(a2, _mm256_add_epi64(_mm256_load_si256((const __m256i *) (p + 2 * q + i)), _mm256_load_si256((const __m256i *) (p + 2 * q + i + 32))));
        a3 = _mm256_add_epi64(a3, _mm256_add_epi64(_mm256_load_si256((const __m256i *) (p + 3 * q + i)), _mm256_load_si256((const __m256i *) (p + 3 * q + i + 32))));
    }
    return hsum(_mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3)));
}

static void * worker(void * arg) {
    job_t * j = (job_t *) arg;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    atomic_fetch_add(&g_ready, 1);
    while (!atomic_load(&g_go)) { }
    j->t0 = now_s();
    switch (j->kind) {
        case K_SCALAR:  j->sink = rd_scalar(j->p, j->n); break;
        case K_AVX2:    j->sink = rd_avx2(j->p, j->n, 0); break;
        case K_AVX2_PF: j->sink = rd_avx2(j->p, j->n, 1); break;
        default:        j->sink = rd_avx2_4s(j->p, j->n); break;
    }
    j->t1 = now_s();
    return NULL;
}

static int read_list(const char * path, int * out, int cap) {   // "0-3,32-35"
    FILE * f = fopen(path, "r"); if (!f) { return 0; }
    char b[512]; int n = 0;
    if (fgets(b, sizeof b, f)) {
        for (char * t = strtok(b, ",\n"); t && n < cap; t = strtok(NULL, ",\n")) {
            int a, z; if (sscanf(t, "%d-%d", &a, &z) == 2) { for (int c = a; c <= z && n < cap; ++c) { out[n++] = c; } } else if (sscanf(t, "%d", &a) == 1) { out[n++] = a; }
        }
    }
    fclose(f); return n;
}

static int g_ncpu, g_order_spread[1024], g_order_packed[1024];

// one CPU per physical core per L3 group, groups in order; then the siblings
static void topology(void) {
    g_ncpu = (int) sysconf(_SC_NPROCESSORS_ONLN);
    static int grp_of[1024], first_sib[1024], ngrp = 0, gfirst[1024];
    for (int c = 0; c < g_ncpu; ++c) { grp_of[c] = -1; }
    for (int c = 0; c < g_ncpu; ++c) {
        char path[256]; int l[1024];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        const int ns = read_list(path, l, 1024); first_sib[c] = ns > 0 ? l[0] : c;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", c);
        const int nl = read_list(path, l, 1024); const int lead = nl > 0 ? l[0] : 0;
        int g = -1; for (int k = 0; k < ngrp; ++k) { if (gfirst[k] == lead) { g = k; } }
        if (g < 0) { g = ngrp; gfirst[ngrp++] = lead; }
        grp_of[c] = g;
    }
    int n = 0;
    for (int pass = 0; pass < 2; ++pass) {   // cores (first siblings), then the other siblings
        for (int round = 0; round < g_ncpu; ++round) {
            for (int g = 0; g < ngrp; ++g) {
                int seen = 0;
                for (int c = 0; c < g_ncpu; ++c) {
                    if (grp_of[c] != g || (pass == 0) != (first_sib[c] == c)) { continue; }
                    if (seen++ == round) { g_order_spread[n++] = c; }
                }
            }
        }
    }
    int m = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int g = 0; g < ngrp; ++g) { for (int c = 0; c < g_ncpu; ++c) { if (grp_of[c] == g && (pass == 0) == (first_sib[c] == c)) { g_order_packed[m++] = c; } } }
    }
    printf("%d CPUs, %d L3 groups; spread order:", g_ncpu, ngrp);
    for (int i = 0; i < n && i < 16; ++i) { printf(" %d", g_order_spread[i]); }
    printf(" ...\n");
}

static double run(const char * buf, size_t bytes, int nt, const int * order, int kind) {
    pthread_t th[1024]; job_t j[1024];
    const size_t per = (bytes / nt) & ~(size_t) 4095;
    atomic_store(&g_ready, 0); atomic_store(&g_go, 0);
    for (int i = 0; i < nt; ++i) { j[i] = (job_t) { buf + (size_t) i * per, per, order[i], kind, 0, 0, 0 }; pthread_create(&th[i], NULL, worker, &j[i]); }
    while (atomic_load(&g_ready) < nt) { }
    atomic_store(&g_go, 1);
    double t0 = 1e30, t1 = 0; uint64_t sink = 0;
    for (int i = 0; i < nt; ++i) { pthread_join(th[i], NULL); if (j[i].t0 < t0) { t0 = j[i].t0; } if (j[i].t1 > t1) { t1 = j[i].t1; } sink += j[i].sink; }
    if (sink == 42) { printf(" "); }
    return (double) per * nt / (t1 - t0) / 1e9;
}

int main(int argc, char ** argv) {
    const size_t gib = argc > 1 ? (size_t) atoi(argv[1]) : 4;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    const size_t bytes = gib << 30;
    topology();
    const int counts[] = { 4, 8, 16, 24, 32, 48, 64 };
    for (int huge = 0; huge < 2; ++huge) {
        char * buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (buf == MAP_FAILED) { perror("mmap"); return 1; }
        madvise(buf, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        for (size_t i = 0; i < bytes; i += 4096) { buf[i] = (char) i; }   // fault every page in
        printf("\n%s pages, %zu GiB, best of %d (GB/s)\n%-8s %-7s", huge ? "2 MiB" : "4 KiB", gib, reps, "threads", "place");
        for (int k = 0; k < K_N; ++k) { printf(" %16s", k_name[k]); }
        printf("\n");
        for (size_t ci = 0; ci < sizeof counts / sizeof counts[0]; ++ci) {
            const int nt = counts[ci];
            if (nt > g_ncpu) { continue; }
            for (int pl = 0; pl < 2; ++pl) {
                if (pl == 1 && (nt < 8 || nt > 16)) { continue; }   // packed only where it differs and matters: one or two chiplets' worth
                printf("%-8d %-7s", nt, pl ? "packed" : "spread");
                for (int k = 0; k < K_N; ++k) {
                    double best = 0;
                    for (int r = 0; r < reps; ++r) { const double g = run(buf, bytes, nt, pl ? g_order_packed : g_order_spread, k); if (g > best) { best = g; } }
                    printf(" %16.1f", best);
                    fflush(stdout);
                }
                printf("\n");
            }
        }
        munmap(buf, bytes);
    }
    return 0;
}
