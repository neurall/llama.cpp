// memlat: idle DRAM load-to-use latency by pointer chasing a random cyclic permutation of cache lines over a buffer
// far bigger than L3 (default 1 GiB, transparent huge pages so TLB misses don't count), plus the same with N
// streaming reader threads loading memory (latency under bandwidth load, what a decode-time miss sees).
// Build: cc -O2 -pthread tools/moe-bench/memlat.c -o memlat ; run: ./memlat [MiB] [loaded threads]
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

static atomic_int stop;
static uint8_t * bw_buf; static size_t bw_n;
static void * reader(void * arg) {
    size_t o = (size_t) (uintptr_t) arg * (bw_n / 16) & ~(size_t) 63; uint64_t x = 0;
    while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
        for (size_t i = 0; i < (64u << 20); i += 64) { x += *(volatile uint64_t *) (bw_buf + (o + i) % bw_n); }
        o += 64u << 20;
    }
    return (void *) (uintptr_t) x;
}

static double chase(void ** p, size_t steps) {
    const double t0 = now_ns();
    for (size_t i = 0; i < steps; ++i) { p = (void **) *p; }
    const double dt = now_ns() - t0;
    if (!p) { puts(""); }
    return dt / steps;
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) atoi(argv[1]) : 1024, n = mib << 20, lines = n / 64;
    const int nload = argc > 2 ? atoi(argv[2]) : 4;
    uint8_t * buf = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(buf, n, MADV_HUGEPAGE);
    memset(buf, 1, n);
    size_t * perm = malloc(lines * sizeof(size_t));
    for (size_t i = 0; i < lines; ++i) { perm[i] = i; }
    srand(1);
    for (size_t i = lines - 1; i > 0; --i) { size_t j = ((size_t) rand() << 16 ^ rand()) % (i + 1), t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (size_t i = 0; i < lines; ++i) { *(void **) (buf + perm[i] * 64) = buf + perm[(i + 1) % lines] * 64; }
    free(perm);
    void ** p = (void **) buf;
    chase(p, 1u << 20);
    printf("idle latency: %.1f ns (%zu MiB, random lines)\n", chase(p, 1u << 23), mib);
    bw_n = 2048u << 20; bw_buf = mmap(NULL, bw_n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(bw_buf, bw_n, MADV_HUGEPAGE); memset(bw_buf, 2, bw_n);
    for (int k = 1; k <= nload; k *= 2) {
        pthread_t th[64]; atomic_store(&stop, 0);
        for (int t = 0; t < k; ++t) { pthread_create(&th[t], NULL, reader, (void *) (uintptr_t) t); }
        chase(p, 1u << 18);
        const double lat = chase(p, 1u << 22);
        atomic_store(&stop, 1);
        for (int t = 0; t < k; ++t) { pthread_join(th[t], NULL); }
        printf("latency with %d streaming readers: %.1f ns\n", k, lat);
    }
    return 0;
}
