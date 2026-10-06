// iobench FILE [--grid] [--sec 0.25] [--engine uring|threads]: find the read settings of this machine's SSD (what fio finds, in 2-3 s).
// O_DIRECT random reads of aligned blocks, block size x queue depth (io_uring) or thread count (pread). Hill climb from 1 MiB / 8: move one
// parameter one step while the gain is at least 3%; the choice is the cheapest setting (fewest bytes in flight) that reaches 95% of the best bandwidth seen.
// --grid measures every point (to compare with fio). gcc -O2 -o iobench iobench.c -luring -lpthread
#define _GNU_SOURCE
#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const size_t BS[] = { 128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20 };
static const int    QD[] = { 1, 2, 4, 8, 16, 32, 64 };
#define NBS 6
#define NQD 7
static int g_fd; static uint64_t g_size; static double g_sec = 0.25; static int g_threads;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rnd(uint64_t * s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static int cmpd(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return (x > y) - (x < y); }

typedef struct { double gbs, p50_us; } res_t;

static res_t run_uring(size_t bs, int qd) {
    struct io_uring ring; io_uring_queue_init(qd, &ring, 0);
    char * buf = aligned_alloc(4096, bs * qd); memset(buf, 1, bs * qd);
    double * t0 = calloc(qd, sizeof(double)); uint64_t seed = 88172645463325252ull;
    size_t nblk = g_size / bs; double * lat = malloc(sizeof(double) * 1000000); size_t nl = 0, bytes = 0;
    double start = now();
    for (int i = 0; i < qd; i++) {
        struct io_uring_sqe * s = io_uring_get_sqe(&ring);
        io_uring_prep_read(s, g_fd, buf + (size_t) i * bs, bs, (rnd(&seed) % nblk) * bs); s->user_data = i; t0[i] = now();
    }
    io_uring_submit(&ring);
    while (now() - start < g_sec) {
        struct io_uring_cqe * c; if (io_uring_wait_cqe(&ring, &c)) break;
        int i = (int) c->user_data; if (c->res != (int) bs) { io_uring_cqe_seen(&ring, c); break; }
        io_uring_cqe_seen(&ring, c);
        double t = now(); if (nl < 1000000) lat[nl++] = (t - t0[i]) * 1e6; bytes += bs;
        struct io_uring_sqe * s = io_uring_get_sqe(&ring);
        io_uring_prep_read(s, g_fd, buf + (size_t) i * bs, bs, (rnd(&seed) % nblk) * bs); s->user_data = i; t0[i] = t; io_uring_submit(&ring);
    }
    double el = now() - start;
    // drain
    struct io_uring_cqe * c; while (io_uring_peek_cqe(&ring, &c) == 0) io_uring_cqe_seen(&ring, c);
    io_uring_queue_exit(&ring);
    res_t r = { bytes / el / 1e9, 0 }; if (nl) { qsort(lat, nl, sizeof(double), cmpd); r.p50_us = lat[nl / 2]; }
    free(buf); free(t0); free(lat); return r;
}

typedef struct { size_t bs; int id; double until; size_t bytes; } th_t;
static void * th_run(void * p) {
    th_t * a = p; char * buf = aligned_alloc(4096, a->bs); uint64_t seed = 88172645463325252ull + a->id * 7919; size_t nblk = g_size / a->bs;
    while (now() < a->until) { if (pread(g_fd, buf, a->bs, (rnd(&seed) % nblk) * a->bs) != (ssize_t) a->bs) break; a->bytes += a->bs; }
    free(buf); return NULL;
}
static res_t run_threads(size_t bs, int n) {
    pthread_t t[64]; th_t a[64]; double start = now();
    for (int i = 0; i < n; i++) { a[i] = (th_t) { bs, i, start + g_sec, 0 }; pthread_create(&t[i], NULL, th_run, &a[i]); }
    size_t bytes = 0; for (int i = 0; i < n; i++) { pthread_join(t[i], NULL); bytes += a[i].bytes; }
    res_t r = { bytes / (now() - start) / 1e9, 0 }; return r;
}
static res_t measure(int bi, int qi) { return g_threads ? run_threads(BS[bi], QD[qi]) : run_uring(BS[bi], QD[qi]); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: iobench FILE [--grid] [--sec S] [--engine uring|threads]\n"); return 1; }
    int grid = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--grid")) grid = 1;
        else if (!strcmp(argv[i], "--sec") && i + 1 < argc) g_sec = atof(argv[++i]);
        else if (!strcmp(argv[i], "--engine") && i + 1 < argc) g_threads = !strcmp(argv[++i], "threads");
    }
    g_fd = open(argv[1], O_RDONLY | O_DIRECT); if (g_fd < 0) { perror("open O_DIRECT"); return 1; }
    struct stat st; fstat(g_fd, &st); g_size = st.st_size;
    double T0 = now(); static res_t R[NBS][NQD]; static int done[NBS][NQD];
    printf("engine %s, %.2f s per point, file %.1f GiB\n%6s %4s %7s %8s\n", g_threads ? "threads(pread)" : "io_uring", g_sec, g_size / 1073741824.0, "bs", "qd", "GB/s", "p50 us");
    #define M(b, q) (done[b][q] ? R[b][q] : (done[b][q] = 1, R[b][q] = measure(b, q), printf("%5zuk %4d %7.2f %8.0f\n", BS[b] >> 10, QD[q], R[b][q].gbs, R[b][q].p50_us), fflush(stdout), R[b][q]))
    int b = 3, q = 3;
    if (grid) { for (int i = 0; i < NBS; i++) for (int j = 0; j < NQD; j++) (void) M(i, j); }
    else for (;;) {
        double cur = M(b, q).gbs; int nb = b, nq = q; double best = cur;
        int mv[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        for (int k = 0; k < 4; k++) {
            int ib = b + mv[k][0], iq = q + mv[k][1]; if (ib < 0 || ib >= NBS || iq < 0 || iq >= NQD) continue;
            double g = M(ib, iq).gbs; if (g > best * 1.03) { best = g; nb = ib; nq = iq; }
        }
        if (nb == b && nq == q) break;
        b = nb; q = nq;
    }
    double mx = 0; for (int i = 0; i < NBS; i++) for (int j = 0; j < NQD; j++) if (done[i][j] && R[i][j].gbs > mx) mx = R[i][j].gbs;
    int cb = -1, cq = -1; size_t cost = ~(size_t) 0;
    for (int i = 0; i < NBS; i++) for (int j = 0; j < NQD; j++) if (done[i][j] && R[i][j].gbs >= 0.95 * mx && BS[i] * QD[j] < cost) { cost = BS[i] * QD[j]; cb = i; cq = j; }
    printf("best %.2f GB/s; choice: %zu KiB x depth %d = %.2f GB/s (p50 %.0f us), %zu KiB in flight; %.1f s\n", mx, BS[cb] >> 10, QD[cq], R[cb][cq].gbs, R[cb][cq].p50_us, cost >> 10, now() - T0);
    return 0;
}
