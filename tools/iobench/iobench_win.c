// iobench_win FILE [--all] [--grid] [--sec 0.25] [--engine iocp|threads]: iobench.c for Windows (unbuffered reads, FILE_FLAG_NO_BUFFERING).
// engine iocp: overlapped ReadFile, `depth` reads in flight on one I/O completion port; engine threads: `depth` threads with their own handle and synchronous ReadFile.
// Hill climb from 1 MiB x 8 (one parameter one step while the gain is >= 3%); choice = the setting with the fewest bytes in flight that reaches 95% of the best bandwidth.
// build: clang -O2 -o iobench_win.exe iobench_win.c     (or: cl /O2 iobench_win.c)
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t BS[] = { 128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20 };
static const int    QD[] = { 1, 2, 4, 8, 16, 32, 64 };
#define NBS 6
#define NQD 7
static const char * g_path; static uint64_t g_size; static double g_sec = 0.25; static int g_threads, g_quiet, g_grid;
typedef struct { double gbs, p50_us, cpu_ms_gb; } res_t;

static double now(void) { LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double) t.QuadPart / f.QuadPart; }
static double cpu_s(void) { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u); return (((uint64_t) k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t) u.dwHighDateTime << 32 | u.dwLowDateTime)) * 1e-7; }
static uint64_t rnd(uint64_t * s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static int cmpd(const void * a, const void * b) { double x = *(const double *) a, y = *(const double *) b; return (x > y) - (x < y); }
static HANDLE opn(int overlapped) { return CreateFileA(g_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | (overlapped ? FILE_FLAG_OVERLAPPED : 0), NULL); }

static res_t run_iocp(size_t bs, int qd) {
    HANDLE h = opn(1); if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "CreateFile failed (%lu)\n", GetLastError()); exit(1); }
    HANDLE port = CreateIoCompletionPort(h, NULL, 0, 0);
    char * buf = VirtualAlloc(NULL, bs * qd, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    OVERLAPPED * ov = calloc(qd, sizeof(OVERLAPPED)); double * t0 = calloc(qd, sizeof(double)); double * lat = malloc(sizeof(double) * 1000000);
    size_t nl = 0, bytes = 0, nblk = g_size / bs; uint64_t seed = 88172645463325252ull;
    double start = now(), c0 = cpu_s();
    for (int i = 0; i < qd; i++) {
        uint64_t off = (rnd(&seed) % nblk) * bs; ov[i].Offset = (DWORD) off; ov[i].OffsetHigh = (DWORD) (off >> 32); t0[i] = now();
        if (!ReadFile(h, buf + (size_t) i * bs, (DWORD) bs, NULL, &ov[i]) && GetLastError() != ERROR_IO_PENDING) { fprintf(stderr, "ReadFile failed (%lu)\n", GetLastError()); exit(1); }
    }
    while (now() - start < g_sec) {
        DWORD n; ULONG_PTR key; OVERLAPPED * po;
        if (!GetQueuedCompletionStatus(port, &n, &key, &po, 5000)) break;
        int i = (int) (po - ov); double t = now(); if (nl < 1000000) lat[nl++] = (t - t0[i]) * 1e6; bytes += bs;
        uint64_t off = (rnd(&seed) % nblk) * bs; memset(&ov[i], 0, sizeof(OVERLAPPED)); ov[i].Offset = (DWORD) off; ov[i].OffsetHigh = (DWORD) (off >> 32); t0[i] = now();
        if (!ReadFile(h, buf + (size_t) i * bs, (DWORD) bs, NULL, &ov[i]) && GetLastError() != ERROR_IO_PENDING) break;
    }
    double el = now() - start, c1 = cpu_s();
    CancelIo(h); Sleep(20); CloseHandle(h); CloseHandle(port);
    res_t r = { bytes / el / 1e9, 0, 0 }; r.cpu_ms_gb = r.gbs > 0 ? (c1 - c0) * 1e3 / (r.gbs * el) : 0;
    if (nl) { qsort(lat, nl, sizeof(double), cmpd); r.p50_us = lat[nl / 2]; }
    VirtualFree(buf, 0, MEM_RELEASE); free(ov); free(t0); free(lat); return r;
}

typedef struct { size_t bs; int id; double until; size_t bytes; } th_t;
static DWORD WINAPI th_run(LPVOID p) {
    th_t * a = p; HANDLE h = opn(0); if (h == INVALID_HANDLE_VALUE) return 1;
    char * buf = VirtualAlloc(NULL, a->bs, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); uint64_t seed = 88172645463325252ull + a->id * 7919; size_t nblk = g_size / a->bs;
    while (now() < a->until) {
        uint64_t off = (rnd(&seed) % nblk) * a->bs; OVERLAPPED ov = { 0 }; ov.Offset = (DWORD) off; ov.OffsetHigh = (DWORD) (off >> 32); DWORD got;
        if (!ReadFile(h, buf, (DWORD) a->bs, &got, &ov) || got != a->bs) break;
        a->bytes += a->bs;
    }
    VirtualFree(buf, 0, MEM_RELEASE); CloseHandle(h); return 0;
}
static res_t run_threads(size_t bs, int n) {
    HANDLE t[64]; th_t a[64]; double start = now(), c0 = cpu_s();
    for (int i = 0; i < n; i++) { a[i] = (th_t) { bs, i, start + g_sec, 0 }; t[i] = CreateThread(NULL, 0, th_run, &a[i], 0, NULL); }
    WaitForMultipleObjects(n, t, TRUE, INFINITE); size_t bytes = 0; for (int i = 0; i < n; i++) { bytes += a[i].bytes; CloseHandle(t[i]); }
    double el = now() - start; res_t r = { bytes / el / 1e9, 0, 0 }; r.cpu_ms_gb = r.gbs > 0 ? (cpu_s() - c0) * 1e3 / (r.gbs * el) : 0; return r;
}
static res_t measure(int b, int q) { return g_threads ? run_threads(BS[b], QD[q]) : run_iocp(BS[b], QD[q]); }

static void search(void) {
    double T0 = now(); static res_t R[NBS][NQD]; static int done[NBS][NQD]; memset(done, 0, sizeof done);
    if (!g_quiet) printf("engine %s, %.2f s per point, file %.1f GiB\n%6s %4s %7s %8s %9s\n", g_threads ? "threads(ReadFile)" : "iocp", g_sec, g_size / 1073741824.0, "bs", "qd", "GB/s", "p50 us", "cpu ms/GB");
    #define M(b, q) (done[b][q] ? R[b][q] : (done[b][q] = 1, R[b][q] = measure(b, q), g_quiet ? 0 : printf("%5zuk %4d %7.2f %8.0f %9.1f\n", BS[b] >> 10, QD[q], R[b][q].gbs, R[b][q].p50_us, R[b][q].cpu_ms_gb), fflush(stdout), R[b][q]))
    int b = 3, q = 3;
    if (g_grid) { for (int i = 0; i < NBS; i++) for (int j = 0; j < NQD; j++) (void) M(i, j); }
    else for (;;) {
        double best = M(b, q).gbs; int nb = b, nq = q; int mv[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
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
    printf("%-18s best %.2f GB/s; choice: %5zu KiB x depth %2d = %.2f GB/s (p50 %6.0f us, %5.1f cpu ms/GB), %zu KiB in flight; %.1f s\n", g_threads ? "threads(ReadFile)" : "iocp",
        mx, BS[cb] >> 10, QD[cq], R[cb][cq].gbs, R[cb][cq].p50_us, R[cb][cq].cpu_ms_gb, cost >> 10, now() - T0);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: iobench_win FILE [--all] [--grid] [--sec S] [--engine iocp|threads]\n"); return 1; }
    int all = 0; g_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--grid")) g_grid = 1;
        else if (!strcmp(argv[i], "--all")) all = 1;
        else if (!strcmp(argv[i], "--sec") && i + 1 < argc) g_sec = atof(argv[++i]);
        else if (!strcmp(argv[i], "--engine") && i + 1 < argc) g_threads = !strcmp(argv[++i], "threads");
    }
    HANDLE h = opn(0); if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "cannot open %s (%lu)\n", g_path, GetLastError()); return 1; }
    LARGE_INTEGER sz; GetFileSizeEx(h, &sz); g_size = sz.QuadPart; CloseHandle(h);
    if (all) { g_quiet = 1; for (int c = 0; c < 2; c++) { g_threads = c; search(); } } else search();
    return 0;
}
