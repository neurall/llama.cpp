// membw: read bandwidth with N threads, portable (MSVC / gcc / clang, x86 AVX2 or scalar fallback).
// Each thread streams its own slice of a 2 GiB buffer (far bigger than any L3) for ~0.5 s per test, twice:
// "touch" = one 8-byte load per 64-byte line (what the moe-cache startup probe does), "full" = every byte with
// 256-bit loads. Build: g++ -O2 -mavx2 -pthread membw.cpp -o membw ; cl /O2 /arch:AVX2 /EHsc membw.cpp
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const size_t n = (size_t) (argc > 1 ? atoi(argv[1]) : 2048) << 20;
    uint8_t * buf = (uint8_t *) malloc(n + 64);
    uint8_t * a = (uint8_t *) (((uintptr_t) buf + 63) & ~(uintptr_t) 63);
    memset(a, 1, n);
    const int hw = (int) std::thread::hardware_concurrency();
    for (int mode = 0; mode < 2; ++mode) {
        for (int nt : { 1, 2, 4, hw / 4, hw / 2 - 1, hw / 2, hw }) {
            if (nt < 1) continue;
            std::atomic<bool> stop{false};
            std::vector<double> got(nt, 0);
            std::vector<uint64_t> sink(nt, 0);
            std::vector<std::thread> th;
            for (int t = 0; t < nt; ++t) {
                th.emplace_back([&, t] {
                    const size_t len = (n / nt) & ~(size_t) 4095, s0 = len * t; // 4 KB aligned slices (aligned AVX2 loads)
                    uint64_t x = 0;
                    while (!stop) {
                        const uint8_t * p = a + s0;
                        if (mode == 0) {
                            for (size_t o = 0; o < len; o += 64) { x += *(const volatile uint64_t *) (p + o); }
                        } else {
#if defined(__AVX2__)
                            __m256i acc = _mm256_setzero_si256();
                            for (size_t o = 0; o < len; o += 128) {
                                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i *) (p + o)));
                                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i *) (p + o + 32)));
                                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i *) (p + o + 64)));
                                acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i *) (p + o + 96)));
                            }
                            x += (uint64_t) _mm256_extract_epi64(acc, 0);
#else
                            for (size_t o = 0; o < len; o += 8) { x ^= *(const uint64_t *) (p + o); }
#endif
                        }
                        got[t] += (double) len;
                    }
                    sink[t] = x;
                });
            }
            const double t0 = now_s();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            stop = true;
            for (auto & x : th) x.join();
            const double dt = now_s() - t0;
            double tot = 0; for (double g : got) tot += g;
            printf("%-5s %2d threads: %6.1f GB/s\n", mode ? "full" : "touch", nt, tot / dt / 1e9);
        }
    }
    free(buf);
    return 0;
}
