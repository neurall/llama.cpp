// fork: lazy row tables read straight from the file instead of through the mapping.
//
// DeepSeek V4.1's Engram table holds 768M rows of 144 bytes (103 GiB). Each token reads 48 of them at hashed, random places. Through the mapping every row cost a
// 4 KiB page of page cache (28x its size) next to the RAM pool, read by madvise(WILLNEED) or by a page fault with readahead (DS 13.5k-token call: about 115k major
// faults of 24 KiB in its first 4 s; user 2026-11-10: "the pressure was caused by our mmap use"). Here the rows a pass needs are read as O_DIRECT pages in one batch
// (ggml-read-fork.h sr_read_pages: kernel AIO on Linux, overlapped ReadFile on Windows) into a compact table that the graph gathers from. No page cache, nothing mapped is read.
//
// LLAMA_ROWREAD=0 keeps the mapping (A/B).

#include "llama-impl.h"
#include "../ggml/src/ggml-read-fork.h"

#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
struct row_src { int fd = -1; uint64_t off = 0; };
uint8_t * page_alloc(size_t n) {
#ifdef _WIN32
    return (uint8_t *) _aligned_malloc(n, direct_align());
#else
    void * p = nullptr;
    return posix_memalign(&p, direct_align(), n) == 0 ? (uint8_t *) p : nullptr;
#endif
}
void page_free(uint8_t * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
std::mutex g_mu;
std::unordered_map<const ggml_tensor *, row_src> g_src;
}

// the table t starts at byte `off` of the file open as fd_file (the loader's descriptor): opened once more past the page cache
bool llama_rowread_add_fork(const ggml_tensor * t, int fd_file, uint64_t off) {
    const char * e = getenv("LLAMA_ROWREAD");
    if (e && atoi(e) == 0) { return false; }
    const int fd = open_direct_fd(fd_file);
    if (fd < 0) { return false; }
    // the engine must work here: one page read now, or the caller keeps the mapping
    const size_t pg = direct_align();
    uint8_t * buf = page_alloc(pg);
    const uint64_t p0 = off / pg * pg;
    const bool ok = buf && sr_read_pages(fd, &p0, 1, pg, buf);
    page_free(buf);
    if (!ok) { LLAMA_LOG_WARN("%s: %s stays read through the mapping (no async page reads here)\n", __func__, ggml_get_name(t)); return false; }
    std::lock_guard<std::mutex> lk(g_mu);
    g_src[t] = { fd, off };
    LLAMA_LOG_INFO("%s: %s read straight from the file (O_DIRECT pages, kernel AIO), not through the mapping\n", __func__, ggml_get_name(t));
    return true;
}

bool llama_rowread_has_fork(const ggml_tensor * t) {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_src.count(t) > 0;
}

// rows (sorted, unique) of t into dst, row i at dst + i*row size. False: no source or a read failed (dst incomplete).
bool llama_rowread_gather_fork(const ggml_tensor * t, const int32_t * rows, size_t n, uint8_t * dst) {
    row_src s;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const auto it = g_src.find(t);
        if (it == g_src.end()) { return false; }
        s = it->second;
    }
    const size_t rs = ggml_row_size(t->type, t->ne[0]);
    const uint64_t pg = direct_align();
    constexpr size_t MAXP = 16384;   // pages per batch: 64 MiB of bounce buffer
    thread_local uint8_t * buf = nullptr;
    if (!buf) { buf = page_alloc(MAXP*pg); if (!buf) { return false; } }   // one per gathering thread, kept
    std::vector<uint64_t> pages;
    pages.reserve(MAXP);
    size_t i0 = 0;
    while (i0 < n) {
        // the rows of this batch and their pages, in file order; a row across a page boundary takes both pages (adjacent in the buffer, as in the file)
        pages.clear();
        size_t i1 = i0;
        while (i1 < n) {
            const uint64_t o = s.off + (uint64_t) rows[i1]*rs;
            const uint64_t a = o / pg * pg, b = (o + rs - 1) / pg * pg;
            const size_t add = (pages.empty() || pages.back() != a ? 1 : 0) + (b != a ? 1 : 0);
            if (pages.size() + add > MAXP) { break; }
            if (pages.empty() || pages.back() != a) { pages.push_back(a); }
            if (b != a) { pages.push_back(b); }
            ++i1;
        }
        if (!sr_read_pages(s.fd, pages.data(), pages.size(), pg, buf)) { return false; }
        size_t j = 0;
        for (size_t i = i0; i < i1; ++i) {
            const uint64_t o = s.off + (uint64_t) rows[i]*rs;
            const uint64_t a = o / pg * pg;
            while (pages[j] != a) { ++j; }
            memcpy(dst + i*rs, buf + j*pg + (o - a), rs);
        }
        i0 = i1;
    }
    return true;
}
