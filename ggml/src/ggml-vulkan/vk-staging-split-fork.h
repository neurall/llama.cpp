#pragma once
// fork: host reads and writes of device-local Vulkan buffers go through a staging buffer of at most 32 MiB (the landing rule). The staging was
// sized to the largest single region: the server's prompt save read one 195 MiB KV region on the Radeon 780M, the staging allocation failed
// (ErrorOutOfDeviceMemory) and the uncaught exception ended the server (W10, 2026-11-25). Regions are split into pieces of at most the cap;
// the copy loop already submits whenever the staging is full.
// test: tests/test-vk-staging-split-fork.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

static inline size_t vk_staging_cap_fork() { return (size_t) 32 << 20; }

// R: { buf, offset, host, size } (vk_buffer_region)
template <class R>
static std::vector<R> vk_staging_split_fork(const std::vector<R> & in, size_t cap) {
    std::vector<R> out;
    out.reserve(in.size());
    for (const R & r : in) {
        for (size_t off = 0; off < r.size; off += cap) {
            R p = r;
            p.offset = r.offset + off;
            p.host   = (uint8_t *) r.host + off;
            p.size   = std::min(cap, r.size - off);
            out.push_back(p);
        }
    }
    return out;
}
